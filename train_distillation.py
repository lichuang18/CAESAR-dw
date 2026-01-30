"""
单步扩散蒸馏训练脚本
将多步Teacher蒸馏为单步Student
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CAESAR.models.frequency_compressor import FrequencyCompressor
from CAESAR.models.video_diffusion_interpo import Unet3D, GaussianDiffusion
from CAESAR.models.single_step_diffusion import SingleStepStudent, SingleStepDiffusion
from dataset import ScientificDataset


class DistillationTrainer:
    """单步蒸馏训练器"""

    def __init__(
        self,
        freq_compressor: FrequencyCompressor,
        teacher: GaussianDiffusion,
        student: SingleStepStudent,
        device: str = 'cuda',
        lr: float = 1e-4,
        interpo_rate: int = 3,
        n_frame: int = 16,
        use_perceptual_loss: bool = True,
        use_frequency_loss: bool = True,
    ):
        self.freq_compressor = freq_compressor.to(device).eval()
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.device = device
        self.interpo_rate = interpo_rate
        self.n_frame = n_frame

        # 冻结频率压缩器和Teacher
        for p in self.freq_compressor.parameters():
            p.requires_grad = False
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 关键帧和插值帧索引
        self.cond_idx = torch.arange(0, n_frame, interpo_rate)
        self.pred_idx = ~torch.isin(torch.arange(n_frame), self.cond_idx)

        # 损失配置
        self.use_perceptual_loss = use_perceptual_loss
        self.use_frequency_loss = use_frequency_loss

        # 优化器
        self.optimizer = optim.AdamW(student.parameters(), lr=lr)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def normalize_latent(self, x):
        """归一化潜在变量到[-1, 1]"""
        x_min = torch.amin(x, dim=(1, 2, 3, 4), keepdim=True)
        x_max = torch.amax(x, dim=(1, 2, 3, 4), keepdim=True)
        scale = (x_max - x_min + 1e-8) / 2
        offset = x_min + scale
        x_norm = (x - offset) / scale
        return x_norm, offset, scale

    @torch.no_grad()
    def get_latent(self, x):
        """获取频率空间的低频潜在变量"""
        B, C, T, H, W = x.shape

        # 频率变换
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        freq_coeffs = self.freq_compressor.freq_encoder(x_2d)

        if self.freq_compressor.use_dwt:
            LL_dct = freq_coeffs['LL_dct']
        else:
            LL_dct = freq_coeffs['dct']

        # 提取低频
        low_freq_size = self.freq_compressor.low_freq_size
        low_freq = LL_dct[:, :, :low_freq_size, :low_freq_size]

        # 编码
        latent = self.freq_compressor.low_freq_processor.encode(low_freq)

        # 重塑为5D
        _, c, h, w = latent.shape
        latent = latent.view(B, T, c, h, w).permute(0, 2, 1, 3, 4)

        return latent

    def compute_loss(self, student_output, teacher_output):
        """计算蒸馏损失"""
        losses = {}

        # MSE损失
        losses['mse'] = nn.functional.mse_loss(student_output, teacher_output)

        # 感知损失（多尺度MSE）
        if self.use_perceptual_loss:
            perceptual_loss = 0
            s_down = student_output
            t_down = teacher_output
            for scale in [1, 2, 4]:
                if scale > 1:
                    s_down = nn.functional.avg_pool3d(s_down, (1, 2, 2))
                    t_down = nn.functional.avg_pool3d(t_down, (1, 2, 2))
                perceptual_loss += nn.functional.mse_loss(s_down, t_down)
            losses['perceptual'] = perceptual_loss / 3

        # 频率损失
        if self.use_frequency_loss:
            s_fft = torch.fft.rfft2(student_output)
            t_fft = torch.fft.rfft2(teacher_output)
            losses['frequency'] = nn.functional.mse_loss(s_fft.abs(), t_fft.abs())

        # 总损失
        total_loss = losses['mse']
        if self.use_perceptual_loss:
            total_loss = total_loss + 0.1 * losses['perceptual']
        if self.use_frequency_loss:
            total_loss = total_loss + 0.1 * losses['frequency']
        losses['total'] = total_loss

        return losses

    def train_step(self, batch):
        """单步训练"""
        self.student.train()

        x = batch["input"].to(self.device)

        # 获取潜在变量
        latent = self.get_latent(x)
        B = latent.shape[0]

        # 归一化
        latent_norm, offset, scale = self.normalize_latent(latent)

        # 构建条件输入（关键帧）
        condition = latent_norm[:, :, self.cond_idx]

        # 生成噪声
        noise = torch.randn_like(latent_norm[:, :, self.pred_idx])

        # Teacher生成目标（无梯度）
        with torch.no_grad():
            # 构建完整输入
            input_latent = latent_norm.clone()
            input_latent[:, :, self.pred_idx] = noise

            # Teacher采样
            teacher_output = self.teacher.sample(
                input_latent,
                self.interpo_rate,
                batch_size=B
            )

        # Student单步预测
        # 将条件和噪声拼接
        student_input = torch.cat([noise, condition], dim=1)
        student_output = self.student(
            x=student_input,
            condition=None,  # 条件已经拼接在输入中
            t=None
        )

        # 计算损失
        losses = self.compute_loss(student_output, teacher_output)

        # 反向传播
        self.optimizer.zero_grad()
        losses['total'].backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()

        total_mse = 0
        total_psnr = 0
        num_batches = 0

        for batch in dataloader:
            x = batch["input"].to(self.device)
            latent = self.get_latent(x)
            B = latent.shape[0]

            latent_norm, offset, scale = self.normalize_latent(latent)
            condition = latent_norm[:, :, self.cond_idx]
            noise = torch.randn_like(latent_norm[:, :, self.pred_idx])

            # Teacher输出
            input_latent = latent_norm.clone()
            input_latent[:, :, self.pred_idx] = noise
            teacher_output = self.teacher.sample(
                input_latent,
                self.interpo_rate,
                batch_size=B
            )

            # Student输出
            student_input = torch.cat([noise, condition], dim=1)
            student_output = self.student(x=student_input, condition=None, t=None)

            # 计算指标
            mse = nn.functional.mse_loss(student_output, teacher_output)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

            total_mse += mse.item()
            total_psnr += psnr.item()
            num_batches += 1

        return {
            "val_mse": total_mse / num_batches,
            "val_psnr": total_psnr / num_batches,
        }

    def save_checkpoint(self, path, epoch, best_loss=None):
        """保存检查点"""
        torch.save({
            "epoch": epoch,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": best_loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


def train_distillation(args):
    """训练单步蒸馏"""

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载频率压缩器
    freq_compressor = FrequencyCompressor(
        wavelet=args.wavelet,
        dct_block_size=args.dct_block_size,
        use_dwt=args.use_dwt,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        low_freq_size=args.low_freq_size,
    )

    if args.freq_compressor_path:
        checkpoint = torch.load(args.freq_compressor_path, map_location=device)
        if "model_state_dict" in checkpoint:
            freq_compressor.load_state_dict(checkpoint["model_state_dict"])
        else:
            freq_compressor.load_state_dict(checkpoint)
        print(f"Loaded frequency compressor from {args.freq_compressor_path}")

    # 加载Teacher扩散模型
    unet = Unet3D(
        dim=args.unet_dim,
        out_dim=args.latent_channels,
        channels=args.latent_channels,
        dim_mults=tuple(args.dim_mults),
        use_bert_text_cond=False
    )

    teacher = GaussianDiffusion(
        unet,
        image_size=args.low_freq_size,
        num_frames=args.n_frame // args.interpo_rate + 1,
        channels=args.latent_channels,
        timesteps=args.teacher_steps,
        loss_type='l2'
    )

    if args.teacher_path:
        checkpoint = torch.load(args.teacher_path, map_location=device)
        if "diffusion_state_dict" in checkpoint:
            teacher.load_state_dict(checkpoint["diffusion_state_dict"])
        else:
            teacher.load_state_dict(checkpoint)
        print(f"Loaded teacher from {args.teacher_path}")

    # 创建Student模型
    # 计算插值帧数
    num_pred_frames = args.n_frame - (args.n_frame // args.interpo_rate + 1)

    student = SingleStepStudent(
        in_channels=args.latent_channels * 2,  # 噪声 + 条件
        out_channels=args.latent_channels,
        base_channels=args.student_base_channels,
        channel_mults=tuple(args.student_channel_mults),
        num_res_blocks=args.student_num_res_blocks,
        use_time_emb=False,
    )

    print(f"Student model parameters: {sum(p.numel() for p in student.parameters()):,}")

    # 创建数据集
    train_dataset = ScientificDataset(
        data_path=args.train_data,
        n_frame=args.n_frame,
        training_size=args.training_size,
        mode="train",
    )

    val_dataset = ScientificDataset(
        data_path=args.val_data if args.val_data else args.train_data,
        n_frame=args.n_frame,
        training_size=args.training_size,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 创建训练器
    trainer = DistillationTrainer(
        freq_compressor=freq_compressor,
        teacher=teacher,
        student=student,
        device=device,
        lr=args.lr,
        interpo_rate=args.interpo_rate,
        n_frame=args.n_frame,
        use_perceptual_loss=args.use_perceptual_loss,
        use_frequency_loss=args.use_frequency_loss,
    )

    # 加载检查点
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, best_loss = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        epoch_losses = []
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            losses = trainer.train_step(batch)
            epoch_losses.append(losses)
            pbar.set_postfix({
                "loss": f"{losses['total']:.4f}",
                "mse": f"{losses['mse']:.4f}",
            })

        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        print(f"Train - Loss: {avg_losses['total']:.4f}, MSE: {avg_losses['mse']:.4f}")

        # 验证
        val_metrics = trainer.validate(val_loader)
        print(f"Val - MSE: {val_metrics['val_mse']:.4f}, PSNR: {val_metrics['val_psnr']:.2f} dB")

        # 更新学习率
        trainer.scheduler.step()

        # 保存检查点
        if (epoch + 1) % args.save_freq == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"student_epoch_{epoch + 1}.pth"),
                epoch + 1,
                best_loss,
            )

        # 保存最佳模型
        if val_metrics['val_mse'] < best_loss:
            best_loss = val_metrics['val_mse']
            trainer.save_checkpoint(
                os.path.join(args.output_dir, "student_best.pth"),
                epoch + 1,
                best_loss,
            )
            print("Best model saved!")

    # 保存最终模型
    trainer.save_checkpoint(
        os.path.join(args.output_dir, "student_final.pth"),
        args.epochs,
        best_loss,
    )

    # 保存可部署的单步模型
    single_step_model = SingleStepDiffusion(trainer.student)
    torch.save(
        single_step_model.state_dict(),
        os.path.join(args.output_dir, "single_step_diffusion.pth")
    )
    print("Training completed! Single-step model saved.")


def main():
    parser = argparse.ArgumentParser(description="Train Single-Step Distillation")

    # 数据参数
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--n_frame", type=int, default=16)
    parser.add_argument("--training_size", type=int, default=256)
    parser.add_argument("--interpo_rate", type=int, default=3)

    # 频率压缩器参数
    parser.add_argument("--freq_compressor_path", type=str, required=True)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--dct_block_size", type=int, default=8)
    parser.add_argument("--use_dwt", action="store_true", default=True)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--hyper_channels", type=int, default=32)
    parser.add_argument("--low_freq_size", type=int, default=16)

    # Teacher参数
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--teacher_steps", type=int, default=32)
    parser.add_argument("--unet_dim", type=int, default=64)
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])

    # Student参数
    parser.add_argument("--student_base_channels", type=int, default=64)
    parser.add_argument("--student_channel_mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--student_num_res_blocks", type=int, default=2)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_perceptual_loss", action="store_true", default=True)
    parser.add_argument("--use_frequency_loss", action="store_true", default=True)

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/distillation")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=10)

    args = parser.parse_args()

    train_distillation(args)


if __name__ == "__main__":
    main()
