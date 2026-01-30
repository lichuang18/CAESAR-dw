"""
扩散模型训练脚本（在频率空间低频上训练）
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
from dataset import ScientificDataset


class DiffusionTrainer:
    """扩散模型训练器（频率空间）"""

    def __init__(
        self,
        freq_compressor: FrequencyCompressor,
        diffusion_model: GaussianDiffusion,
        device: str = 'cuda',
        lr: float = 1e-4,
        interpo_rate: int = 3,
        n_frame: int = 16,
    ):
        self.freq_compressor = freq_compressor.to(device).eval()
        self.diffusion_model = diffusion_model.to(device)
        self.device = device
        self.interpo_rate = interpo_rate
        self.n_frame = n_frame

        # 冻结频率压缩器
        for p in self.freq_compressor.parameters():
            p.requires_grad = False

        # 关键帧和插值帧索引
        self.cond_idx = torch.arange(0, n_frame, interpo_rate)
        self.pred_idx = ~torch.isin(torch.arange(n_frame), self.cond_idx)

        # 优化器
        self.optimizer = optim.AdamW(diffusion_model.parameters(), lr=lr)

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

    def train_step(self, batch):
        """单步训练"""
        self.diffusion_model.train()

        x = batch["input"].to(self.device)

        # 获取潜在变量
        latent = self.get_latent(x)

        # 归一化
        latent_norm, offset, scale = self.normalize_latent(latent)

        # 扩散模型训练
        loss = self.diffusion_model(
            latent_norm,
            cond_idx=self.cond_idx.to(self.device),
            interpo_rate=self.interpo_rate
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.diffusion_model.eval()

        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            x = batch["input"].to(self.device)
            latent = self.get_latent(x)
            latent_norm, _, _ = self.normalize_latent(latent)

            # 计算验证损失
            loss = self.diffusion_model(
                latent_norm,
                cond_idx=self.cond_idx.to(self.device),
                interpo_rate=self.interpo_rate
            )

            total_loss += loss.item()
            num_batches += 1

        return {"val_loss": total_loss / num_batches}

    @torch.no_grad()
    def sample_and_evaluate(self, batch):
        """采样并评估"""
        self.diffusion_model.eval()

        x = batch["input"].to(self.device)
        latent = self.get_latent(x)
        latent_norm, offset, scale = self.normalize_latent(latent)

        # 构建条件输入
        input_latent = latent_norm.clone()
        input_latent[:, :, self.pred_idx] = 0  # 置零待预测帧

        # 采样
        result = self.diffusion_model.sample(
            input_latent,
            self.interpo_rate,
            batch_size=input_latent.shape[0]
        )

        # 计算MSE
        gt = latent_norm[:, :, self.pred_idx]
        mse = nn.functional.mse_loss(result, gt)

        return {"sample_mse": mse.item()}

    def save_checkpoint(self, path, epoch, best_loss=None):
        """保存检查点"""
        torch.save({
            "epoch": epoch,
            "diffusion_state_dict": self.diffusion_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": best_loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint["diffusion_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


def train_diffusion(args):
    """训练扩散模型"""

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

    # 创建扩散模型
    unet = Unet3D(
        dim=args.unet_dim,
        out_dim=args.latent_channels,
        channels=args.latent_channels,
        dim_mults=tuple(args.dim_mults),
        use_bert_text_cond=False
    )

    diffusion = GaussianDiffusion(
        unet,
        image_size=args.low_freq_size,
        num_frames=args.n_frame // args.interpo_rate + 1,
        channels=args.latent_channels,
        timesteps=args.diffusion_steps,
        loss_type='l2'
    )

    print(f"Diffusion model parameters: {sum(p.numel() for p in diffusion.parameters()):,}")

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
    trainer = DiffusionTrainer(
        freq_compressor=freq_compressor,
        diffusion_model=diffusion,
        device=device,
        lr=args.lr,
        interpo_rate=args.interpo_rate,
        n_frame=args.n_frame,
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
            pbar.set_postfix({"loss": f"{losses['loss']:.4f}"})

        avg_loss = np.mean([l["loss"] for l in epoch_losses])
        print(f"Train - Loss: {avg_loss:.4f}")

        # 验证
        val_metrics = trainer.validate(val_loader)
        print(f"Val - Loss: {val_metrics['val_loss']:.4f}")

        # 更新学习率
        trainer.scheduler.step()

        # 保存检查点
        if (epoch + 1) % args.save_freq == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"diffusion_epoch_{epoch + 1}.pth"),
                epoch + 1,
                best_loss,
            )

        # 保存最佳模型
        if val_metrics['val_loss'] < best_loss:
            best_loss = val_metrics['val_loss']
            trainer.save_checkpoint(
                os.path.join(args.output_dir, "diffusion_best.pth"),
                epoch + 1,
                best_loss,
            )
            print("Best model saved!")

    # 保存最终模型
    trainer.save_checkpoint(
        os.path.join(args.output_dir, "diffusion_final.pth"),
        args.epochs,
        best_loss,
    )
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")

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

    # 扩散模型参数
    parser.add_argument("--unet_dim", type=int, default=64)
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--diffusion_steps", type=int, default=32)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/diffusion")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=10)

    args = parser.parse_args()

    train_diffusion(args)


if __name__ == "__main__":
    main()
