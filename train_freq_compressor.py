"""
频率压缩器训练脚本
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
from CAESAR.models.frequency_transform import HybridFrequencyEncoder
from dataset import ScientificDataset


class FrequencyCompressorTrainer:
    """频率压缩器训练器"""

    def __init__(
        self,
        model: FrequencyCompressor,
        device: str = 'cuda',
        lr: float = 1e-4,
        lambda_bpp: float = 0.01,  # bpp损失权重
        lambda_extra: float = 0.01,  # 先验正则化权重
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_bpp = lambda_bpp
        self.lambda_extra = lambda_extra

        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def train_step(self, batch):
        """单步训练"""
        self.model.train()

        x = batch["input"].to(self.device)

        # 前向传播
        outputs = self.model(x)

        # 重建损失
        recon_loss = nn.functional.mse_loss(outputs["output"], x)

        # bpp损失
        bpp_loss = outputs["bpp"].mean()

        # 先验正则化损失
        extra_loss = self.model.get_extra_loss()

        # 总损失
        total_loss = recon_loss + self.lambda_bpp * bpp_loss + self.lambda_extra * extra_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "bpp_loss": bpp_loss.item(),
            "extra_loss": extra_loss.item(),
            "bpp": bpp_loss.item(),
        }

    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.model.eval()

        total_recon_loss = 0
        total_bpp = 0
        num_batches = 0

        for batch in dataloader:
            x = batch["input"].to(self.device)
            outputs = self.model(x)

            recon_loss = nn.functional.mse_loss(outputs["output"], x)
            bpp = outputs["bpp"].mean()

            total_recon_loss += recon_loss.item()
            total_bpp += bpp.item()
            num_batches += 1

        return {
            "val_recon_loss": total_recon_loss / num_batches,
            "val_bpp": total_bpp / num_batches,
            "val_psnr": 10 * np.log10(1.0 / (total_recon_loss / num_batches + 1e-10)),
        }

    def save_checkpoint(self, path, epoch, best_loss=None):
        """保存检查点"""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": best_loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


def train_frequency_compressor(args):
    """训练频率压缩器"""

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型
    model = FrequencyCompressor(
        wavelet=args.wavelet,
        dct_block_size=args.dct_block_size,
        use_dwt=args.use_dwt,
        latent_channels=args.latent_channels,
        hyper_channels=args.hyper_channels,
        low_freq_size=args.low_freq_size,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建数据集配置
    train_args = {
        "data_path": args.train_data,
        "n_frame": args.n_frame,
        "train_size": args.training_size,
        "train": True,
        "variable_idx": [0],  # 使用第一个变量
        "inst_norm": True,
        "norm_type": "mean_range",
    }

    val_args = {
        "data_path": args.val_data if args.val_data else args.train_data,
        "n_frame": args.n_frame,
        "train_size": args.training_size,
        "train": False,
        "test_size": (args.training_size, args.training_size),
        "variable_idx": [0],
        "inst_norm": True,
        "norm_type": "mean_range",
    }

    train_dataset = ScientificDataset(train_args)
    val_dataset = ScientificDataset(val_args)

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
    trainer = FrequencyCompressorTrainer(
        model=model,
        device=device,
        lr=args.lr,
        lambda_bpp=args.lambda_bpp,
        lambda_extra=args.lambda_extra,
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
                "loss": f"{losses['total_loss']:.4f}",
                "recon": f"{losses['recon_loss']:.4f}",
                "bpp": f"{losses['bpp']:.4f}",
            })

        # 计算平均损失
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        print(f"Train - Loss: {avg_losses['total_loss']:.4f}, "
              f"Recon: {avg_losses['recon_loss']:.4f}, "
              f"BPP: {avg_losses['bpp']:.4f}")

        # 验证
        val_metrics = trainer.validate(val_loader)
        print(f"Val - Recon: {val_metrics['val_recon_loss']:.4f}, "
              f"BPP: {val_metrics['val_bpp']:.4f}, "
              f"PSNR: {val_metrics['val_psnr']:.2f} dB")

        # 更新学习率
        trainer.scheduler.step()

        # 保存检查点
        if (epoch + 1) % args.save_freq == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
                epoch + 1,
                best_loss,
            )

        # 保存最佳模型
        if val_metrics['val_recon_loss'] < best_loss:
            best_loss = val_metrics['val_recon_loss']
            trainer.save_checkpoint(
                os.path.join(args.output_dir, "best_model.pth"),
                epoch + 1,
                best_loss,
            )
            print("Best model saved!")

    # 保存最终模型
    trainer.save_checkpoint(
        os.path.join(args.output_dir, "final_model.pth"),
        args.epochs,
        best_loss,
    )
    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Frequency Compressor")

    # 数据参数
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data", type=str, default=None, help="验证数据路径")
    parser.add_argument("--n_frame", type=int, default=16, help="帧数")
    parser.add_argument("--training_size", type=int, default=256, help="训练尺寸")

    # 模型参数
    parser.add_argument("--wavelet", type=str, default="haar", choices=["haar", "cdf97"])
    parser.add_argument("--dct_block_size", type=int, default=8)
    parser.add_argument("--use_dwt", action="store_true", default=True)
    parser.add_argument("--no_dwt", action="store_false", dest="use_dwt")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--hyper_channels", type=int, default=32)
    parser.add_argument("--low_freq_size", type=int, default=16)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_bpp", type=float, default=0.01)
    parser.add_argument("--lambda_extra", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)

    # 其他参数
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/freq_compressor")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--save_freq", type=int, default=10, help="保存检查点的频率")

    args = parser.parse_args()

    train_frequency_compressor(args)


if __name__ == "__main__":
    main()
