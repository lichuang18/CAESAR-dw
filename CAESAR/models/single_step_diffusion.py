"""
单步扩散蒸馏模块
参考 SinSR: Diffusion-Based Image Super-Resolution in a Single Step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class SingleStepStudent(nn.Module):
    """
    单步预测网络 (Student)
    直接从条件预测目标，无需迭代
    """
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        use_time_emb: bool = True,  # 是否使用时间嵌入（可选）
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_time_emb = use_time_emb

        # 时间嵌入（可选，用于兼容Teacher结构）
        if use_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None
            time_emb_dim = None

        # 输入卷积
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # 编码器
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()

        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock3D(ch, out_ch, time_emb_dim))
                ch = out_ch
            if i < len(channel_mults) - 1:
                self.downsample.append(Downsample3D(ch))

        # 中间层
        self.mid_block1 = ResBlock3D(ch, ch, time_emb_dim)
        self.mid_attn = AttentionBlock3D(ch)
        self.mid_block2 = ResBlock3D(ch, ch, time_emb_dim)

        # 解码器
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.decoder.append(ResBlock3D(ch + out_ch if _ == 0 else ch, out_ch, time_emb_dim))
                ch = out_ch
            if i > 0:
                self.upsample.append(Upsample3D(ch))

        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, 3, padding=1),
        )

        # 跳跃连接存储
        self.skip_connections = []

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        单步预测
        Args:
            x: 噪声或初始化 [B, C, T, H, W]
            condition: 条件（关键帧低频） [B, C, T, H, W]
            t: 时间步（可选）
        Returns:
            预测结果 [B, C, T, H, W]
        """
        # 合并输入和条件
        h = torch.cat([x, condition], dim=1) if condition is not None else x
        h = self.input_conv(h)

        # 时间嵌入
        if self.use_time_emb and t is not None:
            t_emb = self.time_mlp(t)
        else:
            t_emb = None

        # 编码器 - 每 2 个 block 后保存一次 skip（对应每个分辨率）
        skips = []
        down_idx = 0
        for i, block in enumerate(self.encoder):
            h = block(h, t_emb)
            if (i + 1) % 2 == 0:
                skips.append(h)  # 在 downsample 之前保存
                if down_idx < len(self.downsample):
                    h = self.downsample[down_idx](h)
                    down_idx += 1

        # 中间层
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 解码器 - 在每个分辨率的第一个 block 前 cat 对应的 skip
        up_idx = 0
        for i, block in enumerate(self.decoder):
            if i % 3 == 0 and skips:
                h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, t_emb)
            if (i + 1) % 3 == 0 and up_idx < len(self.upsample):
                h = self.upsample[up_idx](h)
                up_idx += 1

        return self.output_conv(h)


class SinusoidalPosEmb(nn.Module):
    """正弦位置嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock3D(nn.Module):
    """3D残差块"""
    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: Optional[int] = None):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2),
            )
        else:
            self.time_mlp = None

        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        if self.time_mlp is not None and t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            t_emb = t_emb[:, :, None, None, None]
            scale, shift = t_emb.chunk(2, dim=1)
            h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)


class AttentionBlock3D(nn.Module):
    """3D注意力块"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = self.norm(x)

        # 重塑为序列
        h = h.view(B, C, -1).permute(0, 2, 1)  # [B, T*H*W, C]
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, T, H, W)

        return x + h


class Downsample3D(nn.Module):
    """3D下采样"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=(1, 2, 2), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D上采样"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        return self.conv(x)


class DistillationTrainer:
    """
    蒸馏训练器
    将多步Teacher蒸馏为单步Student
    """
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        device: str = 'cuda',
        lr: float = 1e-4,
        use_perceptual_loss: bool = True,
        use_frequency_loss: bool = True,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device

        # 冻结Teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 优化器
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)

        # 损失权重
        self.use_perceptual_loss = use_perceptual_loss
        self.use_frequency_loss = use_frequency_loss

        # 固定时间步嵌入（用于Student）
        self.register_fixed_t = torch.tensor([0.0], device=device)

    def compute_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失"""
        losses = {}

        # MSE损失
        losses['mse'] = F.mse_loss(student_output, teacher_output)

        # 感知损失（简化版：多尺度MSE）
        if self.use_perceptual_loss:
            perceptual_loss = 0
            s_down = student_output
            t_down = teacher_output
            for scale in [1, 2, 4]:
                if scale > 1:
                    s_down = F.avg_pool3d(s_down, (1, 2, 2))
                    t_down = F.avg_pool3d(t_down, (1, 2, 2))
                perceptual_loss += F.mse_loss(s_down, t_down)
            losses['perceptual'] = perceptual_loss / 3

        # 频率损失
        if self.use_frequency_loss:
            # 简化：使用DCT近似（这里用FFT）
            s_fft = torch.fft.rfft2(student_output)
            t_fft = torch.fft.rfft2(teacher_output)
            losses['frequency'] = F.mse_loss(s_fft.abs(), t_fft.abs())

        # 总损失
        total_loss = losses['mse']
        if self.use_perceptual_loss:
            total_loss = total_loss + 0.1 * losses['perceptual']
        if self.use_frequency_loss:
            total_loss = total_loss + 0.1 * losses['frequency']
        losses['total'] = total_loss

        return losses

    def train_step(
        self,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        单步蒸馏训练
        Args:
            condition: 条件输入（关键帧低频）
            noise: 初始噪声（可选）
        Returns:
            损失字典
        """
        self.student.train()

        B = condition.shape[0]

        # 生成噪声
        if noise is None:
            noise = torch.randn_like(condition)

        # Teacher生成目标（无梯度）
        with torch.no_grad():
            teacher_output = self.teacher.sample(
                condition=condition,
                noise=noise,
            )

        # Student单步预测
        student_output = self.student(
            x=noise,
            condition=condition,
            t=self.register_fixed_t.expand(B),
        )

        # 计算损失
        losses = self.compute_loss(student_output, teacher_output)

        # 反向传播
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ProgressiveDistillation:
    """
    渐进式蒸馏
    32 → 16 → 8 → 4 → 2 → 1
    """
    def __init__(
        self,
        teacher: nn.Module,
        student_config: Dict,
        device: str = 'cuda',
    ):
        self.teacher = teacher
        self.student_config = student_config
        self.device = device

        # 蒸馏阶段
        self.stages = [
            (32, 16),
            (16, 8),
            (8, 4),
            (4, 2),
            (2, 1),
        ]

        self.current_stage = 0
        self.students = {}

    def distill_stage(
        self,
        teacher_steps: int,
        student_steps: int,
        train_loader,
        num_epochs: int = 10,
    ):
        """
        单阶段蒸馏
        """
        print(f"Distilling: {teacher_steps} steps → {student_steps} steps")

        # 创建Student
        student = SingleStepStudent(**self.student_config).to(self.device)

        # 如果不是第一阶段，用上一阶段的Student作为Teacher
        if teacher_steps < 32:
            current_teacher = self.students[teacher_steps]
        else:
            current_teacher = self.teacher

        # 训练
        trainer = DistillationTrainer(
            teacher=current_teacher,
            student=student,
            device=self.device,
        )

        for epoch in range(num_epochs):
            epoch_losses = []
            for batch in train_loader:
                condition = batch['condition'].to(self.device)
                losses = trainer.train_step(condition)
                epoch_losses.append(losses['total'])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

        self.students[student_steps] = student
        return student

    def distill_all(self, train_loader, num_epochs_per_stage: int = 10):
        """执行所有蒸馏阶段"""
        for teacher_steps, student_steps in self.stages:
            self.distill_stage(
                teacher_steps=teacher_steps,
                student_steps=student_steps,
                train_loader=train_loader,
                num_epochs=num_epochs_per_stage,
            )

        return self.students[1]  # 返回单步模型


class SingleStepDiffusion(nn.Module):
    """
    单步扩散模型（部署用）
    封装Student模型，提供简单的推理接口
    """
    def __init__(self, student: SingleStepStudent):
        super().__init__()
        self.student = student

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        单步生成
        Args:
            condition: 条件输入 [B, C, T, H, W]
        Returns:
            生成结果 [B, C, T, H, W]
        """
        # 使用零噪声或学习的初始化
        noise = torch.zeros_like(condition)
        return self.student(x=noise, condition=condition, t=None)

    def sample(self, condition: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        采样接口（兼容Teacher接口）
        """
        if noise is None:
            noise = torch.zeros_like(condition)
        return self.student(x=noise, condition=condition, t=None)
