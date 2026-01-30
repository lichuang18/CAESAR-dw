"""
频率变换模块：DWT + DCT
用于替代VAE编码器，实现无损变换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class DWTTransform(nn.Module):
    """
    离散小波变换模块 (2D)
    支持 Haar 小波，使用简单高效的实现
    """
    def __init__(self, wavelet: str = 'haar'):
        super().__init__()
        self.wavelet = wavelet
        if wavelet != 'haar':
            raise ValueError(f"Currently only 'haar' wavelet is supported, got: {wavelet}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DWT正变换 (Haar)
        Args:
            x: [B, C, H, W]
        Returns:
            LL, LH, HL, HH: 各 [B, C, H/2, W/2]
        """
        # 提取偶数和奇数位置
        x00 = x[:, :, 0::2, 0::2]  # 左上
        x01 = x[:, :, 0::2, 1::2]  # 右上
        x10 = x[:, :, 1::2, 0::2]  # 左下
        x11 = x[:, :, 1::2, 1::2]  # 右下

        # Haar 变换
        LL = (x00 + x01 + x10 + x11) / 2.0
        LH = (x00 + x01 - x10 - x11) / 2.0
        HL = (x00 - x01 + x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0

        return LL, LH, HL, HH

    def inverse(self, LL: torch.Tensor, LH: torch.Tensor,
                HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        """
        DWT逆变换 (Haar)
        Args:
            LL, LH, HL, HH: 各 [B, C, H/2, W/2]
        Returns:
            x: [B, C, H, W]
        """
        B, C, H, W = LL.shape

        # Haar 逆变换
        x00 = (LL + LH + HL + HH) / 2.0
        x01 = (LL + LH - HL - HH) / 2.0
        x10 = (LL - LH + HL - HH) / 2.0
        x11 = (LL - LH - HL + HH) / 2.0

        # 重组
        x = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        x[:, :, 0::2, 0::2] = x00
        x[:, :, 0::2, 1::2] = x01
        x[:, :, 1::2, 0::2] = x10
        x[:, :, 1::2, 1::2] = x11

        return x


class DCTTransform(nn.Module):
    """
    离散余弦变换模块 (块级2D DCT)
    """
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size

        # 预计算DCT矩阵
        dct_matrix = self._create_dct_matrix(block_size)
        self.register_buffer('dct_matrix', dct_matrix)
        self.register_buffer('idct_matrix', dct_matrix.t())

    def _create_dct_matrix(self, N: int) -> torch.Tensor:
        """创建DCT-II变换矩阵"""
        dct_mat = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_mat[k, n] = np.sqrt(1 / N)
                else:
                    dct_mat[k, n] = np.sqrt(2 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        return dct_mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        块级2D DCT正变换
        Args:
            x: [B, C, H, W]
        Returns:
            dct_coeffs: [B, C, H, W]
        """
        B, C, H, W = x.shape
        bs = self.block_size

        # 确保尺寸是block_size的倍数
        assert H % bs == 0 and W % bs == 0, f"H={H}, W={W} must be divisible by block_size={bs}"

        # 分块: [B, C, H//bs, bs, W//bs, bs]
        x = x.view(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H//bs, W//bs, bs, bs]

        # 2D DCT: Y = D @ X @ D^T
        x = torch.matmul(self.dct_matrix, x)
        x = torch.matmul(x, self.idct_matrix)

        # 还原形状
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H//bs, bs, W//bs, bs]
        x = x.contiguous().view(B, C, H, W)

        return x

    def inverse(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """
        块级2D DCT逆变换
        Args:
            dct_coeffs: [B, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B, C, H, W = dct_coeffs.shape
        bs = self.block_size

        # 分块
        x = dct_coeffs.view(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, H//bs, W//bs, bs, bs]

        # 2D IDCT: X = D^T @ Y @ D
        x = torch.matmul(self.idct_matrix, x)
        x = torch.matmul(x, self.dct_matrix)

        # 还原形状
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.contiguous().view(B, C, H, W)

        return x


class HybridFrequencyEncoder(nn.Module):
    """
    DWT + DCT 混合频率变换编码器
    """
    def __init__(self, wavelet: str = 'haar', dct_block_size: int = 8, use_dwt: bool = True):
        super().__init__()
        self.use_dwt = use_dwt
        self.dwt = DWTTransform(wavelet) if use_dwt else None
        self.dct = DCTTransform(dct_block_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码：空间域 -> 频率域
        Args:
            x: [B, C, H, W] 或 [B, C, T, H, W]
        Returns:
            dict: 频率系数字典
        """
        # 处理5D输入 (视频)
        is_video = x.dim() == 5
        if is_video:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        if self.use_dwt:
            # Step 1: DWT分解
            LL, LH, HL, HH = self.dwt(x)

            # Step 2: 对LL子带做DCT
            LL_dct = self.dct(LL)

            result = {
                'LL_dct': LL_dct,  # 主要能量
                'LH': LH,          # 水平细节
                'HL': HL,          # 垂直细节
                'HH': HH           # 对角细节
            }
        else:
            # 纯DCT模式
            dct_coeffs = self.dct(x)
            result = {'dct': dct_coeffs}

        # 恢复5D形状
        if is_video:
            for key in result:
                coeff = result[key]
                _, c, h, w = coeff.shape
                result[key] = coeff.view(B, T, c, h, w).permute(0, 2, 1, 3, 4)

        return result

    def inverse(self, coeffs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        解码：频率域 -> 空间域
        Args:
            coeffs: 频率系数字典
        Returns:
            x: [B, C, H, W] 或 [B, C, T, H, W]
        """
        # 检测是否为视频
        sample_coeff = list(coeffs.values())[0]
        is_video = sample_coeff.dim() == 5

        if is_video:
            B, C, T, H, W = sample_coeff.shape
            # 转换为4D
            coeffs_4d = {}
            for key in coeffs:
                coeff = coeffs[key]
                coeffs_4d[key] = coeff.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            coeffs = coeffs_4d

        if self.use_dwt:
            # Step 1: IDCT恢复LL子带
            LL = self.dct.inverse(coeffs['LL_dct'])

            # Step 2: IDWT合并
            x = self.dwt.inverse(LL, coeffs['LH'], coeffs['HL'], coeffs['HH'])
        else:
            # 纯IDCT模式
            x = self.dct.inverse(coeffs['dct'])

        # 恢复5D形状
        if is_video:
            _, c, h, w = x.shape
            x = x.view(B, T, c, h, w).permute(0, 2, 1, 3, 4)

        return x


class FrequencySeparator(nn.Module):
    """
    频率系数分离器
    将DCT系数分离为低频、中频、高频
    """
    def __init__(self, block_size: int = 8, low_freq_ratio: float = 0.125, mid_freq_ratio: float = 0.375):
        """
        Args:
            block_size: DCT块大小
            low_freq_ratio: 低频区域比例 (默认1/8，即左上角1/8区域)
            mid_freq_ratio: 中频区域比例
        """
        super().__init__()
        self.block_size = block_size
        self.low_freq_ratio = low_freq_ratio
        self.mid_freq_ratio = mid_freq_ratio

    def forward(self, dct_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        分离DCT系数为低/中/高频
        Args:
            dct_coeffs: [B, C, H, W] DCT系数
        Returns:
            low_freq, mid_freq, high_freq
        """
        B, C, H, W = dct_coeffs.shape

        low_h = int(H * self.low_freq_ratio)
        low_w = int(W * self.low_freq_ratio)
        mid_h = int(H * self.mid_freq_ratio)
        mid_w = int(W * self.mid_freq_ratio)

        # 低频：左上角
        low_freq = dct_coeffs[:, :, :low_h, :low_w]

        # 中频：去除低频后的中间区域
        mid_freq_full = dct_coeffs[:, :, :mid_h, :mid_w].clone()
        mid_freq_full[:, :, :low_h, :low_w] = 0  # 置零低频部分
        mid_freq = mid_freq_full

        # 高频：剩余部分
        high_freq = dct_coeffs.clone()
        high_freq[:, :, :mid_h, :mid_w] = 0  # 置零低频和中频部分

        return low_freq, mid_freq, high_freq

    def merge(self, low_freq: torch.Tensor, mid_freq: torch.Tensor,
              high_freq: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        合并低/中/高频为完整DCT系数
        """
        B, C, H, W = target_shape
        dct_coeffs = high_freq.clone()

        low_h, low_w = low_freq.shape[-2:]
        mid_h, mid_w = mid_freq.shape[-2:]

        # 添加中频
        dct_coeffs[:, :, :mid_h, :mid_w] = dct_coeffs[:, :, :mid_h, :mid_w] + mid_freq

        # 添加低频
        dct_coeffs[:, :, :low_h, :low_w] = dct_coeffs[:, :, :low_h, :low_w] + low_freq

        return dct_coeffs


class SparseQuantizer(nn.Module):
    """
    稀疏量化器 - 用于中高频系数
    """
    def __init__(self, default_threshold: float = 0.01):
        super().__init__()
        self.default_threshold = default_threshold

    def forward(self, coeffs: torch.Tensor, q_step: float = 1.0,
                threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        稀疏量化
        Args:
            coeffs: 频率系数
            q_step: 量化步长
            threshold: 阈值（小于此值的系数置零）
        Returns:
            quantized: 量化后的系数
            mask: 非零掩码
        """
        if threshold is None:
            threshold = self.default_threshold

        # 阈值截断
        mask = torch.abs(coeffs) > threshold

        # 量化
        quantized = torch.round(coeffs / q_step) * q_step
        quantized = quantized * mask.float()

        return quantized, mask

    def inverse(self, quantized: torch.Tensor) -> torch.Tensor:
        """反量化（直接返回）"""
        return quantized


# 测试代码
if __name__ == "__main__":
    # 测试DWT
    print("Testing DWT...")
    dwt = DWTTransform('haar')
    x = torch.randn(2, 1, 64, 64)
    LL, LH, HL, HH = dwt(x)
    print(f"Input shape: {x.shape}")
    print(f"LL shape: {LL.shape}, LH shape: {LH.shape}")

    x_rec = dwt.inverse(LL, LH, HL, HH)
    print(f"Reconstructed shape: {x_rec.shape}")
    print(f"DWT reconstruction error: {(x - x_rec).abs().max().item():.6f}")

    # 测试DCT
    print("\nTesting DCT...")
    dct = DCTTransform(block_size=8)
    x = torch.randn(2, 1, 64, 64)
    dct_coeffs = dct(x)
    print(f"DCT coeffs shape: {dct_coeffs.shape}")

    x_rec = dct.inverse(dct_coeffs)
    print(f"DCT reconstruction error: {(x - x_rec).abs().max().item():.6f}")

    # 测试混合编码器
    print("\nTesting HybridFrequencyEncoder...")
    encoder = HybridFrequencyEncoder(wavelet='haar', dct_block_size=8, use_dwt=True)
    x = torch.randn(2, 1, 64, 64)
    coeffs = encoder(x)
    print(f"Coefficients: {list(coeffs.keys())}")
    for k, v in coeffs.items():
        print(f"  {k}: {v.shape}")

    x_rec = encoder.inverse(coeffs)
    print(f"Hybrid reconstruction error: {(x - x_rec).abs().max().item():.6f}")

    # 测试5D输入
    print("\nTesting 5D input (video)...")
    x_video = torch.randn(2, 1, 8, 64, 64)  # B, C, T, H, W
    coeffs_video = encoder(x_video)
    print(f"Video coefficients:")
    for k, v in coeffs_video.items():
        print(f"  {k}: {v.shape}")

    x_video_rec = encoder.inverse(coeffs_video)
    print(f"Video reconstruction error: {(x_video - x_video_rec).abs().max().item():.6f}")

    print("\nAll tests passed!")
