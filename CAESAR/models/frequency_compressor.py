"""
频率空间压缩器
替代原VAE编码器，使用DWT+DCT实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .frequency_transform import HybridFrequencyEncoder, FrequencySeparator, SparseQuantizer
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution
from .RangeEncoding import RangeCoder


class LowFreqProcessor(nn.Module):
    """
    低频处理网络
    - 输入: LL-DCT系数的低频部分
    - 不做空间下采样，保持空间分辨率
    - 只做通道变换用于熵编码
    """
    def __init__(self, in_channels: int = 1, latent_channels: int = 32,
                 hyper_channels: int = 32):
        super().__init__()

        self.latent_channels = latent_channels

        # 编码器：通道扩展，无空间下采样
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, latent_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # 解码器：通道压缩
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, in_channels, 3, padding=1),
        )

        # 超先验编码器
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_channels, hyper_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_channels, hyper_channels, 3, stride=2, padding=1),
        )

        # 超先验解码器
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hyper_channels, latent_channels * 2, 3, padding=1),  # mean and scale
        )

        # 先验模型
        self.prior = FlexiblePrior(hyper_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码低频系数"""
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """解码低频系数"""
        return self.decoder(x)

    def hyper_encode(self, latent: torch.Tensor) -> torch.Tensor:
        """超先验编码"""
        return self.hyper_encoder(latent)

    def hyper_decode(self, hyper_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """超先验解码，返回mean和scale"""
        x = self.hyper_decoder(hyper_latent)
        mean, scale = x.chunk(2, dim=1)
        return mean, scale.clamp(min=0.1)


class FrequencyCompressor(nn.Module):
    """
    频率空间压缩器
    使用DWT+DCT替代VAE编码器
    """
    def __init__(
        self,
        wavelet: str = 'haar',
        dct_block_size: int = 8,
        use_dwt: bool = True,
        latent_channels: int = 32,
        hyper_channels: int = 32,
        low_freq_size: int = 16,  # 低频区域大小
    ):
        super().__init__()

        self.use_dwt = use_dwt
        self.low_freq_size = low_freq_size

        # 频率变换编码器
        self.freq_encoder = HybridFrequencyEncoder(
            wavelet=wavelet,
            dct_block_size=dct_block_size,
            use_dwt=use_dwt
        )

        # 低频处理网络
        self.low_freq_processor = LowFreqProcessor(
            in_channels=1,
            latent_channels=latent_channels,
            hyper_channels=hyper_channels
        )

        # 稀疏量化器（用于中高频）
        self.sparse_quantizer = SparseQuantizer(default_threshold=0.01)

        # Range编码器
        self.range_coder = None

        # 量化步长（可学习或固定）
        self.register_buffer('q_step_mid', torch.tensor(1.0))
        self.register_buffer('q_step_high', torch.tensor(2.0))

    @property
    def prior(self):
        return self.low_freq_processor.prior

    def _extract_low_freq(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        """提取DCT系数的低频部分"""
        return dct_coeffs[:, :, :self.low_freq_size, :self.low_freq_size]

    def _embed_low_freq(self, low_freq: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """将低频嵌入到完整DCT系数中"""
        B, C, H, W = target_shape
        dct_coeffs = torch.zeros(B, C, H, W, device=low_freq.device, dtype=low_freq.dtype)
        dct_coeffs[:, :, :self.low_freq_size, :self.low_freq_size] = low_freq
        return dct_coeffs

    def forward(self, x: torch.Tensor) -> Dict:
        """
        前向传播（训练用）
        Args:
            x: [B, C, T, H, W] 输入数据
        Returns:
            dict: 包含输出、bpp等信息
        """
        B, C, T, H, W = x.shape

        # 1. 频率变换
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        freq_coeffs = self.freq_encoder(x_2d)

        if self.use_dwt:
            LL_dct = freq_coeffs['LL_dct']
            LH = freq_coeffs['LH']
            HL = freq_coeffs['HL']
            HH = freq_coeffs['HH']

            # 2. 提取低频并处理
            low_freq = self._extract_low_freq(LL_dct)

            # 3. 低频编码
            latent = self.low_freq_processor.encode(low_freq)
            hyper_latent = self.low_freq_processor.hyper_encode(latent)

            # 4. 量化
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            mean, scale = self.low_freq_processor.hyper_decode(q_hyper_latent)
            q_latent = quantize(latent, "dequantize", mean)

            # 5. 解码低频
            low_freq_rec = self.low_freq_processor.decode(q_latent)

            # 6. 重建LL_dct
            LL_dct_rec = self._embed_low_freq(low_freq_rec, LL_dct.shape)
            # 保留高频部分
            LL_dct_rec[:, :, self.low_freq_size:, :] = LL_dct[:, :, self.low_freq_size:, :]
            LL_dct_rec[:, :, :, self.low_freq_size:] = LL_dct[:, :, :, self.low_freq_size:]

            # 7. 中高频量化（简单量化，训练时可以加噪声）
            if self.training:
                LH_q = LH + torch.randn_like(LH) * 0.5
                HL_q = HL + torch.randn_like(HL) * 0.5
                HH_q = HH + torch.randn_like(HH) * 0.5
            else:
                LH_q = torch.round(LH / self.q_step_mid) * self.q_step_mid
                HL_q = torch.round(HL / self.q_step_mid) * self.q_step_mid
                HH_q = torch.round(HH / self.q_step_high) * self.q_step_high

            # 8. 频率逆变换
            freq_coeffs_rec = {
                'LL_dct': LL_dct_rec,
                'LH': LH_q,
                'HL': HL_q,
                'HH': HH_q
            }
        else:
            # 纯DCT模式
            dct_coeffs = freq_coeffs['dct']
            low_freq = self._extract_low_freq(dct_coeffs)

            latent = self.low_freq_processor.encode(low_freq)
            hyper_latent = self.low_freq_processor.hyper_encode(latent)
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            mean, scale = self.low_freq_processor.hyper_decode(q_hyper_latent)
            q_latent = quantize(latent, "dequantize", mean)
            low_freq_rec = self.low_freq_processor.decode(q_latent)

            dct_rec = self._embed_low_freq(low_freq_rec, dct_coeffs.shape)
            dct_rec[:, :, self.low_freq_size:, :] = dct_coeffs[:, :, self.low_freq_size:, :]
            dct_rec[:, :, :, self.low_freq_size:] = dct_coeffs[:, :, :, self.low_freq_size:]

            freq_coeffs_rec = {'dct': dct_rec}

        # 9. 逆变换回空间域
        output = self.freq_encoder.inverse(freq_coeffs_rec)
        output = output.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        # 10. 计算bpp
        frame_bit, bpp = self._compute_bpp(
            latent, hyper_latent, mean, scale, (B * T, C, H, W)
        )

        return {
            'output': output,
            'bpp': bpp,
            'frame_bit': frame_bit,
            'latent': latent,
            'hyper_latent': hyper_latent,
            'q_latent': q_latent,
            'mean': mean,
            'scale': scale,
        }

    def _compute_bpp(self, latent, hyper_latent, mean, scale, shape):
        """计算比特率"""
        B, C, H, W = shape

        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))

        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)

        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()

        frame_bit_latent = cond_rate.sum(dim=(1, 2, 3))
        frame_bit_hyper = hyper_rate.sum(dim=(1, 2, 3))
        frame_bit = frame_bit_latent + frame_bit_hyper
        bpp = frame_bit / (H * W)

        return frame_bit, bpp

    def compress(self, x: torch.Tensor, return_latent: bool = False) -> Dict:
        """
        压缩
        Args:
            x: [B, C, T, H, W] 输入数据
        Returns:
            dict: 压缩结果
        """
        if self.range_coder is None:
            _quantized_cdf, _cdf_length, _offset = self.prior._update(30)
            self.range_coder = RangeCoder(
                _quantized_cdf=_quantized_cdf,
                _cdf_length=_cdf_length,
                _offset=_offset,
                medians=self.prior.medians.detach()
            )

        B, C, T, H, W = x.shape
        original_shape = x.shape

        # 频率变换
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        freq_coeffs = self.freq_encoder(x_2d)

        if self.use_dwt:
            LL_dct = freq_coeffs['LL_dct']
            LH = freq_coeffs['LH']
            HL = freq_coeffs['HL']
            HH = freq_coeffs['HH']

            # 低频处理
            low_freq = self._extract_low_freq(LL_dct)
            latent = self.low_freq_processor.encode(low_freq)
            hyper_latent = self.low_freq_processor.hyper_encode(latent)
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            mean, scale = self.low_freq_processor.hyper_decode(q_hyper_latent)

            # Range编码
            latent_string = self.range_coder.compress(latent, mean, scale)
            hyper_latent_string = self.range_coder.compress_hyperlatent(hyper_latent)

            # 中高频量化
            LH_q = torch.round(LH / self.q_step_mid) * self.q_step_mid
            HL_q = torch.round(HL / self.q_step_mid) * self.q_step_mid
            HH_q = torch.round(HH / self.q_step_high) * self.q_step_high

            # 高频系数（需要额外存储）
            high_freq_data = {
                'LL_dct_high': LL_dct[:, :, self.low_freq_size:, :].clone(),
                'LL_dct_high2': LL_dct[:, :, :, self.low_freq_size:].clone(),
                'LH': LH_q,
                'HL': HL_q,
                'HH': HH_q,
            }
        else:
            dct_coeffs = freq_coeffs['dct']
            low_freq = self._extract_low_freq(dct_coeffs)
            latent = self.low_freq_processor.encode(low_freq)
            hyper_latent = self.low_freq_processor.hyper_encode(latent)
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            mean, scale = self.low_freq_processor.hyper_decode(q_hyper_latent)

            latent_string = self.range_coder.compress(latent, mean, scale)
            hyper_latent_string = self.range_coder.compress_hyperlatent(hyper_latent)

            high_freq_data = {
                'dct_high': dct_coeffs[:, :, self.low_freq_size:, :].clone(),
                'dct_high2': dct_coeffs[:, :, :, self.low_freq_size:].clone(),
            }

        bpf_real = torch.Tensor([(len(lc) + len(hc)) * 8
                                  for lc, hc in zip(latent_string, hyper_latent_string)])

        compressed_data = {
            'latent_string': latent_string,
            'hyper_latent_string': hyper_latent_string,
            'original_shape': original_shape,
            'hyper_shape': hyper_latent.shape,
            'high_freq_data': high_freq_data,
        }

        result = {
            'compressed': compressed_data,
            'bpf_real': bpf_real,
        }

        if return_latent:
            q_latent = quantize(latent, "dequantize", mean)
            result['q_latent'] = q_latent.reshape(B, T, *q_latent.shape[-3:])

        return result

    def decompress(self, compressed_data: Dict, device: str = 'cuda') -> torch.Tensor:
        """
        解压缩
        Args:
            compressed_data: 压缩数据
            device: 设备
        Returns:
            重建数据 [B, C, T, H, W]
        """
        latent_string = compressed_data['latent_string']
        hyper_latent_string = compressed_data['hyper_latent_string']
        original_shape = compressed_data['original_shape']
        hyper_shape = compressed_data['hyper_shape']
        high_freq_data = compressed_data['high_freq_data']

        B, C, T, H, W = original_shape

        # 解码超先验
        q_hyper_latent = self.range_coder.decompress_hyperlatent(hyper_latent_string, hyper_shape)
        mean, scale = self.low_freq_processor.hyper_decode(q_hyper_latent.to(device))

        # 解码潜在变量
        q_latent = self.range_coder.decompress(latent_string, mean.detach().cpu(), scale.detach().cpu())
        q_latent = q_latent.to(device)

        # 解码低频
        low_freq_rec = self.low_freq_processor.decode(q_latent)

        if self.use_dwt:
            # 重建LL_dct
            LL_h = H // 2
            LL_w = W // 2
            LL_dct_rec = torch.zeros(B * T, C, LL_h, LL_w, device=device)
            LL_dct_rec[:, :, :self.low_freq_size, :self.low_freq_size] = low_freq_rec
            LL_dct_rec[:, :, self.low_freq_size:, :] = high_freq_data['LL_dct_high'].to(device)
            LL_dct_rec[:, :, :, self.low_freq_size:] = high_freq_data['LL_dct_high2'].to(device)

            freq_coeffs_rec = {
                'LL_dct': LL_dct_rec,
                'LH': high_freq_data['LH'].to(device),
                'HL': high_freq_data['HL'].to(device),
                'HH': high_freq_data['HH'].to(device),
            }
        else:
            dct_rec = torch.zeros(B * T, C, H, W, device=device)
            dct_rec[:, :, :self.low_freq_size, :self.low_freq_size] = low_freq_rec
            dct_rec[:, :, self.low_freq_size:, :] = high_freq_data['dct_high'].to(device)
            dct_rec[:, :, :, self.low_freq_size:] = high_freq_data['dct_high2'].to(device)

            freq_coeffs_rec = {'dct': dct_rec}

        # 逆变换
        output = self.freq_encoder.inverse(freq_coeffs_rec)
        output = output.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return output

    def get_extra_loss(self):
        """获取额外损失（先验正则化）"""
        return self.prior.get_extraloss()


class FrequencyCompressorSR(nn.Module):
    """
    带超分辨率的频率压缩器（兼容原接口）
    注意：频率方案不需要SR，此类仅用于接口兼容
    """
    def __init__(
        self,
        wavelet: str = 'haar',
        dct_block_size: int = 8,
        use_dwt: bool = True,
        latent_channels: int = 32,
        hyper_channels: int = 32,
        low_freq_size: int = 16,
    ):
        super().__init__()

        self.compressor = FrequencyCompressor(
            wavelet=wavelet,
            dct_block_size=dct_block_size,
            use_dwt=use_dwt,
            latent_channels=latent_channels,
            hyper_channels=hyper_channels,
            low_freq_size=low_freq_size,
        )

    def forward(self, x: torch.Tensor) -> Dict:
        return self.compressor(x)

    def compress(self, x: torch.Tensor, return_latent: bool = False) -> Dict:
        return self.compressor.compress(x, return_latent)

    def decompress(self, compressed_data: Dict, device: str = 'cuda') -> torch.Tensor:
        return self.compressor.decompress(compressed_data, device)

    def get_extra_loss(self):
        return self.compressor.get_extra_loss()

    def inference_qlatent(self, x: torch.Tensor) -> torch.Tensor:
        """获取量化后的潜在变量（用于扩散模型）"""
        result = self.compressor.compress(x, return_latent=True)
        return result['q_latent']
