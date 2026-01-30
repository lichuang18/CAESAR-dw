"""
CAESAR-Freq: 基于频率空间的压缩器
替代原VAE方案，使用DWT+DCT实现
"""

import torch
from collections import OrderedDict
from .models.run_gae_cuda import PCACompressor
import math
import torch.nn.functional as F
import numpy as np


def normalize_latent(x):
    x_min = torch.amin(x, dim=(1, 2, 3, 4), keepdim=True)
    x_max = torch.amax(x, dim=(1, 2, 3, 4), keepdim=True)

    scale = (x_max - x_min + 1e-8) / 2
    offset = x_min + scale

    x_norm = (x - offset) / scale  # result in [-1, 1]
    return x_norm, offset, scale


class CAESARFreq:
    """
    CAESAR-Freq: 频率空间压缩器
    使用DWT+DCT替代VAE编码器
    """
    def __init__(self,
                 model_path=None,
                 use_diffusion=True,
                 use_single_step=False,  # 是否使用单步扩散
                 device='cuda',
                 n_frame=16,
                 interpo_rate=3,
                 diffusion_steps=32,
                 wavelet='haar',
                 dct_block_size=8,
                 use_dwt=True,
                 low_freq_size=16,
                 ):
        self.pretrained_path = model_path
        self.use_diffusion = use_diffusion
        self.use_single_step = use_single_step
        self.device = device
        self.n_frame = n_frame
        self.diffusion_steps = diffusion_steps

        # 频率变换参数
        self.wavelet = wavelet
        self.dct_block_size = dct_block_size
        self.use_dwt = use_dwt
        self.low_freq_size = low_freq_size

        self._load_models()

        self.interpo_rate = interpo_rate
        self.cond_idx = torch.arange(0, n_frame, interpo_rate)
        self.pred_idx = ~torch.isin(torch.arange(n_frame), self.cond_idx)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def remove_module_prefix(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        return new_state_dict

    def _load_models(self):
        """加载模型"""
        if not self.use_diffusion:
            self._load_freq_compressor()
        else:
            self._load_freq_compressor_with_diffusion()

    def _load_freq_compressor(self):
        """加载频率压缩器（无扩散）"""
        from .models.frequency_compressor import FrequencyCompressor

        print("Loading CAESAR-Freq (without diffusion)")

        model = FrequencyCompressor(
            wavelet=self.wavelet,
            dct_block_size=self.dct_block_size,
            use_dwt=self.use_dwt,
            latent_channels=32,
            hyper_channels=32,
            low_freq_size=self.low_freq_size,
        )

        if self.pretrained_path:
            checkpoint = torch.load(self.pretrained_path, map_location=self.device)
            # 处理不同的 checkpoint 格式
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "freq_compressor" in checkpoint:
                state_dict = checkpoint["freq_compressor"]
            else:
                state_dict = checkpoint
            state_dict = self.remove_module_prefix(state_dict)
            model.load_state_dict(state_dict)

        self.freq_compressor = model.to(self.device).eval()

    def _load_freq_compressor_with_diffusion(self):
        """加载频率压缩器 + 扩散模型"""
        from .models.frequency_compressor import FrequencyCompressor

        print("Loading CAESAR-Freq (with diffusion)")

        # 频率压缩器
        freq_model = FrequencyCompressor(
            wavelet=self.wavelet,
            dct_block_size=self.dct_block_size,
            use_dwt=self.use_dwt,
            latent_channels=32,
            hyper_channels=32,
            low_freq_size=self.low_freq_size,
        )

        if self.pretrained_path:
            pretrained_models = torch.load(self.pretrained_path, map_location=self.device)

            if "freq_compressor" in pretrained_models:
                state_dict = self.remove_module_prefix(pretrained_models["freq_compressor"])
                freq_model.load_state_dict(state_dict)

        self.freq_compressor = freq_model.to(self.device).eval()

        # 扩散模型
        if self.use_single_step:
            self._load_single_step_diffusion()
        else:
            self._load_multi_step_diffusion()

    def _load_multi_step_diffusion(self):
        """加载多步扩散模型"""
        from .models.video_diffusion_interpo import Unet3D, GaussianDiffusion

        # 注意：频率空间的通道数可能不同
        latent_channels = 32  # 低频处理后的通道数

        model = Unet3D(
            dim=64,
            out_dim=latent_channels,
            channels=latent_channels,
            dim_mults=(1, 2, 4, 8),
            use_bert_text_cond=False
        )

        diffusion = GaussianDiffusion(
            model,
            image_size=self.low_freq_size,
            num_frames=10,
            channels=latent_channels,
            timesteps=self.diffusion_steps,
            loss_type='l2'
        )

        if self.pretrained_path:
            pretrained_models = torch.load(self.pretrained_path, map_location=self.device)
            if "diffusion" in pretrained_models:
                state_dict = self.remove_module_prefix(pretrained_models["diffusion"])
                diffusion.load_state_dict(state_dict)

        self.diffusion_model = diffusion.to(self.device).eval()

    def _load_single_step_diffusion(self):
        """加载单步扩散模型"""
        from .models.single_step_diffusion import SingleStepStudent, SingleStepDiffusion

        latent_channels = 32

        student = SingleStepStudent(
            in_channels=latent_channels * 2,  # 输入 + 条件
            out_channels=latent_channels,
            base_channels=64,
            channel_mults=(1, 2, 4),
            num_res_blocks=2,
            use_time_emb=False,
        )

        if self.pretrained_path:
            pretrained_models = torch.load(self.pretrained_path, map_location=self.device)
            if "single_step_diffusion" in pretrained_models:
                state_dict = self.remove_module_prefix(pretrained_models["single_step_diffusion"])
                student.load_state_dict(state_dict)

        self.diffusion_model = SingleStepDiffusion(student).to(self.device).eval()

    def compress(self, dataloader, eb=1e-3):
        """压缩数据"""
        dataset_org = dataloader.dataset
        self.transform_shape = dataset_org.deblocking_hw

        shape = dataset_org.data_input.shape

        if self.use_diffusion:
            compressed_latent, latent_bytes = self.compress_freq_d(dataloader)
            recons_data = self.decompress_freq_d(compressed_latent, shape, dataset_org.filtered_blocks)
        else:
            compressed_latent, latent_bytes = self.compress_freq_v(dataloader)
            recons_data = self.decompress_freq_v(compressed_latent, shape, dataset_org.filtered_blocks)

        recons_data = self.transform_shape(recons_data)

        original_data = dataset_org.original_data()
        original_data, org_padding = self.padding(original_data)
        recons_data, rec_padding = self.padding(recons_data)

        meta_data, compressed_gae = self.postprocessing_encoding(original_data, recons_data, eb)

        return {
            "latent": compressed_latent,
            "postprocess": compressed_gae,
            "meta_data": meta_data,
            "shape": shape,
            "padding": rec_padding,
            "filtered_blocks": dataset_org.filtered_blocks
        }, latent_bytes + meta_data["data_bytes"]

    def decompress(self, compressed):
        """解压缩数据"""
        shape = compressed["shape"]
        filtered_blocks = compressed["filtered_blocks"]

        if self.use_diffusion:
            recons_data = self.decompress_freq_d(compressed["latent"], shape, filtered_blocks)
        else:
            recons_data = self.decompress_freq_v(compressed["latent"], shape, filtered_blocks)

        recons_data = self.transform_shape(recons_data)
        recons_data, rec_padding = self.padding(recons_data)

        recons_data = self.postprocessing_decoding(
            recons_data, compressed["meta_data"], compressed["postprocess"], rec_padding
        )
        return recons_data

    def compress_freq_v(self, dataloader):
        """频率压缩（无扩散）"""
        total_bits = 0
        all_compressed_latent = []

        with torch.no_grad():
            for data in dataloader:
                outputs = self.freq_compressor.compress(data["input"].cuda())
                total_bits += torch.sum(outputs["bpf_real"])

                compressed_latent = {
                    "compressed": outputs["compressed"],
                    "scale": data["scale"],
                    "offset": data["offset"],
                    "index": data["index"]
                }
                all_compressed_latent.append(compressed_latent)

        return all_compressed_latent, total_bits / 8

    def decompress_freq_v(self, all_compressed, shape, filtered_blocks):
        """频率解压（无扩散）"""
        torch.manual_seed(2025)
        torch.cuda.manual_seed_all(2025)

        recons_data = torch.zeros(shape)

        with torch.no_grad():
            for compressed in all_compressed:
                rct_data = self.freq_compressor.decompress(
                    compressed["compressed"], device=self.device
                )
                rct_data = rct_data * compressed["scale"].cuda() + compressed["offset"].cuda()
                rct_data = rct_data.cpu()

                for i in range(rct_data.shape[0]):
                    idx0, idx1, start_t, end_t = compressed["index"]
                    recons_data[idx0[i], idx1[i], start_t[i]:end_t[i]] = rct_data[i]

        # 处理过滤的块
        if filtered_blocks:
            V, S, T, H, W = shape
            n_frame = self.n_frame
            samples = T // n_frame
            for label, value in filtered_blocks:
                v = label // (S * samples)
                remain = label % (S * samples)
                s = remain // samples
                blk_idx = remain % samples
                start = blk_idx * n_frame
                end = (blk_idx + 1) * n_frame
                recons_data[v, s, start:end, :, :] = value

        return recons_data

    def compress_freq_d(self, dataloader):
        """频率压缩（带扩散）"""
        total_bits = 0
        all_compressed_latent = []

        with torch.no_grad():
            for data in dataloader:
                # 只压缩关键帧
                keyframe = data["input"][:, :, self.cond_idx].cuda()
                outputs = self.freq_compressor.compress(keyframe, return_latent=True)
                total_bits += torch.sum(outputs["bpf_real"])

                compressed_latent = {
                    "compressed": outputs["compressed"],
                    "q_latent": outputs.get("q_latent"),
                    "scale": data["scale"],
                    "offset": data["offset"],
                    "index": data["index"]
                }
                all_compressed_latent.append(compressed_latent)

        return all_compressed_latent, total_bits / 8

    def decompress_freq_d(self, all_compressed, shape, filtered_blocks):
        """频率解压（带扩散）"""
        torch.manual_seed(2025)
        torch.cuda.manual_seed_all(2025)

        recons_data = torch.zeros(shape)

        with torch.no_grad():
            for compressed in all_compressed:
                # 解压关键帧的低频
                latent_data = self.freq_compressor.decompress(
                    compressed["compressed"], device=self.device
                )

                B, C, KT, H, W = latent_data.shape

                # 构建完整序列的潜在空间
                input_latent = torch.zeros([B, C, self.n_frame, H, W], device=self.device)
                input_latent[:, :, self.cond_idx] = latent_data

                # 归一化
                input_latent, offset_latent, scale_latent = normalize_latent(input_latent)

                # 扩散模型插值
                if self.use_single_step:
                    # 单步扩散
                    result = self.diffusion_model.sample(
                        condition=input_latent[:, :, self.cond_idx]
                    )
                else:
                    # 多步扩散
                    result = self.diffusion_model.sample(
                        input_latent, self.interpo_rate, batch_size=input_latent.shape[0]
                    )

                input_latent[:, :, self.pred_idx] = result

                # 反归一化
                input_latent = input_latent * scale_latent + offset_latent

                # 频率逆变换重建
                input_latent_2d = input_latent.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

                # 使用频率压缩器的解码器
                rct_data = self.freq_compressor.low_freq_processor.decode(input_latent_2d)

                # 频率逆变换
                # 这里需要完整的频率系数，简化处理：直接用低频重建
                # 实际应用中需要存储和恢复高频系数
                rct_data = rct_data.reshape([B, -1, 1, *rct_data.shape[-2:]])
                rct_data = rct_data * compressed["scale"].cuda() + compressed["offset"].cuda()
                rct_data = rct_data.cpu()

                for i in range(B):
                    idx0, idx1, start_t, end_t = compressed["index"]
                    recons_data[idx0[i], idx1[i], start_t[i]:end_t[i]] = rct_data[i]

        # 处理过滤的块
        if filtered_blocks:
            V, S, T, H, W = shape
            n_frame = self.n_frame
            samples = T // n_frame
            for label, value in filtered_blocks:
                v = label // (S * samples)
                remain = label % (S * samples)
                s = remain // samples
                blk_idx = remain % samples
                start = blk_idx * n_frame
                end = (blk_idx + 1) * n_frame
                recons_data[v, s, start:end, :, :] = value

        return recons_data

    def padding(self, data, block_size=(8, 8)):
        """填充数据"""
        *leading_dims, H, W = data.shape
        h_block, w_block = block_size

        H_target = math.ceil(H / h_block) * h_block
        W_target = math.ceil(W / w_block) * w_block
        dh = H_target - H
        dw = W_target - W
        top, down = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        data_reshaped = data.view(-1, H, W)
        data_padded = F.pad(data_reshaped, (left, right, top, down), mode='reflect')
        padded_data = data_padded.view(*leading_dims, *data_padded.shape[-2:])
        padding = (top, down, left, right)
        return padded_data, padding

    def unpadding(self, padded_data, padding):
        """去除填充"""
        top, down, left, right = padding
        *leading_dims, H, W = padded_data.shape
        unpadded_data = padded_data[..., top:H-down, left:W-right]
        return unpadded_data

    def postprocessing_encoding(self, original_data, recons_data, nrmse):
        """后处理编码"""
        x_min, x_max, offset = original_data.min(), original_data.max(), original_data.mean()
        scale = (x_max - x_min)

        original_data = (original_data - offset) / scale
        recons_data = (recons_data - offset) / scale

        self.compressor = PCACompressor(nrmse, 2, codec_algorithm="Zstd", device=self.device)
        meta_data, compressed_data, _ = self.compressor.compress(original_data, recons_data)

        meta_data["scale"] = scale
        meta_data["offset"] = offset

        return meta_data, compressed_data

    def postprocessing_decoding(self, recons_data, meta_data, compressed_data, padding):
        """后处理解码"""
        recons_data = (recons_data - meta_data["offset"]) / meta_data["scale"]

        if meta_data["data_bytes"] > 0:
            recons_data_gae = self.compressor.decompress(recons_data, meta_data, compressed_data, to_np=False)
        else:
            recons_data_gae = recons_data

        recons_data_gae = self.unpadding(recons_data_gae, padding)
        return recons_data_gae * meta_data["scale"] + meta_data["offset"]
