"""
快速测试脚本 - 验证频率变换模块的正确性
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_dwt():
    """测试DWT变换"""
    from CAESAR.models.frequency_transform import DWTTransform

    print("=" * 50)
    print("Testing DWT Transform")
    print("=" * 50)

    for wavelet in ['haar']:  # cdf97 需要更多调试
        print(f"\nWavelet: {wavelet}")
        dwt = DWTTransform(wavelet)

        # 测试不同尺寸
        for size in [64, 128, 256]:
            x = torch.randn(2, 1, size, size)
            LL, LH, HL, HH = dwt(x)

            print(f"  Input: {x.shape} -> LL: {LL.shape}, LH: {LH.shape}")

            # 逆变换
            x_rec = dwt.inverse(LL, LH, HL, HH)

            # 计算误差
            error = (x - x_rec).abs().max().item()
            print(f"  Reconstruction error: {error:.2e}")

            if error > 1e-5:
                print(f"  WARNING: Large reconstruction error!")


def test_dct():
    """测试DCT变换"""
    from CAESAR.models.frequency_transform import DCTTransform

    print("\n" + "=" * 50)
    print("Testing DCT Transform")
    print("=" * 50)

    for block_size in [8, 16]:
        print(f"\nBlock size: {block_size}")
        dct = DCTTransform(block_size)

        for size in [64, 128, 256]:
            x = torch.randn(2, 1, size, size)
            dct_coeffs = dct(x)

            print(f"  Input: {x.shape} -> DCT: {dct_coeffs.shape}")

            # 逆变换
            x_rec = dct.inverse(dct_coeffs)

            # 计算误差
            error = (x - x_rec).abs().max().item()
            print(f"  Reconstruction error: {error:.2e}")

            if error > 1e-5:
                print(f"  WARNING: Large reconstruction error!")


def test_hybrid_encoder():
    """测试混合编码器"""
    from CAESAR.models.frequency_transform import HybridFrequencyEncoder

    print("\n" + "=" * 50)
    print("Testing Hybrid Frequency Encoder")
    print("=" * 50)

    # 测试DWT+DCT模式
    print("\nMode: DWT + DCT")
    encoder = HybridFrequencyEncoder(wavelet='haar', dct_block_size=8, use_dwt=True)

    x = torch.randn(2, 1, 256, 256)
    coeffs = encoder(x)

    print(f"Input: {x.shape}")
    print("Coefficients:")
    for k, v in coeffs.items():
        print(f"  {k}: {v.shape}")

    x_rec = encoder.inverse(coeffs)
    error = (x - x_rec).abs().max().item()
    print(f"Reconstruction error: {error:.2e}")

    # 测试纯DCT模式
    print("\nMode: DCT only")
    encoder_dct = HybridFrequencyEncoder(wavelet='haar', dct_block_size=8, use_dwt=False)

    coeffs_dct = encoder_dct(x)
    print(f"Input: {x.shape}")
    print("Coefficients:")
    for k, v in coeffs_dct.items():
        print(f"  {k}: {v.shape}")

    x_rec_dct = encoder_dct.inverse(coeffs_dct)
    error_dct = (x - x_rec_dct).abs().max().item()
    print(f"Reconstruction error: {error_dct:.2e}")

    # 测试5D输入（视频）
    print("\nMode: 5D input (video)")
    x_video = torch.randn(2, 1, 8, 256, 256)  # B, C, T, H, W
    coeffs_video = encoder(x_video)

    print(f"Input: {x_video.shape}")
    print("Coefficients:")
    for k, v in coeffs_video.items():
        print(f"  {k}: {v.shape}")

    x_video_rec = encoder.inverse(coeffs_video)
    error_video = (x_video - x_video_rec).abs().max().item()
    print(f"Reconstruction error: {error_video:.2e}")


def test_frequency_compressor():
    """测试频率压缩器"""
    from CAESAR.models.frequency_compressor import FrequencyCompressor

    print("\n" + "=" * 50)
    print("Testing Frequency Compressor")
    print("=" * 50)

    model = FrequencyCompressor(
        wavelet='haar',
        dct_block_size=8,
        use_dwt=True,
        latent_channels=32,
        hyper_channels=32,
        low_freq_size=16,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    x = torch.randn(2, 1, 16, 256, 256)  # B, C, T, H, W
    print(f"\nInput: {x.shape}")

    model.train()
    outputs = model(x)

    print("Outputs:")
    print(f"  output: {outputs['output'].shape}")
    print(f"  bpp: {outputs['bpp'].mean().item():.4f}")
    print(f"  latent: {outputs['latent'].shape}")

    # 计算重建误差
    recon_error = (x - outputs['output']).abs().mean().item()
    print(f"  reconstruction error (mean): {recon_error:.4f}")

    # 测试压缩/解压
    print("\nTesting compress/decompress...")
    model.eval()

    with torch.no_grad():
        compressed = model.compress(x)
        print(f"  Compressed data keys: {list(compressed['compressed'].keys())}")
        print(f"  Bits per frame: {compressed['bpf_real'].mean().item():.1f}")

        # 解压 (使用 CPU)
        x_rec = model.decompress(compressed['compressed'], device='cpu')
        print(f"  Decompressed: {x_rec.shape}")

        decomp_error = (x - x_rec).abs().mean().item()
        print(f"  Decompression error (mean): {decomp_error:.4f}")


def test_single_step_student():
    """测试单步Student网络"""
    from CAESAR.models.single_step_diffusion import SingleStepStudent

    print("\n" + "=" * 50)
    print("Testing Single Step Student")
    print("=" * 50)

    student = SingleStepStudent(
        in_channels=64,  # 32 * 2 (noise + condition)
        out_channels=32,
        base_channels=64,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        use_time_emb=False,
    )

    print(f"Model parameters: {sum(p.numel() for p in student.parameters()):,}")

    # 测试前向传播
    noise = torch.randn(2, 32, 10, 16, 16)  # B, C, T, H, W
    condition = torch.randn(2, 32, 10, 16, 16)  # 条件，T 维度需与 noise 一致

    print(f"\nNoise: {noise.shape}, Condition: {condition.shape}")

    output = student(noise, condition=condition, t=None)
    print(f"Output: {output.shape}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("CAESAR-Freq Module Tests")
    print("=" * 60)

    try:
        test_dwt()
        print("\n✓ DWT test passed")
    except Exception as e:
        print(f"\n✗ DWT test failed: {e}")

    try:
        test_dct()
        print("\n✓ DCT test passed")
    except Exception as e:
        print(f"\n✗ DCT test failed: {e}")

    try:
        test_hybrid_encoder()
        print("\n✓ Hybrid encoder test passed")
    except Exception as e:
        print(f"\n✗ Hybrid encoder test failed: {e}")

    try:
        test_frequency_compressor()
        print("\n✓ Frequency compressor test passed")
    except Exception as e:
        print(f"\n✗ Frequency compressor test failed: {e}")

    try:
        test_single_step_student()
        print("\n✓ Single step student test passed")
    except Exception as e:
        print(f"\n✗ Single step student test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
