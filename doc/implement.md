# CAESAR-Freq 实现细节

## 模块概述

CAESAR-Freq 使用 DWT+DCT 频率变换替代原 VAE 编码器，主要包含以下模块：

```
CAESAR/models/
├── frequency_transform.py    # DWT + DCT 频率变换
├── frequency_compressor.py   # 频率空间压缩器
└── single_step_diffusion.py  # 单步扩散蒸馏
```

---

## 1. DWT 变换 (frequency_transform.py)

### 设计选择

采用 **Haar 小波** 的直接索引实现，而非卷积滤波器方式。

**原因**：
- 卷积实现存在 `F.pad` 的 `mode='reflect'` 兼容性问题
- 卷积后尺寸计算复杂，容易出错
- 直接索引实现简单高效，无精度损失

### 实现

```python
def forward(self, x):
    # 提取 2x2 块的四个位置
    x00 = x[:, :, 0::2, 0::2]  # 左上
    x01 = x[:, :, 0::2, 1::2]  # 右上
    x10 = x[:, :, 1::2, 0::2]  # 左下
    x11 = x[:, :, 1::2, 1::2]  # 右下

    # Haar 变换
    LL = (x00 + x01 + x10 + x11) / 2.0  # 低频 (平均)
    LH = (x00 + x01 - x10 - x11) / 2.0  # 水平细节
    HL = (x00 - x01 + x10 - x11) / 2.0  # 垂直细节
    HH = (x00 - x01 - x10 + x11) / 2.0  # 对角细节

    return LL, LH, HL, HH

def inverse(self, LL, LH, HL, HH):
    # Haar 逆变换
    x00 = (LL + LH + HL + HH) / 2.0
    x01 = (LL + LH - HL - HH) / 2.0
    x10 = (LL - LH + HL - HH) / 2.0
    x11 = (LL - LH - HL + HH) / 2.0

    # 重组到原始位置
    x = torch.zeros(B, C, H * 2, W * 2, ...)
    x[:, :, 0::2, 0::2] = x00
    x[:, :, 0::2, 1::2] = x01
    x[:, :, 1::2, 0::2] = x10
    x[:, :, 1::2, 1::2] = x11

    return x
```

### 性能

| 输入尺寸 | 输出尺寸 | 重建误差 |
|----------|----------|----------|
| [B,C,64,64] | [B,C,32,32] | ~2e-7 |
| [B,C,128,128] | [B,C,64,64] | ~3e-7 |
| [B,C,256,256] | [B,C,128,128] | ~5e-7 |

---

## 2. DCT 变换 (frequency_transform.py)

### 实现

使用预计算的 DCT-II 变换矩阵，块级 2D DCT：

```python
# 2D DCT: Y = D @ X @ D^T
x = torch.matmul(self.dct_matrix, x)
x = torch.matmul(x, self.idct_matrix)
```

### 性能

重建误差: ~1e-6 (浮点精度限制)

---

## 3. SingleStepStudent (single_step_diffusion.py)

### Skip Connection 设计

编码器-解码器结构，关键点是 **skip connection 的保存时机**：

```python
# 编码器：在 downsample 之前保存 skip
for i, block in enumerate(self.encoder):
    h = block(h, t_emb)
    if (i + 1) % 2 == 0:
        skips.append(h)  # 保存当前分辨率的特征
        if down_idx < len(self.downsample):
            h = self.downsample[down_idx](h)
            down_idx += 1

# 解码器：在 upsample 之后使用 skip
for i, block in enumerate(self.decoder):
    if i % 3 == 0 and skips:
        h = torch.cat([h, skips.pop()], dim=1)  # 维度匹配
    h = block(h, t_emb)
    if (i + 1) % 3 == 0 and up_idx < len(self.upsample):
        h = self.upsample[up_idx](h)
        up_idx += 1
```

**关键**：skip 在 downsample 前保存，在 upsample 后使用，确保空间维度匹配。

---

## 4. FrequencyCompressor (frequency_compressor.py)

### 架构

```
输入 [B,C,T,H,W]
    ↓
DWT 分解 → LL, LH, HL, HH
    ↓
DCT(LL) → LL_dct
    ↓
提取低频 → low_freq [B,C,low_freq_size,low_freq_size]
    ↓
LowFreqProcessor 编码 → latent
    ↓
超先验编码 → hyper_latent
    ↓
Range 编码 → 压缩码流
```

### 注意事项

`decompress` 方法的 `device` 参数需要与模型所在设备一致：

```python
# 模型在 CPU 上时
x_rec = model.decompress(compressed['compressed'], device='cpu')

# 模型在 GPU 上时
x_rec = model.decompress(compressed['compressed'], device='cuda')
```

---

## 测试验证

运行测试：

```bash
conda activate caesar
python test_freq_modules.py
```

预期输出：

```
✓ DWT test passed
✓ DCT test passed
✓ Hybrid encoder test passed
✓ Frequency compressor test passed
✓ Single step student test passed
```

---

## 已知限制

1. **DWT 仅支持 Haar 小波**：CDF 9/7 小波需要更复杂的边界处理
2. **输入尺寸要求**：H, W 必须是 2 的倍数（DWT）且是 block_size 的倍数（DCT）
3. **重建误差**：频率压缩器的重建误差约 0.3（有损压缩，可通过调整量化步长控制）
