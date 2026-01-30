"""
S3D 数据集预处理脚本
将原始 .d64 二进制文件转换为训练用的 .npz 格式
"""

import numpy as np
import os
from glob import glob
from tqdm import tqdm
import argparse


def load_s3d_file(filepath, shape=(11, 500, 500, 500), dtype=np.float64):
    """
    加载 S3D 二进制文件

    Args:
        filepath: .d64 文件路径
        shape: 数据形状 (variables, X, Y, Z)
        dtype: 数据类型 (double = float64)

    Returns:
        numpy array of shape (V, X, Y, Z)
    """
    data = np.fromfile(filepath, dtype=dtype)
    data = data.reshape(shape)
    return data


def convert_s3d_to_npz(
    input_dir: str,
    output_path: str,
    variable_indices: list = None,
    spatial_crop: tuple = None,
    z_as_time: bool = True,
    num_slices: int = 10,
):
    """
    将 S3D 数据转换为 npz 格式

    Args:
        input_dir: S3D 数据目录
        output_path: 输出 npz 文件路径
        variable_indices: 要使用的变量索引列表，None 表示全部
        spatial_crop: 空间裁剪 (x_start, x_end, y_start, y_end)
        z_as_time: 是否将 Z 轴作为时间轴
        num_slices: 切片数量（沿 X 轴）

    输出格式: [V, S, T, H, W]
        V: 变量数
        S: 切片数
        T: 时间步数 (Z轴)
        H: 高度 (Y轴)
        W: 宽度 (X轴 或 剩余X)
    """
    # 查找所有 .d64 文件
    d64_files = sorted(glob(os.path.join(input_dir, "*.d64")))

    if len(d64_files) == 0:
        raise FileNotFoundError(f"No .d64 files found in {input_dir}")

    print(f"Found {len(d64_files)} .d64 files")

    all_data = []

    for filepath in tqdm(d64_files, desc="Loading files"):
        print(f"Loading {os.path.basename(filepath)}...")

        # 加载数据 [11, 500, 500, 500]
        data = load_s3d_file(filepath)

        # 选择变量
        if variable_indices is not None:
            data = data[variable_indices]

        # 空间裁剪
        if spatial_crop is not None:
            x_start, x_end, y_start, y_end = spatial_crop
            data = data[:, x_start:x_end, y_start:y_end, :]

        # 转换为 float32 节省空间
        data = data.astype(np.float32)

        all_data.append(data)

    # 合并所有时间步（每个文件是一个时间步）
    # 当前 shape: list of [V, X, Y, Z]

    if z_as_time:
        # 将 Z 轴作为时间轴
        # 输出: [V, S, T, H, W] 其中 T=Z, H=Y, W=X/num_slices

        # 先处理单个文件
        data = all_data[0]  # [V, X, Y, Z]
        V, X, Y, Z = data.shape

        # 沿 X 轴切片
        slice_width = X // num_slices

        slices = []
        for s in range(num_slices):
            x_start = s * slice_width
            x_end = (s + 1) * slice_width
            slice_data = data[:, x_start:x_end, :, :]  # [V, slice_width, Y, Z]
            # 重排为 [V, T, H, W] = [V, Z, Y, slice_width]
            slice_data = slice_data.transpose(0, 3, 2, 1)  # [V, Z, Y, slice_width]
            slices.append(slice_data)

        # 合并切片: [V, S, T, H, W]
        result = np.stack(slices, axis=1)

    else:
        # 将多个文件作为时间步
        # 输出: [V, S, T, H, W] 其中 T=文件数, H=Y, W=X

        # 沿 Z 轴切片
        data = all_data[0]
        V, X, Y, Z = data.shape
        slice_depth = Z // num_slices

        result_slices = []
        for s in range(num_slices):
            z_start = s * slice_depth
            z_end = (s + 1) * slice_depth

            time_steps = []
            for data in all_data:
                # 取 Z 切片的中间层
                z_mid = (z_start + z_end) // 2
                slice_data = data[:, :, :, z_mid]  # [V, X, Y]
                slice_data = slice_data.transpose(0, 2, 1)  # [V, Y, X] = [V, H, W]
                time_steps.append(slice_data)

            # [V, T, H, W]
            slice_result = np.stack(time_steps, axis=1)
            result_slices.append(slice_result)

        # [V, S, T, H, W]
        result = np.stack(result_slices, axis=1)

    print(f"Output shape: {result.shape}")
    print(f"  V (variables): {result.shape[0]}")
    print(f"  S (slices): {result.shape[1]}")
    print(f"  T (time): {result.shape[2]}")
    print(f"  H (height): {result.shape[3]}")
    print(f"  W (width): {result.shape[4]}")

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, data=result)
    print(f"Saved to {output_path}")

    # 打印文件大小
    file_size = os.path.getsize(output_path) / (1024 ** 3)
    print(f"File size: {file_size:.2f} GB")

    return result


def create_small_test_data(output_path: str, shape=(1, 4, 64, 256, 256)):
    """
    创建小型测试数据用于验证训练流程（随机数据）

    Args:
        output_path: 输出路径
        shape: (V, S, T, H, W)
    """
    print(f"Creating random test data with shape {shape}")

    # 生成随机数据
    data = np.random.randn(*shape).astype(np.float32)

    # 添加一些结构（模拟真实数据的空间相关性）
    from scipy.ndimage import gaussian_filter
    for v in range(shape[0]):
        for s in range(shape[1]):
            for t in range(shape[2]):
                data[v, s, t] = gaussian_filter(data[v, s, t], sigma=2)

    # 归一化到 [-1, 1]
    data = (data - data.mean()) / (data.std() + 1e-8)
    data = np.clip(data, -3, 3) / 3

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, data=data)

    print(f"Saved test data to {output_path}")
    print(f"Shape: {data.shape}")
    print(f"Range: [{data.min():.3f}, {data.max():.3f}]")

    return data


def extract_s3d_sample(
    input_dir: str,
    output_path: str,
    variable_idx: int = 6,
    num_slices: int = 4,
    t_frames: int = 64,
    spatial_size: int = 256,
):
    """
    从 S3D 数据中提取小样本用于测试

    Args:
        input_dir: S3D 数据目录
        output_path: 输出路径
        variable_idx: 使用的变量索引 (默认 6 = 温度)
        num_slices: 切片数
        t_frames: 时间帧数 (从 Z 轴采样)
        spatial_size: 空间尺寸 (H, W)

    输出: [1, num_slices, t_frames, spatial_size, spatial_size]
    """
    d64_files = sorted(glob(os.path.join(input_dir, "*.d64")))

    if len(d64_files) == 0:
        raise FileNotFoundError(f"No .d64 files found in {input_dir}")

    # 只读取第一个文件
    filepath = d64_files[0]
    print(f"Loading {os.path.basename(filepath)}...")
    print("This may take a while for large files...")

    # 加载数据 [11, 500, 500, 500]
    data = load_s3d_file(filepath)
    print(f"Loaded shape: {data.shape}")

    # 选择单个变量
    data = data[variable_idx]  # [500, 500, 500] = [X, Y, Z]
    print(f"Selected variable {variable_idx}, shape: {data.shape}")

    X, Y, Z = data.shape

    # 计算采样参数，确保不越界
    # Y 轴居中裁剪
    y_start = (Y - spatial_size) // 2
    y_end = y_start + spatial_size

    # Z 轴均匀采样
    z_indices = np.linspace(0, Z - 1, t_frames, dtype=int)

    # X 轴分成 num_slices 个区域，每个区域取 spatial_size
    usable_x = X - spatial_size  # 可用的起始位置范围
    if num_slices > 1:
        x_starts = np.linspace(0, usable_x, num_slices, dtype=int)
    else:
        x_starts = [(X - spatial_size) // 2]

    print(f"Extracting {num_slices} slices, {t_frames} frames, {spatial_size}x{spatial_size} spatial")
    print(f"Y range: [{y_start}, {y_end})")
    print(f"X starts: {x_starts}")
    print(f"Z indices: {z_indices[0]}...{z_indices[-1]}")

    slices = []
    for s, x_start in enumerate(x_starts):
        x_end = x_start + spatial_size

        frames = []
        for z_idx in z_indices:
            # 提取 [spatial_size, spatial_size]
            frame = data[x_start:x_end, y_start:y_end, z_idx]
            frames.append(frame)

        # [T, H, W]
        slice_data = np.stack(frames, axis=0)
        print(f"  Slice {s}: shape {slice_data.shape}")
        slices.append(slice_data)

    # [S, T, H, W]
    result = np.stack(slices, axis=0)

    # 添加变量维度 [V, S, T, H, W]
    result = result[np.newaxis, ...]

    # 转换为 float32
    result = result.astype(np.float32)

    print(f"Output shape: {result.shape}")
    print(f"  V (variables): {result.shape[0]}")
    print(f"  S (slices): {result.shape[1]}")
    print(f"  T (time): {result.shape[2]}")
    print(f"  H (height): {result.shape[3]}")
    print(f"  W (width): {result.shape[4]}")
    print(f"Value range: [{result.min():.6f}, {result.max():.6f}]")

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, data=result)

    file_size = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Saved to {output_path} ({file_size:.1f} MB)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare S3D data for training")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # 转换命令
    convert_parser = subparsers.add_parser("convert", help="Convert S3D .d64 to .npz")
    convert_parser.add_argument("--input_dir", type=str, required=True,
                                help="S3D data directory containing .d64 files")
    convert_parser.add_argument("--output", type=str, required=True,
                                help="Output .npz file path")
    convert_parser.add_argument("--variables", type=int, nargs="+", default=None,
                                help="Variable indices to use (default: all)")
    convert_parser.add_argument("--num_slices", type=int, default=10,
                                help="Number of slices")
    convert_parser.add_argument("--z_as_time", action="store_true", default=True,
                                help="Use Z axis as time")

    # 提取小样本命令（从真实 S3D 数据）
    extract_parser = subparsers.add_parser("extract", help="Extract small sample from S3D for testing")
    extract_parser.add_argument("--input_dir", type=str, required=True,
                                help="S3D data directory")
    extract_parser.add_argument("--output", type=str, required=True,
                                help="Output .npz file path")
    extract_parser.add_argument("--variable", type=int, default=6,
                                help="Variable index (default: 6=temperature)")
    extract_parser.add_argument("--num_slices", type=int, default=4,
                                help="Number of slices")
    extract_parser.add_argument("--t_frames", type=int, default=64,
                                help="Number of time frames")
    extract_parser.add_argument("--spatial_size", type=int, default=256,
                                help="Spatial size (H, W)")

    # 随机测试数据命令
    test_parser = subparsers.add_parser("random", help="Create random test data")
    test_parser.add_argument("--output", type=str, required=True,
                             help="Output .npz file path")
    test_parser.add_argument("--shape", type=int, nargs=5, default=[1, 4, 64, 256, 256],
                             help="Data shape (V, S, T, H, W)")

    args = parser.parse_args()

    if args.command == "convert":
        convert_s3d_to_npz(
            input_dir=args.input_dir,
            output_path=args.output,
            variable_indices=args.variables,
            num_slices=args.num_slices,
            z_as_time=args.z_as_time,
        )
    elif args.command == "extract":
        extract_s3d_sample(
            input_dir=args.input_dir,
            output_path=args.output,
            variable_idx=args.variable,
            num_slices=args.num_slices,
            t_frames=args.t_frames,
            spatial_size=args.spatial_size,
        )
    elif args.command == "random":
        create_small_test_data(
            output_path=args.output,
            shape=tuple(args.shape),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
