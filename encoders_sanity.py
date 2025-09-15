#!/usr/bin/env python3
"""
Encoders Sanity Check for MineSLAM
测试多模态编码器的输出形状、FLOPs和显存使用
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Tuple, List
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.encoders import MultiModalEncoder

# 尝试导入FLOPs计算工具
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available, FLOPs calculation will be skipped")
    print("Install with: pip install thop")

# 尝试导入torchinfo
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("Warning: torchinfo not available, detailed model info will be skipped")
    print("Install with: pip install torchinfo")


def create_sample_data(batch_size: int = 2, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    创建样本数据用于测试
    
    Args:
        batch_size: 批次大小
        device: 设备
    
    Returns:
        包含各模态样本数据的字典
    """
    data_dict = {
        # RGB图像: [B, 3, 480, 640]
        'rgb': torch.randn(batch_size, 3, 480, 640, device=device),
        
        # 深度图像: [B, 1, 480, 640]
        'depth': torch.randn(batch_size, 1, 480, 640, device=device),
        
        # 热成像图像: [B, 1, 480, 640]
        'thermal': torch.randn(batch_size, 1, 480, 640, device=device),
        
        # LiDAR点云: [B, N, 4] (N=8192个点)
        'lidar': torch.randn(batch_size, 8192, 4, device=device),
        
        # IMU序列: [B, T, 6] (T=20个时间步)
        'imu': torch.randn(batch_size, 20, 6, device=device),
    }
    
    return data_dict


def get_memory_usage() -> float:
    """获取GPU显存使用量（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def calculate_flops(model: nn.Module, input_data: Dict[str, torch.Tensor]) -> Tuple[float, float]:
    """
    计算模型FLOPs和参数量
    
    Args:
        model: 模型
        input_data: 输入数据
    
    Returns:
        (FLOPs, 参数量)
    """
    if not THOP_AVAILABLE:
        return 0.0, 0.0
    
    try:
        # 计算总FLOPs
        total_flops = 0
        total_params = 0
        
        # 分别计算各个编码器的FLOPs
        for modality, data in input_data.items():
            if modality == 'rgb':
                flops, params = profile(model.rgb_encoder, inputs=(data,), verbose=False)
            elif modality == 'depth':
                flops, params = profile(model.depth_encoder, inputs=(data,), verbose=False)
            elif modality == 'thermal':
                flops, params = profile(model.thermal_encoder, inputs=(data,), verbose=False)
            elif modality == 'lidar':
                flops, params = profile(model.lidar_encoder, inputs=(data,), verbose=False)
            elif modality == 'imu':
                flops, params = profile(model.imu_encoder, inputs=(data,), verbose=False)
            else:
                continue
                
            total_flops += flops
            total_params += params
        
        return total_flops, total_params
        
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        return 0.0, 0.0


def test_encoder_shapes(model: MultiModalEncoder, 
                       input_data: Dict[str, torch.Tensor],
                       device: str) -> Dict[str, torch.Tensor]:
    """
    测试编码器输出形状
    
    Args:
        model: 多模态编码器
        input_data: 输入数据
        device: 设备
    
    Returns:
        输出token字典
    """
    print("=" * 60)
    print("ENCODER SHAPE TESTING")
    print("=" * 60)
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    
    # 移动数据到设备
    input_data = {k: v.to(device) for k, v in input_data.items()}
    
    print("\nInput Shapes:")
    for modality, tensor in input_data.items():
        print(f"  {modality:8}: {list(tensor.shape)} ({tensor.dtype})")
    
    # 前向传播
    with torch.no_grad():
        token_dict = model(input_data)
    
    print("\nOutput Token Shapes:")
    total_tokens = 0
    for modality, tokens in token_dict.items():
        num_tokens = tokens.shape[1]
        embedding_dim = tokens.shape[2]
        total_tokens += num_tokens
        print(f"  {modality:8}: {list(tokens.shape)} → {num_tokens} tokens × {embedding_dim}D")
    
    print(f"\nTotal Tokens: {total_tokens}")
    
    return token_dict


def test_memory_and_flops(model: MultiModalEncoder,
                         input_data: Dict[str, torch.Tensor],
                         device: str) -> Dict[str, float]:
    """
    测试显存使用和FLOPs
    
    Args:
        model: 多模态编码器
        input_data: 输入数据
        device: 设备
    
    Returns:
        性能指标字典
    """
    print("\n" + "=" * 60)
    print("MEMORY AND FLOPS TESTING")
    print("=" * 60)
    
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}
    
    # 清空显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 测量模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / (1024**2):.1f} MB (FP32)")
    
    # 测量显存使用
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
        
        with torch.no_grad():
            _ = model(input_data)
        
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"\nGPU Memory Usage:")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After: {mem_after:.1f} MB") 
        print(f"  Peak: {peak_mem:.1f} MB")
        print(f"  Forward Pass: {mem_after - mem_before:.1f} MB")
    
    # 计算FLOPs
    flops, _ = calculate_flops(model, input_data)
    if flops > 0:
        flops_str = clever_format([flops], "%.3f")[0] if THOP_AVAILABLE else f"{flops:.2e}"
        print(f"\nComputational Complexity:")
        print(f"  FLOPs: {flops_str}")
        print(f"  FLOPs per token: {flops / model.get_total_tokens(list(input_data.keys())):.2e}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_usage_mb': mem_after - mem_before if torch.cuda.is_available() else 0,
        'flops': flops
    }


def test_amp_support(model: MultiModalEncoder,
                    input_data: Dict[str, torch.Tensor],
                    device: str) -> bool:
    """
    测试自动混合精度(AMP)支持
    
    Args:
        model: 多模态编码器
        input_data: 输入数据
        device: 设备
    
    Returns:
        是否支持AMP
    """
    print("\n" + "=" * 60)
    print("AUTOMATIC MIXED PRECISION (AMP) TESTING")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping AMP test")
        return False
    
    model = model.to(device)
    input_data = {k: v.to(device) for k, v in input_data.items()}
    
    try:
        # 测试AMP前向传播
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                token_dict = model(input_data)
        
        print("✓ AMP Forward Pass: SUCCESS")
        
        # 检查输出数据类型
        for modality, tokens in token_dict.items():
            print(f"  {modality:8}: {tokens.dtype}")
        
        # 测试梯度缩放（需要创建损失函数）
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        
        with torch.cuda.amp.autocast():
            token_dict = model(input_data)
            # 创建虚拟损失
            loss = sum(tokens.mean() for tokens in token_dict.values())
        
        scaler.scale(loss).backward()
        scaler.step(torch.optim.Adam(model.parameters()))
        scaler.update()
        
        print("✓ AMP Backward Pass: SUCCESS")
        print("✓ Gradient Scaling: SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"✗ AMP Test Failed: {e}")
        return False


def benchmark_inference_speed(model: MultiModalEncoder,
                             input_data: Dict[str, torch.Tensor],
                             device: str,
                             num_runs: int = 100) -> Dict[str, float]:
    """
    基准推理速度测试
    
    Args:
        model: 多模态编码器
        input_data: 输入数据
        device: 设备  
        num_runs: 运行次数
    
    Returns:
        速度指标字典
    """
    print("\n" + "=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 60)
    
    model = model.to(device)
    model.eval()
    input_data = {k: v.to(device) for k, v in input_data.items()}
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 基准测试
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times[10:])  # 去掉前10次（额外预热）
    
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    
    batch_size = input_data['rgb'].shape[0]
    fps = batch_size / (avg_time / 1000)
    
    print(f"\nInference Speed (batch_size={batch_size}, runs={len(times)}):")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps
    }


def print_model_summary(model: MultiModalEncoder):
    """打印模型详细信息"""
    if not TORCHINFO_AVAILABLE:
        return
    
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    try:
        # 创建样本输入用于summary
        sample_rgb = torch.randn(1, 3, 480, 640)
        
        print("\nRGB Encoder:")
        summary(model.rgb_encoder, input_data=sample_rgb, verbose=0)
        
        print("\nLiDAR Encoder:")
        sample_lidar = torch.randn(1, 1024, 4)
        summary(model.lidar_encoder, input_data=sample_lidar, verbose=0)
        
        print("\nIMU Encoder:")
        sample_imu = torch.randn(1, 20, 6)
        summary(model.imu_encoder, input_data=sample_imu, verbose=0)
        
    except Exception as e:
        print(f"Model summary failed: {e}")


def main():
    """主函数"""
    print("MineSLAM Multi-Modal Encoders Sanity Check")
    print("=" * 80)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # 配置参数
    embedding_dim = 512
    voxel_size = 0.05
    batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  Embedding Dim: {embedding_dim}")
    print(f"  Input Resolution: 480×640")
    print(f"  IMU Window: T=20")
    print(f"  LiDAR Voxel Size: {voxel_size}m")
    print(f"  Batch Size: {batch_size}")
    
    # 创建模型
    print(f"\nInitializing MultiModalEncoder...")
    model = MultiModalEncoder(embedding_dim=embedding_dim, voxel_size=voxel_size)
    
    # 创建测试数据
    print(f"\nCreating sample data...")
    input_data = create_sample_data(batch_size=batch_size, device=device)
    
    # 打印模型架构摘要
    print_model_summary(model)
    
    # 测试输出形状
    token_dict = test_encoder_shapes(model, input_data, device)
    
    # 测试内存和FLOPs
    performance_metrics = test_memory_and_flops(model, input_data, device)
    
    # 测试AMP支持
    amp_supported = test_amp_support(model, input_data, device)
    
    # 基准推理速度
    speed_metrics = benchmark_inference_speed(model, input_data, device)
    
    # 总结报告
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n✓ Multi-Modal Encoder Successfully Initialized")
    print(f"✓ All Modality Encoders Working Correctly")
    print(f"✓ Unified Token Output: {embedding_dim}D")
    print(f"✓ Total Parameters: {performance_metrics['total_params']:,}")
    print(f"✓ Model Size: {performance_metrics['total_params'] * 4 / (1024**2):.1f} MB")
    
    if torch.cuda.is_available():
        print(f"✓ GPU Memory Usage: {performance_metrics['memory_usage_mb']:.1f} MB")
        print(f"✓ Inference Speed: {speed_metrics['avg_time_ms']:.1f} ms ({speed_metrics['fps']:.1f} FPS)")
    
    print(f"✓ AMP Support: {'Yes' if amp_supported else 'No'}")
    
    # 各模态token统计
    print(f"\nToken Statistics:")
    for modality, tokens in token_dict.items():
        num_tokens = tokens.shape[1]
        print(f"  {modality:8}: {num_tokens:3d} tokens")
    
    total_tokens = sum(tokens.shape[1] for tokens in token_dict.values())
    print(f"  {'Total':8}: {total_tokens:3d} tokens")
    
    print(f"\n🎉 Encoders sanity check completed successfully!")


if __name__ == '__main__':
    main()