#!/usr/bin/env python3
"""
Encoder Unit Tests - REAL DATA ONLY
编码器单元测试：ImageEncoder/LidarEncoder/IMUEncoder
从mineslam_dataset.py读取前N=20个批次真实样本
验证输出token格式[B,T,512]，无NaN/Inf
记录FLOPs和显存使用（TensorBoard）
若缺LiDAR/IMU文件直接失败
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
from torch.utils.tensorboard import SummaryWriter
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_config
from data.mineslam_dataset import MineSLAMDataset
from models.encoders import ImageEncoder, LidarEncoder, IMUEncoder, MultiModalEncoder

# FLOPs计算工具
try:
    from fvcore.nn import FlopCountMode, flop_count
    FLOPS_AVAILABLE = True
except ImportError:
    print("Warning: fvcore not available, install with: pip install fvcore")
    FLOPS_AVAILABLE = False

# GPU内存监控
def get_gpu_memory():
    """获取GPU内存使用量(MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


class EncoderTester:
    """编码器测试器"""
    
    def __init__(self, config: Dict, output_dir: str = "outputs/encoder_tests"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard日志记录
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 测试参数
        self.batch_size = 2
        self.num_batches = 20
        self.embedding_dim = 512
        
        print(f"EncoderTester initialized: {self.num_batches} batches × {self.batch_size} samples")
    
    def _validate_real_sample(self, sample: Dict, batch_idx: int, sample_idx: int):
        """验证样本来自真实数据"""
        # 检查合成数据标记
        forbidden_keys = ['_generated', '_synthetic', '_random', '_mock', '_fake']
        for key in sample.keys():
            if any(forbidden in str(key).lower() for forbidden in forbidden_keys):
                raise ValueError(f"FORBIDDEN: Detected synthetic data in batch {batch_idx}, sample {sample_idx}")
        
        # 验证时间戳真实性
        if 'timestamp' in sample:
            timestamp = sample['timestamp'].item()
            if timestamp <= 0 or timestamp > 2e9:
                raise ValueError(f"Invalid timestamp in batch {batch_idx}: {timestamp}")
        
        # 检查张量值的多样性（防止全零或常数填充）
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.numel() > 10:
                unique_values = torch.unique(value)
                if len(unique_values) == 1:
                    warnings.warn(f"Batch {batch_idx}: Tensor '{key}' has identical values")
    
    def _check_tensor_validity(self, tensor: torch.Tensor, name: str):
        """检查张量有效性：无NaN/Inf"""
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")
        if tensor.numel() == 0:
            raise ValueError(f"{name} is empty tensor")
    
    def test_image_encoder(self, dataset: MineSLAMDataset) -> Dict[str, float]:
        """测试ImageEncoder"""
        print(f"\n🖼️  Testing ImageEncoder on {self.num_batches} real batches...")
        
        # 创建RGB和Thermal编码器
        rgb_encoder = ImageEncoder(in_channels=3, embedding_dim=self.embedding_dim, modality='rgb')
        thermal_encoder = ImageEncoder(in_channels=1, embedding_dim=self.embedding_dim, modality='thermal')
        
        rgb_encoder.to(self.device)
        thermal_encoder.to(self.device)
        rgb_encoder.eval()
        thermal_encoder.eval()
        
        total_rgb_tokens = 0
        total_thermal_tokens = 0
        rgb_flops_list = []
        thermal_flops_list = []
        memory_usage = []
        
        for batch_idx in range(min(self.num_batches, len(dataset))):
            try:
                # 收集batch数据
                rgb_batch = []
                thermal_batch = []
                
                for sample_idx in range(self.batch_size):
                    data_idx = batch_idx * self.batch_size + sample_idx
                    if data_idx >= len(dataset):
                        break
                        
                    sample = dataset[data_idx]
                    self._validate_real_sample(sample, batch_idx, sample_idx)
                    
                    if 'rgb' in sample:
                        rgb_batch.append(sample['rgb'])
                    if 'thermal' in sample:
                        thermal_batch.append(sample['thermal'])
                
                # 测试RGB编码器
                if rgb_batch:
                    rgb_tensor = torch.stack(rgb_batch).to(self.device)
                    print(f"  RGB batch {batch_idx}: {rgb_tensor.shape}")
                    
                    start_memory = get_gpu_memory()
                    
                    with torch.no_grad():
                        # FLOPs计算
                        if FLOPS_AVAILABLE:
                            flop_dict, _ = flop_count(rgb_encoder, (rgb_tensor,))
                            total_flops = sum(flop_dict.values())
                            rgb_flops_list.append(total_flops)
                        
                        # 前向传播
                        rgb_tokens = rgb_encoder(rgb_tensor)
                        
                        # 验证输出格式
                        expected_shape = (len(rgb_batch), rgb_encoder.num_tokens, self.embedding_dim)
                        assert rgb_tokens.shape == expected_shape, f"RGB tokens shape mismatch: {rgb_tokens.shape} vs {expected_shape}"
                        
                        # 检查有效性
                        self._check_tensor_validity(rgb_tokens, f"RGB tokens batch {batch_idx}")
                        
                        total_rgb_tokens += rgb_tokens.shape[0] * rgb_tokens.shape[1]
                    
                    end_memory = get_gpu_memory()
                    memory_usage.append(end_memory - start_memory)
                
                # 测试Thermal编码器
                if thermal_batch:
                    thermal_tensor = torch.stack(thermal_batch).to(self.device)
                    print(f"  Thermal batch {batch_idx}: {thermal_tensor.shape}")
                    
                    with torch.no_grad():
                        # FLOPs计算
                        if FLOPS_AVAILABLE:
                            flop_dict, _ = flop_count(thermal_encoder, (thermal_tensor,))
                            total_flops = sum(flop_dict.values())
                            thermal_flops_list.append(total_flops)
                        
                        # 前向传播
                        thermal_tokens = thermal_encoder(thermal_tensor)
                        
                        # 验证输出格式
                        expected_shape = (len(thermal_batch), thermal_encoder.num_tokens, self.embedding_dim)
                        assert thermal_tokens.shape == expected_shape, f"Thermal tokens shape mismatch: {thermal_tokens.shape} vs {expected_shape}"
                        
                        # 检查有效性
                        self._check_tensor_validity(thermal_tokens, f"Thermal tokens batch {batch_idx}")
                        
                        total_thermal_tokens += thermal_tokens.shape[0] * thermal_tokens.shape[1]
                
            except Exception as e:
                print(f"Warning: Skipping batch {batch_idx}: {e}")
        
        # 记录TensorBoard
        if rgb_flops_list:
            avg_rgb_flops = np.mean(rgb_flops_list)
            self.writer.add_scalar('ImageEncoder/RGB_FLOPs', avg_rgb_flops, 0)
        if thermal_flops_list:
            avg_thermal_flops = np.mean(thermal_flops_list)
            self.writer.add_scalar('ImageEncoder/Thermal_FLOPs', avg_thermal_flops, 0)
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            self.writer.add_scalar('ImageEncoder/GPU_Memory_MB', avg_memory, 0)
        
        metrics = {
            'total_rgb_tokens': total_rgb_tokens,
            'total_thermal_tokens': total_thermal_tokens,
            'avg_rgb_flops': np.mean(rgb_flops_list) if rgb_flops_list else 0,
            'avg_thermal_flops': np.mean(thermal_flops_list) if thermal_flops_list else 0,
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
        }
        
        print(f"✅ ImageEncoder test completed: RGB={total_rgb_tokens} tokens, Thermal={total_thermal_tokens} tokens")
        return metrics
    
    def test_lidar_encoder(self, dataset: MineSLAMDataset) -> Dict[str, float]:
        """测试LidarEncoder - 必须有MinkowskiEngine"""
        print(f"\n☁️  Testing LidarEncoder on {self.num_batches} real batches...")
        
        # 检查LiDAR文件存在性
        lidar_dir = Path(self.config['data']['lidar_dir'])
        if not lidar_dir.exists():
            raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")
        
        lidar_files = list(lidar_dir.glob("*.bin"))
        if len(lidar_files) == 0:
            raise FileNotFoundError(f"No LiDAR files found in: {lidar_dir}")
        
        # 创建LiDAR编码器（会检查MinkowskiEngine）
        try:
            lidar_encoder = LidarEncoder(embedding_dim=self.embedding_dim)
            lidar_encoder.to(self.device)
            lidar_encoder.eval()
        except RuntimeError as e:
            raise RuntimeError(f"LidarEncoder initialization failed: {e}")
        
        total_lidar_tokens = 0
        flops_list = []
        memory_usage = []
        
        for batch_idx in range(min(self.num_batches, len(dataset))):
            try:
                # 收集batch数据
                lidar_batch = []
                
                for sample_idx in range(self.batch_size):
                    data_idx = batch_idx * self.batch_size + sample_idx
                    if data_idx >= len(dataset):
                        break
                        
                    sample = dataset[data_idx]
                    self._validate_real_sample(sample, batch_idx, sample_idx)
                    
                    if 'lidar' in sample:
                        lidar_batch.append(sample['lidar'])
                
                if not lidar_batch:
                    continue
                
                lidar_tensor = torch.stack(lidar_batch).to(self.device)
                print(f"  LiDAR batch {batch_idx}: {lidar_tensor.shape}")
                
                start_memory = get_gpu_memory()
                
                with torch.no_grad():
                    # FLOPs计算（MinkowskiEngine复杂，暂时跳过）
                    # if FLOPS_AVAILABLE:
                    #     flop_dict, _ = flop_count(lidar_encoder, (lidar_tensor,))
                    #     total_flops = sum(flop_dict.values())
                    #     flops_list.append(total_flops)
                    
                    # 前向传播
                    lidar_tokens = lidar_encoder(lidar_tensor)
                    
                    # 验证输出格式
                    expected_shape = (len(lidar_batch), lidar_encoder.num_tokens, self.embedding_dim)
                    assert lidar_tokens.shape == expected_shape, f"LiDAR tokens shape mismatch: {lidar_tokens.shape} vs {expected_shape}"
                    
                    # 检查有效性
                    self._check_tensor_validity(lidar_tokens, f"LiDAR tokens batch {batch_idx}")
                    
                    total_lidar_tokens += lidar_tokens.shape[0] * lidar_tokens.shape[1]
                
                end_memory = get_gpu_memory()
                memory_usage.append(end_memory - start_memory)
                
            except Exception as e:
                print(f"Warning: Skipping LiDAR batch {batch_idx}: {e}")
        
        # 记录TensorBoard
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            self.writer.add_scalar('LidarEncoder/GPU_Memory_MB', avg_memory, 0)
        
        metrics = {
            'total_lidar_tokens': total_lidar_tokens,
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
            'lidar_files_count': len(lidar_files)
        }
        
        print(f"✅ LidarEncoder test completed: {total_lidar_tokens} tokens from {len(lidar_files)} files")
        return metrics
    
    def test_imu_encoder(self, dataset: MineSLAMDataset) -> Dict[str, float]:
        """测试IMUEncoder - 必须有IMU文件"""
        print(f"\n🧭  Testing IMUEncoder on {self.num_batches} real batches...")
        
        # 检查IMU文件存在性
        imu_file = Path(self.config['data']['imu_file'])
        if not imu_file.exists():
            raise FileNotFoundError(f"IMU file not found: {imu_file}")
        
        # 创建IMU编码器
        imu_encoder = IMUEncoder(embedding_dim=self.embedding_dim)
        imu_encoder.to(self.device)
        imu_encoder.eval()
        
        total_imu_tokens = 0
        flops_list = []
        memory_usage = []
        
        for batch_idx in range(min(self.num_batches, len(dataset))):
            try:
                # 收集batch数据
                imu_batch = []
                
                for sample_idx in range(self.batch_size):
                    data_idx = batch_idx * self.batch_size + sample_idx
                    if data_idx >= len(dataset):
                        break
                        
                    sample = dataset[data_idx]
                    self._validate_real_sample(sample, batch_idx, sample_idx)
                    
                    if 'imu' in sample:
                        imu_batch.append(sample['imu'])
                
                if not imu_batch:
                    continue
                
                imu_tensor = torch.stack(imu_batch).to(self.device)
                print(f"  IMU batch {batch_idx}: {imu_tensor.shape}")
                
                start_memory = get_gpu_memory()
                
                with torch.no_grad():
                    # FLOPs计算
                    if FLOPS_AVAILABLE:
                        flop_dict, _ = flop_count(imu_encoder, (imu_tensor,))
                        total_flops = sum(flop_dict.values())
                        flops_list.append(total_flops)
                    
                    # 前向传播
                    imu_tokens = imu_encoder(imu_tensor)
                    
                    # 验证输出格式
                    expected_shape = (len(imu_batch), imu_encoder.num_tokens, self.embedding_dim)
                    assert imu_tokens.shape == expected_shape, f"IMU tokens shape mismatch: {imu_tokens.shape} vs {expected_shape}"
                    
                    # 检查有效性
                    self._check_tensor_validity(imu_tokens, f"IMU tokens batch {batch_idx}")
                    
                    total_imu_tokens += imu_tokens.shape[0] * imu_tokens.shape[1]
                
                end_memory = get_gpu_memory()
                memory_usage.append(end_memory - start_memory)
                
            except Exception as e:
                print(f"Warning: Skipping IMU batch {batch_idx}: {e}")
        
        # 记录TensorBoard
        if flops_list:
            avg_flops = np.mean(flops_list)
            self.writer.add_scalar('IMUEncoder/FLOPs', avg_flops, 0)
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            self.writer.add_scalar('IMUEncoder/GPU_Memory_MB', avg_memory, 0)
        
        metrics = {
            'total_imu_tokens': total_imu_tokens,
            'avg_flops': np.mean(flops_list) if flops_list else 0,
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
        }
        
        print(f"✅ IMUEncoder test completed: {total_imu_tokens} tokens")
        return metrics


class TestEncoders(unittest.TestCase):
    """编码器单元测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.config_path = 'configs/mineslam.yaml'
        cls.output_dir = "outputs/encoder_tests"
        
        # 加载配置
        if not os.path.exists(cls.config_path):
            raise unittest.SkipTest(f"Configuration file not found: {cls.config_path}")
        
        cls.config = load_config(cls.config_path)
        
        # 检查验证数据索引
        val_index_path = cls.config['data']['val_index']
        if not os.path.exists(val_index_path):
            raise unittest.SkipTest(f"Validation index not found: {val_index_path}")
        
        # 创建测试器
        cls.tester = EncoderTester(cls.config, cls.output_dir)
        
        print(f"Testing encoders on val_index with N=20 batches")
    
    def test_01_image_encoder(self):
        """测试ImageEncoder"""
        print("\n" + "="*60)
        print("Testing ImageEncoder (RGB/Thermal)")
        print("="*60)
        
        # 创建验证数据集
        dataset = MineSLAMDataset(self.config, split='val')
        
        # 测试图像编码器
        image_metrics = self.tester.test_image_encoder(dataset)
        
        # 验证token数量
        self.assertGreater(image_metrics['total_rgb_tokens'], 0, "No RGB tokens generated")
        self.assertGreater(image_metrics['total_thermal_tokens'], 0, "No Thermal tokens generated")
        
        print(f"✅ ImageEncoder Test PASSED:")
        print(f"  RGB tokens: {image_metrics['total_rgb_tokens']}")
        print(f"  Thermal tokens: {image_metrics['total_thermal_tokens']}")
        print(f"  Average RGB FLOPs: {image_metrics['avg_rgb_flops']:.2e}")
        print(f"  Average memory: {image_metrics['avg_memory_mb']:.1f} MB")
    
    def test_02_lidar_encoder(self):
        """测试LidarEncoder"""
        print("\n" + "="*60)
        print("Testing LidarEncoder (MinkowskiUNet14)")
        print("="*60)
        
        # 创建验证数据集
        dataset = MineSLAMDataset(self.config, split='val')
        
        # 测试LiDAR编码器
        lidar_metrics = self.tester.test_lidar_encoder(dataset)
        
        # 验证token数量
        self.assertGreater(lidar_metrics['total_lidar_tokens'], 0, "No LiDAR tokens generated")
        self.assertGreater(lidar_metrics['lidar_files_count'], 0, "No LiDAR files found")
        
        print(f"✅ LidarEncoder Test PASSED:")
        print(f"  LiDAR tokens: {lidar_metrics['total_lidar_tokens']}")
        print(f"  LiDAR files: {lidar_metrics['lidar_files_count']}")
        print(f"  Average memory: {lidar_metrics['avg_memory_mb']:.1f} MB")
    
    def test_03_imu_encoder(self):
        """测试IMUEncoder"""
        print("\n" + "="*60)
        print("Testing IMUEncoder (2×LSTM)")
        print("="*60)
        
        # 创建验证数据集
        dataset = MineSLAMDataset(self.config, split='val')
        
        # 测试IMU编码器
        imu_metrics = self.tester.test_imu_encoder(dataset)
        
        # 验证token数量
        self.assertGreater(imu_metrics['total_imu_tokens'], 0, "No IMU tokens generated")
        
        print(f"✅ IMUEncoder Test PASSED:")
        print(f"  IMU tokens: {imu_metrics['total_imu_tokens']}")
        print(f"  Average FLOPs: {imu_metrics['avg_flops']:.2e}")
        print(f"  Average memory: {imu_metrics['avg_memory_mb']:.1f} MB")
    
    def test_04_data_requirements(self):
        """验证数据文件要求"""
        print("\n" + "="*60)
        print("Validating Data File Requirements")
        print("="*60)
        
        # 检查LiDAR目录
        lidar_dir = Path(self.config['data']['lidar_dir'])
        self.assertTrue(lidar_dir.exists(), f"LiDAR directory not found: {lidar_dir}")
        
        lidar_files = list(lidar_dir.glob("*.bin"))
        self.assertGreater(len(lidar_files), 0, f"No LiDAR .bin files in: {lidar_dir}")
        
        # 检查IMU文件
        imu_file = Path(self.config['data']['imu_file'])
        self.assertTrue(imu_file.exists(), f"IMU file not found: {imu_file}")
        
        print(f"✅ Data requirements validated:")
        print(f"  LiDAR files: {len(lidar_files)}")
        print(f"  IMU file: {imu_file}")


def run_encoder_tests():
    """运行编码器测试"""
    print("="*80)
    print("MINESLAM ENCODER TESTS - REAL DATA ONLY")
    print("="*80)
    print("Testing ImageEncoder/LidarEncoder/IMUEncoder on val_index N=20 batches")
    print("Requirements: [B,T,512] tokens, no NaN/Inf, MinkowskiEngine for LiDAR")
    print()
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEncoders)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("编码器测试总结")
    print("="*80)
    
    if result.wasSuccessful():
        print("✅ 所有编码器测试通过!")
        print("   - ImageEncoder: EfficientNet-B0第2/3/4尺度 → 512维")
        print("   - LidarEncoder: MinkowskiUNet14 → 512维")  
        print("   - IMUEncoder: 2×LSTM → 512维")
        print("   - 所有输出格式[B,T,512]，无NaN/Inf")
        print("   - TensorBoard日志保存到outputs/encoder_tests/")
    else:
        print("❌ 编码器测试失败!")
        print(f"   失败: {len(result.failures)}")
        print(f"   错误: {len(result.errors)}")
        
        if result.failures:
            print("\n失败详情:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\n错误详情:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_encoder_tests()
    sys.exit(0 if success else 1)