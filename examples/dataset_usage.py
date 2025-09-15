#!/usr/bin/env python3
"""
MineSLAM Dataset Example Usage
演示如何正确使用MineSLAM数据集加载真实传感器数据
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.mineslam_dataset import MineSLAMDataset
from utils import load_config
import torch


def demonstrate_dataset_usage():
    """演示数据集使用方法"""
    print("="*60)
    print("MineSLAM Dataset Usage Example")
    print("="*60)
    
    # 1. 加载配置
    config = load_config('configs/mineslam.yaml')
    
    # 2. 创建数据集
    print("正在创建训练数据集...")
    try:
        train_dataset = MineSLAMDataset(config, split='train')
        print(f"✓ 训练数据集创建成功: {len(train_dataset)} 个样本")
    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        return
    
    # 3. 获取数据集统计信息
    stats = train_dataset.get_statistics()
    print(f"\n数据集统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  缺帧数量: {stats['missing_frame_count']}")
    print(f"  缺帧比例: {stats['missing_frame_rate']:.2%}")
    print(f"  IMU窗口大小: {stats['imu_window_size']}")
    print(f"  时间阈值: {stats['time_threshold_ms']}ms")
    
    # 4. 加载示例样本
    print(f"\n正在加载前3个样本...")
    for i in range(min(3, len(train_dataset))):
        try:
            sample = train_dataset[i]
            
            print(f"\n样本 {i}:")
            print(f"  时间戳: {sample['timestamp'].item():.6f}")
            print(f"  可用模态: {list(sample.keys())}")
            
            # 显示各模态数据形状
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                    
                    # 显示数据范围
                    if value.numel() > 0:
                        print(f"    范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
            
        except Exception as e:
            print(f"  ✗ 样本 {i} 加载失败: {e}")
    
    # 5. 创建数据加载器
    print(f"\n创建数据加载器...")
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 设置为0避免多进程问题
            collate_fn=custom_collate_fn
        )
        
        print(f"✓ 数据加载器创建成功")
        
        # 测试批次加载
        batch = next(iter(dataloader))
        print(f"  批次大小: {len(batch)}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  批次 {key}: {value.shape}")
        
    except Exception as e:
        print(f"✗ 数据加载器创建失败: {e}")


def custom_collate_fn(batch):
    """自定义批次整理函数"""
    if not batch:
        return {}
    
    # 获取所有键
    keys = set()
    for sample in batch:
        keys.update(sample.keys())
    
    batched = {}
    
    for key in keys:
        # 收集当前键的所有值
        values = []
        for sample in batch:
            if key in sample:
                values.append(sample[key])
        
        if not values:
            continue
        
        # 尝试堆叠张量
        if isinstance(values[0], torch.Tensor):
            try:
                # 对于形状不一致的张量（如激光雷达点云），使用列表
                if key == 'lidar':
                    batched[key] = values  # 保持列表形式
                else:
                    batched[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # 如果无法堆叠，保持列表形式
                batched[key] = values
        else:
            batched[key] = values
    
    return batched


if __name__ == '__main__':
    demonstrate_dataset_usage()