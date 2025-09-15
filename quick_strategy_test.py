#!/usr/bin/env python3
"""
Quick Test Scripts for Different Training Strategies
不同训练策略的快速测试脚本
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.encoders import MultiModalEncoder


def quick_test(strategy_name: str, env_vars: dict):
    """快速测试特定策略"""
    print(f"\n🧪 Quick Test: {strategy_name}")
    print("-" * 50)
    
    # 设置环境变量
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key, None)
        os.environ[key] = str(value)
    
    try:
        # 创建编码器
        encoder = MultiModalEncoder(embedding_dim=256, voxel_size=0.1)  # 更小配置便于测试
        
        # 创建测试数据
        batch_size = 1
        test_data = {
            'rgb': torch.randn(batch_size, 3, 240, 320),      # 更小分辨率
            'depth': torch.randn(batch_size, 1, 240, 320),
            'thermal': torch.randn(batch_size, 1, 240, 320),
            'lidar': torch.randn(batch_size, 1000, 4),        # 更少点
            'imu': torch.randn(batch_size, 10, 6),            # 更短序列
        }
        
        # 测试前向传播
        with torch.no_grad():
            tokens = encoder(test_data)
        
        print(f"✅ Success! Token shapes:")
        for modality, token in tokens.items():
            print(f"  {modality}: {list(token.shape)}")
        
        total_tokens = sum(token.shape[1] for token in tokens.values())
        print(f"  Total: {total_tokens} tokens")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    finally:
        # 恢复环境变量
        for key, original_value in original_env.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value


def main():
    """主测试函数"""
    print("🚀 MineSLAM Training Strategy Quick Tests")
    
    # 测试各种策略
    strategies = [
        ("Default (RGB Pretrained)", {}),
        ("All Scratch", {'MINESLAM_FORCE_SCRATCH': '1'}),
        ("No Pretrained", {'MINESLAM_NO_PRETRAINED': '1'}),
        ("Multi Pretrained", {
            'MINESLAM_DEPTH_PRETRAINED': '1', 
            'MINESLAM_THERMAL_PRETRAINED': '1'
        }),
    ]
    
    for name, env_vars in strategies:
        quick_test(name, env_vars)
    
    print("\n🎉 All tests completed!")
    print("\n📋 Command Reference:")
    print("# 默认策略")
    print("python encoders_sanity.py")
    print("\n# 全部从零开始")
    print("MINESLAM_FORCE_SCRATCH=1 python encoders_sanity.py")
    print("\n# 禁用预训练")  
    print("MINESLAM_NO_PRETRAINED=1 python encoders_sanity.py")
    print("\n# 查看策略演示")
    print("python training_strategy_demo.py")


if __name__ == '__main__':
    main()