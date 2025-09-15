#!/usr/bin/env python3
"""
Training Strategy Demo for MineSLAM Encoders
演示不同训练策略的切换机制
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.encoders import MultiModalEncoder, TrainingStrategy


def demo_strategy(strategy_name: str, env_vars: dict):
    """演示特定训练策略"""
    print("=" * 80)
    print(f"DEMO: {strategy_name}")
    print("=" * 80)
    
    # 设置环境变量
    for key, value in env_vars.items():
        os.environ[key] = str(value)
    
    print(f"Environment Variables:")
    for key, value in env_vars.items():
        print(f"  {key} = {value}")
    
    try:
        # 创建编码器（会自动读取环境变量）
        encoder = MultiModalEncoder(embedding_dim=512, voxel_size=0.05)
        print("✅ Encoder created successfully!")
        
        # 清理环境变量
        for key in env_vars.keys():
            if key in os.environ:
                del os.environ[key]
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """主演示函数"""
    print("MineSLAM Training Strategy Demo")
    print("Demonstrating flexible pretrained/scratch training switches")
    
    # 策略1: 默认策略（RGB预训练，其他从零开始）
    demo_strategy(
        "Default Strategy (RGB Pretrained, Others Scratch)",
        {}
    )
    
    # 策略2: 全部从零开始
    demo_strategy(
        "All From Scratch",
        {'MINESLAM_FORCE_SCRATCH': '1'}
    )
    
    # 策略3: 禁用预训练
    demo_strategy(
        "No Pretrained Weights",
        {'MINESLAM_NO_PRETRAINED': '1'}
    )
    
    # 策略4: 深度和热成像也使用预训练
    demo_strategy(
        "Multi-Modal Pretrained",
        {
            'MINESLAM_DEPTH_PRETRAINED': '1',
            'MINESLAM_THERMAL_PRETRAINED': '1'
        }
    )
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print()
    print("1. 默认策略（RGB预训练）:")
    print("   python encoders_sanity.py")
    print()
    print("2. 全部从零开始:")
    print("   MINESLAM_FORCE_SCRATCH=1 python encoders_sanity.py")
    print()
    print("3. 禁用所有预训练:")
    print("   MINESLAM_NO_PRETRAINED=1 python encoders_sanity.py")
    print()
    print("4. 深度和热成像也用预训练:")
    print("   MINESLAM_DEPTH_PRETRAINED=1 MINESLAM_THERMAL_PRETRAINED=1 python encoders_sanity.py")
    print()
    print("5. 混合策略:")
    print("   MINESLAM_THERMAL_PRETRAINED=1 python encoders_sanity.py")
    print()
    print("Environment Variables:")
    print("  MINESLAM_NO_PRETRAINED     - 禁用RGB预训练")
    print("  MINESLAM_DEPTH_PRETRAINED  - 启用深度预训练")
    print("  MINESLAM_THERMAL_PRETRAINED - 启用热成像预训练")
    print("  MINESLAM_FORCE_SCRATCH     - 强制所有模态从零开始")
    print()
    print("🎯 建议的开发流程:")
    print("  Week 1-2: 使用预训练快速验证架构")
    print("  Week 3-4: 收集矿井数据，微调预训练模型")
    print("  Week 5+:  使用FORCE_SCRATCH在真实数据上从零训练")


if __name__ == '__main__':
    main()