#!/usr/bin/env python3
"""
EfficientNet离线下载和设置工具
提供多种下载方案和环境配置
"""

import os
import sys
import urllib.request
from pathlib import Path
import hashlib


def setup_offline_environment():
    """设置离线环境变量"""
    print("🔧 设置离线环境...")
    
    # 设置torch缓存目录
    cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['TORCH_HOME'] = str(cache_dir.parent)
    os.environ['HF_HUB_OFFLINE'] = '1'  # Hugging Face离线模式
    
    print(f"📁 Torch缓存目录: {cache_dir}")
    return cache_dir


def download_with_mirrors(cache_dir: Path):
    """使用镜像源下载EfficientNet权重"""
    target_file = cache_dir / "efficientnet-b0-355c32eb.pth"
    
    if target_file.exists():
        print(f"✅ 权重文件已存在: {target_file}")
        return True
    
    # 多个下载源
    download_urls = [
        # GitHub镜像源（国内访问友好）
        "https://ghproxy.com/https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
        
        # 其他镜像源
        "https://mirror.ghproxy.com/https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
        
        # 原始GitHub（备用）
        "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
    ]
    
    for i, url in enumerate(download_urls, 1):
        try:
            print(f"🔄 尝试下载源 {i}/{len(download_urls)}: {url}")
            
            # 下载文件
            urllib.request.urlretrieve(url, target_file)
            
            # 验证文件大小
            file_size = target_file.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 至少10MB
                print(f"✅ 下载成功: {target_file} ({file_size / 1024 / 1024:.1f}MB)")
                return True
            else:
                print(f"❌ 文件大小异常: {file_size}字节")
                target_file.unlink()  # 删除异常文件
                
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            if target_file.exists():
                target_file.unlink()
            continue
    
    return False


def create_efficientnet_offline():
    """创建离线EfficientNet架构（不依赖权重文件）"""
    print("🏗️  创建离线EfficientNet架构...")
    
    # 写入离线创建脚本
    offline_script = '''
import torch
import torch.nn as nn
from efficientnet_pytorch.model import EfficientNet
from efficientnet_pytorch.utils import get_model_params

def create_efficientnet_b0_offline():
    """离线创建EfficientNet-B0架构"""
    try:
        # 获取模型参数（不依赖网络）
        blocks_args, global_params = get_model_params('efficientnet-b0', 
                                                     override_params={'num_classes': 1000})
        
        # 直接创建模型
        model = EfficientNet(blocks_args, global_params)
        print("✅ 离线EfficientNet-B0创建成功")
        return model
        
    except Exception as e:
        print(f"❌ 离线创建失败: {e}")
        return None

if __name__ == "__main__":
    model = create_efficientnet_b0_offline()
    if model:
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
'''
    
    script_path = Path(__file__).parent / "efficientnet_offline.py"
    with open(script_path, 'w') as f:
        f.write(offline_script)
    
    print(f"📄 离线脚本已保存: {script_path}")
    return script_path


def main():
    """主函数"""
    print("🚀 EfficientNet离线下载工具")
    print("=" * 50)
    
    # 设置环境
    cache_dir = setup_offline_environment()
    
    # 尝试下载权重
    print("\n📥 下载EfficientNet权重...")
    download_success = download_with_mirrors(cache_dir)
    
    # 创建离线架构脚本
    print("\n🏗️  准备离线架构...")
    offline_script = create_efficientnet_offline()
    
    # 验证测试
    print("\n🧪 验证测试...")
    try:
        from efficientnet_pytorch import EfficientNet
        from efficientnet_pytorch.utils import get_model_params
        
        # 测试架构创建
        blocks_args, global_params = get_model_params('efficientnet-b0')
        model = EfficientNet(blocks_args, global_params)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"✅ EfficientNet-B0架构创建成功")
        print(f"   参数量: {param_count:.1f}M")
        print(f"   权重文件: {'存在' if download_success else '不存在（将从零训练）'}")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    print("\n" + "=" * 50)
    print("✅ 离线设置完成！")
    print("\n使用方法：")
    print("1. 运行此脚本完成离线设置")
    print("2. 运行 python tests/test_moe.py")
    print("3. 如果权重下载成功，RGB将使用预训练；否则从零训练")


if __name__ == "__main__":
    main()