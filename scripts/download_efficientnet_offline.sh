#!/bin/bash
# EfficientNet离线下载脚本
# 使用方法: bash scripts/download_efficientnet_offline.sh

set -e

echo "🔧 EfficientNet离线下载工具"
echo "=================================="

# 创建缓存目录
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"

echo "📁 缓存目录: $CACHE_DIR"

# 方法1：使用国内镜像源下载
echo ""
echo "🌏 方法1：使用国内镜像源（推荐）"
echo "--------------------------------"

# Hugging Face镜像源
HF_MIRROR="https://hf-mirror.com"
EFFICIENTNET_URL="${HF_MIRROR}/timm/efficientnet_b0.ra_in1k/resolve/main/pytorch_model.bin"

# 或者使用GitHub Release镜像
GITHUB_MIRROR="https://ghproxy.com/https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"

echo "🔄 尝试从Hugging Face镜像下载..."
if command -v wget >/dev/null 2>&1; then
    wget -O "$CACHE_DIR/efficientnet-b0-355c32eb.pth" "$GITHUB_MIRROR" || echo "wget下载失败"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$CACHE_DIR/efficientnet-b0-355c32eb.pth" "$GITHUB_MIRROR" || echo "curl下载失败"
else
    echo "❌ 需要安装wget或curl"
fi

# 方法2：使用Python直接下载
echo ""
echo "🐍 方法2：使用Python下载"
echo "------------------------"

python3 << 'EOF'
import os
import urllib.request
from pathlib import Path

cache_dir = Path.home() / ".cache/torch/hub/checkpoints"
cache_dir.mkdir(parents=True, exist_ok=True)

urls = [
    # GitHub镜像源
    ("https://ghproxy.com/https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth", 
     "efficientnet-b0-355c32eb.pth"),
    # 直接GitHub（如果网络允许）
    ("https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
     "efficientnet-b0-355c32eb.pth")
]

for url, filename in urls:
    try:
        target_path = cache_dir / filename
        if not target_path.exists():
            print(f"⬇️  下载中: {url}")
            urllib.request.urlretrieve(url, target_path)
            print(f"✅ 下载完成: {target_path}")
            break
        else:
            print(f"✅ 文件已存在: {target_path}")
            break
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        continue
EOF

# 方法3：手动下载指引
echo ""
echo "👇 方法3：手动下载（如果自动下载失败）"
echo "------------------------------------"
echo "1. 访问以下任一地址下载权重文件："
echo "   - GitHub原址: https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
echo "   - GitHub镜像: https://ghproxy.com/https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
echo ""
echo "2. 将下载的文件重命名为: efficientnet-b0-355c32eb.pth"
echo "3. 移动到目录: $CACHE_DIR/"
echo ""

# 验证下载
echo "🔍 验证下载结果"
echo "----------------"
if [ -f "$CACHE_DIR/efficientnet-b0-355c32eb.pth" ]; then
    FILE_SIZE=$(ls -lh "$CACHE_DIR/efficientnet-b0-355c32eb.pth" | awk '{print $5}')
    echo "✅ EfficientNet-B0权重文件存在"
    echo "   文件大小: $FILE_SIZE"
    echo "   路径: $CACHE_DIR/efficientnet-b0-355c32eb.pth"
else
    echo "❌ 权重文件不存在，请手动下载"
fi

echo ""
echo "🎉 离线下载脚本执行完成！"
echo "现在可以运行MoE测试而不会触发网络下载"