"""
Multi-modal Encoders for MineSLAM
多模态编码器：RGB/Depth/Thermal/LiDAR/IMU → 统一512维token
支持灵活的预训练/从零训练切换策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings

# 尝试导入EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    warnings.warn("EfficientNet not available, using ResNet backbone")

# 尝试导入MinkowskiEngine
try:
    import MinkowskiEngine as ME
    MINKOWSKI_AVAILABLE = True
except ImportError:
    MINKOWSKI_AVAILABLE = False
    warnings.warn("MinkowskiEngine not available, using stub MLP for LiDAR")


# 训练策略配置
class TrainingStrategy:
    """训练策略配置类"""
    
    @staticmethod
    def get_strategy_from_env() -> Dict[str, bool]:
        """从环境变量获取训练策略"""
        return {
            'rgb_pretrained': not bool(os.getenv('MINESLAM_NO_PRETRAINED', False)),
            'depth_pretrained': bool(os.getenv('MINESLAM_DEPTH_PRETRAINED', False)),
            'thermal_pretrained': bool(os.getenv('MINESLAM_THERMAL_PRETRAINED', False)),
            'force_scratch': bool(os.getenv('MINESLAM_FORCE_SCRATCH', False))
        }
    
    @staticmethod
    def print_strategy(strategy: Dict[str, bool]):
        """打印当前训练策略"""
        print("🎯 Training Strategy:")
        print(f"  RGB: {'Pretrained' if strategy['rgb_pretrained'] else 'From Scratch'}")
        print(f"  Depth: {'Pretrained' if strategy['depth_pretrained'] else 'From Scratch'}")
        print(f"  Thermal: {'Pretrained' if strategy['thermal_pretrained'] else 'From Scratch'}")
        if strategy['force_scratch']:
            print("  ⚠️  FORCE_SCRATCH enabled - all modalities train from scratch")


class ImageEncoder(nn.Module):
    """
    图像编码器：RGB/Depth/Thermal → 统一token
    使用EfficientNet-B0提取多尺度特征，输出(B, T_img, 512)
    支持灵活的预训练/从零训练策略
    """
    
    def __init__(self, in_channels: int = 3, embedding_dim: int = 512, 
                 use_pretrained: Optional[bool] = None, modality: str = 'rgb'):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.modality = modality
        
        # 获取训练策略
        strategy = TrainingStrategy.get_strategy_from_env()
        
        # 决定是否使用预训练权重
        if use_pretrained is None:
            # 自动策略：根据模态和环境变量决定
            if strategy['force_scratch']:
                use_pretrained = False
            elif modality == 'rgb':
                use_pretrained = strategy['rgb_pretrained']
            elif modality == 'depth':
                use_pretrained = strategy['depth_pretrained']
            elif modality == 'thermal':
                use_pretrained = strategy['thermal_pretrained']
            else:
                use_pretrained = False
        
        self.use_pretrained = use_pretrained
        
        # 初始化backbone
        if EFFICIENTNET_AVAILABLE:
            self._init_efficientnet_backbone()
        else:
            self._init_simple_backbone()
        
        # 多尺度特征融合
        if EFFICIENTNET_AVAILABLE and hasattr(self, 'backbone') and hasattr(self.backbone, '_blocks'):
            # EfficientNet-B0的特征维度 (修复: 根据实际block输出维度)
            self.feature_dims = [40, 112, 320]  # 对应idx=4,10,15的输出维度
        else:
            # 简化backbone的特征维度
            self.feature_dims = [64, 128, 256]
            
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(dim, embedding_dim, kernel_size=1, bias=False)
            for dim in self.feature_dims
        ])
        
        # 批归一化
        self.scale_norms = nn.ModuleList([
            nn.BatchNorm2d(embedding_dim) for _ in self.feature_dims
        ])
        
        # 自适应池化到固定大小
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((8, 8)) for _ in self.feature_dims
        ])
        
        # 最终的token数量：3个尺度 × 8×8 = 192个token
        self.num_tokens = len(self.feature_dims) * 8 * 8
        
        # 打印初始化信息
        pretrained_status = "✅ Pretrained" if self.use_pretrained else "🆕 From Scratch"
        print(f"{self.modality.upper()} ImageEncoder: {in_channels}ch → {self.num_tokens} tokens × {embedding_dim}D ({pretrained_status})")
    
    def _init_efficientnet_backbone(self):
        """初始化EfficientNet backbone（支持离线模式）"""
        try:
            if self.use_pretrained and self.in_channels == 3:
                # 使用ImageNet预训练权重（仅RGB）
                print(f"📥 Loading ImageNet pretrained EfficientNet-B0 for {self.modality}")
                self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
                # 移除分类头
                self.backbone._fc = nn.Identity()
            else:
                # 从零开始训练 - 只创建架构，不下载权重
                if self.use_pretrained and self.in_channels != 3:
                    print(f"⚠️  Cannot use ImageNet pretrained weights for {self.in_channels}-channel input, using scratch")
                print(f"🆕 Creating EfficientNet-B0 from scratch for {self.modality}")
                
                # 尝试本地创建架构（避免网络下载）
                try:
                    # 方法1：直接创建架构，不通过from_name
                    from efficientnet_pytorch.model import EfficientNet
                    from efficientnet_pytorch.utils import get_model_params
                    
                    # 获取EfficientNet-B0参数
                    blocks_args, global_params = get_model_params('efficientnet-b0', override_params={'num_classes': 1000})
                    self.backbone = EfficientNet(blocks_args, global_params)
                    
                except Exception as e1:
                    print(f"⚠️  Direct architecture creation failed: {e1}")
                    try:
                        # 方法2：使用from_name（可能触发下载）
                        self.backbone = EfficientNet.from_name('efficientnet-b0')
                    except Exception as e2:
                        print(f"⚠️  from_name also failed: {e2}")
                        raise e2
                
                # 修改第一层以适应不同通道数
                if self.in_channels != 3:
                    self.backbone._conv_stem = nn.Conv2d(
                        self.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
                    )
                # 移除分类头
                self.backbone._fc = nn.Identity()
                
        except Exception as e:
            print(f"⚠️  EfficientNet initialization failed: {e}")
            print(f"🔄 Falling back to simple backbone for {self.modality}")
            self._init_simple_backbone()
    
    def _init_simple_backbone(self):
        """初始化简化backbone（fallback选项）"""
        print(f"🔧 Using simple ResNet-like backbone for {self.modality} ({self.in_channels} channels)")
        self.backbone = self._create_simple_backbone(self.in_channels)
    
    def _create_simple_backbone(self, in_channels: int) -> nn.Module:
        """创建简化的backbone（当EfficientNet不可用时）"""
        return nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Stage 2 
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
    
    def extract_multiscale_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取EfficientNet-B0第2/3/4尺度特征"""
        if EFFICIENTNET_AVAILABLE and hasattr(self.backbone, '_blocks'):
            # EfficientNet路径 - 使用EfficientNet特征提取
            x = self.backbone._conv_stem(x)
            x = self.backbone._bn0(x)
            x = self.backbone._swish(x)
            
            stage_features = []
            
            # 遍历各个MBConv block，提取第2/3/4尺度
            for idx, block in enumerate(self.backbone._blocks):
                x = block(x)
                
                # EfficientNet-B0的精确尺度划分：
                # Stage 1: blocks 0-1 (stem->16)
                # Stage 2: blocks 2-4 (24->40) ← 第2尺度，输出40通道
                # Stage 3: blocks 5-10 (40->112) ← 第3尺度，输出112通道
                # Stage 4: blocks 11-15 (112->320) ← 第4尺度，输出320通道
                if idx == 4:  # 第2尺度结束
                    stage_features.append(x)
                elif idx == 10:  # 第3尺度结束
                    stage_features.append(x)
                elif idx == 15:  # 第4尺度结束
                    stage_features.append(x)
            
            return stage_features
            
        else:
            # 简化backbone路径 - 对应stage 2/3/4
            features = []
            x = self.backbone[:4](x)  # Stage 1
            
            x = self.backbone[4:6](x)  # Stage 2
            features.append(x)
            
            x = self.backbone[6:8](x)  # Stage 3
            features.append(x)
            
            x = self.backbone[8:](x)   # Stage 4
            features.append(x)
            
            return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            tokens: [B, T_img, D] 其中T_img=192, D=512
        """
        batch_size = x.shape[0]
        
        # 提取多尺度特征
        multiscale_features = self.extract_multiscale_features(x)
        
        # 处理每个尺度的特征
        scale_tokens = []
        for i, features in enumerate(multiscale_features):
            # 1x1卷积降维到embedding_dim
            features = self.scale_convs[i](features)
            features = self.scale_norms[i](features)
            features = F.relu(features, inplace=True)
            
            # 自适应池化到8x8
            features = self.adaptive_pools[i](features)  # [B, D, 8, 8]
            
            # 展平为token序列
            tokens = features.flatten(2).transpose(1, 2)  # [B, 64, D]
            scale_tokens.append(tokens)
        
        # 拼接所有尺度的token
        all_tokens = torch.cat(scale_tokens, dim=1)  # [B, 192, D]
        
        return all_tokens


class LidarEncoder(nn.Module):
    """
    LiDAR点云编码器
    必须使用MinkowskiUNet14，不可用时直接报错
    """
    
    def __init__(self, embedding_dim: int = 512, voxel_size: float = 0.05):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.voxel_size = voxel_size
        
        if not MINKOWSKI_AVAILABLE:
            raise RuntimeError("MinkowskiEngine is required for LidarEncoder but not available. "
                             "Please install MinkowskiEngine: pip install MinkowskiEngine")
        
        self._init_minkowski_encoder()
        print(f"LidarEncoder initialized: MinkowskiUNet14 → {embedding_dim}D")
    
    def _init_minkowski_encoder(self):
        """初始化MinkowskiUNet14编码器"""
        try:
            # 检查可用的MinkowskiEngine API
            available_nets = []
            
            if hasattr(ME, 'MinkUNet14'):
                available_nets.append('MinkUNet14')
                self.unet = ME.MinkUNet14(
                    in_channels=1,
                    out_channels=self.embedding_dim,
                    D=3
                )
            elif hasattr(ME, 'MinkowskiUNet14'):
                available_nets.append('MinkowskiUNet14')
                self.unet = ME.MinkowskiUNet14(
                    in_channels=1,
                    out_channels=self.embedding_dim,
                    D=3
                )
            elif hasattr(ME, 'MinkowskiUNet'):
                available_nets.append('MinkowskiUNet')
                # 新版本API - 尝试创建UNet14等效结构
                self.unet = ME.MinkowskiUNet(
                    in_nchannel=1,
                    out_nchannel=self.embedding_dim,
                    D=3
                )
            else:
                # 如果没有预定义的UNet，创建自定义14层结构
                print("⚠️  Creating custom 14-layer sparse UNet")
                self.unet = self._create_custom_unet14()
                available_nets.append('Custom14Layer')
            
            # 全局平均池化
            if hasattr(ME, 'MinkowskiGlobalAvgPooling'):
                self.global_pool = ME.MinkowskiGlobalAvgPooling()
            else:
                # 降级到普通池化
                self.global_pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)
            
            # 输出token数量
            self.num_tokens = 1
            
            print(f"✅ MinkowskiEngine encoder initialized: {available_nets[0]}")
            
        except Exception as e:
            print(f"MinkowskiEngine available APIs: {[attr for attr in dir(ME) if 'UNet' in attr]}")
            raise RuntimeError(f"Failed to initialize MinkowskiUNet14: {e}")
    
    def _create_custom_unet14(self):
        """创建自定义14层UNet结构"""
        # 14层UNet的简化实现
        layers = []
        
        # Encoder部分
        in_channels = 1
        base_channels = 32
        
        for i, out_channels in enumerate([32, 64, 128, 256, 512]):
            layers.extend([
                ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1 if i == 0 else 2, dimension=3),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(out_channels),
                ME.MinkowskiReLU(inplace=True)
            ])
            in_channels = out_channels
        
        # 输出层
        layers.extend([
            ME.MinkowskiConvolution(512, self.embedding_dim, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(self.embedding_dim),
            ME.MinkowskiReLU(inplace=True)
        ])
        
        return nn.Sequential(*layers)
    
    def voxelize_pointcloud(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        体素化点云
        
        Args:
            points: [B, N, 4] 点云数据 [x,y,z,intensity]
            
        Returns:
            coords: [M, 4] 体素坐标 [batch_idx, x, y, z]
            features: [M, 1] 体素特征
        """
        batch_size = points.shape[0]
        
        all_coords = []
        all_features = []
        
        for b in range(batch_size):
            pts = points[b]  # [N, 4]
            
            # 过滤无效点
            valid_mask = torch.all(pts[:, :3] != 0, dim=1)
            if valid_mask.sum() == 0:
                continue
                
            pts = pts[valid_mask]
            
            # 体素化
            coords = pts[:, :3] / self.voxel_size
            coords = coords.round().int()  # 修复：显式转换为int类型

            # 添加batch索引
            batch_coords = torch.full((coords.shape[0], 1), b,
                                    dtype=torch.int32, device=coords.device)  # 修复：明确指定int32类型
            coords = torch.cat([batch_coords, coords], dim=1)  # 现在coords是int类型
            
            # 使用强度作为特征
            features = pts[:, 3:4]  # [N, 1]
            
            all_coords.append(coords)
            all_features.append(features)
        
        if len(all_coords) == 0:
            # 没有有效点，返回空张量
            empty_coords = torch.zeros((0, 4), dtype=torch.long, device=points.device)
            empty_features = torch.zeros((0, 1), dtype=torch.float32, device=points.device)
            return empty_coords, empty_features
        
        coords = torch.cat(all_coords, dim=0)
        features = torch.cat(all_features, dim=0)
        
        return coords, features
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            points: [B, N, 4] 点云数据 [x,y,z,intensity]
            
        Returns:
            tokens: [B, T_lidar, D] 其中T_lidar=1, D=512
        """
        batch_size = points.shape[0]
        
        # 体素化点云
        coords, features = self.voxelize_pointcloud(points)
        
        if coords.shape[0] == 0:
            # 没有有效点，返回零向量
            return torch.zeros(batch_size, self.num_tokens, self.embedding_dim, 
                             device=points.device)
        
        # 创建稀疏张量
        sparse_tensor = ME.SparseTensor(features, coords)
        
        # UNet编码
        encoded = self.unet(sparse_tensor)
        
        # 全局池化
        pooled = self.global_pool(encoded)
        
        # 重组为batch格式
        output_features = pooled.F  # [B, D]
        tokens = output_features.unsqueeze(1)  # [B, 1, D]
        
        return tokens


class IMUEncoder(nn.Module):
    """
    IMU编码器：时序IMU数据 → 单个token
    使用2层LSTM + Linear投影
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, embedding_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 2层LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # 投影到embedding维度
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        
        # 输出token数量
        self.num_tokens = 1
        
        print(f"IMUEncoder initialized: {input_dim}D × T=20 → 1 token × {embedding_dim}D")
    
    def forward(self, imu_sequence: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            imu_sequence: [B, T, 6] IMU时序数据 [ax,ay,az,gx,gy,gz]
            
        Returns:
            tokens: [B, 1, D] 其中D=512
        """
        batch_size = imu_sequence.shape[0]
        
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(imu_sequence)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [B, hidden_dim]
        
        # 投影到embedding维度
        tokens = self.projection(last_output)  # [B, embedding_dim]
        tokens = tokens.unsqueeze(1)  # [B, 1, embedding_dim]
        
        return tokens


class MultiModalEncoder(nn.Module):
    """
    多模态编码器集合
    统一管理所有模态的编码器
    支持灵活的预训练/从零训练策略
    """
    
    def __init__(self, embedding_dim: int = 512, voxel_size: float = 0.05, 
                 training_strategy: Optional[Dict[str, bool]] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 获取训练策略
        if training_strategy is None:
            training_strategy = TrainingStrategy.get_strategy_from_env()
        
        # 打印训练策略
        TrainingStrategy.print_strategy(training_strategy)
        
        print(f"\n🔧 Initializing MultiModalEncoder (embedding_dim={embedding_dim})...")
        
        # 各模态编码器（传入模态名称以便智能决定预训练策略）
        self.rgb_encoder = ImageEncoder(
            in_channels=3, 
            embedding_dim=embedding_dim, 
            modality='rgb'
        )
        
        self.depth_encoder = ImageEncoder(
            in_channels=1, 
            embedding_dim=embedding_dim, 
            modality='depth'
        )
        
        self.thermal_encoder = ImageEncoder(
            in_channels=1, 
            embedding_dim=embedding_dim, 
            modality='thermal'
        )
        
        self.lidar_encoder = LidarEncoder(
            embedding_dim=embedding_dim, 
            voxel_size=voxel_size
        )
        
        self.imu_encoder = IMUEncoder(embedding_dim=embedding_dim)
        
        # 统计token数量
        self.token_counts = {
            'rgb': self.rgb_encoder.num_tokens,
            'depth': self.depth_encoder.num_tokens,
            'thermal': self.thermal_encoder.num_tokens,
            'lidar': self.lidar_encoder.num_tokens,
            'imu': self.imu_encoder.num_tokens,
        }
        
        print(f"\n📊 MultiModalEncoder Summary:")
        print(f"  Token counts: {self.token_counts}")
        print(f"  Total tokens: {sum(self.token_counts.values())}")
        print(f"  Memory efficient: {self._estimate_memory_usage():.1f} MB")
    
    def _estimate_memory_usage(self) -> float:
        """估算模型内存使用量"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 ** 2)  # FP32
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播，支持路线B掩码处理

        Args:
            data_dict: 包含各模态数据和掩码的字典
                - 'rgb': [B, 3, H, W]
                - 'depth': [B, 1, H, W]
                - 'thermal': [B, 1, H, W]
                - 'lidar': [B, P, 4]  # P=16384 固定长度
                - 'imu': [B, T, 6]
                - 'present_mask': {'rgb': [B], 'thermal': [B], 'lidar': [B], ...}

        Returns:
            token_dict: 包含各模态token的字典
                - 'rgb': [B, T_rgb, D]
                - 'depth': [B, T_depth, D]
                - 'thermal': [B, T_thermal, D]
                - 'lidar': [B, T_lidar, D]
                - 'imu': [B, T_imu, D]
        """
        token_dict = {}
        present_mask = data_dict.get('present_mask', {})

        if 'rgb' in data_dict:
            mask = present_mask.get('rgb', None)
            token_dict['rgb'] = self._forward_with_mask(
                self.rgb_encoder, data_dict['rgb'], mask
            )

        if 'depth' in data_dict:
            mask = present_mask.get('depth', None)
            token_dict['depth'] = self._forward_with_mask(
                self.depth_encoder, data_dict['depth'], mask
            )

        if 'thermal' in data_dict:
            mask = present_mask.get('thermal', None)
            token_dict['thermal'] = self._forward_with_mask(
                self.thermal_encoder, data_dict['thermal'], mask
            )

        if 'lidar' in data_dict:
            mask = present_mask.get('lidar', None)
            token_dict['lidar'] = self._forward_lidar_with_mask(
                data_dict['lidar'], mask
            )

        if 'imu' in data_dict:
            mask = present_mask.get('imu', None)
            token_dict['imu'] = self._forward_with_mask(
                self.imu_encoder, data_dict['imu'], mask
            )

        return token_dict

    def _forward_with_mask(self, encoder: nn.Module, x: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        带掩码的前向传播（通用图像和IMU编码器）

        Args:
            encoder: 编码器模块
            x: 输入数据 [B, ...]
            mask: 掩码 [B]，True表示真实数据，False表示填充

        Returns:
            tokens: [B, T, D] 编码后的token，填充样本为零向量
        """
        batch_size = x.shape[0]

        if mask is None:
            # 没有掩码，正常处理
            return encoder(x)

        # 正常编码
        tokens = encoder(x)  # [B, T, D]

        # 应用掩码：填充样本置零
        mask_expanded = mask.view(batch_size, 1, 1)  # [B, 1, 1]
        tokens = tokens * mask_expanded.float()  # 广播乘法，填充样本变为零向量

        return tokens

    def _forward_lidar_with_mask(self, points: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        LiDAR编码器的掩码前向传播

        Args:
            points: [B, P, 4] 点云数据（可能包含填充的零点云）
            mask: [B] 布尔掩码，True表示真实数据，False表示填充

        Returns:
            tokens: [B, 1, D] 编码后的token
        """
        batch_size = points.shape[0]

        if mask is None:
            # 没有掩码，正常处理
            return self.lidar_encoder(points)

        # 处理每个样本
        output_tokens = []
        for i in range(batch_size):
            if mask[i]:
                # 真实数据：正常处理
                sample_points = points[i:i+1]  # [1, P, 4]
                coords, features = self.lidar_encoder.voxelize_pointcloud(sample_points)

                if coords.shape[0] > 0:
                    sparse_tensor = ME.SparseTensor(features, coords)
                    encoded = self.lidar_encoder.unet(sparse_tensor)
                    pooled = self.lidar_encoder.global_pool(encoded)
                    token = pooled.F.unsqueeze(1)  # [1, 1, D]
                else:
                    token = torch.zeros(1, 1, self.embedding_dim, device=points.device)
            else:
                # 填充数据：输出零向量
                token = torch.zeros(1, 1, self.embedding_dim, device=points.device)

            output_tokens.append(token)

        return torch.cat(output_tokens, dim=0)  # [B, 1, D]
    
    def get_total_tokens(self, available_modalities: List[str]) -> int:
        """获取给定模态的总token数量"""
        return sum(self.token_counts[mod] for mod in available_modalities 
                  if mod in self.token_counts)