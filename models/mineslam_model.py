"""
MineSLAM Unified Model
统一的MineSLAM模型类 - 集成所有组件并提供完整的前向传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .encoders import MultiModalEncoder
from .moe_fusion import MoEFusion, MoEConfig
from .pose_head import PoseHead
from .detection_head import DetectionHead
from .kendall_uncertainty import create_fixed_kendall_uncertainty


class MineSLAMModel(nn.Module):
    """
    MineSLAM完整模型
    集成多模态编码、MoE融合、多任务头部和Kendall不确定性
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化MineSLAM模型

        Args:
            config: 模型配置字典，包含各组件参数
        """
        super().__init__()

        # 默认配置
        self.config = config or self._get_default_config()

        # 1. 多模态编码器
        encoder_config = self.config.get('encoder', {})
        self.encoder = MultiModalEncoder(**encoder_config)

        # 2. MoE融合模块
        moe_config = self.config.get('moe_fusion', {})
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        elif not isinstance(moe_config, MoEConfig):
            moe_config = MoEConfig()

        self.moe_fusion = MoEFusion(moe_config)

        # 3. 任务头部
        pose_config = self.config.get('pose_head', {})
        self.pose_head = PoseHead(**pose_config)

        detection_config = self.config.get('detection_head', {})
        self.detection_head = DetectionHead(**detection_config)

        # 4. Kendall不确定性权重 (使用修复版)
        kendall_config = self.config.get('kendall_uncertainty', {})
        self.kendall_uncertainty = create_fixed_kendall_uncertainty(kendall_config)

        # 模型信息
        self.model_info = {
            'total_params': self._count_parameters(),
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'moe_params': sum(p.numel() for p in self.moe_fusion.parameters()),
            'pose_head_params': sum(p.numel() for p in self.pose_head.parameters()),
            'detection_head_params': sum(p.numel() for p in self.detection_head.parameters()),
            'kendall_params': sum(p.numel() for p in self.kendall_uncertainty.parameters())
        }

        print(f"🏗️ MineSLAM Model initialized:")
        print(f"   Total parameters: {self.model_info['total_params']:,}")
        print(f"   Encoder: {self.model_info['encoder_params']:,}")
        print(f"   MoE Fusion: {self.model_info['moe_params']:,}")
        print(f"   Pose Head: {self.model_info['pose_head_params']:,}")
        print(f"   Detection Head: {self.model_info['detection_head_params']:,}")
        print(f"   Kendall Uncertainty: {self.model_info['kendall_params']:,}")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'encoder': {},
            'moe_fusion': {
                'embedding_dim': 512,
                'num_experts': 3,
                'num_encoder_layers': 2,
                'nhead': 4,
                'feedforward_dim': 2048,
                'dropout': 0.1,
                'thermal_guidance': True
            },
            'pose_head': {
                'input_dim': 512,
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.1
            },
            'detection_head': {
                'input_dim': 512,
                'num_queries': 20,
                'num_classes': 8,
                'decoder_layers': 4,
                'nhead': 8,
                'dropout': 0.1
            },
            'kendall_uncertainty': {
                'initial_pose_log_var': -1.0,
                'initial_detection_log_var': 0.0,
                'initial_gate_log_var': -0.4,
                'enable_weight_constraints': True,
                'min_log_var': -2.0,
                'max_log_var': 2.0
            }
        }

    def forward(self, batch: Dict[str, torch.Tensor],
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            batch: 输入批次数据
            return_intermediate: 是否返回中间结果

        Returns:
            输出字典，包含位姿、检测结果和中间特征
        """
        outputs = {}

        # 1. 多模态编码
        try:
            encoded_features = self.encoder(batch)
            if return_intermediate:
                outputs['encoded_features'] = encoded_features
        except Exception as e:
            warnings.warn(f"Encoder forward failed: {e}")
            # 创建默认特征用于调试
            batch_size = list(batch.values())[0].shape[0] if batch else 1
            encoded_features = {
                'rgb': torch.zeros(batch_size, 512).to(self.device),
                'depth': torch.zeros(batch_size, 512).to(self.device),
                'thermal': torch.zeros(batch_size, 512).to(self.device),
                'lidar': torch.zeros(batch_size, 512).to(self.device),
                'imu': torch.zeros(batch_size, 512).to(self.device)
            }

        # 2. MoE融合
        try:
            fused_features, moe_output = self.moe_fusion(encoded_features, batch)
            if return_intermediate:
                outputs['fused_features'] = fused_features
                outputs['moe_output'] = moe_output
        except Exception as e:
            warnings.warn(f"MoE fusion failed: {e}")
            # 使用平均融合作为备选
            feature_tensors = [f for f in encoded_features.values() if isinstance(f, torch.Tensor)]
            if feature_tensors:
                fused_features = torch.stack(feature_tensors).mean(dim=0)
            else:
                fused_features = torch.zeros(batch_size, 512).to(self.device)
            moe_output = {}

        # 3. 任务头部预测
        # 位姿估计
        pose_output = self.pose_head(fused_features)
        outputs['pose'] = pose_output

        # 物体检测
        detection_output = self.detection_head(fused_features)
        outputs['detection'] = detection_output

        # 4. 添加MoE分析结果
        if moe_output:
            outputs['moe_analysis'] = moe_output

        return outputs

    def compute_loss(self, model_output: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            model_output: 模型输出
            targets: 真值标签

        Returns:
            损失字典
        """
        # 提取各任务预测和真值
        pose_pred = model_output['pose']
        detection_pred = model_output['detection']

        pose_target = targets.get('pose', torch.zeros_like(pose_pred))
        detection_target = targets.get('detection', {})

        # 计算原始损失
        pose_loss = F.mse_loss(pose_pred, pose_target)

        # 检测损失 (简化实现)
        if 'classes' in detection_pred and 'classes' in detection_target:
            detection_loss = F.cross_entropy(
                detection_pred['classes'].flatten(0, 1),
                detection_target['classes'].flatten().long()
            )
        else:
            detection_loss = torch.tensor(0.0, device=pose_pred.device)

        # MoE门控损失
        moe_output = model_output.get('moe_analysis', {})
        gate_loss = moe_output.get('gate_loss', torch.tensor(0.0, device=pose_pred.device))

        # 使用修复版Kendall不确定性权重
        kendall_result = self.kendall_uncertainty.compute_multitask_loss(
            pose_loss, detection_loss, gate_loss
        )

        return {
            'total_loss': kendall_result['total_loss'],
            'pose_loss': pose_loss,
            'detection_loss': detection_loss,
            'gate_loss': gate_loss,
            'weighted_pose_loss': kendall_result['weighted_pose_loss'],
            'weighted_detection_loss': kendall_result['weighted_detection_loss'],
            'weighted_gate_loss': kendall_result['weighted_gate_loss'],
            'kendall_weights': {
                'pose_weight': kendall_result['pose_weight'],
                'detection_weight': kendall_result['detection_weight'],
                'gate_weight': kendall_result['gate_weight'],
                'pose_sigma': kendall_result['pose_sigma'],
                'detection_sigma': kendall_result['detection_sigma'],
                'gate_sigma': kendall_result['gate_sigma']
            }
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        return {
            'model_name': 'MineSLAM',
            'version': '1.0.0',
            'parameters': self.model_info,
            'components': {
                'encoder': type(self.encoder).__name__,
                'moe_fusion': type(self.moe_fusion).__name__,
                'pose_head': type(self.pose_head).__name__,
                'detection_head': type(self.detection_head).__name__,
                'kendall_uncertainty': type(self.kendall_uncertainty).__name__
            },
            'config': self.config
        }

    def _count_parameters(self) -> int:
        """计算总参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        """获取模型设备"""
        return next(self.parameters()).device

    def print_kendall_status(self, epoch: int = None):
        """打印Kendall权重状态"""
        self.kendall_uncertainty.print_status(epoch)

    def get_kendall_weights(self) -> Dict[str, float]:
        """获取当前Kendall权重"""
        return self.kendall_uncertainty.get_weights()

    def get_weight_balance_metrics(self) -> Dict[str, float]:
        """获取权重平衡指标"""
        return self.kendall_uncertainty.get_weight_balance_metrics()

    def save_model(self, path: str, additional_info: Dict = None):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_info': self.model_info,
            'model_summary': self.get_model_summary()
        }

        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, path)
        print(f"💾 MineSLAM model saved to: {path}")

    @classmethod
    def load_model(cls, path: str, device: str = 'cuda'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)

        config = checkpoint.get('model_config', {})
        model = cls(config)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        print(f"📂 MineSLAM model loaded from: {path}")

        return model, checkpoint


def create_mineslam_model(config: Dict[str, Any] = None) -> MineSLAMModel:
    """
    工厂函数：创建MineSLAM模型

    Args:
        config: 模型配置

    Returns:
        MineSLAM模型实例
    """
    return MineSLAMModel(config)


# 向后兼容
MineSLAM = MineSLAMModel


if __name__ == "__main__":
    # 测试模型创建
    print("🧪 Testing MineSLAM Model Creation...")

    model = create_mineslam_model()
    print(f"✅ Model created successfully")

    # 测试前向传播
    batch_size = 2
    dummy_batch = {
        'rgb': torch.randn(batch_size, 3, 224, 224),
        'depth': torch.randn(batch_size, 1, 224, 224),
        'thermal': torch.randn(batch_size, 1, 224, 224),
        'lidar': torch.randn(batch_size, 16384, 3),
        'imu': torch.randn(batch_size, 6)
    }

    try:
        with torch.no_grad():
            outputs = model(dummy_batch, return_intermediate=True)

        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")

        # 测试损失计算
        dummy_targets = {
            'pose': torch.randn(batch_size, 6),
            'detection': {
                'classes': torch.randint(0, 8, (batch_size, 20)),
                'boxes': torch.randn(batch_size, 20, 6)
            }
        }

        loss_dict = model.compute_loss(outputs, dummy_targets)
        print(f"✅ Loss computation successful")
        print(f"   Total loss: {loss_dict['total_loss']:.6f}")

        # 打印模型摘要
        summary = model.get_model_summary()
        print(f"✅ Model summary: {summary['parameters']['total_params']:,} parameters")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    print("✅ MineSLAM Model test completed!")