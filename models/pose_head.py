"""
PoseHead: Multi-task Pose Estimation Head
多任务姿态估计头：mean-pool → MLP → SE(3) 增量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PoseHead(nn.Module):
    """
    姿态估计头
    输入：多模态融合token [B, T, D]
    输出：SE(3) 6DOF 增量 [B, 6] (translation + rotation)
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dims: list = [512, 256, 128],
                 dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 全局平均池化（mean-pool所有token）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层：6DOF pose (3 translation + 3 rotation)
        layers.append(nn.Linear(prev_dim, 6))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化最后一层权重为小值
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
        
        print(f"PoseHead initialized: {input_dim} → {hidden_dims} → 6DOF")
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens: [B, T, D] 融合后的token序列
            
        Returns:
            pose_delta: [B, 6] SE(3) 增量 [tx, ty, tz, rx, ry, rz]
        """
        batch_size, seq_len, embed_dim = tokens.shape
        
        # 全局平均池化: [B, T, D] → [B, D]
        # 转置到 [B, D, T] 进行池化
        pooled = self.global_pool(tokens.transpose(1, 2))  # [B, D, 1]
        pooled = pooled.squeeze(-1)  # [B, D]
        
        # MLP预测6DOF增量
        pose_delta = self.mlp(pooled)  # [B, 6]
        
        return pose_delta
    
    def compute_loss(self, 
                     pred_pose: torch.Tensor, 
                     target_pose: torch.Tensor,
                     delta: float = 1.0) -> torch.Tensor:
        """
        计算Huber损失
        
        Args:
            pred_pose: [B, 6] 预测的pose增量
            target_pose: [B, 6] 目标pose增量  
            delta: Huber损失的delta参数
            
        Returns:
            loss: scalar Huber损失
        """
        return F.huber_loss(pred_pose, target_pose, delta=delta)
    
    def decompose_pose(self, pose_6dof: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分解6DOF pose为平移和旋转
        
        Args:
            pose_6dof: [B, 6] 6DOF pose
            
        Returns:
            translation: [B, 3] 平移
            rotation: [B, 3] 旋转 (轴角表示)
        """
        translation = pose_6dof[:, :3]  # [B, 3]
        rotation = pose_6dof[:, 3:]     # [B, 3]
        return translation, rotation


class SE3Utils:
    """SE(3)工具函数"""
    
    @staticmethod
    def se3_to_matrix(se3_vec: torch.Tensor) -> torch.Tensor:
        """
        将SE(3) 6维向量转换为4x4变换矩阵
        
        Args:
            se3_vec: [B, 6] SE(3)向量 [tx, ty, tz, rx, ry, rz]
            
        Returns:
            transform_matrix: [B, 4, 4] 变换矩阵
        """
        batch_size = se3_vec.shape[0]
        device = se3_vec.device
        
        # 提取平移和旋转
        translation = se3_vec[:, :3]  # [B, 3]
        rotation_vec = se3_vec[:, 3:]  # [B, 3]
        
        # 计算旋转角度
        angle = torch.norm(rotation_vec, dim=1, keepdim=True)  # [B, 1]
        
        # 避免除零
        safe_angle = torch.where(angle < 1e-6, 
                                torch.ones_like(angle) * 1e-6, 
                                angle)
        
        # 归一化旋转轴
        axis = rotation_vec / safe_angle  # [B, 3]
        
        # 使用Rodrigues公式构建旋转矩阵
        cos_angle = torch.cos(angle).unsqueeze(-1)  # [B, 1, 1]
        sin_angle = torch.sin(angle).unsqueeze(-1)  # [B, 1, 1]
        
        # 单位矩阵
        I = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]
        
        # 反对称矩阵
        K = torch.zeros(batch_size, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
        R = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)
        
        # 构建4x4变换矩阵
        transform_matrix = torch.zeros(batch_size, 4, 4, device=device)
        transform_matrix[:, :3, :3] = R
        transform_matrix[:, :3, 3] = translation
        transform_matrix[:, 3, 3] = 1.0
        
        return transform_matrix
    
    @staticmethod
    def matrix_to_se3(transform_matrix: torch.Tensor) -> torch.Tensor:
        """
        将4x4变换矩阵转换为SE(3) 6维向量
        
        Args:
            transform_matrix: [B, 4, 4] 变换矩阵
            
        Returns:
            se3_vec: [B, 6] SE(3)向量
        """
        batch_size = transform_matrix.shape[0]
        device = transform_matrix.device
        
        # 提取平移
        translation = transform_matrix[:, :3, 3]  # [B, 3]
        
        # 提取旋转矩阵
        R = transform_matrix[:, :3, :3]  # [B, 3, 3]
        
        # 从旋转矩阵计算轴角表示
        trace = R.diagonal(dim1=1, dim2=2).sum(dim=1)  # [B]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))  # [B]
        
        # 计算旋转轴
        axis = torch.zeros(batch_size, 3, device=device)
        axis[:, 0] = R[:, 2, 1] - R[:, 1, 2]
        axis[:, 1] = R[:, 0, 2] - R[:, 2, 0]
        axis[:, 2] = R[:, 1, 0] - R[:, 0, 1]
        
        # 归一化并乘以角度
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        safe_norm = torch.where(axis_norm < 1e-6, 
                               torch.ones_like(axis_norm), 
                               axis_norm)
        rotation_vec = axis / safe_norm * angle.unsqueeze(-1)
        
        # 拼接SE(3)向量
        se3_vec = torch.cat([translation, rotation_vec], dim=1)  # [B, 6]
        
        return se3_vec


# 模块测试
if __name__ == '__main__':
    print("🧪 Testing PoseHead...")
    
    # 创建PoseHead
    pose_head = PoseHead(input_dim=512, hidden_dims=[256, 128])
    
    # 测试数据
    batch_size = 2
    seq_len = 100
    embed_dim = 512
    
    tokens = torch.randn(batch_size, seq_len, embed_dim)
    target_pose = torch.randn(batch_size, 6) * 0.1  # 小的pose增量
    
    print(f"Input tokens shape: {tokens.shape}")
    
    # 前向传播
    with torch.no_grad():
        pred_pose = pose_head(tokens)
        print(f"Predicted pose shape: {pred_pose.shape}")
        print(f"Predicted pose sample: {pred_pose[0]}")
    
    # 计算损失
    pred_pose_grad = pose_head(tokens)
    loss = pose_head.compute_loss(pred_pose_grad, target_pose)
    print(f"Huber loss: {loss.item():.6f}")
    
    # 测试梯度
    loss.backward()
    grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in pose_head.parameters() if p.grad is not None]))
    print(f"Gradient norm: {grad_norm.item():.6f}")
    
    # 测试SE(3)工具
    print("\n🔧 Testing SE(3) utilities...")
    se3_vec = torch.randn(2, 6) * 0.1
    print(f"Original SE(3): {se3_vec}")
    
    transform_matrix = SE3Utils.se3_to_matrix(se3_vec)
    print(f"Transform matrix shape: {transform_matrix.shape}")
    
    recovered_se3 = SE3Utils.matrix_to_se3(transform_matrix)
    print(f"Recovered SE(3): {recovered_se3}")
    
    error = torch.norm(se3_vec - recovered_se3, dim=1)
    print(f"Reconstruction error: {error}")
    
    print("\n✅ PoseHead test completed!")