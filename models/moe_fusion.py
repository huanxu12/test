"""
MoEFusion: Mixture of Experts Fusion Module for MineSLAM
多专家融合模块：Geometric/Semantic/Visual专家 + 热引导机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class MoEConfig:
    """MoE配置类"""
    embedding_dim: int = 512
    num_experts: int = 3
    num_encoder_layers: int = 2
    nhead: int = 4
    feedforward_dim: int = 2048
    dropout: float = 0.1
    gate_hidden_dim: int = 256
    thermal_guidance: bool = True
    gate_entropy_weight: float = 0.01


class ThermalGuidedAttention(nn.Module):
    """热引导注意力机制"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.nhead = config.nhead
        
        # 标准多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 热引导处理
        self.thermal_processor = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def generate_thermal_mask(self, thermal_tokens: torch.Tensor, 
                            target_length: int) -> torch.Tensor:
        """
        从热成像token生成空间mask
        
        Args:
            thermal_tokens: [B, T_thermal, D] 热成像token
            target_length: 目标序列长度
            
        Returns:
            mask: [B, target_length] 空间mask
        """
        batch_size = thermal_tokens.shape[0]
        
        # 生成热重要性分数
        thermal_importance = self.thermal_processor(thermal_tokens)  # [B, T_thermal, 1]
        thermal_importance = thermal_importance.squeeze(-1)  # [B, T_thermal]
        
        # 上采样到目标长度
        if thermal_tokens.shape[1] != target_length:
            # 使用线性插值上采样
            thermal_importance = thermal_importance.unsqueeze(1)  # [B, 1, T_thermal]
            mask = F.interpolate(
                thermal_importance, 
                size=target_length, 
                mode='linear', 
                align_corners=False
            )
            mask = mask.squeeze(1)  # [B, target_length]
        else:
            mask = thermal_importance
        
        return mask
    
    def apply_thermal_guidance(self, attention_weights: torch.Tensor,
                             thermal_mask: torch.Tensor,
                             guidance_type: str = 'additive') -> torch.Tensor:
        """
        应用热引导到注意力权重
        
        Args:
            attention_weights: [B, T, T] 注意力权重 (averaged across heads)
            thermal_mask: [B, T] 热引导mask
            guidance_type: 'additive' 或 'multiplicative'
            
        Returns:
            guided_weights: [B, T, T] 引导后的注意力权重
        """
        if attention_weights.dim() == 4:
            # [B, H, T, T] format - average across heads
            attention_weights = attention_weights.mean(dim=1)
        
        batch_size, seq_len, _ = attention_weights.shape
        
        # 生成mask bias [B, T, T]
        mask_bias = thermal_mask.unsqueeze(-1).expand(-1, -1, seq_len)  # [B, T, T]
        
        if guidance_type == 'additive':
            # 加性bias
            guided_weights = attention_weights + mask_bias * 0.1  # 缩放因子
        elif guidance_type == 'multiplicative':
            # 乘性bias
            guided_weights = attention_weights * (1.0 + mask_bias)
        else:
            raise ValueError(f"Unknown guidance type: {guidance_type}")
        
        # 重新归一化
        guided_weights = F.softmax(guided_weights, dim=-1)
        
        return guided_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                thermal_tokens: Optional[torch.Tensor] = None,
                guidance_type: str = 'additive') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query, key, value: [B, T, D] 注意力输入
            thermal_tokens: [B, T_thermal, D] 热成像token（可选）
            guidance_type: 热引导类型
            
        Returns:
            output: [B, T, D] 输出
            attention_weights: [B, T, T] 注意力权重
        """
        # 标准注意力
        output, attention_weights = self.attention(query, key, value)
        
        # 如果有热成像引导
        if thermal_tokens is not None and self.config.thermal_guidance:
            # 生成热引导mask
            thermal_mask = self.generate_thermal_mask(thermal_tokens, query.shape[1])
            
            # 应用热引导
            guided_weights = self.apply_thermal_guidance(
                attention_weights, thermal_mask, guidance_type
            )
            
            # 重新计算输出使用引导后的权重
            # 注意：PyTorch的attention_weights是[B, T, T]格式，不是[B, H, T, T]
            guided_output = torch.matmul(guided_weights, value)  # [B, T, D]
            
            return guided_output, guided_weights
        
        return output, attention_weights


class ExpertLayer(nn.Module):
    """专家层：Transformer Encoder Layer with Thermal Guidance"""
    
    def __init__(self, config: MoEConfig, expert_type: str = 'generic'):
        super().__init__()
        self.config = config
        self.expert_type = expert_type
        self.embedding_dim = config.embedding_dim
        
        # 热引导注意力（仅Semantic专家使用）
        if expert_type == 'semantic':
            self.self_attention = ThermalGuidedAttention(config)
        else:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=config.embedding_dim,
                num_heads=config.nhead,
                dropout=config.dropout,
                batch_first=True
            )
        
        # 前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        print(f"Expert Layer initialized: {expert_type.upper()}")
    
    def forward(self, x: torch.Tensor, 
                thermal_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, T, D] 输入token序列
            thermal_tokens: [B, T_thermal, D] 热成像token（用于Semantic专家）
            
        Returns:
            output: [B, T, D] 输出
            attention_weights: [B, H, T, T] 注意力权重（用于可视化）
        """
        # Self-attention
        if self.expert_type == 'semantic' and thermal_tokens is not None:
            # Semantic专家使用热引导注意力
            attn_output, attention_weights = self.self_attention(
                x, x, x, thermal_tokens=thermal_tokens
            )
        else:
            # 其他专家使用标准注意力
            attn_output, attention_weights = self.self_attention(x, x, x)
        
        # 残差连接 + LayerNorm
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feedforward(x)
        
        # 残差连接 + LayerNorm
        output = self.norm2(x + ff_output)
        
        return {
            'output': output,
            'attention_weights': attention_weights
        }


class Expert(nn.Module):
    """单个专家：多层Encoder"""
    
    def __init__(self, config: MoEConfig, expert_type: str):
        super().__init__()
        self.expert_type = expert_type
        self.config = config
        
        # 多层Encoder
        self.layers = nn.ModuleList([
            ExpertLayer(config, expert_type) 
            for _ in range(config.num_encoder_layers)
        ])
        
        print(f"{expert_type.upper()} Expert initialized: {config.num_encoder_layers} layers")
    
    def forward(self, x: torch.Tensor, 
                thermal_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, T, D] 输入token序列
            thermal_tokens: [B, T_thermal, D] 热成像token
            
        Returns:
            output: [B, T, D] 专家输出
            attention_maps: List[Tensor] 各层注意力权重
        """
        attention_maps = []
        
        for layer in self.layers:
            result = layer(x, thermal_tokens)
            x = result['output']
            attention_maps.append(result['attention_weights'])
        
        return {
            'output': x,
            'attention_maps': attention_maps
        }


class GatingNetwork(nn.Module):
    """门控网络"""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # 门控MLP
        self.gate_network = nn.Sequential(
            nn.Linear(config.embedding_dim, config.gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_hidden_dim, config.gate_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_hidden_dim // 2, config.num_experts)
        )
        
        print(f"Gating Network initialized: {config.embedding_dim} → {config.num_experts} experts")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, T, D] 输入token序列
            
        Returns:
            gate_weights: [B, T, num_experts] 门控权重
            gate_entropy: [B] 门控熵（用于正则化）
        """
        # 计算门控分数
        gate_logits = self.gate_network(x)  # [B, T, num_experts]
        
        # Softmax归一化
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 计算熵（用于正则化）
        gate_entropy = -torch.sum(
            gate_weights * torch.log(gate_weights + 1e-8), 
            dim=-1
        )  # [B, T]
        gate_entropy = gate_entropy.mean(dim=1)  # [B]
        
        return {
            'gate_weights': gate_weights,
            'gate_entropy': gate_entropy,
            'gate_logits': gate_logits
        }


class MoEFusion(nn.Module):
    """
    MoE融合模块
    包含三个专家：Geometric, Semantic, Visual
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # 三个专家
        self.experts = nn.ModuleDict({
            'geometric': Expert(config, 'geometric'),
            'semantic': Expert(config, 'semantic'),  # 使用热引导
            'visual': Expert(config, 'visual')
        })
        
        # 门控网络
        self.gating_network = GatingNetwork(config)
        
        # 输出投影
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        print(f"MoEFusion initialized: {config.num_experts} experts, thermal_guidance={config.thermal_guidance}")
    
    def compute_gate_entropy_loss(self, gate_entropy: torch.Tensor) -> torch.Tensor:
        """
        计算门控熵损失（负熵，防止专家塌缩）
        
        Args:
            gate_entropy: [B] 门控熵
            
        Returns:
            entropy_loss: scalar 熵损失
        """
        # 负熵损失（鼓励均匀分布）
        max_entropy = math.log(self.config.num_experts)  # 均匀分布的最大熵
        entropy_loss = max_entropy - gate_entropy.mean()  # 负熵
        
        return entropy_loss * self.config.gate_entropy_weight
    
    def forward(self, token_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            token_dict: 各模态token字典
                - 'rgb': [B, T_rgb, D]
                - 'depth': [B, T_depth, D]
                - 'thermal': [B, T_thermal, D]
                - 'lidar': [B, T_lidar, D]
                - 'imu': [B, T_imu, D]
                
        Returns:
            result_dict: 包含融合结果和中间信息
        """
        # 拼接所有token
        all_tokens = []
        modality_boundaries = {}
        current_idx = 0
        
        for modality, tokens in token_dict.items():
            all_tokens.append(tokens)
            modality_boundaries[modality] = (current_idx, current_idx + tokens.shape[1])
            current_idx += tokens.shape[1]
        
        # 拼接：[B, Total_T, D]
        fused_tokens = torch.cat(all_tokens, dim=1)
        batch_size, total_tokens, embedding_dim = fused_tokens.shape
        
        # 门控决策
        gate_result = self.gating_network(fused_tokens)
        gate_weights = gate_result['gate_weights']  # [B, Total_T, 3]
        gate_entropy = gate_result['gate_entropy']  # [B]
        
        # 专家处理
        expert_outputs = {}
        expert_attention_maps = {}
        
        for expert_name, expert in self.experts.items():
            if expert_name == 'semantic' and 'thermal' in token_dict:
                # Semantic专家使用热引导
                expert_result = expert(fused_tokens, token_dict['thermal'])
            else:
                # 其他专家标准处理
                expert_result = expert(fused_tokens)
            
            expert_outputs[expert_name] = expert_result['output']
            expert_attention_maps[expert_name] = expert_result['attention_maps']
        
        # 专家输出加权融合
        expert_names = ['geometric', 'semantic', 'visual']
        weighted_outputs = []
        
        for i, expert_name in enumerate(expert_names):
            expert_output = expert_outputs[expert_name]  # [B, Total_T, D]
            expert_weight = gate_weights[:, :, i:i+1]  # [B, Total_T, 1]
            weighted_output = expert_output * expert_weight
            weighted_outputs.append(weighted_output)
        
        # 融合所有专家输出
        fused_output = sum(weighted_outputs)  # [B, Total_T, D]
        
        # 输出投影
        final_output = self.output_projection(fused_output)
        
        # 计算门控熵损失
        entropy_loss = self.compute_gate_entropy_loss(gate_entropy)
        
        # 分解回各模态
        output_tokens = {}
        for modality, (start_idx, end_idx) in modality_boundaries.items():
            output_tokens[modality] = final_output[:, start_idx:end_idx, :]
        
        return {
            'fused_tokens': output_tokens,
            'gate_weights': gate_weights,
            'gate_entropy': gate_entropy,
            'entropy_loss': entropy_loss,
            'expert_outputs': expert_outputs,
            'expert_attention_maps': expert_attention_maps,
            'modality_boundaries': modality_boundaries
        }


def create_moe_fusion(embedding_dim: int = 512, **kwargs) -> MoEFusion:
    """
    创建MoE融合模块的便捷函数
    
    Args:
        embedding_dim: 嵌入维度
        **kwargs: 其他配置参数
        
    Returns:
        MoEFusion模块
    """
    config = MoEConfig(embedding_dim=embedding_dim, **kwargs)
    return MoEFusion(config)


# 模块测试
if __name__ == '__main__':
    # 测试配置
    config = MoEConfig(
        embedding_dim=512,
        num_experts=3,
        num_encoder_layers=2,
        nhead=4,
        thermal_guidance=True
    )
    
    # 创建模块
    moe_fusion = MoEFusion(config)
    
    # 创建测试数据
    batch_size = 2
    test_tokens = {
        'rgb': torch.randn(batch_size, 192, 512),
        'depth': torch.randn(batch_size, 192, 512),
        'thermal': torch.randn(batch_size, 192, 512),
        'lidar': torch.randn(batch_size, 1, 512),
        'imu': torch.randn(batch_size, 1, 512),
    }
    
    # 前向传播
    with torch.no_grad():
        result = moe_fusion(test_tokens)
    
    print("\n🧪 MoEFusion Test Results:")
    print(f"✓ Input total tokens: {sum(t.shape[1] for t in test_tokens.values())}")
    print(f"✓ Gate weights shape: {result['gate_weights'].shape}")
    print(f"✓ Gate entropy: {result['gate_entropy'].mean().item():.4f}")
    print(f"✓ Entropy loss: {result['entropy_loss'].item():.6f}")
    print(f"✓ Expert outputs: {list(result['expert_outputs'].keys())}")
    print(f"✓ Fused tokens: {list(result['fused_tokens'].keys())}")
    
    print("\n🎉 MoEFusion module test passed!")