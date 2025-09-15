"""
MoEFusion Unit Tests
MoE融合模块的单元测试
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from models.moe_fusion import (
    MoEFusion, MoEConfig, ThermalGuidedAttention, 
    Expert, GatingNetwork, create_moe_fusion
)


class TestMoEFusion(unittest.TestCase):
    """MoEFusion单元测试"""
    
    def setUp(self):
        """测试前设置"""
        self.config = MoEConfig(
            embedding_dim=256,  # 较小配置便于测试
            num_experts=3,
            num_encoder_layers=1,
            nhead=4,
            thermal_guidance=True,
            gate_entropy_weight=0.01
        )
        
        self.batch_size = 2
        self.test_tokens = {
            'rgb': torch.randn(self.batch_size, 96, 256),
            'depth': torch.randn(self.batch_size, 96, 256),
            'thermal': torch.randn(self.batch_size, 96, 256),
            'lidar': torch.randn(self.batch_size, 1, 256),
            'imu': torch.randn(self.batch_size, 1, 256),
        }
    
    def test_config_creation(self):
        """测试配置创建"""
        config = MoEConfig()
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.num_experts, 3)
        self.assertTrue(config.thermal_guidance)
    
    def test_thermal_guided_attention(self):
        """测试热引导注意力"""
        attention = ThermalGuidedAttention(self.config)
        
        seq_len = 100
        query = torch.randn(self.batch_size, seq_len, self.config.embedding_dim)
        key = value = query
        thermal_tokens = torch.randn(self.batch_size, 96, self.config.embedding_dim)
        
        # 测试无热引导
        output1, weights1 = attention(query, key, value)
        self.assertEqual(output1.shape, query.shape)
        self.assertEqual(weights1.shape, (self.batch_size, seq_len, seq_len))
        
        # 测试有热引导
        output2, weights2 = attention(query, key, value, thermal_tokens)
        self.assertEqual(output2.shape, query.shape)
        self.assertEqual(weights2.shape, (self.batch_size, seq_len, seq_len))
        
        # 验证热引导改变了注意力权重
        self.assertFalse(torch.equal(weights1, weights2))
    
    def test_thermal_mask_generation(self):
        """测试热mask生成"""
        attention = ThermalGuidedAttention(self.config)
        thermal_tokens = torch.randn(self.batch_size, 50, self.config.embedding_dim)
        target_length = 100
        
        mask = attention.generate_thermal_mask(thermal_tokens, target_length)
        
        self.assertEqual(mask.shape, (self.batch_size, target_length))
        self.assertTrue(torch.all(mask >= 0))
        self.assertTrue(torch.all(mask <= 1))
    
    def test_expert_layer(self):
        """测试专家层"""
        from models.moe_fusion import ExpertLayer
        
        # 测试普通专家层
        generic_layer = ExpertLayer(self.config, 'generic')
        x = torch.randn(self.batch_size, 100, self.config.embedding_dim)
        
        result = generic_layer(x)
        self.assertEqual(result['output'].shape, x.shape)
        self.assertIn('attention_weights', result)
        
        # 测试语义专家层（带热引导）
        semantic_layer = ExpertLayer(self.config, 'semantic')
        thermal_tokens = torch.randn(self.batch_size, 96, self.config.embedding_dim)
        
        result = semantic_layer(x, thermal_tokens)
        self.assertEqual(result['output'].shape, x.shape)
        self.assertIn('attention_weights', result)
    
    def test_expert(self):
        """测试专家模块"""
        expert = Expert(self.config, 'geometric')
        x = torch.randn(self.batch_size, 100, self.config.embedding_dim)
        
        result = expert(x)
        
        self.assertEqual(result['output'].shape, x.shape)
        self.assertIn('attention_maps', result)
        self.assertEqual(len(result['attention_maps']), self.config.num_encoder_layers)
    
    def test_gating_network(self):
        """测试门控网络"""
        gating = GatingNetwork(self.config)
        x = torch.randn(self.batch_size, 100, self.config.embedding_dim)
        
        result = gating(x)
        
        # 检查门控权重
        gate_weights = result['gate_weights']
        self.assertEqual(gate_weights.shape, (self.batch_size, 100, self.config.num_experts))
        
        # 检查权重和为1
        weight_sums = torch.sum(gate_weights, dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))
        
        # 检查熵
        gate_entropy = result['gate_entropy']
        self.assertEqual(gate_entropy.shape, (self.batch_size,))
        self.assertTrue(torch.all(gate_entropy >= 0))
    
    def test_moe_fusion_forward(self):
        """测试MoE融合前向传播"""
        moe_fusion = MoEFusion(self.config)
        
        result = moe_fusion(self.test_tokens)
        
        # 检查输出格式
        self.assertIn('fused_tokens', result)
        self.assertIn('gate_weights', result)
        self.assertIn('gate_entropy', result)
        self.assertIn('entropy_loss', result)
        self.assertIn('expert_outputs', result)
        
        # 检查融合token形状
        fused_tokens = result['fused_tokens']
        for modality, original_tokens in self.test_tokens.items():
            self.assertIn(modality, fused_tokens)
            self.assertEqual(fused_tokens[modality].shape, original_tokens.shape)
        
        # 检查专家输出
        expert_outputs = result['expert_outputs']
        self.assertEqual(len(expert_outputs), 3)
        self.assertIn('geometric', expert_outputs)
        self.assertIn('semantic', expert_outputs)
        self.assertIn('visual', expert_outputs)
        
        # 检查损失值
        entropy_loss = result['entropy_loss']
        self.assertIsInstance(entropy_loss.item(), float)
        self.assertTrue(entropy_loss.item() >= 0)
    
    def test_different_token_lengths(self):
        """测试不同token长度"""
        # 创建不同长度的token
        varied_tokens = {
            'rgb': torch.randn(self.batch_size, 50, self.config.embedding_dim),
            'depth': torch.randn(self.batch_size, 30, self.config.embedding_dim),
            'thermal': torch.randn(self.batch_size, 40, self.config.embedding_dim),
            'lidar': torch.randn(self.batch_size, 1, self.config.embedding_dim),
            'imu': torch.randn(self.batch_size, 1, self.config.embedding_dim),
        }
        
        moe_fusion = MoEFusion(self.config)
        result = moe_fusion(varied_tokens)
        
        # 验证输出token长度保持不变
        for modality, original_tokens in varied_tokens.items():
            output_tokens = result['fused_tokens'][modality]
            self.assertEqual(output_tokens.shape, original_tokens.shape)
    
    def test_thermal_guidance_toggle(self):
        """测试热引导开关"""
        # 关闭热引导
        config_no_thermal = MoEConfig(
            embedding_dim=256,
            thermal_guidance=False
        )
        
        moe_no_thermal = MoEFusion(config_no_thermal)
        result_no_thermal = moe_no_thermal(self.test_tokens)
        
        # 开启热引导
        moe_with_thermal = MoEFusion(self.config)
        result_with_thermal = moe_with_thermal(self.test_tokens)
        
        # 两种配置都应该正常工作
        self.assertIn('fused_tokens', result_no_thermal)
        self.assertIn('fused_tokens', result_with_thermal)
    
    def test_gradient_flow(self):
        """测试梯度流"""
        moe_fusion = MoEFusion(self.config)
        
        # 启用梯度
        for token_batch in self.test_tokens.values():
            token_batch.requires_grad_(True)
        
        result = moe_fusion(self.test_tokens)
        
        # 创建损失
        total_loss = result['entropy_loss']
        for tokens in result['fused_tokens'].values():
            total_loss = total_loss + tokens.mean()
        
        # 反向传播
        total_loss.backward()
        
        # 检查主要参数的梯度
        gradient_found = False
        for name, param in moe_fusion.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_found = True
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")
        
        # 确保至少有一些梯度被计算
        self.assertTrue(gradient_found, "No gradients found in the model")
    
    def test_create_moe_fusion_function(self):
        """测试便捷创建函数"""
        moe_fusion = create_moe_fusion(
            embedding_dim=256,
            num_experts=3,
            thermal_guidance=True
        )
        
        self.assertIsInstance(moe_fusion, MoEFusion)
        self.assertEqual(moe_fusion.config.embedding_dim, 256)
        self.assertEqual(moe_fusion.config.num_experts, 3)
        self.assertTrue(moe_fusion.config.thermal_guidance)
    
    def test_expert_specialization(self):
        """测试专家特化"""
        moe_fusion = MoEFusion(self.config)
        
        # 多次前向传播，检查门控权重的变化
        gate_weights_history = []
        
        for _ in range(3):
            result = moe_fusion(self.test_tokens)
            gate_weights_history.append(result['gate_weights'].clone())
        
        # 检查门控权重形状一致性
        for gate_weights in gate_weights_history:
            total_tokens = sum(t.shape[1] for t in self.test_tokens.values())
            expected_shape = (self.batch_size, total_tokens, 3)
            self.assertEqual(gate_weights.shape, expected_shape)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_pipeline(self):
        """测试端到端流水线"""
        # 模拟真实尺寸的数据
        batch_size = 1
        real_tokens = {
            'rgb': torch.randn(batch_size, 192, 512),      # 典型RGB token数
            'depth': torch.randn(batch_size, 192, 512),    # 深度token
            'thermal': torch.randn(batch_size, 192, 512),  # 热成像token
            'lidar': torch.randn(batch_size, 1, 512),      # LiDAR token
            'imu': torch.randn(batch_size, 1, 512),        # IMU token
        }
        
        config = MoEConfig(
            embedding_dim=512,
            num_experts=3,
            num_encoder_layers=2,
            nhead=8,
            thermal_guidance=True
        )
        
        moe_fusion = MoEFusion(config)
        
        # 前向传播
        with torch.no_grad():
            result = moe_fusion(real_tokens)
        
        # 验证关键输出
        self.assertIn('fused_tokens', result)
        self.assertIn('gate_weights', result)
        self.assertIn('entropy_loss', result)
        
        # 检查输出尺寸
        for modality, tokens in real_tokens.items():
            output_tokens = result['fused_tokens'][modality]
            self.assertEqual(output_tokens.shape, tokens.shape)
        
        print(f"✓ End-to-end test passed with {sum(t.shape[1] for t in real_tokens.values())} total tokens")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)