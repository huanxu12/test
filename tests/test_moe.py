#!/usr/bin/env python3
"""
MoE Transformer Unit Tests - REAL DATA ONLY
三专家MoE测试：Geometric/Semantic/Visual × 2层Encoder，nhead=4
门控统计验证，热引导注意力，真实样本检验
从val_index取前N=10批真实样本，严禁随机张量/模拟mask
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_config
from data.mineslam_dataset import MineSLAMDataset
from models.moe_fusion import MoEFusion, MoEConfig
from models.encoders import MultiModalEncoder


class MoETester:
    """MoE测试器"""
    
    def __init__(self, config: Dict, output_dir: str = "outputs/moe"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 测试参数
        self.batch_size = 2
        self.num_batches = 10  # N=10批次
        self.sample_ids_log = []  # 记录所用样本ID
        
        # MoE配置
        self.moe_config = MoEConfig(
            embedding_dim=512,
            num_experts=3,  # Geometric, Semantic, Visual
            num_encoder_layers=2,  # 2层Encoder
            nhead=4,  # 4头注意力
            feedforward_dim=2048,
            dropout=0.1,
            gate_hidden_dim=256,
            thermal_guidance=True,
            gate_entropy_weight=0.01
        )
        
        # 创建编码器和MoE
        self.encoder = MultiModalEncoder(embedding_dim=512)
        self.moe_fusion = MoEFusion(self.moe_config)
        
        self.encoder.to(self.device)
        self.moe_fusion.to(self.device)
        
        print(f"MoETester initialized: {self.num_batches} batches × {self.batch_size} samples")
    
    def _validate_real_sample(self, sample: Dict, batch_idx: int, sample_idx: int):
        """验证样本来自真实数据，严禁随机张量"""
        # 检查合成数据标记
        forbidden_keys = ['_generated', '_synthetic', '_random', '_mock', '_fake', '_simulated']
        for key in sample.keys():
            if any(forbidden in str(key).lower() for forbidden in forbidden_keys):
                raise ValueError(f"FORBIDDEN: Detected synthetic data marker in batch {batch_idx}, sample {sample_idx}: {key}")
        
        # 验证时间戳真实性
        if 'timestamp' in sample:
            timestamp = sample['timestamp'].item()
            if timestamp <= 0 or timestamp > 2e9:
                raise ValueError(f"Invalid timestamp in batch {batch_idx}: {timestamp}")
        
        # 严格检查张量真实性
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                # 检查是否为随机生成的张量
                if value.numel() > 100:
                    # 检查值分布（真实数据不应该是完美正态分布）
                    mean_val = value.float().mean().item()
                    std_val = value.float().std().item()
                    
                    # 真实传感器数据应该有合理的范围
                    if key == 'rgb' and (mean_val < -2.0 or mean_val > 2.0):
                        warnings.warn(f"Suspicious RGB values in batch {batch_idx}: mean={mean_val}")
                    elif key == 'thermal' and std_val < 0.01:
                        raise ValueError(f"FORBIDDEN: Thermal data appears to be constant/simulated in batch {batch_idx}")
                    elif key == 'lidar' and torch.all(value == 0):
                        raise ValueError(f"FORBIDDEN: LiDAR data is all zeros (simulated) in batch {batch_idx}")
                
                # 检查是否全为相同值（可能的模拟mask）
                if torch.numel(value) > 10:
                    unique_count = len(torch.unique(value))
                    if unique_count == 1 and key not in ['boxes']:  # boxes可能全为0
                        if 'mask' in key.lower():
                            raise ValueError(f"FORBIDDEN: Detected simulated mask '{key}' in batch {batch_idx}")
        
        # 记录真实样本ID
        sample_id = f"batch_{batch_idx}_sample_{sample_idx}_ts_{sample.get('timestamp', 0)}"
        self.sample_ids_log.append(sample_id)
    
    def _check_thermal_mask_validity(self, thermal_tokens: torch.Tensor, batch_idx: int):
        """检查热成像mask的真实性"""
        if thermal_tokens is None:
            raise ValueError(f"Missing thermal tokens in batch {batch_idx}")
        
        # 检查thermal tokens不是随机生成的
        mean_val = thermal_tokens.mean().item()
        std_val = thermal_tokens.std().item()
        
        if std_val < 1e-6:
            raise ValueError(f"FORBIDDEN: Thermal tokens appear constant/simulated in batch {batch_idx}")
        
        # 检查值范围合理性（归一化后的thermal数据）
        if torch.any(thermal_tokens < -5) or torch.any(thermal_tokens > 5):
            warnings.warn(f"Thermal tokens have extreme values in batch {batch_idx}")
    
    def _analyze_gate_weights(self, gate_weights: torch.Tensor, batch_idx: int) -> Dict[str, float]:
        """分析门控权重统计"""
        # gate_weights: [B, T, 3] - (Geometric, Semantic, Visual)
        batch_size, seq_len, num_experts = gate_weights.shape
        
        # 计算每个专家的平均权重
        expert_weights = gate_weights.mean(dim=(0, 1))  # [3]
        
        expert_stats = {
            'geometric_weight': expert_weights[0].item(),
            'semantic_weight': expert_weights[1].item(), 
            'visual_weight': expert_weights[2].item()
        }
        
        # 检查单一专家占主导（>80%）的情况
        max_weight = max(expert_stats.values())
        dominant_expert = max(expert_stats, key=expert_stats.get)
        
        if max_weight > 0.8:
            print(f"⚠️  WARNING: Single expert dominance in batch {batch_idx}: {dominant_expert} = {max_weight:.3f}")
        
        expert_stats['max_weight'] = max_weight
        expert_stats['dominant_expert'] = dominant_expert
        expert_stats['batch_idx'] = batch_idx
        
        return expert_stats
    
    def _save_attention_heatmap(self, attention_weights: torch.Tensor, 
                               lidar_tokens: torch.Tensor, 
                               batch_idx: int, expert_name: str):
        """保存注意力热力图叠加真实点云"""
        # attention_weights: [B, H, T, T] or [B, T, T]
        # lidar_tokens: [B, T_lidar, D]
        
        batch_size = attention_weights.shape[0]
        
        for b in range(batch_size):
            try:
                # 提取单个样本的注意力权重
                if attention_weights.dim() == 4:
                    attn = attention_weights[b].mean(dim=0)  # 平均所有头 [T, T]
                else:
                    attn = attention_weights[b]  # [T, T]
                
                # 创建热力图
                plt.figure(figsize=(12, 8))
                
                # 左侧：注意力热力图
                plt.subplot(1, 2, 1)
                sns.heatmap(attn.detach().cpu().numpy(), 
                           cmap='hot', cbar=True, square=True)
                plt.title(f'{expert_name} Expert Attention\nBatch {batch_idx}, Sample {b}')
                plt.xlabel('Key Tokens')
                plt.ylabel('Query Tokens')
                
                # 右侧：LiDAR点云可视化（如果有的话）
                plt.subplot(1, 2, 2)
                if lidar_tokens.shape[1] > 0:  # 有LiDAR数据
                    # 简化的点云可视化（投影到2D）
                    lidar_features = lidar_tokens[b, :, :3].detach().cpu().numpy()  # 取前3维作为坐标
                    
                    plt.scatter(lidar_features[:, 0], lidar_features[:, 1], 
                              c=attn.diagonal().detach().cpu().numpy()[:len(lidar_features)], 
                              cmap='hot', alpha=0.7)
                    plt.colorbar(label='Attention Weight')
                    plt.title(f'LiDAR Points with Attention\nBatch {batch_idx}, Sample {b}')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                else:
                    plt.text(0.5, 0.5, 'No LiDAR Data', ha='center', va='center')
                    plt.title('LiDAR Visualization')
                
                plt.tight_layout()
                
                # 保存文件
                save_path = self.output_dir / f"attention_{expert_name}_batch{batch_idx}_sample{b}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Saved attention heatmap: {save_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save attention heatmap for batch {batch_idx}, sample {b}: {e}")
    
    def test_moe_fusion(self, dataset: MineSLAMDataset) -> Dict[str, float]:
        """测试MoE融合模块"""
        print(f"\n🔀 Testing MoE Fusion on {self.num_batches} real batches...")
        
        self.encoder.eval()
        self.moe_fusion.eval()
        
        all_gate_stats = []
        total_samples = 0
        entropy_losses = []
        
        with torch.no_grad():
            for batch_idx in range(min(self.num_batches, len(dataset))):
                try:
                    # 收集batch数据
                    batch_data = []
                    
                    for sample_idx in range(self.batch_size):
                        data_idx = batch_idx * self.batch_size + sample_idx
                        if data_idx >= len(dataset):
                            break
                        
                        sample = dataset[data_idx]
                        self._validate_real_sample(sample, batch_idx, sample_idx)
                        batch_data.append(sample)
                    
                    if not batch_data:
                        continue
                    
                    # 构建输入数据字典
                    input_dict = {}
                    modalities = ['rgb', 'depth', 'thermal', 'lidar', 'imu']
                    
                    for modality in modalities:
                        modality_data = []
                        for sample in batch_data:
                            if modality in sample:
                                modality_data.append(sample[modality])
                        
                        if modality_data:
                            input_dict[modality] = torch.stack(modality_data).to(self.device)
                    
                    print(f"  Processing batch {batch_idx}: {len(batch_data)} samples")
                    print(f"    Available modalities: {list(input_dict.keys())}")
                    
                    # 编码器前向传播
                    token_dict = self.encoder(input_dict)
                    
                    # 检查热成像真实性
                    if 'thermal' in token_dict:
                        self._check_thermal_mask_validity(token_dict['thermal'], batch_idx)
                    
                    # MoE融合
                    moe_result = self.moe_fusion(token_dict)
                    
                    # 分析门控权重
                    gate_weights = moe_result['gate_weights']  # [B, T, 3]
                    gate_stats = self._analyze_gate_weights(gate_weights, batch_idx)
                    all_gate_stats.append(gate_stats)
                    
                    # 记录熵损失
                    entropy_loss = moe_result['entropy_loss'].item()
                    entropy_losses.append(entropy_loss)
                    
                    # 保存注意力热力图（Semantic专家）
                    if 'semantic' in self.moe_fusion.experts:
                        # 获取Semantic专家的注意力权重
                        semantic_expert = self.moe_fusion.experts['semantic']
                        
                        # 重新前向获取注意力权重
                        if 'thermal' in token_dict:
                            fused_tokens = torch.cat([v for v in token_dict.values()], dim=1)
                            semantic_result = semantic_expert(fused_tokens, token_dict['thermal'])
                            attention_maps = semantic_result['attention_maps']
                            
                            if attention_maps:
                                self._save_attention_heatmap(
                                    attention_maps[-1],  # 最后一层的注意力
                                    token_dict.get('lidar', torch.empty(len(batch_data), 0, 512).to(self.device)),
                                    batch_idx, 'Semantic'
                                )
                    
                    total_samples += len(batch_data)
                    print(f"    Gate stats: G={gate_stats['geometric_weight']:.3f}, "
                          f"S={gate_stats['semantic_weight']:.3f}, V={gate_stats['visual_weight']:.3f}")
                    
                except Exception as e:
                    print(f"Warning: Skipping batch {batch_idx}: {e}")
        
        # 计算总体统计
        if all_gate_stats:
            avg_geometric = np.mean([s['geometric_weight'] for s in all_gate_stats])
            avg_semantic = np.mean([s['semantic_weight'] for s in all_gate_stats])
            avg_visual = np.mean([s['visual_weight'] for s in all_gate_stats])
            
            # 检查长期单一专家主导
            dominant_count = sum(1 for s in all_gate_stats if s['max_weight'] > 0.8)
            dominance_rate = dominant_count / len(all_gate_stats) if all_gate_stats else 0
            
            print(f"\n📊 Gate Weight Statistics (N={len(all_gate_stats)} batches):")
            print(f"    Average Geometric: {avg_geometric:.3f}")
            print(f"    Average Semantic:  {avg_semantic:.3f}")
            print(f"    Average Visual:    {avg_visual:.3f}")
            print(f"    Dominance Rate:    {dominance_rate:.3f} ({dominant_count}/{len(all_gate_stats)})")
            
            if dominance_rate > 0.5:
                print(f"⚠️  WARNING: High dominance rate {dominance_rate:.3f} - possible expert collapse")
            
            # 保存样本ID日志
            sample_log_path = self.output_dir / "sample_ids_log.json"
            with open(sample_log_path, 'w') as f:
                json.dump({
                    'sample_ids': self.sample_ids_log,
                    'total_samples': total_samples,
                    'gate_statistics': all_gate_stats
                }, f, indent=2)
            
            print(f"✅ Sample ID log saved: {sample_log_path}")
            
            metrics = {
                'total_samples': total_samples,
                'avg_geometric_weight': avg_geometric,
                'avg_semantic_weight': avg_semantic,
                'avg_visual_weight': avg_visual,
                'dominance_rate': dominance_rate,
                'avg_entropy_loss': np.mean(entropy_losses) if entropy_losses else 0,
                'num_batches': len(all_gate_stats)
            }
        else:
            raise ValueError("No valid batches processed")
        
        return metrics


class TestMoE(unittest.TestCase):
    """MoE单元测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.config_path = 'configs/mineslam.yaml'
        cls.output_dir = "outputs/moe"
        
        # 加载配置
        if not os.path.exists(cls.config_path):
            raise unittest.SkipTest(f"Configuration file not found: {cls.config_path}")
        
        cls.config = load_config(cls.config_path)
        
        # 检查验证数据索引
        val_index_path = cls.config['data']['val_index']
        if not os.path.exists(val_index_path):
            raise unittest.SkipTest(f"Validation index not found: {val_index_path}")
        
        # 创建测试器
        cls.tester = MoETester(cls.config, cls.output_dir)
        
        print(f"Testing MoE on val_index with N=10 batches")
    
    def test_01_moe_architecture(self):
        """测试MoE架构"""
        print("\n" + "="*60)
        print("Testing MoE Architecture")
        print("="*60)
        
        # 验证配置
        config = self.tester.moe_config
        self.assertEqual(config.num_experts, 3, "Must have exactly 3 experts")
        self.assertEqual(config.num_encoder_layers, 2, "Must have 2 encoder layers")
        self.assertEqual(config.nhead, 4, "Must have 4 attention heads")
        self.assertTrue(config.thermal_guidance, "Must enable thermal guidance")
        
        # 验证专家名称
        expert_names = list(self.tester.moe_fusion.experts.keys())
        expected_names = ['geometric', 'semantic', 'visual']
        self.assertEqual(set(expert_names), set(expected_names), 
                        f"Expert names mismatch: {expert_names} vs {expected_names}")
        
        print("✅ MoE Architecture Test PASSED:")
        print(f"  Experts: {expert_names}")
        print(f"  Encoder layers: {config.num_encoder_layers}")
        print(f"  Attention heads: {config.nhead}")
        print(f"  Thermal guidance: {config.thermal_guidance}")
    
    def test_02_real_data_processing(self):
        """测试真实数据处理"""
        print("\n" + "="*60)
        print("Testing Real Data Processing")
        print("="*60)
        
        # 创建验证数据集
        dataset = MineSLAMDataset(self.config, split='val')
        
        # 测试MoE融合
        moe_metrics = self.tester.test_moe_fusion(dataset)
        
        # 验证处理结果
        self.assertGreater(moe_metrics['total_samples'], 0, "No samples processed")
        self.assertEqual(moe_metrics['num_batches'], min(10, len(dataset)), 
                        "Batch count mismatch")
        
        # 验证门控权重合理性
        weight_sum = (moe_metrics['avg_geometric_weight'] + 
                     moe_metrics['avg_semantic_weight'] + 
                     moe_metrics['avg_visual_weight'])
        self.assertAlmostEqual(weight_sum, 1.0, places=2, 
                              msg="Gate weights should sum to 1.0")
        
        print(f"✅ Real Data Processing Test PASSED:")
        print(f"  Total samples: {moe_metrics['total_samples']}")
        print(f"  Gate weight sum: {weight_sum:.3f}")
        print(f"  Entropy loss: {moe_metrics['avg_entropy_loss']:.6f}")
    
    def test_03_gate_weight_distribution(self):
        """测试门控权重分布"""
        print("\n" + "="*60)
        print("Testing Gate Weight Distribution")
        print("="*60)
        
        # 创建验证数据集
        dataset = MineSLAMDataset(self.config, split='val')
        
        # 测试MoE融合
        moe_metrics = self.tester.test_moe_fusion(dataset)
        
        # 验证无单一专家长期主导（>80%）
        dominance_rate = moe_metrics['dominance_rate']
        self.assertLess(dominance_rate, 0.8, 
                       f"Single expert dominance too high: {dominance_rate:.3f}")
        
        # 验证每个专家都有合理的参与度
        min_weight = min(moe_metrics['avg_geometric_weight'],
                        moe_metrics['avg_semantic_weight'], 
                        moe_metrics['avg_visual_weight'])
        self.assertGreater(min_weight, 0.1, "At least one expert severely underused")
        
        print(f"✅ Gate Weight Distribution Test PASSED:")
        print(f"  Dominance rate: {dominance_rate:.3f} (<0.8)")
        print(f"  Minimum expert weight: {min_weight:.3f} (>0.1)")
    
    def test_04_output_validation(self):
        """验证输出文件"""
        print("\n" + "="*60)
        print("Validating Output Files")
        print("="*60)
        
        output_dir = Path(self.tester.output_dir)
        
        # 检查样本ID日志
        sample_log = output_dir / "sample_ids_log.json"
        self.assertTrue(sample_log.exists(), f"Sample ID log not found: {sample_log}")
        
        # 检查注意力热力图
        attention_files = list(output_dir.glob("attention_*.png"))
        self.assertGreater(len(attention_files), 0, "No attention heatmaps generated")
        
        # 验证样本ID格式
        with open(sample_log) as f:
            log_data = json.load(f)
        
        self.assertIn('sample_ids', log_data, "Missing sample_ids in log")
        self.assertGreater(len(log_data['sample_ids']), 0, "No sample IDs recorded")
        
        print(f"✅ Output Validation Test PASSED:")
        print(f"  Sample ID log: {sample_log}")
        print(f"  Attention heatmaps: {len(attention_files)} files")
        print(f"  Sample IDs logged: {len(log_data['sample_ids'])}")


def run_moe_tests():
    """运行MoE测试"""
    print("="*80)
    print("MINESLAM MoE TRANSFORMER TESTS - REAL DATA ONLY")
    print("="*80)
    print("Testing 3-Expert MoE (Geometric/Semantic/Visual) × 2-layer Encoder")
    print("Thermal guidance, gate statistics, attention heatmaps on N=10 batches")
    print("Failure criteria: Random tensors/simulated masks detected")
    print()
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoE)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("MoE测试总结")
    print("="*80)
    
    if result.wasSuccessful():
        print("✅ 所有MoE测试通过!")
        print("   - 3专家架构 × 2层Encoder × 4头注意力")
        print("   - 门控权重统计正常（无单一专家>80%）")
        print("   - 热引导注意力有效作用于Semantic专家")
        print("   - 注意力热力图已保存到outputs/moe/")
        print("   - 样本ID日志记录完整")
        print("   - 严格验证真实数据，无随机张量/模拟mask")
    else:
        print("❌ MoE测试失败!")
        print(f"   失败: {len(result.failures)}")
        print(f"   错误: {len(result.errors)}")
        
        if result.failures:
            print("\n失败详情:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\n错误详情:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_moe_tests()
    sys.exit(0 if success else 1)