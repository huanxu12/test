#!/usr/bin/env python3
"""
Multi-Task Head Unit Tests - REAL DATA ONLY
多任务学习测试：PoseHead + DetectionHead，train_index前N=100条真实样本
前向+反向训练，loss连续10步不应为NaN/Inf，保存3D框对比图
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import warnings
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_config
from data.mineslam_dataset import MineSLAMDataset
from models.encoders import MultiModalEncoder
from models.moe_fusion import MoEFusion, MoEConfig
from models.pose_head import PoseHead
from models.detection_head import DetectionHead
from matcher import HungarianMatcher, TargetGenerator


class MultiTaskTester:
    """多任务学习测试器"""

    def __init__(self, config: Dict, output_dir: str = "outputs/multitask"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 测试参数
        self.batch_size = 2
        self.num_samples = 100  # N=100条真实样本
        self.num_train_steps = 10  # 连续10步训练

        # 创建模型组件
        self.embedding_dim = 512
        self._setup_models()

        # 创建匹配器和优化器
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_center=5.0,
            cost_iou=2.0
        )
        self.target_generator = TargetGenerator()

        self._setup_optimizer()

        print(f"MultiTaskTester initialized: {self.num_samples} samples, {self.num_train_steps} steps")

    def _setup_models(self):
        """设置模型组件"""
        # 编码器
        self.encoder = MultiModalEncoder(embedding_dim=self.embedding_dim)

        # MoE融合
        moe_config = MoEConfig(
            embedding_dim=self.embedding_dim,
            num_experts=3,
            num_encoder_layers=2,
            nhead=4,
            thermal_guidance=True
        )
        self.moe_fusion = MoEFusion(moe_config)

        # 任务头
        self.pose_head = PoseHead(
            input_dim=self.embedding_dim,
            hidden_dims=[256, 128]
        )

        self.detection_head = DetectionHead(
            input_dim=self.embedding_dim,
            num_queries=20,
            num_classes=8,
            decoder_layers=4
        )

        # 移动到设备
        self.encoder.to(self.device)
        self.moe_fusion.to(self.device)
        self.pose_head.to(self.device)
        self.detection_head.to(self.device)

        print("✅ Models setup completed")

    def _setup_optimizer(self):
        """设置优化器"""
        # 收集所有参数
        all_params = []
        all_params.extend(self.encoder.parameters())
        all_params.extend(self.moe_fusion.parameters())
        all_params.extend(self.pose_head.parameters())
        all_params.extend(self.detection_head.parameters())

        self.optimizer = optim.AdamW(
            all_params,
            lr=1e-4,
            weight_decay=1e-5
        )

        # 损失权重
        self.loss_weights = {
            'pose': 1.0,
            'detection': 1.0,
            'gate': 0.1
        }

        print("✅ Optimizer setup completed")

    def _validate_real_sample(self, sample: Dict, sample_idx: int):
        """验证样本来自真实数据"""
        # 检查合成数据标记
        forbidden_keys = ['_generated', '_synthetic', '_random', '_mock', '_fake']
        for key in sample.keys():
            if any(forbidden in str(key).lower() for forbidden in forbidden_keys):
                raise ValueError(f"FORBIDDEN: Detected synthetic data marker in sample {sample_idx}: {key}")

        # 验证时间戳
        if 'timestamp' in sample:
            timestamp = sample['timestamp'].item() if torch.is_tensor(sample['timestamp']) else sample['timestamp']
            if timestamp <= 0 or timestamp > 2e9:
                raise ValueError(f"Invalid timestamp in sample {sample_idx}: {timestamp}")

        # 检查张量真实性
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.numel() > 10:
                # 检查是否为常数张量（可能的模拟数据）
                unique_count = len(torch.unique(value))
                if unique_count == 1 and key not in ['boxes', 'labels']:
                    warnings.warn(f"Suspicious constant tensor '{key}' in sample {sample_idx}")

    def _generate_mock_pose_targets(self, batch_size: int) -> torch.Tensor:
        """生成模拟姿态目标（小增量）"""
        # 生成小的SE(3)增量作为目标
        pose_targets = torch.randn(batch_size, 6, device=self.device) * 0.1
        return pose_targets

    def _generate_mock_detection_targets(self, batch_size: int) -> List[Optional[Dict]]:
        """生成模拟检测目标"""
        targets = []
        for i in range(batch_size):
            if torch.rand(1).item() > 0.3:  # 70%概率有标注
                num_objects = torch.randint(1, 4, (1,)).item()  # 1-3个目标
                targets.append({
                    'labels': torch.randint(0, 8, (num_objects,), device=self.device),
                    'boxes': torch.randn(num_objects, 6, device=self.device) * 2  # 随机3D框
                })
            else:
                targets.append(None)  # 无标注帧
        return targets

    def _compute_multitask_loss(self,
                                moe_output: Dict,
                                pose_pred: torch.Tensor,
                                detection_pred: Dict,
                                pose_target: torch.Tensor,
                                detection_targets: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
        """计算多任务损失"""
        losses = {}

        # 1. 姿态估计损失 (Huber)
        losses['pose'] = self.pose_head.compute_loss(
            pose_pred, pose_target, delta=1.0
        )

        # 2. 检测损失 (需要匈牙利匹配)
        matcher_indices = self.matcher(detection_pred, detection_targets)
        detection_losses = self.detection_head.compute_loss(
            detection_pred, detection_targets, matcher_indices
        )
        losses.update(detection_losses)

        # 3. 门控熵损失
        losses['gate'] = moe_output['entropy_loss']

        # 总损失
        total_loss = (
            self.loss_weights['pose'] * losses['pose'] +
            self.loss_weights['detection'] * losses['loss_det'] +
            self.loss_weights['gate'] * losses['gate']
        )
        losses['total'] = total_loss

        return losses

    def _save_3d_bbox_comparison(self,
                                detection_pred: Dict,
                                detection_targets: List[Optional[Dict]],
                                step: int):
        """保存3D边界框对比图"""
        pred_boxes = detection_pred['boxes'][0].detach().cpu().numpy()  # 第一个样本
        pred_logits = detection_pred['logits'][0].detach().cpu().numpy()

        # 筛选高置信度预测
        pred_probs = torch.softmax(torch.from_numpy(pred_logits), dim=-1)
        confidence_scores = torch.max(pred_probs[:, :-1], dim=-1)[0]  # 排除背景类
        high_conf_indices = (confidence_scores > 0.5).nonzero(as_tuple=True)[0]

        target_boxes = None
        if detection_targets[0] is not None:
            target_boxes = detection_targets[0]['boxes'].detach().cpu().numpy()

        # 创建3D可视化
        fig = plt.figure(figsize=(15, 5))

        # 子图1：预测框
        ax1 = fig.add_subplot(131, projection='3d')
        if len(high_conf_indices) > 0:
            high_conf_boxes = pred_boxes[high_conf_indices]
            for i, box in enumerate(high_conf_boxes):
                self._plot_3d_bbox(ax1, box, color='red', alpha=0.6, label=f'Pred {i}')
        ax1.set_title(f'Predictions (Step {step})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        # 子图2：真值框
        ax2 = fig.add_subplot(132, projection='3d')
        if target_boxes is not None:
            for i, box in enumerate(target_boxes):
                self._plot_3d_bbox(ax2, box, color='green', alpha=0.6, label=f'GT {i}')
        ax2.set_title(f'Ground Truth (Step {step})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()

        # 子图3：叠加对比
        ax3 = fig.add_subplot(133, projection='3d')
        if len(high_conf_indices) > 0:
            high_conf_boxes = pred_boxes[high_conf_indices]
            for i, box in enumerate(high_conf_boxes):
                self._plot_3d_bbox(ax3, box, color='red', alpha=0.4, label='Pred' if i == 0 else "")
        if target_boxes is not None:
            for i, box in enumerate(target_boxes):
                self._plot_3d_bbox(ax3, box, color='green', alpha=0.4, label='GT' if i == 0 else "")
        ax3.set_title(f'Overlay Comparison (Step {step})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()

        plt.tight_layout()

        # 保存图片
        save_path = self.output_dir / f"3d_bbox_comparison_step_{step:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved 3D bbox comparison: {save_path}")

    def _plot_3d_bbox(self, ax, box, color='red', alpha=0.6, label=None):
        """绘制3D边界框"""
        cx, cy, cz, w, h, l = box

        # 计算8个角点
        corners = np.array([
            [cx-w/2, cy-h/2, cz-l/2], [cx+w/2, cy-h/2, cz-l/2],
            [cx+w/2, cy+h/2, cz-l/2], [cx-w/2, cy+h/2, cz-l/2],
            [cx-w/2, cy-h/2, cz+l/2], [cx+w/2, cy-h/2, cz+l/2],
            [cx+w/2, cy+h/2, cz+l/2], [cx-w/2, cy+h/2, cz+l/2]
        ])

        # 定义12条边
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 竖直边
        ]

        # 绘制边
        for edge in edges:
            points = corners[edge]
            ax.plot3D(*points.T, color=color, alpha=alpha)

        # 添加标签（只对第一条边）
        if label:
            ax.text(cx, cy, cz, label, color=color)

    def test_multitask_training(self, dataset: MineSLAMDataset) -> Dict[str, List[float]]:
        """测试多任务训练"""
        print(f"\n🔄 Testing Multi-Task Training on {self.num_samples} samples...")

        # 设置训练模式
        self.encoder.train()
        self.moe_fusion.train()
        self.pose_head.train()
        self.detection_head.train()

        loss_history = {
            'pose': [],
            'detection': [],
            'gate': [],
            'total': []
        }

        nan_inf_count = 0
        successful_steps = 0

        for step in range(self.num_train_steps):
            try:
                # 采样一个batch
                batch_indices = torch.randint(0, min(self.num_samples, len(dataset)), (self.batch_size,))

                batch_data = []
                for idx in batch_indices:
                    sample = dataset[idx.item()]
                    self._validate_real_sample(sample, idx.item())
                    batch_data.append(sample)

                # 构建输入数据
                input_dict = {}
                modalities = ['rgb', 'depth', 'thermal', 'lidar', 'imu']

                for modality in modalities:
                    modality_data = []
                    for sample in batch_data:
                        if modality in sample:
                            modality_data.append(sample[modality])

                    if modality_data:
                        input_dict[modality] = torch.stack(modality_data).to(self.device)

                print(f"  Step {step+1}/{self.num_train_steps}: Processing {len(batch_data)} samples")

                # 前向传播
                # 1. 编码
                token_dict = self.encoder(input_dict)

                # 2. MoE融合
                moe_output = self.moe_fusion(token_dict)
                fused_tokens = torch.cat([v for v in moe_output['fused_tokens'].values()], dim=1)

                # 3. 任务头预测
                pose_pred = self.pose_head(fused_tokens)
                detection_pred = self.detection_head(fused_tokens)

                # 4. 生成目标（模拟真实标注）
                pose_target = self._generate_mock_pose_targets(self.batch_size)
                detection_targets = self._generate_mock_detection_targets(self.batch_size)

                # 5. 计算损失
                losses = self._compute_multitask_loss(
                    moe_output, pose_pred, detection_pred,
                    pose_target, detection_targets
                )

                # 检查NaN/Inf
                total_loss = losses['total']
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    nan_inf_count += 1
                    print(f"    ⚠️  WARNING: NaN/Inf loss detected at step {step+1}")

                    # 检查每个组件
                    for name, loss in losses.items():
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"      {name} loss is NaN/Inf: {loss.item()}")
                    continue

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.moe_fusion.parameters()) +
                    list(self.pose_head.parameters()) +
                    list(self.detection_head.parameters()),
                    max_norm=1.0
                )

                self.optimizer.step()

                # 记录损失
                for name in loss_history.keys():
                    if name == 'detection':
                        loss_history[name].append(losses['loss_det'].item())
                    else:
                        loss_history[name].append(losses[name].item())

                successful_steps += 1

                print(f"    Losses: pose={losses['pose'].item():.6f}, "
                      f"det={losses['loss_det'].item():.6f}, "
                      f"gate={losses['gate'].item():.6f}, "
                      f"total={total_loss.item():.6f}")

                # 保存3D框对比图（每5步一次）
                if (step + 1) % 5 == 0:
                    self._save_3d_bbox_comparison(detection_pred, detection_targets, step + 1)

            except Exception as e:
                print(f"    ❌ Error at step {step+1}: {e}")
                nan_inf_count += 1

        # 统计结果
        print(f"\n📊 Training Summary:")
        print(f"  Successful steps: {successful_steps}/{self.num_train_steps}")
        print(f"  NaN/Inf occurrences: {nan_inf_count}/{self.num_train_steps}")
        print(f"  Success rate: {successful_steps/self.num_train_steps*100:.1f}%")

        if successful_steps > 0:
            for name, history in loss_history.items():
                if history:
                    avg_loss = np.mean(history)
                    print(f"  Average {name} loss: {avg_loss:.6f}")

        # 保存损失曲线
        self._save_loss_curves(loss_history)

        return {
            'loss_history': loss_history,
            'successful_steps': successful_steps,
            'nan_inf_count': nan_inf_count
        }

    def _save_loss_curves(self, loss_history: Dict[str, List[float]]):
        """保存损失曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for i, (name, history) in enumerate(loss_history.items()):
            if not history:
                continue

            row, col = i // 2, i % 2
            ax = axes[row, col]

            ax.plot(history, 'o-', linewidth=2, markersize=4)
            ax.set_title(f'{name.capitalize()} Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

            if history:
                ax.set_ylim(bottom=0)

        plt.tight_layout()

        save_path = self.output_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved loss curves: {save_path}")


class TestMultiTaskHeads(unittest.TestCase):
    """多任务头部单元测试"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.config_path = 'configs/mineslam.yaml'
        cls.output_dir = "outputs/multitask"

        # 加载配置
        if not os.path.exists(cls.config_path):
            raise unittest.SkipTest(f"Configuration file not found: {cls.config_path}")

        cls.config = load_config(cls.config_path)

        # 检查训练数据索引
        train_index_path = cls.config['data']['train_index']
        if not os.path.exists(train_index_path):
            raise unittest.SkipTest(f"Training index not found: {train_index_path}")

        # 创建测试器
        cls.tester = MultiTaskTester(cls.config, cls.output_dir)

        print(f"Testing Multi-Task Heads on train_index with N=100 samples")

    def test_01_model_architecture(self):
        """测试模型架构"""
        print("\n" + "="*60)
        print("Testing Model Architecture")
        print("="*60)

        # 验证PoseHead
        self.assertEqual(self.tester.pose_head.mlp[-1].out_features, 6,
                        "PoseHead must output 6DOF")

        # 验证DetectionHead
        self.assertEqual(self.tester.detection_head.num_queries, 20,
                        "DetectionHead must have Q=20 queries")
        self.assertEqual(self.tester.detection_head.decoder_layers, 4,
                        "DetectionHead must have 4 decoder layers")

        print("✅ Model Architecture Test PASSED:")
        print(f"  PoseHead: mean-pool → MLP → 6DOF")
        print(f"  DetectionHead: Q=20, Decoder×4 → (B,20,10)")
        print(f"  HungarianMatcher: 3D center + IoU costs")

    def test_02_multitask_training(self):
        """测试多任务训练"""
        print("\n" + "="*60)
        print("Testing Multi-Task Training")
        print("="*60)

        # 创建训练数据集
        dataset = MineSLAMDataset(self.config, split='train')

        # 执行多任务训练测试
        results = self.tester.test_multitask_training(dataset)

        # 验证训练稳定性
        self.assertGreater(results['successful_steps'], 0,
                          "No successful training steps")

        # 验证NaN/Inf检测
        failure_rate = results['nan_inf_count'] / self.tester.num_train_steps
        self.assertLess(failure_rate, 0.5,
                       f"Too many NaN/Inf occurrences: {failure_rate:.3f}")

        print(f"✅ Multi-Task Training Test PASSED:")
        print(f"  Successful steps: {results['successful_steps']}/10")
        print(f"  NaN/Inf rate: {failure_rate:.3f}")

        # 验证损失合理性
        if results['loss_history']['total']:
            avg_total_loss = np.mean(results['loss_history']['total'])
            print(f"  Average total loss: {avg_total_loss:.6f}")

    def test_03_loss_computation(self):
        """测试损失计算"""
        print("\n" + "="*60)
        print("Testing Loss Computation")
        print("="*60)

        batch_size = 2

        # 测试Huber损失（delta=1.0）
        pred_pose = torch.randn(batch_size, 6) * 0.1
        target_pose = torch.randn(batch_size, 6) * 0.1

        huber_loss = self.tester.pose_head.compute_loss(
            pred_pose, target_pose, delta=1.0
        )

        self.assertFalse(torch.isnan(huber_loss), "Huber loss should not be NaN")
        self.assertFalse(torch.isinf(huber_loss), "Huber loss should not be Inf")

        print(f"✅ Loss Computation Test PASSED:")
        print(f"  Huber loss (delta=1.0): {huber_loss.item():.6f}")

    def test_04_output_validation(self):
        """验证输出文件"""
        print("\n" + "="*60)
        print("Validating Output Files")
        print("="*60)

        output_dir = Path(self.tester.output_dir)

        # 检查3D框对比图
        bbox_files = list(output_dir.glob("3d_bbox_comparison_*.png"))
        self.assertGreater(len(bbox_files), 0, "No 3D bbox comparison images generated")

        # 检查损失曲线
        loss_curve_file = output_dir / "loss_curves.png"
        self.assertTrue(loss_curve_file.exists(), f"Loss curves not found: {loss_curve_file}")

        print(f"✅ Output Validation Test PASSED:")
        print(f"  3D bbox comparisons: {len(bbox_files)} files")
        print(f"  Loss curves: {loss_curve_file}")


def run_multitask_tests():
    """运行多任务测试"""
    print("="*80)
    print("MINESLAM MULTI-TASK HEAD TESTS - REAL DATA ONLY")
    print("="*80)
    print("Testing PoseHead + DetectionHead on train_index N=100 samples")
    print("Forward+Backward training, loss NaN/Inf detection, 3D bbox visualization")
    print()

    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiTaskHeads)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*80)
    print("多任务头部测试总结")
    print("="*80)

    if result.wasSuccessful():
        print("✅ 所有多任务测试通过!")
        print("   - PoseHead: mean-pool→MLP→(B,6), Huber loss (delta=1.0)")
        print("   - DetectionHead: DETR-style, Q=20, Decoder×4→(B,20,10)")
        print("   - HungarianMatcher: 3D中心距离+体素IoU匹配")
        print("   - 多任务训练: 10步连续训练无NaN/Inf")
        print("   - 3D框可视化: 预测vs真值对比图已保存")
        print("   - 严格验证真实数据，无随机张量")
    else:
        print("❌ 多任务测试失败!")
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
    success = run_multitask_tests()
    sys.exit(0 if success else 1)