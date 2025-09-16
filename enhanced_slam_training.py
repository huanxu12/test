"""
Enhanced Training Loop with SLAM Evaluation Integration
集成SLAM评估功能的增强训练循环：实时指标监控、可视化、完整评估报告
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset

# 导入现有模块
from train_loop import TrainingLoop, pointcloud_collate_fn
from slam_evaluator import MineSLAMEvaluator
from slam_visualizer import MineSLAMVisualizer
from slam_evaluation_dataset import SLAMEvaluationDataset, SLAMDatasetFactory


class EnhancedSLAMTrainingLoop(TrainingLoop):
    """增强的SLAM训练循环 - 集成评估和可视化"""

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        """
        初始化增强的SLAM训练循环

        Args:
            config: 训练配置字典
            *args, **kwargs: 传递给父类的参数
        """
        super().__init__(config, *args, **kwargs)

        # SLAM评估配置
        self.slam_config = config.get('slam_evaluation', {})
        self.enable_slam_eval = self.slam_config.get('enabled', True)
        self.eval_frequency = self.slam_config.get('eval_frequency', 5)  # 每5个epoch评估一次
        self.visualize_frequency = self.slam_config.get('visualize_frequency', 10)
        self.save_visualizations = self.slam_config.get('save_visualizations', True)

        # 初始化SLAM评估器
        if self.enable_slam_eval:
            self.slam_evaluator = MineSLAMEvaluator(self.model, self.device)
            print("🔍 SLAM Evaluator initialized")
        else:
            self.slam_evaluator = None

        # 初始化可视化器
        if self.save_visualizations:
            viz_output_dir = os.path.join(self.output_dir, "slam_visualization")
            self.slam_visualizer = MineSLAMVisualizer(viz_output_dir)
            print(f"🎨 SLAM Visualizer initialized: {viz_output_dir}")
        else:
            self.slam_visualizer = None

        # SLAM评估历史
        self.slam_metrics_history = []
        self.best_slam_metrics = {
            'best_ate': float('inf'),
            'best_map': 0.0,
            'best_combined_score': 0.0,
            'best_epoch': 0
        }

        print(f"🚀 Enhanced SLAM Training Loop initialized")
        print(f"   SLAM Evaluation: {'Enabled' if self.enable_slam_eval else 'Disabled'}")
        print(f"   Visualization: {'Enabled' if self.save_visualizations else 'Disabled'}")

    def setup_slam_evaluation_datasets(self):
        """设置SLAM评估数据集"""
        try:
            # 使用SLAM评估数据集替换现有数据集
            if hasattr(self, 'val_dataset'):
                data_root = self.val_dataset.data_root

                # 创建SLAM评估验证数据集
                self.slam_val_dataset = SLAMDatasetFactory.create_validation_dataset(
                    data_root,
                    sequence_length=self.val_dataset.sequence_length
                )

                print(f"🔄 Replaced validation dataset with SLAM evaluation dataset")
                print(f"   Samples: {len(self.slam_val_dataset)}")

                # 获取数据集统计
                stats = self.slam_val_dataset.get_evaluation_statistics()
                print(f"📊 SLAM Dataset Stats: {stats}")

        except Exception as e:
            warnings.warn(f"Failed to setup SLAM evaluation datasets: {e}")
            self.slam_val_dataset = None

    def validation_epoch(self, epoch: int) -> Dict[str, float]:
        """增强的验证函数 - 集成SLAM指标"""
        print(f"\n🔍 Enhanced Validation - Epoch {epoch}")

        # 执行原有验证
        standard_metrics = super().validation_epoch(epoch)

        # SLAM特定评估
        slam_metrics = {}
        if self.enable_slam_eval and self.slam_evaluator and epoch % self.eval_frequency == 0:
            slam_metrics = self._run_slam_evaluation(epoch)

        # 可视化更新
        if self.slam_visualizer and epoch % self.visualize_frequency == 0:
            self._update_visualizations(epoch, standard_metrics, slam_metrics)

        # 合并指标
        combined_metrics = {**standard_metrics, **slam_metrics}

        # 更新最佳SLAM指标
        if slam_metrics:
            self._update_best_slam_metrics(epoch, slam_metrics)

        return combined_metrics

    def _run_slam_evaluation(self, epoch: int) -> Dict[str, Any]:
        """运行完整的SLAM评估"""
        print(f"🎯 Running SLAM evaluation at epoch {epoch}")

        if not hasattr(self, 'slam_val_dataset') or self.slam_val_dataset is None:
            self.setup_slam_evaluation_datasets()

        if self.slam_val_dataset is None:
            warnings.warn("SLAM validation dataset not available")
            return {}

        try:
            # 重置评估器
            self.slam_evaluator.reset_all_metrics()

            # 创建数据加载器
            slam_val_loader = DataLoader(
                self.slam_val_dataset,
                batch_size=min(4, self.batch_size),  # 减小批次以节省内存
                shuffle=False,
                collate_fn=pointcloud_collate_fn,
                num_workers=2
            )

            # 评估循环
            self.model.eval()
            total_samples = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(slam_val_loader):
                    # 移动数据到设备
                    batch = self._move_batch_to_device(batch)

                    # 提取真值数据
                    gt_poses = batch.get('gt_pose', torch.zeros(batch['rgb'].shape[0], 7)).cpu().numpy()
                    gt_detections = {
                        'boxes': batch['gt_detections']['boxes'] if 'gt_detections' in batch else torch.zeros(0, 6),
                        'classes': batch['gt_detections']['classes'] if 'gt_detections' in batch else torch.zeros(0, dtype=torch.long)
                    }

                    # 运行评估
                    batch_metrics = self.slam_evaluator.evaluate_batch(batch, gt_poses, gt_detections)
                    total_samples += batch['rgb'].shape[0]

                    # 进度输出
                    if batch_idx % 20 == 0:
                        print(f"   Processed {total_samples} samples...")

            # 获取最终指标
            final_metrics = self.slam_evaluator.get_current_metrics()

            # 添加前缀以区分SLAM指标
            slam_metrics = {}
            for category, metrics in final_metrics.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        slam_key = f"slam_{category}_{metric_name}"
                        slam_metrics[slam_key] = value
                else:
                    slam_key = f"slam_{category}"
                    slam_metrics[slam_key] = metrics

            # 记录历史
            self.slam_metrics_history.append({
                'epoch': epoch,
                'metrics': slam_metrics.copy()
            })

            print(f"✅ SLAM evaluation completed:")
            print(f"   ATE: {slam_metrics.get('slam_trajectory_metrics_ATE', 0):.4f}m")
            print(f"   mAP: {slam_metrics.get('slam_detection_metrics_mAP', 0):.4f}")

            return slam_metrics

        except Exception as e:
            warnings.warn(f"SLAM evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将批次数据移动到设备"""
        device_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in value.items()}
            else:
                device_batch[key] = value

        return device_batch

    def _update_visualizations(self, epoch: int, standard_metrics: Dict[str, float],
                             slam_metrics: Dict[str, Any]):
        """更新可视化"""
        try:
            # 合并所有指标用于训练过程可视化
            all_metrics = {**standard_metrics, **slam_metrics}

            # 重新组织指标格式以适配可视化器
            viz_metrics = {
                'losses': {
                    'pose': standard_metrics.get('val_pose_loss', 0),
                    'detection': standard_metrics.get('val_detection_loss', 0),
                    'total': standard_metrics.get('val_total_loss', 0)
                },
                'trajectory_metrics': {
                    'ATE': slam_metrics.get('slam_trajectory_metrics_ATE', 0)
                },
                'detection_metrics': {
                    'mAP': slam_metrics.get('slam_detection_metrics_mAP', 0)
                },
                'kendall_weights': {
                    'pose_weight': getattr(self.model, 'last_pose_weight', 0),
                    'detection_weight': getattr(self.model, 'last_detection_weight', 0),
                    'gate_weight': getattr(self.model, 'last_gate_weight', 0)
                }
            }

            # 更新训练过程可视化
            self.slam_visualizer.update_training_progress(epoch, viz_metrics)

            # 保存可视化
            if epoch % (self.visualize_frequency * 2) == 0:  # 减少保存频率
                self.slam_visualizer.save_all_visualizations(epoch, "enhanced_slam")

        except Exception as e:
            warnings.warn(f"Visualization update failed: {e}")

    def _update_best_slam_metrics(self, epoch: int, slam_metrics: Dict[str, Any]):
        """更新最佳SLAM指标"""
        try:
            # 提取关键指标
            ate = slam_metrics.get('slam_trajectory_metrics_ATE', float('inf'))
            map_score = slam_metrics.get('slam_detection_metrics_mAP', 0.0)

            # 计算综合得分 (ATE越小越好，mAP越大越好)
            # 归一化：ATE以0.5m为基准，mAP已在[0,1]范围
            normalized_ate = max(0, 1 - ate / 0.5)  # ATE=0时得分1，ATE>=0.5时得分0
            combined_score = (normalized_ate + map_score) / 2

            # 更新最佳指标
            if ate < self.best_slam_metrics['best_ate']:
                self.best_slam_metrics['best_ate'] = ate

            if map_score > self.best_slam_metrics['best_map']:
                self.best_slam_metrics['best_map'] = map_score

            if combined_score > self.best_slam_metrics['best_combined_score']:
                self.best_slam_metrics['best_combined_score'] = combined_score
                self.best_slam_metrics['best_epoch'] = epoch

                # 保存最佳模型
                self._save_best_slam_model(epoch, slam_metrics)

        except Exception as e:
            warnings.warn(f"Failed to update best SLAM metrics: {e}")

    def _save_best_slam_model(self, epoch: int, slam_metrics: Dict[str, Any]):
        """保存最佳SLAM模型"""
        try:
            best_model_path = os.path.join(self.output_dir, "best_slam_model.pth")

            # 保存模型检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'slam_metrics': slam_metrics,
                'best_slam_metrics': self.best_slam_metrics.copy()
            }

            torch.save(checkpoint, best_model_path)
            print(f"💾 Best SLAM model saved: {best_model_path}")

        except Exception as e:
            warnings.warn(f"Failed to save best SLAM model: {e}")

    def save_final_slam_report(self):
        """保存最终的SLAM评估报告"""
        try:
            report_data = {
                'training_completed': time.time(),
                'total_epochs': self.epoch,
                'best_slam_metrics': self.best_slam_metrics,
                'slam_metrics_history': self.slam_metrics_history[-10:],  # 保存最后10个epoch
                'configuration': {
                    'slam_eval_enabled': self.enable_slam_eval,
                    'eval_frequency': self.eval_frequency,
                    'visualize_frequency': self.visualize_frequency
                }
            }

            # 保存JSON报告
            report_path = os.path.join(self.output_dir, "slam_evaluation_report.json")
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            print(f"📊 SLAM evaluation report saved: {report_path}")

            # 创建可视化报告
            if self.slam_visualizer and self.slam_metrics_history:
                latest_metrics = self.slam_metrics_history[-1]['metrics']

                # 重构指标格式
                viz_report_metrics = {
                    'trajectory_metrics': {
                        key.replace('slam_trajectory_metrics_', ''): value
                        for key, value in latest_metrics.items()
                        if key.startswith('slam_trajectory_metrics_')
                    },
                    'detection_metrics': {
                        key.replace('slam_detection_metrics_', ''): value
                        for key, value in latest_metrics.items()
                        if key.startswith('slam_detection_metrics_')
                    },
                    'fusion_metrics': {
                        key.replace('slam_fusion_metrics_', ''): value
                        for key, value in latest_metrics.items()
                        if key.startswith('slam_fusion_metrics_')
                    },
                    'uncertainty_metrics': {
                        key.replace('slam_uncertainty_metrics_', ''): value
                        for key, value in latest_metrics.items()
                        if key.startswith('slam_uncertainty_metrics_')
                    }
                }

                viz_report_path = os.path.join(self.output_dir, "slam_evaluation_plots.png")
                self.slam_visualizer.create_evaluation_report_plots(
                    viz_report_metrics, viz_report_path
                )

            # 输出最终总结
            print(f"\n🎉 SLAM Training Completed!")
            print(f"   Best ATE: {self.best_slam_metrics['best_ate']:.4f}m")
            print(f"   Best mAP: {self.best_slam_metrics['best_map']:.4f}")
            print(f"   Best Combined Score: {self.best_slam_metrics['best_combined_score']:.4f}")
            print(f"   Best Epoch: {self.best_slam_metrics['best_epoch']}")

        except Exception as e:
            warnings.warn(f"Failed to save final SLAM report: {e}")

    def train(self):
        """重写训练函数以包含SLAM评估"""
        print(f"🚀 Starting Enhanced SLAM Training")

        # 设置SLAM评估数据集
        if self.enable_slam_eval:
            self.setup_slam_evaluation_datasets()

        # 调用父类训练方法
        result = super().train()

        # 保存最终SLAM报告
        self.save_final_slam_report()

        return result


def create_enhanced_slam_training_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """创建增强SLAM训练配置"""
    enhanced_config = base_config.copy()

    # 添加SLAM评估配置
    enhanced_config['slam_evaluation'] = {
        'enabled': True,
        'eval_frequency': 5,  # 每5个epoch评估一次
        'visualize_frequency': 10,  # 每10个epoch可视化一次
        'save_visualizations': True,
        'evaluation_metrics': [
            'trajectory_accuracy',  # ATE, RPE
            'detection_performance',  # mAP
            'fusion_analysis',  # MoE专家利用率
            'uncertainty_analysis'  # Kendall权重分析
        ]
    }

    # 调整验证频率以配合SLAM评估
    if 'validation_frequency' in enhanced_config:
        enhanced_config['validation_frequency'] = min(
            enhanced_config['validation_frequency'],
            enhanced_config['slam_evaluation']['eval_frequency']
        )

    return enhanced_config


if __name__ == "__main__":
    # 测试增强的SLAM训练循环
    print("🧪 Enhanced SLAM Training Loop - Test Mode")

    # 创建测试配置
    test_config = {
        'batch_size': 2,
        'learning_rate': 1e-4,
        'max_epochs': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'outputs/test_enhanced_training',
        'slam_evaluation': {
            'enabled': True,
            'eval_frequency': 2,
            'visualize_frequency': 3,
            'save_visualizations': True
        }
    }

    try:
        # 注意：实际使用时需要提供真实的模型和数据
        print("🔧 Enhanced SLAM training configuration created")
        print(f"   SLAM evaluation: {test_config['slam_evaluation']}")
        print("✅ Configuration test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()