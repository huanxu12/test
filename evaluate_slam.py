"""
Complete SLAM Evaluation Script
完整的SLAM评估脚本：基于训练好的模型进行全面的多模态SLAM评估
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings

# 导入评估模块
from slam_evaluator import MineSLAMEvaluator
from slam_visualizer import MineSLAMVisualizer
from slam_evaluation_dataset import SLAMEvaluationDataset, SLAMDatasetFactory
from train_loop import pointcloud_collate_fn

# 导入模型组件
from models.encoders import MultiModalEncoder
from models.moe_fusion import MoEFusion, MoEConfig
from models.pose_head import PoseHead
from models.detection_head import DetectionHead
from models.kendall_uncertainty import create_kendall_uncertainty


class MineSLAMModel(nn.Module):
    """完整的MineSLAM模型 - 用于评估"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # 模型组件
        self.encoder = MultiModalEncoder()

        moe_config = MoEConfig(**config.get('moe_config', {}))
        self.moe_fusion = MoEFusion(moe_config)

        self.pose_head = PoseHead()
        self.detection_head = DetectionHead()

        # Kendall不确定性
        self.kendall_uncertainty = create_kendall_uncertainty()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向推理"""
        # 编码各模态
        encoded_features = self.encoder(batch)

        # MoE融合
        fused_features, moe_output = self.moe_fusion(encoded_features, batch)

        # 任务头预测
        pose_output = self.pose_head(fused_features)
        detection_output = self.detection_head(fused_features)

        # Kendall权重
        kendall_weights = self.kendall_uncertainty.get_weights()

        return {
            'pose': pose_output,
            'detection': detection_output,
            'moe_analysis': moe_output,
            'kendall_weights': kendall_weights,
            'encoded_features': encoded_features,
            'fused_features': fused_features
        }


def load_trained_model(checkpoint_path: str, device: str = 'cuda') -> MineSLAMModel:
    """加载训练好的模型"""
    print(f"📂 Loading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 创建模型配置
    model_config = {
        'moe_config': {
            'embedding_dim': 512,
            'num_experts': 3,
            'thermal_guidance': True
        }
    }

    # 创建模型
    model = MineSLAMModel(model_config)

    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 直接加载模型状态
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"   Trained epochs: {checkpoint['epoch']}")
    if 'best_slam_metrics' in checkpoint:
        print(f"   Best metrics: {checkpoint['best_slam_metrics']}")

    return model


def run_comprehensive_slam_evaluation(
    model: MineSLAMModel,
    dataset: SLAMEvaluationDataset,
    device: str = 'cuda',
    batch_size: int = 4,
    output_dir: str = "outputs/slam_evaluation"
) -> Dict[str, Any]:
    """运行综合SLAM评估"""

    print(f"🎯 Starting comprehensive SLAM evaluation")
    print(f"   Dataset: {len(dataset)} samples")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 初始化评估器和可视化器
    evaluator = MineSLAMEvaluator(model, device)
    visualizer = MineSLAMVisualizer(os.path.join(output_dir, "visualizations"))

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pointcloud_collate_fn,
        num_workers=4
    )

    # 评估循环
    total_samples = 0
    start_time = time.time()

    print(f"\n🔄 Processing evaluation batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到设备
            device_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(device)
                elif isinstance(value, dict):
                    device_batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                       for k, v in value.items()}
                else:
                    device_batch[key] = value

            # 前向推理
            model_output = model(device_batch)

            # 提取真值数据
            gt_poses = batch.get('gt_pose', torch.zeros(device_batch['rgb'].shape[0], 7)).cpu().numpy()
            gt_detections = {
                'boxes': batch['gt_detections']['boxes'] if 'gt_detections' in batch else torch.zeros(0, 6),
                'classes': batch['gt_detections']['classes'] if 'gt_detections' in batch else torch.zeros(0, dtype=torch.long)
            }

            # 运行批次评估
            batch_metrics = evaluator.evaluate_batch(device_batch, gt_poses, gt_detections)

            # 更新可视化（采样部分批次以避免过多输出）
            if batch_idx % 20 == 0 and batch_idx < 100:  # 前5个批次进行可视化
                try:
                    # 准备传感器数据
                    sensor_data = {
                        'rgb': device_batch['rgb'][0].cpu().numpy().transpose(1, 2, 0),
                        'depth': device_batch.get('depth', torch.zeros_like(device_batch['rgb'][:1, :1]))[0, 0].cpu().numpy(),
                        'thermal': device_batch.get('thermal', torch.zeros_like(device_batch['rgb'][:1, :1]))[0, 0].cpu().numpy(),
                        'timestamp': device_batch.get('timestamp', [batch_idx])[0] if 'timestamp' in device_batch else batch_idx
                    }

                    # 准备模型输出
                    viz_model_output = {
                        'expert_weights': model_output.get('moe_analysis', {}).get('expert_weights', torch.zeros(1, 3))[:1],
                        'gate_entropy': model_output.get('moe_analysis', {}).get('gate_entropy', torch.tensor(0.0)),
                        'thermal_guidance_weight': model_output.get('moe_analysis', {}).get('thermal_guidance_weight', torch.tensor(0.0))
                    }

                    # 更新实时可视化
                    visualizer.update_realtime_data(sensor_data, viz_model_output)
                    visualizer.update_moe_analysis(viz_model_output)

                except Exception as e:
                    warnings.warn(f"Visualization update failed for batch {batch_idx}: {e}")

            total_samples += device_batch['rgb'].shape[0]

            # 进度输出
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                print(f"   Batch {batch_idx:4d}: {total_samples:5d} samples processed ({samples_per_sec:.1f} samples/s)")

    # 获取最终评估结果
    final_metrics = evaluator.get_current_metrics()

    # 保存评估报告
    evaluation_time = time.time() - start_time
    report_path = os.path.join(output_dir, "comprehensive_evaluation_report.json")

    full_report = {
        'evaluation_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': total_samples,
            'evaluation_time_seconds': evaluation_time,
            'samples_per_second': total_samples / evaluation_time,
            'dataset_statistics': dataset.get_evaluation_statistics()
        },
        'metrics': final_metrics,
        'configuration': {
            'batch_size': batch_size,
            'device': device,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
    }

    evaluator.save_evaluation_report(report_path, full_report['evaluation_info'])

    # 创建可视化报告
    viz_report_path = os.path.join(output_dir, "evaluation_visualization_report.png")
    visualizer.create_evaluation_report_plots(final_metrics, viz_report_path)

    # 保存最终可视化
    visualizer.save_all_visualizations(0, "final_evaluation")

    print(f"\n✅ Comprehensive SLAM evaluation completed!")
    print(f"   Total time: {evaluation_time:.1f}s")
    print(f"   Processing speed: {total_samples/evaluation_time:.1f} samples/s")
    print(f"   Report saved: {report_path}")

    return full_report


def main():
    """主评估函数"""
    parser = argparse.ArgumentParser(description="Comprehensive MineSLAM Evaluation")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='sr_B_route2_deep_learning',
                       help='Path to dataset root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Evaluation batch size')
    parser.add_argument('--output_dir', type=str, default='outputs/comprehensive_slam_evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Limit number of samples to evaluate (for testing)')

    args = parser.parse_args()

    print(f"🚀 MineSLAM Comprehensive Evaluation")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data root: {args.data_root}")
    print(f"   Split: {args.split}")
    print(f"   Output: {args.output_dir}")

    try:
        # 加载模型
        model = load_trained_model(args.checkpoint, args.device)

        # 创建评估数据集
        eval_dataset = SLAMDatasetFactory.create_evaluation_dataset(
            args.data_root,
            split=args.split,
            load_ground_truth=True
        )

        # 限制样本数量（用于测试）
        if args.num_samples is not None:
            from torch.utils.data import Subset
            indices = list(range(min(args.num_samples, len(eval_dataset))))
            eval_dataset = Subset(eval_dataset, indices)
            print(f"⚠️ Limited to {len(eval_dataset)} samples for testing")

        # 运行评估
        evaluation_report = run_comprehensive_slam_evaluation(
            model=model,
            dataset=eval_dataset,
            device=args.device,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )

        # 输出关键指标
        metrics = evaluation_report['metrics']
        print(f"\n📊 Key Evaluation Results:")

        if 'trajectory_metrics' in metrics:
            traj = metrics['trajectory_metrics']
            print(f"   🎯 Trajectory Accuracy:")
            print(f"      ATE: {traj.get('ATE', 0):.4f}m")
            print(f"      RPE: {traj.get('RPE', 0):.4f}m")
            print(f"      Translation RMSE: {traj.get('translation_rmse', 0):.4f}m")
            print(f"      Rotation RMSE: {traj.get('rotation_rmse', 0):.4f}°")

        if 'detection_metrics' in metrics:
            det = metrics['detection_metrics']
            print(f"   🏷️ Detection Performance:")
            print(f"      mAP: {det.get('mAP', 0):.4f}")

        if 'fusion_metrics' in metrics:
            fusion = metrics['fusion_metrics']
            print(f"   🔄 Fusion Analysis:")
            expert_util = fusion.get('expert_utilization', {})
            for expert, util in expert_util.items():
                print(f"      {expert.capitalize()} Expert: {util:.3f}")

        print(f"\n🎉 Evaluation completed successfully!")
        print(f"   Detailed report: {args.output_dir}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())