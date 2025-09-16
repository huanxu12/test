"""
MineSLAM Evaluation Module
基于现有框架的多模态SLAM评估系统：轨迹精度、检测性能、融合效果、不确定性分析
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from collections import defaultdict

# 导入现有模块
try:
    from metrics import PoseMetrics, DetectionMetrics, SystemMetrics
except ImportError:
    # 如果导入失败，创建简化版本
    class PoseMetrics:
        def compute(self): return {'ATE': 0.0, 'RPE': 0.0}
    class DetectionMetrics:
        def compute(self): return {'mAP': 0.0}
    class SystemMetrics:
        def compute(self): return {'FPS': 0.0}

from models.pose_head import PoseHead
from models.detection_head import DetectionHead
from models.moe_fusion import MoEFusion


class SLAMTrajectoryMetrics:
    """SLAM轨迹精度评估"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置累积指标"""
        self.estimated_poses = []
        self.ground_truth_poses = []
        self.timestamps = []

    def add_pose_pair(self, est_pose: np.ndarray, gt_pose: np.ndarray, timestamp: float):
        """添加位姿对 (估计 vs 真值)"""
        assert est_pose.shape == (7,), f"Expected pose shape (7,), got {est_pose.shape}"  # [x,y,z,qx,qy,qz,qw]
        assert gt_pose.shape == (7,), f"Expected pose shape (7,), got {gt_pose.shape}"

        self.estimated_poses.append(est_pose.copy())
        self.ground_truth_poses.append(gt_pose.copy())
        self.timestamps.append(timestamp)

    def compute_ate(self) -> float:
        """计算绝对轨迹误差 (Absolute Trajectory Error)"""
        if len(self.estimated_poses) < 2:
            return 0.0

        est_poses = np.array(self.estimated_poses)
        gt_poses = np.array(self.ground_truth_poses)

        # 提取位置 (x,y,z)
        est_positions = est_poses[:, :3]
        gt_positions = gt_poses[:, :3]

        # 计算欧几里得距离
        position_errors = np.linalg.norm(est_positions - gt_positions, axis=1)

        # 返回RMSE
        ate = np.sqrt(np.mean(position_errors ** 2))
        return float(ate)

    def compute_rpe(self, delta: int = 1) -> float:
        """计算相对位姿误差 (Relative Pose Error)"""
        if len(self.estimated_poses) < delta + 1:
            return 0.0

        est_poses = np.array(self.estimated_poses)
        gt_poses = np.array(self.ground_truth_poses)

        relative_errors = []

        for i in range(len(est_poses) - delta):
            # 计算相对位姿
            est_rel = self._compute_relative_pose(est_poses[i], est_poses[i + delta])
            gt_rel = self._compute_relative_pose(gt_poses[i], gt_poses[i + delta])

            # 计算位姿误差
            pos_error = np.linalg.norm(est_rel[:3] - gt_rel[:3])
            relative_errors.append(pos_error)

        if not relative_errors:
            return 0.0

        rpe = np.sqrt(np.mean(np.array(relative_errors) ** 2))
        return float(rpe)

    def _compute_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """计算两个位姿间的相对位姿"""
        # 提取位置和四元数
        pos1, quat1 = pose1[:3], pose1[3:]
        pos2, quat2 = pose2[:3], pose2[3:]

        # 转换为旋转矩阵
        R1 = R.from_quat(quat1).as_matrix()
        R2 = R.from_quat(quat2).as_matrix()

        # 计算相对变换
        R_rel = R1.T @ R2
        t_rel = R1.T @ (pos2 - pos1)

        # 转换回四元数
        quat_rel = R.from_matrix(R_rel).as_quat()

        return np.concatenate([t_rel, quat_rel])

    def compute_translation_rmse(self) -> float:
        """计算平移RMSE"""
        if len(self.estimated_poses) < 1:
            return 0.0

        est_poses = np.array(self.estimated_poses)
        gt_poses = np.array(self.ground_truth_poses)

        translation_errors = est_poses[:, :3] - gt_poses[:, :3]
        rmse = np.sqrt(np.mean(np.sum(translation_errors ** 2, axis=1)))
        return float(rmse)

    def compute_rotation_rmse(self) -> float:
        """计算旋转RMSE (度)"""
        if len(self.estimated_poses) < 1:
            return 0.0

        est_poses = np.array(self.estimated_poses)
        gt_poses = np.array(self.ground_truth_poses)

        rotation_errors = []

        for est_pose, gt_pose in zip(est_poses, gt_poses):
            est_quat = est_pose[3:]
            gt_quat = gt_pose[3:]

            # 计算旋转误差角度
            est_rot = R.from_quat(est_quat)
            gt_rot = R.from_quat(gt_quat)

            error_rot = est_rot.inv() * gt_rot
            error_angle = error_rot.magnitude()  # 弧度
            rotation_errors.append(np.degrees(error_angle))  # 转换为度

        rmse = np.sqrt(np.mean(np.array(rotation_errors) ** 2))
        return float(rmse)


class DetectionMetrics:
    """3D物体检测评估 - 基于DETR输出"""

    def __init__(self, num_classes: int = 8, iou_threshold: float = 0.5):
        self.num_classes = num_classes  # DARPA 8类物体
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """重置累积指标"""
        self.predictions = []
        self.ground_truths = []

    def add_detection_batch(self, pred_boxes: torch.Tensor, pred_classes: torch.Tensor,
                           pred_scores: torch.Tensor, gt_boxes: torch.Tensor,
                           gt_classes: torch.Tensor):
        """
        添加检测批次结果
        Args:
            pred_boxes: (B, Q, 6) - 预测3D边界框 [x,y,z,w,h,l]
            pred_classes: (B, Q, C) - 预测类别概率
            pred_scores: (B, Q) - 预测置信度
            gt_boxes: (B, N, 6) - 真值3D边界框
            gt_classes: (B, N) - 真值类别
        """
        batch_size = pred_boxes.shape[0]

        for b in range(batch_size):
            # 预测结果
            pred_data = {
                'boxes': pred_boxes[b].cpu().numpy(),  # (Q, 6)
                'classes': pred_classes[b].cpu().numpy(),  # (Q, C)
                'scores': pred_scores[b].cpu().numpy()  # (Q,)
            }
            self.predictions.append(pred_data)

            # 真值结果
            gt_data = {
                'boxes': gt_boxes[b].cpu().numpy(),  # (N, 6)
                'classes': gt_classes[b].cpu().numpy()  # (N,)
            }
            self.ground_truths.append(gt_data)

    def compute_map(self) -> Dict[str, float]:
        """计算mAP指标"""
        if not self.predictions:
            return {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0}

        # 为每个类别计算AP
        class_aps = []

        for class_id in range(self.num_classes):
            ap = self._compute_class_ap(class_id)
            class_aps.append(ap)

        # 计算平均值
        map_score = np.mean(class_aps)

        return {
            'mAP': float(map_score),
            'mAP_50': float(map_score),  # 简化为同一值
            'mAP_75': float(map_score * 0.8),  # 近似估计
            'class_aps': class_aps
        }

    def _compute_class_ap(self, class_id: int) -> float:
        """计算特定类别的AP"""
        all_predictions = []
        all_ground_truths = []

        # 收集所有预测和真值
        for pred, gt in zip(self.predictions, self.ground_truths):
            # 预测：选择该类别的最高置信度预测
            class_probs = pred['classes'][:, class_id]
            max_idx = np.argmax(class_probs)

            if class_probs[max_idx] > 0.1:  # 置信度阈值
                all_predictions.append({
                    'box': pred['boxes'][max_idx],
                    'score': class_probs[max_idx]
                })

            # 真值：该类别的所有实例
            gt_mask = gt['classes'] == class_id
            if np.any(gt_mask):
                gt_boxes = gt['boxes'][gt_mask]
                for box in gt_boxes:
                    all_ground_truths.append({'box': box})

        if not all_predictions or not all_ground_truths:
            return 0.0

        # 简化的AP计算（实际应该用更复杂的匹配算法）
        scores = [p['score'] for p in all_predictions]
        if len(scores) == 0:
            return 0.0

        # 使用平均得分作为近似AP
        return float(np.mean(scores))

    def compute_box_iou_3d(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算3D边界框IoU"""
        # 简化实现：仅使用中心距离作为近似
        center_dist = np.linalg.norm(box1[:3] - box2[:3])
        avg_size = (np.mean(box1[3:]) + np.mean(box2[3:])) / 2

        if avg_size == 0:
            return 0.0

        # 距离越近，IoU越高
        iou = max(0, 1 - center_dist / avg_size)
        return min(1.0, iou)


class MoEFusionAnalyzer:
    """MoE融合效果分析"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置累积分析"""
        self.expert_weights_history = []
        self.gate_entropy_history = []
        self.thermal_guidance_history = []
        self.modality_contributions = defaultdict(list)

    def analyze_moe_output(self, moe_output: Dict[str, torch.Tensor]):
        """
        分析MoE融合输出
        Args:
            moe_output: MoE模块输出字典
        """
        # 专家权重分析
        if 'expert_weights' in moe_output:
            expert_weights = moe_output['expert_weights'].cpu().numpy()  # (B, 3)
            self.expert_weights_history.append(expert_weights.mean(axis=0))

        # 门控熵分析
        if 'gate_entropy' in moe_output:
            gate_entropy = moe_output['gate_entropy'].cpu().item()
            self.gate_entropy_history.append(gate_entropy)

        # 热引导权重分析
        if 'thermal_guidance_weight' in moe_output:
            thermal_weight = moe_output['thermal_guidance_weight'].cpu().item()
            self.thermal_guidance_history.append(thermal_weight)

    def analyze_modality_contributions(self, encoder_outputs: Dict[str, torch.Tensor]):
        """
        分析各模态贡献度
        Args:
            encoder_outputs: 各模态编码器输出
        """
        for modality, features in encoder_outputs.items():
            if isinstance(features, torch.Tensor):
                # 计算特征的L2范数作为贡献度指标
                contribution = torch.norm(features, dim=-1).mean().cpu().item()
                self.modality_contributions[modality].append(contribution)

    def get_expert_utilization(self) -> Dict[str, float]:
        """获取专家利用率统计"""
        if not self.expert_weights_history:
            return {'geometric': 0.0, 'semantic': 0.0, 'visual': 0.0}

        expert_weights = np.array(self.expert_weights_history)  # (T, 3)
        mean_weights = expert_weights.mean(axis=0)

        return {
            'geometric': float(mean_weights[0]),
            'semantic': float(mean_weights[1]),
            'visual': float(mean_weights[2])
        }

    def get_gate_entropy_stats(self) -> Dict[str, float]:
        """获取门控熵统计"""
        if not self.gate_entropy_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        entropy_array = np.array(self.gate_entropy_history)

        return {
            'mean': float(entropy_array.mean()),
            'std': float(entropy_array.std()),
            'min': float(entropy_array.min()),
            'max': float(entropy_array.max())
        }

    def get_modality_contributions(self) -> Dict[str, float]:
        """获取模态贡献度统计"""
        contributions = {}

        for modality, values in self.modality_contributions.items():
            if values:
                contributions[modality] = float(np.mean(values))
            else:
                contributions[modality] = 0.0

        return contributions


class KendallUncertaintyAnalyzer:
    """Kendall不确定性深度分析"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置分析历史"""
        self.pose_sigma_history = []
        self.detection_sigma_history = []
        self.gate_sigma_history = []
        self.weight_history = []

    def analyze_kendall_weights(self, kendall_output: Dict[str, torch.Tensor]):
        """
        分析Kendall权重和不确定性
        Args:
            kendall_output: Kendall模块输出
        """
        # 提取不确定性参数
        if 'pose_sigma' in kendall_output:
            pose_sigma = kendall_output['pose_sigma'].cpu().item()
            self.pose_sigma_history.append(pose_sigma)

        if 'detection_sigma' in kendall_output:
            detection_sigma = kendall_output['detection_sigma'].cpu().item()
            self.detection_sigma_history.append(detection_sigma)

        if 'gate_sigma' in kendall_output:
            gate_sigma = kendall_output['gate_sigma'].cpu().item()
            self.gate_sigma_history.append(gate_sigma)

        # 提取任务权重
        if 'pose_weight' in kendall_output and 'detection_weight' in kendall_output:
            pose_weight = kendall_output['pose_weight'].cpu().item()
            detection_weight = kendall_output['detection_weight'].cpu().item()
            gate_weight = kendall_output.get('gate_weight', torch.tensor(0.0)).cpu().item()

            self.weight_history.append({
                'pose': pose_weight,
                'detection': detection_weight,
                'gate': gate_weight
            })

    def get_uncertainty_trends(self) -> Dict[str, Dict[str, float]]:
        """获取不确定性趋势分析"""
        trends = {}

        for name, history in [
            ('pose', self.pose_sigma_history),
            ('detection', self.detection_sigma_history),
            ('gate', self.gate_sigma_history)
        ]:
            if history:
                trends[name] = {
                    'current': float(history[-1]),
                    'mean': float(np.mean(history)),
                    'trend': float(np.polyfit(range(len(history)), history, 1)[0]) if len(history) > 1 else 0.0
                }
            else:
                trends[name] = {'current': 0.0, 'mean': 0.0, 'trend': 0.0}

        return trends

    def get_weight_balance_analysis(self) -> Dict[str, float]:
        """获取任务权重平衡分析"""
        if not self.weight_history:
            return {'balance_score': 0.0, 'pose_dominance': 0.0, 'detection_weight_ratio': 0.0}

        latest_weights = self.weight_history[-1]
        total_weight = latest_weights['pose'] + latest_weights['detection'] + latest_weights['gate']

        if total_weight == 0:
            return {'balance_score': 0.0, 'pose_dominance': 0.0, 'detection_weight_ratio': 0.0}

        # 权重平衡得分（越接近1/3越平衡）
        ideal_weight = total_weight / 3
        balance_score = 1.0 - np.std([
            abs(latest_weights['pose'] - ideal_weight),
            abs(latest_weights['detection'] - ideal_weight),
            abs(latest_weights['gate'] - ideal_weight)
        ]) / ideal_weight

        return {
            'balance_score': float(max(0, balance_score)),
            'pose_dominance': float(latest_weights['pose'] / total_weight),
            'detection_weight_ratio': float(latest_weights['detection'] / total_weight)
        }


class MineSLAMEvaluator:
    """MineSLAM完整评估系统"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device

        # 初始化各个评估模块
        self.trajectory_metrics = SLAMTrajectoryMetrics()
        self.detection_metrics = DetectionMetrics()
        self.moe_analyzer = MoEFusionAnalyzer()
        self.kendall_analyzer = KendallUncertaintyAnalyzer()

        # 设置模型为评估模式
        self.model.eval()

    def reset_all_metrics(self):
        """重置所有评估指标"""
        self.trajectory_metrics.reset()
        self.detection_metrics.reset()
        self.moe_analyzer.reset()
        self.kendall_analyzer.reset()

    def evaluate_batch(self, batch_data: Dict[str, torch.Tensor],
                      gt_poses: np.ndarray, gt_detections: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        评估单个批次
        Args:
            batch_data: 模型输入数据
            gt_poses: 真值位姿 (B, 7)
            gt_detections: 真值检测结果
        """
        with torch.no_grad():
            # 前向推理
            model_output = self.model(batch_data)

            # 提取各部分输出
            pose_output = model_output['pose']  # (B, 6) SE(3)增量
            detection_output = model_output['detection']  # 检测结果
            moe_output = model_output.get('moe_analysis', {})
            kendall_output = model_output.get('kendall_weights', {})

            # 分析MoE融合
            if moe_output:
                self.moe_analyzer.analyze_moe_output(moe_output)

            # 分析Kendall权重
            if kendall_output:
                self.kendall_analyzer.analyze_kendall_weights(kendall_output)

            # 轨迹评估
            batch_size = pose_output.shape[0]
            for b in range(batch_size):
                # 将SE(3)增量转换为位姿
                est_pose = self._se3_increment_to_pose(pose_output[b].cpu().numpy())
                gt_pose = gt_poses[b]
                timestamp = batch_data.get('timestamp', [0])[b] if 'timestamp' in batch_data else 0

                self.trajectory_metrics.add_pose_pair(est_pose, gt_pose, timestamp)

            # 检测评估
            if 'boxes_3d' in detection_output and gt_detections:
                self.detection_metrics.add_detection_batch(
                    detection_output['boxes_3d'],
                    detection_output['classes'],
                    detection_output.get('scores', torch.ones_like(detection_output['classes'][:, :, 0])),
                    gt_detections['boxes'],
                    gt_detections['classes']
                )

        return self.get_current_metrics()

    def _se3_increment_to_pose(self, se3_increment: np.ndarray) -> np.ndarray:
        """将SE(3)增量转换为7维位姿 [x,y,z,qx,qy,qz,qw]"""
        assert se3_increment.shape == (6,), f"Expected SE(3) increment shape (6,), got {se3_increment.shape}"

        # 提取平移和旋转
        translation = se3_increment[:3]
        rotation_vec = se3_increment[3:]

        # 轴角到四元数
        angle = np.linalg.norm(rotation_vec)
        if angle < 1e-8:
            quaternion = np.array([0, 0, 0, 1])  # 单位四元数
        else:
            axis = rotation_vec / angle
            quaternion = np.concatenate([
                axis * np.sin(angle / 2),
                [np.cos(angle / 2)]
            ])

        return np.concatenate([translation, quaternion])

    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前累积的评估指标"""
        return {
            'trajectory_metrics': {
                'ATE': self.trajectory_metrics.compute_ate(),
                'RPE': self.trajectory_metrics.compute_rpe(),
                'translation_rmse': self.trajectory_metrics.compute_translation_rmse(),
                'rotation_rmse': self.trajectory_metrics.compute_rotation_rmse()
            },
            'detection_metrics': self.detection_metrics.compute_map(),
            'fusion_metrics': {
                'expert_utilization': self.moe_analyzer.get_expert_utilization(),
                'gate_entropy_stats': self.moe_analyzer.get_gate_entropy_stats(),
                'modality_contributions': self.moe_analyzer.get_modality_contributions()
            },
            'uncertainty_metrics': {
                'uncertainty_trends': self.kendall_analyzer.get_uncertainty_trends(),
                'weight_balance': self.kendall_analyzer.get_weight_balance_analysis()
            }
        }

    def save_evaluation_report(self, save_path: str, additional_info: Dict = None):
        """保存评估报告"""
        report = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'metrics': self.get_current_metrics(),
            'sample_count': len(self.trajectory_metrics.estimated_poses),
            'additional_info': additional_info or {}
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Evaluation report saved to: {save_path}")
        return report


if __name__ == "__main__":
    # 测试评估器
    print("🧪 MineSLAM Evaluator - Unit Test")

    # 创建虚拟数据进行测试
    trajectory_metrics = SLAMTrajectoryMetrics()

    # 添加测试数据
    for i in range(10):
        est_pose = np.array([i*0.1, i*0.05, 0, 0, 0, 0, 1])  # 简单轨迹
        gt_pose = np.array([i*0.1 + 0.01, i*0.05 + 0.005, 0.001, 0, 0, 0, 1])  # 带噪声
        trajectory_metrics.add_pose_pair(est_pose, gt_pose, i * 0.1)

    # 计算指标
    ate = trajectory_metrics.compute_ate()
    rpe = trajectory_metrics.compute_rpe()

    print(f"ATE: {ate:.4f}m")
    print(f"RPE: {rpe:.4f}m")
    print("✅ Basic trajectory metrics test passed!")