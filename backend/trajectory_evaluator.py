"""
Trajectory Evaluation and Visualization Module
轨迹评估和可视化模块：ATE计算，优化前后对比，地标可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json


class TrajectoryEvaluator:
    """轨迹评估器 - 计算ATE/RPE等SLAM指标"""

    def __init__(self):
        self.ground_truth: Optional[np.ndarray] = None
        self.estimated_trajectory: Optional[np.ndarray] = None

    def set_ground_truth(self, gt_trajectory: np.ndarray):
        """设置真值轨迹
        Args:
            gt_trajectory: N×7 [timestamp, x, y, z, rx, ry, rz]
        """
        self.ground_truth = gt_trajectory

    def set_estimated_trajectory(self, estimated_trajectory: np.ndarray):
        """设置估计轨迹
        Args:
            estimated_trajectory: N×7 [timestamp, x, y, z, rx, ry, rz]
        """
        self.estimated_trajectory = estimated_trajectory

    def compute_ate(self, align_trajectories: bool = True) -> Dict[str, float]:
        """计算绝对轨迹误差 (Absolute Trajectory Error)"""
        if self.ground_truth is None or self.estimated_trajectory is None:
            raise ValueError("Ground truth and estimated trajectory must be set")

        # 时间戳对齐
        gt_aligned, est_aligned = self._align_timestamps()

        if len(gt_aligned) == 0:
            raise ValueError("No overlapping timestamps found")

        # 位置对齐（如果需要）
        gt_positions = gt_aligned[:, 1:4]  # x, y, z
        est_positions = est_aligned[:, 1:4]

        if align_trajectories:
            # Umeyama对齐（7DoF：尺度、旋转、平移）
            est_positions_aligned = self._umeyama_alignment(est_positions, gt_positions)
        else:
            est_positions_aligned = est_positions

        # 计算ATE
        position_errors = np.linalg.norm(gt_positions - est_positions_aligned, axis=1)

        ate_stats = {
            'ate_mean': np.mean(position_errors),
            'ate_median': np.median(position_errors),
            'ate_std': np.std(position_errors),
            'ate_min': np.min(position_errors),
            'ate_max': np.max(position_errors),
            'ate_rmse': np.sqrt(np.mean(position_errors ** 2)),
            'num_poses': len(position_errors)
        }

        return ate_stats

    def compute_rpe(self, delta: float = 1.0) -> Dict[str, float]:
        """计算相对位姿误差 (Relative Pose Error)"""
        if self.ground_truth is None or self.estimated_trajectory is None:
            raise ValueError("Ground truth and estimated trajectory must be set")

        gt_aligned, est_aligned = self._align_timestamps()

        # 计算相对位姿
        gt_relative_poses = self._compute_relative_poses(gt_aligned[:, 1:7], delta)
        est_relative_poses = self._compute_relative_poses(est_aligned[:, 1:7], delta)

        if len(gt_relative_poses) == 0:
            return {'rpe_mean': float('inf'), 'rpe_median': float('inf')}

        # 计算RPE
        rpe_errors = []
        for gt_rel, est_rel in zip(gt_relative_poses, est_relative_poses):
            # 只计算平移误差
            translation_error = np.linalg.norm(gt_rel[:3] - est_rel[:3])
            rpe_errors.append(translation_error)

        rpe_stats = {
            'rpe_mean': np.mean(rpe_errors),
            'rpe_median': np.median(rpe_errors),
            'rpe_std': np.std(rpe_errors),
            'rpe_rmse': np.sqrt(np.mean(np.array(rpe_errors) ** 2)),
            'num_relative_poses': len(rpe_errors)
        }

        return rpe_stats

    def _align_timestamps(self) -> Tuple[np.ndarray, np.ndarray]:
        """时间戳对齐"""
        gt_times = self.ground_truth[:, 0]
        est_times = self.estimated_trajectory[:, 0]

        # 找到重叠时间范围
        start_time = max(gt_times.min(), est_times.min())
        end_time = min(gt_times.max(), est_times.max())

        # 筛选重叠时间段的数据
        gt_mask = (gt_times >= start_time) & (gt_times <= end_time)
        est_mask = (est_times >= start_time) & (est_times <= end_time)

        gt_filtered = self.ground_truth[gt_mask]
        est_filtered = self.estimated_trajectory[est_mask]

        # 最近邻插值对齐
        aligned_gt = []
        aligned_est = []

        for est_pose in est_filtered:
            est_time = est_pose[0]
            # 找到最近的GT时间戳
            time_diffs = np.abs(gt_filtered[:, 0] - est_time)
            closest_idx = np.argmin(time_diffs)

            if time_diffs[closest_idx] < 0.1:  # 100ms容忍度
                aligned_gt.append(gt_filtered[closest_idx])
                aligned_est.append(est_pose)

        return np.array(aligned_gt), np.array(aligned_est)

    def _umeyama_alignment(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Umeyama算法进行3D点云对齐"""
        # 计算质心
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)

        # 去质心
        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # 计算协方差矩阵
        H = source_centered.T @ target_centered

        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 确保为旋转矩阵
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 计算尺度
        scale = np.sum(S) / np.sum(source_centered ** 2)

        # 计算平移
        t = target_centroid - scale * R @ source_centroid

        # 应用变换
        aligned_source = scale * (source @ R.T) + t

        return aligned_source

    def _compute_relative_poses(self, poses: np.ndarray, delta: float) -> List[np.ndarray]:
        """计算相对位姿"""
        relative_poses = []

        for i in range(len(poses) - 1):
            if poses[i + 1, 0] - poses[i, 0] <= delta + 0.1:  # 时间间隔检查
                # 简化：只计算平移差异
                relative_pose = poses[i + 1] - poses[i]
                relative_poses.append(relative_pose)

        return relative_poses

    def evaluate_optimization_improvement(self, before_trajectory: np.ndarray,
                                        after_trajectory: np.ndarray) -> Dict[str, Any]:
        """评估优化改进效果"""
        # 计算优化前ATE
        self.set_estimated_trajectory(before_trajectory)
        ate_before = self.compute_ate()

        # 计算优化后ATE
        self.set_estimated_trajectory(after_trajectory)
        ate_after = self.compute_ate()

        # 计算改进
        ate_improvement = (ate_before['ate_rmse'] - ate_after['ate_rmse']) / ate_before['ate_rmse']

        return {
            'ate_before': ate_before,
            'ate_after': ate_after,
            'ate_improvement_percentage': ate_improvement * 100,
            'improvement_achieved': ate_improvement > 0.1  # 10%改进阈值
        }


class TrajectoryVisualizer:
    """轨迹可视化器"""

    def __init__(self, output_dir: str = "outputs/backend"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_trajectory_comparison(self, ground_truth: np.ndarray,
                                 before_optimization: np.ndarray,
                                 after_optimization: np.ndarray,
                                 save_path: Optional[str] = None) -> str:
        """绘制轨迹对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 2D轨迹对比 (XY平面)
        ax = axes[0, 0]
        ax.plot(ground_truth[:, 1], ground_truth[:, 2], 'g-', linewidth=2, label='Ground Truth')
        ax.plot(before_optimization[:, 1], before_optimization[:, 2], 'r--', linewidth=1.5, label='Before Optimization')
        ax.plot(after_optimization[:, 1], after_optimization[:, 2], 'b-', linewidth=1.5, label='After Optimization')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Comparison (XY Plane)')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # 3D轨迹对比
        ax = axes[0, 1]
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.plot(ground_truth[:, 1], ground_truth[:, 2], ground_truth[:, 3], 'g-', linewidth=2, label='Ground Truth')
        ax.plot(before_optimization[:, 1], before_optimization[:, 2], before_optimization[:, 3], 'r--', linewidth=1.5, label='Before')
        ax.plot(after_optimization[:, 1], after_optimization[:, 2], after_optimization[:, 3], 'b-', linewidth=1.5, label='After')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory Comparison')
        ax.legend()

        # 误差分析
        ax = axes[1, 0]
        times = before_optimization[:, 0]
        errors_before = np.linalg.norm(ground_truth[:, 1:4] - before_optimization[:, 1:4], axis=1)
        errors_after = np.linalg.norm(ground_truth[:, 1:4] - after_optimization[:, 1:4], axis=1)

        ax.plot(times, errors_before, 'r-', label='Before Optimization', alpha=0.7)
        ax.plot(times, errors_after, 'b-', label='After Optimization', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Position Error vs Time')
        ax.legend()
        ax.grid(True)

        # 误差分布直方图
        ax = axes[1, 1]
        ax.hist(errors_before, bins=30, alpha=0.7, label='Before', color='red', density=True)
        ax.hist(errors_after, bins=30, alpha=0.7, label='After', color='blue', density=True)
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        # 保存图像
        if save_path is None:
            save_path = self.output_dir / 'trajectory_comparison.png'
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def plot_landmarks_3d(self, landmarks: Dict[int, np.ndarray],
                         trajectory: np.ndarray,
                         save_path: Optional[str] = None) -> str:
        """绘制3D地标和轨迹"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制轨迹
        ax.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], 'b-', linewidth=2, label='Trajectory')

        # 绘制地标
        if landmarks:
            landmark_positions = np.array(list(landmarks.values()))
            ax.scatter(landmark_positions[:, 0], landmark_positions[:, 1], landmark_positions[:, 2],
                      c='red', s=50, marker='o', label='Landmarks')

            # 添加地标ID标签
            for landmark_id, position in landmarks.items():
                ax.text(position[0], position[1], position[2], f'L{landmark_id}', fontsize=8)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory and Landmarks')
        ax.legend()

        # 保存图像
        if save_path is None:
            save_path = self.output_dir / 'landmarks_3d.png'
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def plot_pose_uncertainties(self, trajectory: np.ndarray,
                               uncertainties: Dict[int, np.ndarray],
                               save_path: Optional[str] = None) -> str:
        """绘制位姿不确定性椭圆"""
        fig, ax = plt.subplots(figsize=(12, 9))

        # 绘制轨迹
        ax.plot(trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, label='Trajectory')

        # 绘制不确定性椭圆
        for i, (pose_id, uncertainty) in enumerate(uncertainties.items()):
            if i < len(trajectory):
                x, y = trajectory[i, 1], trajectory[i, 2]

                # 使用位置不确定性 (σx, σy)
                if len(uncertainty) >= 2:
                    width = uncertainty[0] * 2  # 2σ椭圆
                    height = uncertainty[1] * 2

                    ellipse = Ellipse((x, y), width, height, alpha=0.3, color='red')
                    ax.add_patch(ellipse)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory with Pose Uncertainties')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # 保存图像
        if save_path is None:
            save_path = self.output_dir / 'pose_uncertainties.png'
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(save_path)

    def save_evaluation_report(self, evaluation_results: Dict[str, Any],
                             optimization_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
        """保存评估报告"""
        if save_path is None:
            save_path = self.output_dir / 'evaluation_report.json'
        else:
            save_path = Path(save_path)

        report = {
            'evaluation_results': evaluation_results,
            'optimization_results': optimization_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(save_path)