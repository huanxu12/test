"""
MineSLAM Visualization System
基于现有框架的多模态SLAM可视化：实时监控、3D地图、MoE分析、训练过程可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict, deque
import warnings

# 尝试导入Open3D用于3D可视化
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("Open3D not available, 3D visualization will be limited")


class RealTimeSensorDashboard:
    """实时传感器数据监控面板"""

    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        self.figsize = figsize
        self.setup_dashboard()
        self.data_history = {
            'timestamps': deque(maxlen=100),
            'imu_accel': deque(maxlen=100),
            'imu_gyro': deque(maxlen=100),
            'trajectory_est': deque(maxlen=100),
            'trajectory_gt': deque(maxlen=100)
        }

    def setup_dashboard(self):
        """设置监控面板布局"""
        self.fig = plt.figure(figsize=self.figsize)

        # 创建子图网格 (3x4)
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # RGB + 检测框预览
        self.ax_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_rgb.set_title('RGB + Detection')
        self.ax_rgb.axis('off')

        # 深度图热力图
        self.ax_depth = self.fig.add_subplot(gs[0, 1])
        self.ax_depth.set_title('Depth Heatmap')
        self.ax_depth.axis('off')

        # 热成像叠加
        self.ax_thermal = self.fig.add_subplot(gs[0, 2])
        self.ax_thermal.set_title('Thermal Overlay')
        self.ax_thermal.axis('off')

        # 点云3D显示
        self.ax_pointcloud = self.fig.add_subplot(gs[0, 3], projection='3d')
        self.ax_pointcloud.set_title('LiDAR PointCloud')

        # IMU加速度曲线
        self.ax_imu_accel = self.fig.add_subplot(gs[1, 0:2])
        self.ax_imu_accel.set_title('IMU Acceleration')
        self.ax_imu_accel.set_xlabel('Time (s)')
        self.ax_imu_accel.set_ylabel('Acceleration (m/s²)')

        # IMU角速度曲线
        self.ax_imu_gyro = self.fig.add_subplot(gs[1, 2:4])
        self.ax_imu_gyro.set_title('IMU Angular Velocity')
        self.ax_imu_gyro.set_xlabel('Time (s)')
        self.ax_imu_gyro.set_ylabel('Angular Velocity (rad/s)')

        # 轨迹对比 (估计 vs 真值)
        self.ax_trajectory = self.fig.add_subplot(gs[2, 0:2])
        self.ax_trajectory.set_title('Trajectory: Estimated vs Ground Truth')
        self.ax_trajectory.set_xlabel('X (m)')
        self.ax_trajectory.set_ylabel('Y (m)')

        # 有效像素统计
        self.ax_pixel_stats = self.fig.add_subplot(gs[2, 2])
        self.ax_pixel_stats.set_title('Valid Pixel Statistics')

        # 温度分布
        self.ax_temp_dist = self.fig.add_subplot(gs[2, 3])
        self.ax_temp_dist.set_title('Temperature Distribution')

    def update_rgb_detection(self, rgb_image: np.ndarray, detections: List[Dict] = None):
        """更新RGB图像 + 检测框显示"""
        self.ax_rgb.clear()
        self.ax_rgb.imshow(rgb_image)
        self.ax_rgb.set_title('RGB + Detection')
        self.ax_rgb.axis('off')

        # 绘制检测框
        if detections:
            for det in detections:
                if 'bbox_2d' in det:  # 2D边界框
                    x, y, w, h = det['bbox_2d']
                    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                           edgecolor='red', facecolor='none')
                    self.ax_rgb.add_patch(rect)

                    # 添加标签
                    if 'class_name' in det:
                        self.ax_rgb.text(x, y-5, det['class_name'],
                                       color='red', fontsize=8, fontweight='bold')

    def update_depth_heatmap(self, depth_image: np.ndarray):
        """更新深度图热力图"""
        self.ax_depth.clear()

        # 创建热力图
        im = self.ax_depth.imshow(depth_image, cmap='viridis', aspect='auto')
        self.ax_depth.set_title('Depth Heatmap')
        self.ax_depth.axis('off')

        # 添加colorbar
        if not hasattr(self, 'depth_colorbar'):
            self.depth_colorbar = plt.colorbar(im, ax=self.ax_depth, shrink=0.8)
            self.depth_colorbar.set_label('Depth (m)')

    def update_thermal_overlay(self, thermal_image: np.ndarray, rgb_image: np.ndarray = None):
        """更新热成像叠加显示"""
        self.ax_thermal.clear()

        if rgb_image is not None:
            # RGB作为背景
            self.ax_thermal.imshow(rgb_image, alpha=0.7)
            # 热成像叠加
            self.ax_thermal.imshow(thermal_image, cmap='hot', alpha=0.5)
        else:
            self.ax_thermal.imshow(thermal_image, cmap='hot')

        self.ax_thermal.set_title('Thermal Overlay')
        self.ax_thermal.axis('off')

    def update_pointcloud_3d(self, points: np.ndarray, colors: np.ndarray = None):
        """更新LiDAR点云3D显示"""
        self.ax_pointcloud.clear()

        if points.shape[0] > 0:
            # 降采样以提高性能
            if points.shape[0] > 1000:
                indices = np.random.choice(points.shape[0], 1000, replace=False)
                points = points[indices]
                if colors is not None:
                    colors = colors[indices]

            if colors is not None:
                scatter = self.ax_pointcloud.scatter(points[:, 0], points[:, 1], points[:, 2],
                                                   c=colors, s=1, alpha=0.6)
            else:
                scatter = self.ax_pointcloud.scatter(points[:, 0], points[:, 1], points[:, 2],
                                                   c=points[:, 2], cmap='viridis', s=1, alpha=0.6)

        self.ax_pointcloud.set_title('LiDAR PointCloud')
        self.ax_pointcloud.set_xlabel('X (m)')
        self.ax_pointcloud.set_ylabel('Y (m)')
        self.ax_pointcloud.set_zlabel('Z (m)')

    def update_imu_data(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        """更新IMU数据曲线"""
        self.data_history['timestamps'].append(timestamp)
        self.data_history['imu_accel'].append(accel)
        self.data_history['imu_gyro'].append(gyro)

        if len(self.data_history['timestamps']) > 1:
            timestamps = list(self.data_history['timestamps'])

            # 加速度曲线
            self.ax_imu_accel.clear()
            accel_data = np.array(list(self.data_history['imu_accel']))
            self.ax_imu_accel.plot(timestamps, accel_data[:, 0], 'r-', label='X', alpha=0.8)
            self.ax_imu_accel.plot(timestamps, accel_data[:, 1], 'g-', label='Y', alpha=0.8)
            self.ax_imu_accel.plot(timestamps, accel_data[:, 2], 'b-', label='Z', alpha=0.8)
            self.ax_imu_accel.set_title('IMU Acceleration')
            self.ax_imu_accel.set_xlabel('Time (s)')
            self.ax_imu_accel.set_ylabel('Acceleration (m/s²)')
            self.ax_imu_accel.legend()
            self.ax_imu_accel.grid(True, alpha=0.3)

            # 角速度曲线
            self.ax_imu_gyro.clear()
            gyro_data = np.array(list(self.data_history['imu_gyro']))
            self.ax_imu_gyro.plot(timestamps, gyro_data[:, 0], 'r-', label='Roll', alpha=0.8)
            self.ax_imu_gyro.plot(timestamps, gyro_data[:, 1], 'g-', label='Pitch', alpha=0.8)
            self.ax_imu_gyro.plot(timestamps, gyro_data[:, 2], 'b-', label='Yaw', alpha=0.8)
            self.ax_imu_gyro.set_title('IMU Angular Velocity')
            self.ax_imu_gyro.set_xlabel('Time (s)')
            self.ax_imu_gyro.set_ylabel('Angular Velocity (rad/s)')
            self.ax_imu_gyro.legend()
            self.ax_imu_gyro.grid(True, alpha=0.3)

    def update_trajectory_comparison(self, est_pose: np.ndarray, gt_pose: np.ndarray):
        """更新轨迹对比显示"""
        self.data_history['trajectory_est'].append(est_pose[:2])  # x, y
        self.data_history['trajectory_gt'].append(gt_pose[:2])

        if len(self.data_history['trajectory_est']) > 1:
            self.ax_trajectory.clear()

            est_traj = np.array(list(self.data_history['trajectory_est']))
            gt_traj = np.array(list(self.data_history['trajectory_gt']))

            self.ax_trajectory.plot(est_traj[:, 0], est_traj[:, 1], 'r-',
                                  label='Estimated', linewidth=2, alpha=0.8)
            self.ax_trajectory.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-',
                                  label='Ground Truth', linewidth=2, alpha=0.8)

            # 标记当前位置
            self.ax_trajectory.plot(est_traj[-1, 0], est_traj[-1, 1], 'ro', markersize=8)
            self.ax_trajectory.plot(gt_traj[-1, 0], gt_traj[-1, 1], 'go', markersize=8)

            self.ax_trajectory.set_title('Trajectory: Estimated vs Ground Truth')
            self.ax_trajectory.set_xlabel('X (m)')
            self.ax_trajectory.set_ylabel('Y (m)')
            self.ax_trajectory.legend()
            self.ax_trajectory.grid(True, alpha=0.3)
            self.ax_trajectory.axis('equal')

    def update_pixel_statistics(self, depth_image: np.ndarray):
        """更新有效像素统计"""
        self.ax_pixel_stats.clear()

        # 计算有效像素统计
        valid_pixels = depth_image[depth_image > 0]
        total_pixels = depth_image.size
        valid_ratio = len(valid_pixels) / total_pixels

        # 饼图显示
        labels = ['Valid', 'Invalid']
        sizes = [valid_ratio, 1 - valid_ratio]
        colors = ['lightgreen', 'lightcoral']

        self.ax_pixel_stats.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        self.ax_pixel_stats.set_title('Valid Pixel Statistics')

    def update_temperature_distribution(self, thermal_image: np.ndarray):
        """更新温度分布直方图"""
        self.ax_temp_dist.clear()

        # 计算温度分布
        temperatures = thermal_image.flatten()
        self.ax_temp_dist.hist(temperatures, bins=50, alpha=0.7, color='orange', edgecolor='black')
        self.ax_temp_dist.set_title('Temperature Distribution')
        self.ax_temp_dist.set_xlabel('Temperature')
        self.ax_temp_dist.set_ylabel('Frequency')
        self.ax_temp_dist.grid(True, alpha=0.3)

    def refresh_display(self):
        """刷新显示"""
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def save_dashboard(self, save_path: str):
        """保存当前监控面板"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Dashboard saved to: {save_path}")


class MoEFusionVisualizer:
    """MoE融合过程可视化"""

    def __init__(self):
        self.expert_names = ['Geometric', 'Semantic', 'Visual']
        self.setup_plots()
        self.gate_entropy_history = deque(maxlen=100)
        self.expert_weights_history = deque(maxlen=100)
        self.thermal_guidance_history = deque(maxlen=100)

    def setup_plots(self):
        """设置MoE可视化图表"""
        self.fig_moe = plt.figure(figsize=(15, 10))
        gs = self.fig_moe.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 专家权重饼图
        self.ax_expert_pie = self.fig_moe.add_subplot(gs[0, 0])

        # 门控熵变化曲线
        self.ax_entropy = self.fig_moe.add_subplot(gs[0, 1])

        # 热引导权重变化
        self.ax_thermal_guidance = self.fig_moe.add_subplot(gs[0, 2])

        # 专家权重时间序列
        self.ax_weights_time = self.fig_moe.add_subplot(gs[1, :])

    def update_expert_weights(self, expert_weights: np.ndarray, timestamp: float = None):
        """更新专家权重可视化"""
        if timestamp is None:
            timestamp = len(self.expert_weights_history)

        self.expert_weights_history.append(expert_weights)

        # 专家权重饼图
        self.ax_expert_pie.clear()
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        wedges, texts, autotexts = self.ax_expert_pie.pie(
            expert_weights, labels=self.expert_names, colors=colors,
            autopct='%1.1f%%', startangle=90
        )

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        self.ax_expert_pie.set_title('Expert Weight Distribution', fontsize=12, fontweight='bold')

    def update_gate_entropy(self, gate_entropy: float, timestamp: float = None):
        """更新门控熵可视化"""
        if timestamp is None:
            timestamp = len(self.gate_entropy_history)

        self.gate_entropy_history.append((timestamp, gate_entropy))

        if len(self.gate_entropy_history) > 1:
            self.ax_entropy.clear()
            timestamps, entropies = zip(*self.gate_entropy_history)

            self.ax_entropy.plot(timestamps, entropies, 'b-', linewidth=2, alpha=0.8)
            self.ax_entropy.fill_between(timestamps, entropies, alpha=0.3)
            self.ax_entropy.set_title('Gate Entropy Evolution')
            self.ax_entropy.set_xlabel('Time Step')
            self.ax_entropy.set_ylabel('Entropy')
            self.ax_entropy.grid(True, alpha=0.3)

    def update_thermal_guidance(self, thermal_weight: float, timestamp: float = None):
        """更新热引导权重可视化"""
        if timestamp is None:
            timestamp = len(self.thermal_guidance_history)

        self.thermal_guidance_history.append((timestamp, thermal_weight))

        if len(self.thermal_guidance_history) > 1:
            self.ax_thermal_guidance.clear()
            timestamps, weights = zip(*self.thermal_guidance_history)

            self.ax_thermal_guidance.plot(timestamps, weights, 'r-', linewidth=2, alpha=0.8)
            self.ax_thermal_guidance.fill_between(timestamps, weights, alpha=0.3, color='red')
            self.ax_thermal_guidance.set_title('Thermal Guidance Weight')
            self.ax_thermal_guidance.set_xlabel('Time Step')
            self.ax_thermal_guidance.set_ylabel('Weight')
            self.ax_thermal_guidance.grid(True, alpha=0.3)

    def update_weights_timeline(self):
        """更新专家权重时间序列"""
        if len(self.expert_weights_history) > 1:
            self.ax_weights_time.clear()

            weights_array = np.array(list(self.expert_weights_history))
            timestamps = range(len(weights_array))

            colors = ['blue', 'green', 'red']
            for i, (expert_name, color) in enumerate(zip(self.expert_names, colors)):
                self.ax_weights_time.plot(timestamps, weights_array[:, i],
                                        color=color, label=expert_name,
                                        linewidth=2, alpha=0.8)

            self.ax_weights_time.set_title('Expert Weights Evolution')
            self.ax_weights_time.set_xlabel('Time Step')
            self.ax_weights_time.set_ylabel('Weight')
            self.ax_weights_time.legend()
            self.ax_weights_time.grid(True, alpha=0.3)

    def visualize_attention_heatmap(self, attention_weights: np.ndarray,
                                  save_path: str = None):
        """可视化512维token注意力热力图"""
        plt.figure(figsize=(12, 8))

        # 重塑为2D以便可视化
        if attention_weights.ndim == 1:
            # 假设512维，重塑为16x32
            attention_2d = attention_weights.reshape(16, 32)
        else:
            attention_2d = attention_weights

        sns.heatmap(attention_2d, cmap='viridis', cbar=True,
                   xticklabels=False, yticklabels=False)
        plt.title('Token Attention Heatmap (512D)')
        plt.xlabel('Token Dimension')
        plt.ylabel('Token Dimension')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def refresh_moe_display(self):
        """刷新MoE可视化"""
        self.update_weights_timeline()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)


class TrainingProcessVisualizer:
    """训练过程可视化监控"""

    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.setup_training_plots()

    def setup_training_plots(self):
        """设置训练过程可视化"""
        self.fig_training = plt.figure(figsize=(16, 12))
        gs = self.fig_training.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Kendall权重动态变化
        self.ax_kendall = self.fig_training.add_subplot(gs[0, 0])

        # 多任务损失分解
        self.ax_losses = self.fig_training.add_subplot(gs[0, 1])

        # 不确定性σ值趋势
        self.ax_uncertainty = self.fig_training.add_subplot(gs[0, 2])

        # ATE误差曲线
        self.ax_ate = self.fig_training.add_subplot(gs[1, 0])

        # mAP检测精度
        self.ax_map = self.fig_training.add_subplot(gs[1, 1])

        # 专家利用率统计
        self.ax_expert_util = self.fig_training.add_subplot(gs[1, 2])

        # 收敛分析
        self.ax_convergence = self.fig_training.add_subplot(gs[2, :])

    def update_training_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """更新训练指标"""
        # 记录历史数据
        self.metrics_history['epoch'].append(epoch)

        # Kendall权重
        if 'kendall_weights' in metrics:
            kendall = metrics['kendall_weights']
            self.metrics_history['pose_weight'].append(kendall.get('pose_weight', 0))
            self.metrics_history['detection_weight'].append(kendall.get('detection_weight', 0))
            self.metrics_history['gate_weight'].append(kendall.get('gate_weight', 0))

        # 损失值
        if 'losses' in metrics:
            losses = metrics['losses']
            self.metrics_history['pose_loss'].append(losses.get('pose', 0))
            self.metrics_history['detection_loss'].append(losses.get('detection', 0))
            self.metrics_history['total_loss'].append(losses.get('total', 0))

        # 评估指标
        if 'trajectory_metrics' in metrics:
            traj = metrics['trajectory_metrics']
            self.metrics_history['ate'].append(traj.get('ATE', 0))

        if 'detection_metrics' in metrics:
            det = metrics['detection_metrics']
            self.metrics_history['map'].append(det.get('mAP', 0))

    def plot_kendall_weights(self):
        """绘制Kendall权重变化"""
        if 'epoch' not in self.metrics_history or len(self.metrics_history['epoch']) < 2:
            return

        self.ax_kendall.clear()
        epochs = self.metrics_history['epoch']

        if 'pose_weight' in self.metrics_history:
            self.ax_kendall.plot(epochs, self.metrics_history['pose_weight'],
                               'r-', label='Pose', linewidth=2)
        if 'detection_weight' in self.metrics_history:
            self.ax_kendall.plot(epochs, self.metrics_history['detection_weight'],
                               'g-', label='Detection', linewidth=2)
        if 'gate_weight' in self.metrics_history:
            self.ax_kendall.plot(epochs, self.metrics_history['gate_weight'],
                               'b-', label='Gate', linewidth=2)

        self.ax_kendall.set_title('Kendall Weight Evolution')
        self.ax_kendall.set_xlabel('Epoch')
        self.ax_kendall.set_ylabel('Weight')
        self.ax_kendall.legend()
        self.ax_kendall.grid(True, alpha=0.3)
        self.ax_kendall.set_yscale('log')

    def plot_loss_curves(self):
        """绘制损失曲线"""
        if 'epoch' not in self.metrics_history or len(self.metrics_history['epoch']) < 2:
            return

        self.ax_losses.clear()
        epochs = self.metrics_history['epoch']

        if 'pose_loss' in self.metrics_history:
            self.ax_losses.plot(epochs, self.metrics_history['pose_loss'],
                              'r-', label='Pose Loss', linewidth=2)
        if 'detection_loss' in self.metrics_history:
            self.ax_losses.plot(epochs, self.metrics_history['detection_loss'],
                              'g-', label='Detection Loss', linewidth=2)
        if 'total_loss' in self.metrics_history:
            self.ax_losses.plot(epochs, self.metrics_history['total_loss'],
                              'k--', label='Total Loss', linewidth=2)

        self.ax_losses.set_title('Multi-Task Loss Decomposition')
        self.ax_losses.set_xlabel('Epoch')
        self.ax_losses.set_ylabel('Loss')
        self.ax_losses.legend()
        self.ax_losses.grid(True, alpha=0.3)
        self.ax_losses.set_yscale('log')

    def plot_performance_metrics(self):
        """绘制性能指标"""
        if 'epoch' not in self.metrics_history or len(self.metrics_history['epoch']) < 2:
            return

        epochs = self.metrics_history['epoch']

        # ATE曲线
        if 'ate' in self.metrics_history:
            self.ax_ate.clear()
            self.ax_ate.plot(epochs, self.metrics_history['ate'], 'b-', linewidth=2)
            self.ax_ate.set_title('ATE Error Evolution')
            self.ax_ate.set_xlabel('Epoch')
            self.ax_ate.set_ylabel('ATE (m)')
            self.ax_ate.grid(True, alpha=0.3)

        # mAP曲线
        if 'map' in self.metrics_history:
            self.ax_map.clear()
            self.ax_map.plot(epochs, self.metrics_history['map'], 'g-', linewidth=2)
            self.ax_map.set_title('mAP Detection Accuracy')
            self.ax_map.set_xlabel('Epoch')
            self.ax_map.set_ylabel('mAP')
            self.ax_map.grid(True, alpha=0.3)

    def update_full_training_display(self):
        """更新完整训练可视化"""
        self.plot_kendall_weights()
        self.plot_loss_curves()
        self.plot_performance_metrics()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def save_training_plots(self, save_dir: str, epoch: int):
        """保存训练图表"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'training_plots_epoch_{epoch:03d}.png')
        self.fig_training.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 Training plots saved to: {save_path}")


class SemanticMap3DVisualizer:
    """3D语义地图可视化"""

    def __init__(self):
        self.setup_3d_viewer()
        self.darpa_colors = {
            0: [1.0, 0.0, 0.0],  # fiducial - 红色
            1: [0.0, 1.0, 0.0],  # extinguisher - 绿色
            2: [0.0, 0.0, 1.0],  # phone - 蓝色
            3: [1.0, 1.0, 0.0],  # backpack - 黄色
            4: [1.0, 0.0, 1.0],  # survivor - 品红
            5: [0.0, 1.0, 1.0],  # drill - 青色
            6: [0.5, 0.5, 0.5],  # other - 灰色
            7: [1.0, 0.5, 0.0]   # unknown - 橙色
        }

    def setup_3d_viewer(self):
        """设置3D查看器"""
        if OPEN3D_AVAILABLE:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="MineSLAM 3D Semantic Map", width=1200, height=800)

            # 设置渲染选项
            render_option = self.vis.get_render_option()
            render_option.show_coordinate_frame = True
            render_option.background_color = np.array([0.1, 0.1, 0.1])

            # 初始化几何体容器
            self.geometries = {}
        else:
            warnings.warn("Open3D not available, using matplotlib 3D instead")
            self.setup_matplotlib_3d()

    def setup_matplotlib_3d(self):
        """设置matplotlib 3D可视化作为备选"""
        self.fig_3d = plt.figure(figsize=(12, 10))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

    def update_semantic_pointcloud(self, points: np.ndarray, semantic_labels: np.ndarray):
        """更新语义着色点云"""
        if not OPEN3D_AVAILABLE:
            self._update_pointcloud_matplotlib(points, semantic_labels)
            return

        # 创建颜色数组
        colors = np.zeros((len(points), 3))
        for i, label in enumerate(semantic_labels):
            label_int = int(label) if label < len(self.darpa_colors) else 7
            colors[i] = self.darpa_colors[label_int]

        # 创建或更新点云
        if 'pointcloud' not in self.geometries:
            self.geometries['pointcloud'] = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.geometries['pointcloud'])

        self.geometries['pointcloud'].points = o3d.utility.Vector3dVector(points)
        self.geometries['pointcloud'].colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.geometries['pointcloud'])

    def _update_pointcloud_matplotlib(self, points: np.ndarray, semantic_labels: np.ndarray):
        """使用matplotlib更新点云（备选方案）"""
        self.ax_3d.clear()

        # 为每个语义类别绘制点
        for label_id, color in self.darpa_colors.items():
            mask = semantic_labels == label_id
            if np.any(mask):
                label_points = points[mask]
                self.ax_3d.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2],
                                 c=[color], s=1, alpha=0.6, label=f'Class {label_id}')

        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Semantic PointCloud')

    def update_trajectory(self, estimated_trajectory: np.ndarray,
                         ground_truth_trajectory: np.ndarray = None):
        """更新机器人轨迹显示"""
        if not OPEN3D_AVAILABLE:
            return

        # 估计轨迹
        if 'est_trajectory' not in self.geometries:
            self.geometries['est_trajectory'] = o3d.geometry.LineSet()
            self.vis.add_geometry(self.geometries['est_trajectory'])

        # 创建线段
        lines = [[i, i+1] for i in range(len(estimated_trajectory)-1)]
        self.geometries['est_trajectory'].points = o3d.utility.Vector3dVector(estimated_trajectory)
        self.geometries['est_trajectory'].lines = o3d.utility.Vector2iVector(lines)
        self.geometries['est_trajectory'].colors = o3d.utility.Vector3dVector(
            [[1, 0, 0] for _ in lines]  # 红色
        )

        # 真值轨迹
        if ground_truth_trajectory is not None:
            if 'gt_trajectory' not in self.geometries:
                self.geometries['gt_trajectory'] = o3d.geometry.LineSet()
                self.vis.add_geometry(self.geometries['gt_trajectory'])

            gt_lines = [[i, i+1] for i in range(len(ground_truth_trajectory)-1)]
            self.geometries['gt_trajectory'].points = o3d.utility.Vector3dVector(ground_truth_trajectory)
            self.geometries['gt_trajectory'].lines = o3d.utility.Vector2iVector(gt_lines)
            self.geometries['gt_trajectory'].colors = o3d.utility.Vector3dVector(
                [[0, 1, 0] for _ in gt_lines]  # 绿色
            )

        self.vis.update_geometry(self.geometries['est_trajectory'])
        if ground_truth_trajectory is not None:
            self.vis.update_geometry(self.geometries['gt_trajectory'])

    def update_detection_boxes(self, boxes_3d: np.ndarray, class_ids: np.ndarray,
                              scores: np.ndarray = None):
        """更新3D检测边界框"""
        if not OPEN3D_AVAILABLE:
            return

        # 清除之前的检测框
        for key in list(self.geometries.keys()):
            if key.startswith('bbox_'):
                self.vis.remove_geometry(self.geometries[key])
                del self.geometries[key]

        # 添加新的检测框
        for i, (box, class_id) in enumerate(zip(boxes_3d, class_ids)):
            if scores is not None and scores[i] < 0.5:  # 置信度过滤
                continue

            # 创建边界框
            center = box[:3]
            extent = box[3:6]

            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=center - extent/2,
                max_bound=center + extent/2
            )

            # 设置颜色
            color = self.darpa_colors.get(int(class_id), [0.5, 0.5, 0.5])
            bbox.color = color

            bbox_key = f'bbox_{i}'
            self.geometries[bbox_key] = bbox
            self.vis.add_geometry(bbox)

    def refresh_3d_display(self):
        """刷新3D显示"""
        if OPEN3D_AVAILABLE:
            self.vis.poll_events()
            self.vis.update_renderer()
        else:
            plt.draw()
            plt.pause(0.01)

    def save_3d_view(self, save_path: str):
        """保存3D视图"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if OPEN3D_AVAILABLE:
            self.vis.capture_screen_image(save_path)
        else:
            self.fig_3d.savefig(save_path, dpi=150, bbox_inches='tight')

        print(f"🗺️ 3D map view saved to: {save_path}")


class MineSLAMVisualizer:
    """MineSLAM完整可视化系统"""

    def __init__(self, output_dir: str = "outputs/visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各个可视化模块
        self.sensor_dashboard = RealTimeSensorDashboard()
        self.moe_visualizer = MoEFusionVisualizer()
        self.training_visualizer = TrainingProcessVisualizer()
        self.map_3d_visualizer = SemanticMap3DVisualizer()

        print(f"🎨 MineSLAM Visualizer initialized. Output directory: {self.output_dir}")

    def update_realtime_data(self, sensor_data: Dict[str, Any],
                           model_output: Dict[str, Any]):
        """更新实时数据可视化"""
        # 传感器数据更新
        if 'rgb' in sensor_data:
            detections = model_output.get('detections_2d', [])
            self.sensor_dashboard.update_rgb_detection(sensor_data['rgb'], detections)

        if 'depth' in sensor_data:
            self.sensor_dashboard.update_depth_heatmap(sensor_data['depth'])
            self.sensor_dashboard.update_pixel_statistics(sensor_data['depth'])

        if 'thermal' in sensor_data:
            rgb_bg = sensor_data.get('rgb', None)
            self.sensor_dashboard.update_thermal_overlay(sensor_data['thermal'], rgb_bg)
            self.sensor_dashboard.update_temperature_distribution(sensor_data['thermal'])

        if 'pointcloud' in sensor_data:
            points = sensor_data['pointcloud']
            colors = sensor_data.get('pointcloud_colors', None)
            self.sensor_dashboard.update_pointcloud_3d(points, colors)

        if 'imu' in sensor_data:
            timestamp = sensor_data.get('timestamp', 0)
            imu_data = sensor_data['imu']
            accel = imu_data[:3]
            gyro = imu_data[3:]
            self.sensor_dashboard.update_imu_data(timestamp, accel, gyro)

    def update_moe_analysis(self, moe_output: Dict[str, torch.Tensor]):
        """更新MoE融合分析可视化"""
        if 'expert_weights' in moe_output:
            weights = moe_output['expert_weights'].cpu().numpy().mean(axis=0)
            self.moe_visualizer.update_expert_weights(weights)

        if 'gate_entropy' in moe_output:
            entropy = moe_output['gate_entropy'].cpu().item()
            self.moe_visualizer.update_gate_entropy(entropy)

        if 'thermal_guidance_weight' in moe_output:
            thermal_weight = moe_output['thermal_guidance_weight'].cpu().item()
            self.moe_visualizer.update_thermal_guidance(thermal_weight)

        self.moe_visualizer.refresh_moe_display()

    def update_training_progress(self, epoch: int, metrics: Dict[str, Any]):
        """更新训练过程可视化"""
        self.training_visualizer.update_training_metrics(epoch, metrics)
        self.training_visualizer.update_full_training_display()

    def update_3d_semantic_map(self, map_data: Dict[str, Any]):
        """更新3D语义地图"""
        if 'pointcloud' in map_data and 'semantic_labels' in map_data:
            self.map_3d_visualizer.update_semantic_pointcloud(
                map_data['pointcloud'], map_data['semantic_labels']
            )

        if 'estimated_trajectory' in map_data:
            gt_traj = map_data.get('ground_truth_trajectory', None)
            self.map_3d_visualizer.update_trajectory(
                map_data['estimated_trajectory'], gt_traj
            )

        if 'detection_boxes' in map_data:
            boxes = map_data['detection_boxes']
            classes = map_data.get('detection_classes', np.zeros(len(boxes)))
            scores = map_data.get('detection_scores', np.ones(len(boxes)))
            self.map_3d_visualizer.update_detection_boxes(boxes, classes, scores)

        self.map_3d_visualizer.refresh_3d_display()

    def save_all_visualizations(self, epoch: int, prefix: str = "mineslam"):
        """保存所有可视化结果"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # 保存传感器监控面板
        dashboard_path = self.output_dir / f"{prefix}_dashboard_epoch_{epoch:03d}_{timestamp}.png"
        self.sensor_dashboard.save_dashboard(str(dashboard_path))

        # 保存训练图表
        training_dir = self.output_dir / "training_plots"
        self.training_visualizer.save_training_plots(str(training_dir), epoch)

        # 保存3D地图视图
        map_path = self.output_dir / f"{prefix}_3d_map_epoch_{epoch:03d}_{timestamp}.png"
        self.map_3d_visualizer.save_3d_view(str(map_path))

        print(f"💾 All visualizations saved for epoch {epoch}")

    def create_evaluation_report_plots(self, evaluation_results: Dict[str, Any],
                                     save_path: str):
        """创建评估报告图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MineSLAM Evaluation Report', fontsize=16, fontweight='bold')

        # 轨迹精度指标
        traj_metrics = evaluation_results.get('trajectory_metrics', {})
        metric_names = ['ATE', 'RPE', 'Translation RMSE', 'Rotation RMSE']
        metric_values = [
            traj_metrics.get('ATE', 0),
            traj_metrics.get('RPE', 0),
            traj_metrics.get('translation_rmse', 0),
            traj_metrics.get('rotation_rmse', 0)
        ]

        axes[0, 0].bar(metric_names, metric_values, color='lightblue', alpha=0.8)
        axes[0, 0].set_title('Trajectory Accuracy Metrics')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 检测精度指标
        det_metrics = evaluation_results.get('detection_metrics', {})
        axes[0, 1].bar(['mAP'], [det_metrics.get('mAP', 0)], color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Detection Accuracy')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_ylim(0, 1)

        # 专家利用率
        fusion_metrics = evaluation_results.get('fusion_metrics', {})
        expert_util = fusion_metrics.get('expert_utilization', {})
        expert_names = list(expert_util.keys())
        expert_values = list(expert_util.values())

        axes[0, 2].pie(expert_values, labels=expert_names, autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Expert Utilization')

        # 不确定性趋势
        uncertainty_metrics = evaluation_results.get('uncertainty_metrics', {})
        uncertainty_trends = uncertainty_metrics.get('uncertainty_trends', {})

        uncertainty_names = list(uncertainty_trends.keys())
        uncertainty_values = [trends.get('current', 0) for trends in uncertainty_trends.values()]

        axes[1, 0].bar(uncertainty_names, uncertainty_values, color='orange', alpha=0.8)
        axes[1, 0].set_title('Current Uncertainty Values')
        axes[1, 0].set_ylabel('Uncertainty (σ)')

        # 权重平衡分析
        weight_balance = uncertainty_metrics.get('weight_balance', {})
        balance_score = weight_balance.get('balance_score', 0)
        pose_dominance = weight_balance.get('pose_dominance', 0)

        axes[1, 1].bar(['Balance Score', 'Pose Dominance'],
                      [balance_score, pose_dominance],
                      color=['purple', 'red'], alpha=0.8)
        axes[1, 1].set_title('Weight Balance Analysis')
        axes[1, 1].set_ylabel('Score/Ratio')
        axes[1, 1].set_ylim(0, 1)

        # 模态贡献度
        modality_contrib = fusion_metrics.get('modality_contributions', {})
        if modality_contrib:
            contrib_names = list(modality_contrib.keys())
            contrib_values = list(modality_contrib.values())

            axes[1, 2].bar(contrib_names, contrib_values, color='cyan', alpha=0.8)
            axes[1, 2].set_title('Modality Contributions')
            axes[1, 2].set_ylabel('Contribution')
            axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📊 Evaluation report plots saved to: {save_path}")


if __name__ == "__main__":
    # 测试可视化系统
    print("🎨 MineSLAM Visualizer - Test Mode")

    # 创建可视化器
    visualizer = MineSLAMVisualizer("outputs/test_visualization")

    # 模拟数据测试
    sensor_data = {
        'rgb': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        'depth': np.random.rand(480, 640) * 10,
        'thermal': np.random.rand(480, 640) * 50 + 273,  # 温度(K)
        'pointcloud': np.random.rand(1000, 3) * 10,
        'imu': np.random.rand(6),
        'timestamp': 1.0
    }

    model_output = {
        'expert_weights': torch.rand(1, 3),
        'gate_entropy': torch.rand(1),
        'thermal_guidance_weight': torch.rand(1)
    }

    # 更新可视化
    visualizer.update_realtime_data(sensor_data, model_output)
    visualizer.update_moe_analysis(model_output)

    print("✅ Visualization test completed!")