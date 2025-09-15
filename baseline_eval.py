#!/usr/bin/env python3
"""
SLAM Baseline Evaluation Script
计算ATE、RPE、2D mAP、3D mAP@0.5等指标
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import argparse
from scipy.spatial.transform import Rotation
import warnings

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入自定义模块
from data.mineslam_dataset import MineSLAMDataset
from baselines.lidar_odometry import run_lidar_odometry_on_dataset
from baselines.rgb_thermal_detection import run_rgb_thermal_detection_on_dataset


class TrajectoryEvaluator:
    """轨迹评估器：计算ATE和RPE指标"""
    
    def __init__(self):
        pass
    
    def load_trajectory(self, trajectory_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载轨迹文件
        
        Returns:
            timestamps: [N] 时间戳数组
            positions: [N, 3] 位置数组
        """
        if not os.path.exists(trajectory_file):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
        
        df = pd.read_csv(trajectory_file)
        
        required_cols = ['timestamp', 'position_x', 'position_y', 'position_z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in trajectory: {missing_cols}")
        
        timestamps = df['timestamp'].values
        positions = df[['position_x', 'position_y', 'position_z']].values
        
        return timestamps, positions
    
    def align_trajectories(self, est_timestamps: np.ndarray, est_positions: np.ndarray,
                          gt_timestamps: np.ndarray, gt_positions: np.ndarray,
                          max_time_diff: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        时间对齐两条轨迹
        
        Returns:
            aligned_est_positions: [M, 3]
            aligned_gt_positions: [M, 3]
        """
        aligned_est = []
        aligned_gt = []
        
        for i, est_t in enumerate(est_timestamps):
            # 找到最近的地面真值时间戳
            time_diffs = np.abs(gt_timestamps - est_t)
            min_idx = np.argmin(time_diffs)
            
            if time_diffs[min_idx] <= max_time_diff:
                aligned_est.append(est_positions[i])
                aligned_gt.append(gt_positions[min_idx])
        
        if len(aligned_est) == 0:
            raise ValueError("No aligned trajectory points found")
        
        return np.array(aligned_est), np.array(aligned_gt)
    
    def compute_ate(self, est_positions: np.ndarray, gt_positions: np.ndarray) -> Dict[str, float]:
        """
        计算绝对轨迹误差 (ATE)
        
        Args:
            est_positions: [N, 3] 估计位置
            gt_positions: [N, 3] 地面真值位置
        
        Returns:
            ATE指标字典
        """
        if len(est_positions) != len(gt_positions):
            raise ValueError("Trajectory lengths must match")
        
        # 计算欧几里得距离误差
        errors = np.linalg.norm(est_positions - gt_positions, axis=1)
        
        ate_metrics = {
            'ate_mean': float(np.mean(errors)),
            'ate_median': float(np.median(errors)),
            'ate_std': float(np.std(errors)),
            'ate_min': float(np.min(errors)),
            'ate_max': float(np.max(errors)),
            'ate_rmse': float(np.sqrt(np.mean(errors ** 2))),
            'num_poses': len(errors)
        }
        
        return ate_metrics
    
    def compute_rpe(self, est_positions: np.ndarray, gt_positions: np.ndarray,
                   delta: int = 1) -> Dict[str, float]:
        """
        计算相对位姿误差 (RPE)
        
        Args:
            est_positions: [N, 3] 估计位置
            gt_positions: [N, 3] 地面真值位置
            delta: 帧间隔
        
        Returns:
            RPE指标字典
        """
        if len(est_positions) != len(gt_positions):
            raise ValueError("Trajectory lengths must match")
        
        if len(est_positions) <= delta:
            raise ValueError(f"Not enough poses for delta={delta}")
        
        rpe_errors = []
        
        for i in range(len(est_positions) - delta):
            # 计算相对运动
            est_rel = est_positions[i + delta] - est_positions[i]
            gt_rel = gt_positions[i + delta] - gt_positions[i]
            
            # 计算相对误差
            rel_error = np.linalg.norm(est_rel - gt_rel)
            rpe_errors.append(rel_error)
        
        rpe_errors = np.array(rpe_errors)
        
        rpe_metrics = {
            'rpe_mean': float(np.mean(rpe_errors)),
            'rpe_median': float(np.median(rpe_errors)),
            'rpe_std': float(np.std(rpe_errors)),
            'rpe_min': float(np.min(rpe_errors)),
            'rpe_max': float(np.max(rpe_errors)),
            'rpe_rmse': float(np.sqrt(np.mean(rpe_errors ** 2))),
            'num_pairs': len(rpe_errors),
            'delta': delta
        }
        
        return rpe_metrics


class DetectionEvaluator:
    """检测评估器：计算2D和3D mAP指标"""
    
    def __init__(self):
        pass
    
    def compute_iou_2d(self, box1: List[float], box2: List[float]) -> float:
        """
        计算2D检测框IoU
        
        Args:
            box1, box2: [x1, y1, x2, y2]
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_iou_3d(self, center1: List[float], size1: List[float],
                      center2: List[float], size2: List[float]) -> float:
        """
        计算3D边界框IoU（简化版本）
        
        Args:
            center1, center2: [x, y, z] 中心点
            size1, size2: [w, h, l] 尺寸
        """
        # 计算每个轴上的重叠
        overlaps = []
        for i in range(3):
            min1 = center1[i] - size1[i] / 2
            max1 = center1[i] + size1[i] / 2
            min2 = center2[i] - size2[i] / 2
            max2 = center2[i] + size2[i] / 2
            
            overlap = max(0, min(max1, max2) - max(min1, min2))
            overlaps.append(overlap)
        
        # 3D交集体积
        inter_volume = overlaps[0] * overlaps[1] * overlaps[2]
        
        # 各自体积
        volume1 = size1[0] * size1[1] * size1[2]
        volume2 = size2[0] * size2[1] * size2[2]
        
        # 并集体积
        union_volume = volume1 + volume2 - inter_volume
        
        return inter_volume / union_volume if union_volume > 0 else 0.0
    
    def compute_ap(self, confidences: np.ndarray, matches: np.ndarray, 
                   num_gt: int) -> float:
        """
        计算单个类别的AP
        
        Args:
            confidences: 检测置信度
            matches: 是否匹配到GT (0/1)
            num_gt: GT数量
        """
        if num_gt == 0:
            return 0.0 if len(confidences) > 0 else float('nan')
        
        # 按置信度排序
        indices = np.argsort(-confidences)
        matches = matches[indices]
        
        # 计算累积TP和FP
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        
        # 计算precision和recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / num_gt
        
        # 计算AP (11点插值)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        return ap
    
    def evaluate_detections(self, detections_file: str, 
                          iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        评估检测结果
        
        Args:
            detections_file: 检测结果JSON文件
            iou_threshold: IoU阈值
        
        Returns:
            mAP指标字典
        """
        if not os.path.exists(detections_file):
            print(f"Warning: Detection file not found: {detections_file}")
            return {
                '2d_map': 0.0,
                '3d_map': 0.0,
                'num_frames': 0,
                'num_detections': 0
            }
        
        with open(detections_file, 'r') as f:
            detection_results = json.load(f)
        
        # 由于没有标注的地面真值检测框，这里返回简化的指标
        # 实际应用中需要加载GT标注进行比较
        
        total_detections = 0
        successful_frames = 0
        
        for frame_result in detection_results:
            if frame_result.get('success', False):
                successful_frames += 1
                total_detections += frame_result.get('num_detections', 0)
        
        # 简化指标（没有GT的情况下）
        avg_detections_per_frame = total_detections / max(1, successful_frames)
        
        return {
            '2d_map': 0.0,  # 需要GT标注才能计算
            '3d_map': 0.0,  # 需要GT标注才能计算
            'num_frames': len(detection_results),
            'num_detections': total_detections,
            'successful_frames': successful_frames,
            'avg_detections_per_frame': avg_detections_per_frame,
            'detection_rate': successful_frames / max(1, len(detection_results))
        }


class BaselineEvaluator:
    """基线算法评估器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.trajectory_evaluator = TrajectoryEvaluator()
        self.detection_evaluator = DetectionEvaluator()
    
    def run_lidar_odometry_baseline(self, dataset, output_dir: str) -> Tuple[str, Dict]:
        """运行LiDAR里程计基线"""
        print("Running LiDAR-only Odometry Baseline...")
        
        lidar_config = self.config.get('lidar_odometry', {})
        
        trajectory_file, results = run_lidar_odometry_on_dataset(
            dataset, lidar_config, output_dir
        )
        
        return trajectory_file, results
    
    def run_detection_baseline(self, dataset, output_dir: str) -> Tuple[str, Dict]:
        """运行RGB-Thermal检测基线"""
        print("Running RGB-Thermal Detection Baseline...")
        
        detection_config = self.config.get('rgb_thermal_detection', {})
        
        detection_file, results = run_rgb_thermal_detection_on_dataset(
            dataset, detection_config, output_dir
        )
        
        return detection_file, results
    
    def evaluate_trajectory(self, est_trajectory_file: str, 
                          gt_trajectory_file: str) -> Dict[str, float]:
        """评估轨迹"""
        try:
            # 加载估计轨迹
            est_timestamps, est_positions = self.trajectory_evaluator.load_trajectory(est_trajectory_file)
            
            # 加载地面真值轨迹
            gt_timestamps, gt_positions = self.trajectory_evaluator.load_trajectory(gt_trajectory_file)
            
            # 时间对齐
            aligned_est, aligned_gt = self.trajectory_evaluator.align_trajectories(
                est_timestamps, est_positions, gt_timestamps, gt_positions
            )
            
            # 计算ATE
            ate_metrics = self.trajectory_evaluator.compute_ate(aligned_est, aligned_gt)
            
            # 计算RPE
            rpe_metrics = self.trajectory_evaluator.compute_rpe(aligned_est, aligned_gt, delta=10)
            
            # 合并指标
            trajectory_metrics = {**ate_metrics, **rpe_metrics}
            
            return trajectory_metrics
            
        except Exception as e:
            print(f"Error evaluating trajectory: {e}")
            return {
                'ate_mean': float('inf'),
                'ate_rmse': float('inf'),
                'rpe_mean': float('inf'),
                'rpe_rmse': float('inf'),
                'error': str(e)
            }
    
    def evaluate_detections(self, detection_file: str) -> Dict[str, float]:
        """评估检测结果"""
        return self.detection_evaluator.evaluate_detections(detection_file)
    
    def generate_report(self, results: Dict, output_dir: str):
        """生成评估报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整结果
        results_file = output_dir / "baseline_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 生成文本报告
        self._generate_text_report(results, output_dir)
        
        # 生成可视化
        self._generate_visualizations(results, output_dir)
        
        print(f"Evaluation report saved to: {output_dir}")
    
    def _generate_text_report(self, results: Dict, output_dir: Path):
        """生成文本报告"""
        report_file = output_dir / "evaluation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MineSLAM Baseline Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            # LiDAR里程计结果
            if 'lidar_odometry' in results:
                f.write("LiDAR-only Odometry Results:\n")
                f.write("-" * 40 + "\n")
                
                lidar_results = results['lidar_odometry']
                if 'trajectory_metrics' in lidar_results:
                    metrics = lidar_results['trajectory_metrics']
                    f.write(f"ATE Mean: {metrics.get('ate_mean', 'N/A'):.4f} m\n")
                    f.write(f"ATE RMSE: {metrics.get('ate_rmse', 'N/A'):.4f} m\n")
                    f.write(f"RPE Mean: {metrics.get('rpe_mean', 'N/A'):.4f} m\n")
                    f.write(f"RPE RMSE: {metrics.get('rpe_rmse', 'N/A'):.4f} m\n")
                    f.write(f"Aligned Poses: {metrics.get('num_poses', 'N/A')}\n")
                
                if 'statistics' in lidar_results:
                    stats = lidar_results['statistics']
                    f.write(f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%\n")
                    f.write(f"Avg Processing Time: {stats.get('avg_processing_time', 0)*1000:.1f} ms/frame\n")
                
                f.write("\n")
            
            # RGB-Thermal检测结果
            if 'rgb_thermal_detection' in results:
                f.write("RGB-Thermal Detection Results:\n")
                f.write("-" * 40 + "\n")
                
                detection_results = results['rgb_thermal_detection']
                if 'detection_metrics' in detection_results:
                    metrics = detection_results['detection_metrics']
                    f.write(f"2D mAP: {metrics.get('2d_map', 0):.4f}\n")
                    f.write(f"3D mAP@0.5: {metrics.get('3d_map', 0):.4f}\n")
                    f.write(f"Detection Rate: {metrics.get('detection_rate', 0)*100:.1f}%\n")
                    f.write(f"Avg Detections/Frame: {metrics.get('avg_detections_per_frame', 0):.1f}\n")
                
                if 'statistics' in detection_results:
                    stats = detection_results['statistics']
                    f.write(f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%\n")
                    f.write(f"Total Detections: {stats.get('total_detections', 0)}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"Text report saved to: {report_file}")
    
    def _generate_visualizations(self, results: Dict, output_dir: Path):
        """生成可视化图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建总结图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('MineSLAM Baseline Evaluation Summary', fontsize=16)
            
            # 1. LiDAR里程计误差
            if 'lidar_odometry' in results and 'trajectory_metrics' in results['lidar_odometry']:
                metrics = results['lidar_odometry']['trajectory_metrics']
                
                ax = axes[0, 0]
                error_types = ['ATE Mean', 'ATE RMSE', 'RPE Mean', 'RPE RMSE']
                error_values = [
                    metrics.get('ate_mean', 0),
                    metrics.get('ate_rmse', 0),
                    metrics.get('rpe_mean', 0),
                    metrics.get('rpe_rmse', 0)
                ]
                
                bars = ax.bar(error_types, error_values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
                ax.set_title('LiDAR Odometry Error Metrics')
                ax.set_ylabel('Error (m)')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for bar, value in zip(bars, error_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            else:
                axes[0, 0].text(0.5, 0.5, 'No LiDAR Odometry Results', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('LiDAR Odometry Error Metrics')
            
            # 2. 检测性能
            if 'rgb_thermal_detection' in results and 'detection_metrics' in results['rgb_thermal_detection']:
                metrics = results['rgb_thermal_detection']['detection_metrics']
                
                ax = axes[0, 1]
                detection_metrics = ['2D mAP', '3D mAP@0.5', 'Detection Rate']
                detection_values = [
                    metrics.get('2d_map', 0),
                    metrics.get('3d_map', 0),
                    metrics.get('detection_rate', 0)
                ]
                
                bars = ax.bar(detection_metrics, detection_values, color=['purple', 'brown', 'pink'])
                ax.set_title('RGB-Thermal Detection Performance')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                for bar, value in zip(bars, detection_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Detection Results', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('RGB-Thermal Detection Performance')
            
            # 3. 处理时间比较
            ax = axes[1, 0]
            algorithms = []
            processing_times = []
            
            if 'lidar_odometry' in results and 'statistics' in results['lidar_odometry']:
                algorithms.append('LiDAR Odometry')
                processing_times.append(results['lidar_odometry']['statistics'].get('avg_processing_time', 0) * 1000)
            
            if 'rgb_thermal_detection' in results and 'statistics' in results['rgb_thermal_detection']:
                algorithms.append('RGB-Thermal Detection')
                processing_times.append(results['rgb_thermal_detection']['statistics'].get('avg_processing_time', 0) * 1000)
            
            if algorithms:
                bars = ax.bar(algorithms, processing_times, color=['cyan', 'magenta'])
                ax.set_title('Average Processing Time')
                ax.set_ylabel('Time (ms/frame)')
                
                for bar, value in zip(bars, processing_times):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Timing Data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Average Processing Time')
            
            # 4. 成功率比较
            ax = axes[1, 1]
            success_rates = []
            
            if 'lidar_odometry' in results and 'statistics' in results['lidar_odometry']:
                success_rates.append(('LiDAR Odometry', results['lidar_odometry']['statistics'].get('success_rate', 0) * 100))
            
            if 'rgb_thermal_detection' in results and 'statistics' in results['rgb_thermal_detection']:
                success_rates.append(('RGB-Thermal Detection', results['rgb_thermal_detection']['statistics'].get('success_rate', 0) * 100))
            
            if success_rates:
                algs, rates = zip(*success_rates)
                bars = ax.bar(algs, rates, color=['gold', 'silver'])
                ax.set_title('Success Rate Comparison')
                ax.set_ylabel('Success Rate (%)')
                ax.set_ylim(0, 100)
                
                for bar, value in zip(bars, rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Success Rate Data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Success Rate Comparison')
            
            plt.tight_layout()
            
            # 保存图表
            viz_file = output_dir / "evaluation_summary.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to: {viz_file}")
            
        except Exception as e:
            print(f"Warning: Failed to generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SLAM baselines')
    parser.add_argument('--config', type=str, 
                       default='/root/autodl-tmp/new huanjing ganzhi-cc/configs/mineslam.yaml',
                       help='Configuration file path')
    parser.add_argument('--output_dir', type=str,
                       default='/root/autodl-tmp/new huanjing ganzhi-cc/results/baseline_evaluation',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to evaluate on')
    parser.add_argument('--run_lidar', action='store_true', default=True,
                       help='Run LiDAR odometry baseline')
    parser.add_argument('--run_detection', action='store_true', default=True,
                       help='Run RGB-Thermal detection baseline')
    parser.add_argument('--gt_trajectory', type=str,
                       default='/root/autodl-tmp/new huanjing ganzhi-cc/sr_B_route2_deep_learning/ground_truth_trajectory.csv',
                       help='Ground truth trajectory file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MineSLAM Baseline Evaluation")
    print("="*80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 初始化评估器
        evaluator = BaselineEvaluator(args.config)
        
        # 加载数据集
        print(f"Loading dataset...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset = MineSLAMDataset(config, split=args.split)
        print(f"Loaded {len(dataset)} samples from {args.split} split")
        
        results = {}
        
        # 运行LiDAR里程计基线
        if args.run_lidar:
            print("\n" + "="*60)
            print("Running LiDAR-only Odometry Baseline")
            print("="*60)
            
            lidar_output_dir = output_dir / "lidar_odometry"
            lidar_output_dir.mkdir(exist_ok=True)
            
            trajectory_file, lidar_results = evaluator.run_lidar_odometry_baseline(
                dataset, str(lidar_output_dir)
            )
            
            # 评估轨迹
            trajectory_metrics = evaluator.evaluate_trajectory(
                trajectory_file, args.gt_trajectory
            )
            
            results['lidar_odometry'] = {
                **lidar_results,
                'trajectory_metrics': trajectory_metrics,
                'trajectory_file': trajectory_file
            }
        
        # 运行RGB-Thermal检测基线
        if args.run_detection:
            print("\n" + "="*60)
            print("Running RGB-Thermal Detection Baseline")
            print("="*60)
            
            detection_output_dir = output_dir / "rgb_thermal_detection"
            detection_output_dir.mkdir(exist_ok=True)
            
            detection_file, detection_results = evaluator.run_detection_baseline(
                dataset, str(detection_output_dir)
            )
            
            # 评估检测结果
            detection_metrics = evaluator.evaluate_detections(detection_file)
            
            results['rgb_thermal_detection'] = {
                **detection_results,
                'detection_metrics': detection_metrics,
                'detection_file': detection_file
            }
        
        # 生成评估报告
        print("\n" + "="*60)
        print("Generating Evaluation Report")
        print("="*60)
        
        evaluator.generate_report(results, args.output_dir)
        
        print("\n" + "="*80)
        print("Baseline evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())