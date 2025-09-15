"""
LiDAR-only Odometry Baseline
基于点云的里程计算法，包含体素下采样、法线估计和point-to-plane ICP
"""

import numpy as np
import torch
import open3d as o3d
from typing import Tuple, List, Optional, Dict
import time
from scipy.spatial.transform import Rotation
from pathlib import Path
import json


class LiDAROdometry:
    """
    LiDAR-only里程计基线算法
    实现简化的LOAM前端：体素下采样 + 法线估计 + point-to-plane ICP
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 算法参数
        self.voxel_size = config.get('voxel_size', 0.3)  # 体素下采样大小
        self.max_correspondence_distance = config.get('max_correspondence_distance', 1.0)
        self.normal_radius = config.get('normal_radius', 0.5)  # 法线估计半径
        self.normal_max_nn = config.get('normal_max_nn', 30)  # 法线估计最大邻居数
        self.icp_max_iteration = config.get('icp_max_iteration', 50)
        self.icp_convergence_threshold = config.get('icp_convergence_threshold', 1e-6)
        
        # 状态变量
        self.previous_cloud = None
        self.current_pose = np.eye(4)  # 4x4变换矩阵
        self.trajectory = []  # 存储轨迹
        self.timestamps = []  # 存储时间戳
        
        print(f"LiDAR Odometry initialized with voxel_size={self.voxel_size}")
    
    def preprocess_pointcloud(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        """
        点云预处理：体素下采样和法线估计
        
        Args:
            points: 原始点云数据 [N, 3] 或 [N, 4]
        
        Returns:
            预处理后的点云对象
        """
        # 转换为Open3D点云格式
        if points.shape[1] >= 3:
            xyz = points[:, :3]  # 只取xyz坐标
        else:
            raise ValueError(f"Point cloud must have at least 3 coordinates, got {points.shape[1]}")
        
        # 过滤无效点
        valid_mask = ~np.any(np.isnan(xyz), axis=1)
        xyz = xyz[valid_mask]
        
        # 距离过滤（移除过近和过远的点）
        distances = np.linalg.norm(xyz, axis=1)
        distance_mask = (distances > 0.5) & (distances < 100.0)
        xyz = xyz[distance_mask]
        
        if len(xyz) < 100:
            raise ValueError(f"Too few valid points after filtering: {len(xyz)}")
        
        # 创建Open3D点云
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        
        # 体素下采样
        cloud_downsampled = cloud.voxel_down_sample(voxel_size=self.voxel_size)
        
        # 估计法线
        cloud_downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )
        
        # 法线方向统一（朝向原点）
        cloud_downsampled.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        
        return cloud_downsampled
    
    def point_to_plane_icp(self, source: o3d.geometry.PointCloud, 
                          target: o3d.geometry.PointCloud) -> Tuple[np.ndarray, float]:
        """
        Point-to-plane ICP配准
        
        Args:
            source: 源点云
            target: 目标点云
        
        Returns:
            transformation: 4x4变换矩阵
            fitness: 配准质量评分
        """
        # 初始变换矩阵
        init_transformation = np.eye(4)
        
        # Point-to-plane ICP
        reg_p2plane = o3d.pipelines.registration.registration_icp(
            source, target, 
            max_correspondence_distance=self.max_correspondence_distance,
            init=init_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iteration,
                relative_fitness=self.icp_convergence_threshold,
                relative_rmse=self.icp_convergence_threshold
            )
        )
        
        return reg_p2plane.transformation, reg_p2plane.fitness
    
    def process_frame(self, points: np.ndarray, timestamp: float) -> Dict:
        """
        处理单帧点云数据
        
        Args:
            points: 点云数据 [N, 4] (x, y, z, intensity)
            timestamp: 时间戳
        
        Returns:
            处理结果字典
        """
        start_time = time.time()
        
        try:
            # 预处理点云
            current_cloud = self.preprocess_pointcloud(points)
            
            result = {
                'timestamp': timestamp,
                'pose': self.current_pose.copy(),
                'num_points_raw': len(points),
                'num_points_processed': len(current_cloud.points),
                'processing_time': 0.0,
                'icp_fitness': 0.0,
                'success': True
            }
            
            # 如果是第一帧，直接保存
            if self.previous_cloud is None:
                self.previous_cloud = current_cloud
                self.trajectory.append(self.current_pose.copy())
                self.timestamps.append(timestamp)
                result['processing_time'] = time.time() - start_time
                return result
            
            # ICP配准
            transformation, fitness = self.point_to_plane_icp(current_cloud, self.previous_cloud)
            
            # 更新位姿
            self.current_pose = self.current_pose @ transformation
            
            # 保存轨迹
            self.trajectory.append(self.current_pose.copy())
            self.timestamps.append(timestamp)
            
            # 更新前一帧
            self.previous_cloud = current_cloud
            
            # 填充结果
            result['pose'] = self.current_pose.copy()
            result['icp_fitness'] = fitness
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            print(f"Error processing frame at timestamp {timestamp}: {e}")
            
            # 返回失败结果
            result = {
                'timestamp': timestamp,
                'pose': self.current_pose.copy(),
                'num_points_raw': len(points),
                'num_points_processed': 0,
                'processing_time': time.time() - start_time,
                'icp_fitness': 0.0,
                'success': False,
                'error': str(e)
            }
            
            return result
    
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取完整轨迹
        
        Returns:
            poses: [N, 4, 4] 位姿矩阵数组
            timestamps: [N] 时间戳数组
        """
        poses = np.array(self.trajectory)
        timestamps = np.array(self.timestamps)
        
        return poses, timestamps
    
    def get_trajectory_positions(self) -> np.ndarray:
        """
        获取轨迹位置序列
        
        Returns:
            positions: [N, 3] 位置坐标
        """
        if not self.trajectory:
            return np.empty((0, 3))
        
        positions = np.array([pose[:3, 3] for pose in self.trajectory])
        return positions
    
    def get_trajectory_orientations(self) -> np.ndarray:
        """
        获取轨迹姿态序列（欧拉角）
        
        Returns:
            orientations: [N, 3] 欧拉角 (roll, pitch, yaw)
        """
        if not self.trajectory:
            return np.empty((0, 3))
        
        orientations = []
        for pose in self.trajectory:
            rotation_matrix = pose[:3, :3]
            euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
            orientations.append(euler)
        
        return np.array(orientations)
    
    def save_trajectory(self, output_path: str):
        """
        保存轨迹到CSV文件
        
        Args:
            output_path: 输出文件路径
        """
        if not self.trajectory:
            print("No trajectory to save")
            return
        
        positions = self.get_trajectory_positions()
        orientations = self.get_trajectory_orientations()
        
        # 创建CSV内容
        import pandas as pd
        
        data = {
            'timestamp': self.timestamps,
            'position_x': positions[:, 0],
            'position_y': positions[:, 1], 
            'position_z': positions[:, 2],
            'orientation_roll': orientations[:, 0],
            'orientation_pitch': orientations[:, 1],
            'orientation_yaw': orientations[:, 2]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"Trajectory saved to: {output_path}")
    
    def reset(self):
        """重置里程计状态"""
        self.previous_cloud = None
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.timestamps = []


def run_lidar_odometry_on_dataset(dataset, config: Dict, output_dir: str):
    """
    在数据集上运行LiDAR里程计
    
    Args:
        dataset: MineSLAM数据集对象
        config: 算法配置
        output_dir: 输出目录
    """
    print("="*60)
    print("Running LiDAR-only Odometry Baseline")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建里程计算法实例
    odometry = LiDAROdometry(config)
    
    # 处理统计
    total_frames = 0
    successful_frames = 0
    total_processing_time = 0.0
    
    # 逐帧处理
    print(f"Processing {len(dataset)} frames...")
    
    for i, sample in enumerate(dataset):
        if 'lidar' not in sample:
            print(f"Frame {i}: No LiDAR data, skipping")
            continue
        
        # 提取数据
        lidar_points = sample['lidar'].numpy()  # [N, 4]
        timestamp = sample['timestamp'].item()
        
        # 处理帧
        result = odometry.process_frame(lidar_points, timestamp)
        
        total_frames += 1
        if result['success']:
            successful_frames += 1
        
        total_processing_time += result['processing_time']
        
        # 打印进度
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} frames, "
                  f"Success rate: {successful_frames/max(1,total_frames)*100:.1f}%")
    
    # 保存结果
    poses, timestamps = odometry.get_trajectory()
    
    if len(poses) == 0:
        print("No valid trajectory generated")
        return
    
    # 保存轨迹
    trajectory_file = output_dir / "lidar_odometry_trajectory.csv"
    odometry.save_trajectory(str(trajectory_file))
    
    # 保存详细结果
    results = {
        'algorithm': 'LiDAR-only Odometry',
        'config': config,
        'statistics': {
            'total_frames': total_frames,
            'successful_frames': successful_frames,
            'success_rate': successful_frames / max(1, total_frames),
            'avg_processing_time': total_processing_time / max(1, total_frames),
            'total_processing_time': total_processing_time,
            'trajectory_length': len(poses)
        }
    }
    
    results_file = output_dir / "lidar_odometry_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print("LiDAR Odometry Results")
    print("="*60)
    print(f"Total frames processed: {total_frames}")
    print(f"Successful frames: {successful_frames}")
    print(f"Success rate: {successful_frames/max(1,total_frames)*100:.1f}%")
    print(f"Average processing time: {total_processing_time/max(1,total_frames)*1000:.1f} ms/frame")
    print(f"Trajectory length: {len(poses)} poses")
    print(f"Results saved to: {output_dir}")
    
    return str(trajectory_file), results