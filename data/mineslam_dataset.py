"""
MineSLAM Dataset Implementation - REAL DATA ONLY
严格基于真实传感器数据的数据集实现，禁止任何合成/随机数据生成
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import bisect
from pathlib import Path
import warnings


class RealDataContract:
    """真实数据契约定义"""
    
    @staticmethod
    def validate_rgb_tensor(rgb: torch.Tensor) -> bool:
        """验证RGB张量契约: (3,H,W) in [0,1]"""
        if rgb.dim() != 3 or rgb.shape[0] != 3:
            raise ValueError(f"RGB must be (3,H,W), got {rgb.shape}")
        if not (0 <= rgb.min() and rgb.max() <= 1):
            raise ValueError(f"RGB values must be in [0,1], got range [{rgb.min():.3f}, {rgb.max():.3f}]")
        return True
    
    @staticmethod
    def validate_depth_tensor(depth: torch.Tensor) -> bool:
        """验证深度张量契约: (1,H,W) in meters"""
        if depth.dim() != 3 or depth.shape[0] != 1:
            raise ValueError(f"Depth must be (1,H,W), got {depth.shape}")
        if depth.min() < 0:
            raise ValueError(f"Depth values must be >= 0, got min {depth.min():.3f}")
        return True
    
    @staticmethod
    def validate_thermal_tensor(thermal: torch.Tensor) -> bool:
        """验证热成像张量契约: (1,H,W) normalized"""
        if thermal.dim() != 3 or thermal.shape[0] != 1:
            raise ValueError(f"Thermal must be (1,H,W), got {thermal.shape}")
        return True
    
    @staticmethod
    def validate_lidar_tensor(lidar: torch.Tensor) -> bool:
        """验证激光雷达张量契约: (N,4) = [x,y,z,intensity]"""
        if lidar.dim() != 2 or lidar.shape[1] != 4:
            raise ValueError(f"LiDAR must be (N,4), got {lidar.shape}")
        return True
    
    @staticmethod
    def validate_imu_tensor(imu: torch.Tensor) -> bool:
        """验证IMU张量契约: (T,6) = [ax,ay,az,gx,gy,gz]"""
        if imu.dim() != 2 or imu.shape[1] != 6:
            raise ValueError(f"IMU must be (T,6), got {imu.shape}")
        if imu.shape[0] == 0:
            raise ValueError("IMU sequence cannot be empty")
        return True
    
    @staticmethod
    def validate_pose_delta_tensor(pose_delta: torch.Tensor) -> bool:
        """验证位姿增量张量契约: (6,) = [tx,ty,tz,rx,ry,rz]"""
        if pose_delta.dim() != 1 or pose_delta.shape[0] != 6:
            raise ValueError(f"Pose delta must be (6,), got {pose_delta.shape}")
        return True
    
    @staticmethod
    def validate_boxes_tensor(boxes: torch.Tensor) -> bool:
        """验证检测框张量契约: (Q,10)，无目标时conf=-1"""
        if boxes.dim() != 2 or boxes.shape[1] != 10:
            raise ValueError(f"Boxes must be (Q,10), got {boxes.shape}")
        return True


class RealSensorDataLoader:
    """真实传感器数据加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root_path = config['data']['root']
        
        # 验证所有数据路径存在
        self._validate_real_data_paths()
        
        # 加载标定数据
        self.calibration = self._load_real_calibration()
        
        # 时间同步参数
        self.imu_window_size = 20  # IMU滑窗长度
        self.time_threshold_ms = 100  # 时间对齐阈值（毫秒）
        
        # 统计信息
        self.missing_frame_count = 0
        self.total_frame_count = 0
    
    def _validate_real_data_paths(self):
        """验证所有真实数据路径存在"""
        required_paths = [
            self.config['data']['images']['thermal'],
            self.config['data']['images']['rgb_left'],
            self.config['data']['images']['depth'],
            self.config['data']['pointclouds']['ouster'],
            self.config['data']['imu'],
            self.config['data']['ground_truth']['trajectory']
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"REAL DATA PATH MISSING: {path}")
    
    def _load_real_calibration(self) -> Dict[str, Any]:
        """加载真实标定数据"""
        calib_path = self.config['calib']['source_calibration']
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        # 验证标定矩阵
        self._validate_calibration_matrices(calib_data)
        
        return calib_data
    
    def _validate_calibration_matrices(self, calib_data: Dict[str, Any]):
        """验证标定矩阵的有效性"""
        for camera_name, camera_info in calib_data.items():
            if 'K' in camera_info and 'R' in camera_info:
                K = np.array(camera_info['K']).reshape(3, 3)
                R = np.array(camera_info['R']).reshape(3, 3)
                
                # 检查内参矩阵可逆性
                if np.linalg.det(K) == 0:
                    raise ValueError(f"Intrinsic matrix K is singular for {camera_name}")
                
                # 检查旋转矩阵正交性
                if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6):
                    raise ValueError(f"Rotation matrix R is not orthogonal for {camera_name}")
                
                # 检查行列式约等于1
                det_R = np.linalg.det(R)
                if not np.isclose(det_R, 1.0, atol=1e-6):
                    raise ValueError(f"Rotation matrix determinant is {det_R}, expected ~1.0 for {camera_name}")
    
    def load_real_rgb_image(self, image_path: str) -> torch.Tensor:
        """加载真实RGB图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"RGB image not found: {image_path}")
        
        # 使用OpenCV加载图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load RGB image: {image_path}")
        
        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为张量并归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # 验证契约
        RealDataContract.validate_rgb_tensor(image_tensor)
        
        return image_tensor
    
    def load_real_depth_image(self, depth_path: str) -> torch.Tensor:
        """加载真实深度图像"""
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        
        # 加载深度图像（假设为16位PNG）
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to load depth image: {depth_path}")
        
        # 转换为米为单位（根据实际标定调整比例因子）
        depth_scale = 0.001  # 毫米转米
        depth = depth.astype(np.float32) * depth_scale
        
        # 转换为张量
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)  # 添加通道维度
        
        # 验证契约
        RealDataContract.validate_depth_tensor(depth_tensor)
        
        return depth_tensor
    
    def load_real_thermal_image(self, thermal_path: str) -> torch.Tensor:
        """加载真实热成像图像"""
        if not os.path.exists(thermal_path):
            raise FileNotFoundError(f"Thermal image not found: {thermal_path}")
        
        # 加载热成像图像
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        if thermal is None:
            raise ValueError(f"Failed to load thermal image: {thermal_path}")
        
        # 归一化处理
        thermal = thermal.astype(np.float32)
        thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
        
        # 转换为张量
        thermal_tensor = torch.from_numpy(thermal).unsqueeze(0)  # 添加通道维度
        
        # 验证契约
        RealDataContract.validate_thermal_tensor(thermal_tensor)
        
        return thermal_tensor
    
    def load_real_lidar_data(self, lidar_path: str) -> torch.Tensor:
        """加载真实激光雷达数据"""
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
        
        try:
            # 尝试加载不同格式的点云文件
            if lidar_path.endswith('.bin'):
                # 二进制格式 - Ouster LiDAR使用这种格式
                # 每个点包含 [x, y, z, intensity] = 4个float32值
                points = np.fromfile(lidar_path, dtype=np.float32)
                
                # 检查数据长度是否为4的倍数
                if len(points) % 4 != 0:
                    raise ValueError(f"Invalid binary LiDAR file format: {lidar_path} (length: {len(points)})")
                
                # 重塑为(N, 4)格式
                points = points.reshape(-1, 4)
                
            elif lidar_path.endswith('.txt'):
                # 文本格式
                points = np.loadtxt(lidar_path, dtype=np.float32)
            elif lidar_path.endswith('.pcd'):
                # PCD格式 - 如果需要的话
                raise NotImplementedError("PCD format not implemented yet")
            else:
                raise ValueError(f"Unsupported LiDAR file format: {lidar_path}")
            
            # 确保点云数据格式正确
            if len(points.shape) != 2:
                raise ValueError(f"LiDAR data must be 2D array, got shape: {points.shape}")
            
            if points.shape[1] < 3:
                raise ValueError(f"LiDAR data must have at least 3 columns [x,y,z], got {points.shape[1]}")
            
            # 如果只有3列，添加强度列（全1）
            if points.shape[1] == 3:
                intensity = np.ones((points.shape[0], 1), dtype=np.float32)
                points = np.concatenate([points, intensity], axis=1)
            elif points.shape[1] > 4:
                # 如果超过4列，只取前4列
                points = points[:, :4]
            
            # 过滤无效点（所有坐标为0或nan的点）
            valid_mask = ~(np.all(points[:, :3] == 0, axis=1) | np.any(np.isnan(points), axis=1))
            points = points[valid_mask]
            
            if len(points) == 0:
                raise ValueError(f"No valid points found in LiDAR file: {lidar_path}")
            
            # 转换为张量
            lidar_tensor = torch.from_numpy(points.astype(np.float32))
            
            # 验证契约
            RealDataContract.validate_lidar_tensor(lidar_tensor)
            
            return lidar_tensor
            
        except Exception as e:
            raise ValueError(f"Failed to load LiDAR data from {lidar_path}: {e}")
    
    def load_real_imu_sequence(self, imu_data: pd.DataFrame, 
                              t_start: float, t_end: float) -> torch.Tensor:
        """加载真实IMU序列，基于时间窗口切片"""
        # 基于真实时间戳筛选IMU数据
        mask = (imu_data['timestamp'] >= t_start) & (imu_data['timestamp'] <= t_end)
        imu_window = imu_data[mask]
        
        if len(imu_window) == 0:
            raise ValueError(f"No IMU data found in time window [{t_start}, {t_end}]")
        
        # 提取6维IMU数据 [ax,ay,az,gx,gy,gz]
        imu_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # 检查列是否存在
        missing_cols = [col for col in imu_columns if col not in imu_window.columns]
        if missing_cols:
            raise ValueError(f"Missing IMU columns: {missing_cols}")
        
        imu_values = imu_window[imu_columns].values.astype(np.float32)
        
        # 如果数据量不足滑窗大小，进行线性插值
        if len(imu_values) < self.imu_window_size:
            warnings.warn(f"IMU sequence length {len(imu_values)} < window size {self.imu_window_size}, "
                         f"using interpolation")
            
            # 时间戳插值
            time_range = np.linspace(t_start, t_end, self.imu_window_size)
            imu_interp = np.zeros((self.imu_window_size, 6), dtype=np.float32)
            
            for i in range(6):
                imu_interp[:, i] = np.interp(time_range, imu_window['timestamp'].values, 
                                           imu_values[:, i])
            
            imu_values = imu_interp
        elif len(imu_values) > self.imu_window_size:
            # 等间隔采样到目标长度
            indices = np.linspace(0, len(imu_values)-1, self.imu_window_size, dtype=int)
            imu_values = imu_values[indices]
        
        # 转换为张量
        imu_tensor = torch.from_numpy(imu_values)
        
        # 验证契约
        RealDataContract.validate_imu_tensor(imu_tensor)
        
        return imu_tensor
    
    def apply_extrinsic_calibration(self, points: torch.Tensor, 
                                   from_frame: str, to_frame: str = 'lidar') -> torch.Tensor:
        """应用外参标定，将坐标统一到LiDAR坐标系"""
        if from_frame == to_frame:
            return points
        
        # 这里应该根据实际标定数据实现坐标变换
        # 暂时返回原始数据，实际实现需要根据标定矩阵进行变换
        return points
    
    def find_temporal_matches(self, target_timestamp: float, 
                             timestamps: List[float], 
                             threshold_ms: float = None) -> Tuple[int, float]:
        """查找时间戳最近邻匹配"""
        if threshold_ms is None:
            threshold_ms = self.time_threshold_ms
        
        threshold_s = threshold_ms / 1000.0
        
        # 使用二分查找找到最近的时间戳
        idx = bisect.bisect_left(timestamps, target_timestamp)
        
        best_idx = -1
        best_diff = float('inf')
        
        # 检查左右两个候选
        for candidate_idx in [idx-1, idx]:
            if 0 <= candidate_idx < len(timestamps):
                diff = abs(timestamps[candidate_idx] - target_timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = candidate_idx
        
        if best_diff > threshold_s:
            raise ValueError(f"No timestamp match within {threshold_ms}ms threshold. "
                           f"Best match: {best_diff*1000:.1f}ms")
        
        return best_idx, best_diff


class MineSLAMDataset(Dataset):
    """
    MineSLAM数据集 - 仅使用真实传感器数据
    严格禁止任何合成/随机数据生成
    """
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        self.config = config
        self.split = split
        self.data_loader = RealSensorDataLoader(config)
        
        # 加载真实数据索引
        self.samples = self._load_real_data_index()
        
        # 加载地面真值轨迹
        self.ground_truth_poses = self._load_ground_truth_trajectory()
        
        # 加载IMU数据
        self.imu_data = self._load_imu_data()
        
        # 时间戳列表（用于快速查找）
        self._build_timestamp_indices()
        
        print(f"Loaded {len(self.samples)} real samples for {split} split")
        print(f"Missing frame rate: {self.data_loader.missing_frame_count / max(1, self.data_loader.total_frame_count) * 100:.2f}%")
    
    def _load_real_data_index(self) -> List[Dict[str, Any]]:
        """从真实jsonl文件加载数据索引"""
        index_path = self.config['data'][f'{self.split}_index']
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"CRITICAL: Real data index not found: {index_path}")
        
        samples = []
        with open(index_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    sample = json.loads(line.strip())
                    
                    # 验证样本包含必要字段
                    required_fields = ['timestamp', 'rgb_image', 'thermal_image']
                    missing_fields = [field for field in required_fields if field not in sample]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                    
                    # 验证所有文件路径存在
                    file_paths = [
                        sample.get('rgb_image'),
                        sample.get('thermal_image'),
                        sample.get('depth_image'),
                        sample.get('lidar_file')
                    ]
                    
                    missing_files = []
                    for file_path in file_paths:
                        if file_path and not os.path.exists(file_path):
                            missing_files.append(file_path)
                    
                    if missing_files:
                        self.data_loader.missing_frame_count += 1
                        warnings.warn(f"Missing files in sample {line_num}: {missing_files}")
                        continue
                    
                    samples.append(sample)
                    self.data_loader.total_frame_count += 1
                    
                except (json.JSONDecodeError, ValueError) as e:
                    warnings.warn(f"Invalid sample at line {line_num}: {e}")
                    continue
        
        if not samples:
            raise ValueError(f"No valid real samples found in {index_path}")
        
        return samples
    
    def _load_ground_truth_trajectory(self) -> pd.DataFrame:
        """加载真实地面真值轨迹"""
        gt_path = self.config['data']['ground_truth']['trajectory']
        
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth trajectory not found: {gt_path}")
        
        gt_df = pd.read_csv(gt_path)
        
        # 验证必要列存在
        required_columns = ['timestamp', 'position_x', 'position_y', 'position_z', 
                          'orientation_roll', 'orientation_pitch', 'orientation_yaw']
        missing_cols = [col for col in required_columns if col not in gt_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ground truth: {missing_cols}")
        
        # 按时间戳排序
        gt_df = gt_df.sort_values('timestamp').reset_index(drop=True)
        
        return gt_df
    
    def _load_imu_data(self) -> pd.DataFrame:
        """加载真实IMU数据"""
        imu_dir = self.config['data']['imu']
        
        # 查找IMU文件 - 使用main_imu.csv作为主要IMU数据源
        main_imu_file = os.path.join(imu_dir, 'main_imu.csv')
        
        if not os.path.exists(main_imu_file):
            # 如果main_imu.csv不存在，尝试其他文件
            imu_files = []
            for ext in ['.csv', '.txt']:
                imu_files.extend(Path(imu_dir).glob(f'*{ext}'))
            
            if not imu_files:
                raise FileNotFoundError(f"No IMU files found in {imu_dir}")
            
            main_imu_file = str(imu_files[0])
        
        print(f"Loading IMU data from: {main_imu_file}")
        imu_df = pd.read_csv(main_imu_file)
        
        # 检查实际的IMU数据列格式
        print(f"IMU columns: {list(imu_df.columns)}")
        
        # 根据实际数据格式映射列名
        # 实际格式: timestamp,orientation_x,orientation_y,orientation_z,orientation_w,angular_velocity_x,angular_velocity_y,angular_velocity_z,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z
        column_mapping = {
            'linear_acceleration_x': 'accel_x',
            'linear_acceleration_y': 'accel_y', 
            'linear_acceleration_z': 'accel_z',
            'angular_velocity_x': 'gyro_x',
            'angular_velocity_y': 'gyro_y',
            'angular_velocity_z': 'gyro_z'
        }
        
        # 重命名列以匹配预期格式
        for old_name, new_name in column_mapping.items():
            if old_name in imu_df.columns:
                imu_df.rename(columns={old_name: new_name}, inplace=True)
        
        # 验证重命名后的IMU数据列
        required_imu_cols = ['timestamp', 'accel_x', 'accel_y', 'accel_z', 
                           'gyro_x', 'gyro_y', 'gyro_z']
        missing_cols = [col for col in required_imu_cols if col not in imu_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required IMU columns after mapping: {missing_cols}")
        
        # 按时间戳排序
        imu_df = imu_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(imu_df)} IMU measurements")
        
        return imu_df
    
    def _build_timestamp_indices(self):
        """构建时间戳索引用于快速查找"""
        self.gt_timestamps = self.ground_truth_poses['timestamp'].tolist()
        self.imu_timestamps = self.imu_data['timestamp'].tolist()
    
    def _get_pose_delta(self, current_timestamp: float) -> torch.Tensor:
        """获取位姿增量（相对于前一帧）"""
        try:
            # 找到当前时间戳对应的位姿
            current_idx, _ = self.data_loader.find_temporal_matches(
                current_timestamp, self.gt_timestamps)
            
            if current_idx == 0:
                # 第一帧，返回零增量
                pose_delta = torch.zeros(6, dtype=torch.float32)
            else:
                # 计算相对于前一帧的位姿增量
                current_pose = self.ground_truth_poses.iloc[current_idx]
                prev_pose = self.ground_truth_poses.iloc[current_idx - 1]
                
                # 位置增量
                pos_delta = np.array([
                    current_pose['position_x'] - prev_pose['position_x'],
                    current_pose['position_y'] - prev_pose['position_y'],
                    current_pose['position_z'] - prev_pose['position_z']
                ], dtype=np.float32)
                
                # 旋转增量
                rot_delta = np.array([
                    current_pose['orientation_roll'] - prev_pose['orientation_roll'],
                    current_pose['orientation_pitch'] - prev_pose['orientation_pitch'],
                    current_pose['orientation_yaw'] - prev_pose['orientation_yaw']
                ], dtype=np.float32)
                
                pose_delta = torch.from_numpy(np.concatenate([pos_delta, rot_delta]))
            
            # 验证契约
            RealDataContract.validate_pose_delta_tensor(pose_delta)
            
            return pose_delta
            
        except Exception as e:
            raise ValueError(f"Failed to get pose delta for timestamp {current_timestamp}: {e}")
    
    def _get_detection_boxes(self, sample: Dict[str, Any]) -> torch.Tensor:
        """获取检测框标注（Q,10格式）"""
        # 如果样本中没有检测框标注，返回无目标的标记
        if 'detections' not in sample or not sample['detections']:
            # 无目标时返回单个框，confidence=-1
            boxes = torch.tensor([[-1.0] * 10], dtype=torch.float32)
        else:
            detections = sample['detections']
            boxes_list = []
            
            for det in detections:
                # 假设检测框格式为 [x, y, z, w, h, l, class_id, confidence, attr1, attr2]
                box = [
                    det.get('x', 0.0),
                    det.get('y', 0.0), 
                    det.get('z', 0.0),
                    det.get('width', 0.0),
                    det.get('height', 0.0),
                    det.get('length', 0.0),
                    det.get('class_id', 0.0),
                    det.get('confidence', 0.0),
                    det.get('attr1', 0.0),
                    det.get('attr2', 0.0)
                ]
                boxes_list.append(box)
            
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
        
        # 验证契约
        RealDataContract.validate_boxes_tensor(boxes)
        
        return boxes
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个真实数据样本
        
        严格禁止任何合成/随机数据生成
        """
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self.samples)})")
        
        sample = self.samples[idx]
        timestamp = sample['timestamp']
        
        try:
            # 检查是否有任何自动生成/随机替代的迹象
            if any(key.startswith('_generated') or key.startswith('_synthetic') 
                   for key in sample.keys()):
                raise ValueError("FORBIDDEN: Detected synthetic/generated data markers in sample")
            
            # 加载真实传感器数据
            data_dict = {}
            
            # 1. RGB图像
            if 'rgb_image' in sample:
                data_dict['rgb'] = self.data_loader.load_real_rgb_image(sample['rgb_image'])
            
            # 2. 深度图像
            if 'depth_image' in sample:
                data_dict['depth'] = self.data_loader.load_real_depth_image(sample['depth_image'])
            
            # 3. 热成像图像
            if 'thermal_image' in sample:
                data_dict['thermal'] = self.data_loader.load_real_thermal_image(sample['thermal_image'])
            
            # 4. 激光雷达数据
            if 'lidar_file' in sample:
                data_dict['lidar'] = self.data_loader.load_real_lidar_data(sample['lidar_file'])
            
            # 5. IMU数据（基于时间窗口）
            imu_window_half = 0.5  # 半秒窗口
            t_imu_start = timestamp - imu_window_half
            t_imu_end = timestamp + imu_window_half
            
            data_dict['imu'] = self.data_loader.load_real_imu_sequence(
                self.imu_data, t_imu_start, t_imu_end)
            
            # 6. 位姿增量
            data_dict['pose_delta'] = self._get_pose_delta(timestamp)
            
            # 7. 检测框
            data_dict['boxes'] = self._get_detection_boxes(sample)
            
            # 8. 元数据
            data_dict['timestamp'] = torch.tensor(timestamp, dtype=torch.float64)
            data_dict['sample_idx'] = torch.tensor(idx, dtype=torch.long)
            
            # 验证所有数据契约
            self._validate_sample_contracts(data_dict)
            
            # 检测缺帧情况
            missing_modalities = []
            expected_modalities = ['rgb', 'thermal', 'lidar', 'imu']
            for mod in expected_modalities:
                if mod not in data_dict:
                    missing_modalities.append(mod)
            
            if missing_modalities:
                missing_ratio = len(missing_modalities) / len(expected_modalities)
                warnings.warn(f"Sample {idx}: Missing modalities {missing_modalities}, "
                             f"missing ratio: {missing_ratio:.2f}")
            
            return data_dict
            
        except Exception as e:
            raise RuntimeError(f"Failed to load real data sample {idx} (timestamp: {timestamp}): {e}")
    
    def _validate_sample_contracts(self, data_dict: Dict[str, torch.Tensor]):
        """验证样本数据契约"""
        contract_validators = {
            'rgb': RealDataContract.validate_rgb_tensor,
            'depth': RealDataContract.validate_depth_tensor,
            'thermal': RealDataContract.validate_thermal_tensor,
            'lidar': RealDataContract.validate_lidar_tensor,
            'imu': RealDataContract.validate_imu_tensor,
            'pose_delta': RealDataContract.validate_pose_delta_tensor,
            'boxes': RealDataContract.validate_boxes_tensor
        }
        
        for key, validator in contract_validators.items():
            if key in data_dict:
                try:
                    validator(data_dict[key])
                except Exception as e:
                    raise ValueError(f"Contract validation failed for {key}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        total_samples = len(self.samples)
        missing_rate = self.data_loader.missing_frame_count / max(1, self.data_loader.total_frame_count)
        
        return {
            'total_samples': total_samples,
            'missing_frame_count': self.data_loader.missing_frame_count,
            'missing_frame_rate': missing_rate,
            'split': self.split,
            'imu_window_size': self.data_loader.imu_window_size,
            'time_threshold_ms': self.data_loader.time_threshold_ms
        }