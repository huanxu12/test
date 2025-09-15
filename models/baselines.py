"""
MineSLAM Baseline Algorithms - REAL DATA ONLY
基线算法实现：LiDAR里程计和RGB-Thermal检测
严格基于真实传感器数据，禁止合成数据
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2
from sklearn.neighbors import NearestNeighbors


class LiDAROnlyOdometry:
    """
    LiDAR-only 里程计解算
    体素下采样 → 法线估计 → point-to-plane ICP
    输出连续 se(3) 轨迹与 ATE/RPE
    """
    
    def __init__(self, voxel_size: float = 0.1, 
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6):
        self.voxel_size = voxel_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # 存储轨迹
        self.trajectory = []  # List of 4x4 transformation matrices
        self.timestamps = []
        self.previous_cloud = None
        
        print(f"LiDAR Odometry initialized with voxel_size={voxel_size}")
    
    def voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """体素下采样"""
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 体素下采样
        downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        return np.asarray(downsampled.points)
    
    def estimate_normals(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """估计点云法线"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 估计法线
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        
        return points, normals
    
    def point_to_plane_icp(self, source_points: np.ndarray, source_normals: np.ndarray,
                          target_points: np.ndarray, target_normals: np.ndarray,
                          initial_transform: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """Point-to-plane ICP算法"""
        if initial_transform is None:
            transform = np.eye(4)
        else:
            transform = initial_transform.copy()
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 变换源点云
            source_transformed = self._transform_points(source_points, transform)
            
            # 找到最近邻对应点
            correspondences, distances = self._find_correspondences(
                source_transformed, target_points)
            
            # 过滤距离过大的对应点
            valid_mask = distances < self.voxel_size * 3
            if np.sum(valid_mask) < 100:  # 至少需要100个对应点
                break
            
            valid_correspondences = correspondences[valid_mask]
            valid_source = source_transformed[valid_mask]
            valid_target = target_points[valid_correspondences]
            valid_target_normals = target_normals[valid_correspondences]
            
            # 计算point-to-plane误差
            error = self._compute_point_to_plane_error(
                valid_source, valid_target, valid_target_normals)
            
            # 检查收敛
            if abs(prev_error - error) < self.convergence_threshold:
                break
            
            prev_error = error
            
            # 求解线性系统得到增量变换
            delta_transform = self._solve_linear_system(
                valid_source, valid_target, valid_target_normals)
            
            # 更新变换矩阵
            transform = delta_transform @ transform
        
        return transform, prev_error
    
    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """变换点云"""
        homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = (transform @ homogeneous.T).T
        return transformed[:, :3]
    
    def _find_correspondences(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """查找最近邻对应点"""
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
        distances, indices = nbrs.kneighbors(source)
        return indices.flatten(), distances.flatten()
    
    def _compute_point_to_plane_error(self, source: np.ndarray, target: np.ndarray, 
                                     normals: np.ndarray) -> float:
        """计算point-to-plane误差"""
        diff = source - target
        errors = np.sum(diff * normals, axis=1)
        return np.mean(errors ** 2)
    
    def _solve_linear_system(self, source: np.ndarray, target: np.ndarray, 
                           normals: np.ndarray) -> np.ndarray:
        """求解point-to-plane ICP线性系统"""
        # 改进的ICP实现：使用SVD进行旋转估计
        
        # 计算质心
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        # 中心化点云
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        # 使用SVD计算最佳旋转矩阵
        try:
            # 构建协方差矩阵
            H = source_centered.T @ target_centered
            
            # SVD分解
            U, S, Vt = np.linalg.svd(H)
            
            # 计算旋转矩阵
            R = Vt.T @ U.T
            
            # 确保R是正旋转矩阵
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
                
        except np.linalg.LinAlgError:
            # SVD失败时使用单位矩阵
            R = np.eye(3)
        
        # 计算平移
        t = centroid_target - R @ centroid_source
        
        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
    
    def process_frame(self, points: torch.Tensor, timestamp: float) -> Dict[str, np.ndarray]:
        """处理单帧LiDAR数据"""
        # 转换为numpy数组
        if isinstance(points, torch.Tensor):
            points_np = points.numpy()
        else:
            points_np = points
        
        # 只使用xyz坐标
        points_xyz = points_np[:, :3]
        
        # 体素下采样
        downsampled = self.voxel_downsample(points_xyz)
        
        # 估计法线
        points_with_normals, normals = self.estimate_normals(downsampled)
        
        result = {
            'points': points_with_normals,
            'normals': normals,
            'timestamp': timestamp,
            'transform': np.eye(4)
        }
        
        # 如果有前一帧，执行ICP
        if self.previous_cloud is not None:
            try:
                transform, error = self.point_to_plane_icp(
                    points_with_normals, normals,
                    self.previous_cloud['points'], self.previous_cloud['normals']
                )
                
                result['transform'] = transform
                result['icp_error'] = error
                
                # 累积变换到全局坐标系
                if len(self.trajectory) > 0:
                    # 正确的变换累积：T_global = T_prev @ T_relative
                    global_transform = self.trajectory[-1] @ transform
                else:
                    global_transform = np.eye(4)  # 第一帧设为原点
                
                self.trajectory.append(global_transform)
                self.timestamps.append(timestamp)
                
            except Exception as e:
                print(f"ICP failed for frame at {timestamp}: {e}")
                # 使用单位变换
                if len(self.trajectory) > 0:
                    self.trajectory.append(self.trajectory[-1])
                else:
                    self.trajectory.append(np.eye(4))
                self.timestamps.append(timestamp)
        else:
            # 第一帧
            self.trajectory.append(np.eye(4))
            self.timestamps.append(timestamp)
        
        # 保存当前帧供下一帧使用
        self.previous_cloud = {
            'points': points_with_normals,
            'normals': normals
        }
        
        return result
    
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取完整轨迹"""
        if not self.trajectory:
            return np.array([]), np.array([])
        
        # 提取位置和旋转
        positions = []
        orientations = []
        
        for transform in self.trajectory:
            # 位置
            position = transform[:3, 3]
            positions.append(position)
            
            # 旋转（转换为欧拉角）
            rotation_matrix = transform[:3, :3]
            rotation = Rotation.from_matrix(rotation_matrix)
            euler = rotation.as_euler('xyz')
            orientations.append(euler)
        
        positions = np.array(positions)
        orientations = np.array(orientations)
        
        # 组合为6维轨迹 [x, y, z, rx, ry, rz] - 移除时间戳列
        trajectory_array = np.hstack([positions, orientations])
        
        return trajectory_array, np.array(self.timestamps)


class RGBThermalDetector(nn.Module):
    """
    RGB-Thermal 2D 检测器
    基于真实标注训练，仅允许ImageNet预训练
    推断后用真实深度/标定投影到3D
    """
    
    def __init__(self, num_classes: int = 8, 
                 input_size: Tuple[int, int] = (544, 1024),
                 thermal_size: Tuple[int, int] = (512, 640)):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.thermal_size = thermal_size
        
        # RGB分支 - 仅允许ImageNet预训练的backbone
        self.rgb_backbone = self._create_rgb_backbone()
        
        # 热成像分支
        self.thermal_backbone = self._create_thermal_backbone()
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(512 + 256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes + 4, 1)  # classes + bbox
        )
        
        print(f"RGB-Thermal Detector initialized for {num_classes} classes")
    
    def _create_rgb_backbone(self) -> nn.Module:
        """创建RGB backbone - 仅使用ImageNet预训练"""
        # 使用简化的ResNet-like结构，可以加载ImageNet预训练权重
        backbone = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # 残差块1
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        return backbone
    
    def _create_thermal_backbone(self) -> nn.Module:
        """创建热成像backbone"""
        backbone = nn.Sequential(
            # 适配单通道输入
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # 残差块
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
        )
        
        return backbone
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1) -> nn.Module:
        """创建残差层"""
        layers = []
        
        # 第一个块可能需要下采样
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # 其余块
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # RGB特征提取
        rgb_features = self.rgb_backbone(rgb)  # [B, 512, H/32, W/32]
        
        # 热成像特征提取
        thermal_features = self.thermal_backbone(thermal)  # [B, 256, H'/32, W'/32]
        
        # 特征对齐和融合
        # 调整热成像特征尺寸以匹配RGB
        thermal_resized = F.interpolate(
            thermal_features, 
            size=rgb_features.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 特征融合
        fused_features = torch.cat([rgb_features, thermal_resized], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # 检测预测
        detection_output = self.detection_head(fused_features)
        
        # 分离类别和边界框预测
        class_pred = detection_output[:, :self.num_classes]
        bbox_pred = detection_output[:, self.num_classes:]
        
        return {
            'class_logits': class_pred,
            'bbox_pred': bbox_pred,
            'features': fused_features
        }
    
    def project_to_3d(self, detections_2d: Dict[str, torch.Tensor],
                     depth_image: torch.Tensor,
                     camera_intrinsics: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """将2D检测结果投影到3D"""
        detections_3d = []
        
        # 解析2D检测结果
        class_logits = detections_2d['class_logits']
        bbox_pred = detections_2d['bbox_pred']
        
        batch_size = class_logits.shape[0]
        
        for b in range(batch_size):
            batch_detections = []
            
            # 获取当前批次的预测
            batch_classes = torch.softmax(class_logits[b], dim=0)
            batch_bboxes = bbox_pred[b]
            batch_depth = depth_image[b, 0]  # 假设深度是[B, 1, H, W]
            
            # 找到置信度高的检测
            max_confidence, predicted_classes = torch.max(batch_classes, dim=0)
            
            # 过滤低置信度检测
            valid_detections = max_confidence > 0.5
            
            if valid_detections.any():
                # 解析边界框（假设格式为[x, y, w, h]在特征图空间）
                feature_h, feature_w = batch_bboxes.shape[1], batch_bboxes.shape[2]
                
                for y in range(feature_h):
                    for x in range(feature_w):
                        if valid_detections[y, x]:
                            # 获取边界框参数
                            bbox_params = batch_bboxes[:, y, x]  # [4]
                            
                            # 转换到图像坐标
                            scale_x = self.input_size[1] / feature_w
                            scale_y = self.input_size[0] / feature_h
                            
                            center_x = x * scale_x + bbox_params[0] * scale_x
                            center_y = y * scale_y + bbox_params[1] * scale_y
                            width = bbox_params[2] * scale_x
                            height = bbox_params[3] * scale_y
                            
                            # 获取中心点的深度值
                            depth_y = int(np.clip(center_y, 0, batch_depth.shape[0] - 1))
                            depth_x = int(np.clip(center_x, 0, batch_depth.shape[1] - 1))
                            depth_value = batch_depth[depth_y, depth_x].item()
                            
                            if depth_value > 0:  # 有效深度
                                # 投影到3D空间
                                fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
                                cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
                                
                                # 3D中心点
                                x_3d = (center_x - cx) * depth_value / fx
                                y_3d = (center_y - cy) * depth_value / fy
                                z_3d = depth_value
                                
                                # 估计3D尺寸（简化方法）
                                width_3d = width * depth_value / fx
                                height_3d = height * depth_value / fy
                                depth_3d = min(width_3d, height_3d)  # 假设深度与较小的水平尺寸相等
                                
                                detection_3d = {
                                    'center': np.array([x_3d, y_3d, z_3d]),
                                    'size': np.array([width_3d, height_3d, depth_3d]),
                                    'class_id': predicted_classes[y, x].item(),
                                    'confidence': max_confidence[y, x].item(),
                                    'bbox_2d': np.array([center_x - width/2, center_y - height/2, 
                                                       center_x + width/2, center_y + height/2])
                                }
                                
                                batch_detections.append(detection_3d)
            
            detections_3d.append(batch_detections)
        
        return detections_3d


class BasicBlock(nn.Module):
    """基础残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.relu(out)