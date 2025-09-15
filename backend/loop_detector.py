"""
Loop Detection Module for MineSLAM
回环检测模块：基于NetVLAD或简化全局描述符的关键帧相似度检测
严格使用真实传感器数据，防止合成图像或伪造点云
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from pathlib import Path
import cv2
import faiss


@dataclass
class KeyFrame:
    """关键帧数据结构"""
    frame_id: int
    timestamp: float
    rgb_image: np.ndarray  # H×W×3
    depth_image: Optional[np.ndarray] = None  # H×W
    thermal_image: Optional[np.ndarray] = None  # H×W
    lidar_points: Optional[np.ndarray] = None  # N×4
    pose: Optional[np.ndarray] = None  # 6DoF [x,y,z,rx,ry,rz]
    global_descriptor: Optional[np.ndarray] = None
    data_source: str = "real_sensors"  # 必须是真实传感器


@dataclass
class LoopCandidate:
    """回环候选结果"""
    query_id: int
    match_id: int
    similarity_score: float
    descriptor_distance: float
    geometric_verification: bool = False
    relative_pose: Optional[np.ndarray] = None


class RealDataValidator:
    """真实数据验证器 - 防止合成数据进入回环检测"""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.forbidden_markers = [
            'synthetic', 'generated', 'fake', 'mock', 'random',
            'artificial', 'simulated', 'dummy', 'test'
        ]

    def validate_keyframe(self, keyframe: KeyFrame) -> bool:
        """验证关键帧数据的真实性"""
        # 1. 检查数据源标记
        if keyframe.data_source.lower() != "real_sensors":
            if self.strict_mode:
                raise ValueError(f"FORBIDDEN: Non-real data source '{keyframe.data_source}' detected")
            return False

        # 2. 检查图像特征
        if keyframe.rgb_image is not None:
            if not self._validate_rgb_image(keyframe.rgb_image):
                if self.strict_mode:
                    raise ValueError("FORBIDDEN: Suspicious RGB image detected")
                return False

        # 3. 检查点云特征
        if keyframe.lidar_points is not None:
            if not self._validate_lidar_points(keyframe.lidar_points):
                if self.strict_mode:
                    raise ValueError("FORBIDDEN: Suspicious LiDAR points detected")
                return False

        return True

    def _validate_rgb_image(self, rgb_image: np.ndarray) -> bool:
        """验证RGB图像真实性"""
        # 检查尺寸合理性
        h, w = rgb_image.shape[:2]
        if h < 100 or w < 100 or h > 2000 or w > 2000:
            warnings.warn(f"Suspicious image size: {h}×{w}")

        # 检查数值范围
        if rgb_image.dtype == np.uint8:
            if rgb_image.min() == rgb_image.max():
                warnings.warn("Constant RGB image detected")
                return False
        elif rgb_image.dtype == np.float32:
            if rgb_image.min() < -1.0 or rgb_image.max() > 1.0:
                warnings.warn(f"RGB range [{rgb_image.min():.3f}, {rgb_image.max():.3f}] suspicious")

        return True

    def _validate_lidar_points(self, lidar_points: np.ndarray) -> bool:
        """验证LiDAR点云真实性"""
        if lidar_points.shape[1] < 3:
            warnings.warn("LiDAR points must have at least XYZ coordinates")
            return False

        # 检查点云范围
        xyz = lidar_points[:, :3]
        ranges = np.linalg.norm(xyz, axis=1)

        if np.all(ranges == 0):
            warnings.warn("All LiDAR points at origin - suspicious")
            return False

        if np.max(ranges) > 200:  # 200米最大检测距离
            warnings.warn(f"LiDAR range {np.max(ranges):.1f}m exceeds realistic limit")

        return True


class SimpleGlobalDescriptor(nn.Module):
    """简化的全局描述符提取器"""

    def __init__(self, descriptor_dim: int = 256):
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # 简单的CNN backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.descriptor_head = nn.Sequential(
            nn.Linear(512, descriptor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.L2Norm(dim=1)  # L2 normalize
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB图像 (B, 3, H, W)
        Returns:
            global_descriptor: (B, descriptor_dim)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        descriptor = self.descriptor_head(features)
        return descriptor


class NetVLADLoopDetector(nn.Module):
    """基于NetVLAD的回环检测器"""

    def __init__(self, descriptor_dim: int = 256, num_clusters: int = 64):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.num_clusters = num_clusters

        # 使用简化的backbone（实际项目中可用预训练网络）
        self.backbone = SimpleGlobalDescriptor(descriptor_dim=512)

        # NetVLAD层
        self.netvlad = nn.Sequential(
            nn.Conv2d(512, num_clusters, 1),  # soft assignment
            nn.Softmax(dim=1)
        )

        self.centroids = nn.Parameter(torch.randn(num_clusters, 512))
        self.output_dim = num_clusters * 512

        # 最终描述符
        self.final_descriptor = nn.Sequential(
            nn.Linear(self.output_dim, descriptor_dim),
            nn.ReLU(),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.L2Norm(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """NetVLAD前向传播"""
        # 提取局部特征
        features = self.backbone.backbone(x)  # (B, 512, H', W')
        B, C, H, W = features.shape

        # Soft assignment
        soft_assign = self.netvlad(features)  # (B, K, H', W')
        soft_assign = soft_assign.view(B, self.num_clusters, -1)  # (B, K, N)

        # Reshape features
        features = features.view(B, C, -1)  # (B, 512, N)

        # VLAD encoding
        vlad_encoding = []
        for k in range(self.num_clusters):
            # Residual vector
            residual = features - self.centroids[k:k+1, :].unsqueeze(2)  # (B, 512, N)
            weighted_residual = soft_assign[:, k:k+1, :] * residual  # (B, 512, N)
            vlad_k = weighted_residual.sum(dim=2)  # (B, 512)
            vlad_encoding.append(vlad_k)

        vlad_encoding = torch.stack(vlad_encoding, dim=1)  # (B, K, 512)
        vlad_encoding = vlad_encoding.view(B, -1)  # (B, K*512)

        # Final descriptor
        descriptor = self.final_descriptor(vlad_encoding)
        return descriptor


class LoopDetector:
    """
    回环检测器主类
    支持多种描述符提取方法和相似度匹配
    """

    def __init__(self, descriptor_type: str = "simple",
                 similarity_threshold: float = 0.8,
                 temporal_consistency: int = 50):
        """
        Args:
            descriptor_type: 描述符类型 ["simple", "netvlad"]
            similarity_threshold: 相似度阈值
            temporal_consistency: 时间一致性约束（帧间隔）
        """
        self.descriptor_type = descriptor_type
        self.similarity_threshold = similarity_threshold
        self.temporal_consistency = temporal_consistency

        # 初始化描述符提取器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if descriptor_type == "netvlad":
            self.descriptor_extractor = NetVLADLoopDetector().to(self.device)
        else:
            self.descriptor_extractor = SimpleGlobalDescriptor().to(self.device)

        self.descriptor_extractor.eval()

        # 数据存储
        self.keyframes: List[KeyFrame] = []
        self.descriptor_database: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None

        # 验证器
        self.data_validator = RealDataValidator(strict_mode=True)

        print(f"LoopDetector initialized: {descriptor_type}, threshold={similarity_threshold}")

    def add_keyframe(self, keyframe: KeyFrame) -> bool:
        """添加关键帧"""
        # 验证数据真实性
        if not self.data_validator.validate_keyframe(keyframe):
            warnings.warn(f"Keyframe {keyframe.frame_id} failed validation")
            return False

        # 提取全局描述符
        descriptor = self._extract_global_descriptor(keyframe)
        if descriptor is None:
            warnings.warn(f"Failed to extract descriptor for keyframe {keyframe.frame_id}")
            return False

        keyframe.global_descriptor = descriptor
        self.keyframes.append(keyframe)

        # 更新数据库
        self._update_descriptor_database()

        return True

    def _extract_global_descriptor(self, keyframe: KeyFrame) -> Optional[np.ndarray]:
        """提取全局描述符"""
        if keyframe.rgb_image is None:
            return None

        try:
            # 预处理图像
            rgb_tensor = self._preprocess_image(keyframe.rgb_image)

            with torch.no_grad():
                descriptor = self.descriptor_extractor(rgb_tensor)
                return descriptor.cpu().numpy().flatten()

        except Exception as e:
            warnings.warn(f"Descriptor extraction failed: {e}")
            return None

    def _preprocess_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 转换为tensor格式
        if rgb_image.dtype == np.uint8:
            rgb_image = rgb_image.astype(np.float32) / 255.0

        # 调整尺寸
        rgb_resized = cv2.resize(rgb_image, (224, 224))

        # 转换为tensor (1, 3, H, W)
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0)
        return rgb_tensor.to(self.device)

    def _update_descriptor_database(self):
        """更新描述符数据库"""
        if not self.keyframes:
            return

        # 构建描述符矩阵
        descriptors = []
        for kf in self.keyframes:
            if kf.global_descriptor is not None:
                descriptors.append(kf.global_descriptor)

        if descriptors:
            self.descriptor_database = np.vstack(descriptors)

            # 构建FAISS索引
            descriptor_dim = self.descriptor_database.shape[1]
            self.faiss_index = faiss.IndexFlatIP(descriptor_dim)  # Inner product (cosine similarity)
            self.faiss_index.add(self.descriptor_database.astype(np.float32))

    def detect_loop_closure(self, query_keyframe: KeyFrame,
                          k_candidates: int = 5) -> List[LoopCandidate]:
        """检测回环闭合"""
        if not self.data_validator.validate_keyframe(query_keyframe):
            raise ValueError("FORBIDDEN: Query keyframe failed validation")

        if self.faiss_index is None or len(self.keyframes) < self.temporal_consistency:
            return []

        # 提取查询描述符
        query_descriptor = self._extract_global_descriptor(query_keyframe)
        if query_descriptor is None:
            return []

        # 搜索相似关键帧
        query_descriptor = query_descriptor.reshape(1, -1).astype(np.float32)
        similarities, indices = self.faiss_index.search(query_descriptor, k_candidates + 1)

        candidates = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= len(self.keyframes):
                continue

            candidate_kf = self.keyframes[idx]

            # 时间一致性检查
            frame_gap = abs(query_keyframe.frame_id - candidate_kf.frame_id)
            if frame_gap < self.temporal_consistency:
                continue

            # 相似度阈值检查
            if similarity < self.similarity_threshold:
                continue

            candidate = LoopCandidate(
                query_id=query_keyframe.frame_id,
                match_id=candidate_kf.frame_id,
                similarity_score=similarity,
                descriptor_distance=1.0 - similarity
            )

            # 几何验证（可选）
            if self._geometric_verification(query_keyframe, candidate_kf):
                candidate.geometric_verification = True
                candidate.relative_pose = self._estimate_relative_pose(query_keyframe, candidate_kf)

            candidates.append(candidate)

        return sorted(candidates, key=lambda x: x.similarity_score, reverse=True)

    def _geometric_verification(self, kf1: KeyFrame, kf2: KeyFrame) -> bool:
        """几何验证"""
        if kf1.rgb_image is None or kf2.rgb_image is None:
            return False

        try:
            # 简化的SIFT特征匹配
            sift = cv2.SIFT_create()

            # 提取特征点
            kp1, des1 = sift.detectAndCompute(cv2.cvtColor(kf1.rgb_image, cv2.COLOR_RGB2GRAY), None)
            kp2, des2 = sift.detectAndCompute(cv2.cvtColor(kf2.rgb_image, cv2.COLOR_RGB2GRAY), None)

            if des1 is None or des2 is None:
                return False

            # 特征匹配
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)

            # 比率测试
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # 几何验证阈值
            return len(good_matches) > 20

        except Exception:
            return False

    def _estimate_relative_pose(self, kf1: KeyFrame, kf2: KeyFrame) -> Optional[np.ndarray]:
        """估计相对位姿"""
        if kf1.pose is not None and kf2.pose is not None:
            # 使用已知位姿计算相对变换
            pose1 = kf1.pose  # [x, y, z, rx, ry, rz]
            pose2 = kf2.pose

            # 简化：只计算平移差异
            relative_translation = pose2[:3] - pose1[:3]
            relative_rotation = pose2[3:] - pose1[3:]
            return np.concatenate([relative_translation, relative_rotation])

        return None

    def get_database_size(self) -> int:
        """获取数据库大小"""
        return len(self.keyframes)

    def save_database(self, filepath: str):
        """保存描述符数据库"""
        save_data = {
            'keyframes': self.keyframes,
            'descriptor_database': self.descriptor_database
        }
        np.savez(filepath, **save_data)

    def load_database(self, filepath: str) -> bool:
        """加载描述符数据库"""
        try:
            data = np.load(filepath, allow_pickle=True)
            self.keyframes = data['keyframes'].tolist()
            self.descriptor_database = data['descriptor_database']
            self._update_descriptor_database()
            return True
        except Exception as e:
            warnings.warn(f"Failed to load database: {e}")
            return False