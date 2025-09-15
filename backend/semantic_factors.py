"""
Semantic Factors Module for MineSLAM Backend
语义因子模块：语义观测因子，信息矩阵=diag(k·conf)，基于真实检测置信度
"""

import numpy as np
import g2o
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class SemanticObservation:
    """语义观测数据结构"""
    pose_id: int
    landmark_id: int
    semantic_class: str
    confidence: float
    image_coordinates: np.ndarray  # 2D图像坐标 [u, v]
    world_coordinates: Optional[np.ndarray] = None  # 3D世界坐标 [x, y, z]
    detection_method: str = "real_detection"  # 必须是真实检测


class SemanticFactor:
    """
    语义观测因子
    连接位姿顶点和地标顶点，信息矩阵基于检测置信度
    """

    def __init__(self, k_confidence_scale: float = 1000.0):
        """
        Args:
            k_confidence_scale: 置信度缩放因子，用于信息矩阵计算
        """
        self.k_confidence_scale = k_confidence_scale

    def create_semantic_edge(self, observation: SemanticObservation,
                           camera_params: Dict[str, float]) -> g2o.EdgeSE3ProjectXYZ:
        """创建语义观测边"""

        # 验证观测数据
        if observation.detection_method != "real_detection":
            raise ValueError(f"FORBIDDEN: Non-real detection method '{observation.detection_method}'")

        if observation.confidence <= 0 or observation.confidence > 1.0:
            raise ValueError(f"Invalid confidence: {observation.confidence}")

        # 创建投影边
        edge = g2o.EdgeSE3ProjectXYZ()

        # 设置观测值（2D图像坐标）
        edge.set_measurement(observation.image_coordinates)

        # 计算信息矩阵：对角矩阵，元素为 k * confidence
        info_weight = self.k_confidence_scale * observation.confidence
        information_matrix = np.eye(2) * info_weight

        edge.set_information(information_matrix)

        # 设置相机参数
        focal_length = camera_params.get('focal_length', 525.0)
        principal_point = camera_params.get('principal_point', [320.0, 240.0])
        baseline = camera_params.get('baseline', 0.0)

        cam_params = g2o.CameraParameters(
            focal_length,
            np.array(principal_point),
            baseline
        )
        edge.set_parameter_id(0, 0)

        return edge

    def validate_semantic_observation(self, observation: SemanticObservation) -> bool:
        """验证语义观测的真实性"""
        # 检查检测方法
        forbidden_methods = [
            'synthetic', 'generated', 'random', 'mock', 'fake',
            'simulated', 'artificial', 'dummy'
        ]

        detection_method_lower = observation.detection_method.lower()
        for forbidden in forbidden_methods:
            if forbidden in detection_method_lower:
                raise ValueError(f"FORBIDDEN: Synthetic detection method '{observation.detection_method}'")

        # 检查置信度范围
        if not (0.0 < observation.confidence <= 1.0):
            raise ValueError(f"Invalid confidence range: {observation.confidence}")

        # 检查图像坐标合理性
        u, v = observation.image_coordinates
        if not (0 <= u <= 2000 and 0 <= v <= 2000):  # 合理的图像尺寸范围
            raise ValueError(f"Suspicious image coordinates: [{u}, {v}]")

        # 检查世界坐标合理性（如果提供）
        if observation.world_coordinates is not None:
            world_coords = observation.world_coordinates
            max_distance = np.linalg.norm(world_coords)
            if max_distance > 100:  # 100米最大检测距离
                raise ValueError(f"Unrealistic detection distance: {max_distance:.1f}m")

        return True

    def compute_reprojection_error(self, observation: SemanticObservation,
                                 pose_estimate: np.ndarray,
                                 landmark_estimate: np.ndarray,
                                 camera_params: Dict[str, float]) -> float:
        """计算重投影误差"""
        # 从世界坐标变换到相机坐标
        R = self._euler_to_rotation_matrix(pose_estimate[3:6])
        t = pose_estimate[0:3]

        # 世界点到相机坐标系
        camera_point = R.T @ (landmark_estimate - t)

        if camera_point[2] <= 0:  # 点在相机后方
            return float('inf')

        # 投影到图像平面
        focal_length = camera_params.get('focal_length', 525.0)
        principal_point = camera_params.get('principal_point', [320.0, 240.0])

        u_proj = focal_length * camera_point[0] / camera_point[2] + principal_point[0]
        v_proj = focal_length * camera_point[1] / camera_point[2] + principal_point[1]

        # 计算重投影误差
        u_obs, v_obs = observation.image_coordinates
        error = np.sqrt((u_proj - u_obs)**2 + (v_proj - v_obs)**2)

        return error

    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """欧拉角转旋转矩阵"""
        rx, ry, rz = euler_angles

        # 绕X轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        # 绕Y轴旋转
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # 绕Z轴旋转
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = Rz @ Ry @ Rx
        return R


class LandmarkVertex:
    """3D地标顶点扩展"""

    def __init__(self, landmark_id: int, position: np.ndarray,
                 semantic_class: str, confidence: float,
                 observations: List[SemanticObservation]):
        self.landmark_id = landmark_id
        self.position = position  # [x, y, z]
        self.semantic_class = semantic_class
        self.confidence = confidence
        self.observations = observations
        self.covariance: Optional[np.ndarray] = None

    def add_observation(self, observation: SemanticObservation):
        """添加语义观测"""
        # 验证观测
        semantic_factor = SemanticFactor()
        if semantic_factor.validate_semantic_observation(observation):
            self.observations.append(observation)

    def compute_landmark_uncertainty(self) -> np.ndarray:
        """计算地标位置不确定性"""
        if len(self.observations) == 0:
            return np.eye(3) * 1000.0  # 高不确定性

        # 基于观测数量和置信度的简化不确定性模型
        total_confidence = sum(obs.confidence for obs in self.observations)
        num_observations = len(self.observations)

        # 信息矩阵 ∝ 观测数量 × 平均置信度
        information_scale = num_observations * (total_confidence / num_observations)
        uncertainty_scale = 1.0 / max(information_scale, 0.1)

        uncertainty_matrix = np.eye(3) * uncertainty_scale
        return uncertainty_matrix

    def get_semantic_info(self) -> Dict[str, Any]:
        """获取语义信息摘要"""
        return {
            'landmark_id': self.landmark_id,
            'semantic_class': self.semantic_class,
            'position': self.position.tolist(),
            'confidence': self.confidence,
            'num_observations': len(self.observations),
            'observation_poses': [obs.pose_id for obs in self.observations]
        }