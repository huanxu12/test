"""
Graph Optimizer Module for MineSLAM
基于g2o的图优化器：VertexSE3/VertexPointXYZ，邻接位姿因子、回环因子、语义观测因子
"""

import os
import numpy as np
import torch
import g2o
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class PoseVertex:
    """SE3位姿顶点"""
    id: int
    timestamp: float
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [rx, ry, rz] Euler angles
    information_matrix: Optional[np.ndarray] = None

    def to_se3(self) -> g2o.SE3Quat:
        """转换为SE3四元数表示"""
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', self.orientation)
        quat = R.as_quat()  # [x, y, z, w]
        # g2o uses [w, x, y, z] quaternion format
        return g2o.SE3Quat([quat[3], quat[0], quat[1], quat[2]], self.position)


@dataclass
class LandmarkVertex:
    """3D地标顶点"""
    id: int
    position: np.ndarray  # [x, y, z]
    semantic_class: str
    confidence: float
    timestamp: float
    information_matrix: Optional[np.ndarray] = None


@dataclass
class PoseEdge:
    """位姿边（相邻帧或回环）"""
    from_id: int
    to_id: int
    relative_pose: g2o.SE3Quat
    information_matrix: np.ndarray  # 6x6
    edge_type: str  # "odometry" or "loop_closure"


@dataclass
class SemanticEdge:
    """语义观测边"""
    pose_id: int
    landmark_id: int
    observation: np.ndarray  # 2D/3D observation
    information_matrix: np.ndarray  # confidence-weighted
    semantic_class: str


class GraphOptimizer:
    """
    基于g2o的图优化器
    支持VertexSE3（位姿）和VertexPointXYZ（地标）
    """

    def __init__(self, solver_type: str = "lm_var"):
        """
        Args:
            solver_type: 求解器类型 ["lm_var", "lm_fix", "gn_var", "gn_fix"]
        """
        self.optimizer = g2o.SparseOptimizer()
        self.solver_type = solver_type
        self._setup_solver()

        # 存储顶点和边
        self.pose_vertices: Dict[int, PoseVertex] = {}
        self.landmark_vertices: Dict[int, LandmarkVertex] = {}
        self.pose_edges: List[PoseEdge] = []
        self.semantic_edges: List[SemanticEdge] = []

        # 优化参数
        self.max_iterations = 20
        self.verbose = True

    def _setup_solver(self):
        """设置g2o求解器"""
        # 线性求解器
        linear_solver = g2o.BlockSolverSE3.LinearSolverType()
        linear_solver = g2o.LinearSolverCholmodSE3()

        # 块求解器
        block_solver = g2o.BlockSolverSE3(linear_solver)

        # 优化算法
        if self.solver_type == "lm_var":
            algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
        elif self.solver_type == "gn_var":
            algorithm = g2o.OptimizationAlgorithmGaussNewton(block_solver)
        else:
            algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)

        self.optimizer.set_algorithm(algorithm)

    def add_pose_vertex(self, pose: PoseVertex, fixed: bool = False):
        """添加位姿顶点"""
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(pose.id)
        vertex.set_estimate(pose.to_se3())
        vertex.set_fixed(fixed)

        self.optimizer.add_vertex(vertex)
        self.pose_vertices[pose.id] = pose

    def add_landmark_vertex(self, landmark: LandmarkVertex, fixed: bool = False):
        """添加地标顶点"""
        vertex = g2o.VertexSBAPointXYZ()
        vertex.set_id(landmark.id)
        vertex.set_estimate(landmark.position)
        vertex.set_fixed(fixed)

        self.optimizer.add_vertex(vertex)
        self.landmark_vertices[landmark.id] = landmark

    def add_pose_edge(self, edge: PoseEdge):
        """添加位姿边（里程计或回环）"""
        g2o_edge = g2o.EdgeSE3Expmap()
        g2o_edge.set_vertex(0, self.optimizer.vertex(edge.from_id))
        g2o_edge.set_vertex(1, self.optimizer.vertex(edge.to_id))
        g2o_edge.set_measurement(edge.relative_pose)
        g2o_edge.set_information(edge.information_matrix)

        # 设置鲁棒核
        if edge.edge_type == "loop_closure":
            robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))  # Chi2 threshold
            g2o_edge.set_robust_kernel(robust_kernel)

        self.optimizer.add_edge(g2o_edge)
        self.pose_edges.append(edge)

    def add_semantic_edge(self, edge: SemanticEdge):
        """添加语义观测边"""
        # 3D点到位姿的投影边
        g2o_edge = g2o.EdgeSE3ProjectXYZ()
        g2o_edge.set_vertex(0, self.optimizer.vertex(edge.landmark_id))
        g2o_edge.set_vertex(1, self.optimizer.vertex(edge.pose_id))
        g2o_edge.set_measurement(edge.observation[:2])  # 2D观测
        g2o_edge.set_information(edge.information_matrix)

        # 相机参数（假设已知内参）
        cam_params = g2o.CameraParameters(525.0, np.array([320.0, 240.0]), 0.0)
        g2o_edge.set_parameter_id(0, 0)

        self.optimizer.add_edge(g2o_edge)
        self.semantic_edges.append(edge)

    def optimize(self, iterations: Optional[int] = None) -> Dict[str, Any]:
        """执行图优化"""
        if iterations is None:
            iterations = self.max_iterations

        # 初始化优化器
        self.optimizer.initialize_optimization()

        # 计算初始误差
        initial_chi2 = self.optimizer.chi2()

        # 执行优化
        self.optimizer.set_verbose(self.verbose)
        converged = self.optimizer.optimize(iterations)

        # 计算最终误差
        final_chi2 = self.optimizer.chi2()

        # 提取优化后的位姿
        optimized_poses = self._extract_optimized_poses()
        optimized_landmarks = self._extract_optimized_landmarks()

        return {
            'converged': converged,
            'initial_chi2': initial_chi2,
            'final_chi2': final_chi2,
            'iterations': iterations,
            'optimized_poses': optimized_poses,
            'optimized_landmarks': optimized_landmarks,
            'chi2_reduction': (initial_chi2 - final_chi2) / initial_chi2 if initial_chi2 > 0 else 0.0
        }

    def _extract_optimized_poses(self) -> Dict[int, np.ndarray]:
        """提取优化后的位姿"""
        optimized_poses = {}

        for pose_id in self.pose_vertices.keys():
            vertex = self.optimizer.vertex(pose_id)
            if vertex is not None:
                se3_estimate = vertex.estimate()

                # 提取位置和四元数
                position = se3_estimate.translation()
                quaternion = se3_estimate.rotation().coeffs()  # [x, y, z, w]

                # 转换为Euler角
                from scipy.spatial.transform import Rotation
                R = Rotation.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
                euler = R.as_euler('xyz')

                # 6DoF: [x, y, z, rx, ry, rz]
                optimized_poses[pose_id] = np.concatenate([position, euler])

        return optimized_poses

    def _extract_optimized_landmarks(self) -> Dict[int, np.ndarray]:
        """提取优化后的地标"""
        optimized_landmarks = {}

        for landmark_id in self.landmark_vertices.keys():
            vertex = self.optimizer.vertex(landmark_id)
            if vertex is not None:
                optimized_landmarks[landmark_id] = vertex.estimate()

        return optimized_landmarks

    def get_covariance_matrix(self, vertex_id: int) -> Optional[np.ndarray]:
        """获取顶点的协方差矩阵"""
        try:
            # 计算边缘协方差
            self.optimizer.compute_marginals()
            vertex = self.optimizer.vertex(vertex_id)
            if vertex is not None:
                return self.optimizer.get_marginal_covariance(vertex)
        except Exception as e:
            warnings.warn(f"Cannot compute covariance for vertex {vertex_id}: {e}")
        return None

    def save_graph(self, filename: str):
        """保存图到g2o格式文件"""
        self.optimizer.save(filename)

    def load_graph(self, filename: str) -> bool:
        """从g2o格式文件加载图"""
        return self.optimizer.load(filename)


class PoseGraphOptimizer:
    """
    专门的位姿图优化器
    针对SLAM后端优化进行了优化
    """

    def __init__(self):
        self.graph_optimizer = GraphOptimizer()
        self.pose_sequence: List[PoseVertex] = []
        self.loop_closures: List[Tuple[int, int, g2o.SE3Quat, np.ndarray]] = []

    def add_pose_sequence(self, poses: List[PoseVertex]):
        """添加位姿序列"""
        self.pose_sequence = poses

        # 添加到图中
        for i, pose in enumerate(poses):
            is_first = (i == 0)
            self.graph_optimizer.add_pose_vertex(pose, fixed=is_first)

        # 添加序列边（里程计约束）
        for i in range(len(poses) - 1):
            self._add_odometry_edge(poses[i], poses[i + 1])

    def _add_odometry_edge(self, pose_from: PoseVertex, pose_to: PoseVertex):
        """添加里程计边"""
        # 计算相对变换
        se3_from = pose_from.to_se3()
        se3_to = pose_to.to_se3()
        relative_pose = se3_from.inverse() * se3_to

        # 信息矩阵（较高精度）
        information = np.eye(6) * 100.0  # 较强的里程计约束

        edge = PoseEdge(
            from_id=pose_from.id,
            to_id=pose_to.id,
            relative_pose=relative_pose,
            information_matrix=information,
            edge_type="odometry"
        )

        self.graph_optimizer.add_pose_edge(edge)

    def add_loop_closure(self, from_id: int, to_id: int,
                        relative_pose: g2o.SE3Quat,
                        confidence: float = 1.0):
        """添加回环约束"""
        # 基于置信度的信息矩阵
        base_information = np.eye(6) * 10.0  # 基础信息矩阵
        information = base_information * confidence

        edge = PoseEdge(
            from_id=from_id,
            to_id=to_id,
            relative_pose=relative_pose,
            information_matrix=information,
            edge_type="loop_closure"
        )

        self.graph_optimizer.add_pose_edge(edge)
        self.loop_closures.append((from_id, to_id, relative_pose, information))

    def add_semantic_constraints(self, semantic_observations: List[Dict]):
        """添加语义约束"""
        landmark_id_counter = 1000000  # 避免与位姿ID冲突

        for obs in semantic_observations:
            # 创建地标顶点
            landmark = LandmarkVertex(
                id=landmark_id_counter,
                position=obs['world_position'],
                semantic_class=obs['class'],
                confidence=obs['confidence'],
                timestamp=obs['timestamp']
            )

            self.graph_optimizer.add_landmark_vertex(landmark)

            # 创建观测边
            # 信息矩阵根据检测置信度调整
            info_scale = obs['confidence'] * 1000.0
            information = np.eye(2) * info_scale

            semantic_edge = SemanticEdge(
                pose_id=obs['pose_id'],
                landmark_id=landmark_id_counter,
                observation=obs['image_observation'],
                information_matrix=information,
                semantic_class=obs['class']
            )

            self.graph_optimizer.add_semantic_edge(semantic_edge)
            landmark_id_counter += 1

    def backend_optimize(self, max_iterations: int = 20) -> Dict[str, Any]:
        """执行后端优化"""
        result = self.graph_optimizer.optimize(iterations=max_iterations)

        # 更新位姿序列
        optimized_poses = result['optimized_poses']
        for pose in self.pose_sequence:
            if pose.id in optimized_poses:
                optimized_6dof = optimized_poses[pose.id]
                pose.position = optimized_6dof[:3]
                pose.orientation = optimized_6dof[3:]

        return result

    def get_trajectory(self) -> np.ndarray:
        """获取优化后的轨迹"""
        trajectory = []
        for pose in sorted(self.pose_sequence, key=lambda p: p.timestamp):
            trajectory.append([
                pose.timestamp,
                pose.position[0], pose.position[1], pose.position[2],
                pose.orientation[0], pose.orientation[1], pose.orientation[2]
            ])
        return np.array(trajectory)

    def get_pose_uncertainties(self) -> Dict[int, np.ndarray]:
        """获取位姿的不确定性"""
        uncertainties = {}
        for pose in self.pose_sequence:
            cov = self.graph_optimizer.get_covariance_matrix(pose.id)
            if cov is not None:
                uncertainties[pose.id] = np.sqrt(np.diag(cov))
        return uncertainties