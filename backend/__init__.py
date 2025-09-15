"""
MineSLAM Backend Optimization Module
后端优化模块：基于g2o的因子图优化，支持位姿图优化和语义地标
"""

from .graph_optimizer import GraphOptimizer, PoseGraphOptimizer, PoseVertex
from .loop_detector import LoopDetector, NetVLADLoopDetector, KeyFrame
from .semantic_factors import SemanticFactor, LandmarkVertex, SemanticObservation
from .trajectory_evaluator import TrajectoryEvaluator, TrajectoryVisualizer
from .backend_interface import backend_optimize

__all__ = [
    'GraphOptimizer',
    'PoseGraphOptimizer',
    'PoseVertex',
    'LoopDetector',
    'NetVLADLoopDetector',
    'KeyFrame',
    'SemanticFactor',
    'LandmarkVertex',
    'SemanticObservation',
    'TrajectoryEvaluator',
    'TrajectoryVisualizer',
    'backend_optimize'
]