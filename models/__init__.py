"""
Models Module for MineSLAM
多模态编码器、MoE融合、多任务头部等组件
"""

from .encoders import (
    ImageEncoder,
    LidarEncoder,
    IMUEncoder,
    MultiModalEncoder
)

from .moe_fusion import (
    MoEFusion,
    MoEConfig,
    ThermalGuidedAttention,
    Expert,
    GatingNetwork,
    create_moe_fusion
)

from .pose_head import (
    PoseHead,
    SE3Utils
)

from .detection_head import (
    DetectionHead,
    BBox3DUtils
)

__all__ = [
    # Encoders
    'ImageEncoder',
    'LidarEncoder',
    'IMUEncoder',
    'MultiModalEncoder',

    # MoE Fusion
    'MoEFusion',
    'MoEConfig',
    'ThermalGuidedAttention',
    'Expert',
    'GatingNetwork',
    'create_moe_fusion',

    # Task Heads
    'PoseHead',
    'SE3Utils',
    'DetectionHead',
    'BBox3DUtils'
]

# 保持原有模型兼容性
from .encoders import MultiModalEncoder as NewMultiModalEncoder