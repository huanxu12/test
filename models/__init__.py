"""
Models Module for MineSLAM
多模态编码器、MoE融合、多任务头部等组件
"""

# 核心组件导入
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

# Kendall不确定性 (修复版)
from .kendall_uncertainty import (
    FixedKendallUncertainty,
    KendallUncertainty,
    create_fixed_kendall_uncertainty,
    create_kendall_uncertainty
)

# 统一模型类
from .mineslam_model import (
    MineSLAMModel,
    create_mineslam_model,
    MineSLAM  # 向后兼容别名
)

# 完整导出列表
__all__ = [
    # 核心组件
    'ImageEncoder',
    'LidarEncoder',
    'IMUEncoder',
    'MultiModalEncoder',

    # MoE融合
    'MoEFusion',
    'MoEConfig',
    'ThermalGuidedAttention',
    'Expert',
    'GatingNetwork',
    'create_moe_fusion',

    # 任务头部
    'PoseHead',
    'SE3Utils',
    'DetectionHead',
    'BBox3DUtils',

    # Kendall不确定性
    'FixedKendallUncertainty',
    'KendallUncertainty',
    'create_fixed_kendall_uncertainty',
    'create_kendall_uncertainty',

    # 统一模型
    'MineSLAMModel',
    'create_mineslam_model',
    'MineSLAM'
]

# 保持原有模型兼容性
from .encoders import MultiModalEncoder as NewMultiModalEncoder

# 版本信息
__version__ = "1.0.0"
__author__ = "MineSLAM Team"

# 模型工厂函数
def get_model_info():
    """获取模型模块信息"""
    return {
        'version': __version__,
        'components': len(__all__),
        'available_models': ['MineSLAMModel'],
        'encoders': ['ImageEncoder', 'LidarEncoder', 'IMUEncoder', 'MultiModalEncoder'],
        'fusion': ['MoEFusion'],
        'heads': ['PoseHead', 'DetectionHead'],
        'uncertainty': ['FixedKendallUncertainty', 'KendallUncertainty']
    }

def create_model(model_type: str = 'MineSLAMModel', config: dict = None):
    """
    统一模型创建接口

    Args:
        model_type: 模型类型 (目前支持 'MineSLAMModel')
        config: 模型配置

    Returns:
        模型实例
    """
    if model_type == 'MineSLAMModel' or model_type == 'MineSLAM':
        return create_mineslam_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# 打印模块信息
print(f"📦 MineSLAM Models Module v{__version__} loaded")
print(f"   Available components: {len(__all__)}")
print(f"   Main model: MineSLAMModel (fixed Kendall uncertainty)")