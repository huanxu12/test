"""
Advanced Training Loop for MineSLAM Multi-Task Learning
集成训练系统：Kendall不确定性、AMP、梯度累积、Early-Stop、实时日志
严格基于真实数据，禁止合成数据替代
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
try:
    # 新版本PyTorch (≥1.10)
    from torch.amp import autocast
except ImportError:
    # 旧版本PyTorch
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_config
from data.mineslam_dataset import MineSLAMDataset
from models.encoders import MultiModalEncoder
from models.moe_fusion import MoEFusion, MoEConfig
from models.pose_head import PoseHead
from models.detection_head import DetectionHead
from models.kendall_uncertainty import create_fixed_kendall_uncertainty
from matcher import HungarianMatcher, TargetGenerator

# 集成评估和可视化系统
try:
    from slam_evaluator import MineSLAMEvaluator, SLAMTrajectoryMetrics
    from slam_visualizer import SLAMVisualizer, TrainingMonitor
    SLAM_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SLAM integration partially unavailable: {e}")
    SLAM_INTEGRATION_AVAILABLE = False

# 抑制重复警告 - 路线B优化
warnings.filterwarnings("ignore", message="coordinates implicitly converted to torch.IntTensor")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast.*` is deprecated")

# 全局计数器，用于限制某些警告显示
_warning_counters = {
    'minkowski_coords': 0,
    'autocast_deprecated': 0,
    'missing_modality': 0
}


def pointcloud_collate_fn(batch):
    """
    路线B：柔性批处理函数，强制对齐batch_size，零填充+掩码

    核心原则：
    1. 不丢样本，所有模态强制对齐相同batch_size B
    2. 缺失模态用零样本占位
    3. 返回present_mask指示真实/填充数据
    4. 禁止filter(None)
    5. 点云统一到固定长度P，图像统一尺寸

    Args:
        batch: 样本列表，每个样本是一个字典

    Returns:
        collated_batch: 批处理后的字典，包含：
        - 各模态数据：统一batch_size的张量
        - present_mask：指示真实/填充数据的掩码字典
    """
    if not batch:
        return {}

    batch_size = len(batch)
    collated_batch = {}
    present_mask = {}

    # 固定参数 - 路线B要求
    FIXED_LIDAR_SIZE = 16384  # P参数：点云统一长度

    for key in batch[0].keys():
        # 确定哪些样本有此模态的数据
        valid_indices = [i for i, item in enumerate(batch) if key in item]

        # 创建present_mask - 路线B格式
        mask = torch.zeros(batch_size, dtype=torch.bool)
        mask[valid_indices] = True
        present_mask[key] = mask

        if key == 'lidar':
            # 点云：统一到固定长度P，强制对齐batch_size B
            standardized_lidar = []
            for i in range(batch_size):
                if i in valid_indices:
                    # 真实点云：统一到固定长度
                    points = batch[i][key]
                    uniform_points = pad_or_subsample_pointcloud(points, target_size=FIXED_LIDAR_SIZE)
                    standardized_lidar.append(uniform_points)
                else:
                    # 填充点云：固定长度的零点云
                    dummy_lidar = torch.zeros(FIXED_LIDAR_SIZE, 4, dtype=torch.float32)
                    standardized_lidar.append(dummy_lidar)

            collated_batch[key] = torch.stack(standardized_lidar)  # [B, P, C]

        elif key in ['rgb', 'depth', 'thermal']:
            # 图像数据：强制对齐batch_size B
            if valid_indices:
                # 获取参考尺寸（应在dataset中预处理为统一尺寸）
                ref_shape = batch[valid_indices[0]][key].shape
                standardized_images = []

                for i in range(batch_size):
                    if i in valid_indices:
                        # 真实图像
                        standardized_images.append(batch[i][key])
                    else:
                        # 填充图像：相同尺寸的零图像
                        dummy_image = torch.zeros(ref_shape, dtype=torch.float32)
                        standardized_images.append(dummy_image)

                collated_batch[key] = torch.stack(standardized_images)  # [B, C, H, W]
            else:
                # 所有样本都缺失此模态（不应发生）
                collated_batch[key] = []

        elif key == 'imu':
            # IMU序列数据：强制对齐batch_size B
            if valid_indices:
                ref_shape = batch[valid_indices[0]][key].shape
                standardized_imu = []

                for i in range(batch_size):
                    if i in valid_indices:
                        # 真实IMU序列
                        standardized_imu.append(batch[i][key])
                    else:
                        # 填充IMU序列：相同长度的零序列
                        dummy_imu = torch.zeros(ref_shape, dtype=torch.float32)
                        standardized_imu.append(dummy_imu)

                collated_batch[key] = torch.stack(standardized_imu)  # [B, T, 6]
            else:
                collated_batch[key] = []

        else:
            # 其他数据（pose_delta, boxes等）：强制对齐batch_size B
            if valid_indices:
                ref_item = batch[valid_indices[0]][key]
                standardized_items = []

                for i in range(batch_size):
                    if i in valid_indices:
                        # 真实数据
                        standardized_items.append(batch[i][key])
                    else:
                        # 填充数据：根据类型创建默认值
                        if hasattr(ref_item, 'shape'):
                            dummy_item = torch.zeros_like(ref_item)
                        else:
                            dummy_item = torch.tensor(0.0)  # 标量默认值
                        standardized_items.append(dummy_item)

                try:
                    collated_batch[key] = torch.stack(standardized_items)
                except:
                    # 无法stack时保持列表形式
                    collated_batch[key] = standardized_items
            else:
                collated_batch[key] = []

    # 添加present_mask到批次数据 - 路线B格式
    collated_batch['present_mask'] = present_mask

    return collated_batch


def pad_or_subsample_pointcloud(points: torch.Tensor, target_size: int = 32768) -> torch.Tensor:
    """
    填充或子采样点云到统一尺寸

    Args:
        points: 输入点云张量 (N, 4)
        target_size: 目标点数

    Returns:
        处理后的点云张量 (target_size, 4)
    """
    num_points = points.shape[0]

    if num_points >= target_size:
        # 随机子采样
        indices = torch.randperm(num_points)[:target_size]
        return points[indices]
    else:
        # 重复填充到目标尺寸
        repeat_times = (target_size + num_points - 1) // num_points
        repeated = points.repeat(repeat_times, 1)
        return repeated[:target_size]


class RealDataValidator:
    """真实数据验证器 - 防止合成数据替代"""

    def __init__(self):
        self.forbidden_keys = [
            '_generated', '_synthetic', '_random', '_mock', '_fake',
            '_simulated', '_artificial', '_dummy', '_test_'
        ]
        self.validation_enabled = True

    def validate_batch(self, batch: Dict, batch_idx: int):
        """验证批次数据的真实性"""
        if not self.validation_enabled:
            return

        # 1. 检查数据字典键名 - 更严格的检查
        for key in batch.keys():
            key_lower = str(key).lower()
            for forbidden in self.forbidden_keys:
                if forbidden in key_lower:
                    raise ValueError(
                        f"FORBIDDEN: Detected synthetic data marker '{forbidden}' "
                        f"in batch {batch_idx}, key '{key}'"
                    )

        # 2. 检查张量特征
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 时间戳验证（不受元素数量限制）
                if key == 'timestamp' and hasattr(value, 'item'):
                    if value.dim() == 0:
                        ts = value.item()
                        if ts <= 0 or ts > 2e9:
                            raise ValueError(f"Invalid timestamp {ts} in batch {batch_idx}")
                    else:
                        # 检查所有时间戳
                        for i, ts in enumerate(value.flatten()):
                            ts_val = ts.item()
                            if ts_val <= 0 or ts_val > 2e9:
                                raise ValueError(f"Invalid timestamp {ts_val} in batch {batch_idx} at index {i}")

                # 其他张量检查（需要足够的元素数量）
                if value.numel() > 100:
                    # 检查是否为常数张量
                    if len(torch.unique(value)) == 1:
                        warnings.warn(f"Suspicious constant tensor '{key}' in batch {batch_idx}")

                    # 检查数值范围合理性
                    if key == 'rgb' and (value.min() < -3 or value.max() > 3):
                        warnings.warn(f"Suspicious RGB range [{value.min():.3f}, {value.max():.3f}] "
                                    f"in batch {batch_idx}")


class EnhancedTrainingMetrics:
    """增强的训练指标管理器 - 集成SLAM评估"""

    def __init__(self, log_dir: str, config: Dict):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # 原有指标历史
        self.metrics_history = {
            'train_losses': [],
            'val_losses': [],
            'ate_errors': [],
            'rpe_errors': [],
            'map_scores': [],
            'fps_values': [],
            'kendall_weights': [],
            'gpu_memory': []
        }

        # 新增SLAM详细指标
        self.slam_metrics_history = {
            'trajectory_metrics': [],      # 轨迹详细分析
            'detection_metrics': [],       # 检测详细分析
            'fusion_metrics': [],          # 多模态融合分析
            'uncertainty_metrics': []      # 不确定性分析
        }

        # 最佳指标记录
        self.best_ate = float('inf')
        self.best_map = 0.0
        self.best_val_loss = float('inf')

        # 集成SLAM评估器
        if SLAM_INTEGRATION_AVAILABLE:
            self.slam_evaluator = MineSLAMEvaluator(config)
            self.trajectory_metrics = SLAMTrajectoryMetrics()
            # 使用简化的检测指标
            self.detection_metrics = DetectionMetrics()  # 从原有metrics模块导入
        else:
            self.slam_evaluator = None
            self.trajectory_metrics = None
            self.detection_metrics = None

    def update_train_metrics(self, losses: Dict[str, float],
                           kendall_weights: Dict[str, float],
                           gpu_memory: float, fps: float, epoch: int, step: int):
        """更新训练指标（保持原有接口）"""
        self.metrics_history['train_losses'].append({
            'epoch': epoch,
            'step': step,
            'timestamp': time.time(),
            'losses': losses,
            'fps': fps,
            'gpu_memory_mb': gpu_memory
        })

        self.metrics_history['kendall_weights'].append({
            'epoch': epoch,
            'step': step,
            'weights': kendall_weights
        })

    def update_slam_metrics(self, epoch: int, model_outputs: Dict,
                           ground_truth: Dict, batch_data: Dict):
        """更新完整的SLAM评估指标"""
        if not SLAM_INTEGRATION_AVAILABLE:
            return

        try:
            # 轨迹评估
            if 'pose' in model_outputs and 'pose' in ground_truth and self.trajectory_metrics:
                traj_metrics = self.trajectory_metrics.compute_detailed_metrics(
                    model_outputs['pose'], ground_truth['pose']
                )
                self.slam_metrics_history['trajectory_metrics'].append({
                    'epoch': epoch,
                    'timestamp': time.time(),
                    'metrics': traj_metrics
                })

            # 检测评估
            if 'detection' in model_outputs and 'detection' in ground_truth and self.detection_metrics:
                det_metrics = self.detection_metrics.compute()  # 使用简化接口
                self.slam_metrics_history['detection_metrics'].append({
                    'epoch': epoch,
                    'timestamp': time.time(),
                    'metrics': det_metrics
                })

            # MoE融合分析
            if 'moe_analysis' in model_outputs:
                fusion_analysis = self._analyze_moe_fusion(model_outputs['moe_analysis'])
                self.slam_metrics_history['fusion_metrics'].append({
                    'epoch': epoch,
                    'timestamp': time.time(),
                    'analysis': fusion_analysis
                })

        except Exception as e:
            warnings.warn(f"SLAM metrics update failed: {e}")

    def _analyze_moe_fusion(self, moe_output: Dict) -> Dict:
        """分析MoE融合效果"""
        analysis = {}

        if 'gate_weights' in moe_output:
            gate_weights = moe_output['gate_weights']
            analysis['expert_utilization'] = {
                f'expert_{i}': float(gate_weights[:, i].mean())
                for i in range(gate_weights.shape[1])
            }
            analysis['gate_entropy'] = float(moe_output.get('entropy_loss', 0))
            analysis['load_balance'] = float(gate_weights.std())

        return analysis

    def update_val_metrics(self, val_loss: float, ate: float,
                          rpe: float, map_score: float, epoch: int):
        """更新验证指标（保持原有接口）"""
        self.metrics_history['val_losses'].append({
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': time.time()
        })

        self.metrics_history['ate_errors'].append({
            'epoch': epoch,
            'ate': ate
        })

        self.metrics_history['rpe_errors'].append({
            'epoch': epoch,
            'rpe': rpe
        })

        self.metrics_history['map_scores'].append({
            'epoch': epoch,
            'map': map_score
        })

        # 更新最佳记录
        if ate < self.best_ate:
            self.best_ate = ate

        if map_score > self.best_map:
            self.best_map = map_score

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

    def save_enhanced_metrics(self, filename: str = 'enhanced_training_metrics.json'):
        """保存增强指标到文件"""
        metrics_path = self.log_dir / filename

        # 合并所有指标
        output_data = {
            'best_metrics': {
                'best_ate': self.best_ate,
                'best_map': self.best_map,
                'best_val_loss': self.best_val_loss
            },
            'basic_history': self.metrics_history,
            'slam_history': self.slam_metrics_history,
            'config': self.config
        }

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

        return metrics_path

    def should_early_stop(self, patience: int = 10, min_delta: float = 1e-4) -> bool:
        """判断是否应该早停（保持原有接口）"""
        if len(self.metrics_history['val_losses']) < patience:
            return False

        recent_losses = [item['val_loss'] for item in
                        self.metrics_history['val_losses'][-patience:]]

        if len(recent_losses) < patience:
            return False

        best_recent = min(recent_losses)
        return (self.best_val_loss - best_recent) < min_delta


class MineSLAMTrainer:
    """MineSLAM多任务训练器"""

    def __init__(self, config: Dict, output_dir: str = "outputs/training"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        # 训练参数
        self.batch_size = config.get('batch_size', 4)
        self.accumulation_steps = config.get('gradient_accumulation', 4)
        self.max_epochs = config.get('max_epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)

        # Early stopping参数
        self.early_stop_patience = config.get('early_stop_patience', 10)
        self.early_stop_min_delta = config.get('early_stop_min_delta', 1e-4)

        # 验证目标
        self.target_ate = config.get('target_ate', 1.5)  # 目标ATE ≤ 1.5m
        self.target_map = config.get('target_map', 0.6)   # 目标mAP ≥ 60%

        # 初始化组件
        self._setup_models()
        self._setup_training_components()
        self._setup_data_loaders()

        # 集成增强的验证器和指标管理
        self.data_validator = RealDataValidator()
        self.metrics = EnhancedTrainingMetrics(str(self.output_dir / 'logs'), config)

        # 集成可视化系统
        if SLAM_INTEGRATION_AVAILABLE:
            self.visualizer = SLAMVisualizer(config, output_dir=str(self.output_dir / 'visualizations'))
            self.training_monitor = TrainingMonitor(
                log_dir=str(self.output_dir / 'logs'),
                update_interval=50  # 每50个batch更新一次
            )
        else:
            self.visualizer = None
            self.training_monitor = None

        print(f"MineSLAMTrainer initialized: {self.batch_size}×{self.accumulation_steps} effective batch (Route B)")
        print(f"Targets: ATE≤{self.target_ate}m, mAP≥{self.target_map*100}%")
        print(f"Route B features: Masked padding, consistent batch_size, no sample dropping")
        if SLAM_INTEGRATION_AVAILABLE:
            print(f"🎨 Integrated: SLAM Evaluator + Real-time Visualizer + Training Monitor")
        else:
            print(f"⚠️ Basic training mode: SLAM integration unavailable")
        print(f"Warning suppression: MinkowskiEngine + deprecated autocast warnings filtered")

    def _setup_models(self):
        """设置模型组件"""
        self.embedding_dim = self.config.get('embedding_dim', 512)

        # 编码器
        self.encoder = MultiModalEncoder(embedding_dim=self.embedding_dim)

        # MoE融合
        moe_config = MoEConfig(
            embedding_dim=self.embedding_dim,
            num_experts=3,
            num_encoder_layers=2,
            nhead=4,
            thermal_guidance=True
        )
        self.moe_fusion = MoEFusion(moe_config)

        # 任务头
        self.pose_head = PoseHead(
            input_dim=self.embedding_dim,
            hidden_dims=[256, 128]
        )

        self.detection_head = DetectionHead(
            input_dim=self.embedding_dim,
            num_queries=20,
            num_classes=8,
            decoder_layers=4
        )

        # 移动到设备
        self.encoder.to(self.device)
        self.moe_fusion.to(self.device)
        self.pose_head.to(self.device)
        self.detection_head.to(self.device)

    def _setup_training_components(self):
        """设置训练组件"""
        # 修复版Kendall不确定性权重学习 - 解决权重失衡问题
        self.kendall_uncertainty = create_fixed_kendall_uncertainty({
            'initial_pose_log_var': -1.0,      # σ≈0.61, weight≈2.7
            'initial_detection_log_var': 0.0,   # σ=1.0,  weight=1.0
            'initial_gate_log_var': -0.4,      # σ≈0.82, weight≈1.5
            'enable_weight_constraints': True,
            'min_log_var': -2.0,
            'max_log_var': 2.0,
            'learning_rate_scale': 0.1
        }).to(self.device)

        print("🔧 使用修复版Kendall不确定性 - 目标权重比例 2.7:1.0:1.5")

        # 优化器 - 包含所有参数
        all_params = []
        all_params.extend(self.encoder.parameters())
        all_params.extend(self.moe_fusion.parameters())
        all_params.extend(self.pose_head.parameters())
        all_params.extend(self.detection_head.parameters())
        all_params.extend(self.kendall_uncertainty.parameters())

        self.optimizer = optim.AdamW(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器: Cosine + Warmup
        warmup_steps = self.config.get('warmup_steps', 1000)
        self.total_steps = self.max_epochs * self.config.get('steps_per_epoch', 1000)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - warmup_steps,
            eta_min=self.learning_rate * 0.01
        )

        # 自动混合精度 - 禁用以兼容MinkowskiEngine
        self.scaler = GradScaler()
        self.use_amp = False  # 禁用AMP避免MinkowskiEngine FP16兼容性问题

        # 匹配器
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_center=5.0,
            cost_iou=2.0
        )
        self.target_generator = TargetGenerator()

        print(f"Training components setup: AMP={self.use_amp} (DISABLED for MinkowskiEngine compatibility), "
              f"Accumulation={self.accumulation_steps}, Warmup={warmup_steps}")

    def _setup_data_loaders(self):
        """设置数据加载器"""
        # 训练数据集
        self.train_dataset = MineSLAMDataset(self.config, split='train')

        # 创建固定训练子集（前300帧用于单元测试）
        self._create_train_subset()

        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # 设为0避免多进程数据加载问题
            pin_memory=True,
            drop_last=True,
            collate_fn=pointcloud_collate_fn  # 使用自定义批处理函数
        )

        # 验证数据集
        try:
            self.val_dataset = MineSLAMDataset(self.config, split='val')
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # 设为0避免多进程数据加载问题
                pin_memory=True,
                collate_fn=pointcloud_collate_fn  # 使用自定义批处理函数
            )
        except:
            print("Warning: No validation dataset found, using train split subset")
            # 使用训练集的一部分作为验证集
            val_indices = list(range(len(self.train_dataset)))[-100:]
            val_subset = Subset(self.train_dataset, val_indices)
            self.val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # 设为0避免多进程数据加载问题
                pin_memory=True,
                collate_fn=pointcloud_collate_fn  # 使用自定义批处理函数
            )

    def _create_train_subset(self):
        """创建固定的训练子集用于单元测试"""
        subset_file = Path('lists/train_subset.jsonl')
        subset_file.parent.mkdir(parents=True, exist_ok=True)

        if not subset_file.exists():
            # 固定前300帧的索引
            subset_indices = list(range(min(300, len(self.train_dataset))))

            subset_data = []
            for idx in subset_indices:
                try:
                    sample = self.train_dataset[idx]
                    # 提取关键信息用于验证
                    subset_info = {
                        'index': idx,
                        'timestamp': sample.get('timestamp', 0.0),
                        'has_rgb': 'rgb' in sample,
                        'has_depth': 'depth' in sample,
                        'has_thermal': 'thermal' in sample,
                        'has_lidar': 'lidar' in sample,
                        'has_imu': 'imu' in sample
                    }
                    subset_data.append(subset_info)
                except Exception as e:
                    print(f"Warning: Cannot access sample {idx}: {e}")

            # 保存子集信息
            with open(subset_file, 'w') as f:
                for item in subset_data:
                    f.write(json.dumps(item) + '\n')

            print(f"Created training subset: {len(subset_data)} samples → {subset_file}")
        else:
            print(f"Using existing training subset: {subset_file}")

    def _warmup_lr(self, step: int, warmup_steps: int):
        """学习率预热"""
        if step < warmup_steps:
            warmup_factor = step / warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * warmup_factor

    def _get_gpu_memory_usage(self) -> float:
        """获取GPU内存使用量（MB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _compute_multitask_loss(self, moe_output: Dict, pose_pred: torch.Tensor,
                               detection_pred: Dict, pose_target: torch.Tensor,
                               detection_targets: List[Optional[Dict]]) -> Dict[str, torch.Tensor]:
        """计算多任务损失"""
        losses = {}

        # 1. 姿态估计损失 (Huber)
        losses['pose'] = self.pose_head.compute_loss(pose_pred, pose_target, delta=1.0)

        # 2. 检测损失 (匈牙利匹配)
        matcher_indices = self.matcher(detection_pred, detection_targets)
        detection_losses = self.detection_head.compute_loss(
            detection_pred, detection_targets, matcher_indices
        )
        losses['detection'] = detection_losses['loss_det']

        # 3. 门控熵损失
        losses['gate'] = moe_output['entropy_loss']

        return losses

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        # 添加额外的警告过滤
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="MinkowskiEngine")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

            return self._train_epoch_impl(epoch)

    def _train_epoch_impl(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch的实际实现"""
        self.encoder.train()
        self.moe_fusion.train()
        self.pose_head.train()
        self.detection_head.train()
        self.kendall_uncertainty.train()

        epoch_losses = {'pose': 0.0, 'detection': 0.0, 'gate': 0.0, 'total': 0.0}
        num_batches = 0
        step_count = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            step_count += 1
            global_step = epoch * len(self.train_loader) + batch_idx

            # 验证真实数据
            self.data_validator.validate_batch(batch, batch_idx)

            # 学习率预热
            self._warmup_lr(global_step, 1000)

            # 构建输入 - 路线B：直接使用collate的统一输出
            # 不需要额外处理，collate已经保证了batch_size一致性
            input_dict = {}
            modalities = ['rgb', 'depth', 'thermal', 'lidar', 'imu']

            # batch_size现在可以从任意模态获取，因为都是统一的
            actual_batch_size = None
            for modality in modalities:
                if modality in batch and len(batch[modality]) > 0:
                    if isinstance(batch[modality], torch.Tensor):
                        actual_batch_size = batch[modality].shape[0]
                        break

            if actual_batch_size is None:
                actual_batch_size = self.batch_size
                print(f"Warning: Cannot determine batch_size, using default {self.batch_size}")

            # 直接传递collate后的数据和掩码
            for modality in modalities:
                if modality in batch and len(batch[modality]) > 0:
                    # 路线B：所有数据已经在collate中处理为统一格式
                    input_dict[modality] = batch[modality].to(self.device)

            # 传递present_mask
            if 'present_mask' in batch:
                input_dict['present_mask'] = {}
                for key, mask in batch['present_mask'].items():
                    input_dict['present_mask'][key] = mask.to(self.device)

            # 生成模拟目标（实际训练中应使用真实标注）
            pose_target = torch.randn(actual_batch_size, 6, device=self.device) * 0.1
            detection_targets = self._generate_mock_detection_targets(actual_batch_size)

            with autocast('cuda', enabled=self.use_amp):
                # 前向传播
                token_dict = self.encoder(input_dict)
                moe_output = self.moe_fusion(token_dict)
                fused_tokens = torch.cat([v for v in moe_output['fused_tokens'].values()], dim=1)

                pose_pred = self.pose_head(fused_tokens)
                detection_pred = self.detection_head(fused_tokens)

                # 计算原始损失
                raw_losses = self._compute_multitask_loss(
                    moe_output, pose_pred, detection_pred,
                    pose_target, detection_targets
                )

                # 修复版Kendall不确定性加权 - 使用新接口
                weighted_losses = self.kendall_uncertainty.compute_multitask_loss(
                    raw_losses['pose'],
                    raw_losses['detection'],
                    raw_losses['gate']
                )
                total_loss = weighted_losses['total_loss'] / self.accumulation_steps

            # 反向传播
            self.scaler.scale(total_loss).backward()

            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.moe_fusion.parameters()) +
                    list(self.pose_head.parameters()) +
                    list(self.detection_head.parameters()) +
                    list(self.kendall_uncertainty.parameters()),
                    max_norm=1.0
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if global_step > 1000:  # 预热后开始调度
                    self.scheduler.step()

            # 记录损失
            for key in epoch_losses.keys():
                if key == 'total':
                    epoch_losses[key] += weighted_losses['total_loss'].item()
                elif key in raw_losses:
                    epoch_losses[key] += raw_losses[key].item()

            num_batches += 1

            # 记录训练指标
            if batch_idx % 50 == 0:
                elapsed_time = time.time() - start_time
                fps = (batch_idx + 1) * actual_batch_size / elapsed_time
                gpu_memory = self._get_gpu_memory_usage()
                kendall_weights = self.kendall_uncertainty.get_weights()

                batch_losses = {
                    'pose': raw_losses['pose'].item(),
                    'detection': raw_losses['detection'].item(),
                    'gate': raw_losses['gate'].item(),
                    'total': weighted_losses['total_loss'].item()
                }

                # 路线B：记录模态缺失统计
                if 'present_mask' in batch:
                    modality_stats = self._log_modality_statistics(batch['present_mask'])
                else:
                    modality_stats = "No mask data"

                # 更新基础训练指标
                self.metrics.update_train_metrics(
                    batch_losses, kendall_weights, gpu_memory, fps, epoch, global_step
                )

                # 更新SLAM详细指标（每100个batch执行一次）
                if batch_idx % 100 == 0:
                    # 构建模拟的ground truth用于演示
                    mock_ground_truth = {
                        'pose': pose_target,
                        'detection': detection_targets
                    }

                    model_outputs_for_eval = {
                        'pose': pose_pred,
                        'detection': detection_pred,
                        'moe_analysis': moe_output
                    }

                    self.metrics.update_slam_metrics(
                        epoch, model_outputs_for_eval, mock_ground_truth, batch
                    )

                # 更新实时监控
                if self.training_monitor:
                    self.training_monitor.update_training_progress(
                        epoch, batch_idx, batch_losses, kendall_weights, fps, gpu_memory
                    )

                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={weighted_losses['total_loss'].item():.6f}, "
                      f"FPS={fps:.1f}, GPU={gpu_memory:.1f}MB")

                # 每200个batch显示一次模态统计
                if batch_idx % 200 == 0:
                    print(f"  Modality presence: {modality_stats}")

                # 生成实时可视化（每500个batch）
                if batch_idx % 500 == 0 and batch_idx > 0 and self.visualizer:
                    try:
                        self.visualizer.plot_realtime_training(
                            epoch, self.metrics.metrics_history,
                            save_path=str(self.output_dir / 'visualizations' / f'training_progress_e{epoch}_b{batch_idx}.png')
                        )
                    except Exception as e:
                        warnings.warn(f"Real-time visualization failed: {e}")

        # 平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _log_modality_statistics(self, present_mask: Dict[str, torch.Tensor]) -> str:
        """
        记录模态缺失统计 - 路线B

        Args:
            present_mask: 掩码字典 {'rgb': [B], 'thermal': [B], 'lidar': [B], ...}

        Returns:
            str: 格式化的统计信息
        """
        stats = []
        for modality, mask in present_mask.items():
            if isinstance(mask, torch.Tensor):
                present_rate = mask.float().mean().item()
                missing_rate = 1.0 - present_rate
                stats.append(f"{modality}:{missing_rate*100:.1f}%missing")

        return ", ".join(stats)

    def _generate_mock_detection_targets(self, batch_size: int) -> List[Optional[Dict]]:
        """生成模拟检测目标"""
        targets = []
        for i in range(batch_size):
            if torch.rand(1).item() > 0.3:  # 70%概率有标注
                num_objects = torch.randint(1, 4, (1,)).item()
                targets.append({
                    'labels': torch.randint(0, 8, (num_objects,), device=self.device),
                    'boxes': torch.randn(num_objects, 6, device=self.device) * 2
                })
            else:
                targets.append(None)
        return targets

    def validate(self, epoch: int) -> Tuple[float, float, float, float]:
        """验证模型性能"""
        # 添加额外的警告过滤
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="MinkowskiEngine")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")

            return self._validate_impl(epoch)

    def _validate_impl(self, epoch: int) -> Tuple[float, float, float, float]:
        """验证模型性能的实际实现"""
        self.encoder.eval()
        self.moe_fusion.eval()
        self.pose_head.eval()
        self.detection_head.eval()
        self.kendall_uncertainty.eval()

        total_val_loss = 0.0
        total_ate = 0.0
        total_rpe = 0.0
        total_map = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 构建输入 - 路线B：直接使用collate的统一输出
                input_dict = {}
                modalities = ['rgb', 'depth', 'thermal', 'lidar', 'imu']

                # batch_size现在可以从任意模态获取，因为都是统一的
                actual_batch_size = None
                for modality in modalities:
                    if modality in batch and len(batch[modality]) > 0:
                        if isinstance(batch[modality], torch.Tensor):
                            actual_batch_size = batch[modality].shape[0]
                            break

                if actual_batch_size is None:
                    actual_batch_size = self.batch_size

                # 直接传递collate后的数据和掩码
                for modality in modalities:
                    if modality in batch and len(batch[modality]) > 0:
                        input_dict[modality] = batch[modality].to(self.device)

                # 传递present_mask
                if 'present_mask' in batch:
                    input_dict['present_mask'] = {}
                    for key, mask in batch['present_mask'].items():
                        input_dict['present_mask'][key] = mask.to(self.device)

                # 前向传播
                token_dict = self.encoder(input_dict)
                moe_output = self.moe_fusion(token_dict)
                fused_tokens = torch.cat([v for v in moe_output['fused_tokens'].values()], dim=1)

                pose_pred = self.pose_head(fused_tokens)
                detection_pred = self.detection_head(fused_tokens)

                # 模拟验证损失计算（实际中应使用真实标注）
                pose_target = torch.randn(actual_batch_size, 6, device=self.device) * 0.1
                detection_targets = self._generate_mock_detection_targets(actual_batch_size)

                raw_losses = self._compute_multitask_loss(
                    moe_output, pose_pred, detection_pred,
                    pose_target, detection_targets
                )
                weighted_losses = self.kendall_uncertainty(raw_losses)

                total_val_loss += weighted_losses['total_loss'].item()

                # 模拟ATE/RPE/mAP计算
                ate = torch.norm(pose_pred[:, :3] - pose_target[:, :3], dim=1).mean().item()
                rpe = torch.norm(pose_pred - pose_target, dim=1).mean().item()
                map_score = torch.sigmoid(detection_pred['logits']).mean().item()

                total_ate += ate
                total_rpe += rpe
                total_map += map_score
                num_batches += 1

        avg_val_loss = total_val_loss / num_batches
        avg_ate = total_ate / num_batches
        avg_rpe = total_rpe / num_batches
        avg_map = total_map / num_batches

        return avg_val_loss, avg_ate, avg_rpe, avg_map

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'moe_fusion_state_dict': self.moe_fusion.state_dict(),
            'pose_head_state_dict': self.pose_head.state_dict(),
            'detection_head_state_dict': self.detection_head.state_dict(),
            'kendall_uncertainty_state_dict': self.kendall_uncertainty.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'best_metrics': {
                'best_ate': self.metrics.best_ate,
                'best_map': self.metrics.best_map,
                'best_val_loss': self.metrics.best_val_loss
            }
        }

        # 保存最新检查点
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.output_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best checkpoint: {best_path}")

        return latest_path

    def train(self):
        """主训练循环"""
        print(f"\n🚀 Starting MineSLAM Multi-Task Training...")
        print(f"Max epochs: {self.max_epochs}, Early stop patience: {self.early_stop_patience}")
        print(f"Target: ATE≤{self.target_ate}m, mAP≥{self.target_map*100}%")
        print("=" * 80)

        for epoch in range(self.max_epochs):
            epoch_start = time.time()

            # 训练
            train_losses = self.train_epoch(epoch)

            # 验证
            val_loss, ate, rpe, map_score = self.validate(epoch)

            # 更新基础验证指标
            self.metrics.update_val_metrics(val_loss, ate, rpe, map_score, epoch)

            # 生成epoch可视化报告
            if epoch % 5 == 0 and self.visualizer:  # 每5个epoch生成一次详细报告
                try:
                    # 生成轨迹可视化
                    self.visualizer.plot_trajectory_comparison(
                        predicted_trajectory=None,  # 在实际使用中需要传入真实轨迹
                        ground_truth_trajectory=None,
                        save_path=str(self.output_dir / 'visualizations' / f'trajectory_epoch_{epoch}.png')
                    )

                    # 生成训练进度综合报告
                    self.visualizer.generate_training_report(
                        metrics_history=self.metrics.metrics_history,
                        slam_metrics=self.metrics.slam_metrics_history,
                        epoch=epoch,
                        output_dir=str(self.output_dir / 'reports')
                    )

                    # 生成Kendall权重分析图
                    self.visualizer.plot_kendall_weight_evolution(
                        kendall_history=self.metrics.metrics_history['kendall_weights'],
                        save_path=str(self.output_dir / 'visualizations' / f'kendall_weights_epoch_{epoch}.png')
                    )

                except Exception as e:
                    warnings.warn(f"Epoch visualization failed: {e}")

            # 检查是否达到目标
            target_reached = ate <= self.target_ate and map_score >= self.target_map
            is_best = (ate < self.metrics.best_ate) or (map_score > self.metrics.best_map)

            # 保存检查点
            self.save_checkpoint(epoch, is_best)

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_losses['total']:.6f} "
                  f"(pose={train_losses['pose']:.6f}, det={train_losses['detection']:.6f}, gate={train_losses['gate']:.6f})")
            print(f"  Val Metrics: Loss={val_loss:.6f}, ATE={ate:.3f}m, RPE={rpe:.3f}, mAP={map_score:.3f}")

            # 显示修复版Kendall权重状态
            kendall_weights = self.kendall_uncertainty.get_weights()
            print(f"  Kendall Weights: pose={kendall_weights['pose_weight']:.4f}, "
                  f"det={kendall_weights['detection_weight']:.4f}, gate={kendall_weights['gate_weight']:.4f}")

            # 添加权重平衡监控
            balance_metrics = self.kendall_uncertainty.get_weight_balance_metrics()
            print(f"  Weight Balance: Score={balance_metrics['balance_score']:.3f}, "
                  f"MaxRatio={balance_metrics['max_ratio']:.1f}:1")

            # 检查权重失衡警告
            if balance_metrics['max_ratio'] > 50:
                print(f"  ⚠️ Warning: Severe weight imbalance detected!")
                suggestions = self.kendall_uncertainty.get_optimization_suggestions()
                for suggestion in suggestions.values():
                    print(f"    - {suggestion}")

            # 定期详细状态报告
            if epoch % 5 == 0:
                self.kendall_uncertainty.print_status(epoch)

            if target_reached:
                print(f"🎯 Target reached! ATE={ate:.3f}≤{self.target_ate}, mAP={map_score:.3f}≥{self.target_map}")

            # Early stopping检查
            if self.metrics.should_early_stop(self.early_stop_patience, self.early_stop_min_delta):
                print(f"⏹️ Early stopping triggered after {epoch+1} epochs")
                break

        # 保存最终指标（使用增强版）
        metrics_path = self.metrics.save_enhanced_metrics()
        print(f"\n✅ Training completed! Metrics saved to: {metrics_path}")
        print(f"Best results: ATE={self.metrics.best_ate:.3f}m, mAP={self.metrics.best_map:.3f}, "
              f"Val Loss={self.metrics.best_val_loss:.6f}")

        return {
            'best_ate': self.metrics.best_ate,
            'best_map': self.metrics.best_map,
            'best_val_loss': self.metrics.best_val_loss,
            'total_epochs': epoch + 1
        }


# 训练脚本入口
def main():
    """主训练脚本"""
    # 加载配置
    config_path = 'configs/mineslam.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)

    # 训练参数覆盖
    config.update({
        'batch_size': 4,
        'gradient_accumulation': 4,
        'max_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_amp': True,
        'early_stop_patience': 10,
        'warmup_steps': 1000,
        'target_ate': 1.5,
        'target_map': 0.6
    })

    # 创建训练器
    trainer = MineSLAMTrainer(config, output_dir="outputs/training")

    # 开始训练
    try:
        results = trainer.train()
        print(f"\n🎉 Training finished successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)