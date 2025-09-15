"""
Advanced Training Strategy for MineSLAM
集成自动混合精度、梯度累积、学习率调度、早停等先进训练策略
"""

import os
import sys
import math
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler


class CosineLRWithWarmup(_LRScheduler):
    """
    Cosine Annealing LR with Linear Warmup
    先线性预热，再余弦退火
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            total_steps: 总训练步数
            eta_min: 最小学习率
            last_epoch: 上次epoch编号
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [0.0 for _ in self.base_lrs]

        if self.last_epoch < self.warmup_steps:
            # 线性预热阶段
            return [base_lr * self.last_epoch / self.warmup_steps
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            cosine_steps = self.total_steps - self.warmup_steps
            current_cosine_step = self.last_epoch - self.warmup_steps

            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * current_cosine_step / cosine_steps)) / 2
                    for base_lr in self.base_lrs]


class EarlyStopping:
    """
    Early Stopping with Multiple Metrics
    监控多个指标的早停机制
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 restore_best_weights: bool = True, mode: str = 'min'):
        """
        Args:
            patience: 容忍的无改善epoch数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        检查是否早停

        Args:
            score: 当前分数
            model: 模型（用于保存最佳状态）

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self._is_improved(score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
                print(f"✅ Restored best weights (score: {self.best_score:.6f})")

        return self.early_stop

    def _is_improved(self, score: float) -> bool:
        """判断分数是否改善"""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def _save_checkpoint(self, model: nn.Module):
        """保存最佳模型状态"""
        if self.restore_best_weights:
            self.best_state = model.state_dict().copy()


class GradientAccumulator:
    """
    Gradient Accumulation Manager
    梯度累积管理器
    """

    def __init__(self, accumulation_steps: int = 1, clip_grad_norm: float = 1.0):
        """
        Args:
            accumulation_steps: 梯度累积步数
            clip_grad_norm: 梯度裁剪阈值
        """
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.current_step = 0

    def should_step(self) -> bool:
        """检查是否应该执行优化器步骤"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失用于梯度累积"""
        return loss / self.accumulation_steps

    def clip_gradients(self, model: nn.Module) -> float:
        """梯度裁剪"""
        if self.clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.clip_grad_norm
            )
            return grad_norm.item()
        return 0.0

    def reset(self):
        """重置步数计数"""
        self.current_step = 0


class AMPManager:
    """
    Automatic Mixed Precision Manager
    自动混合精度管理器
    """

    def __init__(self, enabled: bool = True, init_scale: float = 2**16,
                 growth_factor: float = 2.0, backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        Args:
            enabled: 是否启用AMP
            init_scale: 初始缩放因子
            growth_factor: 增长因子
            backoff_factor: 回退因子
            growth_interval: 增长间隔
        """
        self.enabled = enabled
        self.scaler = None

        if self.enabled and torch.cuda.is_available():
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            print(f"✅ AMP enabled with init_scale={init_scale}")
        else:
            print("❌ AMP disabled (CUDA not available or manually disabled)")

    def autocast_context(self):
        """返回autocast上下文管理器"""
        return autocast(enabled=self.enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_gradients(self, optimizer: optim.Optimizer):
        """反缩放梯度（用于梯度裁剪）"""
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step_optimizer(self, optimizer: optim.Optimizer):
        """执行优化器步骤"""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.enabled and self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


class TrainingStrategy:
    """
    Integrated Training Strategy
    集成训练策略管理器
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config

        # 训练参数
        self.max_epochs = config.get('max_epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)

        # 梯度累积
        accumulation_config = config.get('gradient_accumulation', {})
        self.accumulator = GradientAccumulator(
            accumulation_steps=accumulation_config.get('steps', 4),
            clip_grad_norm=accumulation_config.get('clip_norm', 1.0)
        )

        # AMP设置
        amp_config = config.get('amp', {})
        self.amp_manager = AMPManager(
            enabled=amp_config.get('enabled', True),
            init_scale=amp_config.get('init_scale', 2**16)
        )

        # 早停设置
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 1e-4),
            restore_best_weights=early_stop_config.get('restore_best_weights', True),
            mode=early_stop_config.get('mode', 'min')
        )

        # 学习率调度器（在create_scheduler中初始化）
        self.scheduler = None

        print(f"TrainingStrategy initialized:")
        print(f"  Max epochs: {self.max_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Gradient accumulation: {self.accumulator.accumulation_steps} steps")
        print(f"  AMP enabled: {self.amp_manager.enabled}")
        print(f"  Early stopping patience: {self.early_stopping.patience}")

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adamw')

        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999)),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        print(f"✅ Created {optimizer_type.upper()} optimizer")
        return optimizer

    def create_scheduler(self, optimizer: optim.Optimizer,
                        total_steps: int) -> _LRScheduler:
        """创建学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine_warmup')

        if scheduler_type == 'cosine_warmup':
            warmup_steps = scheduler_config.get('warmup_steps', 1000)
            eta_min = scheduler_config.get('eta_min', self.learning_rate * 0.01)

            self.scheduler = CosineLRWithWarmup(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                eta_min=eta_min
            )
            print(f"✅ Created Cosine LR with Warmup scheduler (warmup: {warmup_steps} steps)")

        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=scheduler_config.get('eta_min', self.learning_rate * 0.01)
            )
            print(f"✅ Created Cosine Annealing LR scheduler")

        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', total_steps // 3),
                gamma=scheduler_config.get('gamma', 0.1)
            )
            print(f"✅ Created Step LR scheduler")

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        return self.scheduler

    def forward_pass(self, model: nn.Module, inputs: Dict[str, torch.Tensor]) -> Any:
        """带AMP的前向传播"""
        with self.amp_manager.autocast_context():
            outputs = model(inputs)
        return outputs

    def backward_pass(self, loss: torch.Tensor, model: nn.Module,
                     optimizer: optim.Optimizer) -> Dict[str, float]:
        """带梯度累积和AMP的反向传播"""
        # 缩放损失用于梯度累积
        scaled_loss = self.accumulator.scale_loss(loss)

        # AMP缩放损失
        scaled_loss = self.amp_manager.scale_loss(scaled_loss)

        # 反向传播
        scaled_loss.backward()

        metrics = {'loss': loss.item(), 'scaled_loss': scaled_loss.item()}

        # 检查是否应该执行优化器步骤
        if self.accumulator.should_step():
            # 反缩放梯度（用于梯度裁剪）
            self.amp_manager.unscale_gradients(optimizer)

            # 梯度裁剪
            grad_norm = self.accumulator.clip_gradients(model)
            metrics['grad_norm'] = grad_norm

            # 优化器步骤
            self.amp_manager.step_optimizer(optimizer)
            optimizer.zero_grad()

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            metrics['lr'] = optimizer.param_groups[0]['lr']
            metrics['amp_scale'] = self.amp_manager.get_scale()

        return metrics

    def check_early_stopping(self, score: float, model: nn.Module) -> bool:
        """检查早停条件"""
        return self.early_stopping(score, model)

    def get_training_state(self) -> Dict[str, Any]:
        """获取训练状态（用于保存检查点）"""
        state = {
            'accumulator_step': self.accumulator.current_step,
            'early_stopping_counter': self.early_stopping.counter,
            'early_stopping_best_score': self.early_stopping.best_score,
        }

        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()

        if self.amp_manager.scaler is not None:
            state['scaler_state'] = self.amp_manager.scaler.state_dict()

        return state

    def load_training_state(self, state: Dict[str, Any],
                           scheduler: _LRScheduler = None):
        """加载训练状态（用于恢复检查点）"""
        self.accumulator.current_step = state.get('accumulator_step', 0)
        self.early_stopping.counter = state.get('early_stopping_counter', 0)
        self.early_stopping.best_score = state.get('early_stopping_best_score', None)

        if 'scheduler_state' in state and scheduler is not None:
            scheduler.load_state_dict(state['scheduler_state'])

        if 'scaler_state' in state and self.amp_manager.scaler is not None:
            self.amp_manager.scaler.load_state_dict(state['scaler_state'])


def create_training_strategy(config: Dict) -> TrainingStrategy:
    """
    Factory function to create training strategy

    Args:
        config: 训练配置

    Returns:
        配置好的训练策略
    """
    return TrainingStrategy(config)


# 示例配置
EXAMPLE_CONFIG = {
    'max_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,

    'gradient_accumulation': {
        'steps': 4,
        'clip_norm': 1.0
    },

    'amp': {
        'enabled': True,
        'init_scale': 2**16
    },

    'early_stopping': {
        'patience': 10,
        'min_delta': 1e-4,
        'restore_best_weights': True,
        'mode': 'min'
    },

    'optimizer': {
        'type': 'adamw',
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },

    'scheduler': {
        'type': 'cosine_warmup',
        'warmup_steps': 1000,
        'eta_min': 1e-6
    }
}


if __name__ == '__main__':
    print("🧪 Testing Training Strategy...")

    # 创建示例模型
    model = torch.nn.Linear(10, 1)

    # 创建训练策略
    strategy = create_training_strategy(EXAMPLE_CONFIG)

    # 创建优化器和调度器
    optimizer = strategy.create_optimizer(model)
    scheduler = strategy.create_scheduler(optimizer, total_steps=5000)

    print(f"\n✅ Training strategy test completed!")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__}")
    print(f"AMP enabled: {strategy.amp_manager.enabled}")
    print(f"Gradient accumulation: {strategy.accumulator.accumulation_steps} steps")