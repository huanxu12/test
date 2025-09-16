"""
Kendall Uncertainty Loss Weighting for Multi-Task Learning
基于Kendall不确定性的多任务损失权重学习
参考: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional


class KendallUncertainty(nn.Module):
    """
    Kendall不确定性损失权重学习器
    通过学习任务相关的不确定性参数σ来动态平衡多任务损失
    """

    def __init__(self, num_tasks: int = 3,
                 init_log_var: float = 0.0,
                 min_log_var: float = -10.0,
                 max_log_var: float = 5.0):
        """
        Args:
            num_tasks: 任务数量（pose, detection, gate）
            init_log_var: log(σ²)的初始值
            min_log_var: log(σ²)的最小值（避免权重过大）
            max_log_var: log(σ²)的最大值（避免权重过小）
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # 可学习的log(σ²)参数
        self.log_vars = nn.Parameter(
            torch.full((num_tasks,), init_log_var, dtype=torch.float32)
        )

        # 任务名称映射
        self.task_names = ['pose', 'detection', 'gate']

        print(f"KendallUncertainty initialized: {num_tasks} tasks, "
              f"init_log_var={init_log_var}, range=[{min_log_var}, {max_log_var}]")

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算带不确定性权重的总损失

        Args:
            losses: 各任务损失字典
                - 'pose': 姿态估计损失
                - 'detection': 检测损失
                - 'gate': 门控损失

        Returns:
            weighted_losses: 加权损失字典
        """
        # 限制log_vars在合理范围内
        clamped_log_vars = torch.clamp(self.log_vars, self.min_log_var, self.max_log_var)

        # 计算不确定性权重: w_i = 1/(2σ²)
        uncertainties = torch.exp(clamped_log_vars)  # σ²
        weights = 1.0 / (2.0 * uncertainties)       # 1/(2σ²)

        weighted_losses = {}
        total_loss = 0.0

        for i, task_name in enumerate(self.task_names):
            if task_name in losses:
                # 主要损失项: w_i * L_i
                task_loss = losses[task_name]
                weighted_task_loss = weights[i] * task_loss

                # 正则化项: log(σ)（防止σ趋向0）
                regularization = 0.5 * clamped_log_vars[i]

                # 总的加权损失
                final_task_loss = weighted_task_loss + regularization

                weighted_losses[f'weighted_{task_name}'] = final_task_loss
                weighted_losses[f'{task_name}_weight'] = weights[i]
                weighted_losses[f'{task_name}_sigma'] = torch.sqrt(uncertainties[i])

                total_loss += final_task_loss

        weighted_losses['total_loss'] = total_loss

        return weighted_losses

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重值（用于日志记录）"""
        with torch.no_grad():
            clamped_log_vars = torch.clamp(self.log_vars, self.min_log_var, self.max_log_var)
            uncertainties = torch.exp(clamped_log_vars)
            weights = 1.0 / (2.0 * uncertainties)

            weight_dict = {}
            for i, task_name in enumerate(self.task_names):
                weight_dict[f'{task_name}_weight'] = weights[i].item()
                weight_dict[f'{task_name}_sigma'] = torch.sqrt(uncertainties[i]).item()
                weight_dict[f'{task_name}_log_var'] = clamped_log_vars[i].item()

            return weight_dict

    def reset_parameters(self, init_log_var: float = 0.0):
        """重置参数"""
        with torch.no_grad():
            self.log_vars.fill_(init_log_var)


class AdaptiveKendallUncertainty(KendallUncertainty):
    """
    自适应Kendall不确定性
    根据训练过程动态调整不确定性参数的学习率
    """

    def __init__(self, num_tasks: int = 3,
                 init_log_var: float = 0.0,
                 min_log_var: float = -10.0,
                 max_log_var: float = 5.0,
                 adaptation_rate: float = 0.01,
                 target_balance: float = 0.33):
        """
        Args:
            adaptation_rate: 自适应学习率
            target_balance: 目标平衡比例（理想情况下每个任务损失占总损失的1/3）
        """
        super().__init__(num_tasks, init_log_var, min_log_var, max_log_var)

        self.adaptation_rate = adaptation_rate
        self.target_balance = target_balance

        # 记录损失历史用于自适应调整
        self.loss_history = {name: [] for name in self.task_names}
        self.history_length = 100

        print(f"AdaptiveKendallUncertainty initialized: "
              f"adaptation_rate={adaptation_rate}, target_balance={target_balance}")

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        自适应计算带不确定性权重的总损失
        """
        # 更新损失历史
        with torch.no_grad():
            for task_name in self.task_names:
                if task_name in losses:
                    self.loss_history[task_name].append(losses[task_name].item())
                    if len(self.loss_history[task_name]) > self.history_length:
                        self.loss_history[task_name].pop(0)

        # 自适应调整log_vars
        self._adaptive_adjustment()

        # 计算加权损失
        return super().forward(losses)

    def _adaptive_adjustment(self):
        """基于损失历史自适应调整log_vars"""
        if not self.training:
            return

        with torch.no_grad():
            # 计算各任务的相对损失比例
            recent_losses = {}
            total_recent_loss = 0.0

            for task_name in self.task_names:
                if len(self.loss_history[task_name]) >= 10:  # 至少有10个样本
                    recent_avg = sum(self.loss_history[task_name][-10:]) / 10
                    recent_losses[task_name] = recent_avg
                    total_recent_loss += recent_avg

            if total_recent_loss > 0:
                for i, task_name in enumerate(self.task_names):
                    if task_name in recent_losses:
                        current_ratio = recent_losses[task_name] / total_recent_loss

                        # 如果某任务损失比例过高，增加其log_var（降低权重）
                        if current_ratio > self.target_balance * 1.5:
                            self.log_vars[i] += self.adaptation_rate
                        # 如果某任务损失比例过低，降低其log_var（增加权重）
                        elif current_ratio < self.target_balance * 0.5:
                            self.log_vars[i] -= self.adaptation_rate

                        # 确保在合理范围内
                        self.log_vars[i] = torch.clamp(
                            self.log_vars[i], self.min_log_var, self.max_log_var
                        )


# 工厂函数
def create_kendall_uncertainty(uncertainty_type: str = 'basic',
                              num_tasks: int = 3,
                              **kwargs) -> KendallUncertainty:
    """
    创建Kendall不确定性模块

    Args:
        uncertainty_type: 'basic' 或 'adaptive'
        num_tasks: 任务数量
        **kwargs: 其他参数

    Returns:
        uncertainty_module: Kendall不确定性模块
    """
    if uncertainty_type == 'adaptive':
        return AdaptiveKendallUncertainty(num_tasks=num_tasks, **kwargs)
    else:
        return KendallUncertainty(num_tasks=num_tasks, **kwargs)


# 模块测试
if __name__ == '__main__':
    print("🧪 Testing Kendall Uncertainty...")

    # 创建基础Kendall不确定性
    kendall = KendallUncertainty(num_tasks=3, init_log_var=0.0)

    # 模拟损失
    losses = {
        'pose': torch.tensor(0.1),
        'detection': torch.tensor(0.5),
        'gate': torch.tensor(0.02)
    }

    print(f"Input losses: {losses}")

    # 计算加权损失
    weighted_losses = kendall(losses)

    print(f"\nWeighted losses:")
    for key, value in weighted_losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value}")

    # 获取权重信息
    weights = kendall.get_weights()
    print(f"\nCurrent weights:")
    for key, value in weights.items():
        print(f"  {key}: {value:.6f}")

    # 测试自适应版本
    print(f"\n🔧 Testing Adaptive Kendall Uncertainty...")
    adaptive_kendall = AdaptiveKendallUncertainty(
        num_tasks=3,
        adaptation_rate=0.01,
        target_balance=0.33
    )

    adaptive_kendall.train()

    # 模拟多轮训练
    for step in range(20):
        # 模拟不平衡的损失
        losses = {
            'pose': torch.tensor(0.1 * (1 + 0.1 * step)),
            'detection': torch.tensor(0.5 * (1 - 0.05 * step)),
            'gate': torch.tensor(0.02)
        }

        weighted_losses = adaptive_kendall(losses)

        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Total loss: {weighted_losses['total_loss'].item():.6f}")
            weights = adaptive_kendall.get_weights()
            for task in ['pose', 'detection', 'gate']:
                w = weights[f'{task}_weight']
                s = weights[f'{task}_sigma']
                print(f"  {task}: weight={w:.4f}, sigma={s:.4f}")

    print("\n✅ Kendall Uncertainty test completed!")