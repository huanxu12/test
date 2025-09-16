"""
Fixed Kendall Uncertainty Loss Weighting for Multi-Task Learning
修复版Kendall不确定性多任务损失权重学习 - 解决严重权重失衡问题

原问题: 位姿权重11013 vs 检测权重0.003 (比例3,269,017:1)
修复方案: 保守平衡初始化 + 权重约束 + 监控诊断

参考: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple


class FixedKendallUncertainty(nn.Module):
    """
    修复版Kendall不确定性权重学习器
    解决权重严重失衡问题，实现平衡的多任务学习
    """

    def __init__(self,
                 initial_pose_log_var: float = -1.0,
                 initial_detection_log_var: float = 0.0,
                 initial_gate_log_var: float = -0.4,
                 enable_weight_constraints: bool = True,
                 min_log_var: float = -2.0,
                 max_log_var: float = 2.0,
                 learning_rate_scale: float = 0.1):
        """
        Args:
            initial_pose_log_var: 位姿任务初始log方差 (-1.0 → σ≈0.61, weight≈2.7)
            initial_detection_log_var: 检测任务初始log方差 (0.0 → σ=1.0, weight=1.0)
            initial_gate_log_var: 门控任务初始log方差 (-0.4 → σ≈0.82, weight≈1.5)
            enable_weight_constraints: 是否启用权重范围约束
            min_log_var: log方差最小值 (防止权重过大)
            max_log_var: log方差最大值 (防止权重过小)
            learning_rate_scale: Kendall参数学习率缩放因子
        """
        super().__init__()

        # 可学习的log方差参数 (保守平衡初始化)
        self.pose_log_var = nn.Parameter(torch.tensor(initial_pose_log_var, dtype=torch.float32))
        self.detection_log_var = nn.Parameter(torch.tensor(initial_detection_log_var, dtype=torch.float32))
        self.gate_log_var = nn.Parameter(torch.tensor(initial_gate_log_var, dtype=torch.float32))

        # 约束参数
        self.enable_constraints = enable_weight_constraints
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.lr_scale = learning_rate_scale

        # 统计信息
        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('weight_history', torch.zeros(100, 3))  # 记录最近100次权重

        print(f"🔧 Fixed Kendall Uncertainty initialized:")
        print(f"   Pose: log_var={initial_pose_log_var:.2f} → σ={np.exp(initial_pose_log_var/2):.3f}, weight≈{1/np.exp(initial_pose_log_var):.2f}")
        print(f"   Detection: log_var={initial_detection_log_var:.2f} → σ={np.exp(initial_detection_log_var/2):.3f}, weight≈{1/np.exp(initial_detection_log_var):.2f}")
        print(f"   Gate: log_var={initial_gate_log_var:.2f} → σ={np.exp(initial_gate_log_var/2):.3f}, weight≈{1/np.exp(initial_gate_log_var):.2f}")

    def apply_constraints(self):
        """应用log_var范围约束"""
        if self.enable_constraints:
            with torch.no_grad():
                self.pose_log_var.clamp_(self.min_log_var, self.max_log_var)
                self.detection_log_var.clamp_(self.min_log_var, self.max_log_var)
                self.gate_log_var.clamp_(self.min_log_var, self.max_log_var)

    def get_weights_and_sigmas(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算当前权重和不确定性
        Returns:
            weights: (pose_weight, detection_weight, gate_weight)
            sigmas: (pose_sigma, detection_sigma, gate_sigma)
            log_vars: (pose_log_var, detection_log_var, gate_log_var)
        """
        # 应用约束
        self.apply_constraints()

        # 计算σ = exp(log_var / 2)
        pose_sigma = torch.exp(self.pose_log_var / 2)
        detection_sigma = torch.exp(self.detection_log_var / 2)
        gate_sigma = torch.exp(self.gate_log_var / 2)

        # 计算权重 = 1 / σ²
        pose_weight = 1.0 / (pose_sigma ** 2)
        detection_weight = 1.0 / (detection_sigma ** 2)
        gate_weight = 1.0 / (gate_sigma ** 2)

        weights = torch.stack([pose_weight, detection_weight, gate_weight])
        sigmas = torch.stack([pose_sigma, detection_sigma, gate_sigma])
        log_vars = torch.stack([self.pose_log_var, self.detection_log_var, self.gate_log_var])

        # 更新历史记录
        self._update_history(weights)

        return weights, sigmas, log_vars

    def _update_history(self, weights: torch.Tensor):
        """更新权重历史记录"""
        with torch.no_grad():
            idx = self.update_count % 100
            self.weight_history[idx] = weights.detach()
            self.update_count += 1

    def compute_multitask_loss(self,
                              pose_loss: torch.Tensor,
                              detection_loss: torch.Tensor,
                              gate_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        Args:
            pose_loss: 位姿任务损失
            detection_loss: 检测任务损失
            gate_loss: 门控任务损失
        Returns:
            包含总损失和各组件的字典
        """
        weights, sigmas, log_vars = self.get_weights_and_sigmas()

        pose_weight, detection_weight, gate_weight = weights
        pose_sigma, detection_sigma, gate_sigma = sigmas

        # Kendall多任务损失公式: L = (1/σ²)*loss + log(σ)
        weighted_pose_loss = pose_weight * pose_loss + self.pose_log_var / 2
        weighted_detection_loss = detection_weight * detection_loss + self.detection_log_var / 2
        weighted_gate_loss = gate_weight * gate_loss + self.gate_log_var / 2

        total_loss = weighted_pose_loss + weighted_detection_loss + weighted_gate_loss

        return {
            'total_loss': total_loss,
            'weighted_pose_loss': weighted_pose_loss,
            'weighted_detection_loss': weighted_detection_loss,
            'weighted_gate_loss': weighted_gate_loss,
            'pose_weight': pose_weight,
            'detection_weight': detection_weight,
            'gate_weight': gate_weight,
            'pose_sigma': pose_sigma,
            'detection_sigma': detection_sigma,
            'gate_sigma': gate_sigma,
            'pose_log_var': self.pose_log_var,
            'detection_log_var': self.detection_log_var,
            'gate_log_var': self.gate_log_var
        }

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        兼容原有接口的前向传播

        Args:
            losses: 各任务损失字典
                - 'pose': 姿态估计损失
                - 'detection': 检测损失
                - 'gate': 门控损失

        Returns:
            weighted_losses: 加权损失字典
        """
        pose_loss = losses.get('pose', torch.tensor(0.0, device=self.pose_log_var.device))
        detection_loss = losses.get('detection', torch.tensor(0.0, device=self.detection_log_var.device))
        gate_loss = losses.get('gate', torch.tensor(0.0, device=self.gate_log_var.device))

        result = self.compute_multitask_loss(pose_loss, detection_loss, gate_loss)

        # 转换为原有格式以保持兼容性
        return {
            'total_loss': result['total_loss'],
            'weighted_pose': result['weighted_pose_loss'],
            'weighted_detection': result['weighted_detection_loss'],
            'weighted_gate': result['weighted_gate_loss'],
            'pose_weight': result['pose_weight'],
            'detection_weight': result['detection_weight'],
            'gate_weight': result['gate_weight'],
            'pose_sigma': result['pose_sigma'],
            'detection_sigma': result['detection_sigma'],
            'gate_sigma': result['gate_sigma']
        }

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重值（用于日志记录）"""
        weights, sigmas, log_vars = self.get_weights_and_sigmas()

        return {
            'pose_weight': float(weights[0]),
            'detection_weight': float(weights[1]),
            'gate_weight': float(weights[2]),
            'pose_sigma': float(sigmas[0]),
            'detection_sigma': float(sigmas[1]),
            'gate_sigma': float(sigmas[2]),
            'pose_log_var': float(log_vars[0]),
            'detection_log_var': float(log_vars[1]),
            'gate_log_var': float(log_vars[2])
        }

    def get_weight_balance_metrics(self) -> Dict[str, float]:
        """获取权重平衡分析指标"""
        weights, _, _ = self.get_weights_and_sigmas()

        total_weight = weights.sum()
        if total_weight == 0:
            return {'balance_score': 0.0, 'max_ratio': float('inf'), 'std_ratio': float('inf')}

        # 权重比例
        weight_ratios = weights / total_weight

        # 平衡得分 (理想情况下每个任务权重应该相近)
        ideal_ratio = 1.0 / 3.0
        balance_score = 1.0 - torch.std(weight_ratios - ideal_ratio).item()

        # 最大权重比例
        max_ratio = torch.max(weights) / torch.min(weights)

        # 权重标准差比例
        std_ratio = torch.std(weights) / torch.mean(weights)

        return {
            'balance_score': float(max(0, balance_score)),
            'max_ratio': float(max_ratio),
            'std_ratio': float(std_ratio),
            'pose_ratio': float(weight_ratios[0]),
            'detection_ratio': float(weight_ratios[1]),
            'gate_ratio': float(weight_ratios[2])
        }

    def get_optimization_suggestions(self) -> Dict[str, str]:
        """获取优化建议"""
        balance_metrics = self.get_weight_balance_metrics()
        suggestions = {}

        if balance_metrics['max_ratio'] > 10:
            suggestions['severe_imbalance'] = f"权重比例过大 ({balance_metrics['max_ratio']:.1f}:1), 建议调整初始log_var"

        if balance_metrics['detection_ratio'] < 0.1:
            suggestions['detection_underweight'] = f"检测任务权重过低 ({balance_metrics['detection_ratio']:.3f}), 建议降低detection_log_var"

        if balance_metrics['pose_ratio'] > 0.7:
            suggestions['pose_overweight'] = f"位姿任务权重过高 ({balance_metrics['pose_ratio']:.3f}), 建议提高pose_log_var"

        if balance_metrics['balance_score'] < 0.5:
            suggestions['general_imbalance'] = f"整体权重不平衡 (得分{balance_metrics['balance_score']:.3f}), 建议重新初始化"

        return suggestions

    def print_status(self, epoch: int = None):
        """打印当前状态"""
        weights, sigmas, log_vars = self.get_weights_and_sigmas()
        balance_metrics = self.get_weight_balance_metrics()

        header = f"Kendall Status (Epoch {epoch})" if epoch is not None else "Kendall Status"
        print(f"\n📊 {header}:")
        print(f"   Weights: Pose={weights[0]:.2f}, Detection={weights[1]:.4f}, Gate={weights[2]:.2f}")
        print(f"   Sigmas:  Pose={sigmas[0]:.4f}, Detection={sigmas[1]:.4f}, Gate={sigmas[2]:.4f}")
        print(f"   LogVars: Pose={log_vars[0]:.3f}, Detection={log_vars[1]:.3f}, Gate={log_vars[2]:.3f}")
        print(f"   Balance: Score={balance_metrics['balance_score']:.3f}, MaxRatio={balance_metrics['max_ratio']:.1f}:1")

        # 打印建议
        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print(f"   ⚠️ Suggestions:")
            for key, suggestion in suggestions.items():
                print(f"      - {suggestion}")

    def reset_parameters(self, initial_pose_log_var: float = -1.0,
                        initial_detection_log_var: float = 0.0,
                        initial_gate_log_var: float = -0.4):
        """重置参数到修复版初始值"""
        with torch.no_grad():
            self.pose_log_var.fill_(initial_pose_log_var)
            self.detection_log_var.fill_(initial_detection_log_var)
            self.gate_log_var.fill_(initial_gate_log_var)
            self.update_count.fill_(0)
            self.weight_history.zero_()


# 原有类的修复版别名 (保持向后兼容)
class KendallUncertainty(FixedKendallUncertainty):
    """向后兼容的别名"""
    def __init__(self, num_tasks: int = 3, init_log_var: float = 0.0,
                 min_log_var: float = -10.0, max_log_var: float = 5.0):
        # 将旧参数映射到新的修复版参数
        if init_log_var == 0.0 and min_log_var == -10.0 and max_log_var == 5.0:
            # 使用修复版默认值
            super().__init__(
                initial_pose_log_var=-1.0,
                initial_detection_log_var=0.0,
                initial_gate_log_var=-0.4,
                enable_weight_constraints=True,
                min_log_var=-2.0,
                max_log_var=2.0
            )
            print("⚠️ 使用修复版Kendall不确定性 - 旧接口已自动转换为平衡初始化")
        else:
            # 使用用户指定的参数
            super().__init__(
                initial_pose_log_var=init_log_var,
                initial_detection_log_var=init_log_var,
                initial_gate_log_var=init_log_var,
                enable_weight_constraints=True,
                min_log_var=min_log_var,
                max_log_var=max_log_var
            )


def create_fixed_kendall_uncertainty(config: Dict = None) -> FixedKendallUncertainty:
    """创建修复版Kendall不确定性模块"""
    default_config = {
        'initial_pose_log_var': -1.0,      # 保守平衡
        'initial_detection_log_var': 0.0,   # 基准
        'initial_gate_log_var': -0.4,      # 适中
        'enable_weight_constraints': True,
        'min_log_var': -2.0,
        'max_log_var': 2.0,
        'learning_rate_scale': 0.1
    }

    if config:
        default_config.update(config)

    return FixedKendallUncertainty(**default_config)


# 工厂函数 (更新为使用修复版)
def create_kendall_uncertainty(uncertainty_type: str = 'fixed',
                              num_tasks: int = 3,
                              **kwargs) -> FixedKendallUncertainty:
    """
    创建Kendall不确定性模块 (现在默认使用修复版)

    Args:
        uncertainty_type: 'fixed' (推荐) 或 'basic' (向后兼容)
        num_tasks: 任务数量 (固定为3)
        **kwargs: 其他参数

    Returns:
        uncertainty_module: 修复版Kendall不确定性模块
    """
    if uncertainty_type == 'basic':
        # 向后兼容模式
        return KendallUncertainty(num_tasks=num_tasks, **kwargs)
    else:
        # 推荐的修复版
        return create_fixed_kendall_uncertainty(kwargs)


# 模块测试
if __name__ == '__main__':
    print("🧪 Testing Fixed Kendall Uncertainty...")

    # 测试修复版Kendall不确定性
    print("\n=== 修复版Kendall不确定性测试 ===")
    kendall = create_fixed_kendall_uncertainty()

    # 模拟当前问题的损失值
    pose_loss = torch.tensor(0.005)
    detection_loss = torch.tensor(1.8)
    gate_loss = torch.tensor(0.00001)

    print(f"\n输入损失值:")
    print(f"  Pose Loss: {pose_loss.item():.6f}")
    print(f"  Detection Loss: {detection_loss.item():.6f}")
    print(f"  Gate Loss: {gate_loss.item():.8f}")

    # 计算多任务损失
    result = kendall.compute_multitask_loss(pose_loss, detection_loss, gate_loss)

    print(f"\n修复后的损失计算:")
    print(f"  Total Loss: {result['total_loss']:.6f}")
    print(f"  Weighted Losses:")
    print(f"    Pose: {result['weighted_pose_loss']:.6f}")
    print(f"    Detection: {result['weighted_detection_loss']:.6f}")
    print(f"    Gate: {result['weighted_gate_loss']:.6f}")

    # 显示权重状态
    kendall.print_status()

    # 测试权重平衡指标
    balance_metrics = kendall.get_weight_balance_metrics()
    print(f"\n权重平衡分析:")
    print(f"  平衡得分: {balance_metrics['balance_score']:.3f}")
    print(f"  最大比例: {balance_metrics['max_ratio']:.1f}:1")
    print(f"  权重分布: Pose={balance_metrics['pose_ratio']:.3f}, Detection={balance_metrics['detection_ratio']:.3f}, Gate={balance_metrics['gate_ratio']:.3f}")

    # 测试向后兼容性
    print(f"\n=== 向后兼容性测试 ===")
    legacy_kendall = KendallUncertainty()

    # 使用旧接口
    legacy_losses = {
        'pose': pose_loss,
        'detection': detection_loss,
        'gate': gate_loss
    }

    legacy_result = legacy_kendall(legacy_losses)
    print(f"旧接口总损失: {legacy_result['total_loss']:.6f}")

    # 对比测试：模拟原有问题配置
    print(f"\n=== 问题配置对比测试 ===")
    problem_kendall = FixedKendallUncertainty(
        initial_pose_log_var=-10.0,  # 原问题配置
        initial_detection_log_var=5.0,
        initial_gate_log_var=-10.0,
        enable_weight_constraints=False
    )

    problem_result = problem_kendall.compute_multitask_loss(pose_loss, detection_loss, gate_loss)
    problem_kendall.print_status()

    print(f"\n对比结果:")
    print(f"  修复版总损失: {result['total_loss']:.6f}")
    print(f"  问题版总损失: {problem_result['total_loss']:.6f}")
    print(f"  权重比例改善:")
    print(f"    修复版: {result['pose_weight']:.2f} : {result['detection_weight']:.4f} : {result['gate_weight']:.2f}")
    print(f"    问题版: {problem_result['pose_weight']:.0f} : {problem_result['detection_weight']:.6f} : {problem_result['gate_weight']:.0f}")

    print("\n✅ Fixed Kendall Uncertainty test completed!")
    print("\n🎯 关键改进:")
    print("   1. 权重比例从 3,269,017:1 降低到 2.7:1")
    print("   2. 检测任务重新获得合理权重")
    print("   3. 添加权重范围约束和监控")
    print("   4. 保持向后兼容性")