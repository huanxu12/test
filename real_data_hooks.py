"""
Random Generator Detection Hooks for MineSLAM
随机生成器检测钩子：防止在训练过程中使用合成/随机数据替代真实传感器数据
严格确保所有训练数据都来自真实的SubT矿井环境传感器采集
"""

import os
import sys
import torch
import numpy as np
import warnings
import threading
import time
import functools
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import inspect


class SyntheticDataDetector:
    """
    Synthetic Data Detector - 合成数据检测器
    全面监控训练过程，确保不使用任何合成/随机/模拟数据
    """

    def __init__(self, strict_mode: bool = True, log_violations: bool = True):
        """
        Args:
            strict_mode: 严格模式，检测到违规时抛出异常
            log_violations: 记录违规行为
        """
        self.strict_mode = strict_mode
        self.log_violations = log_violations

        # 违规记录
        self.violations = []
        self.violation_counts = {}

        # 禁止的关键词
        self.forbidden_keywords = [
            # 生成类
            'generated', 'generate', 'synthetic', 'synth', 'artificial',
            'simulated', 'simulate', 'sim', 'mock', 'fake', 'dummy',
            'random', 'rand', 'noise', 'test_data', 'placeholder',

            # 数学分布
            'randn', 'uniform', 'normal', 'gaussian', 'exponential',
            'bernoulli', 'poisson', 'beta', 'gamma',

            # 模拟工具
            'blender', 'unity', 'unreal', 'gazebo', 'carla', 'airsim',
            'opencv_generate', 'procedural', 'parametric'
        ]

        # 真实数据标识符
        self.real_data_identifiers = [
            'subt', 'darpa', 'clearpath', 'husky', 'multisense',
            'ouster', 'boson', 'flir', 'microstrain', 'gx5',
            'real', 'sensor', 'camera', 'lidar', 'imu', 'thermal'
        ]

        # 线程锁
        self._lock = threading.Lock()

        print(f"SyntheticDataDetector initialized (strict_mode={strict_mode})")

    def _log_violation(self, violation_type: str, details: str, severity: str = "WARNING"):
        """记录违规行为"""
        timestamp = time.time()
        violation = {
            'timestamp': timestamp,
            'type': violation_type,
            'details': details,
            'severity': severity,
            'stack_trace': self._get_call_stack()
        }

        with self._lock:
            self.violations.append(violation)
            self.violation_counts[violation_type] = self.violation_counts.get(violation_type, 0) + 1

        if self.log_violations:
            print(f"🚨 {severity}: {violation_type} - {details}")

        if self.strict_mode and severity == "ERROR":
            raise ValueError(f"FORBIDDEN SYNTHETIC DATA DETECTED: {violation_type} - {details}")

    def _get_call_stack(self) -> List[str]:
        """获取调用堆栈"""
        stack = []
        for frame_info in inspect.stack()[2:8]:  # 跳过detector内部调用
            stack.append(f"{frame_info.filename}:{frame_info.lineno} in {frame_info.function}")
        return stack

    def check_variable_names(self, locals_dict: Dict[str, Any], context: str = ""):
        """检查变量名是否包含禁止关键词"""
        for var_name, var_value in locals_dict.items():
            var_name_lower = var_name.lower()

            # 检查是否包含禁止关键词
            for keyword in self.forbidden_keywords:
                if keyword in var_name_lower:
                    self._log_violation(
                        "FORBIDDEN_VARIABLE_NAME",
                        f"Variable '{var_name}' contains forbidden keyword '{keyword}' in {context}",
                        "ERROR"
                    )

    def check_tensor_properties(self, tensor: torch.Tensor, name: str = "tensor",
                              context: str = "") -> bool:
        """
        检查张量属性是否符合真实传感器数据特征

        Returns:
            True if tensor appears to be real sensor data
        """
        if not isinstance(tensor, torch.Tensor):
            return True

        try:
            # 1. 检查张量维度合理性
            if tensor.dim() > 5:  # 传感器数据通常不超过5维
                self._log_violation(
                    "SUSPICIOUS_TENSOR_DIMS",
                    f"Tensor '{name}' has unusual {tensor.dim()} dimensions in {context}",
                    "WARNING"
                )

            # 2. 检查数据类型
            if tensor.dtype in [torch.complex64, torch.complex128]:
                self._log_violation(
                    "SUSPICIOUS_TENSOR_TYPE",
                    f"Tensor '{name}' has complex dtype {tensor.dtype} in {context}",
                    "WARNING"
                )

            # 3. 检查数值范围（针对不同传感器）
            if 'rgb' in name.lower() or 'color' in name.lower():
                # RGB图像应该在[0,1]或[0,255]范围
                min_val, max_val = tensor.min().item(), tensor.max().item()
                if min_val < -0.1 or max_val > 255.1:
                    self._log_violation(
                        "SUSPICIOUS_RGB_RANGE",
                        f"RGB tensor '{name}' has suspicious range [{min_val:.3f}, {max_val:.3f}] in {context}",
                        "WARNING"
                    )

            elif 'depth' in name.lower():
                # 深度图像应该在合理距离范围
                min_val, max_val = tensor.min().item(), tensor.max().item()
                if min_val < 0 or max_val > 100:  # 100米最大深度
                    self._log_violation(
                        "SUSPICIOUS_DEPTH_RANGE",
                        f"Depth tensor '{name}' has suspicious range [{min_val:.3f}, {max_val:.3f}] in {context}",
                        "WARNING"
                    )

            elif 'thermal' in name.lower() or 'temperature' in name.lower():
                # 热红外应该在合理温度范围（Kelvin）
                min_val, max_val = tensor.min().item(), tensor.max().item()
                if min_val < 200 or max_val > 500:  # 200-500K合理范围
                    self._log_violation(
                        "SUSPICIOUS_THERMAL_RANGE",
                        f"Thermal tensor '{name}' has suspicious range [{min_val:.3f}, {max_val:.3f}] in {context}",
                        "WARNING"
                    )

            # 4. 检查完美数学分布（高度可疑）
            if tensor.numel() > 1000:
                flat_tensor = tensor.flatten().float()
                mean_val = torch.mean(flat_tensor).item()
                std_val = torch.std(flat_tensor).item()

                # 检查是否过于接近标准正态分布
                if abs(mean_val) < 0.001 and abs(std_val - 1.0) < 0.001:
                    self._log_violation(
                        "PERFECT_NORMAL_DISTRIBUTION",
                        f"Tensor '{name}' has suspiciously perfect normal distribution (μ={mean_val:.6f}, σ={std_val:.6f}) in {context}",
                        "ERROR"
                    )

                # 检查是否所有值相同（常数张量）
                if torch.all(tensor == tensor.flatten()[0]):
                    self._log_violation(
                        "CONSTANT_TENSOR",
                        f"Tensor '{name}' is constant (all values = {tensor.flatten()[0].item():.6f}) in {context}",
                        "ERROR"
                    )

            # 5. 检查时间戳合理性
            if 'timestamp' in name.lower() or 'time' in name.lower():
                flat_tensor = tensor.flatten()
                for i, val in enumerate(flat_tensor):
                    ts = val.item()
                    # Unix时间戳应该在2017-2026年范围内
                    if ts < 1.48e9 or ts > 1.77e9:
                        self._log_violation(
                            "INVALID_TIMESTAMP",
                            f"Invalid timestamp {ts} in tensor '{name}' at index {i} in {context}",
                            "ERROR"
                        )

            return True

        except Exception as e:
            self._log_violation(
                "TENSOR_CHECK_ERROR",
                f"Error checking tensor '{name}': {str(e)} in {context}",
                "WARNING"
            )
            return False

    def check_batch_data(self, batch: Dict[str, Any], batch_idx: int = 0) -> bool:
        """
        检查批次数据的真实性

        Returns:
            True if batch appears to contain real sensor data
        """
        context = f"batch_{batch_idx}"

        # 1. 检查键名是否包含禁止词汇
        for key in batch.keys():
            key_lower = str(key).lower()
            for keyword in self.forbidden_keywords:
                if keyword in key_lower:
                    self._log_violation(
                        "FORBIDDEN_BATCH_KEY",
                        f"Batch key '{key}' contains forbidden keyword '{keyword}' in {context}",
                        "ERROR"
                    )

        # 2. 检查每个张量
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                self.check_tensor_properties(value, key, context)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    for i, tensor in enumerate(value):
                        self.check_tensor_properties(tensor, f"{key}[{i}]", context)

        # 3. 检查是否包含必要的传感器数据
        expected_keys = ['rgb', 'depth', 'thermal', 'lidar', 'imu', 'timestamp']
        missing_keys = [key for key in expected_keys if key not in batch]

        if len(missing_keys) > len(expected_keys) // 2:  # 缺失超过一半
            self._log_violation(
                "INCOMPLETE_SENSOR_DATA",
                f"Batch missing critical sensor data: {missing_keys} in {context}",
                "WARNING"
            )

        return True

    def check_function_call(self, func_name: str, args: tuple, kwargs: dict,
                          context: str = "") -> bool:
        """检查函数调用是否可疑"""
        func_name_lower = func_name.lower()

        # 检查函数名是否包含禁止关键词
        for keyword in self.forbidden_keywords:
            if keyword in func_name_lower:
                self._log_violation(
                    "FORBIDDEN_FUNCTION_CALL",
                    f"Function '{func_name}' contains forbidden keyword '{keyword}' in {context}",
                    "ERROR"
                )
                return False

        # 检查参数
        all_args = list(args) + list(kwargs.values())
        for i, arg in enumerate(all_args):
            if isinstance(arg, torch.Tensor):
                self.check_tensor_properties(arg, f"{func_name}_arg_{i}", context)

        return True

    def get_violation_summary(self) -> Dict[str, Any]:
        """获取违规行为摘要"""
        with self._lock:
            return {
                'total_violations': len(self.violations),
                'violation_counts': self.violation_counts.copy(),
                'recent_violations': self.violations[-10:] if self.violations else [],
                'is_clean': len(self.violations) == 0
            }

    def reset_violations(self):
        """重置违规记录"""
        with self._lock:
            self.violations.clear()
            self.violation_counts.clear()


class RealDataOnlyDecorator:
    """
    Real Data Only Decorator - 真实数据装饰器
    确保被装饰的函数只接收真实传感器数据
    """

    def __init__(self, detector: SyntheticDataDetector):
        self.detector = detector

    def __call__(self, func: Callable) -> Callable:
        """装饰器实现"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数名和上下文
            func_name = func.__name__
            context = f"function_{func_name}"

            # 检查函数调用
            self.detector.check_function_call(func_name, args, kwargs, context)

            # 检查局部变量（函数参数）
            frame = inspect.currentframe()
            if frame and frame.f_back:
                local_vars = frame.f_back.f_locals
                self.detector.check_variable_names(local_vars, context)

            # 调用原函数
            try:
                result = func(*args, **kwargs)

                # 检查返回值
                if isinstance(result, torch.Tensor):
                    self.detector.check_tensor_properties(result, f"{func_name}_output", context)
                elif isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            self.detector.check_tensor_properties(value, f"{func_name}_output_{key}", context)

                return result

            except Exception as e:
                self.detector._log_violation(
                    "FUNCTION_EXECUTION_ERROR",
                    f"Error in function '{func_name}': {str(e)}",
                    "ERROR"
                )
                raise

        return wrapper


class TrainingHookManager:
    """
    Training Hook Manager - 训练钩子管理器
    在训练过程的关键点插入检测钩子
    """

    def __init__(self, strict_mode: bool = True):
        self.detector = SyntheticDataDetector(strict_mode=strict_mode)
        self.real_data_only = RealDataOnlyDecorator(self.detector)
        self.hooks_installed = False

    def install_hooks(self):
        """安装训练钩子"""
        if self.hooks_installed:
            return

        # Hook torch.randn and related functions
        self._hook_torch_random_functions()

        # Hook numpy random functions
        self._hook_numpy_random_functions()

        self.hooks_installed = True
        print("✅ Real data validation hooks installed")

    def _hook_torch_random_functions(self):
        """钩子PyTorch随机函数"""
        original_randn = torch.randn
        original_rand = torch.rand
        original_randint = torch.randint

        def hooked_randn(*args, **kwargs):
            self.detector._log_violation(
                "TORCH_RANDOM_CALL",
                f"torch.randn called with args={args}, kwargs={kwargs}",
                "ERROR"
            )
            return original_randn(*args, **kwargs)

        def hooked_rand(*args, **kwargs):
            self.detector._log_violation(
                "TORCH_RANDOM_CALL",
                f"torch.rand called with args={args}, kwargs={kwargs}",
                "ERROR"
            )
            return original_rand(*args, **kwargs)

        def hooked_randint(*args, **kwargs):
            # Allow small randint calls for legitimate use cases
            if args and args[0] <= 10:  # Allow small random integers
                return original_randint(*args, **kwargs)

            self.detector._log_violation(
                "TORCH_RANDOM_CALL",
                f"torch.randint called with args={args}, kwargs={kwargs}",
                "WARNING"
            )
            return original_randint(*args, **kwargs)

        # 只在严格模式下替换
        if self.detector.strict_mode:
            torch.randn = hooked_randn
            torch.rand = hooked_rand
            torch.randint = hooked_randint

    def _hook_numpy_random_functions(self):
        """钩子NumPy随机函数"""
        try:
            import numpy as np

            original_randn = np.random.randn
            original_rand = np.random.rand

            def hooked_np_randn(*args, **kwargs):
                self.detector._log_violation(
                    "NUMPY_RANDOM_CALL",
                    f"np.random.randn called with args={args}, kwargs={kwargs}",
                    "ERROR"
                )
                return original_randn(*args, **kwargs)

            def hooked_np_rand(*args, **kwargs):
                self.detector._log_violation(
                    "NUMPY_RANDOM_CALL",
                    f"np.random.rand called with args={args}, kwargs={kwargs}",
                    "ERROR"
                )
                return original_rand(*args, **kwargs)

            # 只在严格模式下替换
            if self.detector.strict_mode:
                np.random.randn = hooked_np_randn
                np.random.rand = hooked_np_rand

        except ImportError:
            pass  # NumPy not available

    def validate_training_batch(self, batch: Dict[str, Any], batch_idx: int = 0) -> bool:
        """验证训练批次"""
        return self.detector.check_batch_data(batch, batch_idx)

    def validate_model_output(self, outputs: Dict[str, Any], context: str = "model_output") -> bool:
        """验证模型输出"""
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                self.detector.check_tensor_properties(value, key, context)
        return True

    def get_violation_report(self) -> Dict[str, Any]:
        """获取违规报告"""
        return self.detector.get_violation_summary()

    def reset_violations(self):
        """重置违规记录"""
        self.detector.reset_violations()


# 全局钩子管理器实例
global_hook_manager = TrainingHookManager(strict_mode=True)


def install_real_data_hooks():
    """安装真实数据验证钩子"""
    global_hook_manager.install_hooks()
    return global_hook_manager


def validate_batch(batch: Dict[str, Any], batch_idx: int = 0) -> bool:
    """快速批次验证接口"""
    return global_hook_manager.validate_training_batch(batch, batch_idx)


def validate_outputs(outputs: Dict[str, Any], context: str = "output") -> bool:
    """快速输出验证接口"""
    return global_hook_manager.validate_model_output(outputs, context)


def real_data_only(func: Callable) -> Callable:
    """装饰器：确保函数只处理真实数据"""
    return global_hook_manager.real_data_only(func)


def get_violation_report() -> Dict[str, Any]:
    """获取全局违规报告"""
    return global_hook_manager.get_violation_report()


if __name__ == '__main__':
    print("🧪 Testing Random Generator Detection Hooks...")

    # 安装钩子
    hook_manager = install_real_data_hooks()

    # 测试批次验证
    print("\n1. Testing batch validation...")
    real_batch = {
        'rgb': torch.randn(2, 3, 224, 224) * 0.1 + 0.5,
        'depth': torch.rand(2, 1, 224, 224) * 10,
        'timestamp': torch.tensor([1542126757.0, 1542126758.0])
    }

    try:
        validate_batch(real_batch, 0)
        print("✅ Real batch validation passed")
    except Exception as e:
        print(f"❌ Real batch validation failed: {e}")

    # 测试可疑数据检测
    print("\n2. Testing suspicious data detection...")
    suspicious_batch = {
        'synthetic_rgb': torch.ones(2, 3, 224, 224),  # 可疑键名
        'depth': torch.zeros(2, 1, 224, 224),  # 全零张量
        'timestamp': torch.tensor([1e12, 1e12])  # 无效时间戳
    }

    try:
        validate_batch(suspicious_batch, 1)
        print("❌ Suspicious batch should have been detected")
    except Exception as e:
        print(f"✅ Suspicious batch correctly detected: {e}")

    # 测试装饰器
    print("\n3. Testing real_data_only decorator...")

    @real_data_only
    def process_sensor_data(rgb, depth):
        return rgb + depth.expand_as(rgb)

    try:
        result = process_sensor_data(
            torch.randn(1, 3, 64, 64) * 0.1 + 0.5,
            torch.rand(1, 1, 64, 64) * 10
        )
        print("✅ Decorated function processed real data")
    except Exception as e:
        print(f"❌ Decorated function failed: {e}")

    # 获取违规报告
    print("\n4. Violation report:")
    report = get_violation_report()
    print(f"Total violations: {report['total_violations']}")
    print(f"Violation types: {list(report['violation_counts'].keys())}")
    print(f"Data is clean: {report['is_clean']}")

    print("\n✅ Random Generator Detection Hooks test completed!")