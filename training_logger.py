"""
Advanced Training Logger for MineSLAM
高级训练日志记录系统：记录真实数据的ATE/RPE/mAP/FPS、门控权重、GPU显存
"""

import os
import sys
import json
import time
import csv
import threading
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import logging

import torch
import numpy as np
from collections import defaultdict, deque


class MetricsBuffer:
    """
    Metrics Buffer with Statistical Analysis
    带统计分析的指标缓冲区
    """

    def __init__(self, maxlen: int = 1000):
        """
        Args:
            maxlen: 缓冲区最大长度
        """
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)

    def add(self, value: float, timestamp: Optional[float] = None):
        """添加数据点"""
        if timestamp is None:
            timestamp = time.time()

        self.data.append(value)
        self.timestamps.append(timestamp)

    def get_statistics(self, window: Optional[int] = None) -> Dict[str, float]:
        """获取统计信息"""
        if not self.data:
            return {}

        # 选择窗口数据
        if window is not None and window < len(self.data):
            data = list(self.data)[-window:]
        else:
            data = list(self.data)

        if not data:
            return {}

        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'count': len(data),
            'latest': data[-1] if data else 0.0
        }

    def get_recent_trend(self, window: int = 10) -> Optional[float]:
        """获取最近趋势（正数表示上升，负数表示下降）"""
        if len(self.data) < window:
            return None

        recent_data = list(self.data)[-window:]
        x = np.arange(len(recent_data))
        y = np.array(recent_data)

        # 简单线性回归求斜率
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)

        return None


class GPUMonitor:
    """
    GPU Memory and Utilization Monitor
    GPU内存和利用率监控器
    """

    def __init__(self):
        self.available = torch.cuda.is_available()
        if self.available:
            self.device_count = torch.cuda.device_count()
            self.device_names = [torch.cuda.get_device_name(i)
                               for i in range(self.device_count)]
        else:
            self.device_count = 0
            self.device_names = []

    def get_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """获取GPU内存信息（MB）"""
        if not self.available or device_id >= self.device_count:
            return {'allocated': 0.0, 'cached': 0.0, 'total': 0.0}

        try:
            allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
            cached = torch.cuda.memory_reserved(device_id) / 1024 / 1024
            total = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024

            return {
                'allocated': allocated,
                'cached': cached,
                'total': total,
                'free': total - cached,
                'utilization': allocated / total * 100 if total > 0 else 0.0
            }
        except:
            return {'allocated': 0.0, 'cached': 0.0, 'total': 0.0}

    def get_all_devices_info(self) -> Dict[str, Dict[str, float]]:
        """获取所有GPU设备信息"""
        info = {}
        for i in range(self.device_count):
            info[f'gpu_{i}'] = self.get_memory_info(i)
            info[f'gpu_{i}']['name'] = self.device_names[i]

        return info


class TrainingLogger:
    """
    Comprehensive Training Logger
    综合训练日志记录器
    """

    def __init__(self, log_dir: str, experiment_name: str = None,
                 save_interval: int = 100, buffer_size: int = 1000):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            save_interval: 保存间隔（步数）
            buffer_size: 缓冲区大小
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 实验名称和时间戳
        if experiment_name is None:
            experiment_name = f"mineslam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        self.save_interval = save_interval
        self.step_count = 0

        # 日志文件路径
        self.log_files = {
            'training': self.log_dir / f'{experiment_name}_training.log',
            'metrics': self.log_dir / f'{experiment_name}_metrics.json',
            'csv': self.log_dir / f'{experiment_name}_metrics.csv',
            'weights': self.log_dir / f'{experiment_name}_weights.json',
            'gpu': self.log_dir / f'{experiment_name}_gpu.json'
        }

        # 设置Python日志
        self.logger = self._setup_logger()

        # 指标缓冲区
        self.metrics_buffers = {
            'train_loss': MetricsBuffer(buffer_size),
            'val_loss': MetricsBuffer(buffer_size),
            'ate': MetricsBuffer(buffer_size),
            'rpe': MetricsBuffer(buffer_size),
            'map': MetricsBuffer(buffer_size),
            'fps': MetricsBuffer(buffer_size),
            'gpu_memory': MetricsBuffer(buffer_size),
            'learning_rate': MetricsBuffer(buffer_size)
        }

        # GPU监控器
        self.gpu_monitor = GPUMonitor()

        # 历史记录
        self.training_history = []
        self.kendall_weights_history = []
        self.gpu_history = []

        # 最佳指标记录
        self.best_metrics = {
            'best_ate': float('inf'),
            'best_rpe': float('inf'),
            'best_map': 0.0,
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # 初始化CSV文件
        self._init_csv_file()

        self.logger.info(f"TrainingLogger initialized: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"GPU available: {self.gpu_monitor.available}")

    def _setup_logger(self) -> logging.Logger:
        """设置Python日志记录器"""
        logger = logging.getLogger(f'MineSLAM_{self.experiment_name}')
        logger.setLevel(logging.INFO)

        # 避免重复添加handler
        if logger.handlers:
            logger.handlers.clear()

        # 文件handler
        file_handler = logging.FileHandler(self.log_files['training'])
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def _init_csv_file(self):
        """初始化CSV文件"""
        csv_path = self.log_files['csv']
        if not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'epoch', 'step', 'phase',
                    'total_loss', 'pose_loss', 'detection_loss', 'gate_loss',
                    'ate', 'rpe', 'map_score', 'fps',
                    'learning_rate', 'gpu_memory_mb', 'gpu_utilization',
                    'kendall_pose_weight', 'kendall_det_weight', 'kendall_gate_weight'
                ])

    def log_training_step(self, epoch: int, step: int, losses: Dict[str, float],
                         kendall_weights: Optional[Dict[str, float]] = None,
                         learning_rate: float = 0.0, fps: float = 0.0):
        """记录训练步骤"""
        timestamp = time.time()
        self.step_count += 1

        # 更新缓冲区
        if 'total_loss' in losses:
            self.metrics_buffers['train_loss'].add(losses['total_loss'], timestamp)
        self.metrics_buffers['learning_rate'].add(learning_rate, timestamp)
        self.metrics_buffers['fps'].add(fps, timestamp)

        # GPU信息
        gpu_info = self.gpu_monitor.get_memory_info()
        gpu_memory = gpu_info.get('allocated', 0.0)
        gpu_utilization = gpu_info.get('utilization', 0.0)
        self.metrics_buffers['gpu_memory'].add(gpu_memory, timestamp)

        # 记录到历史
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'step': step,
            'phase': 'train',
            'losses': losses,
            'fps': fps,
            'learning_rate': learning_rate,
            'gpu_memory_mb': gpu_memory,
            'gpu_utilization': gpu_utilization
        }

        if kendall_weights:
            log_entry['kendall_weights'] = kendall_weights

        self.training_history.append(log_entry)

        # 写入CSV
        self._write_csv_row(log_entry)

        # 日志信息
        if step % 50 == 0:
            self.logger.info(
                f"Epoch {epoch}, Step {step}: "
                f"Loss={losses.get('total_loss', 0):.6f}, "
                f"FPS={fps:.1f}, GPU={gpu_memory:.1f}MB"
            )

        # 定期保存
        if self.step_count % self.save_interval == 0:
            self._save_metrics()

    def log_validation_epoch(self, epoch: int, val_loss: float, ate: float,
                           rpe: float, map_score: float):
        """记录验证epoch"""
        timestamp = time.time()

        # 更新缓冲区
        self.metrics_buffers['val_loss'].add(val_loss, timestamp)
        self.metrics_buffers['ate'].add(ate, timestamp)
        self.metrics_buffers['rpe'].add(rpe, timestamp)
        self.metrics_buffers['map'].add(map_score, timestamp)

        # 检查最佳指标
        is_best = False
        if ate < self.best_metrics['best_ate']:
            self.best_metrics['best_ate'] = ate
            self.best_metrics['best_epoch'] = epoch
            is_best = True

        if rpe < self.best_metrics['best_rpe']:
            self.best_metrics['best_rpe'] = rpe

        if map_score > self.best_metrics['best_map']:
            self.best_metrics['best_map'] = map_score

        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss

        # 记录到历史
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'step': self.step_count,
            'phase': 'val',
            'val_loss': val_loss,
            'ate': ate,
            'rpe': rpe,
            'map_score': map_score,
            'is_best': is_best
        }

        self.training_history.append(log_entry)

        # 写入CSV
        self._write_csv_row(log_entry)

        # 日志信息
        status = "🎯 NEW BEST" if is_best else ""
        self.logger.info(
            f"Epoch {epoch} Validation: "
            f"Loss={val_loss:.6f}, ATE={ate:.3f}m, RPE={rpe:.3f}, mAP={map_score:.3f} {status}"
        )

        # 记录目标达成
        if ate <= 1.5 and map_score >= 0.6:
            self.logger.info(f"🎉 TARGET REACHED! ATE={ate:.3f}≤1.5m, mAP={map_score:.3f}≥60%")

    def log_kendall_weights(self, epoch: int, weights: Dict[str, float]):
        """记录Kendall权重"""
        timestamp = time.time()

        weight_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'step': self.step_count,
            'weights': weights
        }

        self.kendall_weights_history.append(weight_entry)

        # 每10个epoch详细记录权重变化
        if epoch % 10 == 0:
            self.logger.info(f"Kendall Weights (Epoch {epoch}):")
            for key, value in weights.items():
                if 'weight' in key:
                    self.logger.info(f"  {key}: {value:.6f}")

    def log_gpu_stats(self):
        """记录GPU统计"""
        if not self.gpu_monitor.available:
            return

        timestamp = time.time()
        gpu_info = self.gpu_monitor.get_all_devices_info()

        self.gpu_history.append({
            'timestamp': timestamp,
            'gpu_info': gpu_info
        })

        # 只保留最近1000条记录
        if len(self.gpu_history) > 1000:
            self.gpu_history = self.gpu_history[-1000:]

    def _write_csv_row(self, log_entry: Dict):
        """写入CSV行"""
        csv_path = self.log_files['csv']

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 准备数据行
            row = [
                log_entry.get('timestamp', time.time()),
                log_entry.get('epoch', 0),
                log_entry.get('step', 0),
                log_entry.get('phase', 'unknown'),
                log_entry.get('losses', {}).get('total_loss', log_entry.get('val_loss', 0)),
                log_entry.get('losses', {}).get('pose_loss', 0),
                log_entry.get('losses', {}).get('detection_loss', 0),
                log_entry.get('losses', {}).get('gate_loss', 0),
                log_entry.get('ate', 0),
                log_entry.get('rpe', 0),
                log_entry.get('map_score', 0),
                log_entry.get('fps', 0),
                log_entry.get('learning_rate', 0),
                log_entry.get('gpu_memory_mb', 0),
                log_entry.get('gpu_utilization', 0),
                log_entry.get('kendall_weights', {}).get('pose_weight', 0),
                log_entry.get('kendall_weights', {}).get('detection_weight', 0),
                log_entry.get('kendall_weights', {}).get('gate_weight', 0)
            ]

            writer.writerow(row)

    def _save_metrics(self):
        """保存指标到JSON文件"""
        metrics_data = {
            'experiment_name': self.experiment_name,
            'timestamp': time.time(),
            'step_count': self.step_count,
            'best_metrics': self.best_metrics,
            'recent_statistics': {
                name: buffer.get_statistics(window=100)
                for name, buffer in self.metrics_buffers.items()
            },
            'trends': {
                name: buffer.get_recent_trend(window=20)
                for name, buffer in self.metrics_buffers.items()
            }
        }

        # 保存主要指标
        with open(self.log_files['metrics'], 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        # 保存Kendall权重历史
        with open(self.log_files['weights'], 'w', encoding='utf-8') as f:
            json.dump(self.kendall_weights_history, f, indent=2, ensure_ascii=False)

        # 保存GPU历史
        if self.gpu_history:
            with open(self.log_files['gpu'], 'w', encoding='utf-8') as f:
                json.dump(self.gpu_history[-100:], f, indent=2, ensure_ascii=False)

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        summary = {
            'experiment_name': self.experiment_name,
            'total_steps': self.step_count,
            'best_metrics': self.best_metrics,
            'current_statistics': {}
        }

        # 当前统计
        for name, buffer in self.metrics_buffers.items():
            stats = buffer.get_statistics()
            if stats:
                summary['current_statistics'][name] = stats

        return summary

    def save_final_report(self):
        """保存最终报告"""
        self._save_metrics()

        # 生成最终报告
        report = {
            'experiment_info': {
                'name': self.experiment_name,
                'start_time': self.training_history[0]['timestamp'] if self.training_history else time.time(),
                'end_time': time.time(),
                'total_steps': self.step_count
            },
            'final_metrics': self.best_metrics,
            'training_summary': self.get_training_summary(),
            'convergence_analysis': self._analyze_convergence()
        }

        report_path = self.log_dir / f'{self.experiment_name}_final_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✅ Final report saved: {report_path}")
        self.logger.info(f"📊 Training completed with {self.step_count} steps")
        self.logger.info(f"🏆 Best ATE: {self.best_metrics['best_ate']:.3f}m")
        self.logger.info(f"🏆 Best mAP: {self.best_metrics['best_map']:.3f}")

        return report_path

    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析收敛性"""
        analysis = {}

        for name, buffer in self.metrics_buffers.items():
            if len(buffer.data) < 10:
                continue

            stats = buffer.get_statistics()
            trend = buffer.get_recent_trend(window=min(50, len(buffer.data)))

            analysis[name] = {
                'converged': abs(trend) < 1e-5 if trend is not None else False,
                'trend_slope': trend,
                'stability': stats['std'] / max(abs(stats['mean']), 1e-8),
                'final_value': stats['latest']
            }

        return analysis


def create_training_logger(log_dir: str, experiment_name: str = None,
                          **kwargs) -> TrainingLogger:
    """
    Factory function to create training logger

    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        **kwargs: 其他参数

    Returns:
        配置好的训练日志记录器
    """
    return TrainingLogger(log_dir, experiment_name, **kwargs)


if __name__ == '__main__':
    print("🧪 Testing Training Logger...")

    # 创建测试日志记录器
    logger = create_training_logger('test_logs', 'test_experiment')

    # 模拟训练过程
    for epoch in range(5):
        for step in range(10):
            # 模拟训练步骤
            losses = {
                'total_loss': 0.5 + 0.1 * np.random.randn(),
                'pose_loss': 0.1 + 0.02 * np.random.randn(),
                'detection_loss': 0.3 + 0.05 * np.random.randn(),
                'gate_loss': 0.1 + 0.01 * np.random.randn()
            }

            kendall_weights = {
                'pose_weight': 0.5 + 0.1 * np.random.randn(),
                'detection_weight': 1.0 + 0.2 * np.random.randn(),
                'gate_weight': 0.3 + 0.05 * np.random.randn()
            }

            logger.log_training_step(
                epoch, step, losses, kendall_weights,
                learning_rate=1e-4, fps=15.0
            )

        # 模拟验证
        val_loss = 0.4 + 0.05 * np.random.randn()
        ate = 1.0 + 0.2 * np.random.randn()
        rpe = 0.8 + 0.1 * np.random.randn()
        map_score = 0.7 + 0.05 * np.random.randn()

        logger.log_validation_epoch(epoch, val_loss, ate, rpe, map_score)

    # 保存最终报告
    report_path = logger.save_final_report()
    print(f"✅ Test completed! Report saved to: {report_path}")