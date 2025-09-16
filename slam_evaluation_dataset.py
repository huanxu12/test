"""
SLAM Evaluation Dataset Extension
扩展现有MineSLAMDataset以支持完整的SLAM评估：ground truth加载、DARPA标注、轨迹分析
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import bisect
from pathlib import Path
import warnings
from scipy.spatial.transform import Rotation as R

# 导入现有数据集
from data.mineslam_dataset import MineSLAMDataset, RealDataContract


class SLAMGroundTruthLoader:
    """SLAM真值数据加载器"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.gt_trajectory_file = self.data_root / "ground_truth_trajectory.csv"
        self.gt_stats_file = self.data_root / "ground_truth_stats.json"
        self.artifacts_file = self.data_root / "sr_B_route2.bag.artifacts"
        self.fiducial_file = self.data_root / "fiducial_sr"

        # 加载真值数据
        self.ground_truth_poses = self._load_ground_truth_trajectory()
        self.ground_truth_stats = self._load_ground_truth_stats()
        self.artifact_annotations = self._load_artifact_annotations()
        self.fiducial_positions = self._load_fiducial_positions()

        print(f"📍 Ground truth loaded: {len(self.ground_truth_poses)} poses")

    def _load_ground_truth_trajectory(self) -> pd.DataFrame:
        """加载真值轨迹数据"""
        if not self.gt_trajectory_file.exists():
            warnings.warn(f"Ground truth trajectory file not found: {self.gt_trajectory_file}")
            return pd.DataFrame()

        try:
            # 读取CSV文件
            gt_df = pd.read_csv(self.gt_trajectory_file)

            # 验证必需的列
            required_cols = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            missing_cols = [col for col in required_cols if col not in gt_df.columns]

            if missing_cols:
                warnings.warn(f"Missing columns in ground truth: {missing_cols}")
                return pd.DataFrame()

            # 排序并去重
            gt_df = gt_df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

            print(f"✅ Loaded {len(gt_df)} ground truth poses")
            print(f"   Time range: {gt_df['timestamp'].min():.2f} - {gt_df['timestamp'].max():.2f}s")
            print(f"   Trajectory length: {self._compute_trajectory_length(gt_df):.2f}m")

            return gt_df

        except Exception as e:
            warnings.warn(f"Error loading ground truth trajectory: {e}")
            return pd.DataFrame()

    def _compute_trajectory_length(self, gt_df: pd.DataFrame) -> float:
        """计算轨迹总长度"""
        if len(gt_df) < 2:
            return 0.0

        positions = gt_df[['x', 'y', 'z']].values
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return float(np.sum(distances))

    def _load_ground_truth_stats(self) -> Dict[str, Any]:
        """加载真值统计信息"""
        if not self.gt_stats_file.exists():
            warnings.warn(f"Ground truth stats file not found: {self.gt_stats_file}")
            return {}

        try:
            with open(self.gt_stats_file, 'r') as f:
                stats = json.load(f)
            print(f"📊 Ground truth stats: {stats}")
            return stats
        except Exception as e:
            warnings.warn(f"Error loading ground truth stats: {e}")
            return {}

    def _load_artifact_annotations(self) -> List[Dict[str, Any]]:
        """加载DARPA物体标注"""
        if not self.artifacts_file.exists():
            warnings.warn(f"Artifacts file not found: {self.artifacts_file}")
            return []

        try:
            artifacts = []
            with open(self.artifacts_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 解析标注格式（具体格式需要根据实际文件确定）
                        parts = line.split(',')
                        if len(parts) >= 4:
                            artifact = {
                                'class_name': parts[0].strip(),
                                'x': float(parts[1]),
                                'y': float(parts[2]),
                                'z': float(parts[3]),
                                'confidence': float(parts[4]) if len(parts) > 4 else 1.0
                            }
                            artifacts.append(artifact)

            print(f"🏷️ Loaded {len(artifacts)} artifact annotations")
            return artifacts

        except Exception as e:
            warnings.warn(f"Error loading artifact annotations: {e}")
            return []

    def _load_fiducial_positions(self) -> List[Dict[str, Any]]:
        """加载基准标志位置"""
        if not self.fiducial_file.exists():
            warnings.warn(f"Fiducial file not found: {self.fiducial_file}")
            return []

        try:
            fiducials = []
            with open(self.fiducial_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 3:
                            fiducial = {
                                'id': len(fiducials),
                                'x': float(parts[0]),
                                'y': float(parts[1]),
                                'z': float(parts[2])
                            }
                            fiducials.append(fiducial)

            print(f"📍 Loaded {len(fiducials)} fiducial markers")
            return fiducials

        except Exception as e:
            warnings.warn(f"Error loading fiducial positions: {e}")
            return []

    def get_pose_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """获取指定时间戳的位姿"""
        if self.ground_truth_poses.empty:
            return None

        # 找到最近的时间戳
        timestamps = self.ground_truth_poses['timestamp'].values
        idx = np.argmin(np.abs(timestamps - timestamp))

        # 检查时间差是否在合理范围内
        time_diff = abs(timestamps[idx] - timestamp)
        if time_diff > 0.1:  # 100ms容差
            warnings.warn(f"Large time difference: {time_diff:.3f}s for timestamp {timestamp}")

        # 提取7维位姿 [x,y,z,qx,qy,qz,qw]
        pose_row = self.ground_truth_poses.iloc[idx]
        pose = np.array([
            pose_row['x'], pose_row['y'], pose_row['z'],
            pose_row['qx'], pose_row['qy'], pose_row['qz'], pose_row['qw']
        ])

        return pose

    def get_artifacts_in_view(self, pose: np.ndarray, max_distance: float = 10.0) -> List[Dict[str, Any]]:
        """获取视野内的标注物体"""
        if not self.artifact_annotations:
            return []

        pose_position = pose[:3]
        visible_artifacts = []

        for artifact in self.artifact_annotations:
            artifact_position = np.array([artifact['x'], artifact['y'], artifact['z']])
            distance = np.linalg.norm(pose_position - artifact_position)

            if distance <= max_distance:
                artifact_copy = artifact.copy()
                artifact_copy['distance'] = distance
                visible_artifacts.append(artifact_copy)

        return visible_artifacts


class DARPAClassMapping:
    """DARPA SubT挑战赛类别映射"""

    # DARPA 8类物体映射
    CLASS_NAMES = [
        'fiducial',      # 0 - 基准标志
        'extinguisher',  # 1 - 灭火器
        'phone',         # 2 - 电话
        'backpack',      # 3 - 背包
        'survivor',      # 4 - 幸存者
        'drill',         # 5 - 钻头
        'rope',          # 6 - 绳索（如果有）
        'other'          # 7 - 其他物体
    ]

    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

    @classmethod
    def get_class_id(cls, class_name: str) -> int:
        """获取类别ID"""
        # 处理各种可能的命名变体
        class_name = class_name.lower().strip()

        # 直接匹配
        if class_name in cls.CLASS_TO_ID:
            return cls.CLASS_TO_ID[class_name]

        # 模糊匹配
        if 'fiducial' in class_name or 'marker' in class_name:
            return 0
        elif 'extinguisher' in class_name or 'fire' in class_name:
            return 1
        elif 'phone' in class_name or 'cellphone' in class_name:
            return 2
        elif 'backpack' in class_name or 'bag' in class_name:
            return 3
        elif 'survivor' in class_name or 'person' in class_name or 'human' in class_name:
            return 4
        elif 'drill' in class_name or 'tool' in class_name:
            return 5
        elif 'rope' in class_name or 'cable' in class_name:
            return 6
        else:
            return 7  # other

    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        """获取类别名称"""
        return cls.ID_TO_CLASS.get(class_id, 'unknown')


class SLAMEvaluationDataset(MineSLAMDataset):
    """扩展的SLAM评估数据集"""

    def __init__(self, data_root: str, split: str = "train", sequence_length: int = 1,
                 load_ground_truth: bool = True, **kwargs):
        """
        初始化SLAM评估数据集

        Args:
            data_root: 数据根目录
            split: 数据分割 ("train", "val", "test")
            sequence_length: 序列长度
            load_ground_truth: 是否加载真值数据
        """
        super().__init__(data_root, split, sequence_length, **kwargs)

        self.load_ground_truth = load_ground_truth

        # 加载真值数据
        if self.load_ground_truth:
            self.gt_loader = SLAMGroundTruthLoader(data_root)
        else:
            self.gt_loader = None

        # DARPA类别映射
        self.class_mapping = DARPAClassMapping()

        print(f"🔍 SLAM Evaluation Dataset initialized:")
        print(f"   Split: {split}")
        print(f"   Samples: {len(self)}")
        print(f"   Ground truth: {'Enabled' if load_ground_truth else 'Disabled'}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取增强的评估样本"""
        # 获取基础样本
        sample = super().__getitem__(idx)

        # 添加SLAM评估相关信息
        if self.load_ground_truth and self.gt_loader:
            sample = self._add_ground_truth_info(sample, idx)

        return sample

    def _add_ground_truth_info(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """添加真值信息到样本"""
        # 获取时间戳
        timestamp = sample.get('timestamp', 0.0)

        # 获取真值位姿
        gt_pose = self.gt_loader.get_pose_at_timestamp(timestamp)
        if gt_pose is not None:
            sample['gt_pose'] = torch.from_numpy(gt_pose).float()
        else:
            # 创建单位位姿作为默认值
            sample['gt_pose'] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)

        # 获取视野内的标注物体
        if gt_pose is not None:
            visible_artifacts = self.gt_loader.get_artifacts_in_view(gt_pose)
            sample['gt_detections'] = self._process_artifacts_to_detections(visible_artifacts)
        else:
            sample['gt_detections'] = {
                'boxes': torch.zeros(0, 6),  # (N, 6) - 3D边界框
                'classes': torch.zeros(0, dtype=torch.long),  # (N,) - 类别
                'scores': torch.zeros(0)  # (N,) - 置信度
            }

        # 添加评估元数据
        sample['eval_metadata'] = {
            'sample_idx': idx,
            'timestamp': timestamp,
            'has_ground_truth': gt_pose is not None,
            'num_visible_artifacts': len(visible_artifacts) if gt_pose is not None else 0
        }

        return sample

    def _process_artifacts_to_detections(self, artifacts: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """将标注物体转换为检测格式"""
        if not artifacts:
            return {
                'boxes': torch.zeros(0, 6),
                'classes': torch.zeros(0, dtype=torch.long),
                'scores': torch.zeros(0)
            }

        boxes = []
        classes = []
        scores = []

        for artifact in artifacts:
            # 3D边界框 (中心位置 + 尺寸，简化处理)
            center = [artifact['x'], artifact['y'], artifact['z']]
            size = [1.0, 1.0, 1.0]  # 默认尺寸，实际应该根据物体类别调整
            box = center + size
            boxes.append(box)

            # 类别ID
            class_id = self.class_mapping.get_class_id(artifact['class_name'])
            classes.append(class_id)

            # 置信度
            scores.append(artifact.get('confidence', 1.0))

        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),  # (N, 6)
            'classes': torch.tensor(classes, dtype=torch.long),  # (N,)
            'scores': torch.tensor(scores, dtype=torch.float32)  # (N,)
        }

    def get_trajectory_sequence(self, start_idx: int, length: int) -> List[Dict[str, Any]]:
        """获取轨迹序列用于评估"""
        trajectory = []

        for i in range(start_idx, min(start_idx + length, len(self))):
            sample = self[i]
            trajectory.append(sample)

        return trajectory

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估数据集统计信息"""
        stats = {
            'total_samples': len(self),
            'ground_truth_available': self.load_ground_truth,
            'class_distribution': defaultdict(int),
            'trajectory_coverage': 0.0
        }

        if self.load_ground_truth and self.gt_loader:
            # 统计类别分布
            for artifact in self.gt_loader.artifact_annotations:
                class_name = artifact['class_name']
                class_id = self.class_mapping.get_class_id(class_name)
                stats['class_distribution'][self.class_mapping.get_class_name(class_id)] += 1

            # 轨迹覆盖度
            if not self.gt_loader.ground_truth_poses.empty:
                total_time = (self.gt_loader.ground_truth_poses['timestamp'].max() -
                            self.gt_loader.ground_truth_poses['timestamp'].min())
                stats['trajectory_coverage'] = total_time
                stats['trajectory_length'] = self.gt_loader._compute_trajectory_length(
                    self.gt_loader.ground_truth_poses
                )

        return dict(stats)


def create_slam_evaluation_splits(data_root: str, train_ratio: float = 0.7,
                                val_ratio: float = 0.15, test_ratio: float = 0.15,
                                save_splits: bool = True) -> Dict[str, List[int]]:
    """
    创建SLAM评估数据分割

    Args:
        data_root: 数据根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        save_splits: 是否保存分割信息

    Returns:
        包含各分割索引的字典
    """
    # 创建临时数据集以获取总样本数
    temp_dataset = SLAMEvaluationDataset(data_root, split="train", load_ground_truth=False)
    total_samples = len(temp_dataset)

    # 计算分割大小
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    # 创建随机分割
    np.random.seed(42)  # 固定种子以保证可重复性
    indices = np.random.permutation(total_samples)

    splits = {
        'train': indices[:train_size].tolist(),
        'val': indices[train_size:train_size + val_size].tolist(),
        'test': indices[train_size + val_size:].tolist()
    }

    print(f"📊 Created SLAM evaluation splits:")
    print(f"   Train: {len(splits['train'])} samples ({train_ratio:.1%})")
    print(f"   Val: {len(splits['val'])} samples ({val_ratio:.1%})")
    print(f"   Test: {len(splits['test'])} samples ({test_ratio:.1%})")

    # 保存分割信息
    if save_splits:
        splits_file = Path(data_root) / "slam_evaluation_splits.json"
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"💾 Splits saved to: {splits_file}")

    return splits


def load_slam_evaluation_splits(data_root: str) -> Optional[Dict[str, List[int]]]:
    """加载已保存的SLAM评估分割"""
    splits_file = Path(data_root) / "slam_evaluation_splits.json"

    if not splits_file.exists():
        return None

    try:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        print(f"📂 Loaded SLAM evaluation splits from: {splits_file}")
        return splits
    except Exception as e:
        warnings.warn(f"Error loading splits: {e}")
        return None


class SLAMDatasetFactory:
    """SLAM数据集工厂类"""

    @staticmethod
    def create_evaluation_dataset(data_root: str, split: str = "train",
                                **kwargs) -> SLAMEvaluationDataset:
        """创建SLAM评估数据集"""
        return SLAMEvaluationDataset(data_root, split=split, **kwargs)

    @staticmethod
    def create_training_dataset(data_root: str, **kwargs) -> SLAMEvaluationDataset:
        """创建训练数据集（含真值）"""
        return SLAMEvaluationDataset(data_root, split="train", load_ground_truth=True, **kwargs)

    @staticmethod
    def create_validation_dataset(data_root: str, **kwargs) -> SLAMEvaluationDataset:
        """创建验证数据集（含真值）"""
        return SLAMEvaluationDataset(data_root, split="val", load_ground_truth=True, **kwargs)

    @staticmethod
    def create_test_dataset(data_root: str, **kwargs) -> SLAMEvaluationDataset:
        """创建测试数据集（含真值）"""
        return SLAMEvaluationDataset(data_root, split="test", load_ground_truth=True, **kwargs)


if __name__ == "__main__":
    # 测试SLAM评估数据集
    print("🧪 SLAM Evaluation Dataset - Test Mode")

    data_root = "sr_B_route2_deep_learning"

    try:
        # 测试真值加载器
        gt_loader = SLAMGroundTruthLoader(data_root)

        # 测试数据集
        eval_dataset = SLAMEvaluationDataset(data_root, split="train", load_ground_truth=True)

        # 获取样本
        sample = eval_dataset[0]
        print(f"📦 Sample keys: {list(sample.keys())}")

        if 'gt_pose' in sample:
            print(f"🎯 Ground truth pose shape: {sample['gt_pose'].shape}")

        if 'gt_detections' in sample:
            gt_det = sample['gt_detections']
            print(f"🏷️ Ground truth detections:")
            print(f"   Boxes: {gt_det['boxes'].shape}")
            print(f"   Classes: {gt_det['classes'].shape}")

        # 获取统计信息
        stats = eval_dataset.get_evaluation_statistics()
        print(f"📊 Dataset statistics: {stats}")

        print("✅ SLAM evaluation dataset test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()