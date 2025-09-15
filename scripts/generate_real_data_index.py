#!/usr/bin/env python3
"""
Real Data Index Generator for MineSLAM
从实际传感器数据生成训练/验证/测试索引文件
严格基于真实文件时间戳进行匹配
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict
import bisect


class RealDataIndexGenerator:
    """
    真实数据索引生成器
    基于实际传感器文件时间戳生成数据索引
    """
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.time_threshold_ms = 50  # 多模态时间对齐阈值（毫秒）
        
        # 验证数据集根目录存在
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        
        # 数据路径
        self.thermal_dir = self.dataset_root / "images" / "boson_thermal"
        self.rgb_left_dir = self.dataset_root / "images" / "multisense_left_color"
        self.rgb_right_dir = self.dataset_root / "images" / "multisense_right"
        self.depth_dir = self.dataset_root / "images" / "multisense_depth"
        self.ouster_dir = self.dataset_root / "pointclouds" / "ouster_lidar"
        self.multisense_lidar_dir = self.dataset_root / "pointclouds" / "multisense_lidar"
        self.imu_dir = self.dataset_root / "imu"
        self.gt_trajectory_file = self.dataset_root / "ground_truth_trajectory.csv"
        self.calibration_file = self.dataset_root / "calibration" / "calibration_data.json"
        
        print(f"Initializing real data index generator for: {dataset_root}")
        self._validate_data_directories()
    
    def _validate_data_directories(self):
        """验证所有必要的数据目录存在"""
        required_dirs = [
            self.thermal_dir,
            self.rgb_left_dir, 
            self.ouster_dir,
            self.imu_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required data directory not found: {dir_path}")
        
        # 验证关键文件存在
        if not self.gt_trajectory_file.exists():
            raise FileNotFoundError(f"Ground truth trajectory not found: {self.gt_trajectory_file}")
        
        print("✓ All required data directories validated")
    
    def _extract_timestamp_from_filename(self, filename: str) -> float:
        """从文件名提取时间戳"""
        try:
            # 文件名格式: timestamp.extension
            return float(filename.split('.')[0])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Cannot extract timestamp from filename: {filename}")
    
    def _scan_sensor_files(self) -> Dict[str, List[Tuple[float, str]]]:
        """扫描所有传感器文件并提取时间戳"""
        print("Scanning real sensor files...")
        
        sensor_files = {}
        
        # 扫描热成像图像
        thermal_files = []
        for f in self.thermal_dir.glob("*.png"):
            timestamp = self._extract_timestamp_from_filename(f.name)
            thermal_files.append((timestamp, str(f)))
        sensor_files['thermal'] = sorted(thermal_files)
        print(f"Found {len(thermal_files)} thermal images")
        
        # 扫描RGB图像
        rgb_left_files = []
        for f in self.rgb_left_dir.glob("*.png"):
            timestamp = self._extract_timestamp_from_filename(f.name)
            rgb_left_files.append((timestamp, str(f)))
        sensor_files['rgb_left'] = sorted(rgb_left_files)
        print(f"Found {len(rgb_left_files)} RGB left images")
        
        # 扫描右侧RGB图像（如果存在）
        if self.rgb_right_dir.exists():
            rgb_right_files = []
            for f in self.rgb_right_dir.glob("*.png"):
                timestamp = self._extract_timestamp_from_filename(f.name)
                rgb_right_files.append((timestamp, str(f)))
            sensor_files['rgb_right'] = sorted(rgb_right_files)
            print(f"Found {len(rgb_right_files)} RGB right images")
        
        # 扫描深度图像（如果存在）
        if self.depth_dir.exists():
            depth_files = []
            for f in self.depth_dir.glob("*.png"):
                timestamp = self._extract_timestamp_from_filename(f.name)
                depth_files.append((timestamp, str(f)))
            sensor_files['depth'] = sorted(depth_files)
            print(f"Found {len(depth_files)} depth images")
        
        # 扫描Ouster激光雷达
        ouster_files = []
        for f in self.ouster_dir.glob("*.bin"):
            timestamp = self._extract_timestamp_from_filename(f.name)
            ouster_files.append((timestamp, str(f)))
        sensor_files['ouster'] = sorted(ouster_files)
        print(f"Found {len(ouster_files)} Ouster LiDAR files")
        
        # 扫描Multisense激光雷达（如果存在）
        if self.multisense_lidar_dir.exists():
            multisense_files = []
            for f in self.multisense_lidar_dir.glob("*.bin"):
                timestamp = self._extract_timestamp_from_filename(f.name)
                multisense_files.append((timestamp, str(f)))
            sensor_files['multisense_lidar'] = sorted(multisense_files)
            print(f"Found {len(multisense_files)} Multisense LiDAR files")
        
        return sensor_files
    
    def _load_ground_truth_data(self) -> pd.DataFrame:
        """加载真实地面真值数据"""
        print(f"Loading ground truth from: {self.gt_trajectory_file}")
        
        gt_df = pd.read_csv(self.gt_trajectory_file)
        
        # 验证必要列存在
        required_columns = ['timestamp', 'position_x', 'position_y', 'position_z',
                          'orientation_roll', 'orientation_pitch', 'orientation_yaw']
        missing_cols = [col for col in required_columns if col not in gt_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ground truth: {missing_cols}")
        
        # 按时间戳排序
        gt_df = gt_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(gt_df)} ground truth poses")
        print(f"Time range: {gt_df['timestamp'].min():.6f} - {gt_df['timestamp'].max():.6f}")
        
        return gt_df
    
    def _find_temporal_matches(self, reference_timestamp: float, 
                              target_files: List[Tuple[float, str]]) -> Optional[Tuple[float, str]]:
        """查找时间戳最近邻匹配"""
        if not target_files:
            return None
        
        timestamps = [t for t, _ in target_files]
        
        # 使用二分查找
        idx = bisect.bisect_left(timestamps, reference_timestamp)
        
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps[idx - 1] - reference_timestamp)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - reference_timestamp)))
        
        if not candidates:
            return None
        
        # 选择时间差最小的
        best_idx, best_diff = min(candidates, key=lambda x: x[1])
        
        # 检查是否在阈值内
        threshold_s = self.time_threshold_ms / 1000.0
        if best_diff <= threshold_s:
            return target_files[best_idx]
        
        return None
    
    def _generate_matched_samples(self, sensor_files: Dict[str, List[Tuple[float, str]]], 
                                 gt_df: pd.DataFrame) -> List[Dict]:
        """生成时间对齐的样本"""
        print("Generating temporally aligned samples...")
        
        # 以热成像为基准时间戳（因为帧率相对稳定）
        thermal_files = sensor_files['thermal']
        
        samples = []
        skipped_count = 0
        
        for thermal_timestamp, thermal_path in thermal_files:
            try:
                sample = {
                    'timestamp': thermal_timestamp,
                    'thermal_image': thermal_path
                }
                
                # 查找RGB图像匹配
                rgb_match = self._find_temporal_matches(thermal_timestamp, sensor_files['rgb_left'])
                if rgb_match:
                    sample['rgb_image'] = rgb_match[1]
                    sample['rgb_timestamp'] = rgb_match[0]
                
                # 查找右侧RGB图像匹配（如果存在）
                if 'rgb_right' in sensor_files:
                    rgb_right_match = self._find_temporal_matches(thermal_timestamp, sensor_files['rgb_right'])
                    if rgb_right_match:
                        sample['rgb_right_image'] = rgb_right_match[1]
                
                # 查找深度图像匹配（如果存在）
                if 'depth' in sensor_files:
                    depth_match = self._find_temporal_matches(thermal_timestamp, sensor_files['depth'])
                    if depth_match:
                        sample['depth_image'] = depth_match[1]
                
                # 查找激光雷达匹配
                lidar_match = self._find_temporal_matches(thermal_timestamp, sensor_files['ouster'])
                if lidar_match:
                    sample['lidar_file'] = lidar_match[1]
                    sample['lidar_timestamp'] = lidar_match[0]
                
                # 查找地面真值位姿匹配
                gt_timestamps = gt_df['timestamp'].values
                gt_idx = np.argmin(np.abs(gt_timestamps - thermal_timestamp))
                gt_row = gt_df.iloc[gt_idx]
                
                # 检查时间差是否在阈值内
                time_diff = abs(gt_row['timestamp'] - thermal_timestamp)
                if time_diff <= self.time_threshold_ms / 1000.0:
                    sample['pose'] = {
                        'position': [gt_row['position_x'], gt_row['position_y'], gt_row['position_z']],
                        'orientation': [gt_row['orientation_roll'], gt_row['orientation_pitch'], gt_row['orientation_yaw']],
                        'gt_timestamp': gt_row['timestamp']
                    }
                
                # 验证样本有效性（至少需要热成像和一个其他模态）
                required_modalities = ['thermal_image']
                optional_modalities = ['rgb_image', 'lidar_file', 'pose']
                
                has_optional = any(mod in sample for mod in optional_modalities)
                has_required = all(mod in sample for mod in required_modalities)
                
                if has_required and has_optional:
                    # 添加元数据
                    sample['sample_id'] = len(samples)
                    sample['data_source'] = 'real_sensors'
                    
                    samples.append(sample)
                else:
                    skipped_count += 1
                
            except Exception as e:
                print(f"Warning: Skipping sample at timestamp {thermal_timestamp}: {e}")
                skipped_count += 1
        
        print(f"Generated {len(samples)} valid samples")
        print(f"Skipped {skipped_count} incomplete samples")
        
        return samples
    
    def _split_samples(self, samples: List[Dict], 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分割样本为训练/验证/测试集"""
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # 按时间戳排序确保一致性
        samples = sorted(samples, key=lambda x: x['timestamp'])
        
        total_samples = len(samples)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        print(f"Split summary:")
        print(f"  Train: {len(train_samples)} samples ({len(train_samples)/total_samples*100:.1f}%)")
        print(f"  Val: {len(val_samples)} samples ({len(val_samples)/total_samples*100:.1f}%)")
        print(f"  Test: {len(test_samples)} samples ({len(test_samples)/total_samples*100:.1f}%)")
        
        return train_samples, val_samples, test_samples
    
    def generate_indices(self, output_dir: str, 
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15):
        """生成完整的数据索引文件"""
        print("="*60)
        print("Generating MineSLAM Real Data Indices")
        print("="*60)
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 扫描传感器文件
        sensor_files = self._scan_sensor_files()
        
        # 2. 加载地面真值
        gt_df = self._load_ground_truth_data()
        
        # 3. 生成时间对齐样本
        samples = self._generate_matched_samples(sensor_files, gt_df)
        
        if not samples:
            raise ValueError("No valid samples generated from real data")
        
        # 4. 分割数据集
        train_samples, val_samples, test_samples = self._split_samples(
            samples, train_ratio, val_ratio, test_ratio)
        
        # 5. 保存索引文件
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            output_file = output_dir / f"{split_name}.jsonl"
            
            with open(output_file, 'w') as f:
                for sample in split_samples:
                    f.write(json.dumps(sample) + '\n')
            
            print(f"✓ Saved {split_name} index: {output_file} ({len(split_samples)} samples)")
        
        # 6. 生成统计信息
        self._generate_statistics(samples, output_dir)
        
        print("="*60)
        print("Real data index generation completed successfully!")
        print("="*60)
    
    def _generate_statistics(self, samples: List[Dict], output_dir: Path):
        """生成数据集统计信息"""
        stats = {
            'total_samples': len(samples),
            'time_range': {
                'start': min(s['timestamp'] for s in samples),
                'end': max(s['timestamp'] for s in samples),
                'duration': max(s['timestamp'] for s in samples) - min(s['timestamp'] for s in samples)
            },
            'modality_coverage': {},
            'temporal_alignment_threshold_ms': self.time_threshold_ms,
            'data_source': str(self.dataset_root)
        }
        
        # 统计各模态覆盖率
        modalities = ['thermal_image', 'rgb_image', 'depth_image', 'lidar_file', 'pose']
        for modality in modalities:
            count = sum(1 for s in samples if modality in s)
            stats['modality_coverage'][modality] = {
                'count': count,
                'coverage': count / len(samples) * 100
            }
        
        # 保存统计信息
        stats_file = output_dir / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Generated statistics: {stats_file}")
        
        # 打印摘要
        print("\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Time duration: {stats['time_range']['duration']:.1f} seconds")
        print("  Modality coverage:")
        for modality, info in stats['modality_coverage'].items():
            print(f"    {modality}: {info['count']} samples ({info['coverage']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate real data indices for MineSLAM')
    parser.add_argument('--dataset_root', type=str, 
                       default='/root/autodl-tmp/new huanjing ganzhi-cc/sr_B_route2_deep_learning',
                       help='Root directory of real dataset')
    parser.add_argument('--output_dir', type=str,
                       default='/root/autodl-tmp/new huanjing ganzhi-cc/lists',
                       help='Output directory for index files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--time_threshold_ms', type=int, default=50,
                       help='Temporal alignment threshold in milliseconds')
    
    args = parser.parse_args()
    
    try:
        generator = RealDataIndexGenerator(args.dataset_root)
        generator.time_threshold_ms = args.time_threshold_ms
        
        generator.generate_indices(
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
    except Exception as e:
        print(f"Error generating indices: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())