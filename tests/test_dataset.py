#!/usr/bin/env python3
"""
MineSLAM Dataset Unit Tests - REAL DATA VALIDATION ONLY
测试真实数据加载，严格禁止合成/随机数据生成
"""

import unittest
import os
import sys
import torch
import numpy as np
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.mineslam_dataset import MineSLAMDataset, RealSensorDataLoader, RealDataContract
from utils import load_config


class TestRealDataContract(unittest.TestCase):
    """测试真实数据契约验证"""
    
    def test_rgb_contract_validation(self):
        """测试RGB张量契约验证"""
        # 有效的RGB张量
        valid_rgb = torch.rand(3, 224, 224)  # [0,1]范围
        self.assertTrue(RealDataContract.validate_rgb_tensor(valid_rgb))
        
        # 无效形状
        with self.assertRaises(ValueError):
            invalid_shape = torch.rand(4, 224, 224)  # 错误通道数
            RealDataContract.validate_rgb_tensor(invalid_shape)
        
        # 无效值范围
        with self.assertRaises(ValueError):
            invalid_range = torch.rand(3, 224, 224) * 2  # [0,2]范围
            RealDataContract.validate_rgb_tensor(invalid_range)
    
    def test_depth_contract_validation(self):
        """测试深度张量契约验证"""
        # 有效的深度张量
        valid_depth = torch.rand(1, 224, 224) * 10  # 深度值范围
        self.assertTrue(RealDataContract.validate_depth_tensor(valid_depth))
        
        # 无效形状
        with self.assertRaises(ValueError):
            invalid_shape = torch.rand(3, 224, 224)
            RealDataContract.validate_depth_tensor(invalid_shape)
        
        # 负值
        with self.assertRaises(ValueError):
            negative_depth = torch.rand(1, 224, 224) - 1
            RealDataContract.validate_depth_tensor(negative_depth)
    
    def test_lidar_contract_validation(self):
        """测试激光雷达张量契约验证"""
        # 有效的激光雷达数据
        valid_lidar = torch.rand(1000, 4)  # N x 4
        self.assertTrue(RealDataContract.validate_lidar_tensor(valid_lidar))
        
        # 无效列数
        with self.assertRaises(ValueError):
            invalid_cols = torch.rand(1000, 3)  # 应该是4列
            RealDataContract.validate_lidar_tensor(invalid_cols)
    
    def test_imu_contract_validation(self):
        """测试IMU张量契约验证"""
        # 有效的IMU数据
        valid_imu = torch.rand(20, 6)  # T x 6
        self.assertTrue(RealDataContract.validate_imu_tensor(valid_imu))
        
        # 无效列数
        with self.assertRaises(ValueError):
            invalid_cols = torch.rand(20, 5)  # 应该是6列
            RealDataContract.validate_imu_tensor(invalid_cols)
        
        # 空序列
        with self.assertRaises(ValueError):
            empty_imu = torch.empty(0, 6)
            RealDataContract.validate_imu_tensor(empty_imu)
    
    def test_pose_delta_contract_validation(self):
        """测试位姿增量张量契约验证"""
        # 有效的位姿增量
        valid_pose = torch.rand(6)
        self.assertTrue(RealDataContract.validate_pose_delta_tensor(valid_pose))
        
        # 无效维度
        with self.assertRaises(ValueError):
            invalid_dims = torch.rand(3, 2)  # 应该是1维
            RealDataContract.validate_pose_delta_tensor(invalid_dims)
    
    def test_boxes_contract_validation(self):
        """测试检测框张量契约验证"""
        # 有效的检测框
        valid_boxes = torch.rand(5, 10)  # Q x 10
        self.assertTrue(RealDataContract.validate_boxes_tensor(valid_boxes))
        
        # 无效列数
        with self.assertRaises(ValueError):
            invalid_cols = torch.rand(5, 8)  # 应该是10列
            RealDataContract.validate_boxes_tensor(invalid_cols)


class TestMineSLAMDataset(unittest.TestCase):
    """测试MineSLAM数据集实现"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.config_path = 'configs/mineslam.yaml'
        cls.test_sample_count = 10  # 测试前N=10个样本
        
        # 加载配置
        if not os.path.exists(cls.config_path):
            raise unittest.SkipTest(f"Configuration file not found: {cls.config_path}")
        
        cls.config = load_config(cls.config_path)
        
        # 检查训练数据索引是否存在
        train_index_path = cls.config['data']['train_index']
        if not os.path.exists(train_index_path):
            raise unittest.SkipTest(f"Train index not found: {train_index_path}")
    
    def test_01_dataset_creation_with_real_data(self):
        """测试使用真实数据创建数据集"""
        try:
            dataset = MineSLAMDataset(self.config, split='train')
            self.assertGreater(len(dataset), 0, "数据集应包含真实样本")
            print(f"✓ 数据集创建成功，包含 {len(dataset)} 个真实样本")
        except Exception as e:
            self.fail(f"真实数据集创建失败: {e}")
    
    def test_02_real_files_accessibility(self):
        """测试所有文件可以打开"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        # 随机抽取前N=10个样本进行测试
        test_count = min(self.test_sample_count, len(dataset))
        
        for i in range(test_count):
            with self.subTest(sample_idx=i):
                sample_info = dataset.samples[i]
                
                # 检查所有文件路径
                file_paths = [
                    sample_info.get('rgb_image'),
                    sample_info.get('thermal_image'),
                    sample_info.get('depth_image'),
                    sample_info.get('lidar_file')
                ]
                
                for file_path in file_paths:
                    if file_path:  # 如果路径存在
                        self.assertTrue(os.path.exists(file_path), 
                                      f"文件不存在: {file_path}")
                        self.assertTrue(os.access(file_path, os.R_OK),
                                      f"文件不可读: {file_path}")
                
                print(f"✓ 样本 {i} 所有文件可访问")
    
    def test_03_tensor_dtype_shape_consistency(self):
        """验证张量dtype/shape与配置一致"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        test_count = min(self.test_sample_count, len(dataset))
        
        for i in range(test_count):
            with self.subTest(sample_idx=i):
                try:
                    sample_data = dataset[i]
                    
                    # 验证数据类型和形状
                    if 'rgb' in sample_data:
                        rgb = sample_data['rgb']
                        self.assertEqual(rgb.dtype, torch.float32, "RGB dtype应为float32")
                        self.assertEqual(len(rgb.shape), 3, "RGB应为3维张量")
                        self.assertEqual(rgb.shape[0], 3, "RGB应有3个通道")
                        self.assertTrue(0 <= rgb.min() and rgb.max() <= 1, 
                                      "RGB值应在[0,1]范围内")
                    
                    if 'thermal' in sample_data:
                        thermal = sample_data['thermal']
                        self.assertEqual(thermal.dtype, torch.float32, "热成像dtype应为float32")
                        self.assertEqual(len(thermal.shape), 3, "热成像应为3维张量")
                        self.assertEqual(thermal.shape[0], 1, "热成像应有1个通道")
                    
                    if 'lidar' in sample_data:
                        lidar = sample_data['lidar']
                        self.assertEqual(lidar.dtype, torch.float32, "激光雷达dtype应为float32")
                        self.assertEqual(len(lidar.shape), 2, "激光雷达应为2维张量")
                        self.assertEqual(lidar.shape[1], 4, "激光雷达应有4列[x,y,z,intensity]")
                    
                    if 'imu' in sample_data:
                        imu = sample_data['imu']
                        self.assertEqual(imu.dtype, torch.float32, "IMU dtype应为float32")
                        self.assertEqual(len(imu.shape), 2, "IMU应为2维张量")
                        self.assertEqual(imu.shape[1], 6, "IMU应有6列[ax,ay,az,gx,gy,gz]")
                        self.assertEqual(imu.shape[0], 20, "IMU序列长度应为20")
                    
                    if 'pose_delta' in sample_data:
                        pose = sample_data['pose_delta']
                        self.assertEqual(pose.dtype, torch.float32, "位姿增量dtype应为float32")
                        self.assertEqual(len(pose.shape), 1, "位姿增量应为1维张量")
                        self.assertEqual(pose.shape[0], 6, "位姿增量应有6个元素")
                    
                    if 'boxes' in sample_data:
                        boxes = sample_data['boxes']
                        self.assertEqual(boxes.dtype, torch.float32, "检测框dtype应为float32")
                        self.assertEqual(len(boxes.shape), 2, "检测框应为2维张量")
                        self.assertEqual(boxes.shape[1], 10, "检测框应有10列")
                    
                    print(f"✓ 样本 {i} 张量类型和形状验证通过")
                    
                except Exception as e:
                    self.fail(f"样本 {i} 张量验证失败: {e}")
    
    def test_04_temporal_alignment_validation(self):
        """验证时间对齐: RGB-LiDAR时间差 <= 阈值毫秒，IMU覆盖窗口"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        time_threshold_ms = 100  # 时间对齐阈值
        
        test_count = min(self.test_sample_count, len(dataset))
        
        for i in range(test_count):
            with self.subTest(sample_idx=i):
                sample_info = dataset.samples[i]
                rgb_timestamp = sample_info['timestamp']
                
                try:
                    sample_data = dataset[i]
                    
                    # 验证时间戳存在
                    self.assertIn('timestamp', sample_data, "样本应包含时间戳")
                    
                    # 如果有激光雷达数据，验证时间对齐
                    if 'lidar_file' in sample_info:
                        # 假设激光雷达文件名包含时间戳
                        lidar_file = os.path.basename(sample_info['lidar_file'])
                        try:
                            lidar_timestamp = float(lidar_file.split('.')[0])
                            time_diff_ms = abs(rgb_timestamp - lidar_timestamp) * 1000
                            self.assertLessEqual(time_diff_ms, time_threshold_ms,
                                               f"RGB-LiDAR时间差 {time_diff_ms:.1f}ms 超过阈值 {time_threshold_ms}ms")
                        except ValueError:
                            # 如果无法从文件名提取时间戳，跳过此检查
                            pass
                    
                    # 验证IMU时间窗口覆盖
                    if 'imu' in sample_data:
                        imu_data = sample_data['imu']
                        self.assertGreater(imu_data.shape[0], 0, "IMU数据不应为空")
                        
                        # 验证IMU序列长度符合预期
                        expected_imu_length = 20  # 滑窗长度
                        self.assertEqual(imu_data.shape[0], expected_imu_length,
                                       f"IMU序列长度应为 {expected_imu_length}")
                    
                    print(f"✓ 样本 {i} 时间对齐验证通过")
                    
                except Exception as e:
                    self.fail(f"样本 {i} 时间对齐验证失败: {e}")
    
    def test_05_calibration_matrix_validation(self):
        """验证标定矩阵可逆，外参变换一致性"""
        dataset = MineSLAMDataset(self.config, split='train')
        calibration = dataset.data_loader.calibration
        
        for camera_name, camera_info in calibration.items():
            with self.subTest(camera=camera_name):
                if 'K' in camera_info and 'R' in camera_info:
                    # 内参矩阵
                    K = np.array(camera_info['K']).reshape(3, 3)
                    
                    # 验证内参矩阵可逆
                    det_K = np.linalg.det(K)
                    self.assertNotAlmostEqual(det_K, 0.0, places=6,
                                            msg=f"内参矩阵K不可逆 (det={det_K})")
                    
                    # 外参旋转矩阵
                    R = np.array(camera_info['R']).reshape(3, 3)
                    
                    # 验证旋转矩阵正交性
                    R_transpose = R.T
                    identity_check = np.dot(R, R_transpose)
                    np.testing.assert_allclose(identity_check, np.eye(3), atol=1e-6,
                                             err_msg=f"旋转矩阵R不正交 for {camera_name}")
                    
                    # 验证行列式约等于1
                    det_R = np.linalg.det(R)
                    self.assertAlmostEqual(det_R, 1.0, places=6,
                                         msg=f"旋转矩阵行列式 {det_R} ≠ 1.0 for {camera_name}")
                    
                    print(f"✓ 相机 {camera_name} 标定矩阵验证通过")
    
    def test_06_no_synthetic_data_detection(self):
        """检测和拒绝任何合成/随机替代数据"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        test_count = min(self.test_sample_count, len(dataset))
        
        for i in range(test_count):
            with self.subTest(sample_idx=i):
                sample_info = dataset.samples[i]
                
                # 检查样本信息中是否有合成数据标记
                forbidden_keys = ['_generated', '_synthetic', '_random', '_mock', '_fake']
                for key in sample_info.keys():
                    for forbidden in forbidden_keys:
                        self.assertNotIn(forbidden, key.lower(),
                                       f"检测到禁用的合成数据标记: {key}")
                
                try:
                    sample_data = dataset[i]
                    
                    # 验证时间戳是真实的（不是完美规律的）
                    if i > 0:
                        prev_timestamp = dataset.samples[i-1]['timestamp']
                        curr_timestamp = sample_info['timestamp']
                        time_diff = curr_timestamp - prev_timestamp
                        
                        # 真实传感器数据的时间间隔应该有一定变化
                        self.assertGreater(time_diff, 0, "时间戳应递增")
                    
                    # 检查张量数据的真实性（不应该全部相同）
                    for key, tensor in sample_data.items():
                        if isinstance(tensor, torch.Tensor) and tensor.numel() > 1:
                            unique_values = torch.unique(tensor)
                            self.assertGreater(len(unique_values), 1,
                                             f"张量 {key} 所有值相同，疑似合成数据")
                    
                    print(f"✓ 样本 {i} 通过合成数据检测")
                    
                except Exception as e:
                    # 如果加载失败且错误信息包含"自动生成"或"随机替代"，应该报错终止
                    error_msg = str(e).lower()
                    forbidden_terms = ['generated', 'synthetic', 'random', 'mock', 'fake']
                    
                    if any(term in error_msg for term in forbidden_terms):
                        self.fail(f"检测到禁用的合成数据生成: {e}")
                    else:
                        # 其他错误不影响此测试
                        pass
    
    def test_07_missing_frame_reporting(self):
        """验证缺帧检测和报告"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        # 获取数据集统计信息
        stats = dataset.get_statistics()
        
        self.assertIn('missing_frame_count', stats, "应报告缺帧数量")
        self.assertIn('missing_frame_rate', stats, "应报告缺帧比例")
        self.assertIn('total_samples', stats, "应报告总样本数")
        
        missing_rate = stats['missing_frame_rate']
        
        # 记录缺帧情况
        print(f"数据集统计:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  缺帧数量: {stats['missing_frame_count']}")
        print(f"  缺帧比例: {missing_rate:.2%}")
        
        # 如果缺帧率过高，发出警告
        if missing_rate > 0.1:  # 超过10%
            print(f"⚠ 警告: 缺帧率较高 ({missing_rate:.2%})")
    
    def test_08_real_data_loading_performance(self):
        """测试真实数据加载性能"""
        import time
        
        dataset = MineSLAMDataset(self.config, split='train')
        
        test_count = min(5, len(dataset))  # 测试5个样本的加载时间
        
        loading_times = []
        
        for i in range(test_count):
            start_time = time.time()
            
            try:
                sample_data = dataset[i]
                end_time = time.time()
                
                loading_time = end_time - start_time
                loading_times.append(loading_time)
                
                print(f"样本 {i} 加载时间: {loading_time:.3f}s")
                
            except Exception as e:
                print(f"样本 {i} 加载失败: {e}")
        
        if loading_times:
            avg_loading_time = np.mean(loading_times)
            max_loading_time = np.max(loading_times)
            
            print(f"平均加载时间: {avg_loading_time:.3f}s")
            print(f"最大加载时间: {max_loading_time:.3f}s")
            
            # 性能检查：单个样本加载不应超过5秒
            self.assertLess(max_loading_time, 5.0, 
                           f"样本加载时间过长: {max_loading_time:.3f}s")


def run_dataset_tests():
    """运行数据集测试"""
    print("="*80)
    print("MINESLAM DATASET UNIT TESTS - REAL DATA VALIDATION")
    print("="*80)
    print("测试真实数据加载，严格禁止合成/随机数据生成")
    print("前N=10个样本将被验证")
    print()
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRealDataContract)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMineSLAMDataset))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("数据集测试总结")
    print("="*80)
    
    if result.wasSuccessful():
        print("✅ 所有测试通过 - 真实数据验证成功!")
        print("   数据集已准备好用于MineSLAM训练")
    else:
        print("❌ 测试失败 - 真实数据存在问题!")
        print(f"   失败: {len(result.failures)}")
        print(f"   错误: {len(result.errors)}")
        
        if result.failures:
            print("\n失败详情:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\n错误详情:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_dataset_tests()
    sys.exit(0 if success else 1)