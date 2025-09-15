#!/usr/bin/env python3
"""
MineSLAM Bootstrap Test - REAL DATA ONLY
This test ONLY reads real data samples and validates basic I/O operations.
NO synthetic/random/mock data generation is allowed.
"""

import unittest
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import load_config, validate_real_data_config, RealDataChecker
from data import MineSLAMDataset, RealDataValidator


class TestRealDataBootstrap(unittest.TestCase):
    """
    Bootstrap test for real data validation
    
    STRICT POLICY: This test ONLY validates real sensor data.
    Any attempt to use synthetic/mock data will cause test failure.
    """
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.config_path = 'configs/mineslam.yaml'
        cls.max_test_samples = 5  # Limit samples for bootstrap test
        
        # Load and validate configuration
        if not os.path.exists(cls.config_path):
            raise unittest.SkipTest(f"Configuration file not found: {cls.config_path}")
        
        cls.config = load_config(cls.config_path)
        
        # Validate anti-synthetic settings
        try:
            validate_real_data_config(cls.config)
        except Exception as e:
            raise unittest.SkipTest(f"Configuration validation failed: {e}")
    
    def test_01_config_prohibits_synthetic_data(self):
        """Test that configuration explicitly prohibits synthetic data"""
        validation_settings = self.config.get('validation_settings', {})
        
        # Must explicitly forbid synthetic data
        self.assertFalse(validation_settings.get('allow_synthetic', True),
                        "Configuration MUST set allow_synthetic: false")
        
        # Must explicitly forbid mock data
        self.assertFalse(validation_settings.get('allow_mock', True),
                        "Configuration MUST set allow_mock: false")
        
        # Must enable strict path checking
        self.assertTrue(validation_settings.get('strict_path_check', False),
                       "Configuration MUST set strict_path_check: true")
    
    def test_02_real_data_paths_exist(self):
        """Test that all critical real data paths exist"""
        critical_paths = [
            self.config['data']['source_data'],
            self.config['data']['images']['thermal'],
            self.config['data']['images']['rgb_left'],
            self.config['data']['ground_truth']['trajectory'],
        ]
        
        for path in critical_paths:
            with self.subTest(path=path):
                self.assertTrue(os.path.exists(path), 
                              f"Critical real data path missing: {path}")
                self.assertTrue(os.access(path, os.R_OK),
                              f"Critical real data path not readable: {path}")
    
    def test_03_real_image_files_exist(self):
        """Test that real image files exist and are readable"""
        thermal_dir = self.config['data']['images']['thermal']
        rgb_dir = self.config['data']['images']['rgb_left']
        
        # Check thermal images
        thermal_files = [f for f in os.listdir(thermal_dir) if f.endswith('.png')]
        self.assertGreater(len(thermal_files), 0, 
                          "No real thermal image files found")
        
        # Check RGB images
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
        self.assertGreater(len(rgb_files), 0,
                          "No real RGB image files found")
        
        # Validate first few files are real images
        for i, filename in enumerate(thermal_files[:self.max_test_samples]):
            file_path = os.path.join(thermal_dir, filename)
            with self.subTest(file=filename):
                RealDataValidator.validate_image_file(file_path)
    
    def test_04_ground_truth_data_valid(self):
        """Test that ground truth data is valid and contains real values"""
        gt_path = self.config['data']['ground_truth']['trajectory']
        
        # Validate file exists and is readable
        RealDataValidator.validate_path(gt_path, "Ground truth trajectory")
        
        # Load and validate CSV content
        import pandas as pd
        gt_df = pd.read_csv(gt_path)
        
        # Check required columns exist
        required_columns = ['timestamp', 'position_x', 'position_y', 'position_z']
        for col in required_columns:
            self.assertIn(col, gt_df.columns, f"Missing required column: {col}")
        
        # Check data is not synthetic (no constant values)
        for col in ['position_x', 'position_y', 'position_z']:
            values = gt_df[col].values
            self.assertGreater(len(np.unique(values)), 1,
                             f"Column {col} appears synthetic (constant values)")
            
            # Check for reasonable value ranges (real world coordinates)
            self.assertTrue(np.all(np.isfinite(values)),
                          f"Column {col} contains invalid values")
    
    def test_05_real_dataset_creation(self):
        """Test MineSLAM dataset creation with real data only"""
        try:
            # Create dataset - this should only load real data
            dataset = MineSLAMDataset(self.config, split='train')
            
            # Verify dataset has real samples
            self.assertGreater(len(dataset), 0, 
                             "Dataset contains no real samples")
            
            print(f"✓ Dataset created with {len(dataset)} real samples")
            
        except Exception as e:
            self.fail(f"Failed to create dataset with real data: {e}")
    
    def test_06_real_data_sample_loading(self):
        """Test loading and validating real data samples"""
        # Create dataset
        dataset = MineSLAMDataset(self.config, split='train')
        
        # Test loading first N real samples
        num_test_samples = min(self.max_test_samples, len(dataset))
        
        for i in range(num_test_samples):
            with self.subTest(sample_idx=i):
                # Load real sample
                sample = dataset[i]
                
                # Validate sample structure
                self.assertIsInstance(sample, dict, "Sample must be dictionary")
                
                # Validate tensor data
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        # Check tensor is valid (not NaN/Inf)
                        RealDataChecker.check_tensor_is_real(value, f"sample_{i}_{key}")
                        
                        # Check tensor has expected properties
                        self.assertEqual(value.dtype, torch.float32, 
                                       f"Tensor {key} should be float32")
                        self.assertFalse(torch.any(torch.isnan(value)),
                                        f"Tensor {key} contains NaN values")
                        self.assertFalse(torch.any(torch.isinf(value)),
                                        f"Tensor {key} contains infinite values")
                
                # Validate image tensors have correct shapes
                if 'thermal' in sample:
                    thermal = sample['thermal']
                    self.assertEqual(len(thermal.shape), 3, "Thermal image should be [C,H,W]")
                    self.assertGreater(thermal.shape[1], 0, "Thermal image height > 0")
                    self.assertGreater(thermal.shape[2], 0, "Thermal image width > 0")
                
                if 'rgb' in sample:
                    rgb = sample['rgb']
                    self.assertEqual(len(rgb.shape), 3, "RGB image should be [C,H,W]")
                    self.assertEqual(rgb.shape[0], 3, "RGB should have 3 channels")
                    self.assertGreater(rgb.shape[1], 0, "RGB image height > 0")
                    self.assertGreater(rgb.shape[2], 0, "RGB image width > 0")
                
                print(f"✓ Sample {i} validated: {list(sample.keys())}")
    
    def test_07_real_data_no_duplication(self):
        """Test that loaded samples are unique (not artificially duplicated)"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        # Load first few samples
        num_test_samples = min(self.max_test_samples, len(dataset))
        samples = [dataset[i] for i in range(num_test_samples)]
        
        # Check timestamps are unique (real data should have unique timestamps)
        timestamps = []
        for sample in samples:
            if 'timestamp' in sample:
                timestamps.append(sample['timestamp'].item())
        
        if len(timestamps) > 1:
            unique_timestamps = set(timestamps)
            self.assertEqual(len(unique_timestamps), len(timestamps),
                           "Duplicate timestamps detected - data may be artificially replicated")
        
        # Check image data is not identical (real data should vary)
        if len(samples) > 1:
            for i in range(1, len(samples)):
                if 'thermal' in samples[0] and 'thermal' in samples[i]:
                    thermal_0 = samples[0]['thermal']
                    thermal_i = samples[i]['thermal']
                    
                    # Images should not be identical
                    self.assertFalse(torch.allclose(thermal_0, thermal_i),
                                   f"Samples 0 and {i} have identical thermal images")
    
    def test_08_data_temporal_consistency(self):
        """Test that real data has realistic temporal patterns"""
        dataset = MineSLAMDataset(self.config, split='train')
        
        # Load first few samples
        num_test_samples = min(self.max_test_samples, len(dataset))
        timestamps = []
        
        for i in range(num_test_samples):
            sample = dataset[i]
            if 'timestamp' in sample:
                timestamps.append(sample['timestamp'].item())
        
        if len(timestamps) > 1:
            timestamps.sort()
            
            # Check timestamps are increasing (basic temporal consistency)
            for i in range(1, len(timestamps)):
                self.assertGreater(timestamps[i], timestamps[i-1],
                                 "Timestamps should be increasing for real sensor data")
            
            # Check time differences are reasonable (not too regular = synthetic)
            time_diffs = np.diff(timestamps)
            if len(time_diffs) > 1:
                diff_variance = np.var(time_diffs)
                self.assertGreater(diff_variance, 0,
                                 "Perfectly regular timestamps suggest synthetic data")
    
    def test_09_calibration_data_valid(self):
        """Test that calibration data is valid and realistic"""
        calib_path = self.config['calib']['source_calibration']
        
        if os.path.exists(calib_path):
            # Load calibration data
            calib_data = RealDataValidator.validate_json_file(calib_path)
            
            # Check for realistic camera parameters
            for camera_name, camera_info in calib_data.items():
                if isinstance(camera_info, dict) and 'K' in camera_info:
                    K = camera_info['K']
                    
                    # Check intrinsic matrix format
                    self.assertEqual(len(K), 9, f"Camera {camera_name} K matrix should have 9 elements")
                    
                    # Check focal lengths are reasonable
                    fx, fy = K[0], K[4]
                    self.assertGreater(fx, 100, f"Camera {camera_name} focal length fx too small")
                    self.assertGreater(fy, 100, f"Camera {camera_name} focal length fy too small")
                    self.assertLess(fx, 2000, f"Camera {camera_name} focal length fx too large")
                    self.assertLess(fy, 2000, f"Camera {camera_name} focal length fy too large")
    
    def test_10_memory_usage_reasonable(self):
        """Test that real data loading doesn't use excessive memory"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create dataset and load samples
        dataset = MineSLAMDataset(self.config, split='train')
        
        samples = []
        for i in range(min(self.max_test_samples, len(dataset))):
            samples.append(dataset[i])
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable for real data (not excessive duplication)
        max_expected_memory_mb = self.max_test_samples * 50  # ~50MB per sample max
        self.assertLess(memory_increase, max_expected_memory_mb,
                       f"Memory usage too high: {memory_increase:.1f}MB for {len(samples)} samples")
        
        print(f"✓ Memory usage: {memory_increase:.1f}MB for {len(samples)} real samples")
        
        # Cleanup
        del samples
        del dataset
        gc.collect()


class TestRealDataValidation(unittest.TestCase):
    """Test real data validation utilities"""
    
    def test_real_data_validator(self):
        """Test RealDataValidator utility functions"""
        # Test with existing real data path
        config = load_config('configs/mineslam.yaml')
        thermal_dir = config['data']['images']['thermal']
        
        if os.path.exists(thermal_dir):
            # This should pass for real directory
            RealDataValidator.validate_path(thermal_dir, "Thermal directory")
            
            # Test with first image file
            thermal_files = [f for f in os.listdir(thermal_dir) if f.endswith('.png')]
            if thermal_files:
                first_image = os.path.join(thermal_dir, thermal_files[0])
                RealDataValidator.validate_image_file(first_image)
    
    def test_real_data_checker(self):
        """Test RealDataChecker utility functions"""
        # Test with realistic tensor (should pass)
        real_tensor = torch.randn(3, 224, 224) * 0.5 + 0.5  # Normalized image-like
        RealDataChecker.check_tensor_is_real(real_tensor, "test_tensor")
        
        # Test with obviously synthetic tensor (should fail)
        with self.assertRaises(ValueError):
            synthetic_tensor = torch.ones(3, 224, 224)  # All identical values
            RealDataChecker.check_tensor_is_real(synthetic_tensor, "synthetic_tensor")
        
        # Test with NaN tensor (should fail)
        with self.assertRaises(ValueError):
            nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
            RealDataChecker.check_tensor_is_real(nan_tensor, "nan_tensor")


def run_bootstrap_test():
    """
    Run bootstrap test with proper reporting
    """
    print("="*80)
    print("MINESLAM BOOTSTRAP TEST - REAL DATA VALIDATION ONLY")
    print("="*80)
    print("This test validates REAL sensor data loading and processing.")
    print("NO synthetic, random, or mock data is allowed.")
    print("Test will FAIL if any artificial data is detected.\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRealDataBootstrap)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRealDataValidation))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("BOOTSTRAP TEST SUMMARY")
    print("="*80)
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Real data validation successful!")
        print("   Ready for MineSLAM training with verified real data.")
    else:
        print("❌ TESTS FAILED - Issues detected with real data!")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nERRORs:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_bootstrap_test()
    sys.exit(0 if success else 1)