#!/usr/bin/env python3
"""
MineSLAM Path Checker
Validates all real data paths exist and are readable
STRICT POLICY: Exits immediately if any path is missing
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any


class RealDataPathValidator:
    """Validates that all configured paths point to real, existing data"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.errors = []
        self.warnings = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_all_paths(self) -> bool:
        """
        Validate all paths in configuration
        
        Returns:
            True if all paths are valid, False otherwise
        """
        print("="*80)
        print("MINESLAM REAL DATA PATH VALIDATION")
        print("="*80)
        print(f"Validating configuration: {self.config_path}")
        print("STRICT MODE: Will exit if ANY path is missing\n")
        
        # Validate anti-synthetic settings first
        if not self._validate_anti_synthetic_settings():
            return False
        
        # Validate critical data paths
        if not self._validate_data_paths():
            return False
        
        # Validate calibration paths
        if not self._validate_calibration_paths():
            return False
        
        # Validate output directories (create if needed)
        self._validate_output_directories()
        
        # Print summary
        self._print_validation_summary()
        
        return len(self.errors) == 0
    
    def _validate_anti_synthetic_settings(self) -> bool:
        """Validate anti-synthetic data settings"""
        print("🚫 ANTI-SYNTHETIC DATA VALIDATION:")
        
        validation_settings = self.config.get('validation_settings', {})
        
        # Check allow_synthetic flag
        if validation_settings.get('allow_synthetic', True):
            self.errors.append("Configuration allows synthetic data! Set allow_synthetic: false")
            print("  ✗ allow_synthetic is enabled (FORBIDDEN)")
            return False
        else:
            print("  ✓ allow_synthetic is disabled")
        
        # Check allow_mock flag
        if validation_settings.get('allow_mock', True):
            self.errors.append("Configuration allows mock data! Set allow_mock: false")
            print("  ✗ allow_mock is enabled (FORBIDDEN)")
            return False
        else:
            print("  ✓ allow_mock is disabled")
        
        # Check strict_path_check flag
        if not validation_settings.get('strict_path_check', False):
            self.errors.append("strict_path_check must be enabled for real data validation")
            print("  ✗ strict_path_check is disabled (MUST BE ENABLED)")
            return False
        else:
            print("  ✓ strict_path_check is enabled")
        
        print("  ✅ Anti-synthetic settings validated\n")
        return True
    
    def _validate_data_paths(self) -> bool:
        """Validate all data paths point to real files/directories"""
        print("📁 REAL DATA PATH VALIDATION:")
        
        data_config = self.config.get('data', {})
        
        # Critical paths that MUST exist
        critical_paths = [
            ('data.root', data_config.get('root')),
            ('data.source_data', data_config.get('source_data')),
            ('data.ground_truth.trajectory', data_config.get('ground_truth', {}).get('trajectory')),
            ('data.ground_truth.stats', data_config.get('ground_truth', {}).get('stats')),
        ]
        
        # Image directories
        images_config = data_config.get('images', {})
        image_paths = [
            ('data.images.thermal', images_config.get('thermal')),
            ('data.images.rgb_left', images_config.get('rgb_left')),
            ('data.images.rgb_right', images_config.get('rgb_right')),
            ('data.images.depth', images_config.get('depth')),
        ]
        
        # Point cloud directories
        pointclouds_config = data_config.get('pointclouds', {})
        pointcloud_paths = [
            ('data.pointclouds.multisense', pointclouds_config.get('multisense')),
            ('data.pointclouds.ouster', pointclouds_config.get('ouster')),
            ('data.pointclouds.assembled', pointclouds_config.get('assembled')),
        ]
        
        # Other sensor data
        sensor_paths = [
            ('data.imu', data_config.get('imu')),
            ('data.odometry', data_config.get('odometry')),
        ]
        
        all_paths = critical_paths + image_paths + pointcloud_paths + sensor_paths
        
        for path_name, path_value in all_paths:
            if path_value is None:
                self.errors.append(f"Path not configured: {path_name}")
                print(f"  ✗ {path_name}: NOT CONFIGURED")
                continue
            
            if not os.path.exists(path_value):
                self.errors.append(f"Path does not exist: {path_name} = {path_value}")
                print(f"  ✗ {path_name}: NOT FOUND")
                print(f"      Path: {path_value}")
            else:
                # Check if it's readable
                if not os.access(path_value, os.R_OK):
                    self.errors.append(f"Path not readable: {path_name} = {path_value}")
                    print(f"  ✗ {path_name}: NOT READABLE")
                else:
                    print(f"  ✓ {path_name}: EXISTS & READABLE")
                    
                    # For directories, check if they contain files
                    if os.path.isdir(path_value):
                        file_count = len([f for f in os.listdir(path_value) 
                                        if os.path.isfile(os.path.join(path_value, f))])
                        if file_count == 0:
                            self.warnings.append(f"Directory is empty: {path_name}")
                            print(f"      ⚠ Directory is empty ({file_count} files)")
                        else:
                            print(f"      Contains {file_count} files")
        
        # Validate data indices if they exist
        index_paths = [
            ('data.train_index', data_config.get('train_index')),
            ('data.val_index', data_config.get('val_index')),
            ('data.test_index', data_config.get('test_index')),
        ]
        
        print("\n  📋 DATA INDEX VALIDATION:")
        for path_name, path_value in index_paths:
            if path_value and os.path.exists(path_value):
                sample_count = self._count_jsonl_samples(path_value)
                print(f"    ✓ {path_name}: {sample_count} samples")
            else:
                print(f"    - {path_name}: Will be generated from real data")
        
        print()
        return len(self.errors) == 0
    
    def _validate_calibration_paths(self) -> bool:
        """Validate calibration file paths"""
        print("🎯 CALIBRATION DATA VALIDATION:")
        
        calib_config = self.config.get('calib', {})
        
        calib_paths = [
            ('calib.extrinsics', calib_config.get('extrinsics')),
            ('calib.intrinsics', calib_config.get('intrinsics')),
            ('calib.source_calibration', calib_config.get('source_calibration')),
        ]
        
        for path_name, path_value in calib_paths:
            if path_value is None:
                self.warnings.append(f"Calibration path not configured: {path_name}")
                print(f"  - {path_name}: NOT CONFIGURED")
                continue
            
            if not os.path.exists(path_value):
                # For extrinsics/intrinsics, these may not exist yet (will be created)
                if path_name in ['calib.extrinsics', 'calib.intrinsics']:
                    print(f"  - {path_name}: Will be generated from source calibration")
                else:
                    self.errors.append(f"Calibration file missing: {path_name} = {path_value}")
                    print(f"  ✗ {path_name}: NOT FOUND")
            else:
                # Validate JSON format for calibration files
                if path_value.endswith('.json'):
                    try:
                        with open(path_value, 'r') as f:
                            calib_data = json.load(f)
                        print(f"  ✓ {path_name}: VALID JSON ({len(calib_data)} entries)")
                    except json.JSONDecodeError as e:
                        self.errors.append(f"Invalid JSON in calibration file: {path_name}")
                        print(f"  ✗ {path_name}: INVALID JSON - {e}")
                elif path_value.endswith('.yaml') or path_value.endswith('.yml'):
                    try:
                        with open(path_value, 'r') as f:
                            calib_data = yaml.safe_load(f)
                        print(f"  ✓ {path_name}: VALID YAML")
                    except yaml.YAMLError as e:
                        self.errors.append(f"Invalid YAML in calibration file: {path_name}")
                        print(f"  ✗ {path_name}: INVALID YAML - {e}")
                else:
                    print(f"  ✓ {path_name}: EXISTS")
        
        print()
        return True
    
    def _validate_output_directories(self):
        """Validate and create output directories"""
        print("📤 OUTPUT DIRECTORY VALIDATION:")
        
        output_dirs = [
            ('logging.log_dir', self.config.get('logging', {}).get('log_dir')),
            ('logging.tensorboard_dir', self.config.get('logging', {}).get('tensorboard_dir')),
            ('checkpoint.save_dir', self.config.get('checkpoint', {}).get('save_dir')),
        ]
        
        for dir_name, dir_path in output_dirs:
            if dir_path is None:
                self.warnings.append(f"Output directory not configured: {dir_name}")
                print(f"  - {dir_name}: NOT CONFIGURED")
                continue
            
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"  ✓ {dir_name}: CREATED")
                except OSError as e:
                    self.errors.append(f"Cannot create output directory: {dir_name} = {dir_path}")
                    print(f"  ✗ {dir_name}: CANNOT CREATE - {e}")
            else:
                if os.access(dir_path, os.W_OK):
                    print(f"  ✓ {dir_name}: EXISTS & WRITABLE")
                else:
                    self.errors.append(f"Output directory not writable: {dir_name}")
                    print(f"  ✗ {dir_name}: NOT WRITABLE")
        
        print()
    
    def _count_jsonl_samples(self, jsonl_path: str) -> int:
        """Count samples in JSONL file"""
        try:
            with open(jsonl_path, 'r') as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0
    
    def _print_validation_summary(self):
        """Print validation summary and exit if errors"""
        print("="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        if self.errors:
            print(f"❌ CRITICAL ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
            
            print("\n🚨 VALIDATION FAILED - CANNOT PROCEED WITH TRAINING")
            print("   Please fix all path issues before running MineSLAM")
            
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ ALL PATHS VALIDATED SUCCESSFULLY")
            print("   Ready for MineSLAM training with REAL data")
        elif not self.errors:
            print("✅ VALIDATION PASSED (with warnings)")
            print("   Ready for MineSLAM training with REAL data")
        
        print("="*80)


def validate_specific_paths(paths: List[str]) -> bool:
    """Validate specific paths provided as arguments"""
    print("Validating specific paths...")
    
    all_valid = True
    
    for path in paths:
        if os.path.exists(path):
            if os.access(path, os.R_OK):
                print(f"✓ {path}: EXISTS & READABLE")
                
                if os.path.isdir(path):
                    file_count = len([f for f in os.listdir(path) 
                                    if os.path.isfile(os.path.join(path, f))])
                    print(f"    Contains {file_count} files")
            else:
                print(f"✗ {path}: NOT READABLE")
                all_valid = False
        else:
            print(f"✗ {path}: NOT FOUND")
            all_valid = False
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description='MineSLAM Real Data Path Validator')
    parser.add_argument('--config', type=str, default='configs/mineslam.yaml',
                       help='Path to configuration file')
    parser.add_argument('--paths', nargs='*', 
                       help='Specific paths to validate (optional)')
    parser.add_argument('--exit-on-error', action='store_true', default=True,
                       help='Exit with error code if validation fails')
    
    args = parser.parse_args()
    
    try:
        if args.paths:
            # Validate specific paths
            success = validate_specific_paths(args.paths)
        else:
            # Validate full configuration
            validator = RealDataPathValidator(args.config)
            success = validator.validate_all_paths()
        
        if not success and args.exit_on_error:
            print("\n🚨 EXITING DUE TO VALIDATION ERRORS")
            sys.exit(1)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error during path validation: {e}")
        if args.exit_on_error:
            sys.exit(1)
        return 1


if __name__ == '__main__':
    sys.exit(main())