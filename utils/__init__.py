"""
MineSLAM Utilities Module
Configuration loading, transformations, and helper functions
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_real_data_config(config: Dict[str, Any]) -> bool:
    """
    Validate that configuration points to real data paths only
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises exception otherwise
    """
    # Check anti-synthetic flags
    validation_settings = config.get('validation_settings', {})
    
    if validation_settings.get('allow_synthetic', True):
        raise ValueError("Configuration allows synthetic data! Set allow_synthetic: false")
    
    if validation_settings.get('allow_mock', True):
        raise ValueError("Configuration allows mock data! Set allow_mock: false")
    
    if not validation_settings.get('strict_path_check', False):
        raise ValueError("Configuration must enable strict_path_check: true")
    
    # Validate critical real data paths exist
    critical_paths = [
        config['data']['source_data'],
        config['data']['images']['thermal'],
        config['data']['images']['rgb_left'],
        config['data']['ground_truth']['trajectory']
    ]
    
    for path in critical_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical real data path missing: {path}")
    
    print("✓ Configuration validated - ALL paths point to REAL data")
    return True


def setup_directories(config: Dict[str, Any]) -> None:
    """
    Setup output directories for logging, checkpoints, etc.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['logging']['log_dir'],
        config['logging']['tensorboard_dir'],
        config['checkpoint']['save_dir'],
        config['data']['root']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory ready: {directory}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


class RealDataTransforms:
    """
    Data transforms that only work with REAL data
    NO synthetic data generation allowed
    """
    
    @staticmethod
    def normalize_image(image: torch.Tensor, 
                       mean: List[float] = [0.485, 0.456, 0.406],
                       std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
        """Normalize REAL image with ImageNet stats"""
        if image.dim() == 3:  # Single image
            for i in range(image.size(0)):
                image[i] = (image[i] - mean[i % len(mean)]) / std[i % len(std)]
        return image
    
    @staticmethod
    def resize_image(image: torch.Tensor, target_size: List[int]) -> torch.Tensor:
        """Resize REAL image to target size"""
        import torch.nn.functional as F
        
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dim
        
        resized = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
        
        if resized.size(0) == 1:
            resized = resized.squeeze(0)  # Remove batch dim
        
        return resized
    
    @staticmethod
    def augment_real_image(image: torch.Tensor, 
                          brightness: float = 0.2,
                          contrast: float = 0.2) -> torch.Tensor:
        """
        Apply augmentation to REAL images only
        NO synthetic/artificial generation
        """
        # Only modify existing real pixel values
        if np.random.random() < 0.5:
            # Brightness adjustment
            brightness_factor = 1.0 + np.random.uniform(-brightness, brightness)
            image = torch.clamp(image * brightness_factor, 0.0, 1.0)
        
        if np.random.random() < 0.5:
            # Contrast adjustment
            contrast_factor = 1.0 + np.random.uniform(-contrast, contrast)
            mean = image.mean()
            image = torch.clamp((image - mean) * contrast_factor + mean, 0.0, 1.0)
        
        return image
    
    @staticmethod
    def add_real_noise_to_pointcloud(points: torch.Tensor, 
                                    noise_std: float = 0.01) -> torch.Tensor:
        """
        Add realistic noise to REAL point cloud data
        Simulates real sensor noise characteristics
        """
        if noise_std > 0:
            noise = torch.randn_like(points) * noise_std
            points = points + noise
        
        return points


class RealDataChecker:
    """Utility class to verify data is real (not synthetic/mock)"""
    
    @staticmethod
    def check_tensor_is_real(tensor: torch.Tensor, name: str = "") -> bool:
        """
        Check if tensor contains real data characteristics
        
        Args:
            tensor: Input tensor to check
            name: Name for error messages
        
        Returns:
            True if passes checks
        """
        # Check for obvious synthetic patterns
        if torch.all(tensor == tensor[0]):
            raise ValueError(f"Tensor '{name}' appears synthetic - all values identical")
        
        # Check for NaN/Inf values
        if torch.any(torch.isnan(tensor)):
            raise ValueError(f"Tensor '{name}' contains NaN values")
        
        if torch.any(torch.isinf(tensor)):
            raise ValueError(f"Tensor '{name}' contains infinite values")
        
        # Check data range is reasonable for sensor data
        if tensor.dtype == torch.float32:
            if torch.min(tensor) < -1000 or torch.max(tensor) > 1000:
                print(f"Warning: Tensor '{name}' has unusual value range [{torch.min(tensor):.3f}, {torch.max(tensor):.3f}]")
        
        return True
    
    @staticmethod
    def verify_real_file_timestamps(file_list: List[str]) -> bool:
        """
        Verify files have realistic timestamps
        Real sensor data should have consistent temporal patterns
        """
        timestamps = []
        for file_path in file_list:
            filename = os.path.basename(file_path)
            try:
                # Extract timestamp from filename (assuming format: timestamp.ext)
                timestamp = float(filename.split('.')[0])
                timestamps.append(timestamp)
            except ValueError:
                continue  # Skip files without timestamp format
        
        if len(timestamps) < 2:
            return True  # Can't verify with insufficient data
        
        timestamps.sort()
        time_diffs = np.diff(timestamps)
        
        # Check for realistic time intervals (should be > 0 and not too regular)
        if np.any(time_diffs <= 0):
            raise ValueError("Files have invalid temporal ordering")
        
        # Real sensor data should have some variance in timing
        if len(set(time_diffs)) == 1:
            print("Warning: File timestamps are perfectly regular (may be synthetic)")
        
        return True


def create_real_sample_index(source_dir: str, 
                           output_path: str,
                           max_samples: int = 1000) -> None:
    """
    Create sample index from real data directory
    
    Args:
        source_dir: Directory containing real sensor data
        output_path: Output path for sample index
        max_samples: Maximum number of samples to index
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source data directory not found: {source_dir}")
    
    print(f"Creating real data index from {source_dir}...")
    
    # Find real image files
    image_dirs = {
        'thermal': os.path.join(source_dir, 'images', 'boson_thermal'),
        'rgb_left': os.path.join(source_dir, 'images', 'multisense_left_color'),
        'rgb_right': os.path.join(source_dir, 'images', 'multisense_right')
    }
    
    samples = []
    
    # Get thermal images as reference
    thermal_dir = image_dirs['thermal']
    if os.path.exists(thermal_dir):
        thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.endswith('.png')])
        
        for i, thermal_file in enumerate(thermal_files[:max_samples]):
            timestamp = thermal_file.split('.')[0]
            
            sample = {
                'timestamp': float(timestamp),
                'thermal_image': os.path.join(thermal_dir, thermal_file),
                'rgb_left_image': os.path.join(image_dirs['rgb_left'], f"{timestamp}.png"),
                'rgb_right_image': os.path.join(image_dirs['rgb_right'], f"{timestamp}.png")
            }
            
            # Verify files exist
            if all(os.path.exists(sample[key]) for key in ['thermal_image', 'rgb_left_image']):
                samples.append(sample)
    
    # Save index
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Created index with {len(samples)} real samples: {output_path}")


def print_real_data_summary(config: Dict[str, Any]) -> None:
    """Print summary of real data configuration"""
    print("\n" + "="*60)
    print("MineSLAM REAL DATA CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Project: {config['project']['name']}")
    print(f"Root: {config['project']['root']}")
    
    print("\nREAL DATA SOURCES:")
    print(f"  Source Dataset: {config['data']['source_data']}")
    print(f"  Thermal Images: {config['data']['images']['thermal']}")
    print(f"  RGB Images: {config['data']['images']['rgb_left']}")
    print(f"  Point Clouds: {config['data']['pointclouds']['ouster']}")
    print(f"  Ground Truth: {config['data']['ground_truth']['trajectory']}")
    
    print("\nVALIDATION SETTINGS:")
    validation = config.get('validation_settings', {})
    print(f"  Synthetic Data: {'FORBIDDEN' if not validation.get('allow_synthetic', True) else 'ALLOWED'}")
    print(f"  Mock Data: {'FORBIDDEN' if not validation.get('allow_mock', True) else 'ALLOWED'}")
    print(f"  Strict Path Check: {'ENABLED' if validation.get('strict_path_check', False) else 'DISABLED'}")
    
    print("="*60)