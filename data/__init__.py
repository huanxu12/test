"""
MineSLAM Dataset Module - REAL DATA ONLY
This module STRICTLY prohibits any synthetic/random data generation.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

class RealDataValidator:
    """Validates that all data paths point to real, existing files"""
    
    @staticmethod
    def validate_path(path: str, description: str = "") -> bool:
        """Validate a single path exists and is readable"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"REAL DATA PATH MISSING: {description} at {path}")
        
        if not os.access(path, os.R_OK):
            raise PermissionError(f"REAL DATA NOT READABLE: {description} at {path}")
        
        return True
    
    @staticmethod
    def validate_image_file(path: str) -> bool:
        """Validate image file exists and can be decoded"""
        RealDataValidator.validate_path(path, "Image file")
        
        try:
            # Try to open and verify it's a valid image
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as e:
            raise ValueError(f"INVALID REAL IMAGE FILE: {path} - {e}")
    
    @staticmethod
    def validate_json_file(path: str) -> Dict:
        """Validate JSON file exists and can be parsed"""
        RealDataValidator.validate_path(path, "JSON file")
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise ValueError(f"INVALID REAL JSON FILE: {path} - {e}")


class MineSLAMDataset(Dataset):
    """
    MineSLAM Dataset for real multi-modal sensor data
    
    STRICT POLICY: This dataset ONLY loads real sensor data.
    NO synthetic, random, or mock data is allowed.
    """
    
    def __init__(self, 
                 config: Dict,
                 split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Initialize dataset with real data paths
        
        Args:
            config: Configuration dictionary with real data paths
            split: 'train', 'val', or 'test'
            transform: Data transforms (applied to REAL data only)
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # Validate all critical paths exist
        self._validate_config_paths()
        
        # Load real data index
        self.samples = self._load_real_data_index()
        
        if len(self.samples) == 0:
            raise ValueError(f"NO REAL DATA FOUND for split '{split}'. "
                           f"Check data paths in config.")
        
        print(f"Loaded {len(self.samples)} REAL samples for {split} split")
    
    def _validate_config_paths(self):
        """Validate all configuration paths point to real data"""
        required_paths = [
            ('data.root', self.config['data']['root']),
            ('data.source_data', self.config['data']['source_data']),
            ('data.images.thermal', self.config['data']['images']['thermal']),
            ('data.images.rgb_left', self.config['data']['images']['rgb_left']),
            ('data.pointclouds.ouster', self.config['data']['pointclouds']['ouster']),
            ('data.ground_truth.trajectory', self.config['data']['ground_truth']['trajectory']),
        ]
        
        for name, path in required_paths:
            RealDataValidator.validate_path(path, f"Config path {name}")
    
    def _load_real_data_index(self) -> List[Dict]:
        """Load index of real data samples"""
        index_path = self.config['data'][f'{self.split}_index']
        
        if not os.path.exists(index_path):
            # If index doesn't exist, generate from real data
            return self._generate_real_data_index()
        
        # Load existing index and validate all entries point to real data
        samples = []
        with open(index_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Validate each sample points to real files
                if self._validate_sample_paths(sample):
                    samples.append(sample)
        
        return samples
    
    def _generate_real_data_index(self) -> List[Dict]:
        """Generate data index from real sensor data files"""
        print(f"Generating real data index for {self.split} split...")
        
        # Load ground truth trajectory
        gt_path = self.config['data']['ground_truth']['trajectory']
        RealDataValidator.validate_path(gt_path, "Ground truth trajectory")
        
        gt_df = pd.read_csv(gt_path)
        
        # Get real image timestamps
        thermal_dir = self.config['data']['images']['thermal']
        rgb_left_dir = self.config['data']['images']['rgb_left']
        
        thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.endswith('.png')])
        rgb_files = sorted([f for f in os.listdir(rgb_left_dir) if f.endswith('.png')])
        
        samples = []
        for i, (thermal_file, rgb_file) in enumerate(zip(thermal_files[:100], rgb_files[:100])):  # Limit for demo
            # Extract timestamp from filename
            timestamp = float(thermal_file.split('.')[0])
            
            # Find closest ground truth pose
            gt_idx = np.argmin(np.abs(gt_df['timestamp'].values - timestamp))
            gt_row = gt_df.iloc[gt_idx]
            
            sample = {
                'timestamp': timestamp,
                'thermal_image': os.path.join(thermal_dir, thermal_file),
                'rgb_image': os.path.join(rgb_left_dir, rgb_file),
                'pose': {
                    'position': [gt_row['position_x'], gt_row['position_y'], gt_row['position_z']],
                    'orientation': [gt_row['orientation_roll'], gt_row['orientation_pitch'], gt_row['orientation_yaw']]
                }
            }
            
            # Validate sample points to real files
            if self._validate_sample_paths(sample):
                samples.append(sample)
        
        return samples
    
    def _validate_sample_paths(self, sample: Dict) -> bool:
        """Validate all paths in a sample point to real files"""
        try:
            RealDataValidator.validate_image_file(sample['thermal_image'])
            RealDataValidator.validate_image_file(sample['rgb_image'])
            return True
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping invalid sample: {e}")
            return False
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a real data sample
        
        Returns:
            Dictionary containing real sensor data tensors
        """
        sample = self.samples[idx]
        
        # Load REAL thermal image
        thermal_img = self._load_real_image(sample['thermal_image'], expected_channels=1)
        
        # Load REAL RGB image  
        rgb_img = self._load_real_image(sample['rgb_image'], expected_channels=3)
        
        # Get REAL pose data
        pose = torch.tensor(sample['pose']['position'] + sample['pose']['orientation'], 
                          dtype=torch.float32)
        
        data = {
            'thermal': thermal_img,
            'rgb': rgb_img,
            'pose': pose,
            'timestamp': torch.tensor(sample['timestamp'], dtype=torch.float64)
        }
        
        # Apply transforms to REAL data only
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _load_real_image(self, path: str, expected_channels: int) -> torch.Tensor:
        """Load and validate a real image file"""
        RealDataValidator.validate_image_file(path)
        
        # Load with OpenCV for consistency
        if expected_channels == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load real image: {path}")
            img = np.expand_dims(img, axis=0)  # Add channel dimension
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load real image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img.astype(np.float32)) / 255.0
        
        # Validate tensor properties
        assert img_tensor.dtype == torch.float32, f"Invalid tensor dtype: {img_tensor.dtype}"
        assert img_tensor.shape[0] == expected_channels, f"Invalid channels: expected {expected_channels}, got {img_tensor.shape[0]}"
        assert not torch.any(torch.isnan(img_tensor)), "Real image contains NaN values"
        
        return img_tensor


def create_real_data_loader(config: Dict, 
                           split: str, 
                           batch_size: int,
                           num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    Create data loader for REAL data only
    
    Args:
        config: Configuration with real data paths
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        DataLoader for real MineSLAM data
    """
    dataset = MineSLAMDataset(config, split=split)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader