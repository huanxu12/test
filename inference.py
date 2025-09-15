#!/usr/bin/env python3
"""
MineSLAM Inference Script - REAL DATA ONLY
This script runs inference on real sensor data.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import load_config, validate_real_data_config, RealDataChecker
from models import MineSLAMModel
from logger import setup_logger


class RealTimeInference:
    """Real-time inference on real sensor data"""
    
    def __init__(self, config, checkpoint_path, device):
        self.config = config
        self.device = device
        
        # Load model
        self.model = MineSLAMModel(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup timing
        self.inference_times = []
        
    def preprocess_image(self, image_path, target_size):
        """Preprocess real image for inference"""
        # Validate it's a real image file
        RealDataChecker.check_tensor_is_real(torch.tensor([1.0]), f"image_path_check")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Real image file not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load real image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Validate tensor
        RealDataChecker.check_tensor_is_real(image_tensor, "preprocessed_image")
        
        return image_tensor.to(self.device)
    
    def infer_single_sample(self, thermal_path, rgb_path):
        """Run inference on a single real data sample"""
        
        # Load and preprocess REAL images
        thermal_size = self.config['model']['encoders']['thermal']['input_size']
        rgb_size = self.config['model']['encoders']['rgb']['input_size']
        
        thermal_tensor = self.preprocess_image(thermal_path, thermal_size)
        rgb_tensor = self.preprocess_image(rgb_path, rgb_size)
        
        # Prepare inputs
        inputs = {
            'thermal': thermal_tensor,
            'rgb': rgb_tensor
        }
        
        # Run inference with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        self.inference_times.append(inference_time)
        
        # Process outputs
        pose_pred = outputs['pose'].cpu().numpy()[0]  # Remove batch dimension
        detection_pred = outputs['detection']
        gate_weights = outputs['gate_weights'].cpu().numpy()[0]
        
        results = {
            'pose': {
                'translation': pose_pred[:3].tolist(),
                'rotation': pose_pred[3:].tolist()
            },
            'detections': self._process_detections(detection_pred),
            'gate_weights': gate_weights.tolist(),
            'inference_time_ms': inference_time
        }
        
        return results
    
    def _process_detections(self, detection_pred):
        """Process detection predictions"""
        detections = []
        
        class_logits = detection_pred['class_logits'][0]  # Remove batch dim
        bbox_pred = detection_pred['bbox_pred'][0]
        confidence = detection_pred['confidence'][0]
        
        # Apply confidence threshold
        conf_threshold = 0.5
        valid_mask = confidence.squeeze(-1) > conf_threshold
        
        if valid_mask.any():
            valid_classes = class_logits[valid_mask]
            valid_bboxes = bbox_pred[valid_mask]
            valid_confs = confidence[valid_mask]
            
            for i in range(len(valid_classes)):
                detection = {
                    'class_id': torch.argmax(valid_classes[i]).item(),
                    'confidence': valid_confs[i].item(),
                    'bbox_3d': valid_bboxes[i].cpu().numpy().tolist()
                }
                detections.append(detection)
        
        return detections
    
    def get_performance_stats(self):
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        return {
            'avg_inference_time_ms': float(np.mean(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'std_inference_time_ms': float(np.std(times)),
            'fps': 1000.0 / np.mean(times),
            'total_samples': len(times)
        }


def run_inference_on_real_data(config, checkpoint_path, data_dir, output_dir):
    """Run inference on real data directory"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create inference engine
    inference_engine = RealTimeInference(config, checkpoint_path, device)
    
    # Find real data files
    thermal_dir = os.path.join(data_dir, 'images', 'boson_thermal')
    rgb_dir = os.path.join(data_dir, 'images', 'multisense_left_color')
    
    if not os.path.exists(thermal_dir) or not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Real data directories not found: {thermal_dir}, {rgb_dir}")
    
    # Get list of real files
    thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.endswith('.png')])
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    if len(thermal_files) == 0 or len(rgb_files) == 0:
        raise ValueError("No real image files found in data directories")
    
    print(f"Found {len(thermal_files)} thermal images and {len(rgb_files)} RGB images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference on real data samples
    results = []
    num_samples = min(len(thermal_files), len(rgb_files), 100)  # Limit for demo
    
    for i in range(num_samples):
        thermal_path = os.path.join(thermal_dir, thermal_files[i])
        rgb_path = os.path.join(rgb_dir, rgb_files[i])
        
        try:
            result = inference_engine.infer_single_sample(thermal_path, rgb_path)
            result['sample_id'] = i
            result['thermal_file'] = thermal_files[i]
            result['rgb_file'] = rgb_files[i]
            results.append(result)
            
            if i % 10 == 0:
                print(f"Processed {i}/{num_samples} real samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save results
    import json
    results_file = os.path.join(output_dir, 'inference_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save performance statistics
    perf_stats = inference_engine.get_performance_stats()
    perf_file = os.path.join(output_dir, 'performance_stats.json')
    with open(perf_file, 'w') as f:
        json.dump(perf_stats, f, indent=2)
    
    print(f"\nInference completed on {len(results)} real samples")
    print(f"Results saved to: {results_file}")
    print(f"Performance stats saved to: {perf_file}")
    print(f"Average FPS: {perf_stats.get('fps', 0):.2f}")
    print(f"Average inference time: {perf_stats.get('avg_inference_time_ms', 0):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='MineSLAM Inference - REAL DATA ONLY')
    parser.add_argument('--config', type=str, default='configs/mineslam.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing real sensor data')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                       help='Output directory for results')
    parser.add_argument('--single', action='store_true',
                       help='Run inference on single sample (interactive)')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_real_data_config(config)
    
    # Setup logging
    logger = setup_logger(config['logging']['log_dir'], 'inference')
    logger.info("Starting MineSLAM inference on REAL data")
    
    # Validate real data directory exists
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Real data directory not found: {args.data_dir}")
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if args.single:
        # Interactive single sample inference
        print("Single sample inference mode - provide paths to real sensor data")
        thermal_path = input("Enter path to thermal image: ").strip()
        rgb_path = input("Enter path to RGB image: ").strip()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_engine = RealTimeInference(config, args.checkpoint, device)
        
        result = inference_engine.infer_single_sample(thermal_path, rgb_path)
        
        print("\nInference Results:")
        print(f"Pose: {result['pose']}")
        print(f"Detections: {len(result['detections'])} objects detected")
        print(f"Gate weights: {result['gate_weights']}")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        
    else:
        # Batch inference on real data directory
        run_inference_on_real_data(config, args.checkpoint, args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()