#!/usr/bin/env python3
"""
MineSLAM Validation Script - REAL DATA ONLY
This script validates trained models on real sensor data.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import load_config, validate_real_data_config, RealDataChecker
from data import create_real_data_loader
from models import MineSLAMModel
from losses import MineSLAMLoss
from metrics import PoseMetrics, DetectionMetrics, SystemMetrics
from logger import setup_logger


def validate_model(model, dataloader, criterion, device, logger):
    """Validate model on real data"""
    model.eval()
    
    pose_metrics = PoseMetrics()
    detection_metrics = DetectionMetrics()
    system_metrics = SystemMetrics()
    
    total_loss = 0.0
    num_samples = 0
    
    logger.info("Starting validation on REAL data...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Validate batch contains REAL data
            for key, tensor in batch_data.items():
                if isinstance(tensor, torch.Tensor):
                    RealDataChecker.check_tensor_is_real(tensor, f"val_batch_{key}")
            
            # Move data to device
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device)
            
            batch_size = next(iter(batch_data.values())).size(0)
            num_samples += batch_size
            
            # Prepare inputs and targets
            inputs = {}
            targets = {}
            
            if 'thermal' in batch_data:
                inputs['thermal'] = batch_data['thermal']
            if 'rgb' in batch_data:
                inputs['rgb'] = batch_data['rgb']
            
            if 'pose' in batch_data:
                targets['pose'] = batch_data['pose']
            
            # Forward pass with timing
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = model(inputs)
            end_time.record()
            
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)  # milliseconds
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict['total_loss'].item()
            
            # Update metrics
            if 'pose' in outputs and 'pose' in targets:
                pose_metrics.update(outputs['pose'], targets['pose'])
            
            if 'detection' in outputs:
                detection_metrics.update(outputs['detection'], targets.get('detection'))
            
            system_metrics.update(inference_time, batch_size)
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Compute final metrics
    avg_loss = total_loss / len(dataloader)
    pose_results = pose_metrics.compute()
    detection_results = detection_metrics.compute()
    system_results = system_metrics.compute()
    
    # Log results
    logger.info("="*60)
    logger.info("VALIDATION RESULTS ON REAL DATA")
    logger.info("="*60)
    logger.info(f"Average Loss: {avg_loss:.6f}")
    logger.info(f"Total Samples: {num_samples}")
    
    logger.info("\nPOSE ESTIMATION METRICS:")
    logger.info(f"  ATE (Absolute Trajectory Error): {pose_results.get('ATE', 0):.6f} m")
    logger.info(f"  RPE (Relative Pose Error): {pose_results.get('RPE', 0):.6f} %")
    
    logger.info("\nDETECTION METRICS:")
    logger.info(f"  3D mAP@0.5: {detection_results.get('mAP@0.5', 0):.4f}")
    logger.info(f"  3D mAP@0.7: {detection_results.get('mAP@0.7', 0):.4f}")
    
    logger.info("\nSYSTEM PERFORMANCE:")
    logger.info(f"  Average FPS: {system_results.get('FPS', 0):.2f}")
    logger.info(f"  Average Inference Time: {system_results.get('avg_inference_time', 0):.2f} ms")
    logger.info(f"  GPU Memory Used: {system_results.get('gpu_memory_mb', 0):.1f} MB")
    
    return {
        'loss': avg_loss,
        'pose_metrics': pose_results,
        'detection_metrics': detection_results,
        'system_metrics': system_results
    }


def main():
    parser = argparse.ArgumentParser(description='MineSLAM Validation - REAL DATA ONLY')
    parser.add_argument('--config', type=str, default='configs/mineslam.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Data split to validate on')
    parser.add_argument('--output', type=str, default='validation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    validate_real_data_config(config)
    
    # Setup logging
    logger = setup_logger(config['logging']['log_dir'], 'validation')
    logger.info("Starting MineSLAM validation on REAL data")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create REAL data loader
    logger.info(f"Loading REAL data for {args.split} split...")
    dataloader = create_real_data_loader(
        config, args.split,
        batch_size=config['validation']['batch_size'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Load model
    logger.info("Loading MineSLAM model...")
    model = MineSLAMModel(config).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create loss function
    criterion = MineSLAMLoss(config)
    
    # Run validation
    results = validate_model(model, dataloader, criterion, device, logger)
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {args.output}")
    logger.info("Validation completed!")


if __name__ == '__main__':
    main()