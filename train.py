#!/usr/bin/env python3
"""
MineSLAM Training Script - REAL DATA ONLY
This script STRICTLY uses real sensor data for training.
NO synthetic, random, or mock data is allowed.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import load_config, validate_real_data_config, setup_directories, print_real_data_summary
from data import create_real_data_loader, RealDataValidator
from models import MineSLAMModel
from losses import MineSLAMLoss
from metrics import PoseMetrics, DetectionMetrics, SystemMetrics
from logger import setup_logger
from seeding import set_deterministic_seed


def validate_training_environment():
    """Validate training environment before starting"""
    print("Validating training environment...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Real data training requires GPU.")
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check for critical imports
    try:
        import MinkowskiEngine
        print("✓ MinkowskiEngine available for point cloud processing")
    except ImportError:
        print("⚠ MinkowskiEngine not available - point cloud processing limited")


def create_real_data_indices(config):
    """Create data indices from real sensor data"""
    from utils import create_real_sample_index
    
    source_dir = config['data']['source_data']
    lists_dir = os.path.join(config['project']['root'], 'lists')
    os.makedirs(lists_dir, exist_ok=True)
    
    # Create train/val split from real data
    train_path = config['data']['train_index']
    val_path = config['data']['val_index']
    
    if not os.path.exists(train_path):
        print("Creating real data training index...")
        create_real_sample_index(source_dir, train_path, max_samples=800)
    
    if not os.path.exists(val_path):
        print("Creating real data validation index...")
        create_real_sample_index(source_dir, val_path, max_samples=200)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, writer):
    """Train one epoch on REAL data"""
    model.train()
    
    total_loss = 0.0
    pose_loss_total = 0.0
    detection_loss_total = 0.0
    gate_loss_total = 0.0
    
    num_batches = len(dataloader)
    
    for batch_idx, batch_data in enumerate(dataloader):
        # Validate batch contains REAL data
        for key, tensor in batch_data.items():
            if isinstance(tensor, torch.Tensor):
                from utils import RealDataChecker
                RealDataChecker.check_tensor_is_real(tensor, f"batch_{key}")
        
        # Move data to device
        inputs = {}
        targets = {}
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(device)
        
        # Prepare inputs
        if 'thermal' in batch_data:
            inputs['thermal'] = batch_data['thermal']
        if 'rgb' in batch_data:
            inputs['rgb'] = batch_data['rgb']
        
        # Prepare targets
        if 'pose' in batch_data:
            targets['pose'] = batch_data['pose']
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        total_batch_loss = loss_dict['total_loss']
        
        # Backward pass
        total_batch_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        pose_loss_total += loss_dict['pose_loss'].item()
        detection_loss_total += loss_dict['detection_loss'].item()
        gate_loss_total += loss_dict['gate_loss'].item()
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                       f"Loss: {total_batch_loss.item():.4f}")
        
        # Tensorboard logging
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar('Train/BatchLoss', total_batch_loss.item(), global_step)
    
    # Epoch averages
    avg_total_loss = total_loss / num_batches
    avg_pose_loss = pose_loss_total / num_batches
    avg_detection_loss = detection_loss_total / num_batches
    avg_gate_loss = gate_loss_total / num_batches
    
    logger.info(f"Epoch {epoch} Training - Total: {avg_total_loss:.4f}, "
               f"Pose: {avg_pose_loss:.4f}, Detection: {avg_detection_loss:.4f}, "
               f"Gate: {avg_gate_loss:.4f}")
    
    # Log to tensorboard
    writer.add_scalar('Train/EpochLoss', avg_total_loss, epoch)
    writer.add_scalar('Train/PoseLoss', avg_pose_loss, epoch)
    writer.add_scalar('Train/DetectionLoss', avg_detection_loss, epoch)
    writer.add_scalar('Train/GateLoss', avg_gate_loss, epoch)
    
    return avg_total_loss


def validate_epoch(model, dataloader, criterion, device, epoch, logger, writer):
    """Validate on REAL data"""
    model.eval()
    
    total_loss = 0.0
    pose_metrics = PoseMetrics()
    detection_metrics = DetectionMetrics()
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Validate batch contains REAL data
            for key, tensor in batch_data.items():
                if isinstance(tensor, torch.Tensor):
                    from utils import RealDataChecker
                    RealDataChecker.check_tensor_is_real(tensor, f"val_batch_{key}")
            
            # Move data to device
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(device)
            
            # Prepare inputs
            inputs = {}
            targets = {}
            
            if 'thermal' in batch_data:
                inputs['thermal'] = batch_data['thermal']
            if 'rgb' in batch_data:
                inputs['rgb'] = batch_data['rgb']
            
            if 'pose' in batch_data:
                targets['pose'] = batch_data['pose']
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            total_loss += loss_dict['total_loss'].item()
            
            # Update metrics with REAL data
            if 'pose' in outputs and 'pose' in targets:
                pose_metrics.update(outputs['pose'], targets['pose'])
    
    # Compute final metrics
    avg_loss = total_loss / len(dataloader)
    pose_results = pose_metrics.compute()
    
    logger.info(f"Epoch {epoch} Validation - Loss: {avg_loss:.4f}")
    logger.info(f"Pose Metrics - ATE: {pose_results.get('ATE', 0):.4f}, "
               f"RPE: {pose_results.get('RPE', 0):.4f}")
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/ATE', pose_results.get('ATE', 0), epoch)
    writer.add_scalar('Val/RPE', pose_results.get('RPE', 0), epoch)
    
    return avg_loss, pose_results


def save_checkpoint(model, optimizer, epoch, loss, config, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    save_dir = config['checkpoint']['save_dir']
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path}")


def main():
    parser = argparse.ArgumentParser(description='MineSLAM Training - REAL DATA ONLY')
    parser.add_argument('--config', type=str, default='configs/mineslam.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint for resuming')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    print("Loading configuration...")
    config = load_config(args.config)
    validate_real_data_config(config)
    print_real_data_summary(config)
    
    # Setup directories
    setup_directories(config)
    
    # Setup logging
    logger = setup_logger(config['logging']['log_dir'], 'train')
    logger.info("Starting MineSLAM training with REAL data")
    
    # Set deterministic seed
    set_deterministic_seed(42)
    
    # Validate environment
    validate_training_environment()
    
    # Create data indices if needed
    create_real_data_indices(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create REAL data loaders
    logger.info("Creating real data loaders...")
    train_loader = create_real_data_loader(
        config, 'train', 
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers']
    )
    
    val_loader = create_real_data_loader(
        config, 'val',
        batch_size=config['validation']['batch_size'],
        num_workers=config['hardware']['num_workers']
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating MineSLAM model...")
    model = MineSLAMModel(config).to(device)
    
    from utils import count_parameters, get_model_size_mb
    logger.info(f"Model parameters: {count_parameters(model):,}")
    logger.info(f"Model size: {get_model_size_mb(model):.2f} MB")
    
    # Create loss and optimizer
    criterion = MineSLAMLoss(config)
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['learning_rate'],
                          weight_decay=config['training']['weight_decay'])
    
    # Setup tensorboard
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    best_val_loss = float('inf')
    num_epochs = config['training']['num_epochs']
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                               device, epoch, logger, writer)
        
        # Validation
        if epoch % config['validation']['eval_every'] == 0:
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion,
                                                 device, epoch, logger, writer)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if epoch % config['checkpoint']['save_every'] == 0 or is_best:
                save_checkpoint(model, optimizer, epoch, val_loss, config, is_best)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    logger.info("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()