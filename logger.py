"""
MineSLAM Logger Module
Centralized logging for training, validation, and inference
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
import time
from datetime import datetime


def setup_logger(log_dir: str, 
                 logger_name: str, 
                 level: str = 'INFO',
                 console_output: bool = True) -> logging.Logger:
    """
    Setup logger for MineSLAM training/inference
    
    Args:
        log_dir: Directory to save log files
        logger_name: Name of the logger (e.g., 'train', 'val', 'inference')
        level: Logging level
        console_output: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(f'MineSLAM.{logger_name}')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler for detailed logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{logger_name}_{timestamp}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for important messages
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # Add initial log entry
    logger.info(f"Logger '{logger_name}' initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {level}")
    
    return logger


class TrainingLogger:
    """Specialized logger for training progress"""
    
    def __init__(self, log_dir: str, logger_name: str = 'training'):
        self.logger = setup_logger(log_dir, logger_name)
        self.start_time = time.time()
        self.epoch_start_time = None
        
        # Training statistics
        self.epoch_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def log_training_start(self, config: dict, model_info: dict):
        """Log training initialization"""
        self.logger.info("="*80)
        self.logger.info("STARTING MINESLAM TRAINING - REAL DATA ONLY")
        self.logger.info("="*80)
        
        # Log configuration summary
        self.logger.info("Training Configuration:")
        self.logger.info(f"  Batch Size: {config['training']['batch_size']}")
        self.logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
        self.logger.info(f"  Epochs: {config['training']['num_epochs']}")
        self.logger.info(f"  Device: {config['hardware']['device']}")
        
        # Log model information
        self.logger.info("Model Information:")
        self.logger.info(f"  Parameters: {model_info.get('num_params', 'Unknown'):,}")
        self.logger.info(f"  Model Size: {model_info.get('size_mb', 'Unknown'):.2f} MB")
        
        # Log data paths validation
        self.logger.info("Real Data Validation:")
        self.logger.info(f"  Source Data: {config['data']['source_data']}")
        self.logger.info(f"  Train Index: {config['data']['train_index']}")
        self.logger.info(f"  Val Index: {config['data']['val_index']}")
        
        self.logger.info("Training started with REAL sensor data only!")
    
    def log_epoch_start(self, epoch: int, num_epochs: int):
        """Log start of training epoch"""
        self.epoch_start_time = time.time()
        self.logger.info(f"\nEpoch {epoch+1}/{num_epochs} started")
    
    def log_batch_progress(self, epoch: int, batch_idx: int, num_batches: int, 
                          loss: float, lr: float):
        """Log batch training progress"""
        if batch_idx % 50 == 0 or batch_idx == num_batches - 1:
            progress = (batch_idx + 1) / num_batches * 100
            self.logger.info(
                f"Epoch {epoch+1} [{batch_idx+1:4d}/{num_batches:4d}] "
                f"({progress:5.1f}%) - Loss: {loss:.6f}, LR: {lr:.2e}"
            )
    
    def log_epoch_end(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                      val_metrics: Optional[dict] = None):
        """Log end of training epoch"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        self.epoch_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        self.logger.info(f"  Train Loss: {train_loss:.6f}")
        
        if val_loss is not None:
            self.logger.info(f"  Val Loss: {val_loss:.6f}")
        
        if val_metrics:
            self.logger.info("  Validation Metrics:")
            for metric_name, value in val_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {metric_name}: {value:.6f}")
        
        # Log best metrics so far
        if len(self.epoch_losses) > 0:
            best_train_loss = min(self.epoch_losses)
            self.logger.info(f"  Best Train Loss: {best_train_loss:.6f}")
        
        if len(self.val_losses) > 0:
            best_val_loss = min(self.val_losses)
            self.logger.info(f"  Best Val Loss: {best_val_loss:.6f}")
    
    def log_checkpoint_saved(self, epoch: int, checkpoint_path: str, is_best: bool = False):
        """Log checkpoint saving"""
        checkpoint_type = "BEST MODEL" if is_best else "checkpoint"
        self.logger.info(f"✓ Saved {checkpoint_type} at epoch {epoch+1}: {checkpoint_path}")
    
    def log_training_end(self, total_epochs: int):
        """Log training completion"""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        self.logger.info("="*80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total Epochs: {total_epochs}")
        self.logger.info(f"Total Time: {hours}h {minutes}m")
        
        if self.epoch_losses:
            final_train_loss = self.epoch_losses[-1]
            best_train_loss = min(self.epoch_losses)
            self.logger.info(f"Final Train Loss: {final_train_loss:.6f}")
            self.logger.info(f"Best Train Loss: {best_train_loss:.6f}")
        
        if self.val_losses:
            final_val_loss = self.val_losses[-1]
            best_val_loss = min(self.val_losses)
            self.logger.info(f"Final Val Loss: {final_val_loss:.6f}")
            self.logger.info(f"Best Val Loss: {best_val_loss:.6f}")


class InferenceLogger:
    """Specialized logger for inference"""
    
    def __init__(self, log_dir: str, logger_name: str = 'inference'):
        self.logger = setup_logger(log_dir, logger_name)
        self.start_time = time.time()
        self.sample_count = 0
        self.inference_times = []
    
    def log_inference_start(self, config: dict, data_path: str):
        """Log inference initialization"""
        self.logger.info("="*80)
        self.logger.info("STARTING MINESLAM INFERENCE - REAL DATA ONLY")
        self.logger.info("="*80)
        
        self.logger.info(f"Model Config: {config['project']['name']}")
        self.logger.info(f"Real Data Path: {data_path}")
        self.logger.info(f"Device: {config['hardware']['device']}")
    
    def log_sample_processed(self, sample_id: int, inference_time_ms: float,
                           pose_pred: dict, detections: list):
        """Log processing of individual sample"""
        self.sample_count += 1
        self.inference_times.append(inference_time_ms)
        
        if sample_id % 10 == 0:  # Log every 10th sample
            avg_time = sum(self.inference_times) / len(self.inference_times)
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            
            self.logger.info(f"Processed sample {sample_id}: {inference_time_ms:.2f}ms")
            self.logger.info(f"  Pose: {pose_pred}")
            self.logger.info(f"  Detections: {len(detections)} objects")
            self.logger.info(f"  Running avg: {avg_time:.2f}ms, FPS: {fps:.2f}")
    
    def log_inference_end(self, total_samples: int, results_path: str):
        """Log inference completion"""
        total_time = time.time() - self.start_time
        
        if self.inference_times:
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            avg_fps = 1000.0 / avg_inference_time
        else:
            avg_inference_time = 0
            avg_fps = 0
        
        self.logger.info("="*80)
        self.logger.info("INFERENCE COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total Samples: {total_samples}")
        self.logger.info(f"Total Time: {total_time:.2f}s")
        self.logger.info(f"Average Inference Time: {avg_inference_time:.2f}ms")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Results saved to: {results_path}")


class ValidationLogger:
    """Specialized logger for validation"""
    
    def __init__(self, log_dir: str, logger_name: str = 'validation'):
        self.logger = setup_logger(log_dir, logger_name)
        self.start_time = time.time()
    
    def log_validation_start(self, split: str, num_samples: int):
        """Log validation start"""
        self.logger.info("="*60)
        self.logger.info(f"STARTING VALIDATION ON {split.upper()} SPLIT - REAL DATA ONLY")
        self.logger.info("="*60)
        self.logger.info(f"Number of samples: {num_samples}")
    
    def log_batch_progress(self, batch_idx: int, num_batches: int):
        """Log validation batch progress"""
        if batch_idx % 20 == 0 or batch_idx == num_batches - 1:
            progress = (batch_idx + 1) / num_batches * 100
            self.logger.info(f"Validation progress: [{batch_idx+1:4d}/{num_batches:4d}] ({progress:5.1f}%)")
    
    def log_validation_results(self, results: dict):
        """Log final validation results"""
        total_time = time.time() - self.start_time
        
        self.logger.info("="*60)
        self.logger.info("VALIDATION RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Validation Time: {total_time:.2f}s")
        self.logger.info(f"Average Loss: {results.get('loss', 0):.6f}")
        
        # Log pose metrics
        pose_metrics = results.get('pose_metrics', {})
        if pose_metrics:
            self.logger.info("\nPose Estimation Metrics:")
            for metric, value in pose_metrics.items():
                self.logger.info(f"  {metric}: {value:.6f}")
        
        # Log detection metrics
        detection_metrics = results.get('detection_metrics', {})
        if detection_metrics:
            self.logger.info("\nDetection Metrics:")
            for metric, value in detection_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
        
        # Log system metrics
        system_metrics = results.get('system_metrics', {})
        if system_metrics:
            self.logger.info("\nSystem Performance:")
            for metric, value in system_metrics.items():
                if 'time' in metric.lower():
                    self.logger.info(f"  {metric}: {value:.2f} ms")
                elif 'fps' in metric.lower():
                    self.logger.info(f"  {metric}: {value:.2f}")
                elif 'memory' in metric.lower():
                    self.logger.info(f"  {metric}: {value:.1f} MB")
                else:
                    self.logger.info(f"  {metric}: {value}")


def create_logger(log_type: str, log_dir: str, **kwargs) -> logging.Logger:
    """
    Factory function to create appropriate logger
    
    Args:
        log_type: Type of logger ('train', 'val', 'inference', 'general')
        log_dir: Directory for log files
        **kwargs: Additional arguments for specific logger types
    
    Returns:
        Configured logger
    """
    if log_type == 'train':
        return TrainingLogger(log_dir, 'training')
    elif log_type == 'val':
        return ValidationLogger(log_dir, 'validation')
    elif log_type == 'inference':
        return InferenceLogger(log_dir, 'inference')
    else:
        return setup_logger(log_dir, log_type, **kwargs)