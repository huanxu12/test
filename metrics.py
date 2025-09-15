"""
MineSLAM Metrics Module
Implements ATE, RPE, 3D mAP, and FPS metrics for real data evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from scipy.spatial.transform import Rotation


class PoseMetrics:
    """Pose estimation metrics: ATE and RPE"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.predicted_poses = []
        self.ground_truth_poses = []
        self.timestamps = []
    
    def update(self, pred_poses: torch.Tensor, gt_poses: torch.Tensor, 
               timestamps: Optional[torch.Tensor] = None):
        """
        Update metrics with batch of poses
        
        Args:
            pred_poses: Predicted poses [B, 6] (x, y, z, roll, pitch, yaw)
            gt_poses: Ground truth poses [B, 6]
            timestamps: Optional timestamps [B]
        """
        pred_poses_np = pred_poses.detach().cpu().numpy()
        gt_poses_np = gt_poses.detach().cpu().numpy()
        
        self.predicted_poses.extend(pred_poses_np)
        self.ground_truth_poses.extend(gt_poses_np)
        
        if timestamps is not None:
            timestamps_np = timestamps.detach().cpu().numpy()
            self.timestamps.extend(timestamps_np)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics
        
        Returns:
            Dictionary containing ATE and RPE values
        """
        if len(self.predicted_poses) == 0:
            return {'ATE': 0.0, 'RPE': 0.0}
        
        pred_poses = np.array(self.predicted_poses)
        gt_poses = np.array(self.ground_truth_poses)
        
        # Compute ATE (Absolute Trajectory Error)
        ate = self._compute_ate(pred_poses, gt_poses)
        
        # Compute RPE (Relative Pose Error)
        rpe = self._compute_rpe(pred_poses, gt_poses)
        
        return {
            'ATE': float(ate),
            'RPE': float(rpe),
            'num_poses': len(self.predicted_poses)
        }
    
    def _compute_ate(self, pred_poses: np.ndarray, gt_poses: np.ndarray) -> float:
        """
        Compute Absolute Trajectory Error
        
        Args:
            pred_poses: Predicted poses [N, 6]
            gt_poses: Ground truth poses [N, 6]
        
        Returns:
            ATE in meters
        """
        # Extract positions
        pred_positions = pred_poses[:, :3]  # x, y, z
        gt_positions = gt_poses[:, :3]
        
        # Compute Euclidean distances
        position_errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
        
        # Return RMSE of position errors
        ate = np.sqrt(np.mean(position_errors ** 2))
        
        return ate
    
    def _compute_rpe(self, pred_poses: np.ndarray, gt_poses: np.ndarray, 
                     delta: int = 1) -> float:
        """
        Compute Relative Pose Error
        
        Args:
            pred_poses: Predicted poses [N, 6]
            gt_poses: Ground truth poses [N, 6]
            delta: Frame delta for relative pose computation
        
        Returns:
            RPE as percentage
        """
        if len(pred_poses) <= delta:
            return 0.0
        
        relative_errors = []
        
        for i in range(len(pred_poses) - delta):
            # Compute relative transformations
            pred_rel = self._compute_relative_pose(pred_poses[i], pred_poses[i + delta])
            gt_rel = self._compute_relative_pose(gt_poses[i], gt_poses[i + delta])
            
            # Compute error in relative transformation
            position_error = np.linalg.norm(pred_rel[:3] - gt_rel[:3])
            relative_errors.append(position_error)
        
        if len(relative_errors) == 0:
            return 0.0
        
        # Return RMSE as percentage
        rpe = np.sqrt(np.mean(np.array(relative_errors) ** 2))
        
        return rpe * 100  # Convert to percentage
    
    def _compute_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """Compute relative pose between two absolute poses"""
        # Extract positions and orientations
        pos1, rot1 = pose1[:3], pose1[3:]
        pos2, rot2 = pose2[:3], pose2[3:]
        
        # Convert Euler angles to rotation matrices
        R1 = Rotation.from_euler('xyz', rot1).as_matrix()
        R2 = Rotation.from_euler('xyz', rot2).as_matrix()
        
        # Compute relative transformation
        R_rel = R2 @ R1.T
        t_rel = pos2 - pos1
        
        # Convert back to Euler angles
        rot_rel = Rotation.from_matrix(R_rel).as_euler('xyz')
        
        return np.concatenate([t_rel, rot_rel])


class DetectionMetrics:
    """3D object detection metrics: mAP, precision, recall"""
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.7]):
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        """Reset accumulated detections"""
        self.predictions = []
        self.ground_truths = []
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               ground_truths: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update metrics with batch of detections
        
        Args:
            predictions: Model predictions containing bboxes, classes, confidence
            ground_truths: Ground truth detections (if available)
        """
        if ground_truths is None:
            # For inference without ground truth, just accumulate predictions
            self.predictions.append(predictions)
            return
        
        # Process predictions and ground truths
        batch_size = predictions['class_logits'].size(0)
        
        for b in range(batch_size):
            pred_dict = {
                'bboxes': predictions['bbox_pred'][b],
                'scores': torch.max(torch.softmax(predictions['class_logits'][b], dim=-1), dim=-1)[0],
                'labels': torch.argmax(predictions['class_logits'][b], dim=-1)
            }
            
            self.predictions.append(pred_dict)
            
            if ground_truths is not None:
                gt_dict = {
                    'bboxes': ground_truths.get('bboxes', torch.empty(0, 6)),
                    'labels': ground_truths.get('labels', torch.empty(0, dtype=torch.long))
                }
                self.ground_truths.append(gt_dict)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute 3D detection metrics
        
        Returns:
            Dictionary containing mAP values
        """
        if len(self.predictions) == 0:
            return {f'mAP@{th}': 0.0 for th in self.iou_thresholds}
        
        # If no ground truths, return dummy metrics
        if len(self.ground_truths) == 0:
            return {
                f'mAP@{th}': 0.0 for th in self.iou_thresholds
            }
        
        results = {}
        
        # Compute mAP for each IoU threshold
        for iou_th in self.iou_thresholds:
            map_value = self._compute_map_3d(iou_th)
            results[f'mAP@{iou_th}'] = float(map_value)
        
        # Overall 3D mAP (average across thresholds)
        results['mAP_3D'] = float(np.mean(list(results.values())))
        
        return results
    
    def _compute_map_3d(self, iou_threshold: float) -> float:
        """
        Compute 3D mean Average Precision
        
        Args:
            iou_threshold: IoU threshold for positive detection
        
        Returns:
            mAP value
        """
        # Simplified mAP computation
        # In practice, you'd implement proper AP calculation with precision-recall curves
        
        total_tp = 0
        total_fp = 0
        total_gt = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if len(gt['bboxes']) == 0:
                continue
            
            # Compute IoU between predictions and ground truths
            ious = self._compute_iou_3d(pred['bboxes'], gt['bboxes'])
            
            # Find matches above threshold
            max_ious, matches = torch.max(ious, dim=1)
            tp = torch.sum(max_ious > iou_threshold).item()
            
            total_tp += tp
            total_fp += len(pred['bboxes']) - tp
            total_gt += len(gt['bboxes'])
        
        if total_gt == 0:
            return 0.0
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt
        
        # Simplified AP (should be proper precision-recall curve integration)
        ap = precision * recall
        
        return ap
    
    def _compute_iou_3d(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D IoU between two sets of bounding boxes
        
        Args:
            boxes1: First set of boxes [N, 6] (x, y, z, w, h, l)
            boxes2: Second set of boxes [M, 6]
        
        Returns:
            IoU matrix [N, M]
        """
        # Simplified 3D IoU computation
        # In practice, you'd implement proper 3D box intersection computation
        
        N, M = boxes1.size(0), boxes2.size(0)
        ious = torch.zeros(N, M)
        
        for i in range(N):
            for j in range(M):
                # Compute center distance as simple proxy for IoU
                center1 = boxes1[i, :3]
                center2 = boxes2[j, :3]
                
                distance = torch.norm(center1 - center2)
                
                # Convert distance to IoU-like score (simplified)
                iou = 1.0 / (1.0 + distance)
                ious[i, j] = iou
        
        return ious


class SystemMetrics:
    """System performance metrics: FPS, inference time, GPU memory"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.inference_times = []
        self.batch_sizes = []
        self.gpu_memory_usage = []
    
    def update(self, inference_time_ms: float, batch_size: int):
        """
        Update system metrics
        
        Args:
            inference_time_ms: Inference time in milliseconds
            batch_size: Batch size processed
        """
        self.inference_times.append(inference_time_ms)
        self.batch_sizes.append(batch_size)
        
        # Record GPU memory usage if available
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            self.gpu_memory_usage.append(gpu_memory_mb)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute system performance metrics
        
        Returns:
            Dictionary containing FPS, inference time, memory usage
        """
        if len(self.inference_times) == 0:
            return {
                'FPS': 0.0,
                'avg_inference_time': 0.0,
                'gpu_memory_mb': 0.0
            }
        
        inference_times = np.array(self.inference_times)
        batch_sizes = np.array(self.batch_sizes)
        
        # Compute FPS
        avg_inference_time_ms = np.mean(inference_times)
        avg_batch_size = np.mean(batch_sizes)
        fps = (avg_batch_size * 1000) / avg_inference_time_ms
        
        results = {
            'FPS': float(fps),
            'avg_inference_time': float(avg_inference_time_ms),
            'min_inference_time': float(np.min(inference_times)),
            'max_inference_time': float(np.max(inference_times)),
            'std_inference_time': float(np.std(inference_times)),
        }
        
        # Add GPU memory usage if available
        if self.gpu_memory_usage:
            results['gpu_memory_mb'] = float(np.mean(self.gpu_memory_usage))
            results['max_gpu_memory_mb'] = float(np.max(self.gpu_memory_usage))
        else:
            results['gpu_memory_mb'] = 0.0
        
        return results


class MetricsLogger:
    """Utility class for logging and tracking metrics"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics_history = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], split: str = 'train'):
        """Log metrics for an epoch"""
        timestamp = time.time()
        
        log_entry = {
            'epoch': epoch,
            'timestamp': timestamp,
            'split': split,
            'metrics': metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # Save to file
        import json
        metrics_file = os.path.join(self.log_dir, f'{split}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_best_metrics(self, metric_name: str, split: str = 'val') -> Tuple[float, int]:
        """Get best value and epoch for a specific metric"""
        split_metrics = [m for m in self.metrics_history if m['split'] == split]
        
        if not split_metrics:
            return 0.0, 0
        
        values = [m['metrics'].get(metric_name, 0.0) for m in split_metrics]
        epochs = [m['epoch'] for m in split_metrics]
        
        if metric_name in ['ATE', 'RPE']:  # Lower is better
            best_idx = np.argmin(values)
        else:  # Higher is better (mAP, FPS)
            best_idx = np.argmax(values)
        
        return values[best_idx], epochs[best_idx]