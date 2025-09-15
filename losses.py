"""
MineSLAM Loss Functions
Multi-task loss for pose estimation and 3D object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class MineSLAMLoss(nn.Module):
    """
    Multi-task loss for MineSLAM with Kendall Uncertainty Weighting
    Combines pose estimation loss, detection loss, and gate regularization
    with learnable uncertainty parameters (σ) for dynamic balancing
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Individual loss functions
        self.pose_loss_fn = PoseLoss()
        self.detection_loss_fn = DetectionLoss()
        self.gate_loss_fn = GateRegularizationLoss()

        # Kendall uncertainty weighting (learnable σ parameters)
        from models.kendall_uncertainty import create_kendall_uncertainty
        uncertainty_config = config.get('kendall_uncertainty', {})

        self.use_kendall = uncertainty_config.get('enabled', True)
        if self.use_kendall:
            self.kendall_module = create_kendall_uncertainty(
                uncertainty_type=uncertainty_config.get('type', 'adaptive'),
                num_tasks=3,
                init_log_var=uncertainty_config.get('init_log_var', 0.0),
                adaptation_rate=uncertainty_config.get('adaptation_rate', 0.01),
                target_balance=uncertainty_config.get('target_balance', 0.33)
            )
        else:
            # Fixed weights fallback
            self.pose_weight = config['training']['loss_weights']['pose']
            self.detection_weight = config['training']['loss_weights']['detection']
            self.gate_weight = config['training']['loss_weights']['gate']
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with Kendall uncertainty weighting

        Args:
            outputs: Model outputs containing pose, detection, gate_weights
            targets: Ground truth targets

        Returns:
            Dictionary containing individual and total losses with Kendall weights
        """
        # Compute individual task losses
        raw_losses = {}

        # Pose estimation loss
        if 'pose' in outputs and 'pose' in targets:
            pose_loss = self.pose_loss_fn(outputs['pose'], targets['pose'])
            raw_losses['pose'] = pose_loss
        else:
            raw_losses['pose'] = torch.tensor(0.0, device=outputs[list(outputs.keys())[0]].device)

        # Detection loss
        if 'detection' in outputs:
            detection_targets = targets.get('detection', None)
            detection_loss = self.detection_loss_fn(outputs['detection'], detection_targets)
            raw_losses['detection'] = detection_loss
        else:
            raw_losses['detection'] = torch.tensor(0.0, device=outputs[list(outputs.keys())[0]].device)

        # Gate regularization loss
        if 'gate_weights' in outputs:
            gate_loss = self.gate_loss_fn(outputs['gate_weights'])
            raw_losses['gate'] = gate_loss
        else:
            raw_losses['gate'] = torch.tensor(0.0, device=outputs[list(outputs.keys())[0]].device)

        # Apply Kendall uncertainty weighting
        if self.use_kendall:
            weighted_losses = self.kendall_module(raw_losses)

            # Add raw losses for monitoring
            for key, value in raw_losses.items():
                weighted_losses[f'raw_{key}_loss'] = value

            return weighted_losses
        else:
            # Fixed weighting fallback
            total_loss = (self.pose_weight * raw_losses['pose'] +
                         self.detection_weight * raw_losses['detection'] +
                         self.gate_weight * raw_losses['gate'])

            return {
                'total_loss': total_loss,
                'pose_loss': raw_losses['pose'],
                'detection_loss': raw_losses['detection'],
                'gate_loss': raw_losses['gate']
            }

    def get_uncertainty_weights(self) -> Optional[Dict[str, float]]:
        """Get current Kendall uncertainty weights for logging"""
        if self.use_kendall:
            return self.kendall_module.get_weights()
        return None


class PoseLoss(nn.Module):
    """Pose estimation loss (translation + rotation)"""
    
    def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
    
    def forward(self, pred_poses: torch.Tensor, gt_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute pose loss
        
        Args:
            pred_poses: Predicted poses [B, 6] (x, y, z, roll, pitch, yaw)
            gt_poses: Ground truth poses [B, 6]
        
        Returns:
            Pose loss value
        """
        # Split into translation and rotation components
        pred_translation = pred_poses[:, :3]  # x, y, z
        pred_rotation = pred_poses[:, 3:]     # roll, pitch, yaw
        
        gt_translation = gt_poses[:, :3]
        gt_rotation = gt_poses[:, 3:]
        
        # Translation loss (L2)
        translation_loss = F.mse_loss(pred_translation, gt_translation)
        
        # Rotation loss (L2 on Euler angles, could be improved with quaternions)
        rotation_loss = F.mse_loss(pred_rotation, gt_rotation)
        
        # Combine weighted losses
        total_pose_loss = (self.translation_weight * translation_loss + 
                          self.rotation_weight * rotation_loss)
        
        return total_pose_loss


class DetectionLoss(nn.Module):
    """3D object detection loss (classification + localization + confidence)"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma  # Focal loss gamma parameter
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute detection loss
        
        Args:
            predictions: Model predictions (class_logits, bbox_pred, confidence)
            targets: Ground truth targets (if available)
        
        Returns:
            Detection loss value
        """
        if targets is None:
            # For unsupervised/self-supervised training, use confidence regularization
            return self._confidence_regularization_loss(predictions)
        
        class_logits = predictions['class_logits']  # [B, num_queries, num_classes]
        bbox_pred = predictions['bbox_pred']        # [B, num_queries, 6]
        confidence = predictions['confidence']       # [B, num_queries, 1]
        
        # Get targets
        gt_classes = targets.get('classes', None)   # [B, num_gt]
        gt_bboxes = targets.get('bboxes', None)     # [B, num_gt, 6]
        
        # Classification loss (focal loss)
        if gt_classes is not None:
            class_loss = self._focal_loss(class_logits, gt_classes)
        else:
            class_loss = torch.tensor(0.0, device=class_logits.device)
        
        # Bounding box regression loss
        if gt_bboxes is not None:
            bbox_loss = self._bbox_loss(bbox_pred, gt_bboxes, confidence)
        else:
            bbox_loss = torch.tensor(0.0, device=bbox_pred.device)
        
        # Confidence loss
        conf_loss = self._confidence_loss(confidence, gt_classes)
        
        total_detection_loss = class_loss + bbox_loss + conf_loss
        
        return total_detection_loss
    
    def _focal_loss(self, class_logits: torch.Tensor, gt_classes: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for classification"""
        # Convert to probabilities
        probs = torch.softmax(class_logits, dim=-1)
        
        # For simplicity, assume background class is 0
        # In practice, you'd need proper target assignment
        batch_size, num_queries, num_classes = class_logits.shape
        
        # Create dummy targets for focal loss computation
        targets = torch.zeros(batch_size, num_queries, dtype=torch.long, device=class_logits.device)
        
        ce_loss = F.cross_entropy(class_logits.view(-1, num_classes), targets.view(-1), reduction='none')
        
        # Focal loss weights
        pt = torch.exp(-ce_loss)
        focal_weights = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weights * ce_loss
        
        return focal_loss.mean()
    
    def _bbox_loss(self, bbox_pred: torch.Tensor, gt_bboxes: torch.Tensor, 
                   confidence: torch.Tensor) -> torch.Tensor:
        """Compute bounding box regression loss"""
        # Simplified bbox loss - in practice, need proper target assignment
        # Use confidence to weight the bbox loss
        
        batch_size, num_queries, bbox_dim = bbox_pred.shape
        
        # Create dummy targets (in practice, assign based on IoU)
        targets = torch.zeros_like(bbox_pred)
        
        # L1 loss weighted by confidence
        bbox_loss = F.l1_loss(bbox_pred, targets, reduction='none')
        
        # Weight by confidence (higher confidence should have lower loss tolerance)
        weights = confidence.squeeze(-1).unsqueeze(-1).expand(-1, -1, bbox_dim)
        weighted_loss = bbox_loss * weights
        
        return weighted_loss.mean()
    
    def _confidence_loss(self, confidence: torch.Tensor, 
                        gt_classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute confidence loss"""
        # Encourage high confidence for positive detections, low for negative
        # Simplified version - encourage moderate confidence values
        
        # Penalty for extreme confidence values (too high or too low)
        entropy_penalty = -torch.mean(confidence * torch.log(confidence + 1e-8) + 
                                    (1 - confidence) * torch.log(1 - confidence + 1e-8))
        
        return entropy_penalty
    
    def _confidence_regularization_loss(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Regularization loss when no ground truth is available"""
        confidence = predictions['confidence']
        
        # Encourage sparsity in detections
        sparsity_loss = torch.mean(confidence)
        
        # Encourage diversity in predicted classes
        class_logits = predictions['class_logits']
        class_probs = torch.softmax(class_logits, dim=-1)
        diversity_loss = -torch.mean(torch.sum(class_probs * torch.log(class_probs + 1e-8), dim=-1))
        
        return sparsity_loss + 0.1 * diversity_loss


class GateRegularizationLoss(nn.Module):
    """Regularization loss for MoE gate weights"""
    
    def __init__(self, entropy_weight: float = 1.0, balance_weight: float = 1.0):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.balance_weight = balance_weight
    
    def forward(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute gate regularization loss
        
        Args:
            gate_weights: Gate weights [B, num_experts]
        
        Returns:
            Gate regularization loss
        """
        # Entropy regularization - encourage diversity in expert usage
        entropy_loss = self._entropy_regularization(gate_weights)
        
        # Balance regularization - encourage balanced expert usage
        balance_loss = self._balance_regularization(gate_weights)
        
        total_gate_loss = (self.entropy_weight * entropy_loss + 
                          self.balance_weight * balance_loss)
        
        return total_gate_loss
    
    def _entropy_regularization(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Encourage diversity in gate weights"""
        # Entropy of gate weight distribution
        entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=-1)
        
        # Encourage high entropy (diverse expert usage)
        entropy_loss = -torch.mean(entropy)
        
        return entropy_loss
    
    def _balance_regularization(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Encourage balanced usage of experts across batches"""
        # Average gate weights across batch
        avg_gates = torch.mean(gate_weights, dim=0)  # [num_experts]
        
        # Encourage uniform distribution
        num_experts = gate_weights.size(-1)
        target_weight = 1.0 / num_experts
        
        balance_loss = F.mse_loss(avg_gates, torch.full_like(avg_gates, target_weight))
        
        return balance_loss


class GeometricLoss(nn.Module):
    """Geometric consistency loss for multi-view constraints"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pose_pred: torch.Tensor, 
                stereo_baseline: float = 0.07) -> torch.Tensor:
        """
        Compute geometric consistency loss
        
        Args:
            pose_pred: Predicted pose sequence [B, T, 6]
            stereo_baseline: Stereo camera baseline in meters
        
        Returns:
            Geometric consistency loss
        """
        if pose_pred.size(1) < 2:  # Need at least 2 poses
            return torch.tensor(0.0, device=pose_pred.device)
        
        # Extract consecutive poses
        pose_t = pose_pred[:, :-1]  # [B, T-1, 6]
        pose_t1 = pose_pred[:, 1:]  # [B, T-1, 6]
        
        # Compute relative motion
        rel_translation = pose_t1[:, :, :3] - pose_t[:, :, :3]
        rel_rotation = pose_t1[:, :, 3:] - pose_t[:, :, 3:]
        
        # Smoothness constraint - penalize sudden changes
        translation_smoothness = torch.mean(torch.norm(rel_translation, dim=-1) ** 2)
        rotation_smoothness = torch.mean(torch.norm(rel_rotation, dim=-1) ** 2)
        
        smoothness_loss = translation_smoothness + rotation_smoothness
        
        return smoothness_loss


def create_loss_function(config: Dict) -> MineSLAMLoss:
    """
    Factory function to create loss function based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured loss function
    """
    return MineSLAMLoss(config)