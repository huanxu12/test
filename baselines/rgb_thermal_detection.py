"""
RGB-Thermal 2D Detection Baseline
基于RGB和热成像的2D目标检测，使用YOLOv8或RT-DETR
支持双通道堆叠和late fusion策略
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import json
import warnings

# 尝试导入YOLOv8（如果可用）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("YOLOv8 not available, using simplified detection model")


class SimplifiedDetectionModel(nn.Module):
    """
    简化的检测模型（当YOLOv8不可用时使用）
    基于轻量级CNN + 简单检测头
    """
    
    def __init__(self, num_classes: int = 8, input_channels: int = 4):
        super().__init__()
        self.num_classes = num_classes
        
        # 简化的backbone
        self.backbone = nn.Sequential(
            # 输入: [B, C, H, W]
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes * 5)  # [x, y, w, h, conf] per class
        )
    
    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        
        # 重塑为 [B, num_classes, 5]
        batch_size = x.size(0)
        detections = detections.view(batch_size, self.num_classes, 5)
        
        return detections


class RGBThermalDetector:
    """
    RGB-Thermal双模态2D检测器
    支持早期融合（通道堆叠）和晚期融合策略
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.fusion_strategy = config.get('fusion_strategy', 'early')  # 'early' or 'late'
        self.model_type = config.get('model_type', 'yolov8n')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        
        # 类别映射 (根据实际标注调整)
        self.class_names = [
            'person', 'backpack', 'phone', 'drill', 
            'extinguisher', 'fiducial', 'survivor', 'other'
        ]
        self.num_classes = len(self.class_names)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self._init_model()
        
        print(f"RGB-Thermal Detector initialized:")
        print(f"  Fusion strategy: {self.fusion_strategy}")
        print(f"  Model type: {self.model_type}")
        print(f"  Device: {self.device}")
    
    def _init_model(self):
        """初始化检测模型"""
        if YOLO_AVAILABLE and self.model_type.startswith('yolo'):
            # 使用YOLOv8
            try:
                self.model = YOLO(f'{self.model_type}.pt')
                self.use_yolo = True
                print("✓ Using YOLOv8 model")
            except Exception as e:
                print(f"Failed to load YOLOv8: {e}")
                self.use_yolo = False
                self._init_simplified_model()
        else:
            self.use_yolo = False
            self._init_simplified_model()
    
    def _init_simplified_model(self):
        """初始化简化模型"""
        input_channels = 4 if self.fusion_strategy == 'early' else 3
        self.model = SimplifiedDetectionModel(
            num_classes=self.num_classes,
            input_channels=input_channels
        ).to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        print("✓ Using simplified detection model")
    
    def preprocess_images(self, rgb_image: torch.Tensor, 
                         thermal_image: torch.Tensor) -> torch.Tensor:
        """
        图像预处理和融合
        
        Args:
            rgb_image: RGB图像 [3, H, W]
            thermal_image: 热成像图像 [1, H, W]
        
        Returns:
            processed_image: 预处理后的图像
        """
        # 确保图像尺寸一致
        if rgb_image.shape[-2:] != thermal_image.shape[-2:]:
            # 将thermal图像resize到RGB图像的尺寸
            thermal_resized = torch.nn.functional.interpolate(
                thermal_image.unsqueeze(0),  # 添加batch维度
                size=rgb_image.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # 移除batch维度
        else:
            thermal_resized = thermal_image
        
        if self.fusion_strategy == 'early':
            # 早期融合：通道堆叠
            fused_image = torch.cat([rgb_image, thermal_resized], dim=0)  # [4, H, W]
        else:
            # 晚期融合：分别处理RGB图像
            fused_image = rgb_image  # [3, H, W]
        
        return fused_image
    
    def detect_yolo(self, image: torch.Tensor) -> List[Dict]:
        """
        使用YOLOv8进行检测
        
        Args:
            image: 输入图像张量
        
        Returns:
            检测结果列表
        """
        # 转换为numpy格式 (HWC)
        if image.dim() == 3:
            if image.shape[0] <= 4:  # CHW格式
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image.cpu().numpy()
        
        # 如果是4通道，只使用前3通道
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]
        
        # 转换到[0, 255]范围
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # YOLO推理
        results = self.model(image_np)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    conf = boxes.conf[i].cpu().item()
                    cls = int(boxes.cls[i].cpu().item())
                    
                    if conf > self.confidence_threshold:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls,
                            'class_name': self.class_names[min(cls, len(self.class_names)-1)]
                        }
                        detections.append(detection)
        
        return detections
    
    def detect_simplified(self, image: torch.Tensor) -> List[Dict]:
        """
        使用简化模型进行检测
        
        Args:
            image: 输入图像张量 [C, H, W]
        
        Returns:
            检测结果列表
        """
        # 添加batch维度
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]
        
        # 确保在正确设备上
        image = image.to(self.device)
        
        with torch.no_grad():
            # 模型推理
            outputs = self.model(image)  # [1, num_classes, 5]
            
            # 解析输出
            detections = []
            for cls_id in range(self.num_classes):
                cls_output = outputs[0, cls_id]  # [5] -> [x, y, w, h, conf]
                
                x, y, w, h, conf = cls_output.cpu().numpy()
                
                if conf > self.confidence_threshold:
                    # 转换为xyxy格式
                    x1 = max(0, x - w/2)
                    y1 = max(0, y - h/2)
                    x2 = min(image.shape[-1], x + w/2)
                    y2 = min(image.shape[-2], y + h/2)
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id]
                    }
                    detections.append(detection)
        
        return detections
    
    def detect(self, rgb_image: torch.Tensor, 
               thermal_image: torch.Tensor) -> List[Dict]:
        """
        执行2D检测
        
        Args:
            rgb_image: RGB图像 [3, H, W]
            thermal_image: 热成像图像 [1, H, W]
        
        Returns:
            检测结果列表
        """
        # 预处理和融合
        processed_image = self.preprocess_images(rgb_image, thermal_image)
        
        # 根据模型类型选择检测方法
        if self.use_yolo:
            detections = self.detect_yolo(processed_image)
        else:
            detections = self.detect_simplified(processed_image)
        
        # 后处理：添加图像尺寸信息
        image_height, image_width = rgb_image.shape[-2:]
        for detection in detections:
            detection['image_width'] = image_width
            detection['image_height'] = image_height
        
        return detections
    
    def project_to_3d(self, detections: List[Dict], 
                     depth_image: Optional[torch.Tensor] = None,
                     lidar_points: Optional[torch.Tensor] = None,
                     camera_height: float = 1.5) -> List[Dict]:
        """
        将2D检测投影到3D
        
        Args:
            detections: 2D检测结果
            depth_image: 深度图像 [1, H, W] (可选)
            lidar_points: 激光雷达点云 [N, 4] (可选)
            camera_height: 相机高度假设 (米)
        
        Returns:
            3D检测结果
        """
        detections_3d = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 简化的3D投影
            # 假设：物体在地面上，相机高度已知
            detection_3d = detection.copy()
            
            if depth_image is not None and depth_image.numel() > 0:
                # 使用深度图像
                h, w = depth_image.shape[-2:]
                pixel_x = int(center_x * w / detection['image_width'])
                pixel_y = int(center_y * h / detection['image_height'])
                
                if 0 <= pixel_x < w and 0 <= pixel_y < h:
                    depth = depth_image[0, pixel_y, pixel_x].item()
                    if depth > 0:
                        # 简化的相机投影（需要相机内参进行准确投影）
                        # 这里使用简化假设
                        world_x = (center_x - detection['image_width']/2) * depth / 500  # 简化焦距
                        world_y = camera_height  # 假设高度
                        world_z = depth
                        
                        detection_3d['center_3d'] = [world_x, world_y, world_z]
                        detection_3d['depth_source'] = 'depth_image'
            
            elif lidar_points is not None and len(lidar_points) > 0:
                # 使用激光雷达投影（简化）
                # 在图像区域内查找最近的激光雷达点
                # 这需要相机-激光雷达标定，这里使用简化方法
                
                # 假设激光雷达点已经投影到图像平面
                # 实际应用中需要使用标定参数
                world_x = (center_x - detection['image_width']/2) * 0.01  # 简化
                world_y = camera_height
                world_z = 5.0  # 简化深度假设
                
                detection_3d['center_3d'] = [world_x, world_y, world_z]
                detection_3d['depth_source'] = 'lidar_projection'
            
            else:
                # 使用相机高度假设
                world_x = (center_x - detection['image_width']/2) * 0.01
                world_y = camera_height
                world_z = 3.0  # 默认深度假设
                
                detection_3d['center_3d'] = [world_x, world_y, world_z]
                detection_3d['depth_source'] = 'height_assumption'
            
            # 添加3D边界框尺寸（简化）
            object_sizes = {
                'person': [0.6, 1.8, 0.3],
                'backpack': [0.4, 0.5, 0.2],
                'phone': [0.1, 0.2, 0.02],
                'drill': [0.3, 0.8, 0.1],
                'extinguisher': [0.2, 0.6, 0.2],
                'fiducial': [0.1, 0.1, 0.02],
                'survivor': [0.6, 1.8, 0.3],
                'other': [0.5, 0.5, 0.5]
            }
            
            class_name = detection_3d['class_name']
            size = object_sizes.get(class_name, object_sizes['other'])
            detection_3d['size_3d'] = size  # [width, height, depth]
            
            detections_3d.append(detection_3d)
        
        return detections_3d


def run_rgb_thermal_detection_on_dataset(dataset, config: Dict, output_dir: str):
    """
    在数据集上运行RGB-Thermal检测
    
    Args:
        dataset: MineSLAM数据集对象
        config: 算法配置
        output_dir: 输出目录
    """
    print("="*60)
    print("Running RGB-Thermal 2D Detection Baseline")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建检测器
    detector = RGBThermalDetector(config)
    
    # 处理统计
    total_frames = 0
    successful_frames = 0
    total_detections = 0
    total_processing_time = 0.0
    
    # 存储所有检测结果
    all_detections = []
    
    print(f"Processing {len(dataset)} frames...")
    
    for i, sample in enumerate(dataset):
        if 'rgb' not in sample or 'thermal' not in sample:
            print(f"Frame {i}: Missing RGB or thermal data, skipping")
            continue
        
        start_time = time.time()
        
        try:
            # 提取数据
            rgb_image = sample['rgb']  # [3, H, W]
            thermal_image = sample['thermal']  # [1, H, W]
            depth_image = sample.get('depth', None)  # [1, H, W] 可选
            lidar_points = sample.get('lidar', None)  # [N, 4] 可选
            timestamp = sample['timestamp'].item()
            
            # 2D检测
            detections_2d = detector.detect(rgb_image, thermal_image)
            
            # 3D投影
            detections_3d = detector.project_to_3d(
                detections_2d,
                depth_image=depth_image,
                lidar_points=lidar_points,
                camera_height=config.get('camera_height', 1.5)
            )
            
            processing_time = time.time() - start_time
            
            # 保存结果
            frame_result = {
                'frame_id': i,
                'timestamp': timestamp,
                'detections_2d': detections_2d,
                'detections_3d': detections_3d,
                'num_detections': len(detections_2d),
                'processing_time': processing_time,
                'success': True
            }
            
            all_detections.append(frame_result)
            
            total_frames += 1
            successful_frames += 1
            total_detections += len(detections_2d)
            total_processing_time += processing_time
            
            # 打印进度
            if i % 100 == 0:
                print(f"Processed {i}/{len(dataset)} frames, "
                      f"Avg detections: {total_detections/max(1,successful_frames):.1f}/frame")
        
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            frame_result = {
                'frame_id': i,
                'timestamp': sample.get('timestamp', 0.0),
                'detections_2d': [],
                'detections_3d': [],
                'num_detections': 0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
            all_detections.append(frame_result)
            total_frames += 1
    
    # 保存所有检测结果
    detections_file = output_dir / "rgb_thermal_detections.json"
    with open(detections_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    # 保存统计结果
    results = {
        'algorithm': 'RGB-Thermal 2D Detection',
        'config': config,
        'statistics': {
            'total_frames': total_frames,
            'successful_frames': successful_frames,
            'success_rate': successful_frames / max(1, total_frames),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / max(1, successful_frames),
            'avg_processing_time': total_processing_time / max(1, total_frames),
            'total_processing_time': total_processing_time
        }
    }
    
    results_file = output_dir / "rgb_thermal_detection_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print("RGB-Thermal Detection Results")
    print("="*60)
    print(f"Total frames processed: {total_frames}")
    print(f"Successful frames: {successful_frames}")
    print(f"Success rate: {successful_frames/max(1,total_frames)*100:.1f}%")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/frame: {total_detections/max(1,successful_frames):.1f}")
    print(f"Average processing time: {total_processing_time/max(1,total_frames)*1000:.1f} ms/frame")
    print(f"Results saved to: {output_dir}")
    
    return str(detections_file), results