"""
Hungarian Matcher for 3D Object Detection
基于真实3D标注与中心距离/体素IoU的匈牙利匹配器
若某帧无标注则跳过该帧的det loss
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import warnings


class HungarianMatcher(nn.Module):
    """
    3D目标检测的匈牙利匹配器
    基于分类成本、中心距离成本、体素IoU成本进行最优匹配
    """

    def __init__(self,
                 cost_class: float = 1.0,
                 cost_center: float = 5.0,
                 cost_iou: float = 2.0,
                 center_threshold: float = 2.0):
        """
        Args:
            cost_class: 分类成本权重
            cost_center: 3D中心距离成本权重
            cost_iou: 体素IoU成本权重
            center_threshold: 中心距离阈值（米），超过此距离认为不匹配
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_center = cost_center
        self.cost_iou = cost_iou
        self.center_threshold = center_threshold

        print(f"HungarianMatcher initialized: "
              f"cost_class={cost_class}, cost_center={cost_center}, "
              f"cost_iou={cost_iou}, threshold={center_threshold}m")

    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: List[Dict[str, torch.Tensor]]) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        执行匈牙利匹配

        Args:
            predictions: 模型预测结果
                - 'logits': [B, Q, num_classes+1] 分类logits
                - 'boxes': [B, Q, 6] 3D边界框 [cx, cy, cz, w, h, l]
            targets: 真实标注列表，每个元素包含：
                - 'labels': [N] 真实类别标签
                - 'boxes': [N, 6] 真实3D边界框
                - None 表示该帧无标注

        Returns:
            indices: 匹配结果列表，每个元素为：
                - (pred_indices, target_indices) 或 None（无标注帧）
        """
        batch_size = predictions['logits'].shape[0]
        device = predictions['logits'].device

        pred_logits = predictions['logits']  # [B, Q, num_classes+1]
        pred_boxes = predictions['boxes']    # [B, Q, 6]

        indices = []

        for i in range(batch_size):
            target_i = targets[i]

            # 检查是否有标注
            if target_i is None or len(target_i.get('labels', [])) == 0:
                # 无标注帧：跳过匹配，返回None
                indices.append(None)
                continue

            # 提取当前批次的预测和目标
            pred_logits_i = pred_logits[i]  # [Q, num_classes+1]
            pred_boxes_i = pred_boxes[i]    # [Q, 6]

            target_labels_i = target_i['labels']  # [N]
            target_boxes_i = target_i['boxes']    # [N, 6]

            # 执行匹配
            matched_indices = self._match_single_frame(
                pred_logits_i, pred_boxes_i,
                target_labels_i, target_boxes_i
            )

            indices.append(matched_indices)

        return indices

    def _match_single_frame(self,
                            pred_logits: torch.Tensor,
                            pred_boxes: torch.Tensor,
                            target_labels: torch.Tensor,
                            target_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单帧匈牙利匹配

        Args:
            pred_logits: [Q, num_classes+1] 预测分类logits
            pred_boxes: [Q, 6] 预测3D边界框
            target_labels: [N] 真实类别标签
            target_boxes: [N, 6] 真实3D边界框

        Returns:
            (pred_indices, target_indices): 匹配的索引对
        """
        num_queries = pred_logits.shape[0]
        num_targets = len(target_labels)

        if num_targets == 0:
            # 没有目标：返回空匹配
            return torch.empty(0, dtype=torch.int64, device=pred_logits.device), \
                   torch.empty(0, dtype=torch.int64, device=pred_logits.device)

        # 计算分类成本
        # 使用softmax概率计算成本，不包括背景类
        pred_probs = torch.softmax(pred_logits, dim=-1)  # [Q, num_classes+1]

        # 排除背景类的概率
        pred_probs_fg = pred_probs[:, :-1]  # [Q, num_classes]

        # 分类成本矩阵：负对数概率
        class_costs = []
        for target_label in target_labels:
            if target_label < pred_probs_fg.shape[1]:
                cost = -pred_probs_fg[:, target_label]  # [Q]
            else:
                # 标签超出范围，设置高成本
                cost = torch.ones(num_queries, device=pred_logits.device)
            class_costs.append(cost)

        class_cost_matrix = torch.stack(class_costs, dim=1)  # [Q, N]

        # 计算中心距离成本
        pred_centers = pred_boxes[:, :3]  # [Q, 3]
        target_centers = target_boxes[:, :3]  # [N, 3]

        # 计算所有对之间的L2距离
        center_distances = torch.cdist(pred_centers, target_centers)  # [Q, N]

        # 应用距离阈值：超过阈值的设置为无穷大
        center_cost_matrix = torch.where(
            center_distances <= self.center_threshold,
            center_distances,
            torch.full_like(center_distances, float('inf'))
        )

        # 计算体素IoU成本
        iou_cost_matrix = self._compute_3d_iou_cost(pred_boxes, target_boxes)  # [Q, N]

        # 组合总成本
        total_cost_matrix = (
                self.cost_class * class_cost_matrix +
                self.cost_center * center_cost_matrix +
                self.cost_iou * iou_cost_matrix
        )

        # 处理无穷大值（无法匹配的对）
        total_cost_matrix = torch.where(
            torch.isinf(total_cost_matrix),
            torch.tensor(1e6, device=total_cost_matrix.device),
            total_cost_matrix
        )

        # 执行匈牙利算法
        cost_matrix_np = total_cost_matrix.detach().cpu().numpy()

        try:
            pred_indices_np, target_indices_np = linear_sum_assignment(cost_matrix_np)
        except Exception as e:
            warnings.warn(f"Hungarian assignment failed: {e}, using greedy fallback")
            # 备用方案：贪心匹配
            pred_indices_np, target_indices_np = self._greedy_assignment(cost_matrix_np)

        # 过滤掉成本过高的匹配
        valid_matches = []
        for pi, ti in zip(pred_indices_np, target_indices_np):
            if cost_matrix_np[pi, ti] < 1e5:  # 过滤极高成本的匹配
                valid_matches.append((pi, ti))

        if valid_matches:
            pred_indices = torch.tensor([m[0] for m in valid_matches],
                                       dtype=torch.int64, device=pred_logits.device)
            target_indices = torch.tensor([m[1] for m in valid_matches],
                                         dtype=torch.int64, device=pred_logits.device)
        else:
            # 没有有效匹配
            pred_indices = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
            target_indices = torch.empty(0, dtype=torch.int64, device=pred_logits.device)

        return pred_indices, target_indices

    def _compute_3d_iou_cost(self,
                             pred_boxes: torch.Tensor,
                             target_boxes: torch.Tensor) -> torch.Tensor:
        """
        计算3D体素IoU成本

        Args:
            pred_boxes: [Q, 6] 预测边界框
            target_boxes: [N, 6] 真实边界框

        Returns:
            iou_cost: [Q, N] IoU成本矩阵
        """
        num_preds = pred_boxes.shape[0]
        num_targets = target_boxes.shape[0]

        iou_matrix = torch.zeros(num_preds, num_targets, device=pred_boxes.device)

        for i in range(num_preds):
            for j in range(num_targets):
                iou = self._compute_single_3d_iou(pred_boxes[i], target_boxes[j])
                iou_matrix[i, j] = iou

        # IoU成本：1 - IoU（IoU越高，成本越低）
        iou_cost = 1.0 - iou_matrix

        return iou_cost

    def _compute_single_3d_iou(self,
                               box1: torch.Tensor,
                               box2: torch.Tensor) -> torch.Tensor:
        """
        计算两个3D边界框的IoU

        Args:
            box1: [6] 第一个边界框 [cx, cy, cz, w, h, l]
            box2: [6] 第二个边界框 [cx, cy, cz, w, h, l]

        Returns:
            iou: scalar IoU值
        """
        # 提取中心和尺寸
        center1, size1 = box1[:3], box1[3:]
        center2, size2 = box2[:3], box2[3:]

        # 计算边界框的最小和最大坐标
        min1 = center1 - size1 / 2
        max1 = center1 + size1 / 2
        min2 = center2 - size2 / 2
        max2 = center2 + size2 / 2

        # 计算交集
        intersection_min = torch.max(min1, min2)
        intersection_max = torch.min(max1, max2)

        # 检查是否有交集
        intersection_size = torch.clamp(intersection_max - intersection_min, min=0)
        intersection_volume = intersection_size.prod()

        # 计算各自体积
        volume1 = size1.prod()
        volume2 = size2.prod()

        # 计算并集
        union_volume = volume1 + volume2 - intersection_volume

        # 计算IoU
        if union_volume > 0:
            iou = intersection_volume / union_volume
        else:
            iou = torch.tensor(0.0, device=box1.device)

        return iou

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        贪心匹配算法（备用方案）

        Args:
            cost_matrix: [Q, N] 成本矩阵

        Returns:
            (pred_indices, target_indices): 匹配索引
        """
        pred_indices = []
        target_indices = []
        used_targets = set()

        # 按成本排序所有可能的匹配
        q, n = cost_matrix.shape
        matches = []
        for i in range(q):
            for j in range(n):
                matches.append((cost_matrix[i, j], i, j))

        matches.sort()  # 按成本从小到大排序

        # 贪心选择
        for cost, pred_idx, target_idx in matches:
            if target_idx not in used_targets and cost < 1e5:
                pred_indices.append(pred_idx)
                target_indices.append(target_idx)
                used_targets.add(target_idx)

        return np.array(pred_indices), np.array(target_indices)


class TargetGenerator:
    """
    目标标注生成器
    将DARPA数据集标注转换为匹配器需要的格式
    """

    def __init__(self,
                 class_mapping: Optional[Dict[str, int]] = None):
        """
        Args:
            class_mapping: 类别名称到ID的映射
        """
        # DARPA SubT数据集的标准类别
        if class_mapping is None:
            self.class_mapping = {
                'fiducial': 0,      # 基准标志
                'extinguisher': 1,   # 灭火器
                'phone': 2,         # 电话
                'backpack': 3,      # 背包
                'survivor': 4,      # 幸存者
                'drill': 5,         # 钻头
                'rope': 6,          # 绳索
                'helmet': 7         # 头盔
            }
        else:
            self.class_mapping = class_mapping

        print(f"TargetGenerator initialized with {len(self.class_mapping)} classes: {list(self.class_mapping.keys())}")

    def generate_targets_from_darpa(self,
                                    annotations: List[Dict],
                                    device: torch.device = None) -> List[Dict[str, torch.Tensor]]:
        """
        从DARPA标注生成目标

        Args:
            annotations: DARPA标注列表，每个元素包含该帧的标注信息
            device: 目标设备

        Returns:
            targets: 目标列表
        """
        if device is None:
            device = torch.device('cpu')

        targets = []

        for annotation in annotations:
            if annotation is None or len(annotation.get('objects', [])) == 0:
                # 无标注帧
                targets.append(None)
                continue

            labels = []
            boxes = []

            for obj in annotation['objects']:
                # 提取类别
                obj_class = obj.get('class', 'unknown')
                if obj_class in self.class_mapping:
                    class_id = self.class_mapping[obj_class]
                else:
                    continue  # 跳过未知类别

                # 提取3D边界框
                bbox_3d = obj.get('bbox_3d')
                if bbox_3d is None:
                    continue

                # 转换为中心-尺寸格式 [cx, cy, cz, w, h, l]
                if 'center' in bbox_3d and 'size' in bbox_3d:
                    center = bbox_3d['center']  # [x, y, z]
                    size = bbox_3d['size']      # [w, h, l]
                    box = center + size
                elif 'min' in bbox_3d and 'max' in bbox_3d:
                    # 从最小最大坐标转换
                    min_coords = bbox_3d['min']
                    max_coords = bbox_3d['max']
                    center = [(min_coords[i] + max_coords[i]) / 2 for i in range(3)]
                    size = [max_coords[i] - min_coords[i] for i in range(3)]
                    box = center + size
                else:
                    continue  # 跳过格式不正确的标注

                labels.append(class_id)
                boxes.append(box)

            if labels:
                targets.append({
                    'labels': torch.tensor(labels, dtype=torch.long, device=device),
                    'boxes': torch.tensor(boxes, dtype=torch.float32, device=device)
                })
            else:
                targets.append(None)

        return targets


# 模块测试
if __name__ == '__main__':
    print("🧪 Testing HungarianMatcher...")

    # 创建匹配器
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_center=5.0,
        cost_iou=2.0
    )

    # 创建测试数据
    batch_size = 2
    num_queries = 20
    num_classes = 8

    # 模拟预测结果
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_boxes = torch.randn(batch_size, num_queries, 6) * 2  # 限制在合理范围

    predictions = {
        'logits': pred_logits,
        'boxes': pred_boxes
    }

    # 模拟目标标注
    targets = [
        {
            'labels': torch.tensor([0, 1, 2], dtype=torch.long),
            'boxes': torch.tensor([
                [0, 0, 0, 1, 1, 1],
                [2, 2, 2, 0.5, 0.5, 0.5],
                [4, 4, 4, 2, 2, 2]
            ], dtype=torch.float32)
        },
        None  # 第二帧无标注
    ]

    print(f"Predictions shape: logits={pred_logits.shape}, boxes={pred_boxes.shape}")
    print(f"Targets: {len(targets)} frames, frame 0: {len(targets[0]['labels']) if targets[0] else 0} objects")

    # 执行匹配
    with torch.no_grad():
        indices = matcher(predictions, targets)

    print(f"\nMatching results:")
    for i, idx_pair in enumerate(indices):
        if idx_pair is None:
            print(f"  Frame {i}: No annotations (skipped)")
        else:
            pred_idx, target_idx = idx_pair
            print(f"  Frame {i}: {len(pred_idx)} matches")
            if len(pred_idx) > 0:
                print(f"    Pred indices: {pred_idx}")
                print(f"    Target indices: {target_idx}")

    # 测试目标生成器
    print(f"\n🎯 Testing TargetGenerator...")
    target_gen = TargetGenerator()

    # 模拟DARPA标注
    darpa_annotations = [
        {
            'objects': [
                {
                    'class': 'fiducial',
                    'bbox_3d': {
                        'center': [1.0, 2.0, 0.5],
                        'size': [0.2, 0.2, 0.1]
                    }
                },
                {
                    'class': 'extinguisher',
                    'bbox_3d': {
                        'min': [5.0, 5.0, 0.0],
                        'max': [6.0, 6.0, 1.0]
                    }
                }
            ]
        },
        {
            'objects': []  # 无标注帧
        }
    ]

    generated_targets = target_gen.generate_targets_from_darpa(darpa_annotations)

    print(f"Generated targets:")
    for i, target in enumerate(generated_targets):
        if target is None:
            print(f"  Frame {i}: No targets")
        else:
            print(f"  Frame {i}: {len(target['labels'])} targets")
            print(f"    Labels: {target['labels']}")
            print(f"    Boxes shape: {target['boxes'].shape}")

    print("\n✅ HungarianMatcher test completed!")