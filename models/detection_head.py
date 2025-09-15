"""
DetectionHead: DETR-style 3D Object Detection Head
DETR风格3D目标检测头：Q=20查询，TransformerDecoder×4，输出(B,Q,10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class DetectionHead(nn.Module):
    """
    DETR风格3D目标检测头
    输入：多模态融合token [B, T, D]
    输出：检测结果 [B, Q, 10] (Q=20查询，10=类别+3D框)
    """

    def __init__(self,
                 input_dim: int = 512,
                 num_queries: int = 20,
                 num_classes: int = 8,  # 基于DARPA标注：fiducial, extinguisher, phone, backpack, survivor, drill等
                 decoder_layers: int = 4,
                 nhead: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.decoder_layers = decoder_layers

        # DETR查询嵌入 - 可学习的Q=20个查询
        self.query_embeddings = nn.Parameter(
            torch.randn(num_queries, input_dim) * 0.1
        )

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1000, input_dim) * 0.1  # 支持最多1000个token
        )

        # TransformerDecoder ×4层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )

        # 输出头
        # 3D边界框：中心(x,y,z) + 尺寸(w,h,l) = 6维
        self.bbox_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 6)  # 3D bbox: center(3) + size(3)
        )

        # 分类头：num_classes + 1 (背景类)
        self.class_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes + 1)
        )

        # 初始化权重
        self._init_weights()

        print(f"DetectionHead initialized: Q={num_queries}, Classes={num_classes}, "
              f"Decoder={decoder_layers}×{nhead}head → (B,{num_queries},10)")

    def _init_weights(self):
        """初始化模型权重"""
        # 查询嵌入
        nn.init.normal_(self.query_embeddings, std=0.01)

        # 位置编码
        nn.init.normal_(self.position_embedding, std=0.01)

        # 输出头
        for module in [self.bbox_head, self.class_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        # bbox头最后一层使用小的初始化（DETR惯例）
        nn.init.xavier_uniform_(self.bbox_head[-1].weight, gain=0.01)

    def forward(self, tokens: torch.Tensor,
                token_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            tokens: [B, T, D] 融合后的token序列
            token_mask: [B, T] token掩码（可选）

        Returns:
            result_dict: 包含预测结果的字典
                - 'logits': [B, Q, num_classes+1] 分类logits
                - 'boxes': [B, Q, 6] 3D边界框 [cx, cy, cz, w, h, l]
                - 'pred_combined': [B, Q, 10] 组合输出（兼容要求）
        """
        batch_size, seq_len, embed_dim = tokens.shape
        device = tokens.device

        # 添加位置编码到memory tokens
        pos_embed = self.position_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        memory = tokens + pos_embed

        # 准备查询：复制到batch
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Q, D]

        # 准备mask（如果需要）
        if token_mask is not None:
            # token_mask: [B, T] -> memory_key_padding_mask
            memory_key_padding_mask = ~token_mask  # 反转：True表示padding
        else:
            memory_key_padding_mask = None

        # TransformerDecoder处理
        # queries作为tgt，memory作为encoder输出
        decoder_output = self.transformer_decoder(
            tgt=queries,  # [B, Q, D]
            memory=memory,  # [B, T, D]
            memory_key_padding_mask=memory_key_padding_mask  # [B, T]
        )  # Output: [B, Q, D]

        # 预测分类和回归
        class_logits = self.class_head(decoder_output)  # [B, Q, num_classes+1]
        bbox_preds = self.bbox_head(decoder_output)     # [B, Q, 6]

        # 3D边界框后处理：应用sigmoid约束到合理范围
        # center可以在较大范围内，size需要为正值
        bbox_centers = bbox_preds[:, :, :3]  # [B, Q, 3] 中心点
        bbox_sizes = torch.sigmoid(bbox_preds[:, :, 3:]) * 10.0  # [B, Q, 3] 尺寸，最大10米

        bbox_processed = torch.cat([bbox_centers, bbox_sizes], dim=-1)  # [B, Q, 6]

        # 构建组合输出 [B, Q, 10] = [classes(1) + bbox(6) + confidence(1) + reserved(2)]
        # 为了兼容(B, Q, 10)的要求，我们需要重新组织输出
        pred_classes = torch.argmax(class_logits, dim=-1, keepdim=True).float()  # [B, Q, 1]
        pred_confidence = torch.max(F.softmax(class_logits, dim=-1), dim=-1, keepdim=True)[0]  # [B, Q, 1]
        reserved_dims = torch.zeros(batch_size, self.num_queries, 2, device=device)  # [B, Q, 2]

        pred_combined = torch.cat([
            pred_classes,      # [B, Q, 1] 预测类别
            bbox_processed,    # [B, Q, 6] 3D边界框
            pred_confidence,   # [B, Q, 1] 置信度
            reserved_dims      # [B, Q, 2] 保留维度
        ], dim=-1)  # [B, Q, 10]

        return {
            'logits': class_logits,        # [B, Q, num_classes+1] 用于损失计算
            'boxes': bbox_processed,       # [B, Q, 6] 用于损失计算
            'pred_combined': pred_combined, # [B, Q, 10] 符合接口要求
            'decoder_features': decoder_output  # [B, Q, D] 解码器特征（可用于可视化）
        }

    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                     targets: List[Dict[str, torch.Tensor]],
                     matcher_indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        计算检测损失（需要与matcher.py配合使用）

        Args:
            predictions: 模型预测结果字典
            targets: 真实标注列表
            matcher_indices: 匈牙利匹配结果

        Returns:
            loss_dict: 损失字典
        """
        pred_logits = predictions['logits']  # [B, Q, num_classes+1]
        pred_boxes = predictions['boxes']    # [B, Q, 6]

        batch_size = pred_logits.shape[0]
        device = pred_logits.device

        # 准备匹配后的目标
        target_classes_list = []
        target_boxes_list = []

        for i, (indices_i, target_i) in enumerate(zip(matcher_indices, targets)):
            if indices_i is None or target_i is None:
                # 无标注帧：全部视为背景
                target_classes_list.append(
                    torch.full((self.num_queries,), self.num_classes,
                             dtype=torch.long, device=device)
                )
                target_boxes_list.append(
                    torch.zeros((self.num_queries, 6), device=device)
                )
            else:
                pred_idx, target_idx = indices_i

                # 构建匹配后的目标类别
                target_classes = torch.full((self.num_queries,), self.num_classes,
                                           dtype=torch.long, device=device)
                target_classes[pred_idx] = target_i['labels'][target_idx]
                target_classes_list.append(target_classes)

                # 构建匹配后的目标框
                target_boxes = torch.zeros((self.num_queries, 6), device=device)
                if len(pred_idx) > 0:
                    target_boxes[pred_idx] = target_i['boxes'][target_idx]
                target_boxes_list.append(target_boxes)

        # 堆叠目标
        target_classes = torch.stack(target_classes_list)  # [B, Q]
        target_boxes = torch.stack(target_boxes_list)      # [B, Q, 6]

        # 分类损失（交叉熵）
        loss_class = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_classes.reshape(-1)
        )

        # 回归损失（仅对非背景类计算）
        # 筛选出非背景的预测和目标
        foreground_mask = (target_classes != self.num_classes)

        if foreground_mask.sum() > 0:
            # L1损失
            loss_bbox = F.l1_loss(
                pred_boxes[foreground_mask],
                target_boxes[foreground_mask]
            )

            # IoU损失（3D情况下使用中心距离作为代理）
            pred_centers = pred_boxes[foreground_mask][:, :3]
            target_centers = target_boxes[foreground_mask][:, :3]
            loss_center = F.mse_loss(pred_centers, target_centers)
        else:
            loss_bbox = torch.tensor(0.0, device=device)
            loss_center = torch.tensor(0.0, device=device)

        return {
            'loss_class': loss_class,
            'loss_bbox': loss_bbox,
            'loss_center': loss_center,
            'loss_det': loss_class + loss_bbox + loss_center  # 总检测损失
        }


# 3D边界框工具函数
class BBox3DUtils:
    """3D边界框工具函数"""

    @staticmethod
    def center_size_to_corners(boxes: torch.Tensor) -> torch.Tensor:
        """
        将中心-尺寸格式转换为角点格式

        Args:
            boxes: [N, 6] [cx, cy, cz, w, h, l]

        Returns:
            corners: [N, 8, 3] 8个角点坐标
        """
        centers = boxes[:, :3]  # [N, 3]
        sizes = boxes[:, 3:]    # [N, 3]

        # 半尺寸
        half_sizes = sizes / 2  # [N, 3]

        # 8个角点的偏移
        offsets = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]
        ], dtype=boxes.dtype, device=boxes.device)  # [8, 3]

        # 广播计算角点
        corners = centers.unsqueeze(1) + offsets.unsqueeze(0) * half_sizes.unsqueeze(1)

        return corners

    @staticmethod
    def compute_3d_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        计算3D边界框IoU（简化版本）

        Args:
            boxes1: [N, 6] 第一组框
            boxes2: [M, 6] 第二组框

        Returns:
            iou: [N, M] IoU矩阵
        """
        # 简化实现：使用中心距离的倒数作为IoU的代理
        centers1 = boxes1[:, :3]  # [N, 3]
        centers2 = boxes2[:, :3]  # [M, 3]

        # 计算中心距离
        distances = torch.cdist(centers1, centers2)  # [N, M]

        # 转换为相似度（距离越小，相似度越高）
        iou_proxy = 1.0 / (1.0 + distances)

        return iou_proxy


# 模块测试
if __name__ == '__main__':
    print("🧪 Testing DetectionHead...")

    # 创建DetectionHead
    detection_head = DetectionHead(
        input_dim=512,
        num_queries=20,
        num_classes=8,
        decoder_layers=4
    )

    # 测试数据
    batch_size = 2
    seq_len = 100
    embed_dim = 512

    tokens = torch.randn(batch_size, seq_len, embed_dim)
    token_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f"Input tokens shape: {tokens.shape}")
    print(f"Token mask shape: {token_mask.shape}")

    # 前向传播
    with torch.no_grad():
        predictions = detection_head(tokens, token_mask)

        print(f"\nPrediction shapes:")
        for key, value in predictions.items():
            print(f"  {key}: {value.shape}")

        print(f"\nCombined output shape: {predictions['pred_combined'].shape}")
        print(f"Expected shape: (B={batch_size}, Q=20, 10)")
        assert predictions['pred_combined'].shape == (batch_size, 20, 10)

    # 测试梯度
    print("\n🔄 Testing gradients...")
    predictions_grad = detection_head(tokens, token_mask)

    # 模拟损失
    dummy_loss = predictions_grad['pred_combined'].sum()
    dummy_loss.backward()

    # 检查梯度
    grad_norm = torch.norm(torch.cat([
        p.grad.flatten() for p in detection_head.parameters()
        if p.grad is not None
    ]))
    print(f"Gradient norm: {grad_norm.item():.6f}")

    # 测试3D工具函数
    print("\n🔧 Testing 3D BBox utilities...")
    test_boxes = torch.tensor([[0, 0, 0, 2, 2, 2], [1, 1, 1, 1, 1, 1]])
    corners = BBox3DUtils.center_size_to_corners(test_boxes)
    print(f"Corners shape: {corners.shape}")

    iou_matrix = BBox3DUtils.compute_3d_iou(test_boxes, test_boxes)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    print(f"IoU matrix:\n{iou_matrix}")

    print("\n✅ DetectionHead test completed!")