# MineSLAM多模态SLAM评估系统使用说明

## 📋 系统概述

基于现有MineSLAM框架实现的完整多模态SLAM评估系统，支持：
- 轨迹精度评估 (ATE/RPE/RMSE)
- 3D物体检测评估 (mAP/IoU)
- MoE融合效果分析
- Kendall不确定性分析
- 实时可视化监控
- 3D语义地图展示

## 🚀 快速开始

### 1. 基于现有模型进行评估

```bash
# 使用训练好的模型进行完整评估
python evaluate_slam.py \
    --checkpoint outputs/training/latest_checkpoint.pth \
    --data_root sr_B_route2_deep_learning \
    --split test \
    --batch_size 4 \
    --output_dir outputs/slam_evaluation
```

### 2. 集成到训练过程

```python
from enhanced_slam_training import EnhancedSLAMTrainingLoop, create_enhanced_slam_training_config

# 创建增强配置
config = create_enhanced_slam_training_config(base_config)

# 启动增强训练
trainer = EnhancedSLAMTrainingLoop(config)
trainer.train()
```

### 3. 独立评估脚本

```python
from slam_evaluator import MineSLAMEvaluator
from slam_evaluation_dataset import SLAMEvaluationDataset

# 加载数据和模型
dataset = SLAMEvaluationDataset("sr_B_route2_deep_learning", split="test")
evaluator = MineSLAMEvaluator(model)

# 运行评估
metrics = evaluator.evaluate_batch(batch_data, gt_poses, gt_detections)
```

## 📊 可实现的评估指标

### SLAM轨迹精度
- **ATE (绝对轨迹误差)**: 全局位姿精度
- **RPE (相对位姿误差)**: 局部运动精度
- **平移RMSE**: 位置估计误差
- **旋转RMSE**: 姿态估计误差

### 3D物体检测
- **mAP**: 基于DARPA 8类物体的检测精度
- **3D边界框IoU**: 空间定位精度
- **类别准确率**: 分类性能
- **DETR查询效率**: 20个查询的利用率

### 多模态融合分析
- **MoE门控熵**: 专家选择多样性
- **专家利用率**: Geometric/Semantic/Visual专家分布
- **热引导权重**: 热成像引导效果
- **模态贡献度**: RGB/Depth/Thermal/LiDAR/IMU贡献

### Kendall不确定性
- **位姿/检测/门控不确定性**: σ值分析
- **任务权重平衡度**: 多任务权重分布
- **动态权重趋势**: 训练过程权重变化

## 🎨 可视化功能

### 实时传感器监控
- RGB相机预览 + 检测框叠加
- 深度图热力图 + 有效像素统计
- 热成像叠加显示 + 温度分布
- LiDAR点云3D显示 + 分割结果
- IMU数据实时曲线
- 轨迹对比 (估计vs真值)

### MoE融合过程可视化
- 3个专家权重实时饼图
- 门控熵变化曲线
- 热引导注意力热力图
- 512维token注意力可视化

### 训练过程监控
- Kendall权重动态变化曲线
- 多任务损失分解
- ATE/mAP精度提升曲线
- 专家利用率统计
- 收敛分析 + 早停触发点

### 3D语义地图
- 基于DARPA 8类物体的语义着色点云
- 机器人轨迹叠加 (真值vs估计)
- 3D边界框显示
- SLAM关键帧标记
- 检测置信度热力图

## 📁 文件结构

```
MineSLAM评估系统/
├── slam_evaluator.py              # 核心评估指标计算
├── slam_visualizer.py             # 完整可视化系统
├── slam_evaluation_dataset.py     # 数据集扩展
├── enhanced_slam_training.py      # 增强训练循环
├── evaluate_slam.py               # 完整评估脚本
└── README_SLAM_EVALUATION.md      # 本文档
```

## ⚙️ 配置选项

### SLAM评估配置
```python
slam_config = {
    'enabled': True,                    # 启用SLAM评估
    'eval_frequency': 5,                # 每5个epoch评估一次
    'visualize_frequency': 10,          # 每10个epoch可视化
    'save_visualizations': True,        # 保存可视化结果
    'evaluation_metrics': [
        'trajectory_accuracy',          # 轨迹精度
        'detection_performance',        # 检测性能
        'fusion_analysis',              # 融合分析
        'uncertainty_analysis'          # 不确定性分析
    ]
}
```

### 可视化配置
```python
viz_config = {
    'realtime_dashboard': True,         # 实时监控面板
    'moe_analysis': True,               # MoE分析可视化
    'training_curves': True,            # 训练曲线
    '3d_semantic_map': True,            # 3D语义地图
    'save_frequency': 10,               # 保存频率
    'output_format': ['png', 'html']    # 输出格式
}
```

## 🔧 依赖要求

```bash
# 核心依赖
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0

# 可视化依赖
plotly>=5.0.0
open3d>=0.13.0  # 可选，用于3D可视化

# 现有MineSLAM依赖
efficientnet-pytorch
# (其他现有依赖...)
```

## 📝 使用示例

### 完整评估流程
```bash
# 1. 使用现有检查点进行评估
python evaluate_slam.py \
    --checkpoint outputs/training/latest_checkpoint.pth \
    --data_root sr_B_route2_deep_learning \
    --split test \
    --output_dir outputs/final_evaluation

# 2. 查看评估结果
ls outputs/final_evaluation/
# comprehensive_evaluation_report.json
# evaluation_visualization_report.png
# visualizations/
```

### 集成训练评估
```python
# train_with_slam_eval.py
from enhanced_slam_training import EnhancedSLAMTrainingLoop

config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'max_epochs': 50,
    'slam_evaluation': {
        'enabled': True,
        'eval_frequency': 5,
        'save_visualizations': True
    }
}

trainer = EnhancedSLAMTrainingLoop(config)
trainer.train()
```

## 📊 输出报告格式

### JSON评估报告
```json
{
  "evaluation_info": {
    "timestamp": "2024-01-15 14:30:00",
    "total_samples": 1500,
    "evaluation_time_seconds": 125.6
  },
  "metrics": {
    "trajectory_metrics": {
      "ATE": 0.158,
      "RPE": 0.236,
      "translation_rmse": 0.145,
      "rotation_rmse": 2.34
    },
    "detection_metrics": {
      "mAP": 0.505,
      "class_aps": [0.6, 0.4, 0.5, ...]
    },
    "fusion_metrics": {
      "expert_utilization": {
        "geometric": 0.34,
        "semantic": 0.33,
        "visual": 0.33
      }
    }
  }
}
```

## 🎯 最佳实践

1. **评估频率**: 训练时每5个epoch进行SLAM评估
2. **批次大小**: 评估时使用较小批次(2-4)以节省内存
3. **可视化采样**: 仅对部分批次进行可视化以提高效率
4. **结果保存**: 定期保存最佳模型和可视化结果
5. **内存管理**: 长序列评估时注意GPU内存使用

## 🔍 故障排除

### 常见问题
1. **Open3D未安装**: 3D可视化将降级到matplotlib
2. **内存不足**: 减小batch_size或使用梯度检查点
3. **真值文件缺失**: 检查ground_truth_trajectory.csv是否存在
4. **DARPA标注格式**: 确认sr_B_route2.bag.artifacts格式正确

### 调试模式
```bash
# 使用小数据集测试
python evaluate_slam.py \
    --checkpoint outputs/training/latest_checkpoint.pth \
    --num_samples 100 \
    --batch_size 2
```

## 📈 性能基准

基于RTX 4090 (24GB)的性能参考：
- **评估速度**: ~15-20 samples/s
- **内存使用**: ~8-12GB (batch_size=4)
- **可视化开销**: ~10-15% 额外时间
- **存储需求**: ~500MB/epoch (含可视化)

---

💡 **提示**: 首次运行建议使用小数据集验证系统功能，确认无误后再进行完整评估。