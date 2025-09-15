# MineSLAM: Multi-modal Adaptive 3D Semantic SLAM

基于真实DARPA SubT数据集的多模态自适应3D语义SLAM系统，专为退化矿井环境设计。

🚨 **STRICT REAL DATA POLICY** 🚨  
**本工程严禁使用合成/随机数据跑通测试**  
**This project STRICTLY PROHIBITS using synthetic/random data for testing**

## 项目概述

MineSLAM是一个端到端的SLAM系统，融合RGB、热成像、深度、LiDAR和IMU多模态传感器数据，使用Mixture of Experts (MoE)架构实现环境自适应。

### 关键特性

- **多模态传感器融合**: RGB、热成像、深度、LiDAR、IMU
- **真实数据驱动**: 基于DARPA SubT Challenge真实矿井数据
- **环境自适应**: MoE架构根据环境条件动态选择算法
- **严格数据契约**: 禁止合成数据，确保真实性
- **完整评估框架**: ATE、RPE、mAP等标准指标
- **基线算法**: LiDAR里程计和RGB-Thermal检测

## 目录结构

```
MineSLAM/
├── configs/                    # 配置文件
│   └── mineslam.yaml          # 主配置文件
├── data/                      # 数据处理模块  
│   └── mineslam_dataset.py    # 数据集加载器
├── models/                    # 模型定义
├── baselines/                 # 基线算法
│   ├── lidar_odometry.py      # LiDAR里程计
│   └── rgb_thermal_detection.py # RGB-热成像检测
├── scripts/                   # 工具脚本
│   └── generate_real_data_index.py # 数据索引生成
├── baseline_eval.py           # 基线评估脚本
├── sr_B_route2_deep_learning/ # 真实数据集
└── lists/                     # 数据索引文件
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n mineslam python=3.8
conda activate mineslam

# 安装基础依赖
pip install torch torchvision torchaudio
pip install numpy pandas opencv-python
pip install open3d scipy scikit-learn
pip install matplotlib seaborn
pip install pyyaml tqdm

# 安装可选依赖（用于完整功能）
pip install ultralytics  # YOLOv8检测模型
```

### 2. 数据准备

项目使用真实的DARPA SubT数据集，已包含在`sr_B_route2_deep_learning/`目录中。

```bash
# 生成数据索引（如果需要）
python scripts/generate_real_data_index.py \
    --dataset_root ./sr_B_route2_deep_learning \
    --output_dir ./lists \
    --time_threshold_ms 50
```

### 3. 验证数据加载

```bash
# 测试数据集加载器
python -c "
import yaml
from data.mineslam_dataset import MineSLAMDataset

# 加载配置
with open('configs/mineslam.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建数据集
dataset = MineSLAMDataset(config, split='train')
print(f'数据集大小: {len(dataset)}')

# 测试单个样本
sample = dataset[0]
print(f'样本包含的模态: {list(sample.keys())}')
for key, tensor in sample.items():
    if hasattr(tensor, 'shape'):
        print(f'{key}: {tensor.shape} ({tensor.dtype})')
"
```

## 基线算法使用

### A. LiDAR-only 里程计

基于体素下采样、法线估计和point-to-plane ICP的里程计算法。

```bash
# 单独运行LiDAR里程计
python -c "
import yaml
from data.mineslam_dataset import MineSLAMDataset
from baselines.lidar_odometry import run_lidar_odometry_on_dataset

# 加载配置和数据集
with open('configs/mineslam.yaml', 'r') as f:
    config = yaml.safe_load(f)
dataset = MineSLAMDataset(config, split='test')

# 运行算法
lidar_config = config.get('lidar_odometry', {})
trajectory_file, results = run_lidar_odometry_on_dataset(
    dataset, lidar_config, './results/lidar_test'
)
print(f'轨迹保存至: {trajectory_file}')
"
```

**参数配置** (在`configs/mineslam.yaml`中):
```yaml
lidar_odometry:
  voxel_size: 0.3                    # 体素下采样大小
  max_correspondence_distance: 1.0    # ICP对应点最大距离
  normal_radius: 0.5                 # 法线估计半径
  icp_max_iteration: 50              # ICP最大迭代次数
```

### B. RGB-Thermal 2D检测

融合RGB和热成像的2D目标检测，支持早期融合和晚期融合策略。

```bash
# 单独运行RGB-Thermal检测
python -c "
import yaml
from data.mineslam_dataset import MineSLAMDataset
from baselines.rgb_thermal_detection import run_rgb_thermal_detection_on_dataset

# 加载配置和数据集
with open('configs/mineslam.yaml', 'r') as f:
    config = yaml.safe_load(f)
dataset = MineSLAMDataset(config, split='test')

# 运行算法
detection_config = config.get('rgb_thermal_detection', {})
detection_file, results = run_rgb_thermal_detection_on_dataset(
    dataset, detection_config, './results/detection_test'
)
print(f'检测结果保存至: {detection_file}')
"
```

**参数配置**:
```yaml
rgb_thermal_detection:
  fusion_strategy: 'early'           # 'early' 或 'late'
  model_type: 'yolov8n'             # 'yolov8n' 或 'simplified'
  confidence_threshold: 0.5          # 置信度阈值
  camera_height: 1.5                # 相机高度假设(米)
```

### C. 完整基线评估

运行完整的基线评估，计算ATE、RPE、2D mAP、3D mAP@0.5等指标：

```bash
# 运行完整评估
python baseline_eval.py \
    --config configs/mineslam.yaml \
    --output_dir results/baseline_evaluation \
    --split test \
    --run_lidar \
    --run_detection \
    --gt_trajectory sr_B_route2_deep_learning/ground_truth_trajectory.csv
```

**评估输出**:
- `results/baseline_evaluation/baseline_evaluation_results.json`: 完整JSON结果
- `results/baseline_evaluation/evaluation_report.txt`: 文本报告
- `results/baseline_evaluation/evaluation_summary.png`: 可视化图表
- `results/baseline_evaluation/lidar_odometry/`: LiDAR里程计详细结果
- `results/baseline_evaluation/rgb_thermal_detection/`: 检测详细结果
## 配置文件说明

主配置文件`configs/mineslam.yaml`包含以下主要部分：

```yaml
# 数据配置
data:
  root: "./sr_B_route2_deep_learning"
  train_index: "./lists/train.jsonl"
  val_index: "./lists/val.jsonl"
  test_index: "./lists/test.jsonl"
  # ... 其他数据路径

# LiDAR里程计配置
lidar_odometry:
  voxel_size: 0.3
  max_correspondence_distance: 1.0
  # ... 其他参数

# RGB-Thermal检测配置  
rgb_thermal_detection:
  fusion_strategy: 'early'
  model_type: 'yolov8n'
  # ... 其他参数

# 反合成数据约束
anti_synthetic:
  enforce_real_data_only: true
  validate_file_existence: true
```

## 数据集统计

当前数据集包含14,790个真实样本：

- **时间跨度**: 1479秒 (~24.6分钟)
- **热成像覆盖率**: 100% (14,790样本)
- **RGB覆盖率**: 71.2% (10,537样本)  
- **LiDAR覆盖率**: 97.7% (14,450样本)
- **位姿覆盖率**: 99.9% (14,782样本)
- **深度覆盖率**: 0% (传感器故障)

## 评估指标

### 里程计指标
- **ATE (Absolute Trajectory Error)**: 绝对轨迹误差
  - ATE Mean/Median/RMSE: 平均/中位数/均方根误差
- **RPE (Relative Pose Error)**: 相对位姿误差  
  - RPE Mean/Median/RMSE: 帧间相对误差

### 检测指标  
- **2D mAP**: 2D检测平均精度
- **3D mAP@0.5**: 3D检测平均精度(IoU=0.5)
- **检测率**: 成功检测帧比例
- **平均检测数**: 每帧平均检测目标数

## 故障排除

### 常见问题

1. **ModuleNotFoundError**: 确保安装了所需依赖
```bash
pip install -r requirements.txt  # 如果有的话
```

2. **数据加载失败**: 检查数据路径和文件权限
```bash
ls -la sr_B_route2_deep_learning/
```

3. **内存不足**: 调整批大小或使用数据分批处理
```python
# 在配置中调整批大小
batch_size: 1  # 减少批大小
```

4. **GPU内存不足**: 强制使用CPU
```python
# 在代码中添加
import torch
torch.cuda.is_available = lambda: False
```

### 性能优化建议

1. **LiDAR处理优化**:
   - 增大体素大小减少点云密度
   - 调整ICP迭代次数平衡精度和速度

2. **检测模型优化**:
   - 使用简化模型代替YOLOv8
   - 调整输入图像分辨率

3. **数据加载优化**:
   - 使用多进程数据加载
   - 预计算和缓存重复操作

## 扩展开发

### 添加新的基线算法

1. 在`baselines/`目录创建新文件
2. 实现标准接口:
```python
def run_new_algorithm_on_dataset(dataset, config, output_dir):
    # 算法实现
    return result_file, metrics
```

3. 在`baseline_eval.py`中注册新算法

### 自定义评估指标

在`DetectionEvaluator`或`TrajectoryEvaluator`类中添加新的评估方法。

## 引用

如果使用此代码，请引用相关论文和数据集：

```bibtex
@article{mineslam2024,
  title={MineSLAM: Multi-modal Adaptive 3D Semantic SLAM for Degraded Mining Environments},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 许可证

本项目使用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请提交GitHub Issue或联系维护者。
# test
测试
