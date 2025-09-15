# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

本代码库包含 **MineSLAM** 项目 - 面向矿井退化环境的多模态自适应三维语义 SLAM 系统。该系统设计用于在极端矿井场景中工作（高粉尘、低光照、重复拓扑、多传感器退化），同时保持鲁棒的 6-DoF 位姿估计、高精度三维语义地图构建和实时人员检测。

## 代码库结构

主项目位于 `sr_B_route2_deep_learning/` 目录，包含来自 DARPA SubT（地下）隧道赛道的综合机器人数据集。这是基于 ROS 的数据集，在安全研究（SR）课程中通过遥操作在 Clearpath Husky 机器人上收集。

### 关键目录

- **`images/`** - 多模态传感器数据：
  - `boson_thermal/` - FLIR Boson 热红外图像（640x512，16位）
  - `multisense_left_color/` - 左立体相机图像（1024x544 PNG）
  - `multisense_right/` - 右立体相机图像
  - `multisense_depth/` - 立体视觉压缩深度图像（约1Hz）

- **`pointclouds/`** - 激光雷达数据：
  - `multisense_lidar/` - 来自 MultiSense 的 Hokuyo 激光雷达点云
  - `ouster_lidar/` - Ouster 激光雷达点云（约10Hz，每旋转一次一个点云）
  - `assembled_cloud/` - 带里程计位姿的内部建图系统点云

- **`calibration/`** - JSON格式的相机标定参数
- **`timestamps/`** - 时间同步数据
- **`imu/`** - IMU传感器数据（Microstrain GX5-25）
- **`odometry/`** - 机器人里程计数据
- **`control/`** - 机器人控制命令

### 数据文件

- **`ground_truth_trajectory.csv`** - 包含时间戳和6-DoF位姿的完整真值轨迹
- **`ground_truth_stats.json`** - 数据集统计信息（15,426个位姿，1,479秒，838米轨迹）
- **`sr_B_route2.bag.artifacts`** - 标注的人工制品位置（基准标志、灭火器、电话、背包、幸存者、钻头）
- **`fiducial_sr`** - 用于DARPA坐标系对齐的基准地标位置

## 数据使用和ROS集成

此数据集设计用于ROS（机器人操作系统）。`usage.txt` 文件包含详细说明：

1. **设置**：从引用仓库克隆 stix_ws catkin 工作空间
2. **构建**：使用 catkin 构建系统，正确扩展工作空间
3. **回放**：使用 roslaunch 与不同建图系统的特定参数

### 关键ROS启动参数

- `bag`：指定要打开的bag文件（需要相对路径）
- `rate`：Bag回放速率倍数（默认：2.0）
- `course`：安全研究用"sr"或实验用"ex"
- `config`："A"或"B"（所有当前bag都是配置B）
- `reproject`：启用Ouster点云重投影
- `mark_artifacts`：启用人工制品位置编码界面

### 支持的建图系统

- `omnimapper`：内部建图系统
- `cartographer`：Google Cartographer SLAM
- `odom_only`：无建图，静态地图到里程计变换

## 传感器话题和数据类型

数据集包含具有精确时间同步的丰富多模态传感器数据：

- **RGB/热成像相机**：标定的立体对 + 热成像
- **激光雷达**：双激光雷达系统（Hokuyo + Ouster）用于不同扫描模式
- **IMU**：带加速度计/陀螺仪数据的高频惯性测量
- **里程计**：来自Husky平台的滤波和原始里程计
- **WiFi**：用于基于无线电定位的信号强度扫描

## MineSLAM架构

项目实现四层架构：

1. **模态编码层**：从RGB/深度/热成像/激光雷达/IMU进行多尺度特征提取
2. **MoE融合层**：带门控网络和热成像引导注意力的专家混合
3. **任务解码层**：位姿估计和3D物体检测头
4. **后端优化**：带语义因子的g2o因子图

## 开发环境

- **训练平台**：RTX 4090（24GB），i9 CPU，128GB RAM
- **部署目标**：Jetson AGX Orin，30W功耗模式，TensorRT推理
- **性能要求**：<100ms推理延迟，≥10Hz帧率，<200MB模型大小

## 数据集引用和使用

此数据集由陆军研究实验室代表DARPA收集，以支持通过离线组件测试进行系统开发。使用此数据集时，请参考相关的ICRA论文获取更多技术细节。

数据代表退化环境中的真实机器人挑战，特别适用于：
- 多模态SLAM算法开发
- 挑战条件下的传感器融合
- 传感器退化下的鲁棒性测试
- 地下环境中的语义物体检测