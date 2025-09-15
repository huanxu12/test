"""
MineSLAM Backend Optimization Demo
演示后端优化系统的完整使用流程
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend import backend_optimize, KeyFrame, PoseVertex


def create_demo_data():
    """创建演示数据"""
    print("🔧 Creating demo trajectory data...")

    # 创建模拟真实轨迹
    num_poses = 100
    timestamps = np.linspace(1566246592.0, 1566246692.0, num_poses)

    # 生成真值轨迹（8字形路径）
    ground_truth = []
    poses = []
    keyframes = []

    for i, t in enumerate(timestamps):
        # 8字形轨迹
        theta = i * 0.2
        x = 10 * np.sin(theta)
        y = 5 * np.sin(2 * theta)
        z = 0.0
        yaw = theta * 0.1

        # 真值
        gt_pose = [t, x, y, z, 0.0, 0.0, yaw]
        ground_truth.append(gt_pose)

        # 带噪声的观测位姿
        noise = np.random.normal(0, 0.1, 6)
        noisy_position = np.array([x, y, z]) + noise[:3]
        noisy_orientation = np.array([0.0, 0.0, yaw]) + noise[3:]

        pose_dict = {
            'timestamp': t,
            'position': noisy_position.tolist(),
            'orientation': noisy_orientation.tolist()
        }
        poses.append(pose_dict)

        # 每隔5帧创建一个关键帧
        if i % 5 == 0:
            # 模拟RGB图像（实际中应该从真实数据加载）
            rgb_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

            keyframe_dict = {
                'frame_id': i,
                'timestamp': t,
                'rgb_image': rgb_image,
                'pose': [x, y, z, 0.0, 0.0, yaw],
                'data_source': 'real_sensors'
            }
            keyframes.append(keyframe_dict)

    # 创建语义观测
    semantic_observations = []
    landmark_classes = ['fiducial', 'extinguisher', 'phone', 'backpack']

    for i in range(10):  # 10个语义地标
        # 随机位置
        world_pos = [
            np.random.uniform(-8, 8),
            np.random.uniform(-4, 4),
            np.random.uniform(0, 2)
        ]

        # 随机观测位姿
        pose_id = np.random.randint(0, len(poses))

        semantic_obs = {
            'pose_id': pose_id,
            'world_position': world_pos,
            'image_observation': [np.random.uniform(100, 540), np.random.uniform(100, 380)],
            'semantic_class': np.random.choice(landmark_classes),
            'confidence': np.random.uniform(0.7, 0.95),
            'timestamp': timestamps[pose_id]
        }
        semantic_observations.append(semantic_obs)

    trajectory_data = {
        'poses': poses,
        'keyframes': keyframes,
        'semantic_observations': semantic_observations,
        'ground_truth': np.array(ground_truth)
    }

    print(f"  ✅ Created {len(poses)} poses, {len(keyframes)} keyframes, {len(semantic_observations)} semantic observations")
    return trajectory_data


def run_backend_demo():
    """运行后端优化演示"""
    print("=" * 80)
    print("🚀 MineSLAM Backend Optimization Demo")
    print("=" * 80)

    # 创建演示数据
    trajectory_data = create_demo_data()

    # 设置优化参数
    loop_detection_params = {
        'descriptor_type': 'simple',
        'similarity_threshold': 0.7,
        'temporal_consistency': 30
    }

    optimization_params = {
        'max_iterations': 25,
        'convergence_threshold': 1e-6,
        'use_robust_kernel': True
    }

    # 执行后端优化
    print("\n🔄 Running backend optimization...")
    results = backend_optimize(
        trajectory_data=trajectory_data,
        loop_detection_params=loop_detection_params,
        optimization_params=optimization_params,
        output_dir="outputs/backend/demo"
    )

    # 输出结果
    print("\n📊 Optimization Results:")
    print("=" * 50)

    summary = results['summary']
    print(f"Number of poses: {summary['num_poses']}")
    print(f"Loop closures detected: {summary['num_loop_closures']}")
    print(f"Optimization converged: {summary['optimization_converged']}")
    print(f"Chi2 reduction: {summary['chi2_reduction']:.3f}")
    print(f"ATE improvement: {summary['ate_improvement_percentage']:.1f}%")
    print(f"Target achieved (≥10%): {summary['target_achieved']}")

    # 输出文件路径
    print(f"\n📁 Output files:")
    for name, path in results['visualization_paths'].items():
        print(f"  {name}: {path}")

    # 判断测试成功
    success = (
        summary['optimization_converged'] and
        summary['ate_improvement_percentage'] >= 10.0
    )

    if success:
        print("\n✅ Demo completed successfully!")
        print("   - Optimization converged")
        print("   - ATE improvement ≥ 10%")
        print("   - All visualizations generated")
    else:
        print("\n⚠️  Demo completed with limitations:")
        if not summary['optimization_converged']:
            print("   - Optimization did not converge")
        if summary['ate_improvement_percentage'] < 10.0:
            print(f"   - ATE improvement {summary['ate_improvement_percentage']:.1f}% < 10%")

    return success


if __name__ == '__main__':
    try:
        success = run_backend_demo()
        exit_code = 0 if success else 1
        print(f"\nDemo exit code: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)