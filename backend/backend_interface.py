"""
MineSLAM Backend Optimization Main Interface
MineSLAM后端优化主接口：backend_optimize()函数，集成完整流水线
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings
import json

from .graph_optimizer import PoseGraphOptimizer, PoseVertex
from .loop_detector import LoopDetector, KeyFrame
from .trajectory_evaluator import TrajectoryEvaluator, TrajectoryVisualizer
from .semantic_factors import SemanticFactor, SemanticObservation, LandmarkVertex


def backend_optimize(trajectory_data: Dict[str, Any],
                    loop_detection_params: Optional[Dict] = None,
                    optimization_params: Optional[Dict] = None,
                    output_dir: str = "outputs/backend") -> Dict[str, Any]:
    """
    MineSLAM后端优化主函数

    Args:
        trajectory_data: 轨迹数据字典，包含：
            - poses: List[Dict] 位姿序列
            - keyframes: List[Dict] 关键帧数据
            - semantic_observations: List[Dict] 语义观测
            - ground_truth: Optional[np.ndarray] 真值轨迹
        loop_detection_params: 回环检测参数
        optimization_params: 优化参数
        output_dir: 输出目录

    Returns:
        Dict包含优化结果、评估指标、可视化路径
    """

    # 设置默认参数
    if loop_detection_params is None:
        loop_detection_params = {
            'descriptor_type': 'simple',
            'similarity_threshold': 0.8,
            'temporal_consistency': 50
        }

    if optimization_params is None:
        optimization_params = {
            'max_iterations': 30,
            'convergence_threshold': 1e-6,
            'use_robust_kernel': True
        }

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting MineSLAM Backend Optimization...")
    print(f"Output directory: {output_path}")

    # 步骤1: 构建位姿图
    print("📊 Building pose graph...")
    optimizer = PoseGraphOptimizer()

    # 解析位姿数据
    poses = _parse_pose_data(trajectory_data['poses'])
    optimizer.add_pose_sequence(poses)

    print(f"  ✅ Added {len(poses)} poses to graph")

    # 记录优化前轨迹
    trajectory_before = optimizer.get_trajectory()

    # 步骤2: 回环检测
    print("🔍 Performing loop detection...")
    loop_detector = LoopDetector(**loop_detection_params)

    # 添加关键帧
    keyframes = _parse_keyframe_data(trajectory_data['keyframes'])
    valid_keyframes = 0

    for keyframe in keyframes:
        if loop_detector.add_keyframe(keyframe):
            valid_keyframes += 1

    print(f"  ✅ Added {valid_keyframes}/{len(keyframes)} valid keyframes")

    # 检测回环
    loop_closures = _detect_loop_closures(loop_detector, keyframes, poses)
    print(f"  🔗 Detected {len(loop_closures)} loop closures")

    # 添加回环约束
    for loop_closure in loop_closures:
        optimizer.add_loop_closure(
            from_id=loop_closure['from_id'],
            to_id=loop_closure['to_id'],
            relative_pose=loop_closure['relative_pose'],
            confidence=loop_closure['confidence']
        )

    # 步骤3: 语义约束
    if 'semantic_observations' in trajectory_data:
        print("🎯 Adding semantic constraints...")
        semantic_observations = _parse_semantic_data(trajectory_data['semantic_observations'])
        optimizer.add_semantic_constraints(semantic_observations)
        print(f"  ✅ Added {len(semantic_observations)} semantic constraints")

    # 步骤4: 图优化
    print("⚙️ Performing graph optimization...")
    optimization_result = optimizer.backend_optimize(
        max_iterations=optimization_params['max_iterations']
    )

    print(f"  ✅ Optimization converged: {optimization_result['converged']}")
    print(f"  📉 Chi2 reduction: {optimization_result['chi2_reduction']:.3f}")

    # 记录优化后轨迹
    trajectory_after = optimizer.get_trajectory()

    # 步骤5: 评估优化效果
    print("📈 Evaluating optimization results...")
    evaluator = TrajectoryEvaluator()

    # 如果有真值，计算ATE改进
    evaluation_results = {}
    if 'ground_truth' in trajectory_data and trajectory_data['ground_truth'] is not None:
        gt_trajectory = trajectory_data['ground_truth']
        evaluator.set_ground_truth(gt_trajectory)

        improvement = evaluator.evaluate_optimization_improvement(
            trajectory_before, trajectory_after
        )
        evaluation_results = improvement

        print(f"  📊 ATE improvement: {improvement['ate_improvement_percentage']:.1f}%")
        print(f"  🎯 Target achieved (≥10%): {improvement['improvement_achieved']}")
    else:
        print("  ⚠️  No ground truth available for ATE evaluation")

    # 步骤6: 可视化和保存结果
    print("🎨 Generating visualizations...")
    visualizer = TrajectoryVisualizer(str(output_path))

    visualization_paths = {}

    # 轨迹对比图
    if 'ground_truth' in trajectory_data and trajectory_data['ground_truth'] is not None:
        comparison_path = visualizer.plot_trajectory_comparison(
            trajectory_data['ground_truth'],
            trajectory_before,
            trajectory_after,
            save_path=output_path / "trajectory_comparison.png"
        )
        visualization_paths['trajectory_comparison'] = comparison_path

    # 3D地标图
    optimized_landmarks = optimization_result.get('optimized_landmarks', {})
    if optimized_landmarks:
        landmarks_path = visualizer.plot_landmarks_3d(
            optimized_landmarks,
            trajectory_after,
            save_path=output_path / "landmarks_3d.png"
        )
        visualization_paths['landmarks_3d'] = landmarks_path

    # 位姿不确定性
    pose_uncertainties = optimizer.get_pose_uncertainties()
    if pose_uncertainties:
        uncertainties_path = visualizer.plot_pose_uncertainties(
            trajectory_after,
            pose_uncertainties,
            save_path=output_path / "pose_uncertainties.png"
        )
        visualization_paths['pose_uncertainties'] = uncertainties_path

    # 保存评估报告
    report_path = visualizer.save_evaluation_report(
        evaluation_results,
        optimization_result,
        save_path=output_path / "optimization_report.json"
    )
    visualization_paths['evaluation_report'] = report_path

    print(f"  ✅ Visualizations saved to: {output_path}")

    # 返回完整结果
    results = {
        'optimization_result': optimization_result,
        'evaluation_results': evaluation_results,
        'trajectory_before': trajectory_before,
        'trajectory_after': trajectory_after,
        'loop_closures': loop_closures,
        'visualization_paths': visualization_paths,
        'output_directory': str(output_path),
        'summary': {
            'num_poses': len(poses),
            'num_loop_closures': len(loop_closures),
            'optimization_converged': optimization_result['converged'],
            'chi2_reduction': optimization_result['chi2_reduction'],
            'ate_improvement_percentage': evaluation_results.get('ate_improvement_percentage', 0.0),
            'target_achieved': evaluation_results.get('improvement_achieved', False)
        }
    }

    print("✅ Backend optimization completed successfully!")
    return results


def _parse_pose_data(poses_data: List[Dict]) -> List[PoseVertex]:
    """解析位姿数据"""
    poses = []

    for i, pose_dict in enumerate(poses_data):
        try:
            # 兼容不同格式
            if 'position' in pose_dict and 'orientation' in pose_dict:
                position = np.array(pose_dict['position'])
                orientation = np.array(pose_dict['orientation'])
            elif 'pose' in pose_dict:
                pose_array = np.array(pose_dict['pose'])
                position = pose_array[:3]
                orientation = pose_array[3:6] if len(pose_array) >= 6 else np.zeros(3)
            else:
                # 假设直接是6DoF数组
                pose_array = np.array(pose_dict)
                position = pose_array[:3]
                orientation = pose_array[3:6] if len(pose_array) >= 6 else np.zeros(3)

            pose = PoseVertex(
                id=i,
                timestamp=pose_dict.get('timestamp', i * 0.1),
                position=position,
                orientation=orientation
            )
            poses.append(pose)

        except Exception as e:
            warnings.warn(f"Failed to parse pose {i}: {e}")

    return poses


def _parse_keyframe_data(keyframes_data: List[Dict]) -> List[KeyFrame]:
    """解析关键帧数据"""
    keyframes = []

    for i, kf_dict in enumerate(keyframes_data):
        try:
            # 创建关键帧
            keyframe = KeyFrame(
                frame_id=kf_dict.get('frame_id', i),
                timestamp=kf_dict.get('timestamp', i * 0.1),
                rgb_image=kf_dict.get('rgb_image'),
                depth_image=kf_dict.get('depth_image'),
                thermal_image=kf_dict.get('thermal_image'),
                lidar_points=kf_dict.get('lidar_points'),
                pose=kf_dict.get('pose'),
                data_source=kf_dict.get('data_source', 'real_sensors')
            )
            keyframes.append(keyframe)

        except Exception as e:
            warnings.warn(f"Failed to parse keyframe {i}: {e}")

    return keyframes


def _detect_loop_closures(loop_detector: LoopDetector,
                         keyframes: List[KeyFrame],
                         poses: List[PoseVertex]) -> List[Dict]:
    """检测回环闭合"""
    loop_closures = []

    # 对每个关键帧检测回环
    for i, keyframe in enumerate(keyframes):
        if i < 50:  # 跳过前50帧以确保时间一致性
            continue

        try:
            candidates = loop_detector.detect_loop_closure(keyframe, k_candidates=3)

            for candidate in candidates:
                # 验证回环质量
                if candidate.similarity_score > 0.8 and candidate.geometric_verification:

                    # 计算相对位姿（简化版本）
                    from_pose = poses[candidate.match_id].to_se3()
                    to_pose = poses[candidate.query_id].to_se3()
                    relative_pose = from_pose.inverse() * to_pose

                    loop_closure = {
                        'from_id': candidate.match_id,
                        'to_id': candidate.query_id,
                        'relative_pose': relative_pose,
                        'confidence': candidate.similarity_score,
                        'similarity_score': candidate.similarity_score
                    }

                    loop_closures.append(loop_closure)

        except Exception as e:
            warnings.warn(f"Loop detection failed for keyframe {i}: {e}")

    return loop_closures


def _parse_semantic_data(semantic_data: List[Dict]) -> List[Dict]:
    """解析语义观测数据"""
    observations = []

    for obs_dict in semantic_data:
        try:
            observation = {
                'pose_id': obs_dict['pose_id'],
                'world_position': np.array(obs_dict['world_position']),
                'image_observation': np.array(obs_dict['image_observation']),
                'class': obs_dict['semantic_class'],
                'confidence': obs_dict['confidence'],
                'timestamp': obs_dict.get('timestamp', 0.0)
            }
            observations.append(observation)

        except Exception as e:
            warnings.warn(f"Failed to parse semantic observation: {e}")

    return observations