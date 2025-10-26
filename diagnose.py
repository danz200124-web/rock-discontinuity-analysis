"""
诊断脚本：分析为什么检测覆盖率低
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

import numpy as np
import json
from pathlib import Path
from point_cloud_processing.loader import PointCloudLoader
from point_cloud_processing.preprocessing import PointCloudPreprocessor
from point_cloud_processing.normal_estimation import NormalEstimator
from discontinuity_detection.stereonet import StereonetAnalyzer
from discontinuity_detection.meanshift_analyzer import AdaptiveDensityAnalyzer

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 加载点云
print("加载点云...")
loader = PointCloudLoader()
point_cloud = loader.load('data/test.txt')
print(f"原始点数: {len(point_cloud.points)}")

# 预处理
print("\n预处理...")
preprocessor = PointCloudPreprocessor()
if config["preprocessing"]["voxel_size"] > 0:
    point_cloud = preprocessor.voxel_downsample(
        point_cloud,
        config["preprocessing"]["voxel_size"]
    )
print(f"下采样后点数: {len(point_cloud.points)}")

# 估计法向量
print("\n估计法向量...")
normal_estimator = NormalEstimator()
normals = normal_estimator.estimate(
    point_cloud,
    search_radius=config["normal_estimation"]["search_radius"],
    max_nn=config["normal_estimation"]["max_nn"]
)

# 立体投影分析
print("\n立体投影分析...")
stereonet_analyzer = StereonetAnalyzer()
poles = stereonet_analyzer.compute_poles(normals)

# Mean-Shift分析（替代旧的KDE）
print("\nMean-Shift密度分析...")
density_analyzer = AdaptiveDensityAnalyzer()
main_poles, pole_labels = density_analyzer.analyze(
    poles,
    method='meanshift',
    quantile=0.15
)

print(f"\n检测到 {len(main_poles)} 个主要极点:")
for i, pole in enumerate(main_poles):
    print(f"  极点 {i+1}: 倾向={pole['dip_direction']:.1f}°, 倾角={pole['dip']:.1f}°")

# 测试不同的角度阈值
print("\n测试不同角度阈值的点分配情况:")
from discontinuity_detection.detector import DiscontinuityDetector

detector = DiscontinuityDetector()
points = np.asarray(point_cloud.points)

for angle_threshold in [15, 20, 30, 45, 60]:
    total_assigned = 0
    for pole in main_poles:
        mask = detector._assign_points_to_pole(normals, pole, angle_threshold)
        total_assigned += np.sum(mask)

    # 去重（有些点可能被分配到多个组）
    all_assigned = np.zeros(len(points), dtype=bool)
    for pole in main_poles:
        mask = detector._assign_points_to_pole(normals, pole, angle_threshold)
        all_assigned |= mask

    unique_assigned = np.sum(all_assigned)
    print(f"  角度阈值={angle_threshold}°: 分配了 {unique_assigned}/{len(points)} 个点 ({unique_assigned/len(points)*100:.1f}%)")

print("\n诊断完成！")
