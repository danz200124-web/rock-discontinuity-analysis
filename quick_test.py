"""
快速测试脚本 - 绕过MKL问题直接测试新算法
"""

import numpy as np
import sys
import os

# 设置环境变量避免MKL问题
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

print("=" * 60)
print("岩石裂隙识别 - 新算法快速测试")
print("=" * 60)

# 生成测试数据
print("\n1. 生成测试数据...")
np.random.seed(42)

# 模拟3个平面的点云
n_points_per_plane = 1000
all_points = []
all_normals = []

for i in range(3):
    # 平面法向量
    normal = np.array([
        np.random.randn(),
        np.random.randn(),
        abs(np.random.randn())  # 朝上
    ])
    normal = normal / np.linalg.norm(normal)

    # 生成平面上的点
    x = np.random.uniform(-5, 5, n_points_per_plane)
    y = np.random.uniform(-5, 5, n_points_per_plane)
    d = np.random.uniform(-2, 2)
    z = -(normal[0] * x + normal[1] * y + d) / (normal[2] + 1e-6)

    points = np.column_stack([x, y, z])
    points += np.random.randn(*points.shape) * 0.02  # 添加噪声

    normals = np.tile(normal, (n_points_per_plane, 1))

    all_points.append(points)
    all_normals.append(normals)

all_points = np.vstack(all_points)
all_normals = np.vstack(all_normals)

print(f"✓ 生成了 {len(all_points)} 个点")

# 测试新算法
print("\n2. 测试HDBSCAN聚类算法...")
try:
    from src.discontinuity_detection.hdbscan_clusterer import HDBSCANClusterer

    clusterer = HDBSCANClusterer()
    labels = clusterer.cluster(all_points, all_normals, min_cluster_size=50)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"✓ HDBSCAN聚类成功!")
    print(f"  - 识别簇数: {n_clusters}")
    print(f"  - 噪声点数: {n_noise}")

    # 获取概率
    probs = clusterer.get_cluster_probabilities()
    if probs is not None:
        print(f"  - 平均置信度: {np.mean(probs):.3f}")

except ImportError as e:
    print(f"✗ HDBSCAN未安装: {e}")
    print("  请运行: pip install hdbscan")
except Exception as e:
    print(f"✗ HDBSCAN测试失败: {e}")

# 测试Mean-Shift
print("\n3. 测试Mean-Shift密度分析...")
try:
    from src.discontinuity_detection.meanshift_analyzer import MeanShiftPoleAnalyzer

    # 转换为极点
    poles = []
    for normal in all_normals[::10]:  # 采样减少计算量
        nx, ny, nz = normal / (np.linalg.norm(normal) + 1e-10)
        nz = abs(nz)
        if nz < 0.9999:
            x = nx / (1 + nz)
            y = ny / (1 + nz)
        else:
            x, y = 0, 0
        poles.append([x, y])
    poles = np.array(poles)

    analyzer = MeanShiftPoleAnalyzer()
    main_poles, labels = analyzer.analyze(poles, quantile=0.15)

    print(f"✓ Mean-Shift分析成功!")
    print(f"  - 识别主要组数: {len(main_poles)}")
    for i, pole in enumerate(main_poles):
        print(f"  - 组{i+1}: 倾向={pole['dip_direction']:.1f}°, 倾角={pole['dip']:.1f}°")

except ImportError as e:
    print(f"✗ scikit-learn未正确安装: {e}")
except Exception as e:
    print(f"✗ Mean-Shift测试失败: {e}")

# 测试MAGSAC++
print("\n4. 测试MAGSAC++平面拟合...")
try:
    from src.discontinuity_detection.magsac_plane_fitter import MAGSACPlaneFitter

    # 取第一个簇测试
    test_points = all_points[:n_points_per_plane]
    test_normals = all_normals[:n_points_per_plane]

    fitter = MAGSACPlaneFitter()
    plane_params, info = fitter.fit(test_points, test_normals)

    if plane_params is not None:
        print(f"✓ MAGSAC++拟合成功!")
        print(f"  - 内点比例: {info['inlier_ratio']:.2%}")
        print(f"  - 噪声水平σ: {info['sigma']:.4f}")
    else:
        print(f"✗ MAGSAC++拟合失败")

except Exception as e:
    print(f"✗ MAGSAC++测试失败: {e}")

# 测试Transformer (需要PyTorch)
print("\n5. 测试Transformer法向量精炼...")
try:
    import torch
    from src.point_cloud_processing.transformer_normal_refiner import TransformerNormalRefiner

    # 使用CPU避免GPU问题
    refiner = TransformerNormalRefiner(device='cpu')

    # 只取部分点测试
    test_size = 500
    refined_normals, confidence = refiner.refine_normals(
        all_points[:test_size],
        all_normals[:test_size],
        batch_size=500
    )

    print(f"✓ Transformer精炼成功!")
    print(f"  - 平均置信度: {np.mean(confidence):.3f}")
    print(f"  - 法向量变化量: {np.mean(np.linalg.norm(refined_normals - all_normals[:test_size], axis=1)):.4f}")

except ImportError as e:
    print(f"⚠ PyTorch未安装,跳过Transformer测试")
    print(f"  安装: pip install torch")
except Exception as e:
    print(f"✗ Transformer测试失败: {e}")

# 测试GNN (需要PyTorch)
print("\n6. 测试GNN裂隙精炼...")
try:
    import torch
    from src.discontinuity_detection.gnn_refiner import GNNDiscontinuityRefiner

    print("⚠ GNN需要构建裂隙图,跳过此测试")
    print("  (在完整流程中会自动使用)")

except ImportError:
    print(f"⚠ PyTorch未安装,跳过GNN测试")
except Exception as e:
    print(f"✗ GNN测试失败: {e}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n总结:")
print("- ✓ 表示算法可用")
print("- ✗ 表示需要修复")
print("- ⚠ 表示需要额外依赖")
print("\n建议:")
print("1. 安装缺失依赖: pip install hdbscan")
print("2. 如果PyTorch已安装,深度学习模块应该可用")
print("3. 运行完整流程请使用: python src/main.py")
