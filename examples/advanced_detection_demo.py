"""
高级算法集成示例
展示如何使用新引入的先进算法提升裂隙识别精度

新算法包括:
1. PointNet++ - 深度学习特征提取
2. HDBSCAN - 层级密度聚类
3. Transformer - 法向量精炼
4. MAGSAC++ - 自适应RANSAC
5. GNN - 图神经网络关系建模
6. Mean-Shift - 自适应密度峰检测
"""

import numpy as np
import open3d as o3d
import logging
from typing import Dict, List

# 导入新算法模块
from point_cloud_processing.pointnet2_feature_extractor import PointNet2Wrapper
from point_cloud_processing.transformer_normal_refiner import TransformerNormalRefiner
from discontinuity_detection.hdbscan_clusterer import HDBSCANClusterer, AdaptiveHDBSCAN
from discontinuity_detection.magsac_plane_fitter import MAGSACPlaneFitter, AdaptiveRANSAC
from discontinuity_detection.gnn_refiner import GNNDiscontinuityRefiner
from discontinuity_detection.meanshift_analyzer import MeanShiftPoleAnalyzer, AdaptiveDensityAnalyzer

# 导入原有模块
from point_cloud_processing.loader import PointCloudLoader
from point_cloud_processing.preprocessing import PointCloudPreprocessor
from point_cloud_processing.normal_estimation import NormalEstimator

logger = logging.getLogger(__name__)


class AdvancedDiscontinuityDetector:
    """
    高级裂隙检测器
    集成最新算法的完整流程
    """

    def __init__(self, use_gpu: bool = True, use_advanced: bool = True):
        """
        参数:
            use_gpu: 是否使用GPU加速
            use_advanced: 是否使用高级算法(False则使用传统算法)
        """
        self.use_gpu = use_gpu
        self.use_advanced = use_advanced

        # 初始化各模块
        self.loader = PointCloudLoader()
        self.preprocessor = PointCloudPreprocessor()
        self.normal_estimator = NormalEstimator()

        if use_advanced:
            logger.info("🚀 启用高级算法模式")
            # 深度学习模块(需要GPU)
            if use_gpu:
                self.pointnet2 = PointNet2Wrapper(device='cuda')
                self.transformer_refiner = TransformerNormalRefiner(device='cuda')
            else:
                logger.warning("GPU未启用,跳过深度学习模块")
                self.pointnet2 = None
                self.transformer_refiner = None

            # 高级聚类
            self.adaptive_hdbscan = AdaptiveHDBSCAN()

            # 高级平面拟合
            self.adaptive_ransac = AdaptiveRANSAC()

            # 图神经网络
            if use_gpu:
                self.gnn_refiner = GNNDiscontinuityRefiner(device='cuda')
            else:
                self.gnn_refiner = None

            # Mean-Shift密度分析
            self.density_analyzer = AdaptiveDensityAnalyzer()
        else:
            logger.info("使用传统算法模式")

    def detect(self, point_cloud_path: str, config: Dict = None) -> Dict:
        """
        完整的裂隙检测流程

        参数:
            point_cloud_path: 点云文件路径
            config: 配置参数

        返回:
            results: 检测结果字典
        """
        if config is None:
            config = self._get_default_config()

        logger.info("=" * 60)
        logger.info("高级裂隙检测流程启动")
        logger.info("=" * 60)

        # 步骤1: 加载点云
        logger.info("\n【步骤1/7】加载点云...")
        point_cloud = self.loader.load(point_cloud_path)
        points = np.asarray(point_cloud.points)
        logger.info(f"点云大小: {len(points)} 个点")

        # 步骤2: 预处理
        logger.info("\n【步骤2/7】点云预处理...")
        point_cloud = self.preprocessor.downsample(point_cloud, voxel_size=config['voxel_size'])
        point_cloud = self.preprocessor.remove_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(point_cloud.points)
        logger.info(f"预处理后: {len(points)} 个点")

        # 步骤3: 法向量估计
        logger.info("\n【步骤3/7】法向量估计...")
        normals = self.normal_estimator.estimate(
            point_cloud,
            search_radius=config['normal_radius'],
            max_nn=config['normal_max_nn']
        )

        # 步骤3.5: Transformer法向量精炼(高级模式)
        if self.use_advanced and self.transformer_refiner is not None:
            logger.info("\n【步骤3.5/7】Transformer法向量精炼...")
            normals, confidence = self.transformer_refiner.refine_normals(points, normals)
            logger.info(f"平均置信度: {np.mean(confidence):.3f}")

        # 步骤4: 极点投影
        logger.info("\n【步骤4/7】立体投影...")
        poles = self._normals_to_poles(normals)

        # 步骤5: 密度分析识别主要组
        logger.info("\n【步骤5/7】密度分析识别主要不连续面组...")
        if self.use_advanced:
            # Mean-Shift自适应分析
            main_poles, pole_labels = self.density_analyzer.analyze(
                poles,
                method='meanshift',
                quantile=config['meanshift_quantile']
            )
        else:
            # 使用Mean-Shift作为替代（旧KDE已废弃）
            logger.warning("传统KDE已废弃，使用Mean-Shift替代")
            main_poles, pole_labels = self.density_analyzer.analyze(
                poles,
                method='meanshift',
                quantile=config['meanshift_quantile']
            )

        logger.info(f"识别到 {len(main_poles)} 个主要不连续面组")

        # 步骤6: 聚类识别单个裂隙
        logger.info("\n【步骤6/7】聚类识别单个裂隙...")
        discontinuities = []

        for i, pole in enumerate(main_poles):
            logger.info(f"\n处理第 {i+1} 组...")

            # 分配点到该组
            group_mask = self._assign_points_to_pole(
                normals, pole,
                angle_threshold=config['angle_threshold']
            )
            group_points = points[group_mask]
            group_normals = normals[group_mask]

            if len(group_points) < config['min_cluster_size']:
                continue

            # 聚类
            if self.use_advanced:
                # HDBSCAN自适应聚类
                clusterer = HDBSCANClusterer()
                labels = clusterer.cluster(
                    group_points, group_normals,
                    min_cluster_size=config['min_cluster_size'],
                    min_samples=config['min_samples']
                )
            else:
                # 使用HDBSCAN作为替代（旧DBSCAN已废弃）
                logger.warning("传统DBSCAN已废弃，使用HDBSCAN替代")
                clusterer = HDBSCANClusterer()
                labels = clusterer.cluster(
                    group_points, group_normals,
                    min_cluster_size=config['min_cluster_size'],
                    min_samples=config['min_samples']
                )

            # 平面拟合
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:
                    continue

                cluster_mask = labels == label
                cluster_points = group_points[cluster_mask]

                if len(cluster_points) < 10:
                    continue

                # 拟合平面
                if self.use_advanced:
                    # MAGSAC++自适应拟合
                    plane_params, info = self.adaptive_ransac.fit(
                        cluster_points,
                        normals=group_normals[cluster_mask],
                        method='magsac'
                    )
                    if plane_params is None:
                        continue
                else:
                    # 传统RANSAC
                    from discontinuity_detection.plane_fitting import PlaneFitter
                    fitter = PlaneFitter()
                    plane_params = fitter.fit(cluster_points, method='ransac')
                    info = {}

                # 保存裂隙信息
                discontinuity = {
                    'set_id': i + 1,
                    'cluster_id': label,
                    'points': cluster_points,
                    'plane_params': plane_params,
                    'normal': plane_params[:3],
                    'orientation': self._normal_to_orientation(plane_params[:3]),
                    'num_points': len(cluster_points),
                    'fit_info': info
                }
                discontinuities.append(discontinuity)

        logger.info(f"\n检测到 {len(discontinuities)} 个裂隙")

        # 步骤7: GNN精炼(高级模式)
        if self.use_advanced and self.gnn_refiner is not None and len(discontinuities) > 0:
            logger.info("\n【步骤7/7】GNN图神经网络精炼...")
            refined_groups = self.gnn_refiner.refine_discontinuities(
                discontinuities,
                similarity_threshold=config['gnn_threshold']
            )
            logger.info(f"精炼后: {len(refined_groups)} 个裂隙组")
        else:
            refined_groups = None

        # 汇总结果
        results = {
            'point_cloud': point_cloud,
            'points': points,
            'normals': normals,
            'main_poles': main_poles,
            'discontinuities': discontinuities,
            'refined_groups': refined_groups,
            'n_discontinuities': len(discontinuities),
            'config': config
        }

        logger.info("\n" + "=" * 60)
        logger.info("✅ 高级裂隙检测完成!")
        logger.info(f"主要组数: {len(main_poles)}")
        logger.info(f"裂隙总数: {len(discontinuities)}")
        logger.info("=" * 60)

        return results

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'voxel_size': 0.05,
            'normal_radius': 0.3,
            'normal_max_nn': 30,
            'meanshift_quantile': 0.15,
            'angle_threshold': 30,
            'min_cluster_size': 50,
            'min_samples': 10,
            'eps': 0.1,
            'gnn_threshold': 0.7
        }

    def _normals_to_poles(self, normals: np.ndarray) -> np.ndarray:
        """法向量转立体投影极点"""
        poles = []
        for normal in normals:
            nx, ny, nz = normal / (np.linalg.norm(normal) + 1e-10)
            nz = abs(nz)

            # 立体投影
            if nz < 0.9999:
                x = nx / (1 + nz)
                y = ny / (1 + nz)
            else:
                x, y = 0, 0

            poles.append([x, y])

        return np.array(poles)

    def _assign_points_to_pole(self, normals: np.ndarray, pole: Dict,
                               angle_threshold: float) -> np.ndarray:
        """分配点到极点"""
        pole_normal = self._orientation_to_normal(
            pole['dip_direction'],
            pole['dip']
        )

        angles = np.arccos(np.clip(np.abs(np.dot(normals, pole_normal)), -1, 1))
        angles_deg = np.degrees(angles)

        return angles_deg < angle_threshold

    def _orientation_to_normal(self, dip_direction: float, dip: float) -> np.ndarray:
        """产状转法向量"""
        dd_rad = np.radians(dip_direction)
        dip_rad = np.radians(dip)

        nx = np.sin(dip_rad) * np.sin(dd_rad)
        ny = np.sin(dip_rad) * np.cos(dd_rad)
        nz = np.cos(dip_rad)

        return np.array([nx, ny, nz])

    def _normal_to_orientation(self, normal: np.ndarray) -> Dict:
        """法向量转产状"""
        nx, ny, nz = normal / np.linalg.norm(normal)

        dip_direction = np.degrees(np.arctan2(nx, ny))
        if dip_direction < 0:
            dip_direction += 360

        dip = np.degrees(np.arccos(np.abs(nz)))

        return {'dip_direction': dip_direction, 'dip': dip}


def main():
    """主函数 - 使用示例"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建高级检测器
    detector = AdvancedDiscontinuityDetector(
        use_gpu=True,      # 启用GPU加速
        use_advanced=True  # 启用高级算法
    )

    # 执行检测
    results = detector.detect(
        point_cloud_path='data/rock_outcrop.ply',
        config={
            'voxel_size': 0.05,
            'normal_radius': 0.3,
            'meanshift_quantile': 0.15,  # Mean-Shift带宽
            'angle_threshold': 30,
            'min_cluster_size': 50,
            'gnn_threshold': 0.7
        }
    )

    # 输出结果
    print("\n检测结果摘要:")
    print(f"主要不连续面组: {len(results['main_poles'])} 个")
    print(f"识别的裂隙: {results['n_discontinuities']} 个")

    # 详细信息
    for i, disc in enumerate(results['discontinuities'][:5]):  # 只显示前5个
        print(f"\n裂隙 {i+1}:")
        print(f"  组ID: {disc['set_id']}")
        print(f"  点数: {disc['num_points']}")
        print(f"  倾向: {disc['orientation']['dip_direction']:.1f}°")
        print(f"  倾角: {disc['orientation']['dip']:.1f}°")


if __name__ == '__main__':
    main()
