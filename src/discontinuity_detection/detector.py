"""
不连续面检测主模块 - 使用高级算法
整合HDBSCAN、MAGSAC++、GNN等先进算法
"""

import numpy as np
import logging
from .hdbscan_clusterer import HDBSCANClusterer, AdaptiveHDBSCAN
from .magsac_plane_fitter import MAGSACPlaneFitter, AdaptiveRANSAC

logger = logging.getLogger(__name__)


class DiscontinuityDetector:
    """
    不连续面检测器 - 高级算法版本
    使用HDBSCAN聚类和MAGSAC++平面拟合
    """

    def __init__(self, use_adaptive=False):
        """
        初始化检测器

        参数:
            use_adaptive: 是否使用自适应算法(目前禁用,保持接口一致)
        """
        # 使用标准HDBSCAN聚类器
        self.clusterer = HDBSCANClusterer()
        self.plane_fitter = MAGSACPlaneFitter()
        logger.info("使用高级算法: HDBSCAN + MAGSAC++")

    def detect(self, point_cloud, normals, main_poles,
               min_cluster_size=50, min_samples=10, angle_threshold=30):
        """
        检测不连续面

        参数:
            point_cloud: 点云对象
            normals: 法向量数组
            main_poles: 主要极点列表
            min_cluster_size: HDBSCAN最小簇大小 (自动确定簇数)
            min_samples: HDBSCAN最小样本数
            angle_threshold: 角度阈值（度）

        返回:
            discontinuity_sets: 不连续面组列表
        """
        logger.info("开始检测不连续面 (使用高级算法)...")
        logger.info(f"参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}, angle_threshold={angle_threshold}°")

        points = np.asarray(point_cloud.points)
        discontinuity_sets = []

        # 创建原始点云索引数组
        original_indices = np.arange(len(points))

        # 统计分配情况
        all_assigned_mask = np.zeros(len(points), dtype=bool)
        total_noise_points = 0

        # 对每个主要极点进行处理
        for i, pole in enumerate(main_poles):
            logger.info(f"处理第 {i + 1} 组...")

            # 获取属于该组的点
            group_mask = self._assign_points_to_pole(
                normals, pole, angle_threshold
            )

            group_points_count = np.sum(group_mask)
            logger.info(f"第 {i + 1} 组: 分配了 {group_points_count} 个点 (占总点数 {group_points_count/len(points)*100:.1f}%)")

            # 记录分配的点
            all_assigned_mask |= group_mask

            if group_points_count < min_cluster_size:
                logger.warning(f"第 {i + 1} 组点数太少({group_points_count} < {min_cluster_size})，跳过")
                continue

            group_points = points[group_mask]
            group_normals = normals[group_mask]
            group_indices = original_indices[group_mask]

            # HDBSCAN聚类分析 - 自动确定聚类数
            logger.info(f"使用HDBSCAN进行层级密度聚类...")
            clusters = self.clusterer.cluster(
                group_points,
                group_normals,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )

            # 获取聚类概率和离群分数
            probabilities = self.clusterer.get_cluster_probabilities()
            outlier_scores = self.clusterer.get_outlier_scores()

            # 对每个聚类进行MAGSAC++平面拟合
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters[unique_clusters != -1])
            n_noise = np.sum(clusters == -1)
            total_noise_points += n_noise

            logger.info(f"第 {i + 1} 组聚类结果: {n_clusters} 个聚类, {n_noise} 个噪声点 (占该组 {n_noise/group_points_count*100:.1f}%)")

            if probabilities is not None:
                valid_probs = probabilities[clusters != -1]
                if len(valid_probs) > 0:
                    logger.info(f"聚类置信度: 平均={np.mean(valid_probs):.3f}, 最小={np.min(valid_probs):.3f}")

            for cluster_id in unique_clusters:
                if cluster_id == -1:  # 噪声点
                    continue

                cluster_mask = clusters == cluster_id
                cluster_points = group_points[cluster_mask]
                cluster_normals = group_normals[cluster_mask]
                cluster_indices = group_indices[cluster_mask]

                # 确保cluster_points只包含3D坐标
                if cluster_points.ndim == 2 and cluster_points.shape[1] != 3:
                    logger.warning(f"聚类点维度异常: {cluster_points.shape}，取前3列")
                    cluster_points = cluster_points[:, :3]

                # MAGSAC++自适应平面拟合
                plane_params, fit_info = self.plane_fitter.fit(
                    cluster_points,
                    normals=cluster_normals
                )

                # 如果平面拟合失败，跳过该聚类
                if plane_params is None:
                    logger.warning(f"第{i+1}组聚类{cluster_id}平面拟合失败，跳过")
                    continue

                # 记录拟合质量
                if 'inlier_ratio' in fit_info:
                    logger.info(f"  聚类{cluster_id}: {len(cluster_points)}点, 内点率={fit_info['inlier_ratio']:.2%}, σ={fit_info.get('sigma', 0):.4f}")

                # 计算聚类质量指标
                cluster_quality = {
                    'mean_probability': np.mean(probabilities[cluster_mask]) if probabilities is not None else None,
                    'mean_outlier_score': np.mean(outlier_scores[cluster_mask]) if outlier_scores is not None else None,
                    'inlier_ratio': fit_info.get('inlier_ratio', None),
                    'fit_sigma': fit_info.get('sigma', None)
                }

                # 计算不连续面属性
                discontinuity = {
                    'set_id': i + 1,
                    'cluster_id': cluster_id,
                    'points': cluster_points,
                    'indices': cluster_indices,
                    'plane_params': plane_params,
                    'normal': plane_params[:3],
                    'orientation': self._normal_to_orientation(plane_params[:3]),
                    'num_points': len(cluster_points),
                    'quality': cluster_quality,  # 添加质量指标
                    'fit_info': fit_info  # MAGSAC++拟合信息
                }

                discontinuity_sets.append(discontinuity)

        # 计算实际的唯一点数（去重）
        unique_indices = set()
        for disc in discontinuity_sets:
            unique_indices.update(disc['indices'])
        total_unique_points = len(unique_indices)

        total_assigned = np.sum(all_assigned_mask)

        logger.info(f"检测完成，共找到 {len(discontinuity_sets)} 个不连续面")
        logger.info(f"===== 检测统计 =====")
        logger.info(f"总点数: {len(points)}")
        logger.info(f"分配到组的点数: {total_assigned} ({total_assigned/len(points)*100:.1f}%)")
        logger.info(f"未分配的点数: {len(points) - total_assigned} ({(len(points) - total_assigned)/len(points)*100:.1f}%)")
        logger.info(f"聚类噪声点数: {total_noise_points} ({total_noise_points/len(points)*100:.1f}%)")
        logger.info(f"包含在不连续面中的点数（含重复）: {sum(len(d['indices']) for d in discontinuity_sets)}")
        logger.info(f"包含在不连续面中的唯一点数: {total_unique_points} ({total_unique_points/len(points)*100:.1f}%)")
        logger.info(f"==================")

        return discontinuity_sets

    def _assign_points_to_pole(self, normals, pole, angle_threshold):
        """
        将点分配到最近的极点

        参数:
            normals: 法向量数组
            pole: 极点信息
            angle_threshold: 角度阈值（度）

        返回:
            mask: 布尔掩码
        """
        # 从极点信息中提取法向量
        pole_normal = self._orientation_to_normal(
            pole['dip_direction'],
            pole['dip']
        )

        # 计算夹角
        angles = np.arccos(np.clip(np.abs(np.dot(normals, pole_normal)), -1, 1))
        angles_deg = np.degrees(angles)

        # 返回角度小于阈值的点
        return angles_deg < angle_threshold

    def _orientation_to_normal(self, dip_direction, dip):
        """产状转换为法向量"""
        dd_rad = np.radians(dip_direction)
        dip_rad = np.radians(dip)

        nx = np.sin(dip_rad) * np.sin(dd_rad)
        ny = np.sin(dip_rad) * np.cos(dd_rad)
        nz = np.cos(dip_rad)

        return np.array([nx, ny, nz])

    def _normal_to_orientation(self, normal):
        """法向量转换为产状"""
        nx, ny, nz = normal / np.linalg.norm(normal)

        # 计算倾向
        dip_direction = np.degrees(np.arctan2(nx, ny))
        if dip_direction < 0:
            dip_direction += 360

        # 计算倾角
        dip = np.degrees(np.arccos(np.abs(nz)))

        return {
            'dip_direction': dip_direction,
            'dip': dip
        }
