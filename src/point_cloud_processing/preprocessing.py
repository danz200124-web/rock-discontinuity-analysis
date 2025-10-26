"""
点云预处理模块
包括下采样、去噪、滤波等操作
"""

import numpy as np
import open3d as o3d
import logging

logger = logging.getLogger(__name__)


class PointCloudPreprocessor:
    """点云预处理器"""

    def __init__(self):
        pass

    def voxel_downsample(self, point_cloud, voxel_size):
        """
        体素下采样

        参数:
            point_cloud: Open3D点云对象
            voxel_size: 体素大小（米）

        返回:
            downsampled: 下采样后的点云
        """
        logger.info(f"体素下采样，体素大小: {voxel_size}m")

        downsampled = point_cloud.voxel_down_sample(voxel_size)

        reduction_rate = 1 - len(downsampled.points) / len(point_cloud.points)
        logger.info(f"下采样完成，点数减少 {reduction_rate:.1%}")

        return downsampled

    def remove_outliers(self, point_cloud, nb_neighbors=20, std_ratio=2.0):
        """
        移除离群点（统计滤波）

        参数:
            point_cloud: Open3D点云对象
            nb_neighbors: 邻近点数量
            std_ratio: 标准差倍数阈值

        返回:
            filtered: 滤波后的点云
        """
        logger.info(f"统计滤波去除离群点...")

        filtered, ind = point_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        outliers = len(point_cloud.points) - len(filtered.points)
        logger.info(f"移除 {outliers} 个离群点")

        return filtered

    def remove_radius_outliers(self, point_cloud, nb_points=16, radius=0.5):
        """
        半径滤波去除离群点

        参数:
            point_cloud: Open3D点云对象
            nb_points: 半径内最少点数
            radius: 搜索半径（米）

        返回:
            filtered: 滤波后的点云
        """
        logger.info(f"半径滤波去除离群点...")

        filtered, ind = point_cloud.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )

        outliers = len(point_cloud.points) - len(filtered.points)
        logger.info(f"移除 {outliers} 个离群点")

        return filtered

    def crop_roi(self, point_cloud, min_bound, max_bound):
        """
        裁剪感兴趣区域

        参数:
            point_cloud: Open3D点云对象
            min_bound: 最小边界 [x_min, y_min, z_min]
            max_bound: 最大边界 [x_max, y_max, z_max]

        返回:
            cropped: 裁剪后的点云
        """
        logger.info(f"裁剪ROI区域...")

        # 创建边界框
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )

        # 裁剪点云
        cropped = point_cloud.crop(bbox)

        logger.info(f"裁剪完成，保留 {len(cropped.points)} 个点")

        return cropped

    def estimate_and_orient_normals(self, point_cloud, radius=0.3):
        """
        估计并调整法向量方向

        参数:
            point_cloud: Open3D点云对象
            radius: 搜索半径

        返回:
            point_cloud: 包含法向量的点云
        """
        logger.info(f"估计法向量...")

        # 估计法向量
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=30
            )
        )

        # 调整法向量方向（使其一致）
        point_cloud.orient_normals_consistent_tangent_plane(30)

        logger.info(f"法向量估计完成")

        return point_cloud