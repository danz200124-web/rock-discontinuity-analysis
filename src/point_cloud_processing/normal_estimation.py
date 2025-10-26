"""
法向量估计模块
实现鲁棒的点云法向量估计算法，支持多进程并行计算
极致性能优化版 - 160GB内存
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree  # 使用cKDTree替代sklearn KDTree (快3-5倍)
from scipy.linalg import svd
import logging
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import sys

logger = logging.getLogger(__name__)


class NormalEstimator:
    """
    点云法向量估计器
    使用Open3D加速 + 自定义迭代重加权(可选)
    极致性能优化版
    """

    def __init__(self):
        self.normals = None
        self.kdtree = None

    def estimate(self, point_cloud, search_radius=0.3, max_nn=30, use_parallel=True, n_jobs=None):
        """
        估计点云法向量（极致性能优化版）

        参数:
            point_cloud: Open3D点云对象
            search_radius: 搜索半径（米）
            max_nn: 最大邻近点数
            use_parallel: 是否使用并行计算（默认True）
            n_jobs: 使用的CPU核心数，None表示使用所有核心

        返回:
            normals: 法向量数组 (N, 3)
        """
        points = np.asarray(point_cloud.points)
        n_points = points.shape[0]

        logger.info(f"开始估计 {n_points} 个点的法向量...")

        # 方法1: 使用Open3D内置高性能法向量估计 (最快)
        logger.info("使用Open3D高性能法向量估计...")
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=max_nn
            )
        )

        # 法向量一致性调整
        point_cloud.orient_normals_consistent_tangent_plane(k=15)

        self.normals = np.asarray(point_cloud.normals)

        # 确保法向量朝上
        z_components = self.normals[:, 2]
        flip_mask = z_components < 0
        self.normals[flip_mask] *= -1

        logger.info("法向量估计完成")
        return self.normals

    def _iterative_reweighted_fitting(self, points, center, max_iter=5):
        """
        迭代重加权平面拟合

        参数:
            points: 邻近点坐标
            center: 中心点坐标
            max_iter: 最大迭代次数

        返回:
            normal: 拟合平面法向量
        """
        # 中心化
        centered_points = points - center

        # 初始权重（均匀）
        weights = np.ones(len(points))

        for _ in range(max_iter):
            # 加权协方差矩阵
            weighted_cov = np.dot(
                centered_points.T * weights,
                centered_points
            ) / np.sum(weights)

            # SVD分解
            _, _, vt = svd(weighted_cov)
            normal = vt[2]  # 最小特征值对应的特征向量

            # 更新权重（基于残差）
            residuals = np.abs(np.dot(centered_points, normal))
            sigma = np.median(residuals) * 1.4826  # 鲁棒标准差估计

            if sigma > 0:
                weights = np.exp(-(residuals / (2 * sigma)) ** 2)
            else:
                break

        # 确保法向量朝上
        if normal[2] < 0:
            normal = -normal

        return normal / np.linalg.norm(normal)

    def _orient_normals_consistently(self):
        """法向量一致性调整"""
        # 使用最小生成树方法确保法向量方向一致
        # 这里简化处理，确保大部分法向量朝上
        z_components = self.normals[:, 2]
        flip_mask = z_components < 0
        self.normals[flip_mask] *= -1


def _compute_normals_batch(batch_info, points, kdtree_data, search_radius, max_nn):
    """
    批量计算法向量（用于多进程）

    参数:
        batch_info: (start_idx, end_idx) 批次信息
        points: 所有点坐标
        kdtree_data: KDTree构建数据
        search_radius: 搜索半径
        max_nn: 最大邻近点数

    返回:
        (start_idx, end_idx, normals) 批次结果
    """
    start_idx, end_idx = batch_info

    # 构建局部KDTree
    kdtree = KDTree(kdtree_data, leaf_size=30)

    # 批次法向量
    batch_normals = np.zeros((end_idx - start_idx, 3))

    for i in range(start_idx, end_idx):
        local_idx = i - start_idx

        # 查找邻近点
        indices = kdtree.query_radius(
            points[i:i + 1],
            r=search_radius,
            return_distance=False
        )[0]

        # 限制邻近点数量
        if len(indices) > max_nn:
            distances = np.linalg.norm(points[indices] - points[i], axis=1)
            sorted_indices = np.argsort(distances)
            indices = indices[sorted_indices[:max_nn]]

        if len(indices) < 3:
            # 邻近点太少，使用默认法向量
            batch_normals[local_idx] = [0, 0, 1]
            continue

        # 迭代重加权平面拟合
        normal = _iterative_reweighted_fitting_static(
            points[indices],
            points[i]
        )

        batch_normals[local_idx] = normal

    return start_idx, end_idx, batch_normals


def _compute_normals_batch_optimized(batch_info, points, all_indices):
    """
    批量计算法向量（内存优化版，用于多进程）
    使用预计算的邻近点索引，避免重复构建KDTree

    参数:
        batch_info: (start_idx, end_idx) 批次信息
        points: 所有点坐标
        all_indices: 预计算的所有邻近点索引列表

    返回:
        (start_idx, end_idx, normals) 批次结果
    """
    start_idx, end_idx = batch_info

    # 批次法向量
    batch_normals = np.zeros((end_idx - start_idx, 3))

    for i in range(start_idx, end_idx):
        local_idx = i - start_idx

        # 使用预计算的邻近点索引
        indices = all_indices[i]

        if len(indices) < 3:
            # 邻近点太少，使用默认法向量
            batch_normals[local_idx] = [0, 0, 1]
            continue

        # 迭代重加权平面拟合
        normal = _iterative_reweighted_fitting_static(
            points[indices],
            points[i]
        )

        batch_normals[local_idx] = normal

    return start_idx, end_idx, batch_normals


def _iterative_reweighted_fitting_static(points, center, max_iter=5):
    """
    迭代重加权平面拟合（静态版本，用于多进程）

    参数:
        points: 邻近点坐标
        center: 中心点坐标
        max_iter: 最大迭代次数

    返回:
        normal: 拟合平面法向量
    """
    # 中心化
    centered_points = points - center

    # 初始权重（均匀）
    weights = np.ones(len(points))

    for _ in range(max_iter):
        # 加权协方差矩阵
        weighted_cov = np.dot(
            centered_points.T * weights,
            centered_points
        ) / np.sum(weights)

        # SVD分解
        _, _, vt = svd(weighted_cov)
        normal = vt[2]  # 最小特征值对应的特征向量

        # 更新权重（基于残差）
        residuals = np.abs(np.dot(centered_points, normal))
        sigma = np.median(residuals) * 1.4826  # 鲁棒标准差估计

        if sigma > 0:
            weights = np.exp(-(residuals / (2 * sigma)) ** 2)
        else:
            break

    # 确保法向量朝上
    if normal[2] < 0:
        normal = -normal

    return normal / np.linalg.norm(normal)