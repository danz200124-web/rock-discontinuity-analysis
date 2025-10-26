"""
平面拟合模块
使用RANSAC算法拟合不连续面
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor
import logging

logger = logging.getLogger(__name__)


class PlaneFitter:
    """平面拟合器"""

    def __init__(self):
        pass

    def fit(self, points, method='ransac'):
        """
        拟合平面

        参数:
            points: 点坐标数组 (N, 3)
            method: 拟合方法 ('ransac', 'svd')

        返回:
            plane_params: 平面参数 [a, b, c, d]
                         满足 ax + by + cz + d = 0
        """
        # 确保points是正确的形状 (N, 3)
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        if points.shape[1] != 3:
            logger.warning(f"点云维度异常: {points.shape}，取前3列作为坐标")
            points = points[:, :3]

        if method == 'ransac':
            return self._fit_ransac(points)
        elif method == 'svd':
            return self._fit_svd(points)
        else:
            raise ValueError(f"未知的拟合方法: {method}")

    def _fit_ransac(self, points, residual_threshold=0.1, max_trials=1000):
        """
        RANSAC平面拟合

        参数:
            points: 点坐标数组
            residual_threshold: 残差阈值
            max_trials: 最大迭代次数

        返回:
            plane_params: 平面参数
        """
        n_points = points.shape[0]

        if n_points < 3:
            logger.warning("点数太少，无法拟合平面")
            return None

        best_params = None
        best_inliers = 0

        for _ in range(max_trials):
            # 随机选择3个点
            sample_indices = np.random.choice(n_points, 3, replace=False)
            sample_points = points[sample_indices]

            # 计算平面参数
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])

            # 计算所有点到平面的距离
            distances = np.abs(np.dot(points, normal) + d)

            # 统计内点数量
            inliers = np.sum(distances < residual_threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_params = np.append(normal, d)

        if best_params is None:
            logger.warning("RANSAC拟合失败")
            return self._fit_svd(points)

        # 使用所有内点重新拟合
        distances = np.abs(np.dot(points, best_params[:3]) + best_params[3])
        inlier_mask = distances < residual_threshold

        if np.sum(inlier_mask) >= 3:
            return self._fit_svd(points[inlier_mask])
        else:
            return best_params

    def _fit_svd(self, points):
        """
        SVD平面拟合

        参数:
            points: 点坐标数组

        返回:
            plane_params: 平面参数
        """
        # 确保points是2D数组且有正确的形状
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(-1, 3)

        # 再次检查列数
        if points.shape[1] != 3:
            logger.warning(f"SVD拟合时点云维度异常: {points.shape}, 取前3列")
            points = points[:, :3]

        # 检查点数
        if points.shape[0] < 3:
            logger.warning("点数太少，无法拟合平面")
            return None

        # 计算质心
        centroid = np.mean(points, axis=0)

        # 中心化
        centered = points - centroid

        # SVD分解 - 对 (N, 3)矩阵做SVD
        # U: (N, N), S: (3,), Vt: (3, 3)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)

        # 法向量是最小奇异值对应的右奇异向量（Vt的最后一行）
        normal = vt[2]  # vt是(3,3)矩阵，最后一行是法向量

        # 确保法向量朝上
        if normal[2] < 0:
            normal = -normal

        # 计算d
        d = -np.dot(normal, centroid)

        return np.append(normal, d)

    def calculate_fitting_error(self, points, plane_params):
        """
        计算拟合误差

        参数:
            points: 点坐标数组
            plane_params: 平面参数

        返回:
            rmse: 均方根误差
        """
        normal = plane_params[:3]
        d = plane_params[3]

        # 计算点到平面的距离
        distances = np.abs(np.dot(points, normal) + d)

        # RMSE
        rmse = np.sqrt(np.mean(distances ** 2))

        return rmse