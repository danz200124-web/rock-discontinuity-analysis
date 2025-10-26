"""
迹线分析模块
计算不连续面迹长
"""

import numpy as np
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)


class TraceAnalyzer:
    """迹线分析器"""

    @staticmethod
    def analyze(discontinuity_set):
        """
        分析不连续面迹线

        参数:
            discontinuity_set: 不连续面数据

        返回:
            trace_params: 迹线参数
        """
        if 'points' not in discontinuity_set:
            logger.warning("缺少点云数据")
            return None

        points = discontinuity_set['points']

        # 计算暴露长度（最大边界尺寸）
        exposed_length = TraceAnalyzer._calculate_exposed_length(points)

        # 计算等效圆盘直径
        disc_diameter = TraceAnalyzer._calculate_disc_diameter(points)

        # 计算迹线统计
        trace_stats = TraceAnalyzer._calculate_trace_statistics(points)

        return {
            'exposed_length': exposed_length,
            'disc_diameter': disc_diameter,
            'area': trace_stats['area'],
            'perimeter': trace_stats['perimeter'],
            'shape_factor': trace_stats['shape_factor']
        }

    @staticmethod
    def _calculate_exposed_length(points):
        """
        计算暴露长度（最大顶点间距离）

        参数:
            points: 点坐标数组

        返回:
            max_length: 最大长度
        """
        if len(points) < 2:
            return 0

        # 投影到最佳拟合平面
        projected = TraceAnalyzer._project_to_plane(points)

        # 计算凸包
        try:
            hull = ConvexHull(projected[:, :2])
            vertices = projected[hull.vertices]

            # 计算所有顶点对之间的距离
            max_length = 0
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    dist = np.linalg.norm(vertices[i] - vertices[j])
                    max_length = max(max_length, dist)

            return max_length

        except Exception as e:
            logger.warning(f"凸包计算失败: {e}")
            # 退化情况：返回点云的最大范围
            return np.max(np.ptp(points, axis=0))

    @staticmethod
    def _calculate_disc_diameter(points):
        """
        计算等效圆盘直径

        参数:
            points: 点坐标数组

        返回:
            diameter: 等效直径
        """
        # 计算点云面积
        projected = TraceAnalyzer._project_to_plane(points)

        try:
            hull = ConvexHull(projected[:, :2])
            area = hull.volume  # 2D中volume就是面积

            # 等效圆盘直径
            diameter = 2 * np.sqrt(area / np.pi)

            return diameter

        except:
            return 0

    @staticmethod
    def _calculate_trace_statistics(points):
        """
        计算迹线统计参数

        参数:
            points: 点坐标数组

        返回:
            stats: 统计参数字典
        """
        projected = TraceAnalyzer._project_to_plane(points)

        try:
            hull = ConvexHull(projected[:, :2])

            # 面积
            area = hull.volume

            # 周长
            perimeter = 0
            for simplex in hull.simplices:
                p1 = projected[simplex[0]]
                p2 = projected[simplex[1]]
                perimeter += np.linalg.norm(p2 - p1)

            # 形状因子（圆度）
            if perimeter > 0:
                shape_factor = 4 * np.pi * area / (perimeter ** 2)
            else:
                shape_factor = 0

            return {
                'area': area,
                'perimeter': perimeter,
                'shape_factor': shape_factor
            }

        except:
            return {
                'area': 0,
                'perimeter': 0,
                'shape_factor': 0
            }

    @staticmethod
    def _project_to_plane(points):
        """
        将点投影到最佳拟合平面

        参数:
            points: 点坐标数组

        返回:
            projected: 投影后的点
        """
        # 计算质心
        centroid = np.mean(points, axis=0)

        # 中心化
        centered = points - centroid

        # PCA获取主方向 - 对(N, 3)矩阵做SVD
        _, _, vt = np.linalg.svd(centered, full_matrices=False)

        # 前两个主成分作为平面坐标系 (vt是3x3矩阵，行是主成分)
        u = vt[0]  # 第一主成分
        v = vt[1]  # 第二主成分

        # 投影
        projected = np.zeros_like(points)
        projected[:, 0] = np.dot(centered, u)
        projected[:, 1] = np.dot(centered, v)
        projected[:, 2] = 0

        return projected