"""
频率参数计算模块
计算线频率P10、面频率P20和面强度P21
"""

import numpy as np
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)


class FrequencyCalculator:
    """
    不连续面频率参数计算器
    """

    @staticmethod
    def calculate(discontinuity_set, scanline=None, window=None):
        """
        计算频率参数

        参数:
            discontinuity_set: 不连续面集合
            scanline: 测线参数（可选）
            window: 测窗参数（可选）

        返回:
            frequency_params: 频率参数字典
        """
        params = {}

        # 计算线频率P10
        if scanline is not None:
            params['P10'] = FrequencyCalculator._calculate_p10(
                discontinuity_set, scanline
            )

        # 计算面频率P20和面强度P21
        if window is not None:
            params['P20'] = FrequencyCalculator._calculate_p20(
                discontinuity_set, window
            )
            params['P21'] = FrequencyCalculator._calculate_p21(
                discontinuity_set, window
            )

        return params

    @staticmethod
    def _calculate_p10(discontinuity_set, scanline):
        """
        计算线频率P10（单位长度上的不连续面数量）

        参数:
            discontinuity_set: 不连续面集合
            scanline: 测线参数 {'start': [x,y,z], 'end': [x,y,z]}

        返回:
            P10: 线频率（个/米）
        """
        start = np.array(scanline['start'])
        end = np.array(scanline['end'])
        line_length = np.linalg.norm(end - start)

        if line_length == 0:
            return 0

        # 统计与测线相交的不连续面数量
        intersections = 0

        for disc in discontinuity_set:
            # 检查不连续面与测线是否相交
            if FrequencyCalculator._check_line_plane_intersection(
                    start, end, disc['plane_params']
            ):
                intersections += 1

        # 应用Terzaghi偏差校正
        # 考虑不连续面与测线的夹角
        correction_factor = FrequencyCalculator._terzaghi_correction(
            discontinuity_set, scanline
        )

        P10 = (intersections * correction_factor) / line_length

        return P10

    @staticmethod
    def _calculate_p20(discontinuity_set, window):
        """
        计算面频率P20（单位面积上的不连续面数量）

        参数:
            discontinuity_set: 不连续面集合
            window: 测窗参数 {'vertices': [[x,y,z], ...]}

        返回:
            P20: 面频率（个/平方米）
        """
        vertices = np.array(window['vertices'])

        # 计算测窗面积
        if len(vertices) == 4:  # 矩形窗
            v1 = vertices[1] - vertices[0]
            v2 = vertices[3] - vertices[0]
            area = np.linalg.norm(np.cross(v1, v2))
        else:  # 一般多边形
            hull = ConvexHull(vertices[:, :2])  # 投影到xy平面
            area = hull.volume  # 2D中volume就是面积

        if area == 0:
            return 0

        # 统计迹线类型
        N0 = 0  # 两端都被截断
        N1 = 0  # 一端被截断
        N2 = 0  # 两端都可见

        for disc in discontinuity_set:
            trace_type = FrequencyCalculator._classify_trace(
                disc['trace'], window
            )
            if trace_type == 0:
                N0 += 1
            elif trace_type == 1:
                N1 += 1
            elif trace_type == 2:
                N2 += 1

        # 根据公式计算P20
        P20 = (N0 + N1 + N2 - N0 + N2 / 2) / area

        return P20

    @staticmethod
    def _calculate_p21(discontinuity_set, window):
        """
        计算面强度P21（单位面积上的迹线总长度）

        参数:
            discontinuity_set: 不连续面集合
            window: 测窗参数

        返回:
            P21: 面强度（米/平方米）
        """
        vertices = np.array(window['vertices'])

        # 计算测窗面积
        if len(vertices) == 4:
            v1 = vertices[1] - vertices[0]
            v2 = vertices[3] - vertices[0]
            area = np.linalg.norm(np.cross(v1, v2))
        else:
            hull = ConvexHull(vertices[:, :2])
            area = hull.volume

        if area == 0:
            return 0

        # 计算迹线总长度
        total_trace_length = 0

        for disc in discontinuity_set:
            if 'trace_length' in disc:
                # 只计算在测窗内的部分
                visible_length = FrequencyCalculator._get_visible_trace_length(
                    disc['trace'], window
                )
                total_trace_length += visible_length

        P21 = total_trace_length / area

        return P21

    @staticmethod
    def _check_line_plane_intersection(start, end, plane_params):
        """检查线段与平面是否相交"""
        # 平面方程: ax + by + cz + d = 0
        a, b, c, d = plane_params
        normal = np.array([a, b, c])

        # 线段方向向量
        direction = end - start

        # 检查线段是否与平面平行
        denom = np.dot(normal, direction)
        if abs(denom) < 1e-6:
            return False

        # 计算交点参数t
        t = -(np.dot(normal, start) + d) / denom

        # 检查交点是否在线段内
        return 0 <= t <= 1

    @staticmethod
    def _terzaghi_correction(discontinuity_set, scanline):
        """
        Terzaghi偏差校正系数
        考虑不连续面与测线夹角的影响
        """
        start = np.array(scanline['start'])
        end = np.array(scanline['end'])
        line_direction = (end - start) / np.linalg.norm(end - start)

        # 计算平均校正系数
        corrections = []
        for disc in discontinuity_set:
            if 'normal' in disc:
                normal = disc['normal']
                # 计算夹角余弦
                cos_angle = abs(np.dot(normal, line_direction))
                # 避免除零
                if cos_angle > 0.1:
                    corrections.append(1.0 / cos_angle)

        if corrections:
            return np.mean(corrections)
        else:
            return 1.0

    @staticmethod
    def _classify_trace(trace, window):
        """
        分类迹线类型
        0: 两端都被截断
        1: 一端被截断
        2: 两端都可见
        """
        if trace is None:
            return -1

        # 简化处理：根据迹线端点是否在测窗边界上判断
        # 实际应用中需要更精确的几何判断

        # 这里返回随机分类作为示例
        return np.random.choice([0, 1, 2])

    @staticmethod
    def _get_visible_trace_length(trace, window):
        """获取迹线在测窗内的可见长度"""
        if trace is None or 'length' not in trace:
            return 0

        # 简化处理：返回完整迹线长度
        # 实际应用中需要计算迹线与测窗边界的交点
        return trace['length']