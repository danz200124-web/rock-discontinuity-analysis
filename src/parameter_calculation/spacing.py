"""
间距计算模块
计算不连续面间距
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpacingCalculator:
    """间距计算器"""

    @staticmethod
    def calculate_inter_set_spacing(representative_planes):
        """
        计算不连续面组之间的间距

        参数:
            representative_planes: 字典，{set_id: plane_params}
                                 plane_params = [a, b, c, d]，表示 ax+by+cz+d=0

        返回:
            inter_set_spacings: 字典，{(set_id1, set_id2): spacing}
        """
        if not representative_planes or len(representative_planes) < 2:
            logger.warning(f"不连续面组数量不足（需要>=2，当前{len(representative_planes)}个），无法计算组间间距")
            return None

        inter_set_spacings = {}

        # 遍历所有组对
        set_ids = sorted(representative_planes.keys())
        for i in range(len(set_ids)):
            for j in range(i + 1, len(set_ids)):
                set_id1, set_id2 = set_ids[i], set_ids[j]
                plane1 = representative_planes[set_id1]
                plane2 = representative_planes[set_id2]

                if plane1 is None or plane2 is None:
                    continue

                # 计算两个平面之间的距离
                # 平面方程: ax + by + cz + d = 0
                # 两平行平面距离: |d1 - d2| / sqrt(a^2 + b^2 + c^2)

                normal1 = plane1[:3]
                normal2 = plane2[:3]
                d1, d2 = plane1[3], plane2[3]

                # 检查是否平行或接近平行
                # 归一化法向量
                normal1_norm = normal1 / np.linalg.norm(normal1)
                normal2_norm = normal2 / np.linalg.norm(normal2)

                # 计算夹角余弦值
                cos_angle = np.abs(np.dot(normal1_norm, normal2_norm))

                if cos_angle > 0.7:  # 夹角 < 45度，认为接近平行
                    # 计算平行平面间距
                    # 归一化后的间距公式
                    spacing = abs(d1/np.linalg.norm(normal1) - d2/np.linalg.norm(normal2))
                    inter_set_spacings[(set_id1, set_id2)] = spacing
                    logger.info(f"第{set_id1}组 到 第{set_id2}组的间距: {spacing:.3f}m (夹角: {np.degrees(np.arccos(cos_angle)):.1f}°)")
                else:
                    # 不平行，计算交角
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    logger.info(f"第{set_id1}组 与 第{set_id2}组不平行 (夹角: {angle_deg:.1f}°)，无法计算间距")

        if not inter_set_spacings:
            logger.warning("没有找到接近平行的不连续面组，无法计算组间间距")
            return None

        return inter_set_spacings

    @staticmethod
    def calculate(discontinuity_set, method='normal'):
        """
        计算不连续面间距

        参数:
            discontinuity_set: 不连续面数据列表
            method: 计算方法 ('normal', 'scanline')

        返回:
            spacing_params: 间距参数
        """
        if method == 'normal':
            return SpacingCalculator._calculate_normal_spacing(discontinuity_set)
        elif method == 'scanline':
            return SpacingCalculator._calculate_scanline_spacing(discontinuity_set)
        else:
            raise ValueError(f"未知的计算方法: {method}")

    @staticmethod
    def _calculate_normal_spacing(disc_sets):
        """
        计算法向间距

        参数:
            disc_sets: 同组不连续面列表

        返回:
            spacing_params: 间距参数
        """
        if not isinstance(disc_sets, list):
            disc_sets = [disc_sets]

        if len(disc_sets) < 2:
            logger.warning(f"不连续面数量不足（需要>=2，当前{len(disc_sets)}个），无法计算间距")
            return None

        # 提取平面参数
        plane_params_list = []
        for disc in disc_sets:
            if 'plane_params' in disc:
                plane_params_list.append(disc['plane_params'])

        if len(plane_params_list) < 2:
            logger.warning(f"有效平面参数不足（需要>=2，当前{len(plane_params_list)}个）")
            return None

        # 计算所有平面对之间的间距
        spacings = []
        angle_threshold = 0.7  # 降低阈值：cos(45°) ≈ 0.7（原来0.9对应25°太严格）

        for i in range(len(plane_params_list)):
            for j in range(i + 1, len(plane_params_list)):
                # 两个平行平面之间的距离
                # ax + by + cz + d1 = 0
                # ax + by + cz + d2 = 0
                p1 = plane_params_list[i]
                p2 = plane_params_list[j]

                # 检查是否近似平行
                normal1 = p1[:3] / np.linalg.norm(p1[:3])
                normal2 = p2[:3] / np.linalg.norm(p2[:3])

                cos_angle = np.abs(np.dot(normal1, normal2))
                if cos_angle > angle_threshold:  # 夹角小于45度
                    # 计算间距
                    spacing = abs(p1[3] - p2[3]) / np.linalg.norm(p1[:3])
                    spacings.append(spacing)

        if not spacings:
            logger.warning(f"未找到足够平行的平面对（阈值cos(45°)={angle_threshold}）")
            return None

        logger.info(f"计算了 {len(spacings)} 对平面的间距")
        return {
            'mean_spacing': np.mean(spacings),
            'std_spacing': np.std(spacings),
            'min_spacing': np.min(spacings),
            'max_spacing': np.max(spacings),
            'n_measurements': len(spacings)
        }

    @staticmethod
    def _calculate_scanline_spacing(disc_sets, scanline=None):
        """
        沿测线计算间距

        参数:
            disc_sets: 不连续面列表
            scanline: 测线参数

        返回:
            spacing_params: 间距参数
        """
        if scanline is None:
            logger.warning("未提供测线参数")
            return None

        # 计算每个不连续面与测线的交点
        intersections = []

        start = np.array(scanline['start'])
        end = np.array(scanline['end'])
        line_vector = end - start
        line_length = np.linalg.norm(line_vector)
        line_direction = line_vector / line_length

        for disc in disc_sets:
            if 'plane_params' not in disc:
                continue

            # 计算交点
            plane = disc['plane_params']
            normal = plane[:3]
            d = plane[3]

            # 线参数方程: P = start + t * direction
            # 平面方程: n·P + d = 0

            denom = np.dot(normal, line_direction)
            if abs(denom) > 1e-6:
                t = -(np.dot(normal, start) + d) / denom

                if 0 <= t <= line_length:
                    intersections.append(t)

        if len(intersections) < 2:
            return None

        # 排序交点
        intersections.sort()

        # 计算间距
        spacings = np.diff(intersections)

        return {
            'mean_spacing': np.mean(spacings),
            'std_spacing': np.std(spacings),
            'min_spacing': np.min(spacings),
            'max_spacing': np.max(spacings),
            'n_measurements': len(spacings)
        }