"""
产状计算模块
计算不连续面的倾向和倾角
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class OrientationCalculator:
    """产状计算器"""

    @staticmethod
    def calculate(discontinuity_set):
        """
        计算不连续面产状

        参数:
            discontinuity_set: 不连续面数据

        返回:
            orientation: 产状参数字典
        """
        if 'plane_params' in discontinuity_set:
            normal = discontinuity_set['plane_params'][:3]
        elif 'normal' in discontinuity_set:
            normal = discontinuity_set['normal']
        else:
            logger.warning("缺少平面参数或法向量")
            return None

        # 归一化
        normal = normal / np.linalg.norm(normal)

        # 确保法向量朝下（地质习惯）
        if normal[2] > 0:
            normal = -normal

        # 计算倾向（dip direction）
        dip_direction = np.degrees(np.arctan2(normal[1], normal[0]))
        if dip_direction < 0:
            dip_direction += 360

        # 计算倾角（dip）
        dip = np.degrees(np.arccos(np.abs(normal[2])))

        # 计算走向（strike，右手定则）
        strike = dip_direction - 90
        if strike < 0:
            strike += 360

        return {
            'dip_direction': round(dip_direction, 1),
            'dip': round(dip, 1),
            'strike': round(strike, 1),
            'normal': normal.tolist()
        }

    @staticmethod
    def calculate_fisher_statistics(orientations):
        """
        计算Fisher统计参数

        参数:
            orientations: 产状列表

        返回:
            fisher_params: Fisher统计参数
        """
        if not orientations:
            return None

        # 转换为单位向量
        vectors = []
        for ori in orientations:
            dd_rad = np.radians(ori['dip_direction'])
            dip_rad = np.radians(ori['dip'])

            x = np.sin(dip_rad) * np.sin(dd_rad)
            y = np.sin(dip_rad) * np.cos(dd_rad)
            z = np.cos(dip_rad)

            vectors.append([x, y, z])

        vectors = np.array(vectors)

        # 计算平均向量
        mean_vector = np.mean(vectors, axis=0)
        mean_length = np.linalg.norm(mean_vector)

        # Fisher集中参数K
        n = len(vectors)
        if mean_length > 0.999:
            K = 1000  # 避免除零
        else:
            K = (n - 1) / (n - mean_length)

        # 95%置信锥角
        alpha95 = np.degrees(np.arccos(1 - (n - mean_length) / mean_length *
                                       ((1 / 0.05) ** (1 / (n - 1)) - 1)))

        return {
            'mean_vector': mean_vector / mean_length,
            'concentration': K,
            'confidence_cone': alpha95,
            'n_samples': n
        }