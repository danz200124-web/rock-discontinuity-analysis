"""
立体投影分析模块
"""

import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
import logging

logger = logging.getLogger(__name__)


class StereonetAnalyzer:
    """立体投影分析器"""

    def __init__(self):
        self.poles = None

    def compute_poles(self, normals):
        """
        计算极点（立体投影坐标）

        参数:
            normals: 法向量数组 (N, 3)

        返回:
            poles: 极点坐标数组 (N, 2)
        """
        logger.info("计算立体投影极点...")

        n_points = normals.shape[0]
        poles = np.zeros((n_points, 2))

        for i, normal in enumerate(normals):
            x, y = self._vector_to_stereo(normal)
            poles[i] = [x, y]

        self.poles = poles
        return poles

    def _vector_to_stereo(self, vector):
        """
        将单位向量转换为立体投影坐标（下半球投影）

        参数:
            vector: 单位向量 [x, y, z]

        返回:
            stereo_x, stereo_y: 立体投影坐标
        """
        x, y, z = vector / np.linalg.norm(vector)

        # 如果向量朝上，翻转
        if z > 0:
            x, y, z = -x, -y, -z

        # 下半球投影
        denominator = 1 - z

        if abs(denominator) < 1e-10:
            return 0, 0

        stereo_x = x / denominator
        stereo_y = y / denominator

        # 限制在单位圆内
        r = np.sqrt(stereo_x ** 2 + stereo_y ** 2)
        if r > 1:
            stereo_x /= r
            stereo_y /= r

        return stereo_x, stereo_y

    def plot_stereonet(self, normals=None, discontinuity_sets=None):
        """
        绘制立体投影图

        参数:
            normals: 法向量数组（可选）
            discontinuity_sets: 不连续面组列表（可选）
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='stereonet')

        # 绘制所有极点
        if normals is not None:
            for normal in normals:
                # 转换为走向和倾角
                strike, dip = self._normal_to_strike_dip(normal)
                ax.pole(strike, dip, 'k.', markersize=2, alpha=0.5)

        # 绘制分组后的极点
        if discontinuity_sets is not None:
            colors = plt.cm.Set1(np.linspace(0, 1, len(discontinuity_sets)))

            for i, disc_set in enumerate(discontinuity_sets):
                if 'normal' in disc_set:
                    strike, dip = self._normal_to_strike_dip(disc_set['normal'])
                    ax.pole(strike, dip, 'o',
                            color=colors[i], markersize=8,
                            label=f"Set {disc_set['set_id']}")

        ax.grid()
        ax.set_title('立体投影图（下半球投影）')
        ax.legend()

        plt.tight_layout()
        return fig

    def _normal_to_strike_dip(self, normal):
        """法向量转换为走向和倾角"""
        nx, ny, nz = normal / np.linalg.norm(normal)

        # 计算走向（右手定则）
        strike = np.degrees(np.arctan2(-nx, -ny))
        if strike < 0:
            strike += 360

        # 计算倾角
        dip = np.degrees(np.arccos(np.abs(nz)))

        return strike, dip