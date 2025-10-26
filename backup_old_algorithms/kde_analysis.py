"""
核密度估计分析模块
用于识别主要不连续面组
极致性能优化版 - 支持32核并行 + NumPy向量化
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import sys

logger = logging.getLogger(__name__)


class KDEAnalyzer:
    """
    基于核密度估计的极点分析器
    在立体投影上进行密度分析，识别主要不连续面组
    极致性能优化版
    """

    def __init__(self):
        self.density = None
        self.main_poles = None
        self.kde_model = None

    def analyze(self, poles, bin_size=256, min_angle=20, max_sets=10,
                use_parallel=True, n_jobs=None, subsample_size=None):
        """
        分析极点分布，识别主要不连续面组（极致性能优化版）

        参数:
            poles: 极点坐标数组 (N, 2) - 立体投影坐标
            bin_size: 密度网格大小
            min_angle: 主要组之间最小角度（度）
            max_sets: 最大不连续面组数
            use_parallel: 是否使用并行计算
            n_jobs: 使用的CPU核心数，None表示使用所有核心
            subsample_size: 子采样大小，None表示使用全部数据

        返回:
            density: 密度矩阵
            main_poles: 主要极点列表
        """
        logger.info("开始核密度估计分析(极致性能优化模式)...")

        # 创建网格
        x_grid = np.linspace(-1, 1, bin_size)
        y_grid = np.linspace(-1, 1, bin_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

        # 向量化过滤有效极点（在单位圆内）
        valid_mask = np.sum(poles**2, axis=1) <= 1
        valid_poles = poles[valid_mask]

        if len(valid_poles) < 10:
            logger.warning("有效极点数量太少，无法进行密度分析")
            return None, []

        logger.info(f"有效极点数量: {len(valid_poles)}")

        # 160GB大内存模式: 不进行子采样,使用全部数据获得最高精度
        if subsample_size is not None and len(valid_poles) > subsample_size:
            logger.info(f"数据量较大,子采样至{subsample_size}个点")
            indices = np.random.choice(len(valid_poles), subsample_size, replace=False)
            valid_poles = valid_poles[indices]
        else:
            logger.info(f"大内存模式:使用全部{len(valid_poles)}个极点(不子采样,最高精度)")

        # 核密度估计 - 使用scott带宽(自适应)
        logger.info("构建KDE模型(使用scipy优化实现)...")
        self.kde_model = gaussian_kde(valid_poles.T, bw_method='scott')

        # 并行计算密度
        if use_parallel and grid_points.shape[1] > 10000:
            if n_jobs is None:
                n_jobs = cpu_count()

            # 激进模式:使用更多核心
            if sys.platform == 'win32':
                n_jobs = min(n_jobs, 64)  # Windows提升到64核
            else:
                n_jobs = min(n_jobs, 128)  # Linux提升到128核

            logger.info(f"⚡ 使用 {n_jobs} 个CPU核心并行计算KDE (极致性能模式)...")

            # 增大批次减少开销
            n_points = grid_points.shape[1]
            batch_size = max(10000, n_points // (n_jobs * 2))
            batches = []
            for i in range(0, n_points, batch_size):
                end_idx = min(i + batch_size, n_points)
                batches.append((i, end_idx, grid_points[:, i:end_idx]))

            logger.info(f"分为 {len(batches)} 个批次(每批{batch_size}个网格点)")

            # 并行计算
            density_flat = np.zeros(n_points)
            with Pool(processes=n_jobs) as pool:
                compute_func = partial(_compute_kde_batch_fast, kde_data=self.kde_model.dataset, bw=self.kde_model.factor)

                for i, (start_idx, end_idx, batch_density) in enumerate(pool.imap(compute_func, batches)):
                    density_flat[start_idx:end_idx] = batch_density
                    if (i + 1) % max(1, len(batches) // 10) == 0:
                        logger.info(f"KDE进度: {100 * (i + 1) // len(batches)}%")

            self.density = density_flat.reshape(xx.shape)
        else:
            logger.info("计算密度分布...")
            self.density = self.kde_model(grid_points).reshape(xx.shape)

        # 寻找密度峰值
        self.main_poles = self._find_density_peaks(
            xx, yy, self.density,
            min_angle=min_angle,
            max_sets=max_sets
        )

        logger.info(f"识别到 {len(self.main_poles)} 个主要不连续面组")

        return self.density, self.main_poles

    def _find_density_peaks(self, xx, yy, density, min_angle=20, max_sets=10):
        """
        寻找密度峰值点

        参数:
            xx, yy: 网格坐标
            density: 密度矩阵
            min_angle: 峰值之间最小角度（度）
            max_sets: 最大峰值数

        返回:
            peaks: 峰值点列表
        """
        # 寻找局部最大值
        flat_density = density.flatten()
        sorted_indices = np.argsort(flat_density)[::-1]

        peaks = []
        min_angle_rad = np.radians(min_angle)

        for idx in sorted_indices:
            if len(peaks) >= max_sets:
                break

            # 获取峰值坐标
            i, j = np.unravel_index(idx, density.shape)
            x, y = xx[i, j], yy[i, j]

            # 检查是否在单位圆内
            if x ** 2 + y ** 2 > 1:
                continue

            # 检查与已有峰值的角度
            too_close = False
            for px, py, _ in peaks:
                angle = self._calculate_angle_between_poles(
                    (x, y), (px, py)
                )
                if angle < min_angle_rad:
                    too_close = True
                    break

            if not too_close:
                peaks.append((x, y, flat_density[idx]))

        # 转换为产状
        main_poles = []
        for x, y, density_value in peaks:
            dip_dir, dip = self._stereo_to_orientation(x, y)
            main_poles.append({
                'stereo_x': x,
                'stereo_y': y,
                'dip_direction': dip_dir,
                'dip': dip,
                'density': density_value
            })

        return main_poles

    def _calculate_angle_between_poles(self, pole1, pole2):
        """计算两个极点之间的角度"""
        # 将立体投影坐标转换为单位向量
        v1 = self._stereo_to_vector(pole1[0], pole1[1])
        v2 = self._stereo_to_vector(pole2[0], pole2[1])

        # 计算夹角
        cos_angle = np.clip(np.dot(v1, v2), -1, 1)
        return np.arccos(cos_angle)

    def _stereo_to_vector(self, x, y):
        """立体投影坐标转换为单位向量"""
        r2 = x ** 2 + y ** 2
        if r2 > 1:
            return np.array([0, 0, 1])

        z = (1 - r2) / (1 + r2)
        scale = 2 / (1 + r2)
        return np.array([x * scale, y * scale, z])

    def _stereo_to_orientation(self, x, y):
        """立体投影坐标转换为产状"""
        vector = self._stereo_to_vector(x, y)

        # 计算倾向
        dip_direction = np.degrees(np.arctan2(vector[1], vector[0]))
        if dip_direction < 0:
            dip_direction += 360

        # 计算倾角
        dip = np.degrees(np.arccos(np.abs(vector[2])))

        return dip_direction, dip

    def plot_density_map(self, save_path=None):
        """
        绘制密度云图

        参数:
            save_path: 保存路径（可选）
        """
        if self.density is None:
            logger.warning("尚未进行密度分析")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制密度等值线
        x = np.linspace(-1, 1, self.density.shape[0])
        y = np.linspace(-1, 1, self.density.shape[1])

        # 只在单位圆内绘制
        mask = np.sqrt(x[:, np.newaxis] ** 2 + y[np.newaxis, :] ** 2) <= 1
        masked_density = np.ma.masked_where(~mask.T, self.density)

        contour = ax.contour(x, y, masked_density, levels=20, colors='black', alpha=0.3)
        contourf = ax.contourf(x, y, masked_density, levels=20, cmap='hot')

        # 绘制单位圆
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)

        # 标记主要极点
        if self.main_poles:
            for i, pole in enumerate(self.main_poles):
                ax.plot(pole['stereo_x'], pole['stereo_y'], 'b*', markersize=15)
                ax.text(pole['stereo_x'] + 0.05, pole['stereo_y'] + 0.05,
                        f"S{i + 1}", fontsize=12, fontweight='bold')

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('极点密度分布图（下半球投影）')

        plt.colorbar(contourf, ax=ax, label='密度')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"密度图已保存至: {save_path}")
        else:
            plt.show()


def _compute_kde_batch_fast(batch_info, kde_data, bw):
    """
    批量计算KDE（激进优化版,用于32核并行）

    参数:
        batch_info: (start_idx, end_idx, grid_points_batch) 批次信息
        kde_data: KDE模型的数据集
        bw: 带宽因子

    返回:
        (start_idx, end_idx, batch_density) 批次结果
    """
    start_idx, end_idx, grid_points_batch = batch_info

    # 重建KDE模型（每个进程独立）
    kde = gaussian_kde(kde_data, bw_method=bw)

    # 计算密度
    batch_density = kde(grid_points_batch)

    return start_idx, end_idx, batch_density