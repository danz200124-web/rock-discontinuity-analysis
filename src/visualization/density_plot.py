"""
密度云图绘制器
基于KDE的密度分布可视化
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DensityPlotter:
    """密度云图绘制器"""

    def __init__(self):
        pass

    def plot_contour(self, normals, title="法向量密度分布"):
        """
        绘制法向量密度等高线图

        参数:
            normals: 法向量数组 (N, 3)
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制密度等高线图...")

        # 将法向量转换为球面坐标
        azimuth, elevation = self._normals_to_spherical(normals)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))

        # 计算2D KDE，根据样本量调整带宽
        kde_data = np.vstack([azimuth, elevation])
        bw_method = self._adaptive_bandwidth(len(normals))
        kde = stats.gaussian_kde(kde_data, bw_method=bw_method)

        # 创建网格
        azimuth_range = np.linspace(0, 360, 100)
        elevation_range = np.linspace(0, 90, 50)
        A, E = np.meshgrid(azimuth_range, elevation_range)
        positions = np.vstack([A.ravel(), E.ravel()])

        # 计算密度
        density = kde(positions).reshape(A.shape)

        # 动态调整等高线等级数
        n_levels = self._adaptive_levels(len(normals))

        # 绘制等高线
        contour = ax.contourf(A, E, density, levels=n_levels, cmap='viridis', alpha=0.8)
        ax.contour(A, E, density, levels=n_levels, colors='black', alpha=0.4, linewidths=0.5)

        # 叠加原始数据点
        ax.scatter(azimuth, elevation, c='red', s=1, alpha=0.6, label='数据点')

        # 添加颜色条
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('密度', fontsize=12)

        # 设置图形属性
        ax.set_xlabel('方位角 (度)', fontsize=12)
        ax.set_ylabel('仰角 (度)', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)
        ax.legend()

        return fig

    def plot_3d_density(self, normals, title="3D密度分布"):
        """
        绘制3D密度分布图

        参数:
            normals: 法向量数组 (N, 3)
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制3D密度分布图...")

        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 计算密度
        kde = stats.gaussian_kde(normals.T)
        density = kde(normals.T)

        # 创建散点图，颜色表示密度
        scatter = ax.scatter(normals[:, 0], normals[:, 1], normals[:, 2],
                           c=density, cmap='viridis', s=20, alpha=0.6)

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('密度', fontsize=12)

        # 设置图形属性
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=16)

        # 添加单位球面
        self._add_unit_sphere(ax)

        return fig

    def plot_heatmap(self, normals, grid_size=50, title="法向量热力图"):
        """
        绘制球面投影热力图

        参数:
            normals: 法向量数组 (N, 3)
            grid_size: 网格大小
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制球面投影热力图...")

        # 将法向量投影到单位球面
        azimuth, elevation = self._normals_to_spherical(normals)

        # 创建2D直方图
        az_bins = np.linspace(0, 360, grid_size + 1)
        el_bins = np.linspace(0, 90, grid_size // 2 + 1)

        H, az_edges, el_edges = np.histogram2d(azimuth, elevation, bins=[az_bins, el_bins])

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制热力图
        im = ax.imshow(H.T, extent=[0, 360, 0, 90], origin='lower',
                      cmap='hot', aspect='auto', interpolation='bilinear')

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('计数', fontsize=12)

        # 设置图形属性
        ax.set_xlabel('方位角 (度)', fontsize=12)
        ax.set_ylabel('仰角 (度)', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)

        return fig

    def plot_density_by_sets(self, normals, discontinuity_sets, title="分组密度分布"):
        """
        绘制分组密度分布图

        参数:
            normals: 法向量数组 (N, 3)
            discontinuity_sets: 不连续面组字典
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制分组密度分布图...")

        n_sets = len(discontinuity_sets)
        if n_sets == 0:
            logger.warning("没有不连续面组数据")
            return self.plot_contour(normals, title)

        # 计算子图布局
        cols = min(3, n_sets)
        rows = (n_sets + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_sets == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'brown']

        for i, (set_id, set_data) in enumerate(discontinuity_sets.items()):
            if i >= len(axes):
                break

            ax = axes[i]

            if 'indices' in set_data:
                indices = set_data['indices']
                set_normals = normals[indices]

                if len(set_normals) > 3:
                    # 计算该组的密度
                    azimuth, elevation = self._normals_to_spherical(set_normals)

                    # 创建等高线图，根据样本量调整带宽和等级数
                    kde_data = np.vstack([azimuth, elevation])
                    bw_method = self._adaptive_bandwidth(len(set_normals))
                    kde = stats.gaussian_kde(kde_data, bw_method=bw_method)

                    azimuth_range = np.linspace(0, 360, 100)
                    elevation_range = np.linspace(0, 90, 50)
                    A, E = np.meshgrid(azimuth_range, elevation_range)
                    positions = np.vstack([A.ravel(), E.ravel()])
                    density = kde(positions).reshape(A.shape)

                    # 动态调整等高线等级数
                    n_levels = self._adaptive_levels(len(set_normals))
                    contour = ax.contourf(A, E, density, levels=n_levels, cmap='viridis', alpha=0.8)
                    ax.scatter(azimuth, elevation, c=colors[i % len(colors)],
                             s=10, alpha=0.7, label=f'组 {set_id}')

                    ax.set_title(f'组 {set_id} (n={len(set_normals)})', fontsize=12)
                else:
                    ax.text(0.5, 0.5, f'组 {set_id}\n数据点不足',
                           transform=ax.transAxes, ha='center', va='center')

            ax.set_xlabel('方位角 (度)', fontsize=10)
            ax.set_ylabel('仰角 (度)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 90)

        # 隐藏多余的子图
        for i in range(n_sets, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        return fig

    def _adaptive_bandwidth(self, n_samples):
        """
        根据样本量自适应调整KDE带宽

        参数:
            n_samples: 样本数量

        返回:
            带宽方法（数值或字符串）
        """
        if n_samples < 50:
            # 样本少时，使用较大带宽，避免过度拟合
            return 0.5
        elif n_samples < 200:
            # 中等样本，使用中等带宽
            return 0.3
        elif n_samples < 1000:
            # 较多样本，使用scott方法
            return 'scott'
        else:
            # 大样本，使用silverman方法（带宽更小）
            return 'silverman'

    def _adaptive_levels(self, n_samples):
        """
        根据样本量自适应调整等高线等级数

        参数:
            n_samples: 样本数量

        返回:
            等高线等级数
        """
        if n_samples < 50:
            # 样本少时，使用较少等级数
            return 6
        elif n_samples < 200:
            # 中等样本
            return 10
        elif n_samples < 1000:
            # 较多样本
            return 15
        else:
            # 大样本，使用更多等级
            return 20

    def _normals_to_spherical(self, normals):
        """
        将法向量转换为球面坐标

        参数:
            normals: 法向量数组 (N, 3)

        返回:
            azimuth: 方位角数组 (度)
            elevation: 仰角数组 (度)
        """
        # 确保法向量是单位向量
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # 计算方位角 (0-360度)
        azimuth = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
        azimuth = (azimuth + 360) % 360

        # 计算仰角 (0-90度)
        elevation = np.degrees(np.arcsin(np.abs(normals[:, 2])))

        return azimuth, elevation

    def _add_unit_sphere(self, ax, alpha=0.1):
        """
        在3D图中添加单位球面

        参数:
            ax: 3D坐标轴
            alpha: 透明度
        """
        # 创建球面
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # 绘制球面
        ax.plot_surface(x, y, z, alpha=alpha, color='lightgray')

    def save_density_plots(self, normals, discontinuity_sets, output_dir):
        """
        保存所有密度图

        参数:
            normals: 法向量数组
            discontinuity_sets: 不连续面组字典
            output_dir: 输出目录
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 密度等高线图
        fig1 = self.plot_contour(normals)
        fig1.savefig(output_dir / "density_contour.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # 3D密度图
        fig2 = self.plot_3d_density(normals)
        fig2.savefig(output_dir / "density_3d.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # 热力图
        fig3 = self.plot_heatmap(normals)
        fig3.savefig(output_dir / "heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # 分组密度图
        if discontinuity_sets:
            fig4 = self.plot_density_by_sets(normals, discontinuity_sets)
            fig4.savefig(output_dir / "density_by_sets.png", dpi=300, bbox_inches='tight')
            plt.close(fig4)

        logger.info(f"密度图已保存至: {output_dir}")