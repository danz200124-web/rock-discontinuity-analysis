"""
立体投影图绘制器
基于mplstereonet的立体投影可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import mplstereonet
import logging

logger = logging.getLogger(__name__)


class StereonetPlotter:
    """立体投影图绘制器"""

    def __init__(self):
        pass

    def plot(self, normals, discontinuity_sets=None, title="立体投影图"):
        """
        绘制立体投影图

        参数:
            normals: 法向量数组 (N, 3)
            discontinuity_sets: 不连续面组字典（可选）
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制立体投影图...")

        if normals is None or len(normals) == 0:
            logger.error("法向量数据为空，无法绘制立体投影图")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, '无数据可显示', ha='center', va='center', fontsize=16)
            ax.set_title(title, fontsize=16, pad=20)
            return fig

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='stereonet'))

        # 将法向量转换为倾向和倾角
        strikes, dips = self._normals_to_strike_dip(normals)

        logger.info(f"转换得到 {len(strikes)} 个极点, 走向范围: [{strikes.min():.1f}, {strikes.max():.1f}], 倾角范围: [{dips.min():.1f}, {dips.max():.1f}]")

        if discontinuity_sets is None or len(discontinuity_sets) == 0:
            # 如果没有分组，绘制所有点
            ax.pole(strikes, dips, 'k+', markersize=4, label='极点')
        else:
            # 按组绘制不同颜色的点
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'brown']

            # 如果是列表类型的discontinuity_sets（从detector返回）
            if isinstance(discontinuity_sets, list):
                # 按set_id分组
                set_groups = {}
                for disc in discontinuity_sets:
                    set_id = disc.get('set_id', 0)
                    if set_id not in set_groups:
                        set_groups[set_id] = []
                    set_groups[set_id].append(disc)

                # 绘制每组
                for i, (set_id, discs) in enumerate(sorted(set_groups.items())):
                    color = colors[i % len(colors)]

                    # 收集该组所有不连续面的法向量
                    group_normals = []
                    for disc in discs:
                        if 'normal' in disc:
                            group_normals.append(disc['normal'])

                    if len(group_normals) > 0:
                        group_normals = np.array(group_normals)
                        group_strikes, group_dips = self._normals_to_strike_dip(group_normals)

                        # 绘制该组的极点
                        ax.pole(group_strikes, group_dips,
                               color=color, marker='+', markersize=4,
                               label=f'组 {set_id}')

                        # 计算并绘制最佳拟合平面
                        if len(group_strikes) > 3:
                            mean_strike, mean_dip = self._calculate_mean_orientation(
                                group_strikes, group_dips
                            )
                            ax.plane(mean_strike, mean_dip, color=color, linewidth=2, alpha=0.7)

            # 如果是字典类型（旧格式）
            elif isinstance(discontinuity_sets, dict):
                for i, (set_id, set_data) in enumerate(discontinuity_sets.items()):
                    color = colors[i % len(colors)]

                    if 'indices' in set_data:
                        indices = set_data['indices']
                        ax.pole(strikes[indices], dips[indices],
                               color=color, marker='+', markersize=4,
                               label=f'组 {set_id}')

                        if len(indices) > 3:
                            mean_strike, mean_dip = self._calculate_mean_orientation(
                                strikes[indices], dips[indices]
                            )
                            ax.plane(mean_strike, mean_dip, color=color, linewidth=2, alpha=0.7)

        # 设置图形属性
        ax.set_title(title, fontsize=16, pad=20)
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        return fig

    def plot_density(self, normals, title="立体投影密度图"):
        """
        绘制立体投影密度图

        参数:
            normals: 法向量数组 (N, 3)
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制立体投影密度图...")

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='stereonet'))

        # 将法向量转换为倾向和倾角
        strikes, dips = self._normals_to_strike_dip(normals)

        # 绘制密度等高线
        cax = ax.density_contourf(strikes, dips, measurement='poles', cmap='Reds')
        ax.pole(strikes, dips, 'k+', markersize=2, alpha=0.5)

        # 添加颜色条
        fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.8)

        # 设置图形属性
        ax.set_title(title, fontsize=16, pad=20)
        ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_rose_diagram(self, normals, bins=36, title="走向玫瑰图"):
        """
        绘制走向玫瑰图

        参数:
            normals: 法向量数组 (N, 3)
            bins: 角度区间数
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制走向玫瑰图...")

        # 转换为走向
        strikes, _ = self._normals_to_strike_dip(normals)

        # 创建极坐标图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 计算直方图
        theta_bins = np.linspace(0, 2*np.pi, bins+1)
        hist, _ = np.histogram(np.radians(strikes), bins=theta_bins)

        # 绘制玫瑰图
        theta = theta_bins[:-1] + np.pi/(2*bins)
        bars = ax.bar(theta, hist, width=2*np.pi/bins, alpha=0.7, color='skyblue', edgecolor='navy')

        # 设置图形属性
        ax.set_theta_direction(-1)  # 顺时针
        ax.set_theta_zero_location('N')  # 北向为0度
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('走向 (度)', fontsize=12)

        return fig

    def plot_contour_map(self, normals, title="不连续面产状分布图"):
        """
        绘制产状分布等高线图

        参数:
            normals: 法向量数组 (N, 3)
            title: 图标题

        返回:
            matplotlib图形对象
        """
        logger.info("绘制产状分布等高线图...")

        # 转换为倾向和倾角
        strikes, dips = self._normals_to_strike_dip(normals)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))

        # 创建2D直方图
        x_bins = np.linspace(0, 360, 37)  # 倾向
        y_bins = np.linspace(0, 90, 19)   # 倾角

        H, xedges, yedges = np.histogram2d(strikes, dips, bins=[x_bins, y_bins])

        # 创建网格
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

        # 绘制等高线
        contour = ax.contourf(X, Y, H.T, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, H.T, levels=20, colors='black', alpha=0.4, linewidths=0.5)

        # 添加颜色条
        fig.colorbar(contour, ax=ax)

        # 设置图形属性
        ax.set_xlabel('倾向 (度)', fontsize=12)
        ax.set_ylabel('倾角 (度)', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)

        return fig

    def _normals_to_strike_dip(self, normals):
        """
        将法向量转换为走向和倾角（用于mplstereonet）

        参数:
            normals: 法向量数组 (N, 3)

        返回:
            strikes: 走向角度数组 (0-360度)
            dips: 倾角数组 (0-90度)
        """
        # 确保法向量是单位向量
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除以0
        normals = normals / norms

        # 对于mplstereonet，我们需要将法向量转换为平面的走向和倾角
        # 走向是平面走向线的方向（右手定则）
        # 倾角是平面与水平面的夹角

        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]

        # 确保法向量指向下半球（如果指向上，反转）
        mask = nz > 0
        nx = np.where(mask, -nx, nx)
        ny = np.where(mask, -ny, ny)
        nz = np.where(mask, -nz, nz)

        # 计算倾角（与水平面的夹角）
        dips = 90 - np.degrees(np.arccos(np.clip(-nz, -1, 1)))

        # 计算走向（平面走向线的方向）
        strikes = np.degrees(np.arctan2(nx, ny))
        strikes = (strikes + 360) % 360

        return strikes, dips

    def _calculate_mean_orientation(self, strikes, dips):
        """
        计算平均产状

        参数:
            strikes: 走向数组
            dips: 倾角数组

        返回:
            mean_strike: 平均走向
            mean_dip: 平均倾角
        """
        # 转换为向量形式
        strike_rads = np.radians(strikes)
        dip_rads = np.radians(dips)

        # 计算法向量
        nx = np.sin(dip_rads) * np.cos(strike_rads)
        ny = np.sin(dip_rads) * np.sin(strike_rads)
        nz = np.cos(dip_rads)

        # 计算平均法向量
        mean_nx = np.mean(nx)
        mean_ny = np.mean(ny)
        mean_nz = np.mean(nz)

        # 归一化
        norm = np.sqrt(mean_nx**2 + mean_ny**2 + mean_nz**2)
        mean_nx /= norm
        mean_ny /= norm
        mean_nz /= norm

        # 转换回走向和倾角
        mean_dip = np.degrees(np.arccos(abs(mean_nz)))
        mean_strike = np.degrees(np.arctan2(mean_ny, mean_nx))
        mean_strike = (mean_strike + 360) % 360

        return mean_strike, mean_dip

    def save_plots(self, normals, discontinuity_sets, output_dir):
        """
        保存所有类型的投影图

        参数:
            normals: 法向量数组
            discontinuity_sets: 不连续面组字典
            output_dir: 输出目录
        """
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 立体投影图
        fig1 = self.plot(normals, discontinuity_sets)
        fig1.savefig(output_dir / "stereonet.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # 密度图
        fig2 = self.plot_density(normals)
        fig2.savefig(output_dir / "density_stereonet.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)

        # 玫瑰图
        fig3 = self.plot_rose_diagram(normals)
        fig3.savefig(output_dir / "rose_diagram.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)

        # 等高线图
        fig4 = self.plot_contour_map(normals)
        fig4.savefig(output_dir / "contour_map.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)

        logger.info(f"投影图已保存至: {output_dir}")