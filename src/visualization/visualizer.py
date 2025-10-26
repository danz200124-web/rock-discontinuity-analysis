"""
可视化主模块
整合各种可视化功能
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
from .point_cloud_viewer import PointCloudViewer
from .stereonet_plot import StereonetPlotter
from .density_plot import DensityPlotter
from .font_config import configure_chinese_font

# 配置中文字体
configure_chinese_font()

logger = logging.getLogger(__name__)


class Visualizer:
    """综合可视化器"""

    def __init__(self):
        self.pc_viewer = PointCloudViewer()
        self.stereonet_plotter = StereonetPlotter()
        self.density_plotter = DensityPlotter()

    def show_point_cloud(self, point_cloud, discontinuity_sets=None):
        """显示3D点云"""
        self.pc_viewer.show(point_cloud, discontinuity_sets)

    def show_point_cloud_with_planes(self, point_cloud, discontinuity_sets=None):
        """显示3D点云及裂隙面"""
        self.pc_viewer.show_with_planes(point_cloud, discontinuity_sets)

    def plot_stereonet(self, normals, discontinuity_sets=None):
        """绘制立体投影图"""
        fig = self.stereonet_plotter.plot(normals, discontinuity_sets, title="不连续面立体投影图")
        plt.show()
        return fig

    def plot_density_contour(self, normals):
        """绘制密度云图"""
        fig = self.density_plotter.plot_contour(normals, title="法向量密度分布图")
        plt.show()
        return fig

    def plot_statistics(self, parameters):
        """
        绘制统计图表

        参数:
            parameters: 参数字典
        """
        n_sets = len(parameters)

        if n_sets == 0:
            logger.warning("没有数据可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('不连续面参数统计图表', fontsize=18, fontweight='bold', y=0.995)

        # 提取数据
        set_ids = []
        dip_dirs = []
        dips = []
        trace_lengths = []
        spacings = []

        for set_id, params in parameters.items():
            set_ids.append(set_id)

            if 'orientation' in params:
                dip_dirs.append(params['orientation']['dip_direction'])
                dips.append(params['orientation']['dip'])
            else:
                dip_dirs.append(0)
                dips.append(0)

            if 'trace_length' in params and params['trace_length']:
                trace_lengths.append(params['trace_length']['exposed_length'])
            else:
                trace_lengths.append(0)

            if 'spacing' in params and params['spacing']:
                spacings.append(params['spacing']['mean_spacing'])
            else:
                spacings.append(0)

        # 产状玫瑰图
        ax = axes[0, 0]
        ax.bar(range(n_sets), dip_dirs, color='blue', alpha=0.6, label='倾向')
        ax.set_xlabel('不连续面组')
        ax.set_ylabel('倾向（度）')
        ax.set_title('倾向分布')
        ax.set_xticks(range(n_sets))
        ax.set_xticklabels(set_ids)
        ax.grid(True, alpha=0.3)

        # 倾角分布
        ax = axes[0, 1]
        ax.bar(range(n_sets), dips, color='red', alpha=0.6, label='倾角')
        ax.set_xlabel('不连续面组')
        ax.set_ylabel('倾角（度）')
        ax.set_title('倾角分布')
        ax.set_xticks(range(n_sets))
        ax.set_xticklabels(set_ids)
        ax.grid(True, alpha=0.3)

        # 迹长分布
        ax = axes[1, 0]
        ax.bar(range(n_sets), trace_lengths, color='green', alpha=0.6)
        ax.set_xlabel('不连续面组')
        ax.set_ylabel('暴露长度（米）')
        ax.set_title('迹长分布')
        ax.set_xticks(range(n_sets))
        ax.set_xticklabels(set_ids)
        ax.grid(True, alpha=0.3)

        # 间距分布
        ax = axes[1, 1]
        ax.bar(range(n_sets), spacings, color='orange', alpha=0.6)
        ax.set_xlabel('不连续面组')
        ax.set_ylabel('平均间距（米）')
        ax.set_title('间距分布')
        ax.set_xticks(range(n_sets))
        ax.set_xticklabels(set_ids)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def create_report(self, parameters, output_path):
        """
        创建分析报告

        参数:
            parameters: 参数字典
            output_path: 输出路径
        """
        # TODO: 使用reportlab或其他库生成PDF报告
        logger.info(f"生成报告: {output_path}")
        pass