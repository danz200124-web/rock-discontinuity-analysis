"""
岩体不连续面自动检测与表征系统
基于UAV-SfM摄影测量和3D点云处理
"""

import numpy as np
import argparse
import logging
from pathlib import Path
import json
import open3d as o3d
import sys
import io

# 设置标准输出为UTF-8编码（解决Windows控制台乱码问题）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from point_cloud_processing.loader import PointCloudLoader
from point_cloud_processing.normal_estimation import NormalEstimator
from point_cloud_processing.preprocessing import PointCloudPreprocessor
from discontinuity_detection.detector import DiscontinuityDetector
from discontinuity_detection.stereonet import StereonetAnalyzer
from discontinuity_detection.meanshift_analyzer import MeanShiftPoleAnalyzer, AdaptiveDensityAnalyzer
from parameter_calculation.orientation import OrientationCalculator
from parameter_calculation.trace_analysis import TraceAnalyzer
from parameter_calculation.spacing import SpacingCalculator
from parameter_calculation.frequency import FrequencyCalculator
from visualization.visualizer import Visualizer

# 配置日志（使用UTF-8编码）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RockDiscontinuityAnalyzer:
    """
    岩体不连续面分析主类
    整合所有分析模块，提供完整的工作流程
    """

    def __init__(self, config_file=None, use_gpu=True):
        """
        初始化分析器

        参数:
            config_file: 配置文件路径（可选）
            use_gpu: 是否使用GPU加速（默认True）
        """
        self.config = self._load_config(config_file)
        self.use_gpu = use_gpu
        self.point_cloud = None
        self.normals = None
        self.discontinuity_sets = None
        self.parameters = {}

        # 初始化各模块
        self.loader = PointCloudLoader()
        self.preprocessor = PointCloudPreprocessor()
        self.normal_estimator = NormalEstimator()
        self.detector = DiscontinuityDetector(use_adaptive=True)  # 使用高级自适应算法
        self.stereonet_analyzer = StereonetAnalyzer()

        # 使用Mean-Shift自适应密度分析器(替代传统KDE)
        self.density_analyzer = AdaptiveDensityAnalyzer()
        logger.info("✓ 使用Mean-Shift自适应密度分析器(替代传统KDE)")

        self.visualizer = Visualizer()

    def _load_config(self, config_file):
        """加载配置文件"""
        if config_file and Path(config_file).exists():
            logger.info(f"✓ 加载配置文件: {config_file}")
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"配置参数: voxel_size={config['preprocessing']['voxel_size']}, "
                          f"eps={config['clustering']['eps']}, "
                          f"min_samples={config['clustering']['min_samples']}")
                return config
        else:
            logger.warning(f"⚠️  未指定配置文件或文件不存在，使用默认配置")
            # 默认配置
            return {
                "preprocessing": {
                    "voxel_size": 0.05,  # 体素大小（米）
                    "remove_outliers": True,
                    "outlier_nb_neighbors": 20,
                    "outlier_std_ratio": 2.0
                },
                "normal_estimation": {
                    "search_radius": 0.3,  # 搜索半径（米）
                    "max_nn": 30,  # 最大邻近点数
                    "use_parallel": True,  # 使用并行计算
                    "n_jobs": None  # None表示使用所有CPU核心
                },
                "clustering": {
                    "eps": 1.0,  # DBSCAN参数（增大以提高聚类效果）
                    "min_samples": 30,  # 减小最小样本数
                    "angle_threshold": 30  # 角度阈值（度）
                },
                "kde": {
                    "bin_size": 256,  # KDE网格大小
                    "min_angle_between_sets": 20,  # 主要组之间最小角度（度）
                    "max_sets": 10  # 最大不连续面组数
                },
                "performance": {
                    "use_parallel": True,
                    "n_jobs": None,
                    "batch_size": "auto"
                }
            }

    def load_point_cloud(self, file_path):
        """
        加载点云数据

        参数:
            file_path: 点云文件路径（支持.txt, .xyz, .ply, .pcd等格式）
        """
        logger.info(f"正在加载点云: {file_path}")
        self.point_cloud = self.loader.load(file_path)
        logger.info(f"成功加载 {len(self.point_cloud.points)} 个点")

    def preprocess(self):
        """预处理点云"""
        logger.info("开始预处理点云...")

        # 下采样
        if self.config["preprocessing"]["voxel_size"] > 0:
            self.point_cloud = self.preprocessor.voxel_downsample(
                self.point_cloud,
                self.config["preprocessing"]["voxel_size"]
            )

        # 移除离群点
        if self.config["preprocessing"]["remove_outliers"]:
            self.point_cloud = self.preprocessor.remove_outliers(
                self.point_cloud,
                nb_neighbors=self.config["preprocessing"]["outlier_nb_neighbors"],
                std_ratio=self.config["preprocessing"]["outlier_std_ratio"]
            )

        logger.info(f"预处理完成，剩余 {len(self.point_cloud.points)} 个点")

    def estimate_normals(self):
        """估计点云法向量"""
        logger.info("开始估计法向量...")

        # 获取并行配置
        use_parallel = self.config["normal_estimation"].get("use_parallel", True)
        n_jobs = self.config["normal_estimation"].get("n_jobs", None)

        self.normals = self.normal_estimator.estimate(
            self.point_cloud,
            search_radius=self.config["normal_estimation"]["search_radius"],
            max_nn=self.config["normal_estimation"]["max_nn"],
            use_parallel=use_parallel,
            n_jobs=n_jobs
        )
        logger.info("法向量估计完成")

    def detect_discontinuity_sets(self):
        """检测不连续面组"""
        logger.info("开始检测不连续面组...")

        # 立体投影分析
        poles = self.stereonet_analyzer.compute_poles(self.normals)

        # Mean-Shift自适应密度分析(替代传统KDE)
        logger.info("使用Mean-Shift进行自适应密度峰检测...")
        main_poles, pole_labels = self.density_analyzer.analyze(
            poles,
            method='meanshift',
            quantile=self.config.get("meanshift", {}).get("quantile", 0.15)
        )

        logger.info(f"识别到 {len(main_poles)} 个主要不连续面组")

        # 聚类分析 (使用HDBSCAN)
        self.discontinuity_sets = self.detector.detect(
            self.point_cloud,
            self.normals,
            main_poles,
            min_cluster_size=self.config["clustering"].get("min_cluster_size", 50),
            min_samples=self.config["clustering"]["min_samples"],
            angle_threshold=self.config["clustering"]["angle_threshold"]
        )

        logger.info(f"检测到 {len(self.discontinuity_sets)} 个不连续面组")

    def calculate_parameters(self):
        """计算不连续面参数"""
        logger.info("开始计算不连续面参数...")

        # 按set_id分组
        set_groups = {}
        for disc in self.discontinuity_sets:
            set_id = disc.get('set_id', 0)
            if set_id not in set_groups:
                set_groups[set_id] = []
            set_groups[set_id].append(disc)

        # 为每组计算参数
        for set_id, disc_list in set_groups.items():
            logger.info(f"计算第 {set_id} 组参数（共 {len(disc_list)} 个不连续面）...")

            # 计算平均产状（基于该组所有不连续面）
            normals = []
            for disc in disc_list:
                if 'normal' in disc:
                    normals.append(disc['normal'])

            if len(normals) > 0:
                mean_normal = np.mean(normals, axis=0)
                mean_normal = mean_normal / np.linalg.norm(mean_normal)
                orientation = OrientationCalculator.calculate({'normal': mean_normal})
            else:
                orientation = None

            # 计算平均迹长
            trace_lengths = []
            for disc in disc_list:
                trace = TraceAnalyzer.analyze(disc)
                if trace:
                    trace_lengths.append(trace)

            if len(trace_lengths) > 0:
                trace_length = {
                    'exposed_length': np.mean([t['exposed_length'] for t in trace_lengths]),
                    'disc_diameter': np.mean([t['disc_diameter'] for t in trace_lengths]),
                    'area': np.mean([t['area'] for t in trace_lengths]),
                    'perimeter': np.mean([t['perimeter'] for t in trace_lengths]),
                    'shape_factor': np.mean([t['shape_factor'] for t in trace_lengths])
                }
            else:
                trace_length = None

            # 间距（同一组内不连续面之间的间距）
            spacing = SpacingCalculator.calculate(disc_list)
            if spacing is None and len(disc_list) >= 2:
                logger.warning(f"第 {set_id} 组有 {len(disc_list)} 个不连续面，但间距计算失败（可能是平面不够平行）")

            # 频率
            frequency = FrequencyCalculator.calculate(disc_list)

            self.parameters[f"set_{set_id}"] = {
                "orientation": orientation,
                "trace_length": trace_length,
                "spacing": spacing,
                "frequency": frequency,
                "n_discontinuities": len(disc_list)
            }

        logger.info("参数计算完成")

    def visualize_results(self):
        """可视化结果"""
        logger.info("生成可视化结果...")

        # 3D点云可视化 - 显示裂隙面
        self.visualizer.show_point_cloud_with_planes(
            self.point_cloud,
            self.discontinuity_sets
        )

        # 立体投影图
        self.visualizer.plot_stereonet(self.normals, self.discontinuity_sets)

        # 密度云图
        self.visualizer.plot_density_contour(self.normals)

        # 参数统计图表
        self.visualizer.plot_statistics(self.parameters)

        logger.info("可视化完成")

    def save_results(self, output_dir):
        """
        保存分析结果

        参数:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存参数
        with open(output_path / "parameters.json", 'w') as f:
            json.dump(self.parameters, f, indent=2)

        # 保存不连续面点云
        for i, disc_set in enumerate(self.discontinuity_sets):
            # disc_set是字典，包含points等信息
            if 'points' in disc_set:
                # 创建点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(disc_set['points'])
                # 保存点云
                o3d.io.write_point_cloud(
                    str(output_path / f"discontinuity_set_{i + 1}.ply"),
                    pcd
                )

        logger.info(f"结果已保存至: {output_path}")

    def run_analysis(self, input_file, output_dir):
        """
        运行完整分析流程

        参数:
            input_file: 输入点云文件
            output_dir: 输出目录
        """
        # 1. 加载数据
        self.load_point_cloud(input_file)

        # 2. 预处理
        self.preprocess()

        # 3. 估计法向量
        self.estimate_normals()

        # 4. 检测不连续面组
        self.detect_discontinuity_sets()

        # 5. 计算参数
        self.calculate_parameters()

        # 6. 可视化结果
        self.visualize_results()

        # 7. 保存结果
        self.save_results(output_dir)

        logger.info("分析完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='岩体不连续面自动检测与表征系统')
    parser.add_argument('input', help='输入点云文件路径')
    parser.add_argument('-o', '--output', default='./output', help='输出目录')
    parser.add_argument('-c', '--config', help='配置文件路径')
    parser.add_argument('--gpu', action='store_true', default=True, help='使用GPU加速（默认开启）')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false', help='禁用GPU加速')

    args = parser.parse_args()

    # 创建分析器并运行
    analyzer = RockDiscontinuityAnalyzer(config_file=args.config, use_gpu=args.gpu)
    analyzer.run_analysis(args.input, args.output)


if __name__ == '__main__':
    import os
    # 确保工作目录是项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"当前工作目录: {os.getcwd()}")

    main()