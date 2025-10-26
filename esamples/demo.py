"""
演示脚本
展示系统的基本使用方法
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from main import RockDiscontinuityAnalyzer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    """主函数"""

    # 创建分析器
    analyzer = RockDiscontinuityAnalyzer()

    # 配置参数（可选）
    config = {
        "preprocessing": {
            "voxel_size": 0.05,
            "remove_outliers": True,
            "outlier_nb_neighbors": 20,
            "outlier_std_ratio": 2.0
        },
        "normal_estimation": {
            "search_radius": 0.3,
            "max_nn": 30
        },
        "clustering": {
            "eps": 0.1,
            "min_samples": 50,
            "angle_threshold": 30
        },
        "kde": {
            "bin_size": 256,
            "min_angle_between_sets": 20
        }
    }

    analyzer.config = config

    # 运行分析
    # 注意：需要提供实际的点云文件路径
    input_file = "data-files/sample_data/zzm.ply"
    output_dir = "data-files/output/"

    try:
        analyzer.run_analysis(input_file, output_dir)
        print("分析完成！结果已保存至:", output_dir)
    except Exception as e:
        print(f"分析失败: {e}")


if __name__ == "__main__":
    main()