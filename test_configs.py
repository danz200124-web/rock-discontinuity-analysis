"""
配置对比测试脚本
快速测试不同配置的着色覆盖率效果
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

from main import RockDiscontinuityAnalyzer
import time

# 测试配置列表
configs = [
    ('config_balanced.json', '平衡配置'),
    ('config_hdbscan_optimized.json', 'HDBSCAN优化配置'),
    ('config_max_coverage.json', '最大覆盖率配置'),
]

# 输入文件
input_file = 'data-files/sample_data/zzm.ply'

print("=" * 80)
print("配置对比测试")
print("=" * 80)

for config_file, description in configs:
    print(f"\n\n{'='*80}")
    print(f"测试配置: {description} ({config_file})")
    print(f"{'='*80}\n")

    try:
        # 创建分析器
        analyzer = RockDiscontinuityAnalyzer(config_file=config_file, use_gpu=False)

        # 记录开始时间
        start_time = time.time()

        # 1. 加载数据
        analyzer.load_point_cloud(input_file)

        # 2. 预处理
        analyzer.preprocess()

        # 3. 估计法向量
        analyzer.estimate_normals()

        # 4. 检测不连续面组
        analyzer.detect_discontinuity_sets()

        # 记录结束时间
        elapsed_time = time.time() - start_time

        # 统计结果
        n_discontinuities = len(analyzer.discontinuity_sets)
        total_points = len(analyzer.point_cloud.points)

        # 计算着色点数
        colored_indices = set()
        for disc in analyzer.discontinuity_sets:
            if 'indices' in disc:
                colored_indices.update(disc['indices'])

        n_colored = len(colored_indices)
        coverage_rate = (n_colored / total_points * 100) if total_points > 0 else 0

        print(f"\n{'='*80}")
        print(f"结果统计 - {description}")
        print(f"{'='*80}")
        print(f"总点数: {total_points:,}")
        print(f"检测到的不连续面数量: {n_discontinuities}")
        print(f"着色点数: {n_colored:,}")
        print(f"着色覆盖率: {coverage_rate:.2f}%")
        print(f"处理时间: {elapsed_time:.2f}秒")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}\n")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("所有测试完成！")
print("="*80)
