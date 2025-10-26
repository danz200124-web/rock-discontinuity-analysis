"""
性能测试脚本 - 充分利用64线程 + 4xRTX3090
"""
import time
import os
import sys

# 确保工作目录正确
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from main import RockDiscontinuityAnalyzer

def test_performance():
    """性能测试"""

    configs = [
        ("CPU 64线程", None, False),
        ("单GPU", "config_high_performance.json", True),
    ]

    results = []

    for name, config_file, use_gpu in configs:
        print(f"\n{'='*60}")
        print(f"🔥 测试配置: {name}")
        print(f"{'='*60}\n")

        start_time = time.time()

        try:
            analyzer = RockDiscontinuityAnalyzer(
                config_file=config_file,
                use_gpu=use_gpu
            )

            analyzer.run_analysis(
                input_file="data-files/sample_data/zzm.ply",
                output_dir=f"./output_{name.replace(' ', '_')}"
            )

            elapsed = time.time() - start_time
            results.append((name, elapsed, "成功"))
            print(f"\n✅ {name} 完成！耗时: {elapsed:.2f}秒")

        except Exception as e:
            elapsed = time.time() - start_time
            results.append((name, elapsed, f"失败: {str(e)[:50]}"))
            print(f"\n❌ {name} 失败: {e}")

    # 输出性能报告
    print(f"\n\n{'='*60}")
    print("📊 性能测试报告")
    print(f"{'='*60}")
    print(f"{'配置':<20} {'耗时(秒)':<12} {'状态'}")
    print("-" * 60)

    for name, elapsed, status in results:
        print(f"{name:<20} {elapsed:>10.2f}s  {status}")

    print(f"{'='*60}\n")

if __name__ == '__main__':
    test_performance()
