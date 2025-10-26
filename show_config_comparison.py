"""
快速查看配置参数对比
"""
import json
import os

configs = [
    'config.json',
    'config_balanced.json',
    'config_hdbscan_optimized.json',
    'config_max_coverage.json'
]

print("=" * 100)
print("配置参数对比表")
print("=" * 100)
print(f"{'配置文件':<35} {'min_cluster':<15} {'min_samples':<15} {'angle_th':<12} {'voxel':<10}")
print("-" * 100)

for config_file in configs:
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            desc = cfg.get('description', 'N/A')
            min_cluster = cfg['clustering']['min_cluster_size']
            min_samples = cfg['clustering']['min_samples']
            angle_th = cfg['clustering']['angle_threshold']
            voxel = cfg['preprocessing']['voxel_size']

            print(f"{config_file:<35} {min_cluster:<15} {min_samples:<15} {angle_th:<12} {voxel:<10}")
            print(f"  描述: {desc}")
    else:
        print(f"{config_file:<35} 文件不存在")

print("=" * 100)
print("\n参数说明:")
print("  - min_cluster_size: HDBSCAN最小簇大小 (越小越多簇，覆盖率越高)")
print("  - min_samples: HDBSCAN最小样本数 (越小噪声点越少)")
print("  - angle_threshold: 角度阈值/度 (越大分配到组的点越多)")
print("  - voxel_size: 体素下采样尺寸/米 (越小保留细节越多)")
print("=" * 100)
