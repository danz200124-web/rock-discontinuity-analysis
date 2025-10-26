# 配置文件使用指南

## 📋 可用配置文件

根据不同的使用场景，我们提供了以下优化配置：

### 1. **config.json** (默认配置 - 推荐)
- **用途**: 日常使用的标准配置
- **特点**: HDBSCAN优化，平衡精度和覆盖率
- **着色覆盖率**: 预估 50-70%
- **运行速度**: 中等

### 2. **config_balanced.json** (平衡配置)
- **用途**: 兼顾检测精度和覆盖率
- **特点**: 适中的聚类阈值
- **着色覆盖率**: 预估 45-65%
- **运行速度**: 中等

### 3. **config_hdbscan_optimized.json** (HDBSCAN优化)
- **用途**: 针对HDBSCAN算法特别优化
- **特点**: 降低噪声点，提高聚类效果
- **着色覆盖率**: 预估 55-75%
- **运行速度**: 中等

### 4. **config_max_coverage.json** (最大覆盖率)
- **用途**: 最大化点云着色覆盖率
- **特点**: 超低聚类阈值，检测更多细小裂隙
- **着色覆盖率**: 预估 70-90%
- **运行速度**: 较慢
- **注意**: 可能产生一些过度分割

### 5. **config_high_sensitivity.json** (高灵敏度 - 旧版)
- **用途**: 检测更多细小特征
- **特点**: 高灵敏度参数
- **注意**: 此配置可能需要更新以适配HDBSCAN

### 6. **config_optimized.json** (优化配置 - 旧版)
- **用途**: 通用优化配置
- **注意**: 此配置可能需要更新以适配HDBSCAN

---

## 🎯 关键参数说明

### 聚类参数 (clustering)

| 参数 | 说明 | 推荐值 | 影响 |
|-----|------|-------|------|
| `min_cluster_size` | HDBSCAN最小簇大小 | 20-50 | ⬇️ 降低 → 更多簇，更高覆盖率 |
| `min_samples` | HDBSCAN最小样本数 | 3-10 | ⬇️ 降低 → 减少噪声点 |
| `angle_threshold` | 法向量角度阈值(度) | 30-40 | ⬆️ 提高 → 更多点分配到组 |

### 预处理参数 (preprocessing)

| 参数 | 说明 | 推荐值 | 影响 |
|-----|------|-------|------|
| `voxel_size` | 体素下采样尺寸(米) | 0.005-0.01 | ⬇️ 降低 → 更多细节，更慢 |
| `remove_outliers` | 是否移除离群点 | true | 建议开启 |

### Mean-Shift参数 (meanshift)

| 参数 | 说明 | 推荐值 | 影响 |
|-----|------|-------|------|
| `quantile` | 密度分位数阈值 | 0.08-0.15 | ⬇️ 降低 → 检测更多密度峰 |

---

## 🚀 使用方法

### 方法1: 命令行指定配置
```bash
python src/main.py data-files/sample_data/zzm.ply -c config_max_coverage.json -o output_max_coverage
```

### 方法2: 测试多个配置
```bash
python test_configs.py
```

### 方法3: 在代码中使用
```python
from main import RockDiscontinuityAnalyzer

analyzer = RockDiscontinuityAnalyzer(config_file='config_max_coverage.json')
analyzer.run_analysis('data.ply', 'output')
```

---

## 🔧 参数调优建议

### 如果着色覆盖率太低 (<30%)
1. ✅ 降低 `min_cluster_size`: 30 → 20 → 15
2. ✅ 降低 `min_samples`: 5 → 3 → 2
3. ✅ 提高 `angle_threshold`: 35 → 40 → 45
4. ✅ 降低 `quantile`: 0.12 → 0.10 → 0.08

### 如果检测到太多噪声/过度分割
1. ⬆️ 提高 `min_cluster_size`: 30 → 40 → 50
2. ⬆️ 提高 `min_samples`: 5 → 8 → 10
3. ⬇️ 降低 `angle_threshold`: 35 → 30 → 25
4. ⬆️ 提高 `quantile`: 0.12 → 0.15 → 0.18

### 如果运行速度太慢
1. ⬆️ 提高 `voxel_size`: 0.008 → 0.01 → 0.015
2. ✅ 确保 `use_parallel: true` 和 `n_jobs: -1`
3. ⬇️ 降低 `max_nn`: 40 → 30 → 20

---

## 📊 配置对比表

| 配置 | min_cluster_size | min_samples | angle_threshold | voxel_size | 覆盖率 | 速度 |
|-----|-----------------|-------------|-----------------|-----------|-------|------|
| balanced | 25 | 5 | 35° | 0.008 | ⭐⭐⭐ | ⭐⭐⭐ |
| hdbscan_optimized | 30 | 5 | 35° | 0.008 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| max_coverage | 20 | 3 | 40° | 0.005 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 默认config.json | 30 | 5 | 35° | 0.008 | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## ⚠️ 重要提示

1. **HDBSCAN vs DBSCAN**: 新配置专为HDBSCAN设计，不再使用 `eps` 参数
2. **Mean-Shift vs KDE**: 使用Mean-Shift密度分析替代传统KDE
3. **旧配置兼容性**: `config_high_sensitivity.json` 和 `config_optimized.json` 可能需要手动更新

---

## 📝 更新记录

- 2025-10-26: 创建HDBSCAN优化配置，移除旧算法依赖
- 2025-10-26: 添加 `config_max_coverage.json` 最大覆盖率配置
- 2025-10-26: 更新默认 `config.json` 使用HDBSCAN参数
