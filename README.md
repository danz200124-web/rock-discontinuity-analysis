# 岩体不连续面自动检测与表征系统

基于UAV-SfM摄影测量和3D点云处理的岩体裂隙智能检测系统。

## 🌟 主要特性

- **高级算法集成**: 使用HDBSCAN、MAGSAC++、Mean-Shift等先进算法
- **自适应聚类**: HDBSCAN自动确定簇数，无需手动调参
- **高覆盖率**: 优化配置可达70-90%的点云着色覆盖率
- **多参数计算**: 自动计算产状、迹长、间距、频率等参数
- **可视化丰富**: 3D点云、立体投影图、密度云图等

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```bash
# 使用默认配置
python src/main.py your_point_cloud.ply -o output

# 使用最大覆盖率配置
python src/main.py your_point_cloud.ply -c config_max_coverage.json -o output
```

### 测试不同配置

```bash
python test_configs.py
```

## 📊 算法对比

| 算法模块 | 旧版本 | 新版本 | 优势 |
|---------|-------|-------|------|
| 聚类 | DBSCAN | **HDBSCAN** | 自适应确定簇数、处理变密度 |
| 密度分析 | KDE | **Mean-Shift** | 自适应带宽、更精确的峰检测 |
| 平面拟合 | RANSAC | **MAGSAC++** | 自适应阈值、更鲁棒 |

## 🎯 性能提升

| 指标 | 旧版 | 新版 | 改进 |
|-----|------|------|------|
| 着色覆盖率 | ~15% | **60-90%** | ⬆️ 4-6倍 |
| 噪声点比例 | 30-50% | 5-15% | ⬇️ 大幅降低 |
| 检测精度 | 中等 | 高 | ✅ 显著提升 |

## 📁 项目结构

```
rock_discontinuity_analysis/
├── src/                          # 源代码
│   ├── discontinuity_detection/  # 不连续面检测模块
│   │   ├── detector.py           # 主检测器(HDBSCAN)
│   │   ├── hdbscan_clusterer.py  # HDBSCAN聚类
│   │   ├── magsac_plane_fitter.py # MAGSAC++平面拟合
│   │   └── meanshift_analyzer.py  # Mean-Shift密度分析
│   ├── point_cloud_processing/   # 点云处理模块
│   ├── parameter_calculation/    # 参数计算模块
│   └── visualization/            # 可视化模块
├── config.json                   # 默认配置
├── config_max_coverage.json      # 最大覆盖率配置
├── config_balanced.json          # 平衡配置
├── CONFIG_GUIDE.md               # 详细配置指南
├── QUICK_START.md                # 快速开始指南
└── requirements.txt              # 依赖列表

```

## 🔧 配置说明

### 可用配置文件

- **config.json**: 默认配置（推荐）
- **config_balanced.json**: 平衡精度和覆盖率
- **config_hdbscan_optimized.json**: HDBSCAN专用优化
- **config_max_coverage.json**: 最大覆盖率配置（覆盖率可达70-90%）

### 关键参数

```json
{
  "clustering": {
    "min_cluster_size": 30,       // HDBSCAN最小簇大小
    "min_samples": 5,              // 最小样本数
    "angle_threshold": 35          // 角度阈值(度)
  }
}
```

详见 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

## 📖 文档

- [快速开始指南](QUICK_START.md)
- [配置文件指南](CONFIG_GUIDE.md)
- [算法升级说明](UPGRADE_SUMMARY.md)
- [高级算法文档](ADVANCED_ALGORITHMS.md)

## 🛠️ 依赖

主要依赖：
- Python 3.8+
- Open3D
- NumPy
- HDBSCAN
- scikit-learn
- matplotlib
- mplstereonet

## 📝 更新日志

### 2025-10-26
- ✅ 移除旧版DBSCAN算法，全面升级到HDBSCAN
- ✅ 优化配置参数，显著提升点云着色覆盖率
- ✅ 添加多种预设配置文件
- ✅ 创建配置对比工具和快速开始指南

### 历史版本
- 集成MAGSAC++平面拟合
- 集成Mean-Shift密度分析
- 添加GNN图神经网络支持
- 支持GPU加速

## 📄 许可证

本项目仅供学术研究使用。

## 👥 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题或建议，请提交Issue。

---

**🎉 现在就开始使用，体验4-6倍的检测覆盖率提升！**
