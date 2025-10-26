#!/bin/bash
# 激活岩体不连续面分析系统环境

echo "正在激活rock_analysis环境..."
source E:\dzw\rock_discontinuity_analysis\activate rock

echo "环境已激活！"
echo "Python版本: $(python --version)"
echo "当前工作目录: $(pwd)"
echo ""
echo "可用命令："
echo "  python src/main.py <input_file> -o <output_dir> -c <config_file>  # 运行分析"
echo "  python -m unittest discover test/                                # 运行测试"
echo "  pip list                                                         # 查看已安装包"
echo ""