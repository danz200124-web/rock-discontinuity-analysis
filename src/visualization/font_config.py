"""
配置matplotlib中文显示
"""
import matplotlib.pyplot as plt
import matplotlib

def configure_chinese_font():
    """配置matplotlib以支持中文显示"""
    try:
        # Windows系统字体配置
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 设置全局字体大小
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10

    except Exception as e:
        print(f"警告: 中文字体配置失败: {e}")
        print("图表将使用默认字体，中文可能无法正常显示")
