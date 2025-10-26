"""安装配置文件"""

from setuptools import setup, find_packages

setup(
    name="rock_discontinuity_analysis",
    version="1.0.0",
    author="Rock Analysis Team",
    description="岩体不连续面自动检测与表征系统",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "open3d>=0.13.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "mplstereonet>=0.6.2",
        "tqdm>=4.62.0"
    ],
    python_requires=">=3.7",
)