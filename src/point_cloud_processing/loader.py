"""
点云数据加载模块
支持多种格式的点云文件加载
"""

import numpy as np
import open3d as o3d
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PointCloudLoader:
    """点云数据加载器"""

    def __init__(self):
        self.supported_formats = ['.txt', '.xyz', '.ply', '.pcd', '.pts']

    def load(self, file_path):
        """
        加载点云文件

        参数:
            file_path: 文件路径

        返回:
            point_cloud: Open3D点云对象
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = file_path.suffix.lower()

        if ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {ext}")

        logger.info(f"加载点云文件: {file_path}")

        if ext in ['.txt', '.xyz', '.pts']:
            return self._load_ascii(file_path)
        elif ext == '.ply':
            return self._load_ply(file_path)
        elif ext == '.pcd':
            return self._load_pcd(file_path)

    def _load_ascii(self, file_path):
        """加载ASCII格式点云"""
        try:
            # 读取数据
            data = np.loadtxt(file_path)

            # 检查数据维度
            if data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("点云数据必须至少包含XYZ三列")

            # 提取XYZ坐标
            points = data[:, :3]

            # 创建Open3D点云对象
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            # 如果有颜色信息（第4-6列）
            if data.shape[1] >= 6:
                colors = data[:, 3:6]
                # 归一化到[0, 1]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                point_cloud.colors = o3d.utility.Vector3dVector(colors)

            logger.info(f"成功加载 {len(points)} 个点")
            return point_cloud

        except Exception as e:
            logger.error(f"加载ASCII文件失败: {e}")
            raise

    def _load_ply(self, file_path):
        """加载PLY格式点云"""
        try:
            point_cloud = o3d.io.read_point_cloud(str(file_path))
            logger.info(f"成功加载 {len(point_cloud.points)} 个点")
            return point_cloud
        except Exception as e:
            logger.error(f"加载PLY文件失败: {e}")
            raise

    def _load_pcd(self, file_path):
        """加载PCD格式点云"""
        try:
            point_cloud = o3d.io.read_point_cloud(str(file_path))
            logger.info(f"成功加载 {len(point_cloud.points)} 个点")
            return point_cloud
        except Exception as e:
            logger.error(f"加载PCD文件失败: {e}")
            raise

    def save(self, point_cloud, file_path):
        """
        保存点云文件

        参数:
            point_cloud: Open3D点云对象
            file_path: 保存路径
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == '.txt':
            points = np.asarray(point_cloud.points)
            np.savetxt(file_path, points, fmt='%.6f')
        else:
            o3d.io.write_point_cloud(str(file_path), point_cloud)

        logger.info(f"点云已保存至: {file_path}")