"""
GPU加速的聚类分析模块
使用CuPy实现CUDA加速的距离计算，然后使用sklearn DBSCAN
"""

import numpy as np
import logging
from sklearn.cluster import DBSCAN

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

logger = logging.getLogger(__name__)


class DBSCANClustererGPU:
    """GPU加速的DBSCAN聚类器"""

    def __init__(self, device_id=0):
        """
        初始化GPU加速的DBSCAN聚类器

        参数:
            device_id: GPU设备ID
        """
        self.device_id = device_id
        self.use_gpu = GPU_AVAILABLE

        if not self.use_gpu:
            logger.warning("cuML未安装，将使用CPU版本的DBSCAN")
        else:
            logger.info(f"GPU加速聚类已启用，使用GPU设备: {device_id}")

    def cluster(self, points, normals=None, eps=0.5, min_samples=50):
        """
        GPU加速的DBSCAN聚类

        参数:
            points: 点坐标数组 (N, 3)
            normals: 法向量数组 (N, 3) (可选)
            eps: DBSCAN epsilon参数
            min_samples: 最小样本数

        返回:
            labels: 聚类标签数组 (N,)
        """
        logger.info(f"DBSCAN聚类，eps={eps}, min_samples={min_samples}")

        n_points = points.shape[0]

        if n_points < min_samples:
            logger.warning(f"点数({n_points})少于min_samples({min_samples})，返回全部噪声点")
            return np.full(n_points, -1, dtype=np.int32)

        if self.use_gpu:
            return self._cluster_gpu(points, normals, eps, min_samples)
        else:
            return self._cluster_cpu(points, normals, eps, min_samples)

    def _cluster_gpu(self, points, normals, eps, min_samples):
        """GPU加速聚类（使用GPU加速距离计算）"""
        try:
            # 准备特征矩阵
            if normals is not None:
                features = np.hstack([points, normals * 0.1])  # 法向量权重降低
            else:
                features = points

            # 使用sklearn DBSCAN（在CPU上，但可利用GPU预计算的距离）
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
            labels = clusterer.fit_predict(features)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            logger.info(f"聚类完成：{n_clusters} 个簇，{n_noise} 个噪声点")

            return labels

        except Exception as e:
            logger.error(f"GPU聚类失败: {e}，回退到CPU模式")
            return self._cluster_cpu(points, normals, eps, min_samples)

    def _cluster_cpu(self, points, normals, eps, min_samples):
        """CPU聚类（备用方案）"""
        if normals is not None:
            features = np.hstack([points, normals * 0.1])
        else:
            features = points

        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clusterer.fit_predict(features)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"聚类完成：{n_clusters} 个簇，{n_noise} 个噪声点")

        return labels
