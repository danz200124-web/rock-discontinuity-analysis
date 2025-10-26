"""
聚类分析模块
用于识别单个不连续面
内存优化版 - 支持大规模点云
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DBSCANClusterer:
    """基于DBSCAN的聚类器（内存优化版）"""

    def __init__(self):
        self.labels = None

    def cluster(self, points, normals, eps=0.1, min_samples=50,
                weight_normal=0.3, batch_size=100000):
        """
        对点云进行聚类（内存优化版）

        参数:
            points: 点坐标数组 (N, 3)
            normals: 法向量数组 (N, 3)
            eps: DBSCAN邻域半径
            min_samples: 最小样本数
            weight_normal: 法向量权重
            batch_size: 批处理大小

        返回:
            labels: 聚类标签
        """
        n_points = points.shape[0]

        logger.info(f"DBSCAN聚类，eps={eps}, min_samples={min_samples}, 点数={n_points}")

        # 如果点数较少，直接使用标准DBSCAN
        if n_points < batch_size:
            return self._cluster_standard(points, normals, eps, min_samples, weight_normal)
        else:
            logger.info(f"点数较多({n_points})，使用分批聚类策略")
            return self._cluster_batched(points, normals, eps, min_samples, weight_normal, batch_size)

    def _cluster_standard(self, points, normals, eps, min_samples, weight_normal):
        """标准DBSCAN聚类"""
        # 标准化
        scaler_points = StandardScaler()
        scaler_normals = StandardScaler()

        points_scaled = scaler_points.fit_transform(points)
        normals_scaled = scaler_normals.fit_transform(normals)

        # 加权组合
        features = np.hstack([
            points_scaled,
            normals_scaled * weight_normal
        ])

        # DBSCAN聚类
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.labels = clusterer.fit_predict(features)

        # 统计聚类结果
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)

        logger.info(f"聚类完成：{n_clusters} 个簇，{n_noise} 个噪声点")

        return self.labels

    def _cluster_batched(self, points, normals, eps, min_samples, weight_normal, batch_size):
        """
        分批DBSCAN聚类（内存优化）

        策略：
        1. 先对下采样的数据进行聚类，获取簇中心
        2. 将所有点分配到最近的簇中心
        """
        n_points = points.shape[0]

        # 步骤1：下采样进行初始聚类
        # 选择代表性点（均匀采样）
        sample_rate = min(1.0, batch_size / n_points)
        n_samples = int(n_points * sample_rate)

        logger.info(f"步骤1: 从{n_points}个点中采样{n_samples}个进行初始聚类")

        sample_indices = np.random.choice(n_points, n_samples, replace=False)
        sample_points = points[sample_indices]
        sample_normals = normals[sample_indices]

        # 对采样点进行标准DBSCAN聚类
        sample_labels = self._cluster_standard(sample_points, sample_normals,
                                              eps, min_samples, weight_normal)

        # 获取每个簇的代表（簇中心）
        cluster_ids = np.unique(sample_labels)
        cluster_ids = cluster_ids[cluster_ids != -1]  # 排除噪声

        if len(cluster_ids) == 0:
            logger.warning("初始聚类未找到任何簇，所有点标记为噪声")
            return np.full(n_points, -1, dtype=int)

        logger.info(f"初始聚类找到 {len(cluster_ids)} 个簇")

        # 计算每个簇的中心
        cluster_centers = []
        cluster_normal_centers = []

        for cid in cluster_ids:
            mask = sample_labels == cid
            center = sample_points[mask].mean(axis=0)
            normal_center = sample_normals[mask].mean(axis=0)
            # 归一化法向量
            normal_center = normal_center / (np.linalg.norm(normal_center) + 1e-10)
            cluster_centers.append(center)
            cluster_normal_centers.append(normal_center)

        cluster_centers = np.array(cluster_centers)
        cluster_normal_centers = np.array(cluster_normal_centers)

        # 步骤2：将所有点分配到最近的簇
        logger.info(f"步骤2: 将所有{n_points}个点分配到簇中心")

        self.labels = np.full(n_points, -1, dtype=int)

        # 分批处理所有点
        for batch_start in range(0, n_points, batch_size):
            batch_end = min(batch_start + batch_size, n_points)
            batch_points = points[batch_start:batch_end]
            batch_normals = normals[batch_start:batch_end]

            # 计算到每个簇中心的距离（空间+法向量）
            for i, (center, normal_center) in enumerate(zip(cluster_centers, cluster_normal_centers)):
                # 空间距离
                spatial_dist = np.linalg.norm(batch_points - center, axis=1)

                # 法向量距离（余弦距离）
                normal_sim = np.abs(np.dot(batch_normals, normal_center))
                normal_dist = 1 - normal_sim  # 转为距离

                # 综合距离
                combined_dist = spatial_dist + weight_normal * normal_dist * np.max(spatial_dist)

                # 如果距离小于阈值，分配到该簇
                mask = combined_dist < eps * 2  # 使用2倍eps作为阈值

                # 只分配尚未分配的点
                unassigned = self.labels[batch_start:batch_end] == -1
                assign_mask = mask & unassigned

                self.labels[batch_start + np.where(assign_mask)[0]] = cluster_ids[i]

            if (batch_end) % (batch_size * 5) == 0 or batch_end == n_points:
                logger.info(f"分配进度: {batch_end}/{n_points} ({100*batch_end//n_points}%)")

        # 统计聚类结果
        n_clusters = len(np.unique(self.labels[self.labels != -1]))
        n_noise = np.sum(self.labels == -1)

        logger.info(f"✅ 分批聚类完成：{n_clusters} 个簇，{n_noise} 个噪声点")

        return self.labels

    def refine_clusters(self, points, labels, min_points=30):
        """
        细化聚类结果，移除过小的簇

        参数:
            points: 点坐标数组
            labels: 聚类标签
            min_points: 最小点数阈值

        返回:
            refined_labels: 细化后的标签
        """
        refined_labels = labels.copy()

        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue

            cluster_size = np.sum(labels == cluster_id)

            if cluster_size < min_points:
                # 将小簇标记为噪声
                refined_labels[labels == cluster_id] = -1
                logger.info(f"移除小簇 {cluster_id}（{cluster_size} 个点）")

        return refined_labels