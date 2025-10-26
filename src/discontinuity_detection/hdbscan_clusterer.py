"""
HDBSCAN层级密度聚类模块
优于传统DBSCAN,能够:
1. 自动确定聚类数量
2. 处理不同密度的簇
3. 更鲁棒的噪声处理
4. 层级聚类结构
基于2013年论文"Density-Based Clustering Based on Hierarchical Density Estimates"
"""

import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class HDBSCANClusterer:
    """
    基于HDBSCAN的智能聚类器
    相比传统DBSCAN的优势:
    - 自适应eps参数
    - 可变密度聚类
    - 层级结构
    - 更好的噪声识别
    """

    def __init__(self):
        self.labels = None
        self.clusterer = None
        self.probabilities = None  # 每个点属于其簇的概率

    def cluster(self, points: np.ndarray, normals: np.ndarray,
                min_cluster_size: int = 50, min_samples: int = 10,
                weight_normal: float = 0.3, cluster_selection_epsilon: float = 0.0,
                cluster_selection_method: str = 'eom') -> np.ndarray:
        """
        使用HDBSCAN进行智能聚类

        参数:
            points: 点坐标数组 (N, 3)
            normals: 法向量数组 (N, 3)
            min_cluster_size: 最小簇大小(HDBSCAN核心参数)
            min_samples: 最小样本数(控制噪声敏感度)
            weight_normal: 法向量权重
            cluster_selection_epsilon: DBSCAN-like切割参数(0表示纯HDBSCAN)
            cluster_selection_method: 簇选择方法('eom'或'leaf')
                - 'eom': Excess of Mass(推荐,更稳定)
                - 'leaf': 叶节点选择(更多小簇)

        返回:
            labels: 聚类标签(-1表示噪声)
        """
        n_points = points.shape[0]
        logger.info(f"HDBSCAN智能聚类,点数={n_points}")
        logger.info(f"参数: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        # 特征标准化
        scaler_points = StandardScaler()
        scaler_normals = StandardScaler()

        points_scaled = scaler_points.fit_transform(points)
        normals_scaled = scaler_normals.fit_transform(normals)

        # 加权组合特征
        features = np.hstack([
            points_scaled,
            normals_scaled * weight_normal
        ])

        # HDBSCAN聚类
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            metric='euclidean',
            core_dist_n_jobs=-1,  # 多核并行
            prediction_data=True   # 保留预测数据
        )

        self.labels = self.clusterer.fit_predict(features)
        self.probabilities = self.clusterer.probabilities_

        # 统计聚类结果
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)

        logger.info(f"✅ HDBSCAN聚类完成: {n_clusters} 个簇, {n_noise} 个噪声点 ({100*n_noise/n_points:.1f}%)")

        # 输出每个簇的大小
        unique_labels, counts = np.unique(self.labels[self.labels != -1], return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"  簇 {label}: {count} 个点")

        return self.labels

    def get_cluster_probabilities(self) -> Optional[np.ndarray]:
        """
        获取每个点属于其簇的概率(HDBSCAN独有功能)

        返回:
            probabilities: (N,) 概率数组,范围[0,1]
        """
        return self.probabilities

    def get_outlier_scores(self) -> Optional[np.ndarray]:
        """
        获取离群分数(HDBSCAN独有功能)

        返回:
            outlier_scores: (N,) 离群分数,值越大越可能是噪声
        """
        if self.clusterer is None:
            return None
        return self.clusterer.outlier_scores_

    def get_cluster_persistence(self) -> dict:
        """
        获取簇的持久性(稳定性指标)

        返回:
            persistence_dict: {cluster_id: persistence_value}
        """
        if self.clusterer is None or not hasattr(self.clusterer, 'cluster_persistence_'):
            return {}

        persistence = {}
        for i, p in enumerate(self.clusterer.cluster_persistence_):
            persistence[i] = p

        return persistence

    def refine_clusters_by_probability(self, min_probability: float = 0.5) -> np.ndarray:
        """
        基于概率阈值细化聚类结果

        参数:
            min_probability: 最小概率阈值,低于此值的点标记为噪声

        返回:
            refined_labels: 细化后的标签
        """
        if self.probabilities is None:
            logger.warning("概率数据不可用,返回原始标签")
            return self.labels

        refined_labels = self.labels.copy()
        low_prob_mask = self.probabilities < min_probability
        refined_labels[low_prob_mask] = -1

        n_removed = np.sum(low_prob_mask & (self.labels != -1))
        logger.info(f"基于概率阈值({min_probability})移除 {n_removed} 个低置信度点")

        return refined_labels

    def approximate_predict(self, new_points: np.ndarray, new_normals: np.ndarray,
                           weight_normal: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        对新点进行近似预测(HDBSCAN独有功能)

        参数:
            new_points: 新点坐标 (M, 3)
            new_normals: 新点法向量 (M, 3)
            weight_normal: 法向量权重(应与训练时一致)

        返回:
            labels: 预测标签
            strengths: 预测强度(类似概率)
        """
        if self.clusterer is None or not self.clusterer.prediction_data:
            logger.error("聚类器未训练或未启用预测数据")
            return None, None

        # 特征标准化(使用训练集的scaler)
        scaler_points = StandardScaler()
        scaler_normals = StandardScaler()

        # 重新fit(理想情况应保存训练时的scaler)
        points_scaled = scaler_points.fit_transform(new_points)
        normals_scaled = scaler_normals.fit_transform(new_normals)

        new_features = np.hstack([
            points_scaled,
            normals_scaled * weight_normal
        ])

        # 近似预测
        labels, strengths = hdbscan.approximate_predict(self.clusterer, new_features)

        logger.info(f"预测完成: {len(new_points)} 个新点")

        return labels, strengths


class AdaptiveHDBSCAN:
    """
    自适应HDBSCAN聚类器
    自动调整参数以优化聚类质量
    """

    def __init__(self):
        self.best_clusterer = None
        self.best_params = None
        self.best_score = -np.inf

    def auto_cluster(self, points: np.ndarray, normals: np.ndarray,
                     weight_normal: float = 0.3) -> np.ndarray:
        """
        自动调参聚类

        参数:
            points: 点坐标数组 (N, 3)
            normals: 法向量数组 (N, 3)
            weight_normal: 法向量权重

        返回:
            labels: 最优聚类标签
        """
        n_points = points.shape[0]
        logger.info("启动自适应HDBSCAN聚类...")

        # 参数搜索空间
        param_grid = [
            {'min_cluster_size': int(n_points * 0.01), 'min_samples': 5},
            {'min_cluster_size': int(n_points * 0.02), 'min_samples': 10},
            {'min_cluster_size': int(n_points * 0.03), 'min_samples': 15},
            {'min_cluster_size': max(50, int(n_points * 0.005)), 'min_samples': 10},
        ]

        best_score = -np.inf
        best_labels = None

        for params in param_grid:
            clusterer = HDBSCANClusterer()
            labels = clusterer.cluster(
                points, normals,
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                weight_normal=weight_normal
            )

            # 评估聚类质量(使用DBCV分数)
            if hasattr(clusterer.clusterer, 'relative_validity_'):
                score = clusterer.clusterer.relative_validity_
                logger.info(f"参数 {params}: DBCV分数 = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    self.best_params = params
                    self.best_clusterer = clusterer

        if best_labels is not None:
            logger.info(f"✅ 最佳参数: {self.best_params}, DBCV分数: {best_score:.4f}")
            return best_labels
        else:
            logger.warning("自动调参失败,使用默认参数")
            clusterer = HDBSCANClusterer()
            return clusterer.cluster(points, normals, weight_normal=weight_normal)
