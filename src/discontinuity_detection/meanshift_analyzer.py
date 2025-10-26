"""
Mean-Shift自适应密度峰检测模块
优于固定网格KDE的特点:
1. 自适应带宽
2. 非参数化峰检测
3. 无需预设峰值数量
4. 更精确的峰值定位
基于1995年论文"Mean Shift: A Robust Approach toward Feature Space Analysis"
"""

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial.distance import cdist
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class MeanShiftPoleAnalyzer:
    """
    基于Mean-Shift的极点分析器
    自适应识别主要不连续面组
    """

    def __init__(self):
        self.cluster_centers = None
        self.labels = None
        self.bandwidth = None

    def analyze(self, poles: np.ndarray, quantile: float = 0.15,
                n_samples: Optional[int] = None, min_bin_freq: int = 10,
                min_angle: float = 15.0) -> Tuple[List[Dict], np.ndarray]:
        """
        使用Mean-Shift分析极点分布

        参数:
            poles: (N, 2) 立体投影极点坐标
            quantile: 带宽估计分位数(0-1),值越小带宽越小,峰越多
            n_samples: 带宽估计的采样数量(None表示全部)
            min_bin_freq: 最小簇频率(过滤小簇)
            min_angle: 主要组之间最小角度(度)

        返回:
            main_poles: 主要极点列表
            labels: 每个极点的簇标签
        """
        logger.info(f"Mean-Shift极点分析: {len(poles)} 个极点")

        # 过滤有效极点(单位圆内)
        valid_mask = np.sum(poles ** 2, axis=1) <= 1.0
        valid_poles = poles[valid_mask]

        if len(valid_poles) < 10:
            logger.warning("有效极点太少,无法分析")
            return [], np.array([])

        logger.info(f"有效极点: {len(valid_poles)} 个")

        # 自适应带宽估计
        self.bandwidth = estimate_bandwidth(
            valid_poles,
            quantile=quantile,
            n_samples=n_samples if n_samples else min(len(valid_poles), 10000)
        )

        logger.info(f"自适应带宽: {self.bandwidth:.4f}")

        # 处理带宽过小的情况
        if self.bandwidth < 1e-4:
            logger.warning("带宽过小,使用默认值")
            self.bandwidth = 0.05

        # Mean-Shift聚类
        ms = MeanShift(
            bandwidth=self.bandwidth,
            bin_seeding=True,
            min_bin_freq=min_bin_freq,
            cluster_all=False,  # 不强制分配离群点
            n_jobs=-1
        )

        self.labels = ms.fit_predict(valid_poles)
        self.cluster_centers = ms.cluster_centers_

        n_clusters = len(self.cluster_centers)
        logger.info(f"Mean-Shift发现 {n_clusters} 个极点簇")

        # 统计每个簇的大小
        unique_labels, counts = np.unique(self.labels[self.labels != -1], return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"  簇 {label}: {count} 个极点 ({100*count/len(valid_poles):.1f}%)")

        # 过滤并排序簇中心
        main_poles = self._filter_and_rank_poles(
            self.cluster_centers,
            valid_poles,
            self.labels,
            min_angle
        )

        logger.info(f"✅ 识别到 {len(main_poles)} 个主要不连续面组")

        return main_poles, self.labels

    def _filter_and_rank_poles(self, centers: np.ndarray, poles: np.ndarray,
                               labels: np.ndarray, min_angle: float) -> List[Dict]:
        """
        过滤并排序极点簇

        参数:
            centers: 簇中心
            poles: 原始极点
            labels: 簇标签
            min_angle: 最小角度阈值

        返回:
            filtered_poles: 过滤后的主要极点列表
        """
        if len(centers) == 0:
            return []

        # 计算每个簇的统计信息
        cluster_stats = []

        for i, center in enumerate(centers):
            # 属于该簇的极点
            cluster_mask = labels == i
            cluster_poles = poles[cluster_mask]
            n_points = len(cluster_poles)

            if n_points < 10:  # 过滤小簇
                continue

            # 计算簇的密度(紧密度)
            distances = np.linalg.norm(cluster_poles - center, axis=1)
            density = n_points / (np.mean(distances) + 1e-6)

            # 计算簇的方差(稳定性)
            variance = np.var(distances)

            cluster_stats.append({
                'center': center,
                'n_points': n_points,
                'density': density,
                'variance': variance,
                'score': n_points * density / (variance + 1.0)  # 综合分数
            })

        if len(cluster_stats) == 0:
            return []

        # 按分数排序
        cluster_stats.sort(key=lambda x: x['score'], reverse=True)

        # 过滤角度过近的簇
        main_poles = []
        min_angle_rad = np.radians(min_angle)

        for stat in cluster_stats:
            center = stat['center']

            # 检查与已选簇的角度
            too_close = False
            for selected_pole in main_poles:
                angle = self._calculate_angle(center, selected_pole['stereo_coords'])
                if angle < min_angle_rad:
                    too_close = True
                    break

            if not too_close:
                # 转换为产状
                dip_dir, dip = self._stereo_to_orientation(center[0], center[1])

                main_poles.append({
                    'stereo_x': center[0],
                    'stereo_y': center[1],
                    'stereo_coords': center,
                    'dip_direction': dip_dir,
                    'dip': dip,
                    'n_points': stat['n_points'],
                    'density': stat['density'],
                    'score': stat['score']
                })

        return main_poles

    def _calculate_angle(self, pole1: np.ndarray, pole2: np.ndarray) -> float:
        """计算两个立体投影极点之间的角度"""
        v1 = self._stereo_to_vector(pole1[0], pole1[1])
        v2 = self._stereo_to_vector(pole2[0], pole2[1])

        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(cos_angle)

    def _stereo_to_vector(self, x: float, y: float) -> np.ndarray:
        """立体投影转3D向量"""
        r2 = x ** 2 + y ** 2
        if r2 > 1:
            return np.array([0, 0, 1])

        z = (1 - r2) / (1 + r2)
        scale = 2 / (1 + r2)
        return np.array([x * scale, y * scale, z])

    def _stereo_to_orientation(self, x: float, y: float) -> Tuple[float, float]:
        """立体投影转产状"""
        vector = self._stereo_to_vector(x, y)

        # 倾向
        dip_direction = np.degrees(np.arctan2(vector[0], vector[1]))
        if dip_direction < 0:
            dip_direction += 360

        # 倾角
        dip = np.degrees(np.arccos(np.abs(vector[2])))

        return dip_direction, dip

    def predict_new_poles(self, new_poles: np.ndarray) -> np.ndarray:
        """
        为新极点预测簇标签

        参数:
            new_poles: (M, 2) 新极点坐标

        返回:
            labels: (M,) 预测的簇标签
        """
        if self.cluster_centers is None:
            logger.warning("模型未训练,无法预测")
            return np.full(len(new_poles), -1)

        # 计算到所有簇中心的距离
        distances = cdist(new_poles, self.cluster_centers)

        # 分配到最近的簇
        labels = np.argmin(distances, axis=1)

        # 如果距离过大,标记为噪声
        min_distances = np.min(distances, axis=1)
        labels[min_distances > 2 * self.bandwidth] = -1

        return labels


class AdaptiveDensityAnalyzer:
    """
    自适应密度分析器
    结合Mean-Shift和KDE的优势
    """

    def __init__(self):
        self.ms_analyzer = MeanShiftPoleAnalyzer()

    def analyze(self, poles: np.ndarray, method: str = 'auto',
                **kwargs) -> Tuple[List[Dict], np.ndarray]:
        """
        自适应密度分析

        参数:
            poles: 极点坐标
            method: 'auto', 'meanshift', 'kde'
            **kwargs: 方法特定参数

        返回:
            main_poles: 主要极点列表
            labels: 簇标签
        """
        n_poles = len(poles)
        logger.info(f"自适应密度分析: {n_poles} 个极点, 方法={method}")

        if method == 'auto':
            # 自动选择:数据量适中用Mean-Shift,大数据用KDE
            if 100 < n_poles < 50000:
                logger.info("选择Mean-Shift方法(自适应)")
                return self.ms_analyzer.analyze(poles, **kwargs)
            else:
                logger.info("选择KDE方法(网格)")
                # 回退到KDE(这里可以调用原有的KDE分析器)
                return [], np.array([])
        elif method == 'meanshift':
            return self.ms_analyzer.analyze(poles, **kwargs)
        else:
            return [], np.array([])

    def multi_scale_analysis(self, poles: np.ndarray,
                            quantiles: List[float] = [0.1, 0.15, 0.2]) -> List[Tuple[List[Dict], np.ndarray]]:
        """
        多尺度分析(不同带宽)

        参数:
            poles: 极点坐标
            quantiles: 带宽分位数列表

        返回:
            results: [(main_poles, labels), ...] 多个尺度的结果
        """
        logger.info(f"多尺度分析: {len(quantiles)} 个尺度")

        results = []
        for q in quantiles:
            logger.info(f"--- 尺度 quantile={q} ---")
            main_poles, labels = self.ms_analyzer.analyze(poles, quantile=q)
            results.append((main_poles, labels))

        # 选择最佳尺度(综合评估)
        best_idx = self._select_best_scale(results, poles)
        logger.info(f"✅ 最佳尺度: quantile={quantiles[best_idx]}")

        return results

    def _select_best_scale(self, results: List[Tuple[List[Dict], np.ndarray]],
                          poles: np.ndarray) -> int:
        """选择最佳尺度"""
        scores = []

        for main_poles, labels in results:
            if len(main_poles) == 0:
                scores.append(-np.inf)
                continue

            # 评分标准:
            # 1. 主要组数量(3-6个较好)
            n_groups = len(main_poles)
            group_score = -abs(n_groups - 4)

            # 2. 覆盖率(非噪声点比例)
            coverage = np.sum(labels != -1) / len(labels) if len(labels) > 0 else 0

            # 3. 分组平衡度(各组大小的均衡性)
            if n_groups > 0:
                group_sizes = [p['n_points'] for p in main_poles]
                balance = -np.std(group_sizes) / (np.mean(group_sizes) + 1)
            else:
                balance = -np.inf

            # 综合分数
            score = group_score + coverage * 10 + balance
            scores.append(score)

        return int(np.argmax(scores))
