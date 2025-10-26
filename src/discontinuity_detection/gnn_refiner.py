"""
基于图神经网络(GNN)的裂隙关系建模
学习裂隙之间的空间拓扑关系,提升识别精度
使用Graph Attention Network (GAT)架构
基于ICLR 2018论文"Graph Attention Networks"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """图注意力层(GAT Layer)"""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4,
                 concat: bool = True, dropout: float = 0.3, alpha: float = 0.2):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            n_heads: 注意力头数量
            concat: 是否拼接多头输出
            dropout: Dropout率
            alpha: LeakyReLU负斜率
        """
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        # 多头注意力权重
        self.W = nn.Parameter(torch.zeros(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_features, 1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: (N, in_features) 节点特征
            adj: (N, N) 邻接矩阵

        输出:
            out: (N, out_features * n_heads) 或 (N, out_features)
        """
        N = x.size(0)
        out_list = []

        for i in range(self.n_heads):
            # 线性变换: (N, in_features) @ (in_features, out_features) -> (N, out_features)
            h = torch.matmul(x, self.W[i])  # (N, out_features)

            # 注意力系数计算
            h_i = h.repeat(N, 1).view(N * N, -1)  # (N*N, out_features)
            h_j = h.repeat(1, N).view(N * N, -1)  # (N*N, out_features)
            a_input = torch.cat([h_i, h_j], dim=1).view(N, N, 2 * self.out_features)

            # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            e = self.leakyrelu(torch.matmul(a_input, self.a[i]).squeeze(-1))  # (N, N)

            # 掩码:只考虑连接的边
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            # Softmax归一化
            attention = F.softmax(attention, dim=1)  # (N, N)
            attention = self.dropout_layer(attention)

            # 加权聚合
            h_prime = torch.matmul(attention, h)  # (N, out_features)
            out_list.append(h_prime)

        # 多头拼接或平均
        if self.concat:
            return torch.cat(out_list, dim=1)  # (N, out_features * n_heads)
        else:
            return torch.mean(torch.stack(out_list), dim=0)  # (N, out_features)


class DiscontinuityGNN(nn.Module):
    """裂隙识别图神经网络"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64,
                 output_dim: int = 3, n_heads: int = 4, n_layers: int = 3):
        """
        参数:
            input_dim: 输入特征维度(法向量+几何特征)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度(裂隙类别数或嵌入维度)
            n_heads: 注意力头数
            n_layers: GAT层数
        """
        super(DiscontinuityGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 输入特征编码
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT层堆叠
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_dim = hidden_dim
            else:
                in_dim = hidden_dim * n_heads

            # 最后一层不拼接
            concat = (i != n_layers - 1)
            out_dim = hidden_dim if i != n_layers - 1 else output_dim

            self.gat_layers.append(
                GraphAttentionLayer(in_dim, out_dim, n_heads, concat)
            )

        # 分类头
        final_dim = output_dim if n_layers > 0 else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 二分类:是否属于同一裂隙组
        )

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            node_features: (N, input_dim) 节点特征
            adj_matrix: (N, N) 邻接矩阵

        输出:
            embeddings: (N, output_dim) 节点嵌入
            edge_probs: (N, N) 边概率(是否属于同一裂隙组)
        """
        # 特征编码
        x = self.feature_encoder(node_features)

        # GAT层前向传播
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adj_matrix)
            x = F.elu(x)

        embeddings = x  # (N, output_dim)

        # 边分类(成对节点是否属于同一组)
        N = embeddings.size(0)
        embed_i = embeddings.repeat(N, 1).view(N * N, -1)
        embed_j = embeddings.repeat(1, N).view(N * N, -1)

        # 计算边特征(差异+相似度)
        edge_features = torch.abs(embed_i - embed_j)  # (N*N, output_dim)
        edge_probs = self.classifier(edge_features).view(N, N)  # (N, N)

        return embeddings, edge_probs


class DiscontinuityGraphBuilder:
    """裂隙图构建器"""

    def __init__(self, k_neighbors: int = 10, distance_threshold: float = 1.0):
        """
        参数:
            k_neighbors: K近邻数量
            distance_threshold: 距离阈值
        """
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

    def build_graph(self, discontinuities: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从裂隙列表构建图

        参数:
            discontinuities: 裂隙列表,每个元素包含:
                - 'points': 点坐标
                - 'normal': 法向量
                - 'plane_params': 平面参数
                - 'orientation': 产状

        返回:
            node_features: (N, feature_dim) 节点特征矩阵
            adj_matrix: (N, N) 邻接矩阵
        """
        n_discs = len(discontinuities)

        if n_discs == 0:
            return np.array([]), np.array([])

        logger.info(f"构建裂隙图: {n_discs} 个裂隙")

        # 提取节点特征
        node_features = []
        centroids = []

        for disc in discontinuities:
            features = self._extract_features(disc)
            node_features.append(features)

            # 计算质心
            centroid = np.mean(disc['points'], axis=0)
            centroids.append(centroid)

        node_features = np.array(node_features)
        centroids = np.array(centroids)

        # 构建邻接矩阵(基于空间距离+法向量相似度)
        adj_matrix = self._build_adjacency(discontinuities, centroids)

        return node_features, adj_matrix

    def _extract_features(self, disc: Dict) -> np.ndarray:
        """
        提取裂隙特征

        特征包括:
        - 法向量 (3维)
        - 倾向倾角 (2维)
        - 面积/点数 (1维)
        - 平坦度 (1维)
        - 密度 (1维)
        - 平均曲率 (1维)
        - 方向性 (1维)
        共10维
        """
        features = []

        # 法向量
        normal = disc['normal']
        features.extend(normal)

        # 产状
        orientation = disc['orientation']
        features.append(orientation['dip_direction'] / 360.0)  # 归一化
        features.append(orientation['dip'] / 90.0)

        # 规模特征
        n_points = len(disc['points'])
        features.append(np.log10(n_points + 1) / 5.0)  # log归一化

        # 几何特征
        points = disc['points']
        plane_params = disc['plane_params']

        # 平坦度(到平面的RMS距离)
        distances = np.abs(np.dot(points, plane_params[:3]) + plane_params[3])
        flatness = np.std(distances)
        features.append(min(flatness, 1.0))

        # 密度(点密度)
        pairwise_dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        mean_dist = np.mean(pairwise_dist[pairwise_dist > 0])
        density = 1.0 / (mean_dist + 1e-6)
        features.append(min(density, 10.0) / 10.0)

        # 曲率(法向量变化)
        if 'curvature' in disc:
            curvature = disc['curvature']
        else:
            curvature = flatness  # 近似
        features.append(min(curvature, 1.0))

        # 方向性(主方向强度)
        cov = np.cov(points.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        directionality = eigenvalues[2] / (np.sum(eigenvalues) + 1e-6)
        features.append(directionality)

        return np.array(features, dtype=np.float32)

    def _build_adjacency(self, discontinuities: List[Dict], centroids: np.ndarray) -> np.ndarray:
        """构建邻接矩阵"""
        n = len(discontinuities)
        adj_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            # 计算到其他裂隙的距离
            distances = np.linalg.norm(centroids - centroids[i], axis=1)

            # 法向量相似度
            normal_i = discontinuities[i]['normal']
            similarities = np.array([
                np.abs(np.dot(normal_i, discontinuities[j]['normal']))
                for j in range(n)
            ])

            # 综合距离
            combined_dist = distances / (self.distance_threshold + 1e-6) + (1 - similarities)

            # K近邻
            k = min(self.k_neighbors, n - 1)
            nearest_indices = np.argsort(combined_dist)[1:k + 1]  # 排除自己

            # 连接边
            for j in nearest_indices:
                # 边权重(距离越近,法向量越相似,权重越大)
                weight = similarities[j] / (distances[j] + 1e-3)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # 无向图

        return adj_matrix


class GNNDiscontinuityRefiner:
    """GNN裂隙精炼器"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = DiscontinuityGNN(
            input_dim=10,
            hidden_dim=64,
            output_dim=16,
            n_heads=4,
            n_layers=3
        ).to(device)

        self.graph_builder = DiscontinuityGraphBuilder(k_neighbors=10, distance_threshold=1.0)
        self.model.eval()

        logger.info(f"GNN裂隙精炼器已加载到设备: {device}")

    def refine_discontinuities(self, discontinuities: List[Dict],
                               similarity_threshold: float = 0.7) -> List[List[int]]:
        """
        使用GNN精炼裂隙分组

        参数:
            discontinuities: 裂隙列表
            similarity_threshold: 相似度阈值

        返回:
            refined_groups: 精炼后的裂隙分组(每组是裂隙索引列表)
        """
        if len(discontinuities) == 0:
            return []

        logger.info(f"使用GNN精炼 {len(discontinuities)} 个裂隙的分组...")

        # 构建图
        node_features, adj_matrix = self.graph_builder.build_graph(discontinuities)

        # 转换为张量
        node_features_t = torch.from_numpy(node_features).float().to(self.device)
        adj_matrix_t = torch.from_numpy(adj_matrix).float().to(self.device)

        # GNN前向传播
        with torch.no_grad():
            embeddings, edge_probs = self.model(node_features_t, adj_matrix_t)

        # 转换回NumPy
        edge_probs_np = edge_probs.cpu().numpy()

        # 基于边概率重新分组
        refined_groups = self._cluster_by_edges(edge_probs_np, similarity_threshold)

        logger.info(f"✅ GNN精炼完成: {len(refined_groups)} 个精炼组")

        return refined_groups

    def _cluster_by_edges(self, edge_probs: np.ndarray,
                         threshold: float) -> List[List[int]]:
        """基于边概率聚类"""
        n = edge_probs.shape[0]
        visited = np.zeros(n, dtype=bool)
        groups = []

        for i in range(n):
            if visited[i]:
                continue

            # BFS构建连通分量
            group = []
            queue = [i]
            visited[i] = True

            while queue:
                node = queue.pop(0)
                group.append(node)

                # 找到所有相连的节点
                neighbors = np.where((edge_probs[node] > threshold) & (~visited))[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)

            groups.append(group)

        return groups
