"""
基于Transformer的法向量精炼模块
使用自注意力机制学习点云局部几何关系,提升法向量估计精度
基于Point Transformer架构(ICCV 2021)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PointTransformerLayer(nn.Module):
    """Point Transformer层 - 核心自注意力模块"""

    def __init__(self, in_channels: int, out_channels: int, num_neighbors: int = 16):
        """
        参数:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            num_neighbors: K近邻数量
        """
        super(PointTransformerLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors

        # 线性变换: Q, K, V
        self.fc_q = nn.Linear(in_channels, out_channels)
        self.fc_k = nn.Linear(in_channels, out_channels)
        self.fc_v = nn.Linear(in_channels, out_channels)

        # 位置编码MLP
        self.fc_delta = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

        # 注意力权重MLP (γ)
        self.fc_gamma = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

        # 输出变换
        self.fc_out = nn.Linear(out_channels, out_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor,
                neighbor_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入:
            xyz: (B, N, 3) 点坐标
            features: (B, N, C_in) 点特征
            neighbor_indices: (B, N, K) K近邻索引

        输出:
            out_features: (B, N, C_out) 输出特征
        """
        B, N, _ = xyz.shape

        # 计算K近邻
        if neighbor_indices is None:
            neighbor_indices = self._compute_knn(xyz, self.num_neighbors)  # (B, N, K)

        # Query, Key, Value变换
        q = self.fc_q(features)  # (B, N, C_out)
        k = self.fc_k(features)  # (B, N, C_out)
        v = self.fc_v(features)  # (B, N, C_out)

        # 收集邻居特征
        k_neighbors = self._gather_neighbors(k, neighbor_indices)  # (B, N, K, C_out)
        v_neighbors = self._gather_neighbors(v, neighbor_indices)  # (B, N, K, C_out)
        xyz_neighbors = self._gather_neighbors(xyz, neighbor_indices)  # (B, N, K, 3)

        # 位置编码
        pos_enc = self.fc_delta(xyz.unsqueeze(2) - xyz_neighbors)  # (B, N, K, C_out)

        # 注意力计算: γ(q_i - k_j + δ)
        q_expanded = q.unsqueeze(2)  # (B, N, 1, C_out)
        attn_features = q_expanded - k_neighbors + pos_enc  # (B, N, K, C_out)
        attn_weights = self.fc_gamma(attn_features)  # (B, N, K, C_out)

        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=2)  # (B, N, K, C_out)

        # 加权聚合
        out = torch.sum(attn_weights * (v_neighbors + pos_enc), dim=2)  # (B, N, C_out)

        # 输出变换
        out = self.fc_out(out)

        return out

    @staticmethod
    def _compute_knn(xyz: torch.Tensor, k: int) -> torch.Tensor:
        """计算K近邻索引"""
        B, N, _ = xyz.shape

        # 计算点对距离
        dist = torch.cdist(xyz, xyz)  # (B, N, N)

        # Top-K最近邻
        _, indices = torch.topk(dist, k + 1, dim=-1, largest=False)  # (B, N, K+1)

        return indices[:, :, 1:]  # 排除自身, (B, N, K)

    @staticmethod
    def _gather_neighbors(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """根据索引收集邻居特征"""
        B, N, C = features.shape
        _, _, K = indices.shape

        # 扩展索引
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, K, C)

        # 收集特征
        neighbors = torch.gather(features.unsqueeze(2).expand(-1, -1, K, -1),
                                dim=1, index=indices_expanded)

        return neighbors


class NormalRefinementTransformer(nn.Module):
    """法向量精炼Transformer网络"""

    def __init__(self, feature_dim: int = 128, num_layers: int = 3,
                 num_neighbors: int = 16):
        """
        参数:
            feature_dim: 特征维度
            num_layers: Transformer层数
            num_neighbors: K近邻数量
        """
        super(NormalRefinementTransformer, self).__init__()

        # 输入特征嵌入 (xyz + normal + curvature)
        self.input_embedding = nn.Sequential(
            nn.Linear(7, feature_dim),  # 3(xyz) + 3(normal) + 1(curvature)
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        # Transformer层堆叠
        self.transformer_layers = nn.ModuleList([
            PointTransformerLayer(feature_dim, feature_dim, num_neighbors)
            for _ in range(num_layers)
        ])

        # 法向量预测头
        self.normal_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, xyz: torch.Tensor, normals: torch.Tensor,
                curvature: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            xyz: (B, N, 3) 点坐标
            normals: (B, N, 3) 初始法向量
            curvature: (B, N, 1) 曲率 (可选)

        输出:
            refined_normals: (B, N, 3) 精炼后的法向量
            confidence: (B, N, 1) 置信度分数
        """
        B, N, _ = xyz.shape

        # 计算曲率(如果未提供)
        if curvature is None:
            curvature = self._estimate_curvature(xyz, normals)

        # 输入特征拼接
        input_features = torch.cat([xyz, normals, curvature], dim=-1)  # (B, N, 7)

        # 特征嵌入
        features = self.input_embedding(input_features)  # (B, N, feature_dim)

        # Transformer层
        for layer in self.transformer_layers:
            residual = features
            features = layer(xyz, features)
            features = features + residual  # 残差连接

        # 法向量预测
        delta_normals = self.normal_head(features)  # (B, N, 3)
        refined_normals = F.normalize(normals + delta_normals, p=2, dim=-1)

        # 置信度预测
        confidence = self.confidence_head(features)  # (B, N, 1)

        return refined_normals, confidence

    @staticmethod
    def _estimate_curvature(xyz: torch.Tensor, normals: torch.Tensor,
                           k: int = 8) -> torch.Tensor:
        """估计局部曲率"""
        B, N, _ = xyz.shape

        # 计算K近邻
        dist = torch.cdist(xyz, xyz)
        _, indices = torch.topk(dist, k + 1, dim=-1, largest=False)
        neighbor_indices = indices[:, :, 1:]  # (B, N, k)

        # 收集邻居法向量
        indices_expanded = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbor_normals = torch.gather(normals.unsqueeze(2).expand(-1, -1, k, -1),
                                       dim=1, index=indices_expanded)

        # 法向量变化量 = 平均角度偏差
        normal_expanded = normals.unsqueeze(2)  # (B, N, 1, 3)
        cosine_sim = torch.sum(normal_expanded * neighbor_normals, dim=-1)  # (B, N, k)
        curvature = 1 - torch.mean(torch.abs(cosine_sim), dim=-1, keepdim=True)  # (B, N, 1)

        return curvature


class TransformerNormalRefiner:
    """Transformer法向量精炼器 - 包装类"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 pretrained_path: Optional[str] = None):
        self.device = device
        self.model = NormalRefinementTransformer(
            feature_dim=128,
            num_layers=3,
            num_neighbors=16
        ).to(device)

        # 加载预训练模型
        if pretrained_path is not None:
            try:
                self.model.load_state_dict(torch.load(pretrained_path, map_location=device))
                logger.info(f"已加载预训练模型: {pretrained_path}")
            except Exception as e:
                logger.warning(f"加载预训练模型失败: {e}, 使用随机初始化")

        self.model.eval()
        logger.info(f"Transformer法向量精炼器已加载到设备: {device}")

    def refine_normals(self, points: np.ndarray, normals: np.ndarray,
                      batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """
        精炼法向量

        参数:
            points: (N, 3) 点云坐标
            normals: (N, 3) 初始法向量
            batch_size: 批处理大小

        返回:
            refined_normals: (N, 3) 精炼后的法向量
            confidence: (N,) 置信度分数
        """
        logger.info("使用Transformer精炼法向量...")

        n_points = points.shape[0]
        refined_normals = np.zeros_like(normals)
        confidence = np.zeros(n_points)

        # 分批处理
        n_batches = (n_points + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_points)

                # 转换为张量
                xyz_batch = torch.from_numpy(points[start_idx:end_idx]).float().unsqueeze(0).to(self.device)
                normal_batch = torch.from_numpy(normals[start_idx:end_idx]).float().unsqueeze(0).to(self.device)

                # 前向传播
                refined_batch, conf_batch = self.model(xyz_batch, normal_batch)

                # 转换回NumPy
                refined_normals[start_idx:end_idx] = refined_batch.squeeze(0).cpu().numpy()
                confidence[start_idx:end_idx] = conf_batch.squeeze().cpu().numpy()

                if (i + 1) % 10 == 0:
                    logger.info(f"精炼进度: {i + 1}/{n_batches} 批次")

        logger.info("✅ 法向量精炼完成")
        return refined_normals, confidence

    def filter_by_confidence(self, normals: np.ndarray, confidence: np.ndarray,
                            threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据置信度过滤法向量

        参数:
            normals: (N, 3) 法向量
            confidence: (N,) 置信度
            threshold: 置信度阈值

        返回:
            filtered_normals: 高置信度法向量
            mask: 布尔掩码
        """
        mask = confidence >= threshold
        n_filtered = np.sum(~mask)

        logger.info(f"置信度过滤: 阈值={threshold}, 过滤掉 {n_filtered} 个点 ({100*n_filtered/len(mask):.1f}%)")

        return normals[mask], mask
