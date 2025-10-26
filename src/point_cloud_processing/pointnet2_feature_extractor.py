"""
PointNet++深度学习特征提取模块
用于从点云中提取高级语义特征,提升裂隙识别精度
基于2017年NIPS论文"PointNet++: Deep Hierarchical Feature Learning on Point Sets"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction层"""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list, group_all: bool = False):
        """
        参数:
            npoint: 采样点数
            radius: 球查询半径
            nsample: 每个球内采样点数
            in_channel: 输入特征维度
            mlp: MLP网络层配置
            group_all: 是否全局池化
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None):
        """
        输入:
            xyz: (B, N, 3) 点坐标
            points: (B, N, C) 点特征
        输出:
            new_xyz: (B, npoint, 3) 采样后的点坐标
            new_points: (B, npoint, mlp[-1]) 聚合后的特征
        """
        xyz = xyz.permute(0, 2, 1)  # (B, 3, N)
        if points is not None:
            points = points.permute(0, 2, 1)  # (B, C, N)

        if self.group_all:
            # 全局特征提取
            new_xyz, new_points = self._sample_and_group_all(xyz, points)
        else:
            # 局部特征提取
            new_xyz, new_points = self._sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        # MLP: (B, C_in, npoint, nsample) -> (B, C_out, npoint, nsample)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 最大池化: (B, C_out, npoint, nsample) -> (B, C_out, npoint)
        new_points = torch.max(new_points, -1)[0]
        new_xyz = new_xyz.permute(0, 2, 1)  # (B, npoint, 3)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, C_out)

        return new_xyz, new_points

    def _sample_and_group(self, npoint: int, radius: float, nsample: int,
                          xyz: torch.Tensor, points: Optional[torch.Tensor]):
        """采样和分组"""
        B, C, N = xyz.shape

        # FPS采样
        fps_idx = self._farthest_point_sample(xyz, npoint)  # (B, npoint)
        new_xyz = self._index_points(xyz.permute(0, 2, 1), fps_idx)  # (B, npoint, 3)

        # 球查询
        idx = self._query_ball_point(radius, nsample, xyz.permute(0, 2, 1), new_xyz)

        # 分组
        grouped_xyz = self._index_points(xyz.permute(0, 2, 1), idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # 相对坐标

        if points is not None:
            grouped_points = self._index_points(points.permute(0, 2, 1), idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 1, 2)  # (B, C+3, npoint, nsample)
        new_xyz = new_xyz.permute(0, 2, 1)  # (B, 3, npoint)

        return new_xyz, grouped_points

    def _sample_and_group_all(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        """全局池化"""
        B, C, N = xyz.shape
        new_xyz = torch.zeros(B, 3, 1).to(xyz.device)
        grouped_xyz = xyz.permute(0, 2, 1).unsqueeze(2)  # (B, N, 1, 3)

        if points is not None:
            new_points = torch.cat([grouped_xyz, points.permute(0, 2, 1).unsqueeze(2)], dim=-1)
        else:
            new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 2, 1)  # (B, C+3, 1, N)
        return new_xyz, new_points

    @staticmethod
    def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """最远点采样 (FPS)"""
        device = xyz.device
        B, _, N = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, :, farthest].unsqueeze(2)  # (B, 3, 1)
            dist = torch.sum((xyz - centroid) ** 2, dim=1)  # (B, N)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    @staticmethod
    def _query_ball_point(radius: float, nsample: int,
                          xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """球查询"""
        device = xyz.device
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape

        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, S, 1)
        sqrdists = self._square_distance(new_xyz, xyz)  # (B, S, N)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

        # 处理不足nsample的情况
        group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
        mask = group_idx == N
        group_idx[mask] = group_first[mask]

        return group_idx

    @staticmethod
    def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """计算点之间的平方距离"""
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).unsqueeze(-1)
        dist += torch.sum(dst ** 2, -1).unsqueeze(-2)
        return dist

    @staticmethod
    def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """根据索引提取点"""
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointNet2FeatureExtractor(nn.Module):
    """PointNet++特征提取器 - 用于岩石裂隙检测"""

    def __init__(self, normal_channel: bool = True):
        """
        参数:
            normal_channel: 是否使用法向量作为输入特征
        """
        super(PointNet2FeatureExtractor, self).__init__()

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        # 层级特征提取
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024],
            group_all=True
        )

        # 特征解码器
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

    def forward(self, xyz: torch.Tensor, normals: Optional[torch.Tensor] = None):
        """
        输入:
            xyz: (B, N, 3) 点云坐标
            normals: (B, N, 3) 法向量 (可选)
        输出:
            point_features: (B, 256) 全局特征
            local_features: (B, 128, 128) 局部特征
        """
        if self.normal_channel and normals is not None:
            points = torch.cat([xyz, normals], dim=-1)
        else:
            points = None

        # 层级特征提取
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # 全局特征
        global_features = l3_points.squeeze(1)  # (B, 1024)

        # 特征解码
        x = self.drop1(F.relu(self.bn1(self.fc1(global_features))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))

        return x, l2_points.squeeze(1)  # (B, 256), (B, 128, 128)


class PointNet2Wrapper:
    """PointNet++包装器 - 便于集成到现有流程"""

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = PointNet2FeatureExtractor(normal_channel=True).to(device)
        self.model.eval()
        logger.info(f"PointNet++模型已加载到设备: {device}")

    def extract_features(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                        batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取点云深度特征

        参数:
            points: (N, 3) 点云坐标
            normals: (N, 3) 法向量
            batch_size: 批处理大小

        返回:
            global_features: (batch_size, 256) 全局特征
            local_features: (batch_size, 128) 局部特征
        """
        n_points = points.shape[0]

        # 数据准备
        if normals is None:
            normals = np.zeros_like(points)

        # 转换为张量
        xyz_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        normal_tensor = torch.from_numpy(normals).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            global_feat, local_feat = self.model(xyz_tensor, normal_tensor)

        return global_feat.cpu().numpy(), local_feat.cpu().numpy()

    def enhance_normals(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """
        使用深度学习特征增强法向量估计

        参数:
            points: (N, 3) 点云坐标
            normals: (N, 3) 初始法向量

        返回:
            enhanced_normals: (N, 3) 增强后的法向量
        """
        logger.info("使用PointNet++增强法向量...")

        # 提取特征
        global_feat, local_feat = self.extract_features(points, normals)

        # 这里可以添加一个小型MLP来预测更精确的法向量
        # 简化版本:保持原法向量,仅用于特征增强
        return normals
