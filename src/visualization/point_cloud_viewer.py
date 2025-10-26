"""
3D点云查看器
基于Open3D的点云可视化
"""

import open3d as o3d
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PointCloudViewer:
    """3D点云查看器"""

    def __init__(self):
        pass

    def show(self, point_cloud, discontinuity_sets=None, window_name="点云查看器"):
        """
        显示3D点云

        参数:
            point_cloud: Open3D点云对象
            discontinuity_sets: 不连续面组字典（可选）
            window_name: 窗口名称
        """
        logger.info("显示3D点云...")

        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=800)

        # 添加点云
        vis.add_geometry(point_cloud)

        # 如果有不连续面组，用不同颜色显示
        if discontinuity_sets:
            self._color_by_sets(point_cloud, discontinuity_sets)

        # 设置视角
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 2.0

        # 运行可视化器
        logger.info("按Q键退出可视化器")
        vis.run()
        vis.destroy_window()

    def show_with_planes(self, point_cloud, discontinuity_sets=None, window_name="点云与裂隙面查看器"):
        """
        显示3D点云及裂隙面

        参数:
            point_cloud: Open3D点云对象
            discontinuity_sets: 不连续面组字典（可选）
            window_name: 窗口名称
        """
        logger.info("显示3D点云及裂隙面...")

        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=800)

        # 创建点云副本（避免修改原始点云）
        pcd_copy = o3d.geometry.PointCloud(point_cloud)

        # 如果有不连续面组，用不同颜色显示
        if discontinuity_sets:
            self._color_by_sets(pcd_copy, discontinuity_sets)

        # 添加点云
        vis.add_geometry(pcd_copy)

        # 添加裂隙面平面
        if discontinuity_sets:
            self._add_discontinuity_planes(vis, discontinuity_sets)

        # 设置视角
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 2.0

        # 运行可视化器
        logger.info("按Q键退出可视化器")
        vis.run()
        vis.destroy_window()

    def _add_discontinuity_planes(self, vis, discontinuity_sets):
        """
        添加不连续面平面到可视化器

        参数:
            vis: Open3D可视化器
            discontinuity_sets: 不连续面组字典
        """
        # 预定义颜色（半透明）
        set_colors = [
            [1, 0, 0],      # 红色
            [0, 1, 0],      # 绿色
            [0, 0, 1],      # 蓝色
            [1, 1, 0],      # 黄色
            [1, 0, 1],      # 紫色
            [0, 1, 1],      # 青色
            [1, 0.5, 0],    # 橙色
            [0.5, 0, 1],    # 紫蓝色
        ]

        items = discontinuity_sets.items() if isinstance(discontinuity_sets, dict) else enumerate(discontinuity_sets)

        for i, (set_id, set_data) in enumerate(items):
            if isinstance(discontinuity_sets, list):
                set_id, set_data = i, set_data

            # 如果有平面参数，创建平面网格
            if 'plane' in set_data:
                plane_params = set_data['plane']
                mesh = self._create_plane_mesh(plane_params, size=5.0)

                # 设置颜色
                color_idx = i % len(set_colors)
                mesh.paint_uniform_color(set_colors[color_idx])

                vis.add_geometry(mesh)

    def _color_by_sets(self, point_cloud, discontinuity_sets):
        """
        根据不连续面组给点云着色

        参数:
            point_cloud: Open3D点云对象
            discontinuity_sets: 不连续面组列表或字典
        """
        points = np.asarray(point_cloud.points)

        # 检查点云是否为空
        if len(points) == 0:
            logger.warning("点云为空，跳过着色")
            return

        colors = np.ones((len(points), 3), dtype=np.float64) * 0.5  # 确保形状正确

        # 预定义颜色（扩展到20种）
        set_colors = [
            [1, 0, 0],        # 红色
            [0, 1, 0],        # 绿色
            [0, 0, 1],        # 蓝色
            [1, 1, 0],        # 黄色
            [1, 0, 1],        # 紫色
            [0, 1, 1],        # 青色
            [1, 0.5, 0],      # 橙色
            [0.5, 0, 1],      # 紫蓝色
            [1, 0.75, 0.8],   # 粉色
            [0.5, 1, 0],      # 黄绿色
            [0, 0.5, 1],      # 天蓝色
            [1, 0, 0.5],      # 玫红色
            [0.75, 0.75, 0],  # 橄榄色
            [0.5, 0, 0.5],    # 深紫色
            [0, 0.75, 0.75],  # 深青色
            [1, 0.65, 0],     # 深橙色
            [0.6, 0.4, 0.2],  # 棕色
            [0.8, 0.8, 0.8],  # 浅灰色
            [0.3, 0.7, 0.3],  # 深绿色
            [0.7, 0.3, 0.7],  # 深粉色
        ]

        # 如果是列表类型（从detector返回）
        if isinstance(discontinuity_sets, list):
            # 使用索引直接着色，避免KD树查询的不准确性
            logger.info("开始为不连续面分组着色...")
            logger.info(f"总点数: {len(points)}, 不连续面数量: {len(discontinuity_sets)}")

            # 按set_id分组
            set_groups = {}
            for disc in discontinuity_sets:
                set_id = disc.get('set_id', 0)
                if set_id not in set_groups:
                    set_groups[set_id] = []
                set_groups[set_id].append(disc)

            logger.info(f"分组数量: {len(set_groups)}")

            # 统计着色信息
            total_colored = 0
            colored_by_set = {}
            colored_indices = set()  # 记录已着色的点，避免重复计数

            # 为每组着色
            for i, (set_id, discs) in enumerate(sorted(set_groups.items())):
                color_idx = (set_id - 1) % len(set_colors)
                color = set_colors[color_idx]
                set_colored = 0

                # 遍历该组的所有不连续面
                for disc in discs:
                    # 优先使用索引（新版detector返回）
                    if 'indices' in disc:
                        indices = disc['indices']
                        colors[indices] = color
                        # 只统计新着色的点
                        for idx in indices:
                            if idx not in colored_indices:
                                colored_indices.add(idx)
                                total_colored += 1
                        set_colored += len(indices)
                    # 向后兼容：使用KD树查找（旧版本）
                    elif 'points' in disc:
                        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
                        disc_points = disc['points']
                        for disc_pt in disc_points:
                            [k, idx, dist] = pcd_tree.search_knn_vector_3d(disc_pt, 3)
                            if k > 0:
                                for j in range(k):
                                    if dist[j] < 0.01:
                                        if np.allclose(colors[idx[j]], 0.5) or dist[j] < 0.001:
                                            colors[idx[j]] = color
                                            if idx[j] not in colored_indices:
                                                colored_indices.add(idx[j])
                                                total_colored += 1
                                            set_colored += 1

                colored_by_set[set_id] = set_colored
                logger.info(f"第{set_id}组: {len(discs)}个不连续面, 着色{set_colored}个点")

            if len(points) > 0:
                colored_ratio = total_colored / len(points) * 100
                logger.info(f"着色完成：{total_colored}/{len(points)} 个点 ({colored_ratio:.1f}%)")
            else:
                logger.warning("点云为空，无法着色")

        # 如果是字典类型（旧格式）
        elif isinstance(discontinuity_sets, dict):
            for i, (set_id, set_data) in enumerate(discontinuity_sets.items()):
                color_idx = i % len(set_colors)
                if 'indices' in set_data:
                    colors[set_data['indices']] = set_colors[color_idx]

        # 设置颜色
        try:
            logger.info(f"准备设置颜色：points={len(points)}, colors shape={colors.shape}, dtype={colors.dtype}")
            logger.info(f"colors范围: min={colors.min()}, max={colors.max()}")
            logger.info(f"colors示例: {colors[:3] if len(colors) > 0 else 'empty'}")
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            logger.info("颜色设置成功")
        except Exception as e:
            logger.error(f"设置颜色失败: {type(e).__name__}: {str(e)}")
            logger.error(f"详细信息 - colors: shape={colors.shape}, dtype={colors.dtype}, is_contiguous={colors.flags['C_CONTIGUOUS']}")
            raise

    def save_screenshot(self, point_cloud, output_path, discontinuity_sets=None):
        """
        保存点云截图

        参数:
            point_cloud: Open3D点云对象
            output_path: 输出路径
            discontinuity_sets: 不连续面组字典（可选）
        """
        logger.info(f"保存点云截图: {output_path}")

        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 不显示窗口

        # 添加点云
        if discontinuity_sets:
            self._color_by_sets(point_cloud, discontinuity_sets)
        vis.add_geometry(point_cloud)

        # 设置视角
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([1, 1, 1])  # 白色背景
        opt.point_size = 3.0

        # 保存截图
        vis.capture_screen_image(str(output_path))
        vis.destroy_window()

    def create_mesh_visualization(self, point_cloud, discontinuity_planes=None):
        """
        创建包含不连续面的可视化

        参数:
            point_cloud: Open3D点云对象
            discontinuity_planes: 不连续面平面列表
        """
        geometries = [point_cloud]

        if discontinuity_planes:
            for i, plane in enumerate(discontinuity_planes):
                # 创建平面网格
                mesh = self._create_plane_mesh(plane, size=5.0)

                # 设置半透明颜色
                color_idx = i % 8
                colors = [
                    [1, 0, 0, 0.3],      # 红色
                    [0, 1, 0, 0.3],      # 绿色
                    [0, 0, 1, 0.3],      # 蓝色
                    [1, 1, 0, 0.3],      # 黄色
                    [1, 0, 1, 0.3],      # 紫色
                    [0, 1, 1, 0.3],      # 青色
                    [1, 0.5, 0, 0.3],    # 橙色
                    [0.5, 0, 1, 0.3],    # 紫蓝色
                ]
                mesh.paint_uniform_color(colors[color_idx][:3])
                geometries.append(mesh)

        # 显示
        o3d.visualization.draw_geometries(
            geometries,
            window_name="点云与不连续面",
            width=1200,
            height=800
        )

    def _create_plane_mesh(self, plane_params, size=5.0):
        """
        创建平面网格

        参数:
            plane_params: 平面参数 [a, b, c, d] (ax + by + cz + d = 0)
            size: 平面大小

        返回:
            Open3D三角网格对象
        """
        a, b, c, d = plane_params
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # 找到平面上的一点
        if abs(c) > 1e-6:
            point_on_plane = np.array([0, 0, -d/c])
        elif abs(b) > 1e-6:
            point_on_plane = np.array([0, -d/b, 0])
        else:
            point_on_plane = np.array([-d/a, 0, 0])

        # 创建两个正交向量
        if abs(normal[2]) < 0.9:
            v1 = np.cross(normal, [0, 0, 1])
        else:
            v1 = np.cross(normal, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)

        # 创建平面顶点
        vertices = [
            point_on_plane + size * (-v1 - v2),
            point_on_plane + size * (v1 - v2),
            point_on_plane + size * (v1 + v2),
            point_on_plane + size * (-v1 + v2),
        ]

        # 创建三角形
        triangles = [[0, 1, 2], [0, 2, 3]]

        # 创建网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()

        return mesh

    def show_with_normals(self, point_cloud, normals=None, normal_scale=0.1):
        """
        显示带法向量的点云

        参数:
            point_cloud: Open3D点云对象
            normals: 法向量数组（可选）
            normal_scale: 法向量显示比例
        """
        if normals is None and not point_cloud.has_normals():
            logger.warning("点云没有法向量信息")
            self.show(point_cloud)
            return

        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="点云与法向量", width=1200, height=800)

        # 添加点云
        vis.add_geometry(point_cloud)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        # 如果有自定义法向量，创建线段几何体
        if normals is not None:
            points = np.asarray(point_cloud.points)
            # 采样显示法向量（避免太密集）
            step = max(1, len(points) // 1000)
            sampled_points = points[::step]
            sampled_normals = normals[::step]

            lines = []
            line_points = []
            for i, (point, normal) in enumerate(zip(sampled_points, sampled_normals)):
                line_points.append(point)
                line_points.append(point + normal * normal_scale)
                lines.append([2*i, 2*i+1])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([1, 0, 0])  # 红色法向量

            vis.add_geometry(line_set)

        # 运行可视化器
        logger.info("按Q键退出可视化器")
        vis.run()
        vis.destroy_window()