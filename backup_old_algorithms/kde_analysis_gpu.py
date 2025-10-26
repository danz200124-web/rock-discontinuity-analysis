"""
GPU加速的核密度估计分析模块
使用CuPy实现CUDA加速，支持多GPU并行
用于识别主要不连续面组
"""

import numpy as np
import logging
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError as e:
    GPU_AVAILABLE = False
    cp = np
    logging.warning(f"CuPy导入失败: {e}")

logger = logging.getLogger(__name__)


class KDEAnalyzerGPU:
    """
    GPU加速的基于核密度估计的极点分析器
    在立体投影上进行密度分析，识别主要不连续面组
    """

    def __init__(self, device_ids=None, use_multi_gpu=False):
        """
        初始化GPU加速的KDE分析器

        参数:
            device_ids: 使用的GPU设备ID列表，None表示使用所有可用GPU
            use_multi_gpu: 是否启用多GPU并行
        """
        self.density = None
        self.main_poles = None
        self.kde_model = None

        if not GPU_AVAILABLE:
            logger.warning("CuPy未安装，将回退到CPU模式")
            self.use_gpu = False
            return

        self.use_gpu = True
        self.use_multi_gpu = use_multi_gpu

        # 设置GPU设备
        if device_ids is None:
            self.device_ids = list(range(cp.cuda.runtime.getDeviceCount()))
        else:
            self.device_ids = device_ids

        # 获取GPU内存信息
        gpu_info = []
        for gpu_id in self.device_ids:
            with cp.cuda.Device(gpu_id):
                mem_info = cp.cuda.Device().mem_info
                total_mem = mem_info[1] / (1024**3)  # GB
                gpu_info.append(f"GPU {gpu_id}: {total_mem:.1f}GB")

        logger.info(f"🚀 GPU加速已启用 - {'多GPU并行' if use_multi_gpu else '单GPU'}")
        logger.info(f"   设备: {', '.join(gpu_info)}")

    def analyze(self, poles, bin_size=256, min_angle=20, max_sets=10,
                subsample_size=None, primary_gpu=0):
        """
        GPU加速分析极点分布，识别主要不连续面组

        参数:
            poles: 极点坐标数组 (N, 2) - 立体投影坐标
            bin_size: 密度网格大小
            min_angle: 主要组之间最小角度（度）
            max_sets: 最大不连续面组数
            subsample_size: 子采样大小，None表示使用全部数据
            primary_gpu: 主GPU设备ID

        返回:
            density: 密度矩阵 (CPU数组)
            main_poles: 主要极点列表
        """
        logger.info("开始GPU加速核密度估计分析...")

        if not self.use_gpu:
            logger.warning("GPU不可用，请使用CPU版本的KDEAnalyzer")
            return None, []

        # 设置主GPU
        with cp.cuda.Device(primary_gpu):
            # 创建网格（GPU）
            x_grid = cp.linspace(-1, 1, bin_size)
            y_grid = cp.linspace(-1, 1, bin_size)
            xx, yy = cp.meshgrid(x_grid, y_grid)
            grid_points = cp.vstack([xx.ravel(), yy.ravel()])

            # 转移数据到GPU
            poles_gpu = cp.asarray(poles, dtype=cp.float32)

            # 向量化过滤有效极点（在单位圆内）
            valid_mask = cp.sum(poles_gpu**2, axis=1) <= 1
            valid_poles = poles_gpu[valid_mask]

            if len(valid_poles) < 10:
                logger.warning("有效极点数量太少，无法进行密度分析")
                return None, []

            logger.info(f"有效极点数量: {len(valid_poles)}")

            # 子采样（如果需要）
            if subsample_size is not None and len(valid_poles) > subsample_size:
                logger.info(f"子采样至 {subsample_size} 个点")
                indices = cp.random.choice(len(valid_poles), subsample_size, replace=False)
                valid_poles = valid_poles[indices]
            else:
                logger.info(f"使用全部 {len(valid_poles)} 个极点")

            # GPU加速KDE计算
            logger.info("GPU加速KDE计算...")
            density_flat = self._compute_kde_gpu(valid_poles, grid_points)

            # 转换为2D密度矩阵
            self.density = cp.asnumpy(density_flat.reshape(xx.shape))
            xx_cpu = cp.asnumpy(xx)
            yy_cpu = cp.asnumpy(yy)

        # 寻找密度峰值（在CPU上）
        self.main_poles = self._find_density_peaks(
            xx_cpu, yy_cpu, self.density,
            min_angle=min_angle,
            max_sets=max_sets
        )

        logger.info(f"识别到 {len(self.main_poles)} 个主要不连续面组")

        return self.density, self.main_poles

    def _compute_kde_gpu(self, poles, grid_points):
        """
        多GPU加速的KDE计算（真正的多GPU数据并行）

        参数:
            poles: 有效极点 (GPU数组)
            grid_points: 网格点 (GPU数组)

        返回:
            density: 密度值 (GPU数组)
        """
        n_poles = poles.shape[0]
        n_grid = grid_points.shape[1]

        # 计算带宽 (Scott's rule)
        d = poles.shape[1]  # 维度
        bw_factor = n_poles ** (-1.0 / (d + 4))
        cov = cp.cov(poles.T)
        bandwidth = bw_factor * cp.sqrt(cp.diag(cov))

        logger.info(f"KDE带宽: {cp.asnumpy(bandwidth)}")

        # 如果启用多GPU并行
        if self.use_multi_gpu and len(self.device_ids) > 1:
            return self._compute_kde_multi_gpu(poles, grid_points, bandwidth)
        else:
            return self._compute_kde_single_gpu(poles, grid_points, bandwidth)

    def _compute_kde_multi_gpu(self, poles, grid_points, bandwidth):
        """
        真正的多GPU并行KDE计算 - 使用线程池实现并行

        将评估点分配到多个GPU上并行计算
        """
        import threading

        n_poles = poles.shape[0]
        n_grid = grid_points.shape[1]
        n_gpus = len(self.device_ids)

        logger.info(f"🚀 启动多GPU并行计算: {n_gpus} 个GPU")

        # 将网格点均匀分配到各个GPU
        grid_per_gpu = (n_grid + n_gpus - 1) // n_gpus

        # 在主GPU上准备数据（转为CPU避免跨GPU复制问题）
        poles_cpu = cp.asnumpy(poles)
        grid_cpu = cp.asnumpy(grid_points)
        bandwidth_cpu = cp.asnumpy(bandwidth)

        # 存储每个GPU的结果
        gpu_results = {}

        def compute_on_gpu(gpu_id, start_idx, end_idx):
            """在指定GPU上计算密度"""
            with cp.cuda.Device(gpu_id):
                # 将数据复制到该GPU
                poles_gpu = cp.array(poles_cpu, dtype=cp.float32)
                grid_gpu = cp.array(grid_cpu[:, start_idx:end_idx], dtype=cp.float32)
                bandwidth_gpu = cp.array(bandwidth_cpu, dtype=cp.float32)

                logger.info(f"GPU {gpu_id}: 开始处理网格点 {start_idx}-{end_idx} (共{end_idx-start_idx}个)")

                # 在该GPU上计算密度
                density_gpu = self._kde_kernel_gpu(poles_gpu, grid_gpu, bandwidth_gpu)

                # 将结果转为CPU数组（避免跨GPU传输）
                density_cpu = cp.asnumpy(density_gpu)

                # 存储结果
                gpu_results[gpu_id] = (start_idx, end_idx, density_cpu)

                logger.info(f"GPU {gpu_id}: 计算完成")

                # 清理GPU内存
                del poles_gpu, grid_gpu, bandwidth_gpu, density_gpu
                cp.get_default_memory_pool().free_all_blocks()

        # 创建线程池，每个GPU一个线程
        threads = []
        for gpu_idx, gpu_id in enumerate(self.device_ids):
            # 计算该GPU负责的网格点范围
            start_idx = gpu_idx * grid_per_gpu
            end_idx = min((gpu_idx + 1) * grid_per_gpu, n_grid)

            if start_idx >= n_grid:
                break

            # 启动线程
            thread = threading.Thread(
                target=compute_on_gpu,
                args=(gpu_id, start_idx, end_idx)
            )
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 在主GPU上合并所有结果
        with cp.cuda.Device(self.device_ids[0]):
            density = cp.zeros(n_grid, dtype=cp.float32)
            for gpu_id in sorted(gpu_results.keys()):
                start_idx, end_idx, density_cpu = gpu_results[gpu_id]
                density[start_idx:end_idx] = cp.array(density_cpu)
                logger.info(f"GPU {gpu_id}: 结果已合并")

        logger.info(f"✅ 多GPU并行计算完成")
        return density

    def _compute_kde_single_gpu(self, poles, grid_points, bandwidth):
        """
        单GPU的KDE计算（原有逻辑）
        """
        n_grid = grid_points.shape[1]

        # 分批计算以节省显存
        batch_size = min(50000, n_grid)
        n_batches = (n_grid + batch_size - 1) // batch_size

        density = cp.zeros(n_grid, dtype=cp.float32)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_grid)
            batch_grid = grid_points[:, start_idx:end_idx]

            # 计算每个网格点的密度
            batch_density = self._kde_kernel_gpu(poles, batch_grid, bandwidth)
            density[start_idx:end_idx] = batch_density

            if (i + 1) % max(1, n_batches // 10) == 0:
                logger.info(f"KDE进度: {100 * (i + 1) // n_batches}%")

        return density

    def _kde_kernel_gpu(self, data, eval_points, bandwidth):
        """
        GPU加速的KDE核函数计算（高斯核）- 内存优化版

        参数:
            data: 数据点 (n_data, d) GPU数组
            eval_points: 评估点 (d, n_eval) GPU数组
            bandwidth: 带宽 (d,) GPU数组

        返回:
            density: 密度值 (n_eval,) GPU数组
        """
        n_data = data.shape[0]
        n_eval = eval_points.shape[1]
        d = data.shape[1]

        # 根据可用显存动态调整批次大小
        mem_info = cp.cuda.Device().mem_info
        available_mem = mem_info[0]  # 可用显存（字节）

        # 更保守的内存估算
        # 预估内存需求：data_batch * eval_batch * 4 bytes (float32) * 5 (中间数组+安全余量)
        # 只使用30%的可用显存
        safe_mem = available_mem * 0.3

        # 首先确定评估批次大小（较小）
        eval_batch_size = 100  # 固定较小的评估批次

        # 然后计算数据批次大小
        max_data_batch = int(safe_mem / (eval_batch_size * 4 * 5))
        data_batch_size = max(500, min(max_data_batch, 50000))  # 限制在500-50000之间

        logger.info(f"GPU KDE内存优化: 可用显存={available_mem/(1024**3):.2f}GB, "
                   f"数据批次={data_batch_size}, 评估批次={eval_batch_size}, "
                   f"预估每批内存={(data_batch_size * eval_batch_size * 4 * 5)/(1024**3):.2f}GB")

        density = cp.zeros(n_eval, dtype=cp.float32)
        normalization = (2 * cp.pi) ** (-d / 2) * cp.prod(bandwidth) ** (-1)

        total_eval_batches = (n_eval + eval_batch_size - 1) // eval_batch_size
        logger.info(f"KDE计算: {n_data} 数据点 × {n_eval} 评估点, "
                   f"共 {total_eval_batches} 个评估批次")

        # 双层分批：先分批评估点，再分批数据点
        for eval_batch_idx, i in enumerate(range(0, n_eval, eval_batch_size)):
            end_idx = min(i + eval_batch_size, n_eval)
            batch_eval = eval_points[:, i:end_idx]  # (d, current_eval_batch)
            batch_density = cp.zeros(end_idx - i, dtype=cp.float32)

            # 对数据点分批处理并累加（保持精度）
            total_data_batches = (n_data + data_batch_size - 1) // data_batch_size
            for data_batch_idx, j in enumerate(range(0, n_data, data_batch_size)):
                data_end = min(j + data_batch_size, n_data)
                batch_data = data[j:data_end, :]  # (current_data_batch, d)

                # 广播计算距离
                # batch_data: (current_data_batch, d) -> (current_data_batch, d, 1)
                # batch_eval: (d, current_eval_batch) -> (1, d, current_eval_batch)
                data_expanded = batch_data[:, :, cp.newaxis]
                eval_expanded = batch_eval[cp.newaxis, :, :]

                # 计算标准化距离
                diff = (data_expanded - eval_expanded) / bandwidth[cp.newaxis, :, cp.newaxis]

                # 高斯核
                exponent = -0.5 * cp.sum(diff ** 2, axis=1)  # (current_data_batch, current_eval_batch)
                kernels = cp.exp(exponent)

                # 累加这批数据点的贡献（保持精度）
                batch_density += cp.sum(kernels, axis=0)

                # 清理显存
                del data_expanded, eval_expanded, diff, exponent, kernels
                cp.get_default_memory_pool().free_all_blocks()

                # 进度报告
                if (data_batch_idx + 1) % max(1, total_data_batches // 5) == 0:
                    progress = (eval_batch_idx * total_data_batches + data_batch_idx + 1) / (total_eval_batches * total_data_batches) * 100
                    logger.info(f"KDE进度: {progress:.1f}% (评估批次 {eval_batch_idx+1}/{total_eval_batches}, "
                              f"数据批次 {data_batch_idx+1}/{total_data_batches})")

            # 归一化：除以总数据点数
            density[i:end_idx] = normalization * batch_density / n_data

            # 清理显存
            del batch_eval, batch_density
            cp.get_default_memory_pool().free_all_blocks()

        return density

    def _find_density_peaks(self, xx, yy, density, min_angle=20, max_sets=10):
        """
        寻找密度峰值点（CPU版本）

        参数:
            xx, yy: 网格坐标 (CPU数组)
            density: 密度矩阵 (CPU数组)
            min_angle: 峰值之间最小角度（度）
            max_sets: 最大峰值数

        返回:
            peaks: 峰值点列表
        """
        # 寻找局部最大值
        flat_density = density.flatten()
        sorted_indices = np.argsort(flat_density)[::-1]

        peaks = []
        min_angle_rad = np.radians(min_angle)

        for idx in sorted_indices:
            if len(peaks) >= max_sets:
                break

            # 获取峰值坐标
            i, j = np.unravel_index(idx, density.shape)
            x, y = xx[i, j], yy[i, j]

            # 检查是否在单位圆内
            if x ** 2 + y ** 2 > 1:
                continue

            # 检查与已有峰值的角度
            too_close = False
            for px, py, _ in peaks:
                angle = self._calculate_angle_between_poles((x, y), (px, py))
                if angle < min_angle_rad:
                    too_close = True
                    break

            if not too_close:
                peaks.append((x, y, flat_density[idx]))

        # 转换为产状
        main_poles = []
        for x, y, density_value in peaks:
            dip_dir, dip = self._stereo_to_orientation(x, y)
            main_poles.append({
                'stereo_x': x,
                'stereo_y': y,
                'dip_direction': dip_dir,
                'dip': dip,
                'density': density_value
            })

        return main_poles

    def _calculate_angle_between_poles(self, pole1, pole2):
        """计算两个极点之间的角度"""
        v1 = self._stereo_to_vector(pole1[0], pole1[1])
        v2 = self._stereo_to_vector(pole2[0], pole2[1])
        cos_angle = np.clip(np.dot(v1, v2), -1, 1)
        return np.arccos(cos_angle)

    def _stereo_to_vector(self, x, y):
        """立体投影坐标转换为单位向量"""
        r2 = x ** 2 + y ** 2
        if r2 > 1:
            return np.array([0, 0, 1])

        z = (1 - r2) / (1 + r2)
        scale = 2 / (1 + r2)
        return np.array([x * scale, y * scale, z])

    def _stereo_to_orientation(self, x, y):
        """立体投影坐标转换为产状"""
        vector = self._stereo_to_vector(x, y)

        # 计算倾向
        dip_direction = np.degrees(np.arctan2(vector[1], vector[0]))
        if dip_direction < 0:
            dip_direction += 360

        # 计算倾角
        dip = np.degrees(np.arccos(np.abs(vector[2])))

        return dip_direction, dip
