"""
生成更真实的岩石边坡点云测试数据
"""

import numpy as np

def generate_rock_slope_data(filename, n_points=1000):
    """生成模拟岩石边坡点云数据"""

    # 设置随机种子以便重现结果
    np.random.seed(42)

    points = []

    # 主要坡面 (倾向70度，倾角45度)
    for i in range(n_points // 3):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 8)
        z = 5 - 0.8 * x + np.random.normal(0, 0.1)
        points.append([x, y, z])

    # 第一组节理面 (近垂直)
    for i in range(n_points // 3):
        x = np.random.uniform(2, 8) + np.random.normal(0, 0.1)
        y = np.random.uniform(0, 8)
        z = np.random.uniform(0, 6)
        points.append([x, y, z])

    # 第二组节理面 (水平)
    for i in range(n_points // 3):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 8)
        z = 3 + np.random.normal(0, 0.1)
        points.append([x, y, z])

    # 保存数据
    points = np.array(points)
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
    print(f"已生成 {len(points)} 个点的测试数据: {filename}")

if __name__ == "__main__":
    generate_rock_slope_data("data-files/sample_data/rock_slope.txt", 1000)