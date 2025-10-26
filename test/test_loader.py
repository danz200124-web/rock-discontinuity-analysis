"""
点云加载器测试
"""

import unittest
import numpy as np
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.point_cloud_processing.loader import PointCloudLoader


class TestPointCloudLoader(unittest.TestCase):
    """测试点云加载器"""

    def setUp(self):
        """初始化测试"""
        self.loader = PointCloudLoader()

    def test_load_ascii(self):
        """测试ASCII文件加载"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # 写入测试数据
            f.write("0.0 0.0 0.0\n")
            f.write("1.0 0.0 0.0\n")
            f.write("0.0 1.0 0.0\n")
            f.write("0.0 0.0 1.0\n")
            temp_file = f.name

        try:
            # 加载点云
            pc = self.loader.load(temp_file)

            # 验证
            self.assertIsNotNone(pc)
            self.assertEqual(len(pc.points), 4)

        finally:
            # 清理
            os.unlink(temp_file)

    def test_invalid_file(self):
        """测试无效文件"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load("nonexistent.txt")


if __name__ == '__main__':
    unittest.main()