"""report 模块（统计与 --quiet 输出）的单元测试。"""

import os
import sys
import unittest

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tokprobe.report import (mean, median, percentile,
                             prefill_quiet_line, serve_quiet_line)


class TestMean(unittest.TestCase):
    def test_mean_basic(self):
        self.assertEqual(mean([1, 2, 3]), 2.0)

    def test_mean_float(self):
        self.assertAlmostEqual(mean([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_mean_single(self):
        self.assertEqual(mean([7]), 7.0)

    def test_mean_empty_returns_zero(self):
        self.assertEqual(mean([]), 0.0)


class TestMedian(unittest.TestCase):
    def test_median_odd(self):
        self.assertEqual(median([3, 1, 2]), 2.0)

    def test_median_single(self):
        self.assertEqual(median([5]), 5.0)

    def test_median_even_takes_midpoint(self):
        # 偶数个样本必须取中间两个值的算术平均（修正点）
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_median_even_two(self):
        self.assertEqual(median([3, 7]), 5.0)

    def test_median_even_float(self):
        self.assertAlmostEqual(median([1.0, 10.0, 20.0, 100.0]), 15.0)

    def test_median_empty_returns_zero(self):
        self.assertEqual(median([]), 0.0)


class TestPercentile(unittest.TestCase):
    def test_p50_equals_median_odd(self):
        xs = [9, 1, 5]
        self.assertEqual(percentile(xs, 50), median(xs))
        self.assertEqual(percentile(xs, 50), 5.0)

    def test_p50_equals_median_even(self):
        # P50 必须与修正后的中位数一致（偶数样本取中点）
        xs = [1, 2, 3, 4]
        self.assertEqual(percentile(xs, 50), median(xs))
        self.assertEqual(percentile(xs, 50), 2.5)

    def test_percentile_min_max(self):
        xs = [4, 2, 8, 6]
        self.assertEqual(percentile(xs, 0), 2.0)
        self.assertEqual(percentile(xs, 100), 8.0)

    def test_percentile_single(self):
        self.assertEqual(percentile([3], 50), 3.0)

    def test_percentile_empty_returns_zero(self):
        self.assertEqual(percentile([], 50), 0.0)


class TestQuietLines(unittest.TestCase):
    def test_serve_quiet_line_format(self):
        line = serve_quiet_line(10.0, 8, 10, 800, 50, 80.0, 100.5, 95.0)
        # 单行、无换行符
        self.assertNotIn("\n", line)
        # 字段名与原脚本保持一致
        for field in ("wall=", "requests=8/10", "total_ct=800",
                      "total_pt=50", "avg_tps=80.0",
                      "per_req_mean=100.5", "per_req_p50=95.0"):
            self.assertIn(field, line)

    def test_prefill_quiet_line_format(self):
        line = prefill_quiet_line(131072, 2, 3, 133000, 45210.5, 45000.2)
        self.assertNotIn("\n", line)
        for field in ("target=131072", "ok=2/3", "emp_pt=133000",
                      "prefill_mean=45210.5", "prefill_med=45000.2",
                      "tok/s"):
            self.assertIn(field, line)


if __name__ == "__main__":
    unittest.main()
