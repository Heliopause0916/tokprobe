"""bench_serve（run_single 网络异常与参数校验）的单元测试。

全部基于 mock（patch tokprobe.bench_serve.post_json），
不发起任何真实网络请求。
"""

import os
import sys
import unittest
from unittest.mock import patch

import requests

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tokprobe.bench_serve import run_single, validate_cli_args
from tokprobe.report import mean, percentile, serve_quiet_line


def _args():
    return ("http://127.0.0.1:8082", "test-model", "hi", 16, 30)


class TestRunSingleNetworkErrors(unittest.TestCase):
    @patch("tokprobe.bench_serve.post_json")
    def test_connection_error_returns_error_dict(self, mock_post):
        mock_post.side_effect = \
            requests.exceptions.ConnectionError("connection refused")
        r = run_single(*_args())
        self.assertIn("error", r)
        self.assertIn("ConnectionError", r["error"])
        self.assertEqual(r["prompt_tokens"], 0)
        self.assertEqual(r["completion_tokens"], 0)
        self.assertEqual(r["tok_per_s"], 0.0)
        self.assertGreaterEqual(r["elapsed"], 0.0)

    @patch("tokprobe.bench_serve.post_json")
    def test_httperror_returns_error_dict(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error")
        r = run_single(*_args())
        self.assertIn("error", r)
        self.assertIn("HTTPError", r["error"])
        self.assertEqual(r["completion_tokens"], 0)
        self.assertEqual(r["tok_per_s"], 0.0)

    @patch("tokprobe.bench_serve.post_json")
    def test_success_returns_full_dict_without_error(self, mock_post):
        mock_post.return_value = {"usage": {"prompt_tokens": 8,
                                            "completion_tokens": 16}}
        r = run_single(*_args())
        self.assertIsNone(r["error"])
        self.assertEqual(r["prompt_tokens"], 8)
        self.assertEqual(r["completion_tokens"], 16)
        self.assertGreater(r["tok_per_s"], 0.0)


class TestAggregationWithAllFailures(unittest.TestCase):
    @patch("tokprobe.bench_serve.post_json")
    def test_all_failed_run_single_results_aggregate_safely(self, mock_post):
        """全部请求失败时：聚合统计不抛异常，空样本输出 0.0 风格值。"""
        mock_post.side_effect = \
            requests.exceptions.ConnectionError("boom")
        results = [run_single(*_args()) for _ in range(3)]
        self.assertTrue(all(r.get("error") for r in results))

        # 模拟 main 的聚合统计逻辑
        ok = [r for r in results if not r.get("error")]
        err = [r for r in results if r.get("error")]
        wall = 1.0
        total_ct = sum(r["completion_tokens"] for r in ok)
        total_pt = sum(r["prompt_tokens"] for r in ok)
        avg_tps = total_ct / wall if wall > 0 else 0.0
        per_req_tps = [r["tok_per_s"] for r in ok]
        mean_per_req = mean(per_req_tps)
        p50 = percentile(per_req_tps, 50)

        self.assertEqual(len(ok), 0)
        self.assertEqual(len(err), 3)
        self.assertIn("ConnectionError", err[0]["error"])
        self.assertEqual(total_ct, 0)
        self.assertEqual(total_pt, 0)
        self.assertEqual(avg_tps, 0.0)
        self.assertEqual(mean_per_req, 0.0)
        self.assertEqual(p50, 0.0)

        line = serve_quiet_line(wall, len(ok), len(results), total_ct,
                                total_pt, avg_tps, mean_per_req, p50)
        self.assertIn("requests=0/3", line)


class TestValidateCliArgs(unittest.TestCase):
    def test_concurrency_zero_rejected(self):
        with self.assertRaises(ValueError):
            validate_cli_args(0, 0)

    def test_n_requests_negative_rejected(self):
        with self.assertRaises(ValueError):
            validate_cli_args(4, -1)

    def test_valid_values_accepted(self):
        validate_cli_args(1, 0)      # n_requests=0 表示回落为并发数，合法
        validate_cli_args(4, 8)


if __name__ == "__main__":
    unittest.main()
