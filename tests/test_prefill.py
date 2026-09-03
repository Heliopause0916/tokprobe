"""bench_prefill（build_prompt / measure_prefill / parse_target_tokens）单元测试。

被测逻辑不涉及任何真实网络请求（网络部分一律 mock post_json）。
"""

import os
import sys
import unittest
from unittest.mock import patch

import requests

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tokprobe.bench_prefill import (build_prompt, measure_prefill,
                                    parse_target_tokens)
from tokprobe.report import mean, median, prefill_quiet_line


class TestBuildPrompt(unittest.TestCase):
    def test_reproducible_same_seed(self):
        # 同一 seed 输出完全一致
        a = build_prompt(42, 4000)
        b = build_prompt(42, 4000)
        self.assertEqual(a, b)

    def test_reproducible_different_target_same_seed(self):
        # 同一 seed、不同目标长度的前缀一致（同 PRNG 流）
        shorter = build_prompt(7, 800)
        longer = build_prompt(7, 8000)
        self.assertTrue(longer.startswith(shorter[:800]))

    def test_different_seed_prefix_differs(self):
        # 不同 seed 必须产生不同文本前缀（规避 vLLM 前缀缓存）
        a = build_prompt(0, 4000)
        b = build_prompt(1, 4000)
        # 取靠近文本开头的一段比较，切勿复用同一常量导致字符串拼接误判
        k = 300
        self.assertNotEqual(a[:k], b[:k])

    def test_length_within_target(self):
        for seed, target in [(0, 1), (3, 100), (12, 4096), (99, 131072)]:
            text = build_prompt(seed, target)
            self.assertGreaterEqual(len(text), 1)
            self.assertLessEqual(len(text), target)

    def test_small_target_nonempty(self):
        text = build_prompt(0, 1)
        self.assertGreaterEqual(len(text), 1)

    def test_all_ascii(self):
        text = build_prompt(5, 2000)
        # ASCII：便于 BPE 稳定、贴近 1 字符 ≈ 1 token
        text.encode("ascii")


class TestMeasurePrefillNetworkErrors(unittest.TestCase):
    def _call(self):
        return measure_prefill("http://127.0.0.1:8082",
                               "test-model", "hi", 1, 30)

    @patch("tokprobe.bench_prefill.post_json")
    def test_connection_error_returns_error_dict(self, mock_post):
        mock_post.side_effect = \
            requests.exceptions.ConnectionError("connection refused")
        r = self._call()
        self.assertIn("error", r)
        self.assertIn("ConnectionError", r["error"])
        self.assertEqual(r["prompt_tokens"], 0)
        self.assertEqual(r["completion_tokens"], 0)
        self.assertEqual(r["prefill_tps"], 0.0)
        self.assertEqual(r["ms_per_token"], 0.0)
        self.assertGreaterEqual(r["elapsed"], 0.0)

    @patch("tokprobe.bench_prefill.post_json")
    def test_httperror_returns_error_dict(self, mock_post):
        mock_post.side_effect = requests.exceptions.HTTPError(
            "400 Bad Request")
        r = self._call()
        self.assertIn("error", r)
        self.assertIn("HTTPError", r["error"])
        self.assertEqual(r["prefill_tps"], 0.0)

    @patch("tokprobe.bench_prefill.post_json")
    def test_all_failed_samples_aggregate_safely(self, mock_post):
        """全部样本失败时：summary 统计不抛异常，空样本输出 0.0 风格值。"""
        mock_post.side_effect = \
            requests.exceptions.ConnectionError("boom")
        runs = [self._call() for _ in range(2)]
        self.assertTrue(all(r.get("error") for r in runs))

        # 模拟 main 的 summary 统计逻辑
        ok = [r for r in runs if not r.get("error")]
        pt_list = [r["prompt_tokens"] for r in ok]
        tps = [r["prefill_tps"] for r in ok]
        mean_tps = mean(tps)
        med_tps = median(tps)
        mean_pt = mean(pt_list)

        self.assertEqual(len(ok), 0)
        self.assertEqual(mean_pt, 0.0)
        self.assertEqual(mean_tps, 0.0)
        self.assertEqual(med_tps, 0.0)

        line = prefill_quiet_line(131072, len(ok), len(runs),
                                  mean_pt, mean_tps, med_tps)
        self.assertIn("ok=0/2", line)


class TestParseTargetTokens(unittest.TestCase):
    def test_multi_tier(self):
        self.assertEqual(parse_target_tokens("65536,131072"),
                         [65536, 131072])

    def test_single_tier(self):
        self.assertEqual(parse_target_tokens("1024"), [1024])

    def test_whitespace_ignored(self):
        self.assertEqual(parse_target_tokens(" 1024 , 2048 "),
                         [1024, 2048])

    def test_empty_raises(self):
        for s in ("", ",,", " , , "):
            with self.assertRaises(ValueError):
                parse_target_tokens(s)

    def test_invalid_token_raises(self):
        for s in ("abc", "1024,xyz"):
            with self.assertRaises(ValueError):
                parse_target_tokens(s)


if __name__ == "__main__":
    unittest.main()
