"""bench_serve（run_single 网络异常与参数校验）的单元测试。

全部基于 mock（patch tokprobe.bench_serve.chat_request），
不发起任何真实网络请求。
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tokprobe.bench_serve import main, run_single, validate_cli_args
from tokprobe.report import mean, percentile, serve_quiet_line


def _run_single(mock_chat, stream=False):
    # chat_request 被 patch 后，client 不被实际使用，传 MagicMock 占位
    return run_single(MagicMock(), "test-model", "hi", 16, 30, stream=stream)


class TestRunSingleNetworkErrors(unittest.TestCase):
    @patch("tokprobe.bench_serve.chat_request")
    def test_exception_returns_error_dict(self, mock_chat):
        mock_chat.side_effect = RuntimeError("boom")
        r = _run_single(mock_chat)
        self.assertIn("error", r)
        self.assertIn("RuntimeError: boom", r["error"])
        self.assertEqual(r["prompt_tokens"], 0)
        self.assertEqual(r["completion_tokens"], 0)
        self.assertEqual(r["tok_per_s"], 0.0)
        self.assertEqual(r["ttft"], 0.0)
        self.assertEqual(r["decode_tps"], 0.0)
        self.assertEqual(r["chunk_count"], 0)
        self.assertGreaterEqual(r["elapsed"], 0.0)

    @patch("tokprobe.bench_serve.time.monotonic", side_effect=[0.0, 1.0])
    @patch("tokprobe.bench_serve.chat_request")
    def test_success_returns_full_dict_without_error(self, mock_chat,
                                                     mock_monotonic):
        # mock time.monotonic 为确定值，避免真实计时抖动导致 elapsed==0
        mock_chat.return_value = {
            "usage": {"prompt_tokens": 8, "completion_tokens": 16},
            "ttft": None,
            "chunk_count": 0,
        }
        r = _run_single(mock_chat)
        self.assertIsNone(r["error"])
        self.assertEqual(r["prompt_tokens"], 8)
        self.assertEqual(r["completion_tokens"], 16)
        self.assertEqual(r["elapsed"], 1.0)
        self.assertEqual(r["tok_per_s"], 16.0)
        # 非流式未测得 TTFT：ttft/decode_tps 保留 None（不可计算），
        # 不再用 0.0 冒充测得值
        self.assertIsNone(r["ttft"])
        self.assertIsNone(r["decode_tps"])
        self.assertEqual(r["chunk_count"], 0)

    @patch("tokprobe.bench_serve.time.monotonic", side_effect=[0.0, 1.0])
    @patch("tokprobe.bench_serve.chat_request")
    def test_stream_success_returns_ttft_and_decode_tps(self, mock_chat,
                                                        mock_monotonic):
        mock_chat.return_value = {
            "usage": {"prompt_tokens": 8, "completion_tokens": 16},
            "ttft": 0.2,
            "chunk_count": 18,
        }
        r = _run_single(mock_chat, stream=True)
        self.assertIsNone(r["error"])
        self.assertEqual(r["ttft"], 0.2)
        self.assertEqual(r["chunk_count"], 18)
        self.assertEqual(r["elapsed"], 1.0)
        self.assertEqual(r["tok_per_s"], 16.0)
        # decode: (16-1)/(1.0-0.2) = 18.75
        self.assertAlmostEqual(r["decode_tps"], 18.75, places=9)

    @patch("tokprobe.bench_serve.time.monotonic", side_effect=[0.0, 1.0])
    @patch("tokprobe.bench_serve.chat_request")
    def test_stream_missing_ttft_returns_none_decode_tps(self, mock_chat,
                                                         mock_monotonic):
        # 流式但未测得 TTFT（如模型只输出 reasoning_content 且被 max_tokens
        # 截断，content 从未非空）：decode_tps 应为 None（不可计算），
        # 不得用 0.0 冒充测得值
        mock_chat.return_value = {
            "usage": {"prompt_tokens": 8, "completion_tokens": 16},
            "ttft": None,
            "chunk_count": 18,
        }
        r = _run_single(mock_chat, stream=True)
        self.assertIsNone(r["error"])
        self.assertEqual(r["tok_per_s"], 16.0)
        self.assertIsNone(r["ttft"])
        self.assertIsNone(r["decode_tps"])


class TestAggregationWithAllFailures(unittest.TestCase):
    @patch("tokprobe.bench_serve.chat_request")
    def test_all_failed_run_single_results_aggregate_safely(self, mock_chat):
        """全部请求失败时：聚合统计不抛异常，空样本输出 0.0 风格值。"""
        mock_chat.side_effect = RuntimeError("boom")
        results = [_run_single(mock_chat) for _ in range(3)]
        self.assertTrue(all(r.get("error") for r in results))

        # 失败样本 dict 含新键，聚合取值不会 KeyError
        for r in results:
            self.assertEqual(r["ttft"], 0.0)
            self.assertEqual(r["decode_tps"], 0.0)
            self.assertEqual(r["chunk_count"], 0)

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
        self.assertIn("RuntimeError", err[0]["error"])
        self.assertEqual(total_ct, 0)
        self.assertEqual(total_pt, 0)
        self.assertEqual(avg_tps, 0.0)
        self.assertEqual(mean_per_req, 0.0)
        self.assertEqual(p50, 0.0)

        line = serve_quiet_line(wall, len(ok), len(results), total_ct,
                                total_pt, avg_tps, mean_per_req, p50)
        self.assertIn("requests=0/3", line)


class TestDecodeAggregation(unittest.TestCase):
    """流式聚合：未测得（None）样本不参与 ttft / decode 均值。"""

    def _result(self, ttft, decode_tps, ct=16, elapsed=1.0, chunk_count=18):
        return {
            "elapsed": elapsed,
            "prompt_tokens": 8,
            "completion_tokens": ct,
            "tok_per_s": ct / elapsed if elapsed > 0 else 0.0,
            "ttft": ttft,
            "decode_tps": decode_tps,
            "chunk_count": chunk_count,
            "error": None,
        }

    @patch("tokprobe.bench_serve.run_single")
    @patch("tokprobe.bench_serve.make_client")
    @patch("tokprobe.bench_serve.health_ok", return_value=True)
    def test_stream_decode_mean_only_uses_measurable_samples(
            self, mock_health, mock_client, mock_run):
        # 混合样本：一个未测得（None），一个有效（ttft=0.2 / decode=18.75）
        mock_run.side_effect = [
            self._result(None, None),
            self._result(0.2, 18.75),
        ]
        buf = io.StringIO()
        with patch.object(sys, "argv", ["bench_serve", "--model", "test-model",
                                        "--stream", "--quiet",
                                        "--concurrency", "2"]):
            with redirect_stdout(buf):
                main()
        out = buf.getvalue()
        # 均值只统计有效样本（0.2 / 18.75），未测得样本不以 0.0 拉低均值
        self.assertIn("ttft_mean=0.200", out)
        self.assertIn("decode_tps_mean=18.8", out)
        self.assertNotIn("None", out)


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


class TestNoTemperatureCli(unittest.TestCase):
    """--no-temperature：main 将该开关折算为 temperature=None 传给 run_single。"""

    def _ok_result(self):
        return {
            "elapsed": 1.0,
            "prompt_tokens": 8,
            "completion_tokens": 16,
            "tok_per_s": 16.0,
            "ttft": 0.0,
            "decode_tps": 0.0,
            "chunk_count": 0,
            "error": None,
        }

    @patch("tokprobe.bench_serve.run_single")
    @patch("tokprobe.bench_serve.make_client")
    @patch("tokprobe.bench_serve.health_ok", return_value=True)
    def test_no_temperature_flag_passes_none(self, mock_health,
                                              mock_client, mock_run):
        mock_run.return_value = self._ok_result()
        with patch.object(sys, "argv", ["bench_serve", "--model", "test-model",
                                        "--no-temperature", "--quiet"]):
            main()
        mock_run.assert_called_once()
        # 第 7 个位置参数（temperature）为 None -> 请求不发送 temperature 字段
        self.assertIsNone(mock_run.call_args.args[6])

    @patch("tokprobe.bench_serve.run_single")
    @patch("tokprobe.bench_serve.make_client")
    @patch("tokprobe.bench_serve.health_ok", return_value=True)
    def test_without_flag_keeps_default_zero(self, mock_health,
                                             mock_client, mock_run):
        # 未指定 --no-temperature 时行为与现状一致：temperature=0.0
        mock_run.return_value = self._ok_result()
        with patch.object(sys, "argv", ["bench_serve", "--model", "test-model",
                                        "--quiet"]):
            main()
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[6], 0.0)


if __name__ == "__main__":
    unittest.main()
