"""http 模块（post_json / health_ok）的单元测试。

全部基于 mock（patch urllib.request.urlopen），
不发起任何真实网络请求。
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import patch

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tokprobe.http import health_ok, post_json


class TestPostJson(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_post_json_returns_parsed_dict(self, mock_urlopen):
        body = json.dumps({"usage": {"completion_tokens": 42}})
        mock_resp = io.StringIO(body)
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        result = post_json("http://127.0.0.1:8082/v1/chat/completions",
                           {"model": "m", "max_tokens": 1}, 30)

        self.assertEqual(result, {"usage": {"completion_tokens": 42}})

    @patch("urllib.request.urlopen")
    def test_post_json_builds_request_correctly(self, mock_urlopen):
        mock_resp = io.StringIO("{}")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        url = "http://127.0.0.1:8082/v1/chat/completions"
        payload = {"model": "m", "messages": [{"role": "user",
                                               "content": "hi"}]}
        post_json(url, payload, 60)

        (req,), kwargs = mock_urlopen.call_args
        self.assertEqual(req.full_url, url)
        self.assertEqual(req.method, "POST")
        # urllib 会把头名规范化（Content-Type -> Content-type），语义等价
        self.assertEqual(req.get_header("Content-type"), "application/json")
        self.assertEqual(req.data, json.dumps(payload).encode())
        self.assertEqual(kwargs, {"timeout": 60})

    @patch("urllib.request.urlopen")
    def test_post_json_non_dict_response_raises(self, mock_urlopen):
        # 响应体不是 JSON 对象（如数组/标量）时抛 ValueError，
        # 避免调用方 .get() 时 AttributeError
        mock_resp = io.StringIO("[1, 2, 3]")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with self.assertRaises(ValueError) as ctx:
            post_json("http://127.0.0.1:8082/v1/chat/completions", {}, 30)
        self.assertIn("unexpected response type", str(ctx.exception))


class TestHealthOk(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_health_ok_true_on_status_200(self, mock_urlopen):
        mock_resp = mock_urlopen.return_value.__enter__.return_value
        mock_resp.status = 200
        self.assertTrue(health_ok("http://127.0.0.1:8082"))

    @patch("urllib.request.urlopen")
    def test_health_ok_false_on_non_200(self, mock_urlopen):
        mock_resp = mock_urlopen.return_value.__enter__.return_value
        mock_resp.status = 503
        self.assertFalse(health_ok("http://127.0.0.1:8082"))

    @patch("urllib.request.urlopen")
    def test_health_ok_false_on_exception(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("connection refused")
        self.assertFalse(health_ok("http://127.0.0.1:8082"))


if __name__ == "__main__":
    unittest.main()
