"""http 模块（post_json / health_ok）的单元测试。

全部基于 mock（patch requests.post / requests.get），
不发起任何真实网络请求。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import requests

from tokprobe.http import health_ok, post_json


def _mock_resp(status_code=200, body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


class TestPostJson(unittest.TestCase):
    @patch("tokprobe.http.requests.post")
    def test_post_json_returns_parsed_dict(self, mock_post):
        mock_post.return_value = _mock_resp(
            body={"usage": {"completion_tokens": 42}})

        result = post_json("http://127.0.0.1:8082/v1/chat/completions",
                           {"model": "m", "max_tokens": 1}, 30)

        self.assertEqual(result, {"usage": {"completion_tokens": 42}})
        mock_post.return_value.raise_for_status.assert_called_once()

    @patch("tokprobe.http.requests.post")
    def test_post_json_builds_request_correctly(self, mock_post):
        mock_post.return_value = _mock_resp(body={})

        url = "http://127.0.0.1:8082/v1/chat/completions"
        payload = {"model": "m", "messages": [{"role": "user",
                                               "content": "hi"}]}
        post_json(url, payload, 60)

        call = mock_post.call_args
        self.assertEqual(call.args[0], url)
        self.assertEqual(call.kwargs["json"], payload)
        self.assertEqual(call.kwargs["timeout"], 60)
        # 未传 api_key 时不携带 Authorization 头，headers 为空 dict
        self.assertEqual(call.kwargs["headers"], {})
        mock_post.return_value.raise_for_status.assert_called_once()

    @patch("tokprobe.http.requests.post")
    def test_post_json_with_api_key(self, mock_post):
        mock_post.return_value = _mock_resp(body={})

        post_json("http://127.0.0.1:8082/v1/chat/completions", {}, 30,
                  api_key="sk-abc")

        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-abc")

    @patch("tokprobe.http.requests.post")
    def test_post_json_raises_on_non_2xx(self, mock_post):
        mock_post.return_value = _mock_resp(status_code=500)
        mock_post.return_value.raise_for_status.side_effect = \
            requests.exceptions.HTTPError("500 Server Error")

        with self.assertRaises(requests.exceptions.HTTPError):
            post_json("http://127.0.0.1:8082/v1/chat/completions", {}, 30)

    @patch("tokprobe.http.requests.post")
    def test_post_json_non_dict_response_raises(self, mock_post):
        # 响应体不是 JSON 对象（如数组/标量）时抛 ValueError，
        # 避免调用方 .get() 时 AttributeError
        mock_post.return_value = _mock_resp(body=[1, 2, 3])

        with self.assertRaises(ValueError) as ctx:
            post_json("http://127.0.0.1:8082/v1/chat/completions", {}, 30)
        self.assertIn("unexpected response type", str(ctx.exception))


class TestHealthOk(unittest.TestCase):
    @patch("tokprobe.http.requests.get")
    def test_health_ok_true_on_status_200(self, mock_get):
        mock_get.return_value = _mock_resp(status_code=200)
        self.assertTrue(health_ok("http://127.0.0.1:8082"))

    @patch("tokprobe.http.requests.get")
    def test_health_ok_false_on_non_200(self, mock_get):
        mock_get.return_value = _mock_resp(status_code=503)
        self.assertFalse(health_ok("http://127.0.0.1:8082"))

    @patch("tokprobe.http.requests.get")
    def test_health_ok_false_on_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "connection refused")
        self.assertFalse(health_ok("http://127.0.0.1:8082"))

    @patch("tokprobe.http.requests.get")
    def test_health_ok_with_api_key(self, mock_get):
        mock_get.return_value = _mock_resp(status_code=200)
        self.assertTrue(
            health_ok("http://127.0.0.1:8082", api_key="sk-xyz"))
        headers = mock_get.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer sk-xyz")


if __name__ == "__main__":
    unittest.main()
