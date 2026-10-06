"""http 模块（post_json / health_ok）的单元测试。

全部基于 mock（patch requests.post / requests.get），
不发起任何真实网络请求。
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# src layout：不安装包即可让 `python -m unittest discover -s tests` 找到 tokprobe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import requests

from tokprobe.http import chat_request, health_ok, make_client, post_json


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


class TestMakeClient(unittest.TestCase):
    def test_make_client_joins_v1_and_disables_retries(self):
        # OpenAI 构造不联网，可直接实例化
        client = make_client("http://127.0.0.1:8082", api_key="sk-abc")
        # 对字符串形式断言（base_url 可能为 httpx 类型），不依赖 SDK 是否补尾斜杠
        self.assertTrue(str(client.base_url).startswith("http://127.0.0.1:8082/v1"))
        self.assertEqual(client.max_retries, 0)
        self.assertEqual(client.api_key, "sk-abc")

    def test_make_client_default_api_key_is_empty(self):
        # 无 key 时用占位值 "EMPTY"
        client = make_client("http://127.0.0.1:8082")
        self.assertEqual(client.api_key, "EMPTY")


class TestChatRequestNonStream(unittest.TestCase):
    def test_chat_request_returns_usage_and_no_stream_kwarg(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=16,
                                  total_tokens=24))

        r = chat_request(client, model="test-model",
                         messages=[{"role": "user", "content": "hi"}],
                         max_tokens=16, timeout=30)

        self.assertEqual(r["usage"], {"prompt_tokens": 8,
                                      "completion_tokens": 16,
                                      "total_tokens": 24})
        self.assertIsNone(r["ttft"])
        self.assertEqual(r["chunk_count"], 0)

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["max_tokens"], 16)
        self.assertEqual(kwargs["timeout"], 30)
        self.assertNotIn("stream", kwargs)


class _FakeStream(list):
    """承载 chunk 列表的假流对象，记录 close() 是否被调用。"""

    def __init__(self, chunks):
        super().__init__(chunks)
        self.closed = False

    def __iter__(self):
        # 遍历 list 内容；不能写 `for chunk in self`，否则会递归调用自身
        for chunk in list.__iter__(self):
            yield chunk

    def close(self):
        self.closed = True


class TestChatRequestStream(unittest.TestCase):
    @patch("tokprobe.http.time.monotonic", side_effect=[0.0, 0.2])
    def test_stream_success_records_usage_ttft_and_closes(self, mock_monotonic):
        # 第 1 次 monotonic 在 create 前（t0=0.0），第 2 次在首内容 chunk（ttft=0.2）
        usage = SimpleNamespace(prompt_tokens=8, completion_tokens=16,
                                total_tokens=24)
        chunks = _FakeStream([
            # role-only chunk：content 为空串，跳过 TTFT
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""),
                                         finish_reason=None)],
                usage=None),
            # 首个内容 chunk
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="你"),
                                         finish_reason=None)],
                usage=None),
            # 末 usage chunk：choices 为空数组
            SimpleNamespace(choices=[], usage=usage),
        ])
        client = MagicMock()
        client.chat.completions.create.return_value = chunks

        r = chat_request(client, model="test-model",
                         messages=[{"role": "user", "content": "hi"}],
                         max_tokens=16, timeout=30, stream=True)

        self.assertEqual(r["usage"], {"prompt_tokens": 8,
                                      "completion_tokens": 16,
                                      "total_tokens": 24})
        self.assertEqual(r["chunk_count"], 3)
        self.assertGreater(r["ttft"], 0)
        self.assertTrue(chunks.closed)

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_stream_without_usage_chunk_returns_zero_usage(self):
        chunks = _FakeStream([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="你"),
                                         finish_reason=None)],
                usage=None),
        ])
        client = MagicMock()
        client.chat.completions.create.return_value = chunks

        r = chat_request(client, model="test-model",
                         messages=[{"role": "user", "content": "hi"}],
                         max_tokens=16, timeout=30, stream=True)

        self.assertEqual(r["usage"], {"prompt_tokens": 0,
                                      "completion_tokens": 0,
                                      "total_tokens": 0})
        self.assertEqual(r["chunk_count"], 1)

    def test_stream_interrupted_propagates_and_closes(self):
        class _BrokenStream(_FakeStream):
            def __iter__(self):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(
                        content="你"), finish_reason=None)],
                    usage=None)
                raise RuntimeError("stream interrupted")

        stream = _BrokenStream([])
        client = MagicMock()
        client.chat.completions.create.return_value = stream

        with self.assertRaises(RuntimeError) as ctx:
            chat_request(client, model="test-model",
                         messages=[{"role": "user", "content": "hi"}],
                         max_tokens=16, timeout=30, stream=True)
        self.assertIn("stream interrupted", str(ctx.exception))
        # finally 生效：即使迭代中途异常，close 也已调用
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
