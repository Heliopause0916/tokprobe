"""公共 HTTP 网络辅助（仅标准库 urllib，零第三方依赖）。

提供两个函数：

- ``post_json``：向 OpenAI 兼容接口发送 JSON 请求并解析响应；
- ``health_ok``：探测服务的 /health 端点是否就绪。

统一网络请求与健康检查，供两个 benchmark 命令共用（含超时、
错误处理与返回约定）。
"""

import json
import urllib.request


def post_json(url: str, payload: dict, timeout: int) -> dict:
    """POST 一个 JSON payload，返回解析后的响应 dict。

    - 非 2xx 状态码抛出 :class:`urllib.error.HTTPError`，连接失败等抛出
      :class:`urllib.error.URLError`（由调用方决定如何兜底，见 run_single /
      measure_prefill）；
    - 响应体解析后若不是 dict，抛出 ``ValueError("unexpected response
      type: ...")``，避免调用方 ``.get()`` 时 AttributeError。
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.load(resp)
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected response type: {type(obj).__name__}")
    return obj


def health_ok(base: str) -> bool:
    """检查 ``{base}/health`` 是否返回 200。

    - 超时固定 5 秒（与原实现一致）；
    - 任何异常（连接失败 / 超时 / 非 200）统一返回 ``False``。
    """
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False
