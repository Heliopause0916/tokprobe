"""公共 HTTP 网络辅助（基于 requests）。

提供两个函数：

- ``post_json``：向 OpenAI 兼容接口发送 JSON 请求并解析响应；
- ``health_ok``：探测服务的 /health 端点是否就绪。

统一网络请求与健康检查，供两个 benchmark 命令共用（含超时、
错误处理与返回约定）。
"""

from typing import Optional

import requests


def post_json(url: str, payload: dict, timeout: int,
              api_key: Optional[str] = None) -> dict:
    """POST 一个 JSON payload，返回解析后的响应 dict。

    - 非 2xx 状态码抛出 :class:`requests.exceptions.HTTPError`，
      连接失败等抛出 :class:`requests.exceptions.RequestException`
      （由调用方决定如何兜底，见 run_single / measure_prefill）；
    - ``api_key`` 非空时附加 ``Authorization: Bearer <api_key>`` 头，
      否则不携带任何认证头；
    - 响应体解析后若不是 dict，抛出 ``ValueError("unexpected response
      type: ...")``，避免调用方 ``.get()`` 时 AttributeError。
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    obj = resp.json()
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected response type: {type(obj).__name__}")
    return obj


def health_ok(base: str, api_key: Optional[str] = None) -> bool:
    """检查 ``{base}/health`` 是否返回 200。

    - 超时固定 5 秒（与原实现一致）；
    - 任何异常（连接失败 / 超时 / 非 200 / 认证失败）统一返回 ``False``。
    """
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(f"{base}/health", headers=headers, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
