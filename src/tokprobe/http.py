"""公共 HTTP / API 请求辅助（requests + openai SDK）。

提供四个函数：

- ``post_json``：以 requests 向 OpenAI 兼容接口发送 JSON 请求并解析响应；
- ``make_client``：构造 openai 同步客户端（base_url 拼接 /v1）；
- ``chat_request``：以 openai SDK 发起 chat completions 请求（支持流式）；
- ``health_ok``：探测服务的 /health 端点是否就绪。

统一网络请求与健康检查，供两个 benchmark 命令共用（含超时、
错误处理与返回约定）。
"""

import time
from typing import Optional

import requests
from openai import OpenAI


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


def make_client(base: str, api_key: Optional[str] = None) -> OpenAI:
    """构造 openai 同步客户端（供 chat_request 使用）。

    - SDK 不会自动为 base_url 补 ``/v1``（已实测），需手动拼接；
    - ``api_key`` 为空时用占位值 ``"EMPTY"``（SDK 要求非空，vLLM 默认
      不校验该值）；
    - ``max_retries=0``：禁用 SDK 默认重试，避免重试污染计时统计。
    """
    return OpenAI(
        base_url=f"{base.rstrip('/')}/v1",
        api_key=api_key or "EMPTY",
        max_retries=0,
    )


def chat_request(client, *, model, messages, max_tokens, timeout,
                 temperature: Optional[float] = 0.0, stream=False) -> dict:
    """以 openai SDK 发起一次 chat completions 请求，统一返回统计 dict。

    返回结构（流式与非流式一致）::

        {"usage": {"prompt_tokens": int, "completion_tokens": int,
                   "total_tokens": int},
         "ttft": float | None,  # 流式首个内容 token 到达时刻（秒）；非流式 None
         "chunk_count": int}    # 流式收到的 chunk 数；非流式恒为 0

    - usage 一律取响应 usage 字段（不做逐 token 文本解析），
      usage 缺失（None）时三字段计 0；
    - ``temperature`` 为 None 时请求体不携带 temperature 键，
      由服务端使用自身默认温度（如 vLLM 默认 1.0）生成；
    - 任何异常一律上抛，由调用方（run_single）兜底折算。
    """
    # temperature 为 None 时不发送该键（服务端按其默认温度生成）
    gen_kwargs = {"model": model, "messages": messages,
                  "max_tokens": max_tokens, "timeout": timeout}
    if temperature is not None:
        gen_kwargs["temperature"] = temperature

    if not stream:
        resp = client.chat.completions.create(**gen_kwargs)
        u = resp.usage
        if u is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0,
                     "total_tokens": 0}
        else:
            usage = {"prompt_tokens": u.prompt_tokens,
                     "completion_tokens": u.completion_tokens,
                     "total_tokens": u.total_tokens}
        return {"usage": usage, "ttft": None, "chunk_count": 0}

    # 流式：t0 必须在 create 之前取，TTFT 才覆盖排队与首包延迟
    t0 = time.monotonic()
    s = client.chat.completions.create(
        **gen_kwargs,
        stream=True,
        stream_options={"include_usage": True},
    )
    usage = None
    ttft = None
    chunk_count = 0
    try:
        for chunk in s:
            chunk_count += 1
            # 末 usage chunk 的 choices 为空数组，必须先判 usage 再碰 choices
            if chunk.usage is not None:
                usage = chunk.usage
            if ttft is None and chunk.choices:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:  # role-only / 空串 chunk 跳过，非空内容才算
                    ttft = time.monotonic() - t0
    finally:
        s.close()

    if usage is None:
        usage = {"prompt_tokens": 0, "completion_tokens": 0,
                 "total_tokens": 0}
    else:
        usage = {"prompt_tokens": usage.prompt_tokens,
                 "completion_tokens": usage.completion_tokens,
                 "total_tokens": usage.total_tokens}
    return {"usage": usage, "ttft": ttft, "chunk_count": chunk_count}


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
