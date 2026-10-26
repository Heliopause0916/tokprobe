#!/usr/bin/env python3
"""
vLLM (OpenAI 兼容接口) 测速工具
=================================
基于 OpenAI 兼容接口，支持单请求测速与并发压测。

用法示例:
  # 单请求测速（默认）
  python -m tokprobe.bench_serve --model my-model

  # 指定服务地址 / 模型 / 问题
  python -m tokprobe.bench_serve --base-url http://127.0.0.1:8082 --model my-model --prompt "介绍一下QUIC协议"

  # 并发 4、每个请求最大生成 1024 token
  python -m tokprobe.bench_serve --model my-model --concurrency 4 --max-tokens 1024

  # 并发 4、共发 8 个请求（2 轮）
  python -m tokprobe.bench_serve --model my-model --concurrency 4 --n-requests 8

  # 只返回数字（便于脚本拼接 / 自动化对比）
  python -m tokprobe.bench_serve --model my-model --quiet --concurrency 4
"""

import argparse
import concurrent.futures
import os
import sys
import time
from typing import Optional

from .http import chat_request, health_ok, make_client
from .report import mean, percentile, serve_quiet_line


def validate_cli_args(concurrency: int, n_requests: int) -> None:
    """校验并发参数；非法时抛 ValueError（由 main 转为友好报错 + exit 2）。"""
    if concurrency < 1:
        raise ValueError(f"--concurrency 必须 >= 1（当前值: {concurrency}）")
    if n_requests < 0:
        raise ValueError(f"--n-requests 必须 >= 0（当前值: {n_requests}）")


def run_single(client, model: str, prompt: str, max_tokens: int,
               timeout: int, stream: bool = False,
               temperature: Optional[float] = 0.0) -> dict:
    """发送单个请求，返回耗时与 token 统计。

    网络/HTTP 异常（非 2xx、连接失败、超时等）不抛出，统一折算为
    带 ``error`` 键的结果 dict，由调用方按 err/ok 分流统计。

    返回键统一（流式与非流式一致）：elapsed、prompt_tokens、
    completion_tokens、tok_per_s、ttft、decode_tps、chunk_count、error。
    """
    t0 = time.monotonic()
    try:
        r = chat_request(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            stream=stream,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "elapsed": elapsed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tok_per_s": 0.0,
            "ttft": 0.0,
            "decode_tps": 0.0,
            "chunk_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }
    elapsed = time.monotonic() - t0
    usage = r["usage"]
    ct = usage["completion_tokens"]
    pt = usage["prompt_tokens"]
    # ttft 为 None 表示未测得（非流式 / 流式无输出 token），保留该区分
    ttft = r["ttft"]
    # decode 段口径：第 1 个 token 于 ttft 时刻到达，其余 ct-1 个 token
    # 落在 (ttft, 流结束) 区间内，故分母为 elapsed - ttft
    # 未测得 TTFT 或样本不可计算时返回 None（不参与均值），而非 0.0 冒充
    decode_tps = ((ct - 1) / (elapsed - ttft)
                  if (stream and ttft is not None and ct > 1
                      and elapsed > ttft) else None)
    return {
        "elapsed": elapsed,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "tok_per_s": ct / elapsed if elapsed > 0 else 0.0,
        "ttft": ttft,
        "decode_tps": decode_tps,
        "chunk_count": r["chunk_count"],
        "error": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM / OpenAI 兼容接口测速")
    ap.add_argument("--base-url", default="http://127.0.0.1:8082",
                    help="服务基础地址，含协议与端口，如 http://192.168.6.3:8098")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                    type=str, help="可选：Bearer 认证密钥")
    ap.add_argument("--model", required=True, help="模型名（必填）")
    ap.add_argument("--prompt", default="介绍一下QUIC协议")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同时并发请求数")
    ap.add_argument("--n-requests", type=int, default=0,
                    help="总请求数（默认 = concurrency，即 1 轮）")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--stream", action="store_true",
                    help="流式模式（SSE）：附加输出 TTFT 与 decode 段吞吐（默认非流式）")
    ap.add_argument("--no-temperature", action="store_true",
                    help="不发送 temperature 字段，由服务端使用默认温度生成（输出不可复现，适用于模拟真实随机生成负载）")
    ap.add_argument("--quiet", action="store_true",
                    help="只输出核心数字")
    args = ap.parse_args()

    try:
        validate_cli_args(args.concurrency, args.n_requests)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)

    base = args.base_url.rstrip("/")
    if not health_ok(base, args.api_key):
        print(f"错误: {base}/health 不可用", file=sys.stderr)
        sys.exit(1)
    client = make_client(base, args.api_key)

    n_total = args.n_requests if args.n_requests > 0 else args.concurrency
    results = []

    t0_all = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(run_single, client, args.model, args.prompt,
                      args.max_tokens, args.timeout, args.stream,
                      None if args.no_temperature else 0.0)
            for _ in range(n_total)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    wall = time.monotonic() - t0_all

    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]

    total_ct = sum(r["completion_tokens"] for r in ok)
    total_pt = sum(r["prompt_tokens"] for r in ok)
    avg_tps = total_ct / wall if wall > 0 else 0.0
    per_req_tps = [r["tok_per_s"] for r in ok]
    mean_per_req = mean(per_req_tps)
    p50 = percentile(per_req_tps, 50)
    # 只对实测得（非 None）的样本求均值；mean(空列表)=0.0，语义为"未测得"
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    # decode 均值与 ttfts 口径一致：过滤未测得（None）的样本
    dtps = [r["decode_tps"] for r in ok if r["decode_tps"] is not None]
    chunks = [r["chunk_count"] for r in ok]

    if args.quiet:
        stream_stats = None
        if args.stream:
            stream_stats = {"ttft_mean": mean(ttfts),
                            "decode_tps_mean": mean(dtps),
                            "chunks_mean": mean(chunks)}
        print(serve_quiet_line(wall, len(ok), len(results), total_ct,
                               total_pt, avg_tps, mean_per_req, p50,
                               stream_stats=stream_stats))
        return

    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"并发: {args.concurrency}  总请求: {len(results)}  墙钟: {wall:.2f}s")
    print(f"prompt: {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    print("-" * 60)
    if err:
        print(f"失败 {len(err)} 个，首个错误: {err[0].get('error')}")
    if ok:
        print(f"成功 {len(ok)} 个")
        print(f"总生成 tokens: {total_ct}")
        print(f"总 prompt tokens: {total_pt}")
        print(f"总吞吐 (aggregate): {avg_tps:.1f} tok/s")
        print(f"每请求平均速度: {mean_per_req:.1f} tok/s")
        print(f"每请求 P50 速度: {p50:.1f} tok/s")
        if args.stream:
            print(f"TTFT 均值: {mean(ttfts):.3f} s（首个输出 token）")
            print(f"decode 段吞吐均值: {mean(dtps):.1f} tok/s（剔除首 token）")
            print(f"平均 chunk 数: {mean(chunks):.1f}")
        for i, r in enumerate(ok, 1):
            line = (f"  [req{i}] {r['completion_tokens']:>5} tok  "
                    f"{r['elapsed']:6.2f}s  {r['tok_per_s']:6.1f} tok/s")
            if args.stream:
                # 未测得（None）显示为 0.0 占位，保持列结构与原格式不变
                line += (f"  {r['ttft'] or 0.0:6.3f}s  "
                         f"{r['decode_tps'] or 0.0:6.1f} tok/s(decode)")
            print(line)
    print("=" * 60)


if __name__ == "__main__":
    main()
