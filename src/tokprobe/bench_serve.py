#!/usr/bin/env python3
"""
vLLM (OpenAI 兼容接口) 测速工具
=================================
基于 OpenAI 兼容接口，支持单请求测速与并发压测。

用法示例:
  # 单请求测速（默认）
  python -m tokprobe.bench_serve --model my-model

  # 指定模型 / 端口 / 问题
  python -m tokprobe.bench_serve --port 8082 --model my-model --prompt "介绍一下QUIC协议"

  # 并发 4、每个请求最大生成 1024 token
  python -m tokprobe.bench_serve --model my-model --concurrency 4 --max-tokens 1024

  # 并发 4、共发 8 个请求（2 轮）
  python -m tokprobe.bench_serve --model my-model --concurrency 4 --n-requests 8

  # 只返回数字（便于脚本拼接 / 自动化对比）
  python -m tokprobe.bench_serve --model my-model --quiet --concurrency 4
"""

import argparse
import concurrent.futures
import sys
import time

from .http import health_ok, post_json
from .report import mean, percentile, serve_quiet_line


def validate_cli_args(concurrency: int, n_requests: int) -> None:
    """校验并发参数；非法时抛 ValueError（由 main 转为友好报错 + exit 2）。"""
    if concurrency < 1:
        raise ValueError(f"--concurrency 必须 >= 1（当前值: {concurrency}）")
    if n_requests < 0:
        raise ValueError(f"--n-requests 必须 >= 0（当前值: {n_requests}）")


def run_single(base: str, model: str, prompt: str, max_tokens: int,
               timeout: int) -> dict:
    """发送单个请求，返回耗时与 token 统计。

    网络/HTTP 异常（非 2xx、连接失败、超时等）不抛出，统一折算为
    带 ``error`` 键的结果 dict，由调用方按 err/ok 分流统计。
    """
    t0 = time.monotonic()
    try:
        resp = post_json(
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "elapsed": elapsed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tok_per_s": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }
    elapsed = time.monotonic() - t0
    usage = resp.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    pt = usage.get("prompt_tokens", 0)
    return {
        "elapsed": elapsed,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "tok_per_s": ct / elapsed if elapsed > 0 else 0.0,
        "error": resp.get("error"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM / OpenAI 兼容接口测速")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--model", required=True, help="模型名（必填）")
    ap.add_argument("--prompt", default="介绍一下QUIC协议")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同时并发请求数")
    ap.add_argument("--n-requests", type=int, default=0,
                    help="总请求数（默认 = concurrency，即 1 轮）")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--quiet", action="store_true",
                    help="只输出核心数字")
    args = ap.parse_args()

    try:
        validate_cli_args(args.concurrency, args.n_requests)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)

    base = f"http://{args.host}:{args.port}"
    if not health_ok(base):
        print(f"错误: {base}/health 不可用", file=sys.stderr)
        sys.exit(1)

    n_total = args.n_requests if args.n_requests > 0 else args.concurrency
    results = []

    t0_all = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(run_single, base, args.model, args.prompt,
                      args.max_tokens, args.timeout)
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

    if args.quiet:
        print(serve_quiet_line(wall, len(ok), len(results), total_ct,
                               total_pt, avg_tps, mean_per_req, p50))
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
        for i, r in enumerate(ok, 1):
            print(f"  [req{i}] {r['completion_tokens']:>5} tok  "
                  f"{r['elapsed']:6.2f}s  {r['tok_per_s']:6.1f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
