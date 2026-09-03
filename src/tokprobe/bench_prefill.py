#!/usr/bin/env python3
"""
vLLM（OpenAI 兼容接口）预填充（prefill）测速工具
==================================================
参考 tokprobe.bench_serve 的风格（仅用标准库 urllib、零额外依赖），
针对"预填充阶段"单独测速：

  prefill_tok/s = usage.prompt_tokens / 单请求墙钟

关键设计：
  1) max_tokens=1 ：只让服务做预填充并吐 1 个 token，
     避免把大量 decode 时间混入 prefill 耗时。
  2) PRNG（可复现）生成"规律一致、但每 seed 前缀不同"的数列文本：
     - 同一 --seed 完全可复现，便于横向对比；
     - 不同 --seed 前缀/内容不同 -> 规避 vLLM 前缀缓存（radix cache）命中，
       保证测到的是真实 prefill 计算量。
  3) 用 /tokenize 自动校准字符/token 比（默认开启，可用
     --char-per-token 覆盖）：脚本生成的 ASCII 数字数列文本 BPE 后
     接近 1 字符 ≈ 1 token，而非默认参数假设的 4.0；直接用服务端
     分词器校准，避免因估算误差导致 prompt 超 max_model_len 而返回 400。
     真实计数仍取响应里的 usage.prompt_tokens，测速本身依然精确。

用法示例:
  # 默认：128K 目标、3 个 seed 取均值
  python -m tokprobe.bench_prefill --model my-model

  # 指定端口/模型（与 .env 对齐）
  python -m tokprobe.bench_prefill --port 8082 --model my-model

  # 同时测 64K 与 128K 两个档位
  python -m tokprobe.bench_prefill --model my-model --target-tokens 65536,131072

  # 5 个 seed（更稳），只输出数字
  python -m tokprobe.bench_prefill --model my-model --seeds 5 --quiet

  # 输出 JSON 到文件
  python -m tokprobe.bench_prefill --model my-model --json-output ./prefill_results.json
"""

import argparse
import concurrent.futures
import json
import random
import secrets
import sys
import time

from .http import health_ok, post_json
from .report import mean, median, prefill_quiet_line


def parse_target_tokens(s: str) -> list:
    """解析逗号分隔的目标 token 档位（如 "65536,131072" -> [65536, 131072]）。

    - 空白段忽略；
    - 非法整数值抛 ``ValueError``（含出错字段）；
    - 全部为空（如 ",," 或空串）抛 ``ValueError``。
    """
    targets = []
    for tok in s.split(","):
        if not tok.strip():
            continue
        try:
            targets.append(int(tok))
        except ValueError:
            raise ValueError(f"非法目标 token 值: {tok!r}") from None
    if not targets:
        raise ValueError("--target-tokens 为空，请至少指定一个目标 token 档位")
    return targets


def calibrate_char_per_token(base: str, model: str,
                             sample_chars: int = 4096) -> float:
    """
    用 /tokenize 精确校准当前模型的字符/token 比。

    脚本生成的 ASCII 数字数列文本 BPE 后接近 1 字符 ≈ 1 token
    （实测 char/token≈1.043），而非默认参数假设的 4.0。
    直接用服务端分词器校准，避免因估算误差导致 prompt
    超 max_model_len 而返回 400。
    """
    p = build_prompt(0, sample_chars)
    count = post_json(f"{base}/tokenize",
                      {"model": model, "prompt": p}, 60)["count"]
    ratio = len(p) / count if count > 0 else 1.0
    return ratio


# ---------- 生成防缓存的规律数列文本 ----------

def build_prompt(seed: int, target_chars: int,
                 cols: int = 6, width: int = 8) -> str:
    """
    生成"规律一致、但每 seed 前缀不同"的数列表格文本。

    用 random.Random(seed) 代替全局 random，保证：
      - 同一 seed 输出完全可复现；
      - 不同 seed，哪怕是同一种"行格式/数列规律"，
        第一段 token 流也完全不同 -> 规避前缀缓存命中。

    文本形如（ASCII，便于 BPE 稳定、贴近目标 token）：
      row 000123: 84210973, 46213058, ...
    target_chars 只是粗略目标，之后按响应的 usage.prompt_tokens 取真实值。
    """
    rng = random.Random(seed)
    lines = []
    total = 0
    n = 0
    # 每行大体固定宽度，便于估算；enough = 目标字符 + 一行余量
    row_len = len(f"row 000000: ") + cols * (width + 2)  # 逗号+空格
    n_rows = target_chars // max(1, row_len) + 1
    while n < n_rows:
        nums = ", ".join(f"{rng.randrange(0, 10**width):0{width}d}"
                         for _ in range(cols))
        line = f"row {n:06d}: {nums}"
        lines.append(line)
        n += 1
    return "\n".join(lines)[:target_chars]


# ---------- 单请求 prefill 测量 ----------

def measure_prefill(base: str, model: str, prompt: str,
                    max_tokens: int, timeout: int) -> dict:
    t0 = time.monotonic()
    try:
        resp = post_json(
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,   # 1 -> 只预填充，几乎不 decode
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
            "prefill_tps": 0.0,
            "ms_per_token": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }
    elapsed = time.monotonic() - t0
    usage = resp.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    # 注意：chat template 会把 messages 包装成完整 prompt，
    # 因此 usage.prompt_tokens 通常比裸文本 token 略多一点点。
    return {
        "elapsed": elapsed,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "prefill_tps": pt / elapsed if elapsed > 0 else 0.0,
        "ms_per_token": elapsed * 1000 / pt if pt > 0 else 0.0,
        "error": resp.get("error"),
    }


def run_sample(base: str, model: str, seed: int, target_tokens: int,
               char_per_token: float, max_tokens: int,
               timeout: int) -> dict:
    """根据 seed 生成文本并测量一次 prefill。"""
    target_chars = max(1, int(target_tokens * char_per_token))
    prompt = build_prompt(seed, target_chars)
    r = measure_prefill(base, model, prompt, max_tokens, timeout)
    r.update({"seed": seed, "target_tokens": target_tokens,
              "gen_chars": len(prompt)})
    return r


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM / OpenAI 兼容接口预填充测速")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--model", required=True, help="模型名（必填）")
    ap.add_argument("--target-tokens", default="131072",
                    help="目标 token 档位，逗号分隔多值，如 65536,131072")
    ap.add_argument("--char-per-token", type=float, default=0,
                    help="ASCII 数列文本字符/token 估算比；默认 0 表示用 /tokenize 自动校准")
    ap.add_argument("--seeds", type=int, default=3,
                    help="每个档位跑的随机 seed 数（取均值/中位数，抗抖动）")
    ap.add_argument("--random-seeds", action="store_true",
                    help="用系统熵生成真正随机的 seed（重复运行文本也不同，彻底防缓存）")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="请求 max_tokens（默认 1：只测 prefill）")
    ap.add_argument("--warmup", action="store_true",
                    help="先发一个短请求暖机（避免首次 kernel 编译混入计时）")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json-output", default=None, help="结果写入 JSON 文件")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        targets = parse_target_tokens(args.target_tokens)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)

    base = f"http://{args.host}:{args.port}"
    if not health_ok(base):
        print(f"错误: {base}/health 不可用", file=sys.stderr)
        sys.exit(1)

    # 用 /tokenize 自动校准字符/token 比，避免默认估算超 max_model_len
    if not args.char_per_token:
        try:
            args.char_per_token = calibrate_char_per_token(base, args.model)
        except Exception as e:
            print(f"警告: /tokenize 校准失败(用回 1.0): {e}", file=sys.stderr)
            args.char_per_token = 1.0

    if args.warmup:
        try:
            post_json(
                f"{base}/v1/chat/completions",
                {"model": args.model,
                 "messages": [{"role": "user", "content": "Reply with: ok"}],
                 "max_tokens": 8, "temperature": 0},
                timeout=120,
            )
        except Exception as e:
            print(f"警告: 暖机请求失败: {e}", file=sys.stderr)

    samples = []          # 所有 (target, seed) 的原始结果
    summary = []          # 每个档位的统计

    for target_tokens in targets:
        runs = []
        for se in range(args.seeds):
            seed = secrets.randbits(63) if args.random_seeds else se
            r = run_sample(base, args.model, seed, target_tokens,
                           args.char_per_token, args.max_tokens,
                           args.timeout)
            samples.append(r)
            runs.append(r)

        ok = [r for r in runs if not r.get("error")]
        pt_list = [r["prompt_tokens"] for r in ok]
        tps = [r["prefill_tps"] for r in ok]
        mean_tps = mean(tps)
        med_tps = median(tps)
        mean_pt = mean(pt_list)
        s = {
            "target_tokens": target_tokens,
            "runs": len(runs), "ok": len(ok),
            "mean_prompt_tokens": mean_pt,
            "mean_prefill_tps": mean_tps,
            "median_prefill_tps": med_tps,
        }
        summary.append(s)

    if args.json_output:
        try:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "samples": samples},
                          f, indent=2, sort_keys=True)
        except OSError as e:
            print(f"错误: 写入 JSON 文件失败 {args.json_output}: {e}",
                  file=sys.stderr)
            sys.exit(1)

    if args.quiet:
        for s in summary:
            print(prefill_quiet_line(s["target_tokens"], s["ok"], s["runs"],
                                     s["mean_prompt_tokens"],
                                     s["mean_prefill_tps"],
                                     s["median_prefill_tps"]))
        return

    print("=" * 66)
    print(f"模型: {args.model}   目标档位: {targets}")
    print(f"每档 seed 数: {args.seeds}   max_tokens(prefill): {args.max_tokens}")
    print(f"char/token 估算: {args.char_per_token}   "
          f"墙钟计时: 单请求 elapsed")
    print("-" * 66)
    for s in summary:
        print(f"\n目标 {s['target_tokens']} tok | 成功 {s['ok']}/{s['runs']} | "
              f"实测 prompt_token 均值 {s['mean_prompt_tokens']:.0f}")
        print(f"  prefill 均值: {s['mean_prefill_tps']:.1f} tok/s "
              f"(每 token {1000/s['mean_prefill_tps'] if s['mean_prefill_tps'] else 0:.2f} ms)")
        print(f"  prefill 中位数: {s['median_prefill_tps']:.1f} tok/s")
    print("-" * 66)
    # 逐样本明细
    for r in samples:
        err = f"  error={r['error']}" if r.get("error") else ""
        print(f"  target={r['target_tokens']:<7} seed={r['seed']} "
              f"pt={r['prompt_tokens']:>6}  {r['elapsed']:6.2f}s  "
              f"{r['prefill_tps']:7.1f} tok/s{err}")
    print("=" * 66)


if __name__ == "__main__":
    main()
