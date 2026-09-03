# tokprobe

轻量 vLLM 测速 / prefill 探针（网络层基于 requests）。

tokprobe 定位为**轻量测量工具**而非压测平台：用最少的代码、最快的上手路径，
对 OpenAI 兼容接口（vLLM / vLLM 系服务）做两类测量并给出可归因的结论——

- **serve**：通用端到端并发压测（含 decode），回答「服务整体吞吐与单请求速度」；
- **prefill**：prefill 阶段专项测速，回答「长 prompt 的预填充算力」。

区别于重负载压测平台，tokprobe 强调单机、单脚本、可复现、便于脚本拼接与
自动化对比（`--quiet` 单行输出、`--json-output` 结构化输出）。

## 安装

第三方依赖仅 requests（Python 3.9+）。

```bash
# 方式一：可编辑安装（提供 tokprobe-serve / tokprobe-prefill 两个命令）
pip install -e .

# 方式二：不安装、零副作用，用 PYTHONPATH 指向 src 直接运行
#（PowerShell 用 $env:PYTHONPATH="src"；bash 用 PYTHONPATH=src 前缀）
python -m tokprobe.bench_serve --model my-model --quiet
python -m tokprobe.bench_prefill --model my-model --quiet
```

> 说明：`python -m tokprobe.xxx` 依赖包可被导入。方式一安装后直接可用；
> 方式二需要把 `src/` 加入 `PYTHONPATH`（或先 `pip install -e .`）。

## 用法

### tokprobe-serve（通用端到端并发压测）

```bash
# 单请求测速（默认）
python -m tokprobe.bench_serve --model my-model

# 指定服务地址 / 模型 / 问题
python -m tokprobe.bench_serve --base-url http://127.0.0.1:8082 --model my-model --prompt "介绍一下QUIC协议"

# 并发 4、每个请求最大生成 1024 token
python -m tokprobe.bench_serve --model my-model --concurrency 4 --max-tokens 1024

# 并发 4、共发 8 个请求（2 轮）
python -m tokprobe.bench_serve --model my-model --concurrency 4 --n-requests 8

# 只输出数字（便于脚本拼接 / 自动化对比）
python -m tokprobe.bench_serve --model my-model --quiet --concurrency 4
```

`--quiet` 输出示例（单行，字段名稳定）：

```
wall=12.34s requests=8/8 total_ct=2048 total_pt=64 avg_tps=166.0 per_req_mean=92.5 per_req_p50=95.0
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8082` | 服务基础地址（含协议与端口），如 `http://192.168.6.3:8098` |
| `--api-key` | 无 | 可选，Bearer 认证密钥 |
| `--model` | 必填 | 模型名（必填，无默认值） |
| `--prompt` | `介绍一下QUIC协议` | 请求内容 |
| `--max-tokens` | `4096` | 每个请求最大生成 token |
| `--concurrency` | `1` | 同时并发请求数 |
| `--n-requests` | `0` | 总请求数（默认 = concurrency，即 1 轮） |
| `--timeout` | `1800` | 单请求超时（秒） |
| `--quiet` | — | 只输出核心数字（单行） |

### tokprobe-prefill（prefill 专项测速）

```bash
# 默认：128K 目标、3 个 seed 取均值
python -m tokprobe.bench_prefill --model my-model

# 同时测 64K 与 128K 两个档位
python -m tokprobe.bench_prefill --model my-model --target-tokens 65536,131072

# 5 个 seed（更稳），只输出数字
python -m tokprobe.bench_prefill --model my-model --seeds 5 --quiet

# 输出 JSON 到文件
python -m tokprobe.bench_prefill --model my-model --json-output ./prefill_results.json
```

`--quiet` 输出示例（每档位一行）：

```
target=131072 ok=3/3 emp_pt=134912 prefill_mean=45210.5 prefill_med=45000.2 tok/s
```

`--json-output` 写出的文件结构：

```json
{
  "summary": [
    {
      "target_tokens": 131072,
      "runs": 3,
      "ok": 3,
      "mean_prompt_tokens": 134912.0,
      "mean_prefill_tps": 45210.5,
      "median_prefill_tps": 45000.2
    }
  ],
  "samples": [
    {
      "seed": 0,
      "target_tokens": 131072,
      "gen_chars": 136640,
      "prompt_tokens": 134912,
      "completion_tokens": 1,
      "elapsed": 2.9,
      "prefill_tps": 46521.4,
      "ms_per_token": 0.0215,
      "error": null
    }
  ]
}
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 模型名（必填，无默认值） |
| `--target-tokens` | `131072` | 目标 token 档位，逗号分隔多值 |
| `--char-per-token` | `0` | 字符/token 估算比；默认 0 表示用 `/tokenize` 自动校准 |
| `--seeds` | `3` | 每档位随机 seed 数（取均值/中位数，抗抖动） |
| `--random-seeds` | 关 | 用系统熵生成真正随机的 seed（重复运行文本也不同） |
| `--max-tokens` | `1` | 请求 max_tokens（默认 1：只测 prefill） |
| `--warmup` | 关 | 先发一个短请求暖机（避免首次 kernel 编译混入计时） |
| `--timeout` | `1800` | 单请求超时（秒） |
| `--json-output` | — | 结果写入 JSON 文件 |
| `--quiet` | — | 每档位一行数字 |

> `--seeds` 与 `--random-seeds` 组合：`--seeds` 始终决定每档位的样本数；
> `--random-seeds` 只改变 seed 来源（系统熵），开启后同一命令重复运行的文本
> 不同，彻底防前缀缓存（但样本数仍由 `--seeds` 决定）。

## prefill 测量的三个设计点

1. **`max_tokens=1` 剥离 decode**：预填充阶段只计算一次完整 prompt 的
   KV cache，把 `max_tokens` 压到 1 可避免大量 decode 时间混入计时，
   得到的是纯粹的 prefill 算力（`prefill_tok/s = prompt_tokens / 单请求墙钟`）。
2. **防缓存伪随机文本**：用 `random.Random(seed)` 生成"规律一致、但每
   seed 前缀不同"的 ASCII 数列文本——同一 `--seed` 完全可复现便于横向对比；
   不同 seed 首段 token 流不同，规避 vLLM 前缀缓存（radix cache）命中，
   保证测到的是真实 prefill 计算量。
3. **`/tokenize` 自动校准**：数列文本 BPE 后接近 1 字符 ≈ 1 token（实测
   char/token≈1.043），而默认假设常是 4.0；脚本直接用服务端 `/tokenize`
   精确校准字符/token 比，避免因估算误差导致 prompt 超 `max_model_len`
   返回 400。最终测速仍取响应的 `usage.prompt_tokens` 真实值。

## 开发

运行全部单元测试（标准库 unittest，不依赖 pytest）：

```bash
python -m unittest discover -s tests
```

静态检查（可选，需自行安装 pyright）：

```bash
pyright
```

## 项目结构

```
src/tokprobe/
  __init__.py       包元信息
  http.py           公共网络辅助：post_json / health_ok
  bench_serve.py    通用端到端并发压测（python -m tokprobe.bench_serve）
  bench_prefill.py  prefill 专项测速（python -m tokprobe.bench_prefill）
  report.py         统计与输出辅助：mean / median / percentile、--quiet 单行输出
tests/
  test_report.py    统计函数与 --quiet 输出格式
  test_http.py      post_json / health_ok（mock，不发真实网络请求）
  test_prefill.py   build_prompt 可复现性与防缓存特性
```
