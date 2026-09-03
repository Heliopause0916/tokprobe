"""tokprobe：轻量 vLLM 测速 / prefill 探针。

两个命令行入口（见各模块 ``main()``）：

- ``python -m tokprobe.bench_serve``   通用 OpenAI 兼容接口并发压测
- ``python -m tokprobe.bench_prefill``  prefill 阶段专项测速
"""

__version__ = "0.1.0"
