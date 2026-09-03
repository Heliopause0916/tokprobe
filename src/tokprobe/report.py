"""统计与输出辅助：均值 / 中位数 / 分位数，以及 --quiet 单行输出。

数值口径：

- ``mean``：算术平均，空样本返回 0.0（与原脚本 ``if xs else 0.0`` 一致）；
- ``median``：**对偶数个样本取中间两个值的算术平均**（修正了原实现
  ``sorted(tps)[len // 2]`` 在偶数样本下取不到真中位数的问题）；
- ``percentile``：按线性插值计算，**P50 与修正后的 median 完全一致**。

--quiet 输出函数保持原有字段名与格式不变，仅将格式化逻辑集中于此。
"""

from typing import Sequence


def mean(xs: Sequence[float]) -> float:
    """算术平均；空样本返回 0.0。"""
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def median(xs: Sequence[float]) -> float:
    """中位数；偶数个样本取中间两个值的算术平均；空样本返回 0.0。"""
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2.0


def percentile(xs: Sequence[float], p: float) -> float:
    """分位数（线性插值法，范围按 numpy 默认口径）。

    对 p=50 的结果与 :func:`median` 完全一致（含偶数样本的中位修正）。
    空样本返回 0.0。
    """
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    if n == 1:
        return ys[0]
    k = (n - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def serve_quiet_line(wall: float, ok_count: int, total: int,
                     total_ct: int, total_pt: int, avg_tps: float,
                     per_req_mean: float, per_req_p50: float) -> str:
    """bench_serve 的 --quiet 单行输出（字段名与格式与原脚本一致）。"""
    return (f"wall={wall:.2f}s requests={ok_count}/{total} "
            f"total_ct={total_ct} total_pt={total_pt} "
            f"avg_tps={avg_tps:.1f} "
            f"per_req_mean={per_req_mean:.1f} per_req_p50={per_req_p50:.1f}")


def prefill_quiet_line(target_tokens: int, ok: int, runs: int,
                       mean_pt: float, mean_tps: float,
                       med_tps: float) -> str:
    """bench_prefill 的 --quiet 单行输出（每档位一行，格式与原脚本一致）。"""
    return (f"target={target_tokens} ok={ok}/{runs} "
            f"emp_pt={mean_pt:.0f} "
            f"prefill_mean={mean_tps:.1f} "
            f"prefill_med={med_tps:.1f} tok/s")
