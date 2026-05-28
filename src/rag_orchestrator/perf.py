from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LatencyStats:
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


def percentile_ms(values_ms: list[float], p: float) -> float:
    if not values_ms:
        return 0.0
    if p <= 0:
        return min(values_ms)
    if p >= 100:
        return max(values_ms)

    ordered = sorted(values_ms)
    # Linear interpolation percentile to reduce jump artifacts on small samples.
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latencies(values_ms: list[float]) -> LatencyStats:
    if not values_ms:
        return LatencyStats(count=0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)

    avg = sum(values_ms) / len(values_ms)
    return LatencyStats(
        count=len(values_ms),
        avg_ms=avg,
        p50_ms=percentile_ms(values_ms, 50),
        p95_ms=percentile_ms(values_ms, 95),
        p99_ms=percentile_ms(values_ms, 99),
    )
