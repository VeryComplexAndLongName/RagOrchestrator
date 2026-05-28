from __future__ import annotations

from rag_orchestrator.perf import percentile_ms, summarize_latencies


def test_percentile_and_summary() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile_ms(values, 0) == 1.0
    assert percentile_ms(values, 100) == 5.0

    stats = summarize_latencies(values)
    assert stats.count == 5
    assert stats.avg_ms == 3.0
    assert stats.p50_ms == 3.0
    assert stats.p95_ms >= stats.p50_ms
