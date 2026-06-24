from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _query_runs(conn: sqlite3.Connection, group_by: str, metric: str, limit: int) -> list[tuple]:
    if group_by not in {"scenario", "provider_kind", "embedder_provider", "pipeline_preset"}:
        raise ValueError("Unsupported --group-by for run metrics")
    if metric not in {"total_duration_ms", "chunks_per_second", "total_chunks", "duplicate_chunks_skipped"}:
        raise ValueError("Unsupported --metric for run metrics")

    sql = f"""
        SELECT
            {group_by} AS group_name,
            COUNT(*) AS runs,
            AVG({metric}) AS avg_metric,
            MIN({metric}) AS min_metric,
            MAX({metric}) AS max_metric,
            MAX(timestamp_utc) AS latest_ts
        FROM template_runs
        GROUP BY {group_by}
        ORDER BY avg_metric DESC
        LIMIT ?
    """
    return conn.execute(sql, (limit,)).fetchall()


def _query_quality(conn: sqlite3.Connection, metric: str, limit: int) -> list[tuple]:
    if metric not in {"precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"}:
        raise ValueError("Unsupported --metric for quality")

    sql = f"""
        SELECT
            q.strategy_name AS group_name,
            COUNT(*) AS runs,
            AVG(q.{metric}) AS avg_metric,
            MIN(q.{metric}) AS min_metric,
            MAX(q.{metric}) AS max_metric,
            MAX(r.timestamp_utc) AS latest_ts
        FROM template_quality q
        JOIN template_runs r ON r.id = q.run_id
        GROUP BY q.strategy_name
        ORDER BY avg_metric DESC
        LIMIT ?
    """
    return conn.execute(sql, (limit,)).fetchall()


def _print(rows: list[tuple], metric: str) -> None:
    print(f"group | runs | avg_{metric} | min_{metric} | max_{metric} | latest")
    print("--- | --- | --- | --- | --- | ---")
    for group_name, runs, avg_metric, min_metric, max_metric, latest_ts in rows:
        print(
            f"{group_name} | {runs} | {avg_metric:.4f} | {min_metric:.4f} | {max_metric:.4f} | {latest_ts}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare template run trends from experiment log")
    parser.add_argument("--db", default="loadtest/experiments.sqlite")
    parser.add_argument("--group-by", default="scenario")
    parser.add_argument("--metric", default="chunks_per_second")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Experiment DB not found: {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        if args.group_by == "strategy_name":
            rows = _query_quality(conn, metric=args.metric, limit=args.limit)
        else:
            rows = _query_runs(conn, group_by=args.group_by, metric=args.metric, limit=args.limit)
    finally:
        conn.close()

    if not rows:
        print("No rows found.")
        return 0

    _print(rows, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

