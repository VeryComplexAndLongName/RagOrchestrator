from __future__ import annotations

import argparse
import json

from ragflow_orchestrator.graph import SqlGraphStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Query repository graph")
    parser.add_argument("--db", default="rag_graph.sqlite", help="Path to graph sqlite db")

    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Find repositories by keyword")
    p_search.add_argument("keyword")
    p_search.add_argument("--limit", type=int, default=10)

    p_count = sub.add_parser("contributors", help="Get contributor count for repository")
    p_count.add_argument("full_name")

    sub.add_parser("popular", help="Get most popular repository")

    args = parser.parse_args()
    store = SqlGraphStore(args.db)

    if args.command == "search":
        result = store.find_repositories_by_keyword(args.keyword, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "contributors":
        count = store.get_contributor_count(args.full_name)
        print(json.dumps({"full_name": args.full_name, "contributors": count}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "popular":
        result = store.get_most_popular_repository()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

