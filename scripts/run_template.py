from __future__ import annotations

import json
import sys

from ragflow_orchestrator.templates.runner import run_template_from_json


def main() -> int:
    config_path = "templates.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    report = run_template_from_json(config_path)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
