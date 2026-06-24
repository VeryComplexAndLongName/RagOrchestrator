from __future__ import annotations

import os
import subprocess
import sys


def run_step(label: str, args: list[str], env: dict[str, str]) -> int:
    print(f"\n=== {label} ===")
    print("Command:", " ".join(args))
    completed = subprocess.run(args, env=env)
    return completed.returncode


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()

    env["NO_PROXY"] = env.get("NO_PROXY", "localhost,127.0.0.1,::1")
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["PGVECTOR_DSN"] = env.get(
        "PGVECTOR_DSN",
        "postgresql+psycopg://postgres:N0th1ing@localhost:5432/app",
    )

    python_exe = sys.executable

    preflight_cmd = [python_exe, os.path.join(root, "scripts", "preflight_check.py")]
    tests_cmd = [python_exe, "-m", "pytest", "-q", "tests/integration", "-ra"]

    rc = run_step("Preflight", preflight_cmd, env=env)
    if rc != 0:
        print("\nPreflight failed.")
        return rc

    rc = run_step("Integration tests", tests_cmd, env=env)
    if rc != 0:
        print("\nIntegration tests failed.")
        return rc

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

