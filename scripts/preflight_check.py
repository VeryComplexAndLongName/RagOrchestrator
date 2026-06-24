from __future__ import annotations

import os
import urllib.error
import urllib.request

from sqlalchemy import create_engine, text


def _status(label: str, ok: bool, details: str) -> None:
    marker = "OK" if ok else "FAIL"
    print(f"[{marker}] {label}: {details}")


def check_proxy() -> None:
    proxies = urllib.request.getproxies()
    active_transport_proxies = {key: value for key, value in proxies.items() if key.lower() in {"http", "https", "all"}}
    if active_transport_proxies:
        _status("effective proxy", False, str(proxies))
    else:
        _status("effective proxy", True, str(proxies) if proxies else "no proxy detected")


def _probe_url(url: str, timeout: int = 8, disable_proxy: bool = False) -> tuple[bool, str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if disable_proxy else urllib.request.build_opener()
    req = urllib.request.Request(url=url, method="GET")
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        return True, f"HTTP {response.status}; body={body[:180]}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTPError {exc.code}: {exc.reason}"
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"


def check_qdrant() -> None:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collections_url = qdrant_url.rstrip("/") + "/collections"

    ok_proxy, details_proxy = _probe_url(collections_url, disable_proxy=False)
    _status("Qdrant (default path)", ok_proxy, details_proxy)

    ok_direct, details_direct = _probe_url(collections_url, disable_proxy=True)
    _status("Qdrant (direct/no-proxy)", ok_direct, details_direct)


def check_postgres() -> None:
    dsn = os.getenv("PGVECTOR_DSN")
    if not dsn:
        user = os.getenv("PGUSER", "postgres")
        password = os.getenv("PGPASSWORD", "")
        host = os.getenv("PGHOST", "localhost")
        port = os.getenv("PGPORT", "5432")
        db = os.getenv("PGDATABASE", "app")
        auth = user if not password else f"{user}:{password}"
        dsn = f"postgresql+psycopg://{auth}@{host}:{port}/{db}"

    try:
        engine = create_engine(dsn)
        with engine.connect() as conn:
            row = conn.execute(text("select current_database(), current_user")).fetchone()
            ext = conn.execute(text("select 1 from pg_extension where extname='vector' limit 1")).fetchone()
        _status("PostgreSQL auth", True, f"db={row[0]} user={row[1]}")
        _status("pgvector extension", bool(ext), "vector extension found" if ext else "vector extension NOT found")
    except Exception as exc:  # pragma: no cover
        _status("PostgreSQL auth", False, str(exc))


def print_fix_commands() -> None:
    print("\nSuggested quick fixes (PowerShell):")
    print("$env:NO_PROXY='localhost,127.0.0.1,::1'")
    print("$env:HTTP_PROXY='' ; $env:HTTPS_PROXY=''")
    print("$env:PGVECTOR_DSN='postgresql+psycopg://postgres:N0th1ing@localhost:5432/app'")
    print("python -m pytest -q tests/integration -ra")


def main() -> None:
    print("== RAG Orchestrator Preflight ==")
    check_proxy()
    check_qdrant()
    check_postgres()
    print_fix_commands()


if __name__ == "__main__":
    main()

