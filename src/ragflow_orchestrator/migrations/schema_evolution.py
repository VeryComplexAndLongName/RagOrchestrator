from __future__ import annotations

from typing import Literal

Backend = Literal["sqlite", "pgvector", "qdrant"]


def add_field_sql(backend: Backend, table_name: str, field_name: str, sql_type: str) -> str | None:
    if backend == "qdrant":
        return None
    return f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {field_name} {sql_type};"


def drop_field_sql(backend: Backend, table_name: str, field_name: str) -> str | None:
    if backend == "qdrant":
        return None
    if backend == "sqlite":
        # SQLite does not support DROP COLUMN in old versions; create-copy-swap is required.
        return f"-- recreate table workflow required to drop column {field_name}"
    return f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {field_name};"


def rename_field_sql(backend: Backend, table_name: str, old_field: str, new_field: str) -> str | None:
    if backend == "qdrant":
        return None
    return f"ALTER TABLE {table_name} RENAME COLUMN {old_field} TO {new_field};"
