from __future__ import annotations

import json
from pathlib import Path


class JsonFileMigrationStore:
    """Simple migration state store that works for any backend."""

    def __init__(self, file_path: str = ".rag_migrations.json") -> None:
        self.file_path = Path(file_path)

    def get_current_version(self, namespace: str) -> int:
        if not self.file_path.exists():
            return 0
        payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        return int(payload.get(namespace, 0))

    def set_current_version(self, namespace: str, version: int) -> None:
        payload: dict[str, int] = {}
        if self.file_path.exists():
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        payload[namespace] = version
        self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
