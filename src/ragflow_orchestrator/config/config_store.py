from __future__ import annotations

from ragflow_orchestrator.config.module_config import ModuleConfig


class ConfigStore:
    def __init__(self, config: ModuleConfig | None = None) -> None:
        self._config = config or ModuleConfig()

    def get(self, path: str, default: object | None = None) -> object | None:
        current: object = self._config.model_dump()
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def get_config(self) -> ModuleConfig:
        return self._config

    def set_config(self, config: ModuleConfig) -> None:
        self._config = config

    def as_dict(self) -> dict[str, object]:
        return self._config.model_dump()
