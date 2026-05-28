from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class MigrationStepDef:
    version: int
    description: str
    up: Callable[[], None]
    down: Callable[[], None]


class MigrationManager:
    def __init__(self, namespace: str, store: object, steps: list[MigrationStepDef]) -> None:
        self.namespace = namespace
        self.store = store
        self.steps = sorted(steps, key=lambda x: x.version)

    def current_version(self) -> int:
        return int(self.store.get_current_version(self.namespace))

    def upgrade(self, target_version: int | None = None) -> int:
        current = self.current_version()
        planned = [step for step in self.steps if step.version > current]
        if target_version is not None:
            planned = [step for step in planned if step.version <= target_version]

        for step in planned:
            step.up()
            self.store.set_current_version(self.namespace, step.version)
            current = step.version
        return current

    def downgrade(self, target_version: int) -> int:
        current = self.current_version()
        planned = [step for step in self.steps if target_version < step.version <= current]
        for step in reversed(planned):
            step.down()
            self.store.set_current_version(self.namespace, step.version - 1)
            current = step.version - 1
        return current
