from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Common project paths."""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"


def find_project_root(start: Path | None = None) -> Path:
    """Find repo root by walking upwards until we see expected markers."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "notebooks").exists() and (p / "data").exists():
            return p
        if (p / ".git").exists():
            return p
    return cur


