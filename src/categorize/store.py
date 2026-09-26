"""Where an OpenChoice keeps its options between runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol


class Store(Protocol):
    """Loads and saves one OpenChoice's state, a JSON-serializable dict."""

    def load(self) -> dict[str, Any] | None:
        """The saved state, or None if nothing has been saved yet."""
        ...

    def save(self, state: dict[str, Any]) -> None: ...


class JsonFileStore:
    """Keeps the state in a JSON file, rewritten atomically on every save.

    Use one process per file: writers in different processes overwrite each other's changes.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"JsonFileStore({str(self.path)!r})"

    def load(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return None

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(state, file, indent=2, ensure_ascii=False)
                file.write("\n")
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise
