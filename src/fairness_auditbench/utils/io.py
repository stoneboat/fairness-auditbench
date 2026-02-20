"""File I/O helpers."""

import json
import os
from pathlib import Path
from typing import Any, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Save a JSON-serialisable object to *path*."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)
