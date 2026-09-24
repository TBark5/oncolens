"""Small helpers for writing results to disk in a consistent format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python types so json can serialize them."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(data: dict[str, Any], path: Path) -> Path:
    """Write a dictionary to pretty-printed JSON, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(data), indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_to_markdown(df: "pd.DataFrame", index_label: str = "") -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table (no extra dependency)."""
    header = [index_label or (df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:g}" if isinstance(v, float) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
