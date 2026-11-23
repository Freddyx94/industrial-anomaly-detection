"""Read config.yaml and resolve its paths.

Why: every notebook, script and test reads settings from one file, so a change (for example the
image size) is made once. Paths in the file are relative to the project root, so the project
works wherever it is cloned.
"""

from pathlib import Path
from typing import Any

import yaml

# src/inspection/config.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml (or another YAML file, e.g. in tests) as a plain dict."""
    path = path or PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    """Turn a path from config.yaml into an absolute path under the project root."""
    return PROJECT_ROOT / relative
