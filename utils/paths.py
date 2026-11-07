"""Resolve local paths relative to the HiST-FEAST project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
