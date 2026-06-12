"""Repository path constants shared by helper modules."""

from pathlib import Path

_HELPERS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _HELPERS_DIR.parent.parent
