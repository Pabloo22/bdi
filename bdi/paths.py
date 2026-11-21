"""Filesystem locations used across the project."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_DIR = REPO_ROOT / "bdi" / "vendored"
OEL_DIR = VENDORED_DIR / "oel"
OPCD_DIR = VENDORED_DIR / "opcd"
