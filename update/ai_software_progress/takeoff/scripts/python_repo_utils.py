from __future__ import annotations

from pathlib import Path


def resolve_python_repo_root(current_file: Path) -> Path:
    """Return the repository root that contains the Python model code."""
    branch_root = current_file.resolve().parents[1]
    candidate = branch_root / "model_config.py"
    if candidate.exists():
        return branch_root
    raise RuntimeError("model_config.py not found in repository root; the Python model is missing")
