"""Exclude execution caches from the deliberately public website files."""
from pathlib import Path
import shutil

RUNTIME_NAMES = frozenset({"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store"})
RUNTIME_SUFFIXES = frozenset({".pyc", ".pyo"})


def is_runtime_artifact(name: str) -> bool:
    return name in RUNTIME_NAMES or Path(name).suffix in RUNTIME_SUFFIXES


def ignore_runtime_artifacts(directory, names):
    """The copytree ignore callback; scientific source and result files remain."""
    return {name for name in names if is_runtime_artifact(name)}


def remove_runtime_artifacts(root: Path) -> None:
    """Prune generated output only, without following links into other folders."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Public output root must not be a symlink")
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if not is_runtime_artifact(path.name):
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)
