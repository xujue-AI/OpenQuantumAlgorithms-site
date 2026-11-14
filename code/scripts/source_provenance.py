"""Optional Git provenance for real checkouts, never ancestor repositories.

Public code downloads intentionally omit Git history. Their content hashes
remain authoritative; absence of Git provenance is recorded as ``None``.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from collections.abc import Iterable


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # A caller's repository selectors must not redirect this metadata lookup.
    for name in (
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
        "GIT_NAMESPACE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS",
    ):
        environment.pop(name, None)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def git_revision(root: Path | str) -> str | None:
    """Return HEAD only for a Git marker and top-level at this exact root."""
    directory = Path(root).resolve()
    marker = directory / ".git"
    if not (marker.is_dir() or marker.is_file()):
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if result.returncode or len(lines) != 2:
        return None
    if Path(lines[0]).resolve() != directory:
        return None
    revision = lines[1].strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) else None


def git_dirty(root: Path | str, paths: Iterable[str]) -> bool | None:
    """Return owned-path status only when the exact checkout is available."""
    directory = Path(root).resolve()
    if git_revision(directory) is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "-c", "core.fsmonitor=false", "status",
             "--porcelain", "--", *paths],
            capture_output=True, text=True, check=False, timeout=5,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None
