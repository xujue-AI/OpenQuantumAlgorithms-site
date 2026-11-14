#!/usr/bin/env python3
"""Build the pinned book with bounded renderer concurrency and owned processes.

The cache-only compatibility patch limits complete render/read/write tasks to
five at a time. It changes neither the installed SDK nor the client bundle.
Publication remains a separate step after the HTML and artifact checks.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_URL = "https://github.com/myst-templates/book-theme/archive/08951401cf4aa6710f040fe34dcf0c1351a08639.zip"
TEMPLATE_CACHE_ID = "59e676854fab5281680c180252d0b63de39f27b300085de4e49e28c8d162167d"
TEMPLATE_DIRECTORY = Path("_build/templates/site") / TEMPLATE_CACHE_ID
RENDERER_RELATIVE = TEMPLATE_DIRECTORY / "renderer/server/index.js"
RENDER_COMMAND = "node --env-file=./.env.prod renderer/server/index.js"
ORIGINAL_SHA256 = "e86340e4b26370b57a6b88922db57e0cf8f81814863f1c6b7d33622fd3d0fc80"
PATCHED_SHA256 = "bc820e4b0e3ef30c1b99d2b879536de74f3225d956bc5987b29b7efbeec404c6"
ORIGINAL_EXPRESSION = b"await Promise.all(A.map(e=>renderRenderablePath(N,e,t,f)))"
PATCHED_EXPRESSION = (
    b"for(let bookRenderOffset=0;bookRenderOffset<A.length;bookRenderOffset+=5)"
    b"await Promise.all(A.slice(bookRenderOffset,bookRenderOffset+5).map(e=>renderRenderablePath(N,e,t,f)))"
)
BOOTSTRAP_TIMEOUT_SECONDS = 600
HTML_TIMEOUT_SECONDS = 1200
STOP_GRACE_SECONDS = 2


def checked_path(root: Path, relative: Path, *, required: bool = True) -> Path:
    """Reject redirected cache/config paths before any bootstrap or write."""
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Book root must be a real directory")
    path = root
    for index, part in enumerate(relative.parts):
        if part in {".", ".."} or relative.is_absolute():
            raise ValueError("Book cache path must be project-relative")
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in book build path: {relative}")
        if index < len(relative.parts) - 1 and path.exists() and not path.is_dir():
            raise ValueError(f"Non-directory in book build path: {relative}")
    if path.exists() and not path.is_file():
        raise ValueError(f"Book input must be a regular file: {relative}")
    if required and not path.is_file():
        raise ValueError(f"Missing book build input: {relative}")
    return path


def verify_theme(root: Path) -> None:
    path = checked_path(root, Path("myst.yml"))
    configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(configuration, dict) or not isinstance(configuration.get("site"), dict)
            or configuration["site"].get("template") != TEMPLATE_URL):
        raise ValueError("Book build requires the exact pinned theme URL")
    if hashlib.sha256(TEMPLATE_URL.encode()).hexdigest() != TEMPLATE_CACHE_ID:
        raise ValueError("Pinned theme URL does not match its cache identity")


def guarded_renderer(data: bytes) -> tuple[bytes, bool]:
    """Return the exact checked patch, or recognize the already patched bytes."""
    digest = hashlib.sha256(data).hexdigest()
    if digest == PATCHED_SHA256:
        if data.count(PATCHED_EXPRESSION) != 1 or data.count(ORIGINAL_EXPRESSION) != 0:
            raise ValueError("Patched renderer expression is ambiguous")
        return data, False
    if digest != ORIGINAL_SHA256:
        raise ValueError("Pinned renderer checksum differs; inspect the upstream bundle")
    if data.count(ORIGINAL_EXPRESSION) != 1 or data.count(PATCHED_EXPRESSION) != 0:
        raise ValueError("Original renderer expression is ambiguous")
    patched = data.replace(ORIGINAL_EXPRESSION, PATCHED_EXPRESSION)
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise ValueError("Patched renderer checksum differs from the verified fix")
    return patched, True


def verify_cached_paths(root: Path, *, required: bool) -> Path:
    renderer = checked_path(root, RENDERER_RELATIVE, required=required)
    metadata = checked_path(root, TEMPLATE_DIRECTORY / "template.yml", required=required)
    checked_path(root, TEMPLATE_DIRECTORY / ".env.prod", required=required)
    if metadata.is_file():
        template = yaml.safe_load(metadata.read_text(encoding="utf-8"))
        if (not isinstance(template, dict) or not isinstance(template.get("build"), dict)
                or template["build"].get("render") != RENDER_COMMAND):
            raise ValueError("Pinned template does not execute the guarded renderer")
    if renderer.is_file():
        guarded_renderer(renderer.read_bytes())
    return renderer


def patch_renderer(root: Path) -> bool:
    verify_theme(root)
    path = verify_cached_paths(root, required=True)
    original = path.read_bytes()
    patched, changed = guarded_renderer(original)
    if not changed:
        return False
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary = tempfile.mkstemp(prefix=".book-renderer-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(patched)
        os.chmod(temporary_path, mode)
        # Recheck the original target and ancestors before atomic replacement.
        checked_path(root, RENDERER_RELATIVE)
        if path.read_bytes() != original:
            raise ValueError("Renderer changed while applying the guarded patch")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def stop_owned_group(process: subprocess.Popen) -> None:
    """Stop only this wrapper's fresh POSIX session, including descendants."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except OSError as error:
            print(f"Could not signal the owned book subprocess group: {error}", file=sys.stderr)
        try:
            process.wait(timeout=STOP_GRACE_SECONDS)
        except BaseException:
            # Cleanup must still attempt SIGKILL and preserve the original
            # failed status/exception, even if waiting itself fails.
            pass


def run_step(command: list[str], root: Path, timeout: float, environment: dict) -> int:
    if os.name != "posix":
        raise ValueError("The publication build wrapper requires POSIX process groups")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Book build timeout must be finite and positive")
    process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)
    try:
        status = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_owned_group(process)
        print(f"Book build step exceeded its {timeout:g}-second limit", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        stop_owned_group(process)
        return 130
    except BaseException:
        stop_owned_group(process)
        raise
    if status:
        stop_owned_group(process)
    return status if status >= 0 else 128 - status


def reset_generated_site_cache(root: Path) -> None:
    checked_path(root, Path("_build/site/config.json"), required=False)
    site = Path(root).absolute() / "_build/site"
    if site.exists():
        if not site.is_dir():
            raise ValueError("Generated site cache must be a directory")
        if any(path.is_symlink() for path in site.rglob("*")):
            raise ValueError("Symlink in generated site cache")
        shutil.rmtree(site)


def build_book(root: Path = ROOT) -> int:
    root = Path(root).absolute()
    verify_theme(root)
    # A redirected _build must fail before MyST can install into it.
    verify_cached_paths(root, required=False)
    reset_generated_site_cache(root)
    environment = dict(os.environ)
    environment.setdefault("HOST", "127.0.0.1")
    command = [sys.executable, "-m", "jupyter_book", "build"]
    print("Bootstrapping the pinned book renderer", flush=True)
    status = run_step([*command, "--site", "--strict", "--ci"], root,
                      BOOTSTRAP_TIMEOUT_SECONDS, environment)
    if status:
        return status
    changed = patch_renderer(root)
    print("Applied verified renderer concurrency guard" if changed else
          "Verified existing renderer concurrency guard", flush=True)
    return run_step([*command, "--html", "--strict", "--ci", "--keep-host"], root,
                    HTML_TIMEOUT_SECONDS, environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        status = build_book(args.root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Book build stopped: {error}", file=sys.stderr)
        status = 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
