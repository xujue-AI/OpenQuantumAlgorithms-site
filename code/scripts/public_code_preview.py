#!/usr/bin/env python3
"""Prepare and verify a partial source preview against the complete download.

The archive remains the complete runnable package. The website's ``code/``
directory retains its non-asset files and full archive provenance; PREVIEW.json
identifies every omitted asset at its identical canonical site path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from verify_public_package import checked_path, extract_package, verify_manifest

SCHEMA = "openquantum-source-preview-v1"
ARCHIVE_PATH = "assets/generated/openquantum-code.zip"
SCOPE = "Partial source preview; use the complete archive for fixtures, tests and notebook execution"


def site_directory(root: Path) -> Path:
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Public site must be a regular directory")
    return root.resolve()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def regular_file(root: Path, name: str) -> Path:
    root = Path(root).resolve()
    path = checked_path(root, name)
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Linked source-preview path: {name}")
    return path


def verify_archive(archive_path: Path, package: Path, provenance: dict) -> None:
    """Require every original archive member and its exact full-package hash."""
    expected = dict(provenance["file_sha256"])
    expected["PROVENANCE.json"] = digest(regular_file(package, "PROVENANCE.json"))
    expected = {"openquantum-examples/" + name: value for name, value in expected.items()}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != len(expected) or {entry.filename for entry in members} != set(expected):
            raise ValueError("Complete archive members differ from the full-package manifest")
        for entry in members:
            if entry.is_dir() or stat.S_IFMT(entry.external_attr >> 16) not in (0, stat.S_IFREG):
                raise ValueError(f"Complete archive member is not a regular file: {entry.filename}")
            hasher = hashlib.sha256()
            with archive.open(entry) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected[entry.filename]:
                raise ValueError(f"Complete archive member differs from its manifest: {entry.filename}")


def expected_preview(package: Path, site_root: Path) -> dict:
    """Verify the full package and bind its assets to identical site files."""
    provenance = verify_manifest(package)
    archive_path = regular_file(site_root, ARCHIVE_PATH)
    verify_archive(archive_path, package, provenance)
    retained = {}
    omitted = {}
    for name, expected in provenance["file_sha256"].items():
        if name.startswith("assets/"):
            if digest(regular_file(site_root, name)) != expected:
                raise ValueError(f"Canonical asset differs from the complete archive: {name}")
            omitted[name] = {"site_path": name, "sha256": expected}
        else:
            retained[name] = expected
    retained["PROVENANCE.json"] = digest(regular_file(package, "PROVENANCE.json"))
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "source_commit": provenance["source_commit"],
        "archive": {"path": ARCHIVE_PATH, "sha256": digest(archive_path)},
        "retained_sha256": retained,
        "omitted_assets": omitted,
    }


def verify_preview(site_root: Path, *, preview: Path | None = None, package: Path | None = None) -> dict:
    """Verify retained sources, omitted canonical assets and the full archive.

    ``package`` is an already extracted archive, for preparation to reuse its
    extraction. Its complete manifest is still checked here. Ordinary public
    verification always extracts and verifies the complete archive itself.
    """
    site_root = site_directory(site_root)
    preview = site_root / "code" if preview is None else Path(preview)
    if preview.is_symlink() or not preview.is_dir():
        raise ValueError("Source preview must be a regular directory")
    if package is None:
        with tempfile.TemporaryDirectory(prefix="openquantum-preview-check-") as temporary:
            package = extract_package(regular_file(site_root, ARCHIVE_PATH), Path(temporary))
            return verify_preview(site_root, preview=preview, package=package)
    expected = expected_preview(Path(package), site_root)
    actual = json.loads(regular_file(preview, "PREVIEW.json").read_text())
    if actual != expected:
        raise ValueError("Source-preview manifest differs from the complete archive and canonical assets")
    names = {path.relative_to(preview).as_posix() for path in preview.rglob("*") if path.is_file()}
    if names != set(expected["retained_sha256"]) | {"PREVIEW.json"}:
        raise ValueError("Source-preview contents differ from its retained-file manifest")
    for name, expected_hash in expected["retained_sha256"].items():
        if digest(regular_file(preview, name)) != expected_hash:
            raise ValueError(f"Source-preview file differs from the complete archive: {name}")
    if any(path.is_symlink() for path in preview.rglob("*")):
        raise ValueError("Source preview contains a linked path")
    return actual


def write_preview(package: Path, site_root: Path) -> dict:
    """Atomically replace only ``code/`` after validating the complete inputs."""
    site_root = site_directory(site_root)
    destination = site_root / "code"
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Public code output must be a regular directory")
    manifest = expected_preview(Path(package), site_root)
    temporary = Path(tempfile.mkdtemp(prefix=".openquantum-preview-", dir=site_root.parent))
    remove_staging = True
    try:
        staging = temporary / "code"
        staging.mkdir()
        for name in manifest["retained_sha256"]:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(regular_file(Path(package), name), target)
        (staging / "PREVIEW.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_preview(site_root, preview=staging, package=package)
        previous = temporary / "previous"
        if destination.exists():
            destination.rename(previous)
        try:
            staging.rename(destination)
        except BaseException:
            if previous.exists():
                try:
                    previous.rename(destination)
                except BaseException as rollback_error:
                    remove_staging = False
                    raise RuntimeError(f"Source-preview replacement failed; previous code preserved at {previous}") from rollback_error
            raise
    finally:
        if remove_staging:
            shutil.rmtree(temporary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_root", type=Path, help="Prepared public site containing code/ and the complete archive")
    args = parser.parse_args()
    record = verify_preview(args.site_root)
    print(f"Verified source preview: {len(record['retained_sha256'])} retained files, "
          f"{len(record['omitted_assets'])} canonical assets, complete archive manifest")


if __name__ == "__main__":
    main()
