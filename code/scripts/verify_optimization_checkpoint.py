#!/usr/bin/env python3
"""Rebuild the frozen optimization exports without repeating numerical training.

Use the locked Python environment. The archived producer and implementation
remain immutable; --resume writes only to a temporary output directory. All
scientific exports are compared byte-for-byte. Only runs.json's top-level
artifact_generation_wall_seconds is excluded from its JSON comparison, since
that field measures rebuilding exports rather than the original CPU study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "assets/generated/optimization/quality-scaling"
PRODUCER = "generate_optimization_quality_scaling.py"
IMPLEMENTATION = "optimization.py"
EXPECTED_SOURCE_HASHES = {
    "generator_sha256": "55202cf674f486e9c641eb83dbc5e1d658a16663335d0bd98b1fec41b6091861",
    "implementation_sha256": "b959c0852ddddcdba4de20dddec2df09bdcd14dae2bd2ef277fde032eab0f392",
}
EXPECTED_QASM_COUNT = 1873
BYTE_COMPARED_FILES = (
    "resources.json", "results.csv", "resources.csv", "groups.csv",
    "evaluations.csv", "circuit-manifest.json",
    "raw-study.json", "recipe.json", PRODUCER, IMPLEMENTATION, "LICENSE.txt",
)
REBUILD_TIMING_FIELD = "artifact_generation_wall_seconds"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON value: {value}")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def canonical_json(value: object) -> bytes:
    """Preserve JSON numeric types while ignoring object order/whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def inventory(directory: Path) -> dict:
    """Record all file bytes and directory entries, rejecting linked/special files."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Checkpoint must be a regular directory")
    files, directories = {}, []
    for path in sorted(directory.rglob("*")):
        mode = path.lstat().st_mode
        relative = path.relative_to(directory).as_posix()
        if stat.S_ISDIR(mode):
            directories.append(relative)
        elif stat.S_ISREG(mode):
            files[relative] = sha256(path)
        else:
            raise ValueError(f"Linked or special checkpoint entry: {relative}")
    return {"file_sha256": files, "directories": directories}


def assert_preserved(directory: Path, before: dict) -> None:
    if inventory(directory) != before:
        raise ValueError("Original checkpoint inventory or file hashes changed during verification")


def checked_file(directory: Path, name: str) -> Path:
    path = directory / name
    if (path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(directory.resolve())):
        raise ValueError(f"Missing or unsafe checkpoint file: {name}")
    return path


def verify_source_bindings(directory: Path, checkpoint: dict) -> dict:
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Raw checkpoint lacks source provenance")
    bindings = {}
    for name, key in ((PRODUCER, "generator_sha256"), (IMPLEMENTATION, "implementation_sha256")):
        expected = provenance.get(key)
        if expected != EXPECTED_SOURCE_HASHES[key]:
            raise ValueError(f"Raw checkpoint changes the frozen source binding: {name}")
        if sha256(checked_file(directory, name)) != expected:
            raise ValueError(f"Archived source differs from its checkpoint binding: {name}")
        bindings[name] = expected
    recipe = load_json(checked_file(directory, "recipe.json"))
    if canonical_json(recipe) != canonical_json(checkpoint.get("recipe")):
        raise ValueError("Recipe differs from the raw numerical checkpoint")
    return bindings


def verify_locked_interpreter(python: Path, lock_file: Path, checkpoint: dict) -> dict:
    provenance = checkpoint["provenance"]
    if sha256(lock_file) != provenance.get("environment_lock_sha256"):
        raise ValueError("Environment lock differs from the original checkpoint")
    packages = provenance.get("packages")
    if not isinstance(packages, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in packages.items()):
        raise ValueError("Checkpoint lacks exact package versions")
    code = (
        "import importlib.metadata,json,platform,sys; "
        "print(json.dumps({'python':platform.python_version(),"
        "'packages':{name:importlib.metadata.version(name) for name in json.loads(sys.argv[1])}}))"
    )
    result = subprocess.run([str(python), "-I", "-B", "-c", code, json.dumps(list(packages))],
                            check=True, capture_output=True, text=True, timeout=30)
    actual = json.loads(result.stdout, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    # CI requests the supported 3.12 family, not an obsolete exact security
    # patch. Original provenance stays untouched; the actual replay version is
    # reported, while exact package/lock and artifact checks remain mandatory.
    actual_python = actual.get("python") if isinstance(actual, dict) else None
    for version in (provenance.get("python"), actual_python):
        if not isinstance(version, str) or re.fullmatch(r"3\.12\.(0|[1-9][0-9]*)", version) is None:
            raise ValueError("Interpreter must use the Python 3.12 family of the locked checkpoint environment")
    expected = {"python": actual_python, "packages": packages}
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("Interpreter or installed packages differ from the locked checkpoint environment")
    return actual


def compare_runs_json(recorded: Path, rebuilt: Path) -> None:
    values = [load_json(recorded), load_json(rebuilt)]
    for value in values:
        timing = value.get(REBUILD_TIMING_FIELD)
        if (isinstance(timing, bool) or not isinstance(timing, (int, float))
                or not math.isfinite(timing) or timing < 0):
            raise ValueError("runs.json requires a finite nonnegative top-level rebuild wall time")
        del value[REBUILD_TIMING_FIELD]
    if canonical_json(values[0]) != canonical_json(values[1]):
        raise ValueError("runs.json differs beyond its top-level rebuild wall-time field")


def compare_exports(recorded: Path, rebuilt: Path) -> dict:
    # Inventory both trees first so links cannot hide missing/extra QASM files.
    original = inventory(recorded)
    replay = inventory(rebuilt)
    original_qasm = {name for name in original["file_sha256"] if name.endswith(".qasm")}
    replay_qasm = {name for name in replay["file_sha256"] if name.endswith(".qasm")}
    if original_qasm != replay_qasm:
        raise ValueError("Rebuilt QASM inventory differs from the complete checkpoint inventory")
    for name in (*BYTE_COMPARED_FILES, *sorted(original_qasm)):
        expected = checked_file(recorded, name).read_bytes()
        actual = checked_file(rebuilt, name).read_bytes()
        if expected != actual:
            raise ValueError(f"Rebuilt artifact bytes differ: {name}")
    compare_runs_json(checked_file(recorded, "runs.json"), checked_file(rebuilt, "runs.json"))
    return {"byte_compared_files": list(BYTE_COMPARED_FILES),
            "qasm_files_compared": len(original_qasm),
            "runs_json_ignored_top_level_fields": [REBUILD_TIMING_FIELD]}


def run_archived_producer(python: Path, producer: Path, output: Path, workdir: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([str(python), "-I", "-B", str(producer), "--resume", "--output-dir", str(output)],
                   cwd=workdir, env=environment, check=True, timeout=150)


def verify_checkpoint(directory: Path = DEFAULT_CHECKPOINT, *, python: Path | None = None,
                      lock_file: Path = ROOT / "requirements-lock.txt") -> dict:
    directory = directory.absolute()
    # Keep the venv executable path: resolving its symlink can discard the venv.
    python = Path(sys.executable) if python is None else python.absolute()
    before = inventory(directory)
    try:
        checkpoint = load_json(checked_file(directory, "raw-study.json"))
        bindings = verify_source_bindings(directory, checkpoint)
        environment = verify_locked_interpreter(python, lock_file, checkpoint)
        qasm_count = sum(name.endswith(".qasm") for name in before["file_sha256"])
        manifest = load_json(checked_file(directory, "circuit-manifest.json"))
        if qasm_count != EXPECTED_QASM_COUNT or manifest.get("unique_circuit_count") != qasm_count:
            raise ValueError("Checkpoint does not contain all 1873 frozen QASM outputs")
        with tempfile.TemporaryDirectory(prefix="openquantum-optimization-checkpoint-") as work:
            workdir = Path(work)
            if workdir.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Temporary replay must be outside the original checkpoint")
            output = workdir / "rebuild"
            output.mkdir()
            for name in ("raw-study.json", "recipe.json"):
                (output / name).write_bytes(checked_file(directory, name).read_bytes())
            run_archived_producer(python, checked_file(directory, PRODUCER), output, workdir)
            comparison = compare_exports(directory, output)
    finally:
        assert_preserved(directory, before)
    return {"status": "verified immutable optimization checkpoint replay",
            "training_repeated": False, "original_inventory_preserved": True,
            "original_files": len(before["file_sha256"]),
            "original_inventory_sha256": hashlib.sha256(canonical_json(before)).hexdigest(),
            "archived_source_sha256": bindings, "verified_environment": environment,
            "python_version_policy": "Python 3.12 patch releases; exact packages, lock and output verification; original recorded provenance preserved",
            **comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lock-file", type=Path, default=ROOT / "requirements-lock.txt")
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint(args.checkpoint_dir, python=args.python, lock_file=args.lock_file),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
