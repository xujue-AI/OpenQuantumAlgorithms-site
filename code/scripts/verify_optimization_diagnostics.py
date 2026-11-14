#!/usr/bin/env python3
"""Replay the frozen 36-run diagnostics without changing its numerical records.

The complete saved producer bundle is copied outside the original directory.
Its preserved producer runs with --resume only after all 60 completed numerical
checkpoints pass preflight. Every replayed file is compared byte-for-byte except
the finite nonnegative top-level export wall time in runs.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "assets/generated/optimization/diagnostics"
PRODUCER = "generate_optimization_diagnostics.py"
IMPLEMENTATION = "optimization.py"
CAPS = {"files": 1500, "entries": 1600, "file_bytes": 32 * 1024**2,
        "total_bytes": 128 * 1024**2, "replay_seconds": 150}
TOP_LEVEL_FILES = frozenset({
    "LICENSE.txt", "circuit-manifest.json", "control-evaluations.csv", "controls.csv",
    "evaluations.csv", "freeze.json", PRODUCER, "groups.csv", "margins.svg",
    "numerical-timing.json", IMPLEMENTATION, "overview.md", "quality.svg",
    "raw-study.json", "recipe.json", "requirements-lock.txt", "resources.csv",
    "resources.json", "results.csv", "runs.json", "summary.md", "workload.svg",
})
EXPECTED_FILE_HASHES = {
    "recipe.json": "3b991eaece34b43116afbbbc85d5e8da9e76fe7c9ff217a45c254989c40f5dd9",
    IMPLEMENTATION: "a617b5e8d43b648427e5ceaa5e35d3f50ea14c02464465084f23f457ffb5855d",
    PRODUCER: "0c0f3ff0f177dca99a00cc611261c3b6c82ce39be251e8d3a54026b0f5346f3e",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE.txt": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
    "freeze.json": "2e18c7677ebd992a2642c754397e8d48fdea8287dd9aa65474fa3ef3cc76a3b7",
    "raw-study.json": "a8649f45dd948302f201654e9ad341824363357fd37f8ed343281cdf9d846688",
    "numerical-timing.json": "a4b73ad6aa27b6a31e9822c1dcfd075334cf94e4b38234b131125bda5d03b186",
    "circuit-manifest.json": "3ac01d8e8c60469a2ea0effadf11f21efa2fb747f3e61cc3d28ce67337ff1c94",
}
EXPECTED_QASM_COUNT = 1147

# Reuse the existing JSON type/duplicate-key and environment checks without
# changing the earlier quality-scaling replay contract.
_SPEC = importlib.util.spec_from_file_location(
    "_openquantum_checkpoint_common", Path(__file__).with_name("verify_optimization_checkpoint.py"))
_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMON)
load_json = _COMMON.load_json
canonical_json = _COMMON.canonical_json
checked_file = _COMMON.checked_file
compare_runs_json = _COMMON.compare_runs_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(directory: Path) -> dict:
    """Hash a bounded regular tree before any producer is invoked."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Diagnostics checkpoint must be a regular directory")
    files, directories, total, entries = {}, [], 0, 0
    for current, folders, names in os.walk(directory, followlinks=False):
        for name in sorted(folders + names):
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            entries += 1
            if entries > CAPS["entries"]:
                raise ValueError("Diagnostics checkpoint exceeds its entry cap")
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(relative)
            elif stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
                if (len(files) >= CAPS["files"] or metadata.st_size > CAPS["file_bytes"]
                        or total > CAPS["total_bytes"]):
                    raise ValueError("Diagnostics checkpoint exceeds its file/byte cap")
                files[relative] = sha256(path)
            else:
                raise ValueError(f"Linked or special diagnostics checkpoint entry: {relative}")
    return {"file_sha256": files, "directories": sorted(directories), "total_bytes": total}


def assert_preserved(directory: Path, before: dict) -> None:
    if inventory(directory) != before:
        raise ValueError("Original diagnostics inventory or file hashes changed during verification")


def _expected_checkpoints(recipe: dict, raw: dict, identity: dict) -> dict[str, dict]:
    expected = {}
    cases = recipe.get("cases")
    seeds = recipe.get("seeds")
    if not isinstance(cases, list) or not isinstance(seeds, list) or len(set(seeds)) != len(seeds):
        raise ValueError("Frozen diagnostics recipe lacks its complete case/seed schedule")
    case_ids = [case.get("case_id") for case in cases]
    if (not all(isinstance(case, str) and re.fullmatch(r"[a-z0-9-]+", case) for case in case_ids)
            or len(set(case_ids)) != len(case_ids)):
        raise ValueError("Invalid or repeated diagnostic case ID")
    expected_sampled = {f"{case}-{start}-seed-{seed}"
                        for case in case_ids for start in ("original", "alternate") for seed in seeds}
    expected_controls = {f"{case}-{start}-{mode}"
                         for case in case_ids for start in ("original", "alternate")
                         for mode in ("exact-objective", "expected-margin")}
    sampled = [run for case in raw.get("cases", []) for run in case.get("runs", [])]
    controls = raw.get("controls", [])
    for kind, runs, required in (("sampled", sampled, expected_sampled),
                                 ("control", controls, expected_controls)):
        if not isinstance(runs, list):
            raise ValueError("Missing completed numerical diagnostics")
        keys = [run.get("run_id") for run in runs]
        if len(keys) != len(required) or set(keys) != required:
            raise ValueError("Raw study lacks the complete prescribed numerical runs")
        for key, run in zip(keys, runs):
            expected[f"checkpoints/{kind}-{key}.json"] = {
                "schema_version": 1, "kind": kind, "key": key, "identity": identity,
                "status": "completed", "run": run,
            }
    return expected


def preflight(directory: Path, before: dict) -> dict:
    for name, expected in EXPECTED_FILE_HASHES.items():
        if before["file_sha256"].get(name) != expected:
            raise ValueError(f"Missing or changed frozen diagnostics file: {name}")
    recipe = load_json(checked_file(directory, "recipe.json"))
    freeze = load_json(checked_file(directory, "freeze.json"))
    raw = load_json(checked_file(directory, "raw-study.json"))
    identity = {key: EXPECTED_FILE_HASHES[name] for key, name in (
        ("recipe_sha256", "recipe.json"), ("implementation_sha256", IMPLEMENTATION),
        ("generator_sha256", PRODUCER), ("environment_lock_sha256", "requirements-lock.txt"))}
    if (freeze.get("status") != "recipe_and_producers_frozen_before_execution"
            or freeze.get("identity") != identity or raw.get("checkpoint_identity") != identity
            or canonical_json(raw.get("recipe")) != canonical_json(recipe)
            or raw.get("execution_status") != "completed_bounded_ideal_sampling_and_exact_cpu_controls"):
        raise ValueError("Diagnostics numerical record is incomplete or differs from its freeze")
    provenance = raw.get("provenance")
    if (not isinstance(provenance, dict) or any(provenance.get(key) != value for key, value in identity.items())
            or provenance.get("freeze_sha256") != EXPECTED_FILE_HASHES["freeze.json"]
            or raw.get("numerical_timing_sha256") != EXPECTED_FILE_HASHES["numerical-timing.json"]):
        raise ValueError("Raw diagnostics source/timing bindings differ from the frozen files")
    timing = load_json(checked_file(directory, "numerical-timing.json"))
    if timing.get("status") != "complete" or timing.get("identity") != identity:
        raise ValueError("Numerical timing record is incomplete")
    checkpoints = _expected_checkpoints(recipe, raw, identity)
    for name, record in checkpoints.items():
        if canonical_json(load_json(checked_file(directory, name))) != canonical_json(record):
            raise ValueError(f"Missing, changed or incomplete numerical checkpoint: {name}")
    manifest = load_json(checked_file(directory, "circuit-manifest.json"))
    unique = manifest.get("unique_circuits")
    if (not isinstance(unique, list) or len(unique) != EXPECTED_QASM_COUNT
            or manifest.get("unique_circuit_count") != EXPECTED_QASM_COUNT):
        raise ValueError("Diagnostics circuit inventory is incomplete")
    qasm_files = set()
    for entry in unique:
        digest = entry.get("sha256")
        name = entry.get("file")
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or name != f"circuits/{digest}.qasm" or name in qasm_files
                or before["file_sha256"].get(name) != digest):
            raise ValueError("Missing or changed source QASM binding")
        qasm_files.add(name)
    required = TOP_LEVEL_FILES | qasm_files | checkpoints.keys()
    if set(before["file_sha256"]) != required or before["directories"] != ["checkpoints", "circuits"]:
        raise ValueError("Diagnostics checkpoint file/directory inventory is incomplete or unexpected")
    compare_runs_json(checked_file(directory, "runs.json"), checked_file(directory, "runs.json"))
    return {"raw": raw, "sampled_runs": sum(name.startswith("checkpoints/sampled-") for name in checkpoints),
            "controls": sum(name.startswith("checkpoints/control-") for name in checkpoints),
            "qasm_files": len(qasm_files), "archived_source_sha256": identity}


def compare_exports(recorded: Path, replay: Path) -> dict:
    original, rebuilt = inventory(recorded), inventory(replay)
    if (set(original["file_sha256"]) != set(rebuilt["file_sha256"])
            or original["directories"] != rebuilt["directories"]):
        raise ValueError("Replayed diagnostics file/directory inventory differs")
    for name, expected in original["file_sha256"].items():
        if name != "runs.json" and rebuilt["file_sha256"][name] != expected:
            raise ValueError(f"Replayed diagnostics artifact bytes differ: {name}")
    compare_runs_json(checked_file(recorded, "runs.json"), checked_file(replay, "runs.json"))
    return {"byte_compared_files": len(original["file_sha256"]) - 1,
            "runs_json_ignored_top_level_fields": ["artifact_generation_wall_seconds"]}


def run_archived_producer(python: Path, producer: Path, output: Path, workdir: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen([str(python), "-I", "-B", str(producer), "--resume",
                                "--output-dir", str(output)], cwd=workdir, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name == "posix")
    try:
        stdout, stderr = process.communicate(timeout=CAPS["replay_seconds"])
    except BaseException:
        # A descendant can still hold a pipe after the direct child exits.
        # Terminate the entire group created for this replay on interruption.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, "preserved diagnostics --resume",
                                            output=stdout, stderr=stderr)


def verify_checkpoint(directory: Path = DEFAULT_CHECKPOINT, *, python: Path | None = None) -> dict:
    directory = directory.absolute()
    # Resolving a venv interpreter symlink would discard its environment.
    python = Path(sys.executable) if python is None else python.absolute()
    before = inventory(directory)
    try:
        record = preflight(directory, before)
        environment = _COMMON.verify_locked_interpreter(
            python, checked_file(directory, "requirements-lock.txt"), record["raw"])
        if environment["python"] != record["raw"]["provenance"]["python"]:
            raise ValueError("The frozen diagnostic producer requires its recorded Python patch version")
        with tempfile.TemporaryDirectory(prefix="openquantum-diagnostics-replay-") as temporary:
            workdir = Path(temporary)
            if workdir.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Temporary diagnostics replay must be outside the original checkpoint")
            replay = workdir / "replay"
            shutil.copytree(directory, replay)
            if inventory(replay) != before:
                raise ValueError("Diagnostics source changed while copying the temporary replay")
            run_archived_producer(python, checked_file(replay, PRODUCER), replay, workdir)
            comparison = compare_exports(directory, replay)
    finally:
        assert_preserved(directory, before)
    return {"status": "verified immutable optimization diagnostics replay", "training_repeated": False,
            "original_inventory_preserved": True, "original_files": len(before["file_sha256"]),
            "original_bytes": before["total_bytes"],
            "original_inventory_sha256": hashlib.sha256(canonical_json(before)).hexdigest(),
            "sampled_runs": record["sampled_runs"], "controls": record["controls"],
            "qasm_files_compared": record["qasm_files"],
            "archived_source_sha256": record["archived_source_sha256"],
            "verified_environment": environment, "caps": CAPS, **comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint(args.checkpoint_dir, python=args.python), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
