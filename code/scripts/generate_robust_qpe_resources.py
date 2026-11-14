#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a finite source family or the single reviewed robust-QPE cost task.

Outputs and caches must be fresh and outside assets/generated. There is no
automatic retry, alternate task, package installation or workflow mutation.
The wall/RSS checks are cooperative; a reviewed caller must provide an outer
process/time/memory boundary before approving native synthesis or estimation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAX_WALL_SECONDS = 180
MAX_RSS_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_FILES = 40
MAX_CACHE_BYTES = 64 * 1024 * 1024
MAX_CACHE_FILES = 128
SOURCE_PATHS = (
    "scripts/generate_robust_qpe_resources.py",
    "src/openquantum/robust_qpe_resources.py",
    "src/openquantum/instrument_resources.py",
    "src/openquantum/hadamard_qpe.py",
    "src/openquantum/robust_qpe.py",
    "src/openquantum/__init__.py",
    "src/openquantum/grover.py",
    "requirements-lock.txt", "LICENSE",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_fingerprints() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "assets/generated").rglob("*"))
            if p.is_file()}


def aggregate_digest(values: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_revision() -> tuple[str | None, bool | None]:
    """Copied source packets retain hash identity even without a Git checkout."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT.resolve():
            return None, None
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            return None, None
        revision = result.stdout.strip()
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            return None, None
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_PATHS],
                               cwd=ROOT, capture_output=True, text=True)
        return revision, bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except OSError:
        return None, None


def peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(output: Path) -> tuple[int, int]:
    files = []
    for path in output.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("The new output contains a symlink")
        if path.is_file():
            files.append(path)
    total = sum(p.stat().st_size for p in files)
    if len(files) > MAX_OUTPUT_FILES or total > MAX_OUTPUT_BYTES:
        raise RuntimeError("The finite output exceeded its file/byte cap")
    return len(files), total


def cache_inventory(cache: Path) -> tuple[int, int]:
    """Cooperative private-cache check; the native caller owns the outer cap."""
    files = []
    for path in cache.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("The private native cache contains a symlink")
        if path.is_file():
            files.append(path)
    total = sum(p.stat().st_size for p in files)
    if len(files) > MAX_CACHE_FILES or total > MAX_CACHE_BYTES:
        raise RuntimeError("The private native cache exceeded its observed cap")
    return len(files), total


def selected_output(path: Path) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("Use a fresh output directory; an existing attempt is never overwritten")
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError("Output/cache parents must not redirect through symlinks")
    path = path.resolve(strict=False)
    generated = (ROOT / "assets/generated").resolve()
    if path == ROOT or path in ROOT.parents or path == generated or generated in path.parents:
        raise ValueError("Keep new work outside existing source and generated artifacts")
    if any(path == (ROOT / rel) or path in (ROOT / rel).parents for rel in SOURCE_PATHS):
        raise ValueError("Output/cache would cover a protected source")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source-only", "physical"), required=True)
    parser.add_argument("--target-qubits", type=int, choices=(1, 2, 4, 8, 16, 32), default=1)
    parser.add_argument("--levels", type=int, choices=(2, 3, 4, 5, 6), default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-wall-seconds", type=int, choices=(180,), default=180)
    args = parser.parse_args(argv)
    if platform.python_version() != "3.12.14":
        parser.error("Use the locked Python 3.12.14 runtime")
    versions = {p: importlib.metadata.version(p) for p in ("qiskit", "numpy")}
    if versions["qiskit"] != "2.6.2":
        parser.error("Use the existing locked Qiskit 2.6.2 environment")
    if args.mode == "physical" and (args.target_qubits != 1 or args.levels != 2 or args.cache_dir is None):
        parser.error("Physical mode permits only n=1, L=2 and a fresh private cache")
    if args.mode == "source-only" and args.cache_dir is not None:
        parser.error("Source-only mode performs no native query and needs no cache")
    try:
        output = selected_output(args.output_dir)
        cache = selected_output(args.cache_dir) if args.cache_dir is not None else None
        if cache is not None and (cache == output or cache in output.parents or output in cache.parents):
            raise ValueError("Keep private native cache and result output disjoint")
    except ValueError as error:
        parser.error(str(error))
    started = time.monotonic()
    def check():
        if time.monotonic() - started >= args.max_wall_seconds:
            raise TimeoutError("Cooperative producer wall limit exceeded")
        if peak_rss_bytes() > MAX_RSS_BYTES:
            raise MemoryError("Observed producer peak RSS exceeded the reviewed cap")
        if output.exists():
            inventory(output)
        if cache is not None and cache.exists():
            cache_inventory(cache)
    def alarm(_signum, _frame):
        raise TimeoutError("POSIX producer wall timer expired")
    previous_handler = None
    if hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"):
        previous_handler = signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, args.max_wall_seconds)
    try:
        before = protected_fingerprints()
        check()
        output.mkdir(parents=True, exist_ok=False)
        from qiskit import qasm3
        from openquantum.robust_qpe_resources import (
            RobustResourceTask, build_generation_macrocycle, build_macrocycle,
            check_macrocycle_source, classical_exclusions, design_ledger,
            estimate_quantum_schedule, split_measured_trials,
            synthesize_macrocycle, validate_compiled_macrocycle,
        )
        task = RobustResourceTask()
        macro, trials = (build_macrocycle(task) if args.mode == "physical"
                         else build_generation_macrocycle(args.target_qubits, args.levels))
        ledger = design_ledger(args.target_qubits, args.levels)
        (output / "source-macrocycle.qasm").write_text(qasm3.dumps(macro))
        for index, trial in enumerate(trials):
            (output / f"source-fringe-{index}.qasm").write_text(qasm3.dumps(trial))
        write_json(output / "source-ledger.json", ledger)
        if args.mode == "source-only":
            del macro, trials, trial
        check()
        physical = checks = synthesis = None
        synthesis_seconds = estimation_seconds = None
        unavailable = False
        if args.mode == "physical":
            versions["qdk"] = importlib.metadata.version("qdk")
            if versions["qdk"] != "1.34.0":
                raise RuntimeError("Use the existing locked QDK 1.34.0 environment")
            cache.mkdir(parents=True, exist_ok=False)
            checkpoint = time.monotonic()
            compiled = synthesize_macrocycle(macro, task)
            synthesis_seconds = time.monotonic() - checkpoint
            check()
            checks = validate_compiled_macrocycle(compiled, task)
            synthesis = compiled.report
            (output / "clifford-t-macrocycle.qasm").write_text(qasm3.dumps(compiled.circuit))
            for index, trial in enumerate(split_measured_trials(compiled.circuit)):
                (output / f"clifford-t-fringe-{index}.qasm").write_text(qasm3.dumps(trial))
            write_json(output / "compiled-fringe-checks.json", checks)
            check()
            checkpoint = time.monotonic()
            try:
                physical = estimate_quantum_schedule(compiled, task, cache_dir=cache)
            except RuntimeError as error:
                if not str(error).startswith("No feasible point in the declared"):
                    raise
                unavailable = True
                physical = {"status": "no_feasible_point_in_the_frozen_quantum_schedule_domain",
                            "physical_resource_estimate_computed": False,
                            "classical": classical_exclusions(task),
                            "complete_end_to_end_cost": False,
                            "end_to_end_confidence_established": False}
            estimation_seconds = time.monotonic() - checkpoint
            check()
            physical["compiled_fringe_check_executed"] = True
            physical["compiled_fringe_checks_sha256"] = sha(output / "compiled-fringe-checks.json")
            write_json(output / "quantum-schedule-resources.json", physical)
        # Package only ordinary source/lock/license paths. No operational
        # documents, cache, grant, raw logs, private coordinates or .git are copied.
        for relative in SOURCE_PATHS:
            destination = output / "code" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        source_hashes = {p: sha(ROOT / p) for p in SOURCE_PATHS}
        after = protected_fingerprints()
        if before != after:
            raise RuntimeError("An existing generated artifact changed")
        (output / "README.md").write_text(
            "# Robust-QPE quantum schedule\n\n"
            "Source-only mode supplies actual primitive generators and design counts. "
            "Physical mode is bounded to one target, two dyadic levels and 400 trials per fringe. "
            "Its four fresh-preparation experiments run in one macrocycle repeated 400 times.\n\n"
            "Physical totals cover this quantum schedule, including input preparation, literal "
            "controlled calls, resets, readout, protected workspace, actual synthesis and capped "
            "factory supply. Count aggregation, reliable classical storage, rational phase "
            "inference, calibration, communication and their extra idle exposure are excluded. "
            "No full end-to-end cost or hardware confidence is established.\n\n"
            "The compiled checks use four tiny density evolutions and independent stored-angle "
            "fringes. They perform no shot sampling, phase recovery, complete population replay "
            "or formal proof. Guarded numerical synthesis is not an interval certificate for "
            "the robust probability promise. The existing baseline and records remain unchanged.\n\n"
            "Original source is MIT. Use the included repository-shaped code snapshot with the "
            "existing Python 3.12.14 lock. Qiskit 2.6.2, its recorded rsgridsynth 0.2.2 backend and "
            "QDK 1.34.0 supply circuits, actual synthesis and the conditional factory model. "
            "The installed native-wheel identity limitation is retained in synthesis metadata.\n")
        revision, dirty = source_revision()
        report = {
            "schema": "openquantum.robust_qpe_quantum_schedule_producer.v1",
            "status": ("physical_point_unavailable" if unavailable else
                       "bounded_compiled_fringe_checks_and_conditional_quantum_schedule" if args.mode == "physical" else
                       "generation_only_no_phase_synthesis_or_physical_result"),
            "mode": args.mode, "source_commit": revision, "source_dirty": dirty,
            "source_identity_scope": "matching repository root and source bytes" if revision is not None else "source byte snapshot only; no matching Git root",
            "source_sha256": source_hashes,
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "architecture": platform.machine(), "packages": versions,
                            "device": "local CPU; no quantum hardware or GPU"},
            "design_ledger": ledger,
            "fixed_physical_task": task.contract() if args.mode == "physical" else None,
            "source_check": check_macrocycle_source(macro, task) if args.mode == "physical" else None,
            "synthesis": synthesis, "synthesis_wall_seconds": synthesis_seconds,
            "native_estimation_wall_seconds": estimation_seconds,
            "compiled_fringe_check_executed": checks is not None,
            "physical_resource_estimate_computed": physical is not None and not unavailable,
            "classical": classical_exclusions(task) if args.mode == "physical" else {
                "status": "generation only; no classical phase inference or timing"},
            "complete_end_to_end_cost": False, "end_to_end_confidence_established": False,
            "sampling_performed": False, "phase_output_computed": False,
            "formal_proof_reproduced": False, "hardware_executed": False,
            "matched_improvement_computed": False, "retry": False,
            "protected_generated_file_count": len(before),
            "protected_generated_digest": aggregate_digest(before),
            "protected_generated_unchanged": True,
            "compute_caps": {"cooperative_wall_seconds": args.max_wall_seconds,
                "observed_self_peak_rss_bytes_maximum": MAX_RSS_BYTES,
                "output_bytes": MAX_OUTPUT_BYTES, "output_files": MAX_OUTPUT_FILES,
                "private_cache_bytes_observed_maximum": MAX_CACHE_BYTES,
                "private_cache_files_observed_maximum": MAX_CACHE_FILES,
                "physical_density_trial_checks_maximum": 4,
                "instrument_estimator_invocations_maximum": 1,
                "odd_factory_distance_searches_maximum": 7,
                "native_supply_candidates_counted": False,
                "native_calls_interruptibility_certified": False},
            "observed_wall_seconds": time.monotonic() - started,
            "wait_derived_self_peak_rss_bytes": peak_rss_bytes(),
            "artifact_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        }
        write_json(output / "manifest.json", report)
        check()
        print(json.dumps({"status": report["status"], "mode": args.mode,
                          "complete_end_to_end_cost": False,
                          "compiled_fringe_check_executed": checks is not None,
                          "physical_resource_estimate_computed": report["physical_resource_estimate_computed"],
                          "retry": False, "wall_seconds": time.monotonic() - started}))
        return 2 if unavailable else 0
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
