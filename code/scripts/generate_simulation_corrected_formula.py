#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Unexecuted bounded producer prototype for disjoint-pair symplectic CPF2.

The default suite has 32 frozen tiny cases. Larger exports require an explicit
pair count and --generation-only. All fresh runs require their own directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import qasm3
from openquantum.simulation.corrected_product_formula import (
    PairModel, METHODS, canonical_cases, compare_case, complete_trial,
    logical_ledger, negative_control_contrasts, request,
)

MAX_CASES = 32
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_FILES = 128
MAX_PRODUCER_WALL_SECONDS = 90
REFERENCE_TOLERANCE = 1e-11
SOURCE_FILES = (
    "src/openquantum/simulation/corrected_product_formula.py",
    "scripts/generate_simulation_corrected_formula.py",
    "tests/test_simulation_corrected_formula.py",
    "notebooks/simulation_corrected_formula.ipynb",
    "requirements-lock.txt", "LICENSE",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def output_directory(path):
    destination = Path(path).absolute()
    current = destination
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("output path may not contain a symlink")
        current = current.parent
    destination = destination.resolve()
    if destination == ROOT or destination.exists():
        raise ValueError("output must be a new dedicated directory")
    for folder in (".git", "src", "scripts", "tests", "notebooks", "algorithms", "docs", "references", "assets"):
        protected = ROOT / folder
        if destination == protected or protected in destination.parents:
            raise ValueError("prototype output cannot overwrite project source or historical assets")
    return destination


def committed_sources():
    snapshots = {name: (ROOT / name).read_bytes() for name in SOURCE_FILES}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise ValueError("commit the frozen source before scientific generation")
    commit = result.stdout.strip()
    for name, content in snapshots.items():
        old = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT,
                             capture_output=True, timeout=5)
        if old.returncode or old.stdout != content:
            raise ValueError(f"uncommitted or changed source: {name}")
    return commit, snapshots


class Writer:
    def __init__(self, directory):
        self.directory = directory
        self.files = []
        self.used = 0

    def write(self, name, content):
        payload = content.encode("utf8") if isinstance(content, str) else content
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("payload path must be a safe relative file")
        if len(self.files) >= MAX_OUTPUT_FILES or self.used + len(payload) > MAX_OUTPUT_BYTES - 65536:
            raise ValueError("output reservation exceeded; no completion manifest")
        path = self.directory / relative
        if path.exists():
            raise ValueError("producer must not replace an existing payload")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.files.append({"path": relative.as_posix(), "bytes": len(payload), "sha256": digest(payload)})
        self.used += len(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--alpha")
    parser.add_argument("--a")
    parser.add_argument("--b")
    parser.add_argument("--time")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.suite and (args.pairs is not None or args.generation_only):
        parser.error("--suite has a fixed tiny case vector and cannot be changed")
    if args.suite and any(value is not None for value in (args.alpha, args.a, args.b, args.time, args.repetitions)):
        parser.error("--suite rejects parameter overrides; use a separate single-case command")
    if args.generation_only and args.pairs is None:
        parser.error("--generation-only requires an explicit pair count")
    pairs = args.pairs if args.pairs is not None else 1
    cases = canonical_cases() if args.suite else ((PairModel(pairs,
                                                 args.a if args.a is not None else "1",
                                                 args.b if args.b is not None else "1",
                                                 args.alpha if args.alpha is not None else "1/5"),
                                                 args.time if args.time is not None else "1",
                                                 args.repetitions if args.repetitions is not None else 4),)
    if len(cases) > MAX_CASES:
        parser.error("case count exceeds the fixed cap")
    for model, duration, repetitions in cases:
        for method in METHODS:
            request(model, duration, repetitions, method, dense=not args.generation_only)
    if sys.version_info[:3] != (3, 12, 14) or importlib.metadata.version("qiskit") != "2.6.2":
        raise ValueError("the frozen producer requires Python3.12.14 and locked Qiskit2.6.2")
    commit, sources = committed_sources()
    output = output_directory(args.output_dir)
    output.mkdir(parents=True)
    writer = Writer(output)
    started = time.monotonic()
    rows = []
    for index, (model, duration, repetitions) in enumerate(cases):
        if time.monotonic() - started > MAX_PRODUCER_WALL_SECONDS:
            raise TimeoutError("cooperative producer deadline exceeded")
        row = {"case_index": index, "model": model.as_dict(), "time": str(duration),
               "repetitions": repetitions, "simulation_performed": not args.generation_only,
               "ledgers": {method: logical_ledger(model, duration, repetitions, method) for method in METHODS},
               "comparison": None if args.generation_only else compare_case(model, duration, repetitions)}
        if row["comparison"] is not None:
            comparison = row["comparison"]
            residual = comparison["independent_expm_max_entry_discrepancy"]
            if not math.isfinite(residual) or residual > REFERENCE_TOLERANCE:
                raise ValueError("independent exact-dynamics references disagree")
            for checked in comparison["methods"].values():
                for name in ("operator_formula_max_entry_discrepancy", "state_preparation_max_entry_discrepancy", "reset_density_max_entry_discrepancy"):
                    if not math.isfinite(checked[name]) or checked[name] > REFERENCE_TOLERANCE:
                        raise ValueError(f"independent check failed: {name}")
        directory = f"case-{index:02d}"
        for method in METHODS:
            writer.write(f"{directory}/{method}.qasm", qasm3.dumps(complete_trial(model, duration, repetitions, method)))
        writer.write(f"{directory}/record.json", json.dumps(row, indent=2, allow_nan=False) + "\n")
        rows.append(row)
    for name, content in sources.items():
        writer.write(f"source/{name}", content)
    if args.suite:
        controls = negative_control_contrasts()
        if not math.isfinite(controls["correct_formula_max_entry_discrepancy"]) or controls["correct_formula_max_entry_discrepancy"] > REFERENCE_TOLERANCE:
            raise ValueError("correct formula failed the frozen negative-control reference")
        for name in ("missing_boundaries_max_entry_contrast", "wrong_sign_boundaries_max_entry_contrast"):
            if not math.isfinite(controls[name]) or controls[name] <= 1e-6:
                raise ValueError("frozen noncommuting negative control was not discriminating")
        writer.write("negative-controls.json", json.dumps(controls, indent=2, allow_nan=False) + "\n")
    if any((ROOT / name).read_bytes() != content for name, content in sources.items()):
        raise ValueError("scientific source changed during generation")
    if time.monotonic() - started > MAX_PRODUCER_WALL_SECONDS:
        raise TimeoutError("cooperative producer deadline exceeded before completion")
    study = {"source_commit": commit, "cases": rows,
             "wall_seconds": time.monotonic()-started,
             "physical_estimation_performed": False, "sampling_performed": False,
             "source_only_generation_scope": bool(args.generation_only),
             "scope": "Given first symplectic CPF2 for disjoint ZZ/X pairs. No general fourth-order, algorithm advantage, synthesis or hardware certificate.",
             "limits": {"cases": MAX_CASES, "output_bytes": MAX_OUTPUT_BYTES,
                        "output_files": MAX_OUTPUT_FILES, "producer_cooperative_wall_seconds": MAX_PRODUCER_WALL_SECONDS}}
    writer.write("study.json", json.dumps(study, indent=2) + "\n")
    manifest = {"status": "complete", "source_commit": commit,
                "source_sha256": {name: digest(content) for name, content in sources.items()},
                "files": writer.files, "payload_bytes": writer.used,
                "scientific_scope": study["scope"],
                "limits_qualification": "Cooperative checkpoints and payload reservation; a reviewed outer process/kernel controller supplies run wall and sampled RSS limits."}
    writer.write("manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "cases": len(rows), "payload_bytes": writer.used,
                      "source_commit": commit, "simulation_performed": not args.generation_only}))


if __name__ == "__main__":
    main()
