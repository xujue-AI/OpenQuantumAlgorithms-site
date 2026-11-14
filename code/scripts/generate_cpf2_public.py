#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepared, unexecuted standalone wrapper for the immutable CPF2 module.

Small numerical checks and larger generation have separate source bounds.
This adaptation has not executed; saved evidence belongs to the original
repository producer and original notebook, not to this wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

MODULE_SHA256 = "63fffc29105a2162ba5d0b37f8638738dad00161065734099b43654089839540"
LOCK_SHA256 = "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"
LICENSE_SHA256 = "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_CASES = 32
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_FILES = 128
MAX_PRODUCER_WALL_SECONDS = 90
REFERENCE_TOLERANCE = 1e-11
RUNTIME_PINS = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def packet_directory():
    here = Path(__file__).resolve().parent
    if (here / "corrected_product_formula.py").is_file():
        return here
    packet = here.parent / "assets/generated/simulation/corrected-formula"
    if not (packet / "corrected_product_formula.py").is_file():
        raise ValueError("download the complete public CPF2 packet beside this wrapper")
    return packet


def load_science(packet=None):
    """Authenticate local source before a normal standalone module import."""
    directory = packet_directory() if packet is None else Path(packet).resolve()
    pins = {"corrected_product_formula.py": MODULE_SHA256,
            "requirements-lock.txt": LOCK_SHA256, "LICENSE.txt": LICENSE_SHA256}
    sources = {}
    for name, expected in pins.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("public inputs must be local regular files")
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES or digest(raw) != expected:
            raise ValueError(f"public input differs from its frozen original bytes: {name}")
        sources[name] = raw
    if sys.version_info[:3] != (3, 12, 14):
        raise ValueError("use the recorded Python 3.12.14 environment")
    for name, expected in RUNTIME_PINS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"runtime package differs from the recorded lock: {name}")
    module_path = directory / "corrected_product_formula.py"
    spec = importlib.util.spec_from_file_location("_openquantum_cpf2_public", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the authenticated standalone CPF2 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, directory, sources


def check_comparison(comparison):
    residual = comparison["independent_expm_max_entry_discrepancy"]
    if not math.isfinite(residual) or residual > REFERENCE_TOLERANCE:
        raise ValueError("independent exact-dynamics references disagree")
    for checked in comparison["methods"].values():
        for name in ("operator_formula_max_entry_discrepancy",
                     "state_preparation_max_entry_discrepancy",
                     "reset_density_max_entry_discrepancy"):
            if not math.isfinite(checked[name]) or checked[name] > REFERENCE_TOLERANCE:
                raise ValueError(f"independent check failed: {name}")


class Writer:
    def __init__(self, directory):
        self.directory, self.files, self.used = directory, [], 0

    def write(self, name, content):
        raw = content.encode() if isinstance(content, str) else content
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("payload path must be a safe relative file")
        if len(self.files) >= MAX_OUTPUT_FILES or self.used + len(raw) > MAX_OUTPUT_BYTES - 65536:
            raise ValueError("bounded output reservation exceeded")
        path = self.directory / relative
        if path.exists() or path.is_symlink():
            raise ValueError("generation cannot replace an existing payload")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        self.files.append({"path": relative.as_posix(), "bytes": len(raw), "sha256": digest(raw)})
        self.used += len(raw)


def fresh_output(path, packet):
    candidate = Path(path).absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("output path cannot traverse a symlink")
    candidate = candidate.resolve()
    if candidate.exists() or candidate == packet or packet in candidate.parents:
        raise ValueError("use a new output directory outside the immutable public packet")
    if packet in candidate.parents or candidate in packet.parents:
        raise ValueError("output cannot replace the packet or an ancestor")
    candidate.mkdir(parents=True)
    return candidate


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    overrides = (args.pairs, args.alpha, args.a, args.b, args.time, args.repetitions)
    if args.suite and (args.generation_only or any(value is not None for value in overrides)):
        parser.error("--suite retains the original 32 cases and rejects all overrides")
    if args.generation_only and args.pairs is None:
        parser.error("--generation-only requires an explicit pair count")
    cpf, packet, sources = load_science()
    pairs = args.pairs if args.pairs is not None else 1
    cases = cpf.canonical_cases() if args.suite else ((
        cpf.PairModel(pairs, args.a if args.a is not None else "1",
                      args.b if args.b is not None else "1",
                      args.alpha if args.alpha is not None else "1/5"),
        args.time if args.time is not None else "1",
        args.repetitions if args.repetitions is not None else 4),)
    if len(cases) > MAX_CASES:
        raise ValueError("case count exceeds the original bounded vector")
    for model, duration, repetitions in cases:
        for method in cpf.METHODS:
            cpf.request(model, duration, repetitions, method, dense=not args.generation_only)
    from qiskit import qasm3
    output = fresh_output(args.output_dir, packet)
    writer, rows = Writer(output), []
    started = time.monotonic()
    for index, (model, duration, repetitions) in enumerate(cases):
        if time.monotonic() - started > MAX_PRODUCER_WALL_SECONDS:
            raise TimeoutError("cooperative producer deadline exceeded")
        comparison = None if args.generation_only else cpf.compare_case(model, duration, repetitions)
        if comparison is not None:
            check_comparison(comparison)
        row = {"case_index": index, "model": model.as_dict(), "time": str(duration),
               "repetitions": repetitions, "simulation_performed": not args.generation_only,
               "ledgers": {method: cpf.logical_ledger(model, duration, repetitions, method)
                           for method in cpf.METHODS}, "comparison": comparison}
        for method in cpf.METHODS:
            writer.write(f"case-{index:02d}/{method}.qasm",
                         qasm3.dumps(cpf.complete_trial(model, duration, repetitions, method)))
        writer.write(f"case-{index:02d}/record.json", json.dumps(row, indent=2, allow_nan=False) + "\n")
        rows.append(row)
    if args.suite:
        controls = cpf.negative_control_contrasts()
        if not math.isfinite(controls["correct_formula_max_entry_discrepancy"]) or controls["correct_formula_max_entry_discrepancy"] > REFERENCE_TOLERANCE:
            raise ValueError("correct formula failed its independent negative-control reference")
        for name in ("missing_boundaries_max_entry_contrast", "wrong_sign_boundaries_max_entry_contrast"):
            if not math.isfinite(controls[name]) or controls[name] <= 1e-6:
                raise ValueError("original noncommuting negative control is not discriminating")
        writer.write("negative-controls.json", json.dumps(controls, indent=2, allow_nan=False) + "\n")
    for name, original in sources.items():
        if (packet / name).read_bytes() != original:
            raise ValueError("immutable public source changed during generation")
    if time.monotonic() - started > MAX_PRODUCER_WALL_SECONDS:
        raise TimeoutError("cooperative deadline exceeded before completion")
    study = {"schema": "openquantum.cpf2.public-run.v1", "cases": rows,
             "scientific_module_sha256": MODULE_SHA256, "wrapper_sha256": digest(Path(__file__).read_bytes()),
             "runtime_versions": dict(RUNTIME_PINS), "physical_estimation_performed": False,
             "sampling_performed": False, "generation_only": bool(args.generation_only),
             "scope": "Disjoint-pair CPF2 component; no generic fourth-order, application-advantage, synthesis or hardware certificate."}
    writer.write("study.json", json.dumps(study, indent=2, allow_nan=False) + "\n")
    manifest = {"schema": "openquantum.cpf2.public-output.v1", "status": "complete",
                "scientific_module_sha256": MODULE_SHA256, "source_hashes": {name: digest(raw) for name, raw in sources.items()},
                "files": writer.files, "payload_bytes": writer.used,
                "bounds_qualification": "Cooperative 90-second generation and finite source/output caps; no hard RSS or process containment guarantee."}
    writer.write("manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "cases": len(rows),
                      "simulation_performed": not args.generation_only, "payload_bytes": writer.used}))


if __name__ == "__main__":
    main()
