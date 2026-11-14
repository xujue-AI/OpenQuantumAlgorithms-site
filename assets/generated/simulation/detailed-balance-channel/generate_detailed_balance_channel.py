#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded original finite zero-C weak channel and repeated-circuit export."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ADJACENT = (HERE / "detailed_balance_channel.py").is_file()
SOURCE_DIR = HERE if ADJACENT else ROOT / "src/openquantum/simulation"
sys.path.insert(0, str(SOURCE_DIR))
import detailed_balance_channel as dbc
import detailed_balance_access as dba
if Path(dbc.__file__).resolve() != (SOURCE_DIR / "detailed_balance_channel.py").resolve():
    raise ValueError("channel module differs from selected source snapshot")
if Path(dba.__file__).resolve() != (SOURCE_DIR / "detailed_balance_access.py").resolve():
    raise ValueError("access module differs from selected source snapshot")
from qiskit import qasm3

MAX_OUTPUT_BYTES = 8 * 1024 * 1024


def model(n_system):
    n = dba._integer(n_system, "n_system", 1, dba.MAX_SOURCE_SYSTEM_WIRES)
    if n == 1:
        return ((1., "Z"),), ("X", "Z")
    # A scalable source family, with no claimed rapid-mixing coverage. The
    # endpoint pair of X proposals does not connect all basis states for n>2.
    def label(q, p):
        return "I" * q + p + "I" * (n - q - 1)
    terms = tuple((.4 + .01 * q, label(q, "Z")) for q in range(n))
    proposals = (label(0, "X"), label(0, "Z"), label(n - 1, "X"), label(n - 1, "Z"))
    return terms, proposals


def _git_environment():
    env = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                 "GIT_NAMESPACE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"):
        env.pop(name, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _revision(paths):
    if ADJACENT or not (ROOT / ".git").exists():
        return None, None
    try:
        probe = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel", "HEAD"],
                               check=True, capture_output=True, text=True, timeout=5, env=_git_environment())
        directory, revision = probe.stdout.splitlines()
        if Path(directory).resolve() != ROOT.resolve():
            return None, None
        for path in paths:
            saved = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:" + str(path.relative_to(ROOT))],
                                   check=True, capture_output=True, timeout=5, env=_git_environment()).stdout
            if saved != path.read_bytes():
                return revision, True
        return revision, False
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def _write(output, name, raw, inventory):
    if sum(x["bytes"] for x in inventory.values()) + len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("weak-channel export exceeds8MiB")
    (output / name).write_bytes(raw)
    inventory[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def generate(output: Path, *, n_system=1, clock_bits=2, beta=1., delta=.125,
             circuit_rounds=8, check_rounds=64, simulate=True, grid=False):
    if not isinstance(simulate, bool) or not isinstance(grid, bool):
        raise ValueError("simulate/grid must be boolean")
    if grid and not simulate:
        raise ValueError("grid requires bounded simulation")
    dba._integer(check_rounds, "check_rounds", 1, dbc.MAX_SOURCE_ROUNDS)
    terms, proposals = model(n_system)
    if simulate and n_system != 1:
        raise ValueError("numerical endpoint checks are capped at one system wire; use --no-sim")
    if output.exists():
        raise ValueError("export requires an absent output directory; existing records are preserved")
    cases = [(clock_bits, beta, delta)]
    if grid:
        cases = [(m, 1., step) for m in (2, 3, 4) for step in (.0625, .125, .25)]
        cases += [(3, temperature, .125) for temperature in (.5, 2.)]
    # All source dimensions and preparation counts are preflighted before the
    # output directory is created. Larger sources do not simulate matrices.
    preflight = []
    for m, temperature, step in cases:
        dilation = dbc.build_weak_dilation(terms, proposals, clock_bits=m, beta=temperature, delta=step)
        round_circuit = dbc.build_weak_round(terms, proposals, clock_bits=m, beta=temperature, delta=step)
        preparation = dbc.build_preparation(terms, proposals, clock_bits=m, beta=temperature,
                                            delta=step, rounds=circuit_rounds)
        preflight.append((m, temperature, step, dilation, round_circuit, preparation))
    output.mkdir(parents=True)
    inventory, records = {}, []
    start = time.perf_counter()
    for m, temperature, step, dilation, round_circuit, preparation in preflight:
        name = f"n{n_system}-m{m}-beta{temperature:g}-delta{step:g}"
        for label, circuit in (("dilation", dilation), ("reset-round", round_circuit), ("preparation", preparation)):
            _write(output, f"{name}-{label}.qasm", qasm3.dumps(circuit).encode(), inventory)
        record = {"name": name, "terms": terms, "proposals": proposals,
                  "dilation": dilation.metadata, "reset_round": round_circuit.metadata,
                  "preparation": preparation.metadata,
                  "preparation_primitive_counts": dict(preparation.count_ops()),
                  "preparation_depth": preparation.depth(),
                  "numerical_checks_performed": simulate}
        if simulate:
            checked = dbc.channel_check(terms, proposals, clock_bits=m, beta=temperature,
                                        delta=step, rounds=check_rounds)
            for key in ("actual_vs_independent_kraus_max_entry_error",
                        "actual_vs_independent_channel_frobenius_error",
                        "trace_preservation_max_entry_error", "continuum_generator_gibbs_residual",
                        "decay_commutator"):
                if checked[key] > 1e-10:
                    raise AssertionError(f"bounded channel check failed: {key}")
            if checked["minimum_choi_eigenvalue"] < -1e-10:
                raise AssertionError("channel Choi positivity check failed")
            # Nonzero finite Gibbs bias is a retained result, not an acceptance failure.
            record["check"] = checked
        epsilon, failure = .1, .05
        draws = math.ceil(2 * math.log(2 / failure) / epsilon ** 2)
        record["logical_readout_contract"] = {
            "observable": "Z on q0 after the finite circuit's explicit K rounds",
            "mean_additive_sampling_error": epsilon, "sampling_failure_probability": failure,
            "independent_complete_preparations": draws,
            "hoeffding_formula": "ceil(2*ln(2/failure)/epsilon^2), Z in[-1,1]",
            "total_primitive_counts": {k: int(v) * draws for k, v in preparation.count_ops().items()},
            "total_rounds": draws * circuit_rounds,
            "synthesis_and_physical_faults_included": False,
            "continuum_gibbs_accuracy_certificate": False,
            "exclusions": ["arbitrary rotation synthesis", "physical code/factory/reset/controller model",
                           "general mixing and finite mesh certificate", "classical preprocessing timing"]}
        records.append(record)
    sources = [SOURCE_DIR / "detailed_balance_channel.py", SOURCE_DIR / "detailed_balance_access.py", Path(__file__).resolve()]
    license_file = HERE / "LICENSE.txt" if (HERE / "LICENSE.txt").is_file() else ROOT / "LICENSE"
    lock_file = HERE / "requirements-lock.txt" if (HERE / "requirements-lock.txt").is_file() else ROOT / "requirements-lock.txt"
    for source in sources:
        _write(output, source.name, source.read_bytes(), inventory)
    for name, path in (("LICENSE.txt", license_file), ("requirements-lock.txt", lock_file)):
        if not path.is_file():
            raise ValueError("export needs the exact license and environment lock")
        _write(output, name, path.read_bytes(), inventory)
    revision, dirty = _revision([*sources, license_file, lock_file])
    report = {
        "status": "checked_small_finite_diagonal_channel" if simulate else "generated_diagonal_channel_without_simulation",
        "paper": {"doi": "10.1038/s41586-025-09583-x", "weak_round_equations": [28, 29],
                  "spectral_correction_equation": 19, "coherent_correction": "exactly zero for the declared diagonal-Pauli family"},
        "source_commit": revision, "source_dirty": dirty,
        "source_revision_is_exact": revision is not None and dirty is False,
        "environment": {"python": platform.python_version(), "qiskit": importlib.metadata.version("qiskit"),
                        "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy"),
                        "device": "CPU", "seed": None, "random_sampling_performed": False},
        "caps": {"source_rounds": dbc.MAX_SOURCE_ROUNDS, "primitive_operations_each": dba.MAX_PRIMITIVE_OPERATIONS,
                 "reference_system_wires": 1, "reference_clock_bits": 4, "reference_total_wires": 7,
                 "output_bytes": MAX_OUTPUT_BYTES},
        "source_files": {p.name: inventory[p.name] for p in sources},
        "environment_lock_sha256": inventory["requirements-lock.txt"]["sha256"],
        "wall_seconds": time.perf_counter() - start, "cases": records,
        "scope": "complete finite weak reset channel in a restricted zero-C family; all outcomes retained; channel-power checks separately labeled",
        "exact_finite_kms": False, "general_sampler_implemented": False,
        "rotations_synthesized": False, "physical_resources_computed": False,
        "formal_proof_reproduced": False}
    _write(output, "results.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(), inventory)
    manifest = {"status": report["status"], "files": dict(inventory),
                "total_payload_bytes": sum(v["bytes"] for v in inventory.values()),
                "general_sampler_implemented": False, "exact_finite_kms": False}
    _write(output, "manifest.json", (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(), inventory)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-system", type=int, default=1)
    parser.add_argument("--clock-bits", type=int, default=2)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--delta", type=float, default=.125)
    parser.add_argument("--circuit-rounds", type=int, default=8)
    parser.add_argument("--check-rounds", type=int, default=64)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    args = parser.parse_args()
    report = generate(args.output_dir, n_system=args.n_system, clock_bits=args.clock_bits,
                      beta=args.beta, delta=args.delta, circuit_rounds=args.circuit_rounds,
                      check_rounds=args.check_rounds, grid=args.grid, simulate=not args.no_sim)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "general_sampler_implemented": False, "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
