#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded primitive Gaussian access export; no channel/sampler execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE = HERE / "detailed_balance_access.py"
ADJACENT_SNAPSHOT = MODULE.is_file()
if ADJACENT_SNAPSHOT:
    # An exported snapshot wins over any newer parent checkout.
    sys.path.insert(0, str(HERE))
    import detailed_balance_access as dba
else:
    MODULE = ROOT / "src/openquantum/simulation/detailed_balance_access.py"
    sys.path.insert(0, str(ROOT / "src"))
    from openquantum.simulation import detailed_balance_access as dba
if Path(dba.__file__).resolve() != MODULE.resolve():
    raise ValueError("imported access module differs from the selected source snapshot")
from qiskit import qasm3

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
ONE = ((1.0, "Z"),)
TWO = ((.7, "ZI"), (.4, "XX"), (.3, "IX"), (.19, "YZ"), (.23, "IZ"))
TWO_PROPOSALS = ("XI", "ZI", "IX", "IZ")


def _label(n, operators):
    result = ["I"] * n
    for q, p in operators:
        result[q] = p
    return "".join(result)


def model(n):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= dba.MAX_SOURCE_SYSTEM_WIRES:
        raise ValueError("n_system must be an integer from 1 to 16")
    if n == 1:
        return ONE, ("X", "Z")
    if n == 2:
        return TWO, TWO_PROPOSALS
    terms = tuple((.4 + .01 * q, _label(n, [(q, "Z")])) for q in range(n)) + tuple(
        (.2, _label(n, [(q, "X"), (q + 1, "X")])) for q in range(n - 1))
    proposals = tuple(_label(n, [(q, p)]) for q, p in ((0, "X"), (0, "Z"), (n - 1, "X"), (n - 1, "Z")))
    return terms, proposals


def _write(path: Path, raw: bytes, inventory: dict) -> None:
    if sum(v["bytes"] for v in inventory.values()) + len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("export exceeds the 64 MiB output cap")
    path.write_bytes(raw)
    inventory[path.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _source_commit():
    """Optional provenance only when this exact directory is the Git root."""
    if ADJACENT_SNAPSHOT:
        return None
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                                check=True, capture_output=True, text=True, timeout=5)
        if Path(result.stdout.strip()).resolve() != ROOT.resolve():
            return None
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_dirty(input_paths):
    try:
        for path in (MODULE, Path(__file__), *input_paths):
            if not path.is_file():
                continue
            relative = str(path.resolve().relative_to(ROOT.resolve()))
            saved = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:" + relative],
                                   check=True, capture_output=True, timeout=5).stdout
            if saved != path.read_bytes():
                return True
        return False
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def generate(output: Path, *, n_system=1, clock_bits=2, product_steps=1,
             beta=1., simulate=True, grid=False):
    if grid and not simulate:
        raise ValueError("the bounded check grid requires simulation")
    terms, proposals = model(n_system)
    if simulate and n_system > dba.MAX_DENSE_SYSTEM_WIRES:
        raise ValueError("simulation requires one or two system wires; use --no-sim for a larger source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("export needs a fresh or empty output directory; old checked records are not relabeled")
    output.mkdir(parents=True, exist_ok=True)
    inventory = {}
    start = time.perf_counter()
    cases = [(n_system, clock_bits, product_steps, beta)]
    if grid:
        cases = [(1, m, 1, 1.) for m in (2, 3, 4)] + [(2, m, r, 1.) for m in (2, 3) for r in (1, 2, 4)]
    records = []
    for n, m, r, temperature in cases:
        inputs, jumps = model(n)
        circuit = dba.build_gaussian_access(inputs, jumps, beta=temperature,
                                           clock_bits=m, product_steps=r)
        label = f"n{n}-m{m}-r{r}"
        _write(output / f"{label}-access.qasm", qasm3.dumps(circuit).encode(), inventory)
        record = {"name": label, "terms": inputs, "proposals": jumps,
                  "metadata": circuit.metadata, "primitive_counts": dict(circuit.count_ops()),
                  "numerical_checks_performed": bool(simulate)}
        if simulate:
            record["check"] = dba.access_check(inputs, jumps, beta=temperature,
                                              clock_bits=m, product_steps=r)
            if max(record["check"]["actual_vs_finite_trotter_max_entry_error"],
                   record["check"]["isometry_max_entry_error"],
                   record["check"]["continuum_reference"]["gibbs_stationarity_residual"]) > 1e-10:
                raise AssertionError("small circuit/reference checks failed")
        records.append(record)
    for source in (MODULE, Path(__file__).resolve()):
        _write(output / source.name, source.read_bytes(), inventory)
    license_file = HERE / "LICENSE.txt" if (HERE / "LICENSE.txt").is_file() else ROOT / "LICENSE"
    if license_file.is_file():
        _write(output / "LICENSE.txt", license_file.read_bytes(), inventory)
    lock_file = HERE / "requirements-lock.txt" if (HERE / "requirements-lock.txt").is_file() else ROOT / "requirements-lock.txt"
    if lock_file.is_file():
        _write(output / "requirements-lock.txt", lock_file.read_bytes(), inventory)
    source_commit = _source_commit()
    source_dirty = _source_dirty((license_file, lock_file)) if source_commit is not None else None
    report = {"status": "checked_small_gaussian_access_only" if simulate else "generated_gaussian_access_without_simulation",
              "paper": {"doi": "10.1038/s41586-025-09583-x", "approximated_equations": [6, 13, 25, 26, 27],
                        "coherent_reference_equation": 19, "coherent_circuit_implemented": False},
              "environment": {"python": platform.python_version(), "qiskit": importlib.metadata.version("qiskit"),
                              "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy"),
                              "seed": None, "device": "CPU", "random_sampling_performed": False},
              "source_commit": source_commit, "source_dirty": source_dirty,
              "source_revision_is_exact": source_commit is not None and source_dirty is False,
              "source_files": {name: value for name, value in inventory.items() if name.endswith(".py")},
              "environment_lock_sha256": inventory.get("requirements-lock.txt", {}).get("sha256"),
              "caps": {"source_operations_each": dba.MAX_PRIMITIVE_OPERATIONS,
                       "simulation_total_wires": dba.MAX_SIMULATION_WIRES, "dense_system_wires": dba.MAX_DENSE_SYSTEM_WIRES,
                       "output_bytes": MAX_OUTPUT_BYTES}, "wall_seconds": time.perf_counter() - start,
              "cases": records, "sampler_implemented": False, "channel_implemented": False,
              "physical_estimation_performed": False, "rotations_synthesized": False,
              "scope": "normalized primitive access and bounded classical references; no exact finite KMS or mixing claim",
              "remaining_endpoint": ["conditional weak uncompute and complete channel/reset instrument",
                                     "actual coherent-correction access/evolution where required",
                                     "uniform mesh/channel/synthesis error budget and mixing/output contract",
                                     "complete preparation/repetition/readout resources and physical baseline"]}
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    _write(output / "results.json", raw, inventory)
    manifest = {"status": report["status"], "files": inventory,
                "source_hash": inventory[MODULE.name]["sha256"],
                "sampler_implemented": False, "channel_implemented": False,
                "total_payload_bytes": sum(v["bytes"] for v in inventory.values())}
    _write(output / "manifest.json", (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(), inventory)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-system", type=int, default=1)
    parser.add_argument("--clock-bits", type=int, default=2)
    parser.add_argument("--product-steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.output_dir, n_system=args.n_system, clock_bits=args.clock_bits,
                      product_steps=args.product_steps, beta=args.beta, simulate=not args.no_sim, grid=args.grid)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]),
                      "sampler_implemented": False, "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
