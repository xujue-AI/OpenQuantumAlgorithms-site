#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export original primitive finite weighted-decay block access.

Numerical mode is restricted to the three frozen beta=1, m=2, r=1 cases.
Source-only mode never imports the new dense reference or calls a simulator.
This is a block oracle, not coherent-correction evolution or a Gibbs sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
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
SOURCE_NAMES = (
    "detailed_balance_decay.py",
    "detailed_balance_decay_reference.py",
    "detailed_balance_access.py",
)
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES_EACH = 2 * 1024 * 1024
NUMERICAL_TOLERANCE = 1e-10
TINY_CASES = {
    "diagonal": (((1.0, "Z"),), ("X", "Z")),
    "tilted": (((0.6, "Z"), (0.8, "X")), ("Z",)),
    "complex": (
        ((0.7, "ZI"), (0.4, "XX"), (0.3, "IX"), (0.19, "YZ"), (0.23, "IZ")),
        ("XI", "ZI", "IX", "IZ"),
    ),
}
EXPECTED_TINY_WIRES = {"diagonal": 6, "tilted": 5, "complex": 8}
SCOPE_FLAGS = {
    "all_components_verified": False,
    "new_algorithm_result": False,
    "coherent_correction_implemented": False,
    "coherent_correction_evolution_implemented": False,
    "general_channel_implemented": False,
    "sampler_implemented": False,
    "exact_finite_kms": False,
    "mixing_certificate": False,
    "rotations_synthesized": False,
    "physical_resources_computed": False,
    "formal_proof_reproduced": False,
    "random_sampling_performed": False,
}


def _source_directory() -> tuple[Path, bool]:
    """An adjacent exported snapshot wins; incomplete snapshots fail closed."""
    adjacent = (HERE / SOURCE_NAMES[0]).is_file()
    directory = HERE if adjacent else ROOT / "src/openquantum/simulation"
    for name in SOURCE_NAMES:
        source = directory / name
        if not source.is_file() or source.is_symlink():
            raise ValueError("export needs regular exact source inputs")
        if source.stat().st_size > MAX_SOURCE_BYTES_EACH:
            raise ValueError("a source input exceeds the bounded source-read cap")
    return directory, adjacent


def _load_module(directory: Path, name: str):
    """Resolve the chosen source before accepting its target-module import."""
    sys.path.insert(0, str(directory))
    module = importlib.import_module(name)
    expected = (directory / (name + ".py")).resolve()
    if Path(module.__file__).resolve() != expected:
        raise ValueError("imported module differs from the selected source snapshot")
    return module


def _source_inputs(directory: Path, adjacent: bool) -> list[tuple[str, Path]]:
    license_file = HERE / "LICENSE.txt" if adjacent else ROOT / "LICENSE"
    lock_file = HERE / "requirements-lock.txt" if adjacent else ROOT / "requirements-lock.txt"
    paths = [(name, directory / name) for name in SOURCE_NAMES]
    paths += [
        (Path(__file__).name, Path(__file__).resolve()),
        ("LICENSE.txt", license_file),
        ("requirements-lock.txt", lock_file),
    ]
    for _, path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError("source export requires the exact regular sources, lock and license")
        if path.stat().st_size > MAX_SOURCE_BYTES_EACH:
            raise ValueError("a source input exceeds the bounded source-read cap")
    return paths


def _git_environment():
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _revision(paths, adjacent):
    """A copied-source replay never borrows provenance from a parent checkout."""
    if adjacent or not (ROOT / ".git").exists():
        return None, None
    try:
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5, env=_git_environment(),
        )
        directory, revision = probe.stdout.splitlines()
        if Path(directory).resolve() != ROOT.resolve():
            return None, None
        for _, path in paths:
            saved = subprocess.run(
                ["git", "-C", str(ROOT), "show", "HEAD:" + str(path.relative_to(ROOT))],
                check=True, capture_output=True, timeout=5, env=_git_environment(),
            ).stdout
            if saved != path.read_bytes():
                return revision, True
        return revision, False
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def _label(n, operators):
    characters = ["I"] * n
    for wire, pauli in operators:
        characters[wire] = pauli
    return "".join(characters)


def chain_model(n_system):
    """Parameterized source family; no dense check or convergence promise."""
    if isinstance(n_system, bool) or not isinstance(n_system, int) or not 3 <= n_system <= 16:
        raise ValueError("the chain source family requires an integer n_system in [3,16]")
    n = n_system
    terms = tuple(
        (1.0, _label(n, ((q, "Z"), (q + 1, "Z")))) for q in range(n - 1)
    ) + tuple((0.7, _label(n, ((q, "X"),))) for q in range(n))
    proposals = tuple(
        _label(n, ((q, pauli),)) for q, pauli in ((0, "X"), (0, "Z"), (n - 1, "X"), (n - 1, "Z"))
    )
    return terms, proposals


def _parameters(case, tiny_grid, n_system, beta, clock_bits, product_steps, simulate):
    if not isinstance(tiny_grid, bool) or not isinstance(simulate, bool):
        raise ValueError("tiny_grid and simulate must be booleans")
    if case is not None and case not in TINY_CASES:
        raise ValueError("case must be diagonal, tilted or complex")
    if sum((case is not None, tiny_grid, n_system is not None)) > 1:
        raise ValueError("choose one case, the tiny grid or the chain source family")
    if n_system is not None:
        if simulate:
            raise ValueError("chain source export requires --no-sim")
        terms, proposals = chain_model(n_system)
        return [(f"chain-n{n_system}", terms, proposals)]
    if tiny_grid and not simulate:
        raise ValueError("the frozen tiny grid requires numerical mode")
    if simulate and (beta != 1.0 or clock_bits != 2 or product_steps != 1):
        raise ValueError("numerical export is restricted to beta=1, clock_bits=2, product_steps=1")
    names = list(TINY_CASES) if tiny_grid else [case or "diagonal"]
    return [(name, *TINY_CASES[name]) for name in names]


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _write(output, name, raw, inventory):
    if not isinstance(raw, bytes) or "/" in name or "\\" in name or name in ("", ".", ".."):
        raise ValueError("exports use bounded regular flat-file payloads")
    if name in inventory:
        raise ValueError("an output payload name was repeated")
    if sum(pin["bytes"] for pin in inventory.values()) + len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("weighted-decay export exceeds the 64 MiB total output cap")
    (output / name).write_bytes(raw)
    inventory[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _check_acceptance(check):
    """Fixed tolerances authenticate this circuit/reference boundary only."""
    if (check.get("checks_passed") is not True or check.get("failed_checks") != []
            or check.get("acceptance_tolerance") != NUMERICAL_TOLERANCE):
        raise AssertionError("the independent decay-reference acceptance predicates failed")
    fields = [
        check["actual_vs_finite_trotter_max_entry_error"],
        check["actual_vs_finite_trotter_frobenius_error"],
        check["normalization"]["eta_absolute_error"],
        check["normalization"]["alpha_absolute_error"],
        check["normalization"]["dt_dw_residual"],
        check["finite_diagnostics"]["isometry_max_entry_error"],
        check["finite_diagnostics"]["decay_hermiticity_residual"],
        check["finite_diagnostics"]["correction_hermiticity_residual"],
        check["actual_diagnostics"]["hermiticity_residual"],
        check["actual_diagnostics"]["maximum_eigenvalue_imaginary_part"],
    ]
    if any(not math.isfinite(value) or value > NUMERICAL_TOLERANCE for value in fields):
        raise AssertionError("the bounded decay-block circuit/reference check failed")
    if check["finite_diagnostics"]["minimum_decay_eigenvalue"] < -NUMERICAL_TOLERANCE:
        raise AssertionError("the finite decay positivity diagnostic exceeded its fixed tolerance")
    if check["finite_diagnostics"]["alpha_upper_bound_excess"] > NUMERICAL_TOLERANCE:
        raise AssertionError("the independent decay normalization bound failed")
    if check["actual_diagnostics"]["spectral_norm"] > 1.0 + NUMERICAL_TOLERANCE:
        raise AssertionError("the actual selected-block norm bound failed")
    # Exact-H differences, raw negative roundoff and Gibbs-related quantities
    # are retained as reported; they are not clipped or made acceptance targets.


def generate(output: Path, *, case=None, tiny_grid=False, n_system=None,
             beta=1.0, clock_bits=2, product_steps=1, simulate=True):
    start = time.perf_counter()
    if not isinstance(output, Path):
        raise ValueError("output must be a Path")
    if output.exists() or output.is_symlink():
        raise ValueError("export requires an absent output directory; saved records are preserved")
    cases = _parameters(case, tiny_grid, n_system, beta, clock_bits, product_steps, simulate)
    directory, adjacent = _source_directory()
    sources = _source_inputs(directory, adjacent)
    decay = _load_module(directory, "detailed_balance_decay")
    # Parameter/whole-oracle guards run before any circuit or output directory.
    plans = [
        (name, terms, proposals, decay.preflight_decay(
            terms, proposals, beta=beta, clock_bits=clock_bits, product_steps=product_steps
        )) for name, terms, proposals in cases
    ]
    if simulate and any(plan["n_system"] > 2 or plan["total_wires"] > 8 for _, _, _, plan in plans):
        raise ValueError("actual numerical recovery is capped at two system and eight total wires")
    reference = _load_module(directory, "detailed_balance_decay_reference") if simulate else None
    from qiskit import qasm3

    output.mkdir(parents=True, exist_ok=False)
    inventory, records = {}, []
    for name, terms, proposals, plan in plans:
        circuit = decay.build_decay_block(
            terms, proposals, beta=beta, clock_bits=clock_bits, product_steps=product_steps
        )
        if simulate and circuit.num_qubits != EXPECTED_TINY_WIRES[name]:
            raise AssertionError("the frozen tiny case has an unexpected actual source width")
        _write(output, name + "-decay-block.qasm", qasm3.dumps(circuit).encode(), inventory)
        record = {
            "name": name, "terms": terms, "proposals": proposals,
            "beta": beta, "clock_bits": clock_bits, "product_steps": product_steps,
            "metadata": circuit.metadata, "primitive_counts": dict(circuit.count_ops()),
            "source_depth": circuit.depth(), "numerical_checks_performed": simulate,
            "selected_auxiliary_state": "all_zero",
            "output_contract": "all-zero clock/proposal/weight/selector block equals D_fin/alpha",
            "standalone_initialization_performed": False,
            "measurement_or_sampling_performed": False,
            "logical_resource_scope": (
                "one complete unitary oracle, including both realized access copies and inverses; "
                "caller supplies system input and zero auxiliaries; no readout or amplification"
            ),
            **SCOPE_FLAGS,
        }
        if simulate:
            record["check"] = reference.decay_check(
                terms, proposals, beta=beta, clock_bits=clock_bits,
                product_steps=product_steps, circuit=circuit,
            )
            _check_acceptance(record["check"])
        records.append(record)
    for name, path in sources:
        _write(output, name, path.read_bytes(), inventory)
    revision, dirty = _revision(sources, adjacent)
    report = {
        "status": "checked_small_finite_decay_block" if simulate else "generated_decay_block_without_simulation",
        "paper": {
            "doi": "10.1038/s41586-025-09583-x",
            "weighted_decay_equation": 8, "classical_spectral_correction_equation": 19,
            "construction": "original finite weighted-access reflection; no correction time-integral circuit",
        },
        "source_commit": revision, "source_dirty": dirty,
        "source_revision_is_exact": revision is not None and dirty is False,
        "adjacent_snapshot_selected": adjacent,
        "source_files": {name: inventory[name] for name, _ in sources if name.endswith(".py")},
        "environment_lock_sha256": inventory["requirements-lock.txt"]["sha256"],
        "environment": {
            "python": platform.python_version(),
            "qiskit": importlib.metadata.version("qiskit"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "device": "CPU", "seed": None, "random_sampling_performed": False,
        },
        "caps": {
            "primitive_operations_each_source": 100_000,
            "dense_system_wires": 2, "actual_recovery_total_wires": 8,
            "output_bytes_total_including_manifest": MAX_OUTPUT_BYTES,
            "source_input_bytes_each": MAX_SOURCE_BYTES_EACH,
        },
        "acceptance": {
            "fixed_circuit_reference_tolerance": NUMERICAL_TOLERANCE,
            "exact_hamiltonian_finite_error_is_acceptance_target": False,
            "negative_roundoff_clipped": False,
        },
        "cases": records, "wall_seconds": time.perf_counter() - start,
        "scope": "finite-Trotter weighted-decay block access; every selected auxiliary wire is zero",
        "remaining": [
            "actual coherent correction and its evolution",
            "noncommuting full channel and Gibbs preparation",
            "continuum/KMS/mesh and mixing/output certificates",
            "standalone input preparation, readout, repetitions/amplification and processing costs",
            "rotation synthesis and physical fault/resource models",
        ],
        **SCOPE_FLAGS,
    }
    _write(output, "results.json", _json_bytes(report), inventory)
    manifest = {
        "schema_version": 1, "status": report["status"],
        "files": dict(inventory), "total_payload_bytes": sum(pin["bytes"] for pin in inventory.values()),
        "manifest_and_sidecar_are_separate": True,
        **SCOPE_FLAGS,
    }
    manifest_raw = _json_bytes(manifest)
    _write(output, "manifest.json", manifest_raw, inventory)
    sidecar = (hashlib.sha256(manifest_raw).hexdigest() + "  manifest.json\n").encode("ascii")
    _write(output, "manifest.sha256", sidecar, inventory)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", choices=tuple(TINY_CASES))
    group.add_argument("--tiny-grid", action="store_true")
    group.add_argument("--n-system", type=int)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--clock-bits", type=int, default=2)
    parser.add_argument("--product-steps", type=int, default=1)
    parser.add_argument("--no-sim", action="store_true")
    args = parser.parse_args()
    report = generate(
        args.output_dir, case=args.case, tiny_grid=args.tiny_grid, n_system=args.n_system,
        beta=args.beta, clock_bits=args.clock_bits, product_steps=args.product_steps,
        simulate=not args.no_sim,
    )
    print(json.dumps({
        "status": report["status"], "cases": len(report["cases"]),
        "all_components_verified": False, "new_algorithm_result": False,
        "wall_seconds": report["wall_seconds"],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
