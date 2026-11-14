#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded multiproduct source exports, numerical checks and complete ledgers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

LOCAL = Path(__file__).resolve().with_name("multiproduct.py")
if LOCAL.is_file():
    ROOT = LOCAL.parent
    MODULE_PATH = LOCAL
else:
    ROOT = Path(__file__).resolve().parents[1]
    MODULE_PATH = ROOT / "src/openquantum/simulation/multiproduct.py"
MODULE_BYTES = MODULE_PATH.read_bytes()
mpf = types.ModuleType("openquantum_fixed_mpf_runtime")
mpf.__file__ = str(MODULE_PATH)
mpf.__package__ = ""
sys.modules[mpf.__name__] = mpf
exec(compile(MODULE_BYTES, str(MODULE_PATH), "exec"), mpf.__dict__)

import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator

MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_FILE_BYTES = 4 * 1024 * 1024
GRID = tuple((n, r) for n in (1, 2, 3) for r in (1, 2, 4, 8)) + ((2, 32), (3, 32))


def qasm_text(circuit):
    """Keep the circuit's global phase explicit, including the OAA minus sign."""
    text = qasm3.dumps(circuit)
    phase = float(circuit.global_phase)
    inserted = False
    if phase and not any(line.strip().startswith("gphase(") for line in text.splitlines()):
        lines = text.splitlines()
        declarations = [i for i, line in enumerate(lines) if line.strip().startswith(("qubit[", "qubit "))]
        if not declarations:
            raise ValueError("QASM export has no qubit declaration")
        lines.insert(declarations[-1] + 1, f"gphase({phase!r});")
        text = "\n".join(lines) + "\n"
        inserted = True
    return text, {"global_phase_radians": phase, "explicit_gphase_inserted": inserted,
                  "convention": "OpenQASM3 gphase retains the Python circuit's phase; no controlled-evolution export has been checked"}


def _json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _write(output, filename, raw, inventory):
    if not isinstance(raw, bytes) or len(raw) > MAX_FILE_BYTES:
        raise ValueError("one multiproduct export exceeds4MiB")
    if Path(filename).name != filename or filename in inventory:
        raise ValueError("invalid or repeated output filename")
    if sum(item["bytes"] for item in inventory.values()) + len(raw) > MAX_EXPORT_BYTES:
        raise ValueError("multiproduct aggregate exports exceed32MiB")
    with (output / filename).open("xb") as handle:
        handle.write(raw)
    inventory[filename] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _source_record():
    source = Path(mpf.__file__).resolve()
    generator = Path(__file__).resolve()
    if source.read_bytes() != MODULE_BYTES:
        raise ValueError("loaded algorithm bytes differ from source on disk")
    sources = {source.name: MODULE_BYTES, generator.name: generator.read_bytes()}
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()}
    commit, dirty = None, None
    if not LOCAL.exists():
        try:
            top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                                 text=True, capture_output=True, timeout=5, check=True).stdout.strip()
            if Path(top).resolve() == ROOT:
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        text=True, capture_output=True, timeout=5, check=True).stdout.strip()
                paths = [str(path.relative_to(ROOT)) for path in (source, generator)]
                state = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
                                       text=True, capture_output=True, timeout=5, check=True).stdout
                dirty = bool(state)
                if not dirty:
                    for path, raw in zip(paths, sources.values()):
                        committed = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT,
                                                   capture_output=True, timeout=5, check=True).stdout
                        if committed != raw:
                            raise ValueError("current Git source blob differs from executed source")
        except (OSError, subprocess.SubprocessError):
            commit, dirty = None, None
    record = {"source_commit": commit, "owned_source_dirty": dirty, "source_sha256": hashes,
              "source_scope": "two actual algorithm/generator source files; standalone exports never borrow ancestor Git",
              "python": platform.python_version(), "device": "CPU", "seed": 17,
              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "qiskit")}}
    return sources, record


def _binomial(rng, samples, mean):
    if not math.isfinite(mean) or not -1 - 1e-10 <= mean <= 1 + 1e-10:
        raise ValueError("circuit Z mean is outside its physical range")
    # Only roundoff in a physically bounded circuit probability is corrected.
    p = min(1., max(0., (mean + 1) / 2))
    count = int(rng.binomial(samples, p))
    return {"represented_samples": samples, "q0_zero_count": count,
            "estimated_Z": 2 * count / samples - 1, "expected_Z": mean}


def _fixture_record():
    records, circuits = {}, {}
    for amplified in (False, True):
        circuit = mpf.opposite_identity_fixture(amplified)
        matrix = Operator(circuit).data
        block = matrix[:2, :2]
        scalar = 5 / 27 if amplified else 5 / 6
        residual = float(np.linalg.norm(block - scalar * np.eye(2), 2))
        leakage = float(np.linalg.norm(matrix[8:, :8], 2))
        if max(residual, leakage) > 1e-10:
            raise AssertionError("opposite-identity source fixture failed")
        state = block[:, 0]
        p = float(np.vdot(state, state).real)
        state /= math.sqrt(p)
        key = "oaa" if amplified else "lcu"
        records[key] = {"block_reference_error": residual, "clean_work_leakage": leakage,
                        "acceptance": p, "ideal_acceptance": "25/729" if amplified else "25/36",
                        "normalized_state_error": float(np.linalg.norm(state - np.array([1, 0]))),
                        "raw_operator_error_to_identity": abs(scalar * (1 if amplified else 2) - 1),
                        "scope": "explicitphase fixture Uf=I,Uc=-I outside common-H Strang family; not a conditional-state failure"}
        circuits[key] = circuit
    return records, circuits


def _csv(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def generate(output, *, n=2, steps=4, time_value=1., J=1., h=.7, grid=False, no_sim=False, figures=True):
    started = time.monotonic()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new absent directory; existing records are preserved")
    if grid and no_sim:
        raise ValueError("the fixed grid is a bounded numerical study")
    if grid and (time_value != 1. or J != 1. or h != .7):
        raise ValueError("fixed grid requires t=1,J=1,h=.7")
    parameters = GRID if grid else ((n, steps),)
    # Every source and complete-attempt bound is checked before output creation.
    for system, r in parameters:
        for method in mpf.METHODS:
            mpf.preflight(system, time_value, r, method, J, h)
    sources, provenance = _source_record()
    output.mkdir(parents=True)
    inventory, cases, rows = {}, [], []
    for name, raw in sources.items():
        _write(output, name, raw, inventory)
    lock = ROOT / "requirements-lock.txt"
    license_path = ROOT / ("LICENSE.txt" if LOCAL.exists() else "LICENSE")
    companions = {}
    for name, path in (("requirements-lock.txt", lock), ("LICENSE.txt", license_path)):
        if not path.is_file():
            raise ValueError(f"missing reproducible source companion: {name}")
        raw = path.read_bytes()
        companions[name] = (path, raw)
        _write(output, name, raw, inventory)
    rng = np.random.default_rng(17)
    for system, r in parameters:
        case_id = f"n{system}-r{r}"
        performed = not no_sim and system <= mpf.MAX_REFERENCE_SYSTEM_QUBITS
        record = {"case_id": case_id, "n_system": system, "steps": r, "time": time_value,
                  "J": J, "h": h, "initial_basis": 0, "readout": "Z(q0)", "methods": {}}
        validation = mpf.compare(system, time_value, r, J, h) if performed else {
            "performed": False, "reason": "explicit --no-sim or three-system reference cap"}
        record["validation"] = validation
        for method in mpf.METHODS:
            circuit = mpf.build_attempt(system, time_value, r, method, J, h)
            raw_qasm, convention = qasm_text(circuit)
            filename = f"{case_id}-{method}.qasm"
            _write(output, filename, raw_qasm.encode(), inventory)
            record["methods"][method] = {"qasm": filename, "circuit": circuit.metadata,
                                          "qasm_convention": convention,
                                          "resource_proposal": mpf.resource_proposal(system, time_value, r, method, J, h)}
        if performed:
            sampling = mpf.sampling_plan()
            draws = {method: _binomial(rng, sampling["single_accepted_samples"],
                                       validation["conditional"][method]["Z"]) for method in ("lcu", "oaa")}
            draws["oaa-channel"] = _binomial(rng, sampling["single_accepted_samples"], validation["unconditional_oaa_Z"])
            draws["fine"] = _binomial(rng, sampling["classical_fine_samples"], validation["classical_fine_Z"])
            draws["coarse"] = _binomial(rng, sampling["classical_coarse_samples"], validation["classical_coarse_Z"])
            record["illustrative_sampling"] = {
                "kind": "seeded classical binomial counts from independently checked ideal circuit means",
                "draws": draws,
                "raw_weighted_estimated_Z": (4 * draws["fine"]["estimated_Z"] - draws["coarse"]["estimated_Z"]) / 3,
                "weighted_estimator_clipped": False,
                "quantum_trials_or_rejections_executed": False,
                "scope": "represented sampling populations; no hardware or literal long circuit stream"}
            rows.append({"n_system": system, "steps": r, "fine_error": validation["fine_operator_error"],
                         "coarse_error": validation["coarse_operator_error"], "A_error": validation["A_operator_error"],
                         "F_error": validation["F_operator_error"], "lcu_pmin": validation["lcu_min_acceptance_numeric"],
                         "oaa_pmin": validation["oaa_min_acceptance_numeric"],
                         "classical_Z_error": validation["classical_weighted_Z_error"],
                         "conditional_oaa_Z_error": validation["conditional"]["oaa"]["Z_error"],
                         "unconditional_oaa_Z_error": validation["unconditional_oaa_Z_error"]})
        cases.append(record)
    negative = None
    if not no_sim:
        negative, fixture_circuits = _fixture_record()
        for method, circuit in fixture_circuits.items():
            text, _ = qasm_text(circuit)
            _write(output, f"opposite-identity-{method}.qasm", text.encode(), inventory)
    state_instruments = {}
    state_n, state_r = (2, 4) if grid else (n, steps)
    for method in ("lcu", "oaa", "oaa-channel"):
        instrument = mpf.build_state_preparation(state_n, time_value, state_r, method, J, h)
        filename = f"state-n{state_n}-r{state_r}-{method}.qasm"
        text, convention = qasm_text(instrument)
        _write(output, filename, text.encode(), inventory)
        state_instruments[method] = {"qasm": filename, "circuit": instrument.metadata,
                                     "qasm_convention": convention,
                                     "scope": "complete known-input instrument; system notmeasured; numerical modelcheck scope remainsn<=3"}
    study = {"schema": "openquantum-fixed-multiproduct-study-v1",
             "status": "generated_and_small_checked" if rows else "generated_only_unvalidated_numerically",
             "source": provenance, "hamiltonian": "openchain J*sumZZ+h*sumX; no onsiteZ",
             "family": "fixedtwo-term Richardson; optimizedhigherorder MPF notimplemented",
             "coefficients": [str(value) for value in mpf.COEFFICIENTS],
             "prepare_probabilities": [str(value) for value in mpf.PROBABILITIES],
             "normalization": "2", "cases": cases, "state_instruments": state_instruments, "negative_fixture": negative,
             "theory_scope": "fixed-H,t asymptoticfourthorder; supplied finite certificate is conservative secondorder",
             "physical_or_synthesis_or_formal_executed": False,
             "device": "CPU", "seed": 17, "elapsed_seconds": time.monotonic() - started}
    _write(output, "study.json", _json(study), inventory)
    if rows:
        _write(output, "convergence.csv", _csv(rows), inventory)
    scale = [mpf.scalar_counts(system, 1, method) for system in (1, 2, 3, 8, 16, 32, 61)
             for method in ("lcu", "oaa", "fine", "coarse")]
    _write(output, "scalar-scaling.json", _json({"counts": scale, "scope": "exactscalar sourcecounts; no new large numerical validation"}), inventory)
    if figures and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["svg.hashsalt"] = "openquantum-fixed-mpf"
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for system, color in ((2, "#346b8a"), (3, "#be6b35")):
            group = [row for row in rows if row["n_system"] == system]
            if not group:
                continue
            for key, style in (("coarse_error", "--"), ("A_error", "-"), ("F_error", ":")):
                axes[0].loglog([row["steps"] for row in group], [max(1e-16, row[key]) for row in group],
                               "o" + style, color=color, label=f"n={system} {key}")
            axes[1].semilogx([row["steps"] for row in group], [row["oaa_pmin"] for row in group],
                            "o-", color=color, label=f"n={system} OAA")
        axes[0].set(xlabel="Coarse Strang steps r", ylabel="Operatorerror")
        axes[1].set(xlabel="Coarse Strang steps r", ylabel="Numericalminimum acceptance")
        for ax in axes:
            ax.grid(alpha=.18)
            ax.legend(fontsize=7)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None})
        plt.close(fig)
        _write(output, "convergence.svg", buffer.getvalue(), inventory)
        demo = mpf.build_attempt(2, 1., 1, "lcu")
        fig = demo.draw(output="mpl", fold=40, scale=.75)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
        _write(output, "small-lcu-circuit.svg", buffer.getvalue(), inventory)
    readme = """# Fixed two-term multiproduct implementation

This original MIT implementation uses NumPy, SciPy and pinned Qiskit2.6.2.
Install requirements-lock.txt in a Python3.12.14 virtual environment. Run
`python generate_simulation_multiproduct.py --output-dir NEW_DIRECTORY`.
The generator refuses existing directories. A large circuit uses `--qubits 32
--steps 1 --no-sim --no-figures`; bounded numerical references stop at 3 spins.

Complete measured-trial QASMs contain literal primitive access, input resets
and readouts. State-instrument QASMs leave the system unmeasured; the separate
opposite-identity counterexamples are unitary fixtures.
Nonzero circuit phases are retained through explicit OpenQASM3 gphase.
LCU/OAA postselection, unconditional discarded-ancilla OAA, and classical
weighted means are different outputs. Both SELECT branches, padding,
PREPARE/uncomputation, reflections, work, repetitions and readout are counted.
The opposite-identity fixture is outside the common-H Strang family and
retains OAA's possible acceptance decrease without claiming state failure.

The grid checks implementation on smallinputs. Fixed-H,t fourthorder is an
asymptotic statement; the concrete commutator certificate is conservative
secondorder. Finite counts and proposed synthesis/physical allowances are
not executed physical estimates or hardware/accuracy certificates. Projected
sampling and rejection populations are unexecuted. No formal proof is built.

References: Low etal1907.11679v2; CarreraVazquez etalQuantum7,1067,
DOI10.22331/q-2023-07-25-1067; MizutaQuantum10,1974,
DOI10.22331/q-2026-01-19-1974. Upstream Qiskit is Apache2.0.
"""
    _write(output, "README.md", readme.encode(), inventory)
    if _source_record()[1]["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("algorithm source changed during generation")
    if any(path.read_bytes() != raw for path, raw in companions.values()):
        raise ValueError("lock/license companions changed during generation")
    manifest = {"schema": "openquantum-fixed-multiproduct-bundle-v1", "source": provenance,
                "files": inventory, "source_parameters_fixed_for_grid": bool(grid),
                "exports_bytes_without_manifest": sum(item["bytes"] for item in inventory.values()),
                "scope": "scientific source/QASM/numericalrecords; no private controllers, logs or credentials"}
    raw_manifest = _json(manifest)
    if sum(item["bytes"] for item in inventory.values()) + len(raw_manifest) > MAX_EXPORT_BYTES:
        raise ValueError("final manifest makesbundle exceed32MiB")
    with (output / "manifest.json").open("xb") as handle:
        handle.write(raw_manifest)
    return {"output": str(output), "cases": len(cases), "numerical_cases": len(rows),
            "files": len(inventory) + 1, "bytes": sum(item["bytes"] for item in inventory.values()) + len(raw_manifest),
            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--time", type=float, default=1.)
    parser.add_argument("--J", type=float, default=1.)
    parser.add_argument("--h", type=float, default=.7)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/multiproduct")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, n=args.qubits, steps=args.steps, time_value=args.time,
                              J=args.J, h=args.h, grid=args.grid, no_sim=args.no_sim,
                              figures=not args.no_figures), indent=2))


if __name__ == "__main__":
    main()
