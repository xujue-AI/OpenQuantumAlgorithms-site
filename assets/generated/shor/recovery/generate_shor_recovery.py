#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare bounded classical recovery on the same actual arithmetic outputs.

Default cases use t=min(2n,18-(2n+2)), selected from a fixed statevector width
cap before looking at the order. No Native estimation or physical search runs.
User-supplied large classical frequencies use --outcome and skip circuits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from source_provenance import git_revision
except ModuleNotFoundError:
    def git_revision(_root):
        return None

from openquantum.shor_recovery import (
    MAX_CERTIFICATE_PERIOD, MAX_MODULAR_POW_CALLS, MAX_OFFSET_RADIUS,
    MAX_MODULUS_BITS, MAX_PHASE_BITS, MAX_SMOOTH_BOUND, RecoveryConfig,
    baseline_frequency, recover_frequency,
)

MAX_DEMO_QUBITS = 18
MAX_DEMO_PHASE_BITS = 12
MAX_DEMO_CASES = 3
MAX_DEMO_OUTCOMES = 4096
MAX_DEMO_SECONDS = 180
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
DEFAULT_OUTPUT = ROOT / "assets/generated/shor/recovery"
SOURCE_FILES = ("src/openquantum/shor_recovery.py", "src/openquantum/shor.py",
                "scripts/generate_shor_recovery.py", "requirements-lock.txt", "LICENSE")
UPSTREAM_COMMIT = "8040affdf08814ef7d451d9797d28f7645856151"
UPSTREAM_LICENSE = 'Copyright (c) 2020–2026 by Martin Ekerå <<ekera@kth.se>>\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_directory(path: Path) -> Path:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("output has a linked ancestor")
    generated = ROOT / "assets/generated"
    if generated == path or generated in path.parents:
        if path != DEFAULT_OUTPUT and DEFAULT_OUTPUT not in path.parents:
            raise ValueError("output cannot overwrite another algorithm's generated artifacts")
    path.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() or (p.is_file() and p.stat().st_nlink != 1) for p in path.rglob("*")):
        raise ValueError("output contains linked files")
    return path


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance() -> dict:
    return {"source_commit": git_revision(ROOT),
            "source_commit_scope": "checkout at execution; source hashes identify actual working bytes",
            "source_sha256": {name: sha((ROOT / name).read_bytes()) for name in SOURCE_FILES},
            "environment": {"python": platform.python_version(), "device": "CPU",
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ("qiskit", "numpy", "matplotlib", "scipy")}},
            "upstream_reference": {"paper_doi": "10.1145/3655026", "quaspy_commit": UPSTREAM_COMMIT,
                                   "license": "MIT", "adoption": "original integer-only implementation; no upstream package imported"}}


def _source_copies(output: Path) -> None:
    for relative in ("src/openquantum/shor.py", "src/openquantum/shor_recovery.py", "scripts/generate_shor_recovery.py"):
        (output / Path(relative).name).write_bytes((ROOT / relative).read_bytes())
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    (output / "Quaspy-LICENSE.txt").write_text(UPSTREAM_LICENSE)


def generate(output: Path = DEFAULT_OUTPUT, *, cases=None, config: RecoveryConfig = RecoveryConfig(),
             simulate: bool = True, seed: int = 20261113) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "openquantum-order-recovery"
    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit import qasm3
    from openquantum.shor import build_order_finding_circuit, decompose, reconstruct_order, simulate_phase_distribution

    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2^63)")
    if (cases is not None or not simulate) and output.absolute() == DEFAULT_OUTPUT:
        raise ValueError("custom/materialization-only runs need a separate output directory")
    if cases is None:
        cases = tuple((N, 2, min(2 * (N - 1).bit_length(), MAX_DEMO_QUBITS - 2 * (N - 1).bit_length() - 2))
                      for N in (15, 21, 33))
    if not 1 <= len(cases) <= MAX_DEMO_CASES:
        raise ValueError("demo is bounded to three cases")
    output = output_directory(output)
    started = time.perf_counter_ns()
    records = []
    original_shor_sha = sha((ROOT / "src/openquantum/shor.py").read_bytes())
    for case_index, (N, a, t) in enumerate(cases):
        if time.perf_counter_ns() - started > MAX_DEMO_SECONDS * 1_000_000_000:
            raise TimeoutError("complete demo deadline exceeded")
        n = (N - 1).bit_length()
        if type(t) is not int or not 1 <= t <= MAX_DEMO_PHASE_BITS or 1 << t > MAX_DEMO_OUTCOMES:
            raise ValueError("demo frequency grid exceeds the bounded reference")
        width = t + 2 * n + 2
        if simulate and width > MAX_DEMO_QUBITS:
            raise ValueError("full actual-circuit statevector exceeds the18-wire demo cap")
        begin = time.perf_counter_ns()
        circuit = build_order_finding_circuit(N, a, t)
        measured = decompose(build_order_finding_circuit(N, a, t, measure=True))
        construction_ns = time.perf_counter_ns() - begin
        prefix = f"n{N}-a{a}-t{t}"
        qasm = qasm3.dumps(measured)
        (output / f"{prefix}.qasm").write_text(qasm)
        record = {"modulus": N, "base": a, "phase_bits": t, "logical_qubits": width,
                  "phase_bits_selected_using_reference_order": False,
                  "phase_rule": "min(2n,18-(2n+2)) for the fixed default cases; explicit t otherwise",
                  "qasm_file": f"{prefix}.qasm", "qasm_sha256": sha(qasm.encode()),
                  "primitive_gate_counts_per_trial": dict(measured.count_ops()),
                  "primitive_operations_per_trial": len(measured.data), "dependency_depth": measured.depth(),
                  "readouts_per_trial": t, "construction_and_lowering_ns": construction_ns,
                  "circuit_source_sha256": original_shor_sha,
                  "simulation": {"performed": False, "reason": "explicitly disabled"},
                  "physical_estimation": {"status": "not_performed", "native_calls": 0},
                  "included_quantum_work": ["logical-zero input assumption and X for modular input1", "phase Hadamards",
                      "every actual controlled modular multiply", "accumulator/borrow cleanup", "inverse QFT", "all phase readouts"],
                  "excluded_physical_costs": ["calibrated initialization/reset", "synthesis certificate", "fault-tolerant scheduling/factories",
                      "routing", "classical readout/feedback communication"]}
        if simulate:
            begin = time.perf_counter_ns()
            actual = simulate_phase_distribution(N, a, t)
            validation_ns = time.perf_counter_ns() - begin
            raw = np.asarray(actual["probabilities"], dtype=float)
            normalization = float(raw.sum())
            probabilities = raw / normalization
            if not np.all(np.isfinite(raw)) or np.any(raw < 0) or abs(normalization - 1) > 1e-10:
                raise AssertionError("actual circuit probabilities failed normalization")
            # The reference order is read ONLY after circuit generation/simulation.
            # It never enters either recovery function or the phase-width choice.
            reference_order = actual["classical_reference_order"]
            rows = []
            old_order, new_order, old_factor, new_factor = 0.0, 0.0, 0.0, 0.0
            for frequency, probability in enumerate(probabilities):
                old = baseline_frequency(N, a, frequency, t, config)
                new = recover_frequency(N, a, frequency, t, config)
                if old["period"] != reconstruct_order(frequency, t, N, a):
                    raise AssertionError("counted baseline changed the original recovery rule")
                if new["status"] == "resource_limit":
                    raise AssertionError("default complete comparison hit a recovery cap")
                if new["certified_order"] is not None and new["certified_order"] != reference_order:
                    raise AssertionError("exact-order certificate disagrees with independent reference")
                if old["period"] == reference_order:
                    old_order += float(probability)
                if new["certified_order"] == reference_order:
                    new_order += float(probability)
                if old["factors"]:
                    old_factor += float(probability)
                if new["factors"]:
                    new_factor += float(probability)
                rows.append({"outcome": frequency, "probability": float(probability), "baseline": old, "recovery": new})
            weighted_work = {}
            for method in ("baseline", "recovery"):
                weighted_work[method] = {name: sum(row["probability"] * row[method]["work"][name] for row in rows)
                                         for name in rows[0][method]["work"]}
            rng_seed = seed + case_index
            rng = np.random.default_rng(rng_seed)
            draws = rng.choice(1 << t, size=16, p=probabilities)
            samples = [{"trial": i, "outcome": int(y), "baseline_period": rows[int(y)]["baseline"]["period"],
                        "baseline_factors": rows[int(y)]["baseline"]["factors"],
                        "recovery_period": rows[int(y)]["recovery"]["period"],
                        "recovery_certified_order": rows[int(y)]["recovery"]["certified_order"],
                        "recovery_factors": rows[int(y)]["recovery"]["factors"]} for i, y in enumerate(draws)]
            record.update({"simulation": {**actual, "performed": True, "simulation_and_independent_check_ns": validation_ns,
                                           "probability_normalization_for_classical_draws": normalization},
                           "all_frequency_rows": rows, "all_outcomes_retained": True,
                           "reference_order_is_diagnostic_only": True,
                           "actual_order_squared_below_grid_diagnostic": reference_order**2 < 1 << t,
                           "all_frequency_recovery_uses_smooth_only_pretest": all(row["recovery"]["smooth_only_pretest"] for row in rows),
                           "weighted_success": {"baseline_exact_order": old_order, "recovery_certified_exact_order": new_order,
                                                "baseline_verified_factor": old_factor, "recovery_verified_factor": new_factor},
                           "probability_weighted_classical_work": weighted_work,
                           "samples": {"seed": rng_seed, "trial_count": 16, "same_draws_for_both_methods": True,
                                       "source": "classical draws from freshly validated actual-circuit probabilities; no QPU execution",
                                       "records": samples},
                           "complete_quantum_work_for_fixed_samples": {"trials": 16, "primitive_operations": 16 * len(measured.data),
                                "phase_readouts": 16 * t, "peak_logical_qubits": width,
                                "unchanged_between_postprocessors": True}})
        records.append(record)
        if case_index == 0:
            fig = circuit.draw(output="mpl", fold=40, scale=.65)
            fig.savefig(output / "circuit.svg", bbox_inches="tight", metadata={"Date": None})
            plt.close(fig)
    _source_copies(output)
    study = {"schema": "openquantum-shor-recovery-v1", "status": "completed_bounded_classical_recovery_comparison" if simulate else "completed_circuit_materialization_only",
             "config": asdict(config), "cases": records, **provenance(),
             "bounds": {"demo_qubits": MAX_DEMO_QUBITS, "cases": MAX_DEMO_CASES, "outcomes_per_case": MAX_DEMO_OUTCOMES,
                        "complete_demo_seconds": MAX_DEMO_SECONDS, "output_bytes": MAX_OUTPUT_BYTES,
                        "modulus_bits_for_classical_recovery": MAX_MODULUS_BITS, "phase_bits_for_classical_recovery": MAX_PHASE_BITS,
                        "offset_radius": MAX_OFFSET_RADIUS, "smooth_bound": MAX_SMOOTH_BOUND,
                        "modular_pow_calls_per_frequency": MAX_MODULAR_POW_CALLS, "certificate_period": MAX_CERTIFICATE_PERIOD},
             "work_counter_scope": "selected algorithm operations: Euclidean divisions, sieve assignments, scheduled/CF products, modular pow calls/exponent lengths, candidate-period divisions and gcd calls; not every Python multiplication, CPython instruction or bit-operation counts",
             "new_native_calls": 0, "new_hardware_execution": False, "exact_gate_angles_are_numerical": True,
             "classical_only_smooth_case_warning": "All default orders are cm-smooth: the retained trivial pretest can recover them without using the measured frequency. This is an educational algorithm/control-flow comparison, not quantum advantage.",
             "original_negative_case_preserved": "N33/base2 has recovered order10 and half-power32=-1mod33; no nontrivial factor is obtained",
             "formalization_status": "no reproduced proof or Python/circuit correspondence theorem"}
    if simulate:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
        positions = np.arange(len(records))
        labels = [f"N={r['modulus']}, t={r['phase_bits']}" for r in records]
        for ax, suffix, title in ((axes[0], "exact_order", "Exact order"), (axes[1], "verified_factor", "Verified factor")):
            key = "recovery_certified_exact_order" if suffix == "exact_order" else "recovery_verified_factor"
            ax.bar(positions - .17, [r["weighted_success"]["baseline_" + suffix] for r in records], .34, label="original CF rule", color="#67758b")
            ax.bar(positions + .17, [r["weighted_success"][key] for r in records], .34, label="bounded completion + certificate", color="#247c82")
            ax.set(xticks=positions, xticklabels=labels, ylim=(0, 1.08), title=title, ylabel="Ideal probability")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "success.svg", metadata={"Date": None})
        plt.close(fig)
        table = []
        for r in records:
            table.append({"N": r["modulus"], "a": r["base"], "t": r["phase_bits"], "wires": r["logical_qubits"],
                          **r["weighted_success"], "smooth_only_pretest_all_outcomes": r["all_frequency_recovery_uses_smooth_only_pretest"]})
        with (output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(table)
        summary = ["| N | a | t | Logical wires | Original exact-order probability | Certified completion probability | Original factor probability | Completion factor probability |",
                   "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in table:
            summary.append(f"| {r['N']} | {r['a']} | {r['t']} | {r['wires']} | {r['baseline_exact_order']:.6f} | {r['recovery_certified_exact_order']:.6f} | {r['baseline_verified_factor']:.6f} | {r['recovery_verified_factor']:.6f} |")
        summary += ["", "All three orders are small and cm-smooth; the completion shortcut can recover them classically before using a frequency. N33/base2 remains a failed factoring base. These rows do not establish quantum advantage or generic factoring confidence."]
        (output / "summary.md").write_text("\n".join(summary) + "\n")
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    readme = "# Bounded order recovery\n\nOriginal integer-only CF/offset/filter/Algorithm3 completion and bounded exact-order certificate. See results.json for executed circuits, retained frequency rows, same sampled outputs, work, limits and provenance. Default orders are all cm-smooth; this does not show quantum advantage. No Native/hardware/formal execution.\n"
    study["execution_wall_ns_before_record_serialization"] = time.perf_counter_ns() - started
    encoded = json.dumps(study, indent=2, allow_nan=False) + "\n"
    other_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and p.name not in {"results.json", "README.md"})
    if other_bytes + len(encoded.encode()) + len(readme.encode()) > MAX_OUTPUT_BYTES:
        raise ValueError("complete generated output exceeds64MiB")
    if time.perf_counter_ns() - started > MAX_DEMO_SECONDS * 1_000_000_000:
        raise TimeoutError("complete demo deadline exceeded")
    (output / "results.json").write_text(encoded)
    (output / "README.md").write_text(readme)
    print(f"Completed {len(records)} cases; Native calls0; simulation={simulate}")
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modulus", type=int)
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--phase-bits", type=int)
    parser.add_argument("--outcome", type=int, action="append", help="explicit classical-only frequency; no circuit/simulator")
    parser.add_argument("--offset-radius", type=int, default=2)
    parser.add_argument("--smooth-multiplier", type=int, default=1)
    parser.add_argument("--certificate-period-cap", type=int, default=MAX_CERTIFICATE_PERIOD)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--no-sim", action="store_true")
    args = parser.parse_args()
    config = RecoveryConfig(args.offset_radius, args.smooth_multiplier, args.certificate_period_cap)
    if args.outcome:
        if args.modulus is None or args.phase_bits is None or len(args.outcome) > 64:
            parser.error("classical-only input needs N,t and at most64 explicit outcomes")
        if args.output_dir.absolute() == DEFAULT_OUTPUT:
            parser.error("classical-only inputs need a separate --output-dir")
        output = output_directory(args.output_dir)
        results = [recover_frequency(args.modulus, args.base, j, args.phase_bits, config) for j in args.outcome]
        dump(output / "classical-only.json", {"frequency_source": "user supplied; no quantum execution or distribution claim",
                                             "results": results, **provenance(), "native_calls": 0})
        _source_copies(output)
    else:
        cases = None
        if args.modulus is not None:
            if args.phase_bits is None:
                parser.error("a custom circuit needs explicit --phase-bits")
            cases = ((args.modulus, args.base, args.phase_bits),)
        generate(args.output_dir, cases=cases, config=config, simulate=not args.no_sim, seed=args.seed)


if __name__ == "__main__":
    main()
