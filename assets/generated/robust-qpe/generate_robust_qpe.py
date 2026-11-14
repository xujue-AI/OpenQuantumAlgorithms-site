#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded robust-QPE circuits, readout-distorted counts and complete records."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT = Path(__file__).resolve()
PORTABLE = (SCRIPT.parent / "robust_qpe.py").is_file()
ROOT = SCRIPT.parent if PORTABLE else SCRIPT.parents[1]
sys.path.insert(0, str(ROOT if PORTABLE else ROOT / "src"))
if PORTABLE:
    import robust_qpe as rq
else:
    from openquantum import robust_qpe as rq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_WALL_SECONDS = 120
PHASES = (.3, .995, .125, .5)
SEEDS = (7, 19)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_files() -> dict[str, Path]:
    if PORTABLE:
        files = {name: ROOT / name for name in ("robust_qpe.py", "hadamard_qpe.py", "requirements-lock.txt", "LICENSE.txt")}
    else:
        files = {"robust_qpe.py": ROOT / "src/openquantum/robust_qpe.py",
                 "hadamard_qpe.py": ROOT / "src/openquantum/hadamard_qpe.py",
                 "requirements-lock.txt": ROOT / "requirements-lock.txt", "LICENSE.txt": ROOT / "LICENSE"}
    files["generate_robust_qpe.py"] = SCRIPT
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size > 1024 * 1024 for p in files.values()):
        raise ValueError("Complete bounded regular source, lock and license files required")
    return files


def checkpoint(start: float, output: Path) -> None:
    if time.perf_counter() - start > MAX_WALL_SECONDS:
        raise TimeoutError("Generator exceeded its 120-second allowance")
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > MAX_OUTPUT_BYTES:
        raise ValueError("Generator exceeded its 16 MiB output cap")


def export(path: Path, circuit) -> dict:
    lowered = rq.lower_experiment(circuit)
    raw = qasm3.dumps(lowered).encode()
    path.write_bytes(raw)
    return {"file": "circuits/" + path.name, "sha256": digest(raw),
            "logical_qubits": circuit.num_qubits, "readout_bits": circuit.num_clbits,
            "source_gate_counts": dict(circuit.count_ops()), "source_dependency_depth": circuit.depth(),
            "lowered_gate_counts": dict(lowered.count_ops()), "lowered_dependency_depth": lowered.depth(),
            "lowered_global_phase_radians": float(lowered.global_phase),
            "scope": "measured fresh-reset primitive program; continuous RZ unsynthesized; QASM omits irrelevant overall phase"}


def generate(output: Path, *, target_qubits: int = 1, phase: float = .3, levels: int = 6,
             failure_target: Fraction = Fraction(1, 100), additive_bound: Fraction = Fraction(1, 50),
             zero_to_one: Fraction = Fraction(3, 200), one_to_zero: Fraction = Fraction(1, 100),
             seed: int = 7, canonical: bool = True, simulate: bool = True) -> dict:
    start = time.perf_counter()
    if not all(isinstance(v, bool) for v in (canonical, simulate)):
        raise ValueError("canonical and simulate must be booleans")
    rq.sampling_tolerance(additive_bound)
    # Validate the concrete readout channel independently of the bound.
    rq.readout_zero_probability(.5, zero_to_one, one_to_zero)
    if max(zero_to_one, one_to_zero) > additive_bound:
        raise ValueError("Readout channel is outside the supplied additive bound")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 63):
        raise ValueError("Bounded nonnegative integer seed required")
    if canonical and (target_qubits, phase, levels, failure_target, additive_bound, zero_to_one, one_to_zero, seed) != (1, .3, 6, Fraction(1,100), Fraction(1,50), Fraction(3,200), Fraction(1,100), 7):
        raise ValueError("Canonical study has frozen defaults; use --single")
    phases, seeds = (PHASES, SEEDS) if canonical else ((phase,), (seed,))
    run_count = len(phases) * len(seeds)
    m = rq.shots_for_failure(levels, failure_target / run_count, additive_bound)
    ledger = rq.protocol_ledger(target_qubits, levels, m, additive_bound)
    proposed_trials = run_count * ledger["complete_fresh_preparations"]
    if len(phases) * 2 * levels + 1 > rq.MAX_EXPERIMENT_CIRCUITS or proposed_trials > rq.MAX_MODELED_DRAWS:
        raise ValueError("Whole invocation exceeds circuit or modeled-trial cap")
    if simulate and target_qubits > rq.MAX_DENSE_TARGET_QUBITS:
        raise ValueError("Use --no-sim beyond three data wires")
    # Destination must be new; no existing evidence is silently replaced.
    destination = output.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output must be a new directory")
    if not PORTABLE:
        protected = (ROOT / "assets/generated").resolve()
        resolved = destination.resolve()
        if resolved.is_relative_to(protected) and resolved != protected / "robust-qpe":
            raise ValueError("Output cannot overwrite another algorithm's generated tree")
        if resolved == ROOT or any(resolved.is_relative_to((ROOT / name).resolve()) for name in ("src", "scripts", "notebooks", "algorithms", "references")):
            raise ValueError("Output cannot overwrite source")
    files = source_files()
    hashes = {name: digest(path.read_bytes()) for name, path in files.items()}
    commit = None
    if not PORTABLE:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False).stdout.strip() or None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robust-qpe-stage-", dir=destination.parent) as temp:
        stage = Path(temp)
        (stage / "circuits").mkdir()
        for name, path in files.items():
            shutil.copyfile(path, stage / name)
        cases, runs = [], []
        for index, current_phase in enumerate(phases):
            circuits = rq.build_schedule(target_qubits, current_phase, levels)
            case = {"case_id": f"phase-{index}", "nominal_phase": current_phase,
                    "actual_angle_phase_reference": rq.actual_angle_phase(target_qubits, current_phase),
                    "reference_scope": "numerical binary64 stored-angle diagnostic; not an exact phase certificate", "experiments": []}
            probabilities = []
            for j, circuit in enumerate(circuits):
                k, basis = 1 << (j // 2), "sin" if j % 2 else "cos"
                entry = {"power": k, "basis": basis, **export(stage / "circuits" / f"phase-{index}-{basis}-k{k}.qasm", circuit)}
                if simulate:
                    entry["check"] = rq.check_experiment(target_qubits, current_phase, k, basis)
                    raw = entry["check"]["zero_probability"]
                    p = min(1.0, max(0.0, raw))
                    if abs(p - raw) > 2e-11:
                        raise AssertionError("Probability clipping exceeded numerical allowance")
                    q = rq.readout_zero_probability(p, zero_to_one, one_to_zero)
                    confusion = np.array([[1-float(zero_to_one), float(one_to_zero)], [float(zero_to_one), 1-float(one_to_zero)]])
                    independent = float((confusion @ np.array([p, 1-p]))[0])
                    if abs(q-independent) > 1e-15 or abs(q-p) > float(additive_bound) + 1e-15:
                        raise AssertionError("Independent readout/Bernoulli law failed")
                    entry.update(ideal_sampling_zero_probability=p, observed_zero_probability=q,
                                 readout_bias=q-p, independent_confusion_zero_probability=independent,
                                 probability_clipping_absolute_error=abs(p-raw))
                    probabilities.append(q)
                else:
                    entry["check"] = {"performed": False, "reason": "--no-sim; no sampled output or validation at this size"}
                case["experiments"].append(entry)
                checkpoint(start, stage)
            cases.append(case)
            if simulate:
                for current_seed in seeds:
                    tick = time.perf_counter()
                    counts = rq.sample_counts(probabilities, levels, m, current_seed, stream=index)
                    after_sampling = time.perf_counter()
                    result = rq.recover_phase(counts, additive_bound)
                    elapsed = time.perf_counter() - after_sampling
                    value = result["phase_estimate"]
                    error = None if value is None else rq.circular_distance(value, case["actual_angle_phase_reference"])
                    outcome = "inconclusive" if value is None else ("within_radius" if error <= float(Fraction(ledger["phase_radius_exact"])) else "miss")
                    runs.append({"case_id": case["case_id"], "seed": current_seed, "stream": index,
                                 "seed_scope": "SeedSequence([seed, predefined phase index]); every output retained",
                                 "result": result, "circular_error_vs_actual_angle_reference": error,
                                 "outcome": outcome, "classical_binomial_sampling_seconds": after_sampling-tick,
                                 "classical_rational_inference_seconds": elapsed})
                    checkpoint(start, stage)
        large = rq.build_experiment(8, .3, 32, "sin")
        large_record = export(stage / "circuits/large-n8-sin-k32.qasm", large)
        large_record["check"] = {"performed": False, "reason": "separate nine-wire source/count example, no dense simulation"}
        rows = []
        for n in (1, 2, 4, 8):
            for length in (3, 6, 9):
                for beta in (Fraction(0), Fraction(1,50), Fraction(3,50)):
                    shots = rq.shots_for_failure(length, Fraction(1,100), beta)
                    item = rq.protocol_ledger(n, length, shots, beta)
                    rows.append({"target_qubits": n, "levels": length, "additive_bound_exact": str(beta),
                                 "failure_target_exact": "1/100", "shots_per_fringe": shots,
                                 "logical_qubits": n+1, "phase_radius_exact": item["phase_radius_exact"],
                                 "maximum_controlled_power": item["maximum_controlled_U_power"],
                                 "fresh_preparations_and_readout_bits": item["complete_fresh_preparations"],
                                 "controlled_base_U_calls": item["shot_weighted_controlled_U_calls"],
                                 "raw_P_rotations": item["raw_continuous_phase_rotations"],
                                 "source_operations": sum(item["source_gate_counts_all_trials"].values()),
                                 "scope": "analytic construction/workload; no physical estimate or simulation at this row"})
        with (stage / "scaling.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        statuses = {name: sum(r["outcome"] == name for r in runs) for name in ("within_radius", "miss", "inconclusive")}
        per_bound = rq.failure_bound(levels, m, additive_bound)
        data = {"schema": "openquantum-robust-qpe-v1",
                "parameters": {"target_qubits": target_qubits, "levels": levels, "phases": list(phases), "seeds": list(seeds),
                               "canonical_study": canonical, "shots_per_fringe": m, "whole_study_failure_target_exact": str(failure_target),
                               "per_output_failure_target_exact": str(failure_target / run_count), "additive_probability_bound_exact": str(additive_bound),
                               "readout_zero_to_one_exact": str(zero_to_one), "readout_one_to_zero_exact": str(one_to_zero)},
                "provenance": {"source_commit": commit, "source_sha256": hashes, "scope": "exact snapshots identify producer; commit may precede working edits"},
                "environment": {"python": platform.python_version(), "device": "CPU", "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "numpy", "matplotlib")}},
                "simulation": {"performed": simulate, "modeled_fresh_trials_sampled": proposed_trials if simulate else 0,
                               "scope": "classical binomial counts from numerically checked quantum circuits followed by specified independent readout confusion; no QPU"},
                "confidence": {"phase_radius_exact": ledger["phase_radius_exact"], "per_output_failure_upper_bound_exact": str(per_bound),
                               "whole_predeclared_study_failure_upper_bound_exact": str(min(Fraction(1), run_count*per_bound)),
                               "scope": "independently derived conservative conditional uniform bound; not the original RPE constants, posterior coverage or physical fault tolerance"},
                "cases": cases, "runs": runs, "outcomes": statuses, "common_per_output_ledger": ledger,
                "whole_study_modeled_workload": {"proposed_outputs": run_count, "fresh_preparations": proposed_trials,
                                                "controlled_base_U_calls": run_count*ledger["shot_weighted_controlled_U_calls"],
                                                "source_gate_counts": {g: run_count*c for g,c in ledger["source_gate_counts_all_trials"].items()},
                                                "status": "all represented noisy-readout trials sampled" if simulate else "proposal only; no trials sampled"},
                "large_source_only_export": large_record, "physical_estimate_performed": False, "formal_proof_reproduced": False}
        dump(stage / "results.json", data)
        dump(stage / "software.json", data["environment"])
        summary = ["| Phase | Seed | Status | Estimate | Circular error |", "|---:|---:|:---|---:|---:|"]
        for run in runs:
            nominal = next(c["nominal_phase"] for c in cases if c["case_id"] == run["case_id"])
            value, error = run["result"]["phase_estimate"], run["circular_error_vs_actual_angle_reference"]
            summary.append(f"| {nominal} | {run['seed']} | {run['outcome']} | {'—' if value is None else format(value,'.10f')} | {'—' if error is None else format(error,'.8g')} |")
        summary += ["", f"{len(runs)} retained outputs; {statuses['miss']} misses and {statuses['inconclusive']} inconclusive outputs. "
                    f"{m} shots per fringe; radius `{ledger['phase_radius_exact']}`; {proposed_trials if simulate else 0:,} sampled represented trials. "
                    f"The conditional union-bound upper bound is {float(min(Fraction(1),run_count*per_bound)):.8g}. The observed outcomes do not establish coverage."]
        (stage / "summary.md").write_text("\n".join(summary) + "\n")
        fig = rq.build_experiment(target_qubits, phases[0], 2, "sin").draw(output="mpl", fold=32)
        fig.savefig(stage / "circuit.svg", bbox_inches="tight"); plt.close(fig)
        if runs:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.bar(range(len(runs)), [r["circular_error_vs_actual_angle_reference"] if r["circular_error_vs_actual_angle_reference"] is not None else 0 for r in runs], color="#216c72")
            for i, run in enumerate(runs):
                if run["outcome"] == "inconclusive":
                    ax.text(i, 0, "inconclusive", rotation=90)
            ax.axhline(float(Fraction(ledger["phase_radius_exact"])), color="#b5643e", linestyle="--", label="Conditional robust radius")
            ax.set(xticks=range(len(runs)), xticklabels=[f"{r['case_id']}/s{r['seed']}" for r in runs], ylabel="Circular phase error", title="Every prescribed noisy-readout output")
            ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout()
            fig.savefig(stage / "recovery.svg"); plt.close(fig)
        checkpoint(start, stage)
        manifest = {"schema": "openquantum-robust-qpe-files-v1", "file_sha256": {p.relative_to(stage).as_posix(): digest(p.read_bytes()) for p in sorted(stage.rglob("*")) if p.is_file()}}
        dump(stage / "manifest.json", manifest)
        checkpoint(start, stage)
        shutil.move(str(stage), destination)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--target-qubits", type=int, default=1)
    parser.add_argument("--phase", type=float, default=.3)
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--failure-target", type=Fraction, default=Fraction(1,100))
    parser.add_argument("--additive-bound", type=Fraction, default=Fraction(1,50))
    parser.add_argument("--zero-to-one", type=Fraction, default=Fraction(3,200))
    parser.add_argument("--one-to-zero", type=Fraction, default=Fraction(1,100))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/robust-qpe" if not PORTABLE else ROOT / "new-robust-run")
    args = parser.parse_args()
    data = generate(args.output_dir, target_qubits=args.target_qubits, phase=args.phase, levels=args.levels,
                    failure_target=args.failure_target, additive_bound=args.additive_bound,
                    zero_to_one=args.zero_to_one, one_to_zero=args.one_to_zero, seed=args.seed,
                    canonical=not args.single, simulate=not args.no_sim)
    print(json.dumps({"schema": data["schema"], "simulation_performed": data["simulation"]["performed"], "outcomes": data["outcomes"]}))


if __name__ == "__main__":
    main()
