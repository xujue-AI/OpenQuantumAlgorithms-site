#!/usr/bin/env python3
"""Bounded Grover scaling with explicit circuits and conditional physical costs.

The frozen n=3 report is read for hardware/failure assumptions and never written.
Default run covers n=2..8. Synthesis costs are the audited QDK mean-fit model,
not concrete approximation certificates or certified runtime-tail bounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

# Source layout is independent of optional Git checkout metadata.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from source_provenance import git_dirty, git_revision
except ModuleNotFoundError as error:
    if error.name != "source_provenance":
        raise
    # Individual source downloads can omit this optional metadata helper.
    def git_revision(_root):
        return None
    def git_dirty(_root, _paths):
        return None

import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import qasm3
from qiskit.quantum_info import Statevector

from openquantum.grover import build_grover_circuit, expected_success_probability, optimal_iterations
from openquantum.resource_estimation import (
    CircuitResourceBudget,
    SurfaceCodeProfile,
    estimate_circuit_resources,
    prepare_measured_circuit,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def equality_target(n_qubits: int) -> int:
    """Alternating bits: q[0], q[2], ... are one, giving n=3 target 101."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or not 2 <= n_qubits <= 8:
        raise ValueError("This bounded scaling study supports integer n=2..8")
    return sum(1 << bit for bit in range(0, n_qubits, 2))


def stopped_trial_expectation(success: float, cap: int) -> float:
    """E[min(G, cap)] for IID geometric success, including failed capped runs."""
    if not math.isfinite(success) or not 0 < success <= 1:
        raise ValueError("Success must lie in (0, 1]")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("Trial cap must be a positive integer")
    if success == 1:
        return 1.0
    return -math.expm1(cap * math.log1p(-success)) / success


def trial_contract(
    ideal_success: float,
    synthesis_allowance: float,
    sampling_failure_allowance: float,
    physical_failure_allowance: float,
    *,
    max_trials: int = 10_000,
) -> dict:
    """Allocate capped trials under an explicitly conditional probability model.

    Subtracting the synthesis allowance from ideal success assumes that this
    allowance bounds output-distribution TV. The mean-fit helper does not certify
    that assumption. Physical faults are allocated separately across the cap.
    """
    if not math.isfinite(ideal_success) or not 0 < ideal_success <= 1:
        raise ValueError("Ideal success must lie in (0, 1]")
    if not math.isfinite(synthesis_allowance) or synthesis_allowance < 0:
        raise ValueError("Synthesis allowance must be finite and nonnegative")
    for value in (sampling_failure_allowance, physical_failure_allowance):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Failure allowances must lie in (0, 1)")
    if sampling_failure_allowance + physical_failure_allowance >= 1:
        raise ValueError("Combined failure target must be below one")
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer")
    lower = max(0.0, ideal_success - synthesis_allowance)
    if lower == 0:
        raise ValueError("No finite cap meets confidence when conditional success lower bound is zero")
    if lower < 1 and max_trials * math.log1p(-lower) > math.log(sampling_failure_allowance):
        raise ValueError(f"No confidence-compliant cap at or below {max_trials}; confidence was not truncated")
    required = 1 if lower == 1 else math.ceil(math.log(sampling_failure_allowance) / math.log1p(-lower))
    required = max(1, required)
    if required > max_trials:
        raise ValueError(f"Required cap {required} exceeds supported maximum {max_trials}; confidence was not truncated")
    miss = (1 - lower) ** required
    if miss > sampling_failure_allowance + 1e-14:
        raise RuntimeError("Computed trial cap failed its sampling allowance")
    return {
        "ideal_success_probability_per_trial": ideal_success,
        "configured_synthesis_model_allowance_per_trial": synthesis_allowance,
        "configured_conditional_success_lower_per_trial": lower,
        "sampling_failure_allowance": sampling_failure_allowance,
        "physical_failure_allowance_all_trials": physical_failure_allowance,
        "total_failure_target": sampling_failure_allowance + physical_failure_allowance,
        "trial_cap": required,
        "physical_failure_allowance_per_trial": physical_failure_allowance / required,
        "ideal_expected_stopped_trials": stopped_trial_expectation(ideal_success, required),
        "configured_conditional_expected_stopped_trials_upper": stopped_trial_expectation(lower, required),
        "configured_conditional_miss_probability": miss,
        "configured_conditional_full_failure": miss + physical_failure_allowance,
        "certified_confidence_lower_bound": None,
        "success_model_assumption": "Synthesis model scalar upper-bounds output-distribution total variation; not certified by the mean fit",
        "stopping_assumptions": "Fresh IID trials, serial reuse, correct immediate classical equality verification; verification/control time unmeasured",
        "timing_status": "Capped and expected trial counts multiply QDK modeled per-trial time, not a certified worst-case wall time",
    }


def build_logical_case(n_qubits: int) -> tuple:
    """Materialize preparation, equality predicate, diffusion, and final readout."""
    from qdk.qre import instruction_name
    from qdk.qre.application import OpenQASMApplication

    started = time.perf_counter()
    marked = equality_target(n_qubits)
    iterations = optimal_iterations(n_qubits)
    circuit = build_grover_circuit(n_qubits, marked, iterations, measure=True)
    compiled = prepare_measured_circuit(circuit)
    qasm = qasm3.dumps(compiled)
    trace = next(OpenQASMApplication(qasm).enumerate_traces())
    counts = {instruction_name(key): value for key, value in trace.gate_counts.items()}
    state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    ideal = expected_success_probability(n_qubits, iterations)
    simulated = float(state.probabilities()[marked])
    if abs(simulated - ideal) > 1e-10:
        raise RuntimeError("Independent ideal statevector disagrees with Grover's analytic probability")
    compiled_state = Statevector.from_instruction(compiled.remove_final_measurements(inplace=False))
    if not state.equiv(compiled_state, atol=1e-10):
        raise RuntimeError("Logical lowering changed the ideal state")
    problem = {
        "algorithm": "Grover single-target equality search",
        "n_qubits": n_qubits,
        "number_of_items": 2**n_qubits,
        "marked": marked,
        "marked_bitstring": f"{marked:0{n_qubits}b}",
        "number_of_marked_states": 1,
        "iterations": iterations,
        "target_family": "Alternating q[even]=1, q[odd]=0; n=3 is the original target 101",
    }
    raw = dict(circuit.count_ops())
    lowered = dict(compiled.count_ops())
    logical = {
        "problem": problem,
        "input_sha256": content_sha256(problem),
        "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "oracle_calls_per_trial": iterations,
        "diffusion_calls_per_trial": iterations,
        "preparation": "Explicit uniform-superposition Hadamards on all n input qubits",
        "oracle": "Explicit equality phase predicate with X masks and a decomposed multi-controlled phase",
        "source_gate_counts": raw,
        "decomposed_gate_counts": lowered,
        "decomposed_unitary_gate_count": sum(v for k, v in lowered.items() if k not in {"barrier", "measure"}),
        "decomposed_depth": compiled.depth(),
        "algorithm_qubits": circuit.num_qubits,
        "compiled_qubits": compiled.num_qubits,
        "output_measurements_per_trial": lowered.get("measure", 0),
        "canonical_gate_counts": counts,
        "canonical_application_trace": str(trace),
        "ideal_success_probability": ideal,
        "statevector_success_probability": simulated,
        "statevector_checked": True,
        "lowered_state_checked_up_to_global_phase": True,
        "classical_generation_lowering_reference_check_time_s": time.perf_counter() - started,
        "included_quantum_work": ["initial preparation", "explicit equality oracle", "diffusion", "ancillas if compiler introduces them", "final readout"],
        "classical_verification_time_s": None,
        "classical_control_and_external_loading_time_s": None,
    }
    if logical["output_measurements_per_trial"] != n_qubits:
        raise RuntimeError("All input qubits must have final readout")
    return circuit, qasm, logical


def add_execution_metrics(point: dict, contract: dict, actual_synthesis: float) -> dict:
    """Retain capped and expected timing without multiplying serial peak qubits."""
    ideal = contract["ideal_success_probability_per_trial"]
    lower = max(0.0, ideal - actual_synthesis)
    if lower <= 0:
        raise ValueError("Actual conditional success bound is zero")
    cap = contract["trial_cap"]
    miss = (1 - lower) ** cap
    expected_ideal = stopped_trial_expectation(ideal, cap)
    expected_upper = stopped_trial_expectation(lower, cap)
    physical = point["physical_failure_union_bound_all_executions"]
    full = miss + physical
    if full > contract["total_failure_target"] + 1e-12:
        raise RuntimeError("Physical point violates the conditional whole-run allocation")
    q = point["peak_physical_qubits"]
    t = point["runtime_per_execution_ns"]
    capped_t = cap * t
    return {
        **point,
        "actual_synthesis_model_error_per_trial": actual_synthesis,
        "actual_conditional_success_lower_per_trial": lower,
        "conditional_miss_probability_at_cap": miss,
        "conditional_full_failure_model": full,
        "conditional_confidence_model": 1 - full,
        "certified_confidence_lower_bound": None,
        "trial_cap": cap,
        "ideal_expected_stopped_trials": expected_ideal,
        "conditional_expected_stopped_trials_upper": expected_upper,
        "capped_quantum_model_runtime_ns": capped_t,
        "ideal_expected_quantum_model_runtime_ns": expected_ideal * t,
        "conditional_expected_quantum_model_runtime_upper_ns": expected_upper * t,
        "peak_times_capped_runtime_qubit_ns": q * capped_t,
        "peak_times_ideal_expected_runtime_qubit_ns": q * expected_ideal * t,
        "peak_times_conditional_expected_runtime_upper_qubit_ns": q * expected_upper * t,
        "spacetime_metric": "Peak physical footprint times modeled runtime; not an integrated occupation schedule",
        "runtime_status": "Mean-fit/layout model; no certified runtime-tail or worst-case wall-time bound",
    }


def frozen_fingerprints() -> dict[str, str]:
    """Protect canonical frozen data/QASM and the original source, not page layout."""
    names = [
        "scripts/estimate_resources.py", "src/openquantum/grover.py",
        "assets/generated/grover/circuit.qasm",
        "assets/generated/resources/grover-surface-code.json",
        "assets/generated/resources/grover-surface-code.csv",
    ]
    return {name: file_sha256(ROOT / name) for name in names}


def guard_output_directory(output: Path) -> None:
    """Reject a target within frozen baseline asset directories before any write."""
    target = output.resolve()
    for name in ("assets/generated/grover", "assets/generated/resources"):
        protected = (ROOT / name).resolve()
        if target == protected or target.is_relative_to(protected):
            raise ValueError("Scaling output must not overlap frozen Grover baseline assets")


def provenance() -> dict:
    code_commit = git_revision(ROOT)
    source_paths = [
        "scripts/generate_grover_scaling.py", "src/openquantum/grover.py",
        "src/openquantum/resource_estimation.py", "requirements-lock.txt",
    ]
    return {
        "code_commit": code_commit,
        "source_sha256": {p: file_sha256(ROOT / p) for p in source_paths if (ROOT / p).exists()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine_architecture": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in ["qdk", "qiskit", "numpy", "matplotlib", "pandas"]},
        "simulation": "CPU ideal Statevector, at most eight qubits; no random sampling",
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7, "parameters_fully_bound": True},
    }


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_tables(output: Path, report: dict) -> None:
    logical_rows, physical_rows, summary_rows = [], [], []
    for case in report["cases"]:
        logical = case["logical"]
        contract = case["trial_contract"]
        n = logical["problem"]["n_qubits"]
        logical_rows.append({
            "n_qubits": n, "items": 2**n, "marked": logical["problem"]["marked"],
            "iterations": logical["problem"]["iterations"],
            "oracle_calls_per_trial": logical["oracle_calls_per_trial"],
            "primitive_unitary_gates_per_trial": logical["decomposed_unitary_gate_count"],
            "cx_per_trial": logical["decomposed_gate_counts"].get("cx", 0),
            "canonical_T_per_trial": logical["canonical_gate_counts"].get("T", 0),
            "canonical_RZ_per_trial": logical["canonical_gate_counts"].get("RZ", 0),
            "measurements_per_trial": logical["output_measurements_per_trial"],
            "trial_cap": contract["trial_cap"],
            "ideal_success": logical["ideal_success_probability"],
            "qasm_sha256": logical["qasm_sha256"],
        })
        for profile in case["profiles"]:
            common = {
                "n_qubits": n, "physical_error_rate": profile["profile"]["physical_error_rate"],
                "profile_status": profile["status"], "trial_cap": contract["trial_cap"],
            }
            if profile["status"] != "computed_conditional_model":
                physical_rows.append({**common, "point_index": "", "pending_reason": profile.get("reason", "")})
                summary_rows.append({**common, "choice": "pending", "pending_reason": profile.get("reason", "")})
                continue
            points = profile["estimate"]["pareto_points"]
            for i, point in enumerate(points):
                physical_rows.append({**common, "point_index": i, **{
                    k: point[k] for k in [
                        "peak_physical_qubits", "data_code_distance", "runtime_per_execution_ns",
                        "capped_quantum_model_runtime_ns", "ideal_expected_quantum_model_runtime_ns",
                        "conditional_expected_quantum_model_runtime_upper_ns",
                        "peak_times_capped_runtime_qubit_ns", "conditional_full_failure_model",
                        "actual_synthesis_model_error_per_trial", "actual_conditional_success_lower_per_trial",
                        "ideal_expected_stopped_trials", "conditional_expected_stopped_trials_upper",
                    ]
                }})
            selectors = {
                "fewest_qubits": lambda p: (p["peak_physical_qubits"], p["capped_quantum_model_runtime_ns"]),
                "fastest_capped_model": lambda p: (p["capped_quantum_model_runtime_ns"], p["peak_physical_qubits"]),
                "smallest_peak_time_proxy": lambda p: (p["peak_times_capped_runtime_qubit_ns"], p["peak_physical_qubits"]),
            }
            for label, key in selectors.items():
                selected = min(points, key=key)
                summary_rows.append({**common, "choice": label, **{
                    k: selected[k] for k in [
                        "peak_physical_qubits", "data_code_distance", "capped_quantum_model_runtime_ns",
                        "ideal_expected_quantum_model_runtime_ns", "conditional_expected_quantum_model_runtime_upper_ns",
                        "peak_times_capped_runtime_qubit_ns", "conditional_full_failure_model",
                    ]
                }})
    for name, rows in [("logical-scaling.csv", logical_rows), ("physical-pareto.csv", physical_rows), ("physical-summary.csv", summary_rows)]:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "| n | Physical error | Selection | Trial cap | Peak qubits | Capped model time (ms) | Ideal stopped mean time (ms) | Peak × capped time (qubit s) | Conditional failure model |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["choice"] == "smallest_peak_time_proxy":
            continue
        if row["choice"] == "pending":
            lines.append(f"| {row['n_qubits']} | {row['physical_error_rate']:g} | Pending physical model | {row['trial_cap']} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {row['n_qubits']} | {row['physical_error_rate']:g} | {row['choice'].replace('_', ' ')} | "
            f"{row['trial_cap']} | {row['peak_physical_qubits']:,} | {row['capped_quantum_model_runtime_ns']/1e6:.6g} | "
            f"{row['ideal_expected_quantum_model_runtime_ns']/1e6:.6g} | {row['peak_times_capped_runtime_qubit_ns']/1e9:.6g} | "
            f"{row['conditional_full_failure_model']:.6g} |"
        )
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "openquantum-grover-scaling-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sizes = [row["n_qubits"] for row in logical_rows]
    axes[0].plot(sizes, [row["oracle_calls_per_trial"] for row in logical_rows], "o-", color="#176a69")
    axes[0].set(xlabel="Input qubits n", ylabel="Oracle calls per trial", title="Explicit Grover repetitions")
    for key, label in [("primitive_unitary_gates_per_trial", "All primitive unitary gates"), ("cx_per_trial", "CX gates")]:
        axes[1].semilogy(sizes, [row[key] for row in logical_rows], "o-", label=label)
    axes[1].set(xlabel="Input qubits n", ylabel="Compiled gates per trial", title="Actual preparation + oracle + diffusion")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(sizes)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "logical-scaling.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    fields = [
        ("peak_physical_qubits", 1, "Peak physical qubits"),
        ("capped_quantum_model_runtime_ns", 1e6, "Capped model runtime (ms)"),
        ("peak_times_capped_runtime_qubit_ns", 1e9, "Peak × capped time (qubit s)"),
    ]
    colors = {0.001: "#176a69", 0.0001: "#a34f24"}
    for error in sorted({r["physical_error_rate"] for r in summary_rows}, reverse=True):
        for choice, style in [("fewest_qubits", "o-"), ("fastest_capped_model", "s--")]:
            rows = sorted((r for r in summary_rows if r["physical_error_rate"] == error and r["choice"] == choice), key=lambda r:r["n_qubits"])
            for ax, (field, scale, label) in zip(axes, fields):
                if rows:
                    ax.semilogy([r["n_qubits"] for r in rows], [r[field]/scale for r in rows],
                                style, color=colors.get(error), label=f"p={error:g}, {choice.replace('_',' ')}")
                ax.set(xlabel="Input qubits n", ylabel=label)
                ax.set_xticks(sizes)
                ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Conditional surface-code scaling of the explicit equality-oracle circuit", fontsize=11)
    fig.text(.5, .01, "Capped serial trials; modeled synthesis/layout work; no certified wall-time or approximation bound", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(output / "physical-scaling.svg")
    plt.close(fig)


def run_study(
    output: Path, cache_dir: Path, sizes: list[int], *, synthesis_allowance: float = 1e-4,
    max_trials: int = 10_000, logical_only: bool = False,
) -> dict:
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Choose distinct sizes from n=2..8")
    for n in sizes:
        equality_target(n)
    if not math.isfinite(synthesis_allowance) or synthesis_allowance <= 0:
        raise ValueError("Arbitrary-rotation synthesis allowance must be finite and positive")
    guard_output_directory(output)
    protected_before = frozen_fingerprints()
    baseline_path = ROOT / "assets/generated/resources/grover-surface-code.json"
    baseline = json.loads(baseline_path.read_text())
    frozen_budget = baseline["error_budget"]
    sampling = frozen_budget["sampling_failure_allowance"]
    physical = frozen_budget["physical_failure_allowance"]
    profiles = [
        SurfaceCodeProfile(
            name=item["id"], physical_error_rate=item["physical_error_rate"],
            gate_time_ns=item["gate_time_ns"], measurement_time_ns=item["measurement_time_ns"],
        )
        for item in baseline["profiles"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "study": "Explicit equality-oracle Grover scaling n=2..8",
        "sizes": sorted(sizes),
        "provenance": provenance(),
        "frozen_baseline_sha256": file_sha256(baseline_path),
        "frozen_file_sha256_before": protected_before,
        "frozen_baseline_instance": baseline["instance"],
        "failure_contract": {
            "sampling_failure_allowance": sampling, "physical_failure_allowance": physical,
            "total_failure_target": sampling + physical, "serial_trials": True,
        },
        "limitations": [
            "Known-target equality is a pedagogical oracle, not a practical database benchmark or quantum advantage demonstration",
            "The reported boundary includes all explicit quantum preparation/oracle/diffusion/readout work and serial repetitions",
            "Classical verification and control latency, external data access, and hardware-specific routing/decoder engineering are not timed",
            "Generation, lowering, ideal reference checking, and estimator wall times are CPU diagnostics, not physical execution times",
            "Mean-fit arbitrary-rotation costs supply no concrete synthesis sequence, per-instance certificate, or runtime-tail guarantee",
            "Conditional success assumes synthesis scalar bounds output TV; certified confidence remains unspecified",
            "QDK layout/scheduling abstracts Clifford work; modeled runtime is not an exact schedule of the source gates",
            "Peak-footprint times runtime is a spacetime proxy, not an integrated active-qubit schedule",
            "Expected stopping assumes independent fresh trials and immediate correct equality verification",
            "The original n=3 report remains frozen; this extension uses the audited explicit transform and unpruned declared model search",
        ],
        "cases": [],
    }
    for n in sorted(sizes):
        circuit, qasm, logical = build_logical_case(n)
        case_dir = output / f"n{n}"
        case_dir.mkdir(exist_ok=True)
        (case_dir / "circuit.qasm").write_text(qasm, encoding="utf-8")
        logical["matches_frozen_n3_qasm"] = logical["qasm_sha256"] == baseline["qasm_sha256"] if n == 3 else None
        allowance = synthesis_allowance if logical["canonical_gate_counts"].get("RZ", 0) else 0.0
        contract = trial_contract(logical["ideal_success_probability"], allowance, sampling, physical, max_trials=max_trials)
        write_json(case_dir / "logical.json", logical)
        entry = {"logical": logical, "trial_contract": contract, "profiles": []}
        report["cases"].append(entry)
        for profile in profiles:
            profile_entry = {"profile": {
                "name": profile.name, "physical_error_rate": profile.physical_error_rate,
                "gate_time_ns": profile.gate_time_ns, "measurement_time_ns": profile.measurement_time_ns,
                "qec_model": "SurfaceCode", "factory_model": "RoundBasedFactory",
            }}
            entry["profiles"].append(profile_entry)
            if logical_only:
                profile_entry.update(status="not_requested", reason="Logical-only run; physical costs remain uncomputed")
            else:
                started = time.perf_counter()
                try:
                    estimated = estimate_circuit_resources(
                        circuit,
                        budget=CircuitResourceBudget(allowance, physical, executions=contract["trial_cap"]),
                        profile=profile, cache_dir=cache_dir / profile.name, use_graph=False,
                    )
                    if estimated["qasm_sha256"] != logical["qasm_sha256"]:
                        raise RuntimeError("Physical report circuit differs from the checked logical export")
                    actual = estimated["synthesis"]["model_error_per_execution"]
                    if actual > allowance + 1e-14:
                        raise RuntimeError("Synthesis model exceeds the configured allowance")
                    estimated["pareto_points"] = [
                        add_execution_metrics(point, contract, actual) for point in estimated["pareto_points"]
                    ]
                    estimated.pop("qasm3")
                    estimated["qasm_relative_path"] = "circuit.qasm"
                    profile_entry.update(status="computed_conditional_model", estimate=estimated)
                except Exception as exc:
                    profile_entry.update(status="pending_physical_model", reason=f"{type(exc).__name__}: {exc}")
                profile_entry["classical_estimator_wall_time_s"] = time.perf_counter() - started
            write_json(case_dir / f"{profile.name}.json", profile_entry)
            write_json(output / "grover-scaling.json", report)
            print(f"n={n}, profile={profile.name}: {profile_entry['status']}", flush=True)
    protected_after = frozen_fingerprints()
    if protected_before != protected_after:
        raise RuntimeError("Frozen Grover baseline files changed during the study")
    report["frozen_file_sha256_after"] = protected_after
    report["frozen_files_unchanged"] = True
    report["status"] = "computed_conditional_model" if all(
        p["status"] == "computed_conditional_model" for c in report["cases"] for p in c["profiles"]
    ) else "partial_physical_model"
    write_json(output / "grover-scaling.json", report)
    write_tables(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="2,3,4,5,6,7,8")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/grover-scaling")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "openquantum-grover-scaling-qre-cache")
    parser.add_argument("--synthesis-allowance", type=float, default=1e-4)
    parser.add_argument("--max-trials", type=int, default=10_000)
    parser.add_argument("--logical-only", action="store_true")
    args = parser.parse_args()
    report = run_study(
        args.output_dir, args.cache_dir, [int(x) for x in args.sizes.split(",")],
        synthesis_allowance=args.synthesis_allowance, max_trials=args.max_trials, logical_only=args.logical_only,
    )
    print(json.dumps({"status": report["status"], "sizes": report["sizes"], "frozen_files_unchanged": report["frozen_files_unchanged"]}))
    return 0 if report["status"] == "computed_conditional_model" or args.logical_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
