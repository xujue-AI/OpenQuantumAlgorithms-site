#!/usr/bin/env python3
"""Generate bounded complete-instrument Clifford+T and physical-resource reports.

No hardware run, package installation, publication, or frozen-baseline edit.
Use --case to regenerate one task. All resource budgets cover complete attempted
trials and all required magic states, including rejected algorithm trials.
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
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REPOSITORY = (ROOT/"src/openquantum").is_dir() and HERE == ROOT/"scripts"
if REPOSITORY:
    sys.path.insert(0, str(ROOT/"src"))
    from openquantum.instrument_resources import InstrumentProfile, estimate_instrument, simulate_trial, synthesize_instrument
    from openquantum.simulation.open_systems import build_amplitude_damping, input_density, observable_shots
    from openquantum.simulation.lindblad import build_lindblad_circuit, exact_density
    from openquantum.simulation.thermal_filter import DEFAULT_TERMS, build_thermal_trial, exact_gibbs_state
else:
    from instrument_resources import InstrumentProfile, estimate_instrument, simulate_trial, synthesize_instrument
    from open_systems import build_amplitude_damping, input_density, observable_shots
    from lindblad import build_lindblad_circuit, exact_density
    from thermal_filter import DEFAULT_TERMS, build_thermal_trial, exact_gibbs_state

import numpy as np
from qiskit import qasm3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_NAMES = ("damping-Z", "lindblad-Z", "thermal-Z", "thermal-X")
TASK_FILES = ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json")
SOURCE_NAMES = ("instrument_resources.py", "generate_instrument_resources.py", "open_systems.py", "lindblad.py", "thermal_filter.py", "lcu.py")
FROZEN_FILES = ("src/openquantum/grover.py", "scripts/estimate_resources.py",
                "assets/generated/grover/circuit.qasm",
                "assets/generated/resources/grover-surface-code.json",
                "assets/generated/resources/grover-surface-code.csv")


def source_files():
    if not REPOSITORY:
        return {name: HERE/name for name in SOURCE_NAMES}
    return {"instrument_resources.py": ROOT/"src/openquantum/instrument_resources.py",
            "generate_instrument_resources.py": Path(__file__),
            **{name: ROOT/"src/openquantum/simulation"/name for name in SOURCE_NAMES[2:]}}


def source_revision(root, paths):
    """Optional exact-root Git provenance, with no ancestor-HEAD fallback."""
    record = {"source_commit_at_generation": None, "dirty_source_paths": None,
              "revision_note": "Source SHA-256 identities are authoritative; Git revision is optional and records only a verified exact project root."}
    root = Path(root).resolve()
    if not ((root/".git").is_dir() or (root/".git").is_file()):
        return record
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), *args],
                                       text=True, stderr=subprocess.DEVNULL, env=environment, timeout=5).strip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            return record
        revision = git("rev-parse", "--verify", "HEAD")
        if len(revision) not in (40, 64) or any(character not in "0123456789abcdef" for character in revision):
            return record
        relative = [str(Path(path).resolve().relative_to(root)) for path in paths]
        record.update(source_commit_at_generation=revision,
                      dirty_source_paths=git("status", "--porcelain=v1", "--", *relative).splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return record


def frozen_fingerprints(root):
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else None
            for name in FROZEN_FILES}


def validate_paths(output, cache, *, root=ROOT, repository=REPOSITORY, sources=None):
    """Restrict repo writes to this artifact tree and keep the cache private."""
    output, cache, root = Path(output).resolve(), Path(cache).resolve(), Path(root).resolve()
    owned = root/"assets/generated/instrument-resources"
    if repository and output.is_relative_to(root) and not output.is_relative_to(owned):
        raise ValueError("Repository output must be inside assets/generated/instrument-resources; frozen assets and source directories are protected")
    if cache.is_relative_to(output) or output.is_relative_to(cache):
        raise ValueError("Factory cache and public output must be disjoint directories")
    if repository and cache.is_relative_to(root):
        raise ValueError("Private factory cache must remain outside the source checkout")
    for path in (sources or source_files()).values():
        if Path(path).resolve().is_relative_to(output):
            raise ValueError("Output may not contain or overwrite an input source")
    managed = [output/name/file for name in TASK_NAMES for file in TASK_FILES]
    managed += [output/name for name in (*SOURCE_NAMES, "LICENSE.txt", "references.bib", "requirements-lock.txt", "manifest.json", "summary.csv", "summary.md", "README.md", "space-time.svg")]
    if any(path.is_symlink() or path.parent.is_symlink() for path in managed):
        raise ValueError("Managed artifact targets must not be symbolic links")
    return output, cache


def clean_stale_tasks(output, selected):
    """Remove only known managed records omitted from a partial regeneration."""
    for name in set(TASK_NAMES)-set(selected):
        folder = output/name
        for filename in TASK_FILES:
            (folder/filename).unlink(missing_ok=True)
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def readme(selected):
    return f'''# Complete measured instruments: conditional physical models

This bundle computes {", ".join(selected)}. It contains actual Clifford+T
circuits, deterministic small density-matrix checks, explicit operation ledgers
and conditional physical workloads. The shot ensembles and hardware schedules
have **not** been executed. A guarded floating synthesis error is a numerical
proposal, **not** a formal interval or combined confidence certificate.

## Task and construction

The fixed damping-Z task measures the exact finite-time one-qubit absorbing
channel at rate 0.8 and time 1 from a gate-prepared plus state. Lindblad-Z uses
32 positive-time Strang steps by default with H=0.7X+0.3Z and fresh environment
resets. Its reference is the independent matrix-exponential Liouvillian answer.
The thermal X/Z tasks use the explicitly constructed degree-6 Taylor LCU
filter for H=0.6Z+0.4X at beta=1.25, Bell input preparation, actual PREPARE/SELECT
and uncomputation, label readout, reference discard and system readout. The
weighted energy output is 0.6 mean(Z)+0.4 mean(X), from separate groups.
See each report's instance field; --lindblad-steps changes the fixed step count.

Resets and measurements remain in the compiled schedule. SABRE uses a fixed
logical-line layout, seed 7 and optimization level 0. Actual ancilla-free
Ross--Selinger approximants replace every remaining rotation; the algorithm
uses Qiskit's gridsynth_unitary. Its native rsgridsynth backend exposes no seed
parameter. The generated sequences and SHA-256 identities, rather than a
promised seeded rerun, define the saved circuit. Per-rotation Frobenius errors
plus a 1e-12 numerical guard are summed. This is not formal roundoff analysis.
Exported QASM omits global phase and is a measured-trial export; it cannot be
substituted for a phase-preserving controlled oracle.

## Full attempted workload and error allocation

Hoeffding's inequality sets accepted sample counts for one +/-1 Pauli mean
with absolute statistical error 0.03. A fixed schedule runs every rejected
trial in full. Thermal postselection uses the numerical lower proposal
(sqrt(p_ideal)-eta)^2 and conditional mean-bias proposal 4 eta/sqrt(p_ideal).
The conditional budgets separately include statistical failure, acceptance
failure, physical faults and factory availability. X/Z group budgets add to
0.01 for the energy task. Reports include the ideal algorithm discrepancy,
actual synthesized answer and proposed synthesis bias. These estimates rely
on stated perturbation and physical assumptions; no formal success guarantee
or classical advantage is claimed.

## Space and time model

This is a **separate serialized instrument scenario**, not the frozen
Grover/PSSPC surface-code baseline or a co-design improvement. For n logical
source wires and odd distance d, reserve 4n-1 data/workspace/bridge patches,
each of area 2d^2-1. Peak qubits add one QDK native factory's qubits. At fixed
distance the reserved data space is linear in n, but materialized gate counts,
transport, preparation and success depend on the algorithm. This bounded
bundle does not demonstrate an asymptotic Gibbs or Lindblad complexity claim.

Each operation's ledger charges its serialized time, logical preparation,
readouts and controller decisions. Neighbor CNOTs use three protected Pauli
parity slots; nonlocal source CNOTs are routed to actual neighbor operations.
Each T uses native factory supply, capped complete factory retries, neighboring
SWAP transport, injection CNOT, Z readout, conditional S and port initialization.
T-dagger also has a charged inverse Clifford-frame decision. A factory port
residence term is added outside native supply time. QDK's output-quality model
already includes its quantum factory operations and is not charged again as
factory-memory error. The source-level 1% readiness requirement is separated
from output quality; exact decimal-rational arithmetic selects the smallest
retry cap in 1..16 covering every required state.

For M accepted slots and complete-trial cap r, A=M*r attempts are charged.
Total runtime is A times the sum of count(operation)*time(operation), and
data memory exposure uses (4n-1)*ceil(time/cycle) per attempt. Factory states
and all event failure contributions are multiplied by A. The 100 ns gates and
initializations, 500 ns measurements, 1000 ns code cycles, 10000 ns controller
reaction, controller error 1e-15/decision, separate initialization residual,
independent majority-of-disjoint-paths readout and memory-distance fit are
explicit scenarios, not calibrated hardware parameters. Data/factory output
faults are union-bounded; factory readiness has its own budget. Direct native
QDK Trace accounting is checked against the ledger for time and error.

Tracked Clifford frames, protected Pauli bases, T-axis orientation and
factory/data handoff are **conditional geometry/interface assumptions, not
a layout certificate**. This does not model arbitrary source-level if/else or
adaptive quantum feedback. It supports fixed schedules plus the stated reset,
T correction and factory/trial dispatch macros.

## Scope, limitations and reproduction

Source limits are five qubits and 10000 operations; actual synthesis permits
100000 operations. Synthesis allowances are between 1e-9 and 0.1, subject to
the per-gate numerical guard. Dense instrument checks are separately bounded
by that same five-qubit width. Distances are odd through 49 by default and at
most 61; no unbounded GPU job is used. The four small tasks are simulation
checks. The large reported sample/retry workloads are calculated, unexecuted.
Offline coefficient generation, synthesis and final statistical-report wall
time are outside quantum runtime; actual synthesis time is recorded separately.
Decoder throughput, interface certification, formal synthesis proofs and
arbitrary adaptive control remain unavailable.

In the private source checkout, with the locked Python 3.12 environment:

```bash
PYTHONPATH=src python scripts/generate_instrument_resources.py
PYTHONPATH=src python -m pytest tests/test_instrument_resources.py
```

For standalone use, download all six Python files in this directory, the
project requirements-lock.txt and LICENSE.txt into one directory, then run
python generate_instrument_resources.py. --case damping, lindblad or thermal
replaces this bundle's managed task records and summaries with that subset;
unrelated user files are preserved. Use a separate output directory to keep
multiple selections. The private factory cache must remain outside output.
manifest.json records package/Python/device details, optional verified-root
Git revision, exact source hashes and available before/after frozen-baseline
fingerprints. A no-Git download records null revision rather than an ancestor's
HEAD. Local paths, credentials and remote coordinates are not in the bundle.

## References and licenses

Original implementation and generated text are MIT-licensed; see LICENSE.txt.
references.bib credits Litinski (2019), Horsman et al. (2012), Ross and Selinger
(2014), Qiskit 2.6.2 (Apache-2.0), rsgridsynth 0.2.2 (MIT, as pinned by Qiskit's
Cargo.lock) and Microsoft QDK 1.34.0 (MIT). The installed wheel does not directly
certify its native crate version. QDK source semantics were audited at a
separate recorded commit; it is not claimed as the release commit. Exact source
URLs and reference keys are also stored in the physical reports.
'''


def jsonable(value):
    if isinstance(value, np.ndarray):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")


def tasks(selected, steps):
    Z = np.diag([1, -1])
    X = np.array([[0, 1], [1, 0]])
    if selected in ("all", "damping"):
        yield "damping-Z", build_amplitude_damping(observable="Z", reset=True), 0, {}, .001, 1.0, 1-math.exp(-.8), {"rate": .8, "time": 1, "input": "plus", "observable": "Z"}
    if selected in ("all", "lindblad"):
        reference = exact_density(input_density("plus_i"))
        yield "lindblad-Z", build_lindblad_circuit(steps=steps, observable="Z"), 0, {}, .001, 1.0, float(np.trace(Z@reference).real), {"rate": .8, "hx": .7, "hz": .3, "time": 1, "steps": steps, "order": 2, "input": "plus_i", "observable": "Z"}
    if selected in ("all", "thermal"):
        reference = exact_gibbs_state(DEFAULT_TERMS, 1.25)
        for axis, operator, weight in (("Z", Z, .6), ("X", X, .4)):
            circuit = build_thermal_trial(measurement=axis)
            labels = circuit.metadata["n_index"]
            yield f"thermal-{axis}", circuit, labels, dict.fromkeys(range(labels), 0), .0001, weight, float(np.trace(operator@reference).real), {"Hamiltonian": ".6Z+.4X", "beta": 1.25, "degree": 6, "observable": axis, "energy_weight": weight}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "damping", "lindblad", "thermal"), default="all")
    parser.add_argument("--lindblad-steps", type=int, default=32)
    parser.add_argument("--maximum-code-distance", type=int, default=49)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/instrument-resources" if REPOSITORY else HERE/"generated-instrument-resources")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir())/"instrument-resource-factory-cache")
    args = parser.parse_args()
    if not 1 <= args.lindblad_steps <= 64:
        parser.error("This artifact generator is bounded to 1..64 Lindblad steps")
    sources = source_files()
    try:
        output, cache = validate_paths(args.output_dir, args.cache_dir, sources=sources)
    except ValueError as error:
        parser.error(str(error))
    frozen_before = frozen_fingerprints(ROOT) if REPOSITORY else {name: None for name in FROZEN_FILES}
    revision = source_revision(ROOT, list(sources.values())+[ROOT/"references/resources.bib", ROOT/"requirements-lock.txt"]) if REPOSITORY else source_revision(HERE, list(sources.values()))
    output.mkdir(parents=True, exist_ok=True)
    reports, rows = [], []
    for name, source, output_bit, acceptance, allowance, weight, exact_mean, instance in tasks(args.case, args.lindblad_steps):
        began = time.perf_counter()
        synthesized = synthesize_instrument(source, allowance)
        synthesis_seconds = time.perf_counter()-began
        ideal = simulate_trial(source, output_bit, acceptance)
        routed = simulate_trial(synthesized.routed, output_bit, acceptance)
        actual = simulate_trial(synthesized.circuit, output_bit, acceptance)
        if abs(ideal["mean"]-routed["mean"]) > 1e-10 or abs(ideal["acceptance_probability"]-routed["acceptance_probability"]) > 1e-10:
            raise AssertionError("Logical routing changed the measured instrument")
        eta = synthesized.report["guarded_numerical_operator_error_upper_estimate"]
        groups = 2 if acceptance else 1
        statistical_failure = .0025/groups
        physical_failure = .0025/groups
        factory_availability_failure = .0025/groups
        acceptance_failure = .0025/groups if acceptance else 0.0
        accepted_samples = observable_shots(.03, statistical_failure)
        if acceptance:
            p_lower = (math.sqrt(ideal["acceptance_probability"])-eta)**2
            if not eta < math.sqrt(ideal["acceptance_probability"]) or p_lower <= 0:
                raise AssertionError("Numerical synthesis estimate cannot support postselection")
            retries = max(1, math.ceil(math.log(acceptance_failure/accepted_samples)/math.log1p(-p_lower)))
            synthesis_bias = min(2.0, 4*eta/math.sqrt(ideal["acceptance_probability"]))
        else:
            p_lower, retries, synthesis_bias = 1.0, 1, 2*eta
        if abs(actual["mean"]-ideal["mean"]) > synthesis_bias+1e-10 or actual["acceptance_probability"] < p_lower-1e-10:
            raise AssertionError("Actual Clifford+T instrument violates the guarded numerical proposal")
        attempts = accepted_samples*retries
        algorithm_bias = abs(ideal["mean"]-exact_mean)
        if algorithm_bias+synthesis_bias+.03 > .05:
            raise AssertionError("This fixed example does not meet its declared .05 error target")
        folder = output/name
        folder.mkdir(exist_ok=True)
        source_qasm, synthesized_qasm = qasm3.dumps(source), qasm3.dumps(synthesized.circuit)
        (folder/"source-instrument.qasm").write_text(source_qasm)
        (folder/"clifford-t-instrument.qasm").write_text(synthesized_qasm)
        report = {
            "task": name, "instance": instance, "output_clbit": output_bit, "acceptance_clbits": acceptance,
            "actual_synthesis": synthesized.report, "ideal_instrument": ideal, "clifford_t_instrument": actual,
            "source_qasm_sha256": hashlib.sha256(source_qasm.encode()).hexdigest(),
            "clifford_t_qasm_sha256": hashlib.sha256(synthesized_qasm.encode()).hexdigest(),
            "classical_synthesis_wall_seconds_this_machine": synthesis_seconds,
            "workload": {"accepted_samples": accepted_samples, "retry_cap_per_accepted_sample": retries,
                         "complete_attempts": attempts, "numerical_acceptance_lower_estimate": p_lower,
                         "postselection_failure_upper_estimate": accepted_samples*(1-p_lower)**retries if acceptance else 0.0,
                         "statistical_failure_budget": statistical_failure, "physical_failure_budget": physical_failure,
                         "factory_availability_budget": factory_availability_failure, "acceptance_failure_budget": acceptance_failure,
                         "conditional_failure_budget_sum": statistical_failure+physical_failure+factory_availability_failure+acceptance_failure,
                         "statistical_absolute_error": .03, "algorithm_bias_numerical": algorithm_bias,
                         "synthesis_bias_guarded_numerical_upper_estimate": synthesis_bias,
                         "conditional_absolute_error_upper_estimate": algorithm_bias+synthesis_bias+.03,
                         "formal_confidence_certificate": False, "measured_hardware_run": False,
                         "scope": "one fixed mean; thermal X/Z groups are combined with declared weights and separate budgets"},
            "physical_profiles": [],
        }
        for error in (1e-3, 1e-4):
            resource = estimate_instrument(synthesized, attempts,
                                          profile=InstrumentProfile(physical_error_rate=error, maximum_code_distance=args.maximum_code_distance),
                                          physical_failure_budget=physical_failure,
                                          factory_availability_budget=factory_availability_failure,
                                          cache_dir=cache)
            report["physical_profiles"].append(resource)
            minimum = min(resource["pareto_points"], key=lambda p: p["peak_physical_qubits"])
            rows.append({"task": name, "physical_error": error, "data_code_distance": minimum["data_code_distance"],
                         "peak_physical_qubits": minimum["peak_physical_qubits"],
                         "runtime_all_attempts_hours": minimum["worst_case_runtime_all_attempts_ns"]/3.6e12,
                         "complete_attempts": attempts, "accepted_samples": accepted_samples,
                         "actual_t_and_tdg_per_attempt": sum(synthesized.circuit.count_ops().get(g, 0) for g in ("t", "tdg")),
                         "full_physical_failure_model": minimum["full_physical_failure_model"],
                         "factory_availability_failure_model": minimum["factory_supply"]["availability_failure_union_bound"]})
            print(f"Computed {name}, p={error:g}: {len(resource['pareto_points'])} retained points", flush=True)
        write_json(folder/"report.json", report)
        reports.append(report)
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    hashes = {}
    for name, path in sources.items():
        data = path.read_bytes()
        (output/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    license_path = ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    references_path = ROOT/"references/resources.bib" if REPOSITORY else HERE/"references.bib"
    environment_path = ROOT/"requirements-lock.txt" if REPOSITORY else HERE/"requirements-lock.txt"
    for name, path in (("LICENSE.txt", license_path), ("references.bib", references_path), ("requirements-lock.txt", environment_path)):
        if path.is_file():
            data = path.read_bytes()
            (output/name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
    frozen_after = frozen_fingerprints(ROOT) if REPOSITORY else {name: None for name in FROZEN_FILES}
    if frozen_before != frozen_after:
        raise AssertionError("Available frozen Grover baseline fingerprints changed during generation")
    manifest = {"status": "complete measured-instrument conditional physical models computed",
                "models": [r["task"] for r in reports], "source_sha256": hashes,
                "task_artifact_sha256": {r["task"]: {name: hashlib.sha256((output/r["task"]/name).read_bytes()).hexdigest() for name in TASK_FILES} for r in reports},
                "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "qdk", "numpy", "scipy")},
                "runtime_environment": {"python": platform.python_version(), "implementation": platform.python_implementation(),
                                        "operating_system": platform.system(), "architecture": platform.machine(),
                                        "device": "CPU; deterministic small density-matrix checks; no GPU/hardware job", "random_seed": None,
                                        "seed_scope": "SABRE seed 7; gridsynth_unitary has no exposed synthesis seed"},
                **revision,
                "frozen_grover_baseline_changed": False if any(frozen_before.values()) else None,
                "frozen_grover_baseline_fingerprints": {"before": frozen_before, "after": frozen_after,
                                                       "scope": "Only available canonical source/QASM/resource files are checked; absent files are null."},
                "summary_scope": "serialized line-layout and factory-supply baseline; not hardware-calibrated or formally certified",
                "online_control": "reset/correction, factory retry and complete-trial accept/counter dispatch costs are charged",
                "offline_classical_work": "actual synthesis wall time recorded separately; coefficient generation/final report wall time not included in conditional quantum runtime"}
    thermal = [report for report in reports if report["task"].startswith("thermal-")]
    if len(thermal) == 2:
        combined = {
            "task": "noncommuting thermal energy .6Z+.4X",
            "complete_attempts": sum(report["workload"]["complete_attempts"] for report in thermal),
            "total_accepted_samples": sum(report["workload"]["accepted_samples"] for report in thermal),
            "conditional_failure_budget_sum": sum(report["workload"]["conditional_failure_budget_sum"] for report in thermal),
            "conditional_energy_error_upper_estimate": sum(report["instance"]["energy_weight"]*report["workload"]["conditional_absolute_error_upper_estimate"] for report in thermal),
            "clifford_t_energy_mean": sum(report["instance"]["energy_weight"]*report["clifford_t_instrument"]["mean"] for report in thermal),
            "physical_profiles": [], "formal_confidence_certificate": False,
        }
        for index, error in enumerate((1e-3, 1e-4)):
            points = [min(report["physical_profiles"][index]["pareto_points"], key=lambda p: p["peak_physical_qubits"]) for report in thermal]
            combined["physical_profiles"].append({
                "physical_error": error,
                "peak_physical_qubits_sequential_groups": max(point["peak_physical_qubits"] for point in points),
                "worst_case_runtime_all_groups_ns": sum(point["worst_case_runtime_all_attempts_ns"] for point in points),
                "full_physical_failure_model_all_groups": sum(point["full_physical_failure_model"] for point in points),
                "factory_availability_failure_model_all_groups": sum(point["factory_supply"]["availability_failure_union_bound"] for point in points),
                "scope": "sequential groups with per-attempt initialization; peak reused hardware, not summed qubits",
            })
        manifest["thermal_energy_workload"] = combined
    write_json(output/"manifest.json", manifest)
    table = ["| Task | Physical error | Distance | Peak physical qubits | Capped runtime (hours) | Complete attempts |",
             "|---|---:|---:|---:|---:|---:|"]
    table.extend(f"| {r['task']} | {r['physical_error']:g} | {r['data_code_distance']} | {r['peak_physical_qubits']:,} | {r['runtime_all_attempts_hours']:.3f} | {r['complete_attempts']:,} |" for r in rows)
    (output/"summary.md").write_text("\n".join(table)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, error in enumerate((1e-3, 1e-4)):
        chosen = [r for r in rows if r["physical_error"] == error]
        x = np.arange(len(chosen))
        axes[0].bar(x+(i-.5)*.35, [r["peak_physical_qubits"] for r in chosen], width=.35, label=f"p={error:g}")
        axes[1].bar(x+(i-.5)*.35, [r["runtime_all_attempts_hours"] for r in chosen], width=.35, label=f"p={error:g}")
    labels = [r["task"] for r in rows if r["physical_error"] == 1e-3]
    for ax, ylabel in zip(axes, ("Peak physical qubits", "Capped full-workload runtime (hours)")):
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Complete instruments · separate serialized surface-code scenario")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    with matplotlib.rc_context({"svg.hashsalt": "openquantum-instrument-resources"}):
        fig.savefig(output/"space-time.svg", metadata={"Date": None})
    plt.close(fig)
    figure = output/"space-time.svg"
    figure.write_text("\n".join(line.rstrip() for line in figure.read_text().splitlines())+"\n")
    (output/"README.md").write_text(readme(manifest["models"]))
    clean_stale_tasks(output, manifest["models"])
    print(f"Wrote bounded instrument reports to {output}")


if __name__ == "__main__":
    main()
