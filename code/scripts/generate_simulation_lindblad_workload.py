#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate predeclared interacting-chain workloads at common ideal accuracy.

Only n=2/3/4, at most five numerical step points per case, four-step whole
circuit checks and256x256 dense channels. Shot ensembles remain unexecuted.
No physical sizing, publication, GPU job or dependency installation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPOSITORY = HERE == ROOT/"scripts" and (ROOT/"src/openquantum/simulation/lindblad_workload.py").is_file()
sys.path.insert(0, str(HERE))
try:
    from source_provenance import git_dirty, git_revision
except ModuleNotFoundError as error:
    if error.name != "source_provenance":
        raise
    def git_revision(_root):
        return None
    def git_dirty(_root, _paths):
        return None
sys.path.insert(0, str(ROOT/"src" if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-lindblad-workload"
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from qiskit import qasm3

if REPOSITORY:
    from openquantum.simulation.lindblad_workload import (ChainTask, FAMILY, MAX_CONVERGENCE_POINTS,
        SPLITTING_ALLOWANCE, compare, full_circuit_density, gate_density, generator_norm_envelope,
        logical_ledger, measured_trial, output_contract, reference_channel, reference_density,
        select_steps, uniform_split_bound)
else:
    from lindblad_workload import (ChainTask, FAMILY, MAX_CONVERGENCE_POINTS,
        SPLITTING_ALLOWANCE, compare, full_circuit_density, gate_density, generator_norm_envelope,
        logical_ledger, measured_trial, output_contract, reference_channel, reference_density,
        select_steps, uniform_split_bound)

FROZEN = ("src/openquantum/grover.py", "scripts/estimate_resources.py",
          "assets/generated/grover/circuit.qasm", "assets/generated/resources/grover-surface-code.json",
          "assets/generated/resources/grover-surface-code.csv")
CASE_FILES = ("source-trial.qasm", "rotation-basis-trial.qasm", "report.json")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, dict):
        return {name: jsonable(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_paths():
    names = ("lindblad_workload.py", "two_spin_lindblad.py", "open_systems.py")
    return {**{name: ROOT/"src/openquantum/simulation"/name if REPOSITORY else HERE/name for name in names},
            "generate_simulation_lindblad_workload.py": Path(__file__).resolve()}


def fingerprints():
    return {name: sha256((ROOT/name).read_bytes()) if REPOSITORY and (ROOT/name).is_file() else None for name in FROZEN}


def validate_output(output, no_sim):
    output = Path(output).resolve()
    owned = ROOT/"assets/generated/simulation/lindblad-workload"
    if REPOSITORY and output.is_relative_to(ROOT) and not output.is_relative_to(owned):
        raise ValueError("In-repository output is restricted to the owned lindblad-workload artifact tree")
    if no_sim and REPOSITORY and output == owned:
        raise ValueError("Use a separate output directory for unexecuted --no-sim artifacts")
    if any(path.resolve().is_relative_to(output) for path in source_paths().values()):
        raise ValueError("Output may not contain input source files")
    targets = [output/f"n{n}"/name for n in FAMILY for name in CASE_FILES]
    targets += [output/name for name in (*source_paths(), "source_provenance.py", "LICENSE.txt", "requirements-lock.txt", "references.bib", "manifest.json", "summary.csv", "convergence.csv", "summary.md", "README.md", "workload-scaling.svg", "convergence.svg", "one-step-circuit.svg")]
    if any(path.is_symlink() or path.parent.is_symlink() for path in targets):
        raise ValueError("Managed output targets must not be symbolic links")
    return output


def readme(selected, performed):
    return f"""# Interacting-chain workloads at common ideal accuracy

This bundle materializes n={", ".join(map(str, selected))} from the predeclared
family n=2, 3, 4. Numerical checks are {"performed" if performed else "unperformed"}.
It preserves the older two-spin r=32 energy study as a separate task. Here the
output is one **local Z0 mean**, not total energy or tomography.

## Fixed problem and output contract

Use hbar=1, time 1, H=0.6 sum ZZ+0.35 sum X on an open chain, local lowering
jumps with rate 0.8 at site 0 and 0.5 elsewhere, and a product input +i on site 0
and + on the others. Coefficients/rates have inverse-time units. Actual H/S
gates prepare the known input; all systems begin with resets. The measured
circuits read all 2n system/environment wires in Z, retaining only system 0's
score +/-1. Other system bits and every environment bit are ignored.

Each requested mean has splitting allowance 0.01 and statistical error 0.03.
The **three-output family** has statistical failure budget 0.005, split into
1/600 per case, giving 15,756 complete trials per case. A union bound controls
all three output errors simultaneously, each at 0.04 in the ideal exact-angle model.
Positive exponential Taylor
terms verify the Hoeffding count with exact rational arithmetic. The large
shot ensembles are calculated and **unexecuted**. Finite-angle roundoff,
coherent synthesis and physical faults do not have a completed confidence
certificate, so no hardware success guarantee is claimed.

## Original uniform step policy, before observing numerical errors

The literal global Strang palindrome contains ZZ/2, X/2, all local damping
halves except the last, the last damping full step, then the reverse halves.
Each component receives one full step's exposure and all durations are positive.
The dissipator convention is L rho L† - 0.5{{L†L,rho}}.

On Hermitian matrices with Schatten trace norm, Hamiltonian generators have
map norm at most twice the Hamiltonian operator norm, 2||H||_infinity;
each lowering dissipator has norm at most 2 gamma.
Thus A=2(n-1)|J|+2n|hx|+2 sum gamma; this is 5.2, 8.1, 11 for n=2, 3, 4. These are
generator-norm **upper bounds**, not vectorized matrix 1-norms, diamond-norm
measurements or the previous reference-exposure diagnostic.

F(s) denotes the reduced-system channel palindrome after environment discard,
not the joint dilation unitary with square-root small-time angles. Each
component has total weight one, giving F'(0)=G=sum L. Palindrome symmetry
gives F''(0)=G^2. For this positive-time product, each exponential is CPTP and contractive.
The product-rule multinomial expansion bounds ||F'''(s)|| by A^3. The exact
E(s)=exp(s sum L) is CPTP and has ||E'''(s)||<=A^3. The palindrome matches
F/E derivatives through degree 2 at zero. Two integral Taylor remainders give
local trace-norm map error <=A^3 h^3/3; CPTP telescoping across r steps gives
B=min(2,(At)^3/(3r^2)). State trace distance is bounded by B/2, and a unit-norm
Pauli mean bias by B. This is an all-input **exact-arithmetic analytic bound**
for this bounded time-independent positive-time schedule, excluding initial
preparation faults and floating gate representation. It is not a Lean proof
or a formal interval analysis of the implemented gates.

Fraction arithmetic selects the smallest r with B<=0.01, before any numerical
curve is inspected: r=69, 134, 211. The previous integer fails the analytic policy
in each case. The envelope is conservative; it is not optimal locality-aware
complexity. At fixed coefficients A=O(n), this policy gives r=O(n^(3/2) t^(3/2)
/sqrt(epsilon_split)). Gates per trial are O(nr), and complete-workload logical
counts multiply by O(epsilon_stat^(-2) log(3/delta_family)). Fixed-Z0 sampling
has constant score range; the old extensive energy task has different shots.

## Actual circuits and independent bounded classical checks

Quantum implementation uses the original two_spin_lindblad.py generator:
actual CX-RZ-CX and RX layers plus fresh environment reset, CRY and reverse CX.
No dense Hamiltonian, channel or exponential is appended as a quantum gate.
The literal damping halves are retained even when separate-site dissipators
commute. System/environment initialization, discard-before-reuse, final readout
and all input gates are charged. The lowered QASM uses rotation/CX primitives,
with optimization level 1 and transpiler seed 7. Compiler barriers preserve
reset/measurement boundaries. Their count is recorded separately from logical
primitive counts; this bundle contains **no physical timing estimate**.

The independent full GKSL reference uses a column-vectorized dense generator.
The actual reduced schedule is independently evaluated from matrix operators
of the actual small gate components. A fresh environment may be traced
immediately after its dilation: it is untouched until its next reset or ignored
final measurement, so this causal contraction preserves subsequent system
state and readout. Whole2n-wire Qiskit density checks verify this construction
on short schedules. The complete selected schedules are evaluated by powers
of the gate-derived reduced channel, not a full 2n-wire gate-by-gate simulation.
Supplied mixed matrices initialize only the classical verifier; quantum trials
use the declared gate-prepared product input.
Source/circuit correctness, selected-state error and the uniform analytic bound
are reported separately. Small numerical agreement is not the proof of B.

Dense system states are at most 16x16, channels/Liouvillians at most 256x256;
whole-circuit densities are at most 256x256 with four-step checks. Numerical
convergence uses at most five fixed step points per case, with no adaptive
selection. Source generation is capped at 1024 steps and the existing 50,000
primitive-operation bound. This wrapper accepts only 2/3/4 systems. No GPU
or unbounded job is used.

## Missing physical layer

These are **logical, ideal-angle workloads**. The conditional instrument
backend defaults to five source wires and also provides opt-in weighted-Kraus
checks and QSP/thermal resource reports for other circuits. This wrapper uses
no opt-in validation of the six/eight-wire chain workloads and their long
repeated-reset schedules. The width fields below describe the default five-wire
configuration; a width-only check does not supply a chain physical report.
The n=2 width fits the default, but no case in this complete workload has a
reproduced Clifford+T/physical report. No distance, factory count, physical qubit number or
elapsed hardware time is extrapolated. A supported extension needs actual
rotation synthesis and a whole-workload error allocation, reset-aware routing,
checked wider instrument semantics, and a declared data/factory layout and
controller/reset/readout/idle-fault schedule. Generic adapter presence or
changing a width constant would not provide that evidence.

The logical ledger includes all complete trials and score/counter updates,
but excludes offline angle/circuit construction and final statistical wall time,
coherent synthesis, physical routing, factory transport, QEC and physical
initialization/reset/readout/controller faults and times. CPU build/reference
times are recorded separately. Qubit-layer volume is a dependency proxy,
not physical qubit-seconds. No matched co-design or advantage claim is made;
available frozen Grover fingerprints are unchanged.

## Reproduce, provenance and credit

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/generate_simulation_lindblad_workload.py
PYTHONPATH=src python -m pytest tests/test_simulation_lindblad_workload.py
```

For independent use, put lindblad_workload.py, two_spin_lindblad.py,
open_systems.py and the generator together with requirements-lock.txt and
LICENSE.txt. The optional source_provenance.py records a Git revision only
for a verified exact-root checkout; no-Git downloads retain source hashes and
null revision. Partial --sizes runs retain the full-family shot allocation and
remove only stale managed case records, preserving unrelated files. --no-sim
requires separate output and marks numerical checks unperformed.

Original code, diagrams, explanation and bound derivation are MIT work. Pinned
Qiskit 2.6.2 is Apache-2.0; NumPy 2.5.4 and SciPy 1.18.1 use BSD licenses. Exact
versions and source hashes are in manifest.json; upstream references are in
references.bib. Lindblad/GKS 1976, Strang 1968 and the Kliesch 2011/2012 erratum
provide context. No commutator constants are imported from that paper and no
survey wording or figures are copied. Formal/Lean coverage remains absent.
"""


def save_svg(fig, path):
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def generate(output, sizes=FAMILY, no_sim=False):
    sizes = tuple(sizes)
    if not sizes or len(sizes) > len(FAMILY) or len(set(sizes)) != len(sizes) or any(isinstance(n, bool) or not isinstance(n, int) or n not in FAMILY for n in sizes):
        raise ValueError("sizes must be a unique nonempty subset of2/3/4")
    output = validate_output(output, no_sim)
    before = fingerprints()
    paths = source_paths()
    root = ROOT if REPOSITORY else HERE
    revision = git_revision(root)
    dirty = None
    if revision is not None:
        try:
            dirty = git_dirty(root, [str(path.resolve().relative_to(root)) for path in paths.values()])
        except ValueError:
            # Exact hashes remain valid for externally symlinked inputs; a Git
            # owned-path status cannot represent those inputs at this root.
            dirty = None
    output.mkdir(parents=True, exist_ok=True)
    rows, convergence, reports = [], [], []
    for n in sizes:
        task = ChainTask(n)
        steps = select_steps(task)
        began = time.perf_counter()
        source, compiled = measured_trial(task, steps)
        construction_seconds = time.perf_counter()-began
        ledger = logical_ledger(task, source, compiled)
        folder = output/f"n{n}"
        folder.mkdir(exist_ok=True)
        qasms = {"source-trial.qasm": qasm3.dumps(source), "rotation-basis-trial.qasm": qasm3.dumps(compiled)}
        for name, qasm in qasms.items():
            (folder/name).write_text(qasm)
        bound = uniform_split_bound(task, steps)
        report = {"task": asdict(task), "output_contract": output_contract(), "selected_steps": steps,
                  "uniform_bound": {"display": float(bound), "numerator": bound.numerator, "denominator": bound.denominator,
                                    "kind": "all-input analytic exact-angle Hermitian trace-norm bound; not a finite-angle/Lean certificate",
                                    "generator_norm_envelope": float(generator_norm_envelope(task)),
                                    "previous_step_bound": float(uniform_split_bound(task, steps-1)) if steps > 1 else None},
                  "logical_ledger": ledger, "qasm_sha256": {name: sha256(text.encode()) for name, text in qasms.items()},
                  "cpu_circuit_build_lower_seconds": construction_seconds, "numerical_checks_performed": not no_sim,
                  "selected_numerical_check": None, "whole_circuit_short_check": None, "numerical_reference_seconds": None,
                  "physical_estimation": {"status": "unavailable", "existing_instrument_source_width_limit": 5,
                      "task_source_width": 2*n, "generic_width_supported": 2*n <= 5,
                      "actual_common_accuracy_physical_report": False,
                      "missing": ["actual coherent synthesis and whole-workload fault/error allocation", "chain-specific opt-in6/8-wire long repeated-reset validation and physical routing; this wrapper uses no such validation", "data/factory geometry and controller/reset/readout/idle-fault schedule"]}}
        selected = None
        if not no_sim:
            began = time.perf_counter()
            exact = reference_density(task, channel=reference_channel(task))
            selected = compare(task, steps, reference=exact)
            grid = tuple(dict.fromkeys((8, 16, 32, steps)))
            if len(grid) > MAX_CONVERGENCE_POINTS:
                raise ValueError("Predeclared numerical grid exceeds its cap")
            for count in grid:
                check = selected if count == steps else compare(task, count, reference=exact)
                convergence.append({"n_system": n, "steps": count, "uniform_trace_norm_bound": check["uniform_trace_norm_bound_display"],
                                    "observed_trace_norm_error": check["observed_trace_norm_error"], "observed_Z0_bias": check["observed_Z0_bias"]})
            whole = full_circuit_density(task, 2)
            reduced = gate_density(task, 2)
            residual = float(np.linalg.norm(whole-reduced, ord="nuc"))
            if residual > 2e-11:
                raise AssertionError("Whole-circuit gates/reset do not match causal reduced-channel evaluation")
            report.update(selected_numerical_check=selected,
                          whole_circuit_short_check={"steps": 2, "full_qubits": 2*n, "trace_norm_residual": residual,
                                                     "scope": "short whole-circuit verification; selected workload uses gate-contracted channel powers"},
                          numerical_reference_seconds=time.perf_counter()-began)
        write_json(folder/"report.json", report)
        reports.append(report)
        rows.append({"n_system": n, "selected_steps": steps, "logical_qubits": 2*n,
                     "generator_norm_envelope": float(generator_norm_envelope(task)), "uniform_trace_norm_bound": float(bound),
                     "complete_trials": ledger["complete_trials"], "primitive_ops_per_trial": sum(ledger["primitive_counts_per_complete_trial"].values()),
                     "primitive_ops_all_trials": sum(ledger["primitive_counts_all_trials"].values()),
                     "reset_per_trial": ledger["primitive_counts_per_complete_trial"]["reset"], "readout_per_trial": ledger["primitive_counts_per_complete_trial"]["measure"],
                     "observed_trace_norm_error": selected["observed_trace_norm_error"] if selected else None,
                     "observed_Z0_bias": selected["observed_Z0_bias"] if selected else None})
        print(f"Generated n={n}, policy r={steps}, numerical checks={not no_sim}", flush=True)
    hashes = {}
    extras = {"LICENSE.txt": ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt",
              "requirements-lock.txt": ROOT/"requirements-lock.txt" if REPOSITORY else HERE/"requirements-lock.txt",
              "references.bib": ROOT/"references/simulation.bib" if REPOSITORY else HERE/"references.bib",
              "source_provenance.py": HERE/"source_provenance.py"}
    for name, path in {**paths, **extras}.items():
        if path.is_file():
            raw = path.read_bytes()
            (output/name).write_bytes(raw)
            hashes[name] = sha256(raw)
    after = fingerprints()
    if before != after:
        raise AssertionError("Frozen baseline changed during generation")
    manifest = {"schema_version": 1, "status": "checked bounded logical workloads" if not no_sim else "logical generation; numerical checks unperformed",
                "declared_family": list(FAMILY), "generated_sizes": list(sizes), "output_contract": output_contract(),
                "source_commit_at_generation": revision, "dirty_owned_source": dirty, "source_sha256": hashes,
                "case_artifact_sha256": {f"n{n}": {name: sha256((output/f"n{n}"/name).read_bytes()) for name in CASE_FILES} for n in sizes},
                "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "scipy", "matplotlib")},
                "execution": {"python": platform.python_version(), "system": platform.system(), "architecture": platform.machine(),
                              "device": "CPU; bounded dense channels, no GPU/hardware run", "transpiler_seed": 7, "lowering_optimization_level": 1,
                              "sampling_seed": None, "sampling_note": "trial ensembles unexecuted"},
                "frozen_baseline_fingerprints": {"before": before, "after": after},
                "frozen_baseline_changed": False if any(before.values()) else None,
                "physical_report": "unavailable for this chain; default5 and no opt-in6/8 long repeated-reset validation used", "numerical_checks_performed": not no_sim}
    write_json(output/"manifest.json", manifest)
    write_csv(output/"summary.csv", rows)
    if convergence:
        write_csv(output/"convergence.csv", convergence)
    else:
        (output/"convergence.csv").unlink(missing_ok=True)
    (output/"README.md").write_text(readme(sizes, not no_sim))
    table = ["| Systems | Policy steps | Logical wires | Trials | Gates/trial incl. reset/readout | Full attempted logical operations | Uniform trace-norm bound |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    table.extend(f"| {r['n_system']} | {r['selected_steps']} | {r['logical_qubits']} | {r['complete_trials']:,} | {r['primitive_ops_per_trial']:,} | {r['primitive_ops_all_trials']:,} | {r['uniform_trace_norm_bound']:.6f} |" for r in rows)
    (output/"summary.md").write_text("\n".join(table)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for ax, key, title in zip(axes, ("selected_steps", "primitive_ops_per_trial", "primitive_ops_all_trials"),
                              ("Uniform-policy steps", "Logical operations per trial", "Complete-workload logical operations")):
        ax.plot([r["n_system"] for r in rows], [r[key] for r in rows], "o-", color="#247a86")
        ax.set_xticks(list(sizes)); ax.set_xlabel("System qubits n"); ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed local Z0 output · common accuracy and family confidence")
    fig.tight_layout(rect=(0, 0, 1, .93))
    save_svg(fig, output/"workload-scaling.svg")
    if convergence:
        fig, axes = plt.subplots(1, len(sizes), figsize=(3.8*len(sizes), 3.3), squeeze=False)
        for ax, n in zip(axes[0], sizes):
            selected_rows = sorted((r for r in convergence if r["n_system"] == n), key=lambda r: r["steps"])
            x = [r["steps"] for r in selected_rows]
            ax.loglog(x, [r["uniform_trace_norm_bound"] for r in selected_rows], "o--", label="Uniform exact-angle bound", color="#97552a")
            ax.loglog(x, [r["observed_trace_norm_error"] for r in selected_rows], "o-", label="Observed state trace-norm error", color="#247a86")
            ax.axhline(.01, color="#697386", lw=.8, linestyle=":")
            ax.set_xticks(x, [str(count) for count in x])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"n={n}"); ax.set_xlabel("Steps r"); ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel("Full density trace-norm error")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .88))
        save_svg(fig, output/"convergence.svg")
    else:
        (output/"convergence.svg").unlink(missing_ok=True)
    # A one-step actual source diagram; full selected schedules are QASM exports.
    demo, _ = measured_trial(ChainTask(2), 1)
    figure = demo.draw("mpl", fold=28, idle_wires=False)
    figure.suptitle("Two-spin demo · one positive-time palindrome, not the full 69-step workload", fontsize=10)
    save_svg(figure, output/"one-step-circuit.svg")
    for n in set(FAMILY)-set(sizes):
        folder = output/f"n{n}"
        for name in CASE_FILES:
            (folder/name).unlink(missing_ok=True)
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, choices=FAMILY, default=list(FAMILY))
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/lindblad-workload" if REPOSITORY else HERE/"generated-lindblad-workload")
    args = parser.parse_args()
    try:
        generate(args.output_dir, args.sizes, args.no_sim)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
