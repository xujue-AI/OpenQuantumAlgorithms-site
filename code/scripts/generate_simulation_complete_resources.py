#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize five bounded QSP/thermal instruments and conditional workloads.

The repeated schedules and hardware runs are unexecuted. Source intervals,
guarded actual synthesis, a separate SDK reserve and physical assumptions are
recorded separately. This reuses the serialized instrument backend unchanged.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (HERE.name == "scripts" and
              (SOURCE / "openquantum/simulation/complete_resource_workloads.py").is_file() and
              not (HERE / "complete_resource_workloads.py").is_file())
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-complete-simulation-resources"
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import numpy as np
from qiskit import qasm3

if REPOSITORY:
    from openquantum import instrument_resources as ir
    from openquantum.simulation import complete_resource_workloads as cr
else:
    import instrument_resources as ir
    import complete_resource_workloads as cr

SOURCE_NAMES = ("instrument_resources.py", "hamiltonian_simulation.py", "lcu.py",
                "retargetable_qsp.py", "thermal_spectral.py", "qsp_phase_generation.py",
                "qsp_certificate.py", "complete_resource_workloads.py",
                "generate_simulation_complete_resources.py")
CASE_IDS = tuple(f"{kind}-n{n}-{axis}" for kind, n, axis in cr.default_task_specs())
PLOTS = ("attempts.svg", "space-time.svg", "accuracy.svg")
FROZEN_FILES = ("src/openquantum/grover.py", "scripts/estimate_resources.py",
                "assets/generated/grover/circuit.qasm",
                "assets/generated/resources/grover-surface-code.json",
                "assets/generated/resources/grover-surface-code.csv")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def dump(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def source_files():
    if not REPOSITORY:
        return {name: HERE / name for name in SOURCE_NAMES}
    return {name: (Path(__file__) if name.startswith("generate_") else
                   SOURCE / "openquantum" / name if name in SOURCE_NAMES[:2] else
                   SOURCE / "openquantum/simulation" / name) for name in SOURCE_NAMES}


def fingerprints():
    return {name: digest((ROOT / name).read_bytes()) if REPOSITORY and (ROOT / name).is_file() else None
            for name in FROZEN_FILES}


def revision(sources):
    record = {"source_commit_at_generation": None, "dirty_source_paths": None,
              "revision_scope": "optional verified exact checkout root; source SHA-256 identities are authoritative"}
    if not REPOSITORY or not (ROOT / ".git").exists():
        return record
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
    try:
        top = subprocess.check_output(["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True, timeout=5, env=environment, stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve() != ROOT.resolve():
            return record
        record["source_commit_at_generation"] = subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=5, env=environment, stderr=subprocess.DEVNULL).strip()
        record["dirty_source_paths"] = subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(ROOT), "status", "--porcelain=v1", "--",
             *[str(path.relative_to(ROOT)) for path in sources.values()]], text=True, timeout=5, env=environment, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return record


def validate_paths(output, cache, sources):
    output, cache = output.resolve(), cache.resolve()
    owned = ROOT / "assets/generated/simulation/complete-resources"
    if REPOSITORY and output.is_relative_to(ROOT) and not output.is_relative_to(owned):
        raise ValueError("Repository output must remain inside assets/generated/simulation/complete-resources")
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("Public output and private cache must be disjoint")
    if REPOSITORY and cache.is_relative_to(ROOT):
        raise ValueError("Private cache must stay outside the source checkout")
    if any(path.resolve().is_relative_to(output) for path in sources.values()):
        raise ValueError("Output must not overwrite an input source")
    targets = [output / name for name in (*SOURCE_NAMES, *PLOTS, "results.json", "manifest.json", "progress.json", "summary.csv", "summary.md", "README.md", "LICENSE.txt", "references.bib", "requirements-lock.txt")]
    targets += [output / case / name for case in CASE_IDS for name in ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json")]
    if any(path.is_symlink() or path.parent.is_symlink() for path in targets):
        raise ValueError("Managed output targets must not be symbolic links")
    return output, cache


def scenario_id(physical_error, controller_error):
    return f"p{physical_error:.0e}-controller{controller_error:.0e}"


def prepared_key(case_id, sources):
    # Generated privately by this runner. No public or third-party pickle input.
    identity = {name: digest(path.read_bytes()) for name, path in sources.items()
                if not name.startswith("generate_")}
    identity.update(case_id=case_id, python=platform.python_version(),
                    packages={name: importlib.metadata.version(name) for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")},
                    synthesis_allowance=cr.DEFAULT_SYNTHESIS_ALLOWANCE,
                    realization_reserve=cr.DEFAULT_REALIZATION_RESERVE)
    if not REPOSITORY:
        # Standalone dataclass/module names differ from package imports. Never
        # unpickle a same-source package checkpoint into the download runtime.
        identity["standalone_module_imports"] = True
    return digest(json.dumps(identity, sort_keys=True).encode())


def managed_artifact_hashes(output, selected):
    """Hash only published managed outputs, preserving unrelated user files."""
    names = [*SOURCE_NAMES, *PLOTS, "results.json", "summary.csv", "summary.md", "README.md",
             "LICENSE.txt", "references.bib", "requirements-lock.txt"]
    names += [f"{case}/{name}" for case in selected for name in
              ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json")]
    result = {}
    for name in sorted(set(names)):
        path = output / name
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Managed artifact hashes must not follow symbolic links")
        result[name] = digest(path.read_bytes())
    return result


def prepare(case_id, spec, sources, cache, reuse):
    checkpoint = cache / "prepared" / (prepared_key(case_id, sources) + ".pickle")
    if reuse and checkpoint.is_file():
        with checkpoint.open("rb") as handle:
            prepared, seconds = pickle.load(handle)
        return prepared, seconds, True
    start = time.perf_counter()
    prepared = cr.synthesize_task(cr.build_complete_task(*spec))
    seconds = time.perf_counter() - start
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump((prepared, seconds), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(checkpoint)
    return prepared, seconds, False


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_row(task, scenario, resource, attempts, accepted, t_count):
    points = resource.get("pareto_points", [])
    point = min(points, key=lambda p: (p["peak_physical_qubits"], p["worst_case_runtime_all_attempts_ns"])) if points else None
    diagnostic = resource.get("controller_model_diagnostic", {})
    return {"task": task, "scenario": scenario, "status": resource["status"],
            "accepted_slots": accepted, "complete_attempts": attempts,
            "t_and_tdg_per_attempt": t_count,
            "data_code_distance": point["data_code_distance"] if point else None,
            "peak_physical_qubits": point["peak_physical_qubits"] if point else None,
            "runtime_all_attempts_hours": point["worst_case_runtime_all_attempts_ns"] / 3.6e12 if point else None,
            "full_physical_failure_model": point["full_physical_failure_model"] if point else None,
            "controller_model_contribution": diagnostic.get("distance_independent_controller_model_contribution"),
            "pareto_points": len(points)}


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date": None}, bbox_inches="tight")
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def plots(output, cases, rows):
    names = list(cases)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(names, [cases[n]["prepared"]["complete_workload"]["accepted_slots"] for n in names], color="#365e88")
    axes[0].set_ylabel("Required accepted scores")
    axes[1].bar(names, [cases[n]["prepared"]["complete_workload"]["complete_attempts"] for n in names], color="#a56737")
    axes[1].set_ylabel("Capped complete attempts")
    for ax in axes:
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Calculated complete workloads; repeated trials unexecuted")
    fig.tight_layout()
    save_svg(fig, output / "attempts.svg")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    tasks = ("dynamics-n1-Z", "dynamics-n2-Z", "thermal-n1-X", "thermal-n2-energy")
    colors = ("#365e88", "#a56737", "#4f8171", "#876488")
    for ax, physical_error in zip(axes, ("p1e-03", "p1e-04")):
        for task, color in zip(tasks, colors):
            for row in rows:
                if row["task"] == task and row["scenario"].startswith(physical_error) and row["peak_physical_qubits"] is not None:
                    ax.scatter(row["peak_physical_qubits"], row["runtime_all_attempts_hours"], s=65, color=color,
                               marker="o" if "controller1e-15" in row["scenario"] else "^")
        ax.set(xscale="log", yscale="log", xlabel="Peak physical qubits", title=f"Physical error {physical_error[1:]}")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value/1000:,.0f}k"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    axes[0].set_ylabel("All-attempt runtime (hours)")
    handles = [plt.Line2D([], [], color=color, marker="s", linestyle="", label=task) for task, color in zip(tasks, colors)]
    handles += [plt.Line2D([], [], color="#555555", marker=marker, linestyle="", label=label) for marker, label in
                (("o", "Default controller 1e-15"), ("^", "Conditional controller 1e-18"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Conditional serialized models: minimum-space Pareto points\nInfeasible scenarios remain explicit in the report")
    fig.tight_layout(rect=(0, .13, 1, .92))
    save_svg(fig, output / "space-time.svg")
    fig, ax = plt.subplots(figsize=(10, 4))
    source = [cases[n]["prepared"]["error_and_conditioning_contract"]["source_score_bias_upper"] for n in names]
    realization = [cases[n]["prepared"]["error_and_conditioning_contract"]["realization_score_bias_upper_proposal"] for n in names]
    statistical = [cases[n]["prepared"]["statistical_absolute_error"] for n in names]
    x = np.arange(len(names))
    ax.bar(x, source, label="Declared source bound", color="#365e88")
    ax.bar(x, realization, bottom=source, label="Conditional actual synthesis + SDK reserve", color="#a56737")
    ax.bar(x, statistical, bottom=np.array(source)+realization, label="Statistical allowance", color="#85a79a")
    ax.set(xticks=x, xticklabels=names, ylabel="Absolute score error allowance", title="Separate deterministic and statistical stages; no hardware certificate")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.2)
    save_svg(fig, output / "accuracy.svg")


def readme():
    return """# Bounded complete simulation workloads

Five actual measured Clifford+T instruments cover dynamics n=1/2 (Z0)
and thermal n=1 (X), n=2 (X and ZZ). Repeated trial populations and hardware
schedules are calculated, unexecuted. The total task error target is 0.05:
source 0.01, realization 0.01, statistics 0.03. Four failure categories each
receive 0.0025; the two thermal n=2 groups share every task allocation equally.

The source is the explicit retargetable QSP/access algorithm. Every attempt
retains input resets, preparation, access/uncomputation, acceptance readout,
thermal reference discard, basis changes and every system readout. Every
rejected complete attempt and capped factory retry is charged. The scorer
uses exact bounded Kraus/statevector semantics, including correlated bits.

Source interval bounds, guarded actual Ross–Selinger numerical synthesis,
a separate 1e-7 SDK/control/routing reserve, Hoeffding confidence and physical
fault hypotheses remain separate. Actual sequence hashes identify the saved
synthesis (the native backend exposes no seed); SABRE uses seed 7. The reserve
is an explicit norm assumption, not inferred from nominal/routed agreement.
The QASM exporter omits overall phase; exports specify measured trials rather
than phase-preserving controlled oracles. Full in-memory synthesis disables
clean-input lowering. No formal synthesis or hardware confidence certificate
is claimed. Classical preprocessing/report wall time is outside quantum time.

Thermal acceptance uses the exact trace-zero Pauli input and Bell Jensen
bound Tr(exp(-beta H))/d>=1, giving target branch norm>=exp(-a_exact)/4.
The weaker generic exp(-2a_exact)/4 control is also recorded. Neither bound
uses the numerical instance acceptance. Each group has a bounded scalar
score; statistical correlations within a measured bitstring are allowed.

This is a separate serialized instrument scenario, preserving the frozen
Grover/PSSPC baseline. Physical error 1e-3/1e-4 is crossed with the unchanged
default controller 1e-15/decision and a separate conditional 1e-18 scenario.
All other profile assumptions are unchanged. Infeasibility remains explicit;
a controller contribution is a bound within the additive engineering model,
not a lower bound on actual hardware failure. Sequential thermal X/Z results
use a common distance/factory size/supply time/output quality. Peak space is
the maximum and runtime/failure allocations add; no incompatible minima are
combined. Distances are odd 3..49 and factory retries are capped at 16.

Limits: one/two systems; eight total wires; 20,000 source instructions;
500,000 synthesized/check instructions; 256 weighted branches; 1e12 complete
attempts. These bounded examples do not establish scalable hardware advantage.
The physical backend assumes tracked frames, protected Pauli bases and a
factory interface; it does not certify geometry, decoder throughput or
arbitrary adaptive feedback. See each model's assumptions/excluded fields.

Reproduce with the locked Python 3.12 environment:

    PYTHONPATH=src python scripts/generate_simulation_complete_resources.py
    PYTHONPATH=src python -m pytest tests/test_simulation_complete_resource_workloads.py tests/test_simulation_instrument_caps.py

Standalone: save every copied Python source, lock, license and references
beside the generator and notebook. The default standalone output is a new
subdirectory. Use --case for one of the five bounded groups. Partial generation
removes only managed omitted case files. --reuse-prepared explicitly reuses
runner-generated private checkpoints keyed by exact runtime source/package
identities. Cache and public output must be disjoint; the private cache is
outside the checkout. No checkpoint is distributed. manifests record source,
package and QASM hashes; standalone revision is null. Original MIT code/text/
figures; primary algorithm/package references are in references.bib.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASE_IDS, action="append")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/complete-resources" if REPOSITORY else HERE / "generated-complete-resources")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "openquantum-sixth-resource-cache")
    parser.add_argument("--reuse-prepared", action="store_true")
    args = parser.parse_args()
    sources = source_files()
    try:
        output, cache = validate_paths(args.output_dir, args.cache_dir, sources)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    selected = args.case or list(CASE_IDS)
    before = fingerprints()
    source_before = {name: digest(path.read_bytes()) for name, path in sources.items()}
    cases, rows = {}, []
    profiles = {scenario_id(p, c): ir.InstrumentProfile(physical_error_rate=p, controller_failure_per_decision=c)
                for c in (1e-15, 1e-18) for p in (1e-3, 1e-4)}
    for spec in cr.default_task_specs():
        case_id = f"{spec[0]}-n{spec[1]}-{spec[2]}"
        if case_id not in selected:
            continue
        print(f"Preparing {case_id}", flush=True)
        prepared, seconds, reused = prepare(case_id, spec, sources, cache, args.reuse_prepared)
        record = prepared.as_record()
        record.update(classical_preparation_synthesis_check_wall_seconds=seconds,
                      reused_exact_private_prepared_checkpoint=reused)
        folder = output / case_id
        folder.mkdir(exist_ok=True)
        qasm_hashes = {}
        for name, circuit in (("source-instrument.qasm", prepared.task.source),
                              ("clifford-t-instrument.qasm", prepared.synthesized.circuit)):
            raw = qasm3.dumps(circuit).encode()
            (folder / name).write_bytes(raw)
            qasm_hashes[name] = digest(raw)
        record["qasm_sha256"] = qasm_hashes
        scenarios = {}
        for name, profile in profiles.items():
            print(f"Estimating {case_id}: {name}", flush=True)
            report = cr.estimate_workload(prepared, profile=profile, cache_dir=cache / "factories")
            scenarios[name] = report
            rows.append(summary_row(case_id, name, report, prepared.workload["complete_attempts"],
                                    prepared.workload["accepted_slots"],
                                    sum(prepared.synthesized.circuit.count_ops().get(g, 0) for g in ("t", "tdg"))))
            print(f"Computed {case_id}: {name}: {report['status']}", flush=True)
        cases[case_id] = {"prepared": record, "scenarios": scenarios}
        dump(folder / "report.json", cases[case_id])
        dump(output / "progress.json", {"completed_cases": list(cases), "requested_cases": selected})
    energy = {}
    if all(name in cases for name in ("thermal-n2-X", "thermal-n2-Z")):
        for name in profiles:
            report = cr.combine_energy_profiles({axis: cases[f"thermal-n2-{axis}"]["scenarios"][name] for axis in ("X", "Z")})
            energy[name] = report
            rows.append(summary_row("thermal-n2-energy", name, report,
                                    sum(cases[f"thermal-n2-{axis}"]["prepared"]["complete_workload"]["complete_attempts"] for axis in ("X", "Z")),
                                    sum(cases[f"thermal-n2-{axis}"]["prepared"]["complete_workload"]["accepted_slots"] for axis in ("X", "Z")), None))
    for case in set(CASE_IDS) - set(selected):
        folder = output / case
        for name in ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json"):
            (folder / name).unlink(missing_ok=True)
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    dump(output / "results.json", {"schema_version": 1, "cases": cases, "energy": energy,
                                  "profiles": {name: asdict(profile) for name, profile in profiles.items()},
                                  "repeated_workloads_executed": False, "formal_confidence_certificate": False})
    write_csv(output / "summary.csv", rows)
    table = "| Task | Scenario | Status | Capped attempts | Peak qubits | Hours |\n|---|---|---|---:|---:|---:|\n"
    for row in rows:
        hours = f"{row['runtime_all_attempts_hours']:.6g}" if row["runtime_all_attempts_hours"] is not None else "—"
        table += f"| {row['task']} | {row['scenario']} | {row['status']} | {row['complete_attempts']:,} | {row['peak_physical_qubits'] or '—'} | {hours} |\n"
    (output / "summary.md").write_text(table)
    plots(output, cases, rows)
    (output / "README.md").write_text(readme())
    source_hashes = {}
    for name, path in sources.items():
        raw = path.read_bytes()
        if digest(raw) != source_before[name]:
            raise AssertionError("Input sources changed during generation; prepared checkpoints remain private, but this run cannot publish a coherent provenance record")
        (output / name).write_bytes(raw)
        source_hashes[name] = digest(raw)
    extras = {"LICENSE.txt": (ROOT / "LICENSE").read_bytes() if REPOSITORY else (HERE / "LICENSE.txt").read_bytes(),
              "requirements-lock.txt": (ROOT / "requirements-lock.txt").read_bytes() if REPOSITORY else (HERE / "requirements-lock.txt").read_bytes(),
              "references.bib": ((ROOT / "references/simulation.bib").read_bytes() + b"\n" + (ROOT / "references/resources.bib").read_bytes()) if REPOSITORY else (HERE / "references.bib").read_bytes()}
    for name, raw in extras.items():
        (output / name).write_bytes(raw)
        source_hashes[name] = digest(raw)
    after = fingerprints()
    if before != after:
        raise AssertionError("Frozen baseline identities changed during generation")
    (output / "progress.json").unlink(missing_ok=True)
    artifacts = managed_artifact_hashes(output, list(cases))
    dump(output / "manifest.json", {"schema_version": 1, **revision(sources),
                                   "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")},
                                   "python_version": platform.python_version(), "device": "local CPU; ideal bounded numerical instrument checks",
                                   "transpiler_seed": 7, "synthesis_seed": None,
                                   "synthesis_seed_scope": "native backend has no seed API; exact actual sequences/QASM hashes saved",
                                   "source_sha256": source_hashes, "artifacts_sha256": artifacts,
                                   "case_ids": list(cases), "plots": list(PLOTS), "limits": asdict(cr.resource_limits()),
                                   "frozen_baseline_before": before, "frozen_baseline_after": after,
                                   "public_bundle_contains_private_cache": False,
                                   "repeated_workloads_executed": False})
    print(f"Completed {len(cases)} bounded instruments and {len(rows)} scenario rows", flush=True)


if __name__ == "__main__":
    main()
