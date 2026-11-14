#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded complete physical accounting of literal n3/r134/Z0.

Native preparation and each fixed physical profile run in capped children.
Prior n2 results are preserved; n4 remains unimplemented with its allocation reserved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (HERE.name == "scripts" and
              (SOURCE / "openquantum/simulation/lindblad_three_physical.py").is_file() and
              not (HERE / "lindblad_three_physical.py").is_file())
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-literal-lindblad-three"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3

if REPOSITORY:
    from openquantum import instrument_resources as ir
    from openquantum.simulation import lindblad_three_physical as lp
else:
    import instrument_resources as ir
    import lindblad_three_physical as lp

SOURCES = ("instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py",
           "open_systems.py", "bounded_density_instrument.py", "lindblad_three_physical.py", "generate_simulation_lindblad_three.py")
FROZEN_SOURCE_SNAPSHOTS = ("instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py", "open_systems.py")
PLOTS = ("one-step.svg", "counts.svg", "space-time.svg", "failures.svg", "error-budget.svg")
EXTRAS = ("LICENSE.txt", "requirements-lock.txt", "references.bib")
PUBLIC = (*SOURCES, *PLOTS, *EXTRAS, "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
          "results.json", "summary.csv", "family-status.csv", "artifact-inventory.csv", "artifact-inventory.json", "README.md", "manifest.json", "progress.json")
FROZEN = ("src/openquantum/instrument_resources.py", "src/openquantum/grover.py",
          "scripts/estimate_resources.py", "assets/generated/resources/grover-surface-code.json",
          "assets/generated/resources/grover-surface-code.csv")
CAPS = lp.run_caps()
CHECKPOINT_SCHEMA = "literal-n3-density-v1"
INVENTORY_EXCLUSIONS = ("artifact-inventory.csv", "artifact-inventory.json", "manifest.json", "progress.json")
BASELINE_INVENTORY = {
    "tracked_generated_assets_files_at_start": 7133,
    "tracked_generated_assets_bytes_at_start": 351242875,
    "preserved_n2_files": 26,
    "preserved_n2_total_bytes": 3505619,
    "scope": "local tracked generated source assets at worktree creation; not public hosting occupancy",
}


class ExecutionCapExceeded(RuntimeError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def dump(path, value):
    raw = (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > CAPS["managed_json_qasm_bytes"]:
        raise ExecutionCapExceeded("managed JSON exceeds the predeclared 32 MiB cap")
    path.write_bytes(raw)


def source_files():
    return {name: (Path(__file__) if name.startswith("generate_") else
                   SOURCE / "openquantum" / name if name == "instrument_resources.py" else
                   SOURCE / "openquantum/simulation" / name) if REPOSITORY else HERE / name
            for name in SOURCES}


def fingerprints():
    return {name: digest((ROOT / name).read_bytes()) if REPOSITORY and (ROOT / name).is_file() else None for name in FROZEN}


def revision(paths):
    result = {"source_commit_at_generation": None, "dirty_source_paths": None,
              "revision_scope": "optional verified exact-root Git; source hashes are authoritative"}
    if not REPOSITORY or not (ROOT / ".git").exists():
        return result
    env = {key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(ROOT), *args], env=env, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT.resolve():
            result.update(source_commit_at_generation=git("rev-parse", "HEAD"),
                          dirty_source_paths=git("status", "--porcelain=v1", "--", *[str(p.relative_to(ROOT)) for p in paths.values()]).splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return result


def validate_paths(output, cache, sources):
    output, cache = output.resolve(), cache.resolve()
    owned = ROOT / "assets/generated/simulation/lindblad-three"
    if REPOSITORY and output.is_relative_to(ROOT) and not output.is_relative_to(owned):
        raise ValueError("repository output must remain inside assets/generated/simulation/lindblad-three")
    if cache.is_relative_to(output) or output.is_relative_to(cache) or (REPOSITORY and cache.is_relative_to(ROOT)):
        raise ValueError("private cache must be outside the checkout and disjoint from public output")
    if any(path.resolve().is_relative_to(output) for path in sources.values()):
        raise ValueError("output must not overwrite an input source")
    if any((output / name).is_symlink() for name in PUBLIC):
        raise ValueError("managed artifacts must not be symbolic links")
    return output, cache


def managed_hashes(output):
    result = {}
    for name in sorted(set(PUBLIC) - {"manifest.json", "progress.json"}):
        path = output / name
        if path.is_symlink():
            raise ValueError("managed artifact hashes must not follow symbolic links")
        if path.is_file():
            result[name] = digest(path.read_bytes())
    return result


def prepared_key(paths):
    identity = {name: digest(path.read_bytes()) for name, path in paths.items() if not name.startswith("generate_")}
    identity.update(checkpoint_schema=CHECKPOINT_SCHEMA,
                    import_mode="package" if REPOSITORY else "standalone", python=platform.python_version(),
                    packages={name: importlib.metadata.version(name) for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")})
    identity["preparation_execution_policy"] = {
        "caps": {name: CAPS[name] for name in ("native_rotation_synthesis_attempts_upper_bound", "prepare_seconds",
                 "source_instructions", "routed_and_synthesized_instructions", "density_validation_instructions",
                 "density_wires", "private_checkpoint_bytes")},
        "native_guard_source_sha256": digest(inspect.getsource(bounded_rotation_synthesis).encode()),
        "preserve_full_unitary": True,
    }
    return digest(json.dumps(identity, sort_keys=True).encode())


def save_prepared(path, value):
    if path.is_symlink() or path.with_suffix(path.suffix + ".tmp").is_symlink():
        raise ValueError("private checkpoint targets must not be symbolic links")
    temporary = path.with_suffix(path.suffix + ".tmp")
    class BoundedWriter:
        def __init__(self, handle):
            self.handle, self.total = handle, 0
        def write(self, raw):
            self.total += len(raw)
            if self.total > CAPS["private_checkpoint_bytes"]:
                raise ExecutionCapExceeded("private prepared checkpoint exceeds 128 MiB")
            return self.handle.write(raw)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, BoundedWriter(handle), protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_prepared(path):
    if path.is_symlink():
        raise ValueError("private checkpoints must not be symbolic links")
    if path.stat().st_size > CAPS["private_checkpoint_bytes"]:
        raise ExecutionCapExceeded("private prepared checkpoint exceeds 128 MiB")
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepared_hash(path):
    if path.is_symlink():
        raise ValueError("private checkpoints must not be symbolic links")
    if path.stat().st_size > CAPS["private_checkpoint_bytes"]:
        raise ExecutionCapExceeded("private prepared checkpoint exceeds 128 MiB")
    result, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while block := handle.read(65536):
            total += len(block)
            if total > CAPS["private_checkpoint_bytes"]:
                raise ExecutionCapExceeded("private prepared checkpoint exceeds 128 MiB")
            result.update(block)
    return result.hexdigest()


def validate_private_job(job, cache):
    """Preflight containment and every private target before any mutation."""
    cache = cache.resolve()
    # Inspect lexical in-cache parents before resolution: an in-cache alias
    # pointing back to the cache root must not skip the symlink guard.
    job = Path(os.path.abspath(job))
    if job.is_symlink() or not job.is_relative_to(cache) or not job.resolve().is_relative_to(cache):
        raise ValueError("private job directory must remain inside its cache and must not be a symbolic link")
    for parent in job.parents:
        if parent.is_symlink():
            raise ValueError("private job parents must not be symbolic links")
        if parent == cache:
            break
    for name in ("status.json", "value.json", "worker.log", "prepared.pickle", "prepared.pickle.tmp", "prepare-status.json"):
        if (job / name).is_symlink():
            raise ValueError("managed private job targets must not be symbolic links")
    return job.resolve()


def checkpoint_status(job, identity):
    """Verify schema, namespace and bounded content hash before unpickling."""
    status_path = job / "prepare-status.json"
    if not status_path.is_file() or status_path.stat().st_size > CAPS["managed_json_qasm_bytes"]:
        return {}
    try:
        status = json.loads(status_path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(status, dict):
        return {}
    if (status.get("status") != "completed" or
            status.get("checkpoint_schema") != CHECKPOINT_SCHEMA or
            status.get("checkpoint_identity") != identity or
            not (job / "prepared.pickle").is_file()):
        return {}
    if status.get("prepared_checkpoint_sha256") != prepared_hash(job / "prepared.pickle"):
        return {}
    return status


@contextmanager
def bounded_native_queries():
    """Preserve native objects; fail a whole profile on returned-candidate caps.

    Counts are top-level adapter calls, not every internal factory ISA node.
    Internal work before the first yield is bounded by the child wall timeout.
    """
    from qdk.qre.models import RoundBasedFactory, SurfaceCode
    counts = {"top_level_factory_queries": 0, "top_level_surface_code_queries": 0,
              "factory_candidates": 0, "factory_candidates_per_query": []}
    originals = {}
    caller = str(Path(ir.__file__).resolve())
    class LimitedQuery:
        def __init__(self, query):
            self.query = query
        def __getattr__(self, name):
            return getattr(self.query, name)
        def enumerate(self, *args, **kwargs):
            number = 0
            for candidate in self.query.enumerate(*args, **kwargs):
                number += 1
                counts["factory_candidates"] += 1
                if number > CAPS["factory_candidates_per_query"] or counts["factory_candidates"] > CAPS["factory_candidates_per_profile"]:
                    raise ExecutionCapExceeded("native factory candidate cap exceeded; whole profile uncomputed")
                yield candidate
            counts["factory_candidates_per_query"].append(number)
    def install(cls, field, maximum, wrap):
        owned = "q" in cls.__dict__
        descriptor = inspect.getattr_static(cls, "q")
        original = cls.q
        originals[cls] = (owned, descriptor)
        def guarded(*args, **kwargs):
            frame = inspect.currentframe().f_back
            top = frame is not None and str(Path(frame.f_code.co_filename).resolve()) == caller
            if top:
                counts[field] += 1
                if counts[field] > maximum:
                    raise ExecutionCapExceeded("top-level native query cap exceeded; whole profile uncomputed")
            query = original(*args, **kwargs)
            return LimitedQuery(query) if top and wrap else query
        setattr(cls, "q", staticmethod(guarded))
    install(RoundBasedFactory, "top_level_factory_queries", CAPS["factory_queries_per_profile"], True)
    install(SurfaceCode, "top_level_surface_code_queries", CAPS["surface_code_queries_per_profile"], False)
    try:
        yield counts
    finally:
        for cls, (owned, descriptor) in originals.items():
            if owned:
                setattr(cls, "q", descriptor)
            else:
                delattr(cls, "q")


@contextmanager
def bounded_rotation_synthesis():
    """Count actual native attempts, preserving returned approximants."""
    native = ir.gridsynth_unitary
    counts = {"calls": 0}
    def guarded_synthesis(*args, **kwargs):
        counts["calls"] += 1
        if counts["calls"] > CAPS["native_rotation_synthesis_attempts_upper_bound"]:
            raise ExecutionCapExceeded("native rotation synthesis attempt cap10720 exceeded")
        return native(*args, **kwargs)
    ir.gridsynth_unitary = guarded_synthesis
    try:
        yield counts
    finally:
        ir.gridsynth_unitary = native


def stage_worker(stage, job, profile_index, cache):
    began = time.perf_counter()
    try:
        if stage == "family":
            result = lp.family_status_records(check_rejections=True)
            dump(job / "value.json", result)
        elif stage == "prepare":
            preparation_identity = prepared_key(source_files())
            with bounded_rotation_synthesis() as counts:
                prepared = lp.prepare_workload()
            if prepared_key(source_files()) != preparation_identity:
                raise ValueError("preparation source or execution policy changed during the bounded native stage")
            prepared.synthesized.report["native_rotation_synthesis_attempts"] = counts["calls"]
            prepared.synthesized.report["native_rotation_synthesis_attempts_upper_bound"] = CAPS["native_rotation_synthesis_attempts_upper_bound"]
            save_prepared(job / "prepared.pickle", prepared)
        else:
            identity = prepared_key(source_files())
            if not checkpoint_status(job, identity):
                raise ValueError("private profile checkpoint lacks a matching complete identity and hash")
            prepared = load_prepared(job / "prepared.pickle")
            profile = lp.fixed_profiles()[profile_index]
            factory_cache = validate_private_job(cache / "factories", cache)
            with bounded_native_queries() as counts:
                result = lp.estimate_prepared(prepared, profile=profile, cache_dir=factory_cache, include_all_feasible_points=True)
            result["execution_query_counts"] = counts
            result["native_trace_calls_upper_bound"] = counts["factory_candidates"]
            result["native_trace_calls_instrumented"] = False
            dump(job / "value.json", result)
        status = {"status": "completed", "wall_seconds": time.perf_counter()-began}
        if stage == "prepare":
            status["prepared_checkpoint_sha256"] = prepared_hash(job / "prepared.pickle")
            status["checkpoint_schema"] = CHECKPOINT_SCHEMA
            status["checkpoint_identity"] = preparation_identity
    except ExecutionCapExceeded as error:
        status = {"status": "uncomputed_execution_cap", "reason": str(error), "wall_seconds": time.perf_counter()-began}
    except Exception as error:
        status = {"status": "uncomputed_stage_error", "error_type": type(error).__name__,
                  "reason": "bounded stage failed; diagnostic retained only in private local log", "wall_seconds": time.perf_counter()-began}
        import traceback
        traceback.print_exc()
    dump(job / "status.json", status)


def run_stage(stage, job, cache, timeout, *, profile_index=0):
    job = validate_private_job(job, cache)
    job.mkdir(parents=True, exist_ok=True)
    (job / "status.json").unlink(missing_ok=True)
    (job / "value.json").unlink(missing_ok=True)
    if stage == "prepare":
        (job / "prepare-status.json").unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--_stage", stage,
               "--_job-dir", str(job), "--cache-dir", str(cache), "--_profile-index", str(profile_index)]
    env = dict(os.environ)
    env.update(PYTHONPATH=str(SOURCE if REPOSITORY else HERE), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    start = time.perf_counter()
    with (job / "worker.log").open("wb") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT if REPOSITORY else HERE, env=env,
                                       stdout=log, stderr=log, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "uncomputed_timeout", "timeout_seconds": timeout,
                    "wall_seconds": time.perf_counter()-start, "reason": "predeclared hard stage timeout; no partial model retained"}
    if not (job / "status.json").is_file():
        return {"status": "uncomputed_stage_error", "exit_code": completed.returncode,
                "reason": "worker returned no complete status; no partial result retained"}
    return json.loads((job / "status.json").read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_svg(figure, path):
    figure.savefig(path, metadata={"Date": None}, bbox_inches="tight")
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def artifact_inventory(output):
    """Inventory payload bytes without recursive metadata self-hashes.

    The manifest separately attests these inventory files. A final checkout
    inventory can include the four explicitly excluded metadata files.
    """
    def category(name):
        if name in FROZEN_SOURCE_SNAPSHOTS:
            return "copied_frozen_source_dependency"
        if name in SOURCES:
            return "new_original_source_snapshot"
        if name == "requirements-lock.txt":
            return "copied_environment_lock"
        if name in EXTRAS:
            return "copied_license_or_primary_references"
        if name in PLOTS:
            return "original_figure"
        if name.endswith(".qasm"):
            return "actual_circuit_export"
        return "result_or_documentation"
    rows = []
    for name in sorted(set(PUBLIC) - set(INVENTORY_EXCLUSIONS)):
        path = output / name
        if path.is_symlink():
            raise ValueError("inventory must not follow managed artifact symbolic links")
        if path.is_file():
            rows.append({"file": name, "category": category(name), "bytes": path.stat().st_size,
                         "sha256": digest(path.read_bytes())})
    write_csv(output / "artifact-inventory.csv", rows)
    totals = {}
    for row in rows:
        value = totals.setdefault(row["category"], {"files": 0, "bytes": 0})
        value["files"] += 1
        value["bytes"] += row["bytes"]
    result = {"schema_version": 1, "files": rows, "payload_files": len(rows),
              "payload_bytes": sum(row["bytes"] for row in rows), "categories": totals,
              "excluded_metadata": list(INVENTORY_EXCLUSIONS),
              "metadata_scope": "recursive inventory/manifest hashes and volatile progress are excluded here; manifest attests inventory files",
              "baseline_local_inventory": BASELINE_INVENTORY,
              "publication_layout_selected": False, "originals_preserved": True,
              "scope": "managed local payload only; unknown user files omitted; not public hosting occupancy"}
    dump(output / "artifact-inventory.json", result)
    return result


def plots(output, record, reports):
    if record is None:
        for name in PLOTS:
            (output / name).unlink(missing_ok=True)
        return
    source = lp.build_source()
    prefix = QuantumCircuit(6)
    # Seven initialization/preparation instructions, followed by the exact
    # thirty-three-instruction literal first Strang period. Do not include the
    # next period's Hamiltonian prefix while searching for its first reset.
    for item in source.data[:40]:
        prefix.append(item.operation, [source.find_bit(bit).index for bit in item.qubits])
    figure = prefix.draw("mpl", fold=-1, style={"backgroundcolor": "#ffffff"})
    figure.suptitle("Actual initialization and first literal step at dt=1/134\nSource continues for 134 steps before final readout")
    save_svg(figure, output / "one-step.svg")
    fig, ax = plt.subplots(figsize=(9, 4))
    names = ("Source", "Routed rotations", "Actual Clifford+T")
    counts = record["actual_synthesis"]
    values = [sum(counts[key].values()) for key in ("source_counts", "routed_counts", "clifford_t_counts")]
    ax.bar(names, values, color=("#365e88", "#a56737", "#4f8171"))
    ax.set(yscale="log", ylabel="Instructions per complete trial", title="Literal resets/readouts retained; compiler barriers included")
    save_svg(fig, output / "counts.svg")
    fig, ax = plt.subplots(figsize=(8, 4))
    for report in reports:
        points = report.get("pareto_points", [])
        if points:
            ax.plot([p["peak_physical_qubits"] for p in points], [p["worst_case_runtime_all_attempts_ns"]/3.6e12 for p in points], marker="o", label=f"p={report['profile']['physical_error_rate']:g}")
    ax.set(xlabel="Peak physical qubits", ylabel="All 15,756 trials (hours)", title="Conditional serialized model; frozen profiles")
    if any(report.get("pareto_points") for report in reports):
        ax.legend()
    else:
        statuses = "\n".join(f"p={report['profile']['physical_error_rate']:g}: " +
                             report["status"].replace("_", " ") for report in reports)
        ax.text(.5, .5, "No feasible space/time point\n\n" + statuses,
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set(xticks=[], yticks=[])
    ax.grid(alpha=.2)
    save_svg(fig, output / "space-time.svg")
    fig, ax = plt.subplots(figsize=(9, 4))
    component_colors = dict(zip(("data_workspace_code_exposure", "factory_output_handoff_storage",
                                "logical_readout_scenario", "logical_initialization_scenario",
                                "controller_scenario", "factory_output_quality_model"),
                               ("#365e88", "#a56737", "#4f8171", "#b25555", "#876488", "#7a715e")))
    for index, report in enumerate(reports):
        points = report.get("pareto_points", [])
        if points:
            point = min(points, key=lambda p: p["peak_physical_qubits"])
            bottom = 0
            for name, value in point["failure_components_all_attempts"].items():
                ax.bar(index, value, bottom=bottom, color=component_colors[name],
                       label=name.replace("_", " ") if index == 0 else None)
                bottom += value
        elif report.get("controller_model_diagnostic"):
            value = report["controller_model_diagnostic"]["distance_independent_controller_model_contribution"]
            ax.bar(index, value, color=component_colors["controller_scenario"],
                   label="Controller contribution" if index == 0 else None)
            ax.annotate(f"{value:.4g}", (index, value), ha="center", xytext=(0, 4),
                        textcoords="offset points", fontsize=9)
    ax.axhline(record["complete_workload"]["physical_failure_budget"], color="#555555", linestyle="--", label="Allocated physical budget")
    any_points = any(report.get("pareto_points") for report in reports)
    ax.set(xticks=range(2), xticklabels=["p=1e-3", "p=1e-4"], ylabel="Additive physical failure model",
           title="Minimum-space points; factory readiness has a separate budget" if any_points else
                 "Actual controller diagnostic; other faults remain unevaluated")
    if not any_points and not any(report.get("controller_model_diagnostic") for report in reports):
        ax.text(.5, .5, "Physical profiles remain uncomputed", transform=ax.transAxes,
                ha="center", va="center")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_svg(fig, output / "failures.svg")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    c = record["error_contract"]
    labels = ("Uniform splitting", "Conditional realization", "Statistics")
    values = [c["splitting_bias_upper"], c["realization_Z0_bias_upper_proposal"], .03]
    ax.bar(labels, values, color=("#365e88", "#a56737", "#4f8171"))
    ax.set_yscale("log")
    ax.set_ylim(1e-6, .08)
    for index, value in enumerate(values):
        ax.annotate(f"{value:.4g}", (index, value), ha="center", xytext=(0, 4), textcoords="offset points", fontsize=9)
    ax.axhline(.05, color="#555555", linestyle="--", label="Total task target")
    ax.set(ylabel="Absolute Z0 error", title="Separate deterministic premises and statistical allowance")
    ax.legend()
    save_svg(fig, output / "error-budget.svg")


def readme():
    return """# Literal n3/r134 Lindblad physical workload

The fixed three-system, six-wire Z0 task is implemented here. All 15,756
complete fresh +i/+/+ input trials retain 673 resets and six readouts each.
The repeated population and hardware schedules are calculated, unexecuted.
The prior n2/r69 bundle remains unchanged and is referenced without rerunning.
The n4/r211 source ensemble check still rejects at its 256-branch cap;
its eight-wire density check also rejects the explicit six-wire ceiling.
Its synthesis/physical work remains unimplemented and allocation reserved.

Raw source, actual full routed rotations and actual ancilla-free Ross–Selinger
Clifford+T circuits are exported separately. An explicit opt-in six-wire
full-density CPTP checker retains all three system correlations and the full
joint output law, comparing with independent GKSL and gate-derived references.
Verifier-only local-matrix batching does not remove gates from the exported
circuits or resource counts. Resets are unobserved Kraus sums; reset histories
are not supplied as measured classical outputs. No postselection is applied.

Full-unitary lowering disables clean-input assumptions. SABRE seed 7 and level 0
preserve all reset/readout boundaries. Native synthesis exposes no seed: saved
sequences/hashes define its realization. Each QASM omits overall phase and
specifies complete measured trials, not a phase-preserving controlled oracle;
all three in-memory global phases are recorded in results.json.

The target is .01 splitting+.01 realization+.03 statistics=.05 absolute Z0
error. Actual guarded synthesis error, requested allowance 5e-5, representation
reserve 1e-7 and SDK/control/routing reserve 1e-7 remain separate. The reserves
are conditional norm premises, not inferred from density agreement; no full
representation/synthesis/hardware certificate is claimed. Deterministic bias
is not a stochastic physical fault and is not multiplied by the trial count.

Family budgets remain statistics .005, physical .0025 and factory readiness
.0025, divided across three cases: 1/600, 1/1200 and 1/1200 per case. Unused
allocations are not reassigned. There is no algorithm postselection or retry.
Complete factory retries retain their own accounting.

Physical error 1e-3/1e-4 uses the frozen default controller 1e-15 and unchanged
timing/interface profile through the serialized QDK model. Default-controller
infeasibility, if found, is retained explicitly with no fabricated space/time
point or substituted controller scenario. Startup, ignored readouts/resets,
idle/controller exposure, neighbor routing, transport/injection, port recycling
and full factory retries enter the model whenever a profile is evaluated.
The initial distance-independent controller diagnostic can reject a profile
before native factory search; other physical contributions then remain
unevaluated. Frozen Grover/PSSPC remains unchanged. Factory output quality is
charged once; readiness is separate. Offline classical preprocessing/reporting,
layout certification, decoder throughput and hardware calibration remain
explicit exclusions/assumptions, not quantum runtime.

Caps: one six-wire circuit; 20,000 source instructions; 1,000,000 routed,
synthesized and density-validation instructions; preparation/checking 1200 s,
each physical profile 180 s, family diagnostic 60 s and hard parent 1800 s.
Native rotation synthesis permits at most 10,720 attempts, including retries.
Each profile permits 24 top-level factory and 24 SurfaceCode queries, 1000
returned candidates per query and 24,000 total. These count adapter calls;
internal native construction before the first yield is bounded by the child
timeout. No candidate subset is truncated. Direct Trace calls are bounded by
one per returned candidate. Managed JSON/QASM is at most 32 MiB each; private
checkpoints at most 128 MiB. Timeout, overflow, errors and infeasibility remain
explicit with no stale or partial success.

Reproduce in the locked Python 3.12 environment:

    PYTHONPATH=src python scripts/generate_simulation_lindblad_three.py
    PYTHONPATH=src python -m pytest tests/test_simulation_lindblad_three_physical.py tests/test_simulation_bounded_density_instrument.py tests/test_simulation_lindblad_three_runner.py

Standalone: put all seven copied Python sources, requirements-lock.txt,
LICENSE.txt, references.bib and the notebook together. Default output is a new
subfolder. --reuse-prepared uses only runner-created private checkpoints keyed
by exact preparation source, native guard/caps, package versions, schema and
package/standalone import namespace, then checked against their saved hashes.
The default private cache is a task-specific subfolder of Python's canonical
temporary directory. Cache and public output are disjoint; no checkpoint/log
is distributed. Unknown user files are preserved and omitted from manifests.

artifact-inventory.csv/json list each managed payload file, category, logical
bytes and SHA-256. Recursive inventory/manifest metadata and volatile progress
are explicitly excluded there; the manifest attests inventory files. Baseline
asset totals are local measurements from worktree creation, not public-hosting
occupancy. Originals are preserved and publication layout remains undecided.
Source/package/QASM hashes identify the run; no-Git downloads record null
revision. Original MIT code, text and figures; primary algorithms and exact
upstream package sources are credited in references.bib.
"""


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad-three" if REPOSITORY else HERE / "generated-lindblad-three")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "openquantum-literal-lindblad-three-cache")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--_stage", choices=("family", "prepare", "profile"), help=argparse.SUPPRESS)
    parser.add_argument("--_job-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_profile-index", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._stage:
        if args._job_dir is None or not args._job_dir.resolve().is_relative_to(args.cache_dir.resolve()):
            parser.error("private stage directory must be inside its private cache")
        private_job = validate_private_job(args._job_dir, args.cache_dir)
        stage_worker(args._stage, private_job, args._profile_index, args.cache_dir.resolve())
        return
    paths = source_files()
    try:
        output, cache = validate_paths(args.output_dir, args.cache_dir, paths)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    deadline = start + CAPS["parent_seconds"]
    # Invalidate every known old output before a bounded attempt. Unknown
    # user files remain untouched and are never included in the manifest.
    for name in PUBLIC:
        (output / name).unlink(missing_ok=True)
    def parent_timeout(signum, frame):
        raise ExecutionCapExceeded("hard parent generation timeout1800s exceeded")
    signal.signal(signal.SIGALRM, parent_timeout)
    signal.setitimer(signal.ITIMER_REAL, CAPS["parent_seconds"])
    before = fingerprints()
    source_before = {name: digest(path.read_bytes()) for name, path in paths.items()}
    key = prepared_key(paths)
    job = validate_private_job(cache / "prepared" / key, cache)
    job.mkdir(parents=True, exist_ok=True)
    execution = {"run_caps": CAPS, "stage_status": {}, "repeated_trials_executed": False,
                 "whole_family_physical_completed": False}
    def progress(phase):
        dump(output / "progress.json", {"schema_version": 1, "phase": phase,
                                       "execution": execution, "wall_seconds": time.perf_counter()-start})
    progress("family_diagnostic")
    print("Recording prior n2 identity and checking unused n4 source schedule (cap60s)", flush=True)
    family_job = cache / "diagnostics" / key
    status = run_stage("family", family_job, cache, min(CAPS["family_diagnostic_seconds"], max(1, deadline-time.perf_counter())))
    execution["stage_status"]["family"] = status
    family = json.loads((family_job / "value.json").read_text()) if status["status"] == "completed" else lp.family_status_records(check_rejections=False)
    saved_status = checkpoint_status(job, key)
    reusable = bool(saved_status)
    if args.reuse_prepared and reusable:
        status = saved_status
        status = {**status, "reused_exact_private_checkpoint": True}
    else:
        print("Preparing actual full-unitary Clifford+T trial (cap1200s)", flush=True)
        progress("native_preparation")
        status = run_stage("prepare", job, cache, min(CAPS["prepare_seconds"], max(1, deadline-time.perf_counter())))
        if status["status"] == "completed":
            dump(job / "prepare-status.json", status)
            saved_status = checkpoint_status(job, key)
            if not saved_status:
                raise ValueError("new complete preparation checkpoint does not match its source identity and hash")
    execution["stage_status"]["prepare"] = status
    record, reports, rows = None, [], []
    if status["status"] == "completed":
        prepared = load_prepared(job / "prepared.pickle")
        record = prepared.as_record()
        hashes = {}
        for name, circuit in (("source-trial.qasm", prepared.synthesized.source), ("routed-rotation-trial.qasm", prepared.synthesized.routed), ("clifford-t-trial.qasm", prepared.synthesized.circuit)):
            raw = qasm3.dumps(circuit).encode()
            if len(raw) > CAPS["managed_json_qasm_bytes"]:
                raise ExecutionCapExceeded("managed QASM exceeds 32 MiB")
            (output / name).write_bytes(raw)
            hashes[name] = digest(raw)
        record["qasm_sha256"] = hashes
        record["qasm_export"] = {
            "circuits": {"source-trial.qasm": "raw literal source",
                         "routed-rotation-trial.qasm": "actual complete routed rotation-basis circuit",
                         "clifford-t-trial.qasm": "actual complete synthesized Clifford+T circuit"},
            "in_memory_global_phase_radians": {"source-trial.qasm": float(prepared.synthesized.source.global_phase),
                "routed-rotation-trial.qasm": float(prepared.synthesized.routed.global_phase),
                "clifford-t-trial.qasm": float(prepared.synthesized.circuit.global_phase)},
            "scope": "all quantum instructions and instrument events exported; QASM omits overall phase and describes measured trials, not a phase-preserving controlled oracle",
        }
        for index, profile in enumerate(lp.fixed_profiles()):
            name = f"p{profile.physical_error_rate:.0e}"
            print(f"Estimating {name} (cap180s, top-levelqueries24)", flush=True)
            profile_job = validate_private_job(job / name, cache)
            profile_job.mkdir(exist_ok=True)
            shutil_source = job / "prepared.pickle"
            import shutil
            shutil.copyfile(shutil_source, profile_job / "prepared.pickle")
            dump(profile_job / "prepare-status.json", saved_status)
            progress(f"physical_profile_{index}")
            stage_status = run_stage("profile", profile_job, cache, min(CAPS["physical_seconds_per_profile"], max(1, deadline-time.perf_counter())), profile_index=index)
            execution["stage_status"][name] = stage_status
            report = json.loads((profile_job / "value.json").read_text()) if stage_status["status"] == "completed" else {
                **stage_status, "profile": asdict(profile), "pareto_points": [], "all_feasible_points": [],
                "complete_algorithm_attempts": lp.COMPLETE_TRIALS,
                "physical_failure_budget": record["complete_workload"]["physical_failure_budget"],
                "factory_availability_budget": record["complete_workload"]["factory_availability_budget"]}
            reports.append(report)
            point = min(report["pareto_points"], key=lambda p: p["peak_physical_qubits"]) if report.get("pareto_points") else None
            rows.append({"case_id": lp.CASE_ID, "physical_error": profile.physical_error_rate, "status": report["status"],
                         "complete_trials": lp.COMPLETE_TRIALS, "distance": point["data_code_distance"] if point else None,
                         "peak_physical_qubits": point["peak_physical_qubits"] if point else None,
                         "all_trial_runtime_hours": point["worst_case_runtime_all_attempts_ns"]/3.6e12 if point else None,
                         "full_physical_failure_model": point["full_physical_failure_model"] if point else None,
                         "factory_readiness_failure_model": point["factory_supply"]["availability_failure_union_bound"] if point else None,
                         "controller_failure_model": report.get("controller_model_diagnostic", {}).get("distance_independent_controller_model_contribution"),
                         "controller_decisions_all_trials": report.get("controller_model_diagnostic", {}).get("controller_decisions_all_trials"),
                         "pareto_points": len(report.get("pareto_points", []))})
            print(f"Computed {name}: {report['status']}", flush=True)
    else:
        for name in ("source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm"):
            (output / name).unlink(missing_ok=True)
        for profile in lp.fixed_profiles():
            report = {"status": "uncomputed_preparation_unavailable", "profile": asdict(profile), "pareto_points": [], "all_feasible_points": [], "complete_algorithm_attempts": lp.COMPLETE_TRIALS}
            reports.append(report)
            rows.append({"case_id": lp.CASE_ID, "physical_error": profile.physical_error_rate, "status": report["status"], "complete_trials": lp.COMPLETE_TRIALS,
                         "distance": None, "peak_physical_qubits": None, "all_trial_runtime_hours": None, "full_physical_failure_model": None, "factory_readiness_failure_model": None,
                         "controller_failure_model": None, "controller_decisions_all_trials": None, "pareto_points": 0})
    if time.perf_counter() > deadline:
        raise ExecutionCapExceeded("parent generation cap1800s exceeded")
    dump(output / "results.json", {"schema_version": 1, "prepared": record, "physical_profiles": reports,
                                  "family": family, "family_allocations": lp.family_allocations(), "execution": execution})
    progress("artifact_export_and_verification")
    write_csv(output / "summary.csv", rows)
    write_csv(output / "family-status.csv", [{"n_system": r["n_system"], "steps": r["selected_steps"], "status": r["status"],
              "physical_status": r.get("physical_status", "computed_conditional_profiles" if record is not None and all(p["status"].startswith("computed_") for p in reports) else "infeasible_or_uncomputed_profiles"), "complete_trials": r["complete_trials"], "synthesis_performed": r["n_system"] == 3 and record is not None,
              "rejection_reproduced": r.get("rejection_reproduced", False),
              "density_checker_status": r.get("density_checker_status"),
              "density_width_rejection_reproduced": r.get("density_width_rejection_reproduced", False),
              "density_required_wires": r.get("density_required_wires"),
              "density_supported_wires": r.get("density_supported_wires"),
              "allocation_reassigned": False} for r in family.values()])
    plots(output, record, reports)
    (output / "README.md").write_text(readme())
    source_hashes = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        if digest(raw) != source_before[name]:
            raise AssertionError("runtime source changed during bounded generation; no coherent manifest may be issued")
        (output / name).write_bytes(raw)
        source_hashes[name] = digest(raw)
    extras = {"LICENSE.txt": (ROOT / "LICENSE").read_bytes() if REPOSITORY else (HERE / "LICENSE.txt").read_bytes(),
              "requirements-lock.txt": (ROOT / "requirements-lock.txt").read_bytes() if REPOSITORY else (HERE / "requirements-lock.txt").read_bytes(),
              "references.bib": ((ROOT / "references/simulation.bib").read_bytes()+b"\n"+(ROOT / "references/resources.bib").read_bytes()) if REPOSITORY else (HERE / "references.bib").read_bytes()}
    for name, raw in extras.items():
        (output / name).write_bytes(raw)
        source_hashes[name] = digest(raw)
    after = fingerprints()
    if before != after:
        raise AssertionError("frozen shared/backend baseline changed during generation")
    if time.perf_counter() > deadline:
        raise ExecutionCapExceeded("parent generation cap1800s exceeded before manifest attestation")
    inventory = artifact_inventory(output)
    progress("completed_bounded_generation")
    dump(output / "manifest.json", {"schema_version": 1, **revision(paths), "source_sha256": source_hashes,
                                   "artifacts_sha256": managed_hashes(output), "python": platform.python_version(),
                                   "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")},
                                   "device": "local CPU; bounded ideal density checks and native conditional models; no hardware/GPU run",
                                   "transpiler_seed": 7, "synthesis_seed": None,
                                   "synthesis_seed_scope": "native backend has no seed API; saved actual sequences/QASM hashes identify realization",
                                   "run_caps": CAPS, "execution": execution, "wall_seconds": time.perf_counter()-start,
                                   "artifact_inventory": {"payload_files": inventory["payload_files"], "payload_bytes": inventory["payload_bytes"],
                                       "excluded_metadata": inventory["excluded_metadata"], "baseline_local_inventory": BASELINE_INVENTORY},
                                   "frozen_before": before, "frozen_after": after, "private_cache_distributed": False,
                                   "whole_family_physical_completed": False})
    print(f"Recorded bounded n3 attempt: preparation={execution['stage_status']['prepare']['status']}; n2 preserved and n4 unimplemented", flush=True)


def main():
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        _main()
    except (ExecutionCapExceeded, ValueError, AssertionError, OSError) as error:
        # Failed public attempts cannot leave a stale or partial success.
        # Child-stage preflight errors must not mutate the public bundle.
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad-three" if REPOSITORY else HERE / "generated-lindblad-three")
        parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "openquantum-literal-lindblad-three-cache")
        parser.add_argument("--_stage", choices=("family", "prepare", "profile"))
        args, _ = parser.parse_known_args()
        if args._stage is not None:
            raise
        signal.setitimer(signal.ITIMER_REAL, 0)
        output, _ = validate_paths(args.output_dir, args.cache_dir, source_files())
        output.mkdir(parents=True, exist_ok=True)
        for name in PUBLIC:
            (output / name).unlink(missing_ok=True)
        is_cap = isinstance(error, ExecutionCapExceeded)
        reason = str(error) if is_cap else "bounded parent stage failed; no coherent result or source attestation retained"
        failure = {"schema_version": 1, "status": "uncomputed_parent_execution_cap" if is_cap else "uncomputed_parent_stage_error",
                   "reason": reason, "error_type": type(error).__name__,
                   "prepared": None, "physical_profiles": [], "run_caps": CAPS,
                   "whole_family_physical_completed": False, "repeated_trials_executed": False}
        dump(output / "results.json", failure)
        dump(output / "manifest.json", {"status": failure["status"], "artifacts_sha256": {"results.json": digest((output / "results.json").read_bytes())}, "run_caps": CAPS})
        print(f"Recorded uncomputed bounded attempt: {reason}", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
