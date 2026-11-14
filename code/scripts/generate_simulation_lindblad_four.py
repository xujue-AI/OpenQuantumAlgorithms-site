#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded complete physical accounting of literal n4/r211/Z0.

Native preparation and each fixed physical profile run in capped children.
Prior n2/n3 results and historical n4 rejections are preserved.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

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
import tempfile
import time


def configure_import_policy():
    """Use a fresh task prefix when the caller has not supplied one."""
    sys.dont_write_bytecode = True
    if sys.pycache_prefix is None:
        sys.pycache_prefix = str(Path(tempfile.gettempdir()).resolve() /
            "openquantum-literal-lindblad-four-bytecode" / os.urandom(16).hex())


configure_import_policy()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (HERE.name == "scripts" and
              (SOURCE / "openquantum/simulation/lindblad_four_physical.py").is_file() and
              not (HERE / "lindblad_four_physical.py").is_file())
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-literal-lindblad-four"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3

if REPOSITORY:
    from openquantum import instrument_resources as ir
    from openquantum.simulation import lindblad_four_physical as lp
    from openquantum.simulation import lindblad_four_compile as lc
else:
    import instrument_resources as ir
    import lindblad_four_physical as lp
    import lindblad_four_compile as lc

SOURCES = ("instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py",
           "open_systems.py", "bounded_density_instrument.py", "lindblad_four_density.py",
           "lindblad_four_compile.py", "lindblad_four_physical.py", "generate_simulation_lindblad_four.py")
FROZEN_SOURCE_SNAPSHOTS = ("instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py", "open_systems.py", "bounded_density_instrument.py")
PLOTS = ("one-step.svg", "counts.svg", "space-time.svg", "failures.svg", "error-budget.svg")
EXTRAS = ("LICENSE.txt", "requirements-lock.txt", "references.bib")
PUBLIC = (*SOURCES, *PLOTS, *EXTRAS, "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
          "results.json", "source-preflight.json", "summary.csv", "family-status.csv", "artifact-inventory.csv", "artifact-inventory.json", "README.md", "manifest.json", "progress.json")
FROZEN = ("src/openquantum/instrument_resources.py", "src/openquantum/grover.py",
          "scripts/estimate_resources.py", "assets/generated/resources/grover-surface-code.json",
          "assets/generated/resources/grover-surface-code.csv")
CAPS = lp.run_caps()
CHECKPOINT_SCHEMA = "literal-n4-density-v1"
INVENTORY_EXCLUSIONS = ("artifact-inventory.csv", "artifact-inventory.json", "manifest.json", "progress.json")
BASELINE_INVENTORY = {
    "tracked_generated_assets_files_at_start": 7159,
    "tracked_generated_assets_bytes_at_start": 358968120,
    "preserved_prior_files": 66,
    "preserved_prior_total_bytes": 11701228,
    "scope": "local tracked generated source assets at worktree creation; not public hosting occupancy",
}
ACTIVE_PREFLIGHT = None
SOURCE_QASMS = ("source-trial.qasm", "routed-rotation-trial.qasm")


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
    if path.is_symlink():
        raise ValueError("managed JSON must not overwrite a symbolic link")
    raw = (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > CAPS["managed_json_qasm_bytes"]:
        raise ExecutionCapExceeded("managed JSON exceeds the predeclared 32 MiB cap")
    path.write_bytes(raw)


def read_managed_json(path):
    if path.is_symlink() or path.stat().st_size > CAPS["managed_json_qasm_bytes"]:
        raise ValueError("managed JSON must be a bounded regular file")
    with path.open("rb") as handle:
        raw = handle.read(CAPS["managed_json_qasm_bytes"]+1)
    if len(raw) > CAPS["managed_json_qasm_bytes"]:
        raise ExecutionCapExceeded("managed JSON exceeds32MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("managed JSON must contain an object")
    return value


def bounded_file_hash(path):
    if path.is_symlink():
        raise ValueError("managed file hashes must not follow symbolic links")
    if path.stat().st_size > CAPS["managed_json_qasm_bytes"]:
        raise ExecutionCapExceeded("managed JSON/QASM exceeds 32 MiB")
    result, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while block := handle.read(65536):
            total += len(block)
            if total > CAPS["managed_json_qasm_bytes"]:
                raise ExecutionCapExceeded("managed JSON/QASM exceeds 32 MiB")
            result.update(block)
    return result.hexdigest()


def dump_qasm(path, circuit):
    if path.is_symlink():
        raise ValueError("managed QASM must not be a symbolic link")
    raw = qasm3.dumps(circuit).encode()
    if len(raw) > CAPS["managed_json_qasm_bytes"]:
        raise ExecutionCapExceeded("managed QASM exceeds 32 MiB")
    path.write_bytes(raw)
    return digest(raw)


def source_files():
    return {name: (Path(__file__) if name.startswith("generate_") else
                   SOURCE / "openquantum" / name if name == "instrument_resources.py" else
                   SOURCE / "openquantum/simulation" / name) if REPOSITORY else HERE / name
            for name in SOURCES}


def verify_runtime_sources(paths):
    """Verify actual imported modules against this task's intended snapshot."""
    hashes = {}
    for name, expected in paths.items():
        if name.startswith("generate_"):
            if Path(__file__).resolve() != expected.resolve():
                raise ValueError("runner source does not match the intended snapshot")
            continue
        qualified = ("openquantum.instrument_resources" if name == "instrument_resources.py" else
                     "openquantum.simulation." + Path(name).stem) if REPOSITORY else Path(name).stem
        module = importlib.import_module(qualified)
        actual = getattr(module, "__file__", None)
        if actual is None or Path(actual).resolve() != expected.resolve():
            raise ValueError("runtime import differs from the intended task source: " + name)
        hashes[name] = bounded_file_hash(Path(actual))
    return {"import_mode": "package" if REPOSITORY else "standalone",
            "actual_module_paths_verified": True, "source_sha256": hashes,
            "bytecode_writes_disabled": sys.dont_write_bytecode,
            "bytecode_prefix_configured": sys.pycache_prefix is not None,
            "import_policy": "fresh private bytecode prefix; bytecode writes disabled; no environment mutation"}


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
    owned = ROOT / "assets/generated/simulation/lindblad-four"
    if REPOSITORY and output.is_relative_to(ROOT) and not output.is_relative_to(owned):
        raise ValueError("repository output must remain inside assets/generated/simulation/lindblad-four")
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
        "execution_helpers_sha256": {function.__name__: digest(inspect.getsource(function).encode())
            for function in (stage_worker, run_stage, checkpoint_status, preflight_status,
                             save_prepared, load_prepared, validate_private_job, dump_qasm,
                             capture_sources, publish_preflight, recover_preflight, verify_runtime_sources,
                             configure_import_policy, read_managed_json, bounded_file_hash,
                             prepared_hash, export_captured_sources)},
        "preserve_full_unitary": True,
        "import_policy": "fresh task-private PYTHONPYCACHEPREFIX and PYTHONDONTWRITEBYTECODE=1",
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
                raise ExecutionCapExceeded("private prepared checkpoint exceeds 256 MiB")
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
        raise ExecutionCapExceeded("private prepared checkpoint exceeds 256 MiB")
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepared_hash(path):
    if path.is_symlink():
        raise ValueError("private checkpoints must not be symbolic links")
    if path.stat().st_size > CAPS["private_checkpoint_bytes"]:
        raise ExecutionCapExceeded("private prepared checkpoint exceeds 256 MiB")
    result, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while block := handle.read(65536):
            total += len(block)
            if total > CAPS["private_checkpoint_bytes"]:
                raise ExecutionCapExceeded("private prepared checkpoint exceeds 256 MiB")
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
    for name in ("status.json", "value.json", "worker.log", "prepared.pickle", "prepared.pickle.tmp", "prepare-status.json",
                 "source-trial.qasm", "routed-rotation-trial.qasm", "source-preflight.json"):
        if (job / name).is_symlink():
            raise ValueError("managed private job targets must not be symbolic links")
    return job.resolve()


def checkpoint_status(job, identity):
    """Verify schema, namespace and bounded content hash before unpickling."""
    status_path = job / "prepare-status.json"
    if not status_path.is_file() or status_path.stat().st_size > CAPS["managed_json_qasm_bytes"]:
        return {}
    try:
        status = read_managed_json(status_path)
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


def preflight_status(job, identity, *, invocation_id=None):
    """Accept only completed source evidence bound to the current source/policy."""
    if not (job / "status.json").is_file() or not (job / "value.json").is_file():
        return {}
    status = read_managed_json(job / "status.json")
    if (status.get("status") != "completed" or status.get("source_identity") != identity or
            status.get("source_schema") != CHECKPOINT_SCHEMA):
        return {}
    if invocation_id is not None and status.get("invocation_id") != invocation_id:
        return {}
    value = read_managed_json(job / "value.json")
    if status.get("source_record_sha256") != bounded_file_hash(job / "value.json"):
        return {}
    hashes = value.get("qasm_sha256", {})
    if set(hashes) != set(SOURCE_QASMS):
        return {}
    if any(not (job / name).is_file() or bounded_file_hash(job / name) != hashes[name] for name in SOURCE_QASMS):
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
    native = lc.gridsynth_unitary
    counts = {"calls": 0}
    def guarded_synthesis(*args, **kwargs):
        if counts["calls"] >= CAPS["native_rotation_synthesis_attempts_upper_bound"]:
            raise ExecutionCapExceeded("native rotation synthesis attempt cap23632 exceeded")
        counts["calls"] += 1
        return native(*args, **kwargs)
    lc.gridsynth_unitary = guarded_synthesis
    try:
        yield counts
    finally:
        lc.gridsynth_unitary = native


def stage_worker(stage, job, profile_index, cache, invocation_id=None):
    began = time.perf_counter()
    counts = None
    try:
        runtime_sources = verify_runtime_sources(source_files())
        if stage == "preflight":
            source_identity = prepared_key(source_files())
            preflight = lp.source_preflight()
            value = preflight.as_record()
            value["qasm_sha256"] = {name: dump_qasm(job / name, circuit)
                for name, circuit in zip(SOURCE_QASMS, (preflight.source, preflight.routed))}
            value["qasm_export"] = {
                "circuits": {"source-trial.qasm": "complete raw literal source",
                             "routed-rotation-trial.qasm": "complete routed rotation-basis source"},
                "in_memory_global_phase_radians": {"source-trial.qasm": float(preflight.source.global_phase),
                    "routed-rotation-trial.qasm": float(preflight.routed.global_phase)},
                "scope": "all source events retained; measured QASM omits overall phase and is not a controlled oracle",
            }
            if prepared_key(source_files()) != source_identity:
                raise ValueError("source or execution policy changed during source preflight")
            dump(job / "value.json", value)
        elif stage == "family":
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
        status["runtime_source_validation"] = runtime_sources
        if stage == "preflight":
            status.update(source_identity=source_identity, source_schema=CHECKPOINT_SCHEMA,
                          source_record_sha256=bounded_file_hash(job / "value.json"),
                          invocation_id=invocation_id)
        elif stage == "prepare":
            status["prepared_checkpoint_sha256"] = prepared_hash(job / "prepared.pickle")
            status["checkpoint_schema"] = CHECKPOINT_SCHEMA
            status["checkpoint_identity"] = preparation_identity
    except ExecutionCapExceeded as error:
        status = {"status": "uncomputed_execution_cap", "reason": str(error), "wall_seconds": time.perf_counter()-began}
    except lc.CompilerLimitExceeded as error:
        status = {**error.as_record(), "reason": str(error), "wall_seconds": time.perf_counter()-began}
    except Exception as error:
        status = {"status": "uncomputed_stage_error", "error_type": type(error).__name__,
                  "reason": "bounded stage failed; diagnostic retained only in private local log", "wall_seconds": time.perf_counter()-began}
        import traceback
        traceback.print_exc()
    if stage == "prepare" and counts is not None:
        status["native_rotation_synthesis_attempts"] = counts["calls"]
        status["native_rotation_synthesis_attempts_upper_bound"] = CAPS["native_rotation_synthesis_attempts_upper_bound"]
    dump(job / "status.json", status)


def run_stage(stage, job, cache, timeout, *, profile_index=0, invocation_id=None):
    job = validate_private_job(job, cache)
    cache_identity = prepared_key(source_files())
    cache_token = invocation_id or os.urandom(16).hex()
    bytecode = validate_private_job(cache / "python-bytecode" / cache_identity /
                                   ("package" if REPOSITORY else "standalone") / cache_token, cache)
    job.mkdir(parents=True, exist_ok=True)
    (job / "status.json").unlink(missing_ok=True)
    (job / "value.json").unlink(missing_ok=True)
    if stage == "prepare":
        (job / "prepare-status.json").unlink(missing_ok=True)
    elif stage == "preflight":
        for name in SOURCE_QASMS:
            (job / name).unlink(missing_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--_stage", stage,
               "--_job-dir", str(job), "--cache-dir", str(cache), "--_profile-index", str(profile_index)]
    if invocation_id is not None:
        command.extend(("--_invocation-id", invocation_id))
    env = dict(os.environ)
    env.update(PYTHONPATH=str(SOURCE if REPOSITORY else HERE), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX=str(bytecode))
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
    return read_managed_json(job / "status.json")


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


def capture_sources(paths):
    """Capture the intended public inputs before a stage, never private files."""
    sources = {}
    for name, path in paths.items():
        if path.stat().st_size > CAPS["managed_json_qasm_bytes"]:
            raise ExecutionCapExceeded("a source snapshot exceeds its 32 MiB capture cap")
        sources[name] = path.read_bytes()
        if len(sources[name]) > CAPS["managed_json_qasm_bytes"]:
            raise ExecutionCapExceeded("a source snapshot exceeds its 32 MiB capture cap")
    extras = {"LICENSE.txt": (ROOT / "LICENSE").read_bytes() if REPOSITORY else (HERE / "LICENSE.txt").read_bytes(),
              "requirements-lock.txt": (ROOT / "requirements-lock.txt").read_bytes() if REPOSITORY else (HERE / "requirements-lock.txt").read_bytes(),
              "references.bib": ((ROOT / "references/simulation.bib").read_bytes()+b"\n"+(ROOT / "references/resources.bib").read_bytes()) if REPOSITORY else (HERE / "references.bib").read_bytes()}
    if any(len(raw) > CAPS["managed_json_qasm_bytes"] for raw in extras.values()):
        raise ExecutionCapExceeded("a dependency snapshot exceeds its 32 MiB capture cap")
    return sources, extras


def export_captured_sources(output, sources, extras):
    hashes = {}
    for name, raw in {**sources, **extras}.items():
        if (output / name).is_symlink():
            raise ValueError("captured public sources must not overwrite symbolic links")
        (output / name).write_bytes(raw)
        hashes[name] = digest(raw)
    return hashes


def publish_preflight(output, job, identity, invocation_id):
    status = preflight_status(job, identity, invocation_id=invocation_id)
    if not status:
        raise ValueError("current source preflight lacks matching identity, invocation and hashes")
    value = read_managed_json(job / "value.json")
    for name in SOURCE_QASMS:
        raw = (job / name).read_bytes()
        if len(raw) > CAPS["managed_json_qasm_bytes"] or digest(raw) != value["qasm_sha256"][name]:
            raise ValueError("source preflight QASM changed before export")
        if (output / name).is_symlink():
            raise ValueError("preflight exports must not overwrite symbolic links")
        (output / name).write_bytes(raw)
    dump(output / "source-preflight.json", value)
    return value, status


def recover_preflight(output, cache, paths):
    """Recover only this invocation's verified source after a later failure."""
    context = ACTIVE_PREFLIGHT
    if (not context or output != context["output"] or cache != context["cache"]):
        return None
    try:
        current_sources, current_extras = capture_sources(paths)
        if (current_sources != context["sources"] or current_extras != context["extras"] or
                prepared_key(paths) != context["identity"] or fingerprints() != context["frozen_before"]):
            return None
        job = validate_private_job(context["job"], cache)
        value, status = publish_preflight(output, job, context["identity"], context["invocation_id"])
        hashes = export_captured_sources(output, context["sources"], context["extras"])
        (output / "README.md").write_text(context["readme"])
        return {"record": value, "status": status, "source_sha256": hashes,
                "family": context.get("family", {})}
    except (ExecutionCapExceeded, ValueError, AssertionError, OSError, KeyError):
        return None


def plots(output, record, reports, *, source_preflight=None):
    if record is None and source_preflight is None:
        for name in PLOTS:
            (output / name).unlink(missing_ok=True)
        return
    source = lp.build_source()
    prefix = QuantumCircuit(8)
    # Nine initialization/preparation instructions, followed by the exact
    # forty-seven-instruction literal first Strang period. Do not include the
    # next period's Hamiltonian prefix while searching for its first reset.
    for item in source.data[:56]:
        prefix.append(item.operation, [source.find_bit(bit).index for bit in item.qubits])
    figure = prefix.draw("mpl", fold=-1, style={"backgroundcolor": "#ffffff"})
    figure.suptitle("Actual initialization and first literal step at dt=1/211\nSource continues for 211 steps before final readout")
    save_svg(figure, output / "one-step.svg")
    fig, ax = plt.subplots(figsize=(9, 4))
    counts = record["actual_synthesis"] if record is not None else source_preflight["actual_routing"]
    keys = ("source_counts", "routed_counts", "clifford_t_counts") if record is not None else ("source_counts", "routed_counts")
    names = ("Source", "Routed rotations", "Actual Clifford+T")[:len(keys)]
    values = [sum(counts[key].values()) for key in keys]
    ax.bar(names, values, color=("#365e88", "#a56737", "#4f8171")[:len(keys)])
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
        ax.text(.5, .5, "No computed qubit/time point\n\n" + statuses,
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
    physical_budget = (record["complete_workload"]["physical_failure_budget"] if record is not None else
                       lp.family_allocations()["case_allocations"]["4"]["physical_failure_budget"])
    ax.axhline(physical_budget, color="#555555", linestyle="--", label="Allocated physical budget")
    any_points = any(report.get("pareto_points") for report in reports)
    ax.set(xticks=range(2), xticklabels=["p=1e-3", "p=1e-4"], ylabel="Additive physical failure model",
           title="Minimum-space points; factory readiness has a separate budget" if any_points else
                 "Actual controller diagnostic; other faults remain unevaluated")
    if not any_points and not any(report.get("controller_model_diagnostic") for report in reports):
        ax.text(.5, .5, "Physical profiles remain uncomputed", transform=ax.transAxes,
                ha="center", va="center")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_svg(fig, output / "failures.svg")
    if record is None:
        (output / "error-budget.svg").unlink(missing_ok=True)
        return
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
    return """# Literal n4/r211 correlated Lindblad workload

The fixed four-system, eight-wire Z0 task keeps t1, J0.6, hx0.35,
rates0.8/0.5/0.5/0.5 and +i/+/+/+ input. Every one of 15,756 complete
fresh trials retains1,481 resets and eight readouts. Repeated populations and
hardware schedules are calculated and unexecuted. Prior n2/n3 bundles and
all historical n4 rejection labels are preserved; no allocation is reassigned.

source-preflight.json and the source_preflight field in results.json retain
completed raw-source/routed-rotation evidence even if native preparation later
fails. Full Clifford+T and physical records exist only after their own completed
stages; missing/cap/timeout/numerical/unsupported/infeasible statuses remain
explicit. A parent watchdog failure discards all CT/physical success and can
retain only this invocation's hash-verified source evidence and snapshots.

The new density API requires explicit EightWireDensityLimits and preserves the
whole 256x256 complex128 state (1 MiB), all 16 four-system probabilities and the
full 16x16 reduced density. Reset sums both unobserved Kraus terms; ignored
readouts dephase. No reset-history bits, trajectory pruning or product-of-
marginals replacement is used. Classical local-matrix batching does not remove
quantum occurrences. Once-at-output numerical trace normalization and raw trace
are disclosed; they are not algorithm postselection. The predeclared 1e-8 state
trace guard and 1e-10 density-comparison guard are nominal numerical checks,
not representation/compiler/hardware certificates.

A separate fixed-task compiler API requires explicit source 20k/routed 100k/
Clifford+T 2m limits. Existing defaults, the six-wire density ceiling, 256-branch
ensemble ceiling and shared 1m materialization ceiling remain unchanged. The
actual completed CT count is separately tested against the old 1m guard; this
validation is not an executed legacy synthesis attempt. Full-unitary level0
lowering, fixed layout, bidirectional line and SABRE seed 7 preserve every source
reset/readout. Actual Ross--Selinger sequences receive the unchanged whole-
trial 5e-5 request; repeated targets retain every emitted gate. Native synthesis
has no seed API. Complete raw/routed/CT QASM hashes identify realizations;
measured-instrument QASM omits overall phase and is not a controlled oracle.
In-memory phases are recorded for coherent unitary-segment reuse.

Accuracy remains .01 splitting/.01 realization/.03 statistics, target .05.
Representation and SDK/control/routing norm premises remain separately 1e-7
each. Under those premises, unconditional Z0 bias is at most 2 eta; deterministic bias is
not a stochastic fault and is not multiplied by 15,756. Family allocations are
statistics .005/physical .0025/readiness .0025, divided into 1/600, 1/1200, 1/1200
per case. No better controller, physical timing, geometry or allocation is
substituted. Physical p1e-3/p1e-4 and controller 1e-15 are frozen. Actual event
and T-wire counts determine controller/factory-retry diagnostics. An excessive
controller term or unsupported native materialization domain has no fabricated
qubits/runtime. Other fault components remain unevaluated when early rejection
prevents native search. The serialized model includes startup, resets, ignored
readouts, dispatch, active/idle exposure, neighbor routing, transport/injection,
port recycling, handoff and complete capped pipeline retries when evaluated.
Output quality and readiness budgets stay separate; frozen Grover/PSSPC is
unchanged. These engineering-model terms are not measured hardware-failure
lower bounds. Offline preprocessing/reporting, layout/decoder certification
and calibration remain exclusions; no advantage is claimed.

Caps: source preflight and historical diagnostics share 180s, native preparation
1800s, each physical profile 180s and hard parent 2400s through artifact export.
Native attempts 23,632; odd distances 3..49; factory retries <=16. Each profile
permits 24 top-level factory/24 SurfaceCode queries, 1000 candidates/query and 24k
total. Internal native work before yield is timeout-bounded; no candidate
subset is truncated. Direct Trace calls are at most returned candidates.
JSON/QASM 32 MiB each, private prepared pickle 256 MiB with atomic bounded writes
and streaming hashes. Infeasible/failed stages never retain stale success.

Reproduce with the locked Python 3.12 environment:

    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/generate_simulation_lindblad_four.py

Standalone: place all nine Python snapshots, requirements-lock.txt, LICENSE.txt,
references.bib and the notebook together. Default output is a new subfolder.
The runner disables bytecode writes and uses a fresh task-private prefix if the
caller did not supply one; every child uses a fresh private bytecode prefix.
Actual imported module locations and source hashes are verified against the
own snapshot; no paths, logs, checkpoints or credentials are distributed.
--reuse-prepared accepts only exact source/import/schema/native-policy/cap-
bound checkpoints with matching hashes. Source preflight is fresh per invocation
and additionally bound to its invocation ID and both exported QASM hashes.
Private cache and public output are disjoint. Unknown user files are preserved
and omitted from the managed manifest.

artifact-inventory.csv/json itemize local logical bytes, SHA256 and categories.
Recursive inventory/manifest metadata and volatile progress are excluded there;
the manifest attests the inventory files. Baseline 7,159 files/358,968,120 bytes
is local tracked generated storage at worktree creation, not public hosting
occupancy. All originals remain; publication layout is coordinator-owned.
Text/circuits/figures are original MIT work; exact primary package/paper credits
are in references.bib. No main/shared-export/workflow/publication change occurs.
"""


def _main():
    global ACTIVE_PREFLIGHT
    ACTIVE_PREFLIGHT = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad-four" if REPOSITORY else HERE / "generated-lindblad-four")
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "openquantum-literal-lindblad-four-cache")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--_stage", choices=("preflight", "family", "prepare", "profile"), help=argparse.SUPPRESS)
    parser.add_argument("--_job-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_profile-index", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_invocation-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._stage:
        if args._job_dir is None or not args._job_dir.resolve().is_relative_to(args.cache_dir.resolve()):
            parser.error("private stage directory must be inside its private cache")
        private_job = validate_private_job(args._job_dir, args.cache_dir)
        stage_worker(args._stage, private_job, args._profile_index, args.cache_dir.resolve(), args._invocation_id)
        return
    paths = source_files()
    runtime_sources = verify_runtime_sources(paths)
    try:
        output, cache = validate_paths(args.output_dir, args.cache_dir, paths)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    deadline = start + CAPS["parent_seconds"]
    for name in PUBLIC:
        (output / name).unlink(missing_ok=True)
    def parent_timeout(signum, frame):
        raise ExecutionCapExceeded("hard parent generation timeout2400s exceeded")
    signal.signal(signal.SIGALRM, parent_timeout)
    signal.setitimer(signal.ITIMER_REAL, CAPS["parent_seconds"])
    before = fingerprints()
    sources, extras = capture_sources(paths)
    source_hashes = {name: digest(raw) for name, raw in {**sources, **extras}.items()}
    key = prepared_key(paths)
    invocation_id = os.urandom(16).hex()
    execution = {"run_caps": CAPS, "stage_status": {}, "repeated_trials_executed": False,
                 "whole_family_physical_completed": False}
    def progress(phase):
        dump(output / "progress.json", {"schema_version": 1, "phase": phase,
                                       "execution": execution, "wall_seconds": time.perf_counter()-start})
    diagnostic_deadline = time.perf_counter() + CAPS["diagnostic_aggregate_seconds"]
    progress("source_preflight")
    print("Checking complete eight-wire source/routing density (aggregate diagnostics cap180s)", flush=True)
    source_job = validate_private_job(cache / "source-preflight" / key, cache)
    source_status = run_stage("preflight", source_job, cache,
        min(CAPS["source_preflight_seconds"], max(1, deadline-time.perf_counter()),
            max(1, diagnostic_deadline-time.perf_counter())), invocation_id=invocation_id)
    execution["stage_status"]["source_preflight"] = source_status
    source_record = None
    if source_status["status"] == "completed":
        source_record, source_status = publish_preflight(output, source_job, key, invocation_id)
        export_captured_sources(output, sources, extras)
        (output / "README.md").write_text(readme())
        ACTIVE_PREFLIGHT = {"output": output, "cache": cache, "job": source_job,
            "identity": key, "invocation_id": invocation_id, "sources": sources,
            "extras": extras, "frozen_before": before, "readme": readme()}
    else:
        dump(output / "source-preflight.json", {**source_status,
             "source_preflight_completed": False, "source_or_routed_trial_validated": False})
    progress("historical_guard_diagnostic")
    diagnostic_remaining = min(deadline, diagnostic_deadline)-time.perf_counter()
    if diagnostic_remaining > 0:
        family_job = validate_private_job(cache / "diagnostics" / key, cache)
        family_status = run_stage("family", family_job, cache,
            min(CAPS["family_diagnostic_seconds"], diagnostic_remaining))
        family = read_managed_json(family_job / "value.json") if family_status["status"] == "completed" else lp.family_status_records(check_rejections=False)
    else:
        family_status = {"status": "uncomputed_diagnostic_aggregate_cap", "timeout_seconds": CAPS["diagnostic_aggregate_seconds"],
                         "reason": "source and historical diagnostics share one predeclared180s budget"}
        family = lp.family_status_records(check_rejections=False)
    execution["stage_status"]["family"] = family_status
    execution["diagnostic_aggregate_wall_seconds"] = CAPS["diagnostic_aggregate_seconds"]-(diagnostic_deadline-time.perf_counter())
    if ACTIVE_PREFLIGHT is not None:
        ACTIVE_PREFLIGHT["family"] = family
    job = validate_private_job(cache / "prepared" / key, cache)
    job.mkdir(parents=True, exist_ok=True)
    saved_status = checkpoint_status(job, key)
    if source_record is None:
        status = {"status": "uncomputed_source_preflight_unavailable", "reason": "native preparation requires a completed correlated source preflight"}
    elif args.reuse_prepared and saved_status:
        status = {**saved_status, "reused_exact_private_checkpoint": True}
    else:
        print("Preparing actual full-unitary Clifford+T trial (cap1800s)", flush=True)
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
                raise ExecutionCapExceeded("managed QASM exceeds32MiB")
            h = digest(raw)
            if name in SOURCE_QASMS and h != source_record["qasm_sha256"][name]:
                raise AssertionError("prepared source/routed export differs from its completed preflight")
            if (output / name).is_symlink():
                raise ValueError("managed QASM must not overwrite a symbolic link")
            (output / name).write_bytes(raw)
            hashes[name] = h
        record["qasm_sha256"] = hashes
        record["qasm_export"] = {
            "circuits": {"source-trial.qasm": "raw literal source", "routed-rotation-trial.qasm": "actual complete routed rotations", "clifford-t-trial.qasm": "actual complete Clifford+T trial"},
            "in_memory_global_phase_radians": {name: float(circuit.global_phase) for name, circuit in
                (("source-trial.qasm", prepared.synthesized.source), ("routed-rotation-trial.qasm", prepared.synthesized.routed), ("clifford-t-trial.qasm", prepared.synthesized.circuit))},
            "scope": "every quantum event exported; measured QASM omits overall phase and is not a phase-preserving controlled oracle"}
        for index, profile in enumerate(lp.fixed_profiles()):
            name = f"p{profile.physical_error_rate:.0e}"
            print(f"Assessing {name} under frozen controller/model (cap180s)", flush=True)
            profile_job = validate_private_job(job / name, cache)
            profile_job.mkdir(exist_ok=True)
            import shutil
            shutil.copyfile(job / "prepared.pickle", profile_job / "prepared.pickle")
            dump(profile_job / "prepare-status.json", saved_status)
            progress(f"physical_profile_{index}")
            stage_status = run_stage("profile", profile_job, cache,
                min(CAPS["physical_seconds_per_profile"], max(1, deadline-time.perf_counter())), profile_index=index)
            execution["stage_status"][name] = stage_status
            report = read_managed_json(profile_job / "value.json") if stage_status["status"] == "completed" else {
                **stage_status, "profile": asdict(profile), "pareto_points": [], "all_feasible_points": [],
                "complete_algorithm_attempts": lp.COMPLETE_TRIALS,
                "physical_failure_budget": record["complete_workload"]["physical_failure_budget"],
                "factory_availability_budget": record["complete_workload"]["factory_availability_budget"]}
            reports.append(report)
            print(f"Assessed {name}: {report['status']}", flush=True)
    else:
        (output / "clifford-t-trial.qasm").unlink(missing_ok=True)
        for profile in lp.fixed_profiles():
            reports.append({"status": "uncomputed_preparation_unavailable", "profile": asdict(profile), "pareto_points": [],
                            "all_feasible_points": [], "complete_algorithm_attempts": lp.COMPLETE_TRIALS})
    for profile, report in zip(lp.fixed_profiles(), reports):
        point = min(report.get("pareto_points", []), key=lambda p: p["peak_physical_qubits"]) if report.get("pareto_points") else None
        diagnostic = report.get("controller_model_diagnostic", {})
        rows.append({"case_id": lp.CASE_ID, "physical_error": profile.physical_error_rate, "status": report["status"],
             "complete_trials": lp.COMPLETE_TRIALS, "distance": point["data_code_distance"] if point else None,
             "peak_physical_qubits": point["peak_physical_qubits"] if point else None,
             "all_trial_runtime_hours": point["worst_case_runtime_all_attempts_ns"]/3.6e12 if point else None,
             "full_physical_failure_model": point["full_physical_failure_model"] if point else None,
             "factory_readiness_failure_model": point["factory_supply"]["availability_failure_union_bound"] if point else None,
             "controller_failure_model": diagnostic.get("distance_independent_controller_model_contribution"),
             "controller_decisions_all_trials": diagnostic.get("controller_decisions_all_trials"),
             "pareto_points": len(report.get("pareto_points", []))})
    if time.perf_counter() > deadline:
        raise ExecutionCapExceeded("parent generation cap2400s exceeded")
    dump(output / "results.json", {"schema_version": 1, "source_preflight": source_record, "prepared": record,
        "physical_profiles": reports, "family": family, "family_allocations": lp.family_allocations(), "execution": execution})
    progress("artifact_export_and_verification")
    write_csv(output / "summary.csv", rows)
    write_csv(output / "family-status.csv", [{"n_system": r["n_system"], "steps": r["selected_steps"], "status": r["status"],
        "physical_status": r.get("physical_status", "computed_conditional_profiles" if record is not None and all(p["status"].startswith("computed_") for p in reports) else "infeasible_or_uncomputed_profiles"),
        "complete_trials": r["complete_trials"], "synthesis_performed": r["n_system"] == 4 and record is not None,
        "status_scope": r.get("status_scope", "preserved prior bundle"),
        "new_slice_source_preflight_completed": r["n_system"] == 4 and source_record is not None,
        "new_slice_clifford_t_completed": r["n_system"] == 4 and record is not None,
        "new_slice_physical_status": ";".join(p["status"] for p in reports) if r["n_system"] == 4 else "prior_result_preserved",
        "rejection_reproduced": r.get("rejection_reproduced", False), "density_checker_status": r.get("density_checker_status"),
        "density_width_rejection_reproduced": r.get("density_width_rejection_reproduced", False),
        "density_required_wires": r.get("density_required_wires"), "density_supported_wires": r.get("density_supported_wires"),
        "allocation_reassigned": False} for r in family.values()])
    plots(output, record, reports, source_preflight=source_record)
    (output / "README.md").write_text(readme())
    current_sources, current_extras = capture_sources(paths)
    if current_sources != sources or current_extras != extras:
        raise AssertionError("runtime source/dependency changed; no coherent manifest may be issued")
    export_captured_sources(output, sources, extras)
    after = fingerprints()
    if before != after:
        raise AssertionError("frozen shared/backend baseline changed during generation")
    if time.perf_counter() > deadline:
        raise ExecutionCapExceeded("parent generation cap2400s exceeded before manifest attestation")
    inventory = artifact_inventory(output)
    progress("completed_bounded_generation")
    dump(output / "manifest.json", {"schema_version": 1, **revision(paths), "source_sha256": source_hashes,
        "runtime_source_validation": runtime_sources,
        "artifacts_sha256": managed_hashes(output), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")},
        "device": "local CPU; bounded correlated density/native conditional models; no hardware/GPU run",
        "transpiler_seed": 7, "synthesis_seed": None, "synthesis_seed_scope": "native backend has no seed API; saved sequences/QASM hashes identify realization",
        "run_caps": CAPS, "execution": execution, "source_preflight_completed": source_record is not None,
        "full_clifford_t_trial_completed": record is not None, "wall_seconds": time.perf_counter()-start,
        "artifact_inventory": {"payload_files": inventory["payload_files"], "payload_bytes": inventory["payload_bytes"],
            "excluded_metadata": inventory["excluded_metadata"], "baseline_local_inventory": BASELINE_INVENTORY},
        "frozen_before": before, "frozen_after": after, "private_cache_distributed": False,
        "whole_family_physical_completed": False})
    print(f"Recorded bounded n4 attempt: source={source_status['status']}; preparation={status['status']}; prior n2/n3 retained", flush=True)

def main():
    global ACTIVE_PREFLIGHT
    ACTIVE_PREFLIGHT = None
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        _main()
    except (ExecutionCapExceeded, ValueError, AssertionError, OSError) as error:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad-four" if REPOSITORY else HERE / "generated-lindblad-four")
        parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()) / "openquantum-literal-lindblad-four-cache")
        parser.add_argument("--_stage", choices=("preflight", "family", "prepare", "profile"))
        args, _ = parser.parse_known_args()
        if args._stage is not None:
            raise
        signal.setitimer(signal.ITIMER_REAL, 0)
        output, cache = validate_paths(args.output_dir, args.cache_dir, source_files())
        output.mkdir(parents=True, exist_ok=True)
        # Remove all successful or partial prepared/physical artifacts. Only
        # this invocation's current hash-verified source may be reconstructed.
        for name in PUBLIC:
            (output / name).unlink(missing_ok=True)
        recovered = recover_preflight(output, cache, source_files())
        if recovered is None:
            # Recovery may have begun writing before another bounded guard
            # rejected it. No unattested remainder is retained.
            for name in PUBLIC:
                (output / name).unlink(missing_ok=True)
        is_cap = isinstance(error, ExecutionCapExceeded)
        reason = str(error) if is_cap else "bounded parent stage failed; no completed Clifford+T or physical attestation retained"
        failure = {"schema_version": 1,
            "status": "uncomputed_parent_execution_cap" if is_cap else "uncomputed_parent_stage_error",
            "reason": reason, "error_type": type(error).__name__,
            "source_preflight": recovered["record"] if recovered else None,
            "source_preflight_status": recovered["status"] if recovered else {"status": "unavailable_current_verified_source"},
            "source_preflight_completed": bool(recovered), "full_clifford_t_trial_completed": False,
            "prepared": None, "physical_profiles": [], "family": recovered["family"] if recovered else {},
            "run_caps": CAPS, "whole_family_physical_completed": False, "repeated_trials_executed": False,
            "scope": "failed bounded attempt; any retained evidence is verified source/routing only, not a completed quantum workload"}
        dump(output / "results.json", failure)
        inventory = artifact_inventory(output)
        dump(output / "manifest.json", {"schema_version": 1, "status": failure["status"],
            "artifacts_sha256": managed_hashes(output), "run_caps": CAPS,
            "source_sha256": recovered["source_sha256"] if recovered else {},
            "source_preflight_completed": bool(recovered), "full_clifford_t_trial_completed": False,
            "physical_results_completed": False, "private_cache_distributed": False,
            "artifact_inventory": {"payload_files": inventory["payload_files"], "payload_bytes": inventory["payload_bytes"],
                "excluded_metadata": inventory["excluded_metadata"], "baseline_local_inventory": BASELINE_INVENTORY},
            "scope": failure["scope"]})
        print(f"Recorded failed bounded attempt: {reason}; verified source retained={bool(recovered)}", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
