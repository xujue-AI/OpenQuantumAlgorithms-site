#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded, separately gated reconstruction and execution of one port rewrite.

The source stage only reconstructs saved native words and freezes a recipe.
It neither resynthesizes nor runs an old or rewritten density calculation.
Execution uses the same immutable source artifacts and requires a recorded
review proof naming the commit that contains them.  Private caches and logs
are disjoint from the deliberate export bundle.  Original project code.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from contextlib import contextmanager
import contextlib
import csv
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import pickle
import re
import signal
import stat
import subprocess
import tempfile
import threading
import time
from types import CodeType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = HERE.name == "scripts" and (SOURCE / "openquantum/simulation/lindblad_port_words.py").is_file()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))
if sys.pycache_prefix is None:
    sys.pycache_prefix = str(Path(tempfile.gettempdir()) / "openquantum-lindblad-port-bytecode" / os.urandom(16).hex())

CAPS = {
    "word_recipe_seconds": 180, "fixture_seconds": 180,
    "candidate_seconds": 1800, "native_profile_seconds": 180,
    "parent_seconds": 2400, "rss_bytes": 2 * 1024**3,
    "managed_bytes": 32 * 1024**2, "log_bytes": 32 * 1024**2,
    "checkpoint_bytes": 256 * 1024**2, "recipe_bytes": 8 * 1024**2,
    "factory_queries_per_profile": 24, "surface_queries_per_profile": 24,
    "factory_candidates_per_query": 1000, "factory_candidates_per_profile": 24000,
}
PINNED_PACKAGES = {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2"}
CORE_MODULES = ("instrument_resources", "simulation.lindblad_workload", "simulation.two_spin_lindblad",
    "simulation.open_systems", "simulation.bounded_density_instrument", "simulation.lindblad_four_density",
    "simulation.lindblad_four_compile", "simulation.lindblad_four_physical", "simulation.lindblad_port_words",
    "simulation.lindblad_port_rewrite")
SOURCE_OUTPUTS = ("word-catalog.json", "recipe.json", "source-proof.json")
EXECUTION_OUTPUTS = ("candidate-trial.qasm", "results.json", "run_simulation_lindblad_port.py", "lindblad_port_resources.py")
METADATA_OUTPUTS = ("manifest.json", "progress.json", "artifact-inventory.json", "artifact-inventory.csv", "README.md")
WORDS = tuple(f"words/word-{i:02d}.qasm" for i in range(8))
SOURCE_LEAVES = tuple(name.rsplit(".", 1)[-1] + ".py" for name in CORE_MODULES)
EXTRAS = ("LICENSE.txt", "requirements-lock.txt", "references.bib")
MANAGED = (*SOURCE_OUTPUTS, *EXECUTION_OUTPUTS, *METADATA_OUTPUTS, *WORDS, *SOURCE_LEAVES, *EXTRAS)
NATIVE_MONITOR_PROTOCOL = "port-fork-rss-wall-log-v1"


class ExecutionCapExceeded(RuntimeError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        if getattr(value.dtype, "kind", None) == "c":
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _lexical(path):
    return Path(os.path.abspath(path))


def reject_symlink_chain(path):
    """Check lexical parents before resolving, including an alias into itself."""
    path = _lexical(path)
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError("managed paths and parents must not be symbolic links")
    return path


def bounded_file_hash(path, maximum=None):
    maximum = CAPS["managed_bytes"] if maximum is None else maximum
    path = reject_symlink_chain(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("input must be a bounded regular file")
    result, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("input descriptor is not regular")
        while block := handle.read(65536):
            total += len(block)
            if total > maximum:
                raise ExecutionCapExceeded("file exceeds its declared byte cap")
            result.update(block)
        after = os.fstat(handle.fileno())
    current = reject_symlink_chain(path).stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or total != after.st_size or (current.st_ino, current.st_size, current.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during hashing")
    return result.hexdigest()


def bounded_read(path, maximum=None):
    maximum = CAPS["managed_bytes"] if maximum is None else maximum
    path = reject_symlink_chain(path)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("input must be a bounded regular file")
        raw = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
    current = reject_symlink_chain(path).stat()
    if len(raw) > maximum:
        raise ExecutionCapExceeded("input exceeds its declared byte cap")
    if len(raw) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or (current.st_ino, current.st_size, current.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during bounded reading")
    return raw


def read_json(path, maximum=None):
    maximum = CAPS["managed_bytes"] if maximum is None else maximum
    raw = bounded_read(path, maximum)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("managed JSON must contain an object")
    return value


def atomic_bytes(path, raw, maximum=None):
    maximum = CAPS["managed_bytes"] if maximum is None else maximum
    if not isinstance(raw, bytes) or len(raw) > maximum:
        raise ExecutionCapExceeded("artifact exceeds its declared byte cap")
    path = reject_symlink_chain(path)
    temporary = reject_symlink_chain(path.with_name(path.name + ".tmp"))
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return digest(raw)


def dump(path, value, maximum=None):
    return atomic_bytes(path, (json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n").encode(), maximum)


def module(name):
    return importlib.import_module("openquantum." + name if REPOSITORY else name.rsplit(".", 1)[-1])


def source_files():
    return {name.rsplit(".", 1)[-1] + ".py": SOURCE / "openquantum" / (name.replace(".", "/") + ".py") if REPOSITORY
            else HERE / (name.rsplit(".", 1)[-1] + ".py") for name in CORE_MODULES}


def verify_runtime_sources():
    paths = source_files()
    hashes = {}
    for name in CORE_MODULES:
        leaf = name.rsplit(".", 1)[-1] + ".py"
        imported = module(name)
        if Path(getattr(imported, "__file__", "")).resolve() != paths[leaf].resolve():
            raise ValueError("runtime import differs from intended source snapshot: " + leaf)
        raw = bounded_read(paths[leaf])
        compiled = compile(raw, str(paths[leaf].resolve()), "exec", dont_inherit=True, optimize=0)
        functions = {item.co_name: item for item in compiled.co_consts if isinstance(item, CodeType)}
        for name, function in vars(imported).items():
            if inspect.isfunction(function) and function.__module__ == imported.__name__:
                if function.__code__.co_name not in functions or function.__code__ != functions[function.__code__.co_name]:
                    raise ValueError("loaded function code differs from current source: " + leaf + ":" + name)
        hashes[leaf] = bounded_file_hash(paths[leaf])
    return {"source_sha256": hashes, "actual_module_paths_verified": True,
            "import_mode": "package" if REPOSITORY else "standalone", "bytecode_writes_disabled": sys.dont_write_bytecode,
            "bytecode_prefix_configured": sys.pycache_prefix is not None, "loaded_function_code_matches_sources": True}


def verify_runner_code():
    raw = bounded_read(Path(__file__))
    compiled = compile(raw, str(Path(__file__).resolve()), "exec", dont_inherit=True, optimize=0)
    definitions = {item.co_name: item for item in compiled.co_consts if isinstance(item, CodeType) and item.co_name.isidentifier()}
    for name, code in definitions.items():
        value = globals().get(name)
        if inspect.isfunction(value):
            if name in {"parent_watchdog", "stage_deadline"}:
                template = contextlib.contextmanager(lambda: None)
                wrapped = getattr(value, "__wrapped__", None)
                closure = value.__closure__
                if (value.__code__ != template.__code__ or not inspect.isfunction(wrapped) or
                        wrapped.__code__ != code or closure is None or len(closure) != 1 or
                        closure[0].cell_contents is not wrapped or
                        value.__globals__.get("_GeneratorContextManager") is not contextlib._GeneratorContextManager):
                    raise ValueError("loaded runner contextmanager wrapper differs from reviewed source: " + name)
            elif value.__code__ != code:
                raise ValueError("loaded runner helper differs from reviewed source: " + name)
        elif inspect.isclass(value):
            methods = {item.co_name: item for item in code.co_consts if isinstance(item, CodeType)}
            for method, expected in methods.items():
                function = value.__dict__.get(method)
                if isinstance(function, (staticmethod, classmethod)):
                    function = function.__func__
                if not inspect.isfunction(function) or function.__code__ != expected:
                    raise ValueError("loaded runner class helper differs from reviewed source: " + name + ":" + method)
        else:
            raise ValueError("reviewed runner definition is missing or replaced: " + name)
    return digest(raw)


def runtime_record():
    python = platform.python_version()
    packages = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
    if python != "3.12.14" or packages != PINNED_PACKAGES:
        raise ValueError("this recipe requires the unchanged locked Python/package environment")
    if not sys.dont_write_bytecode or sys.pycache_prefix is None:
        raise ValueError("private bytecode policy is required")
    return {"python": python, "packages": packages, "bytecode_writes_disabled": True,
            "bytecode_prefix_configured": True, "device": "local CPU", "hardware_execution": False}


def native_runtime_record():
    version = importlib.metadata.version("psutil")
    if version != "7.2.2":
        raise ValueError("native RSS adapter requires the pinned psutil7.2.2")
    return {"psutil": version, "native_watchdog": NATIVE_MONITOR_PROTOCOL}


def source_identity(hashes, runtime):
    return digest(canonical_json({"schema": "lindblad-port-source-v1", "source_sha256": hashes,
                                 "runtime": runtime, "caps": CAPS}))


def git(*args):
    env = {key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(ROOT), *args], env=env,
        stderr=subprocess.DEVNULL, timeout=5)


def revision():
    if not REPOSITORY:
        return None
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT.resolve():
        raise ValueError("Git selected a different checkout")
    return git("rev-parse", "HEAD").decode().strip()


def validate_paths(output, cache, baseline):
    output, cache, baseline = (reject_symlink_chain(path).resolve() for path in (output, cache, baseline))
    owned = ROOT / "assets/generated/simulation/lindblad-port"
    if REPOSITORY and output.is_relative_to(ROOT) and not output.is_relative_to(owned):
        raise ValueError("repository output must stay inside the new lindblad-port bundle")
    if output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("new output must be disjoint from the preserved baseline")
    if cache.is_relative_to(output) or output.is_relative_to(cache) or cache.is_relative_to(baseline) or baseline.is_relative_to(cache):
        raise ValueError("private cache and artifact inputs/outputs must be disjoint")
    if REPOSITORY and cache.is_relative_to(ROOT):
        raise ValueError("private cache must be outside the checkout")
    for name in MANAGED:
        reject_symlink_chain(output / name)
        temporary = reject_symlink_chain((output / name).with_name(Path(name).name + ".tmp"))
        if temporary.exists():
            raise ValueError("preexisting managed temporary target must be preserved")
    if any(path.resolve().is_relative_to(output) for path in source_files().values()):
        raise ValueError("output must not overwrite source inputs")
    return output, cache, baseline


def capture_source_snapshots():
    return {leaf: bounded_read(path) for leaf, path in source_files().items()}


def extras():
    if not REPOSITORY:
        return {name: bounded_read(HERE / name) for name in EXTRAS}
    references = bounded_read(ROOT / "references/simulation.bib") + b"\n" + bounded_read(ROOT / "references/resources.bib")
    if len(references) > CAPS["managed_bytes"]:
        raise ExecutionCapExceeded("combined references exceed32MiB")
    return {"LICENSE.txt": bounded_read(ROOT / "LICENSE"), "requirements-lock.txt": bounded_read(ROOT / "requirements-lock.txt"),
            "references.bib": references}


def source_stage(output, baseline):
    """Only Gate1/2 source artifacts; no circuit materialization or density call."""
    began = time.monotonic()
    runtime = runtime_record()
    validation = verify_runtime_sources()
    identity = source_identity(validation["source_sha256"], runtime)
    words, rewrite = module("simulation.lindblad_port_words"), module("simulation.lindblad_port_rewrite")
    word_limits, rewrite_limits = words.literal_limits(), rewrite.literal_limits()
    recovered = words.recover_words(baseline, limits=word_limits)
    recipe = rewrite.build_recipe(recovered, limits=rewrite_limits, word_limits=word_limits)
    artifacts = words.recovery_artifacts(recovered, limits=word_limits)
    artifacts["recipe.json"] = recipe.canonical_json
    captured = capture_source_snapshots()
    captured_extras = extras()
    if {leaf: digest(raw) for leaf, raw in captured.items()} != validation["source_sha256"]:
        raise ValueError("source changed during source-only reconstruction")
    if source_identity(verify_runtime_sources()["source_sha256"], runtime_record()) != identity:
        raise ValueError("source/runtime changed during reconstruction")
    if time.monotonic() - began > CAPS["word_recipe_seconds"]:
        raise ExecutionCapExceeded("source reconstruction exceeded180s")
    if {name: digest(raw) for name, raw in extras().items()} != {name: digest(raw) for name, raw in captured_extras.items()}:
        raise ValueError("lock/license/references changed during reconstruction")
    hashes = {name: digest(raw) for name, raw in {**artifacts, **captured, **captured_extras}.items()}
    proof = {"schema_version": 1, "status": "completed_source_only_recipe", "source_identity": identity,
        "source_sha256": validation["source_sha256"], "artifact_sha256": hashes, "runtime": runtime,
        "runtime_source_validation": validation, "source_commit_at_generation": revision(),
        "recipe_sha256": recipe.recipe_sha256, "catalog_recovery_sha256": recovered.recovery_sha256,
        "baseline_artifacts_sha256": recovered.source_artifacts, "limits": {"words": asdict(word_limits), "rewrite": asdict(rewrite_limits)},
        "planned_counts": recipe.as_record().get("plan", recipe.as_record().get("summary", {})), "source_only_wall_seconds": time.monotonic() - began,
        "candidate_materialized": False, "candidate_full_density_executed": False, "new_synthesis_calls": 0,
        "native_physical_queries": 0, "scope": "actual saved phased words and exact structural recipe; no full candidate or feasibility claim"}
    # Every destination is preflighted before the first mutation.
    for name in {**artifacts, **captured, **captured_extras}:
        reject_symlink_chain(output / name)
        reject_symlink_chain((output / name).with_name(Path(name).name + ".tmp"))
    for name, raw in {**artifacts, **captured, **captured_extras}.items():
        atomic_bytes(output / name, raw, CAPS["recipe_bytes"] if name == "recipe.json" else None)
    dump(output / "source-proof.json", proof)
    return proof


def validate_source_proof(output, baseline):
    proof = read_json(output / "source-proof.json")
    if proof.get("status") != "completed_source_only_recipe" or proof.get("candidate_materialized") is not False or proof.get("candidate_full_density_executed") is not False:
        raise ValueError("a completed source-only recipe proof is required")
    runtime, validation = runtime_record(), verify_runtime_sources()
    if proof.get("source_identity") != source_identity(validation["source_sha256"], runtime) or proof.get("runtime") != runtime:
        raise ValueError("source proof has stale source/runtime/caps")
    allowed = set((*WORDS, "word-catalog.json", "recipe.json", *SOURCE_LEAVES, *EXTRAS))
    hashes = proof.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != allowed:
        raise ValueError("source proof lacks the exact deliberate artifact allowlist")
    if any(bounded_file_hash(output / name) != expected for name, expected in hashes.items()):
        raise ValueError("source artifact changed after completed reconstruction")
    if any(bounded_file_hash(baseline / name) != expected for name, expected in proof["baseline_artifacts_sha256"].items()):
        raise ValueError("preserved baseline changed after source reconstruction")
    if {name: digest(raw) for name, raw in extras().items()} != {name: hashes[name] for name in EXTRAS}:
        raise ValueError("source lock/license/references changed after reconstruction")
    return proof


def _review_note(approval):
    if not isinstance(approval, dict) or approval.get("status") != "approved":
        raise ValueError("independent review approval is required")
    name = approval.get("review_note")
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or not name.startswith("docs/coordination/"):
        raise ValueError("review note must be a private coordination path")
    if bounded_file_hash(ROOT / name) != approval.get("review_note_sha256"):
        raise ValueError("review note changed after approval")
    return approval["review_note_sha256"]


def _committed_hash(commit, path):
    name = path.relative_to(ROOT).as_posix()
    spec = commit + ":" + name
    size = int(git("cat-file", "-s", spec).decode().strip())
    if size > CAPS["managed_bytes"]:
        raise ExecutionCapExceeded("frozen commit artifact exceeds32MiB")
    raw = git("show", spec)
    if len(raw) != size:
        raise ValueError("frozen commit artifact size mismatch")
    return digest(raw)


def validate_review_proof(proof_path, output, baseline, *, include_native=False):
    if proof_path is None or not REPOSITORY:
        raise ValueError("execution requires a private recorded repository review proof")
    runner_sha = verify_runner_code()
    source = validate_source_proof(output, baseline)
    approval = read_json(proof_path, 1024 * 1024)
    if approval.get("schema_version") != 1 or approval.get("status") != "approved_frozen_port_recipe":
        raise ValueError("recorded frozen-recipe approval is required")
    for name, expected in (("source_proof_sha256", bounded_file_hash(output / "source-proof.json")),
                           ("recipe_sha256", source["recipe_sha256"]), ("catalog_recovery_sha256", source["catalog_recovery_sha256"]),
                           ("runner_source_sha256", runner_sha)):
        if approval.get(name) != expected:
            raise ValueError("recorded approval does not bind current " + name)
    commit = approval.get("frozen_source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("approval must record an exact frozen source/artifact commit")
    git("merge-base", "--is-ancestor", commit, "HEAD")
    for leaf, path in source_files().items():
        if _committed_hash(commit, path) != source["source_sha256"][leaf]:
            raise ValueError("recorded source freeze does not contain the reviewed source")
    frozen_output = ROOT / "assets/generated/simulation/lindblad-port"
    for name in (*WORDS, "word-catalog.json", "recipe.json", "source-proof.json"):
        if _committed_hash(commit, frozen_output / name) != bounded_file_hash(output / name):
            raise ValueError("recorded source freeze does not contain the reviewed artifact")
    approvals = approval.get("approvals", {})
    _review_note(approvals.get("word_recipe"))
    resource_path = SOURCE / "openquantum/simulation/lindblad_port_resources.py"
    resource_sha = bounded_file_hash(resource_path)
    if approval.get("resource_source_sha256") != resource_sha:
        raise ValueError("candidate verifier source is not bound by the recorded gate proof")
    if include_native:
        _review_note(approvals.get("adapter"))
        if approvals["adapter"].get("source_sha256") != resource_sha:
            raise ValueError("native adapter review does not bind current source")
        _review_note(approvals.get("runner"))
        if approvals["runner"].get("source_sha256") != approval["runner_source_sha256"]:
            raise ValueError("native runner review does not bind current source")
    return source, approval


def rss_peak_bytes():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def checked_boundary(began, seconds):
    if time.monotonic() - began > seconds:
        raise ExecutionCapExceeded("stage exceeded its predeclared wall cap")
    peak = rss_peak_bytes()
    if peak > CAPS["rss_bytes"]:
        raise ExecutionCapExceeded("process peak RSS exceeded2GiB")
    enforce_rss(process_table(), os.getpid())
    return peak


@contextmanager
def stage_deadline(seconds):
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise ExecutionCapExceeded("stage exceeded its predeclared wall cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def parent_watchdog(*, on_failure=None):
    """Keep the hard parent wall/RSS watch active through artifact finalization."""
    began, done, reason = time.monotonic(), threading.Event(), []
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise ExecutionCapExceeded(reason[0] if reason else "parent watchdog expired")
    signal.signal(signal.SIGALRM, expired)
    def monitor():
        while not done.wait(0.25):
            try:
                if time.monotonic() - began > CAPS["parent_seconds"]:
                    raise ExecutionCapExceeded("hard parent timeout2400s through artifact tail")
                enforce_rss(process_table(), os.getpid())
            except (ExecutionCapExceeded, RuntimeError, OSError, subprocess.SubprocessError) as error:
                reason.append(str(error))
                if not done.is_set():
                    os.kill(os.getpid(), signal.SIGALRM)
                return
    thread = threading.Thread(target=monitor, name="port-parent-watchdog", daemon=True)
    thread.start()
    try:
        yield began
        checked_boundary(began, CAPS["parent_seconds"])
    except BaseException as error:
        done.set()
        thread.join(timeout=4)
        if on_failure is not None:
            on_failure(error)
        raise
    finally:
        done.set()
        thread.join(timeout=4)
        signal.signal(signal.SIGALRM, previous)


def clear_failed_parent_artifacts(output, mode, started, error):
    if not started[0]:
        return
    (output / "manifest.json").unlink(missing_ok=True)
    # Preflight rejected every preexisting managed temporary before mutation.
    for name in MANAGED:
        temporary = reject_symlink_chain((output / name).with_name(Path(name).name + ".tmp"))
        temporary.unlink(missing_ok=True)
    status = {"status": "uncomputed_parent_execution_cap" if isinstance(error, ExecutionCapExceeded) else "uncomputed_parent_error",
              "stage": mode, "error_type": type(error).__name__, "reason": "parent or artifact finalization failed; no complete manifest retained"}
    if mode != "source" and (output / "results.json").is_file():
        result = read_json(output / "results.json")
        result["status"] = "completed_candidate_parent_finalization_failed"
        result.setdefault("execution", {})["parent_finalization_failure"] = status
        dump(output / "results.json", result)
    dump(output / "progress.json", status)


def dump_candidate_qasm(path, circuit):
    """Stream every primitive and the baseline whole phase exactly once."""
    path = reject_symlink_chain(path)
    temporary = reject_symlink_chain(path.with_name(path.name + ".tmp"))
    if circuit.num_qubits != 8 or circuit.num_clbits != 8:
        raise ValueError("candidate QASM requires the complete8-wire instrument")
    phase = float(circuit.global_phase)
    if not math.isfinite(phase):
        raise ValueError("candidate phase must be finite")
    total, result, created = 0, hashlib.sha256(), False
    try:
        with temporary.open("xb") as handle:
            created = True
            def write(raw):
                nonlocal total
                total += len(raw)
                if total > CAPS["managed_bytes"]:
                    raise ExecutionCapExceeded("candidate QASM exceeded32MiB")
                handle.write(raw)
                result.update(raw)
            write(("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nbit[8] c;\nqubit[8] q;\n" + f"gphase({phase!r});\n").encode())
            for instruction in circuit.data:
                op = instruction.operation
                wires = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
                bits = [circuit.find_bit(bit).index for bit in instruction.clbits]
                if op.params or getattr(op, "condition", None) is not None or op.name not in {"h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "reset", "measure", "barrier"}:
                    raise ValueError("candidate QASM permits only verified flat primitives/events")
                if op.name == "measure":
                    if len(wires) != 1 or len(bits) != 1:
                        raise ValueError("malformed candidate measurement")
                    line = f"c[{bits[0]}] = measure q[{wires[0]}];\n"
                else:
                    if bits:
                        raise ValueError("unexpected classical operand")
                    line = op.name + " " + ", ".join(f"q[{wire}]" for wire in wires) + ";\n"
                write(line.encode())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return result.hexdigest()


def save_candidate_archive(path, candidate, verification_record):
    """Bounded archival checkpoint; it cannot reissue a verified capability."""
    path, temporary = reject_symlink_chain(path), reject_symlink_chain(path.with_name(path.name + ".tmp"))
    created = False
    class BoundedWriter:
        def __init__(self, handle):
            self.handle, self.total = handle, 0
        def write(self, raw):
            self.total += len(raw)
            if self.total > CAPS["checkpoint_bytes"]:
                raise ExecutionCapExceeded("private candidate checkpoint exceeds256MiB")
            return self.handle.write(raw)
    try:
        with temporary.open("xb") as handle:
            created = True
            pickle.dump({"schema": "port-candidate-archive-v1", "circuit": candidate.circuit,
                "report": candidate.report, "verification_record": verification_record,
                "verified_capability_reissuable": False}, BoundedWriter(handle), protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return bounded_file_hash(path, CAPS["checkpoint_bytes"])


class _ForkProcess:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid
    def poll(self):
        return None if self.process.is_alive() else self.process.exitcode
    def wait(self, timeout):
        self.process.join(timeout)
        if self.process.is_alive():
            raise RuntimeError("native fork child did not terminate after kill")
        return self.process.exitcode


def _native_worker(verified, profile, job, factory_cache, resource, resource_limits, native_caps, activation, capability):
    began = time.monotonic()
    with (job / "worker.log").open("ab") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            if not activation.wait(timeout=5):
                raise ExecutionCapExceeded("native parent watchdog failed to activate")
            checked_boundary(began, CAPS["native_profile_seconds"])
            result = resource.estimate_verified_four(verified, profile=profile, cache_dir=factory_cache, limits=resource_limits,
                native_limits=native_caps, include_all_feasible_points=True, allow_native_search=True, watchdog_capability=capability)
            peak = checked_boundary(began, CAPS["native_profile_seconds"])
            value_sha = dump(job / "value.json", result)
            checked_boundary(began, CAPS["native_profile_seconds"])
            status = {"status": "completed", "value_sha256": value_sha, "peak_rss_bytes": peak, "wall_seconds": time.monotonic() - began}
        except Exception as error:
            import traceback
            traceback.print_exc()
            (job / "value.json").unlink(missing_ok=True)
            cap_types = (ExecutionCapExceeded, getattr(resource, "NativeLimitExceeded", ExecutionCapExceeded))
            status = {"status": "uncomputed_execution_cap" if isinstance(error, cap_types) else "uncomputed_native_error",
                      "reason": "whole profile rejected; diagnostic retained only in private log", "error_type": type(error).__name__}
        dump(job / "status.json", status)


def run_native_profile(verified, profile, profile_index, cache, resource, resource_limits, *, runner_sha256, parent_began):
    """Fork the issued verification and activate the enclosing monitor first."""
    diagnostic = resource.controller_eligibility(verified, profile)
    if diagnostic["budget_exceeded"]:
        return {"status": "infeasible_controller_model", "controller_diagnostic": diagnostic,
            "profile": jsonable(asdict(profile)), "pareto_points": [], "all_feasible_points": [],
            "execution_query_counts": {"top_level_factory_queries": 0, "top_level_surface_code_queries": 0,
                                       "factory_candidates": 0, "native_trace_calls": 0}}
    if verify_runner_code() != runner_sha256:
        raise ValueError("native runner helpers changed after independent review")
    import multiprocessing
    context = multiprocessing.get_context("fork")
    activation = context.Event()
    native_caps = resource.literal_native_limits()
    capability = resource._issue_native_watchdog(activation_event=activation, limits=native_caps, runner_source_sha256=runner_sha256)
    job = validate_private_job(cache / f"profile-{profile_index}-{os.urandom(12).hex()}", cache)
    job.mkdir(parents=True)
    (job / "worker.log").open("xb").close()
    factory_cache = validate_private_job(cache / f"factories-{profile_index}", cache)
    child = context.Process(target=_native_worker, args=(verified, profile, job, factory_cache, resource, resource_limits,
                                                       native_caps, activation, capability))
    child.start()
    failure = watch_process(_ForkProcess(child), CAPS["native_profile_seconds"], job / "worker.log",
                            parent_began=parent_began, activation=activation)
    if failure is not None:
        (job / "value.json").unlink(missing_ok=True)
        return {**failure, "profile": jsonable(asdict(profile)), "pareto_points": [], "all_feasible_points": [],
                "reason": failure.get("reason", "whole profile rejected at hard180s limit")}
    if not (job / "status.json").is_file():
        return {"status": "uncomputed_native_error", "pareto_points": [], "all_feasible_points": [], "reason": "native fork child returned no complete status"}
    status = read_json(job / "status.json")
    if status["status"] != "completed":
        return {**status, "pareto_points": [], "all_feasible_points": []}
    if bounded_file_hash(job / "value.json") != status["value_sha256"]:
        raise ValueError("native profile result changed after bounded completion")
    result = read_json(job / "value.json")
    result["execution_status"] = status
    return result


def candidate_stage(output, baseline, cache, job, proof_path, *, include_native):
    overall_began = time.monotonic()
    source, approval = validate_review_proof(proof_path, output, baseline, include_native=include_native)
    words, rewrite, resource = (module("simulation." + name) for name in ("lindblad_port_words", "lindblad_port_rewrite", "lindblad_port_resources"))
    actual_resource_path = Path(getattr(resource, "__file__", "")).resolve()
    expected_resource_path = (SOURCE / "openquantum/simulation/lindblad_port_resources.py").resolve()
    if actual_resource_path != expected_resource_path:
        raise ValueError("candidate/native adapter import differs from current source")
    native_runtime = native_runtime_record()
    word_limits, rewrite_limits, resource_limits = words.literal_limits(), rewrite.literal_limits(), resource.literal_limits()
    began = time.monotonic()
    with stage_deadline(CAPS["candidate_seconds"]):
        recovered = words.recover_words(baseline, limits=word_limits)
        recipe = rewrite.build_recipe(recovered, limits=rewrite_limits, word_limits=word_limits)
        if recipe.recipe_sha256 != source["recipe_sha256"] or recipe.canonical_json != bounded_read(output / "recipe.json", CAPS["recipe_bytes"]):
            raise ValueError("fresh exact recipe differs from the frozen reviewed recipe")
        artifacts = words.recovery_artifacts(recovered, limits=word_limits)
        if any(digest(raw) != source["artifact_sha256"][name] for name, raw in artifacts.items()):
            raise ValueError("fresh recovered phased words differ from frozen reviewed exports")
        candidate = rewrite.materialize_candidate(recipe, recovered, limits=rewrite_limits, word_limits=word_limits, allow_full_materialization=True)
        verified = resource.verify_candidate(candidate, recipe, recovered, limits=resource_limits, word_limits=word_limits,
                                             rewrite_limits=rewrite_limits, allow_full_density=True)
        verification = verified.as_record()
        peak = checked_boundary(began, CAPS["candidate_seconds"])
        qasm_sha = dump_candidate_qasm(output / "candidate-trial.qasm", candidate.circuit)
        checkpoint_sha = save_candidate_archive(job / "candidate.pickle", candidate, verification)
        checked_boundary(began, CAPS["candidate_seconds"])
    lp = module("simulation.lindblad_four_physical")
    profiles = lp.fixed_profiles()
    eligibility = [resource.controller_eligibility(verified, profile) for profile in profiles]
    result = {"schema_version": 1, "status": "completed_candidate_native_unexecuted", "candidate": verification,
        "candidate_verification_fingerprint": verified.verification_fingerprint, "candidate_qasm_sha256": qasm_sha,
        "candidate_global_phase_binary64_hex": float(candidate.circuit.global_phase).hex(),
        "controller_eligibility": eligibility, "baseline_saved_prepared": words.baseline_record(recovered, limits=word_limits),
        "physical_profiles": [{"status": "unexecuted_native_profiles_not_requested", "pareto_points": [], "all_feasible_points": []} for _ in profiles],
        "execution": {"candidate_density_executed_once": True, "historical_densities_rerun": False, "new_synthesis_calls": 0,
            "candidate_wall_seconds": time.monotonic() - began, "candidate_peak_rss_bytes": peak,
            "private_checkpoint_sha256": checkpoint_sha, "checkpoint_restores_verified_capability": False,
            "frozen_source_commit": approval["frozen_source_commit"], "review_note_sha256": approval["approvals"]["word_recipe"]["review_note_sha256"],
            "runner_source_sha256": approval["runner_source_sha256"], "resource_source_sha256": approval["resource_source_sha256"],
            "rss_watchdog": NATIVE_MONITOR_PROTOCOL, "native_runtime": native_runtime, "run_caps": CAPS},
        "scope": "The repeated trial population and hardware schedule are unexecuted"}
    def publish_completed_candidate():
        validate_review_proof(proof_path, output, baseline, include_native=include_native)
        results_sha = dump(output / "results.json", result)
        dump(job / "candidate-status.json", {"status": "completed_verified_candidate", "results_sha256": results_sha,
            "candidate_qasm_sha256": qasm_sha, "source_proof_sha256": approval["source_proof_sha256"],
            "verification_fingerprint": verified.verification_fingerprint})
    publish_completed_candidate()
    if include_native:
        result["physical_profiles"] = []
        for index, profile in enumerate(profiles):
            # Adapter rechecks actual controller eligibility before dispatch.
            profile_result = run_native_profile(verified, profile, index, cache, resource, resource_limits,
                runner_sha256=approval["runner_source_sha256"], parent_began=overall_began)
            result["physical_profiles"].append(profile_result)
            result["status"] = "completed_candidate_partial_native_profiles"
            publish_completed_candidate()
        result["status"] = "completed_candidate_and_bounded_native_profiles"
        publish_completed_candidate()
    atomic_bytes(output / "run_simulation_lindblad_port.py", bounded_read(Path(__file__)))
    atomic_bytes(output / "lindblad_port_resources.py", bounded_read(actual_resource_path))
    checked_boundary(overall_began, CAPS["candidate_seconds"] + 2 * CAPS["native_profile_seconds"])
    return result


def managed_hashes(output):
    return {name: bounded_file_hash(output / name) for name in MANAGED
            if name not in {"manifest.json", "progress.json", "artifact-inventory.json", "artifact-inventory.csv"} and (output / name).is_file()}


def write_manifest(output, proof, execution=None):
    state = "Source-only recovered words and exact frozen recipe; candidate/native stages remain unexecuted." if execution is None else execution["status"]
    atomic_bytes(output / "README.md", ("# Matched-task physical-wire-0 port rewrite\n\n" + state + "\n\n"
        "The repeated trial population and hardware schedule are unexecuted.\n"
        "The words are recovered from the frozen completed n4 exports without new synthesis.\n"
        "Every word export includes its recorded binary64 global phase. The candidate carries the original complete phase once.\n"
        "Source proof and recipe counts precede execution; only candidate records assert actual materialization/density checks.\n"
        "The preserved original n4 controller-model failures remain unchanged. Native results, when present, are conditional models.\n"
        "The runner requires the locked Python3.12 environment, an explicitly supplied frozen baseline bundle, and private recorded reviews for candidate/native execution.\n"
        "Caches, checkpoints, review notes and raw logs are excluded from this bundle.\n").encode())
    hashes = managed_hashes(output)
    inventory = [{"file": name, "bytes": (output / name).stat().st_size, "sha256": value,
                  "category": "phased_word_export" if name in WORDS else "exact_recipe" if name == "recipe.json" else
                              "source_snapshot" if name.endswith(".py") else "result_or_provenance"} for name, value in sorted(hashes.items())]
    dump(output / "artifact-inventory.json", {"schema_version": 1, "files": inventory,
        "payload_files": len(inventory), "payload_bytes": sum(row["bytes"] for row in inventory),
        "scope": "managed new local assets; metadata self-hashes excluded; not public hosting occupancy"})
    text_csv = io.StringIO(newline="")
    writer = csv.DictWriter(text_csv, fieldnames=("file", "bytes", "sha256", "category"), lineterminator="\n")
    writer.writeheader()
    writer.writerows(inventory)
    atomic_bytes(output / "artifact-inventory.csv", text_csv.getvalue().encode())
    hashes["artifact-inventory.json"] = bounded_file_hash(output / "artifact-inventory.json")
    hashes["artifact-inventory.csv"] = bounded_file_hash(output / "artifact-inventory.csv")
    manifest = {"schema_version": 1, "study": "one matched-task physical-wire-0 port rewrite",
        "status": "completed_source_only_recipe" if execution is None else execution["status"],
        "artifacts_sha256": hashes, "source_sha256": proof["source_sha256"], "runtime": proof["runtime"],
        "python": proof["runtime"]["python"], "packages": proof["runtime"]["packages"],
        "source_commit_at_generation": proof["source_commit_at_generation"], "recipe_sha256": proof["recipe_sha256"],
        "catalog_recovery_sha256": proof["catalog_recovery_sha256"], "run_caps": CAPS,
        "source_only": execution is None, "hardware_execution": False,
        "scope": "The repeated trial population and hardware schedule are unexecuted"}
    dump(output / "manifest.json", manifest)
    return manifest


def process_table():
    """Read only PIDs/parents/RSS; never process commands or environment."""
    raw = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True, timeout=3)
    result = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(item.isdigit() for item in fields):
            pid, parent, kib = map(int, fields)
            result[pid] = (parent, kib * 1024)
    if os.getpid() not in result:
        raise RuntimeError("RSS watchdog could not inspect the parent process")
    return result


def descendant_pids(table, pid):
    found, pending = {pid}, [pid]
    while pending:
        parent = pending.pop()
        children = [child for child, value in table.items() if value[0] == parent and child not in found]
        found.update(children)
        pending.extend(children)
    return found


def enforce_rss(table, pid):
    for current in descendant_pids(table, pid):
        if current in table and table[current][1] > CAPS["rss_bytes"]:
            raise ExecutionCapExceeded("parent/child process RSS exceeded2GiB")


def kill_tree(pid):
    try:
        pids = descendant_pids(process_table(), pid)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        pids = {pid}
    for current in sorted(pids, reverse=True):
        try:
            os.kill(current, signal.SIGKILL)
        except ProcessLookupError:
            pass


def watch_process(process, timeout, log_path, *, parent_began, activation=None):
    try:
        return _watch_process_body(process, timeout, log_path, parent_began=parent_began, activation=activation)
    except BaseException:
        # Parent SIGALRM can arrive even between polling boundaries.
        kill_tree(process.pid)
        try:
            process.wait(timeout=10)
        except (RuntimeError, subprocess.SubprocessError):
            pass
        raise


def _watch_process_body(process, timeout, log_path, *, parent_began, activation=None):
    """Enclose blocked/pre-first-yield native work with wall/RSS/log guards."""
    began = time.monotonic()
    failure = None
    while process.poll() is None:
        try:
            table = process_table()
            enforce_rss(table, os.getpid())
            enforce_rss(table, process.pid)
            if time.monotonic() - parent_began > CAPS["parent_seconds"]:
                raise ExecutionCapExceeded("hard parent timeout2400s")
            if time.monotonic() - began > timeout:
                failure = {"status": "uncomputed_timeout", "timeout_seconds": timeout}
                break
            if Path(log_path).stat().st_size > CAPS["log_bytes"]:
                raise ExecutionCapExceeded("private worker log exceeded32MiB")
            if activation is not None and process.pid in table and not activation.is_set():
                activation.set()
        except (ExecutionCapExceeded, RuntimeError, OSError, subprocess.SubprocessError) as error:
            failure = {"status": "uncomputed_execution_cap", "reason": str(error)}
            break
        time.sleep(0.25)
    if failure is not None:
        kill_tree(process.pid)
    process.wait(timeout=10)
    # Boundary checks cover both ordinary completion and a short transient job.
    if failure is None:
        checked_boundary(parent_began, CAPS["parent_seconds"])
        if time.monotonic() - began > timeout:
            failure = {"status": "uncomputed_timeout", "timeout_seconds": timeout}
        if Path(log_path).stat().st_size > CAPS["log_bytes"]:
            failure = {"status": "uncomputed_execution_cap", "reason": "private log exceeded32MiB"}
    return failure


def validate_private_job(job, cache):
    job, cache = reject_symlink_chain(job), reject_symlink_chain(cache).resolve()
    if not job.is_relative_to(cache) or not job.resolve().is_relative_to(cache):
        raise ValueError("private job must stay inside its cache")
    for name in ("status.json", "candidate-status.json", "worker.log", "value.json", "candidate.pickle", "candidate.pickle.tmp"):
        reject_symlink_chain(job / name)
    return job


def run_stage(stage, output, baseline, cache, *, parent_began, proof_path=None):
    job = validate_private_job(cache / (stage + "-" + os.urandom(12).hex()), cache)
    job.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--_stage", stage, "--_job-dir", str(job),
        "--output-dir", str(output), "--baseline-dir", str(baseline), "--cache-dir", str(cache)]
    if proof_path is not None:
        command.extend(("--review-proof", str(proof_path)))
    env = dict(os.environ)
    env.update(PYTHONPATH=str(SOURCE if REPOSITORY else HERE), PYTHONDONTWRITEBYTECODE="1",
        PYTHONPYCACHEPREFIX=str(cache / "bytecode" / os.urandom(16).hex()), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    timeout = CAPS["word_recipe_seconds"] if stage == "source" else CAPS["candidate_seconds"] + 2 * CAPS["native_profile_seconds"]
    with (job / "worker.log").open("xb") as log:
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT if REPOSITORY else HERE, env=env, stdout=log, stderr=log, start_new_session=True)
            failure = watch_process(process, timeout, job / "worker.log", parent_began=parent_began)
        except BaseException:
            if process is not None and process.poll() is None:
                kill_tree(process.pid)
                process.wait(timeout=10)
            raise
    if failure is not None:
        complete = job / "candidate-status.json"
        if stage != "source" and complete.is_file():
            status = read_json(complete)
            if (status.get("status") == "completed_verified_candidate" and
                    bounded_file_hash(output / "results.json") == status.get("results_sha256") and
                    bounded_file_hash(output / "candidate-trial.qasm") == status.get("candidate_qasm_sha256") and
                    bounded_file_hash(output / "source-proof.json") == status.get("source_proof_sha256")):
                failure["candidate_result_preserved"] = True
        return failure
    if not (job / "status.json").is_file():
        return {"status": "uncomputed_stage_error", "exit_code": process.returncode, "reason": "worker returned no complete status"}
    return read_json(job / "status.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "candidate", "execute"), default="source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad-port")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad-four")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--review-proof", type=Path)
    parser.add_argument("--_stage", choices=("source", "candidate", "execute"), help=argparse.SUPPRESS)
    parser.add_argument("--_job-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output, cache, baseline = validate_paths(args.output_dir, args.cache_dir, args.baseline_dir)
    if args._stage:
        job = validate_private_job(args._job_dir, cache)
        began = time.monotonic()
        try:
            if args._stage == "source":
                with stage_deadline(CAPS["word_recipe_seconds"]):
                    proof = source_stage(output, baseline)
                    validate_source_proof(output, baseline)
                    peak = checked_boundary(began, CAPS["word_recipe_seconds"])
                status = {"status": "completed", "source_proof_sha256": bounded_file_hash(output / "source-proof.json"),
                          "source_identity": proof["source_identity"], "peak_rss_bytes": peak}
            else:
                result = candidate_stage(output, baseline, cache, job, args.review_proof, include_native=args._stage == "execute")
                status = {"status": "completed", "results_sha256": bounded_file_hash(output / "results.json"),
                          "candidate_status": result["status"], "peak_rss_bytes": rss_peak_bytes()}
        except Exception as error:
            import traceback
            traceback.print_exc()
            status = {"status": "uncomputed_execution_cap" if isinstance(error, ExecutionCapExceeded) else "uncomputed_stage_error",
                      "error_type": type(error).__name__, "reason": "bounded stage failed; diagnostic retained in private local log"}
            if args._stage != "source" and not (job / "candidate-status.json").is_file():
                for name in EXECUTION_OUTPUTS:
                    (output / name).unlink(missing_ok=True)
        dump(job / "status.json", status)
        return 0 if status["status"] == "completed" else 1
    started = [False]
    with parent_watchdog(on_failure=lambda error: clear_failed_parent_artifacts(output, args.mode, started, error)) as began:
        if args.mode != "source":
            # Approvals are checked before any success artifact is invalidated.
            validate_review_proof(args.review_proof, output, baseline, include_native=args.mode == "execute")
        started[0] = True
        output.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        for name in EXECUTION_OUTPUTS:
            (output / name).unlink(missing_ok=True)
        if args.mode == "source":
            (output / "source-proof.json").unlink(missing_ok=True)
        (output / "manifest.json").unlink(missing_ok=True)
        dump(output / "progress.json", {"status": "running", "stage": args.mode})
        status = run_stage(args.mode, output, baseline, cache, parent_began=began, proof_path=args.review_proof)
        execution = None
        if status["status"] == "completed":
            proof = validate_source_proof(output, baseline)
            if args.mode == "source":
                if bounded_file_hash(output / "source-proof.json") != status["source_proof_sha256"]:
                    raise ValueError("source proof changed after bounded worker completion")
            else:
                validate_review_proof(args.review_proof, output, baseline, include_native=args.mode == "execute")
                if bounded_file_hash(output / "results.json") != status["results_sha256"]:
                    raise ValueError("candidate results changed after bounded worker completion")
                execution = read_json(output / "results.json")
        elif args.mode != "source" and status.get("candidate_result_preserved"):
            proof = validate_source_proof(output, baseline)
            execution = read_json(output / "results.json")
            execution["status"] = "completed_candidate_native_interrupted"
            execution["execution"]["parent_stage_status"] = status
            while len(execution["physical_profiles"]) < 2:
                execution["physical_profiles"].append({"status": "uncomputed_parent_interruption", "pareto_points": [], "all_feasible_points": []})
            dump(output / "results.json", execution)
        else:
            if args.mode != "source":
                for name in EXECUTION_OUTPUTS:
                    (output / name).unlink(missing_ok=True)
            proof = None
        if proof is not None:
            write_manifest(output, proof, execution)
        dump(output / "progress.json", status)
    print(json.dumps(status, sort_keys=True))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
