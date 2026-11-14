#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate saved dynamics models and replay their exact stored circuits.

No Native estimator, factory search or phase fitter is invoked by this gate.
Fresh-source reproduction is a separate, strict diagnostic route. Its captured
Linux differences never replace a canonical circuit or a frozen cost budget.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

import psutil
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dynamics_record_release as gate

PRODUCER = "scripts/generate_dynamics_resources.py"
CANONICAL_REPLAY = "scripts/dynamics_canonical_replay.py"
LOGICAL_FILES = frozenset({"manifest.json", "summary.md", "dynamics_resources.py",
                          "generate_dynamics_resources.py", "LICENSE.txt"}) | frozenset(
    f"{case}/{name}" for case in gate.CASE_IDS
    for name in ("logical-resources.json", "measured-trial.qasm"))
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_RSS_BYTES = 2 * 1024 * 1024 * 1024
POLL_SECONDS = .25
REPLAY_WALL_SECONDS = 540
CLEANUP_GRACE_SECONDS = 5
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SIGNALS = {signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
PROVENANCE_SCOPE = "Exact checkout root when Git is available; null for standalone packages. Content hashes identify actual source."


def write_json(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    gate.require(len(encoded.encode()) <= gate.MAX_JSON_BYTES, "diagnostic JSON size limit exceeded")
    with path.open("x") as handle:
        handle.write(encoded)


def locked_environment(source_root):
    gate.require(sys.version.split()[0] == gate.NATIVE_ENVIRONMENT["python"],
                 "the recorded Python 3.12.14 interpreter is required")
    versions = {}
    for line in gate.read_bounded(gate.checked_file(source_root, "requirements-lock.txt")).decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        specifiers = list(requirement.specifier)
        gate.require(len(specifiers) == 1 and specifiers[0].operator == "==", "lock requires exact versions")
        actual = importlib.metadata.version(requirement.name)
        gate.require(actual == specifiers[0].version, f"locked package differs: {requirement.name}")
        versions[requirement.name] = actual
    return {"python": sys.version.split()[0], "packages": versions}


def _typed(value):
    if value is None: return {"type": "null"}
    if type(value) is bool: return {"type": "bool", "value": value}
    if type(value) is int: return {"type": "int", "value": value}
    if type(value) is float:
        gate.require(math.isfinite(value), "nonfinite diagnostic number")
        return {"type": "float", "value": value, "hex": value.hex()}
    if type(value) is str:
        return {"type": "str", "length": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}
    if type(value) in (list, dict): return {"type": type(value).__name__, "length": len(value)}
    raise ValueError("Dynamics record: unsupported diagnostic type")


def typed_json_diff(recorded, replayed, *, max_rows=1024, max_nodes=200000, max_depth=64):
    """Keep bounded pointers and finite numeric deltas; never emit string values."""
    rows, total, nodes = [], 0, 0

    def add(path, reason, first=None, second=None, missing=None):
        nonlocal total
        total += 1
        if len(rows) >= max_rows: return
        row = {"path": (path or "/")[:512], "reason": reason}
        if missing != "recorded": row["recorded"] = _typed(first)
        if missing != "replayed": row["replayed"] = _typed(second)
        if missing: row["missing"] = missing
        if type(first) in (int, float) and type(second) in (int, float):
            try: delta = float(second) - float(first)
            except OverflowError: delta = math.inf
            if math.isfinite(delta): row.update(delta=delta, absolute_delta=abs(delta))
        rows.append(row)

    def walk(first, second, pointer, depth):
        nonlocal nodes
        nodes += 1
        gate.require(nodes <= max_nodes and depth <= max_depth, "diagnostic traversal limit exceeded")
        if type(first) is not type(second): add(pointer, "type", first, second)
        elif isinstance(first, dict):
            gate.require(len(first) + len(second) <= max_nodes, "diagnostic key limit exceeded")
            for key in sorted(set(first) | set(second)):
                gate.require(type(key) is str, "diagnostic requires JSON string keys")
                child = (pointer + "/" + key[:512].replace("~", "~0").replace("/", "~1"))[:512]
                if key not in first: add(child, "missing", second=second[key], missing="recorded")
                elif key not in second: add(child, "missing", first=first[key], missing="replayed")
                else: walk(first[key], second[key], child, depth + 1)
        elif isinstance(first, list):
            if len(first) != len(second): add(pointer, "shape", first, second)
            for index, (old, new) in enumerate(zip(first, second)):
                walk(old, new, (pointer + "/" + str(index))[:512], depth + 1)
        elif gate.canonical(first) != gate.canonical(second): add(pointer, "value", first, second)

    walk(recorded, replayed, "", 0)
    return {"different_entries": total, "retained_entries": len(rows), "truncated": total > len(rows),
            "visited_nodes": nodes, "differences": rows}


class LogicalReplayMismatch(ValueError):
    def __init__(self, details):
        self.details = details
        first = next((row for item in details["files"] for row in item["diff"].get("differences", [])), None)
        pointer = first["path"] if first else "/<byte-or-inventory-difference>"
        super().__init__("Dynamics record: logical replay differs at " + pointer)


def normalized_provenance(value, expected_sources):
    value = copy.deepcopy(value)
    gate.require(isinstance(value, dict) and set(value) == {
        "source_commit", "source_revision_scope", "source_sha256", "environment"}, "invalid logical provenance")
    commit = value["source_commit"]
    gate.require(commit is None or isinstance(commit, str) and COMMIT.fullmatch(commit), "invalid replay Git commit")
    gate.require(value["source_revision_scope"] == PROVENANCE_SCOPE, "replay provenance scope differs")
    gate.require(value["source_sha256"] == expected_sources, "replay producer source bindings differ")
    gate.require(gate.canonical(value["environment"]) == gate.canonical(gate.NATIVE_ENVIRONMENT),
                 "logical interpreter/package/device/seed differs")
    value.pop("source_commit"); value.pop("source_sha256")
    return value


def normalized_case(value, expected_sources, *, replay):
    value = copy.deepcopy(value)
    case_id = value.get("case_id")
    gate.require(case_id in gate.CASE_IDS, "unexpected logical case")
    value["provenance"] = normalized_provenance(value.get("provenance"), expected_sources)
    reports = value.pop("physical_reports")
    expected_reports = [] if replay else [p["name"] + ".json" for p in gate.PROFILES]
    gate.require(reports == expected_reports, "logical replay unexpectedly contains physical reports")
    timings = value.pop("cpu_timings_seconds")
    keys = {"construction_and_lowering", "independent_dense_validation"} if value.get("method") == "product_formula" else {
        "construction_validation_lowering"}
    gate.require(isinstance(timings, dict) and set(timings) == keys
                 and all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in timings.values()),
                 "invalid bounded logical CPU timing metadata")
    return value


def normalized_study(value, expected_sources, *, replay):
    value = copy.deepcopy(value)
    gate.require(type(value.get("canonical_cases")) is int and value["canonical_cases"] == 24
                 and type(value.get("actual_physical_queries")) is int
                 and value["actual_physical_queries"] == (0 if replay else 48), "logical study coverage/query count differs")
    gate.require(value["status"] == ("logical_only" if replay else "computed_conditional_model"), "logical study status differs")
    size_scope = value.pop("QSP_physical_size_sweep")
    gate.require(size_scope == (
        "Not computed; n2,n3,n4 logical circuit checks at fixed J1,h.7,t1, with no physical projection" if replay else
        "Computed n2,n3,n4 at fixed J1,h.7,t1; retargeted14/15phases, numerical circuit evidence and conditional physical model, not generic phase/size certification"),
        "logical-only physical-size disclosure differs")
    value.pop("status"); value.pop("actual_physical_queries")
    value["provenance"] = normalized_provenance(value.get("provenance"), expected_sources)
    cases = value.get("cases")
    gate.require(isinstance(cases, list) and [case.get("case_id") for case in cases] == list(gate.CASE_IDS),
                 "logical case roster/order differs")
    value["cases"] = [normalized_case(case, expected_sources, replay=replay) for case in cases]
    return value


def output_inventory(output, workdir=None, *, complete=False):
    """Check partial writes while running, then require the exact 53-file output."""
    inventory, total, entries = {}, 0, 0
    if output.exists():
        gate.require(output.is_dir() and not output.is_symlink() and output.absolute() == output.resolve(), "unsafe replay output")
        for path in output.rglob("*"):
            entries += 1
            gate.require(entries <= len(LOGICAL_FILES) + len(gate.CASE_IDS), "replay entry limit exceeded")
            name = path.relative_to(output).as_posix()
            gate.require(not path.is_symlink(), "linked replay output")
            if path.is_dir():
                gate.require(name in gate.CASE_IDS, "unexpected replay directory")
                continue
            gate.require(name in LOGICAL_FILES and path.is_file() and stat.S_ISREG(path.stat().st_mode), "unexpected replay file")
            size = path.stat().st_size
            total += size
            gate.require(total <= MAX_OUTPUT_BYTES, "replay output size limit exceeded")
            gate.require(not name.endswith(".json") or size <= gate.MAX_JSON_BYTES, "replay JSON size limit exceeded")
            inventory[name] = {"bytes": size}
            if complete:
                gate.require(size > 0, "empty completed replay file")
                inventory[name]["sha256"] = gate.sha256(path, MAX_OUTPUT_BYTES)
    if workdir is not None:
        for name in ("producer.stdout", "producer.stderr"):
            path = workdir / name
            if path.exists():
                gate.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_LOG_BYTES,
                             "replay log size/type limit exceeded")
    if complete: gate.require(set(inventory) == LOGICAL_FILES, "completed logical output inventory differs")
    return inventory


def compare_logical_outputs(records, replay, record_manifest, source_root):
    output_inventory(replay, complete=True)
    recorded = gate.load_json(gate.checked_file(records, "manifest.json"))
    rebuilt = gate.load_json(gate.checked_file(replay, "manifest.json"))
    native_sources = record_manifest["native_provenance"]["source_sha256"]
    replay_sources = {name: record_manifest["replay_sources"][name] for name in gate.NATIVE_SOURCE_FILES}
    files, retained_rows = [], 0
    def compare(name, first, second):
        nonlocal retained_rows
        diff = typed_json_diff(first, second, max_rows=max(0, 1024 - retained_rows))
        retained_rows += diff["retained_entries"]
        if diff["different_entries"]: files.append({"file": name, "diff": diff})
    compare("manifest.json", normalized_study(recorded, native_sources, replay=False),
            normalized_study(rebuilt, replay_sources, replay=True))
    for case_id in gate.CASE_IDS:
        name = f"{case_id}/logical-resources.json"
        original = gate.load_json(gate.checked_file(records, name))
        actual = gate.load_json(gate.checked_file(replay, name))
        gate.require(gate.canonical(actual) == gate.canonical(next(case for case in rebuilt["cases"] if case["case_id"] == case_id)),
                     "rebuilt case ledger differs from its manifest")
        compare(name, normalized_case(original, native_sources, replay=False), normalized_case(actual, replay_sources, replay=True))
        qasm = f"{case_id}/measured-trial.qasm"
        first, second = gate.checked_file(records, qasm), gate.checked_file(replay, qasm)
        if gate.read_bounded(first, MAX_OUTPUT_BYTES) != gate.read_bounded(second, MAX_OUTPUT_BYTES):
            files.append({"file": qasm, "diff": {"different_entries": 1, "differences": [
                {"path": "/", "reason": "bytes", "recorded_sha256": gate.sha256(first), "replayed_sha256": gate.sha256(second)}]}})
        gate.require(actual.get("qasm_sha256") == gate.sha256(second), "rebuilt QASM ledger binding differs")
    for name, source in (("dynamics_resources.py", "src/openquantum/dynamics_resources.py"),
                         ("generate_dynamics_resources.py", PRODUCER), ("LICENSE.txt", "LICENSE")):
        gate.require(gate.sha256(gate.checked_file(replay, name)) == record_manifest["replay_sources"][source],
                     "replay source copy differs from authenticated current source")
    expected_summary = "\n".join([
        "| Method | Cases | Measured circuit/approximation evidence | Physical evidence |",
        "|---|---:|---|---|",
        "| Product formulas | 18 | Actual n=2–6 size cases, t=.5/1/2 and accuracy .025/.05/.1 slices; analytic step bounds | 0 conditional profile queries |",
        "| Complex QSP and integer OAA | 4 | Three-spin t=1 accepted blocks, rounds 0–3; every failed attempt charged | 0 conditional profile queries; three-spin amplification slice |",
        "| Retargeted complex QSP | 2 additional | n2/n4 at fixed J1,h.7,t1; all actual accepted-block basis columns checked, alongside n3 above | 0 conditional profile queries; bounded three-size slice |",
        "", "The mean synthesis fit is not an error certificate. Initialization/reset and classical communication costs remain excluded.",
    ]) + "\n"
    gate.require(gate.read_bounded(gate.checked_file(replay, "summary.md")).decode() == expected_summary,
                 "logical summary must disclose zero physical queries")
    details = {"status": "exact logical comparison passed" if not files else "exact logical comparison failed",
               "numeric_allowances": [], "files": files, "original_cost_budgets_repriced": False}
    if files: raise LogicalReplayMismatch(details)
    return details


@contextmanager
def wall_limit(seconds):
    gate.require(signal.SIGALRM not in signal.pthread_sigmask(signal.SIG_BLOCK, set()), "blocked wall timer")
    gate.require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "cannot replace existing wall timer")
    previous = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP)}
    def expire(number, frame): raise TimeoutError("Dynamics verifier total wall limit or termination")
    for number in previous: signal.signal(number, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in previous.items(): signal.signal(number, handler)


def _owned_processes(child, tracked):
    try:
        leader = psutil.Process(child.pid)
        processes = [leader] + leader.children(recursive=True)
    except psutil.NoSuchProcess: processes = []
    processes = {process.pid: process for process in processes}
    for process in _known_processes(tracked): processes.setdefault(process.pid, process)
    rss = 0
    for process in processes.values():
        try:
            created = process.create_time()
            tracked[process.pid] = created
            rss += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
    return rss


def _known_processes(tracked):
    result = []
    for pid, created in tracked.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created: result.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
    return result


def _signal_owned_group(child, tracked, number):
    # A reaped PID may be reused. Never signal a group whose visible leader
    # is a different process from the acquired replay handle.
    try:
        leader = psutil.Process(child.pid)
        if tracked.get(child.pid) != leader.create_time(): return
    except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
    try: os.killpg(child.pid, number)
    except ProcessLookupError: pass


def cleanup_owned_group(child, tracked, grace=CLEANUP_GRACE_SECONDS):
    """Always force a bounded KILL/reap phase, even if TERM cleanup fails."""
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    error = None

    def phase(number):
        nonlocal error
        deadline = time.monotonic() + grace
        try: _signal_owned_group(child, tracked, number)
        except BaseException as caught: error = error or caught
        try: processes = _known_processes(tracked)
        except BaseException as caught:
            error = error or caught
            processes = []
        for process in processes:
            try: process.send_signal(number)
            except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
            except BaseException as caught: error = error or caught
        try: child.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired: pass
        except BaseException as caught: error = error or caught
        try: psutil.wait_procs(processes, timeout=max(0., deadline - time.monotonic()))
        except BaseException as caught: error = error or caught

    try:
        try: phase(signal.SIGTERM)
        finally: phase(signal.SIGKILL)
        try: child.wait(timeout=.001)
        except subprocess.TimeoutExpired as caught:
            raise RuntimeError("Dynamics replay child was not reaped after bounded cleanup") from caught
        remaining = []
        for process in _known_processes(tracked):
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE: remaining.append(process)
            except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
        gate.require(not remaining, "owned replay descendants survived bounded cleanup")
        if error is not None: raise error
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def run_producer(source_root, output, workdir, timeout, *, rss_limit=MAX_RSS_BYTES,
                 output_limit=MAX_OUTPUT_BYTES):
    gate.require(not output.exists(), "logical output must be a new directory")
    gate.require(0 < timeout <= REPLAY_WALL_SECONDS, "replay wall allowance exceeds 540 seconds")
    gate.require(rss_limit == MAX_RSS_BYTES and output_limit == MAX_OUTPUT_BYTES, "production replay limits are fixed")
    command = [sys.executable, "-I", "-B", str(gate.checked_file(source_root, PRODUCER)),
               "--no-estimate", "--output-dir", str(output)]
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="")
    started, child, tracked, peak, status, error = time.monotonic(), None, {}, 0, None, None
    with (workdir / "producer.stdout").open("w") as stdout, (workdir / "producer.stderr").open("w") as stderr:
        try:
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
            try:
                restore = ("import json,os,signal,sys; signal.pthread_sigmask(signal.SIG_SETMASK,set(json.loads(sys.argv[1]))); "
                           "os.execv(sys.argv[2],sys.argv[2:])")
                launch = [sys.executable, "-I", "-B", "-c", restore,
                          json.dumps(sorted(int(number) for number in old_mask)), *command]
                child = subprocess.Popen(launch, cwd=workdir, env=environment, stdout=stdout, stderr=stderr,
                                         start_new_session=True)
                try: tracked[child.pid] = psutil.Process(child.pid).create_time()
                except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
            finally: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            while True:
                peak = max(peak, _owned_processes(child, tracked))
                gate.require(peak <= rss_limit, "replay process-tree RSS abort threshold exceeded")
                output_inventory(output, workdir)
                if time.monotonic() - started > timeout: raise TimeoutError("Dynamics logical replay exceeded its wall allowance")
                status = child.poll()
                if status is not None: break
                time.sleep(min(POLL_SECONDS, max(.001, timeout - (time.monotonic() - started))))
        except BaseException as caught:
            error = caught
        finally:
            if child is not None:
                try: cleanup_owned_group(child, tracked)
                except BaseException as caught: error = caught
    metrics = {"argv": command, "exit_status": child.returncode if child is not None else status,
               "wall_seconds": time.monotonic() - started,
               "sampled_process_tree_peak_rss_bytes": peak, "rss_abort_threshold_bytes": rss_limit,
               "rss_sampling_seconds": POLL_SECONDS,
               "rss_scope": "sampled sum of replay process/descendant RSS; shared pages may be counted more than once",
               "Native_estimates_run": False, "phase_fitter_run": False, "factory_search_repeated": False}
    if error is None:
        try:
            gate.require(status == 0, "actual logical producer failed")
            output_inventory(output, workdir, complete=True)
            summary = gate.load_json(workdir / "producer.stdout")
            gate.require(gate.canonical(summary) == gate.canonical({"status": "logical_only", "cases": 24, "queries": 0}),
                         "producer did not return a complete no-Native replay")
        except BaseException as caught: error = caught
    if error is not None:
        error.replay_metrics = metrics
        raise error
    return metrics


def _capture_root(capture_dir, records, source_root):
    if capture_dir is None: return None
    path = Path(capture_dir).absolute()
    gate.require(not path.exists() and not path.is_symlink(), "capture directory must be new")
    gate.require(path.parent.is_dir() and not path.parent.is_symlink() and path.parent == path.parent.resolve(), "unsafe capture parent")
    gate.require(not path.is_relative_to(records) and not records.is_relative_to(path), "capture overlaps immutable records")
    if path.is_relative_to(source_root):
        gate.require(path.is_relative_to(source_root / ".cache"), "in-checkout captures must stay in private .cache")
    path.mkdir()
    return path


def _retain(workdir, output, capture, metadata, error=None):
    if capture is None: return
    for name in ("producer.stdout", "producer.stderr"):
        path = workdir / name
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as source, (capture / name).open("xb") as destination:
                destination.write(source.read(MAX_LOG_BYTES))
    # Preserve bounded scientific JSON before checking the aggregate export.
    # A large QASM/extra entry must not erase already completed logical checks.
    copied, failures, captured_bytes = {}, [], 0
    for name in sorted(item for item in LOGICAL_FILES if item.endswith(".json")):
        path = output / name
        if not path.exists() and not path.is_symlink(): continue
        try:
            payload = gate.read_bounded(gate.checked_file(output, name), gate.MAX_JSON_BYTES)
            gate.parse_json_bytes(payload)
            gate.require(captured_bytes + len(payload) <= MAX_OUTPUT_BYTES, "partial JSON capture limit exceeded")
            destination = capture / "logical-replay" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle: handle.write(payload)
            captured_bytes += len(payload)
            copied[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        except (OSError, ValueError, UnicodeError) as capture_error:
            failures.append({"file": name, "type": type(capture_error).__name__, "message": str(capture_error)[:512]})
    write_json(capture / "partial-json-capture.json", {"files": copied, "bytes": captured_bytes,
               "partial": True, "skipped_or_invalid": failures})
    try:
        inventory = output_inventory(output)
        write_json(capture / "replay-output-inventory.json", inventory)
        for name in inventory:
            destination = capture / "logical-replay" / name
            if destination.exists(): continue
            payload = gate.read_bounded(gate.checked_file(output, name), MAX_OUTPUT_BYTES)
            gate.require(captured_bytes + len(payload) <= MAX_OUTPUT_BYTES, "capture aggregate output limit exceeded")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle: handle.write(payload)
            captured_bytes += len(payload)
    except (OSError, ValueError) as capture_error:
        write_json(capture / "capture-output-error.json", {"type": type(capture_error).__name__,
                   "message": str(capture_error)[:512]})
        if error is None: raise
    if isinstance(error, LogicalReplayMismatch): write_json(capture / "logical-differences.json", error.details)


def verify_record(records, manifest_path, expected_sha256, *, source_root=ROOT,
                  max_wall_seconds=600, capture_dir=None):
    started = time.monotonic()
    gate.require(type(max_wall_seconds) in (int, float) and math.isfinite(max_wall_seconds)
                 and 0 < max_wall_seconds <= 600, "total wall cap must be within 600 seconds")
    reserve = min(20., max_wall_seconds * .2)
    before, manifest_hash, capture, error = None, None, None, None
    metadata = {"status": "logical replay not yet verified", "manifest_sha256": expected_sha256,
                "original_core_files": 105, "Native_estimates_run": False,
                "phase_fitter_run": False, "factory_search_repeated": False, "publication_performed": False}
    try:
        # The hard timer remains live through retention after a child/work
        # deadline. Replay itself still leaves the explicit cleanup reserve.
        with wall_limit(max_wall_seconds):
            source_root, records = gate.checked_directory(source_root), gate.checked_directory(records)
            manifest_path = Path(manifest_path).absolute()
            gate.require(manifest_path == records / gate.MANIFEST_NAME, "only the root immutable manifest is accepted")
            capture = _capture_root(capture_dir, records, source_root)
            before = gate.core_inventory(records)
            manifest_hash = gate.sha256(gate.checked_file(records, gate.MANIFEST_NAME))
            identity = gate.verify_dynamics_record_identity(records, source_root, expected_sha256=expected_sha256)
            record_manifest = gate.load_json(manifest_path)
            environment = locked_environment(source_root)
            kernel = os.uname()
            metadata.update(verified_environment=environment, replay_source_sha256=record_manifest["replay_sources"],
                            native_source_sha256=record_manifest["native_provenance"]["source_sha256"],
                            verifier_source_sha256=record_manifest["verifier_sources"],
                            frozen_baseline_sha256=record_manifest["frozen_files"],
                            physical_validation=identity["physical_validation"],
                            source_root=str(source_root), machine=kernel.machine,
                            kernel={"system": kernel.sysname, "release": kernel.release, "version": kernel.version},
                            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                            inherited_thread_environment={name: os.environ.get(name) for name in (
                                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")})
            with tempfile.TemporaryDirectory(prefix="openquantum-dynamics-replay-") as temporary:
                workdir = Path(temporary).resolve()
                output = workdir / "logical-replay"
                gate.require(not workdir.is_relative_to(records), "temporary output overlaps immutable records")
                replay_error = None
                try:
                    remaining = min(REPLAY_WALL_SECONDS, max_wall_seconds - reserve - (time.monotonic() - started))
                    gate.require(remaining > 0, "wall allowance exhausted before replay")
                    metadata["producer"] = run_producer(source_root, output, workdir, remaining)
                    metadata["comparison"] = compare_logical_outputs(records, output, record_manifest, source_root)
                    metadata["status"] = "verified immutable dynamics record and exact logical replay"
                except BaseException as caught:
                    replay_error = caught
                    if hasattr(caught, "replay_metrics"): metadata["producer"] = caught.replay_metrics
                    raise
                finally:
                    _retain(workdir, output, capture, metadata, replay_error)
    except BaseException as caught:
        error = caught
    finally:
        remaining = max_wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            error = TimeoutError("Dynamics verifier exceeded its entire wall allowance")
        else:
            try:
                with wall_limit(remaining):
                    try:
                        if before is not None:
                            gate.require(gate.core_inventory(records) == before
                                         and gate.sha256(gate.checked_file(records, gate.MANIFEST_NAME)) == manifest_hash,
                                         "immutable dynamics files/manifest changed during verification")
                            metadata["original_core_bytes_preserved"] = True
                        if "replay_source_sha256" in metadata:
                            gate.require(gate.source_hashes(source_root, gate.REPLAY_SOURCE_FILES) == metadata["replay_source_sha256"]
                                         and gate.source_hashes(source_root, gate.VERIFIER_FILES) == metadata["verifier_source_sha256"]
                                         and gate.native_source_hashes(records, source_root) == metadata["native_source_sha256"]
                                         and gate.frozen_hashes(records, source_root) == metadata["frozen_baseline_sha256"],
                                         "authenticated replay/Native/frozen sources changed during replay")
                        gate.require(time.monotonic() - started <= max_wall_seconds, "verifier exceeded total wall allowance")
                    except BaseException as caught:
                        error = caught
                    metadata["wall_seconds"] = time.monotonic() - started
                    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    metadata["wait_peak_rss_kib"] = int(usage.ru_maxrss / (1024 if sys.platform == "darwin" else 1))
                    metadata["wait_rss_scope"] = "kernel wait-derived child/descendant maximum; not a concurrent process-tree RSS sum"
                    if error is not None: metadata["status"] = "dynamics verification failed"
                    if capture is not None:
                        write_json(capture / "receipt.json", {**metadata, "error": None if error is None else {
                            "type": type(error).__name__, "message": str(error)[:512]}, "strict_comparison_passed": error is None})
            except BaseException as caught:
                error = caught
    if error is not None: raise error
    return metadata


def canonical_output_inventory(workdir, *, complete=False):
    """Bound the canonical child to one JSON result and two private logs."""
    allowed = {"canonical.stdout", "canonical.stderr", "canonical-replay.json"}
    inventory, total = {}, 0
    for path in workdir.iterdir():
        gate.require(path.name in allowed, "unexpected canonical replay output")
        checked = gate.checked_file(workdir, path.name)
        size = checked.stat().st_size
        total += size
        cap = gate.MAX_JSON_BYTES if path.name.endswith(".json") else MAX_LOG_BYTES
        gate.require(size <= cap and total <= MAX_OUTPUT_BYTES, "canonical replay output size limit exceeded")
        inventory[path.name] = {"bytes": size}
        if complete:
            inventory[path.name]["sha256"] = gate.sha256(checked, cap)
    if complete:
        gate.require(set(inventory) == allowed and inventory["canonical-replay.json"]["bytes"] > 0,
                     "incomplete canonical replay output")
    return inventory


def run_canonical_replay(source_root, records, expected_sha256, workdir, timeout):
    """Execute the authenticated stored-QASM validator, with no fresh producer."""
    gate.require(0 < timeout <= REPLAY_WALL_SECONDS, "canonical replay wall allowance exceeds 540 seconds")
    output = workdir / "canonical-replay.json"
    gate.require(not output.exists(), "canonical replay output must be new")
    command = [sys.executable, "-I", "-B", str(gate.checked_file(source_root, CANONICAL_REPLAY)),
               "--records-dir", str(records), "--source-root", str(source_root),
               "--manifest-sha256", expected_sha256, "--output-json", str(output)]
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    started, child, tracked, peak, status, error = time.monotonic(), None, {}, 0, None, None
    with (workdir / "canonical.stdout").open("x") as stdout, (workdir / "canonical.stderr").open("x") as stderr:
        try:
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
            try:
                restore = ("import json,os,signal,sys; signal.pthread_sigmask(signal.SIG_SETMASK,set(json.loads(sys.argv[1]))); "
                           "os.execv(sys.argv[2],sys.argv[2:])")
                launch = [sys.executable, "-I", "-B", "-c", restore,
                          json.dumps(sorted(int(number) for number in old_mask)), *command]
                child = subprocess.Popen(launch, cwd=workdir, env=environment, stdout=stdout, stderr=stderr,
                                         start_new_session=True)
                try: tracked[child.pid] = psutil.Process(child.pid).create_time()
                except (psutil.NoSuchProcess, psutil.ZombieProcess): pass
            finally: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            while True:
                peak = max(peak, _owned_processes(child, tracked))
                gate.require(peak <= MAX_RSS_BYTES, "canonical replay process-tree RSS abort threshold exceeded")
                canonical_output_inventory(workdir)
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Canonical QASM replay exceeded its wall allowance")
                status = child.poll()
                if status is not None: break
                time.sleep(min(POLL_SECONDS, max(.001, timeout - (time.monotonic() - started))))
        except BaseException as caught:
            error = caught
        finally:
            if child is not None:
                try: cleanup_owned_group(child, tracked)
                except BaseException as caught: error = caught
    metrics = {"argv": command, "exit_status": child.returncode if child is not None else status,
               "wall_seconds": time.monotonic() - started,
               "sampled_process_tree_peak_rss_bytes": peak, "rss_abort_threshold_bytes": MAX_RSS_BYTES,
               "rss_sampling_seconds": POLL_SECONDS,
               "rss_scope": "sampled sum of canonical replay process/descendant RSS; shared pages may be counted more than once",
               "Native_estimates_run": False, "phase_fitter_run": False, "factory_search_repeated": False,
               "fresh_source_generator_executed": False}
    if error is None:
        try:
            gate.require(status == 0, "actual canonical QASM replay failed")
            canonical_output_inventory(workdir, complete=True)
            summary = gate.load_json(workdir / "canonical.stdout")
            gate.require(gate.canonical(summary) == gate.canonical(
                {"status": "canonical_qasm_replay_passed", "cases": 24, "native_estimates": 0}),
                         "canonical child did not return complete no-Native replay")
        except BaseException as caught: error = caught
    if error is not None:
        error.replay_metrics = metrics
        raise error
    return metrics


def _retain_canonical(workdir, capture):
    if capture is None: return
    failures = []
    for name in ("canonical.stdout", "canonical.stderr", "canonical-replay.json"):
        path = workdir / name
        if not path.exists(): continue
        try:
            cap = gate.MAX_JSON_BYTES if name.endswith(".json") else MAX_LOG_BYTES
            payload = gate.read_bounded(gate.checked_file(workdir, name), cap)
            if name.endswith(".json"): gate.parse_json_bytes(payload)
            with (capture / name).open("xb") as destination: destination.write(payload)
        except (OSError, ValueError) as caught:
            failures.append({"file": name, "type": type(caught).__name__, "message": str(caught)[:512]})
    if failures: write_json(capture / "canonical-capture-errors.json", {"errors": failures})


def verify_canonical_record(records, manifest_path, expected_sha256, *, source_root=ROOT,
                            max_wall_seconds=600, capture_dir=None):
    """Validate frozen records and their actual canonical circuit snapshots."""
    started = time.monotonic()
    gate.require(type(max_wall_seconds) in (int, float) and math.isfinite(max_wall_seconds)
                 and 0 < max_wall_seconds <= 600, "total wall cap must be within 600 seconds")
    reserve = min(20., max_wall_seconds * .2)
    before, manifest_hash, capture, record_manifest, error = None, None, None, None, None
    metadata = {"status": "canonical dynamics replay not yet verified", "manifest_sha256": expected_sha256,
                "original_core_files": 105, "Native_estimates_run": False, "phase_fitter_run": False,
                "factory_search_repeated": False, "fresh_source_generator_executed": False,
                "physical_budgets_repriced": False, "publication_performed": False}
    try:
        with wall_limit(max_wall_seconds):
            source_root, records = gate.checked_directory(source_root), gate.checked_directory(records)
            manifest_path = Path(manifest_path).absolute()
            gate.require(manifest_path == records / gate.MANIFEST_NAME, "only the root immutable manifest is accepted")
            capture = _capture_root(capture_dir, records, source_root)
            before = gate.core_inventory(records)
            manifest_hash = gate.sha256(gate.checked_file(records, gate.MANIFEST_NAME))
            identity = gate.verify_dynamics_record_identity(records, source_root, expected_sha256=expected_sha256)
            record_manifest = gate.load_json(manifest_path, expected_sha256=expected_sha256)
            environment = locked_environment(source_root)
            kernel = os.uname()
            metadata.update(verified_environment=environment, replay_source_sha256=record_manifest["replay_sources"],
                            native_source_sha256=record_manifest["native_provenance"]["source_sha256"],
                            verifier_source_sha256=record_manifest["verifier_sources"],
                            frozen_baseline_sha256=record_manifest["frozen_files"],
                            source_reproduction_evidence=record_manifest["source_reproduction_evidence"],
                            recorded_source_reproduction=identity["source_reproduction_validation"],
                            physical_validation=identity["physical_validation"], source_root=str(source_root),
                            machine=kernel.machine,
                            kernel={"system": kernel.sysname, "release": kernel.release, "version": kernel.version},
                            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                            inherited_thread_environment={name: os.environ.get(name) for name in (
                                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")})
            with tempfile.TemporaryDirectory(prefix="openquantum-dynamics-canonical-") as temporary:
                workdir = Path(temporary).resolve()
                try:
                    remaining = min(REPLAY_WALL_SECONDS, max_wall_seconds - reserve - (time.monotonic() - started))
                    gate.require(remaining > 0, "wall allowance exhausted before canonical replay")
                    metadata["canonical_child"] = run_canonical_replay(source_root, records, expected_sha256, workdir, remaining)
                    result = gate.load_json(gate.checked_file(workdir, "canonical-replay.json"))
                    gate.require(result.get("schema") == "openquantum-dynamics-canonical-qasm-replay-v1"
                                 and result.get("status") == "canonical_qasm_replay_passed"
                                 and type(result.get("case_count")) is int and result["case_count"] == 24,
                                 "canonical replay result coverage/schema differs")
                    for key in ("Native_estimates_run", "phase_fitter_run", "factory_search_repeated",
                                "physical_budgets_repriced", "formal_certificate"):
                        gate.require(result.get(key) is False, "canonical replay cannot claim changed costing or a certificate")
                    gate.require(isinstance(result.get("cases"), list)
                                 and len(result["cases"]) == 24
                                 and all(isinstance(case, dict) for case in result["cases"])
                                 and [case.get("case_id") for case in result["cases"]] == list(gate.CASE_IDS),
                                 "canonical replay result case roster differs")
                    metadata["canonical_qasm_validation"] = result
                    metadata["status"] = "verified immutable dynamics record and canonical QASM numerical replay"
                except BaseException as caught:
                    if hasattr(caught, "replay_metrics"): metadata["canonical_child"] = caught.replay_metrics
                    raise
                finally:
                    _retain_canonical(workdir, capture)
    except BaseException as caught:
        error = caught
    finally:
        remaining = max_wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            error = TimeoutError("Dynamics canonical verifier exceeded its entire wall allowance")
        else:
            try:
                with wall_limit(remaining):
                    try:
                        if before is not None:
                            gate.require(gate.core_inventory(records) == before
                                         and gate.sha256(gate.checked_file(records, gate.MANIFEST_NAME)) == manifest_hash,
                                         "immutable dynamics files/manifest changed during canonical replay")
                            metadata["original_core_bytes_preserved"] = True
                        if record_manifest is not None:
                            final_identity = gate.verify_dynamics_record_identity(records, source_root,
                                                                                expected_sha256=expected_sha256)
                            if "recorded_source_reproduction" in metadata:
                                gate.require(final_identity["source_reproduction_validation"] == metadata["recorded_source_reproduction"],
                                             "recorded source reproduction evidence changed during canonical replay")
                        gate.require(time.monotonic() - started <= max_wall_seconds, "canonical verifier exceeded total wall allowance")
                    except BaseException as caught:
                        error = caught
                    metadata["wall_seconds"] = time.monotonic() - started
                    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                    metadata["wait_peak_rss_kib"] = int(usage.ru_maxrss / (1024 if sys.platform == "darwin" else 1))
                    metadata["wait_rss_scope"] = "kernel wait-derived canonical child/descendant maximum; not a concurrent process-tree RSS sum"
                    metadata["canonical_qasm_replay_passed"] = error is None
                    metadata["fresh_source_byte_reproduction_claimed"] = False
                    if error is not None: metadata["status"] = "dynamics canonical verification failed"
                    if capture is not None:
                        write_json(capture / "receipt.json", {**metadata, "error": None if error is None else {
                            "type": type(error).__name__, "message": str(error)[:512]}})
            except BaseException as caught:
                error = caught
    if error is not None: raise error
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / gate.RECORD_DIRECTORY)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--capture-dir", type=Path)
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--canonical-qasm-replay", action="store_true",
                        help="Validate exact stored QASM against frozen task contracts (the default)")
    routes.add_argument("--fresh-source-reproduction", action="store_true",
                        help="Run the separate original strict fresh-source reproduction diagnostic")
    args = parser.parse_args()
    route = verify_record if args.fresh_source_reproduction else verify_canonical_record
    result = route(args.records_dir, args.records_dir / gate.MANIFEST_NAME, args.manifest_sha256,
                   max_wall_seconds=args.max_wall_seconds, capture_dir=args.capture_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__": main()
