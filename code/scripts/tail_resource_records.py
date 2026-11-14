#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded authentication and saved-only verification of two resource records.

Only the standard library loads before every input has been authenticated.
Native factory results and unavailable Trace JSON remain recorded evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types

SIDECAR = "saved-resource-record.json"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 1024
MAX_LOG_BYTES = 1024 * 1024
MAX_WALL_SECONDS = 600
MAX_REPLAY_SECONDS = 540
MAX_TREE_RSS_KIB = 2 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
PRODUCER_COMMIT = "d4d0aedafc5e92f9a33c3bff34a5e79d83a63af1"
FROZEN_FILES = (
    "src/openquantum/grover.py", "scripts/estimate_resources.py",
    "assets/generated/grover/circuit.qasm",
    "assets/generated/resources/grover-surface-code.json",
    "assets/generated/resources/grover-surface-code.csv",
)
FROZEN_ASSET_FILES = frozenset({
    "assets/generated/grover/circuit.qasm",
    "assets/generated/resources/grover-surface-code.json",
    "assets/generated/resources/grover-surface-code.csv",
})
NUMERICAL_POLICY_FILES = {
    "instrument": "scripts/tail_resource_instrument_policy.json",
    "complete": "scripts/tail_resource_complete_policy.json",
}
ENERGY_FREEZE_FILES = (
    "scripts/generate_energy_estimation.py", "scripts/source_provenance.py",
    "src/openquantum/__init__.py", "src/openquantum/grover.py",
    "src/openquantum/energy_estimation.py", "src/openquantum/qft_qpe.py",
    "src/openquantum/resource_estimation.py", "requirements-lock.txt", "LICENSE",
    "scripts/verify_energy_estimation.py", "scripts/freeze_energy_estimation.py",
    "scripts/energy_physical_records.py",
)
VERIFIER_FILES = (
    "scripts/tail_resource_records.py", "scripts/tail_resource_model.py",
    "scripts/tail_resource_qasm.py", "scripts/tail_resource_replay.py",
    "scripts/tail_resource_numerical.py",
)
KINDS = {
    "instrument": {
        "directory": "assets/generated/instrument-resources",
        "tasks": ("damping-Z", "lindblad-Z", "thermal-Z", "thermal-X"),
        "sources": {
            "instrument_resources.py": "src/openquantum/instrument_resources.py",
            "generate_instrument_resources.py": "scripts/generate_instrument_resources.py",
            **{name: "src/openquantum/simulation/" + name for name in
               ("open_systems.py", "lindblad.py", "thermal_filter.py", "lcu.py")},
            "LICENSE.txt": "LICENSE", "requirements-lock.txt": "requirements-lock.txt",
        },
        "root_files": ("README.md", "manifest.json", "references.bib", "space-time.svg",
                       "summary.csv", "summary.md"),
        "scope": {"stored_points": 23, "reported_feasible_candidates": 859,
                  "unavailable_candidate_ledgers": 836, "positive_profiles": 8,
                  "controller_negatives": 0},
    },
    "complete": {
        "directory": "assets/generated/simulation/complete-resources",
        "tasks": ("dynamics-n1-Z", "dynamics-n2-Z", "thermal-n1-X", "thermal-n2-X", "thermal-n2-Z"),
        "sources": {
            "instrument_resources.py": "src/openquantum/instrument_resources.py",
            "hamiltonian_simulation.py": "src/openquantum/hamiltonian_simulation.py",
            "generate_simulation_complete_resources.py": "scripts/generate_simulation_complete_resources.py",
            **{name: "src/openquantum/simulation/" + name for name in
               ("lcu.py", "retargetable_qsp.py", "thermal_spectral.py", "qsp_phase_generation.py",
                "qsp_certificate.py", "complete_resource_workloads.py")},
            "LICENSE.txt": "LICENSE", "requirements-lock.txt": "requirements-lock.txt",
        },
        "root_files": ("README.md", "manifest.json", "references.bib", "accuracy.svg",
                       "attempts.svg", "space-time.svg", "summary.csv", "summary.md", "results.json"),
        "scope": {"stored_points": 1168, "positive_profiles": 14, "controller_negatives": 6,
                  "positive_common_energy_joins": 2, "negative_common_energy_joins": 2},
    },
}
TRUST_BOUNDARIES = [
    "Native factory properties and absent raw Trace JSON are producer-attested; Trace hashes are identities only.",
    "No Native estimation, factory search, phase fitting or rotation synthesis is executed.",
    "Only retained instrument point ledgers are available; omitted search candidates are not replayed.",
    "Frozen source/Clifford+T replay checks the recorded fixed input, not arbitrary-input channel correctness.",
    "Per-rotation approximant words and routing boundaries are absent; their producer claims are not renewed.",
    "Large repeated workloads, physical hardware and complete-shot ensembles remain unexecuted.",
]

# A stdlib-only bootstrap runs under the aggregate watchdog. It authenticates
# only the pinned core bytes before compiling the core, which then authenticates
# the complete record/source inventory before loading any model or science.
# Public callers authenticate this enclosing file via the complete package
# manifest before this bootstrap is used.
BOOTSTRAP = r'''
import argparse, hashlib, json, math, os, stat, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--worker', required=True)
p.add_argument('--records-dir', required=True)
p.add_argument('--source-root', required=True)
p.add_argument('--canonical-root')
p.add_argument('--sha256', required=True)
p.add_argument('--replay', action='store_true')
a = p.parse_args()
def need(ok, message):
    if not ok: raise ValueError(message)
def read(root, parts, cap):
    root = Path(root).absolute()
    need(root == root.resolve(), 'aliased bootstrap root')
    handles, edges = [], []
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    handles.append(os.open(root.anchor, flags))
    fd = None
    try:
        for part in (*root.parts[1:], *parts[:-1]):
            parent = handles[-1]
            old = os.stat(part, dir_fd=parent, follow_symlinks=False)
            need(stat.S_ISDIR(old.st_mode), 'linked bootstrap ancestor')
            child = os.open(part, flags, dir_fd=parent)
            handles.append(child)
            info = os.fstat(child)
            key = (info.st_dev, info.st_ino, info.st_mode)
            need((old.st_dev, old.st_ino, old.st_mode) == key, 'bootstrap ancestor race')
            edges.append((parent, part, child, key))
        parent = handles[-1]
        old = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        def identity(s):
            return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        need(stat.S_ISREG(old.st_mode) and old.st_nlink == 1 and 0 < old.st_size <= cap, 'unsafe bootstrap file')
        fd = os.open(parts[-1], os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK, dir_fd=parent)
        need(identity(old) == identity(os.fstat(fd)), 'bootstrap file race')
        data = bytearray()
        while True:
            block = os.read(fd, min(65536,cap + 1 - len(data)))
            if not block: break
            data.extend(block)
            need(len(data) <= cap, 'bootstrap byte limit')
        need(len(data) == old.st_size and identity(old) == identity(os.fstat(fd)) ==
             identity(os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)), 'bootstrap file changed')
        for parent,name,child,key in edges:
            item = os.stat(name,dir_fd=parent,follow_symlinks=False)
            info = os.fstat(child)
            need((item.st_dev,item.st_ino,item.st_mode) == key ==
                 (info.st_dev,info.st_ino,info.st_mode), 'bootstrap ancestor changed')
        return bytes(data)
    finally:
        if fd is not None: os.close(fd)
        for handle in reversed(handles): os.close(handle)
side = read(a.records_dir, ('saved-resource-record.json',), 1048576)
need(hashlib.sha256(side).hexdigest() == a.sha256, 'bootstrap sidecar identity differs')
def unique(pairs):
    out = {}
    for k,v in pairs:
        need(k not in out, 'duplicate bootstrap JSON key')
        out[k] = v
    return out
def reject(v): raise ValueError('nonfinite bootstrap JSON')
manifest = json.loads(side,object_pairs_hook=unique,parse_constant=reject)
need(isinstance(manifest,dict) and manifest.get('schema') == 'openquantum-saved-tail-resource-v1'
     and manifest.get('kind') == a.worker, 'bootstrap sidecar schema differs')
entry = manifest['sources']['scripts/tail_resource_records.py']
need(isinstance(entry,dict) and set(entry) == {'bytes','sha256'} and
     type(entry['bytes']) is int and 0 < entry['bytes'] <= 1048576, 'bootstrap core entry invalid')
core = read(a.source_root, ('scripts','tail_resource_records.py'), entry['bytes'])
need(len(core) == entry['bytes'] and hashlib.sha256(core).hexdigest() == entry['sha256'], 'bootstrap core identity differs')
exec(compile(core,'<authenticated saved-tail core>','exec'),
     {'__name__':'__main__','__file__':'<authenticated saved-tail core>'})
'''


class RecordVerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RecordVerificationError(message)


def checked_root(root):
    root = Path(root).absolute()
    require(root == root.resolve(), "linked or aliased root")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), "linked/non-directory ancestor")
    return root


def relative_name(name):
    require(isinstance(name, str) and 0 < len(name) <= 512 and "\\" not in name and "\x00" not in name,
            "unsafe relative path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name
            and len(path.parts) <= 12 and all(p not in ("", ".", "..") for p in path.parts),
            "unsafe relative path")
    return path.parts


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _directory_chain(root, parts=()):
    """Anchor every ancestor at '/', retaining identities until reads finish."""
    root = checked_root(root)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    handles = [os.open(root.anchor, flags)]
    edges = []
    try:
        for part in (*root.parts[1:], *parts):
            parent = handles[-1]
            before = os.stat(part, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode), "linked/non-directory ancestor")
            child = os.open(part, flags, dir_fd=parent)
            handles.append(child)
            after = os.fstat(child)
            require((before.st_dev, before.st_ino, before.st_mode) ==
                    (after.st_dev, after.st_ino, after.st_mode), "directory changed while opening")
            edges.append((parent, part, child, (after.st_dev, after.st_ino, after.st_mode)))
        return handles, edges
    except BaseException:
        for handle in reversed(handles):
            os.close(handle)
        raise


def _check_directory_chain(edges):
    for parent, name, child, expected in edges:
        path = os.stat(name, dir_fd=parent, follow_symlinks=False)
        opened = os.fstat(child)
        require((path.st_dev, path.st_ino, path.st_mode) == expected ==
                (opened.st_dev, opened.st_ino, opened.st_mode), "ancestor identity changed")


def read_regular(root, name, *, max_bytes=MAX_FILE_BYTES):
    """Read through no-follow descriptors; never open a FIFO in blocking mode."""
    parts = relative_name(name)
    handles, edges = _directory_chain(root, parts[:-1])
    directory = handles[-1]
    handle = None
    try:
        before = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "input must be singly linked regular file: " + name)
        require(0 < before.st_size <= max_bytes, "input size outside limit: " + name)
        handle = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        opened = os.fstat(handle)
        require(_identity(before) == _identity(opened), "input changed before read: " + name)
        chunks, size = [], 0
        while True:
            chunk = os.read(handle, min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            require(size <= max_bytes, "input grew beyond limit: " + name)
        after = os.fstat(handle)
        path_after = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        require(_identity(before) == _identity(after) == _identity(path_after)
                and size == before.st_size, "input changed while reading: " + name)
        _check_directory_chain(edges)
        return b"".join(chunks)
    finally:
        if handle is not None:
            os.close(handle)
        for directory in reversed(handles):
            os.close(directory)


def tree_names(root, *, allowed_directories=None):
    """Bound enumeration before reading any entry; reject all special files."""
    root = checked_root(root)
    found, pending = set(), [((), "", 0)]
    count = 0
    while pending:
        parts, prefix, depth = pending.pop()
        require(depth <= 8, "input directory depth limit")
        handles, edges = _directory_chain(root, parts)
        try:
            with os.scandir(handles[-1]) as entries:
                for entry in entries:
                    count += 1
                    require(count <= MAX_ENTRIES, "input directory entry limit")
                    name = prefix + entry.name
                    relative_name(name)
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        require(allowed_directories is None or name in allowed_directories,
                                "unexpected input directory: " + name)
                        pending.append(((*parts, entry.name), name + "/", depth + 1))
                    else:
                        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                                "special or linked input: " + name)
                        found.add(name)
            _check_directory_chain(edges)
        finally:
            for handle in reversed(handles):
                os.close(handle)
    return found


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(data):
    return {name: {"bytes": len(value), "sha256": digest(value)} for name, value in sorted(data.items())}


def parse_json(data):
    require(isinstance(data, bytes) and len(data) <= MAX_FILE_BYTES, "JSON size limit")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate JSON key: " + key[:100])
            value[key] = item
        return value
    def reject(value):
        raise RecordVerificationError("nonfinite JSON constant: " + value)
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise RecordVerificationError("malformed JSON") from error
    require(isinstance(value, dict), "expected JSON object")
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        require(nodes <= 2_000_000 and depth <= 64, "JSON structure limit")
        if isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
        elif type(item) is float:
            require(math.isfinite(item), "nonfinite JSON number")
        elif isinstance(item, str):
            require(len(item) <= MAX_FILE_BYTES, "JSON string size limit")
    return value


def expected_files(kind):
    config = KINDS[kind]
    return set(config["root_files"]) | set(config["sources"]) | {
        task + "/" + name for task in config["tasks"] for name in
        ("report.json", "source-instrument.qasm", "clifford-t-instrument.qasm")}


def collect_inputs(kind, records_dir, source_root, *, with_sidecar=True, canonical_root=None):
    """Authenticate-bound reads only; this never imports executable helpers."""
    require(kind in KINDS, "unknown resource record kind")
    records_dir, source_root = checked_root(records_dir), checked_root(source_root)
    canonical_root = source_root if canonical_root is None else checked_root(canonical_root)
    names = expected_files(kind)
    require(tree_names(records_dir, allowed_directories=set(KINDS[kind]["tasks"])) == names | ({SIDECAR} if with_sidecar else set()),
            "missing/unexpected record entries")
    records, sources, total = {}, {}, 0
    for name in sorted(names):
        value = read_regular(records_dir, name, max_bytes=min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
        records[name] = value
        total += len(value)
    source_names = (set(KINDS[kind]["sources"].values()) | set(FROZEN_FILES) | set(VERIFIER_FILES)
                    | set(ENERGY_FREEZE_FILES) | {NUMERICAL_POLICY_FILES[kind]})
    for name in sorted(source_names):
        input_root = canonical_root if name in FROZEN_ASSET_FILES else source_root
        value = read_regular(input_root, name, max_bytes=min(MAX_FILE_BYTES, MAX_TOTAL_BYTES - total))
        sources[name] = value
        total += len(value)
    # Snapshot citations are historical inputs; scientific sources/lock/license
    # must match current code exactly. Current citations may grow independently.
    manifest = parse_json(records["manifest.json"])
    require(manifest.get("source_commit_at_generation") == PRODUCER_COMMIT
            and manifest.get("dirty_source_paths") == [], "recorded producer identity differs")
    require(manifest.get("source_sha256") == {
        name: digest(records[name]) for name in (*KINDS[kind]["sources"], "references.bib")},
        "producer source snapshot manifest differs")
    for snapshot, source in KINDS[kind]["sources"].items():
        require(records[snapshot] == sources[source], "producer/lock source changed: " + source)
    before = manifest.get("frozen_baseline_before") if kind == "complete" else manifest.get("frozen_grover_baseline_fingerprints", {}).get("before")
    after = manifest.get("frozen_baseline_after") if kind == "complete" else manifest.get("frozen_grover_baseline_fingerprints", {}).get("after")
    frozen = {name: digest(sources[name]) for name in FROZEN_FILES}
    require(before == after == frozen, "Frozen5 identities differ")
    if kind == "complete":
        require(manifest.get("artifacts_sha256") == {name: digest(data) for name, data in records.items() if name != "manifest.json"},
                "complete artifact manifest differs")
        require(manifest.get("case_ids") == list(KINDS[kind]["tasks"])
                and manifest.get("repeated_workloads_executed") is False
                and manifest.get("public_bundle_contains_private_cache") is False,
                "complete execution/publication scope differs")
    else:
        require(manifest.get("models") == list(KINDS[kind]["tasks"])
                and manifest.get("task_artifact_sha256") == {
                    task: {name: digest(records[task + "/" + name]) for name in
                           ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json")}
                    for task in KINDS[kind]["tasks"]}, "instrument task manifest differs")
    return records, sources, manifest


def load_checked_module(name, data):
    """Compile exact already-authenticated bytes, without bytecache reuse."""
    module = types.ModuleType(name)
    module.__file__ = "<authenticated " + name + ">"
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(data, module.__file__, "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _authenticate_record(kind, records_dir, source_root, expected_sha256, *, canonical_root=None):
    require(isinstance(expected_sha256, str) and SHA.fullmatch(expected_sha256), "reviewed sidecar digest required")
    sidecar_bytes = read_regular(records_dir, SIDECAR, max_bytes=MAX_LOG_BYTES)
    require(digest(sidecar_bytes) == expected_sha256, "sidecar differs from reviewed immutable identity")
    sidecar = parse_json(sidecar_bytes)
    require(set(sidecar) == {"schema", "kind", "producer_commit", "files", "sources", "scope", "trust_boundaries"}
            and sidecar["schema"] == "openquantum-saved-tail-resource-v1"
            and sidecar["kind"] == kind and sidecar["producer_commit"] == PRODUCER_COMMIT,
            "invalid saved record sidecar")
    records, sources, manifest = collect_inputs(kind, records_dir, source_root, canonical_root=canonical_root)
    require(sidecar["files"] == inventory(records) and sidecar["sources"] == inventory(sources),
            "saved record/source identity differs")
    require(sidecar["scope"] == KINDS[kind]["scope"] and sidecar["trust_boundaries"] == TRUST_BOUNDARIES,
            "saved verification scope differs")
    return records, sources, manifest, sidecar_bytes


def _verify_in_worker(kind, records_dir, source_root, expected_sha256, replay, *, canonical_root=None):
    started = time.monotonic()
    records, sources, manifest, sidecar_bytes = _authenticate_record(
        kind, records_dir, source_root, expected_sha256, canonical_root=canonical_root)
    # The complete result/report duplication is part of the immutable record.
    reports = {task: parse_json(records[task + "/report.json"]) for task in KINDS[kind]["tasks"]}
    results = parse_json(records["results.json"]) if kind == "complete" else None
    if results is not None:
        require(results.get("cases") == reports, "complete results/report records differ")
    model = load_checked_module("_tail_verified_model", sources["scripts/tail_resource_model.py"])
    qasms = {task: records[task + "/clifford-t-instrument.qasm"].decode("utf-8") for task in reports}
    coverage = (model.verify_instrument_models(reports, qasms) if kind == "instrument"
                else model.verify_complete_models(results, qasms))
    replay_result = numerical_validation = None
    if replay:
        require(time.monotonic() - started < MAX_WALL_SECONDS - 10, "authentication exhausted replay budget")
        numerical = load_checked_module("_tail_verified_numerical", sources["scripts/tail_resource_numerical.py"])
        numerical_policy = parse_json(sources[NUMERICAL_POLICY_FILES[kind]])
        require(numerical_policy.get("schema") == numerical.POLICY_SCHEMA
                and numerical_policy.get("status") == numerical.POLICY_STATUS
                and numerical_policy.get("kind") == kind, "numerical reproduction policy is unreviewed or incompatible")
        parser = load_checked_module("_tail_verified_qasm", sources["scripts/tail_resource_qasm.py"])
        worker = load_checked_module("_tail_verified_replay", sources["scripts/tail_resource_replay.py"])
        replay_started = time.monotonic()
        def expired(signum, frame):
            raise RecordVerificationError("numerical replay exceeded budget")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(MAX_REPLAY_SECONDS, MAX_WALL_SECONDS - 10 - (replay_started - started)))
        try:
            replay_result = worker.run_replay(kind, records, sources, parser)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        require(time.monotonic() - replay_started <= MAX_REPLAY_SECONDS, "numerical replay exceeded budget")
        numerical_validation = numerical.verify_numerical_replay(replay_result, numerical_policy)
    # Re-read to reject path/byte races and leave all canonical inputs unchanged.
    final_records, final_sources, _ = collect_inputs(kind, records_dir, source_root, canonical_root=canonical_root)
    require(inventory(final_records) == inventory(records) and inventory(final_sources) == inventory(sources)
            and read_regular(records_dir, SIDECAR, max_bytes=MAX_LOG_BYTES) == sidecar_bytes,
            "immutable inputs changed during verification")
    return {"status": "verified_saved_record", "kind": kind, "files": len(records),
            "original_bytes": sum(map(len, records.values())), "sidecar_sha256": expected_sha256,
            "coverage": coverage, "replay": replay_result, "numerical_validation": numerical_validation, "native_calls": 0,
            "synthesis_calls": 0, "phase_fitting_calls": 0, "trust_boundaries": TRUST_BOUNDARIES}


def _tree_rss(root_pid, *, timeout=0.5):
    with tempfile.TemporaryFile() as output:
        try:
            subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,rss="], stdout=output,
                           stderr=subprocess.DEVNULL, timeout=timeout, check=True)
        except subprocess.TimeoutExpired as error:
            raise RecordVerificationError("aggregate verifier wall budget exceeded during process inventory") from error
        require(os.fstat(output.fileno()).st_size <= 2 * MAX_LOG_BYTES, "process inventory limit")
        output.seek(0)
        data = output.read(2 * MAX_LOG_BYTES + 1)
        require(len(data) <= 2 * MAX_LOG_BYTES, "process inventory limit")
    entries = []
    for line in data.decode("ascii").splitlines():
        values = line.split()
        require(len(values) == 3, "malformed process inventory")
        entries.append(tuple(map(int, values)))
    require(len(entries) <= 65536, "process inventory entry limit")
    family = {root_pid}
    for _ in range(512):
        new = {pid for pid, ppid, _ in entries if ppid in family} - family
        if not new:
            break
        family.update(new)
        require(len(family) <= 512, "verifier process-tree limit")
    return sum(rss for pid, _, rss in entries if pid in family)


def _stop_group(process, deadline):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.001, min(1, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            continue
    require(process.poll() is not None, "verifier subprocess could not be reaped")


def verify_bundle(records_dir, source_root, *, kind, expected_sha256, replay=False,
                  max_wall_seconds=MAX_WALL_SECONDS, max_tree_rss_kib=MAX_TREE_RSS_KIB, canonical_root=None):
    """Run complete verification under one aggregate wall/RSS/log watchdog."""
    started = time.monotonic()
    require(type(max_wall_seconds) in (int, float) and 0 < max_wall_seconds <= MAX_WALL_SECONDS,
            "wall budget outside 0..600 seconds")
    require(type(max_tree_rss_kib) is int and 0 < max_tree_rss_kib <= MAX_TREE_RSS_KIB,
            "RSS budget outside 0..2 GiB")
    if replay:
        # A parent deadline also bounds blocked numerical/C calls that cannot
        # receive the worker's Python alarm promptly.
        max_wall_seconds = min(max_wall_seconds, MAX_REPLAY_SECONDS)
    deadline = started + max_wall_seconds
    cleanup_reserve = min(2.1, max_wall_seconds * 0.4)
    require(kind in KINDS and isinstance(expected_sha256, str) and SHA.fullmatch(expected_sha256), "invalid verifier arguments")
    records_dir, source_root = checked_root(records_dir), checked_root(source_root)
    canonical_root = source_root if canonical_root is None else checked_root(canonical_root)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    with tempfile.TemporaryDirectory(prefix="tail-resource-watchdog-") as temporary:
        command = [sys.executable, "-I", "-B", "-c", BOOTSTRAP, "--worker", kind, "--records-dir", str(records_dir),
                   "--source-root", str(source_root), "--canonical-root", str(canonical_root), "--sha256", expected_sha256]
        if replay:
            command.append("--replay")
        log = Path(temporary) / "worker.json"
        with log.open("xb") as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                       cwd=source_root, env=environment, start_new_session=True)
            peak = 0
            try:
                while process.poll() is None:
                    remaining = deadline - time.monotonic()
                    require(remaining > cleanup_reserve + 0.01, "aggregate verifier wall budget exceeded")
                    require(log.stat().st_size <= MAX_LOG_BYTES, "verifier output limit exceeded")
                    peak = max(peak, _tree_rss(os.getpid(), timeout=min(0.5, remaining - cleanup_reserve)))
                    require(peak <= max_tree_rss_kib, "sampled verifier process-tree RSS budget exceeded")
                    time.sleep(min(0.1, max(0, deadline - cleanup_reserve - time.monotonic())))
                require(log.stat().st_size <= MAX_LOG_BYTES, "verifier output limit exceeded")
                result_bytes = read_regular(log.parent.resolve(), log.name, max_bytes=MAX_LOG_BYTES)
                require(process.returncode == 0, "saved verifier failed: " + result_bytes[-4000:].decode("utf-8", errors="replace"))
                require(time.monotonic() < deadline - cleanup_reserve, "aggregate verifier wall budget exceeded")
                result = parse_json(result_bytes)
            finally:
                _stop_group(process, deadline)
        result["watchdog"] = {"wall_seconds": time.monotonic() - started, "sampled_peak_tree_rss_kib": peak,
                              "max_wall_seconds": max_wall_seconds, "max_replay_seconds": MAX_REPLAY_SECONDS,
                              "max_tree_rss_kib": max_tree_rss_kib, "max_log_bytes": MAX_LOG_BYTES,
                              "cleanup": "finite process-group TERM/KILL and direct-child reap"}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=KINDS, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    result = _verify_in_worker(args.worker, args.records_dir, args.source_root, args.sha256, args.replay,
                               canonical_root=args.canonical_root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
