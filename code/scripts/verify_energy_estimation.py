#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify an immutable energy study without repeating large native estimation.

The physical models were computed in a separately recorded bounded job. This
verifier checks their identities and accounting, and runs the current producer
with --no-estimate in a temporary directory. It does not execute QDK estimates
or repeat a native factory search. Publication must supply the reviewed SHA256
of the manifest; creating a manifest never overwrites computed native records.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
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
import xml.etree.ElementTree as ET

from packaging.requirements import Requirement

from energy_physical_records import validate_physical_records

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "generate_energy_estimation.py"
MANIFEST_NAME = "resource-record-manifest.json"
CASES = ("default", "n2", "n4", "n6", "n8", "analytic-powers")
PROFILE = {"name": "illustrative-p-1e-4", "physical_error_rate": 1e-4,
           "gate_time_ns": 100, "measurement_time_ns": 500}
SOURCES = ("scripts/generate_energy_estimation.py", "scripts/source_provenance.py",
           "src/openquantum/__init__.py", "src/openquantum/grover.py",
           "src/openquantum/energy_estimation.py", "src/openquantum/qft_qpe.py",
           "src/openquantum/resource_estimation.py", "requirements-lock.txt", "LICENSE")
VERIFIER_SOURCES = ("scripts/verify_energy_estimation.py", "scripts/freeze_energy_estimation.py",
                    "scripts/energy_physical_records.py")
LEDGER_SOURCES = ("scripts/generate_energy_estimation.py", "src/openquantum/energy_estimation.py",
                  "src/openquantum/qft_qpe.py", "src/openquantum/resource_estimation.py",
                  "requirements-lock.txt")
ROOT_FILES = ("logical-resources.json", "measured-group.qasm", "molecular-fixture.json",
              "results.json", "surface-code.json", "surface-code-scaling.json",
              "access-comparison.json", "recycled-clock-resources.json", "recycled-clock.qasm",
              "dimer-access.svg", "small-qpe.svg", "recycled-clock.svg",
              "energy-distributions.svg", "scaling.svg", "scaling.csv", "energy_estimation.py",
              "qft_qpe.py", "resource_estimation.py", PRODUCER, "LICENSE.txt")
CORE_FILES = frozenset(ROOT_FILES) | frozenset(
    f"{case}/{name}" for case in CASES[1:]
    for name in ("logical-resources.json", "measured-group.qasm", "surface-code.json"))
PHYSICAL_FILES = frozenset("surface-code.json" if case == "default"
                           else f"{case}/surface-code.json" for case in CASES)
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
CONTRACT = "Native models computed separately; routine CI replays logical work and validates recorded models, without native re-estimation or factory search."
REFERENCE_ATOL = 1e-12
REFERENCE_ARRAYS = ("eigenvalues", "phases", "marginal", "conditional_laws")
SMALL_COMPLETE_PARAMETERS = {"dimers": 1, "phase_bits": 3, "rounds": 3}
SMALL_COMPLETE_RESIDUAL = "small_complete_11wire_joint_max_error"
SMALL_COMPLETE_ATOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def load_json(path):
    require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
            f"JSON input must be a regular nonlinked file: {path.name}")
    require(path.stat().st_size <= 128 * 1024 * 1024, f"JSON input size limit exceeded: {path.name}")
    value = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject_constant)
    require(isinstance(value, dict), f"Expected JSON object: {path.name}")
    def finite(item):
        if isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
        elif type(item) is float:
            require(math.isfinite(item), "Nonfinite JSON number")
    finite(value)
    return value


def checked_file(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and all(part not in ("", ".", "..")
            for part in relative.parts), f"Unsafe relative path: {name}")
    require(root.is_dir() and not root.is_symlink(), "Records/source root must be a regular directory")
    require(root.absolute() == root.resolve(), "Symlinked/unnormalized root ancestor")
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), f"Linked file/directory: {name}")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode)
            and path.resolve().is_relative_to(root.resolve()), f"Missing/unsafe file: {name}")
    return path


def core_inventory(root):
    # The separately owned Pauli tree is not a producer input. No unexplained
    # core extras are accepted, including PNGs (this producer writes SVGs).
    result = {}
    for name in sorted(CORE_FILES):
        path = checked_file(root, name)
        result[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == "pauli-access" or relative.startswith("pauli-access/"):
            continue
        if path.is_dir() and not path.is_symlink():
            require(relative in CASES[1:], f"Unexpected core directory: {relative}")
        elif relative not in CORE_FILES and relative != MANIFEST_NAME:
            raise ValueError(f"Unexpected core entry: {relative}")
        elif relative in CORE_FILES:
            checked_file(root, relative)
    return result


def source_hashes(source_root):
    return {name: sha256(checked_file(source_root, name)) for name in SOURCES}


def verifier_hashes(source_root):
    return {name: sha256(checked_file(source_root, name)) for name in VERIFIER_SOURCES}


def locked_environment(source_root):
    require(sys.version.split()[0].startswith("3.12."), "Python 3.12 is required")
    versions = {}
    for line in checked_file(source_root, "requirements-lock.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        specifiers = list(requirement.specifier)
        require(len(specifiers) == 1 and specifiers[0].operator == "==",
                "Lock must contain exact package versions")
        actual = importlib.metadata.version(requirement.name)
        require(actual == specifiers[0].version, f"Locked package mismatch: {requirement.name}")
        versions[requirement.name] = actual
    return {"python": sys.version.split()[0], "packages": versions}


def ledger_path(case):
    return "logical-resources.json" if case == "default" else f"{case}/logical-resources.json"


def load_ledgers(records, sources):
    ledgers = {case: load_json(checked_file(records, ledger_path(case))) for case in CASES}
    expected = {name: sources[name] for name in LEDGER_SOURCES}
    for case, ledger in ledgers.items():
        require(ledger.get("source_sha256") == expected, f"Stale producer/adapter/lock: {case}")
        require(ledger.get("source_dirty") is None or ledger.get("source_dirty") is False,
                f"Dirty/invalid native source metadata: {case}")
        commit = ledger.get("source_commit")
        require(commit is None or isinstance(commit, str) and COMMIT.fullmatch(commit),
                f"Invalid source Git metadata: {case}")
    contract = ledgers["default"]["contract"]
    require((contract["dimers"], contract["phase_bits"], contract["requested_ideal_failure"])
            == (1, 5, .005), "Frozen core record must contain the full default experiment")
    require(ledgers["default"]["sequential_equivalent_simulation"]["performed"] is True,
            "Default simulation was not performed")
    for filename, source in ((PRODUCER, "scripts/" + PRODUCER),
                             ("energy_estimation.py", "src/openquantum/energy_estimation.py"),
                             ("qft_qpe.py", "src/openquantum/qft_qpe.py"),
                             ("resource_estimation.py", "src/openquantum/resource_estimation.py"),
                             ("LICENSE.txt", "LICENSE")):
        require(sha256(checked_file(records, filename)) == sources[source],
                f"Archived source differs from current producer: {filename}")
    return ledgers


def freeze_record(records, output, native_environment, native_measurement, *, source_root=ROOT):
    """Freeze operator-reviewed computed records, not attest execution cryptographically."""
    records, output = records.absolute(), output.absolute()
    require(not output.exists() and not output.is_symlink(), "Manifest output already exists")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "Unsafe manifest output parent")
    require(output.parent.absolute() == output.parent.resolve(), "Symlinked manifest output ancestor")
    require(output == records / MANIFEST_NAME or not output.resolve().is_relative_to(records.resolve()),
            "Manifest output must be the root manifest or outside all recorded trees")
    sources = source_hashes(source_root)
    inventory = core_inventory(records)
    ledgers = load_ledgers(records, sources)
    environment = load_json(native_environment)
    measurement = load_json(native_measurement)
    require(isinstance(environment.get("source_commit"), str)
            and COMMIT.fullmatch(environment["source_commit"]), "Native run lacks producer commit")
    require(all(environment.get("source_sha256", {}).get(name) == digest
                for name, digest in sources.items()), "Native input/source/lock binding mismatch")
    require(type(measurement.get("conventional_exit_status")) is int
            and measurement["conventional_exit_status"] == 0
            and type(measurement.get("child_return_code")) is int
            and measurement["child_return_code"] == 0, "Native run did not complete successfully")
    command = measurement.get("command")
    require(isinstance(command, list) and len(command) in (2, 4)
            and all(isinstance(token, str) and token for token in command)
            and Path(command[1]).name == PRODUCER
            and (len(command) == 2 or command[2] == "--output-dir" and not command[3].startswith("-")),
            "Native receipt must show the complete default producer, without disabling work")
    require(environment.get("device") == "CPU", "Recorded device is unsupported by this frozen study")
    require(type(environment.get("seed")) is int and environment["seed"] == 20261112,
            "Native sample seed differs")
    version = ledgers["default"]["environment"]["python"]
    require(environment.get("python", "").split()[0] == version, "Native Python binding mismatch")
    packages = ledgers["default"]["environment"]["packages"]
    require(environment.get("packages") == packages, "Native package binding mismatch")
    require(type(measurement.get("wall_seconds")) in (int, float)
            and math.isfinite(measurement["wall_seconds"]) and measurement["wall_seconds"] > 0
            and type(measurement.get("ru_maxrss_kib")) is int and measurement["ru_maxrss_kib"] > 0,
            "Native receipt lacks finite wall/peak measurements")
    validation = validate_physical_records(records, ledgers, PROFILE)
    before = core_inventory(records)
    require(before == inventory, "Computed records changed during freeze")
    manifest = {"schema_version": 1, "status": "frozen_computed_conditional_energy_record",
                "files": inventory, "sources": sources, "verifier_sources": verifier_hashes(source_root),
                "physical_profile": PROFILE,
                "native_execution": {"producer_commit": environment["source_commit"],
                    "python": version, "packages": packages, "device": "CPU", "seed": 20261112,
                    "complete_default_producer": True, "native_estimation_performed": True,
                    "exit_status": 0, "wall_seconds": measurement["wall_seconds"],
                    "peak_rss_kib": measurement["ru_maxrss_kib"]},
                "physical_validation": validation,
                "contract": CONTRACT}
    # O_EXCL prevents accidental replacement even if another writer races us.
    with output.open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {"manifest": str(output), "manifest_sha256": sha256(output),
            "native_records_overwritten": False, "files": len(inventory)}


def normalized_ledger(value, *, default):
    value = json.loads(canonical(value))
    timing = value.pop("construction_and_lowering_wall_seconds")
    require(isinstance(timing, (int, float)) and not isinstance(timing, bool)
            and math.isfinite(timing) and timing >= 0, "Invalid construction timing")
    commit, dirty = value["source_commit"], value["source_dirty"]
    require(commit is None or isinstance(commit, str) and COMMIT.fullmatch(commit), "Invalid replay Git commit")
    require(dirty is None or type(dirty) is bool, "Invalid replay Git dirty flag")
    require(value["source_commit_scope"] == "exact checkout at generation, if available; content hashes identify working source",
            "Git provenance scope differs")
    for field in ("source_commit", "source_dirty", "source_commit_scope"):
        value.pop(field)
    if default:
        value.pop("physical_resource_estimate")
    return value


def normalized_svg(path):
    root = ET.parse(path).getroot()
    ids = {node.attrib["id"]: f"stable{index}" for index, node in enumerate(root.iter()) if "id" in node.attrib}
    for node in root.iter():
        if node.tag == "{http://purl.org/dc/elements/1.1/}date":
            node.text = "host-render-date"
        for key, value in list(node.attrib.items()):
            if key == "id": node.attrib[key] = ids[value]
            elif value.startswith("#") and value[1:] in ids: node.attrib[key] = "#" + ids[value[1:]]
            elif value.startswith("url(#") and value.endswith(")") and value[5:-1] in ids:
                node.attrib[key] = "url(#" + ids[value[5:-1]] + ")"
    return ET.tostring(root)


def first_json_difference_pointer(first, second):
    """Bound error diagnostics to a JSON pointer, without exposing values."""
    remaining = 4096

    def bounded(pointer):
        return (pointer or "/")[:256]

    def visit(left, right, pointer, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 32:
            return bounded(pointer + "/<diagnostic-limit>")
        if type(left) is not type(right):
            return bounded(pointer)
        if isinstance(left, dict):
            if len(left) > 4096 or len(right) > 4096:
                return bounded(pointer + "/<diagnostic-limit>")
            for key in sorted(set(left) | set(right)):
                child = pointer + "/" + key[:256].replace("~", "~0").replace("/", "~1")
                if key not in left or key not in right:
                    return bounded(child)
                difference = visit(left[key], right[key], child, depth + 1)
                if difference is not None:
                    return difference
        elif isinstance(left, list):
            if len(left) != len(right):
                return bounded(pointer)
            for index, (old, new) in enumerate(zip(left, right)):
                difference = visit(old, new, pointer + "/" + str(index), depth + 1)
                if difference is not None:
                    return difference
        elif canonical(left) != canonical(right):
            return bounded(pointer)
        return None

    return visit(first, second, "", 0)


def compare_energy_results(recorded, replayed):
    """Check four nominal arrays and the fixed 11-wire validation residual.

    The producer checks all complete-circuit probabilities at absolute 1e-12,
    relative zero. Each recorded/replayed maximum residual must satisfy that
    same bound for the exact fixed experiment. Sampled outcomes, weights,
    estimates, other JSON and every immutable file hash remain exact.
    """
    values = [json.loads(canonical(value)) for value in (recorded, replayed)]
    references = []
    for value in values:
        require(canonical(value.get("small_complete_parameters")) == canonical(SMALL_COMPLETE_PARAMETERS),
                "Small complete validation parameters differ")
        residual = value.get(SMALL_COMPLETE_RESIDUAL)
        require(type(residual) is float and math.isfinite(residual)
                and 0 <= residual <= SMALL_COMPLETE_ATOL,
                "Small complete 11-wire residual must be a finite float in [0, 1e-12]")
        value.pop(SMALL_COMPLETE_RESIDUAL)
        reference = value.get("single_round_reference")
        require(isinstance(reference, dict), "Missing single-round spectral reference")
        checked = {}
        for name in REFERENCE_ARRAYS:
            array = reference.get(name)
            rows = array if name == "conditional_laws" else [array]
            require(isinstance(array, list) and len(array) == (32 if name == "marginal" else 4),
                    f"Spectral reference shape differs: {name}")
            for row in rows:
                require(isinstance(row, list) and len(row) == (32 if name in ("marginal", "conditional_laws") else 4),
                        f"Spectral reference row shape differs: {name}")
                require(all(type(item) is float and math.isfinite(item) for item in row),
                        f"Spectral reference requires finite floats: {name}")
                if name in ("marginal", "conditional_laws"):
                    require(all(0 <= item <= 1 for item in row), f"Spectral probability range differs: {name}")
                    require(abs(math.fsum(row) - 1) <= REFERENCE_ATOL,
                            f"Spectral probability normalization differs: {name}")
                elif name == "phases":
                    require(all(.25 <= item <= .75 for item in row), "Spectral phase range differs")
                    require(row == sorted(row), "Spectral phase order differs")
                else:
                    require(row == sorted(row), "Spectral eigenvalue order differs")
            checked[name] = reference.pop(name)
        weights = reference.get("weights")
        require(isinstance(weights, list) and len(weights) == 4
                and all(type(item) is float and math.isfinite(item) and 0 <= item <= 1 for item in weights)
                and abs(math.fsum(weights) - 1) <= REFERENCE_ATOL, "Invalid fixed spectral weights")
        require(all(abs(checked["marginal"][index] - math.fsum(
                    weights[row] * checked["conditional_laws"][row][index] for row in range(4))) <= REFERENCE_ATOL
                    for index in range(32)), "Spectral marginal consistency differs")
        references.append(checked)
    for name in REFERENCE_ARRAYS:
        first, second = references[0][name], references[1][name]
        pairs = zip(first, second) if name == "conditional_laws" else [(first, second)]
        for old, new in pairs:
            require(all(abs(left - right) <= REFERENCE_ATOL for left, right in zip(old, new)),
                    f"Spectral reference exceeds absolute 1e-12 allowance: {name}")
    if canonical(values[0]) != canonical(values[1]):
        pointer = first_json_difference_pointer(values[0], values[1])
        raise ValueError("Results differ outside four spectral-reference arrays and validated 11-wire residual at "
                         + (pointer or "/<diagnostic-limit>"))


def compare_logical_outputs(records, replay, ledgers):
    replay_ledgers = {}
    for case in CASES:
        rebuilt = load_json(checked_file(replay, ledger_path(case)))
        if case == "default":
            require(rebuilt["physical_resource_estimate"] == {
                "status": "unavailable", "kind": None, "pareto_points": 0,
                "hardware_energy_confidence_certified": False}, "Replay unexpectedly contains native estimates")
        require(canonical(normalized_ledger(ledgers[case], default=case == "default"))
                == canonical(normalized_ledger(rebuilt, default=case == "default")),
                f"Logical replay differs: {case}")
        replay_ledgers[case] = rebuilt
    require(not any((replay / name).exists() for name in PHYSICAL_FILES), "Replay wrote native reports")
    access = load_json(checked_file(replay, "access-comparison.json"))
    recorded_access = load_json(checked_file(records, "access-comparison.json"))
    for name in ("repeated", "analytic_powers"):
        require(access[name]["status"] == "unavailable" and access[name]["pareto_points"] == [],
                "Replay access comparison unexpectedly estimates physical costs")
        for value in (access, recorded_access):
            value[name].pop("status"); value[name].pop("pareto_points")
    require(canonical(access) == canonical(recorded_access), "Logical access comparison differs")
    scaling = load_json(checked_file(replay, "surface-code-scaling.json"))
    require(scaling == {"status": "unavailable", "common_absolute_energy_error_target_Ha": .5,
            "same_ideal_failure": .005, "hardware_energy_confidence_certified": False, "points": []},
            "Replay scaling unexpectedly contains estimates")
    ignored = PHYSICAL_FILES | {"access-comparison.json", "surface-code-scaling.json"} | {
        ledger_path(case) for case in CASES}
    for name in sorted(CORE_FILES - ignored):
        first, second = checked_file(records, name), checked_file(replay, name)
        if name == "results.json":
            compare_energy_results(load_json(first), load_json(second))
            equal = True
        elif name.endswith(".json"):
            equal = canonical(load_json(first)) == canonical(load_json(second))
        elif name.endswith(".svg"):
            equal = normalized_svg(first) == normalized_svg(second)
        else:
            equal = first.read_bytes() == second.read_bytes()
        require(equal, f"Logical/sample/circuit export differs: {name}")
    return replay_ledgers


def cleanup_owned_group(child, grace):
    """Kill the owned session even after its leader exits; bound both waits."""
    cleanup_signals = {signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, cleanup_signals)
    try:
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try:
            child.wait(timeout=grace)
        except BaseException:
            # An interrupt during TERM must not bypass the following KILL.
            pass
        finally:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Owned replay child was not reaped within bounded KILL grace") from error
    finally:
        # A pending outer alarm is delivered only after group cleanup.
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


def run_producer(source_root, output, workdir, timeout, *, cleanup_grace=2):
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "2", "MPLBACKEND": "Agg"})
    command = [sys.executable, "-I", "-B", str(checked_file(source_root, "scripts/" + PRODUCER)),
               "--no-estimate", "--output-dir", str(output)]
    with (workdir / "producer.stdout").open("w") as stdout, (workdir / "producer.stderr").open("w") as stderr:
        child = None
        try:
            # Acquire the handle before pending cancellation. A normal-exec
            # wrapper restores the mask then execs the exact producer argv;
            # no Python preexec_fn runs in a possibly threaded forked process.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK,
                {signal.SIGALRM, signal.SIGTERM, signal.SIGINT, signal.SIGHUP})
            try:
                restore_and_exec = ("import json,os,signal,sys; "
                    "signal.pthread_sigmask(signal.SIG_SETMASK,set(json.loads(sys.argv[1]))); "
                    "os.execv(sys.argv[2],sys.argv[2:])")
                launch = [sys.executable, "-I", "-B", "-c", restore_and_exec,
                          json.dumps(sorted(int(number) for number in old_mask)), *command]
                child = subprocess.Popen(launch, cwd=workdir, env=environment, stdout=stdout, stderr=stderr,
                                         start_new_session=True)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            result = child.wait(timeout=timeout)
        except BaseException:
            if child is not None:
                cleanup_owned_group(child, cleanup_grace)
            raise
        else:
            # A completed leader must not leave owned descendants behind.
            cleanup_owned_group(child, cleanup_grace)
    require(result == 0, "Logical producer failed; no verification completed")
    summary = load_json(workdir / "producer.stdout")
    require(summary["physical_points"] == 0, "Producer unexpectedly returned native points")


@contextmanager
def wall_limit(seconds):
    """Bound validation and replay together; run_producer owns child cleanup."""
    current_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    require(signal.SIGALRM not in current_mask, "Cannot enforce a wall deadline with blocked SIGALRM")
    previous = signal.getsignal(signal.SIGALRM)
    termination_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Cannot replace an existing process wall timer")
    def expired(number, frame):
        raise TimeoutError("Energy verifier exceeded its total wall allowance")
    def terminated(number, frame):
        raise TimeoutError("Energy verifier received termination; cleaning up owned replay")
    signal.signal(signal.SIGALRM, expired)
    for number in termination_handlers: signal.signal(number, terminated)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        for number, handler in termination_handlers.items(): signal.signal(number, handler)


def verify_record(records, manifest_path, expected_sha256, *, source_root=ROOT, max_wall_seconds=600):
    require(isinstance(expected_sha256, str) and SHA.fullmatch(expected_sha256), "Reviewed manifest SHA256 required")
    require(isinstance(max_wall_seconds, (int, float)) and not isinstance(max_wall_seconds, bool)
            and math.isfinite(max_wall_seconds) and 0 < max_wall_seconds <= 600, "Wall cap must be within 600 seconds")
    # Reserve time for finite TERM/KILL grace and immutable-byte audits.
    reserve = min(10, max_wall_seconds * .2)
    with wall_limit(max_wall_seconds - reserve):
        return _verify_record(records, manifest_path, expected_sha256, source_root, max_wall_seconds, reserve)


def _verify_record(records, manifest_path, expected_sha256, source_root, max_wall_seconds, reserve):
    started = time.monotonic()
    records, manifest_path = records.absolute(), manifest_path.absolute()
    require(not manifest_path.is_symlink() and manifest_path.is_file(), "Missing/linked manifest")
    require(manifest_path.parent.absolute() == manifest_path.parent.resolve(), "Symlinked manifest ancestor")
    require(sha256(manifest_path) == expected_sha256, "Manifest differs from reviewed immutable identity")
    manifest = load_json(manifest_path)
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
            and manifest.get("status") == "frozen_computed_conditional_energy_record",
            "Missing/pending/unsupported record manifest")
    before = core_inventory(records)
    try:
        require(set(manifest) == {"schema_version", "status", "files", "sources", "verifier_sources",
                "physical_profile", "native_execution", "physical_validation", "contract"},
                "Manifest has missing/unexpected fields")
        require(manifest["files"] == before and set(manifest["files"]) == CORE_FILES, "Recorded artifact identities differ")
        sources = source_hashes(source_root)
        require(manifest["sources"] == sources and set(sources) == set(SOURCES), "Current producer/adapter/lock is stale")
        require(manifest["verifier_sources"] == verifier_hashes(source_root), "Verifier/model audit source differs")
        require(manifest["physical_profile"] == PROFILE, "Frozen physical profile changed")
        require(manifest["contract"] == CONTRACT, "Manifest changes the recorded-model verification contract")
        execution = manifest["native_execution"]
        require(set(execution) == {"producer_commit", "python", "packages", "device", "seed",
                "complete_default_producer", "native_estimation_performed", "exit_status", "wall_seconds", "peak_rss_kib"},
                "Native receipt has missing/unexpected fields")
        require(execution["native_estimation_performed"] is True and execution["complete_default_producer"] is True
                and type(execution["exit_status"]) is int and execution["exit_status"] == 0,
                "Missing/computation-pending native receipt")
        require(isinstance(execution["producer_commit"], str) and COMMIT.fullmatch(execution["producer_commit"])
                and execution["device"] == "CPU" and type(execution["seed"]) is int and execution["seed"] == 20261112,
                "Native source/device/seed receipt differs")
        require(type(execution["wall_seconds"]) in (int, float) and execution["wall_seconds"] > 0
                and type(execution["peak_rss_kib"]) is int and execution["peak_rss_kib"] > 0,
                "Native wall/peak receipt is invalid")
        environment = locked_environment(source_root)
        require(environment["python"] == execution["python"], "Interpreter differs from recorded native study")
        require(isinstance(execution["packages"], dict) and set(execution["packages"]) ==
                {"qiskit", "numpy", "scipy", "matplotlib", "qdk"}
                and all(isinstance(version, str) for version in execution["packages"].values()),
                "Native receipt lacks exact producer package versions")
        for name, version in execution["packages"].items():
            require(importlib.metadata.version(name) == version, f"Recorded package mismatch: {name}")
        ledgers = load_ledgers(records, sources)
        require(all(ledger["environment"]["python"] == execution["python"]
                    and ledger["environment"]["packages"] == execution["packages"]
                    and ledger["environment"]["device"] == execution["device"]
                    and ledger["environment"]["seed"] == execution["seed"]
                    for ledger in ledgers.values()), "Native receipt/ledger environments differ")
        with tempfile.TemporaryDirectory(prefix="openquantum-energy-verify-") as temporary:
            # macOS /var is itself an OS symlink. Use its canonical real path
            # before enforcing the same strict ancestor rule as input roots.
            workdir = Path(temporary).resolve()
            require(not workdir.resolve().is_relative_to(records.resolve()), "Temporary replay overlaps immutable records")
            output = workdir / "logical-replay"
            remaining = max_wall_seconds - reserve - (time.monotonic() - started)
            require(remaining > 0, "Verifier wall allowance exhausted before replay")
            run_producer(source_root, output, workdir, remaining, cleanup_grace=min(2, reserve / 4))
            replay_ledgers = compare_logical_outputs(records, output, ledgers)
            physical = validate_physical_records(records, replay_ledgers, PROFILE)
        require(physical == manifest["physical_validation"], "Frozen physical validation summary changed")
        require(time.monotonic() - started <= max_wall_seconds, "Verifier exceeded its wall allowance")
    finally:
        require(core_inventory(records) == before and sha256(manifest_path) == expected_sha256,
                "Immutable records/manifest changed during verification")
    return {"status": "verified immutable core-energy record", "native_reestimation_performed": False,
            "factory_search_repeated": False, "logical_producer_executed": True,
            "original_core_bytes_preserved": True, "manifest_sha256": expected_sha256,
            "core_files": len(before), "verified_environment": environment,
            "spectral_reference_comparison": {"file": "results.json", "arrays": list(REFERENCE_ARRAYS),
                "absolute_tolerance": REFERENCE_ATOL, "relative_tolerance": 0,
                "all_other_results_and_record_hashes": "all original record hashes exact; other results exact except separately validated 11-wire residual"},
            "small_complete_validation": {"field": SMALL_COMPLETE_RESIDUAL,
                "parameters": dict(SMALL_COMPLETE_PARAMETERS), "absolute_error_upper_bound": SMALL_COMPLETE_ATOL,
                "comparison": "both residuals satisfy the fixed bound; original record bytes remain exact"},
            "wall_seconds": time.monotonic() - started, "physical_validation": physical}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "assets/generated/energy-estimation")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    args = parser.parse_args()
    result = verify_record(args.records_dir, args.manifest or args.records_dir / MANIFEST_NAME,
                           args.manifest_sha256, max_wall_seconds=args.max_wall_seconds)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
