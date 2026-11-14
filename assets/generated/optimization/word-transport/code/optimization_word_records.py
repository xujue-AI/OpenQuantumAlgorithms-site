# SPDX-License-Identifier: MIT
"""Portable authentication and saved revalidation of one QAOA word shuttle.

This module imports only the Python standard library.  Authenticated source
snapshots are compiled from their checked bytes only after the complete public
bundle manifest has passed.  The revalidation path never calls QDK, synthesis,
an optimizer, or the private producer/runner.  Operational clock and permit
evidence remains an explicitly audited private trust boundary.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import zlib

SCHEMA = "openquantum-optimization-word-records-v1"
TRANSCRIPT = "qaoa-cycle-4-p2-alternate-seed-7"
PRODUCER_COMMIT = "12c9d03853a8e5e6e6fd8d532eaaffeb87d47675"
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_DECODED_BYTES = 64 * 1024 * 1024
MAX_CLASS_BYTES = 4 * 1024 * 1024
MAX_FILES = 512
MAX_DIRECTORIES = 1024
MAX_PATH_DEPTH = 32
HEX = re.compile(r"[0-9a-f]{64}\Z")
CLASS_HEX = HEX
STUDY = dict(transcript_id=TRANSCRIPT, algorithm="qaoa", data_qubits=4,
             depth=2, training_groups=109, holdout_groups=1, groups=110,
             classes=54, parents=49, preparations=82032, T_states=182646240)
CODE = {
    "native": "code/optimization_word_native.py",
    "accounting": "code/optimization_word_accounting.py",
    "transport": "code/optimization_word_transport.py",
    "codec": "code/optimization_compiled_codec.py",
    "schema": "code/optimization_word_record_schema.py",
}
FIXED_FILES = frozenset({
    "source-recipe.json", "source-accounting.json", "source-validation.json",
    "native-plan.json", "source-artifact-manifest.json", "execution-summary.json", "record-transformations.json",
    "native-inputs.json.gz", "producer-returns.jsonl.gz", "native-result.json",
    "baseline/manifest.json", "baseline/physical-dag.json.gz", "baseline/recipe.json",
    "baseline/negative-controls.csv", *CODE.values(),
})


class RecordError(ValueError):
    """A public saved record failed authentication or its fixed contract."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fail(message):
    raise RecordError(message)


def _hash(value, label):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        _fail(label + " must be an exact SHA-256")
    return value


def _int(value, label, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        _fail(label + " must retain its bounded integer type")
    return value


def _finite(value, label, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(label + " must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        _fail(label + " exceeds its stated bound")
    return value


def typed_equal(actual, expected):
    """JSON equality that does not accept bool/int or int/float substitution."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(typed_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def strict_json(raw: bytes, *, maximum=MAX_JSON_BYTES):
    if not isinstance(raw, bytes) or len(raw) > maximum:
        _fail("JSON exceeds its decoded byte bound")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("Duplicate JSON object key")
            result[key] = value
        return result

    def number(token):
        value = float(token)
        if not math.isfinite(value):
            _fail("Nonfinite or overflowing JSON number")
        return value

    def integer(token):
        if len(token.lstrip("-")) > 1024:
            _fail("JSON integer exceeds its digit bound")
        return int(token)

    def constant(_token):
        _fail("Nonfinite JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                           parse_int=integer, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as error:
        raise RecordError("Malformed bounded JSON") from error

    def walk(item, depth=0):
        if depth > 128:
            _fail("JSON nesting exceeds its depth bound")
        if isinstance(item, dict):
            if len(item) > 100000:
                _fail("JSON object exceeds its item bound")
            for key, child in item.items():
                if not isinstance(key, str):
                    _fail("JSON object key is not a string")
                walk(child, depth + 1)
        elif isinstance(item, list):
            if len(item) > 100000:
                _fail("JSON array exceeds its item bound")
            for child in item:
                walk(child, depth + 1)
        elif type(item) is float and not math.isfinite(item):
            _fail("JSON has a nonfinite value")
        elif item is not None and type(item) not in (bool, int, float, str):
            _fail("JSON has an unsupported scalar")
    walk(value)
    return value


def relative_path(value):
    if (not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value
            or value.startswith("/") or "\x00" in value):
        _fail("Record path must be a bounded lexical relative path")
    parts = value.split("/")
    if (len(parts) > MAX_PATH_DEPTH or any(part in {"", ".", ".."} for part in parts)
            or PurePosixPath(value).as_posix() != value):
        _fail("Record path is redirected or normalized")
    return parts


def checked_root(value):
    root = Path(os.path.abspath(os.fspath(value)))
    for part in (*reversed(root.parents), root):
        if part.is_symlink():
            _fail("Linked record root or ancestor")
    if not root.is_dir():
        _fail("Record root must be a real directory")
    return root


def checked_path(root, relative):
    current = root
    parts = relative_path(relative)
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            _fail("Linked record path or ancestor")
        if index < len(parts) - 1 and not current.is_dir():
            _fail("Record ancestor is not a directory")
    return current


def read_regular(root, relative, maximum):
    path = checked_path(root, relative)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RecordError("Record file is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            _fail("Record file is linked, nonregular or oversized")
        pieces, remaining = [], maximum + 1
        while remaining:
            piece = os.read(descriptor, min(1024 * 1024, remaining))
            if not piece:
                break
            pieces.append(piece)
            remaining -= len(piece)
        raw = b"".join(pieces)
        after = os.fstat(descriptor)
        if (len(raw) != before.st_size or len(raw) > maximum
                or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)):
            _fail("Record changed while being read")
        final = checked_path(root, relative).stat(follow_symlinks=False)
        if ((final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_nlink)
                != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)):
            _fail("Record path changed while being read")
        return raw
    finally:
        os.close(descriptor)


def decode_gzip(raw, maximum=MAX_DECODED_BYTES):
    try:
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        result = decoder.decompress(raw, maximum + 1)
        if (len(result) > maximum or not decoder.eof or decoder.unconsumed_tail
                or decoder.unused_data):
            _fail("Compressed record is truncated, concatenated or oversized")
        return result
    except zlib.error as error:
        raise RecordError("Malformed compressed record") from error


@dataclass(frozen=True)
class AuthenticatedBundle:
    root: Path
    manifest: dict
    raw: dict
    manifest_sha256: str

    def json(self, name, maximum=MAX_JSON_BYTES):
        return strict_json(self.raw[name], maximum=maximum)


def verify_inventory_tree(root, expected_files, expected_directories):
    """Stream bounded entries and never descend into an unlisted directory."""
    if len(expected_directories) > MAX_DIRECTORIES:
        _fail("Inventory directory count exceeds its bound")
    seen_files, seen_directories = set(), {""}
    visits = 0
    descriptor_walk = os.scandir in os.supports_fd
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))

    def visit(relative, descriptor=None, depth=0):
        nonlocal visits
        if depth > MAX_PATH_DEPTH:
            _fail("Actual inventory depth exceeds its bound")
        path = root if not relative else checked_path(root, relative)
        iterator_input = descriptor if descriptor_walk else path
        with os.scandir(iterator_input) as entries:
            for entry in entries:
                visits += 1
                if visits > MAX_FILES + MAX_DIRECTORIES + 1:
                    _fail("Actual inventory entry count exceeds its bound")
                name = entry.name if not relative else relative + "/" + entry.name
                if entry.is_symlink():
                    _fail("Linked inventory entry")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    if name not in expected_directories:
                        _fail("Public bundle contains an unlisted directory")
                    seen_directories.add(name)
                    child = None
                    try:
                        if descriptor_walk:
                            child = os.open(entry.name, flags, dir_fd=descriptor)
                            opened = os.fstat(child)
                            if (not stat.S_ISDIR(opened.st_mode)
                                    or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)):
                                _fail("Inventory directory changed before enumeration")
                        visit(name, child, depth + 1)
                    finally:
                        if child is not None:
                            os.close(child)
                elif stat.S_ISREG(metadata.st_mode):
                    if name not in expected_files or metadata.st_nlink != 1:
                        _fail("Public bundle contains an unlisted or linked file")
                    seen_files.add(name)
                else:
                    _fail("Public inventory contains a nonregular file")
    descriptor = None
    try:
        if descriptor_walk:
            descriptor = os.open(root, flags)
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                _fail("Inventory root changed before enumeration")
        visit("", descriptor)
    except OSError as error:
        raise RecordError("Inventory entry changed or cannot be enumerated") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if seen_files != expected_files or seen_directories != expected_directories:
        _fail("Public bundle contains omitted files or directories")


def authenticate_bundle(bundle, expected_manifest_sha256):
    """Authenticate the entire immutable inventory without compiling sources."""
    root = checked_root(bundle)
    expected = _hash(expected_manifest_sha256, "Trusted manifest pin")
    manifest_raw = read_regular(root, "record-manifest.json", MAX_JSON_BYTES)
    if digest(manifest_raw) != expected:
        _fail("Immutable public record manifest changed")
    manifest = strict_json(manifest_raw)
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("status") != "completed_fixed_native_resource_model"):
        _fail("Public record manifest status/schema changed")
    study = manifest.get("study")
    if not isinstance(study, dict) or any(key not in study or not typed_equal(study[key], value) for key, value in STUDY.items()):
        _fail("The manifest is not the one frozen C4/depth-two transcript")
    if manifest.get("producer_source_commit") != PRODUCER_COMMIT:
        _fail("Recorded Native producer source changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES or "record-manifest.json" in files:
        _fail("Immutable inventory is absent or malformed")
    if not FIXED_FILES <= files.keys():
        _fail("Immutable inventory omitted a required source/scientific file")
    pins = manifest.get("source_pins")
    if not isinstance(pins, dict) or not set(CODE.values()) <= pins.keys():
        _fail("Immutable manifest omitted a compiled source pin")
    for name, pin in pins.items():
        relative_path(name)
        if name not in files or not name.endswith(".py") or pin != files[name].get("sha256"):
            _fail("Compiled source pin disagrees with the authenticated inventory")
    raw, total = {}, len(manifest_raw)
    directories = {""}
    for relative, entry in files.items():
        parts = relative_path(relative)
        for length in range(1, len(parts)):
            directories.add("/".join(parts[:length]))
        if len(directories) > MAX_DIRECTORIES:
            _fail("Inventory directory count exceeds its bound")
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
            _fail("Immutable file descriptor changed")
        size = _int(entry["bytes"], "Record byte count", 1, MAX_BUNDLE_BYTES)
        _hash(entry["sha256"], "Record content hash")
        maximum = MAX_BUNDLE_BYTES if relative.endswith(".gz") else MAX_JSON_BYTES
        content = read_regular(root, relative, maximum)
        if len(content) != size or digest(content) != entry["sha256"]:
            _fail("An immutable scientific/source file changed")
        total += len(content)
        if total > MAX_BUNDLE_BYTES:
            _fail("Complete public bundle exceeded 16 MiB")
        raw[relative] = content
    verify_inventory_tree(root, set(files) | {"record-manifest.json"}, directories)
    return AuthenticatedBundle(root, manifest, raw, expected)


def _module(bundle, role):
    relative = CODE[role]
    raw = bundle.raw[relative]
    name = "_oq_word_saved_" + role + "_" + digest(raw)[:20]
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(bundle.root / relative))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(bundle.root / relative)
    sys.modules[name] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def _bind_cached_native_source_reads(bundle, native):
    """Keep historical hash guards while making saved-only source reads bounded.

    Only this freshly compiled verification snapshot is changed in memory.
    Production files and execution guards are untouched.  The historical
    verifier's own-source/accounting reads receive the exact bytes already
    authenticated by the complete manifest, under its original hash checks.
    No arbitrary disk path, import cache or changed on-disk source is used.
    """
    permitted = {bundle.root / CODE[role]: bundle.raw[CODE[role]] for role in ("native", "accounting")}

    def regular(path, expected=None):
        requested = Path(path)
        if not requested.is_absolute() or requested not in permitted or ".." in requested.parts:
            _fail("Saved Native verifier requested an unauthenticated source path")
        content = permitted[requested]
        if len(content) > MAX_JSON_BYTES:
            _fail("Cached verification source exceeded its original read bound")
        if expected is not None and digest(content) != expected:
            _fail("Cached verification source changed its original hash guard")
        return content

    native._regular = regular


def _tagged_types(value):
    """Enforce the actual Python scalar type behind saved Native type tags."""
    if isinstance(value, dict):
        kind = value.get("type")
        if kind == "int" and type(value.get("value")) is not int:
            _fail("Saved Native integer tag hides a noninteger value")
        if kind == "float":
            token = value.get("hex")
            if not isinstance(token, str):
                _fail("Saved Native float tag omitted its exact bits")
            try:
                parsed = float.fromhex(token)
            except ValueError as error:
                raise RecordError("Saved Native float bits are malformed") from error
            if not math.isfinite(parsed):
                _fail("Saved Native float tag is nonfinite")
        if "integer" in value and type(value["integer"]) is not int:
            _fail("Saved ISA integer field changed scalar type")
        if "float_hex" in value:
            token = value["float_hex"]
            try:
                parsed = float.fromhex(token) if isinstance(token, str) else float("nan")
            except (ValueError, OverflowError) as error:
                raise RecordError("Saved ISA float field is malformed") from error
            if not math.isfinite(parsed):
                _fail("Saved ISA float field is nonfinite or malformed")
        for child in value.values():
            _tagged_types(child)
    elif isinstance(value, list):
        for child in value:
            _tagged_types(child)


def verify_trace(request, enriched):
    """Compare full Trace semantics, rather than trusting a matching digest."""
    trace = strict_json(enriched["native_trace_json"].encode(), maximum=MAX_DECODED_BYTES)
    expected = dict(
        block=dict(operations=[dict(BlockOperation=dict(
            operations=[dict(GateOperation=dict(id=row["instruction_id"],
                       qubits=list(range(15)), params=[]))], repetitions=row["count"]))
            for row in request["macro_ledger_per_preparation"]], repetitions=1),
        base_error=float.fromhex(request["native_base_error_hex"]), compute_qubits=15,
        memory_qubits=None, resource_states={"1028": request["t_states_per_preparation"]}, properties={})
    if not typed_equal(trace, expected):
        _fail("Native Trace structure, counts or scalar types changed")


def verify_raw_types(raw, enriched):
    """Close Python-equality loopholes in the historical saved validator."""
    _tagged_types(raw)
    if not typed_equal(raw["native_isa_cost_records"], enriched["native_isa_cost_records"]):
        _fail("Returned ISA costs changed exact scalar types or values")
    properties = raw["properties"]
    expected = {8: 6735, 13: 0, 12: 15, 9: 236623}
    observed = {}
    for row in properties:
        if set(row) != {"key", "value"}:
            _fail("Returned property record has unexpected fields")
        key, value = row["key"], row["value"]
        if (not isinstance(key, dict) or not isinstance(value, dict)
                or key.get("type") != "int" or value.get("type") != "int"
                or type(key.get("value")) is not int or type(value.get("value")) is not int
                or key["value"] in observed):
            _fail("Returned properties lost unique exact integer fields")
        observed[key["value"]] = value["value"]
    if observed != expected:
        _fail("Returned compute/memory/factory property values changed")


def _execution_summary(bundle):
    summary = bundle.json("execution-summary.json")
    if (summary.get("schema") != "openquantum-word-native-execution-summary-v1"
            or summary.get("status") != "completed_local_native_resource_model"
            or summary.get("transcript_id") != TRANSCRIPT
            or summary.get("source_commit") != PRODUCER_COMMIT
            or summary.get("hardware_execution") is not False
            or summary.get("new_optimizer") is not False or summary.get("new_synthesis") is not False):
        _fail("Audited execution summary changed its scope/status")
    execution = summary["execution"]
    for key, expected in dict(classes=54, parents=49, preparations=82032, T_states=182646240,
                              trace_calls=54, worker_returncode=0, factory_frontier_queries=0,
                              pipeline_candidates=0, synthesis_calls=0).items():
        if type(execution.get(key)) is not int or execution[key] != expected:
            _fail("Audited execution lost exact class/call/work coverage")
    if execution.get("cleanup_complete") is not True:
        _fail("Audited cleanup attestation changed")
    for name in ("internal_parent_wall_seconds", "external_process_wall_seconds", "post_export_guardian_wall_seconds"):
        _finite(execution[name], "Audited wall time", 0, 60)
    saved = summary["saved_verification"]
    if saved.get("status") != "verified_fixed_native_result" or type(saved.get("exit_code")) is not int or saved["exit_code"] != 0 or type(saved.get("new_native_calls")) is not int or saved["new_native_calls"] != 0:
        _fail("Saved verification attestation changed")
    if "one_use_state_summary" in summary or "retries_permitted" in execution:
        _fail("Private operational state was reintroduced into public science data")
    expected_operational = {"attempt-claim.json", "attempt-journal.jsonl", "completion-seal.json",
        "guardian-config.json", "guardian-receipt.json", "parent-receipt.json", "worker-config.json", "worker-pid.json"}
    hashes = summary["original_operational_file_sha256"]
    if set(hashes) != expected_operational:
        _fail("Audited operational hash inventory changed")
    for value in hashes.values():
        _hash(value, "Private operational evidence hash")
    for name in ("native-inputs.json.gz", "producer-returns.jsonl.gz", "native-result.json"):
        if summary["original_scientific_file_sha256"].get(name) != digest(bundle.raw[name]):
            _fail("Public Native preimage is not the exact captured scientific file")
    return summary


def _recorded_source_checks(validation, classes):
    checks = validation.get("class_checks")
    if (not isinstance(checks, list) or len(checks) != 54
            or [row["class_id"] for row in checks] != [row["class_id"] for row in classes]
            or [row["shots"] for row in checks] != [row["shots"] for row in classes]
            or validation.get("classes") != 54 or validation.get("groups") != 110
            or validation.get("preparations") != 82032):
        _fail("Recorded full-source checks lost transcript/class coverage")
    for row in checks:
        _int(row["num_qubits"], "Recorded data wires", 4, 4)
        tolerance = _finite(row["spectral_tolerance"], "Recorded operator tolerance", 0, 2e-11)
        _finite(row["operator_spectral_error"], "Recorded operator residual", 0, tolerance)
        _finite(row["joint_z_maximum_probability_error"], "Recorded joint-readout residual", 0, tolerance)
        if (row.get("global_phase_fitting") is not False
                or row.get("global_phase_metadata_bit_identical") is not True
                or row.get("structural_word_boundaries_verified") is not True
                or row.get("event_subsequence_preserved") is not True):
            _fail("Recorded source checks omit phase/word/event coverage")
    density = validation["density_checks"]
    _int(density["case_count"], "Recorded density cases", 1, 32)
    _int(density["maximum_qubits"], "Recorded density wires", 1, 6)
    _int(density["maximum_micro_operations"], "Recorded density operations", 1, 64)
    tolerance = _finite(density["spectral_tolerance"], "Recorded density tolerance", 0, 2e-11)
    _finite(density["maximum_spectral_error"], "Recorded density residual", 0, tolerance)
    _finite(density["maximum_branch_probability_error"], "Recorded branch residual", 0, tolerance)


def _source_pairs(bundle, recipe, baseline_manifest, transport, native):
    entries = {entry["class_id"]: entry for entry in baseline_manifest["classes"]}
    classes = recipe["classes"]
    if len(entries) != 54 or set(entries) != {row["class_id"] for row in classes}:
        _fail("Original source manifest changed the 54-class identity set")
    counted = Counter()
    for planned in classes:
        key = planned["class_id"]
        if not CLASS_HEX.fullmatch(key):
            _fail("Class identifier is not a saved SHA-256")
        entry = entries[key]
        checkpoint_path = "baseline/" + entry["checkpoint"]
        packed = bundle.raw[checkpoint_path]
        if digest(packed) != planned["checkpoint_sha256"]:
            _fail("Original source checkpoint changed")
        checkpoint = strict_json(decode_gzip(packed, MAX_CLASS_BYTES), maximum=MAX_CLASS_BYTES)
        original = checkpoint["record"]
        if (checkpoint["key"] != key or original["class_id"] != key
                or not typed_equal(checkpoint["identity"], baseline_manifest["original_identity"])
                or type(original["shots"]) is not int or original["shots"] != planned["shots"]
                or not typed_equal(original["batches"], planned["batches"])
                or not typed_equal(original["parameters"], planned["parameters"])
                or type(original["clifford_t_global_phase"]) is not float
                or original["clifford_t_global_phase"].hex() != planned["global_phase_hex"]
                or digest(original["clifford_t_qasm"].encode()) != planned["source_qasm_sha256"]):
            _fail("Original source/checkpoint phase, batches or QASM changed")
        packed_candidate = bundle.raw[f"candidate/circuits/{key}.json.gz"]
        candidate = strict_json(decode_gzip(packed_candidate, MAX_CLASS_BYTES), maximum=MAX_CLASS_BYTES)
        expected = transport.build_candidate(original["clifford_t_operations"])
        if (candidate["class_id"] != key or not typed_equal(candidate["operations"], expected["operations"])
                or not typed_equal(candidate["segments"], expected["segments"])
                or type(candidate["exact_global_phase"]) is not float
                or candidate["exact_global_phase"].hex() != planned["global_phase_hex"]
                or candidate["source_qasm_sha256"] != planned["source_qasm_sha256"]
                or transport.operation_digest(candidate["operations"]) != planned["candidate_operations_sha256"]
                or digest((json.dumps(candidate["segments"], sort_keys=True, indent=2,
                                      allow_nan=False) + "\n").encode()) != planned["segments_sha256"]):
            _fail("Candidate exact words, SWAPs, phases or source events changed")
        qasm = bundle.raw[f"candidate/circuits/{key}.qasm"]
        if digest(qasm) != planned["candidate_qasm_sha256"] or native._qasm(candidate["operations"]) != qasm:
            _fail("Candidate QASM no longer matches its full primitive sequence")
        if (sorted(expected["reset_wire_order"]) != [0, 1, 2, 3]
                or canonical(expected["measurement_map"]) != canonical([[0, 0], [3, 1], [2, 2], [1, 3]])):
            _fail("Candidate changed initial data order or the routed classical readout map")
        counted.update({"selected_words": sum(segment["shuttled"] for segment in expected["segments"]),
                        "added_cx": expected["added_data_transport_cx"],
                        "source_operations": len(original["clifford_t_operations"]),
                        "candidate_operations": len(expected["operations"])})
    return counted


def _verify_records(bundle, expected_manifest_sha256, *, replay=True):
    """Verify one immutable bundle, optionally rederiving its saved models."""
    if type(replay) is not bool:
        _fail("Replay choice must be a boolean")
    start = time.perf_counter()
    authenticated = authenticate_bundle(bundle, expected_manifest_sha256)
    summary = _execution_summary(authenticated)
    if not replay:
        return dict(status="verified_word_record_identity", transcript_id=TRANSCRIPT,
                    classes=54, parents=49, preparations=82032, t_states=182646240,
                    native_api_calls=0, writes_performed=False,
                    verification_scope="immutable file/source identities and audited execution-summary boundaries; no saved model rederivation",
                    elapsed_seconds=time.perf_counter() - start)
    schema = _module(authenticated, "schema")
    wrappers = {name: authenticated.json(path) for name, path in {
        "plan": "native-plan.json", "recipe": "source-recipe.json",
        "accounting": "source-accounting.json", "validation": "source-validation.json",
        "artifact_manifest": "source-artifact-manifest.json"}.items()}
    restored = {name: schema.restore_record(value) for name, value in wrappers.items()}
    plan, recipe, accounting, validation = (restored[name] for name in ("plan", "recipe", "accounting", "validation"))
    if (not typed_equal(plan["frozen_source_recipe"], recipe)
            or not typed_equal(plan["frozen_source_accounting"], accounting)
            or plan["plan_sha256"] != summary["original_plan_logical_sha256"]
            or wrappers["plan"]["original_sha256"] != summary["original_plan_file_sha256"]):
        _fail("Restored scientific records disagree with their original exact plan")
    native, transport, codec = (_module(authenticated, name) for name in ("native", "transport", "codec"))
    _bind_cached_native_source_reads(authenticated, native)
    if (wrappers["recipe"]["original_sha256"] != native.RECIPE_SHA256
            or wrappers["accounting"]["original_sha256"] != native.ACCOUNTING_SHA256
            or wrappers["validation"]["original_sha256"] != native.VALIDATION_SHA256
            or wrappers["artifact_manifest"]["original_sha256"] != native.MANIFEST_SHA256):
        _fail("Historical source/validation records changed their original hash anchors")
    original_files = restored["artifact_manifest"]["files"]
    expected_source_files = {f"circuits/{row['class_id']}.{suffix}" for row in recipe["classes"]
                             for suffix in ("qasm", "json.gz")} | {"validation.json", "accounting.json"}
    if set(original_files) != expected_source_files:
        _fail("Historical source manifest changed its exact 110 scientific files")
    for name, descriptor in original_files.items():
        if name.startswith("circuits/"):
            content = authenticated.raw["candidate/" + name]
        else:
            kind = "validation" if name == "validation.json" else "accounting"
            content = (json.dumps(restored[kind], sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if len(content) != descriptor["bytes"] or digest(content) != descriptor["sha256"]:
            _fail("A historical exact scientific preimage changed")
    native._bound_plan(plan, check_runtime=False)
    if (not typed_equal(plan["fixed_machine"], authenticated.manifest["fixed_machine"])
            or plan["fixed_machine"]["peak_physical_qubits"] != 243358):
        _fail("The one frozen common machine changed")
    baseline_recipe = authenticated.json("baseline/recipe.json")
    expected_planned = dict(training_calls=109, holdout_calls=1, measured_groups=110,
        unique_source_classes=54, training_preparations=65648, holdout_preparations=16384,
        total_preparations=82032)
    if (baseline_recipe["run_id"] != TRANSCRIPT or baseline_recipe["case_id"] != "qaoa-cycle-4-p2"
            or baseline_recipe["output"]["n_qubits"] != 4
            or any(not typed_equal(baseline_recipe["planned"].get(key), value) for key, value in expected_planned.items())
            or not typed_equal(baseline_recipe["budget"], plan["unchanged_budget"])):
        _fail("The matched baseline is not the same fixed C4 transcript/task/budget")
    baseline_manifest = authenticated.json("baseline/manifest.json")
    if digest(authenticated.raw["baseline/manifest.json"]) != recipe["baseline_manifest_sha256"]:
        _fail("Original compiled source manifest changed")
    counts = _source_pairs(authenticated, recipe, baseline_manifest, transport, native)
    _recorded_source_checks(validation, recipe["classes"])
    original_dag = authenticated.raw["baseline/physical-dag.json.gz"]
    if digest(original_dag) != plan["baseline_physical_dag_sha256"]:
        _fail("Original selected physical record changed")
    physical = codec.loads(decode_gzip(original_dag))
    baseline = physical["selected_minimum_runtime"]
    if (len(physical["physical_classes"]) != 49
            or baseline["worst_case_serial_runtime_ns"] != 797328306784000
            or baseline["peak_physical_qubits"] != 243358
            or baseline["data_code_distance"] != 15 or baseline["global_factory_retry_cap"] != 6
            or baseline["total_preparations"] != 82032 or baseline["total_t_states"] != 182646240):
        _fail("Baseline selected machine/workload changed")
    enriched = strict_json(decode_gzip(authenticated.raw["native-inputs.json.gz"]), maximum=MAX_DECODED_BYTES)
    decoded_returns = decode_gzip(authenticated.raw["producer-returns.jsonl.gz"])
    if not decoded_returns.endswith(b"\n"):
        _fail("Raw Native return transport is truncated")
    rows = [strict_json(line, maximum=MAX_CLASS_BYTES) for line in decoded_returns.splitlines()]
    result = authenticated.json("native-result.json")
    ids = [request["class_id"] for request in plan["requests"]]
    if (not isinstance(enriched, list) or len(enriched) != 54 or len(rows) != 54
            or [row["class_id"] for row in enriched] != ids or [row["class_id"] for row in rows] != ids):
        _fail("Full Native input/return preimages lost exact ordered class coverage")
    for request, value, row in zip(plan["requests"], enriched, rows):
        _tagged_types(value)
        _tagged_types(row)
        verify_trace(request, value)
        verify_raw_types(row["producer_record"], value)
    replayed = native.validate_saved_execution(plan, [row["producer_record"] for row in rows], enriched, result)
    for key in ("permitted", "confirmed_started", "returned", "validated"):
        if type(result["audit"].get(key)) is not int or result["audit"][key] != 54:
            _fail("Recorded Native audit lacks all 54 exact calls/results")
    for key in ("factory_frontier_queries", "pipeline_candidates", "synthesis_calls"):
        if type(result.get(key)) is not int or result[key] != 0:
            _fail("Excluded Native query counter lost its integer-zero type")
    model = summary["conditional_model"]
    if (replayed["serial_runtime_ns"] != 602466084947200
            or replayed["serial_runtime_ns"] != model["serial_runtime_ns"]
            or replayed["whole_physical_failure_exact"] != model["whole_physical_failure_exact"]
            or replayed["factory_availability_failure_exact"] != model["factory_availability_failure_exact"]
            or baseline["worst_case_serial_runtime_ns"] != model["baseline_serial_model_time_ns"]
            or Fraction(baseline["whole_factory_availability_failure_bound_exact"]) != Fraction(replayed["factory_availability_failure_exact"])):
        _fail("Actual saved candidate/baseline aggregate accounting changed")
    if (plan["totals"]["data_exposure_cycles"] != 602466111572
            or counts["selected_words"] != 744 or counts["added_cx"] != 7020
            or counts["source_operations"] != 306508 or counts["candidate_operations"] != 313528):
        _fail("Source rewrite or per-preparation-ceiled exposure totals changed")
    ratio = Fraction(replayed["serial_runtime_ns"], baseline["worst_case_serial_runtime_ns"])
    reduction = 1 - ratio
    if str(ratio) != model["model_runtime_ratio_exact"] or str(reduction) != model["model_runtime_reduction_fraction_exact"]:
        _fail("The matched runtime/spacetime comparison changed")
    if not typed_equal(summary["factory_result_metadata_scope"], native.factory_metadata_scope(copies_validated=True)):
        _fail("Factory metadata was relabeled as semantically validated")
    again = authenticate_bundle(authenticated.root, expected_manifest_sha256)
    if any(again.raw[name] != value for name, value in authenticated.raw.items()):
        _fail("Record bundle changed during saved revalidation")
    return dict(status="verified_saved_word_native_model", transcript_id=TRANSCRIPT,
        classes=54, parents=49, groups=110, preparations=82032, t_states=182646240,
        serial_runtime_ns=replayed["serial_runtime_ns"], baseline_serial_runtime_ns=baseline["worst_case_serial_runtime_ns"],
        peak_physical_qubits=243358, code_distance=15, retry_cap=6,
        whole_physical_failure_exact=replayed["whole_physical_failure_exact"],
        baseline_physical_failure_exact=baseline["whole_physical_failure_model_exact_binary_rational_sum"],
        factory_availability_failure_exact=replayed["factory_availability_failure_exact"],
        model_runtime_ratio_exact=str(ratio), model_runtime_reduction_fraction_exact=str(reduction),
        spacetime_qubit_ns=243358*replayed["serial_runtime_ns"],
        baseline_spacetime_qubit_ns=243358*baseline["worst_case_serial_runtime_ns"],
        selected_words=counts["selected_words"], added_distinct_cx=counts["added_cx"],
        factory_result_metadata_scope=native.factory_metadata_scope(copies_validated=True),
        native_api_calls=0, writes_performed=False, hardware_execution=False,
        verification_scope="exact saved scientific preimages, source word/QASM reconstruction and conditional Native/parent/whole-workload accounting; private clock/permit/cleanup represented by audited hashes",
        logical_validation_scope="hash-bound recorded full-operator/density checks; no new simulation or kernel proof",
        original_pipeline_layout_equivalence=False, global_frontier_claim=False,
        elapsed_seconds=time.perf_counter() - start)


def verify_records(bundle, expected_manifest_sha256, *, replay=True):
    """Fail with one bounded public error class for malformed saved records."""
    try:
        return _verify_records(bundle, expected_manifest_sha256, replay=replay)
    except RecordError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError, ValueError, OverflowError,
            RecursionError, OSError) as error:
        raise RecordError("Saved word-transport record is malformed or inconsistent") from error
