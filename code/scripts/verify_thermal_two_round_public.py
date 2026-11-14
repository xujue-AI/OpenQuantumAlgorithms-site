#!/usr/bin/env python3
"""Authenticate frozen public thermal bytes without scientific replay.

Only manifest metadata is parsed. Results, QASM, scientific source copies and
all other payloads are raw bytes. This gate does not run the separate metadata
display, a scientific reader, a simulator, a notebook or an experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import time

BUNDLE_RELATIVE = "assets/generated/simulation/detailed-balance-two-round"
SCOPE = "fixed one-system two-round finite composition; no Gibbs/mixing/physical/formal guarantee"
MANIFEST_SCHEMA = "openquantum.fixed-two-round.public-manifest.v1"
MANIFEST_BYTES = 3497
MANIFEST_SHA256 = "e2f3c836fa05c6a0ae6be5daa18684a6b9ec96c6033f5bd34420a2413fe54377"
SIDECAR_BYTES = 80
SIDECAR_SHA256 = "f3bfd5181075899ac7fe182716c1546ab90e9c8058a382207c27338df34aa4ed"
PAYLOAD_BYTES = 1559231
CORE_BYTES = 1562808
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_TREE_ENTRIES = 64
MAX_WALL_SECONDS = 10
MAX_OUTPUT_BYTES = 16 * 1024
ACCEPTED_SOURCE_COMMIT = "02c6bce80784b249927223b08b39174396d6e184"
PAYLOAD_NAMES = frozenset({
    "LICENSE.txt", "README.md", "UPSTREAM-LICENSES.md", "requirements-lock.txt",
    "results.json", "summary.json", "two-round-circuit.svg",
    "circuits/correction-heralded-step.qasm", "circuits/correction_lcu.qasm",
    "circuits/round_macro.qasm", "circuits/two_round_trial.qasm",
    "circuits/weak_dilation.qasm", "circuits/weak_reset_prefix.qasm",
    "sources/detailed_balance_access.py", "sources/detailed_balance_channel.py",
    "sources/detailed_balance_correction.py", "sources/detailed_balance_decay.py",
    "sources/detailed_balance_thermal_two_round.py",
    "sources/detailed_balance_thermal_two_round_reference.py",
    "sources/thermal_two_round_contract.py",
})
CORE_NAMES = PAYLOAD_NAMES | {"manifest.json", "manifest.sha256"}
DIRECTORY_NAMES = frozenset({"circuits", "sources"})
INACTIVE_FLAGS = {
    "scientific_work_executed": False,
    "scientific_work_reexecuted": False,
    "scientific_acceptance_recomputed": False,
    "reader_executed": False,
    "scientific_reader_executed": False,
    "metadata_display_executed": False,
    "numerical_work_executed": False,
    "numerical_replay_performed": False,
    "replay_performed": False,
    "notebook_executed": False,
    "kernel_started": False,
    "bundled_sources_imported": False,
    "bundled_sources_compiled": False,
}


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved-byte authentication exceeded its wall allowance")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _regular(info):
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex_digest(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _platform():
    if (not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW")
            or os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd
            or not hasattr(signal, "pthread_sigmask") or not hasattr(signal, "SIGALRM")):
        raise ValueError("Saved-byte authentication requires no-follow directory descriptors")


def _close_handles(descriptors):
    """Attempt every owned close before restoring a pending cancellation."""
    cleanup_signals = {getattr(signal, name) for name in
                       ("SIGALRM", "SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name)}
    old_mask, first_error = None, None
    try:
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, cleanup_signals)
    except BaseException as error:
        first_error = error
    try:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except BaseException as error:
                if first_error is None:
                    first_error = error
    finally:
        if old_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            except BaseException as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error


def _json_metadata(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate manifest metadata key")
            result[key] = value
        return result
    def bad_constant(_value):
        raise ValueError("Nonfinite manifest metadata constant")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=bad_constant)


def _manifest(raw):
    if len(raw) != MANIFEST_BYTES or _digest(raw) != MANIFEST_SHA256:
        raise ValueError("Frozen thermal manifest identity differs")
    document = _json_metadata(raw)
    keys = {"schema", "accepted_source_commit", "scope", "raw_flags_unchanged",
            "original_accepted_payloads", "summary_path", "scientific_result_path",
            "files", "file_count", "selected_payload_bytes"}
    if (type(document) is not dict or set(document) != keys
            or document["schema"] != MANIFEST_SCHEMA
            or document["accepted_source_commit"] != ACCEPTED_SOURCE_COMMIT
            or document["scope"] != SCOPE or document["raw_flags_unchanged"] is not True
            or document["original_accepted_payloads"] != {"files": 23, "bytes": 1755875}
            or document["summary_path"] != "summary.json"
            or document["scientific_result_path"] != "results.json"
            or type(document["file_count"]) is not int or document["file_count"] != 20
            or type(document["selected_payload_bytes"]) is not int
            or document["selected_payload_bytes"] != PAYLOAD_BYTES
            or type(document["files"]) is not dict
            or set(document["files"]) != PAYLOAD_NAMES):
        raise ValueError("Frozen thermal manifest contract differs")
    total = 0
    for record in document["files"].values():
        if (type(record) is not dict or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int
                or not 0 < record["bytes"] <= MAX_TOTAL_BYTES
                or not _hex_digest(record["sha256"])):
            raise ValueError("Thermal payload binding is not a bounded byte identity")
        total += record["bytes"]
    if total != PAYLOAD_BYTES or total + MANIFEST_BYTES + SIDECAR_BYTES != CORE_BYTES:
        raise ValueError("Frozen thermal byte totals differ")
    return document["files"]


def _provenance(provenance, omitted_assets):
    if provenance is None:
        if omitted_assets is not None:
            raise ValueError("Canonical preview joins require complete package provenance")
        return None
    if (type(provenance) is not dict
            or provenance.get("schema") != "openquantum-public-package-v2"
            or type(provenance.get("file_sha256")) is not dict):
        raise ValueError("Complete thermal package provenance is required")
    if omitted_assets is not None and type(omitted_assets) is not dict:
        raise ValueError("Canonical thermal preview joins must be a mapping")
    return provenance["file_sha256"]


def _join(name, digest, hashes, omitted_assets):
    canonical = BUNDLE_RELATIVE + "/" + name
    if hashes is not None and hashes.get(canonical) != digest:
        raise ValueError("Frozen thermal package provenance differs: " + name)
    if omitted_assets is not None:
        record = omitted_assets.get(canonical)
        if (type(record) is not dict or set(record) != {"site_path", "sha256"}
                or record["site_path"] != canonical or record["sha256"] != digest):
            raise ValueError("Canonical thermal preview join differs: " + name)


def _snapshot_tree(root, directories, deadline):
    """Inspect only the two admitted directory descriptors and exact filenames."""
    files, found_directories, entries = {}, set(), 0
    for directory_name, descriptor in [("", root), *sorted(directories.items())]:
        _deadline(deadline)
        with os.scandir(descriptor) as children:
            for child in children:
                _deadline(deadline)
                entries += 1
                if entries > MAX_TREE_ENTRIES:
                    raise ValueError("Thermal bundle exceeds its closed tree bound")
                relative = (directory_name + "/" if directory_name else "") + child.name
                info = child.stat(follow_symlinks=False)
                if not directory_name and child.name in DIRECTORY_NAMES:
                    if (not stat.S_ISDIR(info.st_mode)
                            or _identity(info) != _identity(os.fstat(directories[child.name]))):
                        raise ValueError("Thermal directory is linked or changed")
                    found_directories.add(child.name)
                elif (relative in CORE_NAMES and _regular(info)
                      and 0 < info.st_size <= MAX_TOTAL_BYTES):
                    files[relative] = _identity(info)
                else:
                    raise ValueError("Unlisted, linked, nonregular or oversized thermal entry")
    if set(files) != CORE_NAMES or found_directories != DIRECTORY_NAMES:
        raise ValueError("Thermal bundle tree is incomplete")
    return files


def _read_file(directory, name, maximum, deadline):
    _deadline(deadline)
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if not _regular(before) or not 0 < before.st_size <= maximum:
        raise ValueError("Thermal file is linked, nonregular or exceeds its byte bound")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(name, flags, dir_fd=directory)
    try:
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise ValueError("Thermal file changed before open")
        raw = bytearray()
        while len(raw) <= maximum:
            _deadline(deadline)
            chunk = os.read(descriptor, min(65536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (len(raw) != before.st_size or len(raw) > maximum
                or _identity(os.fstat(descriptor)) != _identity(before)
                or _identity(after) != _identity(before)):
            raise ValueError("Thermal file changed during read")
        _deadline(deadline)
        return bytes(raw)
    finally:
        _close_handles([descriptor])


def verify_full_bundle(bundle, *, provenance=None, omitted_assets=None):
    """Authenticate the frozen 22-file bundle and optional verified caller joins.

    A supplied provenance is joined, not independently certified here: package
    callers must first verify their complete manifest; served-site callers must
    first authenticate their source preview and retained helper/provenance.
    This function parses only the frozen inventory manifest, never results JSON.
    """
    _platform()
    deadline = time.monotonic() + MAX_WALL_SECONDS
    hashes = _provenance(provenance, omitted_assets)
    path = Path(bundle).absolute()
    if path != path.resolve(strict=True):
        raise ValueError("Thermal bundle must have an unaliased canonical path")
    before = path.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("Thermal bundle must be a regular directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    root = os.open(path, flags)
    directories = {}
    try:
        if _identity(os.fstat(root)) != _identity(before):
            raise ValueError("Thermal bundle changed before open")
        for name in sorted(DIRECTORY_NAMES):
            _deadline(deadline)
            info = os.stat(name, dir_fd=root, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Thermal payload directory must be regular and unlinked")
            descriptor = os.open(name, flags, dir_fd=root)
            directories[name] = descriptor
            if _identity(os.fstat(descriptor)) != _identity(info):
                raise ValueError("Thermal payload directory changed before open")
        directory_identities = {name: _identity(os.fstat(fd)) for name, fd in directories.items()}
        original_tree = _snapshot_tree(root, directories, deadline)
        manifest_raw = _read_file(root, "manifest.json", MAX_MANIFEST_BYTES, deadline)
        records = _manifest(manifest_raw)
        _join("manifest.json", MANIFEST_SHA256, hashes, omitted_assets)
        sidecar = _read_file(root, "manifest.sha256", SIDECAR_BYTES, deadline)
        if (len(sidecar) != SIDECAR_BYTES or _digest(sidecar) != SIDECAR_SHA256
                or sidecar != (MANIFEST_SHA256 + "  manifest.json\n").encode("ascii")):
            raise ValueError("Frozen thermal manifest sidecar differs")
        _join("manifest.sha256", SIDECAR_SHA256, hashes, omitted_assets)
        total = len(manifest_raw) + len(sidecar)
        for name in sorted(records):
            _deadline(deadline)
            parts = name.split("/")
            descriptor = root if len(parts) == 1 else directories[parts[0]]
            record = records[name]
            raw = _read_file(descriptor, parts[-1], record["bytes"], deadline)
            total += len(raw)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Thermal bundle exceeds its aggregate byte bound")
            digest = _digest(raw)
            if len(raw) != record["bytes"] or digest != record["sha256"]:
                raise ValueError("Frozen thermal payload identity differs: " + name)
            _join(name, digest, hashes, omitted_assets)
        if (total != CORE_BYTES or original_tree != _snapshot_tree(root, directories, deadline)
                or any(_identity(os.fstat(fd)) != directory_identities[name]
                       for name, fd in directories.items())
                or _identity(os.fstat(root)) != _identity(before)
                or _identity(path.lstat()) != _identity(before)):
            raise ValueError("Thermal bundle changed during full-byte authentication")
        _deadline(deadline)
        return {
            "status": "saved_thermal_two_round_full_byte_authenticated",
            "scope": SCOPE,
            "bundle": BUNDLE_RELATIVE,
            "full_byte_authenticated": True,
            "files": 22,
            "payload_files": 20,
            "bytes": total,
            "manifest_sha256": MANIFEST_SHA256,
            "provenance_joined": hashes is not None,
            "canonical_site_joined": omitted_assets is not None,
            **INACTIVE_FLAGS,
        }
    finally:
        _close_handles([*directories.values(), root])


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError("Invalid saved-byte gate arguments")


def main():
    old_handler, installed = None, False
    cli_deadline = time.monotonic() + MAX_WALL_SECONDS
    try:
        try:
            if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
                raise ValueError("The saved-byte CLI requires a bounded POSIX wall timer")
            if (signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)
                    or signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, set())):
                raise ValueError("The saved-byte CLI cannot replace or bypass a wall timer")
            def expired(_number, _frame):
                raise TimeoutError("Saved-byte CLI exceeded its wall allowance")
            old_handler = signal.signal(signal.SIGALRM, expired)
            installed = True
            remaining = cli_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Saved-byte CLI exceeded its wall allowance")
            signal.setitimer(signal.ITIMER_REAL, remaining)
            parser = _Parser(description=__doc__, add_help=False)
            parser.add_argument("--bundle", type=Path,
                                default=Path(__file__).resolve().parents[1] / BUNDLE_RELATIVE)
            args = parser.parse_args()
            report = verify_full_bundle(args.bundle)
            exit_code = 0
        except (OSError, ValueError, TypeError, TimeoutError, RecursionError) as error:
            report = {"status": "saved_thermal_two_round_full_byte_rejected",
                      "full_byte_authenticated": False,
                      "error_type": type(error).__name__, **INACTIVE_FLAGS}
            exit_code = 1
        if installed and time.monotonic() >= cli_deadline:
            return 1
        output = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            output = json.dumps({"full_byte_authenticated": False,
                                 "status": "saved_byte_output_rejected", **INACTIVE_FLAGS},
                                sort_keys=True) + "\n"
            exit_code = 1
        try:
            print(output, end="", flush=True)
        except (OSError, TimeoutError):
            return 1
        return exit_code
    except TimeoutError:
        return 1
    finally:
        if installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    raise SystemExit(main())
