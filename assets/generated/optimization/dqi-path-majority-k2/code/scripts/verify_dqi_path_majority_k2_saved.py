#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the fixed k=2 public projection from saved bytes only.

This checker imports only Python's standard library. It hashes the externally
pinned public manifest before parsing, authenticates the complete public file
set, and streams all five original vector ZIP members without extracting or
decoding them. It neither imports a scientific target nor constructs, evolves,
scores, normalizes or compares numerical states. The later curated fixed-run
acceptance and the unchanged raw false validation/parent flags are distinct.

The API needs only a bundle directory and its independently supplied manifest
SHA-256. The CLI additionally installs a real-time alarm and a non-enlarging
55-second kernel CPU limit. Cooperative API checks have the same finite
55-second operation ceiling; they do not certify blocked I/O or interpreter
startup. No checker invocation, test, probe or replay follows from this source.
"""
from __future__ import annotations

import time

_ENTRY_TIME = time.monotonic()

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import struct
import sys
import zipfile


BASE = "assets/generated/optimization/dqi-path-majority-k2"
MANIFEST_NAME = "public-manifest.json"
SCHEMA = "openquantum.dqi.path_majority_k2_public_projection.v1"
SOURCE_COMMIT = "d80a2be51c5156283bc4912dc6ed9b48f8543d43"
PARENT_ACCEPTANCE_SHA256 = "c8f5933fc6a4ddedc4802652ca2833f1b75a0ed429f925d1abf08c3bf466427d"
ARCHIVE_NAME = "saved-full-vectors.zip"
MAX_SECONDS = 55.0
CHUNK_BYTES = 1 << 20
MAX_NORMAL_BYTES = 1 << 20
MAX_MANIFEST_BYTES = 128 << 10
MAX_ARCHIVE_BYTES = 96 << 20
MAX_BUNDLE_BYTES = 128 << 20
MAX_FILES = 64
MAX_ENTRIES = 192
MAX_OUTPUT_BYTES = 32 << 10
VECTOR_BYTES = 67108864
VECTOR_TOTAL_BYTES = 335544320

# These are the unchanged original scientific bytes, not newly calculated
# reference values. Paths describe the public projection's repository shape.
SCIENCE_PINS = {
    "code/src/openquantum/optimization_dqi_path_majority_k2.py":
        (26863, "5fe7a85cf7046e50b072d46cbb865a39dbe3890567d09cd86f524aa9873dcd06"),
    "code/src/openquantum/optimization_dqi_path_majority_k2_verify.py":
        (45182, "57b207fb570321c6e44b6ed14fed4de915087f4017a83bb060c5e9017bb2c2f3"),
    "code/scripts/run_optimization_dqi_path_majority_k2.py":
        (44738, "3599a419ea418fb8f49f69bbb998e5fd089ef1b7d24212a6123f1d91152d09a8"),
    "scientific/source/input-k2.json":
        (56, "9c4c61f83b3e0aa4de78831b943d0638a63a92b269a96ef9b76bf223e57401a2"),
    "scientific/source/program-ledger.json":
        (5253, "95461d02d482973c88b50c386573ae4038adfca95275d497481ed7aec33f16b7"),
    "scientific/source/program.ndjson":
        (62117, "0a2dbf3c7f18e302ab0b47a820c2ec97d5feea7351d24ab2483c61c3f6d644e4"),
    "scientific/source/program.qasm":
        (29377, "067601cd2172f3b80e8c0617018027e133d1bc954a9945d94100d1de6f0c7918"),
    "scientific/verification/diagnostics.json":
        (6648, "37d27ff87830abdfbb8ea827925bf00ffbf6bde2d67da2bbcee6d01860e00df5"),
    "scientific/verification/output-table.json":
        (129536, "754201439c91683b6840d1a94c8d123459671f387c97e2cf11716c44eaafda9f"),
}
VECTOR_PINS = {
    "verification/initial-state.bin":
        (VECTOR_BYTES, "c279a5b16c71b79cf32c8478b16301897ac8234acd4f4990102a8b3f1f9ae1d2"),
    "verification/preparation-reference.bin":
        (VECTOR_BYTES, "8b5887679893fe7150dd15293520cd7ea707854c2a020ca52559facd15f7233c"),
    "verification/preparation-actual.bin":
        (VECTOR_BYTES, "29dcfdbd4bf118b45aae2c201fd338bc771bd1fbf542623431a23cf99688a4e7"),
    "verification/final-reference.bin":
        (VECTOR_BYTES, "74849bc46f31c6a3524a2bf2face69536e44eaad0aaf3d8815e0fd670277284a"),
    "verification/final-actual.bin":
        (VECTOR_BYTES, "0033d2f1be3d53bcca1617aff9ef6b9857745e7bf232eb9410d3b64c543d38d3"),
}
LICENSE_PINS = {
    "LICENSE.txt":
        (1093, "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"),
    "licenses/psutil-LICENSE.txt":
        (1548, "b89c063b3786e28e0c0a38f1931db61fed35e69dd2a2966fbecffee0f46c8d10"),
    "licenses/numpy/LICENSE.txt":
        (46826, "649a589ab31076e79e9ce62ba24eae9959598124176640dbc95f87bd10f34fcf"),
    "licenses/numpy/numpy/_core/include/numpy/libdivide/LICENSE.txt":
        (1018, "fbc539f47d0cf83bc61378080fb873d5c14630126cacbfe754035c3926daa5ec"),
    "licenses/numpy/numpy/_core/src/common/pythoncapi-compat/COPYING":
        (690, "3f2c1ffa76383a55ae0aac01c9737cd70050dff63e1f026694fbe9d662573c98"),
    "licenses/numpy/numpy/_core/src/highway/LICENSE":
        (20785, "e340270d4f64384569a91d546acb5b094d69ce47f0c015db77abb74dc6f815af"),
    "licenses/numpy/numpy/_core/src/multiarray/dragon4_LICENSE.txt":
        (1414, "f7b14309e5f54e5022b26448eefd419988ada2aec01d3c1162d84a0ea8f21f2c"),
    "licenses/numpy/numpy/_core/src/npysort/x86-simd-sort/LICENSE.md":
        (1514, "211e7d737d6179a08f791ebb0a99897ce7cd2457410c542876d125576edfe910"),
    "licenses/numpy/numpy/_core/src/umath/svml/LICENSE":
        (1543, "c17011467b4fe6e20fbfbeda4df09e58c2bca5480a063e55735a4000a379aad9"),
    "licenses/numpy/numpy/fft/pocketfft/LICENSE.md":
        (1498, "a85ca13fdf90160b64a0698215868c13b74d835ad0a4e2ba44713b8c5058a056"),
    "licenses/numpy/numpy/linalg/lapack_lite/LICENSE.txt":
        (2266, "a14cc25e10d40a3aa705b7de2fb764a6535d8ee9b2db4e1724900585457dfd55"),
    "licenses/numpy/numpy/ma/LICENSE":
        (1593, "05f3b88351988ecfad10abe92c0c50e5875c6452d5009a0084cc291551ffcca6"),
    "licenses/numpy/numpy/random/LICENSE.md":
        (3511, "103166b62b80443afb9eb3488e052ea06be0cff566b908f199e561fde49af19f"),
    "licenses/numpy/numpy/random/src/distributions/LICENSE.md":
        (2745, "469778be8f33d54e1bcaab28c281cc472916d2d273f835e55bc487d30503face"),
    "licenses/numpy/numpy/random/src/mt19937/LICENSE.md":
        (2926, "0e111337b25494a773106695f0e799f642b8567aaacca4f116a74a8177179156"),
    "licenses/numpy/numpy/random/src/pcg64/LICENSE.md":
        (1154, "e7d4f4fb20358f314210a892ae17351e04993813646f697ec387385ca2a36918"),
    "licenses/numpy/numpy/random/src/philox/LICENSE.md":
        (1509, "6f5828a57223d4cc7f8ea505a2623e7b978b242752c1515e355f46ba2cec1800"),
    "licenses/numpy/numpy/random/src/sfc64/LICENSE.md":
        (1226, "9d4098846018319cd68e228ee6220361787f73bccbf00237a85e4fc9621b998f"),
    "licenses/numpy/numpy/random/src/splitmix64/LICENSE.md":
        (332, "268b0ca06759d31ea3ca68dd35f20e877916e927b1c2a835e4592a3c4354b1f9"),
}
OTHER_NAMES = frozenset({
    ARCHIVE_NAME, "code/scripts/verify_dqi_path_majority_k2_saved.py",
    "code/requirements-lock.txt", "licenses/CPython-LICENSE.txt",
    "accepted-fixed-run.json", "environment-public.json", "README.md",
    "CREDITS.md", "circuit.svg",
})
FILE_NAMES = frozenset(SCIENCE_PINS) | frozenset(LICENSE_PINS) | OTHER_NAMES
LOCK_PIN = (1974, "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088")
JSON_NAMES = frozenset({
    "scientific/source/input-k2.json", "scientific/source/program-ledger.json",
    "scientific/verification/diagnostics.json", "accepted-fixed-run.json",
})
INPUT = {"k": 2, "signs": [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]}
RAW_DIAGNOSTIC_FLAGS = {
    "full_program_ideal_diagnostics_passed": True,
    "full_program_validated": False, "parent_acceptance_recorded": False,
}
RAW_LEDGER_FIELDS = {
    "status": "source_only_unexecuted_fixed_k2_profile",
    "full_program_validated": False, "optimization_output_validated": False,
    "source_materialized": False, "source_validation_executed": False,
    "source_generated_in_this_record": False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved-byte operation exceeded its finite deadline")


def _digest(value):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
             "An external lowercase SHA-256 is required")
    return value


def _pin(value):
    _require(type(value) is dict and set(value) == {"bytes", "sha256"},
             "A file pin must contain only bytes and SHA-256")
    _require(type(value["bytes"]) is int and 0 < value["bytes"] <= MAX_ARCHIVE_BYTES,
             "A file pin has an invalid finite byte count")
    _digest(value["sha256"])
    return value


def _relative(value):
    _require(type(value) is str and 0 < len(value) <= 256,
             "A bounded bundle-relative name is required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and all(part not in ("", ".", "..") for part in path.parts)
             and chr(0) not in value and chr(92) not in value,
             "A bundle name cannot redirect")
    return value


def _stamp(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink,
            metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


def _root(value):
    _require(isinstance(value, (str, Path)) and len(str(value)) <= 4096,
             "A bounded bundle directory is required")
    path = Path(value).absolute()
    _require(".." not in path.parts and len(path.parts) <= 64,
             "A bundle directory cannot redirect")
    for parent in reversed((path, *path.parents)):
        _require(stat.S_ISDIR(parent.lstat().st_mode),
                 "Bundle ancestors must be ordinary directories")
    return path


def _directories(names):
    result = {""}
    for name in names:
        for parent in PurePosixPath(name).parents:
            if parent.as_posix() != ".":
                result.add(parent.as_posix())
    return result


@contextmanager
def _directory_descriptor(path, deadline):
    """Walk ordinary directories with no-follow descriptor-relative opens."""
    _require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
             "This checker requires no-follow directory opens")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            _check(deadline)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            _require(stat.S_ISDIR(os.fstat(descriptor).st_mode),
                     "An opened ancestor is not an ordinary directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _inventory(root, deadline):
    """Read a finite complete tree, rejecting links and unlisted empty trees."""
    expected = FILE_NAMES | {MANIFEST_NAME}
    allowed_directories = _directories(expected)
    files, directories = {}, {"": _stamp(root.lstat())}
    pending = [(root, "")]
    entries = 0
    total_bytes = 0
    while pending:
        _check(deadline)
        directory, prefix = pending.pop()
        with _directory_descriptor(directory, deadline) as descriptor:
            _require(_stamp(os.fstat(descriptor)) == directories[prefix],
                     "A bundle directory identity changed")
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    _check(deadline)
                    entries += 1
                    _require(entries <= MAX_ENTRIES, "Bundle directory entry bound exceeded")
                    name = _relative(prefix + "/" + entry.name if prefix else entry.name)
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        _require(name in allowed_directories,
                                 "Unexpected directory in the public bundle")
                        directories[name] = _stamp(metadata)
                        pending.append((root / name, name))
                    else:
                        _require(name in expected and stat.S_ISREG(metadata.st_mode)
                                 and metadata.st_nlink == 1,
                                 "Bundle files must be listed ordinary nonlinked files")
                        maximum = (MAX_ARCHIVE_BYTES if name == ARCHIVE_NAME else
                                   MAX_MANIFEST_BYTES if name == MANIFEST_NAME else MAX_NORMAL_BYTES)
                        _require(0 < metadata.st_size <= maximum,
                                 "A public file exceeds its finite byte bound")
                        files[name] = _stamp(metadata)
                        total_bytes += metadata.st_size
                        _require(len(files) <= MAX_FILES and total_bytes <= MAX_BUNDLE_BYTES,
                                 "Compressed public bundle bound exceeded")
            _require(_stamp(os.fstat(descriptor)) == directories[prefix],
                     "A bundle directory changed during inventory")
    _require(set(files) == expected and set(directories) == allowed_directories,
             "Public bundle membership is incomplete or changed")
    return files, directories, total_bytes


@contextmanager
def _opened(root, name, stamp, deadline):
    _check(deadline)
    path = root / _relative(name)
    _require(_stamp(path.lstat()) == stamp, "A public file identity changed")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    with _directory_descriptor(path.parent, deadline) as parent_descriptor:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        try:
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                metadata = os.fstat(stream.fileno())
                _require(_stamp(metadata) == stamp and stat.S_ISREG(metadata.st_mode)
                         and metadata.st_nlink == 1, "An opened public file identity changed")
                yield stream
                _check(deadline)
                final_metadata = os.stat(path.name, dir_fd=parent_descriptor,
                                         follow_symlinks=False)
                _require(_stamp(os.fstat(stream.fileno())) == stamp
                         and _stamp(final_metadata) == stamp,
                         "A public file changed during authentication")
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _hash_file(root, name, stamp, deadline, *, retain=False):
    digest = hashlib.sha256()
    count = 0
    retained = bytearray() if retain else None
    expected_bytes = stamp[4]
    with _opened(root, name, stamp, deadline) as stream:
        while True:
            _check(deadline)
            chunk = stream.read(min(CHUNK_BYTES, expected_bytes - count + 1))
            _check(deadline)
            if not chunk:
                break
            count += len(chunk)
            _require(count <= expected_bytes, "A public file exceeded its admitted length")
            digest.update(chunk)
            if retained is not None:
                retained.extend(chunk)
    _require(count == expected_bytes, "A public file ended before its admitted length")
    return {"bytes": count, "sha256": digest.hexdigest()}, (
        bytes(retained) if retained is not None else None)


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        _require(name not in result, "Duplicate JSON object member")
        result[name] = value
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON constants are not admitted")


def _json(raw):
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                        parse_constant=_constant)
    _require(type(result) is dict, "Saved JSON must contain an ordinary object")
    return result


def _known_pin(pair):
    return {"bytes": pair[0], "sha256": pair[1]}


def _manifest_contract(value):
    _require(value.get("schema") == SCHEMA and value.get("source_commit") == SOURCE_COMMIT
             and value.get("parent_acceptance_sha256") == PARENT_ACCEPTANCE_SHA256,
             "Public manifest provenance differs from the fixed accepted run")
    files = value.get("files")
    _require(type(files) is dict and set(files) == FILE_NAMES,
             "Manifest payload membership differs from the closed projection")
    for name, pin in files.items():
        _relative(name)
        _pin(pin)
        _require(pin["bytes"] <= (MAX_ARCHIVE_BYTES if name == ARCHIVE_NAME else MAX_NORMAL_BYTES),
                 "A manifest file pin exceeds its role bound")
    for name, pair in (SCIENCE_PINS | LICENSE_PINS).items():
        _require(files[name] == _known_pin(pair), "An immutable source or license pin changed")
    _require(files["code/requirements-lock.txt"] == _known_pin(LOCK_PIN),
             "Historical requirements-lock snapshot changed")
    archive = value.get("vector_archive")
    _require(type(archive) is dict
             and set(archive) == {"name", "compression", "members", "uncompressed_bytes"}
             and archive["name"] == ARCHIVE_NAME and archive["compression"] == "ZIP_DEFLATED"
             and type(archive["uncompressed_bytes"]) is int
             and archive["uncompressed_bytes"] == VECTOR_TOTAL_BYTES,
             "Vector archive contract differs from the five original vectors")
    members = archive["members"]
    _require(type(members) is dict and set(members) == set(VECTOR_PINS),
             "The complete five-vector member set is required")
    for name, pair in VECTOR_PINS.items():
        _pin(members[name])
        _require(members[name] == _known_pin(pair), "An original vector pin changed")
    return files


def _subset(value, expected, message):
    _require(type(value) is dict, message)
    for name, wanted in expected.items():
        observed = value.get(name)
        _require(type(observed) is type(wanted) and observed == wanted, message)


def _saved_contracts(raw):
    """Check saved metadata labels only; do no diagnostic arithmetic."""
    input_value = _json(raw["scientific/source/input-k2.json"])
    _require(input_value == INPUT, "Saved fixed input changed")
    ledger = _json(raw["scientific/source/program-ledger.json"])
    _subset(ledger, RAW_LEDGER_FIELDS, "Historical source ledger flags changed")
    _subset(ledger, {"schema": "openquantum.dqi.path_majority_k2_source_plan.v1",
                    "source_wires": 22, "N": 10, "m": 10, "n": 9, "k": 2,
                    "source_unitary_primitives": 2035, "source_materialized_operations": 2073,
                    "source_operational_events": 2066, "primitive_ry_occurrences": 37},
            "Saved source accounting labels changed")
    diagnostics = _json(raw["scientific/verification/diagnostics.json"])
    _subset(diagnostics, RAW_DIAGNOSTIC_FLAGS, "Raw diagnostic acceptance flags changed")
    _subset(diagnostics, {
        "schema": "openquantum.dqi.path_majority_k2_full_validation.v1",
        "status": "diagnostics_completed_not_execution_acceptance",
        "Q": 22, "N": 10, "m": 10, "n": 9, "source_records": 2073,
        "unitary_primitive_applications": 2035, "state_amplitudes": 4194304,
        "operational_events_excluding_barriers": 2066,
        "readout_scope": "nine source mappings and unconditional probability marginal; no shots",
        "reset_scope": "22 ordered resets of the explicitly known zero input",
        "sampling_performed": False, "rotation_synthesis_validated": False,
        "physical_resources_estimated": False, "formal_proof_reproduced": False,
        "hardware_executed": False, "publication_complete": False,
    }, "Saved fixed diagnostic scope changed")
    acceptance = _json(raw["accepted-fixed-run.json"])
    _subset(acceptance, {
        "schema": "openquantum.dqi.path_majority_k2_curated_acceptance.v1",
        "source_commit": SOURCE_COMMIT, "parent_acceptance_sha256": PARENT_ACCEPTANCE_SHA256,
        "accepted_fixed_ideal_run": True, "general_algorithm_validated": False,
        "physical_resources_or_synthesis_validated": False, "formal_or_hardware_claim": False,
    }, "Curated fixed acceptance scope changed")
    flags = acceptance.get("raw_flags")
    _require(type(flags) is dict, "Curated acceptance must disclose the raw flags")
    _subset(flags.get("diagnostics"), RAW_DIAGNOSTIC_FLAGS, "Raw diagnostic disclosure changed")
    _subset(flags.get("ledger"), RAW_LEDGER_FIELDS, "Raw ledger disclosure changed")
    _require(type(acceptance.get("scope")) is str and 0 < len(acceptance["scope"]) <= 4096,
             "Curated acceptance requires a bounded explicit scope")


def _read_exact(stream, count, deadline):
    _require(type(count) is int and 0 <= count <= 1024,
             "ZIP metadata read exceeds its small bound")
    _check(deadline)
    raw = stream.read(count)
    _check(deadline)
    _require(len(raw) == count, "ZIP metadata ended unexpectedly")
    return raw


def _zip_headers(stream, archive, members, archive_bytes, deadline):
    """Reject unlisted prefixes, trailers, extras and local metadata payloads."""
    cursor = 0
    for member in sorted(members, key=lambda item: item.header_offset):
        _check(deadline)
        _require(member.header_offset == cursor, "ZIP has an unlisted prefix or gap")
        stream.seek(cursor)
        header = struct.unpack("<4s5H3I2H", _read_exact(stream, 30, deadline))
        signature, version, flags, method = header[:4]
        crc, compressed, uncompressed, name_bytes, extra_bytes = header[6:]
        _require(signature == b"PK\x03\x04" and version <= 20
                 and flags == member.flag_bits and method == zipfile.ZIP_DEFLATED
                 and flags & ~0x808 == 0 and extra_bytes == 0,
                 "ZIP local header has unsupported metadata")
        encoded = member.filename.encode("utf-8" if flags & 0x800 else "ascii")
        _require(name_bytes == len(encoded)
                 and _read_exact(stream, name_bytes, deadline) == encoded,
                 "ZIP local member name changed")
        if flags & 8:
            _require(crc in (0, member.CRC) and compressed in (0, member.compress_size)
                     and uncompressed in (0, member.file_size),
                     "ZIP descriptor header lengths changed")
        else:
            _require((crc, compressed, uncompressed) ==
                     (member.CRC, member.compress_size, member.file_size),
                     "ZIP local lengths differ from the central directory")
        cursor += 30 + name_bytes + member.compress_size
        if flags & 8:
            stream.seek(cursor)
            first = _read_exact(stream, 4, deadline)
            if first == b"PK\x07\x08":
                values = struct.unpack("<3I", _read_exact(stream, 12, deadline))
                cursor += 16
            else:
                values = struct.unpack("<3I", first + _read_exact(stream, 8, deadline))
                cursor += 12
            _require(values == (member.CRC, member.compress_size, member.file_size),
                     "ZIP data descriptor differs from the central directory")
    _require(cursor == archive.start_dir, "ZIP contains unlisted data before its directory")
    stream.seek(archive.start_dir)
    for member in members:
        central = struct.unpack("<4s6H3I5H2I", _read_exact(stream, 46, deadline))
        _require(central[0] == b"PK\x01\x02" and central[2] <= 20
                 and central[3] == member.flag_bits and central[4] == zipfile.ZIP_DEFLATED
                 and central[7:10] == (member.CRC, member.compress_size, member.file_size)
                 and central[11:14] == (0, 0, 0)
                 and central[15] == member.external_attr
                 and central[16] == member.header_offset,
                 "ZIP central directory has unsupported metadata")
        encoded = member.filename.encode("utf-8" if member.flag_bits & 0x800 else "ascii")
        _require(central[10] == len(encoded)
                 and _read_exact(stream, central[10], deadline) == encoded,
                 "ZIP central member name changed")
    directory_stop = stream.tell()
    _require(directory_stop + 22 == archive_bytes, "ZIP contains an unlisted trailer")
    ending = struct.unpack("<4s4H2IH", _read_exact(stream, 22, deadline))
    _require(ending[0] == b"PK\x05\x06" and ending[1:3] == (0, 0)
             and ending[3:5] == (5, 5) and ending[5] == directory_stop - archive.start_dir
             and ending[6] == archive.start_dir and ending[7] == 0,
             "ZIP end record differs from the closed five-member directory")


def _vectors(root, stamp, deadline):
    results = {}
    with _opened(root, ARCHIVE_NAME, stamp, deadline) as stream:
        with zipfile.ZipFile(stream, "r") as archive:
            _check(deadline)
            members = archive.infolist()
            _require(len(members) == 5 and archive.comment == b"",
                     "Vector ZIP must contain exactly five uncommented members")
            names = [member.filename for member in members]
            _require(len(set(names)) == 5 and set(names) == set(VECTOR_PINS),
                     "Vector ZIP member names are incomplete or duplicated")
            for member in members:
                _check(deadline)
                _relative(member.filename)
                _require(member.orig_filename == member.filename and not member.is_dir()
                         and stat.S_ISREG(member.external_attr >> 16)
                         and member.compress_type == zipfile.ZIP_DEFLATED
                         and member.flag_bits & ~0x808 == 0
                         and member.extra == b"" and member.comment == b""
                         and member.file_size == VECTOR_BYTES
                         and 0 < member.compress_size <= MAX_ARCHIVE_BYTES,
                         "Vector ZIP contains unsupported links, metadata or lengths")
            _require(sum(member.file_size for member in members) == VECTOR_TOTAL_BYTES,
                     "Vector ZIP uncompressed byte bound changed")
            _zip_headers(stream, archive, members, stamp[4], deadline)
            for member in members:
                digest = hashlib.sha256()
                count = 0
                with archive.open(member, "r") as payload:
                    while True:
                        _check(deadline)
                        chunk = payload.read(min(CHUNK_BYTES, VECTOR_BYTES - count + 1))
                        _check(deadline)
                        if not chunk:
                            break
                        count += len(chunk)
                        _require(count <= VECTOR_BYTES, "Vector ZIP member exceeded its byte bound")
                        digest.update(chunk)
                pin = {"bytes": count, "sha256": digest.hexdigest()}
                _require(pin == _known_pin(VECTOR_PINS[member.filename]),
                         "An original saved vector byte identity changed")
                results[member.filename] = pin
    return results


def verify_saved(bundle_root: Path, manifest_sha256: str) -> dict:
    """Authenticate this closed saved bundle; never replay its scientific code."""
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    expected_manifest = _digest(manifest_sha256)
    root = _root(bundle_root)
    files_before, directories_before, total_bytes = _inventory(root, deadline)
    manifest_pin, manifest_raw = _hash_file(
        root, MANIFEST_NAME, files_before[MANIFEST_NAME], deadline, retain=True)
    _require(manifest_pin["sha256"] == expected_manifest,
             "The external public-manifest SHA-256 did not authenticate")
    manifest = _json(manifest_raw)
    declared = _manifest_contract(manifest)
    observed = {MANIFEST_NAME: manifest_pin}
    raw_json = {}
    for name in sorted(FILE_NAMES):
        _check(deadline)
        pin, raw = _hash_file(root, name, files_before[name], deadline, retain=name in JSON_NAMES)
        _require(pin == declared[name], "A public payload differs from its authenticated manifest")
        observed[name] = pin
        if raw is not None:
            raw_json[name] = raw
    _saved_contracts(raw_json)
    vectors = _vectors(root, files_before[ARCHIVE_NAME], deadline)
    files_after, directories_after, final_bytes = _inventory(root, deadline)
    _require(files_before == files_after and directories_before == directories_after
             and final_bytes == total_bytes, "Public bundle changed during authentication")
    _check(deadline)
    return {
        "status": "authenticated_saved_fixed_k2_projection_bytes_without_replay",
        "manifest_sha256": expected_manifest, "source_commit": SOURCE_COMMIT,
        "parent_acceptance_sha256": PARENT_ACCEPTANCE_SHA256,
        "accepted_fixed_ideal_run": True,
        "raw_flags": {"diagnostics": dict(RAW_DIAGNOSTIC_FLAGS),
                      "ledger": dict(RAW_LEDGER_FIELDS)},
        "files_authenticated": len(observed),
        "scientific_files_authenticated": len(SCIENCE_PINS),
        "scientific_source_snapshots_authenticated": 3,
        "vector_members_authenticated": len(vectors),
        "vector_uncompressed_bytes_authenticated": VECTOR_TOTAL_BYTES,
        "compressed_bundle_bytes_authenticated": total_bytes,
        "files": observed, "vector_members": vectors,
        "saved_metadata_contracts_checked": True,
        "scientific_replay_performed": False,
        "scientific_target_imports_performed": False,
        "numpy_import_performed": False,
        "vector_values_decoded": False, "diagnostics_recomputed": False,
        "zip_members_extracted": False,
        "general_algorithm_validated": False,
        "physical_resources_or_synthesis_validated": False,
        "formal_or_hardware_claim": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def _cli_guard():
    import resource

    remaining = MAX_SECONDS - (time.monotonic() - _ENTRY_TIME)
    _require(remaining > 0, "CLI startup exhausted the finite deadline")
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    hard_limit = min(55, hard) if hard != resource.RLIM_INFINITY else 55
    soft_limit = min(hard_limit, soft) if soft != resource.RLIM_INFINITY else hard_limit
    resource.setrlimit(resource.RLIMIT_CPU, (soft_limit, hard_limit))

    def expired(_signal, _frame):
        raise TimeoutError("Saved-byte CLI exceeded its finite real-time limit")

    signal.signal(signal.SIGALRM, expired)
    timer, _ = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, timer) if timer > 0 else remaining)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path,
                        default=Path(__file__).absolute().parents[1] / BASE,
                        help="The closed public k2 bundle directory")
    parser.add_argument("--manifest-sha256", required=True,
                        help="Independently supplied exact public-manifest SHA-256")
    args = parser.parse_args()
    try:
        _cli_guard()
        result = verify_saved(args.bundle_root, args.manifest_sha256)
        output = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
        _require(len(output) <= MAX_OUTPUT_BYTES, "Saved-byte CLI output bound exceeded")
        print(output.decode("utf-8"), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, TimeoutError,
            zipfile.BadZipFile, RuntimeError, EOFError, UnicodeError, struct.error) as error:
        # Static failure text keeps accidental absolute operational paths out
        # of the public result contract. The checker writes no receipt.
        print("Saved k2 byte authentication failed: " + type(error).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
