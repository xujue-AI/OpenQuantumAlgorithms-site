#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the frozen finite weighted-decay packet without numerical replay.

Only retained identities, scope and source/notebook correspondence are checked.
No producer, simulator, estimator, subprocess or scientific library is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat

BUNDLE_RELATIVE = "assets/generated/simulation/detailed-balance-decay"
MANIFEST_SHA256 = "9acc1af928c3484a8ad7c24a5768a1c4ac643cc02c541e5ad0df2b0df1223f94"
PRODUCER_COMMIT = "b1a05ef76a47d8e7ef893cd6bf0ffb66772d4ce1"
BUNDLE_TOTAL_BYTES = 483911
PAYLOAD_TOTAL_BYTES = 479596
MAX_FILE_BYTES = 256 * 1024
MAX_JSON_BYTES = 128 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 50000
MAX_PATH_COMPONENTS = 128
BUNDLE_RECORDS = {'artifact-manifest.json': {'bytes': 4226,
                            'sha256': '9acc1af928c3484a8ad7c24a5768a1c4ac643cc02c541e5ad0df2b0df1223f94'},
 'artifact-manifest.sha256': {'bytes': 89,
                              'sha256': 'c0bc4ceba63f145c7e1126f8c729f8b4ffbb137c3043ca0340adc80bd2768f68'},
 'completion.json': {'bytes': 3347,
                     'sha256': '38438d02427f1ac4368d24b850f3cee7cfb0e8d1dbb9363b6ea10d564b949ad6'},
 'grid/LICENSE.txt': {'bytes': 1093,
                      'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'grid/complex-decay-block.qasm': {'bytes': 11985,
                                   'sha256': '859dd5d7541ca2ab2c96011865e3549120a42ce95b603bb817a767a3ef4e4acb'},
 'grid/detailed_balance_access.py': {'bytes': 25361,
                                     'sha256': '7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77'},
 'grid/detailed_balance_decay.py': {'bytes': 12324,
                                    'sha256': '82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1'},
 'grid/detailed_balance_decay_reference.py': {'bytes': 26002,
                                              'sha256': '71ac5f486ad04cb532a86bf1c4dec4e175d0cfb377a7de26f206cd3f5c5daf53'},
 'grid/diagonal-decay-block.qasm': {'bytes': 2549,
                                    'sha256': '94f063c06242cac81b2c5a9ff994d6b7aea288cfea45643718764bf09439fc92'},
 'grid/generate_detailed_balance_decay.py': {'bytes': 16294,
                                             'sha256': 'e5b0be192e7e1bb48fd7bc8d9148f4c41e19e448edb26a023b78d972b46e7cad'},
 'grid/manifest.json': {'bytes': 2068,
                        'sha256': 'a77996282baad56fc8d15d44635d204dd21764fc09b57d557057c62783fb89de'},
 'grid/manifest.sha256': {'bytes': 80,
                          'sha256': '86d911efff450b9c096f136663392eb16d1c0b691048ac5af53865c3e837e2af'},
 'grid/requirements-lock.txt': {'bytes': 1974,
                                'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'grid/results.json': {'bytes': 79004,
                       'sha256': 'dab75172b425b99c8b0e0fb3d015018c2d3dc2e7e111d15c5060bb005dc1d3dd'},
 'grid/tilted-decay-block.qasm': {'bytes': 4089,
                                  'sha256': '3733789ebe85984d617310dae7ebe0e5d36b6c7142dd49183fc12b7f20e4ba3c'},
 'n12/LICENSE.txt': {'bytes': 1093,
                     'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'n12/chain-n12-decay-block.qasm': {'bytes': 45013,
                                    'sha256': '642c8b66a22458e03c188fd27b62988dc8a55e25c732a084783c35ccea37567f'},
 'n12/detailed_balance_access.py': {'bytes': 25361,
                                    'sha256': '7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77'},
 'n12/detailed_balance_decay.py': {'bytes': 12324,
                                   'sha256': '82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1'},
 'n12/detailed_balance_decay_reference.py': {'bytes': 26002,
                                             'sha256': '71ac5f486ad04cb532a86bf1c4dec4e175d0cfb377a7de26f206cd3f5c5daf53'},
 'n12/generate_detailed_balance_decay.py': {'bytes': 16294,
                                            'sha256': 'e5b0be192e7e1bb48fd7bc8d9148f4c41e19e448edb26a023b78d972b46e7cad'},
 'n12/manifest.json': {'bytes': 1788,
                       'sha256': '13f5f577a2cb419a140f3a5c7122eae769c1498eb7395a582be9b431ed7070a2'},
 'n12/manifest.sha256': {'bytes': 80,
                         'sha256': 'acb40d57faff7356a9795cc06cd4fc7b781eb54d6f3e8c5091ad7be0e99a3915'},
 'n12/requirements-lock.txt': {'bytes': 1974,
                               'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'n12/results.json': {'bytes': 13258,
                      'sha256': 'ff418a00eb7888ca8badf985eca76c2833b30ebae7deea822346602a1a46e718'},
 'notebook/executed-original.ipynb': {'bytes': 67042,
                                      'sha256': '75b3e78a8bab913164cd7a9497b653c3806b1b9a760fb3970474c7708d7f97a4'},
 'notebook/executed.ipynb': {'bytes': 67325,
                             'sha256': '681fce3c1cc2c322761af3096ce1e4e425a3ce8fee75f9d63413efdb382d1d43'},
 'notebook/source.ipynb': {'bytes': 15872,
                           'sha256': 'bf40152ca9d7ae93362594ebc5f36d206aa8d08a9c978cb664229b35d1709021'}}
SOURCE_RECORDS = {'LICENSE': {'bytes': 1093, 'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'requirements-lock.txt': {'bytes': 1974,
                           'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'scripts/generate_detailed_balance_decay.py': {'bytes': 16294,
                                                'sha256': 'e5b0be192e7e1bb48fd7bc8d9148f4c41e19e448edb26a023b78d972b46e7cad'},
 'src/openquantum/simulation/detailed_balance_access.py': {'bytes': 25361,
                                                           'sha256': '7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77'},
 'src/openquantum/simulation/detailed_balance_decay.py': {'bytes': 12324,
                                                          'sha256': '82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1'},
 'src/openquantum/simulation/detailed_balance_decay_reference.py': {'bytes': 26002,
                                                                    'sha256': '71ac5f486ad04cb532a86bf1c4dec4e175d0cfb377a7de26f206cd3f5c5daf53'},
 'tests/test_detailed_balance_decay.py': {'bytes': 19208,
                                          'sha256': '46e118196d083b8ec5014f88fcddd5b0a2ec21279e6da6f1c1c4b903a10fd07a'}}
SNAPSHOT_SOURCES = {'LICENSE.txt': 'LICENSE',
 'detailed_balance_access.py': 'src/openquantum/simulation/detailed_balance_access.py',
 'detailed_balance_decay.py': 'src/openquantum/simulation/detailed_balance_decay.py',
 'detailed_balance_decay_reference.py': 'src/openquantum/simulation/detailed_balance_decay_reference.py',
 'generate_detailed_balance_decay.py': 'scripts/generate_detailed_balance_decay.py',
 'requirements-lock.txt': 'requirements-lock.txt'}

BUNDLE_FILES = frozenset(BUNDLE_RECORDS)
SOURCE_SHA256 = {name: entry["sha256"] for name, entry in SOURCE_RECORDS.items()}
FALSE_FIELDS = frozenset({
    "all_components_verified", "new_algorithm_result", "coherent_correction_implemented",
    "general_coherent_correction_implemented", "coherent_correction_evolution_implemented",
    "general_channel_implemented", "channel_implemented", "channel_or_sampler_implemented",
    "sampler_implemented", "exact_finite_kms", "finite_mesh_exact_kms", "mixing_certificate",
    "rotations_synthesized", "physical_resources_computed", "formal_proof_reproduced",
    "random_sampling_performed", "source_n12_simulated", "notebook_numerical_replay_performed",
    "complete_descendant_lifetime_cleanup_proven",
})
NOTEBOOK_RELATIVE = "notebooks/simulation_detailed_balance_decay.ipynb"
NOTEBOOK_IDS = (
    "decay-introduction", "decay-source-import", "decay-source-figure-intuition",
    "decay-source-figure", "decay-saved-grid-explanation", "decay-read-grid",
    "decay-raw-diagnostics", "decay-resource-scope", "decay-read-source-ledger", "decay-final-boundary",
)


def _equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("Unexpected decay " + label)
    if type(expected) is dict:
        for key in expected:
            _equal(actual[key], expected[key], label)
    elif type(expected) in (tuple, list):
        for left, right in zip(actual, expected):
            _equal(left, right, label)


def _relative(name):
    if type(name) is not str:
        raise ValueError("Unsafe decay relative path")
    path = PurePosixPath(name)
    if (not name or "\\" in name or any(ord(char) < 32 for char in name)
            or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or len(path.parts) > MAX_PATH_COMPONENTS):
        raise ValueError("Unsafe decay relative path")
    return path


def _root(value):
    root = Path(value).absolute()
    if (len(root.parts) > MAX_PATH_COMPONENTS or root != root.resolve()
            or root.is_symlink() or not root.is_dir()):
        raise ValueError("Decay root must be an unaliased directory")
    return root


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory_identity(info):
    # Sibling writes may change directory timestamps without changing the path.
    return info.st_dev, info.st_ino, info.st_mode


def _directory_flags():
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("Secure decay descriptor reads are unsupported")
    return os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _child_directory(parent, name, held):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("Linked or non-directory decay ancestor")
    child = os.open(name, _directory_flags(), dir_fd=parent)
    held.append((child, parent, name, before))
    if _directory_identity(os.fstat(child)) != _directory_identity(before):
        raise ValueError("Decay directory changed before open")
    return child


def _check_directories(held):
    for descriptor, parent, name, before in reversed(held):
        current = (os.lstat(name) if parent is None
                   else os.stat(name, dir_fd=parent, follow_symlinks=False))
        if (_directory_identity(os.fstat(descriptor)) != _directory_identity(before)
                or _directory_identity(current) != _directory_identity(before)):
            raise ValueError("Decay directory changed during authentication")


@contextmanager
def _directory(value):
    """Walk every absolute ancestor with non-following directory descriptors."""
    root = _root(value)
    held = []
    try:
        before = os.lstat(root.anchor)
        descriptor = os.open(root.anchor, _directory_flags())
        held.append((descriptor, None, root.anchor, before))
        if _directory_identity(os.fstat(descriptor)) != _directory_identity(before):
            raise ValueError("Decay filesystem root changed before open")
        for part in root.parts[1:]:
            descriptor = _child_directory(descriptor, part, held)
        yield descriptor, held
        _check_directories(held)
    finally:
        for descriptor, _parent, _name, _before in reversed(held):
            os.close(descriptor)


def _read(root, name, maximum=MAX_FILE_BYTES):
    """Bound allocation and reject links, nonregular files and path/read races."""
    relative = _relative(name)
    if type(maximum) is not int or not 0 <= maximum <= MAX_FILE_BYTES:
        raise ValueError("Invalid decay read cap")
    with _directory(root) as (descriptor, held):
        for part in relative.parts[:-1]:
            descriptor = _child_directory(descriptor, part, held)
        before = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 <= before.st_size <= maximum):
            raise ValueError("Unsafe or oversized decay file")
        handle = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                         | getattr(os, "O_CLOEXEC", 0), dir_fd=descriptor)
        try:
            if _identity(os.fstat(handle)) != _identity(before):
                raise ValueError("Decay file changed before open")
            content = bytearray()
            while len(content) <= maximum:
                chunk = os.read(handle, min(65536, maximum + 1 - len(content)))
                if not chunk:
                    break
                content.extend(chunk)
            after = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
            if (len(content) != before.st_size or len(content) > maximum
                    or _identity(os.fstat(handle)) != _identity(before)
                    or _identity(after) != _identity(before)):
                raise ValueError("Decay file changed during bounded read")
            return bytes(content)
        finally:
            os.close(handle)


def _json(raw):
    if type(raw) is not bytes or len(raw) > MAX_JSON_BYTES:
        raise ValueError("Oversized decay JSON")
    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("Duplicate decay JSON key")
            value[key] = child
        return value
    def constant(_value):
        raise ValueError("Nonfinite decay JSON constant")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid decay JSON") from error
    pending = [(value, 0)]
    visited = 0
    while pending:
        child, depth = pending.pop()
        visited += 1
        if depth > MAX_JSON_DEPTH or visited > MAX_JSON_NODES:
            raise ValueError("Decay JSON structural limit")
        if type(child) is dict:
            pending.extend((item, depth + 1) for item in child.values())
        elif type(child) is list:
            pending.extend((item, depth + 1) for item in child)
        elif type(child) is str and len(child) > MAX_JSON_BYTES:
            raise ValueError("Oversized decay JSON string")
        elif type(child) is float and not math.isfinite(child):
            raise ValueError("Nonfinite decay JSON number")
        elif type(child) is int and child.bit_length() > 4096:
            raise ValueError("Oversized decay JSON integer")
    return value


def _inventory(bundle):
    files = set()
    total = 0
    expected_directories = {"grid", "n12", "notebook"}
    expected_root_files = {name for name in BUNDLE_FILES if "/" not in name}
    with _directory(bundle) as (descriptor, held):
        with os.scandir(descriptor) as entries:
            names = set()
            for entry in entries:
                if entry.name in names or len(names) >= 6:
                    raise ValueError("Overlimit decay root inventory")
                names.add(entry.name)
                info = entry.stat(follow_symlinks=False)
                if entry.name in expected_directories:
                    if not stat.S_ISDIR(info.st_mode):
                        raise ValueError("Unsafe decay bundle directory")
                elif entry.name in expected_root_files:
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("Nonregular decay inventory member")
                    files.add(entry.name)
                    total += info.st_size
                else:
                    raise ValueError("Unrecorded decay root inventory member")
            _equal(names, expected_directories | expected_root_files, "root inventory")
        for folder in sorted(expected_directories):
            child = _child_directory(descriptor, folder, held)
            expected = {name.split("/", 1)[1] for name in BUNDLE_FILES if name.startswith(folder + "/")}
            names = set()
            with os.scandir(child) as entries:
                for entry in entries:
                    if entry.name in names or entry.name not in expected or len(names) >= len(expected):
                        raise ValueError("Unrecorded or overlimit decay nested inventory member")
                    names.add(entry.name)
                    info = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise ValueError("Nonregular decay nested inventory member")
                    files.add(folder + "/" + entry.name)
                    total += info.st_size
                    if total > BUNDLE_TOTAL_BYTES:
                        raise ValueError("Decay bundle byte cap exceeded")
            _equal(names, expected, "nested closed inventory")
    _equal(files, set(BUNDLE_FILES), "closed 28-file inventory")
    _equal(total, BUNDLE_TOTAL_BYTES, "complete bundle bytes")


def _false_flags(value):
    pending = [value]
    while pending:
        child = pending.pop()
        if type(child) is dict:
            for key, item in child.items():
                if key in FALSE_FIELDS:
                    _equal(item, False, "retained false scope flag " + key)
                pending.append(item)
        elif type(child) is list:
            pending.extend(child)


def _authenticate(bundle):
    raw = _read(bundle, "artifact-manifest.json", BUNDLE_RECORDS["artifact-manifest.json"]["bytes"])
    _equal(hashlib.sha256(raw).hexdigest(), MANIFEST_SHA256, "immutable top manifest pin")
    manifest = _json(raw)
    expected_payload = {name: pin for name, pin in BUNDLE_RECORDS.items()
                        if name not in {"artifact-manifest.json", "artifact-manifest.sha256"}}
    _equal(set(manifest), {"schema_version", "component", "files", "total_payload_bytes", "source_commit",
                           "all_components_verified", "new_algorithm_result",
                           "general_coherent_correction_implemented", "channel_or_sampler_implemented",
                           "physical_resources_computed"}, "top manifest fields")
    _equal(manifest["schema_version"], 1, "top schema")
    _equal(manifest["component"], "finite weighted-decay block access", "component")
    _equal(manifest["source_commit"], PRODUCER_COMMIT, "producer commit")
    _equal(manifest["files"], expected_payload, "top payload inventory")
    _equal(manifest["total_payload_bytes"], PAYLOAD_TOTAL_BYTES, "top payload bytes")
    _false_flags(manifest)
    _inventory(bundle)
    data = {"artifact-manifest.json": raw}
    for name, pin in BUNDLE_RECORDS.items():
        if name == "artifact-manifest.json":
            continue
        content = _read(bundle, name, pin["bytes"])
        _equal(len(content), pin["bytes"], "file length")
        _equal(hashlib.sha256(content).hexdigest(), pin["sha256"], "frozen file digest")
        data[name] = content
    _equal(data["artifact-manifest.sha256"], (MANIFEST_SHA256 + "  artifact-manifest.json\n").encode(), "top sidecar bytes")
    for folder, status, total in (("grid", "checked_small_finite_decay_block", 180675),
                                 ("n12", "generated_decay_block_without_simulation", 141319)):
        nested = _json(data[folder + "/manifest.json"])
        pins = {name.split("/", 1)[1]: pin for name, pin in BUNDLE_RECORDS.items()
                if name.startswith(folder + "/") and name.split("/", 1)[1] not in {"manifest.json", "manifest.sha256"}}
        _equal(set(nested), {"schema_version", "status", "files", "total_payload_bytes",
                             "manifest_and_sidecar_are_separate", *(
                                 key for key in FALSE_FIELDS if key in nested)}, "nested manifest fields")
        _equal(nested["schema_version"], 1, "nested schema")
        _equal(nested["status"], status, "nested historical status")
        _equal(nested["files"], pins, "nested payload inventory")
        _equal(nested["total_payload_bytes"], total, "nested payload bytes")
        _equal(sum(pin["bytes"] for pin in pins.values()), total, "nested pin sum")
        _equal(nested["manifest_and_sidecar_are_separate"], True, "nested sidecar separation")
        digest = BUNDLE_RECORDS[folder + "/manifest.json"]["sha256"]
        _equal(data[folder + "/manifest.sha256"], (digest + "  manifest.json\n").encode(), "nested sidecar bytes")
        _false_flags(nested)
    return data


def _source_text(cell):
    source = cell.get("source")
    if type(source) is str:
        return source
    if type(source) is list and all(type(part) is str for part in source):
        return "".join(source)
    raise ValueError("Invalid decay notebook source text")


def _cells(notebook):
    if type(notebook) is not dict or type(notebook.get("cells")) is not list or len(notebook["cells"]) != 10:
        raise ValueError("Invalid decay notebook cell inventory")
    cells = notebook["cells"]
    if any(type(cell) is not dict for cell in cells):
        raise ValueError("Invalid decay notebook cell")
    _equal(tuple(cell.get("id") for cell in cells), NOTEBOOK_IDS, "notebook cell identities")
    for index, cell in enumerate(cells):
        if type(cell) is not dict:
            raise ValueError("Invalid decay notebook cell")
        _equal(cell.get("cell_type"), "code" if index in {1, 3, 5, 6, 8} else "markdown", "notebook cell types")
        _source_text(cell)
    return cells


def _notebooks(data, source_root):
    source = _cells(_json(data["notebook/source.ipynb"]))
    original = _cells(_json(data["notebook/executed-original.ipynb"]))
    presentation = _cells(_json(data["notebook/executed.ipynb"]))
    counts = []
    changed = []
    for index, (before, after, unexecuted) in enumerate(zip(original, presentation, source)):
        _equal(_source_text(before), _source_text(unexecuted), "original/source notebook text")
        _equal({key: value for key, value in before.items() if key != "source"},
               {key: value for key, value in after.items() if key != "source"}, "presentation cell fields")
        if _source_text(before) != _source_text(after):
            changed.append(index)
        if before["cell_type"] == "code":
            _equal(_source_text(before), _source_text(after), "presentation code text")
            counts.append(before.get("execution_count"))
            if type(before.get("outputs")) is not list or any(type(output) is not dict or output.get("output_type") == "error" for output in before["outputs"]):
                raise ValueError("Invalid retained decay notebook outputs")
    _equal(changed, [0, 4], "presentation-only Markdown changes")
    _equal(counts, [1, 2, 3, 4, 5], "five genuine recorded execution counts")
    repository = _cells(_json(_read(source_root, NOTEBOOK_RELATIVE)))
    for before, after in zip(source, repository):
        if before["cell_type"] == "code":
            _equal(_source_text(before), _source_text(after), "repository notebook code text")
            if type(after.get("outputs")) is not list or any(type(output) is not dict or output.get("output_type") == "error" for output in after["outputs"]):
                raise ValueError("Repository decay notebook has invalid/error outputs")


def _bind_sources(data, source_root):
    for name, pin in SOURCE_RECORDS.items():
        content = _read(source_root, name, pin["bytes"])
        _equal(len(content), pin["bytes"], "accepted source length")
        _equal(hashlib.sha256(content).hexdigest(), pin["sha256"], "accepted source digest")
        for folder in ("grid", "n12"):
            for snapshot, selected in SNAPSHOT_SOURCES.items():
                if selected == name:
                    _equal(data[folder + "/" + snapshot], content, "adjacent source snapshot bytes")


def _records(data):
    completion = _json(data["completion.json"])
    _equal(completion["status"], "verified_small_finite_decay_block", "completion status")
    _equal(completion["source_commit"], PRODUCER_COMMIT, "completion producer")
    _equal(completion["completed_cli_exit"], 0, "retained CLI exit")
    _equal(completion["completed_stage_exits"], [0, 0, 0, 0], "retained stage exits")
    _equal(completion["focused_tests_passed"], 70, "retained focused cases")
    _equal(completion["notebook_code_cells"], 5, "retained notebook cells")
    _equal(completion["notebook_execution_counts"], [1, 2, 3, 4, 5], "retained notebook counts")
    _equal(completion["prior_collection_failure"], {
        "source_commit": "e955aa1794fd99132fa53bd3e16145ac8139f114", "cli_exit": 1,
        "pytest_exit": 2, "executed_intended_cases": 0, "declared_scientific_calls": 0, "retained_separately": True,
    }, "prior zero-case collection failure")
    _false_flags(completion)
    reports = {}
    for folder, status, expected_names, wires, counts in (
            ("grid", "checked_small_finite_decay_block", ["diagonal", "tilted", "complex"], [6, 5, 8], [130, 218, 744]),
            ("n12", "generated_decay_block_without_simulation", ["chain-n12"], [18], [2408])):
        report = _json(data[folder + "/results.json"])
        _equal(report["status"], status, "result status")
        _equal(report["source_commit"], PRODUCER_COMMIT, "result producer")
        _equal(report["source_dirty"], False, "frozen source clean")
        _equal(report["source_revision_is_exact"], True, "frozen exact source")
        _equal(report["paper"]["doi"], "10.1038/s41586-025-09583-x", "paper identity")
        _equal(report["environment_lock_sha256"], SOURCE_SHA256["requirements-lock.txt"], "locked environment digest")
        _equal([case["name"] for case in report["cases"]], expected_names, "case identities")
        _equal([case["metadata"]["total_wires"] for case in report["cases"]], wires, "case widths")
        _equal([case["metadata"]["source_operations"] for case in report["cases"]], counts, "primitive ledger")
        for case in report["cases"]:
            _equal(case["numerical_checks_performed"], folder == "grid", "generation versus numerical scope")
            _equal(case["selected_auxiliary_state"], "all_zero", "selected auxiliary scope")
            _equal(case["standalone_initialization_performed"], False, "initialization excluded")
            _equal(case["measurement_or_sampling_performed"], False, "readout excluded")
            _equal(case["metadata"]["primitive_counts"], case["primitive_counts"], "ledger counts")
            _equal(sum(case["primitive_counts"].values()), case["metadata"]["source_operations"], "primitive count sum")
            if folder == "grid":
                _equal(case["check"]["checks_passed"], True, "historical bounded check")
                _equal(case["check"]["failed_checks"], [], "historical check failures")
            elif "check" in case:
                raise ValueError("Generation-only decay source has an unexpected numerical check")
        _false_flags(report)
        reports[folder] = report
    _equal(reports["grid"]["cases"][1]["check"]["actual_vs_exact_finite_frobenius_error"],
           0.09421942439042362, "retained tilted exact-H discrepancy")
    _equal(reports["grid"]["cases"][2]["check"]["actual_vs_exact_finite_frobenius_error"],
           0.038374474372646225, "retained complex exact-H discrepancy")
    return completion


def verify_bundle(bundle, *, source_root):
    """Authenticate saved identities and scope; perform no scientific replay."""
    bundle = _root(bundle)
    source_root = _root(source_root)
    data = _authenticate(bundle)
    _bind_sources(data, source_root)
    completion = _records(data)
    _notebooks(data, source_root)
    # Recheck the closed tree after all reads to detect additions/removals.
    _inventory(bundle)
    return {
        "status": "authenticated_saved_finite_decay_block",
        "manifest_sha256": MANIFEST_SHA256,
        "file_count": len(BUNDLE_FILES),
        "total_bytes": BUNDLE_TOTAL_BYTES,
        "historical_focused_tests": completion["focused_tests_passed"],
        "historical_notebook_code_cells": completion["notebook_code_cells"],
        "grid_cases": 3,
        "generation_only_cases": 1,
        "numerical_replay_performed": False,
        "physical_resources_computed": False,
        "scope": "saved byte/source/notebook identity and historical finite-block scope only",
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=root / BUNDLE_RELATIVE)
    parser.add_argument("--source-root", type=Path, default=root)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle, source_root=args.source_root), indent=2))


if __name__ == "__main__":
    main()
