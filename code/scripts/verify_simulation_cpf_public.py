#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the closed saved CPF2 packet without importing scientific code.

This reader checks saved bytes, case scope, source dependencies and notebook
images. It writes no files and performs no circuit generation, simulation,
numerical residual calculation, test run or notebook execution.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import zipfile

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_DISK_FILES = 122  # Manifest + 120 payloads + ZIP.
MAX_TREE_ENTRIES = 256
MAX_ARCHIVE_BYTES = 1024 * 1024
MAX_WALL_SECONDS = 10
PINS = {
    "manifest.json": (19579, "0e4136c0a60da714fb6bb3a7dbd76fe8acfbef2d0c96a2b9c53142d9a303979f"),
    "cpf2-public-packet.zip": (316251, "a5b718436503880cee05da0d841e747566b684292f998d8a6cee82fd9f369176"),
    "corrected_product_formula.py": (16947, "63fffc29105a2162ba5d0b37f8638738dad00161065734099b43654089839540"),
    "original_repo_producer.py": (10180, "c360f90abd24d78507173677e19b887a33f78f2f29a5e1e09ce6a175339a5576"),
    "generate_cpf2_public.py": (10503, "51bf0ead5b6fa14b2f8fe5e3365866e073507bd3c6e85d8fda33ce31d4ad88ec"),
    "requirements-lock.txt": (1974, "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"),
    "LICENSE.txt": (1093, "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"),
    "original-executed.ipynb": (176661, "fbddffe0c0e31005ed987914880439cadd13c99127c2e6b89abf80ec0517d5b8"),
    "portable_demo.ipynb": (3971, "e5117aa6e11c688b3fb8d8e5c858787c010e0c141d576e1f18e37ee8c0321f4b"),
    "one-pair-complete-trial.png": (58268, "d5864d28b49c14dcc3eeaecdbe95df18550bc03a16128b1d033de02d9611a918"),
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def relative(name):
    require(isinstance(name, str) and 0 < len(name) <= 256, "invalid bounded member name")
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
            "member name must be an unprefixed canonical relative path")
    return path


def directory(path):
    """Open every ancestor without following a symlink; caller closes the fd."""
    path = Path(path).absolute()
    require(".." not in path.parts, "saved input has an unsafe parent component")
    require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
            "saved verification requires no-follow directory access")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(path.anchor, flags)
    try:
        for name in path.parts[1:]:
            next_fd = os.open(name, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


class Reader:
    def __init__(self):
        self.started = time.monotonic()
        self.deadline = self.started + MAX_WALL_SECONDS
        self.total = 0
        self.files = 0

    def tick(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("saved-only check crossed its ten-second envelope")

    def account(self, raw):
        self.total += len(raw)
        require(self.total <= MAX_TOTAL_BYTES, "aggregate saved-read bound exceeded")
        self.tick()
        return raw

    def read(self, path, limit=MAX_FILE_BYTES):
        self.tick()
        self.files += 1
        require(self.files <= MAX_DISK_FILES, "finite saved-file inventory exceeded")
        path = Path(path)
        parent = directory(path.parent)
        try:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                info = os.fstat(fd)
                require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 <= info.st_size <= limit,
                        "saved input must be a bounded unlinked regular file")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    raw = stream.read(limit + 1)
                after = os.fstat(fd)
                before_identity = (info.st_dev, info.st_ino, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                after_identity = (after.st_dev, after.st_ino, after.st_nlink, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                require(before_identity == after_identity and len(raw) == info.st_size and len(raw) <= limit,
                        "saved file identity or size changed while reading")
            finally:
                os.close(fd)
        finally:
            os.close(parent)
        return self.account(raw)

    def inventory(self, packet):
        """Bound entries while streaming, before adding them to the inventory."""
        actual, directories, entries, pending = set(), set(), 0, [Path(packet)]
        while pending:
            self.tick()
            path = pending.pop()
            fd = directory(path)
            try:
                with os.scandir(fd) as children:
                    for item in children:
                        self.tick()
                        entries += 1
                        require(entries <= MAX_TREE_ENTRIES, "packet tree inventory bound exceeded")
                        info = item.stat(follow_symlinks=False)
                        child = path / item.name
                        relative(child.relative_to(packet).as_posix())
                        if stat.S_ISREG(info.st_mode):
                            require(info.st_nlink == 1, "packet contains a hard-linked file")
                            actual.add(child.relative_to(packet).as_posix())
                        elif stat.S_ISDIR(info.st_mode):
                            directories.add(child.relative_to(packet).as_posix())
                            pending.append(child)
                        else:
                            raise ValueError("packet contains a symlink or non-regular entry")
            finally:
                os.close(fd)
        return actual, directories


def source(cell):
    value = cell["source"]
    require(isinstance(value, str) or isinstance(value, list) and all(isinstance(item, str) for item in value),
            "invalid notebook source representation")
    return value if isinstance(value, str) else "".join(value)


def imports(tree):
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            require(node.level == 0, "unexpected relative scientific dependency")
            result.add(node.module)
    return result


def literal_assignments(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def check_model(row, expected, simulate):
    pairs, a, b, alpha, duration, repetitions = expected
    model = row["model"]
    require((model["pairs"], model["a_exact"], model["b_exact"], model["alpha_exact"],
             row["time"], row["repetitions"]) == (pairs, a, b, alpha, duration, repetitions),
            "saved fixed-case parameters differ")
    require(row["simulation_performed"] is simulate and set(row["ledgers"]) == {"strang", "cpf2"},
            "saved numerical/generation scope or paired ledger differs")
    require(isinstance(row["comparison"], dict) if simulate else row["comparison"] is None,
            "saved comparison marker differs")
    if simulate:
        require(set(row["comparison"]["methods"]) == {"strang", "cpf2"}, "saved comparison lacks paired methods")


def verify(bundle):
    """Verify saved packet evidence only and return its bounded scope summary."""
    packet = Path(bundle).absolute()
    reader = Reader()
    manifest_raw = reader.read(packet / "manifest.json")
    require((len(manifest_raw), digest(manifest_raw)) == PINS["manifest.json"], "reviewed manifest bytes differ")
    manifest = json.loads(manifest_raw)
    require(manifest["schema"] == "openquantum.cpf2.public-packet.v1"
            and manifest["status"] == "saved-original-evidence; portable-sources-prepared-unexecuted"
            and len(manifest["files"]) == 120 and manifest["payload_bytes"] == 1309324,
            "reviewed 120-payload packet metadata differs")
    expected = {row["path"]: row for row in manifest["files"]}
    require(len(expected) == 120, "duplicate reviewed payload names")
    allowed_directories = {
        parent.as_posix()
        for name in expected for parent in relative(name).parents
        if parent != Path(".")
    }
    actual, actual_directories = reader.inventory(packet)
    require(actual == set(expected) | {"manifest.json", "cpf2-public-packet.zip"}, "packet has missing or unreviewed files")
    require(actual_directories == allowed_directories, "packet has missing or unreviewed directories")
    payload = {}
    for name, pin in expected.items():
        raw = reader.read(packet / relative(name))
        require(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "reviewed public payload bytes differ")
        if name in PINS:
            require((len(raw), digest(raw)) == PINS[name], "fixed original/source evidence pin differs")
        payload[name] = raw
    archive_raw = reader.read(packet / "cpf2-public-packet.zip", MAX_ARCHIVE_BYTES)
    require((len(archive_raw), digest(archive_raw)) == PINS["cpf2-public-packet.zip"], "reviewed archive bytes differ")
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as bundle:
        members = bundle.infolist()
        require(len(members) == 121 and len({item.filename for item in members}) == 121,
                "archive must contain exactly 121 unique unprefixed entries")
        require({item.filename for item in members} == set(expected) | {"manifest.json"}, "archive membership differs")
        require(sum(item.file_size for item in members) <= 2 * 1024 * 1024, "aggregate archive expansion bound exceeded")
        for item in members:
            reader.tick()
            relative(item.filename)
            require(not item.is_dir() and stat.S_ISREG(item.external_attr >> 16)
                    and item.file_size <= MAX_FILE_BYTES and item.flag_bits & 1 == 0,
                    "archive entry is not a bounded ordinary saved file")
            raw = reader.account(bundle.read(item))
            require(raw == (manifest_raw if item.filename == "manifest.json" else payload[item.filename]),
                    "archive member differs from reviewed served bytes")
    expected_cases = [(m, ["1"] * m, ["1"] * m, alpha, "1", repetitions)
                      for m in (1, 2) for alpha in ("1", "1/5", "1/20") for repetitions in (1, 2, 4, 8)]
    expected_cases += [(m, ["1"] * m, ["1"] * m, "1/5", "-1", repetitions)
                       for m in (1, 2) for repetitions in (1, 4)]
    expected_cases += [(m, ["1"] * m, ["1"] * m, "1/5", "0", 1) for m in (1, 2)]
    expected_cases += [(1, ["1"], ["1"], "0", "1", 4),
                       (2, ["1", "1/2"], ["1", "-3/4"], "1/5", "1", 2)]
    study = json.loads(payload["saved-study.json"])
    require(len(study["cases"]) == 32 and study["physical_estimation_performed"] is False
            and study["sampling_performed"] is False, "saved study scope differs")
    for index, (row, fixed) in enumerate(zip(study["cases"], expected_cases)):
        reader.tick()
        require(row["case_index"] == index and json.loads(payload[f"cases/case-{index:02d}/record.json"]) == row,
                "saved original case and study objects differ")
        check_model(row, fixed, True)
        for method in ("strang", "cpf2"):
            require(payload[f"cases/case-{index:02d}/{method}.qasm"].startswith(b"OPENQASM 3.0;"),
                    "saved paired circuit is not the reviewed QASM export")
    generations = json.loads(payload["saved-generation.json"])
    require(generations["simulation_performed"] is False and len(generations["rows"]) == 3,
            "larger saved generation scope differs")
    for outer, pairs in zip(generations["rows"], (4, 16, 32)):
        require(outer["pairs"] == pairs and len(outer["cases"]) == 1, "saved generation vector differs")
        row = outer["cases"][0]
        check_model(row, (pairs, ["1"] * pairs, ["1"] * pairs, "1/5", "1", 8), False)
        require(json.loads(payload[f"generation/m{pairs}/record.json"]) == row, "saved generation record differs")
        for method in ("strang", "cpf2"):
            require(payload[f"generation/m{pairs}/{method}.qasm"].startswith(b"OPENQASM 3.0;"), "saved larger QASM differs")
    book = json.loads(payload["original-executed.ipynb"])
    codes = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    require(len(codes) == 5 and [cell["execution_count"] for cell in codes] == [1, 2, 3, 4, 5],
            "genuine notebook lacks its exact five completion counts")
    images = []
    for cell in codes:
        ast.parse(source(cell))
        for output in cell["outputs"]:
            require(output["output_type"] != "error", "genuine notebook contains an error output")
            encoded = output.get("data", {}).get("image/png")
            if encoded is not None:
                raw = base64.b64decode(encoded if isinstance(encoded, str) else "".join(encoded), validate=True)
                require(raw == payload["one-pair-complete-trial.png"], "saved PNG differs from genuine notebook image")
                images.append((cell["id"], len(raw), digest(raw)))
    require(images == [("cpf-circuit", *PINS["one-pair-complete-trial.png"])] * 2,
            "genuine notebook PNG correspondence differs")
    portable = json.loads(payload["portable_demo.ipynb"])
    portable_codes = [cell for cell in portable["cells"] if cell["cell_type"] == "code"]
    require(len(portable_codes) == 5 and all(cell["execution_count"] is None and cell["outputs"] == [] for cell in portable_codes),
            "adapted portable notebook inherited execution evidence")
    require(portable["metadata"]["openquantum"]["status"] == "prepared-unexecuted-portable-companion",
            "portable notebook status differs")
    for cell in portable_codes:
        ast.parse(source(cell))
    require("wrapper.load_science(packet)" in source(portable_codes[0])
            and all(source(current) == source(original) for current, original in zip(portable_codes[1:], codes[1:])),
            "prepared portable bootstrap or unchanged scientific cell sources differ")
    module_tree = ast.parse(payload["corrected_product_formula.py"])
    wrapper_tree = ast.parse(payload["generate_cpf2_public.py"])
    require(imports(module_tree) == {"__future__", "dataclasses", "fractions", "math", "numbers", "re", "typing",
                                    "numpy", "qiskit", "qiskit.quantum_info", "scipy.linalg"}, "immutable module static dependency closure differs")
    require(imports(wrapper_tree) == {"__future__", "argparse", "hashlib", "importlib.metadata", "importlib.util",
                                     "json", "math", "pathlib", "sys", "time", "qiskit"}, "public wrapper static dependency closure differs")
    caps = literal_assignments(module_tree)
    for name, value in {"MAX_PAIRS": 32, "MAX_REPETITIONS": 1024, "MAX_SOURCE_OPERATIONS": 100000,
                        "MAX_DENSE_PAIRS": 2, "MAX_DENSE_REPETITIONS": 16, "MAX_DENSE_SOURCE_OPERATIONS": 2000}.items():
        require(caps[name] == value, "immutable module static bounds differ")
    wrapper_text = payload["generate_cpf2_public.py"].decode()
    wrapper_literals = literal_assignments(wrapper_tree)
    require(wrapper_literals["MODULE_SHA256"] == PINS["corrected_product_formula.py"][1]
            and wrapper_literals["LOCK_SHA256"] == PINS["requirements-lock.txt"][1]
            and wrapper_literals["LICENSE_SHA256"] == PINS["LICENSE.txt"][1]
            and wrapper_literals["RUNTIME_PINS"] == {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1"},
            "public wrapper static immutable-input/runtime pins differ")
    require("spec.loader.exec_module(module)" in wrapper_text and "sys.modules[spec.name] = module" in wrapper_text
            and "REFERENCE_TOLERANCE = 1e-11" in wrapper_text and "controls[name] <= 1e-6" in wrapper_text
            and "committed_sources" not in wrapper_text and "subprocess" not in wrapper_text and "docs/coordination" not in wrapper_text,
            "reviewed wrapper static bootstrap/guard or private-contract boundary differs")
    original_tree = ast.parse(payload["original_repo_producer.py"])
    require(literal_assignments(original_tree)["SOURCE_FILES"] == (
        "src/openquantum/simulation/corrected_product_formula.py", "scripts/generate_simulation_corrected_formula.py",
        "tests/test_simulation_corrected_formula.py", "notebooks/simulation_corrected_formula.ipynb",
        "requirements-lock.txt", "LICENSE"), "archived original producer repository dependency closure differs")
    reader.tick()
    return {"status": "passed", "scope": "saved bytes, JSON scope and AST/text only",
                      "payload_files": 120, "archive_entries": 121, "paired_small_cases": 32,
                      "generation_only_pair_sizes": [4, 16, 32], "genuine_notebook_code_cells": 5,
                      "genuine_png_outputs": 2, "portable_execution_performed": False,
                      "scientific_imports_or_calls": 0, "read_bytes": reader.total,
                      "observed_wall_seconds": time.monotonic() - reader.started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    require(sys.version_info[:3] == (3, 12, 14), "use the locked Python 3.12.14 interpreter")
    print(json.dumps(verify(args.bundle)))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)[:1024]}))
        raise SystemExit(1)
