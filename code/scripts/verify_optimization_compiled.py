#!/usr/bin/env python3
"""Check the immutable compiled QAOA replay and its saved-only re-export.

The completion producer's --verify-only path deterministically re-exports
derived files. This wrapper runs it only on a temporary flat copy, forbids new
synthesis/search/optimizer/sampling calls and compares all 301 files afterward.
Saved numerical and conditional model checks are not hardware certificates.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE_RELATIVE = "assets/generated/optimization/compiled-replay"
REVIEWED_MANIFEST_SHA256 = "a4e0bbd7e1063713bd4faa5af82ce2bed0a3c10201f99cbf0d53b61c507d34eb"
REVIEWED_INVENTORY_SHA256 = "3798687d2f3fad7294880dbb26d12d30aa08f3f7de8907175a862183b25de7f2"
ORIGINAL_INVENTORY_SHA256 = "b247290b5c333d94726f2dc7cb935741fae6fd497223f953612fda692cd3446a"
RUN_ID = "qaoa-cycle-4-p2-alternate-seed-7"
PACKAGES = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1",
            "matplotlib": "3.11.2", "qdk": "1.34.0", "jupyter-book": "2.1.7"}
CAPS = {"files": 301, "entries": 310, "bytes": 16 * 1024**2,
        "class_decoded_bytes": 4 * 1024**2, "decoded_total_bytes": 64 * 1024**2,
        "replay_seconds": 120}
COMPLETION_EXPORTS = frozenset({
    "artifact-size.json", "classes.csv", "classical-work.json", "complete_optimization_compiled.py",
    "completed-replay.json.gz", "completion-amendment.json", "completion-freeze.json", "completion-recipe.json",
    "completion-timing.json", "conditional-readout.json", "groups.csv", "manifest.json", "negative-controls.csv",
    "optimization_compiled_codec.py", "physical-dag.json.gz", "physical-summary.csv", "physical-summary.json",
    "resources.json", "summary.md", "workload.svg",
})
DIRECTORIES = frozenset({"checkpoints", "checkpoints/failures", "checkpoints/synthesis", "circuits", "timing"})
SOURCE_FILES = {
    "optimization_compiled.py": "src/openquantum/optimization_compiled.py",
    "optimization_compiled_codec.py": "src/openquantum/optimization_compiled_codec.py",
    "generate_optimization_compiled.py": "scripts/generate_optimization_compiled.py",
    "complete_optimization_compiled.py": "scripts/complete_optimization_compiled.py",
    "instrument_resources.py": "src/openquantum/instrument_resources.py",
    "requirements-lock.txt": "requirements-lock.txt", "LICENSE.txt": "LICENSE",
}
CANONICAL_SOURCES = {
    "parent_optimization.py": "assets/generated/optimization/diagnostics/optimization.py",
    "parent_generate_optimization_diagnostics.py": "assets/generated/optimization/diagnostics/generate_optimization_diagnostics.py",
    "references.bib": "assets/generated/instrument-resources/references.bib",
}
FIRST_COUNTS = {"top_level_calls": 49, "factory_frontier_queries": 1176,
                "factory_frontier_candidates": 12348, "factory_pipeline_candidates": 816, "native_trace_queries": 6174}
SECOND_COUNTS = {"top_level_calls": 49, "factory_frontier_queries": 1176,
                 "factory_frontier_candidates": 12348, "native_trace_queries": 6174}
CUMULATIVE_COUNTS = {"top_level_calls": 98, "factory_frontier_queries": 2352,
                     "factory_frontier_candidates": 24696, "factory_pipeline_candidates": 816, "native_trace_queries": 12348}
PRIVATE_MARKERS = (b"merlin-" + b"ssh-proxy-cn", b"BEGIN " + b"OPENSSH PRIVATE KEY",
                   b"BEGIN " + b"RSA PRIVATE KEY", b"BEGIN " + b"EC PRIVATE KEY", b"BEGIN " + b"DSA PRIVATE KEY")


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def equal(actual, expected, label):
    if isinstance(expected, (set, frozenset)):
        valid = isinstance(actual, (set, frozenset)) and actual == expected
    else:
        valid = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    if not valid:
        raise ValueError(f"Compiled replay {label} differs from the reviewed release")


def checked_path(root: Path, name: str, *, maximum: int | None = None) -> Path:
    maximum = CAPS["bytes"] if maximum is None else maximum
    if not isinstance(name, str):
        raise ValueError("Unsafe compiled-replay path")
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute()
            or ".." in relative.parts or relative.as_posix() != name):
        raise ValueError("Unsafe compiled-replay path")
    path = root / relative
    if any(candidate.is_symlink() for candidate in (root, *root.parents, *path.parents)):
        raise ValueError("Linked compiled-replay ancestor")
    if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > maximum:
        raise ValueError(f"Missing, linked, special or oversized compiled-replay file: {name}")
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def privacy_check(raw: bytes, label: str):
    if any(marker in raw for marker in PRIVATE_MARKERS):
        raise ValueError(f"Private connection/key content in compiled-replay {label}")


def json_bytes(raw: bytes, *, allow_list: bool = False):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate compiled-replay JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite compiled-replay JSON value")
    def finite(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            reject(value)
        return parsed
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=finite)
    if not isinstance(value, dict) and not (allow_list and isinstance(value, list)):
        raise ValueError("Compiled-replay JSON record must be an object")
    return value


def json_record(root: Path, name: str) -> dict:
    return json_bytes(checked_path(root, name).read_bytes())


def compressed_record(path: Path, maximum: int) -> tuple[dict, int]:
    with gzip.GzipFile(fileobj=io.BytesIO(path.read_bytes()), mode="rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Compiled-replay compressed JSON exceeds its decoded-byte cap")
    privacy_check(raw, path.name)
    return json_bytes(raw, allow_list=path.name == "source-classes.json.gz"), len(raw)


def inventory(bundle: Path) -> dict:
    bundle = Path(bundle).absolute()
    if bundle.is_symlink() or not bundle.is_dir() or any(path.is_symlink() for path in bundle.parents):
        raise ValueError("Compiled-replay bundle must be a canonical regular directory")
    files, directories, total, entries = {}, set(), 0, 0
    for current, folders, names in os.walk(bundle, followlinks=False):
        for name in sorted(folders + names):
            path = Path(current) / name
            relative = path.relative_to(bundle).as_posix()
            entries += 1
            if entries > CAPS["entries"]:
                raise ValueError("Compiled-replay tree exceeds its entry cap")
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
            elif stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
                if len(files) >= CAPS["files"] or metadata.st_size > CAPS["bytes"] or total > CAPS["bytes"]:
                    raise ValueError("Compiled-replay tree exceeds its file/byte cap")
                files[relative] = sha(path)
            else:
                raise ValueError("Compiled-replay tree contains a linked or special entry")
    equal(directories, DIRECTORIES, "directory inventory")
    equal(len(files), 301, "complete file count")
    equal(total, 9614924, "complete bundle bytes")
    return {"file_sha256": dict(sorted(files.items())), "directories": sorted(directories), "bytes": total}


def verify_records(manifest: dict, frozen: dict, completion: dict, timing: dict, size: dict) -> None:
    """Semantic contract checked independently of the immutable byte hashes."""
    try:
        for record in (manifest, frozen, completion, timing, size):
            equal(record["schema_version"], 1, "schema")
        for key, expected in (("status", "complete"), ("replay_status", "computed_actual_compiled_fixed_replay_conditional_physical_model"),
                              ("recipe_id", "optimization-qaoa-actual-compiled-replay-v1"), ("run_id", RUN_ID),
                              ("sampled_groups", 110), ("preparations", 82032),
                              ("public_reader_requires_private_pickle", False),
                              ("old_failure_file", "checkpoints/failures/physical-workload.json")):
            equal(manifest[key], expected, key)
        equal(frozen["status"], "recipe_and_producers_frozen_before_compilation", "original freeze status")
        equal(frozen["execution_performed"], False, "original freeze execution")
        equal(completion["status"], "completion_recipe_and_producers_frozen_before_physical_recovery", "completion freeze status")
        equal(completion["native_or_synthesis_executed"], False, "completion freeze execution")
        equal(manifest["identity"], completion["identity"], "completion identity")
        equal(manifest["original_identity"], frozen["identity"], "original identity")
        equal(completion["identity"]["original_identity"], frozen["identity"], "recovery original identity")
        source_hashes = dict(frozen["identity"]["source_sha256"])
        source_hashes.update(completion["identity"]["source_sha256"])
        equal(manifest["provenance"]["source_sha256"], source_hashes, "recorded source provenance")
        equal(manifest["provenance"]["python"], "3.12.14", "recorded interpreter")
        equal(manifest["provenance"]["packages"], PACKAGES, "recorded packages")
        equal(timing["identity"], completion["identity"], "timing identity")
        equal(timing["status"], "complete", "completion timing status")
        equal(timing["error_type"], None, "completion timing error")
        equal(timing["first_query_counts"], FIRST_COUNTS, "first native counters")
        equal(timing["second_query_counts"], SECOND_COUNTS, "saved recovery native counters")
        equal(manifest["recovery"]["first_query_counts"], FIRST_COUNTS, "manifest first counters")
        equal(manifest["recovery"]["second_query_counts"], SECOND_COUNTS, "manifest second counters")
        equal(manifest["recovery"]["cumulative_query_counts"], CUMULATIVE_COUNTS, "cumulative counters")
        for key, value in (("roundtrip_exact_to_second_live_value", True), ("first_complete_result_unavailable", True),
                           ("quantum_preparations_counted_once", 82032)):
            equal(manifest["recovery"][key], value, "recovery " + key)
        equal(manifest["recovery"]["codec_statistics"], {"nodes": 92847, "references": 760148, "maximum_depth": 10,
              "encoded_bytes": 12531648, "logical_sha256": "c84c1c85514d7620482a2fb25f6e2d2c772a6a6bb6f9ceff207e2e31adc2904b"}, "typed transport identity")
        equal(manifest["completed_replay"], {"file": "completed-replay.json.gz", "sha256": "91fe2c423d66e682bcfba4b78142fa4192a60bf02609d4e6b526b946259d3c53"}, "completed transport")
        equal(manifest["physical_storage"], {"encoding": "json-content-dag-v1", "file": "physical-dag.json.gz",
              "sha256": "731791448fcfd3e7fc2df18efa0d7d0db12c49254a709bb58d0286a93d1cca34"}, "physical transport")
        equal(manifest["timing"]["completion"], timing, "manifest timing record")
        equal(manifest["timing"]["synthesis"]["status"], "complete", "historical synthesis completion")
        equal(manifest["timing"]["original_physical"]["status"], "incomplete", "preserved physical failure")
        for key in ("first_wall_seconds", "recovery_wall_seconds", "cumulative_wall_seconds", "parent_cpu_seconds"):
            value = timing[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 120:
                raise ValueError("Compiled-replay timing exceeds its recorded cap")
        if not math.isclose(timing["first_wall_seconds"] + timing["recovery_wall_seconds"], timing["cumulative_wall_seconds"], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Compiled-replay cumulative timing is inconsistent")
        equal(timing["wall_cap_seconds"], 120, "physical wall cap")
        equal(size["files"], 301, "size record count")
        equal(size["total_bytes"], 9614924, "size record bytes")
        equal(size["frozen_byte_cap"], 16 * 1024**2, "frozen artifact cap")
        equal(size["parent_archive_copied"], False, "parent archive copy scope")
        equal(size["private_cache_included"], False, "private cache scope")
        equal(len(manifest["classes"]), 54, "all source classes")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Compiled-replay identity/status record is incomplete or malformed") from error


def preflight(bundle: Path, source_root: Path, canonical_root: Path, before: dict) -> dict:
    files = before["file_sha256"]
    equal(files.get("manifest.json"), REVIEWED_MANIFEST_SHA256, "immutable manifest")
    equal(hashlib.sha256(encoded(files)).hexdigest(), REVIEWED_INVENTORY_SHA256, "all-301-file inventory")
    manifest, frozen, completion, timing, size = (json_record(bundle, name) for name in (
        "manifest.json", "freeze.json", "completion-freeze.json", "completion-timing.json", "artifact-size.json"))
    verify_records(manifest, frozen, completion, timing, size)
    original = completion["original_artifact_inventory"]
    if not isinstance(original, dict):
        raise ValueError("Compiled-replay original inventory must be an object")
    equal(len(original), 281, "preserved first-attempt files")
    equal(set(files), set(original) | COMPLETION_EXPORTS, "original and completion file inventory")
    equal({name: files[name] for name in original}, original, "preserved original bytes")
    anchor = {BUNDLE_RELATIVE + "/" + name: digest for name, digest in sorted(original.items())}
    equal(hashlib.sha256(encoded(anchor)).hexdigest(), ORIGINAL_INVENTORY_SHA256, "original attempt anchor")
    sources = dict(frozen["identity"]["source_sha256"])
    sources.update(completion["identity"]["source_sha256"])
    equal(set(sources), set(SOURCE_FILES) | set(CANONICAL_SOURCES), "complete source inventory")
    for name, relative in SOURCE_FILES.items():
        equal(sha(checked_path(source_root, relative)), sources[name], "current source " + relative)
        equal(files[name], sources[name], "snapshot source " + name)
    for name, relative in CANONICAL_SOURCES.items():
        equal(sha(checked_path(canonical_root, relative)), sources[name], "canonical source " + relative)
        equal(files[name], sources[name], "snapshot source " + name)
    equal(sha(checked_path(canonical_root, "assets/generated/instrument-resources/instrument_resources.py")), sources["instrument_resources.py"], "frozen instrument snapshot")
    for key, relative in (
        ("parent_raw_sha256", "assets/generated/optimization/diagnostics/raw-study.json"),
        ("parent_recipe_sha256", "assets/generated/optimization/diagnostics/recipe.json"),
        ("selected_checkpoint_sha256", f"assets/generated/optimization/diagnostics/checkpoints/sampled-{RUN_ID}.json")):
        maximum = 32 * 1024**2 if key == "parent_raw_sha256" else CAPS["bytes"]
        equal(sha(checked_path(canonical_root, relative, maximum=maximum)), manifest["bindings"][key], "parent binding " + key)
    compressed = [name for name in files if name.endswith(".json.gz")]
    equal(len(compressed), 57, "exact compressed JSON count")
    decoded = 0
    for name in compressed:
        maximum = CAPS["decoded_total_bytes"] if name == "physical-dag.json.gz" else CAPS["class_decoded_bytes"]
        _, count = compressed_record(checked_path(bundle, name), maximum)
        decoded += count
        if decoded > CAPS["decoded_total_bytes"]:
            raise ValueError("Compiled-replay compressed records exceed the aggregate decoded-byte cap")
    for name in files:
        if name.endswith((".py", ".json", ".md", ".txt", ".bib", ".csv", ".qasm")):
            privacy_check(checked_path(bundle, name).read_bytes(), name)
    return {"files": 301, "bytes": before["bytes"], "manifest_sha256": REVIEWED_MANIFEST_SHA256,
            "inventory_sha256": REVIEWED_INVENTORY_SHA256, "original_files_preserved": 281,
            "compressed_json_files": 57, "decoded_json_bytes": decoded}


def guard_module(module, counters):
    categories = {
        "synthesis_calls": ("compile_source_class", "synthesize_instrument", "gridsynth_unitary"),
        "native_estimator_calls": ("run_compiled_physical", "run_seventh_physical_workload", "estimate_instrument", "physical_worker"),
        "optimizer_calls": ("pattern_search", "sampled_pattern_search", "run_demo", "run_sampled_demo", "run_diagnostic_control"),
        "sampling_calls": ("_sample", "_sample_measured_scalar", "sample_qaoa", "sample_vqe", "sample_hypercube_walk",
                           "sampled_qaoa_objective", "sampled_vqe_objective", "sampled_qaoa_layers_objective",
                           "sampled_spin_vqe_objective", "run_diagnostic_sampled"),
    }
    for category, names in categories.items():
        for name in names:
            if hasattr(module, name):
                def forbidden(*args, _category=category, **kwargs):
                    counters[_category] += 1
                    raise ValueError("New computation is forbidden during saved compiled-replay verification")
                setattr(module, name, forbidden)
    if hasattr(module, "_load_module"):
        original = module._load_module
        def guarded_load(*args, **kwargs):
            result = original(*args, **kwargs)
            guard_module(result, counters)
            return result
        module._load_module = guarded_load


def reexport_copy(bundle: Path, before: dict) -> dict:
    for name, version in PACKAGES.items():
        equal(importlib.metadata.version(name), version, "installed " + name)
    equal(".".join(map(str, sys.version_info[:3])), "3.12.14", "installed Python")
    counters = dict(synthesis_calls=0, native_estimator_calls=0, optimizer_calls=0, sampling_calls=0)
    with tempfile.TemporaryDirectory(prefix="openquantum-compiled-saved-") as temporary:
        work = Path(temporary).resolve()
        copied = work / "bundle"
        shutil.copytree(bundle, copied)
        equal(inventory(copied), before, "temporary initial copy")
        path = checked_path(copied, "complete_optimization_compiled.py")
        spec = importlib.util.spec_from_file_location("_guarded_compiled_completion", path)
        completion = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = completion
        exec(compile(path.read_bytes(), str(path), "exec"), completion.__dict__)
        original_load = completion.load_module
        def guarded_load(*args, **kwargs):
            module = original_load(*args, **kwargs)
            guard_module(module, counters)
            return module
        completion.load_module = guarded_load
        guard_module(completion, counters)
        result = completion.complete(copied, work / "private-cache", verify_only=True)
        equal(result["status"], "complete_lossless_physical_recovery", "saved re-export completion")
        equal(result["compiled_synthesis_repeated"], False, "saved synthesis repetition")
        equal(result["new_optimizer_runs"], 0, "saved optimizer repetition")
        equal(inventory(copied), before, "all-301-byte deterministic re-export")
        equal(counters, dict.fromkeys(counters, 0), "forbidden computation counters")
    equal(inventory(bundle), before, "canonical original preserved after replay")
    return {"temporary_reexport_files_compared": 301, "verify_only_reexports_derived_files": True, **counters}


def verify_worker_receipt(worker: dict, expected: dict) -> None:
    mandatory = {"temporary_reexport_files_compared": 301, "verify_only_reexports_derived_files": True,
                 "synthesis_calls": 0, "native_estimator_calls": 0, "optimizer_calls": 0, "sampling_calls": 0,
                 "saved_reexport_run": True}
    equal(set(worker), set(expected) | set(mandatory) | {"verification_wall_seconds"}, "worker receipt fields")
    for key, value in {**expected, **mandatory}.items():
        equal(worker[key], value, "worker receipt " + key)
    value = worker["verification_wall_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= CAPS["replay_seconds"]:
        raise ValueError("Compiled-replay worker receipt has invalid elapsed time")


def verify_bundle(bundle: Path, *, source_root: Path | None = None, canonical_root: Path | None = None,
                  replay: bool = False) -> dict:
    bundle = Path(bundle).absolute()
    source_root = bundle.parents[3] if source_root is None else Path(source_root).absolute()
    canonical_root = source_root if canonical_root is None else Path(canonical_root).absolute()
    before = inventory(bundle)
    receipt = preflight(bundle, source_root, canonical_root, before)
    if replay:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--bundle", str(bundle),
                   "--source-root", str(source_root), "--canonical-root", str(canonical_root)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, env=environment,
                                    timeout=CAPS["replay_seconds"], check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            equal(inventory(bundle), before, "canonical original preserved after failed replay")
            raise ValueError("Bounded saved compiled-replay worker failed or exceeded its 120-second cap") from error
        try:
            worker = json_bytes(result.stdout.encode())
            verify_worker_receipt(worker, receipt)
        finally:
            equal(inventory(bundle), before, "canonical original preserved after worker")
        receipt.update(worker)
    else:
        receipt["saved_reexport_run"] = False
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--identity-only", action="store_true", help="Check saved identities/content only; do not re-export a temporary copy")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    bundle = args.bundle or args.root / BUNDLE_RELATIVE
    source = args.source_root or args.root
    canonical = args.canonical_root or args.root
    started = time.monotonic()
    if args.worker:
        before = inventory(bundle)
        receipt = preflight(bundle, source, canonical, before)
        receipt.update(reexport_copy(bundle, before), saved_reexport_run=True)
    else:
        receipt = verify_bundle(bundle, source_root=source, canonical_root=canonical, replay=not args.identity_only)
    receipt["verification_wall_seconds"] = time.monotonic() - started
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
