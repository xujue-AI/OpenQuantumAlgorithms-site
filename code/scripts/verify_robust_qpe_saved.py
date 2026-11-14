#!/usr/bin/env python3
"""Authenticate saved robust QPE evidence without sampling or numerical replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/robust-qpe"
MANIFEST_SHA256 = "2baf63d0fd410c47df031f50f75f807dc90a85193df27617447bb5f16893d259"
SOURCE_HASHES = {
    "src/openquantum/robust_qpe.py": "cba515031c6c422e197bfc7776eccd9a0cfc46baa4904ab85585d0b30026b39b",
    "src/openquantum/hadamard_qpe.py": "2f0571046d4001af592cb07ce08993290bbdcd14da230315cd3f9722ac881bb5",
    "scripts/generate_robust_qpe.py": "33f63a014d0c6fc55d8978f812ec90cfd53dbd1685227e3d6cb84fe02547b266",
    "tests/test_robust_qpe.py": "5588cdf6dfba2da60c55dc278db885ef6e1a90e97ae6f2a5a3e09e4611a826b1",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
SNAPSHOTS = {"robust_qpe.py": "src/openquantum/robust_qpe.py",
             "hadamard_qpe.py": "src/openquantum/hadamard_qpe.py",
             "generate_robust_qpe.py": "scripts/generate_robust_qpe.py",
             "requirements-lock.txt": "requirements-lock.txt", "LICENSE.txt": "LICENSE"}
BUNDLE_FILES = frozenset({"manifest.json", *SNAPSHOTS, "circuit.svg", "recovery.svg",
                          "results.json", "software.json", "scaling.csv", "summary.md",
                          "circuits/large-n8-sin-k32.qasm"}) | frozenset(
    f"circuits/phase-{phase}-{basis}-k{power}.qasm"
    for phase in range(4) for basis in ("cos", "sin") for power in (1, 2, 4, 8, 16, 32))
NOTEBOOK = "notebooks/robust_qpe.ipynb"
NOTEBOOK_SOURCE_SHA256 = "c64a68bee39ee58d606e62c145ffecd4d1060eb98e3711c9edb3edea83af86b8"
NOTEBOOK_LABELS = {"robust-qpe-circuit-cell": "robust-qpe-circuit",
                   "robust-qpe-recovery-cell": "robust-qpe-recovery"}
MAX_FILE_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 2 * 1024 * 1024


def _root(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path != path.resolve():
        raise ValueError("Unsafe robust QPE root")
    return path


def _read(root, name, limit=MAX_FILE_BYTES):
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name):
        raise ValueError("Unsafe robust QPE member path")
    path = root / relative
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Missing or linked robust QPE member: " + name)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > limit:
        raise ValueError("Unsafe or oversized robust QPE member: " + name)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != metadata.st_size or len(raw) > limit:
        raise ValueError("Robust QPE member changed during read: " + name)
    return raw


def notebook_source_sha256(raw):
    """Preserve ordered cell sources and labels while allowing output refresh."""
    document = json.loads(raw)
    cells = document.get("cells")
    if document.get("nbformat") != 4 or not isinstance(cells, list) or len(cells) != 7:
        raise ValueError("Robust QPE notebook cell contract changed")
    projection, code = [], []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Invalid robust QPE notebook source cell")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if (not isinstance(source, str) or not isinstance(cell.get("id"), str)
                or cell.get("cell_type") not in ("code", "markdown", "raw")):
            raise ValueError("Invalid robust QPE notebook source cell")
        projection.append({"id": cell["id"], "cell_type": cell["cell_type"], "source": source})
        metadata = cell.get("metadata", {})
        if cell["id"] in NOTEBOOK_LABELS and (not isinstance(metadata, dict) or metadata.get("label") != NOTEBOOK_LABELS[cell["id"]]):
            raise ValueError("Robust QPE canonical notebook label changed")
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            if not isinstance(outputs, list) or any(not isinstance(output, dict) for output in outputs):
                raise ValueError("Invalid robust QPE notebook output")
            if any(output.get("output_type") == "error" for output in outputs):
                raise ValueError("Robust QPE notebook contains an error output")
            code.append(cell)
    if [cell.get("execution_count") for cell in code] != [1, 2, 3, 4, 5]:
        raise ValueError("Robust QPE notebook must retain five executed code cells")
    for cell, types in ((code[1], {"image/png", "image/svg+xml"}), (code[3], {"image/svg+xml"})):
        if not any(types & set(output.get("data", {})) for output in cell.get("outputs", [])):
            raise ValueError("Missing robust QPE notebook circuit/recovery output")
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _saved_results(contents):
    """Check retained counts and scope; do not recompute inference or bounds."""
    results = json.loads(contents["results.json"])
    parameters = results.get("parameters", {})
    expected = {"target_qubits": 1, "levels": 6, "phases": [0.3, 0.995, 0.125, 0.5], "seeds": [7, 19],
                "canonical_study": True, "shots_per_fringe": 448, "whole_study_failure_target_exact": "1/100",
                "per_output_failure_target_exact": "1/800", "additive_probability_bound_exact": "1/50",
                "readout_zero_to_one_exact": "3/200", "readout_one_to_zero_exact": "1/100"}
    if (results.get("schema") != "openquantum-robust-qpe-v1" or parameters != expected
            or results.get("provenance", {}).get("source_commit") != "eca0cf40133993369f655dd0477cb6be7732369e"
            or results.get("physical_estimate_performed") is not False
            or results.get("formal_proof_reproduced") is not False):
        raise ValueError("Saved robust QPE result scope changed")
    runs = results.get("runs", [])
    if [(run.get("case_id"), run.get("seed")) for run in runs] != [(f"phase-{i}", seed) for i in range(4) for seed in (7, 19)]:
        raise ValueError("Saved robust QPE prescribed outputs changed")
    for index, run in enumerate(runs):
        counts = run["result"]["counts"]
        if ([(row["power"], row["basis"]) for row in counts] != [(power, basis) for power in (1, 2, 4, 8, 16, 32) for basis in ("cos", "sin")]
                or any(row["shots"] != 448 or type(row["zeros"]) is not int or not 0 <= row["zeros"] <= 448 for row in counts)
                or run["stream"] != index // 2 or len(run["result"]["trace"]) != 6
                or run["result"]["phase_radius_exact"] != "1/384"
                or run["result"]["physical_confidence_certified"] is not False):
            raise ValueError("Saved robust QPE count batches or confidence scope changed")
    ledger = results["common_per_output_ledger"]
    per_output = {"p": 169344, "cx": 112896, "h": 10752, "x": 5376, "sdg": 2688, "reset": 10752, "measure": 5376}
    whole = results["whole_study_modeled_workload"]
    if (ledger["source_gate_counts_all_trials"] != per_output or ledger["complete_fresh_preparations"] != 5376
            or ledger["sampling_tolerance_exact"] != "21/200" or ledger["phase_radius_exact"] != "1/384"
            or ledger["readout_bits"] != 5376 or ledger["shot_weighted_controlled_U_calls"] != 56448
            or whole["fresh_preparations"] != 43008 or whole["controlled_base_U_calls"] != 451584
            or whole["source_gate_counts"] != {gate: 8*count for gate, count in per_output.items()}
            or results["simulation"]["modeled_fresh_trials_sampled"] != 43008
            or results["outcomes"] != {"within_radius": 8, "miss": 0, "inconclusive": 0}):
        raise ValueError("Saved robust QPE whole-study accounting changed")
    large = results["large_source_only_export"]
    if (large["logical_qubits"] != 9 or large["check"]["performed"] is not False
            or sum(large["source_gate_counts"].values()) != 1301
            or sum(large["lowered_gate_counts"].values()) != 1301):
        raise ValueError("Robust QPE larger circuit validation scope changed")
    if len(list(csv.DictReader(io.StringIO(contents["scaling.csv"].decode("utf-8"))))) != 36:
        raise ValueError("Robust QPE logical scaling row count changed")


def verify_saved(root, *, source_root=None, notebook_roots=None, provenance=None):
    """Verify existing public records; perform no new scientific execution."""
    root = _root(root)
    source_root = _root(root if source_root is None else source_root)
    notebook_roots = (source_root,) if notebook_roots is None else notebook_roots
    if not isinstance(notebook_roots, tuple) or len(notebook_roots) not in (1, 2):
        raise ValueError("One or two robust QPE notebook roots are required")
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if provenance is not None and (not isinstance(provenance, dict)
            or provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict)):
        raise ValueError("Complete robust QPE package provenance is required")
    for name, pin in SOURCE_HASHES.items():
        if hashlib.sha256(_read(source_root, name)).hexdigest() != pin or hashes is not None and hashes.get(name) != pin:
            raise ValueError("Promoted robust QPE source identity changed: " + name)
    for notebook_root in notebook_roots:
        raw = _read(_root(notebook_root), NOTEBOOK)
        if hashes is not None and hashlib.sha256(raw).hexdigest() != hashes.get(NOTEBOOK):
            raise ValueError("Refreshed robust QPE notebook differs from current package provenance")
        if notebook_source_sha256(raw) != NOTEBOOK_SOURCE_SHA256:
            raise ValueError("Robust QPE notebook source changed")
    directory = _root(root / BUNDLE)
    expected_members, actual_members = BUNDLE_FILES | {"circuits"}, set()
    for path in directory.rglob("*"):
        name = path.relative_to(directory).as_posix()
        if len(actual_members) >= len(expected_members) or name not in expected_members or path.is_symlink():
            raise ValueError("Robust QPE bundle contains unrecorded members")
        actual_members.add(name)
    if actual_members != expected_members:
        raise ValueError("Robust QPE bundle contains unrecorded members")
    _root(directory / "circuits")
    raw = _read(directory, "manifest.json", 64 * 1024)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Saved robust QPE manifest identity changed")
    manifest = json.loads(raw)
    if manifest.get("schema") != "openquantum-robust-qpe-files-v1" or set(manifest.get("file_sha256", {})) != BUNDLE_FILES - {"manifest.json"}:
        raise ValueError("Saved robust QPE manifest membership changed")
    contents, total = {}, 0
    for name, digest in manifest["file_sha256"].items():
        payload = _read(directory, name)
        total += len(payload)
        if total > MAX_BUNDLE_BYTES or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Saved robust QPE payload identity changed: " + name)
        contents[name] = payload
    source_hashes = {name: SOURCE_HASHES[source] for name, source in SNAPSHOTS.items()}
    if (json.loads(contents["results.json"])["provenance"]["source_sha256"] != source_hashes
            or any(hashlib.sha256(contents[name]).hexdigest() != pin for name, pin in source_hashes.items())):
        raise ValueError("Saved robust QPE source snapshots differ from promoted inputs")
    _saved_results(contents)
    return {"status": "saved_component_verified", "files": len(BUNDLE_FILES), "saved_study_outputs": 8,
            "saved_modeled_fresh_trials": 43008, "saved_literal_base_calls": 451584,
            "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256, "new_modeled_draws": 0,
            "new_quantum_execution": False, "physical_estimation_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_saved(args.root), indent=2))


if __name__ == "__main__":
    main()
