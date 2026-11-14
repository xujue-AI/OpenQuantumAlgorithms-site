#!/usr/bin/env python3
"""Verify the audited Bayesian QPE saved companion without new execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "src/openquantum/bayesian_qpe.py": "019345eac507bb2c6d3190fbfb622d11294f0aaffded03015399d9eb5297ce97",
    "scripts/generate_bayesian_qpe.py": "7664935974c74ac848a025ed8d85d3f685dbdd2a69ffb88a24556ce215149ef7",
    "tests/test_bayesian_qpe.py": "f316bdf2b900174bb8b59756b97bdab4520afd7c286e34947294eccf107e2271",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
SNAPSHOTS = {"bayesian_qpe.py": "src/openquantum/bayesian_qpe.py",
             "generate_bayesian_qpe.py": "scripts/generate_bayesian_qpe.py",
             "requirements-lock.txt": "requirements-lock.txt", "LICENSE.txt": "LICENSE"}
NOTEBOOK = "notebooks/bayesian_qpe.ipynb"
NOTEBOOK_SOURCE_SHA256 = "574a0ab8e76e2862d2bacdc523b252e7967bbd8a08cafe38e93102a886889c12"
EMBED_LABELS = frozenset("bayesian-qpe-" + name for name in ("circuit", "summary", "posterior"))
VALIDATION = "assets/generated/bayesian-qpe-validation/validation.json"
VALIDATION_SHA256 = "ec1f52a54a12c4151e4f8e6fa36803f1d74b743da75b59d5dd318f99853b9b02"
COMMON_FILES = frozenset({"manifest.json", "bayesian_qpe.py", "generate_bayesian_qpe.py",
                         "requirements-lock.txt", "LICENSE.txt", "README.md", "results.json"})
STUDY_FILES = COMMON_FILES | {"demo.qasm", "circuit.svg", "large-n8-k32.qasm", "summary.csv", "posterior.svg"}
PHASE_EXPERIMENTS = (
    ((32, 0), (16, 1), (32, 1), (8, 0), (4, 1), (2, 1), (1, 1), (8, 1), (2, 0)),
    ((32, 0), (16, 0), (8, 0), (4, 0), (2, 0), (1, 0), (32, 1), (16, 1)),
    ((32, 0), (16, 0), (8, 0), (4, 0), (2, 1), (32, 1), (1, 1), (1, 0)),
    ((32, 0), (16, 0), (8, 0), (4, 0), (2, 0), (1, 0), (32, 1)),
)


def experiment_files(phases):
    return frozenset(f"phase{index}-k{power}-offset{offset}{suffix}.qasm"
                     for index, experiments in enumerate(phases) for power, offset in experiments
                     for suffix in ("", "-lowered"))


BUNDLE_FILES = {
    "assets/generated/bayesian-qpe": STUDY_FILES | experiment_files(PHASE_EXPERIMENTS),
    "assets/generated/bayesian-qpe-validation/source32": COMMON_FILES | {"source-only.qasm", "source-only-lowered.qasm"},
    "assets/generated/bayesian-qpe-validation/replay": STUDY_FILES | experiment_files(PHASE_EXPERIMENTS[:1]),
}
MANIFEST_HASHES = {
    "assets/generated/bayesian-qpe": "1c1744f03972d1340b7e13bfa89d252a484ae51c7a160ce3ac3db5eaadb595ad",
    "assets/generated/bayesian-qpe-validation/source32": "b46db9a7d1494ce1c8cbe82edad9724801e49a71723d64b3be2bee9d6d490776",
    "assets/generated/bayesian-qpe-validation/replay": "bbda8cd67eff72cf6afd45a5093e363e8f855415bd5ca77bb8ec05940a9b4670",
}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 16 * 1024 * 1024


def _root(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path != path.resolve():
        raise ValueError("Unsafe Bayesian QPE root")
    return path


def _read(root, name, limit=MAX_FILE_BYTES):
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name):
        raise ValueError("Unsafe Bayesian QPE member path")
    path = root / relative
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Missing or linked Bayesian QPE member: " + name)
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > limit:
        raise ValueError("Unsafe or oversized Bayesian QPE member: " + name)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) != metadata.st_size or len(raw) > limit:
        raise ValueError("Bayesian QPE member changed during read: " + name)
    return raw


def notebook_source_sha256(raw):
    """Bind ordered sources and executed output scope while allowing refresh."""
    notebook = json.loads(raw)
    cells = notebook.get("cells")
    if notebook.get("nbformat") != 4 or not isinstance(cells, list) or len(cells) != 10:
        raise ValueError("Bayesian QPE notebook cell contract changed")
    projection, code = [], []
    for cell in cells:
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if (not isinstance(source, str) or not isinstance(cell.get("id"), str)
                or cell.get("cell_type") not in ("code", "markdown", "raw")):
            raise ValueError("Invalid Bayesian QPE notebook cell")
        projection.append({"id": cell["id"], "cell_type": cell["cell_type"], "source": source})
        if cell["cell_type"] == "code":
            if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
                raise ValueError("Bayesian QPE notebook contains an error output")
            code.append(cell)
    if [cell.get("execution_count") for cell in code] != [1, 2, 3, 4, 5]:
        raise ValueError("Bayesian QPE notebook must retain five executed code cells")
    for cell, mime in ((code[1], "image/svg+xml"), (code[2], "text/markdown"), (code[3], "image/svg+xml")):
        if not any(mime in output.get("data", {}) for output in cell.get("outputs", [])):
            raise ValueError("Missing Bayesian QPE notebook circuit/table/posterior output")
    static_output = "".join("".join(output.get("text", [])) for output in code[4].get("outputs", []))
    if "Static mode: bounded live controls are disabled." not in static_output:
        raise ValueError("Bayesian QPE static live controls are not disabled")
    source_text = "\n".join(cell["source"] for cell in projection)
    if not all(f"({label})=" in source_text for label in EMBED_LABELS):
        raise ValueError("Bayesian QPE notebook embed labels are missing")
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bundle(root, relative):
    directory = _root(root / relative)
    expected = BUNDLE_FILES[relative]
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Bayesian QPE bundle contains unrecorded members")
    raw = _read(directory, "manifest.json", 64 * 1024)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_HASHES[relative]:
        raise ValueError("Saved Bayesian QPE manifest identity changed")
    manifest = json.loads(raw)
    records = manifest.get("files")
    if (manifest.get("schema") != "openquantum-bayesian-qpe-files-v1"
            or not isinstance(records, dict) or set(records) != expected - {"manifest.json"}):
        raise ValueError("Saved Bayesian QPE manifest membership changed")
    contents, total = {}, 0
    for name, record in records.items():
        payload = _read(directory, name)
        total += len(payload)
        if (total > MAX_BUNDLE_BYTES or record.get("bytes") != len(payload)
                or hashlib.sha256(payload).hexdigest() != record.get("sha256")):
            raise ValueError("Saved Bayesian QPE payload identity changed: " + name)
        contents[name] = payload
    source_hashes = {name: SOURCE_HASHES[source] for name, source in SNAPSHOTS.items()}
    if total != manifest.get("payload_bytes") or manifest.get("source_sha256") != source_hashes:
        raise ValueError("Saved Bayesian QPE source or byte accounting changed")
    return contents


def _study(contents, expected_pairs):
    study = json.loads(contents["results.json"])
    records = study.get("records", [])
    if (study.get("schema") != "openquantum-bayesian-qpe-study-v1"
            or [(record.get("nominal_phase"), record.get("root_seed")) for record in records] != expected_pairs
            or study.get("uniform_confidence_certificate") is not False
            or study.get("physical_estimate_performed") is not False
            or study.get("provenance", {}).get("source_commit_at_generation") is not None
            or study.get("provenance", {}).get("hardware_execution") is not False):
        raise ValueError("Saved Bayesian QPE study scope changed")
    base_calls = 0
    for index, record in enumerate(records):
        trace, ledger, n = record["trace"], record["ledger"], record["target_qubits"]
        phase_index = index // 2 if len(records) > 1 else 0
        seed_bytes = hashlib.sha256(f"original-grid-bayes:{record['root_seed']}:{phase_index}".encode()).digest()[:8]
        if record["seed"] != (int.from_bytes(seed_bytes, "big") & (2**63-1)):
            raise ValueError("Saved Bayesian QPE seed identity changed")
        if len(trace) != 48 or len(record["grid"]) != 512 or ledger["classical_policy_calls"] != 49:
            raise ValueError("Saved Bayesian QPE transcript size changed")
        k = sum(row["power"] for row in trace)
        counts = {"p": 3*n*k+48, "cx": 2*n*k, "h": 96, "x": n*48,
                  "reset": (n+1)*48, "measure": 48}
        if (ledger["source_gate_counts_all_trials"] != counts or ledger["source_operations_all_trials"] != sum(counts.values())
                or ledger["shot_weighted_controlled_U_calls"] != k or ledger["complete_fresh_preparations"] != 48
                or ledger["readout_bits"] != 48 or record["physical_execution"] is not False):
            raise ValueError("Saved Bayesian QPE fresh-trial accounting changed")
        base_calls += k
    if study["total_modeled_fresh_trials"] != 48*len(records) or study["total_literal_controlled_U_calls"] != base_calls:
        raise ValueError("Saved Bayesian QPE total accounting changed")
    rows = list(csv.DictReader(io.StringIO(contents["summary.csv"].decode("utf-8"))))
    if len(rows) != len(records):
        raise ValueError("Saved Bayesian QPE summary omits transcripts")
    for row, record in zip(rows, records):
        expected = {"nominal_phase": record["nominal_phase"], "root_seed": record["root_seed"],
                    "estimate": record["estimate"], "circular_error": record["circular_error"],
                    "credible_cell_width": record["summary"]["credible_arc"]["width"],
                    "controlled_U_calls": record["ledger"]["shot_weighted_controlled_U_calls"], "fresh_trials": 48}
        if set(row) != set(expected) or any(float(row[key]) != value for key, value in expected.items()):
            raise ValueError("Saved Bayesian QPE summary differs from its complete transcripts")
    return study


def verify_saved(root, *, source_root=None, notebook_roots=None, provenance=None):
    """Check existing records and source projections; make no new modeled draws."""
    root = _root(root)
    source_root = _root(root if source_root is None else source_root)
    validation_directory = _root(root / "assets/generated/bayesian-qpe-validation")
    if {path.name for path in validation_directory.iterdir()} != {"source32", "replay", "validation.json"}:
        raise ValueError("Bayesian QPE validation tree contains unrecorded members")
    notebook_roots = (source_root,) if notebook_roots is None else notebook_roots
    if not isinstance(notebook_roots, tuple) or len(notebook_roots) not in (1, 2):
        raise ValueError("One or two Bayesian QPE notebook roots are required")
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if provenance is not None and (not isinstance(provenance, dict)
            or provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict)):
        raise ValueError("Complete Bayesian QPE package provenance is required")
    for name, pin in SOURCE_HASHES.items():
        if hashlib.sha256(_read(source_root, name)).hexdigest() != pin or hashes is not None and hashes.get(name) != pin:
            raise ValueError("Promoted Bayesian QPE source identity changed: " + name)
    for notebook_root in notebook_roots:
        raw = _read(_root(notebook_root), NOTEBOOK, 2 * 1024 * 1024)
        if hashes is not None and hashlib.sha256(raw).hexdigest() != hashes.get(NOTEBOOK):
            raise ValueError("Refreshed Bayesian QPE notebook differs from current package provenance")
        if notebook_source_sha256(raw) != NOTEBOOK_SOURCE_SHA256:
            raise ValueError("Bayesian QPE notebook source changed")
    bundles = {relative: _bundle(root, relative) for relative in BUNDLE_FILES}
    for contents in bundles.values():
        if any(hashlib.sha256(contents[name]).hexdigest() != SOURCE_HASHES[source] for name, source in SNAPSHOTS.items()):
            raise ValueError("Saved Bayesian QPE source snapshot differs from promoted source")
    pairs = [(phase, seed) for phase in (0.3, 0.995, 0.125, 0.5) for seed in (7, 19)]
    study = _study(bundles["assets/generated/bayesian-qpe"], pairs)
    replay = _study(bundles["assets/generated/bayesian-qpe-validation/replay"], pairs[:1])
    first = {key: value for key, value in study["records"][0].items() if key != "classical_timing_seconds"}
    repeated = {key: value for key, value in replay["records"][0].items() if key != "classical_timing_seconds"}
    if first != repeated or study["total_literal_controlled_U_calls"] != 10446:
        raise ValueError("Saved Bayesian QPE replay or study total changed")
    source32 = json.loads(bundles["assets/generated/bayesian-qpe-validation/source32"]["results.json"])
    if (source32["source"]["logical_qubits"] != 33 or sum(source32["source"]["source_gate_counts"].values()) != 20549
            or any(source32[key] is not False for key in ("inference_performed", "simulation_performed", "physical_estimate_performed"))):
        raise ValueError("Bayesian QPE source-only companion scope changed")
    validation = _read(root, VALIDATION, 64 * 1024)
    if hashlib.sha256(validation).hexdigest() != VALIDATION_SHA256:
        raise ValueError("Saved Bayesian QPE validation identity changed")
    return {"status": "saved_component_verified", "study_outputs": 8, "saved_validation_modeled_draws": 456,
            "bundles": {name: len(files) for name, files in BUNDLE_FILES.items()},
            "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256, "new_modeled_draws": 0,
            "new_quantum_execution": False, "physical_estimation_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_saved(args.root), indent=2))


if __name__ == "__main__":
    main()
