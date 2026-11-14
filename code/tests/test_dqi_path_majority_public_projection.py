"""Static public path-majority projection boundaries; no saved or quantum replay.

Every target remains inert bytes, JSON or AST. These tests never import or call
the display script, circuit renderer, generator, numerical verifier or science
modules. No state/reference reconstruction or diagnostic recomputation occurs.
The source is authored for a later separately authorized pure test run.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET = "assets/generated/optimization/dqi-path-majority"
MANIFEST = ASSET + "/public-manifest.json"
DISPLAY = "scripts/display_dqi_path_majority_saved.py"
FIGURE_SOURCE = "scripts/plot_dqi_path_majority_component_svg.py"
THIS_TEST = "tests/test_dqi_path_majority_public_projection.py"
ARTICLE = "algorithms/optimization/dqi-degree-two.md"
FAMILY_NAMES = frozenset({
    "generated/component-ledger.json", "generated/component.ndjson",
    "generated/component.qasm", "generated/manifest.json",
    "generated/program-ledger.json", "generated/program.ndjson", "generated/program.qasm",
    "validation/manifest.json", "validation/validation.json",
    "input-k3.json", "component-input-recipe.json", "accepted-components.json",
    "results-summary.json", "environment.json", "README.md", "public-manifest.json",
    "companion-spec.json", "component-circuit.svg",
    "code/src/openquantum/optimization_dqi_path_majority.py",
    "code/src/openquantum/optimization_dqi_path_majority_verify.py",
    "code/scripts/generate_optimization_dqi_path_majority.py",
    "code/scripts/validate_optimization_dqi_path_majority.py",
    "code/tests/test_optimization_dqi_path_majority.py",
    "code/tests/test_optimization_dqi_path_majority_boundaries.py",
})
ORIGINAL_SOURCES = {
    "src/openquantum/optimization_dqi_path_majority.py": (36874, "ba3015dbdf35fe3ccec6a79503f2e911b735c2ea716969f690eb5fbc5e3c61bf"),
    "src/openquantum/optimization_dqi_path_majority_verify.py": (24109, "206c1e705bef8130677f835b68ec8a94a1fd420a53acd62d3d66b3ff8d536606"),
    "scripts/generate_optimization_dqi_path_majority.py": (23690, "01bafe43a205a2b04441ceb8c3aac99f4285b383899a57ed4296c3a6e64c8305"),
    "scripts/validate_optimization_dqi_path_majority.py": (29413, "febe90401a61a332133f3cd9d9229e704fb4e529564508f36ad12c8b510cdd42"),
    "tests/test_optimization_dqi_path_majority.py": (18529, "5b205986d5c71be16ca043897c4f8e0e42a5eda7e1b2e29bf40edf3aa5a9e7b9"),
    "tests/test_optimization_dqi_path_majority_boundaries.py": (14771, "b2f962131f9b0ac60f03c8c0d1640855ae18142c807e0e455f766a1805ed414a"),
}
SOURCE_NAMES = frozenset(ORIGINAL_SOURCES) | {DISPLAY, FIGURE_SOURCE, THIS_TEST}
RAW_PINS = {
    "generated/component-ledger.json": (638, "eae3ccf4c2259bfb8c367619b5d895e4e00e66de5a1a0700a5bfb114b99c77e4"),
    "generated/component.ndjson": (14666, "3cbb0866a388adf23a5b27594d90a92db99cfbab4ad901d597eef39ae9a6f562"),
    "generated/component.qasm": (5921, "7c4359771fd9274d9217cb60ce4066384b8aa2eaeffe219927e1c3bf0bb18091"),
    "generated/manifest.json": (2300, "daccd5efa3afa4a47b707a60a6a4617056d13685e4e0c61bd40a50e2734419a1"),
    "generated/program-ledger.json": (6199, "26f240f45234477a0e739004ccec8921c6c6581b1c631ee10ed0f786ed3324fc"),
    "generated/program.ndjson": (94905, "b1a422a454fd13ece43c21e83bce17cffa919174a1741544ee9445e6c9e1c22b"),
    "generated/program.qasm": (39912, "a43c9d0ae35facdf0f19d68053bff42488db092ca9ba96f24eac787ab52fc1bd"),
    "validation/manifest.json": (1155, "6f8e26cde01ad76ea8e37a9fbbaac6274439377609051121408cfbdc73d0414f"),
    "validation/validation.json": (278512, "4d10497bb5ec2bf335de94212f367c0fa3f4af3740ead4af9725328f4b63179d"),
    "input-k3.json": (71, "3ffac5f985d06910bd5c194a5e56b15beffaa3535d7b76b9e05743717faeab30"),
    "component-input-recipe.json": (785, "d750d0d1d282c80da33548dd0cfa228a2b314db499a4bde7d8b3fc7ea85e12ef"),
}
COMPONENT_COUNTS = {"h": 68, "cx": 219, "t": 136, "tdg": 102}
FULL_COUNTS = {"barrier": 7, "cx": 1272, "h": 385, "measure": 13,
               "reset": 31, "ry": 57, "t": 744, "tdg": 558, "x": 84, "z": 14}
UNSUPPORTED_RESULT_FLAGS = (
    "full_program_validated", "degree_two_preparation_validated",
    "optimization_output_validated", "physical_resources_estimated",
    "rotation_synthesis_validated", "formal_proof_reproduced", "hardware_executed",
    "sampling_performed", "dense_operator_constructed", "notebook_executed",
    "global_phase_fitted", "state_renormalized", "auxiliary_postselection",
)
CURATED_UNSUPPORTED = frozenset({
    "companion_notebook_executed", "degree_two_preparation_validated", "dense_operator_constructed",
    "formal_proof_reproduced", "full_program_validated", "hardware_executed",
    "optimization_output_validated", "physical_resources_estimated", "portable_scientific_replay_validated",
    "publication_complete", "rotation_synthesis_validated", "sampling_performed", "saved_display_executed",
})
EXPECTED_COMPONENT = {
    "arbitrary_clause_targets": 32, "complete_statevector_evolutions": 1,
    "native_estimator_calls": 0, "primitive_unitary_applications": 525,
    "pure_boolean_mappings": 512, "scratch_initially_clean": True,
    "scratch_wires": [9, 10, 11], "small_ccx_reference_cases": 0,
    "state_amplitudes": 4096, "syndrome_wires": [0, 1, 2, 3], "syndromes": 16,
    "target_wires": [4, 5, 6, 7, 8], "validator_boolean_mappings": 512,
    "whole_job_local_mapping_iterations": 1024, "wires": 12,
}
EXPECTED_TEST_COUNTS = {
    "global_weight_at_most_two_error_words": 121, "independent_classical_assignments_scanned": 8192,
    "passed": 259, "private_control_boundary_checks": 75,
    "public_source_contains_private_control_tests": False, "scientific_pure_checks": 184,
}
EXPECTED_VECTORS = {
    "final_state_pairs_saved": 4096, "input_and_reference_defined_by_pinned_source_recipe": True,
    "input_vector_separately_saved": False, "reconstruction_performed_during_public_projection": False,
    "reference_vector_separately_saved": False,
}
EXPECTED_GENERATED = {
    "N": 14, "barrier_records": 7, "component_primitive_counts": COMPONENT_COUNTS,
    "component_primitives": 525, "generation_only": True, "k": 3, "m": 15, "n": 13,
    "observed_source_counts": FULL_COUNTS, "operational_events_excluding_barriers": 3158,
    "quantum_wires": 31, "readout_events": 13, "reset_events": 31, "scope": "generation_only",
    "source_records_including_barriers": 3165, "unitary_primitives": 3114,
}
EXPECTED_DIAGNOSTICS = {
    "auxiliary_leakage_probability": 1.6312041995757796e-30, "auxiliary_postselection": False,
    "final_norm2": 1.0000000000000262, "global_phase_fitted": False,
    "max_signed_complex_entry_error": 1.1529425372364673e-15,
    "preserved_syndrome_probability_error": 1.8457457784393227e-15,
    "state_renormalized": False, "tolerance_absolute": 2e-10, "tolerance_relative": 0.0,
}


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(raw):
    return {"bytes": len(raw), "sha256": _digest(raw)}


def _same_json(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_same_json(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_json(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _assert_exact(actual, expected):
    assert _same_json(actual, expected)


def _assert_pin_shape(value, cap=2 << 20):
    assert type(value) is dict and set(value) == {"bytes", "sha256"}
    assert type(value["bytes"]) is int and 0 < value["bytes"] <= cap
    assert type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])


def _read(path, cap=2 << 20):
    """Bounded ordinary-byte fixture read; no target import or dispatch."""
    path = Path(path)
    for parent in path.parents:
        assert stat.S_ISDIR(parent.lstat().st_mode)
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(fd)
        assert stat.S_ISREG(opened.st_mode)
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(65536, cap + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            assert total <= cap
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns)
    assert identity(before) == identity(opened) == identity(after) and total == after.st_size
    return b"".join(chunks)


def _closed_asset_names(directory):
    names, visited = set(), 0
    stack = [(Path(directory), 0)]
    while stack:
        current, depth = stack.pop()
        assert depth <= 6
        with os.scandir(current) as entries:
            for entry in entries:
                visited += 1
                assert visited <= 64 and not entry.is_symlink()
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
                else:
                    assert entry.is_file(follow_symlinks=False)
                    names.add(Path(entry.path).relative_to(directory).as_posix())
    return names


def _choose_artifact_root(source_root):
    source_root = Path(source_root)
    candidates = (source_root, source_root.parent)
    artifact_root = next((p for p in candidates if (p / MANIFEST).is_file()), None)
    book_source = (source_root / "myst.yml").is_file() or (source_root / ARTICLE).is_file()
    if book_source:
        assert artifact_root is not None, "The required public path-majority manifest is missing from the source book"
    return artifact_root


@pytest.fixture(scope="module")
def packet():
    artifact_root = _choose_artifact_root(ROOT)
    if artifact_root is None:
        pytest.skip("These static fixture tests require the separately admitted public asset family")
    assets = {name: _read(artifact_root / ASSET / name) for name in FAMILY_NAMES}
    return artifact_root, assets, json.loads(assets["public-manifest.json"])


def _assert_component_result(value):
    assert type(value) is dict
    _assert_exact(value["schema"], "openquantum.dqi.path_majority_component_validation.v1")
    _assert_exact(value["status"], "passed_component_only_numerical_check")
    for key, expected in {
        "state_wires": 12, "state_amplitudes": 4096, "verification_invocations": 1,
        "complete_statevector_evolutions": 1, "primitive_unitary_applications": 525,
        "boolean_decoder_mappings": 512, "small_ccx_reference_cases": 0,
        "native_estimator_calls": 0, "primitive_counts": COMPONENT_COUNTS,
        "tolerance_absolute": 2e-10, "tolerance_relative": 0.0,
        "max_signed_complex_entry_error": 1.1529425372364673e-15,
        "auxiliary_leakage_probability": 1.6312041995757796e-30,
        "preserved_syndrome_probability_error": 1.8457457784393227e-15,
        "final_norm2": 1.0000000000000262,
    }.items():
        _assert_exact(value[key], expected)
    for flag in UNSUPPORTED_RESULT_FLAGS:
        assert value[flag] is False
    budget = value["execution_budget"]
    for key, expected in {"active": False, "complete": True, "spent": True,
                          "verification_invocations": 1, "complete_statevector_evolutions": 1,
                          "primitive_unitary_applications": 525, "boolean_decoder_mappings": 512,
                          "small_ccx_reference_cases": 0, "native_estimator_calls": 0}.items():
        _assert_exact(budget[key], expected)
    pairs = value["final_state"]
    assert type(pairs) is list and len(pairs) == 4096
    for pair in pairs:
        assert type(pair) is list and len(pair) == 2
        assert all(type(x) in (int, float) and math.isfinite(x) for x in pair)
    # Only stored scalar/type/length checks: no norm, reference or error recomputation.


def _assert_acceptance(value):
    assert type(value) is dict
    _assert_exact(value["schema"], "openquantum.dqi.path_majority_public_acceptance.v1")
    _assert_exact(value["status"], "parent_accepted_component_and_generation_only")
    _assert_exact(value["accepted_scope"], {"full_algorithm": False, "generated_full_program_source": True,
                                           "local_decoder_component": True})
    _assert_exact(value["component"], EXPECTED_COMPONENT)
    _assert_exact(value["component_diagnostics"], EXPECTED_DIAGNOSTICS)
    _assert_exact(value["tests"], EXPECTED_TEST_COUNTS)
    _assert_exact(value["vectors_saved"], EXPECTED_VECTORS)
    _assert_exact(value["generated_source_counts"], EXPECTED_GENERATED)
    assert set(value["unsupported_claims"]) == CURATED_UNSUPPORTED
    assert all(flag is False for flag in value["unsupported_claims"].values())
    assert value["raw_records_preserved"] is True
    assert value["original_c5_evidence_modified"] is False
    assert value["surface_code_baseline_modified"] is False
    assert value["public_projection_scientific_work_reexecuted"] is False
    _assert_exact(value["source_identities"], {
        "scientific_source_commit": "7a4b1cdaa1354014886f87d14149bd3d2d5edb63",
        "repaired_control_source_commit": "241f3385e0070b5de90288324159a88ea200a00e",
        "accepted_caller_source_commit": "9dad220bb2e18d72b3d267735f380e3fc9f2c352",
    })


def _primitive_records(raw):
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 525
    return [(row["gate"], tuple(row["qubits"])) for row in rows]


def _ccx_word(a, b, target):
    return [("h", (target,)), ("cx", (b, target)), ("tdg", (target,)),
            ("cx", (a, target)), ("t", (target,)), ("cx", (b, target)),
            ("tdg", (target,)), ("cx", (a, target)), ("t", (b,)),
            ("t", (target,)), ("h", (target,)), ("cx", (a, b)),
            ("t", (a,)), ("tdg", (b,)), ("cx", (a, b))]


def _fold_component(raw):
    """Parse only exact gate-word syntax, never evaluate a Boolean or quantum state."""
    rows = _primitive_records(raw)
    result, position = [], 0
    while position < len(rows):
        name, qubits = rows[position]
        if name == "cx":
            stop = position + 1
        else:
            assert name == "h" and len(qubits) == 1 and position + 15 <= len(rows)
            target = qubits[0]
            b, seen_target = rows[position + 1][1]
            a, other_target = rows[position + 3][1]
            assert seen_target == other_target == target
            assert rows[position:position + 15] == _ccx_word(a, b, target)
            name, qubits, stop = "ccx", (a, b, target), position + 15
        result.append((name, qubits, position, stop))
        position = stop
    assert len(result) == 49 and Counter(x[0] for x in result) == {"ccx": 34, "cx": 15}
    return result


def test_asset_family_is_exact_and_contains_no_notebook_archive_or_private_control(packet):
    artifact_root, assets, _ = packet
    assert len(FAMILY_NAMES) == len(assets) == 24
    assert _closed_asset_names(artifact_root / ASSET) == set(FAMILY_NAMES)
    assert not any(name.endswith((".ipynb", ".zip", ".log", ".bin")) for name in FAMILY_NAMES)
    assert not any("controller-repair" in name or "run_path_majority" in name for name in FAMILY_NAMES)


@pytest.mark.parametrize("marker", ["myst.yml", ARTICLE])
def test_missing_public_manifest_is_a_source_book_failure_not_a_fixture_skip(tmp_path, marker):
    root = tmp_path / "book"
    path = root / marker
    path.parent.mkdir(parents=True)
    path.write_text("source book context\n")
    with pytest.raises(AssertionError, match="manifest is missing"):
        _choose_artifact_root(root)


def test_asset_omitted_export_context_is_distinct_from_source_book(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    assert _choose_artifact_root(root) is None


def test_original_raw_evidence_and_scientific_code_are_byte_identical(packet):
    _, assets, _ = packet
    for name, (size, digest) in RAW_PINS.items():
        assert len(assets[name]) == size and _digest(assets[name]) == digest
    for name, (size, digest) in ORIGINAL_SOURCES.items():
        source = _read(ROOT / name)
        snapshot = assets["code/" + name]
        assert source == snapshot and len(source) == size and _digest(source) == digest


def test_manifest_closes_all_payloads_and_source_bytes_without_a_self_hash(packet):
    artifact_root, assets, manifest = packet
    assert type(manifest) is dict and set(manifest) == {"schema", "status", "files", "source_files"}
    _assert_exact(manifest["schema"], "openquantum.dqi.path_majority_public_projection.v1")
    _assert_exact(manifest["status"], "parent_accepted_component_and_generation_only")
    assert set(manifest["files"]) == {ASSET + "/" + name for name in FAMILY_NAMES if name != "public-manifest.json"}
    assert MANIFEST not in manifest["files"] and len(manifest["files"]) == 23
    assert set(manifest["source_files"]) == SOURCE_NAMES and len(manifest["source_files"]) == 9
    for relative, expected in manifest["files"].items():
        _assert_pin_shape(expected)
        assert _pin(assets[PurePosixPath(relative).relative_to(ASSET).as_posix()]) == expected
    for relative, expected in manifest["source_files"].items():
        _assert_pin_shape(expected)
        assert _pin(_read(ROOT / relative)) == expected
    # The test's own bytes are admitted by the manifest; no literal self digest is embedded here.
    assert THIS_TEST in manifest["source_files"]


@pytest.mark.parametrize("value", [True, 1.0, 0, -1, "1", None])
def test_pin_metadata_rejects_boolean_float_and_invalid_byte_allowances(value):
    with pytest.raises(AssertionError):
        _assert_pin_shape({"bytes": value, "sha256": "a" * 64})


def test_raw_ledgers_keep_original_pre_acceptance_status_and_only_observed_source_counts(packet):
    _, assets, _ = packet
    generated = json.loads(assets["generated/manifest.json"])
    ledger = json.loads(assets["generated/program-ledger.json"])
    component = json.loads(assets["generated/component-ledger.json"])
    validation_manifest = json.loads(assets["validation/manifest.json"])
    _assert_exact(generated["status"], "complete_generation_component_pending_invocation_acceptance")
    _assert_exact(validation_manifest["status"], "complete_validation_component_pending_invocation_acceptance")
    assert generated["component_numerically_validated"] is False and generated["simulation_performed"] is False
    assert ledger["source_prediction_only"] is True and ledger["scientific_validated"] is False
    _assert_exact(ledger["observed_source_counts"], FULL_COUNTS)
    _assert_exact(ledger["observed_source_operations"], 3165)
    _assert_exact(ledger["source_unitary_primitives"], 3114)
    assert ledger["observed_source_counts"]["barrier"] == 7
    assert 3114 + 31 + 13 == 3158 and 3158 + 7 == 3165
    _assert_exact(component["primitive_counts"], COMPONENT_COUNTS)
    _assert_exact(component["ccx_blocks"], 34)
    _assert_exact(component["cx_outside_ccx"], 15)
    assert component["full_program_validation"] is False and component["clean_scratch_required"] is True
    assert component["arbitrary_target_xor"] is True and component["total_extension_uses_terminal_syndrome"] is False


def test_raw_validation_is_finite_component_evidence_without_diagnostic_recomputation(packet):
    _, assets, _ = packet
    _assert_component_result(json.loads(assets["validation/validation.json"]))


def test_curated_acceptance_closes_component_and_generation_only_scope(packet):
    _, assets, _ = packet
    value = json.loads(assets["accepted-components.json"])
    _assert_acceptance(value)
    _assert_exact(value["evidence_identities"]["generated_manifest"], {
        "bytes": RAW_PINS["generated/manifest.json"][0], "sha256": RAW_PINS["generated/manifest.json"][1]})
    _assert_exact(value["evidence_identities"]["validation_payload"], {
        "bytes": RAW_PINS["validation/validation.json"][0], "sha256": RAW_PINS["validation/validation.json"][1]})
    _assert_exact(value["evidence_identities"]["validation_manifest"], {
        "bytes": RAW_PINS["validation/manifest.json"][0], "sha256": RAW_PINS["validation/manifest.json"][1]})
    _assert_exact(value["evidence_identities"]["parent_acceptance_record"], {
        "bytes": 2399, "sha256": "3a4694789d2400bbd37add828bc30cc64d6bb21b741244eaffde8a80c648838b"})
    _assert_exact(value["evidence_identities"]["execution_receipt"], {
        "bytes": 12253, "sha256": "a1c80e3eb76d38ed458d682c7b4303e51c45a61e601d3e9afedbe7fe98e68ea8"})


@pytest.mark.parametrize("section, field, replacement", [
    ("accepted_scope", "full_algorithm", True),
    ("component", "wires", 31), ("component", "wires", 12.0),
    ("component", "complete_statevector_evolutions", True),
    ("component", "validator_boolean_mappings", 1024),
    ("component", "whole_job_local_mapping_iterations", 512),
    ("tests", "passed", 259.0), ("tests", "private_control_boundary_checks", 0),
    ("tests", "public_source_contains_private_control_tests", True),
    ("vectors_saved", "input_vector_separately_saved", True),
    ("vectors_saved", "reference_vector_separately_saved", True),
    ("unsupported_claims", "portable_scientific_replay_validated", True),
    ("generated_source_counts", "generation_only", False),
    ("generated_source_counts", "operational_events_excluding_barriers", 3165),
])
def test_curated_acceptance_rejects_mixed_scopes_bool_aliases_or_unearned_claims(
        packet, section, field, replacement):
    _, assets, _ = packet
    value = json.loads(assets["accepted-components.json"])
    value[section][field] = replacement
    with pytest.raises(AssertionError):
        _assert_acceptance(value)


def test_selected_results_copy_saved_fields_without_reconstructed_inputs_or_diagnostics(packet):
    _, assets, _ = packet
    summary = json.loads(assets["results-summary.json"])
    acceptance = json.loads(assets["accepted-components.json"])
    result = json.loads(assets["validation/validation.json"])
    _assert_exact(summary["schema"], "openquantum.dqi.path_majority_public_results.v1")
    _assert_exact(summary["status"], "parent_accepted_component_and_generation_only")
    _assert_exact(summary["component_counts"], acceptance["component"])
    _assert_exact(summary["component_diagnostics"], acceptance["component_diagnostics"])
    _assert_exact(summary["generated_source_counts"], acceptance["generated_source_counts"])
    _assert_exact(summary["test_counts"], acceptance["tests"])
    _assert_exact(summary["vectors_saved"], acceptance["vectors_saved"])
    _assert_exact(summary["unsupported_claims"], acceptance["unsupported_claims"])
    _assert_exact(summary["component_input_recipe"], result["component_input_recipe"])
    _assert_exact(summary["negative_case"], result["boolean_checks"])
    _assert_exact(summary["primitive_counts"], result["primitive_counts"])
    _assert_exact(summary["quantum_source_envelope"], result["source_quantum_envelope"])
    for field, expected in EXPECTED_DIAGNOSTICS.items():
        _assert_exact(summary["component_diagnostics"][field], result[field])
        _assert_exact(summary["component_diagnostics"][field], expected)
    _assert_exact(summary["raw_state_pairs"], 4096)
    _assert_exact(summary["full_state_payload"], ASSET + "/validation/validation.json")
    _assert_exact(summary["validation_source_pin"], _pin(assets["validation/validation.json"]))
    assert "input_state" not in summary and "reference_state" not in summary and "final_state" not in summary


def test_environment_credits_exact_versions_and_selected_coverage_without_replay_claim(packet):
    _, assets, _ = packet
    environment = json.loads(assets["environment.json"])
    _assert_exact(environment["schema"], "openquantum.dqi.path_majority_public_environment.v1")
    _assert_exact(environment["status"], "accepted_selected_environment_identity")
    _assert_exact(environment["python"]["implementation"], "CPython")
    _assert_exact(environment["python"]["version"], "3.12.14")
    assert environment["python"]["native_quantum_sdk_used"] is False
    versions = {"pytest": "9.2.0", "psutil": "7.2.2", "pluggy": "1.7.0", "iniconfig": "2.3.1",
                "packaging": "26.3", "pygments": "2.21.0", "typing_extensions": "4.16.0"}
    assert set(environment["packages"]) == set(versions)
    for package, version in versions.items():
        _assert_exact(environment["packages"][package]["version"], version)
        assert environment["packages"][package]["lock_match"] is True
        assert type(environment["packages"][package]["license"]) is str
        _assert_pin_shape(environment["packages"][package]["installed_metadata_pin"])
    _assert_exact(environment["selected_environment"]["regular_files"], 817)
    _assert_exact(environment["selected_environment"]["selected_total_bytes"], 49419226)
    assert environment["scientific_replay_in_new_download_context_validated"] is False
    for field in ("companion_notebook_executed", "public_operational_content_included",
                  "public_saved_display_executed", "publication_complete", "physical_resources_estimated"):
        assert environment[field] is False
    expected_support = {
        "LICENSE": {"bytes": 1093, "sha256": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"},
        "pyproject.toml": {"bytes": 705, "sha256": "fb96cdbcbb8c032581d62d8d04f261e988fced3e66a0f560ea79d7e9c2d72404"},
        "requirements-lock.txt": {"bytes": 1974, "sha256": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"},
    }
    _assert_exact(environment["root_support_pins"], expected_support)
    # Support pins describe the accepted source context, not a replacement for a site/package bootstrap.


@pytest.mark.parametrize("field, replacement", [
    ("state_wires", 31), ("state_wires", 12.0), ("complete_statevector_evolutions", True),
    ("complete_statevector_evolutions", 2), ("primitive_unitary_applications", 3165),
    ("boolean_decoder_mappings", 1024), ("tolerance_absolute", 1e-3),
    ("full_program_validated", True), ("degree_two_preparation_validated", True),
    ("optimization_output_validated", True), ("physical_resources_estimated", True),
    ("formal_proof_reproduced", True), ("notebook_executed", True),
])
def test_raw_result_contract_rejects_scope_widening_or_inexact_types(packet, field, replacement):
    _, assets, _ = packet
    value = json.loads(assets["validation/validation.json"])
    value[field] = replacement
    with pytest.raises(AssertionError):
        _assert_component_result(value)


def test_input_recipe_and_negative_case_remain_explicit_saved_metadata(packet):
    _, assets, _ = packet
    recipe = json.loads(assets["component-input-recipe.json"])
    result = json.loads(assets["validation/validation.json"])
    _assert_exact(result["component_input_recipe"], recipe)
    _assert_exact(recipe["state_wires"], 12)
    _assert_exact(recipe["state_amplitudes"], 4096)
    _assert_exact(recipe["support_size"], 512)
    assert recipe["all_support_magnitudes_distinct"] is True and recipe["uniform_target_plus"] is False
    assert recipe["scratch_zero_required"] is True and recipe["executed_by_metadata_function"] is False
    negative = result["boolean_checks"]
    for key, expected in {
        "negative_error_bits": [0, 1, 1, 1, 0], "negative_internal_syndrome_bits": [1, 0, 0, 1],
        "negative_returned_error_bits": [1, 0, 0, 0, 1], "terminal_syndrome_bit_of_error": 0,
        "terminal_syndrome_bit_of_returned_error": 1, "syndromes": 16,
        "arbitrary_clause_targets": 32, "mappings": 512,
    }.items():
        _assert_exact(negative[key], expected)
    assert negative["weight_two_promise_extended"] is False


def test_companion_is_only_an_unexecuted_specification_with_explicit_future_authority(packet):
    _, assets, _ = packet
    spec = json.loads(assets["companion-spec.json"])
    _assert_exact(spec["schema"], "openquantum.dqi.path_majority_saved_companion_spec.v1")
    _assert_exact(spec["status"], "specification_only_unexecuted")
    for field in ("execution_authorized", "notebook_created", "notebook_executed",
                  "scientific_work_reexecuted", "publication_complete"):
        assert spec[field] is False
    assert spec["expected_manifest_sha256"] is None and spec["manifest_digest_is_external_argument"] is True
    _assert_exact(spec["source"], DISPLAY)
    _assert_exact(spec["callable"], "display_saved_component")
    _assert_exact(spec["time_limit_seconds"], 5)
    assert len(spec["future_code_cells"]) == 2
    assert "--display" in spec["cli_argv_template"]
    assert spec["source_roots_require_complete_provenance_before_loading"] is True


def test_display_and_figure_sources_are_inert_ast_with_no_target_or_dynamic_imports():
    allowed = {"argparse", "hashlib", "json", "math", "os", "pathlib", "re", "signal", "stat", "sys", "time",
               "html", "xml", "collections", "typing", "__future__"}
    for path in (DISPLAY, FIGURE_SOURCE):
        tree = ast.parse(_read(ROOT / path).decode(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".", 1)[0] in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module and node.module.split(".", 1)[0] in allowed
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile", "__import__"}
        if path == DISPLAY:
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            assert not any(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                           and node.func.value.id == "math" and node.func.attr != "isfinite" for node in calls)
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            guard = next(i for i, node in enumerate(main.body) if isinstance(node, ast.If)
                         and "args.display" in ast.unparse(node.test))
            dispatch = next(i for i, node in enumerate(main.body) if any(
                isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                and item.func.id == "display_saved_component" for item in ast.walk(node)))
            assert guard < dispatch
            condition = ast.unparse(main.body[guard].test)
            assert condition == "args.display is not True"
            assert any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                       and item.func.attr == "error" for item in ast.walk(main.body[guard]))
            forbidden_before_guard = {"_read", "_ordinary", "_inventory", "_digest", "_seconds",
                                      "display_saved_component", "open"}
            for statement in main.body[:guard]:
                for item in ast.walk(statement):
                    if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                        assert item.func.id not in forbidden_before_guard
                    elif isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
                        assert item.func.attr not in {"read", "read_bytes", "open", "stat", "lstat", "resolve"}
            for statement in tree.body:
                if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.If)):
                    continue
                assert not any(isinstance(item, ast.Call) and (
                    isinstance(item.func, ast.Name) and item.func.id in forbidden_before_guard
                    or isinstance(item.func, ast.Attribute) and item.func.attr in {
                        "read", "read_bytes", "open", "stat", "lstat", "resolve"}) for item in ast.walk(statement))


def test_svg_root_metadata_and_every_visible_gate_match_the_accepted_source_word(packet):
    _, assets, _ = packet
    svg = ET.fromstring(assets["component-circuit.svg"])
    for key, expected in {"data-schema": "openquantum.dqi.path_majority_circuit_svg.v1",
                          "data-source-sha256": RAW_PINS["generated/component.ndjson"][1],
                          "data-wires": "12", "data-logical-gates": "49", "data-ccx-blocks": "34",
                          "data-cx-outside-ccx": "15", "data-primitive-operations": "525",
                          "data-full-program-validated": "false"}.items():
        assert svg.attrib[key] == expected
    metadata = next(node for node in svg.iter() if node.attrib.get("id") == "circuit-metadata")
    description = json.loads(metadata.text)
    _assert_exact(description["primitive_counts"], COMPONENT_COUNTS)
    _assert_exact(description["stage_ranges"], {"prefix": [0, 3], "compute_flag": [3, 258],
                                               "xor_targets": [258, 267], "uncompute_flag": [267, 522],
                                               "undo_prefix": [522, 525]})
    assert description["arbitrary_targets"] is True and description["syndrome_preserved"] is True
    assert description["logical_symbols_use_full_phase_exact_ccx"] is True
    assert description["full_program_validated"] is False
    _assert_exact(description["initial_scratch"], "000")
    _assert_exact(description["final_scratch"], "000")
    folded = _fold_component(assets["generated/component.ndjson"])
    gates = [node for node in svg.iter() if "logical-gate" in node.attrib.get("class", "").split()]
    assert len(gates) == 49
    assert [int(node.attrib["data-logical-index"]) for node in gates] == list(range(1, 50))
    for visible, (name, qubits, start, stop), recorded in zip(gates, folded, description["logical_gates"], strict=True):
        assert visible.attrib["data-gate"] == recorded["gate"] == name
        assert int(visible.attrib["data-logical-index"]) == recorded["logical_index"]
        assert tuple(int(x) for x in visible.attrib["data-qubits"].split()) == tuple(recorded["qubits"]) == qubits
        assert int(visible.attrib["data-primitive-start"]) == recorded["primitive_start"] == start
        assert int(visible.attrib["data-primitive-stop"]) == recorded["primitive_stop"] == stop
        assert visible.attrib["data-stage"] == recorded["stage"]
    wires = [node for node in svg.iter() if "circuit-wire" in node.attrib.get("class", "").split()]
    assert len(wires) == 36 and {int(node.attrib["data-qubit"]) for node in wires} == set(range(12))
    assert Counter(int(node.attrib["data-panel-row"]) for node in wires) == {0: 12, 1: 12, 2: 12}


def test_display_failure_output_retains_alarm_and_has_bounded_unexpected_error_exit():
    tree = ast.parse(_read(ROOT / DISPLAY).decode(), filename=DISPLAY)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    protected = next(node for node in main.body if isinstance(node, ast.Try) and any(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
        and item.func.id == "display_saved_component" for statement in node.body for item in ast.walk(statement)))
    assert len(protected.handlers) == 1
    handler = protected.handlers[0]
    assert isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
    failure_write = next(node for node in handler.body if isinstance(node, ast.Try))
    assert any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "_write"
               for statement in failure_write.body for item in ast.walk(statement))
    assert len(failure_write.handlers) == 1
    catch = failure_write.handlers[0]
    assert isinstance(catch.type, ast.Name) and catch.type.id == "BaseException"
    assert any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
               and isinstance(item.func.value, ast.Name) and item.func.value.id == "os"
               and item.func.attr == "_exit" and len(item.args) == 1
               and isinstance(item.args[0], ast.Constant) and item.args[0].value == 1
               for statement in catch.body for item in ast.walk(statement))
    assert not any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                   and item.func.attr == "setitimer" for item in ast.walk(handler))
    expired = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "expired")
    assert any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
               and isinstance(item.func.value, ast.Name) and item.func.value.id == "os"
               and item.func.attr == "_exit" and isinstance(item.args[0], ast.Constant)
               and item.args[0].value == 124 for item in ast.walk(expired))
    assert any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
               and item.func.attr == "setitimer" and len(item.args) == 2
               and isinstance(item.args[1], ast.Constant) and item.args[1].value == 0
               for statement in protected.finalbody for item in ast.walk(statement))


def test_public_assets_contain_no_concrete_private_locations_or_credentials(packet):
    _, assets, _ = packet
    prefixes = [b"/" + b"Users" + b"/", b"/" + b"private" + b"/" + b"tmp" + b"/",
                b"ssh" + b"://", b"BEGIN " + b"OPENSSH PRIVATE KEY", b"PAGES_" + b"DEPLOY_KEY="]
    for name, raw in assets.items():
        assert not any(prefix in raw for prefix in prefixes), name
    manifest = json.loads(assets["public-manifest.json"])
    assert not any("controller-repair" in name or "run_path_majority" in name
                   for name in manifest["source_files"])
