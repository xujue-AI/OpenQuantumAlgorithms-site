"""Immutable n2 release preflight and independent scientific replay regressions."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("n2_release_gate", ROOT / "scripts/verify_simulation_lindblad_physical.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def records():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    return [gate.load_json(bundle / name) for name in ("manifest.json", "results.json")]


@pytest.fixture
def package(tmp_path):
    root = (tmp_path / "package").resolve()
    shutil.copytree(ROOT / gate.BUNDLE_RELATIVE, root / gate.BUNDLE_RELATIVE)
    paths = {relative for relative in gate.SOURCE_FILES.values() if relative is not None}
    paths |= gate.FROZEN_FILES | {"references/simulation.bib", "references/resources.bib"}
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def verify_identity(root):
    return gate.verify_bundle(root / gate.BUNDLE_RELATIVE, source_root=root, canonical_root=root, replay=False)


def set_value(record, keys, value):
    current = record
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def test_exact_preserved_bundle_passes_complete_scientific_replay_without_mutation():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    before = {path.name: gate.sha256(path) for path in bundle.iterdir()}
    receipt = gate.verify_bundle(bundle, source_root=ROOT, canonical_root=ROOT)
    assert receipt["files"] == 21 and receipt["bytes"] == 3201797
    assert receipt["qasm_instructions"] == 206553
    assert receipt["rotation_occurrences"] == 828
    assert receipt["density_replays"] == 3 and receipt["physical_points"] == 182
    assert receipt["native_synthesis_calls"] == receipt["native_factory_queries"] == 0
    assert 0 <= receipt["verification_wall_seconds"] < gate.CAPS["replay_seconds"]
    assert before == {path.name: gate.sha256(path) for path in bundle.iterdir()}


def test_identity_only_preflight_does_not_claim_scientific_replays(package):
    receipt = verify_identity(package)
    assert receipt["manifest_sha256"] == gate.REVIEWED_MANIFEST_SHA256
    assert "density_replays" not in receipt and "physical_points" not in receipt


@pytest.mark.parametrize("relative", sorted({relative for relative in gate.SOURCE_FILES.values() if relative is not None}
                                           | {"references/simulation.bib", "references/resources.bib"}))
def test_current_producer_source_lock_license_or_bibliography_change_rejects(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="current source digest"):
        verify_identity(package)


@pytest.mark.parametrize("name", ["source-trial.qasm", "clifford-t-trial.qasm", "results.json",
                                  "summary.csv", "counts.svg", "instrument_resources.py"])
def test_changed_preserved_artifact_rejects(package, name):
    path = package / gate.BUNDLE_RELATIVE / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest"):
        verify_identity(package)


@pytest.mark.parametrize("name", ["source-trial.qasm", "requirements-lock.txt", "space-time.svg"])
def test_missing_preserved_asset_rejects(package, name):
    (package / gate.BUNDLE_RELATIVE / name).unlink()
    with pytest.raises(ValueError, match="21-file bundle"):
        verify_identity(package)


def test_unrecorded_asset_rejects(package):
    (package / gate.BUNDLE_RELATIVE / "private.log").write_text("unreviewed")
    with pytest.raises(ValueError, match="21-file bundle"):
        verify_identity(package)


def test_consistently_changed_manifest_still_requires_new_review(package):
    path = package / gate.BUNDLE_RELATIVE / "manifest.json"
    manifest = gate.load_json(path)
    manifest["device"] = "unreviewed calibration claim"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="immutable reviewed manifest"):
        verify_identity(package)


def test_changed_frozen_grover_baseline_rejects(package):
    path = package / "assets/generated/resources/grover-surface-code.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="current frozen baseline"):
        verify_identity(package)


@pytest.mark.parametrize("index,keys,value", [
    (0, ["python"], "3.12.15"),
    (0, ["private_cache_distributed"], True),
    (0, ["whole_family_physical_completed"], True),
    (0, ["frozen_after"], {}),
    (0, ["artifacts_sha256"], []),
    (0, ["wall_seconds"], 1201.),
    (0, ["wall_seconds"], float("inf")),
    (1, ["prepared", "case_id"], "lindblad-n3-r134-Z0"),
    (1, ["prepared", "task", "n_system"], 3),
    (1, ["prepared", "task", "time"], 2.),
    (1, ["prepared", "selected_steps"], 68),
    (1, ["prepared", "limits", "max_qubits"], 8),
    (1, ["prepared", "actual_synthesis", "rotation_occurrences"], 760),
    (1, ["prepared", "actual_synthesis", "preserve_full_unitary"], False),
    (1, ["prepared", "complete_workload", "complete_trials"], 1),
    (1, ["prepared", "complete_workload", "source_resets_all_trials"], 1),
    (1, ["prepared", "complete_workload", "whole_workload_executed"], True),
    (1, ["prepared", "hardware_execution"], True),
    (1, ["prepared", "error_contract", "formal_synthesis_certificate"], True),
    (1, ["prepared", "qasm_sha256"], None),
    (1, ["execution", "stage_status", "prepare", "status"], "planned"),
    (1, ["family", "3", "status"], "implemented"),
    (1, ["family", "3", "rejection_reproduced"], False),
    (1, ["family", "4", "physical_estimation_performed"], True),
    (1, ["family", "4", "family_allocation_reallocated"], True),
    (1, ["physical_profiles", 0, "status"], "planned"),
    (1, ["physical_profiles", 0, "profile", "controller_failure_per_decision"], 1e-18),
    (1, ["physical_profiles", 0, "complete_algorithm_attempts"], 1),
    (1, ["physical_profiles", 0, "physical_failure_budget"], .01),
    (1, ["physical_profiles", 0, "factory_availability_budget"], .01),
])
def test_status_input_identity_and_calibration_changes_reject_before_hashes(records, index, keys, value):
    set_value(records[index], keys, value)
    with pytest.raises(ValueError, match="n2"):
        gate.verify_records(*records)


@pytest.mark.parametrize("index,key", [(0, "execution"), (1, "prepared"), (1, "family")])
def test_missing_required_status_records_fail_as_value_errors(records, index, key):
    del records[index][key]
    with pytest.raises(ValueError, match="malformed"):
        gate.verify_records(*records)


@pytest.fixture
def physical_counts():
    instructions = gate.parse_qasm((ROOT / gate.BUNDLE_RELATIVE / "clifford-t-trial.qasm").read_text())
    return gate.physical_event_counts(instructions)


@pytest.mark.parametrize("keys,value", [
    (["all_feasible_points", 0, "macro_ledger_per_complete_attempt", 0, "count"], 1),
    (["all_feasible_points", 0, "macro_ledger_per_complete_attempt", 0, "time_ns"], 0),
    (["all_feasible_points", 0, "macro_ledger_per_complete_attempt", 0, "event_failure_model"], 0.),
    (["all_feasible_points", 0, "runtime_per_complete_attempt_ns"], 0),
    (["all_feasible_points", 0, "worst_case_runtime_all_attempts_ns"], 1),
    (["all_feasible_points", 0, "factory_physical_qubits"], 0),
    (["all_feasible_points", 0, "factory_supply", "pipeline_time_ns"], 1),
    (["all_feasible_points", 0, "factory_supply", "output_error"], 1e-8),
    (["all_feasible_points", 0, "factory_supply", "native_qec_crossing_prefactor"], .1),
    (["all_feasible_points", 0, "factory_supply", "full_pipeline_retry_cap_per_state"], 6),
    (["all_feasible_points", 0, "factory_supply", "states_all_complete_attempts"], 1),
    (["all_feasible_points", 0, "factory_supply", "extra_output_port_hold_ns_per_state"], 0),
    (["all_feasible_points", 0, "failure_components_all_attempts", "controller_scenario"], 0.),
    (["all_feasible_points", 0, "full_physical_failure_model"], 0.),
    (["controller_model_diagnostic", "controller_decisions_all_trials"], 1),
    (["pareto_points"], []),
])
def test_independent_ledger_replay_rejects_changed_costs_calibration_and_frontiers(records, physical_counts, keys, value):
    profile = copy.deepcopy(records[1]["physical_profiles"][0])
    set_value(profile, keys, value)
    with pytest.raises(ValueError, match="n2"):
        gate.replay_physical(profile, physical_counts, time.monotonic() + 60)


def test_actual_t_wire_and_startup_counts_do_not_use_width_only_proxies(physical_counts):
    assert physical_counts["initial_reset"] == 15 and physical_counts["reset"] == 205
    assert physical_counts["measure"] == 4
    assert [physical_counts["t_q" + str(wire)] for wire in range(4)] == [22466, 11503, 33099, 11040]


@pytest.fixture(scope="module")
def compiler_inputs():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    source = gate.parse_qasm((bundle / "source-trial.qasm").read_text())
    compiled = gate.parse_qasm((bundle / "clifford-t-trial.qasm").read_text())
    routed, _ = gate.routed_instructions(source)
    synthesis = gate.load_json(bundle / "results.json")["prepared"]["actual_synthesis"]
    return routed, compiled, synthesis


@pytest.mark.parametrize("change", ["primitive", "phase", "event_wire", "aggregate_guard"])
def test_saved_actual_compiler_sequences_and_guards_reject_changed_semantics(compiler_inputs, change):
    routed, original, saved = compiler_inputs
    compiled, synthesis = original.copy(), copy.deepcopy(saved)
    if change == "primitive":
        index = next(i for i, item in enumerate(compiled) if item[0] == "t")
        compiled[index] = ("s", *compiled[index][1:])
    elif change == "phase":
        synthesis["approximants"][0]["global_phase"] += np.pi / 2
    elif change == "event_wire":
        index = next(i for i, item in enumerate(compiled) if item[0] == "reset")
        name, angle, wires, clbit = compiled[index]
        compiled[index] = (name, angle, ((wires[0] + 1) % 4,), clbit)
    else:
        synthesis["guarded_numerical_operator_error_upper_estimate"] *= 2
    with pytest.raises(ValueError, match="n2"):
        gate.replay_synthesis(routed, compiled, synthesis, time.monotonic() + 60, np)


@pytest.mark.parametrize("body", ["gate custom q;", "opaque q[0];", "cx q[0], q[0];", "reset q[4];",
                                  "rx(NaN) q[0];", "rx(1e400) q[0];", "c[4] = measure q[0];",
                                  "if (c[0]) x q[0];"])
def test_flat_qasm_parser_rejects_opaque_invalid_and_nonfinite_instructions(body):
    text = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[4] c;\nqubit[4] q;\n' + body
    with pytest.raises(ValueError, match="n2"):
        gate.parse_qasm(text)


def test_density_replay_requires_complete_distinct_readout_targets():
    instructions = [("reset", None, (0,), None), ("h", None, (0,), None),
                    ("measure", None, (0,), 0), ("measure", None, (0,), 1)]
    with pytest.raises(ValueError, match="distinct"):
        gate.replay_density(instructions, time.monotonic() + 60, np)


def test_density_replay_reset_is_full_kraus_discard_and_readout_is_unconditional():
    instructions = [("x", None, (0,), None), ("reset", None, (0,), None)]
    instructions += [("measure", None, (wire,), wire) for wire in range(4)]
    result = gate.replay_density(instructions, time.monotonic() + 60, np)
    assert result["acceptance_probability"] == pytest.approx(1.)
    assert result["mean"] == pytest.approx(1.)
    np.testing.assert_array_equal(result["reduced_before_output"], [[1., 0.], [0., 0.]])


def test_deadline_precedes_numerical_replay():
    with pytest.raises(ValueError, match="60-second"):
        gate.replay_density([], time.monotonic() - 1, np)


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"value":NaN}', '{"value":1e400}', '[]'])
def test_duplicate_nonfinite_and_invalid_json_records_reject(tmp_path, text):
    path = tmp_path / "record.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="n2"):
        gate.load_json(path)


def test_linked_bundle_and_file_reject(package):
    bundle = package / gate.BUNDLE_RELATIVE
    alias = package / "alias"
    alias.symlink_to(bundle)
    with pytest.raises(ValueError, match="regular directory"):
        gate.verify_bundle(alias, source_root=package, canonical_root=package, replay=False)
    path = bundle / "counts.svg"
    outside = package / "outside.svg"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="linked"):
        verify_identity(package)
