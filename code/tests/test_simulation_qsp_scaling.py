"""Independent fixed-norm, acceptance-budget and artifact-scope checks."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/generated/simulation/qsp-scaling"
SPEC = importlib.util.spec_from_file_location("qsp_scaling_generator", ROOT / "scripts/generate_simulation_qsp_scaling.py")
scaling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scaling)


@pytest.mark.parametrize("size,expected_width", [(1, 3), (2, 6), (3, 8), (4, 9), (5, 11)])
def test_family_keeps_total_norm_and_implements_the_declared_widths(size, expected_width):
    parameters = scaling.scaling_parameters(size)
    coupling = 4.1 / (1.7 * size - 1)
    assert parameters["J"] == pytest.approx(coupling)
    assert parameters["h"] == pytest.approx(0.7 * coupling)
    assert len(parameters["terms"]) == 2 * size - 1
    assert sum(abs(coefficient) for coefficient, _ in parameters["terms"]) == pytest.approx(4.1, abs=1e-12)
    assert all(len(label) == size for _, label in parameters["terms"])
    circuit = scaling.qe.build_qsp_evolution(parameters["terms"])
    assert circuit.num_qubits == expected_width
    assert circuit.metadata["n_index"] == math.ceil(math.log2(2 * size - 1))
    if size == 3:
        assert parameters["J"] == 1
        assert parameters["h"] == 0.7


@pytest.mark.parametrize("size", [True, 0, 17, 1.0, float("nan")])
def test_sweep_bound_rejects_invalid_sizes(size):
    with pytest.raises(ValueError, match="integer system qubits"):
        scaling.scaling_parameters(size)


def test_one_spin_observable_has_an_independent_analytic_answer():
    parameters = scaling.scaling_parameters(1)
    circuit = scaling.qe.build_qsp_evolution(parameters["terms"])
    checks = scaling._validate(circuit, scaling.decompose_trial(circuit), parameters)
    # For H=4.1X on |0>, <Z(t=1)> = cos(2*4.1), without an expm calculation.
    assert checks["observable_exact"] == pytest.approx(math.cos(8.2), abs=1e-14)
    assert checks["observable_conditional"] == pytest.approx(math.cos(8.2), abs=2e-8)
    assert checks["operator_validation_performed"] is True
    assert checks["acceptance_probability"] == pytest.approx(1 / 16, abs=2e-9)


def test_acceptance_budget_is_minimal_and_charges_every_complete_measured_trial():
    counts = {"cx": 12, "rz": 3, "measure": 5}
    projection = scaling.workload_projection(1 / 16, counts, 5, 10)
    accepted = projection["required_independent_accepted_Z_samples"]
    cap = projection["complete_attempt_cap_per_accepted_sample"]
    target = projection["per_sample_acceptance_failure_allowance"]
    assert accepted == 13315
    assert cap == 230
    assert (15 / 16) ** cap <= target < (15 / 16) ** (cap - 1)
    assert accepted * (15 / 16) ** cap <= 0.005
    total = accepted * cap
    assert projection["complete_attempt_cap_all_samples"] == total == 3062450
    assert projection["primitive_gate_counts_all_capped_attempts"] == {gate: count * total for gate, count in counts.items()}
    assert projection["primitive_operations_all_capped_attempts"] == 20 * total
    assert projection["logical_qubit_layer_volume_all_capped_attempts"] == 5 * 10 * total
    assert scaling.workload_projection(0.5, counts, 5, 10)["complete_attempt_cap_per_accepted_sample"] < cap
    assert projection["certified"] is False


@pytest.mark.parametrize("cap", [True, 0, 200001, 1.0])
def test_primitive_cap_cannot_silently_be_relaxed(tmp_path, cap):
    with pytest.raises(ValueError, match="primitive gate cap"):
        scaling.generate(tmp_path, max_primitive_gates=cap)


def test_constructor_and_simulation_bounds_are_separate(tmp_path):
    assert scaling.scaling_parameters(6)["term_count"] == 11
    with pytest.raises(ValueError, match="use --no-sim"):
        scaling.generate(tmp_path, qubits=6)
    with patch.object(scaling.hs, "dense_hamiltonian", side_effect=AssertionError("dense reference called")), patch.object(scaling, "_validate", side_effect=AssertionError("circuit simulation called")):
        result = scaling.generate(tmp_path, qubits=1, no_sim=True)
    row = result["rows"][0]
    assert row["state_validation_performed"] is False
    assert row["operator_validation_performed"] is False
    assert row["acceptance_probability"] is None
    assert row["complete_attempt_cap_all_samples"] is None
    resources = json.loads((tmp_path / "logical-resources.json").read_text())
    assert resources["instances"][0]["observable_workload_projection"] is None
    assert (tmp_path / "evolution-n1.qasm").exists()


def test_generated_larger_instances_do_not_claim_operator_validation_and_account_all_readout():
    results = json.loads((ASSETS / "results.json").read_text())
    resources = json.loads((ASSETS / "logical-resources.json").read_text())
    assert [item["parameters"]["n_system"] for item in results["instances"]] == [1, 2, 3, 4, 5]
    for item, ledger in zip(results["instances"], resources["instances"], strict=True):
        size = item["parameters"]["n_system"]
        checks = item["checks"]
        assert checks["operator_validation_performed"] is (size <= 3)
        if size >= 4:
            assert checks["encoded_block_error"] is None
            assert checks["decoded_linear_map_operator_error"] is None
            assert checks["finite_H_numerical_minimum_acceptance_probability"] is None
            assert checks["max_entry_error_source_vs_primitive_operator"] is None
            assert "not computed" in checks["validation_scope"]
        counts = ledger["measured_primitive_gate_counts_per_attempt"]
        assert counts["measure"] == ledger["peak_logical_qubits"]
        assert ledger["measured_dependency_depth_per_attempt"] == ledger["decomposed_unitary_dependency_depth_per_attempt"] + 1
        assert ledger["primitive_operations_per_measured_attempt"] == sum(counts.values()) <= 200000
        assert ledger["qasm_global_phase_exported"] is False
        assert ledger["transpiler_qubits_initially_zero"] is False
        assert "controlled oracle" in ledger["qasm_phase_scope"]
        projection = ledger["observable_workload_projection"]
        assert projection["state_specific_numerical_acceptance_probability"] == checks["acceptance_probability"]
        attempts = projection["complete_attempt_cap_all_samples"]
        assert projection["primitive_gate_counts_all_capped_attempts"] == {gate: count * attempts for gate, count in counts.items()}
        assert projection["logical_qubit_layer_volume_all_capped_attempts"] == ledger["peak_logical_qubits"] * ledger["measured_dependency_depth_per_attempt"] * attempts
        assert ledger["certified"] is False


def test_reusing_checked_output_for_no_sim_cleans_only_known_stale_artifacts(tmp_path):
    scaling.generate(tmp_path, qubits=1)
    assert (tmp_path / "accuracy-acceptance.svg").exists()
    assert (tmp_path / "evolution-n1.qasm").exists()
    (tmp_path / "logical-scaling.svg").write_text("obsolete generated plot")
    (tmp_path / "notes.txt").write_text("unrelated user file")
    scaling.generate(tmp_path, qubits=2, no_sim=True)
    assert not (tmp_path / "accuracy-acceptance.svg").exists()
    assert not (tmp_path / "evolution-n1.qasm").exists()
    assert not (tmp_path / "logical-scaling.svg").exists()
    assert (tmp_path / "evolution-n2.qasm").exists()
    assert (tmp_path / "scaling.svg").exists()
    assert (tmp_path / "notes.txt").read_text() == "unrelated user file"
    report = json.loads((tmp_path / "results.json").read_text())
    assert list(report["qasm_sha256"]) == ["evolution-n2.qasm"]
    assert report["family"]["system_sizes_simulated"] == []


def test_exact_copied_source_and_qasm_bytes_match_recorded_hashes():
    report = json.loads((ASSETS / "results.json").read_text())
    for name, expected_hash in {**report["source_sha256"], **report["qasm_sha256"]}.items():
        assert hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() == expected_hash
    assert (ASSETS / "generate_simulation_qsp_scaling.py").read_bytes() == (ROOT / "scripts/generate_simulation_qsp_scaling.py").read_bytes()
    assert b"\r\n" not in (ASSETS / "scaling.csv").read_bytes()


def test_standalone_bundle_uses_its_copied_modules(tmp_path):
    for name in scaling.SOURCE_NAMES:
        shutil.copyfile(ASSETS / name, tmp_path / name)
    script = tmp_path / "generate_simulation_qsp_scaling.py"
    code = (
        "import json,runpy,sys; d=runpy.run_path(sys.argv[1]); "
        "print(json.dumps({'repository':d['REPOSITORY'],'hs':d['hs'].__file__,'qe':d['qe'].__file__}))"
    )
    command = subprocess.run([sys.executable, "-c", code, str(script)], cwd=tmp_path,
                             capture_output=True, text=True, check=True)
    loaded = json.loads(command.stdout)
    assert loaded["repository"] is False
    assert Path(loaded["hs"]).parent == tmp_path
    assert Path(loaded["qe"]).parent == tmp_path
