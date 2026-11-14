"""Audit resource inputs, exact walk lowering, and complete fixed-shot budgets.

These tests import the worktree script explicitly. Native QDK trace import is
used for the exact Clifford+T check; physical resource estimates are mocked.
No full eleven-qubit operator or QPU experiment is performed.
"""

from collections import Counter
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator, Statevector


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/generated/optimization"


@pytest.fixture(scope="module")
def resource_script():
    path = ROOT / "scripts/generate_optimization_annealing_walk_resources.py"
    spec = importlib.util.spec_from_file_location("test_annealing_walk_resource_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.ROOT == ROOT
    return module


@pytest.fixture(scope="module")
def records():
    return {
        method: {
            "runs": json.loads((ASSETS / method / "runs.json").read_text()),
            "resources": json.loads((ASSETS / method / "resources.json").read_text()),
            "folder": ASSETS / method,
        }
        for method in ("annealing", "walk")
    }


@pytest.fixture(scope="module")
def implementation(resource_script, records):
    module, digest = resource_script.load_current_implementation(
        records["annealing"]["runs"], records["walk"]["runs"]
    )
    assert Path(module.__file__).resolve() in {
        ROOT / "src/openquantum/optimization.py", ASSETS / "annealing/optimization.py"}
    assert digest == records["annealing"]["runs"]["provenance"]["implementation_sha256"]
    assert digest == resource_script.sha_file(Path(module.__file__))
    return module


def set_path(value, path, replacement):
    for component in path[:-1]:
        value = value[component]
    value[path[-1]] = replacement


def validate(method, resource_script, records, implementation, *, run=None, resources=None, folder=None):
    entry = records[method]
    return getattr(resource_script, "validate_" + method + "_record")(
        entry["runs"] if run is None else run,
        entry["resources"] if resources is None else resources,
        implementation,
        entry["folder"] if folder is None else folder,
    )


def test_fixed_annealing_selection_counts_all_layers_and_readouts(resource_script, records, implementation):
    source, selection = validate("annealing", resource_script, records, implementation)
    assert selection["total_time"] == 16.0 and selection["steps"] == 256
    assert selection["logical_qubits"] == source.num_qubits == 4
    assert selection["total_preparations"] == 3 * 4096 == 12288
    assert selection["time_step_study_quantum_shots"] == 0
    assert selection["source_gate_counts"] == {
        "h": 4, "rx": 2048, "rz": 1024, "cx": 2048, "measure": 4, "barrier": 1
    }
    assert selection["source_unitary_gate_count"] == 4 + 2048 + 1024 + 2048
    assert selection["weighted_source_gate_counts"]["measure"] == 4 * 12288
    assert "not hardware seconds" in selection["evolution_time_units"]
    for gate in ("h", "rx", "rz", "cx", "measure"):
        assert selection["weighted_source_gate_counts"][gate] == source.count_ops()[gate] * 12288


@pytest.mark.parametrize("path,replacement", [
    (("task", "n_qubits"), 3),
    (("task", "edges"), [[0, 1], [1, 2], [2, 3]]),
    (("task", "driver"), "+sum_i X_i"),
    (("task", "problem"), "C"),
    (("task", "schedule"), "s=(t/T)**2"),
    (("main", "total_time"), 8.0),
    (("main", "steps"), 128),
    (("target", "predeclared_total_time"), 8.0),
    (("target", "predeclared_steps"), 128),
    (("readout", 0, "seed"), 19),
    (("readout", 0, "shots"), 2048),
    (("readout", 0, "optimal_count"), -1),
    (("readout", 0, "measured_qubit_readouts"), 4096),
    (("total_preparations",), 4096),
])
def test_annealing_selector_rejects_changed_problem_main_or_readout(
        path, replacement, resource_script, records, implementation):
    run = copy.deepcopy(records["annealing"]["runs"])
    set_path(run, path, replacement)
    with pytest.raises(ValueError):
        validate("annealing", resource_script, records, implementation, run=run)


@pytest.mark.parametrize("path,replacement", [
    (("main_parameters", "total_time"), 8.0),
    (("main_parameters", "steps"), 128),
    (("total_preparations",), 4096),
    (("logical_qubits",), 3),
    (("measured_qubit_readouts",), 4096),
    (("per_preparation_gate_counts", "rx"), 1024),
    (("circuit_exports", 0, "sha256"), "0" * 64),
])
def test_annealing_selector_rejects_incomplete_or_stale_logical_resource_metadata(
        path, replacement, resource_script, records, implementation):
    resources = copy.deepcopy(records["annealing"]["resources"])
    set_path(resources, path, replacement)
    with pytest.raises(ValueError):
        validate("annealing", resource_script, records, implementation, resources=resources)


def test_fixed_walk_selection_retains_workspace_and_every_oracle_shift(resource_script, records, implementation):
    source, compiled, selection = validate("walk", resource_script, records, implementation)
    assert selection["dimension"] == 4 and selection["marked_vertex"] == 5 and selection["steps"] == 4
    assert source.num_qubits == compiled.num_qubits == selection["logical_qubits"] == 11
    assert selection["active_qubits"] == 6 and selection["clean_workspace_qubits"] == 5
    assert selection["source_ccx_per_preparation"] == 40 + 16 == 56
    assert selection["literal_t_and_tdg_per_preparation"] == 7 * 56 == 392
    assert selection["total_preparations"] == 3 * 4096 == 12288
    assert selection["oracle_coin_shift_calls_all_preparations"] == 4 * 12288
    assert selection["individual_qubit_measurements"] == 11 * 12288
    assert selection["weighted_literal_clifford_t_counts"]["t"] == 224 * 12288
    assert selection["weighted_literal_clifford_t_counts"]["tdg"] == 168 * 12288
    assert selection["no_workspace_postselection"] is True


@pytest.mark.parametrize("path,replacement", [
    (("main", "dimension"), 2),
    (("main", "marked_vertex"), 6),
    (("main", "steps"), 3),
    (("main", "layout", "workspace_qubits"), 4),
    (("readout", 0, "seed"), 19),
    (("readout", 0, "shots"), 2048),
    (("readout", 0, "oracle_calls"), 4096),
    (("readout", 0, "coin_calls"), 4096),
    (("readout", 0, "shift_calls"), 4096),
    (("readout", 0, "circuit_preparations"), 2048),
    (("readout", 0, "individual_qubit_measurements"), 4096),
    (("readout", 0, "marked_samples"), -1),
    (("readout", 0, "position_counts", "0101"), -1),
])
def test_walk_selector_rejects_changed_main_or_missing_readout_work(
        path, replacement, resource_script, records, implementation):
    run = copy.deepcopy(records["walk"]["runs"])
    set_path(run, path, replacement)
    with pytest.raises(ValueError):
        validate("walk", resource_script, records, implementation, run=run)


def test_walk_selector_detects_workspace_outcomes_in_raw_counts(resource_script, records, implementation):
    run = copy.deepcopy(records["walk"]["runs"])
    counts = run["readout"][0]["counts"]
    bits = next(iter(counts))
    # Preserve the total shot count and the position, but set the first work
    # wire. It must be rejected rather than filtered from the marginal.
    changed = f"{int(bits, 2) | (1 << 6):011b}"
    counts[bits] -= 1
    if counts[bits] == 0:
        del counts[bits]
    counts[changed] = counts.get(changed, 0) + 1
    assert sum(counts.values()) == 4096
    with pytest.raises(ValueError, match="filtered|misdecoded|clean-work"):
        validate("walk", resource_script, records, implementation, run=run)


@pytest.mark.parametrize("path,replacement", [
    (("layout", "total_qubits"), 10),
    (("circuits", "source", "sha256"), "0" * 64),
    (("circuits", "clifford_t", "sha256"), "0" * 64),
    (("full_readout_workload", "circuit_preparations"), 4096),
    (("full_readout_workload", "individual_qubit_measurements"), 11 * 4096),
    (("full_readout_workload", "t_count"), 392 * 4096),
    (("full_readout_workload", "oracle_calls"), 4 * 4096),
    (("full_readout_workload", "coin_calls"), 4 * 4096),
    (("full_readout_workload", "shift_calls"), 4 * 4096),
])
def test_walk_selector_rejects_stale_exports_or_incomplete_totals(
        path, replacement, resource_script, records, implementation):
    resources = copy.deepcopy(records["walk"]["resources"])
    set_path(resources, path, replacement)
    with pytest.raises(ValueError):
        validate("walk", resource_script, records, implementation, resources=resources)


@pytest.mark.parametrize("method", ["annealing", "walk"])
@pytest.mark.parametrize("key", ["implementation_sha256", "generator_sha256", "environment_lock_sha256"])
def test_run_and_logical_provenance_must_match(method, key, resource_script, records, implementation):
    resources = copy.deepcopy(records[method]["resources"])
    resources["provenance"][key] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate(method, resource_script, records, implementation, resources=resources)


@pytest.mark.parametrize("method", ["annealing", "walk"])
def test_current_implementation_loader_rejects_stale_source_hash(method, resource_script, records):
    annealing, walk = (copy.deepcopy(records[name]["runs"]) for name in ("annealing", "walk"))
    (annealing if method == "annealing" else walk)["provenance"]["implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="executed implementation"):
        resource_script.load_current_implementation(annealing, walk)


@pytest.mark.parametrize("method,export_kind", [("annealing", "annealing"), ("walk", "source"), ("walk", "clifford_t")])
def test_actual_export_bytes_must_match_the_counted_compilation(
        method, export_kind, tmp_path, resource_script, records, implementation):
    resources = records[method]["resources"]
    exports = resources["circuit_exports"] if method == "annealing" else list(resources["circuits"].values())
    for exported in exports:
        path = records[method]["folder"] / exported["file"]
        (tmp_path / exported["file"]).write_bytes(path.read_bytes())
    target = exports[0] if method == "annealing" else resources["circuits"][export_kind]
    path = tmp_path / target["file"]
    path.write_bytes(path.read_bytes() + b"\n// unrelated stale export\n")
    with pytest.raises(ValueError, match="export hashes|export hashes|export hashes|export"):
        validate(method, resource_script, records, implementation, folder=tmp_path)


@pytest.mark.parametrize("counts", [
    {"0000": 4095},
    {"0000": 4097, "0001": -1},
    {"0000": True, "0001": 4095},
    {"0000": 4095.0, "0001": 1},
    {"000": 4096},
    {"000x": 4096},
])
def test_count_validation_rejects_missing_or_malformed_outcomes(counts, resource_script):
    with pytest.raises(ValueError):
        resource_script.verify_count_record({"shots": 4096, "counts": counts}, width=4)


def test_readout_map_rejects_missing_duplicate_and_midcircuit_measurements(resource_script):
    missing = QuantumCircuit(2, 2)
    missing.measure(0, 0)
    duplicate = QuantumCircuit(2, 2)
    duplicate.measure(0, 0)
    duplicate.measure(1, 0)
    midcircuit = QuantumCircuit(2, 2)
    midcircuit.measure(0, 0)
    midcircuit.x(1)
    midcircuit.measure(1, 1)
    for circuit in (missing, duplicate, midcircuit):
        with pytest.raises(ValueError):
            resource_script.measured_map(circuit)


@pytest.fixture(scope="module")
def walk_pipeline(resource_script, records, implementation):
    source, compiled, _ = validate("walk", resource_script, records, implementation)
    sealed = resource_script.seal_clifford_t(compiled)
    lowered = resource_script.prepare_measured_circuit(sealed)
    return source, compiled, sealed, lowered


def instruction_sequence(circuit):
    return [
        (item.operation.name, tuple(float(p) for p in item.operation.params),
         tuple(circuit.find_bit(q).index for q in item.qubits),
         tuple(circuit.find_bit(c).index for c in item.clbits))
        for item in circuit.data if item.operation.name != "barrier"
    ]


def test_seal_is_no_op_and_preserves_every_literal_gate_and_measurement(resource_script, walk_pipeline):
    _, compiled, sealed, lowered = walk_pipeline
    assert compiled.count_ops()["t"] == sealed.count_ops()["t"] == lowered.count_ops()["t"] == 224
    assert compiled.count_ops()["tdg"] == sealed.count_ops()["tdg"] == lowered.count_ops()["tdg"] == 168
    assert sealed.count_ops()["barrier"] == 974
    assert instruction_sequence(compiled) == instruction_sequence(sealed) == instruction_sequence(lowered)
    assert resource_script.measured_map(sealed) == [[i, i] for i in range(11)]
    assert all(len(item.qubits) == 11 for item in sealed.data if item.operation.name == "barrier")


def test_seal_preserves_nontrivial_global_phase_and_does_not_mutate_source(resource_script, walk_pipeline):
    _, compiled, _, _ = walk_pipeline
    circuit = compiled.copy()
    circuit.global_phase = 0.371
    before = qasm3.dumps(circuit)
    sealed = resource_script.seal_clifford_t(circuit)
    assert float(sealed.global_phase) == pytest.approx(0.371)
    assert qasm3.dumps(circuit) == before
    assert instruction_sequence(sealed) == instruction_sequence(circuit)


def test_seal_rejects_arbitrary_rotations_and_missing_measurements(resource_script):
    rotation = QuantumCircuit(2)
    rotation.rz(0.2, 0)
    rotation.measure_all()
    missing = QuantumCircuit(2)
    missing.t(0)
    for circuit in (rotation, missing):
        with pytest.raises(ValueError):
            resource_script.seal_clifford_t(circuit)


def test_actual_shared_lowered_walk_canonical_trace_has_392_t_and_zero_rotations(resource_script, walk_pipeline):
    _, _, _, lowered = walk_pipeline
    audit = resource_script.canonical_trace_audit(lowered)
    assert audit["canonical_compute_qubits"] == 11
    assert audit["canonical_gate_counts"] == {"T": 392, "MEAS_Z": 11}
    assert audit["base_error"] == 0
    assert audit["qasm_sha256"] == resource_script.sha_text(qasm3.dumps(lowered))
    allowance, description = resource_script.choose_walk_synthesis_allowance(audit)
    assert allowance == 0
    assert "literal Clifford+T" in description
    assert "cost abstraction" in audit["scope"]


@pytest.mark.parametrize("counts", [
    {"T": 391, "MEAS_Z": 11}, {"T": 393, "MEAS_Z": 11},
    {"T": 392, "RZ": 1, "MEAS_Z": 11}, {"T": 392, "CCX": 1, "MEAS_Z": 11},
    {"MEAS_Z": 11},
])
def test_changed_t_or_arbitrary_rotation_representation_retains_positive_model_allowance(counts, resource_script):
    allowance, description = resource_script.choose_walk_synthesis_allowance({"canonical_gate_counts": counts})
    assert allowance == 1e-5
    assert "conditional mean-fit" in description


def test_local_exact_ccx_and_selected_state_scope_never_allocate_11q_operator(
        monkeypatch, resource_script, implementation, walk_pipeline):
    source, compiled, sealed, lowered = walk_pipeline
    original_operator = resource_script.Operator
    operator_widths = []

    def bounded_operator(circuit):
        operator_widths.append(circuit.num_qubits)
        assert circuit.num_qubits <= 4, "a full eleven-qubit operator is outside this check"
        return original_operator(circuit)

    monkeypatch.setattr(resource_script, "Operator", bounded_operator)
    check = resource_script.verify_walk_lowering(source, compiled, sealed, lowered, implementation)
    assert operator_widths == [3]
    assert check["local_ccx_full_operator_equivalence_verified"]
    assert check["local_ccx_t_count"] == 7
    assert check["selected_full_11_qubit_prepared_state_equivalence_verified"]
    assert check["selected_state_max_phase_aligned_amplitude_error"] < 1e-11
    assert check["workspace_leakage"] < 1e-24
    assert check["full_11_qubit_operator_verified"] is False
    # An independent 8-state classical permutation specifies the local gate.
    gate = QuantumCircuit(3)
    gate.ccx(0, 1, 2)
    permutation = np.zeros((8, 8), dtype=complex)
    for column in range(8):
        row = column ^ (4 if column & 3 == 3 else 0)
        permutation[row, column] = 1
    assert Operator(implementation.compile_walk_clifford_t(gate)).equiv(Operator(permutation), atol=1e-12)
    a = Statevector.from_instruction(source.remove_final_measurements(inplace=False))
    b = Statevector.from_instruction(lowered.remove_final_measurements(inplace=False))
    assert a.equiv(b, atol=1e-11)
    assert np.sum(np.abs(b.data[64:]) ** 2) < 1e-24


def test_annealing_operator_check_is_distinct_from_adiabatic_error_bounds(resource_script, records, implementation):
    source, _ = validate("annealing", resource_script, records, implementation)
    lowered = resource_script.prepare_measured_circuit(source)
    check = resource_script.verify_annealing_lowering(source, lowered)
    assert check["full_4_qubit_operator_equivalence_verified"]
    assert check["max_phase_aligned_operator_element_error"] < 1e-10
    assert check["not_an_error_bound_against_continuous_adiabatic_evolution"]
    bad = lowered.remove_final_measurements(inplace=False)
    bad.x(0)
    bad.measure_all()
    with pytest.raises(ValueError, match="Operator"):
        resource_script.verify_annealing_lowering(source, bad)


def test_phase_alignment_discards_only_global_phase(resource_script):
    left = np.array([1.0, 0.0], dtype=complex)
    assert resource_script.phase_aligned_error(left, np.exp(0.371j) * left) < 1e-15
    assert math.isinf(resource_script.phase_aligned_error(left, np.array([0.0, 1.0])))


def test_generate_allocates_all_recorded_preparations_with_mocked_physical_estimates(
        tmp_path, monkeypatch, resource_script, records):
    calls = []

    def mock_estimate(circuit, *, budget, profile, cache_dir, use_graph):
        # This is a contract test only. The synthetic cost point is never a
        # published QDK result; no real physical estimate is invoked here.
        lowered = resource_script.prepare_measured_circuit(circuit)
        audit = resource_script.canonical_trace_audit(lowered)
        calls.append((circuit.num_qubits, budget, profile, cache_dir, use_graph, audit))
        base = 0.0 if budget.synthesis_model_error_per_execution == 0 else 1e-6
        point = {"peak_physical_qubits": 100, "runtime_per_execution_ns": 1000.0,
                 "serial_runtime_all_executions_ns": 1000.0 * budget.executions,
                 "physical_failure_union_bound_all_executions": budget.physical_failure_probability / 2}
        return {"qasm_sha256": audit["qasm_sha256"], "canonical_gate_counts": audit["canonical_gate_counts"],
                "synthesis": {"model_error_per_execution": base}, "pareto_points": [point]}

    monkeypatch.setattr(resource_script, "estimate_circuit_resources", mock_estimate)
    output = tmp_path / "only-test-output"
    cache = tmp_path / "only-test-cache"
    report = resource_script.generate(records["annealing"]["folder"], records["walk"]["folder"], output, cache)
    assert len(calls) == 4
    assert Counter((width, profile.physical_error_rate) for width, _, profile, _, _, _ in calls) == {
        (4, 1e-3): 1, (11, 1e-3): 1, (4, 1e-4): 1, (11, 1e-4): 1
    }
    for width, budget, profile, recorded_cache, use_graph, audit in calls:
        assert budget.executions == 12288
        assert budget.physical_failure_probability == 1e-3
        assert budget.physical_failure_per_execution == pytest.approx(1e-3 / 12288)
        assert profile.gate_time_ns == 100 and profile.measurement_time_ns == 500
        assert recorded_cache == cache and use_graph is False
        assert audit["canonical_gate_counts"]["MEAS_Z"] == width
        assert budget.synthesis_model_error_per_execution == (0 if width == 11 else 1e-5)
    assert report["synthesis_model_scalars_summed_over_shots_as_random_faults"] is False
    assert report["no_hardware_execution_or_success_certificate"] is True
    assert report["no_generic_gap_adiabatic_optimizer_or_speedup_guarantee"] is True
    assert report["methods"]["walk"]["selection"]["weighted_literal_clifford_t_counts"]["t"] == 224 * 12288
    assert report["methods"]["walk"]["selection"]["individual_qubit_measurements"] == 11 * 12288
    assert report["methods"]["annealing"]["selection"]["weighted_source_gate_counts"]["measure"] == 4 * 12288
    assert all(row["executions"] == 12288 and row["serial_runtime_ns"] == 1000 * 12288 for row in report["choices"])
    assert (output / "report.json").is_file() and (output / "summary.csv").is_file()


def test_historical_replay_uses_both_hash_bound_source_copies(resource_script, records, tmp_path, monkeypatch):
    source = tmp_path / "src/openquantum/optimization.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('the unrelated live module must not be imported')\n")
    copies = []
    for method in ("annealing", "walk"):
        path = tmp_path / "assets/generated/optimization" / method / "optimization.py"
        path.parent.mkdir(parents=True)
        path.write_bytes((ASSETS / method / "optimization.py").read_bytes())
        copies.append(path)
    monkeypatch.setattr(resource_script, "ROOT", tmp_path)
    module, digest = resource_script.load_current_implementation(records["annealing"]["runs"], records["walk"]["runs"])
    assert Path(module.__file__) == copies[0]
    assert digest == records["annealing"]["runs"]["provenance"]["implementation_sha256"]
    copies[1].write_text(copies[1].read_text() + "\n# changed source bytes\n")
    with pytest.raises(ValueError, match="recorded source copies"):
        resource_script.load_current_implementation(records["annealing"]["runs"], records["walk"]["runs"])
