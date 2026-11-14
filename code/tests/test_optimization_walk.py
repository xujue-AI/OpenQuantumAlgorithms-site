"""Independent SKW walk, clean-workspace, and complete readout checks."""

import math
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from openquantum import optimization as opt


def dense_skw_step(dimension=4, marked=5):
    """Build the ideal active-space SKW unitary without a quantum generator.

    The active basis is position + 2**dimension * coin. The marked coin is
    minus identity; each unmarked coin is the Grover reflection. The shift
    flips position bit selected by the coin label, which it leaves unchanged.
    """
    positions = 1 << dimension
    size = positions * dimension
    uniform_coin = np.full(dimension, 1 / math.sqrt(dimension))
    grover = 2 * np.outer(uniform_coin, uniform_coin) - np.eye(dimension)
    coin_operator = np.zeros((size, size), dtype=complex)
    shift = np.zeros((size, size), dtype=complex)
    for position in range(positions):
        coin = -np.eye(dimension) if position == marked else grover
        for source_coin in range(dimension):
            source = position + positions * source_coin
            shift[(position ^ (1 << source_coin)) + positions * source_coin, source] = 1
            for output_coin in range(dimension):
                target = position + positions * output_coin
                coin_operator[target, source] = coin[output_coin, source_coin]
    return shift @ coin_operator


def dense_skw_state(dimension, marked, steps):
    state = np.full(dimension * (1 << dimension), 1 / math.sqrt(dimension * (1 << dimension)), dtype=complex)
    step = dense_skw_step(dimension, marked)
    for _ in range(steps):
        state = step @ state
    return state


def independent_position_probabilities(active_state, dimension):
    return (np.abs(np.asarray(active_state).reshape(dimension, 1 << dimension)) ** 2).sum(axis=0)


def source_encoding(ideal_state, dimension):
    # The independent reference above puts position low. The implementation
    # puts coin low, so make the permutation explicit rather than sharing its
    # indexing routine or its independently supplied reference implementation.
    return np.asarray(ideal_state).reshape(dimension, 1 << dimension).T.ravel()


def embed_clean_active(active, total_qubits):
    state = np.zeros(1 << total_qubits, dtype=complex)
    state[:len(active)] = active
    return state


@pytest.mark.parametrize("steps,expected_probability", list(enumerate([1 / 16, 1 / 16, 1 / 4, 1 / 4, 25 / 64, 25 / 64, 1 / 4, 1 / 4, 1 / 256])))
def test_full_walk_matches_independent_skw_state_without_workspace_postselection(steps, expected_probability):
    ideal = dense_skw_state(4, 5, steps)
    expected = (-1) ** steps * source_encoding(ideal, 4)
    circuit = opt.build_hypercube_walk_circuit(4, 5, steps)
    actual = Statevector.from_instruction(circuit).data
    assert circuit.num_qubits == 11
    np.testing.assert_allclose(actual, embed_clean_active(expected, 11), atol=1e-12, rtol=0)
    assert np.sum(np.abs(actual[64:]) ** 2) < 1e-24
    assert np.sum(np.abs(actual[:64]) ** 2) == pytest.approx(1, abs=1e-12)
    # The clean amplitudes are used without normalization or filtering.
    position_probabilities = independent_position_probabilities(ideal, 4)
    assert position_probabilities[5] == pytest.approx(expected_probability, abs=1e-12)
    source_reference = opt.hypercube_walk_reference(4, 5, steps)
    np.testing.assert_allclose(source_reference, source_encoding(ideal, 4), atol=1e-12, rtol=0)


@pytest.mark.parametrize("marked,coin", [(marked, coin) for marked in (0, 5, 15) for coin in range(4)])
def test_joint_mark_oracle_is_a_rank_one_coin_reflection_on_clean_inputs(marked, coin):
    oracle = opt.build_hypercube_mark_oracle(4, marked)
    active = np.zeros(64, dtype=complex)
    active[4 * marked + coin] = 1
    expected = active.copy()
    expected[4 * marked:4 * (marked + 1)] -= 0.5
    actual = Statevector(embed_clean_active(active, 11)).evolve(oracle).data
    np.testing.assert_allclose(actual, embed_clean_active(expected, 11), atol=1e-12, rtol=0)


@pytest.mark.parametrize("position,coin", [(1, 0), (4, 1), (6, 2), (14, 3)])
def test_joint_oracle_leaves_unmarked_active_basis_inputs_unchanged(position, coin):
    active = np.zeros(64, dtype=complex)
    active[4 * position + coin] = 1
    expected = embed_clean_active(active, 11)
    actual = Statevector(expected).evolve(opt.build_hypercube_mark_oracle(4, 5)).data
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_joint_oracle_flips_uniform_coin_and_preserves_orthogonal_coin_at_the_mark():
    uniform = np.zeros(64, dtype=complex)
    uniform[20:24] = 0.5
    orthogonal = np.zeros(64, dtype=complex)
    orthogonal[20:22] = [1 / math.sqrt(2), -1 / math.sqrt(2)]
    oracle = opt.build_hypercube_mark_oracle(4, 5)
    for input_state, expected in ((uniform, -uniform), (orthogonal, orthogonal)):
        actual = Statevector(embed_clean_active(input_state, 11)).evolve(oracle).data
        np.testing.assert_allclose(actual, embed_clean_active(expected, 11), atol=1e-12, rtol=0)


def test_shift_selects_position_bit_using_the_coin_and_uncomputes_workspace():
    shift = opt.build_hypercube_shift(4)
    for position in range(16):
        for coin in range(4):
            actual = Statevector.from_int(4 * position + coin, 2048).evolve(shift).data
            expected = np.zeros(2048, dtype=complex)
            expected[4 * (position ^ (1 << coin)) + coin] = 1
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_main_circuit_counts_include_explicit_oracle_and_clean_and_chain():
    oracle = opt.build_hypercube_mark_oracle(4, 5)
    coin = opt.build_hypercube_coin(4)
    shift = opt.build_hypercube_shift(4)
    assert oracle.count_ops()["ccx"] == 2 * (6 - 1)
    assert oracle.count_ops()["z"] == 1
    assert coin.count_ops()["cz"] == 1
    assert shift.count_ops()["ccx"] == 4
    main = opt.build_hypercube_walk_circuit(4, 5, 4, measure=True)
    assert main.count_ops()["ccx"] == 4 * (10 + 4)
    assert main.count_ops()["measure"] == 11
    assert set(main.count_ops()) <= {"h", "x", "z", "cz", "cx", "ccx", "barrier", "measure"}
    assert main.metadata["oracle_calls"] == main.metadata["coin_calls"] == main.metadata["shift_calls"] == 4


def test_toffoli_lowering_is_an_actual_seven_t_sequence_with_equivalent_operator():
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    lowered = opt.compile_walk_clifford_t(circuit)
    assert Operator(circuit).equiv(Operator(lowered), atol=1e-12)
    assert lowered.count_ops().get("t", 0) + lowered.count_ops().get("tdg", 0) == 7
    assert set(lowered.count_ops()) <= {"h", "x", "z", "s", "sdg", "t", "tdg", "cx"}


def test_full_walk_clifford_t_lowering_preserves_all_clean_state_amplitudes():
    main = opt.build_hypercube_walk_circuit(4, 5, 4)
    lowered = opt.compile_walk_clifford_t(main)
    assert lowered.num_qubits == 11
    assert lowered.count_ops().get("t", 0) + lowered.count_ops().get("tdg", 0) == 7 * 56
    assert set(lowered.count_ops()) <= {"h", "x", "z", "s", "sdg", "t", "tdg", "cx"}
    expected = embed_clean_active(source_encoding(dense_skw_state(4, 5, 4), 4), 11)
    actual = Statevector.from_instruction(lowered).data
    assert Statevector(actual).equiv(Statevector(expected), atol=1e-11)
    assert np.sum(np.abs(actual[64:]) ** 2) < 1e-22
    assert sum(lowered.count_ops().values()) <= main.metadata["primitive_cap_cost"] <= 100_000


def test_simulation_position_marginal_retains_every_workspace_outcome():
    simulation = opt.simulate_hypercube_walk(4, 5, 4)
    probabilities = simulation["probabilities"]
    marginal = np.zeros(16)
    for index, probability in enumerate(probabilities):
        marginal[(index >> 2) & 15] += probability
    np.testing.assert_allclose(simulation["position_probabilities"], marginal, atol=1e-12, rtol=0)
    assert simulation["workspace_leakage"] == pytest.approx(sum(probabilities[64:]), abs=1e-24)
    assert simulation["marked_probability"] == pytest.approx(25 / 64, abs=1e-12)
    assert simulation["reference_overlap_squared"] == pytest.approx(1, abs=1e-12)
    assert simulation["reference_max_amplitude_error"] < 1e-12


@pytest.mark.parametrize("dimension,total_qubits", [(2, 5), (4, 11), (8, 21), (16, 39)])
def test_generator_width_and_primitive_cap_are_separate_from_simulation(dimension, total_qubits):
    layout = opt.walk_register_layout(dimension)
    assert layout["total_qubits"] == total_qubits
    assert layout["workspace_qubits"] == layout["active_qubits"] - 1
    circuit = opt.build_hypercube_walk_circuit(dimension, 1, 4, measure=True)
    assert circuit.num_qubits == total_qubits
    assert circuit.count_ops()["measure"] == total_qubits
    assert circuit.metadata["primitive_cap_cost"] <= 100_000


def test_large_simulation_and_dense_reference_fail_before_statevector_construction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported simulation must fail before constructing a statevector")

    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    with pytest.raises(ValueError):
        opt.simulate_hypercube_walk(8, 5, 4, max_qubits=100)
    with pytest.raises(ValueError):
        opt.hypercube_walk_reference(8, 5, 4, max_active_qubits=100)


@pytest.mark.parametrize("dimension,marked,steps", [(3, 0, 1), (32, 0, 1), (True, 0, 1), (4, -1, 1), (4, 16, 1), (4, True, 1), (4, 5, -1), (4, 5, 33), (4, 5, True)])
def test_invalid_walk_inputs_are_rejected(dimension, marked, steps):
    with pytest.raises(ValueError):
        opt.build_hypercube_walk_circuit(dimension, marked, steps)


def test_fixed_state_repetition_budget_is_minimal_for_a_marked_readout():
    trials = opt.walk_repetitions(25 / 64, 0.99)
    assert 1 - (1 - 25 / 64) ** trials >= 0.99
    assert 1 - (1 - 25 / 64) ** (trials - 1) < 0.99
    assert opt.walk_repetitions(1, 0.99) == 1


@pytest.fixture(scope="module")
def walk_generated(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("walk_generator_test", root / "scripts/generate_optimization_walk.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    output = tmp_path_factory.mktemp("walk-artifact-contract")
    # Figures receive separate visual QA; this test audits their computational
    # source, exports, and complete ledger without rendering duplicates.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generator, "draw_one_step", lambda *args: None)
        patch.setattr(generator, "draw_probability_scan", lambda *args: None)
        generator.generate(output)
    return output, json.loads((output / "runs.json").read_text()), json.loads((output / "resources.json").read_text())


def test_generated_readout_counts_and_ledgers_cover_all_preparations(walk_generated):
    _, runs, resources = walk_generated
    readout = runs["readout"]
    assert [row["seed"] for row in readout] == [7, 19, 41]
    for row in readout:
        assert sum(row["counts"].values()) == row["shots"] == 4096
        independent_position_counts = np.zeros(16, dtype=int)
        for bits, count in row["counts"].items():
            assert len(bits) == 11
            value = int(bits, 2)
            independent_position_counts[(value >> 2) & 15] += count
            assert value >> 6 == 0
        assert row["marked_samples"] == independent_position_counts[5]
        assert row["marked_sample_fraction"] == pytest.approx(independent_position_counts[5] / 4096)
        observed_probability = independent_position_counts[5] / 4096
        assert row["estimated_standard_error"] == pytest.approx(math.sqrt(observed_probability * (1 - observed_probability) / 4096))
        assert row["exact_marked_probability"] == pytest.approx(25 / 64, abs=1e-12)
        assert row["position_counts"] == {f"{position:04b}": int(count) for position, count in enumerate(independent_position_counts) if count}
        assert row["circuit_preparations"] == 4096
        assert row["oracle_calls"] == row["coin_calls"] == row["shift_calls"] == 4 * 4096
        assert row["individual_qubit_measurements"] == 11 * 4096
        assert row["workspace_nonzero_samples"] == 0
        assert row["hoeffding_half_width_95"] == pytest.approx(math.sqrt(math.log(40) / (2 * 4096)))
    preparations = sum(row["circuit_preparations"] for row in readout)
    assert resources["full_readout_workload"]["circuit_preparations"] == preparations == 12288
    assert resources["full_readout_workload"]["individual_qubit_measurements"] == 11 * preparations
    for gate, total in resources["full_readout_source_gate_totals"].items():
        assert total == resources["circuits"]["source"]["gate_counts"].get(gate, 0) * preparations
    for gate, total in resources["full_readout_clifford_t_gate_totals"].items():
        assert total == resources["circuits"]["clifford_t"]["gate_counts"].get(gate, 0) * preparations
    assert resources["full_readout_workload"]["t_count"] == 392 * preparations
    trials = runs["fixed_state_repetition"]["trials"]
    for item, per_preparation in resources["per_preparation"].items():
        assert resources["fixed_state_repetition_workload"][item] == per_preparation * trials


def test_generated_source_hashes_oracle_coverage_and_unsimulated_exports(walk_generated):
    output, runs, resources = walk_generated
    for variant in resources["circuits"].values():
        assert hashlib.sha256((output / variant["file"]).read_bytes()).hexdigest() == variant["sha256"]
    assert hashlib.sha256((output / "optimization.py").read_bytes()).hexdigest() == runs["provenance"]["implementation_sha256"]
    assert runs["clean_subspace_oracle_check"]["checked_inputs"] == 64
    assert runs["clean_subspace_oracle_check"]["max_workspace_leakage"] < 1e-24
    assert resources["synthesis"]["arbitrary_rotations"] == 0
    assert resources["synthesis"]["approximation_error"] == 0
    for export in resources["source_only_scalability_exports"]:
        assert export["dimension"] in (8, 16)
        assert export["simulation_executed"] is False
        assert export["validated_success_probability"] is None
