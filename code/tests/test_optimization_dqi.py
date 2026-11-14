# SPDX-License-Identifier: MIT
"""Deferred DQI checks: circuits are constructed only when tests are invoked.

These references retain complex phases and use literal bit maps or direct
classical parity sums rather than the production decoder/objective helpers.
The source-only checkpoint does not assert that these tests have been run.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator, Statevector

from openquantum.optimization_dqi import (
    DEFAULT_MATRIX,
    DEFAULT_SIGNS,
    DQILimits,
    PRIMITIVE_GATES,
    PROGRAM_STATUS,
    append_controlled_ry,
    append_dirty_c3x,
    append_exact_ccx,
    bounded_decoder_operator,
    bounded_statevector,
    build_decoder_circuit,
    build_dqi_circuit,
    build_preparation_circuit,
    decoder_extension,
    decoder_table,
    logical_resources,
    polynomial_reference,
    primitive_operations,
    register_map,
    validate_instance,
    xor_objective_values,
)


def _stage_circuit(circuit, name):
    """Copy one emitted primitive stage, preserving global wire indices."""
    stage = next(value for value in circuit.metadata["stages"] if value["name"] == name)
    result = QuantumCircuit(circuit.num_qubits)
    for item in circuit.data[stage["start"]:stage["stop"]]:
        result.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    return result


def _direct_values(matrix, signs):
    """Independent binary-equation truth table with integer bit j equal to xj."""
    return tuple(
        sum(sum(row[j] * ((x >> j) & 1) for j in range(len(row))) % 2 == sign
            for row, sign in zip(matrix, signs, strict=True))
        for x in range(1 << len(matrix[0]))
    )


def _direct_final_state(matrix, signs):
    values = _direct_values(matrix, signs)
    n, m = len(matrix[0]), len(matrix)
    result = np.zeros(1 << (n + m + 2), dtype=complex)
    for x, value in enumerate(values):
        result[x << m] = (1 + 2 * value - m) / math.sqrt((1 << n) * (m + 1))
    return result


def test_exact_ccx_has_the_full_truth_table_without_phase_fitting():
    circuit = QuantumCircuit(3)
    append_exact_ccx(circuit, 0, 1, 2)
    expected = np.zeros((8, 8), dtype=complex)
    for column in range(8):
        target = column ^ (4 if (column & 3) == 3 else 0)
        expected[target, column] = 1
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-12, rtol=0)
    assert len(circuit.data) == 15
    assert dict(circuit.count_ops()) == {"h": 2, "cx": 6, "tdg": 3, "t": 4}
    assert circuit.global_phase == 0


@pytest.mark.parametrize("angle", [0.0, 0.37, -1.23, math.pi])
def test_controlled_ry_matches_an_independent_complex_matrix(angle):
    circuit = QuantumCircuit(2)
    append_controlled_ry(circuit, 0, 1, angle)
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    # q0 is the low bit/control, q1 is the target. Control-zero columns are fixed.
    expected = np.array([[1, 0, 0, 0], [0, c, 0, -s],
                         [0, 0, 1, 0], [0, s, 0, c]], dtype=complex)
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-12, rtol=0)
    assert dict(circuit.count_ops()) == {"ry": 2, "cx": 2}


def test_controlled_ry_rejects_inexact_subnormal_halves_before_adding_any_gate(monkeypatch):
    def forbidden_sampling_or_fitting(*args, **kwargs):
        raise AssertionError("Subnormal rejection must not sample or fit")

    for name in ("default_rng", "RandomState", "Generator", "seed", "choice",
                 "multinomial", "binomial", "poisson", "normal", "uniform",
                 "standard_normal", "random", "random_sample", "sample",
                 "rand", "randn", "randint"):
        if hasattr(np.random, name):
            monkeypatch.setattr(np.random, name, forbidden_sampling_or_fitting)
    monkeypatch.setattr(np, "polyfit", forbidden_sampling_or_fitting)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.global_phase = 0.125
    snapshot = circuit.copy()
    before = primitive_operations(circuit)
    angles = (
        float.fromhex("0x0.0000000000001p-1022"),
        -float.fromhex("0x0.0000000000001p-1022"),
        float.fromhex("0x0.0000000000003p-1022"),
        -float.fromhex("0x0.0000000000003p-1022"),
    )
    for angle in angles:
        assert math.isfinite(angle)
        assert 2 * (angle / 2) != angle
        with pytest.raises(ValueError, match="survive halving exactly"):
            append_controlled_ry(circuit, 0, 1, angle)
        assert primitive_operations(circuit) == before
        assert circuit == snapshot


def test_dirty_three_control_x_restores_both_scratch_values_without_phase():
    circuit = QuantumCircuit(5)
    append_dirty_c3x(circuit, [0, 1, 2], 3, 4)
    expected = np.zeros((32, 32), dtype=complex)
    for column in range(32):
        target = column ^ (8 if (column & 7) == 7 else 0)
        expected[target, column] = 1
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-12, rtol=0)
    assert len(circuit.data) == 60
    assert set(circuit.count_ops()) == {"h", "t", "tdg", "cx"}


def test_default_preparation_has_only_five_positive_amplitudes_and_clean_workspace():
    circuit = build_preparation_circuit()
    expected = np.zeros(64, dtype=complex)
    expected[[0, 1, 2, 4, 8]] = 1 / math.sqrt(5)
    np.testing.assert_allclose(bounded_statevector(circuit).data, expected, atol=2e-12, rtol=0)
    assert circuit.num_qubits == 6
    assert dict(circuit.count_ops()) == {"ry": 7, "cx": 42, "x": 12,
                                       "h": 12, "t": 24, "tdg": 18}
    assert len(circuit.data) == 115
    expected_angles = [2 * math.asin(1 / math.sqrt(5 - i)) for i in range(4)]
    assert circuit.metadata["angles"] == expected_angles
    assert circuit.metadata["angle_hex"] == [angle.hex() for angle in expected_angles]
    assert circuit.metadata["status"] == PROGRAM_STATUS


@pytest.mark.parametrize("m", [1, 2, 3])
def test_smaller_bounded_preparations_use_the_same_zero_prefix_law(m):
    circuit = build_preparation_circuit(m)
    expected = np.zeros(1 << (m + 2), dtype=complex)
    expected[[0] + [1 << j for j in range(m)]] = 1 / math.sqrt(m + 1)
    np.testing.assert_allclose(bounded_statevector(circuit).data, expected, atol=2e-12, rtol=0)


def test_sign_and_all_six_syndrome_gates_use_the_declared_bit_order():
    circuit = build_dqi_circuit()
    sign = _stage_circuit(circuit, "clause_sign_phase")
    syndrome = _stage_circuit(circuit, "syndrome_B_transpose_y")
    assert primitive_operations(sign) == [["z", [3], [], []]]
    assert primitive_operations(syndrome) == [
        ["cx", [0, 4], [], []], ["cx", [1, 5], [], []],
        ["cx", [2, 6], [], []], ["cx", [3, 4], [], []],
        ["cx", [3, 5], [], []], ["cx", [3, 6], [], []],
    ]
    # Check all 16 clause inputs, including the 11 outside the cleanup promise.
    for clause in range(16):
        expected_syndrome = ((clause & 1) ^ ((clause >> 3) & 1))
        expected_syndrome |= (((clause >> 1) & 1) ^ ((clause >> 3) & 1)) << 1
        expected_syndrome |= (((clause >> 2) & 1) ^ ((clause >> 3) & 1)) << 2
        expected = np.zeros(512, dtype=complex)
        expected[clause | (expected_syndrome << 4)] = 1
        initial = np.zeros(512, dtype=complex)
        initial[clause] = 1
        np.testing.assert_allclose(Statevector(initial).evolve(syndrome).data,
                                   expected, atol=2e-12, rtol=0)
        sign_expected = initial * (-1 if clause & 8 else 1)
        np.testing.assert_allclose(Statevector(initial).evolve(sign).data,
                                   sign_expected, atol=2e-12, rtol=0)


def test_decoder_is_the_declared_permutation_on_all_512_basis_states():
    circuit = build_decoder_circuit()
    literal_table = {0: 0, 1: 1, 2: 2, 4: 4, 7: 8}
    actual = bounded_decoder_operator(circuit)
    assert actual.shape == (512, 512)
    # One comparison column keeps the combined dense-array allocation bounded.
    expected_column = np.zeros(512, dtype=complex)
    for column in range(512):
        clause, syndrome, scratch = column & 15, (column >> 4) & 7, column >> 7
        decoded = clause ^ literal_table.get(syndrome, 0)
        row = decoded | (syndrome << 4) | (scratch << 7)
        expected_column.fill(0)
        expected_column[row] = 1
        np.testing.assert_allclose(actual[:, column], expected_column, atol=2e-12, rtol=0)
        assert decoder_extension(syndrome, clause, scratch) == (syndrome, decoded, scratch)
    assert decoder_table() == literal_table
    assert dict(circuit.count_ops()) == {"x": 12, "h": 32, "cx": 96, "t": 64, "tdg": 48}
    assert len(circuit.data) == 252
    # Only vacuum/singles promise cleanup after B-transpose; other y remain explicit.
    cleaned = []
    for clause in range(16):
        high = (clause >> 3) & 1
        syndrome = ((clause & 1) ^ high) | ((((clause >> 1) & 1) ^ high) << 1)
        syndrome |= (((clause >> 2) & 1) ^ high) << 2
        if clause ^ literal_table.get(syndrome, 0) == 0:
            cleaned.append(clause)
    assert cleaned == [0, 1, 2, 4, 8]


def test_full_default_state_retains_negative_amplitudes_and_all_rational_probabilities():
    circuit = build_dqi_circuit()
    state = bounded_statevector(circuit).data
    expected = np.zeros(512, dtype=complex)
    polynomial = (3, 3, 3, -1, 3, -1, -1, -1)
    expected[np.arange(8) << 4] = np.asarray(polynomial) / math.sqrt(40)
    np.testing.assert_allclose(state, expected, atol=2e-12, rtol=0)
    probabilities = np.abs(state[np.arange(8) << 4]) ** 2
    np.testing.assert_allclose(probabilities, np.asarray([9, 9, 9, 1, 9, 1, 1, 1]) / 40,
                               atol=2e-12, rtol=0)
    assert np.vdot(state, state).real == pytest.approx(1, abs=2e-12)
    values = (3, 3, 3, 1, 3, 1, 1, 1)
    assert xor_objective_values() == values
    assert _direct_values(DEFAULT_MATRIX, DEFAULT_SIGNS) == values
    assert probabilities @ np.asarray(values) == pytest.approx(14 / 5, abs=2e-12)
    assert probabilities[[0, 1, 2, 4]].sum() == pytest.approx(9 / 10, abs=2e-12)
    reference = polynomial_reference()
    assert reference["objective_values"] == values
    assert reference["polynomial_values"] == polynomial
    assert reference["probability_fractions"] == ("9/40", "9/40", "9/40", "1/40",
                                                   "9/40", "1/40", "1/40", "1/40")
    assert reference["denominator"] == 40
    assert reference["expected_objective_fraction"] == "14/5"
    assert reference["optimal_draw_probability_fraction"] == "9/10"
    assert reference["optimum"] == 3
    assert "analytic_contract" in reference["status"]


def test_nonsymmetric_rows_and_signs_expose_transpose_and_integer_order_errors():
    matrix = ((1, 1, 0), (0, 1, 1), (1, 0, 0))
    signs = (1, 0, 1)
    circuit = build_dqi_circuit(matrix, signs)
    np.testing.assert_allclose(bounded_statevector(circuit).data,
                               _direct_final_state(matrix, signs), atol=2e-12, rtol=0)
    assert decoder_table(matrix) == {0: 0, 3: 1, 6: 2, 1: 4}
    assert primitive_operations(_stage_circuit(circuit, "syndrome_B_transpose_y")) == [
        ["cx", [0, 3], [], []], ["cx", [0, 4], [], []],
        ["cx", [1, 4], [], []], ["cx", [1, 5], [], []], ["cx", [2, 3], [], []],
    ]
    reference = polynomial_reference(matrix, signs)
    values = _direct_values(matrix, signs)
    weights = tuple(Fraction((1 + 2 * value - 3) ** 2, 32) for value in values)
    assert xor_objective_values(matrix, signs) == values
    assert reference["probability_fractions"] == tuple(str(weight) for weight in weights)
    assert sum(weights) == 1


def test_full_gate_ledger_includes_reset_readout_barriers_and_unsynthesized_rotations():
    circuit = build_dqi_circuit()
    counts = {"h": 47, "t": 88, "tdg": 66, "cx": 144, "x": 24,
              "ry": 7, "z": 1, "reset": 9, "measure": 3, "barrier": 7}
    assert dict(circuit.count_ops()) == counts
    assert len(circuit.data) == 396
    assert sum(value for name, value in counts.items() if name != "barrier") == 389
    assert sum(value for name, value in counts.items() if name in PRIMITIVE_GATES) == 377
    registers = {"clause": [0, 1, 2, 3], "assignment": [4, 5, 6],
                 "workspace": [7, 8], "classical": [0, 1, 2], "total_qubits": 9}
    assert register_map(validate_instance()) == circuit.metadata["registers"] == registers
    operations = primitive_operations(circuit)
    assert [entry[1] for entry in operations if entry[0] == "reset"] == [[q] for q in range(9)]
    assert [entry[1:3] for entry in operations if entry[0] == "measure"] == [
        [[4], [0]], [[5], [1]], [[6], [2]],
    ]
    assert [(stage["name"], stage["start"], stage["stop"]) for stage in circuit.metadata["stages"]] == [
        ("input_reset", 0, 9), ("weight_zero_one_preparation", 10, 125),
        ("clause_sign_phase", 126, 127), ("syndrome_B_transpose_y", 128, 134),
        ("reversible_decoder_and_cleanup", 135, 387),
        ("assignment_hadamards", 388, 391), ("assignment_readout", 392, 395),
    ]
    ledger = logical_resources(circuit, draws=2)
    assert ledger["primitive_events_per_draw"] == counts
    assert ledger["primitive_events_all_draws"] == {name: 2 * value for name, value in counts.items()}
    assert ledger["source_unitary_gates_per_draw"] == 377
    assert ledger["source_initializations_per_draw"] == 9
    assert ledger["source_individual_readouts_per_draw"] == 3
    assert ledger["explicit_t_tdg_per_draw"] == 154
    assert ledger["unsynthesized_ry_per_draw"] == 7
    assert not ledger["actual_rotation_synthesis_performed"]
    assert not ledger["native_physical_estimation_performed"]
    assert not circuit.metadata["arbitrary_ry_synthesized"]
    assert not circuit.metadata["upstream_code_adopted"]
    assert circuit.metadata["physical_estimate_status"] == "not implemented or executed"
    assert len(build_dqi_circuit(barriers=False).data) == 389
    unitary = build_dqi_circuit(measure=False, initialize=False, barriers=False)
    assert len(unitary.data) == 377
    assert unitary.num_clbits == 0
    np.testing.assert_allclose(bounded_statevector(unitary).data, _direct_final_state(DEFAULT_MATRIX, DEFAULT_SIGNS),
                               atol=2e-12, rtol=0)


@pytest.mark.parametrize("matrix,signs,ell", [
    ((), (), 1), (((1, 0), (1,)), (0, 0), 1), (((0, 0),), (0,), 1),
    (((1, 0), (1, 0)), (0, 1), 1), (((1, 0, 0, 0),), (0,), 1),
    (((1,),) * 5, (0,) * 5, 1), (((True, 0),), (0,), 1),
    (((1, 0.0),), (0,), 1), (((2, 0),), (0,), 1),
    (((1, 0),), (), 1), (((1, 0),), (False,), 1),
    (((1, 0),), (2,), 1), (((1, 0),), (0,), 0),
    (((1, 0),), (0,), 2), (((1, 0),), (0,), True),
])
def test_invalid_shapes_binary_entries_and_weight_one_promises_are_rejected(matrix, signs, ell):
    with pytest.raises(ValueError):
        validate_instance(matrix, signs, ell=ell)
    with pytest.raises(ValueError):
        build_dqi_circuit(matrix, signs, ell=ell)


@pytest.mark.parametrize("field,value", [
    ("materialized_qubits", 10), ("primitive_operations", 513),
    ("statevector_qubits", 10), ("statevector_amplitudes", 513),
    ("operator_dimension", 513), ("decoder_basis_checks", 513),
    ("primitive_operations", 0), ("operator_dimension", True),
])
def test_limit_contract_cannot_be_relaxed_or_set_to_noninteger_values(field, value):
    with pytest.raises(ValueError):
        DQILimits(**{field: value})


def test_materialization_and_simulation_guards_are_separate_and_enforced():
    circuit = build_dqi_circuit()
    assert len(build_dqi_circuit(limits=DQILimits(primitive_operations=396)).data) == 396
    with pytest.raises(ValueError, match="materialization"):
        build_dqi_circuit(limits=DQILimits(primitive_operations=395))
    with pytest.raises(ValueError, match="materialization"):
        build_dqi_circuit(limits=DQILimits(materialized_qubits=8))
    with pytest.raises(ValueError, match="simulation"):
        bounded_statevector(circuit, limits=DQILimits(statevector_qubits=8))
    with pytest.raises(ValueError, match="simulation"):
        bounded_statevector(circuit, limits=DQILimits(statevector_amplitudes=511))
    decoder = build_decoder_circuit()
    with pytest.raises(ValueError, match="separate guard"):
        bounded_decoder_operator(decoder, limits=DQILimits(operator_dimension=511))
    with pytest.raises(ValueError, match="separate guard"):
        bounded_decoder_operator(decoder, limits=DQILimits(decoder_basis_checks=511))
    with pytest.raises(ValueError, match="limits"):
        build_dqi_circuit(limits={})


def test_helpers_reject_opaque_or_noncanonical_reset_and_readout_workflows():
    opaque = QuantumCircuit(1)
    opaque.append(Gate("hidden_decoder", 1, []), [0])
    with pytest.raises(ValueError, match="opaque"):
        bounded_statevector(opaque)
    with pytest.raises(ValueError, match="opaque"):
        logical_resources(opaque)
    interior_reset = QuantumCircuit(1)
    interior_reset.h(0)
    interior_reset.reset(0)
    with pytest.raises(ValueError, match="input resets"):
        bounded_statevector(interior_reset)
    partial_reset = QuantumCircuit(2)
    partial_reset.reset(0)
    partial_reset.h(1)
    with pytest.raises(ValueError, match="every materialized wire"):
        bounded_statevector(partial_reset)
    wrong_order = QuantumCircuit(2)
    wrong_order.reset(1)
    wrong_order.reset(0)
    with pytest.raises(ValueError, match="Input reset order"):
        bounded_statevector(wrong_order)
    interior_readout = QuantumCircuit(1, 1)
    interior_readout.measure(0, 0)
    interior_readout.h(0)
    with pytest.raises(ValueError, match="final readout"):
        bounded_statevector(interior_readout)
    with pytest.raises(ValueError, match="separate guard"):
        bounded_decoder_operator(build_dqi_circuit())


@pytest.mark.parametrize("syndrome,clause,scratch", [
    (-1, 0, 0), (8, 0, 0), (0, 16, 0), (0, -1, 0),
    (0, 0, 4), (0, 0, -1), (True, 0, 0), (0, False, 0), (0, 0, True),
])
def test_decoder_classical_specification_rejects_out_of_register_values(syndrome, clause, scratch):
    with pytest.raises(ValueError, match="register bounds"):
        decoder_extension(syndrome, clause, scratch)


@pytest.mark.parametrize("m", [0, 5, True, 2.0])
def test_preparation_clause_bounds_are_explicit(m):
    with pytest.raises(ValueError, match="clause wires"):
        build_preparation_circuit(m)


@pytest.mark.parametrize("option,value", [("measure", 1), ("initialize", None), ("barriers", "yes")])
def test_build_options_require_booleans(option, value):
    with pytest.raises(ValueError, match="booleans"):
        build_dqi_circuit(**{option: value})


@pytest.mark.parametrize("draws", [0, -1, 1025, True, 1.0])
def test_repetition_ledger_has_an_explicit_bound(draws):
    circuit = build_dqi_circuit()
    with pytest.raises(ValueError, match="complete draws"):
        logical_resources(circuit, draws=draws)


def test_primitive_helpers_reject_duplicate_invalid_wires_and_nonfinite_angles():
    circuit = QuantumCircuit(5)
    for wires in ((0, 0, 1), (0, 1, 5), (True, 1, 2)):
        with pytest.raises(ValueError):
            append_exact_ccx(circuit, *wires)
    for angle in (True, math.inf, math.nan, "theta"):
        with pytest.raises(ValueError, match="finite real"):
            append_controlled_ry(circuit, 0, 1, angle)
    with pytest.raises(ValueError, match="exactly three"):
        append_dirty_c3x(circuit, [0, 1], 3, 4)
    with pytest.raises(ValueError, match="distinct"):
        append_dirty_c3x(circuit, [0, 1, 2], 3, 2)
