"""Independent ideal-channel, exact-source and accounting regressions."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, Statevector

import openquantum.shor_measured_and as measured


def ccx_permutation():
    """Classical truth table, independent of any quantum gate definition."""
    matrix = np.zeros((8, 8), dtype=complex)
    for value in range(8):
        output = value ^ 4 if (value & 1) and (value & 2) else value
        matrix[output, value] = 1
    return matrix


def arithmetic_destination(value, width):
    mask = (1 << width) - 1
    a, b, carry = value & mask, (value >> width) & mask, (value >> (2 * width)) & 1
    return a | (((a + b + carry) & mask) << width) | (carry << (2 * width))


def feedback_kraus(circuit, clear_mode):
    """Use the emitted premeasurement gates and actual true feedback body."""
    prefix = QuantumCircuit(4)
    feedback_item = None
    for item in circuit.data:
        if item.operation.name == "if_else":
            feedback_item = item
            break
        if item.operation.name == "measure":
            continue
        if item.operation.name == "reset":
            continue
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        prefix.append(item.operation, wires)
    assert feedback_item is not None
    unitary = Operator(prefix).data
    true_body = feedback_item.operation.blocks[0]
    mapping = tuple(circuit.find_bit(bit).index for bit in feedback_item.qubits)
    correction = QuantumCircuit(4)
    for item in true_body.data:
        wires = [mapping[true_body.find_bit(bit).index] for bit in item.qubits]
        correction.append(item.operation, wires)
    correction_matrix = Operator(correction).data
    final_rows = slice(0, 8) if clear_mode == "feedback_x" else slice(8, 16)
    # For literal reset, the row-shift |1>_helper -> |0>_helper adds no data phase.
    return unitary[:8, :8], correction_matrix[final_rows, 8:16] @ unitary[8:16, :8]


def coherent_gadget_input():
    vector = np.zeros(16, complex)
    vector[0] = vector[3] = 1 / math.sqrt(2)
    expected = np.zeros(16, complex)
    expected[0] = expected[7] = 1 / math.sqrt(2)
    return vector, expected


def mutant_without_cz(circuit, *, clear_helper=True):
    """Retain compute/copy/measurement but omit the measured phase correction."""
    result = QuantumCircuit(4, 1)
    for item in circuit.data:
        q = [circuit.find_bit(bit).index for bit in item.qubits]
        c = [circuit.find_bit(bit).index for bit in item.clbits]
        if item.operation.name == "if_else":
            with result.if_test((result.clbits[0], 1)):
                if clear_helper:
                    result.x(3)
                else:
                    result.h(1); result.cx(0, 1); result.h(1)
        else:
            result.append(item.operation, q, c)
    result.metadata = dict(circuit.metadata)
    return result


def test_clean_four_t_and_has_exact_clean_domain_relative_phases():
    circuit = measured.build_temporary_and_compute()
    matrix = Operator(circuit).data
    expected = np.zeros((8, 4), complex)
    for value in range(4):
        output = value | (4 if value == 3 else 0)
        expected[output, value] = 1
    np.testing.assert_allclose(matrix[:, :4], expected, atol=5e-14, rtol=0)
    assert circuit.count_ops().get("t", 0) + circuit.count_ops().get("tdg", 0) == 4
    assert "rccx" not in circuit.count_ops()
    assert "unitary" not in circuit.count_ops()
    # A clean-helper AND is deliberately not an exact CCX on dirty helpers.
    assert np.max(np.abs(matrix - ccx_permutation())) > 0.5


def test_literal_seven_t_baseline_is_full_ccx():
    circuit = QuantumCircuit(3)
    measured.append_seven_t_ccx(circuit, 0, 1, 2)
    np.testing.assert_allclose(Operator(circuit).data, ccx_permutation(), atol=5e-14, rtol=0)
    counts = circuit.count_ops()
    assert counts.get("t", 0) + counts.get("tdg", 0) == 7
    assert counts.get("h", 0) == 2
    assert counts.get("cx", 0) == 6


@pytest.mark.parametrize("clear_mode", ["feedback_x", "reset"])
def test_both_actual_feedback_kraus_blocks_equal_ccx(clear_mode):
    circuit = measured.build_measured_ccx(clear_mode=clear_mode)
    kraus = feedback_kraus(circuit, clear_mode)
    expected = ccx_permutation() / math.sqrt(2)
    for block in kraus:
        np.testing.assert_allclose(block, expected, atol=5e-14, rtol=0)
    np.testing.assert_allclose(sum(block.conj().T @ block for block in kraus), np.eye(8), atol=8e-14, rtol=0)


@pytest.mark.parametrize("clear_mode", ["feedback_x", "reset"])
@pytest.mark.parametrize("value", range(8))
def test_measured_ccx_all_data_basis_states_and_helper_clear(clear_mode, value):
    circuit = measured.build_measured_ccx(clear_mode=clear_mode)
    initial = np.zeros(16, complex); initial[value] = 1
    actual, report = measured.density_output(circuit, initial_state=initial)
    output = value ^ 4 if (value & 1) and (value & 2) else value
    expected = np.zeros((16, 16), complex); expected[output, output] = 1
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    assert report["maximum_classical_branches"] == 2
    assert abs(np.real(np.diag(actual))[8:].sum()) < 1e-12
    assert report["formal_proof"] is False


@pytest.mark.parametrize("clear_mode", ["feedback_x", "reset"])
def test_measured_ccx_preserves_external_entanglement(clear_mode):
    prepare = QuantumCircuit(5)
    prepare.h(4); prepare.cx(4, 0)
    prepare.h(1); prepare.cx(1, 2); prepare.t(4)
    prepared = Statevector.from_instruction(prepare).data
    circuit = QuantumCircuit(5, 1)
    circuit.compose(prepare, qubits=range(5), inplace=True)
    gadget = measured.build_measured_ccx(clear_mode=clear_mode)
    circuit.compose(gadget, qubits=[0, 1, 2, 3], clbits=[0], inplace=True)
    circuit.metadata = dict(gadget.metadata)
    actual, _ = measured.density_output(circuit)
    expected = np.zeros(32, complex)
    for value, amplitude in enumerate(prepared):
        output = value ^ 4 if (value & 1) and (value & 2) else value
        expected[output] = amplitude
    np.testing.assert_allclose(actual, np.outer(expected, expected.conj()), atol=2e-12, rtol=0)


def test_omitting_sdg_is_detected_on_coherence_even_when_basis_map_survives():
    circuit = measured.build_measured_ccx()
    circuit.data = [item for item in circuit.data if item.operation.name != "sdg"]
    initial, expected = coherent_gadget_input()
    actual, _ = measured.density_output(circuit, initial_state=initial)
    assert np.max(np.abs(actual - np.outer(expected, expected.conj()))) > 0.25


def test_omitting_conditional_cz_dephases_data_and_exact_merge_rejects_it():
    circuit = mutant_without_cz(measured.build_measured_ccx())
    initial, expected = coherent_gadget_input()
    actual, _ = measured.density_output(circuit, initial_state=initial)
    assert np.max(np.abs(actual - np.outer(expected, expected.conj()))) > 0.25
    prepared = QuantumCircuit(4, 1)
    prepared.h(0); prepared.cx(0, 1)
    prepared.compose(circuit, qubits=range(4), clbits=[0], inplace=True)
    prepared.metadata = dict(circuit.metadata)
    with pytest.raises(AssertionError, match="branches differ"):
        measured.exact_sparse_output(prepared)


def test_omitting_helper_clear_leaves_a_measured_dirty_helper():
    circuit = mutant_without_cz(measured.build_measured_ccx(), clear_helper=False)
    initial, _ = coherent_gadget_input()
    actual, _ = measured.density_output(circuit, initial_state=initial)
    assert np.real(np.diag(actual))[8:].sum() == pytest.approx(0.5, abs=1e-12)
    with pytest.raises(AssertionError, match="branches differ|dirty helper"):
        measured.exact_sparse_output(circuit)


@pytest.mark.parametrize("clear_mode,minimum", [("feedback_x", 13), ("reset", 14)])
@pytest.mark.parametrize("initialize", [False, True])
def test_gadget_resource_counts_include_branch_correction_and_initialization(clear_mode, minimum, initialize):
    circuit = measured.build_measured_ccx(clear_mode=clear_mode, initialize_helper=initialize)
    ledger = measured.logical_counts(circuit)
    assert ledger["literal_t_or_tdg"] == 4
    assert ledger["all_zero_feedback_primitive_count"] == minimum + initialize
    assert ledger["all_one_feedback_primitive_count"] == 17 + initialize
    assert ledger["classical_feedback_decisions"] == 1
    assert ledger["logical_qubits"] == 4
    assert ledger["classical_bits"] == 1
    extras = ledger["all_one_feedback_extra_counts"]
    assert extras.get("h", 0) == 2 and extras.get("cx", 0) == 1
    assert extras.get("x", 0) == (clear_mode == "feedback_x")
    assert ledger["unconditional_primitive_counts"].get("reset", 0) == initialize + (clear_mode == "reset")
    assert ledger["physical_estimation_status"] == "not computed"


@pytest.mark.parametrize("width", [1, 2, 3, 8, 31])
def test_ripple_materialization_matches_independent_count_formula(width):
    static = measured.logical_counts(measured.build_ripple_core(width, method="static"))
    candidate = measured.logical_counts(measured.build_ripple_core(width))
    assert static["literal_t_or_tdg"] == 14 * width
    assert static["all_one_feedback_primitive_count"] == 34 * width
    assert candidate["literal_t_or_tdg"] == 8 * width
    assert candidate["all_zero_feedback_primitive_count"] == 30 * width + 1
    assert candidate["all_one_feedback_primitive_count"] == 38 * width + 1
    assert candidate["classical_feedback_decisions"] == 2 * width
    assert candidate["logical_qubits"] == 2 * width + 2
    assert candidate["unlowered_phase_operations"] == 0
    assert "4w-4" in measured.build_ripple_core(width).metadata["algorithm_variant"]


@pytest.mark.parametrize("width,a,b,carry", [(1, 1, 1, 1), (2, 3, 1, 0), (3, 7, 6, 1), (8, 255, 129, 1)])
def test_complete_counts_include_data_preparation_and_all_readout(width, a, b, carry):
    weight = a.bit_count() + b.bit_count() + carry
    baseline = measured.logical_counts(measured.build_addition_demo(width, a, b, carry, method="static"))
    candidate = measured.logical_counts(measured.build_addition_demo(width, a, b, carry))
    formula = measured.addition_scaling(width, a, b, carry)
    assert baseline["all_one_feedback_primitive_count"] == 38 * width + 2 + weight
    assert candidate["all_zero_feedback_primitive_count"] == 34 * width + 3 + weight
    assert candidate["all_one_feedback_primitive_count"] == 42 * width + 3 + weight
    assert baseline["unconditional_primitive_counts"]["measure"] == 2 * width + 1
    assert candidate["unconditional_primitive_counts"]["measure"] == 4 * width + 1
    assert candidate["unconditional_primitive_counts"]["reset"] == 2 * width + 2
    assert formula["static"]["complete_primitives"] == baseline["all_one_feedback_primitive_count"]
    assert formula["measured_feedback_x"]["complete_primitives_all_one"] == candidate["all_one_feedback_primitive_count"]
    assert formula["accuracy_inherited"] is False and formula["materialization_performed"] is False


def test_scalar_counts_do_not_dispatch_circuits(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("count-only API dispatched a circuit")
    monkeypatch.setattr(measured, "ripple_source", forbidden)
    monkeypatch.setattr(measured, "build_ripple_core", forbidden)
    report = measured.addition_scaling(100_000)
    assert report["static"]["literal_t_or_tdg"] == 1_400_000
    assert report["measured_feedback_x"]["literal_t_or_tdg"] == 800_000


SMALL_ADDITIONS = [(1, a, b, carry) for a, b, carry in itertools.product(range(2), range(2), range(2))]
SMALL_ADDITIONS += [(2, 0, 3, 1), (2, 3, 0, 1), (2, 3, 3, 1), (2, 1, 2, 0), (2, 2, 1, 0)]
SMALL_ADDITIONS += [(3, 0, 7, 1), (3, 7, 7, 1), (3, 5, 2, 0)]


@pytest.mark.parametrize("method", ["static", "measured"])
@pytest.mark.parametrize("width,a,b,carry", SMALL_ADDITIONS)
def test_actual_small_additions_preserve_a_and_carry(width, a, b, carry, method):
    circuit = measured.build_addition_demo(width, a, b, carry, method=method, readout=False)
    actual, _ = measured.density_output(circuit)
    initial = a | (b << width) | (carry << (2 * width))
    expected_value = arithmetic_destination(initial, width)
    expected = np.zeros_like(actual); expected[expected_value, expected_value] = 1
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    exact, details = measured.exact_sparse_output(circuit)
    assert exact == {expected_value: measured.Cyclotomic16.one()}
    assert details["amplitude_truncation"] is False and details["norm_exactly_one"] is True


@pytest.mark.parametrize("method,clear_mode", [("static", "feedback_x"), ("measured", "feedback_x"), ("measured", "reset")])
def test_arithmetic_coherence_matches_independent_integer_permutation(method, clear_mode):
    width, data_qubits = 2, 5
    prepare = QuantumCircuit(data_qubits)
    prepare.h(range(data_qubits))
    prepare.h(1); prepare.cx(0, 1); prepare.h(1)
    prepare.t(2); prepare.s(4)
    initial = Statevector.from_instruction(prepare).data
    core = measured.build_ripple_core(width, method=method, clear_mode=clear_mode)
    circuit = QuantumCircuit(core.num_qubits, core.num_clbits)
    circuit.compose(prepare, qubits=range(data_qubits), inplace=True)
    circuit.compose(core, qubits=range(core.num_qubits), clbits=range(core.num_clbits), inplace=True)
    circuit.metadata = dict(core.metadata)
    expected = np.zeros(1 << core.num_qubits, complex)
    for value, amplitude in enumerate(initial):
        expected[arithmetic_destination(value, width)] = amplitude
    density, _ = measured.density_output(circuit)
    np.testing.assert_allclose(density, np.outer(expected, expected.conj()), atol=3e-12, rtol=0)
    amplitudes, _ = measured.exact_sparse_output(circuit)
    actual = np.zeros_like(expected)
    for value, amplitude in amplitudes.items():
        actual[value] = amplitude.as_complex()
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


def test_terminal_readout_retains_feedback_and_has_correct_data_mapping():
    circuit = measured.build_addition_demo(2, 3, 1)
    _, report = measured.density_output(circuit)
    expected = arithmetic_destination(3 | (1 << 2), 2) << 1
    probabilities = report["terminal_readout_distribution"]
    assert probabilities[expected] == pytest.approx(0.5, abs=2e-12)
    assert probabilities[expected | 1] == pytest.approx(0.5, abs=2e-12)
    assert probabilities[expected] + probabilities[expected | 1] == pytest.approx(1, abs=2e-12)
    assert sum(probabilities.values()) == pytest.approx(1, abs=2e-12)
    assert circuit.metadata["readout_clbits"] == [1, 2, 3, 4, 5]


def test_terminal_measurement_overwrites_an_old_classical_value():
    circuit = QuantumCircuit(1, 1)
    circuit.x(0); circuit.measure(0, 0)
    circuit.x(0); circuit.measure(0, 0)
    _, report = measured.density_output(circuit)
    assert report["terminal_readout_distribution"][0] == pytest.approx(1, abs=1e-12)
    assert report["terminal_readout_distribution"].get(1, 0) == pytest.approx(0, abs=1e-12)


def test_exact_ring_relations_conjugation_and_normalization():
    one = measured.Cyclotomic16.one()
    zeta = one.shifted(1)
    assert one.shifted(8) == -one
    assert one.shifted(16) == one
    assert zeta * zeta.conjugate() == one
    assert one.inv_sqrt2().sqrt2() == one
    assert one.inv_sqrt2() * one.inv_sqrt2() == measured.Cyclotomic16((1, 0, 0, 0, 0, 0, 0, 0), 1)
    assert measured.Cyclotomic16((2, 0, 0, 0, 0, 0, 0, 0), 1) == one
    assert measured.Cyclotomic16((0,) * 8, 7) == measured.Cyclotomic16.zero()
    assert one.shifted(-2) == one.shifted(14)


@pytest.mark.parametrize("method", ["static", "measured"])
def test_source_global_phase_is_preserved_without_alignment(method):
    source = QuantumCircuit(1); source.h(0)
    source.global_phase = math.pi / 8
    transformed = measured.replace_ccx(source, method=method)
    actual, _ = measured.exact_sparse_output(transformed)
    half = measured.Cyclotomic16.one().inv_sqrt2().shifted(1)
    assert actual == {0: half, 1: half}
    assert transformed.global_phase == source.global_phase
    assert measured.logical_counts(transformed)["unlowered_phase_operations"] == 0


def test_exact_pi_over_eight_phase_and_reject_nonfield_phase():
    circuit = QuantumCircuit(1); circuit.h(0); circuit.p(math.pi / 8, 0)
    amplitudes, _ = measured.exact_sparse_output(circuit)
    half = measured.Cyclotomic16.one().inv_sqrt2()
    assert amplitudes == {0: half, 1: half.shifted(1)}
    unsupported = QuantumCircuit(1); unsupported.p(math.pi / 16, 0)
    with pytest.raises(ValueError, match="pi/8"):
        measured.exact_sparse_output(unsupported)


@pytest.mark.parametrize("N,base,phase_bits", [(15, 2, 4), (21, 2, 4)])
def test_full_measured_order_source_matches_independent_fourier_reference(N, base, phase_bits):
    report = measured.check_order_finding(N, base, phase_bits)
    np.testing.assert_allclose(report["probabilities"], report["reference_probabilities"], atol=1e-10, rtol=0)
    assert report["dirty_work_probability"] == 0
    assert report["exact_sparse"]["merged_uniform_feedback_pairs"] > 0
    assert report["exact_sparse"]["amplitude_truncation"] is False
    assert report["physical_estimation_status"].startswith("not computed")


@pytest.mark.parametrize("bad", [0, 32, True, 1.5, "2"])
def test_bad_materialized_width_rejected(bad):
    with pytest.raises(ValueError, match="width"):
        measured.build_ripple_core(bad)


@pytest.mark.parametrize("kwargs", [{"a": -1}, {"a": 4}, {"b": 4}, {"carry": 2}, {"carry": True}, {"readout": 1}])
def test_bad_addition_input_contract_rejected(kwargs):
    arguments = dict(width=2, a=0, b=0, carry=0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        measured.build_addition_demo(**arguments)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_order_readout_requires_actual_boolean(bad):
    with pytest.raises(ValueError, match="readout"):
        measured.build_order_finding_measured(3, 2, 2, readout=bad)


def test_reject_bad_clear_mode_and_wire_collisions():
    with pytest.raises(ValueError, match="clearing"):
        measured.build_measured_ccx(clear_mode="discard")
    with pytest.raises(ValueError, match="Boolean"):
        measured.build_measured_ccx(initialize_helper=1)
    with pytest.raises(ValueError, match="distinct"):
        measured.append_temporary_and(QuantumCircuit(3), 0, 0, 2)
    with pytest.raises(ValueError, match="distinct"):
        measured.append_seven_t_ccx(QuantumCircuit(3), 0, 1, 1)


def test_reject_unbound_opaque_or_dynamic_source():
    unbound = QuantumCircuit(1); unbound.p(Parameter("angle"), 0)
    with pytest.raises(ValueError, match="bound"):
        measured.replace_ccx(unbound)
    opaque = QuantumCircuit(1); opaque.append(Gate("x", 1, []), [0])
    with pytest.raises(ValueError, match="unsupported"):
        measured.replace_ccx(opaque)
    matrix = QuantumCircuit(1); matrix.unitary(np.eye(2), [0])
    with pytest.raises(ValueError, match="opaque"):
        measured.replace_ccx(matrix)
    dynamic = QuantumCircuit(1, 1); dynamic.measure(0, 0)
    with pytest.raises(ValueError, match="unitary"):
        measured.replace_ccx(dynamic)


@pytest.mark.parametrize("name,n_clbits", [("reset", 0), ("measure", 1), ("barrier", 0)])
def test_instrument_rejects_name_only_nonphysical_instructions(name, n_clbits):
    circuit = QuantumCircuit(1, n_clbits)
    circuit.append(Instruction(name, 1, n_clbits, []), [0], list(range(n_clbits)))
    with pytest.raises(ValueError, match="nonliteral|physical|literal"):
        measured.primitive_events(circuit)


def test_replacement_preflight_and_dense_caps_precede_state_allocation(monkeypatch):
    monkeypatch.setattr(measured, "MAX_SOURCE_EVENTS", 20)
    with pytest.raises(ValueError, match="preflight"):
        measured.replace_ccx(measured.ripple_source(1))
    monkeypatch.setattr(measured, "MAX_SOURCE_EVENTS", 500_000)
    large = measured.build_ripple_core(8)
    with pytest.raises(ValueError, match="ten qubits"):
        measured.density_output(large)
    circuit = QuantumCircuit(2); circuit.h(0)
    monkeypatch.setattr(measured, "MAX_DENSE_EVENT_MATRIX_WORK", 8)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("density allocation preceded work preflight")
    monkeypatch.setattr(measured, "DensityMatrix", forbidden)
    with pytest.raises(ValueError, match="preflight"):
        measured.density_output(circuit)


def test_sparse_support_work_and_nonpure_reset_guards(monkeypatch):
    monkeypatch.setattr(measured, "MAX_EXACT_STATES", 4)
    circuit = QuantumCircuit(3); circuit.h(range(3))
    with pytest.raises(ValueError, match="support"):
        measured.exact_sparse_output(circuit)
    monkeypatch.setattr(measured, "MAX_EXACT_STATES", 8192)
    monkeypatch.setattr(measured, "MAX_EXACT_GATE_STATE_WORK", 1)
    with pytest.raises(ValueError, match="work cap"):
        measured.exact_sparse_output(circuit)
    monkeypatch.setattr(measured, "MAX_EXACT_GATE_STATE_WORK", 20_000_000)
    mixed_reset = QuantumCircuit(1); mixed_reset.h(0); mixed_reset.reset(0)
    with pytest.raises(ValueError, match="known basis"):
        measured.exact_sparse_output(mixed_reset)


def test_reject_invalid_exact_ring_encodings():
    with pytest.raises(ValueError):
        measured.Cyclotomic16((1,) * 7)
    with pytest.raises(ValueError):
        measured.Cyclotomic16((True,) + (0,) * 7)
    with pytest.raises(ValueError):
        measured.Cyclotomic16((1 << 4096,) + (0,) * 7)
    with pytest.raises(ValueError):
        measured.Cyclotomic16((1,) + (0,) * 7, 4097)
    with pytest.raises(ValueError):
        measured.Cyclotomic16((1,) + (0,) * 7, True)


def test_exact_value_does_not_borrow_mutable_coefficients():
    source = [1] + [0] * 7
    try:
        value = measured.Cyclotomic16(source)
    except ValueError:
        # Strict tuple-only input and a defensive tuple copy are both sound.
        return
    source[0] = 2
    assert isinstance(value.coefficients, tuple)
    assert value == measured.Cyclotomic16.one()


def test_root_exponent_bound_and_types_precede_field_shift():
    one = measured.Cyclotomic16.one()
    for exponent in (True, 0.5, (1 << 20) + 1, -(1 << 20) - 1):
        with pytest.raises(ValueError, match="root exponent"):
            one.shifted(exponent)


def test_initial_density_vector_shape_rejected_before_array_conversion(monkeypatch):
    circuit = QuantumCircuit(2)
    nested = [[1, 0, 0, 0]] * 4
    matrix = np.zeros((4, 2))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid vector reached array conversion")
    monkeypatch.setattr(measured.np, "asarray", forbidden)
    for initial in (nested, matrix):
        with pytest.raises(ValueError, match="vector"):
            measured.density_output(circuit, initial_state=initial)


def test_large_generator_and_small_simulators_have_separate_contracts():
    circuit = measured.build_ripple_core(31)
    assert circuit.num_qubits == 64
    assert measured.logical_counts(circuit)["literal_t_or_tdg"] == 248
    with pytest.raises(ValueError, match="width"):
        measured.check_addition(4, 1, 2)
    with pytest.raises(ValueError, match="phase bits"):
        measured.check_order_finding(15, 2, 5)
    with pytest.raises(ValueError, match="modulus"):
        measured.check_order_finding(35, 2, 4)
    with pytest.raises(QiskitError):
        measured.build_measured_ccx().to_gate()
