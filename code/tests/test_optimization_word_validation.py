"""Only pure NumPy checks; this file imports no quantum/native backend."""
import math

import numpy as np
import pytest

from openquantum import optimization_word_validation as validation


def instrument(gates, n=4, mapping=None):
    mapping = list(range(n)) if mapping is None else mapping
    return ([["reset", [q], [], []] for q in range(n)] +
            [[name, list(wires), [], []] for name, wires in gates] +
            [["measure", [q], [mapping[q]], []] for q in range(n)])


def swap(a, b):
    return [("cx", (a, b)), ("cx", (b, a)), ("cx", (a, b))]


def shuttle(word, wire):
    before, after = [], []
    for q in range(wire, 0, -1):
        before.extend(swap(q, q - 1))
    for q in range(1, wire + 1):
        after.extend(swap(q - 1, q))
    return before + [(name, (0,)) for name in word] + after


@pytest.mark.parametrize("wire", [1, 2, 3])
@pytest.mark.parametrize("word", [
    ("t", "h", "tdg", "s", "t"),
    ("tdg", "h", "t", "sdg", "tdg"),
])
def test_shuttle_matches_arbitrary_input_operator_and_permuted_final_destinations(wire, word):
    prefix = [("h", (0,)), ("cx", (0, 1)), ("cx", (1, 2)), ("s", (2,))]
    suffix = [("cx", (2, 3)), ("h", (1,))]
    original = instrument(prefix + [(name, (wire,)) for name in word] + suffix,
                          mapping=[2, 0, 3, 1])
    candidate = instrument(prefix + shuttle(word, wire) + suffix, mapping=[2, 0, 3, 1])
    report = validation.validate_candidate_pair(
        original, candidate, source_global_phase=.125, candidate_global_phase=.125)
    assert report["operator_spectral_error"] < 1e-14
    assert report["joint_z_maximum_probability_error"] < 1e-14
    assert report["measurement_map"] == [[0, 2], [1, 0], [2, 3], [3, 1]]
    assert report["global_phase_fitting"] is False


def test_operator_has_explicit_little_endian_reference_and_phase():
    operations = instrument([("x", (0,)), ("cx", (0, 1))], n=2)
    actual = validation.coherent_operator(operations, num_qubits=2, global_phase=math.pi / 3)
    expected = np.zeros((4, 4), complex)
    for column, row in enumerate((3, 0, 1, 2)):
        expected[row, column] = np.exp(1j * math.pi / 3)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_missing_reverse_shuttle_is_rejected_even_when_initial_data_are_zero():
    original = instrument([("t", (3,)), ("t", (3,)), ("t", (3,))])
    candidate = instrument(shuttle(("t", "t", "t"), 3)[:-9])
    with pytest.raises(ValueError, match="operator"):
        validation.validate_candidate_pair(original, candidate)


def test_entangled_spectator_gate_cannot_be_crossed_without_restoring_locations():
    original = instrument([("h", (0,)), ("cx", (0, 1)), ("t", (1,)), ("h", (0,)), ("t", (1,))])
    candidate = instrument([("h", (0,)), ("cx", (0, 1))] +
                           swap(1, 0) + [("t", (0,)), ("h", (0,)), ("t", (0,))] + swap(0, 1))
    with pytest.raises(ValueError, match="operator"):
        validation.validate_candidate_pair(original, candidate)


def test_exact_global_phase_is_retained_without_fitting():
    operations = instrument([("h", (0,))])
    with pytest.raises(ValueError, match="global-phase"):
        validation.validate_candidate_pair(operations, operations,
                                           source_global_phase=.1, candidate_global_phase=.2)
    with pytest.raises(ValueError, match="global-phase"):
        validation.validate_candidate_pair(operations, operations,
                                           source_global_phase=0.0, candidate_global_phase=-0.0)


def test_measurement_destinations_and_reset_events_cannot_change():
    original = instrument([("h", (0,))])
    candidate = instrument([("h", (0,))], mapping=[1, 0, 2, 3])
    with pytest.raises(ValueError, match="destinations"):
        validation.validate_candidate_pair(original, candidate)
    candidate = [original[1], original[0], *original[2:]]
    with pytest.raises(ValueError, match="resets"):
        validation.validate_candidate_pair(original, candidate)


def test_operator_check_keeps_barrier_position_in_its_separate_structural_scope():
    original = instrument([("h", (0,)), ("t", (0,)), ("t", (0,)), ("t", (0,))])
    candidate = list(original)
    original.insert(5, ["barrier", [0, 1, 2, 3], [], []])
    candidate.insert(4, ["barrier", [0, 1, 2, 3], [], []])
    report = validation.validate_candidate_pair(original, candidate)
    assert report["event_subsequence_preserved"] is True
    assert report["requires_independent_structural_word_boundary_audit"] is True
    assert "events_preserved" not in report


def test_joint_probability_mapper_retains_correlations_and_little_endian_bits():
    vector = np.zeros(16, complex)
    vector[3] = np.sqrt(.25)
    vector[12] = 1j * np.sqrt(.75)
    density = np.outer(vector, vector.conjugate())
    actual = validation.joint_z_probabilities(density, [[0, 2], [1, 0], [2, 3], [3, 1]])
    expected = np.zeros(16)
    expected[5], expected[10] = .25, .75
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("inverse", [False, True])
def test_corrected_injection_kraus_is_complete_and_has_both_expected_branches(inverse):
    sign = -1 if inverse else 1
    target = np.diag([1, np.exp(sign * 1j * np.pi / 4)])
    zero, one = validation.corrected_injection_kraus(inverse)
    np.testing.assert_allclose(zero, target / np.sqrt(2), rtol=0, atol=1e-15)
    np.testing.assert_allclose(one, np.exp(sign * 1j * np.pi / 4) * target / np.sqrt(2), rtol=0, atol=1e-15)
    np.testing.assert_allclose(zero.conjugate().T @ zero + one.conjugate().T @ one,
                               np.eye(2), rtol=0, atol=1e-15)


def test_density_suite_checks_choi_entangled_data_fresh_supply_reset_and_transport():
    report = validation.validate_density_mechanisms()
    assert report["case_count"] == 20 <= validation.MAX_DENSITY_CASES
    assert report["maximum_qubits"] == 6
    assert report["maximum_density_dimension"] == 64
    assert report["maximum_micro_operations"] == 50 <= validation.MAX_DENSITY_MICRO_OPERATIONS
    assert report["maximum_spectral_error"] < 1e-14
    assert report["maximum_branch_probability_error"] < 1e-14
    assert report["optimizer_sampling_synthesis_native_calls"] == 0
    names = {row["case_id"] for row in report["cases"]}
    assert {"injection-choi-t", "injection-choi-tdg", "initialized-port-route-3",
            "data-shuttle-3-mixed-entangled-signed-inverse"} <= names
    assert "no physical-code" in report["coverage"]


def test_uncorrected_injection_is_not_the_t_channel_on_an_entangled_input():
    initial = validation._bell_density(4, 0, 1)
    rho = validation._local_density(initial, validation._GATES["h"], 3)
    rho = validation._local_density(rho, validation._GATES["t"], 3)
    rho = validation._swap_density(rho, 3, 2)
    rho = validation._cx_density(rho, 0, 2)
    wrong = sum((validation._reset_density(validation._local_density(rho, projector, 2), 2)
                 for projector in validation._PROJECTORS), np.zeros_like(rho))
    expected = validation._local_density(initial, validation._GATES["t"], 0)
    assert np.linalg.norm(wrong - expected, 2) > .25


def test_injection_without_port_reset_fails_the_clean_ancilla_channel_invariant():
    initial = validation._bell_density(4, 0, 1)
    rho = validation._local_density(initial, validation._GATES["h"], 3)
    rho = validation._local_density(rho, validation._GATES["t"], 3)
    rho = validation._swap_density(rho, 3, 2)
    rho = validation._cx_density(rho, 0, 2)
    branches = [validation._local_density(rho, projector, 2) for projector in validation._PROJECTORS]
    branches[1] = validation._local_density(branches[1], validation._GATES["s"], 0)
    wrong = branches[0] + branches[1]
    expected = validation._local_density(initial, validation._GATES["t"], 0)
    assert np.linalg.norm(wrong - expected, 2) > .25


@pytest.mark.parametrize("mutation", ["new_reset", "after_readout", "parameter", "nonlocal_cx", "duplicate_bit"])
def test_malformed_saved_instrument_is_rejected(mutation):
    operations = instrument([("h", (0,))])
    if mutation == "new_reset":
        operations.insert(5, ["reset", [0], [], []])
    elif mutation == "after_readout":
        operations.append(["t", [1], [], []])
    elif mutation == "parameter":
        operations[4][3] = [.01]
    elif mutation == "nonlocal_cx":
        operations[4] = ["cx", [0, 3], [], []]
    else:
        operations[-1][2] = [0]
    with pytest.raises(ValueError):
        validation.coherent_operator(operations)


def test_operation_width_density_and_micro_operation_caps_fail_closed():
    operations = instrument([("h", (0,))])
    with pytest.raises(ValueError, match="10,000"):
        validation.coherent_operator(operations * 2_000)
    with pytest.raises(ValueError, match="Data width"):
        validation.coherent_operator(operations, num_qubits=5)
    with pytest.raises(ValueError, match="Density arrays"):
        validation._density(np.eye(128))
    with pytest.raises(ValueError, match="Maximum density width"):
        validation._density(np.eye(128), maximum_qubits=7)
    with pytest.raises(ValueError, match="micro-operation"):
        validation._case("too-long", np.eye(2) / 2, np.eye(2) / 2, 65)


def test_only_parameter_free_saved_primitives_are_accepted():
    with pytest.raises(ValueError, match="Unsupported"):
        validation.coherent_operator(instrument([("rx", (0,))]))
    with pytest.raises(ValueError, match="finite"):
        validation.coherent_operator(instrument([("h", (0,))]), global_phase=float("nan"))


def test_joint_readout_rejects_unnormalized_density():
    with pytest.raises(ValueError, match="unit trace"):
        validation.joint_z_probabilities(np.eye(16), [[q, q] for q in range(4)])


def test_local_density_tensor_contraction_matches_independent_embedded_matrix():
    rho = validation._bell_density(3, 0, 2, 1j)
    gate = np.array([[1, 1j], [1j, 1]], complex) / np.sqrt(2)
    for wire in range(3):
        embedded = validation._embedded_local(gate, 3, wire)
        expected = embedded @ rho @ embedded.conjugate().T
        np.testing.assert_allclose(validation._local_density(rho, gate, wire), expected, rtol=0, atol=1e-15)
