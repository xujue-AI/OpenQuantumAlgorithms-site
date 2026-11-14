"""Actual coupled damping circuits, independent GKSL and grouped readout checks."""
import math
import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace
from scipy.linalg import expm

from openquantum.simulation.two_spin_lindblad import (
    append_x_layer, append_zz, build_chain_circuit, build_two_spin_circuit,
    chain_schedule, compare_two_spin, energy_shot_budget, exact_channel,
    exact_density, expectation_values, group_scores, hamiltonian, initial_density,
    liouvillian, lower_to_rotation_basis, measured_system_probabilities,
    simulate_input_density, simulate_two_spin, split_channel_reference, split_schedule,
)


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)


def on_wire(matrix, wire, n):
    result = np.array([[1]], dtype=complex)
    for j in reversed(range(n)):
        result = np.kron(result, matrix if wire == j else I)
    return result


def test_exact_component_circuits_and_half_step_angles():
    ZZ = QuantumCircuit(2)
    append_zz(ZZ, 0.17, 0.6)
    np.testing.assert_allclose(Operator(ZZ).data, expm(-0.17j*0.6*np.kron(Z, Z)), atol=2e-13)
    XX = QuantumCircuit(2)
    append_x_layer(XX, 0.17, 0.35)
    np.testing.assert_allclose(Operator(XX).data, expm(-0.17j*0.35*(np.kron(I, X)+np.kron(X, I))), atol=2e-13)


def test_global_palindrome_and_total_component_times():
    sequence = split_schedule(2)
    assert sequence == tuple(reversed(sequence))
    for component in ("ZZ", "X", "D0", "D1"):
        assert sum(fraction for name, fraction in sequence if name == component) == 1
    assert chain_schedule(2, 2) == sequence


def test_liouvillian_column_vectorization_matches_direct_gksl_action():
    rng = np.random.default_rng(5)
    M = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    rho = M @ M.conj().T
    rho /= np.trace(rho)
    H = 0.6*np.kron(Z, Z)+0.35*(np.kron(I, X)+np.kron(X, I))
    jumps = [math.sqrt(0.8)*np.kron(I, LOWER), math.sqrt(0.5)*np.kron(LOWER, I)]
    action = -1j*(H @ rho-rho @ H)
    for L in jumps:
        action += L @ rho @ L.conj().T-0.5*(L.conj().T @ L @ rho+rho @ L.conj().T @ L)
    actual = (liouvillian() @ rho.reshape(16, order="F")).reshape((4, 4), order="F")
    np.testing.assert_allclose(actual, action, atol=2e-13)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("initial", ["product", "bell", "zero"])
def test_actual_channel_matches_the_classical_split_and_exact_limits(order, initial):
    rho0 = initial_density(initial)
    actual = simulate_two_spin(steps=4, order=order, initial=initial)
    split = (split_channel_reference(steps=4, order=order) @ rho0.reshape(16, order="F")).reshape((4, 4), order="F")
    np.testing.assert_allclose(actual, split, atol=2e-12)
    for J, hx, rates in [(0, 0, (0.8, 0.5)), (0.6, 0, (0, 0)), (0, 0.35, (0, 0)), (0, 0, (0, 0))]:
        state = simulate_two_spin(J=J, hx=hx, rates=rates, steps=3, order=order, initial=initial)
        reference = exact_density(rho0, J=J, hx=hx, rates=rates)
        np.testing.assert_allclose(state, reference, atol=2e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_positivity_trace_and_complex_inputs(order):
    for time in (0, 0.25, 1, 2):
        for initial in ("product", "bell", "zero"):
            rho = simulate_two_spin(time=time, steps=4, order=order, initial=initial)
            assert np.trace(rho).real == pytest.approx(1, abs=2e-12)
            assert abs(np.trace(rho).imag) < 2e-12
            assert np.linalg.eigvalsh(rho).min() > -2e-12
    rho0 = 0.6*initial_density("bell")+0.4*initial_density("product")
    actual = simulate_input_density(rho0, steps=3, order=order)
    split = (split_channel_reference(steps=3, order=order) @ rho0.reshape(16, order="F")).reshape((4, 4), order="F")
    np.testing.assert_allclose(actual, split, atol=2e-12)


def test_first_second_order_convergence_and_hamiltonian_diagnostic():
    first = [compare_two_spin(steps=r, order=1) for r in (8, 16, 32)]
    second = [compare_two_spin(steps=r, order=2) for r in (8, 16, 32)]
    assert first[0]["trace_distance"] > first[1]["trace_distance"] > first[2]["trace_distance"]
    assert second[0]["trace_distance"] > second[1]["trace_distance"] > second[2]["trace_distance"]
    assert 1.8 < first[1]["trace_distance"]/first[2]["trace_distance"] < 2.2
    assert 3.8 < second[1]["trace_distance"]/second[2]["trace_distance"] < 4.2
    assert second[-1]["trace_distance"] < 0.005
    assert second[-1]["hamiltonian_only_trace_distance_diagnostic"] > 0
    assert max(r["circuit_vs_classical_split_trace_distance"] for r in first+second) < 2e-12


@pytest.mark.parametrize("group", ["X", "Y", "Z"])
def test_measurement_basis_joint_scores_and_full_reset_export(group):
    source = build_two_spin_circuit(steps=4, order=2, observable=group)
    compiled = lower_to_rotation_basis(source)
    assert compiled.count_ops()["reset"] == 14  # two system + eight env0 + four env1.
    assert compiled.count_ops()["measure"] == 4
    probabilities = measured_system_probabilities(compiled)
    assert probabilities.sum() == pytest.approx(1, abs=2e-12)
    s0, s1 = np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1])
    actual = expectation_values(simulate_two_spin(steps=4, order=2))
    assert probabilities @ s0 == pytest.approx(actual[group+"0"], abs=2e-12)
    assert probabilities @ s1 == pytest.approx(actual[group+"1"], abs=2e-12)
    if group == "Z":
        assert probabilities @ (s0*s1) == pytest.approx(actual["ZZ"], abs=2e-12)
    qasm = qasm3.dumps(compiled)
    assert "gate " not in qasm and "reset" in qasm and "measure" in qasm


def test_energy_readout_is_sum_of_two_noncommuting_group_means():
    values = expectation_values(simulate_two_spin(steps=4, order=2))
    energy = 0
    for group in ("X", "Z"):
        circuit = lower_to_rotation_basis(build_two_spin_circuit(steps=4, order=2, observable=group))
        energy += measured_system_probabilities(circuit) @ group_scores(group)
    assert energy == pytest.approx(values["energy"], abs=2e-12)
    budget = energy_shot_budget()
    assert budget["groups"]["X"]["trials"] == 29116
    assert budget["groups"]["Z"]["trials"] == 21391
    assert budget["total_trials"] == 50507
    for allocation in budget["groups"].values():
        n, width, eps, delta = allocation["trials"], allocation["width"], allocation["statistical_error"], allocation["statistical_failure"]
        assert 2*math.exp(-2*n*eps**2/width**2) <= delta
    assert energy_shot_budget(J=0, hx=0)["total_trials"] == 0


@pytest.mark.parametrize("kwargs", [{"J": 1e308}, {"statistical_error": 1e-300},
                                    {"statistical_error": 5e-324}, {"statistical_failure": 5e-324}])
def test_energy_sampling_rejects_unrepresentable_or_unbounded_allocations(kwargs):
    with pytest.raises(ValueError):
        energy_shot_budget(**kwargs)


def test_fresh_environment_resets_are_required():
    source = build_two_spin_circuit(steps=4, order=2)
    altered = source.copy_empty_like()
    first_resets = set()
    for item in source.data:
        if item.operation.name == "reset":
            wire = source.find_bit(item.qubits[0]).index
            if wire >= 2:
                if wire in first_resets:
                    continue
                first_resets.add(wire)
        altered.append(item.operation, item.qubits, item.clbits)
    wrong = partial_trace(DensityMatrix.from_instruction(altered), [2, 3]).data
    assert np.linalg.norm(wrong-simulate_two_spin(steps=4, order=2)) > 0.01


def test_three_spin_generator_matches_an_independent_component_sequence():
    n, time, steps, J, hx, rates = 3, 0.3, 2, 0.6, 0.35, (0.8, 0.5, 0.3)
    source = build_chain_circuit(n, time, J, hx, rates, steps, 2)
    actual = partial_trace(DensityMatrix.from_instruction(source), [3, 4, 5]).data
    plus = np.array([1, 1], dtype=complex)/math.sqrt(2)
    plus_i = np.array([1, 1j], dtype=complex)/math.sqrt(2)
    vector = np.kron(np.kron(plus, plus), plus_i)
    reference = np.outer(vector, vector.conj())
    Hzz = J*sum(on_wire(Z, j, n) @ on_wire(Z, j+1, n) for j in range(n-1))
    Hx = hx*sum(on_wire(X, j, n) for j in range(n))
    sequence = [("ZZ", 0.5), ("X", 0.5), (0, 0.5), (1, 0.5), (2, 1), (1, 0.5), (0, 0.5), ("X", 0.5), ("ZZ", 0.5)]
    for _ in range(steps):
        for component, fraction in sequence:
            duration = time/steps*fraction
            if component in ("ZZ", "X"):
                U = expm(-1j*duration*(Hzz if component == "ZZ" else Hx))
                reference = U @ reference @ U.conj().T
            else:
                rate = rates[component]
                K0 = np.diag([1, math.exp(-rate*duration/2)]).astype(complex)
                K1 = math.sqrt(-math.expm1(-rate*duration))*LOWER
                A, B = on_wire(K0, component, n), on_wire(K1, component, n)
                reference = A @ reference @ A.conj().T+B @ reference @ B.conj().T
    np.testing.assert_allclose(actual, reference, atol=2e-12)


def test_generator_n_and_r_limits_separate_from_fixed_density_checks():
    assert build_chain_circuit(8, steps=4).num_qubits == 16
    assert build_two_spin_circuit(steps=129).num_qubits == 4
    with pytest.raises(ValueError, match="simulation"):
        simulate_two_spin(steps=129)
    with pytest.raises(ValueError, match="materialization"):
        build_chain_circuit(16, steps=1024)
    with pytest.raises(ValueError, match="materialization"):
        build_chain_circuit(16, steps=318, order=1, observable="X")
    with pytest.raises(ValueError):
        build_chain_circuit(3, initial="bell")
    with pytest.raises(ValueError, match="unknown"):
        build_two_spin_circuit(initial=None)


def test_one_spin_generator_preparation_and_dynamics_do_not_use_empty_registers():
    from openquantum.simulation.lindblad import simulate_lindblad
    source = build_chain_circuit(1, rates=(0.8,), steps=4, order=2)
    actual = partial_trace(DensityMatrix.from_instruction(source), [1]).data
    reference = simulate_lindblad(hx=0.35, hz=0, rate=0.8, steps=4, order=2)
    np.testing.assert_allclose(actual, reference, atol=2e-12)
    measured = build_chain_circuit(1, time=0, rates=(0.8,), steps=4, order=2, observable="Y")
    probabilities = DensityMatrix.from_instruction(measured.remove_final_measurements(inplace=False)).probabilities()
    assert probabilities[0::2].sum()-probabilities[1::2].sum() == pytest.approx(1, abs=2e-12)


def test_scaled_reference_corner_cases_reject_raw_overflow_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        np.testing.assert_allclose(exact_channel(1e-308, 1e308, 0, (0, 0)), exact_channel(1, 1, 0, (0, 0)), atol=2e-12)
        np.testing.assert_allclose(exact_channel(1e-308, 0, 0, (1e308, 0)), exact_channel(1, 0, 0, (1, 0)), atol=2e-12)
        with pytest.raises(ValueError, match="overflow"):
            liouvillian(J=1e308)
        with pytest.raises(ValueError, match="overflow"):
            liouvillian(rates=(1e308, 1e308))
    with pytest.raises(ValueError, match="exposure"):
        exact_channel(time=1000)


@pytest.mark.parametrize("kwargs", [{"steps": True}, {"steps": 0}, {"steps": 1025},
    {"steps": 10**9, "time": 0, "J": 0, "hx": 0, "rates": (0, 0)},
    {"order": 2.0}, {"time": -1}, {"J": float("nan")}, {"rates": (-1, 0.5)},
    {"rates": (0.8,)}, {"initial": "invalid"}, {"observable": "invalid"}])
def test_invalid_or_unbounded_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_two_spin_circuit(**kwargs)
