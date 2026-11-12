"""Independent Liouvillian checks of actual positive-time gate/reset schedules."""
import math
import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import DensityMatrix, Operator
from scipy.linalg import expm

from openquantum.simulation.open_systems import INPUT_LABELS, analytic_finite_time_damping, input_density
from openquantum.simulation.lindblad import (
    PAULI_MATRICES, append_exact_hamiltonian, build_lindblad_circuit, compare_lindblad,
    exact_channel, exact_density, liouvillian, lower_to_rotation_basis,
    simulate_input_density, simulate_lindblad,
)


@pytest.mark.parametrize("hx,hz", [(0.7, 0.3), (-0.7, 0.3), (0.7, -0.3), (0, -0.3), (0, 0)])
def test_analytic_hamiltonian_circuit_matches_independent_exponential(hx, hz):
    circuit = QuantumCircuit(1)
    append_exact_hamiltonian(circuit, 0, 0.73, hx, hz)
    H = np.array([[hz, hx], [hx, -hz]], dtype=complex)
    np.testing.assert_allclose(Operator(circuit).data, expm(-0.73j * H), atol=2e-13)


def test_column_vectorization_matches_direct_gksl_action():
    H = np.array([[0.3, 0.7], [0.7, -0.3]], dtype=complex)
    L = math.sqrt(0.8) * np.array([[0, 1], [0, 0]], dtype=complex)
    rho = np.array([[0.6, 0.1-0.2j], [0.1+0.2j, 0.4]], dtype=complex)
    action = -1j*(H @ rho-rho @ H)+L @ rho @ L.conj().T-0.5*(L.conj().T @ L @ rho+rho @ L.conj().T @ L)
    actual = (liouvillian() @ rho.reshape(4, order="F")).reshape((2, 2), order="F")
    np.testing.assert_allclose(actual, action, atol=1e-14)


def test_reference_scales_before_assembly_for_large_coefficients_short_time():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        damping = exact_channel(time=1e-308, rate=1e308, hx=0, hz=0)
        hamiltonian = exact_channel(time=1e-308, rate=0, hx=0, hz=1e308)
        np.testing.assert_allclose(damping, exact_channel(1, 1, 0, 0), atol=2e-13)
        np.testing.assert_allclose(hamiltonian, exact_channel(1, 0, 0, 1), atol=2e-13)
        with pytest.raises(ValueError, match="overflow"):
            liouvillian(rate=0, hx=0, hz=1e308)
        assert np.all(np.isfinite(liouvillian(rate=1e308, hx=0, hz=0)))


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("label", INPUT_LABELS)
def test_commuting_damping_z_hamiltonian_and_zero_components_are_exact(order, label):
    rho = input_density(label)
    for rate, hx, hz in [(0.8, 0, 0), (0.8, 0, 0.3), (0, 0.7, 0.3), (0, 0, 0)]:
        actual = simulate_lindblad(1, rate, hx, hz, 3, order, label)
        reference = exact_density(rho, 1, rate, hx, hz)
        np.testing.assert_allclose(actual, reference, atol=2e-13)
    np.testing.assert_allclose(exact_density(rho, 1, 0.8, 0, 0), analytic_finite_time_damping(rho, 0.8, 1), atol=2e-13)


def test_noncommuting_lie_and_strang_convergence_in_state_and_channel():
    first = [compare_lindblad(steps=r, order=1) for r in (8, 16, 32)]
    second = [compare_lindblad(steps=r, order=2) for r in (8, 16, 32)]
    for key in ("trace_distance", "superoperator_spectral_error"):
        assert first[0][key] > first[1][key] > first[2][key]
        assert second[0][key] > second[1][key] > second[2][key]
        assert all(a[key] > b[key] for a, b in zip(first, second))
        assert 1.8 < first[1][key]/first[2][key] < 2.2
        assert 3.8 < second[1][key]/second[2][key] < 4.2
    assert second[-1]["trace_distance"] < 0.005


@pytest.mark.parametrize("order", [1, 2])
def test_reduced_states_are_trace_one_positive_for_complex_and_mixed_inputs(order):
    for time in (0, 0.25, 1, 2):
        for label in INPUT_LABELS:
            rho = simulate_lindblad(time=time, steps=4, order=order, input_label=label)
            assert np.trace(rho).real == pytest.approx(1, abs=2e-13)
            assert abs(np.trace(rho).imag) < 2e-13
            assert np.linalg.eigvalsh(rho).min() > -2e-13
    mixed = np.array([[0.6, 0.1-0.2j], [0.1+0.2j, 0.4]], dtype=complex)
    actual = simulate_input_density(mixed, steps=32, order=order)
    assert np.trace(actual).real == pytest.approx(1, abs=2e-13)
    assert np.linalg.eigvalsh(actual).min() > 0


@pytest.mark.parametrize("observable", ["X", "Y", "Z"])
def test_export_includes_all_resets_readout_and_correct_measurement_basis(observable):
    source = build_lindblad_circuit(steps=4, order=2, observable=observable)
    compiled = lower_to_rotation_basis(source)
    assert source.count_ops()["reset"] == 5  # One system input reset plus four fresh environments.
    assert compiled.count_ops()["reset"] == 5
    assert compiled.count_ops()["measure"] == 2
    probabilities = DensityMatrix.from_instruction(compiled.remove_final_measurements(inplace=False)).probabilities()
    mean = sum((1 if i % 2 == 0 else -1)*p for i, p in enumerate(probabilities))
    rho = simulate_lindblad(steps=4, order=2)
    assert mean == pytest.approx(np.trace(rho @ PAULI_MATRICES[observable]).real, abs=2e-13)
    exported = qasm3.dumps(compiled)
    assert "gate " not in exported and "reset" in exported and "measure" in exported


def test_environment_must_be_reset_for_each_damping_call():
    source = build_lindblad_circuit(steps=8, order=2)
    altered = source.copy_empty_like()
    kept_environment_reset = False
    for item in source.data:
        env_reset = item.operation.name == "reset" and source.find_bit(item.qubits[0]).index == 1
        if env_reset:
            if kept_environment_reset:
                continue
            kept_environment_reset = True
        altered.append(item.operation, item.qubits, item.clbits)
    actual = simulate_lindblad(steps=8, order=2)
    from qiskit.quantum_info import partial_trace
    missing_resets = partial_trace(DensityMatrix.from_instruction(altered), [1]).data
    assert np.linalg.norm(actual-missing_resets) > 0.01


def test_generation_and_dense_check_have_separate_caps():
    assert build_lindblad_circuit(steps=257).num_qubits == 2
    with pytest.raises(ValueError, match="simulation"):
        simulate_lindblad(steps=257)
    with pytest.raises(ValueError, match="scale"):
        exact_channel(time=1000)
    with pytest.raises(ValueError, match="unknown"):
        build_lindblad_circuit(input_label=None)


@pytest.mark.parametrize("kwargs", [{"steps": True}, {"steps": 0}, {"steps": 4097},
    {"steps": 10**9, "time": 0, "rate": 0, "hx": 0, "hz": 0}, {"order": 2.0},
    {"time": -1}, {"rate": -0.1}, {"hx": float("nan")}, {"hz": True}])
def test_invalid_or_unbounded_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_lindblad_circuit(**kwargs)
