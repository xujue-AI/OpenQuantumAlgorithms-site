"""Independent diagonal thermal checks of actual purification and readout gates."""
import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, partial_trace

from openquantum.simulation.gibbs import (
    bounded_mean_shots, build_gibbs_mixed_preparation, build_gibbs_trial,
    decompose_gibbs, diagonal_gibbs_purification,
    diagonal_ising_energies, exact_diagonal_gibbs, gibbs_check,
    gibbs_probabilities, positive_amplitude_circuit, reduced_gibbs_state,
)


def test_ising_energy_bit_order_and_known_spectrum():
    np.testing.assert_allclose(diagonal_ising_energies(), [0.75, -1.15, -0.05, 0.45])
    np.testing.assert_allclose(diagonal_ising_energies(1, fields=(0.4,)), [0.4, -0.4])


@pytest.mark.parametrize("probabilities", [[.3, .7], [.1, .2, .3, .4], [0, 0, .7, .3], [1, 0, 0, 0]])
def test_positive_amplitudes_and_controlled_prefixes(probabilities):
    actual = Statevector.from_instruction(positive_amplitude_circuit(probabilities)).data
    np.testing.assert_allclose(actual, np.sqrt(probabilities), atol=1e-12)


@pytest.mark.parametrize("energies", [[0, 1], [.75, -1.15, -.05, .45], [0, 0, 1, 1]])
@pytest.mark.parametrize("beta", [0, .25, 1.25, 5])
def test_actual_purification_and_independent_gibbs_reference(energies, beta):
    n = (len(energies)-1).bit_length()
    circuit = diagonal_gibbs_purification(energies, beta)
    actual = Statevector.from_instruction(circuit).data
    weights = np.exp(-beta*(np.array(energies)-min(energies)))
    p = weights/weights.sum()
    reference = np.zeros(2**(2*n), dtype=complex)
    for x, value in enumerate(p):
        reference[x + (x << n)] = np.sqrt(value)
    np.testing.assert_allclose(actual, reference, atol=1e-12)
    reduced = partial_trace(Statevector(actual), list(range(n, 2*n))).data
    np.testing.assert_allclose(reduced, np.diag(p), atol=1e-12)
    np.testing.assert_allclose(exact_diagonal_gibbs(energies, beta), np.diag(p), atol=1e-12)
    result = gibbs_check(energies, beta)
    assert result["trace_distance"] < 1e-12
    assert result["trace_real"] == pytest.approx(1)
    assert result["minimum_eigenvalue"] >= -1e-12
    for key, value in result["actual_observables"].items():
        assert value == pytest.approx(result["reference_observables"][key], abs=1e-12)


def test_energy_shift_invariance_and_uniform_high_temperature_state():
    energies = diagonal_ising_energies()
    np.testing.assert_allclose(reduced_gibbs_state(energies, 1.25), reduced_gibbs_state(energies+31.5, 1.25), atol=1e-12)
    np.testing.assert_allclose(reduced_gibbs_state(energies, 0), np.eye(4)/4, atol=1e-12)
    np.testing.assert_allclose(gibbs_probabilities([-1e308, 1e308], 0), [.5, .5])


def test_explicit_mixed_state_reset_schedule_and_measured_trial():
    energies = diagonal_ising_energies()
    mixed = build_gibbs_mixed_preparation(energies, 1.25)
    assert mixed.count_ops()["reset"] == 6
    density = DensityMatrix.from_instruction(mixed)
    np.testing.assert_allclose(partial_trace(density, [2, 3]).data,
                               exact_diagonal_gibbs(energies, 1.25), atol=1e-12)
    np.testing.assert_allclose(partial_trace(density, [0, 1]).data,
                               np.diag([1, 0, 0, 0]), atol=1e-12)
    trial = decompose_gibbs(build_gibbs_trial(energies, 1.25))
    assert trial.count_ops()["reset"] == 4 and trial.count_ops()["measure"] == 2
    assert "reset" in qasm3.dumps(trial)


def test_decomposition_and_system_only_measured_export():
    energies = diagonal_ising_energies()
    circuit = diagonal_gibbs_purification(energies, 1.25)
    compiled = decompose_gibbs(circuit)
    assert Operator(compiled).equiv(Operator(circuit))
    measured = decompose_gibbs(diagonal_gibbs_purification(energies, 1.25, measure=True))
    assert measured.count_ops()["measure"] == 2
    assert all(measured.find_bit(i.qubits[0]).index < 2 for i in measured.data if i.operation.name == "measure")
    export = qasm3.dumps(measured)
    assert "measure" in export and "gate " not in export
    assert not {"initialize", "state_preparation", "unitary"} & set(compiled.count_ops())


@pytest.mark.parametrize("energies,beta", [([0, 1, 2], 1), ([0, np.nan], 1), ([0, 1j], 1), ([0, 1], -1),
    ([0, 1], np.inf), ([0, 1], True), ([0, 701], 1), (np.zeros(512), 0)])
def test_invalid_or_out_of_domain_thermal_preparation(energies, beta):
    with pytest.raises(ValueError):
        diagonal_gibbs_purification(energies, beta)


def test_classical_preparation_and_dense_simulation_limits_are_distinct():
    energies = np.linspace(0, 1, 128)
    assert diagonal_gibbs_purification(energies, .5).num_qubits == 14
    with pytest.raises(ValueError, match="validation"):
        reduced_gibbs_state(energies, .5)
    with pytest.raises(ValueError, match="dense"):
        exact_diagonal_gibbs(energies, .5)


def test_single_mean_shot_counts_do_not_hide_energy_outcome_width():
    assert bounded_mean_shots(.03, .005) == 13315
    energies = diagonal_ising_energies()
    assert bounded_mean_shots(.03, .005, energies.min(), energies.max()) < 13315
    with pytest.raises(ValueError):
        bounded_mean_shots(.03, 1)


def test_statistical_budget_numeric_boundaries_fail_explicitly():
    with pytest.raises(ValueError, match="shot count"):
        bounded_mean_shots(1e-200, .005)
    with pytest.raises(ValueError, match="shot count"):
        bounded_mean_shots(.03, .005, -1e200, 1e200)
    assert bounded_mean_shots(.03, 1e-320) > 1_000_000
