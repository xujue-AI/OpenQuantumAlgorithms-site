"""Independent algebra, density instruments and conditioned thermal checks."""
import itertools
import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, partial_trace
from scipy.linalg import expm

from openquantum.simulation.thermal_filter import (
    DEFAULT_TERMS, actual_filter_block, build_thermal_filter, build_thermal_lcu,
    build_thermal_trial, conditioned_synthesis_bounds, decompose_thermal_filter,
    postselected_thermal_purification, thermal_energy_workload, thermal_filter_check,
    thermal_pauli_terms,
)

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}


def matrix(terms):
    result = np.zeros((2**len(terms[0][1]),)*2, dtype=complex)
    for c, label in terms:
        product = np.array([[1+0j]])
        for letter in label:
            product = np.kron(product, PAULI[letter])
        result += c*product
    return result


def polynomial(H, beta, degree):
    return sum(np.linalg.matrix_power(-beta*H/2, k)/math.factorial(k) for k in range(degree+1))


@pytest.mark.parametrize("terms", [DEFAULT_TERMS, ((.4, "Y"), (.6, "Z")),
                                  ((.2, "ZI"), (.4, "XX"), (-.3, "IY"))])
@pytest.mark.parametrize("degree", [0, 1, 2, 4, 6, 12])
def test_hermitian_pauli_aggregation_matches_independent_matrix_powers(terms, degree):
    aggregated = thermal_pauli_terms(terms, .7, degree)
    np.testing.assert_allclose(matrix(aggregated), polynomial(matrix(terms), .7, degree), atol=2e-13)
    assert all(c.imag == 0 for c, _ in aggregated)
    assert np.allclose(matrix(aggregated), matrix(aggregated).conj().T)


@pytest.mark.parametrize("terms", [DEFAULT_TERMS, ((.4, "Y"), (.6, "Z")),
                                  ((.2, "ZI"), (.4, "XX"), (-.3, "IY"))])
def test_actual_access_and_bell_purification_with_complex_paulis(terms):
    beta, degree = 1.25, 4
    T = polynomial(matrix(terms), beta, degree)
    access = build_thermal_lcu(terms, beta, degree)
    np.testing.assert_allclose(actual_filter_block(terms, beta, degree), T/access.normalization, atol=1e-12)
    state, p = postselected_thermal_purification(terms, beta, degree)
    d = len(T)
    expected_branch = T.reshape(-1, order="F")/(math.sqrt(d)*access.normalization)
    expected_p = np.vdot(expected_branch, expected_branch).real
    np.testing.assert_allclose(state, expected_branch/np.sqrt(expected_p), atol=1e-12)
    assert p == pytest.approx(expected_p, abs=1e-12)
    reduced = partial_trace(Statevector(state), list(range(access.n_system, 2*access.n_system))).data
    expected = T@T.conj().T
    expected /= np.trace(expected)
    np.testing.assert_allclose(reduced, expected, atol=1e-12)


def test_noncommuting_analytic_bloch_reference_and_convergence():
    H = matrix(DEFAULT_TERMS)
    radius = np.sqrt(.6**2+.4**2)
    assert not np.allclose(PAULI["X"]@PAULI["Z"], PAULI["Z"]@PAULI["X"])
    errors = []
    for degree in (1, 2, 4, 6, 8):
        result = thermal_filter_check(degree=degree)
        assert result["actual_access_block_spectral_error"] < 1e-12
        assert result["trace_real"] == pytest.approx(1)
        assert result["minimum_eigenvalue"] >= -1e-12
        errors.append(result["gibbs_trace_distance"])
    assert all(a > b for a, b in zip(errors, errors[1:]))
    result = thermal_filter_check(degree=8)
    analytic = (np.eye(2)-np.tanh(1.25*radius)*H/radius)/2
    actual = np.array(result["actual_reduced_state_real"])+1j*np.array(result["actual_reduced_state_imag"])
    np.testing.assert_allclose(actual, analytic, atol=1e-9, rtol=0)
    assert errors[-1] < 2e-9


def test_infinite_temperature_is_maximally_mixed_without_postselection_cost():
    for degree in (0, 1, 6):
        result = thermal_filter_check(beta=0, degree=degree)
        assert result["acceptance_probability"] == pytest.approx(1)
        assert result["complete_trials_for_one_measured_output_99_percent"] == 1
        np.testing.assert_allclose(result["actual_reduced_state_real"], np.eye(2)/2, atol=1e-12)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_actual_zero_label_measurement_reference_reset_and_basis_readout(axis):
    # Includes imaginary matrix entries so a transposed/Y-sign mistake is visible.
    terms = ((.4, "Y"), (.6, "Z"))
    trial = build_thermal_trial(terms, measurement=axis)
    n, m = trial.metadata["n_system"], trial.metadata["n_index"]
    assert trial.count_ops()["reset"] == trial.num_qubits+n
    assert trial.count_ops()["measure"] == m+n
    density = DensityMatrix.from_int(0, 2**trial.num_qubits)
    accepted = 1.0
    measured_mean = None
    for item in trial.data:
        qargs = [trial.find_bit(q).index for q in item.qubits]
        if item.operation.name == "measure":
            bit = trial.find_bit(item.clbits[0]).index
            if bit < m:
                keep = (np.arange(len(density.data)) & (1 << qargs[0])) == 0
                branch = density.data * keep[:, None] * keep[None, :]
                p = float(np.trace(branch).real)
                accepted *= p
                density = DensityMatrix(branch/p)
            else:
                zero = (np.arange(len(density.data)) & (1 << qargs[0])) == 0
                p0 = float(np.diag(density.data)[zero].sum().real)
                measured_mean = 2*p0-1
        elif item.operation.name == "reset":
            density = density.reset(qargs)
        else:
            density = density.evolve(item.operation, qargs=qargs)
    result = thermal_filter_check(terms)
    assert accepted == pytest.approx(result["acceptance_probability"], abs=1e-12)
    assert measured_mean == pytest.approx(result["actual_observables"][axis+"0"], abs=1e-12)
    reference = partial_trace(density, [0]+list(range(2, trial.num_qubits))).data
    np.testing.assert_allclose(reference, np.diag([1, 0]), atol=1e-12)


def test_primitive_decomposition_and_readout_export():
    core = build_thermal_filter()
    compiled = decompose_thermal_filter(core)
    np.testing.assert_allclose(Operator(core).data, Operator(compiled).data, atol=1e-12)
    trial = decompose_thermal_filter(build_thermal_trial())
    text = qasm3.dumps(trial)
    assert "reset" in text and "measure" in text and "gate " not in text
    assert not {"unitary", "initialize", "state_preparation"} & set(compiled.count_ops())


@pytest.mark.parametrize("kwargs", [{"beta": -1}, {"beta": True}, {"beta": np.inf}, {"degree": 13},
    {"degree": 2.5}, {"degree": True}, {"terms": ((1j, "X"),)}, {"terms": ()},
    {"terms": ((1, "ZZZZZ"),)}, {"terms": ((1e308, "X"), (1e308, "Z"))}])
def test_invalid_thermal_models_and_polynomial_domains(kwargs):
    with pytest.raises(ValueError):
        build_thermal_filter(**kwargs)


def test_zero_taylor_filter_and_bounded_input_iteration():
    with pytest.raises(ValueError, match="zero"):
        build_thermal_filter(((2.0, "I"),), beta=1, degree=1)
    with pytest.raises(ValueError, match="32"):
        build_thermal_filter(itertools.repeat((.1, "X")), beta=.1)


def test_large_materialization_is_separate_from_statevector_validation():
    labels = ["".join(p) for p in itertools.islice(itertools.product("IXYZ", repeat=4), 32)]
    terms = tuple((.01, p) for p in labels)
    assert build_thermal_filter(terms, .1, 1).num_qubits == 13
    with pytest.raises(ValueError, match="12"):
        postselected_thermal_purification(terms, .1, 1)


def test_grouped_energy_sampling_acceptance_and_physical_budgets():
    result = thermal_filter_check()
    p = result["acceptance_probability"]
    workload = thermal_energy_workload(p)
    assert [g["axis"] for g in workload["groups"]] == ["X", "Z"]
    assert all(g["accepted_shots"] == 14855 for g in workload["groups"])
    assert workload["total_accepted_shots"] == 29710
    assert workload["worst_case_complete_trials"] == 742750
    for group in workload["groups"]:
        cap = group["complete_trial_cap_per_accepted_shot"]
        assert (1-p)**cap <= .005/workload["total_accepted_shots"]
        assert (1-p)**(cap-1) > .005/workload["total_accepted_shots"]
    assert sum(abs(g["coefficient"])*g["pauli_statistical_error"] for g in workload["groups"]) == pytest.approx(.03)
    assert not workload["sampling_performed"]


def test_conditioning_synthesis_bound_is_hypothetical_and_rejects_zero_branch():
    p = .25
    bounds = conditioned_synthesis_bounds(p, .001)
    assert bounds["conditional_state_trace_distance_bound"] == pytest.approx(.004)
    assert bounds["single_pauli_mean_bias_bound"] == pytest.approx(.008)
    assert bounds["perturbed_acceptance_lower_bound"] == pytest.approx(.499**2)
    assert "hypothetical" in bounds["status"]
    with pytest.raises(ValueError):
        conditioned_synthesis_bounds(p, .5)
