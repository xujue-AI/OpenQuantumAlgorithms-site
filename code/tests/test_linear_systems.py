# SPDX-License-Identifier: MIT
"""Independent small matrix/phase/measurement checks for structured HHL."""
import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from openquantum.linear_systems import (
    GridXSystem, accepted_observable_workload, analytic_success_probability, build_hhl, compare_hhl,
    controlled_evolution, decompose, dense_matrix, input_preparation,
    measured_hhl_trial, phase_estimation, qft_circuit, reciprocal_rotation,
)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_qft_matches_fourier_definition(width):
    size = 1 << width
    rows, cols = np.indices((size, size))
    fourier = np.exp(2j*np.pi*rows*cols/size)/math.sqrt(size)
    np.testing.assert_allclose(Operator(qft_circuit(width)).data, fourier, atol=2e-14, rtol=0)
    np.testing.assert_allclose(Operator(qft_circuit(width, inverse=True)).data, fourier.conj().T, atol=2e-14, rtol=0)


@pytest.mark.parametrize("system", [GridXSystem(), GridXSystem((1,2),1,3), GridXSystem((0,1),2,3)])
@pytest.mark.parametrize("power", [1, 2, 4])
def test_controlled_evolution_includes_identity_phase(system, power):
    evolution = expm(2j*np.pi*dense_matrix(system)*power)
    actual = Operator(controlled_evolution(system, power)).data
    expected = np.zeros_like(actual)
    expected[::2, ::2] = np.eye(len(evolution))
    expected[1::2, 1::2] = evolution
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)


@pytest.mark.parametrize("system", [GridXSystem(), GridXSystem((1,2),1,3), GridXSystem((0,1),2,3)])
def test_qpe_assigns_exact_eigenvalue_to_little_endian_clock(system):
    n = system.n_system
    for bits in range(1 << n):
        circuit = QuantumCircuit(n+system.phase_qubits)
        for qubit in range(n):
            if (bits >> qubit) & 1:
                circuit.x(qubit)
        circuit.h(list(range(n)))
        circuit.compose(phase_estimation(system), inplace=True)
        probabilities = Statevector.from_instruction(circuit).probabilities()
        label = system.offset+sum(w*((bits >> j) & 1) for j,w in enumerate(system.weights))
        assert sum(probabilities[index] for index in range(len(probabilities)) if index >> n == label) == pytest.approx(1, abs=3e-14)


@pytest.mark.parametrize("system", [GridXSystem(), GridXSystem((1,1),1,3)])
def test_reciprocal_lookup_has_correct_flag_probability(system):
    m = system.phase_qubits
    for label in range(system.denominator):
        initial = np.zeros(1 << (m+1), complex)
        initial[label] = 1
        result = Statevector(initial).evolve(reciprocal_rotation(system)).data
        expected = (system.offset/label)**2 if label in system.phase_labels else 0
        assert abs(result[label+(1 << m)])**2 == pytest.approx(expected, abs=3e-14)
        assert np.count_nonzero(abs(result)>1e-12) <= 2


@pytest.mark.parametrize("system,angles", [
    (GridXSystem(),None), (GridXSystem(),(.37,)), (GridXSystem(),(-.9,)),
    (GridXSystem((1,1),1,3),(.3,.7)), (GridXSystem((1,2,1),1,3),(.1,-.2,.4)),
    (GridXSystem((0,0),3,3),(1.1,-.4)), (GridXSystem((1,3),2,3),(.2,1.5)),
    (GridXSystem((0,),1,1),(.7,)), (GridXSystem((1,),2,2),(-.2,)),
])
def test_complete_hhl_matches_independent_classical_solve(system, angles):
    result = compare_hhl(system, angles=angles)
    assert result["trace_distance_to_normalized_classical_solution"] < 3e-13
    assert result["state_fidelity"] == pytest.approx(1, abs=3e-13)
    assert result["success_probability"] == pytest.approx(result["analytic_success_probability"], abs=3e-13)
    assert result["clock_leakage_given_flag_probability"] < 3e-13
    assert result["quantum_observable"] == pytest.approx(result["classical_observable"], abs=3e-13)
    assert result["success_probability"] == pytest.approx(analytic_success_probability(system,angles=angles),abs=3e-13)


@pytest.mark.parametrize("observable", ["X", "Y", "Z"])
def test_actual_measurement_basis_matches_classical_pauli(observable):
    measured = measured_hhl_trial(observable=observable)
    probabilities = Statevector.from_instruction(measured.remove_final_measurements(inplace=False)).probabilities()
    success_indices = [8,9]
    mean = (probabilities[8]-probabilities[9])/sum(probabilities[index] for index in success_indices)
    paulis = {"X":np.array([[0,1],[1,0]]), "Y":np.array([[0,-1j],[1j,0]]), "Z":np.diag([1,-1])}
    x = np.array([3,1])/math.sqrt(10)
    assert mean == pytest.approx(np.vdot(x,paulis[observable]@x).real, abs=3e-14)
    assert measured.count_ops()["measure"] == measured.num_qubits


def test_lowered_unitary_matches_original_and_has_no_opaque_gates():
    original = build_hhl(GridXSystem((1,1),1,3),angles=(.1,.2))
    assert Operator(original).equiv(Operator(decompose(original)), atol=2e-12)
    measured = decompose(measured_hhl_trial())
    assert measured.count_ops()["measure"] == 4
    assert set(measured.count_ops()) <= {"h","x","z","s","sdg","t","tdg","rx","ry","rz","cx","measure","barrier"}


def test_cap_budget_and_confidence_are_separate():
    workload = accepted_observable_workload(.625)
    n, cap = workload["accepted_samples"], workload["attempt_cap_per_accepted_slot"]
    assert n*(1-.625)**cap <= .005
    assert n*(1-.625)**(cap-1) > .005
    assert 2*math.exp(-n*.03**2/2) <= .005
    assert workload["total_capped_attempts"] == n*cap
    assert not workload["certified_synthesis_or_physical_success_guarantee"]
    assert accepted_observable_workload(1)["attempt_cap_per_accepted_slot"] == 1


@pytest.mark.parametrize("arguments", [
    {"weights":()}, {"weights":(1,),"offset":0}, {"weights":(3,),"offset":1},
    {"weights":(-1,)}, {"weights":(True,)}, {"phase_qubits":9}, {"phase_qubits":True},
])
def test_invalid_systems_rejected(arguments):
    with pytest.raises(ValueError): GridXSystem(**arguments)


def test_generation_and_simulation_caps_are_separate():
    system = GridXSystem((1,)*10,1,4)
    assert build_hhl(system).num_qubits == 15
    with pytest.raises(ValueError,match="Statevector"): compare_hhl(system)
    with pytest.raises(ValueError,match="Dense"): dense_matrix(system)
    with pytest.raises(ValueError): input_preparation(1,[float("nan")])
    with pytest.raises(ValueError): measured_hhl_trial(observable="I")
    with pytest.raises(ValueError): accepted_observable_workload(0)
