"""Independent permutation/evolution and full measured-workload checks."""
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from openquantum.quantum_walks import (
    MAX_PRIMITIVE_OPERATIONS, build_conditional_shift, build_coined_walk,
    classical_cycle_distribution, coined_observable_budget, lower_walk,
    reference_step_matrix, sample_position_outcomes, simulate_coined_walk,
)


@pytest.mark.parametrize("n", range(1,6))
def test_shift_is_exact_independent_permutation_on_both_coin_branches(n):
    vertices = 1 << n
    expected = np.zeros((2*vertices,2*vertices), dtype=complex)
    for column in range(2*vertices):
        coin, position = divmod(column, vertices)
        row = coin*vertices + (position + (1 if coin else -1)) % vertices
        expected[row,column] = 1
    shift = build_conditional_shift(n)
    np.testing.assert_allclose(Operator(shift).data, expected, atol=2e-13, rtol=0)
    assert shift.num_qubits == n+1 and shift.num_ancillas == 0
    assert len(shift.data) == 2*n+2
    rng = np.random.default_rng(102+n)
    vector = rng.normal(size=2*vertices)+1j*rng.normal(size=2*vertices)
    vector /= np.linalg.norm(vector)
    np.testing.assert_allclose(Statevector(vector).evolve(shift).data, expected@vector, atol=2e-13,rtol=0)


@pytest.mark.parametrize("n,steps", [(n,t) for n in (1,2,3,4,5) for t in (0,1,2,4,7)])
@pytest.mark.parametrize("coin", ["zero","one","balanced"])
def test_actual_walk_matches_independent_full_matrix_and_initial_state(n,steps,coin):
    position=(1<<n)-1
    result=simulate_coined_walk(n,steps,initial_position=position,initial_coin=coin)
    assert result["independent_dense_reference_performed"]
    assert result["state_amplitude_max_error"]<3e-12
    assert result["independent_amplitude_recurrence_max_error"]<3e-12
    assert result["norm"] == pytest.approx(1,abs=3e-13)
    assert result["probabilities"].sum() == pytest.approx(1,abs=3e-13)
    impossible=[x for x in range(1<<n) if (x-position-steps)%2]
    assert result["probabilities"][impossible].sum() < 3e-13


def test_one_step_order_coin_then_shift_and_balanced_cycle_demo():
    reference=reference_step_matrix(3)
    circuit=QuantumCircuit(4)
    circuit.h(3);circuit.compose(build_conditional_shift(3),inplace=True)
    np.testing.assert_allclose(Operator(circuit).data,reference,atol=2e-13,rtol=0)
    actual=simulate_coined_walk(3,4)["probabilities"]
    np.testing.assert_allclose(actual,[1/8,0,3/8,0,1/8,0,3/8,0],atol=3e-13,rtol=0)
    classical=classical_cycle_distribution(3,4)
    np.testing.assert_allclose(classical,[3/8,0,1/4,0,1/8,0,1/4,0],atol=1e-15,rtol=0)
    assert max(abs(actual-classical))>.24


@pytest.mark.parametrize("n", [2,3,4])
def test_complete_lowered_circuit_contains_preparation_and_all_final_readouts(n):
    source=build_coined_walk(n,2,initial_position=3,measure=True)
    compiled=lower_walk(source)
    assert compiled.num_qubits==n+1
    assert compiled.count_ops()["measure"]==n+1
    assert all(item.operation.name=="measure" for item in compiled.data[-(n+1):])
    original_unitary=source.remove_final_measurements(inplace=False)
    lowered_unitary=compiled.remove_final_measurements(inplace=False)
    assert Operator(original_unitary).equiv(Operator(lowered_unitary),atol=3e-12,rtol=0)
    assert len(compiled.data)<=MAX_PRIMITIVE_OPERATIONS


@pytest.mark.parametrize("args", [(0,1),(13,1),(True,1),(2,-1),(2,257),(2,1.5),(12,256)])
def test_generation_limits_reject_invalid_or_unbounded_work(args):
    with pytest.raises(ValueError):build_coined_walk(*args)


@pytest.mark.parametrize("kwargs", [{"initial_position":4},{"initial_position":-1},{"initial_coin":"plus"},{"measure":1}])
def test_input_contracts_are_explicit(kwargs):
    with pytest.raises(ValueError):build_coined_walk(2,1,**kwargs)


def test_generator_and_simulator_limits_are_separate():
    assert build_coined_walk(12,1).num_qubits==13
    with pytest.raises(ValueError,match="Statevector"):simulate_coined_walk(12,1)
    assert build_coined_walk(3,65).num_qubits==4
    with pytest.raises(ValueError,match="Statevector"):simulate_coined_walk(3,65)


def test_reject_opaque_and_dynamic_source_before_lowering():
    for method in ("reset","unitary"):
        circuit=QuantumCircuit(1)
        if method=="reset":circuit.reset(0)
        else:circuit.unitary(np.eye(2),[0])
        with pytest.raises(ValueError,match="supported"):lower_walk(circuit)


@pytest.mark.parametrize("angle", [float("nan"),float("inf"),-float("inf")])
def test_reject_nonfinite_rotations_before_compilation(angle):
    circuit=QuantumCircuit(1);circuit.rz(angle,0)
    with pytest.raises(ValueError,match="finite"):lower_walk(circuit)


def test_reject_postmeasurement_gates():
    circuit=QuantumCircuit(1,1);circuit.measure(0,0);circuit.h(0)
    with pytest.raises(ValueError,match="final measurements"):lower_walk(circuit)


def test_one_fixed_event_budget_and_seeded_sampling():
    budget=coined_observable_budget()
    assert budget.shots_required==math.ceil(math.log(400)/(2*.03**2))==3329
    assert budget.resource_budget().executions==3329
    assert budget.resource_budget().physical_failure_per_execution==pytest.approx(.005/3329)
    assert budget.as_contract()["certified_confidence_lower_bound"] is None
    probabilities=simulate_coined_walk(3,4)["probabilities"]
    counts=sample_position_outcomes(probabilities,3329,seed=731)
    assert counts.sum()==3329
    np.testing.assert_array_equal(counts,sample_position_outcomes(probabilities,3329,seed=731))


@pytest.mark.parametrize("values", [[-.1,1.1],[.5,.4],[np.nan,1],[.5+1j,.5-1j],[[1]]])
def test_sampling_rejects_nonprobabilities(values):
    with pytest.raises(ValueError):sample_position_outcomes(np.asarray(values),10)
