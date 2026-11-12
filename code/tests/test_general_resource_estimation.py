"""Shared FT adapter checks without changing the frozen Grover baseline."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from openquantum.resource_estimation import (
    BoundedObservableBudget,
    CircuitResourceBudget,
    SurfaceCodeProfile,
    choose_num_ts_per_rotation,
    estimate_circuit_resources,
    prepare_measured_circuit,
    synthesis_model_error,
)


@pytest.mark.parametrize("rotations,allowance", [(1, 0.001), (2, 0.0001), (100, 1e-8)])
def test_synthesis_cost_meets_explicit_allowance_in_mean_fit_domain(rotations, allowance):
    count = choose_num_ts_per_rotation(rotations, allowance)
    assert synthesis_model_error(count) < 1e-4
    assert rotations * synthesis_model_error(count) <= allowance
    if count > 12:
        assert rotations * synthesis_model_error(count - 1) > allowance
    assert choose_num_ts_per_rotation(0, 0) == 5
    with pytest.raises(ValueError, match="positive synthesis"):
        choose_num_ts_per_rotation(1, 0)


def test_bound_observable_keeps_bias_precision_and_failure_confidence_separate():
    contract = BoundedObservableBudget(
        absolute_error_target=0.1,
        algorithm_bias_bound=0.02,
        synthesis_bias_allowance=0.01,
        sampling_error_allowance=0.07,
        sampling_failure_probability=0.01,
        physical_failure_probability=0.005,
        outcome_min=0,
        outcome_max=3,
    )
    shots = contract.shots_required
    bound = 2 * math.exp(-2 * shots * 0.07**2 / 3**2)
    assert bound <= 0.01
    assert 2 * math.exp(-2 * (shots - 1) * 0.07**2 / 3**2) > 0.01
    resource = contract.resource_budget()
    assert resource.synthesis_model_error_per_execution == pytest.approx(0.01 / 3)
    assert resource.physical_failure_per_execution * shots == pytest.approx(0.005)
    assert contract.as_contract()["conditional_confidence_lower_bound"] == pytest.approx(0.985)
    with pytest.raises(ValueError, match="absolute error target"):
        BoundedObservableBudget(0.1, 0.05, 0.05, 0.05, 0.01, 0.005)


def test_static_lowering_preserves_the_ideal_nonclifford_state():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(0.213, 0)
    circuit.rz(0.317, 1)
    circuit.measure_all()
    before = dict(circuit.count_ops())
    compiled = prepare_measured_circuit(circuit)
    original_state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    lowered_state = Statevector.from_instruction(compiled.remove_final_measurements(inplace=False))
    assert abs(np.vdot(original_state.data, lowered_state.data)) ** 2 == pytest.approx(1, abs=1e-12)
    assert dict(circuit.count_ops()) == before


def test_unaudited_work_is_rejected_before_estimation():
    unbound = QuantumCircuit(1)
    unbound.rz(Parameter("theta"), 0)
    unbound.measure_all()
    with pytest.raises(ValueError, match="Bind all"):
        prepare_measured_circuit(unbound)
    reset = QuantumCircuit(1)
    reset.reset(0)
    reset.measure_all()
    with pytest.raises(ValueError, match="reset"):
        prepare_measured_circuit(reset)
    mid_measure = QuantumCircuit(1, 1)
    mid_measure.measure(0, 0)
    mid_measure.x(0)
    with pytest.raises(ValueError, match="final measurements"):
        prepare_measured_circuit(mid_measure)


def test_actual_rx_rz_estimation_separates_native_base_error_from_physical_faults(tmp_path):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(0.213, 0)
    circuit.rz(0.317, 1)
    circuit.measure_all()
    budget = CircuitResourceBudget(0.0001, 0.0005, executions=3)
    report = estimate_circuit_resources(
        circuit, budget=budget, profile=SurfaceCodeProfile(), cache_dir=tmp_path / "factory-cache"
    )
    assert report["status"] == "computed_conditional_model"
    assert "rx(" in report["qasm3"] or "ry(" in report["qasm3"]
    assert "rz(" in report["qasm3"]
    synthesis = report["synthesis"]
    assert synthesis["num_arbitrary_rotations"] > 0
    assert synthesis["model_error_per_execution"] <= budget.synthesis_model_error_per_execution
    assert not synthesis["actual_synthesis_sequence_generated"]
    assert not synthesis["instance_approximation_certificate_verified"]
    assert report["pareto_points"]
    for point in report["pareto_points"]:
        assert point["peak_physical_qubits"] > 0
        assert point["serial_runtime_all_executions_ns"] == point["runtime_per_execution_ns"] * 3
        assert point["physical_failure_union_bound_all_executions"] <= 0.0005 + 1e-12
        assert point["qdk_combined_error_score_per_execution"] == pytest.approx(
            point["synthesis_model_error_per_execution"] + point["physical_failure_model_per_execution"]
        )
        assert point["model_properties"]["NUM_TS_PER_ROTATION"] == synthesis["fixed_num_ts_per_rotation"]
