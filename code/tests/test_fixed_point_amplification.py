import math
from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Operator, Statevector

from openquantum.amplitude_estimation import threshold_preparation
from openquantum.fixed_point_amplification import (
    MAX_SIM_QUBITS, build_fixed_point_amplification, fixed_point_schedule,
    good_probability, ideal_success_probability, lower_fixed_point, projector_phase,
)


def independent_chebyshev(p, length, delta):
    """Recurrence independent of production acos/cosh evaluation."""
    if p in (0., 1.):
        return p
    gamma = 1 / np.cosh(np.arccosh(1 / delta) / length)
    x = np.sqrt(1 - p) / gamma
    previous, current = np.longdouble(1), np.longdouble(x)
    for _ in range(2, length + 1):
        previous, current = current, 2 * x * current - previous
    return float(1 - (np.longdouble(delta) * current)**2)


@pytest.mark.parametrize("lower,delta", [(.1,.05),(.01,.1),(.25,.2),(.95,.5),(1.,.05)])
def test_schedule_uses_promise_and_meets_numerical_width(lower, delta):
    schedule = fixed_point_schedule(lower, delta)
    assert schedule.length % 2 == 1
    assert schedule.numerical_threshold_width <= lower
    assert schedule.betas == tuple(-x for x in reversed(schedule.alphas))
    if schedule.length > 1:
        assert math.tanh(math.acosh(1 / delta) / (schedule.length - 2))**2 > lower - 1e-15
    assert schedule.ideal_failure_bound == delta**2
    assert not schedule.as_metadata()["hardware_success_certified"]


@pytest.mark.parametrize("lower,delta", [(.1,.05),(.01,.1),(.25,.2),(.95,.5),(1.,.05)])
def test_actual_one_qubit_gates_match_independent_chebyshev_law(lower, delta):
    schedule = fixed_point_schedule(lower, delta)
    # Endpoints, below-promise values and the guaranteed band all matter.
    probabilities = sorted({0., 1., lower, lower/2, .01, .1, .25, .5, .9})
    for p in probabilities:
        preparation = QuantumCircuit(1)
        preparation.ry(2 * math.asin(math.sqrt(p)), 0)
        circuit = build_fixed_point_amplification(preparation, [0], 1, lower, delta)
        expected = independent_chebyshev(p, schedule.length, delta)
        assert math.isclose(good_probability(circuit, [0], 1), expected, abs_tol=2e-12)
        assert math.isclose(ideal_success_probability(p, schedule), expected, abs_tol=2e-12)
        if p >= lower:
            assert good_probability(circuit, [0], 1) >= 1 - delta**2 - 2e-12


@pytest.mark.parametrize("wires,pattern", [([0],0),([2],1),([2,0],1),([0,1,2],5)])
def test_projector_bit_order_against_direct_diagonal(wires, pattern):
    expected = np.eye(8, dtype=complex)
    for index in range(8):
        if all((index >> q)&1 == (pattern >> bit)&1 for bit, q in enumerate(wires)):
            expected[index,index] = np.exp(.73j)
    np.testing.assert_allclose(Operator(projector_phase(3,wires,pattern,.73)).data,
                               expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("lower,delta", [(.1,.05),(.25,.2),(.95,.5)])
def test_full_complex_operator_against_independent_projector_matrices(lower, delta):
    preparation = QuantumCircuit(3)
    preparation.ry(.8,0)
    preparation.h(1)
    preparation.cx(0,2)
    preparation.rz(.47,1)
    preparation.p(-.2,2)
    preparation.global_phase = .123
    a = Operator(preparation).data
    psi = a[:,0]
    identity = np.eye(8, dtype=complex)
    good = np.diag([int(((i>>2)&1)==1 and (i&1)==0) for i in range(8)])
    schedule = fixed_point_schedule(lower,delta)
    expected = a.copy()
    for alpha,beta in zip(schedule.alphas,schedule.betas,strict=True):
        source_phase = identity + (np.exp(-1j*alpha)-1)*np.outer(psi,psi.conj())
        target_phase = identity + (np.exp(1j*beta)-1)*good
        expected = -source_phase @ target_phase @ expected
    actual = build_fixed_point_amplification(preparation,[2,0],1,lower,delta)
    np.testing.assert_allclose(Operator(actual).data,expected,atol=3e-12,rtol=0)
    # This comparison keeps global phase; a phase-equivalence test is weaker.
    np.testing.assert_allclose(Operator(lower_fixed_point(actual)).data,expected,atol=3e-12,rtol=0)


@pytest.mark.parametrize("n,threshold", [(1,1),(2,1),(3,3),(4,7)])
def test_threshold_state_and_conditional_good_direction(n, threshold):
    preparation = threshold_preparation(n,threshold)
    circuit = build_fixed_point_amplification(preparation, [n],1,.1,.05)
    state = Statevector.from_instruction(circuit).data
    exact = np.zeros_like(state)
    exact[2**n:2**n+threshold] = 1/math.sqrt(threshold)
    projected = state.copy()
    projected[:2**n] = 0
    projected /= np.linalg.norm(projected)
    assert abs(np.vdot(exact,projected))**2 > 1-1e-12
    schedule = fixed_point_schedule(.1,.05)
    assert math.isclose(good_probability(circuit,(x for x in [n]),1),
                        independent_chebyshev(threshold/2**n,schedule.length,.05),abs_tol=3e-12)


def test_call_ledger_and_measured_lowering():
    preparation = threshold_preparation(2,1)
    circuit = build_fixed_point_amplification(preparation,(q for q in [2]),1,.1,.05,measure=True)
    compiled = lower_fixed_point(circuit)
    assert circuit.metadata["length_L"] == 13
    assert circuit.metadata["preparation_calls"] == 7
    assert circuit.metadata["inverse_preparation_calls"] == 6
    assert circuit.metadata["good_projector_phases"] == 6
    assert circuit.metadata["initial_zero_projector_phases"] == 6
    assert not circuit.metadata["physical_initialization_cost_included"]
    assert compiled.count_ops()["measure"] == 3
    for item in compiled.data:
        if item.operation.name == "measure":
            assert compiled.find_bit(item.qubits[0]).index == compiled.find_bit(item.clbits[0]).index
    assert preparation.num_clbits == 0


@pytest.mark.parametrize("lower,delta", [(0,.1),(-1,.1),(True,.1),(math.nan,.1),(.1,0),(.1,1),(.1,True),(.1,math.inf),(.1,1e-200)])
def test_invalid_schedule_probabilities(lower,delta):
    with pytest.raises(ValueError):
        fixed_point_schedule(lower,delta)


def test_corrupt_schedule_rejected_and_length_cap_is_independent():
    schedule = fixed_point_schedule(.1,.05)
    for update in ({"gamma":.9},{"alphas":()},{"betas":tuple(reversed(schedule.betas))},
                   {"length":12},{"success_lower_bound":1e-10}):
        with pytest.raises(ValueError):
            replace(schedule,**update)
    with pytest.raises(ValueError,match="bounded schedule"):
        fixed_point_schedule(1e-8,.05)
    with pytest.raises(ValueError):
        fixed_point_schedule(.1,.05,max_length=True)


@pytest.mark.parametrize("kind", ["measurement","reset","dense","initialize","opaque","unbound","nonfinite","nested_reset","nested_nonfinite","control"])
def test_noncoherent_or_unaccounted_preparation_rejected(kind):
    circuit = QuantumCircuit(1)
    if kind == "measurement":
        circuit.measure_all()
    elif kind == "reset":
        circuit.reset(0)
    elif kind == "dense":
        circuit.unitary(np.eye(2),[0])
    elif kind == "initialize":
        circuit.initialize([1,0],0)
    elif kind == "opaque":
        circuit.append(Gate("hidden",1,[]),[0])
    elif kind == "unbound":
        circuit.ry(Parameter("theta"),0)
    elif kind == "nonfinite":
        circuit.ry(float("nan"),0)
    elif kind.startswith("nested"):
        nested = QuantumCircuit(1)
        if kind == "nested_reset":
            nested.reset(0)
            circuit.append(nested.to_instruction(),[0])
        else:
            nested.ry(float("nan"),0)
            circuit.append(nested.to_gate(),[0])
    else:
        circuit = QuantumCircuit(1,1)
        with circuit.if_test((circuit.clbits[0],True)):
            circuit.x(0)
    with pytest.raises(ValueError):
        build_fixed_point_amplification(circuit,[0],1,.1)


@pytest.mark.parametrize("wires,pattern", [([],0),([0,0],1),([1],1),([True],0),([0],2),([0],True)])
def test_invalid_projector_masks(wires,pattern):
    with pytest.raises(ValueError):
        projector_phase(1,wires,pattern,.1)


def test_lowering_rejects_gates_after_readout_and_bad_source():
    circuit = QuantumCircuit(1,1)
    circuit.measure(0,0)
    circuit.x(0)
    with pytest.raises(ValueError,match="final measurements"):
        lower_fixed_point(circuit)
    for name in ("reset","unitary"):
        circuit = QuantumCircuit(1)
        if name == "reset":
            circuit.reset(0)
        else:
            circuit.unitary(np.eye(2),[0])
        with pytest.raises(ValueError):
            lower_fixed_point(circuit)


def test_simulator_and_source_caps_are_separate():
    with pytest.raises(ValueError,match="bounded simulator"):
        good_probability(QuantumCircuit(MAX_SIM_QUBITS+1),[0],1)
    with pytest.raises(ValueError,match="preflight"):
        build_fixed_point_amplification(QuantumCircuit(12),[0],1,.05,.05)
    with pytest.raises(ValueError,match="measure"):
        build_fixed_point_amplification(QuantumCircuit(1),[0],1,.1,measure=1)


def test_lowering_rejects_expansion_before_transpilation(monkeypatch):
    import openquantum.fixed_point_amplification as module
    def should_not_compile(*args,**kwargs):
        pytest.fail("Oversized multi-control expansion reached the transpiler")
    monkeypatch.setattr(module,"transpile",should_not_compile)
    circuit=QuantumCircuit(12)
    for _ in range(50):circuit.mcp(.1,list(range(11)),11)
    with pytest.raises(ValueError,match="expansion preflight"):
        lower_fixed_point(circuit)


def test_recursive_definition_has_an_explicit_depth_cap():
    circuit=QuantumCircuit(1)
    circuit.h(0)
    for level in range(70):
        gate=Gate(f"wrapper{level}",1,[])
        gate.definition=circuit
        circuit=QuantumCircuit(1)
        circuit.append(gate,[0])
    with pytest.raises(ValueError,match="depth cap"):
        build_fixed_point_amplification(circuit,[0],1,.1)


def test_generator_removes_stale_optional_reports_and_records_no_simulation(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    for name in ("physical-resources.json","estimated-circuit.qasm","physical-scaling.csv","physical-scaling.svg","scaling.csv"):
        (tmp_path/name).write_text("older evidence")
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/"scripts/generate_fixed_point_amplification.py"),
                    "--no-sim","--no-scaling","--output-dir",str(tmp_path)],
                   check=True,capture_output=True,text=True)
    for name in ("physical-resources.json","estimated-circuit.qasm","physical-scaling.csv","physical-scaling.svg","scaling.csv"):
        assert not (tmp_path/name).exists()
    logical=json.loads((tmp_path/"logical-resources.json").read_text())
    assert logical["simulation"]["performed"] is False
    assert logical["physical_projection"]["performed"] is False
    assert logical["parameters"]["exact_preparation_success"] == .25
