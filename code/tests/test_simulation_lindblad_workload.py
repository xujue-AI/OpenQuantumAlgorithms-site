"""Uniform policy, exact sampling budgets and actual interacting-chain checks."""
from fractions import Fraction
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy.linalg import expm

from openquantum.simulation.lindblad_workload import (
    CASE_STATISTICAL_FAILURE, ChainTask, FAMILY, SPLITTING_ALLOWANCE,
    compare, full_circuit_density, gate_channel, gate_density,
    generator_norm_envelope, logical_ledger, measured_trial, on_wire,
    output_contract, reference_channel, reference_density, reference_generators,
    reference_initial, select_steps, uniform_split_bound,
)
from openquantum.simulation.two_spin_lindblad import build_chain_circuit, chain_schedule


def independent_wire(matrix, wire, n):
    result = np.array([[1]], complex)
    for j in reversed(range(n)):
        result = np.kron(result, matrix if j == wire else np.eye(2))
    return result


def random_density(n, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(2**n, 2**n))+1j*rng.normal(size=(2**n, 2**n))
    rho = matrix@matrix.conj().T
    return rho/np.trace(rho)


@pytest.mark.parametrize("n,steps,bound", [(2, 69, Fraction(17576,1785375)),
                                         (3,134,Fraction(177147,17956000)),
                                         (4,211,Fraction(1331,133563))])
def test_exact_uniform_policy_is_minimal_within_the_declared_envelope(n, steps, bound):
    task = ChainTask(n)
    assert select_steps(task) == steps
    assert uniform_split_bound(task, steps) == bound <= SPLITTING_ALLOWANCE
    assert uniform_split_bound(task, steps-1) > SPLITTING_ALLOWANCE
    assert generator_norm_envelope(task) == {2:Fraction(26,5),3:Fraction(81,10),4:Fraction(11)}[n]


def test_exact_statistical_allocation_is_familywise_and_has_constant_score_width():
    contract = output_contract()
    assert contract["complete_trials_per_case"] == 15756
    assert CASE_STATISTICAL_FAILURE*3 == Fraction(1,200)
    q = Fraction(15756)*Fraction(3,100)**2/2
    term, lower = Fraction(1), Fraction(1)
    for degree in range(1,20):
        term *= q/degree
        lower += term
    assert lower >= 1200  # exp(q)>=lower, hence2 exp(-q)<=1/600.
    previous = Fraction(15755)*Fraction(3,100)**2/2
    term, prefix = Fraction(1), Fraction(1)
    for degree in range(1,15):
        term *= previous/degree
        prefix += term
    next_term = term*previous/15
    upper = prefix+next_term/(1-previous/16)
    assert upper < 1200  # Previous N fails, without a floating ln/exp certificate.
    assert contract["ideal_angle_absolute_error_per_output"] == .04
    assert contract["computed_family_union_bound"] < .005
    assert contract["trial_ensemble_executed"] is False


@pytest.mark.parametrize("n", FAMILY)
def test_independent_liouvillian_column_vectorization_matches_direct_action(n):
    task = ChainTask(n)
    rho = random_density(n, 910+n)
    X = np.array([[0,1],[1,0]], complex)
    Z = np.diag([1,-1]).astype(complex)
    lower = np.array([[0,1],[0,0]], complex)
    H = .6*sum(independent_wire(Z,j,n)@independent_wire(Z,j+1,n) for j in range(n-1))
    H += .35*sum(independent_wire(X,j,n) for j in range(n))
    direct = -1j*(H@rho-rho@H)
    for j, rate in enumerate((.8,)+(.5,)*(n-1)):
        L = math.sqrt(rate)*independent_wire(lower,j,n)
        loss = L.conj().T@L
        direct += L@rho@L.conj().T-.5*(loss@rho+rho@loss)
    generators = reference_generators(task)
    matrix = sum(generators.values(),np.zeros_like(generators["ZZ"]))
    actual = (matrix@rho.reshape(4**n,order="F")).reshape((2**n,2**n),order="F")
    np.testing.assert_allclose(actual,direct,atol=2e-13,rtol=0)


@pytest.mark.parametrize("n", FAMILY)
def test_palindrome_matches_first_and_second_generator_derivatives(n):
    task = ChainTask(n)
    generators = reference_generators(task)
    schedule = chain_schedule(n,2)
    assert schedule == tuple(reversed(schedule))
    for name in generators:
        assert sum(Fraction(str(weight)) for component,weight in schedule if component==name) == 1
    zero = np.zeros_like(generators["ZZ"])
    coefficients = [np.eye(4**n,dtype=complex),zero.copy(),zero.copy()]
    for name, weight in schedule:
        A = weight*generators[name]
        prior = coefficients
        coefficients = [prior[0], A@prior[0]+prior[1], (A@A/2)@prior[0]+A@prior[1]+prior[2]]
    G = sum(generators.values(),zero)
    np.testing.assert_allclose(coefficients[1],G,atol=2e-13,rtol=0)
    np.testing.assert_allclose(coefficients[2],G@G/2,atol=2e-13,rtol=0)


@pytest.mark.parametrize("n", FAMILY)
@pytest.mark.parametrize("mixed", [False,True])
def test_gate_derived_causal_channel_matches_whole_circuit_with_fresh_resets(n,mixed):
    task = ChainTask(n)
    rho = random_density(n,420+n) if mixed else None
    contracted = gate_density(task,2,rho)
    whole = full_circuit_density(task,2,rho)
    np.testing.assert_allclose(contracted,whole,atol=2e-12,rtol=0)
    assert np.trace(contracted) == pytest.approx(1,abs=2e-12)
    assert np.linalg.eigvalsh(contracted).min() > -2e-12


@pytest.mark.parametrize("n", FAMILY)
def test_actual_selected_schedule_agrees_with_independent_reference_and_uniform_bound(n):
    task = ChainTask(n)
    steps = select_steps(task)
    channel = gate_channel(task,steps)
    exact_channel = reference_channel(task)
    for rho in (reference_initial(task),random_density(n,103+n),random_density(n,203+n)):
        actual = gate_density(task,steps,rho,channel)
        exact = reference_density(task,rho,exact_channel)
        norm = np.linalg.norm(actual-exact,ord="nuc")
        assert norm <= float(uniform_split_bound(task,steps))+2e-11
        Z0 = independent_wire(np.diag([1,-1]),0,n)
        assert abs(np.trace(Z0@(actual-exact))) <= norm+2e-12
    checked = compare(task,steps)
    assert checked["observed_trace_norm_error"] < .001
    assert checked["floating_interval_certificate"] is False
    assert checked["whole_2n_wire_density_simulation"] is False


@pytest.mark.parametrize("n", FAMILY)
def test_complete_measured_ledger_counts_every_reset_readout_and_trial(n):
    task = ChainTask(n)
    steps = select_steps(task)
    source, compiled = measured_trial(task,steps)
    ledger = logical_ledger(task,source,compiled)
    shots = ledger["complete_trials"]
    assert shots == 15756
    assert ledger["logical_qubits"] == 2*n
    assert ledger["primitive_counts_per_complete_trial"]["reset"] == n+(2*n-1)*steps
    assert ledger["primitive_counts_per_complete_trial"]["measure"] == 2*n
    for name,count in ledger["primitive_counts_per_complete_trial"].items():
        assert ledger["primitive_counts_all_trials"][name] == count*shots
    assert ledger["environment_discard_resets_before_reuse_per_trial"] == [2*steps-1]*(n-1)+[steps-1]
    assert "barrier" not in ledger["primitive_counts_per_complete_trial"]
    measured_bits = lambda c: [c.find_bit(item.clbits[0]).index for item in c.data if item.operation.name=="measure"]
    assert measured_bits(source) == measured_bits(compiled)
    assert ledger["actual_clifford_t_synthesis"] is False


def test_measured_Z0_score_matches_retained_system_observable():
    task = ChainTask(3)
    source,_ = measured_trial(task,2)
    unmeasured = source.remove_final_measurements(inplace=False)
    probabilities = DensityMatrix.from_instruction(unmeasured).probabilities()
    score = np.array([1 if not index&1 else -1 for index in range(len(probabilities))])
    measured = probabilities@score
    rho = gate_density(task,2)
    assert measured == pytest.approx(np.trace(on_wire(np.diag([1,-1]),0,3)@rho).real,abs=2e-12)


def test_ledger_rejects_an_incompatible_task_or_measurement_schedule():
    source, compiled = measured_trial(ChainTask(3),2)
    with pytest.raises(ValueError,match="declared"):
        logical_ledger(ChainTask(2),source,compiled)
    missing_readout = compiled.remove_final_measurements(inplace=False)
    with pytest.raises(ValueError,match="events"):
        logical_ledger(ChainTask(3),source,missing_readout)


def test_missing_environment_resets_changes_the_channel():
    task = ChainTask(3)
    source = build_chain_circuit(3,steps=2)
    wrong = source.copy_empty_like()
    seen = set()
    for item in source.data:
        if item.operation.name == "reset":
            wire = source.find_bit(item.qubits[0]).index
            if wire>=3 and wire in seen:
                continue
            seen.add(wire)
        wrong.append(item.operation,item.qubits,item.clbits)
    incorrect = partial_trace(DensityMatrix.from_instruction(wrong),[3,4,5]).data
    assert np.linalg.norm(incorrect-gate_density(task,2),ord="nuc") > .01


def test_bound_and_reference_accept_normalized_real_values_and_exact_limits():
    task = ChainTask(2,J=Fraction(3,5),hx=Fraction(7,20),time=Fraction(1))
    assert isinstance(task.J,float) and select_steps(task)==69
    np.testing.assert_allclose(reference_channel(task),reference_channel(ChainTask(2)),atol=2e-13)
    task32 = ChainTask(2,J=np.float32(.6))
    assert task32.J == float(np.float32(.6))
    for task in (ChainTask(2,time=0),ChainTask(3,J=0,hx=0,rate0=0,other_rate=0)):
        assert select_steps(task)==1
        assert uniform_split_bound(task,1)==0
        np.testing.assert_allclose(gate_density(task,1),reference_initial(task),atol=2e-12)


@pytest.mark.parametrize("kwargs", [{"n_system":1},{"n_system":5},{"n_system":True},
    {"n_system":2,"time":-1},{"n_system":2,"time":3},{"n_system":2,"J":2},
    {"n_system":2,"rate0":-.1},{"n_system":2,"hx":float("nan")},
    {"n_system":2,"time":True}])
def test_invalid_or_unbounded_problem_definitions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ChainTask(**kwargs)


def test_allocation_and_simulation_caps_fail_before_large_work():
    with pytest.raises(ValueError,match="1024"):
        select_steps(ChainTask(4,time=2,J=1,hx=1,rate0=2,other_rate=2))
    with pytest.raises(ValueError,match="four steps"):
        full_circuit_density(ChainTask(4),5)
    for steps in (0,True,1025):
        with pytest.raises(ValueError):
            gate_channel(ChainTask(2),steps)
    with pytest.raises(ValueError):
        on_wire(np.eye(3),0,2)


def generator_module():
    path = Path(__file__).resolve().parents[1]/"scripts/generate_simulation_lindblad_workload.py"
    spec = importlib.util.spec_from_file_location("lindblad_workload_generator_test",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_protects_original_artifacts_and_unchecked_canonical_output(tmp_path):
    generator = generator_module()
    with pytest.raises(ValueError,match="restricted"):
        generator.validate_output(generator.ROOT/"assets/generated/simulation/two-spin-lindblad",False)
    with pytest.raises(ValueError,match="separate"):
        generator.validate_output(generator.ROOT/"assets/generated/simulation/lindblad-workload",True)
    assert generator.validate_output(tmp_path,True)==tmp_path
