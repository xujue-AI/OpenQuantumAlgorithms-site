"""Independent probability, gate, sampling, confidence and guard checks."""
from fractions import Fraction
import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from openquantum.hadamard_qpe import (
    FringeCounts, MAX_CIRCUIT_OPERATIONS, actual_angle_phase, analytic_zero_probability,
    base_phase_angle, build_experiment, build_schedule, check_experiment,
    circular_distance, dense_experiment_reference, exp_lower_bound, ideal_failure_bound,
    lower_experiment, protocol_ledger, recover_phase, sample_counts, shots_for_failure,
    source_operation_count,
)
from openquantum.qft_qpe import dynamic_distribution


@pytest.mark.parametrize("n,phase,k", [(1,.3,1),(1,.995,32),(1,.125,8),(1,.5,2),(2,.17,8),(3,.07,4)])
@pytest.mark.parametrize("basis", ["cos", "sin"])
def test_full_operator_and_probability(n, phase, k, basis):
    result=check_experiment(n,phase,k,basis)
    assert result["full_operator_max_error"] < 2e-11
    assert result["source_to_lowered_operator_max_error"] < 2e-11
    assert result["probability_absolute_error"] < 2e-11
    circuit=build_experiment(n,phase,k,basis,measure=False,initialize=False)
    # An independent dense gate product is retained, including data preparation.
    np.testing.assert_allclose(Operator(circuit).data,dense_experiment_reference(n,phase,k,basis),atol=2e-11,rtol=0)


@pytest.mark.parametrize("basis", ["cos","sin"])
def test_measured_initial_reset_program(basis):
    circuit=build_experiment(1,.3,4,basis)
    assert circuit.count_ops()["reset"]==2
    assert circuit.count_ops()["measure"]==1
    probability=dynamic_distribution(lower_experiment(circuit))[0]
    assert abs(probability-analytic_zero_probability(1,.3,4,basis))<1e-12


def test_repeated_access_is_materialized_and_scalar_phase_retained():
    circuit=build_experiment(2,.2,8,"sin")
    counts=circuit.count_ops()
    assert counts["p"]==3*2*8
    assert counts["cx"]==2*2*8
    assert counts["sdg"]==1
    assert set(counts)=={"p","cx","h","x","sdg","reset","measure"}
    assert len(circuit.data)==source_operation_count(2,8,"sin")
    angles=[float(x.operation.params[0]) for x in circuit.data if x.operation.name=="p"]
    half=base_phase_angle(2,.2)/2
    assert set(angles)=={half,-half}
    assert actual_angle_phase(2,.2)==pytest.approx(.2,abs=1e-16)


def noiseless_counts(phase,levels,shots=1000000):
    # Independent analytic reference; not used by recovery to access true phase.
    return [FringeCounts(1<<j,b,int(round(shots*(1+(math.cos(2*math.pi*(1<<j)*phase) if b=="cos" else math.sin(2*math.pi*(1<<j)*phase)))/2)),shots)
            for j in range(levels) for b in ("cos","sin")]


@pytest.mark.parametrize("phase", [0,.001,.125,.3,.5,.75,.995,math.nextafter(1.0,0.0)])
def test_numerical_recovery_across_circular_and_ambiguous_fringe_cases(phase):
    counts=noiseless_counts(phase,6,shots=4096)
    result=recover_phase(counts)
    assert result["status"]=="recovered_numerically"
    assert 0<=result["phase_estimate"]<1
    assert circular_distance(result["phase_estimate"],phase)<1e-5
    assert result["numerical_inference_certified"] is False


def test_theoretical_good_event_lift_is_separate_from_sampling():
    # Construct proportion estimates whose corresponding angles stay within
    # the documented 1/8-cycle good-event radius, on both sides of the wrap.
    for phase in (.002,.3,.499,.998):
        for sign in (-1,1):
            records=[]
            for j in range(6):
                theta=2*math.pi*((1<<j)*phase + sign*.03)
                for b,signal in (("cos",math.cos(theta)),("sin",math.sin(theta))):
                    records.append(FringeCounts(1<<j,b,int(round(4096*(1+signal)/2)),4096))
            result=recover_phase(records)
            assert circular_distance(result["phase_estimate"],phase)<=float(Fraction(1,256))


def test_zero_signal_is_explicit_failure():
    records=[FringeCounts(1,b,1,2) for b in ("cos","sin")]
    result=recover_phase(records)
    assert result["status"]=="ambiguous_zero_signal"
    assert result["phase_estimate"] is None


def test_exact_confidence_planner_and_minimality():
    delta=Fraction(1,1200)
    m=shots_for_failure(6,delta)
    assert m==329
    bound=ideal_failure_bound(6,m)
    assert bound<=delta
    assert ideal_failure_bound(6,m-1)>delta
    assert 12*bound<=Fraction(1,100)
    # Exact rational reciprocal is conservative relative to an independently
    # evaluated high-precision real exp, not merely a rounded logarithm.
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec=350
        real=Decimal(24)*(-Decimal(m)/32).exp()
        rational=Decimal(bound.numerator)/Decimal(bound.denominator)
        assert rational>=real
        half=exp_lower_bound(Fraction(1,2))
        assert Decimal(half.numerator)/Decimal(half.denominator) < Decimal('.5').exp()


def test_seeded_counts_and_fresh_trial_accounting():
    m=shots_for_failure(6,Fraction(1,1200))
    probabilities=[analytic_zero_probability(1,.3,1<<j,b) for j in range(6) for b in ("cos","sin")]
    a=sample_counts(probabilities,6,m,7,stream=0)
    assert a==sample_counts(probabilities,6,m,7,stream=0)
    assert a!=sample_counts(probabilities,6,m,7,stream=1)
    assert circular_distance(recover_phase(a)["phase_estimate"],.3)<1/256
    ledger=protocol_ledger(2,6,m)
    M=2*6*m;calls=2*m*63
    assert ledger["complete_fresh_preparations"]==M
    assert ledger["shot_weighted_controlled_U_calls"]==calls
    assert ledger["maximum_controlled_U_power"]==32
    assert ledger["source_gate_counts_all_trials"]==dict(p=6*calls,cx=4*calls,h=2*M,x=2*M,sdg=6*m,reset=3*M,measure=M)
    assert ledger["physical_estimate_performed"] is False


@pytest.mark.parametrize("kwargs", [dict(target_qubits=True),dict(target_qubits=0),dict(target_qubits=33),dict(phase=float('nan')),dict(phase=1),dict(power=True),dict(power=0),dict(power=32769),dict(basis="X"),dict(measure=1),dict(initialize=1)])
def test_circuit_guard_rejections(kwargs):
    values=dict(target_qubits=1,phase=.3,power=1,basis="cos")
    values.update(kwargs)
    with pytest.raises(ValueError):build_experiment(**values)


def test_large_materialization_and_simulation_guards_separate():
    circuit=build_experiment(8,.3,32,"sin")
    assert circuit.num_qubits==9
    assert len(circuit.data)<MAX_CIRCUIT_OPERATIONS
    with pytest.raises(ValueError):check_experiment(8,.3,32,"sin")
    with pytest.raises(ValueError):build_experiment(32,.3,32768,"cos")
    with pytest.raises(ValueError):build_schedule(1,.3,16)
    with pytest.raises(ValueError):protocol_ledger(1,16,50000)


@pytest.mark.parametrize("records", [
    [FringeCounts(1,"cos",1,2)],
    [FringeCounts(1,"cos",1,2),FringeCounts(1,"cos",1,2)],
    [FringeCounts(2,"cos",1,2),FringeCounts(2,"sin",1,2)],
    [FringeCounts(1,"cos",1,2),FringeCounts(1,"sin",1,3)],
    [FringeCounts(1,"cos",True,2),FringeCounts(1,"sin",1,2)],
    [FringeCounts(1,"cos",3,2),FringeCounts(1,"sin",1,2)],
])
def test_malformed_recovery_inputs(records):
    with pytest.raises(ValueError):recover_phase(records)


@pytest.mark.parametrize("delta", [True,0,1,float('nan'),-1])
def test_failure_target_guards(delta):
    with pytest.raises(ValueError):shots_for_failure(6,delta)


def test_recovery_consumes_at_most_33_iterable_records():
    seen=[]
    def oversized():
        i=0
        while True:
            seen.append(i);i+=1
            yield FringeCounts(1,"cos",1,2)
    with pytest.raises(ValueError):recover_phase(oversized())
    assert len(seen)==33


def test_nonexact_subnormal_halving_and_underflow_rejected():
    with pytest.raises(ValueError):build_experiment(1,1e-323,1,"cos")
    with pytest.raises(ValueError):build_experiment(32,5e-324,1,"cos")
    assert base_phase_angle(1,0.0)==0
    assert 2*(base_phase_angle(1,5e-324)/2)==base_phase_angle(1,5e-324)
