# SPDX-License-Identifier: MIT
from fractions import Fraction
import copy
import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator,Statevector
from scipy.stats import binom

from openquantum.iterative_amplitude_estimation import (
    AdaptiveAEConfig,build_amplified_trial,amplified_probability,run_threshold_adaptive,
    run_adaptive_amplitude,adaptive_logical_ledger,clopper_pearson,pi_rational_bounds,
    sin_squared_bounds,inverse_probability_bounds,safe_lobe,choose_power,invert_lobe,
    amplitude_interval,_binomial_tail_numerator,
)


@pytest.mark.parametrize('n,s,p',[(n,s,p) for n in range(1,7) for s in range(n+1) for p in (Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(1))])
def test_exact_binomial_tail_against_direct_rational_sum(n,s,p):
    num,den=_binomial_tail_numerator(n,s,p)
    direct=sum((math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(s,n+1)),Fraction(0))
    assert Fraction(num,den)==direct


def test_machin_pi_bounds_have_exact_tangent_identity():
    # tan(4atan(1/5)-atan(1/239))=1 exactly; branch is first quadrant.
    twice=2*Fraction(1,5)/(1-Fraction(1,5)**2)
    four=2*twice/(1-twice**2)
    assert (four-Fraction(1,239))/(1+four*Fraction(1,239))==1
    low,high=pi_rational_bounds()
    assert Fraction(3)<low<high<Fraction(22,7)
    assert high-low<Fraction(1,2**192)
    # Independent decimal digits bounded at50 places.
    known_low=Fraction('3.14159265358979323846264338327950288419716939937510')
    known_high=Fraction('3.14159265358979323846264338327950288419716939937511')
    assert known_low<low<high<known_high


@pytest.mark.parametrize('angle,expected',[(Fraction(0),Fraction(0)),(Fraction(1,6),Fraction(1,4)),(Fraction(1,4),Fraction(1,2)),(Fraction(1,3),Fraction(3,4)),(Fraction(1,2),Fraction(1))])
def test_certified_trigonometric_bounds_at_exact_angles(angle,expected):
    low,high=sin_squared_bounds(angle)
    assert low<=expected<=high
    assert high-low<Fraction(1,2**180)


@pytest.mark.parametrize('probability',[0.,1e-9,.0001,.01,.25,.5,.9,1-1e-9,1.])
def test_outward_inverse_certificates(probability):
    lo,hi=inverse_probability_bounds(probability)
    assert sin_squared_bounds(Fraction.from_float(lo))[1]<=Fraction.from_float(probability)
    assert sin_squared_bounds(Fraction.from_float(hi))[0]>=Fraction.from_float(probability)
    assert 0<=lo<=hi<=.5


@pytest.mark.parametrize('n',[1,2,5,10,32])
def test_clopper_pearson_exact_coverage_enumeration(n):
    alpha=.05
    intervals=[clopper_pearson(s,n,alpha) for s in range(n+1)]
    for probability in np.linspace(0,1,101):
        missing=sum(binom.pmf(s,n,probability) for s,interval in enumerate(intervals)
                    if probability<interval['lower'] or probability>interval['upper'])
        assert missing<=alpha+1e-14


def test_endpoint_cp_intervals_and_independent_exact_tail_certificates():
    shots=256;delta=.0005
    for s in (0,1,63,128,255,256):
        interval=clopper_pearson(s,shots,delta)
        if s:
            num,den=_binomial_tail_numerator(shots,s,Fraction.from_float(interval['lower']))
            assert Fraction(num,den)<=Fraction.from_float(delta)/2
        else:assert interval['lower']==0
        if s<shots:
            num,den=_binomial_tail_numerator(shots,shots-s,1-Fraction.from_float(interval['upper']))
            assert Fraction(num,den)<=Fraction.from_float(delta)/2
        else:assert interval['upper']==1


@pytest.mark.parametrize('interval,multiplier,lobe',[
    ((0.,.5),1,0),((.1,.2),3,None),((.16,.17),3,None),
    ((.2,.25),3,1),((0.,.05),7,0),((.45,.5),7,6),
    ((.125,.25),2,None),
])
def test_unwrapped_lobes_do_not_alias_crossings(interval,multiplier,lobe):
    if multiplier%2==0:
        with pytest.raises(ValueError):safe_lobe(interval,multiplier)
    else:assert safe_lobe(interval,multiplier)==lobe


def test_selector_is_largest_safe_odd_multiplier():
    interval=(.15,.18)
    chosen=choose_power(interval,32)
    assert safe_lobe(interval,chosen['multiplier'])==chosen['lobe']
    for power in range(chosen['power']+1,33):assert safe_lobe(interval,2*power+1) is None


@pytest.mark.parametrize('multiplier,lobe',[(1,0),(3,0),(3,1),(3,2),(5,3),(9,7)])
def test_inverse_branches_preserve_every_interior_probability(multiplier,lobe):
    lower,upper=invert_lobe((.1,.9),multiplier,lobe)
    for p in np.linspace(.1,.9,17):
        u=math.asin(math.sqrt(p))/math.pi
        reference=(lobe/2+u)/multiplier if lobe%2==0 else ((lobe+1)/2-u)/multiplier
        assert lower<=reference<=upper
        assert lower>=0 and upper<=.5


@pytest.mark.parametrize('n,k,power',[(n,k,power) for n,k in [(1,0),(1,1),(1,2),(2,1),(2,3),(3,2)] for power in (0,1,2,4,8)])
def test_actual_amplified_trial_matches_independent_rotation_law(n,k,power):
    circuit=build_amplified_trial(n,k,power,measure=False)
    actual=Statevector.from_instruction(circuit).probabilities(qargs=[n])[1]
    assert abs(actual-amplified_probability(k/2**n,power))<1e-11
    measured=build_amplified_trial(n,k,power)
    assert measured.count_ops()['measure']==1


def test_default_actual_trajectory_replays_every_bit_and_counts_whole_workload():
    result=run_threshold_adaptive(2,1)
    again=run_threshold_adaptive(2,1)
    assert result==again
    assert result['status']=='conclusive'
    assert [r['power'] for r in result['history']]==[0,0,0,2]
    assert result['total_shots']==1024 and result['total_Q_calls']==512
    assert result['total_threshold_predicate_or_inverse_calls']==2048
    rng=np.random.default_rng(result['seed'])
    for row in result['history']:
        assert rng.bit_generator.state==row['rng_before']
        outcomes=rng.binomial(1,row['sampling_probability'],size=row['shots'])
        assert outcomes.tolist()==row['shot_bitstrings']
        assert int(outcomes.sum())==row['successes']
        assert rng.bit_generator.state==row['rng_after']
        assert row['after_angle_interval'][0]>=row['before_angle_interval'][0]
        assert row['after_angle_interval'][1]<=row['before_angle_interval'][1]
    interval=result['final_amplitude_interval']
    assert interval['lower']<=.25<=interval['upper']
    assert Fraction(interval['radius_fraction'])<=Fraction('.02')
    ledger=adaptive_logical_ledger(2,1,result)
    assert ledger['total_primitive_operations']==46080
    assert ledger['total_shots']==1024
    broken=copy.deepcopy(result);broken['total_Q_calls']+=1
    with pytest.raises(ValueError):adaptive_logical_ledger(2,1,broken)
    with pytest.raises(ValueError):adaptive_logical_ledger(2,2,result)
    with pytest.raises(ValueError):adaptive_logical_ledger(2,1,run_adaptive_amplitude(.25))


@pytest.mark.parametrize('amplitude',[0.,1e-6,.001,.24999,.25,.25001,.5,.9,1-1e-6,1.])
def test_endpoints_and_near_alias_trajectories(amplitude):
    result=run_adaptive_amplitude(amplitude,seed=314159)
    assert result['status'] in {'conclusive','inconclusive_round_cap'}
    assert all(safe_lobe(row['before_angle_interval'],row['multiplier'])==row['lobe'] for row in result['history'])
    if result['status']=='conclusive':
        assert abs(result['final_amplitude_interval']['estimate']-amplitude)<=.02


def test_adaptive_allocation_caps_and_inconclusive_are_not_success():
    config=AdaptiveAEConfig(absolute_error=.00001,shots_per_round=16,max_rounds=2,max_power=0,max_total_shots=32)
    result=run_adaptive_amplitude(.3,config,seed=12)
    assert result['status']=='inconclusive_round_cap'
    assert Fraction(result['confidence_contract']['exact_total_allocation_fraction'])<=Fraction(str(config.failure_probability))
    shotcap=run_adaptive_amplitude(.3,AdaptiveAEConfig(max_total_shots=1))
    assert shotcap['status']=='inconclusive_shot_cap' and shotcap['total_shots']==0
    simcap=run_threshold_adaptive(9,1)
    assert simcap['status']=='inconclusive_simulation_cap' and simcap['cap_reason']


@pytest.mark.parametrize('kwargs',[{'absolute_error':0},{'failure_probability':0},{'max_rounds':0},{'shots_per_round':True},{'max_power':-1},{'max_total_shots':0}])
def test_invalid_protocol_parameters(kwargs):
    with pytest.raises(ValueError):AdaptiveAEConfig(**kwargs)


@pytest.mark.parametrize('interval',[(-.1,.2),(.2,.1),(.2,.6),(float('nan'),.3)])
def test_invalid_alias_intervals(interval):
    with pytest.raises(ValueError):choose_power(interval,64)


@pytest.mark.parametrize('multiplier',[1,3,129])
def test_final_endpoint_uses_the_inclusive_last_lobe(multiplier):
    assert safe_lobe((.5,.5),multiplier)==multiplier-1
    lower,upper=invert_lobe((1.,1.),multiplier,multiplier-1)
    assert lower<=.5<=upper


@pytest.mark.parametrize("epsilon",[.05,.2])
def test_exact_small_adaptive_tree_covers_all_counts_not_just_rng_paths(epsilon):
    # Enumerate ALL binomial count paths. Exact Chebyshev polynomials give each
    # adapted amplified Bernoulli law at a=3/10 without trigonometric floats.
    config=AdaptiveAEConfig(absolute_error=epsilon,failure_probability=.2,
                           shots_per_round=8,max_rounds=3,max_power=4,max_total_shots=24)
    amplitude=Fraction(3,10)
    missing=wrong_conclusive=terminal_mass=Fraction(0)
    leaves=0
    multipliers=set()
    def exact_probability(multiplier):
        c=1-2*amplitude
        previous,current=Fraction(1),c
        for _ in range(1,multiplier):previous,current=current,2*c*current-previous
        return (1-current)/2
    def visit(interval,round_index,weight,missed):
        nonlocal missing,wrong_conclusive,terminal_mass,leaves
        summary=amplitude_interval(interval)
        conclusive=Fraction(summary['radius_fraction'])<=Fraction(str(config.absolute_error))
        if conclusive or round_index==config.max_rounds:
            terminal_mass+=weight;leaves+=1
            if missed:missing+=weight
            if conclusive and abs(Fraction.from_float(summary['estimate'])-amplitude)>Fraction(str(config.absolute_error)):
                wrong_conclusive+=weight
            return
        chosen=choose_power(interval,config.max_power)
        multipliers.add(chosen['multiplier'])
        p=exact_probability(chosen['multiplier'])
        for successes in range(config.shots_per_round+1):
            probability=math.comb(config.shots_per_round,successes)*p**successes*(1-p)**(config.shots_per_round-successes)
            cp=clopper_pearson(successes,config.shots_per_round,config.batch_failure_probability)
            now_missed=missed or p<Fraction.from_float(cp['lower']) or p>Fraction.from_float(cp['upper'])
            candidate=invert_lobe((cp['lower'],cp['upper']),chosen['multiplier'],chosen['lobe'])
            updated=max(interval[0],candidate[0]),min(interval[1],candidate[1])
            if updated[0]>updated[1]:
                terminal_mass+=weight*probability;leaves+=1
                assert now_missed
                missing+=weight*probability
            else:visit(updated,round_index+1,weight*probability,now_missed)
    visit((0.,.5),0,Fraction(1),False)
    assert leaves>1 and terminal_mass==1
    assert wrong_conclusive<=missing<=Fraction(str(config.failure_probability))
    if epsilon==.05:assert max(multipliers)>1
    else:assert wrong_conclusive>0
