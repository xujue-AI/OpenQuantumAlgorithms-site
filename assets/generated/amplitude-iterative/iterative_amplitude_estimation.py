# SPDX-License-Identifier: MIT
"""Bounded IQAE-style branch-adaptive amplitude estimation.

This is a distinct fixed-batch variant inspired by Grinko et al., not their
exact algorithm or convergence theorem. Every fresh batch has a predetermined
failure allocation. CP endpoints, alias tests, trigonometric inversion and the
returned error radius have conservative exact-rational certificates. The ideal
statistical theorem assumes independent exact-law shots; finite PRNG/statevector
experiments and QDK resource projections do not certify hardware confidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.stats import beta

from openquantum.amplitude_estimation import threshold_preparation, threshold_grover_iterate
from openquantum.qft_qpe import lower_circuit

MAX_SOURCE_OPERATIONS = 250_000
MAX_GENERATOR_POWER = 4096
MAX_SIM_POWER = 128
MAX_SIM_DATA = 8
PRECISION_BITS = 192
_SCALE = 1 << PRECISION_BITS
_COS_TERMS = 40
assert 4**82 * _SCALE < math.factorial(82)


def _integer(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def _floor(f: Fraction) -> int:
    return f.numerator // f.denominator


def _ceil(f: Fraction) -> int:
    return -((-f.numerator) // f.denominator)


def _outward_float(value: Fraction, *, upper: bool) -> float:
    result = float(value)
    if upper and Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    elif not upper and Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _atan_bounds(x: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    total = sum(((-1)**k * x**(2*k+1) / (2*k+1) for k in range(terms)), Fraction(0))
    next_term = (-1)**terms * x**(2*terms+1) / (2*terms+1)
    return min(total, total+next_term), max(total, total+next_term)


@lru_cache(maxsize=1)
def pi_rational_bounds() -> tuple[Fraction, Fraction]:
    """Machin identity and alternating arctan tails, no floating π assumption."""
    a_low,a_high = _atan_bounds(Fraction(1,5),48)
    b_low,b_high = _atan_bounds(Fraction(1,239),14)
    return 16*a_low-4*b_high, 16*a_high-4*b_low


def _mul_interval(left, right):
    products = [a*b for a in left for b in right]
    return min(products)//_SCALE, -((-max(products))//_SCALE)


def _divide_interval(interval, denominator):
    return interval[0]//denominator, -((-interval[1])//denominator)


@lru_cache(maxsize=8192)
def sin_squared_bounds(angle_fraction: Fraction) -> tuple[Fraction, Fraction]:
    """Certified sin²(πt), 0<=t<=1/2, using fixed-point interval cos(2πt).

    Each multiply/divide is outward rounded with Python integers. The degree80
    cosine polynomial has remainder <=4**82/82! <2**-192, since |2πt|<4.
    This is an executed arithmetic bound, not a Lean/Rocq proof artifact.
    """
    t = Fraction(angle_fraction)
    if not 0 <= t <= Fraction(1,2):
        raise ValueError("Normalized angle must lie in [0,1/2]")
    if t == 0: return Fraction(0),Fraction(0)
    if t == Fraction(1,2): return Fraction(1),Fraction(1)
    p_low,p_high = pi_rational_bounds()
    x = _floor(2*t*p_low*_SCALE), _ceil(2*t*p_high*_SCALE)
    if x[1] >= 4*_SCALE:
        raise AssertionError("Trigonometric remainder domain exceeded")
    square = _mul_interval(x,x)
    term = (_SCALE,_SCALE)
    total = [_SCALE,_SCALE]
    for k in range(1,_COS_TERMS+1):
        term = _divide_interval(_mul_interval(term,square),(2*k-1)*(2*k))
        if k%2:
            total[0] -= term[1];total[1] -= term[0]
        else:
            total[0] += term[0];total[1] += term[1]
    cos_low,cos_high = total[0]-1,total[1]+1
    return max(Fraction(0),Fraction(_SCALE-cos_high,2*_SCALE)), min(Fraction(1),Fraction(_SCALE-cos_low,2*_SCALE))


def _binomial_tail_numerator(n: int, successes: int, probability: Fraction) -> tuple[int,int]:
    """Exact P(Bin(n,p)>=successes) numerator/denominator at rational p."""
    p = Fraction(probability)
    if not 0<=p<=1: raise ValueError("Probability must lie in [0,1]")
    if successes<=0:return 1,1
    if successes>n:return 0,1
    if p==0:return 0,1
    if p==1:return 1,1
    a,d = p.numerator,p.denominator
    b = d-a
    term = math.comb(n,successes)*a**successes*b**(n-successes)
    numerator = term
    for j in range(successes,n):
        term = term*(n-j)*a//((j+1)*b)
        numerator += term
    return numerator,d**n


def _tail_at_most(n, successes, p, allowance):
    numerator,denominator = _binomial_tail_numerator(n,successes,p)
    return numerator*allowance.denominator <= allowance.numerator*denominator


@lru_cache(maxsize=8192)
def clopper_pearson(successes: int, shots: int, failure_probability: float) -> dict:
    """Conservative CP interval with exact rational binomial-tail certificates.

    SciPy's beta inverse is only an initial guess. Endpoints move outward until
    their exact tail inequalities are verified. If64 ULP steps fail, the bound
    falls back to0/1 conservatively. No equality-coverage claim is made.
    """
    _integer("shots",shots,1,4096);_integer("successes",successes,0,shots)
    if not math.isfinite(failure_probability) or not 0<failure_probability<1:
        raise ValueError("Failure probability must lie in (0,1)")
    alpha = Fraction.from_float(failure_probability)/2
    lower = 0. if successes==0 else float(beta.ppf(float(alpha),successes,shots-successes+1))
    upper = 1. if successes==shots else float(beta.ppf(float(1-alpha),successes+1,shots-successes))
    lower_steps=upper_steps=0
    if successes:
        if not math.isfinite(lower):lower=0.
        lower=max(0.,min(1.,lower))
        while not _tail_at_most(shots,successes,Fraction.from_float(lower),alpha):
            lower=math.nextafter(lower,0.);lower_steps+=1
            if lower_steps>=64:lower=0.;break
    if successes<shots:
        if not math.isfinite(upper):upper=1.
        upper=max(0.,min(1.,upper))
        while not _tail_at_most(shots,shots-successes,1-Fraction.from_float(upper),alpha):
            upper=math.nextafter(upper,1.);upper_steps+=1
            if upper_steps>=64:upper=1.;break
    assert successes==0 or _tail_at_most(shots,successes,Fraction.from_float(lower),alpha)
    assert successes==shots or _tail_at_most(shots,shots-successes,1-Fraction.from_float(upper),alpha)
    return {"lower":lower,"upper":upper,"failure_probability":failure_probability,
            "lower_outward_steps":lower_steps,"upper_outward_steps":upper_steps,
            "certificate":"exact rational binomial tail inequalities; beta inverse is a seed only"}


@lru_cache(maxsize=8192)
def inverse_probability_bounds(probability: float) -> tuple[float,float]:
    """Outward certified bounds for asin(sqrt(p))/π in [0,1/2]."""
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError("Probability must lie in [0,1]")
    if probability==0:return 0.,0.
    if probability==1:return .5,.5
    target=Fraction.from_float(probability)
    seed=math.asin(math.sqrt(probability))/math.pi
    lower=upper=max(0.,min(.5,seed))
    for _ in range(64):
        if sin_squared_bounds(Fraction.from_float(lower))[1] <= target:break
        lower=math.nextafter(lower,0.)
    else:lower=0.
    for _ in range(64):
        if sin_squared_bounds(Fraction.from_float(upper))[0] >= target:break
        upper=math.nextafter(upper,.5)
    else:upper=.5
    assert sin_squared_bounds(Fraction.from_float(lower))[1]<=target
    assert sin_squared_bounds(Fraction.from_float(upper))[0]>=target
    return lower,upper


def _angle_interval(interval):
    if len(interval)!=2 or any(not math.isfinite(v) for v in interval) or not 0<=interval[0]<=interval[1]<=.5:
        raise ValueError("Angle interval must be ordered in [0,1/2]")
    return Fraction.from_float(float(interval[0])),Fraction.from_float(float(interval[1]))


def safe_lobe(interval, multiplier: int) -> int | None:
    """Unwrapped exact-rational test, not endpoints reduced modulo a period."""
    _integer("multiplier",multiplier,1,2*MAX_GENERATOR_POWER+1)
    if multiplier%2!=1:raise ValueError("Multiplier must be odd")
    lower,upper=_angle_interval(interval)
    low,high=multiplier*lower,multiplier*upper
    lobe=min(_floor(2*low),multiplier-1)
    return lobe if low>=Fraction(lobe,2) and high<=Fraction(lobe+1,2) else None


def choose_power(interval, max_power: int) -> dict:
    _integer("max_power",max_power,0,MAX_GENERATOR_POWER)
    for power in range(max_power,-1,-1):
        multiplier=2*power+1
        lobe=safe_lobe(interval,multiplier)
        if lobe is not None:return {"power":power,"multiplier":multiplier,"lobe":lobe,"increasing":lobe%2==0}
    raise AssertionError("Multiplier one must cover a subinterval of [0,1/2]")


def invert_lobe(probability_interval, multiplier: int, lobe: int) -> tuple[float,float]:
    lo,hi=probability_interval
    if not 0<=lo<=hi<=1:raise ValueError("Probability interval must be ordered in [0,1]")
    _integer("lobe",lobe,0,multiplier-1)
    _integer("multiplier",multiplier,1,2*MAX_GENERATOR_POWER+1)
    if multiplier%2!=1:raise ValueError("Multiplier must be odd")
    lower_u=inverse_probability_bounds(float(lo))[0]
    upper_u=inverse_probability_bounds(float(hi))[1]
    if lobe%2==0:
        lower=(Fraction(lobe,2)+Fraction.from_float(lower_u))/multiplier
        upper=(Fraction(lobe,2)+Fraction.from_float(upper_u))/multiplier
    else:
        lower=(Fraction(lobe+1,2)-Fraction.from_float(upper_u))/multiplier
        upper=(Fraction(lobe+1,2)-Fraction.from_float(lower_u))/multiplier
    return _outward_float(lower,upper=False),_outward_float(upper,upper=True)


def amplitude_interval(angle_interval) -> dict:
    lower,upper=_angle_interval(angle_interval)
    a_lower=sin_squared_bounds(lower)[0]
    a_upper=sin_squared_bounds(upper)[1]
    lo=_outward_float(a_lower,upper=False);hi=_outward_float(a_upper,upper=True)
    estimate=float((Fraction.from_float(lo)+Fraction.from_float(hi))/2)
    radius=max(Fraction.from_float(estimate)-Fraction.from_float(lo),Fraction.from_float(hi)-Fraction.from_float(estimate))
    return {"lower":lo,"upper":hi,"estimate":estimate,"radius":_outward_float(radius,upper=True),
            "radius_fraction":str(radius),"arithmetic_certificate":"outward rational sin² enclosure and radius about the reported float"}


@dataclass(frozen=True)
class AdaptiveAEConfig:
    absolute_error: float = .02
    failure_probability: float = .01
    shots_per_round: int = 256
    max_rounds: int = 20
    max_power: int = 64
    max_total_shots: int = 8192

    def __post_init__(self):
        if not math.isfinite(self.absolute_error) or not 1e-6<=self.absolute_error<.5:
            raise ValueError("Absolute error must lie in [1e-6,.5)")
        if not math.isfinite(self.failure_probability) or not 1e-9<=self.failure_probability<1:
            raise ValueError("Failure probability must lie in [1e-9,1)")
        _integer("shots_per_round",self.shots_per_round,1,4096)
        _integer("max_rounds",self.max_rounds,1,64)
        _integer("max_power",self.max_power,0,MAX_GENERATOR_POWER)
        _integer("max_total_shots",self.max_total_shots,1,262144)

    @property
    def batch_failure_probability(self):
        target=Fraction(str(self.failure_probability))/self.max_rounds
        return _outward_float(target,upper=False)


def build_amplified_trial(n_data: int, threshold: int, power: int, *, measure: bool = True) -> QuantumCircuit:
    _integer("power",power,0,MAX_GENERATOR_POWER)
    preparation=threshold_preparation(n_data,threshold)
    iterate=threshold_grover_iterate(n_data,threshold)
    if len(preparation.data)+power*len(iterate.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("Repeated amplified trial exceeds source operation cap; MCX counts atomically")
    circuit=QuantumCircuit(n_data+1,1 if measure else 0,name=f"Q^{power} A")
    circuit.compose(preparation,inplace=True)
    for _ in range(power):circuit.compose(iterate,inplace=True)
    if measure:circuit.measure(n_data,0)
    return circuit


def amplified_probability(amplitude: float, power: int) -> float:
    if not math.isfinite(amplitude) or not 0<=amplitude<=1:
        raise ValueError("Amplitude must lie in [0,1]")
    _integer("power",power,0,MAX_GENERATOR_POWER)
    if amplitude==0:return 0.
    if amplitude==1:return 1.
    return math.sin((2*power+1)*math.asin(math.sqrt(amplitude)))**2


def _threshold_probability(n_data,threshold,power):
    if n_data>MAX_SIM_DATA or power>MAX_SIM_POWER:
        raise ValueError("Bounded actual trial simulation is capped at8 data wires and power128")
    circuit=build_amplified_trial(n_data,threshold,power,measure=False)
    probability=float(Statevector.from_instruction(circuit).probabilities(qargs=[n_data])[1])
    reference=amplified_probability(threshold/2**n_data,power)
    if not math.isclose(probability,reference,abs_tol=2e-11,rel_tol=0):
        raise AssertionError("Actual amplified trial disagrees with independent two-dimensional law")
    return max(0.,min(1.,probability)),reference


def _run(amplitude,config,seed,provider: Callable, *, probability_mode, record_shots):
    _integer("seed",seed,0,2**64-1)
    rng=np.random.default_rng(seed)
    interval=(0.,.5)
    history=[];total_shots=total_q=0;status="inconclusive_round_cap";cap_reason=None
    epsilon=Fraction(str(config.absolute_error))
    for round_index in range(config.max_rounds):
        summary=amplitude_interval(interval)
        if Fraction(summary['radius_fraction'])<=epsilon:status="conclusive";break
        if total_shots+config.shots_per_round>config.max_total_shots:
            status="inconclusive_shot_cap";break
        chosen=choose_power(interval,config.max_power)
        try:probability,reference=provider(chosen['power'])
        except ValueError as error:
            status="inconclusive_simulation_cap";cap_reason=str(error);break
        before=rng.bit_generator.state
        outcomes=rng.binomial(1,probability,size=config.shots_per_round)
        after=rng.bit_generator.state
        successes=int(outcomes.sum())
        confidence=clopper_pearson(successes,config.shots_per_round,config.batch_failure_probability)
        candidate=invert_lobe((confidence['lower'],confidence['upper']),chosen['multiplier'],chosen['lobe'])
        updated=(max(interval[0],candidate[0]),min(interval[1],candidate[1]))
        row={"round":round_index+1,**chosen,"shots":config.shots_per_round,"successes":successes,
             "sampling_probability":probability,"analytic_probability_check":reference,
             "probability_mode":probability_mode,"before_angle_interval":list(interval),
             "probability_confidence_interval":confidence,"candidate_angle_interval":list(candidate),
             "after_angle_interval":list(updated),"rng_before":before,"rng_after":after,
             "rng_algorithm":rng.bit_generator.__class__.__name__,
             "shot_bitstrings":outcomes.tolist() if record_shots else None}
        total_shots+=config.shots_per_round;total_q+=chosen['power']*config.shots_per_round
        if updated[0]>updated[1]:
            row['amplitude_interval']=None;history.append(row);status="contradiction";break
        interval=updated;row['amplitude_interval']=amplitude_interval(interval);history.append(row)
    final=None if status=="contradiction" else amplitude_interval(interval)
    if final is not None and Fraction(final['radius_fraction'])<=epsilon:status="conclusive"
    return {"algorithm":"fixed-batch branch-adaptive amplitude estimation inspired by IQAE",
       "status":status,"cap_reason":cap_reason,"config":asdict(config),"seed":seed,"probability_mode":probability_mode,
       "known_reference_amplitude":amplitude,"angle_convention":"t=asin(sqrt(a))/pi in[0,1/2]",
       "history":history,"final_angle_interval":list(interval) if final else None,"final_amplitude_interval":final,
       "total_shots":total_shots,"total_Q_calls":total_q,"total_A_or_A_inverse_calls":total_shots+2*total_q,
       "total_threshold_predicate_or_inverse_calls":total_shots+2*total_q,"total_flag_measurements":total_shots,
       "confidence_contract":{"batch_failure_probability":config.batch_failure_probability,
          "predetermined_batches":config.max_rounds,"allocated_failure_upper_bound":_outward_float(Fraction.from_float(config.batch_failure_probability)*config.max_rounds,upper=True),
          "exact_total_allocation_fraction":str(Fraction.from_float(config.batch_failure_probability)*config.max_rounds),
          "claim":"P(conclusive AND |reported estimate-a|>absolute_error) <= delta in the iid exact amplified-law model",
          "not_claimed":"coverage conditional on termination, convergence at caps, physical or PRNG confidence certificate",
          "arithmetic":"exact-rational CP tails, single-lobe checks, outward trigonometric/radius bounds",
          "sampling_assumptions":"fresh independent Bernoulli shots; finite PRNG and numerical Statevector experiment not formally certified"}}


def run_threshold_adaptive(n_data:int,threshold:int,config:AdaptiveAEConfig=AdaptiveAEConfig(),*,seed:int=314159,record_shots:bool=True)->dict:
    threshold_preparation(n_data,threshold)
    cache={}
    def provider(power):
        if power not in cache:cache[power]=_threshold_probability(n_data,threshold,power)
        return cache[power]
    result=_run(threshold/2**n_data,config,seed,provider,probability_mode="bounded actual Statevector amplified trials",record_shots=record_shots)
    result['input']={"n_data":n_data,"threshold":threshold,"known_classical_answer":threshold/2**n_data}
    return result


def run_adaptive_amplitude(amplitude:float,config:AdaptiveAEConfig=AdaptiveAEConfig(),*,seed:int=314159,record_shots:bool=False)->dict:
    """Law-only categorical coverage experiment; no actual circuit execution."""
    amplified_probability(amplitude,0)
    return _run(amplitude,config,seed,lambda k:(amplified_probability(amplitude,k),amplified_probability(amplitude,k)),
                probability_mode="analytic law only; no circuit constructed or simulated",record_shots=record_shots)


def adaptive_logical_ledger(n_data:int,threshold:int,result:dict)->dict:
    expected={'n_data':n_data,'threshold':threshold,'known_classical_answer':threshold/2**n_data}
    if result.get('input') != expected or result.get('probability_mode') != 'bounded actual Statevector amplified trials':
        raise ValueError("Ledger requires the matching actual threshold trajectory")
    groups={}
    for row in result['history']:
        _integer('power',row['power'],0,MAX_GENERATOR_POWER)
        _integer('shots',row['shots'],1,4096)
        power=row['power'];groups[power]=groups.get(power,0)+row['shots']
    total_shots=sum(groups.values())
    total_q=sum(power*shots for power,shots in groups.items())
    if (result['total_shots']!=total_shots or result['total_Q_calls']!=total_q
        or result['total_threshold_predicate_or_inverse_calls']!=total_shots+2*total_q
        or result['total_flag_measurements']!=total_shots):
        raise ValueError("Trajectory totals do not match every selected batch")
    circuits=[];total_operations=total_depth=0
    for power,shots in sorted(groups.items()):
        measured=build_amplified_trial(n_data,threshold,power)
        compiled=lower_circuit(measured)
        if len(compiled.data)>250000:raise ValueError("Primitive trial exceeds250000-operation export cap")
        operations=len(compiled.data);depth=compiled.depth()
        total_operations+=operations*shots;total_depth+=depth*shots
        circuits.append({'power':power,'multiplier':2*power+1,'shots':shots,'logical_qubits':compiled.num_qubits,
          'decomposed_gate_counts':dict(compiled.count_ops()),'operations_per_shot':operations,'dependency_depth_per_shot':depth,
          'all_shot_operations':operations*shots,'all_shot_dependency_depth':depth*shots,
          'A_calls_per_shot':power+1,'A_inverse_calls_per_shot':power,'predicate_or_inverse_calls_per_shot':2*power+1,
          'zero_reflections_per_shot':power,'good_reflections_per_shot':power,'measurements_per_shot':1})
    return {'status':'recorded ideal adaptive trajectory; all selected batches charged','circuits':circuits,
       'logical_qubits':n_data+1,'total_shots':result['total_shots'],'total_Q_calls':result['total_Q_calls'],
       'total_threshold_predicate_or_inverse_calls':result['total_threshold_predicate_or_inverse_calls'],
       'total_flag_measurements':result['total_flag_measurements'],'total_primitive_operations':total_operations,
       'serial_dependency_depth_sum':total_depth,'depth_semantics':'serial per-shot circuit dependencies; not physical time',
       'excluded':['classical adaptive inference and compilation timing','hardware initialization faults and device execution',
                   'actual synthesis certificate','application-specific input loading'],
       'arithmetic_certificates_are_not_circuit_or_hardware_proofs':True}
