# SPDX-License-Identifier: MIT
"""Interval reciprocal QSVT with explicit Hermitian Pauli access.

Original regularized geometric polynomial p_K(x)=c*x*sum_(j=0..K)(1-x*x)^j,
c=1/ceil(sqrt(K+1)), on a promised signed normalized spectrum |x|>=delta.
The whole-interval bound and conditional-state bounds are exact rational
facts. Phase fitting is bounded deterministic SciPy numerical preprocessing.
Actual Qiskit PREPARE/SELECT/unprepare, projector phases, preparation and
readout construct every circuit; no dense matrix/inverse constructs a gate.
QSVT: Gilyen et al. (2019), arXiv:1806.01838, doi:10.1145/3313276.3316366.
Qiskit Apache-2.0; NumPy/SciPy BSD-3-Clause; original module MIT.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import least_squares

from .linear_inverse_lcu import _rational, _upper_float, _lower_float
from .linear_systems_qsvt import _preparation
from .reciprocal_phase_certificate import certify_real_qsp_phases
from .simulation.qsp import build_real_qsp_filter, qsp_real_response

MAX_K = 31  # odd degree 2K+1 <= the shared QSP degree cap64
MAX_SYSTEM_QUBITS = 24
MAX_INPUT_TERMS = 48
MAX_REFERENCE_QUBITS = 5
MAX_STATEVECTOR_QUBITS = 12
MAX_SOURCE_OPERATIONS = 100_000
MAX_COMPILED_OPERATIONS = 1_000_000
MAX_RETRY_CAP = 100_000
MAX_TOTAL_ATTEMPTS = 2_000_000_000
DEFAULT_K = 16
DEFAULT_DELTA = Fraction(1, 2)
DEFAULT_TERMS = ((15, "II"), (-2, "IX"), (-2, "ZI"), (-1, "XX"))
DEFAULT_PHASES = (
    -1.6395619459694395, 1.6335626183777725, 1.5355031342762253,
    1.596708228777976, 1.5985898926997628, 1.5065219841214326,
    1.6304065445578757, 1.5165116460519055, 1.6001924938894185,
    1.4997142688452587, 1.5917565973064514, 1.598102594173029,
    1.5065299552207165, 1.7101855201980354, 1.4384868926695997,
    1.6451588017010965, 1.3846320127937268, 1.401717840356853,
    1.7539639881273055, 1.5354009626340417, 1.5312925160975923,
    1.5815870986486347, 1.5688391836317805, 1.5391706853393634,
    1.6458488706356553, 1.540105067349548, 1.6254431582937217,
    1.5110999292306937, 1.6350877326623943, 1.5430000288516401,
    1.5448847634710938, 1.6060894886996189, 1.5080300370075084,
    -3.0728270344666173,
)


def _index(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= MAX_K:
        raise ValueError(f"K must be an integer between0 and{MAX_K}")
    return k


def _gap(delta) -> Fraction:
    result = _rational(delta)
    if not 0 < result <= 1 or max(result.numerator.bit_length(), result.denominator.bit_length()) > 128:
        raise ValueError("promised normalized gap must be in(0,1] and at most128bits")
    return result


@dataclass(frozen=True)
class ReciprocalPolynomial:
    k: int = DEFAULT_K
    delta: Fraction = DEFAULT_DELTA

    def __post_init__(self):
        _index(self.k)
        object.__setattr__(self, "delta", _gap(self.delta))

    @property
    def m(self): return self.k + 1

    @property
    def degree(self): return 2 * self.k + 1

    @property
    def scale(self):
        root = math.isqrt(self.m)
        return Fraction(1, root + (root * root < self.m))

    @property
    def relative_tail(self): return (1 - self.delta**2)**self.m

    @property
    def trace_bound(self): return self.relative_tail / (2 - self.relative_tail)

    @property
    def pauli_bias_bound(self): return 2 * self.trace_bound

    @property
    def minimum_amplitude(self):
        return self.scale * min((1 - self.relative_tail) / self.delta, Fraction(1))

    @property
    def success_lower_bound(self): return self.minimum_amplitude**2

    def as_metadata(self):
        e = self.relative_tail
        return {
            "polynomial": "c*x*sum_(j=0..K)(1-x*x)^j",
            "K": self.k, "m": self.m, "degree": self.degree,
            "scale": float(self.scale), "scale_exact": str(self.scale),
            "delta": float(self.delta), "delta_exact": str(self.delta),
            "condition_upper_promised": float(1/self.delta),
            "spectral_promise": "delta<=|lambda(A/alpha)|<=1; signed Hermitian spectrum",
            "relative_inverse_tail_bound": _upper_float(e),
            "relative_inverse_tail_bound_exact": str(e),
            "scaled_inverse_operator_error_bound": _upper_float(self.scale*e/self.delta),
            "scaled_inverse_operator_error_bound_exact": str(self.scale*e/self.delta),
            "conditional_state_trace_distance_bound": _upper_float(self.trace_bound),
            "conditional_Pauli_bias_bound": _upper_float(self.pauli_bias_bound),
            "success_probability_lower_bound": _lower_float(self.success_lower_bound),
            "success_probability_lower_bound_exact": str(self.success_lower_bound),
            "minimum_accepted_amplitude_exact": str(self.minimum_amplitude),
            "whole_interval_magnitude_bound": 1.0,
            "whole_interval_squared_bound_exact":str(self.m*self.scale**2),
            "analytic_scope": "exact rational target polynomial and promised exact normalized Hermitian matrix; phase realization/access rounding/synthesis separate",
        }


def choose_k(delta=DEFAULT_DELTA, pauli_bias=.01) -> int:
    """Exact bounded scan; no floating-log seed or unsupported huge degree."""
    gap, allowance = _gap(delta), _rational(pauli_bias)
    if not 0 < allowance <= 2:
        raise ValueError("Pauli bias allowance must be in(0,2]")
    for k in range(MAX_K + 1):
        if ReciprocalPolynomial(k, gap).pauli_bias_bound <= allowance:
            return k
    raise ValueError("requested gap/accuracy exceeds bounded odd degree63")


def _cheb_multiply(a: dict[int, Fraction], b: dict[int, Fraction]):
    result: dict[int, Fraction] = {}
    for j, c in a.items():
        for k, d in b.items():
            for index in (j+k, abs(j-k)):
                result[index] = result.get(index, Fraction()) + c*d/2
    return {j: c for j, c in result.items() if c}


def exact_chebyshev_coefficients(k=DEFAULT_K) -> tuple[Fraction, ...]:
    model = ReciprocalPolynomial(k)
    power = {0: Fraction(1)}
    total = power.copy()
    for _ in range(k):
        power = _cheb_multiply(power, {0: Fraction(1, 2), 2: Fraction(-1, 2)})
        for j, c in power.items():
            total[j] = total.get(j, Fraction()) + c
    result = _cheb_multiply(total, {1: model.scale})
    return tuple(result.get(j, Fraction()) for j in range(model.degree+1))


def reciprocal_polynomial(x, k=DEFAULT_K):
    _index(k)
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
        raise ValueError("signal must be finite and in[-1,1]")
    # Evaluate regularized sum directly, including x=0; never divide by zero.
    power, total = np.ones_like(values), np.ones_like(values)
    for _ in range(k):
        power = power * (1-values*values)
        total = total + power
    return float(ReciprocalPolynomial(k).scale)*values*total


def response_and_jacobian(x, phases):
    """Analytic prefix/suffix Jacobian of the shared Hermitian signal model."""
    values = np.asarray(x, dtype=float)
    phases = tuple(float(p) for p in phases)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(np.abs(values)>1):
        raise ValueError("Jacobian signal must be a finite1D array in[-1,1]")
    if not 2 <= len(phases) <= 64 or not all(map(math.isfinite, phases)):
        raise ValueError("Jacobian requires2 to64 finite phases")
    n, length = len(values), len(phases)
    signal = np.zeros((n,2,2),complex)
    signal[:,0,0], signal[:,1,1] = values, -values
    signal[:,0,1] = signal[:,1,0] = np.sqrt(np.maximum(0,1-values*values))
    identity = np.tile(np.eye(2,dtype=complex),(n,1,1))
    left = np.empty((length,n,2,2),complex)
    right = np.empty_like(left)
    e = np.exp(1j*np.asarray(phases))
    diagonal = np.stack([e,e.conj()],axis=-1)
    current = identity.copy()
    for j in range(length):
        left[j] = current
        current = current*diagonal[j][None,None,:]
        if j<length-1: current = current@signal
    response = current[:,0,0].real
    current = identity.copy()
    for j in range(length-1,-1,-1):
        right[j] = current
        current = diagonal[j][None,:,None]*current
        if j: current = signal@current
    derivative = 1j*diagonal
    derivative[:,1] *= -1
    jacobian = ((left*derivative[:,None,None,:])@right)[:,:,0,0].real.T
    return response, jacobian


def fit_reciprocal_phases(k=DEFAULT_K):
    """Deterministic symmetric fit with analytical Jacobian; capped500nfev.

    Hermitian signal initialization is mapped from standard Wx endpoint pi/4
    phases. Starting all phases near zero instead gives a rank-deficient
    Jacobian because the Hermitian signal is an involution.
    """
    model = ReciprocalPolynomial(k)
    if k==0:
        return (0.,0.),{"method":"exact degree1 identity response","starts":0,"nodes":0,
                         "maximum_function_evaluations":0,"function_evaluations":0,"node_residual":0.}
    nodes_count = 4*(model.degree+1)
    nodes = np.cos(np.pi*(np.arange(nodes_count)+.5)/nodes_count)
    target = chebval(nodes, [float(c) for c in exact_chebyshev_coefficients(k)])
    initial = np.full(model.degree+1,np.pi/2)
    initial[0], initial[-1] = -model.degree*np.pi/2, np.pi
    initial = (initial+np.pi)%(2*np.pi)-np.pi
    # Symmetric changes in the standard-Wx convention span the odd target
    # coefficients while avoiding the redundant full-phase rank deficiency.
    def unpack(parameters):
        return initial+np.concatenate([parameters,parameters[::-1]])
    def jacobian(parameters):
        full = response_and_jacobian(nodes,unpack(parameters))[1]
        return full[:,:model.m]+full[:,model.m:][:,::-1]
    result = least_squares(
        lambda p: response_and_jacobian(nodes,unpack(p))[0]-target, np.zeros(model.m),
        jac=jacobian,ftol=1e-14,xtol=1e-14,gtol=None,max_nfev=500,
    )
    phases = tuple(float(p) for p in unpack(result.x))
    residual = float(np.max(np.abs(qsp_real_response(nodes,phases)-target)))
    if residual > 5e-12:
        raise RuntimeError("bounded reciprocal phase fit failed5e-12 node criterion")
    return phases, {
        "method":"scipy.optimize.least_squares with original analytical prefix/suffix Jacobian and symmetric standard-Wx phase changes",
        "starts":1, "seed":None, "nodes":nodes_count,
        "maximum_function_evaluations":500, "function_evaluations":result.nfev,
        "node_residual":residual, "ftol":1e-14,"xtol":1e-14,"gtol":None,
        "initialization":"Hermitian signal: endpoints -d*pi/2,pi; interiors pi/2, modulo2pi",
    }


def verify_reciprocal_phases(k=DEFAULT_K, phases=None):
    model = ReciprocalPolynomial(k)
    phases = DEFAULT_PHASES if phases is None and k==DEFAULT_K else phases
    if phases is None: phases = fit_reciprocal_phases(k)[0]
    phases = tuple(float(p) for p in phases)
    if len(phases)!=model.degree+1 or not all(map(math.isfinite,phases)):
        raise ValueError("phase count must equal degree+1 and all phases finite")
    target = exact_chebyshev_coefficients(k)
    nodes = np.cos(np.pi*(np.arange(4*(model.degree+1))+.5)/(4*(model.degree+1)))
    achieved = chebfit(nodes,qsp_real_response(nodes,phases),model.degree)
    discrepancy = float(np.sum(np.abs(achieved-np.array([float(c) for c in target]))))
    grid = np.linspace(-1,1,10001)
    certificate = certify_real_qsp_phases(phases,target)
    return {
        "K":k,"degree":model.degree,"phases":list(phases),
        "target_chebyshev_coefficients_exact":[str(c) for c in target],
        "achieved_chebyshev_coefficients_numerical":achieved.tolist(),
        "numerical_coefficient_l1_discrepancy":discrepancy,
        "dense_grid_max_error":float(np.max(np.abs(qsp_real_response(grid,phases)-reciprocal_polynomial(grid,k)))),
        "nodes":len(nodes),"grid_points":len(grid),
        "rigorous_ideal_signal_certificate":certificate.as_metadata(),
        "verification_scope":"exact rational uniform ideal-signal certificate plus independent numerical reconstruction/grid evidence; matrix loading and synthesis excluded",
    }


def phase_adjusted_bounds(model,certificate):
    """Rigorous exact-access/stored-phase contract; synthesis/access separate."""
    if certificate.degree!=model.degree or certificate.target_chebyshev_coefficients!=exact_chebyshev_coefficients(model.k):
        raise ValueError("phase certificate must match the chosen reciprocal target")
    eta = certificate.dyadic_upper_bound
    r = eta/model.scale
    if 1-model.relative_tail-r<=0:
        raise ValueError("phase error exceeds usable relative inverse range")
    minimum = model.minimum_amplitude-eta
    if minimum<=0: raise ValueError("no positive certified ideal-signal acceptance lower bound")
    success = minimum**2
    lattice = 1<<48
    retry_success = Fraction((success.numerator*lattice)//success.denominator,lattice)
    return {
        "uniform_ideal_phase_error_bound":certificate.float_upper_bound,
        "phase_adjusted_trace_bound":_upper_float((model.relative_tail+2*r)/(2-model.relative_tail)),
        "phase_adjusted_Pauli_bias_bound":_upper_float(2*(model.relative_tail+2*r)/(2-model.relative_tail)),
        "phase_adjusted_success_lower_bound":_lower_float(success),
        "retry_success_lower_bound_exact":str(retry_success) if retry_success else None,
        "retry_success_rounding":"downwards to2^-48 dyadic; exact target plus certified ideal stored phases",
        "scope":"promised exact matrix/access and exact binary phase gates; numerical access/loading, compiler and physical synthesis require separate certificates",
    }


def _matrix_terms(terms):
    terms = tuple(terms)
    if not 1<=len(terms)<=MAX_INPUT_TERMS:
        raise ValueError(f"supply1 to{MAX_INPUT_TERMS} explicit real Pauli terms")
    width, combined = None, {}
    for coefficient,word in terms:
        if not isinstance(word,str) or not word or set(word)-set("IXYZ"):
            raise ValueError("Pauli words must be nonempty overIXYZ")
        width = len(word) if width is None else width
        if len(word)!=width or width>MAX_SYSTEM_QUBITS:
            raise ValueError("equal Pauli widths of at most24 required")
        combined[word] = combined.get(word,Fraction())+_rational(coefficient)
    result = tuple((c,w) for w,c in sorted(combined.items()) if c)
    if not result: raise ValueError("matrix coefficient sum must be nonzero")
    alpha = sum((abs(c) for c,_ in result),Fraction())
    if max(abs(alpha.numerator).bit_length(),alpha.denominator.bit_length())>512:
        raise ValueError("normalization exceeds512-bit rational cap")
    try: numeric_alpha = float(alpha)
    except OverflowError as error: raise ValueError("normalization exceeds finite gate-loading representation") from error
    if not math.isfinite(numeric_alpha) or numeric_alpha==0:
        raise ValueError("normalization must have finite nonzero float representation")
    return result,alpha


def active_chain_terms(n=2):
    """Every spin active: A=3R I-sum Z_i-sum X_iX_(i+1), R=2n-1.

    alpha=4R and spectrum in[1/2,1] follows by coefficient norm bound.
    This family is distinct from an inactive-padding one-word example.
    """
    if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=MAX_SYSTEM_QUBITS:
        raise ValueError("active-chain data width must lie between1 and24")
    terms = [(3*(2*n-1),"I"*n)]
    for i in range(n):
        word = ["I"]*n; word[n-1-i]="Z"
        terms.append((-1,"".join(word)))
    for i in range(n-1):
        word = ["I"]*n; word[n-1-i]=word[n-2-i]="X"
        terms.append((-1,"".join(word)))
    return tuple(terms)


def default_preparation(n=2):
    if isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=MAX_SYSTEM_QUBITS:
        raise ValueError("preparation width must lie between1 and24")
    circuit = QuantumCircuit(n,name="Gate-built complex b")
    circuit.h(0); circuit.s(0); circuit.ry(.63,0)
    if n>1:
        circuit.ry(.37,n-1); circuit.cx(0,n-1)
    return circuit


def _gate_preparation(preparation,width):
    memo,active = {},set()
    def inspect(definition,depth=0):
        if depth>64: raise ValueError("preparation nesting exceeds64 levels")
        if id(definition) in active: raise ValueError("recursive preparation gate definition")
        if id(definition) in memo: return memo[id(definition)]
        active.add(id(definition))
        if len(definition.data)>MAX_SOURCE_OPERATIONS:
            raise ValueError("preparation exceeds bounded source-operation cap")
        total = 0
        for item in definition.data:
            if item.operation.name in {"unitary","state_preparation","state_preparation_dg","isometry"}:
                raise ValueError("supply explicit gate preparation, not a dense unitary or amplitude loader")
            nested = item.operation.definition
            total += 1 if nested is None else inspect(nested,depth+1)
            if total>MAX_SOURCE_OPERATIONS:
                raise ValueError("expanded preparation exceeds bounded source-operation cap")
        active.remove(id(definition))
        memo[id(definition)] = total
        return total
    if not isinstance(preparation,QuantumCircuit): raise ValueError("preparation must be explicit QuantumCircuit")
    inspect(preparation)
    return _preparation(preparation,width)


def build_reciprocal_qsvt(terms=DEFAULT_TERMS,k=DEFAULT_K,delta=DEFAULT_DELTA,phases=None):
    """Actual filter; the supplied spectral gap is a promise, never inferred."""
    model = ReciprocalPolynomial(k,delta)
    exact,alpha = _matrix_terms(terms)
    phases = DEFAULT_PHASES if phases is None and k==DEFAULT_K else phases
    if phases is None: phases = fit_reciprocal_phases(k)[0]
    phases = tuple(float(p) for p in phases)
    if len(phases)!=model.degree+1 or not all(map(math.isfinite,phases)):
        raise ValueError("supplydegree+1 finite phases")
    certificate = certify_real_qsp_phases(phases,exact_chebyshev_coefficients(k))
    if certificate.realization_error_bound>Fraction(1,10**10):
        raise ValueError("supplied phases do not meet bounded reciprocal realization criterion1e-10")
    phase_contract = phase_adjusted_bounds(model,certificate)
    floats = tuple((float(c),w) for c,w in exact)
    if any(c and float(c)==0 for c,_ in exact):
        raise ValueError("nonzero input coefficient underflows gate-loading representation")
    circuit = build_real_qsp_filter(floats,phases)
    if len(circuit.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("generated filter exceeds100000source operations")
    float_alpha = sum((abs(Fraction.from_float(c)) for c,_ in floats),Fraction())
    rho = sum((abs(Fraction.from_float(f)/float_alpha-c/alpha)
               for (c,w),(f,_) in zip(exact,floats)),Fraction())
    circuit.name = "Interval reciprocal QSVT"
    circuit.metadata.update({
        **model.as_metadata(),
        **phase_contract,
        "rigorous_ideal_signal_certificate":certificate.as_metadata(),
        "algorithm":"general signed-interval reciprocal Hermitian QSVT via real-QSP",
        "alpha_exact":str(alpha),"alpha":float(alpha),
        "input_terms_exact":[[str(c),w] for c,w in exact],
        "block_coefficient_conversion_l1_error_exact":str(rho),
        "block_coefficient_conversion_l1_error_bound":_upper_float(rho),
        "conversion_scope":"exact input versus rationalized float coefficients normalized by their exact l1 sum; probability/libm/angle evaluation and synthesis not covered",
        "spectral_promise_verified":False,
        "postselection":"all Pauli labels and real-part flag zero; data unrestricted",
        "encoded_function":"certified ideal-signal approximation of odd regularized reciprocal polynomial p_K(A/alpha); numeric access rounding separate",
        "status":"implemented interval reciprocal with numerical phases; no complete access/synthesis certificate",
        "amplitude_amplification":"not included; complete fresh-preparation retries counted",
    })
    return circuit


def prepared_reciprocal(terms=DEFAULT_TERMS,k=DEFAULT_K,delta=DEFAULT_DELTA,preparation=None,phases=None):
    block = build_reciprocal_qsvt(terms,k,delta,phases)
    width = block.metadata["n_system"]
    prep = default_preparation(width) if preparation is None else _gate_preparation(preparation,width)
    circuit = QuantumCircuit(block.num_qubits,name="Prepare and interval QSVT")
    circuit.compose(prep,qubits=range(width),inplace=True)
    circuit.compose(block,inplace=True)
    if len(circuit.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("complete preparation/filter exceeds bounded source-operation cap")
    circuit.metadata = {**block.metadata,"preparation":"explicit finite unitary gate circuit from fresh zeros"}
    return circuit


def lower_reciprocal(circuit):
    """Preserve coherent action on arbitrary inputs; rotations unsynthesized."""
    if circuit.num_qubits>31 or len(circuit.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("lowering input exceeds bounded wire/source-operation cap")
    result = transpile(circuit,basis_gates=["x","h","s","sdg","cx","rx","ry","rz","measure"],
                       optimization_level=1,seed_transpiler=7,qubits_initially_zero=False)
    if len(result.data)>MAX_COMPILED_OPERATIONS:
        raise ValueError("lowered circuit exceeds1000000operation cap")
    return result


def measured_reciprocal_trial(terms=DEFAULT_TERMS,k=DEFAULT_K,delta=DEFAULT_DELTA,preparation=None,phases=None,observable="Z"):
    if observable not in {"X","Y","Z"}:
        raise ValueError("observable must beX,Y orZ onq0")
    circuit = prepared_reciprocal(terms,k,delta,preparation,phases)
    if observable=="X": circuit.h(0)
    elif observable=="Y": circuit.sdg(0); circuit.h(0)
    circuit.measure_all()
    circuit.metadata.update({"observable":f"{observable}(q0)","success_predicate":"all measured bits at indices>=n_system zero"})
    return circuit


def compare_reciprocal(terms=DEFAULT_TERMS,k=DEFAULT_K,delta=DEFAULT_DELTA,preparation=None,phases=None):
    model = ReciprocalPolynomial(k,delta)
    exact,alpha = _matrix_terms(terms)
    width = len(exact[0][1])
    # Reject before generating/fitting a circuit outside bounded reference scope.
    labels = max(1,(len(exact)-1).bit_length())
    if width>MAX_REFERENCE_QUBITS or width+labels+1>MAX_STATEVECTOR_QUBITS:
        raise ValueError("dense/reference simulation limited to5 data/12 total wires")
    normalized = SparsePauliOp.from_list([(w,float(c/alpha)) for c,w in exact]).to_matrix()
    values,vectors = np.linalg.eigh(normalized)
    if min(abs(values)) < float(model.delta)-1e-12 or max(abs(values))>1+1e-12:
        raise ValueError("bounded reference matrix violates supplied signed spectral-gap promise")
    if min(abs(values))<1e-12:
        raise ValueError("normalized matrix numerically unresolved or singular")
    circuit = prepared_reciprocal(terms,k,delta,preparation,phases)
    prep = default_preparation(width) if preparation is None else _gate_preparation(preparation,width)
    b = Statevector.from_instruction(prep).data
    branch = Statevector.from_instruction(circuit).data[:1<<width]
    target_branch = vectors@(reciprocal_polynomial(values,k)*(vectors.conj().T@b))
    probability = float(np.vdot(branch,branch).real)
    if probability<1e-25: raise ValueError("numerical acceptance too small for reference")
    state = branch/math.sqrt(probability)
    solution = np.linalg.solve(normalized,b)
    residual = float(np.linalg.norm(normalized@solution-b))
    solution /= np.linalg.norm(solution)
    trace = float(np.linalg.norm(np.outer(state,state.conj())-np.outer(solution,solution.conj()),ord="nuc")/2)
    z = (-1.)**(np.arange(1<<width)%2)
    z_actual,z_target = float(z@abs(state)**2),float(z@abs(solution)**2)
    return {
        "performed":True,"n":width,"logical_qubits":circuit.num_qubits,
        "normalized_eigenvalues":values.tolist(),"actual_condition_number":float(max(abs(values))/min(abs(values))),
        "spectral_promise_numerically_met":True,"independent_solve_residual":residual,
        "success_probability":probability,"target_polynomial_success_probability":float(np.linalg.norm(target_branch)**2),
        "accepted_branch_error_vs_target_polynomial":float(np.linalg.norm(branch-target_branch)),
        "scaled_inverse_operator_error":float(max(abs(reciprocal_polynomial(values,k)-float(model.scale)/values))),
        "solution_trace_distance":trace,"z_actual":z_actual,"z_classical_solution":z_target,
        "z_absolute_bias":abs(z_actual-z_target),
        "analytic_operator_error_bound":_upper_float(model.scale*model.relative_tail/model.delta),
        "analytic_trace_bound":_upper_float(model.trace_bound),
        "analytic_Pauli_bias_bound":_upper_float(model.pauli_bias_bound),
        "analytic_success_lower_bound":_lower_float(model.success_lower_bound),
        "validation_scope":"bounded Qiskit ideal gate simulation versus dense spectral polynomial/classical solve; floating numerical checks, not full physical certificate",
    }


def observable_workload(success=Fraction(1,25),sampling_error=Fraction(3,100),sampling_failure=Fraction(1,250),no_acceptance_failure=Fraction(1,1000)):
    """Bounded complete-attempt proposal with exact retry power verification.

    M fixed accepted slots, at mostR fresh trials per slot. Probability of
    any empty slot <=M*(1-p)^R. Successful slots remain independent with
    the same conditional output law. Cap failure is separate from Hoeffding.
    """
    p,e,f,g = (_rational(v) for v in (success,sampling_error,sampling_failure,no_acceptance_failure))
    if not 0<p<=1 or not 0<e<=2 or not 0<f<1 or not 0<g<1:
        raise ValueError("success/accuracy/confidence probabilities outside range")
    # Cap precision before exact powers; the guard limits even adversarial rational denominators.
    if any(max(v.numerator.bit_length(),v.denominator.bit_length())>128 for v in (p,e,f,g)):
        raise ValueError("workload rational exceeds128-bit precision cap")
    raw_samples = 2*math.log(2/float(f))/float(e)**2
    if not math.isfinite(raw_samples) or raw_samples>MAX_TOTAL_ATTEMPTS:
        raise ValueError("requested sampling workload exceeds bounded attempt cap")
    samples = max(1,math.ceil(raw_samples))
    # Positive rational Taylor partial sum <=exp(t). Verifying the lower sum
    # >=2/f proves Hoeffding2exp(-M e²/2)<=f without trusting rounded logs.
    def hoeffding_certified(count):
        t = count*e*e/2
        term,total = Fraction(1),Fraction(1)
        for j in range(1,257):
            term = term*t/j
            total += term
            if total>=2/f: return True
        return False
    if not hoeffding_certified(samples):
        samples+=1
        if samples>MAX_TOTAL_ATTEMPTS or not hoeffding_certified(samples):
            raise ValueError("bounded rational Hoeffding certificate unavailable")
    if p==1: retries=1
    else:
        raw_retries = math.log(float(g/samples))/math.log1p(-_lower_float(p))
        if not math.isfinite(raw_retries) or raw_retries>MAX_RETRY_CAP:
            raise ValueError("retry proposal unavailable above100000attempts per slot")
        retries = max(1,math.ceil(raw_retries))
        while samples*(1-p)**retries>g:
            retries+=1
            if retries>MAX_RETRY_CAP: raise ValueError("verified retry cap exceeded")
        while retries>1 and samples*(1-p)**(retries-1)<=g: retries-=1
    attempts = samples*retries
    if attempts>MAX_TOTAL_ATTEMPTS: raise ValueError("total complete attempts exceed bounded cap")
    empty = (1-p)**retries
    return {
        "accepted_samples":samples,"retry_cap_per_slot":retries,"total_capped_attempts":attempts,
        "success_lower_bound_used":_lower_float(p),"success_lower_bound_exact":str(p),
        "sampling_error":float(e),"sampling_failure":float(f),"no_acceptance_failure_allowance":float(g),
        "verified_union_empty_slot_bound":_upper_float(samples*empty),
        "verified_minimal_retry_cap":retries==1 or samples*(1-p)**(retries-1)>g,
        "hoeffding_bound_certified":True,
        "hoeffding_certificate":"exact positive Taylor lower sum at M*e²/2 >=2/f; at most256terms",
        "uncapped_expected_attempts_upper":_upper_float(samples/p),
        "finite_cap_expected_attempts_upper":_upper_float(samples*(1-empty)/p),
        "assumptions":"independent fresh preparation, exact target acceptance lower bound; access/phase/synthesis changes require a revised lower bound",
        "status":"unexecuted ideal workload proposal; conditional cost projection, not output-confidence certificate",
    }
