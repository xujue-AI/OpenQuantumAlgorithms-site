# SPDX-License-Identifier: MIT
"""Uniform reciprocal approximation by exact-coefficient Neumann-series LCU.

For a=a_I>sum|a_P|=r, A=aI+sum a_PP is positive definite. With
alpha=a+r, B=A/alpha has promised spectrum[delta,1],delta=(a-r)/alpha.
T_K=delta sum_(k=0..K)(I-B)^k uniformly approximates delta B^-1.
Pauli coefficients are aggregated with exact rational arithmetic; no matrix
inverse or eigendecomposition constructs a gate. Classical aggregation can
grow exponentially, and is capped separately from gate simulation.

Known-input amplitude amplification reflects about the actual fully prepared
input via a full initial-zero reflection. It is not oblivious amplification
of a nonunitary inverse operator. All gates use numerical analytic angles;
the exact-coefficient approximation bound is not a hardware/synthesis proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .linear_systems_qsvt import _preparation as _unitary_preparation
from .simulation.lcu import _multiply_paulis, build_pauli_lcu

MAX_SYSTEM_QUBITS = 24
MAX_INPUT_TERMS = 32
MAX_AGGREGATE_TERMS = 256
MAX_DEGREE = 64
MAX_AMPLIFICATION_ROUNDS = 4
MAX_SOURCE_OPERATIONS = 50_000
MAX_REFERENCE_SYSTEM_QUBITS = 6
MAX_STATEVECTOR_QUBITS = 12
MAX_RATIONAL_INPUT_BITS = 512
MAX_COMMON_DENOMINATOR_BITS = 128
DEFAULT_TERMS = (("3/4", "II"), ("-1/10", "IX"), ("-1/10", "ZI"), ("-1/20", "XX"))


def _rational(value) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Coefficients must be finite real rational values, not booleans")
    if isinstance(value, complex):
        if value.imag or not math.isfinite(value.real):
            raise ValueError("Coefficients must be real and finite")
        value = value.real
    if isinstance(value,Fraction):
        if max(abs(value.numerator).bit_length(),value.denominator.bit_length())>MAX_RATIONAL_INPUT_BITS:
            raise ValueError("Rational input exceeds the512-bit coefficient cap")
        literal=None
    elif isinstance(value,int):
        if abs(value).bit_length()>MAX_RATIONAL_INPUT_BITS:
            raise ValueError("Integer input exceeds the512-bit coefficient cap")
        literal=str(value)
    else:
        literal=str(value).strip()
        if len(literal)>192:
            raise ValueError("Rational input literal exceeds192 characters")
        exponent=re.search(r"[eE]([+-]?\d+)$",literal)
        if exponent and abs(int(exponent.group(1)))>128:
            raise ValueError("Scientific exponents must lie between-128 and128")
    try:
        result = value if isinstance(value,Fraction) else Fraction(literal)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Supply a finite real decimal or rational coefficient") from error
    if max(abs(result.numerator).bit_length(),result.denominator.bit_length())>MAX_RATIONAL_INPUT_BITS:
        raise ValueError("Rational input exceeds the512-bit coefficient cap")
    try:
        numeric = float(result)
    except OverflowError as error:
        raise ValueError("Coefficient exceeds finite gate-loading float representation") from error
    if not math.isfinite(numeric) or (result and numeric == 0):
        raise ValueError("Coefficient is not representable as a finite nonzero gate-loading float")
    return result


def _terms(terms) -> tuple[tuple[Fraction, str], ...]:
    terms = tuple(terms)
    if not 1 <= len(terms) <= MAX_INPUT_TERMS:
        raise ValueError(f"Supply between1 and{MAX_INPUT_TERMS} explicit Pauli terms")
    coefficients = {}
    width = None
    for coefficient, word in terms:
        if not isinstance(word, str) or not word or set(word)-set("IXYZ"):
            raise ValueError("Pauli words must be nonempty strings over IXYZ")
        if width is None:
            width = len(word)
        if len(word) != width or width > MAX_SYSTEM_QUBITS:
            raise ValueError(f"Equal Pauli widths of at most{MAX_SYSTEM_QUBITS} are required")
        coefficients[word] = coefficients.get(word, Fraction(0)) + _rational(coefficient)
    clean = tuple((value,word) for word,value in sorted(coefficients.items()) if value)
    if not clean:
        raise ValueError("The matrix must have a nonzero coefficient")
    if any(max(abs(c.numerator).bit_length(),c.denominator.bit_length())>MAX_RATIONAL_INPUT_BITS for c,_ in clean):
        raise ValueError("Aggregated input exceeds the512-bit coefficient cap")
    identity = "I"*width
    constant = coefficients.get(identity,Fraction(0))
    radius = sum((abs(value) for word,value in coefficients.items() if word != identity),Fraction(0))
    if constant <= radius:
        raise ValueError("Require a positive identity coefficient strictly larger than the nonidentity coefficient l1 sum")
    try:
        normalization=float(constant+radius)
    except OverflowError as error:
        raise ValueError("Input normalization exceeds finite float representation") from error
    if not math.isfinite(normalization):
        raise ValueError("Input normalization must be finite in the gate-loading representation")
    return clean


def _upper_float(value: Fraction) -> float:
    numeric=float(value)
    return numeric if Fraction(numeric)>=value else math.nextafter(numeric,math.inf)


def _lower_float(value: Fraction) -> float:
    numeric=float(value)
    return numeric if Fraction(numeric)<=value else math.nextafter(numeric,-math.inf)


@dataclass(frozen=True)
class InverseLCU:
    input_terms: tuple[tuple[Fraction,str], ...]
    coefficients: tuple[tuple[Fraction,str], ...]
    degree: int
    alpha: Fraction
    delta: Fraction
    gamma: Fraction

    @property
    def n_system(self): return len(self.input_terms[0][1])

    @property
    def normalization(self): return sum((abs(c) for c,_ in self.coefficients),Fraction(0))

    @property
    def tail_bound(self): return self.gamma**(self.degree+1)

    def as_metadata(self):
        error = self.tail_bound
        return {
            "algorithm":"Neumann-series inverse via classically aggregated explicit Pauli LCU",
            "n_system":self.n_system,"degree":self.degree,
            "alpha":float(self.alpha),"alpha_exact":str(self.alpha),
            "delta":float(self.delta),"delta_exact":str(self.delta),
            "gamma":float(self.gamma),"gamma_exact":str(self.gamma),
            "promised_normalized_spectrum":[float(self.delta),1.0],
            "promised_condition_upper":float(1/self.delta),
            "normalization":float(self.normalization),"normalization_exact":str(self.normalization),
            "input_terms_exact":[[str(c),p] for c,p in self.input_terms],
            "polynomial_terms_exact":[[str(c),p] for c,p in self.coefficients],
            "polynomial":"T_K=delta*sum_(k=0..K)(I-A/alpha)^k",
            "term_count":len(self.coefficients),
            "scaled_inverse_operator_error_bound":_upper_float(error),
            "scaled_inverse_operator_error_bound_exact":str(error),
            "unscaled_B_inverse_error_bound":_upper_float(error/self.delta),
            "conditional_state_trace_distance_bound":_upper_float(error/(2-error)),
            "conditional_Pauli_bias_bound":_upper_float(2*error/(2-error)),
            "success_probability_lower_bound":_lower_float((self.delta/self.normalization)**2),
            "coefficient_normalization_upper_bound":_upper_float(1-error),
            "analytical_scope":"exact rational input/polynomial and ideal exact-angle Pauli LCU; uniform spectral bound, not a floating/synthesis certificate",
            "classical_aggregation":"exact Fraction coefficients; Hermitian anticommutator products; capped256 terms anddegree64",
            "rational_precision_limits":{"input_numerator_denominator_bits":MAX_RATIONAL_INPUT_BITS,
                "normalized_common_denominator_bits":MAX_COMMON_DENOMINATOR_BITS,
                "literal_characters":192,"scientific_exponent_magnitude":128},
            "rotation_synthesis":"not generated or certified",
        }


def neumann_coefficients(terms=DEFAULT_TERMS, degree: int=12) -> InverseLCU:
    if isinstance(degree,bool) or not isinstance(degree,int) or not 0 <= degree <= MAX_DEGREE:
        raise ValueError(f"degree must be an integer from0 to{MAX_DEGREE}")
    clean = _terms(terms)
    width = len(clean[0][1]); identity="I"*width
    constant = next(c for c,p in clean if p==identity)
    radius = sum((abs(c) for c,p in clean if p!=identity),Fraction(0))
    alpha=constant+radius;delta=(constant-radius)/alpha;gamma=1-delta
    if max(abs(alpha.numerator).bit_length(),alpha.denominator.bit_length())>MAX_RATIONAL_INPUT_BITS:
        raise ValueError("Input normalization exceeds the512-bit rational cap")
    if float(delta)==0:
        raise ValueError("Promised gap is below the gate-loading float representation")
    complement = {p:-c/alpha for c,p in clean if p!=identity}
    if radius: complement[identity]=radius/alpha
    denominator=math.lcm(delta.denominator,gamma.denominator,*(c.denominator for c in complement.values()))
    if denominator.bit_length()>MAX_COMMON_DENOMINATOR_BITS:
        raise ValueError("Normalized coefficients exceed the128-bit common-denominator cap")
    power={identity:Fraction(1)}
    polynomial={identity:delta}
    for _ in range(degree):
        next_power={}
        # Since power is a polynomial in C=I-B, CP=(CP+PC)/2 exactly.
        # Anticommuting Pauli pairs vanish algebraically rather than by tolerance.
        for left, c_left in complement.items():
            for right, c_right in power.items():
                phase,word=_multiply_paulis(left,right)
                if phase.imag: continue
                value=c_left*c_right*int(phase.real)
                next_power[word]=next_power.get(word,Fraction(0))+value
        power={p:c for p,c in next_power.items() if c}
        if len(power)>MAX_AGGREGATE_TERMS:
            raise ValueError("Neumann power exceeds the256-term classical aggregation cap")
        for word,c in power.items():polynomial[word]=polynomial.get(word,Fraction(0))+delta*c
        polynomial={p:c for p,c in polynomial.items() if c}
        if len(polynomial)>MAX_AGGREGATE_TERMS:
            raise ValueError("Neumann polynomial exceeds the256-term classical aggregation cap")
    model=InverseLCU(clean,tuple((c,p) for p,c in sorted(polynomial.items())),degree,alpha,delta,gamma)
    if not delta <= model.normalization <= 1-model.tail_bound:
        raise RuntimeError("Exact Neumann coefficient normalization violates its analytic contract")
    return model


def degree_for_pauli_bias(delta, allowance: float=.001) -> int:
    gap=_rational(delta)
    if not 0 < gap <= 1 or not math.isfinite(allowance) or not 0 < allowance < 2:
        raise ValueError("Supply gap in(0,1] and positive Pauli-bias allowance below2")
    # Solve 2E/(2-E)<=allowance, E=(1-delta)^(K+1), then verify exactly.
    threshold=2*Fraction(str(allowance))/(2+Fraction(str(allowance)))
    # Small valid gaps can round 1-gap to 1 in binary64. Search the bounded
    # degree domain using exact fractions, without a floating logarithm seed.
    complement=1-gap
    tail=complement
    for degree in range(MAX_DEGREE+1):
        if tail<=threshold:
            return degree
        tail*=complement
    raise ValueError("Requested gap/accuracy needs a polynomial above thedegree64 cap")


def build_neumann_lcu(terms=DEFAULT_TERMS, degree: int=12) -> QuantumCircuit:
    model=neumann_coefficients(terms,degree)
    coefficients=[]
    for coefficient,word in model.coefficients:
        number=float(coefficient)
        if not math.isfinite(number) or (coefficient and number==0):
            raise ValueError("Polynomial coefficient cannot be represented for gate loading")
        coefficients.append((number,word))
    circuit=build_pauli_lcu(coefficients).circuit
    circuit.name="Neumann inverse LCU"
    circuit.metadata.update(model.as_metadata())
    circuit.metadata["n_index"]=(len(coefficients)-1).bit_length()
    circuit.metadata["gate_loading_normalization"]=float(sum(abs(c) for c,_ in coefficients))
    rounding=sum((abs(Fraction(number)-exact) for (number,_),(exact,_) in zip(coefficients,model.coefficients)),Fraction(0))
    circuit.metadata["coefficient_conversion_l1_error_bound_exact"]=str(rounding)
    circuit.metadata["coefficient_conversion_l1_error_bound"]=_upper_float(rounding)
    circuit.metadata["conversion_adjusted_trace_bound_exact_angle_model"]=_upper_float(model.tail_bound/(2-model.tail_bound)+rounding/model.delta)
    circuit.metadata["gate_loading_status"]="exact rational coefficients converted to numerical analytic-angle gates; no conversion/synthesis certificate"
    circuit.metadata["conversion_bound_scope"]="coefficient-to-float l1 error only; probability normalization/libm angle implementation and synthesis need a separate coherent eta certificate"
    if len(circuit.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("Source access exceeds the50,000-operation cap")
    return circuit


def default_preparation(n_system: int=2) -> QuantumCircuit:
    if isinstance(n_system,bool) or not isinstance(n_system,int) or not 2 <= n_system <= MAX_SYSTEM_QUBITS:
        raise ValueError("Default complex preparation requires between2 and24 data qubits")
    circuit=QuantumCircuit(n_system,name="Prepare complex b")
    circuit.ry(.63,0);circuit.rz(.41,0);circuit.ry(.37,n_system-1)
    return circuit


def structured_inverse_terms(n: int=2, delta=.5):
    if isinstance(n,bool) or not isinstance(n,int) or not 2 <= n <= MAX_SYSTEM_QUBITS:
        raise ValueError("Structured inverse family requires between2 and24 data qubits")
    gap=_rational(delta)
    if not 0<gap<1:raise ValueError("Gap must be strictly between zero andone")
    radius=(1-gap)/2;constant=(1+gap)/2
    return ((constant,"I"*n),(-2*radius/5,"I"*(n-1)+"X"),
            (-2*radius/5,"Z"+"I"*(n-1)),(-radius/5,"X"*n))


def zero_reflection(width: int, targets: Sequence[int]) -> QuantumCircuit:
    """I-2Pi_zero using actual X/MCZ/X; unspecified wires pass unchanged."""
    targets=tuple(targets)
    if (isinstance(width,bool) or not isinstance(width,int) or not 1<=width<=32
            or len(set(targets))!=len(targets)
            or any(isinstance(q,bool) or not isinstance(q,int) or not 0<=q<width for q in targets)):
        raise ValueError("Supply a valid width and distinct target qubit indices")
    circuit=QuantumCircuit(width,name="Reflect zeros")
    if not targets:
        circuit.global_phase=math.pi
        return circuit
    circuit.x(list(targets))
    if len(targets)==1:circuit.z(targets[0])
    else:
        circuit.h(targets[-1]);circuit.mcx(list(targets[:-1]),targets[-1]);circuit.h(targets[-1])
    circuit.x(list(targets))
    return circuit


def prepared_inverse(terms=DEFAULT_TERMS, degree: int=12,
                     preparation: QuantumCircuit | None=None, rounds: int=0) -> QuantumCircuit:
    if isinstance(rounds,bool) or not isinstance(rounds,int) or not 0<=rounds<=MAX_AMPLIFICATION_ROUNDS:
        raise ValueError(f"rounds must be an integer from0 to{MAX_AMPLIFICATION_ROUNDS}")
    access=build_neumann_lcu(terms,degree);n=access.metadata["n_system"]
    prep=_unitary_preparation(preparation,n)
    source=QuantumCircuit(access.num_qubits,name="Prepare inverse output")
    source.compose(prep,list(range(n)),inplace=True);source.compose(access,inplace=True)
    good=zero_reflection(source.num_qubits,range(n,source.num_qubits))
    initial=zero_reflection(source.num_qubits,range(source.num_qubits))
    circuit=source.copy()
    for _ in range(rounds):
        circuit.compose(good,inplace=True);circuit.compose(source.inverse(),inplace=True)
        circuit.compose(initial,inplace=True);circuit.compose(source,inplace=True)
        circuit.global_phase+=math.pi
    if len(circuit.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("Amplified source exceeds the50,000-operation cap")
    circuit.name=f"Known-input inverse AA({rounds})"
    circuit.metadata={**access.metadata,"amplification_rounds":rounds,
        "full_source_calls":2*rounds+1,"forward_source_calls":rounds+1,"inverse_source_calls":rounds,
        "good_reflections":rounds,"initial_zero_reflections":rounds,
        "amplification":"known-input amplitude amplification; full initial-state reflection, not oblivious",
        "input_preparation_counts":dict(prep.count_ops()),
        "postselection":"all label ancillas zero; data unrestricted",
        "retry_requirement":"fresh actual re-preparation; cannot copy a single unknown input specimen"}
    return circuit


def measured_inverse_trial(terms=DEFAULT_TERMS,degree: int=12,preparation=None,
                           rounds: int=0,observable: str="Z") -> QuantumCircuit:
    if observable not in {"X","Y","Z"}:raise ValueError("Measure X,Y orZ on dataq0")
    circuit=prepared_inverse(terms,degree,preparation,rounds)
    if observable=="X":circuit.h(0)
    elif observable=="Y":circuit.sdg(0);circuit.h(0)
    circuit.measure_all();circuit.metadata["observable"]=f"{observable}(q0)"
    return circuit


def compare_inverse(terms=DEFAULT_TERMS,degree: int=12,preparation=None,rounds: int=0) -> dict:
    model=neumann_coefficients(terms,degree)
    circuit=prepared_inverse(terms,degree,preparation,rounds);n=model.n_system
    if n>MAX_REFERENCE_SYSTEM_QUBITS or circuit.num_qubits>MAX_STATEVECTOR_QUBITS:
        raise ValueError("Independent dense references capped at6 data/12 total qubits")
    b=Statevector.from_instruction(_unitary_preparation(preparation,n)).data
    # Assemble normalized coefficients directly to avoid scale-dependent
    # overflow/underflow in an independent classical reference.
    B=SparsePauliOp.from_list([(p,float(c/model.alpha)) for c,p in model.input_terms]).to_matrix()
    eigen_normalized=np.linalg.eigvalsh(B)
    if eigen_normalized[0]<=1e-14:
        raise ValueError("Classical comparison requires a numerically resolved positive spectrum above1e-14")
    eigen=eigen_normalized*float(model.alpha);C=np.eye(len(B),dtype=complex)-B
    power=np.eye(len(B),dtype=complex);polynomial=power.copy()
    for _ in range(degree):power=power@C;polynomial+=power
    polynomial*=float(model.delta)
    exact=np.linalg.solve(B,b);target=exact/np.linalg.norm(exact)
    expected_branch=polynomial@b/float(model.normalization)
    p_source=float(np.linalg.norm(expected_branch)**2)
    branch=Statevector.from_instruction(circuit).data[:1<<n]
    success=float(np.vdot(branch,branch).real)
    if success<1e-24:raise ValueError("Numerical amplification has vanishing success; choose anotherround")
    accepted=branch/math.sqrt(success)
    rho=np.outer(accepted,accepted.conj());sigma=np.outer(target,target.conj())
    trace_distance=float(.5*np.sum(abs(np.linalg.eigvalsh(rho-sigma))))
    ideal_accepted=expected_branch/math.sqrt(p_source)
    polynomial_distance=float(.5*np.sum(abs(np.linalg.eigvalsh(rho-np.outer(ideal_accepted,ideal_accepted.conj())))))
    signs=1-2*(np.arange(len(b))&1)
    z=float(np.dot(abs(accepted)**2,signs));z_classical=float(np.dot(abs(target)**2,signs))
    delta=float(model.delta);error=float(np.linalg.norm(polynomial-delta*np.linalg.inv(B),ord=2))
    predicted=math.sin((2*rounds+1)*math.asin(math.sqrt(min(1,max(0,p_source)))))**2
    return {"performed":True,"logical_qubits":circuit.num_qubits,"system_qubits":n,"degree":degree,
        "matrix_eigenvalues":eigen.tolist(),"normalized_eigenvalues":eigen_normalized.tolist(),
        "actual_condition_number":float(eigen_normalized[-1]/eigen_normalized[0]),
        "unamplified_success_probability":p_source,"success_probability":success,"amplification_prediction":predicted,
        "amplification_rounds":rounds,"conditional_state_trace_distance_to_classical_solution":trace_distance,
        "conditional_state_trace_distance_to_unamplified_polynomial":polynomial_distance,
        "z_quantum":z,"z_classical_solution":z_classical,"z_absolute_bias":abs(z-z_classical),
        "scaled_inverse_operator_error":error,
        "analytic_operator_error_bound":model.as_metadata()["scaled_inverse_operator_error_bound"],
        "analytic_conditional_trace_distance_bound":model.as_metadata()["conditional_state_trace_distance_bound"],
        "analytic_conditional_Pauli_bias_bound":model.as_metadata()["conditional_Pauli_bias_bound"],
        "solution_state_real":accepted.real.tolist(),"solution_state_imag":accepted.imag.tolist(),
        "classical_solution_state_real":target.real.tolist(),"classical_solution_state_imag":target.imag.tolist(),
        "classical_residual_norm":float(np.linalg.norm(B@exact-b)),
        "classical_reference_equation":"(A/alpha)y=b; normalized y equals normalized A^-1b; residual uses normalized system",
        "scope":"bounded numerical circuits and independent matrices; analytic approximation bound assumes exact rational gates, not certified synthesis"}
