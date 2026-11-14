# SPDX-License-Identifier: MIT
"""Bounded retargetable complex QSP with explicit access and scoped certificates.

Actual real QSP phases are fitted for each normalization-time parameter. The
zero-ancilla block is (Pc(H/alpha)-i Ps(H/alpha))/2, targeting exp(-iHt)/4.
The coefficient certificate concerns the exact ideal signal and stored phases.
A separate weight certificate encloses the actual binary64-angle loader; the
logical-source bound includes both and exact parameter multiplication error.
Compilation, synthesis and physical faults have separate contracts. No matrix
exponential is a gate.

Original MIT implementation; Qiskit2.6.2 Apache-2.0, SciPy1.18.1 BSD-3-Clause.
Primary algorithms: Low/Chuang2017 doi:10.1103/PhysRevLett.118.010501;
Gilyen et al.2019 doi:10.1145/3313276.3316366.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from fractions import Fraction
import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.special import gammaln, jv

if __package__:
    from .lcu import build_pauli_lcu, postselection_attempts
else:
    from lcu import build_pauli_lcu, postselection_attempts

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_SYSTEM_QUBITS = 16
MAX_TERMS = 32
MAX_PHASE_DEGREE = 32
MAX_ABS_TAU = 8.0
MAX_SOURCE_OPERATIONS = 50_000
MAX_PRIMITIVE_OPERATIONS = 250_000
MAX_STATEVECTOR_QUBITS = 12
MAX_OPERATOR_QUBITS = 8
MAX_REFERENCE_SYSTEM_QUBITS = 6
MIN_DECODED_ERROR = 1e-9
MAX_DECODED_ERROR = 0.1


@dataclass(frozen=True)
class DynamicsPhasePlan:
    tau: float
    requested_decoded_error: float
    even_degree: int
    odd_degree: int
    cosine_phases: tuple[float, ...]
    sine_phases: tuple[float, ...]
    cosine_coefficients: tuple[float, ...]
    sine_coefficients: tuple[float, ...]
    cosine_fit: dict
    sine_fit: dict
    cosine_certificate: dict
    sine_certificate: dict
    decoded_polynomial_error_upper: float
    encoded_polynomial_error_upper: float
    ideal_signal_acceptance_lower: float
    certificate_scope: str

    def as_dict(self) -> dict:
        return asdict(self)


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _int(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _terms(terms) -> tuple[tuple[float, str], ...]:
    values = tuple(terms)
    if not 1 <= len(values) <= MAX_TERMS:
        raise ValueError(f"explicit Pauli input requires1..{MAX_TERMS} terms")
    result = []
    n = None
    for coefficient, label in values:
        coefficient = _finite(coefficient, "Pauli coefficient")
        if not isinstance(label, str) or not label or set(label)-set("IXYZ"):
            raise ValueError("Pauli labels must be nonempty IXYZ strings")
        n = len(label) if n is None else n
        if len(label) != n:
            raise ValueError("Pauli label lengths must agree")
        if coefficient != 0:
            result.append((coefficient, label))
    if not result or n > MAX_SYSTEM_QUBITS:
        raise ValueError("nonzero Hamiltonian or generator width limit invalid")
    alpha = float(sum(abs(c) for c, _ in result))
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("normalization must be finite and positive")
    return tuple(result)


def _phases(phases: Sequence[float], parity: int) -> tuple[float, ...]:
    values = tuple(_finite(p, "QSP phase") for p in phases)
    degree = len(values)-1
    if not 0 <= degree <= MAX_PHASE_DEGREE or degree%2 != parity:
        raise ValueError("QSP phase degree/parity exceeds its materialization cap")
    if any(abs(p) > math.pi for p in values):
        raise ValueError("stored phases must be wrapped into [-pi,pi]")
    return values


def dynamics_coefficients(tau: float, even_degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Floating fit targets only; the certificate independently encloses targets."""
    tau = _finite(tau, "tau")
    _int(even_degree, "even_degree", 0, MAX_PHASE_DEGREE-1)
    if even_degree%2 or abs(tau)>MAX_ABS_TAU:
        raise ValueError("even degree or normalization-time cap invalid")
    cosine, sine = np.zeros(even_degree+1), np.zeros(even_degree+2)
    cosine[0] = jv(0,tau)/2
    for k in range(2,even_degree+1,2):
        cosine[k] = (-1)**(k//2)*jv(k,tau)
    for k in range(1,even_degree+2,2):
        sine[k] = (-1)**((k-1)//2)*jv(k,tau)
    return cosine, sine


def _tail(tau: float, degree: int) -> float:
    """Floating evaluation of an analytic omitted-parity Bessel-tail bound."""
    a = abs(tau)/2
    if a == 0:
        return 0.0
    n = degree+2
    ratio = a*a/((n+1)*(n+2))
    if ratio >= 1:
        return math.inf
    return math.exp(n*math.log(a)-gammaln(n+1)+a*a/(n+1))/(1-ratio)


def choose_dynamics_degree(tau: float, decoded_error: float = 1e-5) -> int:
    tau = _finite(tau, "tau")
    error = _finite(decoded_error, "decoded_error")
    if abs(tau)>MAX_ABS_TAU or not MIN_DECODED_ERROR <= error <= MAX_DECODED_ERROR:
        raise ValueError("normalization-time or requested-error cap invalid")
    for degree in range(0,MAX_PHASE_DEGREE,2):
        if max(_tail(tau,degree),_tail(tau,degree+1)) <= error/8:
            return degree
    raise ValueError("requested truncation exceeds bounded phase-degree support")


def generate_dynamics_phases(tau: float, decoded_error: float = 1e-5,
                             *, seed: int = 37) -> DynamicsPhasePlan:
    """Fit each target, then fail closed against a separate interval validator."""
    if __package__:
        from .qsp_phase_generation import fit_real_qsp_phases
        from .qsp_certificate import certify_real_qsp
    else:
        from qsp_phase_generation import fit_real_qsp_phases
        from qsp_certificate import certify_real_qsp
    _int(seed,"seed",0,2**32-2)
    degree = choose_dynamics_degree(tau,decoded_error)
    cosine, sine = dynamics_coefficients(tau,degree)
    fit_tolerance = min(1e-10,decoded_error/64)
    pc, report_c = fit_real_qsp_phases(cosine,tolerance=fit_tolerance,seed=seed)
    ps, report_s = fit_real_qsp_phases(sine,tolerance=fit_tolerance,seed=seed+1)
    # The interval validator treats the newly stored binary64 phases exactly.
    pc = tuple(math.remainder(p,2*math.pi) for p in pc)
    ps = tuple(math.remainder(p,2*math.pi) for p in ps)
    cert_c = certify_real_qsp(pc,target_kind="cosine",parameter=float(tau))
    cert_s = certify_real_qsp(ps,target_kind="sine",parameter=float(tau))
    component_sum = Fraction.from_float(float(cert_c["uniform_error_upper"]))+Fraction.from_float(float(cert_s["uniform_error_upper"]))
    decoded = _upper_float(2*component_sum)
    if decoded > decoded_error:
        raise ValueError("interval ideal-signal error exceeds requested decoded allowance")
    encoded = _upper_float(component_sum/2)
    lower = _lower_float(max(Fraction(),Fraction(1,4)-Fraction.from_float(encoded))**2)
    return DynamicsPhasePlan(float(tau),float(decoded_error),degree,degree+1,pc,ps,
        tuple(cosine),tuple(sine),report_c,report_s,cert_c,cert_s,decoded,encoded,lower,
        "interval coefficient/error bound for exact ideal signal access and stored binary64 phases/parameter; actual coefficient loading/access rounding, parameter multiplication, gate-angle realization, synthesis and physical faults excluded")



def _upper_float(value: Fraction) -> float:
    number = float(value)
    return math.nextafter(number,math.inf) if Fraction.from_float(number)<value else number


def _lower_float(value: Fraction) -> float:
    number = float(value)
    return math.nextafter(number,0.0) if Fraction.from_float(number)>value else number


def audit_dynamics_phase_plan(plan: DynamicsPhasePlan) -> DynamicsPhasePlan:
    """Revalidate supplied/cache plans from their actual stored phases."""
    if not isinstance(plan,DynamicsPhasePlan):
        raise ValueError("phase plan must be a DynamicsPhasePlan")
    if __package__:
        from .qsp_certificate import certify_real_qsp
    else:
        from qsp_certificate import certify_real_qsp
    tau = _finite(plan.tau, "phase-plan tau")
    allowance = _finite(plan.requested_decoded_error, "phase-plan decoded allowance")
    if abs(tau) > MAX_ABS_TAU or not MIN_DECODED_ERROR <= allowance <= MAX_DECODED_ERROR:
        raise ValueError("phase-plan parameter or requested-error cap invalid")
    _int(plan.even_degree, "phase-plan even degree", 0, MAX_PHASE_DEGREE-1)
    _int(plan.odd_degree, "phase-plan odd degree", 1, MAX_PHASE_DEGREE)
    pc,ps=_phases(plan.cosine_phases,0),_phases(plan.sine_phases,1)
    if len(pc)!=plan.even_degree+1 or len(ps)!=plan.odd_degree+1 or plan.odd_degree!=plan.even_degree+1:
        raise ValueError("phase plan degree fields do not match its stored phases")
    cc=certify_real_qsp(pc,target_kind="cosine",parameter=plan.tau)
    cs=certify_real_qsp(ps,target_kind="sine",parameter=plan.tau)
    total=Fraction.from_float(float(cc["uniform_error_upper"]))+Fraction.from_float(float(cs["uniform_error_upper"]))
    decoded,encoded=_upper_float(2*total),_upper_float(total/2)
    if decoded>plan.requested_decoded_error:
        raise ValueError("stored phase plan fails its requested interval allowance")
    lower=_lower_float(max(Fraction(),Fraction(1,4)-Fraction.from_float(encoded))**2)
    cosine,sine=dynamics_coefficients(tau,plan.even_degree)
    return replace(plan,cosine_coefficients=tuple(cosine),sine_coefficients=tuple(sine),
        cosine_certificate=cc,sine_certificate=cs,
        decoded_polynomial_error_upper=decoded,encoded_polynomial_error_upper=encoded,
        ideal_signal_acceptance_lower=lower)

def _conditioned_phase(circuit,index,flag,selector,phase,value):
    if value==0:
        circuit.x(selector)
    circuit.append(RZGate(2*phase).control(1,annotated=False),[selector,flag])
    if index:
        circuit.x(index)
    circuit.append(RZGate(-4*phase).control(len(index)+1,annotated=False),index+[selector,flag])
    if index:
        circuit.x(index)
    if value==0:
        circuit.x(selector)



def build_hermitian_pauli_access(terms, *, return_components: bool=False):
    """Actual PREPARE and exact logical signed-Pauli SELECT, then unprepare.

    Coefficient magnitudes still load binary64 RY angles. Negative signs use
    exact Clifford pattern operations instead of P(binary64 pi), so SELECT's
    Hermitian-involution identity is algebraic for the ideal named gates.
    """
    terms=_terms(terms)
    loading=build_pauli_lcu(terms)
    n,m=loading.n_system,loading.n_index
    index=list(range(n,n+m))
    select=QuantumCircuit(n+m,name="Exact signed Pauli SELECT")
    for label,(coefficient,pauli) in enumerate(terms):
        zeros=[q for k,q in enumerate(index) if not label&(1<<k)]
        if zeros:select.x(zeros)
        if coefficient<0:
            if m==0:
                # Chronological Z X Z X multiplies to -I, without a pi angle.
                select.z(0);select.x(0);select.z(0);select.x(0)
            elif m==1:
                select.z(index[0])
            else:
                select.h(index[-1]);select.mcx(index[:-1],index[-1]);select.h(index[-1])
        for target,letter in enumerate(reversed(pauli)):
            if letter=="I":continue
            if m==0:
                getattr(select,letter.lower())(target)
            else:
                if letter=="Y":select.sdg(target)
                elif letter=="Z":select.h(target)
                select.mcx(index,target)
                if letter=="Y":select.s(target)
                elif letter=="Z":select.h(target)
        if zeros:select.x(zeros)
    access=QuantumCircuit(n+m,name="Hermitian Pauli access")
    access.compose(loading.prepare,qubits=index,inplace=True)
    access.compose(select,inplace=True)
    access.compose(loading.prepare.inverse(),qubits=index,inplace=True)
    access.metadata={**loading.circuit.metadata,"sign_construction":"exact Clifford zero-pattern phase; no binary64 pi sign angle",
        "ideal_access_property":"Hermitian involution for exact named gates and inverse preparation",
        "access_rounding_scope":"magnitude/probability loading and resulting RY angles need independent error bounds"}
    return (loading.prepare,select,access) if return_components else access

def build_two_parity_filter(terms, even_phases, odd_phases,
                            *, odd_branch_phase: float = -math.pi/2) -> QuantumCircuit:
    """Actual two-real-polynomial selector; no target-specific accuracy claim.

    Zero selector/flag/labels encodes (Pe + relative_factor Po)/2. All common
    B calls are shared; one actual controlled B accounts for the extra odd
    degree. Relative phases0,+/-pi/2 use exact identity/S/S-dagger gates.
    """
    terms = _terms(terms)
    even, odd = _phases(even_phases,0), _phases(odd_phases,1)
    if len(odd)!=len(even)+1:
        raise ValueError("shared access requires adjacent even/odd degrees")
    relative = _finite(odd_branch_phase,"odd_branch_phase")
    block = build_hermitian_pauli_access(terms)
    d = len(even)-1
    estimated = (d+2)*block.size()+8*(len(even)+len(odd))
    if estimated>MAX_SOURCE_OPERATIONS:
        raise ValueError("top-level source materialization estimate exceeds cap")
    n = block.metadata["n_system"]
    flag,selector=block.num_qubits,block.num_qubits+1
    index=list(range(n,flag))
    circuit=QuantumCircuit(block.num_qubits+2,name="Retargetable real-pair QSP")
    circuit.h(flag);circuit.h(selector)
    for j in range(d+1):
        _conditioned_phase(circuit,index,flag,selector,even[d-j],0)
        _conditioned_phase(circuit,index,flag,selector,odd[d+1-j],1)
        if j<d:
            circuit.compose(block,qubits=range(flag),inplace=True)
    circuit.append(block.to_gate(label="Controlled PREPARE SELECT unprepare").control(1,annotated=False),
                   [selector]+list(range(flag)))
    _conditioned_phase(circuit,index,flag,selector,odd[0],1)
    circuit.h(flag)
    if relative==-math.pi/2:
        circuit.sdg(selector)
    elif relative==math.pi/2:
        circuit.s(selector)
    elif relative != 0:
        circuit.p(relative,selector)
    circuit.h(selector)
    circuit.metadata={**block.metadata,"n_flag":1,"n_selector":1,"even_degree":d,"odd_degree":d+1,
        "even_phases":list(even),"odd_phases":list(odd),"odd_branch_phase":relative,
        "ordinary_access_calls":d,"controlled_access_calls":1,
        "branch_conditioned_projector_phase_calls":len(even)+len(odd),
        "source_operation_estimate":estimated,"actual_source_operations":circuit.size(),
        "postselection":"all label, real-part flag and selector bits zero",
        "encoded_operator":"(Pe(H/alpha)+relative_factor Po(H/alpha))/2",
        "floating_access_scope":"binary64 coefficient probabilities and PREPARE angles; exact logical Clifford signs; no loading-rounding certificate",
        "rotations_synthesized":False,"amplification":"none"}
    if circuit.size()>MAX_SOURCE_OPERATIONS:
        raise ValueError("actual top-level source operation count exceeds cap")
    return circuit


def build_retargetable_evolution(terms,time: float=1.0,decoded_error: float=1e-5,
                                 *,phase_plan: DynamicsPhasePlan|None=None,seed:int=37) -> QuantumCircuit:
    terms=_terms(terms);time=_finite(time,"time")
    alpha=float(sum(abs(c) for c,_ in terms));tau=alpha*time
    if not math.isfinite(tau) or abs(tau)>MAX_ABS_TAU:
        raise ValueError("normalization times time exceeds bounded phase-generation domain")
    decoded_error=_finite(decoded_error,"decoded_error")
    if not MIN_DECODED_ERROR<=decoded_error<=MAX_DECODED_ERROR:
        raise ValueError("requested-error cap invalid")
    plan=generate_dynamics_phases(tau,decoded_error,seed=seed) if phase_plan is None else audit_dynamics_phase_plan(phase_plan)
    if not isinstance(plan,DynamicsPhasePlan) or plan.tau!=tau or plan.requested_decoded_error>decoded_error:
        raise ValueError("phase plan must match actual normalization-time and requested allowance")
    circuit=build_two_parity_filter(terms,plan.cosine_phases,plan.sine_phases)
    if __package__:
        from .qsp_certificate import certify_prepared_pauli_weights
    else:
        from qsp_certificate import certify_prepared_pauli_weights
    prepare,_,_=build_hermitian_pauli_access(terms,return_components=True)
    loading=certify_prepared_pauli_weights(terms,prepare)
    exact_alpha=sum((Fraction.from_float(abs(c)) for c,_ in terms),Fraction())
    exact_tau=exact_alpha*Fraction.from_float(time)
    parameter_gap=abs(exact_tau-Fraction.from_float(tau))
    parameter_upper=_upper_float(parameter_gap)
    ideal_upper=_upper_float(Fraction.from_float(plan.decoded_polynomial_error_upper)+parameter_gap)
    access_error=abs(exact_tau)*Fraction.from_float(loading["weight_l1_error_upper"])
    combined_upper=_upper_float(Fraction.from_float(plan.decoded_polynomial_error_upper)+parameter_gap+access_error)
    if combined_upper>decoded_error:
        raise ValueError("logical-source polynomial/loading/parameter error exceeds requested allowance")
    encoded_upper=_upper_float(Fraction.from_float(combined_upper)/4)
    source_acceptance=_lower_float(max(Fraction(),Fraction(1,4)-Fraction.from_float(encoded_upper))**2)
    source_certificate={
        "certified":True,
        "decoded_operator_error_upper":combined_upper,
        "encoded_operator_error_upper":encoded_upper,
        "uniform_acceptance_lower":source_acceptance,
        "polynomial_decoded_error_upper":plan.decoded_polynomial_error_upper,
        "parameter_mismatch_decoded_error_upper":parameter_upper,
        "loading_decoded_error_upper":_upper_float(access_error),
        "prepared_weight_certificate":loading,
        "exact_normalization_numerator":str(exact_alpha.numerator),
        "exact_normalization_denominator":str(exact_alpha.denominator),
        "exact_normalization_time_numerator":str(exact_tau.numerator),
        "exact_normalization_time_denominator":str(exact_tau.denominator),
        "conditional_state_vector_error_upper":_upper_float(2*Fraction.from_float(combined_upper)),
        "conditional_Z_bias_upper":_upper_float(2*Fraction.from_float(combined_upper)),
        "scope":"declared logical source for exact real input numbers represented by supplied binary64 coefficients/time, exact named Cliffords, conditional controls and stored rotation angles; exact signed SELECT and interval-enclosed PREPARE weights; polynomial, loading and parameter errors combined by functional calculus and Duhamel. External model-to-input representation, floating nested ControlledGate/MCX realization, compiler/synthesis and physical faults are separate",
        "excluded":["external model-to-binary64 input representation","floating nested control/MCX realization","compiler control/angle changes","coherent rotation synthesis","initialization/readout/physical faults"],
    }
    circuit.metadata.update({"time":time,"phase_tau":tau,"normalization_alpha":alpha,
        "requested_decoded_polynomial_error":decoded_error,"accepted_branch_normalization":4,
        "target":"exp(-iHt)/4","phase_plan":plan.as_dict(),
        "normalization_sum_rounding_absolute":float(abs(exact_alpha-Fraction.from_float(alpha))),
        "normalization_time_rounding_absolute":float(abs(exact_tau-Fraction.from_float(tau))),
        "parameter_mismatch_operator_error_upper":parameter_upper,
        "ideal_signal_decoded_error_upper_with_parameter_rounding":ideal_upper,
        "logical_source_certificate":source_certificate,
        "precompiler_source_decoded_error_upper":combined_upper,
        "precompiler_source_acceptance_lower":source_acceptance,
        "ideal_signal_normalization_definition":"exact L1 sum of supplied binary64 Pauli coefficients; floating PREPARE is an approximation to this ideal model",
        "accuracy_scope":source_certificate["scope"],
        "phase_plan_origin":"generated" if phase_plan is None else "supplied and interval-recertified",
        "parameterized_generation":"phases target this tau; no frozen4.1 phase reuse"})
    return circuit


def build_measured_trial(circuit:QuantumCircuit,n_system:int|None=None,initial_basis:int=0) -> QuantumCircuit:
    if circuit.num_clbits or circuit.num_parameters or any(i.operation.name in {"reset","measure","initialize"} for i in circuit.data):
        raise ValueError("core must be coherent and parameters bound")
    n=circuit.metadata["n_system"] if n_system is None else n_system
    _int(n,"n_system",1,MAX_SYSTEM_QUBITS);_int(initial_basis,"initial_basis",0,2**n-1)
    if n!=circuit.metadata["n_system"]:
        raise ValueError("system count must match core metadata")
    attempt=QuantumCircuit(circuit.num_qubits,circuit.num_qubits,name="Retargetable dynamics measured trial")
    attempt.reset(range(circuit.num_qubits))
    for q in range(n):
        if initial_basis&(1<<q):attempt.x(q)
    attempt.compose(circuit,inplace=True)
    attempt.measure(range(circuit.num_qubits),range(circuit.num_qubits))
    attempt.metadata={**circuit.metadata,"initial_basis":initial_basis,
        "initial_reset_instructions":circuit.num_qubits,"measurement_instructions":circuit.num_qubits,
        "output":"accepted measured Z trial; system state checked before final readout",
        "acceptance_bit_pattern":"all classical bits n_system..n_total-1 zero",
        "retry_protocol":"restart full known-basis preparation after rejection; charge complete attempt"}
    return attempt


def lower_circuit(circuit:QuantumCircuit) -> QuantumCircuit:
    result=transpile(circuit,basis_gates=ROTATION_BASIS,optimization_level=1,
                     seed_transpiler=7,qubits_initially_zero=False)
    if result.size()>MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive export exceeds operation cap after lowering")
    if set(result.count_ops())-set(ROTATION_BASIS)-{"measure","reset","barrier"}:
        raise ValueError("unsupported primitive operation after lowering")
    return result


def _pauli(label):
    matrices={"I":np.eye(2),"X":np.array([[0,1],[1,0]]),"Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}
    matrix=np.array([[1.+0j]])
    for letter in label:matrix=np.kron(matrix,matrices[letter])
    return matrix


def compare_evolution(circuit,terms,time:float,initial_basis:int=0,*,full_operator:bool=True) -> dict:
    terms=_terms(terms);n=len(terms[0][1]);time=_finite(time,"time")
    if n>MAX_REFERENCE_SYSTEM_QUBITS or circuit.num_qubits>MAX_STATEVECTOR_QUBITS:
        raise ValueError("bounded classical reference/statevector cap exceeded")
    _int(initial_basis,"initial_basis",0,2**n-1)
    if circuit.num_clbits or circuit.num_parameters or circuit.metadata["n_system"]!=n:
        raise ValueError("comparison requires matching coherent core")
    if full_operator and circuit.num_qubits>MAX_OPERATOR_QUBITS:
        raise ValueError("full Operator comparison exceeds total-wire cap")
    dimension=2**n
    h=sum((c*_pauli(p) for c,p in terms),np.zeros((dimension,dimension),complex))
    reference=expm(-1j*time*h)[:,initial_basis]
    input_state=np.zeros(2**circuit.num_qubits,complex);input_state[initial_basis]=1
    full=Statevector(input_state).evolve(circuit).data
    branch=full[:dimension];p=float(np.vdot(branch,branch).real)
    if p<=1e-30:raise ValueError("accepted branch is numerically zero")
    state=branch/math.sqrt(p);z=_pauli("I"*(n-1)+"Z")
    z_ref=float(np.vdot(reference,z@reference).real);z_actual=float(np.vdot(state,z@state).real)
    result={"performed":True,"full_operator_performed":bool(full_operator),
        "conditional_state_scope":"accepted system state before final system readout",
        "conditional_state_vector_error":float(np.linalg.norm(state-reference)),
        "conditional_state_infidelity":float(max(0.,1-abs(np.vdot(reference,state))**2)),
        "acceptance_probability":p,"Z_exact":z_ref,"Z_conditional":z_actual,"Z_bias":abs(z_actual-z_ref),
        "conditional_state_real":state.real.tolist(),"conditional_state_imag":state.imag.tolist(),
        "encoded_block_operator_error":None,"decoded_operator_error":None,
        "finite_instance_minimum_acceptance":None,"finite_instance_maximum_acceptance":None,
        "reference":"independent Kronecker Pauli matrix and scipy.linalg.expm; no matrix is a gate"}
    if full_operator:
        block=Operator(circuit).data[:dimension,:dimension];target=expm(-1j*time*h)
        singular=np.linalg.svd(block,compute_uv=False)
        result.update({"encoded_block_operator_error":float(np.linalg.norm(block-target/4,2)),
            "decoded_operator_error":float(np.linalg.norm(4*block-target,2)),
            "finite_instance_minimum_acceptance":float(singular[-1]**2),
            "finite_instance_maximum_acceptance":float(singular[0]**2)})
    return result


def logical_ledger(core:QuantumCircuit,measured:QuantumCircuit,*,acceptance:float|None=None) -> dict:
    core_counts=dict(core.count_ops());counts=dict(measured.count_ops());n=core.metadata["n_system"]
    d=core.metadata["even_degree"]
    retry=None
    if acceptance is not None:
        acceptance=_finite(acceptance,"acceptance")
        retry={"model":"state-specific numerical precompiler acceptance; compiled/synthesized/physical errors remain separate",
            "probability":acceptance,"one_accepted_trial_failure_target":.01,
            "one_accepted_trial_cap":postselection_attempts(acceptance,.01),"expected_complete_attempts":1/acceptance}
    return {"resource_kind":"actual logical primitive circuit before synthesis/physical scheduling",
        "logical_qubits":measured.num_qubits,"n_system":n,"n_index":core.metadata["n_index"],
        "ordinary_access_calls":d,"controlled_access_calls":1,
        "prepare_calls":d+1,"select_calls":d+1,"prepare_inverse_calls":d+1,
        "projector_phase_calls":core.metadata["branch_conditioned_projector_phase_calls"],
        "core_primitive_counts":core_counts,"core_dependency_depth":core.depth(),
        "full_trial_primitive_counts":counts,"full_trial_dependency_depth":measured.depth(),
        "full_trial_operations":measured.size(),"continuous_rotations":sum(counts.get(g,0) for g in ("rx","ry","rz")),
        "initial_resets":counts.get("reset",0),"final_measurements":counts.get("measure",0),
        "accepted_output":"system Z bitstring conditional on zero labels/flag/selector",
        "state_output_scope":"state checks before final readout; measured export returns classical outcomes",
        "postselection_retry":retry,"physical_resource_estimate":None,"rotations_synthesized":False,
        "logical_source_certificate":core.metadata.get("logical_source_certificate"),
        "exclusions":["compiler changes to source angles/controls","coherent rotation synthesis","physical faults and reset/readout/routing/factory time","classical fit/certificate/preprocessing/readout time in quantum counts"],
        "source_operation_cap_scope":"top-level explicit source instructions; nested lowering may cost more before primitive cap"}
