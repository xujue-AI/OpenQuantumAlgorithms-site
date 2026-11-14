# SPDX-License-Identifier: MIT
"""Frozen, bounded physical sweep for the implemented interval reciprocal.

Every case uses the same normalized-Z contract, independent fresh retries and
hardware/failure assumptions. Exact phase certificates cover ideal access;
numerical loading/lowering and conditional QDK synthesis remain separate.
"""
from __future__ import annotations

import ast
from dataclasses import asdict,dataclass
from fractions import Fraction
import hashlib
import math
import re

import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector

from .linear_systems_qsvt_reciprocal import (
    DEFAULT_PHASES,MAX_K,ReciprocalPolynomial,build_reciprocal_qsvt,choose_k,
    default_preparation,exact_chebyshev_coefficients,fit_reciprocal_phases,
    lower_reciprocal,measured_reciprocal_trial,observable_workload,
    phase_adjusted_bounds,prepared_reciprocal,reciprocal_polynomial,
)
from .reciprocal_phase_certificate import certify_real_qsp_phases
from .resource_estimation import CircuitResourceBudget,SurfaceCodeProfile

MAX_NATIVE_WIRES=8
MAX_NATIVE_PRIMITIVES=25_000
NATIVE_TIMEOUT_SECONDS=60
CONTRACT={
    "observable":"normalized inverse-state Z(q0)",
    "absolute_error_target":.05,"algorithm_bias_allowance":.01,
    "synthesis_bias_allowance":.01,"sampling_error_allowance":.03,
    "statistical_failure":.004,"empty_slot_failure":.001,"physical_failure":.005,
    "synthesis_model_error_per_complete_attempt":1e-6,
    "profile":asdict(SurfaceCodeProfile()),
    "confidence_status":"conditional projection; numerical access/loading and concrete synthesis/physical output confidence are not certified",
    "failure_budget_scope":"the same allocations apply per benchmark case; no joint confidence claim for the full grid",
    "included":"actual gate input preparation, every signed Pauli PREPARE/SELECT/unprepare and projector phase, real extraction, all readout, all failed complete fresh trials",
    "excluded":["fresh-zero initialization/reset timing and faults","classical preprocessing/compilation/readout wall time in physical runtime",
                "arbitrary matrix/vector loading","concrete synthesis sequence and certificate"],
}


@dataclass(frozen=True)
class SweepCase:
    case_id:str
    n:int
    delta:Fraction
    axis:str

    def as_metadata(self):
        return {"id":self.case_id,"n":self.n,"delta_exact":str(self.delta),
                "condition_upper_promised":float(1/self.delta),"axis":self.axis}


CASES=(
    SweepCase("n1-gap1of2",1,Fraction(1,2),"size"),
    SweepCase("n2-gap1of2",2,Fraction(1,2),"size_and_condition"),
    SweepCase("n3-gap1of2",3,Fraction(1,2),"size"),
    SweepCase("n4-gap1of2",4,Fraction(1,2),"size"),
    SweepCase("n2-gap3of4",2,Fraction(3,4),"condition"),
    SweepCase("n2-gap3of5",2,Fraction(3,5),"condition"),
    SweepCase("n2-gap2of5",2,Fraction(2,5),"condition"),
    SweepCase("n2-gap1of3",2,Fraction(1,3),"unsupported_condition"),
    SweepCase("n2-gap1of4",2,Fraction(1,4),"unsupported_condition"),
)


def chain_terms(case:SweepCase):
    """Exact active-spin family with coefficient-norm gap delta=p/q.

    A=R(q+p)I-(q-p)(sum Z_i+sum X_iX_(i+1)),R=2n-1,alpha=2qR.
    All nonidentity words are distinct. Triangle bound gives signed spectrum
    inside[delta,1]; all spin positions participate. Gates use these inputs,
    never eigenvectors or a classical solve.
    """
    if isinstance(case.n,bool) or not isinstance(case.n,int) or not 1<=case.n<=4:
        raise ValueError("physical sweep active width is bounded to1..4")
    if not isinstance(case.delta,Fraction) or not 0<case.delta<1:
        raise ValueError("physical sweep gap must be an exact Fraction in(0,1)")
    if max(case.delta.numerator.bit_length(),case.delta.denominator.bit_length())>16:
        raise ValueError("sweep gap exceeds bounded16-bit precision")
    n,p,q=case.n,case.delta.numerator,case.delta.denominator
    r=2*n-1
    terms=[(r*(q+p),"I"*n)]
    for i in range(n):
        word=["I"]*n;word[n-1-i]="Z"
        terms.append((-(q-p),"".join(word)))
    for i in range(n-1):
        word=["I"]*n;word[n-1-i]=word[n-2-i]="X"
        terms.append((-(q-p),"".join(word)))
    return tuple(terms)


def select_degree(delta:Fraction):
    """Smallest supported degree meeting the phase-adjusted .01 allocation."""
    start=choose_k(delta,Fraction(1,100))
    for k in range(start,MAX_K+1):
        model=ReciprocalPolynomial(k,delta)
        phases=DEFAULT_PHASES if k==16 else fit_reciprocal_phases(k)[0]
        certificate=certify_real_qsp_phases(phases,exact_chebyshev_coefficients(k))
        eta=certificate.dyadic_upper_bound
        relative=eta/model.scale
        bias=2*(model.relative_tail+2*relative)/(2-model.relative_tail)
        if 1-model.relative_tail-relative>0 and bias<=Fraction(1,100):
            bounds=phase_adjusted_bounds(model,certificate)
            if bounds["retry_success_lower_bound_exact"] is None:
                continue
            return k,tuple(phases),certificate,bounds
    raise ValueError("no supported odd degree<=63 meets the fixed phase-adjusted algorithm allocation")


def independent_matrix(terms):
    matrices={"I":np.eye(2),"X":np.array([[0,1],[1,0]]),
              "Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}
    matrix=np.zeros((1<<len(terms[0][1]),)*2,complex)
    alpha=sum(abs(c) for c,w in terms)
    for coefficient,word in terms:
        operator=np.array([[1]],complex)
        for letter in word:operator=np.kron(operator,matrices[letter])
        matrix+=float(Fraction(coefficient,alpha))*operator
    return matrix


def _angle(text):
    def evaluate(node):
        if isinstance(node,ast.Constant) and type(node.value) in {int,float}:return node.value
        if isinstance(node,ast.Name) and node.id=="pi":return math.pi
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=evaluate(node.operand);return value if isinstance(node.op,ast.UAdd) else -value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            return a/b
        raise ValueError("unsupported canonical QASM angle")
    value=float(evaluate(ast.parse(text,mode="eval").body))
    if not math.isfinite(value):raise ValueError("nonfinite angle")
    return value


def replay_qasm(text,initial=None):
    """Independent NumPy vector replay; restricted canonical final-readout QASM."""
    width_match=re.search(r"qubit\[(\d+)\] q;",text)
    if width_match is None:raise ValueError("canonical q register missing")
    width=int(width_match.group(1))
    if not 1<=width<=MAX_NATIVE_WIRES:raise ValueError("QASM replay wire cap exceeded")
    state=np.zeros(1<<width,complex) if initial is None else np.asarray(initial,dtype=complex).copy()
    if initial is None:state[0]=1
    if state.shape!=(1<<width,) or not np.all(np.isfinite(state)) or not np.isclose(np.linalg.norm(state),1,atol=1e-12,rtol=0):
        raise ValueError("replay input must be a finite normalized vector")
    measurements=[];measuring=False
    lines=text.splitlines()
    if len(lines)>MAX_NATIVE_PRIMITIVES+20:raise ValueError("QASM replay operation cap exceeded")
    rows=np.arange(1<<width)
    for line in lines:
        if (line in {'OPENQASM 3.0;','include "stdgates.inc";'}
                or re.fullmatch(r"(?:qubit|bit)\[\d+\] \w+;",line) or line.startswith("barrier ")):continue
        match=re.fullmatch(r"meas\[(\d+)\] = measure q\[(\d+)\];",line)
        if match:
            measurements.append(tuple(map(int,match.groups())));measuring=True;continue
        if measuring:raise ValueError("only final measurements are supported")
        match=re.fullmatch(r"cx q\[(\d+)\], q\[(\d+)\];",line)
        if match:
            control,target=map(int,match.groups())
            if control==target or not 0<=control<width or not 0<=target<width:raise ValueError("invalid controlled-X wires")
            low=rows[(((rows>>control)&1)==1)&(((rows>>target)&1)==0)];high=low^(1<<target)
            state[low],state[high]=state[high].copy(),state[low].copy();continue
        rotation=re.fullmatch(r"(rx|ry|rz)\((.+)\) q\[(\d+)\];",line)
        plain=re.fullmatch(r"(x|h|s|sdg) q\[(\d+)\];",line)
        if rotation:
            name,angle,target=rotation.groups();target=int(target);angle=_angle(angle)
            pauli={"rx":np.array([[0,1],[1,0]]),"ry":np.array([[0,-1j],[1j,0]]),"rz":np.diag([1,-1])}[name]
            gate=math.cos(angle/2)*np.eye(2)-1j*math.sin(angle/2)*pauli
        elif plain:
            name,target=plain.groups();target=int(target)
            gate={"x":np.array([[0,1],[1,0]]),"h":np.array([[1,1],[1,-1]])/math.sqrt(2),
                  "s":np.diag([1,1j]),"sdg":np.diag([1,-1j])}[name]
        else:raise ValueError("unknown canonical QASM line")
        if not 0<=target<width:raise ValueError("invalid target wire")
        low=rows[((rows>>target)&1)==0];high=low^(1<<target)
        a,b=state[low].copy(),state[high].copy()
        state[low]=gate[0,0]*a+gate[0,1]*b;state[high]=gate[1,0]*a+gate[1,1]*b
    if sorted(measurements)!=[(i,i) for i in range(width)]:raise ValueError("measure every wire exactly once with matching bit order")
    return state


def build_case(case:SweepCase):
    terms=chain_terms(case)
    k,phases,certificate,bounds=select_degree(case.delta)
    unitary=prepared_reciprocal(terms,k,case.delta,phases=phases)
    measured=lower_reciprocal(measured_reciprocal_trial(terms,k,case.delta,phases=phases))
    primitive_count=len(measured.data)-measured.count_ops().get("barrier",0)
    if measured.num_qubits>MAX_NATIVE_WIRES or primitive_count>MAX_NATIVE_PRIMITIVES:
        raise ValueError("case exceeds predeclared native circuit materialization caps")
    # Symbolic-pi recognition can round near-pi stored phases. Emit the exact
    # round-tripping decimal literals for each stored numeric gate parameter.
    qasm=qasm3.dumps(measured,disable_constants=True)
    matrix=independent_matrix(terms)
    values,vectors=np.linalg.eigh(matrix)
    if min(values)<float(case.delta)-1e-12 or max(values)>1+1e-12:
        raise RuntimeError("independent matrix violates exact family interval promise")
    b=Statevector.from_instruction(default_preparation(case.n)).data
    state=Statevector.from_instruction(unitary).data
    branch=state[:1<<case.n]
    predicted=vectors@(reciprocal_polynomial(values,k)*(vectors.conj().T@b))
    p=float(np.linalg.norm(branch)**2)
    normalized=branch/math.sqrt(p)
    solution=np.linalg.solve(matrix,b);solve_residual=float(np.linalg.norm(matrix@solution-b));solution/=np.linalg.norm(solution)
    trace=float(np.linalg.norm(np.outer(normalized,normalized.conj())-np.outer(solution,solution.conj()),ord="nuc")/2)
    z=(-1.)**(np.arange(1<<case.n)%2)
    actual_z=float(z@abs(normalized)**2);target_z=float(z@abs(solution)**2)
    replayed=replay_qasm(qasm)
    qasm_phase=np.angle(np.vdot(state,replayed))
    replay_error=float(np.linalg.norm(replayed-state*np.exp(1j*qasm_phase)))
    probability_error=float(np.max(abs(abs(replayed)**2-abs(state)**2)))
    if (np.linalg.norm(branch-predicted)>1e-10 or replay_error>1e-10 or probability_error>1e-11
            or abs(actual_z-target_z)>bounds["phase_adjusted_Pauli_bias_bound"]+1e-11
            or p<bounds["phase_adjusted_success_lower_bound"]-1e-11):
        raise RuntimeError("bounded independent circuit/readout checks failed")
    # Observed acceptance is validation data only. Never use p for retries.
    workload=observable_workload(Fraction(bounds["retry_success_lower_bound_exact"]))
    metadata={
        **case.as_metadata(),"terms_exact":[[str(c),w] for c,w in terms],"matrix_dimension":1<<case.n,
        "alpha_exact":str(sum(abs(c) for c,w in terms)),"K":k,"degree":2*k+1,
        "phase_choice":"existing frozen default" if k==16 else "existing deterministic symmetric fitter",
        "phase_certificate":certificate.as_metadata(),"phase_adjusted_contract":bounds,
        "uniform_contract":CONTRACT,"workload":workload,"retry_probability_source":"rigorous ideal-phase interval lower bound rounded downward; never observed acceptance",
        "logical_qubits":measured.num_qubits,"primitive_operations":primitive_count,
        "primitive_gate_counts":dict(measured.count_ops()),"dependency_depth":measured.depth(),
        "block_access_calls_per_attempt":2*k+1,"input_preparation_counts":dict(default_preparation(case.n).count_ops()),
        "coefficient_conversion_l1_error_exact":unitary.metadata["block_coefficient_conversion_l1_error_exact"],
        "numeric_access_scope":unitary.metadata["conversion_scope"],
        "validation":{"performed":True,"normalized_eigenvalues":values.tolist(),"actual_condition_number":float(max(values)/min(values)),
                      "success_observed_ideal_numeric":p,"branch_error_vs_independent_spectral_polynomial":float(np.linalg.norm(branch-predicted)),
                      "solve_residual":solve_residual,"solution_trace_distance":trace,"z_actual":actual_z,"z_classical_solution":target_z,
                      "z_bias":abs(actual_z-target_z),"qasm_state_error_modulo_global_phase":replay_error,
                      "qasm_full_measured_probability_error":probability_error,
                      "scope":"deterministic declared input; independent Kronecker matrix/solve and restricted NumPy QASM vector replay; no uniform numeric-loading/compiler certificate"},
        "qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),"qasm_global_phase_radians":float(measured.global_phase),
        "qasm_scope":"complete measured trial; pinned export may omit overall phase",
        "qasm_literal_policy":"official Qiskit disable_constants=True; preserve stored numeric angles instead of approximate symbolic-pi recognition",
        "native_caps":{"wires":MAX_NATIVE_WIRES,"primitive_operations":MAX_NATIVE_PRIMITIVES,"timeout_seconds":NATIVE_TIMEOUT_SECONDS},
    }
    return measured,qasm,metadata


def resource_budget(metadata):
    return CircuitResourceBudget(1e-6,.005,metadata["workload"]["total_capped_attempts"])
