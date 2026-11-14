# SPDX-License-Identifier: MIT
"""Known-input fixed-point amplification with explicit projector phases.

Yoder, Low and Chuang (2014), doi:10.1103/PhysRevLett.113.210501.
Convention: G(alpha,beta)=-A S0(-alpha) A† Sgood(beta). Each round
chronologically applies Sgood(beta), A†, S0(-alpha), A and a global pi.
The ideal phase schedule is distinct from its floating-angle realization and
from a certified fault-tolerant synthesis or physical success guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Barrier
from qiskit.circuit.library import (
    HGate, XGate, YGate, ZGate, SGate, SdgGate, TGate, TdgGate,
    RXGate, RYGate, RZGate, PhaseGate, CXGate, CYGate, CZGate,
    CPhaseGate, CRXGate, CRYGate, CRZGate, CCXGate, MCXGate,
    MCPhaseGate, SwapGate,
)
from qiskit.quantum_info import Statevector

MAX_LENGTH = 255
MAX_QUBITS = 12
MAX_SOURCE_OPERATIONS = 200_000
MAX_PRIMITIVE_OPERATIONS = 250_000
MAX_SIM_QUBITS = 10
MAX_DEFINITION_DEPTH = 64
PRIMITIVE_BASIS = ("h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx")
_STANDARD = (HGate, XGate, YGate, ZGate, SGate, SdgGate, TGate, TdgGate,
             RXGate, RYGate, RZGate, PhaseGate, CXGate, CYGate, CZGate,
             CPhaseGate, CRXGate, CRYGate, CRZGate, CCXGate, MCXGate,
             MCPhaseGate, SwapGate)


def _probability(name: str, value: Real, *, include_one: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real probability")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1 or (not include_one and result == 1):
        raise ValueError(f"{name} must lie in (0, {'1]' if include_one else '1)'}")
    return result


@dataclass(frozen=True)
class FixedPointSchedule:
    success_lower_bound: float
    failure_amplitude: float
    length: int
    gamma: float
    alphas: tuple[float, ...]
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        lower = _probability("success_lower_bound", self.success_lower_bound, include_one=True)
        delta = _probability("failure_amplitude", self.failure_amplitude, include_one=False)
        if delta**2 == 0:
            raise ValueError("Failure amplitude is below the binary64 probability reporting floor")
        if (isinstance(self.length, bool) or not isinstance(self.length, int) or
            not 1 <= self.length <= MAX_LENGTH or self.length % 2 != 1):
            raise ValueError("Schedule length must be a bounded positive odd integer")
        scale = _acosh_inverse(delta) / self.length
        gamma = delta if self.length == 1 else 1 / math.cosh(scale)
        expected = tuple(2*math.atan2(1., math.tan(2*math.pi*j/self.length)*math.tanh(scale))
                         for j in range(1, self.rounds + 1))
        if (isinstance(self.gamma,bool) or not isinstance(self.gamma, Real) or not math.isfinite(float(self.gamma)) or
            not math.isclose(float(self.gamma), gamma, rel_tol=1e-14, abs_tol=0)):
            raise ValueError("Schedule gamma does not match delta and length")
        if not isinstance(self.alphas, tuple) or not isinstance(self.betas, tuple):
            raise ValueError("Schedule phases must be immutable tuples")
        for actual, reference in ((self.alphas, expected),
                                  (self.betas, tuple(-x for x in reversed(expected)))):
            if (len(actual) != self.rounds or
                any(not isinstance(x, Real) or not math.isfinite(float(x)) or
                    not math.isclose(float(x), y, rel_tol=1e-14, abs_tol=1e-14)
                    for x, y in zip(actual, reference, strict=True))):
                raise ValueError("Schedule phases do not match the YLC convention")
        if self.numerical_threshold_width > lower:
            raise ValueError("Schedule width exceeds the supplied success promise")

    @property
    def rounds(self) -> int:
        return (self.length - 1) // 2

    @property
    def ideal_failure_bound(self) -> float:
        return self.failure_amplitude**2

    @property
    def numerical_threshold_width(self) -> float:
        if self.length == 1:
            return (1-self.failure_amplitude)*(1+self.failure_amplitude)
        return math.tanh(_acosh_inverse(self.failure_amplitude)/self.length)**2

    def as_metadata(self) -> dict:
        return {
            "success_lower_bound_promised": self.success_lower_bound,
            "failure_amplitude_delta": self.failure_amplitude,
            "ideal_failure_probability_bound": self.ideal_failure_bound,
            "length_L": self.length, "rounds": self.rounds,
            "gamma": self.gamma, "numerical_threshold_width": self.numerical_threshold_width,
            "alpha_phases_radians": list(self.alphas), "beta_phases_radians": list(self.betas),
            "phase_convention": "G=-A S0(-alpha) A† Sgood(beta); chronological beta,A†,-alpha,A,global pi",
            "phase_construction": "binary64 analytic angles; not an interval or synthesis certificate",
            "ideal_theorem_scope": "exact phase functions and unitary preparation, promised input success p>=lower_bound",
            "hardware_success_certified": False,
        }


def _acosh_inverse(delta: float) -> float:
    # acosh(1/delta) without overflowing 1/delta at small positive floats.
    return math.log1p(math.sqrt((1-delta)*(1+delta))) - math.log(delta)


def fixed_point_schedule(success_lower_bound: float, failure_amplitude: float=.05,
                         *, max_length: int=MAX_LENGTH) -> FixedPointSchedule:
    """Choose an odd schedule from a lower bound, never from exact success p.

    L >= acosh(1/delta)/atanh(sqrt(lower)) ensures the ideal width
    1-gamma² <= lower. A bounded numerical width check can add one round.
    Returned phases are floating values, not a rigorous angle certificate.
    """
    lower = _probability("success_lower_bound", success_lower_bound, include_one=True)
    delta = _probability("failure_amplitude", failure_amplitude, include_one=False)
    if delta**2 == 0:
        raise ValueError("Failure amplitude is below the binary64 probability reporting floor")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= MAX_LENGTH:
        raise ValueError(f"max_length must be an integer from 1 to {MAX_LENGTH}")
    if lower == 1:
        length = 1
    else:
        ratio = _acosh_inverse(delta)/math.atanh(math.sqrt(lower))
        length = max(1, math.ceil(ratio))
        if length % 2 == 0:
            length += 1
    if length > max_length:
        raise ValueError("Requested lower-bound/failure pair exceeds the bounded schedule length")
    def width(value: int) -> float:
        return ((1-delta)*(1+delta) if value == 1 else
                math.tanh(_acosh_inverse(delta)/value)**2)
    while width(length) > lower:
        length += 2
        if length > max_length:
            raise ValueError("Numerical width check needs a schedule above the length cap")
    if length == 1:
        gamma = delta
        alphas: tuple[float, ...] = ()
    else:
        scale = _acosh_inverse(delta)/length
        gamma = 1/math.cosh(scale)
        width_root = math.tanh(scale)
        alphas = tuple(2*math.atan2(1., math.tan(2*math.pi*j/length)*width_root)
                       for j in range(1, (length-1)//2 + 1))
    return FixedPointSchedule(lower, delta, length, gamma, alphas,
                              tuple(-angle for angle in reversed(alphas)))


def ideal_success_probability(probability: float, schedule: FixedPointSchedule) -> float:
    """Numerical evaluation of 1-delta² T_L(sqrt(1-p)/gamma)².

    This curve is a reference calculation, not a gate simulator/certificate.
    The polynomial remains defined below the promise, where the failure
    guarantee is unavailable. Exact endpoints avoid cancellation.
    """
    if not isinstance(schedule, FixedPointSchedule):
        raise ValueError("Supply a validated FixedPointSchedule")
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise ValueError("probability must be a real value in [0,1]")
    p=float(probability)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability must be in [0,1]")
    if p in (0.,1.) or schedule.length == 1:
        return p
    x=math.sqrt(1-p)/schedule.gamma
    value=(math.cos(schedule.length*math.acos(x)) if x<=1 else
           math.cosh(schedule.length*math.acosh(x)))
    result=1-(schedule.failure_amplitude*value)**2
    if not -1e-10 <= result <= 1+1e-10:
        raise ArithmeticError("Numerical Chebyshev probability left its valid range")
    return min(1.,max(0.,result))


def _validate_preparation(preparation: QuantumCircuit) -> int:
    if not isinstance(preparation, QuantumCircuit):
        raise ValueError("Supply an explicit QuantumCircuit preparation")
    if not 1 <= preparation.num_qubits <= MAX_QUBITS:
        raise ValueError(f"Preparation width must be 1 to {MAX_QUBITS}")
    if preparation.num_clbits or preparation.num_parameters:
        raise ValueError("Preparation must be bound, coherent and have no classical bits")
    if not math.isfinite(float(preparation.global_phase)):
        raise ValueError("Preparation global phase must be finite")
    visited: set[int] = set()
    count=0
    expansion_proxy=0

    def inspect(circuit: QuantumCircuit, depth: int=0) -> None:
        nonlocal count, expansion_proxy
        if depth>MAX_DEFINITION_DEPTH:
            raise ValueError("Preparation definition nesting exceeds its depth cap")
        if circuit.num_clbits or circuit.num_parameters:
            raise ValueError("Nested preparation must be bound and have no classical bits")
        if not math.isfinite(float(circuit.global_phase)):
            raise ValueError("Nested preparation global phase must be finite")
        if id(circuit) in visited:
            raise ValueError("Recursive circuit definitions are unsupported")
        visited.add(id(circuit))
        for item in circuit.data:
            count+=1
            if count>MAX_SOURCE_OPERATIONS:
                raise ValueError("Preparation exceeds the recursive source-operation cap")
            operation=item.operation
            expansion_proxy += (2**(operation.num_qubits+3)
                                if isinstance(operation,(MCXGate,MCPhaseGate)) else 1)
            if expansion_proxy>MAX_SOURCE_OPERATIONS:
                raise ValueError("Multi-control expansion preflight exceeds the operation cap")
            if isinstance(operation,Barrier):
                continue
            if getattr(operation,"condition",None) is not None or getattr(operation,"blocks",None):
                raise ValueError("Preparation cannot contain classical control")
            if operation.name in {"reset","measure","initialize","unitary","store"}:
                raise ValueError("Preparation needs explicit coherent gates, not reset/measurement/dense loading")
            for value in operation.params:
                if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(float(value)):
                    raise ValueError("Preparation parameters must be finite real numbers")
            if not isinstance(operation,_STANDARD):
                definition=operation.definition
                if definition is None:
                    raise ValueError(f"Opaque preparation operation: {operation.name}")
                inspect(definition, depth+1)
        visited.remove(id(circuit))
    inspect(preparation)
    return count


def projector_phase(n_qubits: int, good_qubits: Sequence[int], good_state: int,
                    angle: float) -> QuantumCircuit:
    """Phase exp(i angle) on the exact register pattern, identity elsewhere."""
    if isinstance(n_qubits,bool) or not isinstance(n_qubits,int) or not 1<=n_qubits<=MAX_QUBITS:
        raise ValueError("Invalid projector circuit width")
    wires=tuple(good_qubits)
    if (not wires or len(set(wires))!=len(wires) or
        any(isinstance(q,bool) or not isinstance(q,int) or not 0<=q<n_qubits for q in wires)):
        raise ValueError("Good qubits must be a nonempty distinct in-range sequence")
    if isinstance(good_state,bool) or not isinstance(good_state,int) or not 0<=good_state<2**len(wires):
        raise ValueError("Good state must fit the selected register")
    if isinstance(angle, bool) or not isinstance(angle,Real) or not math.isfinite(float(angle)):
        raise ValueError("Projector angle must be finite and real")
    circuit=QuantumCircuit(n_qubits,name="projector phase")
    zero_wires=[wire for bit,wire in enumerate(wires) if not ((good_state>>bit)&1)]
    for wire in zero_wires:circuit.x(wire)
    if len(wires)==1:circuit.p(float(angle),wires[0])
    else:circuit.mcp(float(angle),list(wires[:-1]),wires[-1])
    for wire in reversed(zero_wires):circuit.x(wire)
    return circuit


def build_fixed_point_amplification(preparation: QuantumCircuit, good_qubits: Sequence[int],
                                    good_state: int, success_lower_bound: float,
                                    failure_amplitude: float=.05, *, measure: bool=False,
                                    max_length: int=MAX_LENGTH) -> QuantumCircuit:
    """Prepare A|0> and apply the full phased known-input reflections.

    An arbitrary unknown specimen is not a valid preparation oracle. This
    routine never obtains exact p to choose the schedule. Initialization is
    the circuit model's all-zero logical input, not a physical reset protocol.
    """
    if not isinstance(measure, bool):
        raise ValueError("measure must be a bool")
    good_qubits = tuple(good_qubits)
    source_count=_validate_preparation(preparation)
    schedule=fixed_point_schedule(success_lower_bound,failure_amplitude,max_length=max_length)
    width=preparation.num_qubits
    # Preflight guards nested multi-control expansion independently of the
    # post-transpilation primitive cap; it is not a compilation-runtime proof.
    proxy=schedule.length*source_count+schedule.rounds*(2**(width+3)+2**(len(good_qubits)+3))
    if proxy>MAX_SOURCE_OPERATIONS:
        raise ValueError("Fixed-point source/reflector preflight exceeds its operation cap")
    # Validate the mask even when no amplification round is needed.
    projector_phase(width,good_qubits,good_state,0.)
    circuit=preparation.copy()
    circuit.name="fixed-point amplification"
    inverse=preparation.inverse()
    for alpha,beta in zip(schedule.alphas,schedule.betas,strict=True):
        circuit.compose(projector_phase(width,good_qubits,good_state,beta),inplace=True)
        circuit.compose(inverse,inplace=True)
        circuit.compose(projector_phase(width,range(width),0,-alpha),inplace=True)
        circuit.compose(preparation,inplace=True)
        circuit.global_phase+=math.pi
    circuit.metadata={
        **schedule.as_metadata(), "good_qubits":list(good_qubits), "good_state":good_state,
        "good_state_bit_order":"good_qubits[0] is the least significant selected bit",
        "source_preparation_operations":source_count,
        "preparation_calls":schedule.rounds+1,"inverse_preparation_calls":schedule.rounds,
        "good_projector_phases":schedule.rounds,"initial_zero_projector_phases":schedule.rounds,
        "logical_initialization_qubits_assumed":width,
        "physical_initialization_cost_included":False,
        "source_preflight_proxy":proxy,
    }
    if measure:circuit.measure_all()
    return circuit


def lower_fixed_point(circuit: QuantumCircuit) -> QuantumCircuit:
    """Exact-angle primitive lowering, not Clifford+T rotation synthesis."""
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("Supply an explicit QuantumCircuit")
    measured = False
    for item in circuit.data:
        if item.operation.name == "measure":
            measured = True
        elif measured and item.operation.name != "barrier":
            raise ValueError("Only final measurements are supported")
    coherent = circuit.remove_final_measurements(inplace=False)
    _validate_preparation(coherent)
    compiled=transpile(circuit,basis_gates=list(PRIMITIVE_BASIS),optimization_level=0,
                       seed_transpiler=7,qubits_initially_zero=False)
    if len(compiled.data)>MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("Compiled circuit exceeds the primitive-export cap")
    if any(item.operation.name not in {*PRIMITIVE_BASIS,"measure","barrier"} for item in compiled.data):
        raise ValueError("Primitive lowering retained an unsupported operation")
    return compiled


def good_probability(circuit: QuantumCircuit, good_qubits: Sequence[int], good_state: int) -> float:
    good_qubits = tuple(good_qubits)
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("Supply an explicit QuantumCircuit")
    if circuit.num_qubits>MAX_SIM_QUBITS:
        raise ValueError("Statevector probability exceeds the bounded simulator width")
    _validate_preparation(circuit)
    projector_phase(circuit.num_qubits,good_qubits,good_state,0.)
    probabilities=Statevector.from_instruction(circuit).probabilities()
    return float(sum(value for index,value in enumerate(probabilities)
                     if all(((index>>wire)&1)==((good_state>>bit)&1)
                            for bit,wire in enumerate(good_qubits))))
