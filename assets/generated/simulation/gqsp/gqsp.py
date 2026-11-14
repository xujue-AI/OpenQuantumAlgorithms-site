# SPDX-License-Identifier: MIT
"""Original supplied-angle GQSP circuit prototype; numerical validation pending.

Signal: D(W)=|0><0| tensor W + |1><1| tensor I. Ancilla q0 is LSB.
Chronology: E0,D,E1,...,D,Ed; E=RZ(phi) RY(2 theta) RZ(lam).
Thus V=Ed D ... E1 D E0 and first-column blocks are P(W),Q(W).
This forward constructor is not an inverse phase compiler or a complete
Hamiltonian-simulation algorithm. See Motlagh--Wiebe, PRX Quantum5,020368,
doi:10.1103/PRXQuantum.5.020368; arXiv2308.01501v2, Eqs5--11.
Our determinant-one Euler convention differs from the paper's Eq6 by a
specified global factor; polynomial and circuit phases are never aligned away.
Qiskit2.6.2 is Apache-2.0. This source is original MIT code.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import cmath
import math

from qiskit import QuantumCircuit

MAX_DATA_QUBITS = 64
MAX_DEGREE = 64
MAX_SOURCE_OPERATIONS = 120_000
MAX_DENSE_TOTAL_QUBITS = 4
MAX_DENSE_OPERATIONS = 5_000
MAX_ANGLE_ABS = 32 * math.pi
MIN_CONDITIONAL_PROBABILITY = 1e-14
PRIMITIVE_NAMES = frozenset({"x", "h", "s", "t", "tdg", "p", "ry", "rz", "cx", "reset", "measure"})


@dataclass(frozen=True)
class SignalSpec:
    """W=e^(i chi) CZ_chain (tensor_j RZ(2z_j) RY(2y_j))."""

    n: int
    y: tuple[float, ...]
    z: tuple[float, ...]
    chi: float = 0.0


@dataclass(frozen=True)
class EulerRotation:
    theta: float
    phi: float
    lam: float


def _width(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_DATA_QUBITS:
        raise ValueError(f"data width must be an integer in [1,{MAX_DATA_QUBITS}]")
    return n


def _angle(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("angles must be finite real Python numbers")
    value = float(value)
    if not math.isfinite(value) or abs(value) > MAX_ANGLE_ABS:
        raise ValueError("angle is nonfinite or outside the declared 32pi range")
    return value


def signal_spec(n: int, y: Sequence[float], z: Sequence[float], chi: float = 0.0) -> SignalSpec:
    n = _width(n)
    if not isinstance(y, Sequence) or not isinstance(z, Sequence) or len(y) != n or len(z) != n:
        raise ValueError("signal has exactly one Y and Z angle per active data wire")
    return SignalSpec(n, tuple(_angle(a) for a in y), tuple(_angle(a) for a in z), _angle(chi))


def _signal(spec: SignalSpec) -> SignalSpec:
    if not isinstance(spec, SignalSpec):
        raise ValueError("use a SignalSpec with explicit finite primitive access")
    return signal_spec(spec.n, spec.y, spec.z, spec.chi)


def _rotations(values: Sequence[EulerRotation]) -> tuple[EulerRotation, ...]:
    if not isinstance(values, Sequence) or not 1 <= len(values) <= MAX_DEGREE + 1:
        raise ValueError(f"supply 1 to {MAX_DEGREE+1} Euler rotations")
    result = []
    for value in values:
        if not isinstance(value, EulerRotation):
            raise ValueError("rotation entries must be EulerRotation objects")
        result.append(EulerRotation(_angle(value.theta), _angle(value.phi), _angle(value.lam)))
    return tuple(result)


def source_operations(n: int, degree: int, input_mode: str = "zero", measured: bool = False) -> int:
    """Exact named-operation count for this unoptimized construction, not physical work."""
    n = _width(n)
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= MAX_DEGREE:
        raise ValueError("degree outside the declared generator domain")
    if input_mode not in {"zero", "one", "plus_i", "plus_i_x"} or (input_mode == "plus_i_x" and n < 2):
        raise ValueError("invalid implemented input preparation")
    prep = {"zero": 0, "one": 1, "plus_i": 2, "plus_i_x": 3}[input_mode]
    # D: two wrapper X, 4 operations/controlled rotation, 17/CCZ, one P(chi).
    count = degree * (25*n-14) + 3*(degree+1) + prep
    return count + (2*(n+1) if measured else 0)


def _controlled_rotation(circuit: QuantumCircuit, axis: str, angle: float, control: int, target: int) -> None:
    # Chronological R(angle/2),CX,R(-angle/2),CX gives controlled R(angle).
    rotation = circuit.ry if axis == "y" else circuit.rz
    rotation(angle/2, target)
    circuit.cx(control, target)
    rotation(-angle/2, target)
    circuit.cx(control, target)


def _ccz(circuit: QuantumCircuit, a: int, b: int, c: int) -> None:
    # Exact phase polynomial: 4abc=a+b+c-(a^b)-(a^c)-(b^c)+(a^b^c).
    # All parity work is uncomputed in existing wires; no borrowed workspace.
    for q in (a, b, c):
        circuit.t(q)
    for control, target in ((a, b), (a, c), (b, c)):
        circuit.cx(control, target)
        circuit.tdg(target)
        circuit.cx(control, target)
    circuit.cx(a, c)
    circuit.cx(b, c)
    circuit.t(c)
    circuit.cx(b, c)
    circuit.cx(a, c)


def append_signal(circuit: QuantumCircuit, spec: SignalSpec, data: Sequence[int]) -> None:
    """Append W with explicit global phase; no dense gate or free oracle."""
    spec = _signal(spec)
    if len(data) != spec.n or len(set(data)) != len(data):
        raise ValueError("data mapping must cover distinct active wires")
    for q, y, z in zip(data, spec.y, spec.z):
        circuit.ry(2*y, q)
        circuit.rz(2*z, q)
    for a, b in zip(data[:-1], data[1:]):
        circuit.h(b)
        circuit.cx(a, b)
        circuit.h(b)
    # P(chi) X P(chi) X is e^(i chi) I; source/export overall phase stays zero.
    circuit.p(spec.chi, data[0])
    circuit.x(data[0])
    circuit.p(spec.chi, data[0])
    circuit.x(data[0])


def append_controlled_signal(circuit: QuantumCircuit, spec: SignalSpec, control: int,
                             data: Sequence[int], control_state: int = 0) -> None:
    """Literal conditional signal. State1 is exposed only as a distinct convention."""
    spec = _signal(spec)
    if control_state not in (0, 1) or isinstance(control_state, bool) or not isinstance(control_state,int):
        raise ValueError("control state must be integer zero or one")
    if len(data) != spec.n or len(set(data)) != len(data) or control in data:
        raise ValueError("control/data mapping must contain distinct wires")
    if control_state == 0:
        circuit.x(control)
    for q, y, z in zip(data, spec.y, spec.z):
        _controlled_rotation(circuit, "y", 2*y, control, q)
        _controlled_rotation(circuit, "z", 2*z, control, q)
    for a, b in zip(data[:-1], data[1:]):
        _ccz(circuit, control, a, b)
    # The signal's global phase is a CONDITIONAL phase inside this wrapper.
    circuit.p(spec.chi, control)
    if control_state == 0:
        circuit.x(control)


def append_euler(circuit: QuantumCircuit, rotation: EulerRotation, ancilla: int = 0) -> None:
    rotation = _rotations((rotation,))[0]
    circuit.rz(rotation.lam, ancilla)
    circuit.ry(2*rotation.theta, ancilla)
    circuit.rz(rotation.phi, ancilla)


def build_signal(spec: SignalSpec) -> QuantumCircuit:
    spec = _signal(spec)
    circuit = QuantumCircuit(spec.n, name="explicit_unitary_signal")
    append_signal(circuit, spec, tuple(range(spec.n)))
    circuit.metadata = {"signal_global_phase": spec.chi, "source_operations": 5*spec.n+1,
                        "signal_definition": "exp(i chi) CZ_chain tensor[RZ(2z) RY(2y)]"}
    return circuit


def build_gqsp(spec: SignalSpec, rotations: Sequence[EulerRotation]) -> QuantumCircuit:
    spec, rotations = _signal(spec), _rotations(rotations)
    degree = len(rotations)-1
    expected = source_operations(spec.n, degree)
    if expected > MAX_SOURCE_OPERATIONS:
        raise ValueError("source-operation preflight exceeds the materialization cap")
    circuit = QuantumCircuit(spec.n+1, name="supplied_angle_gqsp")
    append_euler(circuit, rotations[0])
    for rotation in rotations[1:]:
        append_controlled_signal(circuit, spec, 0, tuple(range(1, spec.n+1)))
        append_euler(circuit, rotation)
    circuit.metadata = {"status": "given-angle prototype; validation requires a recorded run",
                        "data_qubits": spec.n, "degree_upper_bound": degree, "signal_calls": degree,
                        "ancilla_wire": 0, "source_operations": expected,
                        "chronology": "E0,D(W),E1,...,D(W),Ed", "signal_control_state": 0,
                        "euler_convention": "RZ(phi) RY(2theta) RZ(lambda)",
                        "polynomial_blocks": "P=matrix[0::2,0::2];Q=matrix[1::2,0::2]",
                        "overall_phase": "zero; signal phase retained by literal P gate",
                        "inverse_phase_compiler": False, "hamiltonian_simulation_complete": False}
    return circuit


def _prepare_data(circuit: QuantumCircuit, input_mode: str, n: int) -> None:
    source_operations(n, 0, input_mode)
    if input_mode == "one":
        circuit.x(1)
    elif input_mode in {"plus_i", "plus_i_x"}:
        circuit.h(1)
        circuit.s(1)
        if input_mode == "plus_i_x":
            circuit.x(2)


def build_prepared_gqsp(spec: SignalSpec, rotations: Sequence[EulerRotation], input_mode: str = "zero") -> QuantumCircuit:
    spec, rotations = _signal(spec), _rotations(rotations)
    expected = source_operations(spec.n, len(rotations)-1, input_mode)
    if expected > MAX_SOURCE_OPERATIONS:
        raise ValueError("prepared source-operation cap exceeded")
    circuit = QuantumCircuit(spec.n+1, name="prepared_gqsp")
    _prepare_data(circuit, input_mode, spec.n)
    circuit.compose(build_gqsp(spec, rotations), inplace=True)
    circuit.metadata = {"input_mode": input_mode, "source_operations": expected}
    return circuit


def build_measured_trial(spec: SignalSpec, rotations: Sequence[EulerRotation], input_mode: str = "zero") -> QuantumCircuit:
    spec, rotations = _signal(spec), _rotations(rotations)
    expected = source_operations(spec.n, len(rotations)-1, input_mode, measured=True)
    if expected > MAX_SOURCE_OPERATIONS:
        raise ValueError("complete-trial source-operation cap exceeded")
    circuit = QuantumCircuit(spec.n+1, spec.n+1, name="complete_gqsp_trial")
    circuit.reset(range(spec.n+1))
    circuit.compose(build_prepared_gqsp(spec, rotations, input_mode), inplace=True)
    for q in range(spec.n+1):
        circuit.measure(q, q)
    circuit.metadata = {"status": "complete logical trial; physical synthesis unperformed",
                        "input_mode": input_mode, "source_operations": expected,
                        "accepted_classical_bit": 0, "acceptance_value": 0,
                        "readout_mapping": {str(q): q for q in range(spec.n+1)},
                        "resets": spec.n+1, "measurements": spec.n+1}
    return circuit


def euler_entries(rotation: EulerRotation) -> tuple[complex, complex, complex, complex]:
    r = _rotations((rotation,))[0]
    c, s = math.cos(r.theta), math.sin(r.theta)
    return (c*cmath.exp(-.5j*(r.phi+r.lam)), -s*cmath.exp(-.5j*(r.phi-r.lam)),
            s*cmath.exp(.5j*(r.phi-r.lam)), c*cmath.exp(.5j*(r.phi+r.lam)))


def polynomial_coefficients(rotations: Sequence[EulerRotation]) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    """Forward Euler recurrence in ascending powers; floating construction, no certificate."""
    values = _rotations(rotations)
    a, _, c, _ = euler_entries(values[0])
    p, q = [a], [c]
    for rotation in values[1:]:
        a, b, c, d = euler_entries(rotation)
        p_new, q_new = [0j]*(len(p)+1), [0j]*(len(q)+1)
        for k, value in enumerate(p):
            p_new[k+1] += a*value
            q_new[k+1] += c*value
        for k, value in enumerate(q):
            p_new[k] += b*value
            q_new[k] += d*value
        p, q = p_new, q_new
    return tuple(p), tuple(q)


def polynomial_value(coefficients: Sequence[complex], z: complex) -> complex:
    if not isinstance(coefficients, Sequence) or not 1 <= len(coefficients) <= MAX_DEGREE+1:
        raise ValueError("bounded coefficient sequence required")
    z = complex(z)
    if not math.isfinite(z.real) or not math.isfinite(z.imag) or abs(z) > 2:
        raise ValueError("scalar polynomial evaluation restricted to finite |z|<=2")
    value = 0j
    for coefficient in reversed(coefficients):
        coefficient = complex(coefficient)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("nonfinite polynomial coefficient")
        value = value*z+coefficient
    return value


def require_small(spec: SignalSpec, rotations: Sequence[EulerRotation]) -> None:
    spec, rotations = _signal(spec), _rotations(rotations)
    if spec.n+1 > MAX_DENSE_TOTAL_QUBITS or source_operations(spec.n, len(rotations)-1) > MAX_DENSE_OPERATIONS:
        raise ValueError("dense simulation is separate and bounded to four total wires/5000 operations")
