# SPDX-License-Identifier: MIT
"""Original fixed two-term multiproduct circuits and bounded references.

The generator uses primitive rotations/Cliffords; Hamiltonian and state matrices
exist only in explicitly bounded reference functions. Wire q[0] is the least
significant bit. This is a fixed fourth-order Richardson family, not the
optimized high-order/polylogarithmic-accuracy multiproduct algorithm.
"""
from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

MAX_SYSTEM_QUBITS = 61
MAX_STEPS = 4096
MAX_SOURCE_OPERATIONS = 200_000
MAX_REFERENCE_SYSTEM_QUBITS = 3
PRIMITIVES = {"h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"}
METHODS = ("lcu", "oaa", "oaa-channel", "fine", "coarse")
COEFFICIENTS = (Fraction(4, 3), Fraction(-1, 3), Fraction(1, 6), Fraction(-1, 6))
PROBABILITIES = (Fraction(2, 3), Fraction(1, 6), Fraction(1, 12), Fraction(1, 12))


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    if value.bit_length() > 128:
        raise ValueError(f"{name} exceeds the scalar encoding limit")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and real")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be finite and real") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite and real")
    return converted


def _half(angle):
    half = angle / 2
    if angle and (not half or 2 * half != angle):
        raise ValueError("a nonzero controlled rotation loses its exact half angle")
    return half


def _angles(time, steps, J, h):
    dt = time / steps
    if time and not dt:
        raise ValueError("nonzero step time underflows")
    values = []
    for coefficient, scale in ((J, 1), (h, 2)):
        product = coefficient * dt
        if coefficient and dt and not product:
            raise ValueError("nonzero rotation product underflows")
        angle = scale * product
        if not math.isfinite(angle):
            raise ValueError("rotation angle is not finite")
        _half(angle)
        values.append(angle)
    return tuple(values)


def _parameters(n, time, steps, J, h, *, materialize):
    _integer(n, "n_system", maximum=MAX_SYSTEM_QUBITS if materialize else None)
    _integer(steps, "steps", maximum=MAX_STEPS if materialize else None)
    time, J, h = (_finite(time, "time"), _finite(J, "J"), _finite(h, "h"))
    if materialize:
        _angles(time, steps, J if n > 1 else 0., h)
        _angles(time, 2 * steps, J if n > 1 else 0., h)
    return n, time, steps, J, h


def _ccx7(circuit, a, b, target):
    """Exact Toffoli with seven T/Tdg gates, six CX and two H."""
    circuit.h(target)
    circuit.cx(b, target)
    circuit.tdg(target)
    circuit.cx(a, target)
    circuit.t(target)
    circuit.cx(b, target)
    circuit.tdg(target)
    circuit.cx(a, target)
    circuit.t(b)
    circuit.t(target)
    circuit.h(target)
    circuit.cx(a, b)
    circuit.t(a)
    circuit.tdg(b)
    circuit.cx(a, b)


def _cry(circuit, angle, control, target):
    half = _half(angle)
    circuit.ry(half, target)
    circuit.cx(control, target)
    circuit.ry(-half, target)
    circuit.cx(control, target)


def _crz(circuit, angle, control, target):
    half = _half(angle)
    circuit.rz(half, target)
    circuit.cx(control, target)
    circuit.rz(-half, target)
    circuit.cx(control, target)


def _append_prepare(circuit, low, high):
    # |high,low>: fine, coarse, +I, -I.
    circuit.ry(2 * math.asin(math.sqrt(1 / 6)), high)
    circuit.x(high)
    _cry(circuit, 2 * math.asin(math.sqrt(1 / 5)), high, low)
    circuit.x(high)
    _cry(circuit, math.pi / 2, high, low)


def _append_inverse(circuit, body):
    if set(body.count_ops()) - PRIMITIVES:
        raise ValueError("inverse requires a primitive unitary circuit")
    swaps = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
    for instruction in reversed(body.data):
        name = instruction.operation.name
        wires = [body.find_bit(q).index for q in instruction.qubits]
        if name in {"rx", "ry", "rz"}:
            getattr(circuit, name)(-float(instruction.operation.params[0]), wires[0])
        elif name == "cx":
            circuit.cx(*wires)
        else:
            getattr(circuit, swaps.get(name, name))(wires[0])
    circuit.global_phase -= float(body.global_phase)


def _selected_strang(circuit, n, time, steps, J, h, label, low, high, work):
    if not time or not ((n > 1 and J) or h):
        return
    zeros = [wire for bit, wire in enumerate((low, high)) if not (label >> bit) & 1]
    for wire in zeros:
        circuit.x(wire)
    _ccx7(circuit, low, high, work)
    zz_angle, x_angle = _angles(time, steps, J if n > 1 else 0., h)
    for _ in range(steps):
        for half_layer in (0, 1):
            if J:
                for q in range(n - 1):
                    circuit.cx(q, q + 1)
                    _crz(circuit, zz_angle, work, q + 1)
                    circuit.cx(q, q + 1)
            if half_layer == 0 and h:
                for q in range(n):
                    circuit.h(q)
                    _crz(circuit, x_angle, work, q)
                    circuit.h(q)
    _ccx7(circuit, low, high, work)
    for wire in reversed(zeros):
        circuit.x(wire)


def _reflect_zero(circuit, low, high):
    circuit.x(low)
    circuit.x(high)
    circuit.h(high)
    circuit.cx(low, high)
    circuit.h(high)
    circuit.x(high)
    circuit.x(low)


def scalar_counts(n=2, steps=4, method="oaa", *, time=1., J=1., h=.7):
    """Fixed-size exact source counts; never creates terms, circuits or matrices."""
    n, time, steps, J, h = _parameters(n, time, steps, J, h, materialize=False)
    if method not in METHODS:
        raise ValueError("unknown multiproduct method")
    z = n - 1 if J and time else 0
    x = n if h and time else 0
    if method in ("fine", "coarse"):
        r = steps * (2 if method == "fine" else 1)
        gates = {"cx": 4 * z * r, "rz": 2 * z * r, "rx": x * r}
        qubits = n
    else:
        active = bool(z or x)
        gates = {
            "h": 6 * steps * x + (8 if active else 0),
            "x": 4 + (6 if active else 0), "z": 1,
            "t": 16 if active else 0, "tdg": 12 if active else 0,
            "cx": 3 * steps * (8 * z + 2 * x) + 8 + (24 if active else 0),
            "ry": 10, "rz": 6 * steps * (2 * z + x),
        }
        if method != "lcu":
            forward_t, forward_tdg = gates.get("t", 0), gates.get("tdg", 0)
            gates = {name: 3 * count for name, count in gates.items()}
            gates["t"] = 2 * forward_t + forward_tdg
            gates["tdg"] = 2 * forward_tdg + forward_t
            gates["h"] += 4
            gates["x"] += 8
            gates["cx"] += 2
        qubits = n + 3
    gates = {name: count for name, count in gates.items() if count}
    return {
        "method": method, "system_qubits": n, "logical_qubits": qubits,
        "selector_qubits": 0 if method in ("fine", "coarse") else 2,
        "clean_work_qubits": 0 if method in ("fine", "coarse") else 1,
        "steps": steps, "time": time, "J": J, "h": h,
        "unitary_gate_counts": gates, "unitary_operations": sum(gates.values()),
        "branch_steps_per_B": 3 * steps if method not in ("fine", "coarse") else None,
        "B_calls": 0 if method in ("fine", "coarse") else (1 if method == "lcu" else 3),
        "scalar_only": True, "physical_resource_estimate": None,
    }


def build_evolution(n=2, time=1., steps=4, method="oaa", J=1., h=.7):
    """Explicit grouped-ZZ/X Strang branches, coefficient preparation and OAA."""
    n, time, steps, J, h = _parameters(n, time, steps, J, h, materialize=True)
    ledger = scalar_counts(n, steps, method, time=time, J=J, h=h)
    if ledger["unitary_operations"] > MAX_SOURCE_OPERATIONS:
        raise ValueError("multiproduct source preflight exceeds 200000 operations")
    circuit = QuantumCircuit(ledger["logical_qubits"], name=f"MPF {method} r={steps}")
    if method in ("fine", "coarse"):
        r = steps * (2 if method == "fine" else 1)
        zz_angle, x_angle = _angles(time, r, J if n > 1 else 0., h)
        if time:
            for _ in range(r):
                for half_layer in (0, 1):
                    if J:
                        for q in range(n - 1):
                            circuit.cx(q, q + 1)
                            circuit.rz(zz_angle, q + 1)
                            circuit.cx(q, q + 1)
                    if half_layer == 0 and h:
                        for q in range(n):
                            circuit.rx(x_angle, q)
    else:
        low, high, work = n, n + 1, n + 2
        prepare = QuantumCircuit(n + 3)
        _append_prepare(prepare, low, high)
        body = QuantumCircuit(n + 3)
        body.compose(prepare, inplace=True)
        body.z(low)
        _selected_strang(body, n, time, 2 * steps, J, h, 0, low, high, work)
        _selected_strang(body, n, time, steps, J, h, 1, low, high, work)
        _append_inverse(body, prepare)
        circuit.compose(body, inplace=True)
        if method != "lcu":
            _reflect_zero(circuit, low, high)
            _append_inverse(circuit, body)
            _reflect_zero(circuit, low, high)
            circuit.compose(body, inplace=True)
            circuit.global_phase += math.pi
    actual = {name: int(count) for name, count in circuit.count_ops().items()}
    if actual != ledger["unitary_gate_counts"]:
        raise AssertionError("materialized source differs from scalar operation accounting")
    if set(actual) - PRIMITIVES:
        raise AssertionError("multiproduct circuit contains an opaque/composite operation")
    circuit.metadata = {
        "method": method, "n_system": n, "steps": steps, "time": time, "J": J, "h": h,
        "selector_wires": [] if method in ("fine", "coarse") else [n, n + 1],
        "work_wire": None if method in ("fine", "coarse") else n + 2,
        "work_promise": "work starts in zero and is restored for every selector/system input",
        "normalization": None if method in ("fine", "coarse") else "2",
        "coefficient_strings": [str(x) for x in COEFFICIENTS],
        "probability_strings": [str(x) for x in PROBABILITIES],
        "source_operations": len(circuit.data), "source_gate_counts": actual,
        "source_depth": circuit.depth(), "global_phase_radians": float(circuit.global_phase),
        "physical_estimate": None, "synthesis_performed": False,
    }
    return circuit


def preflight(n=2, time=1., steps=4, method="oaa", J=1., h=.7, *, initial_basis=0, system_readout=True):
    """Validate a complete source request without allocating a circuit."""
    n, time, steps, J, h = _parameters(n, time, steps, J, h, materialize=True)
    _integer(initial_basis, "initial_basis", minimum=0, maximum=(1 << n) - 1)
    if not isinstance(system_readout, bool):
        raise ValueError("system_readout must be bool")
    counts = scalar_counts(n, steps, method, time=time, J=J, h=h)
    measured = ((n if system_readout else 0) if method == "oaa-channel" or method in ("fine", "coarse")
                else (counts["logical_qubits"] if system_readout else 3))
    overhead = counts["logical_qubits"] + initial_basis.bit_count() + measured + (3 if method == "oaa-channel" else 0)
    if counts["unitary_operations"] + overhead > MAX_SOURCE_OPERATIONS:
        raise ValueError("complete multiproduct attempt exceeds200000 operations")
    return counts | {"complete_source_operations": counts["unitary_operations"] + overhead}


def build_attempt(n=2, time=1., steps=4, method="oaa", J=1., h=.7, *, initial_basis=0, system_readout=True):
    """Complete fresh-input experiment, including reset and every readout/discard."""
    n, time, steps, J, h = _parameters(n, time, steps, J, h, materialize=True)
    _integer(initial_basis, "initial_basis", minimum=0)
    if initial_basis >= 1 << n:
        raise ValueError("initial basis is outside the system register")
    if not isinstance(system_readout, bool):
        raise ValueError("system_readout must be bool")
    body = build_evolution(n, time, steps, method, J, h)
    measured = ((n if system_readout else 0) if method == "oaa-channel" or method in ("fine", "coarse")
                else (body.num_qubits if system_readout else 3))
    overhead = body.num_qubits + initial_basis.bit_count() + measured + (3 if method == "oaa-channel" else 0)
    if len(body.data) + overhead > MAX_SOURCE_OPERATIONS:
        raise ValueError("complete multiproduct attempt exceeds 200000 operations")
    circuit = (QuantumCircuit(body.num_qubits, measured, name=f"Complete MPF {method}") if measured
               else QuantumCircuit(body.num_qubits, name=f"Complete MPF {method}"))
    for q in range(body.num_qubits):
        circuit.reset(q)
    for q in range(n):
        if (initial_basis >> q) & 1:
            circuit.x(q)
    circuit.compose(body, inplace=True)
    if method == "oaa-channel":
        for q in range(n, n + 3):
            circuit.reset(q)
        if system_readout:
            for q in range(n):
                circuit.measure(q, q)
        acceptance = "all outcomes; selector/work discarded by explicit reset"
    else:
        readout_wires = list(range(body.num_qubits)) if system_readout else (list(range(n, n + 3)) if method in ("lcu", "oaa") else [])
        for bit, q in enumerate(readout_wires):
            circuit.measure(q, bit)
        acceptance = "all outcomes" if method in ("fine", "coarse") else "accept only selector and work bits all zero"
    measured_wires = list(range(n)) if method == "oaa-channel" and system_readout else [] if method == "oaa-channel" else readout_wires
    circuit.metadata = dict(body.metadata, initial_basis=initial_basis, measured_wires=measured_wires,
                            acceptance=acceptance, source_operations=len(circuit.data),
                            source_gate_counts={k: int(v) for k, v in circuit.count_ops().items()},
                            source_depth=circuit.depth(), initial_reset_count=body.num_qubits,
                            final_discard_reset_count=3 if method == "oaa-channel" else 0,
                            system_readout=system_readout,
                            output_contract="measured Ztrial" if system_readout else "unmeasured systemstate with acceptance/discard instrument")
    return circuit


def build_state_preparation(n=2, time=1., steps=4, method="oaa", J=1., h=.7, *, initial_basis=0):
    """Complete known-input state instrument; system qubits remain unmeasured."""
    if method not in ("lcu", "oaa", "oaa-channel"):
        raise ValueError("state instrument requires a coherent multiproduct method")
    return build_attempt(n, time, steps, method, J, h, initial_basis=initial_basis, system_readout=False)


def combine_classical_means(fine, coarse):
    """Raw Richardson estimate; individual physical means are bounded, sum isn't."""
    fine, coarse = _finite(fine, "fine_mean"), _finite(coarse, "coarse_mean")
    if not -1 <= fine <= 1 or not -1 <= coarse <= 1:
        raise ValueError("individual Z means must be in[-1,1]")
    return (4 * fine - coarse) / 3


def _pauli(label):
    matrices = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
                "Z": np.diag([1., -1.]).astype(complex)}
    matrix = np.ones((1, 1), dtype=complex)
    for character in label:
        matrix = np.kron(matrix, matrices[character])
    return matrix


def reference_operators(n=2, time=1., steps=4, J=1., h=.7):
    """Independent Kronecker Hamiltonian, exponentials and selector completion."""
    n, time, steps, J, h = _parameters(n, time, steps, J, h, materialize=True)
    if n > MAX_REFERENCE_SYSTEM_QUBITS:
        raise ValueError("multiproduct dense references are capped at three system qubits")
    d = 1 << n
    A_H, B_H = np.zeros((d, d), complex), np.zeros((d, d), complex)
    for q in range(n - 1):
        label = ["I"] * n
        label[n - 1 - q] = label[n - 2 - q] = "Z"
        A_H += J * _pauli("".join(label))
    for q in range(n):
        label = ["I"] * n
        label[n - 1 - q] = "X"
        B_H += h * _pauli("".join(label))
    exact = expm(-1j * time * (A_H + B_H))
    def strang(r):
        half = expm(-1j * A_H * time / (2 * r))
        return np.linalg.matrix_power(half @ expm(-1j * B_H * time / r) @ half, r)
    fine, coarse = strang(2 * steps), strang(steps)
    A = (4 * fine - coarse) / 3
    F = (3 * A - A @ A.conj().T @ A) / 2
    high = np.array([[math.sqrt(5 / 6), -math.sqrt(1 / 6)],
                     [math.sqrt(1 / 6), math.sqrt(5 / 6)]])
    low0 = np.array([[math.sqrt(4 / 5), -math.sqrt(1 / 5)],
                    [math.sqrt(1 / 5), math.sqrt(4 / 5)]])
    low1 = np.array([[1., -1.], [1., 1.]]) / math.sqrt(2)
    conditional = np.zeros((4, 4), dtype=complex)
    conditional[:2, :2], conditional[2:, 2:] = low0, low1
    prepare = conditional @ np.kron(high, np.eye(2))
    select = np.zeros((4 * d, 4 * d), dtype=complex)
    for label, unitary in enumerate((fine, -coarse, np.eye(d), -np.eye(d))):
        select[label * d:(label + 1) * d, label * d:(label + 1) * d] = unitary
    P = np.kron(prepare, np.eye(d))
    B = P.conj().T @ select @ P
    R = np.kron(np.diag([-1, 1, 1, 1]), np.eye(d))
    G = -B @ R @ B.conj().T @ R @ B
    return {"exact": exact, "fine": fine, "coarse": coarse, "A": A, "F": F,
            "B_clean": B, "G_clean": G, "prepare": prepare, "H": A_H + B_H}


def _fraction_record(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
        raise ValueError("analytic-bound rational exceeds the 4096-bit export limit")
    return str(value)


def analytic_bounds(n=2, time=1., steps=4, J=1., h=.7):
    """Exact rational ideal-coefficient bounds, separate from their realization."""
    n, time, steps, J, h = _parameters(n, time, steps, J, h, materialize=False)
    j, hx, t = map(Fraction.from_float, (abs(J), abs(h), abs(time)))
    C = Fraction(0) if n == 1 else j * j * hx * (4 * n - 6) / 6 + 4 * j * hx * hx * (n - 1) / 3
    e = 2 * C * t ** 3 / (3 * steps ** 2)
    d = min(Fraction(2), 5 * C * t ** 3 / (4 * steps ** 2))
    x = 4 * d * d / 9
    p = 1 - 3 * x * x / 4 + x ** 3 / 4
    nu = e + 3 * e * e / 2 + e ** 3 / 2
    values = {"strang_C_upper": C, "mpf_operator_error_upper": e,
              "branch_difference_upper": d, "oaa_operator_error_upper": nu,
              "lcu_ideal_acceptance_floor": Fraction(1, 4), "oaa_ideal_acceptance_floor": p}
    return {key: _fraction_record(value) for key, value in values.items()} | {
        "scope": "ideal exact-coefficient/angle mathematics; conservative second-order certificate, not a fourth-order or synthesized-circuit certificate",
        "declared_parameters": {"J": str(Fraction.from_float(J)), "h": str(Fraction.from_float(h)),
                                "time": str(Fraction.from_float(time))},
        "normalization": "2", "uniform_oaa_floor": "25/729",
        "realization_bridge_verified": False,
    }


def _ceil(value):
    return -(-value.numerator // value.denominator)


def log_upper(value):
    """Rational log bound ln(x)<=ceil(log2(x))*7/10 for x>=1."""
    if not isinstance(value, Fraction) or value < 1:
        raise ValueError("rational logarithm argument must be >=1")
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 128:
        raise ValueError("rational logarithm input exceeds128bits")
    k = max(0, value.numerator.bit_length() - value.denominator.bit_length())
    while Fraction(1 << k) < value:
        k += 1
    while k and Fraction(1 << (k - 1)) >= value:
        k -= 1
    return Fraction(7 * k, 10)


def sampling_plan(epsilon=Fraction(3, 100), failure=Fraction(1, 250)):
    if not isinstance(epsilon, Fraction) or not isinstance(failure, Fraction):
        raise ValueError("sampling precision/failure must be exact Fractions")
    if not 0 < epsilon <= 1 or not 0 < failure < 1:
        raise ValueError("invalid sampling precision/failure")
    if max(epsilon.numerator.bit_length(), epsilon.denominator.bit_length(),
           failure.numerator.bit_length(), failure.denominator.bit_length()) > 64:
        raise ValueError("sampling rational exceeds 64-bit input cap")
    single = _ceil(2 * log_upper(2 / failure) / (epsilon * epsilon))
    total = _ceil(2 * Fraction(25, 9) * log_upper(2 / failure) / (epsilon * epsilon))
    fine, coarse = _ceil(Fraction(4, 5) * total), _ceil(Fraction(1, 5) * total)
    if max(single, fine + coarse) > 100_000_000:
        raise ValueError("projected sampling population exceeds100million")
    return {"epsilon": str(epsilon), "failure": str(failure), "log_upper": str(log_upper(2 / failure)),
            "single_accepted_samples": single, "classical_total_target": total,
            "classical_fine_samples": fine, "classical_coarse_samples": coarse,
            "classical_total_samples": fine + coarse, "coefficient_l1": "5/3",
            "coefficient_conditioning_square": "25/9",
            "classical_sum_c2_over_N": str(Fraction(16, 9 * fine) + Fraction(1, 9 * coarse)),
            "population_executed": False, "log_bound_proof": "ln2<7/10, because the first five nonnegative terms of exp(7/10) exceed2"}


def _sqrt_lower(value, bits=32):
    denominator = 1 << bits
    numerator = math.isqrt(value.numerator * denominator * denominator // value.denominator)
    return Fraction(numerator, denominator)


def _sqrt_upper(value, bits=32):
    denominator = 1 << bits
    squared_upper = _ceil(value * denominator * denominator)
    numerator = math.isqrt(squared_upper)
    if numerator * numerator < squared_upper:
        numerator += 1
    return Fraction(numerator, denominator)


def retry_plan(p_floor, samples, failure=Fraction(1, 1000)):
    if not isinstance(p_floor, Fraction) or not 0 < p_floor <= 1:
        raise ValueError("retry acceptance floor must be an exact probability")
    _integer(samples, "samples", maximum=100_000_000)
    if not isinstance(failure, Fraction) or not 0 < failure < 1:
        raise ValueError("retry failure must be an exact Fraction")
    if max(p_floor.numerator.bit_length(), p_floor.denominator.bit_length()) > 128:
        raise ValueError("retry floor exceeds128bits")
    if max(failure.numerator.bit_length(), failure.denominator.bit_length()) > 64:
        raise ValueError("retry failure exceeds64bits")
    q, power, rounds = 1 - p_floor, Fraction(1), 0
    while samples * power > failure:
        rounds += 1
        if rounds > 4096:
            raise ValueError("retry cap exceeds4096")
        power *= q
    expected = (1 - power) / p_floor
    return {"acceptance_floor": str(p_floor), "retry_cap_per_accepted_sample": rounds,
            "capped_complete_attempts": rounds * samples,
            "expected_stopped_attempts_per_sample_upper_float": math.nextafter(float(expected), math.inf),
            "failure_target": str(failure), "failure_bound_upper_float": math.nextafter(float(samples * power), math.inf),
            "fresh_repreparable_input_required": True, "attempts_executed": False}


def resource_proposal(n=2, time=1., steps=4, method="oaa", J=1., h=.7):
    """Complete source/sampling/retry proposal; no synthesis/physical execution."""
    if method not in METHODS:
        raise ValueError("unknown method")
    counts = scalar_counts(n, steps, method, time=time, J=J, h=h)
    plan = sampling_plan()
    bounds = analytic_bounds(n, time, steps, J, h)
    samples = plan["single_accepted_samples"]
    epsilon = Fraction(1, 1000) if method == "lcu" else Fraction(1, 2500)
    retry = None
    if method in ("lcu", "oaa"):
        ideal_p = Fraction(bounds["lcu_ideal_acceptance_floor"] if method == "lcu" else bounds["oaa_ideal_acceptance_floor"])
        amplitude = _sqrt_lower(ideal_p)
        floor = (amplitude - epsilon) ** 2
        retry = retry_plan(floor, samples)
        preparations = retry["capped_complete_attempts"]
        realization_z_bias = 2 * epsilon / (amplitude - epsilon)
    else:
        preparations = samples
        realization_z_bias = 2 * epsilon
    if method in ("fine", "coarse"):
        preparations = plan["classical_fine_samples"] if method == "fine" else plan["classical_coarse_samples"]
        epsilon = Fraction(3, 1000)
        individual_group_realization_z_bias = 2 * epsilon
        realization_z_bias = Fraction(5, 3) * individual_group_realization_z_bias
    e = Fraction(bounds["mpf_operator_error_upper"])
    nu = Fraction(bounds["oaa_operator_error_upper"])
    p1 = Fraction(bounds["oaa_ideal_acceptance_floor"])
    if method == "lcu":
        ideal_Z_bias = min(Fraction(2), 2 * e)
    elif method == "oaa":
        ideal_Z_bias = min(Fraction(2), 2 * nu / _sqrt_lower(p1))
    elif method == "oaa-channel":
        ideal_Z_bias = 2 * _sqrt_upper(min(Fraction(1), nu * nu + 1 - p1))
    else:
        ideal_Z_bias = min(Fraction(8, 3), 2 * e)
    initial_resets = counts["logical_qubits"]
    final_resets = 3 if method == "oaa-channel" else 0
    readout = n if method == "oaa-channel" else counts["logical_qubits"]
    total = {gate: value * preparations for gate, value in counts["unitary_gate_counts"].items()}
    total["reset"] = (initial_resets + final_resets) * preparations
    total["measure"] = readout * preparations
    rotations = sum(counts["unitary_gate_counts"].get(gate, 0) for gate in ("rx", "ry", "rz"))
    classical_group = method in ("fine", "coarse")
    physical_share = Fraction(4, 5) if method == "fine" else Fraction(1, 5) if method == "coarse" else Fraction(1)
    return {"source_counts": counts, "sampling": plan, "analytic_ideal_bounds": bounds,
            "retry": retry, "complete_preparations_capped": preparations,
            "complete_gate_counts_capped": total, "initial_basis": 0,
            "full_attempt_logical_operator_error_request": str(epsilon),
            "per_rotation_operator_error_request": str(epsilon / rotations) if rotations else None,
            "logical_realization_Z_bias_request": str(realization_z_bias),
            "individual_group_realization_Z_bias_request": str(individual_group_realization_z_bias) if classical_group else None,
            "accuracy_contract_scope": "combined (4*fine-coarse)/3 weighted task; neither group alone owns its ideal/statistical/realization bounds" if classical_group else "one complete method task",
            "ideal_algorithm_Z_bias_upper": str(ideal_Z_bias),
            "ideal_algorithm_Z_bias_allowance": "1/100",
            "ideal_algorithm_allowance_supported": ideal_Z_bias <= Fraction(1, 100),
            "statistical_Z_error_request": "3/100", "statistical_failure_request": "1/250",
            "physical_failure_request_whole_workload": "1/200",
            "physical_failure_request_this_group": str(physical_share * Fraction(1, 200)),
            "physical_failure_budget_share": str(physical_share),
            "physical_failure_budget_scope": "fine/coarse allocations sum to the single weighted task's 1/200 budget" if classical_group else "whole complete capped method workload",
            "output": "unconditional reduced-channel Zmean" if method == "oaa-channel" else "postselected Zmean" if method in ("lcu", "oaa") else "one group in weighted classical Zmean",
            "resource_kind": "complete capped logical workload/proposed synthesis and physical budgets",
            "physical_qubits": None, "physical_runtime": None, "synthesis_performed": False,
            "realization_bridge_verified": False, "total_accuracy_verified": False,
            "exclusions": ["rotation synthesis/representation certificate", "physical scheduling/factories/QEC/controller model",
                           "classical coefficient/readout computation latency", "actual projected quantum sampling population"]}


def compare(n=2, time=1., steps=4, J=1., h=.7, *, basis=0):
    """Full small clean-domain operator checks, all accepted blocks and observables."""
    refs = reference_operators(n, time, steps, J, h)
    _integer(basis, "basis", minimum=0, maximum=(1 << n) - 1)
    d = 1 << n
    B = Operator(build_evolution(n, time, steps, "lcu", J, h)).data
    G = Operator(build_evolution(n, time, steps, "oaa", J, h)).data
    fine = Operator(build_evolution(n, time, steps, "fine", J, h)).data
    coarse = Operator(build_evolution(n, time, steps, "coarse", J, h)).data
    if not all(np.isfinite(value).all() for value in (B, G, fine, coarse, *refs.values())):
        raise ValueError("reference/circuit calculation returned a nonfinite matrix")
    X, F = B[:d, :d], G[:d, :d]
    Z = _pauli("I" * (n - 1) + "Z")
    target = refs["exact"][:, basis]
    target_z = float(np.vdot(target, Z @ target).real)
    states = {}
    for method, block in (("lcu", X), ("oaa", F)):
        raw = block[:, basis]
        p = float(np.vdot(raw, raw).real)
        normalized = raw / math.sqrt(p)
        states[method] = {"acceptance": p, "Z": float(np.vdot(normalized, Z @ normalized).real),
                          "state_infidelity": float(max(0., 1 - abs(np.vdot(target, normalized)) ** 2)),
                          "normalized_state_vector_error": float(np.linalg.norm(normalized - target))}
    rho = sum(np.outer(G[label * d:(label + 1) * d, basis], G[label * d:(label + 1) * d, basis].conj()) for label in range(8))
    channel_z = float(np.trace(Z @ rho).real)
    fine_z = float(np.vdot(fine[:, basis], Z @ fine[:, basis]).real)
    coarse_z = float(np.vdot(coarse[:, basis], Z @ coarse[:, basis]).real)
    weighted = (4 * fine_z - coarse_z) / 3
    kraus = [G[label * d:(label + 1) * d, :d] for label in range(8)]
    ref_kraus = [refs["G_clean"][label * d:(label + 1) * d, :d] for label in range(4)]
    choi = sum(np.outer(k.reshape(-1, order="F"), k.reshape(-1, order="F").conj()) for k in kraus)
    ref_choi = sum(np.outer(k.reshape(-1, order="F"), k.reshape(-1, order="F").conj()) for k in ref_kraus)
    for item in states.values():
        item["Z_error"] = abs(item["Z"] - target_z)
    errors = {
        "fine_reference_error": float(np.linalg.norm(fine - refs["fine"], 2)),
        "coarse_reference_error": float(np.linalg.norm(coarse - refs["coarse"], 2)),
        "B_clean_reference_error": float(np.linalg.norm(B[:4 * d, :4 * d] - refs["B_clean"], 2)),
        "G_clean_reference_error": float(np.linalg.norm(G[:4 * d, :4 * d] - refs["G_clean"], 2)),
        "B_work_leakage": float(np.linalg.norm(B[4 * d:, :4 * d], 2)),
        "G_work_leakage": float(np.linalg.norm(G[4 * d:, :4 * d], 2)),
        "lcu_block_error": float(np.linalg.norm(X - refs["A"] / 2, 2)),
        "oaa_block_error": float(np.linalg.norm(F - refs["F"], 2)),
        "B_unitarity_error": float(np.linalg.norm(B.conj().T @ B - np.eye(8 * d), 2)),
        "G_unitarity_error": float(np.linalg.norm(G.conj().T @ G - np.eye(8 * d), 2)),
        "channel_reference_error": float(np.linalg.norm(choi - ref_choi, 2)),
        "channel_trace_preservation_error": float(np.linalg.norm(sum(k.conj().T @ k for k in kraus) - np.eye(d), 2)),
    }
    if max(errors.values()) > 1e-10:
        raise AssertionError("primitive/reference clean-domain agreement failed")
    choi_min = float(np.min(np.linalg.eigvalsh(choi)))
    if choi_min < -1e-10:
        raise AssertionError("unconditional channel Choi matrix is not positive")
    return {"n_system": n, "time": time, "steps": steps, "J": J, "h": h, "basis": basis,
            "performed": True, "reference_scope": "full matrices on at most3system+3access wires",
            "implementation_errors": errors, "exact_Z": target_z, "conditional": states,
            "fine_operator_error": float(np.linalg.norm(fine - refs["exact"], 2)),
            "coarse_operator_error": float(np.linalg.norm(coarse - refs["exact"], 2)),
            "A_operator_error": float(np.linalg.norm(refs["A"] - refs["exact"], 2)),
            "F_operator_error": float(np.linalg.norm(refs["F"] - refs["exact"], 2)),
            "lcu_min_acceptance_numeric": float(np.min(np.linalg.eigvalsh(X.conj().T @ X))),
            "oaa_min_acceptance_numeric": float(np.min(np.linalg.eigvalsh(F.conj().T @ F))),
            "unconditional_oaa_Z": channel_z, "unconditional_oaa_Z_error": abs(channel_z - target_z),
            "unconditional_oaa_trace": float(np.trace(rho).real),
            "unconditional_oaa_choi_min_eigenvalue": choi_min,
            "unconditional_oaa_trace_distance": float(np.linalg.norm(rho - np.outer(target, target.conj()), ord="nuc") / 2),
            "classical_fine_Z": fine_z, "classical_coarse_Z": coarse_z,
            "classical_weighted_Z": weighted, "classical_weighted_Z_error": abs(weighted - target_z),
            "analytic_ideal_bounds": analytic_bounds(n, time, steps, J, h),
            "synthesis_performed": False, "physical_estimation_performed": False,
            "projected_trial_population_executed": False}


def opposite_identity_fixture(amplified=False):
    """Explicit phase counterexample outside the common-H Strang family."""
    n, low, high, work = 1, 1, 2, 3
    prepare = QuantumCircuit(4)
    _append_prepare(prepare, low, high)
    body = QuantumCircuit(4)
    body.compose(prepare, inplace=True)
    body.z(low)
    body.x(high)
    _ccx7(body, low, high, work)
    # Controlled XZXZ is -I only on selected coarse branch.
    for _ in range(2):
        body.cx(work, 0)
        body.h(0)
        body.cx(work, 0)
        body.h(0)
    _ccx7(body, low, high, work)
    body.x(high)
    _append_inverse(body, prepare)
    result = QuantumCircuit(4, name="Opposite identity MPF counterexample")
    result.compose(body, inplace=True)
    if amplified:
        _reflect_zero(result, low, high)
        _append_inverse(result, body)
        _reflect_zero(result, low, high)
        result.compose(body, inplace=True)
        result.global_phase += math.pi
    result.metadata = {"fixture": "Uf=I,Uc=-I; outside physical common-H Strang family",
                       "amplified": bool(amplified), "ideal_acceptance": "25/729" if amplified else "25/36",
                       "normalized_conditional_state": "exact input; not a state-accuracy failure",
                       "expected_block": "5I/27" if amplified else "5I/6"}
    return result
