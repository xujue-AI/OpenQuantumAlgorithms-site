# SPDX-License-Identifier: MIT
"""One authorized, bounded numerical check of the one-spin bath source.

The primitive product, the Qiskit compact Operator, and the full-Hamiltonian
ODE are separate constructions.  No full 192-round quantum schedule is
evolved.  The averaged reduced channel is powered in a raw four-dimensional
Pauli representation, retaining trace drift instead of normalizing it away.

Scientific packages are imported only by run_validation(), inside its first
timed stage.  All counters and the one-use authorization belong to the parent
controller.  This module contains no estimator, synthesis, remote or GPU path.
The resulting GL17/GL65 computations are finite numerical evidence, never a
certificate for the continuum or the total error budget.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any


STATE_GUARD = 1e-10
SAME_SCHEDULE_GUARD = 1e-8
STEPS = 256
ROUNDS = 192
ODE_RTOL = 1e-12
ODE_ATOL = 1e-14
ODE_STEPS_PER_FREQUENCY = 10_000
ODE_RHS_PER_FREQUENCY = 100_000
ARRAY_CACHE_BYTES = 16 * 1024 * 1024
PROBE_NAMES = ("0", "1", "+", "+i")
SDK_TO_MATH = (0, 2, 1, 3)
_NP: Any = None


class ReferenceValidationError(RuntimeError):
    """Terminal error with actual partial evidence for the parent to retain."""

    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


class NumericalGuardError(ValueError):
    """A raw numerical failure, retaining the measured finite matrix."""

    def __init__(self, message: str, *, label: str, guard: str,
                 measured: Any, matrix: Any):
        super().__init__(message)
        np = _np()
        value = np.asarray(matrix)
        self.diagnostic = {"label": label, "guard": guard,
                           "limit": STATE_GUARD, "measured": _diagnostic_scalar(measured)}
        if value.size <= 16:
            self.diagnostic["raw_matrix"] = _diagnostic_matrix(value)
        else:
            self.diagnostic["shape"] = list(value.shape)
            self.diagnostic["raw_matrix_preview"] = _diagnostic_matrix(value.reshape(-1)[:16])


def _consume(budget: Any, name: str, n: int = 1) -> None:
    budget.consume(name, n)
    budget.guard()


def _np() -> Any:
    if _NP is None:
        raise RuntimeError("scientific helpers require the authorized run context")
    return _NP


def _load_libraries() -> dict[str, Any]:
    """Called only after compact_checks has begun under the parent watchdog."""
    global _NP
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from scipy.integrate import DOP853
    from scipy.linalg import expm

    _NP = np
    return {"np": np, "QuantumCircuit": QuantumCircuit, "Operator": Operator,
            "DOP853": DOP853, "expm": expm}


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> Any:
    np = _np()
    array = np.asarray(value, dtype=np.complex128)
    if array.shape != shape:
        raise NumericalGuardError(f"{label}: wrong shape {array.shape!r}, expected {shape!r}",
                                  label=label, guard="shape", measured=array.shape, matrix=array)
    if not np.all(np.isfinite(array)):
        raise NumericalGuardError(f"{label}: raw nonfinite values", label=label,
                                  guard="finite", measured="nonfinite array", matrix=array)
    return array


def _diagnostic_scalar(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_diagnostic_scalar(item) for item in value]
    if isinstance(value, complex):
        return {"real": _diagnostic_scalar(value.real), "imag": _diagnostic_scalar(value.imag)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        if math.isinf(value):
            return {"nonfinite": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _diagnostic_matrix(value: Any) -> dict[str, Any]:
    np = _np()
    array = np.asarray(value)
    return {"real": _diagnostic_scalar(array.real.tolist()),
            "imag": _diagnostic_scalar(array.imag.tolist())}


def encode_matrix(value: Any) -> dict[str, Any]:
    np = _np()
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("cannot encode nonfinite numerical evidence")
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _trace_norm(value: Any) -> float:
    np = _np()
    return float(np.linalg.svd(np.asarray(value), compute_uv=False).sum())


def _check_unitary(value: Any, budget: Any, label: str) -> float:
    np = _np()
    _consume(budget, "state_checks")
    unitary = _finite_array(value, (4, 4), label)
    residual = float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(4), ord=2))
    if residual > STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw unitarity residual {residual!r}",
                                  label=label, guard="unitarity", measured=residual, matrix=unitary)
    return residual


def check_state(value: Any, budget: Any, label: str = "state") -> dict[str, float]:
    """Check raw values; never symmetrize, renormalize or clip the state."""
    np = _np()
    _consume(budget, "state_checks")
    state = _finite_array(value, (2, 2), label)
    hermiticity = float(np.linalg.norm(state - state.conj().T, ord=2))
    trace = complex(np.trace(state))
    if hermiticity > STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw Hermiticity residual {hermiticity!r}",
                                  label=label, guard="Hermiticity", measured=hermiticity, matrix=state)
    if abs(trace - 1) > STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw trace {trace!r}", label=label,
                                  guard="trace", measured=trace, matrix=state)
    minimum = float(np.linalg.eigvalsh(state).min())
    if minimum < -STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw minimum eigenvalue {minimum!r}", label=label,
                                  guard="positivity", measured=minimum, matrix=state)
    return {"hermiticity_residual": hermiticity,
            "trace_real": trace.real, "trace_imag": trace.imag,
            "trace_residual": abs(trace - 1), "minimum_eigenvalue": minimum}


def check_choi(value: Any, budget: Any, label: str = "Choi") -> dict[str, Any]:
    """Unnormalized input⊗output Choi: raw trace two and Tr_output J=I."""
    np = _np()
    _consume(budget, "state_checks")
    choi = _finite_array(value, (4, 4), label)
    hermiticity = float(np.linalg.norm(choi - choi.conj().T, ord=2))
    if hermiticity > STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw Hermiticity residual {hermiticity!r}",
                                  label=label, guard="Hermiticity", measured=hermiticity, matrix=choi)
    eigenvalues = np.linalg.eigvalsh(choi)
    if float(eigenvalues.min()) < -STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw CP minimum eigenvalue {eigenvalues.min()!r}",
                                  label=label, guard="CP", measured=float(eigenvalues.min()), matrix=choi)
    # The first subsystem labels the input; trace the second/output subsystem.
    _consume(budget, "partial_traces")
    trace_out = np.einsum("aibi->ab", choi.reshape(2, 2, 2, 2))
    tp_residual = float(np.linalg.norm(trace_out - np.eye(2), ord=2))
    raw_trace = complex(np.trace(choi))
    if tp_residual > STATE_GUARD or abs(raw_trace - 2) > STATE_GUARD:
        raise NumericalGuardError(f"{label}: raw TP/trace residual {tp_residual!r}/{raw_trace!r}",
                                  label=label, guard="TP/trace", measured=(tp_residual, raw_trace), matrix=choi)
    return {"hermiticity_residual": hermiticity,
            "minimum_eigenvalue": float(eigenvalues.min()),
            "eigenvalues": eigenvalues.tolist(), "TP_residual": tp_residual,
            "trace_real": raw_trace.real, "trace_imag": raw_trace.imag,
            "trace_residual": abs(raw_trace - 2)}


def _paulis() -> tuple[Any, ...]:
    np = _np()
    return (np.eye(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128))


def _probe_states() -> dict[str, Any]:
    np = _np()
    return {"0": np.array([[1, 0], [0, 0]], dtype=np.complex128),
            "1": np.array([[0, 0], [0, 1]], dtype=np.complex128),
            "+": np.array([[.5, .5], [.5, .5]], dtype=np.complex128),
            "+i": np.array([[.5, -.5j], [.5j, .5]], dtype=np.complex128)}


def _raw_pauli_vector(state: Any) -> Any:
    np = _np()
    vector = np.array([np.trace(pauli @ state) for pauli in _paulis()], dtype=np.complex128)
    if float(np.max(np.abs(vector.imag))) > STATE_GUARD:
        raise ValueError("raw Pauli vector has a nonreal component")
    return vector


def _state_from_raw_pauli(vector: Any) -> Any:
    # Its first component is the actual raw trace, not a forced one.
    return sum(component * pauli for component, pauli in zip(vector, _paulis())) / 2


def _array_cache_check(context: Any, budget: Any) -> int:
    np = _np()
    seen: set[int] = set()

    def size(value: Any) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(size(x) for x in value.values())
        if isinstance(value, (tuple, list)):
            return sum(size(x) for x in value)
        return 0

    total = size(context)
    if total > ARRAY_CACHE_BYTES:
        raise ValueError("array cache exceeds the fixed 16 MiB ceiling")
    budget.guard()
    return total


def _manual_gate(operation: Any, gate_cache: dict[Any, Any]) -> Any:
    """Independent matrices in |s,e>; no Qiskit matrix is reused here."""
    np = _np()
    name, qubits, angle, clbit = operation
    if clbit is not None:
        raise ValueError("compact evolution cannot contain classical operands")
    key = (name, qubits, None if angle is None else float(angle).hex())
    if key in gate_cache:
        return gate_cache[key]
    if name == "h" and qubits in ((0,), (1,)) and angle is None:
        single = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    elif name == "rz" and qubits in ((0,), (1,)) and type(angle) is float and math.isfinite(angle):
        single = np.diag([complex(math.cos(angle / 2), -math.sin(angle / 2)),
                          complex(math.cos(angle / 2), math.sin(angle / 2))])
    elif name == "cx" and qubits == (0, 1) and angle is None:
        result = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
        gate_cache[key] = result
        return result
    else:
        raise ValueError(f"unexpected compact primitive {operation!r}")
    result = np.kron(single, np.eye(2)) if qubits == (0,) else np.kron(np.eye(2), single)
    gate_cache[key] = result
    return result


def _compact_unitaries(protocol: Any, nodes: Any, libraries: dict[str, Any],
                       budget: Any, report: dict[str, Any], context: dict[str, Any]) -> None:
    from .reusable_bath import iter_round_operations
    np = libraries["np"]
    QuantumCircuit, Operator = libraries["QuantumCircuit"], libraries["Operator"]
    gate_cache: dict[Any, Any] = {}
    context["gate_cache"] = gate_cache
    for node in nodes:
        report["active_operation"] = {"stage": "compact_checks", "node_index": node.index}
        _consume(budget, "compact_circuits")
        circuit = QuantumCircuit(2)
        manual = np.eye(4, dtype=np.complex128)
        count = 0
        for operation in iter_round_operations(node.omega, STEPS, protocol=protocol,
                                              budget=budget, counter="compact_source_instructions"):
            _consume(budget, "manual_matrix_primitive_visits")
            manual = _manual_gate(operation, gate_cache) @ manual
            _consume(budget, "sdk_primitive_visits")
            name, qubits, angle, clbit = operation
            if name == "h":
                circuit.h(qubits[0])
            elif name == "rz":
                circuit.rz(angle, qubits[0])
            elif name == "cx":
                circuit.cx(*qubits)
            else:
                raise ValueError("compact SDK circuit contains a forbidden primitive")
            count += 1
        if count != 2816 or len(circuit.data) != 2816 or circuit.num_clbits or circuit.num_parameters:
            raise ValueError("compact circuit does not have exactly the fixed 2816 primitives")
        sdk_raw = np.array(Operator(circuit).data, dtype=np.complex128, copy=True)
        sdk_math = sdk_raw[np.ix_(SDK_TO_MATH, SDK_TO_MATH)]
        manual_guard = _check_unitary(manual, budget, "manual midpoint unitary")
        sdk_guard = _check_unitary(sdk_math, budget, "SDK midpoint unitary")
        residual = float(np.linalg.norm(manual - sdk_math, ord=2))
        item = {"node_index": node.index, "omega": node.omega,
                "sdk_instruction_count": count, "sdk_global_phase_radians": float(circuit.global_phase),
                "same_schedule_operator_residual": residual,
                "manual_unitarity_residual": manual_guard, "sdk_unitarity_residual": sdk_guard}
        report["compact_checks"].append(item)
        if residual > SAME_SCHEDULE_GUARD:
            raise ValueError(f"same-midpoint implementation agreement failed at node {node.index}: {residual!r}")
        context["manual_unitaries"][node.index] = manual
        context["sdk_raw_unitaries"][node.index] = sdk_raw
        context["sdk_math_unitaries"][node.index] = sdk_math
        # At most one live circuit; the cached numerical arrays remain bounded.
        del circuit
        context["array_cache_bytes"] = _array_cache_check(context, budget)


def _ode_frequencies(nodes17: Any) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    np = _np()
    x65, weights65 = np.polynomial.legendre.leggauss(65)
    frequencies65 = 1 + x65
    frequencies65[32] = 1.0
    weights65 = weights65 / 2
    if len(frequencies65) != 65 or not np.all(np.isfinite(frequencies65)) or not np.all(weights65 > 0):
        raise ValueError("invalid fixed GL65 coefficient rule")
    if abs(math.fsum(float(w) for w in weights65) - 1) > STATE_GUARD:
        raise ValueError("raw GL65 probability weights do not sum to one")
    union = {node.omega.hex(): node.omega for node in nodes17}
    for omega in frequencies65:
        value = float(omega)
        union[value.hex()] = value
    if len(union) != 81 or sum(value == 1 for value in union.values()) != 1:
        raise ValueError("fixed GL17/GL65 union must contain exactly 81 frequencies and one shared center")
    ordered = [{"omega_hex": key, "omega": value} for key, value in sorted(union.items(), key=lambda x: x[1])]
    return ordered, [(float(omega), float(weight)) for omega, weight in zip(frequencies65, weights65)]


def _ode_unitaries(frequencies: list[dict[str, Any]], libraries: dict[str, Any],
                   budget: Any, report: dict[str, Any], context: dict[str, Any]) -> None:
    np, DOP853 = libraries["np"], libraries["DOP853"]
    identity, x, _, z = _paulis()
    system_z = np.kron(z, identity)
    bath_z = np.kron(identity, z)
    full_xx = np.kron(x, x)
    for frequency in frequencies:
        omega, key = frequency["omega"], frequency["omega_hex"]
        report["active_operation"] = {"stage": "ode_reference", "omega_hex": key, "omega": omega}
        _consume(budget, "ode_solves")
        item = {"omega_hex": key, "omega": omega, "rhs_evaluations": 0,
                "accepted_steps": 0, "status": "started"}
        report["ode_solver"]["solves"].append(item)
        free = .5 * system_z - (omega / 2) * bath_z

        def rhs(t: float, y: Any) -> Any:
            if item["rhs_evaluations"] >= ODE_RHS_PER_FREQUENCY:
                raise ValueError(f"ODE RHS cap would be exceeded for {key}")
            item["rhs_evaluations"] += 1
            _consume(budget, "ode_rhs_evaluations")
            envelope = math.exp(-((t - 8) ** 2) / 8)
            hamiltonian = free + .08 * envelope * full_xx
            return (-1j * hamiltonian @ y.reshape(4, 4)).reshape(16)

        # Constructor calls are included by the RHS wrapper.  No dense output
        # or time-history arrays are requested, and there is no retry loop.
        solver = DOP853(rhs, 0.0, np.eye(4, dtype=np.complex128).reshape(16), 16.0,
                        rtol=ODE_RTOL, atol=ODE_ATOL, vectorized=False)
        while solver.status == "running":
            budget.guard()
            if item["accepted_steps"] >= ODE_STEPS_PER_FREQUENCY:
                raise ValueError(f"ODE accepted-step cap would be exceeded for {key}")
            old_t = float(solver.t)
            message = solver.step()
            if solver.status == "failed":
                item["status"] = "failed"
                item["message"] = str(message)
                raise ValueError(f"DOP853 failed for {key}: {message}")
            if not float(solver.t) > old_t:
                raise ValueError(f"DOP853 did not advance positively for {key}")
            item["accepted_steps"] += 1
            _consume(budget, "ode_accepted_steps")
        if solver.status != "finished" or float(solver.t) != 16.0:
            raise ValueError(f"DOP853 did not finish the declared finite pulse for {key}")
        if int(solver.nfev) != item["rhs_evaluations"]:
            raise ValueError("RHS wrapper count does not match DOP853 nfev")
        unitary = np.array(solver.y.reshape(4, 4), copy=True)
        residual = _check_unitary(unitary, budget, f"ODE unitary at {key}")
        item.update({"status": "finished", "unitarity_residual": residual})
        budget.record_frequency(key, item["accepted_steps"], item["rhs_evaluations"])
        context["ode_unitaries"][key] = unitary
        context["array_cache_bytes"] = _array_cache_check(context, budget)


def _branch_outputs(unitaries: dict[Any, Any], probes: dict[str, Any], budget: Any,
                    label: str) -> dict[Any, dict[str, tuple[Any, Any]]]:
    np = _np()
    bath_states = (np.array([[1, 0], [0, 0]], dtype=np.complex128),
                   np.array([[0, 0], [0, 1]], dtype=np.complex128))
    outputs: dict[Any, dict[str, tuple[Any, Any]]] = {}
    for key, unitary in unitaries.items():
        values = {}
        for probe_name, state in probes.items():
            branches = []
            for bath_bit, bath in enumerate(bath_states):
                _consume(budget, "partial_traces")
                joint = unitary @ np.kron(state, bath) @ unitary.conj().T
                reduced = np.einsum("aibi->ab", joint.reshape(2, 2, 2, 2))
                check_state(reduced, budget, f"{label}/{key}/{probe_name}/bath{bath_bit}")
                branches.append(reduced)
            values[probe_name] = tuple(branches)
        outputs[key] = values
    return outputs


def _weighted_outputs(entries: Any, branch_cache: Any, mode: str) -> dict[str, Any]:
    np = _np()
    result = {name: np.zeros((2, 2), dtype=np.complex128) for name in PROBE_NAMES}
    for key, omega, weight, p1 in entries:
        probability = p1 if mode == "thermal" else 0.0
        for name in PROBE_NAMES:
            zero, one = branch_cache[key][name]
            result[name] += weight * ((1 - probability) * zero + probability * one)
    return result


def _make_channel(outputs: dict[str, Any], budget: Any, label: str) -> dict[str, Any]:
    np = _np()
    _consume(budget, "channel_builds")
    raw_state_checks = {name: check_state(outputs[name], budget, f"{label}({name})") for name in PROBE_NAMES}
    phi_i = outputs["0"] + outputs["1"]
    phi_z = outputs["0"] - outputs["1"]
    phi_x = 2 * outputs["+"] - phi_i
    phi_y = 2 * outputs["+i"] - phi_i
    image_basis = (phi_i, phi_x, phi_y, phi_z)
    # E01=(X+iY)/2, E10=(X-iY)/2 in input⊗output ordering.
    choi = np.block([[outputs["0"], (phi_x + 1j * phi_y) / 2],
                    [(phi_x - 1j * phi_y) / 2, outputs["1"]]])
    choi_check = check_choi(choi, budget, label)
    raw_transfer = np.array([[np.trace(pauli @ image) / 2 for image in image_basis]
                             for pauli in _paulis()], dtype=np.complex128)
    if float(np.max(np.abs(raw_transfer.imag))) > STATE_GUARD:
        raise ValueError(f"{label}: raw Pauli-transfer matrix is nonreal")
    top_row_residual = float(np.linalg.norm(raw_transfer[0] - np.array([1, 0, 0, 0]), ord=2))
    if top_row_residual > STATE_GUARD:
        raise ValueError(f"{label}: raw Pauli-transfer TP row residual {top_row_residual!r}")
    # Preserve the computed top row.  Nothing here forces exact trace one.
    return {"outputs": outputs, "phi_paulis": image_basis, "choi": choi,
            "choi_check": choi_check, "raw_state_checks": raw_state_checks,
            "raw_transfer": raw_transfer, "top_row_residual": top_row_residual,
            "A": raw_transfer[1:, 1:].real, "c": raw_transfer[1:, 0].real}


def _channel_endpoint(channel: dict[str, Any], probes: Any, target: Any,
                      target_z: float, budget: Any, label: str) -> dict[str, Any]:
    np = _np()
    _consume(budget, "channel_powers")
    raw_power = np.linalg.matrix_power(channel["raw_transfer"], ROUNDS)
    final_states = {}
    final_checks = {}
    for name, state in probes.items():
        final = _state_from_raw_pauli(raw_power @ _raw_pauli_vector(state))
        final_checks[name] = check_state(final, budget, f"{label}^{ROUNDS}({name})")
        final_states[name] = final
    channel["raw_power"] = raw_power
    channel["final_states"] = final_states
    a, c = channel["A"], channel["c"]
    _consume(budget, "fixedpoint_solves")
    fixed_equation = np.eye(3) - a
    singular_values = np.linalg.svd(fixed_equation, compute_uv=False)
    if not np.all(np.isfinite(singular_values)) or float(singular_values.min()) <= 0:
        raise ValueError(f"{label}: I-A is not numerically invertible")
    fixed_bloch = np.linalg.solve(fixed_equation, c)
    fixed_state = _state_from_raw_pauli(np.r_[1.0, fixed_bloch])
    fixed_check = check_state(fixed_state, budget, f"{label} numerical fixed point")
    fixed_raw_residual = _trace_norm(_state_from_raw_pauli(channel["raw_transfer"] @
                                                        _raw_pauli_vector(fixed_state)) - fixed_state)
    if fixed_raw_residual > STATE_GUARD:
        raise ValueError(f"{label}: raw fixed-point residual {fixed_raw_residual!r}")
    # Bloch-power information is the lower-right block of the already-computed
    # raw transfer power only when TP is exact.  Compute its own 3x3 power and
    # label it as a numerical affine diagnostic rather than replacing raw data.
    _consume(budget, "affine_powers")
    affine_power = np.linalg.matrix_power(a, ROUNDS)
    mixing_bound = float(np.linalg.norm(affine_power, ord=2) * (1 + np.linalg.norm(fixed_bloch, ord=2)))
    fixed_z = float(fixed_bloch[2])
    final = final_states["0"]
    final_z = float(_raw_pauli_vector(final)[3].real)
    final_to_target = _trace_norm(final - target)
    input0_mixing = _trace_norm(final - fixed_state)
    channel.update({"fixed_state": fixed_state, "fixed_bloch": fixed_bloch,
                    "affine_power": affine_power, "mixing_bound": mixing_bound})
    return {"label": label, "representation": "raw input/output Pauli transfer; no normalization",
            "raw_choi": encode_matrix(channel["choi"]), "choi_check": channel["choi_check"],
            "one_round_state_checks": channel["raw_state_checks"],
            "raw_transfer": encode_matrix(channel["raw_transfer"]), "raw_TP_row_residual": channel["top_row_residual"],
            "raw_power_192": encode_matrix(raw_power), "affine_A": a.tolist(), "affine_c": c.tolist(),
            "I_minus_A_singular_values": singular_values.tolist(),
            "I_minus_A_condition_number": float(singular_values.max() / singular_values.min()),
            "fixedpoint_state": encode_matrix(fixed_state), "fixedpoint_check": fixed_check,
            "fixedpoint_raw_stationarity_trace_norm": fixed_raw_residual,
            "fixedpoint_Z": fixed_z, "fixedpoint_Z_target_residual": abs(fixed_z - target_z),
            "mixing_bound_192": mixing_bound, "actual_input0_mixing_trace_norm": input0_mixing,
            "mixing_bound_scope": "numerical checked-real affine diagnostic; not a certified bound for raw numerical trace drift or exact C",
            "fixedpoint_scope": "numerical checked-real Bloch diagnostic with separately reported raw stationarity residual",
            "outputs_after_192": {name: encode_matrix(value) for name, value in final_states.items()},
            "final_state_checks": final_checks, "final_state": encode_matrix(final),
            "final_Z": final_z, "final_Z_target_residual": abs(final_z - target_z),
            "final_state_trace_norm_target_distance": final_to_target,
            "ground_population": float(final[1, 1].real), "excited_population": float(final[0, 0].real),
            "raw_values_preserved": True, "averaged_channel_power_used": True,
            "continuum_or_physical_certificate": False}


def _endpoints(nodes17: Any, entries65: Any, libraries: dict[str, Any], budget: Any,
               report: dict[str, Any], context: dict[str, Any]) -> None:
    np = _np()
    probes = _probe_states()
    context["probe_inputs"] = probes
    context["ode_branches"] = _branch_outputs(context["ode_unitaries"], probes, budget, "ODE")
    context["source_branches"] = _branch_outputs(context["manual_unitaries"], probes, budget, "midpoint")
    context["array_cache_bytes"] = _array_cache_check(context, budget)
    _, _, _, z = _paulis()
    hamiltonian = .5 * z
    _consume(budget, "spectral_expm_calls")
    unnormalized = libraries["expm"](-hamiltonian)
    thermal_target = unnormalized / np.trace(unnormalized)
    _consume(budget, "spectral_eigh_calls")
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    negative_indices = np.flatnonzero(energies < 0)
    if len(negative_indices) != 1:
        raise ValueError("independent ground spectral projector is not rank one")
    ground_vector = eigenvectors[:, int(negative_indices[0])]
    ground_target = np.outer(ground_vector, ground_vector.conj())
    targets = {"thermal": thermal_target, "ground": ground_target}
    context["targets"] = targets
    for mode, target in targets.items():
        check_state(target, budget, f"independent {mode} target")
        target_z = float(np.trace(z @ target).real)
        report["modes"][mode] = {"target_state": encode_matrix(target), "target_Z": target_z,
                                 "channels": {}, "proxy_comparisons": {}}
        context["channels"][mode] = {}
        entries17ode = [(node.omega.hex(), node.omega, node.weight, node.p1) for node in nodes17]
        entries17source = [(node.index, node.omega, node.weight, node.p1) for node in nodes17]
        entries65ode = [(omega.hex(), omega, weight, 1 / (1 + math.exp(omega))) for omega, weight in entries65]
        for name, entries, branch_cache in (("T17", entries17source, context["source_branches"]),
                                            ("Q17_hat", entries17ode, context["ode_branches"]),
                                            ("C65_hat", entries65ode, context["ode_branches"])):
            report["active_operation"] = {"stage": "channel_endpoints", "mode": mode, "channel": name}
            channel = _make_channel(_weighted_outputs(entries, branch_cache, mode), budget, f"{mode}/{name}")
            context["channels"][mode][name] = channel
            endpoint = _channel_endpoint(channel, probes, target, target_z, budget, f"{mode}/{name}")
            endpoint["evidence_scope"] = ("finite machine-coefficient midpoint channel" if name == "T17"
                                          else "finite full-Hamiltonian numerical ODE proxy")
            report["modes"][mode]["channels"][name] = endpoint
        channels = context["channels"][mode]
        per_input = {}
        for name in PROBE_NAMES:
            t_state = channels["T17"]["final_states"][name]
            q_state = channels["Q17_hat"]["final_states"][name]
            c_state = channels["C65_hat"]["final_states"][name]
            per_input[name] = {"T17_vs_Q17_preparation_trace_norm": _trace_norm(t_state - q_state),
                               "Q17_vs_C65_preparation_trace_norm": _trace_norm(q_state - c_state),
                               "T17_vs_Q17_Z_difference": abs(float(np.trace(z @ (t_state - q_state)).real))}
        c65_endpoint = report["modes"][mode]["channels"]["C65_hat"]
        report["modes"][mode]["proxy_comparisons"] = {
            "per_input": per_input, "input0": per_input["0"],
            "C65_fixedpoint_Z_proxy_allowance_met": c65_endpoint["fixedpoint_Z_target_residual"] <= .02,
            "C65_mixing_proxy_allowance_met": c65_endpoint["mixing_bound_192"] <= .006,
            "Q17_vs_C65_proxy_preparation_allowance_met": per_input["0"]["Q17_vs_C65_preparation_trace_norm"] <= .0015,
            "T17_vs_Q17_proxy_preparation_allowance_met": per_input["0"]["T17_vs_Q17_preparation_trace_norm"] <= .0015,
            "certifies_continuum_quadrature_or_total_budget": False}
    for node, item in zip(nodes17, report["compact_checks"]):
        ode = context["ode_unitaries"][node.omega.hex()]
        item["ode_operator_difference_proxy"] = float(np.linalg.norm(context["manual_unitaries"][node.index] - ode, ord=2))
    context["array_cache_bytes"] = _array_cache_check(context, budget)


def _counter_snapshot(budget: Any) -> dict[str, Any]:
    counters = getattr(budget, "counters", {})
    return {str(key): int(value) for key, value in counters.items()}


def run_validation(protocol: Any, output_dir: str | Path, budget: Any) -> dict[str, Any]:
    """Run only the frozen 17/65 one-spin request under an authorized Budget.

    Success returns JSON evidence and a separate in-memory context for tests.
    A terminal failure carries actual partial evidence; the caller must retain
    it and stop, rather than repeat, normalize, tune or continue later stages.
    """
    from .reusable_bath import _require_protocol, make_quadrature
    if budget is None or not all(callable(getattr(budget, name, None))
                                for name in ("consume", "guard", "stage", "record_frequency")):
        raise ValueError("an authorized parent Budget is required")
    report: dict[str, Any] = {
        "schema_version": 1, "status": "started_bounded_finite_reference_validation",
        "parameters": {"h": .5, "alpha": .08, "duration": 16, "sigma": 2,
                       "frequency_cutoff": 2, "main_frequency_nodes": 17,
                       "reference_frequency_nodes": 65, "rounds": ROUNDS, "steps_per_round": STEPS},
        "basis": {"mathematical": "|s,e>", "SDK": "|e,s>", "SDK_to_mathematical_permutation": list(SDK_TO_MATH)},
        "execution_scope": {"device": "local CPU", "full_schedule_quantum_evolutions": 0,
                            "statistical_population_executed": False, "synthesis_calls": 0,
                            "CT_expansions": 0, "Native_queries": 0, "remote_or_GPU_jobs": 0,
                            "parameter_or_tolerance_retries": 0},
        "compact_checks": [], "ode_solver": {"solver": "SciPy DOP853", "rtol": ODE_RTOL,
                                             "atol": ODE_ATOL, "solves": []}, "modes": {},
        "error_ledger": {"all_components_verified": False,
                         "continuum_bridge": "unavailable; finite node/tolerance comparisons are proxy evidence",
                         "coherent_realization_reserves_verified": False,
                         "physical_budgets": "unallocated/uncomputed",
                         "total_absolute_Z_error_target": .05,
                         "total_target_certified": False}}
    context: dict[str, Any] = {"manual_unitaries": {}, "sdk_raw_unitaries": {},
                              "sdk_math_unitaries": {}, "ode_unitaries": {}, "channels": {}}
    try:
        protocol = _require_protocol(protocol)
        if protocol["rounds"] != ROUNDS or protocol["time_discretization"]["steps_per_round"] != STEPS:
            raise ValueError("reference execution is restricted to the frozen K192/L256 protocol")
        nodes17 = make_quadrature(17, protocol)
        if len(nodes17) != 17:
            raise ValueError("reference execution requires all 17 frozen frequencies")
        report["protocol_sha256"] = protocol.sha256
        with budget.stage("compact_checks"):
            libraries = _load_libraries()
            report["packages"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "qiskit")}
            _compact_unitaries(protocol, nodes17, libraries, budget, report, context)
        with budget.stage("ode_reference"):
            frequencies, entries65 = _ode_frequencies(nodes17)
            report["ode_solver"]["unique_frequency_count"] = len(frequencies)
            report["frequency_rules"] = {
                "GL17": [{"index": node.index, "omega": node.omega, "weight": node.weight,
                          "thermal_p1": node.p1} for node in nodes17],
                "GL65": [{"index": index, "omega": omega, "weight": weight,
                          "thermal_p1": 1 / (1 + math.exp(omega))}
                         for index, (omega, weight) in enumerate(entries65)]}
            _ode_unitaries(frequencies, libraries, budget, report, context)
            report["ode_solver"]["accepted_steps_total"] = sum(x["accepted_steps"] for x in report["ode_solver"]["solves"])
            report["ode_solver"]["rhs_evaluations_total"] = sum(x["rhs_evaluations"] for x in report["ode_solver"]["solves"])
        with budget.stage("channel_endpoints"):
            _endpoints(nodes17, entries65, libraries, budget, report, context)
            report["array_cache_bytes"] = context["array_cache_bytes"]
            report["status"] = "computed_bounded_finite_channel_evidence"
            report.pop("active_operation", None)
            report["counters"] = _counter_snapshot(budget)
            report["counter_snapshot_scope"] = "after channel_endpoints, before cache-only snapshot tests and notebook/export finalization"
            # Verify serialization without replacing any raw numerical values.
            json.dumps(report, allow_nan=False)
            budget.guard()
        budget.context.update({"reference_context": context, "results": report})
        return {"results": report, "context": context}
    except Exception as exc:
        report["status"] = "error_bounded_finite_reference_validation"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, NumericalGuardError):
            report["error"]["diagnostic"] = exc.diagnostic
        report["counters"] = _counter_snapshot(budget)
        raise ReferenceValidationError(str(exc), report) from exc
