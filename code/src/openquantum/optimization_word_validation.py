"""Independent bounded checks of the word-shuttle's ideal logical channel.

This module imports only NumPy. It neither constructs a native physical
estimate nor calls an optimizer, sampler, transpiler or synthesis routine.
The factory supply below is an ideal logical state in a channel check; it is
not a factory circuit, physical encoder, decoder or hardware-noise model.
"""
from __future__ import annotations

import math
import struct
from collections import Counter
from typing import Iterable

import numpy as np

MAX_DATA_QUBITS = 4
MAX_INSTRUMENT_OPERATIONS = 10_000
MAX_DENSITY_QUBITS = 6
MAX_DENSITY_CASES = 32
MAX_DENSITY_MICRO_OPERATIONS = 64
SPECTRAL_TOLERANCE = 2e-11


def _gates() -> dict[str, np.ndarray]:
    phase = np.exp(1j * math.pi / 4)
    values = {
        "h": np.array([[1, 1], [1, -1]], complex) / math.sqrt(2),
        "x": np.array([[0, 1], [1, 0]], complex),
        "y": np.array([[0, -1j], [1j, 0]], complex),
        "z": np.diag([1, -1]).astype(complex),
        "s": np.diag([1, 1j]).astype(complex),
        "sdg": np.diag([1, -1j]).astype(complex),
        "t": np.diag([1, phase]).astype(complex),
        "tdg": np.diag([1, phase.conjugate()]).astype(complex),
    }
    for value in values.values():
        value.flags.writeable = False
    return values


_GATES = _gates()
_RESET = (
    np.array([[1, 0], [0, 0]], complex),
    np.array([[0, 1], [0, 0]], complex),
)
_PROJECTORS = (
    np.array([[1, 0], [0, 0]], complex),
    np.array([[0, 0], [0, 1]], complex),
)


def _bounded_int(value: int, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _phase(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("Global phase must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Global phase must be a finite real scalar")
    return result


def _parse_instrument(operations: Iterable, num_qubits: int) -> dict:
    n = _bounded_int(num_qubits, 1, MAX_DATA_QUBITS, "Data width")
    if not isinstance(operations, (list, tuple)) or not 1 <= len(operations) <= MAX_INSTRUMENT_OPERATIONS:
        raise ValueError("Instrument operations exceed the frozen 10,000-operation cap")
    initialized, measured_wires, measured_bits = set(), set(), set()
    started, readout_started = False, False
    coherent, events, measurement_map, counts = [], [], [], Counter()
    for item in operations:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("An operation needs name, wires, classical bits and parameters")
        name, wires, clbits, parameters = item
        if not isinstance(name, str) or not all(isinstance(part, (list, tuple)) for part in (wires, clbits, parameters)):
            raise ValueError("Malformed primitive operation")
        if parameters:
            raise ValueError("Only saved parameter-free Clifford+T primitives are supported")
        wires = tuple(_bounded_int(q, 0, n - 1, "Wire") for q in wires)
        clbits = tuple(_bounded_int(c, 0, n - 1, "Classical bit") for c in clbits)
        if len(set(wires)) != len(wires):
            raise ValueError("A primitive cannot repeat a wire")
        counts[name] += 1
        if name == "barrier":
            if not wires or clbits:
                raise ValueError("A barrier has only a nonempty wire list")
            events.append((name, wires, clbits))
            continue
        if name == "reset":
            if len(wires) != 1 or clbits or started or readout_started or wires[0] in initialized:
                raise ValueError("Only one initial reset per data wire is permitted")
            initialized.add(wires[0])
            events.append((name, wires, clbits))
            continue
        if initialized != set(range(n)):
            raise ValueError("Every data wire must be reset before coherent work or readout")
        if name == "measure":
            if len(wires) != 1 or len(clbits) != 1 or wires[0] in measured_wires or clbits[0] in measured_bits:
                raise ValueError("Readout must map distinct wires to distinct classical bits")
            readout_started = True
            measured_wires.add(wires[0])
            measured_bits.add(clbits[0])
            measurement_map.append([wires[0], clbits[0]])
            events.append((name, wires, clbits))
            continue
        if readout_started or clbits:
            raise ValueError("Coherent work cannot follow the first output readout")
        if name == "cx":
            if len(wires) != 2 or abs(wires[0] - wires[1]) != 1:
                raise ValueError("Data CNOTs must use distinct neighboring line wires")
        elif name not in _GATES or len(wires) != 1:
            raise ValueError(f"Unsupported saved primitive {name}")
        started = True
        coherent.append((name, wires))
    if initialized != set(range(n)) or measured_wires != set(range(n)) or measured_bits != set(range(n)):
        raise ValueError("The initialized instrument must end in a complete bijective readout")
    return {"num_qubits": n, "coherent": coherent, "events": events,
            "measurement_map": measurement_map, "counts": dict(counts), "operations": len(operations)}


def _embedded_local(gate: np.ndarray, num_qubits: int, wire: int) -> np.ndarray:
    result = np.array([[1]], complex)
    for q in reversed(range(num_qubits)):
        result = np.kron(result, gate if q == wire else np.eye(2))
    return result


def _cx_matrix(num_qubits: int, control: int, target: int) -> np.ndarray:
    size = 1 << num_qubits
    result = np.zeros((size, size), complex)
    for column in range(size):
        row = column ^ ((1 << target) if (column >> control) & 1 else 0)
        result[row, column] = 1
    return result


def _coherent_matrix(parsed: dict, global_phase: float) -> np.ndarray:
    """Contract exact ordered local words, then use explicit Kronecker matrices.

    This differs from the frozen gate-by-gate row-pair operator implementation.
    Every primitive is visited; word contraction removes neither operations
    nor phases from the resource ledger.
    """
    n = parsed["num_qubits"]
    matrix = np.eye(1 << n, dtype=complex)
    pending_wire, pending_word = None, np.eye(2, dtype=complex)

    def flush():
        nonlocal matrix, pending_wire, pending_word
        if pending_wire is not None:
            matrix = _embedded_local(pending_word, n, pending_wire) @ matrix
            pending_wire, pending_word = None, np.eye(2, dtype=complex)

    for name, wires in parsed["coherent"]:
        if name == "cx":
            flush()
            matrix = _cx_matrix(n, *wires) @ matrix
        else:
            if pending_wire != wires[0]:
                flush()
                pending_wire = wires[0]
            pending_word = _GATES[name] @ pending_word
    flush()
    return np.exp(1j * _phase(global_phase)) * matrix


def coherent_operator(operations: Iterable, num_qubits: int = 4, global_phase: float = 0.0) -> np.ndarray:
    """Physical-data-order matrix of the coherent segment, phase restored."""
    return _coherent_matrix(_parse_instrument(operations, num_qubits), global_phase)


def joint_z_probabilities(density: np.ndarray, measurement_map: list) -> np.ndarray:
    """Project a data density to the exact little-endian classical destinations."""
    rho, n = _density(density, maximum_qubits=MAX_DATA_QUBITS)
    if abs(np.trace(rho) - 1) > SPECTRAL_TOLERANCE:
        raise ValueError("The joint-readout density must have unit trace")
    if not isinstance(measurement_map, (list, tuple)) or len(measurement_map) != n:
        raise ValueError("A complete measurement map is required")
    parsed = []
    for pair in measurement_map:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each measurement map entry is a wire/classical-bit pair")
        parsed.append((_bounded_int(pair[0], 0, n - 1, "Wire"),
                       _bounded_int(pair[1], 0, n - 1, "Classical bit")))
    if {q for q, _ in parsed} != set(range(n)) or {c for _, c in parsed} != set(range(n)):
        raise ValueError("The measurement map must be bijective")
    diagonal = np.diag(rho)
    if np.max(np.abs(diagonal.imag)) > SPECTRAL_TOLERANCE or diagonal.real.min() < -SPECTRAL_TOLERANCE:
        raise ValueError("The readout density does not define real nonnegative probabilities")
    result = np.zeros(1 << n, float)
    for physical_assignment, probability in enumerate(diagonal.real):
        classical_assignment = sum(((physical_assignment >> q) & 1) << c for q, c in parsed)
        result[classical_assignment] = probability
    return result


def validate_candidate_pair(source_operations: Iterable, candidate_operations: Iterable, *,
                            source_global_phase: float = 0.0,
                            candidate_global_phase: float | None = None,
                            num_qubits: int = 4) -> dict:
    """Check exact phase/event subsequence and all data-input columns.

    The full operator comparison covers arbitrary input density matrices. The
    initialized joint-Z comparison separately checks the selected trial's
    exact classical measurement destinations. No global-phase fitting occurs.
    Event order is checked, but exact barrier locations relative to coherent
    gates require the separate builder/word-boundary structural audit.
    """
    source = _parse_instrument(source_operations, num_qubits)
    candidate = _parse_instrument(candidate_operations, num_qubits)
    original_phase = _phase(source_global_phase)
    transformed_phase = original_phase if candidate_global_phase is None else _phase(candidate_global_phase)
    if struct.pack(">d", original_phase) != struct.pack(">d", transformed_phase):
        raise ValueError("The exact saved global-phase metadata changed")
    if source["events"] != candidate["events"] or source["measurement_map"] != candidate["measurement_map"]:
        raise ValueError("Saved resets, barriers or final measurement destinations changed")
    a, b = _coherent_matrix(source, original_phase), _coherent_matrix(candidate, transformed_phase)
    operator_error = float(np.linalg.norm(a - b, 2))
    state_a, state_b = a[:, 0], b[:, 0]
    law_a = joint_z_probabilities(np.outer(state_a, state_a.conjugate()), source["measurement_map"])
    law_b = joint_z_probabilities(np.outer(state_b, state_b.conjugate()), candidate["measurement_map"])
    joint_law_error = float(np.max(np.abs(law_a - law_b)))
    if operator_error > SPECTRAL_TOLERANCE or joint_law_error > SPECTRAL_TOLERANCE:
        raise ValueError("The candidate changed the saved coherent operator or initialized joint readout")
    return {"status": "validated_ideal_logical_word_shuttle", "num_qubits": num_qubits,
            "operator_spectral_error": operator_error, "joint_z_maximum_probability_error": joint_law_error,
            "global_phase_fitting": False, "global_phase_metadata_bit_identical": True,
            "measurement_map": source["measurement_map"], "event_subsequence_preserved": True,
            "requires_independent_structural_word_boundary_audit": True,
            "source_operations_visited": source["operations"],
            "candidate_operations_visited": candidate["operations"],
            "spectral_tolerance": SPECTRAL_TOLERANCE,
            "coverage": "ideal complete-data coherent operator and initialized final readout; not a physical-code model"}


def _density(value: np.ndarray, maximum_qubits: int = MAX_DENSITY_QUBITS) -> tuple[np.ndarray, int]:
    _bounded_int(maximum_qubits, 1, MAX_DENSITY_QUBITS, "Maximum density width")
    shape = np.shape(value)
    if len(shape) != 2 or shape[0] != shape[1] or not 2 <= shape[0] <= 1 << maximum_qubits:
        raise ValueError("Density arrays exceed their bounded square dimension")
    n = int(math.log2(shape[0]))
    if 1 << n != shape[0]:
        raise ValueError("Density dimension must be a power of two")
    result = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(result)):
        raise ValueError("Density entries must be finite")
    return result, n


def _local_density(rho: np.ndarray, matrix: np.ndarray, wire: int) -> np.ndarray:
    """Apply a local matrix on both density indices through tensor contractions."""
    rho, n = _density(rho)
    q = _bounded_int(wire, 0, n - 1, "Density wire")
    tensor = rho.reshape((2,) * (2 * n))
    row_axis, column_axis = n - 1 - q, 2 * n - 1 - q
    left = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [row_axis])), 0, row_axis)
    both = np.moveaxis(np.tensordot(matrix.conjugate(), left, axes=([1], [column_axis])), 0, column_axis)
    return both.reshape(rho.shape)


def _cx_density(rho: np.ndarray, control: int, target: int) -> np.ndarray:
    rho, n = _density(rho)
    c, t = (_bounded_int(q, 0, n - 1, "Density wire") for q in (control, target))
    if c == t:
        raise ValueError("CNOT control and target must be distinct")
    indices = np.arange(1 << n)
    permutation = indices ^ (((indices >> c) & 1) << t)
    return rho[np.ix_(permutation, permutation)]


def _swap_density(rho: np.ndarray, a: int, b: int) -> np.ndarray:
    # The validation mechanism uses the same actual three-CNOT SWAP identity.
    return _cx_density(_cx_density(_cx_density(rho, a, b), b, a), a, b)


def _reset_density(rho: np.ndarray, wire: int) -> np.ndarray:
    return sum((_local_density(rho, matrix, wire) for matrix in _RESET), np.zeros_like(rho))


def corrected_injection_kraus(inverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Derive both corrected data Kraus matrices directly from scalar amplitudes."""
    if not isinstance(inverse, bool):
        raise ValueError("Injection direction must be boolean")
    sign = -1 if inverse else 1
    z = np.exp(sign * 1j * math.pi / 4)
    # CNOT's Z=0/1 projections before the conditional S or S-dagger.
    uncorrected_zero = np.diag([1, z]) / math.sqrt(2)
    uncorrected_one = np.diag([z, 1]) / math.sqrt(2)
    correction = np.diag([1, np.exp(sign * 1j * math.pi / 2)])
    return uncorrected_zero, correction @ uncorrected_one


def _inject_density(rho: np.ndarray, data: int, factory: int, port: int, inverse: bool) -> tuple[np.ndarray, dict]:
    # Ideal fresh-state replacement is a test input; its quantum supply cost
    # remains the unchanged native-factory macro in the candidate ledger.
    rho = _reset_density(rho, factory)
    rho = _local_density(rho, _GATES["h"], factory)
    rho = _local_density(rho, _GATES["tdg" if inverse else "t"], factory)
    rho = _swap_density(rho, factory, port)
    rho = _cx_density(rho, data, port)
    branches = []
    probabilities = []
    for outcome, projector in enumerate(_PROJECTORS):
        branch = _local_density(rho, projector, port)
        probabilities.append(float(np.trace(branch).real))
        if outcome:
            branch = _local_density(branch, _GATES["sdg" if inverse else "s"], data)
        branches.append(_reset_density(branch, port))
    # Discard and contract this one two-outcome readout immediately, so a long
    # word never creates an exponentially growing list of branch states.
    result = branches[0] + branches[1]
    return result, {"branch_probability_error": max(abs(p - .5) for p in probabilities),
                    "branch_states": tuple(branches), "micro_operations": 10}


def _bell_density(num_qubits: int, a: int, b: int, phase: complex = 1) -> np.ndarray:
    vector = np.zeros(1 << num_qubits, complex)
    vector[0] = 1 / math.sqrt(2)
    vector[(1 << a) | (1 << b)] = phase / math.sqrt(2)
    return np.outer(vector, vector.conjugate())


def _case(case_id: str, actual: np.ndarray, expected: np.ndarray, micro_operations: int,
          branch_probability_error: float = 0.0) -> dict:
    actual, n = _density(actual)
    expected, expected_n = _density(expected)
    if n != expected_n:
        raise ValueError("Density references have different dimensions")
    _bounded_int(micro_operations, 0, MAX_DENSITY_MICRO_OPERATIONS, "Density micro-operation count")
    error = float(np.linalg.norm(actual - expected, 2))
    trace_error = float(abs(np.trace(actual) - 1))
    hermitian_error = float(np.linalg.norm(actual - actual.conjugate().T, 2))
    if max(error, trace_error, hermitian_error, branch_probability_error) > SPECTRAL_TOLERANCE:
        raise ValueError(f"Ideal density mechanism failed: {case_id}")
    return {"case_id": case_id, "qubits": n, "density_dimension": 1 << n,
            "micro_operations": micro_operations, "spectral_error": error,
            "trace_error": trace_error, "hermiticity_error": hermitian_error,
            "maximum_branch_probability_error": branch_probability_error}


def validate_density_mechanisms() -> dict:
    """Run twenty fixed small ideal-channel cases within the frozen caps.

    Coverage composes a local injection Choi check, arbitrary-data transport
    identity and complete coherent-operator comparisons. It does not expand
    every four-data source circuit into a nine-qubit physical density model.
    """
    cases = []
    for inverse in (False, True):
        sign = -1 if inverse else 1
        t_matrix = _GATES["tdg" if inverse else "t"]
        k0, k1 = corrected_injection_kraus(inverse)
        identities = [k0 - t_matrix / math.sqrt(2),
                      k1 - np.exp(sign * 1j * math.pi / 4) * t_matrix / math.sqrt(2),
                      k0.conjugate().T @ k0 + k1.conjugate().T @ k1 - np.eye(2)]
        error = max(float(np.linalg.norm(value, 2)) for value in identities)
        if error > SPECTRAL_TOLERANCE:
            raise ValueError("Corrected injection Kraus identity failed")
        cases.append({"case_id": f"corrected-kraus-{'tdg' if inverse else 't'}",
                      "qubits": 1, "density_dimension": 2, "micro_operations": 0,
                      "spectral_error": error, "trace_error": 0.0,
                      "hermiticity_error": 0.0, "maximum_branch_probability_error": 0.0})
        initial = _bell_density(4, 0, 1)
        expected = _local_density(initial, t_matrix, 0)
        actual, details = _inject_density(initial, 0, 3, 2, inverse)
        branch_error = max(float(np.linalg.norm(branch - .5 * expected, 2))
                           for branch in details["branch_states"])
        cases.append(_case(f"injection-choi-{'tdg' if inverse else 't'}", actual, expected,
                           details["micro_operations"], max(details["branch_probability_error"], branch_error)))
    for wire in range(4):
        # F, W0,...,W_wire and a reference. At most six qubits. This proves
        # clean-workspace magic transport without including all data qubits.
        n, reference = wire + 3, wire + 2
        initial = _bell_density(n, 0, reference, 1j)
        actual = initial
        for a in range(wire + 1):
            actual = _swap_density(actual, a, a + 1)
        expected = _bell_density(n, wire + 1, reference, 1j)
        cases.append(_case(f"initialized-port-route-{wire}", actual, expected, 3 * (wire + 1)))
    ghz = np.zeros(16, complex)
    ghz[0], ghz[15] = 1 / math.sqrt(2), 1j / math.sqrt(2)
    ghz_density = np.outer(ghz, ghz.conjugate())
    nonreal = np.zeros(16, complex)
    nonreal[[0, 3, 5, 10]] = np.array([1, 1j, -1, .5j])
    nonreal /= np.linalg.norm(nonreal)
    mixed = (2 * np.outer(nonreal, nonreal.conjugate()) + np.eye(16) / 16) / 3
    inputs = (("ghz", ghz_density), ("mixed-entangled", mixed))
    words = (("signed-forward", ("t", "h", "tdg", "s", "t")),
             ("signed-inverse", ("tdg", "h", "t", "sdg", "tdg")))
    ancillas_zero = np.diag([1, 0, 0, 0])
    for wire in (1, 2, 3):
        for input_name, data_density in inputs:
            for word_name, word in words:
                initial = np.kron(ancillas_zero, data_density)
                expected = initial
                for name in word:
                    expected = _local_density(expected, _GATES[name], wire)
                actual, micro_operations, branch_error = initial, 0, 0.0
                for a in range(wire, 0, -1):
                    actual = _swap_density(actual, a, a - 1)
                    micro_operations += 3
                for name in word:
                    if name in {"t", "tdg"}:
                        actual, details = _inject_density(actual, 0, 5, 4, name == "tdg")
                        micro_operations += details["micro_operations"]
                        branch_error = max(branch_error, details["branch_probability_error"])
                    else:
                        actual = _local_density(actual, _GATES[name], 0)
                        micro_operations += 1
                for a in range(1, wire + 1):
                    actual = _swap_density(actual, a - 1, a)
                    micro_operations += 3
                cases.append(_case(f"data-shuttle-{wire}-{input_name}-{word_name}", actual, expected,
                                   micro_operations, branch_error))
    if not 1 <= len(cases) <= MAX_DENSITY_CASES:
        raise ValueError("Density mechanism suite exceeds the frozen case cap")
    return {"status": "validated_ideal_logical_injection_transport_channels",
            "case_count": len(cases), "cases": cases, "maximum_qubits": max(row["qubits"] for row in cases),
            "maximum_density_dimension": max(row["density_dimension"] for row in cases),
            "maximum_micro_operations": max(row["micro_operations"] for row in cases),
            "maximum_spectral_error": max(row["spectral_error"] for row in cases),
            "maximum_branch_probability_error": max(row["maximum_branch_probability_error"] for row in cases),
            "spectral_tolerance": SPECTRAL_TOLERANCE,
            "branch_handling": "each corrected two-outcome injection/reset is contracted immediately",
            "coverage": "ideal logical CPTP mechanisms with ideal magic-state inputs; no physical-code, factory-circuit, hardware-noise or decoder validation",
            "optimizer_sampling_synthesis_native_calls": 0}
