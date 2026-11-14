# SPDX-License-Identifier: MIT
"""Bounded fresh checks of the degree-one structured DQI clean decoder.

This original validator is separate from the previously executed arbitrary-dirty
default. Importing this file imports no circuit/simulator package. Explicit
validation is restricted to n=2 and n=3 using the supplied Python 3.12 lock.
Qiskit 2.6.2 (Apache-2.0) and NumPy 2.5.4 (BSD-3-Clause) supply the numerical
backend; their inspected Python sources are pinned before quantum work begins.
The DQI formulation reference is Nature DOI 10.1038/s41586-025-09527-5. The
authors' archive implementation and benchmarks are not adopted here.

Counts include reset and readout. Coherent simulation starts in declared zero
inputs and computes ideal Born probabilities; it does not sample readouts,
synthesize the explicit RY gates, or estimate physical resources.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


SMALL_NS = (2, 3)
ABSOLUTE_TOLERANCE = 2e-12
MAX_SECONDS = 30.0
MAX_OUTPUT_BYTES = 1 << 20
DEFAULT_OUTPUT_BYTES = 192 << 10
MAX_SOURCE_BYTES = 1 << 20
MAX_CIRCUIT_TEXT_BYTES = 1 << 16
MAX_RECEIPT_BYTES = 1 << 15
MAX_SMALL_QASM_BYTES = 12 << 10
MAX_SMALL_OPERATIONS_BYTES = 9 << 10
MAX_SMALL_LEDGER_BYTES = 8 << 10
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
UNITARIES = frozenset({"h", "x", "z", "t", "tdg", "ry", "cx"})
PACKAGE_VERSIONS = {"qiskit": "2.6.2", "numpy": "2.5.4"}
DEPENDENCY_SOURCE_PINS = {
    "qiskit/quantum_info/operators/operator.py":
        "45ccc994f579615a913e5b15264ceb2e674fd5113fe6c682a5ed00bcca3d62b7",
    "qiskit/quantum_info/states/statevector.py":
        "e36f4b204202cd2538b729a66d753aa7418f2b67fb63d987723fd4969e189ef5",
    "numpy/_core/einsumfunc.py":
        "4b4fc2d54ebe6b533f680fde2fa468d30449c2d44afc73041b6debb6e302dedf",
}

# Independently derived fixed fixtures, rather than production ledger results.
FIXTURE_COUNTS = {
    2: {"h": 12, "cx": 38, "t": 20, "tdg": 15, "x": 10, "ry": 5,
        "z": 1, "reset": 6, "measure": 2, "barrier": 7},
    3: {"h": 39, "cx": 120, "t": 72, "tdg": 54, "x": 24, "ry": 7,
        "z": 1, "reset": 9, "measure": 3, "barrier": 7},
}
FIXTURE_STAGE_COUNTS = {
    2: (6, 45, 1, 4, 49, 2, 2),
    3: (9, 115, 1, 6, 192, 3, 3),
}


class StructuredValidationTimeout(TimeoutError):
    """The explicitly bounded small-validation allowance expired."""


@dataclass(frozen=True)
class SmallValidationLimits:
    """Only tighten the n=2/3 quantum-workload ceilings."""
    max_n: int = 3
    max_qubits: int = 9
    statevector_amplitudes: int = 512
    operator_dimension: int = 512
    clean_basis_columns: int = 128
    quantum_ndarray_bytes: int = 32 << 20

    def __post_init__(self):
        ceilings = {"max_n": 3, "max_qubits": 9, "statevector_amplitudes": 512,
                    "operator_dimension": 512, "clean_basis_columns": 128,
                    "quantum_ndarray_bytes": 32 << 20}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in 1..{ceiling}")


def validate_small_bounds(n: int, limits: SmallValidationLimits | None = None) -> dict:
    """Check small dimensions before loading the generator or numerical backend."""
    if type(n) is not int or n not in SMALL_NS:
        raise ValueError("Fresh coherent validation supports only n=2 and n=3")
    limits = SmallValidationLimits() if limits is None else limits
    if not isinstance(limits, SmallValidationLimits):
        raise ValueError("limits must be SmallValidationLimits")
    qubits, dimension, columns = 3 * n, 1 << (3 * n), 1 << (2 * n + 1)
    # Four full complex matrices cover old/copy/contraction/reshape buffers.
    # The extra term conservatively covers the small column/state comparisons.
    structural_bytes = 4 * dimension * dimension * 16 + 32 * dimension * 16
    if (n > limits.max_n or qubits > limits.max_qubits
            or dimension > limits.statevector_amplitudes
            or dimension > limits.operator_dimension
            or columns > limits.clean_basis_columns
            or structural_bytes > limits.quantum_ndarray_bytes):
        raise ValueError("The new small fixture exceeds its independent quantum ceiling")
    return {"qubits": qubits, "dimension": dimension, "clean_basis_columns": columns,
            "conservative_quantum_ndarray_bytes": structural_bytes,
            "ndarray_scope": "Structural quantum buffers; not total RSS or an allocator peak"}


def _seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("time_limit_seconds must be a finite number in (0,30]")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("time_limit_seconds must be a finite number in (0,30]") from error
    if not math.isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError("time_limit_seconds must be a finite number in (0,30]")
    return value


def _output_cap(value) -> int:
    if type(value) is not int or not 1 <= value <= MAX_OUTPUT_BYTES:
        raise ValueError("max_output_bytes must be an integer in 1..1048576")
    return value


def _commit(value):
    if value is not None and (not isinstance(value, str) or len(value) != 40
                              or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("source_commit must be None or a full lowercase Git commit")
    return value


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise StructuredValidationTimeout("Structured small-validation deadline expired")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _receipt_json(value):
    raw = _json(value)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("The bounded validation receipt exceeds its 32-KiB ceiling")
    return raw


def _read_source(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("A validation source exceeds the fixed byte ceiling")
    return raw


def _source_files():
    here = Path(__file__).resolve().parent
    files = {"optimization_dqi_structured.py": here / "optimization_dqi_structured.py",
             "optimization_dqi_structured_validation.py": Path(__file__).resolve()}
    for candidate in (here / "validate_optimization_dqi_structured.py",
                      here.parent.parent / "scripts/validate_optimization_dqi_structured.py"):
        if candidate.is_file():
            files["validate_optimization_dqi_structured.py"] = candidate
            break
    return files


def _environment_files():
    here = Path(__file__).resolve().parent
    candidates = {
        "requirements-lock.txt": (here / "requirements-lock.txt",
                                  here.parent.parent / "requirements-lock.txt"),
        "LICENSE.txt": (here / "LICENSE.txt", here / "LICENSE",
                        here.parent.parent / "LICENSE"),
    }
    files = {}
    for name, options in candidates.items():
        for candidate in options:
            if candidate.is_file():
                files[name] = candidate
                break
        else:
            raise FileNotFoundError("Keep the ordinary environment lock and project license with the validator")
    return files


def _load_core(raw, path):
    """Compile the exact recorded new source, avoiding the package initializer."""
    name = "_openquantum_structured_dqi_validation_core"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise RuntimeError("Cannot create a new structured-source specification")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _runtime_preflight(deadline):
    _check_deadline(deadline)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the supplied Python 3.12 lock for coherent validation")
    packages, sources = {}, {}
    for name, expected in PACKAGE_VERSIONS.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"The pinned {name}=={expected} backend is required")
        packages[name] = actual
    for relative, expected in DEPENDENCY_SOURCE_PINS.items():
        distribution = importlib.metadata.distribution(relative.split("/", 1)[0])
        raw = _read_source(distribution.locate_file(relative))
        if _sha(raw) != expected:
            raise RuntimeError("Pinned backend Python source changed: " + relative)
        sources[relative] = {"bytes": len(raw), "sha256": expected}
        _check_deadline(deadline)
    return {"python": sys.version.split()[0], "python_implementation": sys.implementation.name,
            "platform": sys.platform, "backend": "Qiskit ideal CPU Statevector/Operator",
            "packages": packages, "dependency_source_files": sources}


def _reference_rows(n):
    # This tiny reference is constructed independently of the production family.
    return [1 << j for j in range(n)] + [(1 << n) - 1], [0] * n + [1]


def _reference_outcomes(n):
    rows, signs = _reference_rows(n)
    denominator = (1 << n) * (n + 2)
    outcomes = []
    for x in range(1 << n):
        parities = [(row & x).bit_count() % 2 for row in rows]
        satisfied = sum(parity == sign for parity, sign in zip(parities, signs, strict=True))
        numerator = 1 + sum((-1 if sign else 1) * (-1 if parity else 1)
                            for sign, parity in zip(signs, parities, strict=True))
        probability = Fraction(numerator * numerator, denominator)
        outcomes.append({"integer": x, "bitstring": format(x, f"0{n}b"),
                         "satisfied": satisfied, "reference_signed_numerator": numerator,
                         "reference_denominator_squared": denominator,
                         "reference_probability": [probability.numerator, probability.denominator]})
    if sum(Fraction(*row["reference_probability"]) for row in outcomes) != 1:
        raise AssertionError("Independent Walsh reference failed exact normalization")
    return outcomes


def _reference_decode(n, syndrome):
    rows, _ = _reference_rows(n)
    for clause, row in enumerate(rows):
        if syndrome == row:
            return 1 << clause
    return 0


def _matrix_rows(n):
    rows, signs = _reference_rows(n)
    return [[(row >> j) & 1 for j in range(n)] for row in rows], signs


def _fraction(value):
    return [value.numerator, value.denominator]


def _check_scalar(value, reference, label):
    value = float(value)
    if not math.isfinite(value):
        raise AssertionError(label + " is not finite")
    error = abs(value - float(reference))
    if error > ABSOLUTE_TOLERANCE:
        raise AssertionError(label + " exceeded the fixed absolute tolerance")
    return error


def _operation_shape(operation, qubits, classical):
    name, wires, clbits, parameters = (operation.name, operation.qubits,
                                     operation.classical, operation.parameters)
    if name not in UNITARIES | {"reset", "measure", "barrier"}:
        raise AssertionError("The new source contains an opaque or unlisted operation")
    if (any(type(q) is not int or not 0 <= q < qubits for q in wires)
            or len(set(wires)) != len(wires)
            or any(type(c) is not int or not 0 <= c < classical for c in clbits)):
        raise AssertionError("Source operands exceed their distinct register wires")
    expected_wires = qubits if name == "barrier" else 2 if name == "cx" else 1
    if len(wires) != expected_wires or len(clbits) != (1 if name == "measure" else 0):
        raise AssertionError("Primitive source arity changed")
    if len(parameters) != (1 if name == "ry" else 0):
        raise AssertionError("Primitive parameter count changed")
    if any(not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p)
           for p in parameters):
        raise AssertionError("Primitive source parameters must be finite real values")


def _plan_case(core, n, limits, deadline):
    bounds = validate_small_bounds(n, limits)
    _check_deadline(deadline)
    operations = list(core.iter_operations(n))
    if len(operations) > 512:
        raise AssertionError("Small source operation ceiling exceeded")
    for operation in operations:
        _operation_shape(operation, 3 * n, n)
    counts = dict(Counter(operation.name for operation in operations))
    if counts != FIXTURE_COUNTS[n]:
        raise AssertionError("Actual new primitives differ from independent fixture counts")
    pieces, start = [], 0
    for index, operation in enumerate(operations):
        if operation.name == "barrier":
            if operation.qubits != tuple(range(3 * n)):
                raise AssertionError("Stage barrier must include the complete register")
            pieces.append((start, index, operations[start:index]))
            start = index + 1
    if len(pieces) != len(STAGES) or start != len(operations):
        raise AssertionError("Seven complete source stages and barriers are required")
    stages, stage_operations = [], {}
    for name, count, (start, stop, values) in zip(STAGES, FIXTURE_STAGE_COUNTS[n], pieces, strict=True):
        if len(values) != count:
            raise AssertionError("Independent stage size changed: " + name)
        if values != list(core.iter_stage_operations(n, name)):
            raise AssertionError("Full chronology differs from its named stage: " + name)
        stages.append({"name": name, "start": start, "stop": stop, "barrier_index": stop})
        stage_operations[name] = values
    reset_records = [(op.name, op.qubits, op.classical, op.parameters)
                     for op in stage_operations["reset"]]
    if reset_records != [("reset", (q,), (), ()) for q in range(3 * n)]:
        raise AssertionError("Every complete draw must reset all logical wires")
    expected_syndrome = [(j, n + 1 + j) for j in range(n)] + [(n, n + 1 + j) for j in range(n)]
    if [(op.name, op.qubits) for op in stage_operations["syndrome"]] != [
            ("cx", pair) for pair in expected_syndrome]:
        raise AssertionError("Syndrome source must implement the independently specified rows")
    if [(op.name, op.qubits) for op in stage_operations["sign"]] != [("z", (n,))]:
        raise AssertionError("The last clause supplies the fixed negative sign")
    if [(op.name, op.qubits) for op in stage_operations["hadamard"]] != [
            ("h", (n + 1 + j,)) for j in range(n)]:
        raise AssertionError("Final assignment transforms changed")
    if [(op.name, op.qubits, op.classical) for op in stage_operations["readout"]] != [
            ("measure", (n + 1 + j,), (j,)) for j in range(n)]:
        raise AssertionError("Assignment readout mapping changed")
    angles = [2 * math.asin(1 / math.sqrt(n + 2 - i)) for i in range(n + 1)]
    reference_rotations = [angles[0]] + [a for theta in angles[1:] for a in (theta / 2, -theta / 2)]
    rotations = [float(op.parameters[0]) for op in operations if op.name == "ry"]
    if [a.hex() for a in rotations] != [a.hex() for a in reference_rotations]:
        raise AssertionError("The primitive RY chronology or exact floating parameters changed")
    unitary = sum(counts.get(name, 0) for name in UNITARIES)
    ledger = {"primitive_counts": counts, "unitary_primitives": unitary,
              "events_excluding_barriers": len(operations) - counts["barrier"],
              "source_operations": len(operations), "unsynthesized_ry": counts["ry"],
              "explicit_t_tdg": counts["t"] + counts["tdg"],
              "compiled_clifford_t_demand_known": False, "stages": stages,
              "workspace": {"allocated_clean_wires": n - 1,
                            "preparation_peak_clean_wires": n - 1,
                            "decoder_peak_clean_wires": n - 2,
                            "arbitrary_dirty_target_action_validated": False}}
    prediction = core.analytic_ledger(n)
    if (prediction["counts"] != counts or prediction["unitary_gates"] != unitary
            or prediction["source_operations"] != len(operations)):
        raise AssertionError("Actual trace and separate production ledger disagree")
    qasm = (core.qasm_header(n) + "".join(core.operation_to_qasm(op) for op in operations)).encode()
    ndjson = b"".join((json.dumps(core.operation_to_record(op), separators=(",", ":"),
                                    allow_nan=False) + "\n").encode() for op in operations)
    if len(qasm) > MAX_SMALL_QASM_BYTES or len(ndjson) > MAX_SMALL_OPERATIONS_BYTES:
        raise ValueError("The bounded small primitive export exceeds its per-file ceiling")
    return {"n": n, "bounds": bounds, "operations": operations,
            "stages": stage_operations, "ledger": ledger, "prediction": prediction,
            "angles_hex": [a.hex() for a in angles], "qasm": qasm, "ndjson": ndjson}


def _circuit(QuantumCircuit, n, operations):
    # Unit-only stage circuits need no unused classical register. The complete
    # initialized/readout circuit retains its actual n classical bits.
    has_readout = any(operation.name == "measure" for operation in operations)
    circuit = QuantumCircuit(3 * n, n if has_readout else 0)
    for operation in operations:
        name, q, c, p = operation.name, operation.qubits, operation.classical, operation.parameters
        if name == "ry":
            circuit.ry(float(p[0]), q[0])
        elif name == "cx":
            circuit.cx(q[0], q[1])
        elif name == "barrier":
            circuit.barrier(*q)
        elif name == "measure":
            circuit.measure(q[0], c[0])
        else:
            getattr(circuit, name)(q[0])
    return circuit


def _vector_error(np, actual, expected, label):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError(label + " has invalid finite dimensions")
    error = float(np.max(np.abs(actual - expected)))
    _check_scalar(error, 0, label)
    return error


def _mass(vector, predicate):
    return sum(abs(complex(amplitude)) ** 2 for index, amplitude in enumerate(vector)
               if predicate(index))


def _check_clean_decoder(np, Operator, circuit, n, deadline):
    """One bounded operator; one clean-input column comparison at a time."""
    dimension = 1 << (3 * n)
    matrix = Operator(circuit).data
    if matrix.shape != (dimension, dimension):
        raise AssertionError("The new decoder operator has an unexpected dimension")
    maximum = 0.0
    columns = 0
    for syndrome in range(1 << n):
        decoded = _reference_decode(n, syndrome)
        for clause_target in range(1 << (n + 1)):
            _check_deadline(deadline)
            source = clause_target | (syndrome << (n + 1))
            target = (clause_target ^ decoded) | (syndrome << (n + 1))
            column = matrix[:, source]
            if not np.isfinite(column).all():
                raise AssertionError("A decoder column contains a nonfinite value")
            difference = column.copy()
            difference[target] -= 1
            error = float(np.max(np.abs(difference)))
            _check_scalar(error, 0, "clean decoder complex column")
            maximum = max(maximum, error)
            columns += 1
    # Column views also retain the full matrix, so release all of them here.
    del column, difference, matrix
    return {"clean_basis_columns": columns, "operator_dimension": dimension,
            "max_complex_entry_error": maximum, "tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": 0, "clean_workspace_only": True,
            "arbitrary_clause_targets": True, "complex_phase_retained": True,
            "off_domain_syndromes": [s for s in range(1 << n)
                                     if s and s not in _reference_rows(n)[0]]}


def _check_case(plan, np, QuantumCircuit, Statevector, Operator, deadline):
    n, stages = plan["n"], plan["stages"]
    _check_deadline(deadline)
    decoder = _circuit(QuantumCircuit, n, stages["decoder"])
    decoder_checks = _check_clean_decoder(np, Operator, decoder, n, deadline)
    dimension, m = 1 << (3 * n), n + 1
    clause_mask = (1 << m) - 1
    workspace_shift = 2 * n + 1
    zero_workspace = lambda index: index >> workspace_shift == 0
    zero_clause_workspace = lambda index: index & clause_mask == 0 and zero_workspace(index)
    amplitude = 1 / math.sqrt(n + 2)
    support = [0] + [1 << i for i in range(n + 1)]
    rows, signs = _reference_rows(n)
    prepared = Statevector.from_instruction(_circuit(QuantumCircuit, n, stages["preparation"]))
    expected = np.zeros(dimension, dtype=complex)
    for clause in support:
        expected[clause] = amplitude
    preparation_error = _vector_error(np, prepared.data, expected, "preparation amplitudes")
    preparation_clean = _mass(prepared.data, zero_workspace)
    preparation_mass_error = _check_scalar(preparation_clean, 1, "preparation workspace cleanup")
    _check_deadline(deadline)
    signed = prepared.evolve(_circuit(QuantumCircuit, n, stages["sign"] + stages["syndrome"]))
    expected.fill(0)
    for clause in support:
        syndrome, sign = 0, 1
        for i, row in enumerate(rows):
            if clause >> i & 1:
                syndrome ^= row
                sign *= -1 if signs[i] else 1
        expected[clause | (syndrome << m)] = sign * amplitude
    signed_error = _vector_error(np, signed.data, expected, "signed syndrome amplitudes")
    signed_clean = _mass(signed.data, zero_workspace)
    signed_mass_error = _check_scalar(signed_clean, 1, "signed workspace cleanup")
    _check_deadline(deadline)
    decoded = signed.evolve(decoder)
    expected.fill(0)
    expected[0] = amplitude
    for row, sign in zip(rows, signs, strict=True):
        expected[row << m] = (-1 if sign else 1) * amplitude
    decoded_error = _vector_error(np, decoded.data, expected, "decoded signed amplitudes")
    decoded_clean = _mass(decoded.data, zero_clause_workspace)
    decoded_mass_error = _check_scalar(decoded_clean, 1, "decoded clause/workspace cleanup")
    _check_deadline(deadline)
    final = decoded.evolve(_circuit(QuantumCircuit, n, stages["hadamard"]))
    reference = _reference_outcomes(n)
    expected.fill(0)
    for row in reference:
        expected[row["integer"] << m] = row["reference_signed_numerator"] / math.sqrt(
            row["reference_denominator_squared"])
    final_error = _vector_error(np, final.data, expected, "final signed Walsh amplitudes")
    final_clean = _mass(final.data, zero_clause_workspace)
    final_mass_error = _check_scalar(final_clean, 1, "final clause/workspace cleanup")
    probability_error, outcomes = 0.0, []
    for row in reference:
        actual = complex(final.data[row["integer"] << m])
        probability = abs(actual) ** 2
        probability_error = max(probability_error, _check_scalar(
            probability, Fraction(*row["reference_probability"]), "final Born probability"))
        outcomes.append(dict(row, amplitude_real=actual.real, amplitude_imag=actual.imag,
                             probability=probability))
    maximum_objective = max(row["satisfied"] for row in reference)
    reference_expectation = sum(Fraction(*row["reference_probability"]) * row["satisfied"]
                                for row in reference)
    reference_maximum = sum(Fraction(*row["reference_probability"]) for row in reference
                            if row["satisfied"] == maximum_objective)
    fixture_expectation, fixture_probability = ((Fraction(3, 2), Fraction(3, 4)) if n == 2
                                               else (Fraction(14, 5), Fraction(9, 10)))
    if (maximum_objective != n or reference_expectation != fixture_expectation
            or reference_maximum != fixture_probability):
        raise AssertionError("Independent tiny fixture quality reference changed")
    expectation = sum(row["probability"] * row["satisfied"] for row in outcomes)
    maximum_probability = sum(row["probability"] for row in outcomes
                              if row["satisfied"] == maximum_objective)
    expectation_error = _check_scalar(expectation, reference_expectation, "expected satisfied count")
    quality_error = _check_scalar(maximum_probability, reference_maximum, "maximum objective probability")
    normalization_error = _check_scalar(sum(row["probability"] for row in outcomes), 1,
                                        "assignment probability normalization")
    matrix_rows, sign_vector = _matrix_rows(n)
    _check_deadline(deadline)
    errors = [decoder_checks["max_complex_entry_error"], preparation_error, preparation_mass_error,
              signed_error, signed_mass_error, decoded_error, decoded_mass_error, final_error,
              final_mass_error, probability_error, expectation_error, quality_error,
              normalization_error]
    return {"n": n, "m": n + 1, "ell": 1,
            "family_spec": {"family": "binary_identity_plus_all_ones_degree_one",
                            "n": n, "m": n + 1, "ell": 1, "matrix": matrix_rows,
                            "signs": sign_vector,
                            "bit_order": "integer bit j is x_j; display x_(n-1)..x_0",
                            "registers": {"clause": list(range(n + 1)),
                                          "assignment": list(range(n + 1, 2 * n + 1)),
                                          "workspace": list(range(2 * n + 1, 3 * n)),
                                          "classical": list(range(n)), "total_qubits": 3 * n}},
            "decoder": decoder_checks,
            "preparation": {"support_count": n + 2, "amplitude_error": preparation_error,
                            "workspace_zero_probability": preparation_clean},
            "signed_syndrome": {"max_complex_error": signed_error,
                                "workspace_zero_probability": signed_clean},
            "decoded_state": {"max_complex_error": decoded_error,
                              "clause_workspace_zero_probability": decoded_clean},
            "final_state": {"max_complex_error": final_error,
                            "max_probability_error": probability_error,
                            "clause_workspace_zero_probability": final_clean,
                            "outcomes": outcomes},
            "quality": {"maximum_satisfied": maximum_objective,
                        "expected_satisfied": expectation,
                        "reference_expected_satisfied": _fraction(reference_expectation),
                        "maximum_probability": maximum_probability,
                        "reference_maximum_probability": _fraction(reference_maximum)},
            "actual_ledger": plan["ledger"], "limits": plan["bounds"],
            "max_error": max(errors), "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": 0, "decoder_operator_constructions": 1,
            "coherent_state_transitions": 4, "outcomes_sampled": False,
            "reset_readout_scope": "Actual primitives counted; ideal coherent calculation starts in zero inputs"}


class _ArtifactWriter:
    def __init__(self, output, cap, deadline):
        self.output, self.cap, self.deadline = output, cap, deadline
        self.files = {}

    def write(self, name, raw, *, bind=True):
        _check_deadline(self.deadline)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Artifact name must be relative to the fresh output")
        target = self.output / relative
        total = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
        # Atomic replacement temporarily retains the old receipt as well as the
        # new .partial file, so both count against this artifact ceiling.
        if total + len(raw) > self.cap:
            raise ValueError("The small-validation artifact byte ceiling would be exceeded")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        with temporary.open("xb") as stream:
            if stream.write(raw) != len(raw):
                raise OSError("Incomplete small-validation artifact write")
            stream.flush()
            os.fsync(stream.fileno())
        _check_deadline(self.deadline)
        os.replace(temporary, target)
        directory = target.parent
        while True:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if directory == self.output:
                break
            directory = directory.parent
        _check_deadline(self.deadline)
        if bind:
            self.files[name] = {"bytes": len(raw), "sha256": _sha(raw)}


def validate_small_structured(output_dir, *, time_limit_seconds=10.0,
                              max_output_bytes=DEFAULT_OUTPUT_BYTES, source_commit=None,
                              limits: SmallValidationLimits | None = None):
    """Run the new n=2/3 clean checks once and retain a bounded fresh result.

    The API checks its deadline between bounded phases and writes. For a hard
    POSIX process deadline use the ordinary CLI. No private recipe, environment
    variable or notebook hook is required. A failed call retains incomplete
    evidence; it never labels a partial case set complete.
    """
    started = time.monotonic()
    seconds, cap, commit = _seconds(time_limit_seconds), _output_cap(max_output_bytes), _commit(source_commit)
    limits = SmallValidationLimits() if limits is None else limits
    for n in SMALL_NS:
        validate_small_bounds(n, limits)
    deadline = started + seconds
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Validation requires a fresh output directory")
    if not output.parent.is_dir():
        raise ValueError("Choose a fresh validation output with an existing parent directory")
    _check_deadline(deadline)
    output.mkdir(mode=0o700)
    fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _check_deadline(deadline)
    writer = _ArtifactWriter(output, cap, deadline)
    receipt = {"schema": "openquantum.dqi.structured.validated_small.v1", "status": "incomplete",
               "scope": "New degree-one clean-workspace n=2/3 coherent checks only",
               "completion_is_provisional": True,
               "invocation_acceptance": "Requires normal API return or raw CLI exit zero; file status alone is insufficient",
               "coherent_validation_performed": False, "old_default_reexecuted": False,
               "large_generation_performed": False, "hardware_or_synthesis_performed": False,
               "cases": [], "files": {}, "source_commit": commit,
               "time_limit_seconds": seconds, "max_output_bytes": cap}
    writer.write("validated-small.json", _receipt_json(receipt), bind=False)
    try:
        source_paths = _source_files()
        raw_sources = {name: _read_source(path) for name, path in source_paths.items()}
        source_bindings = {name: {"bytes": len(raw), "sha256": _sha(raw)}
                           for name, raw in raw_sources.items()}
        environment_paths = _environment_files()
        raw_environment = {name: _read_source(path) for name, path in environment_paths.items()}
        environment_bindings = {name: {"bytes": len(raw), "sha256": _sha(raw)}
                                for name, raw in raw_environment.items()}
        core = _load_core(raw_sources["optimization_dqi_structured.py"],
                          source_paths["optimization_dqi_structured.py"])
        plans = [_plan_case(core, n, limits, deadline) for n in SMALL_NS]
        runtime = _runtime_preflight(deadline)
        receipt["provenance"] = dict(runtime, source_files=source_bindings,
                                     environment_files=environment_bindings)
        # All shape/version/source/deadline gates precede these numerical imports.
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Statevector
        _check_deadline(deadline)
        for plan in plans:
            case = _check_case(plan, np, QuantumCircuit, Statevector, Operator, deadline)
            n = plan["n"]
            writer.write(f"n{n}/dqi.qasm", plan["qasm"])
            writer.write(f"n{n}/operations.ndjson", plan["ndjson"])
            ledger_raw = _json({
                "schema": "openquantum.dqi.structured.primitive_ledger.v1", "n": n,
                "actual_ledger": plan["ledger"], "analytic_prediction": plan["prediction"],
                "preparation_angles_hex": plan["angles_hex"],
                "rotation_synthesis_performed": False})
            if len(ledger_raw) > MAX_SMALL_LEDGER_BYTES:
                raise ValueError("The bounded small ledger exceeds its per-file ceiling")
            writer.write(f"n{n}/ledger.json", ledger_raw)
            if n == 2:
                drawing = str(_circuit(QuantumCircuit, n, plan["operations"]).draw(
                    output="text", fold=120, idle_wires=True, cregbundle=False)).encode()
                if len(drawing) > MAX_CIRCUIT_TEXT_BYTES:
                    raise ValueError("The original small circuit text exceeds its byte ceiling")
                writer.write("n2/circuit.txt", drawing)
            receipt["cases"].append(case)
            receipt["coherent_validation_performed"] = True
            receipt["files"] = dict(writer.files)
            writer.write("validated-small.json", _receipt_json(receipt), bind=False)
        for name, path in source_paths.items():
            if _read_source(path) != raw_sources[name]:
                raise RuntimeError("Recorded validation source changed during execution")
        for name, path in environment_paths.items():
            if _read_source(path) != raw_environment[name]:
                raise RuntimeError("Recorded environment lock/license changed during execution")
        _check_deadline(deadline)
        receipt["status"] = "complete"
        receipt["elapsed_seconds_before_final_receipt"] = time.monotonic() - started
        receipt["files"] = dict(writer.files)
        writer.write("validated-small.json", _receipt_json(receipt), bind=False)
        _check_deadline(deadline)
        return receipt
    except BaseException as error:
        receipt["status"] = "incomplete"
        receipt["reason"] = type(error).__name__
        receipt["files"] = dict(writer.files)
        # Retention is attempted only inside the caller's remaining allowance.
        if time.monotonic() < deadline:
            try:
                writer.write("validated-small.json", _receipt_json(receipt), bind=False)
            except BaseException:
                pass
        raise
