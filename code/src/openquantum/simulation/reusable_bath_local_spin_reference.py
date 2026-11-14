# SPDX-License-Identifier: MIT
"""Independent, bounded checks of the finite local-spin reusable-bath source.

``validate_small_cases(cases, schedules, generator, ledger)`` accepts exactly
the four frozen n=1/n=2, K=4, L=8 cases.  The caller must already have emitted
their complete sources and retained the actual Operation records in
``ledger.context["small_source_operations"]``, keyed by case_id.  Missing
records are an error; this helper never emits a replacement source or draws a
schedule.  A ledger supplies ``consume(name, amount=1)`` and ``guard()``.

All imports at module load are from the standard library.  Numerical and SDK
imports happen only after the small-case, schedule, and actual-source guards.
The matrix reference independently enumerates Ising energies and multiplies
the specified midpoint Strang factors.  It does not call generator field,
pulse, angle, or gate-matrix helpers.  The SDK path consumes the generator's
compact primitives and the caller's actual complete source records.

The output concerns a three-frequency, uniform-site, finite pulse channel.
It provides no continuum bridge, mixing theorem, Gibbs/ground-state accuracy
certificate, synthesis result, physical budget, or total-error certificate.
Raw matrices, state traces, target bias, and failures are retained without
renormalizing, symmetrizing, clipping, tuning, retries, or fixed-point solves.
No ODE, matrix exponential, remote computation, or shot sampling is used.
"""
from __future__ import annotations

import importlib.metadata
import math
from typing import Any


NUMERICAL_GUARD = 1e-10
METHOD_ID = "local-spin-full-xx-reusable-bath-v1"
HARD_CAPS = {
    "source_primitives": 1176,
    "compact_circuits": 9,
    "compact_sdk_primitive_visits": 1176,
    "reference_strang_steps": 72,
    "reference_diagonal_halves": 144,
    "reference_xx_updates": 72,
    "matrix_unit_branch_outputs": 432,
    "averaged_channels": 8,
    "channel_powers": 8,
    "choi_checks": 8,
    "conditional_round_applications": 16,
    "tiny_density_circuits": 4,
    "tiny_density_primitive_visits": 1954,
    "partial_traces": 460,
    "state_checks": 128,
}
_FROZEN_CASES = {
    "n1-ground": (1, "ground", 2026111301),
    "n1-thermal": (1, "thermal", 2026111302),
    "n2-ground": (2, "ground", 2026111303),
    "n2-thermal": (2, "thermal", 2026111304),
}
_FROZEN_CHOICES = {
    "n1-ground": ((0, 0, 0, 0), (0, 2, 0, 2), (0, 0, 0, 0)),
    "n1-thermal": ((0, 0, 0, 0), (0, 2, 1, 2), (1, 0, 0, 0)),
    "n2-ground": ((1, 1, 0, 0), (2, 2, 1, 0), (0, 0, 0, 0)),
    "n2-thermal": ((0, 0, 0, 1), (2, 1, 1, 2), (0, 0, 1, 0)),
}


class ReferenceGuardError(ValueError):
    """A terminal raw failure whose diagnostic is JSON compatible."""

    def __init__(self, message: str, diagnostic: dict[str, Any]):
        super().__init__(message)
        self.diagnostic = diagnostic


def _json_value(value: Any) -> Any:
    """Keep nonfinite values explicitly, rather than emitting invalid JSON."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        if math.isinf(value):
            return {"nonfinite": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an ordinary integer")
    return value


def _operation_signature(operation: Any, n: int) -> tuple[Any, ...]:
    """Validate a bounded primitive without importing a numerical library."""
    name = getattr(operation, "name", None)
    arities = {"rz": 1, "h": 1, "cx": 2, "x": 1, "reset": 1, "measure": 1}
    if name not in arities:
        raise ValueError(f"unsupported source primitive {name!r}")
    qubits = getattr(operation, "qubits", None)
    if not isinstance(qubits, (tuple, list)) or len(qubits) != arities[name]:
        raise ValueError(f"wrong qubit arity for {name!r}")
    indices = tuple(_integer(q, "qubit") for q in qubits)
    if any(q < 0 or q > n for q in indices) or len(set(indices)) != len(indices):
        raise ValueError("source qubit index is outside its guarded register")
    angle = getattr(operation, "angle", None)
    if name == "rz":
        if type(angle) not in (int, float) or not math.isfinite(angle):
            raise ValueError("RZ angle must be a finite real scalar")
        angle = float(angle)
    elif angle is not None:
        raise ValueError(f"unexpected angle on {name!r}")
    clbit = getattr(operation, "clbit", None)
    if name == "measure":
        clbit = _integer(clbit, "classical bit")
        if clbit < 0 or clbit >= n or indices[0] >= n:
            raise ValueError("readout must address a system qubit and output bit")
    elif clbit is not None:
        raise ValueError(f"unexpected classical bit on {name!r}")
    return name, indices, angle, clbit


def _admit_inputs(cases: Any, schedules: Any, ledger: Any) -> dict[str, Any]:
    """Reject non-small inputs and missing actual sources before SDK imports."""
    if not isinstance(cases, (list, tuple)) or len(cases) != 4:
        raise ValueError("reference requires exactly four supplied small cases")
    if not callable(getattr(ledger, "consume", None)) or not callable(getattr(ledger, "guard", None)):
        raise ValueError("reference requires a bounded consume/guard ledger")
    case_map: dict[str, Any] = {}
    for case in cases:
        case_id = getattr(case, "case_id", None)
        if case_id not in _FROZEN_CASES or case_id in case_map:
            raise ValueError("unexpected or duplicate small case_id")
        n = _integer(getattr(case, "n", None), "n")
        rounds = _integer(getattr(case, "rounds", None), "rounds")
        steps = _integer(getattr(case, "steps", None), "steps")
        seed = _integer(getattr(case, "seed", None), "seed")
        mode = getattr(case, "mode", None)
        if n not in (1, 2) or rounds != 4 or steps != 8:
            raise ValueError("reference is restricted to n<=2, K=4, L=8")
        if (n, mode, seed) != _FROZEN_CASES[case_id]:
            raise ValueError("small-case parameters differ from the frozen plan")
        case_map[case_id] = case
    if set(case_map) != set(_FROZEN_CASES):
        raise ValueError("the four frozen small cases must all be supplied")

    if isinstance(schedules, dict):
        schedule_map = schedules
    elif isinstance(schedules, (list, tuple)) and len(schedules) == 4:
        schedule_map = {}
        for schedule in schedules:
            if not isinstance(schedule, dict):
                raise ValueError("a schedule must be a dictionary")
            key = schedule.get("case_id")
            if key in schedule_map:
                raise ValueError("duplicate small schedule")
            schedule_map[key] = schedule
    else:
        raise ValueError("reference requires four supplied schedules")
    if len(schedule_map) != 4 or set(schedule_map) != set(_FROZEN_CASES):
        raise ValueError("schedule keys must be exactly the four small case_ids")
    checked_schedules: dict[str, Any] = {}
    for case_id, case in case_map.items():
        schedule = schedule_map[case_id]
        if not isinstance(schedule, dict) or schedule.get("case_id") != case_id:
            raise ValueError("schedule case_id does not match its case")
        if _integer(schedule.get("seed"), "schedule seed") != case.seed:
            raise ValueError("schedule seed differs from the frozen seed")
        choices = []
        for name, bound in (("sites", case.n), ("frequency_indices", 3), ("bath_bits", 2)):
            values = schedule.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError(f"{name} must have exactly four entries")
            checked = tuple(_integer(item, name) for item in values)
            if any(item < 0 or item >= bound for item in checked):
                raise ValueError(f"{name} has an out-of-range entry")
            choices.append(checked)
        if tuple(choices) != _FROZEN_CHOICES[case_id]:
            raise ValueError("schedule differs from the pre-admitted fixed choices")
        checked_schedules[case_id] = {
            "case_id": case_id, "seed": case.seed,
            "sites": choices[0], "frequency_indices": choices[1], "bath_bits": choices[2],
        }

    context = getattr(ledger, "context", None)
    if not isinstance(context, dict):
        raise ValueError("ledger.context must contain the actual small source cache")
    cache = context.get("small_source_operations")
    if not isinstance(cache, dict) or len(cache) != 4 or set(cache) != set(_FROZEN_CASES):
        raise ValueError("actual source cache must contain exactly the four small cases")
    source_signatures: dict[str, Any] = {}
    source_summaries: dict[str, Any] = {}
    for case_id, case in case_map.items():
        bath_x = sum(checked_schedules[case_id]["bath_bits"])
        expected_counts = {
            "rz": 32 * (4 * case.n + 1), "h": 128, "cx": 32 * (4 * case.n - 2),
            "x": bath_x, "reset": case.n + 5, "measure": case.n,
        }
        expected_count = sum(expected_counts.values())
        operations = cache[case_id]
        if not isinstance(operations, (list, tuple)) or len(operations) != expected_count:
            raise ValueError("actual cached source has the wrong bounded instruction count")
        signatures = tuple(_operation_signature(operation, case.n) for operation in operations)
        counts = dict.fromkeys(expected_counts, 0)
        for signature in signatures:
            counts[signature[0]] += 1
        if counts != expected_counts:
            raise ValueError("actual source primitive counts differ from the frozen task")
        initial = tuple(("reset", (q,), None, None) for q in range(case.n + 1))
        readout = tuple(("measure", (q,), None, q) for q in range(case.n))
        if signatures[:case.n + 1] != initial or signatures[-case.n:] != readout:
            raise ValueError("actual source must reset every qubit and end with q_j -> c_j readout")
        if any(item[0] == "measure" for item in signatures[:-case.n]):
            raise ValueError("only the exact terminal system measurements may be stripped")
        if any(item[0] == "reset" and item[1] != (case.n,)
               for item in signatures[case.n + 1:-case.n]):
            raise ValueError("post-initialization resets must discard the bath only")
        source_signatures[case_id] = signatures
        source_summaries[case_id] = {
            "instruction_count": expected_count, "primitive_counts": counts,
            "stripped_terminal_measurements": case.n,
            "density_primitive_visits": expected_count - case.n,
            "actual_emission_cache_used": True,
        }
    if sum(item["instruction_count"] for item in source_summaries.values()) != 1954:
        raise ValueError("the four fixed complete sources must contain 1954 primitives")
    if sum(item["density_primitive_visits"] for item in source_summaries.values()) != 1948:
        raise ValueError("exactly six terminal measurements must be stripped")
    ledger.guard()
    return {"cases": case_map, "schedules": checked_schedules,
            "source_signatures": source_signatures, "source_summaries": source_summaries}


def validate_small_cases(cases: Any, schedules: Any, generator: Any, ledger: Any) -> dict[str, Any]:
    """Return a JSON-compatible finite-channel report, including partial failures.

    This function performs no schedule or complete-source generation.  It is
    deliberately a bounded caller-owned helper, with no notebook, approval,
    runner, private kernel, or implicit fixture entry point.
    """
    counters = dict.fromkeys(HARD_CAPS, 0)
    diagnostic_checks = 0
    stage = "structural_admission"
    report: dict[str, Any] = {
        "schema_version": 1, "method_id": METHOD_ID, "status": "not_run",
        "validation_passed": False, "all_components_verified": False,
        "claim_limits": {
            "continuum_bridge": False, "mixing_certificate": False,
            "gibbs_accuracy_certificate": False, "ground_accuracy_certificate": False,
            "total_error_certificate": False, "physical_budgets": "uncomputed",
        },
        "scope": {
            "system_sizes": [1, 2], "rounds": 4, "steps": 8,
            "frequency_nodes": 3, "site_law": "uniform",
            "input": "all_zero_from_actual_source_resets", "readout": "all_system_z",
            "comparison": "finite_midpoint_strang_channel",
            "sdk_external_initial_state": "all_ones_to_test_initial_resets",
            "math_basis": "q0_is_most_significant_system_bit_then_bath_bit",
            "sdk_basis_conversion": "reverse_all_n_plus_1_bits_on_rows_and_columns",
            "superoperator_basis": "row_major_Eab_column_a_times_d_plus_b",
            "choi_basis": "unnormalized_input_tensor_output",
            "trace_norm_contract": "sum_singular_values_without_one_half_factor",
            "raw_output_policy": "no_renormalization_symmetrization_or_clipping",
            "max_joint_dimension": 8, "max_system_dimension": 4,
            "max_superoperator_or_choi_dimension": 16,
            "ode_solves": 0, "matrix_exponential_calls": 0, "rng_calls": 0,
            "shot_samples": 0, "synthesis_calls": 0, "native_compilation_calls": 0,
        },
        "numerical_guard": NUMERICAL_GUARD, "hard_caps": dict(HARD_CAPS),
        "compact_unitaries": [], "matrix_unit_branches": [], "channels": [],
        "targets": {}, "conditional_sources": [],
    }

    def consume(name: str, amount: int = 1) -> None:
        if name not in HARD_CAPS or type(amount) is not int or amount < 0:
            raise ValueError("unknown reference counter or invalid consumption")
        proposed = counters[name] + amount
        if proposed > HARD_CAPS[name]:
            raise ValueError(f"reference hard cap exceeded: {name}")
        ledger.consume(name, amount)
        counters[name] = proposed
        ledger.guard()

    try:
        admitted = _admit_inputs(cases, schedules, ledger)
        case_map = admitted["cases"]
        schedule_map = admitted["schedules"]
        source_signatures = admitted["source_signatures"]
        report["source_admission"] = admitted["source_summaries"]
        report["schedules"] = _json_value(schedule_map)
        for name in ("frequencies", "iter_round_operations"):
            if not callable(getattr(generator, name, None)):
                raise ValueError(f"generator is missing {name}")
        stage = "lazy_library_imports"
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import DensityMatrix, Operator

        report["environment"] = {name: importlib.metadata.version(name)
                                 for name in ("numpy", "qiskit")}

        def encode(matrix: Any) -> dict[str, Any]:
            return {"shape": list(matrix.shape), "real": _json_value(matrix.real.tolist()),
                    "imag": _json_value(matrix.imag.tolist())}

        def fail(label: str, guard: str, measured: Any, matrix: Any) -> None:
            raise ReferenceGuardError(f"{label}: raw {guard} guard failed", {
                "label": label, "guard": guard, "limit": NUMERICAL_GUARD,
                "measured": _json_value(measured), "raw_matrix": encode(matrix),
            })

        def finite(matrix: Any, shape: tuple[int, int], label: str) -> Any:
            nonlocal diagnostic_checks
            diagnostic_checks += 1
            if len(shape) != 2 or any(size not in (2, 4, 8, 16) for size in shape):
                raise ValueError("matrix dimension is outside the bounded reference")
            if getattr(matrix, "shape", None) != shape:
                raise ReferenceGuardError(f"{label}: unexpected bounded matrix shape", {
                    "label": label, "guard": "shape", "expected": list(shape),
                    "actual": _json_value(getattr(matrix, "shape", None)),
                })
            if not bool(np.all(np.isfinite(matrix))):
                fail(label, "finite", "nonfinite entries", matrix)
            return matrix

        def norm_difference(left: Any, right: Any, label: str) -> float:
            nonlocal diagnostic_checks
            diagnostic_checks += 1
            finite(left, left.shape, label + "/left")
            finite(right, left.shape, label + "/right")
            residual = float(np.linalg.norm(left - right, ord=2))
            if not math.isfinite(residual) or residual > NUMERICAL_GUARD:
                fail(label, "spectral_norm_difference", residual, left - right)
            return residual

        def check_unitary(matrix: Any, n: int, label: str) -> dict[str, Any]:
            consume("state_checks")
            dimension = 1 << (n + 1)
            finite(matrix, (dimension, dimension), label)
            residual = float(np.linalg.norm(matrix.conj().T @ matrix - np.eye(dimension), ord=2))
            if not math.isfinite(residual) or residual > NUMERICAL_GUARD:
                fail(label, "unitarity", residual, matrix)
            return {"unitarity_residual": residual}

        def check_state(matrix: Any, dimension: int, label: str) -> dict[str, Any]:
            consume("state_checks")
            finite(matrix, (dimension, dimension), label)
            hermiticity = float(np.linalg.norm(matrix - matrix.conj().T, ord=2))
            trace = complex(np.trace(matrix))
            if not math.isfinite(hermiticity) or hermiticity > NUMERICAL_GUARD:
                fail(label, "Hermiticity", hermiticity, matrix)
            if (not math.isfinite(trace.real) or not math.isfinite(trace.imag)
                    or abs(trace - 1) > NUMERICAL_GUARD):
                fail(label, "trace", trace, matrix)
            minimum = float(np.linalg.eigvalsh(matrix).min())
            if not math.isfinite(minimum) or minimum < -NUMERICAL_GUARD:
                fail(label, "positivity", minimum, matrix)
            return {"hermiticity_residual": hermiticity,
                    "trace": _json_value(trace), "trace_residual": abs(trace - 1),
                    "minimum_eigenvalue": minimum}

        def sdk_partial_trace(joint: Any, n: int) -> Any:
            consume("partial_traces")
            dimension = 1 << n
            finite(joint, (2 * dimension, 2 * dimension), "SDK joint partial trace")
            return np.trace(joint.reshape(dimension, 2, dimension, 2), axis1=1, axis2=3)

        def reference_partial_trace(joint: Any, n: int) -> Any:
            consume("partial_traces")
            dimension = 1 << n
            finite(joint, (2 * dimension, 2 * dimension), "reference joint partial trace")
            result = np.zeros((dimension, dimension), dtype=np.complex128)
            for row in range(dimension):
                for column in range(dimension):
                    for bath in (0, 1):
                        result[row, column] += joint[2 * row + bath, 2 * column + bath]
            return result

        def add_primitive(circuit: Any, signature: tuple[Any, ...]) -> None:
            name, qubits, angle, clbit = signature
            if name == "rz":
                circuit.rz(angle, qubits[0])
            elif name == "h":
                circuit.h(qubits[0])
            elif name == "cx":
                circuit.cx(qubits[0], qubits[1])
            elif name == "x":
                circuit.x(qubits[0])
            elif name == "reset":
                circuit.reset(qubits[0])
            elif name == "measure":
                circuit.measure(qubits[0], clbit)
            else:
                raise ValueError("unrecognized already-admitted primitive")

        # Independent exact GL3 constants for the uniform probability law [0,2].
        root = math.sqrt(3 / 5)
        nodes = (1 - root, 1.0, 1 + root)
        weights = (5 / 18, 4 / 9, 5 / 18)
        generated_law = generator.frequencies()
        if (not isinstance(generated_law, (list, tuple)) or len(generated_law) != 2
                or any(not isinstance(values, (list, tuple)) or len(values) != 3
                       for values in generated_law)):
            raise ValueError("generator frequency law has the wrong bounded shape")
        for actual, expected in zip(generated_law, (nodes, weights)):
            if any(type(value) not in (int, float) or not math.isfinite(value)
                   or abs(value - wanted) > 1e-15 for value, wanted in zip(actual, expected)):
                raise ValueError("generator frequency law differs from independent GL3 constants")
        report["frequency_law"] = {"nodes": list(nodes), "weights": list(weights),
                                   "beta": 1.0, "independently_reconstructed": True}

        energies: dict[int, Any] = {}
        z_values: dict[int, Any] = {}
        permutations: dict[int, Any] = {}
        unitaries: dict[str, dict[tuple[int, int, int], Any]] = {"sdk": {}, "reference": {}}
        compact_signatures: dict[tuple[int, int, int], Any] = {}
        branch_cache: dict[str, dict[tuple[int, int, int, int, int, int], Any]] = {
            "sdk": {}, "reference": {},
        }
        for n in (1, 2):
            # n is guarded before any 2**n dimension or array is constructed.
            dimension = 1 << n
            values = []
            signs = []
            for system in range(dimension):
                bits = tuple((system >> (n - 1 - site)) & 1 for site in range(n))
                z = tuple(1 - 2 * bit for bit in bits)
                energy = sum(0.5 * (1 + (site % 3) / 8) * z[site] for site in range(n))
                energy += sum(0.125 * z[site] * z[site + 1] for site in range(n - 1))
                values.append(energy)
                signs.append(z)
            energies[n] = np.array(values, dtype=np.float64)
            z_values[n] = np.array(signs, dtype=np.float64)
            width = n + 1
            permutations[n] = tuple(int(format(index, f"0{width}b")[::-1], 2)
                                    for index in range(1 << width))

        stage = "compact_sdk_and_independent_strang_unitaries"
        for n in (1, 2):
            case = case_map[f"n{n}-ground"]
            dimension = 1 << n
            joint_dimension = 2 * dimension
            for site in range(n):
                for frequency_index, omega in enumerate(nodes):
                    key = (n, site, frequency_index)
                    consume("compact_circuits")
                    signatures = []
                    expected_count = 8 * (8 * n + 3)
                    for operation in generator.iter_round_operations(case, site, frequency_index):
                        if len(signatures) >= expected_count:
                            raise ValueError("compact generator exceeded its exact primitive bound")
                        signature = _operation_signature(operation, n)
                        if signature[0] not in ("rz", "h", "cx"):
                            raise ValueError("compact Operator must contain evolution primitives only")
                        consume("source_primitives")
                        signatures.append(signature)
                    if len(signatures) != expected_count:
                        raise ValueError("compact generator has the wrong exact primitive count")
                    compact_signatures[key] = tuple(signatures)
                    circuit = QuantumCircuit(n + 1)
                    for signature in signatures:
                        consume("compact_sdk_primitive_visits")
                        add_primitive(circuit, signature)
                    sdk_matrix = Operator(circuit).data
                    permutation = permutations[n]
                    sdk_matrix = sdk_matrix[np.ix_(permutation, permutation)]
                    unitaries["sdk"][key] = sdk_matrix

                    reference_matrix = np.eye(joint_dimension, dtype=np.complex128)
                    dt = 2 / 8
                    flip_mask = (1 << (n - site)) | 1
                    for step in range(8):
                        consume("reference_strang_steps")
                        midpoint = (step + 0.5) * dt
                        coupling = 0.08 * math.exp(-((midpoint - 1) ** 2) / 0.5)
                        diagonal = np.zeros((joint_dimension, joint_dimension), dtype=np.complex128)
                        for system in range(dimension):
                            for bath in (0, 1):
                                energy = float(energies[n][system]) - 0.5 * omega * (1 - 2 * bath)
                                angle = -energy * dt / 2
                                diagonal[2 * system + bath, 2 * system + bath] = complex(
                                    math.cos(angle), math.sin(angle))
                        xx = np.zeros((joint_dimension, joint_dimension), dtype=np.complex128)
                        cosine = math.cos(coupling * dt)
                        sine = math.sin(coupling * dt)
                        for index in range(joint_dimension):
                            xx[index, index] = cosine
                            xx[index ^ flip_mask, index] = -1j * sine
                        consume("reference_diagonal_halves")
                        reference_matrix = diagonal @ reference_matrix
                        consume("reference_xx_updates")
                        reference_matrix = xx @ reference_matrix
                        consume("reference_diagonal_halves")
                        reference_matrix = diagonal @ reference_matrix
                    unitaries["reference"][key] = reference_matrix
                    record = {"n": n, "site": site, "frequency_index": frequency_index,
                              "omega": omega, "sdk_to_math_permutation": list(permutation),
                              "primitive_visits": expected_count,
                              "sdk_matrix": encode(sdk_matrix),
                              "reference_matrix": encode(reference_matrix)}
                    report["compact_unitaries"].append(record)
                    record["sdk_checks"] = check_unitary(sdk_matrix, n, f"SDK U {key}")
                    record["reference_checks"] = check_unitary(reference_matrix, n, f"reference U {key}")
                    record["operator_norm_residual"] = norm_difference(sdk_matrix, reference_matrix, f"U {key}")

        # Verify complete actual source structure against the cached compact
        # records, including every preparation X, reset, and terminal readout.
        for case_id, case in case_map.items():
            schedule = schedule_map[case_id]
            expected = [("reset", (q,), None, None) for q in range(case.n + 1)]
            for site, frequency_index, bath in zip(schedule["sites"], schedule["frequency_indices"],
                                                  schedule["bath_bits"]):
                if bath:
                    expected.append(("x", (case.n,), None, None))
                expected.extend(compact_signatures[(case.n, site, frequency_index)])
                expected.append(("reset", (case.n,), None, None))
            expected.extend(("measure", (q,), None, q) for q in range(case.n))
            if tuple(expected) != source_signatures[case_id]:
                raise ValueError(f"actual full source {case_id} differs from its complete recorded schedule")

        stage = "complete_matrix_unit_branch_caches"
        for path in ("sdk", "reference"):
            for n in (1, 2):
                dimension = 1 << n
                for site in range(n):
                    for frequency_index in range(3):
                        unitary = unitaries[path][(n, site, frequency_index)]
                        for bath in (0, 1):
                            for a in range(dimension):
                                for b in range(dimension):
                                    consume("matrix_unit_branch_outputs")
                                    key = (n, site, frequency_index, bath, a, b)
                                    if path == "sdk":
                                        system_input = np.zeros((dimension, dimension), dtype=np.complex128)
                                        system_input[a, b] = 1
                                        bath_input = np.zeros((2, 2), dtype=np.complex128)
                                        bath_input[bath, bath] = 1
                                        joint_input = np.kron(system_input, bath_input)
                                        output = sdk_partial_trace(unitary @ joint_input @ unitary.conj().T, n)
                                    else:
                                        # An independent matrix-unit partial trace,
                                        # with explicit mathematical basis indices.
                                        consume("partial_traces")
                                        output = np.zeros((dimension, dimension), dtype=np.complex128)
                                        for row in range(dimension):
                                            for column in range(dimension):
                                                for discarded_bath in (0, 1):
                                                    output[row, column] += (
                                                        unitary[2 * row + discarded_bath, 2 * a + bath]
                                                        * unitary[2 * column + discarded_bath, 2 * b + bath].conjugate())
                                    branch_cache[path][key] = output
                                    record = {"path": path, "n": n, "site": site,
                                              "frequency_index": frequency_index, "bath_input": bath,
                                              "matrix_unit": [a, b], "raw_output": encode(output)}
                                    report["matrix_unit_branches"].append(record)
                                    finite(output, (dimension, dimension), f"{path} branch {key}")
                                    if path == "reference":
                                        record["sdk_reference_norm_residual"] = norm_difference(
                                            branch_cache["sdk"][key], output, f"branch {key}")

        target_cache: dict[tuple[int, str], Any] = {}
        for n in (1, 2):
            dimension = 1 << n
            ground_index = min(range(dimension), key=lambda index: float(energies[n][index]))
            if ground_index != dimension - 1 or sum(
                    float(value) == float(energies[n][ground_index]) for value in energies[n]) != 1:
                raise ValueError("independent energy enumeration does not give unique all-ones ground state")
            thermal_weights = [math.exp(-float(energy)) for energy in energies[n]]
            partition = sum(thermal_weights)
            thermal = np.diag(np.array([value / partition for value in thermal_weights], dtype=np.complex128))
            ground = np.zeros((dimension, dimension), dtype=np.complex128)
            ground[ground_index, ground_index] = 1
            for mode, target in (("thermal", thermal), ("ground", ground)):
                target_cache[(n, mode)] = target
                report["targets"][f"n{n}-{mode}"] = {
                    "system_energies": energies[n].tolist(), "raw_matrix": encode(target),
                    "normalization": "independently_normalized_analytic_target",
                    "checks": check_state(target, dimension, f"target n{n} {mode}"),
                    "ground_basis_index": ground_index, "unique_ground": True,
                }

        def observables(state: Any, n: int) -> dict[str, Any]:
            values = []
            for site in range(n):
                value = sum(complex(state[index, index]) * float(z_values[n][index, site])
                            for index in range(1 << n))
                values.append(value)
            return {"z_per_site": _json_value(values), "mean_z": _json_value(sum(values) / n)}

        def target_diagnostics(state: Any, n: int, mode: str) -> dict[str, Any]:
            target = target_cache[(n, mode)]
            actual = observables(state, n)
            wanted = observables(target, n)
            actual_values = [sum(complex(state[index, index]) * float(z_values[n][index, site])
                                 for index in range(1 << n)) for site in range(n)]
            target_values = [sum(complex(target[index, index]) * float(z_values[n][index, site])
                                 for index in range(1 << n)) for site in range(n)]
            return {"state_observables": actual, "target_observables": wanted,
                    "target_bias_trace_norm": float(np.linalg.svd(state - target, compute_uv=False).sum()),
                    "absolute_z_error_per_site": [abs(value - goal) for value, goal in zip(actual_values, target_values)],
                    "absolute_mean_z_error": abs(sum(actual_values) / n - sum(target_values) / n),
                    "role": "diagnostic_only_without_accuracy_or_success_certificate"}

        stage = "finite_averaged_channels_choi_and_raw_powers"
        channel_cache: dict[tuple[str, int, str], Any] = {}
        power_cache: dict[tuple[str, int, str], Any] = {}
        endpoint_cache: dict[tuple[str, int, str], Any] = {}
        for path in ("sdk", "reference"):
            for n in (1, 2):
                dimension = 1 << n
                super_dimension = dimension * dimension
                for mode in ("ground", "thermal"):
                    consume("averaged_channels")
                    channel = np.zeros((super_dimension, super_dimension), dtype=np.complex128)
                    for site in range(n):
                        for frequency_index, omega in enumerate(nodes):
                            p1 = 0.0 if mode == "ground" else 1 / (1 + math.exp(omega))
                            for bath, probability in ((0, 1 - p1), (1, p1)):
                                weight = weights[frequency_index] * probability / n
                                for a in range(dimension):
                                    for b in range(dimension):
                                        output = branch_cache[path][(n, site, frequency_index, bath, a, b)]
                                        channel[:, a * dimension + b] += weight * output.reshape(-1, order="C")
                    key = (path, n, mode)
                    channel_cache[key] = channel
                    record = {"path": path, "n": n, "mode": mode,
                              "raw_superoperator": encode(channel)}
                    report["channels"].append(record)
                    finite(channel, (super_dimension, super_dimension), f"channel {key}")
                    consume("choi_checks")
                    consume("state_checks")
                    choi = np.zeros((super_dimension, super_dimension), dtype=np.complex128)
                    for a in range(dimension):
                        for b in range(dimension):
                            for row in range(dimension):
                                for column in range(dimension):
                                    choi[a * dimension + row, b * dimension + column] = channel[
                                        row * dimension + column, a * dimension + b]
                    record["raw_choi"] = encode(choi)
                    finite(choi, (super_dimension, super_dimension), f"Choi {key}")
                    hermiticity = float(np.linalg.norm(choi - choi.conj().T, ord=2))
                    if not math.isfinite(hermiticity) or hermiticity > NUMERICAL_GUARD:
                        fail(f"Choi {key}", "Hermiticity", hermiticity, choi)
                    eigenvalues = np.linalg.eigvalsh(choi)
                    if (not bool(np.all(np.isfinite(eigenvalues)))
                            or float(eigenvalues.min()) < -NUMERICAL_GUARD):
                        fail(f"Choi {key}", "complete_positivity", float(eigenvalues.min()), choi)
                    consume("partial_traces")
                    trace_output = np.zeros((dimension, dimension), dtype=np.complex128)
                    for a in range(dimension):
                        for b in range(dimension):
                            trace_output[a, b] = sum(choi[a * dimension + row, b * dimension + row]
                                                     for row in range(dimension))
                    tp_residual = float(np.linalg.norm(trace_output - np.eye(dimension), ord=2))
                    trace = complex(np.trace(choi))
                    if (not math.isfinite(tp_residual) or not math.isfinite(trace.real)
                            or not math.isfinite(trace.imag) or tp_residual > NUMERICAL_GUARD
                            or abs(trace - dimension) > NUMERICAL_GUARD):
                        fail(f"Choi {key}", "trace_preservation", (tp_residual, trace), choi)
                    record["choi_checks"] = {
                        "hermiticity_residual": hermiticity, "eigenvalues": eigenvalues.tolist(),
                        "minimum_eigenvalue": float(eigenvalues.min()), "tp_residual": tp_residual,
                        "raw_trace": _json_value(trace), "raw_output_partial_trace": encode(trace_output),
                    }
                    consume("channel_powers")
                    powered = np.linalg.matrix_power(channel, 4)
                    power_cache[key] = powered
                    record["raw_fourth_power"] = encode(powered)
                    finite(powered, (super_dimension, super_dimension), f"channel fourth power {key}")
                    initial = np.zeros(super_dimension, dtype=np.complex128)
                    initial[0] = 1
                    endpoint = (powered @ initial).reshape(dimension, dimension, order="C")
                    endpoint_cache[key] = endpoint
                    record["raw_endpoint_state"] = encode(endpoint)
                    record["endpoint_checks"] = check_state(endpoint, dimension, f"averaged endpoint {key}")
                    record["target_diagnostics"] = target_diagnostics(endpoint, n, mode)
                    if path == "reference":
                        sdk_key = ("sdk", n, mode)
                        record["sdk_reference_channel_norm_residual"] = norm_difference(
                            channel_cache[sdk_key], channel, f"averaged channel n{n} {mode}")
                        record["sdk_reference_power_norm_residual"] = norm_difference(
                            power_cache[sdk_key], powered, f"fourth power n{n} {mode}")
                        record["sdk_reference_endpoint_norm_residual"] = norm_difference(
                            endpoint_cache[sdk_key], endpoint, f"averaged endpoint n{n} {mode}")

        stage = "four_conditional_references_and_actual_tiny_sdk_density_sources"
        for case_id, case in case_map.items():
            n = case.n
            dimension = 1 << n
            schedule = schedule_map[case_id]
            conditional_state = np.zeros((dimension, dimension), dtype=np.complex128)
            conditional_state[0, 0] = 1
            record = {"case_id": case_id, "n": n, "mode": case.mode, "rounds": []}
            report["conditional_sources"].append(record)
            for round_index, (site, frequency_index, bath) in enumerate(zip(
                    schedule["sites"], schedule["frequency_indices"], schedule["bath_bits"])):
                consume("conditional_round_applications")
                joint_input = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
                for a in range(dimension):
                    for b in range(dimension):
                        joint_input[2 * a + bath, 2 * b + bath] = conditional_state[a, b]
                unitary = unitaries["reference"][(n, site, frequency_index)]
                joint_output = unitary @ joint_input @ unitary.conj().T
                conditional_state = reference_partial_trace(joint_output, n)
                round_record = {"round": round_index, "site": site,
                                "frequency_index": frequency_index, "bath_input": bath,
                                "raw_state": encode(conditional_state)}
                record["rounds"].append(round_record)
                round_record["checks"] = check_state(conditional_state, dimension,
                                                      f"conditional {case_id} round {round_index}")
            record["raw_conditional_reference_state"] = encode(conditional_state)
            record["target_diagnostics"] = target_diagnostics(conditional_state, n, case.mode)
            consume("tiny_density_circuits")
            source = source_signatures[case_id]
            density_source = source[:-n]  # Strip exactly the prechecked terminal measurements.
            circuit = QuantumCircuit(n + 1)
            for signature in density_source:
                consume("tiny_density_primitive_visits")
                add_primitive(circuit, signature)
            external_initial = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
            external_initial[-1, -1] = 1
            sdk_joint = DensityMatrix(external_initial).evolve(circuit).data
            permutation = permutations[n]
            sdk_joint = sdk_joint[np.ix_(permutation, permutation)]
            record["raw_sdk_final_joint_state"] = encode(sdk_joint)
            record["sdk_joint_checks"] = check_state(sdk_joint, 2 * dimension, f"SDK final joint {case_id}")
            bath_zero = np.array([[1, 0], [0, 0]], dtype=np.complex128)
            expected_joint = np.kron(conditional_state, bath_zero)
            record["raw_expected_final_joint_state"] = encode(expected_joint)
            record["final_joint_norm_residual"] = norm_difference(sdk_joint, expected_joint,
                                                                   f"final joint reset {case_id}")
            record["raw_final_bath_one_population"] = _json_value(sum(
                complex(sdk_joint[2 * system + 1, 2 * system + 1]) for system in range(dimension)))
            sdk_system = sdk_partial_trace(sdk_joint, n)
            record["raw_sdk_final_system_state"] = encode(sdk_system)
            record["sdk_system_checks"] = check_state(sdk_system, dimension, f"SDK final system {case_id}")
            record["final_system_norm_residual"] = norm_difference(sdk_system, conditional_state,
                                                                    f"conditional endpoint {case_id}")
            record["sdk_target_diagnostics"] = target_diagnostics(sdk_system, n, case.mode)
            probabilities = []
            for index in range(dimension):
                probability = complex(sdk_system[index, index])
                if abs(probability.imag) > NUMERICAL_GUARD or probability.real < -NUMERICAL_GUARD:
                    fail(f"Born readout {case_id}", "raw_probability", probability, sdk_system)
                math_bitstring = format(index, f"0{n}b")
                probabilities.append({"system_q0_first_bitstring": math_bitstring,
                                      "classical_register_c_high_first_bitstring": math_bitstring[::-1],
                                      "probability": probability.real, "raw_imaginary_part": probability.imag})
            record["readout"] = {
                "terminal_mapping": [{"qubit": q, "clbit": q} for q in range(n)],
                "mapping_checked_against_actual_source": True,
                "born_probabilities": probabilities, "shot_samples": 0,
                "observables": observables(sdk_system, n),
                "initial_resets_retained": n + 1, "round_end_bath_resets_retained": 4,
                "last_bath_reset_verified_by_joint_state": True,
            }

        stage = "final_counter_guard"
        exact_counts = {"source_primitives": 1176, "compact_circuits": 9,
                        "compact_sdk_primitive_visits": 1176, "reference_strang_steps": 72,
                        "reference_diagonal_halves": 144, "reference_xx_updates": 72,
                        "matrix_unit_branch_outputs": 432, "averaged_channels": 8,
                        "channel_powers": 8, "choi_checks": 8,
                        "conditional_round_applications": 16, "tiny_density_circuits": 4,
                        "tiny_density_primitive_visits": 1948, "partial_traces": 460}
        for name, expected in exact_counts.items():
            if counters[name] != expected:
                raise ValueError(f"completed reference has unexpected {name} count")
        report["sdk_primitive_visits_total"] = (
            counters["compact_sdk_primitive_visits"] + counters["tiny_density_primitive_visits"])
        if report["sdk_primitive_visits_total"] > 3130:
            raise ValueError("completed reference exceeds its combined SDK primitive bound")
        report["partial_trace_accounting"] = {
            "matrix_unit_branches": 432, "choi_trace_preservation": 8,
            "conditional_rounds": 16, "sdk_full_source_system_reductions": 4,
        }
        ledger.guard()
        report["status"] = "finite_component_checks_passed"
        report["validation_passed"] = True
    except Exception as error:
        report["status"] = "failed"
        report["validation_passed"] = False
        report["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        if isinstance(error, ReferenceGuardError):
            report["error"]["raw_diagnostic"] = error.diagnostic
    finally:
        report["counters"] = dict(counters)
        report["additional_diagnostic_checks"] = diagnostic_checks
        report["additional_diagnostic_checks_definition"] = (
            "finite-array checks and raw spectral-norm comparisons; distinct from ledger state_checks")
        report["all_components_verified"] = False
    return _json_value(report)
