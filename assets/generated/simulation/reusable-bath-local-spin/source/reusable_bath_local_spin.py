"""Bounded, standalone source generator for a fixed local-spin bath family.

This module uses only Python's standard library.  It materializes primitive
OpenQASM 2 source, not a state vector, density matrix, numerical reference,
fault-tolerant circuit, or accuracy certificate.  Only reset-prepared all-zero
system inputs and final measurements of every system qubit in Z are supported.

The fixed Hamiltonian is HS = sum_j h_j Z_j / 2 + J sum_j Z_j Z_(j+1),
h_j = 1 + (j % 3)/8, J = 1/8.  The bath is qubit n with HE = -omega Z_n/2.
Each round samples a uniform system site and the three-node Gauss--Legendre
law for the uniform density on [0, 2].  Thermal bath bits use beta=1.  The
full XX coupling is g(t) = .08 exp(-(t-1)^2/(2*.5^2)), over duration T=2.
Positive-time Strang steps retain the seven-primitive XX middle and reverse
the ordering, but never the signs, in the second free half.

The public size domain is the intersection of n<=64, K<=64, L<=128,
100,000 complete-source instructions, and 8 MiB per QASM file.  Repetition
counts are formula-only projections; source emission materializes one complete
preparation.  Synthesis, routing, sampling confidence, mixing, continuum error,
and all physical budgets remain uncomputed.  These source contracts do not
constitute an executed validation claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Iterator


MAX_N = 64
MAX_ROUNDS = 64
MAX_STEPS = 128
MAX_INSTRUCTIONS = 100_000
MAX_QASM_BYTES = 8 * 1024 * 1024
MAX_INPUT_BYTES = 128 * 1024
MAX_CASES = 9
MAX_REPETITIONS = 1_000_000
MAX_COUNT_METADATA_BYTES = 64 * 1024
MAX_SEED = 18_446_744_073_709_551_615

METHOD_ID = "local-spin-full-xx-reusable-bath-v1"
SCHEDULE_RNG = "random.Random(MT19937)"
CHOICE_ORDER = "site/frequency/bath_bit"
_NODES = (1.0 - math.sqrt(3.0 / 5.0), 1.0, 1.0 + math.sqrt(3.0 / 5.0))
_WEIGHTS = (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0)
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_PRIMITIVES = ("rz", "h", "cx", "x", "reset", "measure")
_SCHEDULE_KEYS = {
    "case_id", "seed", "rng", "choice_order", "sites", "frequency_indices",
    "bath_bits", "logical_rng_calls",
}


def _integer(value: object, label: str, lower: int, upper: int) -> int:
    """Accept plain bounded integers, never booleans or integral floats."""
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be a plain integer in [{lower}, {upper}]")
    return value


def _keys(value: object, required: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"{label} must have exactly these keys: {sorted(required)}")
    return value


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def _check_case_values(case: Case) -> None:
    if type(case.case_id) is not str or not _CASE_ID.fullmatch(case.case_id):
        raise ValueError("case_id must be 1..64 safe ASCII identifier characters")
    _integer(case.n, "n", 1, MAX_N)
    _integer(case.rounds, "rounds", 1, MAX_ROUNDS)
    _integer(case.steps, "steps", 1, MAX_STEPS)
    _integer(case.seed, "seed", 1, MAX_SEED)
    if type(case.mode) is not str or case.mode not in ("ground", "thermal"):
        raise ValueError("mode must be ground or thermal")
    upper = (8 * case.n + 3) * case.rounds * case.steps + 2 * case.n + case.rounds + 1
    if case.mode == "thermal":
        upper += case.rounds
    if upper > MAX_INSTRUCTIONS:
        raise ValueError("case exceeds the 100,000 complete-source instruction cap")
    # Every supported primitive line is shorter than 64 ASCII bytes, including
    # a signed 17-significant-digit angle and the largest allowed wire indices.
    if 128 + 64 * upper > MAX_QASM_BYTES:
        raise ValueError("case exceeds the conservative 8 MiB QASM byte bound")


@dataclass(frozen=True)
class Case:
    """One bounded preparation; all model, pulse, input and readout values fixed."""

    case_id: str
    n: int
    rounds: int
    steps: int
    mode: str
    seed: int

    def __post_init__(self) -> None:
        _check_case_values(self)


def _case(case: object) -> Case:
    if type(case) is not Case:
        raise ValueError("case must be a Case instance")
    _check_case_values(case)
    return case


@dataclass(frozen=True)
class Operation:
    """One visible primitive; qubit n is the bath in a case with n system spins."""

    name: str
    qubits: tuple[int, ...]
    angle: float | None = None
    clbit: int | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in _PRIMITIVES:
            raise ValueError("unsupported primitive name")
        arity = 2 if self.name == "cx" else 1
        if type(self.qubits) is not tuple or len(self.qubits) != arity:
            raise ValueError("primitive has the wrong qubit arity")
        for qubit in self.qubits:
            _integer(qubit, "qubit", 0, MAX_N)
        if self.name == "cx" and self.qubits[0] == self.qubits[1]:
            raise ValueError("CX control and target must differ")
        if self.name == "rz":
            _finite_number(self.angle, "RZ angle")
        elif self.angle is not None:
            raise ValueError("only RZ carries an angle")
        if self.name == "measure":
            _integer(self.clbit, "measurement clbit", 0, MAX_N - 1)
        elif self.clbit is not None:
            raise ValueError("only measurement carries a clbit")


def fields(n: int) -> tuple[float, ...]:
    """Return the fixed field coefficients, with zero-based system-site indices."""
    _integer(n, "n", 1, MAX_N)
    return tuple(1.0 + (j % 3) / 8.0 for j in range(n))


def frequencies() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return GL3 nodes and normalized uniform-on-[0,2] quadrature weights."""
    return _NODES, _WEIGHTS


def _rng_counts(case: Case) -> dict[str, int]:
    bit_calls = case.rounds if case.mode == "thermal" else 0
    return {
        "site": case.rounds,
        "frequency": case.rounds,
        "bath_bit": bit_calls,
        "total": 2 * case.rounds + bit_calls,
    }


def make_schedule(case: Case, *, ledger: object | None = None) -> dict:
    """Record seeded site/frequency/bath choices; count logical calls, not words.

    A site call is made even for n=1.  Ground mode makes no bath-bit RNG call.
    The emitter accepts these recorded choices and draws no randomness.
    Ordinary callers need no ledger.  When supplied, consume(name, amount)
    records each completed draw immediately, using site_selection_calls,
    frequency_random_calls, and thermal_bit_random_calls.  An interrupted
    ledger records a completed-draw prefix, not a successful full schedule.
    """
    case = _case(case)
    rng = random.Random(case.seed)
    sites: list[int] = []
    frequency_indices: list[int] = []
    bath_bits: list[int] = []
    for _ in range(case.rounds):
        site = rng.randrange(case.n)
        _ledger_consume(ledger, "site_selection_calls", 1)
        sites.append(site)
        draw = rng.random()
        _ledger_consume(ledger, "frequency_random_calls", 1)
        index = 0 if draw < 5.0 / 18.0 else 1 if draw < 13.0 / 18.0 else 2
        frequency_indices.append(index)
        if case.mode == "thermal":
            bit_draw = rng.random()
            _ledger_consume(ledger, "thermal_bit_random_calls", 1)
            bit = int(bit_draw < 1.0 / (1.0 + math.exp(_NODES[index])))
        else:
            bit = 0
        bath_bits.append(bit)
    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "rng": SCHEDULE_RNG,
        "choice_order": CHOICE_ORDER,
        "sites": sites,
        "frequency_indices": frequency_indices,
        "bath_bits": bath_bits,
        "logical_rng_calls": _rng_counts(case),
    }


def validate_schedule(case: Case, schedule: dict) -> None:
    """Validate every supplied choice before a source iterator or writer starts."""
    case = _case(case)
    schedule = _keys(schedule, _SCHEDULE_KEYS, "schedule")
    for key, expected in (
        ("case_id", case.case_id), ("seed", case.seed),
        ("rng", SCHEDULE_RNG), ("choice_order", CHOICE_ORDER),
    ):
        if type(schedule[key]) is not type(expected) or schedule[key] != expected:
            raise ValueError(f"schedule {key} does not match the case contract")
    for key, upper in (("sites", case.n - 1), ("frequency_indices", 2), ("bath_bits", 1)):
        choices = schedule[key]
        if type(choices) is not list or len(choices) != case.rounds:
            raise ValueError(f"schedule {key} must be a list of exactly rounds choices")
        for choice in choices:
            _integer(choice, f"schedule {key} entry", 0, upper)
    if case.mode == "ground" and any(schedule["bath_bits"]):
        raise ValueError("ground-mode bath bits must all be zero")
    counts = _keys(schedule["logical_rng_calls"], {"site", "frequency", "bath_bit", "total"}, "logical_rng_calls")
    for key, expected in _rng_counts(case).items():
        if type(counts[key]) is not int or counts[key] != expected:
            raise ValueError(f"logical_rng_calls {key} does not match the fixed draw protocol")


def iter_round_operations(case: Case, site: int, frequency_index: int) -> Iterator[Operation]:
    """Yield one round's L*(8*n+3) evolution primitives, with no resets/readout."""
    case = _case(case)
    _integer(site, "site", 0, case.n - 1)
    _integer(frequency_index, "frequency_index", 0, 2)
    dt = 2.0 / case.steps
    omega = _NODES[frequency_index]
    bath = case.n
    free_half: list[Operation] = []
    for j, coefficient in enumerate(fields(case.n)):
        free_half.append(Operation("rz", (j,), coefficient * dt / 2.0))
    for j in range(case.n - 1):
        free_half.extend((
            Operation("cx", (j, j + 1)),
            Operation("rz", (j + 1,), 0.125 * dt),
            Operation("cx", (j, j + 1)),
        ))
    free_half.append(Operation("rz", (bath,), -omega * dt / 2.0))
    # This is a reversal of gate order, not a negative-time inverse.
    reverse_half = tuple(reversed(free_half))
    for step in range(case.steps):
        midpoint = (step + 0.5) * dt
        coupling = 0.08 * math.exp(-((midpoint - 1.0) ** 2) / (2.0 * 0.5 ** 2))
        yield from free_half
        yield Operation("h", (site,))
        yield Operation("h", (bath,))
        yield Operation("cx", (site, bath))
        yield Operation("rz", (bath,), 2.0 * coupling * dt)
        yield Operation("cx", (site, bath))
        yield Operation("h", (site,))
        yield Operation("h", (bath,))
        yield from reverse_half


def iter_operations(case: Case, schedule: dict) -> Iterator[Operation]:
    """Yield complete all-zero preparation, each bath discard/reset, and n readouts."""
    case = _case(case)
    validate_schedule(case, schedule)
    # Snapshot all bounded choices before yielding; caller mutation cannot alter
    # a preparation already being emitted.
    sites = tuple(schedule["sites"])
    frequency_indices = tuple(schedule["frequency_indices"])
    bath_bits = tuple(schedule["bath_bits"])
    for qubit in range(case.n + 1):
        yield Operation("reset", (qubit,))
    for site, frequency_index, bath_bit in zip(sites, frequency_indices, bath_bits, strict=True):
        if bath_bit:
            yield Operation("x", (case.n,))
        yield from iter_round_operations(case, site, frequency_index)
        yield Operation("reset", (case.n,))
    for qubit in range(case.n):
        yield Operation("measure", (qubit,), clbit=qubit)


def qasm_header(n: int) -> str:
    """OpenQASM 2 register declarations; bath index n and c[j] for system q[j]."""
    _integer(n, "n", 1, MAX_N)
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n + 1}];\ncreg c[{n}];\n'


def operation_qasm(operation: Operation) -> str:
    """Serialize one primitive, with a trailing newline and finite RZ angle."""
    if type(operation) is not Operation:
        raise ValueError("operation must be an Operation instance")
    operation.__post_init__()
    name = operation.name
    if name == "rz":
        return f"rz({float(operation.angle):.17g}) q[{operation.qubits[0]}];\n"
    if name == "cx":
        return f"cx q[{operation.qubits[0]}],q[{operation.qubits[1]}];\n"
    if name == "measure":
        return f"measure q[{operation.qubits[0]}] -> c[{operation.clbit}];\n"
    return f"{name} q[{operation.qubits[0]}];\n"


def count_source(case: Case, schedule: dict | None = None, repetitions: int = 1) -> dict:
    """Count by formulas only; no RNG calls, primitive emission or simulation.

    P repetitions rerun all initialization, evolution, bath resets and readout.
    Without a thermal schedule, X and total counts are reported as ranges.
    A supplied schedule fixes only the example preparation's X count.  For
    thermal P>1, fresh unexecuted trials retain the full [0, P*K] X range;
    they do not repeat that example's recorded bath-bit population.
    These are finite workloads, not a mixing or statistical-success bound.
    """
    case = _case(case)
    _integer(repetitions, "repetitions", 1, MAX_REPETITIONS)
    if schedule is not None:
        validate_schedule(case, schedule)
    x_actual = sum(schedule["bath_bits"]) if schedule is not None else 0 if case.mode == "ground" else None
    x_range = [x_actual, x_actual] if x_actual is not None else [0, case.rounds]
    kl = case.rounds * case.steps
    fixed_counts = {
        "rz": (4 * case.n + 1) * kl,
        "h": 4 * kl,
        "cx": (4 * case.n - 2) * kl,
        "reset": case.n + case.rounds + 1,
        "measure": case.n,
    }
    base = (8 * case.n + 3) * kl + 2 * case.n + case.rounds + 1
    instruction_range = [base + x_range[0], base + x_range[1]]
    if case.mode == "thermal" and repetitions > 1:
        repeated_x_range = [0, repetitions * case.rounds]
        repeated_instruction_range = [repetitions * base, repetitions * (base + case.rounds)]
    else:
        repeated_x_range = [repetitions * x_range[0], repetitions * x_range[1]]
        repeated_instruction_range = [repetitions * value for value in instruction_range]
    actual_counts = dict(fixed_counts)
    actual_counts["x"] = x_actual
    result = {
        "method_id": METHOD_ID,
        "status": "formula_only_unexecuted_projection",
        "case_id": case.case_id,
        "system_qubits": case.n,
        "bath_qubits": 1,
        "logical_qubits": case.n + 1,
        "output_bits_per_preparation": case.n,
        "rounds": case.rounds,
        "steps_per_round": case.steps,
        "mode": case.mode,
        "initial": "reset_prepared_all_zero_only",
        "primitive_counts_per_preparation": actual_counts,
        "primitive_count_ranges_per_preparation": {
            **{key: [value, value] for key, value in fixed_counts.items()}, "x": x_range,
        },
        "instruction_count_per_preparation": base + x_actual if x_actual is not None else None,
        "instruction_count_range_per_preparation": instruction_range,
        "unitary_instructions_per_step": 8 * case.n + 3,
        "repetitions": repetitions,
        "complete_preparations_projected": repetitions,
        "repetition_scope": "projection_only_full_preparation_each_trial_emitter_P1_only",
        "repeated_X_scope": (
            "fresh_thermal_P_greater_than_1_full_X_range_does_not_reuse_recorded_example"
            if case.mode == "thermal" and repetitions > 1
            else "P1_example_or_unsupplied_range_ground_X_always_zero"
        ),
        "repeated_primitive_count_ranges": {
            **{key: [repetitions * value, repetitions * value] for key, value in fixed_counts.items()},
            "x": repeated_x_range,
        },
        "repeated_instruction_count_range": repeated_instruction_range,
        "total_system_readouts": repetitions * case.n,
        "rotation_count_per_preparation": fixed_counts["rz"],
        "repeated_rotation_count": repetitions * fixed_counts["rz"],
        "ideal_evolution_time_per_preparation": 2 * case.rounds,
        "ideal_evolution_time_repeated": 2 * case.rounds * repetitions,
        "ideal_time_scope": "Hamiltonian_duration_only_excludes_gates_reset_readout_and_controller",
        "logical_rng_calls_per_schedule": _rng_counts(case),
        "repeated_logical_rng_calls_projection": {
            key: repetitions * value for key, value in _rng_counts(case).items()
        },
        "repeated_draw_scope": "fresh_classical_choices_per_projected_trial_no_trials_or_draws_executed",
        "rng_count_scope": "logical_randrange_and_random_calls_not_MT19937_internal_words",
        "classical_frequency_nodes": 3,
        "frequency_law_scope": "nodes_do_not_multiply_a_sampled_rounds_gate_count",
        "accuracy": {
            "finite_circuit_reference": "unexecuted",
            "trotter_error": "uncertified",
            "quadrature_error": "uncertified",
            "finite_pulse_bias": "uncertified",
            "continuum_bridge": "unavailable",
            "mixing_certificate": False,
            "gibbs_accuracy_certificate": False,
            "sampling_confidence": "uncomputed",
        },
        "excluded_costs": {
            "nonzero_system_input_preparation": "unsupported",
            "rotation_synthesis_T_count_and_depth": "uncomputed_unsynthesized_RZ",
            "CX_routing_and_connectivity": "uncomputed_logical_all_to_all_access_assumed",
            "gate_scheduling_depth_and_idle_exposure": "uncomputed",
            "classical_schedule_generation_time": "unmeasured_logical_draw_calls_only",
            "reset_controller_readout_noise_and_timing": "uncomputed",
            "physical_error_factories_distance_qubits_runtime_and_success": "uncomputed",
        },
        "source_limits": {
            "n_max": MAX_N, "rounds_max": MAX_ROUNDS, "steps_max": MAX_STEPS,
            "instructions_per_file_max": MAX_INSTRUCTIONS, "utf8_bytes_per_file_max": MAX_QASM_BYTES,
            "intersection_enforced": True,
        },
        "all_components_verified": False,
    }
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_COUNT_METADATA_BYTES:
        raise ValueError("count metadata exceeds the 64 KiB cap")
    return result


def _ledger_guard(ledger: object | None) -> None:
    if ledger is not None:
        guard = getattr(ledger, "guard", None)
        if guard is not None:
            if not callable(guard):
                raise ValueError("optional ledger guard must be callable")
            guard()


def _ledger_consume(ledger: object | None, name: str, amount: int) -> None:
    if ledger is not None:
        consume = getattr(ledger, "consume", None)
        if not callable(consume):
            raise ValueError("optional ledger must provide consume(name, amount)")
        consume(name, amount)
    _ledger_guard(ledger)


def write_qasm(case: Case, schedule: dict, path: str | Path, *, ledger: object | None = None) -> dict:
    """Write one bounded preparation with exclusive, atomic final publication.

    All cases/choices/counts are checked before opening a file.  A private
    temporary file in the destination directory is linked to the requested
    name only after byte/count/hash completion; an existing destination is
    never replaced.  Failure removes only this writer's temporary file.

    Ordinary callers need no ledger or private runner.  An optional ledger may
    implement guard(), consume("source_primitives"|"source_bytes", amount),
    and record_source_operation(case, operation).  The record callback sees
    each actual emitted primitive after its line and bytes are accepted.
    No circuit is retained by this writer, and no RNG is called.
    """
    case = _case(case)
    validate_schedule(case, schedule)
    # Immutable snapshots ensure preflight and emission use identical choices.
    snapshot = {
        **schedule,
        "sites": list(schedule["sites"]),
        "frequency_indices": list(schedule["frequency_indices"]),
        "bath_bits": list(schedule["bath_bits"]),
        "logical_rng_calls": dict(schedule["logical_rng_calls"]),
    }
    expected = count_source(case, snapshot)
    expected_count = expected["instruction_count_per_preparation"]
    expected_primitives = expected["primitive_counts_per_preparation"]
    target = Path(path)
    if len(target.name.encode("utf-8")) > 128 or target.name in ("", ".", ".."):
        raise ValueError("destination basename must be nonempty and at most 128 UTF-8 bytes")
    if not target.parent.is_dir():
        raise ValueError("destination parent directory must already exist")
    if os.path.lexists(target):
        raise FileExistsError(f"destination already exists: {target.name}")
    callback = getattr(ledger, "record_source_operation", None) if ledger is not None else None
    if callback is not None and not callable(callback):
        raise ValueError("optional record_source_operation must be callable")
    _ledger_guard(ledger)
    # Header and primitive maximum-line bound precede any file allocation.
    header = qasm_header(case.n).encode("ascii")
    if len(header) + 64 * expected_count > MAX_QASM_BYTES:
        raise ValueError("source preflight exceeds the QASM byte cap")
    if ledger is not None and not callable(getattr(ledger, "consume", None)):
        raise ValueError("optional ledger must provide consume(name, amount)")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    digest = hashlib.sha256()
    byte_count = 0
    instruction_count = 0
    primitive_counts = dict.fromkeys(_PRIMITIVES, 0)
    try:
        with os.fdopen(fd, "wb") as output:
            _ledger_consume(ledger, "source_bytes", len(header))
            output.write(header)
            digest.update(header)
            byte_count += len(header)
            for operation in iter_operations(case, snapshot):
                line = operation_qasm(operation).encode("ascii")
                if len(line) > 64:
                    raise ValueError("primitive line exceeds the preflight serialization bound")
                if instruction_count + 1 > MAX_INSTRUCTIONS or byte_count + len(line) > MAX_QASM_BYTES:
                    raise ValueError("actual emission exceeds the source instruction or byte cap")
                _ledger_consume(ledger, "source_primitives", 1)
                _ledger_consume(ledger, "source_bytes", len(line))
                output.write(line)
                digest.update(line)
                byte_count += len(line)
                instruction_count += 1
                primitive_counts[operation.name] += 1
                if callback is not None:
                    callback(case, operation)
                _ledger_guard(ledger)
            if instruction_count != expected_count or primitive_counts != expected_primitives:
                raise RuntimeError("actual emitted primitives disagree with the complete-source formula")
            output.flush()
            os.fsync(output.fileno())
        _ledger_guard(ledger)
        os.link(temporary_name, target)
    finally:
        if os.path.lexists(temporary_name):
            os.unlink(temporary_name)
    return {
        "case_id": case.case_id,
        "file": target.name,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "instruction_count": instruction_count,
        "primitive_counts": primitive_counts,
        "source_preparations_materialized": 1,
        "status": "source_emitted_no_quantum_simulation_or_accuracy_certificate",
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _fixed_metadata(value: object, expected: dict, label: str) -> None:
    metadata = _keys(value, set(expected), label)
    for key, fixed in expected.items():
        actual = metadata[key]
        if type(fixed) is float:
            if _finite_number(actual, f"{label}.{key}") != fixed:
                raise ValueError(f"{label}.{key} differs from the fixed model")
        elif type(fixed) is int:
            if type(actual) is not int or actual != fixed:
                raise ValueError(f"{label}.{key} differs from the fixed integer contract")
        elif type(fixed) is list:
            if type(actual) is not list or len(actual) != len(fixed):
                raise ValueError(f"{label}.{key} has the wrong fixed array shape")
            for entry, fixed_entry in zip(actual, fixed, strict=True):
                if _finite_number(entry, f"{label}.{key} entry") != fixed_entry:
                    raise ValueError(f"{label}.{key} differs from the fixed law")
        elif type(actual) is not type(fixed) or actual != fixed:
            raise ValueError(f"{label}.{key} differs from the fixed contract")


def load_cases(path: str | Path) -> tuple[Case, ...]:
    """Load at most nine bounded cases from the exact fixed scientific schema.

    Files are limited to 128 KiB before decoding.  Duplicate keys, extra keys,
    nonfinite values, bool/integral-float integer parameters, changed model or
    pulse values, and inconsistent validation roles are rejected.  Role metadata
    is checked here but is not part of the Case dataclass.  The loader performs
    no RNG draw, source generation, reference solve, or numerical state work.
    """
    with Path(path).open("rb") as source:
        encoded = source.read(MAX_INPUT_BYTES + 1)
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("cases JSON exceeds the 128 KiB input cap")
    try:
        payload = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cases file must contain valid UTF-8 JSON") from error
    payload = _keys(payload, {
        "schema_version", "method_id", "status", "system", "bath", "pulse",
        "initial", "readout", "cases", "generation_limits", "claim_limits",
    }, "cases root")
    for key, fixed in {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "status": "unexecuted_circuit_and_reference_cases",
        "initial": "all_zero",
        "readout": "all_system_z",
    }.items():
        if type(payload[key]) is not type(fixed) or payload[key] != fixed:
            raise ValueError(f"cases root {key} differs from the fixed contract")
    _fixed_metadata(payload["system"], {
        "family": "open_diagonal_ising_chain", "field_base": 1.0,
        "field_increment": 0.125, "field_period": 3, "zz_coefficient": 0.125,
    }, "system")
    _fixed_metadata(payload["bath"], {
        "frequency_law": "GL3_uniform_0_2", "frequency_nodes": list(_NODES),
        "frequency_weights": list(_WEIGHTS), "beta": 1.0,
        "coupling_sampler": "uniform_site_x",
    }, "bath")
    _fixed_metadata(payload["pulse"], {
        "duration": 2.0, "center": 1.0, "sigma": 0.5, "alpha": 0.08,
    }, "pulse")
    _fixed_metadata(payload["generation_limits"], {
        "n_max": MAX_N, "rounds_max": MAX_ROUNDS, "steps_max": MAX_STEPS,
        "instructions_per_file_max": MAX_INSTRUCTIONS, "utf8_bytes_per_file_max": MAX_QASM_BYTES,
        "input_bytes_max": MAX_INPUT_BYTES, "cases_max": MAX_CASES,
    }, "generation_limits")
    _fixed_metadata(payload["claim_limits"], {
        "all_components_verified": False, "continuum_bridge": "unavailable",
        "mixing_certificate": False, "gibbs_accuracy_certificate": False,
        "physical_budgets": "uncomputed",
    }, "claim_limits")
    entries = payload["cases"]
    if type(entries) is not list or not 1 <= len(entries) <= MAX_CASES:
        raise ValueError("cases must be a list of one through nine entries")
    cases: list[Case] = []
    identifiers: set[str] = set()
    for entry in entries:
        entry = _keys(entry, {"case_id", "n", "rounds", "steps", "mode", "seed", "role"}, "case entry")
        case = Case(**{key: entry[key] for key in ("case_id", "n", "rounds", "steps", "mode", "seed")})
        expected_role = "small_validation" if case.n <= 2 else "generation_only"
        if type(entry["role"]) is not str or entry["role"] != expected_role:
            raise ValueError("case role must be small_validation for n<=2 and generation_only for n>2")
        if case.case_id in identifiers:
            raise ValueError("case identifiers must be unique")
        identifiers.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def _cli_positive_int(text: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise argparse.ArgumentTypeError("expected a plain positive decimal integer")
    return int(text)


def main(argv: list[str] | None = None) -> int:
    """Ordinary source-generation or formula-count CLI; never a density solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, help="fixed scientific cases JSON, at most 128 KiB")
    parser.add_argument("--output-dir", type=Path, help="directory for exclusive per-case QASM files")
    parser.add_argument("--count", nargs=3, type=_cli_positive_int, metavar=("N", "K", "L"), help="formula-only bounded N K L")
    parser.add_argument("--mode", choices=("ground", "thermal"), default="thermal", help="count mode; source modes come from the cases JSON")
    parser.add_argument("--repetitions", type=_cli_positive_int, default=1, help="formula-only P<=1,000,000; source always P=1")
    args = parser.parse_args(argv)
    try:
        if args.count is not None:
            if args.cases is not None or args.output_dir is not None:
                parser.error("--count cannot be combined with --cases or --output-dir")
            n, rounds, steps = args.count
            result = count_source(Case("count", n, rounds, steps, args.mode, 1), repetitions=args.repetitions)
        else:
            if args.cases is None or args.output_dir is None:
                parser.error("source generation requires --cases PATH --output-dir PATH")
            if args.repetitions != 1:
                parser.error("source generation materializes P=1; use --count for repetition projections")
            cases = load_cases(args.cases)
            schedules = [make_schedule(case) for case in cases]
            for case, schedule in zip(cases, schedules, strict=True):
                validate_schedule(case, schedule)
                count_source(case, schedule)
            targets = [args.output_dir / f"{case.case_id}.qasm" for case in cases]
            if any(os.path.lexists(target) for target in targets):
                parser.error("source destination already exists; choose an unused output directory")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            exports = []
            for case, schedule, target in zip(cases, schedules, targets, strict=True):
                exports.append({"schedule": schedule, "receipt": write_qasm(case, schedule, target)})
            result = {"method_id": METHOD_ID, "status": "source_only_no_validation_claim", "exports": exports}
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_COUNT_METADATA_BYTES:
            raise ValueError("CLI metadata exceeds the 64 KiB cap")
        print(encoded)
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
