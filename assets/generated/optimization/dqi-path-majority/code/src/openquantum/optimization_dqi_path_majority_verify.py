# SPDX-License-Identifier: MIT
"""Independent opt-in verification of one path-local majority decoder.

Source only: this module and its proposed numerical invocation are unexecuted.
The supported component has four syndrome, five arbitrary target and three
initially clean scratch wires. It is not a validator for the 31-wire degree-two
program, its preparation, its optimization distribution or physical resources.

No producer is imported. The reference uses a direct majority of five bits;
the emitted circuit is parsed as exact phase-preserving CCX words and CX gates.
One nonuniform signed coherent input covers all 512 clean-scratch basis inputs.
Uniform target-plus inputs are deliberately excluded because target XOR errors
could leave those inputs unchanged.
"""

from __future__ import annotations

import math
from typing import Callable


STATE_WIRES = 12
STATE_AMPLITUDES = 4096
INPUT_SUPPORT = 512
MAX_RECORDS = 525
MAX_QUANTUM_SOURCE_BYTES = 4 << 20
ABSOLUTE_TOLERANCE = 2e-10
UNITARY_NAMES = frozenset({"h", "cx", "t", "tdg"})
COMPONENT_COUNTS = {"h": 68, "cx": 219, "t": 136, "tdg": 102}
RESULT_SCHEMA = "openquantum.dqi.path_majority_component_validation.v1"


class PathMajorityVerificationError(ValueError):
    """The component input, authority, source structure or check was rejected."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PathMajorityVerificationError(message)


def _integer(value: object, lower: int, upper: int, label: str) -> int:
    _require(type(value) is int and lower <= value <= upper,
             label + " must be an exact bounded integer")
    return value


def _sequence(value: object, maximum: int, label: str) -> list | tuple:
    _require(type(value) in (list, tuple) and len(value) <= maximum,
             label + " requires a bounded ordinary list or tuple")
    return value


def _no_check() -> None:
    pass


def _check(callback: Callable[[], None]) -> None:
    callback()


def validate_quantum_shape(role: str, *, wires=None, amplitudes=None,
                           matrix_dimension=None) -> None:
    """Pure guard: no state, reference, operator or numerical work is created."""
    _require(type(role) is str and role == "component_state"
             and type(wires) is int and wires == STATE_WIRES
             and type(amplitudes) is int and amplitudes == STATE_AMPLITUDES
             and matrix_dimension is None,
             "Only the twelve-wire/4096-amplitude component state is supported")


class ExecutionBudget:
    """Once-only precharged counters, not execution permission or success proof.

    The normal verification API also requires exact Boolean opt-in. A failed
    or aborted invocation remains spent. Copying or mutating a snapshot cannot
    reset these counters. The 512 Boolean mappings are not state evolutions.
    """

    __slots__ = ("_enabled", "_entered", "_active", "_complete", "_counts")

    def __init__(self, execution_enabled: bool = False):
        _require(type(execution_enabled) is bool,
                 "Execution enablement must be an exact Boolean")
        self._enabled = execution_enabled
        self._entered = self._active = self._complete = False
        self._counts = {"verification_invocations": 0,
                        "complete_statevector_evolutions": 0,
                        "boolean_decoder_mappings": 0,
                        "primitive_unitary_applications": 0,
                        "small_ccx_reference_cases": 0,
                        "native_estimator_calls": 0}

    def begin(self) -> None:
        _require(self._enabled is True, "Numerical verification is disabled")
        _require(not self._entered, "The component verification budget is spent")
        self._entered = self._active = True
        self._counts["verification_invocations"] = 1

    def _charge(self, key: str, ceiling: int) -> None:
        _require(self._active, "An active component verification is required")
        _require(self._counts[key] < ceiling, "The fixed " + key + " ceiling was reached")
        self._counts[key] += 1

    def reserve_full_evolution(self, *, wires=12, amplitudes=4096) -> None:
        validate_quantum_shape("component_state", wires=wires, amplitudes=amplitudes)
        self._charge("complete_statevector_evolutions", 1)

    def reserve_boolean_mapping(self) -> None:
        self._charge("boolean_decoder_mappings", INPUT_SUPPORT)

    def charge_unitary_gate(self) -> None:
        _require(self._counts["complete_statevector_evolutions"] == 1,
                 "Primitive applications require the precharged component state")
        self._charge("primitive_unitary_applications", MAX_RECORDS)

    def finish(self) -> None:
        _require(self._active, "No active component verification can finish")
        _require(self._counts == {"verification_invocations": 1,
                                 "complete_statevector_evolutions": 1,
                                 "boolean_decoder_mappings": 512,
                                 "primitive_unitary_applications": 525,
                                 "small_ccx_reference_cases": 0,
                                 "native_estimator_calls": 0},
                 "Complete verification requires the exact fixed role counters")
        self._active = False
        self._complete = True

    def abort(self) -> None:
        self._active = False
        self._complete = False

    def snapshot(self) -> dict:
        return {**self._counts, "spent": self._entered, "active": self._active,
                "complete": self._complete,
                "counter_scope": "charged entries; passed component and complete flag required"}


def _active_budget(budget: ExecutionBudget) -> None:
    _require(type(budget) is ExecutionBudget and budget._enabled is True
             and budget._entered and budget._active,
             "Numerical work requires an enabled active once-only component budget")


class _QuantumEnvelope:
    """Conservative Python source buffers, separate from whole-process RSS.

    The only quantum-sized objects are the actual and independent-reference
    complex lists and the final serialized real/imaginary pairs. The final
    pairs are an output representation, not an additional evolution.
    """

    __slots__ = ("live", "total", "peak")

    def __init__(self):
        self.live = {}
        self.total = self.peak = 0

    def reserve(self, name: str, count: int, bytes_per_element: int) -> None:
        _require(type(name) is str and 1 <= len(name) <= 16
                 and name in {"actual", "reference", "serialized_pairs"},
                 "An undeclared quantum source buffer is forbidden")
        _integer(count, STATE_AMPLITUDES, STATE_AMPLITUDES, "Quantum element count")
        _integer(bytes_per_element, 1, 256, "Quantum per-element envelope")
        _require(name not in self.live, "A quantum source buffer was reserved twice")
        amount = count * bytes_per_element + 4096
        _require(self.total + amount <= MAX_QUANTUM_SOURCE_BYTES,
                 "The four MiB quantum source envelope was exceeded")
        self.live[name] = amount
        self.total += amount
        self.peak = max(self.peak, self.total)

    def release(self, name: str) -> None:
        _require(type(name) is str and 1 <= len(name) <= 16 and name in self.live,
                 "An unknown quantum buffer cannot be released")
        self.total -= self.live.pop(name)


def validate_primitive_record(value: object) -> tuple:
    """Pure bounded source parser; no quantum allocation or mathematics."""
    _require(type(value) is dict and len(value) <= 4
             and all(type(key) is str and 1 <= len(key) <= 14 for key in value)
             and set(value) <= {"gate", "qubits", "classical", "parameters_hex"}
             and {"gate", "qubits"} <= set(value),
             "A component primitive has unknown or missing fields")
    name = value["gate"]
    _require(type(name) is str and 1 <= len(name) <= 3 and name in UNITARY_NAMES,
             "Only H/CX/T/T-dagger component primitives are admitted")
    raw_qs = _sequence(value["qubits"], 2, "Quantum operands")
    qs = tuple(_integer(q, 0, 11, "Quantum wire") for q in raw_qs)
    _require(len(set(qs)) == len(qs), "Quantum operands must be distinct")
    classical = _sequence(value.get("classical", ()), 0, "Classical operands")
    parameters = _sequence(value.get("parameters_hex", ()), 0, "Parameters")
    _require(not classical and not parameters, "The component has no classical operands or rotations")
    _require(len(qs) == (2 if name == "cx" else 1),
             "A component primitive has the wrong arity")
    return name, qs


def component_input_recipe() -> dict:
    """Pure fixed input specification; normalization is deferred to opt-in work."""
    return {"schema": "openquantum.dqi.path_majority_signed_input.v1",
            "state_wires": 12, "state_amplitudes": 4096,
            "syndrome_wires": [0, 1, 2, 3], "target_wires": [4, 5, 6, 7, 8],
            "scratch_wires": [9, 10, 11], "initial_scratch": "000",
            "support_size": 512, "index_formula": "ell=s+16*t",
            "s_range": [0, 15], "t_range": [0, 31],
            "signed_numerator_formula": "(-1)^((popcount(s)+popcount(t)) mod 2)*(ell+1)",
            "normalization_squared_integer_formula": "512*513*1025//6",
            "amplitude_formula": "signed_numerator/sqrt(normalization_squared)",
            "all_support_magnitudes_distinct": True, "uniform_target_plus": False,
            "independent_output_index_formula": "s+16*(t XOR direct_majority_decode(s))",
            "scratch_zero_required": True, "executed_by_metadata_function": False}


def _ccx_word(a: int, b: int, target: int) -> tuple:
    # An independent spelling of the full phase-exact fifteen-primitive CCX.
    return (("h", (target,)), ("cx", (b, target)), ("tdg", (target,)),
            ("cx", (a, target)), ("t", (target,)), ("cx", (b, target)),
            ("tdg", (target,)), ("cx", (a, target)), ("t", (b,)),
            ("t", (target,)), ("h", (target,)), ("cx", (a, b)),
            ("t", (a,)), ("tdg", (b,)), ("cx", (a, b)))


def _component_logic() -> tuple:
    compute = []
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        compute.extend((("ccx", (a, b, 9)), ("ccx", (9, c, 11)),
                        ("ccx", (a, b, 9))))
    compute.extend((("ccx", (0, 1, 9)), ("ccx", (9, 2, 10)),
                    ("ccx", (10, 3, 11)), ("ccx", (9, 2, 10)),
                    ("ccx", (0, 1, 9))))
    logical = [("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3))]
    logical.extend(compute)
    logical.extend(("cx", (11, target)) for target in range(4, 9))
    logical.extend(("cx", (j, 5 + j)) for j in range(4))
    logical.extend(reversed(compute))
    logical.extend((("cx", (2, 3)), ("cx", (1, 2)), ("cx", (0, 1))))
    return tuple(logical)


def _parsed_component(records: object, ledger: object, check: Callable) -> tuple:
    raw_records = _sequence(records, MAX_RECORDS, "Component records")
    _require(len(raw_records) == MAX_RECORDS, "The component requires exactly 525 source primitives")
    parsed = []
    counts = {name: 0 for name in COMPONENT_COUNTS}
    for record in raw_records:
        _check(check)
        operation = validate_primitive_record(record)
        parsed.append(operation)
        counts[operation[0]] += 1
    expected_ledger = {"schema": "openquantum.dqi.path_majority_component_ledger.v1",
                       "status": "generated_source_unvalidated", "qubits": 12,
                       "syndrome_wires": [0, 1, 2, 3], "target_wires": [4, 5, 6, 7, 8],
                       "scratch_wires": [9, 10, 11], "unitary_operations": 525,
                       "ccx_blocks": 34, "cx_outside_ccx": 15,
                       "primitive_counts": COMPONENT_COUNTS,
                       "promise_maximum_error_weight": 2,
                       "total_extension_uses_terminal_syndrome": False,
                       "arbitrary_target_xor": True, "clean_scratch_required": True,
                       "full_program_validation": False}
    _require(type(ledger) is dict and len(ledger) == len(expected_ledger)
             and all(type(key) is str and 1 <= len(key) <= 48 for key in ledger)
             and set(ledger) == set(expected_ledger),
             "A component ledger has unknown or missing fields")
    for key, expected in expected_ledger.items():
        value = ledger[key]
        _require(type(value) is type(expected), "A component ledger field has an inexact type")
        if type(expected) is dict:
            _require(len(value) == len(expected)
                     and all(type(name) is str and 1 <= len(name) <= 3 for name in value)
                     and set(value) == set(expected)
                     and all(type(value[name]) is int for name in expected),
                     "Component primitive counters must be exact integers")
        if type(expected) is list:
            _require(len(value) == len(expected) and all(type(item) is int for item in value),
                     "Component wire entries must be exact bounded integers")
        _require(value == expected, "The component ledger differs from the fixed local contract")
    _require(counts == COMPONENT_COUNTS, "Component primitive counts differ from the fixed lowering")
    logical = _component_logic()
    expected_records = []
    for name, qs in logical:
        if name == "ccx":
            expected_records.extend(_ccx_word(*qs))
        else:
            expected_records.append((name, qs))
    _require(tuple(parsed) == tuple(expected_records),
             "The component source differs from the explicit majority/cleanup lowering")
    return tuple(parsed), logical


def _direct_majority_decode(syndrome: int) -> int:
    # The independent reference uses threshold directly, not the circuit ANF.
    r = [0]
    prefix = 0
    for j in range(4):
        prefix ^= (syndrome >> j) & 1
        r.append(prefix)
    first = int(sum(r) >= 3)
    return sum((first ^ r[j]) << j for j in range(5))


def _boolean_decoder_checks(logical: tuple, budget: ExecutionBudget,
                            check: Callable) -> dict:
    _active_budget(budget)
    for syndrome in range(16):
        decoded = _direct_majority_decode(syndrome)
        for target in range(32):
            budget.reserve_boolean_mapping()
            _check(check)
            word = syndrome | (target << 4)
            for name, qs in logical:
                if name == "cx":
                    if (word >> qs[0]) & 1:
                        word ^= 1 << qs[1]
                else:
                    if ((word >> qs[0]) & 1) and ((word >> qs[1]) & 1):
                        word ^= 1 << qs[2]
            expected = syndrome | ((target ^ decoded) << 4)
            _require(word == expected,
                     "A Boolean mapping violated arbitrary target XOR, syndrome or clean scratch")
    negative_error = (0, 1, 1, 1, 0)
    negative_syndrome = sum((negative_error[j] ^ negative_error[j + 1]) << j for j in range(4))
    negative_decoded = _direct_majority_decode(negative_syndrome)
    _require(negative_syndrome == 9 and negative_decoded == 17,
             "The explicit off-promise negative example is inconsistent")
    return {"mappings": 512, "syndromes": 16, "arbitrary_clause_targets": 32,
            "scratch_initially_clean": True, "syndrome_and_scratch_checked": True,
            "negative_error_bits": list(negative_error), "negative_internal_syndrome_bits": [1, 0, 0, 1],
            "negative_returned_error_bits": [1, 0, 0, 0, 1],
            "negative_error_weight": 3, "negative_returned_weight": 2,
            "terminal_syndrome_bit_of_error": 0, "terminal_syndrome_bit_of_returned_error": 1,
            "path_error_parity": 1, "returned_path_error_parity": 0,
            "other_path_errors_in_negative_case": "all zero",
            "weight_two_promise_extended": False}


def _signed_states(budget: ExecutionBudget, envelope: _QuantumEnvelope,
                   check: Callable) -> tuple:
    _active_budget(budget)
    budget.reserve_full_evolution(wires=12, amplitudes=4096)
    envelope.reserve("actual", 4096, 128)
    actual = [0j] * STATE_AMPLITUDES
    envelope.reserve("reference", 4096, 128)
    reference = [0j] * STATE_AMPLITUDES
    norm_squared = INPUT_SUPPORT * (INPUT_SUPPORT + 1) * (2 * INPUT_SUPPORT + 1) // 6
    normalization = math.sqrt(norm_squared)
    for syndrome in range(16):
        decoded = _direct_majority_decode(syndrome)
        for target in range(32):
            _check(check)
            index = syndrome + 16 * target
            sign = -1 if (syndrome.bit_count() + target.bit_count()) & 1 else 1
            amplitude = complex(sign * (index + 1) / normalization, 0.0)
            actual[index] = amplitude
            reference[syndrome + 16 * (target ^ decoded)] = amplitude
    return actual, reference, norm_squared


def _apply_in_place(state: list, operation: tuple, budget: ExecutionBudget,
                    check: Callable, square_root_half: float) -> None:
    _active_budget(budget)
    budget.charge_unitary_gate()
    _require(type(state) is list and len(state) == STATE_AMPLITUDES,
             "Only the fixed component state can receive a primitive")
    name, qs = operation
    target = qs[-1]
    mask = 1 << target
    if name == "h":
        for base in range(0, STATE_AMPLITUDES, 2 * mask):
            for offset in range(mask):
                index = base + offset
                if not index & 255:
                    _check(check)
                low, high = state[index], state[index + mask]
                state[index] = (low + high) * square_root_half
                state[index + mask] = (low - high) * square_root_half
    elif name == "cx":
        control = 1 << qs[0]
        for index in range(STATE_AMPLITUDES):
            if not index & 255:
                _check(check)
            if index & control and not index & mask:
                state[index], state[index | mask] = state[index | mask], state[index]
    else:
        phase = complex(square_root_half, square_root_half if name == "t" else -square_root_half)
        for index in range(STATE_AMPLITUDES):
            if not index & 255:
                _check(check)
            if index & mask:
                state[index] *= phase


def _diagnostics(actual: list, reference: list, budget: ExecutionBudget,
                  check: Callable) -> dict:
    _active_budget(budget)
    _require(len(actual) == len(reference) == STATE_AMPLITUDES,
             "Diagnostics require the fixed actual/reference shapes")
    maximum = leakage = norm2 = 0.0
    syndrome_actual = [0.0] * 16
    syndrome_reference = [0.0] * 16
    for index in range(STATE_AMPLITUDES):
        if not index & 255:
            _check(check)
        observed, expected = actual[index], reference[index]
        _require(math.isfinite(observed.real) and math.isfinite(observed.imag),
                 "The component state contains a nonfinite amplitude")
        maximum = max(maximum, abs(observed - expected))
        probability = observed.real * observed.real + observed.imag * observed.imag
        expected_probability = expected.real * expected.real + expected.imag * expected.imag
        norm2 += probability
        syndrome_actual[index & 15] += probability
        syndrome_reference[index & 15] += expected_probability
        if index >> 9:
            leakage += probability
    syndrome_error = max(abs(a - b) for a, b in zip(syndrome_actual, syndrome_reference))
    _require(maximum <= ABSOLUTE_TOLERANCE, "Signed complex component amplitudes differ from the independent reference")
    _require(leakage <= ABSOLUTE_TOLERANCE, "Component scratch did not return clean")
    _require(syndrome_error <= ABSOLUTE_TOLERANCE, "The unconditional syndrome distribution changed")
    _require(abs(norm2 - 1.0) <= ABSOLUTE_TOLERANCE, "The component state norm differs from one")
    return {"max_signed_complex_entry_error": maximum,
            "auxiliary_leakage_probability": leakage,
            "preserved_syndrome_probability_error": syndrome_error,
            "final_norm2": norm2, "tolerance_absolute": ABSOLUTE_TOLERANCE,
            "tolerance_relative": 0.0, "global_phase_fitted": False,
            "state_renormalized": False, "auxiliary_postselection": False}


def verify_component_records(records, ledger, *, allow_numerical_execution=False,
                            budget=None, check: Callable[[], None] = _no_check) -> dict:
    """Verify only the fixed local component after explicit one-shot opt-in.

    The default, inexact authority, missing/disabled budget and spent budget
    paths reject before callback or input access. Source status and proposed
    counts are not themselves numerical execution authorization.
    """
    if allow_numerical_execution is not True:
        raise PathMajorityVerificationError("Exact numerical opt-in is required before any input or callback")
    if type(budget) is not ExecutionBudget:
        raise PathMajorityVerificationError("An explicit exact once-only ExecutionBudget is required")
    budget.begin()
    try:
        _require(callable(check), "The bounded-control callback must be callable")
        _check(check)
        operations, logical = _parsed_component(records, ledger, check)
        boolean_checks = _boolean_decoder_checks(logical, budget, check)
        envelope = _QuantumEnvelope()
        actual, reference, normalization_squared = _signed_states(budget, envelope, check)
        square_root_half = math.sqrt(0.5)
        for operation in operations:
            _apply_in_place(actual, operation, budget, check, square_root_half)
        diagnostics = _diagnostics(actual, reference, budget, check)
        envelope.reserve("serialized_pairs", 4096, 256)
        final_state = []
        for index, value in enumerate(actual):
            if not index & 255:
                _check(check)
            final_state.append([value.real, value.imag])
        result = {"schema": RESULT_SCHEMA, "status": "passed_component_only_numerical_check",
                  "scope": "fixed path-local decoder only; one signed forward pass and 512 Boolean clean-scratch mappings",
                  "state_wires": 12, "state_amplitudes": 4096,
                  "primitive_counts": dict(COMPONENT_COUNTS),
                  "component_input_recipe": component_input_recipe(),
                  "normalization_squared_integer": normalization_squared,
                  "boolean_checks": boolean_checks, **diagnostics,
                  "final_state": final_state,
                  "source_quantum_envelope": {"maximum_bytes": MAX_QUANTUM_SOURCE_BYTES,
                                              "peak_bytes": envelope.peak,
                                              "actual_complex_lists": 1, "reference_complex_lists": 1,
                                              "serialized_pair_lists": 1,
                                              "scope": "conservative Python source buffers; whole-process RSS is separate"},
                  "full_program_validated": False, "degree_two_preparation_validated": False,
                  "optimization_output_validated": False, "physical_resources_estimated": False,
                  "rotation_synthesis_validated": False, "formal_proof_reproduced": False,
                  "hardware_executed": False, "sampling_performed": False,
                  "dense_operator_constructed": False, "notebook_executed": False}
        _check(check)
        budget.finish()
        snapshot = budget.snapshot()
        result["execution_budget"] = snapshot
        result.update({key: snapshot[key] for key in ("verification_invocations",
                      "complete_statevector_evolutions", "boolean_decoder_mappings",
                      "primitive_unitary_applications", "small_ccx_reference_cases", "native_estimator_calls")})
        return result
    except BaseException:
        budget.abort()
        raise
