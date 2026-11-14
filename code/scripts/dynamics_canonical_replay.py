#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded ideal replay of authenticated canonical dynamics QASM snapshots.

The caller authenticates records and parser source before this module imports
scientific libraries. Replay never rebuilds a source circuit, fits phases, or
feeds newly evaluated probabilities into physical-resource accounting.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import time
import types

MAX_CASES = 24
MAX_QASM_BYTES = 2 * 1024 * 1024
MAX_AGGREGATE_QASM_BYTES = 8 * 1024 * 1024
MAX_OPERATIONS_PER_CASE = 100_000
MAX_AGGREGATE_OPERATIONS = 300_000
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SYSTEM_QUBITS = 6
MAX_TOTAL_QUBITS = 9
MAX_WALL_SECONDS = 540
PARSER_SOURCE = "scripts/tail_resource_qasm.py"
CANONICAL_PRIMITIVES = frozenset({"cx", "rx", "ry", "rz", "x", "h", "t", "tdg", "sdg", "barrier", "measure"})
NUMERICAL_SANITY_GUARD = 1e-10
OAA_POLYNOMIAL_GUARD = 1e-11
ARCHIVED_ACCEPTANCE_GUARD = 1e-9
OAA_AMPLITUDES = (Fraction(1, 4), Fraction(11, 16), Fraction(61, 64), Fraction(251, 256))


def require(condition, message):
    if not condition:
        raise ValueError("Canonical dynamics replay: " + message)


def _number(value, label, *, lower=None, upper=None):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite")
    if lower is not None:
        require(value >= lower, label + " is below its bound")
    if upper is not None:
        require(value <= upper, label + " is above its bound")
    return float(value)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _scientific():
    # Deliberately deferred until authenticated snapshots reach replay.
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
    from scipy.linalg import expm
    return np, QuantumCircuit, Operator, Statevector, expm


def explicit_hamiltonian(n, coupling, transverse_field):
    """Independent little-endian Kronecker reference, without source builders."""
    require(type(n) is int and 1 <= n <= MAX_SYSTEM_QUBITS, "reference system-size cap exceeded")
    coupling = _number(coupling, "coupling")
    transverse_field = _number(transverse_field, "transverse field")
    np, _, _, _, _ = _scientific()
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)

    def tensor(selected):
        result = np.ones((1, 1), dtype=complex)
        for wire in range(n - 1, -1, -1):
            result = np.kron(result, selected.get(wire, identity))
        return result

    result = np.zeros((1 << n, 1 << n), dtype=complex)
    for wire in range(n - 1):
        result += coupling * tensor({wire: z, wire + 1: z})
    for wire in range(n):
        result += transverse_field * tensor({wire: x})
    require(np.isfinite(result).all() and np.array_equal(result, result.conj().T), "reference Hamiltonian is not finite/Hermitian")
    return result


def _validate_parsed(parsed, case):
    width = case["logical_qubits"]
    require(type(width) is int and 1 <= width <= MAX_TOTAL_QUBITS, "recorded total-width cap exceeded")
    require(parsed.num_qubits == parsed.num_clbits == width, "parsed width differs from recorded complete readout")
    require(parsed.definition_count == 0, "canonical dynamics QASM cannot contain custom definitions")
    require(parsed.operations == parsed.top_level_operations, "canonical dynamics QASM must be an explicit primitive stream")
    require(0 < len(parsed.operations) <= MAX_OPERATIONS_PER_CASE, "canonical operation cap exceeded")
    first_measure = None
    for index, operation in enumerate(parsed.operations):
        require(operation.name in CANONICAL_PRIMITIVES, "canonical opcode is outside the frozen primitive subset")
        require(all(type(wire) is int and 0 <= wire < width for wire in operation.qubits), "invalid quantum operand")
        require(len(set(operation.qubits)) == len(operation.qubits), "repeated quantum operand")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in operation.parameters), "nonfinite primitive parameter")
        if operation.name == "measure":
            if first_measure is None:
                first_measure = index
            require(len(operation.qubits) == len(operation.clbits) == 1 and not operation.parameters, "invalid final measurement")
        elif operation.name == "barrier":
            require(first_measure is None and not operation.parameters and not operation.clbits,
                    "barrier cannot hide a post-measurement operation")
        else:
            require(first_measure is None and not operation.clbits, "unitary operation occurs after final readout")
            arity = 2 if operation.name == "cx" else 1
            parameters = 1 if operation.name in {"rx", "ry", "rz"} else 0
            require(len(operation.qubits) == arity and len(operation.parameters) == parameters, "invalid primitive arity")
    require(first_measure == len(parsed.operations) - width, "every wire must have one terminal measurement")
    readout = parsed.operations[first_measure:]
    require(tuple((item.qubits, item.clbits) for item in readout)
            == tuple(((wire,), (wire,)) for wire in range(width)), "final quantum/classical readout mapping differs")
    require(parsed.counts() == case["compiled_gate_counts"], "parsed primitive counts differ from frozen compiled ledger")


def parsed_circuit(parsed, case):
    """Translate only the already-validated explicit stored primitive stream."""
    _validate_parsed(parsed, case)
    _, QuantumCircuit, _, _, _ = _scientific()
    circuit = QuantumCircuit(parsed.num_qubits, parsed.num_clbits)
    for operation in parsed.operations:
        if operation.name == "measure":
            circuit.measure(operation.qubits[0], operation.clbits[0])
        elif operation.name == "barrier":
            circuit.barrier(*operation.qubits)
        else:
            getattr(circuit, operation.name)(*operation.parameters, *operation.qubits)
    require(dict(circuit.count_ops()) == case["compiled_gate_counts"], "translated primitive counts differ")
    require(circuit.depth() == case["compiled_depth"], "translated dependency depth differs")
    phase = _number(case["compiled_global_phase_radians"], "archived global phase")
    scope = case["qasm_global_phase_scope"]
    require(isinstance(scope, str) and scope.startswith("Measured trial;") and "controlled oracle" in scope,
            "archived phase must retain the measured-trial scope")
    circuit.global_phase = phase
    return circuit.remove_final_measurements(inplace=False)


def _case_parameters(case):
    n = case["system_qubits"]
    require(type(n) is int and 2 <= n <= MAX_SYSTEM_QUBITS, "canonical system-size cap exceeded")
    require(_number(case["J"], "canonical coupling") == 1
            and _number(case["h"], "canonical transverse field") == .7
            and _number(case["hbar"], "canonical hbar") == 1,
            "canonical Hamiltonian parameters differ")
    require(case["initial_state"] == "0" * n and case["observable"] == "Z(q0)",
            "canonical preparation/observable differs")
    time_value = _number(case["time"], "evolution time", lower=.5, upper=2)
    task = case["task"]
    algorithm_bias = _number(task["algorithm_bias"], "frozen algorithm bias", lower=0, upper=.02)
    require(algorithm_bias > 0, "frozen algorithm bias must be positive")
    if case["method"] == "product_formula":
        require(case["logical_qubits"] == n and case["order"] in (1, 2) and type(case["order"]) is int,
                "product-formula parameters differ")
    elif case["method"] == "QSP_oblivious_amplification":
        require(n == 3 and case["logical_qubits"] == 8 and time_value == 1
                and type(case["rounds"]) is int and 0 <= case["rounds"] <= 3, "OAA parameters differ")
    elif case["method"] == "QSP_retargeted":
        require(n in (2, 4) and case["logical_qubits"] == {2: 6, 4: 9}[n]
                and time_value == 1 and type(case["rounds"]) is int and case["rounds"] == 0, "retargeted parameters differ")
    else:
        raise ValueError("Canonical dynamics replay: unsupported canonical method")
    return n, time_value, algorithm_bias


def _reference(n, time_value):
    np, _, _, _, expm = _scientific()
    hamiltonian = explicit_hamiltonian(n, 1, .7)
    unitary = expm(-1j * time_value * hamiltonian)
    residual = float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(1 << n), 2))
    require(np.isfinite(unitary).all() and residual <= NUMERICAL_SANITY_GUARD, "dense-reference unitarity sanity check failed")
    signs = np.array([1 if basis & 1 == 0 else -1 for basis in range(1 << n)], dtype=float)
    return unitary, signs, residual


def _product_result(case, circuit, target, signs):
    np, _, Operator, _, _ = _scientific()
    actual = Operator(circuit).data
    require(actual.shape == target.shape and np.isfinite(actual).all(), "product-formula matrix shape/finite check failed")
    residual = float(np.linalg.norm(actual.conj().T @ actual - np.eye(len(actual)), 2))
    require(residual <= NUMERICAL_SANITY_GUARD, "canonical product-formula unitarity sanity check failed")
    upper = _number(case["analytic_error"]["operator_bound_upper_float"], "frozen product-formula operator allowance", lower=0)
    require(case["analytic_error"]["numerical_lowering_guard_added"] == NUMERICAL_SANITY_GUARD,
            "frozen product-formula lowering guard differs")
    operator_error = float(np.linalg.norm(actual - target, 2))
    state = actual[:, 0]
    reference = target[:, 0]
    vector_error = float(np.linalg.norm(state - reference))
    exact_observable = float(np.vdot(reference, signs * reference).real)
    circuit_observable = float(np.vdot(state, signs * state).real)
    observable_error = abs(circuit_observable - exact_observable)
    require(operator_error <= upper and vector_error <= upper, "canonical product-formula error exceeds frozen operator allowance")
    require(2 * upper <= case["task"]["algorithm_bias"] and observable_error <= case["task"]["algorithm_bias"],
            "canonical product-formula bias exceeds frozen task allocation")
    return {
        "operator_error": operator_error, "state_vector_error": vector_error,
        "state_infidelity": max(0., 1 - float(abs(np.vdot(reference, state))**2)),
        "observable_exact": exact_observable, "observable_canonical_qasm": circuit_observable,
        "observable_error": observable_error, "operator_allowance_frozen": upper,
        "canonical_unitarity_residual": residual, "operator_dimension": len(actual),
        "zero_input_state_checks": 1,
        "acceptance_probability": 1, "physical_contract_repriced": False,
    }, None


def _accepted_block(circuit, n):
    np, _, _, Statevector, _ = _scientific()
    require(2 <= n <= 4 and circuit.num_qubits <= MAX_TOTAL_QUBITS, "accepted-block trajectory cap exceeded")
    dimension = 1 << n
    columns = []
    maximum_norm_residual = 0.
    for basis in range(dimension):
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[basis] = 1
        actual = Statevector(initial).evolve(circuit).data
        require(np.isfinite(actual).all(), "canonical basis trajectory is nonfinite")
        residual = abs(float(np.vdot(actual, actual).real) - 1.)
        maximum_norm_residual = max(maximum_norm_residual, residual)
        require(residual <= NUMERICAL_SANITY_GUARD, "canonical basis trajectory norm sanity check failed")
        # System wires are the low bits. These rows project every high success
        # ancilla to zero; no failed trajectory is relabeled an accepted result.
        columns.append(actual[:dimension])
    return np.array(columns).T, maximum_norm_residual


def _qsp_result(case, circuit, target, signs):
    np, _, _, _, _ = _scientific()
    n = case["system_qubits"]
    block, norm_residual = _accepted_block(circuit, n)
    rounds = case["rounds"]
    amplitude = float(OAA_AMPLITUDES[rounds]) if case["method"] == "QSP_oblivious_amplification" else .25
    eta = float(np.linalg.norm(block - amplitude * target, 2))
    retargeted = case["method"] == "QSP_retargeted"
    lowering_guard = case["numerical_checks"].get("numerical_lowering_guard_added", 0.) if retargeted else 0.
    require(not retargeted or lowering_guard == NUMERICAL_SANITY_GUARD, "frozen retargeted lowering guard differs")
    guarded_eta = eta + lowering_guard
    require(guarded_eta < abs(amplitude), "canonical accepted block cannot support a conditional approximation")
    bias_projection = 4 * guarded_eta / (abs(amplitude) - guarded_eta)
    require(bias_projection <= case["task"]["algorithm_bias"], "canonical conditional bias exceeds frozen task allocation")
    singular = np.linalg.svd(block, compute_uv=False)
    pmin, pmax = float(singular[-1]**2), float(singular[0]**2)
    require(np.isfinite(singular).all() and 0 < pmin <= pmax <= 1 + NUMERICAL_SANITY_GUARD,
            "canonical acceptance spectrum is invalid")
    contract = case["postselection_contract"]
    frozen_floor = _number(contract["reference_acceptance_lower"], "frozen acceptance floor", lower=0, upper=1)
    archived_guard = case["numerical_checks"]["acceptance_guard_subtracted"]
    require(archived_guard == ARCHIVED_ACCEPTANCE_GUARD, "archived acceptance reserve differs")
    reserve = archived_guard / 2
    supported_floor = pmin - reserve
    require(supported_floor >= frozen_floor, "canonical numerical acceptance does not support the frozen floor and reserved margin")
    gram = block.conj().T @ block
    gram_min = float(np.linalg.eigvalsh(gram)[0])
    gram_discrepancy = abs(gram_min - pmin)
    require(math.isfinite(gram_min) and gram_discrepancy <= reserve, "SVD/Gram acceptance consistency check failed")
    output = block[:, 0]
    input_probability = float(np.vdot(output, output).real)
    require(pmin - reserve <= input_probability <= pmax + reserve,
            "zero-input acceptance disagrees with the canonical spectrum")
    conditional = output / math.sqrt(input_probability)
    reference = target[:, 0]
    exact_observable = float(np.vdot(reference, signs * reference).real)
    conditional_observable = float(np.vdot(conditional, signs * conditional).real)
    observable_error = abs(conditional_observable - exact_observable)
    require(observable_error <= case["task"]["algorithm_bias"], "canonical zero-input conditional observable exceeds frozen bias")
    return {
        "accepted_basis_columns": 1 << n, "canonical_basis_norm_residual_maximum": norm_residual,
        "ideal_scaled_amplitude_exact": str(OAA_AMPLITUDES[rounds]) if not retargeted else "1/4",
        "accepted_block_error_vs_scaled_exact_U": eta,
        "archived_retargeted_lowering_guard": lowering_guard,
        "conditional_Pauli_bias_numerical_projection": bias_projection,
        "conditional_state_vector_error": float(np.linalg.norm(conditional - reference)),
        "observable_exact": exact_observable, "observable_canonical_conditional": conditional_observable,
        "observable_error": observable_error, "zero_input_acceptance_probability": input_probability,
        "canonical_minimum_acceptance_probability_numeric": pmin,
        "canonical_maximum_acceptance_probability_numeric": pmax,
        "svd_vs_gram_minimum_acceptance_discrepancy": gram_discrepancy,
        "acceptance_numerical_support": {
            "archived_guard": archived_guard, "reserved_half_guard": reserve,
            "frozen_floor": frozen_floor, "raw_margin": pmin - frozen_floor,
            "spare_margin_after_reserved_half_guard": supported_floor - frozen_floor,
            "support_check": "pmin_numeric - archived_guard/2 >= unchanged frozen floor",
            "proven_floating_evaluation_error_bound": False, "certified_physical_confidence": False,
        },
        "frozen_synthesis_model_error_per_attempt": contract["synthesis_model_error_per_attempt"],
        "frozen_attempts_per_accepted_slot": contract["attempts_per_accepted_slot"],
        "frozen_accepted_slots": contract["accepted_slots"],
        "frozen_maximum_complete_attempts": contract["maximum_complete_attempts"],
        "frozen_physical_failure_allowance_per_complete_attempt": contract["physical_failure_allowance_per_complete_attempt"],
        "physical_contract_repriced": False,
    }, block


def _oaa_polynomial_checks(blocks, cases):
    if not blocks:
        return {}
    require(set(blocks) == set(range(4)), "canonical OAA replay requires the complete four-round slice")
    np, _, _, _, _ = _scientific()
    base = blocks[0]
    identity = np.eye(len(base), dtype=complex)
    gram = base.conj().T @ base
    polynomials = [identity, 3 * identity - 4 * gram]
    for rounds in range(1, 3):
        polynomials.append((2 * identity - 4 * gram) @ polynomials[-1] - polynomials[-2])
    errors = {}
    for rounds in range(4):
        error = float(np.linalg.norm(blocks[rounds] - base @ polynomials[rounds], 2))
        require(error <= OAA_POLYNOMIAL_GUARD,
                f"canonical OAA round {rounds} ({cases[rounds]}) block violates the frozen polynomial guard")
        errors[cases[rounds]] = {"canonical_OAA_polynomial_error": error,
                                "archived_polynomial_guard": OAA_POLYNOMIAL_GUARD}
    return errors


def replay_canonical_qasm(qasm_payloads, case_ledgers, parser_module):
    """Replay caller-authenticated byte snapshots; never reread/rebuild records."""
    require(type(qasm_payloads) is dict and isinstance(case_ledgers, (tuple, list))
            and 0 < len(case_ledgers) <= MAX_CASES, "invalid bounded canonical input collection")
    require(all(type(case) is dict and isinstance(case.get("case_id"), str) for case in case_ledgers),
            "invalid canonical ledger")
    case_ids = [case["case_id"] for case in case_ledgers]
    require(all(0 < len(name) <= 128 and all(char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_" for char in name)
                for name in case_ids), "canonical case identifier is outside the bounded grammar")
    require(len(set(case_ids)) == len(case_ids) and set(qasm_payloads) == set(case_ids),
            "canonical QASM/ledger cases differ")
    require(all(type(payload) is bytes and 0 < len(payload) <= MAX_QASM_BYTES for payload in qasm_payloads.values()),
            "canonical QASM snapshot size/type limit exceeded")
    require(sum(map(len, qasm_payloads.values())) <= MAX_AGGREGATE_QASM_BYTES, "canonical aggregate QASM cap exceeded")
    before = hashlib.sha256(_canonical_json(case_ledgers)).hexdigest()
    started = time.monotonic()
    results, references, blocks, block_cases = [], {}, {}, {}
    aggregate_operations = 0
    for case in case_ledgers:
        case_id = case["case_id"]
        print(json.dumps({"status": "canonical_qasm_case_begin", "case_id": case_id}), file=sys.stderr, flush=True)
        payload = qasm_payloads[case_id]
        require(hashlib.sha256(payload).hexdigest() == case["qasm_sha256"], "canonical QASM snapshot differs from its authenticated ledger")
        n, time_value, algorithm_bias = _case_parameters(case)
        parsed = parser_module.parse_qasm(payload, max_bytes=MAX_QASM_BYTES,
            max_qubits=MAX_TOTAL_QUBITS, max_operations=MAX_OPERATIONS_PER_CASE)
        aggregate_operations += len(parsed.operations)
        require(aggregate_operations <= MAX_AGGREGATE_OPERATIONS, "canonical aggregate operation cap exceeded")
        circuit = parsed_circuit(parsed, case)
        if (n, time_value) not in references:
            references[n, time_value] = _reference(n, time_value)
        target, signs, reference_residual = references[n, time_value]
        if case["method"] == "product_formula":
            diagnostics, block = _product_result(case, circuit, target, signs)
        else:
            diagnostics, block = _qsp_result(case, circuit, target, signs)
            if case["method"] == "QSP_oblivious_amplification":
                blocks[case["rounds"]] = block
                block_cases[case["rounds"]] = case_id
        results.append({
            "case_id": case_id, "method": case["method"], "canonical_qasm_sha256": case["qasm_sha256"],
            "system_qubits": n, "logical_qubits": case["logical_qubits"],
            "parsed_operations": len(parsed.operations), "compiled_counts_and_depth_exact": True,
            "final_readout_mapping_exact": True, "frozen_algorithm_bias": algorithm_bias,
            "reference_unitarity_residual": reference_residual,
            "archived_global_phase_radians_for_phase_sensitive_comparison": case["compiled_global_phase_radians"],
            "global_phase_scope": case["qasm_global_phase_scope"],
            "diagnostics": diagnostics,
        })
        print(json.dumps({"status": "canonical_qasm_case_completed", "case_id": case_id}), file=sys.stderr, flush=True)
    polynomials = _oaa_polynomial_checks(blocks, block_cases)
    for result in results:
        result["diagnostics"].update(polynomials.get(result["case_id"], {}))
    require(hashlib.sha256(_canonical_json(case_ledgers)).hexdigest() == before, "replay mutated frozen ledgers")
    answer = {
        "schema": "openquantum-dynamics-canonical-qasm-replay-v1",
        "status": "canonical_qasm_replay_passed", "case_count": len(results), "cases": results,
        "parsed_aggregate_operations": aggregate_operations, "scientific_wall_seconds": time.monotonic() - started,
        "native_estimates": 0, "phase_fits": 0, "factory_searches": 0,
        "source_circuits_regenerated": 0, "physical_contracts_repriced": False,
        "Native_estimates_run": False, "phase_fitter_run": False,
        "factory_search_repeated": False, "physical_budgets_repriced": False,
        "canonical_input_ledger_sha256": before, "formal_certificate": False,
        "limitations": [
            "Only the authenticated measured-trial QASM is replayed; fresh-source circuit reproduction is separate evidence",
            "Dense and statevector computations are numerical ideal checks, not interval/operator/synthesis/hardware certificates",
            "Half the existing acceptance reserve is a numerical support check, not a proven floating-evaluation error bound",
            "Archived overall phase is restored only for phase-sensitive comparisons; no coherently controlled oracle is specified",
            "Original acceptance/synthesis/retry/physical budgets stay authoritative and are never recomputed from fresh probabilities",
            "Clean zero states, physical initialization/reset and classical communication retain the original exclusions",
        ],
    }
    require(len(_canonical_json(answer)) <= MAX_OUTPUT_BYTES, "canonical replay result exceeds bounded JSON size")
    return answer


@contextmanager
def _wall_limit():
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "cannot replace an existing wall timer")
    require(signal.SIGALRM not in signal.pthread_sigmask(signal.SIG_BLOCK, set()), "canonical wall signal is blocked")
    previous = {number: signal.getsignal(number) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP)}
    def expired(number, frame):
        raise TimeoutError("Canonical dynamics replay exceeded wall cap or was terminated")
    for number in previous:
        signal.signal(number, expired)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number, handler in previous.items():
            signal.signal(number, handler)


def _verified_module(payload, name, filename="<authenticated canonical QASM parser>"):
    """Compile one checked snapshot; sys.modules registration is not authority."""
    module = types.ModuleType(name)
    absent = object()
    prior = sys.modules.get(name, absent)
    sys.modules[name] = module
    try:
        exec(compile(payload, filename, "exec"), module.__dict__)
    finally:
        if prior is absent:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior
    return module


def _identity_snapshot(source_root):
    """Read the trusted stdlib identity entry point without cache/pyc authority."""
    require(source_root.is_dir() and not source_root.is_symlink() and source_root == source_root.resolve(),
            "unsafe trusted source root")
    path = source_root / "scripts" / "dynamics_record_release.py"
    require(path.parent.is_dir() and not path.parent.is_symlink() and path.parent == path.parent.resolve(),
            "unsafe identity source parent")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_OUTPUT_BYTES, "identity entry point must be bounded/regular")
        blocks, remaining = [], MAX_OUTPUT_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(65536, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        payload = b"".join(blocks)
        require(len(payload) <= MAX_OUTPUT_BYTES, "identity source snapshot grew beyond its cap")
        return payload
    finally:
        os.close(descriptor)


@contextmanager
def _pinned_output_parent(output, records, source_root):
    """Anchor a new private output before potentially long numerical replay."""
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent == output.parent.resolve(),
            "result path must be new under a regular parent")
    require(not output.is_relative_to(records), "result path overlaps immutable records")
    if output.is_relative_to(source_root):
        require(output.is_relative_to(source_root / ".cache"), "in-checkout result must remain private")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output.parent, flags)
    try:
        _check_output_parent(output, descriptor)
        yield descriptor
    finally:
        os.close(descriptor)


def _check_output_parent(output, descriptor):
    pinned = os.fstat(descriptor)
    current = os.stat(output.parent, follow_symlinks=False)
    require(stat.S_ISDIR(pinned.st_mode) and stat.S_ISDIR(current.st_mode)
            and (pinned.st_dev, pinned.st_ino) == (current.st_dev, current.st_ino)
            and not output.parent.is_symlink() and output.parent == output.parent.resolve(),
            "result parent path changed after anchoring")


def _write_anchored_result(output, descriptor, payload):
    require(type(payload) is bytes and len(payload) <= MAX_OUTPUT_BYTES, "canonical result JSON cap exceeded")
    _check_output_parent(output, descriptor)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    target = os.open(output.name, flags, 0o600, dir_fd=descriptor)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(target, remaining)
            require(written > 0, "canonical result write made no progress")
            remaining = remaining[written:]
    finally:
        os.close(target)


def main(argv=None):
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--records-dir", type=Path, required=True)
    arguments.add_argument("--source-root", type=Path, required=True)
    arguments.add_argument("--manifest-sha256", required=True)
    arguments.add_argument("--output-json", type=Path, required=True)
    args = arguments.parse_args(argv)
    with _wall_limit():
        # This identity module uses only stdlib. Complete identity/authentication
        # occurs before scientific imports or compilation of parser snapshots.
        source_root = args.source_root.absolute()
        gate = _verified_module(_identity_snapshot(source_root), "_trusted_dynamics_identity_source",
                                "<trusted stdlib dynamics identity entry point>")
        source_root = gate.checked_directory(source_root)
        records = gate.checked_directory(args.records_dir)
        gate.verify_dynamics_record_identity(records, source_root, expected_sha256=args.manifest_sha256)
        manifest = gate.load_json(gate.checked_file(records, gate.MANIFEST_NAME), expected_sha256=args.manifest_sha256)
        parser_payload = gate.read_bounded(gate.checked_file(source_root, PARSER_SOURCE))
        require(hashlib.sha256(parser_payload).hexdigest() == manifest["verifier_sources"][PARSER_SOURCE],
                "parser snapshot differs from authenticated source")
        parser_module = _verified_module(parser_payload, "_authenticated_dynamics_canonical_parser")
        ledgers, snapshots = [], {}
        for case_id in gate.CASE_IDS:
            ledger_name = f"{case_id}/logical-resources.json"
            ledger_payload = gate.read_bounded(gate.checked_file(records, ledger_name))
            require(hashlib.sha256(ledger_payload).hexdigest() == manifest["files"][ledger_name]["sha256"],
                    "ledger changed after complete authentication")
            ledger = json.loads(ledger_payload)
            qasm_name = f"{case_id}/measured-trial.qasm"
            payload = gate.read_bounded(gate.checked_file(records, qasm_name), MAX_QASM_BYTES)
            require(hashlib.sha256(payload).hexdigest() == manifest["files"][qasm_name]["sha256"],
                    "QASM changed after complete authentication")
            ledgers.append(ledger)
            snapshots[case_id] = payload
        require(sys.version_info[:2] == (3, 12), "locked Python 3.12 is required")
        for package in ("qiskit", "numpy", "scipy"):
            require(importlib.metadata.version(package) == gate.NATIVE_ENVIRONMENT["packages"][package],
                    "locked scientific package differs: " + package)
        output = args.output_json.absolute()
        with _pinned_output_parent(output, records, source_root) as descriptor:
            result = replay_canonical_qasm(snapshots, ledgers, parser_module)
            payload = _canonical_json(result) + b"\n"
            _write_anchored_result(output, descriptor, payload)
        print(json.dumps({"status": "canonical_qasm_replay_passed", "cases": len(ledgers), "native_estimates": 0}))


if __name__ == "__main__":
    main()
