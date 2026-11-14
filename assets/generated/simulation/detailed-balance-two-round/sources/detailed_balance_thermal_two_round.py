# SPDX-License-Identifier: MIT
"""Original primitive sources for two fixed, accepted finite thermal rounds.

This source-only endpoint is not an executed simulation or a Gibbs sampler.
It preserves the old diagonal endpoint and the accepted one-step correction
sources. The full weak dilation keeps all eight environment labels, and each
round accepts only the all-zero five-bit correction herald. Failure aborts the
whole known-input trajectory; no continuing macro resets the unknown system.

The mathematical correction specification is Chen et al., Nature 646 (2025),
Eq. 19, DOI 10.1038/s41586-025-09583-x. This primitive construction is original.
No simulation, density matrix, diagonalization, sampling or synthesis occurs
in these builders. Scientific imports and construction happen only on calls.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat

if __package__:
    from . import thermal_two_round_contract as contract
else:
    import thermal_two_round_contract as contract

SOURCE_KEYS = (
    "weak_dilation", "weak_reset_prefix", "correction_lcu",
    "round_macro", "two_round_trial",
)
CORRECTION_PIN = {
    "bytes": 24_902,
    "sha256": "62ede4bbafe79a15e7c66068f0dc2b54a32322bbbff4015f00e7a0ba64e3eb48",
}
PRIMITIVE_ARITY = {
    "h": (1, 0, 0), "x": (1, 0, 0), "s": (1, 0, 0),
    "sdg": (1, 0, 0), "t": (1, 0, 0), "tdg": (1, 0, 0),
    "ry": (1, 0, 1), "rz": (1, 0, 1), "p": (1, 0, 1),
    "cx": (2, 0, 0), "reset": (1, 0, 0), "measure": (1, 1, 0),
}


def _dependencies():
    """Load the pinned construction sources only when explicitly requested."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Measure, Reset
    from qiskit.circuit.controlflow import IfElseOp
    from qiskit.circuit.library import (
        CXGate, HGate, PhaseGate, RYGate, RZGate, SGate, SdgGate,
        TGate, TdgGate, XGate,
    )
    if __package__:
        from . import detailed_balance_access as access
        from . import detailed_balance_channel as control
        from . import detailed_balance_correction as correction
    else:
        import detailed_balance_access as access
        import detailed_balance_channel as control
        import detailed_balance_correction as correction
    native = {
        "h": HGate, "x": XGate, "s": SGate, "sdg": SdgGate,
        "t": TGate, "tdg": TdgGate, "p": PhaseGate,
        "ry": RYGate, "rz": RZGate, "cx": CXGate,
        "reset": Reset, "measure": Measure,
    }
    return (QuantumCircuit, QuantumRegister, ClassicalRegister, IfElseOp,
            native, access, control, correction)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _authenticate_correction(module):
    """Bind reuse to the unchanged accepted regular correction source."""
    path = Path(module.__file__).absolute()
    before = path.lstat()
    if (path.name != "detailed_balance_correction.py" or path != path.resolve()
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != CORRECTION_PIN["bytes"]):
        raise ValueError("two-round construction requires the pinned correction source")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0))
    try:
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise ValueError("correction source changed before its bounded read")
        raw = bytearray()
        while len(raw) <= CORRECTION_PIN["bytes"]:
            chunk = os.read(descriptor, min(65_536, CORRECTION_PIN["bytes"] + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if (len(raw) != before.st_size
                or _identity(os.fstat(descriptor)) != _identity(before)
                or _identity(path.lstat()) != _identity(before)
                or hashlib.sha256(raw).hexdigest() != CORRECTION_PIN["sha256"]):
            raise ValueError("correction source differs from its accepted identity")
    finally:
        os.close(descriptor)
    return dict(CORRECTION_PIN)


def preflight_thermal_two_round() -> dict:
    """Declare the fixed source and caps before allocating a circuit.

    Mesh and angle preprocessing are bounded classical construction work,
    not a numerical validation or a thermal-accuracy calculation.
    """
    contract.validate_fixed_input(
        terms=contract.TERMS, proposals=contract.PROPOSALS, beta=contract.BETA,
        clock_bits=contract.CLOCK_BITS, product_steps=contract.PRODUCT_STEPS,
        delta=contract.DELTA, rounds=contract.ROUNDS,
    )
    if (contract.TAU != 0.125 or contract.M != 4 or contract.J != 1
            or contract.MAX_COMPONENT_OPERATIONS != 5_000
            or contract.MAX_PROGRAM_OPERATIONS != 6_500
            or contract.MAX_SOURCE_WIRES != 6 or contract.CLASSICAL_BITS != 11):
        raise ValueError("the fixed two-round source caps or layout changed")
    deps = _dependencies()
    correction = deps[-1]
    correction_pin = _authenticate_correction(correction)
    old = correction.preflight_correction()
    if (correction.MAX_PRIMITIVE_OPERATIONS != 5_000
            or correction.TAU != contract.TAU
            or correction.TERMS != contract.TERMS
            or correction.PROPOSALS != contract.PROPOSALS):
        raise ValueError("the accepted correction component contract changed")
    probabilities = [contract.DELTA * old["eta"] * gamma for gamma in old["gamma"]]
    if (len(probabilities) != 4
            or any(not math.isfinite(value) or not 0 < value <= 1 for value in probabilities)):
        raise ValueError("the fixed weak probabilities are outside the dilation domain")
    return {
        "fixed_input": {
            "hamiltonian": [[coefficient, pauli] for coefficient, pauli in contract.TERMS],
            "proposals": list(contract.PROPOSALS), "beta": contract.BETA,
            "clock_bits": contract.CLOCK_BITS, "product_steps": contract.PRODUCT_STEPS,
            "delta": contract.DELTA, "tau": contract.TAU, "rounds": contract.ROUNDS,
            "M": contract.M, "J": contract.J,
        },
        "eta": old["eta"], "alpha": old["alpha"], "gamma": list(old["gamma"]),
        "beta": contract.BETA, "delta": contract.DELTA,
        "clock_bits": contract.CLOCK_BITS, "product_steps": contract.PRODUCT_STEPS,
        "weak_angles": [2 * math.asin(math.sqrt(value)) for value in probabilities],
        "dt": old["dt"], "dw": old["dw"],
        "time_mesh": old["time_mesh"], "frequency_mesh": old["frequency_mesh"],
        "source_pins": {**old["source_pins"], "detailed_balance_correction.py": correction_pin},
        "maximum_component_operations": 5_000, "maximum_program_operations": 6_500,
        "reserved_quantum_wires": 6, "classical_bits": 11,
        "weak_active_wires": [0, 1, 2, 3], "weak_idle_reserved_wires": [4, 5],
        "weak_environment_outcomes": 8, "correction_auxiliary_outcomes": 32,
        "source_count_status": "static predictions; two-round sources unexecuted",
        **dict(contract.SCOPE_FLAGS),
    }


def _primitive_signature(circuit, item, native, qmap, cmap):
    name = item.operation.name
    if name not in PRIMITIVE_ARITY or item.operation.base_class is not native[name]:
        raise ValueError("source contains an unsupported or composite primitive")
    nq, nc, nparams = PRIMITIVE_ARITY[name]
    if (len(item.qubits) != nq or len(item.clbits) != nc
            or item.operation.num_qubits != nq or item.operation.num_clbits != nc
            or len(item.operation.params) != nparams
            or getattr(item.operation, "condition", None) is not None):
        raise ValueError("source primitive has hidden controls or the wrong arity")
    if any(isinstance(value, bool) for value in item.operation.params):
        raise ValueError("source primitive parameters must be finite numeric angles")
    try:
        params = tuple(float(value) for value in item.operation.params)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("source primitive parameters must be finite numeric angles") from error
    if any(not math.isfinite(value) for value in params):
        raise ValueError("source primitive parameters must be finite numeric angles")
    qubits = tuple(qmap[circuit.find_bit(bit).index] for bit in item.qubits)
    clbits = tuple(cmap[circuit.find_bit(bit).index] for bit in item.clbits)
    if len(set(qubits)) != len(qubits) or len(set(clbits)) != len(clbits):
        raise ValueError("source primitive aliases one of its operands")
    return (name, params, qubits, clbits)


def _linear_signature(circuit, native, *, qubit_map=None, clbit_map=None):
    """Resolve a literal primitive stream, rejecting composite or hidden work."""
    qmap = list(range(circuit.num_qubits)) if qubit_map is None else qubit_map
    cmap = list(range(circuit.num_clbits)) if clbit_map is None else clbit_map
    if len(qmap) != circuit.num_qubits or len(cmap) != circuit.num_clbits:
        raise ValueError("source operand maps have the wrong widths")
    return tuple(_primitive_signature(circuit, item, native, qmap, cmap)
                 for item in circuit.data)


def _circuit_guard(circuit, deps, *, width, classical_bits, allow_if=False, depth=0):
    QuantumCircuit, _, _, IfElseOp, native, *_ = deps
    if (not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != width
            or circuit.num_clbits != classical_bits
            or any(getattr(circuit, name, 0) != 0 for name in
                   ("num_input_vars", "num_captured_vars", "num_declared_vars"))):
        raise ValueError("source has undeclared wires, classical bits or variables")
    cap = contract.MAX_PROGRAM_OPERATIONS if allow_if else contract.MAX_COMPONENT_OPERATIONS
    if depth > 2 or len(circuit.data) > cap:
        raise ValueError("source exceeds its literal tree-depth or operation cap")
    try:
        phase = float(circuit.global_phase)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("source must retain its literal zero global phase") from error
    if not math.isfinite(phase) or phase != 0:
        raise ValueError("source must retain its literal zero global phase")
    if not allow_if:
        _linear_signature(circuit, native)
        return {
            "wires": width, "classical_bits": classical_bits,
            "static_primitive_instances": len(circuit.data),
            "maximum_executed_primitives": len(circuit.data),
            "minimum_executed_primitives": len(circuit.data), "classical_decisions": 0,
        }
    static_count = maximum = minimum = decisions = 0
    qubit_map, clbit_map = list(range(width)), list(range(classical_bits))
    for item in circuit.data:
        if not isinstance(item.operation, IfElseOp):
            _primitive_signature(circuit, item, native, qubit_map, clbit_map)
            static_count += 1
            maximum += 1
            minimum += 1
            continue
        if (type(item.operation) is not IfElseOp
                or len(item.operation.blocks) != 2 or len(item.qubits) != width
                or len(item.clbits) != classical_bits
                or [circuit.find_bit(q).index for q in item.qubits] != list(range(width))
                or [circuit.find_bit(c).index for c in item.clbits] != list(range(classical_bits))):
            raise ValueError("conditional bodies must preserve all reserved operand maps")
        records = [_circuit_guard(body, deps, width=width, classical_bits=classical_bits,
                                  allow_if=True, depth=depth + 1)
                   for body in item.operation.blocks]
        static_count += sum(record["static_primitive_instances"] for record in records)
        maximum += max(record["maximum_executed_primitives"] for record in records)
        minimum += min(record["minimum_executed_primitives"] for record in records)
        decisions += 1 + sum(record["classical_decisions"] for record in records)
    if (static_count > contract.MAX_PROGRAM_OPERATIONS
            or maximum > contract.MAX_PROGRAM_OPERATIONS):
        raise ValueError("conditional source exceeds the 6500 static/path cap")
    return {"wires": width, "classical_bits": classical_bits,
            "static_primitive_instances": static_count,
            "maximum_executed_primitives": maximum,
            "minimum_executed_primitives": minimum, "classical_decisions": decisions}


def _make_components(plan, deps):
    QuantumCircuit, _, _, _, _, access, control, correction = deps
    unitary = access.build_gaussian_access(
        contract.TERMS, contract.PROPOSALS, beta=contract.BETA,
        clock_bits=contract.CLOCK_BITS, product_steps=contract.PRODUCT_STEPS,
    )
    correction._guard(unitary)
    correction._check_counts(unitary, "gaussian_access", 98)
    if float(unitary.global_phase) != 0:
        raise ValueError("the pinned weak access phase changed and requires refreeze")
    weak = QuantumCircuit(4, name="Fixed tilted weak dilation")
    weak.compose(unitary, [0, 1, 2], inplace=True)
    access._multiplexed_ry(weak, [1, 2], 3, plan["weak_angles"])
    if len(weak.data) != 106:
        raise ValueError("the weak access or multiplexor primitive count changed")
    control._conditional_inverse(weak, unitary, 3)
    correction._guard(weak)
    if len(weak.data) != 989:
        raise ValueError("the complete flag-zero inverse primitive count changed")
    prefix = QuantumCircuit(6, name="Fixed tilted weak reset prefix")
    for qubit in range(1, 6):
        prefix.reset(qubit)
    prefix.compose(weak, [0, 1, 2, 3], inplace=True)
    for qubit in range(1, 4):
        prefix.reset(qubit)
    lcu = correction.build_source_bundle()["heralded_step"].copy()
    lcu.name = "Frozen correction LCU for a mixed input"
    correction._guard(lcu)
    correction._check_counts(lcu, "heralded_step", 2_016)
    macro = QuantumCircuit(6, 5, name="Fixed tilted thermal round macro")
    macro.compose(prefix, range(6), inplace=True)
    macro.compose(lcu, range(6), inplace=True)
    for qubit in range(1, 6):
        macro.measure(qubit, qubit - 1)
    for qubit in range(1, 6):
        macro.reset(qubit)
    return {"weak_dilation": weak, "weak_reset_prefix": prefix,
            "correction_lcu": lcu, "round_macro": macro}


def _same_linear(actual, expected, deps, *, clbit_map=None):
    native = deps[4]
    if _linear_signature(actual, native) != _linear_signature(expected, native, clbit_map=clbit_map):
        raise ValueError("source primitive chronology or operand mapping differs from the fixed recipe")


def _herald_condition(circuit, item, deps, expected_bits):
    ClassicalRegister, IfElseOp = deps[2], deps[3]
    condition = getattr(item.operation, "condition", None)
    if (type(item.operation) is not IfElseOp or not isinstance(condition, tuple)
            or len(condition) != 2 or not isinstance(condition[0], ClassicalRegister)
            or type(condition[1]) is not int or condition[1] != 0):
        raise ValueError("the conditional source must compare a complete herald register with zero")
    bits = tuple(circuit.find_bit(bit).index for bit in condition[0])
    if bits != expected_bits:
        raise ValueError("the conditional source reads the wrong herald bits")
    return item.operation.blocks


def _validate_trial(trial, macro, deps):
    QuantumCircuit, _, _, _, native, *_ = deps
    macro_count = len(macro.data)
    if len(trial.data) != macro_count + 3:
        raise ValueError("the complete trial must have reset/H, one round and one conditional")
    first = QuantumCircuit(6, 11)
    first.reset(0)
    first.h(0)
    first.compose(macro, range(6), range(5), inplace=True)
    actual_first = trial.copy_empty_like()
    for item in trial.data[:-1]:
        actual_first.append(item.operation, item.qubits, item.clbits)
    _same_linear(actual_first, first, deps)
    second, early_abort = _herald_condition(trial, trial.data[-1], deps, tuple(range(5)))
    if len(second.data) != macro_count + 1:
        raise ValueError("the late-success body must contain round two and its terminal decision")
    expected_second = QuantumCircuit(6, 11)
    expected_second.compose(macro, range(6), range(5, 10), inplace=True)
    actual_second = second.copy_empty_like()
    for item in second.data[:-1]:
        actual_second.append(item.operation, item.qubits, item.clbits)
    _same_linear(actual_second, expected_second, deps)
    accepted, late_abort = _herald_condition(second, second.data[-1], deps, tuple(range(5, 10)))
    terminal_reset = (("reset", (), (0,), ()),)
    terminal_z = (("measure", (), (0,), (10,)),)
    if (_linear_signature(early_abort, native) != terminal_reset
            or _linear_signature(late_abort, native) != terminal_reset
            or _linear_signature(accepted, native) != terminal_z):
        raise ValueError("terminal branches must reset failed q0 or measure accepted q0 in Z")


def validate_source_bundle(bundle) -> dict:
    """Validate literal sources and control flow, never trusting metadata counts.

    Canonical primitive components are rebuilt from authenticated construction
    sources. This is source validation, not a backend or matrix-reference call.
    """
    if not isinstance(bundle, dict) or set(bundle) != set(SOURCE_KEYS):
        raise ValueError("the source bundle must contain exactly the five fixed sources")
    plan = preflight_thermal_two_round()
    deps = _dependencies()
    canonical = _make_components(plan, deps)
    records = {}
    for name in SOURCE_KEYS:
        width, classical_bits, operations = contract.EXPECTED_SOURCES[name]
        record = _circuit_guard(bundle[name], deps, width=width, classical_bits=classical_bits,
                                allow_if=name == "two_round_trial")
        if (record["static_primitive_instances"] != operations
                or name != "two_round_trial" and operations > contract.MAX_COMPONENT_OPERATIONS):
            raise ValueError("a fixed source primitive count changed")
        records[name] = record
        if name in canonical:
            _same_linear(bundle[name], canonical[name], deps)
    _validate_trial(bundle["two_round_trial"], canonical["round_macro"], deps)
    trial = records["two_round_trial"]
    if (trial["maximum_executed_primitives"] != 6_049
            or trial["minimum_executed_primitives"] != 3_026
            or trial["classical_decisions"] != 2):
        raise ValueError("the full trial differs from the fixed path/control accounting")
    return {
        "source_counts": records,
        "program_static_primitive_instances": trial["static_primitive_instances"],
        "program_maximum_executed_primitives": trial["maximum_executed_primitives"],
        "program_early_abort_primitives": trial["minimum_executed_primitives"],
        "program_classical_decisions": trial["classical_decisions"],
        "maximum_executed_classical_decisions": 2, "early_abort_classical_decisions": 1,
        "reserved_quantum_wires": 6, "reserved_classical_bits": 11,
        "weak_idle_reserved_wires": [4, 5],
        "system_hold": "q0 remains live through environment/herald reset and classical decisions",
        "physical_pricing": "unavailable; idle, hold, controller, reset and readout timing unpriced",
        "structural_check_scope": "literal primitive and branch source validation only",
        **dict(contract.SCOPE_FLAGS),
    }


def build_source_bundle() -> dict:
    """Materialize the fixed sources, with one whole-known-input attempt.

    Retry repetitions belong to classical whole-attempt bookkeeping. This
    function emits no retry loop and performs no numerical backend calls.
    """
    plan = preflight_thermal_two_round()
    deps = _dependencies()
    QuantumCircuit, QuantumRegister, ClassicalRegister = deps[:3]
    bundle = _make_components(plan, deps)
    quantum = QuantumRegister(6, "q")
    c1 = ClassicalRegister(5, "c1")
    c2 = ClassicalRegister(5, "c2")
    out = ClassicalRegister(1, "out")
    qubits = list(quantum)
    clbits = [*c1, *c2, *out]
    trial = QuantumCircuit(quantum, c1, c2, out, name="Two accepted finite thermal rounds")
    accepted = QuantumCircuit(quantum, c1, c2, out, name="Accepted Z readout")
    accepted.measure(quantum[0], out[0])
    early_abort = QuantumCircuit(quantum, c1, c2, out, name="Early abort system reset")
    early_abort.reset(quantum[0])
    late_abort = QuantumCircuit(quantum, c1, c2, out, name="Late abort system reset")
    late_abort.reset(quantum[0])
    second = QuantumCircuit(quantum, c1, c2, out, name="Reached round two")
    second.compose(bundle["round_macro"], qubits, list(c2), inplace=True)
    second.if_else((c2, 0), accepted, late_abort, qubits, clbits)
    trial.reset(quantum[0])
    trial.h(quantum[0])
    trial.compose(bundle["round_macro"], qubits, list(c1), inplace=True)
    trial.if_else((c1, 0), second, early_abort, qubits, clbits)
    bundle["two_round_trial"] = trial
    structural = validate_source_bundle(bundle)
    for name, circuit in bundle.items():
        circuit.metadata = {
            **plan, **dict(contract.SCOPE_FLAGS),
            "milestone": "fixed_thermal_two_round_source_only", "source_kind": name,
            "finite_toy_source_implemented": True,
            "source_operations": structural["source_counts"][name]["static_primitive_instances"],
            "source_account": structural["source_counts"][name],
            "n_system": 1, "system_wires": [0], "clock_wires": [1, 2],
            "weak_flag": 3, "weak_environment_order": "clock + 4*flag; system q0 is least significant",
            "environment_outcomes": 8,
            "correction_outcome_order": "z=sum bit(qk)*2**(k-1), k=1..5",
            "continuation": "mixed unknown system held; only zero correction herald continues",
            "initialization": "known plus is prepared only once at the start of a whole trajectory",
            "retry_contract": "whole known-plus trajectory reset/reprepare; failed intermediate state is not recovered",
            "record_validity": "round2_reached=(c1==0); accept only reached and c2==0; Z invalid on failures",
            "weak_channel": "all eight labels discarded/reset unconditionally; no weak-flag postselection",
            "reference_target": "finite Psi^2 Z observable; exact finite-C correction after the same weak channel",
            "global_phase_convention": "literal zero phase and full controlled U inverse; no phase fitting",
            "quantum_if_controls": 0, "classical_decisions": structural["source_counts"][name]["classical_decisions"],
            "system_hold": structural["system_hold"], "physical_pricing": structural["physical_pricing"],
        }
    return bundle
