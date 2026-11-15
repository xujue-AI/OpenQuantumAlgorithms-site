# SPDX-License-Identifier: MIT
"""Independent fixed primitive targets and complete numeric-angle provenance.

This scientific helper was used in the accepted fixed calculation. The public
packaging is a source-only derivation that has not been executed as a downloaded
package. The module constructs no Qiskit circuit, imports no scientific package
and performs no work at import. The supplied interval evaluator owns outward
arithmetic; native-class identity and live-circuit rechecks are separate
authentication obligations. Names alone do not attest that identity. Every
operation and serialized token is checked, while repeated preparations are
recorded by exhaustive template/range mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

SCHEMA = "openquantum.finite-gibbs.angle-provenance.v1"
PROGRAM_NAMES = (
    "access", "dilation", "reset_round", "preparation_R64", "preparation_R2",
)
EXPECTED_PROGRAMS = {
    "access": (6, 0, 140),
    "dilation": (7, 0, 1493),
    "reset_round": (7, 0, 1505),
    "preparation_R64": (7, 1, 96323),
    "preparation_R2": (7, 1, 3013),
}
NATIVE_CLASSES = {
    "h": "HGate", "x": "XGate", "s": "SGate", "sdg": "SdgGate",
    "t": "TGate", "tdg": "TdgGate", "ry": "RYGate", "rz": "RZGate",
    "p": "PhaseGate", "cx": "CXGate", "reset": "Reset", "measure": "Measure",
}
PARAMETER_GATES = frozenset({"ry", "rz", "p"})
MAX_TARGETS = 512
MAX_PROGRAM_QASM_BYTES = 8 * 1024 * 1024
MAX_ALL_QASM_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class Angle:
    """An exact rational affine combination of independent nonlinear bases."""

    coefficients: tuple

    @classmethod
    def base(cls, name):
        return cls(((name, 1, 1),))

    def scaled(self, numerator, denominator=1):
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise ValueError("symbolic angle coefficients must be exact integers")
        factor = Fraction(numerator, denominator)
        items = []
        for name, n, d in self.coefficients:
            value = Fraction(n, d) * factor
            if value:
                items.append((name, value.numerator, value.denominator))
        return Angle(tuple(items))


def _sum_angles(angles):
    values = {}
    for angle in angles:
        if type(angle) is not Angle:
            raise ValueError("ideal target must be a symbolic angle")
        for name, n, d in angle.coefficients:
            values[name] = values.get(name, Fraction(0)) + Fraction(n, d)
    return Angle(tuple((name, value.numerator, value.denominator)
                       for name, value in sorted(values.items()) if value))


@dataclass(frozen=True)
class Primitive:
    name: str
    wires: tuple
    clbits: tuple = ()
    angle: Angle | None = None
    origin: str = ""


def _op(name, wires, angle=None, origin="", clbits=()):
    if name not in NATIVE_CLASSES or (name in PARAMETER_GATES) != (angle is not None):
        raise ValueError("invalid ideal primitive descriptor")
    return Primitive(name, tuple(wires), tuple(clbits), angle, origin)


def _inverse(sequence):
    inverse_names = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
    result = []
    for op in reversed(sequence):
        if op.name in {"reset", "measure"}:
            raise ValueError("a discard/readout cannot enter a unitary inverse")
        name = inverse_names.get(op.name, op.name)
        angle = op.angle.scaled(-1) if op.angle is not None else None
        result.append(_op(name, op.wires, angle, "inverse(" + op.origin + ")"))
    return tuple(result)


def _multiplex(controls, target, angles, origin):
    """Independent Gray-Walsh descriptor, including the final Gray closure."""
    count = 1 << len(controls)
    if len(angles) != count:
        raise ValueError("ideal multiplexor angle count is wrong")
    if not controls:
        return (_op("ry", (target,), angles[0], origin + ":uncontrolled"),)
    result = []
    for j in range(count):
        word = j ^ (j >> 1)
        terms = (angle.scaled(-1 if (k & word).bit_count() & 1 else 1, count)
                 for k, angle in enumerate(angles))
        result.append(_op("ry", (target,), _sum_angles(terms),
                          origin + ":gray=" + str(j)))
        next_word = ((j + 1) % count) ^ (((j + 1) % count) >> 1)
        changed = word ^ next_word
        if changed <= 0 or changed & (changed - 1):
            raise ValueError("Gray closure did not change exactly one bit")
        result.append(_op("cx", (controls[changed.bit_length() - 1], target),
                          origin=origin + ":gray-close=" + str(j)))
    return tuple(result)


def _cp(angle, control, target, origin):
    half = angle.scaled(1, 2)
    return (
        _op("p", (control,), half, origin + ":control-phase"),
        _op("cx", (control, target), origin=origin + ":parity-1"),
        _op("p", (target,), half.scaled(-1), origin + ":target-minus"),
        _op("cx", (control, target), origin=origin + ":parity-2"),
        _op("p", (target,), half, origin + ":target-plus"),
    )


def _controlled_rotation(name, angle, control, target, origin):
    if name not in {"ry", "rz"}:
        raise ValueError("unsupported ideal controlled rotation")
    half = angle.scaled(1, 2)
    return (
        _op(name, (target,), half, origin + ":half"),
        _op("cx", (control, target), origin=origin + ":parity-1"),
        _op(name, (target,), half.scaled(-1), origin + ":minus-half"),
        _op("cx", (control, target), origin=origin + ":parity-2"),
    )


def _toffoli(a, b, target, origin):
    specification = (
        ("h", (target,)), ("cx", (b, target)), ("tdg", (target,)),
        ("cx", (a, target)), ("t", (target,)), ("cx", (b, target)),
        ("tdg", (target,)), ("cx", (a, target)), ("t", (b,)),
        ("t", (target,)), ("h", (target,)), ("cx", (a, b)),
        ("t", (a,)), ("tdg", (b,)), ("cx", (a, b)),
    )
    return tuple(_op(name, wires, origin=origin + ":step=" + str(i))
                 for i, (name, wires) in enumerate(specification))


def _controlled_lift(op, flag):
    origin = "flag-controlled(" + op.origin + ")"
    if flag in op.wires:
        raise ValueError("weak flag is an access wire")
    if op.name == "cx":
        return _toffoli(flag, op.wires[0], op.wires[1], origin)
    if len(op.wires) != 1:
        raise ValueError("unsupported ideal controlled arity")
    target = op.wires[0]
    if op.name == "x":
        return (_op("cx", (flag, target), origin=origin),)
    if op.name == "h":
        # H = RY(pi/2) Z exactly. Chronology is controlled Z, then controlled RY.
        cz = (_op("h", (target,), origin=origin + ":CZ-left"),
              _op("cx", (flag, target), origin=origin + ":CZ-CX"),
              _op("h", (target,), origin=origin + ":CZ-right"))
        return cz + _controlled_rotation("ry", Angle.base("pi").scaled(1, 2),
                                         flag, target, origin + ":CRY")
    if op.name in {"ry", "rz"}:
        return _controlled_rotation(op.name, op.angle, flag, target, origin)
    if op.name == "p":
        return _cp(op.angle, flag, target, origin)
    if op.name in {"s", "sdg", "t", "tdg"}:
        numerator = -1 if op.name in {"sdg", "tdg"} else 1
        denominator = 2 if op.name in {"s", "sdg"} else 4
        return _cp(Angle.base("pi").scaled(numerator, denominator), flag, target, origin)
    raise ValueError("unsupported ideal inverse primitive")


def _base_enclosures(ev):
    """Exact finite formulas, independent of all original float helpers."""
    pi = ev.pi()
    root = ev.sqrt(ev.div(ev.mul(2, pi), 16))
    dt, dw = ev.div(root, 2), ev.mul(2, root)
    signed = tuple(range(8)) + tuple(range(-8, 0))
    weights = tuple(ev.exp(ev.neg(ev.div(ev.mul(pi, s * s), 16))) for s in signed)
    eta = ev.div(ev.sum(weights), 4)
    bases = {"pi": pi, "dt": dt}
    formulas = {"pi": "Machin pi enclosure", "dt": "sqrt(2*pi/16)/2"}
    for target in range(3, -1, -1):
        width = 1 << target
        for prefix in range(1 << (3 - target)):
            start = prefix << (target + 1)
            left = ev.sum(weights[start:start + width])
            right = ev.sum(weights[start + width:start + 2 * width])
            if left.lo <= 0 or right.lo <= 0:
                raise ValueError("Gaussian prefix masses are not positive")
            name = "prepare:" + str(target) + ":" + str(prefix)
            bases[name] = ev.mul(2, ev.atan2_positive(ev.sqrt(right), ev.sqrt(left)))
            formulas[name] = "2*atan2(sqrt(sum right w),sqrt(sum left w)); W cancels"
    for label, s in enumerate(signed):
        exponent = ev.nonnegative_part(ev.add(ev.mul(dw, s), "0.5"))
        gamma = ev.exp(ev.neg(exponent))
        probability = ev.mul(ev.div(eta, 8), gamma)
        if probability.lo <= 0 or probability.hi >= 1:
            raise ValueError("weak-angle probability is outside (0,1)")
        name = "weak:" + str(label)
        bases[name] = ev.mul(2, ev.atan2_positive(ev.sqrt(probability),
                                                ev.sqrt(ev.sub(1, probability))))
        formulas[name] = "2*atan2(sqrt(delta*eta*gamma),sqrt(1-delta*eta*gamma))"
    if len(bases) != 33:
        raise ValueError("the fixed nonlinear basis inventory changed")
    return bases, formulas


def build_ideal_ledger(evaluator):
    """Return small fixed templates; no quantum circuit or repeated source is built."""
    bases, formulas = _base_enclosures(evaluator)
    access = [_op("h", (5,), origin="proposal-uniform-H")]
    for target in range(3, -1, -1):
        controls = tuple(range(target + 2, 5))
        angles = tuple(Angle.base("prepare:" + str(target) + ":" + str(prefix))
                       for prefix in range(1 << (3 - target)))
        access.extend(_multiplex(controls, target + 1, angles,
                                 "Gaussian-prepare:target=" + str(target)))
    evolution = []
    for k in range(4):
        signed_power = -(1 << k) if k == 3 else 1 << k
        angle = Angle.base("dt").scaled(signed_power)
        for chronological_term in range(2):
            evolution.extend(_controlled_rotation(
                "rz", angle, k + 1, 0,
                "signed-Z:k=" + str(k) + ":term=" + str(chronological_term)))
    access.extend(evolution)
    access.extend((_op("x", (5,), origin="SELECT-X:zero-control-left"),
                   _op("cx", (5, 0), origin="SELECT-X"),
                   _op("x", (5,), origin="SELECT-X:zero-control-right"),
                   _op("h", (0,), origin="SELECT-Z:left-H"),
                   _op("cx", (5, 0), origin="SELECT-Z"),
                   _op("h", (0,), origin="SELECT-Z:right-H")))
    access.extend(_inverse(evolution))
    positive_dft = []
    for j in range(3, -1, -1):
        positive_dft.append(_op("h", (j + 1,), origin="positive-DFT:H=" + str(j)))
        for k in range(j - 1, -1, -1):
            positive_dft.extend(_cp(Angle.base("pi").scaled(1, 1 << (j - k)),
                                    k + 1, j + 1,
                                    "positive-DFT:phase=" + str(k) + ":" + str(j)))
    for j in range(2):
        a, b = j + 1, 4 - j
        positive_dft.extend((_op("cx", (a, b), origin="DFT-swap:first=" + str(j)),
                             _op("cx", (b, a), origin="DFT-swap:middle=" + str(j)),
                             _op("cx", (a, b), origin="DFT-swap:last=" + str(j))))
    access.extend(_inverse(positive_dft))
    access = tuple(access)
    weak = _multiplex((1, 2, 3, 4), 6,
                      tuple(Angle.base("weak:" + str(l)) for l in range(16)), "weak-frequency")
    controlled = [_op("x", (6,), origin="flag-zero:left-X")]
    for op in _inverse(access):
        controlled.extend(_controlled_lift(op, 6))
    controlled.append(_op("x", (6,), origin="flag-zero:right-X"))
    dilation = access + weak + tuple(controlled)
    before = tuple(_op("reset", (q,), origin="round:initial-aux-reset") for q in range(1, 7))
    after = tuple(_op("reset", (q,), origin="round:terminal-aux-reset") for q in range(1, 7))
    reset_round = before + dilation + after
    if (len(access), len(weak), len(controlled), len(dilation), len(reset_round)) != (140, 32, 1321, 1493, 1505):
        raise ValueError("independent fixed primitive counts differ from the frozen recipe")
    return {"bases": bases, "base_formulas": formulas, "access": access,
            "dilation": dilation, "reset_round": reset_round,
            "global_phase": "0", "weak_operations": 32, "controlled_inverse_operations": 1321}


def _expected_operations(name, ideal):
    if name in {"access", "dilation", "reset_round"}:
        yield from ideal[name]
        return
    rounds = 64 if name == "preparation_R64" else 2
    yield _op("reset", (0,), origin="known-plus:system-reset")
    yield _op("h", (0,), origin="known-plus:H")
    for _round in range(rounds):
        yield from ideal["reset_round"]
    yield _op("measure", (0,), origin="terminal-Z", clbits=(0,))


def _closed(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + " has an unexpected shape")


def _numeric(record):
    _closed(record, ("hex", "token"), "numeric parameter")
    encoded, token = record["hex"], record["token"]
    if (type(encoded) is not str or type(token) is not str or len(encoded) > 128 or len(token) > 128
            or re.fullmatch(r"-?0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+", encoded) is None
            or re.fullmatch(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?", token) is None):
        raise ValueError("numeric source representation is not canonical")
    try:
        value = float.fromhex(encoded)
    except (ValueError, OverflowError) as error:
        raise ValueError("numeric source representation is not finite") from error
    if value.hex() != encoded or format(value, ".17g") != token or not -128 <= value <= 128:
        raise ValueError("native hex/canonical decimal parameter pair differs")
    return value


def _validated_snapshot(program, expected_shape):
    _closed(program, ("num_qubits", "num_clbits", "global_phase", "operations"), "program")
    qubits, clbits, count = expected_shape
    if (type(program["num_qubits"]) is not int or program["num_qubits"] != qubits
            or type(program["num_clbits"]) is not int or program["num_clbits"] != clbits
            or type(program["operations"]) is not list or len(program["operations"]) != count):
        raise ValueError("program dimensions/count differ from the frozen source")
    if _numeric(program["global_phase"]) != 0:
        raise ValueError("fixed source global phase must be exactly zero")
    operations = []
    for item in program["operations"]:
        _closed(item, ("name", "wires", "clbits", "params", "native_class"), "primitive")
        name = item["name"]
        if type(name) is not str or name not in NATIVE_CLASSES:
            raise ValueError("unknown source primitive")
        if type(item["native_class"]) is not str or item["native_class"] != NATIVE_CLASSES[name]:
            raise ValueError("source primitive class label differs")
        if (type(item["wires"]) is not list or type(item["clbits"]) is not list
                or any(type(q) is not int or not 0 <= q < qubits for q in item["wires"])
                or any(type(c) is not int or not 0 <= c < clbits for c in item["clbits"])
                or len(set(item["wires"])) != len(item["wires"])):
            raise ValueError("source primitive operands are malformed")
        arity = 2 if name == "cx" else 1
        classical = 1 if name == "measure" else 0
        parameters = 1 if name in PARAMETER_GATES else 0
        if (len(item["wires"]) != arity or len(item["clbits"]) != classical
                or type(item["params"]) is not list or len(item["params"]) != parameters):
            raise ValueError("source primitive arity differs")
        values = tuple(_numeric(parameter) for parameter in item["params"])
        operations.append((name, tuple(item["wires"]), tuple(item["clbits"]),
                           tuple((p["hex"], p["token"]) for p in item["params"]),
                           item["native_class"], values))
    return tuple(operations)


def _qasm_line(record):
    name, wires, clbits, params, _native_class, _values = record
    if name == "reset":
        return "reset q[" + str(wires[0]) + "];"
    if name == "measure":
        return "c[" + str(clbits[0]) + "] = measure q[" + str(wires[0]) + "];"
    suffix = "(" + ", ".join(token for _encoded, token in params) + ")" if params else ""
    operands = ", ".join("q[" + str(q) + "]" for q in wires)
    return name + suffix + " " + operands + ";"


def _qasm_check(text, records, dimensions):
    if type(text) is not str or not text.endswith("\n") or len(text.encode("utf-8")) > MAX_PROGRAM_QASM_BYTES:
        raise ValueError("serialized source is absent, oversized or unterminated")
    header = ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[" + str(dimensions[0]) + "] q;"]
    if dimensions[1]:
        header.append("bit[" + str(dimensions[1]) + "] c;")
    lines = text.split("\n")[:-1]
    if len(lines) != len(header) + len(records) or lines[:len(header)] != header:
        raise ValueError("serialized declarations/operation count differ")
    for line, record in zip(lines[len(header):], records):
        if line != _qasm_line(record):
            raise ValueError("serialized primitive/token differs from the actual snapshot")


def _signature(records):
    # Omit only the derived native float; its exact hex and token are already bound.
    raw = json.dumps([record[:5] for record in records], sort_keys=True,
                     separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class _Registry:
    def __init__(self, ev, ideal):
        self.ev, self.ideal = ev, ideal
        self.targets, self.target_keys = {}, {}
        self.parameters, self.parameter_keys = {}, {}

    def target(self, angle):
        key = angle.coefficients
        if key in self.target_keys:
            identifier = self.target_keys[key]
            return identifier, self.targets[identifier]["interval_object"]
        if len(self.targets) >= MAX_TARGETS:
            raise ValueError("independent primitive-angle target cap exceeded")
        terms = []
        for name, n, d in key:
            if name not in self.ideal["bases"]:
                raise ValueError("unknown independent angle basis")
            terms.append(self.ev.mul(self.ev.div(n, d), self.ideal["bases"][name]))
        interval = self.ev.sum(terms) if terms else self.ev.point(0)
        identifier = "T" + str(len(self.targets)).zfill(3)
        self.target_keys[key] = identifier
        self.targets[identifier] = {"coefficients": [list(item) for item in key],
                                    "ideal_interval": self.ev.record(interval),
                                    "interval_object": interval}
        return identifier, interval

    def parameter(self, angle, encoded, token, value):
        target_id, target = self.target(angle)
        key = (target_id, encoded, token)
        if key in self.parameter_keys:
            identifier = self.parameter_keys[key]
            return identifier, self.parameters[identifier]["joint_error_upper_object"]
        native = self.ev.native_float(value)
        canonical = self.ev.canonical_decimal(token)
        native_delta, token_delta = self.ev.sub(native, target), self.ev.sub(canonical, target)
        upper = max(self.ev.upper_abs(native_delta), self.ev.upper_abs(token_delta))
        identifier = "P" + str(len(self.parameters)).zfill(3)
        self.parameter_keys[key] = identifier
        self.parameters[identifier] = {
            "target_id": target_id, "native_hex": encoded, "canonical_decimal_token": token,
            "native_enclosure": self.ev.record(native), "canonical_enclosure": self.ev.record(canonical),
            "native_deviation": self.ev.record(native_delta), "canonical_deviation": self.ev.record(token_delta),
            "joint_error_upper": str(upper), "joint_error_upper_object": upper,
        }
        return identifier, upper

    def record(self):
        return {
            "independent_angle_target_count": len(self.targets),
            "nonlinear_bases": {name: {"formula": self.ideal["base_formulas"][name],
                                        "enclosure": self.ev.record(value)}
                                 for name, value in self.ideal["bases"].items()},
            "angle_targets": {name: {key: value for key, value in entry.items() if key != "interval_object"}
                              for name, entry in self.targets.items()},
            "parameter_evidence": {name: {key: value for key, value in entry.items()
                                           if key != "joint_error_upper_object"}
                                   for name, entry in self.parameters.items()},
        }


def verify_angle_provenance(actual_snapshots, evaluator, qasm_sources):
    """Exhaustively bind five actual programs and both numeric representations.

    The caller must authenticate actual native base-class objects and keep the
    live circuits signature-bound before and after each scientific call. This
    function checks the corresponding names, complete chronology and tokens.
    Its returned A counts one dilation; exact repeat equality justifies 64A.
    """
    _closed(actual_snapshots, PROGRAM_NAMES, "source inventory")
    _closed(qasm_sources, PROGRAM_NAMES, "serialized source inventory")
    ideal = build_ideal_ledger(evaluator)
    records, signatures, serialized = {}, {}, {}
    qasm_total = 0
    for name in PROGRAM_NAMES:
        shape = EXPECTED_PROGRAMS[name]
        actual = _validated_snapshot(actual_snapshots[name], shape)
        checked = 0
        for record, expected in zip(actual, _expected_operations(name, ideal)):
            if (record[0] != expected.name or record[1] != expected.wires
                    or record[2] != expected.clbits or record[4] != NATIVE_CLASSES[expected.name]):
                raise ValueError("actual primitive chronology differs from the independent target: " + name)
            checked += 1
        if checked != shape[2]:
            raise ValueError("independent source iterator did not cover every primitive")
        text = qasm_sources[name]
        _qasm_check(text, actual, shape)
        qasm_total += len(text.encode("utf-8"))
        if qasm_total > MAX_ALL_QASM_BYTES:
            raise ValueError("combined serialized source exceeds its admission cap")
        records[name], signatures[name], serialized[name] = actual, _signature(actual), text
    if records["dilation"][:140] != records["access"]:
        raise ValueError("access differs from the actual dilation prefix")
    if records["reset_round"][6:1499] != records["dilation"]:
        raise ValueError("reset round differs from the actual dilation")
    for name, rounds in (("preparation_R64", 64), ("preparation_R2", 2)):
        for r in range(rounds):
            start = 2 + r * 1505
            if records[name][start:start + 1505] != records["reset_round"]:
                raise ValueError("a complete preparation round differs in native bits/tokens")
    registry = _Registry(evaluator, ideal)
    template, error_sum = [], evaluator.point(0)
    for index, (record, expected) in enumerate(zip(records["dilation"], ideal["dilation"])):
        parameters = []
        if expected.angle is not None:
            encoded, token = record[3][0]
            identifier, upper = registry.parameter(expected.angle, encoded, token, record[5][0])
            parameters.append(identifier)
            error_sum = evaluator.add(error_sum, evaluator.interval(0, upper))
        template.append({"index": index, "name": expected.name,
                         "wires": list(expected.wires), "clbits": list(expected.clbits),
                         "native_class": record[4], "origin": expected.origin,
                         "parameters": parameters})
    realization = evaluator.mul(128, error_sum)
    certificate = {
        "schema": SCHEMA,
        "scope": "complete fixed primitive/angle provenance; no full-state/general/physical certification",
        "numeric_models": ["exact native binary64 value", "exact canonical decimal source literal"],
        "error_combination": "maximum native/canonical deviation per primitive, summed once per dilation",
        "one_dilation_angle_error_sum": evaluator.record(error_sum),
        "R64_realization_bound": evaluator.record(realization),
        "realization_formula": "2*64*sum per primitive upper max(native deviation,canonical deviation)",
        "global_phase_target": "0",
        "actual_global_phases": {name: dict(actual_snapshots[name]["global_phase"]) for name in PROGRAM_NAMES},
        "dilation_template": template,
        "source_programs": {
            name: {"qubits": EXPECTED_PROGRAMS[name][0], "classical_bits": EXPECTED_PROGRAMS[name][1],
                   "operations_exhaustively_verified": len(records[name]),
                   "signature_sha256": signatures[name],
                   "canonical_qasm_sha256": hashlib.sha256(serialized[name].encode("utf-8")).hexdigest()}
            for name in PROGRAM_NAMES
        },
        "factorized_complete_mapping": {
            "access": {"dilation_indices": [0, 140]},
            "reset_round": {"initial_reset_indices": [0, 6], "dilation_indices": [6, 1499],
                            "terminal_reset_indices": [1499, 1505], "dilation_index_offset": 6},
            "preparation_R64": {"prefix": ["reset q0", "H q0"], "rounds": 64,
                                 "round_index_start": "2+r*1505", "round_template": "reset_round",
                                 "terminal_measure_index": 96322, "readout": "q0->c0"},
            "preparation_R2": {"prefix": ["reset q0", "H q0"], "rounds": 2,
                                "round_index_start": "2+r*1505", "round_template": "reset_round",
                                "terminal_measure_index": 3012, "readout": "q0->c0"},
            "ranges_are_half_open": True,
            "all_native_hex_and_serialized_tokens_equal_across_repeats": True,
        },
        "caller_live_native_class_identity_required": True,
        "native_class_object_identity_verified_by_this_module": False,
        "interval_counters": evaluator.counters(),
        "full_state_Gibbs_verified": False, "general_mixing_certificate": False,
        "physical_error_certified": False, "formal_proof_reproduced": False,
    }
    certificate.update(registry.record())
    # Re-read caller metadata after the long exhaustive audit; no borrowed mutable
    # dictionary is treated as a stable source binding.
    _closed(actual_snapshots, PROGRAM_NAMES, "source inventory")
    _closed(qasm_sources, PROGRAM_NAMES, "serialized source inventory")
    for name in PROGRAM_NAMES:
        if _signature(_validated_snapshot(actual_snapshots[name], EXPECTED_PROGRAMS[name])) != signatures[name]:
            raise ValueError("actual snapshot changed during provenance verification")
        if qasm_sources[name] != serialized[name]:
            raise ValueError("serialized source changed during provenance verification")
    _closed(actual_snapshots, PROGRAM_NAMES, "source inventory")
    _closed(qasm_sources, PROGRAM_NAMES, "serialized source inventory")
    return {"angle_error_sum": error_sum, "realization_bound": realization, "ledger": certificate}
