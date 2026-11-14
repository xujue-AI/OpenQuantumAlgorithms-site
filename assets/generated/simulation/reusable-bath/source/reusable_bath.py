# SPDX-License-Identifier: MIT
"""Bounded primitive source for the original one-spin reusable-bath example.

This module uses only Python's standard library. Counting never constructs a
quadrature, schedule, circuit, numerical reference or physical estimate.
Emission retains the complete XX interaction through its seven primitive
gates. The frozen recipe is historical evidence; job authorization lives in
the separate validation control record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


RECIPE_SHA256 = "3482c732f1b268b3b3f7b9498fcf19b5978e26e1e5387737c88b6c368a8ed5fb"
PUBLIC_PROTOCOL_SHA256 = "ea7a010d4aac3365361ef8741ee68f94271646411a76d752d3e823c0299c97c7"
ACCEPTED_PROTOCOL_HASHES = frozenset((RECIPE_SHA256, PUBLIC_PROTOCOL_SHA256))
METHOD_ID = "original-one-spin-full-xx-reusable-bath-v1"
MAX_METADATA_BYTES = 64 * 1024
MAX_PROTOCOL_BYTES = 128 * 1024
MAX_ROUNDS = 192
MAX_STEPS = 256
MAX_NODES = 17
MAX_SOURCE_INSTRUCTIONS = 600_000
QASM_HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[1];\n'

Operation = tuple[str, tuple[int, ...], float | None, int | None]


class SourceError(ValueError):
    """Invalid source request, with a stable machine-readable reason."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class SourceLimitError(SourceError):
    """A request exceeds a predeclared source or metadata limit."""


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SourceError(f"invalid_{name}")
    return value


def _mode(value: Any) -> str:
    if type(value) is not str or value not in ("ground", "thermal"):
        raise SourceError("invalid_mode")
    return value


def _encoded_size(value: Any) -> int:
    try:
        return len(json.dumps(value, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))
    except (ValueError, OverflowError) as exc:
        # Do not relax Python's process-wide integer encoding limits.
        raise SourceLimitError("integer_encoding_limit") from exc


def count_source(K: int, L: int, M: int, mode: str, P: int = 1) -> dict[str, Any]:
    """Return exact scalar workload bounds, independently of materialization.

    The default input is system |0>. Thermal totals are bounds until an
    emitted, recorded bath-bit schedule determines the actual X count.
    Arbitrary-precision integers are supported within the encoded input and
    report subcap. No K/L/M-indexed object is created here.
    """
    K = _positive_integer(K, "K")
    L = _positive_integer(L, "L")
    M = _positive_integer(M, "M")
    P = _positive_integer(P, "P")
    mode = _mode(mode)
    input_bytes = _encoded_size({"K": K, "L": L, "M": M, "P": P, "mode": mode})
    if input_bytes > MAX_METADATA_BYTES:
        raise SourceLimitError("metadata_over_cap")
    blocks = K * L
    ground = 11 * blocks + K + 3
    upper = ground + (K if mode == "thermal" else 0)
    single = {
        "rxx_blocks": blocks,
        "rz": 5 * blocks,
        "h": 4 * blocks,
        "cx": 2 * blocks,
        "reset": K + 2,
        "measure": 1,
        "bath_x_min": 0,
        "bath_x_max": K if mode == "thermal" else 0,
        "instructions_min": ground,
        "instructions_max": upper,
        "frequency_draws": K,
        "thermal_bit_draws": K if mode == "thermal" else 0,
        "coupling_draws": 0,
    }
    report = {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "status": "counted_only_not_materialized",
        "K": K, "L": L, "M": M, "P": P, "mode": mode,
        "system_qubits": 1, "bath_qubits": 1, "output_bits": 1,
        "per_preparation": single,
        "population": {key: value * P for key, value in single.items()},
        "rxx_blocks_are_informational": True,
        "M_effect": "coefficient/classical-law storage and reference work",
        "scaling_scope": "one system spin; large-n generator remains planned",
        "accuracy_contract_inherited": False,
        "all_components_verified": False,
        "physical_costs_computed": False,
        "reference_or_native_dispatches": 0,
    }
    if input_bytes + _encoded_size(report) > MAX_METADATA_BYTES:
        raise SourceLimitError("metadata_over_cap")
    return report


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


class Protocol(Mapping[str, Any]):
    """Immutable mapping bound to the recipe or its exact public projection."""

    __slots__ = ("_raw", "_data", "_sha256")

    def __init__(self, raw: bytes):
        if type(raw) is not bytes or len(raw) > MAX_PROTOCOL_BYTES:
            raise SourceError("invalid_protocol_bytes")
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in ACCEPTED_PROTOCOL_HASHES:
            raise SourceError("protocol_hash_mismatch")
        parsed = json.loads(raw)
        if parsed["schema_version"] != 2 or parsed["rounds"] != MAX_ROUNDS:
            raise SourceError("invalid_frozen_protocol")
        if digest == PUBLIC_PROTOCOL_SHA256 and parsed["source_recipe_sha256"] != RECIPE_SHA256:
            raise SourceError("invalid_public_protocol_origin")
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_data", _freeze(parsed))
        object.__setattr__(self, "_sha256", digest)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("Protocol is immutable")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def raw(self) -> bytes:
        return self._raw

    @property
    def sha256(self) -> str:
        return self._sha256


def _require_protocol(protocol: Protocol) -> Protocol:
    if type(protocol) is not Protocol:
        raise SourceError("unsealed_protocol")
    if (type(protocol.raw) is not bytes or len(protocol.raw) > MAX_PROTOCOL_BYTES
            or type(protocol.sha256) is not str
            or protocol.sha256 not in ACCEPTED_PROTOCOL_HASHES
            or hashlib.sha256(protocol.raw).hexdigest() != protocol.sha256):
        raise SourceError("protocol_hash_mismatch")
    # Object-level bypasses must not replace the immutable derived table.
    if type(protocol._data) is not MappingProxyType or protocol._data != _freeze(json.loads(protocol.raw)):
        raise SourceError("protocol_mapping_mismatch")
    # Use a fresh canonical mapping, even if hostile caller-owned equality
    # methods were introduced through object.__setattr__ on this instance.
    return Protocol(protocol.raw)


def _regular_bytes(path: Path, maximum: int) -> bytes:
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise SourceError("symlink_protocol_path")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise SourceError("invalid_protocol_file")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    current = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (len(raw) > maximum or len(raw) != before.st_size
            or identity(before) != identity(after) or identity(after) != identity(current)):
        raise SourceError("protocol_changed_during_read")
    return raw


def load_protocol(path: str | os.PathLike[str] | None = None) -> Protocol:
    """Load the bounded frozen recipe, including in a source download bundle."""
    if path is None:
        sibling = Path(__file__).with_name("reusable_bath_protocol.json")
        parents = Path(__file__).absolute().parents
        candidate = (parents[3] / "docs/coordination/simulation-reusable-bath-recipe.json"
                     if len(parents) > 3 else None)
        path = candidate if candidate is not None and candidate.exists() else sibling
    return Protocol(_regular_bytes(Path(path), MAX_PROTOCOL_BYTES))


@dataclass(frozen=True, slots=True)
class Node:
    index: int
    omega: float
    weight: float
    p1: float


def _legendre(n: int, x: float) -> tuple[float, float]:
    previous, current = 1.0, x
    if n == 1:
        return current, previous
    for degree in range(2, n + 1):
        previous, current = current, ((2 * degree - 1) * x * current
                                     - (degree - 1) * previous) / degree
    return current, previous


def _generic_gl(n: int) -> tuple[tuple[float, float], ...]:
    if n == 1:
        return ((1.0, 1.0),)
    if n == 3:
        offset = math.sqrt(3.0 / 5.0)
        return ((1.0 - offset, 5.0 / 18.0), (1.0, 4.0 / 9.0),
                (1.0 + offset, 5.0 / 18.0))
    pairs: list[tuple[float, float]] = []
    for index in range(1, (n + 1) // 2 + 1):
        if n % 2 and index == (n + 1) // 2:
            root = 0.0
        else:
            root = math.cos(math.pi * (index - 0.25) / (n + 0.5))
            converged = False
            for _ in range(32):
                pn, prior = _legendre(n, root)
                derivative = n * (root * pn - prior) / (root * root - 1.0)
                updated = root - pn / derivative
                if not math.isfinite(updated) or abs(updated) >= 1.0:
                    raise SourceError("quadrature_root_failure")
                if abs(updated - root) <= 4.0 * math.ulp(1.0):
                    root, converged = updated, True
                    break
                root = updated
            if not converged:
                raise SourceError("quadrature_root_iteration_cap")
        pn, prior = _legendre(n, root)
        derivative = n * (root * pn - prior) / (root * root - 1.0)
        if abs(pn) > 5e-13 or not math.isfinite(derivative) or derivative == 0:
            raise SourceError("quadrature_root_residual")
        probability = 1.0 / ((1.0 - root * root) * derivative * derivative)
        pairs.append((1.0 - root, probability))
        if root != 0.0:
            pairs.append((1.0 + root, probability))
    pairs.sort()
    total = math.fsum(weight for _, weight in pairs)
    return tuple((omega, weight / total) for omega, weight in pairs)


def make_quadrature(M: int, protocol: Protocol) -> tuple[Node, ...]:
    """Construct at most 17 nodes; GL17 always uses the frozen hex table."""
    M = _positive_integer(M, "M")
    if M > MAX_NODES:
        raise SourceLimitError("quadrature_over_cap")
    protocol = _require_protocol(protocol)
    if M == 17:
        nodes = tuple(Node(item["index"], float.fromhex(item["omega_binary64_hex"]),
                           float.fromhex(item["weight_binary64_hex"]),
                           float.fromhex(item["bath_excited_probability_beta1_binary64_hex"]))
                      for item in protocol["frequency_quadrature"]["nodes"])
    else:
        nodes = tuple(Node(index, omega, weight, 1.0 / (1.0 + math.exp(omega)))
                      for index, (omega, weight) in enumerate(_generic_gl(M)))
    if (len(nodes) != M or any(node.index != index for index, node in enumerate(nodes))
            or any(not (math.isfinite(node.omega) and 0.0 < node.omega < 2.0
                        and math.isfinite(node.weight) and node.weight > 0.0
                        and math.isfinite(node.p1) and 0.0 < node.p1 < 1.0)
                   for node in nodes)
            or any(a.omega >= b.omega for a, b in zip(nodes, nodes[1:]))
            or abs(math.fsum(node.weight for node in nodes) - 1.0) > 2e-14):
        raise SourceError("invalid_quadrature")
    return nodes


def _steps(value: Any) -> int:
    value = _positive_integer(value, "L")
    if value > MAX_STEPS:
        raise SourceLimitError("steps_over_cap")
    return value


def _omega(value: Any) -> float:
    if type(value) not in (int, float):
        raise SourceError("invalid_frequency")
    try:
        if not math.isfinite(value) or not 0.0 < value < 2.0:
            raise SourceError("invalid_frequency")
        return float(value)
    except OverflowError as exc:
        raise SourceError("invalid_frequency") from exc


def _angles(steps: int, protocol: Protocol | None) -> tuple[float, ...]:
    if steps == 256 and protocol is not None:
        return tuple(float.fromhex(item["rxx_angle_binary64_hex"])
                     for item in protocol["time_discretization"]["interaction_angles"])
    return tuple((64.0 / (25.0 * steps))
                 * math.exp(-8.0 * (2 * j + 1 - steps) ** 2 / (steps * steps))
                 for j in range(steps))


def _round_operations(omega: float, steps: int, angles: tuple[float, ...]) -> Iterator[Operation]:
    system_half = 8.0 / steps
    bath_half = -8.0 * omega / steps
    for angle in angles:
        yield ("rz", (0,), system_half, None)
        yield ("rz", (1,), bath_half, None)
        yield ("h", (0,), None, None)
        yield ("h", (1,), None, None)
        yield ("cx", (0, 1), None, None)
        yield ("rz", (1,), angle, None)
        yield ("cx", (0, 1), None, None)
        yield ("h", (0,), None, None)
        yield ("h", (1,), None, None)
        yield ("rz", (0,), system_half, None)
        yield ("rz", (1,), bath_half, None)


def _charge(budget: Any, counter: str, operations: Iterator[Operation]) -> Iterator[Operation]:
    for operation in operations:
        if budget is not None:
            budget.consume(counter, 1)
        yield operation


def iter_round_operations(omega: float, steps: int, *, protocol: Protocol | None = None,
                          budget: Any = None,
                          counter: str = "compact_source_instructions") -> Iterator[Operation]:
    """Evolution-only primitive iterator; validation occurs before return."""
    steps, omega = _steps(steps), _omega(omega)
    if protocol is not None:
        protocol = _require_protocol(protocol)
    return _charge(budget, counter, _round_operations(omega, steps, _angles(steps, protocol)))


_INITIAL_PREPARATION: dict[str, tuple[Operation, ...]] = {
    "0": (),
    "1": (("x", (0,), None, None),),
    "+": (("h", (0,), None, None),),
    "+i": (("h", (0,), None, None), ("rz", (0,), math.pi / 2.0, None)),
}


def _prepare(protocol: Protocol, frequency_indices: Any, bath_bits: Any, *,
             steps: int | None, nodes: int, mode: str, initial: str) -> tuple[Any, ...]:
    mode = _mode(mode)
    if type(initial) is not str or initial not in _INITIAL_PREPARATION:
        raise SourceError("invalid_initial_state")
    steps = _steps(MAX_STEPS if steps is None else steps)
    nodes = _positive_integer(nodes, "M")
    if nodes > MAX_NODES:
        raise SourceLimitError("quadrature_over_cap")
    if type(frequency_indices) not in (list, tuple):
        raise SourceError("unbounded_schedule_type")
    rounds = len(frequency_indices)
    if not 1 <= rounds <= MAX_ROUNDS:
        raise SourceLimitError("rounds_over_cap")
    # Count before copying schedules or allocating quadrature/pulse tables.
    worst = 11 * rounds * steps + rounds + 3 + len(_INITIAL_PREPARATION[initial])
    if mode == "thermal":
        worst += rounds
    if worst > MAX_SOURCE_INSTRUCTIONS:
        raise SourceLimitError("source_instructions_over_cap")
    if bath_bits is None and mode == "ground":
        bath_bits = (0,) * rounds
    if type(bath_bits) not in (list, tuple) or len(bath_bits) != rounds:
        raise SourceError("invalid_bath_schedule_length")
    # Bounded slices also limit allocation if another thread grows a caller's
    # builtin list after the length preflight. The extra item detects growth.
    frequencies = tuple(frequency_indices[:rounds + 1])
    bits = tuple(bath_bits[:rounds + 1])
    if len(frequencies) != rounds or len(bits) != rounds or not 1 <= len(frequencies) <= MAX_ROUNDS:
        raise SourceError("schedule_changed_during_snapshot")
    if any(type(index) is not int or not 0 <= index < nodes for index in frequencies):
        raise SourceError("invalid_frequency_index")
    if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise SourceError("invalid_bath_bit")
    if mode == "ground" and any(bits):
        raise SourceError("ground_bath_bit_nonzero")
    protocol = _require_protocol(protocol)
    quadrature = make_quadrature(nodes, protocol)
    angles = _angles(steps, protocol)
    return protocol, frequencies, bits, steps, quadrature, angles, initial, mode


def _full_operations(prepared: tuple[Any, ...]) -> Iterator[Operation]:
    _, frequencies, bits, steps, quadrature, angles, initial, _ = prepared
    yield ("reset", (0,), None, None)
    yield ("reset", (1,), None, None)
    yield from _INITIAL_PREPARATION[initial]
    for index, bit in zip(frequencies, bits):
        if bit:
            yield ("x", (1,), None, None)
        yield from _round_operations(quadrature[index].omega, steps, angles)
        yield ("reset", (1,), None, None)
    yield ("measure", (0,), None, 0)


def iter_operations(protocol: Protocol, frequency_indices: Any, bath_bits: Any = None, *,
                    steps: int | None = None, nodes: int = 17, mode: str,
                    initial: str = "0", budget: Any = None,
                    counter: str = "full_source_instructions") -> Iterator[Operation]:
    """Bounded complete source; no QuantumCircuit or simulation is constructed."""
    prepared = _prepare(protocol, frequency_indices, bath_bits, steps=steps, nodes=nodes,
                        mode=mode, initial=initial)
    return _charge(budget, counter, _full_operations(prepared))


def operation_qasm(operation: Operation) -> str:
    """Serialize a validated primitive tuple without opaque matrix operations."""
    if type(operation) is not tuple or len(operation) != 4:
        raise SourceError("invalid_operation")
    name, qubits, angle, clbit = operation
    if type(name) is not str:
        raise SourceError("invalid_operation_name")
    if type(qubits) is not tuple or any(type(q) is not int or q not in (0, 1) for q in qubits):
        raise SourceError("invalid_operation_qubits")
    if name == "rz" and len(qubits) == 1 and clbit is None:
        if type(angle) not in (int, float):
            raise SourceError("invalid_operation_angle")
        try:
            if not math.isfinite(angle):
                raise SourceError("invalid_operation_angle")
        except OverflowError as exc:
            raise SourceError("invalid_operation_angle") from exc
        return f"rz({float(angle)!r}) q[{qubits[0]}];\n"
    if name in ("h", "x", "reset") and len(qubits) == 1 and angle is None and clbit is None:
        return f"{name} q[{qubits[0]}];\n"
    if name == "cx" and qubits == (0, 1) and angle is None and clbit is None:
        return "cx q[0],q[1];\n"
    if name == "measure" and qubits == (0,) and angle is None and type(clbit) is int and clbit == 0:
        return "measure q[0] -> c[0];\n"
    raise SourceError("invalid_operation")


def write_qasm(path: str | os.PathLike[str], protocol: Protocol, frequency_indices: Any,
               bath_bits: Any = None, *, steps: int | None = None, nodes: int = 17,
               mode: str, initial: str = "0", budget: Any,
               counter: str = "full_source_instructions") -> dict[str, Any]:
    """Stream one source export through the required managed-byte controller."""
    if budget is None or not callable(getattr(budget, "open_managed", None)):
        raise SourceError("managed_budget_required")
    prepared = _prepare(protocol, frequency_indices, bath_bits, steps=steps, nodes=nodes,
                        mode=mode, initial=initial)
    budget.consume("full_exports", 1)
    counts: dict[str, int] = {name: 0 for name in ("rz", "h", "cx", "x", "reset", "measure")}
    digest = hashlib.sha256()
    byte_count = instruction_count = 0
    buffer = bytearray(QASM_HEADER.encode("ascii"))
    with budget.open_managed(path, mode="wb") as stream:
        for operation in _charge(budget, counter, _full_operations(prepared)):
            encoded = operation_qasm(operation).encode("ascii")
            buffer.extend(encoded)
            counts[operation[0]] += 1
            instruction_count += 1
            if len(buffer) >= 64 * 1024:
                raw = bytes(buffer[:64 * 1024])
                if stream.write(raw) != len(raw):
                    raise SourceError("short_qasm_write")
                digest.update(raw)
                byte_count += len(raw)
                del buffer[:64 * 1024]
        if buffer:
            raw = bytes(buffer)
            if stream.write(raw) != len(raw):
                raise SourceError("short_qasm_write")
            digest.update(raw)
            byte_count += len(raw)
        stream.flush()
    expected = 11 * len(prepared[1]) * prepared[3] + len(prepared[1]) + 3
    expected += sum(prepared[2]) + len(_INITIAL_PREPARATION[initial])
    if instruction_count != expected or instruction_count > MAX_SOURCE_INSTRUCTIONS:
        raise SourceError("emitted_count_mismatch")
    return {
        "status": "emitted_source_not_quantum_simulated",
        "method_id": METHOD_ID,
        "protocol_sha256": prepared[0].sha256,
        "source_recipe_sha256": RECIPE_SHA256,
        "mode": mode, "K": len(prepared[1]), "L": prepared[3], "M": nodes,
        "initial": initial,
        "instructions": instruction_count, "primitive_counts": counts,
        "rxx_blocks": len(prepared[1]) * prepared[3],
        "bath_X_count": sum(prepared[2]),
        "frequency_draws": len(prepared[1]),
        "thermal_bit_draws": len(prepared[1]) if mode == "thermal" else 0,
        "emitter_rng_draws": 0,
        "schedule_origin": "recorded values supplied by caller; draw counts are algorithm requirements",
        "sha256": digest.hexdigest(), "bytes": byte_count,
        "all_components_verified": False, "physical_costs_computed": False,
    }


def _main(argv: list[str] | None = None) -> int:
    """Standalone scalar CLI; does not enter the legacy package initializer."""
    import argparse
    import sys

    arguments = sys.argv[1:] if argv is None else argv
    if sum(len(item.encode("utf-8")) for item in arguments) > MAX_METADATA_BYTES:
        print(json.dumps({"status": "rejected_count_request", "reason": "metadata_over_cap"}))
        return 2
    class CountParser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            # Do not echo arbitrarily large rejected argument strings.
            raise SourceError("invalid_cli_arguments")

    parser = CountParser(description=__doc__)
    parser.add_argument("--count", nargs=3, type=int, required=True, metavar=("K", "L", "M"))
    parser.add_argument("--mode", choices=("ground", "thermal"), required=True)
    parser.add_argument("--preparations", type=int, default=1, metavar="P")
    try:
        options = parser.parse_args(arguments)
        result = count_source(*options.count, options.mode, P=options.preparations)
        encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
        if len(encoded.encode("utf-8")) + sum(len(item.encode("utf-8"))
                                             for item in arguments) > MAX_METADATA_BYTES:
            raise SourceLimitError("metadata_over_cap")
    except SourceError as exc:
        print(json.dumps({"status": "rejected_count_request", "reason": exc.reason}))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
