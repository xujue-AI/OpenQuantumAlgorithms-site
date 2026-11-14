# SPDX-License-Identifier: MIT
"""Bounded lossless JSON-DAG transport for repeated physical-workload ledgers.

This changes representation only. JSON scalar types, IEEE float values and
container order are retained. Equal subtrees share nodes; unpacked containers
can consequently alias and should be treated as read-only. No expanded JSON
serialization is required to validate or compare the reconstructed graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

CODEC = "json-content-dag-v1"
TAGS = frozenset(("null", "bool", "int", "float", "str", "list", "dict"))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


@dataclass(frozen=True)
class CodecLimits:
    max_encoded_bytes: int = 64 * 1024 * 1024
    max_nodes: int = 500_000
    max_references: int = 4_000_000
    max_depth: int = 128
    max_container_items: int = 100_000
    max_scalar_bytes: int = 4 * 1024 * 1024
    max_integer_digits: int = 1024

    def __post_init__(self):
        ceilings = dict(max_encoded_bytes=64 * 1024 * 1024, max_nodes=500_000,
                        max_references=4_000_000, max_depth=128,
                        max_container_items=100_000, max_scalar_bytes=4 * 1024 * 1024,
                        max_integer_digits=1024)
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")


def _limits(value: CodecLimits) -> CodecLimits:
    if not isinstance(value, CodecLimits):
        raise ValueError("limits must be a CodecLimits instance")
    return value


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _encoded(value: Any, maximum: int) -> bytes:
    """Stream only the flat packed envelope, never the expanded JSON value."""
    result = bytearray()
    encoder = json.JSONEncoder(ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    try:
        for fragment in encoder.iterencode(value):
            raw = fragment.encode("ascii")
            if len(result) + len(raw) > maximum:
                raise ValueError("Packed JSON exceeds its frozen encoded-byte cap")
            result.extend(raw)
    except (TypeError, OverflowError, RecursionError) as error:
        raise ValueError("Invalid packed JSON transport") from error
    return bytes(result)


def _scalar(tag: str, payload: Any, limits: CodecLimits) -> Any:
    if tag == "null":
        if payload is not None:
            raise ValueError("Null payload must be null")
        return None
    if tag == "bool":
        if type(payload) is not bool:
            raise ValueError("Boolean payload must retain its type")
        return payload
    if tag == "int":
        if (type(payload) is not str or len(payload.lstrip("-")) > limits.max_integer_digits
                or not _INTEGER.fullmatch(payload)):
            raise ValueError("Integer payload is not a bounded canonical decimal")
        number = int(payload)
        if str(number) != payload:
            raise ValueError("Integer payload is not canonical")
        return number
    if tag == "float":
        if type(payload) is not str or len(payload) > 64:
            raise ValueError("Float payload must be a bounded IEEE hexadecimal string")
        try:
            number = float.fromhex(payload)
        except (ValueError, OverflowError) as error:
            raise ValueError("Invalid hexadecimal float") from error
        if not math.isfinite(number) or number.hex() != payload:
            raise ValueError("Float payload must be finite and canonical")
        return number
    if tag == "str":
        if (type(payload) is not str or len(payload) > limits.max_scalar_bytes
                or len(_json(payload)) > limits.max_scalar_bytes):
            raise ValueError("String payload exceeds the bounded scalar domain")
        return payload
    raise ValueError("Unknown scalar tag")


def _material(tag: str, payload: Any, hashes: list[str]) -> bytes:
    if tag == "list":
        content = [hashes[index] for index in payload]
    elif tag == "dict":
        content = [[hashes[key], hashes[value]] for key, value in payload]
    else:
        content = payload
    return _json([tag, content])


def pack(value: Any, *, limits: CodecLimits = CodecLimits()) -> dict:
    """Hash-cons an ordered JSON graph, visiting shared objects only once."""
    limits = _limits(limits)
    nodes, hashes, heights = [], [], []
    by_hash: dict[str, int] = {}
    memo: dict[int, tuple[Any, int]] = {}
    active: set[int] = set()
    visited_references = 0

    def intern(tag, payload, height):
        material = _material(tag, payload, hashes)
        digest = hashlib.sha256(material).hexdigest()
        if digest in by_hash:
            previous = by_hash[digest]
            old = nodes[previous]
            if _material(old[1], old[2], hashes) != material:
                raise ValueError("Content-address collision between unequal canonical nodes")
            return previous
        if len(nodes) >= limits.max_nodes:
            raise ValueError("JSON-DAG exceeds its node cap")
        index = len(nodes)
        nodes.append([digest, tag, payload])
        hashes.append(digest)
        heights.append(height)
        by_hash[digest] = index
        return index

    def visit(current, depth):
        nonlocal visited_references
        if depth > limits.max_depth:
            raise ValueError("JSON graph exceeds its depth cap")
        kind = type(current)
        if current is None:
            return intern("null", None, 0)
        if kind is bool:
            return intern("bool", current, 0)
        if kind is int:
            payload = str(current)
            _scalar("int", payload, limits)
            return intern("int", payload, 0)
        if kind is float:
            payload = current.hex()
            _scalar("float", payload, limits)
            return intern("float", payload, 0)
        if kind is str:
            _scalar("str", current, limits)
            return intern("str", current, 0)
        if kind not in (list, dict):
            raise ValueError("Only JSON null/bool/int/float/string/list/dict values are supported")
        identity = id(current)
        if identity in active:
            raise ValueError("Cyclic input is not a JSON value")
        if identity in memo:
            original, index = memo[identity]
            if original is not current:
                raise ValueError("Unexpected object-identity reuse")
            if depth + heights[index] > limits.max_depth:
                raise ValueError("Shared JSON subtree exceeds the depth cap at its new location")
            return index
        if len(current) > limits.max_container_items:
            raise ValueError("JSON container exceeds its item cap")
        visited_references += len(current) * (2 if kind is dict else 1)
        if visited_references > limits.max_references:
            raise ValueError("JSON graph traversal exceeds its reference cap")
        active.add(identity)
        try:
            if kind is list:
                tag = "list"
                payload = [visit(child, depth + 1) for child in current]
                children = payload
            else:
                tag, payload = "dict", []
                for key, child in current.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    payload.append([visit(key, depth + 1), visit(child, depth + 1)])
                children = [index for pair in payload for index in pair]
            height = 1 + max((heights[index] for index in children), default=-1)
            if depth + height > limits.max_depth:
                raise ValueError("JSON graph exceeds its depth cap")
            index = intern(tag, payload, height)
            memo[identity] = (current, index)
            return index
        finally:
            active.remove(identity)

    root = visit(value, 0)
    packed = dict(codec=CODEC, schema_version=1, root=root,
                  logical_sha256=hashes[root], nodes=nodes)
    _encoded(packed, limits.max_encoded_bytes)
    return packed


def _validated(packed: Any, limits: CodecLimits) -> tuple[Any, dict]:
    if (type(packed) is not dict
            or set(packed) != {"codec", "schema_version", "root", "logical_sha256", "nodes"}
            or packed["codec"] != CODEC or type(packed["schema_version"]) is not int
            or packed["schema_version"] != 1):
        raise ValueError("Unsupported JSON-DAG envelope")
    nodes = packed["nodes"]
    if type(nodes) is not list or not 1 <= len(nodes) <= limits.max_nodes:
        raise ValueError("JSON-DAG node table is empty or exceeds its cap")
    root = packed["root"]
    if type(root) is not int or not 0 <= root < len(nodes):
        raise ValueError("Invalid JSON-DAG root reference")
    if type(packed["logical_sha256"]) is not str or not _HEX.fullmatch(packed["logical_sha256"]):
        raise ValueError("Invalid logical content digest")
    # Validate flat payload shapes/scalar bounds before encoding or allocating
    # reconstructed containers. Direct unpack callers receive the same byte
    # guard as byte-oriented loads callers.
    preflight_references = 0
    for index, row in enumerate(nodes):
        if (type(row) is not list or len(row) != 3 or type(row[0]) is not str
                or not _HEX.fullmatch(row[0]) or type(row[1]) is not str or row[1] not in TAGS):
            raise ValueError("Invalid JSON-DAG node")
        _, tag, payload = row
        if tag in ("list", "dict"):
            if type(payload) is not list or len(payload) > limits.max_container_items:
                raise ValueError("JSON-DAG container exceeds its item cap")
            if tag == "dict":
                if any(type(pair) is not list or len(pair) != 2 for pair in payload):
                    raise ValueError("Object payload must contain ordered reference pairs")
                children = [child for pair in payload for child in pair]
            else:
                children = payload
            if any(type(child) is not int or not 0 <= child < index for child in children):
                raise ValueError("References must be preceding nodes; dangling/forward/cyclic references are rejected")
            preflight_references += len(children)
            if preflight_references > limits.max_references:
                raise ValueError("JSON-DAG exceeds its reference cap")
        else:
            _scalar(tag, payload, limits)
    encoded_bytes = len(_encoded(packed, limits.max_encoded_bytes))
    objects, hashes, heights, references = [], [], [], []
    seen_hash: dict[str, int] = {}
    total_references = 0
    for index, row in enumerate(nodes):
        if (type(row) is not list or len(row) != 3 or type(row[0]) is not str
                or not _HEX.fullmatch(row[0]) or type(row[1]) is not str or row[1] not in TAGS):
            raise ValueError("Invalid JSON-DAG node")
        digest, tag, payload = row
        children = []
        if tag in ("list", "dict"):
            if type(payload) is not list or len(payload) > limits.max_container_items:
                raise ValueError("JSON-DAG container exceeds its item cap")
            if tag == "list":
                children = payload
            else:
                if any(type(pair) is not list or len(pair) != 2 for pair in payload):
                    raise ValueError("Object payload must contain ordered key/value reference pairs")
                children = [reference for pair in payload for reference in pair]
            if any(type(reference) is not int or not 0 <= reference < index for reference in children):
                raise ValueError("References must be preceding nodes; dangling/forward/cyclic references are rejected")
            total_references += len(children)
            if total_references > limits.max_references:
                raise ValueError("JSON-DAG exceeds its reference cap")
            height = 1 + max((heights[child] for child in children), default=-1)
            if height > limits.max_depth:
                raise ValueError("JSON-DAG exceeds its depth cap")
            if tag == "list":
                result = [objects[child] for child in payload]
            else:
                result = {}
                for key, child in payload:
                    if nodes[key][1] != "str" or objects[key] in result:
                        raise ValueError("Object keys must be unique decoded strings")
                    result[objects[key]] = objects[child]
        else:
            result = _scalar(tag, payload, limits)
            height = 0
        material = _material(tag, payload, hashes)
        if hashlib.sha256(material).hexdigest() != digest:
            raise ValueError("JSON-DAG content hash mismatch")
        if digest in seen_hash:
            previous = nodes[seen_hash[digest]]
            if _material(previous[1], previous[2], hashes) != material:
                raise ValueError("Content-address collision between unequal nodes")
            raise ValueError("Duplicate canonical content node is not permitted")
        seen_hash[digest] = index
        objects.append(result)
        hashes.append(digest)
        heights.append(height)
        references.append(children)
    if hashes[root] != packed["logical_sha256"]:
        raise ValueError("Root digest differs from the logical content digest")
    reached, todo = set(), [root]
    while todo:
        index = todo.pop()
        if index not in reached:
            reached.add(index)
            todo.extend(references[index])
    if len(reached) != len(nodes):
        raise ValueError("JSON-DAG contains unreachable nodes")
    return objects[root], dict(nodes=len(nodes), references=total_references,
                              maximum_depth=heights[root], encoded_bytes=encoded_bytes,
                              logical_sha256=hashes[root])


def unpack(packed: Any, *, limits: CodecLimits = CodecLimits()) -> Any:
    """Validate every node/hash/reference and reconstruct shared JSON subtrees."""
    return _validated(packed, _limits(limits))[0]


def dumps(packed: Any, *, limits: CodecLimits = CodecLimits()) -> bytes:
    """Serialize validated packed transport compactly, within the 64 MiB cap."""
    limits = _limits(limits)
    _validated(packed, limits)
    return _encoded(packed, limits.max_encoded_bytes)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON transport key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant is unsupported: {value}")


def loads(raw: bytes, *, limits: CodecLimits = CodecLimits()) -> Any:
    """Decode bounded transport and validate its content-addressed graph."""
    limits = _limits(limits)
    if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
        raise ValueError("JSON-DAG transport must be bytes within its encoded cap")
    try:
        packed = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid JSON-DAG transport") from error
    return unpack(packed, limits=limits)


def statistics(packed: Any, *, limits: CodecLimits = CodecLimits()) -> dict:
    """Bounded representation statistics; never expand repeated JSON trees."""
    return _validated(packed, _limits(limits))[1]


def equivalent(left: Any, right: Any, *, limits: CodecLimits = CodecLimits()) -> bool:
    """Compare exact ordered JSON values while memoizing shared container pairs."""
    limits = _limits(limits)
    todo, seen, active, references = [(left, right, 0, False)], set(), set(), 0
    heights, pending_children, checked_keys = {}, {}, set()
    while todo:
        a, b, depth, exiting = todo.pop()
        pair = (id(a), id(b))
        if exiting:
            children = pending_children.pop(pair)
            height = 1 + max((heights[(id(x), id(y))] if type(x) in (list, dict) else 0
                              for x, y in children), default=-1)
            if depth + height > limits.max_depth:
                raise ValueError("Shared comparison subtree exceeds its depth cap")
            heights[pair] = height
            active.remove(pair)
            seen.add(pair)
            continue
        if depth > limits.max_depth:
            raise ValueError("Comparison exceeds its depth cap")
        if type(a) is not type(b):
            return False
        kind = type(a)
        if a is None:
            continue
        if kind is float:
            _scalar("float", a.hex(), limits)
            _scalar("float", b.hex(), limits)
            if a.hex() != b.hex():
                return False
        elif kind in (bool, int, str):
            tag = {bool: "bool", int: "int", str: "str"}[kind]
            _scalar(tag, str(a) if kind is int else a, limits)
            _scalar(tag, str(b) if kind is int else b, limits)
            if a != b:
                return False
        elif kind in (list, dict):
            if pair in active:
                raise ValueError("Cyclic values are not JSON comparison inputs")
            if pair in seen:
                if depth + heights[pair] > limits.max_depth:
                    raise ValueError("Shared comparison subtree exceeds its depth cap at its new location")
                continue
            if len(seen) + len(active) >= limits.max_nodes or max(len(a), len(b)) > limits.max_container_items:
                raise ValueError("Comparison exceeds its container/node cap")
            if len(a) != len(b):
                return False
            if kind is list:
                children = list(zip(a, b))
            else:
                if any(type(key) is not str for key in (*a, *b)):
                    raise ValueError("JSON comparison keys must be strings")
                if list(a) != list(b):
                    return False
                for key in a:
                    if key not in checked_keys:
                        _scalar("str", key, limits)
                        checked_keys.add(key)
                children = [(a[key], b[key]) for key in a]
            references += len(children) * (2 if kind is dict else 1)
            if references > limits.max_references:
                raise ValueError("Comparison exceeds its reference cap")
            active.add(pair)
            pending_children[pair] = children
            todo.append((a, b, depth, True))
            todo.extend((x, y, depth + 1, False) for x, y in children)
        else:
            raise ValueError("Comparison supports only JSON values")
    return True
