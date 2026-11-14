"""Bounded, stdlib-only parser for the authenticated saved OpenQASM subset.

This is deliberately not a general OpenQASM implementation. It accepts explicit
indexed registers, fixed gate definitions, arithmetic angles, reset, measurement,
and barriers. Custom gate bodies are expanded exactly; their names never imply
semantics. Parsing does not import Qiskit or execute quantum operations.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re


class QasmError(ValueError):
    """The saved text is outside the bounded replay grammar."""


# (parameter count, wire count); these are the only numerical replay leaves.
STANDARD_GATES = {
    "id": (0, 1), "h": (0, 1), "x": (0, 1), "y": (0, 1),
    "z": (0, 1), "s": (0, 1), "sdg": (0, 1), "t": (0, 1),
    "tdg": (0, 1), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1),
    "p": (1, 1), "u": (3, 1), "U": (3, 1), "cx": (0, 2),
    "cy": (0, 2), "cz": (0, 2), "ch": (0, 2), "swap": (0, 2),
    "crx": (1, 2), "cry": (1, 2), "crz": (1, 2), "cp": (1, 2),
    "cu": (4, 2), "ccx": (0, 3), "cswap": (0, 3),
}
CLIFFORD_T_GATES = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx"})
_TOKEN = re.compile(
    r"\s+|//[^\n]*(?:\n|$)|/\*[\s\S]*?\*/|"
    r'"[^"\n\\]*"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|'
    r"[A-Za-z_][A-Za-z_0-9]*|[{}()\[\],;=+*/-]"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_INTEGER = re.compile(r"\d+\Z")
_NUMBER = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?\Z")
_RESERVED = frozenset({
    "OPENQASM", "include", "gate", "def", "extern", "qubit", "bit", "if", "else",
    "for", "while", "switch", "case", "default", "break", "continue", "return",
    "measure", "reset", "barrier", "ctrl", "negctrl", "inv", "pow", "gphase",
    "let", "const", "input", "output", "int", "uint", "float", "angle", "bool",
    "complex", "array", "void", "delay", "box", "duration", "stretch", "readonly",
    "mutable", "pragma", "end", "defcal", "cal", "defcalgrammar", "pi",
})


@dataclass(frozen=True)
class Operation:
    name: str
    parameters: tuple[float, ...]
    qubits: tuple[int, ...]
    clbits: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Call:
    name: str
    parameters: tuple[tuple[str, ...], ...]
    qubits: tuple[str | int, ...]
    clbits: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Definition:
    parameters: tuple[str, ...]
    qubits: tuple[str, ...]
    body: tuple[_Call, ...]


@dataclass(frozen=True)
class ParsedCircuit:
    num_qubits: int
    num_clbits: int
    operations: tuple[Operation, ...]
    top_level_operations: tuple[Operation, ...]
    definition_count: int

    def counts(self, *, expanded: bool = True) -> dict[str, int]:
        operations = self.operations if expanded else self.top_level_operations
        return dict(sorted(Counter(item.name for item in operations).items()))

    def events(self) -> tuple[Operation, ...]:
        return tuple(item for item in self.operations if item.name in {"reset", "measure"})


def _bounded_integer(value: int, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise QasmError(f"{name} must be an integer in 1..{maximum}")
    return value


def _tokens(text: str, maximum: int) -> list[str]:
    result = []
    offset = 0
    while offset < len(text):
        match = _TOKEN.match(text, offset)
        if match is None:
            raise QasmError(f"Unsupported QASM character at offset {offset}")
        token = match.group()
        offset = match.end()
        if token.isspace() or token.startswith(("//", "/*")):
            continue
        result.append(token)
        if len(result) > maximum:
            raise QasmError("QASM token cap exceeded")
    return result


def _expression(tokens: tuple[str, ...], bindings: dict[str, float], *, depth_cap: int = 32,
                syntax_only: bool = False) -> float:
    """Evaluate a tiny arithmetic grammar; never eval or import expressions."""
    index = 0

    def take(depth: int, minimum: int = 0) -> tuple[float, bool]:
        nonlocal index
        if depth > depth_cap or index >= len(tokens):
            raise QasmError("Invalid or excessively nested angle expression")
        token = tokens[index]
        index += 1
        symbolic = False
        if token in {"+", "-"}:
            value, symbolic = take(depth + 1, 3)
            if token == "-":
                value = -value
        elif token == "(":
            value, symbolic = take(depth + 1)
            if index >= len(tokens) or tokens[index] != ")":
                raise QasmError("Unclosed angle expression")
            index += 1
        elif token == "pi":
            value = math.pi
        elif _NUMBER.fullmatch(token):
            value = float(token)
        elif token in bindings:
            value = bindings[token]
            symbolic = syntax_only
        else:
            raise QasmError("Unresolved or unsupported angle expression")
        if not math.isfinite(value):
            raise QasmError("Nonfinite angle expression")
        while index < len(tokens):
            operator = tokens[index]
            priority = {"+": 1, "-": 1, "*": 2, "/": 2}.get(operator, -1)
            if priority < minimum:
                break
            index += 1
            right, right_symbolic = take(depth + 1, priority + 1)
            if operator == "/" and not right_symbolic and right == 0:
                raise QasmError("Division by zero in angle expression")
            symbolic = symbolic or right_symbolic
            if syntax_only and symbolic:
                value = 1.0
            elif operator == "+":
                value += right
            elif operator == "-":
                value -= right
            elif operator == "*":
                value *= right
            elif right == 0:
                raise QasmError("Division by zero in angle expression")
            else:
                value /= right
            if not math.isfinite(value):
                raise QasmError("Nonfinite angle expression")
        if not math.isfinite(value):
            raise QasmError("Nonfinite angle expression")
        return value, symbolic

    value, _ = take(0)
    if index != len(tokens):
        raise QasmError("Unsupported angle expression suffix")
    return value


class _Parser:
    def __init__(self, tokens: list[str], max_qubits: int, max_operations: int,
                 max_definitions: int, max_definition_depth: int):
        self.tokens, self.index = tokens, 0
        self.max_qubits, self.max_operations = max_qubits, max_operations
        self.max_definitions, self.max_definition_depth = max_definitions, max_definition_depth
        self.definitions: dict[str, _Definition] = {}
        self.registers: dict[str, tuple[str, int]] = {}
        self.calls: list[_Call] = []
        self.body_operations = 0

    def peek(self) -> str:
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def pop(self, expected: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise QasmError("Unexpected end of QASM")
        value = self.tokens[self.index]
        self.index += 1
        if expected is not None and value != expected:
            raise QasmError(f"Expected {expected!r}")
        return value

    def identifier(self, *, allow_keywords: bool = False) -> str:
        name = self.pop()
        if not _IDENTIFIER.fullmatch(name) or (not allow_keywords and name in _RESERVED):
            raise QasmError("Expected identifier")
        return name

    def integer(self) -> int:
        token = self.pop()
        if not _INTEGER.fullmatch(token) or len(token) > 8:
            raise QasmError("Expected bounded unsigned integer")
        return int(token)

    def parameter_list(self) -> tuple[tuple[str, ...], ...]:
        if self.peek() != "(":
            return ()
        self.pop("(")
        if self.peek() == ")":
            self.pop(")")
            return ()
        result = []
        current: list[str] = []
        depth = 0
        while True:
            token = self.pop()
            if token == "(" :
                depth += 1
            elif token == ")":
                if depth == 0:
                    if not current:
                        raise QasmError("Empty angle expression")
                    result.append(tuple(current))
                    break
                depth -= 1
            elif token == "," and depth == 0:
                if not current:
                    raise QasmError("Empty angle expression")
                result.append(tuple(current))
                current = []
                continue
            if len(current) >= 128 or depth > 32 or len(result) >= 16:
                raise QasmError("Angle expression/list cap exceeded")
            current.append(token)
        return tuple(result)

    def wire(self, locals_: tuple[str, ...] | None, kind: str = "qubit") -> str | int:
        name = self.identifier()
        if locals_ is not None:
            if name not in locals_:
                raise QasmError("Unresolved gate wire")
            return name
        if name not in self.registers or self.registers[name][0] != kind:
            raise QasmError("Unresolved or wrong-kind register")
        self.pop("[")
        index = self.integer()
        self.pop("]")
        if index >= self.registers[name][1]:
            raise QasmError("Register index out of range")
        return index

    def call(self, locals_: tuple[str, ...] | None = None) -> _Call:
        # Only explicit fixed measurement assignments are legal.
        if locals_ is None and self.peek() in self.registers:
            bit = self.wire(None, "bit")
            self.pop("=")
            self.pop("measure")
            wire = self.wire(None)
            self.pop(";")
            return _Call("measure", (), (wire,), (bit,))
        name = self.identifier(allow_keywords=True)
        if name in _RESERVED and name not in {"measure", "reset", "barrier"}:
            raise QasmError("Unsupported statement keyword")
        if name == "measure" or (locals_ is not None and name in {"reset", "barrier"}):
            raise QasmError("Hidden instruments are outside fixed unitary definitions")
        parameters = self.parameter_list()
        wires = [self.wire(locals_)]
        while self.peek() == ",":
            self.pop(",")
            wires.append(self.wire(locals_))
            if len(wires) > self.max_qubits:
                raise QasmError("Wire-list cap exceeded")
        self.pop(";")
        if len(set(wires)) != len(wires):
            raise QasmError("Repeated quantum wire in operation")
        if name == "reset" and (parameters or len(wires) != 1):
            raise QasmError("Reset requires one explicit wire")
        if name == "barrier" and parameters:
            raise QasmError("Barrier cannot have angles")
        return _Call(name, parameters, tuple(wires))

    def definition(self) -> None:
        self.pop("gate")
        name = self.identifier()
        if name in self.definitions or name in STANDARD_GATES or name in {"reset", "measure", "barrier"}:
            raise QasmError("Duplicate or reserved gate definition")
        raw_parameters = self.parameter_list()
        if any(len(item) != 1 or not _IDENTIFIER.fullmatch(item[0]) or item[0] in _RESERVED for item in raw_parameters):
            raise QasmError("Gate parameters must be unique symbolic names")
        parameters = tuple(item[0] for item in raw_parameters)
        wires = [self.identifier()]
        while self.peek() == ",":
            self.pop(",")
            wires.append(self.identifier())
            if len(wires) > self.max_qubits:
                raise QasmError("Gate wire cap exceeded")
        if len(set(parameters + tuple(wires))) != len(parameters) + len(wires):
            raise QasmError("Duplicate gate parameter or wire")
        self.pop("{")
        body = []
        while self.peek() != "}":
            body.append(self.call(tuple(wires)))
            self.body_operations += 1
            if self.body_operations > self.max_operations:
                raise QasmError("Definition-body operation cap exceeded")
        self.pop("}")
        if not body:
            raise QasmError("Empty gate definitions are outside the saved grammar")
        self.definitions[name] = _Definition(parameters, tuple(wires), tuple(body))
        if len(self.definitions) > self.max_definitions:
            raise QasmError("Gate-definition cap exceeded")

    def arity(self, call: _Call) -> None:
        if call.name in {"reset", "measure", "barrier"}:
            return
        if call.name in STANDARD_GATES:
            shape = STANDARD_GATES[call.name]
        elif call.name in self.definitions:
            gate = self.definitions[call.name]
            shape = (len(gate.parameters), len(gate.qubits))
        else:
            raise QasmError(f"Unresolved gate {call.name!r}")
        if (len(call.parameters), len(call.qubits)) != shape:
            raise QasmError("Gate argument count mismatch")

    def validate_definitions(self) -> None:
        # Validate unused definitions too: no unresolved names or cycle escapes.
        checked: dict[str, int] = {}

        def visit(name: str, active: tuple[str, ...]) -> int:
            if name in active:
                raise QasmError("Recursive gate definitions are unsupported")
            if len(active) >= self.max_definition_depth:
                raise QasmError("Gate-definition depth cap exceeded")
            if name in checked:
                height = checked[name]
                if len(active) + height > self.max_definition_depth:
                    raise QasmError("Gate-definition depth cap exceeded")
                return height
            gate = self.definitions[name]
            height = 1
            for call in gate.body:
                self.arity(call)
                # Syntax/name validation is independent of argument values;
                # actual value/division checks happen at each bound expansion.
                for tokens in call.parameters:
                    allowed = set(gate.parameters) | {"pi", "+", "-", "*", "/", "(", ")"}
                    if any(token not in allowed and not _NUMBER.fullmatch(token) for token in tokens):
                        raise QasmError("Unresolved gate angle")
                    _expression(tokens, dict.fromkeys(gate.parameters, 0.125), syntax_only=True)
                if call.name in self.definitions:
                    height = max(height, visit(call.name, active + (name,)) + 1)
            checked[name] = height
            return height

        for name in self.definitions:
            visit(name, ())

    def parse(self) -> ParsedCircuit:
        self.pop("OPENQASM")
        self.pop("3.0")
        self.pop(";")
        self.pop("include")
        self.pop('"stdgates.inc"')
        self.pop(";")
        while self.peek():
            token = self.peek()
            if token == "gate":
                if self.calls:
                    raise QasmError("Gate definitions must precede the fixed schedule")
                self.definition()
            elif token in {"qubit", "bit"}:
                if self.calls:
                    raise QasmError("Register declarations must precede the fixed schedule")
                kind = self.pop()
                self.pop("[")
                width = self.integer()
                self.pop("]")
                name = self.identifier()
                self.pop(";")
                if name in self.registers or any(value[0] == kind for value in self.registers.values()):
                    raise QasmError("Exactly one register of each kind is supported")
                if not 1 <= width <= self.max_qubits:
                    raise QasmError("Register-width cap exceeded")
                self.registers[name] = (kind, width)
            else:
                self.calls.append(self.call())
                if len(self.calls) > self.max_operations:
                    raise QasmError("Top-level operation cap exceeded")
        sizes = {kind: width for kind, width in self.registers.values()}
        if set(sizes) != {"bit", "qubit"}:
            raise QasmError("One quantum and one classical register are required")
        self.validate_definitions()
        operations: list[Operation] = []
        top: list[Operation] = []

        def expand(call: _Call, parameters: dict[str, float], wires: dict[str, int], depth: int) -> None:
            self.arity(call)
            values = tuple(_expression(tokens, parameters) for tokens in call.parameters)
            qubits = tuple(wires[wire] if isinstance(wire, str) else wire for wire in call.qubits)
            if call.name in self.definitions:
                if depth >= self.max_definition_depth:
                    raise QasmError("Gate-definition expansion depth cap exceeded")
                gate = self.definitions[call.name]
                local_parameters = dict(zip(gate.parameters, values, strict=True))
                local_wires = dict(zip(gate.qubits, qubits, strict=True))
                for nested in gate.body:
                    expand(nested, local_parameters, local_wires, depth + 1)
            else:
                operations.append(Operation(call.name, values, qubits, call.clbits))
                if len(operations) > self.max_operations:
                    raise QasmError("Expanded operation cap exceeded")

        for call in self.calls:
            self.arity(call)
            values = tuple(_expression(tokens, {}) for tokens in call.parameters)
            top.append(Operation(call.name, values, tuple(call.qubits), call.clbits))
            expand(call, {}, {}, 0)
        return ParsedCircuit(sizes["qubit"], sizes["bit"], tuple(operations), tuple(top), len(self.definitions))


def parse_qasm(data: bytes | str, *, max_bytes: int = 16 * 1024 * 1024,
               max_qubits: int = 8, max_operations: int = 500_000,
               max_tokens: int = 2_000_000, max_definitions: int = 256,
               max_definition_depth: int = 32) -> ParsedCircuit:
    """Parse authenticated bytes without imports, execution, or name inference."""
    _bounded_integer(max_bytes, "max_bytes", 16 * 1024 * 1024)
    _bounded_integer(max_qubits, "max_qubits", 9)
    _bounded_integer(max_operations, "max_operations", 500_000)
    _bounded_integer(max_tokens, "max_tokens", 2_000_000)
    _bounded_integer(max_definitions, "max_definitions", 256)
    _bounded_integer(max_definition_depth, "max_definition_depth", 32)
    if isinstance(data, bytes):
        if len(data) > max_bytes:
            raise QasmError("QASM byte cap exceeded")
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError as error:
            raise QasmError("Saved QASM must be ASCII") from error
    elif isinstance(data, str):
        if len(data) > max_bytes or not data.isascii():
            raise QasmError("QASM byte/ASCII cap exceeded")
        text = data
    else:
        raise QasmError("QASM input must be bytes or text")
    return _Parser(_tokens(text, max_tokens), max_qubits, max_operations,
                   max_definitions, max_definition_depth).parse()
