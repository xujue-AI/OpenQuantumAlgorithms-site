"""Stdlib parser/preflight checks; these never execute scientific replay."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qasm = _load("tail_qasm_parser_under_test", "tail_resource_qasm.py")
replay = _load("tail_qasm_replay_under_test", "tail_resource_replay.py")


def _text(body, gates="", *, qubits=2, clbits=2):
    return (f'OPENQASM 3.0; include "stdgates.inc"; {gates} '
            f'bit[{clbits}] c; qubit[{qubits}] q; {body}')


def test_bound_parameters_and_baked_custom_bodies_are_preserved():
    gates = "gate bound(a,b) w { rz(-a+2*b/pi) w; } gate c3rz(a) w { ry(pi/4) w; }"
    parsed = qasm.parse_qasm(_text("bound(pi/2,pi/4) q[0]; c3rz(999) q[1]; c[0] = measure q[1];", gates))
    assert parsed.operations[0].parameters == pytest.approx((-math.pi/2+.5,))
    assert parsed.operations[1].parameters == pytest.approx((math.pi/4,))
    assert parsed.counts(expanded=False) == {"bound": 1, "c3rz": 1, "measure": 1}
    assert parsed.counts() == {"measure": 1, "ry": 1, "rz": 1}
    assert parsed.events()[0].qubits == (1,)
    assert parsed.events()[0].clbits == (0,)


def test_full_cu_parameter_list_and_event_maps_reach_circuit_adapter():
    parsed = qasm.parse_qasm(_text("reset q[1]; cu(pi/4,0,-pi/4,pi/2) q[0],q[1]; barrier q[1],q[0]; c[0] = measure q[1];"))

    class RecordingCircuit:
        def __init__(self, qubits, clbits):
            self.widths, self.calls = (qubits, clbits), []

        def __getattr__(self, name):
            def record(*values):
                self.calls.append((name, values))
            return record

    circuit = replay._circuit(parsed, RecordingCircuit)
    assert circuit.widths == (2, 2)
    assert circuit.calls == [
        ("reset", (1,)), ("cu", (math.pi/4, 0.0, -math.pi/4, math.pi/2, 0, 1)),
        ("barrier", (1, 0)), ("measure", (1, 0))]


@pytest.mark.parametrize("body", [
    "rz(__import__(os)) q[0];", "rz(unknown) q[0];", "rz(pi/) q[0];",
    "rz(1e999) q[0];", "rz(1/0) q[0];", "rz(2**3) q[0];",
    "unknown q[0];", "cx q[0],q[0];", "cx q[0];", "h(pi) q[0];",
    "h q[2];", "h c[0];", "c[2] = measure q[0];", "q[0] = measure q[1];",
    "measure q[0];", "reset q;", "if (c[0]) { x q[0]; }", "ctrl @ x q[0],q[1];",
    "for int i in [0:2] { h q[0]; }", "include \"outside.inc\";", "gphase(pi);",
    "bit[2] c;", "qubit[2] r;", "h q[0]; bit[2] d;",
])
def test_unsupported_or_unsafe_source_is_rejected(body):
    with pytest.raises(qasm.QasmError):
        qasm.parse_qasm(_text(body))


@pytest.mark.parametrize("gates", [
    "gate h w { x w; }", "gate a w { x w; } gate a w { h w; }",
    "gate a w { a w; }", "gate a w { b w; } gate b w { a w; }",
    "gate a w { missing w; }", "gate a w { reset w; }", "gate a w { measure w; }",
    "gate a w { rz(missing) w; }", "gate a w { rz(pi/) w; }",
    "gate a w { rz(1e999/1) w; }", "gate a w { rz(1/0) w; }",
    "gate a(v) w { rz(v/0) w; }", "gate a(v) w { rz(v/(1-1)) w; }",
    "gate if w { x w; }", "gate a(if) w { rz(if) w; }",
    "gate a(v,v) w { rz(v) w; }", "gate a(v) v { rz(v) v; }",
    "gate a w { cx w; }", "gate a w { }",
])
def test_unused_invalid_or_hidden_gate_definitions_are_rejected(gates):
    with pytest.raises(qasm.QasmError):
        qasm.parse_qasm(_text("h q[0];", gates))


def test_distinct_definition_expression_and_expansion_caps():
    gates = "gate a w { x w; x w; } gate b w { a w; a w; } gate d w { b w; b w; }"
    with pytest.raises(qasm.QasmError, match="Expanded operation"):
        qasm.parse_qasm(_text("d q[0];", gates), max_operations=7)
    with pytest.raises(qasm.QasmError, match="depth cap"):
        qasm.parse_qasm(_text("d q[0];", gates), max_definition_depth=2)
    with pytest.raises(qasm.QasmError, match="definition cap"):
        qasm.parse_qasm(_text("d q[0];", gates), max_definitions=2)
    with pytest.raises(qasm.QasmError, match="Angle expression"):
        qasm.parse_qasm(_text("rz(" + "+".join(["1"]*70) + ") q[0];"))
    with pytest.raises(qasm.QasmError, match="token cap"):
        qasm.parse_qasm(_text("h q[0];"), max_tokens=10)
    with pytest.raises(qasm.QasmError, match="byte cap"):
        qasm.parse_qasm(_text("h q[0];").encode(), max_bytes=4)
    with pytest.raises(qasm.QasmError, match="ASCII"):
        qasm.parse_qasm(b"\xff")


def test_nine_wire_extension_requires_explicit_parser_opt_in():
    text = 'OPENQASM 3.0; include "stdgates.inc"; bit[9] meas; qubit[9] q; rx(-3*pi/7) q[8]; meas[8] = measure q[8];'
    with pytest.raises(qasm.QasmError, match="width"):
        qasm.parse_qasm(text)
    parsed = qasm.parse_qasm(text, max_qubits=9)
    assert (parsed.num_qubits, parsed.num_clbits) == (9, 9)
    assert parsed.operations[0].parameters == pytest.approx((-3*math.pi/7,))
    assert parsed.operations[-1].qubits == (8,)
    assert parsed.operations[-1].clbits == (8,)


def test_missing_numeric_fields_cannot_be_silently_skipped():
    with pytest.raises(replay.ReplayError, match="missing required"):
        replay._field_residuals({}, {}, None)
    inventory = _inventory("instrument-resources")
    report = json.loads(inventory["damping-Z/report.json"])
    report["ideal_instrument"].pop("mean")
    inventory["damping-Z/report.json"] = json.dumps(report).encode()
    with pytest.raises(replay.ReplayError, match="missing required"):
        replay._preflight("instrument", inventory,
                          {"src/openquantum/instrument_resources.py": inventory["instrument_resources.py"]}, qasm)


def _inventory(folder):
    directory = ROOT / "assets" / "generated" / folder
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kind,folder,case_count", [
    ("instrument", "instrument-resources", 4),
    ("complete", "simulation/complete-resources", 5),
])
def test_all_saved_qasm_and_numeric_helper_ast_preflight_without_execution(kind, folder, case_count):
    inventory = _inventory(folder)
    sources = {"src/openquantum/instrument_resources.py": (ROOT/"src/openquantum/instrument_resources.py").read_bytes()}
    trials, helper_tree = replay._preflight(kind, inventory, sources, qasm)
    assert len(trials) == case_count
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                   for statement in helper_tree.body[1:] for node in ast.walk(statement))
    for trial in trials:
        assert len(trial.source.operations) <= 1000
        assert len(trial.compiled.operations) <= 81_000
        assert trial.compiled.counts() == trial.report["actual_synthesis"]["clifford_t_counts"]


def test_authentication_preflight_rejects_changed_snapshot_and_qasm_bytes():
    inventory = _inventory("instrument-resources")
    sources = {"src/openquantum/instrument_resources.py": inventory["instrument_resources.py"]}
    changed_sources = {"src/openquantum/instrument_resources.py": b"# changed"}
    with pytest.raises(replay.ReplayError, match="differs"):
        replay._preflight("instrument", inventory, changed_sources, qasm)
    changed = dict(inventory)
    changed["damping-Z/source-instrument.qasm"] += b"\n"
    with pytest.raises(replay.ReplayError, match="hash mismatch"):
        replay._preflight("instrument", changed, sources, qasm)


def test_preflight_never_imports_scientific_packages_in_a_clean_interpreter():
    script = """
import importlib.abc, importlib.util, pathlib, sys
root = pathlib.Path(sys.argv[1])
class RejectScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','qiskit','qdk','qsharp','openquantum'}:
            raise AssertionError('scientific import during preflight: ' + fullname)
sys.meta_path.insert(0, RejectScientificImports())
def load(name):
    spec = importlib.util.spec_from_file_location(name, root/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
parser, worker = load('tail_resource_qasm'), load('tail_resource_replay')
source = {'src/openquantum/instrument_resources.py': (root/'src/openquantum/instrument_resources.py').read_bytes()}
count = 0
for kind, folder in [('instrument','instrument-resources'), ('complete','simulation/complete-resources')]:
    directory = root/'assets/generated'/folder
    records = {p.relative_to(directory).as_posix(): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    trials, _ = worker._preflight(kind, records, source, parser)
    count += len(trials)
assert count == 9
print('nine cases preflighted without scientific imports')
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                               check=True, capture_output=True, text=True, timeout=30)
    assert completed.stdout.strip() == "nine cases preflighted without scientific imports"
