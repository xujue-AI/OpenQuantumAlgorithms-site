"""Meaningful policy/guard regressions; native synthesis is mocked throughout."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate
from qiskit.circuit.library import CXGate, RXGate
from qiskit.quantum_info import Operator

from openquantum import instrument_resources as ir
from openquantum.simulation import bounded_density_instrument as legacy_density
from openquantum.simulation import lindblad_four_compile as c
from openquantum.simulation.two_spin_lindblad import build_chain_circuit


@pytest.fixture
def limits():
    return c.FourCompilerLimits(8, 20_000, 100_000, 2_000_000)


@pytest.fixture
def source():
    qc = QuantumCircuit(8, 8)
    qc.reset(range(8))
    qc.rx(math.pi/2, 0)
    qc.rz(math.pi/2, 0)
    qc.rz(math.pi/2, 0)
    qc.measure(range(8), range(8))
    return qc


def exact_clifford(matrix):
    """Independent exact pi/2 rotation circuits, including their global phase."""
    candidates = []
    for name in ("rx", "ry", "rz"):
        value = QuantumCircuit(1)
        if name == "rx":
            value.h(0); value.s(0); value.h(0)
        elif name == "ry":
            value.sdg(0); value.h(0); value.s(0); value.h(0); value.s(0)
        else:
            value.s(0)
        value.global_phase = -math.pi/4
        candidates.append(value)
    for value in candidates:
        if np.linalg.norm(Operator(value).data-matrix, ord="fro") < 1e-12:
            return value
    pytest.fail("Mock received a rotation outside the deterministic Clifford fixture")


@pytest.fixture
def native(monkeypatch):
    calls = []
    def mocked(matrix, epsilon):
        calls.append((matrix.copy(), epsilon))
        return exact_clifford(matrix)
    monkeypatch.setattr(c, "gridsynth_unitary", mocked)
    return calls


def test_limits_are_mandatory_and_do_not_change_old_domains(source, limits):
    with pytest.raises(TypeError):
        c.FourCompilerLimits()
    with pytest.raises(TypeError):
        c.route_source(source)
    with pytest.raises(TypeError):
        c.synthesize_literal_four(source, 5e-5)
    assert asdict(limits) == {"max_qubits": 8, "source_operations": 20_000,
                              "routed_operations": 100_000, "synthesized_operations": 2_000_000}
    assert ir.InstrumentLimits().max_qubits == 5
    assert legacy_density.DenseInstrumentLimits().max_qubits == 5
    with pytest.raises(ValueError):
        ir.InstrumentLimits(synthesized_operations=2_000_000)
    with pytest.raises(ValueError):
        legacy_density.DenseInstrumentLimits(max_qubits=8)


@pytest.mark.parametrize("field,value", [
    ("max_qubits", 7), ("max_qubits", 8.), ("max_qubits", True),
    ("source_operations", 100_001), ("source_operations", 0),
    ("routed_operations", 100_001), ("synthesized_operations", 2_000_001),
    ("synthesized_operations", False), ("synthesized_operations", 2.0),
])
def test_explicit_hard_limits_reject(field, value, limits):
    values = asdict(limits); values[field] = value
    with pytest.raises(ValueError):
        c.FourCompilerLimits(**values)


def test_forged_limits_and_subclasses_reject_before_routing(source, limits, monkeypatch):
    class Override(c.FourCompilerLimits):
        def __post_init__(self):
            pass
    monkeypatch.setattr(c, "transpile", lambda *a, **k: pytest.fail("invalid limits must not route"))
    with pytest.raises(ValueError, match="explicit FourCompilerLimits"):
        c.route_source(source, limits=Override(8, 20_000, 100_000, 9_000_000))
    object.__setattr__(limits, "synthesized_operations", 9_000_000)
    with pytest.raises(ValueError, match="synthesized_operations"):
        c.route_source(source, limits=limits)


@pytest.mark.parametrize("allowance", [True, False, math.nan, math.inf, 1e-6, .1])
def test_fixed_error_request_is_not_relaxed(source, limits, allowance, monkeypatch):
    monkeypatch.setattr(c, "gridsynth_unitary", lambda *a: pytest.fail("invalid allowance must not synthesize"))
    with pytest.raises(ValueError, match="fixed finite 5e-5"):
        c.synthesize_literal_four(source, allowance, limits=limits)


def test_raw_operation_boundary_precedes_transpile(source, limits, monkeypatch):
    monkeypatch.setattr(c, "transpile", lambda *a, **k: pytest.fail("raw cap must precede routing"))
    values = asdict(limits); values["source_operations"] = len(source.data)-1
    with pytest.raises(c.CompilerLimitExceeded) as raised:
        c.route_source(source, limits=c.FourCompilerLimits(**values))
    assert raised.value.as_record()["partial_trial_is_validated"] is False
    assert raised.value.actual_operations == len(source.data)


def test_fixed_frozen_derivation_identity(limits, source, monkeypatch):
    assert hashlib.sha256(Path(ir.__file__).read_bytes()).hexdigest() == c.FROZEN_IR_SOURCE_SHA256
    record = c.derivation_record()
    assert record["shared_IR_limits_modified"] is record["shared_IR_report_limits_forged"] is False
    monkeypatch.setattr(c, "FROZEN_IR_SOURCE_SHA256", "0"*64)
    monkeypatch.setattr(c, "transpile", lambda *a, **k: pytest.fail("wrong derivation must not route"))
    with pytest.raises(ValueError, match="frozen shared"):
        c.route_source(source, limits=limits)


def test_actual_literal_source_route_is_frozen_and_does_not_synthesize(limits, monkeypatch):
    source = build_chain_circuit(4, 1., .6, .35, (.8, .5, .5, .5), 211, 2,
                                 "product", observable="Z")
    original = c.transpile; calls = []
    def routed(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(c, "transpile", routed)
    monkeypatch.setattr(c, "gridsynth_unitary", lambda *a: pytest.fail("source preflight cannot synthesize"))
    result = c.route_source(source, limits=limits)
    assert [len(value.data) for value in (result.source, result.lowered, result.routed)] == [9934, 17343, 24510]
    assert result.routed.count_ops()["cx"] == 14130
    assert result.report["rotation_occurrences"] == 5908
    assert result.report["routed_final_measurement_wires"] == [6, 1, 2, 3, 7, 4, 0, 5]
    assert result.routed.count_ops()["reset"] == 1481 and result.routed.count_ops()["measure"] == 8
    assert all(value["optimization_level"] == 0 and value["seed_transpiler"] == 7
               and value["qubits_initially_zero"] is False for value in calls)
    assert calls[1]["initial_layout"] == list(range(8)) and calls[1]["routing_method"] == "sabre"
    assert result.report["actual_clifford_t_synthesis_performed"] is False
    assert "limits" not in result.report


def test_actual_lowering_cap_precedes_native(source, monkeypatch):
    monkeypatch.setattr(c, "gridsynth_unitary", lambda *a: pytest.fail("routing cap cannot synthesize"))
    with pytest.raises(c.CompilerLimitExceeded) as raised:
        c.route_source(source, limits=c.FourCompilerLimits(8, 20_000, 50, 2_000_000))
    assert raised.value.stage == "lowered_rotation_schedule" and raised.value.actual_operations == 51


def test_cached_targets_preserve_all_occurrences_and_phases(source, limits, native):
    result = c.synthesize_literal_four(source, 5e-5, limits=limits)
    assert len(native) == 2
    assert all(value[1] == 5e-5/3/4 for value in native)
    assert result.report["rotation_occurrences"] == 3
    assert result.report["unique_rotation_targets"] == 2
    assert result.circuit.count_ops()["h"] == 2 and result.circuit.count_ops()["s"] == 3
    assert result.circuit.count_ops()["reset"] == result.circuit.count_ops()["measure"] == 8
    assert len(result.circuit.data) == 53
    assert result.report["clifford_t_counts"] == dict(result.circuit.count_ops())
    assert result.report["materialization_limits"] == asdict(limits) and "limits" not in result.report
    weighted = sum(next(a["guarded_frobenius_error"] for a in result.report["approximants"] if a["gate"] == name)
                   for name in ("rx", "rz", "rz"))
    assert result.report["guarded_numerical_operator_error_upper_estimate"] == weighted
    assert abs(np.exp(1j*result.circuit.global_phase)-np.exp(-3j*math.pi/4)) < 1e-12
    assert result.report["compiled_global_phase"] == result.circuit.global_phase
    assert c._events(result.circuit) == c._events(source)


def test_exact_emitted_operation_boundary(source, native):
    result = c.synthesize_literal_four(source, 5e-5, limits=c.FourCompilerLimits(8, 20_000, 100_000, 53))
    assert len(result.circuit.data) == 53
    with pytest.raises(c.CompilerLimitExceeded) as raised:
        c.synthesize_literal_four(source, 5e-5, limits=c.FourCompilerLimits(8, 20_000, 100_000, 52))
    assert raised.value.actual_operations == 53


def test_next_rotation_segment_cap_is_checked_before_compose(source, native, monkeypatch):
    original = QuantumCircuit.compose; composed = []
    def tracked(self, *args, **kwargs):
        if self.name == "Clifford+T literal four instrument":
            composed.append(len(self.data))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(QuantumCircuit, "compose", tracked)
    with pytest.raises(c.CompilerLimitExceeded) as raised:
        c.synthesize_literal_four(source, 5e-5, limits=c.FourCompilerLimits(8, 20_000, 100_000, 26))
    assert composed == []
    assert raised.value.as_record() == {"status": "uncomputed_compiler_operation_cap",
        "stage": "clifford_t_schedule", "cap": 26, "actual_operations": 27,
        "completed_operations": 24, "next_segment_operations": 3, "partial_trial_is_validated": False}


def test_native_retry_uses_unchanged_quarter_then_halving(source, limits, monkeypatch):
    attempts, calls = {}, []
    def mock(matrix, epsilon):
        key = matrix.tobytes(); attempts[key] = attempts.get(key, 0)+1; calls.append((key, epsilon))
        if attempts[key] == 1:
            bad = QuantumCircuit(1); bad.x(0); return bad
        return exact_clifford(matrix)
    monkeypatch.setattr(c, "gridsynth_unitary", mock)
    c.synthesize_literal_four(source, 5e-5, limits=limits)
    assert len(calls) == 4
    assert all([epsilon for key, epsilon in calls if key == target] == [5e-5/3/4, 5e-5/3/8]
               for target in attempts)


def test_four_attempt_failure_has_no_mean_fit_or_extra_retry(source, limits, monkeypatch):
    calls = []
    def bad(matrix, epsilon):
        calls.append(epsilon); value = QuantumCircuit(1); value.x(0); return value
    monkeypatch.setattr(c, "gridsynth_unitary", bad)
    with pytest.raises(RuntimeError, match="guarded per-gate"):
        c.synthesize_literal_four(source, 5e-5, limits=limits)
    assert calls == [5e-5/3/4, 5e-5/3/8, 5e-5/3/16, 5e-5/3/32]


def test_nonfinite_computed_error_rejects_before_compose(source, limits, monkeypatch):
    value = exact_clifford(Operator(RXGate(math.pi/2)).data)
    monkeypatch.setattr(c, "gridsynth_unitary", lambda *a: value.copy())
    monkeypatch.setattr(c.np.linalg, "norm", lambda *a, **k: math.nan)
    with pytest.raises(ValueError, match="nonfinite guarded"):
        c.synthesize_literal_four(source, 5e-5, limits=limits)


@pytest.mark.parametrize("kind", ["classical", "width", "rotation", "hidden_named_primitive"])
def test_native_output_must_be_flat_ancilla_free_ct(source, limits, monkeypatch, kind):
    def invalid(*args):
        if kind == "classical":
            return QuantumCircuit(1, 1)
        if kind == "width":
            return QuantumCircuit(2)
        result = QuantumCircuit(1)
        if kind == "rotation":
            result.rx(math.pi/2, 0)
        else:
            operation = Gate("x", 1, [])
            definition = QuantumCircuit(1); definition.x(0); definition.x(0); definition.x(0)
            operation.definition = definition; result.append(operation, [0]); result.global_phase = -math.pi/2
        return result
    monkeypatch.setattr(c, "gridsynth_unitary", invalid)
    with pytest.raises(ValueError):
        c.synthesize_literal_four(source, 5e-5, limits=limits)


def test_named_generic_primitive_rejects_in_routed_basis(source, limits, monkeypatch):
    bad = source.copy()
    operation = Gate("rx", 1, [math.pi/2]); definition = QuantumCircuit(1); definition.rx(math.pi/2, 0)
    operation.definition = definition
    bad.data[8] = CircuitInstruction(operation, (bad.qubits[0],), ())
    monkeypatch.setattr(c, "transpile", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="canonical flat primitive"):
        c.route_source(source, limits=limits)


@pytest.mark.parametrize("kind", ["hidden_reset", "nested_duplicate", "canonical_subclass", "nonfinite"])
def test_malformed_or_hidden_source_rejects_before_transpile(source, limits, monkeypatch, kind):
    if kind == "hidden_reset":
        operation = Gate("concealed", 1, []); definition = QuantumCircuit(1); definition.reset(0)
        operation.definition = definition; wires = (source.qubits[0],)
    elif kind == "nested_duplicate":
        operation = Gate("malformed", 2, []); definition = QuantumCircuit(2)
        definition._append(CircuitInstruction(CXGate(), (definition.qubits[0], definition.qubits[0]), ()))
        operation.definition = definition; wires = tuple(source.qubits[:2])
    elif kind == "canonical_subclass":
        class BadRX(RXGate):
            @property
            def base_class(self):
                return RXGate
            def to_matrix(self):
                return np.eye(2)
        operation = BadRX(math.pi/2); wires = (source.qubits[0],)
    else:
        operation = RXGate(math.nan); wires = (source.qubits[0],)
    source.data[8] = CircuitInstruction(operation, wires, ())
    monkeypatch.setattr(c, "transpile", lambda *a, **k: pytest.fail("invalid source must fail before transpile"))
    with pytest.raises(ValueError):
        c.route_source(source, limits=limits)


def test_duplicate_ignored_readout_cannot_pass_count_only_guard(source, limits, monkeypatch):
    item = source.data[-1]
    source.data[-1] = CircuitInstruction(item.operation, (source.qubits[0],), item.clbits)
    monkeypatch.setattr(c, "transpile", lambda *a, **k: pytest.fail("duplicate readout must not route"))
    with pytest.raises(ValueError, match="every physical wire once"):
        c.route_source(source, limits=limits)


def test_legacy_cap_is_validation_not_a_legacy_synthesis_attempt(source, limits, native, monkeypatch):
    result = c.synthesize_literal_four(source, 5e-5, limits=limits)
    original = ir._source_check; calls = []
    def checked(circuit, operation_limit, *, max_qubits):
        calls.append((circuit, operation_limit, max_qubits))
        return original(circuit, operation_limit, max_qubits=max_qubits)
    monkeypatch.setattr(ir, "_source_check", checked)
    record = c.legacy_cap_check(result.circuit)
    assert calls == [(result.circuit, 1_000_000, 8)]
    assert record["status"] == "passed_frozen_operation_guard"
    assert record["legacy_synthesis_performed"] is record["legacy_physical_estimation_performed"] is False
    assert record["actual_operations"] == 53 and record["legacy_operation_cap"] == 1_000_000


def test_legacy_negative_validation_is_not_report_forging(source, limits, native, monkeypatch):
    result = c.synthesize_literal_four(source, 5e-5, limits=limits)
    before = result.report.copy()
    def rejected(*args, **kwargs):
        raise ValueError("bounded fixture rejected by frozen source validator")
    monkeypatch.setattr(ir, "_source_check", rejected)
    record = c.legacy_cap_check(result.circuit)
    assert record["status"] == "rejected_frozen_validation_guard" and record["rejection_reproduced"] is True
    assert record["legacy_synthesis_performed"] is False
    assert result.report == before and "limits" not in result.report
