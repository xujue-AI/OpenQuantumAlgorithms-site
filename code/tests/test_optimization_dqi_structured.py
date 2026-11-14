"""Pure source-contract regressions; no circuit or simulator imports/calls.

The separately approved validator performs each new n=2/3 quantum check once.
These tests exercise input/cap/streaming/quality guards, not the old default.
Direct file loading avoids the repository package initializer and its unrelated
normal dependencies. The locked pytest stage runs with plugin autoload disabled.
"""
import builtins
from fractions import Fraction
import importlib.util
import json
import stat
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    path = ROOT / "src/openquantum" / filename
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


CORE = _load("_structured_contract_core", "optimization_dqi_structured.py")
VALIDATION = _load("_structured_contract_validation", "optimization_dqi_structured_validation.py")


def test_pure_sources_load_without_requesting_a_quantum_dependency(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"qiskit", "numpy", "openquantum", "qdk"}:
            pytest.fail("A pure source import requested an unrelated/quantum backend")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _load("_structured_guarded_core", "optimization_dqi_structured.py")
    _load("_structured_guarded_validation", "optimization_dqi_structured_validation.py")


@pytest.mark.parametrize("n", [0, 1, -1, 65, True, 2.0, "3"])
def test_family_bounds_reject_before_an_operation_is_created(monkeypatch, n):
    monkeypatch.setattr(CORE, "Operation", lambda *a, **k: pytest.fail("Unexpected primitive allocation"))
    with pytest.raises(ValueError):
        CORE.iter_operations(n)


@pytest.mark.parametrize("field,value", [
    ("max_n", 65), ("max_qubits", 193), ("max_operations", 200001), ("max_n", True),
])
def test_materialization_caps_can_only_tighten(field, value):
    with pytest.raises(ValueError):
        CORE.StructuredDQILimits(**{field: value})


@pytest.mark.parametrize("n,limits", [
    (3, {"max_qubits": 8}), (3, {"max_operations": 335}), (64, {"max_n": 63}),
])
def test_complete_draw_cap_rejection_precedes_stream_allocation(monkeypatch, n, limits):
    accepted_limits = CORE.StructuredDQILimits(**limits)
    monkeypatch.setattr(CORE, "Operation", lambda *a, **k: pytest.fail("Unexpected primitive allocation"))
    with pytest.raises(ValueError):
        CORE.iter_operations(n, accepted_limits)


@pytest.mark.parametrize("angle", [
    float.fromhex("0x0.0000000000001p-1022"), -float.fromhex("0x0.0000000000001p-1022"),
    float.fromhex("0x0.0000000000003p-1022"), -float.fromhex("0x0.0000000000003p-1022"),
])
def test_lossy_half_angle_rejects_before_primitive_construction(monkeypatch, angle):
    monkeypatch.setattr(CORE, "Operation", lambda *a, **k: pytest.fail("Unexpected primitive allocation"))
    with pytest.raises(ValueError):
        CORE.controlled_ry_operations(0, 1, angle)


@pytest.mark.parametrize("value", [
    "1e-1000000000000000", "0.01", "١/٢", "1/" + "9" * 65, "1/" + "9" * 63,
])
def test_unbounded_fraction_grammar_rejects_before_fraction_constructor(monkeypatch, value):
    monkeypatch.setattr(CORE, "Fraction", lambda *a, **k: pytest.fail("Unexpected Fraction construction"))
    with pytest.raises(ValueError):
        CORE._fraction(value)


@pytest.mark.parametrize("value", ["1/0", "1/000"])
def test_zero_denominator_rejects_before_fraction_constructor(monkeypatch, value):
    monkeypatch.setattr(CORE, "Fraction", lambda *a, **k: pytest.fail("Unexpected Fraction construction"))
    with pytest.raises(ValueError):
        CORE._fraction(value)


@pytest.mark.parametrize("n,minimum", [(2, 4), (3, 2)])
def test_exact_repetition_boundary_includes_equality(n, minimum):
    plan = CORE.repetition_plan(n, Fraction(1, 100), max_draws=4)
    assert plan["status"] == "feasible"
    assert plan["required_draws"] == minimum
    ratio = plan["failure_probability_after_charged_draws"]
    failure = Fraction(int(ratio["numerator_hex"], 16), int(ratio["denominator_hex"], 16))
    assert failure <= Fraction(1, 100)
    if n == 3:
        assert failure == Fraction(1, 100)


@pytest.mark.parametrize("n", [3, 64])
def test_quality_cap_exhaustion_is_explicit_and_does_not_expand_the_draws(n):
    plan = CORE.repetition_plan(n, Fraction(1, 100), max_draws=1)
    assert plan["status"] == "cap_exceeded"
    assert plan["required_draws"] is None
    assert plan["draws_charged"] == 1
    assert plan["minimum_required_draws_lower_bound"] == 2
    assert plan["quality_feasible"] is False


@pytest.mark.parametrize("n", [1, 4, 8, True, 2.0])
def test_simulation_scope_is_independent_of_larger_generation(n):
    with pytest.raises(ValueError):
        VALIDATION.validate_small_bounds(n)


def test_tightened_small_bound_fails_before_sources_backend_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(VALIDATION, "_runtime_preflight", lambda *a: pytest.fail("Unexpected numerical backend"))
    monkeypatch.setattr(VALIDATION, "_load_core", lambda *a: pytest.fail("Unexpected source execution"))
    output = tmp_path / "never-started"
    with pytest.raises(ValueError):
        VALIDATION.validate_small_structured(output, limits=VALIDATION.SmallValidationLimits(max_qubits=6))
    assert not output.exists()


def test_fresh_validation_requires_existing_parent_before_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(VALIDATION, "_runtime_preflight", lambda *a: pytest.fail("Unexpected numerical backend"))
    monkeypatch.setattr(VALIDATION, "_load_core", lambda *a: pytest.fail("Unexpected source execution"))
    parent = tmp_path / "missing-parent"
    with pytest.raises(ValueError):
        VALIDATION.validate_small_structured(parent / "new-output")
    assert not parent.exists()


def test_opaque_gate_is_not_an_exportable_primitive():
    with pytest.raises(ValueError):
        CORE.Operation("ccx", (0, 1, 2))


def test_postreceipt_directory_sync_failure_cannot_return_acceptance(monkeypatch, tmp_path):
    # This intentionally incomplete toy JSON is not scientific fixture data.
    # It checks the actual file-status/invocation boundary using only stdlib I/O.
    raw = b'{"status":"complete","completion_is_provisional":true}\n'
    original_fsync = VALIDATION.os.fsync
    def fail_directory_sync(fd):
        if stat.S_ISDIR(VALIDATION.os.fstat(fd).st_mode):
            raise OSError("mocked postreplacement directory fsync failure")
        return original_fsync(fd)
    writer = VALIDATION._ArtifactWriter(tmp_path, 4096, VALIDATION.time.monotonic() + 30)
    with monkeypatch.context() as context:
        context.setattr(VALIDATION.os, "fsync", fail_directory_sync)
        with pytest.raises(OSError):
            writer.write("toy-status.json", raw, bind=False)
    retained = json.loads((tmp_path / "toy-status.json").read_bytes())
    assert retained["completion_is_provisional"] is True
