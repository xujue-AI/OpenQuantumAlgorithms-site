"""Pure public-harness boundaries; no frozen source or numerical work is run.

The target loaded by the fixture is the new standard-library harness only.
These tests use tiny synthetic inputs, a fake Operation constructor and
injected clocks/RSS.  They never call reproduce_saved_inputs or load the
accepted generator, reference, cases, schedules or scientific result bundle.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TARGET = (Path(__file__).resolve().parents[1] / "src/openquantum/simulation"
          / "reusable_bath_local_spin_reproduce.py")
_ABSENT = object()


@pytest.fixture
def harness():
    """Fresh-load only the new harness and restore its module slot afterward."""
    name = "openquantum_public_reproduction_boundary_target"
    previous = sys.modules.get(name, _ABSENT)
    spec = importlib.util.spec_from_file_location(name, TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError("new harness loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is _ABSENT:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _ledger(harness, tmp_path, name, *, max_seconds=120):
    output = tmp_path.resolve() / name
    output.mkdir()
    clock = [0.0]
    ledger = harness.BoundedLedger(
        output, max_seconds=max_seconds, clock=lambda: clock[0],
        rss_sampler=lambda: 0,
    )
    return ledger, clock, output


def _header(n=1):
    return ("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
            f"qreg q[{n + 1}];\ncreg c[{n}];\n").encode("ascii")


def test_interpreter_refusals_do_not_mutate_shared_runtime(harness, monkeypatch):
    shared_sys, shared_threading = harness.sys, harness.threading
    main = object()
    valid_threads = SimpleNamespace(
        current_thread=lambda: main, main_thread=lambda: main,
        active_count=lambda: 1,
    )
    cases = (
        (1, (3, 12, 14), valid_threads, "optimized_python_is_not_supported"),
        (2, (3, 12, 14), valid_threads, "optimized_python_is_not_supported"),
        (0, (3, 11, 14), valid_threads, "python_3_12_required"),
        (0, (3, 12, 14), SimpleNamespace(
            current_thread=lambda: object(), main_thread=lambda: main,
            active_count=lambda: 1,
        ), "standalone_main_thread_required"),
        (0, (3, 12, 14), SimpleNamespace(
            current_thread=lambda: main, main_thread=lambda: main,
            active_count=lambda: 2,
        ), "standalone_main_thread_required"),
    )
    for optimize, version, threads, expected in cases:
        with monkeypatch.context() as patch:
            patch.setattr(harness, "sys", SimpleNamespace(
                flags=SimpleNamespace(optimize=optimize), version_info=version,
            ))
            patch.setattr(harness, "threading", threads)
            with pytest.raises(harness.ReproductionError) as error:
                harness.check_interpreter()
            assert error.value.code == expected
        assert harness.sys is shared_sys and harness.threading is shared_threading
    with monkeypatch.context() as patch:
        patch.setattr(harness, "sys", SimpleNamespace(
            flags=SimpleNamespace(optimize=0), version_info=(3, 12, 14),
        ))
        patch.setattr(harness, "threading", valid_threads)
        assert harness.check_interpreter() is None


def test_small_qasm_decodes_literals_and_refuses_nonprimitive_input(harness):
    encoded = _header() + (
        b"reset q[0];\nreset q[1];\nx q[1];\nh q[0];\n"
        b"cx q[0],q[1];\nrz(-1.25e-1) q[1];\n"
        b"measure q[0] -> c[0];\n"
    )
    operations = harness.parse_small_qasm(encoded, 1, SimpleNamespace)
    assert [(item.name, item.qubits, item.angle, item.clbit) for item in operations] == [
        ("reset", (0,), None, None), ("reset", (1,), None, None),
        ("x", (1,), None, None), ("h", (0,), None, None),
        ("cx", (0, 1), None, None), ("rz", (1,), -0.125, None),
        ("measure", (0,), None, 0),
    ]
    invalid_lines = (
        b"rz(1 + 1) q[0];\n", b"rz(float('nan')) q[0];\n",
        b"rz(nan) q[0];\n", b"rz(1e999) q[0];\n",
        b"h q[2];\n", b"h q[-1];\n", b"cx q[1],q[1];\n",
        b"measure q[0] -> c[1];\n", b"u3(0,0,0) q[0];\n",
        b"h q[0]; ignored_statement;\n",
    )
    for line in invalid_lines:
        with pytest.raises(harness.ReproductionError):
            harness.parse_small_qasm(_header() + line, 1, SimpleNamespace)
    for n in (True, 1.0, 0, 3):
        with pytest.raises(harness.ReproductionError) as error:
            harness.parse_small_qasm(encoded, n, SimpleNamespace)
        assert error.value.code == "small_system_size_required"
    for malformed in (encoded[:-1], encoded.replace(b"\n", b"\r\n"), bytearray(encoded)):
        with pytest.raises(harness.ReproductionError):
            harness.parse_small_qasm(malformed, 1, SimpleNamespace)


def test_reader_requires_stable_bounded_unlinked_regular_files(harness, tmp_path):
    root = tmp_path.resolve()
    source = root / "tiny.bin"
    contents = b"\x00abcd"
    source.write_bytes(contents)
    assert harness.read_regular(source, len(contents)) == contents
    for maximum in (0, True, len(contents) - 1):
        with pytest.raises(harness.ReproductionError) as error:
            harness.read_regular(source, maximum)
        assert str(source) not in str(error.value)
    with pytest.raises(harness.ReproductionError):
        harness.read_regular(root, 32)
    link = root / "symbolic.bin"
    link.symlink_to(source)
    with pytest.raises(harness.ReproductionError) as error:
        harness.read_regular(link, 32)
    assert error.value.code == "symlink_input_refused"
    hardlink = root / "hardlinked.bin"
    os.link(source, hardlink)
    for path in (source, hardlink):
        with pytest.raises(harness.ReproductionError) as error:
            harness.read_regular(path, 32)
        assert error.value.code == "bounded_regular_input_required"
    assert source.read_bytes() == contents and link.is_symlink()


def test_counter_refusals_preserve_completed_prefix(harness, tmp_path):
    ledger, _, _ = _ledger(harness, tmp_path, "counter-output")
    initial = dict(ledger.counters)
    for name, amount in (
        ("source_primitives", -1), ("source_primitives", True),
        ("source_primitives", 1.0), ("not_a_counter", 1),
    ):
        with pytest.raises(harness.ReproductionError) as error:
            ledger.consume(name, amount)
        assert error.value.code == "invalid_counter_consumption"
        assert ledger.counters == initial
    ledger.consume("source_primitives", 2_000)
    completed = dict(ledger.counters)
    assert completed["source_primitives"] == 2_000
    for name in ("source_primitives", "full_exports"):
        with pytest.raises(harness.ReproductionError) as error:
            ledger.consume(name)
        assert error.value.code == "counter_limit"
        assert ledger.counters == completed


def test_output_writes_are_exclusive_and_cannot_escape(harness, tmp_path):
    ledger, _, output = _ledger(harness, tmp_path, "exclusive-output")
    ledger.write_json("receipt.json", {"kind": "pure_boundary_test"})
    original = (output / "receipt.json").read_bytes()
    with pytest.raises(harness.ReproductionError) as error:
        ledger.write_json("receipt.json", {"kind": "replacement"})
    assert error.value.code == "existing_output_preserved"
    assert (output / "receipt.json").read_bytes() == original
    for label in ("../outside.json", str(tmp_path.resolve() / "absolute.json")):
        with pytest.raises(harness.ReproductionError) as error:
            ledger.write_json(label, {"kind": "refused"})
        assert error.value.code == "relative_file_label_required"
    assert not (tmp_path.resolve() / "outside.json").exists()
    assert not (tmp_path.resolve() / "absolute.json").exists()
    target = tmp_path.resolve() / "existing-target.json"
    target.write_bytes(b"preserved")
    link = output / "linked.json"
    link.symlink_to(target)
    with pytest.raises(harness.ReproductionError):
        ledger.write_json("linked.json", {"kind": "refused"})
    assert link.is_symlink() and target.read_bytes() == b"preserved"
    assert (output / "receipt.json").read_bytes() == original


def test_injected_deadlines_intersect_stage_and_parent_time(harness, tmp_path):
    admission_output = tmp_path.resolve() / "admission-deadline"
    admission_output.mkdir()
    admission_clock = [10.0]
    arm_requests = []
    admission = harness.BoundedLedger(
        admission_output, max_seconds=120, started=0.0,
        clock=lambda: admission_clock[0], rss_sampler=lambda: 0,
        arm=arm_requests.append,
    )
    with pytest.raises(harness.ReproductionError) as error:
        with admission.stage("admission"):
            assert arm_requests == [5.0]
            admission_clock[0] = 14.0
            admission.guard(force=True)
            admission_clock[0] = 15.0
            admission.guard(force=True)
    assert error.value.code == "cooperative_time_limit"

    parent, parent_clock, _ = _ledger(
        harness, tmp_path, "parent-deadline", max_seconds=2,
    )
    parent_clock[0] = 2.0
    with pytest.raises(harness.ReproductionError) as error:
        parent.guard(force=True)
    assert error.value.code == "cooperative_time_limit"

    stage, stage_clock, _ = _ledger(harness, tmp_path, "stage-deadline")
    with pytest.raises(harness.ReproductionError) as error:
        with stage.stage("reference"):
            stage_clock[0] = 59.0
            stage.guard(force=True)
            stage_clock[0] = 60.0
            stage.guard(force=True)
    assert error.value.code == "cooperative_time_limit"
    assert [event["kind"] for event in stage.events] == ["stage_start", "stage_failed"]

    intersected, clock, _ = _ledger(
        harness, tmp_path, "intersected-deadline", max_seconds=20,
    )
    clock[0] = 14.0
    with pytest.raises(harness.ReproductionError) as error:
        with intersected.stage("reference"):
            clock[0] = 19.0
            intersected.guard(force=True)
            clock[0] = 20.0
            intersected.guard(force=True)
    assert error.value.code == "cooperative_time_limit"
    assert [event["kind"] for event in intersected.events] == ["stage_start"]
