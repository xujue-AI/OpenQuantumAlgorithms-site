"""Synthetic and analytic checks; these tests do not replay the canonical study."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dynamics_canonical_replay_test", ROOT / "scripts/dynamics_canonical_replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def parser():
    source = (ROOT / "scripts/tail_resource_qasm.py").read_bytes()
    return replay._verified_module(source, "_test_dynamics_parser")


def qasm(body, width=2):
    return (f'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[{width}] meas;\nqubit[{width}] q;\n'
            + body + "\n" + ";\n".join(f"meas[{wire}] = measure q[{wire}]" for wire in range(width)) + ";\n").encode()


def ledger(payload, counts, depth, *, case_id="synthetic-pf", n=2):
    return {
        "case_id": case_id, "method": "product_formula", "system_qubits": n,
        "logical_qubits": n, "J": 1, "h": .7, "hbar": 1, "time": .5,
        "order": 2, "initial_state": "0" * n, "observable": "Z(q0)",
        "task": {"algorithm_bias": .01}, "compiled_gate_counts": counts,
        "compiled_depth": depth, "compiled_global_phase_radians": 0.,
        "qasm_global_phase_scope": "Measured trial; overall exported phase is not a coherently controlled oracle specification",
        "qasm_sha256": hashlib.sha256(payload).hexdigest(),
        "analytic_error": {"operator_bound_upper_float": .005,
                           "numerical_lowering_guard_added": 1e-10},
    }


def synthetic_strang(steps=64):
    delta = .5 / steps
    layer = (f"cx q[0], q[1];\nrz({delta!r}) q[1];\ncx q[0], q[1];\n"
             f"rx({1.4 * delta!r}) q[0];\nrx({1.4 * delta!r}) q[1];\n"
             f"cx q[0], q[1];\nrz({delta!r}) q[1];\ncx q[0], q[1];\n")
    payload = qasm(layer * steps + "barrier q[0], q[1];")
    case = ledger(payload, {"cx": 4 * steps, "rz": 2 * steps, "rx": 2 * steps,
                            "barrier": 1, "measure": 2}, 7 * steps + 1)
    return payload, case


def test_module_import_defers_all_scientific_libraries():
    code = ("import pathlib,sys; "
            f"exec(compile(pathlib.Path({str(ROOT / 'scripts/dynamics_canonical_replay.py')!r}).read_bytes(),'helper','exec'),{{'__name__':'fixture'}}); "
            "assert not {'numpy','scipy','qiskit','qdk','openquantum'} & set(sys.modules)")
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_explicit_reference_has_correct_bit_order_and_open_chain():
    import numpy as np
    actual = replay.explicit_hamiltonian(2, 1, .7)
    expected = np.array([[1, .7, .7, 0], [.7, -1, 0, .7],
                         [.7, 0, -1, .7], [0, .7, .7, 1]], dtype=complex)
    assert np.array_equal(actual, expected)
    assert np.array_equal(replay.explicit_hamiltonian(1, 8, .3), np.array([[0, .3], [.3, 0]], dtype=complex))


@pytest.mark.parametrize("n", [0, 7, True])
def test_reference_rejects_size_before_science(n, monkeypatch):
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    with pytest.raises(ValueError, match="size cap"):
        replay.explicit_hamiltonian(n, 1, .7)


def test_stored_bell_stream_and_archived_phase_are_executed(parser):
    import numpy as np
    from qiskit.quantum_info import Statevector
    payload = qasm("h q[0];\ncx q[0], q[1];\nbarrier q[0], q[1];")
    case = ledger(payload, {"h": 1, "cx": 1, "barrier": 1, "measure": 2}, 3)
    case["compiled_global_phase_radians"] = math.pi / 2
    actual = Statevector.from_instruction(replay.parsed_circuit(parser.parse_qasm(payload), case)).data
    expected = 1j * np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    assert np.allclose(actual, expected, atol=1e-15, rtol=0)


def test_real_all_basis_projection_keeps_low_data_and_zero_high_ancillas(parser):
    import numpy as np
    angle = 2 * math.acos(.25)
    payload = qasm(f"x q[0];\nry({angle!r}) q[2];\nbarrier q[0], q[1], q[2], q[3], q[4], q[5];", width=6)
    case = ledger(payload, {"x": 1, "ry": 1, "barrier": 1, "measure": 6}, 2)
    case["logical_qubits"] = 6
    case["compiled_global_phase_radians"] = math.pi / 2
    circuit = replay.parsed_circuit(parser.parse_qasm(payload, max_qubits=9), case)
    actual, residual = replay._accepted_block(circuit, 2)
    x0 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    assert np.allclose(actual, 1j * .25 * x0, atol=1e-15, rtol=0)
    assert residual < 1e-14


@pytest.mark.parametrize("body,counts,depth,expected", [
    ("reset q[0];\nbarrier q[0], q[1];", {"reset": 1, "barrier": 1, "measure": 2}, 2, "opcode"),
    ("gate fixture a { h a; }\nfixture q[0];\nbarrier q[0], q[1];", {"h": 1, "barrier": 1, "measure": 2}, 2, "custom definitions"),
    ("meas[0] = measure q[0];\nh q[0];\nbarrier q[0], q[1];", {"h": 1, "barrier": 1, "measure": 3}, 3, "after final readout"),
])
def test_noncanonical_instruments_or_definitions_are_rejected_before_science(parser, monkeypatch, body, counts, depth, expected):
    payload = qasm(body)
    case = ledger(payload, counts, depth)
    parsed = parser.parse_qasm(payload)
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    with pytest.raises(ValueError, match=expected):
        replay.parsed_circuit(parsed, case)


def test_terminal_readout_mapping_is_exact(parser, monkeypatch):
    payload = qasm("barrier q[0], q[1];").replace(b"meas[0] = measure q[0]", b"meas[1] = measure q[0]")
    case = ledger(payload, {"barrier": 1, "measure": 2}, 1)
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    with pytest.raises(ValueError, match="mapping"):
        replay.parsed_circuit(parser.parse_qasm(payload), case)


def test_frozen_counts_are_exact(parser, monkeypatch):
    payload = qasm("h q[0];\nbarrier q[0], q[1];")
    case = ledger(payload, {"h": 2, "barrier": 1, "measure": 2}, 2)
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    with pytest.raises(ValueError, match="counts"):
        replay.parsed_circuit(parser.parse_qasm(payload), case)


def test_frozen_dependency_depth_is_exact(parser):
    payload = qasm("h q[0];\nbarrier q[0], q[1];")
    case = ledger(payload, {"h": 1, "barrier": 1, "measure": 2}, 99)
    with pytest.raises(ValueError, match="dependency depth"):
        replay.parsed_circuit(parser.parse_qasm(payload), case)


def test_synthetic_strang_checks_full_operator_and_observable_without_repricing(parser):
    payload, case = synthetic_strang()
    before = copy.deepcopy(case)
    result = replay.replay_canonical_qasm({case["case_id"]: payload}, [case], parser)
    diagnostic = result["cases"][0]["diagnostics"]
    assert result["case_count"] == 1
    assert 0 < diagnostic["operator_error"] < .005
    assert diagnostic["observable_error"] < .01
    assert result["Native_estimates_run"] is False
    assert result["physical_budgets_repriced"] is False
    assert case == before


def test_changed_finite_rotation_is_checked_against_task_not_historical_floats(parser):
    payload, case = synthetic_strang()
    bad = payload.replace(b"rx(0.0109375)", b"rx(0.0)")
    assert bad != payload
    case["qasm_sha256"] = hashlib.sha256(bad).hexdigest()
    with pytest.raises(ValueError, match="frozen operator allowance"):
        replay.replay_canonical_qasm({case["case_id"]: bad}, [case], parser)


def test_snapshot_hash_rejection_precedes_parser_or_science(monkeypatch):
    payload, case = synthetic_strang()
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    parser = types.SimpleNamespace(parse_qasm=lambda *a, **k: pytest.fail("parser must not start"))
    with pytest.raises(ValueError, match="snapshot differs"):
        replay.replay_canonical_qasm({case["case_id"]: payload + b"\n"}, [case], parser)


@pytest.mark.parametrize("n,width", [(2, 6), (4, 9)])
def test_frozen_retargeted_parameter_width_without_canonical_execution(monkeypatch, n, width):
    path = ROOT / f"assets/generated/dynamics-resources/qsp-retargeted-n{n}-t1-eps0.05/logical-resources.json"
    case = json.loads(path.read_bytes())
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    assert case["logical_qubits"] == width
    assert replay._case_parameters(case) == (n, 1., .01)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_global_phase_is_rejected(parser, value):
    payload = qasm("barrier q[0], q[1];")
    case = ledger(payload, {"barrier": 1, "measure": 2}, 1)
    case["compiled_global_phase_radians"] = value
    with pytest.raises(ValueError, match="global phase must be finite"):
        replay.parsed_circuit(parser.parse_qasm(payload), case)


def qsp_fixture():
    floor = .0625 - 1e-9
    return {
        "case_id": "synthetic-qsp", "method": "QSP_retargeted", "system_qubits": 2,
        "rounds": 0, "task": {"algorithm_bias": .01},
        "numerical_checks": {"numerical_lowering_guard_added": 1e-10,
                             "acceptance_guard_subtracted": 1e-9},
        "postselection_contract": {
            "reference_acceptance_lower": floor,
            "synthesis_model_error_per_attempt": .005 * floor,
            "attempts_per_accepted_slot": 257, "accepted_slots": 13811,
            "maximum_complete_attempts": 257 * 13811,
            "physical_failure_allowance_per_complete_attempt": .005 / (257 * 13811),
        },
    }


def test_acceptance_spends_only_half_existing_margin_and_preserves_costs(monkeypatch):
    import numpy as np
    case = qsp_fixture()
    before = copy.deepcopy(case)
    target = np.eye(4, dtype=complex)
    monkeypatch.setattr(replay, "_accepted_block", lambda *a: (.25 * target, 0.))
    result, _ = replay._qsp_result(case, object(), target, np.array([1, -1, 1, -1]))
    support = result["acceptance_numerical_support"]
    assert support["reserved_half_guard"] == 5e-10
    assert support["spare_margin_after_reserved_half_guard"] > 0
    assert support["proven_floating_evaluation_error_bound"] is False
    assert support["certified_physical_confidence"] is False
    assert result["frozen_synthesis_model_error_per_attempt"] == before["postselection_contract"]["synthesis_model_error_per_attempt"]
    assert result["frozen_maximum_complete_attempts"] == 3549427
    assert case == before


def test_acceptance_with_insufficient_spare_margin_fails_without_new_floor(monkeypatch):
    import numpy as np
    case = qsp_fixture()
    case["postselection_contract"]["reference_acceptance_lower"] = .0625 - 2e-10
    before = copy.deepcopy(case)
    target = np.eye(4, dtype=complex)
    monkeypatch.setattr(replay, "_accepted_block", lambda *a: (.25 * target, 0.))
    with pytest.raises(ValueError, match="support the frozen floor"):
        replay._qsp_result(case, object(), target, np.array([1, -1, 1, -1]))
    assert case == before


@pytest.mark.parametrize("guard", [0., 5e-10, 2e-9])
def test_archived_acceptance_reserve_cannot_be_replaced(monkeypatch, guard):
    import numpy as np
    case = qsp_fixture()
    case["numerical_checks"]["acceptance_guard_subtracted"] = guard
    target = np.eye(4, dtype=complex)
    monkeypatch.setattr(replay, "_accepted_block", lambda *a: (.25 * target, 0.))
    with pytest.raises(ValueError, match="archived acceptance reserve"):
        replay._qsp_result(case, object(), target, np.array([1, -1, 1, -1]))


def test_oaa_recurrence_has_exact_positive_scaled_amplitudes():
    import numpy as np
    blocks = {rounds: float(value) * np.eye(4, dtype=complex) for rounds, value in enumerate(replay.OAA_AMPLITUDES)}
    result = replay._oaa_polynomial_checks(blocks, {r: str(r) for r in range(4)})
    assert all(value["canonical_OAA_polynomial_error"] == 0 for value in result.values())
    assert tuple(map(str, replay.OAA_AMPLITUDES)) == ("1/4", "11/16", "61/64", "251/256")
    blocks[3][0, 0] += 2e-10
    with pytest.raises(ValueError, match="polynomial guard"):
        replay._oaa_polynomial_checks(blocks, {r: str(r) for r in range(4)})


def test_oaa_recurrence_requires_complete_slice():
    with pytest.raises(ValueError, match="complete four-round"):
        replay._oaa_polynomial_checks({0: object()}, {0: "base"})


@pytest.mark.parametrize("prior", [None, types.ModuleType("poisoned")])
def test_source_snapshot_loading_restores_cache_and_does_not_trust_it(prior):
    name = "_canonical_fixture_cache"
    sys.modules[name] = prior
    try:
        actual = replay._verified_module(b"VALUE=7\n", name)
        assert actual.VALUE == 7
        assert sys.modules[name] is prior
    finally:
        sys.modules.pop(name, None)


def test_source_snapshot_compile_failure_restores_cache():
    name = "_canonical_fixture_failure"
    prior = types.ModuleType(name)
    sys.modules[name] = prior
    try:
        with pytest.raises(SyntaxError):
            replay._verified_module(b"this is ) invalid", name)
        assert sys.modules[name] is prior
    finally:
        sys.modules.pop(name, None)


def test_identity_snapshot_rejects_fifo_without_blocking(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    os.mkfifo(folder / "dynamics_record_release.py")
    with pytest.raises(ValueError, match="bounded/regular"):
        replay._identity_snapshot(tmp_path)


def test_result_write_is_anchored_to_checked_parent_and_exclusive(tmp_path):
    source = tmp_path / "source"
    records = source / "records"
    records.mkdir(parents=True)
    parent = tmp_path / "output"
    parent.mkdir()
    output = parent / "result.json"
    with replay._pinned_output_parent(output, records, source) as descriptor:
        replay._write_anchored_result(output, descriptor, b"{}\n")
        with pytest.raises(FileExistsError):
            replay._write_anchored_result(output, descriptor, b"changed\n")
    assert output.read_bytes() == b"{}\n"
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize("swap", ["parent", "ancestor"])
def test_swapped_result_parent_never_redirects_into_records(tmp_path, swap):
    source = tmp_path / "source"
    records = source / "records"
    records.mkdir(parents=True)
    top = tmp_path / "top"
    parent = top / "output"
    parent.mkdir(parents=True)
    output = parent / "result.json"
    with replay._pinned_output_parent(output, records, source) as descriptor:
        if swap == "parent":
            parent.rename(tmp_path / "parked")
            parent.symlink_to(records, target_is_directory=True)
        else:
            (records / "output").mkdir()
            top.rename(tmp_path / "parked")
            top.symlink_to(records, target_is_directory=True)
        with pytest.raises(ValueError, match="parent path changed"):
            replay._write_anchored_result(output, descriptor, b"{}\n")
    assert not list(records.rglob("result.json"))
    assert not list((tmp_path / "parked").rglob("result.json"))
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_result_parent_closes_after_replay_error(tmp_path):
    source = tmp_path / "source"
    records = source / "records"
    records.mkdir(parents=True)
    parent = tmp_path / "output"
    parent.mkdir()
    with pytest.raises(RuntimeError, match="fixture interruption"):
        with replay._pinned_output_parent(parent / "result.json", records, source) as descriptor:
            raise RuntimeError("fixture interruption")
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_wall_signal_aborts_and_restores_timer():
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match="wall cap"):
        with replay._wall_limit():
            signal.raise_signal(signal.SIGALRM)
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert signal.getsignal(signal.SIGALRM) is before


def test_main_identity_rejection_occurs_before_parser_or_science(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "_scientific", lambda: pytest.fail("science must not start"))
    original = replay._verified_module
    def checked(payload, name, filename="fixture"):
        assert name == "_trusted_dynamics_identity_source", "parser must not compile before identity"
        return original(payload, name, filename)
    monkeypatch.setattr(replay, "_verified_module", checked)
    with pytest.raises(ValueError, match="JSON snapshot identity differs"):
        replay.main(["--records-dir", str(ROOT / "assets/generated/dynamics-resources"),
                     "--source-root", str(ROOT), "--manifest-sha256", "0" * 64,
                     "--output-json", str(tmp_path / "result.json")])
    assert not (tmp_path / "result.json").exists()
