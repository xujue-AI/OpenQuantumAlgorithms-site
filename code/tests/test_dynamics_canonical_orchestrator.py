"""Bounded synthetic-child/control-flow checks; no canonical quantum execution."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dynamics_canonical_orchestrator_test", ROOT / "scripts/verify_dynamics_resources.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def replay_result():
    return {"schema": "openquantum-dynamics-canonical-qasm-replay-v1",
            "status": "canonical_qasm_replay_passed", "case_count": 24,
            "cases": [{"case_id": case, "test_scope": "synthetic control-flow fixture"} for case in v.gate.CASE_IDS],
            **{key: False for key in ("Native_estimates_run", "phase_fitter_run", "factory_search_repeated",
                                     "physical_budgets_repriced", "formal_certificate")}}


def expected_pin(records):
    return v.gate.sha256(records / v.gate.MANIFEST_NAME)


@pytest.fixture
def orchestrator_fixture(tmp_path, monkeypatch):
    records, source = tmp_path / "records", tmp_path / "source"
    records.mkdir(); source.mkdir()
    manifest = {"replay_sources": {}, "native_provenance": {"source_sha256": {}},
                "verifier_sources": {}, "frozen_files": {},
                "source_reproduction_evidence": {"path": "synthetic-report.json", "sha256": "f" * 64, "bytes": 2}}
    (records / v.gate.MANIFEST_NAME).write_text(json.dumps(manifest))
    identity = {"physical_validation": {"test_scope": "synthetic identity fixture"},
                "source_reproduction_validation": {"status": "source_not_byte_reproduced", "test_scope": "synthetic"}}
    monkeypatch.setattr(v.gate, "core_inventory", lambda records: {"test": {"bytes": 1, "sha256": "f" * 64}})
    monkeypatch.setattr(v.gate, "verify_dynamics_record_identity", lambda *args, **kwargs: copy.deepcopy(identity))
    monkeypatch.setattr(v, "locked_environment", lambda source: {"test_scope": "synthetic lock fixture"})
    return records, source, identity


def fake_child(result):
    def run(source, records, expected, workdir, timeout):
        (workdir / "canonical-replay.json").write_text(json.dumps(result))
        (workdir / "canonical.stdout").write_text('{"status":"canonical_qasm_replay_passed","cases":24,"native_estimates":0}\n')
        (workdir / "canonical.stderr").write_text("")
        return {"exit_status": 0, "Native_estimates_run": False, "fresh_source_generator_executed": False}
    return run


def test_canonical_route_keeps_source_difference_and_cost_roles_separate(orchestrator_fixture, tmp_path, monkeypatch):
    records, source, _ = orchestrator_fixture
    monkeypatch.setattr(v, "run_canonical_replay", fake_child(replay_result()))
    monkeypatch.setattr(v, "run_producer", lambda *a, **k: pytest.fail("fresh-source producer must not run"))
    capture = tmp_path / "capture"
    result = v.verify_canonical_record(records, records / v.gate.MANIFEST_NAME, expected_pin(records),
                                       source_root=source, capture_dir=capture, max_wall_seconds=10)
    assert result["canonical_qasm_replay_passed"] is True
    assert result["fresh_source_byte_reproduction_claimed"] is False
    assert result["fresh_source_generator_executed"] is False
    assert result["physical_budgets_repriced"] is False
    assert result["recorded_source_reproduction"]["status"] == "source_not_byte_reproduced"
    assert json.loads((capture / "receipt.json").read_text())["error"] is None


@pytest.mark.parametrize("mutation", ["certificate", "price", "missing-case", "extra-case", "foreign-case", "coverage"])
def test_canonical_result_cannot_claim_new_costing_or_incomplete_coverage(orchestrator_fixture, tmp_path, monkeypatch, mutation):
    records, source, _ = orchestrator_fixture
    result = replay_result()
    if mutation == "certificate": result["formal_certificate"] = True
    elif mutation == "price": result["physical_budgets_repriced"] = True
    elif mutation == "missing-case": result["cases"].pop()
    elif mutation == "extra-case": result["cases"].append(None)
    elif mutation == "foreign-case": result["cases"][0]["case_id"] = "unreviewed"
    else: result["case_count"] = 23
    monkeypatch.setattr(v, "run_canonical_replay", fake_child(result))
    capture = tmp_path / "capture"
    reason = "changed costing or a certificate" if mutation in ("certificate", "price") else (
        "coverage/schema" if mutation == "coverage" else "case roster")
    with pytest.raises(ValueError, match=reason):
        v.verify_canonical_record(records, records / v.gate.MANIFEST_NAME, expected_pin(records),
                                  source_root=source, capture_dir=capture, max_wall_seconds=10)
    receipt = json.loads((capture / "receipt.json").read_text())
    assert receipt["canonical_qasm_replay_passed"] is False and receipt["error"] is not None


def test_bad_initial_identity_prevents_canonical_child(orchestrator_fixture, monkeypatch):
    records, source, _ = orchestrator_fixture
    def reject(*args, **kwargs): raise ValueError("synthetic wrong manifest pin")
    monkeypatch.setattr(v.gate, "verify_dynamics_record_identity", reject)
    monkeypatch.setattr(v, "run_canonical_replay", lambda *a, **k: pytest.fail("child before identity"))
    with pytest.raises(ValueError, match="wrong manifest pin"):
        v.verify_canonical_record(records, records / v.gate.MANIFEST_NAME, expected_pin(records),
                                  source_root=source, max_wall_seconds=10)


def test_final_identity_failure_cannot_produce_success_receipt(orchestrator_fixture, tmp_path, monkeypatch):
    records, source, identity = orchestrator_fixture
    calls = []
    def verify(*args, **kwargs):
        calls.append(True)
        if len(calls) > 1: raise ValueError("synthetic evidence changed after replay")
        return copy.deepcopy(identity)
    monkeypatch.setattr(v.gate, "verify_dynamics_record_identity", verify)
    monkeypatch.setattr(v, "run_canonical_replay", fake_child(replay_result()))
    capture = tmp_path / "capture"
    with pytest.raises(ValueError, match="evidence changed"):
        v.verify_canonical_record(records, records / v.gate.MANIFEST_NAME, expected_pin(records),
                                  source_root=source, capture_dir=capture, max_wall_seconds=10)
    receipt = json.loads((capture / "receipt.json").read_text())
    assert receipt["canonical_qasm_replay_passed"] is False


def test_output_inventory_rejects_links_and_extra_files(tmp_path):
    (tmp_path / "canonical.stdout").write_text("")
    (tmp_path / "foreign").write_text("unreviewed")
    with pytest.raises(ValueError, match="unexpected canonical"):
        v.canonical_output_inventory(tmp_path)
    (tmp_path / "foreign").unlink()
    (tmp_path / "canonical.stderr").symlink_to(tmp_path / "canonical.stdout")
    with pytest.raises(ValueError, match="linked"):
        v.canonical_output_inventory(tmp_path)


def test_canonical_child_uses_finite_caps_and_original_cli_flags(tmp_path, monkeypatch):
    source, records, workdir = tmp_path / "source", tmp_path / "records", tmp_path / "work"
    (source / "scripts").mkdir(parents=True); records.mkdir(); workdir.mkdir()
    program = '''import argparse,json
p=argparse.ArgumentParser()
for name in ("records-dir","source-root","manifest-sha256","output-json"):p.add_argument("--"+name,required=True)
a=p.parse_args()
open(a.output_json,"x").write(json.dumps({"synthetic":True}))
print(json.dumps({"status":"canonical_qasm_replay_passed","cases":24,"native_estimates":0}))
'''
    (source / v.CANONICAL_REPLAY).write_text(program)
    result = v.run_canonical_replay(source, records, "a" * 64, workdir, 2)
    assert result["exit_status"] == 0 and result["Native_estimates_run"] is False
    assert result["argv"][:3] == [sys.executable, "-I", "-B"]
    assert result["argv"][result["argv"].index("--manifest-sha256") + 1] == "a" * 64
    assert result["rss_abort_threshold_bytes"] == 2 * 1024**3


def test_canonical_child_timeout_retains_partial_valid_json(tmp_path, monkeypatch):
    source, records, workdir, capture = tmp_path / "source", tmp_path / "records", tmp_path / "work", tmp_path / "capture"
    (source / "scripts").mkdir(parents=True); records.mkdir(); workdir.mkdir(); capture.mkdir()
    program = '''import argparse,json,time
p=argparse.ArgumentParser()
for name in ("records-dir","source-root","manifest-sha256","output-json"):p.add_argument("--"+name,required=True)
a=p.parse_args()
open(a.output_json,"x").write(json.dumps({"status":"synthetic incomplete"}))
while True:time.sleep(.01)
'''
    (source / v.CANONICAL_REPLAY).write_text(program)
    cleanup = v.cleanup_owned_group
    monkeypatch.setattr(v, "cleanup_owned_group", lambda child, tracked: cleanup(child, tracked, grace=.05))
    # Allow the synthetic child to start and write its partial result before
    # testing timeout retention, as in the neighboring subprocess fixtures.
    with pytest.raises(TimeoutError) as caught:
        v.run_canonical_replay(source, records, "a" * 64, workdir, 2)
    assert caught.value.replay_metrics["exit_status"] != 0
    v._retain_canonical(workdir, capture)
    assert json.loads((capture / "canonical-replay.json").read_text())["status"] == "synthetic incomplete"


@pytest.mark.parametrize("summary", [
    {"status": "canonical_qasm_replay_passed", "cases": 24.0, "native_estimates": 0},
    {"status": "canonical_qasm_replay_passed", "cases": 24, "native_estimates": False},
])
def test_canonical_child_summary_requires_exact_integer_types(tmp_path, summary):
    source, records, workdir = tmp_path / "source", tmp_path / "records", tmp_path / "work"
    (source / "scripts").mkdir(parents=True); records.mkdir(); workdir.mkdir()
    program = '''import argparse,json
p=argparse.ArgumentParser()
for name in ("records-dir","source-root","manifest-sha256","output-json"):p.add_argument("--"+name,required=True)
a=p.parse_args()
open(a.output_json,"x").write(json.dumps({"synthetic":True}))
print(SUMMARY)
'''.replace("SUMMARY", repr(json.dumps(summary)))
    (source / v.CANONICAL_REPLAY).write_text(program)
    with pytest.raises(ValueError, match="complete no-Native replay"):
        v.run_canonical_replay(source, records, "a" * 64, workdir, 2)
