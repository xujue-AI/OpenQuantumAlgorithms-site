"""Immutable temporary-bundle replay and fail-closed preflight checks."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "optimization_diagnostics_checkpoint_verifier", ROOT / "scripts/verify_optimization_diagnostics.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

FIXTURE_PRODUCER = (
    "import argparse,json,pathlib\n"
    "parser=argparse.ArgumentParser()\n"
    "parser.add_argument('--resume',action='store_true')\n"
    "parser.add_argument('--output-dir',type=pathlib.Path,required=True)\n"
    "args=parser.parse_args()\n"
    "assert args.resume, 'numerical training must never be requested'\n"
    "assert args.output_dir == pathlib.Path(__file__).parent\n"
    "assert (args.output_dir/'raw-study.json').is_file()\n"
    "runs=args.output_dir/'runs.json'\n"
    "data=json.loads(runs.read_text())\n"
    "data['artifact_generation_wall_seconds']=123.5\n"
    "runs.write_text(json.dumps(data))\n"
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_checkpoint(tmp_path, monkeypatch, producer_text=FIXTURE_PRODUCER):
    directory = tmp_path / "archive"
    directory.mkdir()
    for name in verifier.TOP_LEVEL_FILES:
        (directory / name).write_text("fixture preserved bytes\n")
    (directory / verifier.PRODUCER).write_text(producer_text)
    (directory / verifier.IMPLEMENTATION).write_text("# matching immutable fixture implementation\n")
    recipe = {"cases": [{"case_id": "fixture-case"}], "seeds": [7]}
    write_json(directory / "recipe.json", recipe)
    identity = {key: verifier.sha256(directory / name) for key, name in (
        ("recipe_sha256", "recipe.json"), ("implementation_sha256", verifier.IMPLEMENTATION),
        ("generator_sha256", verifier.PRODUCER), ("environment_lock_sha256", "requirements-lock.txt"))}
    freeze = {"status": "recipe_and_producers_frozen_before_execution", "identity": identity}
    write_json(directory / "freeze.json", freeze)
    timing = {"status": "complete", "identity": identity, "total_numerical_wall_seconds": 3.5}
    write_json(directory / "numerical-timing.json", timing)
    provenance = {**identity, "python": platform.python_version(), "packages": {},
                  "freeze_sha256": verifier.sha256(directory / "freeze.json")}
    sampled = [{"run_id": f"fixture-case-{start}-seed-7", "start_id": start,
                "seed": 7, "count": 5} for start in ("original", "alternate")]
    controls = [{"run_id": f"fixture-case-{start}-{mode}", "start_id": start,
                 "mode": mode, "quantum_preparations": 0}
                for start in ("original", "alternate")
                for mode in ("exact-objective", "expected-margin")]
    raw = {"recipe": recipe, "checkpoint_identity": identity, "provenance": provenance,
           "execution_status": "completed_bounded_ideal_sampling_and_exact_cpu_controls",
           "cases": [{"case_id": "fixture-case", "runs": sampled}], "controls": controls,
           "numerical_timing_sha256": verifier.sha256(directory / "numerical-timing.json")}
    write_json(directory / "raw-study.json", raw)
    runs = {**deepcopy(raw), "artifact_generation_wall_seconds": 1.5,
            "original_cpu_seconds": 3.0,
            "nested": {"artifact_generation_wall_seconds": .2}}
    write_json(directory / "runs.json", runs)
    (directory / "checkpoints").mkdir()
    for kind, records in (("sampled", sampled), ("control", controls)):
        for run in records:
            write_json(directory / "checkpoints" / f"{kind}-{run['run_id']}.json", {
                "schema_version": 1, "kind": kind, "key": run["run_id"], "identity": identity,
                "status": "completed", "run": run,
            })
    (directory / "circuits").mkdir()
    circuits = []
    for label in ("a", "b"):
        data = f"OPENQASM 3.0; // {label}\n".encode()
        digest = verifier.hashlib.sha256(data).hexdigest()
        name = f"circuits/{digest}.qasm"
        (directory / name).write_bytes(data)
        circuits.append({"file": name, "sha256": digest})
    write_json(directory / "circuit-manifest.json", {
        "unique_circuits": circuits, "unique_circuit_count": len(circuits)})
    monkeypatch.setattr(verifier, "EXPECTED_FILE_HASHES", {
        name: verifier.sha256(directory / name) for name in verifier.EXPECTED_FILE_HASHES})
    monkeypatch.setattr(verifier, "EXPECTED_QASM_COUNT", len(circuits))
    return directory


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    return make_checkpoint(tmp_path, monkeypatch)


def replay_copy(checkpoint, tmp_path):
    replay = tmp_path / "replay"
    shutil.copytree(checkpoint, replay)
    return replay


def test_full_preflight_requires_every_prescribed_completed_record(checkpoint):
    result = verifier.preflight(checkpoint, verifier.inventory(checkpoint))
    assert result["sampled_runs"] == 2
    assert result["controls"] == 4
    assert result["qasm_files"] == 2


def test_preserved_producer_runs_only_in_a_temporary_copy(checkpoint, monkeypatch, tmp_path):
    before = verifier.inventory(checkpoint)
    # An unrelated evolving live source must never select the replay producer.
    live = tmp_path / "live-source.py"
    live.write_text("raise AssertionError('live source must not execute')\n")
    real_run = verifier.run_archived_producer
    calls = []
    def spy(python, producer, output, workdir):
        assert producer.parent == output
        assert not output.resolve().is_relative_to(checkpoint.resolve())
        assert verifier.sha256(producer) == before["file_sha256"][verifier.PRODUCER]
        calls.append((producer, output, workdir))
        real_run(python, producer, output, workdir)
    monkeypatch.setattr(verifier, "run_archived_producer", spy)
    result = verifier.verify_checkpoint(checkpoint, python=Path(sys.executable))
    assert result["training_repeated"] is False
    assert result["original_inventory_preserved"] is True
    assert result["sampled_runs"] == 2 and result["controls"] == 4
    assert len(calls) == 1 and not calls[0][2].exists()
    assert verifier.inventory(checkpoint) == before


@pytest.mark.parametrize("name", ["raw-study.json", "freeze.json", "numerical-timing.json",
                                 "checkpoints/sampled-fixture-case-original-seed-7.json",
                                 "checkpoints/control-fixture-case-original-exact-objective.json"])
def test_missing_records_fail_before_any_producer(checkpoint, monkeypatch, name):
    (checkpoint / name).unlink()
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("no producer allowed"))
    with pytest.raises(ValueError, match="Missing|incomplete"):
        verifier.verify_checkpoint(checkpoint)


@pytest.mark.parametrize("field,value", [("status", "pending"), ("kind", "sampled"),
                                       ("identity", {}), ("key", "different")])
def test_incomplete_or_rebound_checkpoint_fails_before_replay(checkpoint, monkeypatch, field, value):
    path = checkpoint / "checkpoints/control-fixture-case-original-exact-objective.json"
    data = verifier.load_json(path)
    data[field] = value
    write_json(path, data)
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("no producer allowed"))
    with pytest.raises(ValueError, match="changed or incomplete numerical checkpoint"):
        verifier.verify_checkpoint(checkpoint)


@pytest.mark.parametrize("name", [verifier.PRODUCER, verifier.IMPLEMENTATION,
                                 "requirements-lock.txt", "recipe.json", "LICENSE.txt"])
def test_changed_frozen_sources_fail_before_replay(checkpoint, monkeypatch, name):
    with (checkpoint / name).open("ab") as stream:
        stream.write(b" ")
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("no producer allowed"))
    with pytest.raises(ValueError, match="changed frozen diagnostics file"):
        verifier.verify_checkpoint(checkpoint)


def test_unexpected_checkpoint_is_not_treated_as_a_partial_resume(checkpoint, monkeypatch):
    write_json(checkpoint / "checkpoints/extra.json", {"status": "completed"})
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("no producer allowed"))
    with pytest.raises(ValueError, match="inventory is incomplete or unexpected"):
        verifier.verify_checkpoint(checkpoint)


def test_only_top_level_export_wall_time_may_differ(checkpoint, tmp_path):
    replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    runs["artifact_generation_wall_seconds"] = 999.25
    write_json(replay / "runs.json", runs)
    result = verifier.compare_exports(checkpoint, replay)
    assert result["byte_compared_files"] == len(verifier.inventory(checkpoint)["file_sha256"]) - 1
    assert result["runs_json_ignored_top_level_fields"] == ["artifact_generation_wall_seconds"]


@pytest.mark.parametrize("path,value", [
    (("original_cpu_seconds",), 3.1),
    (("provenance", "python"), "3.12.99"),
    (("cases", 0, "runs", 0, "count"), 6),
    (("cases", 0, "runs", 0, "count"), 5.0),
    (("nested", "artifact_generation_wall_seconds"), .3),
])
def test_other_numerical_provenance_or_timing_changes_fail(checkpoint, tmp_path, path, value):
    replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    current = runs
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    write_json(replay / "runs.json", runs)
    with pytest.raises(ValueError, match="differs beyond"):
        verifier.compare_exports(checkpoint, replay)


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_ignored_wall_time_still_requires_finite_nonnegative_scalar(checkpoint, tmp_path, value):
    replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    runs["artifact_generation_wall_seconds"] = value
    write_json(replay / "runs.json", runs)
    with pytest.raises(ValueError):
        verifier.compare_exports(checkpoint, replay)


@pytest.mark.parametrize("name", sorted(verifier.TOP_LEVEL_FILES - {"runs.json"}))
def test_all_other_export_files_remain_byte_exact(checkpoint, tmp_path, name):
    replay = replay_copy(checkpoint, tmp_path)
    with (replay / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="artifact bytes differ"):
        verifier.compare_exports(checkpoint, replay)


@pytest.mark.parametrize("change", ["missing", "extra", "bytes"])
def test_complete_qasm_inventory_is_preserved(checkpoint, tmp_path, change):
    replay = replay_copy(checkpoint, tmp_path)
    path = next((replay / "circuits").glob("*.qasm"))
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (replay / "circuits/extra.qasm").write_text("unreviewed\n")
    else:
        path.write_text("changed\n")
    with pytest.raises(ValueError, match="inventory differs|artifact bytes differ"):
        verifier.compare_exports(checkpoint, replay)


@pytest.mark.parametrize("cap", ["files", "entries", "file_bytes", "total_bytes"])
def test_input_caps_fail_before_hashing_unbounded_data(checkpoint, monkeypatch, cap):
    monkeypatch.setitem(verifier.CAPS, cap, 1)
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("no producer allowed"))
    with pytest.raises(ValueError, match="cap"):
        verifier.verify_checkpoint(checkpoint)


@pytest.mark.parametrize("kind", ["root", "file", "directory"])
def test_symlinked_input_is_rejected(checkpoint, tmp_path, kind):
    target = tmp_path / "foreign"
    target.mkdir()
    if kind == "root":
        link = tmp_path / "archive-link"
        link.symlink_to(checkpoint, target_is_directory=True)
        checkpoint = link
    elif kind == "file":
        (checkpoint / "linked.json").symlink_to(target / "data.json")
    else:
        (checkpoint / "linked-directory").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory|Linked or special"):
        verifier.inventory(checkpoint)


def test_failure_cleans_temporary_copy_and_preserves_original(checkpoint, monkeypatch):
    before, paths = verifier.inventory(checkpoint), []
    def fail(python, producer, output, workdir):
        paths.append(workdir)
        raise RuntimeError("bounded replay failure")
    monkeypatch.setattr(verifier, "run_archived_producer", fail)
    with pytest.raises(RuntimeError, match="bounded replay failure"):
        verifier.verify_checkpoint(checkpoint)
    assert paths and not paths[0].exists()
    assert verifier.inventory(checkpoint) == before


def test_original_mutation_is_reported_even_after_replay_failure(checkpoint, monkeypatch):
    def fail(python, producer, output, workdir):
        next((checkpoint / "circuits").glob("*.qasm")).write_text("changed original\n")
        raise RuntimeError("replay failed")
    monkeypatch.setattr(verifier, "run_archived_producer", fail)
    with pytest.raises(ValueError, match="Original diagnostics inventory"):
        verifier.verify_checkpoint(checkpoint)


def test_timeout_reaps_child_and_removes_temporary_directory(tmp_path, monkeypatch):
    producer = FIXTURE_PRODUCER + "import time\ntime.sleep(10)\n"
    checkpoint = make_checkpoint(tmp_path, monkeypatch, producer)
    before, processes, directories = verifier.inventory(checkpoint), [], []
    real_popen = verifier.subprocess.Popen
    def track(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        if verifier.PRODUCER in str(args[0]):
            processes.append(process)
            directories.append(Path(kwargs["cwd"]))
        return process
    monkeypatch.setattr(verifier.subprocess, "Popen", track)
    monkeypatch.setitem(verifier.CAPS, "replay_seconds", .1)
    with pytest.raises(subprocess.TimeoutExpired):
        verifier.verify_checkpoint(checkpoint)
    assert processes and all(process.poll() is not None for process in processes)
    assert directories and all(not directory.exists() for directory in directories)
    assert verifier.inventory(checkpoint) == before


@pytest.mark.skipif(os.name != "posix" or shutil.which("ps") is None,
                    reason="Owned process-group cleanup is a POSIX publication check")
def test_timeout_kills_pipe_holding_descendant_after_direct_child_exits(tmp_path, monkeypatch):
    marker = tmp_path / "descendant.pid"
    producer = FIXTURE_PRODUCER + (
        "import subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
    )
    checkpoint = make_checkpoint(tmp_path, monkeypatch, producer)
    before = verifier.inventory(checkpoint)
    monkeypatch.setitem(verifier.CAPS, "replay_seconds", .3)
    with pytest.raises(subprocess.TimeoutExpired):
        verifier.verify_checkpoint(checkpoint)
    assert marker.is_file()
    state = subprocess.run(["ps", "-o", "stat=", "-p", marker.read_text()],
                           check=False, capture_output=True, text=True, timeout=5).stdout.strip()
    assert not state or state.startswith("Z"), "replay descendant must not remain running"
    assert verifier.inventory(checkpoint) == before


def test_exact_python_patch_is_required_by_the_frozen_producer(checkpoint, monkeypatch):
    monkeypatch.setattr(verifier._COMMON, "verify_locked_interpreter", lambda *args:
                        {"python": "3.12.99", "packages": {}})
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("no producer allowed"))
    with pytest.raises(ValueError, match="recorded Python patch version"):
        verifier.verify_checkpoint(checkpoint)
