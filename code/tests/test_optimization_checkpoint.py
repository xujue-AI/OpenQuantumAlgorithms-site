"""Fail-closed replay comparison, with immutable temporary checkpoint fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("optimization_checkpoint_verifier",
                                           ROOT / "scripts/verify_optimization_checkpoint.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    directory = tmp_path / "archive"
    directory.mkdir()
    producer = directory / verifier.PRODUCER
    producer.write_text(
        "import argparse,json,pathlib,shutil\n"
        "parser=argparse.ArgumentParser()\n"
        "parser.add_argument('--resume',action='store_true')\n"
        "parser.add_argument('--output-dir',type=pathlib.Path,required=True)\n"
        "args=parser.parse_args()\n"
        "assert args.resume, 'training must never be requested'\n"
        "archive=pathlib.Path(__file__).parent\n"
        "for source in archive.rglob('*'):\n"
        " if source.is_file():\n"
        "  dest=args.output_dir/source.relative_to(archive)\n"
        "  dest.parent.mkdir(parents=True,exist_ok=True)\n"
        "  shutil.copyfile(source,dest)\n"
        "runs=args.output_dir/'runs.json'\n"
        "data=json.loads(runs.read_text())\n"
        "data['artifact_generation_wall_seconds']=123.5\n"
        "runs.write_text(json.dumps(data))\n"
    )
    (directory / verifier.IMPLEMENTATION).write_text("# immutable fixture implementation\n")
    (directory / "LICENSE.txt").write_text("fixture license\n")
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("exact fixture lock\n")
    recipe = {"fixed_published_recipe": True, "seeds": [7, 19, 41]}
    provenance = {"generator_sha256": verifier.sha256(producer),
                  "implementation_sha256": verifier.sha256(directory / verifier.IMPLEMENTATION),
                  "environment_lock_sha256": verifier.sha256(lock),
                  "python": platform.python_version(), "packages": {},
                  "generated_at_utc": "2026-11-12T13:20:28Z", "source_commit": None}
    raw = {"provenance": provenance, "recipe": recipe}
    write_json(directory / "raw-study.json", raw)
    write_json(directory / "recipe.json", recipe)
    runs = {"provenance": deepcopy(provenance), "artifact_generation_wall_seconds": 1.5,
            "artifact_timing_scope": "export timing outside original CPU study",
            "workflow_wall_seconds": 4.704195, "workflow_process_cpu_seconds": 4.701387,
            "hardware_feedback_seconds": None,
            "data": [{"seed": 7, "counts": {"01": 5}, "evaluation_wall_seconds": .004,
                      "nested": {"artifact_generation_wall_seconds": .02}}]}
    write_json(directory / "runs.json", runs)
    write_json(directory / "resources.json", {"total_shots": 25, "provenance": provenance})
    write_json(directory / "circuit-manifest.json", {"unique_circuit_count": 2})
    for name in ("results.csv", "resources.csv", "groups.csv", "evaluations.csv"):
        (directory / name).write_text("seed,count,timing\n7,5,0.004\n")
    (directory / "circuits").mkdir()
    for name in ("a.qasm", "b.qasm"):
        (directory / "circuits" / name).write_text(f"OPENQASM 3.0; // {name}\n")
    monkeypatch.setattr(verifier, "EXPECTED_SOURCE_HASHES", {key: provenance[key] for key in verifier.EXPECTED_SOURCE_HASHES})
    monkeypatch.setattr(verifier, "EXPECTED_QASM_COUNT", 2)
    return directory, lock


def replay_copy(checkpoint, tmp_path):
    directory, _ = checkpoint
    replay = tmp_path / "replay"
    shutil.copytree(directory, replay)
    return directory, replay


def test_only_top_level_rebuild_wall_time_may_change(checkpoint, tmp_path):
    directory, replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    runs[verifier.REBUILD_TIMING_FIELD] = 999.75
    write_json(replay / "runs.json", runs)
    before = (directory / "runs.json").read_bytes()
    result = verifier.compare_exports(directory, replay)
    assert result["qasm_files_compared"] == 2
    assert result["runs_json_ignored_top_level_fields"] == ["artifact_generation_wall_seconds"]
    assert (directory / "runs.json").read_bytes() == before


@pytest.mark.parametrize("path,value", [
    (("workflow_wall_seconds",), 4.7042),
    (("workflow_process_cpu_seconds",), 4.7014),
    (("hardware_feedback_seconds",), 0),
    (("artifact_timing_scope",), "changed timing scope"),
    (("provenance", "source_commit"), "changed"),
    (("provenance", "generated_at_utc"), "2026-11-13"),
    (("provenance", "implementation_sha256"), "0"*64),
    (("data", 0, "seed"), 19),
    (("data", 0, "seed"), 7.0),
    (("data", 0, "counts", "01"), 6),
    (("data", 0, "evaluation_wall_seconds"), .005),
    (("data", 0, "nested", "artifact_generation_wall_seconds"), .03),
])
def test_runs_comparator_rejects_scientific_provenance_and_original_timing_changes(checkpoint, tmp_path, path, value):
    directory, replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    parent = runs
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    write_json(replay / "runs.json", runs)
    with pytest.raises(ValueError, match="differs beyond"):
        verifier.compare_runs_json(directory / "runs.json", replay / "runs.json")


@pytest.mark.parametrize("value", [None, True, -1, float("nan"), float("inf")])
def test_ignored_rebuild_field_still_requires_a_valid_wall_time(checkpoint, tmp_path, value):
    directory, replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    runs[verifier.REBUILD_TIMING_FIELD] = value
    write_json(replay / "runs.json", runs)
    with pytest.raises(ValueError):
        verifier.compare_runs_json(directory / "runs.json", replay / "runs.json")


def test_missing_rebuild_field_is_rejected(checkpoint, tmp_path):
    directory, replay = replay_copy(checkpoint, tmp_path)
    runs = verifier.load_json(replay / "runs.json")
    del runs[verifier.REBUILD_TIMING_FIELD]
    write_json(replay / "runs.json", runs)
    with pytest.raises(ValueError, match="top-level rebuild wall time"):
        verifier.compare_runs_json(directory / "runs.json", replay / "runs.json")


@pytest.mark.parametrize("text", [
    '{"artifact_generation_wall_seconds":1,"data":{"seed":7,"seed":19}}',
    '{"artifact_generation_wall_seconds":1,"data":{"timing":NaN}}',
    '{"artifact_generation_wall_seconds":1,"data":{"timing":1e999}}',
])
def test_ambiguous_or_nonfinite_json_is_rejected(checkpoint, tmp_path, text):
    directory, replay = replay_copy(checkpoint, tmp_path)
    (replay / "runs.json").write_text(text)
    with pytest.raises(ValueError):
        verifier.compare_runs_json(directory / "runs.json", replay / "runs.json")


@pytest.mark.parametrize("name", verifier.BYTE_COMPARED_FILES)
def test_every_required_export_is_compared_byte_for_byte(checkpoint, tmp_path, name):
    directory, replay = replay_copy(checkpoint, tmp_path)
    with (replay / name).open("ab") as stream:
        stream.write(b" ")  # Even insignificant JSON/CSV formatting must match.
    with pytest.raises(ValueError, match="artifact bytes differ"):
        verifier.compare_exports(directory, replay)


@pytest.mark.parametrize("change", ["missing", "extra", "bytes"])
def test_complete_qasm_inventory_and_bytes_are_required(checkpoint, tmp_path, change):
    directory, replay = replay_copy(checkpoint, tmp_path)
    if change == "missing":
        (replay / "circuits/a.qasm").unlink()
    elif change == "extra":
        (replay / "circuits/c.qasm").write_text("OPENQASM 3.0;\n")
    else:
        (replay / "circuits/a.qasm").write_text("OPENQASM 3.0; // altered\n")
    with pytest.raises(ValueError, match="QASM inventory|artifact bytes differ"):
        verifier.compare_exports(directory, replay)


@pytest.mark.parametrize("name", [verifier.PRODUCER, verifier.IMPLEMENTATION])
def test_source_changes_fail_before_replay(checkpoint, monkeypatch, name):
    directory, lock = checkpoint
    (directory / name).write_text("changed source\n")
    monkeypatch.setattr(verifier, "run_archived_producer", lambda *args: pytest.fail("producer should not run"))
    with pytest.raises(ValueError, match="checkpoint binding"):
        verifier.verify_checkpoint(directory, python=Path(sys.executable), lock_file=lock)


def test_rehashed_source_cannot_retune_the_frozen_checkpoint(checkpoint):
    directory, _ = checkpoint
    raw = verifier.load_json(directory / "raw-study.json")
    (directory / verifier.PRODUCER).write_text("changed source\n")
    raw["provenance"]["generator_sha256"] = verifier.sha256(directory / verifier.PRODUCER)
    with pytest.raises(ValueError, match="frozen source binding"):
        verifier.verify_source_bindings(directory, raw)


def test_recipe_changes_are_not_normalized(checkpoint):
    directory, _ = checkpoint
    raw = verifier.load_json(directory / "raw-study.json")
    write_json(directory / "recipe.json", {"fixed_published_recipe": True, "seeds": [19, 7, 41]})
    with pytest.raises(ValueError, match="Recipe differs"):
        verifier.verify_source_bindings(directory, raw)


def test_lock_and_runtime_must_match(checkpoint):
    directory, lock = checkpoint
    raw = verifier.load_json(directory / "raw-study.json")
    raw["provenance"]["python"] = "0.0.0"
    with pytest.raises(ValueError, match="locked checkpoint environment"):
        verifier.verify_locked_interpreter(Path(sys.executable), lock, raw)
    lock.write_text("changed lock")
    with pytest.raises(ValueError, match="Environment lock differs"):
        verifier.verify_locked_interpreter(Path(sys.executable), lock, raw)


@pytest.mark.parametrize("version", ["3.12.0", "3.12.14", "3.12.15", "3.12.99"])
def test_python_312_patch_releases_are_allowed_and_reported_without_rewriting_provenance(checkpoint, monkeypatch, version):
    directory, lock = checkpoint
    raw = verifier.load_json(directory / "raw-study.json")
    original_version = raw["provenance"]["python"]
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"python": version, "packages": {}})))
    actual = verifier.verify_locked_interpreter(Path(sys.executable), lock, raw)
    assert actual["python"] == version
    assert raw["provenance"]["python"] == original_version


@pytest.mark.parametrize("version", ["3.11.99", "3.13.0", "4.12.0", "3.12", "3.12.15rc1", "3.12.014", None, True])
def test_python_family_and_release_boundaries_fail_closed(checkpoint, monkeypatch, version):
    directory, lock = checkpoint
    raw = verifier.load_json(directory / "raw-study.json")
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"python": version, "packages": {}})))
    with pytest.raises(ValueError, match="Python 3.12 family"):
        verifier.verify_locked_interpreter(Path(sys.executable), lock, raw)


def test_patch_version_flexibility_does_not_relax_exact_package_versions(checkpoint, monkeypatch):
    directory, lock = checkpoint
    raw = verifier.load_json(directory / "raw-study.json")
    raw["provenance"]["packages"] = {"qiskit": "2.6.2"}
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"python": "3.12.99", "packages": {"qiskit": "2.6.3"}})))
    with pytest.raises(ValueError, match="installed packages differ"):
        verifier.verify_locked_interpreter(Path(sys.executable), lock, raw)


@pytest.mark.parametrize("change", ["file", "new_file", "empty_directory", "symlink"])
def test_preservation_guard_covers_bytes_inventory_and_links(checkpoint, change):
    directory, _ = checkpoint
    before = verifier.inventory(directory)
    if change == "file":
        (directory / "resources.csv").write_text("changed\n")
    elif change == "new_file":
        (directory / "unexpected.txt").write_text("new")
    elif change == "empty_directory":
        (directory / "unexpected").mkdir()
    else:
        (directory / "linked.qasm").symlink_to(directory / "circuits/a.qasm")
    with pytest.raises(ValueError, match="changed during|Linked or special"):
        verifier.assert_preserved(directory, before)


def test_replay_failure_still_checks_original_preservation(checkpoint, monkeypatch):
    directory, lock = checkpoint
    def failed_replay(*args):
        (directory / "resources.csv").write_text("mutation during failure\n")
        raise RuntimeError("producer failed")
    monkeypatch.setattr(verifier, "run_archived_producer", failed_replay)
    with pytest.raises(ValueError, match="changed during verification"):
        verifier.verify_checkpoint(directory, python=Path(sys.executable), lock_file=lock)


def test_complete_temporary_resume_preserves_every_original_file(checkpoint):
    directory, lock = checkpoint
    before = verifier.inventory(directory)
    result = verifier.verify_checkpoint(directory, python=Path(sys.executable), lock_file=lock)
    assert verifier.inventory(directory) == before
    assert result["original_inventory_preserved"] is True
    assert result["training_repeated"] is False
    assert result["qasm_files_compared"] == 2
    assert result["archived_source_sha256"][verifier.PRODUCER] == verifier.sha256(directory / verifier.PRODUCER)
    assert result["runs_json_ignored_top_level_fields"] == ["artifact_generation_wall_seconds"]
