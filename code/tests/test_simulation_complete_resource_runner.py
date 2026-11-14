"""Artifact-boundary and standalone regressions; no synthesis/physical jobs."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "complete_resource_generator", ROOT / "scripts/generate_simulation_complete_resources.py")
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_repository_outputs_and_private_cache_stay_in_declared_domains(tmp_path):
    sources = generator.source_files()
    owned = ROOT / "assets/generated/simulation/complete-resources"
    output, cache = generator.validate_paths(owned, tmp_path / "private", sources)
    assert output == owned.resolve() and cache == (tmp_path / "private").resolve()
    with pytest.raises(ValueError, match="Repository output"):
        generator.validate_paths(ROOT / "assets/generated/resources", tmp_path / "private", sources)
    with pytest.raises(ValueError, match="outside the source checkout"):
        generator.validate_paths(owned, ROOT / "local-cache", sources)
    for public, private in ((tmp_path / "both", tmp_path / "both"),
                            (tmp_path / "both", tmp_path / "both/private"),
                            (tmp_path / "both/public", tmp_path / "both")):
        with pytest.raises(ValueError, match="disjoint"):
            generator.validate_paths(public, private, sources)


@pytest.mark.parametrize("relative", ["results.json", "progress.json", "thermal-n2-Z/report.json"])
def test_every_written_managed_target_rejects_symlinks(tmp_path, relative):
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "preserved.txt"
    target.write_text("unchanged")
    link = output / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        generator.validate_paths(output, tmp_path / "private", generator.source_files())
    assert target.read_text() == "unchanged"


def test_prepared_checkpoint_key_changes_with_numerical_package_identity(tmp_path, monkeypatch):
    source = tmp_path / "complete_resource_workloads.py"
    source.write_text("fixed source")
    sources = {source.name: source}
    versions = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1",
                "qdk": "1.34.0", "matplotlib": "3.11.2"}
    monkeypatch.setattr(generator.importlib.metadata, "version", lambda name: versions[name])
    initial = generator.prepared_key("dynamics-n1-Z", sources)
    versions["numpy"] = "changed-numpy"
    assert generator.prepared_key("dynamics-n1-Z", sources) != initial
    versions["numpy"] = "2.5.4"
    versions["scipy"] = "changed-scipy"
    assert generator.prepared_key("dynamics-n1-Z", sources) != initial
    versions["scipy"] = "1.18.1"
    source.write_text("changed source")
    assert generator.prepared_key("dynamics-n1-Z", sources) != initial


def test_standalone_and_package_imports_use_distinct_prepared_checkpoints(tmp_path, monkeypatch):
    source = tmp_path / "complete_resource_workloads.py"
    source.write_text("identical source in both import modes")
    sources = {source.name: source}
    monkeypatch.setattr(generator, "REPOSITORY", True)
    package_key = generator.prepared_key("dynamics-n1-Z", sources)
    monkeypatch.setattr(generator, "REPOSITORY", False)
    standalone_key = generator.prepared_key("dynamics-n1-Z", sources)
    assert standalone_key != package_key
    monkeypatch.setattr(generator, "REPOSITORY", True)
    assert generator.prepared_key("dynamics-n1-Z", sources) == package_key


def test_revision_ignores_callers_git_repository_selectors(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    source = tmp_path / "source.py"
    source.write_text("source")
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator, "REPOSITORY", True)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        monkeypatch.setenv(name, "unrelated-repository")
    calls = []

    def git(command, **kwargs):
        calls.append(kwargs)
        if command[-1] == "--show-toplevel":
            return str(tmp_path) + "\n"
        if command[-1] == "HEAD":
            return "1" * 40 + "\n"
        return "\n"

    monkeypatch.setattr(generator.subprocess, "check_output", git)
    record = generator.revision({source.name: source})
    assert record["source_commit_at_generation"] == "1" * 40
    assert calls and all("env" in call for call in calls)
    assert all(not ({"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"} & set(call["env"]))
               for call in calls)


def test_partial_stub_run_preserves_unknown_files_and_excludes_them_from_manifest(tmp_path, monkeypatch):
    output, cache = tmp_path / "output", tmp_path / "private"
    omitted = output / "thermal-n2-Z"
    omitted.mkdir(parents=True)
    for name in ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json"):
        (omitted / name).write_text("stale managed record")
    unrelated = omitted / "user-notes.txt"
    unrelated.write_text("preserve this unrelated file")
    external = tmp_path / "outside.txt"
    external.write_text("outside public bundle")
    (output / "unrelated-link.txt").symlink_to(external)
    source = QuantumCircuit(1, 1)
    source.reset(0)
    source.measure(0, 0)
    prepared = SimpleNamespace(
        task=SimpleNamespace(source=source),
        synthesized=SimpleNamespace(circuit=source),
        workload={"complete_attempts": 5, "accepted_slots": 1},
        as_record=lambda: {"complete_workload": {"complete_attempts": 5, "accepted_slots": 1}})
    report = {"status": "infeasible_declared_controller_model", "pareto_points": [],
              "controller_model_diagnostic": {"distance_independent_controller_model_contribution": .1}}
    monkeypatch.setattr(generator, "prepare", lambda *args: (prepared, 0., False))
    monkeypatch.setattr(generator.cr, "estimate_workload", lambda *args, **kwargs: report)
    def stub_plots(folder, *args):
        for name in generator.PLOTS:
            (folder / name).write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")

    monkeypatch.setattr(generator, "plots", stub_plots)
    monkeypatch.setattr(sys, "argv", [str(generator.HERE / "generate_simulation_complete_resources.py"),
                                    "--case", "dynamics-n1-Z", "--output-dir", str(output),
                                    "--cache-dir", str(cache)])
    generator.main()
    result = json.loads((output / "results.json").read_text())
    assert set(result["cases"]) == {"dynamics-n1-Z"}
    assert len(result["cases"]["dynamics-n1-Z"]["scenarios"]) == 4
    assert result["energy"] == {}
    with (output / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and all(row["status"] == report["status"] for row in rows)
    assert all(row["peak_physical_qubits"] == "" for row in rows)
    assert b"\r" not in (output / "summary.csv").read_bytes()
    assert unrelated.read_text() == "preserve this unrelated file"
    assert external.read_text() == "outside public bundle"
    assert not any((omitted / name).exists() for name in ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert "thermal-n2-Z/user-notes.txt" not in manifest["artifacts_sha256"]
    assert "unrelated-link.txt" not in manifest["artifacts_sha256"]


def test_standalone_download_help_resolves_only_copied_sources(tmp_path):
    for name, path in generator.source_files().items():
        shutil.copyfile(path, tmp_path / name)
    code = (
        "import importlib.util,json,pathlib;"
        "p=pathlib.Path('generate_simulation_complete_resources.py');"
        "s=importlib.util.spec_from_file_location('standalone',p);"
        "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
        "print(json.dumps({'repository':g.REPOSITORY,'revision':g.revision(g.source_files()),"
        "'paths':[str(p.resolve()) for p in g.source_files().values()]}))"
    )
    check = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                           text=True, check=True, timeout=30)
    record = json.loads(check.stdout)
    assert record["repository"] is False
    assert record["revision"]["source_commit_at_generation"] is None
    assert all(Path(path).parent == tmp_path for path in record["paths"])
    help_result = subprocess.run([sys.executable, "generate_simulation_complete_resources.py", "--help"],
                                 cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30)
    assert "thermal-n2-X" in help_result.stdout and "--reuse-prepared" in help_result.stdout


def test_generation_cannot_attest_sources_changed_after_preparation(tmp_path, monkeypatch):
    input_source = tmp_path / "input/complete_resource_workloads.py"
    input_source.parent.mkdir()
    input_source.write_text("original source")
    output, cache = tmp_path / "output", tmp_path / "private"
    source = QuantumCircuit(1, 1)
    source.reset(0)
    source.measure(0, 0)
    prepared = SimpleNamespace(task=SimpleNamespace(source=source),
                               synthesized=SimpleNamespace(circuit=source),
                               workload={"complete_attempts": 5, "accepted_slots": 1},
                               as_record=lambda: {})
    report = {"status": "infeasible_declared_controller_model", "pareto_points": []}
    monkeypatch.setattr(generator, "source_files", lambda: {input_source.name: input_source})
    monkeypatch.setattr(generator, "prepare", lambda *args: (prepared, 0., False))

    def changed_source(*args, **kwargs):
        input_source.write_text("changed source")
        return report

    monkeypatch.setattr(generator.cr, "estimate_workload", changed_source)
    monkeypatch.setattr(generator, "plots", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["generator", "--case", "dynamics-n1-Z",
                                    "--output-dir", str(output), "--cache-dir", str(cache)])
    with pytest.raises(AssertionError, match="Input sources changed during generation"):
        generator.main()
    assert not (output / "manifest.json").exists()
