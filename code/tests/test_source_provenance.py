"""Detached public sources must not fail or claim an ancestor's Git revision."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_provenance_checks", ROOT / "scripts/source_provenance.py")
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def git_command(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
    ).strip()


def make_repository(root):
    root.mkdir(parents=True, exist_ok=True)
    git_command(root, "init", "-q")
    hooks = root / "empty-hooks"
    hooks.mkdir()
    git_command(root, "-c", "user.name=Provenance Test", "-c", "user.email=provenance@example.invalid",
                "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={hooks}",
                "commit", "--allow-empty", "-q", "-m", "fixture")
    return git_command(root, "rev-parse", "HEAD")


def test_detached_root_never_runs_git(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A detached source directory must not invoke Git")
    monkeypatch.setattr(provenance.subprocess, "run", forbidden)
    assert provenance.git_revision(tmp_path) is None
    assert provenance.git_dirty(tmp_path, ["src/example.py"]) is None


def test_ancestor_repository_is_not_package_provenance(tmp_path):
    expected = make_repository(tmp_path / "ancestor")
    package = tmp_path / "ancestor/download"
    package.mkdir()
    assert git_command(package, "rev-parse", "HEAD") == expected
    assert provenance.git_revision(package) is None
    assert provenance.git_dirty(package, ["src/example.py"]) is None


def test_genuine_checkout_and_owned_dirty_status(tmp_path):
    checkout = tmp_path / "checkout"
    expected = make_repository(checkout)
    assert provenance.git_revision(checkout) == expected
    assert provenance.git_dirty(checkout, ["example.py"]) is False
    (checkout / "example.py").write_text("print('fixture')\n")
    assert provenance.git_dirty(checkout, ["example.py"]) is True
    assert provenance.git_dirty(checkout, ["other.py"]) is False


def test_worktree_git_file_is_supported(tmp_path):
    checkout = tmp_path / "checkout"
    expected = make_repository(checkout)
    worktree = tmp_path / "worktree"
    git_command(checkout, "worktree", "add", "--detach", "-q", str(worktree), "HEAD")
    assert (worktree / ".git").is_file()
    assert provenance.git_revision(worktree) == expected


def test_missing_git_executable_is_optional(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    make_repository(checkout)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    assert provenance.git_revision(checkout) is None
    assert provenance.git_dirty(checkout, ["example.py"]) is None


def test_external_git_selectors_cannot_redirect_metadata(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    expected = make_repository(checkout)
    external = tmp_path / "external"
    make_repository(external)
    monkeypatch.setenv("GIT_DIR", str(external / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(external))
    assert provenance.git_revision(checkout) == expected


@pytest.mark.parametrize("stdout,returncode", [
    ("/other/checkout\n" + "a" * 40 + "\n", 0),
    ("{root}\nnot-a-revision\n", 0),
    ("{root}\n" + "a" * 40 + "\n", 128),
    ("{root}\n", 0),
])
def test_invalid_git_result_has_no_revision(tmp_path, monkeypatch, stdout, returncode):
    (tmp_path / ".git").mkdir()
    response = subprocess.CompletedProcess([], returncode, stdout.format(root=tmp_path.resolve()), "")
    monkeypatch.setattr(provenance.subprocess, "run", lambda *args, **kwargs: response)
    assert provenance.git_revision(tmp_path) is None


@pytest.mark.parametrize("error", [FileNotFoundError("git missing"), subprocess.TimeoutExpired("git", 5)])
def test_git_failures_are_optional(tmp_path, monkeypatch, error):
    (tmp_path / ".git").mkdir()
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(provenance.subprocess, "run", fail)
    assert provenance.git_revision(tmp_path) is None


SMOKE_CASES = [
    ("generate_shor.py", ["--modulus", "15", "--phase-bits", "2", "--no-sim", "--no-scaling"]),
    ("generate_shor_driver.py", ["--modulus", "9", "--max-base-draws", "1", "--max-quantum-trials", "1"]),
    ("generate_shor_ripple.py", ["--phase-bits", "2", "--no-sim", "--no-extra"]),
    ("generate_linear_systems.py", ["--qubits", "1", "--phase-qubits", "2", "--no-sim", "--no-scaling", "--no-estimate"]),
    ("generate_linear_systems_qsvt.py", ["--word", "X", "--no-sim", "--no-scaling", "--no-estimate"]),
    ("generate_linear_inverse_lcu.py", ["--qubits", "2", "--degree", "1", "--rounds", "0", "--no-sim", "--no-scaling", "--no-estimate"]),
    ("generate_hamiltonian_simulation.py", ["--qubits", "1", "--steps", "1", "--no-sim"]),
    ("generate_qft_qpe.py", ["--qubits", "1", "--phase-bits", "2", "--no-sim", "--no-scaling"]),
    ("generate_amplitude_estimation.py", ["--data-qubits", "1", "--phase-bits", "2", "--no-sim", "--no-scaling"]),
    ("generate_quantum_walks.py", ["--qubits", "2", "--steps", "1", "--no-sim", "--no-scaling"]),
    ("generate_simulation_word_taylor.py", ["--qubits", "1", "--degree", "1", "--segments", "1", "--rounds", "0", "--no-sim", "--no-padding", "--no-scaling"]),
    ("generate_simulation_qsp_scaling.py", ["--qubits", "1", "--no-sim"]),
    ("generate_simulation_two_spin_lindblad.py", ["--steps", "1", "--grid", "1", "--no-sim", "--no-scaling"]),
    ("generate_simulation_preparation_amplification.py", ["--degree", "1", "--rounds", "0", "--betas", "1.25", "--no-sim", "--no-scaling"]),
    ("generate_optimization.py", ["--no-sim", "--qubits", "3"]),
    ("generate_optimization_annealing.py", ["--no-sim", "--qubits", "3", "--steps", "1"]),
]
REVISION_KEYS = {"source_commit", "source_commit_at_generation", "source_revision_at_generation", "source_base_commit", "code_commit", "source_git_commit_at_generation"}


def values_named(data, keys):
    values = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in keys:
                values.append(value)
            values.extend(values_named(value, keys))
    elif isinstance(data, list):
        for value in data:
            values.extend(values_named(value, keys))
    return values


def hash_values(data, hash_context=False):
    values = []
    if isinstance(data, dict):
        for key, value in data.items():
            context = hash_context or "hash" in key or "sha256" in key
            values.extend(hash_values(value, context))
    elif isinstance(data, list):
        for value in data:
            values.extend(hash_values(value, hash_context))
    elif hash_context and isinstance(data, str) and len(data) == 64:
        if all(character in "0123456789abcdef" for character in data):
            values.append(data)
    return values


@pytest.fixture(scope="module", params=["detached", "ancestor"])
def public_copy(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("public-generator-provenance")
    if request.param == "ancestor":
        make_repository(directory)
    package = directory / "download"
    package.mkdir()
    shutil.copytree(ROOT / "src", package / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (package / "scripts").mkdir()
    for path in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(path, package / "scripts" / path.name)
    for name in ("LICENSE", "requirements-lock.txt"):
        shutil.copy2(ROOT / name, package / name)
    return package


def package_environment(package):
    # Keep inherited process credentials out of pytest fixture repr/tracebacks.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package / "src")
    return environment


@pytest.mark.parametrize("script,arguments", SMOKE_CASES, ids=[case[0] for case in SMOKE_CASES])
def test_public_generator_circuit_only_run(public_copy, script, arguments):
    package = public_copy
    output = package / "results" / script.removesuffix(".py")
    result = subprocess.run([sys.executable, str(package / "scripts" / script), *arguments,
                             "--output-dir", str(output)], cwd=package, env=package_environment(package),
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    documents = [json.loads(path.read_text()) for path in output.rglob("*.json")]
    assert documents, "The standalone generator must write a complete metadata record"
    revisions = [item for document in documents for item in values_named(document, REVISION_KEYS)]
    assert revisions and all(item is None for item in revisions)
    assert any(hash_values(document) for document in documents), "Content hashes must remain present"
    # Outputs retain explicit real gate exports without needing Git or QDK.
    if script != "generate_shor_driver.py":
        assert list(output.rglob("*.qasm")), "Circuit-only output must contain the actual exported circuit"


def test_public_optimization_metadata_keeps_hashes_and_unknown_dirty_status(public_copy):
    package = public_copy
    code = """import hashlib, json, runpy
from pathlib import Path
root=Path.cwd()
result={}
for suffix in ('', '_sampled', '_portability', '_annealing', '_walk'):
    script=root/'scripts'/('generate_optimization'+suffix+'.py')
    result[suffix or 'exact']=runpy.run_path(str(script))['provenance']()
print(json.dumps(result))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=package, env=package_environment(package),
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    records = json.loads(result.stdout)
    expected = hashlib.sha256((package / "src/openquantum/optimization.py").read_bytes()).hexdigest()
    for record in records.values():
        assert record["source_commit"] is None
        assert record["dirty_owned_source"] is None
        assert record["implementation_sha256"] == expected


def test_public_projection_metadata_uses_content_hashes_without_git(public_copy):
    package = public_copy
    snapshot = package / "sampled-input.json"
    snapshot.write_text("{}\n")
    code = """import json, runpy
from pathlib import Path
root=Path.cwd()
module=runpy.run_path(str(root/'scripts/generate_optimization_projections.py'))
print(json.dumps(module['provenance'](root/'sampled-input.json')))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=package, env=package_environment(package),
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(result.stdout)
    assert record["source_commit_at_generation"] is None
    assert record["source_content_sha256"]["sampled_runs"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()


def test_public_mocked_annealing_resource_generation_without_git(public_copy):
    package = public_copy
    (package / "tests").mkdir(exist_ok=True)
    filename = "test_optimization_annealing_walk_resources.py"
    shutil.copy2(ROOT / "tests" / filename, package / "tests" / filename)
    # This regression uses shipped, hashed circuit/input pairs. Its physical
    # estimator is mocked; the real native trace/measurement audits still run.
    for method in ("annealing", "walk"):
        source = ROOT / "assets/generated/optimization" / method
        target = package / "assets/generated/optimization" / method
        shutil.copytree(source, target)
    temporary = package / "mocked-generation-test"
    case = "test_generate_allocates_all_recorded_preparations_with_mocked_physical_estimates"
    # The parent inspects this child's successful report below. Give the child
    # its own pytest options: the CI basetemp is an ancestor of this package,
    # and the successful report must survive until the parent reads it.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "--basetemp", str(temporary),
         "-o", "tmp_path_retention_policy=all",
         str(package / "tests" / filename) + "::" + case],
        cwd=package, env=package_environment(package) | {"PYTEST_ADDOPTS": ""},
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reports = list(temporary.rglob("only-test-output/report.json"))
    assert len(reports) == 1
    record = json.loads(reports[0].read_text())
    assert record["source_provenance"]["source_commit_at_generation"] is None
    # Resource replay uses the run-recorded producer even after the live
    # implementation gains unrelated chapters or diagnostics.
    implementation_hash = record["source_provenance"]["optimization_implementation_sha256"]
    for method in ("annealing", "walk"):
        folder = package / "assets/generated/optimization" / method
        runs = json.loads((folder / "runs.json").read_text())
        assert implementation_hash == runs["provenance"]["implementation_sha256"]
        assert implementation_hash == hashlib.sha256((folder / "optimization.py").read_bytes()).hexdigest()
