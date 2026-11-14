"""Publication gates reject altered records without rerunning native work."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("test_energy_record_release_gate",
                                             ROOT / "scripts/energy_record_release.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(records, manifest):
    path = records / gate.MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return digest(path)


@pytest.fixture
def synthetic_record(tmp_path):
    """Structural protocol fixture; its stub checker performs no science."""
    source = tmp_path / "source"
    records = tmp_path / "records"
    records.mkdir()
    for name in gate.SOURCE_FILES + gate.VERIFIER_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Structural test source: {name}\n")
    checker = source / "scripts/energy_physical_records.py"
    checker.write_text("def validate_physical_records(records, ledgers, profile):\n"
                       "    return {'cases': 6, 'native_reestimation_performed': False}\n")
    sources = {name: digest(source / name) for name in gate.SOURCE_FILES}
    verifiers = {name: digest(source / name) for name in gate.VERIFIER_FILES}
    for name in gate.CORE_FILES:
        path = records / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if name.endswith(".json") else f"Recorded fixture {name}\n")
    for name, original in (
        ("generate_energy_estimation.py", "scripts/generate_energy_estimation.py"),
        ("energy_estimation.py", "src/openquantum/energy_estimation.py"),
        ("qft_qpe.py", "src/openquantum/qft_qpe.py"),
        ("resource_estimation.py", "src/openquantum/resource_estimation.py"),
        ("LICENSE.txt", "LICENSE"),
    ):
        shutil.copyfile(source / original, records / name)
    environment = {"python": "3.12.14", "device": "CPU", "seed": 20261112,
                   "packages": {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4",
                                "scipy": "1.18.1", "matplotlib": "3.11.2"}}
    for case in gate.CASES:
        path = records / ("logical-resources.json" if case == "default"
                          else f"{case}/logical-resources.json")
        path.write_text(json.dumps({
            "source_sha256": {name: sources[name] for name in gate.LEDGER_SOURCE_FILES},
            "source_dirty": None, "source_commit": None, "environment": environment,
            "contract": {"dimers": 1, "phase_bits": 5, "requested_ideal_failure": .005},
            "sequential_equivalent_simulation": {"performed": case == "default"},
        }) + "\n")
    manifest = {"schema_version": 1, "status": "frozen_computed_conditional_energy_record",
                "files": gate.core_inventory(records), "sources": sources, "verifier_sources": verifiers,
                "physical_profile": dict(gate.PROFILE), "contract": gate.CONTRACT,
                "native_execution": {**environment, "producer_commit": "8" * 40,
                    "complete_default_producer": True, "native_estimation_performed": True,
                    "exit_status": 0, "wall_seconds": 30.0, "peak_rss_kib": 12_803_256},
                "physical_validation": {"cases": 6, "native_reestimation_performed": False}}
    expected = write_manifest(records, manifest)
    return records, source, manifest, expected


def test_structural_record_and_partial_preview_gate_use_only_saved_accounting(synthetic_record, monkeypatch):
    records, source, manifest, expected = synthetic_record
    result = gate.verify_energy_record_identity(records, source, expected_sha256=expected)
    assert result["core_files"] == 35
    assert result["logical_producer_executed"] is False
    assert result["native_reestimation_performed"] is False
    assert result["factory_search_repeated"] is False
    assert not list(source.rglob("__pycache__"))
    assert not list(records.rglob("__pycache__"))
    site = source.parent / "site"
    shutil.copytree(source, site / "code")
    shutil.copytree(records, site / gate.RECORD_DIRECTORY)
    assert not (site / "code/assets").exists()
    monkeypatch.setattr(gate, "ENERGY_RECORD_MANIFEST_SHA256", expected)
    assert gate.site_energy_record_errors(site) == []
    assert (records / gate.MANIFEST_NAME).read_text() == json.dumps(manifest, indent=2) + "\n"


@pytest.mark.parametrize("name", ["n8/measured-group.qasm", "scaling.svg", "surface-code.json"])
def test_changed_record_fails_before_checker_loading(synthetic_record, name):
    records, source, _, expected = synthetic_record
    (records / name).write_text("Changed bytes\n")
    with pytest.raises(ValueError, match="record inventory"):
        gate.verify_energy_record_identity(records, source, expected_sha256=expected)


@pytest.mark.parametrize("name", ["src/openquantum/resource_estimation.py", "requirements-lock.txt",
                                  "scripts/energy_physical_records.py", "scripts/verify_energy_estimation.py"])
def test_changed_source_or_model_fails_before_checker_loading(synthetic_record, name):
    records, source, _, expected = synthetic_record
    (source / name).write_text("raise RuntimeError('must never execute changed code')\n")
    with pytest.raises(ValueError, match="sources differ"):
        gate.verify_energy_record_identity(records, source, expected_sha256=expected)


@pytest.mark.parametrize("name", ["unexplained.png", "nested/extra.json", "n2/private.txt"])
def test_unexplained_core_entry_rejects(synthetic_record, name):
    records, source, _, expected = synthetic_record
    path = records / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Unreviewed\n")
    with pytest.raises(ValueError, match="unexpected core"):
        gate.verify_energy_record_identity(records, source, expected_sha256=expected)


def test_separate_pauli_subtree_is_not_a_core_input(synthetic_record):
    records, source, _, expected = synthetic_record
    pauli = records / "pauli-access"
    pauli.mkdir()
    (pauli / "plot.png").write_bytes(b"Separate chapter fixture")
    assert gate.verify_energy_record_identity(records, source, expected_sha256=expected)["core_files"] == 35


@pytest.mark.parametrize("where", ["records", "source", "record-file", "source-directory", "ancestor"])
def test_lexical_links_reject_before_load(synthetic_record, tmp_path, where):
    records, source, _, expected = synthetic_record
    if where in ("records", "source"):
        alias = tmp_path / "alias"
        alias.symlink_to(records if where == "records" else source, target_is_directory=True)
        records = alias if where == "records" else records
        source = alias if where == "source" else source
    elif where == "record-file":
        path = records / "scaling.svg"
        saved = tmp_path / "saved.svg"
        path.rename(saved)
        path.symlink_to(saved)
    elif where == "source-directory":
        path = source / "src"
        saved = tmp_path / "saved-src"
        path.rename(saved)
        path.symlink_to(saved, target_is_directory=True)
    else:
        alias = tmp_path / "ancestor-alias"
        alias.symlink_to(source.parent, target_is_directory=True)
        source = alias / "source"
    with pytest.raises(ValueError, match="linked"):
        gate.verify_energy_record_identity(records, source, expected_sha256=expected)


@pytest.mark.parametrize("keys,value", [
    (("schema_version",), True), (("status",), "pending"), (("contract",), "different"),
    (("physical_profile", "physical_error_rate"), .001),
    (("native_execution", "exit_status"), False),
    (("native_execution", "complete_default_producer"), 1),
    (("native_execution", "native_estimation_performed"), False),
    (("native_execution", "producer_commit"), "unknown"),
    (("native_execution", "seed"), True), (("native_execution", "wall_seconds"), 0),
    (("native_execution", "peak_rss_kib"), True), (("native_execution", "packages"), {}),
    (("files",), {}), (("sources",), {}), (("verifier_sources",), {}),
    (("physical_validation", "cases"), 5),
])
def test_schema_and_accounting_checks_survive_recomputed_outer_identity(synthetic_record, keys, value):
    records, source, manifest, _ = synthetic_record
    current = manifest
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    expected = write_manifest(records, manifest)
    with pytest.raises(ValueError):
        gate.verify_energy_record_identity(records, source, expected_sha256=expected)


@pytest.mark.parametrize("payload", [
    '{"schema_version":1,"schema_version":1}',
    '{"schema_version":NaN}', '{"schema_version":1e400}',
])
def test_json_duplicates_and_nonfinite_numbers_reject(synthetic_record, payload):
    records, source, _, _ = synthetic_record
    path = records / gate.MANIFEST_NAME
    path.write_text(payload)
    with pytest.raises(ValueError, match="JSON"):
        gate.verify_energy_record_identity(records, source, expected_sha256=digest(path))


def test_published_canonical_energy_record_has_reviewed_identity_and_saved_accounting():
    result = gate.verify_energy_record_identity(ROOT / gate.RECORD_DIRECTORY, ROOT)
    assert result["core_files"] == 35
    assert result["manifest_sha256"] == gate.ENERGY_RECORD_MANIFEST_SHA256
    assert result["native_reestimation_performed"] is False


def test_ci_pins_recorded_energy_check_and_keeps_neighboring_commands():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file():
        pytest.skip("Publication workflow is deliberately outside the public example package")
    import yaml
    steps = yaml.safe_load(workflow.read_text())["jobs"]["build"]["steps"]
    body = next(step["run"] for step in steps
                if step.get("name") == "Generate verified circuit and resource artifacts")
    commands = [line.removeprefix("run ") for line in body.splitlines() if line.startswith("run ")]
    command = ("python scripts/verify_energy_estimation.py --records-dir assets/generated/energy-estimation "
               f"--manifest-sha256 {gate.ENERGY_RECORD_MANIFEST_SHA256} --max-wall-seconds 600")
    assert commands.count(command) == 1
    index = commands.index(command)
    assert commands[index - 1] == "python scripts/generate_quantum_walks.py --estimate"
    assert commands[index + 1] == "python scripts/generate_energy_estimation_pauli.py --estimate --compare-controls"
    assert "python scripts/generate_energy_estimation.py" not in commands
    measured_and = "python scripts/verify_shor_measured_and_saved.py"
    assert len(commands) == 81 and commands.count(measured_and) == 1
    assert commands.index(measured_and) == commands.index("python scripts/generate_shor_recovery.py") + 1
    commands = [value for value in commands if value != measured_and]
    feedback = "python scripts/verify_qpe_feedback_public.py"
    assert len(commands) == 80 and commands.count(feedback) == 1
    assert commands.index(feedback) == commands.index("python scripts/verify_robust_qpe_saved.py") + 1
    commands = [value for value in commands if value != feedback]
    robust = "python scripts/verify_robust_qpe_saved.py"
    assert len(commands) == 79 and commands.count(robust) == 1
    assert commands.index(robust) == commands.index("python scripts/verify_bayesian_qpe_saved.py") + 1
    commands = [value for value in commands if value != robust]
    bayesian = "python scripts/verify_bayesian_qpe_saved.py"
    assert len(commands) == 78 and commands.count(bayesian) == 1
    assert commands.index(bayesian) == commands.index("python scripts/verify_hadamard_qpe.py") + 1
    commands = [value for value in commands if value != bayesian]
    assert len(commands) == 77
    assert hashlib.sha256(("\n".join(commands) + "\n").encode()).hexdigest() == (
        "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032")
    multiproduct = "python scripts/verify_simulation_multiproduct_saved.py"
    assert commands.count(multiproduct) == 1
    assert commands.index(multiproduct) == commands.index(
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay") + 1
    commands = [value for value in commands if value != multiproduct]
    assert len(commands) == 76
    assert hashlib.sha256(("\n".join(commands) + "\n").encode()).hexdigest() == (
        "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af")
    replacements = {
        ("python scripts/verify_dynamics_resources.py --canonical-qasm-replay "
         "--records-dir assets/generated/dynamics-resources --manifest-sha256 "
         "0596353af49a0de40176069756cd9398749c21ede68e057ef84611bb6d5f4265 "
         "--max-wall-seconds 600"): "python scripts/generate_dynamics_resources.py",
        "python scripts/verify_instrument_resource_records.py --replay": "python scripts/generate_instrument_resources.py",
        "python scripts/verify_simulation_complete_resource_records.py --replay": "python scripts/generate_simulation_complete_resources.py",
    }
    for replacement, original in replacements.items():
        assert commands.count(replacement) == 1 and original not in commands
    commands = [replacements.get(value, value) for value in commands]
    additions = (
        "python scripts/generate_simulation_suzuki.py --output-dir .cache/suzuki-ci-replay",
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay",
        "python scripts/verify_optimization_word_records.py",
        "python scripts/generate_shor_recovery.py", "python scripts/verify_hadamard_qpe.py",
        "python scripts/generate_simulation_qdrift.py --output-dir .cache/qdrift-ci-replay",
        "python scripts/verify_simulation_qdrift.py --replay-dir .cache/qdrift-ci-replay",
        "python scripts/verify_simulation_lindblad_port.py", "python scripts/verify_detailed_balance_channel.py",
        "python scripts/verify_dqi_saved.py assets/generated/optimization/dqi",
    )
    assert all(commands.count(value) == 1 for value in additions)
    commands = [value for value in commands if value not in additions]
    assert len(commands) == 66
    assert hashlib.sha256(("\n".join(commands) + "\n").encode()).hexdigest() == (
        "6d2b2979fdb474301a25d2e5479efa06c2cb1227ffbca786e8d552cb74391013")
    following_checks = {
        "python scripts/verify_simulation_lindblad_physical.py",
        "python scripts/verify_simulation_lindblad_three.py",
        "python scripts/verify_simulation_lindblad_four.py",
        "python scripts/verify_optimization_diagnostics.py",
        "python scripts/verify_optimization_compiled.py",
    }
    active_following = following_checks & set(commands)
    assert not active_following or active_following == following_checks
    assert len(commands) == 61 + len(active_following)
    if active_following:
        lindblad = commands.index("python scripts/generate_simulation_lindblad_workload.py")
        assert commands[lindblad + 1:lindblad + 4] == [
            "python scripts/verify_simulation_lindblad_physical.py",
            "python scripts/verify_simulation_lindblad_three.py",
            "python scripts/verify_simulation_lindblad_four.py",
        ]
        diagnostics = commands.index("python scripts/verify_optimization_diagnostics.py")
        assert commands[diagnostics + 1:diagnostics + 3] == [
            "python scripts/verify_optimization_compiled.py",
            "python scripts/generate_quantum_walks.py --estimate",
        ]
    assert 'if "$@"; then stage_status=0; else stage_status=$?; fi' in body
    assert 'return "$stage_status"' in body


def test_release_scripts_and_canonical_record_are_registered():
    export_spec = importlib.util.spec_from_file_location("energy_test_export", ROOT / "scripts/export_code.py")
    export = importlib.util.module_from_spec(export_spec)
    export_spec.loader.exec_module(export)
    assert {Path(name).name for name in gate.VERIFIER_FILES} | {"energy_record_release.py"} <= set(export.PUBLIC_SCRIPTS)
    assert export.include_generated_file(f"{gate.RECORD_DIRECTORY}/{gate.MANIFEST_NAME}")
    checker_path = ROOT / "scripts/check_site.py"
    if checker_path.is_file():
        check_spec = importlib.util.spec_from_file_location("energy_test_site", checker_path)
        checker = importlib.util.module_from_spec(check_spec)
        check_spec.loader.exec_module(checker)
        required = set(checker.required_artifacts())
        assert {f"{gate.RECORD_DIRECTORY}/{name}" for name in gate.CORE_FILES | {gate.MANIFEST_NAME}} <= required
        assert {f"code/{name}" for name in gate.SOURCE_FILES + gate.VERIFIER_FILES} <= required
        assert "code/scripts/energy_record_release.py" in required
