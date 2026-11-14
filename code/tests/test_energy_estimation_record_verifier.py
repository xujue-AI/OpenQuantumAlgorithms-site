"""Immutable-record protocol tests using the actually computed archived study.

The fixture keeps that study's archived producer/adapter bytes as its source
authority. A separate coordinator-approved probe covers the current producer.
Native estimation is never run by this test module.
"""
import importlib.util
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_energy_estimation as verifier


@pytest.fixture(scope="module")
def frozen_study(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("energy-record-protocol")
    records, source = temporary / "records", temporary / "source"
    records.mkdir(); source.mkdir()
    archived = ROOT / "assets/generated/energy-estimation"
    for name in verifier.CORE_FILES:
        destination = records / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archived / name, destination)
    archived_sources = {
        "scripts/generate_energy_estimation.py": verifier.PRODUCER,
        "src/openquantum/energy_estimation.py": "energy_estimation.py",
        "src/openquantum/qft_qpe.py": "qft_qpe.py",
        "src/openquantum/resource_estimation.py": "resource_estimation.py",
        "LICENSE": "LICENSE.txt",
    }
    for name in (*verifier.SOURCES, *verifier.VERIFIER_SOURCES):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = records / archived_sources[name] if name in archived_sources else ROOT / name
        shutil.copy2(original, destination)
    ledger = verifier.load_json(records / "logical-resources.json")
    environment = temporary / "native-environment.json"
    environment.write_text(json.dumps({
        "source_commit": ledger["source_commit"] or "a" * 40,
        "source_sha256": verifier.source_hashes(source), "device": "CPU", "seed": 20261112,
        "python": ledger["environment"]["python"], "packages": ledger["environment"]["packages"],
    }))
    measurement = temporary / "native-measurement.json"
    # This is an explicit test receipt for a real archived data fixture, not a
    # claim of new Native execution. Production freeze uses real job receipts.
    measurement.write_text(json.dumps({
        "conventional_exit_status": 0, "child_return_code": 0, "wall_seconds": 1.0,
        "ru_maxrss_kib": 1, "command": [sys.executable, "scripts/" + verifier.PRODUCER],
    }))
    manifest = temporary / verifier.MANIFEST_NAME
    result = verifier.freeze_record(records, manifest, environment, measurement, source_root=source)
    return {"directory": temporary, "records": records, "source": source,
            "manifest": manifest, "sha256": result["manifest_sha256"],
            "environment": environment, "measurement": measurement}


def clone_study(study, tmp_path):
    records = tmp_path / "records"
    shutil.copytree(study["records"], records)
    manifest = tmp_path / verifier.MANIFEST_NAME
    shutil.copy2(study["manifest"], manifest)
    return records, manifest


def emit_disabled_replay(records, output):
    output.mkdir()
    for name in verifier.CORE_FILES - verifier.PHYSICAL_FILES:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(records / name, destination)
    ledger_path = output / "logical-resources.json"
    ledger = verifier.load_json(ledger_path)
    ledger["physical_resource_estimate"] = {"status": "unavailable", "kind": None,
        "pareto_points": 0, "hardware_energy_confidence_certified": False}
    ledger_path.write_text(json.dumps(ledger))
    access_path = output / "access-comparison.json"
    access = verifier.load_json(access_path)
    for mode in ("repeated", "analytic_powers"):
        access[mode]["status"] = "unavailable"; access[mode]["pareto_points"] = []
    access_path.write_text(json.dumps(access))
    (output / "surface-code-scaling.json").write_text(json.dumps({
        "status": "unavailable", "common_absolute_energy_error_target_Ha": .5,
        "same_ideal_failure": .005, "hardware_energy_confidence_certified": False, "points": []}))


def test_actual_no_estimate_producer_replay_preserves_archived_study(frozen_study):
    study = frozen_study
    before = verifier.core_inventory(study["records"])
    result = verifier.verify_record(study["records"], study["manifest"], study["sha256"],
                                    source_root=study["source"], max_wall_seconds=600)
    assert result["logical_producer_executed"] is True
    assert result["native_reestimation_performed"] is False
    assert result["factory_search_repeated"] is False
    assert result["core_files"] == 35
    assert verifier.core_inventory(study["records"]) == before


def test_freeze_refuses_overwrite_and_failed_or_disabled_native_receipts(frozen_study, tmp_path):
    study = frozen_study
    with pytest.raises(ValueError, match="already exists"):
        verifier.freeze_record(study["records"], study["manifest"], study["environment"],
                               study["measurement"], source_root=study["source"])
    for change in ({"conventional_exit_status": 143},
                   {"command": [sys.executable, verifier.PRODUCER, "--no-estimate", "--output-dir", "x"]},
                   {"command": [None, verifier.PRODUCER]},
                   {"command": [sys.executable, verifier.PRODUCER, "--output-dir", None]}):
        receipt = verifier.load_json(study["measurement"]); receipt.update(change)
        path = tmp_path / "bad-receipt.json"; path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError):
            verifier.freeze_record(study["records"], tmp_path / "new-manifest.json",
                                   study["environment"], path, source_root=study["source"])
    for path in (study["records"] / "n8" / verifier.MANIFEST_NAME,
                 study["records"] / "pauli-access" / verifier.MANIFEST_NAME):
        path.parent.mkdir(exist_ok=True)
        with pytest.raises(ValueError, match="outside all recorded"):
            verifier.freeze_record(study["records"], path, study["environment"],
                                   study["measurement"], source_root=study["source"])
        assert not path.exists()


@pytest.mark.parametrize("mutation", ["missing", "pending", "schema", "receipt", "budget", "runtime", "qasm", "sample"])
def test_mutations_fail_closed_even_with_refreshed_file_hashes(frozen_study, tmp_path, monkeypatch, mutation):
    records, manifest_path = clone_study(frozen_study, tmp_path)
    manifest = verifier.load_json(manifest_path)
    if mutation == "missing":
        (records / "n8/surface-code.json").unlink()
    elif mutation == "pending":
        manifest["status"] = "planned"
    elif mutation == "schema":
        manifest["schema_version"] = True
    elif mutation == "receipt":
        manifest["native_execution"]["packages"] = {}
    elif mutation in ("budget", "runtime"):
        path = records / "n8/surface-code.json"
        value = verifier.load_json(path)
        if mutation == "budget": value["budget"]["physical_failure_probability"] = .05
        else: value["pareto_points"][0]["serial_runtime_all_executions_ns"] *= 2
        path.write_text(json.dumps(value))
        manifest["files"]["n8/surface-code.json"] = {"sha256": verifier.sha256(path), "bytes": path.stat().st_size}
    elif mutation == "qasm":
        path = records / "n8/measured-group.qasm"
        path.write_text(path.read_text() + "x q[0];\n")
        manifest["files"]["n8/measured-group.qasm"] = {"sha256": verifier.sha256(path), "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest))
    def fake_producer(source, output, workdir, timeout, **kwargs):
        emit_disabled_replay(records, output)
        if mutation == "sample":
            path = output / "results.json"; value = verifier.load_json(path)
            value["actual_estimate_error"] += .1
            path.write_text(json.dumps(value))
    monkeypatch.setattr(verifier, "run_producer", fake_producer)
    with pytest.raises(ValueError):
        verifier.verify_record(records, manifest_path, verifier.sha256(manifest_path),
                               source_root=frozen_study["source"])


def test_manifest_source_and_links_are_rejected_before_replay(frozen_study, tmp_path, monkeypatch):
    records, manifest = clone_study(frozen_study, tmp_path)
    def forbidden(*args): raise AssertionError("Replay must not start")
    monkeypatch.setattr(verifier, "run_producer", forbidden)
    with pytest.raises(ValueError, match="reviewed immutable"):
        verifier.verify_record(records, manifest, "0" * 64, source_root=frozen_study["source"])
    source = tmp_path / "source"; shutil.copytree(frozen_study["source"], source)
    with (source / "src/openquantum/resource_estimation.py").open("a") as handle:
        handle.write("\n# Unreviewed source change.\n")
    with pytest.raises(ValueError, match="stale"):
        verifier.verify_record(records, manifest, verifier.sha256(manifest), source_root=source)
    path = records / "n8/surface-code.json"; path.unlink()
    path.symlink_to(frozen_study["records"] / "n8/surface-code.json")
    with pytest.raises(ValueError, match="Linked"):
        verifier.verify_record(records, manifest, verifier.sha256(manifest), source_root=frozen_study["source"])


def test_symlinked_ancestor_and_unexplained_core_png_are_rejected(frozen_study, tmp_path):
    records, manifest = clone_study(frozen_study, tmp_path)
    alias = tmp_path / "alias"; alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        verifier.verify_record(alias / "records", manifest, verifier.sha256(manifest),
                               source_root=frozen_study["source"])
    (records / "unexplained.png").write_bytes(b"unexplained")
    with pytest.raises(ValueError, match="Unexpected core"):
        verifier.verify_record(records, manifest, verifier.sha256(manifest), source_root=frozen_study["source"])


@pytest.mark.parametrize("value", [0, -1, 601, float("nan"), True])
def test_wall_cap_is_positive_finite_and_at_most_six_hundred_seconds(value, tmp_path):
    with pytest.raises(ValueError, match="Wall cap"):
        verifier.verify_record(tmp_path, tmp_path / "missing", "0" * 64, max_wall_seconds=value)


def test_child_timeout_reaps_owned_process_and_preserves_other_jobs(tmp_path):
    source = tmp_path / "source"; (source / "scripts").mkdir(parents=True)
    script = source / "scripts" / verifier.PRODUCER
    pid_file = tmp_path / "owned-child.pid"
    script.write_text("import os,time,signal\nfrom pathlib import Path\n"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                      f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
                      "time.sleep(30)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        verifier.run_producer(source, tmp_path / "output", tmp_path, .2, cleanup_grace=.05)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError): os.kill(pid, 0)


@pytest.mark.parametrize("leader_exited", [False, True])
def test_cleanup_kills_owned_group_after_alarm_or_leader_exit(monkeypatch, leader_exited):
    calls = []
    monkeypatch.setattr(verifier.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    class Child:
        pid = 123456789
        waits = 0
        def wait(self, *, timeout):
            assert timeout == .01
            self.waits += 1
            if self.waits == 1 and not leader_exited:
                raise TimeoutError("Outer alarm arrived during TERM")
            return 0
    verifier.cleanup_owned_group(Child(), .01)
    assert calls == [(Child.pid, signal.SIGTERM), (Child.pid, signal.SIGKILL)]


def test_json_receipt_loader_rejects_fifo_and_oversize_without_reading(tmp_path):
    fifo = tmp_path / "receipt-fifo"; os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        verifier.load_json(fifo)
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as handle: handle.truncate(128 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="size limit"):
        verifier.load_json(oversized)


def test_blocked_alarm_is_rejected_before_any_replay():
    original = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        with pytest.raises(ValueError, match="blocked SIGALRM"):
            with verifier.wall_limit(1):
                raise AssertionError("Blocked wall deadline must not enter verification")
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, original)


@pytest.mark.parametrize("name", verifier.REFERENCE_ARRAYS)
def test_only_four_nominal_reference_arrays_allow_small_host_rounding(frozen_study, name):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original)
    value = replay["single_round_reference"][name]
    if name == "conditional_laws": value[3][23] += 5e-15
    elif name == "marginal": value[23] += 5e-16
    elif name == "phases": value[3] += 1e-16
    else: value[2] += 5e-17
    assert replay != original
    verifier.compare_energy_results(original, replay)


@pytest.mark.parametrize("mutation", ["large", "type", "nonfinite", "shape", "rowshape", "order", "range", "normalization", "mixture"])
def test_nominal_reference_validation_remains_fail_closed(frozen_study, mutation):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original); reference = replay["single_round_reference"]
    if mutation == "large": reference["eigenvalues"][2] += 2e-12
    elif mutation == "type": reference["conditional_laws"][0][0] = 0
    elif mutation == "nonfinite": reference["marginal"][0] = float("inf")
    elif mutation == "shape": reference["phases"].pop()
    elif mutation == "rowshape": reference["conditional_laws"][0].pop()
    elif mutation == "order": reference["eigenvalues"].reverse()
    elif mutation == "range": reference["phases"][0] = .249999999999
    elif mutation == "normalization":
        reference["marginal"] = [item + verifier.REFERENCE_ATOL / 2 for item in reference["marginal"]]
    else:
        delta = verifier.REFERENCE_ATOL * .75
        reference["conditional_laws"][0][0] -= delta
        reference["conditional_laws"][0][1] += delta
        reference["marginal"][0] += delta
        reference["marginal"][1] -= delta
    with pytest.raises(ValueError): verifier.compare_energy_results(original, replay)


@pytest.mark.parametrize("field", ["weights", "ground_energy", "actual_estimate_error", "actual_samples"])
def test_nonreference_results_stay_exact_even_for_one_bit_changes(frozen_study, field):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original)
    if field == "weights":
        value = replay["single_round_reference"]["weights"][0]
        replay["single_round_reference"]["weights"][0] += math.ulp(value)
    elif field == "ground_energy":
        value = replay["single_round_reference"]["ground_energy"]
        replay["single_round_reference"]["ground_energy"] += math.ulp(value)
    elif field == "actual_estimate_error":
        replay[field] += math.ulp(replay[field])
    else:
        replay["actual_sequential_samples"]["estimate"] += math.ulp(replay["actual_sequential_samples"]["estimate"])
    with pytest.raises(ValueError, match="outside four"):
        verifier.compare_energy_results(original, replay)


@pytest.mark.parametrize("side", ["recorded", "replayed"])
@pytest.mark.parametrize("residual", [0.0, 1e-12, 6.938893903907228e-16])
def test_fixed_complete_validation_residual_uses_producer_bound(frozen_study, side, residual):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original)
    (original if side == "recorded" else replay)[verifier.SMALL_COMPLETE_RESIDUAL] = residual
    before = (verifier.canonical(original), verifier.canonical(replay))
    verifier.compare_energy_results(original, replay)
    assert (verifier.canonical(original), verifier.canonical(replay)) == before


@pytest.mark.parametrize("side", ["recorded", "replayed"])
@pytest.mark.parametrize("residual", [float("nan"), float("inf"), float("-inf"), True,
                                      0, "0.0", None, -1e-16, math.nextafter(1e-12, math.inf)])
def test_complete_validation_residual_rejects_invalid_values(frozen_study, side, residual):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original)
    (original if side == "recorded" else replay)[verifier.SMALL_COMPLETE_RESIDUAL] = residual
    with pytest.raises(ValueError):
        verifier.compare_energy_results(original, replay)


@pytest.mark.parametrize("side", ["recorded", "replayed"])
@pytest.mark.parametrize("mutation", ["residual_missing", "parameters_missing", "wrong_dimers",
                                      "wrong_phase_bits", "wrong_rounds", "bool", "float",
                                      "extra", "missing"])
def test_complete_validation_binding_is_typed_and_exact(frozen_study, side, mutation):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original)
    value = original if side == "recorded" else replay
    parameters = value["small_complete_parameters"]
    if mutation == "residual_missing": value.pop(verifier.SMALL_COMPLETE_RESIDUAL)
    elif mutation == "parameters_missing": value.pop("small_complete_parameters")
    elif mutation == "wrong_dimers": parameters["dimers"] = 2
    elif mutation == "wrong_phase_bits": parameters["phase_bits"] = 4
    elif mutation == "wrong_rounds": parameters["rounds"] = 5
    elif mutation == "bool": parameters["dimers"] = True
    elif mutation == "float": parameters["phase_bits"] = 3.0
    elif mutation == "extra": parameters["extra"] = 0
    else: parameters.pop("rounds")
    with pytest.raises(ValueError):
        verifier.compare_energy_results(original, replay)


def test_unexpected_remaining_difference_names_only_bounded_json_pointer(frozen_study):
    original = verifier.load_json(frozen_study["records"] / "results.json")
    replay = copy.deepcopy(original)
    probability = replay["robust_reference"]["probabilities"][0]
    replay["robust_reference"]["probabilities"][0] = math.nextafter(probability, math.inf)
    with pytest.raises(ValueError) as caught:
        verifier.compare_energy_results(original, replay)
    assert str(caught.value).endswith(" at /robust_reference/probabilities/0")
    assert str(probability) not in str(caught.value)
    assert verifier.first_json_difference_pointer({"a/b~": 1}, {"a/b~": 1.0}) == "/a~1b~0"
    assert verifier.first_json_difference_pointer(-0.0, 0.0) == "/"
    assert len(verifier.first_json_difference_pointer({"x" * 1024: 1}, {"x" * 1024: 2})) <= 256
    assert "diagnostic-limit" in verifier.first_json_difference_pointer([0] * 4096 + [1], [0] * 4097)
    nested = 0
    for _ in range(40): nested = [nested]
    assert "diagnostic-limit" in verifier.first_json_difference_pointer(nested, nested)
