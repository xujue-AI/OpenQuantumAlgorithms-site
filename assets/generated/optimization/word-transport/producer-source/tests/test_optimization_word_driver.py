"""Pure regression checks of the source-only candidate driver's boundaries.

No saved class is decoded and no density, optimizer, synthesis or native
resource calculation is run by this file.  All candidate receipts are mocked.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver():
    path = ROOT / "scripts/prepare_optimization_word_transport.py"
    name = "source_only_word_driver_test"
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mock_recipe(driver):
    return dict(
        owned_source_sha256={name: "0" * 64 for name in driver.OWNED_PATHS},
        source_caps=deepcopy(driver.SOURCE_CAPS), base_commit=driver.BASE_COMMIT,
        run_id=driver.RUN_ID,
        classes=[dict(class_id=f"class-{index:02d}", shots=1) for index in range(54)],
    )


def forbidden(*args, **kwargs):
    raise AssertionError("This pure boundary check must not execute that path")


def test_module_pin_rejects_changed_bytes_before_execution(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    path = tmp_path / "src/openquantum/untrusted.py"
    path.parent.mkdir(parents=True)
    marker = tmp_path / "executed.txt"
    path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    with pytest.raises(ValueError, match="Frozen candidate source changed"):
        driver.module("untrusted", expected="0" * 64)
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ("missing_pin", "extra_pin", "changed_caps", "changed_base", "changed_run"))
def test_bound_recipe_rejects_scope_drift_before_git_or_imports(driver, tmp_path, monkeypatch, mutation):
    recipe = mock_recipe(driver)
    if mutation == "missing_pin":
        recipe["owned_source_sha256"].pop(next(iter(recipe["owned_source_sha256"])))
    elif mutation == "extra_pin":
        recipe["owned_source_sha256"]["unreviewed.py"] = "0" * 64
    elif mutation == "changed_caps":
        recipe["source_caps"]["classes"] = 55
    elif mutation == "changed_base":
        recipe["base_commit"] = "unreviewed"
    else:
        recipe["run_id"] = "different-selected-run"
    write_json(tmp_path / "recipe.json", recipe)
    monkeypatch.setattr(driver.subprocess, "run", forbidden)
    monkeypatch.setattr(driver, "module", forbidden)
    with pytest.raises(ValueError, match="pin set or execution caps changed"):
        driver.bound_recipe(tmp_path)


def test_verify_rejects_empty_manifest_with_54_frozen_classes(driver, tmp_path, monkeypatch):
    recipe = mock_recipe(driver)
    monkeypatch.setattr(driver, "bound_recipe", lambda output: (recipe, {}))
    monkeypatch.setattr(driver, "module", forbidden)
    monkeypatch.setattr(driver, "verify_inventory", forbidden)
    write_json(tmp_path / "artifact-manifest.json", dict(files={}))
    write_json(tmp_path / "validation.json", dict(classes=54, preparations=82032))
    with pytest.raises(ValueError, match="omitted or added a frozen class artifact"):
        driver.verify(tmp_path)


def test_verify_requires_all_prescribed_artifacts_to_exist(driver, tmp_path, monkeypatch):
    recipe = mock_recipe(driver)
    required = {f"circuits/{c['class_id']}.{suffix}" for c in recipe["classes"]
                for suffix in ("qasm", "json.gz")} | {"validation.json", "accounting.json"}
    monkeypatch.setattr(driver, "bound_recipe", lambda output: (recipe, {}))
    monkeypatch.setattr(driver, "module", forbidden)
    write_json(tmp_path / "artifact-manifest.json", dict(files={name: {} for name in required}))
    write_json(tmp_path / "validation.json", dict(classes=54, preparations=82032))
    with pytest.raises(ValueError, match="output file set differs"):
        driver.verify(tmp_path)


def test_write_files_checks_existing_size_before_reading(driver, tmp_path, monkeypatch):
    output = tmp_path / "candidate"
    output.mkdir()
    with (output / "oversized.bin").open("wb") as stream:
        stream.truncate(driver.BYTE_CAP + 1)
    monkeypatch.setattr(driver, "regular", forbidden)
    with pytest.raises(ValueError, match="Existing private candidate output exceeded"):
        driver.write_files(output, {"new.json": b"{}"})
    assert not (output / "new.json").exists()


def test_write_files_preflights_all_conflicts_before_any_write(driver, tmp_path):
    output = tmp_path / "candidate"
    output.mkdir()
    existing = output / "existing.json"
    existing.write_bytes(b"original")
    with pytest.raises(ValueError, match="cannot be overwritten"):
        driver.write_files(output, {"new.json": b"new", "existing.json": b"changed"})
    assert existing.read_bytes() == b"original"
    assert not (output / "new.json").exists()


def test_write_files_rejects_escape_before_any_write(driver, tmp_path):
    output = tmp_path / "candidate"
    output.mkdir()
    with pytest.raises(ValueError, match="escaped its output"):
        driver.write_files(output, {"new.json": b"new", "../escaped.json": b"changed"})
    assert not (output / "new.json").exists()
    assert not (tmp_path / "escaped.json").exists()


def test_verify_checks_output_byte_cap_before_reading_circuits(driver, tmp_path, monkeypatch):
    recipe = mock_recipe(driver)
    required = {f"circuits/{c['class_id']}.{suffix}" for c in recipe["classes"]
                for suffix in ("qasm", "json.gz")} | {"validation.json", "accounting.json"}
    fixed = {"recipe.json", "input-inventory.json", "accounting-plan.json", "artifact-manifest.json"}
    for name in required | fixed:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    write_json(tmp_path / "artifact-manifest.json", dict(files={name: {} for name in required}))
    write_json(tmp_path / "validation.json", dict(classes=54, preparations=82032))
    with (tmp_path / "circuits/class-00.qasm").open("wb") as stream:
        stream.truncate(driver.BYTE_CAP + 1)
    monkeypatch.setattr(driver, "bound_recipe", lambda output: (recipe, {}))
    monkeypatch.setattr(driver, "regular", forbidden)
    with pytest.raises(ValueError, match="byte cap|byte limit|exceeded"):
        driver.verify(tmp_path)

