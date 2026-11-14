"""Saved source selection and bounded export lifecycle regressions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_detailed_balance_access.py"
MODULE = ROOT / "src/openquantum/simulation/detailed_balance_access.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("bounded_access_generator_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exported_source_and_lock_win_over_a_newer_parent_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    adjacent = checkout / ".cache"
    adjacent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, adjacent / SCRIPT.name)
    shutil.copyfile(MODULE, adjacent / MODULE.name)
    shutil.copyfile(ROOT / "requirements-lock.txt", adjacent / "requirements-lock.txt")
    shutil.copyfile(ROOT / "LICENSE", adjacent / "LICENSE.txt")
    (checkout / "requirements-lock.txt").write_text("newer checkout lock must not be copied")
    (checkout / "LICENSE").write_text("newer checkout license must not be copied")
    foreign = checkout / "src/openquantum/simulation"
    foreign.mkdir(parents=True)
    (foreign.parent / "__init__.py").write_text("")
    (foreign / "__init__.py").write_text("")
    (foreign / MODULE.name).write_text("raise RuntimeError('newer checkout module must not execute')\n")
    (checkout / ".gitignore").write_text(".cache/\n")
    subprocess.run(["git", "init", "-b", "codex/test-snapshot", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "Freeze parent files"], check=True, capture_output=True)
    output = adjacent / "result"
    subprocess.run([sys.executable, str(adjacent / SCRIPT.name), "--no-sim", "--output-dir", str(output)],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=20)
    report = json.loads((output / "results.json").read_text())
    assert report["source_commit"] is None
    assert report["source_revision_is_exact"] is False
    assert report["source_files"][MODULE.name]["sha256"] == hashlib.sha256(MODULE.read_bytes()).hexdigest()
    assert (output / "requirements-lock.txt").read_bytes() == (ROOT / "requirements-lock.txt").read_bytes()
    assert (output / "LICENSE.txt").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert not report["cases"][0]["numerical_checks_performed"]


def test_cached_foreign_module_is_rejected_before_it_can_be_hashed(tmp_path):
    shutil.copyfile(SCRIPT, tmp_path / SCRIPT.name)
    shutil.copyfile(MODULE, tmp_path / MODULE.name)
    code = """import runpy,sys,types
foreign=types.ModuleType('detailed_balance_access')
foreign.__file__='/foreign/source/detailed_balance_access.py'
sys.modules['detailed_balance_access']=foreign
runpy.run_path(sys.argv[1],run_name='standalone_selection_probe')
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / SCRIPT.name)],
                            check=False, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "imported access module differs" in result.stderr


def test_existing_outputs_and_byte_budget_fail_before_overwrite(tmp_path, monkeypatch):
    generator = load_generator()
    (tmp_path / "preserved.txt").write_text("preserve")
    with pytest.raises(ValueError, match="fresh or empty"):
        generator.generate(tmp_path)
    assert (tmp_path / "preserved.txt").read_text() == "preserve"
    monkeypatch.setattr(generator, "MAX_OUTPUT_BYTES", 3)
    with pytest.raises(ValueError, match="64 MiB"):
        generator._write(tmp_path / "too-large.txt", b"four", {})
    assert not (tmp_path / "too-large.txt").exists()


def test_dirty_source_scope_uses_the_actually_selected_adjacent_inputs(tmp_path, monkeypatch):
    generator = load_generator()
    adjacent = tmp_path / "saved"
    adjacent.mkdir()
    lock = adjacent / "requirements-lock.txt"
    license_file = adjacent / "LICENSE.txt"
    lock.write_text("saved lock")
    license_file.write_text("saved license")
    monkeypatch.setattr(generator, "ROOT", ROOT)
    # Selected inputs outside the exact checkout cannot be claimed exact at
    # that checkout's revision, even if its own root lock is unchanged.
    assert generator._source_dirty((license_file, lock)) is None
