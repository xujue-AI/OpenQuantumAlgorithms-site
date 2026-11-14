"""Report formatting cannot mutate authenticated scientific inputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_optimization_word_report.py"


@pytest.fixture
def report():
    spec = importlib.util.spec_from_file_location("word_report_output_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    exec(compile(SCRIPT.read_bytes(), str(SCRIPT), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("suffix", ["", "report", "nested/report"])
def test_external_record_outputs_rejected_before_any_formatting(tmp_path, report, suffix):
    bundle = tmp_path / "extracted-records"
    bundle.mkdir()
    output = bundle / suffix
    before = list(bundle.iterdir())
    with pytest.raises(ValueError, match="must not overwrite immutable"):
        report.format_report(bundle, output, {"status": "verified_saved_word_native_model"})
    assert list(bundle.iterdir()) == before


@pytest.mark.parametrize("suffix", ["", "optimization/word-transport", "other/report"])
def test_checkout_artifact_outputs_remain_protected(tmp_path, report, suffix):
    output = ROOT / "assets/generated" / suffix
    with pytest.raises(ValueError, match="must not overwrite immutable"):
        report._checked_output(tmp_path / "external", output)


def test_alias_to_external_record_output_is_rejected(tmp_path, report):
    bundle = tmp_path / "extracted-records"
    bundle.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="must not overwrite immutable"):
        report._checked_output(bundle, alias / "report")
    assert list(bundle.iterdir()) == []


def test_separate_output_resolution_performs_no_writes(tmp_path, report):
    bundle = tmp_path / "extracted-records"
    output = tmp_path / "report"
    assert report._checked_output(bundle, output) == output.resolve()
    assert not bundle.exists() and not output.exists()


@pytest.mark.parametrize("target_kind", ["regular", "symlink", "hardlink", "fifo", "directory"])
def test_existing_output_cannot_supply_aliased_report_targets(tmp_path, report, target_kind):
    import os
    bundle = tmp_path / "extracted-records"
    bundle.mkdir()
    scientific = bundle / "comparison.csv"
    scientific.write_text("immutable scientific bytes\n")
    output = tmp_path / "output"
    output.mkdir()
    target = output / "comparison.csv"
    if target_kind == "regular":
        target.write_text("prior report\n")
    elif target_kind == "symlink":
        target.symlink_to(scientific)
    elif target_kind == "hardlink":
        os.link(scientific, target)
    elif target_kind == "fifo":
        os.mkfifo(target)
    else:
        target.mkdir()
    before = scientific.read_bytes()
    with pytest.raises(ValueError, match="requires a new directory"):
        report.format_report(bundle, output, {"status": "verified_saved_word_native_model"})
    assert scientific.read_bytes() == before


@pytest.mark.parametrize("suffix", ["", "report"])
def test_cli_rejects_external_record_output_before_loading_release(tmp_path, report, monkeypatch, suffix):
    bundle = tmp_path / "extracted-records"
    bundle.mkdir()
    def forbidden():
        raise AssertionError("Unsafe output reached record authentication")
    monkeypatch.setattr(report, "release_module", forbidden)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--records-dir", str(bundle),
                                      "--output-dir", str(bundle / suffix)])
    with pytest.raises(SystemExit, match="must not overwrite immutable"):
        report.main()
    assert list(bundle.iterdir()) == []
