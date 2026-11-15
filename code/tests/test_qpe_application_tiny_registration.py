"""Corruption tests for the stdlib-only retained QPE packet admission gate."""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qpe_application_tiny_saved_gate", ROOT / "scripts/verify_qpe_application_tiny_saved.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def empty_packet(tmp_path):
    """Only tiny inert placeholders; no scientific module is imported or run."""
    root = tmp_path / "packet"
    root.mkdir()
    for name in GATE.PACKET_NAMES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"inert placeholder\n")
    return root


def verify(root, tmp_path, **kwargs):
    return GATE.verify_saved_packet(root, manifest_sha256=GATE.MANIFEST_SHA256,
        projection_sha256=GATE.PROJECTION_SHA256, figure=tmp_path / "unused.svg",
        figure_sha256=GATE.FIGURE_SHA256, **kwargs)


def unreadable(*args, **kwargs):
    raise AssertionError("Payload reads must follow closed ordinary-file admission")


def test_extra_executable_is_rejected_before_payload_reads(tmp_path, monkeypatch):
    root = empty_packet(tmp_path)
    (root / "extra.py").write_text("raise AssertionError('never import')\n")
    monkeypatch.setattr(GATE, "_read", unreadable)
    with pytest.raises(ValueError, match="Unexpected packet member"):
        verify(root, tmp_path)


def test_source_directory_symlink_is_rejected_before_external_read(tmp_path, monkeypatch):
    root = empty_packet(tmp_path)
    external = tmp_path / "external"
    (root / "src/openquantum").rename(external)
    (root / "src/openquantum").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(GATE, "_read", unreadable)
    with pytest.raises(ValueError, match="Unexpected packet member"):
        verify(root, tmp_path)


def test_hardlinked_manifest_is_rejected_before_payload_reads(tmp_path, monkeypatch):
    root = empty_packet(tmp_path)
    os.link(root / "manifest.json", tmp_path / "manifest-alias.json")
    monkeypatch.setattr(GATE, "_read", unreadable)
    with pytest.raises(ValueError, match="Invalid ordinary packet member"):
        verify(root, tmp_path)


def test_corrupt_manifest_fails_before_checker_source_is_opened(tmp_path, monkeypatch):
    root = empty_packet(tmp_path)
    real_open = GATE.os.open
    opened = []

    def observed_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        if Path(path).suffix == ".py":
            raise AssertionError("Corrupt manifest must reject before reading scientific sources")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(GATE.os, "open", observed_open)
    with pytest.raises(ValueError, match="Member SHA-256 differs: manifest.json"):
        verify(root, tmp_path)
    assert opened == ["manifest.json"]


@pytest.mark.parametrize("pin", [None, "x" * 64, "0" * 64])
def test_unreviewed_external_manifest_pin_rejects_before_filesystem(tmp_path, monkeypatch, pin):
    monkeypatch.setattr(GATE, "_inventory", unreadable)
    with pytest.raises(ValueError, match="exact reviewed external manifest pin"):
        GATE.verify_saved_packet(tmp_path / "missing", manifest_sha256=pin,
            projection_sha256=GATE.PROJECTION_SHA256, figure=tmp_path / "missing.svg",
            figure_sha256=GATE.FIGURE_SHA256)


@pytest.mark.parametrize("seconds", [True, 0, 31])
def test_invalid_time_ceiling_rejects_before_filesystem(tmp_path, monkeypatch, seconds):
    monkeypatch.setattr(GATE, "_inventory", unreadable)
    with pytest.raises(ValueError, match="Saved gate time must be"):
        verify(tmp_path / "missing", tmp_path, max_seconds=seconds)


def test_duplicate_json_cannot_replace_false_claim():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        GATE._json(b'{"formal_proof_reproduced":false,"formal_proof_reproduced":true}')


def test_boolean_cannot_alias_fixed_case_count():
    with pytest.raises(ValueError, match="type differs"):
        GATE._fixed({"case_count": True}, {"case_count": 1}, "Saved case count")
