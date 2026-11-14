import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_mismatched_resume_keeps_recorded_source_and_producer_unchanged(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "quality_resume_guard", ROOT / "scripts/generate_optimization_quality_scaling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "optimization.py"
    producer = tmp_path / "generate_optimization_quality_scaling.py"
    source.write_bytes(b"preserved recorded implementation\n")
    producer.write_bytes(b"preserved recorded producer\n")
    (tmp_path / "raw-study.json").write_text(json.dumps({
        "provenance": {"implementation_sha256": "mismatched", "generator_sha256": "mismatched"}
    }))
    monkeypatch.setattr(module, "preserve_recipe", lambda output: {"planned": {"runs": 72}})
    with pytest.raises(ValueError, match="matching executed"):
        module.generate(tmp_path, resume=True)
    assert source.read_bytes() == b"preserved recorded implementation\n"
    assert producer.read_bytes() == b"preserved recorded producer\n"
