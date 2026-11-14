"""Publication removes runtime caches while preserving reproducible inputs."""
import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_assets", ROOT / "scripts/public_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copy_and_prune_preserve_all_scientific_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = ("module.py", "trial.qasm", "record.json", "results.csv", "figure.svg",
             "notebook.ipynb", "references.bib", "requirements-lock.txt", "C4Reference.lean")
    for name in names:
        (source / name).write_text("original scientific bytes: " + name)
    expected = {name: digest(source / name) for name in names}
    for name in assets.RUNTIME_NAMES:
        path = source / name
        if name == ".DS_Store":
            path.write_text("runtime")
        else:
            path.mkdir()
            (path / "cached.pyc").write_text("runtime")
    (source / "loose.pyo").write_text("runtime")
    original_source = {p.relative_to(source): digest(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "public"
    shutil.copytree(source, output, ignore=assets.ignore_runtime_artifacts)
    assert {p.name: digest(p) for p in output.iterdir()} == expected
    # Also handle caches already emitted by the upstream static renderer.
    (output / "__pycache__").mkdir()
    (output / "__pycache__/cached.pyc").write_text("runtime")
    (output / ".DS_Store").write_text("runtime")
    assets.remove_runtime_artifacts(output)
    assert {p.name: digest(p) for p in output.iterdir()} == expected
    assert {p.relative_to(source): digest(p) for p in source.rglob("*") if p.is_file()} == original_source
    assets.remove_runtime_artifacts(output)
    assert {p.name: digest(p) for p in output.iterdir()} == expected


def test_cache_symlink_is_removed_without_touching_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "original.py"
    retained.write_text("retained")
    output = tmp_path / "public"
    output.mkdir()
    (output / "__pycache__").symlink_to(outside, target_is_directory=True)
    assets.remove_runtime_artifacts(output)
    assert retained.read_text() == "retained"
    assert not (output / "__pycache__").is_symlink()


def test_output_root_symlink_rejects_before_mutation(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "cached.pyc"
    retained.write_text("retained")
    link = tmp_path / "public"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        assets.remove_runtime_artifacts(link)
    assert retained.read_text() == "retained"
