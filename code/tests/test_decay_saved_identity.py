"""Pure saved-byte/structure checks; no scientific producer or backend."""
from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/verify_detailed_balance_decay.py"
spec = importlib.util.spec_from_file_location("_decay_saved_identity_tests", HELPER)
saved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saved)


@pytest.fixture
def packet(tmp_path):
    source = tmp_path.resolve() / "source"
    source.mkdir()
    for name in (*saved.SOURCE_RECORDS, saved.NOTEBOOK_RELATIVE):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    bundle = tmp_path.resolve() / "bundle"
    shutil.copytree(ROOT / saved.BUNDLE_RELATIVE, bundle)
    return source, bundle


def test_saved_reader_authenticates_without_scientific_imports(packet, monkeypatch):
    original = builtins.__import__
    forbidden = {"openquantum", "qiskit", "numpy", "scipy", "qdk", "subprocess"}
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            pytest.fail("Saved checker reached an execution backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    source, bundle = packet
    report = saved.verify_bundle(bundle, source_root=source)
    assert report["status"] == "authenticated_saved_finite_decay_block"
    assert report["file_count"] == 28 and report["total_bytes"] == 483911
    assert report["historical_focused_tests"] == 70
    assert report["historical_notebook_code_cells"] == 5
    assert report["numerical_replay_performed"] is False
    assert report["physical_resources_computed"] is False


def test_helper_imports_are_stdlib_and_contains_no_execution_calls():
    tree = ast.parse(HELPER.read_text())
    roots = {node.module.split(".", 1)[0] for node in ast.walk(tree)
             if isinstance(node, ast.ImportFrom) and node.module}
    roots |= {alias.name.split(".", 1)[0] for node in ast.walk(tree)
              if isinstance(node, ast.Import) for alias in node.names}
    assert roots <= {"__future__", "argparse", "contextlib", "hashlib", "json", "math", "os", "pathlib", "stat"}
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"exec", "eval", "__import__"}


@pytest.mark.parametrize("name", sorted(saved.BUNDLE_FILES))
def test_frozen_member_mutation_is_rejected(packet, name):
    source, bundle = packet
    target = bundle / name
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.parametrize("name", sorted(saved.SOURCE_RECORDS))
def test_selected_source_mutation_is_rejected(packet, name):
    source, bundle = packet
    (source / name).write_bytes((source / name).read_bytes() + b"\n")
    with pytest.raises((ValueError, OSError)):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.parametrize("name", sorted(saved.BUNDLE_FILES))
def test_missing_frozen_member_is_rejected(packet, name):
    source, bundle = packet
    (bundle / name).unlink()
    with pytest.raises((ValueError, OSError)):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.parametrize("name,directory", [
    ("notes.txt", False), ("grid/extra.json", False), ("n12/private.log", False),
    ("notebook/new.ipynb", False), ("extra", True), ("grid/nested", True),
])
def test_unrecorded_files_and_directories_are_rejected(packet, name, directory):
    source, bundle = packet
    target = bundle / name
    if directory:
        target.mkdir()
    else:
        target.write_text("unreviewed")
    with pytest.raises((ValueError, OSError)):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.parametrize("where", ["bundle", "grid", "member", "source", "source-ancestor"])
def test_symlinked_packet_or_selected_source_is_rejected(packet, where):
    source, bundle = packet
    if where == "bundle":
        alias = bundle.parent / "bundle-alias"
        alias.symlink_to(bundle, target_is_directory=True)
        bundle = alias
    else:
        target = {"grid": bundle / "grid", "member": bundle / "completion.json",
                  "source": source / "requirements-lock.txt",
                  "source-ancestor": source / "src"}[where]
        retained = target.with_name(target.name + "-retained")
        target.rename(retained)
        target.symlink_to(retained, target_is_directory=retained.is_dir())
    with pytest.raises((ValueError, OSError)):
        saved.verify_bundle(bundle, source_root=source)


def test_hardlinked_member_is_rejected(packet):
    source, bundle = packet
    os.link(bundle / "completion.json", bundle.parent / "second-completion-link")
    with pytest.raises(ValueError, match="Nonregular|Unsafe"):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO boundary")
def test_fifo_member_is_rejected_without_reading(packet):
    source, bundle = packet
    target = bundle / "grid/results.json"
    target.unlink()
    os.mkfifo(target)
    with pytest.raises(ValueError, match="Nonregular"):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.parametrize("name", ["", "../outside", "grid/../completion.json", "/absolute", "./completion.json", "grid//results.json", "grid\\results.json", "bad\x00name", None])
def test_unsafe_relative_names_are_rejected_before_access(packet, name):
    _source, bundle = packet
    with pytest.raises(ValueError, match="Unsafe"):
        saved._read(bundle, name)


def test_file_read_cap_rejects_before_opening_target(packet, monkeypatch):
    _source, bundle = packet
    target = bundle / "completion.json"
    target.write_bytes(b"x" * (saved.MAX_FILE_BYTES + 1))
    original = os.open
    def guarded(name, *args, **kwargs):
        if name == "completion.json":
            pytest.fail("Oversized target was opened")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(saved.os, "open", guarded)
    with pytest.raises(ValueError, match="oversized"):
        saved._read(bundle, "completion.json")


def test_file_replacement_at_open_is_rejected(packet, monkeypatch):
    _source, bundle = packet
    target = bundle / "completion.json"
    original = os.open
    replaced = False
    def raced(name, *args, **kwargs):
        nonlocal replaced
        if name == "completion.json" and not replaced:
            replaced = True
            raw = target.read_bytes()
            target.rename(target.with_name("retained-completion"))
            target.write_bytes(raw)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(saved.os, "open", raced)
    with pytest.raises(ValueError, match="changed before open"):
        saved._read(bundle, "completion.json")


def test_file_mutation_during_read_is_rejected(packet, monkeypatch):
    _source, bundle = packet
    target = bundle / "completion.json"
    original = os.read
    mutated = False
    def raced(descriptor, size):
        nonlocal mutated
        result = original(descriptor, size)
        if result and not mutated:
            mutated = True
            with target.open("r+b") as stream:
                stream.write(b"!")
        return result
    monkeypatch.setattr(saved.os, "read", raced)
    with pytest.raises(ValueError, match="changed during"):
        saved._read(bundle, "completion.json")


def test_ancestor_replacement_at_open_is_rejected(packet, monkeypatch):
    _source, bundle = packet
    original = os.open
    replaced = False
    def raced(name, *args, **kwargs):
        nonlocal replaced
        if name == "grid" and not replaced:
            replaced = True
            (bundle / "grid").rename(bundle / "retained-grid")
            (bundle / "grid").mkdir()
        return original(name, *args, **kwargs)
    monkeypatch.setattr(saved.os, "open", raced)
    with pytest.raises(ValueError, match="directory changed before open"):
        saved._read(bundle, "grid/results.json")


def test_absolute_root_ancestor_swapped_to_symlink_is_rejected(packet, monkeypatch):
    source, _bundle = packet
    original = os.open
    replaced = False
    def raced(name, *args, **kwargs):
        nonlocal replaced
        if name == "source" and not replaced:
            replaced = True
            retained = source.with_name("retained-source")
            source.rename(retained)
            source.symlink_to(retained, target_is_directory=True)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(saved.os, "open", raced)
    with pytest.raises((ValueError, OSError)):
        saved._read(source, "requirements-lock.txt")


def test_inventory_is_closed_after_all_saved_reads(packet, monkeypatch):
    source, bundle = packet
    original = saved._bind_sources
    def after_binding(data, source_root):
        original(data, source_root)
        (bundle / "late-unrecorded.txt").write_text("unreviewed")
    monkeypatch.setattr(saved, "_bind_sources", after_binding)
    with pytest.raises(ValueError, match="Overlimit|Unrecorded"):
        saved.verify_bundle(bundle, source_root=source)


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e400}',
    b'{"a":', b'"\xff"', b"[" * 33 + b"0" + b"]" * 33,
    json.dumps([0] * saved.MAX_JSON_NODES, separators=(",", ":")).encode(),
    str(1 << 4097).encode(), b" " * (saved.MAX_JSON_BYTES + 1),
])
def test_duplicate_nonfinite_invalid_or_unbounded_json_is_rejected(raw):
    with pytest.raises(ValueError):
        saved._json(raw)


@pytest.mark.parametrize("value", [0, None, "false", True])
def test_scope_false_is_strictly_boolean(value):
    with pytest.raises(ValueError, match="false scope flag"):
        saved._false_flags({"nested": [{"physical_resources_computed": value}]})


def _write_repository_notebook(source, mutate):
    path = source / saved.NOTEBOOK_RELATIVE
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, separators=(",", ":")))


def test_fresh_repository_notebook_accepts_outputs_and_resegmented_same_code(packet):
    source, bundle = packet
    def mutate(notebook):
        count = 0
        notebook["cells"][0]["source"] = "A presentation-only introduction."
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                count += 1
                text = "".join(cell["source"])
                cell["source"] = [text[:17], text[17:]]
                cell["execution_count"] = count
                cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["new saved-reader output\n"]}]
    _write_repository_notebook(source, mutate)
    assert saved.verify_bundle(bundle, source_root=source)["numerical_replay_performed"] is False


@pytest.mark.parametrize("mutation", ["code", "id", "type", "source-type", "error-output", "nonmapping-cell"])
def test_repository_notebook_changes_outside_allowed_presentation_are_rejected(packet, mutation):
    source, bundle = packet
    def mutate(notebook):
        cell = notebook["cells"][1]
        if mutation == "code":
            cell["source"] = "print('different scientific source')"
        elif mutation == "id":
            cell["id"] = notebook["cells"][3]["id"]
        elif mutation == "type":
            cell["cell_type"] = "markdown"
        elif mutation == "source-type":
            cell["source"] = [7]
        elif mutation == "error-output":
            cell["outputs"] = [{"output_type": "error", "ename": "Error", "evalue": "failed", "traceback": []}]
        else:
            notebook["cells"][1] = "not a cell mapping"
    _write_repository_notebook(source, mutate)
    with pytest.raises(ValueError):
        saved.verify_bundle(bundle, source_root=source)
