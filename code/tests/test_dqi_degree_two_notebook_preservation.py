"""Exact DQI companion preservation with mocked notebook libraries only."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRESERVED = "notebooks/optimization_dqi_degree_two.ipynb"
PRESERVED_NAME = "optimization_dqi_degree_two.ipynb"
PRESERVED_BYTES = 19_950
PRESERVED_SHA256 = "e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9"
DECAY_NAME = "simulation_detailed_balance_decay.ipynb"
ORDINARY = ("aa_before.ipynb", DECAY_NAME, "zz_after.ipynb")
DECAY_KEYS = (
    "OPENQUANTUM_SOURCE_ROOT",
    "OPENQUANTUM_DECAY_RESULTS",
    "OPENQUANTUM_DECAY_MANIFEST_SHA256",
    "OPENQUANTUM_DECAY_SOURCE_RESULTS",
    "OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256",
)
OUTER_KEYS = ("JUPYTER_PATH", "OPENQUANTUM_STATIC_BUILD")


def environment():
    return {name: os.environ.get(name) for name in DECAY_KEYS + OUTER_KEYS}


def prior_environment(monkeypatch, mode):
    for name in DECAY_KEYS + OUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    if mode == "mixed":
        monkeypatch.setenv("OPENQUANTUM_SOURCE_ROOT", "prior-source")
        monkeypatch.setenv("OPENQUANTUM_DECAY_RESULTS", "")
        monkeypatch.setenv("OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256", "prior-pin")
        monkeypatch.setenv("JUPYTER_PATH", "prior-kernel-path")
        monkeypatch.setenv("OPENQUANTUM_STATIC_BUILD", "prior-static")
    return environment()


def decay_bindings(root):
    directory = root / "assets/generated/simulation/detailed-balance-decay"
    return {
        "OPENQUANTUM_SOURCE_ROOT": str(root),
        "OPENQUANTUM_DECAY_RESULTS": str(directory / "grid/results.json"),
        "OPENQUANTUM_DECAY_MANIFEST_SHA256": "a77996282baad56fc8d15d44635d204dd21764fc09b57d557057c62783fb89de",
        "OPENQUANTUM_DECAY_SOURCE_RESULTS": str(directory / "n12/results.json"),
        "OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256": "13f5f577a2cb419a140f3a5c7122eae769c1498eb7395a582be9b431ed7070a2",
    }


@pytest.fixture(scope="module")
def accepted_bytes():
    raw = (ROOT / PRESERVED).read_bytes()
    assert len(raw) == PRESERVED_BYTES
    assert hashlib.sha256(raw).hexdigest() == PRESERVED_SHA256
    return raw


@pytest.fixture
def executor(tmp_path, monkeypatch, accepted_bytes):
    root = (tmp_path / "book").resolve()
    directory = root / "notebooks"
    directory.mkdir(parents=True)
    target = root / PRESERVED
    target.write_bytes(accepted_bytes)
    for name in ORDINARY:
        (directory / name).write_bytes(b"ordinary mocked notebook\n")
    events, observed, written = [], [], []
    controls = {"failure": None, "on_read": None}

    def read(path, *, as_version):
        assert as_version == 4
        assert path != target, "The preserved companion reached nbformat.read"
        events.append(("read", path.name))
        if controls["on_read"] is not None:
            controls["on_read"](path)
        if path.name == DECAY_NAME and controls["failure"] == "read":
            raise RuntimeError("mock read failed")
        return SimpleNamespace(path=path, metadata={})

    class Client:
        def __init__(self, notebook, *, timeout, kernel_name, resources):
            assert notebook.path != target, "The preserved companion reached NotebookClient"
            assert timeout == 120 and kernel_name == "openquantum"
            assert resources == {"metadata": {"path": str(root)}}
            self.path = notebook.path
            events.append(("client", self.path.name))
            observed.append((self.path.name, environment()))

        def execute(self):
            events.append(("execute", self.path.name))
            if self.path.name == DECAY_NAME and controls["failure"] == "execute":
                raise RuntimeError("mock execute failed")

    def write(notebook, path):
        assert path != target, "The preserved companion reached nbformat.write"
        events.append(("write", path.name))
        if path.name == DECAY_NAME and controls["failure"] == "write":
            raise RuntimeError("mock write failed")
        written.append(path.name)

    monkeypatch.setitem(sys.modules, "nbformat", SimpleNamespace(read=read, write=write))
    monkeypatch.setitem(sys.modules, "nbclient", SimpleNamespace(NotebookClient=Client))
    spec = importlib.util.spec_from_file_location(
        "_dqi_preservation_mock_executor", ROOT / "scripts/execute_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return SimpleNamespace(
        module=module, root=root, target=target, accepted=accepted_bytes,
        events=events, observed=observed, written=written, controls=controls,
    )


def assert_preserved(executor):
    assert executor.target.read_bytes() == executor.accepted
    assert all(name != PRESERVED_NAME for _, name in executor.events)


def test_policy_binds_one_exact_repository_path_size_and_raw_digest(executor):
    assert executor.module.PRESERVED_NOTEBOOK == PRESERVED
    assert executor.module.PRESERVED_NOTEBOOK_BYTES == PRESERVED_BYTES
    assert executor.module.PRESERVED_NOTEBOOK_SHA256 == PRESERVED_SHA256
    executor.module.verify_preserved_notebook(executor.target)
    assert executor.events == []
    assert_preserved(executor)


@pytest.mark.parametrize("prior_mode", ["absent", "mixed"])
def test_exact_companion_stays_untouched_while_other_notebooks_refresh(
    executor, monkeypatch, capsys, prior_mode
):
    prior = prior_environment(monkeypatch, prior_mode)
    executor.module.main()
    assert [name for name, _ in executor.observed] == list(ORDINARY)
    assert executor.written == list(ORDINARY)
    assert executor.events == [(stage, name) for name in ORDINARY
                               for stage in ("read", "client", "execute", "write")]
    for name, seen in executor.observed:
        assert {key: seen[key] for key in DECAY_KEYS} == (
            decay_bindings(executor.root) if name == DECAY_NAME
            else {key: prior[key] for key in DECAY_KEYS}
        )
        assert seen["OPENQUANTUM_STATIC_BUILD"] == "1"
        assert seen["JUPYTER_PATH"]
        if prior["JUPYTER_PATH"] is not None:
            assert seen["JUPYTER_PATH"].endswith(os.pathsep + prior["JUPYTER_PATH"])
    assert environment() == prior
    assert_preserved(executor)
    output = capsys.readouterr().out
    assert "Preserved " + PRESERVED in output
    assert "Executed " + PRESERVED not in output


@pytest.mark.parametrize("mutation", ["same-size", "larger", "truncated", "empty"])
def test_invalid_preserved_bytes_fail_before_any_notebook_library_call(executor, mutation):
    raw = executor.accepted
    replacements = {
        "same-size": b"!" + raw[1:], "larger": raw + b"\n",
        "truncated": raw[:-1], "empty": b"",
    }
    executor.target.write_bytes(replacements[mutation])
    with pytest.raises(ValueError):
        executor.module.main()
    assert executor.events == []


def test_missing_preserved_file_is_rejected_by_exact_admission(executor):
    executor.target.unlink()
    with pytest.raises(ValueError):
        executor.module.verify_preserved_notebook(executor.target)
    assert executor.events == []


@pytest.mark.parametrize("where", ["other-directory", "relative", "parent-alias"])
def test_same_basename_or_aliased_path_cannot_be_admitted(executor, where):
    if where == "other-directory":
        candidate = executor.root / "elsewhere" / PRESERVED_NAME
        candidate.parent.mkdir()
        candidate.write_bytes(executor.accepted)
    elif where == "relative":
        candidate = Path(PRESERVED)
    else:
        candidate = executor.root / "notebooks" / ".." / "notebooks" / PRESERVED_NAME
    with pytest.raises(ValueError):
        executor.module.verify_preserved_notebook(candidate)
    assert executor.events == []
    assert_preserved(executor)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_nonordinary_preserved_member_is_rejected_before_libraries(executor, kind):
    other = executor.root / "original.ipynb"
    other.write_bytes(executor.accepted)
    executor.target.unlink()
    if kind == "symlink":
        executor.target.symlink_to(other)
    elif kind == "hardlink":
        os.link(other, executor.target)
    else:
        executor.target.mkdir()
    with pytest.raises(ValueError):
        executor.module.main()
    assert executor.events == []


def test_symlinked_notebook_parent_is_rejected_before_libraries(executor):
    original = executor.root / "notebooks"
    moved = executor.root / "original-notebooks"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError):
        executor.module.main()
    assert executor.events == []


def test_file_replacement_during_open_is_rejected_even_with_identical_bytes(executor, monkeypatch):
    replacement = executor.root / "replacement.ipynb"
    replacement.write_bytes(executor.accepted)
    original_open = executor.module.os.open
    swapped = False

    def replace_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == executor.target:
            swapped = True
            executor.target.unlink()
            replacement.rename(executor.target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(executor.module.os, "open", replace_then_open)
    with pytest.raises(ValueError):
        executor.module.verify_preserved_notebook(executor.target)
    assert swapped and executor.events == []


@pytest.mark.parametrize("change", ["contents", "path"])
def test_change_during_bounded_read_is_rejected_before_libraries(executor, monkeypatch, change):
    replacement = executor.root / "replacement.ipynb"
    replacement.write_bytes(executor.accepted)
    original_read = executor.module.os.read
    changed = False

    def change_after_read(descriptor, size):
        nonlocal changed
        raw = original_read(descriptor, size)
        if raw and not changed:
            changed = True
            if change == "contents":
                executor.target.write_bytes(b"!" + executor.accepted[1:])
            else:
                executor.target.unlink()
                replacement.rename(executor.target)
        return raw

    monkeypatch.setattr(executor.module.os, "read", change_after_read)
    with pytest.raises(ValueError):
        executor.module.verify_preserved_notebook(executor.target)
    assert changed and executor.events == []


@pytest.mark.parametrize("prior_mode", ["absent", "mixed"])
@pytest.mark.parametrize("change", ["contents", "missing", "replacement"])
def test_preserved_file_is_rechecked_after_preflight_and_environment_restores(
    executor, monkeypatch, change, prior_mode
):
    prior = prior_environment(monkeypatch, prior_mode)
    changed = False

    def mutate_after_preflight(path):
        nonlocal changed
        if path.name == ORDINARY[0] and not changed:
            changed = True
            if change == "contents":
                executor.target.write_bytes(b"!" + executor.accepted[1:])
            elif change == "missing":
                executor.target.unlink()
            else:
                replacement = executor.root / "replacement.ipynb"
                replacement.write_bytes(b"!" + executor.accepted[1:])
                executor.target.unlink()
                replacement.rename(executor.target)

    executor.controls["on_read"] = mutate_after_preflight
    with pytest.raises(ValueError):
        executor.module.main()
    assert changed
    assert executor.written == [ORDINARY[0]]
    assert all(name != PRESERVED_NAME for _, name in executor.events)
    assert environment() == prior


@pytest.mark.parametrize("prior_mode", ["absent", "mixed"])
@pytest.mark.parametrize("stage", ["read", "execute", "write"])
def test_ordinary_notebook_failure_keeps_preserved_bytes_and_restores_environment(
    executor, monkeypatch, stage, prior_mode
):
    prior = prior_environment(monkeypatch, prior_mode)
    executor.controls["failure"] = stage
    with pytest.raises(RuntimeError, match="mock .* failed"):
        executor.module.main()
    assert environment() == prior
    assert "zz_after.ipynb" not in executor.written
    assert_preserved(executor)


def test_root_without_preserved_companion_keeps_existing_refresh_behavior(executor):
    executor.target.unlink()
    executor.module.main()
    assert [name for name, _ in executor.observed] == list(ORDINARY)
    assert executor.written == list(ORDINARY)


def test_nearby_filename_is_refreshed_instead_of_being_broadly_skipped(executor):
    extra = executor.root / "notebooks/optimization_dqi_degree_two_extra.ipynb"
    extra.write_bytes(executor.accepted)
    executor.module.main()
    assert extra.name in executor.written
    assert ("execute", extra.name) in executor.events
    assert_preserved(executor)
