"""Notebook environment contracts without importing SDKs or starting kernels."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMPANION = "simulation_detailed_balance_decay.ipynb"
KEYS = (
    "OPENQUANTUM_SOURCE_ROOT",
    "OPENQUANTUM_DECAY_RESULTS",
    "OPENQUANTUM_DECAY_MANIFEST_SHA256",
    "OPENQUANTUM_DECAY_SOURCE_RESULTS",
    "OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256",
)
OUTER_KEYS = ("JUPYTER_PATH", "OPENQUANTUM_STATIC_BUILD")


def environment():
    return {name: os.environ.get(name) for name in KEYS + OUTER_KEYS}


@pytest.fixture
def executor(tmp_path, monkeypatch):
    root = tmp_path / "book"
    notebooks = root / "notebooks"
    notebooks.mkdir(parents=True)
    for name in ("aa_before.ipynb", COMPANION, "zz_after.ipynb"):
        (notebooks / name).write_text("unchanged notebook bytes\n")
    observed = []
    written = []
    failure = {"stage": None}

    def read(path, *, as_version):
        assert as_version == 4
        if path.name == COMPANION and failure["stage"] == "read":
            raise RuntimeError("mock read failed")
        return SimpleNamespace(path=path, metadata={})

    def write(notebook, path):
        if path.name == COMPANION and failure["stage"] == "write":
            raise RuntimeError("mock write failed")
        written.append(path.name)

    class Client:
        def __init__(self, notebook, *, timeout, kernel_name, resources):
            assert timeout == 120
            assert kernel_name == "openquantum"
            assert resources == {"metadata": {"path": str(root)}}
            self.path = notebook.path
            observed.append((self.path.name, environment()))

        def execute(self):
            if self.path.name == COMPANION and failure["stage"] == "execute":
                raise RuntimeError("mock execution failed")

    monkeypatch.setitem(sys.modules, "nbformat", SimpleNamespace(read=read, write=write))
    monkeypatch.setitem(sys.modules, "nbclient", SimpleNamespace(NotebookClient=Client))
    spec = importlib.util.spec_from_file_location(
        "_decay_notebook_executor_test", ROOT / "scripts/execute_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return SimpleNamespace(
        module=module, root=root, observed=observed, written=written, failure=failure
    )


def expected_bindings(root):
    decay = root / "assets/generated/simulation/detailed-balance-decay"
    return {
        "OPENQUANTUM_SOURCE_ROOT": str(root),
        "OPENQUANTUM_DECAY_RESULTS": str(decay / "grid/results.json"),
        "OPENQUANTUM_DECAY_MANIFEST_SHA256": "a77996282baad56fc8d15d44635d204dd21764fc09b57d557057c62783fb89de",
        "OPENQUANTUM_DECAY_SOURCE_RESULTS": str(decay / "n12/results.json"),
        "OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256": "13f5f577a2cb419a140f3a5c7122eae769c1498eb7395a582be9b431ed7070a2",
    }


def set_prior_environment(monkeypatch, mode):
    for name in KEYS + OUTER_KEYS:
        monkeypatch.delenv(name, raising=False)
    if mode == "mixed":
        monkeypatch.setenv("OPENQUANTUM_SOURCE_ROOT", "prior-source-root")
        monkeypatch.setenv("OPENQUANTUM_DECAY_RESULTS", "")
        monkeypatch.setenv("OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256", "prior-source-pin")
        monkeypatch.setenv("JUPYTER_PATH", "prior-kernel-path")
        monkeypatch.setenv("OPENQUANTUM_STATIC_BUILD", "prior-static-mode")
    return environment()


@pytest.mark.parametrize("prior_mode", ["absent", "mixed"])
def test_decay_client_gets_exact_bindings_and_other_notebooks_keep_prior_values(
    executor, monkeypatch, prior_mode
):
    prior = set_prior_environment(monkeypatch, prior_mode)
    executor.module.main()
    assert [name for name, _ in executor.observed] == [
        "aa_before.ipynb", COMPANION, "zz_after.ipynb"
    ]
    for name, seen in executor.observed:
        assert {key: seen[key] for key in KEYS} == (
            expected_bindings(executor.root) if name == COMPANION
            else {key: prior[key] for key in KEYS}
        )
        assert seen["OPENQUANTUM_STATIC_BUILD"] == "1"
        assert seen["JUPYTER_PATH"]
        if prior["JUPYTER_PATH"] is not None:
            assert seen["JUPYTER_PATH"].endswith(os.pathsep + prior["JUPYTER_PATH"])
    assert executor.written == ["aa_before.ipynb", COMPANION, "zz_after.ipynb"]
    assert environment() == prior
    assert all(path.read_text() == "unchanged notebook bytes\n"
               for path in (executor.root / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("stage", ["read", "execute", "write"])
@pytest.mark.parametrize("prior_mode", ["absent", "mixed"])
def test_decay_and_outer_environments_restore_after_notebook_failure(
    executor, monkeypatch, stage, prior_mode
):
    prior = set_prior_environment(monkeypatch, prior_mode)
    executor.failure["stage"] = stage
    with pytest.raises(RuntimeError, match="mock .* failed"):
        executor.module.main()
    assert environment() == prior
    assert "zz_after.ipynb" not in executor.written
    assert all(path.read_text() == "unchanged notebook bytes\n"
               for path in (executor.root / "notebooks").glob("*.ipynb"))
