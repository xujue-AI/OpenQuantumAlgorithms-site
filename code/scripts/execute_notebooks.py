#!/usr/bin/env python3
"""Execute published notebooks using this interpreter in an isolated kernel spec."""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import stat
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PRESERVED_NOTEBOOK = "notebooks/optimization_dqi_degree_two.ipynb"
PRESERVED_NOTEBOOK_BYTES = 19950
PRESERVED_NOTEBOOK_SHA256 = "e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9"


def verify_preserved_notebook(path):
    """Admit one accepted notebook by its exact path and frozen raw bytes."""
    if not isinstance(path, Path) or not path.is_absolute() or path != ROOT / PRESERVED_NOTEBOOK:
        raise ValueError("Preserved notebook requires its exact repository path")
    file_identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )
    directory_identity = lambda value: (value.st_dev, value.st_ino, value.st_mode)
    try:
        parents = {}
        for parent in path.parents:
            value = parent.lstat()
            if not stat.S_ISDIR(value.st_mode):
                raise ValueError("Preserved notebook parent is not an ordinary directory")
            parents[parent] = directory_identity(value)
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size != PRESERVED_NOTEBOOK_BYTES):
            raise ValueError("Preserved notebook must be a regular file with its frozen size")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0))
        try:
            opened = os.fstat(descriptor)
            if file_identity(opened) != file_identity(before):
                raise ValueError("Preserved notebook changed during open")
            digest = hashlib.sha256()
            size = 0
            while True:
                piece = os.read(descriptor, min(65536, PRESERVED_NOTEBOOK_BYTES + 1 - size))
                if not piece:
                    break
                size += len(piece)
                if size > PRESERVED_NOTEBOOK_BYTES:
                    raise ValueError("Preserved notebook exceeded its frozen read bound")
                digest.update(piece)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = path.lstat()
        if (file_identity(before) != file_identity(after)
                or file_identity(before) != file_identity(current)
                or size != PRESERVED_NOTEBOOK_BYTES
                or digest.hexdigest() != PRESERVED_NOTEBOOK_SHA256):
            raise ValueError("Preserved notebook bytes or file identity changed")
        for parent, identity in parents.items():
            if directory_identity(parent.lstat()) != identity:
                raise ValueError("Preserved notebook parent changed during read")
    except OSError as error:
        raise ValueError("Preserved notebook could not be admitted") from error


@contextmanager
def notebook_environment(path):
    bindings = {}
    if path.name == "simulation_detailed_balance_decay.ipynb":
        decay = ROOT / "assets/generated/simulation/detailed-balance-decay"
        bindings = {
            "OPENQUANTUM_SOURCE_ROOT": str(ROOT),
            "OPENQUANTUM_DECAY_RESULTS": str(decay / "grid/results.json"),
            "OPENQUANTUM_DECAY_MANIFEST_SHA256": "a77996282baad56fc8d15d44635d204dd21764fc09b57d557057c62783fb89de",
            "OPENQUANTUM_DECAY_SOURCE_RESULTS": str(decay / "n12/results.json"),
            "OPENQUANTUM_DECAY_SOURCE_MANIFEST_SHA256": "13f5f577a2cb419a140f3a5c7122eae769c1498eb7395a582be9b431ed7070a2",
        }
    previous = {name: os.environ.get(name) for name in bindings}
    try:
        os.environ.update(bindings)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main():
    preserved = ROOT / PRESERVED_NOTEBOOK
    if preserved.exists() or preserved.is_symlink():
        verify_preserved_notebook(preserved)
    with tempfile.TemporaryDirectory(prefix="openquantum-kernel-") as temporary:
        kernel = Path(temporary) / "kernels" / "openquantum"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "--matplotlib=inline", "-f", "{connection_file}"],
            "display_name": "Open Quantum Algorithms",
            "language": "python",
        }))
        previous_path = os.environ.get("JUPYTER_PATH")
        previous_static = os.environ.get("OPENQUANTUM_STATIC_BUILD")
        os.environ["JUPYTER_PATH"] = temporary + (os.pathsep + previous_path if previous_path else "")
        os.environ["OPENQUANTUM_STATIC_BUILD"] = "1"
        try:
            for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
                if path == preserved:
                    verify_preserved_notebook(path)
                    print("Preserved", path.relative_to(ROOT), flush=True)
                    continue
                with notebook_environment(path):
                    notebook = nbformat.read(path, as_version=4)
                    NotebookClient(notebook, timeout=120, kernel_name="openquantum",
                                   resources={"metadata": {"path": str(ROOT)}}).execute()
                    notebook.metadata["kernelspec"] = {
                        "display_name": "Python 3", "language": "python", "name": "python3"
                    }
                    nbformat.write(notebook, path)
                    print("Executed", path.relative_to(ROOT), flush=True)
        finally:
            if previous_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_path
            if previous_static is None:
                os.environ.pop("OPENQUANTUM_STATIC_BUILD", None)
            else:
                os.environ["OPENQUANTUM_STATIC_BUILD"] = previous_static


if __name__ == "__main__":
    main()

