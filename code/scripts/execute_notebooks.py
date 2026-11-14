#!/usr/bin/env python3
"""Execute published notebooks using this interpreter in an isolated kernel spec."""
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
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

