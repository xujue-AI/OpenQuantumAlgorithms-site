#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Draw one literal step from the checked local-spin QASM; no simulation runs.

The figure is a presentation excerpt: three initial resets and the first
nineteen unitary instructions of the saved n=2 ground-mode program. Qiskit
2.6.2 supplies its Apache-2.0 circuit drawer; the circuit data are original.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "assets/generated/simulation/reusable-bath-local-spin/source-exports/n2-ground.qasm"
SOURCE_BYTES = 10058
SOURCE_SHA256 = "6191f3555e5c23ac9f0d82afd29b7c2e45fe6baedaa0dfe21e2a0c80bc703eb2"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
HEADER = ('OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[3];', 'creg c[2];')


def parse_excerpt(raw: bytes):
    """Authenticate the stored source and parse only its first-step prefix."""
    if (not isinstance(raw, bytes) or len(raw) != SOURCE_BYTES
            or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256):
        raise ValueError("Saved n2-ground QASM identity changed")
    lines = raw.decode("ascii").splitlines()
    if tuple(lines[:4]) != HEADER:
        raise ValueError("Saved excerpt register/header changed")
    prefix = lines[4:26]
    if len(prefix) != 22 or prefix[:3] != [f"reset q[{i}];" for i in range(3)]:
        raise ValueError("Saved excerpt reset/step boundary changed")
    operations = []
    for line in prefix:
        match = re.fullmatch(r"(reset|h) q\[([0-2])\];", line)
        if match:
            operations.append((match[1], (int(match[2]),), None))
            continue
        match = re.fullmatch(r"cx q\[([0-2])\],q\[([0-2])\];", line)
        if match and match[1] != match[2]:
            operations.append(("cx", (int(match[1]), int(match[2])), None))
            continue
        match = re.fullmatch(r"rz\(([+\-0-9.eE]+)\) q\[([0-2])\];", line)
        if match:
            angle = float(match[1])
            if math.isfinite(angle):
                operations.append(("rz", (int(match[2]),), angle))
                continue
        raise ValueError("Unsupported saved excerpt instruction")
    counts = dict(Counter(name for name, _, _ in operations))
    if counts != {"reset": 3, "rz": 9, "cx": 6, "h": 4}:
        raise ValueError("Saved excerpt has the wrong first-step workload")
    return tuple(operations)


def render(root: Path, output: Path, *, png: bool = False):
    if sys.version_info[:3] != (3, 12, 14):
        raise ValueError("Use the locked Python 3.12.14 environment")
    source = root / SOURCE
    if source.is_symlink() or not source.is_file():
        raise ValueError("Saved QASM must be a regular file")
    with source.open("rb") as handle:
        raw = handle.read(SOURCE_BYTES + 1)
    operations = parse_excerpt(raw)
    import importlib.metadata
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "openquantum-local-spin-excerpt"
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit
    if importlib.metadata.version("qiskit") != "2.6.2":
        raise ValueError("Use the pinned Qiskit 2.6.2 drawer")
    circuit = QuantumCircuit(3, 2)
    for name, qubits, angle in operations:
        if name == "rz":
            circuit.rz(angle, qubits[0])
        else:
            getattr(circuit, name)(*qubits)
    figure = circuit.draw(output="mpl", fold=24, scale=.85, cregbundle=False)
    streams = {}
    try:
        for format_name in (("svg", "png") if png else ("svg",)):
            stream = io.BytesIO()
            options = {"metadata": {"Date": None}} if format_name == "svg" else {}
            figure.savefig(stream, format=format_name, bbox_inches="tight", **options)
            streams["first-step." + format_name] = stream.getvalue()
    finally:
        plt.close(figure)
    record = {
        "kind": "presentation_excerpt_from_saved_qasm",
        "source_file": SOURCE, "source_sha256": SOURCE_SHA256,
        "system_wires": 2, "bath_wire": 2, "classical_bits": 2,
        "initial_resets": 3, "unitary_instructions": 19,
        "counts": dict(Counter(name for name, _, _ in operations)),
        "scope": "First Strang step of the saved n2-ground source; rounded drawer labels",
        "new_source_generation": False, "new_quantum_or_reference_execution": False,
        "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "matplotlib")},
        "figure_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in streams.items()},
    }
    streams["presentation.json"] = (json.dumps(record, indent=2) + "\n").encode()
    if sum(map(len, streams.values())) > MAX_OUTPUT_BYTES:
        raise ValueError("Presentation outputs exceed 8 MiB")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in streams.items():
        with (output / name).open("xb") as handle:
            handle.write(data)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/local-spin-circuit")
    parser.add_argument("--png", action="store_true", help="also write a PNG for visual inspection")
    args = parser.parse_args()
    print(json.dumps(render(args.root.resolve(), args.output_dir, png=args.png), indent=2))


if __name__ == "__main__":
    main()
