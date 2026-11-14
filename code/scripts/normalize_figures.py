#!/usr/bin/env python3
"""Normalize insignificant figure/CSV whitespace after canonical generation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMMUTABLE_SVGS = frozenset({
    "assets/generated/optimization/compiled-replay/workload.svg",
    "assets/generated/shor/measured-and/gadget.svg",
    "assets/generated/shor/measured-and/addition.svg",
    "assets/generated/simulation/reusable-bath-local-spin-presentation/first-step.svg",
    "assets/generated/hadamard-qpe/circuit.svg",
    "assets/generated/hadamard-qpe/recovery.svg",
    "assets/generated/simulation/reusable-bath/fixture-ground.svg",
    "assets/generated/simulation/reusable-bath/fixture-thermal.svg",
    "assets/generated/simulation/reusable-bath/endpoints.svg",
    "assets/generated/simulation/multiproduct/convergence.svg",
    "assets/generated/simulation/multiproduct/small-lcu-circuit.svg",
    "assets/generated/bayesian-qpe/circuit.svg",
    "assets/generated/bayesian-qpe/posterior.svg",
    "assets/generated/bayesian-qpe-validation/replay/circuit.svg",
    "assets/generated/bayesian-qpe-validation/replay/posterior.svg",
    "assets/generated/robust-qpe/circuit.svg",
    "assets/generated/robust-qpe/recovery.svg",
})
IMMUTABLE_CSVS = frozenset({
    "assets/generated/shor/measured-and/scaling.csv",
    "assets/generated/bayesian-qpe/summary.csv",
    "assets/generated/bayesian-qpe-validation/replay/summary.csv",
    "assets/generated/robust-qpe/scaling.csv",
    "assets/generated/qpe-feedback-resources/summary.csv",
})
for path in (ROOT / "assets/generated").rglob("*"):
    if path.relative_to(ROOT).as_posix() in IMMUTABLE_SVGS:
        continue
    if path.relative_to(ROOT).as_posix() in IMMUTABLE_CSVS:
        continue
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    elif path.suffix == ".csv":
        # Text-mode reading converts CRLF; cell values are left untouched.
        path.write_text(path.read_text())
print("Normalized generated SVG/CSV whitespace")
