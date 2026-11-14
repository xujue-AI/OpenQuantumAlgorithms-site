#!/usr/bin/env python3
"""Recompute Grover's physical model without overwriting its frozen artifact."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def physical_record(record: dict) -> dict:
    """Drop only importer wall time and inactive rotation-search bookkeeping."""
    if (record.get("arbitrary_rotation_synthesis_required") is not False
            or record.get("canonical_required_instructions") != ["T", "MEAS_Z"]
            or record.get("canonical_application_trace") !=
            "@compute_qubits(3)\n{\n  repeat 28 {\n    T(0)\n  }\n  repeat 3 {\n    MEAS_Z(0)\n  }\n}\n"):
        raise ValueError("Frozen comparison requires the exact rotation-free Grover trace")
    value = copy.deepcopy(record)
    for profile in value["profiles"]:
        for point in profile["pareto_points"]:
            properties = point["properties"]
            properties.pop("EVALUATION_TIME", None)
            # With no RZ operations, this searched parameter is inactive.
            # All physical quantities and source graphs are still compared.
            properties.pop("NUM_TS_PER_ROTATION", None)
    return value


def compare_records(frozen: dict, fresh: dict) -> None:
    if physical_record(frozen) != physical_record(fresh):
        raise ValueError("Recomputed Grover physical results differ from the frozen baseline")


def main() -> None:
    baseline = ROOT / "assets/generated/resources"
    with tempfile.TemporaryDirectory(prefix="openquantum-baseline-") as temporary:
        output = Path(temporary)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/estimate_resources.py"),
             "--output-dir", str(output)], cwd=ROOT, check=True,
        )
        compare_records(
            json.loads((baseline / "grover-surface-code.json").read_text()),
            json.loads((output / "grover-surface-code.json").read_text()),
        )
        # CSV newline normalization is already part of the publication recipe.
        if ((baseline / "grover-surface-code.csv").read_text()
                != (output / "grover-surface-code.csv").read_text()):
            raise SystemExit("Recomputed physical Pareto CSV differs from the frozen baseline")
    print("Recomputed all nine physical points; frozen Grover artifacts preserved")


if __name__ == "__main__":
    main()
