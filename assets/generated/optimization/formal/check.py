#!/usr/bin/env python3
"""Recheck and export the finite classical Lean artifact without installing tools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


THEOREMS = (
    "all_cuts_le_four",
    "optimal_iff",
    "five_is_optimal",
    "ten_is_optimal",
    "marked_iff",
)
EXPECTED_COMMIT = "6a10ac8c22beadecabdbb0919c2b50214762f91d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elan", default="elan", help="Existing elan executable")
    parser.add_argument("--output", type=Path, required=True, help="Artifact export directory")
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    toolchain = (source_dir / "lean-toolchain").read_text().strip()
    pinned_version = toolchain.rsplit(":v", 1)[1]
    elan = shutil.which(args.elan)
    if elan is None:
        raise SystemExit("Elan executable not found; this check does not install a toolchain.")
    # Unlike the +toolchain proxy shorthand, `elan run` without --install
    # fails if the requested toolchain is absent, so this never downloads it.
    base_command = [elan, "run", toolchain, "lean"]
    version_check = subprocess.run(
        [*base_command, "--version"], cwd=source_dir, text=True,
        capture_output=True, timeout=60, check=True,
    )
    version = version_check.stdout.strip()
    if f"version {pinned_version}," not in version or EXPECTED_COMMIT not in version:
        raise SystemExit(f"Unexpected pinned Lean build: {version}")
    proof_check = subprocess.run(
        [*base_command, "C4Reference.lean"], cwd=source_dir, text=True,
        capture_output=True, timeout=60,
    )
    transcript = proof_check.stdout + proof_check.stderr
    if proof_check.returncode != 0:
        print(transcript, end="", file=sys.stderr)
        raise SystemExit(proof_check.returncode)
    expected_lines = [
        f"'OptimizationReference.{name}' does not depend on any axioms"
        for name in THEOREMS
    ]
    if transcript.splitlines() != expected_lines:
        raise SystemExit(f"Unexpected theorem dependency audit:\n{transcript}")
    files = ("C4Reference.lean", "lean-toolchain", "check.py", "LICENSE.txt", "README.md")
    hashes = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in files
    }
    result = {
        "schema": "optimization-finite-classical-formal-v1",
        "status": "checked",
        "scope": "Encoded finite classical C4 MaxCut reference and known marked vertex only",
        "exclusions": [
            "Equivalence to the Python implementation",
            "Quantum circuit semantics and uncomputation",
            "Adiabatic theorem, spectral gap, or runtime guarantee",
            "Quantum-walk success probability",
            "Sampling confidence and fault-tolerant resource models",
        ],
        "assignment_domain": "Fin 16; vertex 0 is the least significant bit",
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "toolchain": toolchain,
        "lean_version_output": version,
        "upstream_source": f"https://github.com/leanprover/lean4/tree/{EXPECTED_COMMIT}",
        "upstream_license": "Apache-2.0",
        "upstream_license_url": f"https://github.com/leanprover/lean4/blob/{EXPECTED_COMMIT}/LICENSE",
        "original_artifact_license": "MIT",
        "python_version": sys.version.split()[0],
        "check_command": f"elan run {toolchain} lean C4Reference.lean",
        "exit_code": proof_check.returncode,
        "kernel_checked_closed_propositions": True,
        "native_decide_used": False,
        "axioms_by_theorem": {name: [] for name in THEOREMS},
        "source_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name in files:
        destination = args.output / name
        if destination.resolve() != (source_dir / name).resolve():
            shutil.copyfile(source_dir / name, destination)
    (args.output / "check.log").write_text(transcript)
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.md").write_text(
        "**Classical C4 reference coverage only.** "
        "Lean 4 `v4.34.0-rc2` checked the original finite classical reference: "
        "every encoded C4 assignment has cut at most four, and equality holds "
        "exactly at integers `5` and `10`. The saved dependency audit reports "
        "no axioms for all five declarations. This coverage excludes Python "
        "equivalence, quantum circuits, annealing and walk dynamics, sampling, "
        "and physical estimates.\n"
    )
    print(f"Checked {len(THEOREMS)} declarations with {toolchain}; dependency audits empty.")


if __name__ == "__main__":
    main()
