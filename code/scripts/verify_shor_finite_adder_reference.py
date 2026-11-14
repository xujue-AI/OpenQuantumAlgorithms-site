#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate curated finite-adder metadata; never run Lean or parse QASM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets/generated/shor/finite-adder-reference"
MANIFEST_SHA256 = "a1b76407144da82df6065dd666e85da106c3e404dbd3af25b05b6e81e78164eb"
FILES = frozenset({"RippleTwoBitBool.lean", "lean-toolchain", "axiom-audit.txt",
                   "ripple-two-bit-source.qasm", "compilation.json", "README.md", "LICENSE.txt"})
AUDIT_NAMES = ("ripple_bits", "weighted_lt32", "ofState_toState", "toState_ofState",
               "input_values", "decode_ofState", "ripple_two_bit", "ripple_two_bit_carry_one")
PROPEXT_NAMES = frozenset({"ripple_bits", "ripple_two_bit", "ripple_two_bit_carry_one"})
EXPECTED_SCOPE = {"input_domain": "Fin 32", "input_count": 32,
                  "quantum_wires_in_native_source": 5, "native_source_instructions": 12,
                  "wire_order": {"a": [0, 1], "b": [2, 3], "carry": [4]},
                  "mapping": "(a,b,c) -> (a,(a+b+c) mod 4,c)",
                  "carry_in_one_included": True,
                  "proof_model": "finite classical Boolean-wire permutation"}
EXCLUSIONS = ["Correctness of Python and the QASM parser",
              "Coherent quantum amplitudes and phases",
              "Arbitrary-width or modular arithmetic",
              "Whole Shor or order-recovery success",
              "Clifford+T lowering and rotation synthesis",
              "Physical fault tolerance and resource estimates"]
MAX_FILE_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 128 * 1024
PRIVATE_MARKERS = (b"/Users/", b"/private/", b"worker_0.job", b"merlin-ssh",
                   b"PAGES_DEPLOY_KEY", b".grant.json", b"source-freeze")


def _read(path: Path) -> bytes:
    if path.is_symlink() or path != path.resolve():
        raise ValueError("A public reference file redirects")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_FILE_BYTES:
        raise ValueError("A public reference file is not ordinary and bounded")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    after = path.stat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if key(before) != key(after) or len(raw) != before.st_size or len(raw) > MAX_FILE_BYTES:
        raise ValueError("A public reference file changed during read")
    if any(marker in raw for marker in PRIVATE_MARKERS):
        raise ValueError("The scientific public projection contains an operational marker")
    return raw


def validate_record(value: dict, audit: bytes) -> None:
    if value.get("schema") != "openquantum.shor.finite_adder_public_reference.v1" or value.get("status") != "accepted_finite_classical_reference":
        raise ValueError("The record is not the accepted finite reference")
    if value.get("scope") != EXPECTED_SCOPE:
        raise ValueError("The finite classical scope differs")
    lean = value["lean"]
    if lean.get("version_subprocess_exit") != 0 or lean.get("proof_subprocess_exit") != 0:
        raise ValueError("The selected kernel compilation was not successful")
    if lean.get("toolchain") != "leanprover/lean4:v4.34.0-rc2" or lean.get("source_commit") != "6a10ac8c22beadecabdbb0919c2b50214762f91d":
        raise ValueError("The selected Lean/Std source differs")
    foundations = {name: (["propext"] if name in PROPEXT_NAMES else []) for name in AUDIT_NAMES}
    if lean.get("foundation_axioms_by_theorem") != foundations or lean.get("audit_order") != list(AUDIT_NAMES):
        raise ValueError("The explicit foundation inventory differs")
    if lean.get("user_declared_domain_axioms") != [] or lean.get("sorryAx_present") is not False or lean.get("native_decide_used") is not False:
        raise ValueError("An unaccepted proof assumption or execution route appears")
    expected = []
    for name in AUDIT_NAMES:
        suffix = "depends on axioms: [propext]" if name in PROPEXT_NAMES else "does not depend on any axioms"
        expected.append(f"'ShorRippleBooleanReference.{name}' {suffix}")
    if audit.decode("utf-8").splitlines() != expected:
        raise ValueError("The eight recorded scientific audits differ")
    helper = value["original_helper"]
    if helper.get("exit") != 1 or helper.get("helper_complete") is not False or helper.get("empty_audit_contract_satisfied") is not False:
        raise ValueError("The original helper rejection must remain explicit")
    native = value["native_source"]
    if native.get("literal_gate_list_match_observed") is not True or native.get("correspondence_check_exit") != 0 or native.get("parser_formally_verified") is not False:
        raise ValueError("The trusted native-source bridge differs")
    if value.get("projection_actions") != {"new_proof_run": False, "new_source_export": False, "new_quantum_simulation": False}:
        raise ValueError("The projection must not imply new scientific execution")
    if value.get("exclusions") != EXCLUSIONS:
        raise ValueError("The finite reference exclusions are incomplete")


def verify_reference(bundle: Path = BUNDLE) -> dict:
    bundle = Path(bundle).absolute()
    if bundle.is_symlink() or bundle != bundle.resolve() or not bundle.is_dir():
        raise ValueError("Use an ordinary public reference directory")
    if {p.name for p in bundle.iterdir()} != FILES | {"manifest.json"}:
        raise ValueError("The scientific public projection has unexpected members")
    raw_manifest = _read(bundle / "manifest.json")
    if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("The exact curated manifest changed")
    manifest = json.loads(raw_manifest)
    if manifest.get("schema") != "openquantum.shor.finite_adder_public_manifest.v1" or set(manifest["files"]) != FILES or manifest.get("private_operational_payload_included") is not False or manifest.get("new_scientific_execution") is not False:
        raise ValueError("The public manifest scope differs")
    data, total = {}, len(raw_manifest)
    for name in sorted(FILES):
        raw = _read(bundle / name)
        total += len(raw)
        if total > MAX_TOTAL_BYTES or {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != manifest["files"][name]:
            raise ValueError("A selected scientific identity differs")
        data[name] = raw
    value = json.loads(data["compilation.json"])
    validate_record(value, data["axiom-audit.txt"])
    for record, name in ((value["lean"]["proof_source"], "RippleTwoBitBool.lean"),
                         (value["lean"]["audit_transcript"], "axiom-audit.txt"),
                         (value["native_source"]["qasm"], "ripple-two-bit-source.qasm")):
        if record != manifest["files"][name]:
            raise ValueError("A record points to different selected scientific bytes")
    return {"status": "authenticated_recorded_finite_classical_reference",
            "members": len(FILES) + 1, "bytes": total,
            "foundation_dependencies": ["propext"], "helper_complete": False,
            "empty_audit_contract_satisfied": False,
            "lean_executed_by_this_check": False, "qasm_parsed_by_this_check": False,
            "quantum_work_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    print(json.dumps(verify_reference(args.bundle), sort_keys=True))


if __name__ == "__main__":
    main()
