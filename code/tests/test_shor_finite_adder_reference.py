"""Curated finite-adder metadata boundaries, without scientific replay.

Every temporary bundle contains only the eight small public files. Lean and
QASM remain inert bytes; importing the reader is deferred to test execution.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET_NAMES = (
    "LICENSE.txt", "README.md", "RippleTwoBitBool.lean", "axiom-audit.txt",
    "compilation.json", "lean-toolchain", "manifest.json",
    "ripple-two-bit-source.qasm",
)
THEOREMS = (
    "ripple_bits", "weighted_lt32", "ofState_toState", "toState_ofState",
    "input_values", "decode_ofState", "ripple_two_bit", "ripple_two_bit_carry_one",
)
PROPEXT_THEOREMS = {"ripple_bits", "ripple_two_bit", "ripple_two_bit_carry_one"}
FORBIDDEN_EXECUTION_IMPORTS = {
    "subprocess", "openquantum", "qiskit", "numpy", "scipy", "qdk", "lean",
}


def execution_import_guard(original):
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in FORBIDDEN_EXECUTION_IMPORTS:
            raise AssertionError("The saved-reference reader reached an execution backend")
        return original(name, *args, **kwargs)
    return guarded


@pytest.fixture(scope="module")
def reader():
    path = ROOT / "scripts/verify_shor_finite_adder_reference.py"
    specification = importlib.util.spec_from_file_location("shor_finite_reference_reader", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "__import__", execution_import_guard(builtins.__import__))
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload(reader):
    assert set(reader.FILES) | {"manifest.json"} == set(ASSET_NAMES)
    data = {name: (reader.BUNDLE / name).read_bytes() for name in ASSET_NAMES}
    assert sum(map(len, data.values())) <= 20_000
    return data


@pytest.fixture
def bundle(tmp_path, payload):
    destination = (tmp_path / "finite-reference").resolve()
    destination.mkdir()
    for name, raw in payload.items():
        (destination / name).write_bytes(raw)
    return destination


@pytest.fixture
def record_and_audit(payload):
    return json.loads(payload["compilation.json"]), payload["axiom-audit.txt"]


def test_closed_saved_reference_retains_accepted_scope(reader, bundle, record_and_audit):
    value, audit = record_and_audit
    result = reader.verify_reference(bundle)
    assert result["status"] == "authenticated_recorded_finite_classical_reference"
    assert result["members"] == 8 and result["bytes"] <= 20_000
    assert result["foundation_dependencies"] == ["propext"]
    assert result["helper_complete"] is False
    assert result["empty_audit_contract_satisfied"] is False
    assert result["lean_executed_by_this_check"] is False
    assert result["qasm_parsed_by_this_check"] is False
    assert result["quantum_work_performed"] is False
    assert value["scope"] == reader.EXPECTED_SCOPE
    assert value["scope"]["input_domain"] == "Fin 32"
    assert value["scope"]["input_count"] == 32
    assert value["scope"]["quantum_wires_in_native_source"] == 5
    assert value["scope"]["native_source_instructions"] == 12
    assert value["scope"]["wire_order"] == {"a": [0, 1], "b": [2, 3], "carry": [4]}
    assert value["scope"]["carry_in_one_included"] is True
    assert value["scope"]["proof_model"] == "finite classical Boolean-wire permutation"
    assert tuple(reader.AUDIT_NAMES) == THEOREMS
    assert value["lean"]["foundation_axioms_by_theorem"] == {
        name: (["propext"] if name in PROPEXT_THEOREMS else []) for name in THEOREMS
    }
    assert value["original_helper"]["exit"] == 1
    assert value["original_helper"]["helper_complete"] is False
    assert value["original_helper"]["empty_audit_contract_satisfied"] is False
    assert reader.validate_record(value, audit) is None


def test_metadata_reader_never_imports_execution_backends(reader, bundle, monkeypatch):
    monkeypatch.setattr(builtins, "__import__", execution_import_guard(builtins.__import__))
    assert reader.verify_reference(bundle)["quantum_work_performed"] is False


@pytest.mark.parametrize("name", ASSET_NAMES)
def test_byte_tampering_rejects_each_selected_file(reader, bundle, name):
    path = bundle / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        reader.verify_reference(bundle)


@pytest.mark.parametrize("name", ("supervisor.py", "execution-claim.json"))
def test_extra_operational_member_is_rejected(reader, bundle, name):
    (bundle / name).write_bytes(b"{}\n")
    with pytest.raises(ValueError):
        reader.verify_reference(bundle)


@pytest.mark.parametrize("name", ASSET_NAMES)
def test_symlinked_selected_file_is_rejected(reader, bundle, tmp_path, payload, name):
    target = (tmp_path / "redirect-target").resolve()
    target.write_bytes(payload[name])
    path = bundle / name
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError):
        reader.verify_reference(bundle)


def test_symlinked_bundle_is_rejected(reader, bundle, tmp_path):
    redirect = tmp_path / "redirect-bundle"
    redirect.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError):
        reader.verify_reference(redirect)


SEMANTIC_FORGERIES = (
    ("schema", ("schema",), "unaccepted.reference.v1"),
    ("status", ("status",), "complete_quantum_algorithm"),
    ("input-domain", ("scope", "input_domain"), "Fin 64"),
    ("input-count", ("scope", "input_count"), 64),
    ("wire-count", ("scope", "quantum_wires_in_native_source"), 6),
    ("instruction-count", ("scope", "native_source_instructions"), 13),
    ("wire-order", ("scope", "wire_order", "carry"), [5]),
    ("mapping", ("scope", "mapping"), "general modular multiplication"),
    ("proof-model", ("scope", "proof_model"), "whole quantum Shor algorithm"),
    ("carry-one", ("scope", "carry_in_one_included"), False),
    ("helper-exit", ("original_helper", "exit"), 0),
    ("helper-complete", ("original_helper", "helper_complete"), True),
    ("empty-audit-policy", ("original_helper", "empty_audit_contract_satisfied"), True),
    ("new-proof", ("projection_actions", "new_proof_run"), True),
    ("new-export", ("projection_actions", "new_source_export"), True),
    ("new-quantum-work", ("projection_actions", "new_quantum_simulation"), True),
    ("version-exit", ("lean", "version_subprocess_exit"), 1),
    ("proof-exit", ("lean", "proof_subprocess_exit"), 1),
    ("toolchain", ("lean", "toolchain"), "leanprover/lean4:v4.33.1"),
    ("lean-source", ("lean", "source_commit"), "0" * 40),
    ("missing-foundations", ("lean", "foundation_axioms_by_theorem"), {}),
    ("hidden-propext", ("lean", "foundation_axioms_by_theorem", "ripple_bits"), []),
    ("added-foundation", ("lean", "foundation_axioms_by_theorem", "weighted_lt32"), ["Classical.choice"]),
    ("domain-axiom", ("lean", "user_declared_domain_axioms"), ["assumed_arithmetic"]),
    ("sorry", ("lean", "sorryAx_present"), True),
    ("native-decide", ("lean", "native_decide_used"), True),
    ("audit-order", ("lean", "audit_order"), []),
    ("literal-gates", ("native_source", "literal_gate_list_match_observed"), False),
    ("bridge-exit", ("native_source", "correspondence_check_exit"), 1),
    ("parser-proof", ("native_source", "parser_formally_verified"), True),
    ("dropped-exclusions", ("exclusions",), []),
)


@pytest.mark.parametrize("label,path,replacement", SEMANTIC_FORGERIES,
                         ids=[item[0] for item in SEMANTIC_FORGERIES])
def test_semantic_record_forgery_is_rejected(reader, record_and_audit, label, path, replacement):
    value, audit = record_and_audit
    forged = copy.deepcopy(value)
    destination = forged
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = copy.deepcopy(replacement)
    with pytest.raises(ValueError):
        reader.validate_record(forged, audit)


def test_same_length_exclusion_forgery_is_rejected(reader, record_and_audit):
    value, audit = record_and_audit
    forged = copy.deepcopy(value)
    forged["exclusions"][0] = "No excluded claims"
    with pytest.raises(ValueError):
        reader.validate_record(forged, audit)


def test_missing_exclusion_field_is_rejected(reader, record_and_audit):
    value, audit = record_and_audit
    forged = copy.deepcopy(value)
    del forged["exclusions"]
    with pytest.raises(ValueError):
        reader.validate_record(forged, audit)


@pytest.mark.parametrize("mode", ("missing", "reordered", "extra", "hidden-propext", "namespace", "invalid-utf8"))
def test_recorded_audit_forgery_is_rejected(reader, record_and_audit, mode):
    value, audit = record_and_audit
    lines = audit.splitlines()
    if mode == "missing":
        forged = b"\n".join(lines[:-1]) + b"\n"
    elif mode == "reordered":
        forged = b"\n".join(reversed(lines)) + b"\n"
    elif mode == "extra":
        forged = audit + b"\n"
    elif mode == "hidden-propext":
        forged = audit.replace(b"[propext]", b"[]", 1)
    elif mode == "namespace":
        forged = audit.replace(b"ShorRippleBooleanReference", b"UnacceptedReference", 1)
    else:
        forged = b"\xff"
    with pytest.raises(ValueError):
        reader.validate_record(copy.deepcopy(value), forged)


@pytest.mark.parametrize("flag", ("new_scientific_execution", "private_operational_payload_included"))
@pytest.mark.parametrize("replacement", (True, 0, None, "missing"))
def test_manifest_flags_require_explicit_false(reader, bundle, monkeypatch, flag, replacement):
    path = bundle / "manifest.json"
    value = json.loads(path.read_bytes())
    if replacement == "missing":
        del value[flag]
    else:
        value[flag] = replacement
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    # Isolate semantic validation after byte authentication inside this fixture.
    monkeypatch.setattr(reader, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        reader.verify_reference(bundle)
