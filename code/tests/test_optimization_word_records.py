"""Public saved-model integrity tests; no Native, synthesis or simulation."""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/optimization_word_records.py"
BUNDLE = ROOT / "assets/generated/optimization/word-transport"
spec = importlib.util.spec_from_file_location("test_public_word_record_core", MODULE)
records = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = records
spec.loader.exec_module(records)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


@pytest.fixture(scope="module")
def trusted_pin():
    """Read the reviewed literal pin, never learn trust from mutable data."""
    helper = ROOT / "scripts/optimization_word_record_release.py"
    if not helper.is_file():
        pytest.fail("Public release helper/immutable pin is not finalized")
    tree = ast.parse(helper.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    values[name.id] = node.value.value
    for name in ("MANIFEST_SHA256", "RECORD_MANIFEST_SHA256", "WORD_MANIFEST_SHA256"):
        value = values.get(name)
        if isinstance(value, str) and records.HEX.fullmatch(value):
            return value
    pytest.fail("Public helper lacks a reviewed literal manifest pin")


@pytest.fixture(scope="module")
def authenticated(trusted_pin):
    return records.authenticate_bundle(BUNDLE, trusted_pin)


@pytest.fixture(scope="module")
def actual_preimages(authenticated):
    inputs = records.strict_json(records.decode_gzip(authenticated.raw["native-inputs.json.gz"]),
                                 maximum=records.MAX_DECODED_BYTES)
    raw = records.decode_gzip(authenticated.raw["producer-returns.jsonl.gz"])
    returns = [records.strict_json(line, maximum=records.MAX_CLASS_BYTES) for line in raw.splitlines()]
    schema = records._module(authenticated, "schema")
    plan = schema.restore_record(authenticated.json("native-plan.json"))
    return plan, inputs, returns


def fake_bundle(path):
    path = path.resolve()
    path.mkdir()
    files = {}
    for name in records.FIXED_FILES:
        destination = path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = b"{}\n"
        destination.write_bytes(raw)
        files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest = dict(schema=records.SCHEMA, status="completed_fixed_native_resource_model",
                    producer_source_commit=records.PRODUCER_COMMIT,
                    study=deepcopy(records.STUDY), fixed_machine={}, files=files,
                    source_pins={name: files[name]["sha256"] for name in records.CODE.values()})
    raw = encoded(manifest)
    (path / "record-manifest.json").write_bytes(raw)
    return path, manifest, hashlib.sha256(raw).hexdigest()


def rewrite_fake_manifest(path, manifest):
    raw = encoded(manifest)
    (path / "record-manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_core_top_level_imports_are_standard_library():
    tree = ast.parse(MODULE.read_text())
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= sys.stdlib_module_names | {"__future__"}


def test_saved_model_exact_scope_and_portable_no_native(trusted_pin, monkeypatch):
    imports = []
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        imports.append(name)
        if name.split(".")[0] in {"qdk", "qiskit", "numpy", "scipy", "sympy"}:
            raise AssertionError("Public saved validation must not import scientific/Native APIs")
        return original(name, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Public saved validation cannot call Git, processes or Native")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = records.verify_records(BUNDLE, trusted_pin)
    assert result["status"] == "verified_saved_word_native_model"
    assert (result["transcript_id"], result["classes"], result["parents"], result["groups"]) == (
        "qaoa-cycle-4-p2-alternate-seed-7", 54, 49, 110)
    assert (result["preparations"], result["t_states"]) == (82032, 182646240)
    assert result["serial_runtime_ns"] == 602466084947200
    assert result["baseline_serial_runtime_ns"] == 797328306784000
    assert result["peak_physical_qubits"] == 243358
    assert result["whole_physical_failure_exact"] == "63620243034381673613/604462909807314587353088"
    assert result["factory_availability_failure_exact"] == "1141539/6250000000"
    assert result["model_runtime_reduction_fraction_exact"] == "30447222162/124582547935"
    assert (result["selected_words"], result["added_distinct_cx"]) == (744, 7020)
    assert result["native_api_calls"] == 0 and result["writes_performed"] is False
    assert result["hardware_execution"] is False and result["original_pipeline_layout_equivalence"] is False
    assert result["global_frontier_claim"] is False
    assert result["factory_result_metadata_scope"]["fields"]["copies"]["status"] == "validated_exactly_one"
    for name in ("runs", "states", "error_rate"):
        assert result["factory_result_metadata_scope"]["fields"][name] == {
            "status": "captured_unvalidated", "used_in_accounting": False}


def test_identity_only_never_compiles_sources(trusted_pin, monkeypatch):
    monkeypatch.setattr(records, "_module", lambda *_args: pytest.fail("Identity check compiled a source"))
    result = records.verify_records(BUNDLE, trusted_pin, replay=False)
    assert result["status"] == "verified_word_record_identity"
    assert "no saved model rederivation" in result["verification_scope"]


@pytest.mark.parametrize("name", ["source-recipe.json", "native-result.json", "code/optimization_word_native.py",
                                 "producer-returns.jsonl.gz", "baseline/manifest.json"])
def test_authenticated_file_tamper_precedes_any_source_execution(tmp_path, trusted_pin, monkeypatch, name):
    destination = tmp_path.resolve() / "bundle"
    shutil.copytree(BUNDLE, destination)
    path = destination / name
    path.write_bytes(path.read_bytes() + b"x")
    monkeypatch.setattr(records, "_module", lambda *_args: pytest.fail("Tampered source/data was compiled"))
    with pytest.raises(records.RecordError, match="immutable scientific/source"):
        records.verify_records(destination, trusted_pin)


def test_untrusted_rehashed_manifest_is_not_a_new_trust_anchor(tmp_path, trusted_pin):
    destination = tmp_path.resolve() / "bundle"
    shutil.copytree(BUNDLE, destination)
    path = destination / "record-manifest.json"
    value = json.loads(path.read_text())
    value["study"]["preparations"] += 1
    path.write_bytes(encoded(value))
    with pytest.raises(records.RecordError, match="manifest changed"):
        records.authenticate_bundle(destination, trusted_pin)


@pytest.mark.parametrize("change", ["missing", "extra", "empty_directory"])
def test_exact_inventory_rejects_unlisted_or_omitted_files(tmp_path, change):
    path, _, pin = fake_bundle(tmp_path / "bundle")
    if change == "missing":
        (path / "native-result.json").unlink()
    elif change == "extra":
        (path / "extra.txt").write_text("unexpected")
    else:
        (path / "empty").mkdir()
    with pytest.raises(records.RecordError):
        records.authenticate_bundle(path, pin)


@pytest.mark.parametrize("key,value", [("classes", 54.0), ("depth", True), ("algorithm", "vqe"),
                                      ("transcript_id", "all-quality-runs"), ("groups", 49)])
def test_study_contract_is_exact_and_never_broadened(tmp_path, key, value):
    path, manifest, _ = fake_bundle(tmp_path / "bundle")
    manifest["study"][key] = value
    pin = rewrite_fake_manifest(path, manifest)
    with pytest.raises(records.RecordError, match="one frozen C4"):
        records.authenticate_bundle(path, pin)


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "code/../escape", "code//escape", "code\\escape"])
def test_lexical_inventory_paths_cannot_escape(tmp_path, relative):
    path, manifest, _ = fake_bundle(tmp_path / "bundle")
    manifest["files"][relative] = dict(bytes=1, sha256="0" * 64)
    pin = rewrite_fake_manifest(path, manifest)
    with pytest.raises(records.RecordError, match="path"):
        records.authenticate_bundle(path, pin)


@pytest.mark.parametrize("kind", ["file_link", "ancestor_link", "hardlink", "root_link"])
def test_redirected_or_shared_record_paths_are_rejected(tmp_path, kind):
    path, _, pin = fake_bundle(tmp_path / "bundle")
    if kind == "root_link":
        link = tmp_path.resolve() / "linked"
        link.symlink_to(path, target_is_directory=True)
        target = link
    elif kind == "ancestor_link":
        moved = tmp_path.resolve() / "other-code"
        (path / "code").rename(moved)
        (path / "code").symlink_to(moved, target_is_directory=True)
        target = path
    else:
        moved = tmp_path.resolve() / "external.json"
        shutil.copyfile(path / "native-result.json", moved)
        (path / "native-result.json").unlink()
        if kind == "file_link":
            (path / "native-result.json").symlink_to(moved)
        else:
            (path / "native-result.json").hardlink_to(moved)
        target = path
    with pytest.raises(records.RecordError, match="[Ll]inked"):
        records.authenticate_bundle(target, pin)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}', b'{"x":NaN}',
                                b'{"x":Infinity}', b'{"x":1e9999}', b'{"x":'])
def test_strict_json_rejects_ambiguous_or_nonfinite_records(raw):
    with pytest.raises(records.RecordError):
        records.strict_json(raw)


def test_integer_depth_and_byte_caps_are_bounded():
    with pytest.raises(records.RecordError, match="digit"):
        records.strict_json(("1" * 1025).encode())
    with pytest.raises(records.RecordError):
        records.strict_json(b"[" * 130 + b"0" + b"]" * 130)
    with pytest.raises(records.RecordError, match="byte"):
        records.strict_json(b"{}", maximum=1)


@pytest.mark.parametrize("kind", ["truncated", "concatenated", "trailing", "oversize"])
def test_gzip_transport_cannot_hide_extra_or_unbounded_payload(kind):
    packed = gzip.compress(b"123456", mtime=0)
    if kind == "truncated":
        packed = packed[:-1]
    elif kind == "concatenated":
        packed += gzip.compress(b"second", mtime=0)
    elif kind == "trailing":
        packed += b"garbage"
    with pytest.raises(records.RecordError):
        records.decode_gzip(packed, maximum=5 if kind == "oversize" else 100)


@pytest.mark.parametrize("change", ["duplicate", "boolean_repetitions", "wrong_wire", "wrong_resource", "wrong_base"])
def test_full_trace_semantics_are_checked_not_only_its_digest(actual_preimages, change):
    plan, inputs, _ = actual_preimages
    request, value = plan["requests"][0], deepcopy(inputs[0])
    trace = json.loads(value["native_trace_json"])
    if change == "duplicate":
        value["native_trace_json"] = '{"block":{},"block":{}}'
    else:
        if change == "boolean_repetitions":
            trace["block"]["repetitions"] = True
        elif change == "wrong_wire":
            trace["block"]["operations"][0]["BlockOperation"]["operations"][0]["GateOperation"]["qubits"][0] = 14
        elif change == "wrong_resource":
            trace["resource_states"]["1028"] -= 1
        else:
            trace["base_error"] *= 1.01
        value["native_trace_json"] = json.dumps(trace)
    value["native_trace_json_sha256"] = hashlib.sha256(value["native_trace_json"].encode()).hexdigest()
    with pytest.raises(records.RecordError):
        records.verify_trace(request, value)


@pytest.mark.parametrize("change", ["bool_copy", "float_runtime", "float_runs", "bool_arity", "property_value", "duplicate_property"])
def test_native_scalar_tag_and_property_loopholes_are_closed(actual_preimages, change):
    _, inputs, returns = actual_preimages
    raw = deepcopy(returns[0]["producer_record"])
    if change == "bool_copy":
        raw["factories"][0]["copies"]["value"] = True
    elif change == "float_runtime":
        raw["result"]["runtime"]["value"] = float(raw["result"]["runtime"]["value"])
    elif change == "float_runs":
        raw["factories"][0]["runs"]["value"] = float(raw["factories"][0]["runs"]["value"])
    elif change == "bool_arity":
        raw["native_isa_cost_records"][0]["arity"] = True
    elif change == "property_value":
        raw["properties"][0]["value"]["value"] += 1
    else:
        raw["properties"].append(deepcopy(raw["properties"][0]))
    with pytest.raises(records.RecordError):
        records.verify_raw_types(raw, inputs[0])


def test_typed_equality_does_not_coerce_scalar_types():
    assert records.typed_equal({"n": 1, "flag": True}, {"n": 1, "flag": True})
    assert not records.typed_equal({"n": 1}, {"n": True})
    assert not records.typed_equal([1.0], [1])


def test_public_replay_has_no_nonboolean_control_choice(trusted_pin):
    with pytest.raises(records.RecordError, match="boolean"):
        records.verify_records(BUNDLE, trusted_pin, replay=1)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO regression")
@pytest.mark.parametrize("replacement_race", [False, True])
def test_fifo_rejection_is_bounded_even_when_replaced_before_open(tmp_path, replacement_race):
    path, _, pin = fake_bundle(tmp_path / "bundle")
    if not replacement_race:
        (path / "native-result.json").unlink()
        os.mkfifo(path / "native-result.json")
    source = r'''
import importlib.util,sys,os
from pathlib import Path
s=importlib.util.spec_from_file_location("fifo_saved_core",sys.argv[1])
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
root=Path(sys.argv[2]); race=sys.argv[4]=="True"
original=m.os.open; changed=False
def open_checked(value,flags,*args,**kwargs):
    global changed
    if race and not changed and Path(value)==root/"native-result.json":
        changed=True
        (root/"native-result.json").unlink()
        os.mkfifo(root/"native-result.json")
    return original(value,flags,*args,**kwargs)
m.os.open=open_checked
try:m.authenticate_bundle(root,sys.argv[3])
except m.RecordError:print("bounded_nonregular_rejection");sys.exit(0)
raise AssertionError("FIFO was accepted")
'''
    result = subprocess.run([sys.executable, "-c", source, str(MODULE), str(path), pin,
                             str(replacement_race)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded_nonregular_rejection"


def test_inventory_stream_rejects_first_unknown_entry_without_materializing_tail(tmp_path, monkeypatch):
    emitted = []

    class Entries:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            for index in range(1000000):
                emitted.append(index)
                yield SimpleNamespace(name="unlisted", is_symlink=lambda: False,
                                      stat=lambda **_kwargs: SimpleNamespace(st_mode=records.stat.S_IFDIR))

    monkeypatch.setattr(records.os, "scandir", lambda *_args: Entries())
    with pytest.raises(records.RecordError, match="unlisted directory"):
        records.verify_inventory_tree(tmp_path.resolve(), set(), {""})
    assert emitted == [0]


def test_inventory_stream_caps_actual_visits_even_for_malformed_duplicate_entries(tmp_path, monkeypatch):
    emitted = []

    class Entries:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            for index in range(1000000):
                emitted.append(index)
                yield SimpleNamespace(name="known", is_symlink=lambda: False,
                    stat=lambda **_kwargs: SimpleNamespace(st_mode=records.stat.S_IFREG, st_nlink=1))

    monkeypatch.setattr(records.os, "scandir", lambda *_args: Entries())
    with pytest.raises(records.RecordError, match="entry count"):
        records.verify_inventory_tree(tmp_path.resolve(), {"known"}, {""})
    assert len(emitted) == records.MAX_FILES + records.MAX_DIRECTORIES + 2


def test_inventory_path_depth_is_capped_before_traversal():
    with pytest.raises(records.RecordError, match="path"):
        records.relative_path("/".join(["deep"] * (records.MAX_PATH_DEPTH + 1)))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX cached-source regression")
def test_saved_snapshot_uses_only_authenticated_source_bytes_after_disk_replacement(tmp_path, trusted_pin):
    destination = tmp_path.resolve() / "bundle"
    shutil.copytree(BUNDLE, destination)
    authenticated = records.authenticate_bundle(destination, trusted_pin)
    native = records._module(authenticated, "native")
    records._bind_cached_native_source_reads(authenticated, native)
    schema = records._module(authenticated, "schema")
    plan = schema.restore_record(authenticated.json("native-plan.json"))
    changed = destination / records.CODE["accounting"]
    changed.unlink()
    os.mkfifo(changed)
    # This uses the original hash/accounting guards and the exact cached source,
    # rather than opening the now-blocking source path or importing a substitute.
    native._bound_plan(plan, check_runtime=False)
    with pytest.raises(records.RecordError, match="unauthenticated source path"):
        native._regular(tmp_path.resolve() / "unknown.py")
    with pytest.raises(records.RecordError, match="hash guard"):
        native._regular(destination / records.CODE["accounting"], expected="0" * 64)
    # Final bundle preservation still rejects the changed on-disk input.
    with pytest.raises(records.RecordError):
        records.authenticate_bundle(destination, trusted_pin)
