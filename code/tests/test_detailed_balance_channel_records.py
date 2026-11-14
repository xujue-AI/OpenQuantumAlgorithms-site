"""Saved/mocked record checks; no scientific producer or simulation executes."""
from __future__ import annotations
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    path = ROOT / "scripts/verify_detailed_balance_channel.py"
    spec = importlib.util.spec_from_file_location("saved_gibbs_gate_test", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


@pytest.fixture(scope="module")
def gate():
    return load_gate()


@pytest.fixture(scope="module")
def payload(gate):
    return gate._authenticate(ROOT / gate.BUNDLE_RELATIVE)


def mutate_report(gate, payload, edit):
    manifest, data = payload
    copied = dict(data)
    report = json.loads(data["results.json"])
    edit(report)
    copied["results.json"] = json.dumps(report, allow_nan=False).encode()
    return manifest, copied


def test_saved_gate_authenticates_complete_scope_without_scientific_imports(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"qiskit", "numpy", "scipy", "qdk", "qsharp", "openquantum"}:
            pytest.fail("scientific/project import by saved gate")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    gate = load_gate()
    report = gate.verify_bundle(ROOT / gate.BUNDLE_RELATIVE, source_root=ROOT)
    assert report["files"] == 40 and report["cases"] == 11 and report["qasm_circuits"] == 33
    assert report["current_source_inputs_checked"] == 5
    assert report["new_numerical_simulations"] == report["native_estimator_calls"] == 0
    assert report["physical_resource_estimate"] is None
    assert report["formal_certificate"] is report["uniform_Gibbs_accuracy_certificate"] is False


def test_authenticated_record_checks_preserve_finite_and_k64_distinctions(gate,payload):
    manifest,data = payload
    report = gate._records(manifest,data)
    assert report["source_commit"] == gate.PRODUCER_COMMIT
    assert max(case["check"]["finite_stationary_gibbs_trace_distance"] for case in report["cases"]) > .05
    assert max(row["gibbs_trace_distance"] for case in report["cases"]
               for row in case["check"]["spanning_input_channel_power_checks"]) > .14
    assert all(case["preparation"]["rounds"] == 8 and case["check"]["rounds"] == 64
               for case in report["cases"])
    assert all(case["logical_readout_contract"]["independent_complete_preparations"] == 738
               for case in report["cases"])


@pytest.mark.parametrize("field,value", [("source_dirty", True), ("source_revision_is_exact", 1),
                                        ("general_sampler_implemented", True), ("formal_proof_reproduced", True),
                                        ("source_commit", "0"*40), ("exact_finite_kms", 0)])
def test_record_promotion_and_identity_mutations_fail(gate,payload,field,value):
    manifest,data = mutate_report(gate,payload,lambda r: r.__setitem__(field,value))
    with pytest.raises(ValueError): gate._records(manifest,data)


@pytest.mark.parametrize("edit", [
    lambda r: r["cases"][0]["dilation"].__setitem__("beta", True),
    lambda r: r["cases"][0]["dilation"].__setitem__("n_system", 2),
    lambda r: r["cases"][0]["dilation"].__setitem__("weak_flag", 0),
    lambda r: r["cases"][0]["dilation"].__setitem__("coherent_correction_certificate", True),
    lambda r: r["cases"][0]["preparation"].__setitem__("rounds", 64),
    lambda r: r["cases"][0]["reset_round"].__setitem__("resets_per_round", 0),
])
def test_meta_clock_scope_and_reset_mutations_fail(gate,payload,edit):
    manifest,data = mutate_report(gate,payload,edit)
    with pytest.raises(ValueError): gate._records(manifest,data)


@pytest.mark.parametrize("edit", [
    lambda r: r["cases"][0]["check"].__setitem__("rounds", 8),
    lambda r: r["cases"][0]["check"].__setitem__("channel_power_scope", "executed all64 rounds"),
    lambda r: r["cases"][0]["check"].__setitem__("contraction_scope", "certified mixing"),
    lambda r: r["cases"][0]["check"].__setitem__("nonzero_environment_nojump_weight", 0),
    lambda r: r["cases"][0]["check"].__setitem__("actual_vs_independent_channel_frobenius_error", 1),
    lambda r: r["cases"][0]["check"].__setitem__("uniform_gibbs_accuracy", .05),
])
def test_numerical_evidence_scope_mutations_fail(gate,payload,edit):
    manifest,data = mutate_report(gate,payload,edit)
    with pytest.raises(ValueError): gate._records(manifest,data)


@pytest.mark.parametrize("field,value", [("independent_complete_preparations", 738.0),
                                        ("total_rounds", 738), ("mean_additive_sampling_error", True),
                                        ("continuum_gibbs_accuracy_certificate", True),
                                        ("synthesis_and_physical_faults_included", True),
                                        ("exclusions", [])])
def test_sampling_projection_mutations_fail(gate,payload,field,value):
    edit=lambda r: r["cases"][0]["logical_readout_contract"].__setitem__(field,value)
    manifest,data=mutate_report(gate,payload,edit)
    with pytest.raises(ValueError): gate._records(manifest,data)


def test_corrupt_controlled_inverse_and_weak_clock_fail(gate,payload):
    manifest,data=payload
    report=json.loads(data["results.json"]);case=report["cases"][0]
    filename=case["name"]+"-dilation.qasm"
    original=data[filename].decode().splitlines()
    gu=case["dilation"]["access_operations"]
    gw=case["dilation"]["weak_operations"]
    for index,replacement in [(3+gu+1,"cx q[2], q[4];"),
                              (3+gu+gw,"x q[0];")]:
        lines=list(original);lines[index]=replacement
        changed=dict(data);changed[filename]=("\n".join(lines)+"\n").encode()
        with pytest.raises(ValueError): gate._records(manifest,changed)


def test_reset_and_full_preparation_chronology_fail(gate,payload):
    manifest,data=payload
    case=json.loads(data["results.json"])["cases"][0]
    for kind in ("reset-round","preparation"):
        filename=case["name"]+"-"+kind+".qasm"
        lines=data[filename].decode().splitlines()
        index=3 if kind=="reset-round" else 4
        lines[index]="reset q[4];"
        changed=dict(data);changed[filename]=("\n".join(lines)+"\n").encode()
        with pytest.raises(ValueError): gate._records(manifest,changed)


@pytest.mark.parametrize("line", ["if (c[0]) x q[0];", "unitary q[0];", "cx q[1], q[1];",
                                 "rz(nan) q[0];", "ry(1e10000) q[0];", "p(pi/0) q[0];",
                                 "c[1] = measure q[0];", "h q[5];"])
def test_closed_qasm_syntax_and_wire_bounds(gate,line):
    raw=("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[5] q;\n"+line+"\n").encode()
    with pytest.raises(ValueError): gate._qasm(raw,5,False)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'['*20+b'0'+b']'*20,
                                b'{"a":"'+b'x'*8193+b'"}', b'{' ])
def test_json_duplicate_finite_structure_caps(gate,raw):
    with pytest.raises(ValueError): gate._json(raw)


def test_repeated_directory_entry_stream_is_bounded(gate,monkeypatch,tmp_path):
    class Entries:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def __iter__(self):
            for _ in range(41):
                yield SimpleNamespace(name="same",stat=lambda **kwargs: SimpleNamespace(st_mode=stat.S_IFREG,st_nlink=1,st_size=0))
    # Restore the shared os.scandir before pytest removes a passing tmp_path.
    with monkeypatch.context() as scoped:
        scoped.setattr(gate.os,"scandir",lambda *args,**kwargs: Entries())
        with pytest.raises(ValueError,match="Duplicate/overlimit"): gate._inventory(tmp_path)


def test_saved_pin_rejects_changed_manifest_extra_file_and_corrupt_bytes(gate,tmp_path):
    tmp_path=tmp_path.resolve()
    original=ROOT/gate.BUNDLE_RELATIVE
    copied=tmp_path/"bundle";shutil.copytree(original,copied)
    qasm=next(copied.glob("*-dilation.qasm"));old=qasm.read_bytes()
    qasm.write_bytes(old+b"\n")
    with pytest.raises(ValueError): gate.verify_bundle(copied)
    qasm.write_bytes(old)
    (copied/"private.json").write_text("{}")
    with pytest.raises(ValueError): gate.verify_bundle(copied)
    (copied/"private.json").unlink()
    path=copied/"manifest.json";path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError,match="manifest pin"): gate.verify_bundle(copied)


def test_source_five_inputs_are_all_enforced(gate,tmp_path):
    tmp_path=tmp_path.resolve()
    copied=tmp_path/"sources"
    for name in gate.SOURCE_SHA256:
        path=copied/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    gate._sources(copied)
    for name in gate.SOURCE_SHA256:
        path=copied/name;original=path.read_bytes();path.write_bytes(original+b"\n")
        with pytest.raises(ValueError): gate._sources(copied)
        path.write_bytes(original)


def test_reader_rejects_fifo_symlink_hardlink_before_content(gate,tmp_path):
    tmp_path=tmp_path.resolve()
    regular=tmp_path/"regular";regular.write_bytes(b"x")
    linked=tmp_path/"symbolic";linked.symlink_to(regular)
    hard=tmp_path/"hard";os.link(regular,hard)
    fifo=tmp_path/"fifo";os.mkfifo(fifo)
    for name in ("symbolic","hard","fifo"):
        with pytest.raises(ValueError): gate._read(tmp_path,name,10)


@pytest.mark.parametrize("value", [None, [], "report", True])
def test_container_failures_are_value_errors(gate,value):
    with pytest.raises(ValueError): gate._mapping(value,"report")
