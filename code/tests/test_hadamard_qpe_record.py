"""Immutable historical sampling records and zero-sampling current replay."""
from pathlib import Path
import importlib.util
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("hadamard_saved_record_test",ROOT/"scripts/verify_hadamard_qpe.py")
verifier=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(verifier)
CANONICAL=ROOT/"assets/generated/hadamard-qpe"


def test_all_canonical_evidence_replays_without_sampling():
    report=verifier.verify_record(CANONICAL)
    assert report["producer_source_commit"]=="def275c501c8c55a32e219674000b4199eaa16d4"
    assert report["historical_sampled_fresh_trials"]==47376
    assert report["qasm_exports_reproduced"]==49
    assert report["numerical_experiment_checks"]==48
    assert report["saved_runs_recovered"]==12
    assert report["new_sampling_calls"]==report["native_calls"]==0
    assert report["numerical_certificate"] is False


def test_manifest_and_current_source_changes_rejected_before_replay(tmp_path):
    output=tmp_path/"record";shutil.copytree(CANONICAL,output)
    original=(output/"manifest.json").read_bytes()
    (output/"manifest.json").write_bytes(original+b" ")
    with pytest.raises(ValueError,match="Immutable historical"):verifier.verify_record(output)
    source=tmp_path/"hadamard_qpe.py";source.write_bytes((ROOT/"src/openquantum/hadamard_qpe.py").read_bytes()+b"\n")
    with pytest.raises(ValueError,match="beyond the reviewed"):verifier.verify_record(CANONICAL,source)


def test_changed_samples_circuit_or_extra_file_rejected(tmp_path):
    output=tmp_path/"record";shutil.copytree(CANONICAL,output)
    p=output/"results.json";p.write_bytes(p.read_bytes()+b" ")
    with pytest.raises(ValueError,match="Saved file hash"):verifier.verify_record(output)
    p.write_bytes((CANONICAL/"results.json").read_bytes())
    circuit=output/"circuits/phase-0-cos-k1.qasm";circuit.write_bytes(circuit.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="Saved file hash"):verifier.verify_record(output)
    circuit.write_bytes((CANONICAL/"circuits/phase-0-cos-k1.qasm").read_bytes())
    (output/"unknown.json").write_text("{}")
    with pytest.raises(ValueError,match="inventory changed"):verifier.verify_record(output)
