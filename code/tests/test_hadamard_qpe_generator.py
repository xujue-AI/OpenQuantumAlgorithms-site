"""Generator scope, fresh records and stale-output regressions."""
from pathlib import Path
from fractions import Fraction
import hashlib
import importlib.util
import json
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("hadamard_generator_test",ROOT/"scripts/generate_hadamard_qpe.py")
generator=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(generator)


def read_manifest(output):
    m=json.loads((output/"manifest.json").read_text())
    actual={p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    assert actual==set(m["file_sha256"])|{"manifest.json"}
    for name,h in m["file_sha256"].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==h
    return m


def test_no_sim_is_honest_and_preserves_input(tmp_path):
    output=tmp_path/"large"
    result=generator.generate(output,target_qubits=8,phase=.2,levels=3,seed=19,canonical_study=False,simulate=False,scaling=False)
    assert result["parameters"]["phases"]==[.2]
    assert result["simulation"]["performed"] is False
    assert result["simulation"]["modeled_fresh_trials_sampled"]==0
    assert result["runs"]==[]
    assert len(result["cases"][0]["experiments"])==6
    assert all(not x["check"]["performed"] for x in result["cases"][0]["experiments"])
    assert not (output/"circuit.svg").exists()
    assert not result["physical_estimate_performed"]
    read_manifest(output)


def test_owned_record_scope_cleanup_without_new_sampling(tmp_path):
    output=tmp_path/"changed"
    generator.generate(output,phase=.3,levels=3,canonical_study=False,simulate=False,scaling=False)
    # Synthetic prior figure stands for an older owned checked figure.
    path=output/"circuit.svg";path.write_text("historical checked figure")
    m=json.loads((output/"manifest.json").read_text());m["file_sha256"]["circuit.svg"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/"manifest.json").write_text(json.dumps(m))
    generator.generate(output,phase=.2,levels=1,canonical_study=False,simulate=False,scaling=False)
    assert not path.exists()
    assert not (output/"circuits/phase-0-cos-k4.qasm").exists()
    assert len(list((output/"circuits").glob("*.qasm")))==2
    read_manifest(output)


def test_exact_python_api_failure_budget_and_missing_ownership(tmp_path):
    result=generator.generate(tmp_path/"fraction",phase=.2,levels=2,failure_target=.1,canonical_study=False,simulate=False,scaling=False)
    assert result["parameters"]["total_ideal_failure_target_exact"]=="1/10"
    p=tmp_path/"occupied";p.mkdir();(p/"user.txt").write_text("preserve")
    with pytest.raises(ValueError):generator.generate(p,canonical_study=False,simulate=False,scaling=False)
    assert (p/"user.txt").read_text()=="preserve"


@pytest.mark.parametrize("failure",[True,0,1,float("nan")])
def test_invalid_failure_targets_before_outputs(tmp_path,failure):
    p=tmp_path/"bad"
    with pytest.raises(ValueError):generator.generate(p,failure_target=failure,canonical_study=False,simulate=False,scaling=False)
    assert not p.exists()


def test_changed_canonical_arguments_and_protected_outputs(tmp_path):
    with pytest.raises(ValueError):generator.generate(tmp_path/"bad",phase=.2)
    with pytest.raises(ValueError):generator.generate(ROOT/"assets/generated/energy-estimation",canonical_study=False,simulate=False)
    with pytest.raises(ValueError):generator.generate(ROOT/"src",canonical_study=False,simulate=False)
