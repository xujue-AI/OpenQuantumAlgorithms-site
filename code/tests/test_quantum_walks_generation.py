"""Public generator artifact/digest and disabled-mode checks."""
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("walk_generator",ROOT/"scripts/generate_quantum_walks.py")
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def run_generator(output,*arguments,check=True):
    environment=os.environ.copy();environment["PYTHONPATH"]=str(ROOT/"src")
    return subprocess.run([sys.executable,str(ROOT/"scripts/generate_quantum_walks.py"),
        "--output-dir",str(output),*arguments],cwd=ROOT,env=environment,
        capture_output=True,text=True,check=check,timeout=90)


def test_disabled_modes_remove_stale_claims_and_bind_the_actual_circuit(tmp_path):
    (tmp_path/"distribution.svg").write_text("stale")
    (tmp_path/"surface-code-scaling.json").write_text('{"status":"stale computed"}')
    run_generator(tmp_path,"--qubits","1","--steps","0","--no-sim","--no-scaling")
    result=json.loads((tmp_path/"demo-results.json").read_text())
    assert result["physical_query_count"]==0 and result["scaling_rows"]==[]
    assert not (tmp_path/"distribution.svg").exists()
    assert not (tmp_path/"surface-code-scaling.json").exists()
    for model,width in (("coined",2),("continuous-time",1)):
        ledger=json.loads((tmp_path/model/"logical-resources.json").read_text())
        assert not ledger["validation"]["performed"]
        assert ledger["logical_qubits"]==width
        assert ledger["primitive_gate_counts"]["measure"]==width
        assert ledger["qasm_sha256"]==hashlib.sha256((tmp_path/model/"measured-trial.qasm").read_bytes()).hexdigest()
        assert ledger["observable_contract"]["shots_required"]==3329
        assert ledger["observable_contract"]["certified_confidence_lower_bound"] is None


def test_invalid_model_does_not_replace_previous_results(tmp_path):
    sentinel=tmp_path/"distribution.svg";sentinel.write_text("previous result")
    completed=run_generator(tmp_path,"--time","nan",check=False)
    assert completed.returncode!=0
    assert sentinel.read_text()=="previous result"


def test_output_cannot_overwrite_frozen_assets_or_source():
    for folder in (ROOT,ROOT/"src",ROOT/"assets/generated/grover",ROOT/"assets/generated/resources"):
        completed=run_generator(folder,"--no-sim","--no-scaling",check=False)
        assert completed.returncode!=0
        assert "dedicated walk directory" in completed.stderr


def test_demo_provenance_does_not_require_optional_qdk(monkeypatch):
    original=importlib.metadata.version
    def version(name):
        if name=="qdk":raise importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(importlib.metadata,"version",version)
    provenance=generator.source_provenance()
    assert provenance["packages"]["qdk"] is None
    assert provenance["packages"]["qiskit"]==original("qiskit")
