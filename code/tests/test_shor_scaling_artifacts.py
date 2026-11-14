"""Native-input identity, explicit unavailable states, and artifact integrity."""
import hashlib
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from openquantum.shor_scaling import MatchedBudget,ScalingCase,independent_sampling_contract,materialize
from openquantum.resource_estimation import SurfaceCodeProfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("shor_scaling_generator",ROOT/"scripts/generate_shor_scaling.py")
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_output_guard_keeps_frozen_artifacts_isolated():
    for name in ("assets/generated/shor","assets/generated/shor/ripple","assets/generated/grover","assets/generated/resources"):
        with pytest.raises(ValueError):generator.guard_output(ROOT/name)
    generator.guard_output(ROOT/"assets/generated/shor-scaling")


def test_stale_managed_outputs_are_removed_without_touching_other_files(tmp_path):
    names=["study.json","n21-a2-t6/ripple/circuit.qasm","n21-a2-t6/fourier/resources.json",
           "notes.txt","n21-a2-t6/ripple/private.qasm","unrelated/resources.json"]
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("test")
    generator.clear_managed_outputs(tmp_path)
    assert not any((tmp_path/name).exists() for name in names[:3])
    assert all((tmp_path/name).read_text()=="test" for name in names[3:])


def test_native_timeout_is_explicit_and_returns_no_claimed_projection(tmp_path,monkeypatch):
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired("native",.1)
    monkeypatch.setattr(generator.subprocess,"run",timeout)
    case=ScalingCase(15,2,4)
    contract=independent_sampling_contract(case)
    result=generator.isolated_projection(case,"fourier",contract,{"qasm_sha256":"test"},MatchedBudget(),SurfaceCodeProfile(),tmp_path,.1)
    assert result["status"]=="physical_model_unavailable" and "estimate" not in result
    assert result["timeout_seconds"]==.1


def test_worker_rejects_a_native_input_hash_mismatch(tmp_path,monkeypatch):
    case=ScalingCase(3,2,2)
    config={"case":{"modulus":3,"base":2,"phase_bits":2},"method":"ripple","matched_budget":{},
        "trials":8,"profile":{},"cache_dir":str(tmp_path/"cache"),"expected_qasm_sha256":"incorrect",
        "result_path":str(tmp_path/"result.json")}
    (tmp_path/"config.json").write_text(json.dumps(config))
    def forbidden(*args,**kwargs):raise AssertionError("Native query ran despite rebuild mismatch")
    monkeypatch.setattr(generator,"estimate_circuit_resources",forbidden)
    assert generator.estimate_worker(tmp_path/"config.json")==2
    result=json.loads((tmp_path/"result.json").read_text())
    assert result["status"]=="physical_model_unavailable" and "estimate" not in result


def test_zero_success_control_does_not_run_native_queries(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("Zero-success factoring control ran a native query")
    monkeypatch.setattr(generator,"isolated_projection",forbidden)
    # Small prime is a cheap actual-circuit negative control.
    result=generator.generate(tmp_path,(ScalingCase(3,2,2),))
    assert result["native_queries_attempted"]==0
    assert result["status"]=="partial_or_unsupported"
    for record in result["cases"][0]["methods"].values():
        assert record["physical"]["status"]=="not_estimated_unsupported_retry_contract"


def test_generated_scaling_reports_match_complete_circuits_and_frozen_budgets():
    folder=ROOT/"assets/generated/shor-scaling"
    if not (folder/"study.json").exists():pytest.skip("Canonical bounded sweep not yet generated")
    study=json.loads((folder/"study.json").read_text())
    assert study["status"]=="computed_conditional_model_with_explicit_unsupported_controls"
    assert study["native_queries_attempted"]==8
    assert study["frozen_files_unchanged"]
    assert study["frozen_sha256_before"]==study["frozen_sha256_after"]
    for path,digest in study["frozen_sha256_after"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert study["certified_combined_factoring_success"] is None
    for item in study["cases"]:
        contract=item["sampling_contract"]
        for record in item["methods"].values():
            qasm=(folder/record["qasm_relative_path"]).read_bytes()
            assert hashlib.sha256(qasm).hexdigest()==record["qasm_sha256"]
            assert record["data_readouts_per_trial"]==item["case"]["phase_bits"]
            physical=record["physical"]
            if not contract["finite_retry_projection_supported"]:
                assert physical["status"]=="not_estimated_unsupported_retry_contract"
                continue
            assert physical["status"]=="computed_conditional_model"
            model=physical["estimate"]
            r=contract["complete_serial_trials"]
            assert model["qasm_sha256"]==record["qasm_sha256"]
            assert model["budget"]["executions"]==r
            assert model["budget"]["physical_failure_probability"]==.005
            assert model["budget"]["synthesis_model_error_per_execution"]*r==pytest.approx(8e-5)
            assert model["canonical_gate_counts"]["MEAS_Z"]==item["case"]["phase_bits"]
            assert not model["isa_graph_pruning"]
            assert not model["synthesis"]["instance_approximation_certificate_verified"]
            assert model["synthesis"]["model_error_per_execution"]*r<=8e-5
            assert model["synthesis_model_error_all_serial_trials"]<=8e-5
            for point in model["pareto_points"]:
                assert point["physical_failure_union_bound_all_executions"]<=.005+1e-12
                assert point["serial_runtime_all_executions_ns"]==r*point["runtime_per_execution_ns"]
                assert point["peak_times_serial_model_runtime_qubit_ns"]==point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]


def test_materialization_failure_table_preserves_unknown_success(tmp_path):
    study={"cases":[{"case":{"modulus":63,"base":2,"phase_bits":12,"integer_bits":6},
                     "methods":{"fourier":{"declared_qubits":26,"primitive_operations_per_trial":1}}}]}
    generator.write_tables(tmp_path,study)
    table=(tmp_path/"resource-table.md").read_text()
    assert "| fourier | — | — |" in table


def test_symlink_descendants_cannot_overwrite_a_frozen_target(tmp_path):
    output=tmp_path/"output"
    frozen=tmp_path/"frozen-target"
    frozen.mkdir();(frozen/"circuit.qasm").write_text("frozen")
    (output/"n3-a2-t2").mkdir(parents=True)
    (output/"n3-a2-t2/ripple").symlink_to(frozen,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):
        generator.generate(output,(ScalingCase(3,2,2),),physical=False)
    assert (frozen/"circuit.qasm").read_text()=="frozen"
    assert not (output/"study.json").exists()


def test_materialization_error_text_cannot_enter_public_artifacts(tmp_path,monkeypatch):
    sentinel="PRIVATE_PATH_SECRET_SENTINEL"
    def rejected(*args,**kwargs):raise ValueError(sentinel)
    monkeypatch.setattr(generator,"materialize",rejected)
    monkeypatch.setattr(generator,"tiny_variant_checks",lambda:{"status":"checked"})
    result=generator.generate(tmp_path,(ScalingCase(3,2,2),),physical=False)
    assert sentinel not in json.dumps(result)
    assert sentinel not in (tmp_path/"study.json").read_text()


def native_envelope(case,budget,profile,expected_hash="test"):
    resource=budget.resource_budget(8)
    report={"status":"computed_conditional_model","qasm_sha256":expected_hash,
        "budget":asdict(resource),"physical_profile":asdict(profile),"isa_graph_pruning":False,
        "transpiler":{"optimization_level":1},"canonical_gate_counts":{"MEAS_Z":case.phase_bits},
        "synthesis":{"model_error_per_execution":0.0},
        "pareto_points":[{"peak_physical_qubits":100,"data_code_distance":3,
                          "runtime_per_execution_ns":100,"serial_runtime_all_executions_ns":800,
                          "physical_failure_union_bound_all_executions":.004}]}
    return {"status":"computed_conditional_model","native_input_qasm_matches_export":True,"estimate":report}


@pytest.mark.parametrize("mutation",["malformed","list","null","nonzero_exit","no_estimate","false_match","wrong_hash","empty_frontier","bad_transpiler","nonfinite_runtime"])
def test_invalid_isolated_worker_envelopes_are_unavailable(tmp_path,monkeypatch,mutation):
    from types import SimpleNamespace
    case=ScalingCase(15,2,4)
    budget=MatchedBudget();profile=SurfaceCodeProfile()
    envelope=native_envelope(case,budget,profile)
    code=0
    if mutation=="no_estimate":envelope.pop("estimate")
    if mutation=="false_match":envelope["native_input_qasm_matches_export"]=False
    if mutation=="wrong_hash":envelope["estimate"]["qasm_sha256"]="wrong"
    if mutation=="empty_frontier":envelope["estimate"]["pareto_points"]=[]
    if mutation=="bad_transpiler":envelope["estimate"]["transpiler"]=None
    if mutation=="nonfinite_runtime":envelope["estimate"]["pareto_points"][0]["runtime_per_execution_ns"]=float("nan")
    if mutation=="nonzero_exit":code=2
    if mutation=="list":envelope=[]
    if mutation=="null":envelope=None
    def fake_worker(command,**kwargs):
        config=json.loads(Path(command[-1]).read_text())
        Path(config["result_path"]).write_text("malformed" if mutation=="malformed" else json.dumps(envelope))
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(generator.subprocess,"run",fake_worker)
    contract={"complete_serial_trials":8}
    result=generator.isolated_projection(case,"fourier",contract,{"qasm_sha256":"test"},budget,profile,tmp_path,1)
    assert result["status"]=="physical_model_unavailable" and "estimate" not in result


def test_meaningful_computed_worker_envelope_is_accepted(tmp_path,monkeypatch):
    from types import SimpleNamespace
    case=ScalingCase(15,2,4)
    budget=MatchedBudget();profile=SurfaceCodeProfile()
    envelope=native_envelope(case,budget,profile)
    envelope["unreviewed_private_field"]="not included"
    def fake_worker(command,**kwargs):
        config=json.loads(Path(command[-1]).read_text())
        Path(config["result_path"]).write_text(json.dumps(envelope))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(generator.subprocess,"run",fake_worker)
    result=generator.isolated_projection(case,"fourier",{"complete_serial_trials":8},{"qasm_sha256":"test"},budget,profile,tmp_path,1)
    assert result["status"]=="computed_conditional_model"
    assert "unreviewed_private_field" not in result
    assert result["estimate"]["pareto_points"][0]["peak_times_serial_model_runtime_qubit_ns"]==80000
    assert result["estimate"]["synthesis_model_error_all_serial_trials"]==0


def test_derived_worker_fields_are_recomputed_instead_of_trusted(tmp_path,monkeypatch):
    from types import SimpleNamespace
    case=ScalingCase(15,2,4)
    budget=MatchedBudget();profile=SurfaceCodeProfile()
    envelope=native_envelope(case,budget,profile)
    envelope["estimate"]["pareto_points"][0]["peak_times_serial_model_runtime_qubit_ns"]=-1
    envelope["estimate"]["synthesis_model_error_all_serial_trials"]=100
    def fake_worker(command,**kwargs):
        config=json.loads(Path(command[-1]).read_text())
        Path(config["result_path"]).write_text(json.dumps(envelope))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(generator.subprocess,"run",fake_worker)
    result=generator.isolated_projection(case,"fourier",{"complete_serial_trials":8},{"qasm_sha256":"test"},budget,profile,tmp_path,1)
    assert result["estimate"]["pareto_points"][0]["peak_times_serial_model_runtime_qubit_ns"]==80000
    assert result["estimate"]["synthesis_model_error_all_serial_trials"]==0


def test_worker_never_claims_an_empty_pareto_frontier_computed(tmp_path,monkeypatch):
    case=ScalingCase(3,2,2)
    _,_,_,record=materialize(case,"ripple")
    report=native_envelope(case,MatchedBudget(),SurfaceCodeProfile(),record["qasm_sha256"])["estimate"]
    report["pareto_points"]=[]
    monkeypatch.setattr(generator,"estimate_circuit_resources",lambda *args,**kwargs:report)
    config={"case":{"modulus":3,"base":2,"phase_bits":2},"method":"ripple","matched_budget":{},
        "trials":8,"profile":{},"cache_dir":str(tmp_path/"cache"),"expected_qasm_sha256":record["qasm_sha256"],
        "result_path":str(tmp_path/"result.json")}
    (tmp_path/"config.json").write_text(json.dumps(config))
    assert generator.estimate_worker(tmp_path/"config.json")==2
    assert json.loads((tmp_path/"result.json").read_text())["status"]=="physical_model_unavailable"


def test_detached_generator_runs_without_git_or_old_baseline_assets(tmp_path):
    import os
    import shutil
    import sys
    detached=tmp_path/"download"
    (detached/"scripts").mkdir(parents=True)
    shutil.copytree(ROOT/"src",detached/"src")
    shutil.copy2(ROOT/"scripts/generate_shor_scaling.py",detached/"scripts/generate_shor_scaling.py")
    environment=os.environ.copy();environment["PATH"]=str(tmp_path/"no-git")
    environment.pop("PYTHONPATH",None)
    completed=subprocess.run([sys.executable,str(detached/"scripts/generate_shor_scaling.py"),"--case","3:2:2",
        "--no-physical","--output-dir",str(tmp_path/"result")],cwd=detached,env=environment,
        text=True,capture_output=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    result=json.loads((tmp_path/"result/study.json").read_text())
    assert result["environment"]["source_commit"] is None
    assert not result["frozen_reference_files_available"]
    assert result["frozen_files_unchanged"] is None
    assert result["status"]=="logical_only"
