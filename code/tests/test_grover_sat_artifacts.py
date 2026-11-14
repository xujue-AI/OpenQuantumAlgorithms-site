"""Full recorded-prefix resource scope and public SAT artifact integrity."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from openquantum.bbht_search import BBHTConfig,run_bbht
from openquantum.grover_sat import CNF
from openquantum.resource_estimation import SurfaceCodeProfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("sat_generator",ROOT/"scripts/generate_grover_sat.py")
generator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_pending_projection_keeps_the_entire_recorded_quantum_prefix(tmp_path):
    cnf=CNF(4,((1,2),(-1,3),(-2,-3),(4,)))
    run=run_bbht(cnf,BBHTConfig(seed=11))
    (tmp_path/"replay-circuits").mkdir()
    result=generator.project_recorded_path(cnf,run,tmp_path,"test",[],tmp_path/"cache",max_replay_qubits=1)
    assert result["status"]=="pending_physical_model"
    assert result["recorded_attempts"]==len(run["attempts"])
    assert result["logical"]["declared_qubits"]==len(run["attempts"])*cnf.num_qubits
    assert result["logical"]["total_output_measurements"]==len(run["attempts"])*cnf.n_variables
    assert result["logical"]["oracle_calls"]==run["actual_oracle_calls"]
    assert (tmp_path/"replay-circuits/test.qasm").exists()
    assert result["search_failure_confidence"] is None


def test_failed_physical_model_is_explicit_without_hiding_cost(tmp_path,monkeypatch):
    cnf=CNF(2,((1,),(-1,)))
    run=run_bbht(cnf,BBHTConfig(seed=19,max_attempts=2))
    (tmp_path/"replay-circuits").mkdir()
    def unsupported(*args,**kwargs):
        raise RuntimeError("Deliberately unsupported full replay model")
    monkeypatch.setattr(generator,"estimate_circuit_resources",unsupported)
    result=generator.project_recorded_path(cnf,run,tmp_path,"test",[SurfaceCodeProfile()],tmp_path/"cache",128)
    assert result["status"]=="pending_physical_model"
    assert result["profiles"][0]["status"]=="pending_physical_model"
    assert "unsupported" in result["profiles"][0]["reason"]
    assert result["logical"]["source_gate_counts"]["measure"]==4


def test_output_guard_and_export_of_a_larger_unsimulated_circuit(tmp_path):
    for folder in ("assets/generated/grover","assets/generated/resources","assets/generated/grover-scaling"):
        with pytest.raises(ValueError,match="overlap"):
            generator.guard_output_directory(ROOT/folder)
    cnf=CNF(10,((1,-2,3),(-3,4),(5,-6),(7,8),(-9,10),(2,)))
    result=generator.export_iteration(cnf,1,tmp_path/"large.qasm")
    assert result["declared_qubits"]==23 and result["data_measurements"]==10
    source=(tmp_path/"large.qasm").read_text()
    assert "OPENQASM 3.0" in source
    assert "gate " not in source
    assert hashlib.sha256(source.encode()).hexdigest()==result["qasm_sha256"]


def test_generated_sat_paths_have_no_unknown_count_confidence_and_no_baseline_changes():
    folder=ROOT/"assets/generated/grover-sat"
    if not (folder/"replay-resources.json").exists():
        pytest.skip("Generate actual default replay projections before artifact checks")
    resources=json.loads((folder/"replay-resources.json").read_text())
    traces=json.loads((folder/"search-traces.json").read_text())
    assert resources["status"]=="computed_conditional_model"
    assert resources["unknown_solution_count_search_confidence"] is None
    assert resources["frozen_files_unchanged"]
    assert resources["frozen_sha256_before"]==resources["frozen_sha256_after"]
    for relative,digest in resources["frozen_sha256_before"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    runs=traces["runs"]
    assert [run["config"]["seed"] for run in runs]==[7,11,29,19]
    assert traces["bounded_reference_after_search"]["marked_count"]==2
    assert not traces["bounded_reference_after_search"]["used_by_unknown_count_search"]
    assert runs[-1]["status"]=="inconclusive" and runs[-1]["unsat_certificate"] is None
    for run in runs:
        assert not run["marked_count_used_by_search"]
        for attempt in run["attempts"]:
            qasm=(folder/attempt["decomposed_qasm_path"]).read_bytes()
            assert hashlib.sha256(qasm).hexdigest()==attempt["decomposed_qasm_sha256"]
            assert attempt["simulation_input_qasm_sha256"]==attempt["decomposed_qasm_sha256"]
    assert len(resources["paths"])==4
    by_id={run["id"]:run for run in runs}
    for path in resources["paths"]:
        n=by_id[path["id"]]["cnf"]["n_variables"]
        cnf=CNF.from_dict(by_id[path["id"]]["cnf"])
        assert path["status"]=="computed_conditional_model"
        assert path["search_failure_confidence"] is None
        assert path["whole_fixed_replay_budget"]["executions"]==1
        assert path["logical"]["total_output_measurements"]==path["recorded_attempts"]*n
        assert path["logical"]["declared_qubits"]==path["recorded_attempts"]*cnf.num_qubits
        qasm=(folder/path["logical"]["qasm_relative_path"]).read_bytes()
        assert hashlib.sha256(qasm).hexdigest()==path["logical"]["qasm_sha256"]
        for profile in path["profiles"]:
            assert profile["status"]=="computed_conditional_model"
            model=profile["estimate"]
            assert model["transpiler"]["optimization_level"]==0
            assert model["synthesis"]["num_arbitrary_rotations"]==0
            assert model["synthesis"]["model_error_per_execution"]==0
            assert model["canonical_gate_counts"].get("RZ",0)==0
            assert model["canonical_gate_counts"].get("T",0)==7*path["logical"]["source_gate_counts"].get("ccx",0)
            assert not model["synthesis"]["actual_synthesis_sequence_generated"]
            assert not model["synthesis"]["instance_approximation_certificate_verified"]
            assert not model["isa_graph_pruning"]
            assert model["canonical_gate_counts"]["MEAS_Z"]==path["recorded_attempts"]*n
            for point in model["pareto_points"]:
                assert point["physical_failure_union_bound_all_executions"]<=.005+1e-12
                assert point["peak_times_replay_model_runtime_qubit_ns"]==point["peak_physical_qubits"]*point["runtime_per_execution_ns"]


def test_stale_managed_outputs_are_removed_without_touching_unrelated_files(tmp_path):
    paths=["oracle.svg","iteration-circuits/sat-r99.qasm","replay-circuits/sat-seed-999.qasm",
           "notes.txt","iteration-circuits/my-circuit.qasm","custom.svg"]
    for name in paths:
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("test")
    generator.clear_managed_artifacts(tmp_path)
    assert not any((tmp_path/name).exists() for name in paths[:3])
    assert all((tmp_path/name).read_text()=="test" for name in paths[3:])


def test_export_preflight_cannot_write_a_stale_success_artifact(tmp_path):
    with pytest.raises(ValueError,match="32-qubit"):
        generator.export_iteration(CNF(20,(tuple(range(1,21)),)),1,tmp_path/"oversize.qasm")
    assert not (tmp_path/"oversize.qasm").exists()


def test_detached_content_does_not_inherit_an_ancestor_git_commit(tmp_path,monkeypatch):
    import subprocess
    subprocess.run(["git","init","-q",str(tmp_path)],check=True)
    subprocess.run(["git","-C",str(tmp_path),"-c","user.name=Test","-c","user.email=test@example.invalid",
                    "commit","--allow-empty","-q","-m","Unrelated ancestor"],check=True)
    detached=tmp_path/"download"
    detached.mkdir()
    monkeypatch.setattr(generator,"ROOT",detached)
    metadata=generator.environment()
    assert metadata["source_commit"] is None
    assert metadata["source_sha256"]=={}


def test_exact_clifford_t_lowering_is_available_without_changing_default():
    from qiskit import qasm3
    from openquantum.grover_sat import build_sat_grover_circuit
    from openquantum.resource_estimation import prepare_measured_circuit
    circuit=build_sat_grover_circuit(generator.EXAMPLE,1,measure=True)
    default=prepare_measured_circuit(circuit)
    explicit=prepare_measured_circuit(circuit,optimization_level=1)
    assert qasm3.dumps(default)==qasm3.dumps(explicit)
    # Pinned previous behavior captured before the optional adapter change.
    assert hashlib.sha256(qasm3.dumps(default).encode()).hexdigest()=="aaa61812dd4843cb2371037fc5109fa69598a90749c4f7b21a98e6e14889c1a2"
    exact=prepare_measured_circuit(circuit,optimization_level=0)
    assert set(exact.count_ops())<={"h","x","z","s","sdg","t","tdg","cx","measure","barrier"}
    assert exact.count_ops()["t"]+exact.count_ops()["tdg"]==7*circuit.count_ops()["ccx"]


@pytest.mark.parametrize("level", [-1,2,3,True,"0",0.0])
def test_unsupported_compiler_levels_are_rejected(level):
    from qiskit import QuantumCircuit
    from openquantum.resource_estimation import prepare_measured_circuit
    circuit=QuantumCircuit(1,1)
    circuit.measure(0,0)
    with pytest.raises(ValueError,match="optimization levels"):
        prepare_measured_circuit(circuit,optimization_level=level)


def test_iteration_archive_uses_export_whitelist_and_preserves_other_qasm(tmp_path):
    import zipfile
    folder=tmp_path/"iteration-circuits"
    folder.mkdir()
    (folder/"sat-r1.qasm").write_text("generated")
    (folder/"private-notes.qasm").write_text("unrelated")
    generator.archive_iteration_exports(tmp_path,{("sat",1):{}})
    with zipfile.ZipFile(tmp_path/"iteration-circuits.zip") as archive:
        assert archive.namelist()==["sat-r1.qasm"]
        assert archive.read("sat-r1.qasm")==b"generated"
    assert (folder/"private-notes.qasm").read_text()=="unrelated"


def test_detached_generator_runs_without_git_or_provenance_helper(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    detached=tmp_path/"download"
    (detached/"scripts").mkdir(parents=True)
    shutil.copytree(ROOT/"src",detached/"src")
    shutil.copy2(ROOT/"scripts/generate_grover_sat.py",detached/"scripts/generate_grover_sat.py")
    environment=os.environ.copy()
    environment["PATH"]=str(tmp_path/"no-executables")
    environment.pop("PYTHONPATH",None)
    completed=subprocess.run([sys.executable,str(detached/"scripts/generate_grover_sat.py"),
        "--generate-only","--output-dir",str(tmp_path/"result")],cwd=detached,
        env=environment,text=True,capture_output=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    result=json.loads((tmp_path/"result/generation-only.json").read_text())
    assert result["environment"]["source_commit"] is None
    assert result["environment"]["source_sha256"]
    assert "scripts/source_provenance.py" not in result["environment"]["source_sha256"]
    assert result["declared_qubits"]==9
