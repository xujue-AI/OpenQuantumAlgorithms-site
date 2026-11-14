"""Independent retargeting, logical-source contracts and complete trial checks."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from openquantum.simulation import retargetable_qsp as rq
from openquantum.simulation.qsp_phase_generation import real_qsp_response

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("retargetable_generator", ROOT/"scripts/generate_simulation_retargetable_qsp.py")
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


@pytest.mark.parametrize("tau", [-2.4, -0.5, 0.0, 0.175, 1.2, 4.1, 7.5])
def test_fresh_phases_retarget_both_parities_and_pass_separate_interval_bound(tau):
    plan = rq.generate_dynamics_phases(tau, decoded_error=1e-5)
    assert plan.odd_degree == plan.even_degree+1 <= 31
    assert plan.decoded_polynomial_error_upper <= 1e-5
    assert plan.cosine_certificate["certified"] and plan.sine_certificate["certified"]
    x = np.linspace(-1, 1, 257)
    response = (real_qsp_response(x, plan.cosine_phases)-1j*real_qsp_response(x, plan.sine_phases))/2
    assert np.max(abs(response-np.exp(-1j*tau*x)/4)) <= plan.encoded_polynomial_error_upper+2e-14
    assert plan.cosine_fit["certified"] is False
    assert "frozen" not in plan.cosine_fit.get("method", "").lower()


def test_fixed_coupling_size_changes_parameter_and_degree_instead_of_reusing_4p1():
    parameters = [sum(abs(c) for c, _ in generator.hs.spin_terms(n))*0.5 for n in (1, 2, 3, 6)]
    plans = [rq.generate_dynamics_phases(tau) for tau in parameters]
    assert parameters == pytest.approx([0.35, 1.2, 2.05, 4.6])
    assert len({p.cosine_phases for p in plans}) == 4
    assert plans[-1].odd_degree > plans[0].odd_degree


@pytest.mark.parametrize("terms", [((-.7, "I"),), ((-.6,"Y"),(.4,"I"),(-.2,"Z"))])
def test_signed_access_has_exact_logical_label_branches_and_identity_padding(terms):
    _, select, access = rq.build_hermitian_pauli_access(terms, return_components=True)
    matrix = Operator(select).data
    paulis = {"I":np.eye(2), "X":np.array([[0,1],[1,0]]), "Y":np.array([[0,-1j],[1j,0]]), "Z":np.diag([1,-1])}
    dimension = 2
    for label in range(2**access.metadata["n_index"]):
        expected = np.eye(dimension) if label >= len(terms) else np.sign(terms[label][0])*paulis[terms[label][1]]
        np.testing.assert_allclose(matrix[label*dimension:(label+1)*dimension,label*dimension:(label+1)*dimension], expected, atol=2e-14, rtol=0)
    np.testing.assert_allclose(matrix.conj().T, matrix, atol=2e-14, rtol=0)
    np.testing.assert_allclose(matrix@matrix, np.eye(len(matrix)), atol=2e-14, rtol=0)
    assert "p" not in select.count_ops() and "mcp" not in select.count_ops()


@pytest.mark.parametrize("relative", [0.0, -math.pi/2, math.pi/2, 0.37])
def test_relative_branch_phases_preserve_the_combined_complex_block(relative):
    terms = ((.6,"Z"),(-.4,"Y"))
    plan = rq.generate_dynamics_phases(.5)
    circuit = rq.build_two_parity_filter(terms,plan.cosine_phases,plan.sine_phases,odd_branch_phase=relative)
    h = .6*np.diag([1,-1])-.4*np.array([[0,-1j],[1j,0]])
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    scalar = (real_qsp_response(eigenvalues,plan.cosine_phases)+np.exp(1j*relative)*real_qsp_response(eigenvalues,plan.sine_phases))/2
    expected = (eigenvectors*scalar)@eigenvectors.conj().T
    np.testing.assert_allclose(Operator(circuit).data[:2,:2],expected,atol=2e-13,rtol=0)
    if relative==math.pi/2:
        selector=circuit.num_qubits-1
        assert sum(i.operation.name=="s" and [circuit.find_bit(q).index for q in i.qubits]==[selector] for i in circuit.data)==1


def test_source_certificate_includes_loaded_weights_and_exact_parameter_gap():
    terms = ((.6,"Z"),(-.4,"Y"))
    time = .5
    core = rq.build_retargetable_evolution(terms,time)
    cert = core.metadata["logical_source_certificate"]
    assert cert["certified"] and cert["prepared_weight_certificate"]["certified"]
    assert cert["decoded_operator_error_upper"] <= 1e-5
    exact_tau = sum((abs(Fraction.from_float(c)) for c,_ in terms),Fraction())*Fraction.from_float(time)
    assert Fraction(int(cert["exact_normalization_time_numerator"]),int(cert["exact_normalization_time_denominator"])) == exact_tau
    assert cert["loading_decoded_error_upper"] >= 0
    assert "floating nested" in cert["scope"].lower()
    h = .6*np.diag([1,-1])-.4*np.array([[0,-1j],[1j,0]])
    actual = Operator(core).data[:2,:2]
    assert np.linalg.norm(4*actual-expm(-1j*time*h),2) <= cert["decoded_operator_error_upper"]+5e-13
    singular = np.linalg.svd(actual,compute_uv=False)
    assert float(singular[-1]**2)+5e-14 >= cert["uniform_acceptance_lower"] > 0


def test_supplied_plan_is_recertified_instead_of_trusting_reported_bounds():
    plan = rq.generate_dynamics_phases(.5)
    counterfeit = replace(plan,decoded_polynomial_error_upper=0.0,encoded_polynomial_error_upper=0.0,
                          cosine_certificate={"uniform_error_upper":0.0},cosine_coefficients=(99.0,))
    audited = rq.audit_dynamics_phase_plan(counterfeit)
    assert audited.decoded_polynomial_error_upper > 0
    assert audited.cosine_coefficients != (99.0,)
    with pytest.raises(ValueError):
        rq.audit_dynamics_phase_plan(replace(plan,cosine_phases=(0.0,)*len(plan.cosine_phases)))
    with pytest.raises(ValueError,match="must match"):
        rq.build_retargetable_evolution(((1.,"X"),),.6,phase_plan=plan)


@pytest.mark.parametrize("changes", [{"tau":float("nan")},{"tau":9.0},{"requested_decoded_error":float("nan")},
    {"requested_decoded_error":1.0},{"even_degree":True},{"odd_degree":32}])
def test_plan_domain_and_nonfinite_fields_fail_closed(changes):
    plan = rq.generate_dynamics_phases(.5)
    with pytest.raises(ValueError): rq.audit_dynamics_phase_plan(replace(plan,**changes))


def test_known_basis_trial_counts_every_reset_measurement_and_failed_attempt():
    core = rq.build_retargetable_evolution(((.7,"X"),),.25)
    trial = rq.lower_circuit(rq.build_measured_trial(core,initial_basis=1))
    assert trial.count_ops()["reset"] == trial.num_qubits
    assert trial.count_ops()["measure"] == trial.num_qubits
    ledger = rq.logical_ledger(rq.lower_circuit(core),trial,acceptance=1/16)
    assert ledger["postselection_retry"]["one_accepted_trial_cap"]==72
    assert ledger["prepare_calls"] == core.metadata["odd_degree"]
    counts={k:int(v) for k,v in trial.count_ops().items()}
    budget=generator.workload_projection(1/16,counts,trial.depth(),trial.num_qubits)
    attempts=budget["complete_attempt_cap_all_slots"]
    assert budget["required_accepted_Z_samples"]==13315
    assert budget["primitive_counts_all_capped_attempts"]=={k:v*attempts for k,v in counts.items()}
    assert budget["logical_qubit_layer_volume_all_capped_attempts"]==trial.num_qubits*trial.depth()*attempts


def test_full_register_primitive_lowering_keeps_relative_and_global_phases():
    core=rq.build_retargetable_evolution(((.6,"Z"),(-.4,"Y")),.5)
    compiled=rq.lower_circuit(core)
    np.testing.assert_allclose(Operator(core).data,Operator(compiled).data,atol=5e-12,rtol=0)
    rng=np.random.default_rng(93)
    state=rng.normal(size=2**core.num_qubits)+1j*rng.normal(size=2**core.num_qubits)
    state/=np.linalg.norm(state)
    np.testing.assert_allclose(Statevector(state).evolve(core).data,Statevector(state).evolve(compiled).data,atol=5e-12,rtol=0)
    assert not {"unitary","initialize","hamiltonian"}&set(compiled.count_ops())


@pytest.mark.parametrize("kwargs", [{"qubits":17,"time":.001},{"qubits":True},{"time":float("nan")},{"error":float("nan")}])
def test_invalid_generator_inputs_do_not_mutate_output(tmp_path,kwargs):
    output=tmp_path/"untouched"
    with pytest.raises(ValueError): generator.generate(output,no_sim=True,no_sweep=True,**kwargs)
    assert not output.exists()


def test_checked_to_no_sim_reuse_keeps_unrelated_files_and_removes_old_evidence(tmp_path):
    generator.generate(tmp_path,qubits=1,time=.25,no_sweep=True)
    assert (tmp_path/"accuracy-acceptance.svg").exists()
    (tmp_path/"notes.txt").write_text("keep")
    (tmp_path/"evolution-n99-time9.qasm").write_text("unrelated reserved-style name")
    with patch.object(rq,"compare_evolution",side_effect=AssertionError("simulator called")):
        generator.generate(tmp_path,qubits=2,time=.5,no_sim=True,no_sweep=True)
    assert not (tmp_path/"accuracy-acceptance.svg").exists()
    assert not (tmp_path/"evolution-n1-time0p25.qasm").exists()
    assert (tmp_path/"notes.txt").read_text()=="keep"
    assert (tmp_path/"evolution-n99-time9.qasm").read_text()=="unrelated reserved-style name"
    report=json.loads((tmp_path/"results.json").read_text())
    assert report["quantum_simulation_performed"] is False
    assert report["cases"][0]["simulation"]["acceptance_probability"] is None
    ledger=json.loads((tmp_path/"logical-resources.json").read_text())["cases"][0]
    assert ledger["workload_proposal"] is None
    assert ledger["source_acceptance_certificate"] is True


def test_standalone_bundle_runs_without_an_ancestor_git_repository(tmp_path):
    package=tmp_path/"package";package.mkdir()
    generator.source_bundle(package)
    subprocess.run(["git","init",str(tmp_path)],check=True,capture_output=True)
    unrelated_source=tmp_path/"src/openquantum/simulation"
    unrelated_source.mkdir(parents=True)
    (unrelated_source/"retargetable_qsp.py").write_text("raise RuntimeError('foreign source imported')")
    command=subprocess.run([sys.executable,str(package/"generate_simulation_retargetable_qsp.py"),
        "--qubits","1","--time",".25","--no-sim","--no-sweep","--output-dir",str(package/"output")],
        cwd=package,capture_output=True,text=True,check=True,timeout=120)
    assert json.loads(command.stdout)["simulation_performed"] is False
    report=json.loads((package/"output/results.json").read_text())
    assert report["provenance"]["source_revision_at_generation"] is None
    assert report["provenance"]["source_working_tree_dirty"] is None
    for filename,digest in report["provenance"]["source_hashes"].items():
        assert hashlib.sha256((package/"output"/filename).read_bytes()).hexdigest()==digest
