# SPDX-License-Identifier: MIT
from fractions import Fraction
import importlib.util
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Statevector

from openquantum.reciprocal_physical_sweep import (
    CASES,CONTRACT,MAX_NATIVE_PRIMITIVES,MAX_NATIVE_WIRES,
    build_case,chain_terms,independent_matrix,replay_qasm,resource_budget,select_degree,
)
from openquantum.linear_systems_qsvt_reciprocal import (
    ReciprocalPolynomial,exact_chebyshev_coefficients,prepared_reciprocal,
)


RESOURCE_ADAPTER_SOURCE="src/openquantum/resource_estimation.py"
HISTORICAL_RESOURCE_ADAPTER_SHA256="267329af4b100c7f9c82900f1f2713f934ec710e50fd8999d331c0f4b84654de"
HISTORICAL_RESOURCE_ADAPTER_SOURCE=(
    "assets/generated/linear-systems/physical-sweeps/code/openquantum/resource_estimation.py"
)


def recorded_producer_source(root,name,digest):
    """Bind old reports to the exact archived adapter, and fresh ones to live source."""
    source=(root/name).read_bytes()
    if name==RESOURCE_ADAPTER_SOURCE and hashlib.sha256(source).hexdigest()!=digest:
        # This already-public snapshot has exactly the adapter bytes used by
        # the historical reciprocal grid. Later optional context support does
        # not rewrite the old native records or make their provenance current.
        assert digest==HISTORICAL_RESOURCE_ADAPTER_SHA256,"Unknown historical resource adapter"
        source=(root/HISTORICAL_RESOURCE_ADAPTER_SOURCE).read_bytes()
    assert hashlib.sha256(source).hexdigest()==digest,f"Recorded producer digest differs: {name}"
    return source


@pytest.fixture
def recorded_adapter_sources(tmp_path,monkeypatch):
    original_root=Path(__file__).resolve().parents[1]
    archived=(original_root/HISTORICAL_RESOURCE_ADAPTER_SOURCE).read_bytes()
    # Unit fixtures bind the exact supplied archive bytes. A fresh full sweep
    # may update this code copy; the canonical report test independently keeps
    # the actual historical producer pin unless its report matches live source.
    monkeypatch.setitem(globals(),"HISTORICAL_RESOURCE_ADAPTER_SHA256",hashlib.sha256(archived).hexdigest())
    current_path=tmp_path/RESOURCE_ADAPTER_SOURCE
    current_path.parent.mkdir(parents=True)
    current_path.write_bytes(b"# current adapter with optional context support\n")
    archived_path=tmp_path/HISTORICAL_RESOURCE_ADAPTER_SOURCE
    archived_path.parent.mkdir(parents=True)
    archived_path.write_bytes(archived)
    return tmp_path,current_path,archived_path


def test_historical_adapter_source_binds_exact_public_snapshot(recorded_adapter_sources):
    root,current,archived=recorded_adapter_sources
    assert current.read_bytes()!=archived.read_bytes()
    assert recorded_producer_source(root,RESOURCE_ADAPTER_SOURCE,HISTORICAL_RESOURCE_ADAPTER_SHA256)==archived.read_bytes()


def test_fresh_adapter_source_binds_current_without_archive(recorded_adapter_sources):
    root,current,archived=recorded_adapter_sources
    source=current.read_bytes()
    archived.unlink()
    assert recorded_producer_source(root,RESOURCE_ADAPTER_SOURCE,hashlib.sha256(source).hexdigest())==source


@pytest.mark.parametrize("mutation",["missing","tampered","unknown"])
def test_historical_adapter_binding_rejects_missing_tampered_or_unknown(recorded_adapter_sources,mutation):
    root,current,archived=recorded_adapter_sources
    digest=HISTORICAL_RESOURCE_ADAPTER_SHA256
    if mutation=="missing":archived.unlink()
    elif mutation=="tampered":archived.write_bytes(archived.read_bytes()+b"# altered archived producer\n")
    else:digest="0"*64
    with pytest.raises(FileNotFoundError if mutation=="missing" else AssertionError):
        recorded_producer_source(root,RESOURCE_ADAPTER_SOURCE,digest)


def test_other_recorded_sources_keep_exact_current_binding(recorded_adapter_sources):
    root,_,_=recorded_adapter_sources
    name="src/openquantum/reciprocal_phase_certificate.py"
    (root/name).write_bytes(b"# changed certificate implementation\n")
    with pytest.raises(AssertionError,match="Recorded producer digest differs"):
        recorded_producer_source(root,name,hashlib.sha256(b"# recorded certificate implementation\n").hexdigest())


@pytest.mark.parametrize("case",CASES)
def test_exact_family_promised_interval_and_all_spins(case):
    terms=chain_terms(case)
    n,p,q=case.n,case.delta.numerator,case.delta.denominator
    assert len(terms)==2*n
    assert sum(abs(c) for c,w in terms)==2*q*(2*n-1)
    constant=terms[0][0];radius=sum(abs(c) for c,w in terms[1:])
    assert Fraction(constant-radius,constant+radius)==case.delta
    values=np.linalg.eigvalsh(independent_matrix(terms))
    assert min(values)>=float(case.delta)-1e-12
    assert max(values)<=1+1e-12
    assert all(any(w[i]!="I" for c,w in terms) for i in range(n))


@pytest.mark.parametrize("delta,k,retry",[(Fraction(3,4),5,140),(Fraction(3,5),10,255),(Fraction(1,2),16,403),(Fraction(2,5),26,584)])
def test_uniform_phase_adjusted_degree_and_retry_rules(delta,k,retry):
    selected,phases,certificate,bounds=select_degree(delta)
    assert selected==k
    assert certificate.target_chebyshev_coefficients==exact_chebyshev_coefficients(k)
    model=ReciprocalPolynomial(k,delta);eta=certificate.dyadic_upper_bound
    rigorous_bias=2*(model.relative_tail+2*eta/model.scale)/(2-model.relative_tail)
    assert rigorous_bias<=Fraction(1,100)
    assert ReciprocalPolynomial(k-1,delta).pauli_bias_bound>Fraction(1,100)
    chosen=next(c for c in CASES if c.n==2 and c.delta==delta)
    _,_,metadata=build_case(chosen)
    assert metadata["workload"]["accepted_samples"]==13811
    assert metadata["workload"]["retry_cap_per_slot"]==retry
    assert metadata["workload"]["total_capped_attempts"]==13811*retry
    assert metadata["workload"]["success_lower_bound_exact"]==bounds["retry_success_lower_bound_exact"]
    assert metadata["workload"]["hoeffding_bound_certified"]
    assert metadata["workload"]["verified_union_empty_slot_bound"]<=.001
    assert metadata["validation"]["success_observed_ideal_numeric"]>metadata["workload"]["success_lower_bound_used"]


@pytest.mark.parametrize("case",CASES[:7])
def test_actual_bounded_complete_circuits_and_fixed_contract(case):
    measured,qasm,metadata=build_case(case)
    assert measured.num_qubits<=MAX_NATIVE_WIRES
    assert metadata["primitive_operations"]<=MAX_NATIVE_PRIMITIVES
    assert measured.count_ops()["measure"]==measured.num_qubits
    assert metadata["uniform_contract"]==CONTRACT
    assert metadata["coefficient_conversion_l1_error_exact"]=="0"
    assert metadata["validation"]["qasm_state_error_modulo_global_phase"]<1e-10
    assert metadata["validation"]["qasm_full_measured_probability_error"]<1e-11
    assert metadata["validation"]["z_bias"]<=.01
    budget=resource_budget(metadata)
    assert budget.executions==metadata["workload"]["total_capped_attempts"]
    assert budget.physical_failure_probability==.005
    assert budget.synthesis_model_error_per_execution==1e-6
    assert "pi" not in qasm  # numeric literal policy; include filename has no pi


@pytest.mark.parametrize("case",CASES[7:])
def test_unsupported_fixed_accuracy_cases(case):
    with pytest.raises(ValueError,match="degree"):
        build_case(case)


def test_replay_random_complex_input_and_numeric_pi_regression():
    case=CASES[6]  # degree53 exposed default symbolic-pi approximation
    measured,qasm,metadata=build_case(case)
    phases=select_degree(case.delta)[1]
    source=prepared_reciprocal(chain_terms(case),metadata["K"],case.delta,phases=phases)
    rng=np.random.default_rng(73)
    initial=rng.normal(size=1<<measured.num_qubits)+1j*rng.normal(size=1<<measured.num_qubits)
    initial/=np.linalg.norm(initial)
    replayed=replay_qasm(qasm,initial)
    expected=Statevector(initial).evolve(source).data
    phase=np.angle(np.vdot(expected,replayed))
    assert np.linalg.norm(replayed-expected*np.exp(1j*phase))<1e-11
    default_export=qasm3.dumps(measured)
    default_replay=replay_qasm(default_export)
    source_state=Statevector.from_instruction(source).data
    phase=np.angle(np.vdot(source_state,default_replay))
    # The error is reproducibly above the sweep's threshold. It is fixed by
    # preserving literals, not by broadening the validation tolerance.
    assert np.linalg.norm(default_replay-source_state*np.exp(1j*phase))>1e-10


def test_replay_rejects_nonfinal_measurements_and_unknown_syntax():
    header='OPENQASM 3.0;\nqubit[1] q;\nbit[1] meas;\n'
    with pytest.raises(ValueError):replay_qasm(header+'reset q[0];\nmeas[0] = measure q[0];')
    with pytest.raises(ValueError):replay_qasm(header+'meas[0] = measure q[0];\nx q[0];')
    with pytest.raises(ValueError):replay_qasm(header+'rx(__import__("os")) q[0];\nmeas[0] = measure q[0];')


def generator_module():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("reciprocal_physical_generator",root/"scripts/generate_reciprocal_physical_sweep.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_predeclaration_and_export_provenance(tmp_path):
    generator=generator_module()
    plan=generator.predeclared_plan()
    assert len(plan["cases"])==9
    assert plan["native_query_cap"]["cases"]==7
    assert plan["uniform_contract"]==CONTRACT
    assert generator.fingerprint(plan)==generator.fingerprint(generator.predeclared_plan())
    assert generator.source_revision(tmp_path) is None
    assert generator.source_revision(Path(__file__).resolve().parents[1]/"algorithms") is None


def test_precise_exporter_binding_does_not_mutate_shared_globals(monkeypatch):
    generator=generator_module()
    # Function globals and bytecode are isolated while other estimator inputs
    # retain their values. This tests the exporter adapter without native work.
    def probe(circuit,*,budget):
        return qasm3.dumps(circuit),budget
    original_namespace=probe.__globals__["qasm3"]
    monkeypatch.setattr(generator,"estimate_circuit_resources",probe)
    from qiskit import QuantumCircuit
    circuit=QuantumCircuit(1);circuit.rz(np.pi+1e-9,0)
    emitted,budget=generator.numeric_literal_estimate(circuit,budget="same")
    assert "3.141592654589793" in emitted
    assert "pi" not in emitted
    assert budget=="same"
    assert probe.__globals__["qasm3"] is original_namespace


def test_partial_run_keeps_unsupported_reason(tmp_path):
    generator=generator_module()
    manifest=generator.run(tmp_path,no_estimate=True,selected={"n2-gap1of3"})
    assert manifest["cases"][0]["status"]=="unsupported_degree_at_fixed_accuracy"
    assert not manifest["cases"][0]["circuit_generated"]
    assert manifest["cases"][0]["budgets_relaxed"] is False


def test_canonical_native_reports_match_trial_and_total_budgets():
    root=Path(__file__).resolve().parents[1]
    output=root/"assets/generated/linear-systems/reciprocal-qsvt-physical"
    if not (output/"manifest.json").exists():pytest.skip("canonical physical grid not generated in this code-only export")
    manifest=json.loads((output/"manifest.json").read_text())
    assert manifest["native_queries_completed"]==7
    assert manifest["native_queries_requested"]==7
    assert manifest["uniform_contract"]==CONTRACT
    for name,digest in manifest["source_provenance"]["source_hashes_sha256"].items():
        assert hashlib.sha256(recorded_producer_source(root,name,digest)).hexdigest()==digest
    for case in manifest["cases"]:
        if not case["circuit_generated"]:
            assert case["status"]=="unsupported_degree_at_fixed_accuracy"
            continue
        logical=json.loads((output/case["logical_report"]).read_text())
        physical=json.loads((output/case["physical_report"]).read_text())
        trial=(output/case["trial_export"]).read_bytes()
        digest=hashlib.sha256(trial).hexdigest()
        assert digest==logical["qasm_sha256"]==physical["qasm_sha256"]
        assert physical["budget"]["executions"]==logical["workload"]["total_capped_attempts"]
        assert physical["budget"]["physical_failure_probability"]==.005
        assert physical["budget"]["synthesis_model_error_per_execution"]==1e-6
        assert physical["physical_profile"]==CONTRACT["profile"]
        assert physical["pareto_points"]
        for point in physical["pareto_points"]:
            expected=point["runtime_per_execution_ns"]*physical["budget"]["executions"]
            assert math.isclose(point["serial_runtime_all_executions_ns"],expected,rel_tol=1e-12)
            assert point["physical_failure_union_bound_all_executions"]<=.005+1e-12
            assert math.isclose(point["peak_footprint_times_serial_seconds"],point["peak_physical_qubits"]*expected/1e9,rel_tol=1e-12)
