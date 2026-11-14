"""Budget, complete-law, timeout and immutable-artifact checks."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
from qiskit import QuantumCircuit

from openquantum import word_taylor_physical as physical


def exact(record):
    base = 16 if record["integer_encoding"] == "hexadecimal" else 10
    return Fraction(int(record["numerator"], base), int(record["denominator"], base))


@pytest.mark.parametrize("case,cap,attempts", [
    ("size-n1-native", 25, 345275),
    ("size-n1-normalization2-oaa1", 4, 55244),
])
def test_same_task_budget_and_synthesis_adjusted_retry_contract(case, cap, attempts):
    contract = physical.physical_contract(case)
    record = contract.record()
    assert contract.plan.n_system == 1 and contract.plan.total_time == 1
    assert contract.plan.h == Fraction(7, 20)
    assert contract.plan.total_accuracy == Fraction(1, 20)
    assert contract.plan.shots == 13811
    assert contract.retry_cap == cap and contract.complete_attempts == attempts
    allocation = record["failure_allocation"]
    assert sum(exact(allocation[key]) for key in ("statistical", "no_acceptance_across_slots",
                                                "physical_faults", "factory_readiness")) == Fraction(1, 100)
    assert exact(allocation["total"]) == Fraction(1, 100)
    requested = Fraction.from_float(contract.synthesis_allowance_float)
    assert requested+physical.NUMERICAL_ROUTED_ACCEPTED_GUARD <= contract.total_norm_allowance
    assert 4*contract.total_norm_allowance/(contract.accepted_amplitude_lower-contract.total_norm_allowance) == Fraction(1, 100)
    with localcontext() as context:
        context.prec = 90
        p = contract.post_synthesis_acceptance_lower
        failure = Decimal(contract.plan.shots)*(1-Decimal(p.numerator)/Decimal(p.denominator))**cap
        assert failure <= Decimal(".001")
    assert record["hardware_confidence_certified"] is False


@pytest.mark.parametrize("case", ("size-n2-native", "unknown", "", None))
def test_no_wider_case_is_silently_estimated(case):
    with pytest.raises(ValueError):
        physical.physical_contract(case)


def test_complete_law_restores_classical_output_permutation():
    circuit = QuantumCircuit(3, 3)
    circuit.reset(range(3))
    circuit.x(0)
    circuit.h(1)
    circuit.measure(0, 2)
    circuit.measure(1, 0)
    circuit.measure(2, 1)
    probabilities, mapping = physical.full_measured_law(circuit)
    assert mapping == {0: 2, 1: 0, 2: 1}
    assert np.allclose(probabilities, [0, 0, 0, 0, .5, .5, 0, 0], atol=1e-14)


def test_complete_law_rejects_overwritten_or_nonterminal_measurements():
    duplicate = QuantumCircuit(3, 3)
    duplicate.reset(range(3))
    duplicate.measure(0, 0)
    duplicate.measure(1, 0)
    duplicate.measure(2, 2)
    with pytest.raises(ValueError, match="bijective"):
        physical.full_measured_law(duplicate)
    early = QuantumCircuit(3, 3)
    early.reset(range(3))
    early.measure(0, 0)
    early.x(1)
    early.measure(1, 1)
    early.measure(2, 2)
    with pytest.raises(ValueError, match="before all final"):
        physical.full_measured_law(early)


@pytest.fixture(scope="module")
def synthesized_native():
    contract = physical.physical_contract("size-n1-native")
    _, _, source = physical.workloads.build_planned_attempt(contract.plan)
    result = physical.instruments.synthesize_instrument(source, contract.synthesis_allowance_float)
    return contract, result


def test_actual_clifford_t_and_complete_failed_outcome_law(synthesized_native):
    contract, synthesized = synthesized_native
    result = physical.check_instrument_laws(contract, synthesized)
    assert result["all_failed_outcomes_included"]
    assert result["source_routed_semantic_operator_residual"] < 1e-10
    assert result["source_accepted_block_residual_vs_ideal"] < 1e-10
    assert result["routed_accepted_block_residual_vs_ideal"] < 1e-10
    assert result["actual_synthesis_conditional_Z_bias"] < .01
    source = result["complete_measured_laws"]["source"]
    final = result["complete_measured_laws"]["clifford_t"]
    assert len(source["outcome_probabilities"]) == 8
    assert abs(sum(final["outcome_probabilities"])-1) < 1e-10
    assert source["failed_attempt_probability"] > 0
    assert final["acceptance_probability"] >= float(contract.post_synthesis_acceptance_lower)
    assert not set(synthesized.circuit.count_ops())-physical.instruments.CLIFFORD_T-{"reset", "measure", "barrier"}
    assert synthesized.circuit.count_ops()["reset"] == 3
    assert synthesized.circuit.count_ops()["measure"] == 3


@pytest.mark.parametrize("timeout", (0, -1, 61, True, float("inf"), "60"))
def test_worker_deadlines_are_explicitly_bounded(timeout):
    with pytest.raises(ValueError):
        physical.validate_timeout(timeout)


def test_timed_out_native_worker_is_terminated():
    started = time.perf_counter()
    with pytest.raises(TimeoutError):
        physical._bounded(time.sleep, (10,), .05)
    assert time.perf_counter()-started < 5


def test_interrupt_during_join_also_terminates_worker(monkeypatch):
    calls = []
    class FakeProcess:
        pid = 123
        exitcode = None
        alive = True
        joins = 0
        def start(self):
            calls.append("start")
        def join(self, seconds):
            self.joins += 1
            calls.append("join")
            if self.joins == 1:
                raise KeyboardInterrupt()
        def is_alive(self):
            return self.alive
        def terminate(self):
            calls.append("terminate")
            self.alive = False
        def kill(self):
            calls.append("kill")
            self.alive = False
        def close(self):
            calls.append("close")
    class FakeContext:
        def Process(self, **kwargs):
            return FakeProcess()
    monkeypatch.setattr(physical.mp, "get_context", lambda _: FakeContext())
    with pytest.raises(KeyboardInterrupt):
        physical._bounded(time.sleep, (10,), 60)
    assert "terminate" in calls and calls[-1] == "close"


def generator():
    path = Path(__file__).resolve().parents[1]/"scripts/generate_word_taylor_physical.py"
    spec = importlib.util.spec_from_file_location("taylor_physical_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_failure_invalidates_old_completed_records_before_qasm_mutation(
        tmp_path, monkeypatch, synthesized_native):
    _, synthesized = synthesized_native
    module = generator()
    output = tmp_path/"output"
    case = output/"size-n1-native"
    case.mkdir(parents=True)
    (output/"manifest.json").write_text('{"physical_estimation_performed": true}')
    (output/"summary.md").write_text("old completed table")
    (case/"report.json").write_text('{"physical_estimation_performed": true}')
    (case/"source-instrument.qasm").write_text("old qasm")
    monkeypatch.setattr(module.study, "bounded_synthesis", lambda *args: (synthesized, 0))
    def timeout(*args):
        raise TimeoutError("forced profile deadline")
    monkeypatch.setattr(module.study, "bounded_estimate", timeout)
    with pytest.raises(TimeoutError):
        module.generate(output, case="size-n1-native")
    assert not (output/"manifest.json").exists()
    assert not (output/"summary.md").exists()
    assert not (case/"report.json").exists()
    assert (case/"source-instrument.qasm").read_text() != "old qasm"


def test_no_estimate_clears_older_physical_claim_and_preserves_unrelated_files(
        tmp_path, monkeypatch, synthesized_native):
    _, synthesized = synthesized_native
    module = generator()
    output = tmp_path/"output"
    omitted = output/"size-n1-normalization2-oaa1"
    omitted.mkdir(parents=True)
    (omitted/"report.json").write_text("{}")
    (omitted/"notes.txt").write_text("keep")
    monkeypatch.setattr(module.study, "bounded_synthesis", lambda *args: (synthesized, 0))
    result = module.generate(output, case="size-n1-native", estimate=False)
    assert not result["physical_estimation_performed"]
    assert result["complete_model_searches"] == 0
    assert result["records"][0]["physical_profiles"] == []
    assert not (omitted/"report.json").exists()
    assert (omitted/"notes.txt").read_text() == "keep"


def test_individual_download_uses_actual_sequences_with_null_git_provenance(
        tmp_path, monkeypatch, synthesized_native):
    _, synthesized = synthesized_native
    module = generator()
    output = tmp_path/"prepared"
    monkeypatch.setattr(module.study, "bounded_synthesis", lambda *args: (synthesized, 0))
    module.generate(output, case="size-n1-native", estimate=False)
    ancestor = tmp_path/"unrelated"
    ancestor.mkdir()
    subprocess.run(["git", "init", "-q", str(ancestor)], check=True)
    folder = ancestor/"download"
    folder.mkdir()
    for name in (*module.SOURCE_NAMES, "LICENSE.txt"):
        shutil.copy2(output/name, folder/name)
    result = subprocess.run([sys.executable, "generate_word_taylor_physical.py", "--no-estimate",
                             "--case", "size-n1-native"], cwd=folder, check=True,
                            capture_output=True, text=True, timeout=60)
    assert "complete source/routed/Clifford+T laws checked" in result.stdout
    manifest = json.loads((folder/"output/manifest.json").read_text())
    assert manifest["source_revision_at_generation"] is None
    assert manifest["source_dirty_at_generation"] is None
    assert manifest["records"][0]["actual_synthesis"]["actual_synthesis_sequences_generated"]
    assert manifest["complete_model_searches"] == 0


ARTIFACTS = Path(__file__).resolve().parents[1]/"assets/generated/word-taylor-physical"


@pytest.mark.parametrize("case", tuple(physical.CASES))
def test_saved_native_frontiers_are_complete_and_recompute_the_full_caps(case):
    path = ARTIFACTS/case/"report.json"
    if not path.exists():
        pytest.skip("reviewed native profile fixtures are not yet included")
    record = json.loads(path.read_text())
    contract = physical.physical_contract(case)
    assert len(record["physical_profiles"]) == 2
    assert record["contract"]["charged_complete_algorithm_attempts"] == contract.complete_attempts
    qasm = (ARTIFACTS/case/"clifford-t-instrument.qasm").read_text()
    t_from_qasm = sum(line.strip().startswith(("t ", "tdg ")) for line in qasm.splitlines())
    t_from_record = sum(record["logical_clifford_t_gate_counts"].get(name, 0) for name in ("t", "tdg"))
    assert t_from_qasm == t_from_record
    for profile in record["physical_profiles"]:
        assert profile["complete_algorithm_attempts"] == contract.complete_attempts
        assert profile["physical_failure_budget"] == .0025
        assert profile["factory_availability_budget"] == .0025
        for point in profile["pareto_points"]:
            assert point["runtime_per_complete_attempt_ns"]*contract.complete_attempts == point["worst_case_runtime_all_attempts_ns"]
            assert sum(row["count"]*row["time_ns"] for row in point["macro_ledger_per_complete_attempt"]) == point["runtime_per_complete_attempt_ns"]
            assert point["data_workspace_physical_qubits"]+point["factory_physical_qubits"] == point["peak_physical_qubits"]
            assert point["full_physical_failure_model"] <= .0025
            assert sum(point["failure_components_all_attempts"].values()) == pytest.approx(point["full_physical_failure_model"], abs=1e-18)
            factory = point["factory_supply"]
            assert factory["states_all_complete_attempts"] == t_from_qasm*contract.complete_attempts
            exact_availability = Fraction(factory["states_all_complete_attempts"])*Fraction(1, 100)**factory["full_pipeline_retry_cap_per_state"]
            assert float(exact_availability) == factory["availability_failure_union_bound"]
            assert exact_availability <= Fraction(1, 400)
    for filename, expected in record["qasm_sha256"].items():
        assert hashlib.sha256((ARTIFACTS/case/filename).read_bytes()).hexdigest() == expected
