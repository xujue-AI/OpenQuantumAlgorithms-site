from dataclasses import replace
from fractions import Fraction
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from scipy.linalg import expm

from openquantum import linear_systems_physical as sweep
from openquantum.linear_systems import decompose


def test_predeclared_grid_is_bounded_and_shared_neumann_case_is_deduplicated():
    assert len(sweep.CASES) == sweep.MAX_QUERIES == 13
    assert len({case.case_id for case in sweep.CASES}) == 13
    flattened = [case_id for values in sweep.SLICES.values() for case_id in values]
    assert len(flattened) == 14
    assert flattened.count("neumann-size-n2") == 2
    assert set(flattened) == {case.case_id for case in sweep.CASES}
    conditioning = [case for case in sweep.CASES if "hhl-conditioning" in case.slices]
    assert [(case.offset + sum(case.weights)) / (1 << case.phase_qubits) for case in conditioning] == [.5] * 3
    assert [case.degree for case in sweep.CASES if case.method == "neumann-lcu"] == [9, 9, 9, 4, 24, 51]


@pytest.mark.parametrize("case", sweep.CASES, ids=lambda case: case.case_id)
def test_actual_small_complete_trials_match_independent_normalized_solution_contract(case):
    source, ledger = sweep.build_case(case)
    reference = sweep.validate_case(case, source)
    counts = dict(decompose(source).count_ops())
    assert reference["performed"] is True
    assert reference["source_primitive_joint_probability_max_error"] < 1e-10
    assert reference["expected_polynomial_branch_trace_distance"] < 1e-10
    assert reference["z_absolute_bias"] <= ledger["algorithm_bias_bound"] + 2e-10
    assert reference["source_success_probability"] >= ledger["ideal_probability_lower_bound"] - 1e-10
    assert len(reference["reference_identity_sha256"]) == len(reference["matrix_bytes_sha256"]) == 64
    assert ledger["logical_qubits"] <= 8
    assert counts["measure"] == source.num_qubits == ledger["final_measurements_per_attempt"]
    assert ledger["primitive_operations"] <= 4000
    assert ledger["hardware_output_accuracy_certified"] is False
    assert reference["physical_attempts_executed"] == 0
    assert ledger["contract"]["requested_absolute_error"] == .05
    assert ledger["conditional_total_error_upper_bound"] <= .05
    assert ledger["algorithm_bias_bound"] <= .001 + 1e-12
    attempts = ledger["workload"]["total_capped_attempts"]
    assert ledger["capped_measurements"] == attempts * source.num_qubits
    assert ledger["capped_primitive_counts"] == {name: value * attempts for name, value in counts.items()}
    budget = sweep.physical_budget(ledger)
    assert budget.synthesis_model_error_per_execution == 1e-6
    assert budget.physical_failure_probability == .005
    assert budget.executions == attempts
    assert budget.physical_failure_per_execution * attempts == pytest.approx(.005)
    if case.method == "neumann-lcu":
        assert ledger["model"]["amplification_rounds"] == 0
        assert ledger["model"]["full_source_calls"] == 1
        assert ledger["input_preparation_counts"] == {"ry": 2, "rz": 1}
        assert ledger["model"]["coefficient_conversion_Pauli_bias_bound"] < 1e-12
    else:
        assert ledger["input_preparation_counts"] == {}
        assert ledger["algorithm_bias_bound"] == 0


@pytest.mark.parametrize("probability", [Fraction(1, 64), Fraction(1, 25), Fraction(1, 3), Fraction(5, 8), Fraction(1)])
def test_conditioning_and_retry_exports_are_conservative_by_exact_rational_comparison(probability):
    result = sweep.conditioned_workload(probability, .001)
    p_export = Fraction(result["compiled_model_probability_lower_bound"])
    root = sweep._sqrt_lower(probability)
    eta = Fraction("0.000001")
    assert root * root <= probability
    assert p_export <= (root - eta) ** 2 < probability
    assert Fraction(result["postselection_adjusted_synthesis_bias_bound"]) >= 4 * eta / root
    workload = result["workload"]
    M = workload["accepted_samples"]
    R = workload["attempt_cap_per_accepted_slot"]
    assert M == 13315
    assert M * (1 - p_export) ** R <= Fraction("0.005")
    if R > 1:
        assert M * (1 - p_export) ** (R - 1) > Fraction("0.005")
    assert workload["coherent_eta_certificate_available"] is False
    assert workload["certified_synthesis_or_physical_success_guarantee"] is False


@pytest.mark.parametrize("seed", range(8))
def test_hypothetical_operator_error_conditioning_bounds_actual_small_perturbations(seed):
    rng = np.random.default_rng(seed)
    p = [.02, .05, .25, .8][seed % 4]
    angle = .2 + seed / 17
    state = np.array([math.sqrt(p) * math.cos(angle), math.sqrt(p) * math.sin(angle), math.sqrt(1 - p), 0], dtype=complex)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    generator = matrix + matrix.conj().T
    generator /= np.linalg.norm(generator, 2)
    eta = 1e-6
    perturbed = expm(1j * eta * generator) @ state
    actual_p = float(np.linalg.norm(perturbed[:2]) ** 2)
    actual_z = float((abs(perturbed[0]) ** 2 - abs(perturbed[1]) ** 2) / actual_p)
    original_z = math.cos(2 * angle)
    result = sweep.conditioned_workload(Fraction(str(p)), 0)
    assert actual_p >= result["compiled_model_probability_lower_bound"] - 1e-12
    assert abs(actual_z - original_z) <= result["postselection_adjusted_synthesis_bias_bound"] + 1e-12


@pytest.mark.parametrize("probability,bias", [(0, 0), (-.1, 0), (1.1, 0), (Fraction(1, 10**20), 0), (.5, .02), (.5, float("nan"))])
def test_invalid_or_unaffordable_contracts_fail_before_native_queries(probability, bias):
    with pytest.raises(ValueError):
        sweep.conditioned_workload(probability, bias)


def test_undeclared_case_and_oversized_query_are_rejected_before_materialization(monkeypatch):
    with pytest.raises(ValueError):
        sweep.build_case(replace(sweep.CASES[0], n=24))
    def forbidden(*args):
        raise AssertionError("oversized circuit was lowered")
    monkeypatch.setattr(sweep, "decompose", forbidden)
    with pytest.raises(ValueError, match="eight logical"):
        sweep.query_preflight(QuantumCircuit(9))


def test_expanded_operation_cap_is_enforced_before_physical_query(monkeypatch):
    compiled = QuantumCircuit(1, 1)
    for _ in range(4000): compiled.h(0)
    compiled.measure(0, 0)
    monkeypatch.setattr(sweep, "decompose", lambda _: compiled)
    with pytest.raises(ValueError, match="4000"):
        sweep.query_preflight(QuantumCircuit(1))
