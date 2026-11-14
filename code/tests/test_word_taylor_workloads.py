"""Independent arithmetic, matrix and measured-schedule checks for the study."""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from openquantum.simulation import word_taylor as wt
from openquantum.simulation import word_taylor_workloads as work


def decimal(value: Fraction):
    return Decimal(value.numerator)/Decimal(value.denominator)


@pytest.mark.parametrize("value", [Fraction(1), Fraction(1001, 1000), Fraction(3, 2),
                                  Fraction(2), Fraction(500), Fraction(13_811_000), Fraction(10**18)])
def test_exact_log_encloses_high_precision_reference(value):
    with localcontext() as context:
        context.prec = 100
        target = decimal(value).ln()
        assert decimal(work.log_lower(value)) <= target+Decimal("1e-95")
        assert decimal(work.log_upper(value)) >= target-Decimal("1e-95")


@pytest.mark.parametrize("degree", range(9))
@pytest.mark.parametrize("x", [Fraction(0), Fraction(1, 1000), Fraction(1, 5), Fraction(1, 2)])
def test_geometric_majorant_bounds_independent_exponential_remainder(x, degree):
    with localcontext() as context:
        context.prec = 100
        value = decimal(x)
        polynomial = Decimal(1)+sum(value**k/Decimal(math_factorial(k)) for k in range(1, degree+1))
        residual = value.exp()-polynomial
        assert decimal(work.taylor_tail_bound(x, degree)) >= residual-Decimal("1e-95")


def math_factorial(number):
    result = 1
    for integer in range(1, number+1):
        result *= integer
    return result


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", work.MODES)
def test_selected_degree_meets_uniform_bias_and_is_minimal(n, mode):
    plan = work.plan_word_taylor(n, mode=mode)
    assert plan.exposure <= Fraction(1, 2)
    assert plan.bounds["conditional_pauli_bias"] <= plan.total_accuracy/5
    assert 0 < plan.bounds["acceptance_lower"] <= 1
    if plan.degree:
        previous = work.mode_bounds(plan.exposure, plan.degree-1, plan.segments, mode,
                                    plan.bounds["target_exposure_error"])
        assert previous["conditional_pauli_bias"] > plan.total_accuracy/5
    with localcontext() as context:
        context.prec = 100
        p = decimal(work.dyadic_lower(plan.bounds["acceptance_lower"]))
        failure = Decimal(plan.shots)*(1-p)**plan.attempts_per_slot
        assert failure <= decimal(plan.acceptance_failure)
        fixed_mean_failure = 2*(-Decimal(plan.shots)*decimal(plan.statistical_error)**2/2).exp()
        assert fixed_mean_failure <= decimal(plan.statistical_failure)


@pytest.mark.parametrize("mode", work.MODES)
def test_uniform_operator_and_acceptance_bounds_on_full_spectral_interval(mode):
    plan = work.plan_word_taylor(3, mode=mode)
    tau = float(plan.bounds["segment_tail"])
    duration = float(plan.binary_segment_time)
    s = float(plan.bounds["native_taylor_normalization"])
    eigenvalues = np.linspace(-float(plan.alpha), float(plan.alpha), 257)
    terms = np.ones_like(eigenvalues, dtype=complex)
    polynomial = terms.copy()
    for k in range(1, plan.degree+1):
        terms *= -1j*duration*eigenvalues/k
        polynomial += terms
    exact = np.exp(-1j*duration*eigenvalues)
    assert np.max(np.abs(polynomial-exact)) <= tau+1e-14
    block = polynomial/s if mode == "native" else 1.5*polynomial-.5*polynomial*np.abs(polynomial)**2
    full = block**plan.segments
    assert np.min(np.abs(full)**2)+1e-13 >= float(plan.bounds["acceptance_lower"])
    if mode == "native":
        difference = polynomial**plan.segments-exact**plan.segments
    else:
        difference = full-exact**plan.segments
    assert np.max(np.abs(difference)) <= float(plan.bounds["composition_comparison_error"])+1e-13


@pytest.mark.parametrize("mode", work.MODES)
def test_invalid_candidate_does_not_turn_negative_acceptance_into_positive_number(mode):
    bounds = work.mode_bounds(Fraction(1, 2), 0, 32, mode)
    assert bounds["conditional_pauli_bias"] == 4
    if mode == "native":
        assert bounds["acceptance_lower"] == 0
    else:
        assert bounds["segment_comparison_operator_error"] >= 1
        assert bounds["acceptance_lower"] == 0


@pytest.mark.parametrize("mode", work.MODES)
def test_zero_time_has_exact_identity_and_one_retry(mode):
    plan = work.plan_word_taylor(2, total_time=0, mode=mode)
    assert plan.degree == 0
    assert plan.bounds["full_comparison_operator_error"] == 0
    assert plan.bounds["acceptance_lower"] == 1
    assert plan.attempts_per_slot == 1


def test_binary_coefficient_duration_exposure_correction_is_exact():
    plan = work.plan_word_taylor(3, total_time=Fraction(1, 3))
    expected = sum(abs(plan.segments*plan.binary_segment_time*bc-plan.total_time*c)
                   for (c, _), (bc, _) in zip(plan.terms, plan.binary_terms))
    assert plan.bounds["target_exposure_error"] == expected
    assert expected > 0


@pytest.mark.parametrize("mode", work.MODES)
def test_tiny_nonzero_time_near_unit_acceptance_keeps_bounded_retry_arithmetic(mode):
    plan = work.plan_word_taylor(1, Fraction(1, 10**20), mode=mode)
    assert plan.attempts_per_slot == 1
    assert 0 < work.dyadic_lower(plan.bounds["acceptance_lower"]) < 1


def test_segment_policy_selects_smallest_valid_binary_exposure():
    plan = work.plan_word_taylor(1, Fraction(10**20+1, 10**20), h=Fraction(1, 2))
    assert plan.segments == 1
    assert plan.exposure == Fraction(1, 2)


def test_lower_probability_rounding_never_increases_the_claimed_bound():
    for value in (Fraction(0), Fraction(1, 3), Fraction(1), Fraction(10**30-1, 10**30)):
        lowered = work.dyadic_lower(value)
        assert 0 <= lowered <= value
        assert value-lowered < Fraction(1, 1 << 64)


def test_public_mode_bound_api_keeps_displayed_decimal_input_exact():
    bounds = work.mode_bounds(.5, 4, 2, "native")
    assert all(isinstance(value, Fraction) for value in bounds.values())


@pytest.mark.parametrize("kwargs", [{"n": True}, {"n": 0}, {"n": 7},
                                  {"total_time": True}, {"total_time": float("nan")},
                                  {"total_time": 3}, {"accuracy": .0001},
                                  {"accuracy": .5}, {"mode": "unknown"}, {"J": 3}])
def test_unsupported_requested_contract_rejected(kwargs):
    arguments = {"n": 2}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        work.plan_word_taylor(**arguments)


def test_retry_cap_rejects_an_excessive_full_workload():
    with pytest.raises(ValueError, match="retry/shot workload"):
        work.plan_word_taylor(6, total_time=2)


def test_degree_cap_does_not_silently_relax_accuracy(monkeypatch):
    monkeypatch.setattr(work, "MAX_DEGREE", 0)
    with pytest.raises(ValueError, match="no degree"):
        work.plan_word_taylor(1)


def test_large_exact_bounds_have_lossless_portable_serialization():
    plan = work.plan_word_taylor(4, mode="normalization2-oaa1")
    record = json.loads(json.dumps(plan.record()))
    bound = record["uniform_ideal_bounds"]["acceptance_lower"]
    base = 16 if bound["integer_encoding"] == "hexadecimal" else 10
    reconstructed = Fraction(int(bound["numerator"], base), int(bound["denominator"], base))
    assert reconstructed == plan.bounds["acceptance_lower"]


@lru_cache(maxsize=4)
def checked_case(n, mode):
    accuracy = Fraction(1, 10) if n == 2 else Fraction(1, 20)
    plan = work.plan_word_taylor(n, accuracy=accuracy, mode=mode)
    access, segment, attempt = work.build_planned_attempt(plan)
    compiled = wt.decompose_word_circuit(attempt)
    check = work.check_tiny_circuit(plan, access, segment, compiled)
    return plan, access, segment, attempt, compiled, check


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("mode", work.MODES)
def test_actual_gate_blocks_and_complete_measured_accepted_schedule(n, mode):
    plan, _, _, _, compiled, result = checked_case(n, mode)
    assert result["performed"]
    assert result["source_block_residual_vs_independent_polynomial"] < 5e-10
    assert result["complete_measured_branch_probability_residual"] < 5e-10
    assert result["coherent_lowering_random_full_register_residual"] < 5e-10
    assert result["actual_known_input_Z_bias"] <= float(plan.total_accuracy/5)
    assert result["actual_full_map_minimum_acceptance"]+5e-10 >= float(plan.bounds["acceptance_lower"])
    counts = compiled.count_ops()
    ancillas = compiled.num_qubits-plan.n_system
    assert counts["measure"] == plan.segments*ancillas+plan.n_system
    assert counts["reset"] == compiled.num_qubits+(plan.segments-1)*ancillas


@pytest.mark.parametrize("mode", work.MODES)
def test_complete_ledger_charges_source_inverses_reflections_and_full_caps(mode):
    plan, _, _, attempt, compiled, _ = checked_case(1, mode)
    record = work.primitive_ledger(plan, attempt, compiled)
    calls = attempt.metadata
    factor = 1 if mode == "native" else 3
    assert calls["total_A_or_inverse_calls_per_attempt"] == plan.segments*factor
    assert calls["degree_prepare_or_inverse_calls_per_attempt"] == 2*plan.segments*factor
    assert calls["term_prepare_or_inverse_calls_per_attempt"] == 2*plan.segments*factor*plan.degree
    assert calls["SELECT_or_inverse_calls_per_attempt"] == plan.segments*factor
    assert calls["zero_ancilla_reflections_per_attempt"] == (0 if mode == "native" else 2*plan.segments)
    assert record["primitive_operations_full_cap"] == compiled.size()*plan.shots*plan.attempts_per_slot
    assert record["primitive_gate_counts_full_cap"]["measure"] == compiled.count_ops()["measure"]*plan.full_attempts
    assert record["physical_resource_estimate"] is None


def test_measured_checker_rejects_missing_or_early_data_readout():
    from qiskit import QuantumCircuit
    missing = QuantumCircuit(2, 1)
    missing.reset(range(2))
    missing.measure(1, 0)
    with pytest.raises(ValueError, match="every data wire"):
        work.accepted_measured_probabilities(missing, 1)
    early = QuantumCircuit(1, 1)
    early.reset(0)
    early.measure(0, 0)
    early.x(0)
    with pytest.raises(ValueError, match="safely deferred"):
        work.accepted_measured_probabilities(early, 1)
    overwritten = QuantumCircuit(2, 1)
    overwritten.reset(range(2))
    overwritten.h(0)
    overwritten.measure(0, 0)
    overwritten.measure(1, 0)
    with pytest.raises(ValueError, match="distinct classical"):
        work.accepted_measured_probabilities(overwritten, 2)


def generator_module():
    path = Path(__file__).resolve().parents[1]/"scripts/generate_word_taylor_workloads.py"
    spec = importlib.util.spec_from_file_location("word_workload_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_sim_generation_removes_only_managed_stale_outputs(tmp_path):
    generator = generator_module()
    output = tmp_path/"study"
    stale = output/"size-n2-native"
    stale.mkdir(parents=True)
    (stale/"record.json").write_text("{}")
    (stale/"personal-note.txt").write_text("keep")
    result = generator.generate(output, simulate=False, selected_case="size-n1-native")
    assert result["records"][0]["circuit_generation"]["performed"]
    assert result["records"][0]["numerical_validation"]["performed"] is False
    assert not (stale/"record.json").exists()
    assert (stale/"personal-note.txt").read_text() == "keep"


def test_materialization_rejection_is_retained_as_a_partial_cell(tmp_path, monkeypatch):
    generator = generator_module()
    def rejected(_):
        raise ValueError("test primitive export cap")
    monkeypatch.setattr(generator.wt, "decompose_word_circuit", rejected)
    result = generator.generate(tmp_path/"study", selected_case="size-n1-native")
    record = result["records"][0]
    assert record["plan"]["uniform_ideal_bounds"]["acceptance_lower"]["approximate_decimal"] > 0
    assert not record["circuit_generation"]["performed"]
    assert not record["numerical_validation"]["performed"]
    assert not (tmp_path/"study/size-n1-native/measured-attempt.qasm").exists()


def test_individual_download_runs_without_git_or_private_source(tmp_path):
    generator = generator_module()
    generated = tmp_path/"generated"
    generator.generate(generated, simulate=False, selected_case="size-n1-native")
    ancestor = tmp_path/"unrelated"
    ancestor.mkdir()
    subprocess.run(["git", "init", "-q", str(ancestor)], check=True)
    download = ancestor/"download"
    download.mkdir()
    for name in generator.source_files():
        shutil.copy2(generated/name, download/name)
    shutil.copy2(generated/"LICENSE.txt", download/"LICENSE.txt")
    result = subprocess.run([sys.executable, "generate_word_taylor_workloads.py", "--no-sim",
                             "--case", "size-n1-native"], cwd=download, capture_output=True, text=True,
                            check=True, timeout=60)
    assert "size-n1-native: circuit-only" in result.stdout
    manifest = json.loads((download/"output/manifest.json").read_text())
    assert manifest["source_revision_at_generation"] is None
    assert manifest["source_dirty_at_generation"] is None
    assert manifest["records"][0]["sampling"]["performed"] is False
    assert manifest["physical_estimation_performed"] is False
