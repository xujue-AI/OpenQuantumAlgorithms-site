"""Adaptive-order circuit checks and complete bounded factoring-policy traces."""
import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Statevector

from openquantum.shor import build_order_finding_circuit
from openquantum.shor_driver import (
    MAX_DRIVER_INPUT, FactoringConfig, build_iterative_order_circuit,
    factor_with_quantum_trials, integer_nth_root, is_prime_bounded,
    iterative_distribution, numerical_policy_audit, perfect_power,
    preliminary_checks, prime_factorization, recorded_path_ledger,
    recommended_phase_bits, sample_iterative_order,
)


def test_bounded_primality_matches_independent_sieve_exhaustively():
    prime = np.ones(MAX_DRIVER_INPUT + 1, dtype=bool)
    prime[:2] = False
    for divisor in range(2, math.isqrt(MAX_DRIVER_INPUT) + 1):
        if prime[divisor]:
            prime[divisor * divisor::divisor] = False
    assert all(is_prime_bounded(value) == prime[value] for value in range(2, MAX_DRIVER_INPUT + 1))


def test_perfect_power_and_integer_roots_against_exact_bounded_answers():
    powers = {pow(base, exponent) for base in range(2, 64) for exponent in range(2, 13) if pow(base, exponent) <= MAX_DRIVER_INPUT}
    for value in range(2, MAX_DRIVER_INPUT + 1):
        result = perfect_power(value)
        assert (result is not None) == (value in powers)
        if result:
            assert pow(*result) == value
    for value in [2, 3, 9, 27, 225, 2187, 4095]:
        for exponent in [2, 3, 7, 12]:
            root = integer_nth_root(value, exponent)
            assert root**exponent <= value < (root + 1)**exponent


@pytest.mark.parametrize("value,status,factors", [(2, "prime", []), (17, "prime", []), (16, "classical_factor", [2, 8]), (27, "classical_factor", [3, 9]), (225, "classical_factor", [15, 15]), (15, "quantum_needed", [])])
def test_prechecks_exact_and_explicit(value, status, factors):
    result = preliminary_checks(value)
    assert result["decision"] == status
    assert result["factors"] == factors


@pytest.mark.parametrize("modulus,base,phase_bits", [(3, 2, 3), (15, 2, 4), (21, 2, 6)])
def test_adaptive_arithmetic_distribution_matches_independent_full_register_circuit(modulus, base, phase_bits):
    adaptive = iterative_distribution(modulus, base, phase_bits)
    coherent = Statevector.from_instruction(build_order_finding_circuit(modulus, base, phase_bits)).probabilities(list(range(phase_bits)))
    np.testing.assert_allclose(adaptive, coherent, atol=2e-12, rtol=0)
    assert abs(sum(adaptive) - 1) < 2e-13


def test_source_dynamic_circuit_has_real_feedback_and_all_resets():
    circuit = build_iterative_order_circuit(15, 2, 4)
    assert circuit.num_qubits == 11
    assert circuit.num_clbits == 4
    assert circuit.count_ops()["if_else"] == 6
    assert circuit.count_ops()["reset"] == 14
    assert circuit.count_ops()["measure"] == 4
    qasm = qasm3.dumps(circuit)
    assert "if (c[0])" in qasm and "reset q[0]" in qasm
    assert "cM2" in qasm and "cM4" in qasm
    assert "gate" in qasm  # arithmetic boxes have exported definitions


@pytest.mark.parametrize("bits,feedback", [((0, 0, 0, 0), 0), ((1, 1, 1, 1), 6), ((0, 0, 1, 0), 1)])
def test_observed_path_ledger_counts_actual_feedback_resets_and_readout(bits, feedback):
    ledger = recorded_path_ledger(15, 2, bits)
    assert ledger["executed_feedback_phases"] == feedback
    assert ledger["conditional_checks_in_dynamic_program"] == 6
    assert ledger["primitive_gate_counts"]["measure"] == 4
    assert ledger["primitive_gate_counts"]["reset"] == 14
    assert ledger["primitive_operations"] == sum(ledger["primitive_gate_counts"].values())


def test_standard_precision_recycled_sampler_and_reproducibility():
    assert recommended_phase_bits(15) == 8
    assert recommended_phase_bits(21) == 10
    first = sample_iterative_order(21, 10, rng=np.random.default_rng(1))
    second = sample_iterative_order(21, 10, rng=np.random.default_rng(1))
    assert first == second
    assert first["total_qubits"] == 13
    assert first["phase_bits"] == 10
    assert first["order_reference_used"] is False
    assert len(first["steps"]) == 10
    assert first["selected_joint_probability"] > 0
    assert all(step["work_register_zero_probability"] > 1 - 1e-10 for step in first["steps"])


def test_failed_quantum_trials_then_non_coprime_gcd_success_are_counted():
    run = factor_with_quantum_trials(15, config=FactoringConfig(seed=1, max_base_draws=2, shots_per_coprime_base=2, max_quantum_trials=4, phase_bits=4), base_sequence=(14, 5))
    assert run["status"] == "factored_classical_gcd"
    assert run["factors"] == [3, 5]
    assert run["base_draws"] == 2 and run["quantum_trials"] == 2
    assert len(run["trace"][0]["quantum_attempts"]) == 2
    assert all(attempt["failure_reason"] for attempt in run["trace"][0]["quantum_attempts"])
    assert run["trace"][1]["gcd"] == 5 and not run["trace"][1]["quantum_attempts"]
    assert run["executed_quantum_primitive_counts"]["measure"] == 8
    assert run["executed_quantum_primitive_counts"]["reset"] == 28


def test_random_base_quantum_factor_success_and_whole_trace_aggregation():
    run = factor_with_quantum_trials(21, config=FactoringConfig(seed=1, max_base_draws=4, shots_per_coprime_base=1, max_quantum_trials=4))
    assert run["status"] == "factored_quantum_trial"
    assert run["factors"] == [3, 7]
    assert run["phase_bits"] == 10 and run["peak_logical_qubits"] == 13
    assert run["base_draws"] == run["quantum_trials"] == 1
    assert run["trace"][0]["base"] == 10
    assert run["trace"][0]["quantum_attempts"][0]["outcome"] == 853
    counts = {}
    for draw in run["trace"]:
        for attempt in draw["quantum_attempts"]:
            for name, count in attempt["logical_ledger"]["primitive_gate_counts"].items():
                counts[name] = counts.get(name, 0) + count
    assert counts == run["executed_quantum_primitive_counts"]
    assert run["generic_success_certificate"] is None


def test_censored_quantum_cap_and_between_base_gcd_semantics():
    capped = factor_with_quantum_trials(15, config=FactoringConfig(max_base_draws=2, shots_per_coprime_base=2, max_quantum_trials=1, phase_bits=4), base_sequence=(14, 5))
    assert capped["status"] == "exhausted_quantum_trials"
    assert capped["base_draws"] == capped["quantum_trials"] == 1
    one_shot = factor_with_quantum_trials(15, config=FactoringConfig(max_base_draws=2, shots_per_coprime_base=1, max_quantum_trials=1, phase_bits=4), base_sequence=(14, 5))
    assert one_shot["status"] == "factored_classical_gcd"
    assert one_shot["base_draws"] == 2 and one_shot["quantum_trials"] == 1


def test_budget_exhaustion_and_resource_limit_do_not_claim_factor_success():
    run = factor_with_quantum_trials(15, config=FactoringConfig(max_base_draws=1, shots_per_coprime_base=1, max_quantum_trials=1, phase_bits=4), base_sequence=(14,))
    assert run["status"] == "exhausted_base_draws" and run["factors"] == []
    unsupported = factor_with_quantum_trials(511, config=FactoringConfig(max_base_draws=1), base_sequence=(2,))
    assert unsupported["status"] == "quantum_resource_limit"
    assert unsupported["quantum_trials"] == 0 and unsupported["factors"] == []


@pytest.mark.parametrize("value", [2, 16, 27, 225])
def test_recursive_wrapper_complete_product_and_prime_labels(value):
    result = prime_factorization(value, config=FactoringConfig(seed=1))
    assert result["status"] == "complete_prime_factorization"
    assert result["unresolved_components"] == []
    assert math.prod(result["prime_factors"]) == value
    assert all(is_prime_bounded(prime) for prime in result["prime_factors"])
    assert result["quantum_trials"] <= result["global_quantum_cap"]
    if value == 225:
        assert result["prime_factors"] == [3, 3, 5, 5]


def test_recursive_wrapper_unsupported_component_keeps_product_invariant():
    result = prime_factorization(511, config=FactoringConfig(seed=2, max_base_draws=1))
    assert math.prod(result["prime_factors"]) * math.prod(result["unresolved_components"]) == 511
    assert result["quantum_trials"] <= 16


def test_finite_policy_audit_includes_cap_and_all_classical_base_successes():
    uncapped = numerical_policy_audit(15, FactoringConfig(max_base_draws=3, shots_per_coprime_base=1, max_quantum_trials=3))
    assert uncapped["ideal_failure_probability"] == pytest.approx((4 / 13) ** 3)
    censored = numerical_policy_audit(15, FactoringConfig(max_base_draws=3, shots_per_coprime_base=2, max_quantum_trials=1))
    assert censored["ideal_failure_probability"] == pytest.approx(4 / 13)
    between = numerical_policy_audit(15, FactoringConfig(max_base_draws=3, shots_per_coprime_base=1, max_quantum_trials=1))
    assert between["ideal_failure_probability"] == pytest.approx(28 / 169)
    assert censored["is_certified_confidence_bound"] is False
    assert censored["global_quantum_cap_censoring_included"] is True


@pytest.mark.parametrize("kwargs", [{"phase_bits": 0}, {"phase_bits": True}, {"max_quantum_trials": 0}, {"max_base_draws": 17}, {"seed": -1}])
def test_invalid_driver_config_rejected(kwargs):
    with pytest.raises(ValueError):
        FactoringConfig(**kwargs)


def test_validation_guards_cannot_be_raised_by_caller():
    with pytest.raises(ValueError, match="immutable"):
        iterative_distribution(15, 2, 4, max_phase_bits=24)
    with pytest.raises(ValueError, match="memory"):
        iterative_distribution(127, 2, 6)
    with pytest.raises(ValueError):
        numerical_policy_audit(127, FactoringConfig())
    with pytest.raises(ValueError):
        factor_with_quantum_trials(4096)
