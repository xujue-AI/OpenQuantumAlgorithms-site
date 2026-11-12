import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from openquantum.amplitude_estimation import (
    MAX_SIM_QUBITS, amplitude_workload, analytic_amplitude_distribution,
    build_amplitude_estimation, decode_amplitude, exact_median_interval_failure, median_failure_bound,
    repetitions_for_failure, simulate_amplitude_estimation,
    threshold_grover_iterate, threshold_predicate, threshold_preparation,
    uniform_amplitude_error_bound, zero_reflection,
)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_predicate_is_reversible_comparator(n):
    size = 2**n
    for threshold in range(size + 1):
        matrix = Operator(threshold_predicate(n, threshold)).data
        expected = np.zeros_like(matrix)
        for flag in (0, 1):
            for value in range(size):
                output = value + (flag ^ int(value < threshold)) * size
                expected[output, value + flag * size] = 1
        np.testing.assert_allclose(matrix, expected, atol=2e-13, rtol=0)


@pytest.mark.parametrize("n,threshold", [(1, 0), (1, 1), (1, 2), (2, 1), (3, 5), (4, 11)])
def test_preparation_matches_uniform_flagged_state(n, threshold):
    state = Statevector.from_instruction(threshold_preparation(n, threshold)).data
    expected = np.zeros(2**(n + 1), complex)
    for value in range(2**n):
        expected[value + int(value < threshold) * 2**n] = 2**(-n / 2)
    np.testing.assert_allclose(state, expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_zero_reflection_has_exact_sign(width):
    expected = np.eye(2**width)
    expected[0, 0] = -1
    np.testing.assert_allclose(Operator(zero_reflection(width)).data, expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize("n,threshold", [(1, 0), (1, 1), (1, 2), (2, 1), (2, 3), (3, 5)])
def test_iterate_against_independent_state_reflection(n, threshold):
    psi = np.zeros(2**(n + 1), complex)
    for value in range(2**n):
        psi[value + int(value < threshold) * 2**n] = 2**(-n / 2)
    reflection = 2 * np.outer(psi, psi.conj()) - np.eye(len(psi))
    good = np.diag([1] * 2**n + [-1] * 2**n)
    np.testing.assert_allclose(Operator(threshold_grover_iterate(n, threshold)).data,
                               reflection @ good, atol=3e-13, rtol=0)


@pytest.mark.parametrize("n,threshold,bits", [(1, 0, 3), (1, 1, 3), (1, 2, 3), (2, 1, 3), (2, 3, 4), (3, 5, 3)])
def test_full_qpe_distribution_and_phase_sign(n, threshold, bits):
    actual = simulate_amplitude_estimation(n, threshold, bits)
    expected = analytic_amplitude_distribution(threshold / 2**n, bits)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    assert math.isclose(sum(actual), 1, abs_tol=1e-12)
    errors = np.array([abs(decode_amplitude(y, bits) - threshold / 2**n) for y in range(2**bits)])
    assert actual[errors <= uniform_amplitude_error_bound(bits) + 1e-12].sum() >= 8 / math.pi**2 - 1e-12


def test_controlled_global_phase_is_not_discarded():
    # a=1 has Q eigenvalue -1 and an exactly representable phase 1/2.
    # Losing Q's global minus sign would return a=0 instead.
    actual = simulate_amplitude_estimation(1, 2, 3)
    assert actual[4] > 1 - 1e-12
    assert decode_amplitude(4, 3) == 1


def test_exact_median_coverage_against_all_three_run_outcomes():
    from itertools import product

    amplitude, bits, repeats = .25, 3, 3
    law = analytic_amplitude_distribution(amplitude, bits)
    epsilon = uniform_amplitude_error_bound(bits)
    brute = 0.0
    for ys in product(range(2**bits), repeat=repeats):
        median = sorted(decode_amplitude(y, bits) for y in ys)[1]
        if abs(median - amplitude) > epsilon:
            brute += float(np.prod([law[y] for y in ys]))
    assert math.isclose(exact_median_interval_failure(amplitude, bits, repeats, law), brute, abs_tol=2e-13)


def test_inverse_preparation_and_controlled_access_are_actual_gates():
    circuit = build_amplitude_estimation(2, 1, 3, measure=True)
    assert circuit.num_qubits == 6
    assert circuit.count_ops().get("measure") == 3
    assert not any(item.operation.name in {"unitary", "initialize"} for item in circuit.data)


@pytest.mark.parametrize("target", [0.1, 0.01, 0.001, 1e-8])
def test_median_confidence_is_minimal_odd_count(target):
    count = repetitions_for_failure(target)
    assert count % 2 == 1
    assert median_failure_bound(count) <= target
    if count > 1:
        assert median_failure_bound(count - 2) > target


def test_workload_charges_all_access_and_compares_equal_accuracy():
    record = amplitude_workload(3, 5, 4)
    per_run, all_runs = record["per_run"], record["all_runs"]
    assert per_run["controlled_Q_calls"] == 15
    assert per_run["threshold_predicate_or_inverse_calls"] == 31
    assert per_run["controlled_global_minus_signs"] == 15
    assert all_runs["threshold_predicate_or_inverse_calls"] == 31 * record["repetitions"]
    reference = record["classical_same_preparation_comparison"]
    epsilon = record["absolute_amplitude_error_bound"]
    assert 2 * math.exp(-2 * reference["shots_required"] * epsilon**2) <= .01
    assert record["physical_failure_probability"] is None


@pytest.mark.parametrize("n,k", [(0, 0), (True, 1), (2, -1), (2, 5), (21, 1)])
def test_bad_problem_inputs(n, k):
    with pytest.raises(ValueError):
        threshold_preparation(n, k)


def test_simulation_and_generation_limits_are_separate():
    with pytest.raises(ValueError, match="simulation is capped"):
        simulate_amplitude_estimation(MAX_SIM_QUBITS, 1, 1)
    # A 13-wire generated source circuit is permitted but not simulated.
    circuit = build_amplitude_estimation(10, 1, 2)
    assert circuit.num_qubits == 13
    with pytest.raises(ValueError):
        build_amplitude_estimation(2, 1, 11)
    with pytest.raises(ValueError):
        median_failure_bound(2)


def test_generator_cannot_publish_a_stale_optional_resource_report(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    for name in ("physical-resources.json", "surface-code-scaling.json", "scaling.csv", "distribution.svg"):
        (tmp_path / name).write_text("older instance")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/generate_amplitude_estimation.py"),
                    "--data-qubits", "1", "--threshold", "1", "--phase-bits", "1",
                    "--no-sim", "--no-scaling", "--output-dir", str(tmp_path)],
                   check=True, capture_output=True, text=True)
    for name in ("physical-resources.json", "surface-code-scaling.json", "scaling.csv", "distribution.svg"):
        assert not (tmp_path / name).exists()
    result = json.loads((tmp_path / "logical-resources.json").read_text())
    assert result["parameters"]["exact_amplitude"] == .5
    assert result["simulation"]["performed"] is False
    assert result["physical_projection"]["performed"] is False
