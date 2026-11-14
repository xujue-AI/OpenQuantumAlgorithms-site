"""Independent Bayesian arithmetic, circular-summary and circuit checks.

This file is source-only until the coordinator authorizes the frozen bounded
pass. Seeded checks establish deterministic model behavior, not coverage for
future phases, continuous priors, synthesized circuits or noisy hardware.
"""
import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

import openquantum.bayesian_qpe as bq


def _entropy(probabilities):
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0)


def _batch_posterior(prior, grid, transcript):
    """A direct product of scalar likelihoods, not recursive Bayes updates."""
    values = []
    for probability, phase in zip(prior, grid):
        weight = float(probability)
        for power, offset, outcome in transcript:
            p0 = (1 + math.cos(2 * math.pi * power * float(phase) - offset)) / 2
            weight *= p0 if outcome == 0 else 1 - p0
        values.append(weight)
    total = math.fsum(values)
    assert total > 0
    return np.array([value / total for value in values])


def _information_gain(prior, grid, power, offset):
    """Independent scalar enumeration of both prospective outcomes."""
    zero = [(1 + math.cos(2 * math.pi * power * float(phase) - offset)) / 2
            for phase in grid]
    expected_entropy = 0.0
    for outcome in (0, 1):
        joint = [float(w) * (p if outcome == 0 else 1 - p)
                 for w, p in zip(prior, zero)]
        evidence = math.fsum(joint)
        if evidence:
            expected_entropy += evidence * _entropy([p / evidence for p in joint])
    return _entropy(prior) - expected_entropy


def _full_matrix(n, phase, power, offset):
    """Independent closed amplitudes for all computational input columns.

    q0 is the control. The data phase uses the actual stored binary64 angle;
    no gate-matrix product or inference routine is used by this reference.
    """
    dimension = 1 << n
    theta = (2 * math.pi * phase) / n
    expected = np.zeros((2 * dimension, 2 * dimension), complex)
    for data in range(dimension):
        prepared_data = data ^ (dimension - 1)
        angle = power * theta * prepared_data.bit_count() - offset
        phase_factor = complex(math.cos(angle), math.sin(angle))
        for control_in in (0, 1):
            for control_out in (0, 1):
                sign = -1 if control_in != control_out else 1
                expected[2 * prepared_data + control_out, 2 * data + control_in] = (
                    1 + sign * phase_factor
                ) / 2
    return expected


@pytest.mark.parametrize("size", [16, 32, 128, 512, 4096])
def test_regular_circular_grid(size):
    grid = bq.phase_grid(size)
    np.testing.assert_array_equal(grid, np.arange(size) / size)
    assert grid.shape == (size,)
    assert grid[0] == 0
    assert grid[-1] < 1


@pytest.mark.parametrize("size", [True, False, 0, 8, 17, 64.0, 8192, 1 << 25])
def test_grid_rejects_invalid_or_oversized_requests(size):
    with pytest.raises(ValueError):
        bq.phase_grid(size)


def test_normalization_is_stable_without_mutating_inputs():
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    original = raw.copy()
    normalized = bq.normalize_weights(raw)
    np.testing.assert_allclose(normalized, [0.1, 0.2, 0.3, 0.4], atol=1e-16, rtol=0)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(bq.normalize_weights([1e308, 1e308]), [0.5, 0.5])
    smallest = math.nextafter(0.0, 1.0)
    np.testing.assert_array_equal(bq.normalize_weights([smallest, smallest]), [0.5, 0.5])


@pytest.mark.parametrize("weights", [
    [], [0, 0], [-1, 2], [math.nan, 1], [math.inf, 1],
    [[1, 2], [3, 4]], [True, False], [1, True], ["1", "2"], [1 + 0j, 2 + 0j],
])
def test_weight_shape_and_numeric_guards(weights):
    with pytest.raises(ValueError):
        bq.normalize_weights(weights)


def test_one_observation_and_total_probability_law():
    grid = bq.phase_grid(64)
    prior = np.full(64, 1 / 64)
    zero = bq.update_posterior(prior, grid, 1, 0.0, 0)
    one = bq.update_posterior(prior, grid, 1, 0.0, 1)
    reference = np.array([(1 + math.cos(2 * math.pi * p)) / 64 for p in grid])
    np.testing.assert_allclose(zero, reference, atol=2e-16, rtol=0)
    np.testing.assert_allclose(0.5 * zero + 0.5 * one, prior, atol=2e-16, rtol=0)
    np.testing.assert_array_equal(prior, np.full(64, 1 / 64))


def test_recursive_posterior_matches_independent_batch_likelihood():
    grid = bq.phase_grid(64)
    prior = np.array([(i + 1) / 2080 for i in range(64)])
    transcript = [(1, 0.0, 0), (2, math.pi / 2, 1),
                  (4, math.pi, 0), (8, 3 * math.pi / 2, 1)]
    recursive = prior.copy()
    for power, offset, outcome in transcript:
        recursive = bq.update_posterior(recursive, grid, power, offset, outcome)
        assert np.all(recursive >= 0)
        assert math.fsum(recursive) == pytest.approx(1, abs=2e-15)
    np.testing.assert_allclose(recursive, _batch_posterior(prior, grid, transcript),
                               atol=3e-15, rtol=0)


def test_zero_evidence_is_explicit_and_certain_branch_is_unchanged():
    grid = bq.phase_grid(32)
    certain = np.zeros(32)
    certain[0] = 1
    with pytest.raises(ValueError):
        bq.update_posterior(certain, grid, 1, 0.0, 1)
    np.testing.assert_array_equal(bq.update_posterior(certain, grid, 1, 0.0, 0), certain)
    np.testing.assert_array_equal(certain, np.eye(1, 32, 0)[0])


@pytest.mark.parametrize("outcome", [True, -1, 2, 0.0, "0"])
def test_observation_type_and_value_guards(outcome):
    grid = bq.phase_grid(32)
    with pytest.raises(ValueError):
        bq.update_posterior(np.ones(32), grid, 1, 0.0, outcome)


@pytest.mark.parametrize("power,offset", [(True, 0), (0, 0), (-1, 0),
                                         (1, True), (1, math.nan), (1, math.inf),
                                         (1, -0.1), (1, 2 * math.pi)])
def test_likelihood_parameter_guards(power, offset):
    with pytest.raises(ValueError):
        bq.likelihood(bq.phase_grid(32), power, offset)


def test_malformed_grid_and_weight_lengths_are_rejected():
    grid = bq.phase_grid(32)
    with pytest.raises(ValueError):
        bq.update_posterior(np.ones(31), grid, 1, 0, 0)
    malformed = grid.copy()
    malformed[7] = math.nan
    with pytest.raises(ValueError):
        bq.likelihood(malformed, 1, 0)
    with pytest.raises(ValueError):
        bq.likelihood(grid.reshape(4, 8), 1, 0)


def test_policy_matches_independent_two_outcome_utility_and_preserves_inputs():
    grid = bq.phase_grid(64)
    prior = np.array([(i + 1) / 2080 for i in range(64)])
    original_grid, original_prior = grid.copy(), prior.copy()
    powers = (1, 2, 4)
    offsets = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
    scores = [(power, offset, _information_gain(prior, grid, power, offset))
              for power in powers for offset in offsets]
    expected = scores[0]
    for candidate in scores[1:]:
        if candidate[2] > expected[2] + 1e-12:
            expected = candidate
    selected = bq.choose_experiment(prior, grid, powers=powers, offsets=offsets)
    assert selected["power"] == expected[0]
    assert selected["offset"] == expected[1]
    assert selected["expected_information_gain"] == pytest.approx(expected[2], abs=4e-14)
    assert -4e-14 <= selected["expected_information_gain"] <= _entropy(prior) + 4e-14
    np.testing.assert_array_equal(grid, original_grid)
    np.testing.assert_array_equal(prior, original_prior)


def test_zero_evidence_candidate_branches_and_declared_tie_order():
    grid = bq.phase_grid(32)
    certain = np.zeros(32)
    certain[0] = 1
    chosen = bq.choose_experiment(certain, grid, powers=(2, 1), offsets=(0.0, math.pi / 2))
    assert chosen["power"] == 2
    assert chosen["offset"] == 0
    assert chosen["expected_information_gain"] == pytest.approx(0, abs=2e-15)
    # Finite grids need not have identical utilities across different powers.
    # Compare the actual discrete model, retaining its schedule-order tie rule.
    prior = np.full(32, 1 / 32)
    candidates = [(k, beta, _information_gain(prior, grid, k, beta))
                  for k in (1, 2, 4) for beta in (0.0, math.pi / 2)]
    expected = candidates[0]
    for candidate in candidates[1:]:
        if candidate[2] > expected[2] + 1e-12:
            expected = candidate
    uniform = bq.choose_experiment(prior, grid, powers=(1, 2, 4), offsets=(0.0, math.pi / 2))
    assert uniform["power"] == expected[0]
    assert uniform["offset"] == expected[1]
    assert uniform["expected_information_gain"] == pytest.approx(expected[2], abs=4e-14)


@pytest.mark.parametrize("powers,offsets", [
    ((), (0.0,)), ((1, 1), (0.0,)), ((3,), (0.0,)), ((True,), (0.0,)),
    ((1,), ()), ((1,), (True,)), ((1,), (math.nan,)),
    ((1,), (2 * math.pi,)), ((1,), tuple(i / 10 for i in range(9))),
    (tuple(1 << i for i in range(9)), (0.0,)),
])
def test_policy_candidate_guards(powers, offsets):
    with pytest.raises(ValueError):
        bq.choose_experiment(np.ones(512), bq.phase_grid(512), powers=powers, offsets=offsets)


def test_policy_requires_grid_resolution_for_candidate_aliases():
    with pytest.raises(ValueError):
        bq.choose_experiment(np.ones(16), bq.phase_grid(16), powers=(8,), offsets=(0.0,))
    selected = bq.choose_experiment(np.ones(16), bq.phase_grid(16), powers=(4,), offsets=(0.0,))
    assert selected["power"] == 4


def test_cell_arc_for_one_point_has_positive_grid_width():
    grid = bq.phase_grid(64)
    weights = np.zeros(64)
    weights[12] = 1
    summary = bq.posterior_summary(weights, grid)
    assert summary["map_phase"] == 12 / 64
    assert summary["mean_phase"] == pytest.approx(12 / 64, abs=2e-15)
    arc = summary["credible_arc"]
    assert arc["node_count"] == 1
    assert arc["width"] == 1 / 64
    assert arc["start"] == 11.5 / 64
    assert arc["end"] == 12.5 / 64
    assert arc["mass"] == 1
    assert not arc["wraps"]
    assert summary["certified"] is False


def test_shortest_cell_arc_crosses_phase_branch_cut():
    grid = bq.phase_grid(64)
    weights = np.zeros(64)
    weights[[63, 0, 1]] = 1 / 3
    summary = bq.posterior_summary(weights, grid)
    arc = summary["credible_arc"]
    assert arc["node_count"] == 3
    assert arc["width"] == 3 / 64
    assert arc["start"] == 62.5 / 64
    assert arc["end"] == 1.5 / 64
    assert arc["wraps"]
    assert arc["mass"] == pytest.approx(1, abs=2e-15)
    mean = summary["mean_phase"]
    assert mean is not None and 0 <= mean < 1
    assert abs((mean + 0.5) % 1 - 0.5) < 2e-15


def test_uniform_and_opposite_modes_do_not_supply_a_spurious_mean():
    grid = bq.phase_grid(64)
    uniform = bq.posterior_summary(np.ones(64), grid, mass=0.999)
    assert uniform["mean_phase"] is None
    assert uniform["resultant_length"] <= 1e-12
    assert uniform["credible_arc"]["node_count"] == 64
    assert uniform["credible_arc"]["width"] == 1
    assert uniform["credible_arc"]["full_circle"]
    opposite = np.zeros(64)
    opposite[[0, 32]] = 0.5
    summary = bq.posterior_summary(opposite, grid)
    assert summary["mean_phase"] is None
    assert summary["map_phase"] == 0
    assert summary["credible_arc"]["node_count"] == 33
    assert summary["credible_arc"]["width"] == 33 / 64
    assert summary["credible_arc"]["mass"] == 1
    assert summary["certified"] is False


def test_arc_minimality_against_exhaustive_cyclic_windows():
    grid = bq.phase_grid(32)
    weights = np.array([(i + 1) / 528 for i in range(32)])
    target = 0.6
    expected = None
    for count in range(1, 33):
        for start in range(32):
            mass = math.fsum(float(weights[(start + j) % 32]) for j in range(count))
            if mass >= target:
                expected = (start, count, mass)
                break
        if expected is not None:
            break
    assert expected is not None
    arc = bq.posterior_summary(weights, grid, mass=target)["credible_arc"]
    start, count, mass = expected
    assert arc["node_count"] == count
    assert arc["width"] == count / 32
    assert arc["start"] == ((start - 0.5) / 32) % 1
    assert arc["end"] == ((start + count - 0.5) / 32) % 1
    assert arc["mass"] == pytest.approx(mass, abs=2e-15)
    assert arc["target_mass"] == target


def test_near_one_credible_mass_uses_the_actual_normalized_total():
    grid = bq.phase_grid(32)
    weights = np.arange(1, 33, dtype=float)
    summary = bq.posterior_summary(weights, grid, mass=math.nextafter(1.0, 0.0))
    arc = summary["credible_arc"]
    assert arc["full_circle"]
    assert arc["node_count"] == 32
    assert arc["width"] == 1
    assert arc["mass"] == 1
    assert summary["certified"] is False


def test_oversized_weight_vector_is_rejected_before_array_conversion(monkeypatch):
    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("Oversized input reached numerical conversion")

    monkeypatch.setattr(bq.np, "asarray", unexpected_conversion)
    with pytest.raises(ValueError):
        bq.normalize_weights([1.0] * (bq.MAX_GRID_SIZE + 1))


@pytest.mark.parametrize("mass", [True, 0, -0.1, 1, math.nan, math.inf])
def test_credible_mass_guard(mass):
    with pytest.raises(ValueError):
        bq.posterior_summary(np.ones(32), bq.phase_grid(32), mass=mass)


@pytest.mark.parametrize("n,phase,power,offset", [
    (1, 0.0, 1, 0.0), (1, 0.3, 1, math.pi / 2),
    (1, 0.995, 32, 3 * math.pi / 2), (2, 0.17, 8, math.pi / 3),
    (3, 0.07, 4, 0.0), (3, 0.5, 2, math.pi),
])
def test_complete_operator_and_probability_from_independent_matrix(n, phase, power, offset):
    circuit = bq.build_experiment(n, phase, power, offset, measure=False, initialize=False)
    reference = _full_matrix(n, phase, power, offset)
    np.testing.assert_allclose(Operator(circuit).data, reference, atol=3e-11, rtol=0)
    state = Statevector.from_instruction(circuit)
    theta = (2 * math.pi * phase) / n
    expected = (1 + math.cos(power * n * theta - offset)) / 2
    assert float(state.probabilities(qargs=[0])[0]) == pytest.approx(expected, abs=2e-11)
    check = bq.check_experiment(n, phase, power, offset)
    assert check["performed"] is True
    assert check["certified"] is False


def test_literal_access_and_fresh_reset_readout_accounting():
    n, power, offset = 2, 8, math.pi / 3
    circuit = bq.build_experiment(n, 0.3, power, offset)
    counts = dict(circuit.count_ops())
    assert counts == {"p": 3 * n * power + 1, "cx": 2 * n * power,
                      "x": n, "h": 2, "reset": n + 1, "measure": 1}
    assert circuit.num_qubits == n + 1
    assert circuit.num_clbits == 1
    assert len(circuit.data) == 5 * n * power + 2 * n + 5
    assert len(circuit.data) == bq.source_operation_count(n, power, offset)
    angles = [float(item.operation.params[0]) for item in circuit.data if item.operation.name == "p"]
    half = ((2 * math.pi * 0.3) / n) / 2
    assert angles.count(half) == 2 * n * power
    assert angles.count(-half) == n * power
    assert angles.count(-offset) == 1


@pytest.mark.parametrize("initialize,measure", [(False, False), (True, False), (False, True), (True, True)])
def test_source_count_matches_explicit_preparation_flags(initialize, measure):
    circuit = bq.build_experiment(1, 0.3, 2, 0.0, initialize=initialize, measure=measure)
    expected = 5 * 2 + 2 * 1 + 5 - (0 if initialize else 2) - (0 if measure else 1)
    assert len(circuit.data) == expected
    assert bq.source_operation_count(1, 2, 0.0, initialize=initialize, measure=measure) == expected


@pytest.mark.parametrize("n,phase,power,offset", [
    (True, 0.3, 1, 0.0), (0, 0.3, 1, 0.0), (1 << 20, 0.3, 1, 0.0),
    (1, True, 1, 0.0), (1, -0.1, 1, 0.0), (1, 1, 1, 0.0),
    (1, math.nan, 1, 0.0), (1, 0.3, True, 0.0), (1, 0.3, 0, 0.0),
    (1, 0.3, 1 << 30, 0.0), (1, 0.3, 1, math.nan), (1, 0.3, 1, True),
])
def test_experiment_input_guards(n, phase, power, offset):
    with pytest.raises(ValueError):
        bq.build_experiment(n, phase, power, offset)


def test_large_generation_is_separate_from_dense_validation():
    circuit = bq.build_experiment(8, 0.3, 32, math.pi / 2)
    assert circuit.num_qubits == 9
    assert len(circuit.data) == 1301
    with pytest.raises(ValueError):
        bq.check_experiment(8, 0.3, 32, math.pi / 2)


def test_fixed_seed_run_batch_posterior_and_complete_ledger(monkeypatch):
    calls = []
    original = bq.choose_experiment

    def record_policy(weights, grid, *args, **kwargs):
        assert not {"phase", "true_phase", "eigenphase"}.intersection(kwargs)
        calls.append((np.array(weights, copy=True), np.array(grid, copy=True)))
        return original(weights, grid, *args, **kwargs)

    monkeypatch.setattr(bq, "choose_experiment", record_policy)
    run = bq.run_adaptive(0.3, target_qubits=1, steps=12, grid_size=128, seed=7)
    assert len(run["trace"]) == 12
    assert len(calls) == 13  # one bounded policy preflight plus 12 observations
    transcript = [(item["power"], item["offset"], item["outcome"]) for item in run["trace"]]
    assert all(power in (1, 2, 4, 8, 16, 32) and outcome in (0, 1)
               for power, _, outcome in transcript)
    prior = np.full(128, 1 / 128)
    np.testing.assert_allclose(run["final_posterior"],
                               _batch_posterior(prior, run["grid"], transcript), atol=3e-14, rtol=0)
    ledger = run["ledger"]
    work = sum(power for power, _, _ in transcript)
    assert ledger["complete_fresh_preparations"] == 12
    assert ledger["shot_weighted_controlled_U_calls"] == work
    assert ledger["maximum_controlled_U_power"] == max(power for power, _, _ in transcript)
    assert ledger["source_gate_counts_all_trials"] == {
        "p": 3 * work + 12, "cx": 2 * work, "h": 24,
        "x": 12, "reset": 24, "measure": 12,
    }
    assert ledger["source_operations_all_trials"] == 5 * work + 7 * 12
    assert ledger["logical_qubits"] == 2
    assert ledger["classical_posterior_updates"] == 12
    assert ledger["classical_policy_calls"] == 13
    assert ledger["classical_preflight_policy_calls"] == 1
    assert ledger["classical_candidate_likelihood_values"] == 13 * 128 * 24
    assert ledger["classical_observation_likelihood_values"] == 12 * 128
    assert ledger["physical_estimate_performed"] is False
    assert run["summary"]["certified"] is False
    again = bq.run_adaptive(0.3, target_qubits=1, steps=12, grid_size=128, seed=7)
    assert [(item["power"], item["offset"], item["outcome"]) for item in again["trace"]] == transcript
    np.testing.assert_array_equal(run["final_posterior"], again["final_posterior"])


@pytest.mark.parametrize("kwargs", [
    {"steps": True}, {"steps": 0}, {"steps": 1 << 20}, {"target_qubits": True},
    {"seed": True}, {"seed": -1}, {"grid_size": 16}, {"grid_size": 1 << 20},
])
def test_adaptive_run_caps(kwargs):
    settings = {"phase": 0.3, "target_qubits": 1, "steps": 12, "grid_size": 128, "seed": 7}
    settings.update(kwargs)
    with pytest.raises(ValueError):
        bq.run_adaptive(**settings)
