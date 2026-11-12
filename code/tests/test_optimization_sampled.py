"""Check finite-shot decisions independently of ideal objective audits.

The comparisons below reconstruct joint observables from recorded bitstrings
and account for every rejected proposal and independent held-out preparation.
"""

import math

import numpy as np
import pytest

from openquantum import optimization as opt


def _outcome_value(algorithm, group, bits):
    """Evaluate the stated classical cut or Pauli-group outcome directly."""
    bit_assignment = [int(bit) for bit in bits[::-1]]
    if algorithm == "qaoa":
        return sum(bit_assignment[a] != bit_assignment[b]
                   for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)])
    eigenvalues = [1 - 2 * bit for bit in bit_assignment]
    return -sum(eigenvalues) if group == "all_z" else 0.5 * math.prod(eigenvalues)


def _count_moments(algorithm, group):
    """Use integer counts and independent scores, preserving joint covariance."""
    shots = sum(group["counts"].values())
    mean = sum(count * _outcome_value(algorithm, group["group"], bits)
               for bits, count in group["counts"].items()) / shots
    variance = sum(count * (_outcome_value(algorithm, group["group"], bits) - mean) ** 2
                   for bits, count in group["counts"].items()) / (shots - 1)
    return shots, mean, variance


def _without_times(value):
    """Seeded outcomes and decisions are reproducible; measured times are not."""
    if isinstance(value, dict):
        return {key: _without_times(item) for key, item in value.items()
                if "seconds" not in key and "generated_at" not in key}
    if isinstance(value, list):
        return [_without_times(item) for item in value]
    return value


@pytest.fixture(scope="module")
def full_sampled_demo():
    return opt.run_sampled_demo()


@pytest.mark.parametrize("algorithm,parameters", [("qaoa", [math.pi / 4, math.pi / 8]), ("vqe", [-math.atan(0.25)])])
def test_sampled_objectives_reconstruct_joint_counts_and_standard_error(algorithm, parameters):
    objective = opt.sampled_qaoa_objective if algorithm == "qaoa" else opt.sampled_vqe_objective
    result = objective(parameters, shots=512, seed=19, evaluation_index=3)
    group_means, group_variances = [], []
    for group in result["groups"]:
        shots, mean, variance = _count_moments(algorithm, group)
        assert shots == group["shots"] == 512
        assert group["estimate"] == pytest.approx(mean, abs=1e-12)
        assert group["sample_variance"] == pytest.approx(variance, abs=1e-12)
        assert group["estimated_standard_error"] == pytest.approx(math.sqrt(variance / shots), abs=1e-12)
        assert group["circuit_gate_counts"]["measure"] == (4 if algorithm == "qaoa" else 2)
        assert group["measured_qubit_readouts"] == shots * group["logical_qubits"]
        group_means.append(mean)
        group_variances.append(variance / shots)
    expected = -sum(group_means) if algorithm == "qaoa" else sum(group_means)
    assert result["objective_estimate"] == pytest.approx(expected, abs=1e-12)
    assert result["estimated_standard_error"] == pytest.approx(math.sqrt(sum(group_variances)), abs=1e-12)
    assert result["total_shots"] == sum(group["shots"] for group in result["groups"])
    assert "exact_objective_audit" not in result
    assert all("exact_expectation" not in group for group in result["groups"])


def test_sampled_objectives_do_not_call_exact_expectation_oracles(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("exact expectation must not be used by sampled objectives")

    monkeypatch.setattr(opt, "qaoa_expectation", forbidden)
    monkeypatch.setattr(opt, "vqe_expectation", forbidden)
    opt.sampled_pattern_search(opt.sampled_qaoa_objective, [0.2, 0.1], seed=7,
                               schedule=((0.4, 16, 1),), initial_shots=16)
    opt.sampled_pattern_search(opt.sampled_vqe_objective, [0.6], seed=7,
                               schedule=((0.4, 16, 1),), initial_shots=16)


def test_optimizer_ignores_poisoned_exact_diagnostics():
    def make_objective(audit_sign):
        def objective(parameters, *, shots, seed, evaluation_index, family):
            # This callback represents identical noisy observations under two
            # contradictory exact diagnostics. Only observations may decide.
            sampled_value = -math.cos(parameters[0])
            return dict(objective_estimate=sampled_value, estimated_standard_error=0.03,
                        groups=[dict(shots=shots)], total_shots=shots,
                        evaluation_wall_seconds=0.0,
                        exact_objective_audit=audit_sign * parameters[0] * 1e9)
        return objective

    settings = dict(seed=7, schedule=((0.4, 16, 3),), initial_shots=16)
    first = opt.sampled_pattern_search(make_objective(1), [0.6], **settings)
    second = opt.sampled_pattern_search(make_objective(-1), [0.6], **settings)
    assert first["iterations"] == second["iterations"]
    assert first["parameters"] == second["parameters"]
    assert first["accepted_proposals"] == second["accepted_proposals"]


def test_best_sampled_proposal_is_selected_before_the_uncertainty_margin():
    observations = [(0.0, 0.1), (0.0, 0.1), (-1.0, 0.1), (-2.0, 2.0)]

    def objective(parameters, *, shots, seed, evaluation_index, family):
        value, se = observations[evaluation_index]
        return dict(objective_estimate=value, estimated_standard_error=se,
                    groups=[dict(shots=shots)], total_shots=shots,
                    evaluation_wall_seconds=0.0)

    result = opt.sampled_pattern_search(objective, [0.6], seed=7,
                                       schedule=((0.4, 16, 1),), initial_shots=16)
    step = result["iterations"][0]
    assert step["best_proposal_evaluation"] == 3
    assert step["sampled_improvement"] == 2.0
    assert step["comparison_margin"] == pytest.approx(2 * math.hypot(0.1, 2.0))
    assert not step["accepted"]
    assert result["parameters"] == pytest.approx([0.6])


@pytest.mark.parametrize("objective,initial", [(opt.sampled_qaoa_objective, [0.2, 0.1]), (opt.sampled_vqe_objective, [0.6])])
def test_seeded_training_paths_are_deterministic_excluding_timings(objective, initial):
    settings = dict(seed=41, schedule=((0.4, 16, 2), (0.2, 32, 2)), initial_shots=16)
    first = opt.sampled_pattern_search(objective, initial, **settings)
    second = opt.sampled_pattern_search(objective, initial, **settings)
    assert _without_times(first) == _without_times(second)


def test_default_schedule_and_every_training_decision_can_be_replayed(full_sampled_demo):
    assert list(opt.FINITE_SHOT_SCHEDULE) == [(0.4, 256, 8), (0.2, 512, 8), (0.1, 1024, 8), (0.05, 2048, 8)]
    for algorithm, dimensions in (("qaoa", 2), ("vqe", 1)):
        for run in full_sampled_demo[algorithm]["runs"]:
            evaluations = run["evaluations"]
            assert [entry["index"] for entry in evaluations] == list(range(len(evaluations)))
            assert evaluations[0]["role"] == "initial"
            incumbent_parameters = np.asarray(run["initial"])
            incumbent_parameters = (incumbent_parameters + math.pi) % (2 * math.pi) - math.pi
            for sweep_index, sweep in enumerate(run["iterations"]):
                block = sweep_index // 8
                step, shots = [(0.4, 256), (0.2, 512), (0.1, 1024), (0.05, 2048)][block]
                assert sweep["step"] == step
                assert sweep["shots_per_group"] == shots
                incumbent = evaluations[sweep["incumbent_evaluation"]]
                proposals = [evaluations[index] for index in sweep["proposal_evaluations"]]
                assert len(proposals) == 2 * dimensions
                assert incumbent["role"] == "incumbent"
                np.testing.assert_allclose(incumbent["parameters"], incumbent_parameters, atol=1e-12)
                np.testing.assert_allclose(sweep["parameters_before"], incumbent_parameters, atol=1e-12)
                for proposal_index, proposal in enumerate(proposals):
                    axis, direction = proposal_index // 2, -1 if proposal_index % 2 == 0 else 1
                    expected_parameters = incumbent_parameters.copy()
                    expected_parameters[axis] += direction * step
                    expected_parameters = (expected_parameters + math.pi) % (2 * math.pi) - math.pi
                    assert proposal["axis"] == axis
                    assert proposal["direction"] == direction
                    assert proposal["role"] == "proposal"
                    np.testing.assert_allclose(proposal["parameters"], expected_parameters, atol=1e-12)
                best = min(proposals, key=lambda entry: entry["objective_estimate"])
                improvement = incumbent["objective_estimate"] - best["objective_estimate"]
                margin = 2 * math.hypot(incumbent["estimated_standard_error"], best["estimated_standard_error"])
                accepted = improvement > margin
                assert sweep["best_proposal_evaluation"] == best["index"]
                assert sweep["sampled_improvement"] == pytest.approx(improvement, abs=1e-12)
                assert sweep["comparison_margin"] == pytest.approx(margin, abs=1e-12)
                assert sweep["accepted"] == accepted
                assert sum(proposal["accepted"] for proposal in proposals) == int(accepted)
                if accepted:
                    incumbent_parameters = np.asarray(best["parameters"])
                np.testing.assert_allclose(sweep["parameters_after"], incumbent_parameters, atol=1e-12)
                assert all(group["shots"] == shots for call in [incumbent] + proposals for group in call["groups"])
            np.testing.assert_allclose(run["parameters"], incumbent_parameters, atol=1e-12)
            assert run["iteration_count"] == 32
            assert run["stop_reason"] == "fixed_schedule_exhausted"
            assert run["convergence_certified"] is False
            assert run["rejected_proposals"] + run["accepted_proposals"] == 32 * 2 * dimensions


def test_all_rng_streams_are_distinct_across_algorithms_groups_and_holdout(full_sampled_demo):
    stream_keys = []
    for algorithm, algorithm_id in (("qaoa", 0), ("vqe", 1)):
        for run in full_sampled_demo[algorithm]["runs"]:
            for family, calls in ((0, run["evaluations"]), (1, [run["validation"]])):
                for call_index, call in enumerate(calls):
                    for group_index, group in enumerate(call["groups"]):
                        rng = group["rng"]
                        assert rng["entropy"] == [run["seed"], algorithm_id]
                        assert rng["spawn_key"] == [family, call_index, group_index]
                        assert rng["stream_key"] == [run["seed"], algorithm_id, family, call_index, group_index]
                        stream_keys.append(tuple(rng["stream_key"]))
    assert len(stream_keys) == len(set(stream_keys)) == 1074


def test_full_default_budget_counts_all_rejected_proposals_and_holdout(full_sampled_demo):
    expected = {"qaoa": (161, 153856, 4096, 157952), "vqe": (97, 184832, 8192, 193024)}
    for algorithm in ("qaoa", "vqe"):
        calls, training, validation, total = expected[algorithm]
        assert len(full_sampled_demo[algorithm]["runs"]) == 3
        for run in full_sampled_demo[algorithm]["runs"]:
            assert run["objective_evaluations"] == len(run["evaluations"]) == calls
            assert run["total_shots"] == sum(e["total_shots"] for e in run["evaluations"]) == training
            assert run["validation"]["total_shots"] == validation
            assert run["total_shots_including_validation"] == total
            assert run["evaluations"][0]["total_shots"] == (256 if algorithm == "qaoa" else 512)
    assert full_sampled_demo["training_shots"] == 1016064
    assert full_sampled_demo["validation_shots"] == 36864
    assert full_sampled_demo["total_shots"] == 1052928


def test_recorded_exact_audits_and_independent_holdout_have_separate_contracts(full_sampled_demo):
    assert full_sampled_demo["exact_diagnostics_used_for_optimization"] is False
    assert full_sampled_demo["physical_resource_estimate"] is None
    for algorithm in ("qaoa", "vqe"):
        for run in full_sampled_demo[algorithm]["runs"]:
            for evaluation in run["evaluations"] + [run["validation"]]:
                params = evaluation["parameters"]
                exact = -(2 + math.sin(4 * params[1]) * math.sin(2 * params[0])) if algorithm == "qaoa" else (
                    -2 * math.cos(params[0]) + 0.5 * math.sin(params[0]))
                assert evaluation["exact_objective_audit"] == pytest.approx(exact, abs=1e-12)
                means, variances = [], []
                for group in evaluation["groups"]:
                    shots, mean, variance = _count_moments(algorithm, group)
                    assert shots == group["shots"]
                    means.append(mean)
                    variances.append(variance / shots)
                observed = -sum(means) if algorithm == "qaoa" else sum(means)
                assert evaluation["objective_estimate"] == pytest.approx(observed, abs=1e-12)
                assert evaluation["estimated_standard_error"] == pytest.approx(math.sqrt(sum(variances)), abs=1e-12)
            validation = run["validation"]
            np.testing.assert_allclose(validation["parameters"], run["parameters"], atol=1e-12)
            assert validation["role"] == "independent_final_validation"
            assert all(group["shots"] == 4096 for group in validation["groups"])
            assert validation["estimation_error_from_exact"] == pytest.approx(
                validation["objective_estimate"] - validation["exact_objective_audit"], abs=1e-12)
            width_squared_sum = 16 if algorithm == "qaoa" else 17
            assert validation["hoeffding_95_absolute_mean_radius"] == pytest.approx(
                math.sqrt(width_squared_sum * math.log(40) / (2 * 4096)), abs=1e-12)
            if algorithm == "qaoa":
                assert run["target_met_audit"] == (run["exact_expected_cut_audit"] >= 2.90)
            else:
                assert run["ground_energy_gap_audit"] == pytest.approx(
                    run["exact_energy_audit"] + math.sqrt(4.25), abs=1e-12)
                assert run["target_met_audit"] == (run["ground_energy_gap_audit"] <= 0.03)


def test_full_resource_ledger_repeats_every_measured_group(full_sampled_demo):
    resources = opt.sampled_run_resources(full_sampled_demo)
    assert resources["total_shots"] == 1052928
    assert resources["physical_resource_estimate"] is None
    for algorithm, shots, batches, train_calls in (("qaoa", 473856, 486, 483), ("vqe", 579072, 588, 291)):
        ledger = resources["algorithms"][algorithm]
        assert ledger["total_modeled_circuit_shots"] == shots
        assert ledger["measured_group_batches"] == len(ledger["execution_inputs"]) == batches
        assert ledger["objective_calls"] == train_calls
        assert ledger["independent_validation_calls"] == 3
        assert sum(entry["shots"] for entry in ledger["execution_inputs"]) == shots
        assert ledger["total_modeled_qubit_readouts"] == shots * (4 if algorithm == "qaoa" else 2)
        for gate, total in ledger["shot_weighted_gate_counts"].items():
            expected = sum(entry["circuit_gate_counts"].get(gate, 0) * entry["shots"]
                           for entry in ledger["execution_inputs"])
            assert total == expected
    assert resources["algorithms"]["qaoa"]["shot_weighted_gate_counts"]["h"] == 4 * 473856
    assert resources["algorithms"]["qaoa"]["shot_weighted_gate_counts"]["cx"] == 8 * 473856
    assert resources["algorithms"]["vqe"]["shot_weighted_gate_counts"]["ry"] == 579072
    assert resources["algorithms"]["vqe"]["shot_weighted_gate_counts"]["h"] == 579072


@pytest.mark.parametrize("settings", [dict(shots=1), dict(shots=65537), dict(seed=True), dict(seed=-1), dict(evaluation_index=-1), dict(family=-1)])
def test_invalid_finite_sampling_requests_are_rejected(settings):
    request = dict(shots=16, seed=7, evaluation_index=0)
    request.update(settings)
    with pytest.raises(ValueError):
        opt.sampled_qaoa_objective([0.2, 0.1], **request)


@pytest.mark.parametrize("schedule", [(), ((0.4, 16, 129),), ((math.nan, 16, 1),), ((0.4, 1, 1),)])
def test_invalid_schedules_are_rejected_before_sampling(schedule):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid schedule must fail before an objective call")

    with pytest.raises(ValueError):
        opt.sampled_pattern_search(forbidden, [0.6], seed=7, schedule=schedule)
