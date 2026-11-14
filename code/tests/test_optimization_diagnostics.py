"""Independent checks for matched-budget initialization and sampling diagnostics.

The references below use integer bit operations, explicit amplitudes, and raw
joint scores. They do not obtain exact moments from the optimization module.
"""

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Statevector

from openquantum import optimization as opt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "assets/generated/optimization/diagnostics"
SEEDS = [7, 19, 41]
CASE_IDS = [
    "qaoa-cycle-4-p1", "qaoa-cycle-4-p2",
    "qaoa-cycle-6-p1", "qaoa-cycle-6-p2",
    "vqe-open-chain-4", "vqe-open-chain-6",
]


def _case(case_id):
    """The study inputs are frozen independently of the public recipe API."""
    if case_id.startswith("qaoa"):
        _, _, width, depth = case_id.split("-")
        n, p = int(width), int(depth[1:])
        return dict(case_id=case_id, algorithm="qaoa", n_qubits=n, p=p,
                    edges=[(i, i + 1) for i in range(n - 1)] + [(0, n - 1)])
    n = int(case_id.rsplit("-", 1)[1])
    return dict(case_id=case_id, algorithm="vqe", n_qubits=n, p=None)


def _canonical_case(case_id):
    return next(case for case in opt.diagnostic_cases() if case["case_id"] == case_id)


def _starts(case):
    if case["algorithm"] == "qaoa":
        return {"original": [0.2, 0.1] * case["p"],
                "alternate": [0.6, 0.4] * case["p"]}
    return {"original": [0.6] * case["n_qubits"],
            "alternate": [0.0] * case["n_qubits"]}


def _wrap(parameters):
    return (np.asarray(parameters, dtype=float) + math.pi) % (2 * math.pi) - math.pi


def _cut_values(case):
    return np.asarray([sum(((bits >> u) & 1) != ((bits >> v) & 1)
                           for u, v in case["edges"])
                       for bits in range(1 << case["n_qubits"])], dtype=float)


def _reference_state(case, parameters):
    """Use phase/bit-flip evolution or RY amplitudes and ordered CNOTs."""
    n = case["n_qubits"]
    if case["algorithm"] == "qaoa":
        state = np.full(1 << n, 2 ** (-n / 2), dtype=complex)
        cuts = _cut_values(case)
        indexes = np.arange(1 << n)
        for gamma, beta in zip(parameters[0::2], parameters[1::2], strict=True):
            state *= np.exp(-1j * gamma * cuts)
            for wire in range(n):
                state = (math.cos(beta) * state
                         - 1j * math.sin(beta) * state[indexes ^ (1 << wire)])
        return state
    state = np.zeros(1 << n, dtype=complex)
    for assignment in range(1 << n):
        amplitude = math.prod(math.sin(theta / 2) if (assignment >> wire) & 1
                              else math.cos(theta / 2)
                              for wire, theta in enumerate(parameters))
        output = assignment
        for wire in range(n - 1):
            output ^= ((output >> wire) & 1) << (wire + 1)
        state[output] = amplitude
    return state


def _walsh_transform(state):
    """All-wire H via Walsh parity signs in little-endian integer order."""
    size = len(state)
    return np.asarray([sum((-1) ** ((row & column).bit_count()) * amplitude
                           for column, amplitude in enumerate(state)) / math.sqrt(size)
                       for row in range(size)])


def _term_scores(case, group):
    n = case["n_qubits"]
    bits = range(1 << n)
    if case["algorithm"] == "qaoa":
        return np.asarray([[float(((value >> u) & 1) != ((value >> v) & 1))
                            for u, v in case["edges"]] for value in bits])
    signs = [[1 - 2 * ((value >> wire) & 1) for wire in range(n)] for value in bits]
    if group == "all_z":
        return -np.asarray(signs, dtype=float)
    return np.asarray([[0.5 * row[i] * row[i + 1] for i in range(n - 1)]
                       for row in signs])


def _population_moments(case, parameters):
    state = _reference_state(case, parameters)
    result = {}
    groups = ["all_z"] if case["algorithm"] == "qaoa" else ["all_z", "all_x"]
    for group in groups:
        measured_state = state if group == "all_z" else _walsh_transform(state)
        probabilities = np.abs(measured_state) ** 2
        terms = _term_scores(case, group)
        scores = terms.sum(axis=1)
        mean = float(probabilities @ scores)
        second = float(probabilities @ (scores ** 2))
        variance = float(probabilities @ ((scores - mean) ** 2))
        term_means = probabilities @ terms
        centered = terms - term_means
        covariance = (centered.T * probabilities) @ centered
        result[group] = dict(mean=mean, second_moment=second,
                             population_variance=variance,
                             covariance=covariance, probabilities=probabilities)
    return result


def _spin_hamiltonian(n):
    """Diagonal Z scores and adjacent pair-flip matrix entries."""
    matrix = np.zeros((1 << n, 1 << n))
    for source in range(1 << n):
        matrix[source, source] = -n + 2 * source.bit_count()
        for wire in range(n - 1):
            matrix[source ^ (3 << wire), source] += 0.5
    return matrix


def _count_statistics(case, record):
    means, errors = [], []
    for group in record["groups"]:
        shots = sum(group["counts"].values())
        assert shots == group["shots"]
        values = _term_scores(case, group["group"]).sum(axis=1)
        mean = sum(number * values[int(bits, 2)]
                   for bits, number in group["counts"].items()) / shots
        variance = sum(number * (values[int(bits, 2)] - mean) ** 2
                       for bits, number in group["counts"].items()) / (shots - 1)
        assert group["estimate"] == pytest.approx(mean, abs=1e-12)
        assert group["sample_variance"] == pytest.approx(variance, abs=1e-12)
        assert group["estimated_standard_error"] == pytest.approx(math.sqrt(variance / shots), abs=1e-12)
        means.append(mean)
        errors.append(variance / shots)
    objective = -means[0] if case["algorithm"] == "qaoa" else sum(means)
    standard_error = math.sqrt(sum(errors))
    assert record["objective_estimate"] == pytest.approx(objective, abs=1e-12)
    assert record["estimated_standard_error"] == pytest.approx(standard_error, abs=1e-12)
    assert record["total_shots"] == sum(group["shots"] for group in record["groups"])
    return objective, standard_error


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("start_id", ["original", "alternate", "asymmetric"])
def test_joint_populations_match_the_explicit_group_circuits(case_id, start_id):
    case = _case(case_id)
    starts = _starts(case)
    dimensions = 2 * case["p"] if case["algorithm"] == "qaoa" else case["n_qubits"]
    parameters = (starts[start_id] if start_id in starts
                  else [(-1) ** index * (0.17 + 0.09 * index) for index in range(dimensions)])
    for group, reference in _population_moments(case, parameters).items():
        circuit = opt.quality_scaling_group_circuit(case, parameters, group)
        actual = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
        np.testing.assert_allclose(actual, reference["probabilities"], atol=1e-12)
        assert reference["population_variance"] == pytest.approx(reference["covariance"].sum(), abs=1e-12)
        assert reference["second_moment"] == pytest.approx(reference["population_variance"] + reference["mean"] ** 2, abs=1e-12)


def test_spin_joint_variance_retains_nonzero_term_covariance():
    case = _case("vqe-open-chain-4")
    moments = _population_moments(case, [0.6] * 4)
    # Both grouped scores have correlated terms. Dropping these cross terms
    # would change an expected-margin control's standard error.
    for group in ("all_z", "all_x"):
        covariance = moments[group]["covariance"]
        independent_term_variance = float(np.trace(covariance))
        assert abs(moments[group]["population_variance"] - independent_term_variance) > 0.01


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("parameters", [[0.2, 0.1], [0.6, 0.4],
                                       [-0.37, 0.23], [math.pi / 4, math.pi / 8]])
def test_p1_cycle_ceiling_is_an_analytic_constraint_on_the_quality_target(n, parameters):
    case = _case(f"qaoa-cycle-{n}-p1")
    cut_mean = _population_moments(case, parameters)["all_z"]["mean"]
    gamma, beta = parameters
    formula = n / 2 + n / 4 * math.sin(4 * beta) * math.sin(2 * gamma)
    assert cut_mean == pytest.approx(formula, abs=1e-12)
    assert cut_mean / n <= 0.75 + 1e-12
    assert 0.75 < 0.80


@pytest.mark.parametrize("case_id", CASE_IDS)
@pytest.mark.parametrize("start_id", ["original", "alternate", "asymmetric"])
def test_exact_group_moments_use_true_joint_population_variance(case_id, start_id):
    case = _case(case_id)
    dimensions = 2 * case["p"] if case["algorithm"] == "qaoa" else case["n_qubits"]
    starts = _starts(case)
    parameters = (starts[start_id] if start_id in starts
                  else [(-1) ** index * (0.17 + 0.09 * index) for index in range(dimensions)])
    expected = _population_moments(case, parameters)
    result = opt.diagnostic_exact_moments(_canonical_case(case_id), parameters, shots=113)
    means, variance_of_means = [], []
    assert [group["group"] for group in result["groups"]] == list(expected)
    for group in result["groups"]:
        reference = expected[group["group"]]
        assert group["mean"] == pytest.approx(reference["mean"], abs=1e-11)
        assert group["population_variance"] == pytest.approx(reference["population_variance"], abs=1e-11)
        assert group["variance_of_mean"] == pytest.approx(reference["population_variance"] / 113, abs=1e-12)
        assert group["virtual_sampling_shots"] == 113
        assert "counts" not in group and "rng" not in group
        means.append(group["mean"])
        variance_of_means.append(group["variance_of_mean"])
    objective = -means[0] if case["algorithm"] == "qaoa" else sum(means)
    assert result["objective_mean"] == pytest.approx(objective, abs=1e-12)
    assert result["expected_standard_error"] == pytest.approx(math.sqrt(sum(variance_of_means)), abs=1e-12)


@pytest.mark.parametrize("n", [4, 6])
def test_zero_vqe_has_nonzero_x_population_variance_even_without_objective_noise(n):
    result = opt.diagnostic_exact_moments(_canonical_case(f"vqe-open-chain-{n}"), [0.0] * n, shots=64)
    z_group, x_group = result["groups"]
    assert z_group["mean"] == pytest.approx(-n, abs=1e-12)
    assert z_group["population_variance"] == pytest.approx(0, abs=1e-12)
    assert x_group["mean"] == pytest.approx(0, abs=1e-12)
    assert x_group["population_variance"] == pytest.approx((n - 1) / 4, abs=1e-12)
    assert result["expected_standard_error"] == pytest.approx(math.sqrt((n - 1) / 256), abs=1e-12)


def test_ghz_group_variance_exposes_the_z_covariance_terms():
    result = opt.diagnostic_exact_moments(_canonical_case("vqe-open-chain-4"),
                                          [math.pi / 2, 0.0, 0.0, 0.0], shots=97)
    z_group, x_group = result["groups"]
    assert z_group["mean"] == pytest.approx(0, abs=1e-12)
    assert z_group["population_variance"] == pytest.approx(16, abs=1e-12)
    assert x_group["mean"] == pytest.approx(0, abs=1e-12)
    assert x_group["population_variance"] == pytest.approx(1.25, abs=1e-12)


def _assert_polling_trace(run, statistics, *, virtual=False):
    records = run["evaluations"]
    dimensions = len(run["initial"])
    point = _wrap(run["initial"])
    np.testing.assert_allclose(records[0]["parameters"], point, atol=1e-14)
    assert records[0]["role"] == "initial"
    assert len(records) == 1 + len(run["iterations"]) * (1 + 2 * dimensions)
    accepted_indexes = []
    for sweep, iteration in enumerate(run["iterations"]):
        incumbent = 1 + sweep * (1 + 2 * dimensions)
        proposals = list(range(incumbent + 1, incumbent + 1 + 2 * dimensions))
        assert iteration["incumbent_evaluation"] == incumbent
        assert iteration["proposal_evaluations"] == proposals
        assert records[incumbent]["role"] == "incumbent"
        np.testing.assert_allclose(records[incumbent]["parameters"], point, atol=1e-14)
        np.testing.assert_allclose(iteration["parameters_before"], point, atol=1e-14)
        for offset, index in enumerate(proposals):
            axis, direction = offset // 2, -1 if offset % 2 == 0 else 1
            proposed = point.copy()
            proposed[axis] += direction * iteration["step"]
            assert records[index]["axis"] == axis
            assert records[index]["direction"] == direction
            assert records[index]["role"] == "proposal"
            np.testing.assert_allclose(records[index]["parameters"], _wrap(proposed), atol=1e-14)
        best = min(proposals, key=lambda index: statistics[index][0])
        improvement = statistics[incumbent][0] - statistics[best][0]
        margin = 2 * math.hypot(statistics[incumbent][1], statistics[best][1])
        accepted = bool(improvement > margin)
        assert iteration["best_proposal_evaluation"] == best
        assert iteration["exact_improvement" if virtual else "sampled_improvement"] == pytest.approx(improvement, abs=1e-12)
        assert iteration["comparison_margin"] == pytest.approx(margin, abs=1e-12)
        assert iteration["accepted"] is accepted
        assert sum(bool(records[index]["accepted"]) for index in proposals) == int(accepted)
        if accepted:
            point = np.asarray(records[best]["parameters"])
            accepted_indexes.append(best)
        np.testing.assert_allclose(iteration["parameters_after"], point, atol=1e-14)
    np.testing.assert_allclose(run["parameters"], point, atol=1e-14)
    assert [record["index"] for record in records if record["accepted"]] == accepted_indexes
    assert run["accepted_proposals"] == len(accepted_indexes)
    assert run["rejected_proposals"] == 2 * dimensions * len(run["iterations"]) - len(accepted_indexes)
    assert run["stop_reason"] == "fixed_schedule_exhausted"
    assert run["convergence_certified"] is False


def test_mocked_sampled_polling_replays_counts_and_ignores_poisoned_exact_audits():
    case = _case("qaoa-cycle-4-p1")

    def make_objective(poison_sign):
        def objective(parameters, *, shots, seed, evaluation_index, family):
            # Each observation is an explicit mock count table. No circuit,
            # randomness, diagnostic run, or new study is executed here.
            high = min(shots - 1, max(1, round(shots * (0.5 + 0.3 * math.sin(parameters[0])))))
            counts = {"0000": shots - high, "0101": high}
            mean = 4 * high / shots
            variance = ((shots - high) * mean ** 2 + high * (4 - mean) ** 2) / (shots - 1)
            error = math.sqrt(variance / shots)
            return dict(parameters=list(parameters), objective_estimate=-mean,
                        estimated_standard_error=error, groups=[dict(group="all_z", shots=shots,
                        counts=counts, estimate=mean, sample_variance=variance,
                        estimated_standard_error=error)], total_shots=shots,
                        evaluation_wall_seconds=0.0,
                        exact_objective_audit=poison_sign * parameters[0] * 1e12)
        return objective

    settings = dict(seed=7, schedule=((0.4, 16, 2), (0.2, 32, 1)), initial_shots=16)
    first = opt.sampled_pattern_search(make_objective(1), [math.pi - 0.1, 0.1], **settings)
    second = opt.sampled_pattern_search(make_objective(-1), [math.pi - 0.1, 0.1], **settings)
    statistics = [_count_statistics(case, record) for record in first["evaluations"]]
    _assert_polling_trace(first, statistics)
    assert first["iterations"] == second["iterations"]
    assert first["parameters"] == second["parameters"]


def test_mocked_polling_rejects_an_improvement_equal_to_its_margin():
    observations = [(0.0, 0.5), (0.0, 0.5), (-1.0, 0.0), (-0.5, 0.0)]

    def objective(parameters, *, shots, seed, evaluation_index, family):
        mean, error = observations[evaluation_index]
        return dict(objective_estimate=mean, estimated_standard_error=error,
                    groups=[dict(shots=shots)], total_shots=shots,
                    evaluation_wall_seconds=0.0, exact_objective_audit=-1e100)

    run = opt.sampled_pattern_search(objective, [0.0], seed=7,
                                    schedule=((0.4, 16, 1),), initial_shots=16)
    assert run["iterations"][0]["sampled_improvement"] == 1.0
    assert run["iterations"][0]["comparison_margin"] == 1.0
    assert run["iterations"][0]["accepted"] is False
    assert run["parameters"] == [0.0]


def test_frozen_recipe_starts_streams_and_exact_budget_identities():
    recipe = opt.diagnostic_recipe()
    assert recipe["fixed_predeclared_recipe"] is True
    assert recipe["fixed_published_recipe"] is False
    assert recipe["cases"] == opt.diagnostic_cases()
    assert [case["case_id"] for case in recipe["cases"]] == CASE_IDS
    assert recipe["seeds"] == SEEDS
    assert recipe["control_modes"] == ["exact-objective", "expected-margin"]
    assert recipe["comparison_z"] == 2.0
    assert recipe["validation_shots_per_group"] == 16_384
    assert recipe["validation_failure_total"] == 0.05
    assert recipe["validation_failure_per_group"] == pytest.approx(0.05 / 48)
    calls = training_groups = heldout_groups = training_shots = heldout_shots = control_calls = 0
    weighted, readouts = {}, 0
    streams = []
    for index, case in enumerate(recipe["cases"]):
        independent = _case(case["case_id"])
        for key in ("algorithm", "n_qubits", "p"):
            assert case[key] == independent[key]
        groups = ["all_z"] if case["algorithm"] == "qaoa" else ["all_z", "all_x"]
        assert case["groups"] == groups
        assert case["edges"] == ([list(edge) for edge in independent["edges"]]
                                 if case["algorithm"] == "qaoa" else [])
        d = 2 * case["p"] if case["algorithm"] == "qaoa" else case["n_qubits"]
        g = len(groups)
        base = 468 if d == 2 else 260 if case["algorithm"] == "qaoa" else 130 if d == 4 else 90
        initial = 128 if g == 1 else 64
        assert case["parameter_count"] == d
        assert case["initial_shots_per_group"] == initial
        assert case["base_shots_per_group"] == base
        assert case["schedule"] == [[0.4, base, 4], [0.2, 2 * base, 4], [0.1, 4 * base, 4]]
        count = 1 + 12 * (1 + 2 * d)
        assert g * (initial + (1 + 2 * d) * 28 * base) == 65_648
        assert case["planned"]["training_calls_per_seed"] == count
        assert case["planned"]["training_shots_per_seed"] == 65_648
        assert case["planned"]["validation_shots_per_seed"] == g * 16_384
        assert case["planned"]["total_shots_per_seed"] == 65_648 + g * 16_384
        expected_starts = _starts(independent)
        assert [start["start_id"] for start in case["starts"]] == list(expected_starts)
        for start_index, (start_id, parameters) in enumerate(expected_starts.items()):
            start = opt.diagnostic_start(case, start_id)
            assert start["parameters"] == parameters
            assert start["stream_id"] == 30 + 2 * index + start_index
            streams.append(start["stream_id"])
        calls += 6 * count
        training_groups += 6 * g * count
        heldout_groups += 6 * g
        training_shots += 6 * 65_648
        heldout_shots += 6 * g * 16_384
        control_calls += 4 * count
        n, p = case["n_qubits"], case["p"]
        for group in groups:
            per_group_preparations = 6 * (65_648 // g + 16_384)
            if case["algorithm"] == "qaoa":
                per = dict(h=n, rx=n * p, rz=n * p, cx=2 * n * p, measure=n)
            else:
                per = dict(ry=n, cx=n - 1, measure=n)
                if group == "all_x":
                    per["h"] = n
            for gate, number in per.items():
                weighted[gate] = weighted.get(gate, 0) + number * per_group_preparations
            readouts += n * per_group_preparations
    assert streams == list(range(30, 42))
    expected = dict(sampled_runs=36, cpu_controls=24, training_calls=3636,
                    validation_calls=36, objective_calls=3672, training_groups=5232,
                    validation_groups=48, group_batches=5280, training_shots=2_363_328,
                    validation_shots=786_432, total_shots=3_149_760,
                    measured_qubit_readouts=15_748_800,
                    shot_weighted_source_unitary_gates=82_488_288,
                    cpu_control_evaluations_per_mode=1212, cpu_control_evaluations=2424,
                    cpu_control_final_audits=24)
    assert recipe["planned"] == expected
    assert calls == expected["training_calls"]
    assert training_groups == expected["training_groups"]
    assert heldout_groups == expected["validation_groups"]
    assert training_shots == expected["training_shots"]
    assert heldout_shots == expected["validation_shots"]
    assert control_calls == expected["cpu_control_evaluations"]
    assert recipe["expected_weighted_gate_counts"] == weighted
    assert readouts == expected["measured_qubit_readouts"]
    assert sum(number for gate, number in weighted.items() if gate != "measure") == expected["shot_weighted_source_unitary_gates"]
    assert recipe["bounds"] == dict(simulation_qubits=8, dense_qubits=8, actual_max_qubits=6,
                                    source_unitary_gates_per_circuit=100_000, group_batches=10_000,
                                    sampled_shots=10_000_000, numerical_wall_seconds=120)


def test_exact_moment_calculation_cannot_sample_or_construct_source_circuits(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("exact population moments cannot use sampling or circuit-state oracles")

    monkeypatch.setattr(opt, "_sample_measured_scalar", forbidden)
    monkeypatch.setattr(opt, "build_qaoa_circuit", forbidden)
    monkeypatch.setattr(opt, "build_spin_vqe_circuit", forbidden)
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    for case in opt.diagnostic_cases():
        result = opt.diagnostic_exact_moments(case, _starts(case)["original"], shots=64)
        assert result["quantum_preparations"] == 0
        assert result["count_samples"] is None
        assert result["seed"] is None


@pytest.mark.parametrize("mutation", ["start", "schedule", "target", "width", "groups", "stream"])
def test_changed_scientific_case_is_rejected_before_any_run_or_simulator(monkeypatch, mutation):
    case = deepcopy(opt.diagnostic_cases()[0])
    if mutation == "start":
        case["starts"][0]["parameters"] = [math.pi / 4, math.pi / 8]
    elif mutation == "schedule":
        case["schedule"][0][1] += 1
    elif mutation == "target":
        case["target"]["ratio_at_least"] = 0.75
    elif mutation == "width":
        case["n_qubits"] = 32
    elif mutation == "groups":
        case["groups"].append("all_x")
    else:
        case["starts"][0]["stream_id"] += 1

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight rejection must precede sampling or circuit allocation")

    monkeypatch.setattr(opt, "sampled_pattern_search", forbidden)
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    with pytest.raises(ValueError):
        opt.diagnostic_start(case, "original")
    with pytest.raises(ValueError):
        opt.diagnostic_group_circuit(case, [0.2, 0.1], "all_z")
    with pytest.raises(ValueError):
        opt.diagnostic_exact_moments(case, [0.2, 0.1], 64)
    with pytest.raises(ValueError):
        opt.run_diagnostic_sampled(case, "original", 7)


@pytest.mark.parametrize("seed", [True, -1, 0, 20, 7.0])
def test_nonfrozen_seed_rejected_before_sampled_search(monkeypatch, seed):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid seed must be rejected before search")
    monkeypatch.setattr(opt, "sampled_pattern_search", forbidden)
    with pytest.raises(ValueError):
        opt.run_diagnostic_sampled(opt.diagnostic_cases()[0], "original", seed)


@pytest.mark.parametrize("shots", [True, 0, 1, 65_537, 2.5])
def test_invalid_virtual_allocation_is_rejected(shots):
    with pytest.raises(ValueError):
        opt.diagnostic_exact_moments(opt.diagnostic_cases()[0], [0.2, 0.1], shots)


@pytest.fixture(scope="module")
def saved_study():
    if not (ARTIFACTS / "runs.json").is_file():
        pytest.skip("the fixed diagnostic study has not been executed and exported yet")
    return {name: json.loads((ARTIFACTS / filename).read_text())
            for name, filename in (("study", "runs.json"), ("raw", "raw-study.json"),
                                   ("resources", "resources.json"), ("manifest", "circuit-manifest.json"),
                                   ("freeze", "freeze.json"), ("recipe", "recipe.json"))}


def test_all_saved_sampled_moves_replay_from_counts_with_every_start_retained(saved_study):
    study = saved_study["study"]
    assert [case["case_id"] for case in study["cases"]] == CASE_IDS
    assert study["recipe"] == opt.diagnostic_recipe()
    calls = groups = training = validation = 0
    streams = set()
    for case in study["cases"]:
        expected_keys = [(start, seed) for start in ("original", "alternate") for seed in SEEDS]
        assert [(run["start_id"], run["seed"]) for run in case["runs"]] == expected_keys
        for run in case["runs"]:
            start = opt.diagnostic_start(case, run["start_id"])
            assert run["initial"] == _starts(_case(case["case_id"]))[run["start_id"]]
            assert run["stream_id"] == start["stream_id"]
            assert run["run_id"] == f"{case['case_id']}-{run['start_id']}-seed-{run['seed']}"
            statistics = [_count_statistics(case, record) for record in run["evaluations"]]
            _assert_polling_trace(run, statistics)
            assert len(run["iterations"]) == 12
            allocations = [case["initial_shots_per_group"]]
            for step, shots, sweeps in case["schedule"]:
                allocations.extend([shots] * (sweeps * (1 + 2 * case["parameter_count"])))
            assert run["objective_evaluations"] == len(run["evaluations"]) == len(allocations)
            assert run["total_shots"] == 65_648
            assert run["validation"]["parameters"] == run["parameters"]
            assert run["exact_diagnostics_used_for_optimization"] is False
            final = run["evaluations"][-(1 + 2 * case["parameter_count"])]
            best = run["evaluations"][run["iterations"][-1]["best_proposal_evaluation"]]
            selected = best if run["iterations"][-1]["accepted"] else final
            assert run["final_training_estimate"] == selected["objective_estimate"]
            assert run["final_training_standard_error"] == selected["estimated_standard_error"]
            for phase, records in ((0, run["evaluations"]), (1, [run["validation"]])):
                for index, record in enumerate(records):
                    _count_statistics(case, record)
                    calls += 1
                    assert "exact_objective_audit" not in record
                    for group_index, group in enumerate(record["groups"]):
                        allocation = allocations[index] if phase == 0 else 16_384
                        assert group["shots"] == allocation
                        key = (run["seed"], start["stream_id"], phase, index, group_index)
                        assert group["rng"]["stream_key"] == list(key)
                        assert group["rng"]["entropy"] == list(key[:2])
                        assert group["rng"]["spawn_key"] == list(key[2:])
                        assert group["rng"]["bit_generator"] == "PCG64"
                        assert key not in streams
                        streams.add(key)
                        groups += 1
                        if phase == 0:
                            training += allocation
                        else:
                            validation += allocation
    assert calls == 3672
    assert groups == len(streams) == 5280
    assert training == study["training_shots"] == 2_363_328
    assert validation == study["validation_shots"] == 786_432
    assert training + validation == study["total_shots"] == 3_149_760


def test_saved_controls_replay_independent_means_variances_and_zero_preparation_cost(saved_study):
    study = saved_study["study"]
    expected_keys = [(case_id, start, mode) for case_id in CASE_IDS
                     for start in ("original", "alternate")
                     for mode in ("exact-objective", "expected-margin")]
    assert [(run["case_id"], run["start_id"], run["mode"]) for run in study["controls"]] == expected_keys
    population_cache, calls = {}, 0
    by_case = {case["case_id"]: case for case in study["cases"]}
    for run in study["controls"]:
        case = by_case[run["case_id"]]
        assert run["initial"] == _starts(_case(case["case_id"]))[run["start_id"]]
        assert run["seed"] is None
        assert run["kind"] == "cpu-control"
        assert run["quantum_preparations"] == 0
        assert run["count_samples"] is None
        assert run["virtual_training_preparations"] == 65_648
        assert run["final_audit_count"] == 1
        statistics, virtual_preparations = [], 0
        for record in run["evaluations"]:
            key = (case["case_id"], tuple(record["parameters"]))
            if key not in population_cache:
                population_cache[key] = _population_moments(case, record["parameters"])
            reference = population_cache[key]
            allocation = record["virtual_shots_per_group"]
            means, variance_of_means = [], []
            assert record["quantum_preparations"] == 0
            assert record["count_samples"] is None
            assert "rng" not in record and "total_shots" not in record
            for group in record["groups"]:
                expected = reference[group["group"]]
                assert group["mean"] == pytest.approx(expected["mean"], abs=1e-11)
                assert group["population_variance"] == pytest.approx(expected["population_variance"], abs=1e-11)
                assert group["variance_of_mean"] == pytest.approx(expected["population_variance"] / allocation, abs=1e-12)
                assert group["virtual_sampling_shots"] == allocation
                assert group["quantum_preparations"] == 0
                assert group["count_samples"] is None
                assert "counts" not in group and "rng" not in group
                means.append(expected["mean"])
                variance_of_means.append(expected["population_variance"] / allocation)
            mean = -means[0] if case["algorithm"] == "qaoa" else sum(means)
            expected_se = math.sqrt(sum(variance_of_means))
            effective_se = 0.0 if run["mode"] == "exact-objective" else expected_se
            assert record["objective_mean"] == pytest.approx(mean, abs=1e-11)
            assert record["expected_standard_error"] == pytest.approx(expected_se, abs=1e-12)
            assert record["effective_standard_error"] == pytest.approx(effective_se, abs=1e-12)
            # Replay the recorded floating-point oracle after independently
            # checking it. Exact ties can differ in the last rounding bit
            # between algebraically equivalent bounded reference methods.
            statistics.append((record["objective_mean"], record["effective_standard_error"]))
            virtual_preparations += len(record["groups"]) * allocation
        _assert_polling_trace(run, statistics, virtual=True)
        assert virtual_preparations == 65_648
        assert run["objective_evaluations"] == len(run["evaluations"]) == case["planned"]["training_calls_per_seed"]
        selected = run["evaluations"][run["final_evaluation"]]
        assert run["final_exact_objective"] == selected["objective_mean"]
        assert run["final_effective_standard_error"] == selected["effective_standard_error"]
        assert run["final_expected_standard_error"] == selected["expected_standard_error"]
        calls += len(run["evaluations"])
    assert calls == study["cpu_control_objective_evaluations"] == 2424
    assert len(study["controls"]) == study["cpu_control_final_audits"] == 24


def _assert_final_quality(case, run, ground_energies):
    reference = run["diagnostic"]
    state = _reference_state(case, run["parameters"])
    if case["algorithm"] == "qaoa":
        cuts = _cut_values(case)
        probabilities = np.abs(state) ** 2
        mean = float(probabilities @ cuts)
        quality = mean / cuts.max()
        assert reference["cut_values"] == cuts.tolist()
        assert reference["classical_optimum"] == int(cuts.max())
        assert reference["exact_expected_cut"] == pytest.approx(mean, abs=1e-11)
        assert reference["exact_approximation_ratio"] == pytest.approx(quality, abs=1e-11)
        assert reference["exact_optimal_cut_probability"] == pytest.approx(probabilities[cuts == cuts.max()].sum(), abs=1e-11)
        met = quality >= 0.80
        if case["p"] == 1:
            assert quality <= 0.75 + 1e-12
            assert run["target_met_audit"] is False
    else:
        n = case["n_qubits"]
        matrix = _spin_hamiltonian(n)
        if n not in ground_energies:
            ground_energies[n] = float(np.linalg.eigvalsh(matrix)[0])
        ground = ground_energies[n]
        energy = float(np.vdot(state, matrix @ state).real)
        quality = (energy - ground) / n
        assert reference["ground_energy"] == pytest.approx(ground, abs=1e-11)
        assert reference["exact_energy"] == pytest.approx(energy, abs=1e-11)
        assert reference["energy_density_gap"] == pytest.approx(quality, abs=1e-11)
        assert reference["zero_state_energy"] == -n
        assert reference["zero_state_density_gap"] == pytest.approx((-n - ground) / n, abs=1e-11)
        met = quality <= 0.05
    assert run["target_met_audit"] is bool(met)
    assert reference["audit_performed_after_training"] is True
    assert reference["source_state_max_phase_aligned_error"] <= 1e-11
    return float(quality)


def test_every_saved_quality_flag_and_simultaneous_holdout_is_independent(saved_study):
    study = saved_study["study"]
    ground_energies = {}
    summary = {row["run_id"]: row for row in study["summary_rows"]}
    assert len(summary) == 36
    heldout_groups = 0
    for case in study["cases"]:
        for run in case["runs"]:
            quality = _assert_final_quality(case, run, ground_energies)
            validation = run["validation"]
            objective, error = _count_statistics(case, validation)
            denominator = case["n_qubits"]
            value = (-objective / denominator if case["algorithm"] == "qaoa"
                     else (objective - ground_energies[denominator]) / denominator)
            radius = sum((group["outcome_max"] - group["outcome_min"])
                         * math.sqrt(math.log(1920) / (2 * group["shots"]))
                         for group in validation["groups"]) / denominator
            lower, upper = value - radius, value + radius
            if case["algorithm"] == "qaoa":
                status = "supported" if lower >= 0.80 else "missed" if upper < 0.80 else "inconclusive"
            else:
                status = "supported" if upper <= 0.05 else "missed" if lower > 0.05 else "inconclusive"
            bound = validation["quality_bound"]
            assert bound["normalized_estimate"] == pytest.approx(value, abs=1e-12)
            assert bound["normalized_standard_error"] == pytest.approx(error / denominator, abs=1e-12)
            assert bound["simultaneous_radius"] == pytest.approx(radius, abs=1e-12)
            np.testing.assert_allclose(bound["simultaneous_interval"], [lower, upper], atol=1e-12)
            assert bound["group_failure_allocation"] == pytest.approx(0.05 / 48)
            assert bound["target_status"] == status
            assert bound["precision_supported"] is True
            assert radius <= 0.05
            row = summary[run["run_id"]]
            assert row["ideal_quality"] == pytest.approx(quality, abs=1e-11)
            assert row["ideal_target_met"] is run["target_met_audit"]
            assert row["heldout_target_status"] == status
            heldout_groups += len(validation["groups"])
    for run in study["controls"]:
        case = next(case for case in study["cases"] if case["case_id"] == run["case_id"])
        _assert_final_quality(case, run, ground_energies)
    assert heldout_groups == 48
    assert len(study["control_summary_rows"]) == 24
    with (ARTIFACTS / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == [row["run_id"] for row in study["summary_rows"]]
    for row in rows:
        expected = summary[row["run_id"]]
        assert row["target_met_exact"] == str(expected["ideal_target_met"])
        assert row["target_status_simultaneous"] == expected["heldout_target_status"]


def test_every_saved_group_and_qasm_source_is_bound_to_actual_bytes(saved_study):
    study, manifest, resources = (saved_study[name] for name in ("study", "manifest", "resources"))
    assert manifest["total_group_batches"] == 5280
    assert manifest["total_objective_calls"] == 3672
    by_sha = {entry["sha256"]: entry for entry in manifest["unique_circuits"]}
    assert len(by_sha) == manifest["unique_circuit_count"]
    for sha, entry in by_sha.items():
        assert hashlib.sha256((ARTIFACTS / entry["file"]).read_bytes()).hexdigest() == sha
        assert entry["measurement_map"] == [[wire, wire] for wire in range(entry["logical_qubits"])]
    linked_groups = iter(manifest["groups"])
    reconstructed, weighted, readouts = set(), {}, 0
    for case in study["cases"]:
        for run in case["runs"]:
            for phase, records in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for index, record in enumerate(records):
                    for group_index, group in enumerate(record["groups"]):
                        linked = next(linked_groups)
                        assert linked["case_id"] == case["case_id"]
                        assert linked["start_id"] == run["start_id"]
                        assert linked["seed"] == run["seed"]
                        assert linked["phase"] == phase
                        assert linked["evaluation_index"] == index
                        assert linked["group_index"] == group_index
                        assert linked["parameters"] == record["parameters"]
                        assert linked["shots"] == group["shots"]
                        assert linked["counts"] == group["counts"]
                        assert linked["rng"] == group["rng"]
                        sha = group["source_qasm_sha256"]
                        assert linked["source_qasm_sha256"] == sha
                        assert linked["source_gate_counts"] == group["circuit_gate_counts"]
                        assert linked["source_depth"] == group["circuit_depth"]
                        if sha not in reconstructed:
                            source = opt.diagnostic_group_circuit(case, record["parameters"], group["group"])
                            assert hashlib.sha256(qasm3.dumps(source).encode()).hexdigest() == sha
                            assert dict(source.count_ops()) == by_sha[sha]["gate_counts"]
                            assert source.depth() == by_sha[sha]["depth"]
                            reconstructed.add(sha)
                        n, p = case["n_qubits"], case["p"]
                        if case["algorithm"] == "qaoa":
                            per = dict(h=n, rx=n * p, rz=n * p, cx=2 * n * p, measure=n)
                        else:
                            per = dict(ry=n, cx=n - 1, measure=n)
                            if group["group"] == "all_x":
                                per["h"] = n
                        assert {gate: number for gate, number in group["circuit_gate_counts"].items()
                                if gate != "barrier"} == per
                        for gate, number in per.items():
                            weighted[gate] = weighted.get(gate, 0) + number * group["shots"]
                        readouts += n * group["shots"]
    assert next(linked_groups, None) is None
    assert reconstructed == set(by_sha)
    assert resources["weighted_gate_counts"] == weighted
    assert resources["measured_qubit_readouts"] == readouts == 15_748_800
    assert resources["total_preparations"] == 3_149_760
    assert resources["objective_calls"] == 3672
    assert resources["group_batches"] == 5280
    assert resources["logical_peak_qubits"] == 6
    assert hashlib.sha256((ARTIFACTS / "circuit-manifest.json").read_bytes()).hexdigest() == resources["circuit_manifest"]["sha256"]
    for filename, count in (("groups.csv", 5280), ("evaluations.csv", 3672),
                            ("control-evaluations.csv", 2424), ("controls.csv", 24), ("resources.csv", 36)):
        with (ARTIFACTS / filename).open(newline="") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == count


def _load_generator(path=None):
    path = path or ROOT / "scripts/generate_optimization_diagnostics.py"
    specification = importlib.util.spec_from_file_location("diagnostics_generator_test", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_recipe_freeze_preserves_producers_before_any_execution(tmp_path, monkeypatch):
    generator = _load_generator()
    def forbidden(*args, **kwargs):
        raise AssertionError("freezing producers must not execute any numerical run")
    monkeypatch.setattr(generator.opt, "run_diagnostic_sampled", forbidden)
    monkeypatch.setattr(generator.opt, "run_diagnostic_control", forbidden)
    result = generator.freeze(tmp_path)
    assert result["execution_performed"] is False
    assert not (tmp_path / "raw-study.json").exists()
    assert not (tmp_path / "runs.json").exists()
    assert not (tmp_path / "checkpoints").exists()
    expected_sources = {
        "optimization.py": ROOT / "src/openquantum/optimization.py",
        "generate_optimization_diagnostics.py": ROOT / "scripts/generate_optimization_diagnostics.py",
        "requirements-lock.txt": ROOT / "requirements-lock.txt",
        "LICENSE.txt": ROOT / "LICENSE",
    }
    for filename, source in expected_sources.items():
        assert (tmp_path / filename).read_bytes() == source.read_bytes()
    recipe_bytes = (tmp_path / "recipe.json").read_bytes()
    assert hashlib.sha256(recipe_bytes).hexdigest() == result["identity"]["recipe_sha256"]
    assert json.loads(recipe_bytes) == opt.diagnostic_recipe()
    original_freeze = (tmp_path / "freeze.json").read_bytes()
    assert generator.freeze(tmp_path)["identity"] == result["identity"]
    assert (tmp_path / "freeze.json").read_bytes() == original_freeze
    (tmp_path / "optimization.py").write_bytes(b"different producer bytes\n")
    with pytest.raises(ValueError):
        generator.freeze(tmp_path)


@pytest.mark.parametrize("corruption", ["identity", "kind", "key", "status"])
def test_checkpoint_resume_rejects_changed_binding_metadata(tmp_path, corruption):
    generator = _load_generator()
    identity = dict(recipe_sha256="a" * 64, implementation_sha256="b" * 64,
                    generator_sha256="c" * 64, environment_lock_sha256="d" * 64)
    path = tmp_path / "sampled-fixture.json"
    run = dict(run_id="fixture", parameters=[0.2, 0.1])
    generator.save_checkpoint(path, identity, kind="sampled", key="fixture", run=run)
    assert generator.load_checkpoint(path, identity, kind="sampled", key="fixture") == run
    record = json.loads(path.read_text())
    if corruption == "identity":
        record["identity"]["implementation_sha256"] = "e" * 64
    else:
        record[corruption] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        generator.load_checkpoint(path, identity, kind="sampled", key="fixture")


def test_saved_frozen_sources_raw_checkpoint_and_per_run_bindings(saved_study):
    study, raw, freeze = (saved_study[name] for name in ("study", "raw", "freeze"))
    identity = freeze["identity"]
    assert saved_study["recipe"] == study["recipe"] == raw["recipe"] == opt.diagnostic_recipe()
    assert hashlib.sha256((ARTIFACTS / "recipe.json").read_bytes()).hexdigest() == identity["recipe_sha256"]
    for filename, key in (("optimization.py", "implementation_sha256"),
                          ("generate_optimization_diagnostics.py", "generator_sha256"),
                          ("requirements-lock.txt", "environment_lock_sha256")):
        assert hashlib.sha256((ARTIFACTS / filename).read_bytes()).hexdigest() == identity[key]
        assert study["provenance"][key] == identity[key]
        assert raw["provenance"][key] == identity[key]
    raw_hash = hashlib.sha256((ARTIFACTS / "raw-study.json").read_bytes()).hexdigest()
    assert study["numerical_input_sha256"] == saved_study["resources"]["numerical_input_sha256"] == raw_hash
    assert raw["checkpoint_identity"] == identity
    assert raw["execution_status"] == "completed_bounded_ideal_sampling_and_exact_cpu_controls"
    assert study["artifact_status"] == "complete"
    assert study["provenance"]["source_commit"] is not None
    assert study["provenance"]["dirty_owned_source"] is False
    assert study["provenance"]["python"] == "3.12.14"
    assert study["provenance"]["device"]["accelerator"] is None
    expected_files = set()
    for case in raw["cases"]:
        for run in case["runs"]:
            filename = f"sampled-{run['run_id']}.json"
            expected_files.add(filename)
            checkpoint = json.loads((ARTIFACTS / "checkpoints" / filename).read_text())
            assert checkpoint["identity"] == identity
            assert checkpoint["kind"] == "sampled"
            assert checkpoint["key"] == run["run_id"]
            assert checkpoint["status"] == "completed"
            assert checkpoint["run"] == run
    for run in raw["controls"]:
        filename = f"control-{run['run_id']}.json"
        expected_files.add(filename)
        checkpoint = json.loads((ARTIFACTS / "checkpoints" / filename).read_text())
        assert checkpoint["identity"] == identity
        assert checkpoint["kind"] == "control"
        assert checkpoint["key"] == run["run_id"]
        assert checkpoint["status"] == "completed"
        assert checkpoint["run"] == run
    assert {path.name for path in (ARTIFACTS / "checkpoints").glob("*.json")} == expected_files
    assert len(expected_files) == 60
    # Export annotations are the only additions to sampled raw group records.
    exported_copy = deepcopy(study["cases"])
    for case in exported_copy:
        for run in case["runs"]:
            for record in [*run["evaluations"], run["validation"]]:
                for group in record["groups"]:
                    group.pop("source_qasm_sha256")
                    group.pop("source_qasm_file")
    assert exported_copy == raw["cases"]
    assert study["controls"] == raw["controls"]


@pytest.mark.parametrize("shape", ["standalone", "checkout"])
def test_standalone_provenance_does_not_borrow_an_ancestor_git_repository(tmp_path, shape):
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    subprocess.run(["git", "init", "-q", str(ancestor)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(ancestor), "-c", "user.name=Provenance test",
                    "-c", "user.email=provenance-test@example.invalid", "commit", "-q",
                    "--allow-empty", "-m", "Ancestor fixture"], check=True, capture_output=True)
    child = ancestor / "download"
    child.mkdir()
    if shape == "checkout":
        source = child / "src/openquantum/optimization.py"
        destination = child / "scripts/generate_optimization_diagnostics.py"
        source.parent.mkdir(parents=True)
        destination.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/source_provenance.py", child / "scripts/source_provenance.py")
    else:
        source = child / "optimization.py"
        destination = child / "generate_optimization_diagnostics.py"
    shutil.copy2(ROOT / "src/openquantum/optimization.py", source)
    shutil.copy2(ROOT / "scripts/generate_optimization_diagnostics.py", destination)
    generator = _load_generator(destination)
    metadata = generator.provenance()
    assert metadata["source_commit"] is None
    assert metadata["dirty_owned_source"] is None


@pytest.mark.parametrize("corruption", ["iteration", "proposal", "count_mean", "count_variance", "accepted", "final"])
def test_sampled_saved_replay_rejects_count_and_path_corruption(saved_study, corruption):
    case = saved_study["study"]["cases"][0]
    run = deepcopy(case["runs"][0])
    if corruption == "iteration":
        run["iterations"][0]["step"] += 0.1
    elif corruption == "proposal":
        run["evaluations"][2]["parameters"][0] += 0.1
    elif corruption == "count_mean":
        run["evaluations"][0]["groups"][0]["estimate"] += 1
    elif corruption == "count_variance":
        run["evaluations"][0]["groups"][0]["sample_variance"] += 1
    elif corruption == "accepted":
        run["iterations"][0]["accepted"] = not run["iterations"][0]["accepted"]
    else:
        run["parameters"][0] += 0.1
    with pytest.raises(ValueError):
        opt.diagnostic_replay_sampled(case, run)


@pytest.mark.parametrize("corruption", ["actual_preparations", "counts", "variance", "effective_se", "virtual_shots", "proposal", "decision", "final"])
def test_control_saved_replay_rejects_cost_and_moment_corruption(saved_study, corruption):
    case = saved_study["study"]["cases"][0]
    run = deepcopy(saved_study["study"]["controls"][0])
    record = run["evaluations"][0]
    if corruption == "actual_preparations":
        run["quantum_preparations"] = 1
    elif corruption == "counts":
        record["count_samples"] = {"0000": 128}
    elif corruption == "variance":
        record["groups"][0]["population_variance"] += 1
    elif corruption == "effective_se":
        record["effective_standard_error"] = 1
    elif corruption == "virtual_shots":
        record["virtual_shots_per_group"] += 1
    elif corruption == "proposal":
        run["evaluations"][2]["parameters"][0] += 0.1
    elif corruption == "decision":
        run["iterations"][0]["accepted"] = not run["iterations"][0]["accepted"]
    else:
        run["parameters"][0] += 0.1
    with pytest.raises(ValueError):
        opt.diagnostic_replay_control(case, run)


@pytest.mark.parametrize("corruption", ["case_order", "missing_seed", "initial", "group_allocation", "rng_stream", "gate_counts", "run_total", "study_total"])
def test_complete_saved_resource_ledger_rejects_corrupted_allocations(saved_study, corruption):
    study = deepcopy(saved_study["study"])
    run = study["cases"][0]["runs"][0]
    group = run["evaluations"][0]["groups"][0]
    if corruption == "case_order":
        study["cases"][0], study["cases"][1] = study["cases"][1], study["cases"][0]
    elif corruption == "missing_seed":
        study["cases"][0]["runs"].pop()
    elif corruption == "initial":
        run["initial"][0] = math.pi / 4
    elif corruption == "group_allocation":
        group["shots"] += 1
    elif corruption == "rng_stream":
        group["rng"]["stream_key"][1] += 1
    elif corruption == "gate_counts":
        group["circuit_gate_counts"]["cx"] += 1
    elif corruption == "run_total":
        run["total_shots"] += 1
    else:
        study["total_shots"] += 1
    with pytest.raises(ValueError):
        opt.diagnostic_resources(study)
