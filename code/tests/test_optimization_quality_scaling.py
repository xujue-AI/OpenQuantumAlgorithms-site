"""Independent multi-layer behavior, sampled decisions, and coverage budgets."""

import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from openquantum import optimization as opt


def _graph(n_qubits, family):
    edges = [(index, index + 1) for index in range(n_qubits - 1)]
    if family == "cycle":
        edges.append((0, n_qubits - 1))
    return edges


def _cut_values(n_qubits, edges):
    return np.array([sum(((index >> u) & 1) != ((index >> v) & 1)
                         for u, v in edges) for index in range(1 << n_qubits)])


def _bitwise_qaoa_state(n_qubits, edges, gammas, betas):
    """Apply independent classical phases and exact bit-flip exponentials."""
    cuts = _cut_values(n_qubits, edges)
    state = np.full(1 << n_qubits, 2.0 ** (-n_qubits / 2), dtype=complex)
    indices = np.arange(1 << n_qubits)
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * gamma * cuts)
        for wire in range(n_qubits):
            state = math.cos(beta) * state - 1j * math.sin(beta) * state[indices ^ (1 << wire)]
    return state, cuts


@pytest.mark.parametrize("n_qubits", range(3, 9))
@pytest.mark.parametrize("family", ["path", "cycle"])
@pytest.mark.parametrize("layers", [1, 2])
def test_multilayer_path_and_cycle_circuits_match_independent_bitwise_evolution(n_qubits, family, layers):
    edges = _graph(n_qubits, family)
    gammas, betas = [-0.37, 0.51][:layers], [0.23, -0.14][:layers]
    reference, cuts = _bitwise_qaoa_state(n_qubits, edges, gammas, betas)
    source = opt.build_qaoa_circuit(n_qubits, edges, gammas, betas)
    state = Statevector.from_instruction(source)
    assert state.equiv(Statevector(reference), atol=1e-11)
    assert state.probabilities() @ cuts == pytest.approx(np.abs(reference) ** 2 @ cuts, abs=1e-11)
    assert opt.qaoa_expectation(n_qubits, edges, gammas, betas) == pytest.approx(np.abs(reference) ** 2 @ cuts, abs=1e-11)
    optimum = n_qubits - 1 if family == "path" or n_qubits % 2 else n_qubits
    assert cuts.max() == optimum
    assert source.count_ops()["h"] == n_qubits
    assert source.count_ops()["rx"] == n_qubits * layers
    assert source.count_ops()["rz"] == len(edges) * layers
    assert source.count_ops()["cx"] == 2 * len(edges) * layers


from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from qiskit import qasm3

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "assets/generated/optimization/quality-scaling"


def _raw_score(case, bits, group):
    index = int(bits, 2)
    if case["algorithm"] == "qaoa":
        return sum(((index >> u) & 1) != ((index >> v) & 1) for u, v in case["edges"])
    spins = [1 - 2 * ((index >> wire) & 1) for wire in range(case["n_qubits"])]
    if group == "all_z":
        return -sum(spins)
    return 0.5 * sum(a * b for a, b in zip(spins, spins[1:]))


def _record_statistics(case, record):
    means, variances = [], []
    for group in record["groups"]:
        shots = group["shots"]
        scores = {bits: _raw_score(case, bits, group["group"]) for bits in group["counts"]}
        mean = sum(number * scores[bits] for bits, number in group["counts"].items()) / shots
        variance = sum(number * (scores[bits] - mean) ** 2
                       for bits, number in group["counts"].items()) / (shots - 1)
        assert group["estimate"] == pytest.approx(mean, abs=1e-12)
        assert group["sample_variance"] == pytest.approx(variance, abs=1e-12)
        assert group["estimated_standard_error"] == pytest.approx(math.sqrt(variance / shots), abs=1e-12)
        means.append(mean)
        variances.append(variance / shots)
    value = -means[0] if case["algorithm"] == "qaoa" else sum(means)
    error = math.sqrt(sum(variances))
    assert record["objective_estimate"] == pytest.approx(value, abs=1e-12)
    assert record["estimated_standard_error"] == pytest.approx(error, abs=1e-12)
    return value, error


@pytest.fixture(scope="module")
def full_study():
    paths = [ARTIFACTS / name for name in ("recipe.json", "runs.json", "resources.json", "circuit-manifest.json")]
    assert all(path.is_file() for path in paths), "Generate the fixed study artifacts before checking the recorded evidence"
    return tuple(json.loads(path.read_text()) for path in paths)


@pytest.fixture(scope="module")
def small_study():
    return opt.run_quality_scaling_demo(seeds=(7,), schedule=((0.4, 32, 1),), validation_shots=64,
                                       case_ids=("qaoa-cycle-n4-p1", "vqe-open-chain-n3"))


def test_frozen_recipe_has_predetermined_sizes_starts_budgets_and_distinct_namespaces():
    recipe = opt.quality_scaling_recipe()
    assert recipe["fixed_published_recipe"] is True
    assert recipe["seeds"] == [7, 19, 41]
    assert recipe["schedule"] == [[0.4, 256, 4], [0.2, 512, 4], [0.1, 1024, 4]]
    assert recipe["planned"] == dict(runs=72, training_calls=6336, validation_calls=72,
                                      objective_calls=6408, training_groups=7572,
                                      validation_groups=84, group_batches=7656,
                                      training_shots=4483584, validation_shots=1376256,
                                      total_shots=5859840)
    assert recipe["validation_shots_per_group"] == 16384
    assert recipe["validation_failure_per_group"] == pytest.approx(0.05 / 84)
    assert len(recipe["cases"]) == 24
    assert [case["stream_id"] for case in recipe["cases"]] == list(range(6, 30))
    for case in recipe["cases"]:
        if case["algorithm"] == "qaoa":
            assert case["n_qubits"] in (3, 4, 5, 6, 8)
            assert case["p"] in (1, 2)
            assert case["initial"] == [0.2, 0.1] * case["p"]
            assert case["target"]["ratio_at_least"] == 0.80
            assert case["edges"] == [list(edge) for edge in _graph(case["n_qubits"], case["graph_family"])]
        else:
            assert case["n_qubits"] in (2, 3, 4, 6)
            assert case["initial"] == [0.6] * case["n_qubits"]
            assert case["target"]["density_gap_at_most"] == 0.05


def test_recorded_study_preserves_every_fixed_case_and_seed_including_misses(full_study):
    recipe, study, resources, _ = full_study
    assert recipe == opt.quality_scaling_recipe() == study["recipe"]
    assert len(study["cases"]) == 24
    assert study["exact_diagnostics_used_for_optimization"] is False
    assert study["total_shots"] == 5859840
    assert study["training_shots"] == 4483584
    assert study["validation_shots"] == 1376256
    assert len(study["summary_rows"]) == 72
    assert study["hardware_feedback_seconds"] is None
    assert study["workflow_wall_seconds"] > 0
    assert study["workflow_process_cpu_seconds"] > 0
    for expected, case in zip(recipe["cases"], study["cases"], strict=True):
        assert {key: case[key] for key in expected} == expected
        assert [run["seed"] for run in case["runs"]] == recipe["seeds"]
        assert [run["run_index"] for run in case["runs"]] == [0, 1, 2]
        assert len(case["runs"]) == 3
        for run in case["runs"]:
            assert run["stop_reason"] == "fixed_schedule_exhausted"
            assert run["convergence_certified"] is False
            assert run["exact_diagnostics_used_for_optimization"] is False
            assert run["total_shots_including_validation"] == expected["planned"]["total_shots_per_seed"]
            assert run["validation"]["parameters"] == run["parameters"]
            assert run["diagnostic"]["audit_performed_after_training"] is True
    assert resources["total_preparations"] == study["total_shots"]
    assert resources["hardware_feedback_seconds"] is None


def test_all_optimizer_moves_replay_from_recorded_counts_only(full_study):
    _, study, _, _ = full_study
    for case in study["cases"]:
        dimension = len(case["initial"])
        for run in case["runs"]:
            records = run["evaluations"]
            statistics = [_record_statistics(case, record) for record in records]
            assert len(records) == 1 + 12 * (1 + 2 * dimension)
            assert run["objective_evaluations"] == len(records)
            point = (np.array(case["initial"]) + math.pi) % (2 * math.pi) - math.pi
            np.testing.assert_allclose(records[0]["parameters"], point, atol=1e-14)
            accepted_indexes = []
            for sweep, iteration in enumerate(run["iterations"]):
                incumbent_index = 1 + sweep * (1 + 2 * dimension)
                proposal_indexes = list(range(incumbent_index + 1, incumbent_index + 1 + 2 * dimension))
                assert iteration["incumbent_evaluation"] == incumbent_index
                assert iteration["proposal_evaluations"] == proposal_indexes
                np.testing.assert_allclose(iteration["parameters_before"], point, atol=1e-14)
                np.testing.assert_allclose(records[incumbent_index]["parameters"], point, atol=1e-14)
                for offset, index in enumerate(proposal_indexes):
                    axis, direction = offset // 2, -1 if offset % 2 == 0 else 1
                    candidate = point.copy()
                    candidate[axis] += direction * iteration["step"]
                    candidate = (candidate + math.pi) % (2 * math.pi) - math.pi
                    np.testing.assert_allclose(records[index]["parameters"], candidate, atol=1e-14)
                    assert records[index]["axis"] == axis and records[index]["direction"] == direction
                best_index = min(proposal_indexes, key=lambda index: statistics[index][0])
                improvement = statistics[incumbent_index][0] - statistics[best_index][0]
                margin = 2.0 * math.hypot(statistics[incumbent_index][1], statistics[best_index][1])
                accepted = improvement > margin
                assert iteration["best_proposal_evaluation"] == best_index
                assert iteration["sampled_improvement"] == pytest.approx(improvement, abs=1e-12)
                assert iteration["comparison_margin"] == pytest.approx(margin, abs=1e-12)
                assert iteration["accepted"] is accepted
                if accepted:
                    point = np.array(records[best_index]["parameters"])
                    accepted_indexes.append(best_index)
                np.testing.assert_allclose(iteration["parameters_after"], point, atol=1e-14)
            np.testing.assert_allclose(run["parameters"], point, atol=1e-14)
            assert [record["index"] for record in records if record["accepted"]] == accepted_indexes
            assert run["accepted_proposals"] == len(accepted_indexes)
            assert run["rejected_proposals"] == 24 * dimension - len(accepted_indexes)
            assert all("exact_objective_audit" not in record for record in records)


def test_qaoa_sampled_objective_cannot_consult_exact_references(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sampled decisions must not consult exact references")
    monkeypatch.setattr(opt, "qaoa_layers_reference", forbidden)
    monkeypatch.setattr(opt, "qaoa_expectation", forbidden)
    record = opt.sampled_qaoa_layers_objective([0.2, 0.1, -0.3, 0.4], n_qubits=4,
                                              edges=_graph(4, "cycle"), shots=64, seed=7,
                                              evaluation_index=0, stream_id=6)
    assert record["total_shots"] == 64
    _record_statistics(dict(algorithm="qaoa", edges=_graph(4, "cycle")), record)
    assert "exact_objective_audit" not in record


def test_qaoa_postrun_references_and_target_misses_match_independent_states(full_study):
    _, study, _, _ = full_study
    c4 = next(case for case in study["cases"] if case["case_id"] == "qaoa-cycle-n4-p1")
    for case in study["cases"]:
        if case["algorithm"] != "qaoa":
            continue
        n = case["n_qubits"]
        for run in case["runs"]:
            parameters = run["parameters"]
            state, cuts = _bitwise_qaoa_state(n, case["edges"], parameters[0::2], parameters[1::2])
            probabilities = np.abs(state) ** 2
            expected = float(probabilities @ cuts)
            ratio = expected / int(cuts.max())
            diagnostic = run["diagnostic"]
            assert diagnostic["classical_optimum"] == int(cuts.max())
            assert diagnostic["cut_values"] == cuts.tolist()
            assert diagnostic["exact_expected_cut"] == pytest.approx(expected, abs=1e-11)
            assert diagnostic["exact_approximation_ratio"] == pytest.approx(ratio, abs=1e-11)
            assert diagnostic["exact_optimal_cut_probability"] == pytest.approx(probabilities[cuts == cuts.max()].sum(), abs=1e-11)
            assert run["target_met_audit"] is bool(ratio >= 0.80)
    for run in c4["runs"]:
        assert run["diagnostic"]["exact_approximation_ratio"] <= 0.75 + 1e-12
        assert run["target_met_audit"] is False


def test_simultaneous_holdouts_use_raw_joint_scores_and_the_declared_union_bound(full_study):
    recipe, study, _, _ = full_study
    streams = set()
    heldout_groups = 0
    for case in study["cases"]:
        for run in case["runs"]:
            validation = run["validation"]
            _record_statistics(case, validation)
            bound = validation["quality_bound"]
            group_delta = 0.05 / 84
            denominator = run["diagnostic"]["classical_optimum"] if case["algorithm"] == "qaoa" else case["n_qubits"]
            radius = sum((group["outcome_max"] - group["outcome_min"])
                         * math.sqrt(math.log(2 / group_delta) / (2 * group["shots"]))
                         for group in validation["groups"]) / denominator
            value = (-validation["objective_estimate"] / denominator if case["algorithm"] == "qaoa"
                     else (validation["objective_estimate"] - run["diagnostic"]["ground_energy"]) / denominator)
            assert bound["normalized_estimate"] == pytest.approx(value, abs=1e-12)
            assert bound["simultaneous_radius"] == pytest.approx(radius, abs=1e-12)
            np.testing.assert_allclose(bound["simultaneous_interval"], [value - radius, value + radius], atol=1e-12)
            if case["algorithm"] == "qaoa":
                status = "supported" if value - radius >= 0.80 else "missed" if value + radius < 0.80 else "inconclusive"
            else:
                status = "supported" if value + radius <= 0.05 else "missed" if value - radius > 0.05 else "inconclusive"
            assert bound["target_status"] == status
            assert bound["precision_supported"] is True
            assert radius <= 0.05
            assert bound["group_failure_allocation"] == pytest.approx(group_delta)
            for group_index, group in enumerate(validation["groups"]):
                assert group["shots"] == 16384
                expected_stream = (run["seed"], case["stream_id"], 1, 0, group_index)
                assert tuple(group["rng"]["stream_key"]) == expected_stream
                assert expected_stream not in streams
                streams.add(expected_stream)
                heldout_groups += 1
            for call_index, record in enumerate(run["evaluations"]):
                for group_index, group in enumerate(record["groups"]):
                    expected_stream = (run["seed"], case["stream_id"], 0, call_index, group_index)
                    assert tuple(group["rng"]["stream_key"]) == expected_stream
                    assert expected_stream not in streams
                    streams.add(expected_stream)
    assert heldout_groups == recipe["planned"]["validation_groups"] == 84
    assert len(streams) == 7656


def test_complete_logical_workload_matches_all_recorded_group_weights(full_study):
    _, study, resources, _ = full_study
    expected, measured = {}, 0
    for case in study["cases"]:
        n, p = case["n_qubits"], case["p"]
        for run in case["runs"]:
            for record in [*run["evaluations"], run["validation"]]:
                for group in record["groups"]:
                    if case["algorithm"] == "qaoa":
                        per = {"h": n, "rx": n * p, "rz": len(case["edges"]) * p,
                               "cx": 2 * len(case["edges"]) * p, "measure": n}
                    else:
                        per = {"ry": n, "cx": n - 1, "measure": n}
                        if group["group"] == "all_x":
                            per["h"] = n
                    assert {key: count for key, count in group["circuit_gate_counts"].items() if key != "barrier"} == per
                    for gate, count in per.items():
                        expected[gate] = expected.get(gate, 0) + count * group["shots"]
                    measured += n * group["shots"]
    assert resources["weighted_gate_counts"] == expected
    assert measured == resources["measured_qubit_readouts"] == 28528896
    assert resources["objective_calls"] == 6408
    assert resources["group_batches"] == 7656
    assert resources["total_preparations"] == 5859840
    assert resources["logical_peak_qubits"] == 8


@pytest.mark.parametrize("corruption", ["run_index", "seed", "call_index", "holdout_parameters", "group_names",
                                         "rng_stream", "rng_entropy", "rng_spawn", "count_shape", "depth", "gate_counts",
                                         "width", "readout_total"])
def test_complete_ledger_rejects_corrupted_recorded_invariants(small_study, corruption):
    study = deepcopy(small_study)
    run = study["cases"][0]["runs"][0]
    group = run["evaluations"][0]["groups"][0]
    if corruption == "run_index": run["run_index"] += 1
    elif corruption == "seed": run["seed"] += 1
    elif corruption == "call_index": run["evaluations"][0]["index"] += 1
    elif corruption == "holdout_parameters": run["validation"]["parameters"][0] += 0.1
    elif corruption == "group_names": group["group"] = "all_x"
    elif corruption == "rng_stream": group["rng"]["stream_key"][1] += 1
    elif corruption == "rng_entropy": group["rng"]["entropy"][1] += 1
    elif corruption == "rng_spawn": group["rng"]["spawn_key"][0] = 1
    elif corruption == "count_shape":
        key = next(iter(group["counts"])); group["counts"][key + "0"] = group["counts"].pop(key)
    elif corruption == "depth": group["circuit_depth"] += 1
    elif corruption == "gate_counts": group["circuit_gate_counts"]["cx"] += 1
    elif corruption == "width": group["logical_qubits"] += 1
    elif corruption == "readout_total": group["measured_qubit_readouts"] += 1
    with pytest.raises(ValueError):
        opt.quality_scaling_resources(study)


def test_custom_fixture_is_labeled_and_caps_preflight_before_any_simulator(monkeypatch):
    assert opt.quality_scaling_recipe(case_ids=("qaoa-cycle-n4-p1",), seeds=(7,),
                                      schedule=((0.4, 32, 1),), validation_shots=64)["fixed_published_recipe"] is False
    large = opt.quality_scaling_graph(32, "cycle")
    assert opt.build_qaoa_circuit(32, large, [0.2, 0.2], [0.1, 0.1]).num_qubits == 32
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must reject before allocating a quantum state")
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    with pytest.raises(ValueError):
        opt.sampled_qaoa_layers_objective([0.2, 0.1], n_qubits=9, edges=_graph(9, "path"),
                                         shots=64, seed=7, evaluation_index=0)
    with pytest.raises(ValueError):
        opt.qaoa_layers_reference(9, _graph(9, "path"), [0.2, 0.1], max_qubits=32)
    with pytest.raises(ValueError):
        opt.quality_scaling_recipe(schedule=((0.4, 65536, 128),))
    with pytest.raises(RuntimeError):
        opt.run_quality_scaling_demo(case_ids=("qaoa-cycle-n4-p1",), seeds=(7,), max_wall_seconds=1e-12)


@pytest.mark.parametrize("kwargs", [dict(seeds=(7, 7)), dict(validation_shots=65537), dict(case_ids=("missing",)),
                                   dict(schedule=()), dict(schedule=((0.4, 64, 129),))])
def test_invalid_recipe_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        opt.quality_scaling_recipe(**kwargs)


def test_every_source_export_and_manifest_group_is_bound_to_actual_bytes(full_study):
    _, study, resources, manifest = full_study
    assert manifest["total_group_batches"] == 7656
    assert manifest["total_objective_calls"] == 6408
    by_sha = {item["sha256"]: item for item in manifest["unique_circuits"]}
    assert len(by_sha) == manifest["unique_circuit_count"]
    for sha, item in by_sha.items():
        path = ARTIFACTS / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        assert item["measurement_map"] == [[wire, wire] for wire in range(item["logical_qubits"])]
    groups = iter(manifest["groups"])
    reconstructed = set()
    for case in study["cases"]:
        for run in case["runs"]:
            for phase, records in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for call_index, record in enumerate(records):
                    for group_index, group in enumerate(record["groups"]):
                        linked = next(groups)
                        assert linked["case_id"] == case["case_id"] and linked["seed"] == run["seed"]
                        assert linked["phase"] == phase and linked["evaluation_index"] == call_index
                        assert linked["parameters"] == record["parameters"]
                        assert linked["group_index"] == group_index and linked["shots"] == group["shots"]
                        assert linked["counts"] == group["counts"] and linked["rng"] == group["rng"]
                        sha = group["source_qasm_sha256"]
                        assert linked["source_qasm_sha256"] == sha and sha in by_sha
                        assert linked["source_gate_counts"] == group["circuit_gate_counts"]
                        if sha not in reconstructed:
                            source = opt.quality_scaling_group_circuit(case, record["parameters"], group["group"])
                            assert hashlib.sha256(qasm3.dumps(source).encode()).hexdigest() == sha
                            assert dict(source.count_ops()) == by_sha[sha]["gate_counts"]
                            assert source.depth() == by_sha[sha]["depth"]
                            reconstructed.add(sha)
    assert next(groups, None) is None
    assert reconstructed == set(by_sha)
    assert hashlib.sha256((ARTIFACTS / "circuit-manifest.json").read_bytes()).hexdigest() == resources["circuit_manifest"]["sha256"]
    with (ARTIFACTS / "groups.csv").open(newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 7656
    with (ARTIFACTS / "evaluations.csv").open(newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 6408


def test_recorded_module_and_generator_download_hashes_match_the_executed_sources(full_study):
    _, study, _, _ = full_study
    provenance = study["provenance"]
    # The measured study keeps its executed snapshot when later defensive
    # validation changes extend the live module or generator. Circuit reconstruction is
    # independently checked against every exported source above.
    assert hashlib.sha256((ARTIFACTS / "optimization.py").read_bytes()).hexdigest() == provenance["implementation_sha256"]
    assert hashlib.sha256((ARTIFACTS / "generate_optimization_quality_scaling.py").read_bytes()).hexdigest() == provenance["generator_sha256"]
    assert provenance["source_changed_during_generation"] is False


@pytest.mark.parametrize("checkout_shape", [False, True])
def test_standalone_provenance_never_borrows_an_ancestor_git_repository(tmp_path, checkout_shape):
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    subprocess.run(["git", "init", "-q", str(ancestor)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(ancestor), "-c", "user.name=Provenance test",
                    "-c", "user.email=provenance-test@example.invalid", "commit", "-q", "--allow-empty", "-m", "Ancestor fixture"],
                   check=True, capture_output=True)
    child = ancestor / "download"
    child.mkdir()
    if checkout_shape:
        (child / "src/openquantum").mkdir(parents=True)
        (child / "scripts").mkdir()
        shutil.copy2(ARTIFACTS / "optimization.py", child / "src/openquantum/optimization.py")
        destination = child / "scripts/generate_optimization_quality_scaling.py"
        helper = ROOT / "scripts/source_provenance.py"
        if helper.is_file():
            shutil.copy2(helper, child / "scripts/source_provenance.py")
    else:
        shutil.copy2(ARTIFACTS / "optimization.py", child / "optimization.py")
        destination = child / "generate_optimization_quality_scaling.py"
    shutil.copy2(ARTIFACTS / "generate_optimization_quality_scaling.py", destination)
    spec = importlib.util.spec_from_file_location(f"quality_standalone_test_{checkout_shape}", destination)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.provenance()
    assert metadata["source_commit"] is None
    assert metadata["dirty_owned_source"] is None
    assert metadata["source_changed_during_generation"] is False


@pytest.mark.parametrize("corruption", ["shift_group_allocation", "training_total", "study_total"])
def test_complete_ledger_reconciles_prescribed_allocations_and_actual_totals(small_study, corruption):
    study = deepcopy(small_study)
    run = study["cases"][0]["runs"][0]
    if corruption == "shift_group_allocation":
        call = run["evaluations"][0]
        group = call["groups"][0]
        group["shots"] += 1
        bits = next(iter(group["counts"]))
        group["counts"][bits] += 1
        group["measured_qubit_readouts"] += group["logical_qubits"]
        call["total_shots"] += 1
    elif corruption == "training_total":
        run["total_shots"] += 1
    else:
        study["training_shots"] += 1
    with pytest.raises(ValueError):
        opt.quality_scaling_resources(study)
