"""Uncapped versus stopped retry costs, checked against finite geometric sums."""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from openquantum.simulation.thermal_filter import (
    expected_trials_stopped_at_cap, thermal_energy_workload,
)


ROOT = Path(__file__).resolve().parents[1]


def load_generator(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("probability,cap", [(.125, 7), (.5, 3), (.875, 10), (1, 5)])
def test_stopped_expectation_matches_independent_survival_sum(probability, cap):
    # P(T > k) supplies a direct finite sum, independent of the closed form.
    expected = math.fsum((1-probability)**k for k in range(cap))
    assert expected_trials_stopped_at_cap(probability, cap) == pytest.approx(expected, rel=1e-14)


def test_stopped_expectation_retains_tiny_probability_and_certain_success_endpoints():
    assert 1-1e-20 == 1  # Naive 1-(1-p)^R would cancel to zero here.
    assert expected_trials_stopped_at_cap(1e-20, 7) == pytest.approx(7, rel=1e-14)
    assert expected_trials_stopped_at_cap(1, 230) == 1
    assert expected_trials_stopped_at_cap(math.nextafter(1, 0), 230) == pytest.approx(1)


@pytest.mark.parametrize("probability,cap", [(0, 3), (-.1, 3), (1.1, 3), (True, 3),
    (float("nan"), 3), (.5, True), (.5, 0), (.5, -1), (.5, 1.5)])
def test_stopped_expectation_rejects_invalid_probability_or_cap(probability, cap):
    with pytest.raises(ValueError):
        expected_trials_stopped_at_cap(probability, cap)


@pytest.mark.parametrize("probability", [.125, .5, 1])
def test_grouped_workloads_label_uncapped_cost_and_preserve_full_cap_accounting(probability):
    workload = thermal_energy_workload(probability)
    stopped_total = 0
    for group in workload["groups"]:
        slots = group["accepted_shots"]
        cap = group["complete_trial_cap_per_accepted_shot"]
        stopped = slots*math.fsum((1-probability)**k for k in range(cap))
        assert group["expected_complete_trials_uncapped"] == slots/probability
        assert group["expected_complete_trials"] == group["expected_complete_trials_uncapped"]
        assert "uncapped" in group["expected_complete_trials_scope"]
        assert group["expected_complete_trials_stopped_at_cap"] == pytest.approx(stopped, rel=1e-14)
        assert group["worst_case_complete_trials"] == slots*cap
        assert stopped <= group["expected_complete_trials_uncapped"]
        assert stopped <= group["worst_case_complete_trials"]
        stopped_total += stopped
    assert workload["expected_complete_trials_stopped_at_cap"] == pytest.approx(stopped_total, rel=1e-14)
    assert workload["expected_complete_trials"] == workload["expected_complete_trials_uncapped"]
    assert workload["physical_failure_per_complete_trial_proposal"]*workload["worst_case_complete_trials"] == pytest.approx(.005)
    assert "all M slots are attempted" in workload["stopped_expectation_scope"]
    assert "without early abort or conditioning" in workload["stopped_expectation_scope"]
    assert workload["sampling_performed"] is False


def test_constant_energy_has_zero_retry_workload():
    workload = thermal_energy_workload(.5, terms=((.6, "I"),))
    assert workload["groups"] == []
    assert workload["worst_case_complete_trials"] == 0
    assert workload["expected_complete_trials_uncapped"] == 0
    assert workload["expected_complete_trials_stopped_at_cap"] == 0


@pytest.mark.parametrize("probability", [1/16, .5, 1, 1e-20])
def test_qsp_projection_separates_expectations_without_changing_cap_counts(probability):
    scaling = load_generator("generate_simulation_qsp_scaling")
    counts = {"cx": 12, "rz": 3, "measure": 5}
    projection = scaling.workload_projection(probability, counts, 5, 10)
    slots = projection["required_independent_accepted_Z_samples"]
    cap = projection["complete_attempt_cap_per_accepted_sample"]
    if probability == 1e-20:
        # The cap is enormous, so check its stable survival probability instead
        # of enumerating trials. Naive (1-p)^R would incorrectly return one.
        slot_failure = math.exp(cap*math.log1p(-probability))
        expected = slots*(-math.expm1(cap*math.log1p(-probability)))/probability
        assert slot_failure < 1
    else:
        slot_failure = (1-probability)**cap
        expected = slots*math.fsum((1-probability)**k for k in range(cap))
    assert projection["expected_complete_attempts_all_samples_uncapped"] == slots/probability
    assert projection["expected_complete_attempts_all_samples"] == projection["expected_complete_attempts_all_samples_uncapped"]
    assert "uncapped" in projection["expected_complete_attempts_all_samples_scope"]
    assert projection["expected_complete_attempts_all_samples_stopped_at_cap"] == pytest.approx(expected, rel=1e-14)
    assert projection["modeled_no_acceptance_probability_per_slot"] == pytest.approx(slot_failure, rel=1e-14)
    assert projection["union_bound_acceptance_failure_all_samples"] <= .005+1e-15
    assert projection["complete_attempt_cap_all_samples"] == slots*cap
    assert projection["primitive_gate_counts_all_capped_attempts"] == {gate: count*slots*cap for gate, count in counts.items()}
    assert projection["logical_qubit_layer_volume_all_capped_attempts"] == 5*10*slots*cap
    assert "all M slots are attempted" in projection["stopped_expectation_scope"]
    assert projection["certified"] is False


def test_fresh_qsp_scaling_rows_include_both_expectations(tmp_path):
    scaling = load_generator("generate_simulation_qsp_scaling")
    scaling.generate(tmp_path, qubits=1)
    resources = json.loads((tmp_path / "logical-resources.json").read_text())
    projection = resources["instances"][0]["observable_workload_projection"]
    with (tmp_path / "scaling.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    for key in ("expected_complete_attempts_all_samples_uncapped",
                "expected_complete_attempts_all_samples_stopped_at_cap"):
        assert float(row[key]) == projection[key]
    assert row["certified"] == "False"


def test_fresh_amplification_ledgers_propagate_cost_labels_and_keep_cap_counts(tmp_path):
    generator = load_generator("generate_simulation_preparation_amplification")
    generator.generate(tmp_path, rounds=(0, 1), betas=(1.25,), scaling=False)
    report = json.loads((tmp_path / "logical-resources.json").read_text())
    for case in report["cases"]:
        for axis, measured in case["per_measurement_axis"].items():
            counts = measured["primitive_gate_counts_per_complete_attempt"]
            for key, source in (("ideal_numeric_workload", "accepted_shot_workload_proposal"),
                                ("hypothetical_synthesis_workload", "hypothetical_synthesis_workload_proposal")):
                group = next(g for g in case[source]["groups"] if g["axis"] == axis)
                ledger = measured[key]
                assert ledger["expected_complete_attempts"] == ledger["expected_complete_attempts_uncapped"]
                assert ledger["expected_complete_attempts_uncapped"] == group["expected_complete_trials_uncapped"]
                assert ledger["expected_complete_attempts_stopped_at_cap"] == group["expected_complete_trials_stopped_at_cap"]
                assert "uncapped" in ledger["expected_complete_attempts_scope"]
                assert "including exhausted slots" in ledger["stopped_expectation_scope"]
                assert ledger["worst_case_instruction_counts"] == {gate: count*group["worst_case_complete_trials"] for gate, count in counts.items()}
        assert case["acceptance_projection_certified"] is False
        assert case["sampling_performed"] is False
        assert case["physical_resource_estimate"] is None
    with (tmp_path / "rounds.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row, case in zip(rows, report["cases"], strict=True):
        probability = float(row["acceptance"])
        calls = 2*int(row["rounds"])+1
        assert float(row["expected_source_calls_per_accepted_preparation_uncapped"]) == calls/probability
        assert "uncapped" in row["expected_source_calls_per_accepted_preparation_scope"]
        workload = case["accepted_shot_workload_proposal"]
        stopped = calls*workload["expected_complete_trials_stopped_at_cap"]/workload["total_accepted_shots"]
        assert float(row["expected_source_calls_per_measurement_slot_stopped_at_cap"]) == stopped
        assert "without early abort" in row["stopped_source_call_expectation_scope"]


def test_fresh_constant_energy_has_no_requested_measurement_slot(tmp_path):
    generator = load_generator("generate_simulation_preparation_amplification")
    generator.generate(tmp_path, z=0, x=0, rounds=(0,), betas=(1.25,), scaling=False)
    report = json.loads((tmp_path / "logical-resources.json").read_text())
    workload = report["cases"][0]["accepted_shot_workload_proposal"]
    assert workload["total_accepted_shots"] == 0
    assert workload["expected_complete_trials_stopped_at_cap"] == 0
    with (tmp_path / "rounds.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["expected_source_calls_per_accepted_preparation_uncapped"]) == pytest.approx(1)
    assert row["expected_source_calls_per_measurement_slot_stopped_at_cap"] == ""
