# SPDX-License-Identifier: MIT
from copy import deepcopy

import numpy as np
import pytest
from qiskit import QuantumCircuit

from openquantum.instrument_resources import InstrumentProfile, SynthesizedInstrument
from openquantum.qft_qpe import qpe_distribution
from openquantum.qpe_feedback_resources import (PhaseTask, add_feedback_exposure,
    branch_event_counts, build_sources, correction_envelope, decode_median_network,
    measured_distribution, median_controller_decisions, reconstruct_feedback)


@pytest.mark.parametrize("m", [3, 4])
def test_actual_measured_sources_independent_fourier_and_branch_cut(m):
    task = PhaseTask(phase_bits=m)
    sources = build_sources(task)
    reference = qpe_distribution(.3, m)
    for source in sources.values():
        np.testing.assert_allclose(measured_distribution(source), reference, atol=2e-14)
    assert sources["feedback"].count_ops()["reset"] == m+2
    assert sources["textbook"].count_ops()["reset"] == m+1
    assert sources["feedback"].count_ops()["measure"] == m
    assert task.contract()["conditional_failure_upper_estimate"] < .01


@pytest.mark.parametrize("phase,m", [(.0, 3), (.99, 4), (.1, 3)])
def test_scalar_median_branch_cut_rejected(phase, m):
    with pytest.raises(ValueError, match="branch cut"):
        PhaseTask(phase=phase, phase_bits=m)


def toy_feedback():
    source = QuantumCircuit(2, 2)
    source.reset([0, 1])
    source.h(0)
    source.measure(0, 0)
    source.reset(0)
    source.h(0)
    with source.if_test((source.clbits[0], 1)):
        source.z(0)
    source.h(0)
    source.measure(0, 1)
    return source


def test_causal_reconstruction_and_all_transcript_envelope():
    source = toy_feedback()
    envelope, records = correction_envelope(source)
    synthesized = SynthesizedInstrument(envelope, envelope, envelope, {})
    actual, corrections = reconstruct_feedback(synthesized, records)
    np.testing.assert_allclose(measured_distribution(actual), [ .5, 0, 0, .5 ], atol=1e-15)
    np.testing.assert_allclose(measured_distribution(source), measured_distribution(actual), atol=1e-15)
    altered = actual.copy_empty_like()
    for gate in actual.data:
        if gate.operation.name != "if_else":
            altered.append(gate.operation, gate.qubits, gate.clbits)
    assert np.max(np.abs(measured_distribution(altered)-measured_distribution(actual))) > .4
    branches = branch_event_counts(synthesized, corrections, 2)
    assert len(branches) == 4
    assert branches[0]["selected_corrections"] == []
    assert branches[3]["selected_corrections"] == [records[0]["marker"]]
    assert all(b["feedback_control"] == 1 for b in branches)


def test_corrections_cannot_change_layout_or_precede_measurement():
    source = QuantumCircuit(2, 2)
    source.reset([0, 1])
    with source.if_test((source.clbits[0], 1)):
        source.z(0)
    with pytest.raises(ValueError, match="precedes"):
        correction_envelope(source)
    source = toy_feedback()
    envelope, records = correction_envelope(source)
    corrupted = envelope.copy_empty_like()
    for gate in envelope.data:
        corrupted.append(gate.operation, gate.qubits, gate.clbits)
        if gate.operation.name == "barrier" and gate.operation.label.endswith("_begin"):
            corrupted.cx(0, 1)
    with pytest.raises(ValueError, match="layout"):
        reconstruct_feedback(SynthesizedInstrument(envelope, envelope, corrupted, {}), records)


def test_feedback_idle_and_controller_faults_added_under_same_factory():
    profile = InstrumentProfile(maximum_code_distance=7)
    from dataclasses import asdict
    point = {"reserved_logical_patches": 7, "data_code_distance": 7,
        "runtime_per_complete_attempt_ns": 1001, "code_cycles_per_complete_attempt": 2,
        "peak_physical_qubits": 999, "failure_components_all_attempts": {
            "data_workspace_code_exposure": 1e-7, "controller_scenario": 1e-14,
            "factory_output_quality_model": 1e-5},
        "macro_ledger_per_complete_attempt": [],
        "factory_supply": {"pipeline_time_ns": 200, "output_error": 1e-9,
                           "full_pipeline_retry_cap_per_state": 4}}
    original = deepcopy(point)
    resource = {"profile": asdict(profile), "complete_algorithm_attempts": 15,
        "physical_failure_budget": .0025, "all_feasible_points": [point],
        "assumptions": []}
    report = add_feedback_exposure(resource, 3)
    p = report["pareto_points"][0]
    assert p["runtime_per_complete_attempt_ns"] == 31001
    assert p["code_cycles_per_complete_attempt"] == 32
    assert p["factory_supply"] == original["factory_supply"]
    assert p["peak_physical_qubits"] == original["peak_physical_qubits"]
    added = 7*30*profile.patch_failure_per_cycle(7)*15
    assert p["failure_components_all_attempts"]["data_workspace_code_exposure"] == 1e-7+added
    assert p["feedback_adapter"]["added_controller_failure_all_attempts"] == pytest.approx(45e-15)
    assert p["full_physical_failure_model"] == pytest.approx(sum(p["failure_components_all_attempts"].values()))
    assert point == original


def test_feedback_requires_all_candidates_and_refilters_physical_budget():
    with pytest.raises(ValueError, match="complete feasible"):
        add_feedback_exposure({"profile": __import__("dataclasses").asdict(InstrumentProfile()),
                               "complete_algorithm_attempts": 15}, 3)
    profile = InstrumentProfile(maximum_code_distance=7)
    point = {"reserved_logical_patches": 7, "data_code_distance": 7,
        "runtime_per_complete_attempt_ns": 1000, "code_cycles_per_complete_attempt": 1,
        "peak_physical_qubits": 999, "failure_components_all_attempts": {
            "data_workspace_code_exposure": 1e-7, "controller_scenario": 0.,
            "factory_output_quality_model": 1e-5},
        "macro_ledger_per_complete_attempt": [],
        "factory_supply": {"pipeline_time_ns": 200, "output_error": 1e-9,
                           "full_pipeline_retry_cap_per_state": 4}}
    near = deepcopy(point)
    near["failure_components_all_attempts"]["factory_output_quality_model"] = .0024998
    resource = {"profile": __import__("dataclasses").asdict(profile), "complete_algorithm_attempts": 15,
                "physical_failure_budget": .0025, "all_feasible_points": [near, point], "assumptions": []}
    report = add_feedback_exposure(resource, 3)
    assert report["feasible_candidate_count"] == 1
    assert report["pareto_points"][0]["factory_supply"] == point["factory_supply"]


def test_fixed_classical_median_network_and_controller_work():
    values = [7, 0, 1, 6, 3, 5, 2, 4, 7, 0, 6, 1, 5, 2, 4]
    assert decode_median_network(values, 3) == np.median(values)/8
    assert median_controller_decisions(3) == 15*3+105*6+1
    with pytest.raises(ValueError, match="15"):
        decode_median_network(values[:-1], 3)
