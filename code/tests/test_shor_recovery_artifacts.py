"""Saved-circuit provenance and complete exact integer-recovery replay."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import pytest

from openquantum.shor import reconstruct_order
from openquantum.shor_recovery import RecoveryConfig, baseline_frequency, recover_frequency

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets/generated/shor/recovery"


@pytest.fixture(scope="module")
def study():
    return json.loads((DATA / "results.json").read_text())


def test_exact_executed_scope_source_snapshots_and_license(study):
    assert study["status"] == "completed_bounded_classical_recovery_comparison"
    assert study["config"] == asdict(RecoveryConfig())
    assert study["new_native_calls"] == 0 and study["new_hardware_execution"] is False
    assert study["bounds"]["demo_qubits"] == 18
    assert {(c["modulus"], c["base"], c["phase_bits"]) for c in study["cases"]} == {(15, 2, 8), (21, 2, 6), (33, 2, 4)}
    for relative, digest in study["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for name in ("shor.py", "shor_recovery.py", "generate_shor_recovery.py"):
        relative = "scripts/" + name if name.startswith("generate_") else "src/openquantum/" + name
        assert (DATA / name).read_bytes() == (ROOT / relative).read_bytes()
    assert (DATA / "LICENSE.txt").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert "2020–2026 by Martin Ekerå" in (DATA / "Quaspy-LICENSE.txt").read_text()
    assert "Permission is hereby granted" in (DATA / "Quaspy-LICENSE.txt").read_text()
    assert study["upstream_reference"]["quaspy_commit"] == "8040affdf08814ef7d451d9797d28f7645856151"


def test_qasm_primitive_counts_readouts_and_saved_circuit_check(study):
    gate_pattern = re.compile(r"^\s*([a-z][a-z0-9]*)\s*(?:\([^;]*\))?\s+q\[", re.M)
    measure_pattern = re.compile(r"(?:[A-Za-z_]\w*\[(\d+)\])\s*=\s*measure\s+q\[(\d+)\];")
    for case in study["cases"]:
        qasm_bytes = (DATA / case["qasm_file"]).read_bytes()
        assert hashlib.sha256(qasm_bytes).hexdigest() == case["qasm_sha256"]
        qasm = qasm_bytes.decode()
        counts = {}
        for gate in gate_pattern.findall(qasm):
            counts[gate] = counts.get(gate, 0) + 1
        mapping = [(int(q), int(c)) for c, q in measure_pattern.findall(qasm)]
        counts["measure"] = len(mapping)
        assert counts == case["primitive_gate_counts_per_trial"]
        assert mapping == [(i, i) for i in range(case["phase_bits"])]
        assert sum(counts.values()) == case["primitive_operations_per_trial"]
        assert case["logical_qubits"] <= 18
        assert case["simulation"]["performed"] is True
        assert case["simulation"]["max_probability_error"] <= 1e-10
        assert abs(case["simulation"]["work_register_zero_probability"] - 1) <= 1e-10
        assert case["phase_bits_selected_using_reference_order"] is False
        assert case["physical_estimation"] == {"status": "not_performed", "native_calls": 0}


def test_every_saved_frequency_replays_complete_integer_pipeline(study):
    config = RecoveryConfig(**study["config"])
    rows_checked = 0
    for case in study["cases"]:
        N, a, t = case["modulus"], case["base"], case["phase_bits"]
        rows = case["all_frequency_rows"]
        assert [row["outcome"] for row in rows] == list(range(1 << t))
        assert case["all_outcomes_retained"] is True
        for row in rows:
            old = baseline_frequency(N, a, row["outcome"], t, config)
            new = recover_frequency(N, a, row["outcome"], t, config)
            assert old["period"] == reconstruct_order(row["outcome"], t, N, a)
            for method, result in (("baseline", old), ("recovery", new)):
                saved = {k: v for k, v in row[method].items() if k != "wall_ns"}
                replay = {k: v for k, v in result.items() if k != "wall_ns"}
                assert saved == replay
                assert result["order_reference_used"] is False
                assert result["input_modulus_factorization_used"] is False
            assert row["recovery"]["certified_order"] == case["simulation"]["classical_reference_order"]
            assert row["recovery"]["smooth_only_pretest"] is True
            rows_checked += 1
    assert rows_checked == 336


def test_probability_weights_and_complete_same_sample_workload(study):
    for case in study["cases"]:
        rows = case["all_frequency_rows"]
        raw = case["simulation"]["probabilities"]
        normalization = case["simulation"]["probability_normalization_for_classical_draws"]
        # Independent compensated sum; <=256 positive fixed-cohort entries.
        # The 1e-13 bound covers IEEE summation order, without repricing data.
        assert abs(normalization - math.fsum(raw)) <= 1e-13
        assert all(math.isfinite(row["probability"]) and row["probability"] >= 0 for row in rows)
        assert abs(sum(row["probability"] for row in rows) - 1) < 1e-12
        for row, weight in zip(rows, raw):
            assert row["probability"] == weight / normalization
        for method in ("baseline", "recovery"):
            for key, value in case["probability_weighted_classical_work"][method].items():
                assert value == sum(row["probability"] * row[method]["work"][key] for row in rows)
        samples = case["samples"]
        assert samples["same_draws_for_both_methods"] is True
        assert len(samples["records"]) == samples["trial_count"] == 16
        for sample in samples["records"]:
            row = rows[sample["outcome"]]
            assert sample["baseline_period"] == row["baseline"]["period"]
            assert sample["recovery_period"] == row["recovery"]["period"]
            assert sample["baseline_factors"] == row["baseline"]["factors"]
            assert sample["recovery_factors"] == row["recovery"]["factors"]
        assert case["complete_quantum_work_for_fixed_samples"] == {
            "trials": 16, "primitive_operations": 16 * case["primitive_operations_per_trial"],
            "phase_readouts": 16 * case["phase_bits"], "peak_logical_qubits": case["logical_qubits"],
            "unchanged_between_postprocessors": True}


def test_negative_factor_case_and_no_global_success_or_physical_claim(study):
    negative = next(case for case in study["cases"] if case["modulus"] == 33)
    assert negative["actual_order_squared_below_grid_diagnostic"] is False
    assert negative["all_frequency_recovery_uses_smooth_only_pretest"] is True
    assert all(row["recovery"]["factors"] == [] for row in negative["all_frequency_rows"])
    assert negative["weighted_success"]["baseline_verified_factor"] == 0
    assert negative["weighted_success"]["recovery_verified_factor"] == 0
    assert "not quantum advantage" in study["classical_only_smooth_case_warning"]
    assert "no reproduced proof" in study["formalization_status"]
    assert "not every Python multiplication" in study["work_counter_scope"]


def test_generator_output_guards_preserve_other_algorithms(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("recovery_generator_test", ROOT / "scripts/generate_shor_recovery.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="another algorithm"):
        module.output_directory(ROOT / "assets/generated/shor/n21")
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        module.output_directory(linked / "output")
    with pytest.raises(ValueError, match="separate output"):
        module.generate(cases=((15, 2, 4),), simulate=True)
