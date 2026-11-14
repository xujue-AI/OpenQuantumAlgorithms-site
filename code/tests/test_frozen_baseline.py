import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frozen_baseline", ROOT / "scripts/verify_frozen_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def baseline():
    return json.loads((ROOT / "assets/generated/resources/grover-surface-code.json").read_text())


def test_only_inactive_search_and_cpu_timing_are_ignored_without_mutation():
    frozen = baseline()
    original = copy.deepcopy(frozen)
    fresh = copy.deepcopy(frozen)
    for profile in fresh["profiles"]:
        for point in profile["pareto_points"]:
            point["properties"]["EVALUATION_TIME"] += 12345
            point["properties"]["NUM_TS_PER_ROTATION"] += 7
    module.compare_records(frozen, fresh)
    assert frozen == original


@pytest.mark.parametrize("field", ["physical_qubits", "runtime_all_trials_ns", "full_failure_union_bound"])
def test_changed_physical_quantity_is_rejected(field):
    frozen = baseline()
    fresh = copy.deepcopy(frozen)
    fresh["profiles"][0]["pareto_points"][0][field] *= 1.01
    with pytest.raises(ValueError, match="physical results differ"):
        module.compare_records(frozen, fresh)


def test_changed_contract_and_rotated_trace_are_rejected():
    frozen = baseline()
    fresh = copy.deepcopy(frozen)
    fresh["error_budget"]["total_failure_target"] = 0.02
    with pytest.raises(ValueError, match="physical results differ"):
        module.compare_records(frozen, fresh)
    fresh = copy.deepcopy(frozen)
    fresh["arbitrary_rotation_synthesis_required"] = True
    with pytest.raises(ValueError, match="rotation-free"):
        module.compare_records(frozen, fresh)
