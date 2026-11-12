"""Check that circuit repetitions and physical fault budgets share one target."""

import importlib.util
import math
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "resource_script", Path(__file__).resolve().parents[1] / "scripts/estimate_resources.py"
)
resource_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_script)


def test_three_qubit_budget_includes_both_sampling_and_physical_failure():
    budget = resource_script.repetition_budget(3, 2)
    assert budget["ideal_success_probability_per_trial"] == pytest.approx(121 / 128)
    assert budget["serial_trials"] == 2
    assert budget["max_physical_error_per_trial"] == pytest.approx(0.0025)
    assert budget["configured_full_failure_union_bound"] == pytest.approx((7 / 128) ** 2 + 0.005)
    assert budget["configured_full_failure_union_bound"] <= budget["total_failure_target"]


@pytest.mark.parametrize("n_qubits", range(2, 13))
def test_repetition_count_is_minimal_under_sampling_allowance(n_qubits):
    theta = math.asin(2 ** (-n_qubits / 2))
    iterations = max(0, math.floor(math.pi / (4 * theta)))
    budget = resource_script.repetition_budget(n_qubits, iterations)
    failure = 1 - budget["ideal_success_probability_per_trial"]
    trials = budget["serial_trials"]
    allowance = budget["sampling_failure_allowance"]
    assert failure**trials <= allowance + 1e-14
    if trials > 1:
        assert failure ** (trials - 1) > allowance
    assert budget["max_physical_error_per_trial"] * trials == pytest.approx(
        budget["physical_failure_allowance"]
    )


def test_resource_input_rejects_opaque_or_unmeasured_circuits():
    valid = 'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[3] c; c = measure q;'
    resource_script.validate_qasm(valid, 3)
    with pytest.raises(ValueError, match="custom gates"):
        resource_script.validate_qasm(valid + " gate unresolved a { x a; }", 3)
    with pytest.raises(ValueError, match="measurements"):
        resource_script.validate_qasm('OPENQASM 3.0; qubit[3] q; h q[0];', 3)
    with pytest.raises(ValueError, match="qubit count"):
        resource_script.validate_qasm(valid, 4)
