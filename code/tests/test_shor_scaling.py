"""Independent finite Fourier sums, actual circuits, and bounded workload caps."""
import math

import numpy as np
import pytest

from openquantum.shor_scaling import (
    BUILDERS,MatchedBudget,ScalingCase,bounded_ripple_multiplier_check,
    independent_sampling_contract,materialize,tiny_variant_checks,
)


@pytest.mark.parametrize("case,expected_p,expected_trials",[
    (ScalingCase(15,2,8),.5,8),
    (ScalingCase(21,2,6),.28798913703656065,16),
    (ScalingCase(21,2,10),.3308432181936112,14),
    (ScalingCase(35,2,12),.33068609582618,14),
    (ScalingCase(33,5,12),.3982000648755017,11),
])
def test_fixed_cases_have_correct_complete_retry_contract(case,expected_p,expected_trials):
    contract=independent_sampling_contract(case)
    assert contract["ideal_factor_success_probability_per_trial"]==pytest.approx(expected_p,abs=1e-12)
    assert contract["complete_serial_trials"]==expected_trials
    assert contract["ideal_sampling_failure_after_trials"]<=.005
    assert (1-expected_p)**(expected_trials-1)>.005
    assert contract["certified_combined_factor_success"] is None
    assert not contract["reference_used_to_construct_circuit"]
    assert not contract["reference_used_to_choose_precision"]


@pytest.mark.parametrize("case",[ScalingCase(21,2,5),ScalingCase(33,5,6),ScalingCase(15,2,4)])
def test_fft_probabilities_match_independent_explicit_modular_power_pairs(case):
    q=1<<case.phase_bits
    reference=np.zeros(q)
    # Literal exponent-pair interference, independent from FFT/geometric code.
    residues=[pow(case.base,x,case.modulus) for x in range(q)]
    for y in range(q):
        total=0j
        for x in range(q):
            for z in range(q):
                if residues[x]==residues[z]:
                    total+=np.exp(-2j*math.pi*y*(x-z)/q)
        reference[y]=total.real/q**2
    np.testing.assert_allclose(independent_sampling_contract(case)["probabilities"],reference,atol=2e-14,rtol=0)


def test_unfavorable_base_cannot_become_a_finite_retry_factoring_workload():
    result=independent_sampling_contract(ScalingCase(33,2,12))
    assert result["reference_order"]==10
    assert pow(2,5,33)==32
    assert result["ideal_factor_success_probability_per_trial"]==0
    assert result["complete_serial_trials"] is None
    assert not result["finite_retry_projection_supported"]


def test_whole_workload_error_allocations_are_fixed_across_retry_counts():
    budget=MatchedBudget()
    for trials in (8,11,14,16):
        resource=budget.resource_budget(trials)
        assert resource.executions==trials
        assert resource.synthesis_model_error_per_execution*trials==pytest.approx(8e-5)
        assert resource.physical_failure_per_execution*trials==pytest.approx(.005)


def test_tiny_actual_variants_and_coherent_clean_work_match():
    result=tiny_variant_checks()
    assert result["status"]=="checked"
    assert result["variant_probability_difference"]<1e-11
    assert result["ripple"]["qiskit_dense_check_performed"]
    assert result["fourier"]["work_register_zero_probability"]>1-1e-11
    for result in result["coherent_multipliers"].values():
        assert result["max_complex_amplitude_error"]<1e-11


@pytest.mark.parametrize("method",BUILDERS)
def test_materialization_contains_full_preparation_and_final_readout(method):
    case=ScalingCase(3,2,2)
    source,compiled,qasm,record=materialize(case,method)
    assert source.count_ops()["h"]==2 and source.count_ops()["x"]==1
    assert compiled.num_clbits==2 and record["data_readouts_per_trial"]==2
    assert record["readout_mapping"]==[(0,0),(1,1)]
    assert "OPENQASM 3.0" in qasm and "gate " not in qasm
    assert record["primitive_operations_per_trial"]==len(compiled.data)
    assert record["controlled_multiplier_constants"]==[2,1]
    assert record["nonidentity_multipliers"]==1


def test_actual_larger_boolean_multiplier_checks_every_valid_input_branch():
    result=bounded_ripple_multiplier_check(ScalingCase(33,5,12))
    assert result["control_input_cases_per_constant"]==66
    assert result["all_materialized_power_constants_checked"]
    assert set(result["distinct_power_constants"])=={pow(5,1<<j,33) for j in range(12)}
    assert result["all_valid_inputs_tested"]
    assert result["all_tested_work_restored_zero"]


@pytest.mark.parametrize("args",[(65,3,11),(21,2,13),(21,2,True),(21,3,6),(33,5,0)])
def test_bounded_case_validation_rejects_unsupported_requests(args):
    with pytest.raises(ValueError):ScalingCase(*args)


@pytest.mark.parametrize("kwargs",[
    {"ideal_sampling_failure_allowance":0},{"physical_failure_allowance":float("nan")},
    {"synthesis_model_allowance_whole_workload":True},
])
def test_invalid_budget_contracts_are_rejected(kwargs):
    with pytest.raises(ValueError):MatchedBudget(**kwargs)


@pytest.mark.parametrize("trials",[0,65,True,1.5])
def test_repetition_caps_are_enforced_before_native_projection(trials):
    with pytest.raises(ValueError):MatchedBudget().resource_budget(trials)
