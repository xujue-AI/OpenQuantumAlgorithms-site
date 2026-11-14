# SPDX-License-Identifier: MIT
from dataclasses import replace
from fractions import Fraction
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from openquantum.approximate_qft_resources import (
    QPEVariantContract, build_contract_trial, build_variant_qpe, checked_distribution,
    fixed_grid, majority_failure_upper, median_trial_count, omission_tv_upper,
)
from openquantum.qft_qpe import build_qpe,phase_gate_example,qpe_distribution,qft_omission_bound,lower_circuit


def independent_inverse_qft_on_phase_state(m,cutoff,phase):
    """Direct amplitude updates for inverse phases, H and swaps, without Qiskit."""
    size=2**m
    state=np.exp(2j*math.pi*phase*np.arange(size))/math.sqrt(size)
    # Inverse of the explicitly specified positive-sign QFT gate order.
    for low in reversed(range(m//2)):
        high=m-low-1
        new=state.copy()
        for index in range(size):
            bits=((index>>low)&1)^((index>>high)&1)
            new[index^(bits<<low)^(bits<<high)]=state[index]
        state=new
    for target in range(m):
        for control in range(target):
            distance=target-control
            if distance<=cutoff:
                mask=[index for index in range(size) if (index>>target)&1 and (index>>control)&1]
                state[mask]*=np.exp(-1j*math.pi/2**distance)
        for index in range(size):
            if not (index>>target)&1:
                other=index^(1<<target)
                first,second=state[index],state[other]
                state[index]=(first+second)/math.sqrt(2)
                state[other]=(first-second)/math.sqrt(2)
    return np.abs(state)**2


@pytest.mark.parametrize("m",range(1,6))
def test_exact_variant_matches_existing_qpe_operator(m):
    unitary,preparation=phase_gate_example(.31)
    unitary.global_phase=.17
    source=build_variant_qpe(m,unitary,preparation,m-1)
    exact=build_qpe(m,unitary,preparation)
    np.testing.assert_allclose(Operator(source).data,Operator(exact).data,atol=3e-13,rtol=0)


@pytest.mark.parametrize("m,cutoff,phase",[(m,b,p) for m in range(2,6) for b in range(m) for p in (.15625,.3)])
def test_truncated_actual_qpe_against_independent_amplitude_updates(m,cutoff,phase):
    unitary,preparation=phase_gate_example(phase)
    source=build_variant_qpe(m,unitary,preparation,cutoff)
    actual=checked_distribution(source,m)
    expected=independent_inverse_qft_on_phase_state(m,cutoff,phase)
    np.testing.assert_allclose(actual,expected,atol=3e-13,rtol=0)
    exact=qpe_distribution(phase,m)
    tv=.5*float(np.abs(actual-exact).sum())
    assert tv<=float(omission_tv_upper(m,cutoff))+3e-13


def test_grid_contract_holds_all_matched_parameters_fixed():
    grid=fixed_grid()
    assert len(grid)==12
    for contract in grid:
        exact=replace(contract,max_distance=contract.phase_bits-1)
        assert exact.phase==contract.phase and exact.target_qubits==contract.target_qubits
        assert exact.phase_error_target==contract.phase_error_target
        assert exact.physical_failure_allowance==contract.physical_failure_allowance
        assert exact.total_failure_target==contract.total_failure_target
        assert exact.synthesis_model_allowance_per_trial==contract.synthesis_model_allowance_per_trial
        assert exact.repetitions==15
        if contract.max_distance==contract.phase_bits-2:
            assert contract.repetitions=={5:127,6:35,7:21}[contract.phase_bits]
        metadata=contract.as_metadata()
        assert metadata["configured_conditional_full_failure_upper"]<.01
        assert metadata["certified_hardware_confidence"] is None


@pytest.mark.parametrize("m,cutoff",[(m,b) for m in range(1,9) for b in range(m)])
def test_rational_omission_conversion_bounds_existing_angle_sum(m,cutoff):
    assert float(omission_tv_upper(m,cutoff))>=min(1,qft_omission_bound(m,cutoff))-1e-15


def test_exact_majority_tail_against_independent_binary_outcomes():
    from itertools import product
    q=Fraction(1,4)
    brute=sum((q**sum(outcomes)*(1-q)**(3-sum(outcomes))
               for outcomes in product((0,1),repeat=3) if sum(outcomes)>=2),Fraction(0))
    assert majority_failure_upper(q,3)==brute==Fraction(5,32)
    assert median_trial_count(q,Fraction(1,6))==3


def test_selected_repetition_counts_are_minimal_exact_rational():
    for contract in fixed_grid():
        q=contract.single_trial_failure_upper
        assert majority_failure_upper(q,contract.repetitions)<=contract.statistical_failure_allowance
        if contract.repetitions>1:
            assert majority_failure_upper(q,contract.repetitions-2)>contract.statistical_failure_allowance
        assert contract.resource_budget.executions==contract.repetitions
        assert contract.resource_budget.physical_failure_per_execution==contract.physical_failure_allowance/contract.repetitions


def test_invalid_contract_does_not_silently_weaken_output():
    with pytest.raises(ValueError,match="branch cut"):
        QPEVariantContract(5,4,phase=.001)
    with pytest.raises(ValueError,match="at least one half"):
        QPEVariantContract(5,1)
    with pytest.raises(ValueError,match="repetition cap"):
        median_trial_count(Fraction(2,5),Fraction(1,100000),max_repetitions=3)
    with pytest.raises(ValueError):QPEVariantContract(5,4,phase=True)
    with pytest.raises(ValueError):QPEVariantContract(5,4,synthesis_model_allowance_per_trial=0)
    with pytest.raises(ValueError):QPEVariantContract(5,4,statistical_failure_allowance=.005)
    with pytest.raises(ValueError):QPEVariantContract(5,4,physical_failure_allowance=.05)
    with pytest.raises(ValueError):majority_failure_upper(Fraction(1,4),2)


def test_complete_trial_access_readout_and_lowering():
    contract=QPEVariantContract(5,3)
    source=build_contract_trial(contract,measure=True)
    compiled=lower_circuit(source)
    assert source.metadata["controlled_unitary_calls"]==31
    assert source.metadata["input_preparation_calls"]==1
    assert source.metadata["retained_inverse_qft_controlled_phases"]==9
    assert source.metadata["omitted_inverse_qft_controlled_phases"]==1
    assert compiled.count_ops()["measure"]==5
    actual=checked_distribution(source.remove_final_measurements(inplace=False),5)
    lowered=checked_distribution(compiled.remove_final_measurements(inplace=False),5)
    np.testing.assert_allclose(actual,lowered,atol=3e-13,rtol=0)
    assert not source.metadata["physical_initialization_cost_included"]


def test_general_eigenstate_preparation_and_mixture_keep_actual_access():
    unitary=QuantumCircuit(1)
    unitary.ry(-4*math.pi*.3,0)
    preparation=QuantumCircuit(1)
    preparation.h(0);preparation.s(0)
    actual=checked_distribution(build_variant_qpe(4,unitary,preparation,2),4)
    np.testing.assert_allclose(actual,independent_inverse_qft_on_phase_state(4,2,.3),atol=3e-13,rtol=0)
    unitary,preparation=phase_gate_example(.3)
    preparation=QuantumCircuit(1);preparation.h(0)
    actual=checked_distribution(build_variant_qpe(4,unitary,preparation,2),4)
    expected=.5*independent_inverse_qft_on_phase_state(4,2,0)+.5*independent_inverse_qft_on_phase_state(4,2,.3)
    np.testing.assert_allclose(actual,expected,atol=3e-13,rtol=0)


def test_generator_and_simulator_caps_and_dynamic_input_rejection():
    with pytest.raises(ValueError):build_contract_trial(QPEVariantContract(9,8))
    contract=QPEVariantContract(8,7,target_qubits=3)
    generated=build_contract_trial(contract)
    assert generated.num_qubits==11
    with pytest.raises(ValueError,match="ten-wire cap"):checked_distribution(generated,8)
    unitary,preparation=phase_gate_example(.3)
    unitary.reset(0)
    with pytest.raises(ValueError):build_variant_qpe(3,unitary,preparation,2)


def test_large_exact_rational_metadata_has_bounded_safe_encoding():
    from openquantum.approximate_qft_resources import _fraction_text
    value=Fraction(1,2**15000+1)
    record=_fraction_text(value)
    assert record["encoding"]=="exact_hex_rational"
    assert Fraction(int(record["numerator_hex"],16),int(record["denominator_hex"],16))==value


def test_source_preflight_rejects_repeated_expansion_before_constructing_qpe():
    unitary=QuantumCircuit(1)
    for _ in range(100):unitary.h(0)
    with pytest.raises(ValueError,match="source-operation cap"):
        build_variant_qpe(8,unitary,QuantumCircuit(1),7)


def test_generator_removes_optional_physical_outputs_and_marks_unexecuted_simulation(tmp_path):
    import json
    from pathlib import Path
    import subprocess,sys
    for name in ("physical-pareto.csv","matched-comparison.csv","physical-comparison.svg"):
        (tmp_path/name).write_text("old evidence")
    old_case=tmp_path/"cases"/"older"
    old_case.mkdir(parents=True)
    (old_case/"surface-code.json").write_text("old physical evidence")
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/"scripts/generate_qft_variant_comparison.py"),
                    "--no-sim","--output-dir",str(tmp_path)],check=True,capture_output=True,text=True)
    for name in ("physical-pareto.csv","matched-comparison.csv","physical-comparison.svg"):
        assert not (tmp_path/name).exists()
    assert not old_case.exists()
    cases=list((tmp_path/"cases").glob("*/logical.json"))
    assert len(cases)==12
    assert all(not json.loads(path.read_text())["simulation"]["performed"] for path in cases)
