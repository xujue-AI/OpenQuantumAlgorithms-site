"""Retargeted thermal spectra, realization checks and complete accepted trials."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator

from openquantum.simulation import thermal_spectral as ts


@pytest.fixture(scope="module")
def checked_cases():
    cases = {}
    for n in (1, 2, 3):
        for beta in (0.0, 0.75, 1.5):
            core = ts.build_thermal_spectral_filter(ts.spin_chain_terms(n), beta)
            cases[n, beta] = core, ts.thermal_spectral_check(core, operator_check=n <= 2)
    return cases


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("beta", [0.0, 0.75, 1.5])
def test_actual_retargered_filter_matches_gibbs_and_declares_certificate_scope(checked_cases, n, beta):
    core, check = checked_cases[n, beta]
    contract = core.metadata["error_contract"]
    exact_alpha = sum((Fraction.from_float(abs(c)) for c, _ in ts.spin_chain_terms(n)), Fraction())
    assert core.metadata["parameter"] == float(Fraction.from_float(beta)*exact_alpha/2)
    assert core.metadata["odd_degree"] == core.metadata["even_degree"]+1
    assert check["operator_validation_performed"] is (n <= 2)
    assert check["gibbs_trace_distance"] < 2e-7
    assert check["trace_real"] == pytest.approx(1, abs=1e-12)
    assert check["minimum_eigenvalue"] >= -1e-12
    assert 0 < check["actual_acceptance_probability"] <= 1
    assert contract["precompiler_source_decoded_total_error_upper"] <= 1e-6
    assert contract["precompiler_source_Gibbs_trace_distance_upper"] <= 1e-4
    assert contract["loading_weight_certificate"]["certified"] is True
    assert "nested" in contract["precompiler_source_scope"]
    assert check["actual_circuit_error_certificate"] is False
    assert check["actual_Bell_branch_vector_error"] <= contract["precompiler_source_encoded_total_error_upper"]+3e-12
    if n > 2:
        assert check["actual_encoded_filter_operator_error"] is None


def test_beta_zero_has_maximally_mixed_output_and_normalized_quarter_filter(checked_cases):
    for n in (1, 2, 3):
        core, check = checked_cases[n, 0.0]
        density = np.array(check["actual_density_real"])+1j*np.array(check["actual_density_imag"])
        np.testing.assert_allclose(density, np.eye(2**n)/(2**n), atol=3e-14, rtol=0)
        assert check["actual_acceptance_probability"] == pytest.approx(1/16, abs=3e-14)
        assert core.metadata["parameter"] == 0


def test_operator_and_signed_branch_checks_inspect_the_supplied_core(checked_cases):
    core = checked_cases[1,.75][0].copy()
    core.global_phase += np.pi
    result = ts.thermal_spectral_check(core)
    assert result["gibbs_trace_distance"] < 1e-7
    assert result["actual_Bell_branch_vector_error"] > .2
    assert result["actual_encoded_filter_operator_error"] > .2


def test_size_and_temperature_actually_retarget_phase_sequences(checked_cases):
    plans = [checked_cases[key][0].metadata["phase_plan"] for key in [(1,.75),(2,.75),(2,1.5),(3,1.5)]]
    assert len({p["parameter"] for p in plans}) == len(plans)
    assert len({tuple(p["even_phases"]) for p in plans}) == len(plans)
    assert all("least_squares" in p["even_fit"]["method"] for p in plans)
    assert all(p["even_fit"]["certified"] is False for p in plans)


@pytest.mark.parametrize("terms", [((.6,"Z"),(-.4,"Y")), ((.6,"ZY"),(-.4,"XI"))])
def test_noncommuting_y_and_negative_coefficients_keep_complex_gibbs_direction(terms):
    core = ts.build_thermal_spectral_filter(terms, 1.25)
    check = ts.thermal_spectral_check(core)
    assert check["gibbs_trace_distance"] < 2e-7
    assert check["actual_encoded_filter_operator_error"] < 1e-7
    assert np.max(np.abs(check["actual_density_imag"])) > .05
    assert "exact" in core.metadata["sign_construction"]


def test_filter_construction_never_aggregates_pauli_powers_or_loads_dense_eigenvectors():
    with patch("openquantum.simulation.lcu.taylor_pauli_terms", side_effect=AssertionError("aggregated")), \
         patch("openquantum.simulation.thermal_filter.thermal_pauli_terms", side_effect=AssertionError("aggregated")), \
         patch.object(ts, "expm", side_effect=AssertionError("dense reference used to build gate")):
        core = ts.build_thermal_spectral_filter(((.6,"Z"),(.4,"X")), 1.25)
    assert not {"unitary", "hamiltonian", "initialize", "state_preparation"} & set(core.count_ops())


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_full_measured_schedule_retains_correct_conditional_mean_and_discards_reference(basis):
    terms = ((.6,"Z"),(.4,"Y"))
    core = ts.build_thermal_spectral_filter(terms, 1.25)
    trial = ts.build_thermal_spectral_trial(core, basis)
    n, success = core.metadata["n_system"], core.metadata["success_qubits"]
    assert trial.count_ops()["reset"] == core.num_qubits+n
    assert trial.count_ops()["measure"] == len(success)+n
    # The final success measurements commute with reference reset and readout
    # basis changes, so defer them to form independent joint probabilities.
    unmeasured = trial.remove_final_measurements(inplace=False)
    actual = DensityMatrix.from_instruction(unmeasured)
    compiled = ts.lower_thermal_spectral(trial).remove_final_measurements(inplace=False)
    np.testing.assert_allclose(DensityMatrix.from_instruction(compiled).data, actual.data, atol=3e-12, rtol=0)
    probabilities = actual.probabilities(qargs=[0]+success)
    p = float(probabilities[:2].sum())
    mean = float((probabilities[0]-probabilities[1])/p)
    check = ts.thermal_spectral_check(core)
    rho = np.array(check["actual_density_real"])+1j*np.array(check["actual_density_imag"])
    pauli = {"X":np.array([[0,1],[1,0]]),"Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}[basis]
    assert p == pytest.approx(check["actual_acceptance_probability"], abs=3e-12)
    assert mean == pytest.approx(np.trace(rho@pauli).real, abs=3e-12)


def test_shift_invariance_and_stable_huge_coefficient_small_beta_reference():
    first = ts.thermal_spectral_check(ts.build_thermal_spectral_filter(((.6,"Z"),(.4,"X")),1.25))
    shifted = ts.thermal_spectral_check(ts.build_thermal_spectral_filter(((.6,"Z"),(.4,"X"),(.2,"I")),1.25))
    assert first["exact_gibbs_energy"]+.2 == pytest.approx(shifted["exact_gibbs_energy"], abs=1e-12)
    np.testing.assert_allclose(first["actual_density_real"],shifted["actual_density_real"],atol=3e-7,rtol=0)
    with np.errstate(over="raise", invalid="raise"):
        huge = ts.thermal_spectral_check(ts.build_thermal_spectral_filter(((1e308,"Z"),),1e-308))
    assert np.isfinite(huge["actual_energy"])
    assert huge["gibbs_trace_distance"] < 1e-7


def test_conditioning_is_outward_and_independent_of_ambient_decimal_context():
    a, delta = Fraction(7,8), Fraction(1,10**8)
    ordinary = ts.ideal_conditioning_bounds(a,delta)
    with localcontext() as ctx:
        ctx.prec, ctx.Emin, ctx.Emax = 6, -2, 2
        altered = ts.ideal_conditioning_bounds(a,delta)
    assert altered == ordinary
    assert Decimal.from_float(ordinary["ideal_target_Bell_branch_amplitude_lower"]) <= Decimal(ordinary["ideal_target_Bell_branch_amplitude_lower_decimal"])
    assert Decimal.from_float(ordinary["ideal_Gibbs_trace_distance_upper"]) >= Decimal(ordinary["ideal_Gibbs_trace_distance_upper_decimal"])
    no_gap = ts.ideal_conditioning_bounds(Fraction(1),Fraction(1))
    assert no_gap["ideal_approximate_filter_acceptance_lower"] == 0


def test_saved_plan_bounds_are_reconstructed_and_forged_requested_allowances_reject():
    plan = ts.generate_thermal_phases(.65)
    forged = replace(plan, ideal_decoded_filter_error_upper=0,
                     even_certificate={"uniform_error_upper":0},odd_certificate={"uniform_error_upper":0})
    audited = ts.audit_thermal_phase_plan(forged)
    assert audited.ideal_decoded_filter_error_upper > 0
    assert audited.even_certificate["certified"] is True
    with pytest.raises(ValueError,match="allowance"):
        ts.audit_thermal_phase_plan(replace(plan,effective_filter_target=.1,requested_filter_error=1e-6))
    with pytest.raises(ValueError,match="looser"):
        ts.build_thermal_spectral_filter(((.6,"Z"),(.4,"X")),1.3,filter_error=1e-7,plan=plan)
    with pytest.raises(ValueError,match="match"):
        ts.build_thermal_spectral_filter(((.6,"Z"),(.4,"X")),1.25,plan=plan)


def test_materialization_and_quantum_simulator_caps_are_separate():
    core = ts.build_thermal_spectral_filter(ts.spin_chain_terms(4), .75)
    assert core.num_qubits == 13
    with pytest.raises(ValueError,match="statevector"):
        ts.thermal_spectral_check(core)
    with pytest.raises(ValueError,match="parameter"):
        ts.build_thermal_spectral_filter(ts.spin_chain_terms(8),100)


def test_correlated_group_energy_budget_and_complete_retries_are_charged():
    workload = ts.energy_workload(.05,2,.6,.35)
    assert workload["groups"]["X"]["accepted_trials"] == 29116
    assert workload["groups"]["Z"]["accepted_trials"] == 21391
    assert workload["total_accepted_trials"] == 50507
    assert workload["complete_attempt_cap"] == 50507*workload["retry_cap_per_slot"]
    assert (1-.05)**workload["retry_cap_per_slot"] <= .005/50507
    assert workload["sampling_performed"] is False
    assert ts.energy_workload(.5,1,0,0)["complete_attempt_cap"] == 0


@pytest.mark.parametrize("kwargs", [{"beta":-1},{"beta":True},{"filter_error":.5},{"filter_error":1e-15},{"gibbs_error":1e-9},{"gibbs_error":True}])
def test_invalid_requested_domains_fail_closed(kwargs):
    with pytest.raises(ValueError):
        ts.build_thermal_spectral_filter(((.6,"Z"),(.4,"X")),**kwargs)


@pytest.mark.parametrize("probability,error", [(0,.03),(1.1,.03),(.1,0),(.1,1e-300)])
def test_invalid_or_unrepresentable_workload_rejects(probability,error):
    with pytest.raises(ValueError):
        ts.energy_workload(probability,statistical_error=error)
