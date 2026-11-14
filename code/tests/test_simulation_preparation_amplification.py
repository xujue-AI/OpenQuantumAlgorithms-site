"""Known-input thermal amplification, reflections, phases and full measured trials."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector, DensityMatrix

from openquantum.simulation import thermal_filter as tf
from openquantum.simulation import preparation_amplification as pa


def test_good_and_initial_reflections_have_different_projectors():
    full = pa.zero_subregister_reflection(4, range(4))
    good = pa.zero_subregister_reflection(4, [2, 3])
    np.testing.assert_allclose(Operator(full).data, np.diag([-1]+[1]*15), atol=1e-14)
    np.testing.assert_allclose(Operator(good).data, np.diag([-1]*4+[1]*12), atol=1e-14)
    np.testing.assert_allclose(Operator(pa.zero_subregister_reflection(2, [])).data, -np.eye(4), atol=1e-14)
    assert all(good.find_bit(q).index >= 2 for i in good.data for q in i.qubits)


@pytest.mark.parametrize("beta", [0, 1.25, 2])
@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 4])
def test_actual_filter_preserves_signed_purification_and_density(beta, rounds):
    result = pa.preparation_amplification_check(beta=beta, rounds=rounds)
    assert result["raw_signed_accepted_vector_error"] < 5e-13
    assert result["phase_invariant_purification_error"] < 5e-13
    assert result["signed_normalized_purification_error"] < 5e-13
    assert result["reduced_state_trace_distance_from_baseline"] < 5e-13
    assert result["reduced_state_trace_distance_from_polynomial"] < 5e-13
    assert result["actual_acceptance_probability"] == pytest.approx(result["model_success_probability_from_actual_baseline_p"], abs=5e-13)
    assert result["gibbs_trace_distance"] == pytest.approx(result["baseline_gibbs_trace_distance"], abs=5e-13)
    assert result["total_source_calls"] == 2*rounds+1


def test_overrotation_changes_global_sign_but_does_not_reduce_polynomial_bias():
    r0 = pa.preparation_amplification_check(beta=2, rounds=0)
    r3 = pa.preparation_amplification_check(beta=2, rounds=3)
    r4 = pa.preparation_amplification_check(beta=2, rounds=4)
    assert r3["actual_acceptance_probability"] > .99
    assert r4["actual_acceptance_probability"] < .02
    assert r3["expected_global_sign"] == -1
    assert r3["gibbs_trace_distance"] == pytest.approx(r0["gibbs_trace_distance"], abs=5e-13)


@pytest.mark.parametrize("terms", [((.6,"Z"),(.4,"Y")), ((.6,"ZY"),(.4,"XI"))])
def test_y_models_and_full_unitary_primitive_lowering(terms):
    source = pa.build_preparation_amplification(terms, beta=1.25, degree=4, rounds=2)
    assert source.num_qubits <= pa.MAX_OPERATOR_QUBITS
    compiled = pa.decompose_preparation_amplification(source)
    np.testing.assert_allclose(Operator(compiled).data, Operator(source).data, atol=5e-12, rtol=0)
    result = pa.preparation_amplification_check(terms, beta=1.25, degree=4, rounds=2)
    assert result["reduced_state_trace_distance_from_baseline"] < 1e-12
    assert not {"unitary", "hamiltonian", "initialize", "state_preparation"} & set(compiled.count_ops())


@pytest.mark.parametrize("measurement", ["X", "Y", "Z"])
def test_full_trial_readout_and_reference_reset_matches_conditional_state(measurement):
    terms = ((.6,"Z"),(.4,"Y"))
    trial = pa.build_preparation_trial(terms, beta=1.25, degree=6, rounds=3, measurement=measurement)
    compiled = pa.decompose_preparation_amplification(trial)
    assert trial.count_ops()["reset"] == 5
    assert trial.count_ops()["measure"] == 3
    # Label measurements are final on their wires and commute with the later
    # reference reset/system basis change, so defer readout to get joint probs.
    source_density = DensityMatrix.from_instruction(trial.remove_final_measurements(inplace=False))
    compiled_density = DensityMatrix.from_instruction(compiled.remove_final_measurements(inplace=False))
    np.testing.assert_allclose(source_density.data, compiled_density.data, atol=5e-12, rtol=0)
    joint = source_density.probabilities(qargs=[0,2,3])
    p = float(joint[:2].sum())
    actual_mean = float((joint[0]-joint[1])/p)
    check = pa.preparation_amplification_check(terms, beta=1.25, degree=6, rounds=3)
    rho = np.array(check["actual_density_real"])+1j*np.array(check["actual_density_imag"])
    local = {"X":np.array([[0,1],[1,0]]),"Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}
    assert actual_mean == pytest.approx(np.trace(rho@local[measurement]).real, abs=5e-12)
    assert p == pytest.approx(check["actual_acceptance_probability"], abs=5e-12)


def test_known_input_not_oblivious_nonunitary_filter():
    correct = pa.preparation_amplification_check(rounds=1)
    wrong = pa.ancilla_only_distortion_check(rounds=1)
    assert correct["reduced_state_trace_distance_from_baseline"] < 1e-12
    assert wrong["trace_distance_from_original_thermal_polynomial_state"] > .1


def test_exact_quarter_probability_generic_preparation_and_no_probability_encoded_gate():
    source = QuantumCircuit(2)
    source.h(0)
    source.ry(2*np.arccos(.5),1)
    amplified = pa.amplify_known_input(source,[1],1)
    state = Statevector.from_instruction(amplified).data
    assert np.linalg.norm(state[:2])**2 == pytest.approx(1,abs=1e-14)
    np.testing.assert_allclose(state[:2],np.array([1,1])/np.sqrt(2),atol=1e-14)
    assert amplified.metadata["full_initial_reflection_calls"] == 1
    assert amplified.metadata["good_reflection_calls"] == 1
    assert amplified.metadata["forward_A_calls"] == 2
    assert amplified.metadata["inverse_A_calls"] == 1


def test_source_bounds_finiteness_and_opacity():
    for angle in [Parameter("theta"),float("nan"),float("inf")]:
        source = QuantumCircuit(2)
        source.rx(angle,0)
        with pytest.raises(ValueError,match="bound|finite"):
            pa.amplify_known_input(source,[1],1)
    source = QuantumCircuit(2)
    source.global_phase = float("nan")
    with pytest.raises(ValueError,match="phase"):
        pa.amplify_known_input(source,[1],1)
    source = QuantumCircuit(2)
    source.append(UnitaryGate(np.eye(4)),[0,1])
    with pytest.raises(ValueError,match="opaque"):
        pa.amplify_known_input(source,[1],1)
    with pytest.raises(ValueError,match="rounds"):
        pa.build_preparation_amplification(rounds=5)
    with pytest.raises(ValueError,match="distinct"):
        pa.zero_subregister_reflection(4,[2,2])
    with pytest.raises(ValueError,match="source width"):
        pa.amplify_known_input(QuantumCircuit(25),[1],1)


def test_generation_and_statevector_limits_separate():
    source = QuantumCircuit(13)
    source.metadata = {"n_system":1,"n_reference":1,"n_index":11}
    core = pa.amplify_known_input(source,range(2,13),0)
    with pytest.raises(ValueError,match="statevector"):
        pa.accepted_preparation_branch(core)
    assert core.num_qubits == 13


def test_hypothetical_conditioning_counts_all_calls_and_stays_separate_from_faults():
    proposal = pa.synthesis_conditioning_proposal(.5,rounds=3)
    assert proposal["amplified_coherent_operator_error_allowance"] == pytest.approx(13e-5)
    assert proposal["perturbed_acceptance_lower_bound"] < .5
    assert "hypothetical" in proposal["status"]
