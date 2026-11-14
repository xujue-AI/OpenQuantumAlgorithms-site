"""Bounded opt-in joint instrument checks against independent density evolution."""
from dataclasses import asdict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, partial_trace

from openquantum.instrument_resources import (
    MAX_QUBITS, MAX_SOURCE_OPERATIONS, MAX_SYNTHESIZED_OPERATIONS,
    InstrumentLimits, InstrumentProfile, SynthesizedInstrument, estimate_instrument,
    simulate_scored_trial, simulate_trial, synthesize_instrument,
)


WIDE = InstrumentLimits(max_qubits=8, source_operations=20_000,
                        synthesized_operations=500_000, validation_operations=500_000)


def _density_reference(circuit, outcome_clbits, scores, acceptance):
    """Independent full density/Kraus reference for short six/eight-wire circuits."""
    rho = DensityMatrix.from_int(0, 2**circuit.num_qubits)
    reset = Kraus([np.array([[1, 0], [0, 0]], complex),
                   np.array([[0, 1], [0, 0]], complex)])
    projections = [np.diag([1., 0.]).astype(complex), np.diag([0., 1.]).astype(complex)]
    mapping = {circuit.find_bit(i.clbits[0]).index: circuit.find_bit(i.qubits[0]).index
               for i in circuit.data if i.operation.name == "measure"}
    out_wires = tuple(mapping[c] for c in outcome_clbits)
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if name == "barrier":
            continue
        if name == "reset":
            rho = rho.evolve(reset, wires)
        elif name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in acceptance:
                rho = rho.evolve(Operator(projections[acceptance[bit]]), wires)
            elif bit in outcome_clbits:
                probability = float(np.trace(rho.data).real)
                probabilities = rho.probabilities(qargs=out_wires)/probability
                reduced = partial_trace(rho, [q for q in range(circuit.num_qubits) if q not in out_wires]).data/probability
                canonical = sorted(out_wires)
                permutation = [sum(((state >> k) & 1) << canonical.index(q)
                                   for k, q in enumerate(out_wires))
                               for state in range(2**len(out_wires))]
                reduced = reduced[np.ix_(permutation, permutation)]
                return probability, float(np.asarray(scores) @ probabilities), probabilities, reduced
            else:
                rho = rho.evolve(Kraus(projections), wires)
        else:
            rho = rho.evolve(item.operation, wires)
    raise AssertionError("reference did not encounter output")


def _short_wide_trial(width):
    c = QuantumCircuit(width, 6)
    c.reset(range(width))
    c.ry(.6, 4)
    c.ry(.37, 1)
    c.cx(4, 1)
    c.cx(1, width-1)
    c.reset(width-1)  # Discard an environment entangled with both outputs.
    c.h(2)
    c.cz(2, 4)
    c.measure(2, 5)  # Ignore both outcomes, rather than accepting its zero.
    c.ry(.7, 3)
    c.measure(3, 4)
    c.rx(.27, 4)
    c.measure(4, 2)
    c.measure(1, 0)
    c.measure(0, 1)
    return c


def test_original_width_and_operation_defaults_are_preserved():
    assert (MAX_QUBITS, MAX_SOURCE_OPERATIONS, MAX_SYNTHESIZED_OPERATIONS) == (5, 10_000, 100_000)
    assert asdict(InstrumentLimits()) == dict(max_qubits=5, source_operations=10_000,
                                           synthesized_operations=100_000, validation_operations=100_000,
                                           validation_branches=256)


@pytest.mark.parametrize("width", [6, 8])
@pytest.mark.parametrize("outputs", [(2, 0), (0, 2)])
def test_wide_joint_scores_reset_and_ignored_measurements_match_independent_density(width, outputs):
    circuit = _short_wide_trial(width)
    scores = (.95, .25, -.25, -.95)
    expected_p, expected_mean, expected_joint, expected_rho = _density_reference(circuit, outputs, scores, {4: 1})
    actual = simulate_scored_trial(circuit, outputs, scores, {4: 1}, limits=WIDE)
    assert actual["acceptance_probability"] == pytest.approx(expected_p, abs=2e-14)
    assert actual["mean"] == pytest.approx(expected_mean, abs=2e-14)
    np.testing.assert_allclose(actual["joint_probabilities"], expected_joint, atol=2e-14)
    np.testing.assert_allclose(actual["reduced_before_output"], expected_rho, atol=2e-14)
    assert actual["simulation"]["peak_branches"] <= 4
    assert np.trace(actual["reduced_before_output"]).real == pytest.approx(1)


def test_asymmetric_joint_scores_detect_bit_order_instead_of_assuming_independent_spins():
    circuit = _short_wide_trial(6)
    scores = (.95, .25, -.25, -.95)
    forward = simulate_scored_trial(circuit, (2, 0), scores, {4: 1}, limits=WIDE)
    reverse = simulate_scored_trial(circuit, (0, 2), scores, {4: 1}, limits=WIDE)
    assert abs(forward["mean"]-reverse["mean"]) > .001
    assert forward["outcome_wires"] == [4, 1]
    np.testing.assert_allclose(reverse["joint_probabilities"], forward["joint_probabilities"][[0, 2, 1, 3]], atol=1e-14)


def test_correlated_thermal_scores_are_joint_outcomes_not_independent_bit_marginals():
    c = QuantumCircuit(6, 2)
    c.reset(range(6))
    c.h(0)
    c.cx(0, 1)
    c.measure(0, 0)
    c.measure(1, 1)
    zz = simulate_scored_trial(c, (0, 1), (.6, -.6, -.6, .6), limits=WIDE)
    assert zz["mean"] == pytest.approx(.6, abs=1e-14)
    np.testing.assert_allclose(zz["joint_probabilities"], [.5, 0, 0, .5], atol=1e-14)


def test_opt_in_synthesis_preserves_complete_wide_line_routing_and_readout():
    c = QuantumCircuit(6, 6)
    c.reset(range(6))
    c.h(0)
    c.cx(0, 5)
    c.measure(range(6), range(6))
    with pytest.raises(ValueError, match="width/operation"):
        synthesize_instrument(c)
    with pytest.raises(ValueError, match="width/operation"):
        simulate_trial(c, 0)
    compiled = synthesize_instrument(c, limits=WIDE, preserve_full_unitary=True)
    assert compiled.report["limits"] == asdict(WIDE)
    assert compiled.report["transpiler_qubits_initially_zero"] is False
    assert compiled.report["preserve_full_unitary"] is True
    assert compiled.circuit.count_ops()["reset"] == 6
    assert compiled.circuit.count_ops()["measure"] == 6
    ideal = simulate_scored_trial(c, (0, 5), (1, -1, -1, 1), limits=WIDE)
    actual = simulate_scored_trial(compiled.circuit, (0, 5), (1, -1, -1, 1), limits=WIDE)
    np.testing.assert_allclose(actual["joint_probabilities"], ideal["joint_probabilities"], atol=1e-14)
    assert actual["mean"] == pytest.approx(1)
    for item in compiled.circuit.data:
        if item.operation.name == "cx":
            a, b = [compiled.circuit.find_bit(q).index for q in item.qubits]
            assert abs(a-b) == 1


def test_opt_in_single_mean_wrapper_matches_original_density_path():
    c = QuantumCircuit(3, 3)
    c.reset(range(3))
    c.h(0)
    c.cx(0, 1)
    c.reset(1)
    c.ry(.9, 2)
    c.measure(2, 2)
    c.h(0)
    c.measure(0, 0)
    c.measure(1, 1)
    density = simulate_trial(c, 0, {2: 1})
    ensemble = simulate_trial(c, 0, {2: 1}, limits=WIDE)
    assert ensemble["acceptance_probability"] == pytest.approx(density["acceptance_probability"], abs=1e-14)
    assert ensemble["mean"] == pytest.approx(density["mean"], abs=1e-14)
    np.testing.assert_allclose(ensemble["reduced_before_output"], density["reduced_before_output"], atol=1e-14)


@pytest.mark.parametrize("kwargs", [{"max_qubits": 9}, {"max_qubits": True},
                                    {"source_operations": 0}, {"synthesized_operations": 1_000_001},
                                    {"validation_branches": 257}])
def test_explicit_limit_ceiling_rejects_unsupported_domains(kwargs):
    with pytest.raises(ValueError):
        InstrumentLimits(**kwargs)


def test_validation_and_synthesis_operation_caps_are_independent():
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.h(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="operation bound"):
        simulate_scored_trial(c, (0,), (1, -1), limits=InstrumentLimits(validation_operations=2))
    with pytest.raises(ValueError, match="operation bound"):
        synthesize_instrument(c, limits=InstrumentLimits(source_operations=2))
    with pytest.raises(ValueError, match="Routed instrument"):
        synthesize_instrument(c, limits=InstrumentLimits(synthesized_operations=2))
    default = synthesize_instrument(c)
    assert default.report["transpiler_qubits_initially_zero"] is True
    with pytest.raises(ValueError, match="must be Boolean"):
        synthesize_instrument(c, preserve_full_unitary=1)


def test_no_nonzero_kraus_branch_is_pruned_to_fit_the_ensemble_cap():
    c = QuantumCircuit(1, 1)
    c.reset(0)
    for _ in range(3):
        c.h(0)
        c.reset(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="branch cap exceeded; no branches were discarded"):
        simulate_scored_trial(c, (0,), (1, -1), limits=InstrumentLimits(validation_branches=4))
    assert simulate_trial(c, 0)["mean"] == pytest.approx(1)


def test_tiny_nonzero_accepted_branch_is_retained_and_underflow_rejected():
    c = QuantumCircuit(2, 2)
    c.reset(range(2))
    c.ry(1e-12, 1)
    c.measure(1, 1)
    c.measure(0, 0)
    result = simulate_scored_trial(c, (0,), (1, -1), {1: 1})
    assert result["acceptance_probability"] == pytest.approx(2.5e-25, rel=1e-12, abs=0)
    assert result["mean"] == pytest.approx(1)
    tiny = c.copy_empty_like()
    tiny.reset(range(2))
    tiny.ry(1e-200, 1)
    tiny.measure(1, 1)
    tiny.measure(0, 0)
    with pytest.raises(ValueError, match="zero or unrepresentable"):
        simulate_scored_trial(tiny, (0,), (1, -1), {1: 1})


@pytest.mark.parametrize("variant", ["late_acceptance", "late_gate", "late_reset", "reused_bit", "duplicate_wire"])
def test_unsupported_event_schedules_reject_instead_of_dropping_work(variant):
    c = QuantumCircuit(2, 3)
    c.reset(range(2))
    if variant == "late_acceptance":
        c.measure(0, 0)
        c.measure(1, 1)
        outputs, accept = (0,), {1: 0}
    elif variant in {"late_gate", "late_reset"}:
        c.measure(0, 0)
        c.h(1) if variant == "late_gate" else c.reset(1)
        c.measure(1, 1)
        outputs, accept = (0,), {}
    elif variant == "reused_bit":
        c.measure(0, 0)
        c.measure(1, 0)
        outputs, accept = (0,), {}
    else:
        c.measure(0, 0)
        c.measure(0, 1)
        outputs, accept = (0, 1), {}
    with pytest.raises(ValueError):
        simulate_scored_trial(c, outputs, (1, -1) if len(outputs) == 1 else (1, .5, -.5, -1), accept)


@pytest.mark.parametrize("scores", [(1,), (1, float("nan")), (1, float("inf"))])
def test_joint_score_table_must_have_finite_correct_shape(scores):
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Finite score table"):
        simulate_scored_trial(c, (0,), scores)


def test_optional_candidate_view_preserves_default_search_and_pareto_results(tmp_path):
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.measure(0, 0)
    compiled = synthesize_instrument(c)
    profile = InstrumentProfile(maximum_code_distance=9)
    default = estimate_instrument(compiled, 1, profile=profile, cache_dir=tmp_path/"factory")
    candidates = estimate_instrument(compiled, 1, profile=profile, cache_dir=tmp_path/"factory",
                                     include_all_feasible_points=True)
    assert "all_feasible_points" not in default
    assert len(candidates["all_feasible_points"]) == default["feasible_candidate_count"]
    assert candidates["pareto_points"] == default["pareto_points"]
    assert all(point["full_physical_failure_model"] <= default["physical_failure_budget"]
               for point in candidates["all_feasible_points"])
    with pytest.raises(ValueError, match="must be Boolean"):
        estimate_instrument(compiled, 1, cache_dir=tmp_path/"factory", include_all_feasible_points=1)


def test_physical_accounting_does_not_bypass_the_recorded_width_cap(tmp_path):
    c = QuantumCircuit(9, 9)
    c.reset(range(9))
    c.measure(range(9), range(9))
    forged = SynthesizedInstrument(c, c, c, {"limits": asdict(WIDE)})
    with pytest.raises(ValueError, match="width/operation"):
        estimate_instrument(forged, 1, cache_dir=tmp_path/"factory")


@pytest.mark.parametrize("kind", ["reset", "measure"])
def test_hidden_instruments_cannot_be_randomly_sampled_inside_a_unitary_chunk(kind):
    hidden = QuantumCircuit(1, 1 if kind == "measure" else 0, name="hidden")
    hidden.reset(0) if kind == "reset" else hidden.measure(0, 0)
    c = QuantumCircuit(1, 2)
    c.reset(0)
    c.h(0)
    c.append(hidden.to_instruction(), [0], [1] if kind == "measure" else [])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Hidden or conditional instrument"):
        simulate_scored_trial(c, (0,), (1, -1))


@pytest.mark.parametrize("kind", ["reset", "measure"])
def test_noncanonical_multiwire_instruments_cannot_silently_ignore_second_wire(kind):
    c = QuantumCircuit(2, 2)
    c.reset(range(2))
    c.h(0)
    c.cx(0, 1)
    operation = Instruction(kind, 2, 0 if kind == "reset" else 2, [])
    c.append(operation, [0, 1], [] if kind == "reset" else [0, 1])
    if kind == "reset":
        c.measure(0, 0)
        c.measure(1, 1)
    with pytest.raises(ValueError, match="canonical single-wire"):
        simulate_scored_trial(c, (0, 1), (1, 0, 0, -1))


def test_cyclic_composite_definition_rejects_before_statevector_recursion():
    cycle = Gate("cycle", 1, [])
    definition = QuantumCircuit(1)
    definition.append(cycle, [0], copy=False)
    cycle.definition = definition
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(cycle, [0], copy=False)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Cyclic unitary definitions"):
        simulate_scored_trial(c, (0,), (1, -1))


def test_gate_named_barrier_cannot_silently_drop_quantum_work():
    operation = Gate("barrier", 1, [])
    definition = QuantumCircuit(1)
    definition.x(0)
    operation.definition = definition
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(operation, [0])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Only canonical barrier"):
        simulate_scored_trial(c, (0,), (1, -1))


def test_shared_composite_definition_charges_expanded_occurrences():
    definition = QuantumCircuit(1)
    definition.h(0)
    repeated = Gate("leaf", 1, [])
    repeated.definition = definition
    for level in range(8):
        definition = QuantumCircuit(1)
        definition.append(repeated, [0], copy=False)
        definition.append(repeated, [0], copy=False)
        repeated = Gate(f"level_{level}", 1, [])
        repeated.definition = definition
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(repeated, [0], copy=False)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Expanded unitary"):
        simulate_scored_trial(c, (0,), (1, -1), limits=InstrumentLimits(validation_operations=64))


def test_each_top_level_composite_occurrence_counts_and_valid_shared_dag_evolves():
    definition = QuantumCircuit(1)
    definition.h(0)
    shared = definition.to_gate()
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(shared, [0], copy=False)
    c.append(shared, [0], copy=False)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Expanded unitary occurrences"):
        simulate_scored_trial(c, (0,), (1, -1), limits=InstrumentLimits(validation_operations=5))
    checked = simulate_scored_trial(c, (0,), (1, -1), limits=InstrumentLimits(validation_operations=6))
    assert checked["mean"] == pytest.approx(1)
    assert checked["simulation"]["conservative_expanded_operations"] == 6
