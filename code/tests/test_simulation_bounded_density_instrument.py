"""Independent full-matrix/Kraus checks of the bounded density instrument."""
from dataclasses import asdict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate, Instruction
from qiskit.circuit.library import CRYGate, CXGate, CZGate, UnitaryGate

from openquantum import instrument_resources as ir
from openquantum.simulation.bounded_density_instrument import (
    DenseInstrumentLimits, simulate_density_trial, validate_density_source,
)

WIDE = DenseInstrumentLimits(max_qubits=6, source_operations=20_000,
                             validation_operations=1_000_000)


def _embed(matrix, wires, width):
    """Independent scalar bit-index embedding; no checker tensor helpers."""
    d, result = 2**width, np.zeros((2**width, 2**width), complex)
    mask = sum(1 << q for q in wires)
    for column in range(d):
        local_in = sum(((column >> q) & 1) << k for k, q in enumerate(wires))
        for local_out in range(2**len(wires)):
            row = (column & ~mask) | sum(((local_out >> k) & 1) << q for k, q in enumerate(wires))
            result[row, column] = matrix[local_out, local_in]
    return result


def _reference(circuit, outputs, scores, acceptance=None):
    """Explicit full-matrix multiplication and separate reset Kraus sum."""
    acceptance = dict(acceptance or {})
    width, d = circuit.num_qubits, 2**circuit.num_qubits
    rho = np.zeros((d, d), complex)
    rho[0, 0] = 1
    p0, p1 = np.diag([1., 0.]).astype(complex), np.diag([0., 1.]).astype(complex)
    reset_kraus = (np.array([[1, 0], [0, 0]], complex), np.array([[0, 1], [0, 0]], complex))
    mapping = {circuit.find_bit(item.clbits[0]).index: circuit.find_bit(item.qubits[0]).index
               for item in circuit.data if item.operation.name == "measure"}
    wires_out = tuple(mapping[bit] for bit in outputs)
    captured = None
    for item in circuit.data:
        name = item.operation.name
        wires = tuple(circuit.find_bit(q).index for q in item.qubits)
        if name == "barrier":
            continue
        if name == "reset":
            whole = [_embed(K, wires, width) for K in reset_kraus]
            rho = sum((K@rho@K.conj().T for K in whole), np.zeros_like(rho))
        elif name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in acceptance:
                K = _embed(p1 if acceptance[bit] else p0, wires, width)
                rho = K@rho@K.conj().T
            else:
                if bit in outputs and captured is None:
                    probability = float(np.trace(rho).real)
                    reduced = np.zeros((2**len(outputs), 2**len(outputs)), complex)
                    outside = tuple(q for q in range(width) if q not in wires_out)
                    for a in range(len(reduced)):
                        for b in range(len(reduced)):
                            for rest in range(2**len(outside)):
                                outside_word = sum(((rest >> k) & 1) << q for k, q in enumerate(outside))
                                row = outside_word | sum(((a >> k) & 1) << q for k, q in enumerate(wires_out))
                                column = outside_word | sum(((b >> k) & 1) << q for k, q in enumerate(wires_out))
                                reduced[a, b] += rho[row, column]/probability
                    probabilities = np.diag(reduced).real.copy()
                    captured = probability, float(np.asarray(scores)@probabilities), probabilities, reduced
                projectors = [_embed(P, wires, width) for P in (p0, p1)]
                rho = sum((P@rho@P.conj().T for P in projectors), np.zeros_like(rho))
        else:
            U = _embed(np.asarray(item.operation.to_matrix(), complex), wires, width)
            rho = U@rho@U.conj().T
    assert captured is not None
    return captured


def _mixed_correlated_trial():
    c = QuantumCircuit(6, 6)
    c.reset(range(6))
    c.ry(.61, 4)
    c.ry(.37, 1)
    c.cx(4, 1)
    c.cx(1, 5)
    c.reset(5)
    c.h(2)
    c.cz(2, 4)
    c.measure(2, 5)
    c.ry(.7, 3)
    c.measure(3, 4)
    c.rx(.27, 4)
    c.measure(4, 2)
    c.measure(1, 0)
    c.measure(0, 1)
    return c


def test_current_defaults_and_six_wire_opt_in_are_explicit():
    assert asdict(DenseInstrumentLimits()) == dict(max_qubits=5, source_operations=10000, validation_operations=100000, matrix_cache_bytes=1024*1024)
    assert asdict(ir.InstrumentLimits()) == dict(max_qubits=5, source_operations=10000,
                                               synthesized_operations=100000, validation_operations=100000,
                                               validation_branches=256)
    c = _mixed_correlated_trial()
    with pytest.raises(ValueError, match="width/operation"):
        simulate_density_trial(c, (2, 0), (1, .2, -.3, -.6), {4: 1})
    validate_density_source(c, limits=WIDE)


@pytest.mark.parametrize("outputs", [(2, 0), (0, 2)])
@pytest.mark.parametrize("batch", [True, False])
def test_six_wire_joint_density_reset_discard_and_projection_match_independent_full_matrix(outputs, batch):
    c, scores = _mixed_correlated_trial(), (1.1, .2, -.3, -.6)
    p, mean, probabilities, reduced = _reference(c, outputs, scores, {4: 1})
    result = simulate_density_trial(c, outputs, scores, {4: 1}, limits=WIDE, batch_single_qubit=batch)
    assert result["acceptance_probability"] == pytest.approx(p, abs=2e-14)
    assert result["mean"] == pytest.approx(mean, abs=2e-14)
    np.testing.assert_allclose(result["joint_probabilities"], probabilities, atol=2e-14)
    np.testing.assert_allclose(result["reduced_before_output"], reduced, atol=2e-14)
    assert np.max(np.abs(reduced.imag)) > .01  # Capture before output dephasing.
    assert result["simulation"]["density_shape"] == [64, 64]
    assert result["simulation"]["density_state_bytes"] == 65536
    assert result["simulation"]["reset_history_classical_bits_created"] == 0
    assert result["simulation"]["reset_history_measured_outputs"] is False
    assert result["simulation"]["trajectory_branch_pruning"] is False


def test_correlations_and_output_bit_order_are_not_replaced_by_independent_marginals():
    c = QuantumCircuit(6, 3)
    c.reset(range(6))
    c.h(4)
    c.cx(4, 1)
    c.cx(1, 5)
    c.reset(5)
    c.measure(4, 2)
    c.measure(1, 0)
    c.measure(0, 1)
    same = simulate_density_trial(c, (2, 0), (1, -1, -1, 1), limits=WIDE)
    assert same["mean"] == pytest.approx(1)
    np.testing.assert_allclose(same["joint_probabilities"], [.5, 0, 0, .5], atol=1e-14)
    assert same["reduced_before_output"][0, 3] == pytest.approx(0)
    opposite = c.copy_empty_like()
    opposite.reset(range(6))
    opposite.h(4)
    opposite.cx(4, 1)
    opposite.x(1)
    opposite.measure(4, 2)
    opposite.measure(1, 0)
    opposite.measure(0, 1)
    flipped = simulate_density_trial(opposite, (2, 0), (1, -1, -1, 1), limits=WIDE)
    assert flipped["mean"] == pytest.approx(-1)
    assert sum(same["joint_probabilities"][[1, 3]]) == pytest.approx(.5)
    assert sum(flipped["joint_probabilities"][[1, 3]]) == pytest.approx(.5)


@pytest.mark.parametrize("gate", [CXGate, CZGate, CRYGate])
def test_closed_and_open_control_matrix_targets_cannot_collide_in_cache(gate):
    c = QuantumCircuit(2, 2)
    c.reset(range(2))
    c.h(0)
    c.h(1)
    parameters = (.63,) if gate is CRYGate else ()
    c.append(gate(*parameters), [0, 1])
    c.append(gate(*parameters, ctrl_state=0), [0, 1])
    c.ry(.29, 1)
    c.measure(0, 0)
    c.measure(1, 1)
    p, mean, probabilities, reduced = _reference(c, (0, 1), (1, -.2, .3, -.7))
    result = simulate_density_trial(c, (0, 1), (1, -.2, .3, -.7))
    assert result["acceptance_probability"] == pytest.approx(p, abs=2e-14)
    assert result["mean"] == pytest.approx(mean, abs=2e-14)
    np.testing.assert_allclose(result["joint_probabilities"], probabilities, atol=2e-14)
    np.testing.assert_allclose(result["reduced_before_output"], reduced, atol=2e-14)


def test_reset_histories_do_not_need_branch_pruning_or_classical_records():
    c = QuantumCircuit(2, 2)
    c.reset(range(2))
    for _ in range(14):
        c.h(0)
        c.cx(0, 1)
        c.reset(1)
    c.measure(0, 0)
    c.measure(1, 1)
    result = simulate_density_trial(c, (0, 1), (1, -1, 1, -1))
    p, mean, probabilities, reduced = _reference(c, (0, 1), (1, -1, 1, -1))
    np.testing.assert_allclose(result["reduced_before_output"], reduced, atol=2e-14)
    assert result["mean"] == pytest.approx(mean, abs=2e-14)
    assert result["simulation"]["explicit_instrument_counts"] == dict(reset=16, measure=2)
    assert result["simulation"]["reset_history_classical_bits_created"] == 0


def test_batching_retains_noncommuting_same_wire_and_disjoint_wire_order():
    c = QuantumCircuit(3, 3)
    c.reset(range(3))
    for _ in range(3):
        c.h(0)
        c.s(1)
        c.rx(.32, 0)
        c.ry(.41, 1)
        c.t(0)
        c.cx(0, 2)
        c.reset(2)
        c.rz(-.21, 1)
    c.measure(range(3), range(3))
    scores = tuple(1-2*(i&1) for i in range(8))
    direct = simulate_density_trial(c, (0, 1, 2), scores, batch_single_qubit=False)
    batched = simulate_density_trial(c, (0, 1, 2), scores)
    expected = _reference(c, (0, 1, 2), scores)
    np.testing.assert_allclose(batched["reduced_before_output"], expected[3], atol=3e-14)
    np.testing.assert_allclose(batched["reduced_before_output"], direct["reduced_before_output"], atol=3e-14)
    assert batched["simulation"]["unitary_leaf_occurrences"] == direct["simulation"]["unitary_leaf_occurrences"]
    assert batched["simulation"]["density_unitary_updates"] < direct["simulation"]["density_unitary_updates"]


def test_raw_source_and_compiled_validation_caps_are_separate_and_exact():
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.h(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="width/operation"):
        validate_density_source(c, limits=DenseInstrumentLimits(source_operations=2))
    assert simulate_density_trial(c, (0,), (1, -1), limits=DenseInstrumentLimits(source_operations=2, validation_operations=3))["mean"] == pytest.approx(0)
    with pytest.raises(ValueError, match="width/operation"):
        simulate_density_trial(c, (0,), (1, -1), limits=DenseInstrumentLimits(validation_operations=2))
    composite = QuantumCircuit(2, 2)
    composite.reset(range(2))
    composite.cry(.2, 0, 1)
    composite.measure(range(2), range(2))
    with pytest.raises(ValueError, match="Expanded unitary occurrences"):
        simulate_density_trial(composite, (0,), (1, -1), limits=DenseInstrumentLimits(validation_operations=8))
    assert simulate_density_trial(composite, (0,), (1, -1), limits=DenseInstrumentLimits(validation_operations=9))["simulation"]["conservative_expanded_operations"] == 9


@pytest.mark.parametrize("kwargs", [{"max_qubits": 7}, {"max_qubits": True}, {"source_operations": 0},
                                    {"source_operations": 100001}, {"validation_operations": 1000001},
                                    {"validation_operations": False}, {"matrix_cache_bytes": 0},
                                    {"matrix_cache_bytes": 16*1024*1024+1}])
def test_hard_ceiling_and_invalid_limit_types_reject(kwargs):
    with pytest.raises(ValueError):
        DenseInstrumentLimits(**kwargs)


@pytest.mark.parametrize("name", ["reset", "measure", "barrier"])
def test_fake_named_gate_cannot_dispatch_as_an_instrument(name):
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(Gate(name, 1, []), [0])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="canonical"):
        simulate_density_trial(c, (0,), (1, -1))


@pytest.mark.parametrize("variant", ["late_accept", "late_gate", "late_reset", "reuse_bit", "duplicate_output_wire"])
def test_unsupported_readout_histories_reject(variant):
    c = QuantumCircuit(2, 3)
    c.reset(range(2))
    outputs, acceptance, scores = (0,), {}, (1, -1)
    c.measure(0, 0)
    if variant == "late_accept":
        c.measure(1, 1)
        acceptance = {1: 0}
    elif variant == "late_gate":
        c.h(1)
        c.measure(1, 1)
    elif variant == "late_reset":
        c.reset(1)
        c.measure(1, 1)
    elif variant == "reuse_bit":
        c.measure(1, 0)
    else:
        c.measure(0, 1)
        outputs, scores = (0, 1), (1, -.2, .3, -1)
    with pytest.raises(ValueError):
        simulate_density_trial(c, outputs, scores, acceptance)


def test_hidden_instrument_and_cyclic_definitions_reject_before_execution():
    for variant in ("reset", "measure", "cycle"):
        gate = Gate("hidden", 1, [])
        definition = QuantumCircuit(1, 1 if variant == "measure" else 0)
        if variant == "reset":
            definition.reset(0)
        elif variant == "measure":
            definition.measure(0, 0)
        else:
            definition.data.append(CircuitInstruction(gate, (definition.qubits[0],), ()))
        gate.definition = definition
        c = QuantumCircuit(1, 1)
        c.reset(0)
        c.append(gate, [0])
        c.measure(0, 0)
        with pytest.raises(ValueError, match="Hidden|arity|Cyclic"):
            simulate_density_trial(c, (0,), (1, -1))


def test_repeated_shared_definition_work_is_charged_per_occurrence():
    leaf = QuantumCircuit(1)
    leaf.h(0)
    leaf.h(0)
    shared = leaf.to_gate()
    outer = QuantumCircuit(1)
    outer.append(shared, [0])
    outer.append(shared, [0])
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(outer.to_gate(), [0])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Expanded unitary occurrences"):
        simulate_density_trial(c, (0,), (1, -1), limits=DenseInstrumentLimits(validation_operations=8))
    assert simulate_density_trial(c, (0,), (1, -1), limits=DenseInstrumentLimits(validation_operations=9))["mean"] == pytest.approx(1)


@pytest.mark.parametrize("variant", ["duplicate_wire", "missing_wire"])
def test_malformed_nested_instruction_is_rejected_during_preflight(variant):
    definition = QuantumCircuit(2)
    qargs = (definition.qubits[0], definition.qubits[0]) if variant == "duplicate_wire" else (definition.qubits[0],)
    # Deliberately bypass Qiskit's public append validation to exercise the
    # checker's preflight rather than a circuit-construction exception.
    definition._append(CircuitInstruction(CXGate(), qargs, ()))
    gate = Gate("malformed", 2, [])
    gate.definition = definition
    c = QuantumCircuit(2, 2)
    c.reset(range(2))
    c.append(gate, [0, 1])
    c.measure(range(2), range(2))
    with pytest.raises(ValueError, match="Nested instruction arity"):
        simulate_density_trial(c, (0,), (1, -1))


def test_noncanonical_dual_matrix_definition_is_rejected_without_silent_semantic_choice():
    class Ambiguous(Gate):
        def __init__(self):
            super().__init__("ambiguous", 1, [])
            definition = QuantumCircuit(1)
            definition.x(0)
            self.definition = definition
        def to_matrix(self):
            return np.eye(2)
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(Ambiguous(), [0])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="ambiguous representations"):
        simulate_density_trial(c, (0,), (1, -1))


def test_bounded_matrix_cache_eviction_changes_no_joint_density_or_operation_counts():
    c = QuantumCircuit(2, 2)
    c.reset(range(2))
    for angle in (.13, .27, .39, .51, .63, .75):
        c.rx(angle, 0)
        c.ry(angle, 1)
        c.cx(0, 1)
    c.measure(range(2), range(2))
    scores = (1, -.2, .3, -.7)
    expected = _reference(c, (0, 1), scores)
    small = simulate_density_trial(c, (0, 1), scores, limits=DenseInstrumentLimits(matrix_cache_bytes=128))
    uncached = simulate_density_trial(c, (0, 1), scores, limits=DenseInstrumentLimits(matrix_cache_bytes=1))
    for result in (small, uncached):
        np.testing.assert_allclose(result["reduced_before_output"], expected[3], atol=3e-14)
        assert result["simulation"]["top_level_instructions"] == len(c.data)
        assert result["simulation"]["conservative_expanded_operations"] == len(c.data)
        assert result["simulation"]["peak_matrix_cache_bytes"] <= result["simulation"]["matrix_cache_bytes_limit"]
    assert small["simulation"]["matrix_cache_evictions"] > 0
    assert uncached["simulation"]["matrix_cache_skipped_targets"] > 0


def test_large_unitary_temporary_is_width_bounded_and_not_synthesized_or_cached():
    c = QuantumCircuit(6, 6)
    c.reset(range(6))
    c.append(UnitaryGate(np.eye(64)), range(6))
    c.measure(range(6), range(6))
    result = simulate_density_trial(c, (0, 1, 2), tuple(1-2*(i&1) for i in range(8)), limits=WIDE)
    assert result["mean"] == pytest.approx(1)
    assert result["simulation"]["maximum_unitary_matrix_bytes"] == 65536
    assert result["simulation"]["peak_matrix_cache_bytes"] == 0
    assert result["simulation"]["matrix_cache_skipped_targets"] == 1
    assert result["simulation"]["index_group_cache_bytes"] <= 36*64*8


def test_unbounded_output_and_score_iterators_are_rejected_before_consumption():
    class NeverConsume:
        def __iter__(self):
            raise AssertionError("unbounded input was consumed")
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="finite bounded sequence"):
        simulate_density_trial(c, NeverConsume(), (1, -1))
    with pytest.raises(ValueError, match="finite bounded sequence"):
        simulate_density_trial(c, (0,), NeverConsume())
    with pytest.raises(ValueError, match="Output count"):
        simulate_density_trial(c, (0, 1), NeverConsume())


def test_nearly_unitary_repeated_matrix_cannot_hide_an_invalid_probability():
    class SlightGain(Gate):
        def __init__(self):
            super().__init__("slight_gain", 1, [])
        def to_matrix(self):
            return np.diag([1+4e-11, 1]).astype(complex)
    c = QuantumCircuit(1, 1)
    c.reset(0)
    for _ in range(1000):
        c.append(SlightGain(), [0])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="probability guard"):
        simulate_density_trial(c, (0,), (1, -1))


@pytest.mark.parametrize("matrix", [np.array([[1, 0], [0, 2]], complex), np.array([[1, 0], [0, np.nan]], complex)])
def test_invalid_matrix_leaves_reject(matrix):
    class MatrixLeaf(Gate):
        def __init__(self):
            super().__init__("matrix_leaf", 1, [])
        def to_matrix(self):
            return matrix
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.append(MatrixLeaf(), [0])
    c.measure(0, 0)
    with pytest.raises(ValueError, match="unitary|nonfinite"):
        simulate_density_trial(c, (0,), (1, -1))


def test_missing_reset_parameters_and_invalid_scores_reject():
    c = QuantumCircuit(1, 1)
    c.h(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="initial reset"):
        simulate_density_trial(c, (0,), (1, -1))
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.measure(0, 0)
    for scores in ((1,), (1, np.inf), (1, np.nan)):
        with pytest.raises(ValueError, match="score table"):
            simulate_density_trial(c, (0,), scores)
    with pytest.raises(ValueError, match="Boolean"):
        simulate_density_trial(c, (0,), (1, -1), batch_single_qubit=1)


def test_old_n2_density_answer_and_current_default_files_are_unchanged():
    from openquantum.simulation.lindblad_physical_workload import build_source
    c = build_source()
    old = ir.simulate_trial(c, 0)
    new = simulate_density_trial(c, (0,), (1, -1))
    assert new["mean"] == pytest.approx(old["mean"], abs=2e-13)
    np.testing.assert_allclose(new["reduced_before_output"], old["reduced_before_output"], atol=2e-13)
    assert ir.MAX_QUBITS == 5 and ir.MAX_SOURCE_OPERATIONS == 10000 and ir.MAX_SYNTHESIZED_OPERATIONS == 100000


def test_literal_n3_r134_preserves_all_resets_readouts_and_joint_density():
    from openquantum.simulation import lindblad_workload as lw
    from openquantum.simulation.two_spin_lindblad import build_chain_circuit
    task = lw.ChainTask(3)
    c = build_chain_circuit(3, task.time, task.J, task.hx, task.rates, 134, 2, "product", observable="Z")
    scores = tuple(1-2*(i&1) for i in range(8))
    validate_density_source(c, limits=WIDE)
    actual = simulate_density_trial(c, (0, 1, 2), scores, limits=WIDE)
    expected = lw.gate_density(task, 134)
    np.testing.assert_allclose(actual["reduced_before_output"], expected, atol=3e-12)
    np.testing.assert_allclose(actual["joint_probabilities"], np.diag(expected).real, atol=3e-12)
    assert actual["mean"] == pytest.approx(.791052626111264, abs=3e-12)
    assert actual["simulation"]["explicit_instrument_counts"] == dict(reset=673, measure=6)
    assert actual["simulation"]["top_level_instructions"] == 4435
    assert actual["simulation"]["conservative_expanded_operations"] == 7115
    assert actual["simulation"]["density_state_bytes"] == 65536
    assert actual["simulation"]["reset_history_classical_bits_created"] == 0
