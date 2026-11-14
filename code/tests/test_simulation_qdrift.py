# SPDX-License-Identifier: MIT
"""Independent gate, exact-categorical, channel and complete-trial checks."""
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from openquantum.simulation import qdrift as q


# Build Pauli matrices by their action on computational basis bits, rather than
# using either Qiskit Pauli or the implementation's Kronecker construction.
def bit_action_pauli(label):
    dimension = 2 ** len(label)
    result = np.zeros((dimension, dimension), complex)
    for basis in range(dimension):
        output, phase = basis, 1.0 + 0j
        for wire, character in enumerate(reversed(label)):
            bit = (basis >> wire) & 1
            if character == "X":
                output ^= 1 << wire
            elif character == "Y":
                output ^= 1 << wire
                phase *= -1j if bit else 1j
            elif character == "Z" and bit:
                phase *= -1
        result[output, basis] = phase
    return result


def independent_product(hamiltonian, time, steps, schedule):
    result = np.eye(2**hamiltonian.qubits, dtype=complex)
    tau = float(hamiltonian.l1_norm * Fraction(str(time)) / steps)
    for index in schedule:
        coefficient, label = hamiltonian.terms[index]
        sign = 1 if coefficient > 0 else -1
        result = expm(-1j * sign * tau * bit_action_pauli(label)) @ result
    return result


def close(actual, expected, atol=2e-12):
    assert np.allclose(actual, expected, atol=atol, rtol=0)


@pytest.mark.parametrize("label", ["X", "Y", "Z", "IX", "XI", "YZ", "ZY", "XYZ", "IYI"])
@pytest.mark.parametrize("sign,time", [(1, 0.37), (-1, 0.37), (1, -0.37)])
def test_actual_primitive_pauli_evolution_matches_bit_action(label, sign, time):
    h = q.Hamiltonian.from_terms(len(label), [(Fraction(sign, 3), label)])
    schedule = (0, 0, 0)
    circuit = q.qdrift_circuit(h, time, 3, 7, schedule=schedule)
    close(Operator(circuit).data, independent_product(h, time, 3, schedule))
    assert set(circuit.count_ops()) <= q.PRIMITIVES
    assert circuit.count_ops()["rz"] == 3


def test_noncommuting_gate_order_and_reverse_time():
    h = q.Hamiltonian.from_terms(1, [("0.7", "X"), ("-0.2", "Y"), ("0.3", "Z")])
    schedule = (2, 0, 1)
    forward = Operator(q.qdrift_circuit(h, 0.73, 3, 0, schedule=schedule)).data
    close(forward, independent_product(h, 0.73, 3, schedule))
    backward = Operator(q.qdrift_circuit(h, -0.73, 3, 0, schedule=schedule[::-1])).data
    close(backward @ forward, np.eye(2))
    same_order_backward = Operator(q.qdrift_circuit(h, -0.73, 3, 0, schedule=schedule)).data
    assert np.linalg.norm(same_order_backward @ forward - np.eye(2), 2) > 1e-3


def test_identity_terms_retain_global_phase_and_sampling_weight():
    h = q.Hamiltonian.from_terms(2, [("0.3", "II"), ("-0.7", "YX")])
    assert h.l1_norm == 1 and h.integer_weights == (3, 7)
    selected = (0, 1, 0)
    circuit = q.qdrift_circuit(h, 0.49, 3, 0, schedule=selected)
    close(Operator(circuit).data, independent_product(h, 0.49, 3, selected))
    assert circuit.count_ops()["rz"] == 1
    assert not np.isclose(float(circuit.global_phase), 0)
    identity = q.Hamiltonian.from_terms(1, [("-2/3", "I")])
    phase = Operator(q.qdrift_circuit(identity, 0.53, 5, 0)).data
    close(phase, np.exp(1j * (2/3) * 0.53) * np.eye(2))
    close(q.averaged_channel(identity, 0.53, 5), np.eye(4))


def test_exact_decimal_combination_and_categorical_boundaries():
    h = q.Hamiltonian.from_terms(1, [(0.1, "X"), ("0.2", "X"), ("-0.6", "Y"), ("0", "Z")])
    assert h.terms == ((Fraction(3, 10), "X"), (Fraction(-3, 5), "Y"))
    assert h.l1_norm == Fraction(9, 10)
    assert h.integer_weights == (1, 2) and h.total_weight == 3
    assert [q.term_from_ticket(h, ticket) for ticket in range(3)] == [0, 1, 1]
    assert h.as_dict()["probabilities_exact"] == ["1/3", "2/3"]
    for invalid in (-1, 3, True, 0.0):
        with pytest.raises(ValueError):
            q.term_from_ticket(h, invalid)


def test_cancelled_and_zero_hamiltonian_keep_preparation_readout_and_guard():
    h = q.Hamiltonian.from_terms(2, [(1, "XX"), (-1, "XX"), (0, "IZ")])
    assert h.terms == () and h.l1_norm == 0 and h.total_weight == 0
    assert q.sample_schedule(h, 7, 99) == ()
    with pytest.raises(ValueError):
        q.term_from_ticket(h, 0)
    close(q.averaged_channel(h, 0.5, 7), np.eye(16))
    measured = q.qdrift_circuit(h, 0.5, 7, 99, initial_basis=3, measure=True)
    assert dict(measured.count_ops()) == {"reset": 2, "x": 2, "measure": 2}
    ledger = q.logical_ledger(h, 0.5, 7, 99, initial_basis=3)
    assert ledger["categorical_draws"] == 0
    assert ledger["complete_primitive_counts"] == {"reset": 2, "x": 2, "measure": 2}
    with pytest.raises(ValueError, match="steps"):
        q.qdrift_circuit(h, steps=q.MAX_EXPONENTIALS + 1)


def test_seed_reproduction_and_distinct_recorded_streams():
    h = q.Hamiltonian.from_terms(1, [(1, "X"), (2, "Z")])
    assert q.sample_schedule(h, 32, 99) == q.sample_schedule(h, 32, 99)
    pairs = q.paired_seeds(99, 8)
    assert pairs == q.paired_seeds(99, 8)
    assert len(pairs) == 8 and len({seed for pair in pairs for seed in pair}) == 16
    # This is a reproducibility/inventory check, not a statistical test of
    # independence or uniformity of the deterministic PRNG streams.


def test_exhaustive_two_step_noncommuting_mixture_and_choi():
    h = q.Hamiltonian.from_terms(1, [("-0.6", "X"), ("0.4", "Z")])
    steps, time, dimension = 2, 0.43, 2
    probabilities = [Fraction(3, 5), Fraction(2, 5)]
    trajectories = []
    choi_expected = np.zeros((4, 4), complex)
    bell = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    for selected in product(range(2), repeat=steps):
        probability = float(probabilities[selected[0]] * probabilities[selected[1]])
        unitary = independent_product(h, time, steps, selected)
        trajectories.append((probability, unitary))
        vector = np.kron(np.eye(2), unitary) @ bell
        choi_expected += probability * np.outer(vector, vector.conj())
    actual = q.averaged_channel(h, time, steps)
    for a, b in product(range(dimension), repeat=2):
        matrix_unit = np.zeros((dimension, dimension), complex)
        matrix_unit[a, b] = 1
        expected = sum(probability * unitary @ matrix_unit @ unitary.conj().T for probability, unitary in trajectories)
        close(q.linear_channel_action(actual, matrix_unit), expected)
        # Column vectorization: the E_ab input is column a+b*d.
        close(actual[:, a + b*dimension], expected.reshape(-1, order="F"))
    choi = q.normalized_choi(actual, dimension)
    close(choi, choi_expected)
    tensor = choi.reshape(dimension, dimension, dimension, dimension)
    close(np.trace(tensor, axis1=1, axis2=3), np.eye(dimension)/dimension)
    close(np.trace(tensor, axis1=0, axis2=2), np.eye(dimension)/dimension)
    assert np.linalg.eigvalsh(choi).min() >= -2e-12
    assert np.trace(choi).real == pytest.approx(1, abs=2e-12)


def test_full_two_spin_channel_by_direct_kraus_recurrence_on_spanning_units():
    h = q.Hamiltonian.from_terms(2, [("0.7", "IX"), ("0.3", "XI"), ("-0.2", "ZZ")])
    steps, time = 3, 0.31
    tau = float(h.l1_norm * Fraction(str(time))/steps)
    gates = [expm(-1j * (1 if coefficient > 0 else -1) * tau * bit_action_pauli(label)) for coefficient, label in h.terms]
    probabilities = [float(abs(coefficient)/h.l1_norm) for coefficient, _ in h.terms]
    channel = q.averaged_channel(h, time, steps)
    for a, b in product(range(4), repeat=2):
        value = np.zeros((4, 4), complex)
        value[a, b] = 1
        source = value.copy()
        for _ in range(steps):
            value = sum(p * gate @ value @ gate.conj().T for p, gate in zip(probabilities, gates))
        close(q.linear_channel_action(channel, source), value)
    diagnostics = q.channel_diagnostics(channel, 4)
    assert diagnostics["spanning_matrix_units"] == 16
    assert diagnostics["trace_preservation_max_error"] < 2e-12
    assert diagnostics["hermiticity_preservation_max_error"] < 2e-12
    assert diagnostics["unitality_error"] < 2e-12
    assert diagnostics["normalized_choi_min_eigenvalue"] > -2e-12


def test_averaged_channel_is_distinct_from_one_sampled_unitary():
    h = q.Hamiltonian.from_terms(1, [("0.5", "X"), ("0.5", "Z")])
    input_state = np.diag([1, 0]).astype(complex)
    ensemble_state = q.linear_channel_action(q.averaged_channel(h, 0.7, 1), input_state)
    fixed = Operator(q.qdrift_circuit(h, 0.7, 1, 0, schedule=(0,))).data
    fixed_state = fixed @ input_state @ fixed.conj().T
    assert np.trace(ensemble_state @ ensemble_state).real < 0.99
    assert np.trace(fixed_state @ fixed_state).real == pytest.approx(1, abs=2e-12)
    assert np.linalg.norm(ensemble_state - fixed_state) > 0.1
    comparison = q.channel_comparison(h, 0.7, 1)
    assert comparison["hs_induced_superoperator_error"] > 0
    assert "none is a computed diamond norm" in comparison["metric_scope"]


def test_measured_trial_contains_actual_reset_preparation_and_readout():
    h = q.Hamiltonian.from_terms(2, [("0.4", "IX"), ("-0.6", "YZ")])
    selected = (1, 0, 1)
    unitary = q.qdrift_circuit(h, 0.29, 3, 7, schedule=selected, initial_basis=2)
    close(Statevector.from_instruction(unitary).data, independent_product(h, 0.29, 3, selected)[:, 2])
    measured = q.qdrift_circuit(h, 0.29, 3, 7, schedule=selected, initial_basis=2, measure=True)
    counts = dict(measured.count_ops())
    assert counts["reset"] == 2 and counts["x"] == 1 and counts["measure"] == 2
    assert counts["rz"] == 3 and counts["cx"] == 4
    assert counts["h"] == 6 and counts["sdg"] == counts["s"] == 2
    assert set(counts) <= q.PRIMITIVES
    assert measured.num_qubits == measured.num_clbits == 2
    operations = list(measured.data)
    assert all(item.operation.name == "reset" for item in operations[:2])
    assert all(item.operation.name == "measure" for item in operations[-2:])
    exported = qasm3.dumps(measured)
    assert "reset" in exported and "measure" in exported and "unitary" not in exported
    ledger = q.logical_ledger(h, 0.29, 3, 7, schedule=selected, initial_basis=2)
    assert ledger["complete_primitive_counts"] == counts
    assert ledger["physical_resource_estimate"] is None


def test_empirical_trials_redraw_complete_schedules_and_charge_every_trial(monkeypatch):
    h = q.Hamiltonian.from_terms(2, [(1, "ZZ"), ("1/2", "IX"), ("-1/2", "XI")])
    original = q.sample_schedule
    calls = []
    def recorded_schedule(hamiltonian, steps, seed):
        calls.append((steps, seed))
        return original(hamiltonian, steps, seed)
    monkeypatch.setattr(q, "sample_schedule", recorded_schedule)
    result = q.empirical_study(h, 0.23, 3, trajectories=4, master_seed=99, initial_basis=1)
    assert len(calls) == 4 and len({seed for _, seed in calls}) == 4
    assert all(steps == 3 for steps, _ in calls)
    assert result["complete_workload"]["fresh_preparations"] == 4
    assert result["complete_workload"]["term_draws"] == 12
    assert result["complete_workload"]["zero_state_resets"] == 8
    assert result["complete_workload"]["final_measurements"] == 8
    assert result["complete_workload"]["primitive_counts"]["rz"] == 12
    assert result["complete_workload"]["primitive_counts"]["x"] == 4
    aggregate = {}
    for record in result["records"]:
        assert record["classically_sampled_z0"] in (-1, 1)
        assert len(record["schedule_indices"]) == 3
        assert record["source_to_independent_operator_error"] < 2e-12
        assert record["source_to_independent_state_error"] < 2e-12
        assert record["actual_reset_prefix_density_error"] < 2e-12
        assert record["state_probability_norm_before_normalization"] == pytest.approx(1, abs=2e-12)
        outcome = record["classically_sampled_basis_outcome"]
        assert 0 <= outcome < 4
        assert record["readout_bitstring_qiskit_order"] == format(outcome, "02b")
        assert record["classically_sampled_z0"] == (1 if outcome & 1 == 0 else -1)
        for gate, count in record["logical_ledger"]["complete_primitive_counts"].items():
            aggregate[gate] = aggregate.get(gate, 0) + count
    assert aggregate == result["complete_workload"]["primitive_counts"]
    assert result["sample_mean"] == sum(r["classically_sampled_z0"] for r in result["records"])/4
    assert result["ideal_iid_hoeffding"]["absolute_radius"] > 0
    assert "independent ideal categorical" in result["randomness_scope"]
    assert "no QPU execution" in result["sample_kind"]


@pytest.mark.parametrize("coefficient", [True, float("nan"), float("inf"), complex(1, 0), "nan", "1/0"])
def test_invalid_coefficient(coefficient):
    with pytest.raises(ValueError):
        q.Hamiltonian.from_terms(1, [(coefficient, "X")])


@pytest.mark.parametrize("qubits,terms", [(True, [(1, "X")]), (0, []), (65, []), (1, [(1, "XX")]), (1, [(1, "A")]), (1, [(1,)])])
def test_invalid_hamiltonian_shape(qubits, terms):
    with pytest.raises(ValueError):
        q.Hamiltonian.from_terms(qubits, terms)


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": True}, {"steps": 513}, {"seed": True}, {"seed": -1}, {"seed": 2**64}, {"time": float("nan")}, {"time": True}, {"initial_basis": 2}, {"initial_basis": True}, {"measure": 1}, {"schedule": (True,)}])
def test_invalid_trial_arguments(kwargs):
    h = q.Hamiltonian.from_terms(1, [(1, "X")])
    if "schedule" in kwargs:
        kwargs["steps"] = 1
    with pytest.raises(ValueError):
        q.qdrift_circuit(h, **kwargs)


def test_loop_and_dense_guards_fire_before_allocation(monkeypatch):
    h = q.Hamiltonian.from_terms(4, [(1, "IIIX")])
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation reached before guard")
    monkeypatch.setattr(q.np, "zeros", forbidden)
    monkeypatch.setattr(q.np, "eye", forbidden)
    with pytest.raises(ValueError, match="dense"):
        q.dense_hamiltonian(h)
    with pytest.raises(ValueError, match="dense"):
        q.averaged_channel(h, 0.5, 8)
    with pytest.raises(ValueError, match="dimension"):
        q.normalized_choi(np.empty((1, 1)), 9)


def test_primitive_cap_fires_before_circuit_materialization(monkeypatch):
    h = q.Hamiltonian.from_terms(64, [(1, "Y" * 64)])
    def forbidden(*args, **kwargs):
        raise AssertionError("circuit construction reached before operation guard")
    monkeypatch.setattr(q, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError, match="operation guard"):
        q.qdrift_circuit(h, 0.5, 512, 7)


def test_large_generator_does_not_need_dense_reference():
    h = q.Hamiltonian.from_terms(32, [(1, "Z" * 32)])
    circuit = q.qdrift_circuit(h, 0.13, 1, 0, measure=True)
    assert circuit.num_qubits == 32 and circuit.count_ops()["cx"] == 62
    with pytest.raises(ValueError, match="dense"):
        q.channel_comparison(h, 0.13, 1)


def test_input_term_and_exact_weight_representation_caps():
    with pytest.raises(ValueError, match="more than"):
        q.Hamiltonian.from_terms(1, ((1, "X") for _ in range(q.MAX_TERMS + 1)))
    with pytest.raises(ValueError, match="rational representation"):
        q.Hamiltonian.from_terms(1, [(Fraction(1, 2**q.MAX_WEIGHT_BITS), "X")])


@pytest.mark.parametrize("function,args", [(q.unitary_channel, (np.eye(2)*2,)), (q.unitary_channel, (np.array([[np.nan]]),)), (q.linear_channel_action, (np.eye(4), np.eye(3))), (q.linear_channel_action, (np.eye(4), np.array([[np.inf, 0], [0, 0]])))])
def test_invalid_bounded_matrix_inputs(function, args):
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize("changes", [
    {"l1_norm": 1},
    {"l1_norm": True},
    {"integer_weights": [1]},
    {"cumulative_weights": [1]},
    {"cumulative_weights": (True,)},
    {"terms": ([Fraction(1), "X"],)},
    {"total_weight": True},
    {"integer_weights": (2,), "cumulative_weights": (2,), "total_weight": 1},
    {"terms": ((Fraction(1), "X"), (Fraction(2), "Y")), "l1_norm": Fraction(3),
     "integer_weights": (2, 1), "cumulative_weights": (2, 3), "total_weight": 3},
])
def test_canonical_constructor_rejects_untyped_or_inconsistent_metadata(changes):
    fields = {"qubits": 1, "terms": ((Fraction(1), "X"),), "l1_norm": Fraction(1),
              "integer_weights": (1,), "cumulative_weights": (1,), "total_weight": 1}
    fields.update(changes)
    with pytest.raises(ValueError):
        q.Hamiltonian(**fields)


def test_distinct_seed_derivation_has_finite_failure_budget(monkeypatch):
    class ConstantStream:
        calls = 0
        def __init__(self, seed):
            pass
        def getrandbits(self, bits):
            assert bits == 64
            type(self).calls += 1
            return 123
    monkeypatch.setattr(q.random, "Random", ConstantStream)
    with pytest.raises(RuntimeError, match="finite candidate cap"):
        q.paired_seeds(7, 4)
    assert ConstantStream.calls == 8 * 4


def test_explicit_chain_representation_and_normalization():
    h = q.chain_hamiltonian(2, coupling="-1/2", x_field="7/10", z_field="1/5")
    assert h.terms == ((Fraction(-1, 2), "ZZ"), (Fraction(7, 10), "IX"),
                       (Fraction(1, 5), "IZ"), (Fraction(7, 10), "XI"), (Fraction(1, 5), "ZI"))
    assert h.l1_norm == Fraction(23, 10)
    one = q.chain_hamiltonian(1, coupling=0, x_field="7/10", z_field="3/10")
    assert one.terms == ((Fraction(7, 10), "X"), (Fraction(3, 10), "Z"))
    assert one.l1_norm == 1


@pytest.mark.parametrize("value", [Fraction(10**400), Fraction(1, 2**4096)])
def test_canonical_constructor_rejects_extreme_fraction_representations(value):
    with pytest.raises(ValueError, match="finite|representation"):
        q.Hamiltonian(1, ((value, "X"),), abs(value), (1,), (1,), 1)


def _generator_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/generate_simulation_qdrift.py'
    spec = importlib.util.spec_from_file_location('qdrift_generator_atomic_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_generator_study(output, case_id, spec, *args):
    (output / case_id).mkdir()
    return {'ensemble_reference': {'ensemble_observable': 0.5, 'exact_observable': 0.6,
                                  'observable_bias': 0.1, 'hs_induced_superoperator_error': 0.2},
            'sample_mean': 0.5, 'ideal_iid_hoeffding': {'absolute_radius': 1.0}}


def test_late_payload_failure_cannot_leave_completed_manifest(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    module = _generator_module()
    monkeypatch.setattr(module, 'source_commit', lambda: None)
    monkeypatch.setattr(module, 'export_case', _mock_generator_study)
    output = tmp_path / 'late-failure'
    monkeypatch.setattr(sys, 'argv', ['generate_simulation_qdrift.py', '--qubits', '1', '--steps', '1',
                                    '--trajectories', '1', '--output-dir', str(output)])
    original_open = Path.open
    def fail_summary(path, *args, **kwargs):
        if path == output / 'summary.csv':
            raise OSError('injected late summary I/O failure')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_summary)
    with pytest.raises(OSError, match='injected late summary'):
        module.main()
    assert output.exists()
    assert not (output / 'manifest.json').exists()
    assert not (output / '.manifest-write.tmp').exists()


def test_final_atomic_publish_failure_leaves_no_completed_marker(tmp_path, monkeypatch):
    import sys
    module = _generator_module()
    monkeypatch.setattr(module, 'source_commit', lambda: None)
    monkeypatch.setattr(module, 'export_case', _mock_generator_study)
    output = tmp_path / 'atomic-failure'
    monkeypatch.setattr(sys, 'argv', ['generate_simulation_qdrift.py', '--qubits', '1', '--steps', '1',
                                    '--trajectories', '1', '--output-dir', str(output)])
    def fail_replace(*args):
        raise OSError('injected atomic publish failure')
    monkeypatch.setattr(module.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='injected atomic publish'):
        module.main()
    assert (output / 'summary.csv').is_file()
    assert not (output / 'manifest.json').exists()
    assert not (output / '.manifest-write.tmp').exists()


def test_source_commit_does_not_inherit_an_ancestor_checkout(monkeypatch):
    from types import SimpleNamespace
    module = _generator_module()
    calls = []
    def ancestor(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=str(module.ROOT.parent))
    monkeypatch.setattr(module.subprocess, 'run', ancestor)
    assert module.source_commit() is None
    assert calls == [['git', 'rev-parse', '--show-toplevel']]


def test_source_commit_authenticates_actual_producer_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = _generator_module()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'SOURCE_FILES', ('producer.py',))
    (tmp_path / 'producer.py').write_bytes(b'actual producer bytes')
    captured = {'blob': b'wrong historical producer'}
    def git(command, **kwargs):
        if command[-1] == '--show-toplevel':
            return SimpleNamespace(returncode=0, stdout=str(tmp_path))
        if command[-1] == 'HEAD':
            return SimpleNamespace(returncode=0, stdout='a' * 40)
        return SimpleNamespace(returncode=0, stdout=captured['blob'])
    monkeypatch.setattr(module.subprocess, 'run', git)
    with pytest.raises(ValueError, match='producer source differs'):
        module.source_commit()
    captured['blob'] = b'actual producer bytes'
    assert module.source_commit() == 'a' * 40
