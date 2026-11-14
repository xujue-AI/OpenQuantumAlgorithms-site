"""Independent transcript and actual-circuit checks for the compiled QAOA replay.

These checks never run the optimizer, rotation synthesizer, or native estimator.
Compiled artifacts are audited only after the frozen generator has produced them.
"""

from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit import qasm3


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "assets/generated/optimization/diagnostics"
OUTPUT = ROOT / "assets/generated/optimization/compiled-replay"
CASE_ID = "qaoa-cycle-4-p2"
RUN_ID = "qaoa-cycle-4-p2-alternate-seed-7"
EDGES = ((0, 1), (1, 2), (2, 3), (0, 3))
PARENT_RAW_SHA = "a8649f45dd948302f201654e9ad341824363357fd37f8ed343281cdf9d846688"
PARENT_SOURCE_SHA = "a617b5e8d43b648427e5ceaa5e35d3f50ea14c02464465084f23f457ffb5855d"
PARENT_GENERATOR_SHA = "0c0f3ff0f177dca99a00cc611261c3b6c82ce39be251e8d3a54026b0f5346f3e"
BACKEND_SHA = "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"
LOCK_SHA = "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"
FIRST_CT_INVENTORY_SHA = "c37f825c74c909a0f03288d9a24fc5d769d49014e39bfbf09db4e0970de8f3f7"
SOURCE_COUNTS = {"h": 4, "rx": 8, "rz": 8, "cx": 16, "measure": 4, "barrier": 1}
WEIGHTED_SOURCE = {"h": 328128, "rx": 656256, "rz": 656256,
                   "cx": 1312512, "measure": 328128}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def parent_study():
    assert sha(PARENT / "raw-study.json") == PARENT_RAW_SHA
    return json.loads((PARENT / "raw-study.json").read_text())


@pytest.fixture(scope="module")
def selected(parent_study):
    case = next(c for c in parent_study["cases"] if c["case_id"] == CASE_ID)
    run = next(r for r in case["runs"] if r["run_id"] == RUN_ID)
    return case, run


@pytest.fixture(scope="module")
def parent_producer():
    source = PARENT / "optimization.py"
    assert sha(source) == PARENT_SOURCE_SHA
    spec = importlib.util.spec_from_file_location("compiled_test_parent_producer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.dont_write_bytecode = True
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module


def _cut_values():
    return np.asarray([sum(((bits >> u) & 1) != ((bits >> v) & 1)
                           for u, v in EDGES) for bits in range(16)], dtype=float)


def _wrap(parameters):
    return (np.asarray(parameters, dtype=float) + math.pi) % (2 * math.pi) - math.pi


def _qaoa_state(parameters):
    """Independent phase and bit-flip evolution in integer qubit order."""
    state = np.full(16, 0.25, dtype=complex)
    cuts, indexes = _cut_values(), np.arange(16)
    for gamma, beta in zip(parameters[::2], parameters[1::2], strict=True):
        state *= np.exp(-1j * gamma * cuts)
        for wire in range(4):
            state = (math.cos(beta) * state
                     - 1j * math.sin(beta) * state[indexes ^ (1 << wire)])
    return state


def _qaoa_unitary(parameters):
    """Dense mixer and diagonal cost product, independent of source gates."""
    size, indexes = 16, np.arange(16)
    hadamard = np.asarray([[1, 1], [1, -1]], complex) / math.sqrt(2)
    unitary = hadamard
    for _ in range(3):
        unitary = np.kron(unitary, hadamard)
    for gamma, beta in zip(parameters[::2], parameters[1::2], strict=True):
        cost = np.diag(np.exp(-1j * gamma * _cut_values()))
        mixer = np.eye(size, dtype=complex)
        for wire in range(4):
            x = np.eye(size)[indexes ^ (1 << wire)]
            mixer = (math.cos(beta) * np.eye(size) - 1j * math.sin(beta) * x) @ mixer
        unitary = mixer @ cost @ unitary
    return unitary


def _count_statistics(record):
    assert [g["group"] for g in record["groups"]] == ["all_z"]
    group = record["groups"][0]
    counts, shots, values = group["counts"], group["shots"], _cut_values()
    assert sum(counts.values()) == shots
    assert all(len(bits) == 4 and not set(bits) - {"0", "1"}
               and isinstance(number, int) and not isinstance(number, bool) and number > 0
               for bits, number in counts.items())
    mean = sum(number * values[int(bits, 2)] for bits, number in counts.items()) / shots
    variance = sum(number * (values[int(bits, 2)] - mean) ** 2
                   for bits, number in counts.items()) / (shots - 1)
    standard_error = math.sqrt(variance / shots)
    assert group["estimate"] == pytest.approx(mean, abs=1e-12)
    assert group["sample_variance"] == pytest.approx(variance, abs=1e-12)
    assert group["estimated_standard_error"] == pytest.approx(standard_error, abs=1e-12)
    assert record["objective_estimate"] == pytest.approx(-mean, abs=1e-12)
    assert record["estimated_standard_error"] == pytest.approx(standard_error, abs=1e-12)
    assert record["total_shots"] == shots
    return mean, standard_error


def _replay_saved_path(run):
    """Replay decisions from integer counts, without a new sampled draw."""
    evaluations, point = run["evaluations"], _wrap([0.6, 0.4, 0.6, 0.4])
    assert [e["index"] for e in evaluations] == list(range(109))
    assert evaluations[0]["role"] == "initial"
    np.testing.assert_allclose(evaluations[0]["parameters"], point, atol=1e-14, rtol=0)
    index, accepted_indexes, current = 1, set(), evaluations[0]
    for sweep, iteration in enumerate(run["iterations"]):
        step, shots = ((0.4, 260), (0.2, 520), (0.1, 1040))[sweep // 4]
        assert iteration["index"] == sweep and iteration["step"] == step
        assert iteration["shots_per_group"] == shots
        assert iteration["incumbent_evaluation"] == index
        proposal_indexes = list(range(index + 1, index + 9))
        assert iteration["proposal_evaluations"] == proposal_indexes
        incumbent = evaluations[index]
        assert incumbent["role"] == "incumbent" and not incumbent["accepted"]
        np.testing.assert_allclose(incumbent["parameters"], point, atol=1e-14, rtol=0)
        np.testing.assert_allclose(iteration["parameters_before"], point, atol=1e-14, rtol=0)
        proposals = [evaluations[p] for p in proposal_indexes]
        for offset, proposal in enumerate(proposals):
            axis, direction = offset // 2, -1 if offset % 2 == 0 else 1
            assert proposal["role"] == "proposal"
            assert (proposal["axis"], proposal["direction"]) == (axis, direction)
            trial = point.copy()
            trial[axis] += direction * step
            np.testing.assert_allclose(proposal["parameters"], _wrap(trial), atol=1e-14, rtol=0)
        best = min(proposals, key=lambda e: e["objective_estimate"])
        margin = 2 * math.hypot(incumbent["estimated_standard_error"], best["estimated_standard_error"])
        improvement = incumbent["objective_estimate"] - best["objective_estimate"]
        accepted = improvement > margin
        assert accepted == iteration["accepted"]
        assert best["index"] == iteration["best_proposal_evaluation"]
        assert margin == pytest.approx(iteration["comparison_margin"], abs=1e-14)
        assert improvement == pytest.approx(iteration["sampled_improvement"], abs=1e-14)
        for proposal in proposals:
            assert proposal["accepted"] == (accepted and proposal["index"] == best["index"])
        if accepted:
            accepted_indexes.add(best["index"])
        current = best if accepted else incumbent
        point = np.asarray(current["parameters"])
        np.testing.assert_allclose(iteration["parameters_after"], point, atol=1e-14, rtol=0)
        index += 9
    assert index == 109 and len(run["iterations"]) == 12
    np.testing.assert_allclose(run["parameters"], point, atol=1e-14, rtol=0)
    assert run["accepted_proposals"] == len(accepted_indexes) == 4
    assert run["rejected_proposals"] == 96 - len(accepted_indexes) == 92
    assert {e["index"] for e in evaluations if e["accepted"]} == accepted_indexes
    assert run["final_training_estimate"] == current["objective_estimate"]
    assert run["final_training_standard_error"] == current["estimated_standard_error"]
    assert run["stop_reason"] == "fixed_schedule_exhausted"
    assert run["convergence_certified"] is False


def _primitive_state(circuit, *, initial_index=0):
    """Independent four-wire primitive simulation, preserving global phase.

    It accepts canonical initial resets but rejects a reset after coherent work;
    the selected source has exactly one initial reset on every wire.
    """
    n = circuit.num_qubits
    assert n == 4
    size, indexes = 1 << n, np.arange(1 << n)
    state = np.zeros(size, dtype=complex)
    assert not initial_index or "reset" not in circuit.count_ops()
    state[initial_index] = np.exp(1j * float(circuit.global_phase))
    coherent_started, outputs_started = False, False
    initialized, output_map = set(), {}
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if name == "barrier":
            continue
        if name == "reset":
            assert not coherent_started and not outputs_started
            initialized.update(wires)
            continue
        if name == "measure":
            outputs_started = True
            output_map[circuit.find_bit(item.clbits[0]).index] = wires[0]
            continue
        assert not outputs_started
        coherent_started = True
        if name == "cx":
            control, target = wires
            permutation = indexes.copy()
            permutation[(indexes & (1 << control)) != 0] ^= 1 << target
            state = state[permutation]
            continue
        assert len(wires) == 1
        wire = wires[0]
        zeros, ones = indexes[(indexes & (1 << wire)) == 0], indexes[(indexes & (1 << wire)) == 0] ^ (1 << wire)
        matrices = {
            "h": np.asarray([[1, 1], [1, -1]], complex) / math.sqrt(2),
            "x": np.asarray([[0, 1], [1, 0]], complex),
            "y": np.asarray([[0, -1j], [1j, 0]], complex),
            "z": np.diag([1, -1]), "s": np.diag([1, 1j]),
            "sdg": np.diag([1, -1j]), "t": np.diag([1, np.exp(1j * math.pi / 4)]),
            "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
        }
        if name in matrices:
            matrix = matrices[name]
        else:
            theta = float(item.operation.params[0])
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            matrix = ({"rx": np.asarray([[c, -1j * s], [-1j * s, c]]),
                       "ry": np.asarray([[c, -s], [s, c]]),
                       "rz": np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])})[name]
        a, b = state[zeros].copy(), state[ones].copy()
        state[zeros], state[ones] = matrix[0, 0] * a + matrix[0, 1] * b, matrix[1, 0] * a + matrix[1, 1] * b
    assert not initialized or initialized == set(range(n))
    assert set(output_map) == set(range(n)) and set(output_map.values()) == set(range(n))
    probabilities = np.zeros(size)
    for wire_value, probability in enumerate(np.abs(state) ** 2):
        classical_value = sum(((wire_value >> output_map[bit]) & 1) << bit for bit in range(n))
        probabilities[classical_value] += probability
    return state, probabilities, output_map


def _independent_matrix(operations, phase, gate_cache=None):
    """Full coherent matrix from tensor-product primitives and output map.

    This independently applies dense tensor gates; it does not call either
    producer's matrix checker, statevector, compiler, or synthesis routines.
    """
    cache = {} if gate_cache is None else gate_cache
    indexes = np.arange(16)
    matrix, initial, outputs, quantum, measured = np.eye(16, dtype=complex), [], {}, False, False
    identity = np.eye(2, dtype=complex)
    fixed = {"h": np.asarray([[1, 1], [1, -1]], complex) / math.sqrt(2),
        "x": np.asarray([[0, 1], [1, 0]], complex), "y": np.asarray([[0, -1j], [1j, 0]], complex),
        "z": np.diag([1, -1]), "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
        "t": np.diag([1, np.exp(1j * math.pi / 4)]), "tdg": np.diag([1, np.exp(-1j * math.pi / 4)])}
    for name, wires, clbits, params in operations:
        if name == "barrier":
            continue
        if name == "reset":
            assert not quantum and not measured and wires == [len(initial)] and not clbits and not params
            initial.extend(wires)
            continue
        if name == "measure":
            measured = True
            assert len(wires) == len(clbits) == 1 and not params and clbits[0] not in outputs
            outputs[clbits[0]] = wires[0]
            continue
        assert not measured and initial == [0, 1, 2, 3]
        quantum = True
        key = (name, tuple(wires), tuple(float(p).hex() for p in params))
        if key not in cache:
            if name == "cx":
                assert len(wires) == 2 and len(set(wires)) == 2 and not params
                permutation = indexes.copy()
                permutation[(indexes & (1 << wires[0])) != 0] ^= 1 << wires[1]
                gate = np.eye(16, dtype=complex)[permutation]
            else:
                assert len(wires) == 1 and not clbits
                if name in fixed:
                    assert not params
                    local = fixed[name]
                else:
                    assert name in {"rx", "ry", "rz"} and len(params) == 1 and math.isfinite(params[0])
                    theta = params[0]
                    pauli = fixed[{"rx": "x", "ry": "y", "rz": "z"}[name]]
                    local = math.cos(theta / 2) * identity - 1j * math.sin(theta / 2) * pauli
                gate = np.ones((1, 1), dtype=complex)
                for wire in reversed(range(4)):
                    gate = np.kron(gate, local if wire == wires[0] else identity)
            cache[key] = gate
        matrix = cache[key] @ matrix
    assert initial == [0, 1, 2, 3] and set(outputs) == set(range(4)) and set(outputs.values()) == set(range(4))
    permutation = [sum(((bits >> bit) & 1) << outputs[bit] for bit in range(4)) for bits in range(16)]
    return np.exp(1j * phase) * matrix[permutation], outputs


def test_archived_parent_producers_and_backend_are_unchanged():
    assert sha(PARENT / "optimization.py") == PARENT_SOURCE_SHA
    assert sha(PARENT / "generate_optimization_diagnostics.py") == PARENT_GENERATOR_SHA
    assert sha(ROOT / "src/openquantum/instrument_resources.py") == BACKEND_SHA
    assert sha(ROOT / "requirements-lock.txt") == LOCK_SHA
    freeze = json.loads((PARENT / "freeze.json").read_text())
    assert freeze["identity"]["implementation_sha256"] == PARENT_SOURCE_SHA
    assert freeze["identity"]["generator_sha256"] == PARENT_GENERATOR_SHA


def test_selected_run_is_first_genuinely_supported_four_wire_qaoa(parent_study, selected):
    supported = [(case["case_id"], run["run_id"]) for case in parent_study["cases"]
                 for run in case["runs"]
                 if run["validation"]["quality_bound"]["target_status"] == "supported"]
    assert supported == [(CASE_ID, f"qaoa-cycle-4-p2-alternate-seed-{seed}") for seed in (7, 19, 41)]
    assert supported[0][1] == RUN_ID
    case, run = selected
    assert (case["algorithm"], case["n_qubits"], case["p"]) == ("qaoa", 4, 2)
    assert case["edges"] == [list(edge) for edge in EDGES]
    assert (run["start_id"], run["seed"], run["stream_id"]) == ("alternate", 7, 33)
    assert run["initial"] == [0.6, 0.4, 0.6, 0.4]
    assert run["target_met_audit"] is True and run["exact_diagnostics_used_for_optimization"] is False


def test_selected_counts_replay_the_complete_saved_polling_path(selected):
    case, run = selected
    for record in [*run["evaluations"], run["validation"]]:
        _count_statistics(record)
    _replay_saved_path(run)
    assert run["validation"]["parameters"] == run["parameters"]


def test_selected_preparation_readout_and_source_ledger_includes_rejections(selected):
    _, run = selected
    allocations = [128] + [shots for shots in (260, 520, 1040) for _ in range(36)]
    assert len(allocations) == len(run["evaluations"]) == 109
    assert [e["groups"][0]["shots"] for e in run["evaluations"]] == allocations
    assert sum(allocations) == run["total_shots"] == 65648
    assert run["validation"]["total_shots"] == 16384
    assert run["total_shots_including_validation"] == run["quantum_preparations"] == 82032
    weighted = Counter()
    for phase, calls in ((0, run["evaluations"]), (1, [run["validation"]])):
        for index, call in enumerate(calls):
            group = call["groups"][0]
            assert group["rng"]["stream_key"] == [7, 33, phase, index, 0]
            assert group["circuit_gate_counts"] == SOURCE_COUNTS
            assert group["measured_qubit_readouts"] == 4 * group["shots"]
            weighted.update({name: count * group["shots"] for name, count in SOURCE_COUNTS.items() if name != "barrier"})
    assert dict(weighted) == WEIGHTED_SOURCE
    assert sum(value for name, value in weighted.items() if name != "measure") == 2953152
    assert weighted["measure"] == 328128
    assert len({tuple(c["parameters"]) for c in [*run["evaluations"], run["validation"]]}) == 54


def test_selected_holdout_support_and_ideal_quality_are_distinct(selected):
    _, run = selected
    mean, standard_error = _count_statistics(run["validation"])
    bound = run["validation"]["quality_bound"]
    radius = math.sqrt(math.log(2 / (0.05 / 48)) / (2 * 16384))
    assert mean / 4 == pytest.approx(0.9884033203125, abs=1e-15)
    assert bound["simultaneous_radius"] == pytest.approx(radius, abs=1e-15)
    assert bound["normalized_standard_error"] == pytest.approx(standard_error / 4, abs=1e-15)
    assert bound["simultaneous_interval"] == pytest.approx([mean / 4 - radius, mean / 4 + radius], abs=1e-15)
    assert bound["simultaneous_interval"][0] == pytest.approx(0.9732140034673423, abs=1e-15)
    assert bound["simultaneous_interval"][0] >= 0.80 and radius <= 0.05
    assert bound["target_status"] == "supported" and bound["precision_supported"] is True
    probabilities = np.abs(_qaoa_state(run["parameters"])) ** 2
    cut_mean = float(probabilities @ _cut_values())
    assert cut_mean / 4 == pytest.approx(0.9881989495244483, abs=1e-13)
    assert cut_mean == pytest.approx(run["diagnostic"]["exact_expected_cut"], abs=1e-13)
    assert probabilities[[5, 10]].sum() == pytest.approx(run["diagnostic"]["exact_optimal_cut_probability"], abs=1e-13)


def test_negative_controls_and_all_vqe_inconclusive_labels_remain_in_parent(parent_study):
    case = next(c for c in parent_study["cases"] if c["case_id"] == CASE_ID)
    assert [r["validation"]["quality_bound"]["target_status"] for r in case["runs"] if r["start_id"] == "original"] == ["missed"] * 3
    vqe = [r for c in parent_study["cases"] if c["algorithm"] == "vqe" for r in c["runs"]]
    assert len(vqe) == 12
    assert Counter(r["validation"]["quality_bound"]["target_status"] for r in vqe) == {"missed": 6, "inconclusive": 6}
    ideal_hits = [r for r in vqe if r["target_met_audit"]]
    assert len(ideal_hits) == 2
    assert all(r["validation"]["quality_bound"]["target_status"] == "inconclusive" for r in ideal_hits)


def test_whole_transcript_realization_and_selection_safe_statistical_budget(selected):
    _, run = selected
    # Reserve the complete prior family because the supported run is selected
    # from its records; a nominal one-group allocation cannot replace it.
    sampling = Fraction(4, 100)
    realization = Fraction(5, 1000)
    physical = factory = Fraction(25, 10000)
    assert sampling + realization + physical + factory == Fraction(5, 100)
    eta = realization / 82032
    allocations = [e["groups"][0]["shots"] for e in run["evaluations"]] + [16384]
    assert sum(Fraction(shots) * eta for shots in allocations) == realization
    assert eta == Fraction(1, 16406400)
    # The source contains 16 rotations. Its per-rotation error remains above
    # the frozen numerical guard's four-guard minimum, even before routing.
    assert float(eta) / 16 > 4e-12
    radius = math.sqrt(math.log(2 / float(sampling / 48)) / (2 * 16384))
    estimate = run["validation"]["quality_bound"]["normalized_estimate"]
    assert radius < 0.05
    assert estimate - radius - float(eta) > 0.80
    # Whole-transcript TV is a distance/coupling reserve, not an asserted
    # hardware realization event or an adaptive-path correctness theorem.
    assert run["convergence_certified"] is False


@pytest.fixture(scope="module")
def parent_manifest():
    return json.loads((PARENT / "circuit-manifest.json").read_text())


def test_every_selected_group_binds_to_archived_qasm_and_transcript(selected, parent_manifest, parent_producer):
    case, run = selected
    groups = [g for g in parent_manifest["groups"] if g["run_id"] == RUN_ID]
    assert len(groups) == 110
    unique = {entry["sha256"]: entry for entry in parent_manifest["unique_circuits"]}
    source_hashes = set()
    for group, (phase, index, record) in zip(groups, [("training", i, e) for i, e in enumerate(run["evaluations"])]
            + [("validation", 0, run["validation"])], strict=True):
        assert (group["phase"], group["evaluation_index"], group["group_index"]) == (phase, index, 0)
        measured = record["groups"][0]
        assert group["parameters"] == record["parameters"]
        assert group["counts"] == measured["counts"] and group["shots"] == measured["shots"]
        assert group["rng"] == measured["rng"]
        source_hash = group["source_qasm_sha256"]
        path = PARENT / group["source_qasm_file"]
        assert sha(path) == source_hash and unique[source_hash]["file"] == group["source_qasm_file"]
        circuit = parent_producer.quality_scaling_group_circuit(case, record["parameters"], "all_z")
        assert qasm3.dumps(circuit).encode() == path.read_bytes()
        assert dict(circuit.count_ops()) == SOURCE_COUNTS
        state, actual, mapping = _primitive_state(circuit)
        assert mapping == dict(enumerate(range(4)))
        np.testing.assert_allclose(actual, np.abs(_qaoa_state(record["parameters"])) ** 2, atol=1e-12, rtol=0)
        assert abs(np.vdot(state, state).real - 1) < 1e-12
        source_hashes.add(source_hash)
    assert len(source_hashes) == 54


def test_all_source_operators_preserve_the_explicit_cost_identity_phase(selected, parent_producer):
    case, run = selected
    parameters = {tuple(c["parameters"]) for c in [*run["evaluations"], run["validation"]]}
    exact_rotation_keys = set()
    for point in parameters:
        source = parent_producer.quality_scaling_group_circuit(case, point, "all_z")
        actual = np.column_stack([_primitive_state(source, initial_index=column)[0] for column in range(16)])
        expected = np.exp(2j * (point[0] + point[2])) * _qaoa_unitary(point)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        np.testing.assert_allclose(actual.conj().T @ actual, np.eye(16), atol=2e-14, rtol=0)
        exact_rotation_keys.update((item.operation.name, float(item.operation.params[0]).hex())
                                   for item in source.data if item.operation.name in {"rx", "ry", "rz"})
    assert len(parameters) == 54 and len(exact_rotation_keys) == 22
    assert 2 * (run["parameters"][0] + run["parameters"][2]) == pytest.approx(4.0, abs=1e-14)


@pytest.fixture(scope="module")
def generator():
    path = ROOT / "scripts/generate_optimization_compiled.py"
    if not path.is_file():
        pytest.skip("Owned compiled-replay generator has not been written yet")
    spec = importlib.util.spec_from_file_location("compiled_generator_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_store_enforces_immutable_bytes_and_total_output_limit(tmp_path, generator):
    store = generator.ArtifactStore(tmp_path / "artifacts", byte_cap=10)
    store.write("frozen.json", b"123456", immutable=True)
    store.write("frozen.json", b"123456", immutable=True)
    with pytest.raises(ValueError, match="Preserved"):
        store.write("frozen.json", b"654321", immutable=True)
    with pytest.raises(ValueError, match="cap"):
        store.write("additional.json", b"12345")
    assert (store.root / "frozen.json").read_bytes() == b"123456"
    assert not (store.root / "additional.json").exists()
    store.write("derived.json", b"1234")
    assert sum(entry["bytes"] for entry in store.inventory().values()) == 10
    store.write("derived.json", b"12")
    assert sum(entry["bytes"] for entry in store.inventory().values()) == 8


def test_public_store_rejects_traversal_and_linked_targets(tmp_path, generator):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    for relative in ("../private.json", "/private.json", "nested/../../private.json"):
        with pytest.raises(ValueError, match="relative"):
            store.write(relative, b"test")
    target = tmp_path / "private"
    target.mkdir()
    (store.root / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        store.write("link/exposed.json", b"test")
    assert not (target / "exposed.json").exists()
    with pytest.raises(ValueError, match="linked"):
        store.inventory()


def test_public_output_root_and_ancestor_symlinks_are_rejected_before_resolution(tmp_path, generator):
    private = tmp_path / "private"
    private.mkdir()
    alias = tmp_path / "linked-output"
    alias.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="link"):
        generator.validate_output(alias, tmp_path / "cache")
    with pytest.raises(ValueError, match="link"):
        generator.validate_output(alias / "nested", tmp_path / "cache")
    with pytest.raises(ValueError, match="link"):
        generator.ArtifactStore(alias)
    assert list(private.iterdir()) == []


@pytest.mark.parametrize("kind,key,status", [("physical", "class", "completed"),
                                            ("synthesis", "other", "completed"),
                                            ("synthesis", "class", "pending")])
def test_completed_checkpoint_identity_kind_key_and_status_are_bound(tmp_path, generator, kind, key, status):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    identity = {"recipe_sha256": "a" * 64, "selected_transcript_sha256": "b" * 64,
                "source_sha256": {"optimization_compiled.py": "c" * 64}}
    record = {"actual_synthesis_sequences_generated": True, "complete_preparations": 82032}
    generator.save_checkpoint(store, "synthesis", "class", identity, record)
    assert generator.load_checkpoint(store, "synthesis", "class", identity) == record
    path = store.target("checkpoints/synthesis/class.json")
    saved = json.loads(path.read_text())
    saved.update(kind=kind, key=key, status=status)
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Checkpoint"):
        generator.load_checkpoint(store, "synthesis", "class", identity)


def test_checkpoint_producer_or_transcript_drift_is_rejected(tmp_path, generator):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    recipe, transcript = {"fixed": True}, {"run_id": RUN_ID, "parameters": [0.6, 0.4] * 2}
    info = {"source_sha256": {"optimization_compiled.py": "a" * 64}}
    identity = generator.checkpoint_identity(recipe, info, transcript)
    generator.save_checkpoint(store, "synthesis", "class", identity, {"fixed": True})
    changed_transcript = deepcopy(transcript)
    changed_transcript["parameters"][0] = math.nextafter(0.6, 1.0)
    changed_id = generator.checkpoint_identity(recipe, info, changed_transcript)
    assert changed_id != identity
    with pytest.raises(ValueError, match="Checkpoint"):
        generator.load_checkpoint(store, "synthesis", "class", changed_id)
    changed_id = deepcopy(identity)
    changed_id["source_sha256"]["optimization_compiled.py"] = "d" * 64
    with pytest.raises(ValueError, match="Checkpoint"):
        generator.load_checkpoint(store, "synthesis", "class", changed_id)


def test_compressed_checkpoints_are_deterministic_immutable_and_unambiguous(tmp_path, generator):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    identity = {"recipe_sha256": "a" * 64, "source_sha256": {"core": "b" * 64}}
    record = {"actual_operations": [["h", [0], [], []]] * 100, "global_phase": 0.0}
    generator.save_checkpoint(store, "synthesis", "class", identity, record, compressed=True)
    path = store.target("checkpoints/synthesis/class.json.gz")
    frozen_bytes = path.read_bytes()
    assert int.from_bytes(frozen_bytes[4:8], "little") == 0
    decoded = json.loads(gzip.decompress(frozen_bytes))
    assert decoded["record"] == record
    assert generator.load_checkpoint(store, "synthesis", "class", identity) == record
    generator.save_checkpoint(store, "synthesis", "class", identity, record, compressed=True)
    assert path.read_bytes() == frozen_bytes
    with pytest.raises(ValueError, match="Preserved"):
        generator.save_checkpoint(store, "synthesis", "class", identity, {"changed": True}, compressed=True)
    generator.save_checkpoint(store, "synthesis", "class", identity, record)
    with pytest.raises(ValueError, match="Ambiguous"):
        generator.load_checkpoint(store, "synthesis", "class", identity)


def test_producer_export_uses_matching_frozen_parent_and_backend(generator):
    sources = generator.producer_sources(b"# isolated fixture core\n")
    assert hashlib.sha256(sources["parent_optimization.py"]).hexdigest() == PARENT_SOURCE_SHA
    assert hashlib.sha256(sources["instrument_resources.py"]).hexdigest() == BACKEND_SHA
    assert hashlib.sha256(sources["requirements-lock.txt"]).hexdigest() == LOCK_SHA
    assert sources["optimization_compiled.py"] == b"# isolated fixture core\n"
    assert sources["generate_optimization_compiled.py"] == (ROOT / "scripts/generate_optimization_compiled.py").read_bytes()


def test_generator_keeps_output_and_private_native_cache_out_of_other_archives(tmp_path, generator):
    for path in (ROOT / "src", PARENT, ROOT / "assets/generated/optimization/quality-scaling",
                 ROOT / "assets/generated/instrument-resources", ROOT / ".git"):
        with pytest.raises(ValueError):
            generator.validate_output(path, tmp_path / "cache")
    with pytest.raises(ValueError, match="private"):
        generator.validate_output(OUTPUT, OUTPUT / "cache")
    with pytest.raises(ValueError, match="private"):
        generator.validate_output(OUTPUT, ROOT / "private-cache")
    assert generator.validate_output(OUTPUT, tmp_path / "cache") == OUTPUT


@pytest.fixture(scope="module")
def core():
    path = ROOT / "src/openquantum/optimization_compiled.py"
    if not path.is_file():
        pytest.skip("Owned compiled core has not been written yet")
    spec = importlib.util.spec_from_file_location("compiled_core_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compiled_inputs(core):
    return core.load_compiled_inputs(ROOT)


def test_core_input_binding_contains_only_the_immutable_selected_trajectory(core, compiled_inputs, selected):
    case, run = selected
    assert compiled_inputs["case"] == {key: value for key, value in case.items() if key != "runs"}
    assert compiled_inputs["run"] == run
    assert compiled_inputs["bindings"]["parent_raw_sha256"] == PARENT_RAW_SHA
    assert compiled_inputs["bindings"]["parent_implementation_sha256"] == PARENT_SOURCE_SHA
    assert compiled_inputs["bindings"]["instrument_backend_sha256"] == BACKEND_SHA
    assert compiled_inputs["selection_scope"].startswith("data-dependent")
    assert compiled_inputs["negative_controls"]["physical_workload_scope"].startswith("only selected run")


def test_compiled_recipe_preserves_output_failure_and_materialization_contract(core, compiled_inputs):
    recipe = core.compiled_recipe(compiled_inputs)
    assert recipe["run_id"] == RUN_ID and recipe["bindings"] == compiled_inputs["bindings"]
    assert recipe["output"] == {"task": "fixed final-state mean cut / enumerated optimum",
        "quality_at_least": 0.8, "normalized_precision": 0.05, "success_at_least": 0.95,
        "n_qubits": 4, "classical_optimum": 4}
    budget = recipe["budget"]
    assert budget["sampling_failure_family"] == 0.04 and budget["sampling_final_group_count"] == 48
    assert budget["sampling_failure_per_group"] == 0.04 / 48
    assert budget["realization_static_transcript_tv"] == 0.005
    assert budget["physical_failure_whole_workload"] == budget["factory_failure_whole_workload"] == 0.0025
    assert budget["reserved_total"] == 0.05
    assert budget["per_preparation_operator_allowance"] == float(Fraction(1, 16406400))
    assert budget["conditional_ratio_lower"] > 0.8 and budget["conditional_precision_radius"] < 0.05
    assert recipe["planned"]["source_weighted_counts"] == WEIGHTED_SOURCE
    assert recipe["planned"]["total_preparations"] == 82032
    assert recipe["planned"]["explicit_source_initializations"] == recipe["planned"]["source_qubit_readouts"] == 328128
    bounds = recipe["bounds"]
    assert bounds["source_qubits"] == bounds["validation_qubits"] == 4
    assert bounds["source_operations"] == 128 and bounds["routed_operations"] == 1024
    assert bounds["synthesized_operations"] == bounds["validation_operations"] == 10000
    assert bounds["total_synthesized_operations"] == 54 * bounds["synthesized_operations"]
    assert bounds["total_synthesis_native_calls"] == 54 * bounds["synthesis_native_calls_per_class"] == 3456
    assert bounds["physical_top_level_calls"] == 64
    assert bounds["factory_frontier_queries"] == 1536
    assert bounds["native_trace_queries"] == bounds["factory_frontier_candidates"] == 52224
    assert bounds["artifact_bytes"] == 16 * 1024 * 1024
    assert recipe["compiler"]["preserve_full_unitary"] is True
    assert recipe["compiler"]["native_synthesis_seed"] is None


def test_exact_source_partition_preserves_all_counts_parameters_and_rejections(core, compiled_inputs, parent_manifest):
    recipe = core.compiled_recipe(compiled_inputs)
    classes = core.compiled_source_classes(compiled_inputs, recipe)
    expected_groups = [g for g in parent_manifest["groups"] if g["run_id"] == RUN_ID]
    archived_by_hash = {}
    for group in expected_groups:
        archived_by_hash.setdefault(group["source_qasm_sha256"], []).append(group)
    assert [c["class_id"] for c in classes] == list(archived_by_hash)
    assert len(classes) == 54 and sum(c["shots"] for c in classes) == 82032
    assert sum(len(c["batches"]) for c in classes) == 110
    for source in classes:
        assert hashlib.sha256(source["source_qasm"].encode()).hexdigest() == source["class_id"]
        archived = archived_by_hash[source["class_id"]]
        assert source["parameters"] == archived[0]["parameters"]
        assert source["shots"] == sum(g["shots"] for g in archived)
        assert source["source_qasm"].encode() == (PARENT / archived[0]["source_qasm_file"]).read_bytes()
        assert [(b["phase"], b["evaluation_index"], b["shots"]) for b in source["batches"]] == [
            ("holdout" if g["phase"] == "validation" else "training", g["evaluation_index"], g["shots"])
            for g in archived]
    assert sum(b["accepted"] for c in classes for b in c["batches"]) == 4
    assert sum(b["role"] == "proposal" and not b["accepted"] for c in classes for b in c["batches"]) == 92


@pytest.mark.parametrize("key,value", [("per_preparation_operator_allowance", 0.1),
                                     ("sampling_failure_family", 0.05),
                                     ("physical_failure_whole_workload", 0.01)])
def test_changed_recipe_is_rejected_before_any_synthesis(core, compiled_inputs, monkeypatch, key, value):
    recipe = core.compiled_recipe(compiled_inputs)
    source = core.compiled_source_classes(compiled_inputs, recipe)[0]
    recipe["budget"][key] = value
    _, backend = core._producers()
    def forbidden(*args, **kwargs):
        raise AssertionError("Recipe rejection must occur before any synthesis")
    monkeypatch.setattr(backend, "synthesize_instrument", forbidden)
    with pytest.raises(ValueError, match="recipe"):
        core.compile_source_class(source, recipe)


@pytest.mark.parametrize("mutation", ["parameters", "shots", "qasm", "batches"])
def test_changed_source_binding_is_rejected_before_synthesis(core, compiled_inputs, monkeypatch, mutation):
    recipe = core.compiled_recipe(compiled_inputs)
    source = deepcopy(core.compiled_source_classes(compiled_inputs, recipe)[0])
    if mutation == "parameters":
        source["parameters"][0] = math.nextafter(source["parameters"][0], math.inf)
    elif mutation == "shots":
        source["shots"] += 1
    elif mutation == "qasm":
        source["source_qasm"] += "// changed\n"
    else:
        source["batches"][0]["shots"] += 1
    _, backend = core._producers()
    def forbidden(*args, **kwargs):
        raise AssertionError("Source rejection must occur before any synthesis")
    monkeypatch.setattr(backend, "synthesize_instrument", forbidden)
    with pytest.raises(ValueError, match="source"):
        core.compile_source_class(source, recipe)


def test_compilation_prepends_explicit_all_wire_resets_before_backend_dispatch(core, compiled_inputs, monkeypatch):
    recipe = core.compiled_recipe(compiled_inputs)
    original = core.compiled_source_classes(compiled_inputs, recipe)[0]
    _, backend = core._producers()
    captured = {}
    class StopAtDispatch(Exception):
        pass
    def stop(source, allowance, *, limits, preserve_full_unitary):
        captured.update(source=source, allowance=allowance, limits=limits,
                        preserve_full_unitary=preserve_full_unitary)
        raise StopAtDispatch
    monkeypatch.setattr(backend, "synthesize_instrument", stop)
    original_native = backend.gridsynth_unitary
    with pytest.raises(StopAtDispatch):
        core.compile_source_class(original, recipe)
    assert backend.gridsynth_unitary is original_native
    source = captured["source"]
    assert captured["allowance"] == recipe["budget"]["per_preparation_operator_allowance"]
    assert captured["preserve_full_unitary"] is True
    assert captured["limits"].max_qubits == 4
    assert captured["limits"].source_operations == 128
    assert dict(source.count_ops()) == {**SOURCE_COUNTS, "reset": 4}
    assert [(item.operation.name, source.find_bit(item.qubits[0]).index) for item in source.data[:4]] == [
        ("reset", wire) for wire in range(4)]
    actual, probabilities, mapping = _primitive_state(source)
    assert mapping == dict(enumerate(range(4)))
    np.testing.assert_allclose(probabilities, np.abs(_qaoa_state(original["parameters"])) ** 2, atol=1e-12, rtol=0)
    assert abs(np.vdot(actual, actual).real - 1) < 1e-12


@pytest.mark.parametrize("mutation", ["drop_rotations", "wrong_global_phase"])
def test_independent_coherent_audit_rejects_incorrect_backend_results(core, compiled_inputs, monkeypatch, mutation):
    recipe = core.compiled_recipe(compiled_inputs)
    source_class = core.compiled_source_classes(compiled_inputs, recipe)[0]
    _, backend = core._producers()
    def incorrect(source, *args, **kwargs):
        compiled = source.copy_empty_like()
        for item in source.data:
            if mutation != "drop_rotations" or item.operation.name not in {"rx", "ry", "rz"}:
                compiled.append(item.operation, item.qubits, item.clbits)
        compiled.global_phase = source.global_phase + (0.13 if mutation == "wrong_global_phase" else 0.0)
        # A claimed tiny bound must not override the independent actual-matrix
        # check. The phase mutation leaves every readout probability unchanged.
        report = {"rotation_occurrences": 16,
                  "guarded_numerical_operator_error_upper_estimate": recipe["budget"]["per_preparation_operator_allowance"]}
        return backend.SynthesizedInstrument(source.copy(), source.copy(), compiled, report)
    monkeypatch.setattr(backend, "synthesize_instrument", incorrect)
    native = backend.gridsynth_unitary
    with pytest.raises(RuntimeError, match="coherent"):
        core.compile_source_class(source_class, recipe)
    assert backend.gridsynth_unitary is native


def test_primitive_stream_roundtrip_preserves_phase_wire_and_measurement_mapping(core):
    circuit = QuantumCircuit(4, 4)
    circuit.reset(range(4))
    circuit.global_phase = 0.317
    circuit.h(0)
    circuit.ry(-0.431, 2)
    circuit.cx(0, 3)
    circuit.rz(0.139, 1)
    circuit.t(3)
    circuit.tdg(2)
    for qubit, clbit in enumerate((3, 1, 0, 2)):
        circuit.measure(qubit, clbit)
    operations = core.circuit_operations(circuit)
    restored = core.circuit_from_operations(operations, 4, float(circuit.global_phase))
    assert core.circuit_operations(restored) == operations
    assert float(restored.global_phase) == float(circuit.global_phase)
    assert qasm3.dumps(restored) == qasm3.dumps(circuit)
    state, probabilities, mapping = _primitive_state(restored)
    assert mapping == {3: 0, 1: 1, 0: 2, 2: 3}
    coherent = restored.copy_empty_like()
    coherent.global_phase = restored.global_phase
    for item in restored.data:
        if item.operation.name != "reset":
            coherent.append(item.operation, item.qubits, item.clbits)
    expected = np.column_stack([_primitive_state(coherent, initial_index=i)[0] for i in range(16)])
    inverse_rows = [sum(((bits >> c) & 1) << mapping[c] for c in range(4)) for bits in range(16)]
    expected = expected[inverse_rows]
    actual = core.independent_instrument_operator(operations, float(restored.global_phase))
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    dense, dense_mapping = _independent_matrix(operations, float(restored.global_phase))
    assert dense_mapping == mapping
    np.testing.assert_allclose(actual, dense, atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.abs(actual[:, 0]) ** 2, probabilities, atol=1e-13, rtol=0)
    assert abs(np.vdot(state, state).real - 1) < 1e-13


@pytest.mark.parametrize("operations,n,phase", [([["initialize", [0], [], []]], 4, 0),
    ([["cx", [0, 0], [], []]], 4, 0), ([["h", [True], [], []]], 4, 0),
    ([["measure", [0], [4], []]], 4, 0), ([["h", [0], [], []]], 5, 0),
    ([["h", [0], [], []]], 4, math.nan), ([["h", [0], [], []]] * 10001, 4, 0)])
def test_saved_primitive_reconstruction_rejects_opaque_or_out_of_domain_inputs(core, operations, n, phase):
    with pytest.raises(ValueError):
        core.circuit_from_operations(operations, n, phase)


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
def test_saved_primitive_reconstruction_rejects_nonfinite_rotation_angles(core, angle):
    with pytest.raises(ValueError):
        core.circuit_from_operations([["rz", [0], [], [angle]]], 4, 0.0)


def _portable_inputs(root, inputs):
    (root / "selected-transcript.json").write_text(json.dumps(inputs, allow_nan=False))
    (root / "parent_optimization.py").write_bytes((PARENT / "optimization.py").read_bytes())
    (root / "instrument_resources.py").write_bytes((ROOT / "src/openquantum/instrument_resources.py").read_bytes())


def test_portable_selected_transcript_retains_pinned_parent_without_ancestor_git(tmp_path, core, compiled_inputs):
    _portable_inputs(tmp_path, compiled_inputs)
    try:
        actual = core.load_compiled_inputs(tmp_path)
        assert actual == compiled_inputs
        assert core.compiled_recipe(actual) == core.compiled_recipe(compiled_inputs)
    finally:
        core.load_compiled_inputs(ROOT)


@pytest.mark.parametrize("mutation", ["initial", "final", "counts", "target", "case", "negative"])
def test_portable_transcript_and_negative_control_corruption_is_rejected(tmp_path, core, compiled_inputs, mutation):
    changed = deepcopy(compiled_inputs)
    if mutation == "initial":
        changed["run"]["initial"][0] += 0.1
    elif mutation == "final":
        changed["run"]["parameters"][0] += 0.1
    elif mutation == "counts":
        counts = changed["run"]["evaluations"][0]["groups"][0]["counts"]
        counts[next(iter(counts))] += 1
    elif mutation == "target":
        changed["run"]["validation"]["quality_bound"]["target_status"] = "inconclusive"
    elif mutation == "case":
        changed["case"]["edges"][-1] = [0, 2]
    else:
        row = next(row for row in changed["negative_controls"]["sampled"]
                   if row["algorithm"] == "vqe" and row["ideal_target_met"])
        assert row["heldout_target_status"] == "inconclusive"
        row["heldout_target_status"] = "supported"
    _portable_inputs(tmp_path, changed)
    try:
        with pytest.raises(ValueError):
            core.load_compiled_inputs(tmp_path)
    finally:
        core.load_compiled_inputs(ROOT)


@pytest.mark.parametrize("filename", ["parent_optimization.py", "instrument_resources.py"])
def test_portable_producer_hash_changes_are_rejected(tmp_path, core, compiled_inputs, filename):
    _portable_inputs(tmp_path, compiled_inputs)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b"\n# mismatched producer\n")
    try:
        with pytest.raises(ValueError, match="producer"):
            core.load_compiled_inputs(tmp_path)
    finally:
        core.load_compiled_inputs(ROOT)


def _mock_physical_classes(backend):
    """Pure Clifford+T fixtures; not synthesized optimization trajectories."""
    def value(t_wire=None, reverse=False):
        circuit = QuantumCircuit(4, 4)
        circuit.reset(range(4))
        if reverse and t_wire is not None:
            circuit.t(t_wire)
        circuit.h(0)
        circuit.cx(0, 1)
        if not reverse and t_wire is not None:
            circuit.t(t_wire)
        circuit.measure(range(4), range(4))
        limits = {"max_qubits": 4, "source_operations": 128,
                  "synthesized_operations": 10000, "validation_operations": 10000,
                  "validation_branches": 256}
        return backend.SynthesizedInstrument(circuit.copy(), circuit.copy(), circuit,
            {"limits": limits, "actual_synthesis_sequences_generated": True})
    return [{"class_id": "a", "shots": 5, "synthesized": value(0)},
            {"class_id": "b", "shots": 7, "synthesized": value(0, reverse=True)},
            {"class_id": "c", "shots": 3, "synthesized": value(1)},
            {"class_id": "zero", "shots": 2, "synthesized": value()}]


def test_physical_preflight_groups_only_exact_event_costs_and_allocates_complete_workload(core, compiled_inputs):
    _, backend = core._producers()
    classes = _mock_physical_classes(backend)
    prepared = core.prepare_seventh_physical_workload(classes, backend=backend,
        expected_total_preparations=17, expected_total_t_states=15)
    assert prepared["total_preparations"] == 17 and prepared["total_t_states"] == 15
    assert len(prepared["groups"]) == 3
    mapping = prepared["compiled_class_mapping"]
    assert mapping["a"]["physical_class_id"] == mapping["b"]["physical_class_id"]
    assert mapping["a"]["physical_class_id"] != mapping["c"]["physical_class_id"]
    assert core.circuit_operations(classes[0]["synthesized"].circuit) != core.circuit_operations(classes[1]["synthesized"].circuit)
    assert prepared["global_factory_retry_cap"] == 2
    assert Fraction(15) * Fraction("0.01") ** 2 <= Fraction("0.0025")
    physical_allocations, factory_allocations = [], []
    for group in prepared["groups"]:
        expected_physical = Fraction("0.0025") * group["shots"] / 17
        expected_factory = Fraction("0.0025") * group["total_t_states"] / 15
        assert Fraction(group["physical_failure_allocation_exact"]) == expected_physical
        assert Fraction(group["factory_availability_allocation_exact"]) == expected_factory
        assert Fraction.from_float(group["physical_failure_budget"]) <= expected_physical
        assert Fraction(str(group["physical_failure_budget"])) <= expected_physical
        if group["total_t_states"]:
            assert Fraction(str(group["factory_availability_budget"])) <= expected_factory
            assert backend._factory_retry_cap(group["total_t_states"], 0.01, group["factory_availability_budget"]) == 2
        else:
            assert group["factory_availability_budget"] == 1e-300 and expected_factory == 0
        physical_allocations.append(expected_physical)
        factory_allocations.append(expected_factory)
    assert sum(physical_allocations) == sum(factory_allocations) == Fraction("0.0025")
    assert prepared["profile"]["physical_error_rate"] == 1e-4
    assert prepared["profile"]["code_cycle_time_ns"] == 1000
    assert prepared["profile"]["reaction_time_ns"] == 10000
    assert prepared["whole_physical_failure_budget"] == prepared["whole_factory_availability_budget"] == 0.0025
    assert "no unitary/topology equivalence" in prepared["grouping_scope"]


@pytest.mark.parametrize("mutation", ["duplicate", "shots", "total", "magic", "proxy", "non_clifford"])
def test_physical_preflight_rejects_incomplete_or_proxy_workloads_without_native_queries(core, compiled_inputs, monkeypatch, mutation):
    _, backend = core._producers()
    classes = _mock_physical_classes(backend)
    total, magic = 17, 15
    if mutation == "duplicate":
        classes[1]["class_id"] = classes[0]["class_id"]
    elif mutation == "shots":
        classes[0]["shots"] = True
    elif mutation == "total":
        total = 18
    elif mutation == "magic":
        magic = 16
    elif mutation == "proxy":
        classes[0]["synthesized"].report["actual_synthesis_sequences_generated"] = False
    else:
        classes[0]["synthesized"].circuit.rz(0.123, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Physical preflight must not launch a native estimator")
    monkeypatch.setattr(backend, "estimate_instrument", forbidden)
    with pytest.raises(ValueError):
        core.prepare_seventh_physical_workload(classes, backend=backend,
            expected_total_preparations=total, expected_total_t_states=magic)


def test_physical_preflight_cap_failure_preserves_incomplete_status(core, compiled_inputs):
    _, backend = core._producers()
    with pytest.raises(core.SeventhPhysicalLimit) as error:
        core.prepare_seventh_physical_workload(_mock_physical_classes(backend), backend=backend,
            expected_total_preparations=17, caps=core.SeventhPhysicalCaps(top_level_calls=1))
    assert error.value.audit["status"] == "preflight_cap_exceeded"
    assert error.value.audit["physical_classes"] == 3


@pytest.mark.parametrize("field,value", [("top_level_calls", 0), ("top_level_calls", 65),
    ("top_level_calls", True), ("factory_frontier_queries", 1537),
    ("factory_frontier_candidates", 52225), ("factory_pipeline_candidates", 52225),
    ("native_trace_queries", 52225), ("wall_seconds", 0), ("wall_seconds", 121),
    ("wall_seconds", math.nan), ("wall_seconds", math.inf)])
def test_physical_caps_reject_unbounded_or_invalid_values(core, field, value):
    with pytest.raises(ValueError):
        core.SeventhPhysicalCaps(**{field: value})


@pytest.mark.parametrize("field", ["top_level_calls", "factory_frontier_queries", "factory_frontier_candidates",
                                  "factory_pipeline_candidates", "native_trace_queries"])
def test_native_query_counter_rejects_next_query_before_exceeding_cap(core, field):
    audit = core._SeventhNativeAudit(core.SeventhPhysicalCaps(**{field: 1}))
    audit.charge(field)
    assert audit.counts[field] == 1
    with pytest.raises(core.SeventhPhysicalLimit) as error:
        audit.charge(field)
    assert audit.counts[field] == 1
    assert error.value.audit["status"] == "query_cap_exceeded"
    assert error.value.audit["counts"][field] == 1


def test_native_workflow_time_cap_rejects_without_an_unbounded_wait(core):
    audit = core._SeventhNativeAudit(core.SeventhPhysicalCaps(wall_seconds=0.01))
    audit.started -= 1
    with pytest.raises(core.SeventhPhysicalLimit) as error:
        audit.check()
    assert error.value.audit["status"] == "wall_cap_exceeded"


def test_generator_freeze_only_captures_exact_producers_and_never_dispatches_science(tmp_path, generator, core, monkeypatch):
    core_bytes = (ROOT / "src/openquantum/optimization_compiled.py").read_bytes()
    monkeypatch.setattr(generator, "load_core", lambda: (core, core_bytes))
    def forbidden(*args, **kwargs):
        raise AssertionError("Freeze-only must not synthesize or query a native estimator")
    monkeypatch.setattr(core, "compile_source_class", forbidden)
    monkeypatch.setattr(core, "run_compiled_physical", forbidden)
    result = generator.freeze(tmp_path / "frozen-bundle")
    assert result["execution_performed"] is False
    output = tmp_path / "frozen-bundle"
    before = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert before["optimization_compiled.py"] == core_bytes
    assert hashlib.sha256(before["parent_optimization.py"]).hexdigest() == PARENT_SOURCE_SHA
    assert hashlib.sha256(before["instrument_resources.py"]).hexdigest() == BACKEND_SHA
    assert result["artifact_bytes"] == sum(map(len, before.values())) < 16 * 1024 * 1024
    assert not (output / "checkpoints").exists() and not (output / "timing").exists()
    assert not (output / "raw-study.json.gz").exists()
    freeze = json.loads(before["freeze.json"])
    assert freeze["execution_performed"] is False
    assert freeze["identity"]["source_sha256"]["optimization_compiled.py"] == hashlib.sha256(core_bytes).hexdigest()
    classes = json.loads(gzip.decompress(before["source-classes.json.gz"]))
    assert len(classes) == 54 and sum(c["shots"] for c in classes) == 82032
    again = generator.freeze(output)
    assert again == result
    assert before == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_phase_budget_records_cumulative_incomplete_resume_effort(tmp_path, generator, monkeypatch):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    identity, clock = {"frozen": "identity"}, [0.0]
    monkeypatch.setattr(generator, "perf_counter", lambda: clock[0])
    with pytest.raises(RuntimeError, match="interrupted"):
        with generator.PhaseBudget(store, "synthesis", identity, 2.0) as budget:
            clock[0] += 0.75
            assert budget.check() == 0.75 and budget.remaining() == 1.25
            raise RuntimeError("interrupted")
    first = json.loads(store.target("timing/synthesis.json").read_text())
    assert first["status"] == "incomplete" and first["total_wall_seconds"] == 0.75
    with generator.PhaseBudget(store, "synthesis", identity, 2.0) as budget:
        clock[0] += 0.5
        assert budget.check() == 1.25
        budget.complete = True
    final = json.loads(store.target("timing/synthesis.json").read_text())
    assert final["status"] == "complete" and final["total_wall_seconds"] == 1.25
    assert [row["ordinal"] for row in final["invocations"]] == [1, 2]
    assert [row["wall_seconds"] for row in final["invocations"]] == [0.75, 0.5]


def test_phase_budget_cap_exceed_remains_incomplete_and_cannot_reset_on_resume(tmp_path, generator, monkeypatch):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    identity, clock = {"frozen": "identity"}, [0.0]
    monkeypatch.setattr(generator, "perf_counter", lambda: clock[0])
    with pytest.raises(RuntimeError, match="cap exceeded"):
        with generator.PhaseBudget(store, "synthesis", identity, 1.0) as budget:
            clock[0] = 1.25
            budget.check()
    record = json.loads(store.target("timing/synthesis.json").read_text())
    assert record["status"] == "cap_exceeded_incomplete" and record["total_wall_seconds"] == 1.25
    with pytest.raises(RuntimeError, match="exhausted"):
        with generator.PhaseBudget(store, "synthesis", identity, 1.0):
            raise AssertionError("Exhausted phase must not enter another science invocation")
    assert record == json.loads(store.target("timing/synthesis.json").read_text())


@pytest.mark.parametrize("mutation", ["running", "identity", "phase", "nonfinite", "sum", "ordinal"])
def test_phase_budget_rejects_unknown_or_inconsistent_previous_effort(tmp_path, generator, mutation):
    store = generator.ArtifactStore(tmp_path / "artifacts")
    identity = {"frozen": "identity"}
    record = {"schema_version": 1, "phase": "synthesis", "identity": identity, "status": "incomplete",
              "total_wall_seconds": 0.75, "invocations": [{"ordinal": 1, "wall_seconds": 0.75}]}
    if mutation == "running":
        record["status"] = "running"
    elif mutation == "identity":
        record["identity"] = {"changed": True}
    elif mutation == "phase":
        record["phase"] = "physical"
    elif mutation == "nonfinite":
        record["total_wall_seconds"] = math.inf
    elif mutation == "sum":
        record["total_wall_seconds"] = 0.25
    else:
        record["invocations"][0]["ordinal"] = 2
    store.target("timing/synthesis.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        generator.PhaseBudget(store, "synthesis", identity, 2.0)


def test_artifact_size_record_reconciles_all_bytes_including_itself(tmp_path, generator):
    store = generator.ArtifactStore(tmp_path / "artifacts", byte_cap=2000)
    store.write("kept.json", b"{\"fixture\":true}\n", immutable=True)
    record = generator.write_artifact_size(store)
    assert record == json.loads(store.target("artifact-size.json").read_text())
    inventory = store.inventory()
    assert record["total_bytes"] == sum(row["bytes"] for row in inventory.values())
    assert record["files"] == len(inventory) == 2
    assert record["total_bytes"] <= record["frozen_byte_cap"] == 2000
    assert record["parent_archive_copied"] is False and record["private_cache_included"] is False


def test_final_ensemble_json_preserves_complex_density_and_joint_probabilities(core):
    vector = np.asarray([1, 1j, -0.25, 0.5 + 0.5j], dtype=complex)
    vector /= np.linalg.norm(vector)
    density = np.outer(vector, vector.conj())
    original = {"reduced_before_output": density, "joint_probabilities": np.diag(density).real,
                "mean": np.float64(0.123), "scalar": np.complex128(0.2 + 0.3j),
                "outcome_clbits": (0, 1)}
    encoded = json.loads(json.dumps(core._json_values(original), allow_nan=False))
    reduced = encoded["reduced_before_output"]
    restored = np.asarray(reduced["real"]) + 1j * np.asarray(reduced["imag"])
    np.testing.assert_array_equal(restored, density)
    np.testing.assert_array_equal(encoded["joint_probabilities"], np.diag(density).real)
    assert encoded["mean"] == 0.123
    assert encoded["scalar"] == {"real": 0.2, "imag": 0.3}
    assert encoded["outcome_clbits"] == [0, 1]


@pytest.fixture(scope="module")
def compiled_checkpoints(core):
    paths = list((OUTPUT / "checkpoints/synthesis").glob("*.json.gz"))
    if len(paths) != 54:
        pytest.skip("Frozen generator has not completed all 54 actual synthesis checkpoints")
    inputs = core.load_compiled_inputs(ROOT)
    recipe = core.compiled_recipe(inputs)
    original = core.compiled_source_classes(inputs, recipe)
    frozen = json.loads((OUTPUT / "freeze.json").read_text())
    by_class = {}
    for path in paths:
        raw = gzip.decompress(path.read_bytes())
        assert len(raw) <= recipe["bounds"]["decoded_class_bytes"]
        checkpoint = json.loads(raw)
        assert checkpoint["status"] == "completed" and checkpoint["kind"] == "synthesis"
        assert checkpoint["identity"] == frozen["identity"]
        value = checkpoint["record"]
        assert value["class_id"] == checkpoint["key"] == path.name.removesuffix(".json.gz")
        assert value["class_id"] not in by_class
        by_class[value["class_id"]] = value
    assert set(by_class) == {c["class_id"] for c in original}
    return inputs, recipe, original, [by_class[c["class_id"]] for c in original]


def test_saved_actual_clifford_t_synthesis_preserves_entire_source_partition(compiled_checkpoints):
    _, recipe, original, compiled = compiled_checkpoints
    expected = recipe["planned"]
    totals = Counter()
    queries = operations = 0
    guarded, actual_tv = 0.0, 0.0
    for source, value in zip(original, compiled, strict=True):
        assert value["class_id"] == source["class_id"]
        assert value["parameters"] == source["parameters"]
        assert value["shots"] == source["shots"] and value["batches"] == source["batches"]
        assert value["original_source_qasm"] == source["source_qasm"]
        assert hashlib.sha256(value["original_source_qasm"].encode()).hexdigest() == value["class_id"]
        report = value["report"]
        assert report["actual_synthesis_sequences_generated"] is True
        assert report["formal_interval_certificate"] is False
        assert report["preserve_full_unitary"] is True and report["transpiler_qubits_initially_zero"] is False
        assert report["native_synthesis_backend"]["seed"] is None
        assert report["rotation_occurrences"] == 16
        assert report["operator_error_allowance"] == recipe["budget"]["per_preparation_operator_allowance"]
        assert report["guarded_numerical_operator_error_upper_estimate"] <= report["operator_error_allowance"]
        assert len(value["routed_operations"]) <= recipe["bounds"]["routed_operations"]
        assert len(value["clifford_t_operations"]) <= recipe["bounds"]["synthesized_operations"]
        counts = Counter(op[0] for op in value["clifford_t_operations"])
        assert dict(counts) == report["clifford_t_counts"]
        assert set(counts) <= {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "reset", "measure", "barrier"}
        assert counts["reset"] == counts["measure"] == 4
        assert all(abs(op[1][0] - op[1][1]) == 1 for op in value["routed_operations"] if op[0] == "cx")
        assert all(abs(op[1][0] - op[1][1]) == 1 for op in value["clifford_t_operations"] if op[0] == "cx")
        totals.update({name: count * value["shots"] for name, count in counts.items() if name != "barrier"})
        queries += value["query_counters"]["native_synthesis_calls"]
        operations += len(value["clifford_t_operations"])
        guarded += value["shots"] * report["guarded_numerical_operator_error_upper_estimate"]
        actual_tv += value["shots"] * value["checks"]["actual_probability_tv"]
    assert sum(c["shots"] for c in compiled) == expected["total_preparations"] == 82032
    assert sum(len(c["batches"]) for c in compiled) == expected["measured_groups"] == 110
    assert totals["measure"] == totals["reset"] == 328128
    assert totals["t"] + totals["tdg"] > 0
    assert queries <= recipe["bounds"]["total_synthesis_native_calls"]
    assert operations <= recipe["bounds"]["total_synthesized_operations"]
    assert guarded <= recipe["budget"]["realization_static_transcript_tv"] + 1e-12
    assert actual_tv <= guarded + 2e-11 * 82032


def test_first_attempt_actual_synthesis_checkpoints_are_byte_preserved(compiled_checkpoints):
    paths = sorted((OUTPUT / "checkpoints/synthesis").glob("*.json.gz"))
    payload = "".join(path.name + "\0" + sha(path) + "\n" for path in paths).encode()
    assert len(paths) == 54
    assert hashlib.sha256(payload).hexdigest() == FIRST_CT_INVENTORY_SHA


def test_all_saved_compiled_operators_and_joint_readouts_match_independent_tiny_checks(compiled_checkpoints):
    _, recipe, _, compiled = compiled_checkpoints
    gate_cache = {}
    final_checks = 0
    for value in compiled:
        matrices = []
        for prefix, count_key in (("source", "source_counts"), ("routed", "routed_counts"), ("clifford_t", "clifford_t_counts")):
            operations = value[f"{prefix}_operations"]
            assert Counter(op[0] for op in operations) == value["report"][count_key]
            matrix, mapping = _independent_matrix(operations, value[f"{prefix}_global_phase"], gate_cache)
            np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(16), atol=2e-11, rtol=0)
            assert set(mapping) == set(mapping.values()) == set(range(4))
            matrices.append(matrix)
        source, routed, actual = matrices
        expected = np.exp(2j * (value["parameters"][0] + value["parameters"][2])) * _qaoa_unitary(value["parameters"])
        np.testing.assert_allclose(source, expected, atol=2e-13, rtol=0)
        route_error = float(np.linalg.svd(source - routed, compute_uv=False)[0])
        actual_error = float(np.linalg.svd(source - actual, compute_uv=False)[0])
        eta = value["report"]["guarded_numerical_operator_error_upper_estimate"]
        assert route_error < 1e-11
        assert actual_error <= eta + 2e-11
        assert value["checks"]["routed_full_operator_error"] == pytest.approx(route_error, abs=2e-12)
        assert value["checks"]["actual_full_operator_error"] == pytest.approx(actual_error, abs=2e-12)
        probabilities = [np.abs(matrix[:, 0]) ** 2 for matrix in (source, actual)]
        tv = float(np.abs(probabilities[0] - probabilities[1]).sum() / 2)
        means = [float(p @ _cut_values()) for p in probabilities]
        assert tv <= eta + 2e-11 and abs(means[0] - means[1]) / 4 <= tv + 2e-12
        np.testing.assert_allclose(value["checks"]["source_probabilities"], probabilities[0], atol=2e-12, rtol=0)
        np.testing.assert_allclose(value["checks"]["compiled_probabilities"], probabilities[1], atol=2e-12, rtol=0)
        assert value["checks"]["source_mean_cut"] == pytest.approx(means[0], abs=2e-12)
        assert value["checks"]["compiled_mean_cut"] == pytest.approx(means[1], abs=2e-12)
        assert value["checks"]["actual_probability_tv"] == pytest.approx(tv, abs=2e-12)
        if any(b["phase"] == "holdout" for b in value["batches"]):
            final_checks += 1
            ensemble = value["checks"]["frozen_final_ensemble"]
            np.testing.assert_allclose(ensemble["joint_probabilities"], probabilities[1], atol=2e-11, rtol=0)
            reduced = ensemble["reduced_before_output"]
            density = np.asarray(reduced["real"]) + 1j * np.asarray(reduced["imag"])
            np.testing.assert_allclose(np.diag(density).real, probabilities[1], atol=2e-11, rtol=0)
            assert ensemble["acceptance_probability"] == pytest.approx(1.0, abs=2e-11)
            assert ensemble["mean"] == pytest.approx(means[1], abs=2e-11)
            assert means[1] / 4 >= recipe["output"]["quality_at_least"]
    assert final_checks == 1


@pytest.fixture(scope="module")
def saved_replay(core, completion):
    completed = OUTPUT / "completed-replay.json.gz"
    if completed.is_file():
        replay = completion.load_completed_replay(OUTPUT, validate=False)
        assert replay["status"] == "computed_actual_compiled_fixed_replay_conditional_physical_model"
        core.load_compiled_inputs(ROOT)
        return replay
    path = OUTPUT / "raw-study.json.gz"
    if not path.is_file():
        pytest.skip("Frozen generator has not completed the conditional physical replay")
    raw = gzip.decompress(path.read_bytes())
    replay = json.loads(raw)
    assert len(raw) <= replay["recipe"]["bounds"]["decoded_record_bytes"]
    assert replay["status"] == "computed_actual_compiled_fixed_replay_conditional_physical_model"
    core.load_compiled_inputs(ROOT)
    return replay


def _event_counts(operations):
    active, counts = set(), Counter()
    for name, wires, _, _ in operations:
        if name == "barrier":
            continue
        if name == "reset":
            counts["initial_reset" if wires[0] not in active else "reset"] += 1
        elif name in {"t", "tdg"}:
            counts[f"{name}_q{wires[0]}"] += 1
        else:
            counts[name] += 1
        active.update(wires)
    return dict(sorted(counts.items()))


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=32)
def _independent_readout_error(distance, physical_error):
    p = Fraction(str(physical_error))
    path = (1 - (1 - 2 * p) ** distance) / 2
    return float(sum((Fraction(math.comb(distance, count)) * path ** count * (1 - path) ** (distance - count)
                     for count in range((distance + 1) // 2, distance + 1)), Fraction()))


def _independent_macro_cost(profile, distance, name, factory_ns, retry):
    reaction = profile["reaction_time_ns"]
    dtime = distance * profile["code_cycle_time_ns"]
    initial = profile["physical_initialization_time_ns"] + dtime + reaction
    destructive = profile["physical_measurement_time_ns"] + reaction
    parity = 2 * initial + dtime + destructive
    cnot = 3 * parity + 2 * reaction
    if name == "initial_reset":
        return initial, 0, 1, 1
    if name == "reset":
        return parity + 2 * reaction + dtime, 1, 2, 5
    if name == "measure":
        return parity, 1, 2, 3
    if name == "cx":
        return cnot, 3, 6, 11
    if name == "trial_control":
        return reaction, 0, 0, 1
    if name.startswith(("t_q", "tdg_q")):
        wire, inverse = int(name.rsplit("_q", 1)[1]), int(name.startswith("tdg"))
        cnots = 3 * (wire + 1) + 1
        extra = (profile["physical_initialization_time_ns"] + profile["physical_measurement_time_ns"] + reaction)
        supply = retry * (factory_ns + profile["factory_feedback_rounds_reserved"] * extra + reaction)
        return (supply + cnots * cnot + destructive + reaction + initial + inverse * reaction,
                3 * cnots + 1, 6 * cnots + 1,
                11 * cnots + 3 + inverse + retry * (profile["factory_feedback_rounds_reserved"] + 1))
    assert name in {"h", "x", "y", "z", "s", "sdg"}
    return reaction, 0, 0, 1


def test_saved_physical_classes_preserve_actual_event_cost_partition_and_whole_budgets(saved_replay):
    replay = saved_replay
    physical, classes = replay["physical"], replay["source_classes"]
    assert physical["status"] == "complete_conditional_recorded_workload_common_configuration"
    assert physical["instrument_source_sha256"] == BACKEND_SHA
    assert physical["complete_preparations"] == 82032
    total_t = sum(c["shots"] * sum(op[0] in {"t", "tdg"} for op in c["clifford_t_operations"]) for c in classes)
    assert physical["actual_magic_states"] == replay["resources"]["actual_t_states"] == total_t
    assert physical["whole_physical_failure_budget"] == physical["whole_factory_availability_budget"] == 0.0025
    retry = physical["global_factory_retry_cap"]
    failure = Fraction(str(physical["profile"]["factory_pipeline_failure_bound"]))
    assert Fraction(total_t) * failure ** retry <= Fraction("0.0025")
    assert retry == 1 or Fraction(total_t) * failure ** (retry - 1) > Fraction("0.0025")
    expected = {}
    for source in classes:
        signature = {"logical_qubits": 4, "event_counts": _event_counts(source["clifford_t_operations"]),
                     "backend_sha256": BACKEND_SHA}
        identifier = _canonical_sha(signature)
        group = expected.setdefault(identifier, {"signature": signature, "shots": 0, "ids": [], "t": 0})
        group["shots"] += source["shots"]
        group["ids"].append(source["class_id"])
        group["t"] += source["shots"] * sum(op[0] in {"t", "tdg"} for op in source["clifford_t_operations"])
        assert physical["compiled_class_mapping"][source["class_id"]] == {
            "physical_class_id": identifier, "shots": source["shots"], "signature_sha256": identifier}
    assert len(physical["physical_classes"]) == len(expected) <= physical["caps"]["top_level_calls"]
    p_allocations, f_allocations = [], []
    for group in physical["physical_classes"]:
        original = expected[group["physical_class_id"]]
        assert group["signature"] == original["signature"]
        assert group["shots"] == original["shots"] and group["compiled_class_ids"] == original["ids"]
        assert group["total_t_states"] == original["t"]
        p_exact = Fraction("0.0025") * group["shots"] / 82032
        f_exact = Fraction("0.0025") * group["total_t_states"] / total_t
        assert Fraction(group["physical_failure_allocation_exact"]) == p_exact
        assert Fraction(group["factory_availability_allocation_exact"]) == f_exact
        assert Fraction.from_float(group["physical_failure_budget"]) <= p_exact
        if group["total_t_states"]:
            assert Fraction(str(group["factory_availability_budget"])) <= f_exact
        p_allocations.append(p_exact)
        f_allocations.append(f_exact)
    assert sum(p_allocations) == sum(f_allocations) == Fraction("0.0025")
    assert physical["profile"]["physical_error_rate"] == 1e-4
    assert physical["profile"]["physical_gate_time_ns"] == physical["profile"]["physical_initialization_time_ns"] == 100
    assert physical["profile"]["physical_measurement_time_ns"] == 500
    assert physical["profile"]["code_cycle_time_ns"] == 1000 and physical["profile"]["reaction_time_ns"] == 10000
    assert physical["hardware_or_quantum_advantage_guarantee"] is False


def test_saved_native_query_frontiers_and_returned_factory_identities_are_fully_bound(saved_replay):
    physical = saved_replay["physical"]
    audit, caps = physical["query_audit"], physical["caps"]
    assert audit["status"] == "complete" and audit["caps"] == caps
    assert audit["wall_seconds"] <= caps["wall_seconds"]
    assert audit["counts"]["top_level_calls"] == len(physical["physical_classes"])
    assert audit["counts"].get("factory_frontier_queries", 0) == len(audit["factory_frontier_queries"])
    assert audit["counts"].get("native_trace_queries", 0) == len(audit["native_trace_queries"])
    assert audit["counts"].get("factory_frontier_candidates", 0) == sum(len(row["frontier_candidates"]) for row in audit["factory_frontier_queries"])
    assert audit["counts"].get("factory_pipeline_candidates", 0) == sum(row["pipeline_candidates"] for row in audit["factory_frontier_queries"])
    for name, count in audit["counts"].items():
        assert 0 <= count <= caps[name]
    identities = set()
    for index, row in enumerate(audit["factory_frontier_queries"]):
        assert row["index"] == index
        assert row["configuration"]["distance"] in range(3, 50, 2)
        assert row["configuration"]["code_cycle_override"] == 1000
        assert row["configuration"]["error_correction_threshold"] == 0.01
        assert 0 <= row["pipeline_candidates"] <= 34 and len(row["frontier_candidates"]) <= 34
        for identity in row["frontier_candidates"]:
            assert identity["native_factory_isa_sha256"] == _canonical_sha(identity["native_factory_isa"])
            assert identity["native_t_instruction_sha256"] == _canonical_sha(identity["native_t_instruction"])
            assert identity["surface_code_query"] == row["configuration"]
            identities.add(_canonical_sha(identity))
    native_results = audit["native_trace_queries"]
    for index, row in enumerate(native_results):
        assert row["index"] == index and row["status"] in {"native_result_returned", "no_feasible_native_result"}
    for group in physical["physical_classes"]:
        for point in group["report"]["all_feasible_points"]:
            if not group["total_t_states"]:
                continue
            identity = point["native_factory_identity"]
            assert _canonical_sha(identity) in identities
            native = identity["native_t_instruction"]
            assert native["expected_time_ns"] == point["factory_supply"]["pipeline_time_ns"]
            assert native["expected_space"] == point["factory_physical_qubits"]
            assert native["expected_output_error_hex"] == float(point["factory_supply"]["output_error"]).hex()
            assert identity["surface_code_query"]["distance"] == point["data_code_distance"]
            matches = [row for row in native_results if row["physical_class_id"] == group["physical_class_id"]
                       and row["status"] == "native_result_returned"
                       and row["accounting_trace_sha256"] == point["accounting_trace_sha256"]
                       and row["native_t_instruction_sha256"] == identity["native_t_instruction_sha256"]
                       and row["runtime_ns"] == point["runtime_per_complete_attempt_ns"]
                       and row["peak_physical_qubits"] == point["peak_physical_qubits"]]
            assert matches
            assert any(float.fromhex(row["physical_error_hex"]) * group["shots"] == pytest.approx(point["full_physical_failure_model"], rel=1e-10)
                       for row in matches)


def test_every_common_physical_configuration_covers_all_groups_and_reproduces_exact_runtime_and_failures(saved_replay):
    physical = saved_replay["physical"]
    profile, groups = physical["profile"], {g["physical_class_id"]: g for g in physical["physical_classes"]}
    failure = Fraction(str(profile["factory_pipeline_failure_bound"]))
    total_t, retry = physical["actual_magic_states"], physical["global_factory_retry_cap"]
    assert physical["common_configurations"]
    for common in physical["common_configurations"]:
        distance = common["data_code_distance"]
        assert distance in range(3, 50, 2)
        assert common["global_factory_retry_cap"] == retry
        assert common["total_preparations"] == 82032 and common["total_t_states"] == total_t
        assert {p["physical_class_id"] for p in common["per_physical_class"]} == set(groups)
        serial, physical_errors, peaks, availabilities = 0, [], [], []
        for entry in common["per_physical_class"]:
            group, point = groups[entry["physical_class_id"]], entry["point"]
            assert point["data_code_distance"] == distance
            assert point["reserved_logical_patches"] == 15
            assert point["data_workspace_physical_qubits"] == 15 * (2 * distance ** 2 - 1)
            assert point["worst_case_runtime_all_attempts_ns"] == point["runtime_per_complete_attempt_ns"] * group["shots"]
            assert isinstance(point["worst_case_runtime_all_attempts_ns"], int)
            rows = point["macro_ledger_per_complete_attempt"]
            counts = {r["operation"]: r["count"] for r in rows}
            expected = dict(group["signature"]["event_counts"])
            expected["trial_control"] = expected.get("trial_control", 0) + 1
            expected["initial_reset"] += 11
            assert counts == expected
            runtime = sum(r["count"] * r["time_ns"] for r in rows)
            assert runtime == point["runtime_per_complete_attempt_ns"]
            assert point["code_cycles_per_complete_attempt"] == (runtime + 999) // 1000
            readout_error = _independent_readout_error(distance, profile["physical_error_rate"])
            patch_error = profile["memory_prefactor"] * (profile["physical_error_rate"] / profile["memory_threshold"]) ** ((distance + 1) // 2)
            initial_error = profile["initialization_residual_prefactor"] * (profile["physical_error_rate"] / profile["memory_threshold"]) ** ((distance + 1) // 2)
            for row in rows:
                macro = _independent_macro_cost(profile, distance, row["operation"], point["factory_supply"]["pipeline_time_ns"], retry)
                assert (row["time_ns"], row["readouts"], row["preparations"], row["decisions"]) == macro
                event_error = row["readouts"] * readout_error + row["preparations"] * initial_error + row["decisions"] * profile["controller_failure_per_decision"]
                assert row["event_failure_model"] == pytest.approx(event_error, rel=2e-10, abs=1e-40)
            component = point["failure_components_all_attempts"]
            assert component["data_workspace_code_exposure"] == pytest.approx(15 * point["code_cycles_per_complete_attempt"] * patch_error * group["shots"], rel=2e-10)
            assert component["logical_readout_scenario"] == pytest.approx(sum(r["count"] * r["readouts"] for r in rows) * readout_error * group["shots"], rel=2e-10, abs=1e-40)
            assert component["logical_initialization_scenario"] == pytest.approx(sum(r["count"] * r["preparations"] for r in rows) * initial_error * group["shots"], rel=2e-10)
            assert component["controller_scenario"] == pytest.approx(sum(r["count"] * r["decisions"] for r in rows) * profile["controller_failure_per_decision"] * group["shots"], rel=2e-10)
            assert component["factory_output_quality_model"] == pytest.approx(group["total_t_states"] * point["factory_supply"]["output_error"], rel=2e-10)
            assert sum(component.values()) == pytest.approx(point["full_physical_failure_model"], rel=2e-10)
            supply = point["factory_supply"]
            assert group["total_t_states"] % group["shots"] == 0
            states_per_attempt = group["total_t_states"] // group["shots"]
            port_cycles = (supply["extra_output_port_hold_ns_per_state"] + 999) // 1000
            assert component["factory_output_handoff_storage"] == pytest.approx(states_per_attempt * port_cycles * patch_error * group["shots"], rel=2e-10)
            assert supply["states_all_complete_attempts"] == group["total_t_states"]
            assert supply["worst_case_pipeline_attempts"] == group["total_t_states"] * retry
            assert supply["availability_failure_union_bound"] == pytest.approx(float(Fraction(group["total_t_states"]) * failure ** retry), abs=1e-30)
            if group["total_t_states"]:
                identity = point["native_factory_identity"]
                assert identity["native_factory_isa_sha256"] == common["native_factory_isa_sha256"]
                assert identity["native_t_instruction_sha256"] == common["native_t_instruction_sha256"]
                assert supply["full_pipeline_retry_cap_per_state"] == retry
                assert entry["common_factory_reserved_but_unused"] is False
                assert entry["common_peak_physical_qubits"] == point["peak_physical_qubits"]
            else:
                assert entry["common_factory_reserved_but_unused"] is True
                assert entry["common_peak_physical_qubits"] == point["data_workspace_physical_qubits"] + common["factory_physical_qubits"]
                assert point["factory_physical_qubits"] == 0
            assert entry["reserved_common_factory_physical_qubits"] == common["factory_physical_qubits"]
            serial += point["worst_case_runtime_all_attempts_ns"]
            physical_errors.append(point["full_physical_failure_model"])
            availabilities.append(supply["availability_failure_union_bound"])
            peaks.append(entry["common_peak_physical_qubits"])
        assert serial == common["worst_case_serial_runtime_ns"]
        assert max(peaks) == common["peak_physical_qubits"]
        assert Fraction(common["whole_physical_failure_model_exact_binary_rational_sum"]) == sum((Fraction.from_float(e) for e in physical_errors), Fraction())
        assert common["whole_physical_failure_model"] == pytest.approx(math.fsum(physical_errors), abs=1e-18)
        exact_factory = Fraction(total_t) * failure ** retry
        assert Fraction(common["whole_factory_availability_failure_bound_exact"]) == exact_factory <= Fraction("0.0025")
        assert common["whole_factory_availability_failure_bound"] == float(exact_factory)
        assert common["factory_availability_report_float_sum"] == pytest.approx(math.fsum(availabilities), abs=1e-18)
        assert common["whole_physical_failure_model"] <= 0.0025
    minimum_qubits = min(physical["common_configurations"], key=lambda p: (p["peak_physical_qubits"], p["worst_case_serial_runtime_ns"]))
    minimum_runtime = min(physical["common_configurations"], key=lambda p: (p["worst_case_serial_runtime_ns"], p["peak_physical_qubits"]))
    assert physical["selected_minimum_qubits"] == minimum_qubits
    assert physical["selected_minimum_runtime"] == minimum_runtime


@pytest.fixture(scope="module")
def codec():
    path = ROOT / "src/openquantum/optimization_compiled_codec.py"
    if not path.is_file():
        pytest.skip("Owned bounded DAG codec has not been written yet")
    spec = importlib.util.spec_from_file_location("compiled_codec_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dag_codec_roundtrip_preserves_scalar_types_float_bits_list_and_dict_order(codec):
    value = {"first": [None, False, True, 0, 1, -9, 1.0, 0.0, -0.0, 0.1, "λ"],
             "second": {"b": 2, "a": 1}, "ordinary_reserved_keys": {"codec": "json-content-dag-v1", "root": 0, "nodes": []}}
    packed = codec.pack(value)
    assert packed["codec"] == "json-content-dag-v1" and packed["schema_version"] == 1
    assert packed["logical_sha256"] == packed["nodes"][packed["root"]][0]
    assert len(packed["logical_sha256"]) == 64
    restored = codec.unpack(packed)
    assert codec.equivalent(value, restored)
    assert codec.equivalent(value, codec.loads(codec.dumps(packed)))
    assert list(restored) == list(value) and list(restored["second"]) == ["b", "a"]
    assert [type(item) for item in restored["first"]] == [type(item) for item in value["first"]]
    assert [item.hex() for item in restored["first"] if isinstance(item, float)] == [item.hex() for item in value["first"] if isinstance(item, float)]
    assert packed == codec.pack(value)
    assert codec.dumps(packed) == codec.dumps(codec.pack(value))


def test_dag_content_hashes_are_independently_reproducible(codec):
    packed = codec.pack({"ordered": [0, 1, 1.0, -0.0, "λ"], "another": [0, 1, 1.0, -0.0, "λ"]})
    hashes = []
    for index, (expected, tag, payload) in enumerate(packed["nodes"]):
        if tag == "list":
            assert all(type(child) is int and 0 <= child < index for child in payload)
            content = [hashes[child] for child in payload]
        elif tag == "dict":
            assert all(type(child) is int and 0 <= child < index for pair in payload for child in pair)
            content = [[hashes[key], hashes[value]] for key, value in payload]
        else:
            content = payload
        material = json.dumps([tag, content], ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
        actual = hashlib.sha256(material).hexdigest()
        assert actual == expected
        hashes.append(actual)
    assert len(set(hashes)) == len(hashes)
    assert packed["logical_sha256"] == hashes[packed["root"]]


@pytest.mark.parametrize("left,right", [(True, 1), (False, 0), (1, 1.0), (0.0, -0.0),
    ([1, 2], [2, 1]), ({"a": 1, "b": 2}, {"b": 2, "a": 1})])
def test_dag_logical_identity_does_not_coerce_types_floats_or_order(codec, left, right):
    assert not codec.equivalent(left, right)
    assert codec.pack(left)["logical_sha256"] != codec.pack(right)["logical_sha256"]


def test_dag_repeated_subtrees_are_deduplicated_without_expanding_aliases(codec):
    value = ["leaf"]
    for _ in range(24):
        value = [value, value]
    packed = codec.pack(value)
    assert len(packed["nodes"]) == 26
    assert len(codec.dumps(packed)) < 10000
    restored = codec.unpack(packed)
    assert restored[0] is restored[1]
    assert codec.equivalent(value, restored)


def test_dag_pack_and_comparator_reject_cyclic_python_values(codec):
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="yclic"):
        codec.pack(value)
    with pytest.raises(ValueError, match="yclic"):
        codec.equivalent(value, value)


def test_dag_depth_cap_applies_when_a_shared_subtree_reappears_deeper(codec):
    shared = [0]
    value = [shared, [[shared]]]
    limits = codec.CodecLimits(max_depth=3)
    with pytest.raises(ValueError, match="depth"):
        codec.pack(value, limits=limits)
    with pytest.raises(ValueError, match="depth"):
        codec.equivalent(value, value, limits=limits)


@pytest.mark.parametrize("value", [(1, 2), {1: "integer-key"}, np.float64(0.1), complex(1, 2)])
def test_dag_input_is_explicit_json_types_only(codec, value):
    with pytest.raises(ValueError):
        codec.pack(value)


@pytest.mark.parametrize("scalar,payload", [(None, False), (True, 1), (1, "01"), (0, "-0"),
    (1.0, "0x1p+0"), (1.0, "nan"), (1.0, "inf"), ("text", 1)])
def test_dag_scalar_payloads_must_retain_canonical_typed_form(codec, scalar, payload):
    packed = deepcopy(codec.pack(scalar))
    packed["nodes"][0][2] = payload
    with pytest.raises(ValueError):
        codec.unpack(packed)


@pytest.mark.parametrize("mutation", ["duplicate", "nonstring"])
def test_dag_dictionary_structure_is_checked_even_when_content_hash_is_recomputed(codec, mutation):
    packed = deepcopy(codec.pack({"a": 1, "b": "b"} if mutation == "duplicate" else {"a": 1, "b": "a"}))
    root, nodes = packed["root"], packed["nodes"]
    pairs = nodes[root][2]
    if mutation == "duplicate":
        pairs[1][0] = pairs[0][0]
    else:
        pairs[0][0] = pairs[0][1]
    content = [[nodes[key][0], nodes[value][0]] for key, value in pairs]
    digest = hashlib.sha256(json.dumps(["dict", content], ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()
    nodes[root][0] = packed["logical_sha256"] = digest
    with pytest.raises(ValueError):
        codec.unpack(packed)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("schema_version", 2),
    ("root", True), ("root", -1), ("codec", "unknown"), ("logical_sha256", "A" * 64)])
def test_dag_envelope_fields_are_strict_and_bounded(codec, field, value):
    packed = deepcopy(codec.pack([1]))
    packed[field] = value
    with pytest.raises(ValueError):
        codec.unpack(packed)


@pytest.mark.parametrize("mutation", ["self_cycle", "missing", "negative", "bool_ref", "node_hash", "root_hash", "duplicate_hash", "unknown_tag", "duplicate_dict_key", "unreachable"])
def test_dag_decoder_rejects_malformed_graphs_and_identity_drift(codec, mutation):
    packed = deepcopy(codec.pack({"a": [1], "b": [2]}))
    root = packed["root"]
    if mutation in {"self_cycle", "missing", "negative", "bool_ref"}:
        pair = packed["nodes"][root][2][0]
        pair[1] = {"self_cycle": root, "missing": len(packed["nodes"]) + 2,
                   "negative": -1, "bool_ref": True}[mutation]
    elif mutation == "node_hash":
        packed["nodes"][0][0] = "0" * 64
    elif mutation == "root_hash":
        packed["logical_sha256"] = "0" * 64
    elif mutation == "duplicate_hash":
        packed["nodes"].insert(root, deepcopy(packed["nodes"][0]))
        packed["root"] += 1
    elif mutation == "unknown_tag":
        packed["nodes"][0][1] = "opaque"
    elif mutation == "duplicate_dict_key":
        packed["nodes"][root][2][1][0] = packed["nodes"][root][2][0][0]
    else:
        packed = deepcopy(codec.pack(["leaf"]))
        packed["nodes"].insert(1, deepcopy(codec.pack(3.25)["nodes"][0]))
        packed["root"] = 2
    with pytest.raises(ValueError):
        codec.unpack(packed)


@pytest.mark.parametrize("raw", [b'{"codec":"x","codec":"y"}', b'{"node":NaN}', b'{"node":Infinity}'])
def test_dag_json_loader_rejects_duplicate_keys_and_nonfinite_json(codec, raw):
    with pytest.raises(ValueError):
        codec.loads(raw)


@pytest.mark.parametrize("field,value", [("max_nodes", [1, 2]), ("max_references", [1, 2, 3]),
    ("max_depth", [[[1]]]), ("max_container_items", [1, 2, 3]),
    ("max_scalar_bytes", "abc"), ("max_integer_digits", 100)])
def test_dag_codec_enforces_each_declared_materialization_limit(codec, field, value):
    maximum = {"max_nodes": 2, "max_references": 2, "max_depth": 2,
               "max_container_items": 2, "max_scalar_bytes": 2, "max_integer_digits": 2}[field]
    limits = codec.CodecLimits(**{field: maximum})
    with pytest.raises(ValueError):
        codec.pack(value, limits=limits)


def test_dag_codec_bounds_encoded_transport_separately_from_logical_sharing(codec):
    packed = codec.pack("x" * 1024)
    raw = codec.dumps(packed)
    assert len(raw) > 512
    with pytest.raises(ValueError):
        codec.dumps(packed, limits=codec.CodecLimits(max_encoded_bytes=512))
    with pytest.raises(ValueError):
        codec.loads(raw, limits=codec.CodecLimits(max_encoded_bytes=512))


@pytest.mark.parametrize("field,value", [("max_encoded_bytes", 0), ("max_nodes", True),
    ("max_nodes", 500001), ("max_references", 4000001), ("max_depth", 129),
    ("max_container_items", 100001), ("max_scalar_bytes", 4194305), ("max_integer_digits", 1025)])
def test_dag_limits_cannot_exceed_the_frozen_domain(codec, field, value):
    with pytest.raises(ValueError):
        codec.CodecLimits(**{field: value})


@pytest.fixture(scope="module")
def completion():
    path = ROOT / "scripts/complete_optimization_compiled.py"
    if not path.is_file():
        pytest.skip("Owned physical-completion producer has not been written yet")
    spec = importlib.util.spec_from_file_location("compiled_completion_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_physical_recovery_declares_one_pass_and_preserves_original_science_and_failure(completion, compiled_checkpoints, generator):
    store = generator.ArtifactStore(OUTPUT)
    frozen, timing, failure = completion.first_attempt(store)
    recipe = json.loads((OUTPUT / "recipe.json").read_text())
    recovery = completion.recovery_recipe(store, recipe)
    assert recovery["scientific_change"] is False
    assert recovery["synthesis_or_optimizer_repeated"] is False
    assert recovery["maximum_additional_physical_passes"] == 1
    assert recovery["quantum_preparations"] == 82032
    assert recovery["original_model_recipe"] == recipe
    assert recovery["original_identity"] == frozen["identity"]
    assert failure["failure"]["error_type"] == "DecodedByteCapOrSerializationFailure"
    assert failure["failure"]["partial_native_work_unknown"] is False
    assert failure["failure"]["complete_query_records_omitted_due_to_serialization_limit"] is True
    expected_counts = {"top_level_calls": 49, "factory_frontier_queries": 1176,
        "factory_frontier_candidates": 12348, "factory_pipeline_candidates": 816,
        "native_trace_queries": 6174}
    assert timing["status"] == "incomplete"
    assert timing["total_wall_seconds"] == pytest.approx(11.716274958002032, abs=1e-12)
    assert recovery["first_known_query_counts"] == expected_counts
    assert recovery["first_physical_wall_seconds"] == timing["total_wall_seconds"]
    assert recovery["cumulative_query_caps"] == {"top_level_calls": 128, "factory_frontier_queries": 3072,
        "factory_frontier_candidates": 52224, "factory_pipeline_candidates": 52224, "native_trace_queries": 52224}
    assert recovery["second_attempt_effective_caps"] == {"top_level_calls": 64, "factory_frontier_queries": 1536,
        "factory_frontier_candidates": 39876, "factory_pipeline_candidates": 51408, "native_trace_queries": 46050}
    assert recovery["cumulative_physical_wall_cap_seconds"] == 120
    assert recovery["second_remaining_wall_seconds"] == 120 - timing["total_wall_seconds"]
    assert recovery["decoded_transport_bytes"] == 64 * 1024 * 1024
    assert recovery["public_artifact_bytes"] == 16 * 1024 * 1024


def _completion_source_fixture(tmp_path, completion, generator):
    store = generator.ArtifactStore(tmp_path / "bundle")
    sources = {"complete_optimization_compiled.py": completion.SCRIPT.read_bytes(),
               "optimization_compiled_codec.py": completion.CODEC_PATH.read_bytes()}
    frozen = {"identity": {"source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()}}}
    for name, raw in sources.items():
        store.write(name, raw)
    store.json("completion-freeze.json", frozen)
    return store, frozen


def test_completion_decoder_binds_exact_live_and_snapshot_sources_before_execution(tmp_path, completion, generator):
    store, frozen = _completion_source_fixture(tmp_path, completion, generator)
    completion.bind_completion_sources(store, frozen)
    assert sha(store.target("complete_optimization_compiled.py")) == frozen["identity"]["source_sha256"]["complete_optimization_compiled.py"]
    assert sha(store.target("optimization_compiled_codec.py")) == frozen["identity"]["source_sha256"]["optimization_compiled_codec.py"]


@pytest.mark.parametrize("mutation", ["snapshot_script", "snapshot_codec", "live_script", "live_codec"])
def test_direct_saved_loader_rejects_source_drift_before_loading_the_codec(tmp_path, completion, generator, core, compiled_inputs, monkeypatch, mutation):
    store, _ = _completion_source_fixture(tmp_path, completion, generator)
    if mutation.startswith("snapshot"):
        filename = "complete_optimization_compiled.py" if mutation.endswith("script") else "optimization_compiled_codec.py"
        path = store.target(filename)
        path.write_bytes(path.read_bytes() + b"\n# source drift\n")
    else:
        name = "SCRIPT" if mutation.endswith("script") else "CODEC_PATH"
        path = tmp_path / "changed-source.py"
        path.write_bytes(getattr(completion, name).read_bytes() + b"\n# source drift\n")
        monkeypatch.setattr(completion, name, path)
    monkeypatch.setattr(completion, "legacy_modules", lambda output: (generator, core, compiled_inputs))
    def forbidden():
        raise AssertionError("Decoder code must not execute before its frozen source binding is checked")
    monkeypatch.setattr(completion, "load_codec", forbidden)
    with pytest.raises(ValueError, match="producer"):
        completion.load_completed_replay(store.root, validate=False)


def test_private_live_result_reference_roundtrip_is_hash_bound_and_never_public(tmp_path, completion):
    value = {"typed": [True, 1, 1.0, -0.0], "shared": {"factory": ["fixed"]}}
    value["again"] = value["shared"]
    reference = completion.preserve_private_reference(value, tmp_path)
    assert Path(reference["file"]).parent == tmp_path
    assert reference["bytes"] == Path(reference["file"]).stat().st_size <= completion.PRIVATE_REFERENCE_CAP
    assert reference["sha256"] == sha(reference["file"])
    restored = completion.trusted_private_reference(tmp_path, reference)
    assert restored == value and restored["shared"] is restored["again"]
    assert type(restored["typed"][0]) is bool and type(restored["typed"][1]) is int and type(restored["typed"][2]) is float
    assert restored["typed"][3].hex() == (-0.0).hex()


@pytest.mark.parametrize("mutation", ["hash", "bytes", "bound", "filename", "symlink"])
def test_private_reference_corruption_is_rejected_before_unpickling(tmp_path, completion, monkeypatch, mutation):
    reference = completion.preserve_private_reference({"fixed": True}, tmp_path)
    if mutation == "hash":
        reference["sha256"] = "0" * 64
    elif mutation == "bytes":
        reference["bytes"] += 1
    elif mutation == "bound":
        reference["bytes"] = completion.PRIVATE_REFERENCE_CAP + 1
    elif mutation == "filename":
        wrong = tmp_path / "untrusted.pickle"
        wrong.write_bytes(Path(reference["file"]).read_bytes())
        reference["file"] = str(wrong)
    else:
        original = Path(reference["file"])
        moved = tmp_path / "other.pickle"
        original.rename(moved)
        original.symlink_to(moved)
    def forbidden(*args, **kwargs):
        raise AssertionError("Only a bounded regular hash-pinned producer-owned reference may be deserialized")
    monkeypatch.setattr(completion.pickle, "loads", forbidden)
    with pytest.raises(ValueError):
        completion.trusted_private_reference(tmp_path, reference)


@pytest.fixture(scope="module")
def completed_manifest():
    path = OUTPUT / "manifest.json"
    if not path.is_file():
        pytest.skip("Complete public manifest has not been exported yet")
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "complete"
    return manifest


def test_completed_manifest_binds_dag_sources_original_checkpoints_and_preserved_attempt(completed_manifest, saved_replay, codec):
    manifest, replay = completed_manifest, saved_replay
    saved_path = OUTPUT / manifest["completed_replay"]["file"]
    assert sha(saved_path) == manifest["completed_replay"]["sha256"]
    saved = json.loads(gzip.decompress(saved_path.read_bytes()))
    freeze = json.loads((OUTPUT / "completion-freeze.json").read_text())
    old_freeze = json.loads((OUTPUT / "freeze.json").read_text())
    assert manifest["identity"] == saved["identity"] == freeze["identity"]
    assert manifest["original_identity"] == saved["original_identity"] == old_freeze["identity"]
    assert manifest["bindings"] == replay["bindings"]
    assert manifest["run_id"] == RUN_ID and manifest["preparations"] == 82032 and manifest["sampled_groups"] == 110
    for name, expected in old_freeze["identity"]["source_sha256"].items():
        assert sha(OUTPUT / name) == expected
    for name, expected in freeze["identity"]["source_sha256"].items():
        assert sha(OUTPUT / name) == expected
    for name, expected in freeze["original_artifact_inventory"].items():
        assert sha(OUTPUT / name) == expected
    physical_ref = saved["physical_storage"]
    assert physical_ref == manifest["physical_storage"]
    assert physical_ref["encoding"] == "json-content-dag-v1"
    physical_path = OUTPUT / physical_ref["file"]
    assert sha(physical_path) == physical_ref["sha256"]
    packed_raw = gzip.decompress(physical_path.read_bytes())
    assert len(packed_raw) <= 64 * 1024 * 1024
    packed = json.loads(packed_raw)
    assert codec.equivalent(codec.unpack(packed), replay["physical"])
    assert saved["recovery"]["codec_statistics"] == codec.statistics(packed)
    assert saved["recovery"]["roundtrip_exact_to_second_live_value"] is True
    assert saved["recovery"]["first_complete_result_unavailable"] is True
    assert saved["recovery"]["quantum_preparations_counted_once"] == 82032
    refs = saved["source_class_checkpoints"]
    assert [r["class_id"] for r in refs] == [c["class_id"] for c in replay["source_classes"]]
    for ref, compiled in zip(refs, replay["source_classes"], strict=True):
        path = OUTPUT / ref["file"]
        assert sha(path) == ref["sha256"]
        cp = json.loads(gzip.decompress(path.read_bytes()))
        assert cp["identity"] == saved["original_identity"] and cp["record"] == compiled
    assert manifest["public_reader_requires_private_pickle"] is False


def test_recovery_cumulative_queries_times_and_before_query_amendment_are_visible(completed_manifest, saved_replay):
    manifest, physical = completed_manifest, saved_replay["physical"]
    recipe = json.loads((OUTPUT / "completion-recipe.json").read_text())
    amendment = json.loads((OUTPUT / "completion-amendment.json").read_text())
    freeze = json.loads((OUTPUT / "completion-freeze.json").read_text())
    assert amendment["type"] == "explicit_computation_only_recovery_amendment"
    assert amendment["source_angles_shots_model_failure_budgets_unchanged"] is True
    assert amendment["full_first_result_unavailable"] is True
    assert amendment["no_claim_of_equality_to_missing_first_physical_result"] is True
    assert amendment["maximum_additional_physical_passes"] == recipe["maximum_additional_physical_passes"] == 1
    assert amendment["changes"] == {"cumulative_top_level_calls": [64, 128], "cumulative_frontier_queries": [1536, 3072]}
    timing = json.loads((OUTPUT / "completion-timing.json").read_text())
    assert timing["identity"] == freeze["identity"] == manifest["identity"]
    assert timing["status"] == "complete" and timing["wall_cap_seconds"] == 120
    assert timing["first_wall_seconds"] == recipe["first_physical_wall_seconds"]
    assert timing["cumulative_wall_seconds"] == pytest.approx(timing["first_wall_seconds"] + timing["recovery_wall_seconds"], abs=1e-10)
    assert timing["cumulative_wall_seconds"] <= 120
    assert manifest["timing"]["original_physical"]["status"] == "incomplete"
    assert manifest["timing"]["synthesis"]["status"] == "complete"
    assert manifest["timing"]["completion"] == timing
    recovery = manifest["recovery"]
    assert recovery["first_query_counts"] == recipe["first_known_query_counts"]
    assert recovery["second_query_counts"] == physical["query_audit"]["counts"]
    for key, first in recipe["first_known_query_counts"].items():
        expected = first + physical["query_audit"]["counts"].get(key, 0)
        assert recovery["cumulative_query_counts"][key] == expected <= recipe["cumulative_query_caps"][key]
    failure = json.loads((OUTPUT / manifest["old_failure_file"]).read_text())
    assert failure["failure"]["error_type"] == "DecodedByteCapOrSerializationFailure"


def test_completed_public_circuits_match_saved_ops_phase_maps_and_every_group(completed_manifest, saved_replay, core):
    manifest, replay = completed_manifest, saved_replay
    assert len(manifest["classes"]) == 54
    raw_by_class = {c["class_id"]: c for c in replay["source_classes"]}
    unique_files = {}
    for exported in manifest["classes"]:
        c = raw_by_class[exported["class_id"]]
        assert exported["parameters"] == c["parameters"] and exported["shots"] == c["shots"] and exported["batches"] == c["batches"]
        assert exported["actual_synthesis"] == c["report"] and exported["checks"] == c["checks"]
        for role, prefix in (("augmented_source", "source"), ("routed", "routed"), ("clifford_t", "clifford_t")):
            desc = exported["circuits"][role]
            raw = c[f"{prefix}_qasm"].encode()
            assert desc["sha256"] == hashlib.sha256(raw).hexdigest()
            assert (OUTPUT / desc["file"]).read_bytes() == raw and desc["bytes"] == len(raw)
            assert desc["global_phase"] == c[f"{prefix}_global_phase"] and desc["logical_qubits"] == 4
            ops = c[f"{prefix}_operations"]
            assert desc["operations"] == len(ops) and desc["gate_counts"] == dict(Counter(o[0] for o in ops))
            assert desc["measurement_map"] == [[o[1][0], o[2][0]] for o in ops if o[0] == "measure"]
            assert desc["reset_wire_order"] == [0, 1, 2, 3]
            assert desc["controlled_oracle_phase_certificate"] is False
            restored = core.circuit_from_operations(ops, 4, desc["global_phase"])
            if "$" in c[f"{prefix}_qasm"]:
                assert prefix == "routed"
                # Primitive JSON retains the actual routed wire indices but
                # not Qiskit's physical-$wire export marker. Recreate only
                # that presentation metadata, not the original SABRE object.
                from qiskit.transpiler import Layout, TranspileLayout
                before_ops = core.circuit_operations(restored)
                before_phase = float(restored.global_phase)
                restored._layout = TranspileLayout(
                    initial_layout=Layout({bit: index for index, bit in enumerate(restored.qubits)}),
                    input_qubit_mapping={bit: index for index, bit in enumerate(restored.qubits)},
                    final_layout=None, _input_qubit_count=len(restored.qubits),
                    _output_qubit_list=restored.qubits)
                assert core.circuit_operations(restored) == before_ops == ops
                assert float(restored.global_phase) == before_phase == desc["global_phase"]
                assert [[restored.find_bit(item.qubits[0]).index, restored.find_bit(item.clbits[0]).index]
                        for item in restored.data if item.operation.name == "measure"] == desc["measurement_map"]
            assert qasm3.dumps(restored).encode() == raw
            unique_files[desc["file"]] = raw
        source = exported["circuits"]["recorded_source"]
        assert source["sha256"] == c["class_id"]
        assert (OUTPUT / source["file"]).read_bytes() == c["original_source_qasm"].encode()
        unique_files[source["file"]] = c["original_source_qasm"].encode()
    assert set(unique_files) == {str(p.relative_to(OUTPUT)) for p in (OUTPUT / "circuits").glob("*.qasm")}
    assert sum(len(c["batches"]) for c in manifest["classes"]) == 110


def test_completed_public_csv_json_exports_reconcile_all_counts_controls_and_conditionals(completed_manifest, saved_replay, selected, parent_study):
    replay, (_, selected_run) = saved_replay, selected
    assert json.loads((OUTPUT / "resources.json").read_text()) == replay["resources"]
    assert json.loads((OUTPUT / "conditional-readout.json").read_text()) == replay["conditional_readout"]
    assert json.loads((OUTPUT / "classical-work.json").read_text()) == replay["classical_work"]
    assert replay["negative_controls"]["sampled"] == parent_study["summary_rows"]
    assert replay["negative_controls"]["cpu"] == parent_study["control_summary_rows"]
    with (OUTPUT / "groups.csv").open(newline="") as handle:
        groups = list(csv.DictReader(handle))
    assert len(groups) == 110
    source_ids = {c["class_id"] for c in replay["source_classes"]}
    for index, (row, call) in enumerate(zip(groups, [*selected_run["evaluations"], selected_run["validation"]], strict=True)):
        assert int(row["chronological_group_index"]) == index
        assert row["phase"] == ("holdout" if index == 109 else "training")
        assert int(row["shots"]) == call["groups"][0]["shots"]
        assert json.loads(row["counts"]) == call["groups"][0]["counts"]
        assert json.loads(row["rng"]) == call["groups"][0]["rng"]
        assert json.loads(row["parameters"]) == call["parameters"]
        assert float(row["objective_estimate"]) == call["objective_estimate"]
        assert float(row["standard_error"]) == call["estimated_standard_error"]
        assert row["class_id"] in source_ids
    with (OUTPUT / "classes.csv").open(newline="") as handle:
        classes = list(csv.DictReader(handle))
    assert len(classes) == 54 and sum(int(row["shots"]) for row in classes) == 82032
    assert sum(int(row["shots"]) * int(row["actual_t_per_preparation"]) for row in classes) == replay["resources"]["actual_t_states"]
    with (OUTPUT / "negative-controls.csv").open(newline="") as handle:
        controls = list(csv.DictReader(handle))
    assert len(controls) == 60
    assert Counter(row["kind"] for row in controls) == {"sampled": 36, "cpu": 24}
    vqe_hits = [row for row in controls if row["kind"] == "sampled" and row["algorithm"] == "vqe" and row["ideal_target_met"] == "True"]
    assert len(vqe_hits) == 2 and all(row["heldout_target_status"] == "inconclusive" for row in vqe_hits)
    readout = replay["conditional_readout"]
    assert readout["family_group_count"] == 48 and readout["sampling_family_failure"] == 0.04
    assert readout["fixed_final_shots"] == 16384 and readout["normalized_precision_radius"] <= 0.05
    assert readout["future_supported_interval_lower_on_concentration_event"] >= 0.8
    assert readout["compiled_exact_ratio"] >= 0.8
    assert replay["classical_work"]["archived_optimizer_wall_seconds"] == selected_run["optimizer_wall_seconds"]
    assert replay["classical_work"]["archived_control_wall_seconds"] == selected_run["classical_control_seconds"]
    assert replay["classical_work"]["quantum_hardware_optimizer_scoring_latency"] is None


def test_completed_artifact_size_is_exact_and_private_payloads_are_excluded(completed_manifest):
    size = json.loads((OUTPUT / "artifact-size.json").read_text())
    files = [p for p in OUTPUT.rglob("*") if p.is_file()]
    assert size["total_bytes"] == sum(p.stat().st_size for p in files) <= 16 * 1024 * 1024
    assert size["files"] == len(files)
    assert size["parent_archive_copied"] is False and size["private_cache_included"] is False
    assert not any(p.is_symlink() for p in OUTPUT.rglob("*"))
    assert not any(p.suffix in {".pickle", ".log", ".db", ".sqlite", ".pyc"} or "__pycache__" in p.parts for p in files)
    assert completed_manifest["public_reader_requires_private_pickle"] is False
