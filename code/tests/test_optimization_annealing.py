"""Independent matrix, symmetry, ODE, and workload checks for annealing."""

from functools import lru_cache
import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from openquantum import optimization as opt

CYCLE_EDGES = ((0, 1), (1, 2), (2, 3), (0, 3))


def _hamiltonians():
    """Define H_D by bit flips and H_P by classical edge parity."""
    driver = np.zeros((16, 16))
    for column in range(16):
        for qubit in range(4):
            driver[column ^ (1 << qubit), column] = -1
    costs = np.array([sum(((index >> u) & 1) != ((index >> v) & 1)
                         for u, v in CYCLE_EDGES) for index in range(16)])
    problem = np.diag(2 - costs)
    return driver, problem, costs


def _symmetry_sector():
    symmetry = np.zeros((16, 16))
    sector = np.zeros((16, 8))
    for index in range(16):
        symmetry[index ^ 15, index] = 1
    for column in range(8):
        sector[column, column] = 1 / math.sqrt(2)
        sector[column ^ 15, column] = 1 / math.sqrt(2)
    return symmetry, sector


def _dense_strang(total_time, steps):
    driver, problem, _ = _hamiltonians()
    unitary = np.eye(16, dtype=complex)
    dt = total_time / steps
    for index in range(steps):
        s = (index + 0.5) / steps
        half_driver = expm(-0.5j * dt * (1 - s) * driver)
        full_problem = expm(-1j * dt * s * problem)
        unitary = half_driver @ full_problem @ half_driver @ unitary
    return unitary


@lru_cache(maxsize=4)
def _continuous_reference(total_time):
    driver, problem, _ = _hamiltonians()
    initial = np.full(16, 0.25, dtype=complex)

    def derivative(time, state):
        s = time / total_time
        return -1j * ((1 - s) * driver + s * problem) @ state

    result = solve_ivp(derivative, (0, total_time), initial, method="DOP853", rtol=1e-12, atol=1e-13)
    assert result.success
    return result.y[:, -1]


def _phase_invariant_error(first, second):
    overlap = np.vdot(first, second)
    phase = overlap / abs(overlap)
    return np.linalg.norm(second - phase * first)


def test_annealing_hamiltonians_match_independent_bit_flip_and_cut_matrices():
    driver, problem, costs = _hamiltonians()
    source_driver, source_problem = opt.annealing_hamiltonians(4, CYCLE_EDGES)
    np.testing.assert_allclose(source_driver, driver, atol=1e-12)
    np.testing.assert_allclose(source_problem, problem, atol=1e-12)
    np.testing.assert_allclose(np.diag(problem), 2 - costs, atol=1e-12)
    assert np.linalg.eigvalsh(driver)[0] == pytest.approx(-4)
    assert np.linalg.eigvalsh(problem)[0] == pytest.approx(-2)
    assert costs.max() == 4


def test_global_bit_flip_symmetry_and_initial_sector_are_exact_on_this_instance():
    driver, problem, _ = _hamiltonians()
    symmetry, sector = _symmetry_sector()
    np.testing.assert_allclose(symmetry @ driver, driver @ symmetry, atol=1e-12)
    np.testing.assert_allclose(symmetry @ problem, problem @ symmetry, atol=1e-12)
    np.testing.assert_allclose(sector.T @ sector, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(sector @ sector.T, (np.eye(16) + symmetry) / 2, atol=1e-12)
    initial = np.full(16, 0.25)
    np.testing.assert_allclose(symmetry @ initial, initial, atol=1e-12)
    final = opt.annealing_state(4, CYCLE_EDGES, 4.0, 32)
    np.testing.assert_allclose(symmetry @ final, final, atol=1e-11)


def test_full_endpoint_degeneracy_is_distinct_from_the_reachable_sector_gap():
    _, problem, costs = _hamiltonians()
    _, sector = _symmetry_sector()
    full = np.linalg.eigvalsh(problem)
    even = np.linalg.eigvalsh(sector.T @ problem @ sector)
    assert full[1] - full[0] == pytest.approx(0, abs=1e-12)
    assert even[1] - even[0] == pytest.approx(2, abs=1e-12)
    ground = np.zeros(16)
    ground[[5, 10]] = 1 / math.sqrt(2)
    np.testing.assert_allclose(problem @ ground, -2 * ground, atol=1e-12)
    assert all(costs[index] == 4 for index in np.flatnonzero(ground))


def test_gap_grid_matches_independent_full_spectra_and_finite_instance_formula():
    driver, problem, _ = _hamiltonians()
    _, sector = _symmetry_sector()
    rows = opt.annealing_gap_scan(4, CYCLE_EDGES, points=201)
    assert len(rows) == 201
    for index, row in enumerate(rows):
        s = index / 200
        assert row["s"] == pytest.approx(s, abs=1e-12)
        hamiltonian = (1 - s) * driver + s * problem
        full = np.linalg.eigvalsh(hamiltonian)
        even = np.linalg.eigvalsh(sector.T @ hamiltonian @ sector)
        assert row["full_gap"] == pytest.approx(full[1] - full[0], abs=1e-11)
        assert row["plus_sector_gap"] == pytest.approx(even[1] - even[0], abs=1e-11)
        exact_sector_gap = math.sqrt(16 * (1 - s) ** 2 + 4 * s**2 - 8 * math.sqrt(2) * s * (1 - s))
        assert row["plus_sector_gap"] == pytest.approx(exact_sector_gap, abs=1e-11)
    assert rows[-1]["full_gap"] == pytest.approx(0, abs=1e-12)
    assert rows[0]["plus_sector_gap"] == pytest.approx(4, abs=1e-12)
    assert rows[-1]["plus_sector_gap"] == pytest.approx(2, abs=1e-12)
    # The finite grid must not be silently labeled the exact continuous minimum.
    exact_minimum = math.sqrt(8 / (5 + 2 * math.sqrt(2)))
    assert min(row["plus_sector_gap"] for row in rows) >= exact_minimum - 1e-12


def test_chronological_midpoint_angles_and_state_preparation_are_explicit():
    total_time, steps = 1.7, 3
    circuit = opt.build_annealing_circuit(4, CYCLE_EDGES, total_time, steps)
    assert [item.operation.name for item in circuit.data[:4]] == ["h"] * 4
    actual_driver = [float(item.operation.params[0]) for item in circuit.data if item.operation.name == "rx"]
    actual_problem = [float(item.operation.params[0]) for item in circuit.data if item.operation.name == "rz"]
    dt = total_time / steps
    expected_driver, expected_problem = [], []
    for index in range(steps):
        s = (index + 0.5) / steps
        expected_driver.extend([-dt * (1 - s)] * 8)
        expected_problem.extend([dt * s] * 4)
    np.testing.assert_allclose(actual_driver, expected_driver, atol=1e-12)
    np.testing.assert_allclose(actual_problem, expected_problem, atol=1e-12)
    assert circuit.count_ops()["cx"] == 8 * steps
    assert circuit.count_ops()["h"] == 4
    assert circuit.num_qubits == 4


@pytest.mark.parametrize("total_time,steps", [(0.7, 1), (1.7, 3), (4.0, 8)])
def test_explicit_circuit_matches_an_independent_dense_strang_product(total_time, steps):
    initial_hadamard = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    preparation = initial_hadamard
    for _ in range(3):
        preparation = np.kron(preparation, initial_hadamard)
    dense = _dense_strang(total_time, steps) @ preparation
    circuit = opt.build_annealing_circuit(4, CYCLE_EDGES, total_time, steps)
    assert Operator(circuit).equiv(Operator(dense), atol=1e-11, rtol=0)
    expected_state = _dense_strang(total_time, steps) @ np.full(16, 0.25)
    np.testing.assert_allclose(opt.annealing_state(4, CYCLE_EDGES, total_time, steps), expected_state, atol=1e-11)
    compiled = opt.compile_rotation_basis(circuit)
    assert Operator(compiled).equiv(Operator(circuit), atol=1e-11, rtol=0)


def test_continuous_reference_agrees_with_independent_complex_ode():
    independent = _continuous_reference(4.0)
    result = opt.annealing_continuous_reference(4, CYCLE_EDGES, 4.0)
    assert result["solver_success"]
    assert result["nfev"] > 0
    np.testing.assert_allclose(result["state"], independent, atol=2e-9, rtol=0)
    assert np.linalg.norm(result["state"]) == pytest.approx(1, abs=2e-9)
    _, _, costs = _hamiltonians()
    assert np.abs(independent) ** 2 @ costs == pytest.approx(3.22551833549, abs=2e-9)


def test_midpoint_strang_converges_to_the_fixed_time_ode_reference():
    reference = _continuous_reference(4.0)
    errors = []
    for steps in (8, 16, 32, 64, 128):
        state = opt.annealing_state(4, CYCLE_EDGES, 4.0, steps)
        errors.append(_phase_invariant_error(reference, state))
    assert all(later < 0.35 * earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 1e-4


def test_readout_is_explicit_and_generator_bounds_are_separate_from_simulation():
    measured = opt.build_annealing_circuit(4, CYCLE_EDGES, 4.0, 8, measure=True)
    assert measured.count_ops()["measure"] == 4
    assert opt.build_annealing_circuit(13, [(0, 12)], 1.0, 2).num_qubits == 13
    with pytest.raises(ValueError):
        opt.annealing_state(13, [(0, 12)], 1.0, 2, max_qubits=12)
    with pytest.raises(ValueError):
        opt.annealing_hamiltonians(9, [(0, 8)], max_qubits=8)


@pytest.mark.parametrize("total_time,steps", [(math.nan, 4), (math.inf, 4), (-1.0, 4), (1.0, 0), (1.0, True), (1.0, 1000000)])
def test_invalid_or_oversized_annealing_inputs_are_rejected(total_time, steps):
    with pytest.raises(ValueError):
        opt.build_annealing_circuit(4, CYCLE_EDGES, total_time, steps)


@pytest.fixture(scope="module")
def annealing_demo():
    return opt.run_annealing_demo()


def test_predeclared_main_and_diagnostic_studies_report_independent_quantities(annealing_demo):
    main = annealing_demo["main"]
    assert main["total_time"] == annealing_demo["target"]["predeclared_total_time"] == 16.0
    assert main["steps"] == annealing_demo["target"]["predeclared_steps"] == 256
    assert [row["total_time"] for row in annealing_demo["time_study"]] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert all(row["steps"] == 256 for row in annealing_demo["time_study"])
    assert [row["steps"] for row in annealing_demo["step_study"]] == [8, 16, 32, 64, 128]
    assert all(row["total_time"] == 4.0 for row in annealing_demo["step_study"])
    assert annealing_demo["reference"]["cut_values"] == _hamiltonians()[2].tolist()
    assert annealing_demo["reference"]["maximum_cut"] == 4
    assert annealing_demo["reference"]["optimal_bitstrings"] == ["0101", "1010"]
    probabilities = np.asarray(main["probabilities"])
    assert probabilities.sum() == pytest.approx(1, abs=1e-11)
    assert main["expected_cut"] == pytest.approx(probabilities @ _hamiltonians()[2], abs=1e-11)
    assert main["optimal_probability"] == pytest.approx(probabilities[[5, 10]].sum(), abs=1e-11)
    assert annealing_demo["target"]["achieved"] == (
        main["optimal_probability"] >= annealing_demo["target"]["ideal_optimal_probability_at_least"])
    assert main["dense_strang_state_error"] < 1e-10
    assert main["continuous_solver"]["solver_success"] is True
    assert len(annealing_demo["gap_scan"]) == 201


def test_saved_step_convergence_matches_a_separately_computed_ode_reference(annealing_demo):
    reference = _continuous_reference(4.0)
    _, _, cuts = _hamiltonians()
    expected_cut = np.abs(reference) ** 2 @ cuts
    for row in annealing_demo["step_study"]:
        state = opt.annealing_state(4, CYCLE_EDGES, 4.0, row["steps"])
        overlap = abs(np.vdot(reference / np.linalg.norm(reference), state / np.linalg.norm(state)))
        assert row["state_infidelity"] == pytest.approx(1 - overlap**2, abs=2e-9)
        assert row["state_phase_aligned_error"] == pytest.approx(_phase_invariant_error(reference, state), abs=2e-9)
        assert row["cut_mean_absolute_error"] == pytest.approx(abs(row["expected_cut"] - expected_cut), abs=2e-9)


def test_seeded_main_readout_scores_all_outcomes_and_retains_joint_cut_variance(annealing_demo):
    _, _, cuts = _hamiltonians()
    assert [row["seed"] for row in annealing_demo["readout"]] == [7, 19, 41]
    stream_keys = []
    for row in annealing_demo["readout"]:
        counts = row["counts"]
        assert sum(counts.values()) == row["shots"] == 4096
        assert all(len(bits) == 4 for bits in counts)
        mean = sum(number * cuts[int(bits, 2)] for bits, number in counts.items()) / row["shots"]
        variance = sum(number * (cuts[int(bits, 2)] - mean) ** 2 for bits, number in counts.items()) / (row["shots"] - 1)
        assert row["cut_mean"] == pytest.approx(mean, abs=1e-12)
        assert row["sample_variance"] == pytest.approx(variance, abs=1e-12)
        assert row["cut_standard_error"] == pytest.approx(math.sqrt(variance / row["shots"]), abs=1e-12)
        optimal_count = sum(number for bits, number in counts.items() if cuts[int(bits, 2)] == 4)
        assert row["optimal_count"] == optimal_count
        assert row["optimal_frequency"] == optimal_count / row["shots"]
        assert row["rng"]["entropy"] == [row["seed"], 4]
        assert row["rng"]["spawn_key"] == [1, 0, 0]
        stream_keys.append(tuple(row["rng"]["stream_key"]))
    assert len(stream_keys) == len(set(stream_keys)) == 3


def test_main_logical_ledger_counts_all_preparations_rotations_and_readouts(annealing_demo):
    resources = opt.annealing_run_resources(annealing_demo)
    preparations = sum(row["shots"] for row in annealing_demo["readout"])
    assert preparations == annealing_demo["total_preparations"] == 12288
    assert resources["total_preparations"] == preparations
    assert resources["logical_qubits"] == 4
    assert resources["measured_qubit_readouts"] == 4 * preparations
    expected_per_preparation = {"h": 4, "rx": 2 * 4 * 256, "rz": 4 * 256, "cx": 2 * 4 * 256, "measure": 4}
    for gate, expected in expected_per_preparation.items():
        assert resources["per_preparation_gate_counts"][gate] == expected
        assert resources["shot_weighted_gate_counts"][gate] == expected * preparations
    assert resources["rotation_count"] == 3072
    assert len(resources["simulated_rotation_angles"]) == resources["rotation_count"]
    assert resources["main_parameters"] == {"total_time": 16.0, "steps": 256}
    # Abstract Hamiltonian time is retained as a parameter, without presenting
    # it as the duration of a physical Clifford+T or surface-code execution.
    assert resources["physical_resource_estimate"] is None
    assert annealing_demo["workflow_wall_seconds"] >= 0


@pytest.mark.parametrize("points", [1, True, 4098])
def test_gap_scan_bounds_are_checked(points):
    with pytest.raises(ValueError):
        opt.annealing_gap_scan(4, CYCLE_EDGES, points=points)


@pytest.mark.parametrize("rtol,atol", [(0.0, 1e-13), (1e-11, -1.0), (math.nan, 1e-13)])
def test_continuous_reference_rejects_invalid_solver_tolerances(rtol, atol):
    with pytest.raises(ValueError):
        opt.annealing_continuous_reference(4, CYCLE_EDGES, 4.0, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_qubits,total_time,steps", [(33, 1.0, 1), (4, 65.0, 1), (4, 1.0, 4097)])
def test_generator_has_explicit_width_time_and_step_limits(n_qubits, total_time, steps):
    edges = [(0, n_qubits - 1)]
    with pytest.raises(ValueError):
        opt.build_annealing_circuit(n_qubits, edges, total_time, steps)


def test_large_finite_time_is_rejected_before_statevector_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("time preflight must run before statevector allocation")

    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    with pytest.raises(ValueError):
        opt.annealing_state(4, CYCLE_EDGES, 33.0, 1)


def test_large_finite_time_is_rejected_before_dense_reference_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("time preflight must run before a dense reference")

    monkeypatch.setattr("scipy.integrate.solve_ivp", forbidden)
    monkeypatch.setattr("scipy.linalg.expm", forbidden)
    with pytest.raises(ValueError):
        opt.annealing_continuous_reference(4, CYCLE_EDGES, 1e50)
    with pytest.raises(ValueError):
        opt.annealing_dense_strang_reference(4, CYCLE_EDGES, 33.0, 1)
