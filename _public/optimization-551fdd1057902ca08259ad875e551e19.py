# SPDX-License-Identifier: MIT
"""Transparent bounded QAOA/MaxCut and two-qubit VQE workflows.

Only NumPy and Qiskit core are used. Exact reference objectives consume zero
shots. Separate finite-shot objectives and optimizer trajectories use seeded
multinomial outcomes from explicit ideal measured-circuit probabilities.
Circuit materialization, classical enumeration, and simulation have separate
limits. Arbitrary rotations are explicit and unsynthesized.
"""
from __future__ import annotations

from collections import Counter
import math
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

CYCLE_EDGES = ((0, 1), (1, 2), (2, 3), (0, 3))
ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
OPTIMIZER_SETTINGS = dict(initial_step=0.4, step_tolerance=1e-6,
                          improvement_tolerance=1e-12, max_iterations=128)
QAOA_INITIALIZATIONS = ((0.2, 0.1), (0.6, 0.4), (-0.3, 0.25))
VQE_INITIALIZATIONS = ((0.6,), (-1.1,), (2.2,))
SAMPLING_SEEDS = (7, 19, 41)
DEFAULT_VQE_COEFFICIENTS = (1.0, 1.0, 0.5)
PORTABILITY_VQE_COEFFICIENTS = (0.8, 0.6, 0.4)
TRIANGLE_EDGES = ((0, 1), (1, 2), (0, 2))


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _angles(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if not result or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain finite angles")
    return result


def _graph(n_qubits: int, edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    _positive_int(n_qubits, "n_qubits")
    if n_qubits > 4096:
        raise ValueError("circuit materialization supports at most 4096 qubits")
    result = []
    seen = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("each edge must have two endpoints")
        u, v = edge
        if (isinstance(u, bool) or isinstance(v, bool)
                or not isinstance(u, int) or not isinstance(v, int)
                or not 0 <= u < n_qubits or not 0 <= v < n_qubits or u == v):
            raise ValueError("edges require distinct integer endpoints within the register")
        key = tuple(sorted((u, v)))
        if key in seen:
            raise ValueError("parallel or duplicate edges are unsupported")
        seen.add(key)
        result.append(key)
        if len(result) > 100_000:
            raise ValueError("circuit materialization supports at most 100000 edges")
    if not result:
        raise ValueError("at least one edge is required")
    return tuple(result)


def build_qaoa_circuit(n_qubits: int, edges: Iterable[tuple[int, int]],
                       gammas: Sequence[float], betas: Sequence[float], *,
                       measure: bool = False) -> QuantumCircuit:
    """Prepare exp(-i beta sum X) exp(-i gamma C)|+> for each layer.

    C=sum_edges(1-Zu Zv)/2. CX-RZ(-gamma)-CX realizes the phase
    separator up to a global phase. RX(2 beta) uses Qiskit's half-angle
    convention. No graph matrix, opaque oracle, or initialization unitary
    is inserted. Generation is O(n + p(n+m)); at most 100000 unitary
    gates are materialized, before optional final measurements. This bound
    is unrelated to statevector limits. Measurements are counted in the ledger.
    """
    graph = _graph(n_qubits, edges)
    gamma = _angles(gammas, "gammas")
    beta = _angles(betas, "betas")
    if len(gamma) != len(beta):
        raise ValueError("gammas and betas must have the same nonzero length")
    if n_qubits + len(gamma) * (3 * len(graph) + n_qubits) > 100_000:
        raise ValueError("circuit exceeds the 100000 unitary-gate materialization limit")
    circuit = QuantumCircuit(n_qubits, name="QAOA MaxCut")
    circuit.h(range(n_qubits))
    for g, b in zip(gamma, beta):
        for u, v in graph:
            circuit.cx(u, v)
            circuit.rz(-g, v)
            circuit.cx(u, v)
        circuit.rx(2 * b, range(n_qubits))
    if measure:
        circuit.measure_all()
    return circuit


def maxcut_values(n_qubits: int, edges: Iterable[tuple[int, int]], *,
                  max_qubits: int = 16) -> np.ndarray:
    """Enumerate every bit assignment; integer bit q is wire q (little endian)."""
    graph = _graph(n_qubits, edges)
    _positive_int(max_qubits, "max_qubits")
    if n_qubits > min(max_qubits, 20):
        raise ValueError("classical enumeration exceeds its configured limit (hard cap 20)")
    return np.array([sum(((x >> u) & 1) != ((x >> v) & 1) for u, v in graph)
                     for x in range(1 << n_qubits)], dtype=int)


def qaoa_expectation(n_qubits: int, edges: Iterable[tuple[int, int]],
                     gammas: Sequence[float], betas: Sequence[float], *,
                     max_qubits: int = 12) -> float:
    """Exact ideal expectation, with no measurement shots and a hard cap of 16."""
    _positive_int(max_qubits, "max_qubits")
    _positive_int(n_qubits, "n_qubits")
    if n_qubits > min(max_qubits, 16):
        raise ValueError("statevector exceeds its configured limit (hard cap 16)")
    graph = _graph(n_qubits, edges)
    values = maxcut_values(n_qubits, graph, max_qubits=max_qubits)
    state = Statevector.from_instruction(build_qaoa_circuit(n_qubits, graph, gammas, betas))
    return float(state.probabilities() @ values)


def build_vqe_circuit(theta: float, *, basis: str | None = None,
                      measure: bool = False) -> QuantumCircuit:
    """cos(theta/2)|00>+sin(theta/2)|11>, optionally measured in all-Z/all-X."""
    angle = _angles((theta,), "theta")[0]
    if basis not in (None, "z", "x"):
        raise ValueError("basis must be None, 'z', or 'x'")
    circuit = QuantumCircuit(2, name="VQE even-parity ansatz")
    circuit.ry(angle, 0)
    circuit.cx(0, 1)
    if basis == "x":
        circuit.h(range(2))
    if measure:
        circuit.measure_all()
    return circuit


def _vqe_coefficients(coefficients: Sequence[float]) -> tuple[float, float, float]:
    values = _angles(coefficients, "coefficients")
    if len(values) != 3:
        raise ValueError("coefficients must specify a,b,c for H=-a Z0-b Z1+c X0X1")
    return values


def vqe_hamiltonian(coefficients: Sequence[float] = DEFAULT_VQE_COEFFICIENTS) -> np.ndarray:
    """Explicit H=-a Z0-b Z1+c X0X1 in |q1 q0> order.

    The even-parity ansatz need not contain the full ground state for general
    coefficients (opposite-sign a,b can put it in the odd subspace).
    """
    a, b, c = _vqe_coefficients(coefficients)
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    return -a * np.kron(np.eye(2), z) - b * np.kron(z, np.eye(2)) + c * np.kron(x, x)


def vqe_expectation(theta: float, coefficients: Sequence[float] = DEFAULT_VQE_COEFFICIENTS) -> float:
    state = Statevector.from_instruction(build_vqe_circuit(theta)).data
    return float(np.vdot(state, vqe_hamiltonian(coefficients) @ state).real)


def compile_rotation_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Deterministic logical decomposition; rotations remain unsynthesized."""
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS,
                         optimization_level=0, seed_transpiler=7)
    unsupported = set(compiled.count_ops()) - set(ROTATION_BASIS) - {"barrier", "measure"}
    if unsupported:
        raise RuntimeError(f"unsupported operations: {unsupported}")
    return compiled


def pattern_search(objective: Callable[[Sequence[float]], float],
                   initial: Sequence[float], *, initial_step: float = 0.4,
                   step_tolerance: float = 1e-6, improvement_tolerance: float = 1e-12,
                   max_iterations: int = 128, objective_method: str = "provided scalar objective") -> dict:
    """Deterministic coordinate polling, minimizing a periodic scalar objective.

    Each sweep evaluates minus then plus for every coordinate from the same
    incumbent, accepts the best strict improvement, or halves the step.
    Angles are reduced to [-pi,pi). All calls, including rejected candidates,
    are recorded without caching. Stopping is step<tol or the sweep cap;
    no claim of general/global convergence is made.
    """
    parameters = np.array(_angles(initial, "initial"), dtype=float)
    if len(parameters) > 64:
        raise ValueError("optimizer supports at most 64 parameters")
    _positive_int(max_iterations, "max_iterations")
    if max_iterations > 10_000:
        raise ValueError("optimizer supports at most 10000 sweeps")
    if not (math.isfinite(initial_step) and math.isfinite(step_tolerance)
            and 0 < step_tolerance < initial_step
            and math.isfinite(improvement_tolerance) and improvement_tolerance >= 0):
        raise ValueError("require finite 0<step_tolerance<initial_step and nonnegative improvement tolerance")
    evaluations, iterations = [], []
    started = perf_counter()

    def evaluate(point):
        point = (np.asarray(point) + math.pi) % (2 * math.pi) - math.pi
        begin = perf_counter()
        value = float(objective(point.tolist()))
        elapsed = perf_counter() - begin
        if not math.isfinite(value):
            raise ValueError("objective returned a nonfinite value")
        index = len(evaluations)
        evaluations.append(dict(index=index, parameters=point.tolist(), value=value, seconds=elapsed,
                                method=objective_method, shots=0))
        return point, value, index

    parameters, value, accepted_index = evaluate(parameters)
    step = float(initial_step)
    reason = "max_iterations"
    for iteration in range(max_iterations):
        if step < step_tolerance:
            reason = "step_tolerance"
            break
        best_parameters, best_value, best_index = parameters, value, accepted_index
        sweep_step = step
        first_index = len(evaluations)
        for axis in range(len(parameters)):
            for direction in (-1, 1):
                trial = parameters.copy()
                trial[axis] += direction * step
                point, candidate, index = evaluate(trial)
                if candidate < best_value - improvement_tolerance:
                    best_parameters, best_value, best_index = point, candidate, index
        accepted = best_value < value - improvement_tolerance
        if accepted:
            parameters, value, accepted_index = best_parameters, best_value, best_index
        else:
            step /= 2
        iterations.append(dict(index=iteration, step=sweep_step, next_step=step,
                               accepted=accepted, parameters=parameters.tolist(), value=value,
                               incumbent_evaluation=accepted_index,
                               evaluation_indices=list(range(first_index, len(evaluations)))))
    if step < step_tolerance:
        reason = "step_tolerance"
    elapsed = perf_counter() - started
    objective_seconds = sum(x["seconds"] for x in evaluations)
    return dict(parameters=parameters.tolist(), value=value, evaluations=evaluations,
                iterations=iterations, objective_evaluations=len(evaluations),
                iteration_count=len(iterations), stop_reason=reason, final_step=step,
                settings=dict(initial_step=initial_step, step_tolerance=step_tolerance,
                              improvement_tolerance=improvement_tolerance, max_iterations=max_iterations,
                              periodic_domain="[-pi,pi)", polling_order="axis ascending, minus then plus"),
                initial=list(initial), optimizer_wall_seconds=elapsed,
                objective_wall_seconds=objective_seconds,
                classical_control_seconds=max(0.0, elapsed - objective_seconds), shots=0)


def _sample(circuit: QuantumCircuit, values: np.ndarray, *, shots: int, seed: int) -> dict:
    _positive_int(shots, "shots")
    if shots > 10_000_000:
        raise ValueError("sampling supports at most 10000000 shots per group")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    probability_begin = perf_counter()
    probabilities = Statevector.from_instruction(circuit).probabilities()
    probabilities = probabilities / probabilities.sum()
    probability_seconds = perf_counter() - probability_begin
    begin = perf_counter()
    counts = np.random.Generator(np.random.PCG64(seed)).multinomial(shots, probabilities)
    mean = float(counts @ values / shots)
    sample_variance = float(counts @ ((values - mean) ** 2) / (shots - 1)) if shots > 1 else None
    exact = float(probabilities @ values)
    return dict(seed=seed, shots=shots, counts={f"{i:0{circuit.num_qubits}b}": int(n)
                                             for i, n in enumerate(counts) if n},
                estimate=mean, exact_expectation=exact, error=mean - exact,
                sample_variance=sample_variance,
                estimated_standard_error=math.sqrt(sample_variance / shots) if shots > 1 else None,
                sampling_seconds=perf_counter() - begin, probability_generation_seconds=probability_seconds,
                method="ideal multinomial from statevector probabilities; no device noise",
                rng="NumPy Generator(PCG64)")


def sample_qaoa(gamma: float, beta: float, *, shots: int = 2048, seed: int = 7) -> dict:
    """One all-Z group yields all edge costs jointly, including correlations."""
    values = maxcut_values(4, CYCLE_EDGES)
    result = _sample(build_qaoa_circuit(4, CYCLE_EDGES, [gamma], [beta]), values,
                     shots=shots, seed=seed)
    result["group"] = "all_z"
    result["measured_qubits_per_shot"] = 4
    result["best_observed_cut"] = max(int(values[int(bits, 2)]) for bits in result["counts"])
    result["optimal_cut_samples"] = sum(count for bits, count in result["counts"].items()
                                       if values[int(bits, 2)] == 4)
    result["optimal_cut_sample_fraction"] = result["optimal_cut_samples"] / shots
    result["hoeffding_95_mean_error_bound"] = 4 * math.sqrt(math.log(40) / (2 * shots))
    return result


def sample_vqe(theta: float, *, shots: int = 2048, seed: int = 7) -> dict:
    """Two independent groups; Z0 and Z1 share shots, X0X1 uses H,H readout.

    The all-X stream uses seed+1000003. Group variances, including Z term
    covariance, are propagated for a standard error (not a rigorous CI).
    A separate Hoeffding bound uses group ranges 4 and 1.
    """
    values_z = np.array([-2.0, 0.0, 0.0, 2.0])
    values_x = np.array([0.5, -0.5, -0.5, 0.5])
    z = _sample(build_vqe_circuit(theta, basis="z"), values_z, shots=shots, seed=seed)
    x = _sample(build_vqe_circuit(theta, basis="x"), values_x, shots=shots, seed=seed + 1_000_003)
    z.update(group="all_z", observable="-Z0-Z1", measured_qubits_per_shot=2)
    x.update(group="all_x", observable="0.5 X0X1", measured_qubits_per_shot=2)
    estimate = z["estimate"] + x["estimate"]
    exact = z["exact_expectation"] + x["exact_expectation"]
    return dict(seed=seed, groups=[z, x], total_shots=2 * shots,
                estimate=estimate, exact_expectation=exact, error=estimate - exact,
                estimated_standard_error=math.sqrt(z["sample_variance"] / shots + x["sample_variance"] / shots)
                if shots > 1 else None,
                hoeffding_95_energy_error_bound=math.sqrt((16 / shots + 1 / shots) * math.log(40) / 2),
                ground_energy=-math.sqrt(4.25), note="sample energy may fall below the exact ground energy")


def run_demo(*, shots: int = 2048, seeds: Sequence[int] = SAMPLING_SEEDS) -> dict:
    """Run three predeclared initializations and held-out readout at the first.

    Records every rejected/accepted exact objective call. Independent
    analytic identities are checked at every call for these two fixed toys.
    Timing is measured on this CPU/software and is not hardware runtime.
    """
    workflow_begin = perf_counter()
    values = maxcut_values(4, CYCLE_EDGES)
    eigvals = np.linalg.eigvalsh(vqe_hamiltonian())

    def q_objective(point):
        gamma, beta = point
        value = qaoa_expectation(4, CYCLE_EDGES, [gamma], [beta])
        analytic = 2 + math.sin(4 * beta) * math.sin(2 * gamma)
        if abs(value - analytic) > 1e-10:
            raise AssertionError("QAOA expectation disagrees with C4 analytic answer")
        return -value

    def v_objective(point):
        value = vqe_expectation(point[0])
        if abs(value - (-2 * math.cos(point[0]) + 0.5 * math.sin(point[0]))) > 1e-10:
            raise AssertionError("VQE expectation disagrees with analytic answer")
        return value

    q_runs = [pattern_search(q_objective, initial, objective_method="exact_statevector", **OPTIMIZER_SETTINGS)
              for initial in QAOA_INITIALIZATIONS]
    v_runs = [pattern_search(v_objective, initial, objective_method="exact_statevector", **OPTIMIZER_SETTINGS)
              for initial in VQE_INITIALIZATIONS]
    for run in q_runs:
        gamma, beta = run["parameters"]
        probabilities = Statevector.from_instruction(build_qaoa_circuit(4, CYCLE_EDGES, [gamma], [beta])).probabilities()
        probability = float(probabilities[values == values.max()].sum())
        run.update(achieved_expectation=-run["value"], approximation_ratio=-run["value"] / 4,
                   optimal_cut_probability=probability,
                   independent_trials_for_99_percent_one_optimum=math.ceil(math.log(0.01) / math.log1p(-probability)),
                   target=dict(exact_expected_cut_at_least=3 - 1e-8),
                   target_met=-run["value"] >= 3 - 1e-8)
    for run in v_runs:
        gap = run["value"] - float(eigvals[0])
        run.update(ground_energy_gap=gap, target=dict(exact_energy_gap_at_most=1e-8), target_met=gap <= 1e-8)
    gamma, beta = q_runs[0]["parameters"]
    theta = v_runs[0]["parameters"][0]
    result = dict(schema_version=1,
                scope="bounded ideal simulation and ideal finite-shot readout; no device experiment",
                primary_selection="first predeclared initialization, chosen before outcomes",
                qaoa=dict(n_qubits=4, edges=[list(e) for e in CYCLE_EDGES], layers=1,
                          objective="maximize C=sum_edges(1-Zu Zv)/2; optimizer minimizes -<C>",
                          parameters_order=["gamma", "beta"], angles="radians; RZ(-gamma), RX(2 beta)",
                          classical=dict(method="all 16 assignments enumerated", values=values.tolist(), optimum=int(values.max()),
                                         optimal_bitstrings=[f"{i:04b}" for i, v in enumerate(values) if v == values.max()]),
                          optimization_runs=q_runs,
                          readout=[sample_qaoa(gamma, beta, shots=shots, seed=seed) for seed in seeds]),
                vqe=dict(n_qubits=2, hamiltonian="-Z0-Z1+0.5 X0X1", units="dimensionless energy, radians",
                         ansatz="RY(theta) on q0; CX(0,1); initial |00>",
                         reference=dict(method="NumPy eigvalsh of explicit 4x4 matrix", eigenvalues=eigvals.tolist(),
                                        ground_energy=float(eigvals[0]), analytic_ground_energy=-math.sqrt(4.25),
                                        analytic_optimal_theta=-math.atan(0.25)),
                         optimization_runs=v_runs,
                         readout=[sample_vqe(theta, shots=shots, seed=seed) for seed in seeds]),
                exclusions=["device noise and mitigation", "sampled/noisy optimization", "input transfer and classical graph parsing time",
                            "classical reference solver timing separately unmeasured; included in whole CPU workflow timer",
                            "reset/measurement physical time and routing",
                            "rotation synthesis, error correction, factories and hardware runtime"],
                physical_resource_estimate=None)
    result["hybrid_workflow_wall_seconds"] = perf_counter() - workflow_begin
    result["timing_scope"] = ("run_demo CPU wall time includes reference checks, all optimization calls/control, "
                              "post-fit validation and ideal readout probability generation/sampling; "
                              "excludes compilation, export, plotting, input parsing, and file I/O")
    return result


def logical_run_resources(demo: dict) -> dict:
    """Count each simulated exact ansatz and modeled ideal readout shot.

    Exact objectives are statevector simulations, never hardware executions.
    Counts repeat the actual per-evaluation circuits; multinomial draws model
    the stated readout protocol rather than executing hardware. Simulation
    effort is reported by measured timings separately. Depths are per circuit;
    summing depth is an explicitly serial gate-layer sum, not a runtime.
    """
    records = {}
    for algorithm in ("qaoa", "vqe"):
        section = demo[algorithm]
        exact_counts, readout_counts = Counter(), Counter()
        exact_layers = readout_layers = 0
        rotations = []
        evaluation_inputs = []
        for run_index, run in enumerate(section["optimization_runs"]):
            for evaluation in run["evaluations"]:
                params = evaluation["parameters"]
                circuit = (build_qaoa_circuit(4, CYCLE_EDGES, [params[0]], [params[1]])
                           if algorithm == "qaoa" else build_vqe_circuit(params[0]))
                compiled = compile_rotation_basis(circuit)
                exact_counts.update(compiled.count_ops())
                exact_layers += compiled.depth()
                evaluation_inputs.append(dict(run=run_index, evaluation=evaluation["index"], parameters=params,
                                              circuit_gate_counts=dict(compiled.count_ops()), circuit_depth=compiled.depth(), shots=0))
                for instruction in compiled.data:
                    if instruction.operation.name in ("rx", "ry", "rz"):
                        rotations.append(float(instruction.operation.params[0]))
        params = section["optimization_runs"][0]["parameters"]
        groups = [("all_z", build_qaoa_circuit(4, CYCLE_EDGES, [params[0]], [params[1]], measure=True))] if algorithm == "qaoa" else [
            ("all_z", build_vqe_circuit(params[0], basis="z", measure=True)),
            ("all_x", build_vqe_circuit(params[0], basis="x", measure=True))]
        group_inputs = []
        for name, circuit in groups:
            compiled = compile_rotation_basis(circuit)
            matching = section["readout"] if algorithm == "qaoa" else [g for experiment in section["readout"]
                                                                           for g in experiment["groups"] if g["group"] == name]
            group_shots = sum(r["shots"] for r in matching)
            for gate, count in compiled.count_ops().items():
                if gate != "barrier":
                    readout_counts[gate] += count * group_shots
            readout_layers += compiled.depth() * group_shots
            group_inputs.append(dict(group=name, parameters=params, shots=group_shots,
                                     circuit_gate_counts=dict(compiled.count_ops()), circuit_depth=compiled.depth(),
                                     experiment_seeds=[r["seed"] for r in matching]))
        measured_shots = sum(g["shots"] for g in group_inputs)
        records[algorithm] = dict(logical_qubits=section["n_qubits"], ancillas=0,
                                 classical_input_model="explicit classical edge list" if algorithm == "qaoa" else "three explicitly stated Pauli coefficients",
                                 initialization="reset to all-zero assumed once per shot; QAOA H and VQE RY/CX counted",
                                 exact_objective_evaluations=len(evaluation_inputs), exact_statevector_shots=0,
                                 exact_simulated_gate_counts=dict(exact_counts),
                                 exact_simulated_serial_depth_sum=exact_layers,
                                 exact_evaluation_inputs=evaluation_inputs,
                                 held_out_measurement_circuit_shots=measured_shots,
                                 measured_qubit_readouts=measured_shots * section["n_qubits"],
                                 shot_weighted_logical_gate_counts=dict(readout_counts),
                                 shot_weighted_serial_depth_sum=readout_layers,
                                 readout_group_inputs=group_inputs,
                                 optimizer_wall_seconds=sum(r["optimizer_wall_seconds"] for r in section["optimization_runs"]),
                                 objective_wall_seconds=sum(r["objective_wall_seconds"] for r in section["optimization_runs"]),
                                 classical_control_seconds=sum(r["classical_control_seconds"] for r in section["optimization_runs"]),
                                 finite_shot_sampling_seconds=sum(r.get("sampling_seconds", 0) for r in section["readout"])
                                 if algorithm == "qaoa" else sum(g["sampling_seconds"] for r in section["readout"] for g in r["groups"]),
                                 readout_probability_generation_seconds=sum(r["probability_generation_seconds"] for r in section["readout"])
                                 if algorithm == "qaoa" else sum(g["probability_generation_seconds"] for r in section["readout"] for g in r["groups"]),
                                 simulated_rotation_angles=rotations)
    return dict(schema_version=1, resource_kind="pre-synthesis logical ledger for exact simulations and the modeled ideal readout protocol",
                compilation=dict(basis=ROTATION_BASIS, optimization_level=0, seed_transpiler=7,
                                 scheduling="per-circuit Qiskit depth, hypothetical serial depth sum; no routing/device schedule"),
                algorithms=records, physical_resource_estimate=None,
                hybrid_workflow_wall_seconds=demo["hybrid_workflow_wall_seconds"], timing_scope=demo["timing_scope"],
                synthesis_status="pending; arbitrary rotations are neither T gates nor stochastic failures",
                fault_tolerant_handoff=dict(required=["match problem/output/accuracy/success and hardware baseline",
                                                    "synthesize every angle with an explicit coherent approximation metric",
                                                    "separate statistical error, coherent error and physical fault budget",
                                                    "count every optimization/readout repetition and classical latency"],
                                           representative_exports=["qaoa.qasm", "vqe-z.qasm", "vqe-x.qasm"],
                                           per_evaluation_parameters="runs.json and exact_evaluation_inputs; representative exports alone are insufficient"),
                exclusions=demo["exclusions"])


# Each block is (coordinate step in radians, shots per group, fixed sweeps).
FINITE_SHOT_SCHEDULE = ((0.4, 256, 8), (0.2, 512, 8), (0.1, 1024, 8), (0.05, 2048, 8))


def _sample_measured_scalar(circuit: QuantumCircuit, values: np.ndarray, *,
                            shots: int, seed: int, algorithm_id: int,
                            family: int, evaluation_index: int, group_index: int) -> dict:
    """Draw ideal measured outcomes from one explicit final-measurement circuit.

    The statevector backend removes only final measurements to obtain their
    ideal probabilities. The returned estimate/variance use sampled counts;
    no exact observable mean is returned to the optimizer. RNG entropy and
    spawn keys distinguish algorithms, trials, calls, bases and validation.
    """
    _positive_int(shots, "shots")
    if not 2 <= shots <= 65_536:
        raise ValueError("finite-shot objectives require 2..65536 shots per group")
    for name, value in (("seed", seed), ("algorithm_id", algorithm_id), ("family", family),
                        ("evaluation_index", evaluation_index), ("group_index", group_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if circuit.num_qubits > 12:
        raise ValueError("finite-shot statevector sampling is capped at 12 qubits")
    if circuit.count_ops().get("measure", 0) != circuit.num_qubits:
        raise ValueError("finite-shot circuit must measure every qubit at the end")
    started = perf_counter()
    unitary = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector.from_instruction(unitary).probabilities()
    probabilities /= probabilities.sum()
    seed_sequence = np.random.SeedSequence([seed, algorithm_id],
                                          spawn_key=(family, evaluation_index, group_index))
    counts = np.random.Generator(np.random.PCG64(seed_sequence)).multinomial(shots, probabilities)
    estimate = float(counts @ values / shots)
    variance = float(counts @ ((values - estimate) ** 2) / (shots - 1))
    return dict(shots=shots,
                counts={f"{i:0{circuit.num_qubits}b}": int(n) for i, n in enumerate(counts) if n},
                estimate=estimate, sample_variance=variance,
                estimated_standard_error=math.sqrt(variance / shots),
                rng=dict(bit_generator="PCG64", entropy=[seed, algorithm_id],
                         spawn_key=[family, evaluation_index, group_index],
                         stream_key=[seed, algorithm_id, family, evaluation_index, group_index]),
                circuit_gate_counts={gate: int(n) for gate, n in circuit.count_ops().items()},
                circuit_depth=circuit.depth(), logical_qubits=circuit.num_qubits,
                measured_qubit_readouts=shots * circuit.num_qubits,
                group_evaluation_seconds=perf_counter() - started,
                method="ideal circuit-sampling simulation; no QPU execution",
                shot_weighted_counts_semantics="modeled preparations/operations, not measured hardware consumption")


def sampled_qaoa_objective(parameters: Sequence[float], *, shots: int, seed: int,
                           evaluation_index: int, family: int = 0,
                           n_qubits: int = 4, edges: Iterable[tuple[int, int]] = CYCLE_EDGES,
                           stream_id: int = 0) -> dict:
    """Minimize the negative joint MaxCut mean from one all-Z shot group."""
    point = _angles(parameters, "parameters")
    if len(point) != 2:
        raise ValueError("this sampled QAOA workflow requires gamma,beta for one layer")
    if n_qubits > 12:
        raise ValueError("finite-shot statevector sampling is capped at 12 qubits")
    graph = _graph(n_qubits, edges)
    started = perf_counter()
    measured = build_qaoa_circuit(n_qubits, graph, [point[0]], [point[1]], measure=True)
    values = maxcut_values(n_qubits, graph)
    group = _sample_measured_scalar(measured, values, shots=shots, seed=seed, algorithm_id=stream_id,
                                   family=family, evaluation_index=evaluation_index, group_index=0)
    group.update(group="all_z", observable="joint cut C", outcome_min=int(values.min()), outcome_max=int(values.max()))
    return dict(parameters=list(point), objective_estimate=-group["estimate"],
                estimated_standard_error=group["estimated_standard_error"], groups=[group],
                total_shots=shots, evaluation_wall_seconds=perf_counter() - started,
                method="ideal finite-shot objective", objective_convention="minimize -sample mean cut")


def sampled_vqe_objective(parameters: Sequence[float], *, shots: int, seed: int,
                          evaluation_index: int, family: int = 0,
                          coefficients: Sequence[float] = DEFAULT_VQE_COEFFICIENTS,
                          stream_id: int = 1) -> dict:
    """Sum independent Z/X group means, retaining joint Z0/Z1 covariance."""
    point = _angles(parameters, "parameters")
    if len(point) != 1:
        raise ValueError("this VQE ansatz requires one theta")
    a, b, c = _vqe_coefficients(coefficients)
    started = perf_counter()
    groups = []
    z0 = np.array([1.0, -1.0, 1.0, -1.0])
    z1 = np.array([1.0, 1.0, -1.0, -1.0])
    z_width_half = abs(a) + abs(b)
    for group_index, (basis, values, observable, interval) in enumerate((
            ("z", -a * z0 - b * z1, f"joint -{a:g} Z0-{b:g} Z1", (-z_width_half, z_width_half)),
            ("x", c * z0 * z1, f"{c:g} X0X1", (-abs(c), abs(c))))):
        measured = build_vqe_circuit(point[0], basis=basis, measure=True)
        group = _sample_measured_scalar(measured, values, shots=shots, seed=seed, algorithm_id=stream_id,
                                       family=family, evaluation_index=evaluation_index, group_index=group_index)
        group.update(group=f"all_{basis}", observable=observable,
                     outcome_min=interval[0], outcome_max=interval[1])
        groups.append(group)
    return dict(parameters=list(point), objective_estimate=sum(g["estimate"] for g in groups),
                estimated_standard_error=math.sqrt(sum(g["sample_variance"] / g["shots"] for g in groups)),
                groups=groups, total_shots=2 * shots,
                evaluation_wall_seconds=perf_counter() - started,
                method="ideal finite-shot objective", objective_convention="minimize sum of Z/X sample means")


def sampled_pattern_search(objective: Callable[..., dict], initial: Sequence[float], *,
                            seed: int, schedule: Sequence[tuple[float, int, int]] = FINITE_SHOT_SCHEDULE,
                            initial_shots: int = 256, comparison_z: float = 2.0) -> dict:
    """Fixed-budget polling using fresh sampled incumbents and proposals.

    A candidate must beat the fresh incumbent by comparison_z times their
    combined descriptive standard error. Selection over candidates and an
    adaptive parameter path mean this margin is not a hypothesis-test or
    convergence certificate. The only stopping rule is the predeclared end
    of the schedule; no exact objective is consulted by this function.
    """
    point = np.asarray(_angles(initial, "initial"), dtype=float)
    if len(point) > 8:
        raise ValueError("sampled polling supports at most eight coordinates")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(comparison_z) or comparison_z < 0:
        raise ValueError("comparison_z must be finite and nonnegative")
    if not 2 <= _positive_int(initial_shots, "initial_shots") <= 65_536:
        raise ValueError("initial_shots must lie in 2..65536")
    blocks, expanded = [], []
    for step, shots, sweeps in schedule:
        _positive_int(shots, "shots")
        _positive_int(sweeps, "sweeps")
        if not math.isfinite(step) or step <= 0 or not 2 <= shots <= 65_536:
            raise ValueError("schedule requires finite positive steps and 2..65536 shots")
        if len(expanded) + sweeps > 128:
            raise ValueError("sampled schedule is capped at 128 sweeps")
        blocks.append(dict(step=float(step), shots_per_group=shots, sweeps=sweeps))
        expanded.extend([(float(step), shots)] * sweeps)
    if not expanded:
        raise ValueError("schedule cannot be empty")
    # Preflight worst-case two measurement groups; no large sampling job is launched.
    planned_group_shots = initial_shots + (1 + 2 * len(point)) * sum(s for _, s in expanded)
    if 2 * planned_group_shots > 20_000_000:
        raise ValueError("sampled schedule exceeds the 20000000-shot workload cap")
    point = (point + math.pi) % (2 * math.pi) - math.pi
    started = perf_counter()
    evaluations, iterations = [], []

    def evaluate(parameters, shots, role):
        parameters = (np.asarray(parameters) + math.pi) % (2 * math.pi) - math.pi
        index = len(evaluations)
        record = objective(parameters.tolist(), shots=shots, seed=seed,
                           evaluation_index=index, family=0)
        value, standard_error = record["objective_estimate"], record["estimated_standard_error"]
        if not math.isfinite(value) or not math.isfinite(standard_error) or standard_error < 0:
            raise ValueError("sampled objective must return a finite estimate and standard error")
        if any(g["shots"] != shots for g in record["groups"]):
            raise ValueError("objective violated per-group shot allocation")
        if not record["groups"] or record["total_shots"] != sum(g["shots"] for g in record["groups"]):
            raise ValueError("objective must record all measured groups and total shots")
        record.update(index=index, role=role, accepted=False, parameters=parameters.tolist())
        evaluations.append(record)
        return record

    current = evaluate(point, initial_shots, "initial")
    for iteration, (step, shots) in enumerate(expanded):
        before = point.copy()
        incumbent = evaluate(point, shots, "incumbent")
        proposals = []
        for axis in range(len(point)):
            for direction in (-1, 1):
                trial = point.copy()
                trial[axis] += direction * step
                proposal = evaluate(trial, shots, "proposal")
                proposal.update(axis=axis, direction=direction)
                proposals.append(proposal)
        best = min(proposals, key=lambda e: e["objective_estimate"])
        margin = comparison_z * math.hypot(incumbent["estimated_standard_error"], best["estimated_standard_error"])
        improvement = incumbent["objective_estimate"] - best["objective_estimate"]
        accepted = improvement > margin
        if accepted:
            point = np.asarray(best["parameters"])
            best["accepted"] = True
            current = best
        else:
            current = incumbent
        iterations.append(dict(index=iteration, step=step, shots_per_group=shots,
                               parameters_before=before.tolist(), parameters_after=point.tolist(),
                               incumbent_evaluation=incumbent["index"],
                               proposal_evaluations=[e["index"] for e in proposals],
                               best_proposal_evaluation=best["index"], accepted=accepted,
                               sampled_improvement=improvement, comparison_margin=margin))
    wall = perf_counter() - started
    evaluation_seconds = sum(e["evaluation_wall_seconds"] for e in evaluations)
    return dict(parameters=point.tolist(), initial=list(initial), seed=seed,
                final_training_estimate=current["objective_estimate"],
                final_training_standard_error=current["estimated_standard_error"],
                evaluations=evaluations, iterations=iterations,
                objective_evaluations=len(evaluations), iteration_count=len(iterations),
                total_shots=sum(e["total_shots"] for e in evaluations),
                accepted_proposals=sum(e["accepted"] for e in evaluations if e["role"] == "proposal"),
                rejected_proposals=sum(not e["accepted"] for e in evaluations if e["role"] == "proposal"),
                stop_reason="fixed_schedule_exhausted", convergence_certified=False,
                optimizer_wall_seconds=wall, objective_wall_seconds=evaluation_seconds,
                classical_control_seconds=max(0.0, wall - evaluation_seconds),
                settings=dict(schedule=blocks, initial_shots_per_group=initial_shots,
                              comparison_z=comparison_z, periodic_domain="[-pi,pi)",
                              polling_order="axis ascending, minus then plus; fresh incumbent each sweep",
                              acceptance_semantics="descriptive heuristic, not simultaneous confidence/convergence"))


def run_sampled_demo(*, schedule: Sequence[tuple[float, int, int]] = FINITE_SHOT_SCHEDULE,
                      validation_shots: int = 4096, seeds: Sequence[int] = SAMPLING_SEEDS) -> dict:
    """Three predeclared seed/start pairs; exact diagnostics follow decisions.

    Validation is an independent fixed-parameter family of RNG streams.
    Its fixed-mean Hoeffding bounds do not apply to the adaptive training
    trajectory. The paired changes of seed and start do not isolate either
    source of variability; they are three measured teaching cases.
    """
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("the demo requires three distinct predeclared seeds")
    started = perf_counter()
    result = dict(schema_version=1, method="ideal finite-shot optimizer trajectory; no QPU execution",
                  primary_selection="first predeclared seed/start pair before outcomes",
                  rng_contract="SeedSequence([seed,algorithm_id], spawn_key=(family,evaluation_index,group)); 0 training, 1 validation",
                  exact_diagnostics_used_for_optimization=False,
                  physical_resource_estimate=None, whole_trajectory_physical_status="pending",
                  qaoa=dict(n_qubits=4, edges=[list(e) for e in CYCLE_EDGES],
                            classical_optimum=4, p1_exact_optimum=3,
                            target=dict(post_run_exact_expected_cut_at_least=2.90), runs=[]),
                  vqe=dict(n_qubits=2, hamiltonian="-Z0-Z1+0.5 X0X1",
                           exact_ground_energy=float(np.linalg.eigvalsh(vqe_hamiltonian())[0]),
                           target=dict(post_run_exact_ground_energy_gap_at_most=0.03), runs=[]))
    for algorithm, objective, initials in (("qaoa", sampled_qaoa_objective, QAOA_INITIALIZATIONS),
                                           ("vqe", sampled_vqe_objective, VQE_INITIALIZATIONS)):
        for run_index, (seed, initial) in enumerate(zip(seeds, initials)):
            run = sampled_pattern_search(objective, initial, seed=seed, schedule=schedule)
            # Training is finished before either holdout or exact audit is computed.
            validation = objective(run["parameters"], shots=validation_shots, seed=seed,
                                   evaluation_index=0, family=1)
            validation.update(role="independent_final_validation", confidence_scope="one fixed mean conditional on fitted parameters")
            exact = (lambda p: -(2 + math.sin(4 * p[1]) * math.sin(2 * p[0]))) if algorithm == "qaoa" else (
                lambda p: -2 * math.cos(p[0]) + 0.5 * math.sin(p[0]))
            for evaluation in run["evaluations"]:
                evaluation["exact_objective_audit"] = exact(evaluation["parameters"])
            for iteration in run["iterations"]:
                iteration["exact_objective_after_audit"] = exact(iteration["parameters_after"])
            exact_final = exact(run["parameters"])
            validation.update(exact_objective_audit=exact_final,
                              estimation_error_from_exact=validation["objective_estimate"] - exact_final)
            if algorithm == "qaoa":
                validation["hoeffding_95_absolute_mean_radius"] = 4 * math.sqrt(math.log(40) / (2 * validation_shots))
                probabilities = Statevector.from_instruction(build_qaoa_circuit(
                    4, CYCLE_EDGES, [run["parameters"][0]], [run["parameters"][1]])).probabilities()
                values = maxcut_values(4, CYCLE_EDGES)
                run.update(exact_expected_cut_audit=-exact_final, approximation_ratio_audit=-exact_final / 4,
                           optimal_cut_probability_audit=float(probabilities[values == 4].sum()),
                           target_met_audit=-exact_final >= 2.90)
                validation["best_observed_cut"] = max(values[int(bits, 2)].item() for bits in validation["groups"][0]["counts"])
                validation["optimal_cut_sample_fraction"] = sum(
                    count for bits, count in validation["groups"][0]["counts"].items() if values[int(bits, 2)] == 4) / validation_shots
            else:
                validation["hoeffding_95_absolute_mean_radius"] = math.sqrt(17 / validation_shots * math.log(40) / 2)
                gap = exact_final - result["vqe"]["exact_ground_energy"]
                run.update(exact_energy_audit=exact_final, ground_energy_gap_audit=gap, target_met_audit=gap <= 0.03)
            run.update(run_index=run_index, validation=validation,
                       total_shots_including_validation=run["total_shots"] + validation["total_shots"],
                       exact_audit_source="independent scalar identity, computed only after sampled optimizer completion")
            result[algorithm]["runs"].append(run)
    result["workflow_wall_seconds"] = perf_counter() - started
    result["training_shots"] = sum(r["total_shots"] for a in ("qaoa", "vqe") for r in result[a]["runs"])
    result["validation_shots"] = sum(r["validation"]["total_shots"] for a in ("qaoa", "vqe") for r in result[a]["runs"])
    result["total_shots"] = result["training_shots"] + result["validation_shots"]
    result["projection_fixed_readout_selection"] = dict(
        selection_rule="first_predeclared_case", master_seed=seeds[0],
        shots_per_group=validation_shots,
        qaoa_parameters=result["qaoa"]["runs"][0]["parameters"],
        vqe_parameters=result["vqe"]["runs"][0]["parameters"])
    result["limitations"] = ["ideal sampling simulation; no device noise or QPU executions",
                              "fixed-budget heuristic with no certified convergence or optimizer success",
                              "three paired seeds/initializations, not a distributional performance theorem",
                              "training standard-error margins do not provide simultaneous confidence",
                              "validation bounds apply only to independent fixed-parameter readout",
                              "physical initialization time, routing, synthesis certificates and QEC runtime excluded"]
    return result


def sampled_run_resources(demo: dict) -> dict:
    """Count every measured group for every sampled call and heldout validation."""
    algorithms = {}
    for algorithm in ("qaoa", "vqe"):
        inputs, counts, shots = [], Counter(), 0
        for run in demo[algorithm]["runs"]:
            for phase, calls in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for call_index, call in enumerate(calls):
                    for group in call["groups"]:
                        weighted = {gate: number * group["shots"] for gate, number in group["circuit_gate_counts"].items()
                                    if gate != "barrier"}
                        counts.update(weighted)
                        shots += group["shots"]
                        inputs.append(dict(run_index=run["run_index"], seed=run["seed"], phase=phase,
                                           evaluation_index=call_index, parameters=call["parameters"],
                                           group=group["group"], shots=group["shots"], rng=group["rng"],
                                           logical_qubits=group["logical_qubits"],
                                           circuit_gate_counts=group["circuit_gate_counts"], circuit_depth=group["circuit_depth"]))
        algorithms[algorithm] = dict(logical_qubits=demo[algorithm]["n_qubits"], ancillas=0,
                                     objective_calls=sum(r["objective_evaluations"] for r in demo[algorithm]["runs"]),
                                     independent_validation_calls=len(demo[algorithm]["runs"]),
                                     measured_group_batches=len(inputs), total_modeled_circuit_shots=shots,
                                     total_modeled_qubit_readouts=counts.get("measure", 0),
                                     shot_weighted_gate_counts=dict(counts), execution_inputs=inputs,
                                     optimizer_wall_seconds=sum(r["optimizer_wall_seconds"] for r in demo[algorithm]["runs"]),
                                     training_evaluation_wall_seconds=sum(r["objective_wall_seconds"] for r in demo[algorithm]["runs"]),
                                     validation_evaluation_wall_seconds=sum(r["validation"]["evaluation_wall_seconds"] for r in demo[algorithm]["runs"]),
                                     evaluation_wall_seconds=sum(r["objective_wall_seconds"] + r["validation"]["evaluation_wall_seconds"]
                                                                 for r in demo[algorithm]["runs"]),
                                     classical_control_seconds=sum(r["classical_control_seconds"] for r in demo[algorithm]["runs"]),
                                     accepted_proposals=sum(r["accepted_proposals"] for r in demo[algorithm]["runs"]),
                                     rejected_proposals=sum(r["rejected_proposals"] for r in demo[algorithm]["runs"]))
    return dict(schema_version=1, resource_kind="full measured-group logical ledger of an ideal finite-shot trajectory",
                algorithms=algorithms, total_shots=demo["total_shots"],
                workflow_wall_seconds=demo["workflow_wall_seconds"],
                physical_resource_estimate=None, whole_trajectory_physical_status="pending",
                preparation="fresh all-zero input assumed per modeled shot; H or RY/CX preparation counted",
                reset_physical_time="unmeasured; no explicit reset instruction",
                compilation="source primitive H/CX/RX/RY/RZ and final measurements; rotations unsynthesized",
                scheduling="per-circuit Qiskit logical depth; no physical routing or device schedule",
                exact_audit="classical post-run references use zero shots and never affect optimizer decisions",
                fixed_readout_projection_scope="separate conditional model for explicitly identified final variants only",
                limitations=demo["limitations"])


def triangle_expectation_reference(gamma: float, beta: float) -> float:
    """Independent scalar p=1 triangle identity, used only after training."""
    return (1.5 + 0.75 * math.sin(4 * beta) * math.sin(2 * gamma)
            - 1.5 * math.sin(2 * beta) ** 2 * math.sin(gamma) ** 2)


def run_portability_demo(*, schedule: Sequence[tuple[float, int, int]] = FINITE_SHOT_SCHEDULE,
                         validation_shots: int = 4096, seeds: Sequence[int] = SAMPLING_SEEDS) -> dict:
    """Second instances with fixed initializations and predeclared seed repeats.

    Same bounded measured schedule as the original cases, and fresh heldout
    families. Stream IDs 2/3 separate these instances from original IDs 0/1.
    Exact references diagnose results afterward; no target stops training.
    """
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("portability demo requires three distinct predeclared seeds")
    started = perf_counter()
    values = maxcut_values(3, TRIANGLE_EDGES)
    a, b, c = PORTABILITY_VQE_COEFFICIENTS
    matrix = vqe_hamiltonian(PORTABILITY_VQE_COEFFICIENTS)
    eigenvalues = np.linalg.eigvalsh(matrix)
    gamma_star = math.atan(1 / math.sqrt(2))
    result = dict(schema_version=1, method="second-instance ideal finite-shot optimization; no QPU execution",
                  exact_diagnostics_used_for_optimization=False,
                  initialization_policy="same predeclared initialization across all seeds within each instance",
                  primary_selection="first predeclared seed before outcomes",
                  physical_resource_estimate=None,
                  rng_contract="SeedSequence([seed,stream_id],spawn_key=(family,evaluation,group)); instance IDs 2/3",
                  validation_precision_target=dict(absolute_error_at_most=0.10, confidence_per_fixed_estimate=0.95),
                  qaoa=dict(instance_id="triangle-maxcut", n_qubits=3, edges=[list(e) for e in TRIANGLE_EDGES],
                            stream_id=2, classical=dict(values=values.tolist(), optimum=int(values.max()),
                                                       optimal_bitstrings=[f"{i:03b}" for i, v in enumerate(values) if v == values.max()]),
                            reference=dict(p1_maximum=2.0, exact_optimal_parameters=[gamma_star, gamma_star / 2],
                                           method="eight classical assignments plus independent p1 scalar identity"),
                            target=dict(exact_gap_from_p1_optimum_at_most=0.10, exact_expected_cut_at_least=1.90), runs=[]),
                  vqe=dict(instance_id="asymmetric-two-qubit", n_qubits=2, coefficients=list(PORTABILITY_VQE_COEFFICIENTS),
                           stream_id=3, hamiltonian="-0.8 Z0-0.6 Z1+0.4 X0X1",
                           reference=dict(matrix=matrix.tolist(), eigenvalues=eigenvalues.tolist(), ground_energy=float(eigenvalues[0]),
                                          ansatz_minimum=-math.sqrt((a + b) ** 2 + c ** 2),
                                          exact_optimal_theta=-math.atan2(c, a + b),
                                          method="explicit full4x4 diagonalization and independent scalar energy"),
                           target=dict(exact_ground_energy_gap_at_most=0.03), runs=[]))

    def q_objective(parameters, **kwargs):
        return sampled_qaoa_objective(parameters, n_qubits=3, edges=TRIANGLE_EDGES, stream_id=2, **kwargs)

    def v_objective(parameters, **kwargs):
        return sampled_vqe_objective(parameters, coefficients=PORTABILITY_VQE_COEFFICIENTS, stream_id=3, **kwargs)

    for algorithm, objective, initial in (("qaoa", q_objective, (0.2, 0.1)), ("vqe", v_objective, (0.6,))):
        for run_index, seed in enumerate(seeds):
            run = sampled_pattern_search(objective, initial, seed=seed, schedule=schedule)
            validation = objective(run["parameters"], shots=validation_shots, seed=seed,
                                   evaluation_index=0, family=1)
            # Audit only after all measured optimizer decisions have finished.
            exact = (lambda p: -triangle_expectation_reference(p[0], p[1])) if algorithm == "qaoa" else (
                lambda p: -(a + b) * math.cos(p[0]) + c * math.sin(p[0]))
            for call in run["evaluations"]:
                call["exact_objective_audit"] = exact(call["parameters"])
            for iteration in run["iterations"]:
                iteration["exact_objective_after_audit"] = exact(iteration["parameters_after"])
            exact_final = exact(run["parameters"])
            radius = math.sqrt(sum((g["outcome_max"] - g["outcome_min"]) ** 2 / g["shots"]
                                   for g in validation["groups"]) * math.log(40) / 2)
            validation.update(role="independent_final_validation", exact_objective_audit=exact_final,
                              estimation_error_from_exact=validation["objective_estimate"] - exact_final,
                              hoeffding_95_absolute_mean_radius=radius,
                              precision_target_supported_by_bound=radius <= 0.10,
                              confidence_scope="one fixed expectation conditional on training-selected parameters")
            if algorithm == "qaoa":
                mean = -exact_final
                probabilities = Statevector.from_instruction(build_qaoa_circuit(
                    3, TRIANGLE_EDGES, [run["parameters"][0]], [run["parameters"][1]])).probabilities()
                run.update(exact_expected_cut_audit=mean, exact_gap_audit=2 - mean,
                           approximation_ratio_audit=mean / 2,
                           optimal_cut_probability_audit=float(probabilities[values == 2].sum()),
                           target_met_audit=mean >= 1.90)
                validation["best_observed_cut"] = max(int(values[int(bits, 2)]) for bits in validation["groups"][0]["counts"])
                validation["optimal_cut_sample_fraction"] = sum(
                    count for bits, count in validation["groups"][0]["counts"].items() if values[int(bits, 2)] == 2) / validation_shots
            else:
                gap = exact_final - float(eigenvalues[0])
                run.update(exact_energy_audit=exact_final, ground_energy_gap_audit=gap,
                           exact_gap_audit=gap, target_met_audit=gap <= 0.03)
            run.update(run_index=run_index, validation=validation,
                       total_shots_including_validation=run["total_shots"] + validation["total_shots"],
                       exact_audit_source="second-instance scalar identity computed after sampled search")
            result[algorithm]["runs"].append(run)
    result["training_shots"] = sum(r["total_shots"] for name in ("qaoa", "vqe") for r in result[name]["runs"])
    result["validation_shots"] = sum(r["validation"]["total_shots"] for name in ("qaoa", "vqe") for r in result[name]["runs"])
    result["total_shots"] = result["training_shots"] + result["validation_shots"]
    result["workflow_wall_seconds"] = perf_counter() - started
    result["limitations"] = ["two tiny additional instances, not general convergence or advantage evidence",
                             "fixed initialization seed repeats measure conditional sampling variability only",
                             "exact targets evaluated after training; holdout precision is a separate task",
                             "ideal independent circuit sampling; no device noise, mitigation or QPU execution",
                             "full physical layer for these second instances is pending",
                             "even-parity VQE ansatz is not expressive for every coefficient choice"]
    return result
