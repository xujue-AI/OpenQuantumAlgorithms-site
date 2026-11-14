# SPDX-License-Identifier: MIT
"""Transparent bounded variational, digitized-annealing and walk workflows.

Circuits use NumPy and Qiskit core; bounded annealing references additionally
use SciPy. Exact reference objectives consume zero shots. Separate finite-shot
objectives and optimizer trajectories use seeded
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


# Fourth increment: digitized annealing. These circuit generators are distinct
# from the bounded dense reference and ideal simulation routines below.
ANNEALING_MAIN_TIME = 16.0
ANNEALING_MAIN_STEPS = 256
ANNEALING_TIME_STUDY = (1.0, 2.0, 4.0, 8.0, 16.0)
ANNEALING_STEP_STUDY = (8, 16, 32, 64, 128)


def _annealing_parameters(total_time: float, steps: int) -> tuple[float, int]:
    time = _angles((total_time,), "total_time")[0]
    if time <= 0:
        raise ValueError("total_time must be finite and positive")
    layers = _positive_int(steps, "steps")
    if time > 64 or layers > 4096:
        raise ValueError("annealing generation supports total_time<=64 and steps<=4096")
    return time, layers


def build_annealing_circuit(n_qubits: int, edges: Iterable[tuple[int, int]],
                             total_time: float, steps: int, *,
                             measure: bool = False) -> QuantumCircuit:
    """Midpoint Strang evolution of H(s)=-(1-s)sum X+s/2 sum ZZ.

    Prepare |+> with n H gates. At s=(j+1/2)/L, each chronological
    driver half-step is RX(-dt*(1-s)) on every wire and each problem
    full-step is CX; RZ(dt*s); CX on every edge. H_P=m/2-C is the
    explicitly chosen Hamiltonian; no identity contribution is omitted.
    Counts before measurement: n H, 2nL RX, mL RZ, 2mL CX.
    Circuit materialization is capped at 100000 unitary gates; this does
    not allocate a statevector or assert an adiabatic runtime guarantee.
    """
    graph = _graph(n_qubits, edges)
    time, layers = _annealing_parameters(total_time, steps)
    if n_qubits > 32:
        raise ValueError("annealing circuit materialization supports at most 32 qubits")
    if n_qubits + layers * (2 * n_qubits + 3 * len(graph)) > 100_000:
        raise ValueError("circuit exceeds the 100000 unitary-gate materialization limit")
    circuit = QuantumCircuit(n_qubits, name="digitized linear annealing")
    circuit.h(range(n_qubits))
    dt = time / layers
    for j in range(layers):
        s = (j + 0.5) / layers
        circuit.rx(-dt * (1 - s), range(n_qubits))
        for u, v in graph:
            circuit.cx(u, v)
            circuit.rz(dt * s, v)
            circuit.cx(u, v)
        circuit.rx(-dt * (1 - s), range(n_qubits))
    if measure:
        circuit.measure_all()
    return circuit


def annealing_state(n_qubits: int, edges: Iterable[tuple[int, int]],
                     total_time: float, steps: int, *, max_qubits: int = 12) -> np.ndarray:
    """Actual ideal circuit state; configured limit and hard cap 16 qubits."""
    _positive_int(n_qubits, "n_qubits")
    _positive_int(max_qubits, "max_qubits")
    if n_qubits > min(max_qubits, 16):
        raise ValueError("annealing statevector exceeds its configured limit (hard cap 16)")
    if float(total_time) > 32:
        raise ValueError("annealing simulation supports total_time<=32")
    return Statevector.from_instruction(
        build_annealing_circuit(n_qubits, edges, total_time, steps)).data


def annealing_hamiltonians(n_qubits: int, edges: Iterable[tuple[int, int]], *,
                            max_qubits: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Independent dense H_D and H_P from bit flips and classical cut values.

    No circuit matrix is consulted. Dense-reference hard cap is 8 qubits,
    separately bounded before allocating either 2^n by 2^n matrix.
    """
    _positive_int(n_qubits, "n_qubits")
    _positive_int(max_qubits, "max_qubits")
    if n_qubits > min(max_qubits, 8):
        raise ValueError("dense annealing reference exceeds its configured limit (hard cap 8)")
    graph = _graph(n_qubits, edges)
    size = 1 << n_qubits
    driver = np.zeros((size, size), dtype=float)
    for assignment in range(size):
        for wire in range(n_qubits):
            driver[assignment ^ (1 << wire), assignment] = -1.0
    cuts = maxcut_values(n_qubits, graph, max_qubits=max_qubits)
    problem = np.diag(len(graph) / 2 - cuts)
    return driver, problem


def annealing_continuous_reference(n_qubits: int, edges: Iterable[tuple[int, int]],
                                    total_time: float, *, max_qubits: int = 8,
                                    rtol: float = 1e-11, atol: float = 1e-13) -> dict:
    """DOP853 numerical solution of i dpsi/dt=H(t/T)psi, independent of gates.

    Tight solver tolerances are settings, not a certified quantum error bound.
    SciPy is imported only for this bounded independent classical reference.
    """
    from scipy.integrate import solve_ivp
    time, _ = _annealing_parameters(total_time, 1)
    if time > 32:
        raise ValueError("continuous annealing reference supports total_time<=32")
    tolerances = _angles((rtol, atol), "solver tolerances")
    if min(tolerances) <= 0:
        raise ValueError("solver tolerances must be positive")
    driver, problem = annealing_hamiltonians(n_qubits, edges, max_qubits=max_qubits)
    initial = np.full(1 << n_qubits, 1 / math.sqrt(1 << n_qubits), dtype=complex)
    def derivative(t, state):
        s = t / time
        return -1j * ((1 - s) * (driver @ state) + s * (problem @ state))
    solution = solve_ivp(derivative, (0.0, time), initial, method="DOP853",
                         rtol=tolerances[0], atol=tolerances[1], t_eval=[time])
    if not solution.success:
        raise RuntimeError(f"continuous annealing reference failed: {solution.message}")
    state = solution.y[:, -1]
    return dict(state=state, solver="scipy.integrate.solve_ivp DOP853",
                solver_success=bool(solution.success), solver_message=solution.message,
                nfev=int(solution.nfev), rtol=tolerances[0], atol=tolerances[1],
                norm_error=abs(float(np.vdot(state, state).real) - 1.0),
                status="independent numerical reference, not a certified error bound")


def annealing_dense_strang_reference(n_qubits: int, edges: Iterable[tuple[int, int]],
                                      total_time: float, steps: int, *,
                                      max_qubits: int = 8) -> np.ndarray:
    """Independent matrix exponentials for the same midpoint Strang formula."""
    from scipy.linalg import expm
    time, layers = _annealing_parameters(total_time, steps)
    if time > 32:
        raise ValueError("dense Strang reference supports total_time<=32")
    driver, problem = annealing_hamiltonians(n_qubits, edges, max_qubits=max_qubits)
    state = np.full(1 << n_qubits, 1 / math.sqrt(1 << n_qubits), dtype=complex)
    dt = time / layers
    for j in range(layers):
        s = (j + 0.5) / layers
        half = expm(-1j * dt * (1 - s) * driver / 2)
        state = half @ (np.exp(-1j * dt * s * np.diag(problem)) * (half @ state))
    return state


def annealing_gap_scan(n_qubits: int, edges: Iterable[tuple[int, int]], *,
                         points: int = 201, max_qubits: int = 8) -> list[dict]:
    """Finite-grid full and G=product X, +1-sector gaps; no gap certificate.

    Every complement pair contributes (|x>+|~x>)/sqrt2 to the invariant
    + sector. The initial |+> state and this interpolation stay in that
    sector. Numerical eigenspectra on a finite grid are diagnostics only.
    """
    _positive_int(points, "points")
    if not 2 <= points <= 4097:
        raise ValueError("gap scan supports 2..4097 grid points")
    driver, problem = annealing_hamiltonians(n_qubits, edges, max_qubits=max_qubits)
    size = 1 << n_qubits
    basis = np.zeros((size, size // 2))
    for x in range(size // 2):
        basis[x, x] = basis[x ^ (size - 1), x] = 1 / math.sqrt(2)
    plus_driver = basis.T @ driver @ basis
    plus_problem = basis.T @ problem @ basis
    rows = []
    for s in np.linspace(0, 1, points):
        full = np.linalg.eigvalsh((1 - s) * driver + s * problem)
        plus = np.linalg.eigvalsh((1 - s) * plus_driver + s * plus_problem)
        # A one-qubit graph is rejected by _graph, so the sector has >=2 states.
        rows.append(dict(s=float(s), full_gap=float(full[1] - full[0]),
                         plus_sector_gap=float(plus[1] - plus[0])))
    return rows


def _annealing_state_diagnostics(state: np.ndarray, values: np.ndarray) -> dict:
    probabilities = np.abs(state) ** 2
    return dict(probabilities=probabilities.tolist(), expected_cut=float(probabilities @ values),
                optimal_probability=float(probabilities[values == values.max()].sum()),
                norm_error=abs(float(probabilities.sum()) - 1.0))


def _state_comparison(state: np.ndarray, reference: np.ndarray) -> dict:
    left = state / np.linalg.norm(state)
    right = reference / np.linalg.norm(reference)
    overlap = np.vdot(right, left)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    return dict(state_infidelity=max(0.0, 1.0 - float(abs(overlap) ** 2)),
                state_phase_aligned_error=float(np.linalg.norm(left - phase * right)))


def run_annealing_demo(*, shots: int = 4096, seeds: Sequence[int] = SAMPLING_SEEDS) -> dict:
    """Predeclared C4 T=16,L=256 demo and bounded convergence diagnostics."""
    _positive_int(shots, "shots")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    started = perf_counter()
    values = maxcut_values(4, CYCLE_EDGES)
    time_rows = []
    continuous_states = {}
    main = None
    for time in ANNEALING_TIME_STUDY:
        state = annealing_state(4, CYCLE_EDGES, time, ANNEALING_MAIN_STEPS)
        continuous = annealing_continuous_reference(4, CYCLE_EDGES, time)
        continuous_states[time] = continuous["state"]
        digital = _annealing_state_diagnostics(state, values)
        reference = _annealing_state_diagnostics(continuous["state"], values)
        comparison = _state_comparison(state, continuous["state"])
        time_rows.append(dict(total_time=time, steps=ANNEALING_MAIN_STEPS,
                              digitized_expected_cut=digital["expected_cut"],
                              digitized_optimal_probability=digital["optimal_probability"],
                              continuous_expected_cut=reference["expected_cut"],
                              continuous_optimal_probability=reference["optimal_probability"],
                              **comparison, solver_nfev=continuous["nfev"],
                              solver_norm_error=continuous["norm_error"]))
        if time == ANNEALING_MAIN_TIME:
            dense = annealing_dense_strang_reference(4, CYCLE_EDGES, time, ANNEALING_MAIN_STEPS)
            main = dict(total_time=time, steps=ANNEALING_MAIN_STEPS, **digital, **comparison,
                        continuous_expected_cut=reference["expected_cut"],
                        continuous_optimal_probability=reference["optimal_probability"],
                        dense_strang_state_error=_state_comparison(state, dense)["state_phase_aligned_error"],
                        continuous_solver={k: v for k, v in continuous.items() if k != "state"})
    step_rows = []
    continuous = continuous_states[4.0]
    continuous_cut = _annealing_state_diagnostics(continuous, values)["expected_cut"]
    for steps in ANNEALING_STEP_STUDY:
        state = annealing_state(4, CYCLE_EDGES, 4.0, steps)
        diagnostics = _annealing_state_diagnostics(state, values)
        step_rows.append(dict(total_time=4.0, steps=steps, **diagnostics,
                              **_state_comparison(state, continuous),
                              cut_mean_absolute_error=abs(diagnostics["expected_cut"] - continuous_cut)))
    readout = []
    for seed in seeds:
        measured = build_annealing_circuit(4, CYCLE_EDGES, ANNEALING_MAIN_TIME,
                                           ANNEALING_MAIN_STEPS, measure=True)
        group = _sample_measured_scalar(measured, values, shots=shots, seed=seed,
                                       algorithm_id=4, family=1, evaluation_index=0, group_index=0)
        optimal_count = sum(n for key, n in group["counts"].items() if values[int(key, 2)] == 4)
        readout.append(dict(seed=seed, **group, cut_mean=group["estimate"],
                            cut_standard_error=group["estimated_standard_error"],
                            optimal_count=optimal_count, optimal_frequency=optimal_count / shots,
                            best_observed_cut=max(int(values[int(key, 2)]) for key in group["counts"])))
    return dict(schema_version=1, algorithm="digitized linear annealing", instance_id="cycle4-maxcut",
                task=dict(n_qubits=4, edges=[list(e) for e in CYCLE_EDGES],
                          driver="-sum_i X_i", problem="0.5 sum_edges Z_u Z_v=2-C",
                          schedule="s=t/T", units="dimensionless Hamiltonian; hbar=1",
                          preparation="H on every wire from zero", output="all-Z bitstring scored by joint cut"),
                reference=dict(method="enumerate all 16 assignments independently; dense ODE and dense Strang",
                               cut_values=values.tolist(), maximum_cut=4, optimal_integers=[5, 10],
                               optimal_bitstrings=["0101", "1010"]),
                target=dict(predeclared_total_time=ANNEALING_MAIN_TIME, predeclared_steps=ANNEALING_MAIN_STEPS,
                            ideal_optimal_probability_at_least=0.95,
                            achieved=main["optimal_probability"] >= 0.95,
                            scope="fixed ideal circuit; no generic adiabatic or hardware guarantee"),
                main=main, time_study=time_rows, step_study=step_rows,
                gap_scan=annealing_gap_scan(4, CYCLE_EDGES), readout=readout,
                total_preparations=shots * len(seeds),
                workflow_wall_seconds=perf_counter() - started,
                timing_scope="CPU circuit simulation, dense numerical references and ideal seeded sampling; not QPU time",
                exclusions=["device noise, physical reset, connectivity routing and control latency",
                            "generic gap or annealing-time theorem; numerical ODE tolerances are not certified bounds",
                            "coherent Clifford+T synthesis and physical faults in this logical ledger",
                            "larger-instance schedule selection and time-to-solution prediction"])


def annealing_run_resources(demo: dict) -> dict:
    """Full fixed main ideal-readout workload, not abstract evolution time T."""
    task, main = demo["task"], demo["main"]
    circuit = compile_rotation_basis(build_annealing_circuit(
        task["n_qubits"], task["edges"], main["total_time"], main["steps"], measure=True))
    counts = {k: int(v) for k, v in circuit.count_ops().items()}
    repetitions = sum(row["shots"] for row in demo["readout"])
    rotations = [float(inst.operation.params[0]) for inst in circuit.data
                 if inst.operation.name in ("rx", "ry", "rz")]
    return dict(schema_version=1, resource_kind="complete fixed main ideal-readout logical workload",
                main_parameters=dict(total_time=main["total_time"], steps=main["steps"]),
                logical_qubits=circuit.num_qubits, per_preparation_gate_counts=counts,
                per_preparation_depth=circuit.depth(), total_preparations=repetitions,
                shot_weighted_gate_counts={k: v * repetitions for k, v in counts.items()},
                measured_qubit_readouts=circuit.num_qubits * repetitions,
                simulated_rotation_angles=rotations, rotation_count=len(rotations),
                compilation=dict(basis=ROTATION_BASIS, optimization_level=0, seed_transpiler=7),
                scaling=dict(h_preparation="n", driver_rx="2*n*L", problem_rz="m*L", problem_cx="2*m*L",
                             measurement="n per preparation", graph_input="explicit unweighted edge list",
                             classical_scoring="O(m) per readout", materialization_limit=100000,
                             materialization_max_qubits=32, materialization_max_steps=4096,
                             materialization_max_total_time=64, simulation_max_total_time=32,
                             statevector_hard_cap_qubits=16, sampling_hard_cap_qubits=12,
                             dense_reference_hard_cap_qubits=8),
                physical_resource_estimate=None,
                status="arbitrary rotations remain explicit; conditional physical projection recorded separately",
                exclusions=demo["exclusions"])
# ---- Explicit coined hypercube search (bounded SKW-style teaching primitive) ----

WALK_DIMENSIONS = (2, 4, 8, 16)
WALK_CLIFFORD_T_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "cx"]


def walk_register_layout(dimension: int) -> dict:
    """Coin bits are low, then position, then clean AND workspace.

    Dimension d describes a hypercube with N=2**d vertices. The binary coin
    selects one of d directions, so this implementation requires d=2**r.
    Materialization and simulation limits are deliberately separate.
    """
    if (isinstance(dimension, bool) or not isinstance(dimension, int)
            or dimension not in WALK_DIMENSIONS):
        raise ValueError("dimension must be one of 2, 4, 8, 16")
    r = dimension.bit_length() - 1
    active = dimension + r
    return dict(dimension=dimension, vertices=1 << dimension, coin_qubits=r,
                position_qubits=dimension, active_qubits=active,
                workspace_qubits=active - 1, total_qubits=2 * active - 1,
                coin_wires=list(range(r)), position_wires=list(range(r, active)),
                workspace_wires=list(range(active, 2 * active - 1)),
                encoding="integer = coin + 2**r * position + 2**(d+r) * workspace")


def _walk_parameters(dimension: int, marked_vertex: int, steps: int = 0) -> dict:
    layout = walk_register_layout(dimension)
    if (isinstance(marked_vertex, bool) or not isinstance(marked_vertex, int)
            or not 0 <= marked_vertex < layout["vertices"]):
        raise ValueError("marked_vertex must be an integer in [0,2**dimension)")
    if (isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 32):
        raise ValueError("steps must be an integer in [0,32]")
    return layout


def _walk_clean_phase(circuit: QuantumCircuit, controls: Sequence[int],
                      workspace: Sequence[int]) -> None:
    """Apply a -1 phase on all-one controls; clean AND compute/Z/uncompute."""
    if len(controls) == 1:
        circuit.z(controls[0])
    elif len(controls) == 2:
        circuit.cz(controls[0], controls[1])
    else:
        if len(workspace) < len(controls) - 1:
            raise ValueError("insufficient clean workspace")
        chain = [(controls[0], controls[1], workspace[0])]
        chain.extend((workspace[i - 2], controls[i], workspace[i - 1])
                     for i in range(2, len(controls)))
        for gate in chain:
            circuit.ccx(*gate)
        circuit.z(workspace[len(controls) - 2])
        for gate in reversed(chain):
            circuit.ccx(*gate)


def build_hypercube_mark_oracle(dimension: int, marked_vertex: int) -> QuantumCircuit:
    """R=I-2|s_coin,m><s_coin,m|, with supplied teaching equality predicate.

    The known target is an input to the gate construction, not hidden data or
    an opaque oracle. On initially-zero work wires, all work is uncomputed.
    """
    layout = _walk_parameters(dimension, marked_vertex)
    coin, position, work = (layout[key] for key in
                            ("coin_wires", "position_wires", "workspace_wires"))
    circuit = QuantumCircuit(layout["total_qubits"], name="joint mark oracle")
    circuit.h(coin)
    wrappers = coin + [wire for bit, wire in enumerate(position)
                       if not ((marked_vertex >> bit) & 1)]
    circuit.x(wrappers)
    _walk_clean_phase(circuit, coin + position, work)
    circuit.x(wrappers)
    circuit.h(coin)
    return circuit


def build_hypercube_coin(dimension: int) -> QuantumCircuit:
    """Coin I-2|s><s| = minus the usual Grover coin, on clean workspace."""
    layout = walk_register_layout(dimension)
    coin, work = layout["coin_wires"], layout["workspace_wires"]
    circuit = QuantumCircuit(layout["total_qubits"], name="minus Grover coin")
    circuit.h(coin)
    circuit.x(coin)
    _walk_clean_phase(circuit, coin, work)
    circuit.x(coin)
    circuit.h(coin)
    return circuit


def build_hypercube_shift(dimension: int) -> QuantumCircuit:
    """S|coin i,position x>=|i,x XOR 2**i>, with explicit matched controls."""
    layout = walk_register_layout(dimension)
    coin, position, work = (layout[key] for key in
                            ("coin_wires", "position_wires", "workspace_wires"))
    r = layout["coin_qubits"]
    circuit = QuantumCircuit(layout["total_qubits"], name="hypercube shift")
    for direction in range(dimension):
        wrappers = [wire for bit, wire in enumerate(coin)
                    if not ((direction >> bit) & 1)]
        if wrappers:
            circuit.x(wrappers)
        if r == 1:
            circuit.cx(coin[0], position[direction])
        elif r == 2:
            circuit.ccx(coin[0], coin[1], position[direction])
        else:
            chain = [(coin[0], coin[1], work[0])]
            chain.extend((work[i - 2], coin[i], work[i - 1])
                         for i in range(2, r - 1))
            for gate in chain:
                circuit.ccx(*gate)
            circuit.ccx(work[r - 3], coin[-1], position[direction])
            for gate in reversed(chain):
                circuit.ccx(*gate)
        if wrappers:
            circuit.x(wrappers)
    return circuit


def build_hypercube_walk_circuit(dimension: int = 4, marked_vertex: int = 5,
                                 steps: int = 4, *, measure: bool = False,
                                 stage_barriers: bool = False) -> QuantumCircuit:
    """Prepare uniform coin+position, repeat joint oracle/coin/shift, read all.

    Each implemented step differs from the standard SKW step by a global -1;
    this circuit is intended for uncontrolled search, not phase-correct
    controlled reuse. The conservative primitive cap charges CCX=15, CZ=3,
    and ordinary Clifford gates=1, before optional final measurements.
    """
    layout = _walk_parameters(dimension, marked_vertex, steps)
    oracle = build_hypercube_mark_oracle(dimension, marked_vertex)
    coin = build_hypercube_coin(dimension)
    shift = build_hypercube_shift(dimension)
    def primitive_cost(c):
        return sum(count * (15 if name == "ccx" else 3 if name == "cz" else 1)
                   for name, count in c.count_ops().items())
    cap_cost = layout["active_qubits"] + steps * sum(
        primitive_cost(c) for c in (oracle, coin, shift))
    if cap_cost > 100_000:
        raise ValueError("walk circuit exceeds the 100000 primitive-gate materialization limit")
    circuit = QuantumCircuit(layout["total_qubits"], name="SKW-style hypercube search")
    circuit.h(range(layout["active_qubits"]))
    for _ in range(steps):
        for operation in (oracle, coin, shift):
            circuit.compose(operation, inplace=True)
            if stage_barriers:
                circuit.barrier()
    if measure:
        circuit.measure_all()
    circuit.metadata = dict(walk=layout, marked_vertex=marked_vertex, steps=steps,
                            oracle_calls=steps, coin_calls=steps, shift_calls=steps,
                            primitive_cap_cost=cap_cost,
                            global_phase_vs_standard_skw="(-1)**steps")
    return circuit


def compile_walk_clifford_t(circuit: QuantumCircuit) -> QuantumCircuit:
    """Exact deterministic CCX/CZ lowering; no arbitrary rotations or synthesis."""
    compiled = transpile(circuit, basis_gates=WALK_CLIFFORD_T_BASIS,
                         optimization_level=0, seed_transpiler=7)
    unsupported = set(compiled.count_ops()) - set(WALK_CLIFFORD_T_BASIS) - {"barrier", "measure"}
    if unsupported:
        raise RuntimeError(f"unsupported walk operations: {unsupported}")
    return compiled


def hypercube_walk_reference(dimension: int = 4, marked_vertex: int = 5,
                             steps: int = 4, *, max_active_qubits: int = 6) -> np.ndarray:
    """Independent dense standard SKW S*C_marked matrix on the active space.

    C is the usual (2/d)J-I except the marked position block is -I. S is an
    independently constructed permutation. Hard cap six active bits prevents
    mistaking this small classical check for a scalable simulator.
    """
    layout = _walk_parameters(dimension, marked_vertex, steps)
    _positive_int(max_active_qubits, "max_active_qubits")
    if layout["active_qubits"] > min(max_active_qubits, 6):
        raise ValueError("dense walk reference exceeds the active-qubit limit (hard cap 6)")
    size = 1 << layout["active_qubits"]
    grover = 2.0 / dimension * np.ones((dimension, dimension)) - np.eye(dimension)
    marked_coin = np.zeros((size, size), dtype=complex)
    shift = np.zeros_like(marked_coin)
    for position in range(layout["vertices"]):
        block = slice(position * dimension, (position + 1) * dimension)
        marked_coin[block, block] = -np.eye(dimension) if position == marked_vertex else grover
        for direction in range(dimension):
            shift[(position ^ (1 << direction)) * dimension + direction,
                  position * dimension + direction] = 1
    step = shift @ marked_coin
    state = np.ones(size, dtype=complex) / math.sqrt(size)
    for _ in range(steps):
        state = step @ state
    return state


def simulate_hypercube_walk(dimension: int = 4, marked_vertex: int = 5,
                            steps: int = 4, *, max_qubits: int = 12) -> dict:
    """Bounded ideal statevector, without postselection on workspace outcomes."""
    layout = _walk_parameters(dimension, marked_vertex, steps)
    _positive_int(max_qubits, "max_qubits")
    if layout["total_qubits"] > min(max_qubits, 12):
        raise ValueError("walk statevector exceeds the total-qubit limit (hard cap 12)")
    begin = perf_counter()
    circuit = build_hypercube_walk_circuit(dimension, marked_vertex, steps)
    state = Statevector.from_instruction(circuit)
    probabilities = state.probabilities()
    active_size = 1 << layout["active_qubits"]
    workspace_leakage = float(np.sum(probabilities[active_size:]))
    # Sum over every measured workspace value: this is not postselection.
    indices = np.arange(len(probabilities))
    positions = (indices >> layout["coin_qubits"]) & (layout["vertices"] - 1)
    position_probabilities = np.bincount(positions, weights=probabilities,
                                         minlength=layout["vertices"])
    reference = hypercube_walk_reference(dimension, marked_vertex, steps)
    clean_state = state.data[:active_size]
    overlap = np.vdot(reference, clean_state)
    global_phase = overlap / abs(overlap) if abs(overlap) > 0 else complex(1)
    return dict(layout=layout, dimension=dimension, marked_vertex=marked_vertex, steps=steps,
                probabilities=probabilities, position_probabilities=position_probabilities,
                marked_probability=float(position_probabilities[marked_vertex]),
                workspace_leakage=workspace_leakage,
                reference_max_amplitude_error=float(np.max(np.abs(clean_state - global_phase * reference))),
                reference_overlap_squared=float(abs(overlap) ** 2),
                observed_global_phase_real=float(global_phase.real),
                observed_global_phase_imag=float(global_phase.imag),
                statevector_seconds=perf_counter() - begin,
                validation="all workspace outcomes retained; zero-work leakage is a check, not postselection")


def sample_hypercube_walk(dimension: int = 4, marked_vertex: int = 5,
                          steps: int = 4, *, shots: int = 4096, seed: int = 7,
                          max_qubits: int = 12) -> dict:
    """Seeded ideal measurement draws for one fixed final state, namespace 5."""
    _positive_int(shots, "shots")
    if shots > 1_000_000:
        raise ValueError("walk sampling is capped at 1000000 shots per call")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    simulation = simulate_hypercube_walk(dimension, marked_vertex, steps, max_qubits=max_qubits)
    layout = simulation["layout"]
    begin = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
    probabilities = np.array(simulation["probabilities"], dtype=float)
    probabilities /= np.sum(probabilities)
    outcomes = rng.multinomial(shots, probabilities)
    positions = (np.arange(len(outcomes)) >> layout["coin_qubits"]) & (layout["vertices"] - 1)
    position_counts = np.bincount(positions, weights=outcomes,
                                  minlength=layout["vertices"]).astype(int)
    marked_count = int(position_counts[marked_vertex])
    leaked_count = int(np.sum(outcomes[1 << layout["active_qubits"]:]))
    return dict(seed=seed, rng_contract="NumPy PCG64; SeedSequence([seed,5])", shots=shots,
                counts={format(i, f"0{layout['total_qubits']}b"): int(value)
                        for i, value in enumerate(outcomes) if value},
                position_counts={format(i, f"0{dimension}b"): int(value)
                                 for i, value in enumerate(position_counts) if value},
                marked_samples=marked_count, marked_sample_fraction=marked_count / shots,
                exact_marked_probability=simulation["marked_probability"],
                workspace_nonzero_samples=leaked_count,
                workspace_leakage=simulation["workspace_leakage"],
                estimated_standard_error=math.sqrt((marked_count / shots) * (1 - marked_count / shots) / shots),
                hoeffding_half_width_95=math.sqrt(math.log(40) / (2 * shots)),
                confidence_scope="one fixed marked-position frequency per seed; no adaptive selection",
                circuit_preparations=shots, oracle_calls=shots * steps,
                coin_calls=shots * steps, shift_calls=shots * steps,
                individual_qubit_measurements=shots * layout["total_qubits"],
                sampling_seconds=perf_counter() - begin,
                statevector_seconds=simulation["statevector_seconds"],
                method="ideal Statevector probabilities followed by multinomial measurement draws; no device noise")


def walk_repetitions(marked_probability: float, target_success: float = 0.99) -> int:
    """Independent fixed-state trials for at least one marked readout.

    A measured position is checked by equality with the supplied marked value;
    no amplitude amplification or coherent inter-trial state is assumed.
    """
    if not (math.isfinite(marked_probability) and 0 < marked_probability <= 1
            and math.isfinite(target_success) and 0 < target_success < 1):
        raise ValueError("require 0<marked_probability<=1 and 0<target_success<1")
    if marked_probability == 1:
        return 1
    return math.ceil(math.log1p(-target_success) / math.log1p(-marked_probability))


# Fifth increment: predetermined finite-shot size studies, separate from the
# historical demonstrations above. Targets are evaluated only after training.
QUALITY_SCALING_QAOA_SIZES = (3, 4, 5, 6, 8)
QUALITY_SCALING_DEPTHS = (1, 2)
QUALITY_SCALING_GRAPH_FAMILIES = ("path", "cycle")
QUALITY_SCALING_VQE_SIZES = (2, 3, 4, 6)
QUALITY_SCALING_SCHEDULE = ((0.4, 256, 4), (0.2, 512, 4), (0.1, 1024, 4))
QUALITY_SCALING_INITIAL_SHOTS = 128
QUALITY_SCALING_VALIDATION_SHOTS = 16384
QUALITY_SCALING_VALIDATION_FAILURE = 0.05
QUALITY_SCALING_MAX_GROUP_BATCHES = 10_000
QUALITY_SCALING_MAX_SHOTS = 10_000_000
QUALITY_SCALING_MAX_WALL_SECONDS = 120.0


def quality_scaling_graph(n_qubits: int, family: str) -> tuple[tuple[int, int], ...]:
    """Explicit path/cycle edge lists, without allocating a simulator.

    This separate family generator supports 3..4096 vertices and the existing
    circuit constructor enforces its own aggregate 100000-unitary-gate cap.
    The empirical study itself only simulates at most eight qubits.
    """
    _positive_int(n_qubits, "n_qubits")
    if not 3 <= n_qubits <= 4096:
        raise ValueError("path/cycle generation supports 3..4096 vertices")
    if family not in QUALITY_SCALING_GRAPH_FAMILIES:
        raise ValueError("family must be 'path' or 'cycle'")
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    if family == "cycle":
        edges.append((0, n_qubits - 1))
    return tuple(edges)


def quality_scaling_cases() -> list[dict]:
    """The 24 predetermined input/ansatz cases; no outcome-dependent selection."""
    cases = []
    for family in QUALITY_SCALING_GRAPH_FAMILIES:
        for n in QUALITY_SCALING_QAOA_SIZES:
            for p in QUALITY_SCALING_DEPTHS:
                cases.append(dict(case_id=f"qaoa-{family}-n{n}-p{p}", algorithm="qaoa",
                                  graph_family=family, n_qubits=n, p=p,
                                  edges=[list(e) for e in quality_scaling_graph(n, family)],
                                  initial=[angle for _ in range(p) for angle in (0.2, 0.1)],
                                  parameter_order="gamma0,beta0,gamma1,beta1,...",
                                  stream_id=6 + len(cases), groups=["all_z"],
                                  target=dict(kind="post_training_ideal_expected_cut_ratio",
                                              ratio_at_least=0.80, selection="predetermined")))
    for n in QUALITY_SCALING_VQE_SIZES:
        cases.append(dict(case_id=f"vqe-open-chain-n{n}", algorithm="vqe",
                          graph_family="open-chain", n_qubits=n, p=1,
                          hamiltonian="-sum_i Z_i+0.5 sum_i X_i X_(i+1)",
                          ansatz="one RY per qubit followed by ascending nearest-neighbor CX chain",
                          initial=[0.6] * n, parameter_order="theta0,theta1,...",
                          stream_id=6 + len(cases), groups=["all_z", "all_x"],
                          target=dict(kind="post_training_ideal_ground_energy_density_gap",
                                      density_gap_at_most=0.05, selection="predetermined")))
    return cases


def quality_scaling_recipe(*, seeds: Sequence[int] = SAMPLING_SEEDS,
                             schedule: Sequence[tuple[float, int, int]] = QUALITY_SCALING_SCHEDULE,
                             validation_shots: int = QUALITY_SCALING_VALIDATION_SHOTS,
                             case_ids: Sequence[str] | None = None) -> dict:
    """Record and preflight the fixed design before any quantum simulation.

    Optional smaller case/seed/schedule allocations support bounded tests.
    They are labeled custom and are not substitutes for the published fixed
    recipe. No target or initialization can be changed by this API.
    """
    seeds = tuple(seeds)
    if (not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds)
            or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32 for x in seeds)):
        raise ValueError("recipe requires one to three distinct nonnegative 32-bit seeds")
    if not 2 <= _positive_int(validation_shots, "validation_shots") <= 65536:
        raise ValueError("validation_shots must lie in 2..65536")
    blocks = []
    sweeps = shot_sweep_sum = 0
    for step, shots, repetitions in schedule:
        step = _angles((step,), "step")[0]
        _positive_int(shots, "shots")
        _positive_int(repetitions, "sweeps")
        if step <= 0 or not 2 <= shots <= 65536 or sweeps + repetitions > 128:
            raise ValueError("invalid bounded sampled schedule")
        blocks.append([step, shots, repetitions])
        sweeps += repetitions
        shot_sweep_sum += shots * repetitions
    if not blocks:
        raise ValueError("recipe schedule cannot be empty")
    cases = quality_scaling_cases()
    if case_ids is not None:
        ids = tuple(case_ids)
        known = {c["case_id"] for c in cases}
        if not ids or len(set(ids)) != len(ids) or set(ids) - known:
            raise ValueError("case_ids must be a nonempty distinct subset of the fixed case IDs")
        cases = [c for c in cases if c["case_id"] in ids]
    training_calls = training_groups = training_shots = holdout_groups = 0
    for case in cases:
        dimension = len(case["initial"])
        group_count = len(case["groups"])
        calls = 1 + (1 + 2 * dimension) * sweeps
        shots = QUALITY_SCALING_INITIAL_SHOTS + (1 + 2 * dimension) * shot_sweep_sum
        case["planned"] = dict(training_calls_per_seed=calls,
                               training_shots_per_seed=shots * group_count,
                               holdout_shots_per_seed=validation_shots * group_count,
                               total_shots_per_seed=(shots + validation_shots) * group_count)
        training_calls += calls * len(seeds)
        training_groups += calls * group_count * len(seeds)
        training_shots += shots * group_count * len(seeds)
        holdout_groups += group_count * len(seeds)
    validation_total = validation_shots * holdout_groups
    total_groups = training_groups + holdout_groups
    if total_groups > QUALITY_SCALING_MAX_GROUP_BATCHES or training_shots + validation_total > QUALITY_SCALING_MAX_SHOTS:
        raise ValueError("recipe exceeds the study's 10000-group or 10000000-shot cap")
    fixed = (seeds == SAMPLING_SEEDS and tuple(tuple(x) for x in blocks) == QUALITY_SCALING_SCHEDULE
             and validation_shots == QUALITY_SCALING_VALIDATION_SHOTS and case_ids is None)
    return dict(schema_version=1, recipe_id="quality-scaling-v1", fixed_published_recipe=fixed,
                design="predetermined before study execution; no quality-target retuning",
                cases=cases, seeds=list(seeds), schedule=blocks,
                initial_shots_per_group=QUALITY_SCALING_INITIAL_SHOTS, comparison_z=2.0,
                validation_shots_per_group=validation_shots,
                validation_failure_all_final_groups=QUALITY_SCALING_VALIDATION_FAILURE,
                validation_failure_per_group=QUALITY_SCALING_VALIDATION_FAILURE / holdout_groups,
                normalized_holdout_precision_target=0.05,
                planned=dict(runs=len(cases) * len(seeds), training_calls=training_calls,
                             validation_calls=len(cases) * len(seeds),
                             objective_calls=training_calls + len(cases) * len(seeds),
                             training_groups=training_groups, validation_groups=holdout_groups,
                             group_batches=total_groups, training_shots=training_shots,
                             validation_shots=validation_total, total_shots=training_shots + validation_total),
                bounds=dict(simulation_max_qubits=8, dense_reference_max_qubits=8,
                            group_batches=QUALITY_SCALING_MAX_GROUP_BATCHES,
                            total_shots=QUALITY_SCALING_MAX_SHOTS,
                            workflow_wall_seconds=QUALITY_SCALING_MAX_WALL_SECONDS,
                            unitary_gates_per_circuit=100000),
                workload_comparison="same per-call schedule/sweeps; p2 and larger VQE parameter vectors consume more calls/shots",
                statistical_scope="union bound for independent final fixed-state groups conditional on training; no optimizer guarantee")


def _layer_parameters(parameters: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    point = _angles(parameters, "parameters")
    if len(point) not in (2, 4):
        raise ValueError("this size study supports p=1 or2 with interleaved gamma,beta")
    return point[0::2], point[1::2]


def sampled_qaoa_layers_objective(parameters: Sequence[float], *, n_qubits: int,
                                    edges: Iterable[tuple[int, int]], shots: int, seed: int,
                                    evaluation_index: int, family: int = 0, stream_id: int = 6) -> dict:
    """Raw sampled joint cut objective at p1/p2; no classical optimum lookup."""
    _positive_int(n_qubits, "n_qubits")
    if n_qubits > 8:
        raise ValueError("quality-study sampling is capped at eight qubits")
    gammas, betas = _layer_parameters(parameters)
    graph = _graph(n_qubits, edges)
    begin = perf_counter()
    measured = build_qaoa_circuit(n_qubits, graph, gammas, betas, measure=True)
    values = maxcut_values(n_qubits, graph, max_qubits=8)
    group = _sample_measured_scalar(measured, values, shots=shots, seed=seed, algorithm_id=stream_id,
                                   family=family, evaluation_index=evaluation_index, group_index=0)
    group.update(group="all_z", observable="joint raw cut C", outcome_min=0, outcome_max=len(graph))
    return dict(parameters=list(parameters), objective_estimate=-group["estimate"],
                estimated_standard_error=group["estimated_standard_error"], groups=[group], total_shots=shots,
                evaluation_wall_seconds=perf_counter() - begin,
                objective_convention="minimize negative sample mean of joint cut; optimum not used in decisions",
                method="ideal finite-shot circuit sampling")


def quality_scaling_group_circuit(case: dict, parameters: Sequence[float], group_name: str) -> QuantumCircuit:
    """Reconstruct an actual measured source circuit for every recorded group."""
    if case["algorithm"] == "qaoa":
        if group_name != "all_z":
            raise ValueError("QAOA has only its joint all-Z group")
        gammas, betas = _layer_parameters(parameters)
        if len(gammas) != case["p"]:
            raise ValueError("recorded QAOA parameter dimension does not match p")
        return build_qaoa_circuit(case["n_qubits"], case["edges"], gammas, betas, measure=True)
    if case["algorithm"] == "vqe" and group_name in ("all_z", "all_x"):
        return build_spin_vqe_circuit(case["n_qubits"], parameters, basis=group_name[-1], measure=True)
    raise ValueError("unknown case algorithm or measurement group")


def qaoa_layers_reference(n_qubits: int, edges: Iterable[tuple[int, int]],
                           parameters: Sequence[float], *, max_qubits: int = 8) -> dict:
    """Independent finite assignment enumeration and pairwise mixer evolution.

    Uses the full e^{-i gamma C} phase including its identity contribution,
    followed by algebraic bit-pair X rotations. No Qiskit circuit matrix or
    statevector implementation is used to construct this reference state.
    """
    _positive_int(n_qubits, "n_qubits")
    _positive_int(max_qubits, "max_qubits")
    if n_qubits > min(max_qubits, 8):
        raise ValueError("quality reference is capped at eight qubits")
    graph = _graph(n_qubits, edges)
    gammas, betas = _layer_parameters(parameters)
    begin = perf_counter()
    values = np.fromiter((sum(((z >> u) ^ (z >> v)) & 1 for u, v in graph)
                          for z in range(1 << n_qubits)), dtype=int)
    enumeration_seconds = perf_counter() - begin
    evolution_begin = perf_counter()
    state = np.full(1 << n_qubits, 1 / math.sqrt(1 << n_qubits), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * gamma * values)
        for q in range(n_qubits):
            mask = 1 << q
            lo = np.array([z for z in range(len(state)) if not (z & mask)], dtype=int)
            hi = lo ^ mask
            a, b = state[lo].copy(), state[hi].copy()
            state[lo] = math.cos(beta) * a - 1j * math.sin(beta) * b
            state[hi] = math.cos(beta) * b - 1j * math.sin(beta) * a
    probabilities = np.abs(state) ** 2
    optimum = int(values.max())
    return dict(reference_state=state, cut_values=values.tolist(), classical_optimum=optimum,
                optimal_bitstrings=[format(z, f"0{n_qubits}b") for z in np.flatnonzero(values == optimum)],
                exact_expected_cut=float(probabilities @ values),
                exact_approximation_ratio=float(probabilities @ values / optimum),
                exact_optimal_cut_probability=float(probabilities[values == optimum].sum()),
                enumeration_seconds=enumeration_seconds, evolution_seconds=perf_counter() - evolution_begin,
                reference_wall_seconds=perf_counter() - begin,
                method="independent all-assignment XOR enumeration and algebraic phase/bit-pair mixer evolution")


def _quality_holdout_bounds(case: dict, validation: dict, reference: dict, recipe: dict) -> dict:
    group_delta = recipe["validation_failure_per_group"]
    global_radii = [float(g["outcome_max"] - g["outcome_min"])
                    * math.sqrt(math.log(2 / group_delta) / (2 * g["shots"])) for g in validation["groups"]]
    per_run_delta = 0.05 / len(validation["groups"])
    per_run_radii = [float(g["outcome_max"] - g["outcome_min"])
                     * math.sqrt(math.log(2 / per_run_delta) / (2 * g["shots"])) for g in validation["groups"]]
    if case["algorithm"] == "qaoa":
        denominator = reference["classical_optimum"]
        value = -validation["objective_estimate"] / denominator
        radius, local_radius = sum(global_radii) / denominator, sum(per_run_radii) / denominator
        threshold = case["target"]["ratio_at_least"]
        status = "supported" if value - radius >= threshold else "missed" if value + radius < threshold else "inconclusive"
        standard_error = validation["estimated_standard_error"] / denominator
    else:
        denominator = case["n_qubits"]
        value = (validation["objective_estimate"] - reference["ground_energy"]) / denominator
        radius, local_radius = sum(global_radii) / denominator, sum(per_run_radii) / denominator
        threshold = case["target"]["density_gap_at_most"]
        status = "supported" if value + radius <= threshold else "missed" if value - radius > threshold else "inconclusive"
        standard_error = validation["estimated_standard_error"] / denominator
    return dict(normalized_estimate=value, normalized_standard_error=standard_error,
                simultaneous_radius=radius, per_run_95_radius=local_radius,
                simultaneous_interval=[value - radius, value + radius],
                group_failure_allocation=group_delta, target_status=status,
                precision_target=recipe["normalized_holdout_precision_target"],
                precision_supported=radius <= recipe["normalized_holdout_precision_target"],
                scope="fixed final states, conditional on training; simultaneous ideal-sampling union bound over every final group",
                excludes="optimizer convergence, physical faults and coherent synthesis approximation")


def run_quality_scaling_demo(*, seeds: Sequence[int] = SAMPLING_SEEDS,
                              schedule: Sequence[tuple[float, int, int]] = QUALITY_SCALING_SCHEDULE,
                              validation_shots: int = QUALITY_SCALING_VALIDATION_SHOTS,
                              case_ids: Sequence[str] | None = None,
                              max_wall_seconds: float = QUALITY_SCALING_MAX_WALL_SECONDS) -> dict:
    """Run every fixed recipe, then independently audit achieved final quality.

    Each optimizer sees only fresh sampled raw objectives. All exact/dense
    answers, quality-target flags and confidence statements follow training.
    Missed targets are retained; they are never stopping or retuning signals.
    """
    from time import process_time
    recipe = quality_scaling_recipe(seeds=seeds, schedule=schedule,
                                    validation_shots=validation_shots, case_ids=case_ids)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= QUALITY_SCALING_MAX_WALL_SECONDS:
        raise ValueError("workflow wall cap must lie in (0,120] seconds")
    begin, cpu_begin = perf_counter(), process_time()
    cases, rows = [], []
    for original in recipe["cases"]:
        case = {**original, "runs": []}
        def objective(parameters, **kwargs):
            if perf_counter() - begin > max_wall_seconds:
                raise RuntimeError("quality study exceeded its bounded CPU workflow wall allowance")
            if case["algorithm"] == "qaoa":
                return sampled_qaoa_layers_objective(parameters, n_qubits=case["n_qubits"],
                                                     edges=case["edges"], stream_id=case["stream_id"], **kwargs)
            return sampled_spin_vqe_objective(parameters, n_qubits=case["n_qubits"],
                                              stream_id=case["stream_id"], **kwargs)
        for run_index, seed in enumerate(recipe["seeds"]):
            if perf_counter() - begin > max_wall_seconds:
                raise RuntimeError("quality study exceeded its bounded CPU workflow wall allowance")
            run = sampled_pattern_search(objective, case["initial"], seed=seed,
                                         schedule=recipe["schedule"], initial_shots=QUALITY_SCALING_INITIAL_SHOTS,
                                         comparison_z=recipe["comparison_z"])
            validation = objective(run["parameters"], shots=validation_shots, seed=seed, evaluation_index=0, family=1)
            validation.update(role="independent_final_validation", confidence_scope="conditional on fixed final parameters")
            reference = (qaoa_layers_reference(case["n_qubits"], case["edges"], run["parameters"])
                         if case["algorithm"] == "qaoa" else spin_vqe_reference(case["n_qubits"], run["parameters"]))
            reference_state = reference.pop("reference_state")
            audit_begin = perf_counter()
            prepared = quality_scaling_group_circuit(case, run["parameters"], "all_z").remove_final_measurements(inplace=False)
            circuit_state = Statevector.from_instruction(prepared).data
            overlap = np.vdot(reference_state, circuit_state)
            phase = overlap / abs(overlap) if abs(overlap) else 1.0
            amplitude_error = float(np.max(np.abs(circuit_state - phase * reference_state)))
            if amplitude_error > 1e-11:
                raise RuntimeError("final source circuit disagrees with its independent bounded reference")
            reference.update(source_state_max_phase_aligned_error=amplitude_error,
                             source_state_audit_wall_seconds=perf_counter() - audit_begin,
                             audit_performed_after_training=True)
            if case["algorithm"] == "qaoa":
                target_met = reference["exact_approximation_ratio"] >= case["target"]["ratio_at_least"]
                values = np.array(reference["cut_values"])
                group = validation["groups"][0]
                reference.update(zero_assignment_cut=0,
                                 uniform_assignment_mean_cut=len(case["edges"]) / 2)
                validation.update(best_observed_cut=int(max(values[int(bits, 2)] for bits in group["counts"])),
                                  optimal_cut_count=sum(number for bits, number in group["counts"].items()
                                                        if values[int(bits, 2)] == reference["classical_optimum"]))
            else:
                target_met = reference["energy_density_gap"] <= case["target"]["density_gap_at_most"]
                reference.update(zero_state_energy=-case["n_qubits"],
                                 zero_state_density_gap=(-case["n_qubits"] - reference["ground_energy"]) / case["n_qubits"])
            validation["quality_bound"] = _quality_holdout_bounds(case, validation, reference, recipe)
            run.update(run_index=run_index, validation=validation, diagnostic=reference,
                       target_met_audit=bool(target_met),
                       total_shots_including_validation=run["total_shots"] + validation["total_shots"],
                       exact_diagnostics_used_for_optimization=False)
            if run["total_shots_including_validation"] != case["planned"]["total_shots_per_seed"]:
                raise RuntimeError("recorded shot use disagrees with the predeclared recipe")
            case["runs"].append(run)
            rows.append(dict(case_id=case["case_id"], algorithm=case["algorithm"], graph_family=case["graph_family"],
                             n_qubits=case["n_qubits"], p=case["p"], seed=seed,
                             ideal_quality=(reference["exact_approximation_ratio"] if case["algorithm"] == "qaoa"
                                            else reference["energy_density_gap"]),
                             ideal_target_met=bool(target_met),
                             heldout_estimate=validation["quality_bound"]["normalized_estimate"],
                             heldout_simultaneous_radius=validation["quality_bound"]["simultaneous_radius"],
                             heldout_target_status=validation["quality_bound"]["target_status"],
                             precision_supported=validation["quality_bound"]["precision_supported"],
                             training_calls=run["objective_evaluations"], training_shots=run["total_shots"],
                             validation_shots=validation["total_shots"], total_shots=run["total_shots_including_validation"],
                             optimizer_wall_seconds=run["optimizer_wall_seconds"],
                             objective_wall_seconds=run["objective_wall_seconds"],
                             classical_control_seconds=run["classical_control_seconds"],
                             independent_reference_wall_seconds=reference["reference_wall_seconds"],
                             source_state_audit_wall_seconds=reference["source_state_audit_wall_seconds"]))
        cases.append(case)
    wall = perf_counter() - begin
    if wall > max_wall_seconds:
        raise RuntimeError("quality study exceeded its bounded CPU workflow wall allowance")
    return dict(schema_version=1, recipe=recipe, cases=cases, summary_rows=rows,
                exact_diagnostics_used_for_optimization=False,
                training_shots=sum(r["training_shots"] for r in rows),
                validation_shots=sum(r["validation_shots"] for r in rows),
                total_shots=sum(r["total_shots"] for r in rows),
                workflow_wall_seconds=wall, workflow_process_cpu_seconds=process_time() - cpu_begin,
                hardware_feedback_seconds=None,
                timing_scope="local CPU sampling, controller, heldout validation and independent classical audits; not quantum hardware time",
                quality_selection="all predetermined inputs/seeds including misses; no best-seed or best-size selection",
                physical_resource_estimate=None,
                limitations=["small deterministic ideal CPU study; no hardware noise or advantage claim",
                             "descriptive fixed-budget polling margin, not a convergence theorem",
                             "post-training exact diagnostics do not drive optimizer decisions",
                             "simultaneous holdout coverage excludes adaptive training queries",
                             "VQE restricted ansatz may not contain the ground state; no global ansatz optimum certified",
                             "physical initialization/reset, routing and feedback timing unmeasured",
                             "conditional synthesis and fault models require a separate complete recorded-workload report",
                             "existing Lean proof covers classical C4 only, not these families or quantum dynamics"])


def quality_scaling_resources(demo: dict) -> dict:
    """Complete source-operation ledger for all accepted/rejected/heldout shots."""
    totals, per_run = Counter(), []
    training_groups = validation_groups = calls = measured_bits = 0
    actual_training_shots = actual_validation_shots = 0
    for case in demo["cases"]:
        for run_index, run in enumerate(case["runs"]):
            if run["run_index"] != run_index or run["seed"] != demo["recipe"]["seeds"][run_index]:
                raise ValueError("recorded run indexes or seeds differ from the recipe")
            weighted = Counter()
            groups = 0
            expected_training_allocations = [demo["recipe"]["initial_shots_per_group"]]
            dimension = len(case["initial"])
            for _, shots, sweeps in demo["recipe"]["schedule"]:
                expected_training_allocations.extend([shots] * ((1 + 2 * dimension) * sweeps))
            if len(run["evaluations"]) != len(expected_training_allocations):
                raise ValueError("recorded training calls differ from the prescribed schedule")
            run_training_shots = run_validation_shots = 0
            for phase, records in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for call_index, record in enumerate(records):
                    calls += 1
                    if phase == "training" and record["index"] != call_index:
                        raise ValueError("recorded training call indexes are inconsistent")
                    if phase == "validation" and record["parameters"] != run["parameters"]:
                        raise ValueError("holdout parameters differ from the final fitted state")
                    if [g["group"] for g in record["groups"]] != case["groups"]:
                        raise ValueError("recorded measurement groups differ from the recipe")
                    if record["total_shots"] != sum(g["shots"] for g in record["groups"]):
                        raise ValueError("recorded group shot totals are inconsistent")
                    for group_index, group in enumerate(record["groups"]):
                        expected_shots = (expected_training_allocations[call_index] if phase == "training"
                                          else demo["recipe"]["validation_shots_per_group"])
                        if group["shots"] != expected_shots:
                            raise ValueError("group shot allocation differs from the prescribed recipe")
                        expected_stream = [run["seed"], case["stream_id"], 0 if phase == "training" else 1,
                                           call_index, group_index]
                        if (group["rng"]["stream_key"] != expected_stream
                                or group["rng"]["entropy"] != expected_stream[:2]
                                or group["rng"]["spawn_key"] != expected_stream[2:]):
                            raise ValueError("group RNG lineage differs from the recipe")
                        if sum(group["counts"].values()) != group["shots"]:
                            raise ValueError("group counts omit recorded shots")
                        if any(len(bits) != case["n_qubits"] or set(bits) - {"0", "1"}
                               or isinstance(number, bool) or not isinstance(number, int) or number <= 0
                               for bits, number in group["counts"].items()):
                            raise ValueError("recorded bitstring counts are invalid")
                        source = quality_scaling_group_circuit(case, record["parameters"], group["group"])
                        counts = {k: int(v) for k, v in source.count_ops().items()}
                        if (counts != group["circuit_gate_counts"] or source.num_qubits != group["logical_qubits"]
                                or source.depth() != group["circuit_depth"]
                                or group["measured_qubit_readouts"] != source.num_qubits * group["shots"]):
                            raise ValueError("source group metadata differs from actual reconstruction")
                        weighted.update({gate: number * group["shots"] for gate, number in counts.items() if gate != "barrier"})
                        measured_bits += source.num_qubits * group["shots"]
                        groups += 1
                        if phase == "training":
                            training_groups += 1
                            run_training_shots += group["shots"]
                        else:
                            validation_groups += 1
                            run_validation_shots += group["shots"]
            if (run_training_shots != run["total_shots"]
                    or run_training_shots + run_validation_shots != run["total_shots_including_validation"]
                    or run_training_shots + run_validation_shots != case["planned"]["total_shots_per_seed"]):
                raise ValueError("actual group shots differ from declared run totals")
            actual_training_shots += run_training_shots
            actual_validation_shots += run_validation_shots
            totals.update(weighted)
            per_run.append(dict(case_id=case["case_id"], seed=run["seed"], n_qubits=case["n_qubits"],
                                p=case["p"], algorithm=case["algorithm"], group_batches=groups,
                                total_shots=run["total_shots_including_validation"], weighted_gate_counts=dict(weighted),
                                optimizer_wall_seconds=run["optimizer_wall_seconds"],
                                classical_control_seconds=run["classical_control_seconds"],
                                hardware_feedback_seconds=None))
    planned = demo["recipe"]["planned"]
    if (calls != planned["objective_calls"] or training_groups + validation_groups != planned["group_batches"]
            or demo["total_shots"] != planned["total_shots"]
            or actual_training_shots != demo["training_shots"]
            or actual_validation_shots != demo["validation_shots"]
            or actual_training_shots + actual_validation_shots != demo["total_shots"]):
        raise ValueError("complete ledger does not match the predeclared workload")
    return dict(schema_version=1, kind="complete recorded source logical workload",
                recipe_id=demo["recipe"]["recipe_id"], fixed_published_recipe=demo["recipe"]["fixed_published_recipe"],
                per_run=per_run, objective_calls=calls, training_groups=training_groups,
                validation_groups=validation_groups, group_batches=training_groups + validation_groups,
                total_preparations=demo["total_shots"], measured_qubit_readouts=measured_bits,
                weighted_gate_counts=dict(totals),
                logical_peak_qubits=max(c["n_qubits"] for c in demo["cases"]),
                hardware_feedback_seconds=None, physical_resource_estimate=None,
                scaling=dict(qaoa_source="H=n,RX=np,RZ=mp,CX=2mp,measure=n per group",
                             vqe_source="RY=n,CX=n-1,basis H=n for all-X only,measure=n per group",
                             training_calls="1+(1+2d)*S for d parameters and S prescribed sweeps",
                             total_shots="sum each recorded group allocation including initial, rejected candidates, and final validation",
                             score_cost="O(m) QAOA or O(n) joint VQE score per readout",
                             classical_statevector_memory="O(2^n)", dense_reference_memory="O(4^n)"),
                exclusions=demo["limitations"])


# Fifth increment: a specified open-chain multi-spin VQE family. Construction
# and the bounded classical references below have separate resource limits.
def _spin_vqe_width(n_qubits: int, *, limit: int = 32) -> int:
    width = _positive_int(n_qubits, "n_qubits")
    if not 2 <= width <= limit:
        raise ValueError(f"spin VQE supports 2..{limit} qubits in this operation")
    return width


def _spin_vqe_parameters(n_qubits: int, parameters: Sequence[float]) -> tuple[float, ...]:
    _spin_vqe_width(n_qubits)
    # Check the declared parameter count before materializing an angle tuple.
    if len(parameters) != n_qubits:
        raise ValueError("spin VQE requires one finite RY angle per qubit")
    return _angles(parameters, "parameters")


def _spin_vqe_dense_bound(n_qubits: int, max_qubits: int) -> None:
    _positive_int(max_qubits, "max_qubits")
    _spin_vqe_width(n_qubits, limit=min(max_qubits, 8))


def build_spin_vqe_circuit(n_qubits: int, parameters: Sequence[float], *,
                           basis: str | None = None,
                           measure: bool = False) -> QuantumCircuit:
    """Explicit RY product state followed by an ascending nearest-neighbor CX chain.

    Each wire starts in zero, receives RY(theta_i), and then CX(i,i+1)
    is applied for i=0..n-2. All-X measurement adds H to every qubit;
    all-Z measurement uses the computational basis directly. This is one
    specified n-parameter real ansatz, not a general ground-state-preparation
    circuit. The source contains n RY and n-1 CX, plus n H for all-X and
    n measurements when requested. Physical zero initialization/reset time
    is excluded from source gate counts and must be supplied separately.
    Generation supports 2..32 qubits and at most 100000 unitary gates,
    independently of the eight-qubit sampling/dense-reference limits.
    """
    width = _spin_vqe_width(n_qubits)
    point = _spin_vqe_parameters(width, parameters)
    if basis not in (None, "z", "x"):
        raise ValueError("basis must be None, 'z', or 'x'")
    unitary_gates = width + (width - 1) + (width if basis == "x" else 0)
    if unitary_gates > 100_000:
        raise ValueError("spin VQE circuit exceeds the 100000 unitary-gate materialization limit")
    circuit = QuantumCircuit(width, name="one-layer open-chain spin VQE")
    for wire, theta in enumerate(point):
        circuit.ry(theta, wire)
    for wire in range(width - 1):
        circuit.cx(wire, wire + 1)
    if basis == "x":
        circuit.h(range(width))
    if measure:
        circuit.measure_all()
    return circuit


def spin_vqe_hamiltonian(n_qubits: int, *, max_qubits: int = 8) -> np.ndarray:
    """Independent dense H=-sum_i Z_i+0.5 sum_i X_i X_(i+1), open chain.

    Basis integer bit i is qubit i. Pauli tensor factors are constructed in
    descending wire order, without obtaining a matrix from the circuit.
    The configured dense limit has a hard cap of eight qubits and is checked
    before allocating any 2**n by 2**n matrix. Coefficients are dimensionless.
    """
    _spin_vqe_dense_bound(n_qubits, max_qubits)
    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    size = 1 << n_qubits
    matrix = np.zeros((size, size))
    def product(terms):
        result = np.ones((1, 1))
        for wire in reversed(range(n_qubits)):
            result = np.kron(result, terms.get(wire, identity))
        return result
    for wire in range(n_qubits):
        matrix -= product({wire: z})
    for wire in range(n_qubits - 1):
        matrix += 0.5 * product({wire: x, wire + 1: x})
    return matrix


def spin_vqe_energy_formula(n_qubits: int, parameters: Sequence[float]) -> float:
    """Independent scalar expectation for this particular RY/CX-chain ansatz.

    Pulling the Hamiltonian through the ascending CNOT chain gives
    E=-sum_j product_(k<=j) cos(theta_k)
      +0.5*(sum_(j=0..n-3) sin(theta_j)sin(theta_(j+2)) +sin(theta_(n-2))).
    This ansatz identity does not calculate a Hamiltonian ground energy or
    certify expressibility, parameter-search convergence, or larger-system
    VQE accuracy. Its computation does not allocate an exponential state.
    """
    point = _spin_vqe_parameters(n_qubits, parameters)
    prefix, z_sum = 1.0, 0.0
    for theta in point:
        prefix *= math.cos(theta)
        z_sum += prefix
    x_sum = math.sin(point[n_qubits - 2])
    x_sum += sum(math.sin(point[j]) * math.sin(point[j + 2])
                 for j in range(n_qubits - 2))
    return -z_sum + 0.5 * x_sum


def sampled_spin_vqe_objective(parameters: Sequence[float], *, n_qubits: int,
                               shots: int, seed: int, evaluation_index: int,
                               family: int = 0, stream_id: int) -> dict:
    """Sample two joint-score measurement groups; no exact reference is consulted.

    All-Z scores -sum Z_i jointly, including all term covariance. All-X
    applies H on all wires and scores 0.5 sum adjacent X_i X_(i+1) jointly.
    Separate fresh preparations are charged to both groups. The stream ID
    is supplied by the frozen study recipe, while family=0/1 separates
    training from held-out validation. Ideal circuit probabilities followed
    by seeded multinomial draws are CPU simulation, not QPU execution.
    """
    _spin_vqe_width(n_qubits, limit=8)
    point = _spin_vqe_parameters(n_qubits, parameters)
    started = perf_counter()
    integers = np.arange(1 << n_qubits)
    signs = np.array([1.0 - 2.0 * ((integers >> wire) & 1)
                      for wire in range(n_qubits)])
    z_values = -np.sum(signs, axis=0)
    x_values = 0.5 * np.sum(signs[:-1] * signs[1:], axis=0)
    groups = []
    for group_index, (basis, values, observable, interval) in enumerate((
            ("z", z_values, "joint -sum_i Z_i", (-float(n_qubits), float(n_qubits))),
            ("x", x_values, "joint 0.5 sum_i X_i X_(i+1)",
             (-0.5 * (n_qubits - 1), 0.5 * (n_qubits - 1))))):
        measured = build_spin_vqe_circuit(n_qubits, point, basis=basis, measure=True)
        group = _sample_measured_scalar(
            measured, values, shots=shots, seed=seed, algorithm_id=stream_id,
            family=family, evaluation_index=evaluation_index, group_index=group_index)
        group.update(group=f"all_{basis}", observable=observable,
                     outcome_min=interval[0], outcome_max=interval[1],
                     joint_term_covariance_retained=True)
        groups.append(group)
    return dict(parameters=list(point), n_qubits=n_qubits,
                objective_estimate=sum(group["estimate"] for group in groups),
                estimated_standard_error=math.sqrt(sum(group["sample_variance"] / group["shots"]
                                                       for group in groups)),
                groups=groups, total_shots=2 * shots,
                evaluation_wall_seconds=perf_counter() - started,
                method="ideal finite-shot spin VQE objective; no QPU execution",
                objective_convention="minimize sum of joint all-Z and all-X sample means",
                exact_reference_used_for_objective=False)


def spin_vqe_reference(n_qubits: int, parameters: Sequence[float], *,
                       max_qubits: int = 8) -> dict:
    """Bounded post-run dense/scalar reference, independent of the circuit API.

    The state is constructed from product RY amplitudes and an integer CNOT
    permutation, rather than Qiskit Statevector. Dense diagonalization supplies
    the independent Hamiltonian ground energy. These are numerical references,
    not formal quantum proofs. Raw reference_state is an in-memory complex
    array; omit it or encode it deliberately before JSON publication.
    """
    _spin_vqe_dense_bound(n_qubits, max_qubits)
    point = _spin_vqe_parameters(n_qubits, parameters)
    started = perf_counter()
    stage = perf_counter()
    matrix = spin_vqe_hamiltonian(n_qubits, max_qubits=max_qubits)
    hamiltonian_seconds = perf_counter() - stage
    stage = perf_counter()
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigensolver_seconds = perf_counter() - stage
    stage = perf_counter()
    product_state = np.ones(1, dtype=complex)
    for wire in reversed(range(n_qubits)):
        product_state = np.kron(product_state,
                                [math.cos(point[wire] / 2), math.sin(point[wire] / 2)])
    state = np.zeros(1 << n_qubits, dtype=complex)
    for assignment, amplitude in enumerate(product_state):
        output = assignment
        for wire in range(n_qubits - 1):
            if (output >> wire) & 1:
                output ^= 1 << (wire + 1)
        state[output] = amplitude
    state_preparation_seconds = perf_counter() - stage
    stage = perf_counter()
    energy = float(np.vdot(state, matrix @ state).real)
    scalar_energy = spin_vqe_energy_formula(n_qubits, point)
    scalar_energy_error = abs(energy - scalar_energy)
    if scalar_energy_error > 1e-10:
        raise RuntimeError("spin VQE dense and independent scalar references disagree")
    expectation_seconds = perf_counter() - stage
    return dict(n_qubits=n_qubits,
                hamiltonian="-sum_i Z_i+0.5 sum_i X_i X_(i+1); open chain",
                ground_energy=float(eigenvalues[0]), exact_energy=energy,
                energy_density_gap=(energy - float(eigenvalues[0])) / n_qubits,
                scalar_energy=scalar_energy, scalar_energy_error=scalar_energy_error,
                eigenvalues=eigenvalues.tolist(), reference_state=state,
                norm_error=abs(float(np.vdot(state, state).real) - 1.0),
                hamiltonian_seconds=hamiltonian_seconds,
                eigensolver_seconds=eigensolver_seconds,
                state_preparation_seconds=state_preparation_seconds,
                expectation_seconds=expectation_seconds,
                reference_wall_seconds=perf_counter() - started,
                method="independent Pauli-tensor Hamiltonian/eigvalsh and RY-product/CNOT-permutation state",
                timing_scope="bounded classical reference CPU wall time; no hardware preparation or feedback",
                accuracy_scope="numerical dense/scalar reference; no formal quantum/optimizer guarantee")


# Sixth increment: frozen matched-training-preparation diagnostics. Earlier
# studies and their archived producers are deliberately left unchanged.
DIAGNOSTIC_RECIPE_ID = "optimization-matched-preparations-v1"
DIAGNOSTIC_CONTROL_MODES = ("exact-objective", "expected-margin")


def diagnostic_cases() -> list[dict]:
    """Six fixed inputs and two starts, declared without inspecting outcomes."""
    cases = []
    for n, p in ((4, 1), (4, 2), (6, 1), (6, 2)):
        graph = [[i, i + 1] for i in range(n - 1)] + [[0, n - 1]]
        cases.append(dict(case_id=f"qaoa-cycle-{n}-p{p}", algorithm="qaoa",
                          graph_family="cycle", n_qubits=n, p=p, edges=graph,
                          groups=["all_z"], target=dict(ratio_at_least=0.80),
                          initial_shots_per_group=128,
                          base_shots_per_group=468 if p == 1 else 260,
                          parameter_count=2 * p,
                          starts=[dict(start_id="original", parameters=[0.2, 0.1] * p),
                                  dict(start_id="alternate", parameters=[0.6, 0.4] * p)]))
    for n in (4, 6):
        cases.append(dict(case_id=f"vqe-open-chain-{n}", algorithm="vqe",
                          graph_family="open-chain", n_qubits=n, p=None, edges=[],
                          groups=["all_z", "all_x"], target=dict(density_gap_at_most=0.05),
                          initial_shots_per_group=64,
                          base_shots_per_group=130 if n == 4 else 90,
                          parameter_count=n,
                          hamiltonian="-sum_i Z_i+0.5 sum_i X_i X_(i+1); open chain",
                          ansatz="RY on each wire, then ascending CX(i,i+1)",
                          starts=[dict(start_id="original", parameters=[0.6] * n),
                                  dict(start_id="alternate", parameters=[0.0] * n)]))
    for index, case in enumerate(cases):
        b, d, g = case["base_shots_per_group"], case["parameter_count"], len(case["groups"])
        case["schedule"] = [[0.4, b, 4], [0.2, 2 * b, 4], [0.1, 4 * b, 4]]
        for start_index, start in enumerate(case["starts"]):
            start["stream_id"] = 30 + 2 * index + start_index
        calls = 1 + (1 + 2 * d) * 12
        training = g * (case["initial_shots_per_group"] + (1 + 2 * d) * 28 * b)
        if training != 65_648:
            raise RuntimeError("frozen matched-preparation identity is inconsistent")
        case["planned"] = dict(training_calls_per_seed=calls, training_groups_per_seed=g * calls,
                               training_shots_per_seed=training, validation_shots_per_seed=g * 16_384,
                               total_shots_per_seed=training + g * 16_384,
                               controls_per_start=2)
    return cases


def diagnostic_recipe() -> dict:
    """Outcome-independent recipe; virtual CPU allocations are not shot use."""
    cases = diagnostic_cases()
    return dict(schema_version=1, recipe_id=DIAGNOSTIC_RECIPE_ID,
                fixed_predeclared_recipe=True, fixed_published_recipe=False,
                cases=cases, seeds=list(SAMPLING_SEEDS), comparison_z=2.0,
                control_modes=list(DIAGNOSTIC_CONTROL_MODES),
                execution_order="all 36 sampled runs, then 24 deterministic CPU controls",
                budget_matched="training preparations only; not source gates, synthesis or physical cost",
                starts_policy="predeclared starts; no fifth winning angles or diagnostic witnesses",
                validation_shots_per_group=16_384, validation_failure_total=0.05,
                validation_failure_per_group=0.05 / 48,
                normalized_holdout_precision_target=0.05,
                planned=dict(sampled_runs=36, cpu_controls=24, training_calls=3636,
                             validation_calls=36, objective_calls=3672, training_groups=5232,
                             validation_groups=48, group_batches=5280, training_shots=2_363_328,
                             validation_shots=786_432, total_shots=3_149_760,
                             measured_qubit_readouts=15_748_800,
                             shot_weighted_source_unitary_gates=82_488_288,
                             cpu_control_evaluations_per_mode=1212, cpu_control_evaluations=2424,
                             cpu_control_final_audits=24),
                expected_weighted_gate_counts=dict(h=12_796_320, rx=14_765_760, rz=14_765_760,
                                                   ry=5_904_960, cx=34_255_488, measure=15_748_800),
                bounds=dict(simulation_qubits=8, dense_qubits=8, actual_max_qubits=6,
                            source_unitary_gates_per_circuit=100_000, group_batches=10_000,
                            sampled_shots=10_000_000, numerical_wall_seconds=120),
                control_semantics="classical exact-moment diagnostics, not zero-shot quantum algorithms or expected stochastic paths",
                formal_coverage="Classical C4 reference coverage only")


def _diagnostic_case(case: dict) -> dict:
    canonical = next((c for c in diagnostic_cases() if c["case_id"] == case.get("case_id")), None)
    if canonical is None or any(case.get(key) != value for key, value in canonical.items()):
        raise ValueError("case differs from the frozen diagnostic recipe")
    return canonical


def diagnostic_start(case: dict, start_id: str) -> dict:
    canonical = _diagnostic_case(case)
    for start in canonical["starts"]:
        if start["start_id"] == start_id:
            return start
    raise ValueError("start_id must be original or alternate")


def diagnostic_group_circuit(case: dict, parameters: Sequence[float], group_name: str) -> QuantumCircuit:
    """Actual source preparation/basis/readout, within the frozen input family."""
    canonical = _diagnostic_case(case)
    return quality_scaling_group_circuit(canonical, parameters, group_name)


def diagnostic_exact_moments(case: dict, parameters: Sequence[float], shots: int) -> dict:
    """Independent population moments for deterministic CPU controls only.

    The virtual allocation divides a true population variance, never a sample
    variance. All terms in each measured group are scored jointly. No random
    draws or quantum preparations occur; these are simulator calculations.
    """
    canonical = _diagnostic_case(case)
    if not 2 <= _positive_int(shots, "virtual sampling shots") <= 65_536:
        raise ValueError("virtual sampling shots must lie in 2..65536")
    point = _angles(parameters, "parameters")
    if len(point) != canonical["parameter_count"]:
        raise ValueError("parameter count differs from frozen diagnostic case")
    started = perf_counter()
    n = canonical["n_qubits"]
    integers = np.arange(1 << n)
    if canonical["algorithm"] == "qaoa":
        reference = qaoa_layers_reference(n, canonical["edges"], point)
        state = reference["reference_state"]
        scores = [("all_z", np.asarray(reference["cut_values"], dtype=float), state)]
    else:
        # RY tensor product and integer CNOT permutation, independent of Qiskit.
        product = np.ones(1, dtype=complex)
        for wire in reversed(range(n)):
            product = np.kron(product, [math.cos(point[wire] / 2), math.sin(point[wire] / 2)])
        state = np.zeros(1 << n, dtype=complex)
        for assignment, amplitude in enumerate(product):
            output = assignment
            for wire in range(n - 1):
                if (output >> wire) & 1:
                    output ^= 1 << (wire + 1)
            state[output] = amplitude
        x_state = state.copy()
        for wire in range(n):
            lo = integers[(integers & (1 << wire)) == 0]
            hi = lo ^ (1 << wire)
            a, b = x_state[lo].copy(), x_state[hi].copy()
            x_state[lo], x_state[hi] = (a + b) / math.sqrt(2), (a - b) / math.sqrt(2)
        signs = np.array([1.0 - 2.0 * ((integers >> wire) & 1) for wire in range(n)])
        scores = [("all_z", -signs.sum(axis=0), state),
                  ("all_x", 0.5 * (signs[:-1] * signs[1:]).sum(axis=0), x_state)]
    groups = []
    for group, values, measured_state in scores:
        probabilities = np.abs(measured_state) ** 2
        probabilities /= probabilities.sum()
        mean = float(probabilities @ values)
        variance = float(probabilities @ ((values - mean) ** 2))
        groups.append(dict(group=group, mean=mean, population_variance=variance,
                           variance_of_mean=variance / shots, virtual_sampling_shots=shots,
                           outcome_min=float(values.min()), outcome_max=float(values.max()),
                           quantum_preparations=0, count_samples=None,
                           joint_term_covariance_retained=True))
    mean = sum(g["mean"] for g in groups)
    if canonical["algorithm"] == "qaoa":
        mean = -mean
    elif abs(mean - spin_vqe_energy_formula(n, point)) > 1e-10:
        raise RuntimeError("independent group moments disagree with the scalar VQE identity")
    return dict(parameters=list(point), objective_mean=mean,
                expected_standard_error=math.sqrt(sum(g["variance_of_mean"] for g in groups)),
                groups=groups, quantum_preparations=0, count_samples=None, seed=None,
                evaluation_wall_seconds=perf_counter() - started,
                method="independent ideal population moments; virtual allocations only; CPU diagnostic")


def diagnostic_final_audit(case: dict, parameters: Sequence[float]) -> dict:
    """Final independent quality/circuit audit; never an optimizer input."""
    canonical = _diagnostic_case(case)
    reference = (qaoa_layers_reference(canonical["n_qubits"], canonical["edges"], parameters)
                 if canonical["algorithm"] == "qaoa"
                 else spin_vqe_reference(canonical["n_qubits"], parameters))
    reference_state = reference.pop("reference_state")
    started = perf_counter()
    source = diagnostic_group_circuit(canonical, parameters, "all_z").remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(source).data
    overlap = np.vdot(reference_state, state)
    phase = overlap / abs(overlap) if abs(overlap) else 1.0
    error = float(np.max(np.abs(state - phase * reference_state)))
    if error > 1e-11:
        raise RuntimeError("diagnostic final source disagrees with independent state algebra")
    reference.update(source_state_max_phase_aligned_error=error,
                     source_state_audit_wall_seconds=perf_counter() - started,
                     audit_performed_after_training=True)
    if canonical["algorithm"] == "qaoa":
        met = reference["exact_approximation_ratio"] >= canonical["target"]["ratio_at_least"]
        reference.update(zero_assignment_cut=0, uniform_assignment_mean_cut=len(canonical["edges"]) / 2)
    else:
        met = reference["energy_density_gap"] <= canonical["target"]["density_gap_at_most"]
        reference.update(zero_state_energy=-canonical["n_qubits"],
                         zero_state_density_gap=(-canonical["n_qubits"] - reference["ground_energy"]) / canonical["n_qubits"])
    return dict(diagnostic=reference, target_met_audit=bool(met))


def run_diagnostic_sampled(case: dict, start_id: str, seed: int) -> dict:
    """One frozen seeded run, retained in full including every rejection."""
    canonical = _diagnostic_case(case)
    start, recipe = diagnostic_start(canonical, start_id), diagnostic_recipe()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in recipe["seeds"]:
        raise ValueError("seed must be one of the frozen diagnostic seeds")
    begin = perf_counter()
    def objective(parameters, **kwargs):
        if perf_counter() - begin > recipe["bounds"]["numerical_wall_seconds"]:
            raise RuntimeError("diagnostic run exceeded its bounded CPU wall allowance")
        if canonical["algorithm"] == "qaoa":
            return sampled_qaoa_layers_objective(parameters, n_qubits=canonical["n_qubits"],
                                                 edges=canonical["edges"], stream_id=start["stream_id"], **kwargs)
        return sampled_spin_vqe_objective(parameters, n_qubits=canonical["n_qubits"],
                                          stream_id=start["stream_id"], **kwargs)
    run = sampled_pattern_search(objective, start["parameters"], seed=seed,
                                 schedule=canonical["schedule"], initial_shots=canonical["initial_shots_per_group"],
                                 comparison_z=recipe["comparison_z"])
    validation = objective(run["parameters"], shots=recipe["validation_shots_per_group"],
                           seed=seed, evaluation_index=0, family=1)
    validation.update(role="independent_final_validation", confidence_scope="conditional on fixed final parameters")
    audit = diagnostic_final_audit(canonical, run["parameters"])
    validation["quality_bound"] = _quality_holdout_bounds(canonical, validation, audit["diagnostic"], recipe)
    run.update(**audit, case_id=canonical["case_id"], start_id=start_id, stream_id=start["stream_id"],
               run_id=f"{canonical['case_id']}-{start_id}-seed-{seed}",
               validation=validation, total_shots_including_validation=run["total_shots"] + validation["total_shots"],
               quantum_preparations=run["total_shots"] + validation["total_shots"],
               exact_diagnostics_used_for_optimization=False, kind="sampled",
               run_wall_seconds=perf_counter() - begin)
    if (run["total_shots"] != canonical["planned"]["training_shots_per_seed"]
            or run["objective_evaluations"] != canonical["planned"]["training_calls_per_seed"]
            or run["total_shots_including_validation"] != canonical["planned"]["total_shots_per_seed"]):
        raise RuntimeError("sampled run allocation disagrees with frozen recipe")
    return run


def diagnostic_aggregate(recipe: dict, cases: list[dict], controls: list[dict]) -> dict:
    """Assemble separately labeled sampled and CPU-control records."""
    if recipe != diagnostic_recipe():
        raise ValueError("recipe differs from frozen diagnostic design")
    rows, control_rows = [], []
    for case in cases:
        _diagnostic_case(case)
        for run in case["runs"]:
            reference, bound = run["diagnostic"], run["validation"]["quality_bound"]
            rows.append(dict(case_id=case["case_id"], algorithm=case["algorithm"], n_qubits=case["n_qubits"],
                             p=case["p"], start_id=run["start_id"], seed=run["seed"], run_id=run["run_id"],
                             ideal_quality=reference["exact_approximation_ratio"] if case["algorithm"] == "qaoa" else reference["energy_density_gap"],
                             ideal_target_met=run["target_met_audit"], heldout_estimate=bound["normalized_estimate"],
                             heldout_simultaneous_radius=bound["simultaneous_radius"],
                             heldout_target_status=bound["target_status"], precision_supported=bound["precision_supported"],
                             training_calls=run["objective_evaluations"], training_shots=run["total_shots"],
                             validation_shots=run["validation"]["total_shots"], total_shots=run["total_shots_including_validation"],
                             accepted_proposals=run["accepted_proposals"], rejected_proposals=run["rejected_proposals"],
                             optimizer_wall_seconds=run["optimizer_wall_seconds"], objective_wall_seconds=run["objective_wall_seconds"],
                             classical_control_seconds=run["classical_control_seconds"],
                             independent_reference_wall_seconds=reference["reference_wall_seconds"],
                             source_state_audit_wall_seconds=reference["source_state_audit_wall_seconds"]))
    by_id = {case["case_id"]: case for case in cases}
    for run in controls:
        case, reference = by_id[run["case_id"]], run["diagnostic"]
        control_rows.append(dict(case_id=run["case_id"], algorithm=case["algorithm"], n_qubits=case["n_qubits"],
                                 p=case["p"], start_id=run["start_id"], mode=run["mode"],
                                 ideal_quality=reference["exact_approximation_ratio"] if case["algorithm"] == "qaoa" else reference["energy_density_gap"],
                                 ideal_target_met=run["target_met_audit"], objective_evaluations=run["objective_evaluations"],
                                 accepted_proposals=run["accepted_proposals"], rejected_proposals=run["rejected_proposals"],
                                 quantum_preparations=0, seed=None, count_samples=None,
                                 optimizer_wall_seconds=run["optimizer_wall_seconds"]))
    return dict(schema_version=1, recipe=recipe, cases=cases, controls=controls,
                summary_rows=rows, control_summary_rows=control_rows,
                training_shots=sum(r["training_shots"] for r in rows),
                validation_shots=sum(r["validation_shots"] for r in rows),
                total_shots=sum(r["total_shots"] for r in rows),
                cpu_control_objective_evaluations=sum(r["objective_evaluations"] for r in controls),
                cpu_control_final_audits=len(controls), exact_diagnostics_used_for_optimization=False,
                exact_diagnostics_scope="sampled optimizer decisions only; CPU controls explicitly use exact moments",
                hardware_feedback_seconds=None, physical_resource_estimate=None,
                quality_selection="every frozen case/start/seed, every miss and inconclusive holdout retained",
                limitations=["ideal CPU sampling only; no hardware-calibrated noise, predictive convergence or quantum advantage",
                             "matched training preparations do not match gate, synthesis or physical costs",
                             "CPU controls use exact moments, not zero-shot quantum algorithms or expected stochastic paths",
                             "no global ansatz optimum or infeasibility certified by failed local polling",
                             "holdout confidence is conditional on frozen final states and excludes adaptive training",
                             "source ledger excludes unmeasured physical reset, routing, synthesis, fault correction and feedback",
                             "Classical C4 reference coverage only"])


def diagnostic_replay_sampled(case: dict, run: dict) -> None:
    """Count-only validation of moments and every prescribed polling choice.

    This audit uses stored integer counts, not a new draw or ideal expectation.
    It cannot change a sampled start or make an optimizer decision.
    """
    canonical = _diagnostic_case(case)
    point = np.asarray(diagnostic_start(canonical, run["start_id"])["parameters"])
    point = (point + math.pi) % (2 * math.pi) - math.pi
    records = run["evaluations"]
    for record in [*records, run["validation"]]:
        means, variances = [], []
        for group in record["groups"]:
            counts, shots = group["counts"], group["shots"]
            def score(bits):
                assignment = int(bits, 2)
                if canonical["algorithm"] == "qaoa":
                    return float(sum(((assignment >> u) ^ (assignment >> v)) & 1 for u, v in canonical["edges"]))
                signs = [1 - 2 * ((assignment >> wire) & 1) for wire in range(canonical["n_qubits"])]
                return (-float(sum(signs)) if group["group"] == "all_z"
                        else 0.5 * sum(signs[i] * signs[i + 1] for i in range(len(signs) - 1)))
            mean = sum(number * score(bits) for bits, number in counts.items()) / shots
            variance = sum(number * (score(bits) - mean) ** 2 for bits, number in counts.items()) / (shots - 1)
            if any(not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12) for actual, expected in (
                    (group["estimate"], mean), (group["sample_variance"], variance),
                    (group["estimated_standard_error"], math.sqrt(variance / shots)))):
                raise ValueError("recorded measured moments do not replay from integer counts")
            means.append(mean)
            variances.append(variance / shots)
        objective = (-sum(means) if canonical["algorithm"] == "qaoa" else sum(means))
        if (not math.isclose(record["objective_estimate"], objective, rel_tol=1e-11, abs_tol=1e-12)
                or not math.isclose(record["estimated_standard_error"], math.sqrt(sum(variances)), rel_tol=1e-11, abs_tol=1e-12)):
            raise ValueError("joint objective does not replay from measured groups")
    if records[0]["role"] != "initial" or not np.allclose(records[0]["parameters"], point, atol=1e-14, rtol=0):
        raise ValueError("initial evaluation differs from the frozen start")
    expected_index, accepted_indexes = 1, set()
    current = records[0]
    expanded = [(step, shots) for step, shots, sweeps in canonical["schedule"] for _ in range(sweeps)]
    if len(run["iterations"]) != len(expanded):
        raise ValueError("polling iterations do not exhaust the declared schedule")
    for iteration, (step, shots) in zip(run["iterations"], expanded):
        incumbent = records[expected_index]
        proposal_indexes = list(range(expected_index + 1, expected_index + 1 + 2 * len(point)))
        if (incumbent["role"] != "incumbent" or iteration["incumbent_evaluation"] != expected_index
                or iteration["proposal_evaluations"] != proposal_indexes or iteration["step"] != step
                or iteration["shots_per_group"] != shots
                or not np.allclose(iteration["parameters_before"], point, atol=1e-14, rtol=0)
                or not np.allclose(incumbent["parameters"], point, atol=1e-14, rtol=0)):
            raise ValueError("incumbent or proposal indexes differ from fixed polling schedule")
        proposals = [records[i] for i in proposal_indexes]
        for offset, proposal in enumerate(proposals):
            axis, direction = offset // 2, -1 if offset % 2 == 0 else 1
            trial = point.copy()
            trial[axis] += direction * step
            trial = (trial + math.pi) % (2 * math.pi) - math.pi
            if (proposal["role"] != "proposal" or proposal["axis"] != axis or proposal["direction"] != direction
                    or not np.allclose(proposal["parameters"], trial, atol=1e-14, rtol=0)):
                raise ValueError("candidate parameters or order differ from fixed polling schedule")
        best = min(proposals, key=lambda record: record["objective_estimate"])
        improvement = incumbent["objective_estimate"] - best["objective_estimate"]
        margin = 2.0 * math.hypot(incumbent["estimated_standard_error"], best["estimated_standard_error"])
        accepted = improvement > margin
        if (iteration["accepted"] != accepted or iteration["best_proposal_evaluation"] != best["index"]
                or not math.isclose(iteration["comparison_margin"], margin, rel_tol=1e-12, abs_tol=1e-14)
                or not math.isclose(iteration["sampled_improvement"], improvement, rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError("acceptance decision does not replay from counted objectives")
        if accepted:
            accepted_indexes.add(best["index"])
            point, current = np.asarray(best["parameters"]), best
        else:
            current = incumbent
        if not np.allclose(iteration["parameters_after"], point, atol=1e-14, rtol=0):
            raise ValueError("post-decision parameters disagree with the recorded move")
        expected_index += 1 + 2 * len(point)
    if (expected_index != len(records) or not np.allclose(run["parameters"], point, atol=1e-14, rtol=0)
            or any(record["accepted"] != (record["index"] in accepted_indexes) for record in records)
            or run["accepted_proposals"] != len(accepted_indexes)
            or run["rejected_proposals"] != 24 * len(point) - len(accepted_indexes)
            or run["final_training_estimate"] != current["objective_estimate"]
            or run["final_training_standard_error"] != current["estimated_standard_error"]
            or run["stop_reason"] != "fixed_schedule_exhausted" or run["convergence_certified"] is not False):
        raise ValueError("run final state, rejected proposals or stopping record does not replay")


def diagnostic_resources(study: dict) -> dict:
    """Reconstruct and validate the complete measured source workload."""
    recipe = diagnostic_recipe()
    if study["recipe"] != recipe or [c["case_id"] for c in study["cases"]] != [c["case_id"] for c in recipe["cases"]]:
        raise ValueError("study case order or recipe differs from frozen design")
    weighted, per_run = Counter(), []
    calls = training_groups = validation_groups = training_shots = validation_shots = readouts = 0
    seen_streams = set()
    for case in study["cases"]:
        canonical = _diagnostic_case(case)
        expected_keys = [(s["start_id"], seed) for s in canonical["starts"] for seed in recipe["seeds"]]
        if [(r["start_id"], r["seed"]) for r in case["runs"]] != expected_keys:
            raise ValueError("sampled case/start/seed order or completeness differs from recipe")
        allocations = [canonical["initial_shots_per_group"]]
        for _, shots, sweeps in canonical["schedule"]:
            allocations.extend([shots] * ((1 + 2 * canonical["parameter_count"]) * sweeps))
        for run in case["runs"]:
            start = diagnostic_start(canonical, run["start_id"])
            if run["initial"] != start["parameters"] or run["stream_id"] != start["stream_id"]:
                raise ValueError("sampled start differs from the frozen declaration")
            if len(run["evaluations"]) != len(allocations) or len(run["iterations"]) != 12:
                raise ValueError("sampled run does not exhaust the fixed schedule")
            diagnostic_replay_sampled(canonical, run)
            bound = _quality_holdout_bounds(canonical, run["validation"], run["diagnostic"], recipe)
            if run["validation"]["quality_bound"] != bound:
                raise ValueError("holdout precision/quality classification does not replay")
            run_weighted, run_training, run_validation = Counter(), 0, 0
            for phase, records in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for index, record in enumerate(records):
                    calls += 1
                    if phase == "training" and record["index"] != index:
                        raise ValueError("training evaluation indexes are inconsistent")
                    if phase == "validation" and record["parameters"] != run["parameters"]:
                        raise ValueError("holdout changed the fitted parameters")
                    if [g["group"] for g in record["groups"]] != canonical["groups"]:
                        raise ValueError("measured groups differ from recipe")
                    if record["total_shots"] != sum(g["shots"] for g in record["groups"]):
                        raise ValueError("objective group shot total is inconsistent")
                    for group_index, group in enumerate(record["groups"]):
                        expected_range = ((0, len(canonical["edges"])) if canonical["algorithm"] == "qaoa"
                                          else (-canonical["n_qubits"], canonical["n_qubits"]) if group["group"] == "all_z"
                                          else (-0.5 * (canonical["n_qubits"] - 1), 0.5 * (canonical["n_qubits"] - 1)))
                        if (group["outcome_min"], group["outcome_max"]) != expected_range:
                            raise ValueError("measurement score range differs from actual fixed observable")
                        shots = allocations[index] if phase == "training" else recipe["validation_shots_per_group"]
                        key = [run["seed"], start["stream_id"], 0 if phase == "training" else 1, index, group_index]
                        if (group["shots"] != shots or group["rng"]["stream_key"] != key
                                or group["rng"]["entropy"] != key[:2] or group["rng"]["spawn_key"] != key[2:]
                                or tuple(key) in seen_streams):
                            raise ValueError("group shot allocation or RNG lineage differs from recipe")
                        seen_streams.add(tuple(key))
                        counts = group["counts"]
                        if not counts or sum(counts.values()) != shots or any(
                                len(bits) != canonical["n_qubits"] or set(bits) - {"0", "1"}
                                or isinstance(number, bool) or not isinstance(number, int) or number <= 0
                                for bits, number in counts.items()):
                            raise ValueError("sampled counts are incomplete or invalid")
                        source = diagnostic_group_circuit(canonical, record["parameters"], group["group"])
                        source_counts = {name: int(number) for name, number in source.count_ops().items()}
                        if (group["circuit_gate_counts"] != source_counts or group["circuit_depth"] != source.depth()
                                or group["logical_qubits"] != source.num_qubits
                                or group["measured_qubit_readouts"] != shots * source.num_qubits):
                            raise ValueError("source metadata disagrees with actual circuit reconstruction")
                        if sum(v for k, v in source_counts.items() if k not in ("measure", "barrier")) > recipe["bounds"]["source_unitary_gates_per_circuit"]:
                            raise ValueError("source exceeds materialization bound")
                        run_weighted.update({k: v * shots for k, v in source_counts.items() if k != "barrier"})
                        readouts += shots * source.num_qubits
                        if phase == "training":
                            training_groups += 1
                            run_training += shots
                        else:
                            validation_groups += 1
                            run_validation += shots
            if (run_training != 65_648 or run_training != run["total_shots"]
                    or run_training + run_validation != run["total_shots_including_validation"]
                    or run["quantum_preparations"] != run_training + run_validation):
                raise ValueError("run does not satisfy matched training preparation budget")
            weighted.update(run_weighted)
            training_shots += run_training
            validation_shots += run_validation
            per_run.append(dict(case_id=canonical["case_id"], start_id=run["start_id"], seed=run["seed"],
                                n_qubits=canonical["n_qubits"], p=canonical["p"], algorithm=canonical["algorithm"],
                                training_preparations=run_training, validation_preparations=run_validation,
                                total_shots=run_training + run_validation, weighted_gate_counts=dict(run_weighted),
                                optimizer_wall_seconds=run["optimizer_wall_seconds"],
                                hardware_feedback_seconds=None))
    expected_controls = [(c["case_id"], s["start_id"], mode) for c in recipe["cases"] for s in c["starts"] for mode in recipe["control_modes"]]
    if [(r["case_id"], r["start_id"], r["mode"]) for r in study["controls"]] != expected_controls:
        raise ValueError("CPU control order or completeness differs from recipe")
    control_calls = 0
    for run in study["controls"]:
        diagnostic_replay_control(next(c for c in recipe["cases"] if c["case_id"] == run["case_id"]), run)
        case = next(c for c in recipe["cases"] if c["case_id"] == run["case_id"])
        if (run["quantum_preparations"] != 0 or run["seed"] is not None or run["count_samples"] is not None
                or run["objective_evaluations"] != case["planned"]["training_calls_per_seed"]
                or len(run["evaluations"]) != run["objective_evaluations"]):
            raise ValueError("CPU control masquerades as sampled use or omits evaluations")
        control_calls += run["objective_evaluations"]
        if any(e["quantum_preparations"] != 0 or e["count_samples"] is not None for e in run["evaluations"]):
            raise ValueError("CPU control evaluations must have no sampled preparations")
    actual = dict(objective_calls=calls, training_groups=training_groups, validation_groups=validation_groups,
                  group_batches=training_groups + validation_groups, training_shots=training_shots,
                  validation_shots=validation_shots, total_shots=training_shots + validation_shots,
                  measured_qubit_readouts=readouts, cpu_control_evaluations=control_calls)
    if any(value != recipe["planned"][key] for key, value in actual.items()) or dict(weighted) != recipe["expected_weighted_gate_counts"]:
        raise ValueError("complete actual workload differs from frozen budget")
    if (study["training_shots"] != training_shots or study["validation_shots"] != validation_shots
            or study["total_shots"] != training_shots + validation_shots):
        raise ValueError("aggregate sampled totals are inconsistent")
    return dict(schema_version=1, kind="complete unsynthesized recorded source logical workload",
                recipe_id=recipe["recipe_id"], **actual, per_run=per_run,
                total_preparations=training_shots + validation_shots,
                weighted_gate_counts=dict(weighted), logical_peak_qubits=6,
                shot_weighted_source_unitary_gates=sum(v for k, v in weighted.items() if k != "measure"),
                cpu_controls=len(study["controls"]), cpu_control_final_audits=len(study["controls"]),
                cpu_control_quantum_preparations=0, cpu_control_virtual_allocations_excluded=True,
                physical_resource_estimate=None, hardware_feedback_seconds=None,
                scaling=dict(qaoa="H=n,RX=np,RZ=mp,CX=2mp,measure=n per group",
                             vqe="RY=n,CX=n-1,H=n for all-X,measure=n per group",
                             calls="1+(1+2D)*12 plus one holdout; g groups per call",
                             training_preparations="g*[initial+(1+2D)*28b]=65648",
                             statevector_memory="O(2^n)", dense_reference_memory="O(4^n)"),
                exclusions=study["limitations"])


def run_diagnostic_control(case: dict, start_id: str, mode: str, *,
                           max_wall_seconds: float = 120.0) -> dict:
    """Run one frozen deterministic CPU control, without RNG or quantum shots.

    exact-objective uses exact means and zero effective comparison SE while
    retaining every state's population variance. expected-margin uses the same
    means and the population-based sqrt(sum(variance/virtual_shots)) supplied
    by diagnostic_exact_moments. Virtual allocations annotate this CPU control;
    they are not real or modeled consumption of quantum preparations. Neither
    mode certifies convergence, hardware performance, or sampling confidence.
    """
    run_started = perf_counter()
    from time import process_time
    if mode not in ("exact-objective", "expected-margin"):
        raise ValueError("diagnostic control mode must be exact-objective or expected-margin")
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= 120.0:
        raise ValueError("diagnostic control wall allowance must lie in (0,120] seconds")
    if case["n_qubits"] > 8:
        raise ValueError("diagnostic CPU controls are capped at eight qubits")
    start = diagnostic_start(case, start_id)
    initial = _angles(start["parameters"], "initial")
    if len(initial) > 8:
        raise ValueError("diagnostic CPU controls support at most eight coordinates")
    initial_virtual = _positive_int(case["initial_shots_per_group"], "initial virtual shots")
    schedule, expanded = [], []
    for step, virtual_shots, sweeps in case["schedule"]:
        _positive_int(virtual_shots, "virtual shots")
        _positive_int(sweeps, "sweeps")
        if not math.isfinite(step) or step <= 0 or not 2 <= virtual_shots <= 65_536:
            raise ValueError("diagnostic virtual schedule requires positive steps and 2..65536 allocations")
        schedule.append(dict(step=float(step), virtual_shots_per_group=virtual_shots, sweeps=sweeps))
        expanded.extend([(float(step), virtual_shots)] * sweeps)
    if len(expanded) != 12:
        raise ValueError("diagnostic CPU control requires the frozen twelve-sweep schedule")
    group_names = list(case["groups"])
    if group_names != (["all_z"] if case["algorithm"] == "qaoa" else ["all_z", "all_x"]):
        raise ValueError("diagnostic CPU control requires the frozen joint-score groups")
    planned_virtual = len(group_names) * (initial_virtual + (1 + 2 * len(initial))
                                         * sum(virtual for _, virtual in expanded))
    if planned_virtual != 65_648:
        raise ValueError("diagnostic CPU control requires the frozen 65648 virtual-preparation annotation")
    point = (np.asarray(initial, dtype=float) + math.pi) % (2 * math.pi) - math.pi
    started, cpu_started = perf_counter(), process_time()
    evaluations, iterations = [], []

    def evaluate(parameters, virtual_shots, role, iteration_index):
        if perf_counter() - started > max_wall_seconds:
            raise RuntimeError("diagnostic CPU control exceeded its bounded wall allowance")
        parameters = (np.asarray(parameters) + math.pi) % (2 * math.pi) - math.pi
        evaluation_cpu_started = process_time()
        record = dict(diagnostic_exact_moments(case, parameters.tolist(), virtual_shots))
        record["groups"] = [dict(group) for group in record["groups"]]
        record["evaluation_process_cpu_seconds"] = process_time() - evaluation_cpu_started
        effective_se = 0.0 if mode == "exact-objective" else record["expected_standard_error"]
        if (not math.isfinite(record["objective_mean"]) or not math.isfinite(effective_se)
                or effective_se < 0):
            raise RuntimeError("diagnostic CPU control produced invalid exact moments")
        if [group["group"] for group in record["groups"]] != group_names:
            raise RuntimeError("diagnostic CPU control changed its measurement-group annotation")
        if any(group["virtual_sampling_shots"] != virtual_shots for group in record["groups"]):
            raise RuntimeError("diagnostic moments changed the virtual allocation")
        expected_variance = sum(group["variance_of_mean"] for group in record["groups"])
        if (any(not math.isfinite(group["population_variance"]) or group["population_variance"] < 0
                for group in record["groups"])
                or not math.isclose(record["expected_standard_error"] ** 2,
                                    expected_variance, rel_tol=1e-10, abs_tol=1e-12)):
            raise RuntimeError("diagnostic population variances and expected mean SE disagree")
        for group in record["groups"]:
            group.update(quantum_preparations=0, count_samples=None,
                         allocation_scope="virtual CPU-control annotation; no measurement draws")
        record.update(index=len(evaluations), role=role, iteration_index=iteration_index,
                      accepted=False, parameters=parameters.tolist(),
                      virtual_shots_per_group=virtual_shots,
                      virtual_preparations=len(group_names) * virtual_shots,
                      effective_standard_error=effective_se,
                      quantum_preparations=0, count_samples=None, hardware_wall_seconds=None,
                      effective_standard_error_scope=("zero comparison uncertainty for exact-objective control"
                          if mode == "exact-objective" else
                          "population-based hypothetical mean SE under virtual allocation; no samples"))
        evaluations.append(record)
        return record

    current = evaluate(point, initial_virtual, "initial", None)
    for iteration, (step, virtual_shots) in enumerate(expanded):
        before = point.copy()
        incumbent = evaluate(point, virtual_shots, "incumbent", iteration)
        proposals = []
        for axis in range(len(point)):
            for direction in (-1, 1):
                trial = point.copy()
                trial[axis] += direction * step
                proposal = evaluate(trial, virtual_shots, "proposal", iteration)
                proposal.update(axis=axis, direction=direction)
                proposals.append(proposal)
        best = min(proposals, key=lambda record: record["objective_mean"])
        improvement = incumbent["objective_mean"] - best["objective_mean"]
        margin = 2.0 * math.hypot(incumbent["effective_standard_error"],
                                best["effective_standard_error"])
        accepted = improvement > margin
        if accepted:
            point = np.asarray(best["parameters"], dtype=float)
            best["accepted"] = True
            current = best
        else:
            current = incumbent
        iterations.append(dict(index=iteration, step=step,
                               virtual_shots_per_group=virtual_shots,
                               parameters_before=before.tolist(), parameters_after=point.tolist(),
                               incumbent_evaluation=incumbent["index"],
                               proposal_evaluations=[proposal["index"] for proposal in proposals],
                               best_proposal_evaluation=best["index"], accepted=accepted,
                               exact_improvement=improvement, comparison_margin=margin,
                               quantum_preparations=0, count_samples=None))
    optimizer_wall = perf_counter() - started
    optimizer_cpu = process_time() - cpu_started
    objective_wall = sum(record["evaluation_wall_seconds"] for record in evaluations)
    objective_cpu = sum(record["evaluation_process_cpu_seconds"] for record in evaluations)
    virtual_preparations = sum(record["virtual_preparations"] for record in evaluations)
    if virtual_preparations != planned_virtual:
        raise RuntimeError("diagnostic control record violates its virtual budget annotation")
    audit_started, audit_cpu_started = perf_counter(), process_time()
    final_audit = diagnostic_final_audit(case, point.tolist())
    final_audit_wall = perf_counter() - audit_started
    final_audit_cpu = process_time() - audit_cpu_started
    workflow_wall = perf_counter() - started
    workflow_cpu = process_time() - cpu_started
    if workflow_wall > max_wall_seconds:
        raise RuntimeError("diagnostic CPU control exceeded its bounded wall allowance")
    result = dict(case_id=case["case_id"], algorithm=case["algorithm"], n_qubits=case["n_qubits"],
                  p=case["p"], start_id=start_id, mode=mode, seed=None, kind="cpu-control",
                  run_id=f"{case['case_id']}-{start_id}-{mode}", initial=list(initial),
                  parameters=point.tolist(), final_parameters=point.tolist(),
                  final_exact_objective=current["objective_mean"],
                  final_effective_standard_error=current["effective_standard_error"],
                  final_expected_standard_error=current["expected_standard_error"],
                  final_evaluation=current["index"], evaluations=evaluations, iterations=iterations,
                  objective_evaluations=len(evaluations), oracle_evaluations=len(evaluations),
                  oracle_evaluation_scope="exact bounded CPU polling-objective calls; final audit counted separately; no quantum oracle calls",
                  final_audit_count=1,
                  iteration_count=len(iterations),
                  accepted_proposals=sum(record["accepted"] for record in evaluations if record["role"] == "proposal"),
                  rejected_proposals=sum(not record["accepted"] for record in evaluations if record["role"] == "proposal"),
                  virtual_training_preparations=virtual_preparations,
                  planned_virtual_training_preparations=planned_virtual,
                  virtual_budget_scope="CPU-control annotations only; no real or modeled quantum shot consumption",
                  quantum_preparations=0, count_samples=None, hardware_wall_seconds=None,
                  hardware_feedback_seconds=None, stop_reason="fixed_schedule_exhausted",
                  convergence_certified=False,
                  method="deterministic exact-moment CPU control; no sampling or QPU execution",
                  optimizer_wall_seconds=optimizer_wall, objective_wall_seconds=objective_wall,
                  classical_control_seconds=max(0.0, optimizer_wall - objective_wall),
                  optimizer_process_cpu_seconds=optimizer_cpu, objective_process_cpu_seconds=objective_cpu,
                  classical_control_process_cpu_seconds=max(0.0, optimizer_cpu - objective_cpu),
                  final_audit_wall_seconds=final_audit_wall, workflow_wall_seconds=workflow_wall,
                  final_audit_process_cpu_seconds=final_audit_cpu, process_cpu_seconds=workflow_cpu,
                  timing_scope="measured CPU reference and controller times; no hardware performance measurement",
                  settings=dict(schedule=schedule, initial_virtual_shots_per_group=initial_virtual,
                                comparison_z=2.0, periodic_domain="[-pi,pi)",
                                polling_order="axis ascending, minus then plus; fresh incumbent each sweep",
                                tie_break="first minimum in frozen polling order",
                                acceptance_semantics="strict exact reduction exceeds descriptive margin; no confidence/convergence claim",
                                limits=dict(qubits=8, parameters=8, source_unitary_gates=100_000,
                                            workflow_wall_seconds=max_wall_seconds)),
                  limitations=["deterministic exact-reference CPU controls; no RNG, counts or quantum preparations",
                               "virtual allocations enter only the expected-margin annotation",
                               "fixed schedule exhaustion supplies no convergence or hardware-performance certificate",
                               "population variances remain state properties even when effective comparison SE is zero"])
    result.update(final_audit)
    result["run_wall_seconds"] = perf_counter() - run_started
    if result["run_wall_seconds"] > max_wall_seconds:
        raise RuntimeError("diagnostic CPU control exceeded its bounded wall allowance")
    return result


def diagnostic_replay_control(case: dict, run: dict) -> None:
    """Validate stored CPU moments/decisions without new objective calls."""
    canonical = _diagnostic_case(case)
    start = diagnostic_start(canonical, run["start_id"])
    mode = run["mode"]
    if (mode not in DIAGNOSTIC_CONTROL_MODES or run["initial"] != start["parameters"]
            or run["seed"] is not None or run["count_samples"] is not None or run["quantum_preparations"] != 0
            or run["final_audit_count"] != 1 or run["convergence_certified"] is not False):
        raise ValueError("CPU control identity, initialization or accounting differs from recipe")
    expanded = [(step, shots) for step, shots, sweeps in canonical["schedule"] for _ in range(sweeps)]
    allocations = [canonical["initial_shots_per_group"]]
    for _, shots in expanded:
        allocations.extend([shots] * (1 + 2 * canonical["parameter_count"]))
    records = run["evaluations"]
    if len(records) != len(allocations) or len(run["iterations"]) != 12:
        raise ValueError("CPU control does not exhaust the fixed schedule")
    virtual_total = 0
    for index, (record, shots) in enumerate(zip(records, allocations)):
        if (record["index"] != index or record["virtual_shots_per_group"] != shots
                or record["quantum_preparations"] != 0 or record["count_samples"] is not None
                or [g["group"] for g in record["groups"]] != canonical["groups"]):
            raise ValueError("CPU moment record changed its order, groups or virtual allocation")
        for group in record["groups"]:
            if (group["virtual_sampling_shots"] != shots or group["quantum_preparations"] != 0
                    or group["count_samples"] is not None or group["population_variance"] < 0
                    or not math.isfinite(group["population_variance"])
                    or not math.isclose(group["variance_of_mean"], group["population_variance"] / shots, rel_tol=1e-12, abs_tol=1e-14)):
                raise ValueError("CPU population variance or virtual allocation is inconsistent")
        expected_se = math.sqrt(sum(g["variance_of_mean"] for g in record["groups"]))
        objective = sum(g["mean"] for g in record["groups"])
        if canonical["algorithm"] == "qaoa":
            objective = -objective
        effective_se = 0.0 if mode == "exact-objective" else expected_se
        if (not math.isclose(record["objective_mean"], objective, rel_tol=1e-12, abs_tol=1e-14)
                or not math.isclose(record["expected_standard_error"], expected_se, rel_tol=1e-12, abs_tol=1e-14)
                or record["effective_standard_error"] != effective_se
                or record["virtual_preparations"] != len(canonical["groups"]) * shots):
            raise ValueError("CPU objective or effective SE is inconsistent with its mode")
        virtual_total += record["virtual_preparations"]
    point = (np.asarray(start["parameters"]) + math.pi) % (2 * math.pi) - math.pi
    expected_index, accepted_indexes, current = 1, set(), records[0]
    if current["role"] != "initial" or not np.allclose(current["parameters"], point, atol=1e-14, rtol=0):
        raise ValueError("CPU initial evaluation differs from start")
    for iteration, (step, shots) in zip(run["iterations"], expanded):
        incumbent = records[expected_index]
        proposal_indexes = list(range(expected_index + 1, expected_index + 1 + 2 * len(point)))
        proposals = [records[index] for index in proposal_indexes]
        if (incumbent["role"] != "incumbent" or iteration["incumbent_evaluation"] != expected_index
                or iteration["proposal_evaluations"] != proposal_indexes or iteration["step"] != step
                or iteration["virtual_shots_per_group"] != shots
                or not np.allclose(incumbent["parameters"], point, atol=1e-14, rtol=0)
                or not np.allclose(iteration["parameters_before"], point, atol=1e-14, rtol=0)):
            raise ValueError("CPU incumbent or polling order differs from fixed schedule")
        for offset, proposal in enumerate(proposals):
            axis, direction = offset // 2, -1 if offset % 2 == 0 else 1
            trial = point.copy()
            trial[axis] += direction * step
            trial = (trial + math.pi) % (2 * math.pi) - math.pi
            if (proposal["role"] != "proposal" or proposal["axis"] != axis or proposal["direction"] != direction
                    or not np.allclose(proposal["parameters"], trial, atol=1e-14, rtol=0)):
                raise ValueError("CPU candidate parameters differ from prescribed polling")
        best = min(proposals, key=lambda record: record["objective_mean"])
        improvement = incumbent["objective_mean"] - best["objective_mean"]
        margin = 2.0 * math.hypot(incumbent["effective_standard_error"], best["effective_standard_error"])
        accepted = improvement > margin
        if (iteration["accepted"] != accepted or iteration["best_proposal_evaluation"] != best["index"]
                or iteration["exact_improvement"] != improvement or iteration["comparison_margin"] != margin):
            raise ValueError("CPU acceptance does not replay from stored exact moments")
        if accepted:
            accepted_indexes.add(best["index"])
            point, current = np.asarray(best["parameters"]), best
        else:
            current = incumbent
        if not np.allclose(iteration["parameters_after"], point, atol=1e-14, rtol=0):
            raise ValueError("CPU post-decision parameters disagree with selected move")
        expected_index += 1 + 2 * len(point)
    if (not np.allclose(run["parameters"], point, atol=1e-14, rtol=0)
            or run["final_evaluation"] != current["index"] or run["final_exact_objective"] != current["objective_mean"]
            or run["final_effective_standard_error"] != current["effective_standard_error"]
            or run["final_expected_standard_error"] != current["expected_standard_error"]
            or any(record["accepted"] != (record["index"] in accepted_indexes) for record in records)
            or run["accepted_proposals"] != len(accepted_indexes)
            or run["rejected_proposals"] != 24 * len(point) - len(accepted_indexes)
            or virtual_total != 65_648 or run["virtual_training_preparations"] != virtual_total
            or run["stop_reason"] != "fixed_schedule_exhausted"):
        raise ValueError("CPU final state, decisions or virtual annotation does not replay")
