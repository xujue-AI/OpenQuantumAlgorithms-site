# SPDX-License-Identifier: MIT
"""Transparent bounded QAOA/MaxCut and two-qubit VQE workflows.

Only NumPy and Qiskit core are used. Exact statevector objective calls consume
zero shots; the separate seeded multinomial experiments model ideal readout.
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


def vqe_hamiltonian() -> np.ndarray:
    """H=-Z0-Z1+0.5 X0 X1 in |q1 q0> order; dimensionless energies."""
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    return -np.kron(np.eye(2), z) - np.kron(z, np.eye(2)) + 0.5 * np.kron(x, x)


def vqe_expectation(theta: float) -> float:
    state = Statevector.from_instruction(build_vqe_circuit(theta)).data
    return float(np.vdot(state, vqe_hamiltonian() @ state).real)


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
