# SPDX-License-Identifier: MIT
"""Known-input amplitude amplification of explicit thermal preparation.

A prepares a Bell purification and applies an actual Pauli LCU filter from
all-zero wires. Good Sf=I-2Pi_labels0 leaves system/reference untouched.
Initial S0=I-2|0_all><0_all| reflects about the FULL known input. Each
chronological Sf,A†,S0,A round and global pi implements Q=-A S0 A† Sf.
Unlike ancilla-only oblivious amplification, this preserves the normalized
accepted purification even when the filter is nonunitary. Integer rounds can
overshoot; amplitude amplification does not reduce thermal-polynomial bias.

Primary source: Brassard, Hoyer, Mosca, Tapp, Quantum Amplitude Amplification
and Estimation, quant-ph/0005055, published2002 DOI10.1090/conm/305/05215,
Eq8. Original module MIT; Qiskit Apache-2.0; SciPy BSD-3-Clause.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace

if __package__:
    from . import thermal_filter as tf
else:
    import thermal_filter as tf

MAX_ROUNDS = 4
MAX_GENERATOR_QUBITS = 24
MAX_SOURCE_TOP_LEVEL_INSTRUCTIONS = 50000
MAX_PRIMITIVE_EXPORT_OPERATIONS = 250000
MAX_STATEVECTOR_QUBITS = 12
MAX_OPERATOR_QUBITS = 6
_PRIMITIVES = {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
               "p", "rx", "ry", "rz", "u", "u1", "u2", "u3", "cx", "cy", "cz", "swap", "barrier"}
_FORBIDDEN = {"unitary", "hamiltonian", "initialize", "state_preparation", "measure", "reset"}


def _int(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _finite(value, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{context} must be a finite real scalar") from None
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _explicit_source(source: QuantumCircuit):
    if source.num_clbits or source.num_parameters:
        raise ValueError("coherent source must have no classical wires and all parameters bound")
    _finite(source.global_phase, "source global phase")
    visited = set()
    def inspect(operation):
        if operation.name in _FORBIDDEN:
            raise ValueError(f"opaque or nonunitary source operation: {operation.name}")
        if id(operation) in visited:
            return
        visited.add(id(operation))
        for parameter in operation.params:
            _finite(parameter, f"parameter of {operation.name}")
        if operation.name in _PRIMITIVES:
            return
        base = getattr(operation, "base_gate", None)
        if base is not None:
            inspect(base)
            return
        definition = operation.definition
        if definition is None:
            raise ValueError(f"source operation has no explicit definition: {operation.name}")
        _finite(definition.global_phase, f"definition phase of {operation.name}")
        for instruction in definition.data:
            inspect(instruction.operation)
    for instruction in source.data:
        inspect(instruction.operation)


def zero_subregister_reflection(n_total: int, qubits: Sequence[int]) -> QuantumCircuit:
    """I-2Pi_selected0 using actual X/MCZ/X, identity on unselected wires."""
    _int(n_total, "n_total", 1, MAX_GENERATOR_QUBITS)
    qubits = tuple(qubits)
    for q in qubits:
        _int(q, "reflection qubit", 0, n_total-1)
    if len(set(qubits)) != len(qubits):
        raise ValueError("reflection qubits must be distinct")
    selected = sorted(qubits)
    circuit = QuantumCircuit(n_total, name="Zero-pattern reflection")
    if not selected:
        circuit.global_phase = math.pi  # Empty constraint Pi=I.
    else:
        circuit.x(selected)
        if len(selected) == 1:
            circuit.z(selected[0])
        else:
            target = selected[-1]
            circuit.h(target)
            circuit.mcx(selected[:-1], target)
            circuit.h(target)
        circuit.x(selected)
    circuit.metadata = {"reflection_qubits": selected, "reflection": "I-2Pi_selected0"}
    return circuit


def amplify_known_input(source: QuantumCircuit, success_qubits: Sequence[int],
                         rounds: int = 1) -> QuantumCircuit:
    """Q^r A from known all-zero input; no probability is encoded in a gate.

    Source/inverse and two reflections are actual unitary circuits. The 50,000
    top-level instruction cap does not bound nested lowering cost; the separate
    250,000-operation export cap is checked after primitive lowering.
    """
    _int(source.num_qubits, "source width", 1, MAX_GENERATOR_QUBITS)
    _int(rounds, "rounds", 0, MAX_ROUNDS)
    _explicit_source(source)
    good = zero_subregister_reflection(source.num_qubits, success_qubits)
    initial = zero_subregister_reflection(source.num_qubits, range(source.num_qubits))
    instructions = (2*rounds+1)*source.size()+rounds*(good.size()+initial.size())
    if instructions > MAX_SOURCE_TOP_LEVEL_INSTRUCTIONS:
        raise ValueError("top-level source instruction cap exceeded")
    circuit = QuantumCircuit(source.num_qubits, name=f"Known-input preparation amplification r={rounds}")
    circuit.compose(source, inplace=True)
    if rounds:
        inverse = source.inverse()
        for _ in range(rounds):
            circuit.compose(good, inplace=True)
            circuit.compose(inverse, inplace=True)
            circuit.compose(initial, inplace=True)
            circuit.compose(source, inplace=True)
            circuit.global_phase += math.pi
    circuit.metadata = {
        **(source.metadata or {}), "amplification_rounds": rounds,
        "success_qubits": good.metadata["reflection_qubits"],
        "initial_reflection_qubits": list(range(source.num_qubits)),
        "known_initial_state": "all-zero computational basis state on every wire",
        "forward_A_calls": rounds+1, "inverse_A_calls": rounds,
        "total_A_or_inverse_calls": 2*rounds+1,
        "good_reflection_calls": rounds, "full_initial_reflection_calls": rounds,
        "operator_convention": "Q=-A S0 A† Sf; A_r=Q^r A",
        "round_global_pi_phases": rounds,
        "amplitude_amplification": "implemented for known input; integer rounds with instance-dependent acceptance",
        "prepared_output": "conditional purification before final system readout",
        "thermal_polynomial_bias": "unchanged by ideal known-input amplification",
    }
    return circuit


def build_preparation_amplification(terms=tf.DEFAULT_TERMS, beta: float = 1.25,
                                    degree: int = 6, rounds: int = 1) -> QuantumCircuit:
    source = tf.build_thermal_filter(terms, beta, degree)
    n, n_ref = source.metadata["n_system"], source.metadata["n_reference"]
    return amplify_known_input(source, range(n+n_ref, source.num_qubits), rounds)


def build_preparation_trial(terms=tf.DEFAULT_TERMS, beta: float = 1.25,
                            degree: int = 6, rounds: int = 1, measurement: str = "Z") -> QuantumCircuit:
    """Fresh reset, amplified unitary, label measure, reference reset, basis/readout.

    Every failed attempt executes the full schedule. Reference reset is never
    conditioned on its state; it discards the purification reference.
    """
    if measurement not in {"X", "Y", "Z"}:
        raise ValueError("measurement must be X, Y or Z")
    core = build_preparation_amplification(terms, beta, degree, rounds)
    n, n_ref, m = (core.metadata[k] for k in ("n_system", "n_reference", "n_index"))
    labels = core.metadata["success_qubits"]
    circuit = QuantumCircuit(core.num_qubits, m+n, name=f"Amplified thermal {measurement} trial")
    circuit.reset(range(core.num_qubits))
    circuit.compose(core, inplace=True)
    if labels:
        circuit.measure(labels, list(range(m)))
    circuit.reset(range(n, n+n_ref))
    if measurement == "Y":
        circuit.sdg(range(n))
    if measurement != "Z":
        circuit.h(range(n))
    circuit.measure(range(n), range(m, m+n))
    circuit.metadata = {
        **core.metadata, "measurement_basis": measurement,
        "output": "accepted measured thermal trial; reference discarded without conditioning",
        "initial_reset_instructions": core.num_qubits, "reference_reset_instructions": n_ref,
        "label_measurement_instructions": m, "system_measurement_instructions": n,
        "classical_bit_mapping": "labels in bits0..m-1; system in following bits; accept zero labels",
        "schedule": "all branches run fully; only zero-label measured outcomes retained",
    }
    return circuit


def decompose_preparation_amplification(circuit: QuantumCircuit) -> QuantumCircuit:
    """Preserve arbitrary full-register unitary inputs during coherent lowering."""
    compiled = transpile(circuit, basis_gates=tf.ROTATION_BASIS, optimization_level=1,
                         seed_transpiler=7, qubits_initially_zero=False)
    unresolved = set(compiled.count_ops())-set(tf.ROTATION_BASIS)-{"reset", "measure", "barrier"}
    if unresolved:
        raise ValueError(f"unresolved lowered operations: {unresolved}")
    if compiled.size() > MAX_PRIMITIVE_EXPORT_OPERATIONS:
        raise ValueError("primitive export operation cap exceeded")
    return compiled


def amplification_factor(acceptance_probability: float, rounds: int) -> float:
    """Signed accepted-vector factor sin((2r+1)theta)/sqrt(p), no unitary-block assumption."""
    p = _finite(acceptance_probability, "acceptance probability")
    _int(rounds, "rounds", 0, MAX_ROUNDS)
    if not 0 < p <= 1:
        raise ValueError("source acceptance must be in (0,1]")
    return math.sin((2*rounds+1)*math.asin(math.sqrt(p)))/math.sqrt(p)


def accepted_preparation_branch(circuit: QuantumCircuit) -> tuple[np.ndarray, float]:
    """Actual circuit from all-zero input; success labels are the final register."""
    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError("statevector validation exceeds 12 total wires")
    n, n_ref = circuit.metadata["n_system"], circuit.metadata["n_reference"]
    expected = list(range(n+n_ref, circuit.num_qubits))
    success = circuit.metadata.get("success_qubits", expected)
    if success != expected:
        raise ValueError("thermal branch helper requires labels after system and reference")
    branch = Statevector.from_instruction(circuit).data[:2**(n+n_ref)]
    raw_probability = float(np.vdot(branch, branch).real)
    if raw_probability < 0 or raw_probability > 1+1e-12:
        raise ValueError("invalid numerical branch probability")
    return branch, min(1.0, raw_probability)


def _density_from_branch(branch: np.ndarray, n: int, n_ref: int) -> np.ndarray:
    norm = np.linalg.norm(branch)
    if norm <= 1e-14:
        raise ValueError("accepted state is undefined at zero numerical probability")
    return partial_trace(Statevector(branch/norm), list(range(n, n+n_ref))).data


def _trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.svd(a-b, compute_uv=False).sum()/2)


def preparation_amplification_check(terms=tf.DEFAULT_TERMS, beta: float = 1.25,
                                     degree: int = 6, rounds: int = 1) -> dict:
    """Actual known-input branch, signed direction, reduced state and Gibbs checks."""
    source = tf.build_thermal_filter(terms, beta, degree)
    circuit = build_preparation_amplification(terms, beta, degree, rounds)
    baseline, p = accepted_preparation_branch(source)
    if p <= 1e-28:
        raise ValueError("baseline accepted purification is numerically undefined")
    branch, amplified_p = accepted_preparation_branch(circuit)
    factor = amplification_factor(p, rounds)
    n, n_ref = source.metadata["n_system"], source.metadata["n_reference"]
    baseline_density = _density_from_branch(baseline, n, n_ref)
    poly = tf.matrix_thermal_polynomial(terms, beta, degree)
    polynomial_density = poly@poly.conj().T
    polynomial_density /= np.trace(polynomial_density)
    gibbs = tf.exact_gibbs_state(terms, beta)
    common = {
        "beta": beta, "degree": degree, "rounds": rounds,
        "n_system": n, "n_reference": n_ref, "n_index": source.metadata["n_index"],
        "total_qubits": source.num_qubits, "baseline_acceptance_probability": p,
        "signed_amplification_factor": factor,
        "expected_global_sign": -1 if factor < 0 else 1,
        "model_success_probability_from_actual_baseline_p": min(1.0, factor*factor*p),
        "actual_acceptance_probability": amplified_p,
        "raw_signed_accepted_vector_error": float(np.linalg.norm(branch-factor*baseline)),
        "baseline_gibbs_trace_distance": _trace_distance(baseline_density, gibbs),
        "conditional_state_scope": "before final system readout; reference traced out without conditioning",
        "conditional_state_defined": amplified_p > 1e-28,
        "source_forward_calls": rounds+1, "source_inverse_calls": rounds,
        "total_source_calls": 2*rounds+1,
        "model_scope": "exact known-input amplitude-amplification identity; floating circuit check, no formal certificate",
        "acceptance_certified": False,
    }
    if amplified_p <= 1e-28:
        return {**common, "phase_invariant_purification_error": None, "signed_normalized_purification_error": None,
                "reduced_state_trace_distance_from_baseline": None, "gibbs_trace_distance": None,
                "actual_energy": None, "exact_gibbs_energy": None, "energy_error": None}
    density = _density_from_branch(branch, n, n_ref)
    baseline_normalized = baseline/np.linalg.norm(baseline)
    normalized = branch/np.linalg.norm(branch)
    overlap = np.vdot(baseline_normalized, normalized)
    phase = overlap/abs(overlap) if abs(overlap) > 1e-14 else 1+0j
    h = tf.dense_hamiltonian(terms, beta)
    actual_energy = float(np.trace(density@h).real)
    exact_energy = float(np.trace(gibbs@h).real)
    return {
        **common,
        "phase_invariant_purification_error": float(np.linalg.norm(normalized-phase*baseline_normalized)),
        "signed_normalized_purification_error": float(np.linalg.norm(normalized-common["expected_global_sign"]*baseline_normalized)),
        "reduced_state_trace_distance_from_baseline": _trace_distance(density, baseline_density),
        "reduced_state_trace_distance_from_polynomial": _trace_distance(density, polynomial_density),
        "gibbs_trace_distance": _trace_distance(density, gibbs),
        "actual_energy": actual_energy, "exact_gibbs_energy": exact_energy,
        "energy_error": abs(actual_energy-exact_energy),
        "actual_density_real": density.real.tolist(), "actual_density_imag": density.imag.tolist(),
        "round_selection_scope": "chosen fixed integer rounds; baseline p acquired only by bounded classical check, no general probability-learning routine",
    }


def synthesis_conditioning_proposal(acceptance_probability: float, rounds: int,
                                    source_operator_error: float = 1e-5,
                                    initial_reflection_error: float = 1e-5,
                                    good_reflection_error: float = 1e-5) -> dict:
    """Hypothetical telescoping bound and conditioning, requiring actual certificates."""
    p = _finite(acceptance_probability, "acceptance probability")
    _int(rounds, "rounds", 0, MAX_ROUNDS)
    allowances = [_finite(value, "coherent operator allowance") for value in
                  (source_operator_error, initial_reflection_error, good_reflection_error)]
    if any(value < 0 for value in allowances) or not 0 < p <= 1:
        raise ValueError("need nonnegative coherent allowances and 0<p<=1")
    eta = (2*rounds+1)*allowances[0]+rounds*(allowances[1]+allowances[2])
    if eta >= math.sqrt(p):
        raise ValueError("hypothetical coherent error is too large for conditioning bound")
    distance = min(1.0, 2*eta/math.sqrt(p))
    return {"source_operator_error_allowance": allowances[0],
            "full_initial_reflection_error_allowance": allowances[1],
            "good_reflection_error_allowance": allowances[2],
            "amplified_coherent_operator_error_allowance": eta,
            "conditional_state_trace_distance_bound": distance,
            "single_pauli_mean_bias_bound": min(2.0, 2*distance),
            "perturbed_acceptance_lower_bound": (math.sqrt(p)-eta)**2,
            "status": "hypothetical only; no synthesis certificates implemented",
            "assumption": "certified operator-norm errors for A, A inverse, S0 and Sf; no bound assigned to reset or readout faults",
            "error_source_scope": "source thermal polynomial bias remains separate"}


def thermal_family_terms(n: int) -> tuple[tuple[float, str], ...]:
    """Structured sum_i(.6Zi+.4Xi), used only for bounded generation cost rows."""
    _int(n, "family n", 1, 3)
    values = []
    for q in range(n):
        for coefficient, axis in ((.6, "Z"), (.4, "X")):
            label = ["I"]*n
            label[n-1-q] = axis
            values.append((coefficient, "".join(label)))
    return tuple(values)


def reflection_scaling_row(n: int, beta: float = 1.25, degree: int = 6) -> dict:
    """Generate actual Pauli access and lower both reflections; no state validation claim."""
    source = tf.build_thermal_filter(thermal_family_terms(n), beta, degree)
    labels = list(range(2*n, source.num_qubits))
    coherent = decompose_preparation_amplification(source)
    trial_x = decompose_preparation_amplification(build_preparation_trial(thermal_family_terms(n), beta, degree, 0, "X"))
    trial_z = decompose_preparation_amplification(build_preparation_trial(thermal_family_terms(n), beta, degree, 0, "Z"))
    s0 = decompose_preparation_amplification(zero_subregister_reflection(source.num_qubits, range(source.num_qubits)))
    sf = decompose_preparation_amplification(zero_subregister_reflection(source.num_qubits, labels))
    return {"n_system": n, "n_reference": n, "label_qubits": len(labels),
            "total_qubits": source.num_qubits, "aggregated_pauli_terms": source.metadata["aggregated_terms"],
            "source_top_level_instructions": source.size(),
            "source_coherent_primitive_gates": coherent.size(), "source_coherent_dependency_depth": coherent.depth(),
            "unamplified_full_X_trial_primitive_operations": trial_x.size(),
            "unamplified_full_Z_trial_primitive_operations": trial_z.size(),
            "unamplified_full_trial_initial_resets": source.num_qubits,
            "unamplified_full_trial_reference_resets": n,
            "unamplified_full_trial_label_measurements": len(labels),
            "unamplified_full_trial_system_measurements": n,
            "energy_coefficient_abs_sum": float(n),
            "accepted_energy_shot_workload": "n>1 unimplemented; one-qubit helper not extrapolated to fixed absolute energy error",
            "precision_scaling_scope": "fixed beta/degree, generation only; no equal-error thermal or fault-tolerant comparison",
            "full_initial_reflection_primitive_gates": s0.size(), "full_initial_reflection_depth": s0.depth(),
            "good_label_reflection_primitive_gates": sf.size(), "good_label_reflection_depth": sf.depth(),
            "status": "access/reflection generation and lowering only; family-row state simulation unperformed",
            "classical_structure": "independent local X/Z spins; classically factorized Gibbs reference is available, no advantage benchmark"}


def ancilla_only_distortion_check(terms=tf.DEFAULT_TERMS, beta: float = 1.25,
                                  degree: int = 6, rounds: int = 1) -> dict:
    """Bounded contrast circuit: replace FULL S0 by labels-only Sf, changing the target."""
    source = tf.build_thermal_filter(terms, beta, degree)
    _int(rounds, "rounds", 0, MAX_ROUNDS)
    n, n_ref = source.metadata["n_system"], source.metadata["n_reference"]
    labels = list(range(n+n_ref, source.num_qubits))
    sf = zero_subregister_reflection(source.num_qubits, labels)
    circuit = source.copy()
    for _ in range(rounds):
        circuit.compose(sf, inplace=True)
        circuit.compose(source.inverse(), inplace=True)
        circuit.compose(sf, inplace=True)
        circuit.compose(source, inplace=True)
        circuit.global_phase += math.pi
    baseline, _ = accepted_preparation_branch(source)
    branch, p = accepted_preparation_branch(circuit)
    density = _density_from_branch(branch, n, n_ref)
    baseline_density = _density_from_branch(baseline, n, n_ref)
    return {"rounds": rounds, "actual_acceptance_probability": p,
            "trace_distance_from_original_thermal_polynomial_state": _trace_distance(density, baseline_density),
            "gibbs_trace_distance": _trace_distance(density, tf.exact_gibbs_state(terms, beta)),
            "scope": "ancilla-only contrast demonstrates distortion of a nonunitary filter; not the known-input preparation algorithm"}
