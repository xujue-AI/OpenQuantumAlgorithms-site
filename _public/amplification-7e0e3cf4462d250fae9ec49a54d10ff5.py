# SPDX-License-Identifier: MIT
"""Original explicit oblivious amplitude amplification for a unitary block.

Pi projects all supplied access ancillas onto0 and acts as identity on unknown
system data. R=I-2Pi is an X/MCZ/X circuit; no dense reflection is constructed.
Starting with A, each chronological R,A†,R,A round and global phase pi gives
A_r=(-A R A† R)^r A. If Pi A Pi=s U exactly, its accepted block is
sin((2r+1)asin(s))U. Our QSP s=1/4 block is approximate, so its numerical
acceptance need not be input-independent or equal to that exact-model formula.

Primary context: Berry et al., arXiv:1312.1414, Lemma3.6; Berry et al.2015,
doi:10.1103/PhysRevLett.114.090502, Eqs13-14. This module is original MIT;
Qiskit is Apache-2.0. Physical faults and continuous-angle synthesis excluded.
"""
from __future__ import annotations

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

if __package__:
    from ..hamiltonian_simulation import decompose
else:
    from hamiltonian_simulation import decompose

MAX_ROUNDS = 4
MAX_GENERATOR_QUBITS = 24
MAX_SOURCE_OPERATIONS = 50000
MAX_PRIMITIVE_GATES = 200000
MAX_DENSE_TOTAL_QUBITS = 8
MAX_STATEVECTOR_QUBITS = 12

_PRIMITIVES = {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "p",
               "rx", "ry", "rz", "u", "u1", "u2", "u3", "cx", "cy", "cz", "swap", "barrier"}
_FORBIDDEN = {"unitary", "hamiltonian", "initialize", "state_preparation", "measure", "reset"}


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in[{minimum},{maximum}]")
    return value


def _check_explicit_unitary_source(circuit: QuantumCircuit):
    """Inspect nested controlled/custom gates without replacing access by a matrix."""
    if circuit.num_parameters:
        raise ValueError("source A parameters must all be bound")
    def check_finite(value, context):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{context} must be a finite real scalar") from None
        if not math.isfinite(numeric):
            raise ValueError(f"{context} must be finite")
    check_finite(circuit.global_phase, "source global phase")
    if circuit.num_clbits:
        raise ValueError("source A must have no classical wires or measurement")
    seen = set()
    def visit(operation):
        if operation.name in _FORBIDDEN:
            raise ValueError(f"opaque or nonunitary source operation:{operation.name}")
        if id(operation) in seen:
            return
        seen.add(id(operation))
        for value in operation.params:
            check_finite(value, f"source parameter of{operation.name}")
        if operation.name in _PRIMITIVES:
            return
        base = getattr(operation, "base_gate", None)
        if base is not None:
            visit(base)
            return
        definition = operation.definition
        if definition is None:
            raise ValueError(f"source operation has no explicit definition:{operation.name}")
        check_finite(definition.global_phase, f"definition global phase of{operation.name}")
        for instruction in definition.data:
            visit(instruction.operation)
    for instruction in circuit.data:
        visit(instruction.operation)


def zero_ancilla_reflection(n_system: int, n_total: int) -> QuantumCircuit:
    """R=I-2(|0...0><0...0|_anc tensor I_data), data wires first.

    Only the ancillas participate. With no ancillas, Pi=I and R=-I is a
    recorded global pi phase. MCZ is H/MCX/H and has a primitive decomposition.
    """
    _integer(n_total, "n_total", 1, MAX_GENERATOR_QUBITS)
    _integer(n_system, "n_system", 1, n_total)
    circuit = QuantumCircuit(n_total, name="R=I-2Pi_anc0")
    ancillas = list(range(n_system, n_total))
    if not ancillas:
        circuit.global_phase = math.pi
    else:
        circuit.x(ancillas)
        target = ancillas[-1]
        if len(ancillas) == 1:
            circuit.z(target)
        else:
            circuit.h(target)
            circuit.mcx(ancillas[:-1], target)
            circuit.h(target)
        circuit.x(ancillas)
    circuit.metadata = {"n_system": n_system, "n_ancillas": len(ancillas),
                        "reflection": "I-2Pi", "projector": "ancillas0 tensor identity_data"}
    return circuit


def build_oblivious_amplification(source: QuantumCircuit, rounds: int = 3, *,
                                  n_system: int | None = None) -> QuantumCircuit:
    """Explicit A/A† and ancilla-reflection generator, bounded independently of simulation.

    The algorithm never reads an input state or Hamiltonian matrix. Source A
    must be an explicit circuit with data first and no measurements. Rounds
    0..4 are permitted; the demonstrated QSP instance uses 0..3. Round 3 success
    is about 0.9613 and remains probabilistic. The 50,000-operation prescreen
    counts top-level instructions; nested definitions are inspected for opaque
    gates and finite parameters, while a separate 200,000-gate cap applies
    after primitive lowering. External nested definitions can make lowering
    costly despite the top-level cap.
    """
    _integer(rounds, "rounds", 0, MAX_ROUNDS)
    _integer(source.num_qubits, "source_qubits", 1, MAX_GENERATOR_QUBITS)
    if n_system is None:
        n_system = (source.metadata or {}).get("n_system")
    _integer(n_system, "n_system", 1, source.num_qubits)
    declared_system = (source.metadata or {}).get("n_system", n_system)
    if declared_system != n_system:
        raise ValueError("n_system disagrees with source metadata")
    _check_explicit_unitary_source(source)
    reflection = zero_ancilla_reflection(n_system, source.num_qubits)
    source_operations = (2*rounds+1)*source.size()+2*rounds*reflection.size()
    if source_operations > MAX_SOURCE_OPERATIONS:
        raise ValueError("top-level source instruction materialization budget exceeded")
    circuit = QuantumCircuit(source.num_qubits, name=f"Oblivious amplification r={rounds}")
    circuit.compose(source, inplace=True)
    if rounds:
        inverse = source.inverse()
        for _ in range(rounds):
            circuit.compose(reflection, inplace=True)
            circuit.compose(inverse, inplace=True)
            circuit.compose(reflection, inplace=True)
            circuit.compose(source, inplace=True)
            circuit.global_phase += math.pi
    circuit.metadata = {
        **(source.metadata or {}), "n_system": n_system,
        "n_amplification_ancillas": source.num_qubits-n_system,
        "encoded_operator": "K p_r(K†K), where K is the source accepted block",
        "source_accepted_branch_normalization": (source.metadata or {}).get("accepted_branch_normalization"),
        "accepted_branch_normalization": None,
        "source_uncontrolled_block_encoding_calls": (source.metadata or {}).get("uncontrolled_block_encoding_calls", 0),
        "source_controlled_block_encoding_calls": (source.metadata or {}).get("controlled_block_encoding_calls", 0),
        "uncontrolled_block_encoding_calls": (2*rounds+1)*(source.metadata or {}).get("uncontrolled_block_encoding_calls", 0),
        "controlled_block_encoding_calls": (2*rounds+1)*(source.metadata or {}).get("controlled_block_encoding_calls", 0),
        "branch_conditioned_projector_phase_calls": (2*rounds+1)*(source.metadata or {}).get("branch_conditioned_projector_phase_calls", 0),
        "oblivious_rounds": rounds, "forward_A_calls": rounds+1, "inverse_A_calls": rounds,
        "total_A_or_inverse_calls": 2*rounds+1, "zero_ancilla_reflections": 2*rounds,
        "reflection_convention": "R=I-2Pi_anc0; Pi_anc0 tensor identity_data",
        "operator_convention": "A_r=(-A R A† R)^r A",
        "round_global_pi_phases": rounds,
        "input_state_knowledge_used_by_amplification": False,
        "amplitude_amplification": "implemented integer-round oblivious path; not deterministic",
        "status": "explicit oblivious amplification; numerical instance checks separate",
    }
    return circuit


def decompose_amplification(circuit: QuantumCircuit) -> QuantumCircuit:
    """Export-bounded primitive circuit; continuous rotations remain unsynthesized."""
    compiled = decompose(circuit)
    if compiled.size() > MAX_PRIMITIVE_GATES:
        raise ValueError(f"primitive export exceeds{MAX_PRIMITIVE_GATES} gates")
    return compiled


def ideal_amplified_amplitude(rounds: int, amplitude: float = .25) -> float:
    """Exact scaled-unitary model only: f_r(s)=sin((2r+1)asin(s))."""
    _integer(rounds, "rounds", 0, MAX_ROUNDS)
    if isinstance(amplitude, bool) or not math.isfinite(amplitude) or not 0 < amplitude <= 1:
        raise ValueError("amplitude must be finite and in(0,1]")
    return math.sin((2*rounds+1)*math.asin(amplitude))


def ideal_oblivious_success(rounds: int, amplitude: float = .25) -> float:
    """Ideal success requires exactly Pi A Pi=sU and unitary U."""
    return ideal_amplified_amplitude(rounds, amplitude)**2


def amplified_block_polynomial(block: np.ndarray, rounds: int) -> np.ndarray:
    """Independent bounded classical reference K p_r(K†K), not a circuit gate.

    p0=1,p1=3-4z,p_(r+1)=(2-4z)p_r-p_(r-1). Repeatedly applying the
    single-round cubic to its own output would give the wrong angle sequence.
    """
    _integer(rounds, "rounds", 0, MAX_ROUNDS)
    k = np.asarray(block, dtype=complex)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] > 2**MAX_DENSE_TOTAL_QUBITS or not np.all(np.isfinite(k)):
        raise ValueError("classical block must be finite, square and bounded")
    identity = np.eye(len(k), dtype=complex)
    if rounds == 0:
        return k.copy()
    z = k.conj().T@k
    previous, current = identity, 3*identity-4*z
    for _ in range(1, rounds):
        previous, current = current, (2*identity-4*z)@current-previous
    return k@current


def postselected_amplified_state(circuit: QuantumCircuit, initial_state: np.ndarray) -> tuple[np.ndarray, float]:
    """Numerical accepted state before final data readout, with a separate statevector cap."""
    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"statevector simulation bounded to{MAX_STATEVECTOR_QUBITS} total qubits")
    n_system = circuit.metadata["n_system"]
    state = np.asarray(initial_state, dtype=complex)
    if state.shape != (2**n_system,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state has invalid dimension or entries")
    if not np.isclose(np.vdot(state, state).real, 1, atol=1e-12, rtol=0):
        raise ValueError("initial_state must be normalized")
    full = np.zeros(2**circuit.num_qubits, dtype=complex)
    full[:len(state)] = state
    branch = Statevector(full).evolve(circuit).data[:len(state)]
    probability = float(np.vdot(branch, branch).real)
    if probability <= 1e-30:
        raise ValueError("accepted branch is numerically zero")
    return branch/math.sqrt(probability), probability


def projected_amplified_block(circuit: QuantumCircuit) -> np.ndarray:
    """Full accepted block from all data-basis inputs, avoiding unused ancilla columns.

    This evolves the actual circuit once per data-basis state. It is a bounded
    classical validation helper, not the generator or a quantum oracle.
    """
    if circuit.num_qubits > MAX_DENSE_TOTAL_QUBITS:
        raise ValueError(f"dense projected-block checks bounded to{MAX_DENSE_TOTAL_QUBITS} total qubits")
    dimension = 2**circuit.metadata["n_system"]
    columns = []
    for j in range(dimension):
        initial = np.zeros(2**circuit.num_qubits, dtype=complex)
        initial[j] = 1
        columns.append(Statevector(initial).evolve(circuit).data[:dimension])
    return np.array(columns).T


def compare_oblivious_amplification(circuit: QuantumCircuit, hamiltonian: np.ndarray,
                                    initial_state: np.ndarray, observable: np.ndarray,
                                    *, ideal_input_amplitude: float = .25) -> dict:
    """Numerical circuit/operator/state check against independent exact evolution.

    Ideal success is conditional on an exact scaled-unitary block, whereas
    minimum/maximum actual acceptance is measured numerically via singular
    values of the implemented finite-H block. Neither is a physical guarantee.
    """
    if circuit.num_qubits > MAX_DENSE_TOTAL_QUBITS:
        raise ValueError(f"dense projected-block validation bounded to{MAX_DENSE_TOTAL_QUBITS} total qubits")
    dimension = 2**circuit.metadata["n_system"]
    h = np.asarray(hamiltonian, dtype=complex)
    obs = np.asarray(observable, dtype=complex)
    if h.shape != (dimension, dimension) or obs.shape != h.shape:
        raise ValueError("classical matrix dimensions invalid")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(obs)):
        raise ValueError("classical matrices must be finite")
    if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0) or not np.allclose(obs, obs.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian and observable must be Hermitian")
    rounds = circuit.metadata["oblivious_rounds"]
    amplitude = ideal_amplified_amplitude(rounds, ideal_input_amplitude)
    target = expm(-1j*circuit.metadata.get("time", 1.0)*h)
    block = projected_amplified_block(circuit)
    state, probability = postselected_amplified_state(circuit, initial_state)
    reference = target@initial_state
    singular_values = np.linalg.svd(block, compute_uv=False)
    exact_observable = float(np.vdot(reference, obs@reference).real)
    conditional_observable = float(np.vdot(state, obs@state).real)
    return {
        "rounds": rounds, "ideal_input_amplitude": ideal_input_amplitude,
        "ideal_scaled_unitary_amplified_amplitude": amplitude,
        "ideal_scaled_unitary_success_probability": amplitude**2,
        "ideal_success_formula_scope": "exactly Pi A Pi=sU with unitaryU; approximate QSP instance tested separately",
        "actual_input_acceptance_probability": probability,
        "actual_minus_ideal_success_probability": probability-amplitude**2,
        "block_error_vs_ideal_amplitude_times_exact_U": float(np.linalg.norm(block-amplitude*target, 2)),
        "decoded_linear_map_operator_error": float(np.linalg.norm(block/amplitude-target, 2)),
        "conditional_state_check_scope": "accepted data state before final system readout",
        "conditional_state_vector_error": float(np.linalg.norm(state-reference)),
        "conditional_state_infidelity": float(max(0, 1-abs(np.vdot(reference, state))**2)),
        "observable_exact": exact_observable, "observable_conditional": conditional_observable,
        "observable_error": abs(conditional_observable-exact_observable),
        "actual_block_singular_values": singular_values.tolist(),
        "finite_H_numerical_minimum_acceptance_probability": float(singular_values[-1]**2),
        "finite_H_numerical_maximum_acceptance_probability": float(singular_values[0]**2),
        "finite_H_numerical_acceptance_input_dependence_span": float(singular_values[0]**2-singular_values[-1]**2),
        "acceptance_certified": False,
    }
