# SPDX-License-Identifier: MIT
"""Independent bounded references for the finite weighted-decay block.

Matrices in this module are classical references, never appended as gates.
Actual recovery compresses every auxiliary wire to zero. A successful small
block check implements neither coherent-correction evolution nor a channel,
continuum detailed balance, Gibbs preparation, mixing or physical resources.
All functions are inert until called; there is no eager source generation.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

if __package__:
    from . import detailed_balance_decay as decay_source
else:
    import detailed_balance_decay as decay_source

MAX_DENSE_SYSTEM_WIRES = 2
MAX_ACTUAL_RECOVERY_WIRES = 8
MAX_REFERENCE_CLOCK_BITS = 6
MAX_REFERENCE_TERMS = 32
MAX_REFERENCE_PRODUCT_STEPS = 128
MAX_PRIMITIVE_OPERATIONS = 100_000
NUMERICAL_TOLERANCE = 1e-10
PRIMITIVE_ARITIES = {
    "h": (1, 0), "x": (1, 0), "s": (1, 0), "sdg": (1, 0),
    "t": (1, 0), "tdg": (1, 0), "ry": (1, 1), "rz": (1, 1),
    "p": (1, 1), "cx": (2, 0),
}


def _integer(value, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _inputs(terms, proposals):
    """Independently parse the bounded dense problem before matrix allocation."""
    if not isinstance(terms, (tuple, list)) or not 1 <= len(terms) <= MAX_REFERENCE_TERMS:
        raise ValueError("dense terms must contain 1 to 32 coefficient/Pauli pairs")
    parsed = []
    for term in terms:
        if not isinstance(term, (tuple, list)) or len(term) != 2:
            raise ValueError("a dense term must be a coefficient/Pauli pair")
        coefficient = _real(term[0], "coefficient")
        label = term[1]
        if not isinstance(label, str) or not label or set(label) - set("IXYZ"):
            raise ValueError("dense labels must be nonempty I/X/Y/Z strings")
        parsed.append((coefficient, label))
    n = _integer(len(parsed[0][1]), "dense system wires", 1, MAX_DENSE_SYSTEM_WIRES)
    if any(len(label) != n for _, label in parsed):
        raise ValueError("dense Hamiltonian labels must have equal width")
    if sum(abs(coefficient) for coefficient, _ in parsed) > 32:
        raise ValueError("dense Hamiltonian coefficient l1 norm exceeds 32")
    if not isinstance(proposals, (tuple, list)) or len(proposals) not in (1, 2, 4):
        raise ValueError("dense proposal count must be 1, 2 or 4")
    if any(not isinstance(label, str) or len(label) != n or set(label) - set("IXYZ")
           for label in proposals):
        raise ValueError("dense proposal labels must match the Hamiltonian width")
    return tuple(parsed), tuple(proposals), n


def independent_mesh(clock_bits: int = 2, beta: float = 1) -> dict:
    """Compute physical f, eta and alpha independently of builder metadata.

    In particular eta is dt*sum(f_j**2), rather than copying the builder's
    stable mass formula. No scale is clipped or silently replaced with one.
    """
    m = _integer(clock_bits, "clock_bits", 1, MAX_REFERENCE_CLOCK_BITS)
    beta = _real(beta, "beta")
    if not 1e-6 <= beta <= 1e6:
        raise ValueError("beta must be in [1e-6, 1e6]")
    size = 1 << m
    signed = np.array([j if j < size // 2 else j - size for j in range(size)], dtype=int)
    sigma = 1 / beta
    dt = math.sqrt(2 * math.pi / size) / (2 * sigma)
    dw = 2 * sigma * math.sqrt(2 * math.pi / size)
    times = signed.astype(float) * dt
    frequencies = signed.astype(float) * dw
    sampled_f = np.exp(-(sigma * times) ** 2) * math.sqrt(sigma * math.sqrt(2 / math.pi))
    squared_sum = float(np.dot(sampled_f, sampled_f))
    eta = dt * squared_sum
    gamma = np.array([math.exp(-max(beta * omega + .5, 0))
                      for omega in frequencies], dtype=float)
    gamma_max = float(gamma.max())
    alpha = eta * gamma_max
    if (not math.isfinite(squared_sum) or squared_sum <= 0
            or not math.isfinite(eta) or eta <= 0
            or not np.all(np.isfinite(gamma)) or np.any(gamma < 0)
            or not math.isfinite(gamma_max) or gamma_max <= 0
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("independent mesh requires finite positive normalization")
    normalized_gamma = gamma / gamma_max
    if np.any(normalized_gamma < 0) or np.any(normalized_gamma > 1):
        raise ValueError("independent normalized rates must be in [0, 1]")
    return {
        "clock_bits": m, "M": size, "beta": beta, "sigma": sigma,
        "signed_indices": signed, "dt": dt, "dw": dw,
        "times": times, "frequencies": frequencies, "sampled_f": sampled_f,
        "physical_squared_sum": squared_sum,
        "amplitudes": sampled_f / math.sqrt(squared_sum),
        "eta": eta, "gamma": gamma, "gamma_max": gamma_max,
        "alpha": alpha, "normalized_gamma": normalized_gamma,
        "dt_dw_residual": float(abs(dt * dw - 2 * math.pi / size)),
    }


def pauli_matrix(label: str) -> np.ndarray:
    """Independent Pauli matrices; character i acts on least-significant qi."""
    if (not isinstance(label, str) or not 1 <= len(label) <= MAX_DENSE_SYSTEM_WIRES
            or set(label) - set("IXYZ")):
        raise ValueError("Pauli matrices are bounded to one/two system wires")
    single = {
        "I": np.array([[1, 0], [0, 1]], complex),
        "X": np.array([[0, 1], [1, 0]], complex),
        "Y": np.array([[0, -1j], [1j, 0]], complex),
        "Z": np.array([[1, 0], [0, -1]], complex),
    }
    matrix = np.ones((1, 1), complex)
    for character in reversed(label):
        matrix = np.kron(matrix, single[character])
    return matrix


def _reference_data(terms, proposals, *, beta, clock_bits, product_steps):
    terms, proposals, n = _inputs(terms, proposals)
    mesh = independent_mesh(clock_bits, beta)
    if product_steps is not None:
        r = _integer(product_steps, "product_steps", 1, MAX_REFERENCE_PRODUCT_STEPS)
    else:
        r = None
    dimension, size, m = 1 << n, mesh["M"], mesh["clock_bits"]
    term_matrices = [(coefficient, pauli_matrix(label)) for coefficient, label in terms]
    proposal_matrices = [pauli_matrix(label) for label in proposals]
    hamiltonian = sum((coefficient * matrix for coefficient, matrix in term_matrices),
                      np.zeros((dimension, dimension), complex))
    intervals = []
    if r is not None:
        # Build each signed controlled interval independently from its
        # mathematical Pauli product. Cache the m matrices, not actual gates.
        for k in range(m):
            interval = mesh["dt"] * (-(1 << k) if k == m - 1 else (1 << k))
            one_step = np.eye(dimension, dtype=complex)
            for coefficient, matrix in (*term_matrices, *reversed(term_matrices)):
                exponential = expm(-1j * coefficient * interval * matrix / (2 * r))
                one_step = exponential @ one_step
            intervals.append(np.linalg.matrix_power(one_step, r))
    conjugates = np.empty((len(proposals), size, dimension, dimension), complex)
    for j, time in enumerate(mesh["times"]):
        if r is None:
            evolution = expm(-1j * hamiltonian * time)
        else:
            evolution = np.eye(dimension, dtype=complex)
            for k, interval_matrix in enumerate(intervals):
                if (j >> k) & 1:
                    evolution = interval_matrix @ evolution
        for a, proposal_matrix in enumerate(proposal_matrices):
            conjugates[a, j] = evolution.conj().T @ proposal_matrix @ evolution
    blocks = np.zeros_like(conjugates)
    normalization = math.sqrt(size * len(proposals))
    # Use physical signed frequencies/times, independently of the circuit's
    # negative QFT and its ordinary computational-index phase convention.
    for a in range(len(proposals)):
        for l, frequency in enumerate(mesh["frequencies"]):
            for j, time in enumerate(mesh["times"]):
                phase = np.exp(-1j * frequency * time)
                blocks[a, l] += phase * mesh["amplitudes"][j] * conjugates[a, j] / normalization
    counters = {
        "system_dimension": dimension, "proposal_frequency_blocks": len(proposals) * size,
        "hamiltonian_term_pauli_matrices": len(terms), "proposal_pauli_matrices": len(proposals),
        "finite_trotter_pauli_expm_evaluations": 0 if r is None else 2 * len(terms) * m,
        "finite_trotter_interval_matrix_power_calls": 0 if r is None else m,
        "finite_trotter_clock_interval_matrix_products": 0 if r is None else m * size // 2,
        "exact_hamiltonian_expm_evaluations": size if r is None else 0,
        "proposal_conjugations": len(proposals) * size,
        "fourier_matrix_summands": len(proposals) * size * size,
    }
    return blocks, hamiltonian, mesh, counters


def reference_blocks(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                     *, beta: float = 1, clock_bits: int = 2,
                     product_steps: int | None = 1) -> np.ndarray:
    """Independent finite Trotter blocks; product_steps=None selects exact H."""
    return _reference_data(terms, proposals, beta=beta, clock_bits=clock_bits,
                           product_steps=product_steps)[0]


def reference_decay(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                    *, beta: float = 1, clock_bits: int = 2,
                    product_steps: int | None = 1) -> dict:
    """Return bounded mathematical arrays and physical normalization/counters."""
    blocks, hamiltonian, mesh, counters = _reference_data(
        terms, proposals, beta=beta, clock_bits=clock_bits, product_steps=product_steps)
    dimension = blocks.shape[-1]
    decay = np.zeros((dimension, dimension), complex)
    isometry = np.zeros_like(decay)
    for a in range(blocks.shape[0]):
        for l, rate in enumerate(mesh["gamma"]):
            product = blocks[a, l].conj().T @ blocks[a, l]
            isometry += product
            decay += mesh["eta"] * rate * product
    return {
        "blocks": blocks, "decay": decay,
        "normalized_decay": decay / mesh["alpha"],
        "hamiltonian": hamiltonian, "mesh": mesh,
        "isometry": isometry, "counters": {
            **counters, "b_dagger_b_matrix_products": blocks.shape[0] * blocks.shape[1]},
    }


def _wire_list(value, expected, name):
    if (not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int)
                                         for x in value) or value != expected):
        raise ValueError(f"actual block requires the exact {name} wire list")


def _validated_circuit_layout(circuit: QuantumCircuit) -> tuple[int, int, int, dict]:
    """Validate typed local bounds and primitive arities before state allocation."""
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("actual block recovery requires a QuantumCircuit")
    metadata = circuit.metadata
    if not isinstance(metadata, dict) or metadata.get("milestone") != "primitive_weighted_decay_block_only":
        raise ValueError("actual block requires the weighted-decay source contract")
    n = _integer(metadata.get("n_system"), "n_system", 1, MAX_DENSE_SYSTEM_WIRES)
    m = _integer(metadata.get("clock_bits"), "clock_bits", 1, MAX_REFERENCE_CLOCK_BITS)
    count = _integer(metadata.get("proposal_count"), "proposal_count", 1, 4)
    p = _integer(metadata.get("proposal_bits"), "proposal_bits", 0, 2)
    total = n + m + p + 2
    if (count not in (1, 2, 4) or p != (count - 1).bit_length()
            or circuit.num_qubits != total or total > MAX_ACTUAL_RECOVERY_WIRES
            or _integer(metadata.get("total_wires"), "total_wires", 1, MAX_ACTUAL_RECOVERY_WIRES) != total
            or circuit.num_clbits):
        raise ValueError("actual block requires the bounded exact source width")
    access_width = n + m + p
    if (_integer(metadata.get("weight_wire"), "weight_wire", 0, MAX_ACTUAL_RECOVERY_WIRES - 1) != access_width
            or _integer(metadata.get("selector_wire"), "selector_wire", 0, MAX_ACTUAL_RECOVERY_WIRES - 1) != access_width + 1
            or metadata.get("selected_auxiliary_state") != "all_zero"
            or metadata.get("selector_zero_only_is_output_contract") is not False):
        raise ValueError("actual block requires all-auxiliary-zero compression and exact flag ordering")
    _wire_list(metadata.get("system_wires"), list(range(n)), "system")
    _wire_list(metadata.get("clock_wires"), list(range(n, n + m)), "clock")
    _wire_list(metadata.get("proposal_wires"), list(range(n + m, access_width)), "proposal")
    _wire_list(metadata.get("auxiliary_wires"), list(range(n, total)), "auxiliary")
    _, declared_proposals, declared_n = _inputs(metadata.get("terms"), metadata.get("proposals"))
    if declared_n != n or len(declared_proposals) != count:
        raise ValueError("actual block problem labels differ from its typed layout")
    beta = _real(metadata.get("beta"), "beta")
    if not 1e-6 <= beta <= 1e6:
        raise ValueError("actual block beta must be in [1e-6, 1e6]")
    _integer(metadata.get("product_steps"), "product_steps", 1, MAX_REFERENCE_PRODUCT_STEPS)
    if min(_real(metadata.get(name), name) for name in ("eta", "gamma_max", "alpha")) <= 0:
        raise ValueError("actual block requires finite positive normalization metadata")
    declared_count = _integer(metadata.get("source_operations"), "source_operations", 1,
                              MAX_PRIMITIVE_OPERATIONS)
    if len(circuit.data) != declared_count or len(circuit.data) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("actual block requires its bounded literal operation count")
    _real(circuit.global_phase, "global phase")
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in PRIMITIVE_ARITIES:
            raise ValueError("actual block contains an unresolved operation")
        wires, parameters = PRIMITIVE_ARITIES[operation.name]
        if (operation.num_qubits != wires or operation.num_clbits or len(instruction.qubits) != wires
                or instruction.clbits or len(operation.params) != parameters):
            raise ValueError("actual block contains an invalid primitive arity")
        addresses = [circuit.find_bit(bit).index for bit in instruction.qubits]
        if len(set(addresses)) != wires or any(not 0 <= address < total for address in addresses):
            raise ValueError("actual block contains invalid primitive wire addresses")
        for parameter in operation.params:
            _real(parameter, "gate angle")
    return n, m, p, metadata


def circuit_block(circuit: QuantumCircuit) -> np.ndarray:
    """Recover <0_all_aux|K|0_all_aux> by two/four basis-column evolutions.

    Only output rows 0..2**n-1 have every high auxiliary bit zero. Retaining
    merely selector-zero rows would be a different map and is not performed.
    """
    n, _, _, _ = _validated_circuit_layout(circuit)
    dimension = 1 << n
    block = np.empty((dimension, dimension), complex)
    for column in range(dimension):
        initial = np.zeros(1 << circuit.num_qubits, complex)
        initial[column] = 1
        evolved = Statevector(initial).evolve(circuit).data
        block[:, column] = evolved[:dimension]
    return block


def matrix_record(matrix) -> dict:
    """JSON-safe real/imaginary representation, without clipping or rounding."""
    values = np.asarray(matrix)
    if not np.all(np.isfinite(values)):
        raise ValueError("recorded matrix must be finite")
    return {"real": values.real.tolist(), "imag": values.imag.tolist()}


def _spectral_correction(hamiltonian, decay, beta):
    """Eq.19 classical diagnostic; this result is never a circuit instruction."""
    energies, vectors = np.linalg.eigh(hamiltonian)
    represented = vectors.conj().T @ decay @ vectors
    differences = energies[:, None] - energies[None, :]
    corrected = .5j * np.tanh(beta * differences / 4) * represented
    return vectors @ corrected @ vectors.conj().T


def decay_check(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                *, beta: float = 1, clock_bits: int = 2, product_steps: int = 1,
                circuit: QuantumCircuit | None = None) -> dict:
    """Compare one actual source against independent finite Trotter decay.

    This function returns raw diagnostics and fixed acceptance predicates.
    Exact-H finite-target differences are reported separately and have no
    tuned acceptance threshold. Passing does not certify the Gibbs task.
    """
    parsed, proposals, n = _inputs(terms, proposals)
    mesh = independent_mesh(clock_bits, beta)
    r = _integer(product_steps, "product_steps", 1, MAX_REFERENCE_PRODUCT_STEPS)
    p = (len(proposals) - 1).bit_length()
    if n + mesh["clock_bits"] + p + 2 > MAX_ACTUAL_RECOVERY_WIRES:
        raise ValueError("the actual block check exceeds eight total wires")
    if circuit is None:
        circuit = decay_source.build_decay_block(parsed, proposals, beta=mesh["beta"],
                                                  clock_bits=mesh["clock_bits"], product_steps=r)
    actual_n, _, actual_p, metadata = _validated_circuit_layout(circuit)
    actual_terms, actual_proposals, _ = _inputs(metadata.get("terms"), metadata.get("proposals"))
    if (actual_n != n or actual_p != p or actual_terms != parsed or actual_proposals != proposals
            or metadata.get("clock_bits") != mesh["clock_bits"]
            or _integer(metadata.get("product_steps"), "product_steps", 1,
                        MAX_REFERENCE_PRODUCT_STEPS) != r
            or _real(metadata.get("beta"), "beta") != mesh["beta"]):
        raise ValueError("actual block metadata describes a different problem")
    builder_eta = _real(metadata.get("eta"), "eta")
    builder_gamma_max = _real(metadata.get("gamma_max"), "gamma_max")
    builder_alpha = _real(metadata.get("alpha"), "alpha")
    if min(builder_eta, builder_gamma_max, builder_alpha) <= 0:
        raise ValueError("actual block metadata requires positive normalization")
    actual = circuit_block(circuit)
    finite = reference_decay(parsed, proposals, beta=mesh["beta"],
                             clock_bits=mesh["clock_bits"], product_steps=r)
    exact = reference_decay(parsed, proposals, beta=mesh["beta"],
                            clock_bits=mesh["clock_bits"], product_steps=None)
    target, exact_target = finite["normalized_decay"], exact["normalized_decay"]
    hamiltonian, decay = finite["hamiltonian"], finite["decay"]
    correction = _spectral_correction(hamiltonian, decay, mesh["beta"])
    eigenvalues = np.linalg.eigvalsh(decay)
    actual_eigenvalues = np.linalg.eigvals(actual)
    dimension = 1 << n
    normalization = {
        "independent_eta": mesh["eta"], "independent_gamma_max": mesh["gamma_max"],
        "independent_alpha": mesh["alpha"], "builder_eta": builder_eta,
        "builder_gamma_max": builder_gamma_max, "builder_alpha": builder_alpha,
        "eta_absolute_error": abs(builder_eta - mesh["eta"]),
        "gamma_max_absolute_error": abs(builder_gamma_max - mesh["gamma_max"]),
        "alpha_absolute_error": abs(builder_alpha - mesh["alpha"]),
        "builder_alpha_product_residual": abs(builder_alpha - builder_eta * builder_gamma_max),
        "dt_dw_residual": mesh["dt_dw_residual"],
        "physical_squared_sum": mesh["physical_squared_sum"],
        "normalized_gamma": mesh["normalized_gamma"].tolist(),
        "scope": "eta=dt*sum(f_j**2) independently; eta does not disappear from physical alpha",
    }
    finite_diagnostics = {
        "isometry_max_entry_error": float(np.max(np.abs(finite["isometry"] - np.eye(dimension)))),
        "decay_hermiticity_residual": float(np.linalg.norm(decay - decay.conj().T, ord="fro")),
        "minimum_decay_eigenvalue": float(eigenvalues.min()),
        "maximum_decay_eigenvalue": float(eigenvalues.max()),
        "alpha_upper_bound_excess": float(eigenvalues.max() - mesh["alpha"]),
        "decay_commutator": float(np.linalg.norm(hamiltonian @ decay - decay @ hamiltonian, ord="fro")),
        "correction_norm": float(np.linalg.norm(correction, ord="fro")),
        "correction_hermiticity_residual": float(np.linalg.norm(correction - correction.conj().T, ord="fro")),
        "coherent_correction_is_classical_only": True,
    }
    actual_diagnostics = {
        "hermiticity_residual": float(np.linalg.norm(actual - actual.conj().T, ord="fro")),
        "spectral_norm": float(np.linalg.norm(actual, ord=2)),
        "eigenvalues": matrix_record(actual_eigenvalues),
        "minimum_eigenvalue_real_part": float(actual_eigenvalues.real.min()),
        "maximum_eigenvalue_real_part": float(actual_eigenvalues.real.max()),
        "maximum_eigenvalue_imaginary_part": float(np.max(np.abs(actual_eigenvalues.imag))),
        "basis_input_all_auxiliary_zero_probabilities":
            [float(np.vdot(actual[:, j], actual[:, j]).real) for j in range(dimension)],
    }
    entry_error = float(np.max(np.abs(actual - target)))
    frobenius_error = float(np.linalg.norm(actual - target, ord="fro"))
    tolerance = NUMERICAL_TOLERANCE
    predicates = {
        "actual_matches_independent_finite_trotter": entry_error <= tolerance,
        "independent_access_isometry": finite_diagnostics["isometry_max_entry_error"] <= tolerance,
        "independent_eta_matches": normalization["eta_absolute_error"] <= tolerance,
        "independent_gamma_max_matches": normalization["gamma_max_absolute_error"] <= tolerance,
        "independent_alpha_matches": normalization["alpha_absolute_error"] <= tolerance,
        "builder_alpha_product_matches": normalization["builder_alpha_product_residual"] <= tolerance,
        "mesh_time_frequency_product": normalization["dt_dw_residual"] <= tolerance,
        "finite_decay_hermitian": finite_diagnostics["decay_hermiticity_residual"] <= tolerance,
        "finite_decay_positive": finite_diagnostics["minimum_decay_eigenvalue"] >= -tolerance,
        "finite_decay_bounded_by_alpha": finite_diagnostics["alpha_upper_bound_excess"] <= tolerance,
        "actual_block_hermitian": actual_diagnostics["hermiticity_residual"] <= tolerance,
        "actual_block_positive": actual_diagnostics["minimum_eigenvalue_real_part"] >= -tolerance,
        "actual_block_contractive": actual_diagnostics["spectral_norm"] <= 1 + tolerance,
        "actual_eigenvalues_real": actual_diagnostics["maximum_eigenvalue_imaginary_part"] <= tolerance,
        "classical_spectral_correction_hermitian": finite_diagnostics["correction_hermiticity_residual"] <= tolerance,
    }
    passed = all(predicates.values())
    return {
        "status": "checked_small_finite_weighted_decay_block" if passed else "failed_small_finite_weighted_decay_block",
        "checks_passed": passed, "acceptance_tolerance": tolerance,
        "acceptance_predicates": predicates,
        "failed_checks": [name for name, value in predicates.items() if not value],
        "actual_vs_finite_trotter_max_entry_error": entry_error,
        "actual_vs_finite_trotter_frobenius_error": frobenius_error,
        "actual_vs_exact_finite_max_entry_error": float(np.max(np.abs(actual - exact_target))),
        "actual_vs_exact_finite_frobenius_error": float(np.linalg.norm(actual - exact_target, ord="fro")),
        "finite_trotter_vs_exact_finite_frobenius_error": float(np.linalg.norm(target - exact_target, ord="fro")),
        "normalization": normalization, "finite_diagnostics": finite_diagnostics,
        "actual_diagnostics": actual_diagnostics,
        "matrices": {
            "actual_selected_block": matrix_record(actual),
            "finite_trotter_decay": matrix_record(decay),
            "finite_trotter_normalized_decay": matrix_record(target),
            "exact_finite_decay": matrix_record(exact["decay"]),
            "exact_finite_normalized_decay": matrix_record(exact_target),
            "hamiltonian": matrix_record(hamiltonian),
            "classical_spectral_correction": matrix_record(correction),
        },
        "counters": {
            "actual_basis_column_statevector_evolutions": dimension,
            "actual_statevector_dimension": 1 << circuit.num_qubits,
            "actual_primitive_instruction_visits": dimension * len(circuit.data),
            "finite_trotter_reference": finite["counters"],
            "exact_hamiltonian_finite_reference": exact["counters"],
            "classical_spectral_correction_eigendecompositions": 1,
        },
        "actual_circuit_metadata": metadata,
        "numerical_checks_performed": True, "all_components_verified": False,
        "new_algorithm_result": False, "coherent_correction_implemented": False,
        "channel_implemented": False, "sampler_implemented": False,
        "finite_mesh_exact_kms": False, "mixing_certificate": False,
        "rotations_synthesized": False, "physical_resources_computed": False,
        "formal_proof_reproduced": False,
        "scope": "actual all-auxiliary-zero block against independent finite-Trotter decay; exact-H finite error separately retained",
        "remaining": ["actual coherent correction access/evolution", "noncommuting weak channel",
                      "uniform finite mesh/Trotter/continuum error bounds", "mixing/output guarantees",
                      "external initialization/readout/repetition/amplification resources",
                      "rotation synthesis and physical resources", "mechanized proof"],
    }
