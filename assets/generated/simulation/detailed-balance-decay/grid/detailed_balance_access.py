# SPDX-License-Identifier: MIT
"""Original bounded Gaussian operator-Fourier access, not a Gibbs sampler.

All circuit instructions are explicit one-qubit gates and CX. Dense matrices
are independent small references only. Finite mesh and Trotter access do not
preserve exact KMS balance. No coherent correction/channel is a circuit here.
Pauli label character i acts on qi (q0 is the least significant state bit).
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from scipy.special import log_ndtr

MAX_SOURCE_SYSTEM_WIRES = 16
MAX_CLOCK_BITS = 6
MAX_TERMS = 32
MAX_PRODUCT_STEPS = 128
MAX_PRIMITIVE_OPERATIONS = 100_000
MAX_DENSE_SYSTEM_WIRES = 2
MAX_SIMULATION_WIRES = 10
PRIMITIVES = frozenset({"h", "x", "s", "sdg", "t", "tdg", "ry", "rz", "p", "cx"})
PAULIS = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex),
          "Y": np.array([[0, -1j], [1j, 0]], complex), "Z": np.diag([1, -1]).astype(complex)}


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _labels(terms: Sequence[tuple[float, str]], proposals: Sequence[str]):
    if not isinstance(terms, (tuple, list)) or not 1 <= len(terms) <= MAX_TERMS:
        raise ValueError(f"terms must contain 1 to {MAX_TERMS} weighted Pauli labels")
    parsed = []
    for term in terms:
        if not isinstance(term, (tuple, list)) or len(term) != 2:
            raise ValueError("each term must be a coefficient and Pauli label")
        coefficient = _real(term[0], "coefficient")
        label = term[1]
        if not isinstance(label, str) or not label or set(label) - set(PAULIS):
            raise ValueError("Pauli labels use nonempty I/X/Y/Z strings")
        parsed.append((coefficient, label))
    n = len(parsed[0][1])
    _integer(n, "system wires", 1, MAX_SOURCE_SYSTEM_WIRES)
    if any(len(label) != n for _, label in parsed):
        raise ValueError("all Hamiltonian labels must have the same width")
    if sum(abs(c) for c, _ in parsed) > 32:
        raise ValueError("Hamiltonian coefficient l1 norm exceeds 32")
    if not isinstance(proposals, (tuple, list)) or len(proposals) not in (1, 2, 4):
        raise ValueError("proposal count must be 1, 2 or 4")
    if any(not isinstance(p, str) or len(p) != n or set(p) - set(PAULIS) for p in proposals):
        raise ValueError("proposal Pauli labels must match the Hamiltonian width")
    return tuple(parsed), tuple(proposals), n


def gaussian_mesh(clock_bits: int = 3, beta: float = 1.0) -> dict:
    """Signed paper mesh, with stable normalized weights and explicit eta."""
    m = _integer(clock_bits, "clock_bits", 1, MAX_CLOCK_BITS)
    beta = _real(beta, "beta")
    if not 1e-6 <= beta <= 1e6:
        raise ValueError("beta must be in the finite source domain [1e-6, 1e6]")
    size = 1 << m
    indices = np.arange(size, dtype=int)
    signed = np.where(indices < size // 2, indices, indices - size)
    dt = beta * .5 * math.sqrt(2 * math.pi / size)
    dw = (2 / beta) * math.sqrt(2 * math.pi / size)
    weights = np.exp(-math.pi * signed.astype(float) ** 2 / size)
    mass = float(weights.sum())
    physical_squared_sum = mass * math.sqrt(2 / math.pi) / beta
    eta = mass / math.sqrt(size)
    return {"clock_bits": m, "M": size, "beta": beta, "sigma": 1 / beta,
            "signed_indices": signed, "dt": dt, "dw": dw,
            "times": dt * signed, "frequencies": dw * signed,
            "sampled_f": np.sqrt(weights * math.sqrt(2 / math.pi) / beta),
            "physical_squared_sum": physical_squared_sum,
            "amplitudes": np.sqrt(weights / mass), "eta": eta,
            "gamma": np.exp(-np.maximum(beta * dw * signed + .5, 0)),
            "normalization": "L_quad[a,l]=sqrt(eta*gamma[l])*B[a,l]; eta is not replaced by one"}


def _guard(circuit: QuantumCircuit) -> None:
    if len(circuit.data) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("source materialization exceeds 100000 primitive operations")


def _multiplexed_ry(circuit: QuantumCircuit, controls: list[int], target: int,
                    angles: Sequence[float]) -> None:
    """Exact real uniformly controlled rotation, including Gray-cycle closure."""
    count = 1 << len(controls)
    if len(angles) != count:
        raise ValueError("multiplexor angle count differs from its controls")
    if not controls:
        circuit.ry(float(angles[0]), target)
        _guard(circuit)
        return
    gray = [j ^ (j >> 1) for j in range(count)]
    for j, word in enumerate(gray):
        alpha = sum((-1 if (k & word).bit_count() & 1 else 1) * theta
                    for k, theta in enumerate(angles)) / count
        circuit.ry(float(alpha), target)
        changed = word ^ gray[(j + 1) % count]
        circuit.cx(controls[changed.bit_length() - 1], target)
        _guard(circuit)


def positive_prepare(amplitudes: Sequence[float]) -> QuantumCircuit:
    """Primitive positive amplitude loading; no StatePreparation/Initialize."""
    values = np.asarray(amplitudes)
    if values.ndim != 1 or values.dtype.kind not in "fiu" or len(values) < 2:
        raise ValueError("amplitudes must be a real power-of-two vector")
    if len(values) & (len(values) - 1):
        raise ValueError("amplitude length must be a power of two")
    m = _integer((len(values) - 1).bit_length(), "clock_bits", 1, MAX_CLOCK_BITS)
    values = values.astype(float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("amplitudes must be finite and nonnegative")
    if not math.isclose(float(values @ values), 1, abs_tol=1e-12, rel_tol=0):
        raise ValueError("amplitudes must have unit squared norm")
    probabilities = values ** 2
    circuit = QuantumCircuit(m, name="Gaussian PREPARE")
    for target in range(m - 1, -1, -1):
        controls = list(range(target + 1, m))
        angles = []
        width = 1 << target
        for prefix in range(1 << len(controls)):
            start = prefix << (target + 1)
            left = float(probabilities[start:start + width].sum())
            right = float(probabilities[start + width:start + 2 * width].sum())
            angles.append(2 * math.atan2(math.sqrt(right), math.sqrt(left))
                          if left + right > 0 else 0.0)
        _multiplexed_ry(circuit, controls, target, angles)
    return circuit


def _cp(circuit: QuantumCircuit, angle: float, control: int, target: int) -> None:
    circuit.p(angle / 2, control)
    circuit.cx(control, target)
    circuit.p(-angle / 2, target)
    circuit.cx(control, target)
    circuit.p(angle / 2, target)


def negative_dft(clock_bits: int) -> QuantumCircuit:
    """Primitive negative-sign DFT in computational-index order."""
    m = _integer(clock_bits, "clock_bits", 1, MAX_CLOCK_BITS)
    positive = QuantumCircuit(m, name="DFT positive")
    for j in range(m - 1, -1, -1):
        positive.h(j)
        for k in range(j - 1, -1, -1):
            _cp(positive, math.pi / (1 << (j - k)), k, j)
    for j in range(m // 2):
        other = m - 1 - j
        positive.cx(j, other)
        positive.cx(other, j)
        positive.cx(j, other)
    inverse = positive.inverse()
    inverse.name = "DFT negative"
    return inverse


def _controlled_pauli_rotation(circuit: QuantumCircuit, label: str,
                               angle: float, control: int) -> None:
    active = [q for q, p in enumerate(label) if p != "I"]
    if not active:
        # Controlled exp(-i angle I/2) is a relative clock phase, not an
        # omitted global phase. Its inverse cancels inside conjugated access.
        circuit.p(-angle / 2, control)
        _guard(circuit)
        return
    for q in active:
        if label[q] == "X":
            circuit.h(q)
        elif label[q] == "Y":
            circuit.sdg(q)
            circuit.h(q)
    pivot = active[-1]
    for q in active[:-1]:
        circuit.cx(q, pivot)
    circuit.rz(angle / 2, pivot)
    circuit.cx(control, pivot)
    circuit.rz(-angle / 2, pivot)
    circuit.cx(control, pivot)
    for q in reversed(active[:-1]):
        circuit.cx(q, pivot)
    for q in reversed(active):
        if label[q] == "X":
            circuit.h(q)
        elif label[q] == "Y":
            circuit.h(q)
            circuit.s(q)
    _guard(circuit)


def _toffoli(circuit: QuantumCircuit, a: int, b: int, target: int) -> None:
    circuit.h(target)
    circuit.cx(b, target)
    circuit.tdg(target)
    circuit.cx(a, target)
    circuit.t(target)
    circuit.cx(b, target)
    circuit.tdg(target)
    circuit.cx(a, target)
    circuit.t(b)
    circuit.t(target)
    circuit.h(target)
    circuit.cx(a, b)
    circuit.t(a)
    circuit.tdg(b)
    circuit.cx(a, b)


def _controlled_x(circuit: QuantumCircuit, controls: list[int], target: int) -> None:
    if not controls:
        circuit.x(target)
    elif len(controls) == 1:
        circuit.cx(controls[0], target)
    elif len(controls) == 2:
        _toffoli(circuit, *controls, target)
    else:
        raise ValueError("the declared proposal cap permits at most two SELECT controls")


def _select(circuit: QuantumCircuit, proposals: tuple[str, ...], controls: list[int]) -> None:
    for a, label in enumerate(proposals):
        zeros = [q for k, q in enumerate(controls) if not (a >> k) & 1]
        for q in zeros:
            circuit.x(q)
        for q, p in enumerate(label):
            if p == "I":
                continue
            if p == "Y":
                circuit.sdg(q)
            elif p == "Z":
                circuit.h(q)
            _controlled_x(circuit, controls, q)
            if p == "Y":
                circuit.s(q)
            elif p == "Z":
                circuit.h(q)
            _guard(circuit)
        for q in zeros:
            circuit.x(q)
    _guard(circuit)


def build_gaussian_access(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                          *, beta: float = 1, clock_bits: int = 3,
                          product_steps: int = 1) -> QuantumCircuit:
    """Generate guarded primitive filtered-jump access with arbitrary rotations.

    Unknown system input is preserved by the access contract; only clock and
    proposal inputs are zero. Its output labels are not postselection flags.
    """
    terms, proposals, n = _labels(terms, proposals)
    r = _integer(product_steps, "product_steps", 1, MAX_PRODUCT_STEPS)
    mesh = gaussian_mesh(clock_bits, beta)
    m = mesh["clock_bits"]
    p = (len(proposals) - 1).bit_length()
    clock = list(range(n, n + m))
    proposal = list(range(n + m, n + m + p))
    # This explicit upper bound is checked before creating the circuit.
    bound = (4 * mesh["M"] + 5 * m * m + 120 * len(proposals) * n
             + 2 * m * r * sum(12 * len(label) + 12 for _, label in terms))
    if bound > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("preflight source materialization exceeds 100000 primitive operations")
    circuit = QuantumCircuit(n + m + p, name="Gaussian filtered-jump access")
    for q in proposal:
        circuit.h(q)
    preparation = positive_prepare(mesh["amplitudes"])
    circuit.compose(preparation, clock, inplace=True)
    evolution = QuantumCircuit(n + m + p, name="Signed Hamiltonian SELECT")
    for k, q in enumerate(clock):
        interval = mesh["dt"] * (-(1 << k) if k == m - 1 else 1 << k)
        for _ in range(r):
            for coefficient, label in (*terms, *reversed(terms)):
                _controlled_pauli_rotation(evolution, label, coefficient * interval / r, q)
    circuit.compose(evolution, inplace=True)
    before_select = len(circuit.data)
    _select(circuit, proposals, proposal)
    select_count = len(circuit.data) - before_select
    circuit.compose(evolution.inverse(), inplace=True)
    dft = negative_dft(m)
    circuit.compose(dft, clock, inplace=True)
    _guard(circuit)
    if set(circuit.count_ops()) - PRIMITIVES:
        raise AssertionError("access contains unresolved composite gates")
    if any(not math.isfinite(float(angle)) for instruction in circuit.data
           for angle in instruction.operation.params):
        raise ValueError("source contains nonfinite rotation angles")
    circuit.metadata = {
        "milestone": "primitive_gaussian_filtered_jump_access_only",
        "n_system": n, "clock_bits": m, "proposal_bits": p,
        "proposal_count": len(proposals), "product_steps": r, "beta": mesh["beta"],
        "pauli_label_order": "character i acts on qi; q0 is least significant",
        "proposal_normalization": "A_a=P_a/sqrt(J); sum A_a^dagger A_a=I",
        "time_mesh": mesh["times"].tolist(), "frequency_mesh": mesh["frequencies"].tolist(),
        "eta": mesh["eta"], "physical_squared_sum": mesh["physical_squared_sum"],
        "dt": mesh["dt"], "dw": mesh["dw"],
        "gaussian_prepare_instructions": len(preparation.data),
        "classical_preprocessing": "explicit Gaussian/prefix masses and naive Gray-Walsh angle transform: O(M^2) time, O(M) storage; wall time is not quantum runtime",
        "select_instructions": select_count,
        "signed_evolution_instructions_each_direction": len(evolution.data),
        "dft_instructions": len(dft.data),
        "requested_signed_interval_magnitude_both_directions": 2 * (mesh["M"] - 1) * mesh["dt"],
        "source_operations": len(circuit.data), "preflight_upper_bound": bound,
        "initialization": "system is caller supplied; clock/proposal start in zero; reset time not included",
        "postselection": "none in this access; output is coherent proposal/frequency labels",
        "finite_mesh_exact_kms": False, "channel_implemented": False,
        "coherent_correction_implemented": False, "sampler_implemented": False,
        "rotations_synthesized": False, "formal_proof_reproduced": False,
        "physical_resources_computed": False,
    }
    return circuit


def pauli_matrix(label: str) -> np.ndarray:
    if not isinstance(label, str) or not 1 <= len(label) <= MAX_DENSE_SYSTEM_WIRES or set(label) - set(PAULIS):
        raise ValueError("dense Pauli references are bounded to one/two system wires")
    result = np.ones((1, 1), complex)
    for p in reversed(label):
        result = np.kron(result, PAULIS[p])
    return result


def _dense_inputs(terms, proposals):
    terms, proposals, n = _labels(terms, proposals)
    if n > MAX_DENSE_SYSTEM_WIRES:
        raise ValueError("dense references are bounded to two system wires")
    h = sum((c * pauli_matrix(label) for c, label in terms), np.zeros((1 << n, 1 << n), complex))
    return terms, proposals, n, h


def _matrix_interval(terms, interval: float, steps: int) -> np.ndarray:
    """Independent mathematical product, with no Qiskit circuit inspection."""
    dim = 1 << len(terms[0][1])
    one = np.eye(dim, dtype=complex)
    for c, label in (*terms, *reversed(terms)):
        one = expm(-1j * c * interval * pauli_matrix(label) / (2 * steps)) @ one
    return np.linalg.matrix_power(one, steps)


def reference_blocks(terms, proposals, *, beta=1, clock_bits=3,
                     product_steps: int | None = None) -> np.ndarray:
    """Finite exact-H or defined-Trotter reference; matrices are never gates."""
    terms, proposals, n, h = _dense_inputs(terms, proposals)
    mesh = gaussian_mesh(clock_bits, beta)
    m, size, dim = mesh["clock_bits"], mesh["M"], 1 << n
    if product_steps is not None:
        _integer(product_steps, "product_steps", 1, MAX_PRODUCT_STEPS)
    conjugates = np.empty((len(proposals), size, dim, dim), complex)
    for j, time in enumerate(mesh["times"]):
        if product_steps is None:
            v = expm(-1j * h * time)
        else:
            v = np.eye(dim, dtype=complex)
            for k in range(m):
                if (j >> k) & 1:
                    interval = mesh["dt"] * (-(1 << k) if k == m - 1 else 1 << k)
                    v = _matrix_interval(terms, interval, product_steps) @ v
        for a, label in enumerate(proposals):
            conjugates[a, j] = v.conj().T @ pauli_matrix(label) @ v
    phase = np.exp(-2j * math.pi * np.outer(np.arange(size), np.arange(size)) / size)
    return np.einsum("lj,j,ajxy->alxy", phase, mesh["amplitudes"], conjugates) / math.sqrt(size * len(proposals))


def circuit_blocks(circuit: QuantumCircuit) -> np.ndarray:
    """Recover all access blocks by bounded actual primitive statevectors."""
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("actual circuit checks require a QuantumCircuit")
    metadata = circuit.metadata or {}
    if metadata.get("milestone") != "primitive_gaussian_filtered_jump_access_only":
        raise ValueError("circuit does not carry this access contract")
    try:
        n = _integer(metadata.get("n_system"), "n_system", 1, MAX_DENSE_SYSTEM_WIRES)
        m = _integer(metadata.get("clock_bits"), "clock_bits", 1, MAX_CLOCK_BITS)
        jumps = _integer(metadata.get("proposal_count"), "proposal_count", 1, 4)
        p = _integer(metadata.get("proposal_bits"), "proposal_bits", 0, 2)
    except ValueError as error:
        raise ValueError("actual circuit checks require typed bounded system/clock/proposal metadata") from error
    if (jumps not in (1, 2, 4) or p != (jumps - 1).bit_length()
            or circuit.num_qubits != n + m + p or circuit.num_qubits > MAX_SIMULATION_WIRES
            or circuit.num_clbits or set(circuit.count_ops()) - PRIMITIVES):
        raise ValueError("actual circuit checks require the exact primitive access width and proposal contract")
    try:
        finite_angles = all(math.isfinite(float(angle)) for instruction in circuit.data
                            for angle in instruction.operation.params)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("actual circuit checks require finite numeric gate angles") from error
    if not finite_angles:
        raise ValueError("actual circuit checks require finite numeric gate angles")
    _guard(circuit)
    dim, size = 1 << n, 1 << m
    blocks = np.empty((jumps, size, dim, dim), complex)
    for column in range(dim):
        initial = np.zeros(1 << circuit.num_qubits, complex)
        initial[column] = 1
        actual = Statevector(initial).evolve(circuit).data
        blocks[:, :, :, column] = actual.reshape(jumps, size, dim)
    return blocks


def spectral_correction(h: np.ndarray, decay: np.ndarray, beta: float) -> np.ndarray:
    """Hermitian paper eq19 reference only; it is not an implemented C gate."""
    beta = gaussian_mesh(1, beta)["beta"]
    h, decay = np.asarray(h), np.asarray(decay)
    if (h.ndim != 2 or h.shape not in ((2, 2), (4, 4)) or decay.shape != h.shape
            or not np.all(np.isfinite(h)) or not np.all(np.isfinite(decay))
            or not np.allclose(h, h.conj().T, atol=1e-12, rtol=0)
            or not np.allclose(decay, decay.conj().T, atol=1e-12, rtol=0)):
        raise ValueError("spectral diagnostics require finite Hermitian one/two-spin matrices")
    values, vectors = np.linalg.eigh(h)
    represented = vectors.conj().T @ decay @ vectors
    differences = values[:, None] - values[None, :]
    c = .5j * np.tanh(beta * differences / 4) * represented
    return vectors @ c @ vectors.conj().T


def finite_generator(terms, proposals, blocks: np.ndarray, *, beta=1, clock_bits=3):
    """Classical finite-jump Lindbladian diagnostics, with spectral C only."""
    _, proposals, n, h = _dense_inputs(terms, proposals)
    mesh = gaussian_mesh(clock_bits, beta)
    blocks = np.asarray(blocks)
    if blocks.shape != (len(proposals), mesh["M"], 1 << n, 1 << n):
        raise ValueError("block shape differs from the finite access contract")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("blocks must be finite")
    jumps = blocks * np.sqrt(mesh["eta"] * mesh["gamma"])[None, :, None, None]
    decay = np.einsum("alji,aljk->ik", jumps.conj(), jumps)
    correction = spectral_correction(h, decay, beta)
    values = np.linalg.eigvalsh(h)
    gibbs = expm(-beta * (h - values[0] * np.eye(1 << n)))
    gibbs /= np.trace(gibbs)
    transition = sum(
        (l @ gibbs @ l.conj().T for l in jumps.reshape(-1, 1 << n, 1 << n)),
        np.zeros_like(h))
    action = transition - .5 * (decay @ gibbs + gibbs @ decay) - 1j * (correction @ gibbs - gibbs @ correction)
    return {"decay": decay, "correction": correction, "gibbs": gibbs,
            "gibbs_stationarity_residual": float(np.linalg.norm(action, ord="fro")),
            "decay_commutator": float(np.linalg.norm(h @ decay - decay @ h, ord="fro")),
            "correction_norm": float(np.linalg.norm(correction, ord="fro")),
            "correction_hermiticity_residual": float(np.linalg.norm(correction - correction.conj().T)),
            "minimum_decay_eigenvalue": float(np.linalg.eigvalsh(decay).min()),
            "interpretation": "finite mesh spectral C does not repair transition KMS error; no sampler/channel is implemented"}


def continuum_generator_reference(terms, proposals, *, beta=1) -> dict:
    """Analytically integrated Gaussian/Metropolis reference on <=2 wires."""
    _, proposals, n, h = _dense_inputs(terms, proposals)
    beta = gaussian_mesh(1, beta)["beta"]
    values, vectors = np.linalg.eigh(h)
    dim = 1 << n
    difference = values[:, None] - values[None, :]
    nu = difference[:, :, None, None]
    mu = difference[None, None, :, :]
    x = beta * (nu + mu) / 2
    log_rate = np.logaddexp(log_ndtr(-x - .5), -x + log_ndtr(x - .5))
    kernel = np.exp(-beta * beta * (nu - mu) ** 2 / 8 + log_rate)
    decay = np.zeros((dim, dim), complex)
    transition = np.zeros((dim, dim), complex)
    probabilities = np.exp(-beta * (values - values[0]))
    probabilities /= probabilities.sum()
    rho = np.diag(probabilities).astype(complex)
    for label in proposals:
        a = vectors.conj().T @ pauli_matrix(label) @ vectors / math.sqrt(len(proposals))
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        transition[i, k] += kernel[i, j, k, l] * a[i, j] * rho[j, l] * a[k, l].conjugate()
                        if i == k:
                            decay[j, l] += kernel[i, j, k, l] * a[i, j].conjugate() * a[k, l]
    correction = .5j * np.tanh(beta * difference / 4) * decay
    action = transition - .5 * (decay @ rho + rho @ decay) - 1j * (correction @ rho - rho @ correction)
    return {"gibbs_stationarity_residual": float(np.linalg.norm(action, ord="fro")),
            "correction_hermiticity_residual": float(np.linalg.norm(correction - correction.conj().T)),
            "decay_commutator": float(np.linalg.norm(difference * decay, ord="fro")),
            "correction_norm": float(np.linalg.norm(correction, ord="fro")),
            "minimum_decay_eigenvalue": float(np.linalg.eigvalsh(decay).min()),
            "reference_only": True, "coherent_correction_implemented": False,
            "interpretation": "analytic continuum Gaussian/Metropolis reference; no dense matrix is a circuit gate"}


def access_check(terms, proposals, *, beta=1, clock_bits=3, product_steps=1) -> dict:
    """Bounded circuit/block/reference evidence; no convergence claim."""
    _, _, n, _ = _dense_inputs(terms, proposals)
    circuit = build_gaussian_access(terms, proposals, beta=beta, clock_bits=clock_bits,
                                   product_steps=product_steps)
    actual = circuit_blocks(circuit)
    finite = reference_blocks(terms, proposals, beta=beta, clock_bits=clock_bits)
    product = reference_blocks(terms, proposals, beta=beta, clock_bits=clock_bits, product_steps=product_steps)
    normalization = np.einsum("alji,aljk->ik", actual.conj(), actual)
    diagnostics = finite_generator(terms, proposals, actual, beta=beta, clock_bits=clock_bits)
    return {"status": "checked_small_primitive_gaussian_access_only", "metadata": circuit.metadata,
            "primitive_counts": dict(circuit.count_ops()), "depth": circuit.depth(),
            "actual_vs_finite_trotter_max_entry_error": float(np.max(np.abs(actual - product))),
            "actual_vs_exact_finite_target_frobenius_error": float(np.linalg.norm(actual - finite)),
            "isometry_max_entry_error": float(np.max(np.abs(normalization - np.eye(1 << n)))),
            "finite_diagnostics": {k: v for k, v in diagnostics.items() if k not in ("decay", "correction", "gibbs")},
            "continuum_reference": continuum_generator_reference(terms, proposals, beta=beta),
            "remaining": ["controlled weak round and complete reset channel", "actual coherent correction access/evolution",
                          "uniform mesh and channel error bounds", "sampler mixing/output contract",
                          "complete repetitions/readout ledger", "Clifford+T synthesis and physical resources"]}
