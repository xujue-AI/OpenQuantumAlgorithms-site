# SPDX-License-Identifier: MIT
"""Original finite weak-channel endpoint for diagonal Pauli Hamiltonians.

The implemented channel retains both weak-flag outcomes and every environment
label. Diagonal I/Z Hamiltonians and Pauli proposals have diagonal decay, hence
the spectral coherent correction is exactly zero. This restricted endpoint is
not the general sampler or an exact finite-mesh KMS construction.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm
from scipy.special import log_ndtr

if __package__:
    from . import detailed_balance_access as access
else:
    import detailed_balance_access as access

MAX_SOURCE_ROUNDS = 256
MAX_REFERENCE_CLOCK_BITS = 4
MAX_REFERENCE_SYSTEM_WIRES = 1
MAX_REFERENCE_WIRES = 7
CHANNEL_PRIMITIVES = access.PRIMITIVES | {"reset", "measure"}


def _inputs(terms, proposals):
    terms, proposals, n = access._labels(terms, proposals)
    if any(set(label) - {"I", "Z"} for _, label in terms):
        raise ValueError("zero-correction endpoint requires diagonal I/Z Hamiltonian terms")
    return terms, proposals, n


def _delta(value: float, mesh: dict) -> float:
    value = access._real(value, "delta")
    if value <= 0 or value * mesh["eta"] * float(mesh["gamma"].max()) > 1:
        raise ValueError("delta must be positive with delta*eta*max(gamma)<=1")
    if value * mesh["eta"] == 0:
        raise ValueError("delta*eta underflows")
    probabilities = value * mesh["eta"] * mesh["gamma"]
    if np.any((mesh["gamma"] > 0) & (probabilities == 0)):
        raise ValueError("a nonzero weak probability underflows")
    return value


def _half(value: float) -> float:
    value = access._real(value, "controlled angle")
    half = value / 2
    if value and not half:
        raise ValueError("controlled angle cannot be halved without underflow")
    return half


def _cry(circuit: QuantumCircuit, angle: float, control: int, target: int):
    half = _half(angle)
    circuit.ry(half, target)
    circuit.cx(control, target)
    circuit.ry(-half, target)
    circuit.cx(control, target)


def _crz(circuit: QuantumCircuit, angle: float, control: int, target: int):
    half = _half(angle)
    circuit.rz(half, target)
    circuit.cx(control, target)
    circuit.rz(-half, target)
    circuit.cx(control, target)


def _controlled_instruction(circuit: QuantumCircuit, name: str, params: list,
                            wires: list[int], control: int):
    """Exact controlled lift of this access primitive alphabet."""
    if control in wires:
        raise ValueError("weak flag must be distinct from every access wire")
    if name == "cx":
        access._toffoli(circuit, control, wires[0], wires[1])
        return
    if len(wires) != 1:
        raise ValueError("controlled inverse contains an unsupported arity")
    target = wires[0]
    if name == "x":
        circuit.cx(control, target)
    elif name == "h":
        # H=RY(pi/2) Z, including its phase. Apply controlled Z first.
        circuit.h(target)
        circuit.cx(control, target)
        circuit.h(target)
        _cry(circuit, math.pi / 2, control, target)
    elif name in ("s", "sdg", "t", "tdg", "p"):
        angle = {"s": math.pi / 2, "sdg": -math.pi / 2,
                 "t": math.pi / 4, "tdg": -math.pi / 4}.get(name)
        if name == "p":
            angle = float(params[0])
        _half(angle)
        access._cp(circuit, angle, control, target)
    elif name == "ry":
        _cry(circuit, float(params[0]), control, target)
    elif name == "rz":
        _crz(circuit, float(params[0]), control, target)
    else:
        raise ValueError(f"controlled inverse cannot lift {name!r}")


def _conditional_inverse(circuit: QuantumCircuit, unitary: QuantumCircuit, flag: int):
    inverse = unitary.inverse()
    circuit.x(flag)
    phase = access._real(float(inverse.global_phase), "inverse global phase")
    if phase:
        circuit.p(phase, flag)
    for instruction in inverse.data:
        wires = [inverse.find_bit(q).index for q in instruction.qubits]
        _controlled_instruction(circuit, instruction.operation.name,
                                instruction.operation.params, wires, flag)
        access._guard(circuit)
    circuit.x(flag)


def build_weak_dilation(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                        *, beta=1.0, clock_bits=3, product_steps=1, delta=.125):
    """Primitive U/weak-RY/flag-zero-U-inverse unitary before discard/reset.

    The caller supplies an arbitrary system state and zero environment. No
    dense unitary is appended. Source materialization and simulation have
    separate limits.
    """
    terms, proposals, n = _inputs(terms, proposals)
    mesh = access.gaussian_mesh(clock_bits, beta)
    delta = _delta(delta, mesh)
    unitary = access.build_gaussian_access(terms, proposals, beta=beta,
                                          clock_bits=clock_bits,
                                          product_steps=product_steps)
    upper = 17 * len(unitary.data) + 2 * mesh["M"] + 4
    if upper > access.MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("weak-round preflight exceeds 100000 primitive operations")
    flag = unitary.num_qubits
    circuit = QuantumCircuit(flag + 1, name="Finite Gaussian weak round")
    circuit.compose(unitary, range(flag), inplace=True)
    angles = [2 * math.asin(math.sqrt(delta * mesh["eta"] * float(g)))
              for g in mesh["gamma"]]
    before = len(circuit.data)
    access._multiplexed_ry(circuit, list(range(n, n + mesh["clock_bits"])), flag, angles)
    weak_count = len(circuit.data) - before
    before = len(circuit.data)
    _conditional_inverse(circuit, unitary, flag)
    controlled_count = len(circuit.data) - before
    access._guard(circuit)
    if set(circuit.count_ops()) - access.PRIMITIVES:
        raise AssertionError("weak dilation contains composite gates")
    if any(not math.isfinite(float(p)) for ins in circuit.data for p in ins.operation.params):
        raise ValueError("weak dilation has a nonfinite gate angle")
    circuit.metadata = {
        "milestone": "finite_diagonal_weak_dilation", "n_system": n,
        "clock_bits": mesh["clock_bits"], "proposal_count": len(proposals),
        "proposal_bits": unitary.metadata["proposal_bits"], "beta": mesh["beta"],
        "delta": delta, "eta": mesh["eta"], "weak_flag": flag,
        "access_operations": len(unitary.data), "weak_operations": weak_count,
        "controlled_inverse_operations": controlled_count,
        "source_operations": len(circuit.data), "preflight_upper_bound": upper,
        "hamiltonian": [[c, p] for c, p in terms], "proposals": list(proposals),
        "product_steps": product_steps,
        "zero_correction_reason": "I/Z Hamiltonian; each Pauli proposal has a fixed computational permutation, so every B_dagger_B is diagonal",
        "environment_outcomes": "all frequency/proposal and both weak flag outcomes retained",
        "nonjump_inverse": "complete primitive U inverse conditional on flag zero, including global phase",
        "finite_mesh_exact_kms": False, "general_coherent_correction_implemented": False,
        "rotations_synthesized": False, "physical_resources_computed": False,
        "formal_proof_reproduced": False,
    }
    return circuit


def build_weak_round(terms, proposals, **kwargs):
    """Explicit initialization and discard/reset instrument on arbitrary system input."""
    dilation = build_weak_dilation(terms, proposals, **kwargs)
    n = dilation.metadata["n_system"]
    circuit = QuantumCircuit(dilation.num_qubits, name="Finite weak reset channel")
    for q in range(n, circuit.num_qubits):
        circuit.reset(q)
    circuit.compose(dilation, inplace=True)
    for q in range(n, circuit.num_qubits):
        circuit.reset(q)
    access._guard(circuit)
    circuit.metadata = dict(dilation.metadata,
        milestone="finite_diagonal_weak_reset_channel",
        resets_per_round=2 * (circuit.num_qubits - n),
        source_operations=len(circuit.data),
        initialization="arbitrary caller system; explicit environment reset before and after round")
    return circuit


def build_preparation(terms, proposals, *, rounds=32, initial="zero", readout=True, **kwargs):
    """Scalable guarded repeated channel, known input and complete final readout.

    This generator does not assign a general mixing certificate or synthesize
    rotations. The number of rounds is a task input, not a convergence theorem.
    """
    rounds = access._integer(rounds, "rounds", 1, MAX_SOURCE_ROUNDS)
    if initial not in ("zero", "one", "plus"):
        raise ValueError("initial must be zero, one or plus")
    if not isinstance(readout, bool):
        raise ValueError("readout must be boolean")
    one = build_weak_round(terms, proposals, **kwargs)
    n = one.metadata["n_system"]
    upper = rounds * len(one.data) + 3 * n
    if upper > access.MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("preparation preflight exceeds 100000 primitive operations")
    circuit = QuantumCircuit(one.num_qubits, n if readout else 0,
                             name="Finite filtered-jump preparation")
    for q in range(n):
        circuit.reset(q)
        if initial == "one":
            circuit.x(q)
        elif initial == "plus":
            circuit.h(q)
    for _ in range(rounds):
        circuit.compose(one, inplace=True)
    if readout:
        for q in range(n):
            circuit.measure(q, q)
    access._guard(circuit)
    circuit.metadata = dict(one.metadata,
        milestone="finite_diagonal_preparation_source", rounds=rounds,
        initial=initial, readout=readout, source_operations=len(circuit.data),
        full_round_operations=len(one.data),
        total_resets=n + rounds * one.metadata["resets_per_round"],
        initialization="explicit known system reset and optional X/H; environment initialized and discarded each round",
        mixing_certificate="none; bounded channel-power checks may be supplied separately",
        repetitions="each observable draw reruns all rounds; no accepted-only branch or algorithm retry factor")
    return circuit


def _reference_inputs(terms, proposals, beta, clock_bits):
    terms, proposals, n = _inputs(terms, proposals)
    if n > MAX_REFERENCE_SYSTEM_WIRES:
        raise ValueError("weak-channel matrix references are capped at one system wire")
    access._integer(clock_bits, "reference clock_bits", 1, MAX_REFERENCE_CLOCK_BITS)
    mesh = access.gaussian_mesh(clock_bits, beta)
    wires = n + clock_bits + (len(proposals) - 1).bit_length() + 1
    if wires > MAX_REFERENCE_WIRES:
        raise ValueError("weak-channel references exceed seven total wires")
    return terms, proposals, n, mesh


def _prefix_prepare_matrix(amplitudes):
    """Independent mathematical multiplexor extension; no circuit inspection."""
    probabilities = np.asarray(amplitudes, float) ** 2
    size = len(probabilities)
    m = (size - 1).bit_length()
    result = np.eye(size, dtype=complex)
    for target in range(m - 1, -1, -1):
        step = np.eye(size, dtype=complex)
        width = 1 << target
        for prefix in range(1 << (m - target - 1)):
            start = prefix << (target + 1)
            left = float(probabilities[start:start + width].sum())
            right = float(probabilities[start + width:start + 2 * width].sum())
            theta = 2 * math.atan2(math.sqrt(right), math.sqrt(left)) if left + right else 0
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            for low in range(width):
                a, b = start + low, start + width + low
                step[a, a], step[a, b], step[b, a], step[b, b] = c, -s, s, c
        result = step @ result
    return result


def reference_access_unitary(terms, proposals, *, beta=1, clock_bits=3):
    """One-spin diagonal-H exact finite access extension, independent of Qiskit."""
    terms, proposals, n, mesh = _reference_inputs(terms, proposals, beta, clock_bits)
    dim, size, count = 1 << n, mesh["M"], len(proposals)
    h = sum((c * access.pauli_matrix(p) for c, p in terms), np.zeros((dim, dim), complex))
    prep_clock = _prefix_prepare_matrix(mesh["amplitudes"])
    hadamard = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
    prep_proposal = np.ones((1, 1), complex)
    for _ in range((count - 1).bit_length()):
        prep_proposal = np.kron(prep_proposal, hadamard)
    full = count * size * dim
    conjugate_select = np.zeros((full, full), complex)
    for a, label in enumerate(proposals):
        pauli = access.pauli_matrix(label)
        for j, time in enumerate(mesh["times"]):
            v = np.diag(np.exp(-1j * np.diag(h).real * float(time)))
            offset = (a * size + j) * dim
            conjugate_select[offset:offset + dim, offset:offset + dim] = v.conj().T @ pauli @ v
    dft = np.exp(-2j * math.pi * np.outer(np.arange(size), np.arange(size)) / size) / math.sqrt(size)
    prepare = np.kron(prep_proposal, np.kron(prep_clock, np.eye(dim)))
    return np.kron(np.eye(count), np.kron(dft, np.eye(dim))) @ conjugate_select @ prepare


def reference_kraus(terms, proposals, *, beta=1, clock_bits=3, delta=.125):
    """All finite nojump/jump Kraus from independently constructed full U."""
    terms, proposals, n, mesh = _reference_inputs(terms, proposals, beta, clock_bits)
    delta = _delta(delta, mesh)
    dim = 1 << n
    u = reference_access_unitary(terms, proposals, beta=beta, clock_bits=clock_bits)
    survival = np.tile(np.repeat(np.sqrt(1 - delta * mesh["eta"] * mesh["gamma"]), dim), len(proposals))
    q0 = u.conj().T @ (survival[:, None] * u)
    jump = np.tile(np.repeat(np.sqrt(delta * mesh["eta"] * mesh["gamma"]), dim), len(proposals))
    q1 = jump[:, None] * u
    return np.concatenate([q0[:, :dim].reshape(-1, dim, dim),
                           q1[:, :dim].reshape(-1, dim, dim)])


def circuit_kraus(circuit: QuantumCircuit):
    """Full actual primitive dilation checked on each system basis column."""
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("actual channel checks require a QuantumCircuit")
    meta = circuit.metadata or {}
    if meta.get("milestone") != "finite_diagonal_weak_dilation":
        raise ValueError("actual channel checks require the weak-dilation contract")
    n = access._integer(meta.get("n_system"), "n_system", 1, MAX_REFERENCE_SYSTEM_WIRES)
    m = access._integer(meta.get("clock_bits"), "clock_bits", 1, MAX_REFERENCE_CLOCK_BITS)
    p = access._integer(meta.get("proposal_bits"), "proposal_bits", 0, 2)
    count = access._integer(meta.get("proposal_count"), "proposal_count", 1, 4)
    if (count not in (1, 2, 4) or p != (count - 1).bit_length()
            or circuit.num_qubits != n + m + p + 1
            or circuit.num_qubits > MAX_REFERENCE_WIRES or circuit.num_clbits
            or set(circuit.count_ops()) - access.PRIMITIVES
            or meta.get("weak_flag") != circuit.num_qubits - 1):
        raise ValueError("actual channel check has an invalid primitive width/flag contract")
    access._guard(circuit)
    try:
        finite_angles = all(math.isfinite(float(v)) for ins in circuit.data for v in ins.operation.params)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("actual channel checks require finite numeric angles") from error
    if not finite_angles:
        raise ValueError("actual channel checks require finite angles")
    dim, env = 1 << n, 1 << (circuit.num_qubits - n)
    kraus = np.empty((env, dim, dim), complex)
    for column in range(dim):
        initial = np.zeros(1 << circuit.num_qubits, complex)
        initial[column] = 1
        kraus[:, :, column] = Statevector(initial).evolve(circuit).data.reshape(env, dim)
    return kraus


def channel_superoperator(kraus):
    kraus = np.asarray(kraus, complex)
    if kraus.ndim != 3 or kraus.shape[1:] != (2, 2) or len(kraus) > 64 or not np.all(np.isfinite(kraus)):
        raise ValueError("channel calculations require at most64 finite one-spin Kraus matrices")
    return sum((np.kron(k.conj(), k) for k in kraus), np.zeros((4, 4), complex))


def apply_channel(superoperator, rho, *, rounds=1):
    rounds = access._integer(rounds, "reference rounds", 0, MAX_SOURCE_ROUNDS)
    s, rho = np.asarray(superoperator, complex), np.asarray(rho, complex)
    if s.shape != (4, 4) or rho.shape != (2, 2) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(rho)):
        raise ValueError("channel iteration requires finite one-spin matrices")
    return (np.linalg.matrix_power(s, rounds) @ rho.reshape(4, order="F")).reshape(2, 2, order="F")


def _generator_from_kernel(h, proposals, kernel):
    """Independent Bohr-component Lindbladian, no circuit or access blocks."""
    values = np.diag(h).real
    decay = np.zeros((2, 2), complex)
    transition = np.zeros((4, 4), complex)
    for label in proposals:
        a = access.pauli_matrix(label) / math.sqrt(len(proposals))
        components = []
        for i in range(2):
            for j in range(2):
                if a[i, j]:
                    e = np.zeros((2, 2), complex)
                    e[i, j] = a[i, j]
                    components.append((values[i] - values[j], e))
        for nu, a_nu in components:
            for mu, a_mu in components:
                coefficient = kernel(nu, mu)
                transition += coefficient * np.kron(a_mu.conj(), a_nu)
                decay += coefficient * a_mu.conj().T @ a_nu
    generator = transition - .5 * (np.kron(np.eye(2), decay) + np.kron(decay.T, np.eye(2)))
    return generator, decay


def independent_generators(terms, proposals, *, beta=1, clock_bits=3):
    """Finite quadrature and analytic continuum references, restricted C=0."""
    terms, proposals, _, mesh = _reference_inputs(terms, proposals, beta, clock_bits)
    h = sum((c * access.pauli_matrix(p) for c, p in terms), np.zeros((2, 2), complex))
    def finite_kernel(nu, mu):
        times, frequencies = mesh["times"], mesh["frequencies"]
        def transform(frequency):
            return np.sum(mesh["amplitudes"] * np.exp(1j * (frequency - frequencies[:, None]) * times), axis=1) / math.sqrt(mesh["M"])
        first, second = transform(nu), transform(mu)
        return np.sum(mesh["eta"] * mesh["gamma"] * first * second.conj())
    def continuum_kernel(nu, mu):
        x = beta * (nu + mu) / 2
        log_rate = np.logaddexp(log_ndtr(-x - .5), -x + log_ndtr(x - .5))
        return float(np.exp(-beta * beta * (nu - mu) ** 2 / 8 + log_rate))
    finite, decay = _generator_from_kernel(h, proposals, finite_kernel)
    continuum, continuum_decay = _generator_from_kernel(h, proposals, continuum_kernel)
    values = np.diag(h).real
    probabilities = np.exp(-beta * (values - values.min()))
    probabilities /= probabilities.sum()
    return {"finite": finite, "continuum": continuum, "decay": decay,
            "continuum_decay": continuum_decay, "h": h,
            "gibbs": np.diag(probabilities).astype(complex)}


def channel_check(terms, proposals, *, beta=1, clock_bits=3, product_steps=1,
                  delta=.125, rounds=32):
    """Bounded full-channel, fixed-point and explicit channel-power evidence."""
    _reference_inputs(terms, proposals, beta, clock_bits)
    rounds = access._integer(rounds, "rounds", 1, MAX_SOURCE_ROUNDS)
    dilation = build_weak_dilation(terms, proposals, beta=beta, clock_bits=clock_bits,
                                  product_steps=product_steps, delta=delta)
    actual = circuit_kraus(dilation)
    reference = reference_kraus(terms, proposals, beta=beta, clock_bits=clock_bits, delta=delta)
    s = channel_superoperator(actual)
    sr = channel_superoperator(reference)
    refs = independent_generators(terms, proposals, beta=beta, clock_bits=clock_bits)
    completeness = np.einsum("aji,ajk->ik", actual.conj(), actual)
    choi = sum((np.outer(k.reshape(4, order="F"), k.reshape(4, order="F").conj()) for k in actual), np.zeros((4, 4), complex))
    populations = s[np.ix_([0, 3], [0, 3])].real
    upward, downward = populations[0, 1], populations[1, 0]
    total = upward + downward
    if total <= 1e-14:
        stationary = None
        stationary_distance = None
        stationary_residual = None
    else:
        stationary = np.diag([upward / total, downward / total]).astype(complex)
        stationary_distance = float(np.linalg.norm(stationary - refs["gibbs"], ord="nuc") / 2)
        stationary_residual = float(np.linalg.norm(apply_channel(s, stationary) - stationary, ord="fro"))
    bloch = [access.PAULIS[x] for x in ("X", "Y", "Z")]
    linear = np.array([[.5 * np.trace(p @ apply_channel(s, q)).real for q in bloch] for p in bloch])
    contraction = float(np.linalg.svd(linear, compute_uv=False)[0])
    inputs = [np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex),
              np.array([[1, 1], [1, 1]], complex) / 2,
              np.array([[1, -1j], [1j, 1]], complex) / 2]
    outputs = []
    for rho in inputs:
        result = apply_channel(s, rho, rounds=rounds)
        record = {"trace_error": float(abs(np.trace(result) - 1)),
                  "minimum_eigenvalue": float(np.linalg.eigvalsh((result + result.conj().T) / 2).min()),
                  "gibbs_trace_distance": float(np.linalg.norm(result - refs["gibbs"], ord="nuc") / 2)}
        if stationary is not None:
            record["finite_stationary_trace_distance"] = float(np.linalg.norm(result - stationary, ord="nuc") / 2)
            residual = float(np.linalg.norm(apply_channel(s, stationary) - stationary, ord="nuc") / 2)
            accumulation = sum(contraction ** k for k in range(rounds))
            record["numerical_contraction_estimate_to_finite_stationary"] = (
                contraction ** rounds * float(np.linalg.norm(rho - stationary, ord="nuc") / 2)
                + residual * accumulation)
        outputs.append(record)
    return {
        "status": "checked_small_finite_diagonal_weak_channel", "metadata": dilation.metadata,
        "primitive_counts": dict(dilation.count_ops()), "depth": dilation.depth(),
        "actual_vs_independent_kraus_max_entry_error": float(np.max(np.abs(actual - reference))),
        "actual_vs_independent_channel_frobenius_error": float(np.linalg.norm(s - sr)),
        "trace_preservation_max_entry_error": float(np.max(np.abs(completeness - np.eye(2)))),
        "minimum_choi_eigenvalue": float(np.linalg.eigvalsh((choi + choi.conj().T) / 2).min()),
        "nonzero_environment_nojump_weight": float(np.sum(np.abs(actual[1:len(actual)//2]) ** 2)),
        "finite_first_order_channel_error": float(np.linalg.norm(s - np.eye(4) - delta * refs["finite"])),
        "finite_semigroup_one_round_error": float(np.linalg.norm(s - expm(delta * refs["finite"]))),
        "continuum_generator_gibbs_residual": float(np.linalg.norm(refs["continuum"] @ refs["gibbs"].reshape(4, order="F"))),
        "finite_generator_gibbs_residual": float(np.linalg.norm(refs["finite"] @ refs["gibbs"].reshape(4, order="F"))),
        "finite_round_gibbs_residual": float(np.linalg.norm(apply_channel(s, refs["gibbs"]) - refs["gibbs"])),
        "decay_commutator": float(np.linalg.norm(refs["h"] @ refs["decay"] - refs["decay"] @ refs["h"])),
        "finite_stationary_populations": None if stationary is None else np.diag(stationary).real.tolist(),
        "finite_stationary_gibbs_trace_distance": stationary_distance,
        "finite_stationary_channel_residual": stationary_residual,
        "bloch_contraction_norm": contraction,
        "contraction_scope": "floating-point small-channel diagnostic with stationary-residual accumulation; no certified rounding enclosure or general mixing certificate",
        "rounds": rounds, "spanning_input_channel_power_checks": outputs,
        "channel_power_scope": "iterate actual full one-round channel after independent Kraus checks; no full statevector/density simulation of the large repeated circuit",
        "remaining": ["general noncommuting coherent-correction circuit", "uniform finite mesh/window/aliasing error bound",
                      "general mixing guarantee", "rotation synthesis and complete frozen surface-code model", "mechanized proof bridge"]}
