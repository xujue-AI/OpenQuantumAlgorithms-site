# SPDX-License-Identifier: MIT
"""Portable reference fragments from the accepted fixed Gibbs-Z calculation.

The four function definitions below are copied byte-for-byte from original
scientific source commit 47ffd10368a738f9576053491a56c6d80a10e602,
scripts/check_finite_gibbs_output.py (SHA-256
dc4666c8f17caf85bc00b0e15df9ea6c90671ac7b39cd1d5c9a10f288083c816).
They contain the full independent Gaussian completion and all 64 Kraus blocks;
no private runtime admission, caller or output-allocation code is included.
The portable module packaging and reproduction wrapper are source-only and
unexecuted. The accepted calculation used these scientific definitions in its
separate original runtime.
"""
from __future__ import annotations

import math


def _matrix_record(np, value):
    array = np.asarray(value, complex)
    if array.size > 16384 or array.ndim > 3 or not np.all(np.isfinite(array)):
        raise ValueError("raw matrix record exceeds the fixed finite envelope")
    return {"shape": list(array.shape), "real": array.real.tolist(), "imag": array.imag.tolist()}


def _max_entry(np, value):
    result = float(np.max(np.abs(value)))
    if not math.isfinite(result):
        raise ValueError("nonfinite raw residual")
    return result


def _full_reference(work):
    """Independent mathematical full completion; no production matrix helper."""
    work.count("independent_full_U_Kraus_constructions")
    np, scalar = work.np, work.math
    work.preprocessing("mesh_constructions", 16)
    indices = np.arange(16, dtype=int)
    signed = np.where(indices < 8, indices, indices - 16)
    dt = .5 * scalar.sqrt(2 * math.pi / 16)
    dw = 2 * scalar.sqrt(2 * math.pi / 16)
    weights = work.exp(-math.pi * signed.astype(float) ** 2 / 16)
    mass = float(weights.sum())
    mesh = {"signed_indices": signed, "times": dt * signed, "frequencies": dw * signed,
            "amplitudes": np.sqrt(weights / mass), "eta": mass / scalar.sqrt(16),
            "gamma": work.exp(-np.maximum(dw * signed + .5, 0))}
    if (not np.all(np.isfinite(mesh["amplitudes"])) or np.any(mesh["gamma"] <= 0)
            or not 0 < .125 * mesh["eta"] * float(mesh["gamma"].max()) <= 1):
        raise ValueError("independent fixed mesh is outside its positive weak-probability domain")
    probabilities = np.asarray(mesh["amplitudes"], float) ** 2
    clock = np.eye(16, dtype=complex)
    for target in range(3, -1, -1):
        step = np.eye(16, dtype=complex)
        width = 1 << target
        for prefix in range(1 << (3 - target)):
            start = prefix << (target + 1)
            left = float(probabilities[start:start + width].sum())
            right = float(probabilities[start + width:start + 2 * width].sum())
            theta = 2 * scalar.atan2(scalar.sqrt(right), scalar.sqrt(left))
            c, s = scalar.cos(theta / 2), scalar.sin(theta / 2)
            for low in range(width):
                first, second = start + low, start + width + low
                step[first, first], step[first, second] = c, -s
                step[second, first], step[second, second] = s, c
        clock = work.mm(step, clock)
    proposal = np.array([[1, 1], [1, -1]], complex) / scalar.sqrt(2)
    paulis = (np.array([[0, 1], [1, 0]], complex), np.diag([1, -1]).astype(complex))
    conjugate_select = np.zeros((64, 64), complex)
    for a, pauli in enumerate(paulis):
        for j, instant in enumerate(mesh["times"]):
            diagonal = np.diag(work.exp(-1j * np.array([1.0, -1.0]) * float(instant)))
            block = work.mm(work.mm(diagonal.conj().T, pauli), diagonal)
            offset = (a * 16 + j) * 2
            conjugate_select[offset:offset + 2, offset:offset + 2] = block
    dft = work.exp(-2j * math.pi * np.outer(np.arange(16), np.arange(16)) / 16) / scalar.sqrt(16)
    prepared = np.kron(proposal, np.kron(clock, np.eye(2)))
    unitary = work.mm(work.mm(np.kron(np.eye(2), np.kron(dft, np.eye(2))),
                              conjugate_select), prepared)
    survival = np.tile(np.repeat(np.sqrt(1 - .125 * mesh["eta"] * mesh["gamma"]), 2), 2)
    jump = np.tile(np.repeat(np.sqrt(.125 * mesh["eta"] * mesh["gamma"]), 2), 2)
    nojump = work.mm(unitary.conj().T, survival[:, None] * unitary)
    jumps = jump[:, None] * unitary
    kraus = np.concatenate((nojump[:, :2].reshape(32, 2, 2),
                            jumps[:, :2].reshape(32, 2, 2)))
    return mesh, unitary, kraus


def _superoperator(np, kraus):
    if kraus.shape != (64, 2, 2) or not np.all(np.isfinite(kraus)):
        raise ValueError("all sixty-four finite Kraus blocks are required")
    return sum((np.kron(block.conj(), block) for block in kraus), np.zeros((4, 4), complex))
