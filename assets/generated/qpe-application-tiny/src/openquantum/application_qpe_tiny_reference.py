# SPDX-License-Identifier: MIT
"""Independent fixed-task references for the PRIVATE tiny application-QPE check.

Source authored for a later once-only bounded grant; this module has not been
imported or executed as part of authoring. Its arrays are numerical references,
not interval certificates, prepared eigenstates, algorithm gates, shot samples,
phase inference, fault-tolerant resources or application confidence evidence.

The supplied model is H=-Z0-Z1+X0X1/4, input |00>, tau0=pi/3 and
U=-exp(-i H tau0). Data index d is q0+2*q1. The full controlled matrix uses
local wire 0 as control and wires 1/2 as data q0/q1: index=control+2*d.
All matrices are assembled from independent original-basis two-by-two formulas;
no circuit constructor, production reference or Qiskit object is imported.

Only powers 1, 2 and 4, fixed two-bit QPE words and fixed two-round records are
admitted. A record index is y0+4*y1, where y0 is the first chronological word.
The reduced law check is not the full m=6/R=41/K=2 application protocol.
"""
from __future__ import annotations

import math

import numpy as np

ADMITTED_POWERS = (1, 2, 4)
WORD_COUNT = 4
RECORD_COUNT = 16


def _power(power: int) -> int:
    if isinstance(power, bool) or not isinstance(power, int) or power not in ADMITTED_POWERS:
        raise ValueError("The tiny reference admits only integer powers 1, 2 and 4")
    return power


def _basis(basis: str) -> str:
    if not isinstance(basis, str) or basis not in ("cos", "sin"):
        raise ValueError("Quadrature must be cos or sin")
    return basis


def _phase(phase: float) -> float:
    if isinstance(phase, bool) or not isinstance(phase, (int, float)):
        raise ValueError("Reference phase must be a finite real scalar in [0, 1)")
    value = float(phase)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("Reference phase must be a finite real scalar in [0, 1)")
    return value


def hamiltonian_matrix() -> np.ndarray:
    """Original data-basis matrix, including both even and odd sectors."""
    return np.array(
        [[-2, 0, 0, 1 / 4],
         [0, 0, 1 / 4, 0],
         [0, 1 / 4, 0, 0],
         [1 / 4, 0, 0, 2]],
        dtype=np.complex128,
    )


def evolution_matrix(power: int = 1) -> np.ndarray:
    """exp(-i H * power*tau0), from original-basis even/odd block formulas.

    This direct longer-time expression is an independent bounded reference.
    Algorithm source must instead call the complete one-step kernel literally.
    No phase fitting or matrix-exponential/eigenvector routine is used.
    """
    power = _power(power)
    time = power * math.pi / 3
    radius = math.sqrt(65) / 4
    even_h = np.array([[-2, 1 / 4], [1 / 4, 2]], dtype=np.complex128)
    even = (math.cos(radius * time) * np.eye(2, dtype=np.complex128)
            - 1j * math.sin(radius * time) * even_h / radius)
    odd = np.array(
        [[math.cos(time / 4), -1j * math.sin(time / 4)],
         [-1j * math.sin(time / 4), math.cos(time / 4)]],
        dtype=np.complex128,
    )
    result = np.zeros((4, 4), dtype=np.complex128)
    for block, indices in ((even, (0, 3)), (odd, (1, 2))):
        for row, original_row in enumerate(indices):
            for column, original_column in enumerate(indices):
                result[original_row, original_column] = block[row, column]
    return result


def base_unitary_matrix(power: int = 1) -> np.ndarray:
    """U**power, retaining the exact scalar reference factor (-1)**power."""
    power = _power(power)
    return ((-1) ** power) * evolution_matrix(power)


def _controlled_block(data_block: np.ndarray) -> np.ndarray:
    """Embed a fixed 4x4 reference into interleaved control-zero/one indices."""
    if not isinstance(data_block, np.ndarray) or data_block.shape != (4, 4):
        raise ValueError("The fixed reference requires a four-dimensional data block")
    result = np.zeros((8, 8), dtype=np.complex128)
    for data_index in range(4):
        result[2 * data_index, 2 * data_index] = 1
    for row in range(4):
        for column in range(4):
            result[2 * row + 1, 2 * column + 1] = data_block[row, column]
    return result


def controlled_unitary_matrix(power: int = 1) -> np.ndarray:
    """Full 8x8 controlled U**power with identity on every control-zero state."""
    return _controlled_block(base_unitary_matrix(_power(power)))


def controlled_evolution_without_reference(power: int = 1) -> np.ndarray:
    """Independent expected operator for the missing-control-Z negative source.

    At power one this omits the essential minus sign on control one. At even
    powers two/four, the omitted factor cancels, so this negative source agrees
    with controlled U**power. The distinguishing check must include power one.
    """
    return _controlled_block(evolution_matrix(_power(power)))


def spectral_components() -> tuple[dict[str, float | str], ...]:
    """Original-model energies, phases and |00> weights as numerical diagnostics.

    Exact expressions are R=sqrt(65)/4 and w0=(1+8/sqrt(65))/2. Zero-weight odd
    components remain present; zero input weight does not exempt their evolution
    from the full controlled-matrix comparison.
    """
    radius = math.sqrt(65) / 4
    ground_weight = (1 + 8 / math.sqrt(65)) / 2
    entries = (
        ("ground-even", -radius, ground_weight),
        ("lower-odd", -1 / 4, 0.0),
        ("upper-odd", 1 / 4, 0.0),
        ("excited-even", radius, 1 - ground_weight),
    )
    return tuple(
        {"label": label, "energy": energy, "phase": (3 - energy) / 6,
         "input_weight": weight}
        for label, energy, weight in entries
    )


def qpe_word_probabilities(phase: float) -> np.ndarray:
    """Independent four-word finite Fourier sum for fixed two-bit exact QPE.

    Each probability is |sum_{r=0}^3 exp(2*pi*i*r*(phase-y/4))/4|**2.
    Values are not renormalized or clipped; future checks must retain their
    ordinary floating residuals and compare normalization explicitly.
    """
    phase = _phase(phase)
    result = np.empty(WORD_COUNT, dtype=float)
    for word in range(WORD_COUNT):
        amplitude = 0j
        for exponent in range(WORD_COUNT):
            angle = 2 * math.pi * exponent * (phase - word / WORD_COUNT)
            amplitude += complex(math.cos(angle), math.sin(angle)) / WORD_COUNT
        result[word] = amplitude.real ** 2 + amplitude.imag ** 2
    return result


def _even_word_laws() -> tuple[float, np.ndarray, np.ndarray]:
    radius = math.sqrt(65) / 4
    weight = (1 + 8 / math.sqrt(65)) / 2
    ground = qpe_word_probabilities((3 + radius) / 6)
    excited = qpe_word_probabilities((3 - radius) / 6)
    return weight, ground, excited


def retained_joint_probabilities() -> np.ndarray:
    """Sixteen records for two rounds on one retained |00> target.

    Conditional on one retained latent eigenvalue, the two words are IID.
    Thus J(y0,y1)=w0*P0(y0)*P0(y1)+(1-w0)*Pe(y0)*Pe(y1), with
    vector index y0+4*y1. The target is not re-prepared between these rounds.
    """
    weight, ground, excited = _even_word_laws()
    return np.array(
        [weight * ground[first] * ground[second]
         + (1 - weight) * excited[first] * excited[second]
         for second in range(WORD_COUNT) for first in range(WORD_COUNT)],
        dtype=float,
    )


def fresh_joint_probabilities() -> np.ndarray:
    """Product-of-mixed-marginals negative reference for target re-preparation.

    This law describes independent |00> preparations between the two rounds;
    it is not the retained-target record law used by the application contract.
    """
    weight, ground, excited = _even_word_laws()
    marginal = weight * ground + (1 - weight) * excited
    return np.array(
        [marginal[first] * marginal[second]
         for second in range(WORD_COUNT) for first in range(WORD_COUNT)],
        dtype=float,
    )


def retained_minus_fresh_reference() -> np.ndarray:
    """Independent factorized difference w0*(1-w0)*(P0-Pe) tensor (P0-Pe).

    The same chronological index convention is retained. This witness exposes
    a target-reset mistake without inferring a ground phase from any records.
    """
    weight, ground, excited = _even_word_laws()
    difference = ground - excited
    return np.array(
        [weight * (1 - weight) * difference[first] * difference[second]
         for second in range(WORD_COUNT) for first in range(WORD_COUNT)],
        dtype=float,
    )


def fresh_zero_probability(power: int, basis: str) -> float:
    """Fresh |00> Hadamard probability from a direct scalar block expression.

    <00|U**k|00>=(-1)**k*[cos(R*k*tau0)+i*(2/R)*sin(R*k*tau0)].
    An H--controlled-U**k--S-dagger--H readout exposes its positive imaginary
    part for the sine quadrature. No eigenphase recovery or sampling occurs.
    """
    power, basis = _power(power), _basis(basis)
    radius = math.sqrt(65) / 4
    angle = radius * power * math.pi / 3
    sign = (-1) ** power
    signal = sign * (math.cos(angle) if basis == "cos"
                     else (2 / radius) * math.sin(angle))
    return (1 + signal) / 2


def ground_zero_probability(power: int, basis: str) -> float:
    """Ideal ground-component fringe; its phase is (3+R)/6."""
    power, basis = _power(power), _basis(basis)
    angle = (math.sqrt(65) / 4) * power * math.pi / 3
    signal = ((-1) ** power) * (math.cos(angle) if basis == "cos" else math.sin(angle))
    return (1 + signal) / 2


def excited_zero_probability(power: int, basis: str) -> float:
    """Ideal excited-even component; its phase is (3-R)/6."""
    power, basis = _power(power), _basis(basis)
    angle = (math.sqrt(65) / 4) * power * math.pi / 3
    signal = ((-1) ** power) * (math.cos(angle) if basis == "cos" else -math.sin(angle))
    return (1 + signal) / 2


def fringe_reference(power: int, basis: str) -> dict[str, int | str | float]:
    """Bounded diagnostic record for each of the six admitted fresh fringes.

    The frozen contamination allowance is q=(1-8/sqrt(65))/2<1/256. Comparing
    measured circuit/reference fringes with q is a tiny numerical check; this
    function does not certify implemented synthesis, physical bias or the
    full robust-inference confidence promise.
    """
    power, basis = _power(power), _basis(basis)
    actual = fresh_zero_probability(power, basis)
    ground = ground_zero_probability(power, basis)
    excited = excited_zero_probability(power, basis)
    contamination = (1 - 8 / math.sqrt(65)) / 2
    return {
        "power": power,
        "basis": basis,
        "input_zero_probability": actual,
        "ground_zero_probability": ground,
        "excited_zero_probability": excited,
        "excited_input_weight": contamination,
        "absolute_ground_fringe_bias": abs(actual - ground),
        "spectral_mixture_zero_probability": ((1 - contamination) * ground
                                              + contamination * excited),
        "contamination_bound_expression": "(1-8/sqrt(65))/2 < 1/256",
    }
