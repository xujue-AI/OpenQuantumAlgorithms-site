# SPDX-License-Identifier: MIT
"""Independent references for the fixed two-round finite thermal proposal.

This source is prepared but unexecuted. Numerical work occurs only through
run_check() under the separately frozen driver: eight 2x2 Pauli exponentials,
two exponent-one interval powers, two four-wire weak basis columns, four
six-wire density stages, and one final 4x4 finite-generator exponential.
The complete prefix-RY completion and signed product chronology are rebuilt
without production matrix helpers. Chen et al., Nature 646 (2025), Eq. 19,
DOI 10.1038/s41586-025-09583-x specifies the energy-basis correction.

All arrays below are tiny algebraic references or bounded density records,
never dense quantum gates. No eigensolver, SVD, fidelity, validity predicate,
to_statevector conversion or full-circuit SuperOp is used. Saved correction
blocks are an authenticated gate-simulation dependence, not fresh independent
failure reconstruction. General thermal, KMS/mixing, numerical-certification,
synthesis, physical and formal claims remain false.
"""
from __future__ import annotations

import math

import numpy as np

if __package__:
    from .thermal_two_round_contract import SCOPE_FLAGS
else:
    from thermal_two_round_contract import SCOPE_FLAGS

TERMS = ((0.6, "Z"), (0.8, "X"))
PROPOSALS = ("Z",)
BETA = 1.0
CLOCK_BITS = 2
PRODUCT_STEPS = 1
DELTA = 0.125
ROUNDS = 2
ACCEPTED_ALPHA = 0.9775450868978823
DEFAULT_TOLERANCES = {"entry": 1e-10, "completeness": 1e-10, "analytic": 1e-9}
BUNDLE_LAYOUT = {
    "weak_dilation": (4, 0),
    "weak_reset_prefix": (6, 0),
    "correction_lcu": (6, 0),
    "round_macro": (6, 5),
    "two_round_trial": (6, 11),
}
DECLARED_WORK_CALLS = {
    "statevector_evolutions": 2,
    "external_density_stage_evolutions": 4,
    "internal_partial_reset_evolve_dispatches": 16,
    "tiny_2x2_expm_calls": 8,
    "finite_4x4_generator_expm_calls": 1,
    "interval_matrix_power_calls": 2,
}


def _finite_real(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label + " must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + " must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(label + " must be a finite real number")
    return result


def _array(value, shape, label):
    result = np.asarray(value, dtype=complex)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(label + " has an incorrect shape or nonfinite entries")
    return result


def _max_entry(value):
    result = float(np.max(np.abs(value)))
    if not math.isfinite(result):
        raise ValueError("a raw entry residual is nonfinite")
    return result


def _frobenius(value):
    """Explicit entry sum, without dispatching a matrix norm/eigensolver."""
    result = math.sqrt(float(np.sum(np.abs(value) ** 2)))
    if not math.isfinite(result):
        raise ValueError("a raw Frobenius residual is nonfinite")
    return result


def _tolerances(tolerances):
    result = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    if set(result) != set(DEFAULT_TOLERANCES):
        raise ValueError("the fixed reference requires entry/completeness/analytic tolerances")
    for name, maximum in DEFAULT_TOLERANCES.items():
        result[name] = _finite_real(result[name], "tolerance " + name)
        if not 0 < result[name] <= maximum:
            raise ValueError("a tolerance exceeds the fixed reference envelope")
    return result


def matrix_record(matrix):
    """Retain raw real/imaginary entries, without clipping or phase fitting."""
    value = np.asarray(matrix, dtype=complex)
    if value.ndim > 3 or value.size > 4096 or not np.all(np.isfinite(value)):
        raise ValueError("a raw matrix record exceeds the six-wire finite envelope")
    return {"shape": list(value.shape), "real": value.real.tolist(),
            "imag": value.imag.tolist()}


def _ry(angle):
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _qubit_coefficients(matrix):
    """Closed coefficients of the Hermitian part; keep residuals separately."""
    matrix = _array(matrix, (2, 2), "qubit matrix")
    return (
        float((matrix[0, 0].real + matrix[1, 1].real) / 2),
        float((matrix[0, 1].real + matrix[1, 0].real) / 2),
        float((matrix[1, 0].imag - matrix[0, 1].imag) / 2),
        float((matrix[0, 0].real - matrix[1, 1].real) / 2),
    )


def _qubit_spectrum(matrix):
    """Two analytic eigenvalue diagnostics, with no diagonalization/clipping."""
    center, x, y, z = _qubit_coefficients(matrix)
    radius = math.sqrt(x * x + y * y + z * z)
    return (center - radius, center + radius)


def _qubit_trace_norm(matrix):
    """Trace norm of an ideal Hermitian qubit difference by the closed formula."""
    lower, upper = _qubit_spectrum(matrix)
    return abs(lower) + abs(upper)


def _channel(blocks, rho):
    """Apply only 2x2 Kraus factors; no full-circuit channel conversion."""
    result = np.zeros((2, 2), dtype=complex)
    for block in blocks:
        result += block @ rho @ block.conj().T
    return result


def _gram(blocks):
    result = np.zeros((2, 2), dtype=complex)
    for block in blocks:
        result += block.conj().T @ block
    return result


def _pullback(blocks, effect):
    result = np.zeros((2, 2), dtype=complex)
    for block in blocks:
        result += block.conj().T @ effect @ block
    return result


def _system_embedding(rho):
    """Algebraically attach five zero auxiliaries; q0 is least significant."""
    rho = _array(rho, (2, 2), "system density input")
    embedded = np.zeros((64, 64), dtype=complex)
    embedded[:2, :2] = rho
    return embedded


def _environment_blocks(rho):
    """All 32 diagonal environment blocks before herald cleanup."""
    rho = _array(rho, (64, 64), "six-wire density output")
    return np.array([rho[2 * z:2 * z + 2, 2 * z:2 * z + 2]
                     for z in range(32)], dtype=complex)


def _system_marginal(rho):
    result = np.zeros((2, 2), dtype=complex)
    for block in _environment_blocks(rho):
        result += block
    return result


def _raw_trace(matrix):
    value = complex(np.trace(matrix))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("a density trace is nonfinite")
    return value


def _linear_matrix(blocks):
    """Only the fixed system's 4x4 column-vectorized Kraus map."""
    result = np.zeros((4, 4), dtype=complex)
    for block in blocks:
        result += np.kron(block.conj(), block)
    return result


def _apply_linear(matrix, rho):
    return (matrix @ rho.reshape(4, order="F")).reshape((2, 2), order="F")


def independent_reference(work):
    """Exactly eight Pauli exponential calls and two exponent-one powers.

    The sole 4x4 semigroup exponential is deliberately deferred until the end
    of run_check(), after all four density stages. nu_j and u_j denote the
    same normalized amplitude sqrt(w_j/W), not a second window convention.
    """
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    hamiltonian = 0.6 * z + 0.8 * x
    signed = (0, 1, -2, -1)
    dt = BETA * math.sqrt(2 * math.pi / 4) / 2
    dw = 2 * math.sqrt(2 * math.pi / 4) / BETA
    weights = [math.exp(-math.pi * index * index / 4) for index in signed]
    mass = math.fsum(weights)
    amplitudes = np.array([math.sqrt(weight / mass) for weight in weights], dtype=float)
    eta = mass / 2
    gamma = np.array([math.exp(-max(BETA * dw * index + 0.5, 0.0))
                      for index in signed], dtype=float)
    alpha = eta * float(np.max(gamma))
    probabilities = DELTA * eta * gamma
    if not 0 < float(np.min(probabilities)) <= float(np.max(probabilities)) <= 1:
        raise ValueError("the fixed weak reference has invalid dilation probabilities")

    root_angle = 2 * math.atan2(
        math.sqrt(float(amplitudes[2] ** 2 + amplitudes[3] ** 2)),
        math.sqrt(float(amplitudes[0] ** 2 + amplitudes[1] ** 2)))
    low_angles = (2 * math.atan2(float(amplitudes[1]), float(amplitudes[0])),
                  2 * math.atan2(float(amplitudes[3]), float(amplitudes[2])))
    branch = np.zeros((4, 4), dtype=complex)
    branch[:2, :2], branch[2:, 2:] = _ry(low_angles[0]), _ry(low_angles[1])
    prepare = branch @ np.kron(_ry(root_angle), identity)

    intervals = []
    for interval in (dt, -2 * dt):
        product = identity.copy()
        # Source chronology is Z,X,X,Z; left multiplication composes that
        # chronology. The high-bit interval is later than the low-bit one.
        for coefficient, pauli in ((0.6, z), (0.8, x), (0.8, x), (0.6, z)):
            exponential = _array(work.expm(-1j * coefficient * interval * pauli / 2),
                                 (2, 2), "independent signed Pauli exponential")
            product = exponential @ product
        intervals.append(_array(work.matrix_power(product, 1), (2, 2),
                                "independent signed interval power"))
    evolution = (identity, intervals[0], intervals[1], intervals[1] @ intervals[0])
    selected = np.zeros((8, 8), dtype=complex)
    for j, product in enumerate(evolution):
        selected[2 * j:2 * j + 2, 2 * j:2 * j + 2] = product.conj().T @ z @ product
    fourier = np.array([[complex(math.cos(-2 * math.pi * l * j / 4),
                                 math.sin(-2 * math.pi * l * j / 4)) / 2
                         for j in range(4)] for l in range(4)], dtype=complex)
    unitary = np.kron(fourier, identity) @ selected @ np.kron(prepare, identity)
    filtered = np.array([unitary[2 * l:2 * l + 2, :2] for l in range(4)])
    survival = np.zeros((8, 8), dtype=complex)
    for l, probability in enumerate(probabilities):
        survival[2 * l, 2 * l] = survival[2 * l + 1, 2 * l + 1] = math.sqrt(1 - float(probability))
    nonjump_columns = unitary.conj().T @ survival @ unitary[:, :2]
    weak = np.empty((8, 2, 2), dtype=complex)
    for label in range(4):
        weak[label] = nonjump_columns[2 * label:2 * label + 2]
        weak[4 + label] = math.sqrt(float(probabilities[label])) * filtered[label]
    decay = np.zeros((2, 2), dtype=complex)
    for rate, block in zip(gamma, filtered):
        decay += eta * float(rate) * block.conj().T @ block

    energy_basis = _ry(math.atan2(0.8, 0.6))
    energy_decay = energy_basis.conj().T @ decay @ energy_basis
    energies = (1.0, -1.0)
    energy_correction = np.empty((2, 2), dtype=complex)
    for row, energy_row in enumerate(energies):
        for column, energy_column in enumerate(energies):
            energy_correction[row, column] = (
                0.5j * math.tanh(BETA * (energy_row - energy_column) / 4)
                * energy_decay[row, column])
    correction = energy_basis @ energy_correction @ energy_basis.conj().T
    c0, cx, cy, cz = _qubit_coefficients(correction)
    c_norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    a = alpha * math.tanh(BETA / 2) / 4
    success = (identity - 1j * DELTA * correction) / (1 + DELTA * a)
    success_probability = (1 + DELTA ** 2 * c_norm ** 2) / (1 + DELTA * a) ** 2
    success_floor = (1 + DELTA * a) ** -2
    accepted_unitary = success / math.sqrt(success_probability)
    correction_multiplier = DELTA if c_norm == 0 else math.sin(DELTA * c_norm) / c_norm
    exact_correction = math.cos(DELTA * c_norm) * identity - 1j * correction_multiplier * correction
    angle = math.atan(DELTA * c_norm)
    angle_multiplier = DELTA if c_norm == 0 else math.sin(angle) / c_norm
    angle_evolution = math.cos(angle) * identity - 1j * angle_multiplier * correction
    phi_factors = np.array([accepted_unitary @ block for block in weak])
    psi_factors = np.array([exact_correction @ block for block in weak])
    phi = _linear_matrix(phi_factors)
    psi = _linear_matrix(psi_factors)
    generator = (-1j * (np.kron(identity, correction) - np.kron(correction.T, identity))
                 - 0.5 * (np.kron(identity, decay) + np.kron(decay.T, identity)))
    for rate, block in zip(gamma, filtered):
        generator += eta * float(rate) * np.kron(block.conj(), block)
    work.checkpoint()
    return {
        "identity": identity, "x": x, "y": y, "z": z,
        "hamiltonian": hamiltonian, "prepare": prepare, "fourier": fourier,
        "access_unitary": unitary, "intervals": np.array(intervals),
        "clock_products": np.array(evolution), "filtered_blocks": filtered,
        "survival": survival, "weak_kraus": weak, "decay": decay,
        "energy_basis": energy_basis, "energy_decay": energy_decay,
        "energy_correction": energy_correction, "correction": correction,
        "correction_coefficients": (c0, cx, cy, cz), "c_norm": c_norm, "a": a,
        "success_block": success, "accepted_unitary": accepted_unitary,
        "exact_correction": exact_correction, "angle_evolution": angle_evolution,
        "success_probability": success_probability, "success_floor": success_floor,
        "phi": phi, "phi_squared": phi @ phi,
        "psi": psi, "psi_squared": psi @ psi, "finite_generator": generator,
        "gibbs": (identity - math.tanh(BETA) * hamiltonian) / 2,
        "mesh": {"dt": dt, "dw": dw, "signed_indices": list(signed),
                 "times": [dt * index for index in signed],
                 "frequencies": [dw * index for index in signed],
                 "weights": weights, "window_mass": mass,
                 "amplitudes_nu_equals_u": amplitudes.tolist(), "eta": eta,
                 "gamma": gamma.tolist(), "alpha": alpha},
    }


def _bundle_guard(bundle):
    """Driver/builder supply the separate literal chronology validation."""
    if not isinstance(bundle, dict) or set(bundle) != set(BUNDLE_LAYOUT):
        raise ValueError("the fixed reference requires the complete five-source bundle")
    for name, (width, classical_width) in BUNDLE_LAYOUT.items():
        source = bundle[name]
        if source.num_qubits != width or source.num_clbits != classical_width:
            raise ValueError("the reference source layout differs: " + name)
    for item in bundle["correction_lcu"].data:
        if item.operation.name in ("measure", "reset", "if_else"):
            raise ValueError("the density correction stage must be reset-free and unmeasured")


def _basis_weak_kraus(circuit, work):
    columns = np.empty((16, 2), dtype=complex)
    for system_bit in range(2):
        initial = np.zeros(16, dtype=complex)
        initial[system_bit] = 1
        columns[:, system_bit] = _array(work.statevector(initial, circuit), (16,),
                                        "actual weak basis-column amplitudes")
    return columns, columns.reshape((8, 2, 2))


def _factorized_terminal(weak, correction):
    """Contract small factors instead of enumerating 2,296 terminal products."""
    accepted = np.array([correction[0] @ block for block in weak])
    failure_effect = _gram(correction[1:])
    first_failure = _pullback(weak, failure_effect)
    first_success = _gram(accepted)
    second_failure = _pullback(accepted, first_failure)
    two_successes = _pullback(accepted, first_success)
    resets = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=complex)
    measurements = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=complex)
    reset_effect = _gram(resets)
    measurement_effect = _gram(measurements)
    reset_failure = _pullback(weak, _pullback(correction[1:], reset_effect))
    reset_late_failure = _pullback(accepted, reset_failure)
    measured_accepted_effect = _pullback(accepted, _pullback(accepted, measurement_effect))
    return {
        "accepted_one_round_factors": accepted,
        "reset_factors": resets, "readout_factors": measurements,
        "early_failure_effect": first_failure,
        "late_failure_effect": second_failure,
        "two_success_effect": two_successes,
        "pre_cleanup_completeness": first_failure + second_failure + two_successes,
        "post_abort_reset_completeness": reset_failure + reset_late_failure + two_successes,
        "full_readout_completeness": reset_failure + reset_late_failure + measured_accepted_effect,
    }


def _retry_formulas(reference):
    """Leave N/R and their total workload unevaluated; no samples are drawn."""
    b = (DELTA * reference["a"]) ** 3 / 3
    return {
        "scope": "ideal correction-only two-round finite-output proposal",
        "operator_error_bound_formula": "b=(delta*a)^3/3",
        "ideal_operator_error_bound_scalar": b,
        "two_round_trace_norm_and_z_bias_bound_formula": "4*b",
        "ideal_two_round_bias_bound_scalar": 4 * b,
        "epsilon_sampling_formula": "0.1-4*b",
        "epsilon_sampling_scalar": 0.1 - 4 * b,
        "delta_sampling": 0.04, "delta_heralding": 0.01,
        "trajectory_success_floor_formula": "p_C_low^2; p_C_low=(1+delta*a)^(-2)",
        "ideal_trajectory_success_floor_scalar": reference["success_floor"] ** 2,
        "N_formula": "ceil(2*log(2/0.04)/(0.1-4*b)^2)",
        "R_formula": "ceil(log(0.01/N)/log(1-p_C_low^2))",
        "maximum_whole_attempts_formula": "N*R",
        "maximum_primitive_work_formula": "N*R*6049",
        "exhaustion_failure_union_bound_formula": "N*(1-p_C_low^2)^R <= 0.01",
        "N_evaluated": False, "R_evaluated": False,
        "whole_known_input_restart_required": True,
        "unknown_intermediate_recovery_implemented": False,
        "sampling_performed": False, "floating_point_bounds_certified": False,
        "weak_step_splitting_mesh_or_gibbs_error_included": False,
        "synthesis_or_hardware_error_included": False,
    }


def run_check(bundle, saved_correction_blocks, work, tolerances=None):
    """Run only the future separately authorized, frozen fixed-toy check.

    The driver authenticates the saved 32-block baseline and all source bytes,
    validates literal source chronology, and reserves every scientific work
    call. All normalization uses the observed zero-block trace; auxiliary
    cleanup is an algebraic embedding and never a system reset. The final
    scientific call is the sole 4x4 exponential, a diagnostic without a
    Lindblad/Gibbs accuracy or mixing acceptance threshold.
    """
    tolerances = _tolerances(tolerances)
    _bundle_guard(bundle)
    saved = _array(saved_correction_blocks, (32, 2, 2), "authenticated correction Kraus blocks")
    reference = independent_reference(work)
    identity, z = reference["identity"], reference["z"]
    weak = reference["weak_kraus"]
    weak_columns, actual_weak = _basis_weak_kraus(bundle["weak_dilation"], work)
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    reference_phi_final = _apply_linear(reference["phi_squared"], rho0)
    reference_psi_final = _apply_linear(reference["psi_squared"], rho0)
    terminal = _factorized_terminal(weak, saved)
    diagnostics = {
        "full_prepare_completion_unitarity_error": _max_entry(reference["prepare"].conj().T @ reference["prepare"] - np.eye(4)),
        "prepare_initial_column_error": _max_entry(reference["prepare"][:, 0] - np.array(reference["mesh"]["amplitudes_nu_equals_u"])),
        "full_access_unitarity_error": _max_entry(reference["access_unitary"].conj().T @ reference["access_unitary"] - np.eye(8)),
        "filtered_isometry_error": _max_entry(_gram(reference["filtered_blocks"]) - identity),
        "reference_weak_kraus_completeness_error": _max_entry(_gram(weak) - identity),
        "actual_weak_kraus_entry_error": _max_entry(actual_weak - weak),
        "actual_weak_kraus_completeness_error": _max_entry(_gram(actual_weak) - identity),
        "actual_weak_basis_columns_isometry_error": _max_entry(weak_columns.conj().T @ weak_columns - identity),
        "nonjump_effect_vs_identity_minus_delta_d_error": _max_entry(_gram(weak[:4]) - identity + DELTA * reference["decay"]),
        "jump_effect_vs_delta_d_error": _max_entry(_gram(weak[4:]) - DELTA * reference["decay"]),
        "energy_basis_hamiltonian_error": _max_entry(reference["energy_basis"].conj().T @ reference["hamiltonian"] @ reference["energy_basis"] - z),
        "correction_hermiticity_error": _max_entry(reference["correction"] - reference["correction"].conj().T),
        "correction_trace_magnitude": abs(_raw_trace(reference["correction"])),
        "correction_square_error": _max_entry(reference["correction"] @ reference["correction"] - reference["c_norm"] ** 2 * identity),
        "closed_c_norm_vs_energy_offdiagonal_error": abs(reference["c_norm"] - abs(reference["energy_correction"][0, 1])),
        "accepted_unitary_vs_analytic_angle_error": _max_entry(reference["accepted_unitary"] - reference["angle_evolution"]),
        "saved_success_block_vs_energy_reference_error": _max_entry(saved[0] - reference["success_block"]),
        "saved_correction_completeness_error": _max_entry(_gram(saved) - identity),
        "saved_success_gram_vs_ideal_p_error": _max_entry(saved[0].conj().T @ saved[0] - reference["success_probability"] * identity),
        "terminal_pre_cleanup_completeness_error": _max_entry(terminal["pre_cleanup_completeness"] - identity),
        "terminal_post_abort_reset_completeness_error": _max_entry(terminal["post_abort_reset_completeness"] - identity),
        "terminal_full_readout_completeness_error": _max_entry(terminal["full_readout_completeness"] - identity),
        "physical_alpha_vs_retained_scalar_error": abs(reference["mesh"]["alpha"] - ACCEPTED_ALPHA),
        "source_eta_vs_independent_error": abs(_finite_real(bundle["weak_dilation"].metadata["eta"], "source eta") - reference["mesh"]["eta"]),
        "source_alpha_vs_independent_error": abs(_finite_real(bundle["weak_dilation"].metadata["alpha"], "source alpha") - reference["mesh"]["alpha"]),
        "source_gamma_vs_independent_error": _max_entry(_array(bundle["weak_dilation"].metadata["gamma"], (4,), "source gamma") - np.array(reference["mesh"]["gamma"])),
    }
    density_records = []
    round_records = []
    all_blocks = []
    actual_system = rho0
    cumulative = 1.0
    independent_system = rho0
    for round_index in range(2):
        # The system input is unknown/mixed after round one. No H or q0
        # reset appears in either continuing density stage.
        embedded = _system_embedding(actual_system)
        weak_output = _array(work.density(embedded, bundle["weak_reset_prefix"]),
                             (64, 64), "actual weak reset-prefix output")
        weak_marginal = _system_marginal(weak_output)
        reference_weak = _channel(weak, actual_system)
        correction_output = _array(work.density(weak_output, bundle["correction_lcu"]),
                                   (64, 64), "actual reset-free correction output")
        blocks = _environment_blocks(correction_output)
        traces = [_raw_trace(block) for block in blocks]
        probabilities = np.array([trace.real for trace in traces], dtype=float)
        success = float(probabilities[0])
        if not 0 < success <= 1 + tolerances["analytic"]:
            raise ValueError("the observed zero-herald trace is not positive and bounded")
        predictions = np.array([block @ reference_weak @ block.conj().T for block in saved])
        expected_traces = np.array([_raw_trace(block).real for block in predictions], dtype=float)
        normalized = blocks[0] / success
        predicted_success = float(expected_traces[0])
        if predicted_success <= 0:
            raise ValueError("the factorized zero-herald prediction has no positive trace")
        predicted_normalized = predictions[0] / predicted_success
        independent_system = reference["accepted_unitary"] @ _channel(weak, independent_system) @ reference["accepted_unitary"].conj().T
        prefix = "round_" + str(round_index + 1) + "_"
        diagnostics.update({
            prefix + "weak_system_entry_error": _max_entry(weak_marginal - reference_weak),
            prefix + "weak_full_reset_embedding_error": _max_entry(weak_output - _system_embedding(reference_weak)),
            prefix + "weak_trace_error": abs(_raw_trace(weak_output) - 1),
            prefix + "weak_hermiticity_error": _max_entry(weak_output - weak_output.conj().T),
            prefix + "all_32_correction_blocks_error": _max_entry(blocks - predictions),
            prefix + "all_32_probabilities_error": _max_entry(probabilities - expected_traces),
            prefix + "probability_sum_error": abs(float(np.sum(probabilities)) - 1),
            prefix + "probability_imaginary_part_error": max(abs(trace.imag) for trace in traces),
            prefix + "correction_hermiticity_error": _max_entry(correction_output - correction_output.conj().T),
            prefix + "accepted_vs_factorized_normalized_error": _max_entry(normalized - predicted_normalized),
            prefix + "accepted_vs_ideal_phi_error": _max_entry(normalized - independent_system),
            prefix + "success_vs_ideal_p_error": abs(success - reference["success_probability"]),
            prefix + "accepted_trace_error": abs(_raw_trace(normalized) - 1),
            prefix + "accepted_hermiticity_error": _max_entry(normalized - normalized.conj().T),
        })
        density_records.extend([
            {"round": round_index + 1, "stage": "weak_reset_prefix",
             "input_trace": {"real": _raw_trace(embedded).real, "imag": _raw_trace(embedded).imag},
             "raw_output": matrix_record(weak_output)},
            {"round": round_index + 1, "stage": "reset_free_correction_lcu",
             "input_trace": {"real": _raw_trace(weak_output).real, "imag": _raw_trace(weak_output).imag},
             "raw_output": matrix_record(correction_output)},
        ])
        round_records.append({
            "round": round_index + 1,
            "incoming_trajectory_probability": cumulative,
            "zero_herald_trace": success,
            "unconditional_success_probability": cumulative * success,
            "weak_input_system": matrix_record(actual_system),
            "actual_weak_system": matrix_record(weak_marginal),
            "independent_weak_system_on_actual_input": matrix_record(reference_weak),
            "normalized_accepted_system": matrix_record(normalized),
            "ideal_phi_system_after_this_round": matrix_record(independent_system),
            "all_32_outcomes": [
                {"herald_z": label, "label_q5_to_q1": format(label, "05b"),
                 "success": label == 0,
                 "conditional_probability_given_round_reached": float(probabilities[label]),
                 "probability_imaginary_part_raw": traces[label].imag,
                 "unconditional_probability": cumulative * float(probabilities[label]),
                 "raw_system_block": matrix_record(blocks[label]),
                 "factorized_reference_block": matrix_record(predictions[label])}
                for label in range(32)],
        })
        all_blocks.append(blocks)
        cumulative *= success
        actual_system = normalized
        work.checkpoint()

    p1, p2 = round_records[0]["zero_herald_trace"], round_records[1]["zero_herald_trace"]
    early_failure = np.sum(all_blocks[0][1:], axis=0)
    late_failure = p1 * np.sum(all_blocks[1][1:], axis=0)
    accepted_subnormalized = p1 * all_blocks[1][0]
    failed_reset = _channel(terminal["reset_factors"], early_failure + late_failure)
    accepted_readout = _channel(terminal["readout_factors"], accepted_subnormalized)
    z_mean = float(_raw_trace(z @ actual_system).real)
    target_z_mean = float(_raw_trace(z @ reference_psi_final).real)
    correction_bias = 4 * (DELTA * reference["a"]) ** 3 / 3
    diagnostics.update({
        "trajectory_success_vs_ideal_p_squared_error": abs(p1 * p2 - reference["success_probability"] ** 2),
        "terminal_probability_sum_error": abs(_raw_trace(early_failure + late_failure + accepted_subnormalized) - 1),
        "terminal_post_reset_readout_trace_error": abs(_raw_trace(failed_reset + accepted_readout) - 1),
        "accepted_final_vs_ideal_phi_squared_error": _max_entry(actual_system - reference_phi_final),
        "ideal_phi_composition_consistency_error": _max_entry(independent_system - reference_phi_final),
        "ideal_phi_vs_psi_squared_trace_norm_diagnostic": _qubit_trace_norm(reference_phi_final - reference_psi_final),
        "ideal_phi_vs_psi_difference_hermiticity_error": _max_entry((reference_phi_final - reference_psi_final) - (reference_phi_final - reference_psi_final).conj().T),
        "actual_z_vs_finite_psi_squared_bias_diagnostic": abs(z_mean - target_z_mean),
        "actual_z_mean": z_mean, "finite_psi_squared_target_z_mean": target_z_mean,
        "ideal_two_round_correction_bias_bound_scalar": correction_bias,
        "c_norm_raw": reference["c_norm"], "a_raw": reference["a"],
        "c_norm_minus_a_raw": reference["c_norm"] - reference["a"],
        "decay_analytic_qubit_eigenvalues_raw": list(_qubit_spectrum(reference["decay"])),
        "accepted_final_analytic_qubit_eigenvalues_raw": list(_qubit_spectrum(actual_system)),
        "actual_final_system_mixedness_diagnostic": 1 - float(_raw_trace(actual_system @ actual_system).real),
        "optional_gibbs_state_residual_frobenius_diagnostic": _frobenius(reference_psi_final - reference["gibbs"]),
        "finite_generator_gibbs_residual_frobenius_diagnostic": _frobenius(_apply_linear(reference["finite_generator"], reference["gibbs"])),
    })
    # The only remaining scientific work call: fixed 4x4 finite semigroup.
    semigroup = _array(work.expm(2 * DELTA * reference["finite_generator"]),
                       (4, 4), "finite generator semigroup diagnostic")
    semigroup_output = _apply_linear(semigroup, rho0)
    diagnostics.update({
        "phi_squared_vs_finite_semigroup_frobenius_diagnostic": _frobenius(reference["phi_squared"] - semigroup),
        "psi_squared_vs_finite_semigroup_frobenius_diagnostic": _frobenius(reference["psi_squared"] - semigroup),
        "psi_squared_state_vs_finite_semigroup_frobenius_diagnostic": _frobenius(reference_psi_final - semigroup_output),
        "finite_semigroup_state_trace_error": abs(_raw_trace(semigroup_output) - 1),
    })
    checks = {}
    for name, residual in diagnostics.items():
        if name.endswith("_error"):
            category = ("completeness" if any(token in name for token in
                                              ("completeness", "isometry", "unitarity", "_gram_"))
                        else "analytic" if "probability" in name or "trace" in name
                        or "success_vs_ideal_p" in name or "trajectory_success" in name
                        else "entry")
            checks[name] = bool(residual <= tolerances[category])
    checks["ideal_c_norm_within_scale"] = reference["c_norm"] <= reference["a"] + tolerances["analytic"]
    checks["ideal_correction_trace_zero"] = diagnostics["correction_trace_magnitude"] <= tolerances["entry"]
    checks["reference_decay_positive_and_bounded_analytic_diagnostic"] = (
        diagnostics["decay_analytic_qubit_eigenvalues_raw"][0] >= -tolerances["analytic"]
        and diagnostics["decay_analytic_qubit_eigenvalues_raw"][1] <= reference["mesh"]["alpha"] + tolerances["analytic"])
    checks["accepted_final_positive_analytic_diagnostic"] = diagnostics["accepted_final_analytic_qubit_eigenvalues_raw"][0] >= -tolerances["analytic"]
    checks["ideal_phi_vs_psi_within_correction_only_bound_plus_tolerance"] = diagnostics["ideal_phi_vs_psi_squared_trace_norm_diagnostic"] <= correction_bias + tolerances["analytic"]
    checks["actual_z_vs_psi_within_correction_only_bound_plus_tolerance"] = diagnostics["actual_z_vs_finite_psi_squared_bias_diagnostic"] <= correction_bias + tolerances["analytic"]
    failed = [name for name, passed in checks.items() if not passed]
    work.checkpoint()
    return {
        "status": "checked_fixed_finite_two_round_toy" if not failed else "failed_fixed_finite_two_round_toy",
        "scope": "fixed finite tilted weak channel composed with two heralded correction steps; no Gibbs preparation or mixing contract",
        "checks_passed": not failed, "failed_checks": failed, "checks": checks,
        "tolerances": tolerances, "declared_work_calls": dict(DECLARED_WORK_CALLS),
        "parameters": {"terms": [[coefficient, label] for coefficient, label in TERMS],
                       "proposals": list(PROPOSALS), "beta": BETA,
                       "clock_bits": CLOCK_BITS, "product_steps": PRODUCT_STEPS,
                       "delta": DELTA, "tau": DELTA, "rounds": ROUNDS,
                       "initial": "known_plus", "system_wire": 0,
                       "weak_clock_wires": [1, 2], "weak_flag_wire": 3,
                       "reserved_total_wires": 6, "idle_weak_selector_wires": [4, 5],
                       "herald_auxiliary_wires": [1, 2, 3, 4, 5]},
        "mesh": reference["mesh"],
        "normalization_convention": "nu_j=u_j=sqrt(w_j/W); full prefix-RY completion, eta retained",
        "floating_diagnostics": diagnostics,
        "ideal_bias_retry_projection": _retry_formulas(reference),
        "actual_round_probability_weights": {"p1": p1, "p2_given_round2_reached": p2,
                                             "p1_times_p2": p1 * p2,
                                             "ideal_p_C_squared_diagnostic": reference["success_probability"] ** 2},
        "rounds": round_records,
        "raw_density_stages": density_records,
        "all_8_weak_kraus_blocks": [
            {"environment_label": label, "clock_label": label % 4,
             "weak_flag": label // 4, "branch": "nonjump" if label < 4 else "jump",
             "discarded_without_postselection": True,
             "actual_matrix": matrix_record(actual_weak[label]),
             "independent_matrix": matrix_record(weak[label])}
            for label in range(8)],
        "all_32_authenticated_correction_blocks": [
            {"herald_z": label, "label_q5_to_q1": format(label, "05b"),
             "success": label == 0, "matrix": matrix_record(saved[label]),
             "dependence": "authenticated retained gate-simulation data; not freshly independently reconstructed"}
            for label in range(32)],
        "terminal_factorization": {
            "accepted_factor_formula": "A_j2*A_j1; A_j=F_0*Q_j",
            "early_failure_factor_formula": "F_z1*Q_j1; z1=1..31",
            "late_failure_factor_formula": "F_z2*Q_j2*A_j1; z2=1..31",
            "weak_index_range": [0, 7], "failure_index_range": [1, 31],
            "pre_cleanup_label_counts": {"accepted": 64, "early_failure": 248,
                                         "late_failure": 1984, "total": 2296},
            "post_abort_reset_before_success_readout_label_count": 4528,
            "full_terminal_readout_label_count": 4592,
            "minimal_kraus_rank_claimed": False, "terminal_paths_enumerated": False,
            "reset_left_factor_formula": "R_s=|0><s|; s=0,1, only on failure",
            "success_readout_left_factor_formula": "M_y=|y><y|; y=0,1",
            "reset_factors": matrix_record(terminal["reset_factors"]),
            "readout_factors": matrix_record(terminal["readout_factors"]),
            "round2_reached_rule": "c1==0",
            "accepted_rule": "round2_reached and c2==0",
            "unused_c2_and_z_bits_are_invalid": True,
            "failed_attempt_has_z_output": False,
            "retry_rule": "reset/reprepare whole known-plus input; no same-round unknown-state retry",
        },
        "terminal_states": {
            "early_failure_pre_cleanup": matrix_record(early_failure),
            "late_failure_pre_cleanup_unconditional": matrix_record(late_failure),
            "two_successes_subnormalized": matrix_record(accepted_subnormalized),
            "failed_abort_reset_output": matrix_record(failed_reset),
            "successful_z_readout_output": matrix_record(accepted_readout),
            "successful_readout_outcome_weights": [float(accepted_subnormalized[y, y].real) for y in range(2)],
            "normalized_accepted_before_z": matrix_record(actual_system),
        },
        "matrices": {
            "full_prepare_completion": matrix_record(reference["prepare"]),
            "full_access_unitary": matrix_record(reference["access_unitary"]),
            "signed_intervals": matrix_record(reference["intervals"]),
            "clock_products_preserving_low_then_high_order": matrix_record(reference["clock_products"]),
            "filtered_B_blocks": matrix_record(reference["filtered_blocks"]),
            "survival_matrix": matrix_record(reference["survival"]),
            "physical_decay_D": matrix_record(reference["decay"]),
            "energy_basis": matrix_record(reference["energy_basis"]),
            "energy_decay": matrix_record(reference["energy_decay"]),
            "energy_correction_Eq19": matrix_record(reference["energy_correction"]),
            "physical_correction_C": matrix_record(reference["correction"]),
            "reference_success_T": matrix_record(reference["success_block"]),
            "reference_accepted_unitary": matrix_record(reference["accepted_unitary"]),
            "analytic_exact_C_evolution": matrix_record(reference["exact_correction"]),
            "actual_weak_basis_columns": matrix_record(weak_columns),
            "reference_phi_squared_4x4": matrix_record(reference["phi_squared"]),
            "reference_psi_squared_4x4": matrix_record(reference["psi_squared"]),
            "reference_phi_squared_state": matrix_record(reference_phi_final),
            "reference_psi_squared_target_state": matrix_record(reference_psi_final),
            "finite_generator_4x4": matrix_record(reference["finite_generator"]),
            "finite_semigroup_4x4_diagnostic": matrix_record(semigroup),
            "finite_semigroup_state_diagnostic": matrix_record(semigroup_output),
            "optional_analytic_gibbs_reference": matrix_record(reference["gibbs"]),
            "terminal_pre_cleanup_effect_sum": matrix_record(terminal["pre_cleanup_completeness"]),
            "terminal_post_abort_reset_effect_sum": matrix_record(terminal["post_abort_reset_completeness"]),
            "terminal_full_readout_effect_sum": matrix_record(terminal["full_readout_completeness"]),
        },
        "c_norm_closed_formula": {"coefficients_identity_x_y_z_raw": list(reference["correction_coefficients"]),
                                  "formula": "sqrt(c_x^2+c_y^2+c_z^2) for ideal traceless Hermitian C",
                                  "spectral_norm_convention": True, "diagonalization_performed": False},
        "diagnostic_exclusions": ["weak-step and splitting error have no certified coefficients",
                                 "finite mesh/Trotter/continuum discrepancy is not a mixing or Gibbs bound",
                                 "4b covers correction approximation only",
                                 "raw floating agreement is not a rounding/synthesis/hardware certificate"],
        **SCOPE_FLAGS,
        "numerical_checks_performed": True,
        "finite_toy_numerical_checks_performed": True,
        "finite_two_round_numerical_checks_performed": True,
    }
