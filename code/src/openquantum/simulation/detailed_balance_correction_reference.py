# SPDX-License-Identifier: MIT
"""Independent references for one fixed finite correction toy, not a sampler.

The signed Fourier/product-formula decay is rebuilt independently. Correction
entries use Chen et al., Nature 646 (2025), DOI 10.1038/s41586-025-09583-x,
Eq. 19 in the known energy basis, not the production commutator expression.
All five Statevector evolutions, twelve tiny matrix exponentials and two
interval matrix powers go through the caller's bounded work interface.

Importing this module performs no circuit construction or scientific check.
The ideal bias/retry formulas are mathematical projections; floating-point
agreement does not certify synthesis, hardware, KMS, mixing or full evolution.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

TERMS = ((0.6, "Z"), (0.8, "X"))
PROPOSALS = ("Z",)
BETA = 1.0
CLOCK_BITS = 2
PRODUCT_STEPS = 1
TAU = 0.125
ACCEPTED_ALPHA = 0.9775450868978823
MAX_WIRES = 6
MAX_AMPLITUDES = 64
MAX_SOURCE_OPERATIONS = 5000
DEFAULT_TOLERANCES = {"entry": 1e-10, "completeness": 1e-10, "analytic": 1e-9}
EXPECTED_WORK_CALLS = {"statevector_evolutions": 5, "tiny_expm_calls": 12,
                       "interval_matrix_power_calls": 2}
EXPECTED_SOURCES = {
    "centered_decay": (4, 213, 0),
    "correction_block": (5, 228, 0),
    "controlled_correction": (6, 2013, 0),
    "heralded_step": (6, 2016, 0),
    "known_plus_prefix": (6, 2017, 0),
    "known_plus_trial": (6, 2029, 6),
}
UNITARY_PRIMITIVES = frozenset({"h", "x", "p", "ry", "rz", "cx",
                                 "s", "sdg", "t", "tdg"})


def _finite_real(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(label + " must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(label + " must be a finite real scalar") from error
    if not math.isfinite(result):
        raise ValueError(label + " must be a finite real scalar")
    return result


def _matrix(value, shape, label):
    result = np.asarray(value, dtype=complex)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(label + " has a wrong shape or nonfinite entries")
    return result


def _max_entry(matrix):
    return float(np.max(np.abs(matrix)))


def matrix_record(matrix):
    """Retain every raw real/imaginary entry without clipping or phase fitting."""
    value = np.asarray(matrix, dtype=complex)
    if value.ndim > 3 or value.size > 256 or not np.all(np.isfinite(value)):
        raise ValueError("raw reference array exceeds its fixed shape/finite cap")
    return {"shape": list(value.shape), "real": value.real.tolist(),
            "imag": value.imag.tolist()}


def _mesh():
    """Physical mesh mass is dt times sum(f_j squared), computed independently."""
    size = 4
    signed = np.array([0, 1, -2, -1], dtype=int)
    sigma = 1.0 / BETA
    dt = math.sqrt(2 * math.pi / size) / (2 * sigma)
    dw = 2 * sigma * math.sqrt(2 * math.pi / size)
    times = signed.astype(float) * dt
    frequencies = signed.astype(float) * dw
    sampled_f = np.exp(-(sigma * times) ** 2) * math.sqrt(
        sigma * math.sqrt(2 / math.pi))
    physical_squared_sum = float(np.dot(sampled_f, sampled_f))
    eta = dt * physical_squared_sum
    gamma = np.array([math.exp(-max(BETA * omega + 0.5, 0.0))
                      for omega in frequencies], dtype=float)
    gamma_max = float(np.max(gamma))
    alpha = eta * gamma_max
    if (physical_squared_sum <= 0 or eta <= 0 or gamma_max <= 0 or alpha <= 0
            or not all(math.isfinite(value) for value in
                       (physical_squared_sum, eta, gamma_max, alpha))
            or not np.all(np.isfinite(gamma)) or np.any(gamma < 0)):
        raise ValueError("fixed independent mesh has invalid normalization")
    return {"M": size, "signed_indices": signed, "dt": dt, "dw": dw,
            "times": times, "frequencies": frequencies, "sampled_f": sampled_f,
            "physical_squared_sum": physical_squared_sum, "eta": eta,
            "gamma": gamma, "gamma_max": gamma_max, "alpha": alpha,
            "amplitudes": sampled_f / math.sqrt(physical_squared_sum)}


def _fourier_decay(conjugates, mesh):
    """Independent physical signed Fourier sum for the one Z proposal."""
    blocks = np.zeros((4, 2, 2), dtype=complex)
    for frequency_index, frequency in enumerate(mesh["frequencies"]):
        for time_index, time in enumerate(mesh["times"]):
            blocks[frequency_index] += (
                np.exp(-1j * frequency * time) * mesh["amplitudes"][time_index]
                * conjugates[time_index] / math.sqrt(4))
    decay = np.zeros((2, 2), dtype=complex)
    isometry = np.zeros((2, 2), dtype=complex)
    for block, rate in zip(blocks, mesh["gamma"]):
        product = block.conj().T @ block
        isometry += product
        decay += mesh["eta"] * rate * product
    return {"blocks": blocks, "decay": decay, "isometry": isometry}


def _energy_basis_correction(decay, basis):
    """Eq. 19 entry by entry in energy order (+1,-1), never a commutator."""
    energy_decay = basis.conj().T @ decay @ basis
    energies = (1.0, -1.0)
    energy_correction = np.empty((2, 2), dtype=complex)
    for row, energy_row in enumerate(energies):
        for column, energy_column in enumerate(energies):
            energy_correction[row, column] = (
                0.5j * math.tanh(BETA * (energy_row - energy_column) / 4)
                * energy_decay[row, column])
    return {"energy_decay": energy_decay, "energy_correction": energy_correction,
            "correction": basis @ energy_correction @ basis.conj().T}


def independent_reference(work):
    """Rebuild both finite decays once using exactly twelve/two work calls.

    work.expm(matrix) and work.matrix_power(matrix, steps) must reserve their
    whole-run budget before the actual library call. No existing producer's
    reference routine is invoked here, so the twelve calls are explicit.
    """
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    hamiltonian = 0.6 * z + 0.8 * x
    term_matrices = ((0.6, z), (0.8, x))
    mesh = _mesh()
    intervals = []
    for clock_bit in range(2):
        interval = mesh["dt"] * (1 if clock_bit == 0 else -2)
        one_step = identity.copy()
        for coefficient, matrix in (*term_matrices, *reversed(term_matrices)):
            exponential = _matrix(
                work.expm(-1j * coefficient * interval * matrix / 2),
                (2, 2), "finite-Trotter Pauli exponential")
            one_step = exponential @ one_step
        intervals.append(_matrix(work.matrix_power(one_step, 1), (2, 2),
                                 "finite-Trotter interval power"))
    finite_conjugates = np.empty((4, 2, 2), dtype=complex)
    exact_conjugates = np.empty((4, 2, 2), dtype=complex)
    for time_index, time in enumerate(mesh["times"]):
        evolution = identity.copy()
        for clock_bit, interval in enumerate(intervals):
            if (time_index >> clock_bit) & 1:
                evolution = interval @ evolution
        finite_conjugates[time_index] = evolution.conj().T @ z @ evolution
        exact_evolution = _matrix(work.expm(-1j * hamiltonian * time), (2, 2),
                                  "exact-H finite exponential")
        exact_conjugates[time_index] = exact_evolution.conj().T @ z @ exact_evolution
    finite = _fourier_decay(finite_conjugates, mesh)
    exact = _fourier_decay(exact_conjugates, mesh)
    theta = math.atan2(0.8, 0.6)
    basis = np.array([[math.cos(theta / 2), -math.sin(theta / 2)],
                      [math.sin(theta / 2), math.cos(theta / 2)]], dtype=complex)
    finite.update(_energy_basis_correction(finite["decay"], basis))
    exact.update(_energy_basis_correction(exact["decay"], basis))
    correction = finite["correction"]
    c_norm = float(np.linalg.norm(correction, ord=2))
    a = mesh["alpha"] * math.tanh(BETA / 2) / 4
    if not math.isfinite(c_norm) or not math.isfinite(a) or a <= 0:
        raise ValueError("fixed correction requires a finite nonnegative norm and positive scale")
    denominator = 1 + TAU * a
    target_block = (identity - 1j * TAU * correction) / denominator
    probability = (1 + TAU * TAU * c_norm * c_norm) / (denominator * denominator)
    accepted_unitary = (identity - 1j * TAU * correction) / math.sqrt(
        1 + TAU * TAU * c_norm * c_norm)
    if c_norm == 0:
        exact_correction_evolution = identity.copy()
        angle_evolution = identity.copy()
    else:
        exact_correction_evolution = (math.cos(TAU * c_norm) * identity
                                      - 1j * math.sin(TAU * c_norm) * correction / c_norm)
        angle = math.atan(TAU * c_norm)
        angle_evolution = (math.cos(angle) * identity
                           - 1j * math.sin(angle) * correction / c_norm)
    return {"finite": finite, "exact_h_finite": exact, "mesh": mesh,
            "hamiltonian": hamiltonian, "energy_basis": basis,
            "c_norm": c_norm, "a": a, "target_block": target_block,
            "accepted_unitary": accepted_unitary, "angle_evolution": angle_evolution,
            "exact_correction_evolution": exact_correction_evolution,
            "success_probability": probability,
            "success_probability_floor": denominator ** -2,
            "declared_work_calls": dict(EXPECTED_WORK_CALLS)}


def _instruction_record(circuit, instruction):
    operation = instruction.operation
    return (operation.name,
            tuple(_finite_real(value, "primitive angle") for value in operation.params),
            tuple(circuit.find_bit(bit).index for bit in instruction.qubits),
            tuple(circuit.find_bit(bit).index for bit in instruction.clbits))


def _source_records(sources):
    """Validate every frozen source and trial boundary before scientific work."""
    if not isinstance(sources, dict) or set(sources) != set(EXPECTED_SOURCES):
        raise ValueError("fixed source bundle has missing or extra members")
    correction_phase = _finite_real(sources["correction_block"].global_phase,
                                    "correction source global phase")
    relative_phase_operations = int(correction_phase != 0)
    phase_lifted_sources = {"controlled_correction", "heralded_step",
                            "known_plus_prefix", "known_plus_trial"}
    records = {}
    for name, (width, zero_phase_count, classical_width) in EXPECTED_SOURCES.items():
        expected_count = zero_phase_count + (relative_phase_operations
                                             if name in phase_lifted_sources else 0)
        source = sources[name]
        if (source.num_qubits != width or source.num_qubits > MAX_WIRES
                or source.num_clbits != classical_width
                or len(source.data) != expected_count
                or len(source.data) > MAX_SOURCE_OPERATIONS):
            raise ValueError("fixed source width/count differs: " + name)
        instructions = [_instruction_record(source, item) for item in source.data]
        names = {item[0] for item in instructions}
        allowed = UNITARY_PRIMITIVES | ({"reset", "measure"} if classical_width else set())
        if names - allowed:
            raise ValueError("source contains an opaque or unsupported operation: " + name)
        metadata = source.metadata
        if not isinstance(metadata, dict):
            raise ValueError("fixed source requires its physical-normalization metadata")
        normalization = {key: _finite_real(metadata[key], "source " + key)
                         for key in ("eta", "alpha", "gamma_max", "correction_scale", "tau")}
        if (any(normalization[key] <= 0 for key in normalization)
                or normalization["tau"] != TAU
                or metadata.get("system_wires") != [0]
                or metadata.get("clock_wires") != [1, 2]
                or metadata.get("weight_wire") != 3
                or metadata.get("selected_auxiliary_state") != "all_zero"):
            raise ValueError("fixed source normalization/input/selection contract differs")
        records[name] = {"width": width, "classical_width": classical_width,
                         "operations": expected_count,
                         "global_phase": _finite_real(source.global_phase, "source global phase"),
                         "primitive_counts": dict(Counter(item[0] for item in instructions)),
                         "physical_normalization": normalization,
                         "instructions": instructions}
    lcu = records["heralded_step"]["instructions"]
    controlled = records["controlled_correction"]["instructions"]
    prefix = records["known_plus_prefix"]["instructions"]
    trial = records["known_plus_trial"]["instructions"]
    expected_relative_phase = [("p", (correction_phase,), (5,), ())] if relative_phase_operations else []
    phase_slice = sources["controlled_correction"].metadata.get("stage_slices", {}).get("source_global_phase")
    if (records["controlled_correction"]["global_phase"] != 0
            or phase_slice != [0, relative_phase_operations]
            or controlled[:relative_phase_operations] != expected_relative_phase):
        raise ValueError("controlled lift must turn the full source global phase into an explicit relative P")
    angle = 2 * math.atan(math.sqrt(
        TAU * records["heralded_step"]["physical_normalization"]["correction_scale"]))
    expected_lcu = [("ry", (angle,), (5,), ()), *controlled,
                    ("p", (-math.pi / 2,), (5,), ()),
                    ("ry", (-angle,), (5,), ())]
    if lcu != expected_lcu:
        raise ValueError("LCU must preserve the complete controlled source and explicit relative phase")
    input_h = ("h", (), (0,), ())
    resets = [("reset", (), (wire,), ()) for wire in range(6)]
    measurements = [("measure", (), (wire,), (wire,)) for wire in range(6)]
    if prefix != [input_h, *lcu] or trial != [*resets, input_h, *lcu, *measurements]:
        raise ValueError("known-plus trial must reset/prepare/run/measure every wire in order")
    if (records["known_plus_prefix"]["global_phase"] != records["heralded_step"]["global_phase"]
            or records["known_plus_trial"]["global_phase"] != records["heralded_step"]["global_phase"]):
        raise ValueError("known-input composition dropped a source global phase")
    expected_kc = {"h": 46, "ry": 18, "rz": 65, "cx": 88, "x": 4, "p": 7}
    if records["correction_block"]["primitive_counts"] != expected_kc:
        raise ValueError("fixed correction primitive inventory differs")
    records["controlled_correction"]["controlled_source_global_phase"] = correction_phase
    records["controlled_correction"]["relative_global_phase_operations"] = relative_phase_operations
    return records


def _tolerances(value):
    if value is None:
        value = DEFAULT_TOLERANCES
    if not isinstance(value, dict) or set(value) != set(DEFAULT_TOLERANCES):
        raise ValueError("tolerance keys must be entry/completeness/analytic")
    result = {name: _finite_real(number, "tolerance " + name)
              for name, number in value.items()}
    if any(number <= 0 or number > DEFAULT_TOLERANCES[name]
           for name, number in result.items()):
        raise ValueError("fixed tolerances may not be nonpositive or relaxed")
    return result


def _column_vectors(circuit, work):
    dimension = 1 << circuit.num_qubits
    if circuit.num_qubits > MAX_WIRES or dimension > MAX_AMPLITUDES or circuit.num_clbits:
        raise ValueError("coherent basis-column evolution exceeds its fixed cap")
    columns = np.empty((dimension, 2), dtype=complex)
    for column in range(2):
        initial = np.zeros(dimension, dtype=complex)
        initial[column] = 1.0
        columns[:, column] = _matrix(work.evolve(initial, circuit), (dimension,),
                                     "actual basis-column amplitudes")
    return columns


def _ideal_retry_projection(reference, full_trial_operations):
    """Evaluate ideal formulas; rounded scalars are not certified error budgets."""
    a, c_norm = reference["a"], reference["c_norm"]
    b = (TAU * a) ** 3 / 3
    epsilon_total, delta_sampling, delta_heralding = 0.1, 0.04, 0.01
    epsilon_sampling = epsilon_total - 2 * b
    p, p_floor = reference["success_probability"], reference["success_probability_floor"]
    if not (0 < epsilon_sampling < epsilon_total and 0 < p_floor < 1 and 0 < p <= 1):
        raise ValueError("ideal fixed retry projection has invalid scalars")
    accepted_outputs = math.ceil(2 * math.log(2 / delta_sampling) / epsilon_sampling ** 2)
    max_attempts = math.ceil(math.log(delta_heralding / accepted_outputs)
                             / math.log1p(-p_floor))
    if accepted_outputs < 1 or max_attempts < 1:
        raise ValueError("ideal fixed retry projection requires positive integer counts")
    trials = accepted_outputs * max_attempts
    return {
        "scope": "ideal mathematical one-step/known-input projection evaluated with floating scalars",
        "operator_error_bound_formula": "b=(tau*a)^3/3",
        "operator_error_bound": b,
        "pauli_mean_and_full_trace_norm_bias_bound_formula": "2*b",
        "pauli_mean_bias_bound": 2 * b,
        "exact_ideal_angle_error": TAU * c_norm - math.atan(TAU * c_norm),
        "epsilon_total": epsilon_total, "epsilon_sampling": epsilon_sampling,
        "delta_sampling": delta_sampling, "delta_heralding": delta_heralding,
        "delta_total": delta_sampling + delta_heralding,
        "success_probability": p, "success_probability_floor": p_floor,
        "accepted_outputs_projection": accepted_outputs,
        "max_attempts_per_output_projection": max_attempts,
        "worst_case_complete_trials_projection": trials,
        "full_trial_operations": full_trial_operations,
        "worst_case_primitive_operations_projection": trials * full_trial_operations,
        "expected_truncated_attempts_per_requested_output": (1 - (1 - p) ** max_attempts) / p,
        "heralding_union_bound_evaluation": accepted_outputs * (1 - p_floor) ** max_attempts,
        "N_formula": "ceil(2*log(2/delta_sampling)/epsilon_sampling^2)",
        "R_formula": "ceil(log(delta_heralding/N)/log(1-p_floor))",
        "unknown_input_recovery_implemented": False,
        "retry_contract": "reset all six wires and reprepare the known plus input after failure; exhaustion is explicit failure",
        "trajectory_chaining_contract": "K chained successes require whole-trajectory restart with p^K or a separate recovery construction",
        "sampling_performed": False,
        "floating_point_bounds_certified": False,
        "synthesis_or_hardware_error_included": False,
    }


def check_sources(sources, work, tolerances=None):
    """Run the fixed five-vector check after the separate source/run freeze.

    The caller enforces the whole-run time/RSS/output and call budgets. Every
    source and numerical tolerance is validated before any reference work.
    Returned flags describe only this executed finite toy, never a general
    correction, channel, sampler, mixing, physical estimate or formal result.
    """
    tolerances = _tolerances(tolerances)
    source_records = _source_records(sources)
    reference = independent_reference(work)
    correction_columns = _column_vectors(sources["correction_block"], work)
    lcu_columns = _column_vectors(sources["heralded_step"], work)
    # q0 is the least-significant system bit: rows 2*z and 2*z+1 share
    # the complete auxiliary outcome z in q1..q5. There are all 32 blocks.
    kraus = lcu_columns.reshape(32, 2, 2)
    actual_correction_block = correction_columns[:2, :]
    actual_success = kraus[0]
    initial_zero = np.zeros(64, dtype=complex)
    initial_zero[0] = 1.0
    plus_vector = _matrix(work.evolve(initial_zero, sources["known_plus_prefix"]),
                          (64,), "actual known-plus amplitudes")
    plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
    identity = np.eye(2, dtype=complex)
    z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    finite, exact, mesh = reference["finite"], reference["exact_h_finite"], reference["mesh"]
    correction, c_norm, a = finite["correction"], reference["c_norm"], reference["a"]
    completeness = np.zeros((2, 2), dtype=complex)
    predicted_plus = np.empty((32, 2), dtype=complex)
    for outcome, block in enumerate(kraus):
        completeness += block.conj().T @ block
        predicted_plus[outcome] = block @ plus
    plus_by_outcome = plus_vector.reshape(32, 2)
    outcome_probabilities = np.array([float(np.vdot(row, row).real)
                                      for row in plus_by_outcome], dtype=float)
    predicted_outcome_probabilities = np.array([float(np.vdot(row, row).real)
                                                for row in predicted_plus], dtype=float)
    success_probability = float(outcome_probabilities[0])
    if success_probability <= 0 or not math.isfinite(success_probability):
        raise ValueError("actual all-zero success probability must be finite and positive")
    actual_accepted = plus_by_outcome[0] / math.sqrt(success_probability)
    reference_accepted = reference["accepted_unitary"] @ plus
    exact_target = reference["exact_correction_evolution"] @ plus
    actual_z_mean = float(np.vdot(actual_accepted, z_matrix @ actual_accepted).real)
    reference_z_mean = float(np.vdot(reference_accepted, z_matrix @ reference_accepted).real)
    exact_z_mean = float(np.vdot(exact_target, z_matrix @ exact_target).real)
    actual_accepted_map = actual_success / math.sqrt(reference["success_probability"])
    ideal_projection = _ideal_retry_projection(
        reference, source_records["known_plus_trial"]["operations"])
    diagnostics = {
        "actual_correction_vs_energy_reference_max_entry_error": _max_entry(actual_correction_block - correction / a),
        "actual_success_vs_reference_max_entry_error": _max_entry(actual_success - reference["target_block"]),
        "all_32_kraus_completeness_max_entry_error": _max_entry(completeness - identity),
        "actual_correction_column_isometry_max_entry_error": _max_entry(correction_columns.conj().T @ correction_columns - identity),
        "actual_lcu_column_isometry_max_entry_error": _max_entry(lcu_columns.conj().T @ lcu_columns - identity),
        "known_plus_linearity_max_entry_error": _max_entry(plus_by_outcome - predicted_plus),
        "all_32_outcome_probability_max_error": _max_entry(outcome_probabilities - predicted_outcome_probabilities),
        "known_plus_probability_sum_error": abs(float(np.sum(outcome_probabilities)) - 1),
        "known_plus_success_probability_error": abs(success_probability - reference["success_probability"]),
        "actual_success_gram_vs_input_independent_probability_max_entry_error": _max_entry(actual_success.conj().T @ actual_success - reference["success_probability"] * identity),
        "actual_conditional_map_vs_reference_max_entry_error": _max_entry(actual_accepted_map - reference["accepted_unitary"]),
        "known_plus_conditional_z_mean_error": abs(actual_z_mean - reference_z_mean),
        "known_plus_z_mean_vs_exact_target_error": abs(actual_z_mean - exact_z_mean),
        "actual_conditional_map_vs_exact_target_operator_error": float(np.linalg.norm(actual_accepted_map - reference["exact_correction_evolution"], ord=2)),
        "reference_accepted_vs_angle_formula_max_entry_error": _max_entry(reference["accepted_unitary"] - reference["angle_evolution"]),
        "reference_correction_hermiticity_max_entry_error": _max_entry(correction - correction.conj().T),
        "reference_correction_trace_magnitude": float(abs(np.trace(correction))),
        "reference_correction_square_max_entry_error": _max_entry(correction @ correction - c_norm ** 2 * identity),
        "energy_basis_hamiltonian_max_entry_error": _max_entry(reference["energy_basis"].conj().T @ reference["hamiltonian"] @ reference["energy_basis"] - z_matrix),
        "finite_fourier_isometry_max_entry_error": _max_entry(finite["isometry"] - identity),
        "exact_h_fourier_isometry_max_entry_error": _max_entry(exact["isometry"] - identity),
        "physical_alpha_vs_accepted_error": abs(mesh["alpha"] - ACCEPTED_ALPHA),
        "physical_eta_source_metadata_max_error": max(abs(record["physical_normalization"]["eta"] - mesh["eta"])
                                                       for record in source_records.values()),
        "physical_alpha_source_metadata_max_error": max(abs(record["physical_normalization"]["alpha"] - mesh["alpha"])
                                                         for record in source_records.values()),
        "physical_gamma_max_source_metadata_max_error": max(abs(record["physical_normalization"]["gamma_max"] - mesh["gamma_max"])
                                                             for record in source_records.values()),
        "physical_correction_scale_source_metadata_max_error": max(abs(record["physical_normalization"]["correction_scale"] - a)
                                                                  for record in source_records.values()),
        "c_norm": c_norm,
        "c_norm_definition": "spectral/operator norm; largest singular value, equal to largest absolute eigenvalue for ideal Hermitian C",
        "c_frobenius_norm_diagnostic": float(np.linalg.norm(correction, ord="fro")),
        "correction_scale_a": a,
        "c_norm_minus_a_raw": c_norm - a,
        "finite_decay_eigenvalues_raw": np.linalg.eigvalsh(finite["decay"]).tolist(),
        "finite_decay_vs_exact_h_finite_frobenius_error": float(np.linalg.norm(finite["decay"] - exact["decay"], ord="fro")),
        "finite_correction_vs_exact_h_finite_frobenius_error": float(np.linalg.norm(correction - exact["correction"], ord="fro")),
        "known_plus_actual_z_mean": actual_z_mean,
        "known_plus_reference_conditional_z_mean": reference_z_mean,
        "known_plus_exact_correction_target_z_mean": exact_z_mean,
        "known_plus_success_probability": success_probability,
        "known_plus_total_failure_probability": float(np.sum(outcome_probabilities[1:])),
        "known_plus_unconditional_z_mean": float(sum(abs(row[0]) ** 2 - abs(row[1]) ** 2 for row in plus_by_outcome)),
    }
    entry_fields = (
        "actual_correction_vs_energy_reference_max_entry_error",
        "actual_success_vs_reference_max_entry_error",
        "actual_correction_column_isometry_max_entry_error",
        "actual_lcu_column_isometry_max_entry_error",
        "known_plus_linearity_max_entry_error",
        "all_32_outcome_probability_max_error",
        "actual_conditional_map_vs_reference_max_entry_error",
        "reference_accepted_vs_angle_formula_max_entry_error",
        "reference_correction_hermiticity_max_entry_error",
        "reference_correction_trace_magnitude",
        "reference_correction_square_max_entry_error",
        "energy_basis_hamiltonian_max_entry_error",
        "finite_fourier_isometry_max_entry_error",
        "exact_h_fourier_isometry_max_entry_error",
        "physical_alpha_vs_accepted_error",
        "physical_eta_source_metadata_max_error",
        "physical_alpha_source_metadata_max_error",
        "physical_gamma_max_source_metadata_max_error",
        "physical_correction_scale_source_metadata_max_error",
    )
    checks = {name: diagnostics[name] <= tolerances["entry"] for name in entry_fields}
    checks["all_32_kraus_completeness"] = diagnostics["all_32_kraus_completeness_max_entry_error"] <= tolerances["completeness"]
    checks["input_independent_success_gram"] = diagnostics["actual_success_gram_vs_input_independent_probability_max_entry_error"] <= tolerances["completeness"]
    for name in ("known_plus_probability_sum_error", "known_plus_success_probability_error",
                 "known_plus_conditional_z_mean_error"):
        checks[name] = diagnostics[name] <= tolerances["analytic"]
    checks["spectral_norm_within_ideal_scale"] = c_norm <= a + tolerances["analytic"]
    checks["actual_conditional_operator_error_within_ideal_bound_plus_numerical_tolerance"] = (
        diagnostics["actual_conditional_map_vs_exact_target_operator_error"]
        <= ideal_projection["operator_error_bound"] + tolerances["analytic"])
    checks["actual_z_mean_error_within_ideal_bias_plus_numerical_tolerance"] = (
        diagnostics["known_plus_z_mean_vs_exact_target_error"]
        <= ideal_projection["pauli_mean_bias_bound"] + tolerances["analytic"])
    checks["ideal_retry_projection_positive"] = (
        ideal_projection["accepted_outputs_projection"] > 0
        and ideal_projection["max_attempts_per_output_projection"] > 0)
    failed_checks = [name for name, passed in checks.items() if not passed]
    passed = not failed_checks
    for value in diagnostics.values():
        if type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite raw floating diagnostic")
    return {
        "status": "checked_fixed_finite_correction_toy" if passed else "failed_fixed_finite_correction_toy",
        "scope": "actual finite-Trotter C access and one known-input heralded correction step; all outcomes retained",
        "checks_passed": passed, "failed_checks": failed_checks,
        "checks": checks, "tolerances": tolerances,
        "parameters": {"terms": [[coefficient, label] for coefficient, label in TERMS],
                       "proposals": list(PROPOSALS), "beta": BETA, "clock_bits": CLOCK_BITS,
                       "product_steps": PRODUCT_STEPS, "tau": TAU,
                       "system_wire": 0, "auxiliary_wires": [1, 2, 3, 4, 5],
                       "success_outcome": "00000"},
        "source_records": {name: {key: value for key, value in record.items() if key != "instructions"}
                           for name, record in source_records.items()},
        "declared_work_calls": dict(EXPECTED_WORK_CALLS),
        "floating_diagnostics": diagnostics,
        "ideal_bias_retry_projection": ideal_projection,
        "all_outcome_probabilities": [{"auxiliary_outcome_integer": outcome,
                                        "auxiliary_label_q5_to_q1": format(outcome, "05b"),
                                        "probability": float(probability),
                                        "predicted_from_kraus_probability": float(predicted_outcome_probabilities[outcome]),
                                        "success": outcome == 0}
                                       for outcome, probability in enumerate(outcome_probabilities)],
        "all_32_kraus_blocks": [{"auxiliary_outcome_integer": outcome,
                                "auxiliary_label_q5_to_q1": format(outcome, "05b"),
                                "matrix": matrix_record(block), "success": outcome == 0}
                               for outcome, block in enumerate(kraus)],
        "matrices": {"finite_trotter_decay": matrix_record(finite["decay"]),
                     "exact_h_finite_decay_diagnostic": matrix_record(exact["decay"]),
                     "energy_basis": matrix_record(reference["energy_basis"]),
                     "finite_energy_decay": matrix_record(finite["energy_decay"]),
                     "finite_energy_correction": matrix_record(finite["energy_correction"]),
                     "finite_correction": matrix_record(correction),
                     "exact_h_finite_correction_diagnostic": matrix_record(exact["correction"]),
                     "actual_correction_all_zero_block": matrix_record(actual_correction_block),
                     "actual_success_block": matrix_record(actual_success),
                     "reference_success_block": matrix_record(reference["target_block"]),
                     "all_kraus_completeness": matrix_record(completeness),
                     "reference_accepted_unitary": matrix_record(reference["accepted_unitary"]),
                     "exact_finite_correction_evolution": matrix_record(reference["exact_correction_evolution"]),
                     "actual_correction_basis_columns": matrix_record(correction_columns),
                     "actual_lcu_basis_columns": matrix_record(lcu_columns),
                     "actual_known_plus_amplitudes": matrix_record(plus_vector)},
        "physical_normalization": {"eta": mesh["eta"], "gamma_max": mesh["gamma_max"],
                                   "alpha": mesh["alpha"], "definition": "alpha=eta*gamma_max",
                                   "physical_squared_sum": mesh["physical_squared_sum"],
                                   "dt": mesh["dt"], "dw": mesh["dw"],
                                   "times": mesh["times"].tolist(),
                                   "frequencies": mesh["frequencies"].tolist(),
                                   "gamma": mesh["gamma"].tolist(),
                                   "physical_eta_independently_checked": True},
        "numerical_checks_performed": True,
        "finite_toy_correction_access_checked": passed,
        "finite_toy_heralded_step_checked": passed,
        "all_components_verified": False, "new_algorithm_result": False,
        "coherent_correction_implemented": False,
        "coherent_correction_evolution_implemented": False,
        "general_channel_implemented": False, "sampler_implemented": False,
        "exact_finite_kms": False, "mixing_certificate": False,
        "rotations_synthesized": False, "physical_resources_computed": False,
        "formal_proof_reproduced": False, "random_sampling_performed": False,
        "floating_point_bounds_certified": False,
        "synthesis_or_hardware_error_certified": False,
    }
