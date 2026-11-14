# SPDX-License-Identifier: MIT
"""Pure boundary contract for one unexecuted finite thermal composition.

This module imports no scientific target and materializes no circuit. Source
implementation and future floating checks do not establish a thermal sampler.
"""
from __future__ import annotations

import math

TERMS = ((0.6, "Z"), (0.8, "X"))
PROPOSALS = ("Z",)
BETA = 1.0
CLOCK_BITS = 2
PRODUCT_STEPS = 1
DELTA = TAU = 0.125
ROUNDS = 2
M = 4
J = 1
MAX_COMPONENT_OPERATIONS = 5_000
MAX_PROGRAM_OPERATIONS = 6_500
MAX_SOURCE_WIRES = 6
CLASSICAL_BITS = 11
EXPECTED_SOURCES = {
    "weak_dilation": (4, 0, 989),
    "weak_reset_prefix": (6, 0, 997),
    "correction_lcu": (6, 0, 2016),
    "round_macro": (6, 5, 3023),
    "two_round_trial": (6, 11, 6051),
}
EXPECTED_PATH_OPERATIONS = {"early_abort": 3026, "late_abort": 6049,
                            "accepted": 6049}
SCOPE_FLAGS = dict.fromkeys((
    "numerical_checks_performed", "finite_toy_numerical_checks_performed",
    "coherent_correction_implemented", "coherent_correction_evolution_implemented",
    "general_coherent_correction_implemented", "general_channel_implemented",
    "channel_implemented", "sampler_implemented", "exact_finite_kms",
    "finite_mesh_exact_kms", "mixing_certificate", "rotations_synthesized",
    "physical_resources_computed", "formal_proof_reproduced",
    "floating_point_error_certified", "hardware_error_certified",
    "all_components_verified", "new_algorithm_result", "random_sampling_performed",
), False)


def _fixed_real(value, expected, name):
    if (type(value) not in (int, float) or value != expected
            or not math.isfinite(value)):
        raise ValueError(f"{name} must equal the fixed finite input")


def validate_fixed_input(*, terms=TERMS, proposals=PROPOSALS, beta=BETA,
                         clock_bits=CLOCK_BITS, product_steps=PRODUCT_STEPS,
                         delta=DELTA, rounds=ROUNDS):
    """Reject nonfixed inputs, booleans, reordered terms and changed scales."""
    if (type(terms) is not tuple or len(terms) != 2
            or any(type(term) is not tuple or len(term) != 2 for term in terms)):
        raise ValueError("terms must be the fixed Z-then-X tuple")
    for (coefficient, label), (expected, expected_label) in zip(terms, TERMS):
        _fixed_real(coefficient, expected, "coefficient")
        if type(label) is not str or label != expected_label:
            raise ValueError("terms must preserve Z-then-X order")
    if (type(proposals) is not tuple or len(proposals) != 1
            or type(proposals[0]) is not str or proposals != PROPOSALS):
        raise ValueError("the only proposal is Z")
    for value, expected, name in ((beta, BETA, "beta"),
                                  (delta, DELTA, "delta")):
        _fixed_real(value, expected, name)
    for value, expected, name in ((clock_bits, CLOCK_BITS, "clock_bits"),
                                  (product_steps, PRODUCT_STEPS, "product_steps"),
                                  (rounds, ROUNDS, "rounds")):
        if type(value) is not int or value != expected:
            raise ValueError(f"{name} must equal the fixed finite input")
    return {"terms": TERMS, "proposals": PROPOSALS, "beta": BETA,
            "clock_bits": CLOCK_BITS, "product_steps": PRODUCT_STEPS,
            "delta": DELTA, "rounds": ROUNDS}


def validate_counts(*, static_operations, maximum_path_operations,
                    quantum_wires, classical_bits, component_operations):
    """Check caps before relying on predicted or walked program counts."""
    for name, value, cap in (
        ("static_operations", static_operations, MAX_PROGRAM_OPERATIONS),
        ("maximum_path_operations", maximum_path_operations, MAX_PROGRAM_OPERATIONS),
        ("quantum_wires", quantum_wires, MAX_SOURCE_WIRES),
        ("classical_bits", classical_bits, CLASSICAL_BITS),
    ):
        if type(value) is not int or not 0 < value <= cap:
            raise ValueError(f"{name} exceeds its fixed cap or has wrong type")
    if quantum_wires != MAX_SOURCE_WIRES or classical_bits != CLASSICAL_BITS:
        raise ValueError("the conditional program reserves six wires and eleven bits")
    if type(component_operations) is not dict or not component_operations:
        raise ValueError("component counts must be a nonempty mapping")
    for name, count in component_operations.items():
        if (type(name) is not str or type(count) is not int
                or not 0 < count <= MAX_COMPONENT_OPERATIONS):
            raise ValueError("a component exceeds the original 5000-operation guard")
    return {"static_operations": static_operations,
            "maximum_path_operations": maximum_path_operations,
            "quantum_wires": quantum_wires, "classical_bits": classical_bits,
            "component_operations": dict(component_operations)}


def classify_record(c1, c2, z_bit=None):
    """Interpret only reached heralds/readout; default-zero unused bits are invalid.

    c1/c2 integers encode qk at bit k-1. Failed attempts have no Z output.
    This is deterministic record processing, not a sampler or retry executor.
    """
    for name, value in (("c1", c1), ("c2", c2)):
        if type(value) is not int or not 0 <= value < 32:
            raise ValueError(f"{name} must be a five-bit integer")
    if z_bit is not None and (type(z_bit) is not int or z_bit not in (0, 1)):
        raise ValueError("z_bit must be absent or a binary integer")
    reached = c1 == 0
    accepted = reached and c2 == 0
    if accepted and z_bit is None:
        raise ValueError("an accepted trajectory must have a final Z readout")
    return {"round2_reached": reached, "round2_herald_valid": reached,
            "accepted": accepted, "z_readout_valid": accepted,
            "z_value": 1 - 2 * z_bit if accepted else None,
            "terminal_event": "accepted" if accepted else
                              ("late_abort" if reached else "early_abort"),
            "retry": "restart whole known-plus trajectory",
            "failed_intermediate_recovery_implemented": False}


def sampling_proposal():
    """Unevaluated ideal two-round budget; no old single-step N/R reuse."""
    return {"target": "Tr(Z Psi^2(|+><+|)); Psi=Ad(exp(-i delta C)) o E_delta",
            "epsilon_total": 0.1, "delta_sampling": 0.04, "delta_heralding": 0.01,
            "b": "(delta*a)^3/3", "correction_only_trace_and_Z_bias": "4*b",
            "epsilon_sampling": "0.1-4*b > 0",
            "N": "ceil(2*ln(2/0.04)/(0.1-4*b)^2)",
            "p_C_low": "(1+delta*a)^(-2)",
            "p_trajectory_low": "p_C_low^2",
            "R": "ceil(ln(0.01/N)/ln(1-p_C_low^2))",
            "maximum_whole_attempts": "N*R", "maximum_primitives": "N*R*6049",
            "exhaustion": "explicit failure, no fabricated Z output",
            "sampling_performed": False, "numerical_repetitions_evaluated": False,
            "weak_splitting_mesh_error_certified": False}


def terminal_label_counts():
    return {"accepted_before_Z": 64, "early_abort_before_reset": 248,
            "late_abort_before_reset": 1984, "pre_cleanup_total": 2296,
            "post_abort_reset_before_success_readout": 4528,
            "full_terminal_readout": 4592, "minimal_kraus_rank_claimed": False}
