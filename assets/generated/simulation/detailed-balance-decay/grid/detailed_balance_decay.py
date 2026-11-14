# SPDX-License-Identifier: MIT
"""Original primitive block access to finite weighted decay, not evolution.

The selected block is D_fin/alpha only when every clock, proposal, weight and
selector auxiliary is compressed to zero. All rotations are source angles;
this module performs no numerical block check, synthesis or Gibbs sampling.
Pauli character i acts on qi; q0 is the least significant state bit.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

from qiskit import QuantumCircuit

if __package__:
    from . import detailed_balance_access as access
else:
    import detailed_balance_access as access

ACCESS_SOURCE_SHA256 = "7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77"
MAX_PRIMITIVE_OPERATIONS = 100_000
PRIMITIVES = access.PRIMITIVES
MILESTONE = "primitive_weighted_decay_block_only"


def _access_source_pin() -> str:
    """The mirrored bound is valid only for the reviewed access source."""
    source = Path(access.__file__).resolve()
    if source.suffix != ".py" or not source.is_file():
        raise ValueError("decay access needs the reviewed Python source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != ACCESS_SOURCE_SHA256:
        raise ValueError("access source differs from the reviewed bound/source pin")
    return digest


def _weight_normalization(eta: float, rates: Sequence[float]) -> dict:
    """Validate finite scales before division; never clip or normalize zero."""
    eta = access._real(eta, "eta")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if isinstance(rates, (str, bytes)):
        raise ValueError("rates must be a nonempty finite nonnegative sequence")
    try:
        values = [access._real(value, "rate") for value in rates]
    except TypeError as error:
        raise ValueError("rates must be a nonempty finite nonnegative sequence") from error
    if not values or any(value < 0 for value in values):
        raise ValueError("rates must be a nonempty finite nonnegative sequence")
    gamma_max = max(values)
    if gamma_max <= 0:
        raise ValueError("gamma_max must be positive; zero normalization is unsupported")
    alpha = eta * gamma_max
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    normalized = [value / gamma_max for value in values]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in normalized):
        raise ValueError("normalized rates must be finite in [0, 1]")
    if any(value > 0 and scaled == 0 for value, scaled in zip(values, normalized)):
        raise ValueError("a positive normalized rate underflows")
    angles = [2 * math.asin(math.sqrt(value)) for value in normalized]
    return {"eta": eta, "gamma": values, "gamma_max": gamma_max,
            "alpha": alpha, "normalized_rates": normalized, "weight_angles": angles}


def preflight_decay(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                    *, beta: float = 1, clock_bits: int = 2,
                    product_steps: int = 1) -> dict:
    """JSON-safe parameters/layout/scales/bounds without circuit allocation.

    Input parsing and the bounded O(M) mesh are classical preprocessing.
    The entire doubled oracle is capped before any QuantumCircuit is made.
    """
    digest = _access_source_pin()
    terms, proposals, n = access._labels(terms, proposals)
    r = access._integer(product_steps, "product_steps", 1, access.MAX_PRODUCT_STEPS)
    mesh = access.gaussian_mesh(clock_bits, beta)
    m, size = mesh["clock_bits"], mesh["M"]
    scales = _weight_normalization(mesh["eta"], mesh["gamma"])
    if len(scales["gamma"]) != size:
        raise ValueError("frequency rate count differs from the clock size")
    p = (len(proposals) - 1).bit_length()
    access_width = n + m + p
    weight, selector = access_width, access_width + 1
    # Mirrors build_gaussian_access at ACCESS_SOURCE_SHA256, checked above.
    access_bound = (4 * size + 5 * m * m + 120 * len(proposals) * n
                    + 2 * m * r * sum(12 * len(label) + 12 for _, label in terms))
    bound = 2 * access_bound + 4 * size + 6
    if bound > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("decay preflight exceeds 100000 primitive operations")
    return {
        "terms": [[coefficient, label] for coefficient, label in terms],
        "proposals": list(proposals), "n_system": n, "clock_bits": m,
        "proposal_bits": p, "proposal_count": len(proposals), "product_steps": r,
        "beta": mesh["beta"], "M": size, "access_width": access_width,
        "system_wires": list(range(n)), "clock_wires": list(range(n, n + m)),
        "proposal_wires": list(range(n + m, access_width)),
        "weight_wire": weight, "selector_wire": selector,
        "auxiliary_wires": list(range(n, selector + 1)), "total_wires": selector + 1,
        "selected_auxiliary_state": "all_zero", "access_source_sha256": digest,
        "access_preflight_upper_bound": access_bound, "preflight_upper_bound": bound,
        "dt": mesh["dt"], "dw": mesh["dw"],
        "time_mesh": mesh["times"].tolist(), "frequency_mesh": mesh["frequencies"].tolist(),
        "physical_squared_sum": mesh["physical_squared_sum"], **scales,
    }


def _guard(circuit: QuantumCircuit) -> None:
    if len(circuit.data) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("decay source exceeds 100000 primitive operations")
    if circuit.num_clbits or set(circuit.count_ops()) - PRIMITIVES:
        raise ValueError("decay block source requires only the declared unitary primitives")
    try:
        phases = [float(circuit.global_phase)]
        phases.extend(float(value) for instruction in circuit.data
                      for value in instruction.operation.params)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("decay block source requires finite numeric angles and phase") from error
    if any(not math.isfinite(value) for value in phases):
        raise ValueError("decay block source requires finite numeric angles and phase")


def build_decay_block(terms: Sequence[tuple[float, str]], proposals: Sequence[str],
                      *, beta: float = 1, clock_bits: int = 2,
                      product_steps: int = 1) -> QuantumCircuit:
    """Emit H_s, U, R_w, CZ_sw, P_s(pi), R_w inverse, U inverse, H_s.

    U and its full inverse are unconditional. The all-zero auxiliary block
    is D_fin/(eta*gamma_max); compressing selector zero alone is insufficient.
    System input and zero auxiliary initialization are supplied by the caller.
    """
    plan = preflight_decay(terms, proposals, beta=beta, clock_bits=clock_bits,
                           product_steps=product_steps)
    unitary = access.build_gaussian_access(plan["terms"], plan["proposals"],
                                          beta=plan["beta"], clock_bits=plan["clock_bits"],
                                          product_steps=plan["product_steps"])
    source_count = len(unitary.data)
    if source_count > plan["access_preflight_upper_bound"]:
        raise AssertionError("access stream exceeds its reviewed conservative bound")
    rotations = QuantumCircuit(plan["total_wires"], name="Frequency weight RY")
    access._multiplexed_ry(rotations, plan["clock_wires"], plan["weight_wire"],
                           plan["weight_angles"])
    if len(rotations.data) != 2 * plan["M"]:
        raise AssertionError("weight multiplexor violates the reviewed primitive count")
    circuit = QuantumCircuit(plan["total_wires"], name="Finite weighted decay block")
    stages = {}

    def stage(name, action):
        start = len(circuit.data)
        action()
        stages[name] = [start, len(circuit.data)]

    weight, selector = plan["weight_wire"], plan["selector_wire"]
    access_wires = list(range(plan["access_width"]))
    stage("selector_open", lambda: circuit.h(selector))
    stage("access_forward", lambda: circuit.compose(unitary, access_wires, inplace=True))
    stage("weight_forward", lambda: circuit.compose(rotations, inplace=True))
    start = len(circuit.data)
    circuit.h(weight)
    circuit.cx(selector, weight)
    circuit.h(weight)
    stages["cz_primitives"] = [start, len(circuit.data)]
    stage("selector_negative_phase", lambda: circuit.p(math.pi, selector))
    stage("weight_inverse", lambda: circuit.compose(rotations.inverse(), inplace=True))
    stage("access_inverse", lambda: circuit.compose(unitary.inverse(), access_wires, inplace=True))
    stage("selector_close", lambda: circuit.h(selector))
    expected = 2 * source_count + 4 * plan["M"] + 6
    if len(circuit.data) != expected or expected > plan["preflight_upper_bound"]:
        raise AssertionError("decay stream violates the reviewed doubled count/bound")
    _guard(circuit)
    upstream = unitary.metadata
    resource_ledger = {
        "access_copies": 2, "access_instructions_each_copy": source_count,
        "access_instructions_total": 2 * source_count,
        "proposal_prepare_h_instructions_total": 2 * plan["proposal_bits"],
        "gaussian_prepare_instructions_total": 2 * upstream["gaussian_prepare_instructions"],
        "pauli_select_instructions_total": 2 * upstream["select_instructions"],
        "signed_evolution_instructions_total": 4 * upstream["signed_evolution_instructions_each_direction"],
        "dft_instructions_total": 2 * upstream["dft_instructions"],
        "requested_signed_interval_magnitude_all_access_directions":
            2 * upstream["requested_signed_interval_magnitude_both_directions"],
        "weight_multiplexor_copies": 2, "weight_ry_instructions_total": 2 * plan["M"],
        "weight_cx_instructions_total": 2 * plan["M"],
        "selector_and_reflection_instructions_total": 6,
        "source_operations_total": expected,
        "classical_preprocessing": upstream["classical_preprocessing"],
        "input_access": "explicit real Pauli list and normalized Pauli proposals; Gaussian loading, signed product-formula evolution and Pauli SELECT included in both access streams",
        "initialization": "unknown system input and every auxiliary zero are caller supplied; preparation/reset gates and time are excluded",
        "readout": "no measurement, discard/reset, repetition or amplification is emitted by this block oracle",
        "exclusions": ["external input preparation and zero-auxiliary initialization/reset",
                       "measurement and classical readout processing", "repetitions or coherent amplification",
                       "arbitrary-angle rotation synthesis", "physical code/factory/routing/controller/fault model",
                       "general coherent correction and its evolution", "uniform mesh/Trotter/mixing/output certificates"],
    }
    circuit.metadata = {
        **plan, "milestone": MILESTONE, "stage_slices": stages,
        "source_operations": len(circuit.data), "access_source_operations_each_copy": source_count,
        "expected_source_operations": expected, "primitive_counts": dict(circuit.count_ops()),
        "resource_ledger": resource_ledger,
        "pauli_label_order": "character i acts on qi; q0 is least significant",
        "proposal_normalization": "A_a=P_a/sqrt(J); sum A_a^dagger A_a=I",
        "decay_definition": "D_fin=eta*sum_(a,l) gamma_l*B_(a,l)^dagger*B_(a,l), with actual finite-Trotter access blocks",
        "selected_block": "D_fin/alpha with every auxiliary compressed to zero; alpha=eta*gamma_max",
        "selection_probability": "norm(D_fin*psi)^2/alpha^2 for normalized caller input psi; input dependent",
        "selector_zero_only_is_output_contract": False,
        "phase_convention": "P(pi) supplies the branch minus sign; both complete access/weight inverses are unconditional, preserving source phases",
        "numerical_checks_performed": False, "all_components_verified": False,
        "new_algorithm_result": False, "coherent_correction_implemented": False,
        "channel_implemented": False, "sampler_implemented": False,
        "finite_mesh_exact_kms": False, "mixing_certificate": False,
        "rotations_synthesized": False, "physical_resources_computed": False,
        "formal_proof_reproduced": False,
    }
    return circuit
