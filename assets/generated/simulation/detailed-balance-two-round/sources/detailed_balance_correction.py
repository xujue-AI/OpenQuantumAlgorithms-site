# SPDX-License-Identifier: MIT
"""Original primitive access and one heralded step for a fixed finite toy.

Only H=0.6 Z+0.8 X, proposal Z, beta=1, m=2, r=1 and tau=1/8 are
implemented. The centered access retains the accepted finite-Trotter decay.
All source phases and all auxiliary outcomes belong to the output contract.
This builder performs no simulation, sampling, synthesis or certification.
The correction specification is Chen et al., Nature 646 (2025), Eq. 19,
DOI 10.1038/s41586-025-09583-x; the primitive gate recipe is original.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat

from qiskit import QuantumCircuit

if __package__:
    from . import detailed_balance_access as access
    from . import detailed_balance_channel as channel
    from . import detailed_balance_decay as decay
else:
    import detailed_balance_access as access
    import detailed_balance_channel as channel
    import detailed_balance_decay as decay

TERMS = ((0.6, "Z"), (0.8, "X"))
PROPOSALS = ("Z",)
BETA = 1.0
CLOCK_BITS = 2
PRODUCT_STEPS = 1
TAU = 0.125
ACCEPTED_ALPHA = 0.9775450868978823
MAX_SOURCE_WIRES = 6
MAX_PRIMITIVE_OPERATIONS = 5_000
MAX_SOURCE_BYTES = 128 * 1024
PRIMITIVES = frozenset({"h", "x", "s", "sdg", "t", "tdg", "ry", "rz", "p", "cx"})
SOURCE_IDENTITIES = {
    "detailed_balance_access.py": {
        "bytes": 25_361,
        "sha256": "7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77",
    },
    "detailed_balance_decay.py": {
        "bytes": 12_324,
        "sha256": "82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1",
    },
    "detailed_balance_channel.py": {
        "bytes": 24_362,
        "sha256": "99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b",
    },
}
EXPECTED_PRIMITIVE_COUNTS = {
    "gaussian_access": {"h": 20, "ry": 3, "rz": 32, "cx": 39, "x": 1, "p": 3},
    "centered_decay": {"h": 40, "ry": 14, "rz": 64, "cx": 86, "x": 2, "p": 7},
    "correction_block": {"h": 46, "ry": 18, "rz": 65, "cx": 88, "x": 4, "p": 7},
}
EXPECTED_ZERO_PHASE_OPERATIONS = {
    "gaussian_access": 98,
    "centered_decay": 213,
    "correction_block": 228,
    "controlled_correction": 2_013,
    "heralded_step": 2_016,
    "known_plus_prefix": 2_017,
    "known_plus_trial": 2_029,
}
CONTROL_COSTS = {"h": 7, "cx": 15, "ry": 4, "rz": 4,
                 "x": 1, "p": 5, "s": 5, "sdg": 5, "t": 5, "tdg": 5}
UNVERIFIED_FLAGS = (
    "numerical_checks_performed", "coherent_correction_implemented",
    "coherent_correction_evolution_implemented", "general_coherent_correction_implemented",
    "general_channel_implemented", "channel_implemented", "sampler_implemented",
    "exact_finite_kms", "finite_mesh_exact_kms", "mixing_certificate",
    "rotations_synthesized", "physical_resources_computed", "formal_proof_reproduced",
    "all_components_verified", "new_algorithm_result", "random_sampling_performed",
)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _source_pin(module, name: str) -> dict:
    """Authenticate a bounded regular source before relying on its gate recipe."""
    expected = SOURCE_IDENTITIES[name]
    path = Path(module.__file__).absolute()
    if path.name != name or path.suffix != ".py" or path != path.resolve():
        raise ValueError("correction access requires unaliased reviewed Python sources")
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_size != expected["bytes"] or info.st_size > MAX_SOURCE_BYTES):
        raise ValueError("correction source pin has an unsafe size or file type")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0))
    try:
        if _identity(os.fstat(descriptor)) != _identity(info):
            raise ValueError("correction source changed before its bounded read")
        content = bytearray()
        while len(content) <= MAX_SOURCE_BYTES:
            chunk = os.read(descriptor, min(65_536, MAX_SOURCE_BYTES + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if (len(content) != info.st_size or len(content) > MAX_SOURCE_BYTES
                or _identity(os.fstat(descriptor)) != _identity(info)):
            raise ValueError("correction source changed during its bounded read")
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("correction source differs from its reviewed identity")
    return {"bytes": len(content), "sha256": digest}


def preflight_correction() -> dict:
    """Return the fixed layout, physical scale and caps before circuit allocation.

    The pinned sources and fixed input bind the static count predictions. The
    O(M) mesh and scalar angles are preprocessing, not a numerical block check.
    There is no argument accepting a general Hamiltonian or changing the step.
    """
    if ((TERMS, PROPOSALS, BETA, CLOCK_BITS, PRODUCT_STEPS, TAU)
            != (((0.6, "Z"), (0.8, "X")), ("Z",), 1.0, 2, 1, 0.125)):
        raise ValueError("only the frozen tilted correction toy is implemented")
    source_pins = {
        "detailed_balance_access.py": _source_pin(access, "detailed_balance_access.py"),
        "detailed_balance_decay.py": _source_pin(decay, "detailed_balance_decay.py"),
        "detailed_balance_channel.py": _source_pin(channel, "detailed_balance_channel.py"),
    }
    if (PRIMITIVES != access.PRIMITIVES or MAX_SOURCE_WIRES != 6
            or MAX_PRIMITIVE_OPERATIONS != 5_000):
        raise ValueError("the fixed correction source contract differs from its reviewed cap")
    if any(type(value) is not int or not 0 < value <= MAX_PRIMITIVE_OPERATIONS
           for value in EXPECTED_ZERO_PHASE_OPERATIONS.values()):
        raise ValueError("fixed correction preflight exceeds 5000 source operations")
    plan = decay.preflight_decay(TERMS, PROPOSALS, beta=BETA, clock_bits=CLOCK_BITS,
                                 product_steps=PRODUCT_STEPS)
    if (plan["n_system"], plan["clock_bits"], plan["proposal_bits"], plan["M"],
            plan["access_width"], plan["weight_wire"] ) != (1, 2, 0, 4, 3, 3):
        raise ValueError("the frozen tilted access layout has changed")
    alpha = float(plan["alpha"])
    if not math.isclose(alpha, ACCEPTED_ALPHA, rel_tol=0, abs_tol=1e-15):
        raise ValueError("the physical finite-decay scale differs from the accepted toy")
    correction_scale = alpha * math.tanh(BETA / 2) / 4
    lcu_weight = TAU * correction_scale
    if (not math.isfinite(correction_scale) or correction_scale <= 0
            or not math.isfinite(lcu_weight) or lcu_weight <= 0):
        raise ValueError("the fixed correction scale and LCU weight must be finite positive")
    # RY preparation amplitudes are sqrt(1/(1+w)) and sqrt(w/(1+w)).
    selector_angle = 2 * math.atan(math.sqrt(lcu_weight))
    return {
        "terms": [[coefficient, label] for coefficient, label in TERMS],
        "proposals": list(PROPOSALS), "beta": BETA, "clock_bits": CLOCK_BITS,
        "product_steps": PRODUCT_STEPS, "tau": TAU, "n_system": 1,
        "proposal_count": 1, "proposal_bits": 0, "M": 4,
        "system_wires": [0], "clock_wires": [1, 2], "proposal_wires": [],
        "weight_wire": 3, "correction_selector_wire": 4, "lcu_selector_wire": 5,
        "source_widths": {"centered_decay": 4, "correction_block": 5,
                          "controlled_correction": 6, "heralded_step": 6,
                          "known_plus_prefix": 6, "known_plus_trial": 6},
        "maximum_source_wires": MAX_SOURCE_WIRES,
        "maximum_source_operations": MAX_PRIMITIVE_OPERATIONS,
        "selected_auxiliary_state": "all_zero", "eta": plan["eta"],
        "gamma": plan["gamma"], "gamma_max": plan["gamma_max"], "alpha": alpha,
        "normalized_rates": plan["normalized_rates"], "weight_angles": plan["weight_angles"],
        "correction_scale": correction_scale, "correction_scale_definition": "alpha*tanh(beta/2)/4",
        "lcu_weight": lcu_weight, "lcu_selector_angle": selector_angle,
        "dt": plan["dt"], "dw": plan["dw"], "time_mesh": plan["time_mesh"],
        "frequency_mesh": plan["frequency_mesh"],
        "physical_squared_sum": plan["physical_squared_sum"],
        "source_pins": source_pins,
        "expected_zero_phase_operations": dict(EXPECTED_ZERO_PHASE_OPERATIONS),
        "source_count_status": "reviewed source predictions; new sources not numerically verified",
        "correction_norm_convention": "c_norm is spectral norm of Hermitian C, not Frobenius norm",
        **{flag: False for flag in UNVERIFIED_FLAGS},
    }


def _guard(circuit: QuantumCircuit, *, readout: bool = False) -> None:
    if not 1 <= circuit.num_qubits <= MAX_SOURCE_WIRES:
        raise ValueError("correction source exceeds six wires")
    if len(circuit.data) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("correction source exceeds 5000 primitive operations")
    allowed = PRIMITIVES | {"reset", "measure"} if readout else PRIMITIVES
    if set(circuit.count_ops()) - allowed:
        raise ValueError("correction source contains an unresolved or unsupported operation")
    if circuit.num_clbits != (6 if readout else 0):
        raise ValueError("correction source has the wrong classical readout width")
    try:
        angles = [float(circuit.global_phase)]
        angles.extend(float(angle) for instruction in circuit.data
                      for angle in instruction.operation.params)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("correction source phases and angles must be finite numbers") from error
    if any(not math.isfinite(angle) for angle in angles):
        raise ValueError("correction source phases and angles must be finite numbers")


def _check_counts(circuit: QuantumCircuit, name: str, expected: int) -> None:
    if len(circuit.data) != expected:
        raise AssertionError("fixed correction source violates its reviewed operation count")
    if name in EXPECTED_PRIMITIVE_COUNTS:
        if dict(circuit.count_ops()) != EXPECTED_PRIMITIVE_COUNTS[name]:
            raise AssertionError("fixed correction source violates its primitive inventory")


def _stage(circuit: QuantumCircuit, stages: dict, name: str, action) -> None:
    start = len(circuit.data)
    action()
    stages[name] = [start, len(circuit.data)]


def _controlled_h(circuit: QuantumCircuit, selector: int, *, zero: bool = False) -> None:
    """Emit RY(-theta), CZ, RY(theta), including both control-zero flips."""
    theta = math.atan2(0.8, 0.6)
    if zero:
        circuit.x(selector)
    circuit.ry(-theta, 0)
    circuit.h(0)
    circuit.cx(selector, 0)
    circuit.h(0)
    circuit.ry(theta, 0)
    if zero:
        circuit.x(selector)


def _metadata(circuit: QuantumCircuit, name: str, plan: dict, stages: dict,
              *, selected_block: str, resource_ledger: dict) -> None:
    circuit.metadata = {
        **plan, "milestone": "fixed_finite_correction_source_only", "source_kind": name,
        "total_wires": circuit.num_qubits, "classical_bits": circuit.num_clbits,
        "auxiliary_wires": list(range(1, circuit.num_qubits)),
        "stage_slices": stages, "source_operations": len(circuit.data),
        "primitive_counts": dict(circuit.count_ops()),
        "source_global_phase": float(circuit.global_phase),
        "selected_block": selected_block,
        "resource_ledger": resource_ledger,
        "finite_toy_source_implemented": True,
        "phase_convention": "literal P/RZ gates and full compose/inverse phases retained; no compensating phase fit",
        "pauli_label_order": "character i acts on qi; q0 is least significant",
        "proposal_normalization": "A=Z; J=1; sum A^dagger A=I",
        "decay_target": "actual accepted signed finite-Trotter Fourier access; no exact-H or continuum substitution",
        "correction_specification": "Chen et al. Eq19, C_ij=(i/2)*tanh(beta*(E_i-E_j)/4)*D_ij in H energy basis",
    }


def build_source_bundle() -> dict[str, QuantumCircuit]:
    """Emit six fixed primitive sources, reusing one accepted Gaussian U build.

    No Statevector, density matrix, matrix exponential or sampling call occurs.
    The coherent known-plus prefix is for one later measurement-probability
    check. The complete trial separately emits all resets and measurements.
    Success of the heralded step requires every one of q1..q5 to be zero.
    """
    plan = preflight_correction()
    unitary = access.build_gaussian_access(TERMS, PROPOSALS, beta=BETA,
                                          clock_bits=CLOCK_BITS, product_steps=PRODUCT_STEPS)
    _guard(unitary)
    _check_counts(unitary, "gaussian_access", EXPECTED_ZERO_PHASE_OPERATIONS["gaussian_access"])
    rotations = QuantumCircuit(4, name="Fixed finite frequency weights")
    access._multiplexed_ry(rotations, [1, 2], 3, plan["weight_angles"])
    _guard(rotations)
    if len(rotations.data) != 8:
        raise AssertionError("fixed weight multiplexor must emit four RY and four CX")
    centered = QuantumCircuit(4, name="Centered finite decay access")
    centered_stages = {}
    _stage(centered, centered_stages, "access_forward",
           lambda: centered.compose(unitary, [0, 1, 2], inplace=True))
    _stage(centered, centered_stages, "weight_forward",
           lambda: centered.compose(rotations, inplace=True))
    _stage(centered, centered_stages, "weight_z_phase", lambda: centered.p(math.pi, 3))
    _stage(centered, centered_stages, "weight_inverse",
           lambda: centered.compose(rotations.inverse(), inplace=True))
    _stage(centered, centered_stages, "access_inverse",
           lambda: centered.compose(unitary.inverse(), [0, 1, 2], inplace=True))
    _guard(centered)
    _check_counts(centered, "centered_decay", EXPECTED_ZERO_PHASE_OPERATIONS["centered_decay"])
    upstream = unitary.metadata
    centered_ledger = {
        "access_copies": 2, "access_instructions_each_copy": len(unitary.data),
        "access_instructions_total": 2 * len(unitary.data),
        "gaussian_prepare_instructions_total": 2 * upstream["gaussian_prepare_instructions"],
        "signed_evolution_instructions_total": 4 * upstream["signed_evolution_instructions_each_direction"],
        "pauli_select_instructions_total": 2 * upstream["select_instructions"],
        "dft_instructions_total": 2 * upstream["dft_instructions"],
        "requested_signed_interval_magnitude_all_access_directions":
            2 * upstream["requested_signed_interval_magnitude_both_directions"],
        "weight_ry_instructions_total": 8, "weight_cx_instructions_total": 8,
        "weight_z_phase_instructions": 1, "source_operations_total": len(centered.data),
        "classical_preprocessing": upstream["classical_preprocessing"],
        "initialization": "caller supplies normalized system input and q1/q2/q3 zero; no reset emitted",
        "readout": "no measurement, repetition or amplification emitted",
        "exclusions": ["external input/zero-auxiliary preparation and reset", "measurement/readout",
                       "repetition/amplification", "arbitrary-angle synthesis", "physical fault/code/factory/routing model"],
    }
    _metadata(centered, "centered_decay", plan, centered_stages,
              selected_block="I-2*D_fin/alpha when q1/q2/q3 all zero",
              resource_ledger=centered_ledger)

    correction = QuantumCircuit(5, name="Fixed finite correction access")
    correction_stages = {}
    _stage(correction, correction_stages, "selector_open", lambda: correction.h(4))
    _stage(correction, correction_stages, "controlled_one_h",
           lambda: _controlled_h(correction, 4))
    _stage(correction, correction_stages, "centered_decay",
           lambda: correction.compose(centered, [0, 1, 2, 3], inplace=True))
    _stage(correction, correction_stages, "controlled_zero_h",
           lambda: _controlled_h(correction, 4, zero=True))
    _stage(correction, correction_stages, "selector_positive_phase",
           lambda: correction.rz(math.pi, 4))
    _stage(correction, correction_stages, "selector_close", lambda: correction.h(4))
    _guard(correction)
    _check_counts(correction, "correction_block", EXPECTED_ZERO_PHASE_OPERATIONS["correction_block"])
    correction_ledger = {
        "centered_decay_instructions": len(centered.data),
        "controlled_one_h_instructions": 5, "controlled_zero_h_instructions": 7,
        "controlled_zero_x_flips": 2, "selector_h_instructions": 2,
        "selector_rz_instructions": 1, "source_operations_total": len(correction.data),
        "centered_decay_resource_ledger": centered_ledger,
        "initialization": "caller supplies normalized input and q1..q4 zero; no reset emitted",
        "readout": "select all four auxiliaries zero; no measurement/repetition emitted",
        "exclusions": list(centered_ledger["exclusions"]),
    }
    _metadata(correction, "correction_block", plan, correction_stages,
              selected_block="C/a when q1..q4 all zero; a=alpha*tanh(beta/2)/4",
              resource_ledger=correction_ledger)

    controlled = QuantumCircuit(6, name="Primitive controlled finite correction")
    controlled_stages = {}
    global_phase = float(correction.global_phase)
    expected_lift = sum(CONTROL_COSTS[name] * count
                        for name, count in correction.count_ops().items())
    phase_instructions = int(global_phase != 0)
    if expected_lift + phase_instructions + 16 > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("controlled correction preflight exceeds 5000 source operations")
    # A source global phase becomes relative on l=1, even though it was global before control.
    _stage(controlled, controlled_stages, "source_global_phase",
           lambda: controlled.p(global_phase, 5) if phase_instructions else None)
    start = len(controlled.data)
    for instruction in correction.data:
        wires = [correction.find_bit(qubit).index for qubit in instruction.qubits]
        channel._controlled_instruction(controlled, instruction.operation.name,
                                        instruction.operation.params, wires, 5)
        _guard(controlled)
    controlled_stages["complete_primitive_lift"] = [start, len(controlled.data)]
    _guard(controlled)
    _check_counts(controlled, "controlled_correction", expected_lift + phase_instructions)
    if expected_lift != EXPECTED_ZERO_PHASE_OPERATIONS["controlled_correction"]:
        raise AssertionError("fixed controlled correction violates its reviewed primitive cost")
    controlled_ledger = {
        "correction_instructions": len(correction.data), "controlled_primitive_costs": dict(CONTROL_COSTS),
        "controlled_primitive_instructions": expected_lift,
        "controlled_source_global_phase": global_phase,
        "relative_global_phase_instructions": phase_instructions,
        "source_operations_total": len(controlled.data),
        "correction_resource_ledger": correction_ledger,
        "initialization": "caller supplies source wires and l=0/1; no reset emitted",
        "readout": "no measurement or repetition emitted",
        "exclusions": list(centered_ledger["exclusions"]),
    }
    _metadata(controlled, "controlled_correction", plan, controlled_stages,
              selected_block="complete K_C on l=1, identity on l=0; relative source phase retained",
              resource_ledger=controlled_ledger)

    heralded = QuantumCircuit(6, name="One heralded finite correction step")
    heralded_stages = {}
    _stage(heralded, heralded_stages, "lcu_prepare",
           lambda: heralded.ry(plan["lcu_selector_angle"], 5))
    _stage(heralded, heralded_stages, "complete_controlled_correction",
           lambda: heralded.compose(controlled, inplace=True))
    _stage(heralded, heralded_stages, "lcu_negative_i_phase", lambda: heralded.p(-math.pi / 2, 5))
    _stage(heralded, heralded_stages, "lcu_unprepare",
           lambda: heralded.ry(-plan["lcu_selector_angle"], 5))
    _guard(heralded)
    _check_counts(heralded, "heralded_step", len(controlled.data) + 3)
    heralded_ledger = {
        "controlled_correction_instructions": len(controlled.data), "lcu_prepare_ry": 1,
        "lcu_unprepare_ry": 1, "lcu_relative_phase_p": 1,
        "source_operations_total": len(heralded.data),
        "controlled_correction_resource_ledger": controlled_ledger,
        "initialization": "caller supplies input and all five auxiliaries zero; no reset emitted",
        "readout": "success is all q1..q5 zero; all 31 failure outcomes retained by instrument contract",
        "repetitions": "none emitted; known-input retry accounting belongs to the bounded readout ledger",
        "exclusions": list(centered_ledger["exclusions"]),
    }
    _metadata(heralded, "heralded_step", plan, heralded_stages,
              selected_block="T_tau=(I-i*tau*C)/(1+tau*a) when q1..q5 all zero",
              resource_ledger=heralded_ledger)
    heralded.metadata["auxiliary_outcome_count"] = 32
    heralded.metadata["success_auxiliary_outcome"] = "00000"
    heralded.metadata["outcome_index_convention"] = "z=sum bit(qk)*2**(k-1), k=1..5; state index=(z<<1)|system_bit"

    prefix = QuantumCircuit(6, name="Known plus coherent correction prefix")
    prefix_stages = {}
    _stage(prefix, prefix_stages, "known_plus_prepare", lambda: prefix.h(0))
    _stage(prefix, prefix_stages, "heralded_step", lambda: prefix.compose(heralded, inplace=True))
    _guard(prefix)
    _check_counts(prefix, "known_plus_prefix", len(heralded.data) + 1)
    _metadata(prefix, "known_plus_prefix", plan, prefix_stages,
              selected_block="H-prepared known plus coherent prefix; measurement probabilities require all outcomes",
              resource_ledger={"known_plus_prepare_h": 1, "heralded_step_instructions": len(heralded.data),
                               "source_operations_total": len(prefix.data),
                               "initialization": "all six wires zero are supplied for one bounded vector check",
                               "readout": "measurement/reset stream is materialized separately in known_plus_trial",
                               "exclusions": ["zero-state initialization/reset in this prefix", "measurement in this prefix",
                                              "sampling/retries", "synthesis", "physical resources"]})

    trial = QuantumCircuit(6, 6, name="Known plus finite correction readout trial")
    trial_stages = {}
    _stage(trial, trial_stages, "initialization_resets", lambda: trial.reset(range(6)))
    _stage(trial, trial_stages, "known_plus_prepare", lambda: trial.h(0))
    _stage(trial, trial_stages, "heralded_step", lambda: trial.compose(heralded, inplace=True))
    _stage(trial, trial_stages, "all_wire_measurements", lambda: trial.measure(range(6), range(6)))
    _guard(trial, readout=True)
    _check_counts(trial, "known_plus_trial", len(heralded.data) + 13)
    _metadata(trial, "known_plus_trial", plan, trial_stages,
              selected_block="known plus trial; accept only auxiliary measurements q1..q5=00000",
              resource_ledger={
                  "all_wire_resets": 6, "known_plus_prepare_h": 1,
                  "heralded_step_instructions": len(heralded.data), "all_wire_measurements": 6,
                  "source_operations_per_complete_attempt": len(trial.data),
                  "heralded_step_resource_ledger": heralded_ledger,
                  "classical_processing": "read five auxiliary bits; on success map system bit0/1 to Z +1/-1; count every failed attempt",
                  "retry_contract": "reset/reprepare known plus after failure; no recovery of unknown intermediate input",
                  "sampling": "no shots drawn; bounded N,R ideal-bias/retry formulas are separate from source emission",
                  "exclusions": ["measured classical-selection duration", "arbitrary-angle synthesis/error allocation",
                                 "scheduling/transport/idle/reset/readout faults", "physical code/factory/runtime model",
                                 "general-H correction/evolution", "sampler/continuum/KMS/mixing/formal certificate"],
              })
    trial.metadata["auxiliary_outcome_count"] = 32
    trial.metadata["success_auxiliary_outcome"] = "00000"
    trial.metadata["system_z_readout_classical_bit"] = 0
    trial.metadata["auxiliary_readout_classical_bits"] = [1, 2, 3, 4, 5]
    return {"centered_decay": centered, "correction_block": correction,
            "controlled_correction": controlled, "heralded_step": heralded,
            "known_plus_prefix": prefix, "known_plus_trial": trial}
