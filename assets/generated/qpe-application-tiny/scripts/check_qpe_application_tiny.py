# SPDX-License-Identifier: MIT
"""Future fixed tiny check, callable only through its separately approved supervisor.

This source has not been imported or executed during authoring.  It performs
no full application output simulation, shots, inference, synthesis, physical
estimation, notebook, website build or remote job.  Its 14 payloads are written
only after every tiny check passes; a failure is retained by the supervisor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys

REPO = Path(__file__).resolve().parents[1]
CONTRACT_SHA256 = "2d5d9a7f60503c65ab09a433679da9449fb805d59110808e778be7cba59164b3"
PAYLOAD_NAMES = (
    "kernel.qasm", "kernel-without-reference-z.qasm", "kernel-without-odd-evolution.qasm",
    "retained-deferred.qasm", "retained-reused.qasm", "retained-reprepared.qasm", "feedback.qasm",
    "fringe-k1-cos.qasm", "fringe-k1-sin.qasm", "fringe-k2-cos.qasm", "fringe-k2-sin.qasm",
    "fringe-k4-cos.qasm", "fringe-k4-sin.qasm", "checks.json",
)
MAX_QASM_BYTES = 64 * 1024
MAX_JSON_BYTES = 128 * 1024
MAX_TOTAL_PAYLOAD_BYTES = 1024 * 1024
MAX_MAP_APPLICATIONS_PER_CASE = 10_000
MAX_DENSITY_DIMENSION = 64
MAX_COMPLEX_ENTRIES_PER_DENSITY = 4096
MAX_RETAINED_DENSITY_ENTRIES = 16 * MAX_COMPLEX_ENTRIES_PER_DENSITY
TOLERANCE = 2e-12
LIMITS = {
    "quantum_wires_max": 6,
    "retained_records_max": 16,
    "controlled_power_max": 4,
    "expanded_validation_instructions_per_case_max": 10_000,
}


def _load_source(name: str, filename: str):
    """Load exactly one pinned module, avoiding openquantum package side effects."""
    path = REPO / "src" / "openquantum" / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("Pinned module must be a regular source file")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Pinned module source cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Budget:
    """Actual primitive whole-state maps, including all retained records.

    An operator is explicitly assembled by ket-column traversals, so each gate
    on each column is charged.  DensityMatrix.evolve applies one whole-density
    primitive map; it is charged once per record, without looping over ket
    columns.  Each reset's two Kraus maps and each measurement's two outcome
    projections are charged separately.  This is not a FLOP/ket-equivalent cost.
    """
    def __init__(self, name: str):
        self.name = name
        self.applications = 0
        self.by_kind: dict[str, int] = {}
        self.maximum_records = 1
        self.maximum_state_dimension = 1

    def charge(self, kind: str, amount: int = 1) -> None:
        self.applications += amount
        self.by_kind[kind] = self.by_kind.get(kind, 0) + amount
        if self.applications > MAX_MAP_APPLICATIONS_PER_CASE:
            raise ValueError(f"{self.name}: expanded validation instruction cap exceeded")

    def report(self) -> dict:
        return {
            "case": self.name,
            "expanded_validation_instructions": self.applications,
            "applications_by_kind": self.by_kind,
            "maximum_retained_records": self.maximum_records,
            "maximum_state_dimension": self.maximum_state_dimension,
            "maximum_complex_entries_per_density": self.maximum_state_dimension ** 2,
            "maximum_retained_density_entries": self.maximum_records * self.maximum_state_dimension ** 2,
            "count_semantics": "one primitive whole-state map per actual record; explicit ket columns charged separately",
        }


def _full_operator(circuit, core, Statevector, np, budget: _Budget):
    ledger = core.source_ledger(circuit)
    if circuit.num_qubits != 3 or circuit.num_clbits or ledger["resets"] or ledger["readouts"] \
            or ledger["feedback_decisions"]:
        raise ValueError("Full operator checks admit only the three-wire primitive kernel")
    dimension = 1 << circuit.num_qubits
    budget.maximum_state_dimension = dimension
    matrix = np.empty((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        state = Statevector.from_int(column, dimension)
        for item in circuit.data:
            qargs = [circuit.find_bit(qubit).index for qubit in item.qubits]
            budget.charge("ket_primitive_map")
            state = state.evolve(item.operation, qargs=qargs)
        matrix[:, column] = state.data
    return matrix


def _instrument_records(circuit, core, DensityMatrix, np, budget: _Budget):
    """Unnormalized densities retain every classical record, including exact zeros.

    Reset uses the trace-preserving sum of two Kraus maps on each density.
    It creates no hidden record branches.  No projection is normalized, clipped
    or pruned.  A target reset therefore genuinely erases retained correlations.
    """
    core.source_ledger(circuit)
    dimension = 1 << circuit.num_qubits
    if dimension > MAX_DENSITY_DIMENSION or dimension * dimension > MAX_COMPLEX_ENTRIES_PER_DENSITY:
        raise ValueError("Deterministic density array shape exceeds the fixed 64-by-64 guard")
    budget.maximum_state_dimension = dimension
    initial = np.zeros((dimension, dimension), dtype=np.complex128)
    initial[0, 0] = 1
    branches = {0: initial}
    written: set[int] = set()
    indices = np.arange(dimension)

    def projection(rho, qubit: int, outcome: int):
        budget.charge("outcome_projection")
        mask = ((indices >> qubit) & 1) == outcome
        result = rho.copy()
        result[~mask, :] = 0
        result[:, ~mask] = 0
        return result

    def reset_map(rho, qubit: int):
        budget.charge("reset_kraus_map", 2)
        low = indices[((indices >> qubit) & 1) == 0]
        high = low | (1 << qubit)
        result = np.zeros_like(rho)
        result[np.ix_(low, low)] = rho[np.ix_(low, low)] + rho[np.ix_(high, high)]
        return result

    def run(block, qmap: list[int], cmap: list[int], incoming, depth: int = 0):
        if depth > 1:
            raise ValueError("Only the single frozen correction body is supported")
        current = incoming
        for item in block.data:
            operation = item.operation
            qargs = [qmap[block.find_bit(qubit).index] for qubit in item.qubits]
            cargs = [cmap[block.find_bit(bit).index] for bit in item.clbits]
            if operation.name == "measure":
                if len(qargs) != 1 or len(cargs) != 1 or cargs[0] in written:
                    raise ValueError("Each single-bit readout must write a fresh record bit")
                next_branches = {}
                for record, rho in current.items():
                    for outcome in (0, 1):
                        next_record = record | (outcome << cargs[0])
                        if next_record in next_branches:
                            raise AssertionError("Distinct unoverwritten records unexpectedly collided")
                        next_branches[next_record] = projection(rho, qargs[0], outcome)
                written.add(cargs[0])
                current = next_branches
            elif operation.name == "reset":
                if len(qargs) != 1 or cargs:
                    raise ValueError("Only explicit single-wire resets are admitted")
                current = {record: reset_map(rho, qargs[0]) for record, rho in current.items()}
            elif operation.name == "if_else":
                bit, value = operation.condition
                bit_index = cmap[block.find_bit(bit).index]
                if bit_index not in written or value != 1 or len(operation.blocks) != 1:
                    raise ValueError("Feedback must use a previously measured bit and have no false body")
                next_branches = {}
                for record, rho in current.items():
                    budget.charge("feedback_predicate")
                    if ((record >> bit_index) & 1) == value:
                        next_branches.update(run(operation.blocks[0], qargs, cargs,
                                                 {record: rho}, depth + 1))
                    else:
                        next_branches[record] = rho
                current = next_branches
            else:
                if cargs or operation.name not in core.PRIMITIVES - {"measure", "reset"}:
                    raise ValueError("Instrument contains unsupported or implicit classical work")
                next_branches = {}
                for record, rho in current.items():
                    budget.charge("density_primitive_map")
                    next_branches[record] = DensityMatrix(rho).evolve(operation, qargs=qargs).data
                current = next_branches
            budget.maximum_records = max(budget.maximum_records, len(current))
            if len(current) > 16:
                raise ValueError("More than 16 retained records would leave the tiny tranche")
            if len(current) * dimension * dimension > MAX_RETAINED_DENSITY_ENTRIES:
                raise ValueError("Deterministic retained density array count exceeds its guard")
        return current

    branches = run(circuit, list(range(circuit.num_qubits)), list(range(circuit.num_clbits)), branches)
    if written != set(range(circuit.num_clbits)) or set(branches) != set(range(1 << circuit.num_clbits)):
        raise AssertionError("Every prescribed output record must be present, including zero records")
    probabilities = np.empty(1 << circuit.num_clbits, dtype=float)
    for record, rho in branches.items():
        trace = np.trace(rho)
        if not np.isfinite(rho).all() or abs(trace.imag) > TOLERANCE:
            raise AssertionError("Instrument density is nonfinite or has an invalid trace")
        probabilities[record] = float(trace.real)
    if float(np.min(probabilities)) < -TOLERANCE or abs(float(np.sum(probabilities)) - 1) > TOLERANCE:
        raise AssertionError("Unnormalized record laws lost probability mass")
    return probabilities


def _error(actual, expected, np) -> float:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Nonfinite or mismatched numerical check arrays")
    return float(np.max(np.abs(actual - expected)))


def _require_close(actual, expected, np, label: str) -> float:
    residual = _error(actual, expected, np)
    if residual > TOLERANCE:
        raise AssertionError(f"{label}: maximum entry residual exceeds {TOLERANCE}")
    return residual


def _output_inodes(directory: Path) -> dict[str, tuple[int, int]]:
    if directory.is_symlink() or not directory.is_dir() or set(os.listdir(directory)) != set(PAYLOAD_NAMES):
        raise ValueError("Supervisor must supply exactly the 14 precreated payload files")
    identities = {}
    for name in PAYLOAD_NAMES:
        info = (directory / name).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != 0 or info.st_nlink != 1:
            raise ValueError("Initial payloads must be empty, unlinked regular inodes")
        identities[name] = (info.st_dev, info.st_ino)
    return identities


def _write_payloads(directory: Path, payloads: dict[str, bytes], identities: dict) -> None:
    if set(payloads) != set(PAYLOAD_NAMES):
        raise AssertionError("Fixed payload closure changed")
    if sum(map(len, payloads.values())) > MAX_TOTAL_PAYLOAD_BYTES:
        raise ValueError("Payload total exceeds 1 MiB")
    for name, data in payloads.items():
        cap = MAX_JSON_BYTES if name == "checks.json" else MAX_QASM_BYTES
        if len(data) > cap:
            raise ValueError(f"Payload {name} exceeds its fixed byte ceiling")
    # Validate all bytes and paths before any artifact is written.
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in PAYLOAD_NAMES:
            descriptor = os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identities[name] \
                        or info.st_nlink != 1 or info.st_size != 0:
                    raise ValueError("Supervisor payload inode changed before write")
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(payloads[name])
                    handle.flush()
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    output = Path(arguments.output_dir)
    if not output.is_absolute() or str(output.resolve()) != str(output):
        raise ValueError("Supervisor output path must be absolute and canonical")
    identities = _output_inodes(output)
    if sys.version_info[:3] != (3, 12, 14) or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError("Tiny checker requires the exact approved Python 3.12.14 -I -B invocation")

    # These are the only scientific imports, made inside the supervised child.
    import numpy as np
    from qiskit import qasm3
    from qiskit.quantum_info import DensityMatrix, Statevector
    core = _load_source("_application_qpe_tiny_core", "application_qpe_tiny.py")
    reference = _load_source("_application_qpe_tiny_reference", "application_qpe_tiny_reference.py")
    if core.CONTRACT_SHA256 != CONTRACT_SHA256:
        raise AssertionError("The accepted private contract pin changed")

    circuits = {}
    cases = []
    kernel = core.build_controlled_kernel()
    circuits["kernel.qasm"] = kernel
    for power in (1, 2, 4):
        repeated = kernel.copy_empty_like()
        for _ in range(power):
            repeated.compose(kernel, inplace=True)
        budget = _Budget(f"full-controlled-k{power}")
        actual = _full_operator(repeated, core, Statevector, np, budget)
        expected = reference.controlled_unitary_matrix(power)
        zero_indices = np.array([0, 2, 4, 6])
        odd_indices = np.array([3, 5])
        cases.append({
            **budget.report(),
            "maximum_full_controlled_entry_error": _require_close(actual, expected, np, "full controlled kernel"),
            "maximum_control_zero_identity_error": _require_close(actual[:, zero_indices],
                                                                  np.eye(8)[:, zero_indices], np, "control-zero identity"),
            "maximum_odd_sector_entry_error": _require_close(actual[np.ix_(odd_indices, odd_indices)],
                                                             expected[np.ix_(odd_indices, odd_indices)], np, "odd sector"),
            "source": core.source_ledger(repeated),
            "global_phase_fit_performed": False,
        })

    expected_base = reference.controlled_unitary_matrix(1)
    missing_z = core.build_kernel_without_reference_z()
    circuits["kernel-without-reference-z.qasm"] = missing_z
    budget = _Budget("negative-missing-reference-z")
    bad_z = _full_operator(missing_z, core, Statevector, np, budget)
    omitted_reference_error = _require_close(bad_z, reference.controlled_evolution_without_reference(1),
                                            np, "missing-Z independent reference")
    difference = _error(bad_z, expected_base, np)
    if difference <= 0.1:
        raise AssertionError("Missing reference Z must change the full controlled operator")
    cases.append({**budget.report(), "negative_control_detected": True,
                  "maximum_difference_from_true_U": difference,
                  "maximum_error_against_deliberately_omitted_reference": omitted_reference_error,
                  "source": core.source_ledger(missing_z)})

    missing_odd = core.build_kernel_without_odd_evolution()
    circuits["kernel-without-odd-evolution.qasm"] = missing_odd
    budget = _Budget("negative-missing-odd-evolution")
    bad_odd = _full_operator(missing_odd, core, Statevector, np, budget)
    odd_indices = np.array([3, 5])
    even_indices = np.array([1, 7])
    odd_difference = _error(bad_odd[np.ix_(odd_indices, odd_indices)],
                            expected_base[np.ix_(odd_indices, odd_indices)], np)
    if odd_difference <= 0.1:
        raise AssertionError("Omitting the odd sector must fail a full controlled comparison")
    cases.append({**budget.report(), "negative_control_detected": True,
                  "maximum_odd_sector_difference": odd_difference,
                  "maximum_even_sector_error": _require_close(bad_odd[np.ix_(even_indices, even_indices)],
                                                              expected_base[np.ix_(even_indices, even_indices)], np,
                                                              "odd negative preserves even sector"),
                  "source": core.source_ledger(missing_odd)})

    retained_reference = reference.retained_joint_probabilities()
    fresh_reference = reference.fresh_joint_probabilities()
    _require_close(retained_reference - fresh_reference,
                   reference.retained_minus_fresh_reference(), np, "factorized retained/fresh difference")
    record_results = {}
    for name, builder, expected in (
        ("retained-deferred.qasm", core.build_reduced_deferred, retained_reference),
        ("retained-reused.qasm", core.build_reduced_textbook, retained_reference),
        ("feedback.qasm", core.build_reduced_feedback, retained_reference),
        ("retained-reprepared.qasm", core.build_reprepared_textbook, fresh_reference),
    ):
        circuit = builder()
        circuits[name] = circuit
        budget = _Budget(name.removesuffix(".qasm"))
        actual = _instrument_records(circuit, core, DensityMatrix, np, budget)
        record_results[name] = actual
        cases.append({**budget.report(),
                      "maximum_joint_record_error": _require_close(actual, expected, np, name),
                      "probabilities": actual.tolist(), "independent_probabilities": expected.tolist(),
                      "record_index": "y0+4*y1; y0=c0+2*c1, y1=c2+2*c3",
                      "zero_probability_records_retained": True,
                      "source": core.source_ledger(circuit)})
    _require_close(record_results["retained-deferred.qasm"], record_results["retained-reused.qasm"],
                   np, "deferred/reused record equivalence")
    _require_close(record_results["retained-reused.qasm"], record_results["feedback.qasm"],
                   np, "textbook/feedback record equivalence")
    observed_retention_difference = _error(record_results["retained-reused.qasm"],
                                          record_results["retained-reprepared.qasm"], np)
    if observed_retention_difference <= 100 * TOLERANCE:
        raise AssertionError("Actual target re-preparation must expose the different joint law")

    for power in (1, 2, 4):
        for basis in ("cos", "sin"):
            name = f"fringe-k{power}-{basis}.qasm"
            circuit = core.build_fresh_experiment(power, basis)
            circuits[name] = circuit
            budget = _Budget(name.removesuffix(".qasm"))
            actual = _instrument_records(circuit, core, DensityMatrix, np, budget)
            expected = reference.fringe_reference(power, basis)
            residual = abs(float(actual[0]) - expected["input_zero_probability"])
            mixture_residual = abs(expected["input_zero_probability"] - expected["spectral_mixture_zero_probability"])
            bias = abs(float(actual[0]) - expected["ground_zero_probability"])
            if residual > TOLERANCE or mixture_residual > TOLERANCE \
                    or bias > expected["excited_input_weight"] + TOLERANCE:
                raise AssertionError("Fresh fringe or frozen contamination diagnostic failed")
            cases.append({**budget.report(), "maximum_zero_probability_error": residual,
                          "spectral_mixture_residual": mixture_residual,
                          "numerical_ground_fringe_bias": bias,
                          "probabilities": actual.tolist(), "independent_reference": expected,
                          "source": core.source_ledger(circuit)})

    result = {
        "status": "passed", "scope": "tiny-ideal-circuit-instrument",
        "contract_sha256": CONTRACT_SHA256, "limits": LIMITS,
        "case_count": len(cases), "cases": cases,
        "numerical_entry_tolerance": TOLERANCE,
        "array_allocation_bounds": {
            "density_dimension_max": MAX_DENSITY_DIMENSION,
            "complex_entries_per_density_max": MAX_COMPLEX_ENTRIES_PER_DENSITY,
            "retained_density_entries_max": MAX_RETAINED_DENSITY_ENTRIES,
            "explicit_controlled_operator_shape": [8, 8],
            "scope": "deterministic scientific state/reference shapes; SDK/import temporary memory uses supervisor postflight",
        },
        "observed_retention_joint_difference": observed_retention_difference,
        "spectral_diagnostics": reference.spectral_components(),
        "frozen_task": {"Hamiltonian": "-Z0-Z1+X0*X1/4", "input": "reset |00>",
                        "tau0": "pi/3", "Eref": "3", "W": "6", "U": "-exp(-i H pi/3)"},
        "full_application_output_simulated": False,
        "full_application_confidence_validated": False,
        "interval_synthesis_certificate": False,
        "physical_estimation_performed": False,
        "classical_inference_performed": False,
        "shot_sampling_performed": False,
        "formal_proof_reproduced": False,
        "evidence_note": "bounded numerical circuit/instrument checks; no compiled/hardware confidence claim",
    }
    payloads = {name: qasm3.dumps(circuit).encode("utf-8") for name, circuit in circuits.items()}
    payloads["checks.json"] = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _write_payloads(output, payloads, identities)
    print("tiny ideal circuit/instrument checks passed; 14 fixed payloads written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
