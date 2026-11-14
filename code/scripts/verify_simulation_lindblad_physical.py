#!/usr/bin/env python3
"""Replay the immutable n2 Lindblad release without native synthesis or queries.

The exact reviewed source/results/QASMs remain inputs. Handwritten density,
GKSL, gate-sequence and serialized-ledger calculations replay their numerical
claims. Native factory calibration and numerical error guards remain premises;
this gate is neither a certificate nor execution of the repeated population.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_RELATIVE = "assets/generated/simulation/lindblad-physical"
REVIEWED_MANIFEST_SHA256 = "1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595"
SOURCE_FILES = {
    "generate_simulation_lindblad_physical.py": "scripts/generate_simulation_lindblad_physical.py",
    "instrument_resources.py": "src/openquantum/instrument_resources.py",
    **{name + ".py": "src/openquantum/simulation/" + name + ".py" for name in (
        "lindblad_physical_workload", "lindblad_workload", "two_spin_lindblad", "open_systems")},
    "LICENSE.txt": "LICENSE", "requirements-lock.txt": "requirements-lock.txt", "references.bib": None,
}
BUNDLE_FILES = frozenset(SOURCE_FILES) | frozenset({
    "README.md", "results.json", "manifest.json", "summary.csv", "family-status.csv",
    "source-trial.qasm", "clifford-t-trial.qasm", "one-step.svg", "counts.svg",
    "error-budget.svg", "failures.svg", "space-time.svg",
})
FROZEN_FILES = frozenset({"src/openquantum/instrument_resources.py", "src/openquantum/grover.py",
    "scripts/estimate_resources.py", "assets/generated/resources/grover-surface-code.json",
    "assets/generated/resources/grover-surface-code.csv"})
PACKAGES = {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2"}
SOURCE_COUNTS = {"cx": 483, "rx": 276, "reset": 209, "cry": 207, "rz": 138, "measure": 4, "h": 2, "s": 1}
CT_COUNTS = {"h": 80732, "t": 78108, "s": 44437, "x": 414, "cx": 903, "reset": 209, "measure": 4, "barrier": 426}
LIMITS = {"max_qubits": 4, "source_operations": 20000, "synthesized_operations": 500000,
          "validation_operations": 500000, "validation_branches": 256}
PROFILE = {"physical_gate_time_ns": 100, "physical_measurement_time_ns": 500,
    "physical_initialization_time_ns": 100, "code_cycle_time_ns": 1000, "reaction_time_ns": 10000,
    "controller_failure_per_decision": 1e-15, "memory_prefactor": .1,
    "initialization_residual_prefactor": .1, "memory_threshold": .01,
    "factory_pipeline_failure_bound": .01, "factory_feedback_rounds_reserved": 3, "maximum_code_distance": 49}
CAPS = {"replay_seconds": 60, "files": 21, "bundle_bytes": 32 * 1024**2, "file_bytes": 32 * 1024**2,
        "qasm_instructions": 500000, "physical_points": 182}


def equal(actual, expected, label):
    if isinstance(expected, (set, frozenset)):
        valid = isinstance(actual, (set, frozenset)) and actual == expected
    else:
        valid = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    if not valid:
        raise ValueError(f"n2 {label} differs from the frozen release")


def close(actual, expected, label, *, relative=2e-12, absolute=1e-30):
    if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual):
        raise ValueError(f"n2 nonfinite or nonnumeric {label}")
    if not math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute):
        raise ValueError(f"n2 {label} replay differs: {actual!r} versus {expected!r}")


def deadline_check(deadline):
    if time.monotonic() > deadline:
        raise ValueError("n2 replay exceeded its 60-second cap")


def checked_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError(f"Unsafe n2 relative path: {name}")
    path = root / relative
    if any(ancestor.is_symlink() for ancestor in (root, *root.parents, *path.parents)):
        raise ValueError(f"Linked n2 ancestor: {name}")
    if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > CAPS["file_bytes"]:
        raise ValueError(f"Missing, linked, special or oversized n2 file: {name}")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate n2 JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite n2 JSON value: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            reject(value)
        return parsed
    result = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError("n2 JSON must contain an object")
    return result


def verify_records(manifest, results):
    """Validate complete historical status before any scientific replay."""
    try:
        equal(manifest["schema_version"], 1, "manifest schema")
        equal(results["schema_version"], 1, "results schema")
        equal(manifest["python"], "3.12.14", "recorded Python")
        equal(manifest["packages"], PACKAGES, "recorded package versions")
        equal(manifest["private_cache_distributed"], False, "private cache scope")
        equal(manifest["whole_family_physical_completed"], False, "whole-family scope")
        for key, names in (("source_sha256", set(SOURCE_FILES)), ("artifacts_sha256", BUNDLE_FILES - {"manifest.json"}),
                           ("frozen_before", FROZEN_FILES), ("frozen_after", FROZEN_FILES)):
            if not isinstance(manifest[key], dict):
                raise ValueError("n2 manifests must be objects")
            equal(set(manifest[key]), names, key + " inventory")
        equal(manifest["frozen_before"], manifest["frozen_after"], "baseline preservation")
        prepared = results["prepared"]
        equal(prepared["case_id"], "lindblad-n2-r69-Z0", "case identity")
        equal(prepared["task"], {"n_system": 2, "time": 1., "J": .6, "hx": .35, "rate0": .8, "other_rate": .5}, "fixed input")
        equal(prepared["selected_steps"], 69, "step policy")
        equal(prepared["limits"], LIMITS, "instruction limits")
        equal(prepared["source_counts"], SOURCE_COUNTS, "source counts")
        synthesis = prepared["actual_synthesis"]
        for key, value in (("limits", LIMITS), ("source_counts", SOURCE_COUNTS), ("clifford_t_counts", CT_COUNTS),
                           ("qiskit_version", "2.6.2"), ("preserve_full_unitary", True),
                           ("transpiler_qubits_initially_zero", False), ("rotation_occurrences", 828),
                           ("unique_rotation_targets", 6), ("actual_synthesis_sequences_generated", True),
                           ("formal_interval_certificate", False)):
            equal(synthesis[key], value, "compiler " + key)
        equal(results["family_allocations"], prepared["family_allocations"], "family allocation identity")
        allocations = results["family_allocations"]
        equal(allocations["declared_family"], [2, 3, 4], "historical family")
        equal(allocations["implemented_physical_cases"], [2], "historical implemented slice")
        equal(allocations["unused_cases_reallocated"], False, "unused allocations")
        equal(allocations["family_physical_confidence_implemented"], False, "family confidence")
        for key, denominator in (("statistics_family_failure_exact", 200), ("physical_family_failure_exact", 400),
                                 ("factory_family_failure_exact", 400), ("total_conditional_family_failure_exact", 100)):
            equal(allocations[key], {"numerator": 1, "denominator": denominator}, "exact family budget " + key)
        equal(set(allocations["case_allocations"]), {"2", "3", "4"}, "case allocation inventory")
        for n, steps in (("2", 69), ("3", 134), ("4", 211)):
            case = allocations["case_allocations"][n]
            equal(case["selected_steps"], steps, "allocated steps")
            equal(case["complete_trials"], 15756, "allocated complete trials")
            equal(case["physical_status"], "implemented_n2_slice" if n == "2" else "unimplemented_reserved_allocation", "historical allocation status")
            for key, denominator in (("statistical_failure_exact", 600), ("physical_failure_exact", 1200),
                                     ("factory_availability_failure_exact", 1200)):
                equal(case[key], {"numerator": 1, "denominator": denominator}, "exact case budget " + key)
            for key in ("physical_failure_budget", "factory_availability_budget"):
                equal(case[key], .0008333333333333333, "unchanged downward case budget")
                if Fraction.from_float(case[key]) > Fraction(1, 1200):
                    raise ValueError("n2 native budget rounds above its exact case share")
        workload = prepared["complete_workload"]
        for key, value in (("complete_trials", 15756), ("complete_attempts", 15756), ("algorithm_retry_cap", 1),
                           ("postselection", "none"), ("whole_workload_executed", False),
                           ("family_allocations_reallocated", False), ("deterministic_bias_multiplied_by_trials", False),
                           ("reset_per_trial", 209), ("readout_per_trial", 4),
                           ("source_resets_all_trials", 209 * 15756), ("source_readouts_all_trials", 4 * 15756),
                           ("workspace_startup_initializations_per_trial", 11),
                           ("workspace_startup_initializations_all_trials", 11 * 15756),
                           ("system_initializations_per_trial", 2), ("environment_initializations_per_trial", [138, 69]),
                           ("discard_resets_before_environment_reuse_per_trial", [137, 68])):
            equal(workload[key], value, "workload " + key)
        for key in ("physical_failure_budget", "factory_availability_budget", "statistical_failure_exact",
                    "physical_failure_exact", "factory_availability_failure_exact"):
            equal(workload[key], allocations["case_allocations"]["2"][key], "workload allocated budget " + key)
        equal(workload["conditional_case_failure_exact"], {"numerator": 1, "denominator": 300}, "conditional case share")
        primitives = {key: value for key, value in CT_COUNTS.items() if key != "barrier"}
        equal(workload["primitive_counts_per_complete_trial"], primitives, "primitive trial ledger")
        equal(workload["primitive_counts_all_trials"], {key: value * 15756 for key, value in primitives.items()}, "repeated ledger")
        for key in ("hardware_execution", "formal_confidence_certificate"):
            equal(prepared[key], False, key)
        for key in ("full_representation_certificate", "formal_synthesis_certificate", "hardware_confidence_certificate"):
            equal(prepared["error_contract"][key], False, key)
        if not isinstance(prepared["qasm_sha256"], dict):
            raise ValueError("n2 QASM bindings must be an object")
        equal(set(prepared["qasm_sha256"]), {"source-trial.qasm", "clifford-t-trial.qasm"}, "QASM inventory")
        execution, caps = manifest["execution"], manifest["run_caps"]
        equal(results["execution"], execution, "completed execution identity")
        equal(execution["run_caps"], caps, "execution caps")
        equal(prepared["run_caps"], caps, "prepared caps")
        for key, value in (("physical_cases", 1), ("physical_profiles", 2), ("density_wires", 4),
                           ("source_instructions", 20000), ("routed_and_synthesized_instructions", 500000),
                           ("density_validation_instructions", 500000), ("prepare_seconds", 600),
                           ("physical_seconds_per_profile", 180), ("family_diagnostic_seconds", 60),
                           ("parent_seconds", 1200), ("native_rotation_synthesis_attempts_upper_bound", 3312)):
            equal(caps[key], value, "run cap " + key)
        for key in ("repeated_trials_executed", "whole_family_physical_completed"):
            equal(execution[key], False, key)
        equal(set(execution["stage_status"]), {"prepare", "p1e-03", "p1e-04", "family"}, "completed stages")
        for name, stage in execution["stage_status"].items():
            equal(stage["status"], "completed", "stage status " + name)
            cap = caps["prepare_seconds" if name == "prepare" else "family_diagnostic_seconds" if name == "family" else "physical_seconds_per_profile"]
            if isinstance(stage["wall_seconds"], bool) or not isinstance(stage["wall_seconds"], (int, float)) or not 0 <= stage["wall_seconds"] <= cap:
                raise ValueError("n2 stage timing is outside its declared cap")
        if isinstance(manifest["wall_seconds"], bool) or not isinstance(manifest["wall_seconds"], (int, float)) or not 0 <= manifest["wall_seconds"] <= caps["parent_seconds"]:
            raise ValueError("n2 total elapsed time is outside its parent cap")
        equal(set(results["family"]), {"2", "3", "4"}, "historical family inventory")
        equal(results["family"]["2"]["status"], "implemented_four_wire_density_slice", "n2 family status")
        for n, steps in (("3", 134), ("4", 211)):
            record = results["family"][n]
            for key, value in (("selected_steps", steps), ("status", "ensemble_branch_cap_exceeded"),
                               ("physical_status", "unimplemented_reserved_allocation"), ("rejection_reproduced", True),
                               ("synthesis_performed", False), ("physical_estimation_performed", False),
                               ("family_allocation_reallocated", False)):
                equal(record[key], value, "historical n" + n + " " + key)
        equal(len(results["physical_profiles"]), 2, "profile count")
        for profile, rate, count in zip(results["physical_profiles"], (.001, .0001), (63, 119)):
            equal(profile["status"], "computed_conditional_instrument_surface_code_model", "completed physical status")
            equal(profile["profile"], {**PROFILE, "physical_error_rate": rate}, "frozen physical profile")
            for key, value in (("case_id", prepared["case_id"]), ("complete_algorithm_attempts", 15756),
                               ("limits", LIMITS), ("source_event_counts", SOURCE_COUNTS), ("synthesized_event_counts", CT_COUNTS),
                               ("actual_synthesis", synthesis), ("run_caps", caps), ("family_allocations", allocations)):
                equal(profile[key], value, "physical identity " + key)
            for key in ("physical_failure_budget", "factory_availability_budget"):
                equal(profile[key], workload[key], "fixed physical budget " + key)
            equal(profile["feasible_candidate_count"], count, "feasible count")
            equal(len(profile["all_feasible_points"]), count, "feasible inventory")
            equal(len(profile["pareto_points"]), 3, "frontier inventory")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("n2 status/identity record is incomplete or malformed") from error


def verify_inventory(bundle, manifest, source_root, canonical_root):
    equal({path.name for path in bundle.iterdir()}, BUNDLE_FILES, "complete 21-file bundle")
    total = sum(checked_file(bundle, name).stat().st_size for name in BUNDLE_FILES)
    if total > CAPS["bundle_bytes"]:
        raise ValueError("n2 bundle exceeds its byte cap")
    for name, relative in SOURCE_FILES.items():
        original = (checked_file(source_root, relative).read_bytes() if relative is not None else
                    checked_file(source_root, "references/simulation.bib").read_bytes() + b"\n" +
                    checked_file(source_root, "references/resources.bib").read_bytes())
        equal(hashlib.sha256(original).hexdigest(), manifest["source_sha256"][name], "current source digest " + name)
        equal(sha256(checked_file(bundle, name)), manifest["source_sha256"][name], "snapshot digest " + name)
    for name, digest in manifest["artifacts_sha256"].items():
        equal(sha256(checked_file(bundle, name)), digest, "artifact digest " + name)
    for name, digest in manifest["frozen_before"].items():
        location = canonical_root if name.startswith("assets/") else source_root
        equal(sha256(checked_file(location, name)), digest, "current frozen baseline " + name)
    equal(sha256(checked_file(bundle, "manifest.json")), REVIEWED_MANIFEST_SHA256, "immutable reviewed manifest")
    return total


def parse_qasm(text, *, width=4):
    """Parse only the frozen flat gate/reset/readout grammar; no eval or imports."""
    lines = text.splitlines()
    equal(lines[:4], ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{width}] c;", f"qubit[{width}] q;"], "QASM header")
    instructions = []
    for line in lines[4:]:
        measurement = re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];", line)
        if measurement:
            clbit, wire = map(int, measurement.groups())
            instruction = ("measure", None, (wire,), clbit)
        else:
            match = re.fullmatch(r"([a-z]+)(?:\(([^()]*)\))? (q\[\d+\](?:, q\[\d+\])*)\;", line)
            if match is None:
                raise ValueError("n2 QASM contains an unsupported declaration or instruction")
            name, value, operands = match.groups()
            wires = tuple(map(int, re.findall(r"q\[(\d+)\]", operands)))
            angle = float(value) if value is not None else None
            if angle is not None and not math.isfinite(angle):
                raise ValueError("n2 QASM rotation is nonfinite")
            if name not in {"h", "s", "sdg", "x", "y", "z", "t", "tdg", "rx", "ry", "rz", "cx", "cry", "reset", "barrier"}:
                raise ValueError("n2 QASM has an opaque operation")
            arity = width if name == "barrier" else 2 if name in {"cx", "cry"} else 1
            if len(wires) != arity or len(set(wires)) != len(wires) or (name in {"rx", "ry", "rz", "cry"}) != (angle is not None):
                raise ValueError("n2 QASM operands or angle arity are invalid")
            instruction = (name, angle, wires, None)
        if any(not 0 <= wire < width for wire in instruction[2]) or (instruction[3] is not None and not 0 <= instruction[3] < width):
            raise ValueError("n2 QASM wire/readout is outside its width")
        instructions.append(instruction)
        if len(instructions) > CAPS["qasm_instructions"]:
            raise ValueError("n2 QASM instruction cap exceeded")
    return instructions


def local_matrix(name, angle, np):
    i = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    z = np.diag([1, -1]).astype(complex)
    if name == "h":
        return np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
    if name in {"x", "y", "z"}:
        return {"x": x, "y": y, "z": z}[name]
    if name in {"s", "sdg", "t", "tdg"}:
        phase = {"s": math.pi/2, "sdg": -math.pi/2, "t": math.pi/4, "tdg": -math.pi/4}[name]
        return np.diag([1, np.exp(1j * phase)]).astype(complex)
    if name in {"rx", "ry", "rz", "cry"}:
        pauli = {"rx": x, "ry": y, "rz": z, "cry": y}[name]
        return math.cos(angle / 2) * i - 1j * math.sin(angle / 2) * pauli
    raise ValueError("n2 density replay has an unsupported local unitary")


def expanded_matrix(instruction, np, width=4):
    name, angle, wires, _ = instruction
    size = 1 << width
    result = np.zeros((size, size), complex)
    if name == "cx":
        control, target = wires
        for column in range(size):
            result[column ^ ((1 << target) if column & (1 << control) else 0), column] = 1
        return result
    local = local_matrix(name, angle, np)
    target = wires[-1]
    for column in range(size):
        if name == "cry" and not column & (1 << wires[0]):
            result[column, column] = 1
        else:
            low = column & ~(1 << target)
            for value in range(2):
                result[low | (value << target), column] = local[value, (column >> target) & 1]
    return result


def reduced_wire(rho, wire, np, width=4):
    mask = 1 << wire
    indices = [index for index in range(1 << width) if not index & mask]
    return np.array([[sum(rho[index | (a * mask), index | (b * mask)] for index in indices)
                      for b in range(2)] for a in range(2)], complex)


def replay_density(instructions, deadline, np):
    """Full 16x16 density, exact reset Kraus sum, and discarded readout dephasing."""
    deadline_check(deadline)
    rho = np.zeros((16, 16), complex)
    rho[0, 0] = 1
    segment = np.eye(16, dtype=complex)
    matrices, seen_measurements, before = {}, {}, None
    output_wire = next((item[2][0] for item in instructions if item[0] == "measure" and item[3] == 0), None)
    if output_wire is None:
        raise ValueError("n2 replay lacks explicit output bit0")
    for index, item in enumerate(instructions):
        if index % 2048 == 0:
            deadline_check(deadline)
        name, angle, wires, clbit = item
        if name == "barrier":
            continue
        if name in {"reset", "measure"}:
            rho = segment @ rho @ segment.conj().T
            segment = np.eye(16, dtype=complex)
            mask = 1 << wires[0]
            zero = [i for i in range(16) if not i & mask]
            one = [i | mask for i in zero]
            if name == "reset":
                if seen_measurements:
                    raise ValueError("n2 reset follows output readout")
                result = np.zeros_like(rho)
                result[np.ix_(zero, zero)] = rho[np.ix_(zero, zero)] + rho[np.ix_(one, one)]
                rho = result
            else:
                if before is None:
                    before = reduced_wire(rho, output_wire, np)
                if clbit in seen_measurements or wires[0] in seen_measurements.values():
                    raise ValueError("n2 output targets are not distinct")
                seen_measurements[clbit] = wires[0]
                rho[np.ix_(zero, one)] = 0
                rho[np.ix_(one, zero)] = 0
        else:
            if seen_measurements:
                raise ValueError("n2 unitary follows output readout")
            if item not in matrices:
                matrices[item] = expanded_matrix(item, np)
            segment = matrices[item] @ segment
    equal(set(seen_measurements), {0, 1, 2, 3}, "complete readout bits")
    probability = float(np.trace(rho).real)
    if probability <= 0:
        raise ValueError("n2 density replay has invalid normalization")
    normalized = before / probability
    return {"acceptance_probability": probability, "mean": float((normalized[0, 0] - normalized[1, 1]).real),
            "reduced_before_output": normalized, "output_wires": seen_measurements}


def routed_instructions(source):
    """Reconstruct the documented pinned compiler recipe, without synthesis."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
    circuit = QuantumCircuit(4, 4)
    for name, angle, wires, clbit in source:
        if name in {"reset", "measure"}:
            circuit.barrier()
        if name == "measure":
            circuit.measure(wires[0], clbit)
        elif angle is None:
            getattr(circuit, name)(*wires)
        else:
            getattr(circuit, name)(angle, *wires)
        if name in {"reset", "measure"}:
            circuit.barrier()
    basis = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
    lowered = transpile(circuit, basis_gates=basis, optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)
    routed = transpile(lowered, basis_gates=basis, coupling_map=CouplingMap.from_line(4, bidirectional=True),
                       initial_layout=[0, 1, 2, 3], routing_method="sabre", optimization_level=0,
                       seed_transpiler=7, qubits_initially_zero=False)
    result = []
    for item in routed.data:
        wires = tuple(routed.find_bit(bit).index for bit in item.qubits)
        angle = float(item.operation.params[0]) if item.operation.name in {"rx", "ry", "rz"} else None
        clbit = routed.find_bit(item.clbits[0]).index if item.clbits else None
        result.append((item.operation.name, angle, wires, clbit))
    return result, float(routed.global_phase)


def replay_synthesis(routed, compiled, synthesis, deadline, np):
    """Bind every actual CT subsequence to its routed target and saved guard."""
    targets = {(row["gate"], row["angle_hex"]): row for row in synthesis["approximants"]}
    equal(len(targets), 6, "unique synthesis targets")
    cursor, frequencies, first_sequences, errors, phases = 0, Counter(), {}, [], []
    for index, item in enumerate(routed):
        if index % 512 == 0:
            deadline_check(deadline)
        name, angle, wires, _ = item
        if name in {"rx", "ry", "rz"}:
            key = (name, angle.hex())
            if key not in targets:
                raise ValueError("n2 routed rotation is absent from saved synthesis targets")
            target = targets[key]
            size = sum(target["gate_counts"].values())
            sequence = tuple(compiled[cursor:cursor + size])
            if len(sequence) != size or any(row[0] not in {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
                                            or row[2] != wires or row[1] is not None for row in sequence):
                raise ValueError("n2 CT sequence does not act on its routed target")
            equal(dict(Counter(row[0] for row in sequence)), target["gate_counts"], "actual synthesis sequence counts")
            signature = tuple(row[0] for row in sequence)
            if key in first_sequences:
                equal(signature, first_sequences[key], "repeated synthesis sequence identity")
            else:
                first_sequences[key] = signature
                matrix = np.eye(2, dtype=complex)
                for row in sequence:
                    matrix = local_matrix(row[0], None, np) @ matrix
                matrix *= np.exp(1j * target["global_phase"])
                error = float(np.linalg.norm(matrix - local_matrix(name, angle, np), ord="fro")) + synthesis["per_matrix_frobenius_guard"]
                close(target["guarded_frobenius_error"], error, "actual sequence Frobenius guard", absolute=2e-14)
                if error > synthesis["operator_error_allowance"] / 828 + 2e-14:
                    raise ValueError("n2 actual sequence exceeds its allocated numerical guard")
            frequencies[key] += 1
            errors.append(target["guarded_frobenius_error"])
            phases.append(target["global_phase"])
            cursor += size
        else:
            if cursor >= len(compiled) or compiled[cursor] != item:
                raise ValueError("n2 nonrotation event changed across routed/CT schedules")
            cursor += 1
    equal(cursor, len(compiled), "complete CT sequence consumption")
    equal(sum(frequencies.values()), 828, "charged rotation occurrences")
    close(synthesis["guarded_numerical_operator_error_upper_estimate"], math.fsum(errors), "whole synthesis guard", absolute=1e-18)
    return frequencies, phases


def reference_mean(np):
    """Independent column-vectorized 16x16 GKSL exponential for the fixed task."""
    from scipy.linalg import expm
    i = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], complex)
    z = np.diag([1, -1]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], complex)
    x0, x1 = np.kron(i, x), np.kron(x, i)
    z0, z1 = np.kron(i, z), np.kron(z, i)
    h = .6 * z0 @ z1 + .35 * (x0 + x1)
    identity = np.eye(4, dtype=complex)
    generator = -1j * (np.kron(identity, h) - np.kron(h.T, identity))
    for collapse in (math.sqrt(.8) * np.kron(i, lowering), math.sqrt(.5) * np.kron(lowering, i)):
        square = collapse.conj().T @ collapse
        generator += np.kron(collapse.conj(), collapse) - .5 * (np.kron(identity, square) + np.kron(square.T, identity))
    state = np.kron(np.array([1, 1]) / math.sqrt(2), np.array([1, 1j]) / math.sqrt(2))
    rho = np.outer(state, state.conj())
    evolved = (expm(generator) @ rho.reshape(16, order="F")).reshape((4, 4), order="F")
    return float(np.trace(z0 @ evolved).real)


def macro_cost(profile, distance, name, factory_ns, retries):
    """Independent integer implementation of the declared serialized macros."""
    cycle, reaction = profile["code_cycle_time_ns"], profile["reaction_time_ns"]
    init = profile["physical_initialization_time_ns"] + distance * cycle + reaction
    destructive = profile["physical_measurement_time_ns"] + reaction
    parity = 2 * init + distance * cycle + destructive
    cnot = 3 * parity + 2 * reaction
    if name == "initial_reset":
        return {"time_ns": init, "readouts": 0, "preparations": 1, "decisions": 1}
    if name == "reset":
        return {"time_ns": parity + 2 * reaction + distance * cycle, "readouts": 1, "preparations": 2, "decisions": 5}
    if name == "measure":
        return {"time_ns": parity, "readouts": 1, "preparations": 2, "decisions": 3}
    if name == "cx":
        return {"time_ns": cnot, "readouts": 3, "preparations": 6, "decisions": 11}
    if name == "trial_control" or name in {"h", "x", "y", "z", "s", "sdg"}:
        return {"time_ns": reaction, "readouts": 0, "preparations": 0, "decisions": 1}
    if name.startswith(("t_q", "tdg_q")):
        wire = int(name.split("_q")[1])
        cnots, inverse = 3 * (wire + 1) + 1, int(name.startswith("tdg"))
        round_time = profile["physical_initialization_time_ns"] + profile["physical_measurement_time_ns"] + reaction
        supply = retries * (factory_ns + profile["factory_feedback_rounds_reserved"] * round_time + reaction)
        return {"time_ns": supply + cnots * cnot + destructive + reaction + init + inverse * reaction,
                "readouts": 3 * cnots + 1, "preparations": 6 * cnots + 1,
                "decisions": 11 * cnots + 3 + inverse + retries * (profile["factory_feedback_rounds_reserved"] + 1)}
    raise ValueError("n2 ledger contains an unsupported macro")


def physical_event_counts(instructions):
    counts, active = Counter(), set()
    for name, _, wires, _ in instructions:
        if name == "barrier":
            continue
        key = "initial_reset" if name == "reset" and wires[0] not in active else name
        if name in {"t", "tdg"}:
            key = name + "_q" + str(wires[0])
        counts[key] += 1
        active.update(wires)
    counts["trial_control"] += 1
    counts["initial_reset"] += 11
    return counts


def replay_physical(profile, counts, deadline):
    """Recompute every feasible ledger using only the frozen factory calibration."""
    hardware = profile["profile"]
    trials, patches = 15756, 15
    total_t = trials * sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    failure = Fraction(str(hardware["factory_pipeline_failure_bound"]))
    budget = Fraction.from_float(profile["factory_availability_budget"])
    retries = next((cap for cap in range(1, 17) if total_t * failure**cap <= budget), None)
    if retries is None:
        raise ValueError("n2 frozen factory retry domain has no feasible cap")
    availability = float(total_t * failure**retries)
    points = profile["all_feasible_points"]
    for point in points:
        deadline_check(deadline)
        distance, factory = point["data_code_distance"], point["factory_supply"]
        if type(distance) is not int or not 3 <= distance <= 49 or distance % 2 != 1:
            raise ValueError("n2 physical point has invalid code distance")
        equal(point["reserved_logical_patches"], patches, "reserved patches")
        equal(point["data_workspace_physical_qubits"], patches * (2 * distance**2 - 1), "data space")
        equal(point["peak_physical_qubits"], point["data_workspace_physical_qubits"] + point["factory_physical_qubits"], "peak space sum")
        for key in ("factory_physical_qubits", "peak_physical_qubits"):
            if type(point[key]) is not int or point[key] <= 0:
                raise ValueError("n2 physical cost is fabricated or invalid")
        factory_ns, factory_error = factory["pipeline_time_ns"], factory["output_error"]
        if type(factory_ns) is not int or factory_ns <= 0 or not 0 < factory_error < 7.1 * hardware["physical_error_rate"]:
            raise ValueError("n2 saved factory calibration is invalid")
        for key, expected in (("fixed_logical_distillation_distance", distance), ("native_qec_crossing_prefactor", .03),
                              ("native_qec_threshold", .01), ("native_code_cycle_time_ns", 1000),
                              ("full_pipeline_retry_cap_per_state", retries), ("states_all_complete_attempts", total_t),
                              ("worst_case_pipeline_attempts", total_t * retries), ("pipeline_feedback_rounds_reserved", 3)):
            equal(factory[key], expected, "factory calibration " + key)
        close(factory["availability_failure_union_bound"], availability, "factory readiness")
        if availability > profile["factory_availability_budget"] or factory_error * total_t > profile["physical_failure_budget"] * .75:
            raise ValueError("n2 saved factory violates its frozen budget")
        physical_error = hardware["physical_error_rate"]
        parity_error = -math.expm1(distance * math.log1p(-2 * physical_error)) / 2
        readout_error = math.fsum(math.comb(distance, k) * parity_error**k * (1 - parity_error)**(distance-k)
                                  for k in range((distance+1)//2, distance+1))
        patch_error = hardware["memory_prefactor"] * (physical_error / hardware["memory_threshold"])**((distance+1)//2)
        preparation_error = hardware["initialization_residual_prefactor"] * (physical_error / hardware["memory_threshold"])**((distance+1)//2)
        rows = point["macro_ledger_per_complete_attempt"]
        equal([row["operation"] for row in rows], sorted(counts), "macro inventory")
        runtime, readouts, preparations, decisions = 0, 0, 0, 0
        for row in rows:
            name = row["operation"]
            cost = macro_cost(hardware, distance, name, factory_ns, retries)
            equal(row["count"], counts[name], "macro count " + name)
            for key, value in cost.items():
                equal(row[key], value, "macro cost " + name + " " + key)
            close(row["event_failure_model"], cost["readouts"] * readout_error + cost["preparations"] * preparation_error
                  + cost["decisions"] * hardware["controller_failure_per_decision"], "macro fault " + name)
            runtime += counts[name] * cost["time_ns"]
            readouts += counts[name] * cost["readouts"]
            preparations += counts[name] * cost["preparations"]
            decisions += counts[name] * cost["decisions"]
        cycles = -(-runtime // hardware["code_cycle_time_ns"])
        round_time = hardware["physical_initialization_time_ns"] + hardware["physical_measurement_time_ns"] + hardware["reaction_time_ns"]
        extra_dispatch = 3 * round_time + hardware["reaction_time_ns"]
        hold = extra_dispatch + 3 * macro_cost(hardware, distance, "cx", factory_ns, retries)["time_ns"]
        equal(factory["extra_init_readout_feedback_dispatch_ns_per_pipeline"], extra_dispatch, "factory round overhead")
        equal(factory["extra_output_port_hold_ns_per_state"], hold, "factory port residence")
        components = {"data_workspace_code_exposure": patches * cycles * patch_error * trials,
                      "factory_output_handoff_storage": (total_t // trials) * -(-hold // hardware["code_cycle_time_ns"]) * patch_error * trials,
                      "logical_readout_scenario": readouts * readout_error * trials,
                      "logical_initialization_scenario": preparations * preparation_error * trials,
                      "controller_scenario": decisions * hardware["controller_failure_per_decision"] * trials,
                      "factory_output_quality_model": total_t * factory_error}
        equal(set(point["failure_components_all_attempts"]), set(components), "fault-component inventory")
        for name, value in components.items():
            close(point["failure_components_all_attempts"][name], value, "all-trial fault " + name)
        equal(point["runtime_per_complete_attempt_ns"], runtime, "integer runtime")
        equal(point["worst_case_runtime_all_attempts_ns"], runtime * trials, "complete population runtime")
        equal(point["code_cycles_per_complete_attempt"], cycles, "code cycles")
        close(point["full_physical_failure_model"], math.fsum(components.values()), "aggregate physical fault")
        if point["full_physical_failure_model"] > profile["physical_failure_budget"]:
            raise ValueError("n2 reconstructed physical point violates its frozen budget")
    frontier = []
    for point in sorted(points, key=lambda value: (value["peak_physical_qubits"], value["runtime_per_complete_attempt_ns"])):
        if not frontier or point["runtime_per_complete_attempt_ns"] < frontier[-1]["runtime_per_complete_attempt_ns"]:
            frontier.append(point)
    equal(profile["pareto_points"], frontier, "independent Pareto frontier")
    diagnostic = profile["controller_model_diagnostic"]
    trial_decisions = sum(count * macro_cost(hardware, 3, name, 0, retries)["decisions"] for name, count in counts.items())
    equal(diagnostic["controller_decisions_per_trial"], trial_decisions, "controller trial decisions")
    equal(diagnostic["controller_decisions_all_trials"], trial_decisions * trials, "controller population decisions")
    equal(diagnostic["factory_retry_cap"], retries, "minimal factory retry cap")
    close(diagnostic["distance_independent_controller_model_contribution"], trial_decisions * trials * 1e-15, "controller contribution")
    equal(diagnostic["budget_exceeded"], False, "controller feasibility")
    return len(points)


def replay_science(bundle, results, deadline):
    import numpy as np
    for package, version in PACKAGES.items():
        equal(importlib.metadata.version(package), version, "installed " + package)
    equal(".".join(map(str, sys.version_info[:3])), "3.12.14", "installed Python")
    prepared = results["prepared"]
    source = parse_qasm(checked_file(bundle, "source-trial.qasm").read_text())
    compiled = parse_qasm(checked_file(bundle, "clifford-t-trial.qasm").read_text())
    equal(dict(Counter(row[0] for row in source)), SOURCE_COUNTS, "actual source primitives")
    equal(dict(Counter(row[0] for row in compiled)), CT_COUNTS, "actual CT primitives")
    routed, routed_phase = routed_instructions(source)
    deadline_check(deadline)
    synthesis = prepared["actual_synthesis"]
    equal(dict(Counter(row[0] for row in routed)), synthesis["routed_counts"], "reconstructed routed primitives")
    frequencies, phases = replay_synthesis(routed, compiled, synthesis, deadline, np)
    angle_residual = math.remainder(routed_phase + math.fsum(phases) - synthesis["compiled_global_phase"], 2 * math.pi)
    close(angle_residual, 0., "compiled phase restoration", absolute=2e-10)
    checks = {}
    for name, instructions in (("source_instrument_check", source), ("routed_instrument_check", routed),
                               ("clifford_t_instrument_check", compiled)):
        replay = replay_density(instructions, deadline, np)
        saved = prepared[name]
        density = np.array(saved["reduced_before_output"]["real"]) + 1j * np.array(saved["reduced_before_output"]["imag"])
        if np.linalg.norm(replay["reduced_before_output"] - density, ord="nuc") > 2e-10:
            raise ValueError("n2 full-density/output replay differs from its saved check")
        close(saved["acceptance_probability"], replay["acceptance_probability"], name + " trace", absolute=2e-10)
        close(saved["mean"], replay["mean"], name + " Z0", absolute=2e-10)
        equal(saved["normalization_probability_is_postselection"], False, name + " postselection scope")
        checks[name] = replay
    exact = reference_mean(np)
    close(prepared["independent_reference"]["exact_GKSL_Z0"], exact, "independent GKSL", absolute=2e-13)
    contract = prepared["error_contract"]
    equal(contract["splitting_bias_exact"], {"numerator": 17576, "denominator": 1785375}, "exact splitting policy")
    guarded = synthesis["guarded_numerical_operator_error_upper_estimate"]
    eta = Fraction.from_float(guarded) + Fraction("1e-7") + Fraction("1e-7")
    equal(contract["whole_realization_error_exact"], {"numerator": eta.numerator, "denominator": eta.denominator}, "exact realization contract")
    close(contract["realization_Z0_bias_upper_proposal"], float(2 * eta), "unconditional coherent bias", absolute=1e-18)
    bound = Fraction(17576, 1785375) + 2 * eta + Fraction(".03")
    close(contract["absolute_Z0_error_upper_proposal"], float(bound), "total absolute-error proposal", absolute=1e-16)
    if abs(checks["clifford_t_instrument_check"]["mean"] - exact) > float(bound) - .03:
        raise ValueError("n2 reconstructed answer violates the conditional bias contract")
    counts = physical_event_counts(compiled)
    points = sum(replay_physical(profile, counts, deadline) for profile in results["physical_profiles"])
    equal(points, CAPS["physical_points"], "replayed physical point count")
    deadline_check(deadline)
    return {"qasm_instructions": len(source) + len(compiled), "rotation_occurrences": sum(frequencies.values()),
            "density_replays": 3, "physical_points": points, "native_synthesis_calls": 0, "native_factory_queries": 0}


def verify_bundle(bundle: Path, *, source_root: Path | None = None, canonical_root: Path | None = None, replay=True):
    started = time.monotonic()
    deadline = started + CAPS["replay_seconds"]
    bundle = Path(bundle).absolute()
    source_root = bundle.parents[3] if source_root is None else Path(source_root).absolute()
    canonical_root = source_root if canonical_root is None else Path(canonical_root).absolute()
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("n2 bundle must be a regular directory")
    manifest = load_json(checked_file(bundle, "manifest.json"))
    results = load_json(checked_file(bundle, "results.json"))
    verify_records(manifest, results)
    total = verify_inventory(bundle, manifest, source_root, canonical_root)
    for name, digest in results["prepared"]["qasm_sha256"].items():
        equal(sha256(checked_file(bundle, name)), digest, "QASM result binding " + name)
    science = replay_science(bundle, results, deadline) if replay else {}
    return {"files": len(BUNDLE_FILES), "bytes": total, "manifest_sha256": REVIEWED_MANIFEST_SHA256,
            **science, "verification_wall_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        # The CLI's parent bound also interrupts a blocked compiler/import.
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--root", str(args.root.resolve())],
                           check=True, timeout=CAPS["replay_seconds"])
        except subprocess.TimeoutExpired as error:
            raise SystemExit("n2 immutable verification exceeded its 60-second parent cap") from error
        return
    receipt = verify_bundle(args.root / BUNDLE_RELATIVE, source_root=args.root, canonical_root=args.root)
    print("Verified immutable n2 numerical/compiler/instrument/physical replay:", json.dumps(receipt, sort_keys=True),
          "saved factory calibration remains conditional; no native synthesis/query or repeated hardware execution")


if __name__ == "__main__":
    main()
