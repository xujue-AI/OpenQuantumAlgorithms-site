"""Optional numerical replay of authenticated, immutable measured trials.

Call run_replay only from the parent verifier's isolated, monitored worker,
after the parent has authenticated its complete approved inventory. There is
no standalone file-reading replay CLI. This module imports only stdlib until
that call has checked the supplied bytes. It never regenerates QASM, fits
phases, synthesizes rotations, invokes Native/QDK, or reprices saved decisions.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import types


class ReplayError(ValueError):
    """The frozen inputs or their bounded numerical replay did not agree."""


MAX_RECORD_BYTES = 16 * 1024 * 1024
_CASES = {
    "instrument": ("damping-Z", "lindblad-Z", "thermal-X", "thermal-Z"),
    "complete": ("dynamics-n1-Z", "dynamics-n2-Z", "thermal-n1-X", "thermal-n2-X", "thermal-n2-Z"),
}
_HELPER_FUNCTIONS = {"_limits", "_source_check", "reset_kraus", "simulate_trial", "simulate_scored_trial"}
_HELPER_CONSTANTS = {"MAX_QUBITS", "MAX_SOURCE_OPERATIONS", "MAX_SYNTHESIZED_OPERATIONS", "_DIRECT_UNITARY_CLASSES"}
_FORBIDDEN_SYMBOLS = {
    "gridsynth_unitary", "synthesize_instrument", "estimate_instrument", "Native",
    "build_complete_task", "generate_dynamics_phases", "generate_thermal_phases",
    "transpile", "eval", "exec", "compile", "__import__", "open",
}


@dataclass(frozen=True)
class _Trial:
    name: str
    report: dict
    source: object
    compiled: object
    outcome: tuple[int, ...]
    scores: tuple[float, ...]
    accept: dict[int, int]
    source_record: dict
    compiled_record: dict


def _bytes(inventory: dict[str, bytes], name: str) -> bytes:
    data = inventory.get(name)
    if not isinstance(data, bytes) or len(data) > MAX_RECORD_BYTES:
        raise ReplayError(f"Missing, invalid or oversized approved bytes: {name}")
    return data


def _json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReplayError("Duplicate key in approved JSON")
            result[key] = value
        return result

    def nonfinite(_):
        raise ReplayError("Nonfinite literal in approved JSON")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReplayError("Invalid approved JSON") from error
    if not isinstance(value, dict):
        raise ReplayError("Approved report must be a JSON object")
    return value


def _helper_ast(data: bytes) -> ast.Module:
    """Select original numerical definitions, never execute snapshot imports."""
    tree = ast.parse(data, filename="<authenticated instrument_resources.py>")
    selected = []
    found = set()
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
            approved = name in _HELPER_FUNCTIONS or name == "InstrumentLimits"
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            approved = name in _HELPER_CONSTANTS
        else:
            approved = False
        if not approved:
            continue
        if name in found:
            raise ReplayError("Duplicate selected numerical helper definition")
        found.add(name)
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                raise ReplayError("Numerical helper contains an unexpected import")
            if isinstance(child, ast.Name) and child.id in _FORBIDDEN_SYMBOLS:
                raise ReplayError("Numerical helper references a forbidden entry point")
            if isinstance(child, ast.Attribute) and (child.attr in _FORBIDDEN_SYMBOLS or child.attr.startswith("__")):
                raise ReplayError("Numerical helper references a forbidden attribute")
        selected.append(node)
    expected = _HELPER_FUNCTIONS | _HELPER_CONSTANTS | {"InstrumentLimits"}
    if found != expected:
        raise ReplayError("Authenticated snapshot lacks the exact numerical helper allowlist")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    return ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))


def _events(parsed) -> tuple:
    # Routing may permute physical quantum wires. Classical-bit assignment and
    # reset/measurement event order remain explicit; each final wire map is
    # checked independently against that circuit's saved numerical record.
    return tuple((item.name, item.clbits) for item in parsed.events())


def _preflight(kind, records_bytes, source_bytes, parser_module) -> tuple[list[_Trial], ast.Module]:
    if kind not in _CASES or not isinstance(records_bytes, dict) or not isinstance(source_bytes, dict):
        raise ReplayError("Replay requires a known kind and authenticated byte inventories")
    helper = _bytes(records_bytes, "instrument_resources.py")
    if helper != _bytes(source_bytes, "src/openquantum/instrument_resources.py"):
        raise ReplayError("Bundle helper differs from the authenticated project source")
    helper_tree = _helper_ast(helper)
    parsed_trials = []
    for name in _CASES[kind]:
        report = _json(_bytes(records_bytes, f"{name}/report.json"))
        if kind == "complete":
            report = report.get("prepared")
            if not isinstance(report, dict) or report.get("case_id") != name:
                raise ReplayError("Complete prepared case identity mismatch")
            outcome = tuple(report["outcome_clbits"])
            scores = tuple(report["score_values"])
            accept = {int(k): v for k, v in report["accept_clbits"].items()}
            source_record, compiled_record = report["source_instrument_check"], report["clifford_t_instrument_check"]
            hashes = report["qasm_sha256"]
        else:
            if report.get("task") != name:
                raise ReplayError("Instrument task identity mismatch")
            outcome, scores = (report["output_clbit"],), (1.0, -1.0)
            accept = {int(k): v for k, v in report["acceptance_clbits"].items()}
            source_record, compiled_record = report["ideal_instrument"], report["clifford_t_instrument"]
            hashes = {"source-instrument.qasm": report["source_qasm_sha256"],
                      "clifford-t-instrument.qasm": report["clifford_t_qasm_sha256"]}
        circuits = []
        for filename in ("source-instrument.qasm", "clifford-t-instrument.qasm"):
            data = _bytes(records_bytes, f"{name}/{filename}")
            if hashlib.sha256(data).hexdigest() != hashes[filename]:
                raise ReplayError("Approved report/QASM hash mismatch")
            circuits.append(parser_module.parse_qasm(data, max_qubits=8))
        source, compiled = circuits
        if source.num_qubits != compiled.num_qubits or source.num_clbits != compiled.num_clbits:
            raise ReplayError("Source/compiled register widths disagree")
        if _events(source) != _events(compiled):
            raise ReplayError("Source/compiled fixed instrument event sequence disagrees")
        if any(item.name not in parser_module.CLIFFORD_T_GATES | {"reset", "measure", "barrier"}
               for item in compiled.operations):
            raise ReplayError("Saved compiled QASM is outside primitive Clifford+T")
        if compiled.counts() != report["actual_synthesis"]["clifford_t_counts"]:
            raise ReplayError("Saved primitive counts disagree with actual compiled QASM")
        if (not outcome or len(set(outcome)) != len(outcome) or any(isinstance(bit, bool) or not isinstance(bit, int) or not 0 <= bit < source.num_clbits for bit in outcome)
                or set(outcome) & set(accept) or any(bit < 0 or bit >= source.num_clbits or value not in (0, 1) for bit, value in accept.items())):
            raise ReplayError("Invalid saved outcome/acceptance contract")
        if len(scores) != 2 ** len(outcome) or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in scores):
            raise ReplayError("Invalid saved bounded score table")
        required = {"acceptance_probability", "mean", "reduced_before_output"}
        if kind == "complete":
            required |= {"joint_probabilities", "outcome_clbits", "outcome_wires", "score_values"}
        if any(not isinstance(saved, dict) or not required <= saved.keys()
               for saved in (source_record, compiled_record)):
            raise ReplayError("Saved numerical report is missing required evidence fields")
        parsed_trials.append(_Trial(name, report, source, compiled, outcome, scores, accept, source_record, compiled_record))
    return parsed_trials, helper_tree


def _circuit(parsed, quantum_circuit):
    circuit = quantum_circuit(parsed.num_qubits, parsed.num_clbits)
    for item in parsed.operations:
        if item.name == "measure":
            circuit.measure(item.qubits[0], item.clbits[0])
        elif item.name == "reset":
            circuit.reset(item.qubits[0])
        elif item.name == "barrier":
            circuit.barrier(*item.qubits)
        else:
            name = "u" if item.name == "U" else item.name
            getattr(circuit, name)(*item.parameters, *item.qubits)
    return circuit


def _field_residuals(actual, saved, np, *, joint: bool = False) -> dict:
    """Collect typed evidence without approving a numerical compatibility rule."""
    residuals = {}
    required = {"acceptance_probability", "mean", "reduced_before_output"}
    if joint:
        required |= {"joint_probabilities", "outcome_clbits", "outcome_wires", "score_values"}
    if not isinstance(saved, dict) or not required <= saved.keys() or not required <= actual.keys():
        raise ReplayError("Numerical replay is missing required evidence fields")
    for field in ("acceptance_probability", "mean", "joint_probabilities", "reduced_before_output"):
        if field not in saved:
            continue
        left = np.asarray(actual[field])
        right = saved[field]
        if field == "reduced_before_output":
            if not isinstance(right, dict) or set(right) != {"real", "imag"}:
                raise ReplayError("Invalid saved complex matrix encoding")
            right = np.asarray(right["real"]) + 1j * np.asarray(right["imag"])
        else:
            right = np.asarray(right)
        if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ReplayError("Replay/saved numerical shapes or finiteness disagree")
        delta = left-right
        if left.shape == ():
            residuals[field] = {"kind": "scalar", "actual": float(left), "saved": float(right),
                                "signed_difference": float(delta), "maximum_absolute_difference": float(abs(delta))}
        elif field == "reduced_before_output":
            residuals[field] = {"kind": "complex_matrix", "shape": list(left.shape),
                                "maximum_absolute_difference": float(np.max(np.abs(delta))),
                                "maximum_real_absolute_difference": float(np.max(np.abs(delta.real))),
                                "maximum_imaginary_absolute_difference": float(np.max(np.abs(delta.imag))),
                                "frobenius_difference": float(np.linalg.norm(delta, "fro"))}
        else:
            residuals[field] = {"kind": "real_vector", "shape": list(left.shape),
                                "maximum_absolute_difference": float(np.max(np.abs(delta))),
                                "l1_difference": float(np.sum(np.abs(delta)))}
    for field in ("outcome_clbits", "outcome_wires", "score_values"):
        if field in saved and list(actual[field]) != saved[field]:
            raise ReplayError("Replay/saved final output mapping disagrees")
    return residuals


def _independent_reference(trial: _Trial, kind: str, np, expm) -> tuple[float, float, float | None]:
    """Small Kraus/Liouvillian/exponential references; no project generators."""
    x = np.array([[0, 1], [1, 0]], complex)
    z = np.diag([1, -1]).astype(complex)
    identity = np.eye(2, dtype=complex)
    if kind == "instrument":
        if trial.name == "damping-Z":
            survival = math.exp(-0.8)
            kraus = (np.diag([1, math.sqrt(survival)]), np.array([[0, math.sqrt(1-survival)], [0, 0]], complex))
            rho = np.full((2, 2), .5, complex)
            reference = sum(k @ rho @ k.conj().T for k in kraus)
            exact_mean = float(np.trace(z @ reference).real)
        elif trial.name == "lindblad-Z":
            hamiltonian = .7*x + .3*z
            jump = np.array([[0, 1], [0, 0]], complex)
            loss = jump.conj().T @ jump
            generator = -1j*(np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))
            generator += .8*(np.kron(jump.conj(), jump) - .5*np.kron(identity, loss) - .5*np.kron(loss.T, identity))
            rho = .5*np.array([[1, -1j], [1j, 1]], complex)
            reference = (expm(generator) @ rho.reshape(4, order="F")).reshape((2, 2), order="F")
            exact_mean = float(np.trace(z @ reference).real)
        else:
            reference = expm(-1.25*(.6*z+.4*x))
            reference /= np.trace(reference)
            exact_mean = float(np.trace((x if trial.name.endswith("X") else z) @ reference).real)
        return exact_mean, float(trial.report["workload"]["algorithm_bias_numerical"]), None
    n = trial.report["n_system"]
    if n not in (1, 2):
        raise ReplayError("Independent reference is restricted to stored n1/n2 cases")
    hamiltonian = np.zeros((2**n, 2**n), complex)
    matrices = {"I": identity, "X": x, "Z": z}
    for coefficient, pauli in trial.report["Hamiltonian_terms"]:
        if len(pauli) != n or any(letter not in matrices for letter in pauli):
            raise ReplayError("Unsupported saved reference Pauli input")
        term = np.ones((1, 1), complex)
        for letter in pauli:
            term = np.kron(term, matrices[letter])
        hamiltonian += coefficient*term
    if trial.name.startswith("dynamics"):
        state = expm(-.5j*hamiltonian)[:, 0]
        observable = z if n == 1 else np.kron(identity, z)
        exact_mean = float(np.vdot(state, observable @ state).real)
    else:
        reference = expm(-1.5*hamiltonian)
        reference /= np.trace(reference)
        if trial.name.endswith("X"):
            observable = .35*x if n == 1 else .35*(np.kron(identity, x)+np.kron(x, identity))
        else:
            observable = .6*np.kron(z, z)
        exact_mean = float(np.trace(observable @ reference).real)
    return exact_mean, float(trial.report["error_and_conditioning_contract"]["source_score_bias_upper"]), float(trial.report["exact_reference_mean"])


def run_replay(kind, records_bytes, source_bytes, parser_module):
    """Replay approved bytes inside a watchdog child; return bounded JSON data.

    Authentication is the parent's responsibility. This second preflight checks
    source/snapshot identity, report-linked hashes, grammar and primitive counts
    before importing numerical packages. The parent enforces <=540 s replay and
    <=2 GiB process-tree RSS, bounds logs/output, and cleans descendants. The
    replay cannot provide those process protections itself.
    """
    trials, helper_tree = _preflight(kind, records_bytes, source_bytes, parser_module)
    # Numerical imports occur only after byte-inventory and grammar preflight.
    import numpy as np
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.circuit import Barrier, Measure, Reset
    from qiskit.circuit.library import (CXGate, CZGate, HGate, IGate, PhaseGate, RXGate,
                                       RYGate, RZGate, SGate, SdgGate, TGate, TdgGate,
                                       UGate, XGate, YGate, ZGate)
    from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Statevector, partial_trace
    from dataclasses import asdict, dataclass

    module_name = "_tail_authenticated_numerical_instrument"
    if module_name in sys.modules:
        raise ReplayError("Replay worker must start with a fresh numerical helper namespace")
    helper = types.ModuleType(module_name)
    helper.__dict__.update({"np": np, "math": math, "QuantumCircuit": QuantumCircuit,
                           "Barrier": Barrier, "Measure": Measure, "Reset": Reset,
                           "Kraus": Kraus, "Statevector": Statevector,
                           "DensityMatrix": DensityMatrix, "Operator": Operator, "partial_trace": partial_trace,
                           "asdict": asdict, "dataclass": dataclass})
    helper.__dict__.update({gate.__name__: gate for gate in (
        CXGate, CZGate, HGate, IGate, PhaseGate, RXGate, RYGate, RZGate, SGate,
        SdgGate, TGate, TdgGate, UGate, XGate, YGate, ZGate)})
    sys.modules[module_name] = helper
    try:
        exec(compile(helper_tree, "<authenticated numerical helper subset>", "exec"), helper.__dict__)
        # Lindblad32 discarded resets cause exponentially many nonzero pure
        # branches. Preserve the original <=5-wire density path for instruments;
        # max_qubits=8 would dispatch simulate_trial back to the ensemble path.
        limits = helper.InstrumentLimits(max_qubits=5 if kind == "instrument" else 8, source_operations=20_000,
                                        synthesized_operations=500_000,
                                        validation_operations=500_000, validation_branches=256)
        results = []
        for trial in trials:
            checks = []
            source_mean = None
            for parsed, saved in ((trial.source, trial.source_record), (trial.compiled, trial.compiled_record)):
                circuit = _circuit(parsed, QuantumCircuit)
                if kind == "instrument":
                    actual = helper.simulate_trial(circuit, trial.outcome[0], trial.accept, limits=limits)
                else:
                    actual = helper.simulate_scored_trial(circuit, trial.outcome, trial.scores,
                                                          trial.accept, limits=limits)
                checks.append({"saved_record_field_residuals": _field_residuals(actual, saved, np, joint=kind == "complete"),
                               "acceptance_probability": actual["acceptance_probability"],
                               "mean": actual["mean"],
                               "primitive_operations": len(parsed.operations),
                               "numerical_method": "density matrix" if kind == "instrument" else "unnormalized pure-state Kraus ensemble",
                               "peak_branches": actual.get("simulation", {}).get("peak_branches")})
                if source_mean is None:
                    source_mean = actual["mean"]
                del actual, circuit
            exact_mean, saved_bias, saved_exact_mean = _independent_reference(trial, kind, np, expm)
            if not math.isfinite(exact_mean) or not math.isfinite(saved_bias) or (saved_exact_mean is not None and not math.isfinite(saved_exact_mean)):
                raise ReplayError("Nonfinite independent-reference evidence")
            observed_bias = abs(source_mean-exact_mean)
            results.append({"case": trial.name, "source": checks[0], "compiled": checks[1],
                            "independent_reference_mean": exact_mean,
                            "saved_exact_reference_mean": saved_exact_mean,
                            "independent_saved_exact_reference_signed_difference": None if saved_exact_mean is None else exact_mean-saved_exact_mean,
                            "independent_saved_exact_reference_absolute_difference": None if saved_exact_mean is None else abs(exact_mean-saved_exact_mean),
                            "source_reference_absolute_score_difference": observed_bias,
                            "source_reference_signed_score_difference": source_mean-exact_mean,
                            "compiled_reference_absolute_score_difference": abs(checks[1]["mean"]-exact_mean),
                            "frozen_source_score_bias": saved_bias,
                            "source_reference_gap_minus_frozen_score_bias": observed_bias-saved_bias})
        evidence = {"status": "evidence_collected", "kind": kind, "cases": results,
                "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "qiskit")}},
                "authenticated_helper_sha256": hashlib.sha256(records_bytes["instrument_resources.py"]).hexdigest(),
                "authenticated_lock_sha256": hashlib.sha256(_bytes(records_bytes, "requirements-lock.txt")).hexdigest(),
                "numerical_compatibility_policy": "unapproved; collect local and Linux field residuals before freezing an observed policy",
                "scope": "saved initialized fixed measured trials; source and aggregate Clifford+T QASM with explicit reset/readout schedule",
                "excluded": ["arbitrary-input channel theorem", "per-rotation approximant revalidation",
                             "original routing proof", "Native Trace reconstruction", "renewed search or optimality",
                             "fresh fitting, synthesis, estimation or budget/retry repricing"]}
        # Return a strictly finite JSON round-trip, not NumPy objects or NaN/Inf.
        return json.loads(json.dumps(evidence, allow_nan=False))
    finally:
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
    raise SystemExit("Use the authenticated parent verifier's optional replay mode; standalone numerical replay is disabled.")
