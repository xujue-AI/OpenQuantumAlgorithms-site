# SPDX-License-Identifier: MIT
"""Separate, attested n4 port accounting with the frozen instrument ISA.

The completed baseline is read from its hash-verified saved bundle.  Only a
new port candidate can be density checked, under an explicit execution gate.
Native search needs an inherited process-issued proof and an enclosing runner
watchdog: checks between yields cannot interrupt blocked native construction.
No existing materialization limits, budgets or hardware scenarios are changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib
import inspect
import json
import math
import os
import sys
import stat
from pathlib import Path
import time
from types import CodeType
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

if __package__:
    from .. import instrument_resources as ir
    from . import lindblad_four_compile as cc
    from . import lindblad_four_density as di
    from . import lindblad_four_physical as fp
else:
    import instrument_resources as ir
    import lindblad_four_compile as cc
    import lindblad_four_density as di
    import lindblad_four_physical as fp

CASE_ID = "lindblad-n4-r211-Z0"
COMPLETE_TRIALS = 15_756
CANDIDATE_OPERATION_CEILING = 1_926_042
ADDED_CX_CEILING = 248_136
FROZEN_IR_SHA256 = "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"
FROZEN_ESTIMATE_SHA256 = "969f94428f84a11ae280bf6290fac2a188dd91f5b8e9b62fd1e5128e6354f73c"
PHYSICAL_FAILURE = Fraction(1, 1200)
FACTORY_FAILURE = Fraction(1, 1200)
STATISTICAL_FAILURE = Fraction(1, 600)
_ISSUED: dict[int, tuple[Any, str]] = {}
_WATCHDOGS: dict[int, Any] = {}


@dataclass(frozen=True)
class PortResourceLimits:
    """Mandatory explicit domains, independent of InstrumentLimits."""
    max_qubits: int
    source_operations: int
    reference_operations: int
    candidate_operations: int

    def __post_init__(self):
        for name, ceiling in {"max_qubits": 8, "source_operations": 20_000,
                              "reference_operations": 100_000,
                              "candidate_operations": 2_000_000}.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in 1..{ceiling}")


@dataclass(frozen=True)
class NativeSearchLimits:
    factory_queries: int
    surface_code_queries: int
    candidates_per_query: int
    candidates_per_profile: int
    traces_per_candidate: int
    wall_seconds: int
    rss_bytes: int

    def __post_init__(self):
        ceilings = {"factory_queries": 24, "surface_code_queries": 24,
                    "candidates_per_query": 1000, "candidates_per_profile": 24_000,
                    "traces_per_candidate": 1, "wall_seconds": 180,
                    "rss_bytes": 2*1024**3}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in 1..{ceiling}")


def literal_limits() -> PortResourceLimits:
    return PortResourceLimits(8, 20_000, 100_000, 2_000_000)


def native_limits() -> NativeSearchLimits:
    return NativeSearchLimits(24, 24, 1000, 24_000, 1, 180, 2*1024**3)


def literal_native_limits() -> NativeSearchLimits:
    return native_limits()


def _limits(value) -> PortResourceLimits:
    if type(value) is not PortResourceLimits:
        raise ValueError("Explicit PortResourceLimits are required")
    return PortResourceLimits(**asdict(value))


def _native_limits(value) -> NativeSearchLimits:
    if type(value) is not NativeSearchLimits:
        raise ValueError("Explicit NativeSearchLimits are required")
    return NativeSearchLimits(**asdict(value))


def _json_bytes(value) -> bytes:
    raw = (json.dumps(fp._jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)+"\n").encode()
    if len(raw) > 32*1024**2:
        raise ValueError("Managed scientific JSON exceeds its 32 MiB cap")
    return raw


def _sha(value) -> str:
    return hashlib.sha256(value).hexdigest()


def _port_module(name):
    return importlib.import_module("."+name, __package__) if __package__ else importlib.import_module(name)


def _runner_path():
    parent = Path(__file__).resolve().parents[3]/"scripts" if __package__ else Path(__file__).resolve().parent
    return parent/"run_simulation_lindblad_port.py"


def _bounded_source_bytes(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Source must be a regular non-symlink file")
    before = path.stat()
    if before.st_size > 32*1024**2:
        raise ValueError("Source exceeds the 32 MiB managed byte cap")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("Source descriptor identity changed")
        raw = handle.read(32*1024**2+1)
        after_fd = os.fstat(handle.fileno())
    after = path.stat()
    if (len(raw) > 32*1024**2 or path.is_symlink() or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or
            (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) !=
            (after_fd.st_dev, after_fd.st_ino, after_fd.st_size, after_fd.st_mtime_ns, after_fd.st_ctime_ns) or
            len(raw) != before.st_size):
        raise ValueError("Source changed during bounded descriptor read")
    return raw


def _require_loaded_source(module):
    """Compare loaded Python function code with freshly compiled source bytes.

    This executes no replacement module. Disk/source inspection alone would
    allow an older imported code object to survive a same-process file edit.
    """
    raw = _bounded_source_bytes(module.__file__)
    compiled = compile(raw, module.__file__, "exec", dont_inherit=True, optimize=0)
    definitions = {code.co_name: code for code in compiled.co_consts if isinstance(code, CodeType)}
    for name, code in definitions.items():
        current = vars(module).get(name)
        if isinstance(current, type):
            methods = {c.co_name: c for c in code.co_consts if isinstance(c, CodeType)}
            for method_name, expected in methods.items():
                actual = vars(current).get(method_name)
                if isinstance(actual, (staticmethod, classmethod)):
                    actual = actual.__func__
                elif isinstance(actual, property):
                    actual = actual.fget
                if not inspect.isfunction(actual) or actual.__code__ != expected:
                    raise ValueError(f"Loaded method differs from current source: {module.__name__}.{name}.{method_name}")
        else:
            if not inspect.isfunction(current) or current.__code__ != code:
                raise ValueError(f"Loaded function differs from current source: {module.__name__}.{name}")
    if raw != _bounded_source_bytes(module.__file__):
        raise ValueError("Source changed during imported-code verification")
    return _sha(raw)


def derivation_record() -> dict:
    ir_sha = _require_loaded_source(ir)
    function_sha = _sha(inspect.getsource(ir.estimate_instrument).encode())
    if ir_sha != FROZEN_IR_SHA256 or function_sha != FROZEN_ESTIMATE_SHA256:
        raise ValueError("The frozen instrument source or estimate function changed")
    rw, pw = _port_module("lindblad_port_rewrite"), _port_module("lindblad_port_words")
    return {"frozen_ir_source_sha256": ir_sha,
            "frozen_ir_estimate_function_sha256": function_sha,
            "source_sha256": {Path(module.__file__).name: _require_loaded_source(module)
                              for module in (rw, pw, cc, di, fp)},
            "adapter_source_sha256": _require_loaded_source(sys.modules[__name__]),
            "shared_limits_or_formulas_changed": False,
            "legacy_report_limits_forged": False,
            "native_skeleton": "direct Trace, frozen _event_counts/_costs and serial geometry",
            "native_internal_work_scope": "top-level query/candidate counters do not bound internal factory work; an enclosing process timeout/RSS watchdog is required"}


def _fixed_profile(profile) -> ir.InstrumentProfile:
    fields = {"physical_gate_time_ns": 100, "physical_measurement_time_ns": 500,
              "physical_initialization_time_ns": 100, "code_cycle_time_ns": 1000,
              "reaction_time_ns": 10_000, "controller_failure_per_decision": 1e-15,
              "memory_prefactor": .1, "initialization_residual_prefactor": .1,
              "memory_threshold": .01, "factory_pipeline_failure_bound": .01,
              "factory_feedback_rounds_reserved": 3, "maximum_code_distance": 49}
    if type(profile) is not ir.InstrumentProfile or not any(
            asdict(profile) == {"physical_error_rate": rate, **fields}
            for rate in (1e-3, 1e-4)):
        raise ValueError("Use exactly one of the two unchanged literal n4 profiles")
    return profile


def _candidate_domain(circuit, limits):
    limits = _limits(limits)
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != 8 or circuit.num_clbits != 8:
        raise ValueError("Candidate requires eight quantum and classical wires")
    if limits.max_qubits != 8 or len(circuit.data) > min(limits.candidate_operations, CANDIDATE_OPERATION_CEILING):
        raise ValueError("Candidate exceeds the explicit separate operation domain")
    density_limits = di.EightWireDensityLimits(8, limits.source_operations, limits.candidate_operations)
    di._source_check(circuit, density_limits, limits.candidate_operations)
    cc._flat_basis_check(circuit, rotations=False)
    cc._complete_readout(circuit)
    cc._neighbor_check(circuit)
    # The new rewrite guard also checks executable bound matrix methods;
    # a mutable canonical gate can otherwise hide a per-instance override.
    rw = _port_module("lindblad_port_rewrite")
    for _ in rw._circuit_tokens(circuit):
        pass
    if not math.isfinite(float(circuit.global_phase)):
        raise ValueError("Candidate coherent phase must be finite")
    return density_limits


def _circuit_sha(circuit, limits) -> str:
    _candidate_domain(circuit, limits)
    digest = hashlib.sha256()
    layout = circuit.layout
    digest.update(_json_bytes({"name": circuit.name, "qubits": circuit.num_qubits,
                              "clbits": circuit.num_clbits,
                              "qregs": [(r.name, r.size) for r in circuit.qregs],
                              "cregs": [(r.name, r.size) for r in circuit.cregs],
                              "global_phase_hex": float(circuit.global_phase).hex(),
                              "layout": None if layout is None else {
                                  "initial": layout.initial_index_layout(), "final": layout.final_index_layout()}}))
    cache = {}
    for item in circuit.data:
        operation = item.operation
        token = (type(operation).__module__, type(operation).__qualname__,
                 operation.base_class.__module__, operation.base_class.__qualname__,
                 operation.name, operation.num_qubits, operation.num_clbits,
                 getattr(operation, "ctrl_state", None), getattr(operation, "num_ctrl_qubits", None),
                 tuple(float(p).hex() for p in operation.params))
        raw = cache.get(token)
        if raw is None:
            raw = _json_bytes(token)
            cache[token] = raw
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
        digest.update(bytes([len(item.qubits), *(circuit.find_bit(q).index for q in item.qubits),
                             len(item.clbits), *(circuit.find_bit(c).index for c in item.clbits)]))
    return digest.hexdigest()


def _array(value):
    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        return np.asarray(value["real"], float)+1j*np.asarray(value["imag"], float)
    return np.asarray(value)


def _check_density(check):
    rho, probabilities = _array(check["reduced_before_output"]), _array(check["joint_probabilities"])
    trace = check["unnormalized_trace_before_output"]
    if (type(trace) is not float or not math.isfinite(trace) or abs(trace-1) > 1e-8 or
            check["acceptance_probability"] != trace or check["normalization_divisor"] != trace or
            check["normalization_is_postselection"] is not False):
        raise ValueError("Raw trace and one output normalization must retain the frozen no-postselection guard")
    if rho.shape != (16, 16) or probabilities.shape != (16,) or not np.all(np.isfinite(rho)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("A finite complete 16x16 density and 16-outcome law are required")
    if (abs(complex(np.trace(rho))-1) > 1e-10 or abs(float(sum(probabilities))-1) > 1e-10 or
            max(abs(np.diag(rho).real-probabilities)) > 1e-10 or
            np.linalg.norm(rho-rho.conj().T, ord="nuc") > 1e-8 or
            min(np.linalg.eigvalsh((rho+rho.conj().T)/2)) < -1e-8 or min(probabilities) < -1e-8):
        raise ValueError("Candidate normalized density fails trace, correlation, positivity or Hermiticity guards")
    if check["outcome_clbits"] != [0, 1, 2, 3] or check["score_values"] != list(fp.score_values()):
        raise ValueError("The candidate must retain the complete joint Z0 output protocol")
    if abs(float(check["mean"])-float(np.dot(fp.score_values(), probabilities))) > 1e-10:
        raise ValueError("Score does not agree with the complete joint law")
    return rho, probabilities, trace


def _saved_baseline(recovered, word_limits):
    pw = _port_module("lindblad_port_words")
    baseline = pw.baseline_record(recovered, limits=word_limits)
    contract, workload = baseline["error_contract"], baseline["complete_workload"]
    if (baseline["case_id"] != CASE_ID or workload["complete_trials"] != COMPLETE_TRIALS or
            workload["complete_attempts"] != COMPLETE_TRIALS or workload["algorithm_retry_cap"] != 1 or
            workload["reserved_logical_patches"] != 31 or workload["workspace_startup_initializations_per_trial"] != 23 or
            workload["physical_failure_budget"] != fp._downward(PHYSICAL_FAILURE) or
            workload["factory_availability_budget"] != fp._downward(FACTORY_FAILURE) or
            contract["requested_synthesis_allowance"] != 5e-5 or
            contract["representation_norm_reserve_scenario"] != 1e-7 or
            contract["SDK_control_routing_norm_reserve_scenario"] != 1e-7):
        raise ValueError("Saved baseline task, repetitions, geometry or error premises changed")
    return baseline


def _baseline_domain(baseline, limits):
    limits = _limits(limits)
    if (limits.max_qubits != 8 or sum(baseline["source_counts"].values()) > limits.source_operations or
            sum(baseline["actual_synthesis"]["routed_counts"].values()) > limits.reference_operations):
        raise ValueError("Saved source/routed reference exceeds the requested separate resource limits")


@dataclass(frozen=True)
class VerifiedFourCandidate:
    candidate: Any
    recipe: Any
    recovered: Any
    limits: PortResourceLimits
    word_limits: Any
    rewrite_limits: Any
    canonical_verification_json: bytes
    verification_fingerprint: str

    def as_record(self) -> dict:
        return json.loads(self.canonical_verification_json)


def _current_fingerprint(value) -> str:
    if type(value) is not VerifiedFourCandidate:
        raise ValueError("A process-issued VerifiedFourCandidate is required")
    baseline = _saved_baseline(value.recovered, value.word_limits)
    _baseline_domain(baseline, value.limits)
    record = value.as_record()
    if type(value.canonical_verification_json) is not bytes or _json_bytes(record) != value.canonical_verification_json:
        raise ValueError("Verification evidence must retain its exact canonical JSON bytes")
    rw = _port_module("lindblad_port_rewrite")
    if type(value.recipe) is not rw.FrozenPortRecipe or record["recipe_sha256"] != value.recipe.as_record()["recipe_sha256"]:
        raise ValueError("Verified recipe identity changed")
    payload = {"verification_record": record,
               "circuit_sha256": _circuit_sha(value.candidate.circuit, value.limits),
               "candidate_report": value.candidate.report,
               "recipe": value.recipe.as_record(), "recovery": value.recovered.as_record(),
               "baseline_sha256": _sha(_json_bytes(baseline)), "derivation": derivation_record(),
               "limits": asdict(value.limits), "word_limits": asdict(value.word_limits),
               "rewrite_limits": asdict(value.rewrite_limits)}
    return _sha(_json_bytes(payload))


def _require_verified(value):
    if type(value) is not VerifiedFourCandidate:
        raise ValueError("A process-issued VerifiedFourCandidate is required")
    issued = _ISSUED.get(id(value))
    if issued is None or issued[0] is not value or issued[1] != value.verification_fingerprint or value.verification_fingerprint != _current_fingerprint(value):
        raise ValueError("Verified candidate is unissued, stale or mutated")


def verify_candidate(candidate, recipe, recovered, *, limits: PortResourceLimits,
                     word_limits, rewrite_limits, allow_full_density: bool = False) -> VerifiedFourCandidate:
    """Check only the new candidate once; compare saved baseline arrays.

    Process-issued proofs can be inherited by the runner's forked native
    children.  Pickling/restoring a proof does not authorize a new process.
    """
    if type(allow_full_density) is not bool or not allow_full_density:
        raise ValueError("Full candidate density execution requires explicit allow_full_density=True")
    rw = _port_module("lindblad_port_rewrite")
    limits = _limits(limits)
    baseline = _saved_baseline(recovered, word_limits)
    _baseline_domain(baseline, limits)
    report = rw.require_verified_candidate(candidate, recipe, recovered, limits=rewrite_limits, word_limits=word_limits)
    density_limits = _candidate_domain(candidate.circuit, limits)
    if (report["actual_operations"] > CANDIDATE_OPERATION_CEILING or report["added_cx"] > ADDED_CX_CEILING or
            report["global_phase_binary64_hex"] != recovered.baseline_global_phase.hex() or
            float(candidate.circuit.global_phase).hex() != recovered.baseline_global_phase.hex() or
            report["final_measurement_wires"] != [6, 1, 2, 3, 7, 4, 0, 5]):
        raise ValueError("Candidate operation, phase or physical readout contract changed")
    counts = dict(candidate.circuit.count_ops())
    if counts.get("t", 0)+counts.get("tdg", 0) != 642_284 or counts.get("reset") != 1481 or counts.get("measure") != 8:
        raise ValueError("Actual candidate must preserve all frozen T/reset/readout occurrences")
    before = _circuit_sha(candidate.circuit, limits)
    check = di.simulate_density_trial(candidate.circuit, (0, 1, 2, 3), fp.score_values(), {}, limits=density_limits)
    fp._annotate_check(check)
    saved_check = fp._jsonable(check)
    rho, probabilities, trace = _check_density(saved_check)
    ct_rho, ct_probabilities, ct_trace = _check_density(baseline["clifford_t_instrument_check"])
    source_rho, source_probabilities, source_trace = _check_density(baseline["source_instrument_check"])
    comparisons = {"candidate_vs_saved_ct_normalized_trace_norm": float(np.linalg.norm(rho-ct_rho, ord="nuc")),
                   "candidate_vs_saved_ct_raw_trace_norm": float(np.linalg.norm(rho*trace-ct_rho*ct_trace, ord="nuc")),
                   "candidate_vs_saved_ct_joint_maximum_absolute": float(max(abs(probabilities-ct_probabilities))),
                   "candidate_vs_saved_ct_raw_trace_absolute": abs(trace-ct_trace),
                   "candidate_vs_saved_source_normalized_trace_norm": float(np.linalg.norm(rho-source_rho, ord="nuc")),
                   "candidate_vs_saved_source_raw_trace_norm": float(np.linalg.norm(rho*trace-source_rho*source_trace, ord="nuc")),
                   "candidate_vs_saved_source_joint_maximum_absolute": float(max(abs(probabilities-source_probabilities))),
                   "candidate_vs_saved_source_Z0_absolute": abs(float(saved_check["mean"])-float(baseline["source_instrument_check"]["mean"]))}
    gksl = _array(baseline["independent_reference"]["exact_GKSL_system_density"])
    comparisons["candidate_vs_saved_GKSL_normalized_trace_norm"] = float(np.linalg.norm(rho-gksl, ord="nuc"))
    comparisons["candidate_vs_saved_GKSL_raw_trace_norm"] = float(np.linalg.norm(rho*trace-gksl, ord="nuc"))
    comparisons["candidate_vs_saved_GKSL_joint_maximum_absolute"] = float(max(abs(probabilities-np.diag(gksl).real)))
    comparisons["candidate_vs_saved_GKSL_Z0_absolute"] = abs(float(saved_check["mean"])-baseline["independent_reference"]["exact_GKSL_Z0"])
    if (comparisons["candidate_vs_saved_ct_normalized_trace_norm"] > 1e-10 or
            comparisons["candidate_vs_saved_ct_joint_maximum_absolute"] > 1e-10 or
            comparisons["candidate_vs_saved_ct_raw_trace_norm"] > 1e-10 or
            comparisons["candidate_vs_saved_ct_raw_trace_absolute"] > 1e-8 or
            comparisons["candidate_vs_saved_source_normalized_trace_norm"] > baseline["error_contract"]["realization_Z0_bias_upper_proposal"]+1e-10 or
            comparisons["candidate_vs_saved_GKSL_normalized_trace_norm"] > baseline["error_contract"]["splitting_bias_upper"]+baseline["error_contract"]["realization_Z0_bias_upper_proposal"]+1e-10):
        raise ValueError("New candidate fails saved complete raw/normalized baseline agreement")
    if before != _circuit_sha(candidate.circuit, limits) or report != candidate.report:
        raise ValueError("Candidate changed during full density verification")
    workload = dict(baseline["complete_workload"])
    workload["primitive_counts_per_complete_trial"] = {n: c for n, c in counts.items() if n != "barrier"}
    workload["primitive_counts_all_trials"] = {n: c*COMPLETE_TRIALS for n, c in counts.items() if n != "barrier"}
    record = {"schema_version": 1, "status": "verified_complete_port_candidate",
              "case_id": CASE_ID, "candidate_circuit_sha256": before,
              "candidate_report": report, "recipe_sha256": recipe.as_record()["recipe_sha256"],
              "catalog_sha256": recovered.recovery_sha256, "source_artifacts": recovered.source_artifacts,
              "candidate_instrument_check": saved_check, "saved_baseline_comparisons": comparisons,
              "saved_source_instrument_check": baseline["source_instrument_check"],
              "saved_clifford_t_instrument_check": baseline["clifford_t_instrument_check"],
              "saved_independent_reference": baseline["independent_reference"],
              "error_contract": baseline["error_contract"], "complete_workload": workload,
              "saved_source_counts": baseline["source_counts"],
              "saved_baseline_complete_workload": baseline["complete_workload"],
              "derivation": derivation_record(), "materialization_limits": asdict(limits),
              "candidate_full_density_executed": True, "candidate_density_execution_count": 1,
              "historical_density_execution_count": 0, "new_synthesis_calls": 0,
              "native_physical_queries": 0, "physical_feasibility_claimed": False,
              "nominal_normalized_equivalence_guard": 1e-10, "nominal_raw_state_trace_guard": 1e-8}
    value = VerifiedFourCandidate(candidate, recipe, recovered, limits, word_limits, rewrite_limits, _json_bytes(record), "")
    fingerprint = _current_fingerprint(value)
    object.__setattr__(value, "verification_fingerprint", fingerprint)
    _ISSUED[id(value)] = (value, fingerprint)
    return value


def _controller(circuit, profile, attempts, physical_budget, availability_budget):
    counts = ir._event_counts(circuit)
    counts["trial_control"] += 1
    counts["initial_reset"] += 23
    t_count = sum(c for n, c in counts.items() if n.startswith(("t_q", "tdg_q")))
    cap = ir._factory_retry_cap(t_count*attempts, profile.factory_pipeline_failure_bound, availability_budget)
    decisions = sum(c*ir._costs(profile, 3, n, 0, cap)["decisions"] for n, c in counts.items())
    contribution = decisions*attempts*profile.controller_failure_per_decision
    return {"event_counts_per_trial": dict(counts), "required_magic_states_per_trial": t_count,
            "required_magic_states_all_trials": t_count*attempts, "factory_retry_cap": cap,
            "controller_decisions_per_trial": decisions, "controller_decisions_all_trials": decisions*attempts,
            "controller_failure_per_decision_scenario": profile.controller_failure_per_decision,
            "distance_independent_controller_model_contribution": contribution,
            "physical_failure_budget": physical_budget, "factory_availability_budget": availability_budget,
            "maximum_allowed_controller_decisions_per_trial": math.floor(Fraction.from_float(physical_budget)/(attempts*Fraction.from_float(profile.controller_failure_per_decision))) if profile.controller_failure_per_decision else None,
            "factory_readiness_failure_union_bound": float(Fraction(str(profile.factory_pipeline_failure_bound))**cap*t_count*attempts),
            "budget_exceeded": contribution > physical_budget,
            "eligibility_is_physical_feasibility": False}


def controller_eligibility(verified: VerifiedFourCandidate, profile: ir.InstrumentProfile) -> dict:
    _require_verified(verified)
    profile = _fixed_profile(profile)
    diagnostic = _controller(verified.candidate.circuit, profile, COMPLETE_TRIALS,
                             fp._downward(PHYSICAL_FAILURE), fp._downward(FACTORY_FAILURE))
    diagnostic["verification_fingerprint"] = verified.verification_fingerprint
    diagnostic["status"] = "controller_ineligible" if diagnostic["budget_exceeded"] else "controller_eligible_for_bounded_native_search"
    return diagnostic


class NativeLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeWatchdogCapability:
    parent_pid: int
    activation_event: Any
    runner_source_sha256: str
    limits: NativeSearchLimits
    deadline_monotonic: float


def _issue_native_watchdog(*, activation_event, limits: NativeSearchLimits,
                           runner_source_sha256: str) -> NativeWatchdogCapability:
    """Runner-private fork handshake; issuance alone does not activate it.

    The reviewed run_native_profile parent installs its process monitor and
    successfully checks the child's RSS before setting the inherited event.
    """
    limits = _native_limits(limits)
    frame = inspect.currentframe().f_back
    expected = _runner_path()
    actual = Path(frame.f_globals.get("__file__", "")).resolve()
    if actual != expected or frame.f_code.co_name != "run_native_profile":
        raise ValueError("Only the reviewed runner native watchdog wrapper can issue a capability")
    raw = _bounded_source_bytes(expected)
    compiled = compile(raw, str(expected), "exec", dont_inherit=True, optimize=0)
    code = next((c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == "run_native_profile"), None)
    if code != frame.f_code or _sha(raw) != runner_source_sha256:
        raise ValueError("Loaded runner watchdog wrapper differs from its reviewed source")
    if activation_event.is_set():
        raise ValueError("Watchdog activation must follow child monitor installation")
    capability = NativeWatchdogCapability(os.getpid(), activation_event, runner_source_sha256,
                                         limits, time.monotonic()+limits.wall_seconds)
    _WATCHDOGS[id(capability)] = capability
    return capability


def _require_native_watchdog(capability, limits):
    if type(capability) is not NativeWatchdogCapability or _WATCHDOGS.get(id(capability)) is not capability:
        raise ValueError("A runner-issued inherited native watchdog capability is required")
    runner = _runner_path()
    if (os.getpid() == capability.parent_pid or os.getppid() != capability.parent_pid or
            not capability.activation_event.is_set() or capability.limits != limits or
            time.monotonic() >= capability.deadline_monotonic or
            _sha(_bounded_source_bytes(runner)) != capability.runner_source_sha256):
        raise ValueError("Native watchdog capability is inactive, stale or outside its monitored fork")
    import psutil
    if not psutil.Process(capability.parent_pid).is_running():
        raise ValueError("Native watchdog parent is no longer running")


def _rss_bytes() -> int:
    # Current RSS, not ru_maxrss, whose units and peak semantics differ by OS.
    import psutil
    return int(psutil.Process().memory_info().rss)


class _NativeBudget:
    def __init__(self, limits, deadline=None):
        self.limits = _native_limits(limits)
        self.started = time.monotonic()
        self.deadline = min(self.started+self.limits.wall_seconds, deadline) if deadline is not None else self.started+self.limits.wall_seconds
        self.factory_queries = self.surface_code_queries = self.candidates = self.trace_calls = 0
        self.candidate_overflow_sentinel_observed = False

    def check(self):
        if time.monotonic() > self.deadline:
            raise NativeLimitExceeded("native_profile_wall_seconds")
        if _rss_bytes() > self.limits.rss_bytes:
            raise NativeLimitExceeded("native_profile_rss_bytes")

    def dispatch(self, name):
        self.check()
        if getattr(self, name) >= getattr(self.limits, name):
            raise NativeLimitExceeded(name)
        setattr(self, name, getattr(self, name)+1)

    def observe_candidate(self, per_query):
        self.check()
        if per_query > self.limits.candidates_per_query or self.candidates >= self.limits.candidates_per_profile:
            self.candidate_overflow_sentinel_observed = True
            raise NativeLimitExceeded("factory_candidate_count")
        self.candidates += 1

    def as_record(self):
        return {"factory_queries": self.factory_queries, "surface_code_queries": self.surface_code_queries,
                "factory_candidates_observed_within_cap": self.candidates, "trace_estimates": self.trace_calls,
                "candidate_overflow_sentinel_observed": self.candidate_overflow_sentinel_observed,
                "elapsed_seconds": time.monotonic()-self.started,
                "limits": asdict(self.limits), "whole_profile_discarded_on_overflow": True,
                "external_process_watchdog_required": True,
                "query_scope": "actual dispatched top-level queries only; native internal factory construction is governed by the runner watchdog"}


def estimate_verified_four(verified: VerifiedFourCandidate, *, profile: ir.InstrumentProfile,
                           cache_dir: Path, limits: PortResourceLimits,
                           native_limits: NativeSearchLimits,
                           include_all_feasible_points: bool = False,
                           allow_native_search: bool = False,
                           watchdog_capability: NativeWatchdogCapability | None = None) -> dict:
    limits = _limits(limits)
    caps = _native_limits(native_limits)
    if type(include_all_feasible_points) is not bool:
        raise ValueError("include_all_feasible_points must be Boolean")
    profile = _fixed_profile(profile)
    diagnostic = controller_eligibility(verified, profile)
    if limits != verified.limits:
        raise ValueError("Resource limits differ from verified candidate limits")
    if diagnostic["budget_exceeded"]:
        return {"status": "infeasible_controller_model", "pareto_points": [],
                "all_feasible_points": [], "feasible_candidate_count": 0,
                "controller_diagnostic": diagnostic, "profile": asdict(profile),
                "native_execution": {"factory_queries": 0, "surface_code_queries": 0, "trace_estimates": 0}}
    if type(allow_native_search) is not bool or not allow_native_search:
        raise ValueError("Native search requires explicit execution authorization")
    _require_native_watchdog(watchdog_capability, caps)
    return _estimate_accounting(verified.candidate.circuit, COMPLETE_TRIALS,
                                profile=profile, physical_failure_budget=fp._downward(PHYSICAL_FAILURE),
                                factory_availability_budget=fp._downward(FACTORY_FAILURE),
                                cache_dir=cache_dir, native_limits=caps,
                                source_event_counts=verified.as_record()["saved_source_counts"],
                                actual_synthesis=verified.as_record()["candidate_report"],
                                include_all_feasible_points=include_all_feasible_points,
                                verification_fingerprint=verified.verification_fingerprint,
                                watchdog_deadline=watchdog_capability.deadline_monotonic)


def _estimate_accounting(circuit: QuantumCircuit, total_attempts: int, *, profile: ir.InstrumentProfile,
                         physical_failure_budget: float, factory_availability_budget: float,
                         cache_dir: Path, native_limits: NativeSearchLimits,
                         source_event_counts: dict, actual_synthesis: dict,
                         include_all_feasible_points: bool = False,
                         verification_fingerprint: str | None = None,
                         watchdog_deadline: float | None = None) -> dict:
    """Internal frozen-equation core, separately exercised with tiny mocks.

    Production callers enter only through estimate_verified_four. This helper
    changes entry validation and bounded enumeration, never the ISA equations.
    """
    if importlib.metadata.version("qdk") != "1.34.0":
        raise RuntimeError("This native accounting interface was audited with QDK 1.34.0")
    budget = _NativeBudget(native_limits, deadline=watchdog_deadline)
    from qdk.qre import ErrorComposition, LOGICAL, Trace, constant_function, linear_function
    from qdk.qre.instruction_ids import T
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode
    points = []
    try:
        counts = ir._event_counts(circuit)
        counts["trial_control"] += 1
        n = circuit.num_qubits
        # Two interleaved rows for data/bridges and transport/bridges + handoff bridge.
        patches = 4*n-1
        counts["initial_reset"] += patches-n  # Explicit startup of reserved workspace.
        t_count = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
        total_t = t_count*total_attempts
        factory_retry_cap = ir._factory_retry_cap(total_t, profile.factory_pipeline_failure_bound,
                                              factory_availability_budget)
        availability_failure = float(total_t*Fraction(str(profile.factory_pipeline_failure_bound))**factory_retry_cap)
        points = []
        for distance in range(3, profile.maximum_code_distance+1, 2):
            budget.check()
            ctx = GateBased(error_rate=profile.physical_error_rate, gate_time=profile.physical_gate_time_ns,
                            measurement_time=profile.physical_measurement_time_ns).context()
            budget.check()
            if t_count:
                budget.dispatch("surface_code_queries")
                code_query = SurfaceCode.q(distance=distance, code_cycle_override=profile.code_cycle_time_ns,
                                           error_correction_threshold=profile.memory_threshold)
                budget.check()
                budget.dispatch("factory_queries")
                query = RoundBasedFactory.q(code_query=code_query, cache_dir=Path(cache_dir))
                budget.check()
                supplies = query.enumerate(ctx)
                budget.check()
            else:
                supplies = [None]
            per_query = 0
            for supply in supplies:
                if t_count:
                    per_query += 1
                    budget.observe_candidate(per_query)
                else:
                    budget.check()
                factory = supply[T] if supply is not None else None
                factory_error = float(factory.expect_error_rate()) if factory is not None else 0.0
                if factory_error*total_t > physical_failure_budget*.75:
                    continue
                # This strict quality requirement also excludes a physical final stage.
                if factory is not None and factory_error >= 7.1*profile.physical_error_rate:
                    continue
                factory_ns = int(factory.expect_time()) if factory is not None else 0
                patch_size = 2*distance**2-1
                readout_error = ir.majority_path_error(distance, profile.physical_error_rate)
                preparation_error = profile.initialization_failure_scenario(distance)
                rows, nodes = [], []
                trace = Trace(patches)
                for index, (name, count) in enumerate(sorted(counts.items())):
                    cost = ir._costs(profile, distance, name, factory_ns, factory_retry_cap)
                    event_error = cost["readouts"]*readout_error + cost["preparations"]*preparation_error + cost["decisions"]*profile.controller_failure_per_decision
                    instruction_id = 2**40+index
                    nodes.append(ctx.add_instruction(instruction_id, encoding=LOGICAL, arity=None,
                                                     time=constant_function(cost["time_ns"]),
                                                     space=linear_function(patch_size), error_rate=constant_function(event_error)))
                    # Whole-device reservation enforces the declared serial schedule.
                    block = trace.add_block(repetitions=count)
                    block.add_operation(instruction_id, list(range(patches)))
                    rows.append(dict(operation=name, count=count, **cost, event_failure_model=event_error))
                if supply is not None:
                    nodes.append(supply.node_index(T))
                    trace.increment_resource_state(T, t_count)
                isa = ctx.make_isa(*nodes)
                runtime = sum(row["count"]*row["time_ns"] for row in rows)
                cycles = ir._ceil_div(runtime, profile.code_cycle_time_ns)
                memory_error = patches*cycles*profile.patch_failure_per_cycle(distance)
                # Extra output-port residence outside the native factory supply time:
                # reserved round overhead and the first transport SWAP into W0.
                round_overhead = profile.physical_initialization_time_ns + profile.physical_measurement_time_ns + profile.reaction_time_ns
                port_hold_ns = (profile.factory_feedback_rounds_reserved*round_overhead + profile.reaction_time_ns
                                + 3*ir._costs(profile, distance, "cx", factory_ns, factory_retry_cap)["time_ns"]) if t_count else 0
                port_hold_error = t_count*ir._ceil_div(port_hold_ns, profile.code_cycle_time_ns)*profile.patch_failure_per_cycle(distance)
                trace.increment_base_error(memory_error + port_hold_error)
                expected_error = memory_error + port_hold_error + sum(row["count"]*row["event_failure_model"] for row in rows) + t_count*factory_error
                if expected_error*total_attempts > physical_failure_budget:
                    continue
                budget.check()
                budget.trace_calls += 1
                result = trace.estimate(isa, max_error=physical_failure_budget/total_attempts,
                                        composition=ErrorComposition.UnionBound)
                budget.check()
                if result is None:
                    continue
                if int(result.runtime) != runtime or not math.isclose(float(result.error), expected_error, rel_tol=1e-10, abs_tol=1e-30):
                    raise RuntimeError("Native direct accounting differs from the explicit ledger")
                native_factory = result.factories.get(T)
                if t_count and (native_factory is None or native_factory.copies != 1):
                    raise RuntimeError("Serialized just-in-time model requires exactly one native factory")
                if not t_count and native_factory is not None:
                    raise RuntimeError("No-T model unexpectedly reserved a native factory")
                points.append({
                    "data_code_distance": distance, "reserved_logical_patches": patches,
                    "peak_physical_qubits": int(result.qubits), "data_workspace_physical_qubits": patches*patch_size,
                    "factory_physical_qubits": int(factory.expect_space()) if factory is not None else 0,
                    "runtime_per_complete_attempt_ns": runtime, "worst_case_runtime_all_attempts_ns": runtime*total_attempts,
                    "code_cycles_per_complete_attempt": cycles,
                    "full_physical_failure_model": float(result.error)*total_attempts,
                    "failure_components_all_attempts": {
                        "data_workspace_code_exposure": memory_error*total_attempts,
                        "factory_output_handoff_storage": port_hold_error*total_attempts,
                        "logical_readout_scenario": sum(r["count"]*r["readouts"] for r in rows)*readout_error*total_attempts,
                        "logical_initialization_scenario": sum(r["count"]*r["preparations"] for r in rows)*preparation_error*total_attempts,
                        "controller_scenario": sum(r["count"]*r["decisions"] for r in rows)*profile.controller_failure_per_decision*total_attempts,
                        "factory_output_quality_model": total_t*factory_error,
                    },
                    "factory_supply": {"output_error": factory_error, "pipeline_time_ns": factory_ns,
                                       "fixed_logical_distillation_distance": distance,
                                       "native_qec_crossing_prefactor": 0.03,
                                       "native_qec_threshold": profile.memory_threshold,
                                       "native_code_cycle_time_ns": profile.code_cycle_time_ns,
                                       "quality_calibration_scope": "QDK factory model; distinct from the Litinski per-cycle data-exposure fit",
                                       "full_pipeline_retry_cap_per_state": factory_retry_cap,
                                       "states_all_complete_attempts": total_t,
                                       "worst_case_pipeline_attempts": total_t*factory_retry_cap,
                                       "availability_failure_union_bound": availability_failure,
                                       "retry_cap_arithmetic": "exact Fraction search over decimal input strings; smallest cap in 1..16",
                                       "pipeline_feedback_rounds_reserved": profile.factory_feedback_rounds_reserved,
                                       "extra_init_readout_feedback_dispatch_ns_per_pipeline": profile.factory_feedback_rounds_reserved*round_overhead + profile.reaction_time_ns,
                                       "extra_output_port_hold_ns_per_state": port_hold_ns},
                    "macro_ledger_per_complete_attempt": rows,
                    "accounting_trace_sha256": hashlib.sha256(trace.to_json().encode()).hexdigest(),
                    "native_backend": "QDK direct Trace + explicit custom ISA; no import, PSSPC, LatticeSurgery or branch sampling",
                })
            budget.check()
    except NativeLimitExceeded as error:
        return {"schema_version": 1, "status": "native_profile_cap_exceeded",
                "reason": str(error), "pareto_points": [], "all_feasible_points": [],
                "feasible_candidate_count": 0, "profile": asdict(profile),
                "native_execution": budget.as_record(), "verification_fingerprint": verification_fingerprint}
    report = {"schema_version": 1,
            "status": "computed_conditional_port_surface_code_model" if points else "infeasible_declared_native_domain",
            "resource_domain": "literal-n4-port-explicit-opt-in-v1",
            "profile": asdict(profile), "complete_algorithm_attempts": total_attempts,
            "physical_failure_budget": physical_failure_budget,
            "factory_availability_budget": factory_availability_budget,
            "source_event_counts": source_event_counts,
            "synthesized_event_counts": dict(circuit.count_ops()),
            "workspace_initializations_per_attempt": patches-n,
            "actual_synthesis": actual_synthesis,
            "pareto_points": ir._pareto(points), "feasible_candidate_count": len(points),
            **({"all_feasible_points": points} if include_all_feasible_points else {}),
            "native_execution": budget.as_record(), "verification_fingerprint": verification_fingerprint,
            "comparison_scope": "same task, trials, budgets, hardware, controller, 31-patch line and frozen instrument ISA as saved n4",
            "assumptions": ["The repeated trial population and hardware schedule are unexecuted",
                            "Numerical candidate verification is not a formal interval or hardware certificate",
                            "Frozen instrument ISA/readout/initialization/factory assumptions and exclusions apply unchanged",
                            "All block-port added CX gates are counted once in the actual candidate stream",
                            "T at wire0 still pays factory, workspace shuttle, injection, handoff and port residence costs"],
            "excluded": ["offline classical preprocessing and final statistical reporting wall time",
                         "factory/data interface layout certification", "measured hardware and decoder throughput validation"],
            "reference_keys": ["resource_litinski_2019", "resource_horsman_2012", "resource_ross_selinger_2014",
                               "resource_qiskit_262", "resource_rsgridsynth_022", "resource_qdk_134"]}
    try:
        budget.check()
        _json_bytes(report)
        budget.check()
    except (ValueError, NativeLimitExceeded) as error:
        return {"schema_version": 1, "status": "native_profile_cap_exceeded",
                "reason": str(error), "pareto_points": [], "all_feasible_points": [],
                "feasible_candidate_count": 0, "profile": asdict(profile),
                "native_execution": budget.as_record(), "verification_fingerprint": verification_fingerprint}
    return report
