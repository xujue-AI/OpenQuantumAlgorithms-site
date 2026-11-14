"""Pure, frozen-model accounting for the source-only QAOA data shuttle.

No optimizer, sampler, synthesizer, factory search or native estimator is
imported here.  The candidate retains the seventh replay's 49 allocation
groups even when several candidate event ledgers become equal.  Its current
result is an exact source/macro ledger projection, not a new native physical
result.  Source ordering and channel equivalence are checked by the separate
transport and density validators before this accounting layer is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math


BACKEND_SHA256 = "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"
EXPECTED_CLASSES = 54
EXPECTED_PARENT_GROUPS = 49
EXPECTED_PREPARATIONS = 82_032
EXPECTED_MAGIC_STATES = 182_646_240
MAX_OPERATIONS_PER_CLASS = 10_000
MAX_WORKLOAD_OPERATIONS = 540_000
DATA_QUBITS = 4
RESERVED_PATCHES = 15
WORKSPACE_INITIALIZATIONS = 11
GLOBAL_FACTORY_RETRY_CAP = 6
CLIFFORD_GATES = frozenset({"h", "x", "y", "z", "s", "sdg"})
MAGIC_GATES = ("t", "tdg")
_PROFILE = {
    "physical_error_rate": 1e-4,
    "physical_gate_time_ns": 100,
    "physical_measurement_time_ns": 500,
    "physical_initialization_time_ns": 100,
    "code_cycle_time_ns": 1_000,
    "reaction_time_ns": 10_000,
    "controller_failure_per_decision": 1e-15,
    "memory_prefactor": 0.1,
    "initialization_residual_prefactor": 0.1,
    "memory_threshold": 0.01,
    "factory_pipeline_failure_bound": 0.01,
    "factory_feedback_rounds_reserved": 3,
    "maximum_code_distance": 49,
}
_MACHINE = {
    "baseline_selection": "seventh_minimum_runtime",
    "data_code_distance": 15,
    "reserved_logical_patches": RESERVED_PATCHES,
    "native_factory_isa_sha256": "2c94c77a1cb4779ac6b4d2f49d9312d70974b9ff82650b4f252ea52fe939c102",
    "native_t_instruction_sha256": "ad99b8ce36f8b556c05d877305e97e5e654f31c0d3756f0630c5bb2b44b184cb",
    "factory_physical_qubits": 236_623,
    "data_workspace_physical_qubits": 6_735,
    "peak_physical_qubits": 243_358,
    "factory_retry_cap": GLOBAL_FACTORY_RETRY_CAP,
    "baseline_serial_model_time_ns": 797_328_306_784_000,
}
_BUDGETS = {
    "sampling_family_failure": Fraction("0.04"),
    "fixed_script_realization_transfer": Fraction("0.005"),
    "physical_failure": Fraction("0.0025"),
    "factory_availability": Fraction("0.0025"),
}


def frozen_profile() -> dict:
    """Return a fresh copy of every frozen instrument assumption."""
    return deepcopy(_PROFILE)


def frozen_machine() -> dict:
    """Return the one saved seventh machine permitted for candidate comparison."""
    return deepcopy(_MACHINE)


def _check_profile(profile: dict | None) -> dict:
    profile = frozen_profile() if profile is None else profile
    if not isinstance(profile, dict) or set(profile) != set(_PROFILE):
        raise ValueError("Every frozen instrument profile field must be retained")
    if any(type(profile[key]) is not type(value) or profile[key] != value
           for key, value in _PROFILE.items()):
        raise ValueError("The matched-task candidate cannot change the frozen instrument profile")
    return deepcopy(profile)


def _integer(value, name: str, minimum=0, maximum=None) -> int:
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer within its declared bounds")
    return value


def frozen_macro_cost(profile: dict, distance: int, name: str,
                      factory_ns: int, retry_cap: int) -> dict:
    """Mirror only the frozen backend's deterministic ``_costs`` function.

    ``factory_ns`` is a supplied saved calibration, never searched or estimated
    here.  The caller must bind it to the fixed native ISA before physical use.
    Event errors and integer-ceiled exposure are deliberately separate from
    these time/readout/preparation/controller counts.
    """
    p = _check_profile(profile)
    _integer(distance, "code distance", 3, p["maximum_code_distance"])
    if distance % 2 == 0:
        raise ValueError("Code distance must be odd")
    _integer(factory_ns, "saved factory time", 0)
    _integer(retry_cap, "factory retry cap", 0, 16)
    if not isinstance(name, str):
        raise ValueError("Accounting operation name must be a string")
    dtime = distance * p["code_cycle_time_ns"]
    init = p["physical_initialization_time_ns"] + dtime + p["reaction_time_ns"]
    destructive = p["physical_measurement_time_ns"] + p["reaction_time_ns"]
    parity = 2 * init + dtime + destructive
    cnot = 3 * parity + 2 * p["reaction_time_ns"]
    if name == "initial_reset":
        return dict(time_ns=init, readouts=0, preparations=1, decisions=1)
    if name == "reset":
        return dict(time_ns=parity + 2*p["reaction_time_ns"] + dtime,
                    readouts=1, preparations=2, decisions=5)
    if name == "measure":
        return dict(time_ns=parity, readouts=1, preparations=2, decisions=3)
    if name == "cx":
        return dict(time_ns=cnot, readouts=3, preparations=6, decisions=11)
    if name == "trial_control":
        return dict(time_ns=p["reaction_time_ns"], readouts=0, preparations=0, decisions=1)
    magic = next((gate for gate in MAGIC_GATES if name.startswith(gate + "_q")), None)
    if magic is not None:
        suffix = name[len(magic) + 2:]
        if suffix not in {"0", "1", "2", "3"} or retry_cap < 1:
            raise ValueError("Magic events need a data wire in 0..3 and nonzero retries")
        wire = int(suffix)
        cnots = 3*(wire + 1) + 1
        inverse_frame = int(magic == "tdg")
        round_overhead = (p["physical_initialization_time_ns"]
                          + p["physical_measurement_time_ns"] + p["reaction_time_ns"])
        supply = retry_cap*(factory_ns + p["factory_feedback_rounds_reserved"]*round_overhead
                            + p["reaction_time_ns"])
        return dict(time_ns=supply + cnots*cnot + destructive + p["reaction_time_ns"]
                    + init + inverse_frame*p["reaction_time_ns"],
                    readouts=3*cnots + 1, preparations=6*cnots + 1,
                    decisions=11*cnots + 3 + inverse_frame
                    + retry_cap*(p["factory_feedback_rounds_reserved"] + 1))
    if name in CLIFFORD_GATES:
        return dict(time_ns=p["reaction_time_ns"], readouts=0, preparations=0, decisions=1)
    raise ValueError(f"Unsupported frozen accounting operation {name!r}")


def _events(value: dict, label: str) -> dict:
    allowed = CLIFFORD_GATES | {"cx", "initial_reset", "measure"} | {
        f"{gate}_q{wire}" for gate in MAGIC_GATES for wire in range(DATA_QUBITS)}
    if not isinstance(value, dict) or not value or set(value) - allowed:
        raise ValueError(f"{label} must contain only canonical saved source events")
    for name, count in value.items():
        if not isinstance(name, str):
            raise ValueError("Source event names must be strings")
        _integer(count, label + " occurrence count", 1, MAX_OPERATIONS_PER_CLASS)
    if (value.get("initial_reset") != DATA_QUBITS or value.get("measure") != DATA_QUBITS
            or sum(value.values()) > MAX_OPERATIONS_PER_CLASS):
        raise ValueError("Canonical four-wire preparation/readout or operation caps changed")
    return dict(sorted(value.items()))


def _canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _signature(events: dict) -> dict:
    return dict(logical_qubits=DATA_QUBITS, event_counts=events, backend_sha256=BACKEND_SHA256)


def _magic_count(events: dict) -> int:
    return sum(events.get(f"{gate}_q{wire}", 0)
               for gate in MAGIC_GATES for wire in range(DATA_QUBITS))


def _validate_transform(baseline: dict, candidate: dict, segments: list) -> dict:
    if not isinstance(segments, list):
        raise ValueError("Every candidate must retain its selected word segments")
    expected = Counter(baseline)
    total = Counter(selected_words=0, added_data_transport_cx=0,
                    removed_magic_transport_cx=0, net_removed_cx=0)
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Each selected word segment must be a mapping")
        wire = _integer(segment.get("wire"), "selected word wire", 1, DATA_QUBITS - 1)
        t = _integer(segment.get("t_count"), "selected T count", 0, MAX_OPERATIONS_PER_CLASS)
        tdg = _integer(segment.get("tdg_count"), "selected T-dagger count", 0, MAX_OPERATIONS_PER_CLASS)
        magic = t + tdg
        if magic < 3:
            raise ValueError("Only strictly beneficial words with at least three magic gates are selected")
        for gate, count in (("t", t), ("tdg", tdg)):
            key = f"{gate}_q{wire}"
            if expected[key] < count:
                raise ValueError("Selected words duplicate or invent magic occurrences")
            expected[key] -= count
            expected[f"{gate}_q0"] += count
        added = 6*wire
        removed = 3*wire*magic
        net = removed - added
        expected["cx"] += added
        for key, value in (("added_data_transport_cx", added),
                           ("removed_magic_transport_cx", removed), ("net_removed_cx", net)):
            if key in segment and segment[key] != value:
                raise ValueError("Supplied word accounting disagrees with independent arithmetic")
            total[key] += value
        total["selected_words"] += 1
    expected = dict(sorted((name, count) for name, count in expected.items() if count))
    if expected != candidate:
        raise ValueError("Candidate source events do not exactly match the declared word shuttles")
    return dict(total)


def _downward_float(value: Fraction, *, decimal_only=False) -> float:
    result = float(value)
    while Fraction(str(result)) > value or not decimal_only and Fraction.from_float(result) > value:
        result = math.nextafter(result, 0.0)
    return result


def _retry_cap(states: int, budget: Fraction) -> int:
    if states == 0:
        return 0
    bound = Fraction(states)
    for cap in range(1, 17):
        bound *= Fraction("0.01")
        if bound <= budget:
            return cap
    raise ValueError("Frozen factory retry domain is insufficient")


def _macro_counts(events: dict, shots: int, profile: dict) -> Counter:
    # Zero native pipeline time is used only to cancel identical supply terms
    # in a difference.  No absolute hardware/model runtime is emitted from it.
    counts = Counter(events)
    counts["initial_reset"] += WORKSPACE_INITIALIZATIONS
    counts["trial_control"] += 1
    totals = Counter()
    for name, count in counts.items():
        cost = frozen_macro_cost(profile, _MACHINE["data_code_distance"], name, 0,
                                 GLOBAL_FACTORY_RETRY_CAP)
        for key, value in cost.items():
            totals[key] += shots*count*value
    return totals


def build_accounting(classes: list, profile: dict | None = None) -> dict:
    """Validate and account the entire fixed 54-class / 82,032-shot workload.

    Each class is ``{class_id, shots, baseline_event_counts,
    candidate_event_counts, segments}``; selected segments contain ``wire``,
    separate ``t_count`` and ``tdg_count``.  Structural source/phase and density
    checks remain separate.  This function rejects a candidate event-ledger
    divergence inside any old physical group and never merges the old groups.
    """
    p = _check_profile(profile)
    if not isinstance(classes, list) or len(classes) != EXPECTED_CLASSES:
        raise ValueError("All 54 saved source classes are required exactly once")
    seen = set()
    groups = {}
    source_totals = {"baseline": Counter(), "candidate": Counter()}
    macro_totals = {"baseline": Counter(), "candidate": Counter()}
    transform_totals = Counter()
    unique_operations = Counter()
    for item in classes:
        if not isinstance(item, dict):
            raise ValueError("Source classes must be mappings")
        key = item.get("class_id")
        if (not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
                or key in seen):
            raise ValueError("Saved source class identities must be distinct lowercase SHA-256 strings")
        seen.add(key)
        shots = _integer(item.get("shots"), "class preparations", 1, EXPECTED_PREPARATIONS)
        baseline = _events(item.get("baseline_event_counts"), "baseline")
        candidate = _events(item.get("candidate_event_counts"), "candidate")
        change = _validate_transform(baseline, candidate, item.get("segments"))
        signature = _signature(baseline)
        parent_id = _canonical_digest(signature)
        candidate_id = _canonical_digest(_signature(candidate))
        if parent_id not in groups:
            groups[parent_id] = dict(parent_physical_class_id=parent_id,
                baseline_signature=signature, candidate_signature_sha256=candidate_id,
                baseline_event_counts=baseline, candidate_event_counts=candidate,
                shots=0, source_class_ids=[], total_t_states=0)
        group = groups[parent_id]
        if group["candidate_event_counts"] != candidate:
            raise ValueError("Candidate event costs diverge inside an immutable parent allocation group")
        group["shots"] += shots
        group["source_class_ids"].append(key)
        group["total_t_states"] += shots*_magic_count(baseline)
        for role, events in (("baseline", baseline), ("candidate", candidate)):
            source_totals[role].update({name: count*shots for name, count in events.items()})
            macro_totals[role].update(_macro_counts(events, shots, p))
            unique_operations[role] += sum(events.values())
        transform_totals.update({name: value*shots for name, value in change.items()})
    preparations = sum(group["shots"] for group in groups.values())
    t_states = sum(group["total_t_states"] for group in groups.values())
    if (len(groups) != EXPECTED_PARENT_GROUPS or preparations != EXPECTED_PREPARATIONS
            or t_states != EXPECTED_MAGIC_STATES):
        raise ValueError("The immutable parent partition, preparations or actual magic-state demand changed")
    if any(count > MAX_WORKLOAD_OPERATIONS for count in unique_operations.values()):
        raise ValueError("Whole-workload source materialization exceeds its frozen cap")
    retry = _retry_cap(t_states, _BUDGETS["factory_availability"])
    if retry != GLOBAL_FACTORY_RETRY_CAP:
        raise ValueError("The unchanged workload must retain the global six-retry contract")
    for group in groups.values():
        physical = _BUDGETS["physical_failure"]*group["shots"]/preparations
        factory = _BUDGETS["factory_availability"]*group["total_t_states"]/t_states
        group.update(physical_failure_allocation_exact=str(physical),
            factory_availability_allocation_exact=str(factory),
            physical_failure_budget=_downward_float(physical),
            factory_availability_budget=(_downward_float(factory, decimal_only=True)
                                         if group["total_t_states"] else 1e-300),
            global_factory_retry_cap=retry,
            workspace_initializations_per_preparation=WORKSPACE_INITIALIZATIONS,
            trial_control_events_per_preparation=1,
            reserved_logical_patches=RESERVED_PATCHES)
        if group["total_t_states"] and _retry_cap(group["total_t_states"], factory) != retry:
            raise ValueError("A retained parent factory allocation changed the retry cap")
    savings = {key: macro_totals["baseline"][key] - macro_totals["candidate"][key]
               for key in ("time_ns", "readouts", "preparations", "decisions")}
    net_cx = transform_totals["net_removed_cx"]
    if net_cx < 1:
        raise ValueError("The fixed co-design must strictly reduce whole-workload transport/controller work")
    cnot = frozen_macro_cost(p, _MACHINE["data_code_distance"], "cx", 0, retry)
    if savings != {key: net_cx*value for key, value in cnot.items()} or any(v < 0 for v in savings.values()):
        raise ValueError("Whole-workload savings failed exact independent CX-macro reconciliation")
    non_time_macros = {role: {key: macro_totals[role][key]
                              for key in ("readouts", "preparations", "decisions")}
                       for role in ("baseline", "candidate")}
    return dict(schema_version=1, status="source_ledger_only_not_native_physically_executed",
        profile=p, frozen_machine=frozen_machine(), parent_group_count=len(groups),
        candidate_distinct_event_cost_signatures=len({g["candidate_signature_sha256"] for g in groups.values()}),
        parent_groups=list(groups.values()), total_preparations=preparations,
        actual_magic_states=t_states, global_factory_retry_cap=retry,
        factory_availability_bound_exact=str(Fraction(t_states)*Fraction("0.01")**retry),
        allocations_exact={key: str(value) for key, value in _BUDGETS.items()},
        sampling_family_groups=48, sampling_per_group_failure_exact=str(Fraction("0.04")/48),
        statistical_experiment_scope="unchanged archived selection-aware inference and separate prospective fixed script; no new samples",
        source_events_shot_weighted={role: dict(sorted(total.items())) for role, total in source_totals.items()},
        macro_events_shot_weighted=non_time_macros, transform_counts_shot_weighted=dict(transform_totals),
        savings=savings,
        projected_candidate_serial_model_time_ns=_MACHINE["baseline_serial_model_time_ns"] - savings["time_ns"],
        timing_scope="difference in the frozen d15 serialized ledger at identical saved native supply; not a new native result or hardware guarantee",
        prospective_native_caps=dict(trace_estimates=54, expected_parent_trace_estimates=49,
            factory_frontier_queries=0, pipeline_candidates=0, synthesis_calls=0,
            physical_wall_seconds=60, decoded_record_bytes=64*1024*1024,
            candidate_bundle_bytes=16*1024*1024),
        physical_execution_enabled=False,
        physical_execution_requires="coordinator seventh-review clearance and specific frozen-candidate approval",
        unchanged_components=["every factory state and complete retry", "fresh F-to-W0 delivery per relocated T",
            "injection readout/correction/port reset", "separate factory-output handoff residence",
            "15-patch startup and reservation", "all 110 measured groups and source resets/readouts"],
        physical_validation_limitations=["candidate integer-ceiled exposure must be computed for each complete preparation",
            "no factory calibration/search or native trace executed", "conditional macro geometry and encoder/factory internals remain unsupported"])
