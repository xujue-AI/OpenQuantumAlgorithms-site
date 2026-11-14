"""Fixed-candidate complete-draw DQI accounting; native execution is opt-in.

The pure ledger reproduces the frozen project ``instrument_resources.py``
macro equations (SHA-256 fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863).
Only the accepted six-/nine-wire tasks are supported. The shared eight-wire
instrument API is neither imported nor widened. Native estimation requires
an injected, precharged callback; no factory model/search or synthesis runs
from this module. The caller must bind the exact native binary/environment
before supplying that callback. Nothing executes at module import.

Physical errors, readiness, geometry and controller timing are conditional
scenario inputs. Restoring an archived T cost does not restore a distillation
circuit or provenance graph, and the single d=15 point is not optimized.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


QDK_VERSION = "1.34.0"
DISTANCE = 15
T_ID = 1028
T_TIME_NS = 330_000
T_SPACE = 236_623
T_ERROR_HEX = "0x1.88ea67f531f32p-56"
T_SHA256 = "ad99b8ce36f8b556c05d877305e97e5e654f31c0d3756f0630c5bb2b44b184cb"
ISA_SHA256 = "2c94c77a1cb4779ac6b4d2f49d9312d70974b9ff82650b4f252ea52fe939c102"
MAX_COMPILED_OPERATIONS = 20_000
MAX_MACRO_CLASSES = 32
MAX_FACTORY_RETRIES = 16
MAX_SCALAR_INTEGER = (1 << 63) - 1
MAX_TRACE_JSON_BYTES = 64 << 10
PHYSICAL_BUDGET = Fraction(1, 400)
AVAILABILITY_BUDGET = Fraction(1, 400)
_PROFILE_ITEMS = (
    ("physical_error_rate", 1e-4),
    ("physical_gate_time_ns", 100),
    ("physical_measurement_time_ns", 500),
    ("physical_initialization_time_ns", 100),
    ("code_cycle_time_ns", 1000),
    ("reaction_time_ns", 10_000),
    ("controller_failure_per_decision", 1e-15),
    ("memory_prefactor", .1),
    ("initialization_residual_prefactor", .1),
    ("memory_threshold", .01),
    ("factory_pipeline_failure_bound", .01),
    ("factory_feedback_rounds_reserved", 3),
    ("maximum_code_distance", 49),
)
_SOURCE_PINS = (
    ("_isa_enumeration.py", "f35a4248c0d29174daac033d12f8296ad6c3b5a203a4866703d763a5b65aa97e"),
    ("models/factories/_round_based.py", "c5fcb478c86534cf9888bc8a9b9d3e5a733355bf92c41072228d99fdc0d19110"),
    ("models/qec/_surface_code.py", "3530f92f328e325d6d0c3e245f8688c11956fa6f63b1e25455b5d14cd844ed35"),
)
_SINGLE_GATES = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg"})


class NativeExecutionDisabled(PermissionError):
    """An ordinary data/ledger API is not permission to enter QDK."""


class NativeResultError(RuntimeError):
    """Keep a bounded raw return even when its validation fails."""

    def __init__(self, message: str, raw_result: dict):
        super().__init__(message)
        self.raw_result = deepcopy(raw_result)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _same_json(actual, expected) -> bool:
    """Bound validation by the tiny expected schema before serialization."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        if len(actual) != len(expected) or any(type(key) is not str for key in actual):
            return False
        return set(actual) == set(expected) and all(_same_json(actual[key], value)
                                                   for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_json(a, b)
                                                   for a, b in zip(actual, expected, strict=True))
    return actual == expected


def _integer(value, name: str, minimum: int = 0, maximum: int = MAX_SCALAR_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a bounded integer")
    return value


def _case(n: int) -> tuple[int, int]:
    if type(n) is not int or n not in (2, 3):
        raise ValueError("The fixed resource task supports n=2 or n=3 only")
    return 3 * n, 5 if n == 2 else 3


def frozen_profile() -> dict:
    """Return a fresh copy of every frozen scenario field."""
    return dict(_PROFILE_ITEMS)


def check_profile(profile: dict) -> dict:
    expected = frozen_profile()
    if not _same_json(profile, expected):
        raise ValueError("The full frozen instrument profile changed")
    return expected


def _expected_t_record() -> dict:
    return dict(id=T_ID, encoding=1, arity=1,
                time={"integer": T_TIME_NS}, space={"integer": T_SPACE},
                error_rate={"float_hex": T_ERROR_HEX}, expected_time_ns=T_TIME_NS,
                expected_space=T_SPACE, expected_output_error_hex=T_ERROR_HEX)


def check_identity(identity: dict) -> dict:
    """Check all fields of the one archived cost identity, not just its hash."""
    record = _expected_t_record()
    expected = dict(
        identity_scope="exact native factory ISA/T cost-model values; no distillation-layout certificate",
        native_factory_isa=[record], native_t_instruction=record,
        native_factory_isa_sha256=ISA_SHA256, native_t_instruction_sha256=T_SHA256,
        native_sources=dict(_SOURCE_PINS),
        surface_code_query=dict(distance=DISTANCE, code_cycle_override=1000,
                               error_correction_threshold=.01),
    )
    if not _same_json(identity, expected):
        raise ValueError("The exact archived factory identity or source/query fields changed")
    if _digest(record) != T_SHA256 or _digest([record]) != ISA_SHA256:
        raise ValueError("The fixed T/ISA constants do not match their recorded canonical hashes")
    return deepcopy(record)


def majority_path_error(distance: int, physical_error: float) -> float:
    """Frozen independent-readout model: majority of d length-d parity paths."""
    if (type(distance) is not int or distance != DISTANCE
            or type(physical_error) is not float or physical_error != 1e-4):
        raise ValueError("Use the fixed d=15, p=1e-4 readout scenario")
    parity_error = -math.expm1(distance * math.log1p(-2 * physical_error)) / 2
    return math.fsum(math.comb(distance, k) * parity_error ** k * (1 - parity_error) ** (distance - k)
                     for k in range((distance + 1) // 2, distance + 1))


def factory_retry_cap(total_states: int) -> tuple[int, Fraction]:
    """Smallest K<=16 with N_T*(1/100)^K<=1/400; no float/log shortcut."""
    _integer(total_states, "whole-task T states", 1, 100_000)
    bound = Fraction(total_states)
    for cap in range(1, MAX_FACTORY_RETRIES + 1):
        bound *= Fraction(1, 100)
        if bound <= AVAILABILITY_BUDGET:
            return cap, bound
    raise ValueError("The fixed complete-pipeline retry allowance is infeasible")


def macro_cost(name: str, profile: dict, factory_retry: int, source_wires: int) -> dict:
    """Literal frozen time/readout/preparation/decision equations, plus detail."""
    p = check_profile(profile)
    if type(name) is not str or not 1 <= len(name) <= 16:
        raise ValueError("Accounting macro name must be a short string")
    _integer(factory_retry, "factory retry cap", 1, MAX_FACTORY_RETRIES)
    if type(source_wires) is not int or source_wires not in (6, 9):
        raise ValueError("Only six-/nine-wire routing positions are supported")
    dtime = DISTANCE * p["code_cycle_time_ns"]
    init = p["physical_initialization_time_ns"] + dtime + p["reaction_time_ns"]
    destructive = p["physical_measurement_time_ns"] + p["reaction_time_ns"]
    parity = 2 * init + dtime + destructive
    cnot = 3 * parity + 2 * p["reaction_time_ns"]
    detail = {}
    if name == "initial_reset":
        cost = dict(time_ns=init, readouts=0, preparations=1, decisions=1)
    elif name == "reset":
        cost = dict(time_ns=parity + 2 * p["reaction_time_ns"] + dtime,
                    readouts=1, preparations=2, decisions=5)
    elif name == "measure":
        cost = dict(time_ns=parity, readouts=1, preparations=2, decisions=3)
    elif name == "cx":
        cost = dict(time_ns=cnot, readouts=3, preparations=6, decisions=11)
    elif name == "trial_control":
        cost = dict(time_ns=p["reaction_time_ns"], readouts=0, preparations=0, decisions=1)
    elif type(name) is str and name.startswith(("t_q", "tdg_q")):
        prefix, wire_text = name.split("_q", 1)
        if not wire_text.isascii() or not wire_text.isdecimal() or str(int(wire_text)) != wire_text:
            raise ValueError("Malformed magic-state source-wire position")
        wire = _integer(int(wire_text), "magic-state wire", 0, source_wires - 1)
        cnots = 3 * (wire + 1) + 1
        inverse_frame = int(prefix == "tdg")
        round_overhead = (p["physical_initialization_time_ns"]
                          + p["physical_measurement_time_ns"] + p["reaction_time_ns"])
        supply = factory_retry * (T_TIME_NS + p["factory_feedback_rounds_reserved"] * round_overhead
                                  + p["reaction_time_ns"])
        cost = dict(time_ns=supply + cnots * cnot + destructive + p["reaction_time_ns"]
                            + init + inverse_frame * p["reaction_time_ns"],
                    readouts=3 * cnots + 1, preparations=6 * cnots + 1,
                    decisions=11 * cnots + 3 + inverse_frame
                              + factory_retry * (p["factory_feedback_rounds_reserved"] + 1))
        detail = dict(neighbor_transport_cnots=3 * (wire + 1), injection_cnots=1,
                      complete_pipeline_attempts=factory_retry,
                      native_pipeline_time_ns=T_TIME_NS,
                      feedback_rounds_reserved=p["factory_feedback_rounds_reserved"] * factory_retry,
                      full_pipeline_supply_time_ns=supply,
                      inverse_frame_decisions=inverse_frame)
    elif name in _SINGLE_GATES - {"t", "tdg"}:
        cost = dict(time_ns=p["reaction_time_ns"], readouts=0, preparations=0, decisions=1)
    else:
        raise ValueError("Unsupported frozen accounting macro")
    return {**cost, "detail": detail}


def compiled_event_counts(n: int, compiled_ops: list) -> dict:
    """Fail closed on arity, opaque gates, nonneighbor CXs or changed events."""
    wires, _ = _case(n)
    if type(compiled_ops) is not list or not 1 <= len(compiled_ops) <= MAX_COMPILED_OPERATIONS:
        raise ValueError("Compiled operations must be a bounded list")
    counts = Counter()
    resets = 0
    readouts = 0
    output_started = False
    unitary_started = False
    for item in compiled_ops:
        if type(item) is not list or len(item) != 4:
            raise ValueError("Compiled operation records have four list fields")
        name, qargs, cargs, params = item
        if (type(name) is not str or type(qargs) is not list or type(cargs) is not list
                or type(params) is not list or params):
            raise ValueError("Only bound primitive operation records are supported")
        if len(name) > 16 or name not in _SINGLE_GATES | {"cx", "reset", "measure", "barrier"}:
            raise ValueError("An opaque or unsupported primitive remains")
        arity = wires if name == "barrier" else 2 if name == "cx" else 1
        if len(qargs) != arity or len(cargs) != int(name == "measure"):
            raise ValueError("Compiled primitive operand lengths changed")
        for q in qargs:
            _integer(q, "quantum wire", 0, wires - 1)
        for c in cargs:
            _integer(c, "readout bit", 0, n - 1)
        if name == "barrier":
            if qargs != list(range(wires)) or cargs:
                raise ValueError("Source barriers must cover the complete fixed register")
            continue
        if name == "reset":
            if (unitary_started or output_started or resets >= wires
                    or qargs != [resets] or cargs):
                raise ValueError("Every source wire must reset exactly once before all gates")
            resets += 1
            counts["initial_reset"] += 1
            continue
        if resets != wires:
            raise ValueError("A compiled draw omitted initial source reset")
        if name == "measure":
            if readouts >= n or qargs != [n + 1 + readouts] or cargs != [readouts]:
                raise ValueError("Assignment readout order/mapping changed")
            output_started = True
            readouts += 1
            counts["measure"] += 1
            continue
        if output_started or cargs:
            raise ValueError("Only terminal assignment readout may use classical bits")
        if name == "cx":
            if len(qargs) != 2 or abs(qargs[0] - qargs[1]) != 1:
                raise ValueError("A routed CX must use distinct neighboring wires")
        elif name not in _SINGLE_GATES or len(qargs) != 1:
            raise ValueError("An opaque or unsupported primitive remains")
        unitary_started = True
        counts[f"{name}_q{qargs[0]}" if name in {"t", "tdg"} else name] += 1
    if resets != wires or readouts != n:
        raise ValueError("Complete source initialization and assignment readout are required")
    t_count = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    if t_count < (35 if n == 2 else 126):
        raise ValueError("The accepted explicit source T/Tdg occurrences were omitted")
    return dict(counts)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _floor_probability(value: Fraction) -> float:
    number = float(value)
    while Fraction.from_float(number) > value:
        number = math.nextafter(number, 0.0)
    if not math.isfinite(number) or not 0 < number < 1:
        raise ValueError("The native ceiling is outside the fixed probability domain")
    return number


def build_ledger(n: int, compiled_ops: list, identity: dict, profile: dict) -> dict:
    """Prepare the full pure conditional ledger; make no backend/native call."""
    source_wires, draws = _case(n)
    p = check_profile(profile)
    check_identity(identity)
    counts = Counter(compiled_event_counts(n, compiled_ops))
    counts["trial_control"] += 1
    patches = 4 * source_wires - 1
    extra_initializations = patches - source_wires
    counts["initial_reset"] += extra_initializations
    if len(counts) > MAX_MACRO_CLASSES:
        raise ValueError("The fixed ledger exceeded 32 macro classes")
    t_count = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    total_t = t_count * draws
    retry, availability = factory_retry_cap(total_t)
    factory_error = float.fromhex(T_ERROR_HEX)
    if (factory_error * total_t > float(PHYSICAL_BUDGET) * .75
            or factory_error >= 7.1 * p["physical_error_rate"]):
        raise ValueError("The preselected factory fails the frozen output-quality filters")
    readout_error = majority_path_error(DISTANCE, p["physical_error_rate"])
    patch_error = p["memory_prefactor"] * (p["physical_error_rate"] / p["memory_threshold"]) ** ((DISTANCE + 1) // 2)
    initialization_error = p["initialization_residual_prefactor"] * (p["physical_error_rate"] / p["memory_threshold"]) ** ((DISTANCE + 1) // 2)
    rows = []
    for name, count in sorted(counts.items()):
        cost = macro_cost(name, p, retry, source_wires)
        event_error = (cost["readouts"] * readout_error + cost["preparations"] * initialization_error
                       + cost["decisions"] * p["controller_failure_per_decision"])
        rows.append(dict(operation=name, count=count, **cost, event_failure_model=event_error))
    runtime = sum(row["count"] * row["time_ns"] for row in rows)
    cycles = _ceil_div(runtime, p["code_cycle_time_ns"])
    memory_error = patches * cycles * patch_error
    round_overhead = (p["physical_initialization_time_ns"] + p["physical_measurement_time_ns"]
                      + p["reaction_time_ns"])
    port_hold_ns = (p["factory_feedback_rounds_reserved"] * round_overhead + p["reaction_time_ns"]
                    + 3 * macro_cost("cx", p, retry, source_wires)["time_ns"])
    port_hold_error = t_count * _ceil_div(port_hold_ns, p["code_cycle_time_ns"]) * patch_error
    event_error_sum = sum(row["count"] * row["event_failure_model"] for row in rows)
    expected_error = memory_error + port_hold_error + event_error_sum + t_count * factory_error
    if (not math.isfinite(expected_error) or expected_error < 0
            or Fraction.from_float(expected_error) * draws > PHYSICAL_BUDGET):
        raise ValueError("The single frozen candidate is infeasible; no fallback is allowed")
    patch_size = 2 * DISTANCE ** 2 - 1
    maximum_error = _floor_probability(PHYSICAL_BUDGET / draws)
    return dict(
        status="prepared_fixed_candidate_ledger", n=n, source_wires=source_wires,
        complete_draws=draws, source_initializations_per_draw=source_wires,
        assignment_readouts_per_draw=n, reserved_logical_patches=patches,
        additional_patch_initializations_per_draw=extra_initializations,
        initializations_all_draws=patches * draws,
        profile=p, data_code_distance=DISTANCE, factory_count=1,
        compiled_operation_count=len(compiled_ops), macro_counts_per_draw=dict(counts),
        macro_ledger_per_draw=rows, macro_counts_all_draws={name: count * draws for name, count in counts.items()},
        t_states_per_draw=t_count, t_states_all_draws=total_t,
        data_workspace_physical_qubits=patches * patch_size,
        factory_physical_qubits=T_SPACE, expected_peak_physical_qubits=patches * patch_size + T_SPACE,
        runtime_per_draw_ns=runtime, worst_case_runtime_all_draws_ns=runtime * draws,
        code_cycles_per_draw=cycles, base_error_per_draw=memory_error + port_hold_error,
        expected_physical_error_per_draw=expected_error,
        expected_physical_failure_all_draws=expected_error * draws,
        physical_failure_budget_exact=str(PHYSICAL_BUDGET), native_max_error_hex=maximum_error.hex(),
        failure_components_all_draws=dict(
            data_workspace_code_exposure=memory_error * draws,
            factory_output_handoff_storage=port_hold_error * draws,
            logical_readout_scenario=sum(row["count"] * row["readouts"] for row in rows) * readout_error * draws,
            logical_initialization_scenario=sum(row["count"] * row["preparations"] for row in rows) * initialization_error * draws,
            controller_scenario=sum(row["count"] * row["decisions"] for row in rows) * p["controller_failure_per_decision"] * draws,
            factory_output_quality_model=total_t * factory_error,
        ),
        factory_supply=dict(
            native_output_error_hex=T_ERROR_HEX, native_pipeline_time_ns=T_TIME_NS,
            full_pipeline_retry_cap_per_state=retry, worst_case_pipeline_attempts=total_t * retry,
            availability_failure_exact=str(availability), availability_budget_exact=str(AVAILABILITY_BUDGET),
            complete_pipeline_attempts_independent_assumed=True,
            complete_pipeline_rejection_bound_exact="1/100",
            native_per_round_requirement_is_whole_pipeline_certificate=False,
            pipeline_feedback_rounds_reserved=p["factory_feedback_rounds_reserved"],
            extra_output_port_hold_ns_per_state=port_hold_ns,
        ),
        native_factory_isa_sha256=ISA_SHA256, native_t_instruction_sha256=T_SHA256,
        comparison_scope="one predeclared d=15 candidate; no Pareto search or optimality claim",
        hardware_guarantee=False,
        exclusions=[
            "factory internal circuits and exact distillation geometry",
            "startup encoder correctness and factory/data layout certification",
            "hardware decoder throughput and measured controller latency",
            "assignment communication and host scoring/selection reliability or hardware latency",
            "offline coefficient/circuit/synthesis preprocessing and reporting wall time",
            "nominal factory supply timing tails and formal interval certificates",
        ],
    )


def _instruction_record(instruction) -> dict:
    """Recheck the original fixed-arity serializer after construction/return."""
    record = dict(id=_integer(instruction.id, "native instruction id"),
                  encoding=_integer(instruction.encoding, "native encoding"),
                  arity=_integer(instruction.arity, "native arity", 1, 1))
    for name in ("time", "space", "error_rate"):
        value = getattr(instruction, name)()
        if type(value) is int:
            record[name] = {"integer": _integer(value, f"native {name}")}
        elif type(value) is float and math.isfinite(value) and value >= 0:
            record[name] = {"float_hex": value.hex()}
        else:
            raise ValueError("Native fixed-arity cost has an unsupported/nonfinite type")
    record["expected_time_ns"] = _integer(instruction.expect_time(), "native expected time")
    record["expected_space"] = _integer(instruction.expect_space(), "native expected space")
    error = instruction.expect_error_rate()
    if type(error) is not float or not math.isfinite(error) or error < 0:
        raise ValueError("Native output error is not a finite float")
    record["expected_output_error_hex"] = error.hex()
    return record


def _check_runtime_sources(identity: dict) -> Path:
    """Read recorded package-source fingerprints before importing QDK."""
    distribution = importlib.metadata.distribution("qdk")
    if distribution.version != QDK_VERSION:
        raise ValueError("Exact QDK 1.34.0 is required")
    package_root = Path(distribution.locate_file("qdk")).resolve()
    for relative, expected in identity["native_sources"].items():
        path = package_root / "qre" / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("A pinned QDK source is not a regular file")
        with path.open("rb") as stream:
            raw = stream.read((1 << 20) + 1)
        if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("A recorded QDK factory/code/enumeration source changed")
    return package_root


def _typed_scalar(value) -> dict:
    if value is None:
        return {"type": "none"}
    if type(value) is int:
        if not -MAX_SCALAR_INTEGER <= value <= MAX_SCALAR_INTEGER:
            return {"type": "unsupported", "python_type": "int", "reason": "out_of_range"}
        return {"type": "int", "value": value}
    if type(value) is float:
        return {"type": "float", "hex": value.hex()}
    return {"type": "unsupported", "python_type": type(value).__name__[:80]}


def capture_native_result(result) -> dict:
    """Capture bounded public scalar returns before interpreting their meaning."""
    raw = dict(status="producer_returned_before_validation", validated=False)
    if result is None:
        raw["result"] = {"type": "none"}
        return raw
    raw["result"] = {}
    for name in ("qubits", "runtime", "error"):
        try:
            raw["result"][name] = _typed_scalar(getattr(result, name))
        except Exception as error:
            raw["result"][name] = {"type": "access_error", "error_type": type(error).__name__[:80]}
    try:
        factories = []
        for index, (key, value) in enumerate(result.factories.items()):
            if index >= 2:
                raise ValueError("Unexpected extra factories exceed capture bound")
            item = {"instruction_id": _typed_scalar(key)}
            for name in ("copies", "runs", "states", "error_rate"):
                try:
                    item[name] = _typed_scalar(getattr(value, name))
                except Exception as error:
                    item[name] = {"type": "access_error", "error_type": type(error).__name__[:80]}
            factories.append(item)
        raw["factories"] = factories
    except Exception as error:
        raw["factories"] = {"type": "access_error", "error_type": type(error).__name__[:80]}
    raw["factory_result_metadata_scope"] = dict(
        copies=dict(required_value=1, status="captured_pending_validation"),
        uninterpreted_fields=["runs", "states", "error_rate"],
        uninterpreted_fields_used_in_accounting=False,
        authoritative_source="exact archived native T instruction",
        opaque_provenance_scope="not serialized; no original pipeline/layout restoration",
    )
    return raw


def _check_macro_isa(isa, macro_records: list, patches: int) -> None:
    for macro in macro_records:
        instruction = isa[macro["id"]]
        time = instruction.expect_time(patches)
        space = instruction.expect_space(patches)
        error = instruction.expect_error_rate(patches)
        if (instruction.arity is not None or type(instruction.encoding) is not int
                or instruction.encoding != 1
                or type(time) is not int or time != macro["time_ns"]
                or type(space) is not int or space != macro["space"]
                or type(error) is not float or not math.isfinite(error)
                or error.hex() != macro["error_hex"]):
            raise ValueError("Native macro ISA changed a frozen used-arity cost")


def _check_isa_ids(isa, macro_records: list) -> None:
    expected = {T_ID, *(macro["id"] for macro in macro_records)}
    if len(isa) != len(expected):
        raise ValueError("Native ISA contains missing or extra instruction classes")
    actual = set()
    for index, instruction in enumerate(isa):
        if index >= len(expected):
            raise ValueError("Native ISA iteration exceeded the fixed class bound")
        actual.add(_integer(instruction.id, "native instruction class"))
    if actual != expected:
        raise ValueError("Native ISA contains changed instruction classes")


def _validate_native_result(result, raw: dict, ledger: dict, union_bound,
                            macro_records: list) -> None:
    fields = raw.get("result", {})
    if (fields.get("qubits", {}).get("type") != "int"
            or fields.get("runtime", {}).get("type") != "int"
            or fields.get("error", {}).get("type") != "float"):
        raise ValueError("The native result did not return the exact required scalar types")
    qubits = _integer(fields["qubits"]["value"], "native peak qubits", 1)
    runtime = _integer(fields["runtime"]["value"], "native runtime", 1)
    error = float.fromhex(fields["error"]["hex"])
    if (qubits != ledger["expected_peak_physical_qubits"] or runtime != ledger["runtime_per_draw_ns"]
            or not math.isfinite(error) or error < 0
            or error > float.fromhex(ledger["native_max_error_hex"])
            or Fraction.from_float(error) * ledger["complete_draws"] > PHYSICAL_BUDGET
            or not math.isclose(error, ledger["expected_physical_error_per_draw"], rel_tol=1e-10, abs_tol=1e-30)):
        raise ValueError("Native time/space/error disagrees with the complete fixed-candidate ledger")
    if result.error_composition != union_bound:
        raise ValueError("Native physical error composition is not UnionBound")
    factories = raw.get("factories")
    if (type(factories) is not list or len(factories) != 1
            or factories[0].get("instruction_id") != {"type": "int", "value": T_ID}
            or factories[0].get("copies") != {"type": "int", "value": 1}):
        raise ValueError("Exactly one copy of the recorded T factory is required")
    for name in ("runs", "states"):
        item = factories[0].get(name, {})
        if item.get("type") != "int":
            raise ValueError("Captured-uninterpreted factory integer metadata has an invalid type")
        _integer(item["value"], f"captured factory {name}")
    item = factories[0].get("error_rate", {})
    if item.get("type") != "float":
        raise ValueError("Captured-uninterpreted factory error metadata has an invalid type")
    number = float.fromhex(item["hex"])
    if not math.isfinite(number) or number < 0:
        raise ValueError("Captured-uninterpreted factory error metadata is nonfinite/negative")
    if _instruction_record(result.isa[T_ID]) != _expected_t_record():
        raise ValueError("Returned native T cost identity changed")
    _check_isa_ids(result.isa, macro_records)
    _check_macro_isa(result.isa, macro_records, ledger["reserved_logical_patches"])


def estimate_case(n: int, compiled_ops: list, identity: dict, profile: dict,
                  native_call=None) -> dict:
    """One injected direct Trace estimate after complete pure preflight.

    ``native_call(kind, function, *args, **kwargs)`` must precharge the caller's
    per-case/global direct-Trace counter and enforce its reviewed deadline.
    The only kind emitted here is ``direct_trace``. Default calls are disabled
    before package metadata checks or QDK imports; there is no fallback.
    """
    if not callable(native_call):
        raise NativeExecutionDisabled("An injected, precharged direct-Trace callback is required")
    ledger = build_ledger(n, compiled_ops, identity, profile)
    record = check_identity(identity)
    identity_copy = deepcopy(identity)
    package_root = _check_runtime_sources(identity_copy)
    from qdk.qre import ErrorComposition, Instruction, LOGICAL, Trace, constant_function, linear_function
    from qdk.qre.models import GateBased
    import qdk.qre as qre
    import qdk._native as extension

    if (Path(qre.__file__).resolve() != package_root / "qre/__init__.py"
            or Path(extension.__file__).resolve() != package_root / "_native.abi3.so"
            or int(LOGICAL) != 1):
        raise ValueError("Imported QDK origin/encoding differs from the bound runtime")
    p = ledger["profile"]
    context = GateBased(error_rate=p["physical_error_rate"], gate_time=p["physical_gate_time_ns"],
                        measurement_time=p["physical_measurement_time_ns"]).context()
    factory = Instruction.fixed_arity(record["id"], record["encoding"], 1,
                                     record["time"]["integer"], record["space"]["integer"], None,
                                     float.fromhex(record["error_rate"]["float_hex"]))
    factory_node = context.add_instruction(factory, source=[])
    if _instruction_record(factory) != record:
        raise ValueError("Native constructor failed exact recorded T restoration")
    patches = ledger["reserved_logical_patches"]
    patch_size = 2 * DISTANCE ** 2 - 1
    trace = Trace(patches)
    nodes = []
    macro_records = []
    for index, row in enumerate(ledger["macro_ledger_per_draw"]):
        identifier = 2 ** 40 + index
        nodes.append(context.add_instruction(
            identifier, encoding=LOGICAL, arity=None,
            time=constant_function(row["time_ns"]), space=linear_function(patch_size),
            error_rate=constant_function(row["event_failure_model"]),
        ))
        block = trace.add_block(repetitions=row["count"])
        block.add_operation(identifier, list(range(patches)))
        macro_records.append(dict(id=identifier, arity=patches, time_ns=row["time_ns"],
                                  space=patches * patch_size, error_hex=row["event_failure_model"].hex()))
    trace.increment_resource_state(T_ID, ledger["t_states_per_draw"])
    trace.increment_base_error(ledger["base_error_per_draw"])
    isa = context.make_isa(*nodes, factory_node)
    if (_instruction_record(isa[T_ID]) != record
            or _digest([_instruction_record(isa[T_ID])]) != ISA_SHA256):
        raise ValueError("The fresh native ISA failed exact recorded factory restoration")
    _check_isa_ids(isa, macro_records)
    _check_macro_isa(isa, macro_records, patches)
    expected_gates = {macro["id"]: row["count"]
                      for macro, row in zip(macro_records, ledger["macro_ledger_per_draw"], strict=True)}
    if (trace.compute_qubits != patches or trace.gate_counts != expected_gates
            or trace.resource_states != {T_ID: ledger["t_states_per_draw"]}
            or float(trace.base_error).hex() != ledger["base_error_per_draw"].hex()):
        raise ValueError("Native trace changed reservations, counts or base physical exposure")
    trace_json = trace.to_json().encode()
    if len(trace_json) > MAX_TRACE_JSON_BYTES:
        raise ValueError("The direct physical trace exceeds its serialization cap")
    result = native_call("direct_trace", trace.estimate, isa,
                         max_error=float.fromhex(ledger["native_max_error_hex"]),
                         composition=ErrorComposition.UnionBound)
    raw = capture_native_result(result)
    try:
        _validate_native_result(result, raw, ledger, ErrorComposition.UnionBound, macro_records)
    except Exception as error:
        raise NativeResultError(str(error), raw) from error
    validated_metadata_scope = deepcopy(raw["factory_result_metadata_scope"])
    validated_metadata_scope["copies"]["status"] = "validated_exactly_one"
    return {
        **ledger,
        "status": "computed_conditional_fixed_d15_model",
        "validated": True,
        "peak_physical_qubits": raw["result"]["qubits"]["value"],
        "native_error_per_draw_hex": raw["result"]["error"]["hex"],
        "native_physical_failure_all_draws_exact": str(Fraction.from_float(result.error) * ledger["complete_draws"]),
        "native_trace_json_sha256": hashlib.sha256(trace_json).hexdigest(),
        "native_trace_json_bytes": len(trace_json),
        "native_macro_cost_records": macro_records,
        "raw_native_result": raw,
        "factory_result_metadata_scope": validated_metadata_scope,
        "native_backend": "QDK direct Trace with explicit complete-draw ISA and restored fixed T cost",
        "native_error_composition": "UnionBound",
        "factory_identity_restore_scope": "exact recorded cost values in fresh context; no original pipeline/layout graph",
        "native_query_scope": {"direct_trace_calls": 1, "qre_estimate_calls": 0,
                               "factory_frontier_queries": 0, "factory_pipeline_candidates": 0},
    }
