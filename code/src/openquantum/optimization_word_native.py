"""Fixed recorded-value QDK adapter; estimation is disabled by default.

The seventh archive contains evaluated fixed-arity native instruction cost
records, not a serialized ISA/provenance graph.  This adapter uses the SAME
QDK 1.34.0 ``Instruction.fixed_arity`` constructor as the pinned factory
producer, with exact stored integers and IEEE error bits, and checks the
archived serializer identities afterward.  Its new graph is not the original
pipeline graph and is not a distillation/layout certificate.

Planning reads immutable source evidence only.  Execution requires an injected
estimator and durable before/after hooks; this file never calls Trace.estimate
or enumerates factories.  Every producer return is captured before validation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import zlib


RECIPE_SHA256 = "42dc12de0aa666e4fc9054c22b2321e58240959f1f74c647dc368894de79b694"
MANIFEST_SHA256 = "b25408c303fdc1ecef17454914c5f084f9439fe0c605468fb66653b313245385"
VALIDATION_SHA256 = "8446fc8e4e523531b666817bbcd1e33764d440c9d9417690ed7bbe7324e4465b"
ACCOUNTING_SHA256 = "19bd70a79f2103fd1cc141dc142ad954b3d0fad0934c5428f3cdf19a60f1f722"
CODEC_SHA256 = "d722622fd850b5b10d29898eaba38538dd257c870e09df1e28856fa1829cf4f3"
DAG_SHA256 = "731791448fcfd3e7fc2df18efa0d7d0db12c49254a709bb58d0286a93d1cca34"
ISA_SHA256 = "2c94c77a1cb4779ac6b4d2f49d9312d70974b9ff82650b4f252ea52fe939c102"
T_SHA256 = "ad99b8ce36f8b556c05d877305e97e5e654f31c0d3756f0630c5bb2b44b184cb"
PARENT_ALLOCATION_SHA256 = "8676d76e82fc48c0e13c33aa33f93e7591b03960dfaa1e104037628a423702b7"
QDK_VERSION = "1.34.0"
MAX_REQUESTS = 54
DECODED_CAP = 64*1024*1024
CLASS_CAP = 4*1024*1024
BUNDLE_CAP = 16*1024*1024
_NATIVE_PINS = {
    "__init__.py": "82cdfa32496f56dd9ebb379c7610a5c357e2b58ff201dd258c77c0db79a8a38d",
    "qre/__init__.py": "5580dfb76673b1b42c88447c827af7e5048573bb7b76143b5afe21242b84db6f",
    "qre/models/factories/_round_based.py": "c5fcb478c86534cf9888bc8a9b9d3e5a733355bf92c41072228d99fdc0d19110",
    "qre/models/qec/_surface_code.py": "3530f92f328e325d6d0c3e245f8688c11956fa6f63b1e25455b5d14cd844ed35",
    "qre/_isa_enumeration.py": "f35a4248c0d29174daac033d12f8296ad6c3b5a203a4866703d763a5b65aa97e",
    "qre/_architecture.py": "3d188a5a71143a4a023bfe5e06fbabf4de573cbb32abada823373ab1418f98cf",
    "qre/_instruction.py": "9f0c7b7cfa22c68b23aba131c79c0665273b0555cb6ade24ae8eb558eadfb7cb",
    "qre/_qre.pyi": "efe37f94976f8f658cc9fa17c9380533f0639d208546b1628a5dcbd3e4a19791",
    "qre/_qre.py": "58b0a06b4602e8e5737f283a8b21c54434a1df33f8ad1ad5f80ec09343b7a8b2",
    "qre/_trace.py": "75c65fdc24f8e452486fbd022ca1272aca5dfb10f45b237205929343189012d4",
    "qre/models/qubits/_gate_based.py": "8e815aef59c2d724259bb13ddb5e2212332d6febd9bf2e00cca9c33d602b0273",
    "qre/instruction_ids.py": "87a1a9a8286caab890d343f4cc56dd6c7facf8f935bc2fa7c2bc150982b15f6e",
    "_native.abi3.so": "98a99d5bb076a8e306a8ca7d241d9b96348a600e71e14c4f97fd70c6b1986848",
}


class NativeExecutionDisabled(PermissionError):
    """A concrete reviewed adapter is not authorization to run it."""


def factory_metadata_scope(*, copies_validated=False) -> dict:
    """Separate an enforced copy count from captured, uninterpreted metadata.

    The pinned FactoryResult API does not establish whether runs/states/error
    rate are per-state or aggregate quantities.  Their range guards do not
    certify those semantics, and none of them enters resource accounting.
    """
    return dict(
        fields={
            "copies": dict(status="validated_exactly_one" if copies_validated else "captured_pending_validation",
                           required_value=1),
            **{name: dict(status="captured_unvalidated", used_in_accounting=False)
               for name in ("runs", "states", "error_rate")}},
        authoritative_cost_source="exact_native_isa_t_instruction",
        authoritative_cost_fields=["time", "space", "error_rate"],
        metadata_guard_scope="type and finite/nonnegative range guards only; per-state or aggregate semantics are not established")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _regular(path, expected=None) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Native planning inputs must be regular saved files")
    if path.stat().st_size > DECODED_CAP:
        raise ValueError("Saved native planning file exceeds the 64 MiB read guard")
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("An immutable saved input or pinned producer changed")
    return raw


def _within(root: Path, relative: str) -> Path:
    path = root / relative
    if not isinstance(relative, str) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("A saved relative path escaped its input tree")
    return path


def _decoded(raw: bytes, cap: int) -> bytes:
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    value = decoder.decompress(raw, cap + 1)
    if (len(value) > cap or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("Saved compressed record exceeded its bound or has trailing streams")
    return value


def _module(path: Path, expected: str, name: str):
    raw = _regular(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def native_fingerprints() -> dict:
    """Bind the current adapter runtime, without importing or estimating QDK."""
    distribution = importlib.metadata.distribution("qdk")
    if distribution.version != QDK_VERSION:
        raise ValueError("Fixed native restoration requires QDK 1.34.0")
    root = Path(distribution.locate_file("qdk")).resolve()
    for relative, expected in _NATIVE_PINS.items():
        _regular(_within(root, relative), expected)
    return dict(package=QDK_VERSION, files=dict(_NATIVE_PINS),
                scope="current adapter runtime pins; the seventh archive did not pin the native binary")


def instruction_record(instruction) -> dict:
    """The exact archived seventh fixed-arity serializer, including float bits."""
    if instruction.arity != 1:
        raise ValueError("Recorded factory instruction must have arity one")
    values = dict(id=int(instruction.id), encoding=int(instruction.encoding), arity=instruction.arity)
    for name in ("time", "space", "error_rate"):
        value = getattr(instruction, name)()
        if isinstance(value, int):
            values[name] = {"integer": value}
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Native factory instruction has a nonfinite value")
            values[name] = {"float_hex": number.hex()}
    values["expected_time_ns"] = int(instruction.expect_time())
    values["expected_space"] = int(instruction.expect_space())
    values["expected_output_error_hex"] = float(instruction.expect_error_rate()).hex()
    return values


def check_recorded_identity(identity: dict) -> dict:
    """Fail closed on missing/extra fields or any changed native cost value."""
    expected = dict(id=1028, encoding=1, arity=1, time={"integer": 330_000},
        space={"integer": 236_623}, error_rate={"float_hex": "0x1.88ea67f531f32p-56"},
        expected_time_ns=330_000, expected_space=236_623,
        expected_output_error_hex="0x1.88ea67f531f32p-56")
    if (not isinstance(identity, dict) or identity.get("native_t_instruction") != expected
            or identity.get("native_factory_isa") != [expected]
            or identity.get("native_t_instruction_sha256") != T_SHA256
            or identity.get("native_factory_isa_sha256") != ISA_SHA256
            or _digest(identity.get("native_t_instruction")) != T_SHA256
            or _digest(identity.get("native_factory_isa")) != ISA_SHA256):
        raise ValueError("The exact recorded fixed-arity native identity changed")
    if set(identity) != {"native_t_instruction", "native_factory_isa", "native_t_instruction_sha256",
                         "native_factory_isa_sha256", "surface_code_query", "native_sources", "identity_scope"}:
        raise ValueError("The archived native identity field set changed")
    if identity.get("surface_code_query") != dict(distance=15, code_cycle_override=1000,
                                                error_correction_threshold=.01):
        raise ValueError("Native factory code/query assumptions changed")
    sources = {relative.removeprefix("qre/"): digest for relative, digest in _NATIVE_PINS.items()
               if relative in {"qre/models/factories/_round_based.py", "qre/models/qec/_surface_code.py",
                               "qre/_isa_enumeration.py"}}
    if identity.get("native_sources") != sources:
        raise ValueError("Archived native producer fingerprints changed")
    return deepcopy(expected)


def restore_recorded_factory_isa(identity: dict):
    """Restore exact cost-record identities with the original native constructor.

    No factory model or search is instantiated.  The graph/context is fresh;
    original pipeline provenance and a physical distillation circuit were not
    serialized in the archive and are not claimed to be restored.
    """
    recorded = check_recorded_identity(identity)
    native_fingerprints()
    from qdk.qre import Instruction
    from qdk.qre.models import GateBased
    import qdk.qre as qre
    import qdk._native as extension
    package_root = Path(importlib.metadata.distribution("qdk").locate_file("qdk")).resolve()
    if (Path(qre.__file__).resolve() != package_root / "qre/__init__.py"
            or Path(extension.__file__).resolve() != package_root / "_native.abi3.so"):
        raise ValueError("Imported QDK does not match the pinned native runtime")
    context = GateBased(error_rate=1e-4, gate_time=100, measurement_time=500).context()
    instruction = Instruction.fixed_arity(recorded["id"], recorded["encoding"], 1,
        recorded["time"]["integer"], recorded["space"]["integer"], None,
        float.fromhex(recorded["error_rate"]["float_hex"]))
    node = context.add_instruction(instruction, source=[])
    isa = context.make_isa(node)
    values = [instruction_record(value) for value in sorted(isa, key=lambda value: value.id)]
    if values != identity["native_factory_isa"] or _digest(values) != ISA_SHA256:
        raise ValueError("Native constructor did not restore the exact archived ISA cost records")
    if instruction_record(isa[1028]) != recorded or _digest(instruction_record(isa[1028])) != T_SHA256:
        raise ValueError("Native constructor did not restore the exact archived T record")
    return dict(context=context, isa=isa, node=node, instruction=instruction,
                identity_scope="exact recorded cost values; fresh provenance graph; no pipeline/layout restoration")


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _floor_float(value: Fraction) -> float:
    result = float(value)
    while Fraction.from_float(result) > value or Fraction(str(result)) > value:
        result = math.nextafter(result, 0.0)
    if not 0 < result < 1:
        raise ValueError("A positive conservative native ceiling is required")
    return result


def _qasm(operations: list) -> bytes:
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', 'bit[4] c;', 'qubit[4] q;']
    for name, wires, clbits, params in operations:
        if params:
            raise ValueError("Candidate native inputs must be fully materialized primitives")
        operands = ", ".join(f"q[{wire}]" for wire in wires)
        lines.append(f"c[{clbits[0]}] = measure {operands};" if name == "measure" else f"{name} {operands};")
    return ("\n".join(lines) + "\n").encode()


def _load_inputs(candidate_dir: Path, base_root: Path):
    recipe = json.loads(_regular(candidate_dir / "recipe.json", RECIPE_SHA256))
    artifact_manifest = json.loads(_regular(candidate_dir / "artifact-manifest.json", MANIFEST_SHA256))
    validation = json.loads(_regular(candidate_dir / "validation.json", VALIDATION_SHA256))
    if len(recipe["owned_source_sha256"]) != 8:
        raise ValueError("The immutable eighth source pin set changed")
    for relative, expected in recipe["owned_source_sha256"].items():
        _regular(_within(base_root, relative), expected)
    present = {p.relative_to(candidate_dir).as_posix() for p in candidate_dir.rglob("*") if p.is_file()}
    expected_set = set(artifact_manifest["files"]) | {"recipe.json", "input-inventory.json",
                                                    "accounting-plan.json", "artifact-manifest.json"}
    if present != expected_set or len(present) != 114:
        raise ValueError("The complete immutable 114-file candidate bundle changed")
    total_bytes = 0
    for relative in sorted(present):
        descriptor = artifact_manifest["files"].get(relative)
        raw = _regular(_within(candidate_dir, relative), descriptor["sha256"] if descriptor else None)
        total_bytes += len(raw)
        if descriptor and len(raw) != descriptor["bytes"]:
            raise ValueError("Candidate artifact byte count changed")
    if total_bytes != 4_894_762 or total_bytes > BUNDLE_CAP:
        raise ValueError("The candidate bundle byte identity changed")
    inventory_raw = _regular(candidate_dir / "input-inventory.json", recipe["input_inventory_sha256"])
    inventory = json.loads(inventory_raw)
    for relative, expected in inventory.items():
        _regular(_within(base_root, relative), expected)
    if {p.relative_to(base_root).as_posix() for p in (base_root / "assets/generated").rglob("*") if p.is_file()} != set(inventory):
        raise ValueError("Historical scientific archive file membership changed")
    for name in ("accounting.json", "accounting-plan.json"):
        _regular(candidate_dir / name, ACCOUNTING_SHA256)
    accounting = json.loads(_regular(candidate_dir / "accounting.json"))
    bundle = base_root / "assets/generated/optimization/compiled-replay"
    original_manifest = json.loads(_regular(bundle / "manifest.json", recipe["baseline_manifest_sha256"]))
    codec = _module(bundle / "optimization_compiled_codec.py", CODEC_SHA256, "fixed_native_pinned_codec")
    physical = codec.loads(_decoded(_regular(bundle / "physical-dag.json.gz", DAG_SHA256), DECODED_CAP))
    selected = physical["selected_minimum_runtime"]
    if (selected["native_factory_isa_sha256"] != ISA_SHA256 or selected["native_t_instruction_sha256"] != T_SHA256
            or selected["peak_physical_qubits"] != 243_358 or selected["total_preparations"] != 82_032
            or selected["total_t_states"] != 182_646_240 or selected["global_factory_retry_cap"] != 6
            or len(physical["physical_classes"]) != 49):
        raise ValueError("The selected seventh fixed machine/workload changed")
    identities = [item["point"]["native_factory_identity"] for item in selected["per_physical_class"]]
    for identity in identities:
        check_recorded_identity(identity)
    identity = deepcopy(identities[0])
    transport = _module(base_root / "src/openquantum/optimization_word_transport.py",
        recipe["owned_source_sha256"]["src/openquantum/optimization_word_transport.py"], "fixed_native_pinned_transport")
    costs = _module(base_root / "src/openquantum/optimization_word_accounting.py",
        recipe["owned_source_sha256"]["src/openquantum/optimization_word_accounting.py"], "fixed_native_pinned_accounting")
    if (recipe["profile"] != costs.frozen_profile() or recipe["common_machine"] != costs.frozen_machine()
            or physical["profile"] != recipe["profile"] or validation["archived_supports"] != {"qaoa": 3, "vqe": 0}
            or validation["recipe_sha256"] != RECIPE_SHA256 or validation["native_queries"]
            or validation["optimizer_calls"] or validation["synthesis_calls"] or validation["new_holdout_samples"]):
        raise ValueError("Frozen assumptions or source-only receipt boundaries changed")
    old_entries = {entry["class_id"]: entry for entry in original_manifest["classes"]}
    rows = []
    for plan in recipe["classes"]:
        key = plan["class_id"]
        entry = old_entries[key]
        raw_checkpoint = _regular(_within(bundle, entry["checkpoint"]), plan["checkpoint_sha256"])
        checkpoint = json.loads(_decoded(raw_checkpoint, CLASS_CAP))
        original = checkpoint["record"]
        if (checkpoint["identity"] != original_manifest["original_identity"] or checkpoint["key"] != key
                or original["class_id"] != key or original["shots"] != plan["shots"]
                or original["batches"] != plan["batches"] or float(original["clifford_t_global_phase"]).hex() != plan["global_phase_hex"]):
            raise ValueError("Original class/phase/repetition binding changed")
        original_qasm = _regular(_within(bundle, entry["circuits"]["clifford_t"]["file"]), plan["source_qasm_sha256"])
        if original_qasm != original["clifford_t_qasm"].encode():
            raise ValueError("Original saved QASM changed")
        candidate = json.loads(_decoded(_regular(candidate_dir / f"circuits/{key}.json.gz"), CLASS_CAP))
        qasm = _regular(candidate_dir / f"circuits/{key}.qasm", plan["candidate_qasm_sha256"])
        expected = transport.build_candidate(original["clifford_t_operations"])
        if (candidate["class_id"] != key or candidate["operations"] != expected["operations"]
                or candidate["segments"] != expected["segments"] or _qasm(candidate["operations"]) != qasm
                or transport.operation_digest(candidate["operations"]) != plan["candidate_operations_sha256"]
                or float(candidate["exact_global_phase"]).hex() != plan["global_phase_hex"]
                or candidate["source_qasm_sha256"] != plan["source_qasm_sha256"]):
            raise ValueError("Candidate primitives/segments/QASM/phase are not the frozen transformed source")
        rows.append(dict(plan=plan, events=expected["candidate_event_counts"],
                         baseline_events=expected["baseline_event_counts"]))
    if len(rows) != 54 or sum(row["plan"]["shots"] for row in rows) != 82_032:
        raise ValueError("The complete 54-class shot workload changed")
    return recipe, accounting, physical, identity, rows, costs


def _model_request(row: dict, parent: dict, profile: dict, identity: dict, costs) -> dict:
    events = Counter(row["events"])
    events["initial_reset"] += 11
    events["trial_control"] += 1
    distance, patches, retry = 15, 15, 6
    error_rate = profile["physical_error_rate"]
    parity_error = -math.expm1(distance*math.log1p(-2*error_rate))/2
    readout_error = math.fsum(math.comb(distance, k)*parity_error**k*(1-parity_error)**(distance-k)
                             for k in range((distance+1)//2, distance+1))
    patch_failure = profile["memory_prefactor"]*(error_rate/profile["memory_threshold"])**((distance+1)//2)
    initialization_error = profile["initialization_residual_prefactor"]*(error_rate/profile["memory_threshold"])**((distance+1)//2)
    instruction = identity["native_t_instruction"]
    factory_ns = instruction["time"]["integer"]
    factory_error = float.fromhex(instruction["error_rate"]["float_hex"])
    ledger = []
    for index, (name, count) in enumerate(sorted(events.items())):
        cost = costs.frozen_macro_cost(profile, distance, name, factory_ns, retry)
        event_error = cost["readouts"]*readout_error + cost["preparations"]*initialization_error + cost["decisions"]*profile["controller_failure_per_decision"]
        ledger.append(dict(operation=name, count=count, instruction_id=2**40+index,
                           **cost, event_failure_model_hex=event_error.hex()))
    runtime = sum(item["count"]*item["time_ns"] for item in ledger)
    cycles = _ceil_div(runtime, profile["code_cycle_time_ns"])
    t_count = sum(count for name, count in events.items() if name.startswith(("t_q", "tdg_q")))
    handoff_ns = (profile["factory_feedback_rounds_reserved"]*(profile["physical_initialization_time_ns"]
                  + profile["physical_measurement_time_ns"] + profile["reaction_time_ns"])
                  + profile["reaction_time_ns"] + 3*costs.frozen_macro_cost(profile, distance, "cx", factory_ns, retry)["time_ns"])
    handoff_cycles = _ceil_div(handoff_ns, profile["code_cycle_time_ns"])
    memory = patches*cycles*patch_failure
    handoff = t_count*handoff_cycles*patch_failure
    event_total = sum(item["count"]*float.fromhex(item["event_failure_model_hex"]) for item in ledger)
    expected_error = memory + handoff + event_total + t_count*factory_error
    shots = row["plan"]["shots"]
    exact_physical = Fraction(parent["physical_failure_allocation_exact"])*shots/parent["shots"]
    exact_factory = Fraction(parent["factory_availability_allocation_exact"])*(shots*t_count)/parent["total_t_states"]
    # Consume a downward split of the ORIGINAL parent's recorded float as well
    # as its exact fraction; collective class calls cannot exceed either one.
    physical_ceiling = min(exact_physical/shots, Fraction.from_float(parent["physical_failure_budget"])/parent["shots"])
    max_error = _floor_float(physical_ceiling)
    availability = Fraction(shots*t_count)*Fraction(".01")**retry
    if expected_error > max_error or availability > exact_factory:
        raise ValueError("The fixed candidate cannot fit its unchanged parent/class allowances")
    plan = row["plan"]
    request = dict(class_id=plan["class_id"], parent_physical_class_id=parent["physical_class_id"],
        shots=shots, total_t_states=shots*t_count, t_states_per_preparation=t_count,
        physical_failure_allocation_exact=str(exact_physical), factory_availability_allocation_exact=str(exact_factory),
        native_max_error_hex=max_error.hex(), native_max_error_target_exact=str(physical_ceiling),
        source_binding=dict(source_qasm_sha256=plan["source_qasm_sha256"], checkpoint_sha256=plan["checkpoint_sha256"],
            candidate_qasm_sha256=plan["candidate_qasm_sha256"], candidate_operations_sha256=plan["candidate_operations_sha256"],
            source_operations_sha256=plan["source_operations_sha256"], global_phase_hex=plan["global_phase_hex"],
            candidate_operations=plan["candidate_operations"], segments_sha256=plan["segments_sha256"]),
        event_counts=dict(sorted(events.items())), macro_ledger_per_preparation=ledger,
        runtime_per_preparation_ns=runtime, code_cycles_per_preparation=cycles,
        handoff_ns_per_t=handoff_ns, handoff_cycles_per_t=handoff_cycles,
        memory_failure_per_preparation_hex=memory.hex(), handoff_failure_per_preparation_hex=handoff.hex(),
        native_base_error_hex=(memory + handoff).hex(), expected_error_per_preparation_hex=expected_error.hex(),
        factory_availability_failure_exact=str(availability),
        weighted_macro_counts={name: shots*sum(item["count"]*item[name] for item in ledger)
                               for name in ("readouts", "preparations", "decisions")},
        integer_exposure_rule="ceil each complete preparation duration, then multiply by this class's exact shots",
        factory_quality_rule="one saved output-error charge per consumed T; six retries affect supply/availability, not output quality")
    request["trace_input_sha256"] = _digest(request)
    return request


def plan_fixed_requests(candidate_dir, base_root) -> dict:
    """Return a JSON-only plan of 54 individually weighted, immutable requests."""
    candidate_dir, base_root = Path(candidate_dir).resolve(), Path(base_root).resolve()
    recipe, accounting, physical, identity, rows, costs = _load_inputs(candidate_dir, base_root)
    fingerprints = native_fingerprints()
    parents = {group["physical_class_id"]: group for group in physical["physical_classes"]}
    class_parent = physical["compiled_class_mapping"]
    requests = []
    for row in rows:
        parent = parents[class_parent[row["plan"]["class_id"]]["physical_class_id"]]
        if row["baseline_events"] != parent["signature"]["event_counts"]:
            raise ValueError("A source class escaped its immutable original allocation parent")
        requests.append(_model_request(row, parent, recipe["profile"], identity, costs))
    raw_parents = [deepcopy({key: value for key, value in group.items() if key != "report"})
                   for group in physical["physical_classes"]]
    for parent in raw_parents:
        members = [request for request in requests if request["parent_physical_class_id"] == parent["physical_class_id"]]
        if (sum(item["shots"] for item in members) != parent["shots"]
                or sum(item["total_t_states"] for item in members) != parent["total_t_states"]
                or sum(Fraction(item["physical_failure_allocation_exact"]) for item in members) != Fraction(parent["physical_failure_allocation_exact"])
                or sum(Fraction(item["factory_availability_allocation_exact"]) for item in members) != Fraction(parent["factory_availability_allocation_exact"])):
            raise ValueError("Class splitting changed an exact original parent allocation")
    totals = dict(classes=len(requests), parent_groups=len(raw_parents),
        preparations=sum(item["shots"] for item in requests), t_states=sum(item["total_t_states"] for item in requests),
        serial_runtime_ns=sum(item["shots"]*item["runtime_per_preparation_ns"] for item in requests),
        data_exposure_cycles=sum(item["shots"]*item["code_cycles_per_preparation"] for item in requests),
        projected_physical_failure=math.fsum(item["shots"]*float.fromhex(item["expected_error_per_preparation_hex"]) for item in requests),
        macro_counts={key: sum(item["weighted_macro_counts"][key] for item in requests)
                      for key in ("readouts", "preparations", "decisions")})
    if (totals["classes"] != 54 or totals["parent_groups"] != 49 or totals["preparations"] != 82_032
            or totals["t_states"] != 182_646_240 or totals["serial_runtime_ns"] != 602_466_084_947_200
            or totals["macro_counts"] != accounting["macro_events_shot_weighted"]["candidate"]):
        raise ValueError("The fixed candidate native-input totals disagree with its source freeze")
    plan = dict(schema_version=1, status="planned_fixed_native_requests_not_executed", requests=requests,
        recipe_sha256=RECIPE_SHA256, candidate_manifest_sha256=MANIFEST_SHA256, validation_sha256=VALIDATION_SHA256,
        baseline_physical_dag_sha256=DAG_SHA256, source_pins=recipe["owned_source_sha256"],
        adapter_source_sha256=hashlib.sha256(_regular(Path(__file__))).hexdigest(), profile=recipe["profile"],
        fixed_machine=recipe["common_machine"], native_identity=identity, native_fingerprints=fingerprints,
        native_restoration_scope="exact archived evaluated fixed-arity cost records using original constructor; fresh graph; original ISA JSON/pipeline graph unavailable",
        raw_parent_allocation_records=raw_parents, unchanged_budget=recipe["unchanged_budget"], totals=totals,
        frozen_source_recipe=recipe, frozen_source_accounting=accounting,
        caps=dict(native_trace_queries=54, factory_frontier_queries=0, pipeline_candidates=0,
                  synthesis_calls=0, physical_wall_seconds=60, decoded_record_bytes=DECODED_CAP, bundle_bytes=BUNDLE_CAP),
        execution_enabled=False, approval_required="specific coordinator approval of this reviewed fixed adapter/plan",
        physical_result_scope="conditional serial model only; no hardware, geometry, decoder or optimizer-path guarantee")
    plan["factory_result_metadata_scope"] = factory_metadata_scope()
    plan["plan_sha256"] = _digest(plan)
    return plan


def _bound_plan(plan: dict, *, check_runtime=True):
    copied = deepcopy(plan)
    expected = copied.pop("plan_sha256", None)
    if expected != _digest(copied) or len(plan.get("requests", [])) != MAX_REQUESTS:
        raise ValueError("Fixed plan content or request count changed")
    if plan["adapter_source_sha256"] != hashlib.sha256(_regular(Path(__file__))).hexdigest():
        raise ValueError("The planned native adapter source changed")
    fingerprints = native_fingerprints() if check_runtime else dict(package=QDK_VERSION, files=dict(_NATIVE_PINS),
        scope="current adapter runtime pins; the seventh archive did not pin the native binary")
    if plan["recipe_sha256"] != RECIPE_SHA256 or plan["native_fingerprints"] != fingerprints:
        raise ValueError("Fixed native plan provenance changed")
    check_recorded_identity(plan["native_identity"])
    recipe, accounting = plan["frozen_source_recipe"], plan["frozen_source_accounting"]
    encoded = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if (hashlib.sha256(encoded(recipe)).hexdigest() != RECIPE_SHA256
            or hashlib.sha256(encoded(accounting)).hexdigest() != ACCOUNTING_SHA256
            or _canonical(plan["profile"]) != _canonical(recipe["profile"])
            or _canonical(plan["fixed_machine"]) != _canonical(recipe["common_machine"])
            or _canonical(plan["source_pins"]) != _canonical(recipe["owned_source_sha256"])
            or _canonical(plan["unchanged_budget"]) != _canonical(recipe["unchanged_budget"])):
        raise ValueError("Rehashed fixed source/machine/profile/allocations changed")
    if (_canonical(plan["caps"]) != _canonical(dict(native_trace_queries=54, factory_frontier_queries=0, pipeline_candidates=0,
        synthesis_calls=0, physical_wall_seconds=60, decoded_record_bytes=DECODED_CAP, bundle_bytes=BUNDLE_CAP))
            or plan["execution_enabled"] is not False
            or _digest(plan["raw_parent_allocation_records"]) != PARENT_ALLOCATION_SHA256):
        raise ValueError("The fixed Native execution caps changed")
    if _canonical(plan.get("factory_result_metadata_scope")) != _canonical(factory_metadata_scope()):
        raise ValueError("The planned captured-unvalidated factory metadata scope changed")
    if [row["class_id"] for row in plan["requests"]] != [row["class_id"] for row in recipe["classes"]]:
        raise ValueError("The original 54 individual class requests changed")
    costs = _module(Path(__file__).with_name("optimization_word_accounting.py"),
        recipe["owned_source_sha256"]["src/openquantum/optimization_word_accounting.py"], "bound_native_pinned_accounting")
    parents = {row["physical_class_id"]: row for row in plan["raw_parent_allocation_records"]}
    frozen_parents = {row["parent_physical_class_id"]: row for row in accounting["parent_groups"]}
    if (len(plan["raw_parent_allocation_records"]) != 49 or len(parents) != 49
            or len(frozen_parents) != 49 or set(parents) != set(frozen_parents)):
        raise ValueError("The 49 immutable original allocation parents changed")
    class_parent = {}
    for parent_id, parent in parents.items():
        frozen = frozen_parents[parent_id]
        if (parent["compiled_class_ids"] != frozen["source_class_ids"]
                or parent["shots"] != frozen["shots"] or parent["total_t_states"] != frozen["total_t_states"]
                or _canonical(parent["signature"]) != _canonical(frozen["baseline_signature"])):
            raise ValueError("Frozen parent membership, source signature or true weights changed")
        for class_id in parent["compiled_class_ids"]:
            if class_id in class_parent:
                raise ValueError("A source class was duplicated across immutable allocation parents")
            class_parent[class_id] = parent_id
    if set(class_parent) != {source["class_id"] for source in recipe["classes"]}:
        raise ValueError("Immutable parent memberships omit or add a saved source class")
    candidate_events = {parent_id: row["candidate_event_counts"] for parent_id, row in frozen_parents.items()}
    for request in plan["requests"]:
        material = dict(request)
        expected = material.pop("trace_input_sha256", None)
        if expected != _digest(material):
            raise ValueError("A fixed native input changed")
    for source, request in zip(recipe["classes"], plan["requests"]):
        if request["parent_physical_class_id"] != class_parent[source["class_id"]]:
            raise ValueError("Rehashed class request escaped its frozen original allocation parent")
        parent = parents[request["parent_physical_class_id"]]
        expected = _model_request(dict(plan=source, events=candidate_events[parent["physical_class_id"]]),
                                  parent, recipe["profile"], plan["native_identity"], costs)
        if _canonical(expected) != _canonical(request):
            raise ValueError("Rehashed class weights, source binding, base errors or native ledgers changed")
    for parent_id, parent in parents.items():
        members = [request for request in plan["requests"] if request["parent_physical_class_id"] == parent_id]
        if ([request["class_id"] for request in members] != parent["compiled_class_ids"]
                or sum(request["shots"] for request in members) != parent["shots"]
                or sum(request["total_t_states"] for request in members) != parent["total_t_states"]
                or sum(Fraction(request["physical_failure_allocation_exact"]) for request in members)
                    != Fraction(parent["physical_failure_allocation_exact"])
                or sum(Fraction(request["factory_availability_allocation_exact"]) for request in members)
                    != Fraction(parent["factory_availability_allocation_exact"])):
            raise ValueError("A frozen parent lost its exact class membership, weights or allocation sums")
        ceiling = sum(Fraction.from_float(float.fromhex(request["native_max_error_hex"]))*request["shots"]
                      for request in members)
        availability = sum(Fraction(request["factory_availability_failure_exact"]) for request in members)
        if (ceiling > Fraction.from_float(parent["physical_failure_budget"])
                or ceiling > Fraction(parent["physical_failure_allocation_exact"])
                or availability > Fraction(str(parent["factory_availability_budget"]))
                or availability > Fraction(parent["factory_availability_allocation_exact"])):
            raise ValueError("Class request ceilings exceed an unchanged original parent allocation")
    totals = dict(classes=len(plan["requests"]), parent_groups=len(parents),
        preparations=sum(row["shots"] for row in plan["requests"]),
        t_states=sum(row["total_t_states"] for row in plan["requests"]),
        serial_runtime_ns=sum(row["shots"]*row["runtime_per_preparation_ns"] for row in plan["requests"]),
        data_exposure_cycles=sum(row["shots"]*row["code_cycles_per_preparation"] for row in plan["requests"]),
        projected_physical_failure=math.fsum(row["shots"]*float.fromhex(row["expected_error_per_preparation_hex"])
                                             for row in plan["requests"]),
        macro_counts={key: sum(row["weighted_macro_counts"][key] for row in plan["requests"])
                      for key in ("readouts", "preparations", "decisions")})
    if _canonical(totals) != _canonical(plan["totals"]):
        raise ValueError("Rehashed global class, shot, exposure or model totals changed")


def evaluated_isa_records(isa) -> list:
    """Capture every used-arity cost value, without claiming opaque graph identity."""
    values = []
    for instruction in sorted(isa, key=lambda value: value.id):
        if instruction.arity == 1:
            values.append(instruction_record(instruction))
        elif instruction.arity is None:
            values.append(dict(id=int(instruction.id), encoding=int(instruction.encoding), arity=None,
                evaluated_arity=15, time={"integer": instruction.time(15)},
                space={"integer": instruction.space(15)},
                error_rate={"float_hex": float(instruction.error_rate(15)).hex()}))
        else:
            raise ValueError("Returned native ISA has an unexpected fixed arity")
    return values


def _expected_isa_records(request: dict, identity: dict) -> list:
    records = [deepcopy(identity["native_t_instruction"])]
    records += [dict(id=row["instruction_id"], encoding=1, arity=None, evaluated_arity=15,
                     time={"integer": row["time_ns"]}, space={"integer": 6_735},
                     error_rate={"float_hex": row["event_failure_model_hex"]})
                for row in request["macro_ledger_per_preparation"]]
    return sorted(records, key=lambda item: item["id"])


def _composition_record(value) -> dict:
    return dict(type=type(value).__name__, value=str(value))


def validate_restored_input(plan: dict, request: dict, enriched: dict) -> None:
    """Pure parent-side checks for the full input record before a durable grant.

    The pinned worker additionally checks the actual Native object's gate,
    resource and base-error fields against these records.  The parent retains
    the complete Native Trace JSON rather than treating its hash as the trace.
    """
    extras = {"native_trace_json", "native_trace_json_sha256", "native_isa_cost_records",
              "native_isa_cost_records_sha256", "error_composition", "native_factory_isa_sha256",
              "native_t_instruction_sha256", "native_compute_qubits", "native_gate_counts",
              "native_resource_states", "native_base_error_hex"}
    if set(enriched) != set(request) | extras or _canonical({key: enriched[key] for key in request}) != _canonical(request):
        raise ValueError("Restored native input does not bind the exact planned source request")
    trace_json = enriched["native_trace_json"]
    if (not isinstance(trace_json, str) or len(trace_json.encode()) > DECODED_CAP
            or hashlib.sha256(trace_json.encode()).hexdigest() != enriched["native_trace_json_sha256"]):
        raise ValueError("The full restored Native Trace JSON or byte hash changed")
    json.loads(trace_json)
    records = _expected_isa_records(request, plan["native_identity"])
    if (_canonical(enriched["native_isa_cost_records"]) != _canonical(records)
            or enriched["native_isa_cost_records_sha256"] != _digest(records)
            or enriched["native_factory_isa_sha256"] != ISA_SHA256 or enriched["native_t_instruction_sha256"] != T_SHA256):
        raise ValueError("Restored full used-arity native ISA cost identities changed")
    if (enriched["error_composition"] != {"type": "ErrorComposition", "value": "ErrorComposition.UnionBound"}
            or enriched["native_compute_qubits"] != 15
            or enriched["native_gate_counts"] != [dict(instruction_id=row["instruction_id"], count=row["count"])
                                                  for row in request["macro_ledger_per_preparation"]]
            or enriched["native_resource_states"] != [dict(instruction_id=1028, count=request["t_states_per_preparation"])]
            or enriched["native_base_error_hex"] != request["native_base_error_hex"]):
        raise ValueError("Restored Native trace fields or UnionBound composition changed")


def restore_native_request(plan: dict, request: dict) -> dict:
    """Construct checked Native ISA/Trace inputs; never invoke estimation."""
    _bound_plan(plan)
    if request not in plan["requests"]:
        raise ValueError("Native request is not a member of the fixed plan")
    from qdk.qre import Trace, LOGICAL, ErrorComposition, constant_function, linear_function
    restored = restore_recorded_factory_isa(plan["native_identity"])
    context = restored["context"]
    nodes = []
    trace = Trace(15)
    for row in request["macro_ledger_per_preparation"]:
        nodes.append(context.add_instruction(row["instruction_id"], encoding=LOGICAL, arity=None,
            time=constant_function(row["time_ns"]), space=linear_function(449),
            error_rate=constant_function(float.fromhex(row["event_failure_model_hex"]))))
        trace.add_block(repetitions=row["count"]).add_operation(row["instruction_id"], list(range(15)))
    nodes.append(restored["node"])
    isa = context.make_isa(*nodes)
    trace.increment_resource_state(1028, request["t_states_per_preparation"])
    trace.increment_base_error(float.fromhex(request["native_base_error_hex"]))
    if (trace.compute_qubits != 15 or trace.resource_states != {1028: request["t_states_per_preparation"]}
            or trace.gate_counts != {row["instruction_id"]: row["count"] for row in request["macro_ledger_per_preparation"]}
            or float(trace.base_error).hex() != request["native_base_error_hex"]
            or _digest(instruction_record(isa[1028])) != T_SHA256):
        raise ValueError("Native trace/ISA construction changed a checked input")
    isa_records = evaluated_isa_records(isa)
    if isa_records != _expected_isa_records(request, plan["native_identity"]):
        raise ValueError("Native ISA constructor changed a used-arity macro or factory cost")
    enriched = deepcopy(request)
    trace_json = trace.to_json()
    enriched["native_trace_json"] = trace_json
    enriched["native_trace_json_sha256"] = hashlib.sha256(trace_json.encode()).hexdigest()
    enriched["native_isa_cost_records"] = isa_records
    enriched["native_isa_cost_records_sha256"] = _digest(isa_records)
    enriched["error_composition"] = _composition_record(ErrorComposition.UnionBound)
    enriched["native_compute_qubits"] = trace.compute_qubits
    enriched["native_gate_counts"] = [dict(instruction_id=key, count=count) for key, count in sorted(trace.gate_counts.items())]
    enriched["native_resource_states"] = [dict(instruction_id=key, count=count) for key, count in sorted(trace.resource_states.items())]
    enriched["native_base_error_hex"] = float(trace.base_error).hex()
    enriched["native_factory_isa_sha256"] = ISA_SHA256
    enriched["native_t_instruction_sha256"] = T_SHA256
    validate_restored_input(plan, request, enriched)
    return dict(trace=trace, isa=isa, composition=ErrorComposition.UnionBound,
                max_error=float.fromhex(request["native_max_error_hex"]), request=enriched)


def _typed(value):
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "hex": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    return {"type": "unsupported", "python_type": type(value).__name__}


def _capture_result(result, request: dict) -> dict:
    """Capture public raw producer fields even when later validation fails."""
    raw = dict(status="producer_returned_before_validation", class_id=request["class_id"],
               trace_input_sha256=request["trace_input_sha256"], validated=False,
               factory_result_metadata_scope=factory_metadata_scope(),
               opaque_provenance_scope="not serialized; public cost/value fields only")
    if result is None:
        raw["result"] = {"type": "none"}
        return raw
    fields = {}
    for name in ("qubits", "runtime", "error"):
        try:
            fields[name] = _typed(getattr(result, name))
        except Exception as error:
            fields[name] = {"type": "access_error", "error_type": type(error).__name__}
    raw["result"] = fields
    try:
        factories = []
        for key, value in result.factories.items():
            factory = {"instruction_id": _typed(key)}
            for name in ("copies", "runs", "states", "error_rate"):
                try:
                    factory[name] = _typed(getattr(value, name))
                except Exception as error:
                    factory[name] = {"type": "access_error", "error_type": type(error).__name__}
            factories.append(factory)
        raw["factories"] = factories
    except Exception as error:
        raw["factories"] = {"type": "access_error", "error_type": type(error).__name__}
    try:
        raw["properties"] = [{"key": _typed(key), "value": _typed(value)}
                              for key, value in result.properties.items()]
    except Exception as error:
        raw["properties"] = {"type": "access_error", "error_type": type(error).__name__}
    try:
        raw["native_t_instruction"] = instruction_record(result.isa[1028])
        raw["native_factory_isa"] = [raw["native_t_instruction"]]
        raw["native_isa_cost_records"] = evaluated_isa_records(result.isa)
    except Exception as error:
        raw["native_t_instruction"] = {"type": "access_error", "error_type": type(error).__name__}
    try:
        raw["error_composition"] = _composition_record(result.error_composition)
    except Exception as error:
        raw["error_composition"] = {"type": "access_error", "error_type": type(error).__name__}
    return raw


def _validated_result(raw: dict, request: dict) -> dict:
    if (raw.get("status") != "producer_returned_before_validation" or raw.get("validated") is not False
            or raw.get("class_id") != request["class_id"]
            or raw.get("trace_input_sha256") != request["trace_input_sha256"]):
        raise ValueError("Saved producer receipt does not bind this exact native class request")
    if _canonical(raw.get("factory_result_metadata_scope")) != _canonical(factory_metadata_scope()):
        raise ValueError("Captured factory metadata must retain its explicit unvalidated scope")
    result = raw["result"]
    if (result.get("qubits", {}).get("type") != "int" or result.get("runtime", {}).get("type") != "int"
            or result.get("error", {}).get("type") != "float"):
        raise ValueError("Native producer did not return the required exact scalar fields")
    qubits, runtime = result["qubits"]["value"], result["runtime"]["value"]
    error = float.fromhex(result["error"]["hex"])
    expected = float.fromhex(request["expected_error_per_preparation_hex"])
    if (qubits != 243_358 or runtime != request["runtime_per_preparation_ns"] or not math.isfinite(error)
            or error < 0 or error > float.fromhex(request["native_max_error_hex"])
            or not math.isclose(error, expected, rel_tol=1e-10, abs_tol=1e-30)):
        raise ValueError("Native result disagrees with the fixed complete-ledger model/ceiling")
    if raw.get("error_composition") != request["error_composition"]:
        raise ValueError("Native result changed the required UnionBound composition")
    if raw.get("native_isa_cost_records") != request["native_isa_cost_records"]:
        raise ValueError("Returned full used-arity native ISA costs disagree with the checked input")
    factories = raw.get("factories")
    if (not isinstance(factories, list) or len(factories) != 1 or factories[0]["instruction_id"] != {"type": "int", "value": 1028}
            or factories[0]["copies"] != {"type": "int", "value": 1}
            or _digest(raw.get("native_t_instruction")) != T_SHA256 or _digest(raw.get("native_factory_isa")) != ISA_SHA256):
        raise ValueError("Native result changed the one-copy recorded factory/T cost identity")
    for name in ("runs", "states"):
        value = factories[0][name]
        if value.get("type") != "int" or value["value"] < 0:
            raise ValueError("Captured-unvalidated factory metadata failed its integer/nonnegative guard")
    rate = factories[0]["error_rate"]
    if (rate.get("type") != "float" or not math.isfinite(float.fromhex(rate["hex"]))
            or float.fromhex(rate["hex"]) < 0):
        raise ValueError("Captured-unvalidated factory metadata failed its finite/range guard")
    properties = raw.get("properties")
    if not isinstance(properties, list):
        raise ValueError("Native public properties could not be completely captured")
    for item in properties:
        key, value = item.get("key", {}), item.get("value", {})
        if key.get("type") != "int" or value.get("type") not in {"bool", "int", "float", "str"}:
            raise ValueError("Native public properties contain an unsupported or uncaptured value")
        if value["type"] == "float" and not math.isfinite(float.fromhex(value["hex"])):
            raise ValueError("Native public properties contain a nonfinite value")
    return dict(class_id=request["class_id"], parent_physical_class_id=request["parent_physical_class_id"],
        shots=request["shots"], runtime_per_preparation_ns=runtime, serial_runtime_all_preparations_ns=runtime*request["shots"],
        error_per_preparation_hex=error.hex(), full_physical_failure_exact=str(Fraction.from_float(error)*request["shots"]),
        factory_availability_failure_exact=request["factory_availability_failure_exact"],
        code_cycles_per_preparation=request["code_cycles_per_preparation"],
        macro_counts=request["weighted_macro_counts"], peak_physical_qubits=qubits,
        factory_result_metadata_scope=factory_metadata_scope(copies_validated=True),
        raw_receipt_sha256=_digest(raw), native_factory_isa_sha256=ISA_SHA256, native_t_instruction_sha256=T_SHA256)


def validate_saved_result(enriched_request: dict, raw_receipt: dict) -> dict:
    """Pure receipt re-derivation, with no QDK import or Native estimation.

    The caller first validates the complete enriched input using
    ``validate_restored_input(plan, planned_request, enriched_request)``.
    This method checks the post-callback input receipt and reconstructs exactly
    the same checked class result as the execution path.
    """
    expected_post = dict(request_sha256=_digest(enriched_request),
        trace_json_sha256=enriched_request["native_trace_json_sha256"],
        isa_cost_records_sha256=enriched_request["native_isa_cost_records_sha256"])
    if raw_receipt.get("post_callback_input") != expected_post:
        raise ValueError("Saved producer receipt reports a changed or uncaptured native input")
    return _validated_result(raw_receipt, enriched_request)


def validate_saved_execution(plan: dict, raw_receipts: list, enriched_inputs: list, result: dict) -> dict:
    """Rebuild a complete saved execution from raw receipts, without Native APIs.

    Source/plan identities are checked, but the reader does not import QDK or
    require the current machine's native binary merely to revalidate records.
    The production runner separately checks journal/clock/approval evidence.
    """
    _bound_plan(plan, check_runtime=False)
    if (not isinstance(raw_receipts, list) or not isinstance(enriched_inputs, list)
            or len(raw_receipts) != 54 or len(enriched_inputs) != 54
            or not isinstance(result, dict) or result.get("status") != "complete_injected_fixed_result"
            or result.get("plan_sha256") != plan["plan_sha256"]):
        raise ValueError("Saved execution does not contain the complete frozen 54-request run")
    if _canonical(result.get("factory_result_metadata_scope")) != _canonical(factory_metadata_scope(copies_validated=True)):
        raise ValueError("Saved result must disclose captured-unvalidated factory metadata")
    rebuilt = []
    for request, enriched, raw in zip(plan["requests"], enriched_inputs, raw_receipts):
        validate_restored_input(plan, request, enriched)
        rebuilt.append(validate_saved_result(enriched, raw))
    if _canonical(rebuilt) != _canonical(result.get("results")):
        raise ValueError("Saved class results disagree with their captured Native producer records")
    for parent in plan["raw_parent_allocation_records"]:
        members = [row for row in rebuilt if row["parent_physical_class_id"] == parent["physical_class_id"]]
        if ([row["class_id"] for row in members] != parent["compiled_class_ids"]
                or sum(row["shots"] for row in members) != parent["shots"]):
            raise ValueError("Saved class results changed an original parent membership or shot weight")
        physical = sum((Fraction(row["full_physical_failure_exact"]) for row in members), Fraction())
        availability = sum((Fraction(row["factory_availability_failure_exact"]) for row in members), Fraction())
        if (physical > Fraction.from_float(parent["physical_failure_budget"])
                or physical > Fraction(parent["physical_failure_allocation_exact"])
                or availability > Fraction(str(parent["factory_availability_budget"]))
                or availability > Fraction(parent["factory_availability_allocation_exact"])):
            raise ValueError("Saved Native producer results exceed an unchanged original parent budget")
    runtime = sum(row["serial_runtime_all_preparations_ns"] for row in rebuilt)
    physical = sum((Fraction(row["full_physical_failure_exact"]) for row in rebuilt), Fraction())
    availability = sum((Fraction(row["factory_availability_failure_exact"]) for row in rebuilt), Fraction())
    if (result.get("serial_runtime_ns") != runtime or result.get("whole_physical_failure_exact") != str(physical)
            or result.get("factory_availability_failure_exact") != str(availability)
            or any(result.get(name) != 0 for name in ("factory_frontier_queries", "pipeline_candidates", "synthesis_calls"))):
        raise ValueError("Saved execution aggregate totals or excluded-query counters changed")
    return dict(status="verified_saved_fixed_execution", classes=54, parent_groups=49,
        preparations=sum(row["shots"] for row in rebuilt), serial_runtime_ns=runtime,
        whole_physical_failure_exact=str(physical), factory_availability_failure_exact=str(availability),
        factory_result_metadata_scope=factory_metadata_scope(copies_validated=True),
        native_api_calls=0, validation_scope="raw public-field and used-arity cost records; journal/clock/approval verified by runner")


def execute_fixed_requests(plan: dict, *, trace_estimator=None, before_trace=None, after_trace=None,
                           audit: dict | None = None, execution_enabled=False,
                           native_boundary_validator=None) -> dict:
    """Invoke only an injected estimator after a durable permit, capture first.

    Source tests inject mocks.  A production callback requires independent
    coordinator approval and the external runner's cumulative deadline and
    permit journal; passing True here cannot authorize that approval itself.
    No automatic retry, regrouping or completed-result regeneration occurs.
    """
    if execution_enabled is not True:
        raise NativeExecutionDisabled("Fixed Native execution awaits specific coordinator approval")
    if not all(callable(value) for value in (trace_estimator, before_trace, after_trace)):
        raise ValueError("Estimator and durable before/after callbacks are mandatory")
    if native_boundary_validator is not None and not callable(native_boundary_validator):
        raise ValueError("The optional native-boundary validator must be callable")
    _bound_plan(plan)
    audit = {} if audit is None else audit
    if audit.get("execution_started") or any(audit.get(key, 0) for key in ("permitted", "confirmed_started", "returned", "validated")):
        raise ValueError("This execution audit cannot be reset or automatically retried")
    audit.update(execution_started=True, permitted=0, confirmed_started=0, returned=0, validated=0,
        counter_scope="injected callback entry/return; actual Native boundary markers belong to the authorized runner")
    started = time.perf_counter()
    results = []
    for request in plan["requests"]:
        if audit["confirmed_started"] >= MAX_REQUESTS or time.perf_counter() - started >= 60:
            raise ValueError("The fixed Native call/deadline cap was reached; no retry permitted")
        payload = restore_native_request(plan, request)
        if native_boundary_validator is not None:
            if native_boundary_validator(payload) is False:
                raise NativeExecutionDisabled("The exact production Native boundary was refused before its permit")
        enriched = payload["request"]
        expected_request = deepcopy(enriched)
        before_record = deepcopy(expected_request)
        permit = before_trace(before_record)
        if permit is None or permit is False:
            raise NativeExecutionDisabled("The durable parent permit was not granted")
        audit["permitted"] += 1
        if _canonical(before_record) != _canonical(expected_request):
            raise ValueError("A durable-permit callback changed the checked native input")
        if time.perf_counter() - started >= 60:
            raise ValueError("The fixed deadline expired after a permit and before a native call")
        audit["confirmed_started"] += 1
        try:
            producer = trace_estimator(payload)
        except Exception as error:
            after_trace(deepcopy(expected_request), dict(status="native_callback_exception", class_id=expected_request["class_id"],
                trace_input_sha256=expected_request["trace_input_sha256"], validated=False,
                factory_result_metadata_scope=factory_metadata_scope(), error_type=type(error).__name__))
            raise
        audit["returned"] += 1
        raw = _capture_result(producer, expected_request)
        try:
            raw["post_callback_input"] = dict(request_sha256=_digest(payload["request"]),
                trace_json_sha256=hashlib.sha256(payload["trace"].to_json().encode()).hexdigest(),
                isa_cost_records_sha256=_digest(evaluated_isa_records(payload["isa"])))
        except Exception as error:
            raw["post_callback_input"] = {"access_error_type": type(error).__name__}
        # Durable producer receipt is mandatory BEFORE any validity judgement.
        after_request, after_record = deepcopy(expected_request), deepcopy(raw)
        after_trace(after_request, after_record)
        if _canonical(after_request) != _canonical(expected_request) or _canonical(after_record) != _canonical(raw):
            raise ValueError("A receipt callback changed the producer record or its native input")
        if raw["post_callback_input"] != dict(request_sha256=_digest(expected_request),
            trace_json_sha256=expected_request["native_trace_json_sha256"],
            isa_cost_records_sha256=expected_request["native_isa_cost_records_sha256"]):
            raise ValueError("An estimator callback changed a checked native input")
        validated = _validated_result(raw, expected_request)
        audit["validated"] += 1
        results.append(validated)
    for parent in plan["raw_parent_allocation_records"]:
        members = [row for row in results if row["parent_physical_class_id"] == parent["physical_class_id"]]
        physical = sum((Fraction(row["full_physical_failure_exact"]) for row in members), Fraction())
        factory = sum((Fraction(row["factory_availability_failure_exact"]) for row in members), Fraction())
        if (physical > Fraction.from_float(parent["physical_failure_budget"])
                or physical > Fraction(parent["physical_failure_allocation_exact"])
                or factory > Fraction(str(parent["factory_availability_budget"]))
                or factory > Fraction(parent["factory_availability_allocation_exact"])):
            raise ValueError("Validated class results exceeded an unchanged original parent allowance")
    return dict(status="complete_injected_fixed_result", plan_sha256=plan["plan_sha256"], results=results,
        audit=deepcopy(audit), serial_runtime_ns=sum(row["serial_runtime_all_preparations_ns"] for row in results),
        whole_physical_failure_exact=str(sum((Fraction(row["full_physical_failure_exact"]) for row in results), Fraction())),
        factory_availability_failure_exact=str(sum((Fraction(row["factory_availability_failure_exact"]) for row in results), Fraction())),
        estimator_provenance="injected callback; native execution provenance must be supplied by the authorized runner",
        factory_result_metadata_scope=factory_metadata_scope(copies_validated=True),
        factory_frontier_queries=0, pipeline_candidates=0, synthesis_calls=0)
