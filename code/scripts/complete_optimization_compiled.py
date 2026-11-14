#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Complete the frozen QAOA replay with one declared physical-only recovery.

The original synthesis producer, recipe, words, checkpoints and failed physical
attempt remain immutable. Recovery changes representation and explicitly amends
two cumulative native-query ceilings; it never changes the quantum workload.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import os
import pickle
import stat
import subprocess
import sys
import tempfile
from time import perf_counter, process_time

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
CHECKOUT = SCRIPT == PROJECT / "scripts/complete_optimization_compiled.py"
ROOT = PROJECT if CHECKOUT else SCRIPT.parent
DEFAULT_OUTPUT = ROOT / "assets/generated/optimization/compiled-replay" if CHECKOUT else ROOT
CODEC_PATH = ROOT / "src/openquantum/optimization_compiled_codec.py" if CHECKOUT else ROOT / "optimization_compiled_codec.py"
OLD_CORE_SHA = "5802330c76ea71ec46fcd207018b805a2c1da29a91f9b1ee6deb5defcd36f5fc"
OLD_GENERATOR_SHA = "6dafe1aa49c656a732b2d534cefb7aee99a68b215dc3322c769a20bf358e0ecb"
FIRST_WALL = 11.716274958002032
FIRST_COUNTS = dict(top_level_calls=49, factory_frontier_queries=1176,
                    factory_frontier_candidates=12348, factory_pipeline_candidates=816,
                    native_trace_queries=6174)
CUMULATIVE_CAPS = dict(top_level_calls=128, factory_frontier_queries=3072,
                       factory_frontier_candidates=52224, factory_pipeline_candidates=52224,
                       native_trace_queries=52224)
OLD_CAPS = dict(top_level_calls=64, factory_frontier_queries=1536,
                factory_frontier_candidates=52224, factory_pipeline_candidates=52224,
                native_trace_queries=52224)
DECODED_CAP = 64 * 1024 * 1024
PRIVATE_REFERENCE_CAP = 256 * 1024 * 1024
ORIGINAL_INVENTORY_SHA = "b247290b5c333d94726f2dc7cb935741fae6fd497223f953612fda692cd3446a"
ORIGINAL_PREFIX = "assets/generated/optimization/compiled-replay/"


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Recovery producer must be a regular file")
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("The original frozen compilation producer changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def legacy_modules(output):
    output = Path(output).resolve()
    legacy = load_module(output / "generate_optimization_compiled.py", "completion_frozen_generator", OLD_GENERATOR_SHA)
    core = load_module(output / "optimization_compiled.py", "completion_frozen_core", OLD_CORE_SHA)
    inputs = core.load_compiled_inputs(output)
    return legacy, core, inputs


def load_codec():
    return load_module(CODEC_PATH, "completion_lossless_codec")


def bind_completion_sources(store,frozen):
    expected=frozen["identity"]["source_sha256"]
    for name,path in (("complete_optimization_compiled.py",SCRIPT),("optimization_compiled_codec.py",CODEC_PATH)):
        snapshot=store.target(name)
        if (path.is_symlink() or snapshot.is_symlink() or not path.is_file() or not snapshot.is_file()
                or sha(path)!=expected[name] or sha(snapshot)!=expected[name]):
            raise ValueError("Frozen completion producer changed before decoding")


def legacy_inventory(legacy, store):
    return {name: row["sha256"] for name, row in store.inventory().items()}


def verify_preserved(store, inventory):
    for relative, expected in inventory.items():
        path = store.target(relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("An original attempt artifact changed during recovery")


def verify_original_anchor(inventory):
    names = {ORIGINAL_PREFIX + name: digest for name, digest in inventory.items()}
    if len(names) != 281 or hashlib.sha256(encoded(dict(sorted(names.items())))).hexdigest() != ORIGINAL_INVENTORY_SHA:
        raise ValueError("Original first-attempt artifact inventory differs from its preserved anchor")


def first_attempt(store):
    frozen = json.loads(store.target("freeze.json").read_text())
    timing = json.loads(store.target("timing/physical.json").read_text())
    failure = json.loads(store.target("checkpoints/failures/physical-workload.json").read_text())
    audit = failure["failure"]["physical_query_audit"]
    if (timing["identity"] != frozen["identity"] or failure["identity"] != frozen["identity"]
            or timing["status"] != "incomplete" or not math.isclose(timing["total_wall_seconds"], FIRST_WALL, abs_tol=1e-12)
            or audit["status"] != "complete" or audit["counts"] != FIRST_COUNTS
            or failure["failure"]["partial_native_work_unknown"] is not False
            or failure["failure"]["error_type"] != "DecodedByteCapOrSerializationFailure"):
        raise ValueError("Recovery requires the exact measured representation-limit attempt")
    return frozen, timing, failure


def recovery_recipe(store, old_recipe):
    frozen, timing, failure = first_attempt(store)
    effective = {key: min(OLD_CAPS[key], CUMULATIVE_CAPS[key] - FIRST_COUNTS[key]) for key in OLD_CAPS}
    return dict(schema_version=1, recipe_id="qaoa-lossless-physical-completion-v1",
        original_identity=frozen["identity"], original_recipe_sha256=sha(store.target("recipe.json")),
        original_model_recipe=old_recipe, first_physical_timing_sha256=sha(store.target("timing/physical.json")),
        original_failure_sha256=sha(store.target("checkpoints/failures/physical-workload.json")),
        first_known_query_counts=FIRST_COUNTS, first_physical_wall_seconds=FIRST_WALL,
        maximum_additional_physical_passes=1, cumulative_query_caps=CUMULATIVE_CAPS,
        second_attempt_effective_caps=effective, cumulative_physical_wall_cap_seconds=120,
        second_remaining_wall_seconds=120-FIRST_WALL, decoded_transport_bytes=DECODED_CAP,
        private_producer_reference_bytes=PRIVATE_REFERENCE_CAP,
        codec_limits=asdict(load_codec().CodecLimits()),
        routed_qasm_metadata_restoration=dict(known_ops_and_phase_only=True,restores_physical_wire_export_marker=True,
            gates_measurement_maps_global_phases_unchanged=True,all_54_exact_qasm_byte_compares=True,
            original_sabre_internal_layout_recreated=False,physical_geometry_certificate=False),
        public_artifact_bytes=16*1024*1024, representation="lossless content-addressed JSON DAG with shared decoded subtrees",
        scientific_change=False, synthesis_or_optimizer_repeated=False, quantum_preparations=82032,
        quantum_preparations_scope="one fixed recorded workload; classical physical estimation repetition does not double it",
        scope="first complete point/query records were not retained; no recovered-hash equivalence to that missing result is claimed")


def completion_freeze(output):
    legacy, core, inputs = legacy_modules(output)
    codec = load_codec()
    store = legacy.ArtifactStore(legacy.validate_output(Path(output)), 16*1024*1024)
    existing = store.target("completion-freeze.json")
    old_recipe = core.compiled_recipe(inputs)
    recipe = recovery_recipe(store, old_recipe)
    sources = {"complete_optimization_compiled.py": SCRIPT.read_bytes(),
               "optimization_compiled_codec.py": CODEC_PATH.read_bytes()}
    if existing.exists():
        saved = json.loads(existing.read_text())
        inventory = saved["original_artifact_inventory"]
    else:
        inventory = legacy_inventory(legacy, store)
    verify_preserved(store, inventory)
    verify_original_anchor(inventory)
    identity = dict(recipe_sha256=hashlib.sha256(encoded(recipe)).hexdigest(),
                    source_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                    original_identity=recipe["original_identity"])
    amendment = dict(schema_version=1, type="explicit_computation_only_recovery_amendment",
        original_first_attempt_commit="f17e024", original_identity=recipe["original_identity"],
        original_failure_sha256=recipe["original_failure_sha256"],
        changes={"cumulative_top_level_calls": [64,128], "cumulative_frontier_queries": [1536,3072]},
        unchanged_cumulative_caps={key: 52224 for key in ("factory_frontier_candidates", "factory_pipeline_candidates", "native_trace_queries")},
        known_first_query_counts=FIRST_COUNTS, second_attempt_effective_caps=recipe["second_attempt_effective_caps"],
        cumulative_physical_wall_seconds=120, prior_physical_wall_seconds=FIRST_WALL,
        maximum_additional_physical_passes=1, source_angles_shots_model_failure_budgets_unchanged=True,
        full_first_result_unavailable=True, no_claim_of_equality_to_missing_first_physical_result=True)
    store.json("completion-amendment.json", amendment, immutable=True)
    store.json("completion-recipe.json", recipe, immutable=True)
    for name, raw in sources.items():
        store.write(name, raw, immutable=True)
    value = dict(schema_version=1, status="completion_recipe_and_producers_frozen_before_physical_recovery",
        identity=identity, original_artifact_inventory=inventory,
        captured_at_utc=datetime.now(timezone.utc).isoformat(), native_or_synthesis_executed=False)
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved["identity"] != identity:
            raise ValueError("Frozen recovery producer/recipe changed")
        value = saved
    else:
        store.json("completion-freeze.json", value, immutable=True)
    return dict(status=value["status"], identity=identity,
                original_files_preserved=len(inventory), native_or_synthesis_executed=False)


def completion_provenance(legacy,store):
    sources=legacy.producer_sources(store.target("optimization_compiled.py").read_bytes())
    sources.update({"complete_optimization_compiled.py":SCRIPT.read_bytes(),"optimization_compiled_codec.py":CODEC_PATH.read_bytes()})
    value=legacy.provenance(sources)
    value["invocation"]="python scripts/complete_optimization_compiled.py" if CHECKOUT else "python complete_optimization_compiled.py"
    if CHECKOUT and ((ROOT/".git").is_file() or (ROOT/".git").is_dir()):
        environment={key:item for key,item in os.environ.items() if key not in {"GIT_DIR","GIT_WORK_TREE","GIT_COMMON_DIR","GIT_INDEX_FILE"}}
        result=subprocess.run(["git","--no-optional-locks","-C",str(ROOT),"rev-parse","--show-toplevel"],env=environment,capture_output=True,text=True,timeout=5)
        if result.returncode==0 and Path(result.stdout.strip()).resolve()==ROOT:
            revision=subprocess.run(["git","--no-optional-locks","-C",str(ROOT),"rev-parse","HEAD"],env=environment,capture_output=True,text=True,timeout=5)
            if revision.returncode==0:
                value["source_commit_at_generation"]=revision.stdout.strip()
    return value


def load_compiled_classes(legacy, core, store, inputs, original_identity):
    classes = [legacy.load_checkpoint(store, "synthesis", c["class_id"], original_identity)
               for c in core.compiled_source_classes(inputs, core.compiled_recipe(inputs))]
    if len(classes) != 54:
        raise ValueError("Recovery must reuse all 54 actual compiled checkpoints")
    return classes


@contextmanager
def routed_export_marker(core,compiled):
    """Rehydrate only the physical-$wire export marker lost by primitive JSON.

    This leaves every operation, wire, classical output and phase unchanged.
    The identity marker is not the original internal SABRE layout object.
    """
    from qiskit import qasm3
    from qiskit.transpiler import Layout,TranspileLayout
    original=core.circuit_from_operations
    def key(operations,n,phase):
        return (json.dumps(operations,separators=(",",":"),allow_nan=False),n,float(phase).hex())
    markers={}
    def mark(circuit):
        circuit._layout=TranspileLayout(initial_layout=Layout({bit:index for index,bit in enumerate(circuit.qubits)}),
            input_qubit_mapping={bit:index for index,bit in enumerate(circuit.qubits)},final_layout=None,
            _input_qubit_count=len(circuit.qubits),_output_qubit_list=circuit.qubits)
        return circuit
    for item in compiled:
        for stage in ("source","routed","clifford_t"):
            operations,phase=item[stage+"_operations"],item[stage+"_global_phase"]
            circuit=original(operations,4,phase)
            physical="$" in item[stage+"_qasm"]
            signature=key(operations,4,phase)
            if signature in markers and markers[signature]!=physical:
                raise ValueError("One operation-stream/phase has ambiguous physical-wire export metadata")
            markers[signature]=physical
            if physical:
                if stage!="routed":
                    raise ValueError("Only a known routed stream may restore physical-wire export metadata")
                mark(circuit)
            if qasm3.dumps(circuit)!=item[stage+"_qasm"]:
                raise ValueError("Primitive words differ from recorded QASM after marker-only restoration")
    def reconstructed(operations,n,phase=0.0):
        circuit=original(operations,n,phase)
        if markers.get(key(operations,n,phase),False):
            mark(circuit)
        return circuit
    core.circuit_from_operations=reconstructed
    try:
        yield
    finally:
        core.circuit_from_operations=original


def typed_json_graph(value):
    """Validate only producer-created built-in JSON values before private pickle."""
    seen, active = {}, set()
    def visit(item, depth):
        if depth > 128:
            raise ValueError("Private producer graph exceeds its declared depth")
        kind = type(item)
        if kind in (dict, list):
            key=id(item)
            if key in active:
                raise ValueError("Private producer graph is cyclic")
            if key in seen:
                if depth+seen[key]>128:
                    raise ValueError("Private producer graph exceeds its declared depth")
                return seen[key]
            active.add(key)
            if len(seen) > 500000:
                raise ValueError("Private producer graph exceeds its node bound")
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("Private producer graph has a non-JSON key")
                children=item.values()
            else:
                children=item
            height=1+max((visit(child,depth+1) for child in children),default=-1)
            active.remove(key)
            seen[key]=height
            return height
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("Private producer graph has a nonfinite scalar")
        elif kind not in (str, int, bool, type(None)):
            raise ValueError("Private producer graph is not a built-in JSON value graph")
        return 0
    visit(value,0)


def preserve_private_reference(physical, job):
    """Private producer-owned backup; never a public or user-provided pickle."""
    typed_json_graph(physical)
    raw = pickle.dumps(physical, protocol=5)
    if len(raw) > PRIVATE_REFERENCE_CAP:
        raise ValueError("Private producer reference exceeds its frozen byte bound")
    path = Path(job) / "producer-physical-reference.pickle"
    with path.open("xb") as handle:
        handle.write(raw)
    return dict(file=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def trusted_private_reference(job, record):
    """Read only the hash-pinned file just returned by our own worker."""
    path, job = Path(record["file"]), Path(job).resolve()
    if path.parent.resolve() != job or path.name != "producer-physical-reference.pickle" or path.is_symlink():
        raise ValueError("Private reference is not in the producer-owned job directory")
    if (type(record["bytes"]) is not int or not 0 < record["bytes"] <= PRIVATE_REFERENCE_CAP
            or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size != record["bytes"]):
        raise ValueError("Private producer reference is not a bounded regular file")
    with path.open("rb") as handle:
        raw=handle.read(PRIVATE_REFERENCE_CAP+1)
    if len(raw)!=record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Private producer reference byte hash changed")
    value = pickle.loads(raw)
    typed_json_graph(value)
    return value


def physical_worker(input_path, output_path):
    codec = load_codec()
    with input_path.open("rb") as handle:
        with gzip.GzipFile(fileobj=handle) as compressed:
            raw = compressed.read(DECODED_CAP + 1)
    if len(raw) > DECODED_CAP:
        raise ValueError("Private recovery input exceeds decoded transport bound")
    payload = json.loads(raw)
    output = Path(payload["output"]).resolve()
    legacy, core, inputs = legacy_modules(output)
    store = legacy.ArtifactStore(output, 16*1024*1024)
    frozen = json.loads(store.target("completion-freeze.json").read_text())
    verify_original_anchor(frozen["original_artifact_inventory"])
    verify_preserved(store, frozen["original_artifact_inventory"])
    recipe = recovery_recipe(store, core.compiled_recipe(inputs))
    if payload["recipe"] != recipe:
        raise ValueError("Recovery worker received a changed recipe")
    began, cpu_began = perf_counter(), process_time()
    physical = None
    private = None
    try:
        compiled = load_compiled_classes(legacy, core, store, inputs, recipe["original_identity"])
        with routed_export_marker(core,compiled):
            memory_classes = core.compiled_physical_classes(compiled, recipe["original_model_recipe"])
        backend = core._producers()[1]
        caps = core.SeventhPhysicalCaps(**recipe["second_attempt_effective_caps"], wall_seconds=payload["remaining_wall_seconds"])
        t_states = sum(c["shots"]*sum(row[0] in ("t","tdg") for row in c["clifford_t_operations"]) for c in compiled)
        physical = core.run_seventh_physical_workload(memory_classes, backend=backend,
            cache_dir=Path(payload["cache_dir"]), caps=caps,
            expected_total_preparations=82032, expected_total_t_states=t_states)
        private = preserve_private_reference(physical, input_path.parent)
        packed = codec.pack(physical)
        restored = codec.unpack(packed)
        if not codec.equivalent(physical, restored):
            raise ValueError("Lossless physical DAG roundtrip differs from the second live native result")
        value = dict(status="completed", physical_packed=packed,
                     roundtrip_exact_to_second_live_value=True, codec_statistics=codec.statistics(packed),
                     private_reference=private,
                     query_audit={key:physical["query_audit"][key] for key in ("status","caps","counts","wall_seconds")})
    except Exception as error:
        audit = getattr(error,"audit",getattr(error,"seventh_physical_audit",None))
        if audit is None and physical is not None:
            audit = physical["query_audit"]
        value = dict(status="failed_incomplete", error_type=type(error).__name__,
            error_message=legacy.public_error_text(str(error)),
            query_audit=({key:audit[key] for key in ("status","caps","counts","wall_seconds") if key in audit} if isinstance(audit,dict) else None),
            private_reference=private, automatic_additional_physical_pass=False)
    value.update(worker_wall_seconds=perf_counter()-began, worker_process_cpu_seconds=process_time()-cpu_began)
    raw = json.dumps(value,separators=(",",":"),allow_nan=False).encode()
    if len(raw) > DECODED_CAP:
        value = dict(status="failed_incomplete",error_type="PackedTransportLimit",error_message="Packed worker transport exceeds frozen64MiB bound",
                     query_audit={key:physical["query_audit"][key] for key in ("status","caps","counts","wall_seconds")} if physical is not None else None,
                     private_reference=private,worker_wall_seconds=perf_counter()-began,worker_process_cpu_seconds=process_time()-cpu_began,
                     automatic_additional_physical_pass=False)
        raw = json.dumps(value,separators=(",",":"),allow_nan=False).encode()
    output_path.write_bytes(gzip.compress(raw,mtime=0))


def load_completed_replay(output, *, validate=True):
    legacy, core, inputs = legacy_modules(output)
    store = legacy.ArtifactStore(legacy.validate_output(Path(output)),16*1024*1024)
    frozen=json.loads(store.target("completion-freeze.json").read_text())
    bind_completion_sources(store,frozen)
    codec = load_codec()
    saved = legacy.compressed_record(store.target("completed-replay.json.gz"))
    timing=json.loads(store.target("completion-timing.json").read_text())
    if (saved["identity"]!=frozen["identity"] or timing["identity"]!=frozen["identity"] or timing["status"]!="complete"
            or not math.isfinite(timing["cumulative_wall_seconds"]) or timing["cumulative_wall_seconds"]>120
            or not math.isclose(timing["first_wall_seconds"],FIRST_WALL,abs_tol=1e-12)
            or not math.isclose(timing["cumulative_wall_seconds"],FIRST_WALL+timing["recovery_wall_seconds"],abs_tol=1e-9)):
        raise ValueError("Completed recovery lacks its matching bounded timing/freeze")
    verify_original_anchor(frozen["original_artifact_inventory"])
    verify_preserved(store,frozen["original_artifact_inventory"])
    physical_ref = saved["physical_storage"]
    path = store.target(physical_ref["file"])
    if sha(path) != physical_ref["sha256"]:
        raise ValueError("Packed physical record bytes changed")
    packed_raw = legacy.decoded_gzip(path.read_bytes(),DECODED_CAP)
    physical = codec.loads(packed_raw)
    counts=physical["query_audit"]["counts"]
    cumulative={key:FIRST_COUNTS[key]+counts.get(key,0) for key in FIRST_COUNTS}
    if (cumulative!=saved["recovery"]["cumulative_query_counts"]
            or any(cumulative[key]>CUMULATIVE_CAPS[key] for key in cumulative)
            or counts!=timing["second_query_counts"]):
        raise ValueError("Completed native query counts do not match the cumulative amended caps")
    classes = []
    for ref in saved["source_class_checkpoints"]:
        path = store.target(ref["file"])
        if sha(path) != ref["sha256"]:
            raise ValueError("Original compiled checkpoint changed")
        classes.append(legacy.load_checkpoint(store,"synthesis",ref["class_id"],saved["original_identity"]))
    if [c["class_id"] for c in classes]!=[c["class_id"] for c in core.compiled_source_classes(inputs,core.compiled_recipe(inputs))]:
        raise ValueError("Packed replay omitted or reordered immutable source checkpoints")
    replay = dict(saved["aggregate_metadata"],physical=physical,source_classes=classes)
    if validate:
        with routed_export_marker(core,classes):
            core.validate_compiled_record(replay)
    return replay


def complete(output, cache, *, resume=False, verify_only=False):
    started, cpu_started = perf_counter(),process_time()
    legacy, core, inputs = legacy_modules(output)
    output=legacy.validate_output(Path(output),Path(cache))
    cache=Path(cache).resolve()
    if cache.is_relative_to(output) or output.is_relative_to(cache) or (CHECKOUT and cache.is_relative_to(ROOT)):
        raise ValueError("Recovery cache must remain private and outside checkout/output")
    store=legacy.ArtifactStore(output,16*1024*1024)
    frozen=json.loads(store.target("completion-freeze.json").read_text())
    bind_completion_sources(store,frozen)
    codec = load_codec()
    identity=frozen["identity"]
    verify_original_anchor(frozen["original_artifact_inventory"])
    verify_preserved(store,frozen["original_artifact_inventory"])
    recipe=recovery_recipe(store,core.compiled_recipe(inputs))
    if json.loads(store.target("completion-recipe.json").read_text())!=recipe:
        raise ValueError("Frozen completion recipe changed")
    raw_path=store.target("completed-replay.json.gz")
    timing_path=store.target("completion-timing.json")
    if raw_path.exists():
        if not (resume or verify_only):
            raise ValueError("Completed recovery exists; use --resume or --verify-only")
        replay=load_completed_replay(output)
    else:
        if verify_only:
            raise ValueError("Saved complete physical DAG required for verification-only")
        if timing_path.exists():
            raise ValueError("Recovery already attempted; no additional native physical pass is allowed")
        legacy.ensure_committed_freeze(store)
        if CHECKOUT and (ROOT/".git").exists() and output.is_relative_to(ROOT):
            environment={key:value for key,value in os.environ.items() if key not in {"GIT_DIR","GIT_WORK_TREE","GIT_COMMON_DIR","GIT_INDEX_FILE"}}
            top=subprocess.run(["git","--no-optional-locks","-C",str(ROOT),"rev-parse","--show-toplevel"],env=environment,capture_output=True,text=True,timeout=5)
            if top.returncode or Path(top.stdout.strip()).resolve()!=ROOT:
                raise ValueError("Recovery Git provenance does not identify this exact checkout")
            for path in (SCRIPT,CODEC_PATH,store.target("completion-recipe.json"),store.target("completion-freeze.json"),store.target("completion-amendment.json")):
                result=subprocess.run(["git","--no-optional-locks","-C",str(ROOT),"show",f"HEAD:{path.relative_to(ROOT)}"],env=environment,capture_output=True,timeout=5)
                if result.returncode or result.stdout!=path.read_bytes():
                    raise ValueError("Commit the new completion recipe/amendment/producers before native recovery")
        store.json("completion-timing.json",dict(status="running",identity=identity,first_wall_seconds=FIRST_WALL,prior_query_counts=FIRST_COUNTS))
        compiled=load_compiled_classes(legacy,core,store,inputs,recipe["original_identity"])
        job=Path(tempfile.mkdtemp(prefix="optimization-compiled-recovery-job-")).resolve()
        input_path,output_path=job/"input.json.gz",job/"output.json.gz"
        remaining=120-FIRST_WALL-(perf_counter()-started)
        if remaining<=0:
            raise ValueError("Prior work and validation exhausted cumulative physical wall allowance")
        payload=dict(output=str(output),cache_dir=str(cache),recipe=recipe,remaining_wall_seconds=remaining)
        input_path.write_bytes(gzip.compress(json.dumps(payload,separators=(",",":"),allow_nan=False).encode(),mtime=0))
        error=None
        result_value=None
        try:
            environment=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
            with (job/"private-process.log").open("wb") as log:
                result=subprocess.run([sys.executable,str(SCRIPT),"--worker-input",str(input_path),"--worker-output",str(output_path)],
                                      cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            if result.returncode or not output_path.is_file():
                raise RuntimeError("Physical recovery worker failed without structured completion")
            result_value=legacy.compressed_record(output_path)
            if result_value["status"]!="completed":
                raise RuntimeError(result_value.get("error_message","Physical recovery remained incomplete"))
            packed=result_value["physical_packed"]
            physical=codec.unpack(packed)
            reference=trusted_private_reference(job,result_value["private_reference"])
            if not codec.equivalent(reference,physical):
                raise ValueError("Packed physical result differs from our second producer-owned live reference")
            second_counts=physical["query_audit"]["counts"]
            total_counts={key:FIRST_COUNTS[key]+second_counts.get(key,0) for key in FIRST_COUNTS}
            if any(total_counts[key]>CUMULATIVE_CAPS[key] for key in total_counts):
                raise ValueError("Cumulative first+second physical query cap exceeded")
            with routed_export_marker(core,compiled):
                replay=core.compiled_aggregate(inputs,recipe["original_model_recipe"],compiled,physical)
            packed_bytes=codec.dumps(packed)
            store.write("physical-dag.json.gz",gzip.compress(packed_bytes,mtime=0),immutable=True)
            metadata={key:value for key,value in replay.items() if key not in ("physical","source_classes")}
            references=[dict(class_id=c["class_id"],file=f"checkpoints/synthesis/{c['class_id']}.json.gz",
                sha256=sha(store.target(f"checkpoints/synthesis/{c['class_id']}.json.gz"))) for c in compiled]
            saved=dict(schema_version=1,status="completed_lossless_physical_recovery",identity=identity,original_identity=recipe["original_identity"],
                provenance=completion_provenance(legacy,store),
                aggregate_metadata=metadata,source_class_checkpoints=references,
                physical_storage=dict(encoding="json-content-dag-v1",file="physical-dag.json.gz",sha256=sha(store.target("physical-dag.json.gz"))),
                recovery=dict(first_query_counts=FIRST_COUNTS,second_query_counts=second_counts,cumulative_query_counts=total_counts,
                    first_wall_seconds=FIRST_WALL,second_worker_wall_seconds=result_value["worker_wall_seconds"],
                    second_worker_cpu_seconds=result_value["worker_process_cpu_seconds"],
                    roundtrip_exact_to_second_live_value=True,first_complete_result_unavailable=True,
                    quantum_preparations_counted_once=82032,codec_statistics=codec.statistics(packed)))
            store.json_gzip("completed-replay.json.gz",saved,immutable=True)
            verify_preserved(store,frozen["original_artifact_inventory"])
            if FIRST_WALL+perf_counter()-started>120:
                raise ValueError("Recovery packaging/verification exceeded cumulative physical wall cap")
        except Exception as caught:
            error=caught
        wall=perf_counter()-started
        timing=dict(schema_version=1,status="complete" if error is None else "incomplete",identity=identity,
            first_wall_seconds=FIRST_WALL,recovery_wall_seconds=wall,cumulative_wall_seconds=FIRST_WALL+wall,wall_cap_seconds=120,
            parent_cpu_seconds=process_time()-cpu_started,first_query_counts=FIRST_COUNTS,
            second_query_counts=(result_value.get("query_audit",{}).get("counts") if result_value and isinstance(result_value.get("query_audit"),dict) else None),
            error_type=type(error).__name__ if error else None,
            scope="prior physical attempt plus one recovery worker, codec verification and writes; later export-only work separate")
        store.json("completion-timing.json",timing)
        if FIRST_WALL+perf_counter()-started>120 and error is None:
            error=ValueError("Final timing publication exceeded cumulative physical wall cap")
            timing.update(status="incomplete",error_type="ValueError")
            store.json("completion-timing.json",timing)
        if error is not None:
            failure=result_value or dict(status="failed_incomplete",error_type=type(error).__name__,error_message=legacy.public_error_text(str(error)))
            failure={key:value for key,value in failure.items() if key not in ("physical_packed","private_reference")}
            store.json("completion-failure.json",dict(identity=identity,failure=failure,no_additional_native_pass=True),immutable=True)
            raise error
    return export_completed(output,replay)


def export_completed(output,replay):
    legacy,core,inputs=legacy_modules(output)
    store=legacy.ArtifactStore(legacy.validate_output(Path(output)),16*1024*1024)
    frozen=json.loads(store.target("completion-freeze.json").read_text())
    saved=legacy.compressed_record(store.target("completed-replay.json.gz"))
    classes=[legacy.export_class(store,source,compiled,4) for source,compiled in zip(
        core.compiled_source_classes(inputs,replay["recipe"]),replay["source_classes"])]
    rows=[]
    for c in classes:
        for batch in c["batches"]:
            call=inputs["run"]["evaluations"][batch["evaluation_index"]] if batch["phase"]=="training" else inputs["run"]["validation"]
            group=call["groups"][0]
            rows.append({**batch,"class_id":c["class_id"],"parameters":c["parameters"],
                "chronological_group_index":batch["evaluation_index"] if batch["phase"]=="training" else 109,
                "accepted":call.get("accepted",False),"axis":call.get("axis"),"direction":call.get("direction"),
                "objective_estimate":call["objective_estimate"],"standard_error":call["estimated_standard_error"],
                "counts":group["counts"],"rng":group["rng"],
                "recorded_source_sha256":c["circuits"]["recorded_source"]["sha256"],
                "routed_sha256":c["circuits"]["routed"]["sha256"],"clifford_t_sha256":c["circuits"]["clifford_t"]["sha256"]})
    rows.sort(key=lambda row:row["chronological_group_index"])
    legacy.save_csv(store,"groups.csv",rows)
    legacy.save_csv(store,"classes.csv",[dict(class_id=c["class_id"],parameters=c["parameters"],shots=c["shots"],batches=len(c["batches"]),
        actual_t_per_preparation=sum(c["actual_synthesis"]["clifford_t_counts"].get(name,0) for name in ("t","tdg")),
        actual_operator_error=c["checks"]["actual_full_operator_error"],guarded_operator_error=c["actual_synthesis"]["guarded_numerical_operator_error_upper_estimate"],
        source_sha256=c["circuits"]["recorded_source"]["sha256"],routed_sha256=c["circuits"]["routed"]["sha256"],clifford_t_sha256=c["circuits"]["clifford_t"]["sha256"]) for c in classes])
    legacy.save_csv(store,"negative-controls.csv",[dict(kind=kind,**row) for kind,values in inputs["negative_controls"].items()
        if isinstance(values,list) for row in values])
    store.json("resources.json",replay["resources"])
    store.json("conditional-readout.json",replay["conditional_readout"])
    store.json("classical-work.json",replay["classical_work"])
    selections=[]
    for label,key in (("minimum_qubits","selected_minimum_qubits"),("minimum_runtime","selected_minimum_runtime")):
        point=replay["physical"][key]
        selections.append(dict(selection=label,**{name:point[name] for name in (
            "data_code_distance","native_factory_isa_sha256","native_t_instruction_sha256","global_factory_retry_cap",
            "peak_physical_qubits","factory_physical_qubits","worst_case_serial_runtime_ns","whole_physical_failure_model",
            "whole_factory_availability_failure_bound","total_preparations","total_t_states")}))
    legacy.save_csv(store,"physical-summary.csv",selections)
    store.json("physical-summary.json",dict(profile=replay["physical"]["profile"],selections=selections,
        budgets=dict(physical=.0025,factory_availability=.0025),conditional=True,hardware_guarantee=False))
    timing=json.loads(store.target("completion-timing.json").read_text())
    summary=legacy.summary_markdown(replay)
    summary += ("\nThe physical estimator was evaluated twice. The original representation-limit failure remains visible and its measured time/query counts are charged. "
                "One explicitly declared computation-only recovery reused all 54 compiled checkpoints; quantum preparations and T requirements are counted once. "
                "The lossless DAG was checked against the second live producer result. Complete first-pass point records were unavailable, so no equality to that missing result is claimed.\n")
    summary += "\n| Common selection | Distance | Peak physical qubits | Serial model time (s) | Whole physical-fault model | Factory availability bound |\n|---|---:|---:|---:|---:|---:|\n"
    for point in selections:
        summary += f"| {point['selection']} | {point['data_code_distance']} | {point['peak_physical_qubits']:,} | {point['worst_case_serial_runtime_ns']/1e9:.3f} | {point['whole_physical_failure_model']:.6g} | {point['whole_factory_availability_failure_bound']:.6g} |\n"
    store.write("summary.md",summary.encode())
    legacy.export_figures(store,replay)
    manifest=dict(schema_version=1,status="complete",replay_status=replay["status"],recipe_id=replay["recipe"]["recipe_id"],run_id=replay["run_id"],
        identity=frozen["identity"],original_identity=saved["original_identity"],bindings=replay["bindings"],provenance=saved["provenance"],
        completed_replay=dict(file="completed-replay.json.gz",sha256=sha(store.target("completed-replay.json.gz"))),
        physical_storage=saved["physical_storage"],classes=classes,recovery=saved["recovery"],
        timing=dict(synthesis=json.loads(store.target("timing/synthesis.json").read_text()),original_physical=json.loads(store.target("timing/physical.json").read_text()),completion=timing),
        artifact_size_file="artifact-size.json",sampled_groups=110,preparations=82032,
        old_failure_file="checkpoints/failures/physical-workload.json",public_reader_requires_private_pickle=False,
        limitations=replay["limitations"])
    store.json("manifest.json",manifest)
    size=legacy.write_artifact_size(store)
    verify_preserved(store,frozen["original_artifact_inventory"])
    return dict(status="complete_lossless_physical_recovery",run_id=replay["run_id"],artifact_bytes=size["total_bytes"],artifact_byte_cap=16*1024*1024,
        source_classes=54,groups=110,quantum_preparations=82032,actual_t_states=replay["resources"]["actual_t_states"],
        compiled_synthesis_repeated=False,new_optimizer_runs=0,physical_cpu_passes_including_preserved_failure=2,
        cumulative_query_counts=saved["recovery"]["cumulative_query_counts"],cumulative_physical_wall_seconds=timing["cumulative_wall_seconds"])


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir",type=Path,default=Path(tempfile.gettempdir()).resolve()/"optimization-compiled-native-cache")
    choices=parser.add_mutually_exclusive_group()
    choices.add_argument("--freeze-only",action="store_true")
    choices.add_argument("--resume",action="store_true")
    choices.add_argument("--verify-only",action="store_true")
    parser.add_argument("--worker-input",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--worker-output",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_input:
        if not args.worker_output:
            parser.error("Worker output is required")
        physical_worker(args.worker_input,args.worker_output)
        return
    result=completion_freeze(args.output_dir) if args.freeze_only else complete(args.output_dir,args.cache_dir,resume=args.resume,verify_only=args.verify_only)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
