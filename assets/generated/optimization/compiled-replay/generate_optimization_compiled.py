#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export one frozen complete QAOA trajectory as actual Clifford+T instruments.

The recipe and executable bytes must be frozen before synthesis/native queries.
Each completed class has an immutable checkpoint. A completed replay may be
verified or exported again without resampling, reoptimizing or resynthesizing.
Public files contain no private cache, remote coordinates or raw process logs.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
from time import perf_counter, process_time

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
CHECKOUT = (SCRIPT == PROJECT / "scripts/generate_optimization_compiled.py"
            and (PROJECT / "src/openquantum/instrument_resources.py").is_file())
ROOT = PROJECT if CHECKOUT else SCRIPT.parent
CORE_PATH = ROOT / "src/openquantum/optimization_compiled.py" if CHECKOUT else ROOT / "optimization_compiled.py"
DEFAULT_OUTPUT = ROOT / ("assets/generated/optimization/compiled-replay" if CHECKOUT else "compiled-replay-artifacts")
DEFAULT_BYTE_CAP = 16 * 1024 * 1024
DEFAULT_DECODED_RECORD_CAP = 64 * 1024 * 1024
DEFAULT_DECODED_CLASS_CAP = 4 * 1024 * 1024


def encoded(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_gzip(raw: bytes, maximum: int) -> bytes:
    """Bound expansion before parsing a checkpoint or private worker record."""
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("Decoded record cap must be a positive integer")
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as handle:
        decoded = handle.read(maximum + 1)
    if len(decoded) > maximum:
        raise ValueError("Compressed record exceeds its frozen decoded-byte cap")
    return decoded


def load_core():
    if CORE_PATH.is_symlink() or not CORE_PATH.is_file():
        raise ValueError("The matching optimization_compiled.py producer is required")
    raw = CORE_PATH.read_bytes()
    spec = importlib.util.spec_from_file_location("optimization_compiled_artifact_source", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(CORE_PATH), "exec"), module.__dict__)
    return module, raw


def reject_linked_path(path: Path) -> None:
    supplied = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (supplied, *supplied.parents)):
        raise ValueError("Use a canonical path without linked output/cache ancestors")


def validate_output(output: Path, cache: Path | None = None) -> Path:
    reject_linked_path(output)
    output = Path(output).resolve()
    if CHECKOUT and output.is_relative_to(ROOT) and not output.is_relative_to(DEFAULT_OUTPUT):
        raise ValueError("Repository output is restricted to the new compiled-replay artifact tree")
    if output == ROOT or CORE_PATH.is_relative_to(output):
        # A standalone bundle may export into itself only if existing source
        # bytes are preserved by every source publication operation below.
        if CHECKOUT:
            raise ValueError("Output may not contain a source checkout")
    if cache is not None:
        reject_linked_path(cache)
        cache = Path(cache).resolve()
        if cache.is_relative_to(output) or output.is_relative_to(cache) or (CHECKOUT and cache.is_relative_to(ROOT)):
            raise ValueError("Native caches must stay private and outside the output/source checkout")
    if output.is_symlink():
        raise ValueError("Artifact output cannot be a symbolic link")
    return output


class ArtifactStore:
    """Bound the complete public output and publish regular files atomically."""

    def __init__(self, root: Path, byte_cap: int = DEFAULT_BYTE_CAP):
        if isinstance(byte_cap, bool) or not isinstance(byte_cap, int) or byte_cap <= 0:
            raise ValueError("Artifact byte cap must be a positive integer")
        reject_linked_path(root)
        self.root, self.byte_cap = Path(root).resolve(), byte_cap
        self.root.mkdir(parents=True, exist_ok=True)
        self.inventory()

    def inventory(self, *, hashes: bool = True) -> dict:
        records = {}
        for path in sorted(self.root.rglob("*")):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Artifact tree contains a linked or nonregular path")
            if stat.S_ISREG(mode):
                records[str(path.relative_to(self.root))] = dict(bytes=path.stat().st_size,
                    **({"sha256": digest(path)} if hashes else {}))
        if sum(row["bytes"] for row in records.values()) > self.byte_cap:
            raise ValueError("Complete artifact tree exceeds its frozen byte cap")
        return records

    def target(self, relative: str) -> Path:
        candidate = self.root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Artifact path must be relative without parent traversal")
        if any(path.is_symlink() for path in [candidate, *candidate.parents] if path.is_relative_to(self.root)):
            raise ValueError("Managed artifact path cannot be linked")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def write(self, relative: str, raw: bytes, *, immutable: bool = False) -> Path:
        path = self.target(relative)
        if immutable and path.exists():
            if path.read_bytes() != raw:
                raise ValueError(f"Preserved artifact differs: {relative}")
            return path
        old_size = path.stat().st_size if path.exists() else 0
        size = sum(row["bytes"] for row in self.inventory(hashes=False).values()) - old_size + len(raw)
        if size > self.byte_cap:
            raise ValueError(f"Publishing {relative} would exceed the frozen public-artifact cap")
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
        try:
            if immutable:
                os.link(temporary, path)
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def json(self, relative: str, value, *, immutable: bool = False) -> Path:
        return self.write(relative, encoded(value), immutable=immutable)

    def json_gzip(self, relative: str, value, *, immutable: bool = False,
                  decoded_cap: int = DEFAULT_DECODED_RECORD_CAP) -> Path:
        raw = encoded(value)
        if len(raw) > decoded_cap:
            raise ValueError("Serialized checkpoint/aggregate exceeds its frozen decoded-byte cap")
        return self.write(relative, gzip.compress(raw, mtime=0), immutable=immutable)


def producer_sources(core_bytes: bytes) -> dict[str, bytes]:
    if CHECKOUT:
        paths = {
            "instrument_resources.py": ROOT / "assets/generated/instrument-resources/instrument_resources.py",
            "parent_optimization.py": ROOT / "assets/generated/optimization/diagnostics/optimization.py",
            "parent_generate_optimization_diagnostics.py": ROOT / "assets/generated/optimization/diagnostics/generate_optimization_diagnostics.py",
            "requirements-lock.txt": ROOT / "requirements-lock.txt",
            "LICENSE.txt": ROOT / "LICENSE",
            "references.bib": ROOT / "assets/generated/instrument-resources/references.bib",
        }
    else:
        paths = {name: ROOT / name for name in ("instrument_resources.py", "parent_optimization.py", "parent_generate_optimization_diagnostics.py",
                                                "requirements-lock.txt", "LICENSE.txt", "references.bib")}
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError("Matching regular frozen backend/parent producer/lock/license sources are required")
    return {"optimization_compiled.py": core_bytes,
            "generate_optimization_compiled.py": SCRIPT.read_bytes(),
            **{name: path.read_bytes() for name, path in paths.items()}}


def provenance(sources: dict[str, bytes]) -> dict:
    versions = {name: importlib.metadata.version(name) for name in
                ("qiskit", "numpy", "scipy", "matplotlib", "qdk", "jupyter-book")}
    expected = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1",
                "matplotlib": "3.11.2", "qdk": "1.34.0", "jupyter-book": "2.1.7"}
    if platform.python_version() != "3.12.14" or versions != expected:
        raise ValueError("Compiled replay requires the reviewed Python3.12.14 and pinned package environment")
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  packages=versions, source_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                  source_commit_at_generation=None, dirty_owned_source=None,
                  device=dict(backend="CPU actual Ross--Selinger synthesis, independent tiny numerical audits and conditional native QDK model",
                              architecture=platform.machine(), operating_system=platform.system(), accelerator=None),
                  invocation="python scripts/generate_optimization_compiled.py" if CHECKOUT else "python generate_optimization_compiled.py")
    helper = ROOT / "scripts/source_provenance.py"
    if CHECKOUT and helper.is_file():
        spec = importlib.util.spec_from_file_location("compiled_source_provenance", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record.update(source_commit_at_generation=module.git_revision(ROOT),
                      dirty_owned_source=module.git_dirty(ROOT, ["src/openquantum/optimization_compiled.py",
                                         "scripts/generate_optimization_compiled.py", "notebooks/optimization_compiled.ipynb"]),
                      git_provenance_helper_sha256=digest(helper))
    return record


def checkpoint_identity(recipe: dict, info: dict, transcript: dict) -> dict:
    return dict(recipe_sha256=hashlib.sha256(encoded(recipe)).hexdigest(),
                selected_transcript_sha256=hashlib.sha256(encoded(transcript)).hexdigest(),
                source_sha256=info["source_sha256"])


def save_checkpoint(store: ArtifactStore, kind: str, key: str, identity: dict, record: dict, *, compressed: bool = False) -> None:
    value = dict(schema_version=1, kind=kind, key=key, identity=identity, status="completed", record=record)
    maximum = DEFAULT_DECODED_CLASS_CAP if kind == "synthesis" else DEFAULT_DECODED_RECORD_CAP
    if len(encoded(value)) > maximum:
        raise ValueError("Serialized checkpoint exceeds its frozen decoded-byte cap")
    if compressed:
        store.json_gzip(f"checkpoints/{kind}/{key}.json.gz", value, immutable=True, decoded_cap=maximum)
    else:
        store.json(f"checkpoints/{kind}/{key}.json", value, immutable=True)


def load_checkpoint(store: ArtifactStore, kind: str, key: str, identity: dict) -> dict:
    plain = store.target(f"checkpoints/{kind}/{key}.json")
    compressed = store.target(f"checkpoints/{kind}/{key}.json.gz")
    if plain.exists() and compressed.exists():
        raise ValueError("Ambiguous duplicate checkpoint encodings")
    maximum = DEFAULT_DECODED_CLASS_CAP if kind == "synthesis" else DEFAULT_DECODED_RECORD_CAP
    raw = decoded_gzip(compressed.read_bytes(), maximum) if compressed.exists() else plain.read_bytes()
    if len(raw) > maximum:
        raise ValueError("Checkpoint exceeds its frozen decoded-byte cap")
    value = json.loads(raw)
    if (value.get("kind") != kind or value.get("key") != key or value.get("identity") != identity
            or value.get("status") != "completed"):
        raise ValueError("Checkpoint identity/key/completion differs from the frozen replay")
    return value["record"]


def save_csv(store: ArtifactStore, relative: str, rows: list[dict]) -> None:
    import io
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, separators=(",", ":"))
                         if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
    store.write(relative, stream.getvalue().encode())


def export_circuit(store: ArtifactStore, circuit, *, role: str) -> dict:
    """Deduplicate exact exported bytes; keep omitted overall phase separately."""
    from qiskit import qasm3
    raw = qasm3.dumps(circuit).encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    file = f"circuits/{sha}.qasm"
    store.write(file, raw, immutable=True)
    phase = float(circuit.global_phase)
    if not math.isfinite(phase):
        raise ValueError("Bound circuit must have a finite recorded global phase")
    measurements, resets, cx_neighbors = [], [], True
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        if name == "measure":
            measurements.append([wires[0], circuit.find_bit(item.clbits[0]).index])
        elif name == "reset":
            resets.append(wires[0])
        elif name == "cx":
            cx_neighbors = cx_neighbors and abs(wires[0] - wires[1]) == 1
    return dict(role=role, file=file, sha256=sha, bytes=len(raw), global_phase=phase,
                global_phase_scope="QASM exporter omits overall phase; restore this scalar by assignment",
                logical_qubits=circuit.num_qubits, classical_bits=circuit.num_clbits,
                gate_counts={name: int(count) for name, count in circuit.count_ops().items()},
                source_operations=len(circuit.data), depth=circuit.depth(),
                measurement_map=measurements, reset_wire_order=resets,
                all_cx_are_line_neighbors=cx_neighbors,
                controlled_oracle_phase_certificate=False)


def save_compiled_instrument(store: ArtifactStore, instrument, *, original=None) -> dict:
    circuits = {"augmented_source": export_circuit(store, instrument.source, role="augmented_source"),
                "routed": export_circuit(store, instrument.routed, role="routed"),
                "clifford_t": export_circuit(store, instrument.circuit, role="clifford_t")}
    if original is not None:
        circuits["recorded_source"] = export_circuit(store, original, role="recorded_source")
    return dict(circuits=circuits, actual_synthesis=instrument.report)


# The orchestration below is finalized against the owned core's stable API
# before the first recipe/source freeze. No synthesis occurs at import time.


def bound(recipe: dict, key: str, default: int | float) -> int | float:
    value = recipe.get("bounds", {}).get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Frozen {key} bound must be positive and finite")
    return value


def freeze(output: Path) -> dict:
    core, core_bytes = load_core()
    inputs = core.load_compiled_inputs(ROOT)
    recipe = core.compiled_recipe(inputs)
    sources = producer_sources(core_bytes)
    info = provenance(sources)
    store = ArtifactStore(validate_output(output), int(bound(recipe, "artifact_bytes", DEFAULT_BYTE_CAP)))
    identity = checkpoint_identity(recipe, info, inputs)
    store.json("recipe.json", recipe, immutable=True)
    store.json("selected-transcript.json", inputs, immutable=True)
    store.json_gzip("source-classes.json.gz", core.compiled_source_classes(inputs, recipe), immutable=True)
    for name, raw in sources.items():
        store.write(name, raw, immutable=True)
    record = dict(schema_version=1, status="recipe_and_producers_frozen_before_compilation",
                  execution_performed=False, identity=identity,
                  captured_at_utc=info["generated_at_utc"], source_commit_at_capture=info["source_commit_at_generation"],
                  source_files=list(sources),
                  scope="frozen saved QAOA decisions; no resampling/optimization or hardware execution")
    path = store.target("freeze.json")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity or saved["status"] != record["status"]:
            raise ValueError("Frozen replay/producer identity changed")
        record = saved
    else:
        store.json("freeze.json", record, immutable=True)
    return dict(status=record["status"], execution_performed=False,
                identity=identity, artifact_bytes=sum(r["bytes"] for r in store.inventory(hashes=False).values()))


class PhaseBudget:
    """Cumulative actual invocation time, retained across incomplete resumes."""

    def __init__(self, store: ArtifactStore, phase: str, identity: dict, maximum: float):
        self.store, self.phase, self.identity, self.maximum = store, phase, identity, maximum
        self.relative = f"timing/{phase}.json"
        path = store.target(self.relative)
        self.record = json.loads(path.read_text()) if path.exists() else dict(
            schema_version=1, phase=phase, identity=identity, status="not_started",
            total_wall_seconds=0.0, invocations=[])
        if self.record["identity"] != identity or self.record["phase"] != phase:
            raise ValueError("Execution timing identity differs from frozen replay")
        total = self.record["total_wall_seconds"]
        if (not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0
                or any(not isinstance(row["wall_seconds"], (int, float)) or not math.isfinite(row["wall_seconds"])
                       or row["wall_seconds"] < 0 or row["ordinal"] != i + 1
                       for i, row in enumerate(self.record["invocations"]))
                or not math.isclose(total, sum(row["wall_seconds"] for row in self.record["invocations"]), rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError("Cumulative execution timing is invalid or incomplete")
        if self.record["status"] == "running":
            raise ValueError("Prior invocation ended without measured completion; resume cannot reset unknown effort")
        self.previous = total
        self.complete = False

    def __enter__(self):
        if self.previous >= self.maximum:
            raise RuntimeError("Frozen cumulative execution cap has been exhausted")
        self.started, self.cpu_started = perf_counter(), process_time()
        self.record["status"] = "running"
        self.store.json(self.relative, self.record)
        return self

    def check(self):
        total = self.previous + perf_counter() - self.started
        if total > self.maximum:
            raise RuntimeError("Frozen cumulative execution cap exceeded; completed checkpoints remain preserved")
        return total

    def remaining(self):
        return max(0.0, self.maximum - self.check())

    def __exit__(self, error_type, error, traceback):
        wall = perf_counter() - self.started
        total = self.previous + wall
        status = "complete" if self.complete and error_type is None and total <= self.maximum else "incomplete"
        if total > self.maximum:
            status = "cap_exceeded_incomplete"
        self.record["invocations"].append(dict(ordinal=len(self.record["invocations"]) + 1,
            wall_seconds=wall, parent_process_cpu_seconds=process_time() - self.cpu_started,
            status=status, error_type=error_type.__name__ if error_type else None))
        self.record.update(status=status, total_wall_seconds=total, wall_cap_seconds=self.maximum,
            scope="actual parent invocation wall including bounded child workers/checkpoint writes; export verification separate")
        self.store.json(self.relative, self.record)
        if error_type is None and status != "complete":
            raise RuntimeError("Execution did not complete under its frozen cumulative cap")
        return False


def worker(kind: str, input_path: Path, output_path: Path) -> None:
    core, _ = load_core()
    # This configures the identical backend/parent producer paths without
    # making any optimizer, synthesis or resource call.
    core.load_compiled_inputs(ROOT)
    maximum = DEFAULT_DECODED_CLASS_CAP if kind == "synthesis" else DEFAULT_DECODED_RECORD_CAP
    payload = json.loads(decoded_gzip(input_path.read_bytes(), maximum))
    began, cpu_began = perf_counter(), process_time()
    try:
        if kind == "synthesis":
            record = core.compile_source_class(payload["class"], payload["recipe"])
        elif kind == "physical":
            record = core.run_compiled_physical(payload["classes"], payload["recipe"], cache_dir=Path(payload["cache_dir"]))
        else:
            raise ValueError("Unknown private worker phase")
        value = dict(status="completed", record=record)
    except Exception as error:
        audit = getattr(error, "audit", getattr(error, "seventh_physical_audit", None))
        value = dict(status="failed_bounded_phase_incomplete", error_type=type(error).__name__,
                     error_message=public_error_text(str(error)),
                     physical_query_audit=audit, partial_native_work_unknown=audit is None,
                     automatic_retry=False)
    value.update(worker_wall_seconds=perf_counter() - began,
                 worker_process_cpu_seconds=process_time() - cpu_began)
    try:
        raw = encoded(value)
        if len(raw) > maximum:
            raise ValueError("Worker result exceeds its frozen decoded-byte cap")
    except (ValueError, TypeError, OverflowError) as error:
        actual = value.get("record", {})
        audit = actual.get("query_audit", value.get("physical_query_audit"))
        # A serialization-cap failure remains a failed scientific phase rather
        # than silently retrying the native job or publishing a partial result.
        value = dict(status="failed_bounded_phase_incomplete", error_type="DecodedByteCapOrSerializationFailure",
            error_message=public_error_text(str(error)), decoded_byte_cap=maximum,
            observed_query_counters=actual.get("query_counters"),
            physical_query_audit=({key: audit[key] for key in ("status", "caps", "counts", "wall_seconds") if key in audit}
                                  if isinstance(audit, dict) else None),
            complete_query_records_omitted_due_to_serialization_limit=isinstance(audit, dict),
            worker_wall_seconds=perf_counter() - began, worker_process_cpu_seconds=process_time() - cpu_began,
            partial_native_work_unknown=not bool(actual.get("query_counters") or audit), automatic_retry=False)
        raw = encoded(value)
    output_path.write_bytes(gzip.compress(raw, mtime=0))


def public_error_text(message: str) -> str:
    # Worker exceptions contain local producer/cache/temp paths at most. Keep
    # the scientific limit reason, while excluding those paths and tracebacks.
    return re.sub(r"(?<![A-Za-z0-9])/(?:[^\s'\"<>]+)", "<private-path>", message)[:2048]


class BoundedWorkerFailure(RuntimeError):
    def __init__(self, record: dict):
        self.failure_record = record
        super().__init__(record.get("error_message", "Bounded worker did not complete"))


def run_worker(kind: str, payload: dict, timeout: float) -> dict:
    if timeout <= 0:
        raise RuntimeError("No remaining frozen worker allowance")
    with tempfile.TemporaryDirectory(prefix="optimization-compiled-worker-") as temporary:
        folder = Path(temporary).resolve()
        input_path, output_path = folder / "input.json.gz", folder / "result.json.gz"
        maximum = DEFAULT_DECODED_CLASS_CAP if kind == "synthesis" else DEFAULT_DECODED_RECORD_CAP
        raw = encoded(payload)
        if len(raw) > maximum:
            raise ValueError("Worker input exceeds its frozen decoded-byte cap")
        input_path.write_bytes(gzip.compress(raw, mtime=0))
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, str(SCRIPT), "--worker", kind,
                                 "--worker-input", str(input_path), "--worker-output", str(output_path)],
                                cwd=ROOT, env=environment, capture_output=True, timeout=timeout)
        if result.returncode != 0 or not output_path.is_file():
            raise BoundedWorkerFailure(dict(status="failed_bounded_phase_incomplete",
                error_type="WorkerProcessFailure", error_message=f"Bounded {kind} worker exited {result.returncode} without a structured result",
                partial_native_work_unknown=True, automatic_retry=False))
        value = json.loads(decoded_gzip(output_path.read_bytes(), maximum))
        if value.get("status") != "completed":
            raise BoundedWorkerFailure(value)
        return value


def compressed_record(path: Path, maximum: int = DEFAULT_DECODED_RECORD_CAP) -> dict:
    return json.loads(decoded_gzip(path.read_bytes(), maximum))


def phase_timing(store: ArtifactStore, phase: str, identity: dict, maximum: float) -> dict:
    value = json.loads(store.target(f"timing/{phase}.json").read_text())
    if value["identity"] != identity or value["status"] != "complete" or value["total_wall_seconds"] > maximum:
        raise ValueError("Completed replay has no matching bounded timing ledger")
    # Validate finite totals and measured invocation history even for resume.
    PhaseBudget(store, phase, identity, maximum)
    return value


def assert_current_sources(sources: dict[str, bytes]) -> None:
    current = producer_sources(CORE_PATH.read_bytes())
    if current != sources:
        raise RuntimeError("Producer changed during execution; captured checkpoints are preserved")


def export_serialized_circuit(store: ArtifactStore, text: str, operations: list,
                              phase: float, *, role: str, qubits: int) -> dict:
    raw = text.encode("utf-8")
    sha = hashlib.sha256(raw).hexdigest()
    file = f"circuits/{sha}.qasm"
    store.write(file, raw, immutable=True)
    from collections import Counter
    counts = Counter(row[0] for row in operations)
    measurements = [[row[1][0], row[2][0]] for row in operations if row[0] == "measure"]
    resets = [row[1][0] for row in operations if row[0] == "reset"]
    if not math.isfinite(phase):
        raise ValueError("Compiled global phase must be finite")
    return dict(role=role, file=file, sha256=sha, bytes=len(raw), global_phase=phase,
                logical_qubits=qubits, gate_counts=dict(counts), operations=len(operations),
                measurement_map=measurements, reset_wire_order=resets,
                all_cx_are_line_neighbors=all(abs(row[1][0] - row[1][1]) == 1 for row in operations if row[0] == "cx"),
                phase_scope="overall phase omitted by QASM; restore this exact scalar by assignment",
                controlled_oracle_phase_certificate=False)


def export_class(store: ArtifactStore, source: dict, compiled: dict, qubits: int) -> dict:
    circuits = {}
    for stage, role in (("source", "augmented_source"), ("routed", "routed"), ("clifford_t", "clifford_t")):
        circuits[role] = export_serialized_circuit(store, compiled[f"{stage}_qasm"],
            compiled[f"{stage}_operations"], compiled[f"{stage}_global_phase"], role=role, qubits=qubits)
    original_raw = source["source_qasm"].encode()
    original_sha = hashlib.sha256(original_raw).hexdigest()
    store.write(f"circuits/{original_sha}.qasm", original_raw, immutable=True)
    circuits["recorded_source"] = dict(role="recorded_source", file=f"circuits/{original_sha}.qasm",
        sha256=original_sha, bytes=len(original_raw), global_phase=0.0,
        phase_scope="original bound QAOA source; reference identity contribution compared up to global phase")
    return dict(class_id=compiled["class_id"], parameters=compiled["parameters"],
                parameter_hex=[float(value).hex() for value in compiled["parameters"]],
                shots=compiled["shots"], batches=compiled["batches"], circuits=circuits,
                actual_synthesis=compiled["report"], checks=compiled["checks"],
                timing=compiled["timing"], query_counters=compiled["query_counters"],
                checkpoint=f"checkpoints/synthesis/{compiled['class_id']}.json.gz")


def ensure_committed_freeze(store: ArtifactStore) -> None:
    """Require the reviewable freeze in a genuine checkout, not ancestor Git."""
    if not CHECKOUT or not ((ROOT / ".git").is_file() or (ROOT / ".git").is_dir()):
        return
    paths = [CORE_PATH, SCRIPT, store.target("freeze.json"), store.target("recipe.json"),
             store.target("selected-transcript.json")]
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
    top = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                         env=environment, capture_output=True, text=True, timeout=5)
    if top.returncode or Path(top.stdout.strip()).resolve() != ROOT:
        raise ValueError("Compilation checkout Git metadata does not identify this exact source root")
    for path in paths:
        relative = str(path.relative_to(ROOT))
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), "show", f"HEAD:{relative}"],
                                env=environment, capture_output=True, timeout=5)
        if result.returncode or result.stdout != path.read_bytes():
            raise ValueError("Commit the unchanged recipe, selected transcript and producer bytes before execution")


def mark_failure(store: ArtifactStore, phase: str, key: str, identity: dict, error: BaseException) -> None:
    details = getattr(error, "failure_record", None)
    if details is None:
        details = dict(error_type=type(error).__name__, error_message=public_error_text(str(error)),
            timeout_seconds=error.timeout if isinstance(error, subprocess.TimeoutExpired) else None,
            partial_native_work_unknown=True, automatic_retry=False)
    store.json(f"checkpoints/failures/{phase}-{key}.json", dict(schema_version=1, identity=identity,
               phase=phase, key=key, status="failed_bounded_phase_incomplete", error_type=type(error).__name__,
               failure=details,
               raw_process_logs_public=False, automatic_retry=False,
               scope="completed checkpoints retained; unknown partial native work cannot reset the frozen cap"), immutable=True)


def check_checkpoint_inventory(store: ArtifactStore, replay: dict, identity: dict) -> None:
    expected = {f"checkpoints/synthesis/{c['class_id']}.json.gz" for c in replay["source_classes"]}
    expected.add("checkpoints/physical/workload.json.gz")
    actual = {str(path.relative_to(store.root)) for path in (store.root / "checkpoints").rglob("*.json*")}
    if actual != expected:
        raise ValueError("Completed replay checkpoint inventory differs from the full fixed workload")
    for compiled in replay["source_classes"]:
        if load_checkpoint(store, "synthesis", compiled["class_id"], identity) != compiled:
            raise ValueError("Compiled class checkpoint differs from immutable aggregate")
    if load_checkpoint(store, "physical", "workload", identity) != replay["physical"]:
        raise ValueError("Physical workload checkpoint differs from immutable aggregate")


def summary_markdown(replay: dict) -> str:
    resource, readout, recipe = replay["resources"], replay["conditional_readout"], replay["recipe"]
    return f'''The replay fixes **{replay["run_id"]}**, the first heldout-supported four-wire QAOA run in the frozen order. It includes **109 training calls**, **one holdout**, **110 measured groups**, and all **82,032 preparations**, including every rejected proposal. The saved ideal-sampling target status remains **supported**; VQE inconclusive labels and every other archived negative control remain unchanged.

The **54 exact source classes** have actual Clifford+T approximants and actual logical-line routing. Weighted across every recorded group, the replay uses **{resource["actual_t_states"]:,} actual T/T-dagger state consumptions** and **328,128 original output-qubit readouts**. Explicit input resets are added for every preparation; the separate instrument model also charges reserved workspace starts, magic-state supply/injection, macro readout, and classical dispatch.

The summed guarded numerical realization estimate over the fixed full transcript is **{resource["realization_guarded_static_transcript_tv"]:.8g}**, within the frozen 0.005 allocation. The final compiled exact ratio is **{readout["compiled_exact_ratio"]:.10f}**. Its prospective fixed-state precision radius is **{readout["normalized_precision_radius"]:.8f}**, below 0.05. These use actual saved sequences and tiny independent numerical checks, not a mean-fit T proxy or formal interval certificate.

The conditional allocation reserves 0.04 for the full 48-group sampling family, 0.005 for fixed-script realization distance, 0.0025 for whole-workload physical faults, and 0.0025 for factory availability. The surface-code layer remains the frozen serialized instrument scenario with physical error 10^-4. It is separate from the Grover/PSSPC baseline, and its geometry, controller, factory/interface and independence assumptions are not hardware guarantees.

The original counts were generated from ideal source probabilities. They were **not** redrawn from these approximants. A compiled replay conditional on those recorded classical choices does not prove that a changed noisy optimizer would take the same path. Offline CPU optimizer/synthesis/verification timing is recorded separately; hardware optimizer/scoring/communication and decoder latency remain unmeasured. No convergence or quantum advantage follows.
'''


def export_figures(store: ArtifactStore, replay: dict) -> None:
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = replay["resources"]
    with matplotlib.rc_context({"svg.hashsalt": "optimization-actual-compiled", "font.size": 9,
                               "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
        source = counts["source_weighted_counts"]
        compiled = counts["clifford_t_weighted_counts"]
        source_names = ["h", "rx", "rz", "cx"]
        actual_names = [name for name in ("h", "s", "sdg", "t", "tdg", "cx", "reset", "measure") if compiled.get(name)]
        axes[0].bar(source_names, [source.get(name, 0) for name in source_names], color="#176e70")
        axes[1].bar(actual_names, [compiled.get(name, 0) for name in actual_names], color="#71668d")
        axes[0].set(title="Recorded source gates", ylabel="Shot-weighted logical operations")
        axes[1].set(title="Actual compiled instrument operations", ylabel="Shot-weighted logical operations")
        for ax in axes:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("All 110 groups and 82,032 preparations; source and realization are separate workloads", fontsize=10)
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="svg", metadata={"Date": None}, bbox_inches="tight")
        plt.close(fig)
        store.write("workload.svg", stream.getvalue())


def write_artifact_size(store: ArtifactStore) -> dict:
    record = dict(schema_version=1, scope="all regular files in this generated bundle including this size record",
                  total_bytes=0, files=0, frozen_byte_cap=store.byte_cap,
                  parent_archive_copied=False, private_cache_included=False)
    for _ in range(8):
        store.json("artifact-size.json", record)
        inventory = store.inventory(hashes=False)
        size, files = sum(row["bytes"] for row in inventory.values()), len(inventory)
        if size == record["total_bytes"] and files == record["files"]:
            return record
        record.update(total_bytes=size, files=files)
    raise RuntimeError("Artifact size metadata did not reach an exact fixed point")


def export_replay(store: ArtifactStore, core, inputs: dict, replay: dict, identity: dict) -> dict:
    core.validate_compiled_record(replay)
    source_classes = core.compiled_source_classes(inputs, replay["recipe"])
    classes = [export_class(store, original, compiled, 4)
               for original, compiled in zip(source_classes, replay["source_classes"])]
    rows = []
    for c in classes:
        for batch in c["batches"]:
            rows.append(dict(class_id=c["class_id"], parameters=c["parameters"], **batch,
                             recorded_source_sha256=c["circuits"]["recorded_source"]["sha256"],
                             augmented_source_sha256=c["circuits"]["augmented_source"]["sha256"],
                             routed_sha256=c["circuits"]["routed"]["sha256"],
                             clifford_t_sha256=c["circuits"]["clifford_t"]["sha256"]))
    save_csv(store, "groups.csv", rows)
    save_csv(store, "classes.csv", [dict(class_id=c["class_id"], parameters=c["parameters"],
        shots=c["shots"], batches=len(c["batches"]),
        actual_t_per_preparation=sum(c["actual_synthesis"]["clifford_t_counts"].get(name, 0) for name in ("t", "tdg")),
        actual_operator_error=c["checks"]["actual_full_operator_error"],
        guarded_operator_error=c["actual_synthesis"]["guarded_numerical_operator_error_upper_estimate"],
        source_sha256=c["circuits"]["recorded_source"]["sha256"],
        routed_sha256=c["circuits"]["routed"]["sha256"], clifford_t_sha256=c["circuits"]["clifford_t"]["sha256"])
        for c in classes])
    save_csv(store, "negative-controls.csv", [dict(kind=kind, **row)
        for kind, rows in inputs["negative_controls"].items() if isinstance(rows, list) for row in rows])
    store.json("resources.json", replay["resources"])
    store.json("conditional-readout.json", replay["conditional_readout"])
    store.json("classical-work.json", replay["classical_work"])
    store.json_gzip("physical.json.gz", replay["physical"])
    store.write("summary.md", summary_markdown(replay).encode())
    export_figures(store, replay)
    manifest = dict(schema_version=1, status="complete", replay_status=replay["status"],
        recipe_id=replay["recipe"]["recipe_id"], run_id=replay["run_id"], identity=identity,
        provenance=replay["provenance"], bindings=replay["bindings"],
        raw_study=dict(file="raw-study.json.gz", sha256=digest(store.target("raw-study.json.gz"))),
        physical_checkpoint=dict(file="checkpoints/physical/workload.json.gz", sha256=digest(store.target("checkpoints/physical/workload.json.gz"))),
        timing=replay["execution_timing"], classes=classes,
        artifact_size_file="artifact-size.json", sampled_groups=110, preparations=82_032,
        phase_scope="full saved global phases are restored separately from QASM bytes; no controlled-oracle specification",
        conditional_scope="actual logical compilation plus conditional frozen physical macros; recorded classical decisions only",
        limitations=replay["limitations"])
    store.json("manifest.json", manifest)
    return write_artifact_size(store)


def generate(output: Path, cache: Path, *, resume: bool = False, verify_only: bool = False) -> dict:
    core, core_bytes = load_core()
    inputs = core.load_compiled_inputs(ROOT)
    recipe = core.compiled_recipe(inputs)
    sources = producer_sources(core_bytes)
    info = provenance(sources)
    store = ArtifactStore(validate_output(output, cache), int(bound(recipe, "artifact_bytes", DEFAULT_BYTE_CAP)))
    frozen_path = store.target("freeze.json")
    if not frozen_path.is_file():
        raise ValueError("Run --freeze-only and commit the preserved recipe/producers before execution")
    frozen = json.loads(frozen_path.read_text())
    identity = checkpoint_identity(recipe, info, inputs)
    if frozen["identity"] != identity or json.loads(store.target("recipe.json").read_text()) != recipe:
        raise ValueError("Executed recipe or producers differ from the immutable freeze")
    if json.loads(store.target("selected-transcript.json").read_text()) != inputs:
        raise ValueError("Selected transcript differs from the immutable freeze")
    for name, raw in sources.items():
        if store.target(name).read_bytes() != raw:
            raise ValueError("Matching preserved producer bytes are required for replay/resume")
    source_classes = core.compiled_source_classes(inputs, recipe)
    if compressed_record(store.target("source-classes.json.gz")) != source_classes:
        raise ValueError("Frozen source partition differs from immutable transcript")
    raw_path = store.target("raw-study.json.gz")
    existing = list(store.target("checkpoints").rglob("*.json*"))
    expected_paths = {f"checkpoints/synthesis/{c['class_id']}.json.gz" for c in source_classes}
    expected_paths.add("checkpoints/physical/workload.json.gz")
    unexpected = [str(path.relative_to(store.root)) for path in existing
                  if str(path.relative_to(store.root)) not in expected_paths
                  and not path.is_relative_to(store.root / "checkpoints/failures")]
    if unexpected:
        raise ValueError("Unrecognized replay checkpoints exist before execution")
    if (raw_path.exists() or existing) and not resume and not verify_only:
        raise ValueError("Execution records exist; use --resume to preserve completed classes")
    if verify_only and not raw_path.exists():
        raise ValueError("Verification-only requires a completed immutable replay")
    synthesis_executed = physical_executed = 0
    if raw_path.exists():
        replay = compressed_record(raw_path)
        if replay["identity"] != identity or replay["recipe"] != recipe:
            raise ValueError("Immutable replay identity differs from frozen producer/recipe")
        check_checkpoint_inventory(store, replay, identity)
        for phase, maximum in (("synthesis", bound(recipe, "compilation_wall_seconds", 300)),
                               ("physical", bound(recipe, "physical_wall_seconds", 120))):
            if phase_timing(store, phase, identity, maximum) != replay["execution_timing"][phase]:
                raise ValueError("Immutable aggregate timing differs from measured phase ledger")
    else:
        ensure_committed_freeze(store)
        failures = list(store.target("checkpoints/failures").glob("*.json"))
        if failures:
            raise ValueError("A failed bounded native phase is preserved; no automatic retry/reset is permitted")
        compiled = []
        synthesis_checkpoints = list(store.target("checkpoints/synthesis").glob("*.json.gz"))
        if synthesis_checkpoints and not store.target("timing/synthesis.json").is_file():
            raise ValueError("Completed synthesis checkpoints require their cumulative timing ledger")
        if len(synthesis_checkpoints) == len(source_classes):
            phase_timing(store, "synthesis", identity, bound(recipe, "compilation_wall_seconds", 300))
            compiled = [load_checkpoint(store, "synthesis", c["class_id"], identity) for c in source_classes]
        else:
            with PhaseBudget(store, "synthesis", identity, bound(recipe, "compilation_wall_seconds", 300)) as budget:
                for source in source_classes:
                    key = source["class_id"]
                    path = store.target(f"checkpoints/synthesis/{key}.json.gz")
                    if path.exists():
                        result = load_checkpoint(store, "synthesis", key, identity)
                    else:
                        try:
                            response = run_worker("synthesis", dict(recipe=recipe, **{"class": source}),
                                min(bound(recipe, "synthesis_wall_seconds_per_class", 20), budget.remaining()))
                            synthesis_executed += 1
                            result = response["record"]
                            result["worker_timing"] = {key: response[key] for key in ("worker_wall_seconds", "worker_process_cpu_seconds")}
                            save_checkpoint(store, "synthesis", key, identity, result, compressed=True)
                        except BaseException as error:
                            mark_failure(store, "synthesis", key, identity, error)
                            raise
                    export_class(store, source, result, 4)
                    compiled.append(result)
                    budget.check()
                    print(json.dumps(dict(stage="actual synthesis checkpointed", completed=len(compiled), planned=len(source_classes),
                                          class_id=key)), flush=True)
                assert_current_sources(sources)
                budget.complete = True
        physical_path = store.target("checkpoints/physical/workload.json.gz")
        if physical_path.exists():
            physical = load_checkpoint(store, "physical", "workload", identity)
            phase_timing(store, "physical", identity, bound(recipe, "physical_wall_seconds", 120))
        else:
            with PhaseBudget(store, "physical", identity, bound(recipe, "physical_wall_seconds", 120)) as budget:
                try:
                    response = run_worker("physical", dict(recipe=recipe, classes=compiled, cache_dir=str(cache.resolve())), budget.remaining())
                    physical_executed += 1
                    physical = response["record"]
                    physical["worker_timing"] = {key: response[key] for key in ("worker_wall_seconds", "worker_process_cpu_seconds")}
                    save_checkpoint(store, "physical", "workload", identity, physical, compressed=True)
                    budget.check()
                except BaseException as error:
                    mark_failure(store, "physical", "workload", identity, error)
                    raise
                assert_current_sources(sources)
                budget.complete = True
        replay = core.compiled_aggregate(inputs, recipe, compiled, physical)
        replay.update(identity=identity, provenance=info,
            execution_timing={phase: phase_timing(store, phase, identity, maximum) for phase, maximum in
                (("synthesis", bound(recipe, "compilation_wall_seconds", 300)), ("physical", bound(recipe, "physical_wall_seconds", 120)))})
        core.validate_compiled_record(replay)
        check_checkpoint_inventory(store, replay, identity)
        assert_current_sources(sources)
        store.json_gzip("raw-study.json.gz", replay, immutable=True)
    core.validate_compiled_record(replay)
    size = export_replay(store, core, inputs, replay, identity)
    assert_current_sources(sources)
    return dict(status=replay["status"], artifact_status="complete", run_id=replay["run_id"],
                classes=len(replay["source_classes"]), measured_groups=110, preparations=82_032,
                artifact_bytes=size["total_bytes"], artifact_byte_cap=size["frozen_byte_cap"],
                raw_sha256=digest(raw_path), new_optimizer_runs=0,
                synthesis_classes_executed_this_invocation=synthesis_executed,
                physical_workers_executed_this_invocation=physical_executed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir()).resolve() / "optimization-compiled-native-cache")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze-only", action="store_true")
    group.add_argument("--resume", action="store_true")
    group.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker", choices=("synthesis", "physical"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker_input is None or args.worker_output is None:
            parser.error("Private worker input and output are required")
        worker(args.worker, args.worker_input, args.worker_output)
        return
    result = freeze(args.output_dir) if args.freeze_only else generate(args.output_dir, args.cache_dir,
        resume=args.resume, verify_only=args.verify_only)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
