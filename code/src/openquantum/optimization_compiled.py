# SPDX-License-Identifier: MIT
"""Bounded actual Clifford+T replay of one immutable sampled QAOA trajectory.

The archive's classical choices remain fixed. Actual synthesis and logical
routing are numerical circuit artifacts; the surface-code layer is a separate
conditional instrument scenario, never a prediction of a noisy optimizer.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, qasm3

RUN_ID = "qaoa-cycle-4-p2-alternate-seed-7"
PINS = dict(
    parent_raw_sha256="a8649f45dd948302f201654e9ad341824363357fd37f8ed343281cdf9d846688",
    parent_implementation_sha256="a617b5e8d43b648427e5ceaa5e35d3f50ea14c02464465084f23f457ffb5855d",
    parent_generator_sha256="0c0f3ff0f177dca99a00cc611261c3b6c82ce39be251e8d3a54026b0f5346f3e",
    parent_recipe_sha256="3b991eaece34b43116afbbbc85d5e8da9e76fe7c9ff217a45c254989c40f5dd9",
    selected_checkpoint_sha256="7254bcde23f679b55cec00dae613c2dbccfcaefbfb7720ca3a5ed47f89b20cc8",
    selected_run_canonical_sha256="19ef35c53979ac8e7f6a24084ce12d2023e8a371cf39f1eb1b81370514dc96fe",
    negative_controls_sha256="ee9b9b57393c8cf0fee06d9cab938542eceb63521155c8eed7430983670a10dc",
    instrument_backend_sha256="fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863",
    lock_sha256="6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088")
_runtime_root: Path | None = None
_module_cache: dict = {}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_values(value):
    """Lossless explicit JSON representation of tiny numerical check arrays."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return dict(real=value.real.tolist(), imag=value.imag.tolist(), encoding="complex matrix as real/imag arrays")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, complex):
        return dict(real=float(value.real), imag=float(value.imag))
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


def _load_module(path: Path, pin: str, name: str):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Pinned producer must be a regular file")
    raw = path.read_bytes()
    if _sha(raw) != pin:
        raise ValueError("Frozen producer bytes changed")
    key = (str(path.resolve()), pin)
    if key not in _module_cache:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(raw, str(path), "exec"), module.__dict__)
        _module_cache[key] = module
    return _module_cache[key]


def _producers():
    if _runtime_root is None:
        raise ValueError("Call load_compiled_inputs with the checkout or portable bundle first")
    root = _runtime_root
    checkout = (root / "src/openquantum/optimization_compiled.py").is_file()
    parent = (root / "assets/generated/optimization/diagnostics/optimization.py"
              if checkout else root / "parent_optimization.py")
    backend = root / "src/openquantum/instrument_resources.py" if checkout else root / "instrument_resources.py"
    return (_load_module(parent, PINS["parent_implementation_sha256"], "compiled_replay_parent"),
            _load_module(backend, PINS["instrument_backend_sha256"], "compiled_replay_instrument"))


def load_compiled_inputs(root) -> dict:
    """Extract an immutable selected transcript, without repeating optimization."""
    global _runtime_root
    path = Path(root)
    if path.is_symlink():
        raise ValueError("Input root cannot be a symbolic link")
    root = path.resolve()
    _runtime_root = root
    archive = root / "assets/generated/optimization/diagnostics"
    if (archive / "raw-study.json").is_file():
        for name, pin in (("raw-study.json", "parent_raw_sha256"),
                          ("generate_optimization_diagnostics.py", "parent_generator_sha256"),
                          ("recipe.json", "parent_recipe_sha256"), ("requirements-lock.txt", "lock_sha256")):
            if _sha((archive / name).read_bytes()) != PINS[pin]:
                raise ValueError("Parent archive or producer binding changed")
        study = json.loads((archive / "raw-study.json").read_text())
        supported = [(case, run) for case in study["cases"] for run in case["runs"]
                     if case["algorithm"] == "qaoa" and case["n_qubits"] == 4
                     and run["validation"]["quality_bound"]["target_status"] == "supported"]
        if not supported or supported[0][1]["run_id"] != RUN_ID:
            raise ValueError("Frozen first-supported four-wire selection changed")
        case, run = supported[0]
        checkpoint_path = archive / "checkpoints" / f"sampled-{RUN_ID}.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        if (_sha(checkpoint_path.read_bytes()) != PINS["selected_checkpoint_sha256"]
                or checkpoint["run"] != run or checkpoint["status"] != "completed"):
            raise ValueError("Selected parent checkpoint does not bind to the raw transcript")
        inputs = dict(schema_version=1, case={k: v for k, v in case.items() if k != "runs"}, run=run,
                      parent_provenance=study["provenance"], bindings=PINS,
                      selection="first heldout-supported four-wire QAOA run in frozen case/start/seed order",
                      selection_scope="data-dependent choice protected by the full simultaneous final-group family",
                      negative_controls=dict(sampled=study["summary_rows"], cpu=study["control_summary_rows"],
                          physical_workload_scope="only selected run; all other runs/CPU controls remain uncompiled archived evidence"))
    else:
        inputs = json.loads((root / "selected-transcript.json").read_text())
    if inputs["bindings"] != PINS or inputs["run"]["run_id"] != RUN_ID:
        raise ValueError("Selected portable transcript binding changed")
    if _sha(_canonical(inputs["run"])) != PINS["selected_run_canonical_sha256"]:
        raise ValueError("Selected transcript content changed")
    if _sha(_canonical(inputs["negative_controls"])) != PINS["negative_controls_sha256"]:
        raise ValueError("Archived negative-control or VQE labels changed")
    parent, _ = _producers()
    if inputs["case"] != parent.diagnostic_cases()[1]:
        raise ValueError("Selected case differs from frozen C4 depth-two recipe")
    parent.diagnostic_replay_sampled(inputs["case"], inputs["run"])
    if inputs["run"]["validation"]["quality_bound"]["target_status"] != "supported":
        raise ValueError("Selected archived target is not supported")
    return inputs


def compiled_recipe(inputs: dict) -> dict:
    """Predeclared matched-output budgets and bounds; no synthesized results."""
    if inputs["bindings"] != PINS:
        raise ValueError("Inputs are not the frozen selected transcript")
    eta = float(Fraction(1, 200 * 82_032))
    delta = 0.04 / 48
    radius = math.sqrt(math.log(2 / delta) / (2 * 16_384))
    estimate = -inputs["run"]["validation"]["objective_estimate"] / 4
    return dict(schema_version=1, recipe_id="optimization-qaoa-actual-compiled-replay-v1",
        run_id=RUN_ID, bindings=PINS, case_id=inputs["case"]["case_id"],
        output=dict(task="fixed final-state mean cut / enumerated optimum", quality_at_least=0.8,
                    normalized_precision=0.05, success_at_least=0.95, n_qubits=4, classical_optimum=4),
        budget=dict(sampling_failure_family=0.04, sampling_final_group_count=48,
                    sampling_failure_per_group=delta, realization_static_transcript_tv=0.005,
                    physical_failure_whole_workload=0.0025, factory_failure_whole_workload=0.0025,
                    reserved_total=0.05, per_preparation_operator_allowance=eta,
                    final_ratio_bias_allowance=eta, conditional_sampling_radius=radius,
                    saved_estimate=estimate, conditional_ratio_lower=estimate-radius-eta,
                    conditional_precision_radius=radius+eta,
                    selection_coverage="retain all-48-group sampling protection; selection is outcome-dependent",
                    realization_scope="conditional numerical operator bounds transfer fixed-script event probabilities; not a native physical failure or adaptive-path theorem"),
        planned=dict(training_calls=109, holdout_calls=1, measured_groups=110, unique_source_classes=54,
                     training_preparations=65_648, holdout_preparations=16_384, total_preparations=82_032,
                     source_qubit_readouts=328_128, explicit_source_initializations=328_128,
                     source_unitary_gates=2_953_152,
                     source_weighted_counts=dict(h=328_128, rx=656_256, rz=656_256, cx=1_312_512, measure=328_128)),
        bounds=dict(source_qubits=4, source_operations=128, routed_operations=1024,
                    synthesized_operations=10_000, total_synthesized_operations=540_000,
                    validation_qubits=4, validation_operations=10_000, validation_branches=256,
                    rotation_occurrences_per_class=16, synthesis_native_calls_per_class=64,
                    total_synthesis_native_calls=3456, compilation_wall_seconds=300,
                    synthesis_wall_seconds_per_class=20, physical_wall_seconds=120,
                    physical_top_level_calls=64, factory_frontier_queries=1536,
                    factory_frontier_candidates=52224, factory_pipeline_candidates=52224,
                    native_trace_queries=52224, artifact_bytes=16*1024*1024,
                    decoded_record_bytes=64*1024*1024, decoded_class_bytes=4*1024*1024),
        compiler=dict(preserve_full_unitary=True, basis="actual Clifford+T plus reset/readout",
                      routing="bidirectional logical line, fixed initial layout, SABRE seed 7; frozen backend",
                      phase="restore exported global phase; undo measured output permutation for full operator checks",
                      native_synthesis_seed=None, repeated_sequences="preserve actual native synthesis bytes; do not regenerate checkpoints"),
        physical_profile="frozen InstrumentProfile defaults, physical error 1e-4; separate serialized instrument scenario",
        limitations=["compiled replay fixes every archived classical decision; no prediction of new hardware-driven optimization",
                     "original counts are ideal CPU samples, not samples from the newly synthesized circuit",
                     "guarded floating synthesis and tiny operator checks are numerical evidence, not formal certificates",
                     "physical code, factory, controller and interface assumptions are conditional and uncalibrated",
                     "common native factory cost identity does not certify an identical physical factory geometry",
                     "native factory internal preparations/readouts are abstracted by its time/space/output-error ISA, not exported as explicit operation counts",
                     "measured quantum hardware optimizer/scoring/communications/decoder latency is excluded",
                     "all VQE quality labels and negative controls remain unchanged; Classical C4 reference coverage only"])


def _check_recipe(inputs, recipe):
    if recipe != compiled_recipe(inputs):
        raise ValueError("Compilation recipe changed after selection/freeze")


def compiled_source_classes(inputs, recipe) -> list[dict]:
    """Every initial/incumbent/proposal/holdout batch, grouped by exact source."""
    _check_recipe(inputs, recipe)
    parent, _ = _producers()
    classes = {}
    for phase, calls in (("training", inputs["run"]["evaluations"]), ("holdout", [inputs["run"]["validation"]])):
        for index, call in enumerate(calls):
            if len(call["groups"]) != 1 or call["groups"][0]["group"] != "all_z":
                raise ValueError("Selected QAOA must retain every joint all-Z group")
            group = call["groups"][0]
            source = parent.diagnostic_group_circuit(inputs["case"], call["parameters"], "all_z")
            original = qasm3.dumps(source)
            key = _sha(original.encode())
            if dict(source.count_ops()) != group["circuit_gate_counts"]:
                raise ValueError("Source circuit disagrees with immutable sampled batch")
            record = classes.setdefault(key, dict(class_id=key, parameters=call["parameters"], shots=0,
                                                  source_qasm=original, batches=[]))
            if record["source_qasm"] != original or record["parameters"] != call["parameters"]:
                raise ValueError("Exact source hash collision")
            record["shots"] += group["shots"]
            record["batches"].append(dict(phase=phase, evaluation_index=index, shots=group["shots"],
                                         role=call["role"], accepted=call.get("accepted", False), group="all_z"))
    result = list(classes.values())
    if (len(result) != 54 or sum(c["shots"] for c in result) != 82_032
            or sum(len(c["batches"]) for c in result) != 110):
        raise ValueError("Source partition omits the fixed complete workload")
    return result


def circuit_operations(circuit) -> list:
    return [[item.operation.name, [circuit.find_bit(q).index for q in item.qubits],
             [circuit.find_bit(c).index for c in item.clbits], [float(p) for p in item.operation.params]]
            for item in circuit.data]


def circuit_from_operations(operations, n_qubits: int, global_phase: float = 0.0):
    """Rebuild saved primitive circuits without an optional QASM importer."""
    if n_qubits != 4 or not math.isfinite(global_phase) or len(operations) > 10_000:
        raise ValueError("Circuit reconstruction exceeds frozen width/operation bounds")
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.global_phase = global_phase
    fixed = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "reset"}
    for name, qubits, clbits, params in operations:
        if any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < n_qubits for q in qubits+clbits):
            raise ValueError("Invalid primitive wire or classical mapping")
        if name in fixed and len(qubits) == 1 and not clbits and not params:
            getattr(circuit, name)(qubits[0])
        elif name in {"rx", "ry", "rz"} and len(qubits) == 1 and len(params) == 1 and not clbits:
            if not math.isfinite(params[0]):
                raise ValueError("Saved rotation angles must be finite")
            getattr(circuit, name)(params[0], qubits[0])
        elif name == "cx" and len(qubits) == 2 and len(set(qubits)) == 2 and not params and not clbits:
            circuit.cx(*qubits)
        elif name == "measure" and len(qubits) == len(clbits) == 1 and not params:
            circuit.measure(qubits[0], clbits[0])
        elif name == "barrier" and not clbits and not params:
            circuit.barrier(*qubits)
        else:
            raise ValueError("Unsupported saved primitive operation")
    return circuit


def _primitive_matrix(name, params):
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1]).astype(complex)
    fixed = {"h": np.array([[1, 1], [1, -1]], complex)/math.sqrt(2), "x": x, "y": y, "z": z,
             "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
             "t": np.diag([1, np.exp(1j*math.pi/4)]), "tdg": np.diag([1, np.exp(-1j*math.pi/4)])}
    if name in fixed:
        return fixed[name]
    if name in {"rx", "ry", "rz"}:
        angle = params[0]
        pauli = {"rx": x, "ry": y, "rz": z}[name]
        return math.cos(angle/2)*identity-1j*math.sin(angle/2)*pauli
    raise ValueError("Non-unitary or unknown primitive in coherent segment")


def independent_instrument_operator(operations, global_phase: float) -> np.ndarray:
    """Independent tiny full coherent matrix, with output permutation undone."""
    indices = np.arange(16)
    matrix = np.eye(16, dtype=complex)
    initialized, measured, outputs = set(), False, {}
    gate_cache = {}
    for name, wires, clbits, params in operations:
        if name == "barrier":
            continue
        if name == "reset":
            if wires[0] in initialized or measured or initialized != set(range(len(initialized))):
                raise ValueError("Only initial canonical resets are allowed in this tiny coherent audit")
            initialized.add(wires[0])
            continue
        if name == "measure":
            measured = True
            outputs[clbits[0]] = wires[0]
            continue
        if measured or initialized != set(range(4)):
            raise ValueError("Quantum operations must lie after all input resets and before readout")
        if name == "cx":
            control, target = wires
            permutation = indices.copy()
            permutation[(indices & (1 << control)) != 0] ^= 1 << target
            matrix = matrix[permutation]
        else:
            key = (name, tuple(params))
            if key not in gate_cache:
                gate_cache[key] = _primitive_matrix(name, params)
            gate = gate_cache[key]
            zeros = indices[(indices & (1 << wires[0])) == 0]
            ones = zeros ^ (1 << wires[0])
            a, b = matrix[zeros].copy(), matrix[ones].copy()
            matrix[zeros], matrix[ones] = gate[0, 0]*a+gate[0, 1]*b, gate[1, 0]*a+gate[1, 1]*b
    if set(outputs) != set(range(4)) or set(outputs.values()) != set(range(4)):
        raise ValueError("Every output bit must map to one distinct measured wire")
    permutation = [sum(((assignment >> c) & 1) << outputs[c] for c in range(4)) for assignment in range(16)]
    return np.exp(1j*global_phase)*matrix[permutation]


def compile_source_class(source_class, recipe) -> dict:
    """One actual native synthesis, numerical routing/operator/readout audits."""
    parent, backend = _producers()
    inputs = load_compiled_inputs(_runtime_root)
    _check_recipe(inputs, recipe)
    expected = {c["class_id"]: c for c in compiled_source_classes(inputs, recipe)}
    if source_class != expected.get(source_class.get("class_id")):
        raise ValueError("Compilation source differs from the frozen source partition")
    started = perf_counter()
    measured = parent.diagnostic_group_circuit(inputs["case"], source_class["parameters"], "all_z")
    source = QuantumCircuit(4, 4, name="reset-complete recorded QAOA group")
    source.reset(range(4))
    source.compose(measured, inplace=True)
    if len(source.data) > recipe["bounds"]["source_operations"]:
        raise ValueError("Source materialization cap exceeded")
    limits = backend.InstrumentLimits(max_qubits=4, source_operations=128, synthesized_operations=10_000,
                                      validation_operations=10_000, validation_branches=256)
    native_function = backend.gridsynth_unitary
    queries = 0
    def bounded_synthesis(*args, **kwargs):
        nonlocal queries
        queries += 1
        if queries > recipe["bounds"]["synthesis_native_calls_per_class"]:
            raise RuntimeError("Native synthesis-query cap exceeded")
        return native_function(*args, **kwargs)
    backend.gridsynth_unitary = bounded_synthesis
    try:
        value = backend.synthesize_instrument(source, recipe["budget"]["per_preparation_operator_allowance"],
                                              limits=limits, preserve_full_unitary=True)
    finally:
        backend.gridsynth_unitary = native_function
    if (len(value.routed.data) > recipe["bounds"]["routed_operations"]
            or value.report["rotation_occurrences"] != 16):
        raise ValueError("Routing/rotation occurrence cap or source invariant failed")
    operations = [circuit_operations(c) for c in (source, value.routed, value.circuit)]
    phases = [float(c.global_phase) for c in (source, value.routed, value.circuit)]
    matrices = [independent_instrument_operator(o, p) for o, p in zip(operations, phases)]
    route_error = float(np.linalg.norm(matrices[0]-matrices[1], 2))
    realization_error = float(np.linalg.norm(matrices[0]-matrices[2], 2))
    eta = value.report["guarded_numerical_operator_error_upper_estimate"]
    if route_error > 1e-11 or realization_error > eta+2e-11:
        raise RuntimeError("Actual routed/Clifford+T coherent matrices failed numerical checks")
    probabilities = [np.abs(u[:, 0])**2 for u in matrices]
    reference = parent.qaoa_layers_reference(4, inputs["case"]["edges"], source_class["parameters"])
    independent_probs = np.abs(reference["reference_state"])**2
    if np.max(np.abs(probabilities[0]-independent_probs)) > 1e-11:
        raise RuntimeError("Augmented source disagrees with immutable independent QAOA reference")
    scores = np.asarray(reference["cut_values"], dtype=float)
    actual_tv = float(np.abs(probabilities[0]-probabilities[2]).sum()/2)
    source_mean, compiled_mean = (float(p @ scores) for p in (probabilities[0], probabilities[2]))
    if actual_tv > eta+2e-11 or abs(compiled_mean-source_mean)/4 > eta+2e-11:
        raise RuntimeError("Actual compiled readout exceeds conditional numerical realization bound")
    # Independently exercise the frozen ensemble checker on the final class;
    # all other classes have the stronger full coherent matrix/readout audit.
    ensemble = None
    if any(b["phase"] == "holdout" for b in source_class["batches"]):
        ensemble = backend.simulate_scored_trial(value.circuit, range(4), scores, limits=limits)
        if abs(ensemble["mean"]-compiled_mean) > 1e-10:
            raise RuntimeError("Frozen reset/readout ensemble disagrees with independent primitive audit")
        ensemble = _json_values(ensemble)
    return dict(schema_version=1, class_id=source_class["class_id"], parameters=source_class["parameters"],
        shots=source_class["shots"], batches=source_class["batches"], original_source_qasm=source_class["source_qasm"],
        source_qasm=qasm3.dumps(source), routed_qasm=qasm3.dumps(value.routed), clifford_t_qasm=qasm3.dumps(value.circuit),
        source_operations=operations[0], routed_operations=operations[1], clifford_t_operations=operations[2],
        source_global_phase=phases[0], routed_global_phase=phases[1], clifford_t_global_phase=phases[2],
        report=value.report, checks=dict(routed_full_operator_error=route_error, actual_full_operator_error=realization_error,
            source_probabilities=probabilities[0].tolist(), compiled_probabilities=probabilities[2].tolist(),
            source_mean_cut=source_mean, compiled_mean_cut=compiled_mean, actual_probability_tv=actual_tv,
            actual_ratio_bias=abs(compiled_mean-source_mean)/4, frozen_final_ensemble=ensemble,
            scope="independent four-wire coherent matrix/readout checks; numerical evidence, not formal certificates"),
        timing=dict(compilation_and_validation_wall_seconds=perf_counter()-started),
        query_counters=dict(native_synthesis_calls=queries), physical_status="not_estimated")


def compiled_physical_classes(compiled, recipe):
    inputs = load_compiled_inputs(_runtime_root)
    _check_recipe(inputs, recipe)
    expected = compiled_source_classes(inputs, recipe)
    _, backend = _producers()
    if len(compiled) != 54 or [c["class_id"] for c in compiled] != [c["class_id"] for c in expected]:
        raise ValueError("Physical workload must include every exact source class")
    values = []
    for original, c in zip(expected, compiled):
        if any(c[key] != original[key] for key in ("class_id", "parameters", "shots", "batches")) or c["original_source_qasm"] != original["source_qasm"]:
            raise ValueError("Physical class changed the frozen source or repetitions")
        source = circuit_from_operations(c["source_operations"], 4, c["source_global_phase"])
        routed = circuit_from_operations(c["routed_operations"], 4, c["routed_global_phase"])
        circuit = circuit_from_operations(c["clifford_t_operations"], 4, c["clifford_t_global_phase"])
        values.append(dict(class_id=c["class_id"], shots=c["shots"],
                           synthesized=backend.SynthesizedInstrument(source, routed, circuit, c["report"])))
    return values


def run_compiled_physical(compiled, recipe, cache_dir):
    _check_recipe(load_compiled_inputs(_runtime_root), recipe)
    _, backend = _producers()
    classes = compiled_physical_classes(compiled, recipe)
    t_states = sum(c["shots"]*sum(op[0] in ("t", "tdg") for op in c["clifford_t_operations"]) for c in compiled)
    return run_seventh_physical_workload(classes, backend=backend, cache_dir=Path(cache_dir),
                                        expected_total_preparations=82_032, expected_total_t_states=t_states,
                                        caps=SeventhPhysicalCaps())


def compiled_aggregate(inputs, recipe, compiled, physical):
    _check_recipe(inputs, recipe)
    source_classes = compiled_source_classes(inputs, recipe)
    if [c["class_id"] for c in compiled] != [c["class_id"] for c in source_classes]:
        raise ValueError("Compiled source partition is incomplete or reordered")
    source_counts, augmented_counts, routed_counts, ct_counts = (Counter() for _ in range(4))
    realization, actual_realization, batches, total_operations, total_queries = 0.0, 0.0, [], 0, 0
    for original, c in zip(source_classes, compiled):
        if c["shots"] != original["shots"] or c["batches"] != original["batches"]:
            raise ValueError("Compiled batch multiplicities changed")
        for target, counts in zip((source_counts, augmented_counts, routed_counts, ct_counts),
                                 (c["report"]["source_counts"], c["report"]["source_counts"], c["report"]["routed_counts"], c["report"]["clifford_t_counts"])):
            target.update({name: int(n)*c["shots"] for name, n in counts.items() if name != "barrier"})
        # Original immutable source omits explicitly added reset events.
        source_counts.subtract({"reset": 4*c["shots"]})
        realization += c["shots"]*c["report"]["guarded_numerical_operator_error_upper_estimate"]
        actual_realization += c["shots"]*c["checks"]["actual_probability_tv"]
        total_operations += len(c["clifford_t_operations"])
        total_queries += c["query_counters"]["native_synthesis_calls"]
        batches.extend(dict(class_id=c["class_id"], **batch) for batch in c["batches"])
    source_counts.pop("reset", None)
    if (dict(source_counts) != recipe["planned"]["source_weighted_counts"]
            or realization > 0.005+1e-12 or total_operations > 540_000 or total_queries > 3456):
        raise ValueError("Complete source/realization/materialization budget reconciliation failed")
    final = next(c for c in compiled if any(b["phase"] == "holdout" for b in c["batches"]))
    bias = final["report"]["guarded_numerical_operator_error_upper_estimate"]
    radius = recipe["budget"]["conditional_sampling_radius"]
    estimate = recipe["budget"]["saved_estimate"]
    lower = estimate-radius-bias
    if lower < 0.8 or radius+bias > 0.05:
        raise ValueError("Conditional selected-state target/precision allocation failed")
    shots_by_physical = {item["physical_class_id"]: item["shots"] for item in physical["physical_classes"]}
    macro_totals = {}
    for name in ("selected_minimum_qubits", "selected_minimum_runtime"):
        total = Counter()
        for chosen in physical[name]["per_physical_class"]:
            shots = shots_by_physical[chosen["physical_class_id"]]
            for row in chosen["point"]["macro_ledger_per_complete_attempt"]:
                for field in ("readouts", "preparations", "decisions"):
                    total[field] += row["count"]*row[field]*shots
        macro_totals[name] = dict(total, scope="complete serialized data/workspace/injection/control macro ledger; native factory-internal operation counts are abstracted into its time/space/output-error ISA")
    result = dict(schema_version=1, status="computed_actual_compiled_fixed_replay_conditional_physical_model",
        recipe=recipe, bindings=PINS, run_id=RUN_ID, source_classes=compiled, physical=physical,
        batches=batches, selected_archived_target_status="supported", negative_controls=inputs["negative_controls"],
        resources=dict(source_weighted_counts=dict(source_counts), augmented_weighted_counts=dict(augmented_counts),
            routed_weighted_counts=dict(routed_counts), clifford_t_weighted_counts=dict(ct_counts),
            preparations=82_032, original_source_qubit_readouts=328_128, explicit_input_initializations=328_128,
            total_source_classes=54, measured_groups=110, actual_materialized_ct_operations=total_operations,
            native_synthesis_calls=total_queries, actual_t_states=ct_counts["t"]+ct_counts["tdg"],
            physical_macro_totals=macro_totals,
            source_unitary_gates=2_953_152, realization_guarded_static_transcript_tv=realization,
            actual_numerical_static_transcript_tv_sum=actual_realization),
        conditional_readout=dict(sampling_family_failure=0.04, sampling_group_failure=0.04/48,
            family_group_count=48, fixed_final_shots=16_384, compiled_vs_source_ratio_bias_upper_estimate=bias,
            normalized_precision_radius=radius+bias, saved_ideal_ratio_estimate=estimate,
            illustrative_interval_if_compiled_estimator_had_saved_value=[estimate-radius-bias, estimate+radius+bias],
            lower_if_the_compiled_estimator_had_the_saved_value=lower,
            source_exact_ratio=final["checks"]["source_mean_cut"]/4,
            compiled_exact_ratio=final["checks"]["compiled_mean_cut"]/4,
            future_supported_interval_lower_on_concentration_event=final["checks"]["compiled_mean_cut"]/4-2*radius-bias,
            scope="prospective fixed-state concentration plus numerical mean-shift bound; saved counts were ideal and were not rerun on the compiled circuit"),
        classical_work=dict(archived_optimizer_wall_seconds=inputs["run"]["optimizer_wall_seconds"],
            archived_objective_wall_seconds=inputs["run"]["objective_wall_seconds"],
            archived_control_wall_seconds=inputs["run"]["classical_control_seconds"],
            actual_synthesis_validation_wall_seconds=sum(c["timing"]["compilation_and_validation_wall_seconds"] for c in compiled),
            quantum_hardware_optimizer_scoring_latency=None,
            exclusion="CPU archive/synthesis/verification timings are measured separately; no hardware optimizer, scoring, decoder or communication latency measured"),
        limitations=recipe["limitations"])
    validate_compiled_record(result)
    return result


def validate_compiled_record(record):
    """Structural/hash/budget replay; no new native synthesis or search."""
    recipe, classes = record["recipe"], record["source_classes"]
    inputs = load_compiled_inputs(_runtime_root)
    _check_recipe(inputs, recipe)
    expected = compiled_source_classes(inputs, recipe)
    parent, _ = _producers()
    if ([c["class_id"] for c in classes] != [c["class_id"] for c in expected]
            or record["negative_controls"] != inputs["negative_controls"]):
        raise ValueError("Saved compiled source partition or negative controls changed")
    if record["bindings"] != PINS or len(classes) != 54 or sum(c["shots"] for c in classes) != 82_032:
        raise ValueError("Saved compiled workload identity or completeness changed")
    if (record["resources"]["measured_groups"] != 110 or record["selected_archived_target_status"] != "supported"
            or sum(len(c["batches"]) for c in classes) != 110):
        raise ValueError("Saved target status or group coverage changed")
    if not math.isclose(sum((0.04, 0.005, 0.0025, 0.0025)), recipe["budget"]["reserved_total"], abs_tol=1e-15):
        raise ValueError("Combined failure/distance allocation changed")
    weighted, total_ops, total_queries, realization = Counter(), 0, 0, 0.0
    ledgers = {name: Counter() for name in ("source", "augmented", "routed", "clifford_t")}
    scores = np.array([sum(((z >> u) ^ (z >> v)) & 1 for u, v in inputs["case"]["edges"]) for z in range(16)], dtype=float)
    for original, c in zip(expected, classes):
        if any(c[key] != original[key] for key in ("class_id", "parameters", "shots", "batches")) or c["original_source_qasm"] != original["source_qasm"]:
            raise ValueError("Saved class metadata changed its immutable source binding")
        if _sha(c["original_source_qasm"].encode()) != c["class_id"]:
            raise ValueError("Saved original source hash changed")
        augmented = QuantumCircuit(4, 4, name="reset-complete recorded QAOA group")
        augmented.reset(range(4))
        augmented.compose(parent.diagnostic_group_circuit(inputs["case"], c["parameters"], "all_z"), inplace=True)
        if (c["source_operations"] != circuit_operations(augmented) or c["source_qasm"] != qasm3.dumps(augmented)
                or c["source_global_phase"] != float(augmented.global_phase)):
            raise ValueError("Saved augmented source changed its immutable angles or input preparation")
        for prefix in ("source", "routed", "clifford_t"):
            operations = c[f"{prefix}_operations"]
            circuit = circuit_from_operations(operations, 4, c[f"{prefix}_global_phase"])
            if dict(circuit.count_ops()) != c["report"][{"source":"source_counts", "routed":"routed_counts", "clifford_t":"clifford_t_counts"}[prefix]]:
                raise ValueError("Saved operations/counts disagree")
            if sum(op[0] == "reset" for op in operations) != 4 or sum(op[0] == "measure" for op in operations) != 4:
                raise ValueError("Saved circuit lost explicit input/reset/readout events")
            if qasm3.dumps(circuit) != c[f"{prefix}_qasm"]:
                raise ValueError("Saved QASM disagrees with its primitive operation stream")
            initialized, readout, output_bits, output_wires = set(), False, [], []
            for name, wires, clbits, params in operations:
                if name == "barrier":
                    continue
                if name == "reset":
                    if readout or wires[0] in initialized:
                        raise ValueError("Saved reset schedule is not canonical initialization")
                    initialized.add(wires[0])
                elif name == "measure":
                    readout = True
                    output_bits.extend(clbits)
                    output_wires.extend(wires)
                elif readout or initialized != set(range(4)):
                    raise ValueError("Saved quantum operations crossed initialization/readout boundaries")
                if prefix != "source" and name == "cx" and abs(wires[0]-wires[1]) != 1:
                    raise ValueError("Saved routed operation is nonlocal")
            if sorted(output_bits) != list(range(4)) or sorted(output_wires) != list(range(4)):
                raise ValueError("Saved output measurement map changed")
            ledger_name = "augmented" if prefix == "source" else prefix
            ledgers[ledger_name].update({name: int(n)*c["shots"] for name, n in circuit.count_ops().items() if name != "barrier"})
        ledgers["source"].update({name: int(n)*c["shots"] for name, n in augmented.count_ops().items() if name not in ("barrier", "reset")})
        if (c["report"]["guarded_numerical_operator_error_upper_estimate"] > recipe["budget"]["per_preparation_operator_allowance"]
                or set(op[0] for op in c["clifford_t_operations"])-{"h","x","y","z","s","sdg","t","tdg","cx","reset","measure","barrier"}):
            raise ValueError("Actual compiled circuit left Clifford+T or realization budget")
        eta = c["report"]["guarded_numerical_operator_error_upper_estimate"]
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("Saved numerical realization bound is invalid")
        matrices = [independent_instrument_operator(c[f"{stage}_operations"], c[f"{stage}_global_phase"])
                    for stage in ("source", "routed", "clifford_t")]
        route_error = float(np.linalg.norm(matrices[0]-matrices[1], 2))
        actual_error = float(np.linalg.norm(matrices[0]-matrices[2], 2))
        source_prob, compiled_prob = np.abs(matrices[0][:, 0])**2, np.abs(matrices[2][:, 0])**2
        if (route_error > 1e-11 or actual_error > eta+2e-11
                or not np.allclose(c["checks"]["source_probabilities"], source_prob, atol=2e-11, rtol=0)
                or not np.allclose(c["checks"]["compiled_probabilities"], compiled_prob, atol=2e-11, rtol=0)
                or abs(c["checks"]["actual_full_operator_error"]-actual_error) > 2e-11):
            raise ValueError("Saved actual compiled coherent/readout checks do not replay")
        expected_metrics = dict(source_mean_cut=float(source_prob @ scores), compiled_mean_cut=float(compiled_prob @ scores),
                                actual_probability_tv=float(np.abs(source_prob-compiled_prob).sum()/2),
                                actual_ratio_bias=abs(float((source_prob-compiled_prob) @ scores))/4)
        if any(not math.isclose(c["checks"][name], number, abs_tol=2e-11, rel_tol=1e-11) for name, number in expected_metrics.items()):
            raise ValueError("Saved numerical readout scalars do not replay from probabilities")
        weighted.update({name: int(n)*c["shots"] for name, n in c["report"]["clifford_t_counts"].items() if name != "barrier"})
        total_ops += len(c["clifford_t_operations"])
        total_queries += c["query_counters"]["native_synthesis_calls"]
        realization += c["shots"]*c["report"]["guarded_numerical_operator_error_upper_estimate"]
    if (dict(weighted) != record["resources"]["clifford_t_weighted_counts"]
            or total_ops != record["resources"]["actual_materialized_ct_operations"] or total_ops > 540_000
            or total_queries != record["resources"]["native_synthesis_calls"] or total_queries > 3456
            or realization > 0.005+1e-12):
        raise ValueError("Saved complete compiled accounting/caps changed")
    for name in ledgers:
        if dict(ledgers[name]) != record["resources"][f"{name}_weighted_counts"]:
            raise ValueError("Saved source/augmented/routed/compiled weighted ledger changed")
    physical = record["physical"]
    if (physical["status"] != "complete_conditional_recorded_workload_common_configuration"
            or physical["complete_preparations"] != 82_032 or physical["actual_magic_states"] != weighted["t"]+weighted["tdg"]
            or not physical["common_configurations"]):
        raise ValueError("Saved physical workload is not complete under one common configuration")
    for choice in (physical["selected_minimum_qubits"], physical["selected_minimum_runtime"]):
        if choice["total_preparations"] != 82_032 or choice["whole_physical_failure_model"] > 0.0025 or choice["whole_factory_availability_failure_bound"] > 0.0025:
            raise ValueError("Saved common configuration exceeded the whole-workload budget")


# SPDX-License-Identifier: MIT
"""Owned wrapper for a fixed recorded Clifford+T workload.

This fragment delegates the physical scenario to the byte-pinned instrument
backend. Runtime wrappers only count bounded native calls and preserve native
factory ISA identities; they do not edit installed or shared source files.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time


SEVENTH_INSTRUMENT_SOURCE_SHA256 = "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"
SEVENTH_NATIVE_SOURCE_SHA256 = {
    "models/factories/_round_based.py": "c5fcb478c86534cf9888bc8a9b9d3e5a733355bf92c41072228d99fdc0d19110",
    "models/qec/_surface_code.py": "3530f92f328e325d6d0c3e245f8688c11956fa6f63b1e25455b5d14cd844ed35",
    "_isa_enumeration.py": "f35a4248c0d29174daac033d12f8296ad6c3b5a203a4866703d763a5b65aa97e",
}


@dataclass(frozen=True)
class SeventhPhysicalCaps:
    top_level_calls: int = 64
    factory_frontier_queries: int = 1536
    factory_frontier_candidates: int = 52224
    factory_pipeline_candidates: int = 52224
    native_trace_queries: int = 52224
    wall_seconds: float = 120.0

    def __post_init__(self):
        maxima = {"top_level_calls": 64, "factory_frontier_queries": 1536,
                  "factory_frontier_candidates": 52224, "factory_pipeline_candidates": 52224,
                  "native_trace_queries": 52224}
        for name, maximum in maxima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        if isinstance(self.wall_seconds, bool) or not math.isfinite(self.wall_seconds) or not 0 < self.wall_seconds <= 120:
            raise ValueError("physical wall allowance must lie in (0,120] seconds")


class SeventhPhysicalLimit(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _seventh_canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seventh_digest(value):
    return hashlib.sha256(_seventh_canonical(value).encode()).hexdigest()


def _seventh_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seventh_t_count(events):
    return sum(value for key, value in events.items() if key.startswith(("t_q", "tdg_q")))


def _seventh_downward_float(value, *, decimal_only=False):
    """Round down in the interpretation consumed by the frozen backend.

    Physical arithmetic consumes binary floats; factory retry allocation is
    explicitly parsed from str(budget) by the frozen exact-Fraction helper.
    """
    result = float(value)
    while Fraction(str(result)) > value or (not decimal_only and Fraction.from_float(result) > value):
        result = math.nextafter(result, 0.0)
    if not 0 < result < 1:
        raise ValueError("Positive failure allocation is outside the float input domain")
    return result


def prepare_seventh_physical_workload(compiled_classes, *, backend, expected_total_preparations,
                                      expected_total_t_states=None, caps=None):
    """Preflight and allocate a complete static workload; perform no native query.

    Each item is {class_id, synthesized, shots}. The caller separately retains
    every original/augmented/compiled QASM and immutable transcript binding.
    """
    caps = caps or SeventhPhysicalCaps()
    if not isinstance(caps, SeventhPhysicalCaps):
        raise ValueError("caps must be a SeventhPhysicalCaps instance")
    source = Path(backend.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SEVENTH_INSTRUMENT_SOURCE_SHA256:
        raise ValueError("Instrument backend bytes differ from the frozen baseline")
    _seventh_positive_integer(expected_total_preparations, "expected total preparations")
    groups, class_map, seen = {}, {}, set()
    for item in compiled_classes:
        key = item["class_id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Compiled class IDs must be distinct nonempty strings")
        seen.add(key)
        shots = _seventh_positive_integer(item["shots"], "class preparations")
        value = item["synthesized"]
        bound_fields = value.report.get("limits")
        if not isinstance(bound_fields, dict):
            raise ValueError("Every compiled class must retain explicit instrument limits")
        limits = backend.InstrumentLimits(**bound_fields)
        backend._source_check(value.source, operation_limit=limits.source_operations, max_qubits=limits.max_qubits)
        backend._source_check(value.circuit, operation_limit=limits.synthesized_operations, max_qubits=limits.max_qubits)
        names = set(value.circuit.count_ops())
        if names - backend.CLIFFORD_T - {"reset", "measure", "barrier"}:
            raise ValueError("Physical input contains a non-Clifford+T or opaque operation")
        if not value.report.get("actual_synthesis_sequences_generated"):
            raise ValueError("A mean-fit synthesis proxy is not a physical input")
        if value.source.num_qubits != value.circuit.num_qubits:
            raise ValueError("Compilation changed the reserved logical width")
        events = dict(sorted(backend._event_counts(value.circuit).items()))
        # The frozen serializer's costs depend exactly on width and these
        # counts, including T/T-dagger locations; source ordering is retained
        # separately and is not asserted equivalent by this grouping.
        signature = {"logical_qubits": value.circuit.num_qubits, "event_counts": events,
                     "backend_sha256": SEVENTH_INSTRUMENT_SOURCE_SHA256}
        physical_id = _seventh_digest(signature)
        if physical_id not in groups:
            groups[physical_id] = {"physical_class_id": physical_id, "signature": signature,
                                   "synthesized": value, "shots": 0, "compiled_class_ids": []}
        groups[physical_id]["shots"] += shots
        groups[physical_id]["compiled_class_ids"].append(key)
        class_map[key] = {"physical_class_id": physical_id, "shots": shots,
                          "signature_sha256": physical_id}
    if not groups or sum(group["shots"] for group in groups.values()) != expected_total_preparations:
        raise ValueError("Physical classes omit or duplicate the complete preparation workload")
    if len(groups) > caps.top_level_calls:
        raise SeventhPhysicalLimit("Exact physical-ledger classes exceed the frozen top-level cap", {
            "status": "preflight_cap_exceeded", "physical_classes": len(groups), "caps": asdict(caps)})
    total_t = sum(_seventh_t_count(group["signature"]["event_counts"]) * group["shots"] for group in groups.values())
    if expected_total_t_states is not None and total_t != expected_total_t_states:
        raise ValueError("Whole-workload magic-state count disagrees with the complete compiled ledger")
    if total_t < 1:
        raise ValueError("This selected workload requires a nonzero actual magic-state count")
    profile = backend.InstrumentProfile()  # Frozen p=1e-4 scenario, all other fields unchanged.
    physical_budget, factory_budget = 0.0025, 0.0025
    budget_fraction = Fraction("0.0025")
    retry_cap = backend._factory_retry_cap(total_t, profile.factory_pipeline_failure_bound, factory_budget)
    for group in groups.values():
        group_t = _seventh_t_count(group["signature"]["event_counts"]) * group["shots"]
        group["total_t_states"] = group_t
        physical_fraction = budget_fraction * group["shots"] / expected_total_preparations
        factory_fraction = budget_fraction * group_t / total_t
        group["physical_failure_budget"] = _seventh_downward_float(physical_fraction)
        group["physical_failure_allocation_exact"] = str(physical_fraction)
        # Proportional T allocation gives the same exact global retry cap. A
        # zero-T class consumes no factory budget and has no active supply.
        group["factory_availability_budget"] = (_seventh_downward_float(factory_fraction, decimal_only=True) if group_t else 1e-300)
        group["factory_availability_allocation_exact"] = str(factory_fraction)
        if group_t and backend._factory_retry_cap(group_t, profile.factory_pipeline_failure_bound,
                                                  group["factory_availability_budget"]) != retry_cap:
            raise ValueError("Floating factory allocation crosses an exact retry-cap boundary")
    return {"groups": list(groups.values()), "compiled_class_mapping": class_map,
            "total_preparations": expected_total_preparations, "total_t_states": total_t,
            "whole_physical_failure_budget": physical_budget,
            "whole_factory_availability_budget": factory_budget,
            "global_factory_retry_cap": retry_cap, "profile": asdict(profile), "caps": caps,
            "grouping_scope": "identical frozen serialized event-cost ledger only; no unitary/topology equivalence claim"}


def _seventh_instruction_record(instruction):
    """Exact fixed-arity native instruction values, with IEEE identities."""
    if instruction.arity != 1:
        raise ValueError("Native factory ISA contains an unexpected non-single-state instruction")
    values = {"id": int(instruction.id), "encoding": int(instruction.encoding), "arity": instruction.arity}
    for name in ("time", "space", "error_rate"):
        value = getattr(instruction, name)()
        if isinstance(value, int):
            values[name] = {"integer": value}
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Native factory instruction has a nonfinite model value")
            values[name] = {"float_hex": number.hex()}
    values["expected_time_ns"] = int(instruction.expect_time())
    values["expected_space"] = int(instruction.expect_space())
    values["expected_output_error_hex"] = float(instruction.expect_error_rate()).hex()
    return values


class _SeventhNativeAudit:
    def __init__(self, caps):
        self.caps, self.started = caps, time.perf_counter()
        self.counts = Counter()
        self.factory_queries, self.trace_queries = [], []
        self.current_group, self.current_factory_query = None, None
        self.supply_by_t_signature = {}

    def snapshot(self, status="running"):
        return {"status": status, "caps": asdict(self.caps), "counts": dict(self.counts),
                "wall_seconds": time.perf_counter() - self.started,
                "factory_frontier_queries": self.factory_queries, "native_trace_queries": self.trace_queries,
                "instrumentation_scope": "top-level calls, factory frontiers/pipeline candidates, and direct native Trace.estimate calls"}

    def check(self):
        if time.perf_counter() - self.started > self.caps.wall_seconds:
            raise SeventhPhysicalLimit("Physical workflow exceeded its frozen wall allowance", self.snapshot("wall_cap_exceeded"))

    def charge(self, name):
        self.check()
        maximum = getattr(self.caps, name)
        if self.counts[name] >= maximum:
            raise SeventhPhysicalLimit(f"Physical workflow exceeded {name}", self.snapshot("query_cap_exceeded"))
        self.counts[name] += 1


@contextmanager
def _seventh_bounded_native_queries(audit):
    """Reversible runtime instrumentation; no shared/installed source writes.

    Use in a dedicated single-threaded worker process. An outer subprocess
    timeout remains necessary to interrupt a native call that does not return.
    """
    import qdk.qre as qre
    import qdk.qre.models as models
    import qdk.qre.models.factories._round_based as round_based
    from qdk.qre.instruction_ids import T
    base = Path(qre.__file__).resolve().parent
    if importlib.metadata.version("qdk") != "1.34.0":
        raise ValueError("Physical native query audit requires QDK 1.34.0")
    for name, digest in SEVENTH_NATIVE_SOURCE_SHA256.items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned native enumeration source changed: {name}")
    original_trace, original_factory = qre.Trace, models.RoundBasedFactory
    original_pipeline_descriptor = round_based._Pipeline.__dict__["try_create"]
    original_pipeline_bound = round_based._Pipeline.try_create

    class FactoryQuery:
        def __init__(self, inner, kwargs):
            self.inner = inner
            code_query = kwargs.get("code_query")
            code_kwargs = getattr(code_query, "kwargs", None)
            if not isinstance(code_kwargs, dict) or not isinstance(code_kwargs.get("distance"), int):
                raise ValueError("Factory query is not the frozen fixed-distance surface-code query")
            self.configuration = {key: value for key, value in code_kwargs.items() if key in
                                  ("distance", "code_cycle_override", "error_correction_threshold")}

        def enumerate(self, ctx):
            audit.charge("factory_frontier_queries")
            record = {"index": len(audit.factory_queries), "physical_class_id": audit.current_group,
                      "configuration": self.configuration, "pipeline_candidates": 0, "frontier_candidates": []}
            audit.factory_queries.append(record)
            previous = audit.current_factory_query
            audit.current_factory_query = record
            try:
                for supply in self.inner.enumerate(ctx):
                    audit.charge("factory_frontier_candidates")
                    if len(record["frontier_candidates"]) >= 34:
                        raise SeventhPhysicalLimit("Native factory frontier exceeds the pinned 34-candidate domain", audit.snapshot("candidate_cap_exceeded"))
                    instructions = [_seventh_instruction_record(inst) for inst in sorted(supply, key=lambda inst: inst.id)]
                    if len(instructions) != 1 or instructions[0]["id"] != int(T):
                        raise ValueError("Native factory frontier contains an unexpected ISA")
                    t_record = _seventh_instruction_record(supply[T])
                    identity = {"native_factory_isa": instructions,
                                "native_factory_isa_sha256": _seventh_digest(instructions),
                                "native_t_instruction": t_record,
                                "native_t_instruction_sha256": _seventh_digest(t_record),
                                "surface_code_query": self.configuration,
                                "native_sources": SEVENTH_NATIVE_SOURCE_SHA256,
                                "identity_scope": "exact native factory ISA/T cost-model values; no distillation-layout certificate"}
                    record["frontier_candidates"].append(identity)
                    audit.supply_by_t_signature.setdefault(identity["native_t_instruction_sha256"], []).append(identity)
                    yield supply
            finally:
                audit.current_factory_query = previous

    class Factory:
        @staticmethod
        def q(**kwargs):
            return FactoryQuery(original_factory.q(**kwargs), kwargs)

    class Trace:
        def __init__(self, *args, **kwargs):
            self.inner = original_trace(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def estimate(self, isa, *args, **kwargs):
            audit.charge("native_trace_queries")
            native_t = isa.get(T)
            t_signature = _seventh_digest(_seventh_instruction_record(native_t)) if native_t is not None else None
            record = {"index": len(audit.trace_queries), "physical_class_id": audit.current_group,
                      "accounting_trace_sha256": hashlib.sha256(self.inner.to_json().encode()).hexdigest(),
                      "native_t_instruction_sha256": t_signature, "status": "query_started"}
            audit.trace_queries.append(record)
            result = self.inner.estimate(isa, *args, **kwargs)
            audit.check()
            if result is None:
                record["status"] = "no_feasible_native_result"
            else:
                record.update(status="native_result_returned", runtime_ns=int(result.runtime),
                              peak_physical_qubits=int(result.qubits), physical_error_hex=float(result.error).hex())
            return result

    def bounded_pipeline(cls, *args, **kwargs):
        audit.charge("factory_pipeline_candidates")
        if audit.current_factory_query is None:
            raise ValueError("A native factory pipeline bypassed its recorded frontier query")
        audit.current_factory_query["pipeline_candidates"] += 1
        if audit.current_factory_query["pipeline_candidates"] > 34:
            raise SeventhPhysicalLimit("Native pipeline search exceeds the pinned 34-candidate domain", audit.snapshot("candidate_cap_exceeded"))
        return original_pipeline_bound(*args, **kwargs)

    qre.Trace, models.RoundBasedFactory = Trace, Factory
    round_based._Pipeline.try_create = classmethod(bounded_pipeline)
    try:
        yield audit
    finally:
        qre.Trace, models.RoundBasedFactory = original_trace, original_factory
        round_based._Pipeline.try_create = original_pipeline_descriptor


def _seventh_bind_native_identity(point, group, audit):
    """Bind the retained backend point to the native query that produced it."""
    factory = point["factory_supply"]
    if group["total_t_states"] == 0:
        return {"native_factory_isa_sha256": None, "native_t_instruction_sha256": None,
                "identity_scope": "no active magic-state supply for this class"}
    matches = []
    for record in audit.trace_queries:
        if (record["physical_class_id"] != group["physical_class_id"] or record["status"] != "native_result_returned"
                or record["accounting_trace_sha256"] != point["accounting_trace_sha256"]
                or record["runtime_ns"] != point["runtime_per_complete_attempt_ns"]
                or record["peak_physical_qubits"] != point["peak_physical_qubits"]
                or not math.isclose(float.fromhex(record["physical_error_hex"]) * group["shots"],
                                    point["full_physical_failure_model"], rel_tol=1e-10, abs_tol=1e-30)):
            continue
        for identity in audit.supply_by_t_signature.get(record["native_t_instruction_sha256"], []):
            native = identity["native_t_instruction"]
            if (identity["surface_code_query"]["distance"] == point["data_code_distance"]
                    and native["expected_time_ns"] == factory["pipeline_time_ns"]
                    and native["expected_space"] == point["factory_physical_qubits"]
                    and native["expected_output_error_hex"] == float(factory["output_error"]).hex()):
                matches.append(identity)
    unique = {_seventh_digest(value): value for value in matches}
    if len(unique) != 1:
        raise ValueError("Retained physical point lacks one unambiguous recorded native factory ISA")
    return next(iter(unique.values()))


def run_seventh_physical_workload(compiled_classes, backend=None, cache_dir=None, caps=None, *,
                                  expected_total_preparations=82_032, expected_total_t_states=None):
    """Estimate the full replay under one common distance/factory configuration.

    This performs native queries and must run only after the recipe/source
    freeze, in a dedicated process with an outer wall-time watchdog.
    """
    prepared = prepare_seventh_physical_workload(compiled_classes, backend=backend,
                expected_total_preparations=expected_total_preparations,
                expected_total_t_states=expected_total_t_states, caps=caps)
    caps = prepared["caps"]
    cache = Path(cache_dir).resolve() / ("qdk-1.34.0-" + SEVENTH_INSTRUMENT_SOURCE_SHA256
                                        + "-" + _seventh_digest(SEVENTH_NATIVE_SOURCE_SHA256))
    backend_path = Path(backend.__file__).resolve()
    source_root = (backend_path.parents[2] if backend_path.parent.name == "openquantum" and backend_path.parent.parent.name == "src"
                   else backend_path.parent)
    if cache.is_relative_to(source_root):
        raise ValueError("Private factory cache must remain outside the source checkout and public output")
    profile = backend.InstrumentProfile()
    audit = _SeventhNativeAudit(caps)
    results = []
    try:
        with _seventh_bounded_native_queries(audit):
            for group in prepared["groups"]:
                audit.current_group = group["physical_class_id"]
                audit.charge("top_level_calls")
                result = backend.estimate_instrument(group["synthesized"], group["shots"], profile=profile,
                    physical_failure_budget=group["physical_failure_budget"],
                    factory_availability_budget=group["factory_availability_budget"],
                    cache_dir=cache, include_all_feasible_points=True)
                audit.check()
                for point in result["all_feasible_points"]:
                    point["native_factory_identity"] = _seventh_bind_native_identity(point, group, audit)
                    if group["total_t_states"] and point["factory_supply"]["full_pipeline_retry_cap_per_state"] != prepared["global_factory_retry_cap"]:
                        raise ValueError("A physical class changed the common complete-factory retry configuration")
                results.append({"physical_class_id": group["physical_class_id"],
                                "compiled_class_ids": group["compiled_class_ids"], "shots": group["shots"],
                                "total_t_states": group["total_t_states"], "signature": group["signature"],
                                "physical_failure_budget": group["physical_failure_budget"],
                                "physical_failure_allocation_exact": group["physical_failure_allocation_exact"],
                                "factory_availability_budget": group["factory_availability_budget"],
                                "factory_availability_allocation_exact": group["factory_availability_allocation_exact"],
                                "zero_t_factory_budget_is_unused_positive_api_placeholder": group["total_t_states"] == 0,
                                "report": result})
            audit.check()
    except BaseException as error:
        if not isinstance(error, SeventhPhysicalLimit):
            try:
                error.seventh_physical_audit = audit.snapshot("incomplete_error")
            except (AttributeError, TypeError):
                pass
        raise
    by_group, active_factory_groups = [], []
    for result in results:
        options = {}
        for point in result["report"]["all_feasible_points"]:
            identity = point["native_factory_identity"]
            key = ((point["data_code_distance"], identity["native_factory_isa_sha256"],
                    identity["native_t_instruction_sha256"], prepared["global_factory_retry_cap"])
                   if result["total_t_states"] else point["data_code_distance"])
            if key in options and _seventh_canonical(options[key]) != _seventh_canonical(point):
                raise ValueError("One native common configuration has inconsistent duplicate physical points")
            options[key] = point
        by_group.append(options)
        if result["total_t_states"]:
            active_factory_groups.append(options)
    common_keys = set(active_factory_groups[0]).intersection(*(set(options) for options in active_factory_groups[1:]))
    for result, options in zip(results, by_group):
        if not result["total_t_states"]:
            common_keys = {key for key in common_keys if key[0] in options}
    if not common_keys:
        raise SeventhPhysicalLimit("No common distance/native-factory ISA covers every physical class within frozen caps",
                                   audit.snapshot("no_common_configuration"))
    common = []
    for key in sorted(common_keys):
        points = [options[key if result["total_t_states"] else key[0]] for result, options in zip(results, by_group)]
        total_runtime = sum(point["worst_case_runtime_all_attempts_ns"] for point in points)
        physical_failure = math.fsum(point["full_physical_failure_model"] for point in points)
        factory_failure = math.fsum(point["factory_supply"]["availability_failure_union_bound"] for point in points)
        physical_exact = sum((Fraction.from_float(point["full_physical_failure_model"]) for point in points), Fraction())
        factory_exact = (Fraction(prepared["total_t_states"]) * Fraction(str(profile.factory_pipeline_failure_bound))
                         ** prepared["global_factory_retry_cap"])
        if physical_exact > Fraction("0.0025") or factory_exact > Fraction("0.0025"):
            raise ValueError("Common configuration exceeds a whole-workflow failure allocation")
        factory_qubits = {point["factory_physical_qubits"] for result, point in zip(results, points) if result["total_t_states"]}
        if len(factory_qubits) != 1:
            raise ValueError("An identical native factory ISA changed its footprint")
        common_factory_qubits = next(iter(factory_qubits))
        class_points = []
        for result, point in zip(results, points):
            idle = result["total_t_states"] == 0
            class_points.append({"physical_class_id": result["physical_class_id"], "point": point,
                                 "common_factory_reserved_but_unused": idle,
                                 "reserved_common_factory_physical_qubits": common_factory_qubits,
                                 "common_peak_physical_qubits": (point["data_workspace_physical_qubits"] + common_factory_qubits
                                                                  if idle else point["peak_physical_qubits"]),
                                 "idle_factory_scope": ("physical area reserved; no encoded factory/output state is retained, no pipelines run; native complete-pipeline startup is charged on later active supply"
                                                         if idle else None)})
        common.append({"data_code_distance": key[0], "native_factory_isa_sha256": key[1],
                       "native_t_instruction_sha256": key[2], "global_factory_retry_cap": key[3],
                       "peak_physical_qubits": max(point["common_peak_physical_qubits"] for point in class_points),
                       "factory_physical_qubits": common_factory_qubits,
                       "worst_case_serial_runtime_ns": total_runtime,
                       "whole_physical_failure_model": physical_failure,
                       "whole_physical_failure_model_exact_binary_rational_sum": str(physical_exact),
                       "whole_factory_availability_failure_bound": float(factory_exact),
                       "whole_factory_availability_failure_bound_exact": str(factory_exact),
                       "factory_availability_report_float_sum": factory_failure,
                       "total_preparations": prepared["total_preparations"],
                       "total_t_states": prepared["total_t_states"],
                       "per_physical_class": class_points})
    minimum_qubits = min(common, key=lambda point: (point["peak_physical_qubits"], point["worst_case_serial_runtime_ns"]))
    minimum_runtime = min(common, key=lambda point: (point["worst_case_serial_runtime_ns"], point["peak_physical_qubits"]))
    common_pareto, shortest = [], None
    for point in sorted(common, key=lambda item: (item["peak_physical_qubits"], item["worst_case_serial_runtime_ns"])):
        if shortest is None or point["worst_case_serial_runtime_ns"] < shortest:
            common_pareto.append({name: point[name] for name in (
                "data_code_distance", "native_factory_isa_sha256", "native_t_instruction_sha256",
                "global_factory_retry_cap", "peak_physical_qubits", "factory_physical_qubits",
                "worst_case_serial_runtime_ns", "whole_physical_failure_model",
                "whole_factory_availability_failure_bound", "total_preparations", "total_t_states")})
            shortest = point["worst_case_serial_runtime_ns"]
    audit.check()
    return {"schema_version": 1, "status": "complete_conditional_recorded_workload_common_configuration",
            "profile": prepared["profile"], "caps": asdict(caps),
            "instrument_source_sha256": SEVENTH_INSTRUMENT_SOURCE_SHA256,
            "native_source_sha256": SEVENTH_NATIVE_SOURCE_SHA256,
            "complete_preparations": prepared["total_preparations"], "actual_magic_states": prepared["total_t_states"],
            "whole_physical_failure_budget": prepared["whole_physical_failure_budget"],
            "whole_factory_availability_budget": prepared["whole_factory_availability_budget"],
            "global_factory_retry_cap": prepared["global_factory_retry_cap"],
            "compiled_class_mapping": prepared["compiled_class_mapping"],
            "physical_classes": results, "common_configurations": common,
            "common_pareto_configurations": common_pareto,
            "selected_minimum_qubits": minimum_qubits, "selected_minimum_runtime": minimum_runtime,
            "query_audit": audit.snapshot("complete"),
            "grouping_scope": prepared["grouping_scope"],
            "conditional_scope": "fixed recorded classical decisions; no guarantee of the same adaptive optimizer path under changed synthesis/noise",
            "excluded": ["measured optimizer/classical-scoring/communication latency unless separately supplied by the caller",
                         "offline synthesis and export time from the conditional quantum runtime",
                         "certified physical routing/parity/factory-data layout and decoder throughput",
                         "explicit distillation-unit pipeline/layout identity beyond the recorded native ISA/T cost-model values"],
            "unsupported": ["arbitrary adaptive source control; frozen backend supports fixed schedules and reset/T/factory/trial macros",
                            "retaining encoded factory/output states during zero-T groups, measured factory startup/reconfiguration delays beyond native pipeline and reserved feedback macros"],
            "hardware_or_quantum_advantage_guarantee": False}
