#!/usr/bin/env python3
"""Freeze and validate one saved-word candidate; physical execution is absent.

The seventh completion CLI re-exports derived records even with --verify-only.
This driver reads its immutable files directly and never invokes that CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
BASE_COMMIT = "4fb73f6968c3d17c0e505f043b3e3ad400025393"
RUN_ID = "qaoa-cycle-4-p2-alternate-seed-7"
BUNDLE = ROOT / "assets/generated/optimization/compiled-replay"
DEFAULT_OUTPUT = ROOT / "docs/coordination/optimization-eighth-candidate"
BYTE_CAP = 16 * 1024 * 1024
CLASS_DECODED_CAP = 4 * 1024 * 1024
SOURCE_CAPS = dict(classes=54, measured_groups=110, preparations=82032,
                   data_qubits=4, operations_per_class=10000,
                   candidate_operations=540000, validation_primitive_visits=1080000,
                   density_qubits=6, density_cases=32, density_micro_operations=64,
                   density_complex_entries=4096, spectral_tolerance=2e-11,
                   source_validation_wall_seconds=180, output_bytes=BYTE_CAP)
OWNED_PATHS = (
    "scripts/prepare_optimization_word_transport.py",
    "src/openquantum/optimization_word_transport.py",
    "src/openquantum/optimization_word_accounting.py",
    "src/openquantum/optimization_word_validation.py",
    "tests/test_optimization_word_transport.py",
    "tests/test_optimization_word_accounting.py",
    "tests/test_optimization_word_validation.py",
    "tests/test_optimization_word_driver.py",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Input must be a regular saved file")
    return path.read_bytes()


def saved(relative):
    path = BUNDLE / relative
    if not path.resolve().is_relative_to(BUNDLE.resolve()):
        raise ValueError("Saved record path escaped its bundle")
    return path


def module(name, expected=None):
    path = ROOT / "src/openquantum" / (name + ".py")
    raw = regular(path)
    if expected is not None and digest(raw) != expected:
        raise ValueError("Frozen candidate source changed")
    spec = importlib.util.spec_from_file_location("word_candidate_" + name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    exec(compile(raw, str(path), "exec"), result.__dict__)
    return result


def inventory(directory):
    rows = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rows[path.relative_to(ROOT).as_posix()] = digest(regular(path))
    return rows


def verify_inventory(rows):
    for relative, expected in rows.items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or digest(regular(path)) != expected:
            raise ValueError("An immutable input changed")
    if inventory(ROOT / "assets/generated") != rows:
        raise ValueError("The scientific archive's file set changed")


def checked_output(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT / "docs/coordination") or path.is_relative_to(BUNDLE):
        raise ValueError("Source-only evidence belongs in the private coordination tree")
    return path


def load_saved_classes():
    manifest = json.loads(regular(saved("manifest.json")))
    if (manifest["status"] != "complete" or manifest["run_id"] != RUN_ID
            or manifest["preparations"] != 82032 or manifest["sampled_groups"] != 110
            or len(manifest["classes"]) != 54):
        raise ValueError("The saved source workload is not the frozen seventh selection")
    classes = []
    for entry in manifest["classes"]:
        path = saved(entry["checkpoint"])
        with gzip.open(path, "rb") as stream:
            raw = stream.read(CLASS_DECODED_CAP + 1)
        if len(raw) > CLASS_DECODED_CAP:
            raise ValueError("Saved class exceeded its decoded-record cap")
        checkpoint = json.loads(raw)
        record = checkpoint["record"]
        if (checkpoint["identity"] != manifest["original_identity"]
                or checkpoint["key"] != entry["class_id"] or record["class_id"] != entry["class_id"]
                or record["shots"] != entry["shots"] or record["batches"] != entry["batches"]):
            raise ValueError("Checkpoint binding or complete repetition count changed")
        desc = entry["circuits"]["clifford_t"]
        if (digest(regular(saved(desc["file"]))) != desc["sha256"]
                or record["clifford_t_qasm"].encode() != regular(saved(desc["file"]))
                or record["clifford_t_global_phase"] != desc["global_phase"]):
            raise ValueError("Saved sequence, QASM or exact phase identity changed")
        classes.append(dict(class_id=entry["class_id"], shots=entry["shots"],
            batches=entry["batches"], parameters=entry["parameters"],
            operations=record["clifford_t_operations"], global_phase=record["clifford_t_global_phase"],
            source_qasm_sha256=desc["sha256"], checkpoint_sha256=digest(regular(path))))
    if sum(c["shots"] for c in classes) != 82032 or sum(len(c["batches"]) for c in classes) != 110:
        raise ValueError("A training or holdout group was omitted")
    return manifest, classes


def qasm(operations):
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', 'bit[4] c;', 'qubit[4] q;']
    for name, wires, clbits, params in operations:
        if params:
            raise ValueError("Only materialized parameter-free primitives may be exported")
        operands = ", ".join(f"q[{wire}]" for wire in wires)
        if name == "measure":
            lines.append(f"c[{clbits[0]}] = measure {operands};")
        else:
            lines.append(f"{name} {operands};")
    return ("\n".join(lines) + "\n").encode()


def candidate_rows(classes, transform):
    rows = []
    for source in classes:
        candidate = transform.build_candidate(source["operations"])
        audit_segments(source["operations"], candidate)
        rows.append(dict(source=source, candidate=candidate))
    if (len(rows) != 54 or sum(r["candidate"]["candidate_operations"] for r in rows) > 540000
            or sum(r["candidate"]["source_operations"] + r["candidate"]["candidate_operations"] for r in rows) > 1080000):
        raise ValueError("Source materialization or full-pair validation exceeds its frozen cap")
    return rows


def audit_segments(source, candidate):
    """Check the theorem's word boundaries independently of the builder."""
    local = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
    source_cursor, candidate_cursor = 0, 0
    actual = candidate["operations"]
    selected = 0
    for segment in candidate["segments"]:
        start, stop = segment["source_start"], segment["source_stop"]
        out_start, out_stop = segment["candidate_start"], segment["candidate_stop"]
        word = source[start:stop]
        wire = segment["source_wire"]
        if (not source_cursor <= start < stop <= len(source)
                or not candidate_cursor <= out_start < out_stop <= len(actual)
                or source[source_cursor:start] != actual[candidate_cursor:out_start]
                or any(op[0] in local for op in source[source_cursor:start])
                or any(op[0] not in local or op[1] != [wire] for op in word)):
            raise ValueError("A word certificate crossed or omitted a source event")
        if ((start and source[start - 1][0] in local and source[start - 1][1] == [wire])
                or (stop < len(source) and source[stop][0] in local and source[stop][1] == [wire])):
            raise ValueError("A candidate word was not maximal in the saved stream")
        magic = sum(op[0] in {"t", "tdg"} for op in word)
        eligible = wire > 0 and magic >= 3
        if segment["shuttled"] != eligible:
            raise ValueError("Word selection differs from the predeclared threshold")
        edges = [(i, i - 1) for i in range(wire, 0, -1)] if eligible else []
        def swaps(path):
            return [["cx", pair, [], []] for a, b in path for pair in ([a, b], [b, a], [a, b])]
        rewritten = [[op[0], [0], op[2], op[3]] for op in word] if eligible else word
        expected = swaps(edges) + rewritten + swaps(list(reversed(edges)))
        if (actual[out_start:out_stop] != expected
                or segment["forward_swap_edges"] != [list(p) for p in edges]
                or segment["reverse_swap_edges"] != [list(p) for p in reversed(edges)]
                or segment["spectator_zero_assumption"] is not False
                or segment["boundary_data_permutation"] != [0, 1, 2, 3]):
            raise ValueError("Explicit adjacent SWAP/reversal certificate is incorrect")
        selected += eligible
        source_cursor, candidate_cursor = stop, out_stop
    if (source[source_cursor:] != actual[candidate_cursor:]
            or any(op[0] in local for op in source[source_cursor:])):
        raise ValueError("A source suffix or local word was omitted")
    return dict(structural_word_boundaries_verified=True, selected_words=selected,
                original_barriers_and_source_events_preserved=True)


def accounting_input(rows):
    return [dict(class_id=r["source"]["class_id"], shots=r["source"]["shots"],
                 baseline_event_counts=r["candidate"]["baseline_event_counts"],
                 candidate_event_counts=r["candidate"]["candidate_event_counts"],
                 segments=[dict(wire=s["source_wire"], t_count=s["t_count"], tdg_count=s["tdg_count"])
                           for s in r["candidate"]["segments"] if s["shuttled"]]) for r in rows]


def class_plan(row):
    source, candidate = row["source"], row["candidate"]
    return dict(class_id=source["class_id"], shots=source["shots"], batches=source["batches"],
        parameters=source["parameters"], exact_global_phase=source["global_phase"],
        global_phase_hex=float(source["global_phase"]).hex(), source_qasm_sha256=source["source_qasm_sha256"],
        checkpoint_sha256=source["checkpoint_sha256"],
        source_operations_sha256=candidate["source_operations_sha256"],
        candidate_operations_sha256=candidate["candidate_operations_sha256"],
        candidate_qasm_sha256=digest(qasm(candidate["operations"])),
        segments_sha256=digest(encoded(candidate["segments"])),
        source_operations=candidate["source_operations"], candidate_operations=candidate["candidate_operations"],
        selected_words=sum(s["shuttled"] for s in candidate["segments"]))


def write_files(output, files):
    paths = [p for p in output.rglob("*") if p.is_file()] if output.exists() else []
    if sum(p.stat().st_size for p in paths) > BYTE_CAP:
        raise ValueError("Existing private candidate output exceeded its frozen byte cap")
    present = {p.relative_to(output).as_posix(): regular(p) for p in paths}
    combined = dict(present, **files)
    if sum(map(len, combined.values())) > BYTE_CAP:
        raise ValueError("The complete private candidate bundle exceeded its frozen byte cap")
    destinations = []
    for relative, raw in files.items():
        path = output / relative
        if not path.resolve().is_relative_to(output):
            raise ValueError("Candidate evidence path escaped its output")
        if path.exists() and regular(path) != raw:
            raise ValueError("Frozen candidate evidence cannot be overwritten")
        destinations.append((path, raw))
    for path, raw in destinations:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)


def freeze(output):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != BASE_COMMIT:
        raise ValueError("The first candidate freeze must descend directly from the reviewed seventh handoff")
    if (subprocess.run(["git", "diff", "--quiet", BASE_COMMIT, "--", "assets/generated"], cwd=ROOT).returncode
            or subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "assets/generated"], cwd=ROOT)):
        raise ValueError("The reviewed scientific archive changed before candidate freeze")
    before = inventory(ROOT / "assets/generated")
    if len(before) != 8642 or sum((ROOT / name).stat().st_size for name in before) != 441235324:
        raise ValueError("The reviewed complete scientific archive's count or bytes changed")
    owned_pins = {name: digest(regular(ROOT / name)) for name in OWNED_PATHS}
    manifest, classes = load_saved_classes()
    transform = module("optimization_word_transport", owned_pins["src/openquantum/optimization_word_transport.py"])
    accounting = module("optimization_word_accounting", owned_pins["src/openquantum/optimization_word_accounting.py"])
    rows = candidate_rows(classes, transform)
    ledger = accounting.build_accounting(accounting_input(rows))
    profile_record = json.loads(regular(saved("physical-summary.json")))
    chosen = next(r for r in profile_record["selections"] if r["selection"] == "minimum_runtime")
    machine = accounting.frozen_machine()
    if (profile_record["profile"] != accounting.frozen_profile()
            or chosen["native_factory_isa_sha256"] != machine["native_factory_isa_sha256"]
            or chosen["native_t_instruction_sha256"] != machine["native_t_instruction_sha256"]):
        raise ValueError("Prospective machine is not the saved common baseline configuration")
    recipe = dict(schema_version=1, candidate_id=transform.CANDIDATE_ID,
        status="frozen_source_candidate_physical_execution_unapproved", base_commit=BASE_COMMIT,
        run_id=RUN_ID, rule="maximal contiguous single-wire local Clifford+T words, wire>0 and T/Tdg>=3; adjacent DATA SWAP path to D0 and exact reversal",
        eligible_paths="all four data wires explicitly initialized; arbitrary entangled spectator data allowed; no clean spectator assumption",
        source_caps=SOURCE_CAPS, prospective_native_caps=ledger["prospective_native_caps"],
        physical_execution_enabled=False, candidate_approval_received=False,
        seventh_clearance_received=True, common_machine=machine, profile=accounting.frozen_profile(),
        unchanged_budget=json.loads(regular(saved("recipe.json")))["budget"],
        baseline_manifest_sha256=digest(regular(saved("manifest.json"))),
        baseline_physical_dag_sha256=manifest["physical_storage"]["sha256"],
        input_inventory_sha256=digest(encoded(before)),
        owned_source_sha256=owned_pins,
        classes=[class_plan(r) for r in rows],
        accounting_sha256=digest(encoded(ledger)),
        validation_design="independent phase-restored full16x16 operators/all16 initialized joint outputs for54classes; corrected injection Kraus/Choi and bounded6qubit ideal density mechanisms; exact old49allocation parents",
        limitations=["no candidate native Trace or factory calibration", "no hardware or physical-code geometry certificate", "encoder/factory internals and decoder throughput remain unsupported", "archived holdouts and prospective fixed execution remain separate experiments"],
        seventh_cli_receipt_boundary="original --verify-only re-exports derived artifacts; never invoked here; seventh canonical bundle is read-only")
    verify_inventory(before)
    if {name: digest(regular(ROOT / name)) for name in OWNED_PATHS} != owned_pins:
        raise ValueError("Candidate source changed while constructing its freeze")
    write_files(output, {"recipe.json": encoded(recipe), "input-inventory.json": encoded(before),
                         "accounting-plan.json": encoded(ledger)})
    return dict(status=recipe["status"], recipe_sha256=digest(encoded(recipe)), classes=len(rows),
                candidate_operations=sum(r["candidate"]["candidate_operations"] for r in rows),
                selected_words=sum(c["selected_words"] for c in recipe["classes"]), physical_execution_performed=False)


def bound_recipe(output):
    recipe = json.loads(regular(output / "recipe.json"))
    if (set(recipe["owned_source_sha256"]) != set(OWNED_PATHS)
            or recipe["source_caps"] != SOURCE_CAPS or recipe["base_commit"] != BASE_COMMIT
            or recipe["run_id"] != RUN_ID or len(recipe["classes"]) != 54
            or len({c["class_id"] for c in recipe["classes"]}) != 54):
        raise ValueError("Frozen candidate pin set or execution caps changed")
    frozen_names = list(recipe["owned_source_sha256"]) + [
        (output / name).relative_to(ROOT).as_posix()
        for name in ("recipe.json", "input-inventory.json", "accounting-plan.json")]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *frozen_names],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *frozen_names], cwd=ROOT).returncode:
        raise ValueError("Candidate producer and recipe must be committed before source validation")
    for name, expected in recipe["owned_source_sha256"].items():
        if digest(regular(ROOT / name)) != expected:
            raise ValueError("Frozen candidate source or verification specification changed")
    before = json.loads(regular(output / "input-inventory.json"))
    if digest(encoded(before)) != recipe["input_inventory_sha256"]:
        raise ValueError("Frozen scientific input inventory changed")
    if digest(regular(output / "accounting-plan.json")) != recipe["accounting_sha256"]:
        raise ValueError("The frozen source accounting plan changed")
    verify_inventory(before)
    accounting = module("optimization_word_accounting", recipe["owned_source_sha256"]["src/openquantum/optimization_word_accounting.py"])
    if (recipe["profile"] != accounting.frozen_profile()
            or recipe["common_machine"] != accounting.frozen_machine()
            or recipe["unchanged_budget"] != json.loads(regular(saved("recipe.json")))["budget"]):
        raise ValueError("Frozen machine, profile or statistical/failure allocations changed")
    if recipe["physical_execution_enabled"] or recipe["candidate_approval_received"]:
        raise ValueError("This source-only driver cannot execute or authorize physical work")
    return recipe, before


def validate(output):
    if (output / "validation.json").exists():
        return verify(output)
    started = perf_counter()
    recipe, before = bound_recipe(output)
    _, classes = load_saved_classes()
    transform = module("optimization_word_transport", recipe["owned_source_sha256"]["src/openquantum/optimization_word_transport.py"])
    validator = module("optimization_word_validation", recipe["owned_source_sha256"]["src/openquantum/optimization_word_validation.py"])
    accounting = module("optimization_word_accounting", recipe["owned_source_sha256"]["src/openquantum/optimization_word_accounting.py"])
    rows = candidate_rows(classes, transform)
    if [class_plan(r) for r in rows] != recipe["classes"]:
        raise ValueError("Concrete candidate differs from the pre-validation freeze")
    ledger = accounting.build_accounting(accounting_input(rows))
    if digest(encoded(ledger)) != recipe["accounting_sha256"]:
        raise ValueError("Source accounting differs from its frozen candidate")
    checks, files = [], {}
    density = validator.validate_density_mechanisms()
    for row in rows:
        source, candidate = row["source"], row["candidate"]
        if perf_counter() - started > 180:
            raise ValueError("Bounded source validation exceeded its wall allowance")
        checked = validator.validate_candidate_pair(source["operations"], candidate["operations"],
            source_global_phase=source["global_phase"], candidate_global_phase=source["global_phase"])
        checks.append(dict(class_id=source["class_id"], shots=source["shots"],
                           **checked, **audit_segments(source["operations"], candidate)))
        key = source["class_id"]
        files[f"circuits/{key}.qasm"] = qasm(candidate["operations"])
        files[f"circuits/{key}.json.gz"] = gzip.compress(encoded(dict(class_id=key,
            exact_global_phase=source["global_phase"], source_qasm_sha256=source["source_qasm_sha256"],
            operations=candidate["operations"], segments=candidate["segments"])), mtime=0)
    negatives = list(csv.DictReader(regular(saved("negative-controls.csv")).decode().splitlines()))
    supports = Counter(r["algorithm"] for r in negatives if r["kind"] == "sampled" and r["heldout_target_status"] == "supported")
    if supports != {"qaoa": 3}:
        raise ValueError("Archived QAOA/VQE holdout support changed")
    elapsed = perf_counter() - started
    if elapsed > 180 or density["case_count"] > 32 or density["maximum_qubits"] > 6 or density["maximum_micro_operations"] > 64:
        raise ValueError("Independent source/density validation exceeded its frozen scope")
    verify_inventory(before)
    report = dict(schema_version=1, status="validated_source_candidate_physical_unexecuted",
        recipe_sha256=digest(regular(output / "recipe.json")), class_checks=checks, density_checks=density,
        classes=54, groups=110, preparations=82032, actual_magic_states=ledger["actual_magic_states"],
        maximum_operator_spectral_error=max(r["operator_spectral_error"] for r in checks),
        maximum_joint_probability_error=max(r["joint_z_maximum_probability_error"] for r in checks),
        archived_supports=dict(qaoa=3, vqe=0), source_validation_wall_seconds=elapsed,
        source_validation_wall_scope="input binding, ideal density mechanisms and full-class operators; final input recheck/export are additionally covered by the 180s process watchdog",
        physical_execution_performed=False, native_queries=0, synthesis_calls=0,
        optimizer_calls=0, new_holdout_samples=0, immutable_inputs_preserved=True,
        physical_approval_requirement="specific coordinator candidate approval; seventh review clearance alone is insufficient")
    files["validation.json"] = encoded(report)
    files["accounting.json"] = encoded(ledger)
    artifact_inventory = {name: dict(bytes=len(raw), sha256=digest(raw)) for name, raw in files.items()}
    files["artifact-manifest.json"] = encoded(dict(status=report["status"], files=artifact_inventory))
    write_files(output, files)
    return dict(status=report["status"], classes=54, preparations=82032,
                maximum_operator_spectral_error=report["maximum_operator_spectral_error"],
                density_cases=density["case_count"], physical_execution_performed=False)


def verify(output):
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > BYTE_CAP:
        raise ValueError("Saved candidate output exceeded its cap before reading")
    recipe, before = bound_recipe(output)
    manifest = json.loads(regular(output / "artifact-manifest.json"))
    report = json.loads(regular(output / "validation.json"))
    required = {f"circuits/{c['class_id']}.{suffix}" for c in recipe["classes"] for suffix in ("qasm", "json.gz")} | {"validation.json", "accounting.json"}
    if set(manifest["files"]) != required:
        raise ValueError("Candidate manifest omitted or added a frozen class artifact")
    expected_set = required | {"recipe.json", "input-inventory.json", "accounting-plan.json", "artifact-manifest.json"}
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != expected_set:
        raise ValueError("Candidate output file set differs from the frozen complete source bundle")
    for relative, expected in manifest["files"].items():
        path = output / relative
        if not path.resolve().is_relative_to(output) or digest(regular(path)) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Saved candidate evidence changed")
    if ([r["class_id"] for r in report["class_checks"]] != [c["class_id"] for c in recipe["classes"]]
            or [r["shots"] for r in report["class_checks"]] != [c["shots"] for c in recipe["classes"]]
            or report["groups"] != 110 or report["archived_supports"] != {"qaoa": 3, "vqe": 0}
            or report["native_queries"] or report["synthesis_calls"] or report["optimizer_calls"] or report["new_holdout_samples"]):
        raise ValueError("Candidate evidence omitted frozen class/repetition coverage")
    transform = module("optimization_word_transport", recipe["owned_source_sha256"]["src/openquantum/optimization_word_transport.py"])
    for plan in recipe["classes"]:
        raw_qasm = regular(output / f"circuits/{plan['class_id']}.qasm")
        with gzip.open(output / f"circuits/{plan['class_id']}.json.gz", "rb") as stream:
            raw = stream.read(CLASS_DECODED_CAP + 1)
        if len(raw) > CLASS_DECODED_CAP:
            raise ValueError("Candidate class exceeded its decoded-record cap")
        circuit = json.loads(raw)
        if (digest(raw_qasm) != plan["candidate_qasm_sha256"]
                or circuit["class_id"] != plan["class_id"]
                or float(circuit["exact_global_phase"]).hex() != plan["global_phase_hex"]
                or circuit["source_qasm_sha256"] != plan["source_qasm_sha256"]
                or transform.operation_digest(circuit["operations"]) != plan["candidate_operations_sha256"]
                or digest(encoded(circuit["segments"])) != plan["segments_sha256"]
                or qasm(circuit["operations"]) != raw_qasm):
            raise ValueError("Candidate sequence, phase, segment or QASM binding changed")
    if (report["recipe_sha256"] != digest(regular(output / "recipe.json"))
            or report["physical_execution_performed"] or report["classes"] != 54
            or report["preparations"] != 82032 or sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > BYTE_CAP):
        raise ValueError("Saved candidate evidence is outside the frozen source-only scope")
    verify_inventory(before)
    return dict(status="verified_read_only_source_candidate", files=len(manifest["files"]),
                physical_execution_performed=False, writes_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--freeze-only", action="store_true")
    choices.add_argument("--validate", action="store_true")
    choices.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.validate:
        def stop_source_phase(signum, frame):
            raise TimeoutError("The source-only validation process exceeded 180 seconds")
        signal.signal(signal.SIGALRM, stop_source_phase)
        signal.setitimer(signal.ITIMER_REAL, 180)
    output = checked_output(args.output_dir)
    result = freeze(output) if args.freeze_only else validate(output) if args.validate else verify(output)
    if args.validate:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
