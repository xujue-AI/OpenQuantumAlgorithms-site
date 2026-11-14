#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inspect the bounded degree-two DQI bundle using saved files only.

This reader authenticates a closed file graph, component CLI observations,
source bindings and recorded scalar/count contracts. It imports no producer or
quantum SDK, constructs no circuit, and performs no state, matrix, angle or
probability reconstruction. Computational acceptance is separate from notebook
and publication acceptance. Its source alone is not an executed result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat


SCHEMA = "openquantum.dqi.degree_two_saved_bundle.v1"
ACCEPTANCE_SCHEMA = "openquantum.dqi.degree_two_component_invocations.v1"
ACCEPTANCE_STATUS = "passed_component_invocations_pending_notebook"
MAX_MANIFEST_BYTES = 128 << 10
MAX_MEMBER_BYTES = 2 << 20
MAX_NOTEBOOK_BYTES = 512 << 10
MAX_BUNDLE_BYTES = 8 << 20
ATOL = 2e-10
NOTEBOOK = "optimization_dqi_degree_two.ipynb"
SOURCE_FILES = frozenset({
    "optimization_dqi_degree_two.py", "optimization_dqi_degree_two_verify.py",
    "generate_optimization_dqi_degree_two.py", "validate_optimization_dqi_degree_two.py",
    "verify_dqi_degree_two_saved.py",
})
ROOT_FILES = SOURCE_FILES | {
    NOTEBOOK, "requirements-lock.txt", "LICENSE.txt", "input-c5.json",
    "input-figure-eight.json", "invocation-acceptance.json", "bundle.json",
}
GEN_FILES = frozenset({"operations.ndjson", "circuit.qasm", "ledger.json", "manifest.json"})
SUBDIRECTORIES = {
    "c5": GEN_FILES, "figure-eight": GEN_FILES,
    "validation": frozenset({"validation.json", "manifest.json"}),
}
MEMBERS = (ROOT_FILES - {"bundle.json", NOTEBOOK}) | {
    directory + "/" + filename
    for directory, names in SUBDIRECTORIES.items() for filename in names
}
ALL_FILES = MEMBERS | {"bundle.json", NOTEBOOK}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
ANGLE_RE = re.compile(r"-?0x[0-9a-f]+(?:\.[0-9a-f]+)?p[+-][0-9]+\Z", re.ASCII)
PRIVATE_MARKERS = (
    b"/" + b"Users/", b"/private/" + b"tmp/", b"/" + b"tmp/",
    b"docs/" + b"coordination/", b"ssh" + b"://", b"PAGES_" + b"DEPLOY_KEY",
    b"BEGIN " + b"OPENSSH PRIVATE KEY", b"BEGIN " + b"RSA PRIVATE KEY",
    b"gh" + b"p_", b"github_" + b"pat_",
)
PRIVATE_KEYS = frozenset({
    "pid", "ppid", "nonce", "session_id", "control_fd", "cwd", "argv",
    "input_root", "output_dir", "clock_origin", "deadline", "deadline_monotonic",
    "started_at_monotonic", "claim", "raw_stdout", "raw_stderr",
})
COUNTERS = {
    "complete_statevector_evolutions": 1,
    "boolean_decoder_mappings": 512,
    "small_ccx_reference_cases": 1,
}
FALSE_FLAGS = (
    "sampled_outcomes", "auxiliary_postselection_performed", "phase_normalization_performed",
    "native_synthesis_executed", "physical_resources_computed", "qpu_execution",
    "formal_interval_certificate", "old_degree_one_evidence_reexecuted",
    "optimization_advantage_claim", "publication_complete",
)
VALIDATION_FALSE_FLAGS = (
    "global_phase_fitting_performed", "formal_theorem_reproduced",
    "figure_eight_evolution_performed", "figure_eight_state_evolved",
    "accepted_degree_one_producer_rerun", "accepted_degree_one_operator_rerun",
)
C5_COUNTS = {
    "h": 222, "cx": 686, "t": 436, "tdg": 327, "x": 80, "ry": 17,
    "z": 5, "reset": 12, "measure": 4, "barrier": 7,
}
FIGURE_COUNTS = {
    "h": 1636, "cx": 4954, "t": 3256, "tdg": 2442, "x": 646, "ry": 37,
    "z": 10, "reset": 24, "measure": 8, "barrier": 7,
}
SIGNED_NUMERATORS = (0, -1, -1, -1, -1, 6, -1, -1, -1, 6, 6, 6, -1, 6, -1, -1)
C5_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
FIGURE_EDGES = C5_EDGES + ((0, 5), (5, 6), (6, 7), (7, 8), (8, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _int(value, expected, label):
    _require(type(value) is int and value == expected, label + " changed")


def _finite(value, label):
    _require(type(value) in (int, float) and math.isfinite(value), label + " must be finite")
    return value


def _error(value, label):
    _require(0 <= _finite(value, label) <= ATOL, label + " exceeds the recorded tolerance")


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _json(raw, label):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(label + " is not bounded ordinary JSON") from error


def _privacy(value):
    if isinstance(value, dict):
        _require(not (set(value) & PRIVATE_KEYS), "Operational fields must stay outside the public bundle")
        for child in value.values():
            _privacy(child)
    elif isinstance(value, list):
        for child in value:
            _privacy(child)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        _require(not any(marker in raw for marker in PRIVATE_MARKERS), "Private coordinates in saved metadata")


def _directory(directory):
    path = Path(directory)
    _require(not path.is_symlink() and path.is_dir(), "A complete ordinary bundle directory is required")
    return path.resolve()


def _read(directory, name, cap=MAX_MEMBER_BYTES):
    _require(name in ALL_FILES, "Unlisted saved member")
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and len(relative.parts) <= 2
             and all(part not in {".", ".."} for part in relative.parts), "Invalid saved member path")
    path = directory.joinpath(*relative.parts)
    for parent in path.parents:
        if parent == directory:
            break
        _require(not parent.is_symlink() and parent.is_dir(), "Saved subdirectory redirects or is missing")
    _require(not path.is_symlink(), "Saved members must not be symlinks")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= cap, "Saved member exceeds its ordinary-file cap")
        chunks, size = [], 0
        while True:
            raw = os.read(descriptor, min(65536, cap + 1 - size))
            if not raw:
                break
            chunks.append(raw)
            size += len(raw)
            _require(size <= cap, "Saved member grew beyond its cap")
        after = os.fstat(descriptor)
        _require(before.st_size == after.st_size == size
                 and before.st_mtime_ns == after.st_mtime_ns, "Saved member changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _entry_names(directory, expected):
    names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            _require(entry.name in expected and entry.name not in names and len(names) < len(expected),
                     "The closed saved inventory has an unexpected entry")
            names.add(entry.name)
    return names


def _inventory(directory):
    _require(_entry_names(directory, ROOT_FILES | set(SUBDIRECTORIES)) == ROOT_FILES | set(SUBDIRECTORIES),
             "The closed saved bundle has missing or extra root entries")
    total = 0
    for directory_name, names in SUBDIRECTORIES.items():
        child = directory / directory_name
        _require(not child.is_symlink() and child.is_dir(), "Saved subdirectory is not ordinary")
        _require(_entry_names(child, names) == names, "Saved subdirectory has missing or extra members")
    for name in ALL_FILES:
        path = directory / name
        _require(not path.is_symlink() and path.is_file(), "Saved inventory includes a nonordinary member")
        size = path.stat().st_size
        _require(size <= (MAX_NOTEBOOK_BYTES if name == NOTEBOOK else MAX_MEMBER_BYTES), "Saved member is too large")
        total += size
    _require(total <= MAX_BUNDLE_BYTES and len(ALL_FILES) == 22, "Whole saved bundle exceeds its cap")
    return total


def notebook_source_sha256(raw):
    """Hash cell types and sources; ordinary outputs/counters may change."""
    notebook = _json(raw, "Notebook")
    _require(type(notebook) is dict and notebook.get("nbformat") == 4,
             "Expected a version-four notebook")
    cells = notebook.get("cells")
    _require(type(cells) is list and 4 <= len(cells) <= 16, "Notebook cell graph exceeds its cap")
    projection = []
    code_cells = 0
    for cell in cells:
        _require(type(cell) is dict and cell.get("cell_type") in {"code", "markdown"}, "Unlisted notebook cell type")
        source = cell.get("source")
        if type(source) is list and all(type(part) is str for part in source):
            source = "".join(source)
        _require(type(source) is str and len(source.encode("utf-8")) <= 64 << 10, "Notebook source is invalid")
        projection.append({"cell_type": cell["cell_type"], "source": source})
        code_cells += cell["cell_type"] == "code"
    _require(code_cells == 4, "This companion requires exactly four code cells")
    return _sha(json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _pin(raw, binding, label, *, kind=False):
    expected = {"bytes": len(raw), "sha256": _sha(raw)}
    if kind:
        expected["kind"] = "file"
    _require(binding == expected, label + " bytes differ from their saved binding")


def _source_counts(raw, qubits, expected):
    lines = raw.splitlines()
    _require(len(lines) == sum(expected.values()), "Saved operation record count changed")
    counts = Counter()
    records = []
    for line in lines:
        record = _json(line, "Primitive record")
        _require(type(record) is dict and set(record) <= {"gate", "qubits", "classical", "parameters_hex"},
                 "Saved primitive record has unknown fields")
        gate, wires = record.get("gate"), record.get("qubits")
        _require(gate in expected and type(wires) is list and len(set(wires)) == len(wires)
                 and all(type(wire) is int and 0 <= wire < qubits for wire in wires), "Invalid saved primitive")
        _require(len(wires) == (2 if gate == "cx" else qubits if gate == "barrier" else 1), "Primitive arity changed")
        if gate == "measure":
            _require(type(record.get("classical")) is list and len(record["classical"]) == 1
                     and type(record["classical"][0]) is int
                     and 0 <= record["classical"][0] < expected["measure"], "Saved readout target changed")
        else:
            _require("classical" not in record, "Only readout names a classical target")
        if gate == "ry":
            angles = record.get("parameters_hex")
            _require(type(angles) is list and len(angles) == 1 and type(angles[0]) is str
                     and ANGLE_RE.fullmatch(angles[0]) is not None, "Saved RY must retain a binary-angle string")
        else:
            _require("parameters_hex" not in record, "Only RY retains an angle")
        counts[gate] += 1
        records.append(record)
    _require(dict(counts) == expected, "Observed saved primitive counts changed")
    return records


def _component_sources(value, buffers, *, script, module):
    _require(type(value) is dict and set(value) == {"script", "module"}, "Component source roles changed")
    for role, relative in (("script", script), ("module", module)):
        raw = buffers[PurePosixPath(relative).name]
        _require(value[role] == {"path": relative, "bytes": len(raw), "sha256": _sha(raw)},
                 "Component source differs from its saved root source: " + role)


def _generation(buffers, prefix, *, vertices, edges, n, m, wires, unitary, operations, normalization, counts):
    manifest = _json(buffers[prefix + "/manifest.json"], "Generation manifest")
    ledger = _json(buffers[prefix + "/ledger.json"], "Generation ledger")
    _require(manifest.get("schema") == "openquantum.dqi.degree_two_source_manifest.v1"
             and manifest.get("status") == "complete_generation_component_pending_invocation_acceptance"
             and manifest.get("scope") == "generation_only"
             and manifest.get("requires_observed_cli_exit_zero") is True, "Generation component is incomplete")
    _require(set(manifest.get("files", {})) == GEN_FILES - {"manifest.json"}, "Generation file graph changed")
    for name in GEN_FILES - {"manifest.json"}:
        _pin(buffers[prefix + "/" + name], manifest["files"][name], "Generation member")
    _require(ledger.get("schema") == "openquantum.dqi.degree_two_generation_ledger.v1"
             and ledger.get("status") == "generation_only_pending_invocation_acceptance"
             and ledger.get("requires_observed_cli_exit_zero") is True,
             "Expected saved generation-only ledger")
    _component_sources(manifest.get("sources"), buffers,
                       script="scripts/generate_optimization_dqi_degree_two.py",
                       module="src/openquantum/optimization_dqi_degree_two.py")
    _require(ledger.get("sources") == manifest["sources"], "Generation ledger source bindings differ")
    saved_input = _json(buffers["input-" + prefix + ".json"], "Saved graph input")
    _require(type(saved_input) is dict and set(saved_input) == {"schema", "vertices", "edges", "signs"}
             and saved_input.get("schema") == "openquantum.dqi.degree_two_input.v1",
             "Saved graph input schema changed")
    expected_input = {"vertices": vertices, "edges": [list(edge) for edge in edges], "signs": [1] * m}
    _require({key: saved_input[key] for key in expected_input} == expected_input
             and manifest.get("input") == expected_input and ledger.get("input") == expected_input,
             "Saved graph input differs from its generation metadata")
    _require(type(saved_input["vertices"]) is int
             and all(type(v) is int for edge in saved_input["edges"] for v in edge)
             and all(type(v) is int for v in saved_input["signs"]), "Graph fields require integer scalars")
    input_pin = manifest.get("input_file_pin")
    if input_pin is not None:
        _pin(buffers["input-" + prefix + ".json"], input_pin, "Generation graph input")
    for key in ("simulation_performed", "sampled_outcomes", "physical_resources_computed",
                "formal_interval_certificate", "optimization_advantage_claim"):
        _require(ledger.get(key) is False and manifest.get(key) is False,
                 "Generation-only scope changed: " + key)
    _require(manifest.get("publication_complete") is False
             and ledger.get("semicircle_theorem_applied") is False
             and ledger.get("accepted_degree_one_producer_rerun") is False,
             "Generation ledger claims work outside this component")
    for key, expected in (("n", n), ("m", m), ("source_wires", wires),
                          ("source_unitary_primitives", unitary), ("source_materialized_operations", operations),
                          ("normalization_squared", normalization)):
        _int(ledger.get(key), expected, key)
    _int(ledger.get("N", ledger.get("vertices")), vertices, "Graph vertices")
    _require(ledger.get("edges") == [list(edge) for edge in edges]
             and ledger.get("signs") == [1] * m and ledger.get("per_pattern_coefficients") == [m, m, 2],
             "Saved ordered graph, signs or coefficients changed")
    _require(ledger.get("source_counts") == counts and ledger.get("observed_source_counts") == counts,
             "Saved source count ledgers disagree")
    _int(ledger.get("observed_source_operations"), operations, "Observed materialized operations")
    _require(type(ledger.get("preprocessing_counts")) is dict and bool(ledger["preprocessing_counts"]),
             "Classical preprocessing counts are missing")
    records = _source_counts(buffers[prefix + "/operations.ndjson"], wires, counts)
    return ledger, records


def _validation(buffers):
    manifest = _json(buffers["validation/manifest.json"], "Validation manifest")
    result = _json(buffers["validation/validation.json"], "C5 validation")
    _require(manifest.get("schema") == "openquantum.dqi.degree_two_validation_manifest.v1"
             and manifest.get("status") == "complete_validation_component_pending_invocation_acceptance"
             and manifest.get("requires_observed_cli_exit_zero") is True,
             "Validation component is incomplete")
    _require(set(manifest.get("files", {})) == {"validation.json"}, "Validation file graph changed")
    _pin(buffers["validation/validation.json"], manifest["files"]["validation.json"], "Validation member")
    _component_sources(manifest.get("sources"), buffers,
                       script="scripts/validate_optimization_dqi_degree_two.py",
                       module="src/openquantum/optimization_dqi_degree_two_verify.py")
    _require(manifest.get("input_manifest_sha256") == _sha(buffers["c5/manifest.json"]),
             "Validation source-input manifest differs from the saved C5 source")
    _require(result.get("schema") == "openquantum.dqi.degree_two_validation.v1"
             and result.get("status") == "passed_c5_degree_two_component_pending_invocation_acceptance",
             "No successful bounded C5 validation component is saved")
    _require(result.get("source_counts") == C5_COUNTS, "Validated source ledger changed")
    for key, expected in (("source_wires", 12), ("source_unitary_primitives", 1773),
                          ("source_materialized_operations", 1796), ("normalization_squared", 190)):
        _int(result.get(key), expected, key)
    _require(type(result.get("counters")) is dict, "Validation counters missing")
    for key, expected in {**COUNTERS, "verification_invocations": 1,
                          "primitive_unitary_applications": 1773}.items():
        _int(result["counters"].get(key), expected, key)
    _require(result["counters"].get("spent") is True and result["counters"].get("active") is False
             and result["counters"].get("complete") is True
             and result["counters"].get("counter_scope")
             == "charged entries; successful component and complete flag required",
             "Charged roles require a completed successful validator")
    component = result.get("cli_component")
    _require(type(component) is dict and component.get("sources") == manifest["sources"]
             and component.get("source_input_manifest_sha256") == manifest["input_manifest_sha256"]
             and component.get("source_input_files") == _json(buffers["c5/manifest.json"], "C5 manifest")["files"]
             and component.get("budget_snapshot") == result["counters"]
             and component.get("requires_observed_cli_exit_zero") is True
             and component.get("retry") is False and component.get("publication_complete") is False,
             "Validation CLI source/input/completion bindings changed")
    for key in FALSE_FLAGS + VALIDATION_FALSE_FLAGS:
        _require(result.get(key) is False, "Saved scope flag changed: " + key)
    for key in ("native_synthesis_calls", "direct_native_estimator_calls"):
        _int(result.get(key), 0, key)
    _require(result.get("assignment_probability_scope") == "unconditional marginal over every clause and workspace value",
             "Assignment probabilities require every auxiliary branch")
    _require(result.get("numerical_tolerance") == {
        "absolute": ATOL, "relative": 0.0,
        "scope": "floating matrix/state diagnostics; no interval certificate"}, "Recorded tolerance changed")
    for key in ("assignment_total_variation", "full_state_l2_discrepancy",
                "max_state_entry_discrepancy", "clause_workspace_leakage_probability"):
        _error(result.get(key), key)
    _require(result.get("independent_optimum_probability_fraction") == "18/19"
             and result.get("independent_expected_objective_fraction") == "74/19", "Independent scalar references changed")
    _require(0 <= _finite(result.get("optimum_probability"), "Saved optimum probability") <= 1
             and 0 <= _finite(result.get("expected_objective"), "Saved expected objective") <= 5,
             "Saved summary values are outside their ranges")
    _int(result.get("quantum_storage_cap_bytes"), 32 << 20, "Quantum source envelope cap")
    _require(type(result.get("quantum_storage_bound_bytes")) is int
             and 0 < result["quantum_storage_bound_bytes"] <= 32 << 20
             and type(result.get("quantum_storage_scope")) is str, "Saved source-envelope account is missing")
    state = result.get("full_state_real_imag")
    _require(type(state) is list and len(state) == 4096, "Saved full-state shape changed")
    for pair in state:
        _require(type(pair) is list and len(pair) == 2, "Saved complex scalar pair is invalid")
        _finite(pair[0], "Saved real component")
        _finite(pair[1], "Saved imaginary component")
    outcomes = result.get("outcomes")
    _require(type(outcomes) is list and len(outcomes) == 16, "Saved outcome table changed")
    for integer, (outcome, numerator) in enumerate(zip(outcomes, SIGNED_NUMERATORS)):
        _require(type(outcome) is dict, "Saved outcome must be an object")
        _int(outcome.get("assignment_integer"), integer, "Assignment order")
        _require(outcome.get("display_bits") == format(integer, "04b"), "Gauge bit order changed")
        _int(outcome.get("signed_reference_numerator"), numerator, "Signed reference numerator")
        _int(outcome.get("reference_denominator_squared"), 190, "Signed reference denominator")
        _int(outcome.get("objective"), 0 if integer == 0 else 4 if numerator == 6 else 2, "Edge-based objective")
        pair = outcome.get("clean_auxiliary_amplitude_real_imag")
        _require(type(pair) is list and len(pair) == 2, "Saved clean-auxiliary amplitude pair changed")
        _finite(pair[0], "Saved clean real amplitude")
        _finite(pair[1], "Saved clean imaginary amplitude")
        _require(outcome.get("reference_probability_fraction") == ("0" if numerator == 0 else "18/95" if numerator == 6 else "1/190")
                 and 0 <= _finite(outcome.get("unconditional_probability"), "Saved marginal") <= 1,
                 "Saved marginal/reference is invalid")
    _require(type(result.get("checkpoints")) is dict
             and set(result["checkpoints"]) == {"preparation", "sign", "syndrome", "decoder", "final"},
             "Saved cleanup checkpoints are missing")
    for stage, checkpoint in result["checkpoints"].items():
        _require(type(checkpoint) is dict and checkpoint.get("postselection_performed") is False,
                 "Checkpoint cannot use postselection")
        _require(1 - ATOL <= _finite(checkpoint.get("norm_squared"), "Saved checkpoint norm") <= 1 + ATOL,
                 "Saved checkpoint norm exceeds its recorded tolerance")
        for key in ("max_state_entry_discrepancy", "full_state_l2_discrepancy",
                    "max_imaginary_component", "auxiliary_leakage_probability"):
            _error(checkpoint.get(key), "Saved " + stage + " " + key)
        if stage in {"decoder", "final"}:
            _error(checkpoint.get("clause_workspace_leakage_probability"), "Saved cleanup leakage")
        if stage == "preparation":
            _error(checkpoint.get("weight_above_two_probability"), "Saved prepared-support leakage")
            _int(checkpoint.get("expected_prepared_support_size"), 16, "Prepared support")
            _require(checkpoint.get("counter_and_flag_zero_before_sign") is True,
                     "Saved prefix workspace must be clean before sign")
    boolean = result.get("boolean_decoder_check")
    _require(type(boolean) is dict
             and boolean.get("status") == "passed_boolean_clean_arbitrary_target_decoder"
             and boolean.get("initially_dirty_scratch_target_law_promised") is False,
             "Saved Boolean decoder scope changed")
    for key, expected in (("mappings", 512), ("syndromes", 16), ("clause_targets", 32),
                          ("clean_scratch_inputs_per_mapping", 1), ("scratch_bits", 3),
                          ("full_statevector_runs_for_decoder_mappings", 0)):
        _int(boolean.get(key), expected, "Boolean " + key)
    ccx = result.get("small_ccx_reference")
    _require(type(ccx) is dict and ccx.get("status") == "passed_small_phase_exact_ccx_reference"
             and ccx.get("global_phase_fitting_performed") is False, "Saved small CCX scope changed")
    for key, expected in (("matrix_dimension", 8), ("basis_columns", 8), ("primitive_gates", 15)):
        _int(ccx.get(key), expected, "Small CCX " + key)
    _error(ccx.get("max_entry_discrepancy"), "Saved small CCX discrepancy")
    matrix = ccx.get("matrix_real_imag")
    _require(type(matrix) is list and len(matrix) == 8, "Saved small matrix row shape changed")
    for row in matrix:
        _require(type(row) is list and len(row) == 8, "Saved small matrix column shape changed")
        for pair in row:
            _require(type(pair) is list and len(pair) == 2, "Saved small matrix scalar shape changed")
            _finite(pair[0], "Saved small matrix real scalar")
            _finite(pair[1], "Saved small matrix imaginary scalar")
    _require(result.get("output_contract") == {
        "objective_condition": "f(x)=4", "ideal_success_target_fraction": "99/100",
        "ideal_optimum_probability_fraction": "18/19", "ideal_draws_for_99": 2,
        "ideal_miss_two_draws_fraction": "1/361", "prospective_complete_task_draws": 3,
        "ideal_miss_three_draws_fraction": "1/6859", "prospective_realization_ceiling_per_draw": "1/1200",
        "source_explicit_t_tdg_three_draws": 2289, "physical_budget_met": False,
        "factory_availability_budget_met": False, "draws_sampled": False},
        "Saved analytic draw contract scope or literal record changed")
    return result


def _markdown(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _tables(validation, c5, figure):
    outcomes = ["| assignment | objective | signed numerator / sqrt(190) | exact probability | saved Re | saved Im | unconditional probability |",
                "|---|---:|---:|---:|---:|---:|---:|"]
    for row in validation["outcomes"]:
        real, imaginary = row["clean_auxiliary_amplitude_real_imag"]
        outcomes.append(f"| {row['display_bits']} | {row['objective']} | {row['signed_reference_numerator']} | "
                        f"{_markdown(row['reference_probability_fraction'])} | {real:.12g} | {imaginary:.12g} | "
                        f"{row['unconditional_probability']:.12g} |")
    ledger = ["| recorded per draw | C5 checked circuit | figure-eight generation only |",
              "|---|---:|---:|"]
    for key, label in (("N", "graph vertices"), ("m", "edge clauses"), ("n", "gauge variables"),
                       ("source_wires", "logical wires"), ("source_unitary_primitives", "unitary primitives"),
                       ("source_materialized_operations", "operations with reset/readout/barriers")):
        ledger.append(f"| {label} | {c5.get(key, c5.get('vertices'))} | {figure.get(key, figure.get('vertices'))} |")
    for gate in ("h", "cx", "t", "tdg", "x", "ry", "z", "reset", "measure", "barrier"):
        ledger.append(f"| {gate} | {c5['source_counts'][gate]} | {figure['source_counts'][gate]} |")
    preprocessing = ["| saved classical preprocessing field | C5 | figure-eight |", "|---|---:|---:|"]
    keys = sorted(set(c5["preprocessing_counts"]) | set(figure["preprocessing_counts"]))
    _require(len(keys) <= 64, "Preprocessing field graph exceeds its display cap")
    for key in keys:
        preprocessing.append(f"| {_markdown(key)} | {_markdown(c5['preprocessing_counts'].get(key, '—'))} | "
                             f"{_markdown(figure['preprocessing_counts'].get(key, '—'))} |")
    checkpoints = ["| saved single-pass checkpoint | diagnostics |", "|---|---|"]
    for key in ("preparation", "sign", "syndrome", "decoder", "final"):
        if key in validation["checkpoints"]:
            value = json.dumps(validation["checkpoints"][key], sort_keys=True, separators=(",", ":"), allow_nan=False)
            _require(len(value) <= 8192, "Checkpoint display exceeds its cap")
            checkpoints.append(f"| {key} | {_markdown(value)} |")
    return {"outcomes": outcomes, "ledger": ledger, "preprocessing": preprocessing, "checkpoints": checkpoints}


def verify_saved_bundle(directory, *, expected_manifest_sha256=None, expected_source_pins=None):
    """Read saved bytes only; callers may bind the manifest to external provenance."""
    root = _directory(directory)
    total = _inventory(root)
    manifest_raw = _read(root, "bundle.json", MAX_MANIFEST_BYTES)
    manifest_hash = _sha(manifest_raw)
    if expected_manifest_sha256 is not None:
        _require(type(expected_manifest_sha256) is str and SHA_RE.fullmatch(expected_manifest_sha256)
                 and manifest_hash == expected_manifest_sha256, "External bundle manifest identity changed")
    manifest = _json(manifest_raw, "Bundle manifest")
    _require(type(manifest) is dict and manifest.get("schema") == SCHEMA
             and set(manifest.get("files", {})) == MEMBERS, "Closed bundle manifest schema changed")
    _require(type(manifest.get("contract_sha256")) is str and SHA_RE.fullmatch(manifest["contract_sha256"]),
             "Bundle needs its frozen source/command contract identity")
    buffers = {}
    for name in sorted(MEMBERS):
        buffers[name] = _read(root, name)
        _pin(buffers[name], manifest["files"][name], "Bundle member " + name, kind=True)
    notebook_raw = _read(root, NOTEBOOK, MAX_NOTEBOOK_BYTES)
    notebook = manifest.get("notebook")
    _require(type(notebook) is dict and notebook.get("path") == NOTEBOOK
             and notebook.get("code_cells") == 4
             and type(notebook.get("source_template_sha256")) is str
             and SHA_RE.fullmatch(notebook["source_template_sha256"]), "Notebook source provenance is missing")
    projection = notebook_source_sha256(notebook_raw)
    _require(notebook.get("source_projection_sha256") == projection, "Notebook source/type projection changed")
    source_pins = manifest.get("source_pins")
    _require(type(source_pins) is dict and set(source_pins) == SOURCE_FILES, "Public source pin graph changed")
    for name in SOURCE_FILES:
        _pin(buffers[name], source_pins[name], "Source member " + name)
    if expected_source_pins is not None:
        _require(source_pins == expected_source_pins, "Externally frozen source pins changed")
    acceptance = _json(buffers["invocation-acceptance.json"], "Component invocation acceptance")
    _require(type(acceptance) is dict and acceptance.get("schema") == ACCEPTANCE_SCHEMA
             and acceptance.get("status") == ACCEPTANCE_STATUS,
             "Successful component CLI observations are required before reading the notebook")
    _require(acceptance.get("contract_sha256") == manifest["contract_sha256"]
             and acceptance.get("source_pins") == source_pins, "Component source/contract identity changed")
    stages = acceptance.get("stages")
    names = ("generator-c5", "validator-c5", "generator-figure-eight")
    _require(type(stages) is list and len(stages) == 3, "Component sequence must have exactly three observations")
    for observation, name in zip(stages, names):
        _require(type(observation) is dict and observation.get("stage") == name,
                 "Component sequence order changed")
        _int(observation.get("returncode"), 0, "Observed component CLI exit")
        _int(observation.get("invocations"), 1, "Observed component CLI invocations")
    _require(type(acceptance.get("counters")) is dict, "Observed science counters missing")
    for key, expected in COUNTERS.items():
        _int(acceptance["counters"].get(key), expected, "Observed " + key)
    for key in FALSE_FLAGS:
        _require(acceptance.get(key) is False, "Component scope flag changed: " + key)
    for name, raw in buffers.items():
        if name.endswith(".json"):
            _privacy(_json(raw, name))
    _privacy(manifest)
    c5, records = _generation(buffers, "c5", vertices=5, edges=C5_EDGES, n=4, m=5,
                               wires=12, unitary=1773, operations=1796, normalization=190, counts=C5_COUNTS)
    figure, _ = _generation(buffers, "figure-eight", vertices=9, edges=FIGURE_EDGES, n=8, m=10,
                            wires=24, unitary=12981, operations=13020, normalization=1280, counts=FIGURE_COUNTS)
    validation = _validation(buffers)
    _require({key: validation["counters"][key] for key in COUNTERS}
             == {key: acceptance["counters"][key] for key in COUNTERS},
             "Validation and external observation science counters differ")
    tables = _tables(validation, c5, figure)
    excerpt = "\n".join(f"{index:04d} {json.dumps(record, sort_keys=True, separators=(',', ':'))}"
                        for index, record in enumerate(records[:40]))
    return {"schema": "openquantum.dqi.degree_two_saved_reader_report.v1",
            "status": "passed_saved_degree_two_bundle", "files": 22, "bytes": total,
            "manifest_sha256": manifest_hash, "notebook_source_projection_sha256": projection,
            "scope": "saved byte/schema/scalar/count inspection only; no notebook or publication acceptance",
            "scientific_work_reexecuted": False, "publication_complete": False,
            "validation": validation, "c5_ledger": c5, "figure_eight_ledger": figure,
            "acceptance": acceptance, "tables": tables, "source_excerpt": excerpt}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Complete twenty-two-file degree-two saved bundle")
    parser.add_argument("--expected-manifest-sha256")
    arguments = parser.parse_args(argv)
    try:
        result = verify_saved_bundle(arguments.directory,
                                     expected_manifest_sha256=arguments.expected_manifest_sha256)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
        print(json.dumps({"status": "failed_saved_degree_two_bundle", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({key: result[key] for key in ("schema", "status", "files", "bytes", "manifest_sha256",
                                                  "scope", "scientific_work_reexecuted", "publication_complete")},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
