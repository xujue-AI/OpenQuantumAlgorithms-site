#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read the portable structured DQI evidence using only the standard library.

The verifier checks saved primitive records, scalar references and file/source
bindings. It creates no quantum circuit and runs no simulation or generator.
``assemble_saved_bundle`` copies curated public sources and existing evidence;
it never manufactures a numerical validation record.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re


MAX_BUNDLE_BYTES = 1 << 20
MAX_PUBLIC_SOURCE_BYTES = 160 << 10
MAX_MANIFEST_FILES = 32
ATOL = 2e-12
ANGLE_TOLERANCE = 4e-15
SCHEMA = "openquantum.dqi.structured_saved_bundle.v1"
DEPENDENCY_SOURCE_PINS = {
    "qiskit/quantum_info/operators/operator.py": "45ccc994f579615a913e5b15264ceb2e674fd5113fe6c682a5ed00bcca3d62b7",
    "qiskit/quantum_info/states/statevector.py": "e36f4b204202cd2538b729a66d753aa7418f2b67fb63d987723fd4969e189ef5",
    "numpy/_core/einsumfunc.py": "4b4fc2d54ebe6b533f680fde2fa468d30449c2d44afc73041b6debb6e302dedf",
}
SOURCE_PATHS = {
    "optimization_dqi_structured.py": "src/openquantum/optimization_dqi_structured.py",
    "optimization_dqi_structured_validation.py": "src/openquantum/optimization_dqi_structured_validation.py",
    "generate_optimization_dqi_structured.py": "scripts/generate_optimization_dqi_structured.py",
    "validate_optimization_dqi_structured.py": "scripts/validate_optimization_dqi_structured.py",
    "verify_dqi_structured_saved.py": "scripts/verify_dqi_structured_saved.py",
    "optimization_dqi_structured.ipynb": "notebooks/optimization_dqi_structured.ipynb",
    "requirements-lock.txt": "requirements-lock.txt",
    "LICENSE.txt": "LICENSE",
}
SMALL_FILES = {
    "n2/dqi.qasm", "n2/operations.ndjson", "n2/ledger.json", "n2/circuit.txt",
    "n3/dqi.qasm", "n3/operations.ndjson", "n3/ledger.json",
}
LARGE_FILES = {
    "dqi-structured.qasm", "operations.ndjson", "optimization_dqi_structured.py",
    "generate_optimization_dqi_structured.py", "requirements-lock.txt", "LICENSE.txt",
}
REQUIRED_FILES = set(SOURCE_PATHS) | {"invocation-acceptance.json", "small/validated-small.json"} | {
    "small/" + name for name in SMALL_FILES
} | {"large/dqi-structured-generation.json"} | {
    "large/" + name for name in LARGE_FILES
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(directory, name, cap=MAX_BUNDLE_BYTES):
    relative = Path(name)
    _require(isinstance(name, str) and name and not relative.is_absolute()
             and len(relative.parts) <= 3 and all(part not in {".", ".."} for part in relative.parts),
             "Expected a bounded relative bundle path")
    directory = Path(directory).resolve()
    path = directory / relative
    _require(all(not part.is_symlink() for part in [path, *path.parents]
                 if part == directory or part.is_relative_to(directory)),
             "Bundle files must not redirect through symlinks")
    _require(path.resolve().is_relative_to(directory) and path.is_file(),
             "A required ordinary saved file is missing")
    _require(path.stat().st_size <= cap, "Saved file exceeds its byte cap")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    _require(len(raw) <= cap, "Saved file grew beyond its byte cap")
    return raw


def _inventory(directory, expected_names, cap=MAX_BUNDLE_BYTES):
    """Close the ordinary evidence tree; recognized local caches are not evidence."""
    directory = Path(directory).resolve()
    directories = {str(parent) for name in expected_names for parent in Path(name).parents
                   if str(parent) != "."}
    source_stems = {}
    for name in expected_names:
        relative = Path(name)
        if relative.suffix == ".py":
            location = str(relative.parent)
            source_stems.setdefault(location, set()).add(relative.stem)
            directories.add(str(relative.parent/"__pycache__"))
    found, cache_bytes, total_bytes, entries = set(), 0, 0, 0
    pending = [(directory, Path("."))]
    while pending:
        current, relative_parent = pending.pop()
        with os.scandir(current) as children:
            for entry in children:
                entries += 1
                _require(entries <= 64 and not entry.is_symlink(),
                         "Saved tree has too many entries or a symlink")
                relative = relative_parent/entry.name
                name = relative.as_posix()
                if entry.is_dir(follow_symlinks=False):
                    _require(name in directories, "Unexpected saved directory: " + name)
                    pending.append((Path(entry.path), relative))
                    continue
                _require(entry.is_file(follow_symlinks=False), "Saved tree contains a nonordinary file")
                size = entry.stat(follow_symlinks=False).st_size
                total_bytes += size
                _require(size <= cap and total_bytes <= cap, "Whole saved tree exceeds its byte ceiling")
                if name in expected_names:
                    found.add(name)
                    continue
                match = re.fullmatch(r"(.+)\.cpython-[0-9]{2,4}(?:\.opt-[12])?\.pyc", relative.name)
                location = str(relative.parent.parent)
                _require(relative.parent.name == "__pycache__" and match is not None
                         and match.group(1) in source_stems.get(location, set()),
                         "Unexpected unbound saved file: " + name)
                cache_bytes += size
    _require(found == expected_names, "Saved ordinary-file graph is incomplete")
    return {"bytes": total_bytes, "cache_bytes": cache_bytes, "entries": entries}


def _finite(value, label):
    _require(type(value) in {int, float} and math.isfinite(value), label + " must be finite")
    return float(value)


def _close(value, target, label):
    difference = abs(_finite(value, label) - target)
    _require(difference <= ATOL, label + " exceeds the saved-check tolerance")
    return difference


def _error(value, label):
    value = _finite(value, label)
    _require(0 <= value <= ATOL, label + " must be a nonnegative bounded error")
    return value


def _fraction(value, label):
    _require(isinstance(value, list) and len(value) == 2
             and all(type(part) is int for part in value) and value[1] > 0,
             label + " must be an integer numerator/positive denominator pair")
    result = Fraction(*value)
    _require([result.numerator, result.denominator] == value, label + " must be reduced")
    return result


def notebook_source_sha256(raw):
    notebook = json.loads(raw)
    _require(notebook.get("nbformat") == 4 and isinstance(notebook.get("cells"), list)
             and len(notebook["cells"]) <= 32, "Expected a bounded version-four notebook")
    projection = []
    for cell in notebook["cells"]:
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(part, str) for part in source):
            source = "".join(source)
        _require(cell.get("cell_type") in {"code", "markdown", "raw"}
                 and isinstance(source, str), "Notebook cell type/source is invalid")
        projection.append({"cell_type": cell["cell_type"], "source": source})
    return _sha(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode())


def _counts(n):
    return {"h": 6*n*n - 3*n - 6, "cx": 18*n*n - 8*n - 18,
            "t": 12*n*n - 8*n - 12, "tdg": 9*n*n - 6*n - 9,
            "x": 3*n*n - n, "ry": 2*n + 1, "z": 1,
            "reset": 3*n, "measure": n, "barrier": 7}


def _reference_operations(n, angles):
    """Expected saved primitive chronology, not a quantum circuit object."""
    q = 3*n
    clauses = list(range(n + 1))
    assignment = list(range(n + 1, 2*n + 1))
    scratch = list(range(2*n + 1, q))
    operations = []

    def append(name, wires, classical=(), parameters=()):
        operations.append([name, list(wires), list(classical), list(parameters)])

    def ccx(a, b, target):
        for name, wires in [
            ("h", (target,)), ("cx", (b, target)), ("tdg", (target,)),
            ("cx", (a, target)), ("t", (target,)), ("cx", (b, target)),
            ("tdg", (target,)), ("cx", (a, target)), ("t", (b,)),
            ("t", (target,)), ("h", (target,)), ("cx", (a, b)),
            ("t", (a,)), ("tdg", (b,)), ("cx", (a, b)),
        ]:
            append(name, wires)

    def conjunction(controls):
        blocks = [(controls[0], controls[1], scratch[0])]
        blocks += [(scratch[index - 2], controls[index], scratch[index - 1])
                   for index in range(2, len(controls))]
        return blocks

    def controlled_ry(control, target, angle):
        half = angle / 2
        _require(math.isfinite(angle) and 2*half == angle, "Saved angle halving loses information")
        append("ry", (target,), parameters=(half,))
        append("cx", (control, target))
        append("ry", (target,), parameters=(-half,))
        append("cx", (control, target))

    def barrier():
        append("barrier", range(q))

    for wire in range(q):
        append("reset", (wire,))
    barrier()
    for index, target in enumerate(clauses):
        angle = angles[index]
        controls = clauses[:index]
        if not controls:
            append("ry", (target,), parameters=(angle,))
            continue
        for control in controls:
            append("x", (control,))
        blocks = conjunction(controls) if len(controls) >= 2 else []
        for block in blocks:
            ccx(*block)
        controlled_ry(scratch[len(controls) - 2] if blocks else controls[0], target, angle)
        for block in reversed(blocks):
            ccx(*block)
        for control in reversed(controls):
            append("x", (control,))
    barrier()
    append("z", (n,))
    barrier()
    for index in range(n):
        append("cx", (index, assignment[index]))
    for index in range(n):
        append("cx", (n, assignment[index]))
    barrier()
    for index in range(n + 1):
        negatives = [assignment[j] for j in range(n) if index < n and j != index]
        for control in negatives:
            append("x", (control,))
        if n == 2:
            ccx(assignment[0], assignment[1], clauses[index])
        else:
            blocks = conjunction(assignment[:-1])
            for block in blocks:
                ccx(*block)
            ccx(scratch[n - 3], assignment[-1], clauses[index])
            for block in reversed(blocks):
                ccx(*block)
        for control in reversed(negatives):
            append("x", (control,))
    barrier()
    for wire in assignment:
        append("h", (wire,))
    barrier()
    for index, wire in enumerate(assignment):
        append("measure", (wire,), classical=(index,))
    barrier()
    return operations


def _operation_record(record, n):
    _require(isinstance(record, list) and len(record) == 4, "Expected a primitive NDJSON list")
    name, quantum, classical, parameters = record
    _require(name in _counts(n) and isinstance(quantum, list) and isinstance(classical, list)
             and isinstance(parameters, list), "Unknown primitive record")
    _require(all(type(wire) is int and 0 <= wire < 3*n for wire in quantum)
             and len(set(quantum)) == len(quantum), "Quantum indices are invalid")
    arity = 3*n if name == "barrier" else 2 if name == "cx" else 1
    _require(len(quantum) == arity, "Primitive arity changed")
    _require(classical == ([quantum[0] - n - 1] if name == "measure" else []),
             "Readout mapping changed")
    _require(all(type(bit) is int and 0 <= bit < n for bit in classical),
             "Classical readout indices are invalid")
    _require(len(parameters) == (1 if name == "ry" else 0), "Primitive parameters changed")
    for parameter in parameters:
        _finite(parameter, "RY parameter")
    return record


def _qasm_operations(raw, n):
    lines = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    _require(lines[:2] == ["OPENQASM 3.0;", 'include "stdgates.inc";']
             and set(lines[2:4]) == {f"bit[{n}] c;", f"qubit[{3*n}] q;"},
             "Saved QASM declarations changed")
    records = []
    for line in lines[4:]:
        _require(len(records) < 48*n*n - 20*n - 36, "Saved QASM exceeds operation cap")
        measured = re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];", line)
        if measured:
            classical, quantum = map(int, measured.groups())
            record = ["measure", [quantum], [classical], []]
        else:
            match = re.fullmatch(r"([a-z]+)(?:\(([^()]*)\))? (q\[\d+\](?:, q\[\d+\])*)\s*;", line)
            _require(match is not None, "Saved QASM contains an opaque instruction")
            name, parameters, wires = match.groups()
            record = [name, [int(value) for value in re.findall(r"q\[(\d+)\]", wires)], [],
                      [] if parameters is None else [float(value) for value in parameters.split(",")]]
        records.append(_operation_record(record, n))
    return records


def _check_operations(buffers, prefix, n, qasm_name, ledger):
    ndjson = buffers[prefix + "/operations.ndjson"]
    lines = ndjson.splitlines()
    _require(len(lines) == 48*n*n - 20*n - 36, "Saved primitive stream length changed")
    records = [_operation_record(json.loads(line), n) for line in lines]
    _require(records == _qasm_operations(buffers[prefix + "/" + qasm_name], n),
             "QASM and primitive record chronology disagree")
    rotations = [operation[3][0] for operation in records if operation[0] == "ry"]
    _require(len(rotations) == 2*n + 1, "Saved rotation chronology changed")
    angles = [rotations[0]] + [2*rotations[1 + 2*index] for index in range(n)]
    for index, angle in enumerate(angles):
        _require(0 < _finite(angle, "Saved preparation angle") < math.pi
                 and abs(angle - 2*math.asin(1/math.sqrt(n + 2 - index))) <= ANGLE_TOLERANCE,
                 "Saved preparation coefficient/angle reference changed")
    _require(records == _reference_operations(n, angles), "Saved primitive chronology/parameters changed")
    actual_counts = dict(Counter(operation[0] for operation in records))
    _require(actual_counts == _counts(n), "Saved primitive counts changed")
    barriers = [index for index, operation in enumerate(records) if operation[0] == "barrier"]
    starts = [0] + [index + 1 for index in barriers[:-1]]
    if ledger.get("schema") == "openquantum.dqi.structured_source_ledger.v1":
        _require(ledger["n"] == n and ledger["m"] == n + 1 and ledger["ell"] == 1
                 and ledger["total_qubits"] == 3*n and ledger["classical_readout_bits"] == n
                 and ledger["counts"] == actual_counts and ledger["unitary_gates"] == 48*n*n - 24*n - 43
                 and ledger["events_without_barriers"] == 48*n*n - 20*n - 43
                 and ledger["source_operations"] == len(records)
                 and ledger["physical_estimate_status"] == "pending",
                 "Generation source-ledger fields changed")
        intervals = ledger["stage_intervals"]
        names = ["reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout"]
        _require(set(intervals) == set(names), "Source-ledger stage names changed")
        _require([(intervals[name]["start"], intervals[name]["stop"]) for name in names]
                 == list(zip(starts, barriers)), "Generation source-ledger stage intervals changed")
        for name in names:
            item = intervals[name]
            _require(item["barrier_index"] == item["stop"] and item["operations"] == item["stop"] - item["start"],
                     "Generation stage boundary/count changed")
        _require(ledger["workspace"] == {"allocated_clean": n - 1, "preparation_peak_used": n - 1,
                    "decoder_peak_used": n - 2, "serial_reuse": True, "restored_after_each_predicate": True,
                    "clean_target_law_requires_zero_input": True}
                 and ledger["ccx_blocks"] == {"preparation": n*(n - 1), "decoder": (n + 1)*(2*n - 3)},
                 "Generation workspace/CCX accounting changed")
    else:
        _require(ledger["primitive_counts"] == actual_counts
                 and ledger["unitary_primitives"] == 48*n*n - 24*n - 43
                 and ledger["events_excluding_barriers"] == 48*n*n - 20*n - 43
                 and ledger["source_operations"] == len(records)
                 and ledger["compiled_clifford_t_demand_known"] is False,
                 "Saved actual logical-resource fields changed")
        _require([(stage["start"], stage["stop"]) for stage in ledger["stages"]]
                 == list(zip(starts, barriers)), "Saved actual stage intervals changed")
        _require(ledger["workspace"] == {"allocated_clean_wires": n - 1,
                    "preparation_peak_clean_wires": n - 1, "decoder_peak_clean_wires": n - 2,
                    "arbitrary_dirty_target_action_validated": False},
                 "Saved actual workspace scope changed")
    _require(ledger["unsynthesized_ry"] == 2*n + 1 and ledger["explicit_t_tdg"] == 21*n*n - 14*n - 21,
             "Unsynthesized rotation/partial T counts changed")
    return len(records), _sha(ndjson), angles


def _check_hashes(records, buffers, prefix, expected_names):
    _require(set(records) == expected_names, "Saved artifact inventory changed")
    for name, binding in records.items():
        raw = buffers[prefix + "/" + name]
        _require(binding == {"bytes": len(raw), "sha256": _sha(raw)},
                 "Saved artifact hash/length changed: " + name)


def _check_sources(provenance, buffers, names):
    sources = provenance["source_files"]
    _require(set(sources) == set(names), "Saved producer-source inventory changed")
    for name in names:
        binding = sources[name]
        raw = buffers[name]
        _require(binding == {"bytes": len(raw), "sha256": _sha(raw)},
                 "Saved producer source does not match its public copy")


def _check_small(record, buffers):
    _require(record.get("schema") == "openquantum.dqi.structured.validated_small.v1"
             and record.get("status") == "complete"
             and record.get("completion_is_provisional") is True
             and record.get("coherent_validation_performed") is True
             and record.get("old_default_reexecuted") is False,
             "No completed new structured-small validation is present")
    _check_hashes(record["files"], buffers, "small", SMALL_FILES)
    provenance = record["provenance"]
    _check_sources(provenance, buffers, ["optimization_dqi_structured.py",
                  "optimization_dqi_structured_validation.py", "validate_optimization_dqi_structured.py"])
    _require(set(provenance["environment_files"]) == {"requirements-lock.txt", "LICENSE.txt"},
             "Saved environment/license inventory changed")
    for name, binding in provenance["environment_files"].items():
        _require(binding == {"bytes": len(buffers[name]), "sha256": _sha(buffers[name])},
                 "Saved coherent environment/license bytes changed")
    _require(set(provenance["dependency_source_files"]) == set(DEPENDENCY_SOURCE_PINS),
             "Saved pinned backend-source inventory changed")
    for name, binding in provenance["dependency_source_files"].items():
        _require(set(binding) == {"bytes", "sha256"} and type(binding["bytes"]) is int
                 and 0 < binding["bytes"] <= MAX_BUNDLE_BYTES and binding["sha256"] == DEPENDENCY_SOURCE_PINS[name],
                 "Saved pinned backend-source identity changed")
    _require(isinstance(provenance["python"], str) and re.fullmatch(r"3\.12\.[0-9]+", provenance["python"])
             and provenance["packages"] == {"qiskit": "2.6.2", "numpy": "2.5.4"},
             "Saved coherent environment differs from the pinned check")
    commit = record.get("source_commit")
    _require(commit is None or (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)),
             "Optional source commit must be a full Git identity")
    _require(record.get("large_generation_performed") is False and record.get("hardware_or_synthesis_performed") is False,
             "Small validation scope cannot include larger generation or hardware/synthesis")
    _require(0 < _finite(record["time_limit_seconds"], "Small invocation deadline") <= 30
             and 0 <= _finite(record["elapsed_seconds_before_final_receipt"], "Small elapsed time") <= record["time_limit_seconds"],
             "Saved small invocation time/deadline changed")
    _require(type(record["max_output_bytes"]) is int and 1 <= record["max_output_bytes"] <= MAX_BUNDLE_BYTES
             and sum(len(raw) for name, raw in buffers.items() if name.startswith("small/")) <= record["max_output_bytes"],
             "Saved small output-byte ceiling changed")
    _require([case["n"] for case in record["cases"]] == [2, 3], "Expected the two new small fixtures")
    tables = {}
    report_cases = []
    errors = []
    for case in record["cases"]:
        n = case["n"]
        _require(case["m"] == n + 1 and case["ell"] == 1, "Structured family dimensions changed")
        spec = case["family_spec"]
        expected_matrix = [[int(i == j) for j in range(n)] for i in range(n)] + [[1]*n]
        _require(spec["matrix"] == expected_matrix and spec["signs"] == [0]*n + [1]
                 and spec["registers"] == {"clause": list(range(n + 1)),
                    "assignment": list(range(n + 1, 2*n + 1)), "workspace": list(range(2*n + 1, 3*n)),
                    "classical": list(range(n)), "total_qubits": 3*n},
                 "The saved fixture is outside the declared structured family")
        _require(case["absolute_tolerance"] == ATOL and case["relative_tolerance"] == 0
                 and case["decoder_operator_constructions"] == 1 and case["coherent_state_transitions"] == 4
                 and case["outcomes_sampled"] is False, "Saved check/repetition scope changed")
        decoder = case["decoder"]
        _require(decoder["clean_basis_columns"] == (1 << (2*n + 1))
                 and decoder["operator_dimension"] == (1 << (3*n))
                 and decoder["clean_workspace_only"] is True
                 and decoder["tolerance"] == ATOL and decoder["relative_tolerance"] == 0,
                 "Saved clean-decoder coverage/tolerance changed")
        errors.append(_error(decoder["max_complex_entry_error"], "Decoder entry error"))
        prep = case["preparation"]
        _require(prep["support_count"] == n + 2, "Prepared support count changed")
        errors += [_error(prep["amplitude_error"], "Preparation amplitude error"),
                   _close(prep["workspace_zero_probability"], 1, "Preparation workspace cleanup")]
        signed = case["signed_syndrome"]
        errors += [_error(signed["max_complex_error"], "Signed-syndrome error"),
                   _close(signed["workspace_zero_probability"], 1, "Signed-syndrome workspace cleanup")]
        decoded = case["decoded_state"]
        errors += [_error(decoded["max_complex_error"], "Decoded-state error"),
                   _close(decoded["clause_workspace_zero_probability"], 1, "Decoded clause/workspace cleanup")]
        final = case["final_state"]
        errors += [_error(final["max_complex_error"], "Final amplitude error"),
                   _error(final["max_probability_error"], "Final probability error"),
                   _close(final["clause_workspace_zero_probability"], 1, "Final clause/workspace cleanup")]
        _require(len(final["outcomes"]) == 1 << n, "Saved outcome table length changed")
        table = ["| assignment | satisfied | signed numerator | exact probability | saved probability |",
                 "|---|---:|---:|---:|---:|"]
        expectation = Fraction(0)
        optimum_probability = Fraction(0)
        denominator_squared = (1 << n)*(n + 2)
        for integer, row in enumerate(final["outcomes"]):
            bits = [(integer >> index) & 1 for index in range(n)]
            satisfied = sum(int(bit == 0) for bit in bits) + int(sum(bits) % 2 == 1)
            numerator = 1 + sum(1 if bit == 0 else -1 for bit in bits) - (1 if sum(bits) % 2 == 0 else -1)
            reference = Fraction(numerator*numerator, denominator_squared)
            _require(row["integer"] == integer and row["bitstring"] == f"{integer:0{n}b}"
                     and row["satisfied"] == satisfied and row["reference_signed_numerator"] == numerator
                     and row["reference_denominator_squared"] == denominator_squared
                     and _fraction(row["reference_probability"], "Reference probability") == reference,
                     "Saved outcome reference changed")
            errors += [_close(row["amplitude_real"], numerator/math.sqrt(denominator_squared), "Signed amplitude"),
                       _close(row["amplitude_imag"], 0, "Imaginary amplitude"),
                       _close(row["probability"], float(reference), "Outcome probability")]
            expectation += satisfied*reference
            if satisfied == n:
                optimum_probability += reference
            table.append(f"| {row['bitstring']} | {satisfied} | {numerator} | {reference} | {row['probability']:.12f} |")
        quality = case["quality"]
        _require(quality["maximum_satisfied"] == n
                 and _fraction(quality["reference_expected_satisfied"], "Expected objective") == expectation
                 and _fraction(quality["reference_maximum_probability"], "Optimum probability") == optimum_probability,
                 "Saved quality reference changed")
        errors += [_close(quality["expected_satisfied"], float(expectation), "Expected objective"),
                   _close(quality["maximum_probability"], float(optimum_probability), "Optimum probability"),
                   _error(case["max_error"], "Case maximum error")]
        ledger = case["actual_ledger"]
        ledger_file = json.loads(buffers[f"small/n{n}/ledger.json"])
        _require(ledger_file["schema"] == "openquantum.dqi.structured.primitive_ledger.v1"
                 and ledger_file["n"] == n and ledger_file["actual_ledger"] == ledger
                 and ledger_file["rotation_synthesis_performed"] is False,
                 "Saved ledger copy changed")
        count, operation_hash, angles = _check_operations(buffers, f"small/n{n}", n, "dqi.qasm", ledger)
        _check_operations(buffers, f"small/n{n}", n, "dqi.qasm", ledger_file["analytic_prediction"])
        _require(ledger_file["preparation_angles_hex"] == [angle.hex() for angle in angles],
            "Saved preparation coefficient chronology changed")
        tables[n] = table
        report_cases.append({"n": n, "source_operations": count,
                             "operation_stream_sha256": operation_hash,
                             "expected_satisfied": str(expectation), "optimum_probability": str(optimum_probability)})
    return tables, {"cases": report_cases, "maximum_recorded_error": max(errors),
                    "source_commit": commit, "python": provenance["python"], "packages": provenance["packages"]}


def _check_large(record, buffers):
    _require(record.get("schema") == "openquantum.dqi.structured_generation.v1"
             and record.get("status") == "generation_only_exports_complete"
             and record.get("completion_is_provisional") is True
             and record.get("n") == 8 and record.get("m") == 9 and record.get("ell") == 1
             and record.get("coherent_validation_performed") is False,
             "The larger artifact must remain n=8 generation-only evidence")
    _check_hashes(record["artifacts"], buffers, "large", LARGE_FILES)
    _require(buffers["large/optimization_dqi_structured.py"] == buffers["optimization_dqi_structured.py"]
             and buffers["large/generate_optimization_dqi_structured.py"] == buffers["generate_optimization_dqi_structured.py"]
             and buffers["large/requirements-lock.txt"] == buffers["requirements-lock.txt"]
             and buffers["large/LICENSE.txt"] == buffers["LICENSE.txt"],
             "Larger generation source/environment/license copies disagree")
    ledger = record["logical_resources"]
    count, operation_hash, _ = _check_operations(buffers, "large", 8, "dqi-structured.qasm", ledger)
    _require(record["source_operations"] == count and record["observed_operation_counts"] == _counts(8)
             and record["operation_stream_sha256"] == operation_hash
             and record["readout_map"] == [[9 + index, index] for index in range(8)]
             and record["observed_stage_intervals"] == ledger["stage_intervals"],
             "Larger generation observed fields changed")
    instance = record["instance"]
    _require(instance["family"] == "binary_identity_plus_all_ones_degree_one"
             and instance["n"] == 8 and instance["m"] == 9 and instance["ell"] == 1
             and instance["matrix"] == [[int(i == j) for j in range(8)] for i in range(8)] + [[1]*8]
             and instance["signs"] == [0]*8 + [1] and instance["domain_size"] == 10
             and instance["kernel_minimum_distance"] == 9,
             "Larger generation family/decoding promise changed")
    _check_repetition(record["repetition_plan"], 8)
    provenance = record["provenance"]
    _require(provenance["source_module"] == "optimization_dqi_structured.py"
             and provenance["source_sha256"] == _sha(buffers["optimization_dqi_structured.py"])
             and provenance["generator_sha256"] == _sha(buffers["generate_optimization_dqi_structured.py"])
             and provenance["environment_lock_sha256"] == _sha(buffers["requirements-lock.txt"])
             and provenance["upstream_code_adopted"] is False and provenance["random_work_performed"] is False
             and record["simulation_performed"] is False,
             "Larger source/environment/execution scope changed")
    limit_seconds = _finite(record["limits"]["time_limit_seconds"], "Generation deadline")
    _require(0 < limit_seconds <= 30
             and 0 <= _finite(record["elapsed_seconds_before_receipt"], "Generation elapsed time") <= limit_seconds,
             "Generation elapsed/deadline scope changed")
    limits = record["limits"]
    for name, low, high in (("max_n", 8, 64), ("max_qubits", 24, 192), ("max_operations", 2876, 200000)):
        _require(type(limits[name]) is int and low <= limits[name] <= high,
                 "Larger materialization limit changed")
    _require(type(limits["max_output_bytes"]) is int and 1 <= limits["max_output_bytes"] <= 32*(1 << 20)
             and sum(len(raw) for name, raw in buffers.items() if name.startswith("large/")) <= limits["max_output_bytes"],
             "Larger materialization byte limit changed")
    return {"n": 8, "source_operations": count, "coherent_validation_performed": False,
            "operation_stream_sha256": operation_hash}


def _check_repetition(plan, n):
    good = Fraction((n + 1)*n*n, (1 << n)*(n + 2))
    _require(plan["optimal_probability"] == {"numerator": good.numerator, "denominator": good.denominator},
             "Ideal one-draw probability changed")
    target = plan["requested_failure"]
    _require(set(target) == {"numerator", "denominator"}
             and all(type(value) is int and value.bit_length() <= 256 for value in target.values())
             and target["denominator"] > 0, "Ideal confidence target is invalid")
    delta = Fraction(target["numerator"], target["denominator"])
    _require(0 < delta < 1 and target == {"numerator": delta.numerator, "denominator": delta.denominator},
             "Ideal confidence target must be reduced and strictly between zero and one")
    cap = plan["max_draws"]
    _require(type(cap) is int and 1 <= cap <= 1024, "Ideal draw cap exceeds its bound")
    failure, required = Fraction(1), None
    for draws in range(1, cap + 1):
        failure *= 1 - good
        if failure <= delta:
            required = draws
            break
    charged = required if required is not None else cap
    _require(plan["status"] == ("feasible" if required is not None else "cap_exceeded")
             and plan["required_draws"] == required and plan["quality_feasible"] is (required is not None)
             and plan["minimum_required_draws_lower_bound"] == (required if required is not None else cap + 1)
             and plan["draws_charged"] == charged
             and plan["failure_probability_after_charged_draws"] == {
                 "numerator_hex": format(failure.numerator, "x"), "denominator_hex": format(failure.denominator, "x")}
             and plan["serial_peak_qubits"] == 3*n
             and plan["complete_unitary_gates"] == charged*(48*n*n - 24*n - 43)
             and plan["complete_source_operations"] == charged*(48*n*n - 20*n - 36)
             and plan["complete_counts"] == {name: charged*count for name, count in _counts(n).items()},
             "Capped complete-draw ideal repetition accounting changed")


def _accepted_observation(observation):
    _require(isinstance(observation, dict), "Provide an explicit successful caller observation")
    kind = observation.get("kind")
    if kind == "cli_exit":
        allowed = {"kind", "returncode", "stage_receipt_sha256", "raw_waitpid_status"}
        _require(set(observation) <= allowed and {"kind", "returncode", "stage_receipt_sha256"} <= set(observation)
                 and type(observation["returncode"]) is int and observation["returncode"] == 0
                 and isinstance(observation["stage_receipt_sha256"], str)
                 and re.fullmatch(r"[0-9a-f]{64}", observation["stage_receipt_sha256"])
                 and ("raw_waitpid_status" not in observation
                      or type(observation["raw_waitpid_status"]) is int and observation["raw_waitpid_status"] == 0),
                 "CLI invocation acceptance requires observed exit zero and its stage-receipt hash")
    elif kind == "api_return":
        _require(set(observation) == {"kind", "returned"} and observation["returned"] is True,
                 "Normal API acceptance requires explicit successful-return observation")
    else:
        raise ValueError("Invocation acceptance must observe a normal API return or actual CLI exit")
    return dict(observation)


def _acceptance_record(observations, buffers):
    _require(isinstance(observations, dict) and set(observations) == {"small", "large"},
             "Producer file statuses alone are insufficient: supply actual small/large invocation acceptance")
    return {"schema": "openquantum.dqi.structured_invocation_acceptance.v1",
            "scope": "caller-observed small and large producer acceptance; no overall notebook, guardian or publication claim",
            "producers": {name: {"observation": _accepted_observation(observations[name]),
                                 "receipt_sha256": _sha(buffers["small/validated-small.json"]
                                                        if name == "small" else buffers["large/dqi-structured-generation.json"])}
                          for name in ("small", "large")}}


def _check_acceptance(buffers):
    record = json.loads(buffers["invocation-acceptance.json"])
    _require(record.get("schema") == "openquantum.dqi.structured_invocation_acceptance.v1"
             and set(record["producers"]) == {"small", "large"},
             "The complete bundle needs separate producer-invocation acceptance")
    for name, receipt_name in (("small", "small/validated-small.json"), ("large", "large/dqi-structured-generation.json")):
        producer = record["producers"][name]
        _accepted_observation(producer["observation"])
        _require(producer["receipt_sha256"] == _sha(buffers[receipt_name]),
                 "Accepted invocation does not bind the actual producer receipt")
    return record


def _buffers(directory):
    return {name: _read(directory, name) for name in REQUIRED_FILES}


def _check_saved_data(buffers):
    acceptance = _check_acceptance(buffers)
    tables, small_report = _check_small(json.loads(buffers["small/validated-small.json"]), buffers)
    large_report = _check_large(json.loads(buffers["large/dqi-structured-generation.json"]), buffers)
    circuit_text = buffers["small/n2/circuit.txt"].decode("utf-8")
    _require(0 < len(buffers["small/n2/circuit.txt"]) <= 65536, "Saved small circuit text exceeds its cap")
    return {"small": json.loads(buffers["small/validated-small.json"]),
            "large": json.loads(buffers["large/dqi-structured-generation.json"]),
            "tables": tables, "verification": {"small": small_report, "large": large_report},
            "circuit_text": circuit_text, "invocation_acceptance": acceptance}


def verify_saved_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads(_read(directory, "bundle.json"))
    _require(manifest.get("schema") == SCHEMA, "Expected a structured DQI saved bundle")
    _require(manifest.get("status") == "complete",
             "Structured DQI evidence is source-only: the fresh small validation and larger generation must finish first")
    _require(set(manifest["files"]) == REQUIRED_FILES and len(manifest["files"]) <= MAX_MANIFEST_FILES,
             "Saved bundle graph is incomplete or changed")
    tree = _inventory(directory, REQUIRED_FILES | {"bundle.json"})
    buffers = _buffers(directory)
    _require(sum(map(len, buffers.values())) + len(_read(directory, "bundle.json")) + tree["cache_bytes"] <= MAX_BUNDLE_BYTES,
             "Portable saved bundle exceeds its one-MiB cap")
    for name, binding in manifest["files"].items():
        raw = buffers[name]
        if name.endswith(".ipynb"):
            _require(binding == {"kind": "notebook_sources", "source_sha256": notebook_source_sha256(raw)},
                     "Reusable notebook source projection changed")
        else:
            _require(binding == {"kind": "file", "bytes": len(raw), "sha256": _sha(raw)},
                     "Saved bundle file binding changed: " + name)
    return _check_saved_data(buffers)


def assemble_saved_bundle(source_root, small_dir, large_dir, output_dir, *, max_output_bytes=655360,
                          invocation_acceptance=None):
    """Copy existing public files/evidence and write a verified manifest last.

    In-place ``output_dir/small`` and ``output_dir/large`` inputs avoid duplicate
    result copies in a bounded run. Otherwise the output directory must be fresh.
    ``source_root`` may be the ordinary repository or the flat public sources.
    Explicit caller-observed successful API returns or CLI exits are required;
    producer receipt status alone is not invocation acceptance. No project
    module, simulator, notebook or generator is invoked here.
    """
    _require(type(max_output_bytes) is int and 1 <= max_output_bytes <= MAX_BUNDLE_BYTES,
             "Assembly output cap can only tighten one MiB")
    source_root, small_dir, large_dir, output_dir = map(Path, (source_root, small_dir, large_dir, output_dir))
    output_dir = output_dir.resolve()
    in_place = small_dir.resolve() == output_dir/"small" and large_dir.resolve() == output_dir/"large"
    if output_dir.exists():
        _require(in_place and output_dir.is_dir(),
                 "Assembly requires a fresh output or only its exact in-place evidence directories")
        existing = set()
        with os.scandir(output_dir) as children:
            for entry in children:
                _require(entry.name in {"small", "large"} and not entry.is_symlink()
                         and entry.is_dir(follow_symlinks=False),
                         "In-place assembly root may contain only ordinary small/large directories")
                existing.add(entry.name)
        _require(existing == {"small", "large"}, "In-place evidence directories are incomplete")
    small_tree = _inventory(small_dir, SMALL_FILES | {"validated-small.json"}, max_output_bytes)
    large_tree = _inventory(large_dir, LARGE_FILES | {"dqi-structured-generation.json"}, max_output_bytes)
    source_buffers = {name: _read(source_root, name if (source_root/name).is_file() else relative)
                      for name, relative in SOURCE_PATHS.items()}
    _require(sum(map(len, source_buffers.values())) <= MAX_PUBLIC_SOURCE_BYTES,
             "Curated public source/download files exceed their 160-KiB ceiling")
    data_buffers = {"small/validated-small.json": _read(small_dir, "validated-small.json"),
                    "large/dqi-structured-generation.json": _read(large_dir, "dqi-structured-generation.json")}
    data_buffers.update({"small/" + name: _read(small_dir, name) for name in SMALL_FILES})
    data_buffers.update({"large/" + name: _read(large_dir, name) for name in LARGE_FILES})
    buffers = source_buffers | data_buffers
    acceptance_raw = (json.dumps(_acceptance_record(invocation_acceptance, buffers), sort_keys=True, indent=2) + "\n").encode()
    buffers["invocation-acceptance.json"] = acceptance_raw
    cache_bytes = small_tree["cache_bytes"] + large_tree["cache_bytes"] if in_place else 0
    _require(sum(map(len, buffers.values())) + cache_bytes <= max_output_bytes - 32768,
             "Existing public sources/evidence leave insufficient manifest byte reserve")
    verified = _check_saved_data(buffers)
    manifest = {"schema": SCHEMA, "status": "complete",
                "scope": "Saved n=2/3 clean-workspace coherent checks and n=8 generation-only; no hardware, synthesis or physical estimate",
                "completion_scope": "invocation-accepted saved artifact graph only; notebook, guardian and publication completion are separate",
                "cache_policy": "Recognized ordinary Python bytecode caches are not evidence; they are counted in the local byte cap and not read by this verifier",
                "files": {name: ({"kind": "notebook_sources", "source_sha256": notebook_source_sha256(raw)}
                                 if name.endswith(".ipynb") else {"kind": "file", "bytes": len(raw), "sha256": _sha(raw)})
                          for name, raw in sorted(buffers.items())}}
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    _require(sum(map(len, buffers.values())) + len(manifest_raw) + cache_bytes <= max_output_bytes,
             "Assembled bundle exceeds its byte cap")
    output_dir.mkdir(parents=True, exist_ok=in_place)
    for name, raw in buffers.items():
        if in_place and name in data_buffers:
            continue
        path = output_dir/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    with (output_dir/"bundle.json").open("xb") as stream:
        stream.write(manifest_raw)
    final_tree = _inventory(output_dir, REQUIRED_FILES | {"bundle.json"}, max_output_bytes)
    return {"manifest_sha256": _sha(manifest_raw), "files": len(buffers) + 1,
            "bytes": final_tree["bytes"],
            "verification": verified["verification"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    result = verify_saved_bundle(arguments.directory)
    print(json.dumps(result["verification"], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
