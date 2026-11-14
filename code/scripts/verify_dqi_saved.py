#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the portable saved DQI bundle using only Python's standard library.

Run ``python verify_dqi_saved.py PATH_TO_BUNDLE`` or import
``verify_saved_bundle``. This verifies saved files and scalar references;
it performs no circuit construction, simulation, sampling or estimation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

MAX_BUNDLE_BYTES = 1 << 20
MAX_OPERATIONS = 512
ATOL = 2e-12
EXPECTED_COUNTS = {"h": 47, "t": 88, "tdg": 66, "cx": 144, "x": 24,
                   "ry": 7, "z": 1, "reset": 9, "measure": 3, "barrier": 7}
EXPECTED_STAGES = [
    {"name": "input_reset", "start": 0, "stop": 9},
    {"name": "weight_zero_one_preparation", "start": 10, "stop": 125},
    {"name": "clause_sign_phase", "start": 126, "stop": 127},
    {"name": "syndrome_B_transpose_y", "start": 128, "stop": 134},
    {"name": "reversible_decoder_and_cleanup", "start": 135, "stop": 387},
    {"name": "assignment_hadamards", "start": 388, "stop": 391},
    {"name": "assignment_readout", "start": 392, "stop": 395},
]
REQUIRED_FILES = {
    "dqi-validated.json", "dqi.qasm", "dqi-circuit.svg", "optimization_dqi.py",
    "generate_optimization_dqi.py", "verify_dqi_saved.py", "requirements-lock.txt",
    "LICENSE.txt", "optimization_dqi.ipynb",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_file(directory: Path, name: str) -> bytes:
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Bundle file names must be plain basenames")
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError("Bundle files must be ordinary files inside the bundle")
    if not path.is_file() or path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("A saved bundle file is missing or exceeds its byte cap")
    with path.open("rb") as stream:
        data = stream.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("A saved bundle file grew beyond its byte cap")
    return data


def notebook_source_sha256(data: bytes) -> str:
    """Bind reusable notebook source while permitting new execution outputs."""
    notebook = json.loads(data)
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Expected a version-four notebook")
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            raise ValueError("Unexpected notebook cell type")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(part, str) for part in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError("Notebook source must be text")
        cells.append({"cell_type": cell["cell_type"], "source": source})
    encoded = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def qasm_operations(text: str) -> list:
    """Read only the saved default's bounded primitive grammar, retaining order.

    This is not a full OpenQASM conformance parser or another circuit simulation.
    """
    declarations = ["OPENQASM 3.0;", 'include "stdgates.inc";', "bit[3] c;", "qubit[9] q;"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines[:4] != declarations:
        raise ValueError("Saved QASM declarations disagree with the default")
    operations = []
    for line in lines[4:]:
        if len(operations) >= MAX_OPERATIONS:
            raise ValueError("QASM operation cap exceeded")
        measured = re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];", line)
        if measured:
            classical, quantum = map(int, measured.groups())
            if not 0 <= classical < 3 or not 0 <= quantum < 9:
                raise ValueError("Saved readout indices exceed their registers")
            operations.append(["measure", [quantum], [classical], []])
            continue
        match = re.fullmatch(r"([a-z]+)(?:\(([^()]*)\))? (q\[\d+\](?:, q\[\d+\])*)\s*;", line)
        if match is None:
            raise ValueError("Saved QASM contains an unlisted instruction")
        name, arguments, wires = match.groups()
        if name not in EXPECTED_COUNTS or name == "measure":
            raise ValueError("Saved QASM contains an opaque operation")
        quantum = [int(value) for value in re.findall(r"q\[(\d+)\]", wires)]
        if any(not 0 <= wire < 9 for wire in quantum) or len(set(quantum)) != len(quantum):
            raise ValueError("Saved QASM has invalid quantum indices")
        expected_arity = 9 if name == "barrier" else 2 if name == "cx" else 1
        if len(quantum) != expected_arity:
            raise ValueError("Saved primitive arity changed")
        parameters = [] if arguments is None else [float(value) for value in arguments.split(",")]
        if len(parameters) != (1 if name == "ry" else 0) or any(not math.isfinite(p) for p in parameters):
            raise ValueError("Saved primitive parameters changed")
        operations.append([name, quantum, [], parameters])
    if len(operations) != 396 or dict(Counter(op[0] for op in operations)) != EXPECTED_COUNTS:
        raise ValueError("Saved complete primitive ledger changed")
    return operations


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + " must be a finite scalar")
    return float(value)


def _close(value, target, label: str) -> float:
    residual = abs(_finite(value, label) - target)
    if residual > ATOL:
        raise ValueError(label + " exceeds the saved-reference tolerance")
    return residual


def _check_record(record: dict, buffers: dict[str, bytes]) -> tuple[list[str], dict]:
    if (record.get("schema") != "openquantum.dqi.public_validated_component.v1"
            or record.get("status") != "validated_coherent_component"
            or record.get("coherent_component_complete") is not True):
        raise ValueError("The saved record does not identify a completed coherent component")
    if record.get("focused_tests") != {"tests": 63, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("Saved focused-test totals changed")
    provenance = record["provenance"]
    if (provenance["source_commit"] != "76fadb0b308ea4ea27a21382ca328d349314b1c4"
            or provenance["source_sha256"] != "370088484830c23140f14181a3892194fa936e4477327536946423854fcc1f99"
            or provenance["validation_driver_sha256"] != "c9cbe1dc1fe6d6b842f4159bf81904c8f1321b48167cd268bdc8142f4c8ece41"
            or provenance["producer_record_sha256"] != "6357673185c90876b371b1b2958e9a08d25588037377aa7984de70ad105da410"
            or provenance["focused_junit_sha256"] != "0755f124f3f577b569ffcba7b271497fb9bf412a5cd76e875857b2b964cc8ff6"):
        raise ValueError("Saved coherent-validation event identity changed")
    numerical = record["numerical_contract"]
    if (numerical["absolute_elementwise_tolerance"] != ATOL or numerical["relative_tolerance"] != 0
            or numerical["spectral_norm_certificate"] is not False
            or numerical["formal_proof_or_interval_certificate"] is not False):
        raise ValueError("Saved numerical validation contract changed")
    if any(record.get(name) is not False for name in (
            "sampling_performed", "optimizer_or_fit_performed", "rotation_synthesis_performed",
            "native_or_factory_calls", "gpu_or_remote_work")):
        raise ValueError("Saved component scope changed")
    fixture = record["fixture"]
    if (fixture["n"], fixture["m"], fixture["ell"], fixture["matrix"], fixture["signs"]) != (
            3, 4, 1, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], [0, 0, 0, 1]):
        raise ValueError("Saved default input changed")
    if (record["provenance"]["source_sha256"] != _sha256(buffers["optimization_dqi.py"])
            or record["provenance"]["environment_lock_sha256"] != _sha256(buffers["requirements-lock.txt"])):
        raise ValueError("Saved source/environment identity changed")
    circuit = record["circuit"]
    if (circuit["logical_qubits"] != 9 or circuit["classical_bits"] != 3
            or circuit["source_operations"] != 396 or circuit["primitive_counts"] != EXPECTED_COUNTS
            or circuit["stages"] != EXPECTED_STAGES or circuit["readout_map"] != [[4, 0], [5, 1], [6, 2]]
            or circuit["unsynthesized_ry"] != 7 or circuit["explicit_t_tdg"] != 154
            or circuit["compiled_clifford_t_demand_known"] is not False):
        raise ValueError("Saved circuit/resource contract changed")
    operations = qasm_operations(buffers["dqi.qasm"].decode("utf-8"))
    encoded = (json.dumps(operations, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if _sha256(encoded) != circuit["primitive_operations_sha256"]:
        raise ValueError("Saved QASM chronology or floating parameters changed")
    if operations[:9] != [["reset", [q], [], []] for q in range(9)]:
        raise ValueError("Saved initialization order changed")
    for stage in EXPECTED_STAGES:
        if operations[stage["stop"]] != ["barrier", list(range(9)), [], []]:
            raise ValueError("Saved stage barrier changed")
    if operations[392:395] != [["measure", [4 + i], [i], []] for i in range(3)]:
        raise ValueError("Saved readout chronology changed")
    checks = record["checks"]
    decoder = checks["full_decoder"]
    if (decoder["basis_inputs_checked"], decoder["syndromes"], decoder["clause_values"], decoder["scratch_values"]) != (512, 8, 16, 4):
        raise ValueError("Saved decoder coverage changed")
    if decoder["phase_alignment_or_fitting"] is not False or checks["final_state_and_objective"]["phase_alignment_or_fitting"] is not False:
        raise ValueError("Phase-retaining validation scope changed")
    if (checks["qasm"]["primitive_operations_checked"] != 396
            or checks["qasm"]["exact_numeric_parameter_roundtrip"] is not True):
        raise ValueError("Saved QASM check changed")
    errors = []
    for check in checks.values():
        for name, value in check.items():
            if (name.startswith("maximum_") and name.endswith("_error")) or name == "norm_error":
                scalar = _finite(value, name)
                if scalar < 0 or scalar > ATOL:
                    raise ValueError("A saved numerical check exceeds its tolerance")
                errors.append(scalar)
    for name in ("decoded_cleanup_state", "final_state_and_objective"):
        _close(checks[name]["clause_and_workspace_zero_probability"], 1, "saved cleanup mass")
    _close(checks["preparation"]["workspace_zero_probability"], 1, "saved preparation workspace mass")
    if checks["final_state_and_objective"]["statevector_amplitudes"] != 512:
        raise ValueError("Saved statevector coverage changed")
    outcomes = record["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != 8:
        raise ValueError("The saved distribution must contain eight outcomes")
    table = ["| x2x1x0 | f(x) | analytic probability | checked CPU probability |", "|---|---:|---:|---:|"]
    probability_sum, expectation, optimum_probability = 0.0, 0.0, 0.0
    amplitude_error, probability_error = 0.0, 0.0
    for x, outcome in enumerate(outcomes):
        bits = tuple((x >> j) & 1 for j in range(3))
        objective = sum(bit == 0 for bit in bits) + ((bits[0] ^ bits[1] ^ bits[2]) == 1)
        numerator = 2 * objective - 3
        probability = Fraction(numerator * numerator, 40)
        if (outcome["integer"], outcome["bits"], outcome["objective"], outcome["analytic_probability"]) != (
                x, format(x, "03b"), objective, str(probability)):
            raise ValueError("Saved assignment/objective ordering changed")
        amplitude_error = max(amplitude_error, _close(outcome["amplitude_real"], numerator / math.sqrt(40), "saved signed amplitude"),
                              _close(outcome["amplitude_imag"], 0, "saved imaginary amplitude"))
        probability_error = max(probability_error, _close(outcome["probability"], float(probability), "saved probability"))
        p = float(outcome["probability"])
        probability_sum += p
        expectation += p * objective
        if objective == 3:
            optimum_probability += p
        table.append(f"| {outcome['bits']} | {objective} | {probability} | {p:.12f} |")
    _close(probability_sum, 1, "saved total probability")
    _close(expectation, 14 / 5, "saved expected objective")
    _close(optimum_probability, 9 / 10, "saved optimum probability")
    quality = record["quality"]
    if (quality["maximum_objective"] != 3 or quality["analytic_expected_objective"] != "14/5"
            or quality["analytic_maximum_draw_probability"] != "9/10"):
        raise ValueError("Saved quality contract changed")
    _close(quality["expected_objective"], 14 / 5, "reported expected objective")
    _close(quality["maximum_draw_probability"], 9 / 10, "reported optimum probability")
    return table, {"saved_tests": record["focused_tests"], "decoder_basis_inputs": 512,
                   "qasm_operations": len(operations), "maximum_recorded_error": max(errors),
                   "saved_amplitude_reference_error": amplitude_error,
                   "saved_probability_reference_error": probability_error}


def verify_saved_bundle(directory: str | Path) -> dict:
    """Verify portable saved evidence without importing a quantum package.

    Notebook source bindings ignore execution counters, outputs and metadata,
    so a normal Jupyter reexecution remains compatible with the same bundle.
    """
    directory = Path(directory).resolve()
    manifest_bytes = _read_file(directory, "bundle.json")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "openquantum.dqi.public_saved_bundle.v1":
        raise ValueError("Unsupported public DQI bundle schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise ValueError("The portable bundle file set is incomplete or unexpected")
    buffers, notebook_hash = {}, None
    total = len(manifest_bytes)
    for name, binding in files.items():
        data = _read_file(directory, name)
        total += len(data)
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("Saved DQI bundle exceeds one MiB")
        if name == "optimization_dqi.ipynb" and binding.get("kind") != "notebook_sources":
            raise ValueError("Reusable notebook must bind source cells rather than execution outputs")
        if binding.get("kind") == "file":
            if (type(binding.get("bytes")) is not int or binding["bytes"] != len(data)
                    or binding.get("sha256") != _sha256(data)):
                raise ValueError("Saved file hash or length changed: " + name)
        elif binding.get("kind") == "notebook_sources" and name == "optimization_dqi.ipynb":
            notebook_hash = notebook_source_sha256(data)
            if binding.get("source_sha256") != notebook_hash:
                raise ValueError("Reusable notebook source changed")
        else:
            raise ValueError("Unsupported saved file binding")
        buffers[name] = data
    if notebook_hash is None:
        raise ValueError("The reusable notebook source binding is missing")
    record = json.loads(buffers["dqi-validated.json"])
    table, details = _check_record(record, buffers)
    svg_text = buffers["dqi-circuit.svg"].decode("utf-8")
    if not svg_text.lstrip().startswith("<svg ") or "</svg>" not in svg_text:
        raise ValueError("Expected the saved original SVG figure")
    return {"record": record, "verification": {"status": "saved_component_verified",
            "checked_files": len(files), "bundle_bytes": total, "bundle_manifest_sha256": _sha256(manifest_bytes),
            "notebook_source_sha256": notebook_hash, "new_quantum_execution": False, **details},
            "table_lines": table, "svg_text": svg_text}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the portable bundle.json")
    args = parser.parse_args()
    result = verify_saved_bundle(args.directory)
    print(json.dumps(result["verification"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
