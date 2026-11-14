#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce tables and original figures from the saved word-transport record.

This formats authenticated saved data. It performs no optimizer, sampling,
synthesis, Native resource query, or new quantum/classical reference simulation.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def release_module():
    path = Path(__file__).with_name("optimization_word_record_release.py")
    spec = importlib.util.spec_from_file_location("word_report_release", path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    exec(compile(path.read_bytes(), str(path), "exec"), value.__dict__)
    return value


def _csv(rows, columns):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _circuit(operations):
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(4)
    for name, wires, parameters, condition in operations:
        if condition or parameters or name not in {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx"}:
            raise ValueError("The selected word excerpt is not a primitive Clifford+T sequence")
        getattr(circuit, name)(*wires)
    return circuit


def _checked_output(bundle: Path, output: Path) -> Path:
    output = Path(output).resolve()
    for protected in ((ROOT / "assets/generated").resolve(), Path(bundle).resolve()):
        if output == protected or protected in output.parents:
            raise ValueError("Reproduction output must not overwrite immutable scientific records")
    if output.exists() or output.is_symlink():
        raise ValueError("Report output requires a new directory")
    return output


def format_report(bundle: Path, output: Path, verified: dict) -> dict:
    """Format the independently verified fixed record, without repricing it."""
    if verified.get("status") != "verified_saved_word_native_model":
        raise ValueError("Report generation requires full saved-record verification")
    output = _checked_output(bundle, output)
    output.mkdir(parents=True, exist_ok=False)
    candidate = json.loads((bundle / "native-result.json").read_bytes())
    recipe = json.loads((bundle / "source-recipe.json").read_bytes())["record"]
    accounting = json.loads((bundle / "source-accounting.json").read_bytes())["record"]
    reduction = Fraction(verified["model_runtime_reduction_fraction_exact"])
    rows = [dict(configuration="frozen baseline", preparations=82032, physical_qubits=243358,
                 serial_runtime_ns=verified["baseline_serial_runtime_ns"],
                 physical_failure_exact=verified["baseline_physical_failure_exact"],
                 factory_availability_exact=verified["factory_availability_failure_exact"],
                 spacetime_qubit_ns=verified["baseline_spacetime_qubit_ns"]),
            dict(configuration="word data shuttle", preparations=82032, physical_qubits=243358,
                 serial_runtime_ns=verified["serial_runtime_ns"],
                 physical_failure_exact=verified["whole_physical_failure_exact"],
                 factory_availability_exact=verified["factory_availability_failure_exact"],
                 spacetime_qubit_ns=verified["spacetime_qubit_ns"])]
    (output / "comparison.csv").write_text(_csv(rows, list(rows[0])))
    columns = ("class_id", "parent_physical_class_id", "shots", "runtime_per_preparation_ns",
               "serial_runtime_all_preparations_ns", "code_cycles_per_preparation",
               "full_physical_failure_exact", "factory_availability_failure_exact", "peak_physical_qubits")
    (output / "classes.csv").write_text(_csv([{key: row[key] for key in columns} for row in candidate["results"]], columns))
    parent_rows = []
    for parent in accounting["parent_groups"]:
        children = [row for row in candidate["results"] if row["parent_physical_class_id"] == parent["parent_physical_class_id"]]
        parent_rows.append(dict(parent_physical_class_id=parent["parent_physical_class_id"],
            source_classes=len(children), preparations=sum(row["shots"] for row in children),
            serial_runtime_ns=sum(row["serial_runtime_all_preparations_ns"] for row in children),
            physical_failure_exact=str(sum((Fraction(row["full_physical_failure_exact"]) for row in children), Fraction())),
            physical_allocation_exact=parent["physical_failure_allocation_exact"],
            factory_failure_exact=str(sum((Fraction(row["factory_availability_failure_exact"]) for row in children), Fraction())),
            factory_allocation_exact=parent["factory_availability_allocation_exact"]))
    (output / "parents.csv").write_text(_csv(parent_rows, list(parent_rows[0])))
    text = f'''The fixed `qaoa-cycle-4-p2-alternate-seed-7` transcript contains **109 training groups plus one final holdout**, **54 circuit classes**, **49 original cost-allocation parents**, and **82,032 complete preparations**. All **182,646,240 T states** remain charged.

| Fixed serialized model | Physical qubits | Serial model time (s) | Physical-fault score | Factory availability term |
|---|---:|---:|---:|---:|
| Frozen baseline | 243,358 | {verified['baseline_serial_runtime_ns']/1e9:,.6f} | {float(Fraction(verified['baseline_physical_failure_exact'])):.12g} | {float(Fraction(verified['factory_availability_failure_exact'])):.12g} |
| Word data shuttle | 243,358 | {verified['serial_runtime_ns']/1e9:,.6f} | {float(Fraction(verified['whole_physical_failure_exact'])):.12g} | {float(Fraction(verified['factory_availability_failure_exact'])):.12g} |

The matched **conditional serial-model time and spacetime decrease by {float(100*reduction):.6f}%** on this one machine. This is not a new Pareto search, a population optimizer result, hardware performance, or quantum advantage. The full original transcript—including rejected queries—and its selection-aware inference remain fixed.

The actual local resource-model pass returned 54 individually captured results. Public saved verification reconstructs the scientific input/output accounting without a Native call. Private clock, permit and cleanup evidence is represented by audited hashes; its provisional parent/journal records were not rewritten as final receipts.
'''
    (output / "summary.md").write_text(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "openquantum-word-transport"
    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.barh(["Frozen baseline", "Word data shuttle"], [row["serial_runtime_ns"]/1e9/86400 for row in rows], color=["#68798b", "#06766e"])
    axis.invert_yaxis()
    axis.set_xlabel("Serialized conditional model time (days)")
    axis.set_title("One fixed C4 QAOA transcript, same 243,358-qubit machine")
    axis.spines[["top", "right"]].set_visible(False)
    for index, row in enumerate(rows):
        axis.text(row["serial_runtime_ns"]/1e9/86400 + .12, index, f"{row['serial_runtime_ns']/1e9/86400:.3f}", va="center")
    axis.set_xlim(0, 10.8)
    figure.tight_layout()
    figure.savefig(output / "runtime-comparison.svg", metadata={"Date": None})
    plt.close(figure)
    excerpt = None
    for row in recipe["classes"]:
        file = bundle / "candidate/circuits" / (row["class_id"] + ".json.gz")
        item = json.loads(gzip.decompress(file.read_bytes()))
        for segment in item["segments"]:
            if segment["shuttled"] and (excerpt is None or segment["local_operations"] < excerpt[0]):
                excerpt = (segment["local_operations"], row["class_id"], segment,
                           item["operations"][segment["candidate_start"]:segment["candidate_stop"]])
    _, class_id, segment, operations = excerpt
    excerpt_record = dict(class_id=class_id, source_wire=segment["source_wire"], local_operations=segment["local_operations"],
        t_states=segment["magic_count"], added_cx=segment["added_data_transport_cx"], operations=operations,
        candidate_segment_sha256=segment["candidate_segment_sha256"],
        scope="smallest selected exact word excerpt; full four-wire trial, preparation and readout are in the class exports")
    (output / "word-excerpt.json").write_text(json.dumps(excerpt_record, sort_keys=True, indent=2) + "\n")
    from qiskit import qasm3
    circuit = _circuit(operations)
    (output / "word-excerpt.qasm").write_text(qasm3.dumps(circuit))
    figure = circuit.draw(output="mpl", fold=45, idle_wires=False,
                          style={"backgroundcolor": "#ffffff"})
    figure.suptitle(f"Exact selected word: D{segment['source_wire']} to D0 and back ({segment['magic_count']} T)")
    figure.savefig(output / "word-excerpt.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    # Match the book's deterministic canonical whitespace normalization.
    for name in ("runtime-comparison.svg", "word-excerpt.svg"):
        file = output / name
        file.write_text("\n".join(line.rstrip() for line in file.read_text().splitlines()) + "\n")
    return dict(files=8, classes=54, parents=49, new_native_queries=0, new_optimizer_calls=0,
                new_synthesis_calls=0, new_reference_simulations=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "assets/generated/optimization/word-transport")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/optimization-word-report")
    args = parser.parse_args()
    try:
        output = _checked_output(args.records_dir, args.output_dir)
    except ValueError as error:
        raise SystemExit(str(error))
    release = release_module()
    verified = release.verify_public_word_transport(args.records_dir)
    print(json.dumps(format_report(args.records_dir, output, verified), sort_keys=True))


if __name__ == "__main__":
    main()
