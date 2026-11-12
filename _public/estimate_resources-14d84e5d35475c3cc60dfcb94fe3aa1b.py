#!/usr/bin/env python3
"""Estimate the generated Grover circuit with the local Microsoft QDK.

The estimate is conditional on an illustrative rotated surface-code hardware
model. It covers the actual generated oracle, diffusion, measurements, and
serial repetitions, while recording the classical work excluded from timing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def repetition_budget(
    n_qubits: int,
    iterations: int,
    total_failure: float = 0.01,
    physical_fraction: float = 0.5,
) -> dict[str, float | int]:
    """Split sampling and physical fault probabilities without double counting."""
    if n_qubits < 1 or iterations < 0:
        raise ValueError("Require positive qubit count and nonnegative iterations")
    if not 0 < total_failure < 1 or not 0 < physical_fraction < 1:
        raise ValueError("Failure target and physical fraction must lie in (0, 1)")
    theta = math.asin(math.sqrt(1 / (2**n_qubits)))
    success = math.sin((2 * iterations + 1) * theta) ** 2
    sampling_allowance = total_failure * (1 - physical_fraction)
    physical_allowance = total_failure * physical_fraction
    if success <= 0:
        raise ValueError("The configured Grover circuit has zero ideal success")
    trials = (
        1
        if success >= 1 - 1e-15
        else max(1, math.ceil(math.log(sampling_allowance) / math.log1p(-success)))
    )
    if trials > 10_000:
        raise ValueError("Too many repetitions for this demonstration")
    sampling_failure = (1 - success) ** trials
    return {
        "total_failure_target": total_failure,
        "sampling_failure_allowance": sampling_allowance,
        "physical_failure_allowance": physical_allowance,
        "ideal_success_probability_per_trial": success,
        "serial_trials": trials,
        "ideal_sampling_failure_after_trials": sampling_failure,
        "max_physical_error_per_trial": physical_allowance / trials,
        "configured_full_failure_union_bound": sampling_failure + physical_allowance,
    }


def validate_qasm(source: str, n_qubits: int) -> None:
    """Fail closed if the exported demo still contains custom opaque gates."""
    if not re.search(r"OPENQASM\s+3(?:\.0)?\s*;", source):
        raise ValueError("Resource estimation expects OpenQASM 3")
    if re.search(r"\bgate\s+", source):
        raise ValueError("Decompose custom gates before estimating resources")
    registers = re.findall(r"qubit\s*\[(\d+)\]\s*\w+\s*;", source)
    if sum(map(int, registers)) != n_qubits:
        raise ValueError("QASM qubit count differs from the specified problem")
    if "measure" not in source:
        raise ValueError("Include output measurements in the estimated circuit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qasm", type=Path, default=ROOT / "assets/generated/grover/circuit.qasm"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "assets/generated/resources"
    )
    parser.add_argument("--logical-metadata", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/resource-estimator"
    )
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--marked", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--total-failure", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 <= args.marked < 2**args.n_qubits:
        raise ValueError("Marked state must fit in the qubit register")
    source = args.qasm.read_text(encoding="utf-8")
    validate_qasm(source, args.n_qubits)
    metadata_path = args.logical_metadata or args.qasm.with_name("logical-resources.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.encode()).hexdigest()
    expected_metadata = {
        "input_qubits": args.n_qubits,
        "marked_integer": args.marked,
        "iterations": args.iterations,
        "qasm_sha256": digest,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Generated circuit metadata differs at {key}: regenerate matching inputs")
    budget = repetition_budget(args.n_qubits, args.iterations, args.total_failure)

    # These are the current, documented local API names, not Azure job APIs.
    from qdk.qre import estimate, instruction_name, property_name
    from qdk.qre.application import OpenQASMApplication
    from qdk.qre.instruction_ids import LATTICE_SURGERY
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode

    app = OpenQASMApplication(source)
    application_trace = next(app.enumerate_traces())
    required_instructions = list(application_trace.required_isa.as_frame().index)
    if any(name in required_instructions for name in ("RZ", "RX", "RY", "CRZ", "RZZ")):
        raise ValueError(
            "This initial baseline requires an exact Clifford+T trace; "
            "arbitrary rotations need an explicit approximation-error contract"
        )
    isa_query = SurfaceCode.q() * RoundBasedFactory.q(cache_dir=args.cache_dir)
    profiles = []
    flat_rows = []
    for physical_error in (1e-3, 1e-4):
        architecture = GateBased(
            error_rate=physical_error, gate_time=100, measurement_time=500
        )
        estimates = estimate(
            app,
            architecture,
            isa_query=isa_query,
            max_error=budget["max_physical_error_per_trial"],
        )
        if len(estimates) == 0:
            raise RuntimeError(f"No feasible surface-code estimate at p={physical_error}")
        points = []
        for entry in estimates:
            physical_error_bound = int(budget["serial_trials"]) * float(entry.error)
            factories = {
                instruction_name(key): {
                    "copies": factory.copies,
                    "runs": factory.runs,
                    "states": factory.states,
                    "error_rate": factory.error_rate,
                }
                for key, factory in entry.factories.items()
            }
            point = {
                "physical_qubits": int(entry.qubits),
                "data_code_distance": int(entry.source[LATTICE_SURGERY].transform.distance),
                # The QDK EstimationTableEntry runtime is in nanoseconds.
                "runtime_per_trial_ns": float(entry.runtime),
                "runtime_all_trials_ns": float(entry.runtime) * int(budget["serial_trials"]),
                "physical_error_estimate_per_trial": float(entry.error),
                "full_physical_failure_union_bound": physical_error_bound,
                "full_failure_union_bound": float(budget["ideal_sampling_failure_after_trials"])
                + physical_error_bound,
                "factories": factories,
                "properties": {
                    property_name(key): value for key, value in entry.properties.items()
                },
                "instruction_source_graph": str(entry.source),
            }
            if point["full_failure_union_bound"] > args.total_failure + 1e-12:
                raise RuntimeError("Estimator result exceeded the overall failure budget")
            points.append(point)
            flat_rows.append(
                {
                    "physical_error_rate": physical_error,
                    "gate_time_ns": 100,
                    "measurement_time_ns": 500,
                    "serial_trials": budget["serial_trials"],
                    **{key: value for key, value in point.items() if key not in {
                        "factories", "properties", "instruction_source_graph"
                    }},
                }
            )
        profiles.append(
            {
                "id": f"surface-code-p-{physical_error:g}",
                "physical_error_rate": physical_error,
                "gate_time_ns": 100,
                "measurement_time_ns": 500,
                "qec_model": "SurfaceCode",
                "factory_model": "RoundBasedFactory",
                "pareto_points": points,
            }
        )

    output = {
        "schema_version": 1,
        "status": "computed",
        "algorithm": "grover-equality-predicate",
        "instance": {
            "n_qubits": args.n_qubits,
            "marked": args.marked,
            "marked_bitstring": format(args.marked, f"0{args.n_qubits}b"),
            "iterations": args.iterations,
            "number_of_marked_states": 1,
        },
        "qasm_sha256": digest,
        "canonical_application_trace": str(application_trace),
        "canonical_required_instructions": required_instructions,
        "arbitrary_rotation_synthesis_required": False,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("qdk", "qiskit", "pandas")
        },
        "error_budget": budget,
        "included": [
            "initial state preparation", "explicit equality oracle", "diffusion",
            "final measurements", "serial repetitions", "surface-code data qubits",
            "magic-state factories and their production time",
        ],
        "excluded": [
            "classical circuit generation and compilation time",
            "classical predicate verification time",
            "external database loading and arbitrary application predicates",
            "hardware-specific control, routing, and decoder engineering beyond the estimator model",
        ],
        "assumptions": [
            "Illustrative gate-based physical profiles; not measured hardware specifications",
            "The oracle tests equality to a known bitstring; this demonstrates implementation, not search advantage",
            "Ideal sampling formula and ideal classical verification are conditional assumptions",
            "Total failure uses a union bound over sampling and physical faults",
            "All trials run serially; physical qubit count is the peak, not multiplied by trials",
            "The QDK application trace abstracts Clifford work and scheduling into the estimator model",
        ],
        "sources": [
            "https://learn.microsoft.com/en-us/azure/quantum/install-run-resource-estimator",
            "https://learn.microsoft.com/en-us/azure/quantum/qre-supported-applications",
            "https://learn.microsoft.com/en-us/azure/quantum/qre-build-error-correction-models",
            "https://learn.microsoft.com/en-us/azure/quantum/qre-estimation-results",
            "https://github.com/microsoft/qdk",
        ],
        "profiles": profiles,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "grover-surface-code.json").write_text(
        json.dumps(output, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "grover-surface-code.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    table = [
        "| Physical error | Pareto choice | Data code distance | Peak physical qubits | All-trial runtime (ms) | Full failure bound |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        points = profile["pareto_points"]
        for label, point in (
            ("Fewest qubits", min(points, key=lambda point: point["physical_qubits"])),
            ("Shortest runtime", min(points, key=lambda point: point["runtime_all_trials_ns"])),
        ):
            table.append(
                f"| {profile['physical_error_rate']:g} | {label} | {point['data_code_distance']} | {point['physical_qubits']:,} "
                f"| {point['runtime_all_trials_ns'] / 1e6:.4f} | {point['full_failure_union_bound']:.4%} |"
            )
    (args.output_dir / "table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (args.output_dir / "estimate_resources.py").write_text(
        Path(__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for profile, color in zip(profiles, ("#007f86", "#ba5f27")):
        points = sorted(profile["pareto_points"], key=lambda point: point["physical_qubits"])
        ax.plot(
            [point["physical_qubits"] for point in points],
            [point["runtime_all_trials_ns"] / 1e6 for point in points],
            "o-", color=color, label=f"Physical error {profile['physical_error_rate']:g}",
        )
    ax.set_xlabel("Peak physical qubits (data + factories)")
    ax.set_ylabel(f"Runtime for {budget['serial_trials']} serial trials (ms)")
    ax.set_title(f"Grover, {args.n_qubits} qubits · rotated surface-code model")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "grover-surface-code.svg")
    plt.close(fig)
    print(f"Computed {len(flat_rows)} Pareto points; wrote {args.output_dir}")


if __name__ == "__main__":
    main()
