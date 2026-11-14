#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Produce only the bounded new m=3/4 static-versus-feedback QPE records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from qiskit import qasm3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from openquantum.instrument_resources import InstrumentProfile, estimate_instrument, synthesize_instrument
from openquantum.qft_qpe import phase_product_example, qpe_distribution, qpe_ledger
from openquantum.qpe_feedback_resources import (LIMITS, PhaseTask, branch_event_counts,
    add_feedback_exposure, build_sources, decode_median_network, estimate_feedback,
    measured_distribution, median_controller_decisions, synthesize_feedback)

OUTPUT = ROOT/"assets/generated/qpe-feedback-resources"
SOURCES = [ROOT/"src/openquantum/qpe_feedback_resources.py", ROOT/"src/openquantum/qft_qpe.py",
           ROOT/"src/openquantum/instrument_resources.py", Path(__file__),
           ROOT/"scripts/verify_qpe_feedback_saved.py",
           ROOT/"tests/test_qpe_feedback_resources.py", ROOT/"requirements-lock.txt"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def protected_fingerprints():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/"assets/generated").rglob("*"))
            if p.is_file() and OUTPUT not in p.parents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=Path(tempfile.gettempdir())/"qpe-feedback-factory-cache")
    parser.add_argument("--maximum-code-distance", type=int, default=15, choices=(15,))
    args = parser.parse_args()
    if platform.python_version() != "3.12.14":
        parser.error("Use the read-only locked Python3.12.14 runtime")
    if {p: importlib.metadata.version(p) for p in ("qiskit", "qdk")} != {"qiskit": "2.6.2", "qdk": "1.34.0"}:
        parser.error("Use the pinned Qiskit2.6.2/QDK1.34.0 environment")
    output, cache = args.output_dir.resolve(), args.cache_dir.resolve()
    if output == ROOT or output in ROOT.parents or cache == output or output in cache.parents or cache in output.parents:
        parser.error("Keep the private factory cache and result directory disjoint")
    if any(output == p or output in p.parents for p in SOURCES):
        parser.error("Output must not cover an implementation/lock source")
    generated = (ROOT/"assets/generated").resolve()
    if cache == generated or generated in cache.parents:
        parser.error("Private factory caches must stay outside public generated assets")
    if output != OUTPUT.resolve() and (output == generated or generated in output.parents):
        parser.error("Use only the new dedicated generated tree; alternate outputs must be outside assets/generated")
    if output.exists() and output != OUTPUT.resolve() and any(output.iterdir()):
        parser.error("An alternate output must be an empty dedicated directory")
    allowed_root = {"manifest.json", "summary.csv", "summary.md", "logical-size-scaling.json", "README.md",
                    "LICENSE.txt", "references.bib", "primitives.bib"} | {p.name for p in SOURCES}
    allowed_task = {"report.json", "textbook-source.qasm", "feedback-source.qasm", "textbook-clifford-t.qasm",
                    "feedback-clifford-t.qasm", "feedback-all-corrections-accounting-envelope.qasm"}
    allowed = allowed_root | {f"n1-m{m}/{name}" for m in (3, 4) for name in allowed_task}
    if output.exists() and any(p.is_symlink() or (p.is_file() and str(p.relative_to(output)) not in allowed)
                               or (p.is_dir() and str(p.relative_to(output)) not in {"n1-m3", "n1-m4"}) for p in output.rglob("*")):
        parser.error("New QPE output contains unrelated/private members; use a clean dedicated directory")
    baseline_before = protected_fingerprints()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *[str(p.relative_to(ROOT)) for p in SOURCES]], cwd=ROOT, text=True).strip())
    reports, summary = [], []
    profile = InstrumentProfile(physical_error_rate=1e-4, maximum_code_distance=15)
    for m in (3, 4):
        task = PhaseTask(phase_bits=m)
        sources = build_sources(task)
        reference = qpe_distribution(task.phase, m)
        checks, compiled, physical = {}, {}, {}
        began = time.perf_counter()
        static = synthesize_instrument(sources["textbook"], task.operator_error_allowance,
                                      limits=LIMITS, preserve_full_unitary=True)
        static_seconds = time.perf_counter()-began
        began = time.perf_counter()
        envelope, dynamic, corrections = synthesize_feedback(sources["feedback"], task.operator_error_allowance)
        feedback_seconds = time.perf_counter()-began
        compiled = {"textbook": static.circuit, "feedback": dynamic}
        for variant in ("textbook", "feedback"):
            ideal = measured_distribution(sources[variant])
            actual = measured_distribution(compiled[variant])
            if np.max(np.abs(ideal-reference)) > 2e-13:
                raise AssertionError("Original circuit disagrees with independent Fourier law")
            eta = (static if variant == "textbook" else envelope).report["guarded_numerical_operator_error_upper_estimate"]
            tv = float(np.abs(actual-ideal).sum()/2)
            if tv > eta+1e-10:
                raise AssertionError("Actual synthesized instrument exceeds its numerical proposal")
            checks[variant] = {"ideal_probabilities": ideal.tolist(), "clifford_t_probabilities": actual.tolist(),
                "independent_fourier_max_error": float(np.max(np.abs(ideal-reference))),
                "actual_synthesis_total_variation": tv,
                "uniform_synthesis_tv_upper_estimate": eta,
                "branch_pruning": "no nonzero branches pruned; exactly zero floating branches omitted"}
        # A deleted-feedback circuit is a negative control, not another estimate.
        missing = sources["feedback"].copy_empty_like()
        for gate in sources["feedback"].data:
            if gate.operation.name != "if_else":
                missing.append(gate.operation, gate.qubits, gate.clbits)
        negative_error = float(np.max(np.abs(measured_distribution(missing)-reference)))
        if negative_error < .01:
            raise AssertionError("Feedback negative control failed to expose real branch behavior")
        print(f"m={m}: real source/Clifford+T feedback and negative-control checks passed", flush=True)
        static_base = estimate_instrument(static, 15, profile=profile,
            physical_failure_budget=task.physical_failure_budget,
            factory_availability_budget=task.factory_availability_budget, cache_dir=cache,
            include_all_feasible_points=True)
        physical["textbook"] = add_feedback_exposure(static_base, 0,
            postprocessing_decisions=median_controller_decisions(m))
        physical["feedback"] = estimate_feedback(envelope, corrections, task, profile=profile, cache_dir=cache)
        folder = output/f"n1-m{m}"
        folder.mkdir(exist_ok=True)
        artifacts = {"textbook-source.qasm": sources["textbook"], "feedback-source.qasm": sources["feedback"],
                     "textbook-clifford-t.qasm": static.circuit, "feedback-clifford-t.qasm": dynamic,
                     "feedback-all-corrections-accounting-envelope.qasm": envelope.circuit}
        for filename, circuit in artifacts.items():
            (folder/filename).write_text(qasm3.dumps(circuit))
        draws = [int(y) for y in np.random.default_rng(20261113+m).choice(2**m, size=15, p=checks["feedback"]["clifford_t_probabilities"])]
        decode_began = time.perf_counter()
        median = decode_median_network(draws, m)
        decode_seconds = time.perf_counter()-decode_began
        if median != float(np.median(draws))/2**m:
            raise AssertionError("Fixed classical decoder disagrees with independent median")
        report = {"task": task.contract(), "independent_fourier_probabilities": reference.tolist(),
            "checks": checks, "feedback_removed_max_error": negative_error,
            "physical": physical, "synthesis": {"textbook": static.report, "feedback_envelope": envelope.report},
            "synthesis_wall_seconds": {"textbook": static_seconds, "feedback": feedback_seconds},
            "classical_median_demo": {"seed": 20261113+m, "draws": draws, "decoded_scalar_median": median,
                "independent_median": float(np.median(draws))/2**m, "actual_python_decode_wall_seconds": decode_seconds,
                "source": "seeded draws from the enumerated actual Clifford+T ideal instrument; not hardware observations"},
            "feedback_branch_ledgers": branch_event_counts(envelope, corrections, m),
            "common_schedule": "fixed all-correction slots, one factory/configuration/retry cap per selected Pareto point; skip gates and retain whole-device idle padding on false predicates",
            "coherent_phase_scope": "full source controls retain supplied global phase; conditional-body overall phases are irrelevant after measured classical branching; exports are complete instruments, not coherent oracles",
            "expected_cost": "not reported; every padded allowed transcript has the reserved worst-case wall time",
            "artifact_sha256": {name: digest(folder/name) for name in artifacts}}
        write_json(folder/"report.json", report)
        reports.append(report)
        for variant, resource in physical.items():
            selected = min(resource["pareto_points"], key=lambda p: (p["peak_physical_qubits"], p["runtime_per_complete_attempt_ns"]))
            summary.append({"target_qubits": 1, "phase_bits": m, "variant": variant,
                "phase_error_target": 2**-m, "complete_runs": 15,
                "data_code_distance": selected["data_code_distance"],
                "peak_physical_qubits": selected["peak_physical_qubits"],
                "capped_runtime_all_runs_seconds": selected["worst_case_runtime_all_attempts_ns"]/1e9,
                "full_physical_failure_model": selected["full_physical_failure_model"],
                "factory_availability_failure_model": selected["factory_supply"]["availability_failure_union_bound"],
                "actual_t_and_tdg_per_run_upper_bound": sum((static if variant == "textbook" else envelope).circuit.count_ops().get(g, 0) for g in ("t", "tdg"))})
        print(f"m={m}: both new conditional physical records computed", flush=True)
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    # Logical n scaling uses the scalable access generator; it is not simulated
    # or synthesized at these widths and is not physical evidence.
    logical = []
    for n in (1, 2, 4, 8):
        unitary, preparation = phase_product_example(n, .3)
        for variant in (False, True):
            ledger = qpe_ledger(3, unitary, preparation, iterative=variant)
            logical.append({"n": n, "m": 3, "variant": ledger["variant"],
                "logical_qubits": ledger["logical_qubits"], "controlled_U_calls": 7,
                "validation_status": "generator/source ledger only; not simulated/synthesized/physically estimated here",
                "source_ledger": ledger})
    write_json(output/"logical-size-scaling.json", logical)
    for source in SOURCES:
        (output/source.name).write_bytes(source.read_bytes())
    for source, name in ((ROOT/"LICENSE", "LICENSE.txt"), (ROOT/"references/resources.bib", "references.bib"),
                         (ROOT/"references/primitives.bib", "primitives.bib")):
        (output/name).write_bytes(source.read_bytes())
    baseline_after = protected_fingerprints()
    if baseline_before != baseline_after:
        raise AssertionError("An existing generated artifact changed")
    manifest = {"status": "two new matched source/synthesis/common-schedule physical comparisons executed",
        "source_commit": revision, "source_dirty": dirty,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in SOURCES},
        "runtime_environment": {"python": platform.python_version(), "operating_system": platform.system(),
            "architecture": platform.machine(), "device": "CPU; no hardware/GPU job",
            "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "qdk", "numpy", "scipy")},
            "routing_seed": 7, "native_synthesis_seed": None},
        "compute_caps": {"phase_bits": [3, 4], "target_qubits": 1, "physical_error": 1e-4,
            "odd_code_distances": list(range(3, 16, 2)), "estimator_invocation_count": 4,
            "maximum_factory_distance_queries": 28,
            "limits": static.report["limits"], "factory_retry_ceiling": 16},
        "protected_generated_artifact_count": len(baseline_before),
        "protected_generated_artifacts_unchanged": True,
        "comparison_selection": "minimum peak qubits per matched task/variant; all computed Pareto points retained",
        "formal_confidence_certificate": False, "hardware_execution": False}
    write_json(output/"manifest.json", manifest)
    table = ["| m | Variant | Distance | Peak physical qubits | Capped15-run time (s) |",
             "|---:|---|---:|---:|---:|"]
    table += [f"| {r['phase_bits']} | {r['variant']} | {r['data_code_distance']} | {r['peak_physical_qubits']:,} | {r['capped_runtime_all_runs_seconds']:.6f} |" for r in summary]
    (output/"summary.md").write_text("\n".join(table)+"\n")
    (output/"README.md").write_text("""# Complete feedback-QPE accounting

These are two bounded, original MIT implementations, n=1 and m=3/4,
phi=.3, U=P(2pi phi), X eigenstate preparation, and15 fresh full runs.
The executable feedback QASM contains actual measurements, all round resets
and conditional Clifford+T corrections. The separately named all-corrections
envelope is accounting input and must not be run as feedback QPE.

Both variants use the same serialized surface-code instrument model and
hardware scenario: p=1e-4,100ns gates/init,500ns measurements,1000ns code
cycles,10000ns controller reaction,1e-15 controller fault per decision.
The existing static QDK mean-fit baseline and all earlier artifacts are
preserved. Every full run initializes every wire and repeats actual supplied
controlled U calls; workspace startup, target idling, preparation, readout,
reset, local Clifford frames, neighbor routing, T transport/injection and
capped factory retries are charged. Each predicate, even false, adds one
controller slot and whole-device idle exposure. A common distance/factory/
retry cap covers every allowed transcript. False bodies pad their reserved
slot; expected wall time is not presented as a lower resource estimate.

Scalar medians need phi in[2^-m,1-2^-m]. The ideal Fourier/majority argument,
actual guarded numerical Ross-Selinger synthesis, modeled physical faults
and factory availability are distinct. No floating check certifies hardware
confidence, a decoder, a lattice layout or a formal synthesis interval.
The exact sequences, hashes, source controls, all branch ledgers, failure
components and complete Pareto points are retained in the task reports.

Offline synthesis time is separately measured. The15-sample scalar median
is decoded by a fixed105-comparator m-bit network and charged once after all
runs:15*m bit-decoding slots,105*(m+3) comparison/write slots and1 selection
slot. Each uses the controller reaction/fault scenario; all quantum patches
remain charged during that time. Compilation and presentation-only final
reporting time are excluded from quantum wall time. Geometry,
tracked Pauli bases, factory/data handoff and encoder/readout/controller
reliability retain the explicit conditional assumptions of instrument_resources.
Classical record/logic reliability is part of the per-decision controller
scenario; independent idle classical-memory calibration and classical
controller transistor area are not supplied or included in physical qubits.
Conditional-body overall phases are discarded only across measured classical
branches; controlled source global phase is preserved. QASM exports are
complete measured instruments, not reusable coherent oracle specifications.

The logical-size file generates active phase-product access at n=1,2,4,8;
those extra widths were not simulated/synthesized/physically estimated.
Scalable generation retains the independent qft_qpe.py caps of64 wires,
16 phase bits,200000 source operations and1000000 decomposed operations.
The new physical/simulation adapter is limited to two feedback wires,
at most4 record bits,100000 total validation operations and256 branches.
Only previously measured one-bit, local one-arm corrections are supported.

Reproduce in the repository with the locked Python3.12.14 environment:
python scripts/generate_qpe_feedback_resources.py
pytest tests/test_qpe_feedback_resources.py
The producer's four new native queries search odd distances3..15. Keep the
private factory cache outside the public output. No existing producer is run.

Qiskit2.6.2 is Apache-2.0; rsgridsynth0.2.2 and QDK1.34.0 are MIT.
Exact upstream identities and the installed-wheel limitation appear in
synthesis/physical reports. references.bib and primitives.bib credit the
semiclassical QFT, QPE, Ross-Selinger and surface-code sources. All prose and
circuits here are original; LICENSE.txt covers our implementation.
""")
    if output == OUTPUT.resolve():
        import nbformat
        from nbclient import NotebookClient
        path = ROOT/"notebooks/qpe_feedback_resources.ipynb"
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=60, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(notebook, path)
        print("Executed record-reading comparison notebook without native reruns", flush=True)
    notebook_path = ROOT/"notebooks/qpe_feedback_resources.ipynb"
    manifest["notebook"] = {"path": str(notebook_path.relative_to(ROOT)), "sha256": digest(notebook_path)}
    manifest["artifact_sha256"] = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                                  if p.is_file() and p.name != "manifest.json"}
    write_json(output/"manifest.json", manifest)
    from verify_qpe_feedback_saved import verify_saved
    if output == OUTPUT.resolve():
        print(verify_saved(ROOT)["status"], flush=True)
    print("Completed bounded new QPE feedback comparison", flush=True)


if __name__ == "__main__":
    main()
