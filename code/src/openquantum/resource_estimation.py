"""Conditional surface-code resource models for measured, bound circuits.

This module does not synthesize or certify Clifford+T circuits. QDK 1.34.0's
mixed-fallback synthesis formula is a resource model. Its approximation scalar
is kept separate from the modelled probability of physical faults, from
algorithmic bias, and from sampling confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import math
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qasm3, transpile


PINNED_QDK_VERSION = "1.34.0"
AUDITED_SOURCE_COMMIT = "81de0d2e5347e1e9f10e05066320fffdb3b7058c"
ROTATION_BASIS = ("h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx")
SOURCES = (
    f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace/transforms/psspc.rs",
    f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace.rs",
    f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/result.rs",
    "https://learn.microsoft.com/en-us/azure/quantum/qre-estimation-results",
    "https://arxiv.org/abs/2203.10064",
)


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class CircuitResourceBudget:
    """Separate approximation-model allowance and whole-run physical faults.

    ``executions`` counts identical, serial circuit executions. The physical
    failure allowance is divided among these executions using a union bound.
    The approximation allowance applies to each prepared circuit output; it
    is not an independent random fault or an optimizer convergence guarantee.
    """

    synthesis_model_error_per_execution: float
    physical_failure_probability: float
    executions: int = 1

    def __post_init__(self) -> None:
        _finite_nonnegative("synthesis_model_error_per_execution", self.synthesis_model_error_per_execution)
        if not 0 < self.physical_failure_probability < 1:
            raise ValueError("physical_failure_probability must lie in (0, 1)")
        if isinstance(self.executions, bool) or not isinstance(self.executions, int) or self.executions < 1:
            raise ValueError("executions must be a positive integer")
        if self.synthesis_model_error_per_execution + self.physical_failure_per_execution >= 1:
            raise ValueError("Per-execution model error allowance must be below one")

    @property
    def physical_failure_per_execution(self) -> float:
        return self.physical_failure_probability / self.executions


@dataclass(frozen=True)
class BoundedObservableBudget:
    """Precision for one mean of independent, identically distributed outcomes.

    Each shot produces one scalar in [outcome_min, outcome_max]. This covers a
    measured MaxCut score or a single Pauli expectation. A noncommuting energy
    sum, adaptive optimizer, or multiple parameter settings needs a separate
    measurement/evaluation allocation; this class does not infer one.

    The synthesis-to-observable conversion is conditional on the QDK model
    scalar bounding output-distribution total variation. No concrete synthesis
    sequence or approximation certificate is provided by this module.
    """

    absolute_error_target: float
    algorithm_bias_bound: float
    synthesis_bias_allowance: float
    sampling_error_allowance: float
    sampling_failure_probability: float
    physical_failure_probability: float
    outcome_min: float = -1.0
    outcome_max: float = 1.0

    def __post_init__(self) -> None:
        for name in ("absolute_error_target", "algorithm_bias_bound", "synthesis_bias_allowance", "sampling_error_allowance"):
            _finite_nonnegative(name, getattr(self, name))
        if self.absolute_error_target <= 0 or self.sampling_error_allowance <= 0:
            raise ValueError("Absolute target and sampling allowance must be positive")
        if not math.isfinite(self.outcome_min) or not math.isfinite(self.outcome_max) or self.outcome_max <= self.outcome_min:
            raise ValueError("Outcome interval must be finite and have positive width")
        if self.total_bias_and_sampling_allowance > self.absolute_error_target and not math.isclose(
            self.total_bias_and_sampling_allowance, self.absolute_error_target, rel_tol=1e-12
        ):
            raise ValueError("Bias and sampling allowances exceed the absolute error target")
        for name in ("sampling_failure_probability", "physical_failure_probability"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in (0, 1)")
        if self.sampling_failure_probability + self.physical_failure_probability >= 1:
            raise ValueError("Combined sampling and physical confidence allowance must be below one")

    @property
    def outcome_width(self) -> float:
        return self.outcome_max - self.outcome_min

    @property
    def total_bias_and_sampling_allowance(self) -> float:
        return self.algorithm_bias_bound + self.synthesis_bias_allowance + self.sampling_error_allowance

    @property
    def shots_required(self) -> int:
        # Hoeffding: P(|sample mean - mean| >= e) <= 2 exp(-2 N e² / W²).
        return max(1, math.ceil(
            self.outcome_width**2
            * math.log(2 / self.sampling_failure_probability)
            / (2 * self.sampling_error_allowance**2)
        ))

    def resource_budget(self) -> CircuitResourceBudget:
        return CircuitResourceBudget(
            synthesis_model_error_per_execution=self.synthesis_bias_allowance / self.outcome_width,
            physical_failure_probability=self.physical_failure_probability,
            executions=self.shots_required,
        )

    def as_contract(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "kind": "one_bounded_scalar_mean",
            "shots_required": self.shots_required,
            "confidence_status": "conditional_only; independent approximation evidence is required",
            "certified_confidence_lower_bound": None,
            "conditional_confidence_lower_bound": 1 - self.sampling_failure_probability - self.physical_failure_probability,
            "assumptions": [
                "Independent identically distributed bounded shots at one fixed circuit/parameter setting",
                "The caller independently establishes the algorithm_bias_bound",
                "The synthesis model scalar upper-bounds output-distribution total variation; this module does not certify it",
                "The declared physical-fault model upper-bounds faults and the classical measurement interpretation is correct",
            ],
            "does_not_cover": ["optimizer convergence", "noncommuting observable grouping", "multiple parameter settings", "adaptive stopping"],
        }


@dataclass(frozen=True)
class SurfaceCodeProfile:
    name: str = "illustrative-p-1e-4"
    physical_error_rate: float = 1e-4
    gate_time_ns: int = 100
    measurement_time_ns: int = 500

    def __post_init__(self) -> None:
        if not 0 < self.physical_error_rate < 0.01:
            raise ValueError("Physical error must be below the model's 1% surface-code threshold")
        if self.gate_time_ns <= 0 or self.measurement_time_ns <= 0:
            raise ValueError("Operation times must be positive nanosecond values")


def synthesis_model_error(num_ts_per_rotation: int) -> float:
    """QDK 1.34.0 PSSPC mixed-fallback model; not a synthesis certificate."""
    if isinstance(num_ts_per_rotation, bool) or not isinstance(num_ts_per_rotation, int) or num_ts_per_rotation < 5:
        raise ValueError("num_ts_per_rotation must be an integer at least five")
    return 2 ** ((4.86 - num_ts_per_rotation) / 0.53)


def choose_num_ts_per_rotation(rotation_count: int, allowance: float) -> int:
    """Choose a fixed synthesis cost whose summed model scalar meets allowance."""
    if isinstance(rotation_count, bool) or not isinstance(rotation_count, int) or rotation_count < 0:
        raise ValueError("rotation_count must be a nonnegative integer")
    _finite_nonnegative("allowance", allowance)
    if rotation_count == 0:
        return 5  # Irrelevant to cost; no rotation consumes synthesized T states.
    if allowance == 0:
        raise ValueError("Arbitrary rotations require a positive synthesis-model allowance")
    # The cited mean linear fit is for epsilon < 1e-4. Twelve T states is
    # the smallest integer whose inverted model lies in that fit domain.
    count = max(12, math.ceil(4.86 + 0.53 * (math.log2(rotation_count) - math.log2(allowance))))
    while rotation_count * synthesis_model_error(count) > allowance:
        count += 1
    return count


def prepare_measured_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Lower a bound static circuit to explicit rotations and Clifford gates.

    Reject mid-circuit measurements, resets, dynamic control, and unbound
    parameters because the current QDK import can otherwise choose a branch or
    omit work. Input preparation and final basis changes must already be part
    of the supplied circuit. The input circuit is not modified.
    """
    if circuit.num_parameters:
        raise ValueError("Bind all circuit parameters before resource estimation")
    measured = False
    for item in circuit.data:
        operation = item.operation
        if operation.name in {"reset", "initialize", "if_else", "while_loop", "for_loop", "switch_case", "store"}:
            raise ValueError(f"Unsupported dynamic or reset operation: {operation.name}")
        if getattr(operation, "condition", None) is not None:
            raise ValueError("Classically conditioned operations need a branch-aware estimator")
        if operation.name == "measure":
            measured = True
        elif operation.name != "barrier" and measured:
            raise ValueError("Only final measurements are supported")
        for parameter in operation.params:
            if not math.isfinite(float(parameter)):
                raise ValueError("Circuit rotation parameters must be finite")
    if not measured:
        raise ValueError("Supply explicit final output measurements")
    compiled = transpile(
        circuit, basis_gates=list(ROTATION_BASIS), optimization_level=1, seed_transpiler=7
    )
    if any(item.operation.name not in {*ROTATION_BASIS, "measure", "barrier"} for item in compiled.data):
        raise ValueError("Circuit lowering left an unsupported opaque operation")
    return compiled


def estimate_circuit_resources(
    circuit: QuantumCircuit,
    *,
    budget: CircuitResourceBudget,
    profile: SurfaceCodeProfile = SurfaceCodeProfile(),
    cache_dir: Path,
    observable_budget: BoundedObservableBudget | None = None,
    use_graph: bool = False,
) -> dict[str, Any]:
    """Run a conditional, explicit-synthesis surface-code resource model.

    Returns computed Pareto points; no output files or site baseline are changed.
    ``use_graph=False`` avoids the documented ISA-pruning incompleteness.
    The report separates QDK's approximation base scalar from its physical-fault
    contribution by using the explicitly additive UnionBound composition.
    """
    installed_version = importlib.metadata.version("qdk")
    if installed_version != PINNED_QDK_VERSION:
        raise RuntimeError(f"Audited QDK {PINNED_QDK_VERSION} required; found {installed_version}")
    if observable_budget is not None and budget != observable_budget.resource_budget():
        raise ValueError("Circuit budget must match the supplied bounded-observable allocation")
    compiled = prepare_measured_circuit(circuit)
    qasm = qasm3.dumps(compiled)

    from qdk.qre import ErrorComposition, LatticeSurgery, PSSPC, estimate, instruction_name, property_name
    from qdk.qre.application import OpenQASMApplication
    from qdk.qre.instruction_ids import LATTICE_SURGERY, RZ
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode

    app = OpenQASMApplication(qasm)
    raw_trace = next(app.enumerate_traces())
    counts = {instruction_name(key): count for key, count in raw_trace.gate_counts.items()}
    if set(counts) - {"RZ", "T", "CCX", "MEAS_Z"}:
        raise ValueError(f"Unaudited canonical instructions: {sorted(set(counts) - {'RZ', 'T', 'CCX', 'MEAS_Z'})}")
    if raw_trace.base_error != 0:
        raise ValueError("Pre-existing trace base error requires a separate provenance allocation")
    rotation_count = raw_trace.gate_counts.get(RZ, 0)
    num_ts = choose_num_ts_per_rotation(rotation_count, budget.synthesis_model_error_per_execution)
    synthesized_trace = PSSPC(num_ts_per_rotation=num_ts).transform(raw_trace)
    if synthesized_trace is None:
        raise RuntimeError("PSSPC could not transform the measured circuit trace")
    synthesis_base = float(synthesized_trace.base_error)
    expected_base = rotation_count * synthesis_model_error(num_ts)
    if not math.isclose(synthesis_base, expected_base, rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("Pinned QDK synthesis semantics differ from the audited formula")
    maximum_qdk_error_score = synthesis_base + budget.physical_failure_per_execution
    estimates = estimate(
        app,
        GateBased(
            error_rate=profile.physical_error_rate,
            gate_time=profile.gate_time_ns,
            measurement_time=profile.measurement_time_ns,
        ),
        isa_query=SurfaceCode.q() * RoundBasedFactory.q(cache_dir=Path(cache_dir)),
        trace_query=PSSPC.q(num_ts_per_rotation=num_ts) * LatticeSurgery.q(),
        max_error=maximum_qdk_error_score,
        composition=ErrorComposition.UnionBound,
        use_graph=use_graph,
    )
    if len(estimates) == 0:
        raise RuntimeError("No feasible estimate under the explicit synthesis and physical budgets")
    points = []
    for entry in estimates:
        physical_per_execution = float(entry.error) - synthesis_base
        if physical_per_execution < -1e-12 or physical_per_execution > budget.physical_failure_per_execution + 1e-12:
            raise RuntimeError("Estimated physical contribution violates the explicit additive contract")
        physical_per_execution = max(0.0, physical_per_execution)
        points.append({
            "peak_physical_qubits": int(entry.qubits),
            "data_code_distance": int(entry.source[LATTICE_SURGERY].transform.distance),
            "runtime_per_execution_ns": float(entry.runtime),
            "serial_runtime_all_executions_ns": float(entry.runtime) * budget.executions,
            "qdk_combined_error_score_per_execution": float(entry.error),
            "synthesis_model_error_per_execution": synthesis_base,
            "physical_failure_model_per_execution": physical_per_execution,
            "physical_failure_union_bound_all_executions": physical_per_execution * budget.executions,
            "factories": {
                instruction_name(key): {
                    "copies": factory.copies, "runs": factory.runs,
                    "states": factory.states, "error_rate": factory.error_rate,
                }
                for key, factory in entry.factories.items()
            },
            "model_properties": {property_name(key): value for key, value in entry.properties.items()},
            "instruction_source_graph": str(entry.source),
        })
    return {
        "schema_version": 1,
        "status": "computed_conditional_model",
        "resource_estimate_kind": "conditional_mean_fit_resource_projection",
        "qdk_version": installed_version,
        "audited_source_commit": AUDITED_SOURCE_COMMIT,
        "source_commit_is_package_release_commit": False,
        "qiskit_version": importlib.metadata.version("qiskit"),
        "qasm3": qasm,
        "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "source_gate_counts": dict(circuit.count_ops()),
        "decomposed_gate_counts": dict(compiled.count_ops()),
        "canonical_gate_counts": counts,
        "canonical_application_trace": str(raw_trace),
        "synthesized_application_trace": str(synthesized_trace),
        "synthesis": {
            "model": "QDK PSSPC mixed-fallback Clifford+T approximation",
            "cost_semantics": "mean linear fit for random angles; not worst-case or certified per angle",
            "fit_domain": "per-rotation modeled diamond-distance accuracy epsilon < 1e-4",
            "num_arbitrary_rotations": rotation_count,
            "fixed_num_ts_per_rotation": num_ts,
            "model_error_per_rotation": synthesis_model_error(num_ts) if rotation_count else 0.0,
            "model_error_per_execution": synthesis_base,
            "actual_synthesis_sequence_generated": False,
            "instance_approximation_certificate_verified": False,
        },
        "budget": asdict(budget),
        "observable_contract": observable_budget.as_contract() if observable_budget else None,
        "physical_profile": asdict(profile),
        "error_composition": "UnionBound; approximation base and physical contribution explicitly separated",
        "isa_graph_pruning": use_graph,
        "pareto_completeness": "pruning may miss points" if use_graph else "enumerated ISA combinations; limited to declared models/domains",
        "included": ["all operations in supplied bound circuit", "explicit input preparation and output basis changes if supplied", "final measurements", "serial executions", "PSSPC logical layout", "surface-code data qubits", "round-based magic-state factories"],
        "limitations": [
            "The synthesis formula is a mean linear fit for random angles, not a worst-case bound or certified concrete Clifford+T implementation",
            "The fit models expected fallback-protocol work; no synthesis runtime-tail or budget-overrun confidence is certified",
            "The QDK trace abstracts Clifford work and scheduling into PSSPC rather than preserving an exact gate schedule",
            "Physical reliability is conditional on the declared hardware/QEC/factory models",
            "No algorithm approximation, optimizer convergence, or observable precision is inferred from the circuit",
            "Classical optimizer/preprocessing/compilation/verification costs and external data loading are excluded",
            "A bounded-observable contract covers one fixed scalar mean only when its declared assumptions hold",
        ],
        "sources": list(SOURCES),
        "pareto_points": points,
    }
