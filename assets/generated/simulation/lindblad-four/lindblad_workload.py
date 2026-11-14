# SPDX-License-Identifier: MIT
"""Bounded accuracy-preserving interacting-chain workloads.

The step policy is an original exact-arithmetic trace-norm bound for the
literal positive-time global palindrome. It is separate from floating circuit
and Liouvillian checks. Quantum circuits use the existing explicit ZZ/RX and
reset/dilation generator; dense matrices are only classical verification.
No synthesis or physical resource estimate is inferred from these logical
workloads. Original MIT implementation on pinned Qiskit/NumPy/SciPy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace
from scipy.linalg import expm

if __package__:
    from .open_systems import finite_time_dilation
    from .two_spin_lindblad import (append_x_layer, append_zz, build_chain_circuit,
                                   chain_schedule, lower_to_rotation_basis)
else:
    from open_systems import finite_time_dilation
    from two_spin_lindblad import (append_x_layer, append_zz, build_chain_circuit,
                                  chain_schedule, lower_to_rotation_basis)

FAMILY = (2, 3, 4)
MAX_STEPS = 1024
MAX_DENSE_SYSTEM_QUBITS = 4
MAX_FULL_CIRCUIT_STEPS = 4
MAX_CONVERGENCE_POINTS = 5
SPLITTING_ALLOWANCE = Fraction(1, 100)
STATISTICAL_ERROR = Fraction(3, 100)
FAMILY_STATISTICAL_FAILURE = Fraction(1, 200)
CASE_STATISTICAL_FAILURE = FAMILY_STATISTICAL_FAILURE / len(FAMILY)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)


@dataclass(frozen=True)
class ChainTask:
    """Small family, with finite explicit limits separate from the old generator."""
    n_system: int
    time: float = 1.0
    J: float = 0.6
    hx: float = 0.35
    rate0: float = 0.8
    other_rate: float = 0.5

    def __post_init__(self):
        if isinstance(self.n_system, bool) or not isinstance(self.n_system, int) or self.n_system not in FAMILY:
            raise ValueError("Workload family is bounded to 2, 3 or 4 systems")
        for name in ("time", "J", "hx", "rate0", "other_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite and real")
            try:
                value = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError(f"{name} must have a finite float representation") from error
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite and real")
            object.__setattr__(self, name, value)
        if not 0 <= self.time <= 2 or abs(self.J) > 1 or abs(self.hx) > 1:
            raise ValueError("Bounded domain: time in [0,2], |J| and |hx| at most one")
        if not 0 <= self.rate0 <= 2 or not 0 <= self.other_rate <= 2:
            raise ValueError("Local damping rates must lie in [0,2]")

    @property
    def rates(self):
        return (float(self.rate0),)+(float(self.other_rate),)*(self.n_system-1)


def _steps(steps: int):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError("steps must be an integer from 1 to 1024")


def generator_norm_envelope(task: ChainTask) -> Fraction:
    """Hermitian trace-norm map envelope, not a vectorized matrix 1/2-norm.

    Commutators satisfy ||-i[H,.]|| <=2||H||_operator. Each standard-GKSL lowering
    dissipator satisfies ||D_j|| <=2 gamma_j. These estimates are uniform
    over input states and use decimal interpretations of parameter strings.
    """
    n = task.n_system
    return (2*abs(Fraction(str(task.J)))*(n-1)
            + 2*abs(Fraction(str(task.hx)))*n
            + 2*sum((Fraction(str(rate)) for rate in task.rates), Fraction()))


def uniform_split_bound(task: ChainTask, steps: int) -> Fraction:
    """Full density trace-norm bound for ideal exact-angle evolution.

    F(s) is the reduced-system channel palindrome after environment discard,
    not the joint dilation unitary with square-root small-time angles. Each
    component has total weight one. Product Leibniz and component
    contraction give ||F'''(s)|| <= A^3. The exact channel E(s) has the same
    bound. F/E derivatives through degree two coincide at zero; two integral
    remainders give local <= A^3 h^3/3. CPTP telescoping gives total
    <= (A*t)^3/(3*r^2), capped by the universal state-distance ceiling two.
    Preparation/reset faults, angle rounding, synthesis and hardware faults
    are outside this exact-arithmetic evolution statement.
    """
    _steps(steps)
    exposure = generator_norm_envelope(task)*Fraction(str(task.time))
    return min(Fraction(2), exposure**3/(3*steps**2))


def select_steps(task: ChainTask, allowance: Fraction = SPLITTING_ALLOWANCE) -> int:
    """Minimal integer r from the declared bound, with exact rational comparison."""
    allowance = Fraction(allowance)
    if not 0 < allowance < 1:
        raise ValueError("Splitting allowance must lie in (0,1)")
    exposure = generator_norm_envelope(task)*Fraction(str(task.time))
    target = exposure**3/(3*allowance)
    steps = max(1, math.isqrt(target.numerator//target.denominator))
    if steps**2*target.denominator < target.numerator:
        steps += 1
    if steps > MAX_STEPS:
        raise ValueError("Uniform-policy step count exceeds the declared 1024-step domain")
    return steps


def output_contract() -> dict:
    shots = math.ceil(2*(math.log(2)-math.log(float(CASE_STATISTICAL_FAILURE)))/float(STATISTICAL_ERROR)**2)
    exponent = shots*STATISTICAL_ERROR**2/2
    term, lower = Fraction(1), Fraction(1)
    for degree in range(1, 33):
        term *= exponent/degree
        lower += term
    # Positive Taylor terms give an exact lower bound on exp(exponent).
    # This verifies the chosen count without treating a float log as a proof.
    if lower < 2/CASE_STATISTICAL_FAILURE:
        raise ArithmeticError("Exact rational statistical-budget check failed")
    tail = 2*math.exp(-shots*float(STATISTICAL_ERROR)**2/2)
    if tail > float(CASE_STATISTICAL_FAILURE):
        raise ArithmeticError("Chosen shot count does not meet the stated Hoeffding proposal")
    return {
        "observable": "Z on system wire 0", "score": "+1 for output bit 0, -1 for bit 1",
        "initial": "product: +i on system0, + on every other system; actual H/S gates",
        "family": list(FAMILY), "splitting_bias_allowance": float(SPLITTING_ALLOWANCE),
        "statistical_absolute_error_per_output": float(STATISTICAL_ERROR),
        "ideal_angle_absolute_error_per_output": float(SPLITTING_ALLOWANCE+STATISTICAL_ERROR),
        "family_statistical_failure": float(FAMILY_STATISTICAL_FAILURE),
        "case_statistical_failure": float(CASE_STATISTICAL_FAILURE),
        "case_failure_exact": {"numerator": CASE_STATISTICAL_FAILURE.numerator, "denominator": CASE_STATISTICAL_FAILURE.denominator},
        "complete_trials_per_case": shots, "computed_hoeffding_tail_per_case": tail,
        "hoeffding_threshold_exactly_verified": True,
        "hoeffding_arithmetic": "positive exponential Taylor sum through degree32 proves exp(N*eps^2/2)>=2/delta using Fraction arithmetic",
        "computed_family_union_bound": len(FAMILY)*tail,
        "simultaneous_scope": "all three predeclared local means; error .04 for each output, family failure at most .005 in the ideal-angle model",
        "trial_ensemble_executed": False, "hardware_execution": False,
        "finite_angle_confidence_certificate": False,
    }


def on_wire(matrix: np.ndarray, wire: int, n: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("Embedding requires a finite two-by-two matrix")
    if isinstance(wire, bool) or isinstance(n, bool) or not isinstance(wire, int) or not isinstance(n, int) or not 0 <= wire < n <= MAX_DENSE_SYSTEM_QUBITS:
        raise ValueError("Matrix embedding exceeds the four-system dense domain")
    result = np.array([[1]], dtype=complex)
    for position in reversed(range(n)):
        result = np.kron(result, matrix if position == wire else I2)
    return result


def reference_initial(task: ChainTask) -> np.ndarray:
    plus = np.array([1, 1], dtype=complex)/math.sqrt(2)
    plus_i = np.array([1, 1j], dtype=complex)/math.sqrt(2)
    vector = np.array([1], dtype=complex)
    for wire in reversed(range(task.n_system)):
        vector = np.kron(vector, plus_i if wire == 0 else plus)
    return np.outer(vector, vector.conj())


def _density(rho, n):
    rho = np.asarray(rho, dtype=complex)
    d = 2**n
    if n not in FAMILY or rho.shape != (d, d) or not np.all(np.isfinite(rho)):
        raise ValueError("Density shape/values exceed the four-system domain")
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0) or not np.isclose(np.trace(rho), 1, atol=1e-12, rtol=0):
        raise ValueError("Density must be Hermitian with trace one")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("Density must be positive semidefinite")
    return rho


def reference_generators(task: ChainTask) -> dict[str, np.ndarray]:
    """Independent column-vectorized generators; no circuit construction calls."""
    n, d = task.n_system, 2**task.n_system
    identity = np.eye(d, dtype=complex)
    hzz = task.J*sum((on_wire(Z, j, n) @ on_wire(Z, j+1, n) for j in range(n-1)), np.zeros((d, d), complex))
    hx = task.hx*sum((on_wire(X, j, n) for j in range(n)), np.zeros((d, d), complex))
    matrices = {name: -1j*(np.kron(identity, H)-np.kron(H.T, identity)) for name, H in (("ZZ", hzz), ("X", hx))}
    for wire, rate in enumerate(task.rates):
        jump = on_wire(LOWER, wire, n)
        loss = jump.conj().T @ jump
        matrices[f"D{wire}"] = rate*(np.kron(jump.conj(), jump)-.5*(np.kron(identity, loss)+np.kron(loss.T, identity)))
    return matrices


def reference_channel(task: ChainTask) -> np.ndarray:
    matrices = reference_generators(task)
    return expm(task.time*sum(matrices.values(), np.zeros_like(matrices["ZZ"])))


def reference_density(task: ChainTask, rho=None, channel=None) -> np.ndarray:
    rho = _density(reference_initial(task) if rho is None else rho, task.n_system)
    channel = reference_channel(task) if channel is None else channel
    d = len(rho)
    if channel.shape != (d*d, d*d):
        raise ValueError("Reference channel shape mismatch")
    return (channel @ rho.reshape(d*d, order="F")).reshape((d, d), order="F")


def component_gate_channel(task: ChainTask, name: str, duration: float) -> np.ndarray:
    """Classical reduced channel extracted from actual small gate operators."""
    if isinstance(duration, bool) or not isinstance(duration, Real) or not math.isfinite(duration) or not 0 <= duration <= 2:
        raise ValueError("Component duration must lie in the bounded interval [0,2]")
    n = task.n_system
    if name in ("ZZ", "X"):
        circuit = QuantumCircuit(n)
        if name == "ZZ":
            append_zz(circuit, duration, task.J, n)
        else:
            append_x_layer(circuit, duration, task.hx, n)
        U = Operator(circuit).data
        return np.kron(U.conj(), U)
    if not name.startswith("D") or not name[1:].isdigit() or not 0 <= int(name[1:]) < n:
        raise ValueError("Unknown split component")
    wire = int(name[1:])
    # System is q0, fresh environment q1 in this actual two-wire gate operator.
    unitary = Operator(finite_time_dilation(task.rates[wire], duration)).data
    kraus = [on_wire(unitary[:2, :2], wire, n), on_wire(unitary[2:, :2], wire, n)]
    return sum((np.kron(K.conj(), K) for K in kraus), np.zeros((4**n, 4**n), complex))


def gate_channel(task: ChainTask, steps: int) -> np.ndarray:
    """Compose the complete reduced schedule from gate-derived components.

    Each environment is untouched after its dilation until its next reset or
    ignored final readout. Tracing it immediately therefore preserves every
    later system observable. This exact causal contraction bounds density
    storage at 4^n entries; the composed channel itself has 16^n entries
    (at most256 by256). It is not a full2n-wire gate-by-gate density simulation.
    """
    _steps(steps)
    schedule = chain_schedule(task.n_system, 2)
    one_step = np.eye(4**task.n_system, dtype=complex)
    cache = {}
    for name, fraction in schedule:
        key = (name, fraction)
        if key not in cache:
            cache[key] = component_gate_channel(task, name, task.time/steps*fraction)
        one_step = cache[key] @ one_step
    return np.linalg.matrix_power(one_step, steps)


def gate_density(task: ChainTask, steps: int, rho=None, channel=None) -> np.ndarray:
    rho = _density(reference_initial(task) if rho is None else rho, task.n_system)
    channel = gate_channel(task, steps) if channel is None else channel
    d = len(rho)
    return (channel @ rho.reshape(d*d, order="F")).reshape((d, d), order="F")


def full_circuit_density(task: ChainTask, steps: int, rho=None) -> np.ndarray:
    """Small whole-2n-wire check, separately capped at four steps/four systems."""
    _steps(steps)
    if steps > MAX_FULL_CIRCUIT_STEPS:
        raise ValueError("Whole-circuit density checks are capped at four steps")
    n = task.n_system
    if rho is None:
        circuit = build_chain_circuit(n, task.time, task.J, task.hx, task.rates, steps, 2, "product")
        joint = DensityMatrix.from_instruction(circuit)
    else:
        rho = _density(rho, n)
        circuit = build_chain_circuit(n, task.time, task.J, task.hx, task.rates, steps, 2, None, initial_system_reset=False)
        environment = np.zeros((2**n, 2**n), complex)
        environment[0, 0] = 1
        joint = DensityMatrix(np.kron(environment, rho)).evolve(circuit)
    return partial_trace(joint, list(range(n, 2*n))).data


def measured_trial(task: ChainTask, steps: int | None = None):
    steps = select_steps(task) if steps is None else steps
    _steps(steps)
    source = build_chain_circuit(task.n_system, task.time, task.J, task.hx, task.rates, steps, 2, "product", observable="Z")
    scheduled = source.copy_empty_like()
    for item in source.data:
        event = item.operation.name in {"reset", "measure"}
        if event:
            scheduled.barrier()
        scheduled.append(item.operation, item.qubits, item.clbits)
        if event:
            scheduled.barrier()
    compiled = lower_to_rotation_basis(scheduled)
    expected_resets = task.n_system+(2*task.n_system-1)*steps
    if compiled.count_ops().get("reset") != expected_resets or compiled.count_ops().get("measure") != 2*task.n_system:
        raise RuntimeError("Lowering changed the complete initialization/readout contract")
    return source, compiled


def logical_ledger(task: ChainTask, source, compiled) -> dict:
    metadata = source.metadata or {}
    expected = {"n_system": task.n_system, "time": task.time, "J": task.J,
                "hx": task.hx, "rates": list(task.rates), "initial": "product",
                "order": 2, "observable_group": "Z"}
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError("Ledger source does not match the declared Strang/local-Z0 task")
    if source.num_qubits != 2*task.n_system or compiled.num_qubits != source.num_qubits:
        raise ValueError("Ledger circuit width does not match the declared task")
    for event in ("reset", "measure"):
        if compiled.count_ops().get(event, 0) != source.count_ops().get(event, 0):
            raise ValueError("Ledger compiled initialization/readout events differ from source")
    shots = output_contract()["complete_trials_per_case"]
    counts = {name: int(count) for name, count in compiled.count_ops().items() if name != "barrier"}
    no_barriers = compiled.copy_empty_like()
    for item in compiled.data:
        if item.operation.name != "barrier":
            no_barriers.append(item.operation, item.qubits, item.clbits)
    return {
        "logical_qubits": compiled.num_qubits, "source_counts": dict(source.count_ops()),
        "primitive_counts_per_complete_trial": counts,
        "primitive_counts_all_trials": {name: count*shots for name, count in counts.items()},
        "complete_trials": shots, "algorithm_retry_cap": 1, "postselection": "none",
        "system_initializations_per_trial": task.n_system,
        "environment_initializations_per_trial": source.metadata["environment_initializations"],
        "environment_discard_resets_before_reuse_per_trial": source.metadata["environment_discard_resets_before_reuse"],
        "system_readouts_per_trial": task.n_system, "ignored_environment_readouts_per_trial": task.n_system,
        "output_clbit": 0, "other_system_bits_ignored": list(range(1, task.n_system)),
        "basis_rotation": "none for the fixed Z0 mean; all systems are measured in Z",
        "compiler_event_barriers": int(compiled.count_ops().get("barrier", 0)),
        "dependency_depth_without_barriers": no_barriers.depth(),
        "logical_qubit_layer_volume_all_trials": compiled.num_qubits*no_barriers.depth()*shots,
        "volume_scope": "dependency-layer proxy; no physical qubit-seconds or hardware time",
        "rotation_count_per_trial": sum(counts.get(name, 0) for name in ("rx", "ry", "rz")),
        "actual_clifford_t_synthesis": False,
        "classical_online_work": "one Z0 score and accumulator/counter update per full trial; wall time unmeasured",
        "excluded": ["offline circuit/angle preparation and final statistics wall time", "coherent Clifford+T synthesis", "physical routing and schedule", "surface-code distance/decoder calibration", "magic-state factories and transport", "physical initialization/reset/readout/feedback fault and timing model"],
    }


def compare(task: ChainTask, steps: int, reference=None) -> dict:
    actual = gate_density(task, steps)
    exact = reference_density(task) if reference is None else reference
    difference = actual-exact
    trace_norm = float(np.linalg.norm(difference, ord="nuc"))
    op = on_wire(Z, 0, task.n_system)
    mean = float(np.trace(op@actual).real)
    exact_mean = float(np.trace(op@exact).real)
    bound = uniform_split_bound(task, steps)
    if trace_norm > float(bound)+1e-10:
        raise AssertionError("Numerical state check violates the declared analytic envelope")
    if abs(np.trace(actual)-1) > 2e-11 or np.linalg.eigvalsh(actual).min() < -2e-11:
        raise AssertionError("Gate-contracted numerical state is not a valid density")
    return {
        "n_system": task.n_system, "steps": steps, "generator_norm_envelope": float(generator_norm_envelope(task)),
        "uniform_trace_norm_bound_exact": {"numerator": bound.numerator, "denominator": bound.denominator},
        "uniform_trace_norm_bound_display": float(bound), "observed_trace_norm_error": trace_norm,
        "observed_trace_distance": trace_norm/2, "observed_Z0_mean": mean, "reference_Z0_mean": exact_mean,
        "observed_Z0_bias": abs(mean-exact_mean), "trace_real": float(np.trace(actual).real),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(actual).min()),
        "actual_density": actual, "reference_density": exact,
        "numerical_check_kind": "full reduced schedule composed from actual gate-derived channels; independent full GKSL exponential",
        "whole_2n_wire_density_simulation": False,
        "uniform_bound_scope": "all-input exact-angle evolution trace norm; preparation/roundoff/synthesis/physical faults excluded",
        "floating_interval_certificate": False, "formal_lean_certificate": False,
    }
