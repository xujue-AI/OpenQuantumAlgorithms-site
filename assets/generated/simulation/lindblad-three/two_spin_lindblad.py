# SPDX-License-Identifier: MIT
"""Global Lie/Strang circuits for a two-spin Hamiltonian and local damping.

The executed density study uses H=J ZZ+hx(X0+X1), system wires0/1 and
environment wires2/3. A separate open-chain generator supports up to16
systems with J sum ZZ+hx sum X and one local damping environment per system.
Each Hamiltonian component is an actual CX-RZ-CX or RX layer. Each damping
component is a survival-aware CRY/CX dilation with a fresh environment reset.
No dense matrix is appended as a quantum operation. Independent4x4 density
and16x16 column-vectorized GKSL references are bounded classical checks.

Original MIT code. Qiskit2.6.2 is Apache-2.0, SciPy1.18.1 is BSD-3-Clause:
https://github.com/Qiskit/qiskit/tree/2.6.2
https://github.com/scipy/scipy/tree/v1.18.1
Primary context: Lindblad1976 doi:10.1007/BF01608499; Strang1968
doi:10.1137/0705041; Kliesch2011 doi:10.1103/PhysRevLett.107.120501 with
erratum2012 doi:10.1103/PhysRevLett.109.119904.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy.linalg import expm

if __package__:
    from .open_systems import ROTATION_BASIS, finite_time_dilation, lower_to_rotation_basis as _base_lower
else:
    from open_systems import ROTATION_BASIS, finite_time_dilation, lower_to_rotation_basis as _base_lower

MAX_STEPS = 1024
MAX_SYSTEM_QUBITS = 16
MAX_PRIMITIVE_OPERATIONS = 50_000
MAX_SIMULATION_STEPS = 128
MAX_VALIDATION_POINTS = 16
MAX_REFERENCE_EXPOSURE = 100.0
MAX_ENERGY_BUDGET_TRIALS = 10**12
INITIAL_LABELS = ("product", "bell", "zero")


def lower_to_rotation_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Lower the actual circuit and enforce the compiled materialization cap."""
    if circuit.num_qubits > 2*MAX_SYSTEM_QUBITS:
        raise ValueError("circuit exceeds the supported chain width")
    result = _base_lower(circuit)
    if result.size() > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("compiled primitive materialization exceeds its cap")
    return result


def _real(value: float, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _rates(rates) -> tuple[float, float]:
    if not isinstance(rates, (tuple, list, np.ndarray)) or np.ndim(rates) != 1 or len(rates) != 2:
        raise ValueError("rates must contain two nonnegative local rates")
    return _real(rates[0], "rate0", nonnegative=True), _real(rates[1], "rate1", nonnegative=True)


def _problem(time: float, J: float, hx: float, rates) -> tuple[float, float, float, tuple[float, float]]:
    return _real(time, "time", nonnegative=True), _real(J, "J"), _real(hx, "hx"), _rates(rates)


def split_schedule(order: int) -> tuple[tuple[str, float], ...]:
    """Chronological positive-time schedule; each component receives dt total."""
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if order == 1:
        return (("ZZ", 1.0), ("X", 1.0), ("D0", 1.0), ("D1", 1.0))
    return (("ZZ", 0.5), ("X", 0.5), ("D0", 0.5), ("D1", 1.0),
            ("D0", 0.5), ("X", 0.5), ("ZZ", 0.5))


def chain_schedule(n_system: int, order: int) -> tuple[tuple[str, float], ...]:
    if isinstance(n_system, bool) or not isinstance(n_system, int) or not 1 <= n_system <= MAX_SYSTEM_QUBITS:
        raise ValueError(f"n_system must be an integer from 1 to {MAX_SYSTEM_QUBITS}")
    split_schedule(order)
    if order == 1:
        return (("ZZ", 1.0), ("X", 1.0), *((f"D{j}", 1.0) for j in range(n_system)))
    outer_damping = tuple((f"D{j}", 0.5) for j in range(n_system-1))
    return (("ZZ", 0.5), ("X", 0.5), *outer_damping, (f"D{n_system-1}", 1.0),
            *reversed(outer_damping), ("X", 0.5), ("ZZ", 0.5))


def _steps(steps: int, order: int) -> None:
    split_schedule(order)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer from 1 to {MAX_STEPS}")
    if steps * (28 if order == 2 else 17) + 16 > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive materialization estimate exceeds its explicit cap")


def prepare_initial(circuit: QuantumCircuit, initial: str, n_system: int = 2) -> None:
    """Prepare named states from system00 with actual H/S/CX gates."""
    if initial not in INITIAL_LABELS:
        raise ValueError("initial must be product, bell or zero")
    if initial == "product":
        circuit.h(0)
        circuit.s(0)
        if n_system > 1:
            circuit.h(range(1, n_system))
    elif initial == "bell":
        if n_system != 2:
            raise ValueError("Bell preparation is implemented only for two system wires")
        circuit.h(0)
        circuit.cx(0, 1)


def append_zz(circuit: QuantumCircuit, time: float, J: float, n_system: int = 2) -> None:
    time, J = _real(time, "time", nonnegative=True), _real(J, "J")
    if isinstance(n_system, bool) or not isinstance(n_system, int) or not 1 <= n_system <= circuit.num_qubits:
        raise ValueError("n_system must fit the circuit's system register")
    if n_system == 1 or time == 0 or J == 0:
        return
    angle = 2 * (J * time)
    if not math.isfinite(angle):
        raise ValueError("ZZ rotation angle overflowed finite arithmetic")
    for system in range(n_system-1):
        circuit.cx(system, system+1)
        circuit.rz(angle, system+1)
        circuit.cx(system, system+1)


def append_x_layer(circuit: QuantumCircuit, time: float, hx: float, n_system: int = 2) -> None:
    time, hx = _real(time, "time", nonnegative=True), _real(hx, "hx")
    if isinstance(n_system, bool) or not isinstance(n_system, int) or not 1 <= n_system <= circuit.num_qubits:
        raise ValueError("n_system must fit the circuit's system register")
    if time == 0 or hx == 0:
        return
    angle = 2 * (hx * time)
    if not math.isfinite(angle):
        raise ValueError("X rotation angle overflowed finite arithmetic")
    circuit.rx(angle, range(n_system))


def build_chain_circuit(n_system: int = 2, time: float = 1.0, J: float = 0.6, hx: float = 0.35,
                           rates=None, steps: int = 32, order: int = 2,
                           initial: str | None = "product", *,
                           observable: str | None = None,
                           initial_system_reset: bool = True) -> QuantumCircuit:
    """Explicit open-chain gate generator, separate from the fixed n=2 checks.

    The global palindrome covers ZZlayer, Xlayer and every local D as components; there is no
    separately approximated combined-H substep. Its error includes H-term and
    Hamiltonian/dissipator commutators. D0 half means rate0*dt/2, not p_full/2.
    Unknown caller input uses initial=None and initial_system_reset=False.
    """
    schedule = chain_schedule(n_system, order)
    time, J, hx = _real(time, "time", nonnegative=True), _real(J, "J"), _real(hx, "hx")
    if rates is not None and (not isinstance(rates, (tuple, list, np.ndarray)) or np.ndim(rates) != 1):
        raise ValueError("rates must contain one rate per system wire")
    rates = (0.8,)+(0.5,)*(n_system-1) if rates is None else tuple(rates)
    if len(rates) != n_system:
        raise ValueError("rates must contain one rate per system wire")
    rates = tuple(_real(rate, "rate", nonnegative=True) for rate in rates)
    _steps(steps, order)
    per_step = 10*n_system-3 if order == 1 else 20*n_system-12
    if steps*per_step+6*n_system+8 > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("n-dependent primitive materialization estimate exceeds its cap")
    if initial is not None and initial not in INITIAL_LABELS:
        raise ValueError("unsupported initial-state label")
    if initial is None and initial_system_reset:
        raise ValueError("do not erase an unknown caller-supplied system input")
    if observable is not None and observable not in ("X", "Y", "Z"):
        raise ValueError("observable group must be X, Y, Z or None")
    circuit = QuantumCircuit(2*n_system, 2*n_system if observable else 0,
                             name=f"Spin-chain PF{order} n={n_system} r={steps}")
    if initial_system_reset:
        circuit.reset(range(n_system))
    if initial is not None:
        prepare_initial(circuit, initial, n_system)
    dt = time / steps
    for _ in range(steps):
        for component, fraction in schedule:
            subtime = dt * fraction
            if component == "ZZ":
                append_zz(circuit, subtime, J, n_system)
            elif component == "X":
                append_x_layer(circuit, subtime, hx, n_system)
            else:
                system = int(component[1:])
                environment = system + n_system
                circuit.reset(environment)
                circuit.compose(finite_time_dilation(rates[system], subtime),
                                qubits=[system, environment], inplace=True)
    if observable:
        for system in range(n_system):
            if observable == "Y":
                circuit.sdg(system)
            if observable in ("X", "Y"):
                circuit.h(system)
        circuit.measure(range(2*n_system), range(2*n_system))
    damping_calls = [steps if order == 1 or j == n_system-1 else 2*steps for j in range(n_system)]
    circuit.metadata = {
        "n_system": n_system, "systems": list(range(n_system)),
        "environments": list(range(n_system, 2*n_system)), "time": time,
        "J": J, "hx": hx, "rates": list(rates), "steps": steps, "order": order,
        "chronological_schedule": list(schedule), "initial": initial or "caller supplied",
        "system_initializations": n_system if initial_system_reset else 0,
        "component_calls": {"ZZ": steps if order == 1 else 2*steps,
                            "Xlayer": steps if order == 1 else 2*steps,
                            **{f"D{j}": calls for j, calls in enumerate(damping_calls)}},
        "environment_initializations": damping_calls,
        "environment_discard_resets_before_reuse": [calls-1 for calls in damping_calls],
        "final_environment_handling": "trace for retained state; both Z records ignored for measured trial",
        "postselection": "none", "observable_group": observable,
        "splitting_contract": "global component splitting; no separate inner combined-H approximation",
    }
    return circuit


def build_two_spin_circuit(time: float = 1.0, J: float = 0.6, hx: float = 0.35,
                           rates=(0.8, 0.5), steps: int = 32, order: int = 2,
                           initial: str | None = "product", *,
                           observable: str | None = None,
                           initial_system_reset: bool = True) -> QuantumCircuit:
    """The fixed two-system/two-environment checked instance of the generator."""
    return build_chain_circuit(2, time, J, hx, rates, steps, order, initial,
                                observable=observable, initial_system_reset=initial_system_reset)


def observable_operators() -> dict[str, np.ndarray]:
    """Independent small matrix observables, with rightmost factor system0."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    return {"X0": np.kron(I, X), "X1": np.kron(X, I),
            "Y0": np.kron(I, Y), "Y1": np.kron(Y, I),
            "Z0": np.kron(I, Z), "Z1": np.kron(Z, I), "ZZ": np.kron(Z, Z)}


def initial_density(initial: str = "product") -> np.ndarray:
    """Classical input reference, not a state-loading gate."""
    if initial == "product":
        plus = np.array([1, 1], dtype=complex)/math.sqrt(2)
        plus_i = np.array([1, 1j], dtype=complex)/math.sqrt(2)
        vector = np.kron(plus, plus_i)
    elif initial == "bell":
        vector = np.array([1, 0, 0, 1], dtype=complex)/math.sqrt(2)
    elif initial == "zero":
        vector = np.array([1, 0, 0, 0], dtype=complex)
    else:
        raise ValueError("unsupported initial-state reference")
    return np.outer(vector, vector.conj())


def _density(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4) or not np.all(np.isfinite(rho)):
        raise ValueError("density must be finite and4x4")
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0) or not np.isclose(np.trace(rho), 1, atol=1e-12, rtol=0):
        raise ValueError("density must be Hermitian with trace one")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("density must be positive semidefinite")
    return rho


def hamiltonian(J: float = 0.6, hx: float = 0.35) -> np.ndarray:
    _, J, hx, _ = _problem(0, J, hx, (0, 0))
    operators = observable_operators()
    return J*operators["ZZ"]+hx*(operators["X0"]+operators["X1"])


def liouvillian(J: float = 0.6, hx: float = 0.35, rates=(0.8, 0.5)) -> np.ndarray:
    """Independent standard-GKSL16x16 matrix in column-vectorization order."""
    _, J, hx, rates = _problem(0, J, hx, rates)
    maximum = np.finfo(float).max
    if abs(J) > maximum/2 or rates[0] > maximum-rates[1]:
        raise ValueError("raw Liouvillian entries would overflow finite arithmetic")
    H = hamiltonian(J, hx)
    I4 = np.eye(4, dtype=complex)
    result = -1j*(np.kron(I4, H)-np.kron(H.T, I4))
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    loss = np.diag([0.0, 1.0]).astype(complex)
    I2 = np.eye(2, dtype=complex)
    for rate, jump, K in (
        (rates[0], np.kron(I2, lowering), np.kron(I2, loss)),
        (rates[1], np.kron(lowering, I2), np.kron(loss, I2)),
    ):
        template = np.kron(jump.conj(), jump)-0.5*(np.kron(I4, K)+np.kron(K.T, I4))
        result += rate*template
    if not np.all(np.isfinite(result)):
        raise ValueError("raw Liouvillian entries overflowed finite arithmetic")
    return result


def reference_exposure(time: float, J: float, hx: float, rates) -> float:
    time, J, hx, rates = _problem(time, J, hx, rates)
    return 2*(abs(time*J)+2*abs(time*hx))+time*rates[0]+time*rates[1]


def exact_channel(time: float = 1.0, J: float = 0.6, hx: float = 0.35,
                   rates=(0.8, 0.5)) -> np.ndarray:
    """Bounded dense answer formed from scaled coefficients before assembly."""
    time, J, hx, rates = _problem(time, J, hx, rates)
    if time == 0:
        return np.eye(16, dtype=complex)
    if reference_exposure(time, J, hx, rates) > MAX_REFERENCE_EXPOSURE:
        raise ValueError("reference exposure exceeds100")
    return expm(liouvillian(time*J, time*hx, (time*rates[0], time*rates[1])))


def exact_density(rho: np.ndarray, time: float = 1.0, J: float = 0.6,
                   hx: float = 0.35, rates=(0.8, 0.5)) -> np.ndarray:
    rho = _density(rho)
    return (exact_channel(time, J, hx, rates) @ rho.reshape(16, order="F")).reshape((4, 4), order="F")


def split_channel_reference(time: float = 1.0, J: float = 0.6, hx: float = 0.35,
                             rates=(0.8, 0.5), steps: int = 32, order: int = 2) -> np.ndarray:
    """Classical matrix reference of the stated split schedule; not a circuit."""
    time, J, hx, rates = _problem(time, J, hx, rates)
    _steps(steps, order)
    if reference_exposure(time, J, hx, rates) > MAX_REFERENCE_EXPOSURE:
        raise ValueError("reference exposure exceeds100")
    dt = time/steps
    one_step = np.eye(16, dtype=complex)
    for component, fraction in split_schedule(order):
        subtime = dt*fraction
        if component == "ZZ":
            matrix = liouvillian(subtime*J, 0, (0, 0))
        elif component == "X":
            matrix = liouvillian(0, subtime*hx, (0, 0))
        elif component == "D0":
            matrix = liouvillian(0, 0, (subtime*rates[0], 0))
        else:
            matrix = liouvillian(0, 0, (0, subtime*rates[1]))
        one_step = expm(matrix) @ one_step
    return np.linalg.matrix_power(one_step, steps)


def simulate_two_spin(time: float = 1.0, J: float = 0.6, hx: float = 0.35,
                       rates=(0.8, 0.5), steps: int = 32, order: int = 2,
                       initial: str = "product") -> np.ndarray:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps > MAX_SIMULATION_STEPS:
        raise ValueError(f"density simulation is capped at {MAX_SIMULATION_STEPS} steps")
    circuit = build_two_spin_circuit(time, J, hx, rates, steps, order, initial)
    return partial_trace(DensityMatrix.from_instruction(circuit), [2, 3]).data


def simulate_input_density(rho: np.ndarray, time: float = 1.0, J: float = 0.6,
                           hx: float = 0.35, rates=(0.8, 0.5),
                           steps: int = 32, order: int = 2) -> np.ndarray:
    """Supplied mixed input initializes the bounded classical checker only."""
    rho = _density(rho)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps > MAX_SIMULATION_STEPS:
        raise ValueError(f"density simulation is capped at {MAX_SIMULATION_STEPS} steps")
    circuit = build_two_spin_circuit(time, J, hx, rates, steps, order, None,
                                    initial_system_reset=False)
    zero_environment = np.zeros((4, 4), dtype=complex)
    zero_environment[0, 0] = 1
    joint = DensityMatrix(np.kron(zero_environment, rho)).evolve(circuit)
    return partial_trace(joint, [2, 3]).data


def expectation_values(rho: np.ndarray, J: float = 0.6, hx: float = 0.35) -> dict[str, float]:
    operators = observable_operators()
    values = {name: float(np.trace(rho @ operator).real) for name, operator in operators.items()}
    values["energy"] = J*values["ZZ"]+hx*(values["X0"]+values["X1"])
    if not math.isfinite(values["energy"]):
        raise ValueError("energy mean exceeds finite floating representation")
    return values


def compare_two_spin(time: float = 1.0, J: float = 0.6, hx: float = 0.35,
                      rates=(0.8, 0.5), steps: int = 32, order: int = 2,
                      initial: str = "product", *, h_diagnostic: bool = True) -> dict:
    actual = simulate_two_spin(time, J, hx, rates, steps, order, initial)
    rho0 = initial_density(initial)
    reference = exact_density(rho0, time, J, hx, rates)
    split_reference = (split_channel_reference(time, J, hx, rates, steps, order) @ rho0.reshape(16, order="F")).reshape((4, 4), order="F")
    actual_means, exact_means = expectation_values(actual, J, hx), expectation_values(reference, J, hx)
    diagnostic = None
    if h_diagnostic:
        h_actual = simulate_two_spin(time, J, hx, (0, 0), steps, order, initial)
        h_exact = exact_density(rho0, time, J, hx, (0, 0))
        diagnostic = float(np.linalg.norm(h_actual-h_exact, ord="nuc")/2)
    return {
        "order": order, "steps": steps, "time": time, "J": J, "hx": hx,
        "rates": list(rates), "initial": initial,
        "trace_distance": float(np.linalg.norm(actual-reference, ord="nuc")/2),
        "circuit_vs_classical_split_trace_distance": float(np.linalg.norm(actual-split_reference, ord="nuc")/2),
        "matrix_entry_max_error": float(np.max(abs(actual-reference))),
        "trace_real": float(np.trace(actual).real), "trace_imag": float(np.trace(actual).imag),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(actual).min()),
        "circuit_observables": actual_means, "exact_observables": exact_means,
        "observable_errors": {name: abs(actual_means[name]-exact_means[name]) for name in actual_means},
        "hamiltonian_only_trace_distance_diagnostic": diagnostic,
        "diagnostic_note": "not additive; global splitting error already contains H-term and H/D commutators",
        "actual_density_real": actual.real.tolist(), "actual_density_imag": actual.imag.tolist(),
    }


def measured_system_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    """Bounded ideal probabilities of system bits; all environment bits summed."""
    if circuit.num_qubits != 4 or circuit.metadata is None or circuit.metadata.get("steps", MAX_SIMULATION_STEPS+1) > MAX_SIMULATION_STEPS:
        raise ValueError("measured circuit exceeds the bounded two-spin checker")
    unmeasured = circuit.remove_final_measurements(inplace=False)
    probabilities = DensityMatrix.from_instruction(unmeasured).probabilities()
    return np.array([sum(probabilities[i] for i in range(16) if i % 4 == outcome) for outcome in range(4)])


def group_scores(group: str, J: float = 0.6, hx: float = 0.35) -> np.ndarray:
    """Scalar score for each q1q0 outcome; within-trial correlations allowed."""
    _, J, hx, _ = _problem(0, J, hx, (0, 0))
    s0 = np.array([1, -1, 1, -1], dtype=float)
    s1 = np.array([1, 1, -1, -1], dtype=float)
    if group == "X":
        return hx*(s0+s1)
    if group == "Z":
        return J*s0*s1
    raise ValueError("energy group must be X or Z")


def energy_shot_budget(J: float = 0.6, hx: float = 0.35,
                        statistical_error: float = 0.03,
                        statistical_failure: float = 0.005) -> dict:
    """Independent bounded scalar-score means plus a union bound across groups."""
    _, J, hx, _ = _problem(0, J, hx, (0, 0))
    error = _real(statistical_error, "statistical_error", nonnegative=True)
    failure = _real(statistical_failure, "statistical_failure", nonnegative=True)
    if error <= 0 or not 0 < failure < 1:
        raise ValueError("statistical error must be positive and failure in(0,1)")
    active = [("X", 4*abs(hx)), ("Z", 2*abs(J))]
    active = [(group, width) for group, width in active if width]
    groups = {}
    for group, width in active:
        allowance, delta = error/len(active), failure/len(active)
        if not math.isfinite(width) or allowance <= 0 or delta <= 0:
            raise ValueError("energy sampling allocation is not representable")
        # Logarithms avoid width**2 overflow and allowance**2 underflow.
        log_trials = 2*(math.log(width)-math.log(allowance))+math.log(math.log(2)-math.log(delta))-math.log(2)
        if log_trials > math.log(MAX_ENERGY_BUDGET_TRIALS):
            raise ValueError("energy sampling proposal exceeds the bounded trial cap")
        trials = max(1, math.ceil(math.exp(log_trials)))
        groups[group] = {"outcome_min": -width/2, "outcome_max": width/2, "width": width,
                         "statistical_error": allowance, "statistical_failure": delta, "trials": trials}
    total = sum(g["trials"] for g in groups.values())
    if total > MAX_ENERGY_BUDGET_TRIALS:
        raise ValueError("energy sampling proposal exceeds the bounded trial cap")
    return {"groups": groups, "total_trials": total,
            "statistical_error": error, "statistical_failure": failure,
            "scope": "one fixed energy mean; correlated bits within a trial are combined into one bounded scalar score"}
