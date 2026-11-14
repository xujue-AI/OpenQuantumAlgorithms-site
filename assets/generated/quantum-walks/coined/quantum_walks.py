# SPDX-License-Identifier: MIT
"""Explicit bounded coined quantum walks on a binary-encoded cycle.

Position occupies the least-significant n wires; the final wire is the coin.
One step is S (H_coin tensor I). S moves coin 0 by -1 and coin 1 by +1,
modulo 2**n. Binary carry gates implement the shift; no matrix gate is used.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

MAX_POSITION_QUBITS = 12
MAX_STEPS = 256
MAX_SOURCE_WORK = 2_000_000
MAX_PRIMITIVE_OPERATIONS = 250_000
MAX_STATEVECTOR_QUBITS = 12
MAX_SIMULATION_STEPS = 64
MAX_DENSE_POSITION_QUBITS = 6
PRIMITIVE_BASIS = ("h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx")
CoinInput = Literal["zero", "one", "balanced"]


def _integer(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _parameters(n: int, steps: int = 0, initial_position: int = 0, initial_coin: CoinInput = "balanced") -> None:
    _integer("position qubits", n, 1, MAX_POSITION_QUBITS)
    _integer("steps", steps, 0, MAX_STEPS)
    _integer("initial_position", initial_position, 0, (1 << n) - 1)
    if initial_coin not in ("zero", "one", "balanced"):
        raise ValueError("initial_coin must be zero, one, or balanced")
    # Conservative source-work guard before controlled-gate expansion. It is
    # not a certified bound on the transpiler's temporary memory or runtime.
    if steps * (1 << (n + 3)) > MAX_SOURCE_WORK:
        raise ValueError("Walk exceeds conservative controlled-shift source-work guard")


def _mcx(circuit: QuantumCircuit, controls: list[int], target: int) -> None:
    if len(controls) == 1:
        circuit.cx(controls[0], target)
    else:
        circuit.mcx(controls, target)


def _controlled_increment(circuit: QuantumCircuit, n: int, *, inverse: bool = False) -> None:
    """Increment/decrement position conditional on coin 1, without scratch."""
    coin = n
    if inverse:
        circuit.cx(coin, 0)
        for bit in range(1, n):
            _mcx(circuit, [coin, *range(bit)], bit)
    else:
        for bit in range(n - 1, 0, -1):
            _mcx(circuit, [coin, *range(bit)], bit)
        circuit.cx(coin, 0)


def build_conditional_shift(n: int) -> QuantumCircuit:
    """Return the actual coin-controlled +/-1 permutation circuit."""
    _parameters(n)
    circuit = QuantumCircuit(n + 1, name="cycle_shift")
    _controlled_increment(circuit, n)
    circuit.x(n)
    _controlled_increment(circuit, n, inverse=True)
    circuit.x(n)
    circuit.metadata = {"position_qubits": n, "coin_wire": n,
                        "coin_0_direction": -1, "coin_1_direction": 1,
                        "scratch_qubits": 0, "modulus": 1 << n}
    return circuit


def build_coined_walk(n: int, steps: int, *, initial_position: int = 0,
                      initial_coin: CoinInput = "balanced", measure: bool = False) -> QuantumCircuit:
    """Prepare and execute a Hadamard-coin walk; optionally measure all wires.

    balanced means (|0>+i|1>)/sqrt(2), prepared by H then S. Every complete
    measured trial reads the coin as well; the position statistic ignores its
    outcome, without postselection.
    """
    _parameters(n, steps, initial_position, initial_coin)
    if not isinstance(measure, bool):
        raise ValueError("measure must be a bool")
    circuit = QuantumCircuit(n + 1, n + 1) if measure else QuantumCircuit(n + 1)
    for bit in range(n):
        if (initial_position >> bit) & 1:
            circuit.x(bit)
    if initial_coin == "one":
        circuit.x(n)
    elif initial_coin == "balanced":
        circuit.h(n)
        circuit.s(n)
    shift = build_conditional_shift(n)
    for _ in range(steps):
        circuit.h(n)
        circuit.compose(shift, inplace=True)
    if measure:
        circuit.measure(range(n + 1), range(n + 1))
    circuit.metadata = {"model": "Hadamard coined cycle walk", "position_qubits": n,
        "vertices": 1 << n, "steps": steps, "initial_position": initial_position,
        "initial_coin": initial_coin, "coin_wire": n,
        "step_chronology": "Hadamard coin, then +/-1 controlled shift",
        "output_contract": "all measured coin/position outcomes; ignore coin for position mean",
        "source_shift_operations_per_step": 2*n+2, "scratch_qubits": 0}
    return circuit


def lower_walk(circuit: QuantumCircuit) -> QuantumCircuit:
    """Lower our bound circuits to explicit primitive gates and check size."""
    if circuit.num_parameters:
        raise ValueError("Bind circuit parameters before lowering")
    allowed_source = {*PRIMITIVE_BASIS, "ccx", "mcx", "measure", "barrier"}
    if any(item.operation.name not in allowed_source for item in circuit.data):
        raise ValueError("Only explicitly supported walk gates can be lowered")
    if not math.isfinite(float(circuit.global_phase)):
        raise ValueError("Circuit global phase must be finite")
    measured = False
    for item in circuit.data:
        if item.operation.name == "measure":
            measured = True
        elif measured and item.operation.name != "barrier":
            raise ValueError("Walk readout must use final measurements")
        if any(not math.isfinite(float(parameter)) for parameter in item.operation.params):
            raise ValueError("Circuit rotation parameters must be finite")
    compiled = transpile(circuit, basis_gates=list(PRIMITIVE_BASIS),
                         optimization_level=1, seed_transpiler=7)
    if len(compiled.data) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("Compiled circuit exceeds primitive-operation export limit")
    if any(item.operation.name not in {*PRIMITIVE_BASIS, "measure", "barrier"} for item in compiled.data):
        raise ValueError("Lowering retained an unsupported operation")
    return compiled


def reference_step_matrix(n: int) -> np.ndarray:
    """Independent permutation and Kronecker-product reference, bounded."""
    _integer("dense-reference position qubits", n, 1, MAX_DENSE_POSITION_QUBITS)
    vertices = 1 << n
    shift = np.zeros((2*vertices, 2*vertices), dtype=complex)
    for coin in range(2):
        direction = -1 if coin == 0 else 1
        for position in range(vertices):
            shift[coin*vertices + (position+direction) % vertices,
                  coin*vertices + position] = 1
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex)/math.sqrt(2)
    return shift @ np.kron(hadamard, np.eye(vertices))


def reference_initial_state(n: int, initial_position: int = 0,
                            initial_coin: CoinInput = "balanced") -> np.ndarray:
    _parameters(n, 0, initial_position, initial_coin)
    _integer("dense-reference position qubits", n, 1, MAX_DENSE_POSITION_QUBITS)
    position = np.zeros(1 << n, dtype=complex)
    position[initial_position] = 1
    coin = {"zero": np.array([1, 0]), "one": np.array([0, 1]),
            "balanced": np.array([1, 1j])/math.sqrt(2)}[initial_coin]
    return np.kron(coin, position)


def reference_roll_state(n: int, steps: int, initial_position: int = 0,
                         initial_coin: CoinInput = "balanced") -> np.ndarray:
    """Independent amplitude recurrence, without circuit or carry operations."""
    _parameters(n,steps,initial_position,initial_coin)
    if n+1>MAX_STATEVECTOR_QUBITS or steps>MAX_SIMULATION_STEPS:
        raise ValueError("Amplitude recurrence exceeds bounded wire/step limits")
    state=np.zeros((2,1<<n),dtype=complex)
    state[:,initial_position]={"zero":np.array([1,0]),"one":np.array([0,1]),
        "balanced":np.array([1,1j])/math.sqrt(2)}[initial_coin]
    for _ in range(steps):
        left=(state[0]+state[1])/math.sqrt(2)
        right=(state[0]-state[1])/math.sqrt(2)
        state=np.vstack((np.roll(left,-1),np.roll(right,1)))
    return state.reshape(-1)


def simulate_coined_walk(n: int, steps: int, *, initial_position: int = 0,
                        initial_coin: CoinInput = "balanced") -> dict:
    """Evaluate the actual circuit, separately bounded from construction."""
    _parameters(n, steps, initial_position, initial_coin)
    if n + 1 > MAX_STATEVECTOR_QUBITS or steps > MAX_SIMULATION_STEPS:
        raise ValueError("Statevector validation exceeds bounded wire/step limits")
    circuit = build_coined_walk(n, steps, initial_position=initial_position, initial_coin=initial_coin)
    state = Statevector.from_instruction(circuit).data
    probabilities = np.sum(abs(state.reshape(2, 1 << n))**2, axis=0)
    recurrence=reference_roll_state(n,steps,initial_position,initial_coin)
    recurrence_error=float(np.max(abs(state-recurrence)))
    np.testing.assert_allclose(state,recurrence,atol=3e-12,rtol=0)
    result = {"performed": True, "statevector": state, "probabilities": probabilities,
              "return_probability": float(probabilities[initial_position]),
              "norm": float(np.linalg.norm(state)), "independent_dense_reference_performed": False,
              "independent_amplitude_recurrence_max_error":recurrence_error}
    if n <= MAX_DENSE_POSITION_QUBITS:
        reference = np.linalg.matrix_power(reference_step_matrix(n), steps) @ reference_initial_state(n, initial_position, initial_coin)
        discrepancy = float(np.max(abs(state-reference)))
        np.testing.assert_allclose(state, reference, atol=3e-12, rtol=0)
        result.update(independent_dense_reference_performed=True, state_amplitude_max_error=discrepancy,
                      reference_probabilities=np.sum(abs(reference.reshape(2, 1 << n))**2, axis=0))
    return result


def classical_cycle_distribution(n: int, steps: int, initial_position: int = 0) -> np.ndarray:
    """Independent unbiased stochastic walk; a comparator, not quantum check."""
    _parameters(n, steps, initial_position)
    probabilities = np.zeros(1 << n)
    probabilities[initial_position] = 1
    for _ in range(steps):
        probabilities = (np.roll(probabilities, 1)+np.roll(probabilities, -1))/2
    return probabilities


def coined_observable_budget():
    """One fixed return-event mean, with distinct sampling/model/fault inputs."""
    from .resource_estimation import BoundedObservableBudget
    return BoundedObservableBudget(absolute_error_target=.0301, algorithm_bias_bound=0,
        synthesis_bias_allowance=1e-4, sampling_error_allowance=.03,
        sampling_failure_probability=.005, physical_failure_probability=.005,
        outcome_min=0, outcome_max=1)


def sample_position_outcomes(probabilities: np.ndarray, shots: int, *, seed: int = 731) -> np.ndarray:
    """Seeded classical categorical draws from an ideal validated law."""
    _integer("shots", shots, 1, 100_000)
    _integer("seed", seed, 0, (1 << 32)-1)
    if np.iscomplexobj(probabilities):
        raise ValueError("Probabilities must be real")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Probabilities must be a finite nonnegative vector")
    if not np.isclose(values.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Probabilities must sum to one")
    return np.random.default_rng(seed).multinomial(shots, values/values.sum())
