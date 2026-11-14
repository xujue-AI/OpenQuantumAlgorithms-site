# SPDX-License-Identifier: MIT
"""Original bounded weight-zero/one DQI circuits; execution evidence recorded separately.

The implementation is not adopted from the authors' xortools archive. Every
preparation, syndrome and reversible-decoder operation is materialized below.
No simulator, optimizer, sampler, synthesizer or Native estimator runs at import.
Arbitrary RY gates remain explicit and unsynthesized. An external execution
receipt records performed checks and their fixture scope; this source does not
independently validate the bounded generator family or an efficient general decoder.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

from qiskit import QuantumCircuit


DEFAULT_MATRIX = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
DEFAULT_SIGNS = (0, 0, 0, 1)
PRIMITIVE_GATES = frozenset({"h", "x", "z", "t", "tdg", "ry", "cx"})
PROGRAM_STATUS = "bounded_dqi_execution_evidence_external"
SOURCE_REFERENCE = "Nature DOI 10.1038/s41586-025-09527-5; formulation reference only"


@dataclass(frozen=True)
class DQILimits:
    """Separate materialization, statevector and dense-operator ceilings."""
    materialized_qubits: int = 9
    primitive_operations: int = 512
    statevector_qubits: int = 9
    statevector_amplitudes: int = 512
    operator_dimension: int = 512
    decoder_basis_checks: int = 512

    def __post_init__(self):
        ceilings = {"materialized_qubits": 9, "primitive_operations": 512,
                    "statevector_qubits": 9, "statevector_amplitudes": 512,
                    "operator_dimension": 512, "decoder_basis_checks": 512}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in 1..{ceiling}")


def _bounds(limits: DQILimits | None) -> DQILimits:
    limits = DQILimits() if limits is None else limits
    if not isinstance(limits, DQILimits):
        raise ValueError("limits must be a DQILimits instance")
    return limits


def validate_instance(matrix=DEFAULT_MATRIX, signs=DEFAULT_SIGNS, *, ell: int = 1) -> dict:
    """Accept only the explicitly bounded binary, weight-at-most-one family.

    Distinct nonzero rows make the vacuum and every weight-one clause string
    have a distinct syndrome. A bounded truth-table decoder is not an efficient
    general decoding family or a higher-degree DQI implementation.
    """
    if type(ell) is not int or ell != 1:
        raise ValueError("Only the weight-zero/one (ell=1) construction is implemented")
    if not isinstance(matrix, (tuple, list)) or not 1 <= len(matrix) <= 4:
        raise ValueError("The bounded generator supports 1..4 matrix rows")
    if not isinstance(matrix[0], (tuple, list)) or not 1 <= len(matrix[0]) <= 3:
        raise ValueError("The bounded generator supports 1..3 assignment wires")
    n, m = len(matrix[0]), len(matrix)
    rows = []
    for row in matrix:
        if not isinstance(row, (tuple, list)) or len(row) != n:
            raise ValueError("Binary matrix rows must have the same length")
        if any(type(value) is not int or value not in (0, 1) for value in row):
            raise ValueError("Matrix entries must be integer binary values")
        rows.append(tuple(row))
    if any(not any(row) for row in rows) or len(set(rows)) != m:
        raise ValueError("Weight-one decoding requires distinct nonzero matrix rows")
    if not isinstance(signs, (tuple, list)) or len(signs) != m:
        raise ValueError("The sign vector must have one binary entry per matrix row")
    if any(type(value) is not int or value not in (0, 1) for value in signs):
        raise ValueError("Sign entries must be integer binary values")
    return dict(matrix=tuple(rows), signs=tuple(signs), n=n, m=m, ell=1)


def register_map(instance: dict) -> dict:
    """Integer bit q is qubit q; displayed assignment strings are x[n-1]..x[0]."""
    m, n = instance["m"], instance["n"]
    return dict(clause=list(range(m)), assignment=list(range(m, m + n)),
                workspace=[m + n, m + n + 1], classical=list(range(n)),
                total_qubits=m + n + 2)


def _wires(circuit: QuantumCircuit, *wires: int):
    if any(type(q) is not int or not 0 <= q < circuit.num_qubits for q in wires):
        raise ValueError("Gate wires must be distinct integer indices inside the circuit")
    if len(set(wires)) != len(wires):
        raise ValueError("Gate wires must be distinct")


def append_exact_ccx(circuit: QuantumCircuit, control0: int, control1: int, target: int):
    """Materialize the exact 15-gate Clifford+T CCX, without relative phases."""
    _wires(circuit, control0, control1, target)
    circuit.h(target)
    circuit.cx(control1, target)
    circuit.tdg(target)
    circuit.cx(control0, target)
    circuit.t(target)
    circuit.cx(control1, target)
    circuit.tdg(target)
    circuit.cx(control0, target)
    circuit.t(control1)
    circuit.t(target)
    circuit.h(target)
    circuit.cx(control0, control1)
    circuit.t(control0)
    circuit.tdg(control1)
    circuit.cx(control0, control1)


def append_controlled_ry(circuit: QuantumCircuit, control: int, target: int, angle: float):
    """Two explicit RY rotations and two CXs; no opaque controlled-rotation gate."""
    _wires(circuit, control, target)
    if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle):
        raise ValueError("The controlled-RY angle must be a finite real number")
    if 2 * (angle / 2) != angle:
        raise ValueError("The controlled-RY angle must survive halving exactly")
    circuit.ry(angle / 2, target)
    circuit.cx(control, target)
    circuit.ry(-angle / 2, target)
    circuit.cx(control, target)


def append_dirty_c3x(circuit: QuantumCircuit, controls: Sequence[int], target: int, scratch: int):
    """Exact three-control X, leaving a possibly entangled dirty scratch intact.

    The target toggle is r*c2 xor (r xor c0*c1)*c2 = c0*c1*c2.
    Four exact CCXs restore r and introduce no data/scratch phase.
    """
    if len(controls) != 3:
        raise ValueError("This helper requires exactly three controls")
    c0, c1, c2 = controls
    _wires(circuit, c0, c1, c2, target, scratch)
    append_exact_ccx(circuit, scratch, c2, target)
    append_exact_ccx(circuit, c0, c1, scratch)
    append_exact_ccx(circuit, scratch, c2, target)
    append_exact_ccx(circuit, c0, c1, scratch)


def _zero_prefix_ry(circuit, controls, target, angle, workspace):
    """Compute the negative prefix predicate, rotate, and return workspace to zero."""
    if len(controls) > 3 or len(workspace) != 2:
        raise ValueError("Prefix preparation is bounded to three controls and two clean scratch wires")
    _wires(circuit, *controls, target, *workspace)
    for q in controls:
        circuit.x(q)
    if not controls:
        circuit.ry(angle, target)
    elif len(controls) == 1:
        append_controlled_ry(circuit, controls[0], target, angle)
    elif len(controls) == 2:
        append_exact_ccx(circuit, controls[0], controls[1], workspace[0])
        append_controlled_ry(circuit, workspace[0], target, angle)
        append_exact_ccx(circuit, controls[0], controls[1], workspace[0])
    else:
        append_exact_ccx(circuit, controls[0], controls[1], workspace[0])
        append_exact_ccx(circuit, workspace[0], controls[2], workspace[1])
        append_controlled_ry(circuit, workspace[1], target, angle)
        append_exact_ccx(circuit, workspace[0], controls[2], workspace[1])
        append_exact_ccx(circuit, controls[0], controls[1], workspace[0])
    for q in reversed(controls):
        circuit.x(q)


def _append_preparation(circuit, clause, workspace):
    angles = []
    for index, target in enumerate(clause):
        # A remaining zero-prefix mass (m+1-index)/(m+1) emits exactly 1/(m+1).
        angle = 2 * math.asin(1 / math.sqrt(len(clause) + 1 - index))
        _zero_prefix_ry(circuit, clause[:index], target, angle, workspace)
        angles.append(angle)
    return angles


def decoder_table(matrix=DEFAULT_MATRIX) -> dict[int, int]:
    """Classical specification only: nonpromised syndromes map to zero."""
    if not isinstance(matrix, (tuple, list)):
        raise ValueError("Matrix must be a bounded row sequence")
    instance = validate_instance(matrix, [0] * len(matrix))
    table = {0: 0}
    for index, row in enumerate(instance["matrix"]):
        table[sum(bit << j for j, bit in enumerate(row))] = 1 << index
    return table


def decoder_extension(syndrome: int, clause: int, scratch: int = 0,
                      matrix=DEFAULT_MATRIX) -> tuple[int, int, int]:
    """The declared total reversible map (s,y,a) -> (s,y xor d(s),a)."""
    instance = validate_instance(matrix, [0] * len(matrix))
    for value, ceiling in ((syndrome, 1 << instance["n"]), (clause, 1 << instance["m"]), (scratch, 4)):
        if type(value) is not int or not 0 <= value < ceiling:
            raise ValueError("Decoder basis values exceed their register bounds")
    return syndrome, clause ^ decoder_table(matrix).get(syndrome, 0), scratch


def _append_decoder(circuit, instance, registers):
    controls = registers["assignment"]
    for target, row in zip(registers["clause"], instance["matrix"], strict=True):
        negative = [q for q, bit in zip(controls, row, strict=True) if not bit]
        for q in negative:
            circuit.x(q)
        if len(controls) == 1:
            circuit.cx(controls[0], target)
        elif len(controls) == 2:
            append_exact_ccx(circuit, controls[0], controls[1], target)
        else:
            append_dirty_c3x(circuit, controls, target, registers["workspace"][0])
        for q in reversed(negative):
            circuit.x(q)


def _check_materialized(circuit, limits):
    if circuit.num_qubits > limits.materialized_qubits or len(circuit.data) > limits.primitive_operations:
        raise ValueError("DQI circuit exceeds its materialization ceiling")
    if set(circuit.count_ops()) - PRIMITIVE_GATES - {"reset", "measure", "barrier"}:
        raise ValueError("DQI circuit contains an opaque or unlisted operation")


def build_preparation_circuit(m: int = 4, *, limits: DQILimits | None = None) -> QuantumCircuit:
    """Primitive preparation on m clause wires plus two initially clean scratch wires."""
    limits = _bounds(limits)
    if type(m) is not int or not 1 <= m <= 4:
        raise ValueError("Weight-zero/one preparation supports 1..4 clause wires")
    circuit = QuantumCircuit(m + 2, name="DQI weight zero or one")
    angles = _append_preparation(circuit, list(range(m)), [m, m + 1])
    _check_materialized(circuit, limits)
    circuit.metadata = dict(status=PROGRAM_STATUS, component="weight_zero_one_preparation",
        input_requirement="all clause and workspace wires initially zero", clause=list(range(m)),
        workspace=[m, m + 1], angles=angles, angle_hex=[angle.hex() for angle in angles],
        target_amplitudes="equal positive 1/sqrt(m+1) on vacuum and all single excitations",
        workspace_cleanup="zero on the specified input; executed checks and fixture scope belong to an external execution receipt")
    return circuit


def build_decoder_circuit(matrix=DEFAULT_MATRIX, *, limits: DQILimits | None = None) -> QuantumCircuit:
    """Explicit global XOR oracle, with no initialization or hidden decoder instruction."""
    limits = _bounds(limits)
    instance = validate_instance(matrix, [0] * len(matrix))
    registers = register_map(instance)
    circuit = QuantumCircuit(registers["total_qubits"], name="DQI reversible syndrome decoder")
    _append_decoder(circuit, instance, registers)
    _check_materialized(circuit, limits)
    circuit.metadata = dict(status=PROGRAM_STATUS, component="decoder", instance=instance,
        registers=registers, reversible_extension="(s,y,a) -> (s,y xor d(s),a)",
        workspace_requirement="arbitrary state; restored including spectator entanglement",
        decoder_table=decoder_table(matrix), correctness_scope="declared bounded XOR extension; executed checks and fixture scope belong to an external execution receipt")
    return circuit


def build_dqi_circuit(matrix=DEFAULT_MATRIX, signs=DEFAULT_SIGNS, *, ell: int = 1,
                      measure: bool = True, initialize: bool = True,
                      barriers: bool = True, limits: DQILimits | None = None) -> QuantumCircuit:
    """Materialize the complete bounded preparation/syndrome/decoder/H/readout workflow."""
    limits = _bounds(limits)
    if any(type(value) is not bool for value in (measure, initialize, barriers)):
        raise ValueError("Circuit options must be booleans")
    instance = validate_instance(matrix, signs, ell=ell)
    registers = register_map(instance)
    circuit = QuantumCircuit(registers["total_qubits"], instance["n"] if measure else 0, name="DQI ell one")
    stages = []
    def stage(name, action):
        start = len(circuit.data)
        result = action()
        stages.append(dict(name=name, start=start, stop=len(circuit.data)))
        if barriers:
            circuit.barrier(*range(circuit.num_qubits))
        return result
    if initialize:
        stage("input_reset", lambda: circuit.reset(range(circuit.num_qubits)))
    angles = stage("weight_zero_one_preparation", lambda: _append_preparation(circuit, registers["clause"], registers["workspace"]))
    def sign_phase():
        for q, bit in zip(registers["clause"], instance["signs"], strict=True):
            if bit:
                circuit.z(q)
    stage("clause_sign_phase", sign_phase)
    def syndrome():
        for q, row in zip(registers["clause"], instance["matrix"], strict=True):
            for target, bit in zip(registers["assignment"], row, strict=True):
                if bit:
                    circuit.cx(q, target)
    stage("syndrome_B_transpose_y", syndrome)
    stage("reversible_decoder_and_cleanup", lambda: _append_decoder(circuit, instance, registers))
    stage("assignment_hadamards", lambda: circuit.h(registers["assignment"]))
    if measure:
        stage("assignment_readout", lambda: circuit.measure(registers["assignment"], registers["classical"]))
    _check_materialized(circuit, limits)
    circuit.metadata = dict(status=PROGRAM_STATUS, instance=instance, registers=registers,
        stages=stages, preparation_angles=angles, preparation_angle_hex=[angle.hex() for angle in angles],
        input_requirement="all-zero data/workspace; explicit resets" if initialize else "all-zero input externally required",
        decoder_extension="(s,y,a) -> (s,y xor d(s),a), d(s)=0 off vacuum/single-row syndromes",
        output_order="integer bit j is assignment qubit m+j and classical bit j",
        source_reference=SOURCE_REFERENCE, upstream_code_adopted=False,
        physical_estimate_status="not implemented or executed", arbitrary_ry_synthesized=False)
    return circuit


def primitive_operations(circuit: QuantumCircuit) -> list:
    return [[item.operation.name, [circuit.find_bit(q).index for q in item.qubits],
             [circuit.find_bit(c).index for c in item.clbits], [float(p) for p in item.operation.params]]
            for item in circuit.data]


def logical_resources(circuit: QuantumCircuit, *, draws: int = 1) -> dict:
    """Actual primitive source ledger; rotations and all physical costs remain unresolved."""
    if type(draws) is not int or not 1 <= draws <= 1024:
        raise ValueError("The bounded source ledger supports 1..1024 complete draws")
    _check_materialized(circuit, _bounds(None))
    counts = dict(sorted(circuit.count_ops().items()))
    unitary = sum(value for name, value in counts.items() if name in PRIMITIVE_GATES)
    return dict(status="primitive_source_ledger_rotations_unsynthesized", logical_qubits=circuit.num_qubits,
        complete_draws=draws, primitive_events_per_draw=counts,
        primitive_events_all_draws={name: value * draws for name, value in counts.items()},
        source_unitary_gates_per_draw=unitary, source_initializations_per_draw=counts.get("reset", 0),
        source_individual_readouts_per_draw=counts.get("measure", 0),
        explicit_t_tdg_per_draw=counts.get("t", 0) + counts.get("tdg", 0),
        unsynthesized_ry_per_draw=counts.get("ry", 0), actual_rotation_synthesis_performed=False,
        native_physical_estimation_performed=False,
        exclusions=["arbitrary-RY Clifford+T synthesis and its approximation allocation",
                    "physical reset/readout/transport/controller/factory/decoder model",
                    "hardware input delivery, classical scoring and reporting time"],
        scaling_scope="bounded n<=3,m<=4,ell=1 truth-table predicate family; no efficient general decoder claim")


def xor_objective_values(matrix=DEFAULT_MATRIX, signs=DEFAULT_SIGNS) -> tuple[int, ...]:
    """Independent exhaustive constraint scoring, with no circuit or decoder calls."""
    instance = validate_instance(matrix, signs)
    values = []
    for assignment in range(1 << instance["n"]):
        satisfied = 0
        for row, target in zip(instance["matrix"], instance["signs"], strict=True):
            parity = sum(bit * ((assignment >> j) & 1) for j, bit in enumerate(row)) % 2
            satisfied += parity == target
        values.append(int(satisfied))
    return tuple(values)


def polynomial_reference(matrix=DEFAULT_MATRIX, signs=DEFAULT_SIGNS) -> dict:
    """Exact rational probabilities for equal vacuum/single amplitudes.

    Orthogonality of the distinct nonzero Walsh characters gives normalization
    2**n*(m+1).  The chosen degree-one polynomial is 1+S=1+2*f-m.
    This function calculates an analytic contract, not sampled results.
    """
    instance = validate_instance(matrix, signs)
    values = xor_objective_values(matrix, signs)
    denominator = (1 << instance["n"]) * (instance["m"] + 1)
    weights = tuple((1 + 2 * value - instance["m"]) ** 2 for value in values)
    if sum(weights) != denominator:
        raise ValueError("Distinct-row polynomial normalization failed")
    probabilities = tuple(Fraction(weight, denominator) for weight in weights)
    optimum = max(values)
    expectation = sum((probability * value for probability, value in zip(probabilities, values, strict=True)), Fraction())
    success = sum((p for p, value in zip(probabilities, values, strict=True) if value == optimum), Fraction())
    return dict(status="analytic_contract_not_a_sampled_or_executed_circuit_result",
        objective_values=values, polynomial_values=tuple(1 + 2 * value - instance["m"] for value in values),
        probability_fractions=tuple(str(value) for value in probabilities), denominator=denominator,
        expected_objective_fraction=str(expectation), optimum=optimum, optimal_draw_probability_fraction=str(success),
        distribution_scope="one ideal initialized DQI draw; no hardware, convergence or generic advantage guarantee")


def bounded_statevector(circuit: QuantumCircuit, *, limits: DQILimits | None = None):
    """Deferred exact ideal helper; deterministic all-zero resets are handled analytically."""
    limits = _bounds(limits)
    _check_materialized(circuit, limits)
    if circuit.num_qubits > limits.statevector_qubits or 1 << circuit.num_qubits > limits.statevector_amplitudes:
        raise ValueError("Statevector exceeds its independent simulation ceiling")
    from qiskit.quantum_info import Statevector
    coherent = QuantumCircuit(circuit.num_qubits)
    active, measured = False, False
    reset_wires = []
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        if name == "reset":
            if active or measured:
                raise ValueError("Only explicit all-zero input resets may be removed analytically")
            reset_wires.append(circuit.find_bit(item.qubits[0]).index)
            continue
        if name == "measure":
            measured = True
            continue
        if measured:
            raise ValueError("The bounded helper requires final readout only")
        active = True
        coherent.append(item.operation, [coherent.qubits[circuit.find_bit(q).index] for q in item.qubits])
    if reset_wires and reset_wires != list(range(circuit.num_qubits)):
        raise ValueError("Input reset order must initialize every materialized wire")
    return Statevector.from_instruction(coherent)


def bounded_decoder_operator(circuit: QuantumCircuit, *, limits: DQILimits | None = None):
    """Deferred dense check of the primitive decoder's total reversible extension."""
    limits = _bounds(limits)
    _check_materialized(circuit, limits)
    if (1 << circuit.num_qubits > limits.operator_dimension
            or 1 << circuit.num_qubits > limits.decoder_basis_checks
            or set(circuit.count_ops()) - PRIMITIVE_GATES):
        raise ValueError("Dense decoder verification exceeds its separate guard")
    from qiskit.quantum_info import Operator
    return Operator(circuit).data
