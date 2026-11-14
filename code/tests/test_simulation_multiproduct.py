"""Independent small quantum checks and source/resource boundary regressions.

This suite is part of the parent's frozen, bounded scientific pass. Its matrix
tests never exceed three system plus three access qubits; the larger source
test explicitly blocks all matrix/reference evaluation.
"""
from fractions import Fraction
import ast
import importlib.util
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator

from openquantum.simulation import multiproduct as mpf


ATOL = 1e-11


def _qasm_angle(text):
    """Only bounded literal arithmetic/pi in the generated test specimen."""
    if len(text) > 128:
        raise AssertionError("test angle exceeds parser bound")
    root = ast.parse(text, mode="eval")
    if len(list(ast.walk(root))) > 32:
        raise AssertionError("test angle expression exceeds parser bound")
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if b == 0:
                raise AssertionError("zero denominator in test QASM")
            return a / b
        raise AssertionError("unsupported test QASM expression")
    result = evaluate(root.body)
    if not math.isfinite(result):
        raise AssertionError("nonfinite test QASM angle")
    return result


def _primitive_qasm_unitary(text):
    """Independent local-matrix replay of the small unitary export only.

    It accepts no custom definitions, measurements, resets, controls, variables,
    loops or arbitrary Python expressions. This is a deliberately narrow test
    reader, not a general OpenQASM importer.
    """
    if len(text.encode()) > 65536:
        raise AssertionError("test QASM exceeds64KiB")
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("//")]
    if len(lines) > 1000:
        raise AssertionError("test QASM exceeds1000statements")
    if lines[:2] != ['OPENQASM 3.0;', 'include "stdgates.inc";']:
        raise AssertionError("unexpected test QASM header/include")
    n, result, pending_phase = None, None, 1 + 0j
    constants = {
        "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "z": np.diag([1, -1]).astype(complex),
        "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
        "t": np.diag([1, np.exp(1j * math.pi / 4)]),
        "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
    }
    for line in lines[2:]:
        declaration = re.fullmatch(r"qubit\[(\d+)\]\s+q;", line)
        if declaration:
            if n is not None:
                raise AssertionError("duplicate test quantum register")
            n = int(declaration.group(1))
            if not 1 <= n <= 4:
                raise AssertionError("test QASM matrix exceedsfourqubits")
            result = pending_phase * np.eye(1 << n, dtype=complex)
            continue
        phase = re.fullmatch(r"gphase\s*\(([^;]+)\);", line)
        if phase:
            scalar = np.exp(1j * _qasm_angle(phase.group(1)))
            if result is None:
                pending_phase *= scalar
            else:
                result *= scalar
            continue
        if result is None:
            raise AssertionError("gate precedes the test quantum register")
        controlled = re.fullmatch(r"(?:cx|CX)\s+q\[(\d+)\],\s*q\[(\d+)\];", line)
        if controlled:
            control, target = map(int, controlled.groups())
            if not (0 <= control < n and 0 <= target < n and control != target):
                raise AssertionError("invalid test controlled-X wires")
            for basis in range(1 << n):
                if basis & (1 << control) and not basis & (1 << target):
                    partner = basis ^ (1 << target)
                    result[[basis, partner], :] = result[[partner, basis], :]
            continue
        simple = re.fullmatch(r"(h|x|z|s|sdg|t|tdg)\s+q\[(\d+)\];", line)
        rotation = re.fullmatch(r"(rx|ry|rz)\s*\(([^;]+)\)\s+q\[(\d+)\];", line)
        if simple:
            name, target = simple.group(1), int(simple.group(2))
            matrix = constants[name]
        elif rotation:
            name, target = rotation.group(1), int(rotation.group(3))
            angle = _qasm_angle(rotation.group(2))
            c, s = math.cos(angle / 2), math.sin(angle / 2)
            matrix = (_ry(angle) if name == "ry" else
                      np.array([[c, -1j * s], [-1j * s, c]]) if name == "rx" else
                      np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)]))
        else:
            raise AssertionError(f"unsupported test QASM statement: {line}")
        if not 0 <= target < n:
            raise AssertionError("invalid test single-qubit wire")
        for basis in range(1 << n):
            if not basis & (1 << target):
                partner = basis ^ (1 << target)
                a, b = result[basis].copy(), result[partner].copy()
                result[basis] = matrix[0, 0] * a + matrix[0, 1] * b
                result[partner] = matrix[1, 0] * a + matrix[1, 1] * b
    if result is None:
        raise AssertionError("test QASM has no quantum register")
    return result


def _ry(angle):
    return np.array([[math.cos(angle / 2), -math.sin(angle / 2)],
                     [math.sin(angle / 2), math.cos(angle / 2)]], dtype=complex)


def _probability(matrix, input_column, output_slice):
    vector = matrix[output_slice, input_column]
    return float(np.vdot(vector, vector).real)


def test_seven_t_toffoli_preserves_all_basis_phases():
    circuit = QuantumCircuit(3)
    mpf._ccx7(circuit, 0, 1, 2)
    expected = np.eye(8, dtype=complex)
    expected[:, [3, 7]] = expected[:, [7, 3]]
    assert dict(circuit.count_ops()) == {"h": 2, "cx": 6, "tdg": 3, "t": 4}
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=ATOL, rtol=0)


def test_prepare_full_unitary_and_all_four_probabilities():
    circuit = QuantumCircuit(2)
    mpf._append_prepare(circuit, low=0, high=1)
    high = np.array([[math.sqrt(5 / 6), -math.sqrt(1 / 6)],
                     [math.sqrt(1 / 6), math.sqrt(5 / 6)]])
    low0 = np.array([[math.sqrt(4 / 5), -math.sqrt(1 / 5)],
                     [math.sqrt(1 / 5), math.sqrt(4 / 5)]])
    low1 = np.array([[1., -1.], [1., 1.]]) / math.sqrt(2)
    controlled = np.zeros((4, 4), dtype=complex)
    controlled[:2, :2], controlled[2:, 2:] = low0, low1
    expected = controlled @ np.kron(high, np.eye(2))
    actual = Operator(circuit).data
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    np.testing.assert_allclose(abs(actual[:, 0]) ** 2,
                               [2 / 3, 1 / 6, 1 / 12, 1 / 12], atol=ATOL, rtol=0)
    assert len(circuit.data) == 11
    assert circuit.num_qubits == 2


@pytest.mark.parametrize("control_state", [0, 1])
def test_primitive_controlled_ry_full_matrix(control_state):
    angle = 0.371
    circuit = QuantumCircuit(2)
    if control_state == 0:
        circuit.x(1)
    mpf._cry(circuit, angle, control=1, target=0)
    if control_state == 0:
        circuit.x(1)
    expected = np.eye(4, dtype=complex)
    selected = slice(2 * control_state, 2 * control_state + 2)
    expected[selected, selected] = _ry(angle)
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("label", [0, 1, 2, 3])
def test_selected_strang_has_correct_relative_phase_and_clean_work(label):
    n, d = 2, 4
    circuit = QuantumCircuit(n + 3)
    mpf._selected_strang(circuit, n, .5, 2, 1., .7, label,
                        low=n, high=n + 1, work=n + 2)
    actual = Operator(circuit).data
    branch = mpf.reference_operators(n, .5, 2)["coarse"]
    expected = np.eye(4 * d, dtype=complex)
    selected = slice(label * d, (label + 1) * d)
    expected[selected, selected] = branch
    np.testing.assert_allclose(actual[:4 * d, :4 * d], expected, atol=ATOL, rtol=0)
    np.testing.assert_allclose(actual[4 * d:, :4 * d], 0, atol=ATOL, rtol=0)


def test_zero_selector_reflection_has_only_one_negative_label():
    circuit = QuantumCircuit(2)
    mpf._reflect_zero(circuit, 0, 1)
    np.testing.assert_allclose(Operator(circuit).data,
                               np.diag([-1, 1, 1, 1]), atol=ATOL, rtol=0)


@pytest.mark.parametrize("method", ["lcu", "oaa"])
def test_exported_primitive_qasm_preserves_full_operator_and_global_minus(method):
    path = Path(__file__).resolve().parents[1] / "scripts/generate_simulation_multiproduct.py"
    name = "multiproduct_generator_test"
    specification = importlib.util.spec_from_file_location(name, path)
    generator = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(name)
    sys.modules[name] = generator
    try:
        specification.loader.exec_module(generator)
        circuit = mpf.build_evolution(1, .5, 1, method)
        qasm, metadata = generator.qasm_text(circuit)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    if method == "oaa":
        assert "gphase(" in qasm or "gphase (" in qasm
        assert len(re.findall(r"^gphase\s*\(", qasm, flags=re.MULTILINE)) == 1
    assert metadata["global_phase_radians"] == pytest.approx(float(circuit.global_phase), abs=ATOL)
    assert isinstance(metadata["explicit_gphase_inserted"], bool)
    assert "no controlled-evolution export has been checked" in metadata["convention"]
    replayed = _primitive_qasm_unitary(qasm)
    np.testing.assert_allclose(replayed, Operator(circuit).data, atol=ATOL, rtol=0)


def test_test_qasm_reader_handles_phases_without_a_circuit_library():
    text = ('OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\n'
            'gphase(pi);\nrz(pi/2) q[0];\n')
    expected = -np.diag([np.exp(-1j * math.pi / 4), np.exp(1j * math.pi / 4)])
    np.testing.assert_allclose(_primitive_qasm_unitary(text), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("statement", ["qubit[5] q;", "qubit[1] q;\nreset q[0];",
                                       "qubit[1] q;\nry(sin(pi)) q[0];"])
def test_test_qasm_reader_rejects_unbounded_or_unsupported_specimens(statement):
    text = 'OPENQASM 3.0;\ninclude "stdgates.inc";\n' + statement + '\n'
    with pytest.raises(AssertionError):
        _primitive_qasm_unitary(text)


@pytest.mark.parametrize("n,time,steps", [(1, 1., 1), (2, 1., 1), (2, 1., 2), (3, .5, 2)])
def test_complete_clean_domains_and_accepted_blocks(n, time, steps):
    refs = mpf.reference_operators(n, time, steps)
    d = 1 << n
    B = Operator(mpf.build_evolution(n, time, steps, "lcu")).data
    G = Operator(mpf.build_evolution(n, time, steps, "oaa")).data
    # These comparisons inspect every selector/system input with clean work,
    # rather than only one prepared state or measurement distribution.
    np.testing.assert_allclose(B[:4 * d, :4 * d], refs["B_clean"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(G[:4 * d, :4 * d], refs["G_clean"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(B[4 * d:, :4 * d], 0, atol=ATOL, rtol=0)
    np.testing.assert_allclose(G[4 * d:, :4 * d], 0, atol=ATOL, rtol=0)
    np.testing.assert_allclose(B[:d, :d], refs["A"] / 2, atol=ATOL, rtol=0)
    expected_F = (3 * refs["A"] - refs["A"] @ refs["A"].conj().T @ refs["A"]) / 2
    np.testing.assert_allclose(G[:d, :d], expected_F, atol=ATOL, rtol=0)
    assert math.isclose(float(mpf.build_evolution(n, time, steps, "oaa").global_phase),
                        math.pi, abs_tol=ATOL)
    assert np.linalg.norm(expected_F + G[:d, :d], 2) > .1  # catches loss of minus sign


@pytest.mark.parametrize("amplified,p,coefficient", [
    (False, Fraction(25, 36), Fraction(5, 6)),
    (True, Fraction(25, 729), Fraction(5, 27)),
])
def test_opposite_identity_counterexample_preserves_conditional_state(amplified, p, coefficient):
    circuit = mpf.opposite_identity_fixture(amplified)
    actual = Operator(circuit).data
    block = actual[:2, :2]
    np.testing.assert_allclose(block, float(coefficient) * np.eye(2), atol=ATOL, rtol=0)
    for basis in (0, 1):
        assert math.isclose(_probability(actual, basis, slice(0, 2)), float(p), abs_tol=ATOL)
        normalized = block[:, basis] / math.sqrt(float(p))
        np.testing.assert_allclose(normalized, np.eye(2)[:, basis], atol=ATOL, rtol=0)
    # A scalar accepted block can preserve conditional states while failing
    # the unscaled coherent-operator contract.
    assert np.linalg.norm(block - np.eye(2), 2) > .1


@pytest.mark.parametrize("n,J,h,time", [(1, 1., .7, 1.), (2, 1., 0., .5),
                                      (2, 0., .7, .5), (2, 1., .7, 0.)])
def test_commuting_or_zero_evolution_has_exact_multiproduct(n, J, h, time):
    refs = mpf.reference_operators(n, time, 2, J, h)
    np.testing.assert_allclose(refs["A"], refs["exact"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(refs["F"], refs["exact"], atol=ATOL, rtol=0)
    G = Operator(mpf.build_evolution(n, time, 2, "oaa", J, h)).data
    d = 1 << n
    np.testing.assert_allclose(G[:d, :d], refs["exact"], atol=ATOL, rtol=0)
    assert math.isclose(_probability(G, 0, slice(0, d)), 1., abs_tol=ATOL)


def test_inactive_one_spin_J_does_not_create_a_rotation():
    usual = mpf.build_evolution(1, 4., 1, "lcu", J=1., h=.7)
    unused_large_J = mpf.build_evolution(1, 4., 1, "lcu", J=1e308, h=.7)
    np.testing.assert_allclose(Operator(unused_large_J).data,
                               Operator(usual).data, atol=ATOL, rtol=0)


@pytest.mark.parametrize("basis", [0, 1])
def test_complete_channel_attempt_resets_access_and_preserves_full_reduced_state(basis):
    n, d, steps = 2, 4, 1
    attempt = mpf.build_attempt(n, 1., steps, "oaa-channel", initial_basis=basis)
    actual = DensityMatrix.from_instruction(attempt.remove_final_measurements(inplace=False)).data
    ref_G = mpf.reference_operators(n, 1., steps)["G_clean"]
    target = sum(np.outer(ref_G[label * d:(label + 1) * d, basis],
                          ref_G[label * d:(label + 1) * d, basis].conj()) for label in range(4))
    expected = np.zeros((8 * d, 8 * d), dtype=complex)
    expected[:d, :d] = target
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    assert attempt.num_clbits == n
    assert dict(attempt.count_ops())["reset"] == n + 6
    assert dict(attempt.count_ops())["measure"] == n
    assert attempt.metadata["acceptance"].startswith("all outcomes")


def test_unconditional_channel_retains_coherences_for_entangled_system_input():
    n, d = 2, 4
    psi = np.array([1, 0, 0, 1j], dtype=complex) / math.sqrt(2)
    joint = np.zeros(8 * d, dtype=complex)
    joint[:d] = psi
    circuit = mpf.build_evolution(n, .5, 1, "oaa-channel")
    for q in range(n, n + 3):
        circuit.reset(q)
    actual = DensityMatrix(np.outer(joint, joint.conj())).evolve(circuit).data
    ref_G = mpf.reference_operators(n, .5, 1)["G_clean"]
    target = np.zeros((d, d), dtype=complex)
    for label in range(4):
        vector = ref_G[label * d:(label + 1) * d, :d] @ psi
        target += np.outer(vector, vector.conj())
    expected = np.zeros((8 * d, 8 * d), dtype=complex)
    expected[:d, :d] = target
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    assert abs(target[0, 3].imag) > .05


def test_independent_diagnostics_match_conservative_bounds_without_accuracy_claims():
    result = mpf.compare(2, 1., 2, basis=1)
    bounds = result["analytic_ideal_bounds"]
    assert max(result["implementation_errors"].values()) < ATOL
    assert result["A_operator_error"] <= float(Fraction(bounds["mpf_operator_error_upper"])) + ATOL
    assert result["F_operator_error"] <= float(Fraction(bounds["oaa_operator_error_upper"])) + ATOL
    assert result["lcu_min_acceptance_numeric"] >= .25 - ATOL
    assert result["oaa_min_acceptance_numeric"] >= float(Fraction(bounds["oaa_ideal_acceptance_floor"])) - ATOL
    assert result["classical_weighted_Z"] == pytest.approx(
        (4 * result["classical_fine_Z"] - result["classical_coarse_Z"]) / 3, abs=ATOL)
    assert result["unconditional_oaa_trace"] == pytest.approx(1., abs=ATOL)
    assert result["synthesis_performed"] is False
    assert result["physical_estimation_performed"] is False
    assert result["projected_trial_population_executed"] is False


def test_exact_source_counts_follow_materialized_architecture():
    expected_B = {"h": 20, "x": 10, "z": 1, "t": 16, "tdg": 12,
                  "cx": 68, "ry": 10, "rz": 24}
    expected_G = {"h": 64, "x": 38, "z": 3, "t": 44, "tdg": 40,
                  "cx": 206, "ry": 30, "rz": 72}
    for method, expected in (("lcu", expected_B), ("oaa", expected_G), ("oaa-channel", expected_G)):
        circuit = mpf.build_evolution(2, 1., 1, method)
        assert dict(circuit.count_ops()) == expected
        counted = mpf.scalar_counts(2, 1, method)
        assert counted["unitary_gate_counts"] == expected
        assert counted["unitary_operations"] == sum(expected.values())
        assert counted["branch_steps_per_B"] == 3
        assert counted["B_calls"] == (1 if method == "lcu" else 3)
        assert set(circuit.count_ops()) <= mpf.PRIMITIVES


@pytest.mark.parametrize("method", mpf.METHODS)
def test_full_experiments_include_initialization_and_every_readout(method):
    circuit = mpf.build_attempt(2, .5, 1, method, initial_basis=1)
    counts = dict(circuit.count_ops())
    assert counts["measure"] == (2 if method in ("fine", "coarse", "oaa-channel") else 5)
    assert counts["reset"] == (8 if method == "oaa-channel" else circuit.num_qubits)
    assert circuit.metadata["initial_basis"] == 1
    assert circuit.metadata["initial_reset_count"] == circuit.num_qubits
    assert circuit.metadata["source_operations"] == len(circuit.data)
    assert sum(circuit.metadata["source_gate_counts"].values()) == len(circuit.data)


def test_large_source_construction_is_separate_from_matrix_validation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("large source attempted matrix evaluation")
    monkeypatch.setattr(mpf, "Operator", forbidden)
    monkeypatch.setattr(mpf, "reference_operators", forbidden)
    monkeypatch.setattr(mpf, "compare", forbidden)
    circuit = mpf.build_attempt(32, 1., 1, "oaa")
    assert circuit.num_qubits == 35
    assert len(circuit.data) == 5427
    assert circuit.count_ops()["t"] + circuit.count_ops()["tdg"] == 84
    assert circuit.metadata["work_promise"].startswith("work starts in zero")


def test_scalar_scaling_does_not_allocate_terms_circuits_or_matrices(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("scalar count allocated scientific state")
    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    monkeypatch.setattr(mpf, "Operator", forbidden)
    monkeypatch.setattr(mpf, "reference_operators", forbidden)
    n, r = 100_000_000, 32
    result = mpf.scalar_counts(n, r, "oaa")
    assert result["logical_qubits"] == n + 3
    assert result["unitary_operations"] == 54 * r * (3 * n - 2) + 281
    assert result["physical_resource_estimate"] is None
    assert result["scalar_only"]


@pytest.mark.parametrize("kwargs", [{"n": True}, {"n": 0}, {"n": 62}, {"steps": True},
                                   {"steps": 0}, {"steps": 4097}, {"time": float("nan")},
                                   {"time": float("inf")}, {"J": True}, {"h": float("inf")},
                                   {"method": "unknown"}])
def test_invalid_source_inputs_reject_before_circuit_allocation(monkeypatch, kwargs):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid request allocated a circuit")
    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError):
        mpf.build_evolution(**kwargs)


def test_complete_attempt_rejects_huge_n_before_basis_shift_or_circuit_allocation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("huge source request allocated a circuit")
    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError):
        mpf.build_attempt(n=100_000_000)


def test_operation_preflight_rejects_before_circuit_allocation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("over-budget source request allocated a circuit")
    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError, match="preflight"):
        mpf.build_evolution(61, 1., 4096, "oaa")


def test_non_halvable_controlled_rotation_rejects():
    smallest = float.fromhex("0x0.0000000000001p-1022")
    with pytest.raises(ValueError, match="underflow|half"):
        mpf.build_evolution(2, smallest, 1, "lcu")


def test_nonminimal_odd_subnormal_controlled_rotation_rejects_before_allocation(monkeypatch):
    odd_subnormal = float.fromhex("0x0.0000000000003p-1022")
    rounded_half = odd_subnormal / 2
    assert rounded_half != 0
    assert 2 * rounded_half != odd_subnormal

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inexact controlled angle allocated a circuit")

    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError, match="exact half angle"):
        mpf.build_evolution(n=2, time=1., steps=1, method="lcu", J=odd_subnormal, h=0.)


def test_dense_validation_is_bounded_independently():
    with pytest.raises(ValueError, match="three system"):
        mpf.reference_operators(4)


def test_exact_classical_sampling_conditioning_and_log_bound():
    plan = mpf.sampling_plan()
    assert plan["single_accepted_samples"] == 14_000
    assert plan["classical_total_target"] == 38_889
    assert plan["classical_fine_samples"] == 31_112
    assert plan["classical_coarse_samples"] == 7778
    assert plan["classical_total_samples"] == 38_890
    assert plan["coefficient_conditioning_square"] == "25/9"
    assert Fraction(plan["classical_sum_c2_over_N"]) <= Fraction(3, 100) ** 2 / (2 * Fraction(plan["log_upper"]))
    assert not plan["population_executed"]
    exp_partial = sum(Fraction(7, 10) ** k / math.factorial(k) for k in range(5))
    assert exp_partial > 2
    assert mpf.log_upper(Fraction(500)) == Fraction(63, 10)


def test_classical_weighted_estimator_retains_values_outside_state_observable_range():
    assert mpf.combine_classical_means(1., -1.) == pytest.approx(5 / 3, abs=ATOL)
    assert mpf.combine_classical_means(-1., 1.) == pytest.approx(-5 / 3, abs=ATOL)
    assert mpf.combine_classical_means(.5, .5) == pytest.approx(.5, abs=ATOL)


@pytest.mark.parametrize("fine,coarse", [(float("nan"), 0.), (1.01, 0.), (0., -1.01),
                                       (True, 0.), (0., float("inf"))])
def test_classical_group_mean_validation(fine, coarse):
    with pytest.raises(ValueError):
        mpf.combine_classical_means(fine, coarse)


@pytest.mark.parametrize("value", [Fraction(1), Fraction(3, 2), Fraction(2), Fraction(500)])
def test_rational_log_upper_dominates_independent_math_log(value):
    assert float(mpf.log_upper(value)) >= math.log(float(value))


def test_retry_cap_is_minimal_and_counts_complete_preparations():
    p, samples, delta = Fraction(1, 4), 3, Fraction(1, 1000)
    plan = mpf.retry_plan(p, samples, delta)
    r = plan["retry_cap_per_accepted_sample"]
    assert r == 28
    assert samples * (1 - p) ** r <= delta
    assert samples * (1 - p) ** (r - 1) > delta
    assert plan["capped_complete_attempts"] == r * samples
    assert plan["fresh_repreparable_input_required"]
    assert not plan["attempts_executed"]


def test_probability_one_retry_has_one_complete_attempt_per_sample():
    plan = mpf.retry_plan(Fraction(1), 3)
    assert plan["retry_cap_per_accepted_sample"] == 1
    assert plan["capped_complete_attempts"] == 3


@pytest.mark.parametrize("call", [
    lambda: mpf.log_upper(Fraction(1 << 129)),
    lambda: mpf.log_upper(Fraction(1, 2)),
    lambda: mpf.retry_plan(Fraction(1, 1 << 129), 1),
    lambda: mpf.retry_plan(Fraction(1, 2), True),
    lambda: mpf.retry_plan(Fraction(1, 2), 1, Fraction(1, 1 << 65)),
    lambda: mpf.sampling_plan(.03, Fraction(1, 250)),
    lambda: mpf.sampling_plan(Fraction(1, 1 << 64), Fraction(1, 250)),
    lambda: mpf.sampling_plan(Fraction(1, 1 << 63), Fraction(1, 250)),
])
def test_resource_rational_and_population_guards(call):
    with pytest.raises(ValueError):
        call()


def test_exact_analytic_floors_and_commuting_bounds():
    commuting = mpf.analytic_bounds(1, 1., 2)
    assert Fraction(commuting["strang_C_upper"]) == 0
    assert Fraction(commuting["mpf_operator_error_upper"]) == 0
    assert Fraction(commuting["oaa_ideal_acceptance_floor"]) == 1
    difficult = mpf.analytic_bounds(3, 1., 1)
    assert Fraction(difficult["branch_difference_upper"]) == 2
    assert Fraction(difficult["oaa_ideal_acceptance_floor"]) == Fraction(25, 729)
    assert Fraction(difficult["lcu_ideal_acceptance_floor"]) == Fraction(1, 4)
    assert not difficult["realization_bridge_verified"]


@pytest.mark.parametrize("method", mpf.METHODS)
def test_resource_projection_counts_complete_workload_and_retains_exclusions(method):
    plan = mpf.resource_proposal(2, 1., 32, method)
    source = plan["source_counts"]
    preparations = plan["complete_preparations_capped"]
    for gate, count in source["unitary_gate_counts"].items():
        assert plan["complete_gate_counts_capped"][gate] == count * preparations
    assert plan["complete_gate_counts_capped"]["measure"] >= 2 * preparations
    assert plan["complete_gate_counts_capped"]["reset"] >= source["logical_qubits"] * preparations
    if method in ("lcu", "oaa"):
        assert plan["retry"]["capped_complete_attempts"] == preparations
        assert plan["retry"]["fresh_repreparable_input_required"]
    else:
        assert plan["retry"] is None
    assert plan["physical_qubits"] is None
    assert plan["physical_runtime"] is None
    assert not plan["synthesis_performed"]
    assert not plan["total_accuracy_verified"]
    assert not plan["realization_bridge_verified"]
    assert any("classical" in item for item in plan["exclusions"])


@pytest.mark.parametrize("n,method", [(1, "lcu"), (1, "oaa"), (2, "lcu"), (2, "oaa")])
def test_state_preparation_measures_only_three_access_bits_in_the_declared_order(n, method):
    state = mpf.build_state_preparation(n, .5, 1, method)
    assert state.num_clbits == 3
    readouts = [(state.find_bit(instruction.qubits[0]).index,
                 state.find_bit(instruction.clbits[0]).index)
                for instruction in state.data if instruction.operation.name == "measure"]
    assert readouts == [(n, 0), (n + 1, 1), (n + 2, 2)]
    assert all(q >= n for q, _bit in readouts)
    assert state.metadata["measured_wires"] == [n, n + 1, n + 2]
    assert not state.metadata["system_readout"]
    assert state.metadata["initial_reset_count"] == n + 3
    assert state.metadata["final_discard_reset_count"] == 0
    assert state.count_ops()["reset"] == n + 3
    assert state.count_ops()["measure"] == 3
    assert "unmeasured systemstate" in state.metadata["output_contract"]


@pytest.mark.parametrize("n,method", [(1, "lcu"), (1, "oaa"), (1, "oaa-channel"),
                                      (2, "lcu"), (2, "oaa"), (2, "oaa-channel")])
def test_state_preparation_density_matches_independent_conditional_or_channel_reference(n, method):
    basis, d = 1, 1 << n
    state = mpf.build_state_preparation(n, .5, 1, method, initial_basis=basis)
    stripped = state.remove_final_measurements(inplace=False)
    actual = DensityMatrix.from_instruction(stripped).data
    references = mpf.reference_operators(n, .5, 1)
    if method == "oaa-channel":
        assert state.num_clbits == 0
        assert not any(instruction.operation.name == "measure" for instruction in state.data)
        assert state.metadata["measured_wires"] == []
        assert state.metadata["final_discard_reset_count"] == 3
        assert state.count_ops()["reset"] == n + 6
        expected_system = np.zeros((d, d), dtype=complex)
        for label in range(4):
            vector = references["G_clean"][label * d:(label + 1) * d, basis]
            expected_system += np.outer(vector, vector.conj())
        expected_joint = np.zeros((8 * d, 8 * d), dtype=complex)
        expected_joint[:d, :d] = expected_system
        np.testing.assert_allclose(actual, expected_joint, atol=ATOL, rtol=0)
    else:
        # Stripping the final ancilla measurements exposes the actual joint
        # preparation. Its zero-access block gives the accepted system state.
        accepted = actual[:d, :d]
        probability = float(np.trace(accepted).real)
        reference_block = references["A"] / 2 if method == "lcu" else references["F"]
        vector = reference_block[:, basis]
        expected_probability = float(np.vdot(vector, vector).real)
        expected_system = np.outer(vector, vector.conj()) / expected_probability
        assert probability == pytest.approx(expected_probability, abs=ATOL)
        np.testing.assert_allclose(accepted / probability, expected_system, atol=ATOL, rtol=0)
    assert np.trace(actual).real == pytest.approx(1., abs=ATOL)
    assert not state.metadata["system_readout"]
    assert state.metadata["initial_basis"] == basis


@pytest.mark.parametrize("system_readout", [0, 1, None, "false", np.bool_(True)])
def test_system_readout_flag_rejects_non_bool_before_circuit_allocation(monkeypatch, system_readout):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid readout flag allocated a circuit")
    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError, match="system_readout"):
        mpf.build_attempt(2, .5, 1, "lcu", system_readout=system_readout)


@pytest.mark.parametrize("method", ["fine", "coarse", "unknown"])
def test_state_preparation_rejects_classical_or_unknown_methods_before_allocation(monkeypatch, method):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid state method allocated a circuit")
    monkeypatch.setattr(mpf, "QuantumCircuit", forbidden)
    with pytest.raises(ValueError, match="coherent"):
        mpf.build_state_preparation(2, .5, 1, method)


def test_default_system_readout_is_identical_to_explicit_true():
    implicit = mpf.build_attempt(2, .5, 1, "lcu")
    explicit = mpf.build_attempt(2, .5, 1, "lcu", system_readout=True)
    def signature(circuit):
        return [(instruction.operation.name, tuple(instruction.operation.params),
                 tuple(circuit.find_bit(q).index for q in instruction.qubits),
                 tuple(circuit.find_bit(c).index for c in instruction.clbits))
                for instruction in circuit.data]
    assert signature(implicit) == signature(explicit)
    assert implicit.num_clbits == explicit.num_clbits == 5
    assert dict(implicit.count_ops()) == dict(explicit.count_ops())
    assert implicit.metadata == explicit.metadata


def test_classical_groups_share_one_weighted_task_with_additive_physical_allocations(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("allocation ledger attempted scientific state construction")

    for name in ("QuantumCircuit", "Operator", "reference_operators", "compare"):
        monkeypatch.setattr(mpf, name, forbidden)
    fine = mpf.resource_proposal(2, 1., 32, "fine")
    coarse = mpf.resource_proposal(2, 1., 32, "coarse")
    assert fine["sampling"] == coarse["sampling"]
    assert (fine["complete_preparations_capped"] + coarse["complete_preparations_capped"]
            == fine["sampling"]["classical_total_samples"])

    physical_whole = Fraction(1, 200)
    fine_share = Fraction(fine["physical_failure_budget_share"])
    coarse_share = Fraction(coarse["physical_failure_budget_share"])
    assert (fine_share, coarse_share) == (Fraction(4, 5), Fraction(1, 5))
    assert fine_share + coarse_share == 1
    fine_physical = Fraction(fine["physical_failure_request_this_group"])
    coarse_physical = Fraction(coarse["physical_failure_request_this_group"])
    assert (fine_physical, coarse_physical) == (Fraction(1, 250), Fraction(1, 1000))
    assert fine_physical == fine_share * physical_whole
    assert coarse_physical == coarse_share * physical_whole
    assert fine_physical + coarse_physical == physical_whole

    fine_bias = Fraction(fine["individual_group_realization_Z_bias_request"])
    coarse_bias = Fraction(coarse["individual_group_realization_Z_bias_request"])
    assert fine_bias == coarse_bias == Fraction(3, 500)
    weighted_bias = Fraction(4, 3) * fine_bias + Fraction(1, 3) * coarse_bias
    assert weighted_bias == Fraction(1, 100)
    shared_requests = {
        "logical_realization_Z_bias_request": weighted_bias,
        "ideal_algorithm_Z_bias_allowance": Fraction(1, 100),
        "statistical_Z_error_request": Fraction(3, 100),
        "statistical_failure_request": Fraction(1, 250),
        "physical_failure_request_whole_workload": physical_whole,
    }
    for group in (fine, coarse):
        for field, expected in shared_requests.items():
            assert Fraction(group[field]) == expected
        assert Fraction(group["full_attempt_logical_operator_error_request"]) == Fraction(3, 1000)
        assert "combined (4*fine-coarse)/3 weighted task" in group["accuracy_contract_scope"]
        assert "neither group alone owns" in group["accuracy_contract_scope"]
        assert "single weighted task" in group["physical_failure_budget_scope"]
        assert group["output"] == "one group in weighted classical Zmean"
        assert group["total_accuracy_verified"] is False
        assert group["realization_bridge_verified"] is False
    assert fine["ideal_algorithm_Z_bias_upper"] == coarse["ideal_algorithm_Z_bias_upper"]
    assert fine["accuracy_contract_scope"] == coarse["accuracy_contract_scope"]


@pytest.mark.parametrize("changed_companion", ["requirements-lock.txt", "LICENSE"])
def test_companion_mutation_prevents_manifest_without_scientific_execution(tmp_path, changed_companion):
    """Replay only export functions with fixture sources and no circuit runtime.

    This tests late companion rejection, not generated scientific content.
    Parsing selected function definitions avoids the generator's module-loading
    code; every state/count/sampling input below is an inert fixture.
    """
    import hashlib
    import json
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "scripts/generate_simulation_multiproduct.py"
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"generate", "_write", "_json"}]
    assert {node.name for node in definitions} == {"generate", "_write", "_json"}

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    originals = {"requirements-lock.txt": b"fixture-lock-before\n", "LICENSE": b"fixture-license-before\n"}
    for name, raw in originals.items():
        (source_dir / name).write_bytes(raw)
    sources = {"fixture-source.py": b"# inert fixture source\n"}
    provenance = {"source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()}}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("companion guard test attempted scientific execution")

    fixture_runtime = SimpleNamespace(
        METHODS=(), MAX_REFERENCE_SYSTEM_QUBITS=3, COEFFICIENTS=(), PROBABILITIES=(),
        preflight=forbidden, compare=forbidden, build_attempt=forbidden,
        build_state_preparation=lambda *_args, **_kwargs: SimpleNamespace(metadata={"fixture": True}),
        scalar_counts=lambda *_args, **_kwargs: {"fixture": True},
    )
    namespace = {
        "Path": Path, "ROOT": source_dir, "LOCAL": source_dir / "absent-multiproduct.py",
        "mpf": fixture_runtime, "time": SimpleNamespace(monotonic=lambda: 0.),
        "np": SimpleNamespace(random=SimpleNamespace(default_rng=lambda _seed: object())),
        "_source_record": lambda: (sources, provenance),
        "qasm_text": lambda _circuit: ("// inert export fixture\n", {"fixture": True}),
        "json": json, "hashlib": hashlib,
        "MAX_FILE_BYTES": 4 * 1024 * 1024, "MAX_EXPORT_BYTES": 32 * 1024 * 1024,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    original_write = namespace["_write"]
    mutation = b"fixture-companion-changed-during-generation\n"

    def write_then_mutate(output, filename, raw, inventory):
        original_write(output, filename, raw, inventory)
        if filename == "README.md":
            (source_dir / changed_companion).write_bytes(mutation)

    namespace["_write"] = write_then_mutate
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="lock/license companions changed during generation"):
        namespace["generate"](output, n=1, steps=1, no_sim=True, figures=False)
    assert (source_dir / changed_companion).read_bytes() == mutation
    assert (output / "requirements-lock.txt").read_bytes() == originals["requirements-lock.txt"]
    assert (output / "LICENSE.txt").read_bytes() == originals["LICENSE"]
    assert not (output / "manifest.json").exists()
