"""Independent small references and explicit access-only scope contracts."""
import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from qiskit import QuantumCircuit

from openquantum.simulation import detailed_balance_access as dba

ONE = ((1.0, "Z"),)
TWO = ((.7, "ZI"), (.4, "XX"), (.3, "IX"), (.19, "YZ"), (.23, "IZ"))
PROPOSALS = ("XI", "ZI", "IX", "IZ")


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("beta", [.5, 1, 2])
def test_signed_mesh_and_physical_mass(bits, beta):
    mesh = dba.gaussian_mesh(bits, beta)
    size = 1 << bits
    assert set(mesh["signed_indices"]) == set(range(-size // 2, size // 2))
    assert mesh["dt"] * mesh["dw"] == pytest.approx(2 * math.pi / size)
    f = np.exp(-(mesh["times"] / beta) ** 2) * math.sqrt(math.sqrt(2 / math.pi) / beta)
    np.testing.assert_allclose(mesh["amplitudes"], f / np.linalg.norm(f), atol=1e-15)
    assert mesh["eta"] == pytest.approx(mesh["dt"] * float(f @ f), abs=1e-15)
    phases = np.exp(-1j * np.outer(mesh["frequencies"], mesh["times"]))
    ordinary = np.exp(-2j * math.pi * np.outer(np.arange(size), np.arange(size)) / size)
    np.testing.assert_allclose(phases, ordinary, atol=1e-14)


@pytest.mark.parametrize("weights", [[.3, .7], [.01, .09, .2, .7],
                                     [0, .05, .1, .15, .2, .25, .1, .15],
                                     [0, 0, .7, .3], [1, 0, 0, 0]])
def test_nonpalindromic_gaussian_loader_and_zero_prefix(weights):
    amplitudes = np.sqrt(weights)
    actual = Statevector.from_instruction(dba.positive_prepare(amplitudes)).data
    np.testing.assert_allclose(actual, amplitudes, atol=1e-12)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_primitive_dft_against_explicit_complex_matrix(bits):
    size = 1 << bits
    expected = np.exp(-2j * math.pi * np.outer(np.arange(size), np.arange(size)) / size) / math.sqrt(size)
    actual = Operator(dba.negative_dft(bits)).data
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert set(dba.negative_dft(bits).count_ops()) <= dba.PRIMITIVES


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_one_spin_exact_conjugation_and_zero_correction(bits):
    result = dba.access_check(ONE, ("X", "Z"), clock_bits=bits)
    assert result["actual_vs_finite_trotter_max_entry_error"] < 1e-11
    assert result["actual_vs_exact_finite_target_frobenius_error"] < 1e-11
    assert result["isometry_max_entry_error"] < 1e-11
    assert result["finite_diagnostics"]["correction_norm"] < 1e-12
    assert result["continuum_reference"]["gibbs_stationarity_residual"] < 1e-12
    assert result["metadata"]["eta"] != 1
    assert not result["metadata"]["sampler_implemented"]


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_two_spin_complex_access_and_nonzero_correction(steps):
    result = dba.access_check(TWO, PROPOSALS, clock_bits=2, product_steps=steps)
    assert result["actual_vs_finite_trotter_max_entry_error"] < 1e-11
    assert result["isometry_max_entry_error"] < 1e-11
    assert result["finite_diagnostics"]["minimum_decay_eigenvalue"] >= -1e-12
    assert result["finite_diagnostics"]["correction_hermiticity_residual"] < 1e-12
    assert result["finite_diagnostics"]["decay_commutator"] > 1e-5
    assert result["finite_diagnostics"]["gibbs_stationarity_residual"] > 1e-5
    assert result["continuum_reference"]["gibbs_stationarity_residual"] < 1e-12


def test_product_formula_convergence_is_separate_from_mesh_error():
    exact = dba.reference_blocks(TWO, PROPOSALS, clock_bits=2)
    errors = [np.linalg.norm(dba.reference_blocks(TWO, PROPOSALS, clock_bits=2,
                                                 product_steps=r) - exact) for r in (1, 2, 4)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] > 0


def test_reordered_hamiltonian_has_its_own_actual_product_reference():
    reordered = tuple(reversed(TWO))
    actual = dba.circuit_blocks(dba.build_gaussian_access(reordered, PROPOSALS, clock_bits=2))
    expected = dba.reference_blocks(reordered, PROPOSALS, clock_bits=2, product_steps=1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(dba.reference_blocks(TWO, PROPOSALS, clock_bits=2),
                               dba.reference_blocks(reordered, PROPOSALS, clock_bits=2), atol=1e-12)


@pytest.mark.parametrize("proposals", [("Y",), ("Y", "I"), ("Y", "X", "Z", "I")])
def test_actual_select_phase_proposals_and_inverse(proposals):
    terms = ((.4, "Z"), (.3, "Y"), (.1, "I"))
    circuit = dba.build_gaussian_access(terms, proposals, clock_bits=2, product_steps=2)
    actual = dba.circuit_blocks(circuit)
    expected = dba.reference_blocks(terms, proposals, clock_bits=2, product_steps=2)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    inverse_pair = circuit.compose(circuit.inverse())
    initial = Statevector.from_int(1, 1 << circuit.num_qubits)
    np.testing.assert_allclose(initial.evolve(inverse_pair).data, initial.data, atol=1e-12)
    assert set(circuit.count_ops()) <= dba.PRIMITIVES
    assert "unitary" not in qasm3.dumps(circuit)


def test_identity_hamiltonian_clock_phase_is_explicit_and_cancels():
    base = dba.circuit_blocks(dba.build_gaussian_access(ONE, ("X", "Z"), clock_bits=2))
    shifted = dba.circuit_blocks(dba.build_gaussian_access((*ONE, (.37, "I")), ("X", "Z"), clock_bits=2))
    np.testing.assert_allclose(base, shifted, atol=1e-12)


def test_continuum_kernel_with_extreme_scaled_gap_remains_finite():
    for beta in (1e-6, 1e6):
        report = dba.continuum_generator_reference(((1, "Z"),), ("X", "Z"), beta=beta)
        assert math.isfinite(report["gibbs_stationarity_residual"])
        assert report["gibbs_stationarity_residual"] < 1e-12
        assert report["minimum_decay_eigenvalue"] >= -1e-12


@pytest.mark.parametrize("kwargs", [{"beta": 0}, {"beta": True}, {"beta": math.inf},
                                   {"clock_bits": True}, {"clock_bits": 7},
                                   {"product_steps": 0}, {"product_steps": 129}])
def test_source_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        dba.build_gaussian_access(ONE, ("X",), **kwargs)


@pytest.mark.parametrize("terms,proposals", [(((math.nan, "Z"),), ("X",)),
                                             (((1, "Z"),), ("x",)),
                                             (((1, "ZZ"),), ("X",)),
                                             (((33, "Z"),), ("X",)),
                                             (((1, "Z"),), ("X", "Y", "Z"))])
def test_pauli_input_validation(terms, proposals):
    with pytest.raises(ValueError):
        dba.build_gaussian_access(terms, proposals)


def test_source_generator_and_dense_references_have_distinct_caps():
    circuit = dba.build_gaussian_access(((1, "Z" + "I" * 11),), ("X" + "I" * 11,), clock_bits=2)
    assert circuit.num_qubits == 14
    assert len(circuit.data) < dba.MAX_PRIMITIVE_OPERATIONS
    with pytest.raises(ValueError, match="actual circuit checks"):
        dba.circuit_blocks(circuit)
    with pytest.raises(ValueError, match="dense references"):
        dba.reference_blocks(((1, "ZZZ"),), ("XXX",))
    with pytest.raises(ValueError, match="preflight"):
        dba.build_gaussian_access(tuple((1, "ZZ") for _ in range(32)), PROPOSALS,
                                  clock_bits=6, product_steps=128)


@pytest.mark.parametrize("field,value", [("clock_bits", 40), ("clock_bits", True),
                                         ("n_system", True), ("n_system", 16),
                                         ("proposal_count", 3), ("proposal_count", 2 ** 30),
                                         ("proposal_bits", 1), ("proposal_bits", False)])
def test_untrusted_metadata_rejected_before_any_block_allocation(field, value, monkeypatch):
    circuit = dba.build_gaussian_access(ONE, ("X", "Y", "Z", "I"), clock_bits=2)
    circuit.metadata[field] = value
    monkeypatch.setattr(dba.np, "empty", lambda *args, **kwargs: pytest.fail("allocation happened before metadata validation"))
    with pytest.raises(ValueError, match="actual circuit checks"):
        dba.circuit_blocks(circuit)


def test_external_width_gates_and_nonfinite_angles_fail_before_simulation():
    original = dba.build_gaussian_access(ONE, ("X",), clock_bits=2)
    wrong_width = QuantumCircuit(original.num_qubits + 1)
    wrong_width.metadata = dict(original.metadata)
    bad_angle = original.copy()
    bad_angle.rz(math.nan, 0)
    composite = original.copy()
    composite.append(QuantumCircuit(1, name="opaque").to_gate(), [0])
    for circuit in (wrong_width, bad_angle, composite):
        with pytest.raises(ValueError, match="actual circuit checks"):
            dba.circuit_blocks(circuit)


@pytest.mark.parametrize("h,d", [(np.array([[1, 1], [0, 1]]), np.eye(2)),
                                (np.eye(2), np.array([[1, 0], [0, math.nan]])),
                                (np.eye(8), np.eye(8))])
def test_spectral_reference_requires_bounded_hermitian_inputs(h, d):
    with pytest.raises(ValueError, match="finite Hermitian"):
        dba.spectral_correction(h, d, 1)
