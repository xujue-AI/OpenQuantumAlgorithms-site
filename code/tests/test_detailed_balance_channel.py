"""Independent full-channel checks, phases, reset semantics and scoped budgets."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from openquantum.simulation import detailed_balance_access as dba
from openquantum.simulation import detailed_balance_channel as dbc

ONE = ((1., "Z"),)
PROPOSALS = ("X", "Z")


@pytest.mark.parametrize("proposals", [("X",), ("Y", "Z"), ("Y", "X", "Z", "I")])
def test_conditional_inverse_preserves_entire_unitary_extension_and_phases(proposals):
    unitary = dba.build_gaussian_access((*ONE, (.37, "I")), proposals, clock_bits=1)
    unitary.global_phase = .23
    circuit = QuantumCircuit(unitary.num_qubits + 1)
    dbc._conditional_inverse(circuit, unitary, unitary.num_qubits)
    dim = 1 << unitary.num_qubits
    expected = np.eye(2 * dim, dtype=complex)
    expected[:dim, :dim] = Operator(unitary).data.conj().T
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-12)
    assert set(circuit.count_ops()) <= dba.PRIMITIVES


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_independent_prefix_prepare_matches_full_matrix_extension(bits):
    mesh = dba.gaussian_mesh(bits)
    expected = dbc._prefix_prepare_matrix(mesh["amplitudes"])
    np.testing.assert_allclose(Operator(dba.positive_prepare(mesh["amplitudes"])).data,
                               expected, atol=2e-12)


@pytest.mark.parametrize("bits,delta,beta", [(1, .125, 1.), (2, .125, 1.),
                                           (3, .0625, 1.), (3, .25, .5), (4, .125, 2.)])
def test_entire_actual_channel_including_all_nojump_labels(bits, delta, beta):
    circuit = dbc.build_weak_dilation(ONE, PROPOSALS, clock_bits=bits, delta=delta, beta=beta)
    actual = dbc.circuit_kraus(circuit)
    expected = dbc.reference_kraus(ONE, PROPOSALS, clock_bits=bits, delta=delta, beta=beta)
    np.testing.assert_allclose(actual, expected, atol=5e-12)
    np.testing.assert_allclose(np.einsum("aji,ajk->ik", actual.conj(), actual), np.eye(2), atol=5e-12)
    # The environmental no-jump labels are real channel contributions. An
    # accepted-only K0,0 replacement would fail both this and completeness.
    assert np.sum(np.abs(actual[1:len(actual)//2]) ** 2) > 1e-7
    s = dbc.channel_superoperator(actual)
    for rho in [np.diag([1, 0]), np.diag([0, 1]),
                np.array([[1, 1], [1, 1]]) / 2,
                np.array([[1, -1j], [1j, 1]]) / 2]:
        result = dbc.apply_channel(s, rho)
        np.testing.assert_allclose(result, dbc.apply_channel(dbc.channel_superoperator(expected), rho), atol=5e-12)
        assert np.linalg.eigvalsh(result).min() >= -5e-12
        assert np.trace(result) == pytest.approx(1, abs=5e-12)


def test_reset_instrument_on_plus_input_matches_channel_squared():
    kwargs = {"clock_bits": 1, "delta": .125}
    one = dbc.build_weak_round(ONE, PROPOSALS, **kwargs)
    circuit = QuantumCircuit(one.num_qubits)
    circuit.h(0)
    circuit.compose(one, inplace=True)
    circuit.compose(one, inplace=True)
    actual = partial_trace(DensityMatrix.from_instruction(circuit), list(range(1, circuit.num_qubits))).data
    s = dbc.channel_superoperator(dbc.circuit_kraus(dbc.build_weak_dilation(ONE, PROPOSALS, **kwargs)))
    expected = dbc.apply_channel(s, np.array([[1, 1], [1, 1]]) / 2, rounds=2)
    np.testing.assert_allclose(actual, expected, atol=5e-12)
    assert one.count_ops()["reset"] == 2 * (one.num_qubits - 1)


def test_one_spin_population_update_and_delta_independent_fixed_point():
    refs = dbc.independent_generators(ONE, PROPOSALS, clock_bits=2)
    downward, upward = refs["finite"][3, 0].real, refs["finite"][0, 3].real
    stationary = np.diag([upward, downward]) / (upward + downward)
    for delta in (.0625, .125, .25):
        kraus = dbc.reference_kraus(ONE, PROPOSALS, clock_bits=2, delta=delta)
        s = dbc.channel_superoperator(kraus)
        expected = np.array([[1 - delta * downward, delta * upward],
                             [delta * downward, 1 - delta * upward]])
        np.testing.assert_allclose(s[np.ix_([0, 3], [0, 3])].real, expected, atol=3e-12)
        np.testing.assert_allclose(dbc.apply_channel(s, stationary), stationary, atol=3e-12)
    gibbs = refs["gibbs"]
    assert np.linalg.norm(refs["continuum"] @ gibbs.reshape(4, order="F")) < 1e-12
    assert np.linalg.norm(refs["finite"] @ gibbs.reshape(4, order="F")) > 1e-5


def test_first_order_error_scales_quadratically_without_claiming_exact_gibbs():
    refs = dbc.independent_generators(ONE, PROPOSALS, clock_bits=2)
    errors = []
    for delta in (.125, .0625, .03125):
        s = dbc.channel_superoperator(dbc.reference_kraus(ONE, PROPOSALS, clock_bits=2, delta=delta))
        errors.append(np.linalg.norm(s - np.eye(4) - delta * refs["finite"]))
    assert 3.5 < errors[0] / errors[1] < 4.7
    assert 3.5 < errors[1] / errors[2] < 4.7


def test_channel_power_bound_checks_coherences_and_reports_gibbs_bias():
    result = dbc.channel_check(ONE, PROPOSALS, clock_bits=2, rounds=64)
    assert result["bloch_contraction_norm"] < 1
    assert result["finite_stationary_gibbs_trace_distance"] > 0
    assert result["finite_stationary_channel_residual"] < 1e-11
    for row in result["spanning_input_channel_power_checks"]:
        assert row["finite_stationary_trace_distance"] <= row["numerical_contraction_estimate_to_finite_stationary"] + 1e-11
    assert "no certified" in result["contraction_scope"]
    assert "no full statevector" in result["channel_power_scope"]
    assert not result["metadata"]["physical_resources_computed"]


@pytest.mark.parametrize("delta", [0, -1, True, math.nan, math.inf, 2, 5e-324])
def test_weak_step_domain(delta):
    with pytest.raises(ValueError):
        dbc.build_weak_dilation(ONE, PROPOSALS, delta=delta)


def test_nonzero_angle_halving_underflow_rejected():
    with pytest.raises(ValueError, match="halved"):
        dbc._half(5e-324)


@pytest.mark.parametrize("terms", [((1., "X"),), ((.2, "YZ"),), ((1., "IZ"), (.2, "XX"))])
def test_noncommuting_endpoint_requires_real_correction_instead_of_relabeling(terms):
    with pytest.raises(ValueError, match="diagonal"):
        dbc.build_weak_dilation(terms, ("X" * len(terms[0][1]),))


@pytest.mark.parametrize("field,value", [("n_system", True), ("n_system", 2),
                                        ("clock_bits", 20), ("clock_bits", True),
                                        ("proposal_count", 3), ("proposal_bits", False),
                                        ("weak_flag", 0)])
def test_metadata_cannot_trigger_large_simulation(field, value, monkeypatch):
    circuit = dbc.build_weak_dilation(ONE, PROPOSALS, clock_bits=2)
    circuit.metadata[field] = value
    monkeypatch.setattr(dbc, "Statevector", lambda *args, **kwargs: pytest.fail("simulation before validation"))
    with pytest.raises(ValueError):
        dbc.circuit_kraus(circuit)


def test_unresolved_angles_and_composite_operations_rejected_before_simulation(monkeypatch):
    original = dbc.build_weak_dilation(ONE, PROPOSALS, clock_bits=2)
    bad = original.copy()
    bad.rz(Parameter("angle"), 0)
    composite = original.copy()
    composite.append(QuantumCircuit(1, name="opaque").to_gate(), [0])
    monkeypatch.setattr(dbc, "Statevector", lambda *args, **kwargs: pytest.fail("simulation before validation"))
    for circuit in (bad, composite):
        with pytest.raises(ValueError):
            dbc.circuit_kraus(circuit)


def test_repeated_source_includes_initialization_resets_readout_and_all_rounds():
    circuit = dbc.build_preparation(ONE, PROPOSALS, clock_bits=1, rounds=3, initial="plus")
    meta = circuit.metadata
    assert circuit.count_ops()["reset"] == meta["total_resets"]
    assert circuit.count_ops()["measure"] == 1
    assert len(circuit.data) == 3 * meta["full_round_operations"] + 3
    assert set(circuit.count_ops()) <= dbc.CHANNEL_PRIMITIVES
    assert not meta["rotations_synthesized"]
    assert "no accepted-only" in meta["repetitions"]


def test_scalable_source_and_bounded_matrix_checks_are_distinct():
    terms = ((1., "Z" + "I" * 11),)
    proposals = ("X" + "I" * 11,)
    source = dbc.build_preparation(terms, proposals, clock_bits=2, rounds=1, readout=False)
    assert source.num_qubits == 15
    assert len(source.data) <= dba.MAX_PRIMITIVE_OPERATIONS
    with pytest.raises(ValueError, match="one system wire"):
        dbc.reference_kraus(terms, proposals, clock_bits=2)
    with pytest.raises(ValueError, match="preflight"):
        dbc.build_preparation(ONE, PROPOSALS, clock_bits=4, rounds=256)
    assert "unitary" not in qasm3.dumps(source)


def _generator():
    path = Path(__file__).resolve().parents[1] / "scripts/generate_detailed_balance_channel.py"
    spec = importlib.util.spec_from_file_location("channel_generator_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_source_only_is_explicit_and_does_not_relabel_existing_output(tmp_path):
    generator = _generator()
    output = tmp_path / "source-only"
    report = generator.generate(output, simulate=False, n_system=12, clock_bits=2, circuit_rounds=1)
    assert report["status"] == "generated_diagonal_channel_without_simulation"
    assert not report["cases"][0]["numerical_checks_performed"]
    assert not report["general_sampler_implemented"]
    contract = report["cases"][0]["logical_readout_contract"]
    assert contract["independent_complete_preparations"] == 738
    assert not contract["synthesis_and_physical_faults_included"]
    before = (output / "results.json").read_bytes()
    with pytest.raises(ValueError, match="absent"):
        generator.generate(output, simulate=False)
    assert (output / "results.json").read_bytes() == before
    with pytest.raises(ValueError, match="capped"):
        generator.generate(tmp_path / "invalid-large", n_system=2)
    assert not (tmp_path / "invalid-large").exists()
