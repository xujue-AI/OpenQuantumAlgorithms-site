"""Fixed decay-block checks and guards; prepared source, not an execution receipt."""
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PhaseGate, RYGate
from qiskit.quantum_info import Statevector

from openquantum.simulation import detailed_balance_decay as decay
from openquantum.simulation import detailed_balance_decay_reference as reference


TOLERANCE = 1e-10
CASES = (
    ("diagonal", ((1.0, "Z"),), ("X", "Z"), 6),
    ("tilted", ((0.6, "Z"), (0.8, "X")), ("Z",), 5),
    ("complex", ((0.7, "ZI"), (0.4, "XX"), (0.3, "IX"),
                 (0.19, "YZ"), (0.23, "IZ")), ("XI", "ZI", "IX", "IZ"), 8),
)
STAGES = (
    "selector_open", "access_forward", "weight_forward", "cz_primitives",
    "selector_negative_phase", "weight_inverse", "access_inverse", "selector_close",
)
UNFINISHED = (
    "all_components_verified", "new_algorithm_result", "coherent_correction_implemented",
    "channel_implemented", "sampler_implemented", "finite_mesh_exact_kms",
    "mixing_certificate", "rotations_synthesized", "physical_resources_computed",
    "formal_proof_reproduced", "numerical_checks_performed",
)


@pytest.fixture(scope="module")
def sources():
    """Exactly three actual sources, shared without mutation across the tests."""
    return {
        name: decay.build_decay_block(terms, proposals, beta=1.0,
                                     clock_bits=2, product_steps=1)
        for name, terms, proposals, _ in CASES
    }


@pytest.fixture(scope="module")
def checked(sources):
    """Core comparisons use 2/2/4 actual basis columns, eight in total."""
    results = {}
    for name, terms, proposals, _ in CASES:
        finite = reference.reference_decay(terms, proposals, beta=1.0,
                                           clock_bits=2, product_steps=1)
        exact_h = reference.reference_decay(terms, proposals, beta=1.0,
                                            clock_bits=2, product_steps=None)
        results[name] = {
            "finite": finite, "exact_h": exact_h,
            "actual": reference.circuit_block(sources[name]),
        }
    return results


def _signature(circuit, instruction, *, inverse=False):
    operation = instruction.operation.inverse() if inverse else instruction.operation
    return (
        operation.name, tuple(float(value) for value in operation.params),
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
        tuple(circuit.find_bit(bit).index for bit in instruction.clbits),
    )


def _stage(circuit, name):
    start, stop = circuit.metadata["stage_slices"][name]
    return list(circuit.data[start:stop])


@pytest.mark.parametrize("name,terms,proposals,width", CASES, ids=[case[0] for case in CASES])
def test_fixed_finite_trotter_block_and_independent_physical_alpha(
        name, terms, proposals, width, sources, checked):
    circuit, result = sources[name], checked[name]
    finite, actual = result["finite"], result["actual"]
    expected = finite["normalized_decay"]
    dim = 1 << len(terms[0][1])
    assert circuit.num_qubits == width
    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(actual, actual.conj().T, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(finite["decay"], finite["decay"].conj().T,
                               atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(
        np.einsum("alji,aljk->ik", finite["blocks"].conj(), finite["blocks"]),
        np.eye(dim), atol=TOLERANCE, rtol=0,
    )
    # These are raw eigensolver outputs. Tolerance admits tiny negative
    # roundoff; no clipping, phase alignment or post-run tuning is applied.
    eigenvalues = np.linalg.eigvalsh(actual)
    assert eigenvalues.min() >= -TOLERANCE
    assert eigenvalues.max() <= 1 + TOLERANCE
    assert np.linalg.norm(actual, ord=2) <= 1 + TOLERANCE
    np.testing.assert_allclose(eigenvalues, np.linalg.eigvalsh(expected),
                               atol=TOLERANCE, rtol=0)

    # A second physical normalization derivation is separate from the
    # normalized block: dropping eta can otherwise pass the block comparison.
    signed = np.array([0, 1, -2, -1])
    dt = 0.5 * math.sqrt(2 * math.pi / 4)
    dw = 2 * math.sqrt(2 * math.pi / 4)
    times = dt * signed
    sampled_f = np.exp(-times ** 2) * math.sqrt(math.sqrt(2 / math.pi))
    physical_eta = dt * float(sampled_f @ sampled_f)
    gamma = np.exp(-np.maximum(dw * signed + 0.5, 0))
    physical_alpha = physical_eta * float(gamma.max())
    assert circuit.metadata["eta"] == pytest.approx(physical_eta, abs=1e-15, rel=0)
    assert circuit.metadata["alpha"] == pytest.approx(physical_alpha, abs=1e-15, rel=0)
    assert finite["mesh"]["eta"] == pytest.approx(physical_eta, abs=1e-15, rel=0)
    assert finite["mesh"]["alpha"] == pytest.approx(physical_alpha, abs=1e-15, rel=0)
    assert circuit.metadata["eta"] != 1
    np.testing.assert_allclose(actual * physical_alpha, finite["decay"],
                               atol=TOLERANCE, rtol=0)

    h, d = finite["hamiltonian"], finite["decay"]
    commutator = float(np.linalg.norm(h @ d - d @ h))
    assert math.isfinite(commutator)
    if name == "diagonal":
        assert commutator < TOLERANCE
    elif name == "complex":
        # Same frozen complex B/D family as the original access checks.
        assert commutator > 1e-5
    # The tilted case's commutator remains a measured diagnostic, with no
    # predeclared nonzero threshold or general coherent-correction claim.
    exact_difference = np.linalg.norm(expected - result["exact_h"]["normalized_decay"])
    assert math.isfinite(float(exact_difference))
    if name == "diagonal":
        assert exact_difference < TOLERANCE


@pytest.mark.parametrize("name,terms,proposals,width", CASES, ids=[case[0] for case in CASES])
def test_literal_inverse_chronology_counts_and_unfinished_scope(
        name, terms, proposals, width, sources):
    circuit = sources[name]
    meta = circuit.metadata
    assert meta["milestone"] == decay.MILESTONE
    assert meta["selected_auxiliary_state"] == "all_zero"
    assert meta["auxiliary_wires"] == list(range(len(terms[0][1]), width))
    assert meta["selector_wire"] == width - 1
    assert meta["weight_wire"] == width - 2
    assert not meta["selector_zero_only_is_output_contract"]
    assert not circuit.num_clbits
    assert set(circuit.count_ops()) <= decay.PRIMITIVES
    assert all(meta[field] is False for field in UNFINISHED)

    cursor = 0
    assert tuple(meta["stage_slices"]) == STAGES
    for name_of_stage in STAGES:
        start, stop = meta["stage_slices"][name_of_stage]
        assert start == cursor
        assert stop > start
        cursor = stop
    assert cursor == len(circuit.data)
    for forward, backward in (("access_forward", "access_inverse"),
                              ("weight_forward", "weight_inverse")):
        expected_inverse = [
            _signature(circuit, instruction, inverse=True)
            for instruction in reversed(_stage(circuit, forward))
        ]
        assert [_signature(circuit, instruction)
                for instruction in _stage(circuit, backward)] == expected_inverse

    weight, selector = meta["weight_wire"], meta["selector_wire"]
    assert [_signature(circuit, instruction) for instruction in _stage(circuit, "cz_primitives")] == [
        ("h", (), (weight,), ()), ("cx", (), (selector, weight), ()),
        ("h", (), (weight,), ()),
    ]
    assert [_signature(circuit, instruction)
            for instruction in _stage(circuit, "selector_negative_phase")] == [
        ("p", (math.pi,), (selector,), ()),
    ]
    for stage in ("selector_open", "selector_close"):
        assert [_signature(circuit, instruction) for instruction in _stage(circuit, stage)] == [
            ("h", (), (selector,), ()),
        ]
    size = 1 << meta["clock_bits"]
    for stage in ("weight_forward", "weight_inverse"):
        names = [instruction.operation.name for instruction in _stage(circuit, stage)]
        assert names.count("ry") == size
        assert names.count("cx") == size
        assert len(names) == 2 * size
    expected_operations = 2 * meta["access_source_operations_each_copy"] + 4 * size + 6
    assert len(circuit.data) == meta["source_operations"] == expected_operations
    assert meta["expected_source_operations"] == expected_operations
    assert expected_operations <= meta["preflight_upper_bound"] <= 100_000
    ledger = meta["resource_ledger"]
    assert ledger["access_copies"] == ledger["weight_multiplexor_copies"] == 2
    assert ledger["access_instructions_total"] == 2 * meta["access_source_operations_each_copy"]
    assert sum(ledger[field] for field in (
        "proposal_prepare_h_instructions_total", "gaussian_prepare_instructions_total",
        "pauli_select_instructions_total", "signed_evolution_instructions_total",
        "dft_instructions_total",
    )) == ledger["access_instructions_total"]
    assert ledger["weight_ry_instructions_total"] == ledger["weight_cx_instructions_total"] == 2 * size
    assert ledger["selector_and_reflection_instructions_total"] == 6
    assert ledger["source_operations_total"] == expected_operations
    assert "caller supplied" in ledger["initialization"]
    assert "no measurement" in ledger["readout"]
    assert "arbitrary-angle rotation synthesis" in ledger["exclusions"]
    assert "repetitions or coherent amplification" in ledger["exclusions"]


@pytest.mark.parametrize("name,terms,proposals,width", CASES, ids=[case[0] for case in CASES])
def test_missing_selector_phase_is_complement_and_selector_zero_is_not_all_zero(
        name, terms, proposals, width, sources, checked):
    circuit = sources[name]
    expected = checked[name]["finite"]["normalized_decay"]
    n = len(terms[0][1])
    dim = 1 << n
    start, stop = circuit.metadata["stage_slices"]["selector_negative_phase"]
    assert stop == start + 1
    shortened = circuit.copy()
    del shortened.data[start]
    # The extractor rejects this shortened stream because its declared
    # source count differs; count validation does not authenticate chronology.
    with pytest.raises(ValueError):
        reference.circuit_block(shortened)
    complement = circuit.copy()
    complement.data[start] = complement.data[start].replace(operation=PhaseGate(0.0))
    assert len(complement.data) == len(circuit.data)
    # A same-count P(0) substitution still satisfies the bounded extractor's
    # layout/arity/count guards. The independent block comparison exposes its
    # complementary operator rather than treating metadata as authentication.
    complement_block = reference.circuit_block(complement)
    np.testing.assert_allclose(complement_block, np.eye(dim) - expected,
                               atol=TOLERANCE, rtol=0)

    # One original |0> column per case, separate from the eight core columns.
    original = Statevector.from_int(0, 1 << width).evolve(circuit).data
    selector_zero_probability = float(np.sum(np.abs(original[:1 << circuit.metadata["selector_wire"]]) ** 2))
    all_auxiliary_zero_probability = float(np.sum(np.abs(original[:dim]) ** 2))
    assert selector_zero_probability == pytest.approx(float(expected[0, 0].real), abs=TOLERANCE, rel=0)
    assert all_auxiliary_zero_probability == pytest.approx(
        float(np.vdot(expected[:, 0], expected[:, 0]).real), abs=TOLERANCE, rel=0,
    )
    assert selector_zero_probability - all_auxiliary_zero_probability > 1e-8


def test_unconditional_access_inverse_cancels_global_phase(monkeypatch):
    original_builder = decay.access.build_gaussian_access

    def phase_shifted_access(*args, **kwargs):
        unitary = original_builder(*args, **kwargs)
        unitary.global_phase = 0.23
        return unitary

    monkeypatch.setattr(decay.access, "build_gaussian_access", phase_shifted_access)
    terms, proposals = CASES[0][1:3]
    circuit = decay.build_decay_block(terms, proposals, beta=1.0, clock_bits=2, product_steps=1)
    assert math.remainder(float(circuit.global_phase), 2 * math.pi) == pytest.approx(0, abs=1e-15)
    expected = reference.reference_decay(terms, proposals, beta=1.0,
                                         clock_bits=2, product_steps=1)["normalized_decay"]
    np.testing.assert_allclose(reference.circuit_block(circuit), expected,
                               atol=TOLERANCE, rtol=0)


def _forbid_circuit_construction(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("circuit construction entered before rejecting the source input")
    monkeypatch.setattr(decay, "QuantumCircuit", unexpected)
    monkeypatch.setattr(decay.access, "build_gaussian_access", unexpected)


@pytest.mark.parametrize("kwargs", [
    {"beta": 0}, {"beta": True}, {"beta": math.nan}, {"beta": math.inf},
    {"beta": 1e-7}, {"beta": 1e7}, {"clock_bits": 0}, {"clock_bits": 7},
    {"clock_bits": True}, {"product_steps": 0}, {"product_steps": 129},
    {"product_steps": True},
])
def test_parameter_rejection_precedes_circuit_allocation(kwargs, monkeypatch):
    _forbid_circuit_construction(monkeypatch)
    with pytest.raises(ValueError):
        decay.build_decay_block(((1.0, "Z"),), ("X",), **kwargs)


@pytest.mark.parametrize("terms,proposals", [
    ((), ("X",)), (((True, "Z"),), ("X",)), (((math.nan, "Z"),), ("X",)),
    (((1.0, "x"),), ("X",)), (((1.0, ""),), ("X",)),
    (((1.0, "ZZ"),), ("X",)), (((33.0, "Z"),), ("X",)),
    (((1.0, "Z"),), ("X", "Y", "Z")),
    (((1.0, "Z" + "I" * 16),), ("X" + "I" * 16,)),
    (tuple((0.01, "Z") for _ in range(33)), ("X",)),
])
def test_pauli_domain_rejection_precedes_circuit_allocation(terms, proposals, monkeypatch):
    _forbid_circuit_construction(monkeypatch)
    with pytest.raises(ValueError):
        decay.build_decay_block(terms, proposals)


@pytest.mark.parametrize("eta,rates", [
    (0, (1,)), (-1, (1,)), (True, (1,)), (math.nan, (1,)), (math.inf, (1,)),
    (1, (-1,)), (1, (math.nan,)), (1, (math.inf,)), (1, (True,)),
    (1, ()), (1, (0, 0)), (1, "not-rates"), (1, None),
    (1e308, (1e308,)), (5e-324, (5e-324,)), (1, (5e-324, 1e308)),
])
def test_invalid_weight_scales_are_rejected_without_clipping(eta, rates):
    with pytest.raises(ValueError):
        decay._weight_normalization(eta, rates)


def test_weight_normalization_retains_zero_rates_and_physical_scale():
    result = decay._weight_normalization(0.7, (0, 0.25, 1))
    assert result["eta"] == result["alpha"] == 0.7
    assert result["gamma_max"] == 1
    assert result["gamma"] == result["normalized_rates"] == [0, 0.25, 1]
    np.testing.assert_allclose(result["weight_angles"], [0, math.pi / 3, math.pi],
                               atol=1e-15, rtol=0)


def test_doubled_oracle_cap_rejects_an_access_below_its_own_cap(monkeypatch):
    # Five one-spin terms: B_U=61,596, while B_K=123,214. This exercises the
    # doubled guard rather than relying on rejection by the old U endpoint.
    _forbid_circuit_construction(monkeypatch)
    access_bound = 4 * 4 + 5 * 2 ** 2 + 120 + 2 * 2 * 128 * 5 * 24
    assert access_bound == 61_596 < decay.MAX_PRIMITIVE_OPERATIONS
    assert 2 * access_bound + 4 * 4 + 6 == 123_214 > decay.MAX_PRIMITIVE_OPERATIONS
    with pytest.raises(ValueError, match="preflight"):
        decay.build_decay_block(tuple((1.0, "Z") for _ in range(5)), ("X",),
                                clock_bits=2, product_steps=128)


def test_mirrored_bound_refuses_a_different_access_source_pin(monkeypatch):
    _forbid_circuit_construction(monkeypatch)
    monkeypatch.setattr(decay, "ACCESS_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="source pin"):
        decay.build_decay_block(((1.0, "Z"),), ("X",))


def test_n12_scalable_preflight_has_no_circuit_or_dense_allocation(monkeypatch):
    n = 12

    def label(operators):
        result = ["I"] * n
        for qubit, pauli in operators:
            result[qubit] = pauli
        return "".join(result)

    terms = tuple((1.0, label(((q, "Z"), (q + 1, "Z")))) for q in range(n - 1))
    terms += tuple((0.7, label(((q, "X"),))) for q in range(n))
    proposals = tuple(label(((q, pauli),)) for q, pauli in ((0, "X"), (0, "Z"), (11, "X"), (11, "Z")))
    _forbid_circuit_construction(monkeypatch)
    plan = decay.preflight_decay(terms, proposals, beta=1.0, clock_bits=2, product_steps=1)
    assert plan["total_wires"] == 18
    assert plan["preflight_upper_bound"] <= 100_000
    monkeypatch.setattr(reference, "expm", lambda *args, **kwargs: pytest.fail("dense exponential before width rejection"))
    with pytest.raises(ValueError):
        reference.reference_decay(terms, proposals, beta=1.0, clock_bits=2, product_steps=1)


def _forbid_actual_allocation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("actual state/block allocation entered before contract rejection")
    monkeypatch.setattr(reference, "Statevector", unexpected)
    monkeypatch.setattr(reference.np, "empty", unexpected)
    monkeypatch.setattr(reference.np, "zeros", unexpected)


@pytest.mark.parametrize("field,value", [
    ("n_system", True), ("n_system", 16), ("clock_bits", 40), ("clock_bits", True),
    ("proposal_count", 3), ("proposal_count", 2 ** 30), ("proposal_bits", False),
    ("weight_wire", 0), ("selector_wire", 0), ("auxiliary_wires", [0]),
    ("selected_auxiliary_state", "selector_zero_only"), ("milestone", "weak_channel"),
])
def test_spoofed_metadata_cannot_raise_actual_recovery_limits(field, value, sources, monkeypatch):
    circuit = sources["diagonal"].copy()
    circuit.metadata[field] = value
    _forbid_actual_allocation(monkeypatch)
    with pytest.raises(ValueError):
        reference.circuit_block(circuit)


@pytest.mark.parametrize("kind", [
    "wrong_width", "classical_wire", "symbolic_angle", "nonfinite_angle", "opaque_gate",
])
def test_invalid_source_stream_is_rejected_before_actual_allocation(kind, sources, monkeypatch):
    source = sources["diagonal"]
    if kind in ("wrong_width", "classical_wire"):
        circuit = QuantumCircuit(source.num_qubits + (kind == "wrong_width"),
                                 int(kind == "classical_wire"))
        circuit.metadata = dict(source.metadata)
    else:
        circuit = source.copy()
        if kind in ("symbolic_angle", "nonfinite_angle"):
            index = next(index for index, instruction in enumerate(circuit.data)
                         if instruction.operation.name == "ry")
            angle = Parameter("unresolved") if kind == "symbolic_angle" else math.nan
            circuit.data[index] = circuit.data[index].replace(operation=RYGate(angle))
        else:
            circuit.data[0] = circuit.data[0].replace(operation=QuantumCircuit(1, name="opaque").to_gate())
    _forbid_actual_allocation(monkeypatch)
    with pytest.raises(ValueError):
        reference.circuit_block(circuit)


def test_dense_and_actual_recovery_caps_are_independent_of_source_domain():
    assert reference.MAX_DENSE_SYSTEM_WIRES == 2
    assert reference.MAX_ACTUAL_RECOVERY_WIRES == 8
    assert decay.access.MAX_SOURCE_SYSTEM_WIRES == 16
    assert decay.MAX_PRIMITIVE_OPERATIONS == 100_000
