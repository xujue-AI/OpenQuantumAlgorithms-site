"""Pure verification guards, defined without any scientific execution.

Only permission checks, strict scalar/shape parsing and integer reservation
counters are called by these tests. No test enables verify_c5_records or runs
a Boolean decoder mapping, phase reference, operation generator or state pass.
"""
import builtins
import importlib.util
import math
from pathlib import Path
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/openquantum/optimization_dqi_degree_two_verify.py"
FORBIDDEN_IMPORTS = frozenset({"openquantum", "qiskit", "numpy", "qdk",
                               "nbclient", "ipykernel", "psutil"})


def _load():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in FORBIDDEN_IMPORTS:
            raise AssertionError("The pure verifier guard imported a backend")
        return original(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location("_degree_two_pure_verifier_guards", SOURCE)
    if spec is None or spec.loader is None:
        raise AssertionError("Standalone verifier source is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original
    return module


VERIFY = _load()


class Unreadable:
    """A missing authority check must fail before touching input or callbacks."""

    def __iter__(self):
        raise AssertionError("Unauthorized verification read records")

    def __len__(self):
        raise AssertionError("Unauthorized verification counted records")

    def __getitem__(self, key):
        raise AssertionError("Unauthorized verification indexed input")

    def get(self, *args, **kwargs):
        raise AssertionError("Unauthorized verification read ledger")


def forbidden_callback():
    raise AssertionError("Unauthorized verification called a deadline or engine")


@pytest.mark.parametrize("allow", [False, None, 0, 1, "true"])
def test_verification_requires_exact_opt_in_before_any_input_or_callback(allow):
    with pytest.raises((PermissionError, ValueError)):
        VERIFY.verify_c5_records(Unreadable(), Unreadable(),
                                 allow_execution=allow, check=forbidden_callback)


def test_default_verification_cannot_enter_scientific_work():
    with pytest.raises((PermissionError, ValueError)):
        VERIFY.verify_c5_records(Unreadable(), Unreadable(), check=forbidden_callback)


@pytest.mark.parametrize("enabled", [False, None, 0, 1, "true"])
def test_budget_authority_uses_boolean_identity(enabled):
    with pytest.raises((PermissionError, ValueError)):
        budget = VERIFY.ExecutionBudget(execution_enabled=enabled)
        budget.begin()


def test_budget_reservation_requires_begin_without_allocating_a_state():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    with pytest.raises(ValueError):
        budget.reserve_full_evolution()


def test_budget_reentry_is_rejected_before_a_second_validation():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    with pytest.raises(ValueError):
        budget.begin()


def test_a_reserved_full_evolution_slot_cannot_be_reused():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution(wires=12, amplitudes=4096)
    with pytest.raises(ValueError):
        budget.reserve_full_evolution(wires=12, amplitudes=4096)


def test_abort_does_not_turn_a_spent_budget_into_a_fresh_attempt():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    budget.abort()
    with pytest.raises(ValueError):
        budget.begin()
    with pytest.raises(ValueError):
        budget.reserve_full_evolution()


def test_a_small_ccx_reference_slot_is_also_once_only():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_ccx_reference(matrix_dimension=8)
    with pytest.raises(ValueError):
        budget.reserve_ccx_reference(matrix_dimension=8)


def test_finish_cannot_substitute_for_missing_scientific_checks():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    with pytest.raises(ValueError):
        budget.finish()


def test_counter_only_boolean_reservations_are_bounded():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    for _ in range(512):
        budget.reserve_boolean_mapping()
    with pytest.raises(ValueError):
        budget.reserve_boolean_mapping()


def test_snapshot_mutation_cannot_reset_reservations():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    snapshot = budget.snapshot()
    snapshot.clear()
    assert budget.snapshot()
    with pytest.raises(ValueError):
        budget.reserve_full_evolution()


def test_exact_counter_completion_is_not_a_scientific_validation_result():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_ccx_reference()
    for _ in range(512):
        budget.reserve_boolean_mapping()
    budget.reserve_full_evolution()
    for _ in range(1773):
        budget.charge_unitary_gate()
    budget.finish()
    snapshot = budget.snapshot()
    assert snapshot["complete"] is True and snapshot["active"] is False
    assert snapshot["complete_statevector_evolutions"] == 1
    assert snapshot["boolean_decoder_mappings"] == 512
    assert snapshot["small_ccx_reference_cases"] == 1
    assert snapshot["primitive_unitary_applications"] == 1773
    assert "charged entries" in snapshot["counter_scope"]
    with pytest.raises(ValueError):
        budget.reserve_boolean_mapping()


def test_unitary_application_counter_rejects_an_extra_gate_without_applying_one():
    budget = VERIFY.ExecutionBudget(execution_enabled=True)
    budget.begin()
    for _ in range(1773):
        budget.charge_unitary_gate()
    with pytest.raises(ValueError):
        budget.charge_unitary_gate()


def test_quantum_envelope_rejects_duplicate_and_unknown_reservations_without_allocating():
    envelope = VERIFY._QuantumEnvelope()
    envelope.reserve("dummy", 4, 128)
    with pytest.raises(ValueError):
        envelope.reserve("dummy", 4, 128)
    with pytest.raises(ValueError):
        envelope.release("missing")
    envelope.release("dummy")
    assert envelope.total == 0 and envelope.live == {}


def test_source_quantum_envelope_is_a_hard_total_byte_limit():
    envelope = VERIFY._QuantumEnvelope()
    for index in range(31):
        envelope.reserve("dummy_" + str(index), 4096, 256)
    with pytest.raises(ValueError):
        envelope.reserve("overflow", 4096, 256)
    assert envelope.total <= 32 << 20
    assert len(envelope.live) == 31


@pytest.mark.parametrize("role,arguments", [
    ("dense_operator", {"wires": 12, "matrix_dimension": 4096}),
    ("full_circuit", {"wires": 12, "matrix_dimension": 4096}),
    ("sampled_state", {"wires": 12, "amplitudes": 4096}),
    ("physical_model", {}), ("native_synthesis", {}),
    ("full_state", {"wires": 13, "amplitudes": 8192}),
    ("full_state", {"wires": 12, "amplitudes": 4097}),
    ("full_state", {"wires": True, "amplitudes": 4096}),
    ("full_state", {"wires": 12, "amplitudes": True}),
    ("full_state", {"wires": 12, "amplitudes": 4096, "matrix_dimension": 8}),
    ("small_ccx", {"wires": 12, "matrix_dimension": 4096}),
    ("small_ccx", {"wires": 3, "matrix_dimension": 9}),
    ("small_ccx", {"wires": 3, "matrix_dimension": True}),
    ("small_ccx", {"wires": 3, "matrix_dimension": 8, "amplitudes": 8}),
])
def test_shape_guard_has_no_dense_sampled_or_relaxed_mode(role, arguments):
    with pytest.raises(ValueError):
        VERIFY.validate_quantum_shape(role, **arguments)


def test_only_declared_full_state_and_small_reference_shapes_pass_pure_guards():
    VERIFY.validate_quantum_shape("full_state", wires=12, amplitudes=4096)
    VERIFY.validate_quantum_shape("small_ccx", wires=3, matrix_dimension=8)


@pytest.mark.parametrize("record", [
    {"gate": "ccx", "qubits": [0, 1, 2]},
    {"gate": "mcx", "qubits": [0, 1, 2, 3]},
    {"gate": "decoder_oracle", "qubits": [0]},
    {"gate": "h", "qubits": [True]},
    {"gate": "h", "qubits": [-1]},
    {"gate": "h", "qubits": [12]},
    {"gate": "cx", "qubits": [0, 0]},
    {"gate": "cx", "qubits": [0]},
    {"gate": "measure", "qubits": [5], "classical": [True]},
    {"gate": "measure", "qubits": [5], "classical": [4]},
    {"gate": "measure", "qubits": [5]},
    {"gate": "h", "qubits": [0], "parameters_hex": ["0x1p-3"]},
    {"gate": "ry", "qubits": [0], "parameters_hex": ["nan"]},
    {"gate": "ry", "qubits": [0], "parameters_hex": ["inf"]},
    {"gate": "ry", "qubits": [0], "parameters_hex": [True]},
    {"gate": "ry", "qubits": [0], "parameters_hex": []},
    {"gate": "ry", "qubits": [0], "parameters_hex": ["0x1p-3", "0x1p-2"]},
])
def test_primitive_parser_rejects_invalid_scalar_or_opaque_work_without_math(record):
    with pytest.raises(ValueError):
        VERIFY.validate_primitive_record(record)


@pytest.mark.parametrize("field,value", [
    ("sampled_outcomes", True), ("physical_resources_computed", True),
    ("formal_interval_certificate", True), ("postselection", True),
    ("global_phase_fitted", True), ("new_instruction", "unitary"),
])
def test_primitive_schema_cannot_smuggle_scope_or_certificate_claims(field, value):
    record = {"gate": "h", "qubits": [0], field: value}
    with pytest.raises(ValueError):
        VERIFY.validate_primitive_record(record)


def test_valid_single_records_are_parsed_without_reference_or_trigonometry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Primitive parsing requested scientific arithmetic")

    class MathGuard:
        def __getattr__(self, name):
            return forbidden if name in {"sin", "cos", "sqrt", "asin", "exp"} else getattr(math, name)

    monkeypatch.setattr(VERIFY, "math", MathGuard())
    VERIFY.validate_primitive_record({"gate": "h", "qubits": [0]})
    VERIFY.validate_primitive_record({"gate": "ry", "qubits": [0], "parameters_hex": ["0x1.0000000000000p-3"]})
    VERIFY.validate_primitive_record({"gate": "measure", "qubits": [5], "classical": [0]})
