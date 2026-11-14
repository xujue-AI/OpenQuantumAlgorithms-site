"""Pure component-verification boundaries, authored but not executed.

Only disabled/default API entries, strict scalar/record/shape checks and
integer reservation counters may be exercised by a later authorized pytest
command. No test enables a numerical component call, allocates a state or runs
a kernel. Collection does not load the target verifier.
"""
import builtins
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/openquantum/optimization_dqi_path_majority_verify.py"
FORBIDDEN_IMPORTS = frozenset({
    "openquantum", "qiskit", "numpy", "scipy", "qdk", "nbclient",
    "ipykernel", "psutil", "torch", "cupy",
})


def _load_standalone(name):
    original = builtins.__import__

    def guarded(module_name, *args, **kwargs):
        if module_name.split(".", 1)[0] in FORBIDDEN_IMPORTS:
            raise AssertionError("A pure verifier boundary imported a scientific backend")
        return original(module_name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise AssertionError("Standalone path-majority verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original
    return module


@pytest.fixture(scope="module")
def verifier():
    name = "_path_majority_pure_verifier_boundaries"
    module = _load_standalone(name)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


class Unreadable:
    def __iter__(self):
        raise AssertionError("Unauthorized verification iterated source input")

    def __len__(self):
        raise AssertionError("Unauthorized verification counted source input")

    def __getitem__(self, key):
        raise AssertionError("Unauthorized verification indexed source input")

    def get(self, *args, **kwargs):
        raise AssertionError("Unauthorized verification read the ledger")


class EnabledLookingBudget:
    @property
    def _enabled(self):
        raise AssertionError("The verifier read an untrusted budget attribute")


def forbidden_callback():
    raise AssertionError("Unauthorized verification invoked its callback")


@pytest.mark.parametrize("allow", [False, None, 0, 1, "true", (), [], {}])
def test_exact_opt_in_is_required_before_inputs_callbacks_or_budget_reads(verifier, allow):
    with pytest.raises((PermissionError, ValueError)):
        verifier.verify_component_records(
            Unreadable(), Unreadable(), allow_numerical_execution=allow,
            budget=EnabledLookingBudget(), check=forbidden_callback,
        )


def test_default_entry_cannot_read_source_or_call_a_deadline(verifier):
    with pytest.raises((PermissionError, ValueError)):
        verifier.verify_component_records(Unreadable(), Unreadable(), check=forbidden_callback)


@pytest.mark.parametrize("budget", [None, True, False, 0, 1, {}, EnabledLookingBudget()])
def test_exact_opt_in_alone_does_not_supply_an_explicit_budget(verifier, budget):
    with pytest.raises((PermissionError, ValueError)):
        verifier.verify_component_records(
            Unreadable(), Unreadable(), allow_numerical_execution=True,
            budget=budget, check=forbidden_callback,
        )


def test_disabled_explicit_budget_cannot_read_source_or_call_a_deadline(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=False)
    with pytest.raises((PermissionError, ValueError)):
        verifier.verify_component_records(
            Unreadable(), Unreadable(), allow_numerical_execution=True,
            budget=budget, check=forbidden_callback,
        )
    assert budget.snapshot()["verification_invocations"] == 0


def test_budget_subclass_is_not_exact_execution_authority(verifier):
    class BudgetSubclass(verifier.ExecutionBudget):
        pass

    budget = BudgetSubclass(execution_enabled=True)
    with pytest.raises((PermissionError, ValueError)):
        verifier.verify_component_records(
            Unreadable(), Unreadable(), allow_numerical_execution=True,
            budget=budget, check=forbidden_callback,
        )


@pytest.mark.parametrize("enabled", [None, 0, 1, "true", (), [], {}])
def test_budget_enablement_rejects_truthy_and_falsey_nonbooleans(verifier, enabled):
    with pytest.raises((PermissionError, ValueError)):
        verifier.ExecutionBudget(execution_enabled=enabled)


def test_default_budget_cannot_begin(verifier):
    budget = verifier.ExecutionBudget()
    with pytest.raises((PermissionError, ValueError)):
        budget.begin()
    assert budget.snapshot()["spent"] is False


def test_reservations_require_a_begin_but_allocate_no_quantum_object(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    for reserve in (budget.reserve_full_evolution, budget.reserve_boolean_mapping, budget.charge_unitary_gate):
        with pytest.raises(ValueError):
            reserve()


def test_begin_cannot_reenter_an_active_or_completed_budget(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    with pytest.raises(ValueError):
        budget.begin()
    budget.abort()
    with pytest.raises(ValueError):
        budget.begin()


def test_one_component_forward_reservation_cannot_be_reused(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution(wires=12, amplitudes=4096)
    with pytest.raises(ValueError):
        budget.reserve_full_evolution(wires=12, amplitudes=4096)


def test_abort_keeps_the_once_only_budget_spent(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    budget.abort()
    snapshot = budget.snapshot()
    assert snapshot["spent"] is True and snapshot["active"] is False and snapshot["complete"] is False
    with pytest.raises(ValueError):
        budget.reserve_boolean_mapping()
    with pytest.raises(ValueError):
        budget.begin()


def test_boolean_mapping_counter_has_a_fixed_512_entry_ceiling(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    for _ in range(512):
        budget.reserve_boolean_mapping()
    with pytest.raises(ValueError):
        budget.reserve_boolean_mapping()


def test_gate_counter_has_a_fixed_525_entry_ceiling(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    for _ in range(525):
        budget.charge_unitary_gate()
    with pytest.raises(ValueError):
        budget.charge_unitary_gate()


def test_gate_charges_require_the_precharged_forward_slot(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    with pytest.raises(ValueError):
        budget.charge_unitary_gate()
    with pytest.raises(ValueError):
        budget.finish()


def test_finish_cannot_replace_missing_boolean_roles(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    for _ in range(525):
        budget.charge_unitary_gate()
    with pytest.raises(ValueError):
        budget.finish()


def test_complete_integer_counter_snapshot_is_not_a_component_result(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    for _ in range(512):
        budget.reserve_boolean_mapping()
    for _ in range(525):
        budget.charge_unitary_gate()
    budget.finish()
    snapshot = budget.snapshot()
    assert snapshot["verification_invocations"] == 1
    assert snapshot["complete_statevector_evolutions"] == 1
    assert snapshot["boolean_decoder_mappings"] == 512
    assert snapshot["primitive_unitary_applications"] == 525
    assert snapshot["small_ccx_reference_cases"] == snapshot["native_estimator_calls"] == 0
    assert snapshot["spent"] is True and snapshot["active"] is False and snapshot["complete"] is True
    assert "charged entries" in snapshot["counter_scope"]
    assert "status" not in snapshot and "maximum_state_entry_error" not in snapshot
    with pytest.raises(ValueError):
        budget.begin()


def test_snapshot_mutation_cannot_reset_reservations(verifier):
    budget = verifier.ExecutionBudget(execution_enabled=True)
    budget.begin()
    budget.reserve_full_evolution()
    copied = budget.snapshot()
    copied.clear()
    assert budget.snapshot()["complete_statevector_evolutions"] == 1
    with pytest.raises(ValueError):
        budget.reserve_full_evolution()


@pytest.mark.parametrize("role,arguments", [
    ("component_state", {"wires": 31, "amplitudes": 1 << 31}),
    ("component_state", {"wires": 13, "amplitudes": 8192}),
    ("component_state", {"wires": 12, "amplitudes": 4095}),
    ("component_state", {"wires": True, "amplitudes": 4096}),
    ("component_state", {"wires": 12, "amplitudes": True}),
    ("component_state", {"wires": 12, "amplitudes": 4096, "matrix_dimension": 8}),
    ("component_state", {"wires": 12, "amplitudes": 4096, "matrix_dimension": 4096}),
    ("full_state", {"wires": 12, "amplitudes": 4096}),
    ("small_ccx", {"wires": 3, "matrix_dimension": 8}),
    ("dense_operator", {"wires": 12, "matrix_dimension": 4096}),
    ("sampled_state", {"wires": 12, "amplitudes": 4096}),
    ("physical_model", {}), ("native_synthesis", {}),
])
def test_shape_gate_has_only_the_fixed_local_component_role(verifier, role, arguments):
    with pytest.raises(ValueError):
        verifier.validate_quantum_shape(role, **arguments)


def test_only_12_wire_4096_amplitude_shape_passes_the_pure_shape_gate(verifier):
    verifier.validate_quantum_shape("component_state", wires=12, amplitudes=4096)


@pytest.mark.parametrize("record", [
    {"gate": "ccx", "qubits": [0, 1, 2]},
    {"gate": "mcx", "qubits": [0, 1, 2, 3]},
    {"gate": "majority_oracle", "qubits": [0]},
    {"gate": "reset", "qubits": [0]},
    {"gate": "measure", "qubits": [4], "classical": [0]},
    {"gate": "barrier", "qubits": [0]},
    {"gate": "x", "qubits": [0]},
    {"gate": "z", "qubits": [0]},
    {"gate": "ry", "qubits": [0], "parameters_hex": ["0x1p-3"]},
    {"gate": "h", "qubits": [True]},
    {"gate": "h", "qubits": [-1]},
    {"gate": "h", "qubits": [12]},
    {"gate": "h", "qubits": [0, 1]},
    {"gate": "cx", "qubits": [0, 0]},
    {"gate": "cx", "qubits": [0]},
    {"gate": "cx", "qubits": [0, 12]},
    {"gate": "t", "qubits": [0], "parameters_hex": ["0x1p-3"]},
    {"gate": "h", "qubits": [0], "classical": [0]},
    {"gate": "h", "qubits": [0], "unexpected_instruction": "unitary"},
])
def test_component_parser_rejects_opaque_or_out_of_scope_records_without_math(verifier, record):
    with pytest.raises(ValueError):
        verifier.validate_primitive_record(record)


@pytest.mark.parametrize("gate,wires", [("h", [0]), ("cx", [0, 1]), ("t", [11]), ("tdg", [9])])
def test_valid_explicit_component_records_parse_without_any_state(verifier, gate, wires):
    assert verifier.validate_primitive_record({"gate": gate, "qubits": wires}) == (gate, tuple(wires))
    assert verifier.validate_primitive_record({
        "gate": gate, "qubits": wires, "classical": [], "parameters_hex": [],
    }) == (gate, tuple(wires))


def test_signed_input_recipe_is_pure_metadata_and_excludes_uniform_target_plus(verifier):
    recipe = verifier.component_input_recipe()
    assert recipe["state_wires"] == 12 and recipe["state_amplitudes"] == 4096
    assert recipe["syndrome_wires"] == [0, 1, 2, 3]
    assert recipe["target_wires"] == [4, 5, 6, 7, 8]
    assert recipe["scratch_wires"] == [9, 10, 11]
    assert recipe["initial_scratch"] == "000" and recipe["support_size"] == 512
    assert recipe["s_range"] == [0, 15] and recipe["t_range"] == [0, 31]
    assert recipe["index_formula"] == "ell=s+16*t"
    assert recipe["signed_numerator_formula"] == "(-1)^((popcount(s)+popcount(t)) mod 2)*(ell+1)"
    assert recipe["normalization_squared_integer_formula"] == "512*513*1025//6"
    assert recipe["amplitude_formula"] == "signed_numerator/sqrt(normalization_squared)"
    assert recipe["independent_output_index_formula"] == "s+16*(t XOR direct_majority_decode(s))"
    assert recipe["all_support_magnitudes_distinct"] is True
    assert recipe["uniform_target_plus"] is False
    assert recipe["scratch_zero_required"] is True
    assert recipe["executed_by_metadata_function"] is False
    recipe["support_size"] = 1
    assert verifier.component_input_recipe()["support_size"] == 512


def test_signed_input_numerators_have_distinct_magnitudes_without_normalization(verifier):
    recipe = verifier.component_input_recipe()
    assert recipe["executed_by_metadata_function"] is False
    numerators = tuple((-1 if (syndrome.bit_count() + target.bit_count()) % 2 else 1)
                       * (syndrome + 16 * target + 1)
                       for target in range(32) for syndrome in range(16))
    assert len(numerators) == len({abs(value) for value in numerators}) == 512
    assert {value > 0 for value in numerators} == {False, True}
    assert sum(value * value for value in numerators) == 512 * 513 * 1025 // 6
    for syndrome in range(16):
        assert len({abs(numerators[syndrome + 16 * target]) for target in range(32)}) == 32


def test_quantum_source_envelope_reserves_counters_only_and_rejects_duplicates(verifier):
    envelope = verifier._QuantumEnvelope()
    envelope.reserve("actual", 4096, 256)
    envelope.reserve("reference", 4096, 256)
    envelope.reserve("serialized_pairs", 4096, 256)
    assert envelope.total == envelope.peak == 3 * (4096 * 256 + 4096)
    assert envelope.total <= verifier.MAX_QUANTUM_SOURCE_BYTES == 4 << 20
    assert len(envelope.live) == 3
    with pytest.raises(ValueError):
        envelope.reserve("actual", 4096, 256)
    with pytest.raises(ValueError):
        envelope.release("missing")
    for name in ("actual", "reference", "serialized_pairs"):
        envelope.release(name)
    assert envelope.total == 0 and envelope.live == {}


@pytest.mark.parametrize("name,count,width", [
    ("dummy", 4096, 32), ("dense_operator", 4096, 32),
    ("actual", 4095, 32), ("actual", 4097, 32),
    ("actual", True, 32), ("actual", 4096, True),
    ("actual", 4096, 0), ("actual", 4096, 257),
])
def test_quantum_source_buffer_guard_rejects_undeclared_or_widened_roles(verifier, name, count, width):
    envelope = verifier._QuantumEnvelope()
    with pytest.raises(ValueError):
        envelope.reserve(name, count, width)
    assert envelope.live == {} and envelope.total == envelope.peak == 0
