"""Pure path-majority source tests, authored but not executed.

An independently authorized pytest command may later perform integer graph,
Boolean decoder and classical objective checks. These tests never emit the
full RY program, allocate quantum states, enable numerical verification or run
a kernel. Collection itself does not load the new target source.
"""
from collections import Counter
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import combinations
import ast
import builtins
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/openquantum/optimization_dqi_path_majority.py"
FORBIDDEN_IMPORTS = frozenset({
    "openquantum", "qiskit", "numpy", "scipy", "qdk", "nbclient",
    "ipykernel", "psutil", "torch", "cupy",
})
SIGNS = (1, 1, 1, 1, 0) + (1,) * 10


def _load_standalone(name):
    original = builtins.__import__

    def guarded(module_name, *args, **kwargs):
        if module_name.split(".", 1)[0] in FORBIDDEN_IMPORTS:
            raise AssertionError("A pure source test requested a scientific backend")
        return original(module_name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise AssertionError("Standalone path-majority source is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original
    return module


@pytest.fixture(scope="module")
def core():
    name = "_path_majority_pure_source_tests"
    module = _load_standalone(name)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


def _independent_edges(k):
    edges = []
    for path in range(k):
        vertices = (0, *(2 + 4 * path + j for j in range(4)), 1)
        edges.extend(zip(vertices[:-1], vertices[1:]))
    return tuple(edges)


def _independent_syndrome(edges, errors, vertices):
    bits = [0] * (vertices - 1)
    for (a, b), error in zip(edges, errors, strict=True):
        if error:
            for vertex in (a, b):
                if vertex:
                    bits[vertex - 1] ^= 1
    return tuple(bits)


def _independent_local_decoder(syndrome):
    prefix = [0]
    for bit in syndrome:
        prefix.append(prefix[-1] ^ bit)
    first_edge = int(sum(prefix) >= 3)
    return tuple(first_edge ^ bit for bit in prefix)


def _ccx_word(a, b, target):
    """Independent exact phase-preserving lowering, compared as source bytes."""
    return (
        ("h", (target,)), ("cx", (b, target)), ("tdg", (target,)),
        ("cx", (a, target)), ("t", (target,)), ("cx", (b, target)),
        ("tdg", (target,)), ("cx", (a, target)), ("t", (b,)),
        ("t", (target,)), ("h", (target,)), ("cx", (a, b)),
        ("t", (a,)), ("tdg", (b,)), ("cx", (a, b)),
    )


def _fold_component(records):
    """Recognize complete exact CCX words before Boolean folding their action."""
    sequence = tuple((record["gate"], tuple(record["qubits"])) for record in records)
    logic = []
    position = 0
    while position < len(sequence):
        gate, wires = sequence[position]
        if gate == "h":
            assert position + 15 <= len(sequence)
            target = wires[0]
            a = sequence[position + 12][1][0]
            b = sequence[position + 1][1][0]
            assert len({a, b, target}) == 3
            assert sequence[position:position + 15] == _ccx_word(a, b, target)
            logic.append(("ccx", (a, b, target)))
            position += 15
        else:
            assert gate == "cx" and len(wires) == 2
            logic.append((gate, wires))
            position += 1
    return tuple(logic)


def _apply_boolean(logic, initial):
    bits = list(initial)
    for gate, wires in logic:
        if gate == "ccx":
            a, b, target = wires
            bits[target] ^= bits[a] & bits[b]
        else:
            a, target = wires
            bits[target] ^= bits[a]
    return tuple(bits)


def _score(edges, signs, assignment):
    return sum((assignment[a] ^ assignment[b]) == sign
               for (a, b), sign in zip(edges, signs, strict=True))


def test_plan_is_classical_and_does_not_evaluate_rotations(core, monkeypatch):
    def no_rotation(*args, **kwargs):
        pytest.fail("A source preflight evaluated a rotation")

    monkeypatch.setattr(core, "math", SimpleNamespace(
        isfinite=math.isfinite, gcd=math.gcd, comb=math.comb,
        asin=no_rotation, sqrt=no_rotation, sin=no_rotation, cos=no_rotation,
    ))
    plan = core.make_plan(3, SIGNS)
    assert (plan.k, plan.N, plan.n, plan.m, plan.source_wires) == (3, 14, 13, 15, 31)
    assert plan.shared_clean_workspace == 3
    assert plan.per_pattern_coefficients == (15, 15, 2)
    assert plan.normalization_squared == 4020


@pytest.mark.parametrize("k", [3, 4, 12])
def test_family_has_two_terminals_and_four_internal_vertices_per_path(core, k):
    edges = _independent_edges(k)
    assert core.family_edges(k) == edges
    plan = core.make_plan(k, (0,) * (5 * k))
    assert plan.edges == edges
    assert (plan.N, plan.n, plan.m, plan.source_wires) == (2 + 4 * k, 1 + 4 * k, 5 * k, 9 * k + 4)
    degrees = Counter(vertex for edge in edges for vertex in edge)
    assert degrees[0] == degrees[1] == k
    assert all(degrees[vertex] == 2 for vertex in range(2, plan.N))


def test_fixed_fixture_has_mixed_path_parities_and_complete_source_prediction(core):
    assert core.fixture_input() == (3, SIGNS)
    plan = core.make_plan(*core.fixture_input())
    assert dict(plan.source_counts) == {
        "h": 385, "cx": 1272, "t": 744, "tdg": 558, "x": 84,
        "ry": 57, "z": 14, "reset": 31, "measure": 13, "barrier": 7,
    }
    assert plan.source_unitary_primitives == 3114
    assert plan.source_materialized_operations == 3165
    assert tuple(sum(SIGNS[5 * path:5 * path + 5]) % 2 for path in range(3)) == (0, 1, 1)


def test_source_summary_retains_the_weight_two_promise_and_unexecuted_scope(core):
    summary = core.plan_summary(core.make_plan(3, SIGNS))
    assert summary["source_prediction_only"] is True
    assert summary["scientific_validated"] is False
    assert summary["full_program_validated"] is False
    assert summary["publication_complete"] is False
    assert summary["decoder"]["scope"] == "weight_at_most_two"
    assert summary["decoder"]["total_extension"] == "local_majority_ignores_terminal"
    assert summary["graph"]["girth"] == 10
    assert summary["graph"]["cycle_space_dimension"] == 2
    assert summary["graph"]["every_neighbor_recovery_claimed"] is False
    assert summary["scaling"]["decoder_lookup_entries"] == 0
    assert summary["repetitions"]["generic_success_probability_available"] is False
    assert summary["repetitions"]["generic_repetition_bound_available"] is False
    assert all(value is True for value in summary["exclusions"].values())
    component = core.component_summary()
    assert component["source_wires"] == 12
    assert component["syndrome_wires"] == [0, 1, 2, 3]
    assert component["target_wires"] == [4, 5, 6, 7, 8]
    assert component["scratch_wires"] == [9, 10, 11]
    assert component["source_prediction_only"] is True
    assert component["component_executed"] is False
    assert component["scientific_validated"] is False
    assert component["full_program_validated"] is False
    assert component["rotation_synthesis_available"] is False
    assert component["physical_resources_available"] is False


@pytest.mark.parametrize("k", [3, 4, 8, 12])
def test_source_scaling_charges_preparation_decoder_signs_syndrome_and_readout(core, k):
    signs = (0,) * (5 * k)
    plan = core.make_plan(k, signs)
    ccx_blocks = 6 * (plan.m - 1) + 34 * k
    counts = dict(plan.source_counts)
    assert counts["h"] == 2 * ccx_blocks + plan.n
    assert counts["t"] == 4 * ccx_blocks
    assert counts["tdg"] == 3 * ccx_blocks
    assert counts["cx"] == 6 * ccx_blocks + 6 * (plan.m - 1) + 15 * k + 9 * k
    assert counts["x"] == 6 * (plan.m - 1)
    assert counts["ry"] == 4 * plan.m - 3
    assert counts["z"] == 0
    assert counts["reset"] == plan.source_wires and counts["measure"] == plan.n
    assert counts["barrier"] == 7
    assert plan.source_unitary_primitives == 1068 * k - 104
    assert plan.source_materialized_operations == plan.source_unitary_primitives + plan.source_wires + plan.n + 7


def test_weight_two_branch_ratios_are_exact_integer_preflight(core):
    plan = core.make_plan(3, SIGNS)
    coefficients = (plan.m, plan.m, 2)

    def branch_mass(remaining, prefix_weight):
        return sum(math.comb(remaining, tail) * coefficients[prefix_weight + tail] ** 2
                   for tail in range(3 - prefix_weight))

    ratios = {(position, prefix): Fraction(numerator, denominator)
              for position, prefix, numerator, denominator in plan.rotation_ratios}
    assert len(ratios) == 2 * plan.m - 1
    for (position, prefix), ratio in ratios.items():
        assert ratio == Fraction(branch_mass(plan.m - position - 1, prefix + 1),
                                 branch_mass(plan.m - position, prefix))


def test_component_has_34_exact_ccx_blocks_and_15_cx_without_full_preparation(core):
    records = tuple(operation.record() for operation in core.iter_component_operations())
    assert len(records) == 525
    assert Counter(record["gate"] for record in records) == {
        "h": 68, "cx": 219, "t": 136, "tdg": 102,
    }
    assert all(set(record) == {"gate", "qubits"} for record in records)
    assert all(0 <= wire < 12 for record in records for wire in record["qubits"])
    logic = _fold_component(records)
    assert Counter(gate for gate, _ in logic) == {"ccx": 34, "cx": 15}


def test_all_local_syndromes_preserve_arbitrary_targets_and_clean_scratch(core):
    logic = _fold_component(tuple(operation.record() for operation in core.iter_component_operations()))
    checked = 0
    for syndrome_integer in range(16):
        syndrome = tuple((syndrome_integer >> bit) & 1 for bit in range(4))
        decoded = _independent_local_decoder(syndrome)
        assert core.decode_path_bits(syndrome) == decoded
        for target_integer in range(32):
            target = tuple((target_integer >> bit) & 1 for bit in range(5))
            actual = _apply_boolean(logic, syndrome + target + (0, 0, 0))
            assert actual[:4] == syndrome
            assert actual[4:9] == tuple(bit ^ error for bit, error in zip(target, decoded, strict=True))
            assert actual[9:] == (0, 0, 0)
            checked += 1
    assert checked == 512


def test_all_121_global_weight_two_inputs_reconstruct_with_the_terminal_bit_ignored(core):
    plan = core.make_plan(3, SIGNS)
    checked = 0
    for weight in (0, 1, 2):
        for support in combinations(range(plan.m), weight):
            errors = tuple(int(edge in support) for edge in range(plan.m))
            syndrome = _independent_syndrome(plan.edges, errors, plan.N)
            mask = sum(bit << index for index, bit in enumerate(syndrome))
            assert core.reduced_incidence_syndrome(3, errors) == mask
            assert core.decode_syndrome(3, syndrome) == errors
            inconsistent_terminal = (syndrome[0] ^ 1,) + syndrome[1:]
            assert core.decode_syndrome(3, inconsistent_terminal) == errors
            checked += 1
    assert checked == 121


def test_three_errors_have_the_specified_weight_two_complement_and_wrong_terminal(core):
    errors = (0, 1, 1, 1, 0)
    syndrome = tuple(a ^ b for a, b in zip(errors[:-1], errors[1:], strict=True))
    assert syndrome == (1, 0, 0, 1)
    decoded = core.decode_path_bits(syndrome)
    assert decoded == (1, 0, 0, 0, 1)
    assert decoded == tuple(bit ^ 1 for bit in errors)
    assert decoded[-1] != errors[-1]
    global_errors = errors + (0,) * 10
    global_decoded = decoded + (0,) * 10
    edges = _independent_edges(3)
    before = _independent_syndrome(edges, global_errors, 14)
    after = _independent_syndrome(edges, global_decoded, 14)
    assert before[1:] == after[1:] and before[0] != after[0]
    assert core.decode_syndrome(3, before) == global_decoded


def test_full_path_kernel_words_require_an_even_path_count_and_have_weight_ten(core):
    edges = _independent_edges(3)
    nonzero_weights = []
    for active_paths in range(1, 8):
        errors = tuple((active_paths >> (edge // 5)) & 1 for edge in range(15))
        syndrome = _independent_syndrome(edges, errors, 14)
        assert all(bit == 0 for bit in syndrome[1:])
        assert syndrome[0] == active_paths.bit_count() % 2
        if not any(syndrome):
            nonzero_weights.append(sum(errors))
    assert nonzero_weights and min(nonzero_weights) == 10


def test_linear_classical_comparator_matches_an_independent_full_k3_scan(core):
    result = core.classical_optimum(3, SIGNS)
    edges = _independent_edges(3)
    optimum = max(_score(edges, SIGNS, (0,) + tuple((word >> bit) & 1 for bit in range(13)))
                  for word in range(1 << 13))
    assert result["score"] == optimum == 14
    assert _score(edges, SIGNS, result["assignment"]) == result["score"]
    assert result["assignment"][0] == 0 and result["terminal_bit"] == 1
    assert result["path_parities"] == (0, 1, 1)
    assert result["minimum_violations"] == 1
    assert result["no_advantage_claim"] is True


def test_classical_comparator_uses_deterministic_tie_and_first_edge_violation(core):
    signs = (0,) * 10 + (1, 0, 0, 0, 0) * 2
    result = core.classical_optimum(4, signs)
    assert result["terminal_bit"] == 0 and result["score"] == 18
    assignment = result["assignment"]
    violations = tuple(index for index, ((a, b), sign) in enumerate(zip(_independent_edges(4), signs, strict=True))
                       if (assignment[a] ^ assignment[b]) != sign)
    assert violations == (10, 15)
    assert result["gauge_assignment"] == assignment[1:]
    assert result["classical_work"] == {
        "path_parity_xors": 20, "assignment_internal_steps": 16,
        "independent_edge_scores": 20, "full_vertex_output_writes": 18,
    }


def test_classical_optimum_accepts_only_the_canonical_edge_sign_pairing(core):
    edges = _independent_edges(3)
    assert core.classical_optimum(3, SIGNS, edges=edges)["edges"] == edges
    with pytest.raises(ValueError):
        core.classical_optimum(3, SIGNS, edges=edges[::-1])
    with pytest.raises(ValueError):
        core.classical_optimum(3, SIGNS, edges=list(edges))


def test_plan_and_source_prediction_are_immutable(core):
    plan = core.make_plan(3, SIGNS)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        plan.k = 4
    with pytest.raises(TypeError):
        plan.source_counts["ry"] = 0
    summary = core.plan_summary(plan)
    summary["source_counts"]["ry"] = 0
    assert dict(plan.source_counts)["ry"] == 57
    assert core.plan_summary(plan)["source_counts"]["ry"] == 57


def test_a_mutated_or_unsealed_plan_is_rejected_by_summary_before_emission(core):
    with pytest.raises(ValueError):
        core.plan_summary(SimpleNamespace())
    plan = core.make_plan(3, SIGNS)
    object.__setattr__(plan, "k", 4)
    with pytest.raises(ValueError):
        core.plan_summary(plan)


@pytest.mark.parametrize("k", [False, True, 3.0, "3", None, 0, 2, 13])
def test_family_size_requires_a_supported_strict_integer(core, k):
    with pytest.raises(ValueError):
        core.make_plan(k, SIGNS)
    with pytest.raises(ValueError):
        core.family_edges(k)


@pytest.mark.parametrize("signs", [
    list(SIGNS), (), SIGNS[:-1], SIGNS + (0,),
    (True,) + SIGNS[1:], (1.0,) + SIGNS[1:], (2,) + SIGNS[1:],
    (-1,) + SIGNS[1:], (None,) + SIGNS[1:],
])
def test_every_edge_requires_one_strict_binary_target(core, signs):
    with pytest.raises(ValueError):
        core.make_plan(3, signs)
    with pytest.raises(ValueError):
        core.classical_optimum(3, signs)


@pytest.mark.parametrize("field,hard", [
    ("max_k", 12), ("max_qubits", 112), ("max_operations", 20000),
    ("max_export_bytes", 8388608),
])
@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.0, "1", None])
def test_limit_fields_reject_boolean_aliases_and_nonpositive_or_noninteger_values(core, field, hard, bad):
    with pytest.raises(ValueError):
        core.PathMajorityLimits(**{field: bad})
    with pytest.raises(ValueError):
        core.PathMajorityLimits(**{field: hard + 1})


@pytest.mark.parametrize("allowance", [True, False, 0, -1, 30.01, "1", None, float("nan"), float("inf")])
def test_api_allowance_is_finite_positive_and_cannot_widen_its_ceiling(core, allowance):
    with pytest.raises(ValueError):
        core.PathMajorityLimits(api_wall_seconds=allowance)


@pytest.mark.parametrize("limits", [
    {"max_k": 2}, {"max_qubits": 30}, {"max_operations": 3164},
])
def test_tighter_limits_reject_the_full_fixture_before_emission(core, limits):
    with pytest.raises(ValueError):
        core.make_plan(3, SIGNS, limits=core.PathMajorityLimits(**limits))


@pytest.mark.parametrize("syndrome", [
    [0, 0, 0, 0], (), (0, 0, 0), (0, 0, 0, 0, 0),
    (True, 0, 0, 0), (1.0, 0, 0, 0), (2, 0, 0, 0),
])
def test_local_total_decoder_has_a_strict_four_bit_domain(core, syndrome):
    with pytest.raises(ValueError):
        core.decode_path_bits(syndrome)


@pytest.mark.parametrize("gate,qubits,kwargs", [
    ("ccx", (0, 1, 2), {}), ("majority_oracle", (0,), {}),
    (True, (0,), {}), ("h", [0], {}),
    ("h", (True,), {}), ("h", (-1,), {}), ("h", (112,), {}),
    ("cx", (0, 0), {}), ("cx", (0,), {}), ("barrier", (), {}),
    ("h", (0,), {"classical": (0,)}),
    ("measure", (0,), {"classical": (49,)}),
    ("ry", (0,), {"parameters": (True,)}),
    ("ry", (0,), {"parameters": (float("nan"),)}),
])
def test_operation_value_guard_rejects_opaque_or_malformed_primitives(core, gate, qubits, kwargs):
    with pytest.raises(ValueError):
        core.Operation(gate, qubits, **kwargs)


def test_source_tests_have_no_call_to_full_generation_or_numerical_verification():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"iter_operations", "iter_ndjson", "verify_component_records", "_angle_from_ratio"}
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not (calls & forbidden)
