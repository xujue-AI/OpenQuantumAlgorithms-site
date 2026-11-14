"""Pure degree-two input/source-plan boundaries, defined but not executed.

The later frozen pytest command may exercise integer graph checks and primitive
value validation only.  This file never invokes an operation generator, angle
converter, state evolution, Boolean decoder mapping or CCX matrix reference.
"""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import builtins
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/openquantum/optimization_dqi_degree_two.py"
FORBIDDEN_IMPORTS = frozenset({"openquantum", "qiskit", "numpy", "qdk",
                               "nbclient", "ipykernel", "psutil"})


def _load(name):
    """Load a standalone source file without the repository package initializer."""
    original = builtins.__import__

    def guarded(module_name, *args, **kwargs):
        if module_name.split(".", 1)[0] in FORBIDDEN_IMPORTS:
            raise AssertionError("A pure source plan requested a scientific backend")
        return original(module_name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise AssertionError("Standalone degree-two source is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original
    return module


CORE = _load("_degree_two_pure_source_plan_tests")
C5 = (5, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)), (1, 1, 1, 1, 1))
FIGURE_EIGHT = (9, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                    (0, 5), (5, 6), (6, 7), (7, 8), (8, 0)), (1,) * 10)


def _plan(limits=None, **kwargs):
    return CORE.make_plan(*C5, limits=limits, **kwargs)


def test_standalone_import_and_classical_plan_request_no_backend_or_trigonometry(monkeypatch):
    pure = _load("_degree_two_pure_reload_guard")

    def no_trigonometry(*args, **kwargs):
        pytest.fail("A classical source plan evaluated a rotation")

    monkeypatch.setattr(pure, "math", SimpleNamespace(
        isfinite=math.isfinite, gcd=math.gcd, comb=math.comb,
        asin=no_trigonometry, sqrt=no_trigonometry,
        sin=no_trigonometry, cos=no_trigonometry,
    ))
    plan = pure.make_plan(*C5)
    assert (plan.N, plan.n, plan.m) == (5, 4, 5)
    assert plan.normalization_squared == 190


def test_five_cycle_uses_graph_vertices_and_four_gauge_variables():
    plan = _plan()
    assert (plan.N, plan.n, plan.m, plan.source_wires) == (5, 4, 5, 12)
    assert CORE.c5_input() == C5
    assert plan.normalization_squared == 190


def test_five_cycle_preflight_charges_reset_readout_and_unsynthesized_rotations():
    summary = CORE.plan_summary(_plan())
    assert summary["source_counts"] == {
        "h": 222, "cx": 686, "t": 436, "tdg": 327, "x": 80,
        "z": 5, "ry": 17, "reset": 12, "measure": 4, "barrier": 7,
    }
    assert summary["source_unitary_primitives"] == 1773
    assert summary["source_materialized_operations"] == 1796


def test_generation_only_figure_eight_has_its_own_width_and_counts():
    plan = CORE.make_plan(*FIGURE_EIGHT)
    summary = CORE.plan_summary(plan)
    assert CORE.figure_eight_input() == FIGURE_EIGHT
    assert (plan.N, plan.n, plan.m, plan.source_wires) == (9, 8, 10, 24)
    assert summary["source_unitary_primitives"] == 12981
    assert summary["source_materialized_operations"] == 13020
    assert summary["source_counts"]["ry"] == 37
    assert summary.get("physical_resources_computed") is False
    assert summary.get("sampled_outcomes") is False
    assert "optimum_probability" not in summary
    assert "fixed_complete_draws" not in summary


def test_decoder_compilation_stores_sparse_supports_with_all_nonzero_syndromes():
    plan = _plan()
    entries = plan.lookup_entries
    assert len(entries) == 15
    assert [mask for mask, _ in entries] == list(range(1, 16))
    rows = (1, 3, 6, 12, 8)
    for mask, support in entries:
        assert type(mask) is int and type(support) is tuple
        assert len(support) in (1, 2)
        assert tuple(sorted(support)) == support
        assert all(type(edge) is int and 0 <= edge < 5 for edge in support)
        expected = 0
        for edge in support:
            expected ^= rows[edge]
        assert mask == expected


def test_classical_graph_girth_and_code_lookup_checks_are_separately_charged():
    counts = _plan().preprocessing_counts
    assert counts["edge_endpoint_records"] == 10
    assert counts["sign_records"] == 5
    assert counts["connectivity_vertices_visited"] == 5
    assert counts["connectivity_adjacency_entries"] == 10
    assert counts["wedge_entries"] == counts["triangle_membership_queries"] == 5
    assert counts["D2_masks_including_vacuum"] == 16
    assert counts["adjacent_D2_collision_checks"] == counts["lookup_entries"] == 15
    assert counts["integer_rotation_ratios"] == 9


def test_counter_preparation_ratios_remain_exact_integers_without_angle_conversion():
    ratios = {(i, counter): Fraction(numerator, denominator)
              for i, counter, numerator, denominator in _plan().rotation_ratios}
    assert ratios == {
        (0, 0): Fraction(41, 190),
        (1, 0): Fraction(37, 149), (1, 1): Fraction(4, 41),
        (2, 0): Fraction(33, 112), (2, 1): Fraction(4, 37),
        (3, 0): Fraction(29, 79), (3, 1): Fraction(4, 33),
        (4, 0): Fraction(1, 2), (4, 1): Fraction(4, 29),
    }


def test_summary_reuses_the_prepared_plan_without_another_graph_lookup_pass(monkeypatch):
    plan = _plan()
    monkeypatch.setattr(CORE, "make_plan", lambda *a, **k: pytest.fail("The graph plan was rebuilt"))
    assert CORE.plan_summary(plan)["source_materialized_operations"] == 1796


def test_plan_is_immutable_and_summary_mutation_does_not_change_it():
    plan = _plan()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        plan.n = 3
    summary = CORE.plan_summary(plan)
    summary["source_counts"]["ry"] = 0
    assert CORE.plan_summary(plan)["source_counts"]["ry"] == 17


def test_ordered_edges_and_their_target_bits_are_not_sorted_apart():
    edges = ((4, 0), (3, 4), (2, 3), (1, 2), (0, 1))
    signs = (1, 0, 1, 0, 0)
    plan = CORE.make_plan(5, edges, signs)
    assert plan.edges == edges
    assert plan.signs == signs
    assert CORE.plan_summary(plan)["source_counts"]["z"] == 2


@pytest.mark.parametrize("vertices", [True, 5.0, "5", None, 4, 66])
def test_graph_vertex_domain_rejects_boolean_alias_and_unsupported_sizes(vertices):
    with pytest.raises(ValueError):
        CORE.make_plan(vertices, C5[1], C5[2])


@pytest.mark.parametrize("edges", [list(C5[1]), (), C5[1][:-1],
    ((0, 1, 2),) + C5[1][1:], ([0, 1],) + C5[1][1:],
    ((True, 1),) + C5[1][1:], ((-1, 1),) + C5[1][1:],
    ((0, 5),) + C5[1][1:]])
def test_ordered_edge_shape_and_endpoint_domain_are_strict(edges):
    with pytest.raises(ValueError):
        CORE.make_plan(5, edges, C5[2])


@pytest.mark.parametrize("signs", [list(C5[2]), (), C5[2][:-1],
    (True, 1, 1, 1, 1), (0.0, 1, 1, 1, 1),
    (2, 1, 1, 1, 1), (None, 1, 1, 1, 1)])
def test_one_exact_binary_target_per_edge_is_required(signs):
    with pytest.raises(ValueError):
        CORE.make_plan(5, C5[1], signs)


@pytest.mark.parametrize("first", [(0, 0), (0, 4)])
def test_loop_and_reverse_duplicate_reject_before_source_emission(first):
    edges = (first,) + C5[1][1:]
    with pytest.raises(ValueError):
        CORE.make_plan(5, edges, C5[2])


def test_connectivity_is_checked_before_accepting_a_cycle_code():
    edges = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
    with pytest.raises(ValueError, match="connect"):
        CORE.make_plan(6, edges, (1,) * 6)


@pytest.mark.parametrize("edges", [
    ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4)),
    ((0, 1), (1, 2), (2, 3), (3, 0), (3, 4)),
])
def test_triangle_or_four_cycle_cannot_supply_degree_two_uniqueness(edges):
    with pytest.raises(ValueError):
        CORE.make_plan(5, edges, (1,) * 5)


@pytest.mark.parametrize("field,value", [
    ("max_n", 0), ("max_n", True), ("max_n", 65),
    ("max_m", 0), ("max_m", 65),
    ("max_qubits", 0), ("max_qubits", 193),
    ("max_operations", 0), ("max_operations", 200001),
    ("max_export_bytes", 0), ("max_export_bytes", 33554433),
    ("api_wall_seconds", 0), ("api_wall_seconds", True),
    ("api_wall_seconds", 31), ("api_wall_seconds", math.inf),
    ("api_wall_seconds", math.nan),
])
def test_limits_cannot_relax_hard_caps_or_use_boolean_aliases(field, value):
    with pytest.raises(ValueError):
        CORE.DegreeTwoLimits(**{field: value})


@pytest.mark.parametrize("limits", [
    {"max_n": 3}, {"max_m": 4}, {"max_qubits": 11},
    {"max_operations": 1795},
])
def test_caller_can_tighten_caps_and_reject_before_any_rotation(limits, monkeypatch):
    monkeypatch.setattr(CORE, "math", SimpleNamespace(
        isfinite=math.isfinite, gcd=math.gcd, comb=math.comb,
        asin=lambda *a, **k: pytest.fail("Rotation evaluated after failing preflight"),
        sqrt=lambda *a, **k: pytest.fail("Rotation evaluated after failing preflight"),
    ))
    with pytest.raises(ValueError):
        _plan(CORE.DegreeTwoLimits(**limits))


def test_exact_materialization_and_wire_limits_accept_the_classical_plan():
    plan = _plan(CORE.DegreeTwoLimits(max_n=4, max_m=5,
                                     max_qubits=12, max_operations=1796))
    assert plan.source_wires == 12
    assert CORE.plan_summary(plan)["source_materialized_operations"] == 1796


def test_expired_caller_deadline_rejects_before_graph_or_angle_work(monkeypatch):
    monkeypatch.setattr(CORE, "time", SimpleNamespace(monotonic=lambda: 10.0))
    with pytest.raises((TimeoutError, ValueError)):
        _plan(deadline=10.0)


@pytest.mark.parametrize("name", ["ccx", "mcx", "unitary", "decoder_oracle",
                                  "sample", "measure_all"])
def test_primitive_values_do_not_accept_opaque_or_sampled_work(name):
    with pytest.raises(ValueError):
        CORE.Operation(name, (0,))


@pytest.mark.parametrize("name,wires", [
    ("h", ()), ("h", (0, 1)), ("cx", (0,)), ("cx", (0, 1, 2)),
    ("barrier", ()), ("cx", (0, 0)), ("h", [0]),
    ("h", (True,)), ("h", (-1,)), ("h", (192,)),
])
def test_primitive_arity_wire_type_and_distinctness_are_checked(name, wires):
    with pytest.raises(ValueError):
        CORE.Operation(name, wires)


@pytest.mark.parametrize("parameters", [(math.nan,), (math.inf,),
    (-math.inf,), (True,), ("0.1",), (), (0.1, 0.2)])
def test_rotation_value_guards_use_only_finite_single_scalars(parameters):
    with pytest.raises(ValueError):
        CORE.Operation("ry", (0,), parameters=parameters)


@pytest.mark.parametrize("name,classical,parameters", [
    ("h", (0,), ()), ("h", (), (0.1,)),
    ("measure", (), ()), ("measure", (True,), ()),
    ("measure", (64,), ()), ("measure", (0, 1), ()),
])
def test_readout_and_rotation_metadata_cannot_hide_in_other_primitives(name, classical, parameters):
    with pytest.raises(ValueError):
        CORE.Operation(name, (0,), classical=classical, parameters=parameters)


def test_single_primitive_record_preserves_binary_parameter_and_readout_index():
    rotation = CORE.Operation("ry", (0,), parameters=(0.125,))
    assert rotation.record() == {"gate": "ry", "qubits": [0],
                                 "parameters_hex": [float(0.125).hex()]}
    readout = CORE.Operation("measure", (5,), classical=(0,))
    assert readout.record() == {"gate": "measure", "qubits": [5], "classical": [0]}


@pytest.mark.parametrize("target,draws,failure", [
    (Fraction(1, 100), 2, Fraction(1, 361)),
    (Fraction(1, 400), 3, Fraction(1, 6859)),
    (Fraction(1, 19), 1, Fraction(1, 19)),
])
def test_c5_ideal_draw_plan_matches_exact_success_and_whole_budget_targets(target, draws, failure):
    plan = CORE.c5_ideal_draw_plan(target)
    assert plan["draws"] == draws
    assert plan["failure"] == failure <= target
    assert plan["success"] == 1 - failure
    assert plan["p_opt"] == Fraction(18, 19)
    assert Fraction(1, 19) ** (draws - 1) > target
    assert plan["scope"] == "fixed_C5_analytic_ideal_reference_only"
    for name in ("sampled_outcomes", "quantum_execution", "physical_resources_computed", "validated"):
        assert plan[name] is False


def test_ideal_draw_cap_reports_shortfall_instead_of_accepting_a_weaker_target():
    with pytest.raises(ValueError):
        CORE.c5_ideal_draw_plan(Fraction(1, 100), max_draws=1)


@pytest.mark.parametrize("target", [False, 0, 0.01, "1/100", Fraction(0),
    Fraction(1), Fraction(-1, 2), Fraction(2),
    Fraction(1, 1 << 256), Fraction((1 << 256) + 1, (1 << 257) + 1)])
def test_ideal_draw_target_requires_native_bounded_fraction(target):
    with pytest.raises(ValueError):
        CORE.c5_ideal_draw_plan(target)


@pytest.mark.parametrize("max_draws", [True, 0, 1025, 1.0, "3"])
def test_ideal_draw_count_bound_does_not_accept_aliases_or_relaxations(max_draws):
    with pytest.raises(ValueError):
        CORE.c5_ideal_draw_plan(Fraction(1, 100), max_draws=max_draws)


def test_ideal_draw_result_is_fresh_and_does_not_mark_execution():
    first = CORE.c5_ideal_draw_plan(Fraction(1, 100))
    first["draws"] = 1
    first["validated"] = True
    second = CORE.c5_ideal_draw_plan(Fraction(1, 100))
    assert second["draws"] == 2 and second["validated"] is False
