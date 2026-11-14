"""Pure source tests for the single predeclared word-shuttle transformation."""
from copy import deepcopy
import hashlib
import json

import pytest

from openquantum.optimization_word_transport import build_candidate, operation_digest


def local(name, wire):
    return [name, [wire], [], []]


def measured(body, mapping=(2, 0, 3, 1)):
    return ([local("reset", wire) for wire in range(4)] + body
            + [["measure", [wire], [mapping[wire]], []] for wire in range(4)])


@pytest.mark.parametrize("wire", [1, 2, 3])
@pytest.mark.parametrize("magic_count", [3, 4, 17])
def test_single_word_charges_both_routes_and_retains_fresh_magic_demand(wire, magic_count):
    body = [local("h", wire)]
    for index in range(magic_count):
        body.extend([local("t" if index % 2 else "tdg", wire), local("s", wire)])
    original = measured(body)
    unchanged = deepcopy(original)
    result = build_candidate(original)
    assert original == unchanged
    assert result["status"] == "source_only_not_physically_executed"
    assert result["physical_execution_performed"] is False
    assert result["changed"] is True
    assert result["source_operations"] == len(original)
    assert result["candidate_operations"] == len(original) + 6 * wire
    assert result["baseline_magic_transport_swaps"] == magic_count * (wire + 1)
    assert result["candidate_magic_transport_swaps"] == magic_count
    assert result["added_data_transport_swaps"] == 2 * wire
    assert result["net_removed_transport_swaps"] == wire * (magic_count - 2)
    assert result["added_data_transport_cx"] == 6 * wire
    assert result["removed_magic_transport_cx"] == 3 * wire * magic_count
    assert result["net_removed_cx"] == 3 * wire * (magic_count - 2)
    assert result["candidate_event_counts"]["cx"] == 6 * wire
    assert result["candidate_event_counts"]["initial_reset"] == 4
    assert result["candidate_event_counts"]["measure"] == 4
    assert result["measurement_map"] == [[0, 2], [1, 0], [2, 3], [3, 1]]
    segment = result["segments"][0]
    assert segment["source_start"] == 4 and segment["source_stop"] == 4 + len(body)
    assert segment["t_count"] == magic_count // 2
    assert segment["tdg_count"] == (magic_count + 1) // 2
    assert segment["magic_count"] == magic_count
    assert segment["candidate_start"] == 4 and segment["candidate_stop"] == 4 + len(body) + 6 * wire
    assert segment["boundary_data_permutation"] == [0, 1, 2, 3]
    assert segment["spectator_zero_assumption"] is False
    assert segment["initialized_data_path"] == list(range(wire, -1, -1))
    forward = segment["forward_swap_edges"]
    assert forward == [[right, right - 1] for right in range(wire, 0, -1)]
    assert segment["reverse_swap_edges"] == list(reversed(forward))
    segment_ops = result["operations"][segment["candidate_start"]:segment["candidate_stop"]]
    assert operation_digest(segment_ops) == segment["candidate_segment_sha256"]
    assert operation_digest(body) == segment["source_word_sha256"]
    assert all(abs(item[1][0] - item[1][1]) == 1 for item in segment_ops if item[0] == "cx")
    # Removing the explicitly charged shuttle CXs leaves exactly the saved
    # local gate names and order, all performed at the fixed data-zero port.
    moved_word = [item for item in segment_ops if item[0] != "cx"]
    assert [item[0] for item in moved_word] == [item[0] for item in body]
    assert all(item[1] == [0] for item in moved_word)
    assert result["operations"][:4] == original[:4]
    assert result["operations"][-4:] == original[-4:]


@pytest.mark.parametrize("wire,magic_count", [(0, 3), (0, 17), (1, 0), (2, 1), (3, 2)])
def test_nonpositive_savings_words_are_retained_exactly(wire, magic_count):
    original = measured([local("h", wire)] + [local("t", wire)] * magic_count)
    result = build_candidate(original)
    assert result["changed"] is False
    assert result["operations"] == original and result["operations"] is not original
    assert result["baseline_event_counts"] == result["candidate_event_counts"]
    assert result["net_removed_cx"] == result["added_data_transport_cx"] == 0
    assert result["segments"][0]["shuttled"] is False


@pytest.mark.parametrize("boundary", [
    ["barrier", [0, 1, 2, 3], [], []],
    ["cx", [2, 3], [], []],
    ["h", [0], [], []],
])
def test_no_source_boundary_is_crossed_to_create_an_eligible_word(boundary):
    original = measured([local("t", 3), local("t", 3), boundary, local("t", 3)])
    result = build_candidate(original)
    assert result["changed"] is False
    assert result["operations"] == original
    assert not any(segment["shuttled"] for segment in result["segments"])


def test_two_words_restore_order_before_the_intervening_source_cx():
    boundary = ["cx", [1, 2], [], []]
    original = measured([local("t", 3)] * 3 + [boundary] + [local("tdg", 2)] * 4)
    result = build_candidate(original)
    assert len(result["segments"]) == 2
    first, second = result["segments"]
    assert result["operations"][first["candidate_stop"]] == boundary
    assert second["candidate_start"] == first["candidate_stop"] + 1
    assert result["added_data_transport_cx"] == 6 * (3 + 2)
    assert result["removed_magic_transport_cx"] == 3 * (3 * 3 + 2 * 4)
    assert result["net_removed_cx"] == 3 * (3 * 1 + 2 * 2)
    assert result["candidate_event_counts"]["cx"] == 1 + 6 * (3 + 2)
    assert result["candidate_event_counts"]["t_q0"] == 3
    assert result["candidate_event_counts"]["tdg_q0"] == 4


def test_frozen_reset_and_readout_barriers_are_retained_in_exact_source_order():
    prefix = []
    for wire in range(4):
        prefix.extend([["barrier", [0, 1, 2, 3], [], []], local("reset", wire),
                       ["barrier", [0, 1, 2, 3], [], []]])
    suffix = []
    for wire, clbit in ((0, 0), (3, 1), (2, 2), (1, 3)):
        suffix.extend([["barrier", [0, 3, 2, 1], [], []],
                       ["measure", [wire], [clbit], []],
                       ["barrier", [0, 3, 2, 1], [], []]])
    original = prefix + [local("t", 3)] * 3 + suffix
    result = build_candidate(original)
    assert result["changed"] is True
    assert result["operations"][:len(prefix)] == prefix
    assert result["operations"][-len(suffix):] == suffix
    assert [item for item in result["operations"] if item[0] == "barrier"] == [
        item for item in original if item[0] == "barrier"]
    assert result["reset_wire_order"] == [0, 1, 2, 3]
    assert result["measurement_map"] == [[0, 0], [3, 1], [2, 2], [1, 3]]
    assert result["baseline_event_counts"]["initial_reset"] == 4
    assert result["candidate_event_counts"]["initial_reset"] == 4
    assert result["segments"][0]["source_start"] == len(prefix)
    assert result["segments"][0]["source_stop"] == len(prefix) + 3


def test_interleaved_barriers_do_not_hide_invalid_initialization_or_readout():
    original = measured([local("t", 3)] * 3)
    with_barriers = []
    for operation in original:
        with_barriers.extend([["barrier", [0, 1, 2, 3], [], []], operation])
    assert build_candidate(with_barriers)["changed"] is False  # They split the T word.
    corrupted = deepcopy(with_barriers)
    corrupted[1], corrupted[3] = corrupted[3], corrupted[1]
    with pytest.raises(ValueError, match="canonical initial resets"):
        build_candidate(corrupted)
    corrupted = deepcopy(with_barriers)
    corrupted[-1][2] = deepcopy(corrupted[-3][2])
    with pytest.raises(ValueError, match="bijections"):
        build_candidate(corrupted)


def test_output_and_segment_lists_do_not_alias_the_input():
    original = measured([local("t", 3)] * 3)
    result = build_candidate(original)
    result["operations"][0][1][0] = 2
    result["segments"][0]["forward_swap_edges"][0][0] = 0
    assert original[0] == ["reset", [0], [], []]
    assert original[4:] == measured([local("t", 3)] * 3)[4:]


def test_source_digest_binds_exact_primitive_order_and_list_fields():
    original = measured([local("t", 1), local("h", 1), local("t", 1), local("t", 1)])
    result = build_candidate(original)
    expected = hashlib.sha256(json.dumps(original, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert result["source_operations_sha256"] == expected
    assert result["candidate_operations_sha256"] == operation_digest(result["operations"])
    reversed_word = measured([local("t", 1), local("t", 1), local("h", 1), local("t", 1)])
    assert operation_digest(reversed_word) != expected


@pytest.mark.parametrize("num_qubits", [True, 3, 5, 4.0, "4"])
def test_width_is_an_exact_fixed_integer(num_qubits):
    with pytest.raises(ValueError, match="four data wires"):
        build_candidate(measured([local("h", 0)]), num_qubits=num_qubits)


@pytest.mark.parametrize("bad_operation", [
    ["rx", [1], [], [0.1]],
    ["id", [1], [], []],
    ["t", [True], [], []],
    ["t", [4], [], []],
    ["t", [1], [0], []],
    ["t", [1, 2], [], []],
    ["t", [1], [], [0]],
    ["cx", [0, 2], [], []],
    ["cx", [1, 1], [], []],
    ["barrier", [1, 1], [], []],
    ["barrier", [1], [0], []],
    ["t", [1], []],
    ("t", [1], [], []),
    ["t", (1,), [], []],
])
def test_unsupported_or_malformed_streams_fail_before_transform(bad_operation):
    with pytest.raises(ValueError):
        build_candidate(measured([bad_operation]))


@pytest.mark.parametrize("mutation", ["reset_order", "missing_reset", "interior_reset", "interior_measure",
                                      "duplicate_readout_wire", "duplicate_classical_destination", "missing_readout"])
def test_initialization_and_readout_contract_is_required(mutation):
    original = measured([local("t", 3)] * 3)
    if mutation == "reset_order":
        original[0], original[1] = original[1], original[0]
    elif mutation == "missing_reset":
        original.pop(0)
    elif mutation == "interior_reset":
        original.insert(5, local("reset", 2))
    elif mutation == "interior_measure":
        original.insert(5, ["measure", [2], [2], []])
    elif mutation == "duplicate_readout_wire":
        original[-1][1] = [2]
    elif mutation == "duplicate_classical_destination":
        original[-1][2] = original[-2][2]
    else:
        original.pop()
    with pytest.raises(ValueError):
        build_candidate(original)


def test_materialization_cap_cannot_be_silently_raised_by_the_shuttle():
    # Input at the old bound is valid; even the smallest profitable data
    # roundtrip would exceed the unchanged candidate bound and must fail.
    original = measured([local("t", 1)] * (10_000 - 8))
    assert len(original) == 10_000
    with pytest.raises(ValueError, match="Candidate materialization"):
        build_candidate(original)


@pytest.mark.parametrize("operations", [[], (), "saved", [local("h", 0)] * 10_001])
def test_source_list_and_bound_are_checked(operations):
    with pytest.raises(ValueError):
        build_candidate(operations)
