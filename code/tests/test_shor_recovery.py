"""Independent integer, promise, minimality and original-baseline checks."""
from __future__ import annotations

import math
import pytest

from openquantum.shor import factors_from_order, reconstruct_order
from openquantum.shor_recovery import (
    MAX_MODULUS_BITS, MAX_OFFSET_RADIUS, MAX_PHASE_BITS, MAX_SMOOTH_BOUND,
    RecoveryConfig, Work, _certificate, _complete, _prime_powers,
    baseline_frequency, last_small_convergent, recover_frequency,
)


def independent_order(N, a):
    value = 1
    for r in range(1, N):
        value = value * a % N
        if value == 1:
            return r
    raise AssertionError("reference orbit did not close")


def smooth(d, bound):
    # Independent definition: each complete prime power dividing d <= bound.
    p = 2
    while p * p <= d:
        power = 1
        while d % p == 0:
            d //= p
            power *= p
        if power > bound:
            return False
        p += 1
    return d <= bound


@pytest.mark.parametrize("frequency,bits,expected", [(0, 4, 1), (4, 4, 1), (6, 5, 5), (32, 6, 2), (11, 6, 6)])
def test_last_cf_strict_squared_bound(frequency, bits, expected):
    candidate = last_small_convergent(frequency, bits)
    assert candidate == expected
    assert candidate * candidate < 1 << bits


def test_integer_cf_large_odd_precision_has_no_floating_threshold():
    bits = 255
    frequency = (1 << bits) // 17
    assert last_small_convergent(frequency, bits) == 17


@pytest.mark.parametrize("N,a", [(3, 2), (7, 2), (15, 2), (21, 2), (33, 2), (35, 2), (47, 2), (59, 2), (63, 2)])
def test_counted_baseline_matches_existing_rule_for_every_frequency(N, a):
    for bits in (2, 4, 6, 8):
        for frequency in range(1 << bits):
            result = baseline_frequency(N, a, frequency, bits)
            expected = reconstruct_order(frequency, bits, N, a)
            assert result["period"] == expected
            assert result["factors"] == factors_from_order(N, a, expected)
            assert result["order_reference_used"] is False
            assert result["input_modulus_factorization_used"] is False


def test_literal_algorithm3_prime_power_smoothness_and_false_candidate():
    work = Work()
    powers = _prime_powers(5, work)
    assert powers == ((2, 4, 2), (3, 3, 1), (5, 5, 1))
    assert _complete(1, 3, 17, powers, work, 8192) is None  # order16 is not5-smooth
    assert _complete(4, 3, 17, powers, work, 8192) == 16  # missing component4 is5-smooth
    assert _complete(22, 2, 23, powers, work, 8192) == 22  # period multiple, notorder11
    certificate = _certificate(22, 2, 23, RecoveryConfig(), work)
    assert certificate["factorization_complete"] is True
    assert certificate["exact_order_certified"] is False
    assert certificate["status"] == "period_is_not_minimal"
    invalid = _certificate(9, 2, 23, RecoveryConfig(), work)
    assert invalid["status"] == "not_a_verified_period"
    assert invalid["exact_order_certified"] is False


def test_false_frequency_never_labels_order_multiple_exact():
    result = recover_frequency(23, 2, 46, 10, RecoveryConfig(offset_radius=0))
    assert result["period"] == 22
    assert result["certified_order"] is None
    assert result["status"] == "verified_period_only"
    assert result["certificate"]["status"] == "period_is_not_minimal"


def test_complete_offset_search_changes_recovery_without_repeating_quantum():
    direct = recover_frequency(59, 2, 70, 12, RecoveryConfig(offset_radius=0))
    nearby = recover_frequency(59, 2, 70, 12, RecoveryConfig(offset_radius=1))
    assert direct["certified_order"] is None
    assert nearby["certified_order"] == 58
    assert [item["offset"] for item in nearby["examined_offsets"]] == [0, 1, -1]
    assert nearby["work"]["modular_pow_calls"] > direct["work"]["modular_pow_calls"]
    assert nearby["outcome"] == direct["outcome"] == 70


def test_optimal_frequency_conditional_claim_against_independent_orders():
    checks = 0
    for N in range(3, 36, 2):
        for a in range(2, N):
            if math.gcd(a, N) != 1:
                continue
            order = independent_order(N, a)
            m = (N - 1).bit_length()
            bits = 2 * m
            for z in range(order):
                frequency = ((2 * (1 << bits) * z + order) // (2 * order)) % (1 << bits)
                result = recover_frequency(N, a, frequency, bits, RecoveryConfig(offset_radius=0))
                if result["certified_order"] is not None:
                    assert result["certified_order"] == order
                if smooth(math.gcd(order, z), m):
                    assert result["period"] == order
                    assert result["certified_order"] == order
                    checks += 1
    assert checks > 1000


def test_certificate_cap_keeps_verified_period_without_exact_order_claim():
    result = recover_frequency(15, 2, 0, 8, RecoveryConfig(certificate_period_cap=0))
    assert result["period"] == 4
    assert result["certified_order"] is None
    assert result["status"] == "verified_period_only"
    assert result["certificate"]["factorization_complete"] is False
    assert result["factors"] == [3, 5]


def test_zero_frequency_discloses_whole_order_smooth_classical_shortcut():
    result = recover_frequency(21, 2, 0, 6)
    assert result["smooth_only_pretest"] is True
    assert result["examined_offsets"] == []
    assert result["candidate_records"][0]["source"] == "whole_order_smooth_pretest"
    assert result["certified_order"] == 6
    assert result["work"]["certificate_trial_divisions"] > 0


def test_bad_half_order_base_negative_case_is_preserved_for_all_outcomes():
    for frequency in range(16):
        result = recover_frequency(33, 2, frequency, 4)
        assert result["certified_order"] == 10
        assert result["factors"] == []
        assert result["paper_l_on_1_through_m"] is False
        assert result["full_2m_precision_sufficient_for_all_orders_below_2_to_m"] is False


def test_modular_pow_cap_discards_partial_success():
    result = recover_frequency(15, 2, 0, 8, RecoveryConfig(modular_pow_call_cap=1))
    assert result["status"] == "resource_limit"
    assert result["period"] is None and result["certified_order"] is None
    assert result["factors"] == []
    assert result["work"]["modular_pow_calls"] == 1


@pytest.mark.parametrize("field,value", [("offset_radius", True), ("offset_radius", MAX_OFFSET_RADIUS + 1),
    ("smooth_multiplier", 0), ("smooth_multiplier", 1.0), ("certificate_period_cap", 1_000_001),
    ("modular_pow_call_cap", 0), ("modular_pow_call_cap", 8193)])
def test_config_bounds_and_types(field, value):
    with pytest.raises(ValueError):
        RecoveryConfig(**{field: value})


@pytest.mark.parametrize("args", [(15, 2, -1, 4), (15, 2, 16, 4), (15, 2, 0, True),
    (15, 3, 0, 4), (1 << MAX_MODULUS_BITS, 2, 0, 4), (15, 2, 0, MAX_PHASE_BITS + 1)])
def test_instance_guards(args):
    with pytest.raises(ValueError):
        recover_frequency(*args)


def test_smooth_bound_cap_is_not_silently_clamped():
    with pytest.raises(ValueError, match="smooth bound"):
        recover_frequency((1 << 127) - 1, 2, 0, 10, RecoveryConfig(smooth_multiplier=MAX_SMOOTH_BOUND))
