"""Bounded integer-only frequency recovery with explicit work and proof scope.

The candidate pipeline follows Ekerå, ACM TQC 5(2):11 (2024),
doi:10.1145/3655026: the continued-fraction variant, complete offset search,
Algorithm 4's smooth-completion filter, and Algorithm 3's product tree.
It is an original implementation cross-checked against the MIT Quaspy source
at 8040affdf08814ef7d451d9797d28f7645856151. No Quaspy code is imported.

For arbitrary frequencies the result can be an order multiple. A separately
charged, bounded factorization/minimality check certifies an exact order.
Recovery never enumerates the order or factors the input modulus.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

MAX_MODULUS_BITS = 128
MAX_PHASE_BITS = 256
MAX_OFFSET_RADIUS = 32
MAX_SMOOTH_BOUND = 512
MAX_MODULAR_POW_CALLS = 8192
MAX_EXPONENT_BITS = 4096
MAX_CERTIFICATE_PERIOD = 1_000_000
MAX_CERTIFICATE_DIVISIONS = 4096


def _int(name: str, value: int, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class RecoveryConfig:
    offset_radius: int = 2
    smooth_multiplier: int = 1
    certificate_period_cap: int = MAX_CERTIFICATE_PERIOD
    modular_pow_call_cap: int = MAX_MODULAR_POW_CALLS

    def __post_init__(self):
        _int("offset_radius", self.offset_radius, 0, MAX_OFFSET_RADIUS)
        _int("smooth_multiplier", self.smooth_multiplier, 1, MAX_SMOOTH_BOUND)
        _int("certificate_period_cap", self.certificate_period_cap, 0, MAX_CERTIFICATE_PERIOD)
        _int("modular_pow_call_cap", self.modular_pow_call_cap, 1, MAX_MODULAR_POW_CALLS)


class RecoveryLimit(RuntimeError):
    """An explicit work cap stopped a recovery before a complete result."""


@dataclass
class Work:
    euclidean_divisions: int = 0
    sieve_marks: int = 0
    integer_products: int = 0
    modular_pow_calls: int = 0
    modular_pow_exponent_bits: int = 0
    maximum_exponent_bits: int = 0
    certificate_trial_divisions: int = 0
    gcd_calls: int = 0

    def pow(self, base: int, exponent: int, modulus: int, cap: int) -> int:
        bits = exponent.bit_length()
        if self.modular_pow_calls >= cap or bits > MAX_EXPONENT_BITS:
            raise RecoveryLimit("modular exponentiation work cap exceeded")
        self.modular_pow_calls += 1
        self.modular_pow_exponent_bits += bits
        self.maximum_exponent_bits = max(self.maximum_exponent_bits, bits)
        return pow(base, exponent, modulus)


def _instance(modulus: int, base: int, outcome: int, phase_bits: int) -> int:
    _int("modulus", modulus, 3, (1 << MAX_MODULUS_BITS) - 1)
    _int("base", base, 2, modulus - 1)
    if math.gcd(modulus, base) != 1:
        raise ValueError("order recovery requires a coprime base")
    _int("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    _int("outcome", outcome, 0, (1 << phase_bits) - 1)
    return (modulus - 1).bit_length()


def _primes(bound: int, work: Work) -> tuple[int, ...]:
    sieve = bytearray(b"\x01") * (bound + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, math.isqrt(bound) + 1):
        if sieve[p]:
            for composite in range(p * p, bound + 1, p):
                sieve[composite] = 0
                work.sieve_marks += 1
    return tuple(p for p in range(2, bound + 1) if sieve[p])


def _prime_powers(bound: int, work: Work) -> tuple[tuple[int, int, int], ...]:
    result = []
    for p in _primes(bound, work):
        power, exponent = p, 1
        while power <= bound // p:
            power *= p
            work.integer_products += 1
            exponent += 1
        result.append((p, power, exponent))
    return tuple(result)


def _product(values, work: Work) -> int:
    product = 1
    for value in values:
        product *= value
        work.integer_products += 1
    return product


def last_small_convergent(outcome: int, phase_bits: int, *, work: Work | None = None) -> int:
    """Last CF denominator q with q*q < 2**phase_bits, using exact integers.

    The paper's optimal-frequency and precision promises make this r/gcd(r,z).
    This function neither checks those unavailable promises nor returns an
    asserted order. A zero frequency yields the candidate one.
    """
    _int("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    _int("outcome", outcome, 0, (1 << phase_bits) - 1)
    work = Work() if work is None else work
    numerator, denominator = outcome, 1 << phase_bits
    previous, current, last = 1, 0, 1
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        work.euclidean_divisions += 1
        work.integer_products += 2
        candidate = quotient * current + previous
        if candidate * candidate >= 1 << phase_bits:
            break
        last = candidate
        previous, current = current, candidate
        numerator, denominator = denominator, remainder
    return last


def _complete(candidate: int, base: int, modulus: int, powers, work: Work, cap: int) -> int | None:
    """Literal Algorithm 3; arbitrary candidates can yield order multiples."""
    _int("candidate", candidate, 1, (1 << MAX_MODULUS_BITS) - 1)
    x = work.pow(base, candidate, modulus, cap)

    def leaves(value: int, subset):
        if len(subset) == 1:
            return [(subset[0], value)]
        half = len(subset) // 2
        left, right = subset[:half], subset[half:]
        left_exponent = _product((entry[1] for entry in right), work)
        right_exponent = _product((entry[1] for entry in left), work)
        x_left = work.pow(value, left_exponent, modulus, cap)
        x_right = work.pow(value, right_exponent, modulus, cap)
        return leaves(x_left, left) + leaves(x_right, right)

    missing = 1
    for (prime, _, exponent_cap), isolated in leaves(x, powers):
        exponent = 0
        while isolated != 1:
            if exponent == exponent_cap:
                return None
            isolated = work.pow(isolated, prime, modulus, cap)
            missing *= prime
            work.integer_products += 1
            exponent += 1
    work.integer_products += 1
    period = candidate * missing
    if work.pow(base, period, modulus, cap) != 1:
        raise AssertionError("Algorithm 3 returned an unverified period")
    return period


def _certificate(period: int, base: int, modulus: int, config: RecoveryConfig, work: Work) -> dict:
    _int("period", period, 1, (1 << MAX_EXPONENT_BITS) - 1)
    if period > config.certificate_period_cap:
        return {"status": "unavailable_period_cap", "period_cap": config.certificate_period_cap,
                "factorization_complete": False, "exact_order_certified": False}
    if work.pow(base, period, modulus, config.modular_pow_call_cap) != 1:
        return {"status": "not_a_verified_period", "factorization_complete": False,
                "exact_order_certified": False}
    remainder, divisor = period, 2
    factors = []
    while divisor * divisor <= remainder:
        while True:
            if work.certificate_trial_divisions >= MAX_CERTIFICATE_DIVISIONS:
                raise RecoveryLimit("candidate-period certificate division cap exceeded")
            work.certificate_trial_divisions += 1
            quotient, residue = divmod(remainder, divisor)
            if residue:
                break
            factors.append(divisor)
            remainder = quotient
        divisor = 3 if divisor == 2 else divisor + 2
    if remainder > 1:
        factors.append(remainder)
    distinct = sorted(set(factors))
    witnesses = []
    for prime in distinct:
        residue = work.pow(base, period // prime, modulus, config.modular_pow_call_cap)
        witnesses.append({"prime": prime, "exponent": period // prime, "residue": residue})
    exact = all(witness["residue"] != 1 for witness in witnesses)
    return {"status": "exact_order_certified" if exact else "period_is_not_minimal",
            "factorization_complete": True, "candidate_period_prime_factors": factors,
            "minimality_witnesses": witnesses, "exact_order_certified": exact,
            "scope": "bounded factorization of the returned period, not of the input modulus"}


def _factors(period: int | None, modulus: int, base: int, config: RecoveryConfig, work: Work) -> list[int]:
    if period is None or period % 2:
        return []
    if work.pow(base, period, modulus, config.modular_pow_call_cap) != 1:
        raise AssertionError("factor extraction received an unverified period")
    halfway = work.pow(base, period // 2, modulus, config.modular_pow_call_cap)
    divisors = []
    for offset in (-1, 1):
        work.gcd_calls += 1
        divisor = math.gcd(halfway + offset, modulus)
        if 1 < divisor < modulus:
            if modulus % divisor:
                raise AssertionError("invalid factor output")
            divisors.append(divisor)
    return sorted(set(divisors))


def recover_frequency(modulus: int, base: int, outcome: int, phase_bits: int,
                      config: RecoveryConfig = RecoveryConfig()) -> dict:
    """Complete bounded offset/CF/filter/completion pipeline and certificate.

    An optimal frequency in the examined offsets and the cm-smooth missing
    component promise imply exact recovery under the paper's CF precision
    assumptions. Arbitrary input is reported as a verified period unless the
    separately charged minimality certificate establishes an exact order.
    No peak isolation, speculative stopping or lattice enumeration is used.
    """
    started = time.perf_counter_ns()
    if not isinstance(config, RecoveryConfig):
        raise ValueError("config must be RecoveryConfig")
    m = _instance(modulus, base, outcome, phase_bits)
    bound = config.smooth_multiplier * m
    if bound > MAX_SMOOTH_BOUND:
        raise ValueError("cm smooth bound exceeds the bounded implementation")
    work = Work(gcd_calls=1)
    result = {"method": "bounded_Ekera_CF_Algorithm4_Algorithm3", "modulus": modulus,
              "base": base, "outcome": outcome, "phase_bits": phase_bits, "m": m,
              "smooth_bound": bound, "config": asdict(config), "period": None,
              "certified_order": None, "factors": [], "examined_offsets": [],
              "candidate_records": [], "smooth_only_pretest": False,
              "full_2m_precision_sufficient_for_all_orders_below_2_to_m": phase_bits >= 2 * m,
              "paper_l_on_1_through_m": 1 <= phase_bits - m <= m,
              "order_reference_used": False, "input_modulus_factorization_used": False}
    try:
        powers = _prime_powers(bound, work)
        smooth_exponent = _product((entry[1] for entry in powers), work)
        g_smooth = work.pow(base, smooth_exponent, modulus, config.modular_pow_call_cap)
        result["prime_power_schedule"] = [{"prime": p, "power": q, "exponent_cap": e}
                                          for p, q, e in powers]
        result["smooth_exponent"] = smooth_exponent
        periods = []
        if g_smooth == 1:
            result["smooth_only_pretest"] = True
            period = _complete(1, base, modulus, powers, work, config.modular_pow_call_cap)
            if period is None:
                raise AssertionError("smooth pretest and completion disagree")
            periods.append(period)
            result["candidate_records"].append({"candidate": 1, "source": "whole_order_smooth_pretest",
                                               "completion_period": period})
        else:
            seen_frequencies, candidate_periods = set(), {}
            grid = 1 << phase_bits
            for radius in range(config.offset_radius + 1):
                for offset in ((0,) if radius == 0 else (radius, -radius)):
                    shifted = (outcome + offset) % grid
                    if shifted in seen_frequencies:
                        continue
                    seen_frequencies.add(shifted)
                    candidate = last_small_convergent(shifted, phase_bits, work=work)
                    result["examined_offsets"].append({"offset": offset, "frequency": shifted,
                                                       "candidate": candidate})
                    if candidate >= 1 << m or candidate in candidate_periods:
                        continue
                    # Algorithm 4's filter, without peak-isolation heuristics.
                    if work.pow(g_smooth, candidate, modulus, config.modular_pow_call_cap) != 1:
                        candidate_periods[candidate] = None
                        result["candidate_records"].append({"candidate": candidate, "filter_passed": False})
                        continue
                    period = _complete(candidate, base, modulus, powers, work, config.modular_pow_call_cap)
                    candidate_periods[candidate] = period
                    result["candidate_records"].append({"candidate": candidate, "filter_passed": True,
                                                       "completion_period": period})
                    if period is not None:
                        periods.append(period)
        if not periods:
            result["status"] = "no_period_found_within_configured_search"
        else:
            period = min(periods)
            certificate = _certificate(period, base, modulus, config, work)
            result.update({"period": period, "certificate": certificate,
                           "certified_order": period if certificate["exact_order_certified"] else None,
                           "factors": _factors(period, modulus, base, config, work),
                           "status": "exact_order_certified" if certificate["exact_order_certified"] else "verified_period_only"})
    except RecoveryLimit as error:
        result.update({"status": "resource_limit", "reason": str(error), "period": None,
                       "certified_order": None, "factors": []})
    result["work"] = asdict(work)
    result["wall_ns"] = time.perf_counter_ns() - started
    return result


def baseline_frequency(modulus: int, base: int, outcome: int, phase_bits: int,
                       config: RecoveryConfig = RecoveryConfig()) -> dict:
    """The book's original convergent-only rule with its actual work counted.

    It excludes denominator one, tries convergents below N in their original
    order, and returns the first modularly verified period. No extra order
    certificate or smooth/offset search is added to this baseline.
    """
    started = time.perf_counter_ns()
    if not isinstance(config, RecoveryConfig):
        raise ValueError("config must be RecoveryConfig")
    _instance(modulus, base, outcome, phase_bits)
    work = Work(gcd_calls=1)
    numerator, denominator = outcome, 1 << phase_bits
    previous, current = 1, 0
    candidates = []
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        work.euclidean_divisions += 1
        work.integer_products += 1
        previous, current = current, quotient * current + previous
        if 1 < current < modulus:
            candidates.append(current)
        numerator, denominator = denominator, remainder
    period = None
    for candidate in candidates:
        if work.pow(base, candidate, modulus, config.modular_pow_call_cap) == 1:
            period = candidate
            break
    return {"method": "existing_convergent_only_rule", "period": period,
            "factors": _factors(period, modulus, base, config, work),
            "candidates": candidates, "work": asdict(work),
            "wall_ns": time.perf_counter_ns() - started,
            "order_reference_used": False, "input_modulus_factorization_used": False}
