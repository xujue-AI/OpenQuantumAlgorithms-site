# SPDX-License-Identifier: MIT
"""Outward intervals for the fixed one-spin Gibbs-Z source contract.

This scientific helper was used in the accepted fixed calculation. The public
packaging is a source-only derivation that has not been executed as a downloaded
package. Import defines classes and constants only; arithmetic starts when an
Evaluator is constructed and invoked. No scientific package, native job, file,
environment, signal, random draw or subprocess is used here.

Every basic Decimal operation uses an explicit precision-64 directed context.
Transcendentals use bounded real series/brackets with analytic remainder
bounds, never Decimal.exp/sqrt or ordinary floating transcendental functions.
The fixed output check is conditional on an authenticated complete angle
ledger and the stated ideal circuit/iid measurement assumptions. It supplies
no full-state, generic mixing, floating-backend, hardware or formal proof claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Clamped, Context, Decimal, DivisionByZero, FloatOperation,
    InvalidOperation, Overflow, ROUND_CEILING, ROUND_FLOOR, Subnormal,
    Underflow,
)
import re
from typing import Callable, Iterable

PRECISION = 64
MAX_SCALAR_REQUESTS = 4096
MACHIN_TERMS = 64
TRIG_ATAN_TERMS = 128
EXP_TERMS = 512
SQRT_STEPS = 256
MAX_TRIG_ARGUMENT = 16
MAX_EXP_ARGUMENT = 128
MAX_POWER = 512
MAX_SUM_ITEMS = 4096
MAX_ENDPOINT_DIGITS = 4096
MAX_ENDPOINT_ADJUSTED = 4096
MAX_INTEGER_BITS = 16384
NUMERIC_TOKEN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


class IntervalError(ValueError):
    """A bound, type, arithmetic domain or fixed-budget condition was refused."""


def _endpoint(value) -> Decimal:
    """Parse an exact point without using the ambient Decimal context."""
    if type(value) is Decimal:
        result = value
    elif type(value) is int:
        if value.bit_length() > MAX_INTEGER_BITS:
            raise IntervalError("integer endpoint exceeds the fixed input bound")
        result = Decimal(value)
    elif type(value) is str:
        if (not 0 < len(value) <= MAX_ENDPOINT_DIGITS
                or NUMERIC_TOKEN.fullmatch(value) is None):
            raise IntervalError("interval endpoint requires a bounded numeric token")
        parts = value.lower().split("e")
        if len(parts) == 2:
            exponent_digits = parts[1].lstrip("+-").lstrip("0")
            if (len(exponent_digits) > 4
                    or int(exponent_digits or "0") > MAX_ENDPOINT_ADJUSTED + MAX_ENDPOINT_DIGITS):
                raise IntervalError("numeric token exponent exceeds its fixed parse bound")
        result = Decimal(value)
    else:
        raise IntervalError("use exact int/string/Decimal points or native_float explicitly")
    if (not result.is_finite()
            or len(result.as_tuple().digits) > MAX_ENDPOINT_DIGITS
            or result and abs(result.adjusted()) > MAX_ENDPOINT_ADJUSTED):
        raise IntervalError("interval endpoint is nonfinite or outside its bound")
    return result


@dataclass(frozen=True, slots=True)
class Interval:
    lo: Decimal
    hi: Decimal

    def __post_init__(self):
        if type(self.lo) is not Decimal or type(self.hi) is not Decimal:
            raise IntervalError("Interval endpoints must be exact Decimal objects")
        _endpoint(self.lo)
        _endpoint(self.hi)
        if self.lo > self.hi:
            raise IntervalError("reversed interval endpoints")


class Evaluator:
    """A fixed-precision arithmetic context and monotone admission ledger.

    A scalar request is one public pi/sqrt/atan/atan2_positive/sin/cos/exp
    call. Private endpoint/series work is separately recorded. Basic interval
    operations never dispatch through the ambient Decimal context.
    """

    def __init__(self, checkpoint: Callable[[], None] | None = None):
        if checkpoint is not None and not callable(checkpoint):
            raise IntervalError("checkpoint must be callable or absent")
        traps = [InvalidOperation, DivisionByZero, Overflow, Underflow,
                 Subnormal, Clamped, FloatOperation]
        self._down = Context(prec=PRECISION, rounding=ROUND_FLOOR,
                             Emin=-999999, Emax=999999, capitals=1, clamp=0,
                             flags=[], traps=traps)
        self._up = Context(prec=PRECISION, rounding=ROUND_CEILING,
                           Emin=-999999, Emax=999999, capitals=1, clamp=0,
                           flags=[], traps=traps)
        self._expected_traps = dict(self._down.traps)
        self._checkpoint = checkpoint
        self._scalar_requests = 0
        self._scalar_kinds: dict[str, int] = {}
        self._series_kinds: dict[str, int] = {}
        self._basic_operations = 0
        self._sqrt_bracket_iterations = 0
        self._certificate_calls = 0
        self._pi_value: Interval | None = None

    def _contexts_intact(self):
        for context, rounding in ((self._down, ROUND_FLOOR), (self._up, ROUND_CEILING)):
            if (context.prec != PRECISION or context.rounding != rounding
                    or context.Emin != -999999 or context.Emax != 999999
                    or context.capitals != 1 or context.clamp != 0
                    or dict(context.traps) != self._expected_traps):
                raise IntervalError("the frozen directed arithmetic context was changed")

    def _basic(self):
        self._contexts_intact()
        self._basic_operations += 1
        if self._checkpoint is not None and self._basic_operations % 64 == 0:
            self._checkpoint()

    def _request(self, name: str):
        self._contexts_intact()
        self._scalar_requests += 1
        self._scalar_kinds[name] = self._scalar_kinds.get(name, 0) + 1
        if self._scalar_requests > MAX_SCALAR_REQUESTS:
            raise IntervalError("the fixed scalar enclosure request budget was exceeded")
        if self._checkpoint is not None:
            self._checkpoint()

    def _series(self, name: str):
        self._series_kinds[name] = self._series_kinds.get(name, 0) + 1

    def _as_interval(self, value) -> Interval:
        return value if type(value) is Interval else self.point(value)

    def point(self, value) -> Interval:
        point = _endpoint(value)
        return Interval(point, point)

    def native_float(self, value: float) -> Interval:
        """The exact finite binary value, not a context-rounded conversion."""
        if type(value) is not float:
            raise IntervalError("native_float requires a literal float")
        return self.point(Decimal.from_float(value))

    def canonical_decimal(self, token: str) -> Interval:
        """An exact decimal numeric token; symbolic QASM expressions refuse."""
        if type(token) is not str:
            raise IntervalError("canonical decimal requires a string token")
        return self.point(token)

    def interval(self, lo, hi) -> Interval:
        return Interval(_endpoint(lo), _endpoint(hi))

    def add(self, left, right) -> Interval:
        self._basic()
        a, b = self._as_interval(left), self._as_interval(right)
        return Interval(self._down.add(a.lo, b.lo), self._up.add(a.hi, b.hi))

    def sub(self, left, right) -> Interval:
        self._basic()
        a, b = self._as_interval(left), self._as_interval(right)
        return Interval(self._down.subtract(a.lo, b.hi), self._up.subtract(a.hi, b.lo))

    def neg(self, value) -> Interval:
        self._basic()
        a = self._as_interval(value)
        return Interval(a.hi.copy_negate(), a.lo.copy_negate())

    def mul(self, left, right) -> Interval:
        self._basic()
        a, b = self._as_interval(left), self._as_interval(right)
        lower = [self._down.multiply(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        upper = [self._up.multiply(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return Interval(min(lower), max(upper))

    def div(self, left, right) -> Interval:
        self._basic()
        a, b = self._as_interval(left), self._as_interval(right)
        if b.lo <= Decimal(0) <= b.hi:
            raise IntervalError("division denominator interval contains zero")
        lower = [self._down.divide(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        upper = [self._up.divide(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return Interval(min(lower), max(upper))

    def square(self, value) -> Interval:
        self._basic()
        a = self._as_interval(value)
        zero = Decimal(0)
        if a.lo >= zero:
            return Interval(self._down.multiply(a.lo, a.lo), self._up.multiply(a.hi, a.hi))
        if a.hi <= zero:
            return Interval(self._down.multiply(a.hi, a.hi), self._up.multiply(a.lo, a.lo))
        return Interval(zero, max(self._up.multiply(a.lo, a.lo), self._up.multiply(a.hi, a.hi)))

    def pow_int(self, value, exponent: int) -> Interval:
        """Bounded repeated squaring; no Decimal power/transcendental call."""
        if type(exponent) is not int or not 0 <= exponent <= MAX_POWER:
            raise IntervalError("integer power is outside its fixed bound")
        a = self._as_interval(value)
        result, remaining = self.point(1), exponent
        while remaining:
            if remaining & 1:
                result = self.mul(result, a)
            remaining >>= 1
            if remaining:
                a = self.square(a)
        return result

    def sum(self, values: Iterable) -> Interval:
        result = self.point(0)
        for index, value in enumerate(values):
            if index >= MAX_SUM_ITEMS:
                raise IntervalError("interval sum exceeds its fixed item count")
            result = self.add(result, value)
        return result

    def abs_interval(self, value) -> Interval:
        self._basic()
        a = self._as_interval(value)
        zero = Decimal(0)
        if a.lo >= zero:
            return a
        if a.hi <= zero:
            return Interval(a.hi.copy_abs(), a.lo.copy_abs())
        return Interval(zero, max(a.lo.copy_abs(), a.hi.copy_abs()))

    def upper_abs(self, value) -> Decimal:
        a = self._as_interval(value)
        return max(a.lo.copy_abs(), a.hi.copy_abs())

    def nonnegative_part(self, value) -> Interval:
        """Enclose max(x,0) as an exact monotone function, not float clipping."""
        self._basic()
        a = self._as_interval(value)
        return Interval(max(a.lo, Decimal(0)), max(a.hi, Decimal(0)))

    def intersect(self, left, right) -> Interval:
        a, b = self._as_interval(left), self._as_interval(right)
        lo, hi = max(a.lo, b.lo), min(a.hi, b.hi)
        if lo > hi:
            raise IntervalError("empty intersection with an independently proved range")
        return Interval(lo, hi)

    def hull(self, values: Iterable) -> Interval:
        result = None
        for index, value in enumerate(values):
            if index >= MAX_SUM_ITEMS:
                raise IntervalError("interval hull exceeds its fixed item count")
            a = self._as_interval(value)
            result = a if result is None else Interval(min(result.lo, a.lo), max(result.hi, a.hi))
        if result is None:
            raise IntervalError("an empty interval hull is undefined")
        return result

    def _sqrt_endpoint(self, point: Decimal) -> Interval:
        if point < Decimal(0):
            raise IntervalError("negative square-root endpoint")
        if point == Decimal(0):
            return self.point(0)
        low, high = Decimal(0), max(Decimal(1), point)
        for _ in range(SQRT_STEPS):
            self._basic()
            self._sqrt_bracket_iterations += 1
            middle = self._down.divide(self._down.add(low, high), Decimal(2))
            if not low < middle < high:
                continue
            square_lo = self._down.multiply(middle, middle)
            square_hi = self._up.multiply(middle, middle)
            if square_hi <= point:
                low = middle
            elif square_lo >= point:
                high = middle
            else:
                # sqrt(x) is between m and x/m for every positive m.
                quotient_lo = self._down.divide(point, middle)
                quotient_hi = self._up.divide(point, middle)
                low = max(low, min(middle, quotient_lo))
                high = min(high, max(middle, quotient_hi))
                if low > high:
                    raise IntervalError("square-root bracket invariant was lost")
        return Interval(low, high)

    def sqrt(self, value) -> Interval:
        self._request("sqrt")
        a = self._as_interval(value)
        if a.lo < Decimal(0):
            raise IntervalError("square-root interval has a negative lower endpoint")
        lower, upper = self._sqrt_endpoint(a.lo), self._sqrt_endpoint(a.hi)
        return Interval(lower.lo, upper.hi)

    def _atan_series(self, value: Interval, terms: int) -> Interval:
        if terms not in (MACHIN_TERMS, TRIG_ATAN_TERMS):
            raise IntervalError("arctangent term count is not frozen")
        if self.upper_abs(value) > Decimal("0.5"):
            raise IntervalError("arctangent series argument was not reduced")
        self._series("atan_" + str(terms))
        square = self.square(value)
        power, total = value, self.point(0)
        for index in range(terms):
            term = self.div(power, 2 * index + 1)
            total = self.add(total, term if index % 2 == 0 else self.neg(term))
            power = self.mul(power, square)
        # The alternating tail is bounded by the next term for |x| <= 1/2.
        radius = self.div(self.pow_int(self.upper_abs(value), 2 * terms + 1), 2 * terms + 1).hi
        return self.add(total, self.interval(radius.copy_negate(), radius))

    def _pi_interval(self) -> Interval:
        if self._pi_value is None:
            first = self._atan_series(self.div(1, 5), MACHIN_TERMS)
            second = self._atan_series(self.div(1, 239), MACHIN_TERMS)
            value = self.sub(self.mul(16, first), self.mul(4, second))
            # The exact Machin identity and elementary 3<pi<4 justify this.
            self._pi_value = self.intersect(value, self.interval(3, 4))
        return self._pi_value

    def pi(self) -> Interval:
        self._request("pi")
        return self._pi_interval()

    def _atan_endpoint(self, point: Decimal) -> Interval:
        if point < Decimal(0):
            return self.neg(self._atan_endpoint(point.copy_negate()))
        if point <= Decimal("0.5"):
            return self._atan_series(self.point(point), TRIG_ATAN_TERMS)
        if point <= Decimal(2):
            reduced = self.div(self.sub(point, 1), self.add(point, 1))
            return self.add(self.div(self._pi_interval(), 4), self._atan_series(reduced, TRIG_ATAN_TERMS))
        reduced = self.div(1, point)
        return self.sub(self.div(self._pi_interval(), 2), self._atan_series(reduced, TRIG_ATAN_TERMS))

    def _atan_interval(self, value: Interval) -> Interval:
        lower, upper = self._atan_endpoint(value.lo), self._atan_endpoint(value.hi)
        return Interval(lower.lo, upper.hi)

    def atan(self, value) -> Interval:
        self._request("atan")
        return self._atan_interval(self._as_interval(value))

    def atan2_positive(self, y, x) -> Interval:
        """atan2(y,x) for y>=0 and x>0; callers multiply by 2 for an RY angle."""
        self._request("atan2_positive")
        yy, xx = self._as_interval(y), self._as_interval(x)
        if yy.lo < Decimal(0) or xx.lo <= Decimal(0):
            raise IntervalError("positive atan2 requires y>=0 and strictly positive x")
        return self._atan_interval(self.div(yy, xx))

    @staticmethod
    def _factorial(number: int) -> int:
        if type(number) is not int or not 0 <= number <= MAX_POWER:
            raise IntervalError("factorial remainder index is outside its bound")
        result = 1
        for value in range(2, number + 1):
            result *= value
        return result

    def _trig(self, value: Interval, sine: bool) -> Interval:
        if self.upper_abs(value) > Decimal(MAX_TRIG_ARGUMENT):
            raise IntervalError("trigonometric interval exceeds the fixed argument range")
        self._series("sin" if sine else "cos")
        square = self.square(value)
        term = value if sine else self.point(1)
        total = term
        for index in range(1, TRIG_ATAN_TERMS):
            denominator = (2 * index) * (2 * index + 1) if sine else (2 * index - 1) * (2 * index)
            term = self.neg(self.div(self.mul(term, square), denominator))
            total = self.add(total, term)
        # Both polynomials are T_255; cos has zero odd coefficients.
        # All real sin/cos derivatives have magnitude <=1.
        radius = self.div(self.pow_int(self.upper_abs(value), 256), self._factorial(256)).hi
        enclosure = self.add(total, self.interval(radius.copy_negate(), radius))
        return self.intersect(enclosure, self.interval(-1, 1))

    def sin(self, value) -> Interval:
        self._request("sin")
        return self._trig(self._as_interval(value), True)

    def cos(self, value) -> Interval:
        self._request("cos")
        return self._trig(self._as_interval(value), False)

    def _exp_positive(self, point: Decimal) -> Interval:
        if not Decimal(0) <= point <= Decimal(MAX_EXP_ARGUMENT):
            raise IntervalError("positive exponential endpoint exceeds its fixed range")
        self._series("exp_positive")
        term_lo = term_hi = Decimal(1)
        total_lo = total_hi = Decimal(1)
        for index in range(1, EXP_TERMS):
            self._basic()
            term_lo = self._down.divide(self._down.multiply(term_lo, point), Decimal(index))
            term_hi = self._up.divide(self._up.multiply(term_hi, point), Decimal(index))
            total_lo = self._down.add(total_lo, term_lo)
            total_hi = self._up.add(total_hi, term_hi)
        # First omitted term is x^512/512!. Subsequent ratios <=x/513.
        first_hi = self._up.divide(self._up.multiply(term_hi, point), Decimal(EXP_TERMS))
        ratio_hi = self._up.divide(point, Decimal(EXP_TERMS + 1))
        denominator_lo = self._down.subtract(Decimal(1), ratio_hi)
        if denominator_lo <= Decimal(0):
            raise IntervalError("exponential tail denominator is nonpositive")
        tail_hi = self._up.divide(first_hi, denominator_lo)
        return Interval(total_lo, self._up.add(total_hi, tail_hi))

    def _exp_endpoint(self, point: Decimal) -> Interval:
        if point.copy_abs() > Decimal(MAX_EXP_ARGUMENT):
            raise IntervalError("exponential endpoint exceeds its fixed range")
        if point < Decimal(0):
            positive = self._exp_positive(point.copy_negate())
            return Interval(self._down.divide(Decimal(1), positive.hi),
                            self._up.divide(Decimal(1), positive.lo))
        return self._exp_positive(point)

    def exp(self, value) -> Interval:
        self._request("exp")
        a = self._as_interval(value)
        lower, upper = self._exp_endpoint(a.lo), self._exp_endpoint(a.hi)
        return Interval(lower.lo, upper.hi)

    def record(self, value) -> dict[str, str]:
        a = self._as_interval(value)
        return {"lo": str(a.lo), "hi": str(a.hi)}

    def counters(self) -> dict:
        self._contexts_intact()
        return {
            "precision": PRECISION,
            "scalar_enclosure_requests": self._scalar_requests,
            "scalar_enclosure_requests_max": MAX_SCALAR_REQUESTS,
            "scalar_requests_by_kind": dict(self._scalar_kinds),
            "private_series_evaluations_by_kind": dict(self._series_kinds),
            "basic_interval_operations": self._basic_operations,
            "basic_operation_counter_scope": "logical API/block checkpoints; not an enumerated native arithmetic count",
            "sqrt_bracket_iterations": self._sqrt_bracket_iterations,
            "fixed_certificate_evaluations": self._certificate_calls,
            "machin_terms": MACHIN_TERMS,
            "trig_atan_terms": TRIG_ATAN_TERMS,
            "exp_terms": EXP_TERMS,
            "sqrt_steps_max_per_endpoint": SQRT_STEPS,
            "precision_or_term_escalation_performed": False,
        }


def evaluate_fixed_output(ev: Evaluator, angle_error_sum: Interval) -> dict:
    """One fixed finite-output inequality check, conditional on the angle ledger.

    d/u are exact-ideal-reference rate enclosures. No floating quantum result
    is accepted as a rate/certificate input. The caller must authenticate and
    combine the complete primitive ledger and separate raw implementation
    checks; a numerical inequality pass here alone does not establish them.
    """
    if type(ev) is not Evaluator or type(angle_error_sum) is not Interval:
        raise IntervalError("fixed certificate requires its evaluator and interval angle sum")
    ev._certificate_calls += 1
    if ev._certificate_calls != 1:
        raise IntervalError("the fixed interval certificate may be evaluated only once")
    if angle_error_sum.lo < Decimal(0):
        raise IntervalError("angle-error sum requires a nonnegative interval")

    pi = ev.pi()
    spacing = ev.sqrt(ev.div(ev.mul(2, pi), 16))
    dt, dw = ev.mul("0.5", spacing), ev.mul(2, spacing)
    signed = tuple(range(8)) + tuple(range(-8, 0))
    times = [ev.mul(dt, index) for index in signed]
    sqrt_weights = [ev.exp(ev.neg(ev.div(ev.mul(pi, index * index), 32)))
                    for index in signed]
    gamma = [ev.exp(ev.neg(ev.nonnegative_part(ev.add(ev.mul(dw, index), "0.5"))))
             for index in signed]

    contributions = {"d": [], "u": []}
    for kind in ("d", "u"):
        for frequency in range(16):
            real_parts, imag_parts = [], []
            for clock in range(16):
                # Exact integer periodicity precedes enclosing pi.
                residue = (frequency * clock) % 16
                fourier = ev.div(ev.mul(ev.mul(2, pi), residue), 16)
                gap = ev.mul(2, times[clock])
                phase = ev.neg(ev.add(gap, fourier)) if kind == "d" else ev.sub(gap, fourier)
                real_parts.append(ev.mul(sqrt_weights[clock], ev.cos(phase)))
                imag_parts.append(ev.mul(sqrt_weights[clock], ev.sin(phase)))
            real_sum, imag_sum = ev.sum(real_parts), ev.sum(imag_parts)
            norm_squared = ev.add(ev.square(real_sum), ev.square(imag_sum))
            # 1/(J*M*sqrt(M)) =1/(2*16*4). PREPARE normalization appears once.
            contributions[kind].append(ev.div(ev.mul(gamma[frequency], norm_squared), 128))
    d, u = ev.sum(contributions["d"]), ev.sum(contributions["u"])
    rate_sum = ev.add(d, u)
    population_step = ev.mul("0.125", rate_sum)
    domains = {
        "d_positive_lower": d.lo > Decimal(0),
        "u_positive_lower": u.lo > Decimal(0),
        "rate_sum_positive_lower": rate_sum.lo > Decimal(0),
        "population_step_nonnegative": population_step.lo >= Decimal(0),
        "population_step_at_most_one": population_step.hi <= Decimal(1),
    }
    interval_records = {
        "pi": ev.record(pi), "dt": ev.record(dt), "dw": ev.record(dw),
        "d": ev.record(d), "u": ev.record(u), "rate_sum": ev.record(rate_sum),
        "population_step": ev.record(population_step),
        "angle_error_sum": ev.record(angle_error_sum),
    }
    report = {
        "schema": "openquantum.finite-gibbs.fixed-output-interval.v1",
        "scope": "fixed Gibbs Z-mean inequalities conditional on complete authenticated primitive-angle ledger and ideal iid measurement",
        "fixed_case": {"H": "Z", "beta": 1, "M": 16, "r": 1,
                       "delta": "1/8", "R": 64, "input": "known_plus", "N": 16384},
        "intervals": interval_records,
        "rate_frequency_contributions": {
            kind: [ev.record(value) for value in values]
            for kind, values in contributions.items()
        },
        "mesh_intervals": {"sqrt_weights": [ev.record(value) for value in sqrt_weights],
                           "gamma": [ev.record(value) for value in gamma]},
        "domain_checks": domains,
        "budget_checks": {},
        "certificate_passed": False,
        "status": "interval_domain_refused",
        "angle_sum_requires_authenticated_source_ledger": True,
        "rate_proposal_normalization": "included exactly once as 1/(2*16*4); no extra division of full-Kraus rates",
        "source_realization_bound": "2*64*sum absolute per-primitive numeric parameter deviations",
        "all_components_verified": False,
        "whole_invocation_validated": False,
        "new_algorithm_result": False,
        "full_state_Gibbs_accuracy_certified": False,
        "general_thermal_sampler_implemented": False,
        "general_mixing_theorem_reproduced": False,
        "coherent_correction_implemented": False,
        "coherent_correction_evolution_implemented": False,
        "general_coherent_correction_implemented": False,
        "general_channel_implemented": False,
        "channel_implemented": False,
        "sampler_implemented": False,
        "exact_finite_kms": False,
        "finite_mesh_exact_kms": False,
        "mixing_certificate": False,
        "rotations_synthesized": False,
        "physical_resources_computed": False,
        "floating_point_error_certified": False,
        "floating_backend_error_certified": False,
        "hardware_error_certified": False,
        "formal_proof_reproduced": False,
        "random_sampling_performed": False,
        "shots_drawn": False,
        "parameter_or_precision_retuning": False,
    }
    if not all(domains.values()):
        report["counters"] = ev.counters()
        return report

    z_star = ev.div(ev.sub(u, d), rate_sum)
    decay_discrete = ev.pow_int(ev.sub(1, population_step), 64)
    decay_semigroup = ev.exp(ev.neg(ev.mul(64, population_step)))
    z_D = ev.add(z_star, ev.mul(decay_discrete, ev.neg(z_star)))
    z_S = ev.add(z_star, ev.mul(decay_semigroup, ev.neg(z_star)))
    exp_two = ev.exp(2)
    mu_beta = ev.neg(ev.div(ev.sub(exp_two, 1), ev.add(exp_two, 1)))
    angle_bias = ev.mul(128, angle_error_sum)
    digital_bias = ev.abs_interval(ev.sub(z_D, z_S))
    mixing_bias = ev.abs_interval(ev.sub(z_S, z_star))
    mesh_bias = ev.abs_interval(ev.sub(z_star, mu_beta))
    preparation_total = ev.sum([angle_bias, digital_bias, mixing_bias, mesh_bias])
    measurement = ev.point("0.030")
    total_error_allowance = ev.add(preparation_total, measurement)
    exponent = ev.neg(ev.div(ev.mul(16384, ev.square(measurement)), 2))
    hoeffding_failure_bound = ev.mul(2, ev.exp(exponent))
    bounds = {
        "z_star": z_star, "z_D": z_D, "z_S": z_S, "mu_beta": mu_beta,
        "angle_bias": angle_bias, "digital_bias": digital_bias,
        "mixing_bias": mixing_bias, "mesh_bias": mesh_bias,
        "preparation_total": preparation_total,
        "measurement_error_allowance": measurement,
        "total_error_allowance": total_error_allowance,
        "hoeffding_failure_bound": hoeffding_failure_bound,
    }
    report["intervals"].update({name: ev.record(value) for name, value in bounds.items()})
    checks = {
        "angle_realization_at_most_0.002": angle_bias.hi <= Decimal("0.002"),
        "mesh_stationary_at_most_0.004": mesh_bias.hi <= Decimal("0.004"),
        "digital_transient_at_most_0.002": digital_bias.hi <= Decimal("0.002"),
        "finite_time_transient_at_most_0.012": mixing_bias.hi <= Decimal("0.012"),
        "preparation_total_at_most_0.020": preparation_total.hi <= Decimal("0.020"),
        "total_error_at_most_0.050": total_error_allowance.hi <= Decimal("0.050"),
        "measurement_failure_at_most_0.050": hoeffding_failure_bound.hi <= Decimal("0.050"),
    }
    report["budget_checks"] = checks
    report["certificate_passed"] = all(checks.values())
    report["status"] = "fixed_output_inequalities_passed" if report["certificate_passed"] else "fixed_output_budget_refused"
    report["counters"] = ev.counters()
    return report
