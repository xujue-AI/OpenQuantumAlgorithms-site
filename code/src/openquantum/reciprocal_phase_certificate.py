# SPDX-License-Identifier: MIT
"""Exact rational certificate for an ideal, real-extracted QSP polynomial.

The signal convention is B(x)=[[x,sqrt(1-x*x)],[sqrt(1-x*x),-x]] and
D(phi)=diag(exp(i*phi),exp(-i*phi)).  Stored binary float phases are treated
as exact real numbers.  Rational Taylor enclosures and an exact polynomial
recurrence certify the entire interval [-1,1], independently of grid checks.

This certifies the mathematical signal sequence only.  Loading/access errors,
gate decomposition, concrete angle synthesis and physical faults require
separate bounds; this module does not certify a compiled or physical circuit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real

MAX_DEGREE = 63
TRIG_PRECISION_BITS = 160
MAX_PHASE_MAGNITUDE = 32
MAX_TAYLOR_DEGREE = 256
_DYADIC_SCALE = 1 << TRIG_PRECISION_BITS
_TAYLOR_TOLERANCE = Fraction(1, _DYADIC_SCALE)

GaussianInteger = tuple[int, int]
ExactCoefficient = Fraction | int


def _dyadic_upper(value: Fraction) -> Fraction:
    """Round a nonnegative rational upwards to a 160-bit dyadic lattice."""
    if value < 0:
        raise ValueError("an error bound must be nonnegative")
    quotient, remainder = divmod(value.numerator * _DYADIC_SCALE, value.denominator)
    return Fraction(quotient + bool(remainder), _DYADIC_SCALE)


def _compact_upper(value: Fraction) -> dict[str, str | int]:
    rounded = _dyadic_upper(value)
    return {
        "numerator": str(rounded.numerator),
        "denominator": str(rounded.denominator),
        "rounding": "upwards on a 2^-160 lattice",
    }


def _float_upper(value: Fraction) -> float:
    """Return a floating upper bound, including overflow/underflow cases."""
    try:
        rounded = float(value)
    except OverflowError:
        return math.inf
    # Conversion uses nearest rounding.  The following float is always an
    # upper bound, including when a positive exact value underflows to zero.
    return math.nextafter(rounded, math.inf)


@dataclass(frozen=True)
class PhaseCertificate:
    """Exact bounds and coefficients; metadata contains compact upper bounds."""

    degree: int
    phase_hex: tuple[str, ...]
    target_chebyshev_coefficients: tuple[Fraction, ...]
    reconstructed_chebyshev_coefficients: tuple[Fraction, ...]
    coefficient_l1_error: Fraction
    phase_exponential_error_bound: Fraction
    realization_error_bound: Fraction
    taylor_degrees: tuple[int, ...]

    @property
    def float_upper_bound(self) -> float:
        return _float_upper(self.realization_error_bound)

    @property
    def dyadic_upper_bound(self) -> Fraction:
        return _dyadic_upper(self.realization_error_bound)

    def as_metadata(self) -> dict:
        """JSON-safe evidence without enormous exact coefficient fractions.

        The full exact Fractions remain available on this immutable result.
        To reproduce the bound, retain the target coefficients alongside the
        exact binary phase hex strings.  Every serialized rational bound is
        rounded upwards, and can therefore be used conservatively itself.
        """
        floating = self.float_upper_bound
        return {
            "method": "exact rational Taylor enclosure and polynomial recurrence",
            "degree": self.degree,
            "phase_count": len(self.phase_hex),
            "exact_binary_phase_hex": list(self.phase_hex),
            "target_coefficient_count": len(self.target_chebyshev_coefficients),
            "coefficient_l1_error_upper": _compact_upper(self.coefficient_l1_error),
            "phase_exponential_error_upper": _compact_upper(self.phase_exponential_error_bound),
            "uniform_realization_error_upper": _compact_upper(self.realization_error_bound),
            "uniform_realization_error_float_upper": floating if math.isfinite(floating) else None,
            "float_conversion": "math.nextafter towards +infinity; null means overflow",
            "trig_precision_bits": TRIG_PRECISION_BITS,
            "maximum_phase_magnitude": MAX_PHASE_MAGNITUDE,
            "maximum_taylor_degree": max(self.taylor_degrees),
            "taylor_degree_cap": MAX_TAYLOR_DEGREE,
            "interval": [-1, 1],
            "signal_convention": "D(phi0) B(x) ... B(x) D(phid); real top-left entry",
            "verification_scope": "rigorous rational bound for the ideal signal sequence with exact stored binary phases",
            "excluded_errors": ["matrix loading/access", "gate decomposition or compiler", "concrete angle synthesis", "physical faults"],
        }


def _phase_exponential(phi: float) -> tuple[GaussianInteger, Fraction, int]:
    """Approximate exp(i*phi) by a dyadic Gaussian rational with a bound.

    For real r, the integral Taylor remainder of exp(i*r) has modulus at
    most |r|^(N+1)/(N+1)! because all real-path derivatives have modulus one.
    Both Taylor components are exact rational numbers.  Rounding each to the
    nearest 2^-160 lattice contributes at most 2^-160 in complex modulus
    (the component errors are each <=2^-161).  No libm trig result is used.
    """
    r = Fraction.from_float(phi)
    if abs(r) > MAX_PHASE_MAGNITUDE:
        raise ValueError(f"phase magnitude must not exceed {MAX_PHASE_MAGNITUDE}")
    real, imaginary, term = Fraction(1), Fraction(0), Fraction(1)
    for degree in range(1, MAX_TAYLOR_DEGREE + 1):
        term *= r / degree
        if degree % 4 == 0:
            real += term
        elif degree % 4 == 1:
            imaginary += term
        elif degree % 4 == 2:
            real -= term
        else:
            imaginary -= term
        remainder = abs(term * r / (degree + 1))
        if remainder <= _TAYLOR_TOLERANCE:
            # Fraction.__round__ performs exact nearest rounding, including
            # ties; its integer result incurs <=half a lattice unit.
            dyadic = (round(real * _DYADIC_SCALE), round(imaginary * _DYADIC_SCALE))
            return dyadic, remainder + _TAYLOR_TOLERANCE, degree
    raise ValueError("rational phase Taylor enclosure exceeded its bounded degree")


def _trim(polynomial: list[GaussianInteger]) -> list[GaussianInteger]:
    while len(polynomial) > 1 and polynomial[-1] == (0, 0):
        polynomial.pop()
    return polynomial


def _add(left: list[GaussianInteger], right: list[GaussianInteger], sign: int = 1) -> list[GaussianInteger]:
    result = [(0, 0)] * max(len(left), len(right))
    for k, value in enumerate(left):
        result[k] = value
    for k, (real, imaginary) in enumerate(right):
        r, s = result[k]
        result[k] = (r + sign * real, s + sign * imaginary)
    return _trim(result)


def _multiply_scalar(polynomial: list[GaussianInteger], scalar: GaussianInteger) -> list[GaussianInteger]:
    c, s = scalar
    return _trim([(c * a - s * b, s * a + c * b) for a, b in polynomial])


def _multiply_x(polynomial: list[GaussianInteger]) -> list[GaussianInteger]:
    return _trim([(0, 0)] + polynomial)


def _multiply_one_minus_x_squared(polynomial: list[GaussianInteger]) -> list[GaussianInteger]:
    return _add(polynomial, [(0, 0), (0, 0)] + polynomial, -1)


def _exact_target(coefficients: Mapping[int, ExactCoefficient] | Sequence[ExactCoefficient], degree: int) -> tuple[Fraction, ...]:
    if isinstance(coefficients, Mapping):
        items = tuple(coefficients.items())
    elif isinstance(coefficients, Sequence) and not isinstance(coefficients, (str, bytes)):
        if len(coefficients) > degree + 1:
            raise ValueError("target degree must not exceed the QSP degree")
        items = tuple(enumerate(coefficients))
    else:
        raise ValueError("target must be an exact rational coefficient mapping or sequence")
    result = [Fraction(0)] * (degree + 1)
    for index, value in items:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= degree:
            raise ValueError("target indices must be integers within the QSP degree")
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise ValueError("target coefficients must be exact Fraction or integer values")
        exact = Fraction(value)
        if index % 2 == 0 and exact:
            raise ValueError("the reciprocal target must have odd parity")
        result[index] = exact
    return tuple(result)


def certify_real_qsp_phases(
    phases: Sequence[float],
    target_coefficients: Mapping[int, ExactCoefficient] | Sequence[ExactCoefficient],
) -> PhaseCertificate:
    """Certify uniform real-polynomial error for an odd degree <=63 sequence.

    The first row has the form (a(x),sqrt(1-x*x)*b(x)).  Right multiplication
    by D(phi)B gives a'=z*x*a+conj(z)*(1-x*x)*b and
    b'=z*a-conj(z)*x*b.  All arithmetic in the dyadic recurrence is Gaussian
    integer arithmetic with a known common denominator.  Exact conversion
    of odd powers to Chebyshev coefficients yields a uniform coefficient
    L1 bound because |T_k(x)|<=1 on [-1,1].

    For per-phase exponential errors eps_j, exact D and B have norm one,
    and approximate D has norm <=1+eps_j.  Telescoping therefore gives
    gamma=product_j(1+eps_j)-1.  The returned exact uniform bound is L1+gamma.
    Invalid degree, nonfinite phases, unbounded phase magnitude, or inexact
    targets fail explicitly; numerical phase fitting is not performed here.
    """
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        raise ValueError("phases must be a finite sequence of real numbers")
    degree = len(phases) - 1
    if not 1 <= degree <= MAX_DEGREE or degree % 2 == 0:
        raise ValueError(f"QSP degree must be odd and lie between 1 and {MAX_DEGREE}")
    stored = []
    for phase in phases:
        if isinstance(phase, bool) or not isinstance(phase, Real):
            raise ValueError("phases must be finite real numbers")
        try:
            value = float(phase)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("phases must be finite real numbers") from error
        if not math.isfinite(value):
            raise ValueError("phases must be finite real numbers")
        stored.append(value)
    target = _exact_target(target_coefficients, degree)
    a, b = [(1, 0)], [(0, 0)]
    product_error = Fraction(1)
    taylor_degrees = []
    for index, phase in enumerate(stored):
        z, error, taylor_degree = _phase_exponential(phase)
        conjugate = (z[0], -z[1])
        product_error *= 1 + error
        taylor_degrees.append(taylor_degree)
        if index < degree:
            next_a = _add(_multiply_x(_multiply_scalar(a, z)), _multiply_one_minus_x_squared(_multiply_scalar(b, conjugate)))
            next_b = _add(_multiply_scalar(a, z), _multiply_x(_multiply_scalar(b, conjugate)), -1)
            a, b = next_a, next_b
        else:
            a = _multiply_scalar(a, z)

    # Each phase contributes 2^160 to the denominator.  For odd r,
    # x^r=2^(1-r) sum_{k=0}^{(r-1)/2} binom(r,k) T_(r-2k).
    # An additional denominator factor 2^degree accommodates all powers.
    denominator = 1 << (TRIG_PRECISION_BITS * len(stored) + degree)
    numerators = [0] * (degree + 1)
    for power, (real, imaginary) in enumerate(a):
        if power % 2 == 0:
            if real or imaginary:
                raise ArithmeticError("exact QSP recurrence violated odd parity")
            continue
        if power > degree:
            raise ArithmeticError("exact QSP recurrence exceeded its degree")
        for k in range((power - 1) // 2 + 1):
            numerators[power - 2 * k] += real * math.comb(power, k) * (1 << (degree + 1 - power))
    achieved = tuple(Fraction(numerator, denominator) for numerator in numerators)
    l1 = sum((abs(actual - wanted) for actual, wanted in zip(achieved, target)), Fraction(0))
    gamma = product_error - 1
    return PhaseCertificate(
        degree=degree,
        phase_hex=tuple(phase.hex() for phase in stored),
        target_chebyshev_coefficients=target,
        reconstructed_chebyshev_coefficients=achieved,
        coefficient_l1_error=l1,
        phase_exponential_error_bound=gamma,
        realization_error_bound=l1 + gamma,
        taylor_degrees=tuple(taylor_degrees),
    )
