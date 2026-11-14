# SPDX-License-Identifier: MIT
"""Directed-decimal coefficient certificates for an exact ideal QSP signal.

Inputs are the exact real numbers represented by the supplied binary64 values.
This module does not fit phases or simulate a circuit. Ordinary decimal endpoint
arithmetic rounds outward; sin/cos/exp are enclosed by Taylor polynomials and
real Lagrange remainder bounds, never by Decimal's nearest-rounded exp/sqrt.
The coefficient l1 bound is uniform on [-1,1]. Physical synthesis, compiler
angle changes and a caller's implemented Hamiltonian access need their own
contracts. Original MIT implementation; Python 3.12 decimal is the arithmetic
runtime, not an integrated formal proof assistant.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import math
from typing import Sequence

MAX_DEGREE = 48
MAX_ARGUMENT = 16.0
MAX_PHASE_MAGNITUDE = 4.0
MAX_TAYLOR_DEGREE = 256
MAX_PREPARE_INSTRUCTIONS = 1024
TARGET_KINDS = {"cosine": 0, "sine": 1, "thermal_even": 0, "thermal_odd": 1}


def _precision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 40 <= value <= 160:
        raise ValueError("decimal precision must be an integer from 40 to 160")
    return value


def _point(value) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Boolean is not an exact numeric input")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("numeric input must be finite")
        return Decimal.from_float(value)
    result = value if isinstance(value, Decimal) else Decimal(value)
    if not result.is_finite():
        raise ValueError("numeric input must be finite")
    return result


@dataclass(frozen=True)
class DecimalInterval:
    """Closed interval; each operation uses its own directed Context."""
    lo: Decimal
    hi: Decimal
    precision: int = 80

    def __post_init__(self):
        _precision(self.precision)
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo > self.hi:
            raise ValueError("invalid finite decimal interval")

    @classmethod
    def point(cls, value, precision=80):
        d = _point(value)
        return cls(d, d, precision)

    @classmethod
    def rational(cls, value: Fraction, precision=80):
        return cls.point(value.numerator, precision) / value.denominator

    def _other(self, value):
        if isinstance(value, DecimalInterval):
            if value.precision != self.precision:
                raise ValueError("interval precisions must match")
            return value
        return DecimalInterval.point(value, self.precision)

    def _contexts(self):
        return (Context(prec=self.precision, rounding=ROUND_FLOOR),
                Context(prec=self.precision, rounding=ROUND_CEILING))

    def __add__(self, value):
        other = self._other(value)
        lower, upper = self._contexts()
        return DecimalInterval(lower.add(self.lo, other.lo), upper.add(self.hi, other.hi), self.precision)

    __radd__ = __add__

    def __neg__(self):
        return DecimalInterval(self.hi.copy_negate(), self.lo.copy_negate(), self.precision)

    def __sub__(self, value):
        return self + (-self._other(value))

    def __rsub__(self, value):
        return self._other(value) - self

    def __mul__(self, value):
        other = self._other(value)
        lower, upper = self._contexts()
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return DecimalInterval(min(lower.multiply(a, b) for a, b in pairs),
                               max(upper.multiply(a, b) for a, b in pairs), self.precision)

    __rmul__ = __mul__

    def __truediv__(self, value):
        other = self._other(value)
        if other.lo <= 0 <= other.hi:
            raise ValueError("interval division by an interval containing zero")
        lower, upper = self._contexts()
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return DecimalInterval(min(lower.divide(a, b) for a, b in pairs),
                               max(upper.divide(a, b) for a, b in pairs), self.precision)

    def power(self, exponent: int):
        if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
            raise ValueError("interval power requires a nonnegative integer")
        result, base = self.point(1, self.precision), self
        while exponent:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent //= 2
        return result

    def abs_upper(self) -> Decimal:
        return max(self.lo.copy_abs(), self.hi.copy_abs())

    def record(self):
        return {"lower": str(self.lo), "upper": str(self.hi)}


def _outward_float(upper: Decimal) -> float:
    value = float(upper)
    if not math.isfinite(value):
        raise ValueError("bound is not representable by a finite binary64 upper bound")
    if Decimal.from_float(value) < upper:
        value = math.nextafter(value, math.inf)
    return value


def _power_factorial(value: DecimalInterval, degree: int, divisor=1) -> DecimalInterval:
    return value.power(degree) / (math.factorial(degree) * divisor)


def _remainder_interval(upper: Decimal, precision: int):
    return DecimalInterval(upper.copy_negate(), upper, precision)


def _trig_enclosure(angle: DecimalInterval, odd: bool) -> tuple[DecimalInterval, int]:
    """Real Taylor enclosure; the next zero coefficient permits D+2 remainder."""
    magnitude = DecimalInterval.point(angle.abs_upper(), angle.precision)
    goal = Decimal("1e-" + str(angle.precision-8))
    degree = 1 if odd else 0
    term = angle if odd else DecimalInterval.point(1, angle.precision)
    total = term
    square = angle * angle
    while True:
        tail = _power_factorial(magnitude, degree+2).hi
        if tail <= goal:
            return total + _remainder_interval(tail, angle.precision), degree
        degree += 2
        if degree > MAX_TAYLOR_DEGREE:
            raise ValueError("trigonometric Taylor materialization cap exceeded")
        term = -(term * square / (degree*(degree-1)))
        total = total + term


def _negative_exp_enclosure(argument: DecimalInterval) -> tuple[DecimalInterval, int]:
    """exp(-a), a>=0; real remainder <= a^(D+1)/(D+1)! on [-a,0]."""
    if argument.lo < 0:
        raise ValueError("negative-exp enclosure requires nonnegative argument")
    magnitude = DecimalInterval.point(argument.abs_upper(), argument.precision)
    goal = Decimal("1e-" + str(argument.precision-8))
    term = total = DecimalInterval.point(1, argument.precision)
    for degree in range(MAX_TAYLOR_DEGREE+1):
        tail = _power_factorial(magnitude, degree+1).hi
        if tail <= goal:
            return total + _remainder_interval(tail, argument.precision), degree
        term = -(term * argument / (degree+1))
        total = total + term
    raise ValueError("exponential Taylor materialization cap exceeded")


@dataclass(frozen=True)
class _ComplexInterval:
    real: DecimalInterval
    imag: DecimalInterval

    @classmethod
    def zero(cls, precision):
        z = DecimalInterval.point(0, precision)
        return cls(z, z)

    def __add__(self, other):
        return _ComplexInterval(self.real+other.real, self.imag+other.imag)

    def __neg__(self):
        return _ComplexInterval(-self.real, -self.imag)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return _ComplexInterval(self.real*other.real-self.imag*other.imag,
                                self.real*other.imag+self.imag*other.real)

    def conjugate(self):
        return _ComplexInterval(self.real, -self.imag)


def _qsp_monomial_intervals(phases: tuple[float, ...], precision: int):
    """Top row is [a(x), sqrt(1-x²)b(x)]; no sampled reconstruction."""
    zero = _ComplexInterval.zero(precision)
    one = _ComplexInterval(DecimalInterval.point(1, precision), zero.imag)
    a, b, trig_degrees = [one], [], []
    for position, phase in enumerate(phases):
        angle = DecimalInterval.point(phase, precision)
        cosine, dc = _trig_enclosure(angle, False)
        sine, ds = _trig_enclosure(angle, True)
        z = _ComplexInterval(cosine, sine)
        a = [value*z for value in a]
        b = [value*z.conjugate() for value in b]
        trig_degrees.append({"cosine": dc, "sine": ds})
        if position == len(phases)-1:
            break
        # Right multiply by Bx: a'=x a+(1-x²)b, b'=a-x b.
        anew = [zero for _ in range(max(len(a)+1, len(b)+2))]
        bnew = [zero for _ in range(max(len(a), len(b)+1))]
        for k, value in enumerate(a):
            anew[k+1] = anew[k+1]+value
            bnew[k] = bnew[k]+value
        for k, value in enumerate(b):
            anew[k] = anew[k]+value
            anew[k+2] = anew[k+2]-value
            bnew[k+1] = bnew[k+1]-value
        a, b = anew, bnew
    return [value.real for value in a], trig_degrees


def _monomial_to_chebyshev(values: Sequence[DecimalInterval]):
    precision = values[0].precision
    result = [DecimalInterval.point(0, precision) for _ in values]
    for k, coefficient in enumerate(values):
        if k % 2 == 0:
            result[0] = result[0] + coefficient*DecimalInterval.rational(
                Fraction(math.comb(k, k//2), 2**k), precision)
        for j in range((k+1)//2):
            result[k-2*j] = result[k-2*j] + coefficient*DecimalInterval.rational(
                Fraction(2*math.comb(k, j), 2**k), precision)
    return result


def _target_taylor(kind: str, parameter: DecimalInterval, degree: int):
    parity = TARGET_KINDS[kind]
    degree += (degree-parity) % 2
    if degree > MAX_TAYLOR_DEGREE:
        raise ValueError("target Taylor degree cap exceeded")
    zero = DecimalInterval.point(0, parameter.precision)
    values = [zero for _ in range(degree+1)]
    exp_degree = None
    scale = DecimalInterval.point(1, parameter.precision) / 2
    if kind.startswith("thermal"):
        exponential, exp_degree = _negative_exp_enclosure(parameter)
        scale = exponential / 2
        if kind == "thermal_odd":
            scale = -scale
    for k in range(parity, degree+1, 2):
        coefficient = _power_factorial(parameter, k)*scale
        if kind == "cosine" and (k//2) % 2:
            coefficient = -coefficient
        if kind == "sine" and ((k-1)//2) % 2:
            coefficient = -coefficient
        values[k] = coefficient
    magnitude = DecimalInterval.point(parameter.abs_upper(), parameter.precision)
    tail = _power_factorial(magnitude, degree+2, 2).hi
    # For thermal derivatives, e^-a cosh(ax), e^-a sinh(ax) <=1 in magnitude.
    return _monomial_to_chebyshev(values), tail, degree, exp_degree


def certify_real_qsp(phases: Sequence[float], *, target_kind: str,
                     parameter: float, precision: int = 80,
                     target_taylor_degree: int | None = None) -> dict:
    """Uniform interval error certificate for exact supplied phases and signal.

    Targets: cos(tau*x)/2, sin(tau*x)/2, or e^-a cosh(ax)/2 and
    -e^-a sinh(ax)/2. The phases, tau/a are exact binary64 real values.
    Certificate validity does not require the numerical fit to have converged.
    Caller enforces its requested allowance using the outward upper bound.
    """
    precision = _precision(precision)
    if not isinstance(target_kind, str) or target_kind not in TARGET_KINDS:
        raise ValueError("unsupported analytic QSP target kind")
    values = tuple(phases)
    if not values or len(values)-1 > MAX_DEGREE:
        raise ValueError("QSP certificate degree cap exceeded")
    if (len(values)-1) % 2 != TARGET_KINDS[target_kind]:
        raise ValueError("phase degree has the wrong target parity")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or abs(p) > MAX_PHASE_MAGNITUDE for p in values):
        raise ValueError("phases must be finite binary64 values with magnitude at most4")
    values = tuple(float(p) for p in values)
    if isinstance(parameter, bool) or not isinstance(parameter, (int, float)) or not math.isfinite(parameter):
        raise ValueError("target parameter must be a finite binary64 value")
    if abs(parameter) > MAX_ARGUMENT or (target_kind.startswith("thermal") and parameter < 0):
        raise ValueError("target parameter outside analytic certificate cap")
    parameter = float(parameter)
    argument = DecimalInterval.point(parameter, precision)
    chosen = max(64, len(values)+32) if target_taylor_degree is None else target_taylor_degree
    if isinstance(chosen, bool) or not isinstance(chosen, int) or chosen < len(values)-1:
        raise ValueError("target Taylor degree must be integer and at least QSP degree")
    monomials, trig_degrees = _qsp_monomial_intervals(values, precision)
    achieved = _monomial_to_chebyshev(monomials)
    target, tail, chosen, exp_degree = _target_taylor(target_kind, argument, chosen)
    zero = DecimalInterval.point(0, precision)
    residuals, total = [], zero
    for k in range(max(len(achieved), len(target))):
        observed = achieved[k] if k < len(achieved) else zero
        expected = target[k] if k < len(target) else zero
        residual = (observed-expected).abs_upper()
        residuals.append(residual)
        total = total + residual
    upper = (total+tail).hi
    return {
        "certified": True,
        "certificate_scope": "uniform mathematical real-QSP response on[-1,1] for exact supplied binary64 phases/parameter and an exact Hermitian-involution signal; not a physical or compiler certificate",
        "target_kind": target_kind, "parameter": parameter,
        "exact_parameter_decimal": str(Decimal.from_float(parameter)),
        "degree": len(values)-1, "decimal_precision": precision,
        "uniform_error_upper": _outward_float(upper),
        "uniform_error_upper_decimal": str(upper),
        "coefficient_residual_l1_upper_decimal": str(total.hi),
        "analytic_target_taylor_remainder_upper_decimal": str(tail),
        "target_taylor_degree": chosen, "negative_exp_taylor_degree": exp_degree,
        "phase_trig_taylor_degrees": trig_degrees,
        "achieved_chebyshev_intervals": [value.record() for value in achieved],
        "target_taylor_chebyshev_intervals": [value.record() for value in target],
        "coefficient_residual_upper_decimals": [str(value) for value in residuals],
        "method": "outward ordinary Decimal arithmetic; analytic real Taylor sin/cos/exp enclosures; exact binomial monomial-to-Chebyshev identity; |T_k(x)|<=1",
        "excluded": ["error of implemented Pauli access/loader around the intended normalizedH",
                     "normalization/time/beta multiplication rounding relative to separately stated exact inputs",
                     "compiler changes to source phases or controls", "rotation synthesis", "reset/readout faults", "physical errors"],
        "formal_proof_assistant": None,
    }


def exact_parameter_rounding(terms, multiplier: float, *, thermal: bool = False) -> dict:
    """Exact-rational normalization/product gap, separately from access loading."""
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier):
        raise ValueError("multiplier must be finite")
    coefficients = tuple(coefficient for coefficient, _ in terms)
    if not coefficients or any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for c in coefficients):
        raise ValueError("finite real Pauli coefficients required")
    alpha = sum((abs(Fraction.from_float(float(c))) for c in coefficients), Fraction(0))
    if alpha == 0 or (thermal and multiplier < 0):
        raise ValueError("positive normalization and valid multiplier required")
    exact = alpha*Fraction.from_float(float(multiplier))/(2 if thermal else 1)
    try:
        parameter, alpha_float = float(exact), float(alpha)
    except OverflowError:
        raise ValueError("normalization/parameter overflowed binary64") from None
    if not math.isfinite(parameter) or not math.isfinite(alpha_float):
        raise ValueError("parameter overflowed binary64")
    gap = abs(Fraction.from_float(parameter)-exact)
    interval = DecimalInterval.rational(gap)
    return {
        "normalization_exact_numerator": str(alpha.numerator),
        "normalization_exact_denominator": str(alpha.denominator),
        "normalization_float": alpha_float,
        "selected_parameter_float": parameter,
        "exact_parameter_numerator": str(exact.numerator),
        "exact_parameter_denominator": str(exact.denominator),
        "parameter_rounding_error_upper": _outward_float(interval.hi),
        "parameter_rounding_error_upper_decimal": str(interval.hi),
        "decoded_target_parameter_error_upper": _outward_float((interval*(2 if thermal else 1)).hi),
        "scope": "exact binary64 input coefficients/multiplier; idealH/alpha_exact. Thermal |d exp[-a(x+1)]/da|<=2, dynamics |d exp[-itau x]/dtau|<=1. Access loading not included",
    }


def _same_definition_exact(left, right) -> bool:
    """Structural comparison without Qiskit's tolerance-based gate equality."""
    if left is None or right is None:
        return left is right
    if (left.num_qubits != right.num_qubits or left.num_clbits != right.num_clbits
            or float(left.global_phase) != float(right.global_phase) or len(left.data) != len(right.data)):
        return False
    for li, ri in zip(left.data, right.data, strict=True):
        a, b = li.operation, ri.operation
        if (a.base_class is not b.base_class or a.name != b.name
                or a.num_qubits != b.num_qubits or a.num_clbits != b.num_clbits
                or len(a.params) != len(b.params)):
            return False
        if any(type(x) is not type(y) or x != y for x, y in zip(a.params, b.params, strict=True)):
            return False
        if [left.find_bit(q).index for q in li.qubits] != [right.find_bit(q).index for q in ri.qubits]:
            return False
        if [left.find_bit(c).index for c in li.clbits] != [right.find_bit(c).index for c in ri.clbits]:
            return False
    return True


def certify_prepared_pauli_weights(terms, preparation, *, precision: int = 80) -> dict:
    """Enclose an actual binary64-angle RY/X loader, without circuit simulation.

    The input is its label-only logical PREPARE circuit, not its decomposition.
    Supported logical operations are X, RY and controlled RY with explicit
    controls. For an exact indexed signed-Pauli SELECT with identity padding,
    the returned l1 error bounds the accepted normalized-Hamiltonian error.
    Controls have their exact declared logical meaning. Floating decompositions
    of those controls still require a separate realization error contract.
    This is not a whole-unitary PREPARE or compiler/synthesis certificate.
    """
    precision = _precision(precision)
    # Gate names are not semantic evidence: custom gates can be named "x" or
    # "ry". Accept only the canonical library primitives/control definitions.
    from qiskit.circuit import ControlledGate
    from qiskit.circuit.library import XGate, RYGate, CRYGate
    values = tuple(terms)
    if not values or len(values) > 64:
        raise ValueError("prepared-weight certificate requires1..64 Pauli terms")
    coefficients = []
    n_system = None
    for coefficient, label in values:
        if (isinstance(coefficient, bool) or not isinstance(coefficient, (int, float, complex))
                or complex(coefficient).imag != 0 or not math.isfinite(complex(coefficient).real)):
            raise ValueError("prepared weights require finite real Pauli coefficients")
        if not isinstance(label, str) or not label or any(letter not in "IXYZ" for letter in label):
            raise ValueError("invalid Pauli label")
        n_system = len(label) if n_system is None else n_system
        if len(label) != n_system or coefficient == 0:
            raise ValueError("weights must use the same nonzero term list as indexed SELECT")
        coefficients.append(abs(Fraction.from_float(float(complex(coefficient).real))))
    alpha = sum(coefficients, Fraction(0))
    m = (len(values)-1).bit_length()
    if preparation.num_qubits != m or preparation.num_clbits or preparation.num_parameters:
        raise ValueError("PREPARE label width/classical/parameter mismatch")
    if len(preparation.data) > MAX_PREPARE_INSTRUCTIONS:
        raise ValueError("PREPARE instruction certificate cap exceeded")
    if float(preparation.global_phase) != 0:
        raise ValueError("prepared-weight certificate supports zero global phase")
    zero = DecimalInterval.point(0, precision)
    state = [zero for _ in range(2**m)]
    state[0] = DecimalInterval.point(1, precision)
    for instruction in preparation.data:
        operation = instruction.operation
        qubits = [preparation.find_bit(q).index for q in instruction.qubits]
        if operation.base_class is XGate and len(qubits) == 1 and not operation.params:
            mask = 1 << qubits[0]
            for index in range(len(state)):
                if not index & mask:
                    state[index], state[index | mask] = state[index | mask], state[index]
            continue
        base = getattr(operation, "base_gate", operation)
        controls = int(getattr(operation, "num_ctrl_qubits", 0))
        if (base.base_class is not RYGate or len(qubits) != controls+1 or len(base.params) != 1
                or (controls == 0 and operation.base_class is not RYGate)
                or (controls and operation.base_class not in (CRYGate, ControlledGate))):
            raise ValueError("unsupported PREPARE operation for weight certificate")
        angle = float(base.params[0])
        if not math.isfinite(angle) or abs(angle) > MAX_PHASE_MAGNITUDE:
            raise ValueError("PREPARE angle outside finite certificate cap")
        half = DecimalInterval.point(angle, precision) / 2
        cosine, _ = _trig_enclosure(half, False)
        sine, _ = _trig_enclosure(half, True)
        target_mask = 1 << qubits[-1]
        control_state = int(getattr(operation, "ctrl_state", 0)) if controls else 0
        if not 0 <= control_state < 2**controls:
            raise ValueError("invalid controlled-RY predicate")
        if controls:
            expected = RYGate(angle).control(controls, ctrl_state=control_state, annotated=False)
            if not _same_definition_exact(operation.definition, expected.definition):
                raise ValueError("noncanonical controlled-RY definition")
        for index in range(len(state)):
            if index & target_mask:
                continue
            if any(bool(index & (1 << q)) != bool(control_state & (1 << k))
                   for k, q in enumerate(qubits[:-1])):
                continue
            left, right = state[index], state[index | target_mask]
            state[index] = cosine*left-sine*right
            state[index | target_mask] = sine*left+cosine*right
    probabilities, target_weights, errors = [], [], []
    total, padding = zero, zero
    lower_context, upper_context = zero._contexts()
    for index, amplitude in enumerate(state):
        square_upper = max(upper_context.multiply(v, v) for v in (amplitude.lo, amplitude.hi))
        square_lower = (Decimal(0) if amplitude.lo <= 0 <= amplitude.hi else
                        min(lower_context.multiply(v, v) for v in (amplitude.lo, amplitude.hi)))
        probability = DecimalInterval(max(Decimal(0), square_lower), min(Decimal(1), square_upper), precision)
        target = (DecimalInterval.rational(coefficients[index]/alpha, precision)
                  if index < len(values) else zero)
        error = (probability-target).abs_upper()
        probabilities.append(probability)
        target_weights.append(target)
        errors.append(error)
        total = total + error
        if index >= len(values):
            padding = padding + probability
    return {
        "certified": True,
        "certificate_scope": "logical RY/X PREPARE with exact binary64 angles and exact declared controls; accepted Hamiltonian block error requires exact indexed signed-Pauli SELECT with identity padding. Floating nested control decompositions, whole PREPARE-unitary and compiled/synthesized realization are not certified",
        "weight_l1_error_upper": _outward_float(total.hi),
        "weight_l1_error_upper_decimal": str(total.hi),
        "accepted_hamiltonian_block_error_upper": _outward_float(total.hi),
        "padding_probability_upper_decimal": str(padding.hi),
        "label_qubits": m, "term_count": len(values), "decimal_precision": precision,
        "normalization_exact_numerator": str(alpha.numerator),
        "normalization_exact_denominator": str(alpha.denominator),
        "prepared_probability_intervals": [p.record() for p in probabilities],
        "ideal_normalized_weight_intervals": [p.record() for p in target_weights],
        "weight_residual_upper_decimals": [str(value) for value in errors],
        "method": "directed interval RY/X amplitude propagation; squared probability enclosures; exact-rational coefficient normalization; triangle inequality for norm-one Paulis",
        "excluded": ["floating nested control decompositions", "compiler angle/control changes", "rotation synthesis", "physical/reset/readout faults"],
    }
