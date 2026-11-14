"""Independent analytic checks of the rational QSP realization certificate."""
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from openquantum.reciprocal_phase_certificate import (
    TRIG_PRECISION_BITS,
    _dyadic_upper,
    _float_upper,
    _phase_exponential,
    certify_real_qsp_phases,
)
from openquantum.simulation.qsp import qsp_real_response


def test_zero_phase_sequences_have_exact_chebyshev_polynomial():
    # B(x) is an involution: an odd number of B calls is exactly B(x).
    for degree in (1, 3, 17, 63):
        certificate = certify_real_qsp_phases([0.0] * (degree + 1), {1: Fraction(1)})
        assert certificate.reconstructed_chebyshev_coefficients[1] == 1
        assert all(coefficient == 0 for k, coefficient in enumerate(certificate.reconstructed_chebyshev_coefficients) if k != 1)
        assert certificate.coefficient_l1_error == 0
        epsilon = Fraction(1, 1 << TRIG_PRECISION_BITS)
        assert certificate.phase_exponential_error_bound == (1 + epsilon) ** (degree + 1) - 1
        assert 0 < certificate.realization_error_bound < Fraction(1, 10**45)


def test_linear_signal_has_analytic_phase_sum_response():
    phases = (0.125, -0.375)
    certificate = certify_real_qsp_phases(phases, [0, Fraction(1)])
    actual = float(certificate.reconstructed_chebyshev_coefficients[1])
    assert actual == pytest.approx(math.cos(sum(phases)), abs=2e-16)
    assert certificate.reconstructed_chebyshev_coefficients[0] == 0
    assert float(certificate.realization_error_bound) == pytest.approx(1 - math.cos(sum(phases)), abs=2e-16)


def test_old_cubic_frozen_phases_receive_tight_uniform_certificate():
    phases = (0.043750941193310945, 0.4773037119407437, 0.5955800476346843, 0.20148137088407908)
    target = {1: Fraction(1, 2), 3: Fraction(-1, 4)}
    certificate = certify_real_qsp_phases(phases, target)
    assert certificate.realization_error_bound < Fraction(1, 10**15)
    assert certificate.phase_exponential_error_bound < Fraction(1, 10**46)
    assert certificate.phase_hex == tuple(value.hex() for value in phases)
    assert certificate.realization_error_bound == certificate.coefficient_l1_error + certificate.phase_exponential_error_bound
    # The independent binary64 signal evaluator corroborates the result; its
    # own rounding error is not part of the rigorous rational certificate.
    grid = np.linspace(-1, 1, 101)
    achieved = np.polynomial.chebyshev.chebval(grid, [float(c) for c in certificate.reconstructed_chebyshev_coefficients])
    assert np.max(np.abs(achieved - qsp_real_response(grid, phases))) < 2e-15


def test_high_degree_exact_recurrence_agrees_with_independent_signal_model():
    phases = tuple(np.random.default_rng(71).uniform(-1, 1, 64))
    certificate = certify_real_qsp_phases(phases, {1: Fraction(1, 3)})
    assert certificate.degree == 63
    assert all(value == 0 for k, value in enumerate(certificate.reconstructed_chebyshev_coefficients) if k % 2 == 0)
    assert certificate.phase_exponential_error_bound < Fraction(1, 10**45)
    grid = np.linspace(-1, 1, 257)
    achieved = np.polynomial.chebyshev.chebval(grid, [float(c) for c in certificate.reconstructed_chebyshev_coefficients])
    assert np.max(np.abs(achieved - qsp_real_response(grid, phases))) < 4e-15


def test_taylor_enclosure_is_bounded_for_maximum_phase_and_is_conjugate_symmetric():
    positive, positive_error, positive_degree = _phase_exponential(32.0)
    negative, negative_error, negative_degree = _phase_exponential(-32.0)
    assert positive_degree <= 256
    assert positive_degree == negative_degree
    assert positive_error == negative_error
    assert negative == (positive[0], -positive[1])
    assert positive_error <= Fraction(2, 1 << TRIG_PRECISION_BITS)


def test_serialized_bounds_and_float_conversion_never_round_down():
    certificate = certify_real_qsp_phases((0.125, -0.375), {1: Fraction(1)})
    assert Fraction.from_float(certificate.float_upper_bound) >= certificate.realization_error_bound
    assert certificate.dyadic_upper_bound >= certificate.realization_error_bound
    metadata = certificate.as_metadata()
    json.dumps(metadata, allow_nan=False)
    for field, bound in (
        ("coefficient_l1_error_upper", certificate.coefficient_l1_error),
        ("phase_exponential_error_upper", certificate.phase_exponential_error_bound),
        ("uniform_realization_error_upper", certificate.realization_error_bound),
    ):
        record = metadata[field]
        outward = Fraction(int(record["numerator"]), int(record["denominator"]))
        assert outward >= bound
        assert outward.denominator.bit_length() <= 161
        assert outward - bound < Fraction(1, 1 << TRIG_PRECISION_BITS)
    tiny = Fraction(1, 1 << 2000)
    assert Fraction.from_float(_float_upper(tiny)) >= tiny
    assert _float_upper(Fraction(10**1000)) == math.inf
    assert _dyadic_upper(Fraction(0)) == 0
    with pytest.raises(ValueError):
        _dyadic_upper(Fraction(-1))


@pytest.mark.parametrize("phases", [(), (0.0,), (0.0, 0.0, 0.0), (0.0,) * 66, (math.nan, 0.0), (math.inf, 0.0), (33.0, 0.0), (True, 0.0), (1j, 0.0), (np.complex128(1 + 1j), 0.0), (np.bool_(True), 0.0), ("0.25", 0.0)])
def test_invalid_phase_contract_fails(phases):
    with pytest.raises(ValueError):
        certify_real_qsp_phases(phases, {1: Fraction(1)})


@pytest.mark.parametrize("target", [{0: Fraction(1)}, {2: Fraction(1)}, {-1: Fraction(1)}, {True: Fraction(1)}, {1: 0.5}, {1: True}, [0, 1, 0], "0,1"])
def test_target_must_be_exact_and_odd_within_degree(target):
    with pytest.raises(ValueError):
        certify_real_qsp_phases((0.0, 0.0), target)
