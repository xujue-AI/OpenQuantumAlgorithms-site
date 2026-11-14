"""Retargetable phase algebra, numerical fits and explicit bounded failures."""
import math
from unittest.mock import patch

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval
from scipy.special import ive, jv

from openquantum.simulation import qsp_phase_generation as phase
from openquantum.simulation.qsp import qsp_real_response


def _coefficients(kind, parameter, degree):
    result = np.zeros(degree+1)
    if kind == "cosine":
        result[0] = jv(0, parameter)/2
        for k in range(2, degree+1, 2):
            result[k] = (-1)**(k//2)*jv(k, parameter)
    elif kind == "sine":
        for k in range(1, degree+1, 2):
            result[k] = (-1)**((k-1)//2)*jv(k, parameter)
    elif kind == "thermal_even":
        result[0] = ive(0, parameter)/2
        for k in range(2, degree+1, 2):
            result[k] = ive(k, parameter)
    elif kind == "thermal_odd":
        for k in range(1, degree+1, 2):
            result[k] = -ive(k, parameter)
    else:
        raise AssertionError(kind)
    return result


@pytest.mark.parametrize("kind,degree", [("cosine", 24), ("sine", 25),
                                         ("thermal_even", 24), ("thermal_odd", 25)])
@pytest.mark.parametrize("parameter", [0, .5, 1.3, 4.1, 8, 12])
def test_actual_phase_generation_retargets_independent_chebyshev_polynomials(kind, degree, parameter):
    coefficients = _coefficients(kind, parameter, degree)
    phases, report = phase.fit_real_qsp_phases(coefficients, tolerance=1e-11)
    grid = np.linspace(-1, 1, 10001)
    # Existing independent implementation fixes the written B/D convention.
    np.testing.assert_allclose(qsp_real_response(grid, phases), chebval(grid, coefficients), atol=2e-13, rtol=0)
    assert len(phases) == degree+1
    assert report["degree"] == degree and report["parity"] == degree%2
    assert report["total_function_evaluations"] <= 6
    assert report["grid_max_residual"] < 2e-13
    assert report["numerically_reconstructed_coefficient_l1_discrepancy"] < 2e-13
    assert report["certified"] is False
    assert "interval" in report["validation_scope"]


@pytest.mark.parametrize("degree", [0, 1, 2, 7, 16, 31, 32])
def test_full_ordered_phase_jacobian_matches_independent_central_differences(degree):
    rng = np.random.default_rng(4+degree)
    phases = rng.normal(0, .4, degree+1)
    grid = np.array([-1., -.61, .0, .37, 1.])
    actual, jacobian = phase.qsp_response_and_jacobian(grid, phases)
    np.testing.assert_allclose(actual, qsp_real_response(grid, phases), atol=2e-14, rtol=0)
    h = 2e-7
    for j in range(degree+1):
        offset = np.zeros(degree+1)
        offset[j] = h
        numeric = (qsp_real_response(grid, phases+offset)-qsp_real_response(grid, phases-offset))/(2*h)
        np.testing.assert_allclose(jacobian[:, j], numeric, atol=3e-8, rtol=0)


@pytest.mark.parametrize("degree", [1, 2, 3, 8, 15, 16, 31, 32])
def test_reduced_phase_conversion_matches_standard_complex_signal(degree):
    rng = np.random.default_rng(93+degree)
    theta = rng.normal(0, .2, degree//2+1)
    phases = phase.symmetric_qsp_phases(theta, degree)
    for x in [-1., -.65, .0, .28, 1.]:
        s = np.sqrt(1-x*x)
        w = np.array([[x, 1j*s], [1j*s, x]])
        matrix = np.eye(2, dtype=complex)
        for j in range(degree+1):
            value = theta[min(j, degree-j)]
            matrix = matrix@np.diag([np.exp(1j*value), np.exp(-1j*value)])
            if j < degree:
                matrix = matrix@w
        expected = (-1)**(degree//2+1)*matrix[0, 0].imag
        assert float(phase.real_qsp_response(x, phases)) == pytest.approx(expected, abs=3e-14)


@pytest.mark.parametrize("degree", [1, 2, 8, 15, 24, 31, 32])
def test_general_bounded_parity_targets_and_seed_reproducibility(degree):
    rng = np.random.default_rng(217+degree)
    coefficients = np.zeros(degree+1)
    active = rng.normal(size=degree//2+1)
    active *= .49/sum(abs(active))
    coefficients[degree%2::2] = active
    phases, report = phase.fit_real_qsp_phases(coefficients, seed=31)
    repeat, repeated_report = phase.fit_real_qsp_phases(coefficients, seed=31)
    np.testing.assert_array_equal(phases, repeat)
    assert report["grid_max_residual"] < 2e-13
    assert repeated_report["total_function_evaluations"] == report["total_function_evaluations"]
    assert report["target_chebyshev_coefficients"] == coefficients.tolist()


@pytest.mark.parametrize("tau", [-.7, -4.1, -10.])
def test_negative_times_preserve_cosine_and_reverse_the_sine_target(tau):
    grid = np.linspace(-1, 1, 1001)
    for kind, degree, sign in [("cosine", 24, 1), ("sine", 25, -1)]:
        coefficients = _coefficients(kind, tau, degree)
        phases, report = phase.fit_real_qsp_phases(coefficients)
        positive = _coefficients(kind, -tau, degree)
        np.testing.assert_allclose(qsp_real_response(grid, phases),
                                   sign*chebval(grid, positive), atol=2e-13, rtol=0)
        assert report["grid_max_residual"] < 2e-13


@pytest.mark.parametrize("degree,value", [(0, .5), (2, -.3), (12, .0),
                                         (1, -.4), (3, .0), (23, .5)])
def test_zero_and_constant_linear_targets_retain_requested_degree(degree, value):
    coefficients = np.zeros(degree+1)
    coefficients[degree%2] = value
    phases, report = phase.fit_real_qsp_phases(coefficients)
    assert len(phases) == degree+1
    assert report["total_function_evaluations"] == 0
    assert report["method"] == "analytic constant/linear phases"
    target = value*np.linspace(-1, 1, 57) if degree%2 else np.full(57, value)
    np.testing.assert_allclose(qsp_real_response(np.linspace(-1, 1, 57), phases), target, atol=5e-14, rtol=0)


def test_multidimensional_signal_jacobian_shape_and_no_endpoint_singularity():
    values = np.array([[-1., -.2], [.3, 1.]])
    response, jacobian = phase.qsp_response_and_jacobian(values, [.2, -.1, .3])
    assert response.shape == (2, 2) and jacobian.shape == (2, 2, 3)
    assert np.all(np.isfinite(jacobian))
    np.testing.assert_allclose(response, qsp_real_response(values, [.2, -.1, .3]), atol=1e-14)


@pytest.mark.parametrize("coefficients", [[], [.2]*34, [.2, .1, .2], [np.nan],
                                            [np.inf], [1j], [True], [.9]])
def test_unsupported_or_nonreal_targets_fail_closed(coefficients):
    with pytest.raises(ValueError):
        phase.fit_real_qsp_phases(coefficients)


@pytest.mark.parametrize("kwargs", [{"tolerance": 1e-16}, {"tolerance": True},
                                      {"max_restarts": 0}, {"max_restarts": 9},
                                      {"max_nfev": 0}, {"max_nfev": 5001},
                                      {"max_seconds": 0}, {"max_seconds": np.inf},
                                      {"max_seconds": 121}, {"seed": -1}, {"seed": True},
                                      {"fit_nodes": 2}, {"fit_nodes": 257},
                                      {"grid_points": 32}, {"grid_points": 16386}])
def test_fit_budgets_have_explicit_ceilings(kwargs):
    with pytest.raises(ValueError):
        phase.fit_real_qsp_phases([.1, 0, .2], **kwargs)


@pytest.mark.parametrize("signal", [1.00001, np.inf, np.nan, [1j], [0]*16386])
def test_signal_domain_and_sample_size_fail_closed(signal):
    with pytest.raises(ValueError):
        phase.real_qsp_response(signal, [.2])


def test_exhausted_optimizer_budget_raises_instead_of_returning_unverified_phases():
    coefficients = _coefficients("cosine", 4.1, 14)
    with pytest.raises(RuntimeError, match="failed residual"):
        phase.fit_real_qsp_phases(coefficients, tolerance=1e-11, max_restarts=1, max_nfev=1)


def test_elapsed_time_guard_is_checked_inside_the_optimizer():
    coefficients = _coefficients("sine", 3.0, 15)
    # A deterministic clock progression avoids timing-sensitive wall-clock tests.
    with patch.object(phase.time, "perf_counter", side_effect=[0., 2.]):
        with pytest.raises(RuntimeError, match="elapsed-time"):
            phase.fit_real_qsp_phases(coefficients, max_seconds=1.)


def test_elapsed_time_guard_also_covers_direct_phase_final_validation():
    with patch.object(phase.time, "perf_counter", side_effect=[0., 0., 2.]):
        with pytest.raises(RuntimeError, match="final validation"):
            phase.fit_real_qsp_phases([.5], max_seconds=1.)
