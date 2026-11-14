"""Independent exact/high-precision checks of the scoped interval validator."""
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction
import json
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, ControlledGate
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector
from scipy.special import jv, ive

from openquantum.simulation import qsp_certificate as qc
from openquantum.simulation.lcu import build_pauli_lcu
from openquantum.simulation.qsp import COSINE_PHASES
from openquantum.simulation.qsp_evolution import SINE_PHASES
from openquantum.simulation.qsp_phase_generation import fit_real_qsp_phases


def _trig_reference(x, odd=False):
    """110-digit nearest-rounded independent scalar reference, not a certificate."""
    term = x if odd else Decimal(1)
    result = term
    for k in range(1, 150):
        denominator = (2*k)*(2*k+1) if odd else (2*k-1)*(2*k)
        term = -term*x*x/Decimal(denominator)
        result += term
    return result


def _complex_product(a, b):
    return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])


def _signal_reference(phases, x):
    with localcontext() as context:
        context.prec = 110
        context.Emax, context.Emin = 999999, -999999
        context.rounding = ROUND_HALF_EVEN
        x = Decimal.from_float(float(x))
        s = (Decimal(1)-x*x).sqrt()
        zero, one = (Decimal(0), Decimal(0)), (Decimal(1), Decimal(0))
        response = [[one, zero], [zero, one]]
        signal = [[(x, Decimal(0)), (s, Decimal(0))], [(s, Decimal(0)), (-x, Decimal(0))]]
        for position, phase in enumerate(phases):
            phi = Decimal.from_float(float(phase))
            c, z = _trig_reference(phi), _trig_reference(phi, True)
            for row in range(2):
                response[row][0] = _complex_product(response[row][0], (c, z))
                response[row][1] = _complex_product(response[row][1], (c, -z))
            if position < len(phases)-1:
                product = [[zero, zero], [zero, zero]]
                for i in range(2):
                    for j in range(2):
                        parts = [_complex_product(response[i][k], signal[k][j]) for k in range(2)]
                        product[i][j] = (sum(p[0] for p in parts), sum(p[1] for p in parts))
                response = product
        return response[0][0][0]


def _analytic_reference(kind, parameter, x):
    with localcontext() as context:
        context.prec = 110
        p, x = Decimal.from_float(float(parameter)), Decimal.from_float(float(x))
        if kind == "cosine":
            return _trig_reference(p*x)/2
        if kind == "sine":
            return _trig_reference(p*x, True)/2
        positive, negative = (p*x).exp(), (-p*x).exp()
        if kind == "thermal_even":
            return (-p).exp()*(positive+negative)/4
        return -(-p).exp()*(positive-negative)/4


def _coefficient_target(kind, parameter, degree):
    c = np.zeros(degree+1)
    if kind == "cosine":
        c[0] = jv(0, parameter)/2
        for k in range(2, degree+1, 2):
            c[k] = (-1)**(k//2)*jv(k, parameter)
    elif kind == "sine":
        for k in range(1, degree+1, 2):
            c[k] = (-1)**((k-1)//2)*jv(k, parameter)
    elif kind == "thermal_even":
        c[0] = ive(0, parameter)/2
        for k in range(2, degree+1, 2):
            c[k] = ive(k, parameter)
    else:
        for k in range(1, degree+1, 2):
            c[k] = -ive(k, parameter)
    return c


@pytest.mark.parametrize("left,right", [
    ((Fraction(-2, 7), Fraction(3, 5)), (Fraction(-9, 11), Fraction(-1, 3))),
    ((Fraction(1, 17), Fraction(5, 13)), (Fraction(2, 19), Fraction(4, 7))),
])
def test_interval_operations_enclose_independent_rational_endpoints(left, right):
    a0, a1 = [qc.DecimalInterval.rational(v, 40) for v in left]
    b0, b1 = [qc.DecimalInterval.rational(v, 40) for v in right]
    a = qc.DecimalInterval(a0.lo, a1.hi, 40)
    b = qc.DecimalInterval(b0.lo, b1.hi, 40)
    for operation, reference in [
        (a+b, [x+y for x in left for y in right]),
        (a-b, [x-y for x in left for y in right]),
        (a*b, [x*y for x in left for y in right]),
        (a/b, [x/y for x in left for y in right]),
    ]:
        assert Fraction(operation.lo) <= min(reference)
        assert Fraction(operation.hi) >= max(reference)


def test_ambient_context_does_not_change_interval_certificate():
    before = qc.certify_real_qsp(COSINE_PHASES, target_kind="cosine", parameter=4.1)
    with localcontext() as context:
        context.prec = 5
        context.Emax, context.Emin = 5, -5
        context.rounding = ROUND_HALF_EVEN
        after = qc.certify_real_qsp(COSINE_PHASES, target_kind="cosine", parameter=4.1)
    assert before == after


@pytest.mark.parametrize("angle", [-4.0, -math.pi, -0.3, 0.0, 0.2, math.pi, 4.0])
@pytest.mark.parametrize("odd", [False, True])
def test_phase_trig_enclosures_include_high_precision_independent_reference(angle, odd):
    interval, _ = qc._trig_enclosure(qc.DecimalInterval.point(angle), odd)
    with localcontext() as context:
        context.prec = 110
        reference = _trig_reference(Decimal.from_float(angle), odd)
    assert interval.lo <= reference <= interval.hi


@pytest.mark.parametrize("parameter", [0.0, 0.1, 1.3, 8.0, 16.0])
def test_negative_exponential_is_enclosed_without_using_decimal_exp(parameter):
    interval, _ = qc._negative_exp_enclosure(qc.DecimalInterval.point(parameter))
    with localcontext() as context:
        context.prec = 110
        reference = Decimal.from_float(-parameter).exp()
    assert interval.lo <= reference <= interval.hi


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 8])
def test_qsp_coefficient_intervals_include_independent_complex_signal_matrices(degree):
    phases = tuple(np.random.default_rng(degree+12).uniform(-1.2, 1.2, degree+1))
    monomials, _ = qc._qsp_monomial_intervals(phases, 80)
    chebyshev = qc._monomial_to_chebyshev(monomials)
    for x in [-1.0, -0.7, 0.0, 0.25, 1.0]:
        xi = qc.DecimalInterval.point(x)
        t0, t1 = qc.DecimalInterval.point(1), xi
        value = chebyshev[0]*t0
        for k in range(1, len(chebyshev)):
            tk = t1 if k == 1 else 2*xi*t1-t0
            if k > 1:
                t0, t1 = t1, tk
            value = value+chebyshev[k]*tk
        reference = _signal_reference(phases, x)
        assert value.lo <= reference <= value.hi


@pytest.mark.parametrize("kind", ["cosine", "sine", "thermal_even", "thermal_odd"])
@pytest.mark.parametrize("parameter", [0.0, 1.3, 6.0])
def test_fitted_phase_certificate_bounds_true_target_at_high_precision(kind, parameter):
    degree = 24 if qc.TARGET_KINDS[kind] == 0 else 25
    phases, _ = fit_real_qsp_phases(_coefficient_target(kind, parameter, degree))
    certificate = qc.certify_real_qsp(phases, target_kind=kind, parameter=parameter)
    upper = Decimal(certificate["uniform_error_upper_decimal"])
    assert Decimal.from_float(certificate["uniform_error_upper"]) >= upper
    for x in [-1.0, -0.75, -0.1, 0.0, 0.25, 0.8, 1.0]:
        with localcontext() as context:
            context.prec = 100
            error = abs(_signal_reference(phases, x)-_analytic_reference(kind, parameter, x))
        assert error <= upper
    assert certificate["formal_proof_assistant"] is None


def test_validator_can_certify_a_poor_phase_sequence_with_a_large_bound():
    result = qc.certify_real_qsp((0., 0., 0.), target_kind="cosine", parameter=1.3)
    assert result["uniform_error_upper"] > .5
    assert result["certified"]  # Valid error bound; not an accuracy acceptance decision.


def test_known_frozen_phases_have_new_interval_evidence_independent_of_grid_fit():
    c = qc.certify_real_qsp(COSINE_PHASES, target_kind="cosine", parameter=4.1)
    s = qc.certify_real_qsp(SINE_PHASES, target_kind="sine", parameter=4.1)
    assert 3e-9 < c["uniform_error_upper"] < 5e-9
    assert 1e-10 < s["uniform_error_upper"] < 1e-9


@pytest.mark.parametrize("terms,multiplier,thermal", [
    (((.6, "Z"), (.4, "X")), .3, False),
    (((.6, "ZZ"), (.35, "IX"), (.35, "XI")), 1.25, True),
    (((-.7, "Y"),), -.25, False),
])
def test_exact_parameter_rounding_encloses_independent_rational_gap(terms, multiplier, thermal):
    result = qc.exact_parameter_rounding(terms, multiplier, thermal=thermal)
    alpha = sum((abs(Fraction.from_float(c)) for c, _ in terms), Fraction())
    exact = alpha*Fraction.from_float(multiplier)/(2 if thermal else 1)
    gap = abs(Fraction.from_float(result["selected_parameter_float"])-exact)
    assert Fraction(Decimal(result["parameter_rounding_error_upper_decimal"])) >= gap
    assert Fraction.from_float(result["parameter_rounding_error_upper"]) >= gap


@pytest.mark.parametrize("terms", [
    ((-.7, "Y"),), ((.8, "X"), (.2, "Z")),
    ((.6, "ZZ"), (-.35, "IX"), (.35, "XI")),
    ((.2, "I"), (.3, "X"), (.4, "Y"), (.1, "Z")),
])
def test_prepared_weight_certificate_covers_canonical_loader_and_exact_normalization(terms):
    prepare = build_pauli_lcu(terms).prepare
    result = qc.certify_prepared_pauli_weights(terms, prepare)
    assert result["weight_l1_error_upper"] < 2e-14
    assert Decimal.from_float(result["weight_l1_error_upper"]) >= Decimal(result["weight_l1_error_upper_decimal"])
    probabilities = Statevector.from_instruction(prepare).probabilities()
    for value, interval in zip(probabilities, result["prepared_probability_intervals"], strict=True):
        # Qiskit's floating simulation is separately numerical, not an interval oracle.
        assert float(interval["lower"])-3e-15 <= value <= float(interval["upper"])+3e-15
    if len(terms) == 1:
        assert result["weight_l1_error_upper"] == 0


def test_loader_angles_are_certified_as_supplied_and_error_is_detected():
    prepare = QuantumCircuit(1)
    prepare.ry(.7, 0)
    result = qc.certify_prepared_pauli_weights(((.8, "X"), (.2, "Z")), prepare)
    with localcontext() as context:
        context.prec = 110
        c = _trig_reference(Decimal.from_float(.7)/2)
        p = c*c
        actual_l1 = 2*abs(p-Decimal(".8"))
    assert result["weight_l1_error_upper"] > .1
    assert Decimal(result["weight_l1_error_upper_decimal"]) >= actual_l1


@pytest.mark.parametrize("name", ["x", "ry"])
def test_custom_gate_names_cannot_impersonate_certified_primitive_semantics(name):
    custom = Gate(name, 1, [.7] if name == "ry" else [])
    wrong = QuantumCircuit(1)
    wrong.h(0)
    custom.definition = wrong
    prepare = QuantumCircuit(1)
    prepare.append(custom, [0])
    with pytest.raises(ValueError, match="unsupported PREPARE"):
        qc.certify_prepared_pauli_weights(((.8, "X"), (.2, "Z")), prepare)


def test_noncanonical_controlled_definition_is_not_accepted_with_tolerant_gate_equality():
    prepare = QuantumCircuit(2)
    gate = RYGate(.7).control(1, annotated=False).to_mutable()
    definition = gate.definition.copy()
    for index, instruction in enumerate(definition.data):
        if instruction.operation.name == "ry":
            changed = instruction.operation.to_mutable()
            changed.params[0] += 1e-12
            definition.data[index] = instruction.replace(operation=changed)
            break
    gate.definition = definition
    prepare.append(gate, [0, 1])
    with pytest.raises(ValueError, match="noncanonical"):
        qc.certify_prepared_pauli_weights(((1., "X"), (.2, "Y"), (.3, "Z")), prepare)


@pytest.mark.parametrize("kwargs", [
    {"phases": [True]}, {"phases": [float("nan")]}, {"phases": [4.1]},
    {"parameter": float("inf")}, {"parameter": 16.1}, {"precision": 39},
    {"target_taylor_degree": -1}, {"target_taylor_degree": 257},
    {"target_kind": "unsupported"},
])
def test_invalid_certificate_domains_fail_closed(kwargs):
    arguments = {"phases": [0.], "target_kind": "cosine", "parameter": 1.}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        qc.certify_real_qsp(**arguments)


def test_finite_serialization_and_positive_subnormal_upper_conversion():
    report = qc.certify_real_qsp((0.,), target_kind="cosine", parameter=0.)
    json.dumps(report, allow_nan=False)
    tiny = Decimal("1e-400")
    assert qc._outward_float(tiny) > 0
    assert Decimal.from_float(qc._outward_float(tiny)) >= tiny


def test_unrepresentable_normalization_rejects_cleanly_even_at_zero_time():
    with pytest.raises(ValueError, match="overflowed"):
        qc.exact_parameter_rounding([(1e308, "X")]*64, 0.)


def test_open_control_predicate_and_zero_padding_have_explicit_weight_bounds():
    prepare = QuantumCircuit(2)
    prepare.ry(.7, 0)
    prepare.append(RYGate(.9).control(1, ctrl_state=0, annotated=False), [0, 1])
    prepare.append(RYGate(.6).control(1, ctrl_state=1, annotated=False), [0, 1])
    terms = ((.5, "X"), (.3, "Y"), (.2, "Z"))
    result = qc.certify_prepared_pauli_weights(terms, prepare)
    probabilities = Statevector.from_instruction(prepare).probabilities()
    assert result["weight_l1_error_upper"] >= sum(abs(p-t) for p, t in zip(probabilities, [.5,.3,.2,0.]))-2e-14
    assert Decimal(result["padding_probability_upper_decimal"]) > 0


def test_prepare_instruction_cap_is_separate_from_label_width():
    prepare = QuantumCircuit(1)
    for _ in range(qc.MAX_PREPARE_INSTRUCTIONS+1):
        prepare.ry(.1, 0)
    with pytest.raises(ValueError, match="instruction certificate cap"):
        qc.certify_prepared_pauli_weights(((.5, "X"), (.5, "Z")), prepare)
