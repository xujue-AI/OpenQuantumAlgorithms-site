"""Independent splitting, conditioning and full-workload resource checks."""
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from qiskit.circuit.library import RXGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from openquantum.dynamics_resources import (
    DynamicsTask, attempts_per_slot, choose_steps, commutator_l1_bounds,
    estimate_static_dynamics_resources, DISTANCE_DOMAIN, RETARGETED_PHASES,
    MAX_PHASE_FIT_SIGNAL_VALUES, fit_retargeted_phases, qsp_response_jacobian,
    postselected_contract, product_formula_error_bound, rational,
    source_angle_rounding_bound,
)
from openquantum.hamiltonian_simulation import (
    compare_product_formula, pauli_matrix, product_formula_circuit,
)
from openquantum.resource_estimation import CircuitResourceBudget, SurfaceCodeProfile, synthesis_model_error

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dynamic_resource_script", ROOT / "scripts/generate_dynamics_resources.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def independent_groups(n, J, h):
    a = np.zeros((2**n, 2**n), complex)
    b = a.copy()
    for j in range(n - 1):
        label = ["I"] * n
        label[n - j - 1] = label[n - j - 2] = "Z"
        a += J * pauli_matrix("".join(label))
    for j in range(n):
        label = ["I"] * n
        label[n - j - 1] = "X"
        b += h * pauli_matrix("".join(label))
    return a, b


def commutator(a, b):
    return a @ b - b @ a


def independent_scalar_qsp_response(x, phases):
    signal = np.array([[x, math.sqrt(1-x*x)], [math.sqrt(1-x*x), -x]], complex)
    response = np.eye(2, dtype=complex)
    for index, phase in enumerate(phases):
        response = response @ np.diag([np.exp(1j*phase), np.exp(-1j*phase)])
        if index < len(phases)-1:
            response = response @ signal
    return response[0, 0].real


@pytest.mark.parametrize("phases", [(.3,), (.2, -.7, .4), (.2, .1, -.3, .7),
                                   RETARGETED_PHASES[2]["cosine"], RETARGETED_PHASES[4]["sine"]])
def test_analytic_phase_jacobian_matches_independent_matrix_finite_differences(phases):
    values = np.array([-1., -.79, -.1, 0., .34, .93, 1.])
    response, jacobian = qsp_response_jacobian(values, phases)
    expected = [independent_scalar_qsp_response(x, phases) for x in values]
    assert np.allclose(response, expected, atol=3e-14, rtol=0)
    epsilon = 1e-6
    for column in range(len(phases)):
        upper, lower = np.array(phases), np.array(phases)
        upper[column] += epsilon
        lower[column] -= epsilon
        derivative = [(independent_scalar_qsp_response(x, upper)-independent_scalar_qsp_response(x, lower))
                      / (2*epsilon) for x in values]
        assert np.allclose(jacobian[:, column], derivative, atol=2e-9, rtol=0)


@pytest.mark.parametrize("values,phases", [([], [0]), ([1.01], [0]), ([float("nan")], [0]),
                                         ([[0]], [0]), ([0], []), ([0], [float("inf")]),
                                         ([0], [0]*19), ([0]* (MAX_PHASE_FIT_SIGNAL_VALUES+1), [0])])
def test_phase_jacobian_rejects_invalid_or_unbounded_arrays(values, phases):
    with pytest.raises(ValueError):
        qsp_response_jacobian(values, phases)


@pytest.mark.parametrize("n", [2, 4])
def test_stored_and_reproduced_phases_have_fixed_coupling_targets(n):
    phases = RETARGETED_PHASES[n]
    assert phases["tau"] == pytest.approx(n-1+.7*n)
    assert len(phases["cosine"]) == 15 and len(phases["sine"]) == 16
    fitted = fit_retargeted_phases(n)
    assert fitted["certified"] is False
    values = np.linspace(-1, 1, 1001)
    for kind, function in (("cosine", np.cos), ("sine", np.sin)):
        stored_response = qsp_response_jacobian(values, phases[kind])[0]
        fitted_response = qsp_response_jacobian(values, fitted[kind]["phases"])[0]
        assert np.max(np.abs(stored_response-function(phases["tau"]*values)/2)) < phases["constructor_tolerance"]
        assert np.max(np.abs(stored_response-fitted_response)) < 1e-8
        assert fitted[kind]["verification"]["certified"] is False


def test_continuation_uses_parity_sized_variables_and_shared_response_cache(monkeypatch):
    from scipy import optimize
    from openquantum import dynamics_resources
    original_solver = optimize.least_squares
    original_response = dynamics_resources.qsp_response_jacobian
    response_calls = [0]
    variable_counts = []
    def counted_response(*args):
        response_calls[0] += 1
        return original_response(*args)
    def checked_solver(fun, initial, **options):
        variable_counts.append(len(initial))
        assert len(initial) == 8 and options["max_nfev"] == 500
        candidate = initial + np.linspace(.0001, .0008, len(initial))
        fun(candidate)
        after_residual = response_calls[0]
        assert options["jac"](candidate).shape[1] == 8
        assert response_calls[0] == after_residual
        return original_solver(fun, initial, **options)
    monkeypatch.setattr(dynamics_resources, "qsp_response_jacobian", counted_response)
    monkeypatch.setattr(optimize, "least_squares", checked_solver)
    fitted = fit_retargeted_phases(2)
    assert variable_counts == [8]*18
    assert response_calls[0] == sum(fitted[kind]["response_jacobian_evaluations"] for kind in ("cosine", "sine"))
    assert all(fitted[kind]["phase_variables_per_step"] == 8 for kind in ("cosine", "sine"))


@pytest.mark.parametrize("n", [2, 4])
def test_continuation_acceptance_is_invariant_to_rowspace_coordinates(n, monkeypatch):
    original_svd = np.linalg.svd
    rotation = np.linalg.qr(np.random.default_rng(271).normal(size=(8, 8)))[0]
    changed_bases = []
    def rotated_rowspace(matrix, *args, **options):
        left, singular_values, right = original_svd(matrix, *args, **options)
        if matrix.shape in ((33, 15), (35, 16)):
            # Any orthonormal coordinates span the same eight-dimensional
            # rowspace. The solver must depend on that space, not SVD signs.
            right = right.copy()
            right[:8] = rotation @ right[:8]
            changed_bases.append(matrix.shape)
        return left, singular_values, right
    monkeypatch.setattr(np.linalg, "svd", rotated_rowspace)
    fitted = fit_retargeted_phases(n)
    assert len(changed_bases) == 18
    values = np.linspace(-1, 1, 1001)
    for kind in ("cosine", "sine"):
        expected = qsp_response_jacobian(values, RETARGETED_PHASES[n][kind])[0]
        actual = qsp_response_jacobian(values, fitted[kind]["phases"])[0]
        assert np.max(np.abs(actual-expected)) < 1e-8
        assert fitted[kind]["verification"]["certified"] is False


def test_continuation_enforces_combined_response_evaluation_cap(monkeypatch):
    from scipy import optimize
    from openquantum import dynamics_resources
    original_response = dynamics_resources.qsp_response_jacobian
    calls = [0]
    def counted_response(*args):
        calls[0] += 1
        return original_response(*args)
    def exhausted_solver(fun, initial, **options):
        for count in range(501):
            fun(initial + np.linspace(-1, 1, len(initial)) * ((count+1)*.0001))
        raise AssertionError("Combined evaluation cap was not enforced")
    monkeypatch.setattr(dynamics_resources, "qsp_response_jacobian", counted_response)
    monkeypatch.setattr(optimize, "least_squares", exhausted_solver)
    with pytest.raises(RuntimeError, match="response-evaluation cap"):
        fit_retargeted_phases(2)
    assert calls[0] == 500


@pytest.mark.parametrize("n", [True, 2., 3, 5])
def test_retargeted_cases_and_fitter_reject_unsupported_sizes(n):
    with pytest.raises(ValueError):
        script.retargeted_qsp_case(n)
    with pytest.raises(ValueError):
        fit_retargeted_phases(n)


@pytest.mark.parametrize("timeout", [True, 0, 61, float("inf"), "60"])
def test_retargeted_fitter_rejects_unbounded_or_invalid_timeouts(timeout):
    with pytest.raises(ValueError):
        fit_retargeted_phases(2, component_timeout_seconds=timeout)


def test_retargeted_component_deadline_includes_final_phase_verification(monkeypatch):
    import time
    from openquantum.simulation import qsp
    clock = [0.]
    original_verify = qsp.verify_cosine_phases
    def late_verify(*args):
        result = original_verify(*args)
        clock[0] = 61.
        return result
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(qsp, "verify_cosine_phases", late_verify)
    with pytest.raises(TimeoutError, match="during verification"):
        fit_retargeted_phases(2, component_timeout_seconds=60)


@pytest.mark.parametrize("n", [2, 4])
def test_actual_retargeted_circuits_meet_independent_dynamics_and_full_attempt_contract(n):
    measured, compiled, ledger, budget, observable_budget = script.retargeted_qsp_case(n)
    a, b = independent_groups(n, 1, .7)
    reference = expm(-1j*(a+b))[:, 0]
    z = pauli_matrix("I"*(n-1)+"Z")
    check = ledger["numerical_checks"]
    assert check["observable_exact"] == pytest.approx(np.vdot(reference, z@reference).real, abs=1e-13)
    assert check["observable_error"] < ledger["task"]["algorithm_bias"]
    assert check["all_input_conditional_Pauli_bias_projection"] < ledger["task"]["algorithm_bias"]
    assert check["accepted_block_basis_trajectories_per_source_or_lowering"] == 1 << n
    assert check["source_to_compiled_accepted_block_operator_error"] < 1e-10
    assert check["seeded_full_state_lowering_error"] < 1e-10
    assert check["guard_is_certificate"] is False
    assert measured.num_qubits == compiled.num_qubits == ledger["logical_qubits"] <= 9
    assert compiled.count_ops()["measure"] == ledger["logical_qubits"]
    assert ledger["access"]["total_A_or_inverse_calls"] == 1
    assert ledger["access"]["uncontrolled_block_encoding_calls"] == 14
    assert ledger["access"]["controlled_block_encoding_calls"] == 1
    assert ledger["phase_tau"] == pytest.approx(n-1+.7*n)
    assert budget.executions == ledger["complete_attempts"] == ledger["attempts_per_accepted_slot"]*13811
    assert observable_budget is None
    assert budget.synthesis_model_error_per_execution == ledger["postselection_contract"]["synthesis_model_error_per_attempt"]
    assert ledger["postselection_contract"]["certified_confidence_lower_bound"] is None
    if n == 2:
        unitary = compiled.remove_final_measurements(inplace=False)
        unitary.metadata = {"n_system": n}
        assert np.allclose(script.projected_retargeted_block(unitary), Operator(unitary).data[:4, :4], atol=1e-12)


@pytest.mark.parametrize("n,width", [(3, 8), (4, 10), (4, 3)])
def test_retargeted_basis_trajectory_checks_have_explicit_width_and_size_caps(n, width):
    circuit = QuantumCircuit(width)
    circuit.metadata = {"n_system": n}
    with pytest.raises(ValueError):
        script.projected_retargeted_block(circuit)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("J,h", [(1, .7), (-.4, .3), (.2, -.8)])
def test_analytic_Pauli_bounds_cover_independent_dense_commutators(n, J, h):
    a, b = independent_groups(n, J, h)
    values = commutator_l1_bounds(n, J, h)
    matrices = {"AB": commutator(a, b), "AAB": commutator(a, commutator(a, b)),
                "BBA": commutator(b, commutator(b, a))}
    for key, matrix in matrices.items():
        assert np.linalg.norm(matrix, 2) <= float(values[key]) + 1e-12


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n,time", [(2, -.5), (3, .3), (4, 1.)])
def test_selected_steps_are_minimal_and_actual_unitary_respects_the_bound(order, n, time):
    allowance = Fraction(1, 200)
    steps = choose_steps(n, time, order, allowance)
    assert product_formula_error_bound(n, time, steps, order) <= allowance
    if steps > 1:
        assert product_formula_error_bound(n, time, steps - 1, order) > allowance
    result = compare_product_formula(n, time, steps, order)
    assert result["operator_error"] <= float(product_formula_error_bound(n, time, steps, order)) + 1e-11
    assert result["observable_error"] <= 2 * result["operator_error"] + 1e-12


def test_exact_fraction_step_boundaries_and_commuting_limits():
    assert choose_steps(2, 1, 1, Fraction(1, 200)) == 280
    assert choose_steps(3, 1, 2, Fraction(1, 200)) == 21
    assert choose_steps(6, 1, 2, Fraction(1, 200)) == 33
    assert choose_steps(4, 0, 1, Fraction(1, 200)) == 1
    assert choose_steps(1, 1, 2, Fraction(1, 200)) == 1
    assert choose_steps(5, 1, 2, Fraction(1, 200), h=0) == 1


def test_rounding_checker_does_not_accept_an_empty_or_wrong_width_formula():
    with pytest.raises(ValueError, match="counts"):
        source_angle_rounding_bound(QuantumCircuit(3), n=3, time=1, steps=1, order=1)
    with pytest.raises(ValueError, match="width"):
        source_angle_rounding_bound(QuantumCircuit(2), n=3, time=1, steps=1, order=1)


@pytest.mark.parametrize("order", [1, 2])
def test_source_binary_angles_have_a_separate_small_rational_rounding_bound(order):
    source = product_formula_circuit(3, time=Fraction(1, 3), steps=19, order=order)
    bound = source_angle_rounding_bound(source, n=3, time=Fraction(1, 3), steps=19, order=order)
    assert isinstance(bound, Fraction)
    assert 0 <= bound < Fraction(1, 10**12)
    altered = source.copy()
    # Changing one actual rotation must appear in this ledger.
    for index, instruction in enumerate(altered.data):
        if instruction.operation.name == "rx":
            replacement = RXGate(float(instruction.operation.params[0]) + .1)
            altered.data[index] = instruction.replace(operation=replacement)
            break
    large = source_angle_rounding_bound(altered, n=3, time=Fraction(1, 3), steps=19, order=order)
    assert float(large) >= .05 - 1e-12


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "0.7", Fraction(10**1000)])
def test_bad_or_unrepresentable_scalars_are_rejected(value):
    with pytest.raises(ValueError):
        rational(value)


@pytest.mark.parametrize("arguments", [(True, 1, 1, .005), (2, 1, True, .005), (65, 1, 2, .005),
                                      (2, 1, 2, 0), (6, 10000, 1, .005)])
def test_bad_or_unbounded_splitting_request_is_rejected(arguments):
    with pytest.raises(ValueError):
        choose_steps(*arguments)


def test_default_task_uses_a_single_shared_one_percent_failure_contract():
    task = DynamicsTask()
    assert task.shots == 13811
    assert task.failure_allowance == pytest.approx(.01)
    assert task.algorithm_bias + task.synthesis_bias + task.sampling_error == pytest.approx(.05)
    fine = DynamicsTask.at_accuracy(.025)
    assert fine.shots > task.shots
    assert fine.algorithm_bias == .005
    assert fine.synthesis_bias == .005


def test_conditioning_TV_bound_on_independent_finite_laws():
    rng = np.random.default_rng(31)
    for _ in range(1000):
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        event_p, event_q = sum(p[:3]), sum(q[:3])
        tv = np.sum(np.abs(p - q)) / 2
        conditional_tv = np.sum(np.abs(p[:3] / event_p - q[:3] / event_q)) / 2
        assert event_q >= event_p - tv - 1e-14
        assert conditional_tv <= tv / event_p + 1e-14


@pytest.mark.parametrize("p", [.0625, .47265625, .908447265625, .961318970063])
def test_capped_conditioned_workload_charges_every_failed_attempt(p):
    task = DynamicsTask()
    contract = postselected_contract(p, task)
    lower = contract["conditional_acceptance_lower_after_synthesis_model"]
    assert lower == pytest.approx(.995 * p)
    assert contract["conditional_synthesis_Pauli_bias_allowance"] == pytest.approx(.01)
    cap = contract["attempts_per_accepted_slot"]
    assert task.shots * (1 - lower)**cap <= task.acceptance_failure
    assert cap == 1 or task.shots * (1 - lower)**(cap - 1) > task.acceptance_failure
    assert contract["maximum_complete_attempts"] == cap * task.shots
    assert contract["physical_failure_allowance_per_complete_attempt"] * contract["maximum_complete_attempts"] == pytest.approx(.005)
    independent = task.shots * sum((1 - lower)**j for j in range(cap))
    assert contract["conditional_expected_stopped_attempts_all_slots"] == pytest.approx(independent)
    assert contract["certified_confidence_lower_bound"] is None
    assert "complete final joint" in " ".join(contract["assumptions"])


def test_acceptance_cap_handles_perfect_and_tiny_probability():
    assert attempts_per_slot(1, 13811, .001) == 1
    with pytest.raises(ValueError, match="bounded"):
        attempts_per_slot(1e-10, 13811, .001)
    for p in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            attempts_per_slot(p, 1, .001)


def test_canonical_grid_coverage_and_complete_static_PF_case():
    rows = script.canonical_product_parameters()
    assert len(rows) == 18 and len(set(rows)) == 18
    assert {n for n, t, order, e in rows if t == 1 and e == .05} == set(range(2, 7))
    measured, compiled, ledger, budget, observable = script.product_case(2, 1, 2, .05)
    assert measured.num_qubits == 2
    assert compiled.count_ops()["measure"] == 2
    assert budget.executions == ledger["complete_attempts"] == 13811
    assert budget == observable.resource_budget()
    assert ledger["numerical_checks"]["performed"]
    assert float(Fraction(ledger["analytic_error"]["product_formula_operator_bound_exact"])) < .005


def test_rectangular_spacetime_and_serial_peak_are_not_integrated_occupancy():
    point = {"peak_physical_qubits": 100, "serial_runtime_all_executions_ns": 30}
    value = script.add_point_metrics(point, {"complete_attempts": 3})
    assert value["peak_physical_qubits"] == 100
    assert value["peak_times_capped_runtime_qubit_seconds"] == pytest.approx(3e-6)
    assert "not integrated" in value["spacetime_scope"]


def test_actual_native_adapter_records_common_data_and_factory_domains(tmp_path):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rx(.213, 0)
    circuit.cx(0, 1)
    circuit.rz(.317, 1)
    circuit.measure_all()
    budget = CircuitResourceBudget(1e-4, .001, 100)
    result = estimate_static_dynamics_resources(
        circuit, budget=budget, profile=SurfaceCodeProfile(physical_error_rate=1e-4), cache_dir=tmp_path,
    )
    assert result["surface_code_data_distance_domain"] == list(DISTANCE_DOMAIN)
    assert result["factory_surface_code_distance_domain"] == list(DISTANCE_DOMAIN)
    assert result["isa_graph_pruning"] is False
    assert result["canonical_gate_counts"]["MEAS_Z"] == 2
    model = result["synthesis"]
    assert model["model_error_per_execution"] == pytest.approx(
        model["num_arbitrary_rotations"] * synthesis_model_error(model["fixed_num_ts_per_rotation"])
    )
    for point in result["pareto_points"]:
        assert point["data_code_distance"] in DISTANCE_DOMAIN
        assert point["serial_runtime_all_executions_ns"] == 100 * point["runtime_per_execution_ns"]
        assert point["physical_failure_union_bound_all_executions"] <= .001 + 1e-10


@pytest.mark.parametrize("distances", [(), (2,), (3, 3), (True,), (63,), tuple(range(3, 100, 2))])
def test_native_distance_search_is_bounded_and_explicit(distances, tmp_path):
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    with pytest.raises(ValueError, match="distances"):
        estimate_static_dynamics_resources(
            circuit, budget=CircuitResourceBudget(0., .01), profile=SurfaceCodeProfile(),
            cache_dir=tmp_path, distance_domain=distances,
        )


@pytest.mark.parametrize("relative", ["assets/generated/grover", "assets/generated/resources/new", "assets/generated/simulation/new"])
def test_output_guard_protects_frozen_and_prior_algorithm_artifacts(relative):
    with pytest.raises(ValueError):
        script.guard_output(ROOT / relative)


def test_actual_generated_data_and_all_physical_cases():
    path = ROOT / "assets/generated/dynamics-resources/manifest.json"
    if not path.exists():
        pytest.skip("Canonical dynamics physical artifacts have not been generated yet")
    report = json.loads(path.read_text())
    assert report["status"] == "computed_conditional_model"
    assert report["canonical_cases"] == 24
    assert report["actual_physical_queries"] == 48
    assert report["QSP_retargeted_size_cases"] == 2
    assert {case["system_qubits"] for case in report["cases"]
            if case["method"] == "QSP_retargeted"} == {2, 4}
    assert report["frozen_files_unchanged"]
    assert report["frozen_file_sha256_before"] == report["frozen_file_sha256_after"]
    import hashlib
    for name, digest in report["frozen_file_sha256_before"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for case in report["cases"]:
        directory = path.parent / case["case_id"]
        assert hashlib.sha256((directory / "measured-trial.qasm").read_bytes()).hexdigest() == case["qasm_sha256"]
        assert len(case["physical_reports"]) == 2
        for filename in case["physical_reports"]:
            projection = json.loads((directory / filename).read_text())
            assert projection["resource_estimate_kind"] == "conditional_mean_fit_resource_projection"
            assert projection["canonical_gate_counts"]["MEAS_Z"] == case["logical_qubits"]
            assert projection["budget"]["executions"] == case["complete_attempts"]
            for point in projection["pareto_points"]:
                assert point["physical_failure_union_bound_all_executions"] <= .005 + 1e-9


def test_logical_only_generation_removes_stale_physical_claims_and_results(tmp_path):
    output = tmp_path / "logical-only"
    old_case = output / "pf2-n2-t1-eps0.05"
    old_case.mkdir(parents=True)
    (old_case / "frozen-p-1e-3.json").write_text("stale")
    (output / "qsp-size-resources.svg").write_text("stale")
    report = script.generate(output, estimate=False)
    assert report["status"] == "logical_only"
    assert report["canonical_cases"] == 24
    assert report["actual_physical_queries"] == 0
    assert report["QSP_physical_size_sweep"].startswith("Not computed")
    assert not list(output.glob("*.svg"))
    for case in report["cases"]:
        assert not case["physical_reports"]
        assert not list((output / case["case_id"]).glob("frozen-p-*.json"))
