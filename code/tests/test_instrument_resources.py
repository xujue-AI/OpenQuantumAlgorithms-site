"""Complete-instrument semantics, actual synthesis, and physical ledger checks."""
import itertools
import importlib.util
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

from openquantum.instrument_resources import (
    InstrumentProfile, _ceil_div, _costs, _factory_retry_cap, estimate_instrument, majority_path_error,
    reset_kraus, simulate_trial, synthesize_instrument, t_injection_kraus,
)
from openquantum.simulation.open_systems import build_amplitude_damping
from openquantum.simulation.lindblad import build_lindblad_circuit
from openquantum.simulation.thermal_filter import build_thermal_trial


def generator_module():
    path = Path(__file__).resolve().parents[1]/"scripts/generate_instrument_resources.py"
    spec = importlib.util.spec_from_file_location("instrument_generator_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry_caps_and_time_ceilings_are_exact_at_large_integer_boundaries():
    assert _factory_retry_cap(0, .01, .001) == 0
    assert _factory_retry_cap(1, .01, .0001) == 2
    assert _factory_retry_cap(10**29, .01, .001) == 16
    with pytest.raises(ValueError, match="retry cap"):
        _factory_retry_cap(10**29+1, .01, .001)
    with pytest.raises(ValueError, match="retry cap"):
        _factory_retry_cap(10**400, .01, .001)
    assert _ceil_div(10**30+1, 1000) == 10**27+1
    assert _ceil_div(10**30, 1000) == 10**27


def test_generator_revision_requires_exact_git_root(tmp_path):
    generator = generator_module()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "root"], check=True)
    source = tmp_path/"source.py"
    source.write_text("# test\n")
    record = generator.source_revision(tmp_path, [source])
    assert len(record["source_commit_at_generation"]) == 40
    assert record["dirty_source_paths"] == ["?? source.py"]
    nested = tmp_path/"download"
    nested.mkdir()
    assert generator.source_revision(nested, []) == generator.source_revision(tmp_path/"absent", [])
    assert generator.source_revision(nested, [source])["source_commit_at_generation"] is None


def test_generator_protects_source_frozen_assets_and_public_output_from_cache(tmp_path):
    generator = generator_module()
    source = tmp_path/"src/module.py"
    source.parent.mkdir()
    source.write_text("# test\n")
    cache = tmp_path.parent/"private-factory-cache"
    for relative in ("src", "assets/generated/grover", "assets/generated/resources", "scripts", ".git"):
        with pytest.raises(ValueError, match="protected"):
            generator.validate_paths(tmp_path/relative, cache, root=tmp_path, repository=True, sources={"module.py": source})
    output = tmp_path/"assets/generated/instrument-resources"
    with pytest.raises(ValueError, match="disjoint"):
        generator.validate_paths(output, output/"cache", root=tmp_path, repository=True, sources={"module.py": source})
    checked, _ = generator.validate_paths(output, cache, root=tmp_path, repository=True, sources={"module.py": source})
    assert checked == output
    with pytest.raises(ValueError, match="outside"):
        generator.validate_paths(output, tmp_path/"src/cache", root=tmp_path, repository=True, sources={"module.py": source})


def test_partial_generation_cleanup_preserves_unrelated_files(tmp_path):
    generator = generator_module()
    stale = tmp_path/"thermal-X"
    stale.mkdir()
    for name in generator.TASK_FILES:
        (stale/name).write_text("old")
    (stale/"user-notes.txt").write_text("preserve")
    generator.clean_stale_tasks(tmp_path, ["damping-Z"])
    assert sorted(path.name for path in stale.iterdir()) == ["user-notes.txt"]


def test_synthesis_preserves_acceptance_output_boundaries():
    circuit = QuantumCircuit(3, 3)
    circuit.reset(range(3))
    for wire, angle in enumerate((-.9545334665779891, -1.0046981887284008, .8424988502395834)):
        circuit.h(wire)
        circuit.rz(angle, wire)
    circuit.cx(0, 2)
    circuit.ry(-2.0827027446216535, 2)
    circuit.cx(2, 1)
    circuit.reset(2)
    circuit.rx(-1.0530785518149375, 0)
    circuit.ry(.4317590043242818, 2)
    circuit.cx(2, 0)
    circuit.measure(1, 1)
    circuit.s(0)
    circuit.measure(2, 2)
    circuit.h(0)
    circuit.measure(0, 0)
    ideal = simulate_trial(circuit, 0, {2: 1})
    compiled = synthesize_instrument(circuit)
    actual = simulate_trial(compiled.circuit, 0, {2: 1})
    routed = simulate_trial(compiled.routed, 0, {2: 1})
    assert routed["mean"] == pytest.approx(ideal["mean"], abs=1e-12)
    eta = compiled.report["guarded_numerical_operator_error_upper_estimate"]
    assert actual["acceptance_probability"] >= (math.sqrt(ideal["acceptance_probability"])-eta)**2
    assert abs(actual["mean"]-ideal["mean"]) <= 4*eta/math.sqrt(ideal["acceptance_probability"])
    measured = lambda value: [value.find_bit(item.clbits[0]).index for item in value.data if item.operation.name == "measure"]
    assert measured(compiled.circuit) == measured(circuit)


def test_reset_protocol_discards_entangled_input_without_postselection():
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    rho = DensityMatrix.from_instruction(bell).evolve(reset_kraus(), [1])
    np.testing.assert_allclose(rho.data, np.kron(np.diag([1, 0]), np.eye(2)/2), atol=1e-14)
    assert np.trace(rho.data) == pytest.approx(1)


@pytest.mark.parametrize("inverse", [False, True])
def test_actual_magic_state_injection_channel(inverse):
    c = QuantumCircuit(2)
    c.h(0)
    c.cx(0, 1)
    rho = DensityMatrix.from_instruction(c)
    angle = (-1 if inverse else 1)*np.pi/4
    expected = rho.evolve(Operator(np.diag([1, np.exp(1j*angle)])), [0])
    actual = rho.evolve(t_injection_kraus(inverse), [0])
    np.testing.assert_allclose(actual.data, expected.data, atol=1e-14)


def test_simple_readout_decoder_matches_exhaustive_independent_errors():
    p, total = .001, 0.0
    for errors in itertools.product((0, 1), repeat=9):
        noisy_paths = sum(sum(errors[3*j:3*j+3]) % 2 for j in range(3))
        if noisy_paths >= 2:
            total += p**sum(errors)*(1-p)**(9-sum(errors))
    assert majority_path_error(3, p) == pytest.approx(total, rel=1e-12)


@pytest.mark.parametrize("factory_ns,cap", [(0, 0), (500000, 6)])
def test_macro_latency_charges_every_reserved_controller_decision(factory_ns, cap):
    p = InstrumentProfile()
    for name in ["initial_reset", "reset", "measure", "cx", "h", "s", "t_q0", "tdg_q1"]:
        cost = _costs(p, 15, name, factory_ns, cap)
        assert cost["time_ns"] >= cost["decisions"]*p.reaction_time_ns
    reset = _costs(p, 15, "reset", factory_ns, cap)
    assert reset["readouts"] == 1 and reset["preparations"] == 2 and reset["decisions"] == 5


@pytest.mark.parametrize("kind", ["damping", "lindblad", "thermal"])
def test_synthesis_preserves_complete_instrument_and_checked_output(kind):
    if kind == "damping":
        source = build_amplitude_damping(observable="Z", reset=True)
        output, acceptance = 0, {}
    elif kind == "lindblad":
        source = build_lindblad_circuit(steps=2, observable="Z")
        output, acceptance = 0, {}
    else:
        source = build_thermal_trial(beta=.4, degree=2, measurement="Y")
        m = source.metadata["n_index"]
        output, acceptance = m, dict.fromkeys(range(m), 0)
    compiled = synthesize_instrument(source, .001)
    assert compiled.circuit.count_ops()["reset"] == source.count_ops()["reset"]
    assert compiled.circuit.count_ops()["measure"] == source.count_ops()["measure"]
    assert not set(compiled.circuit.count_ops()) & {"rx", "ry", "rz"}
    ideal = simulate_trial(source, output, acceptance)
    routed = simulate_trial(compiled.routed, output, acceptance)
    actual = simulate_trial(compiled.circuit, output, acceptance)
    assert routed["acceptance_probability"] == pytest.approx(ideal["acceptance_probability"], abs=1e-12)
    assert routed["mean"] == pytest.approx(ideal["mean"], abs=1e-12)
    eta = compiled.report["guarded_numerical_operator_error_upper_estimate"]
    assert eta <= .001
    assert abs(actual["mean"]-ideal["mean"]) <= 4*eta/math.sqrt(ideal["acceptance_probability"])+1e-12
    assert actual["acceptance_probability"] >= (math.sqrt(ideal["acceptance_probability"])-eta)**2-1e-12


def test_actual_native_resource_ledger_charges_all_events_and_factory_retries(tmp_path):
    source = build_amplitude_damping(observable="Z", reset=True)
    compiled = synthesize_instrument(source, .001)
    report = estimate_instrument(compiled, 100, profile=InstrumentProfile(maximum_code_distance=21),
                                 cache_dir=tmp_path/"factory")
    assert report["status"] == "computed_conditional_instrument_surface_code_model"
    assert report["source_event_counts"]["reset"] == 2
    for point in report["pareto_points"]:
        assert point["full_physical_failure_model"] <= report["physical_failure_budget"]
        assert point["full_physical_failure_model"] == pytest.approx(sum(point["failure_components_all_attempts"].values()), rel=1e-10)
        rows = point["macro_ledger_per_complete_attempt"]
        assert point["runtime_per_complete_attempt_ns"] == sum(row["count"]*row["time_ns"] for row in rows)
        assert point["worst_case_runtime_all_attempts_ns"] == point["runtime_per_complete_attempt_ns"]*100
        initial = next(row for row in rows if row["operation"] == "initial_reset")
        assert initial["count"] == 2+report["workspace_initializations_per_attempt"]
        supply = point["factory_supply"]
        assert supply["full_pipeline_retry_cap_per_state"] > 1
        assert supply["availability_failure_union_bound"] <= report["factory_availability_budget"]
        assert supply["worst_case_pipeline_attempts"] == supply["states_all_complete_attempts"]*supply["full_pipeline_retry_cap_per_state"]


def test_no_magic_state_source_still_charges_initialization_and_readout(tmp_path):
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.measure(0, 0)
    report = estimate_instrument(synthesize_instrument(c), 1, profile=InstrumentProfile(maximum_code_distance=9),
                                 cache_dir=tmp_path/"factory")
    point = report["pareto_points"][0]
    assert point["factory_supply"]["states_all_complete_attempts"] == 0
    assert point["factory_physical_qubits"] == 0
    assert point["peak_physical_qubits"] == point["data_workspace_physical_qubits"]
    assert point["runtime_per_complete_attempt_ns"] > 0
    assert point["failure_components_all_attempts"]["logical_initialization_scenario"] > 0


def test_compiled_instrument_checker_has_a_separate_expansion_limit():
    c = QuantumCircuit(1, 1)
    c.reset(0)
    c.h(0)
    for _ in range(10_001):
        c.t(0)
    c.measure(0, 0)
    checked = simulate_trial(c, 0)
    assert checked["mean"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match="operation bound"):
        synthesize_instrument(c)


def test_unsafe_or_unaudited_scenarios_are_rejected():
    with pytest.raises(ValueError):
        InstrumentProfile(factory_pipeline_failure_bound=.001)
    with pytest.raises(ValueError):
        InstrumentProfile(memory_prefactor=-.1)
    with pytest.raises(ValueError):
        majority_path_error(4, .001)
    c = QuantumCircuit(1, 1)
    c.h(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="initialize"):
        synthesize_instrument(c)
