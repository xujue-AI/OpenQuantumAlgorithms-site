"""Source-only port adapter tests: all native and full-candidate work is mocked."""
from dataclasses import dataclass, replace
import copy
import hashlib
from functools import wraps
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Reset
from qiskit.circuit.library import CXGate, HGate

from openquantum import instrument_resources as ir
from openquantum.simulation import lindblad_port_resources as pr


def tiny_trial(t=True):
    circuit = QuantumCircuit(8, 8)
    circuit.reset(range(8))
    circuit.h(0)
    if t:
        circuit.t(7)
        circuit.tdg(0)
    circuit.cx(0, 1)
    circuit.reset(7)
    circuit.measure(range(8), range(8))
    return circuit


@pytest.fixture
def fake_native(monkeypatch):
    state = {"factory_queries": 0, "surface_queries": 0, "traces": 0,
             "candidates": 1, "factory_error": 1e-18, "result_fault": None,
             "copies": 1, "missing_factory": False}
    native = ModuleType("qdk.qre")
    ids = ModuleType("qdk.qre.instruction_ids")
    models = ModuleType("qdk.qre.models")
    ids.T = "T"
    native.LOGICAL = "logical"
    native.ErrorComposition = SimpleNamespace(UnionBound="union")
    native.constant_function = lambda value: value
    native.linear_function = lambda value: value

    class Context:
        def __init__(self):
            self.instructions = {}
            self.factory = None

        def add_instruction(self, instruction_id, **kwargs):
            self.instructions[instruction_id] = kwargs
            return instruction_id

        def make_isa(self, *nodes):
            return self

    class GateBased:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def context(self):
            return Context()

    class Factory:
        def __init__(self, distance):
            self.distance = distance

        def expect_error_rate(self):
            return state["factory_error"]

        def expect_time(self):
            return 500_000+self.distance*1000

        def expect_space(self):
            return 100+self.distance

    class Supply:
        def __init__(self, distance, ctx):
            self.factory = Factory(distance)
            self.ctx = ctx

        def __getitem__(self, item):
            assert item == ids.T
            return self.factory

        def node_index(self, item):
            self.ctx.factory = self.factory
            return "supply"

    class SurfaceCode:
        @staticmethod
        def q(**kwargs):
            state["surface_queries"] += 1
            return kwargs

    class RoundBasedFactory:
        @staticmethod
        def q(code_query, cache_dir):
            state["factory_queries"] += 1
            distance = code_query["distance"]
            def enumerate_(ctx):
                for _ in range(state["candidates"]):
                    yield Supply(distance, ctx)
                if state.get("on_exhaustion"):
                    state["on_exhaustion"]()
            return SimpleNamespace(enumerate=enumerate_)

    class Trace:
        def __init__(self, patches):
            self.patches, self.blocks, self.resources, self.base_error = patches, [], {}, 0.

        def add_block(self, repetitions):
            def add_operation(instruction_id, wires):
                assert wires == list(range(self.patches))
                self.blocks.append((repetitions, instruction_id))
            return SimpleNamespace(add_operation=add_operation)

        def increment_resource_state(self, item, count):
            self.resources[item] = count

        def increment_base_error(self, error):
            self.base_error += error

        def to_json(self):
            return json.dumps([self.patches, self.blocks, self.resources, self.base_error], sort_keys=True)

        def estimate(self, isa, max_error, composition):
            state["traces"] += 1
            assert composition == "union"
            runtime = sum(c*isa.instructions[i]["time"] for c, i in self.blocks)
            error = self.base_error+sum(c*isa.instructions[i]["error_rate"] for c, i in self.blocks)
            if isa.factory:
                error += self.resources[ids.T]*isa.factory.expect_error_rate()
            assert error <= max_error
            patch_size = next(iter(isa.instructions.values()))["space"]
            qubits = self.patches*patch_size+(isa.factory.expect_space() if isa.factory else 0)
            if state["result_fault"] == "runtime":
                runtime += 1
            if state["result_fault"] == "error":
                error += 1e-10
            factories = {ids.T: SimpleNamespace(copies=state["copies"])} if isa.factory and not state["missing_factory"] else {}
            return SimpleNamespace(runtime=runtime, error=error, qubits=qubits, factories=factories)

    native.Trace = Trace
    models.GateBased, models.RoundBasedFactory, models.SurfaceCode = GateBased, RoundBasedFactory, SurfaceCode
    for name, module in (("qdk.qre", native), ("qdk.qre.instruction_ids", ids), ("qdk.qre.models", models)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(pr, "_rss_bytes", lambda: 1)
    monkeypatch.setattr(ir.importlib.metadata, "version", lambda name: "1.34.0")
    return state


def core(circuit, profile, tmp_path, *, limits=None, attempts=100):
    return pr._estimate_accounting(circuit, attempts, profile=profile,
                                  physical_failure_budget=.0025, factory_availability_budget=.0025,
                                  cache_dir=tmp_path, native_limits=limits or pr.native_limits(),
                                  source_event_counts=dict(circuit.count_ops()), actual_synthesis={},
                                  include_all_feasible_points=True)


@pytest.mark.parametrize("p", [.001, .0001])
@pytest.mark.parametrize("magic", [False, True])
def test_complete_mock_loop_matches_frozen_estimator_exactly(fake_native, tmp_path, p, magic):
    circuit = tiny_trial(magic)
    profile = ir.InstrumentProfile(physical_error_rate=p)
    bounds = ir.InstrumentLimits(8, 20_000, 1_000_000, 1_000_000, 256)
    old = ir.estimate_instrument(ir.SynthesizedInstrument(circuit, circuit, circuit, {"limits": ir.asdict(bounds)}),
                                 100, profile=profile, cache_dir=tmp_path, include_all_feasible_points=True)
    new = core(circuit, profile, tmp_path)
    for name in ("all_feasible_points", "pareto_points", "feasible_candidate_count", "profile",
                 "complete_algorithm_attempts", "physical_failure_budget", "factory_availability_budget",
                 "source_event_counts", "synthesized_event_counts", "workspace_initializations_per_attempt"):
        assert new[name] == old[name]
    assert new["native_execution"]["factory_queries"] == (24 if magic else 0)
    assert new["native_execution"]["surface_code_queries"] == (24 if magic else 0)


@pytest.mark.parametrize("limit_field", ["candidates_per_query", "candidates_per_profile"])
def test_candidate_overflow_discards_entire_profile(fake_native, tmp_path, limit_field):
    fake_native["candidates"] = 3
    limits = replace(pr.native_limits(), **{limit_field: 2})
    report = core(tiny_trial(), ir.InstrumentProfile(), tmp_path, limits=limits)
    assert report["status"] == "native_profile_cap_exceeded"
    assert report["pareto_points"] == report["all_feasible_points"] == []
    assert report["feasible_candidate_count"] == 0
    assert report["native_execution"]["candidate_overflow_sentinel_observed"]
    assert report["native_execution"]["factory_candidates_observed_within_cap"] == 2


def test_later_overflow_clears_previously_feasible_points(fake_native, tmp_path):
    result = core(tiny_trial(), ir.InstrumentProfile(), tmp_path,
                  limits=replace(pr.native_limits(), candidates_per_profile=12))
    assert fake_native["traces"] > 0
    assert result["status"] == "native_profile_cap_exceeded"
    assert result["all_feasible_points"] == result["pareto_points"] == []
    assert result["feasible_candidate_count"] == 0


def test_empty_iterator_still_checks_resources_after_native_work(fake_native, monkeypatch, tmp_path):
    fake_native["candidates"] = 0
    fake_native["on_exhaustion"] = lambda: monkeypatch.setattr(pr, "_rss_bytes", lambda: 2*1024**3+1)
    result = core(tiny_trial(), ir.InstrumentProfile(), tmp_path)
    assert result["status"] == "native_profile_cap_exceeded"
    assert fake_native["factory_queries"] == 1 and fake_native["traces"] == 0


def test_profile_json_overflow_discards_computed_points(fake_native, monkeypatch, tmp_path):
    def too_large(value):
        raise ValueError("Managed scientific JSON exceeds its 32 MiB cap")
    monkeypatch.setattr(pr, "_json_bytes", too_large)
    result = core(tiny_trial(), ir.InstrumentProfile(), tmp_path)
    assert fake_native["traces"] > 0
    assert result["status"] == "native_profile_cap_exceeded"
    assert "32 MiB" in result["reason"]
    assert result["all_feasible_points"] == result["pareto_points"] == []


@pytest.mark.parametrize("field,counter", [("factory_queries", "factory_queries"), ("surface_code_queries", "surface_queries")])
def test_query_cap_counts_only_dispatched_calls(fake_native, tmp_path, field, counter):
    report = core(tiny_trial(), ir.InstrumentProfile(), tmp_path, limits=replace(pr.native_limits(), **{field: 1}))
    assert report["status"] == "native_profile_cap_exceeded"
    assert report["native_execution"][field] == fake_native[counter] == 1
    assert report["pareto_points"] == []


@pytest.mark.parametrize("fault", ["runtime", "error", "copies", "missing"])
def test_native_ledger_and_single_factory_contract_rejects(fake_native, tmp_path, fault):
    if fault == "copies":
        fake_native["copies"] = 2
    elif fault == "missing":
        fake_native["missing_factory"] = True
    else:
        fake_native["result_fault"] = fault
    with pytest.raises(RuntimeError, match="ledger|factory"):
        core(tiny_trial(), ir.InstrumentProfile(), tmp_path)


def test_empty_native_domain_is_explicit(fake_native, tmp_path):
    fake_native["factory_error"] = .01
    result = core(tiny_trial(), ir.InstrumentProfile(), tmp_path)
    assert result["status"] == "infeasible_declared_native_domain"
    assert result["pareto_points"] == []
    assert fake_native["traces"] == 0


@pytest.mark.parametrize("resource", ["rss", "wall"])
def test_profile_caps_reject_before_dispatch(fake_native, monkeypatch, tmp_path, resource):
    if resource == "rss":
        monkeypatch.setattr(pr, "_rss_bytes", lambda: 2*1024**3+1)
    else:
        clock = iter([0., 181.])
        monkeypatch.setattr(pr.time, "monotonic", lambda: next(clock, 181.))
    result = core(tiny_trial(), ir.InstrumentProfile(), tmp_path)
    assert result["status"] == "native_profile_cap_exceeded"
    assert fake_native["factory_queries"] == fake_native["surface_queries"] == fake_native["traces"] == 0


def test_limits_and_all_n4_entry_fields_are_explicit():
    with pytest.raises(TypeError):
        pr.PortResourceLimits()
    for name in pr.literal_limits().__dataclass_fields__:
        with pytest.raises(ValueError):
            replace(pr.literal_limits(), **{name: True})
    for name in pr.native_limits().__dataclass_fields__:
        with pytest.raises(ValueError):
            replace(pr.native_limits(), **{name: getattr(pr.native_limits(), name)+1})
    assert ir.InstrumentLimits().max_qubits == 5
    assert ir.InstrumentLimits().synthesized_operations == 100_000


def test_flat_canonical_neighbors_initialization_and_readout_guards():
    pr._candidate_domain(tiny_trial(), pr.literal_limits())
    for mutate in (lambda c: c.cx(0, 7), lambda c: c.data.__delitem__(0), lambda c: c.h(0)):
        circuit = tiny_trial()
        mutate(circuit)
        with pytest.raises(ValueError):
            pr._candidate_domain(circuit, pr.literal_limits())
    circuit = tiny_trial()
    hidden = Gate("t", 1, [])
    hidden.definition = QuantumCircuit(1)
    hidden.definition.x(0)
    circuit.data[9] = circuit.data[9].replace(operation=hidden)
    with pytest.raises(ValueError, match="canonical"):
        pr._candidate_domain(circuit, pr.literal_limits())
    circuit = tiny_trial()
    circuit.data[11] = circuit.data[11].replace(operation=CXGate(ctrl_state=0))
    with pytest.raises(ValueError, match="canonical|Open-control"):
        pr._candidate_domain(circuit, pr.literal_limits())
    circuit = tiny_trial()
    mutable = HGate().to_mutable()
    mutable.to_matrix = lambda: np.array([[0, 1], [1, 0]])
    circuit.data[8] = circuit.data[8].replace(operation=mutable)
    with pytest.raises(ValueError, match="executable"):
        pr._candidate_domain(circuit, pr.literal_limits())


def test_current_loaded_code_detects_foreign_replacement_and_method_change(monkeypatch):
    assert pr._require_loaded_source(ir) == pr.FROZEN_IR_SHA256
    with monkeypatch.context() as m:
        m.setattr(ir, "_costs", lambda *args: {})
        with pytest.raises(ValueError, match="Loaded function"):
            pr._require_loaded_source(ir)
    with monkeypatch.context() as m:
        original = ir._costs
        @wraps(original)
        def changed(*args, **kwargs):
            return {}
        m.setattr(ir, "_costs", changed)
        with pytest.raises(ValueError, match="Loaded function"):
            pr._require_loaded_source(ir)
    with monkeypatch.context() as m:
        m.setattr(ir.InstrumentProfile, "patch_failure_per_cycle", lambda *args: 0.)
        with pytest.raises(ValueError, match="Loaded method"):
            pr._require_loaded_source(ir)


def test_adapter_self_source_is_checked_and_descriptor_reads_reject_symlinks(monkeypatch, tmp_path):
    pr._require_loaded_source(pr)
    with monkeypatch.context() as m:
        m.setattr(pr, "_controller", lambda *args: {})
        with pytest.raises(ValueError, match="Loaded function"):
            pr._require_loaded_source(pr)
    source = tmp_path/"source.py"
    source.write_text("x = 1\n")
    link = tmp_path/"link.py"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        pr._bounded_source_bytes(link)


def test_controller_identity_retains_nonzero_wire_zero_magic_cost():
    profile = ir.InstrumentProfile()
    for wire in range(8):
        assert ir._costs(profile, 3, f"t_q{wire}", 0, 7)["decisions"] == 47+33*wire+4*7
        assert ir._costs(profile, 3, f"tdg_q{wire}", 0, 7)["decisions"] == 48+33*wire+4*7
    cost = ir._costs(profile, 3, "t_q0", 500_000, 7)
    assert cost["time_ns"] > 7*500_000
    assert cost["readouts"] == 13 and cost["preparations"] == 25


def test_profile_literals_do_not_follow_mutated_generated_constructor_defaults(monkeypatch):
    fields = list(ir.InstrumentProfile.__dataclass_fields__)
    defaults = list(ir.InstrumentProfile.__init__.__defaults__)
    defaults[fields.index("memory_prefactor")] = .2
    monkeypatch.setattr(ir.InstrumentProfile.__init__, "__defaults__", tuple(defaults))
    profile = ir.InstrumentProfile()
    assert profile.memory_prefactor == .2
    with pytest.raises(ValueError, match="unchanged"):
        pr._fixed_profile(profile)


def test_fingerprint_rechecks_mutable_gate_methods_after_initial_identity():
    circuit = tiny_trial()
    mutable = HGate().to_mutable()
    circuit.data[8] = circuit.data[8].replace(operation=mutable)
    pr._circuit_sha(circuit, pr.literal_limits())
    mutable.to_matrix = lambda: np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="executable"):
        pr._circuit_sha(circuit, pr.literal_limits())


def density_record():
    rho = np.zeros((16, 16), complex)
    rho[0, 0] = 1
    return {"reduced_before_output": {"real": rho.real.tolist(), "imag": rho.imag.tolist()},
            "joint_probabilities": np.diag(rho).real.tolist(), "unnormalized_trace_before_output": 1.,
            "acceptance_probability": 1., "normalization_divisor": 1., "normalization_is_postselection": False,
            "normalization_probability_is_postselection": False, "outcome_clbits": [0, 1, 2, 3],
            "outcome_wires": [6, 1, 2, 3], "score_values": list(pr.fp.score_values()), "mean": 1.}


@pytest.fixture
def proof_fixture(monkeypatch):
    module = ModuleType("openquantum.simulation.lindblad_port_rewrite")
    @dataclass
    class Candidate:
        circuit: object
        report: dict
    @dataclass
    class Recipe:
        changed: bool = False
        def as_record(self):
            return {"recipe_sha256": "r"*64, "changed": self.changed}
    @dataclass(frozen=True)
    class Limits:
        bound: int = 1
    class Recovery:
        recovery_sha256 = "w"*64
        baseline_global_phase = 0.
        source_artifacts = {"results.json": "frozen"}
        def as_record(self):
            return {"recovery_sha256": self.recovery_sha256}
    module.PortRewriteCandidate, module.FrozenPortRecipe = Candidate, Recipe
    module.require_verified_candidate = lambda candidate, *args, **kwargs: copy.deepcopy(candidate.report)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(pr, "derivation_record", lambda: {"frozen": True})
    record = density_record()
    baseline = {"source_instrument_check": copy.deepcopy(record), "clifford_t_instrument_check": copy.deepcopy(record),
                "error_contract": {"realization_Z0_bias_upper_proposal": 2e-5, "splitting_bias_upper": .01},
                "complete_workload": {}, "source_counts": {"reset": 1481, "h": 8445, "measure": 8},
                "actual_synthesis": {"routed_counts": {"cx": 24_510}},
                "independent_reference": {"exact_GKSL_system_density": record["reduced_before_output"], "exact_GKSL_Z0": 1.}}
    monkeypatch.setattr(pr, "_saved_baseline", lambda *args: copy.deepcopy(baseline))
    monkeypatch.setattr(pr.fp, "_annotate_check", lambda *args: None)
    calls = []
    monkeypatch.setattr(pr.di, "simulate_density_trial", lambda *args, **kwargs: calls.append("new-candidate") or copy.deepcopy(record))
    circuit = tiny_trial()
    circuit.global_phase = 0.
    monkeypatch.setattr(circuit, "count_ops", lambda: {"t": 642_284, "reset": 1481, "measure": 8})
    report = {"actual_operations": 1_800_000, "added_cx": 100_000, "global_phase_binary64_hex": 0.0.hex(),
              "final_measurement_wires": [6, 1, 2, 3, 7, 4, 0, 5]}
    candidate = Candidate(circuit, report)
    # Domain semantics are tested independently; proof tests use a tiny mock
    # standing in for the production n4 candidate, never materialize that trial.
    monkeypatch.setattr(pr, "_candidate_domain", lambda *args: pr.di.EightWireDensityLimits(8, 20_000, 2_000_000))
    return SimpleNamespace(candidate=candidate, recipe=Recipe(), recovered=Recovery(), word_limits=Limits(),
                           rewrite_limits=Limits(), baseline=baseline, record=record, calls=calls)


def verify(fixture, **kwargs):
    return pr.verify_candidate(fixture.candidate, fixture.recipe, fixture.recovered,
                               limits=pr.literal_limits(), word_limits=fixture.word_limits,
                               rewrite_limits=fixture.rewrite_limits, **kwargs)


def test_proof_requires_execution_gate_and_only_checks_new_candidate(proof_fixture):
    with pytest.raises(ValueError, match="explicit"):
        verify(proof_fixture)
    assert proof_fixture.calls == []
    verified = verify(proof_fixture, allow_full_density=True)
    pr._require_verified(verified)
    assert proof_fixture.calls == ["new-candidate"]
    assert verified.as_record()["historical_density_execution_count"] == 0
    assert verified.as_record()["candidate_density_execution_count"] == 1
    assert verified.as_record()["saved_baseline_comparisons"]["candidate_vs_saved_GKSL_joint_maximum_absolute"] == 0
    actual = verified.as_record()["complete_workload"]
    assert actual["primitive_counts_per_complete_trial"] == {"t": 642_284, "reset": 1481, "measure": 8}
    assert actual["primitive_counts_all_trials"]["t"] == 642_284*15_756
    assert verified.as_record()["saved_baseline_complete_workload"] == proof_fixture.baseline["complete_workload"]
    assert verified.as_record()["saved_source_counts"] == proof_fixture.baseline["source_counts"]


@pytest.mark.parametrize("mutation", ["circuit", "report", "recipe", "proof", "fingerprint", "copy", "json_whitespace", "json_type"])
def test_process_proof_rejects_mutation_and_forged_copies(proof_fixture, mutation):
    verified = verify(proof_fixture, allow_full_density=True)
    if mutation == "circuit":
        verified.candidate.circuit.x(0)
    elif mutation == "report":
        verified.candidate.report["actual_operations"] += 1
    elif mutation == "recipe":
        verified.recipe.changed = True
    elif mutation == "proof":
        object.__setattr__(verified, "canonical_verification_json", b'{}')
    elif mutation == "fingerprint":
        object.__setattr__(verified, "verification_fingerprint", "0"*64)
    elif mutation == "json_whitespace":
        object.__setattr__(verified, "canonical_verification_json", verified.canonical_verification_json+b" ")
    elif mutation == "json_type":
        object.__setattr__(verified, "canonical_verification_json", verified.canonical_verification_json.decode())
    else:
        verified = replace(verified)
    with pytest.raises((ValueError, KeyError), match=None):
        pr._require_verified(verified)


@pytest.mark.parametrize("field", ["source_operations", "reference_operations"])
def test_narrowed_saved_baseline_domain_rejects_before_density(proof_fixture, field):
    with pytest.raises(ValueError, match="reference"):
        pr.verify_candidate(proof_fixture.candidate, proof_fixture.recipe, proof_fixture.recovered,
                            limits=replace(pr.literal_limits(), **{field: 1}), word_limits=proof_fixture.word_limits,
                            rewrite_limits=proof_fixture.rewrite_limits, allow_full_density=True)
    assert proof_fixture.calls == []


def test_raw_ct_agreement_has_density_guard_separate_from_state_guard(proof_fixture):
    proof_fixture.baseline["clifford_t_instrument_check"]["unnormalized_trace_before_output"] = 1.-2e-10
    proof_fixture.baseline["clifford_t_instrument_check"]["acceptance_probability"] = 1.-2e-10
    proof_fixture.baseline["clifford_t_instrument_check"]["normalization_divisor"] = 1.-2e-10
    with pytest.raises(ValueError, match="agreement"):
        verify(proof_fixture, allow_full_density=True)


@pytest.mark.parametrize("field,value", [("actual_operations", 1_926_043), ("added_cx", 248_137),
                                       ("global_phase_binary64_hex", 1.0.hex()),
                                       ("final_measurement_wires", list(range(8)))])
def test_candidate_report_contract_rejects_before_density(proof_fixture, field, value):
    proof_fixture.candidate.report[field] = value
    with pytest.raises(ValueError, match="contract"):
        verify(proof_fixture, allow_full_density=True)
    assert proof_fixture.calls == []


@pytest.mark.parametrize("fault", ["trace", "trace_type", "raw_normalization", "shape", "joint", "score", "postselection", "mapping"])
def test_full_density_evidence_rejects_invalid_raw_normalized_contract(fault):
    value = density_record()
    if fault == "trace":
        value["unnormalized_trace_before_output"] = 1.+2e-8
    elif fault == "trace_type":
        value["unnormalized_trace_before_output"] = True
    elif fault == "raw_normalization":
        value["normalization_divisor"] = .5
    elif fault == "shape":
        value["reduced_before_output"] = {"real": [[1.]], "imag": [[0.]]}
    elif fault == "joint":
        value["joint_probabilities"][0] = .5
    elif fault == "score":
        value["mean"] = .5
    elif fault == "postselection":
        value["normalization_is_postselection"] = True
    else:
        value["outcome_clbits"] = [0]
    with pytest.raises(ValueError):
        pr._check_density(value)


def test_controller_rejection_happens_before_native_import(proof_fixture, monkeypatch, tmp_path):
    verified = verify(proof_fixture, allow_full_density=True)
    monkeypatch.setattr(pr, "_controller", lambda *args: {"budget_exceeded": True})
    monkeypatch.setattr(pr, "_estimate_accounting", lambda *args, **kwargs: pytest.fail("native reached"))
    result = pr.estimate_verified_four(verified, profile=ir.InstrumentProfile(), cache_dir=tmp_path,
                                      limits=pr.literal_limits(), native_limits=pr.native_limits())
    assert result["status"] == "infeasible_controller_model"
    assert result["native_execution"]["factory_queries"] == 0


def test_frozen_profile_gate_and_bare_watchdog_boolean_rejects(proof_fixture, monkeypatch, tmp_path):
    verified = verify(proof_fixture, allow_full_density=True)
    for field, value in (("physical_error_rate", .0002), ("controller_failure_per_decision", 0.), ("maximum_code_distance", 47)):
        with pytest.raises(ValueError, match="unchanged"):
            pr.controller_eligibility(verified, replace(ir.InstrumentProfile(), **{field: value}))
    monkeypatch.setattr(pr, "_controller", lambda *args: {"budget_exceeded": False})
    monkeypatch.setattr(pr, "_estimate_accounting", lambda *args, **kwargs: pytest.fail("native reached"))
    with pytest.raises(ValueError, match="capability"):
        pr.estimate_verified_four(verified, profile=ir.InstrumentProfile(), cache_dir=tmp_path,
                                  limits=pr.literal_limits(), native_limits=pr.native_limits(),
                                  allow_native_search=True, watchdog_capability=True)
    with pytest.raises(ValueError, match="reviewed runner"):
        pr._issue_native_watchdog(activation_event=SimpleNamespace(is_set=lambda: False),
                                  limits=pr.native_limits(), runner_source_sha256="0"*64)
