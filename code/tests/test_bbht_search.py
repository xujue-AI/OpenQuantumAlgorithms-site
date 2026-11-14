"""Count-agnostic schedule, trace, and cap semantics for bounded SAT search."""
import copy

import pytest

from openquantum import bbht_search as search
from openquantum import grover_sat as sat

EXAMPLE=sat.CNF(4,((1,2),(-1,3),(-2,-3),(4,)))
UNSAT=sat.CNF(2,((1,),(-1,)))


def deterministic_trace(value):
    result=copy.deepcopy(value)
    for key in ("elapsed_wall_s","classical_verification_cpu_ns"):
        result.pop(key,None)
    for attempt in result["attempts"]:
        for key in list(attempt):
            if key.endswith("_cpu_ns"):
                attempt.pop(key)
    return result


def test_seeded_search_never_uses_an_exact_marked_count(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Search tried to use a bounded classical count")
    monkeypatch.setattr(sat,"exhaustive_solutions",forbidden)
    monkeypatch.setattr(sat,"amplification_probability",forbidden)
    config=search.BBHTConfig(seed=7,max_attempts=16,max_oracle_calls=32)
    first=search.run_bbht(EXAMPLE,config)
    second=search.run_bbht(EXAMPLE,config)
    assert deterministic_trace(first)==deterministic_trace(second)
    assert not first["marked_count_used_by_search"]
    assert first["attempts"][0]["iterations"]==0
    assert first["attempts"][0]["data_measurements"]==4
    assert first["classical_verification_calls"]==first["executed_attempts"]
    if first["status"]=="found":
        assert EXAMPLE.satisfies(first["verified_assignment"])
        assert first["attempts"][-1]["verified"]


def test_unsat_reference_does_not_change_inconclusive_search_status():
    result=search.run_bbht(UNSAT,search.BBHTConfig(seed=19,max_attempts=8))
    assert result["status"]=="inconclusive"
    assert result["termination_reason"]=="attempt_cap"
    assert result["executed_attempts"]==8
    assert result["verified_assignment"] is None
    assert result["unsat_certificate"] is None and result["search_failure_confidence"] is None
    assert sat.exhaustive_solutions(UNSAT)==[]
    assert result["actual_oracle_calls"]==sum(a["iterations"] for a in result["attempts"])
    assert all(a["clean_scratch_probability"]==pytest.approx(1,abs=1e-10) for a in result["attempts"])


def test_windows_and_growth_follow_only_executed_failed_attempts():
    config=search.BBHTConfig(seed=29,max_attempts=8,max_oracle_calls=32)
    result=search.run_bbht(UNSAT,config)
    m=1.
    for attempt in result["attempts"]:
        assert attempt["window_real"]==pytest.approx(m)
        assert 0<=attempt["iterations"]<attempt["window_real"]
        assert attempt["iterations"]<attempt["window_integer_cardinality"]
        m=min(config.growth*m,2.)


def test_query_cap_stops_before_execution_and_does_not_clip_iterations():
    result=search.run_bbht(UNSAT,search.BBHTConfig(seed=7,max_attempts=16,max_oracle_calls=0))
    assert result["status"]=="inconclusive"
    assert result["termination_reason"]=="oracle_call_cap_before_proposed_attempt"
    assert result["actual_oracle_calls"]==0
    assert all(a["iterations"]==0 for a in result["attempts"])
    proposal=result["unexecuted_budget_proposal"]
    assert proposal["proposed_iterations"]>proposal["remaining_oracle_calls"]


def test_soft_time_cap_is_inconclusive_before_a_trial(monkeypatch):
    times=iter([0.,1.,2.])
    monkeypatch.setattr(search.time,"perf_counter",lambda:next(times))
    result=search.run_bbht(UNSAT,search.BBHTConfig(max_wall_seconds=.1))
    assert result["status"]=="inconclusive"
    assert result["termination_reason"]=="soft_wall_cap_between_attempts"
    assert result["executed_attempts"]==0 and result["actual_oracle_calls"]==0
    assert result["unsat_certificate"] is None


def test_simulation_width_cap_does_not_block_raw_circuit_generation():
    cnf=sat.CNF(10,((1,),(-1,),(2,),(3,),(4,)))
    assert sat.build_sat_grover_circuit(cnf,1).num_qubits==22
    with pytest.raises(ValueError,match="simulation bound"):
        search.run_bbht(cnf,search.BBHTConfig(max_simulation_qubits=14))


@pytest.mark.parametrize("kwargs",[
    {"growth":1.},{"growth":4/3},{"growth":float("nan")},{"growth":"1.2"},
    {"seed":-1},{"max_attempts":0},{"max_attempts":129},{"max_oracle_calls":-1},
    {"max_simulation_qubits":17},{"max_wall_seconds":0},{"max_wall_seconds":float("inf")},
])
def test_invalid_schedule_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        search.BBHTConfig(**kwargs)


def test_materialization_preflight_stops_before_constructing_trial(monkeypatch):
    def capped(*args,**kwargs):
        raise ValueError("test operation preflight")
    def forbidden(*args,**kwargs):
        raise AssertionError("Capped proposal was materialized")
    monkeypatch.setattr(search,"preflight_sat_circuit",capped)
    monkeypatch.setattr(search,"build_sat_grover_circuit",forbidden)
    result=search.run_bbht(EXAMPLE)
    assert result["termination_reason"]=="materialization_cap_before_proposed_attempt"
    assert result["executed_attempts"]==0 and result["actual_oracle_calls"]==0
