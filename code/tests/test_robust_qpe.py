# SPDX-License-Identifier: MIT
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from openquantum import robust_qpe as rq
from openquantum.qft_qpe import dynamic_distribution

ROOT = Path(__file__).resolve().parents[1]


def decimal_atan(x):
    # Independent half-angle reduction with Decimal sqrt, not Machin/rational
    # octant reduction. One hundred terms enclose our much wider rational arc.
    y = x / (1 + (1+x*x).sqrt())
    total, term = Decimal(0), y
    for j in range(100):
        total += term / (2*j+1)
        term *= -y*y
    return 2*total


@pytest.mark.parametrize("c,s", [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1),(2,5),(5,2),(-2,5),(-5,2),(-2,-5),(-5,-2),(2,-5),(5,-2)])
def test_rational_angle_encloses_independent_decimal_reference(c,s):
    low, high = rq.angle_interval(Fraction(c,7), Fraction(s,7))
    with localcontext() as context:
        context.prec = 120
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647")
        if s == 0:
            angle = Decimal(0) if c > 0 else Decimal('.5')
        elif c == 0:
            angle = Decimal('.25') if s > 0 else Decimal('.75')
        else:
            magnitude = decimal_atan(Decimal(abs(s))/abs(c))/(2*pi)
            angle = magnitude if c>0 and s>0 else (Decimal('.5')-magnitude if c<0 and s>0 else (Decimal('.5')+magnitude if c<0 else 1-magnitude))
        dl = Decimal(low.numerator)/Decimal(low.denominator)
        dh = Decimal(high.numerator)/Decimal(high.denominator)
        assert dl <= angle <= dh or abs(dh-dl) == 0 and abs(angle-dl)<Decimal('1e-110')
    assert high-low <= rq.MAX_ANGLE_ENCLOSURE_WIDTH


def test_elementary_pi_and_failure_ceiling_are_conservative():
    low, high = rq.pi_interval()
    with localcontext() as context:
        context.prec = 350
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647")
        assert Decimal(low.numerator)/Decimal(low.denominator) < pi < Decimal(high.numerator)/Decimal(high.denominator)
        for beta in (Fraction(0), Fraction(1,50), Fraction(3,50)):
            m = rq.shots_for_failure(6, Fraction(1,800), beta)
            bound = rq.failure_bound(6,m,beta)
            assert bound <= Fraction(1,800) < rq.failure_bound(6,m-1,beta)
            t = rq.sampling_tolerance(beta)
            real = Decimal(24)*(-2*Decimal(m)*(Decimal(t.numerator)/Decimal(t.denominator))**2).exp()
            assert Decimal(bound.numerator)/Decimal(bound.denominator) >= real
    assert rq.shots_for_failure(6,Fraction(1,800),Fraction(1,50)) > rq.shots_for_failure(6,Fraction(1,800),Fraction(0))


@pytest.mark.parametrize("n,phase,k", [(1,.3,1),(1,.995,32),(2,.17,4),(3,.07,2)])
@pytest.mark.parametrize("basis", ["cos","sin"])
def test_actual_preparation_access_full_matrix_and_readout(n,phase,k,basis):
    result = rq.check_experiment(n,phase,k,basis)
    assert result["full_operator_max_error"] < 2e-11
    body = rq.build_experiment(n,phase,k,basis,measure=False,initialize=False)
    np.testing.assert_allclose(Operator(body).data,rq.dense_experiment_reference(n,phase,k,basis),atol=2e-11,rtol=0)
    measured = rq.build_experiment(n,phase,k,basis)
    counts = dict(measured.count_ops())
    assert counts["reset"] == n+1 and counts["x"]==n and counts["measure"]==1
    assert counts["p"]==3*n*k and counts["cx"]==2*n*k
    if n == 1 and k == 1:
        assert abs(dynamic_distribution(rq.lower_experiment(measured))[0]-result["zero_probability"]) < 1e-12
    p = result["zero_probability"]
    q = rq.readout_zero_probability(p,Fraction(3,200),Fraction(1,100))
    law = np.array([[.985,.01],[.015,.99]]) @ np.array([p,1-p])
    assert abs(q-law[0])<1e-15 and abs(q-p)<=.015


def counts_for_fringe_centers(phase,levels,offset=0,shots=4096):
    return [rq.FringeCounts(1<<j,b,int(round(shots*(1+(math.cos(2*math.pi*((1<<j)*phase+offset)) if b=="cos" else math.sin(2*math.pi*((1<<j)*phase+offset))))/2)),shots)
            for j in range(levels) for b in ("cos","sin")]


@pytest.mark.parametrize("phase", [0,.001,.125,.3,.5,.75,.995,math.nextafter(1,0)])
def test_exact_lifts_wrap_and_good_event_with_adversarial_centers(phase):
    for offset in (-.06,0,.06):
        result=rq.recover_phase(counts_for_fringe_centers(phase,6,offset),Fraction(1,50))
        assert result["status"]=="recovered"
        assert 0<=result["phase_estimate"]<1
        assert rq.circular_distance(result["phase_estimate"],phase) <= 1/384
        exact=Fraction(result["phase_estimate_exact"])
        assert float(exact)==result["phase_estimate"]
        assert result["confidence_arc_exact"]


def test_zero_signal_and_half_spacing_tie_retain_every_count():
    zero=[rq.FringeCounts(1,b,1,2) for b in ("cos","sin")]
    result=rq.recover_phase(zero,Fraction(0))
    assert result["status"]=="inconclusive_zero_signal" and len(result["counts"])==2
    assert result["phase_estimate"] is None
    assert rq.exact_lift_interval(Fraction(0),2,(Fraction(1,2),Fraction(1,2))) is None
    tie=[rq.FringeCounts(1,"cos",2,2),rq.FringeCounts(1,"sin",1,2),rq.FringeCounts(2,"cos",0,2),rq.FringeCounts(2,"sin",1,2)]
    result=rq.recover_phase(tie,Fraction(0))
    assert result["status"]=="inconclusive_lift_boundary" and len(result["counts"])==4
    assert result["phase_estimate"] is None


def test_malformed_and_unbounded_records_rejected():
    valid=counts_for_fringe_centers(.3,2)
    for records in (valid[:1],valid+valid[:1],valid[:-1]+valid[:1],[rq.FringeCounts(1,"cos",3,2),rq.FringeCounts(1,"sin",1,2)],itertools.repeat(valid[0])):
        with pytest.raises(ValueError):rq.recover_phase(records,Fraction(0))
    with pytest.raises(ValueError):rq.angle_interval(Fraction(0),Fraction(0))


@pytest.mark.parametrize("beta", [-.01,.125,.2,float('nan'),True,Fraction(1,1<<300)])
def test_bias_model_rejects_unsupported_requests(beta):
    with pytest.raises(ValueError):rq.sampling_tolerance(beta)


def test_separate_source_simulation_and_confidence_caps():
    large=rq.build_experiment(8,.3,32,"sin")
    assert large.num_qubits==9
    with pytest.raises(ValueError):rq.check_experiment(8,.3,32,"sin")
    with pytest.raises(ValueError):rq.build_experiment(32,.3,32768,"sin")
    with pytest.raises(ValueError):rq.shots_for_failure(16,Fraction(1,100),Fraction(124999,1000000))
    with pytest.raises(ValueError):rq.protocol_ledger(1,16,50000,Fraction(0))


def test_ledger_charges_complete_work_against_actual_measured_programs():
    n,L,m=2,4,100
    ledger=rq.protocol_ledger(n,L,m,Fraction(1,50))
    gates={}
    for circuit in rq.build_schedule(n,.3,L):
        for g,c in circuit.count_ops().items():gates[g]=gates.get(g,0)+m*c
    assert ledger["source_gate_counts_all_trials"]==gates
    assert ledger["complete_fresh_preparations"]==ledger["readout_bits"]==2*L*m
    assert ledger["shot_weighted_controlled_U_calls"]==2*m*(2**L-1)
    assert not ledger["physical_estimate_performed"] and not ledger["synthesis_performed"]


def test_executed_canonical_generator_and_notebook(tmp_path):
    spec=importlib.util.spec_from_file_location("robust_generator_test",ROOT/"scripts/generate_robust_qpe.py")
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    output=ROOT/"assets/generated/robust-qpe"
    # The first ordinary validation pass creates the new reviewed record.
    # Later passes read its immutable output, preserving producer/timing hashes.
    data=generator.generate(output) if not output.exists() else json.loads((output/"results.json").read_text())
    assert len(data["runs"])==8 and data["outcomes"]=={"within_radius":8,"miss":0,"inconclusive":0}
    assert data["environment"]["python"]=="3.12.14"
    assert Fraction(data["confidence"]["whole_predeclared_study_failure_upper_bound_exact"])<=Fraction(1,100)
    for run in data["runs"]:
        counts=[rq.FringeCounts(**record) for record in run["result"]["counts"]]
        assert rq.recover_phase(counts,Fraction(1,50))==run["result"]
    manifest=json.loads((output/"manifest.json").read_text())
    for name,sha in manifest["file_sha256"].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==sha
    assert len(list((output/"circuits").glob("*.qasm")))==49
    assert not data["large_source_only_export"]["check"]["performed"]
    # An existing installed/editable package must not replace portable access.
    environment=os.environ.copy(); environment["PYTHONPATH"]=str(ROOT/"src")
    binding=subprocess.run([sys.executable,"-c", "import openquantum.hadamard_qpe; import robust_qpe, hadamard_qpe; from pathlib import Path; assert robust_qpe.build_experiment is hadamard_qpe.build_experiment; print(Path(hadamard_qpe.__file__).resolve())"],
                           cwd=output,env=environment,capture_output=True,text=True,timeout=15,check=True)
    assert binding.stdout.strip()==str((output/"hadamard_qpe.py").resolve())
    with pytest.raises(ValueError):generator.generate(output)
    # Execute the normal companion notebook through the same locked kernel.
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    path=ROOT/"notebooks/robust_qpe.ipynb"
    book=nbformat.read(path,as_version=4)
    kernel=KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(book,timeout=60,km=kernel,resources={"metadata":{"path":str(ROOT)}})
    client.execute()
    nbformat.write(book,path)
