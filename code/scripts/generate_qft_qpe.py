#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit QFT/QPE/feedback circuits, checked demos and scaling.

Default examples are bounded ideal simulations. --no-sim provides exports
without claiming validation. --estimate computes optional conditional QDK
surface-code projections for five bounded static circuits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator
from openquantum.qft_qpe import (build_qft, qft_matrix, qft_omission_bound,
    build_qpe, build_iterative_qpe, phase_product_example, qpe_distribution,
    simulate_qpe, dynamic_distribution, lower_circuit, median_repetitions, qpe_ledger)


def write_json(path, data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def export(path, circuit):
    compiled=lower_circuit(circuit)
    text=qasm3.dumps(compiled)
    path.write_text(text)
    return {'sha256':hashlib.sha256(text.encode()).hexdigest(),'qubits':compiled.num_qubits,
            'gate_counts':dict(compiled.count_ops()),'dependency_depth':compiled.depth(),
            'global_phase':float(compiled.global_phase),
            'qasm_scope':'standalone measured algorithm or reset/feedback instrument; coherent control requires the unmeasured source unitary with global phase preserved'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits',type=int,default=4,help='QFT register width')
    parser.add_argument('--phase-bits',type=int,default=5)
    parser.add_argument('--target-qubits',type=int,default=1)
    parser.add_argument('--phase',type=float,default=.3)
    parser.add_argument('--no-sim',action='store_true')
    parser.add_argument('--no-scaling',action='store_true')
    parser.add_argument('--estimate',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'assets/generated/qft-qpe')
    args=parser.parse_args()
    output=args.output_dir; output.mkdir(parents=True,exist_ok=True)
    for name in ('surface-code-scaling.json','phase-probabilities.svg','qft-circuit.svg','qpe-circuit.svg','iterative-circuit.svg','scaling.csv','qft-scaling.csv'):
        (output/name).unlink(missing_ok=True)
    if (output/'.qdk-cache').is_dir():
        shutil.rmtree(output/'.qdk-cache')
    source=ROOT/'src/openquantum/qft_qpe.py'
    (output/'qft_qpe.py').write_bytes(source.read_bytes())
    (output/'LICENSE.txt').write_bytes((ROOT/'LICENSE').read_bytes())
    environment={'python':platform.python_version(),'device':'CPU','seed':731,
                 'packages':{p:importlib.metadata.version(p) for p in ('qiskit','numpy','matplotlib')}}
    try:
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        source_commit=None
    provenance={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'source_commit':source_commit,
                'source_status':'source hash identifies implementation; HEAD may precede this generated increment',
                'requirements_lock_sha256':hashlib.sha256((ROOT/'requirements-lock.txt').read_bytes()).hexdigest()}
    qft=build_qft(args.qubits)
    measured_qft=QuantumCircuit(args.qubits,args.qubits)
    # Basis x=1; input X is charged. Output distribution is uniform.
    measured_qft.x(0); measured_qft.compose(qft,inplace=True);measured_qft.measure(range(args.qubits),range(args.qubits))
    unitary,preparation=phase_product_example(args.target_qubits,args.phase)
    qpe=build_qpe(args.phase_bits,unitary,preparation)
    iterative=build_iterative_qpe(args.phase_bits,unitary,preparation)
    artifacts={
      'qft':export(output/'qft.qasm',measured_qft),
      'qpe':export(output/'qpe.qasm',build_qpe(args.phase_bits,unitary,preparation,measure=True)),
      'iterative_qpe':export(output/'iterative-qpe.qasm',iterative)}
    result={'parameters':vars(args)|{'output_dir':'generated artifact directory'},'environment':environment,'provenance':provenance,
            'artifacts':artifacts,'simulation':{'performed':False,'reason':'disabled or exceeds independently bounded reference cap'},
            'qpe_ledger':qpe_ledger(args.phase_bits,unitary,preparation),
            'iterative_ledger':qpe_ledger(args.phase_bits,unitary,preparation,iterative=True)}
    precision=1/2**args.phase_bits
    repetitions=median_repetitions(.005)
    result['precision_contract']={'phase_error':precision,'phase_metric':'circular distance for one run; scalar median requires phase in [error,1-error]',
      'known_phase_satisfies_scalar_median_branch_cut_assumption':precision<=args.phase<=1-precision,
      'single_success_lower_bound':8/math.pi**2,'sampling_failure_allowance':.005,
      'independent_odd_repetitions':repetitions,
      'majority_failure_upper_bound':sum(math.comb(repetitions,b)*(1-8/math.pi**2)**b*(8/math.pi**2)**(repetitions-b)
                                         for b in range((repetitions+1)//2,repetitions+1)),
      'classical_method':'sample median of decoded scalar phases; decoding is linear in number of output bits; sorting runtime unmeasured'}
    if not args.no_sim and args.qubits<=8 and qpe.num_qubits<=14 and iterative.num_qubits<=8 and args.phase_bits<=8:
        exact=qft_matrix(args.qubits)
        qft_error=float(np.max(abs(Operator(qft).data-exact)))
        actual=simulate_qpe(qpe,args.phase_bits);reference=qpe_distribution(args.phase,args.phase_bits)
        feedback=dynamic_distribution(iterative)
        np.testing.assert_allclose(actual,reference,atol=3e-13,rtol=0)
        np.testing.assert_allclose(feedback,reference,atol=3e-13,rtol=0)
        labels=np.arange(len(actual))/len(actual)
        success=float(actual[np.abs(labels-args.phase)<=precision+1e-14].sum())
        rng=np.random.default_rng(731)
        samples=rng.choice(len(actual),size=repetitions,p=actual)
        result['simulation']={'performed':True,'method':'static Statevector and actual dynamic measurement/reset/feedback branch enumeration',
          'qft_matrix_max_error':qft_error,'qpe_distribution_max_error':float(np.max(abs(actual-reference))),
          'iterative_distribution_max_error':float(np.max(abs(feedback-reference))),
          'phase_output_probabilities':actual.tolist(),'reference_probabilities':reference.tolist(),
          'numerical_single_success_within_scalar_target':success,
          'ideal_sampled_phase_outputs':samples.tolist(),'median_phase':float(np.median(samples)/len(actual)),
          'sampling_scope':'seeded ideal draws from validated circuit probabilities; not hardware executions'}
        fig,ax=plt.subplots(figsize=(8,3.4));ax.bar(labels,actual,width=.8/len(actual),color='#1c766c')
        ax.axvline(args.phase,color='#c4633d',linestyle='--',label='Exact eigenphase')
        ax.set(xlabel='Decoded phase y / 2^m',ylabel='Ideal probability',title='Textbook and feedback QPE agree');ax.legend();fig.tight_layout();fig.savefig(output/'phase-probabilities.svg');plt.close(fig)
    if args.qubits<=6:
        fig=qft.draw(output='mpl',fold=32);fig.savefig(output/'qft-circuit.svg',bbox_inches='tight');plt.close(fig)
    if args.phase_bits<=6 and args.target_qubits<=3:
        fig=qpe.draw(output='mpl',fold=36);fig.savefig(output/'qpe-circuit.svg',bbox_inches='tight');plt.close(fig)
        fig=iterative.draw(output='mpl',fold=36);fig.savefig(output/'iterative-circuit.svg',bbox_inches='tight');plt.close(fig)
    if not args.no_scaling:
        rows=[]
        for n in (1,2,4,8):
            for m in (3,4,5,6):
                u,p=phase_product_example(n,args.phase)
                ledger=qpe_ledger(m,u,p)
                rows.append({'target_qubits':n,'phase_bits':m,'phase_error':1/2**m,
                  'logical_qubits':n+m,'controlled_U_calls':2**m-1,'input_X_gates':n,
                  'primitive_gates':sum(ledger['decomposed_gate_counts'].values()),
                  'cx':ledger['decomposed_gate_counts'].get('cx',0),
                  'rotations':sum(ledger['decomposed_gate_counts'].get(g,0) for g in ('rx','ry','rz')),
                  'measurements':m,'dependency_depth':ledger['decomposed_dependency_depth'],
                  'independent_repetitions':repetitions,'total_controlled_U_calls':(2**m-1)*repetitions})
        with (output/'scaling.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
        qft_rows=[]
        for n in (2,4,8,16,32):
            cutoff=min(n-1,math.ceil(math.log2(math.pi*n/.01)))
            approximate=build_qft(n,max_distance=cutoff)
            exact=build_qft(n)
            qft_rows.append({'qubits':n,'exact_CP':exact.count_ops().get('cp',0),
              'approx_cutoff':cutoff,'approx_CP':approximate.count_ops().get('cp',0),
              'omission_spectral_norm_bound':qft_omission_bound(n,cutoff),
              'numeric_simulation_performed':False})
        with (output/'qft-scaling.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(qft_rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(qft_rows)
    if args.estimate:
        from openquantum.resource_estimation import CircuitResourceBudget,SurfaceCodeProfile,estimate_circuit_resources
        projected=[]
        with tempfile.TemporaryDirectory(prefix='openquantum-qpe-factory-') as temporary:
          for n,m in ((1,3),(1,4),(1,5),(2,4),(4,4)):
              u,p=phase_product_example(n,args.phase)
              report=estimate_circuit_resources(build_qpe(m,u,p,measure=True),
                  budget=CircuitResourceBudget(1e-5,.005,repetitions),
                  profile=SurfaceCodeProfile(),cache_dir=Path(temporary))
              projected.append({'target_qubits':n,'phase_bits':m,'phase_error':1/2**m,
                                'independent_repetitions':repetitions,
                                'precision_contract':result['precision_contract']|{'phase_error':1/2**m,
                                  'known_phase_satisfies_scalar_median_branch_cut_assumption':1/2**m<=args.phase<=1-1/2**m},
                                'report':report})
        write_json(output/'surface-code-scaling.json',{'status':'conditional_mean_fit_resource_projection',
            'phase':args.phase,'precision_and_repetition_contract':result['precision_contract'],
            'sampling_confidence_status':'ideal median contract only; no concrete synthesis or physical output-confidence certificate',
            'projections':projected})
    write_json(output/'results.json',result)
    # Normalize renderer whitespace without changing SVG path semantics.
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(json.dumps({'output':str(output),'simulation':result['simulation']['performed'],'repetitions':repetitions}))


if __name__=='__main__':main()
