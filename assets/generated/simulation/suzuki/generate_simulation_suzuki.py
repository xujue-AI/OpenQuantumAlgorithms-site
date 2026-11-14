#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate original higher-order Suzuki circuits and fixed-task numerical checks.

Default 36 cases: 1/2/3 spins, orders 2/4/6, repetitions 1/2/4/8, J=1, hX=0.7, hZ=0, t=1.
Use --no-sim with an explicit width for a larger independently bounded export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='OpenQuantumSuzuki'
from qiskit import qasm3
from qiskit.synthesis import SuzukiTrotter

from openquantum.simulation.suzuki import (
    SpinChain,SUPPORTED_ORDERS,MAX_GENERATOR_QUBITS,MAX_DENSE_QUBITS,
    MAX_DENSE_BASE_BLOCKS,MAX_DENSE_PRIMITIVE_OPERATIONS,MAX_BASE_BLOCKS,
    MAX_PRIMITIVE_OPERATIONS,compare_case,logical_ledger,normalized_leaves,
    rational,request,_dense_request,suzuki_circuit,validated_order,
)

MAX_CASES=64
MAX_OUTPUT_BYTES=64*1024*1024
SOURCE_FILES=('src/openquantum/simulation/suzuki.py','scripts/generate_simulation_suzuki.py','requirements-lock.txt','LICENSE')
QISKIT_SOURCE_SHA='546bc06261fb6143a2560c61ef8dfb1b6de274e236916e69876365c3420ae2c1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_payload(output,path,content):
    encoded=content.encode() if isinstance(content,str) else content
    # Reserve space for the atomic completion marker before any payload write.
    reserved=128*1024
    used=sum(item.stat().st_size for item in output.rglob('*') if item.is_file())
    previous=path.stat().st_size if path.exists() else 0
    if used-previous+len(encoded)>MAX_OUTPUT_BYTES-reserved:
        raise RuntimeError('payload would exceed64MiB reservation; no completed manifest written')
    path.write_bytes(encoded)


def save_svg(output,name,figure,**kwargs):
    buffer=io.BytesIO();figure.savefig(buffer,format='svg',**kwargs)
    normalized='\n'.join(line.rstrip() for line in buffer.getvalue().decode().splitlines())+'\n'
    write_payload(output,output/name,normalized)


def committed_source():
    top=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,capture_output=True,text=True,timeout=5)
    if top.returncode!=0 or not top.stdout.strip() or Path(top.stdout.strip()).resolve()!=ROOT.resolve():
        return None
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,timeout=5)
    if result.returncode!=0:
        return None
    head=result.stdout.strip()
    for name in SOURCE_FILES:
        record=subprocess.run(['git','show',f'{head}:{name}'],cwd=ROOT,capture_output=True,timeout=5)
        if record.returncode!=0 or sha(record.stdout)!=sha((ROOT/name).read_bytes()):
            raise ValueError('producer source differs from commit; commit source before generation')
    return head


def output_directory(path):
    path=Path(path).absolute();current=path
    while current!=current.parent:
        if current.is_symlink():
            raise ValueError('output must not use redirected paths')
        current=current.parent
    destination=path.resolve();generated=(ROOT/'assets/generated').resolve();own=generated/'simulation/suzuki'
    if destination==generated or generated in destination.parents:
        if destination!=own and own not in destination.parents:
            raise ValueError('Suzuki generation cannot overwrite another algorithm tree')
    for name in ('src','scripts','tests','notebooks','algorithms','references','.git'):
        protected=(ROOT/name).resolve()
        if destination==protected or protected in destination.parents:
            raise ValueError('output cannot overwrite source/private project data')
    if destination==ROOT.resolve() or destination.exists():
        raise ValueError('output must be a new dedicated directory; preserve prior runs')
    return destination


def coefficient_figure(output):
    fig,axes=plt.subplots(1,3,figsize=(10.5,2.9))
    for order,axis in zip(SUPPORTED_ORDERS,axes):
        values=[leaf.binary64 for leaf in normalized_leaves(order)]
        axis.bar(range(len(values)),values,color=['#346b8a' if v>0 else '#be6b35' for v in values])
        axis.axhline(0,color='#777',linewidth=.6)
        axis.set(title=f'Order {order}',xlabel='Chronological Strang block',ylabel='Normalized signed duration')
    fig.tight_layout();save_svg(output,'signed-substeps.svg',fig);plt.close(fig)


def convergence_figures(output,rows):
    widths=sorted({row['qubits'] for row in rows})
    for xkey,filename,xlabel in [('repetitions','error-vs-steps.svg','Macro-steps r'),('source_operations','error-vs-work.svg','Complete measured primitive operations')]:
        fig,axes=plt.subplots(1,len(widths),figsize=(4.0*len(widths),3.4),squeeze=False)
        for qubits,axis in zip(widths,axes[0]):
            for order in sorted({r['order'] for r in rows}):
                group=sorted((row for row in rows if row['qubits']==qubits and row['order']==order),key=lambda row:row[xkey])
                if group:
                    axis.loglog([row[xkey] for row in group],[max(row['operator_error'],1e-17) for row in group],'o-',label=f'Order {order}')
            axis.set(title=f'{qubits} spins'+(' — commuting limit' if qubits==1 and all(row['z_field_exact']=='0' for row in rows if row['qubits']==1) else ''),xlabel=xlabel,ylabel='Numerical operator error')
            axis.grid(alpha=.18);axis.legend(fontsize=8)
        fig.tight_layout();save_svg(output,filename,fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits',type=int)
    parser.add_argument('--orders',type=int,nargs='+',default=list(SUPPORTED_ORDERS))
    parser.add_argument('--repetitions',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--time',default='1')
    parser.add_argument('--coupling',default='1')
    parser.add_argument('--x-field',default='7/10')
    parser.add_argument('--z-field',default='0')
    parser.add_argument('--initial-basis',type=int,default=0)
    parser.add_argument('--no-sim',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'assets/generated/simulation/suzuki')
    args=parser.parse_args();start=time.perf_counter()
    if len(set(args.orders))!=len(args.orders) or len(set(args.repetitions))!=len(args.repetitions):
        parser.error('orders/repetition schedules must not duplicate cases')
    for order in args.orders:validated_order(order)
    if args.no_sim and args.qubits is None:parser.error('--no-sim requires an explicit width')
    widths=[args.qubits] if args.qubits is not None else [1,2,3]
    if len(widths)*len(args.orders)*len(args.repetitions)>MAX_CASES:parser.error('canonical request exceeds 64 cases')
    chains=[SpinChain(n,args.coupling,args.x_field,args.z_field) for n in widths]
    args.time=str(rational(args.time,'time'))
    args.coupling=str(chains[0].coupling);args.x_field=str(chains[0].x_field);args.z_field=str(chains[0].z_field)
    for chain in chains:
        if not args.no_sim and chain.qubits>MAX_DENSE_QUBITS:parser.error('dense references stop at three spins; use --no-sim')
        for order in args.orders:
            for repetitions in args.repetitions:
                request(chain,args.time,repetitions,order,args.initial_basis)
                if not args.no_sim:
                    _dense_request(chain,args.time,repetitions,order)
    reference_path=Path(inspect.getfile(SuzukiTrotter))
    if importlib.metadata.version('qiskit')!='2.6.2' or sha(reference_path.read_bytes())!=QISKIT_SOURCE_SHA:
        raise ValueError('Qiskit Suzuki reference differs from the inspected pinned 2.6.2 source')
    frozen_producers={name:(ROOT/name).read_bytes() for name in SOURCE_FILES}
    frozen_hashes={name:sha(data) for name,data in frozen_producers.items()}
    source=committed_source()
    if any(sha((ROOT/name).read_bytes())!=digest for name,digest in frozen_hashes.items()):
        raise ValueError('producer source changed during preflight')
    output=output_directory(args.output_dir);output.mkdir(parents=True)
    case_rows=[];cases=[]
    for chain in chains:
        for order in args.orders:
            for repetitions in args.repetitions:
                key=f'n{chain.qubits}-s{order}-r{repetitions}';case=output/key;case.mkdir()
                construction=time.perf_counter()
                ledger=logical_ledger(chain,args.time,repetitions,order,initial_basis=args.initial_basis)
                measured=suzuki_circuit(chain,args.time,repetitions,order,initial_basis=args.initial_basis,measure=True)
                qasm=qasm3.dumps(measured);write_payload(output,case/'measured-trial.qasm',qasm)
                construction_seconds=time.perf_counter()-construction
                result=None;validation_seconds=0
                if not args.no_sim:
                    validation=time.perf_counter();result=compare_case(chain,args.time,repetitions,order,initial_basis=args.initial_basis)
                    validation_seconds=time.perf_counter()-validation
                record={'case_id':key,'logical_ledger':ledger,
                        'validation':result if result is not None else {'performed':False,'reason':'Generated-only larger circuit; dense/reference guard remains independent'},
                        'measured_qasm_sha256':sha(qasm.encode()),
                        'cpu_timings_seconds':{'source_construction_scalar_accounting_export':construction_seconds,'independent_and_qiskit_numerical_checks':validation_seconds},
                        'measurement_execution':{'physical_measurements':False,'finite_shot_samples':False,'scope':('Actual reset/preparation circuit prefix simulated; final expectation calculated from its ideal state. Budgeted repeated trials are not executed.' if not args.no_sim else 'Generated-only measured circuit specification; no state, operator, reset, or measurement simulation was executed.')}}
                write_payload(output,case/'record.json',json.dumps(record,indent=2,allow_nan=False)+'\n');cases.append(key)
                if result is not None:
                    counts=ledger['complete_measured_primitive_counts'];rounding=ledger['source_parameter_rounding']
                    case_rows.append({'case_id':key,'qubits':chain.qubits,'order':order,'repetitions':repetitions,'time_exact':args.time,'z_field_exact':str(chain.z_field),
                                      'source_operations':sum(counts.values()),'rotations':ledger['arbitrary_rotations'],'cx':counts.get('cx',0),'source_depth':ledger['source_dependency_depth'],
                                      'negative_actual_substeps':rounding['negative_actual_substeps'],'absolute_substep_time_exact':rounding['actual_absolute_substep_time_sum_exact'],
                                      'operator_error':result['operator_error_vs_exact_evolution'],'state_error':result['state_vector_error_vs_exact_evolution'],'observable_bias':result['observable_bias'],
                                      'exact_z0':result['exact_z0'],'circuit_z0':result['circuit_z0'],'source_parameter_bound':rounding['total_source_parameter_operator_bound_upper_float'],
                                      'qiskit_reference_operator_difference':result['source_to_qiskit_suzuki_operator_error']})
                if chain.qubits<=3 and order==4 and repetitions==1:
                    figure=measured.draw(output='mpl',fold=35,scale=.8)
                    save_svg(output,f'n{chain.qubits}-order4-circuit.svg',figure,bbox_inches='tight');plt.close(figure)
    coefficient_figure(output)
    if case_rows:
        handle=io.StringIO();writer=csv.DictWriter(handle,fieldnames=list(case_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(case_rows)
        write_payload(output,output/'summary.csv',handle.getvalue())
        convergence_figures(output,case_rows)
        selected=[row for row in case_rows if row['repetitions'] in (1,8)]
        lines=['| n | Order | r | Complete gates | Operator error | Z0 bias | Negative blocks |','|---:|---:|---:|---:|---:|---:|---:|']
        lines += [f"| {row['qubits']} | {row['order']} | {row['repetitions']} | {row['source_operations']} | {row['operator_error']:.4g} | {row['observable_bias']:.4g} | {row['negative_actual_substeps']} |" for row in selected]
        write_payload(output,output/'summary.md','\n'.join(lines)+'\n')
    for name in SOURCE_FILES[:2]:write_payload(output,output/Path(name).name,frozen_producers[name])
    write_payload(output,output/'LICENSE.txt',frozen_producers['LICENSE'])
    files={str(path.relative_to(output)):{'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())} for path in sorted(output.rglob('*')) if path.is_file()}
    manifest={'schema':'openquantum-suzuki-study-v1','status':'generated_and_numerically_checked' if not args.no_sim else 'generated_only_no_simulation',
              'source_commit':source,'source_commit_scope':'Exact checkout root only; null for detached downloads. All four producer bytes authenticate when Git is available.',
              'source_sha256':frozen_hashes,
              'python':sys.version.split()[0],'device':'CPU','seed':None,'randomness':'No stochastic algorithm or optimizer; deterministic circuit construction and numerical checks',
              'packages':{name:importlib.metadata.version(name) for name in ('qiskit','numpy','scipy','matplotlib')},
              'qiskit_reference':{'version':'2.6.2','source_relative':'qiskit/synthesis/evolution/suzuki_trotter.py','source_sha256':QISKIT_SOURCE_SHA,
                                   'grouping':'Nonempty commutingA/B groups; zero blocks omitted only as a declared reference shortcut','primitive_count_scope':'Unitary reference only, not a complete measured/physical resource model'},
              'case_ids':cases,'canonical_case_count':len(cases),'common_task':{'time_exact':args.time,'coupling':args.coupling,'x_field':args.x_field,'z_field':args.z_field,'initial_basis':args.initial_basis,'observable':'Z0'},
              'generation_wall_seconds_after_imports':time.perf_counter()-start,
              'timing_scope':'After CLI parsing through source preflight, input preparation, circuit/validation/figure/file generation; excludes interpreter imports and final manifest serialization',
              'command':{'script':'scripts/generate_simulation_suzuki.py','qubits':args.qubits,'orders':args.orders,'repetitions':args.repetitions,'time_exact':args.time,'coupling':args.coupling,'x_field':args.x_field,'z_field':args.z_field,'initial_basis':args.initial_basis,'no_sim':args.no_sim,'output_path':'omitted; use a dedicated fresh output directory'},
              'generator_guards':{'qubits':MAX_GENERATOR_QUBITS,'base_blocks':MAX_BASE_BLOCKS,'complete_primitive_operations':MAX_PRIMITIVE_OPERATIONS},
              'dense_guards':{'qubits':MAX_DENSE_QUBITS,'base_blocks':MAX_DENSE_BASE_BLOCKS,'evolution_body_operations':MAX_DENSE_PRIMITIVE_OPERATIONS},
              'files':files,'payload_bytes':sum(row['bytes'] for row in files.values()),'native_estimator_calls':0,'factory_searches':0,'gpu_jobs':0,'physical_resource_estimate':None,
              'scope':'Original small ideal circuit and independent numerical work/error study. Mathematical source scalar bounds are separate from splitting/synthesis/physical errors; no generic empirical step certificate, formal proof or unconditional advantage.'}
    if committed_source()!=source or any(sha((ROOT/name).read_bytes())!=digest for name,digest in frozen_hashes.items()):
        raise ValueError('producer source or exact checkout changed during generation; no completed manifest written')
    if sha(reference_path.read_bytes())!=QISKIT_SOURCE_SHA:
        raise ValueError('pinned reference source changed during generation; no completed manifest written')
    encoded=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    if manifest['payload_bytes']+len(encoded)>MAX_OUTPUT_BYTES:raise RuntimeError('output exceeds 64 MiB; no completed manifest written')
    temporary=output/'.manifest-write.tmp'
    try:
        with temporary.open('xb') as handle:handle.write(encoded);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,output/'manifest.json')
    finally:temporary.unlink(missing_ok=True)
    print(json.dumps({'status':manifest['status'],'cases':len(cases),'payload_bytes':manifest['payload_bytes'],'wall_seconds':manifest['generation_wall_seconds_after_imports']},indent=2))


if __name__=='__main__':main()
