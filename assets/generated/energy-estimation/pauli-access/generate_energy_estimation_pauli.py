#!/usr/bin/env python3
"""Generate explicit Pauli energy-QPE circuits and bounded independent checks."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator,Statevector
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
SOURCE_LAYOUT=(ROOT/'src/openquantum/energy_estimation_pauli.py').is_file()
INPUT_DIR=ROOT if SOURCE_LAYOUT else Path(__file__).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
try:
    from source_provenance import git_revision,git_dirty
except ImportError:
    # Individual public downloads may omit the optional provenance helper.
    def git_revision(root):return None
    def git_dirty(root,paths):return None
if SOURCE_LAYOUT:
    sys.path.insert(0,str(ROOT/'src'))
    from openquantum import energy_estimation_pauli as pauli
else:
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import energy_estimation_pauli as pauli


def synthetic_chain(qubits):
    pauli._integer(qubits,'qubits',2,pauli.MAX_GENERATOR_QUBITS)
    terms=[(-.4*qubits,'I'*qubits)]
    for wire in range(qubits):
        for c,p in ((.17,'Z'),(.23,'X')):
            label=['I']*qubits;label[qubits-1-wire]=p;terms.append((c,''.join(label)))
    for wire in range(qubits-1):
        label=['I']*qubits;label[qubits-1-wire]=label[qubits-2-wire]='Z';terms.append((.11,''.join(label)))
    return tuple(terms)


def guard_output(path):
    path=path.resolve()
    protected=ROOT/'assets/generated'
    owned=protected/'energy-estimation/pauli-access'
    if path==INPUT_DIR or INPUT_DIR.is_relative_to(path):raise ValueError('output cannot be the input directory or an ancestor')
    if SOURCE_LAYOUT and path.is_relative_to(protected) and not path.is_relative_to(owned):
        raise ValueError('output inside generated assets must remain in pauli-access')
    return path


def source_commit():
    return git_revision(INPUT_DIR)


def normalize_svg(path):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def save_figure(figure,path):
    figure.savefig(path,bbox_inches='tight');normalize_svg(path)
    figure.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=120)


def scaling_rows():
    rows=[]
    for n in [2,4,6,8,12]:
        for r in [1,4]:
            terms=synthetic_chain(n)
            circuit=pauli.build_pauli_energy_qpe(terms,phase_bits=3,steps=r,order=2,initial_basis=0)
            compiled=pauli.lower_circuit(circuit)
            rows.append({'family':'synthetic local Pauli chain; not molecular H2','qubits':n,'terms':len(terms),
                         'steps':r,'phase_bits':3,'logical_width':compiled.num_qubits,
                         'controlled_U_calls':7,'operations':len(compiled.data),'dependency_depth':compiled.depth(),
                         'CX':compiled.count_ops().get('cx',0),'continuous_rotations':sum(compiled.count_ops().get(x,0) for x in ['rx','ry','rz']),
                         'measurements':compiled.count_ops().get('measure',0),
                         'phase_grid_energy_width':pauli.energy_encoding(terms)['span']/8,
                         'statevector_performed':False,'physical_projection_performed':False})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-bits',type=int,default=3)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--order',type=int,default=2)
    parser.add_argument('--synthetic-chain',action='store_true')
    parser.add_argument('--qubits',type=int,default=4)
    parser.add_argument('--initial-basis',type=int)
    parser.add_argument('--no-sim',action='store_true')
    parser.add_argument('--no-scaling',action='store_true')
    parser.add_argument('--estimate',action='store_true')
    parser.add_argument('--compare-controls',action='store_true')
    parser.add_argument('--control-strategy',choices=['termwise','lift-all-gates'],default='termwise')
    parser.add_argument('--output-dir',type=Path,default=(ROOT/'assets/generated/energy-estimation/pauli-access' if SOURCE_LAYOUT else INPUT_DIR/'pauli-access-results'))
    args=parser.parse_args()
    output=guard_output(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    terms=synthetic_chain(args.qubits) if args.synthetic_chain else pauli.h2_jw_terms()
    initial=args.initial_basis if args.initial_basis is not None else (0 if args.synthetic_chain else pauli.H2_HF_BASIS)
    start=time.perf_counter()
    circuit=pauli.build_pauli_energy_qpe(terms,phase_bits=args.phase_bits,steps=args.steps,order=args.order,initial_basis=initial,
                                       control_strategy=args.control_strategy)
    generated_time=time.perf_counter()-start
    start=time.perf_counter();compiled=pauli.lower_circuit(circuit);lower_time=time.perf_counter()-start
    text=qasm3.dumps(compiled);(output/'qpe.qasm').write_text(text)
    module_path=Path(pauli.__file__).resolve()
    (output/'energy_estimation_pauli.py').write_bytes(module_path.read_bytes())
    (output/'generate_energy_estimation_pauli.py').write_bytes(Path(__file__).read_bytes())
    qft_path=(ROOT/'src/openquantum/qft_qpe.py') if SOURCE_LAYOUT else INPUT_DIR/'qft_qpe.py'
    if qft_path.is_file():(output/'qft_qpe.py').write_bytes(qft_path.read_bytes())
    license_path=INPUT_DIR/'LICENSE'
    if not license_path.exists():license_path=INPUT_DIR/'LICENSE.txt'
    if license_path.exists():(output/'LICENSE.txt').write_bytes(license_path.read_bytes())
    encoding=pauli.energy_encoding(terms)
    bound=pauli.product_formula_error_bound(terms,time=-2*math.pi/encoding['span'],steps=args.steps,order=args.order)
    metadata={
        'schema_version':1,'task':'one approximate-eigenenergy sample from explicit Pauli product-formula QPE',
        'model':'synthetic local Pauli chain' if args.synthetic_chain else 'H2 STO-3G four-spin-orbital Jordan-Wigner fixture',
        'energy_unit':'arbitrary benchmark unit' if args.synthetic_chain else 'Hartree',
        'fixture_provenance':None if args.synthetic_chain else pauli.H2_JW_PROVENANCE,
        'terms':[{'coefficient':c,'label':p} for c,p in terms],
        'parameters':{'phase_bits':args.phase_bits,'steps':args.steps,'order':args.order,'initial_basis':initial,'control_strategy':args.control_strategy},
        'energy_encoding':encoding,'logical_qubits':compiled.num_qubits,'source_counts':dict(circuit.count_ops()),
        'primitive_gate_counts':dict(compiled.count_ops()),'primitive_operations':len(compiled.data),'dependency_depth':compiled.depth(),
        'controlled_unitary_calls':2**args.phase_bits-1,'phase_measurements':args.phase_bits,
        'compiled_global_phase_radians':float(compiled.global_phase),
        'qasm_global_phase_policy':'exporter can omit overall compiled phase; qpe.qasm specifies complete measured trials, not a new coherently controlled oracle',
        'analytic_product_formula_bound':bound,'ground_confidence_guarantee':None,
        'ideal_spectral_sample_contract':pauli.spectral_sample_contract(terms,phase_bits=args.phase_bits,steps=args.steps,order=args.order),
        'simulation':{'performed':False,'reason':'explicitly disabled' if args.no_sim else 'pending'},
        'physical_resource_estimate':{'status':'unavailable','reason':'not requested'},
        'construction_wall_seconds':generated_time,'lowering_wall_seconds':lower_time,
        'source_commit':source_commit(),
        'source_dirty':git_dirty(INPUT_DIR,('src/openquantum/energy_estimation_pauli.py','src/openquantum/qft_qpe.py',
            'scripts/generate_energy_estimation_pauli.py','src/openquantum/resource_estimation.py','requirements-lock.txt')),
        'source_commit_scope':'exact checkout at generation, if available; content hashes identify working source',
        'source_sha256':{'energy_estimation_pauli.py':hashlib.sha256(module_path.read_bytes()).hexdigest(),
                         'generator.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'qft_qpe.py':hashlib.sha256(qft_path.read_bytes()).hexdigest() if qft_path.is_file() else None},
        'qasm_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'environment':{'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['qiskit','numpy','scipy','matplotlib']},'device':'CPU','seed':20261112},
        'limits':{'generator_qubits':pauli.MAX_GENERATOR_QUBITS,'input_terms':pauli.MAX_INPUT_TERMS,'source_operations':pauli.MAX_SOURCE_OPERATIONS,
                  'primitive_operations':pauli.MAX_PRIMITIVE_OPERATIONS,'statevector_qubits':pauli.MAX_STATEVECTOR_QUBITS,'dense_reference_qubits':pauli.MAX_REFERENCE_QUBITS},
        'excluded':['fresh integral/SCF and fermion-mapping computation (input fixture is supplied)',
                    'classical parsing/compilation/readout latency from physical model','physical initialization/reset timing and faults',
                    'concrete Clifford+T synthesis certificate','ground-energy selection/confidence beyond the single-sample contract'],
    }
    if not args.no_sim:
        start=time.perf_counter()
        results=pauli.compare_pauli_energy_qpe(terms,phase_bits=args.phase_bits,steps=args.steps,order=args.order,initial_basis=initial,
                                              sector_indices=None if args.synthetic_chain else pauli.H2_SECTOR_INDICES,
                                              control_strategy=args.control_strategy)
        results['simulation_wall_seconds']=time.perf_counter()-start
        (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        metadata['simulation']={'performed':True,'method':results['reference_method'],'results':'results.json',
                                'distribution_max_error':results['distribution_max_error'],'simulation_width':results['simulation_width']}
        rng=np.random.default_rng(20261112);draws=rng.choice(2**args.phase_bits,size=256,p=np.asarray(results['phase_distribution'])/sum(results['phase_distribution']))
        metadata['illustrative_ideal_readout']={'seed':20261112,'draws':256,'counts':{str(k):v for k,v in sorted(Counter(map(int,draws)).items())},
                    'model':'classical categorical draws from validated ideal gate-circuit distribution','hardware_run':False,
                    'does_not_certify_ground_energy':True}
    else:
        (output/'results.json').unlink(missing_ok=True);(output/'distribution.svg').unlink(missing_ok=True)
        (output/'distribution.png').unlink(missing_ok=True)
    if not args.no_scaling:
        rows=scaling_rows()
        with (output/'scaling.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        metadata['scaling']={'performed':True,'rows':len(rows),'family':'synthetic local Pauli chain; not molecule growth',
                             'all_rows_actually_compiled':True,'rows_statevector_validated':False,'matched_absolute_accuracy':False}
    else:
        (output/'scaling.csv').unlink(missing_ok=True);metadata['scaling']={'performed':False,'reason':'disabled'}
    other_circuit=None
    if args.compare_controls:
        other_strategy='lift-all-gates' if args.control_strategy=='termwise' else 'termwise'
        other_circuit=pauli.build_pauli_energy_qpe(terms,phase_bits=args.phase_bits,steps=args.steps,order=args.order,
                                                  initial_basis=initial,control_strategy=other_strategy)
        other_compiled=pauli.lower_circuit(other_circuit)
        other_qasm=qasm3.dumps(other_compiled);(output/'comparison-qpe.qasm').write_text(other_qasm)
        metadata['control_comparison']={'alternative_strategy':other_strategy,'alternative_primitive_gate_counts':dict(other_compiled.count_ops()),
            'alternative_primitive_operations':len(other_compiled.data),'alternative_dependency_depth':other_compiled.depth(),
            'alternative_qasm_sha256':hashlib.sha256(other_qasm.encode()).hexdigest(),
            'same_model_parameters':True,'same_ideal_access_polynomial':True,'same_phase_precision':True,
            'same_one_trial_output_contract':True,'synthesis_or_physical_accuracy_certificate':False}
        if not args.no_sim:
            alternative_distribution=Statevector.from_instruction(other_circuit.remove_final_measurements(inplace=False)).probabilities(qargs=list(range(args.phase_bits)))
            metadata['control_comparison']['selected_input_distribution_max_difference']=float(np.max(np.abs(alternative_distribution-np.asarray(results['phase_distribution']))))
    else:
        (output/'comparison-qpe.qasm').unlink(missing_ok=True);(output/'comparison-surface-code.json').unlink(missing_ok=True)
    if args.estimate:
        if SOURCE_LAYOUT:
            from openquantum.resource_estimation import CircuitResourceBudget,SurfaceCodeProfile,estimate_circuit_resources
            helper_path=ROOT/'src/openquantum/resource_estimation.py'
        else:
            from resource_estimation import CircuitResourceBudget,SurfaceCodeProfile,estimate_circuit_resources
            helper_path=Path(__file__).resolve().parent/'resource_estimation.py'
        budget=CircuitResourceBudget(synthesis_model_error_per_execution=1e-5,physical_failure_probability=.005,executions=1)
        report=estimate_circuit_resources(circuit,budget=budget,profile=SurfaceCodeProfile(),cache_dir=INPUT_DIR/'.cache/qre-energy-pauli')
        report['energy_output_contract']={'kind':'one approximate-eigenenergy sample','ground_confidence_guarantee':None,
            'algorithm_approximation':bound,'phase_grid_energy_width':encoding['span']/2**args.phase_bits,
            'synthesis_and_physical_model_do_not_certify_energy_accuracy':True}
        (output/'surface-code.json').write_text(json.dumps(report,indent=2)+'\n')
        if helper_path.is_file():(output/'resource_estimation.py').write_bytes(helper_path.read_bytes())
        metadata['source_sha256']['resource_estimation.py']=hashlib.sha256(helper_path.read_bytes()).hexdigest() if helper_path.is_file() else None
        metadata['physical_resource_estimate']={'status':report['status'],'kind':report['resource_estimate_kind'],
                                               'pareto_points':len(report['pareto_points']),'complete_trials':1,
                                               'physical_fault_allowance':.005,'synthesis_model_scalar_allowance':1e-5,
                                               'hardware_ground_confidence_certified':False,'report':'surface-code.json'}
        if other_circuit is not None:
            comparison=estimate_circuit_resources(other_circuit,budget=budget,profile=SurfaceCodeProfile(),cache_dir=INPUT_DIR/'.cache/qre-energy-pauli')
            comparison['comparison_contract']={'same_H_and_HF_input':True,'same_steps_order_and_phase_bits':True,
                'same_one_trial_physical_and_synthesis_model_allowances':True,'same_ideal_spectral_sample_contract':True,
                'comparison_remains_conditional_on_mean_fit_and_hardware_model':True,'energy_accuracy_or_runtime_tail_certificate':False}
            (output/'comparison-surface-code.json').write_text(json.dumps(comparison,indent=2)+'\n')
            metadata['control_comparison']['alternative_physical_pareto_points']=len(comparison['pareto_points'])
    else:
        (output/'surface-code.json').unlink(missing_ok=True)
        (output/'comparison-surface-code.json').unlink(missing_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    unitary=pauli.build_pauli_energy_unitary(terms,steps=args.steps,order=args.order)
    tau=2*math.pi/encoding['span']
    first_offdiag=next(((c,p) for c,p in terms if 'Y' in p or 'X' in p),terms[0])
    factor=pauli.pauli_exponential(*first_offdiag,-tau/args.steps)
    figure=factor.draw(output='mpl',fold=32,style={'backgroundcolor':'#FFFFFF'})
    save_figure(figure,output/'pauli-factor.svg');plt.close(figure)
    figure=circuit.draw(output='mpl',fold=24,style={'backgroundcolor':'#FFFFFF'})
    save_figure(figure,output/'qpe-circuit.svg');plt.close(figure)
    if not args.no_sim:
        energies=encoding['shift']+encoding['span']*np.arange(2**args.phase_bits)/2**args.phase_bits
        figure,axis=plt.subplots(figsize=(8,3.6))
        axis.bar(energies,results['phase_distribution'],width=encoding['span']/2**args.phase_bits*.75,color='#008c9d',alpha=.75,label='Actual ideal circuit')
        axis.plot(energies,results['reference_approximate_phase_distribution'],'ko',markersize=3,label='Independent approximate-V law')
        axis.axvline(results['exact_energy_ground_reference'],color='#bd5c25',linestyle='--',label='True H ground reference')
        axis.set(xlabel='Decoded energy ('+metadata['energy_unit']+')',ylabel='Probability',title='One Pauli-QPE energy sample; no ground-confidence selection')
        axis.legend(fontsize=8);figure.tight_layout();save_figure(figure,output/'distribution.svg');plt.close(figure)
        convergence=[]
        for order in [1,2]:
            for steps in [1,2,4,8,16]:
                access=pauli.product_formula_unitary(terms,time=-tau,steps=steps,order=order)
                exact=expm(1j*tau*pauli.reference_hamiltonian(terms))
                error=float(np.linalg.norm(Operator(access).data-exact,2))
                bound=pauli.product_formula_error_bound(terms,time=-tau,steps=steps,order=order)
                convergence.append({'order':order,'steps':steps,'operator_error':error,'analytic_operator_bound':bound['analytic_operator_error_bound']})
        with (output/'convergence.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(convergence[0]),lineterminator='\n');writer.writeheader();writer.writerows(convergence)
        figure,axis=plt.subplots(figsize=(6.6,3.6))
        for order,color in [(1,'#008c9d'),(2,'#bd5c25')]:
            values=[v for v in convergence if v['order']==order]
            axis.loglog([v['steps'] for v in values],[max(v['operator_error'],1e-16) for v in values],'o-',color=color,label=f'Order {order}: actual circuit')
            axis.loglog([v['steps'] for v in values],[max(v['analytic_operator_bound'],1e-16) for v in values],'--',color=color,label=f'Order {order}: analytic bound')
        axis.set(xlabel='Steps r',ylabel='Operator error',title='Access approximation at the energy-QPE base time')
        axis.legend(fontsize=8);figure.tight_layout();save_figure(figure,output/'convergence.svg');plt.close(figure)
    else:
        (output/'convergence.svg').unlink(missing_ok=True);(output/'convergence.csv').unlink(missing_ok=True)
        (output/'convergence.png').unlink(missing_ok=True)
    (output/'logical-resources.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'model':metadata['model'],'logical_qubits':compiled.num_qubits,'primitive_operations':len(compiled.data),
                      'simulation':metadata['simulation']['performed'],'physical_resource_estimate':metadata['physical_resource_estimate']}))


if __name__=='__main__':main()
