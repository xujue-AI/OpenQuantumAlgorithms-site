#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit H2 energy-estimation circuits and bounded resource studies."""
from pathlib import Path
import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE_LAYOUT=(ROOT/'src/openquantum/energy_estimation.py').is_file()
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
    from openquantum import energy_estimation as energy
    from openquantum import qft_qpe,resource_estimation
else:
    import energy_estimation as energy
    import qft_qpe,resource_estimation
CircuitResourceBudget=resource_estimation.CircuitResourceBudget
SurfaceCodeProfile=resource_estimation.SurfaceCodeProfile
estimate_circuit_resources=resource_estimation.estimate_circuit_resources
from qiskit import qasm3
import numpy as np

SOURCE_FILES={
    'src/openquantum/energy_estimation.py':Path(energy.__file__),
    'src/openquantum/qft_qpe.py':Path(qft_qpe.__file__),
    'src/openquantum/resource_estimation.py':Path(resource_estimation.__file__),
    'scripts/generate_energy_estimation.py':Path(__file__),
    'requirements-lock.txt':INPUT_DIR/'requirements-lock.txt',
}
OWNED_PATHS=('src/openquantum/energy_estimation.py','src/openquantum/qft_qpe.py',
    'src/openquantum/resource_estimation.py','scripts/generate_energy_estimation.py','requirements-lock.txt')


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,default=serializable)+'\n')


def progress(stage,event,**details):
    """Flush stage diagnostics without changing the machine-readable stdout."""
    print(json.dumps({'generator':'energy-estimation','stage':stage,'event':event,**details}),
        file=sys.stderr,flush=True)


def package_version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None


def commit():
    return git_revision(INPUT_DIR)


def guard_output(path):
    path=path.resolve()
    if path==INPUT_DIR or INPUT_DIR.is_relative_to(path):
        raise ValueError('output cannot be the input directory or an ancestor')
    if SOURCE_LAYOUT:
        protected=ROOT/'assets/generated'
        owned=protected/'energy-estimation'
        if path.is_relative_to(protected) and not path.is_relative_to(owned):
            raise ValueError('output inside generated assets must remain in energy-estimation')
        if path.is_relative_to(owned/'pauli-access'):
            raise ValueError('Pauli-access artifacts require the separate Pauli generator')
    return path


def logical_case(dimers,phase_bits,ideal_failure,analytic_powers=False):
    start=time.perf_counter()
    contract=energy.ground_energy_contract(energy.H2_STO3G,dimers,phase_bits,ideal_failure=ideal_failure)
    circuit=energy.build_retained_target_qpe(energy.H2_STO3G,dimers=dimers,phase_bits=phase_bits,
        rounds=contract['qpe_rounds_per_retained_target'],analytic_powers=analytic_powers)
    lowered=energy.lower_circuit(circuit)
    qasm=qasm3.dumps(lowered)
    groups=contract['independent_groups']
    return circuit,lowered,qasm,{'contract':contract,'access_mode':circuit.metadata['access_mode'],'controlled_power_circuits_all_groups':circuit.metadata['controlled_power_circuits']*groups,'logical_qubits_per_group':circuit.num_qubits,
        'source_instruction_counts_per_group':dict(circuit.count_ops()),
        'primitive_counts_per_group':dict(lowered.count_ops()),'primitive_operations_per_group':len(lowered.data),
        'dependency_depth_per_group':lowered.depth(),'primitive_operations_all_groups':len(lowered.data)*groups,
        'qubit_measurements_all_groups':int(lowered.count_ops().get('measure',0))*groups,
        'hartree_fock_preparations_all_groups':groups,'qasm_sha256':hashlib.sha256(qasm.encode()).hexdigest(),
        'compiled_global_phase_radians':float(lowered.global_phase),
        'qasm_phase_policy':'pinned exporter can omit overall compiled global phase; complete measured groups preserve probabilities; not a recontrolled oracle specification',
        'construction_and_lowering_wall_seconds':time.perf_counter()-start,
        'full_static_group_simulated':False,'sequential_equivalent_simulation':None,
        'included':['one HF preparation per group','all repeated controlled exact dimer accesses','identity/shift coherent phases','every clock Hadamard and inverse QFT','all final clock measurements','all serial median groups','classical median/min rule'],
        'excluded_physical_costs':['fresh zero-state initialization/reset faults and timing','classical coefficient/SCF/mapping construction','compilation and decoder/routing implementation','classical readout/median communication latency'],
        'source_commit':commit(),'source_dirty':git_dirty(INPUT_DIR,OWNED_PATHS),
        'source_commit_scope':'exact checkout at generation, if available; content hashes identify working source',
        'source_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for name,path in SOURCE_FILES.items()},
        'environment':{'python':sys.version.split()[0],'device':'CPU','seed':20261112,
            'packages':{name:package_version(name) for name in ['qiskit','numpy','scipy','matplotlib','qdk']}}}


def projected(circuit,contract,cache):
    report=estimate_circuit_resources(circuit,budget=CircuitResourceBudget(
        synthesis_model_error_per_execution=1e-5,physical_failure_probability=.005,
        executions=contract['independent_groups']),profile=SurfaceCodeProfile(),cache_dir=cache,
        isolate_context=True)
    gc.collect()
    report['ideal_energy_contract']=contract
    report['initialization_included']=False
    report['hardware_energy_confidence_certified']=False
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimers',type=int,default=1)
    parser.add_argument('--phase-bits',type=int,default=5)
    parser.add_argument('--ideal-failure',type=float,default=.005)
    parser.add_argument('--no-sim',action='store_true')
    parser.add_argument('--no-scaling',action='store_true')
    parser.add_argument('--no-estimate',action='store_true')
    parser.add_argument('--analytic-powers',action='store_true')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    default_output=(ROOT/'assets/generated/energy-estimation' if SOURCE_LAYOUT else INPUT_DIR/'energy-estimation-results')
    if args.analytic_powers:default_output=default_output/'analytic-powers'
    output=guard_output(args.output_dir or default_output)
    if args.analytic_powers and output==ROOT/'assets/generated/energy-estimation':
        raise ValueError('Use a separate output directory for analytic-power variants')
    output.mkdir(parents=True,exist_ok=True)
    progress('generation','started',dimers=args.dimers,phase_bits=args.phase_bits,
        simulation=not args.no_sim,scaling=not args.no_scaling,estimate=not args.no_estimate,
        analytic_powers=args.analytic_powers)
    progress('default-construction','started',dimers=args.dimers,phase_bits=args.phase_bits)
    circuit,lowered,qasm,ledger=logical_case(args.dimers,args.phase_bits,args.ideal_failure,args.analytic_powers)
    progress('default-construction','completed',logical_qubits=circuit.num_qubits,
        primitive_operations=len(lowered.data))
    (output/'measured-group.qasm').write_text(qasm)
    del lowered,qasm
    gc.collect()
    dump(output/'molecular-fixture.json',{'model':asdict(energy.H2_STO3G),'provenance':energy.H2_FIXTURE_PROVENANCE})
    if not args.no_sim:
        progress('simulation-validation','started',dimers=args.dimers,phase_bits=args.phase_bits)
        reference=energy.reference_spectral_law(energy.H2_STO3G,args.dimers,args.phase_bits)
        contract=ledger['contract']
        samples=energy.sample_retained_target_groups(energy.H2_STO3G,dimers=args.dimers,phase_bits=args.phase_bits,
            rounds=contract['qpe_rounds_per_retained_target'],groups=contract['independent_groups'],analytic_powers=args.analytic_powers)
        robust=energy.robust_energy_reference(energy.H2_STO3G,dimers=args.dimers,phase_bits=args.phase_bits,
            rounds=contract['qpe_rounds_per_retained_target'],groups=contract['independent_groups'])
        small_actual=energy.simulate_complete_group(energy.H2_STO3G,phase_bits=3,rounds=3)
        small_reference=energy.joint_reference_distribution(energy.H2_STO3G,phase_bits=3,rounds=3)
        np.testing.assert_allclose(small_actual,small_reference,atol=1e-12,rtol=0)
        results={'single_round_reference':reference,'robust_reference':robust,'actual_sequential_samples':samples,
            'small_complete_11wire_joint_max_error':float(np.max(np.abs(small_actual-small_reference))),
            'small_complete_parameters':{'dimers':1,'phase_bits':3,'rounds':3},
            'actual_estimate_error':abs(samples['estimate']-reference['ground_energy']),
            'chemical_accuracy_claimed':False}
        dump(output/'results.json',results)
        ledger['sequential_equivalent_simulation']={'performed':True,'width':samples['simulated_width'],
            'rounds':samples['total_qpe_rounds'],'ground_reference':reference['ground_energy'],
            'reported_estimate':samples['estimate'],'error':results['actual_estimate_error'],
            'full_static_group_statevector_allocated':False}
        progress('simulation-validation','completed',simulated_width=samples['simulated_width'],
            total_qpe_rounds=samples['total_qpe_rounds'],
            small_complete_11wire_joint_max_error=results['small_complete_11wire_joint_max_error'])
    else:
        (output/'results.json').unlink(missing_ok=True)
    physical={'status':'unavailable','reason':'explicitly disabled'}
    if not args.no_estimate:
        progress('default-physical-query','started',logical_qubits=circuit.num_qubits)
        start=time.perf_counter();physical=projected(circuit,ledger['contract'],INPUT_DIR/'.cache/energy-qre')
        physical['projection_wall_seconds']=time.perf_counter()-start
        progress('default-physical-query','completed',status=physical['status'],
            pareto_points=len(physical.get('pareto_points',[])))
        physical['ideal_energy_contract']=ledger['contract']
        physical['initialization_included']=False
        physical['hardware_energy_confidence_certified']=False
        dump(output/'surface-code.json',physical)
    else:(output/'surface-code.json').unlink(missing_ok=True)
    ledger['physical_resource_estimate']={'status':physical['status'],
        'kind':physical.get('resource_estimate_kind'),'pareto_points':len(physical.get('pareto_points',[])),
        'hardware_energy_confidence_certified':False}
    if not args.no_scaling:
        import csv
        rows=[];projections=[]
        # Actual compiled circuits at a common coarse absolute energy target.
        # m increases with total spectral width; all dimers remain active.
        for dimers in [1,2,3,4]:
            phase_bits=max(2,math.ceil(math.log2(energy.energy_encoding(energy.H2_STO3G,dimers).span/.5)))
            progress('scaling-construction','started',data_qubits=2*dimers,phase_bits=phase_bits)
            row_circuit,row_lowered,row_qasm,row=logical_case(dimers,phase_bits,args.ideal_failure,args.analytic_powers)
            progress('scaling-construction','completed',data_qubits=2*dimers,phase_bits=phase_bits,
                logical_qubits=row_circuit.num_qubits,primitive_operations=len(row_lowered.data))
            case=output/f'n{2*dimers}';case.mkdir(exist_ok=True)
            (case/'measured-group.qasm').write_text(row_qasm);dump(case/'logical-resources.json',row)
            del row_lowered,row_qasm
            gc.collect()
            rows.append({'data_qubits':2*dimers,'phase_bits':phase_bits,
                'absolute_error_target_Ha':.5,'grid_error_bound_Ha':row['contract']['energy_absolute_error_ideal'],
                'ground_overlap':row['contract']['ground_overlap_lower_bound'],
                'groups':row['contract']['independent_groups'],'rounds_per_group':row['contract']['qpe_rounds_per_retained_target'],
                'logical_qubits':row['logical_qubits_per_group'],'primitive_operations_per_group':row['primitive_operations_per_group'],
                'primitive_operations_all_groups':row['primitive_operations_all_groups'],
                'output_measurements_all_groups':row['qubit_measurements_all_groups'],
                'controlled_U_calls':row_circuit.metadata['controlled_unitary_calls']*row['contract']['independent_groups'],
                'simulation_performed':False})
            if not args.no_estimate:
                progress('scaling-physical-query','started',data_qubits=2*dimers,phase_bits=phase_bits,
                    logical_qubits=row_circuit.num_qubits)
                report=projected(row_circuit,row['contract'],INPUT_DIR/'.cache/energy-qre')
                progress('scaling-physical-query','completed',data_qubits=2*dimers,
                    status=report['status'],pareto_points=len(report.get('pareto_points',[])))
                dump(case/'surface-code.json',report)
                projections.append({'data_qubits':2*dimers,'phase_bits':phase_bits,'contract':row['contract'],
                    'status':report['status'],'pareto_points':report['pareto_points']})
            else:(case/'surface-code.json').unlink(missing_ok=True)
        with (output/'scaling.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        dump(output/'surface-code-scaling.json',{'status':'computed_conditional_model' if projections else 'unavailable',
            'common_absolute_energy_error_target_Ha':.5,'same_ideal_failure':args.ideal_failure,
            'hardware_energy_confidence_certified':False,'points':projections})
    else:
        for name in ['scaling.csv','surface-code-scaling.json','scaling.svg']:(output/name).unlink(missing_ok=True)
        for n in [2,4,6,8]:
            case=output/f'n{n}'
            for name in ['measured-group.qasm','logical-resources.json','surface-code.json']:(case/name).unlink(missing_ok=True)
            if case.exists():
                try:case.rmdir()
                except OSError:pass
    progress('recycled-clock-construction','started',dimers=args.dimers,phase_bits=args.phase_bits)
    dynamic=energy.build_recycled_clock_qpe(energy.H2_STO3G,dimers=args.dimers,phase_bits=args.phase_bits,
        rounds=ledger['contract']['qpe_rounds_per_retained_target'],analytic_powers=args.analytic_powers)
    dynamic_lowered=energy.lower_circuit(dynamic)
    progress('recycled-clock-construction','completed',logical_qubits=dynamic.num_qubits,
        primitive_operations=len(dynamic_lowered.data))
    (output/'recycled-clock.qasm').write_text(qasm3.dumps(dynamic_lowered))
    dump(output/'recycled-clock-resources.json',{'logical_qubits':dynamic.num_qubits,
        'source_counts':dict(dynamic.count_ops()),'primitive_counts':dict(dynamic_lowered.count_ops()),
        'source_resets_per_group':dynamic.count_ops().get('reset',0),'measurement_bits_per_group':dynamic.num_clbits,
        'same_ideal_energy_contract':ledger['contract'],'physical_estimate':None,
        'physical_status':'reset/measurement instrument unsupported by current static importer'})
    if not args.analytic_powers:
        progress('analytic-power-construction','started',dimers=args.dimers,phase_bits=args.phase_bits)
        fast,fast_lowered,fast_qasm,fast_ledger=logical_case(args.dimers,args.phase_bits,args.ideal_failure,True)
        progress('analytic-power-construction','completed',logical_qubits=fast.num_qubits,
            primitive_operations=len(fast_lowered.data))
        fast_output=output/'analytic-powers';fast_output.mkdir(exist_ok=True)
        (fast_output/'measured-group.qasm').write_text(fast_qasm)
        dump(fast_output/'logical-resources.json',fast_ledger)
        fast_report={'status':'unavailable','reason':'explicitly disabled'}
        if not args.no_estimate:
            progress('analytic-power-physical-query','started',logical_qubits=fast.num_qubits)
            fast_report=projected(fast,fast_ledger['contract'],INPUT_DIR/'.cache/energy-qre')
            progress('analytic-power-physical-query','completed',status=fast_report['status'],
                pareto_points=len(fast_report.get('pareto_points',[])))
            dump(fast_output/'surface-code.json',fast_report)
        else:(fast_output/'surface-code.json').unlink(missing_ok=True)
        assert ledger['contract']==fast_ledger['contract']
        dump(output/'access-comparison.json',{'same_problem_and_contract':ledger['contract'],
            'same_physical_profile':asdict(SurfaceCodeProfile()),'same_synthesis_model_allowance':1e-5,
            'same_whole_workload_physical_allowance':.005,'hardware_or_synthesis_certificate':False,
            'repeated':{'logical_qubits':circuit.num_qubits,'operations':ledger['primitive_operations_per_group'],'power_circuits_all_groups':ledger['controlled_power_circuits_all_groups'],
                        'status':physical['status'],'pareto_points':physical.get('pareto_points',[])},
            'analytic_powers':{'logical_qubits':fast.num_qubits,'operations':len(fast_lowered.data),'power_circuits_all_groups':fast_ledger['controlled_power_circuits_all_groups'],
                        'status':fast_report['status'],'pareto_points':fast_report.get('pareto_points',[])},
            'scope':'exact algebraic dimer fast-forwarding; not generic Hamiltonian-oracle access'})
    dump(output/'logical-resources.json',ledger)
    for module in [energy,qft_qpe,resource_estimation]:
        (output/Path(module.__file__).name).write_bytes(Path(module.__file__).read_bytes())
    (output/'generate_energy_estimation.py').write_bytes(Path(__file__).read_bytes())
    license_path=INPUT_DIR/'LICENSE'
    if not license_path.is_file():license_path=INPUT_DIR/'LICENSE.txt'
    if license_path.is_file():(output/'LICENSE.txt').write_bytes(license_path.read_bytes())
    progress('drawing','started')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=energy.dimer_evolution(energy.H2_STO3G,-energy.energy_encoding(energy.H2_STO3G).time).draw('mpl',fold=28)
    figure.savefig(output/'dimer-access.svg',bbox_inches='tight');plt.close(figure)
    small=energy.build_retained_target_qpe(energy.H2_STO3G,phase_bits=3,rounds=1,measure=True)
    figure=small.draw('mpl',fold=24);figure.savefig(output/'small-qpe.svg',bbox_inches='tight');plt.close(figure)
    recycled_demo=energy.build_recycled_clock_qpe(energy.H2_STO3G,phase_bits=2,rounds=3)
    figure=recycled_demo.draw('mpl',fold=24);figure.savefig(output/'recycled-clock.svg',bbox_inches='tight');plt.close(figure)
    if not args.no_sim:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        energies=energy.energy_encoding(energy.H2_STO3G,args.dimers).energy(np.arange(2**args.phase_bits)/2**args.phase_bits)
        axes[0].bar(energies,reference['marginal'],width=energy.energy_encoding(energy.H2_STO3G,args.dimers).span/2**args.phase_bits*.75,color='#167d8d')
        axes[0].axvline(reference['ground_energy'],color='#b55231',label='exact ground energy');axes[0].legend()
        axes[0].set(xlabel='Decoded energy (Ha)',ylabel='Probability',title='One QPE round from HF')
        axes[1].bar(robust['energies'],robust['probabilities'],width=robust['error_tolerance']*.75,color='#7353a2')
        axes[1].axvline(reference['ground_energy'],color='#b55231');axes[1].set(xlabel='Reported energy (Ha)',ylabel='Probability',title='Minimum of retained-target medians')
        fig.tight_layout();fig.savefig(output/'energy-distributions.svg',bbox_inches='tight');plt.close(fig)
    else:(output/'energy-distributions.svg').unlink(missing_ok=True)
    if not args.no_scaling:
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        axes[0].plot([r['data_qubits'] for r in rows],[r['logical_qubits'] for r in rows],'o-',color='#167d8d')
        axes[0].set(xlabel='Active data qubits n',ylabel='Fresh-clock logical width',title='Same ≤0.5 Ha absolute target')
        axes[1].semilogy([r['data_qubits'] for r in rows],[r['primitive_operations_all_groups'] for r in rows],'o-',color='#7353a2')
        axes[1].set(xlabel='Active data qubits n',ylabel='All-group primitive operations',title='Actual compiled workloads')
        fig.tight_layout();fig.savefig(output/'scaling.svg',bbox_inches='tight');plt.close(fig)
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    progress('drawing','completed')
    progress('generation','completed',logical_group_width=circuit.num_qubits,
        primitive_operations=ledger['primitive_operations_per_group'],physical_points=len(physical.get('pareto_points',[])))
    print(json.dumps({'dimers':args.dimers,'data_qubits':2*args.dimers,'phase_bits':args.phase_bits,
        'logical_group_width':circuit.num_qubits,'ideal_contract':ledger['contract'],
        'primitive_operations':ledger['primitive_operations_per_group'],'physical_points':len(physical.get('pareto_points',[]))}))


if __name__=='__main__':main()
