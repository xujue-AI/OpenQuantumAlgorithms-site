import importlib.util
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import weakref
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('energy_generator',ROOT/'scripts/generate_energy_estimation.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)


def test_logical_generation_does_not_require_optional_qdk_metadata(monkeypatch):
    original=generator.importlib.metadata.version
    def version(name):
        if name=='qdk':raise generator.importlib.metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(generator.importlib.metadata,'version',version)
    circuit,lowered,qasm,ledger=generator.logical_case(1,2,.1,True)
    assert ledger['environment']['packages']['qdk'] is None
    assert circuit.num_qubits==ledger['logical_qubits_per_group']
    assert 'OPENQASM 3' in qasm
    assert ledger['access_mode']=='analytic_dimer_powers'


def test_disabled_optional_artifacts_are_cleared_and_paired_power_costs_are_consistent(tmp_path,monkeypatch,capsys):
    for name in ['results.json','surface-code.json','scaling.csv','surface-code-scaling.json','scaling.svg','energy-distributions.svg']:
        (tmp_path/name).write_text('stale')
    case=tmp_path/'n2';case.mkdir()
    for name in ['measured-group.qasm','logical-resources.json','surface-code.json']:(case/name).write_text('stale')
    fast=tmp_path/'analytic-powers';fast.mkdir();(fast/'surface-code.json').write_text('stale')
    monkeypatch.setattr(sys,'argv',['generate_energy_estimation.py','--phase-bits','2','--ideal-failure','.1',
        '--no-sim','--no-scaling','--no-estimate','--output-dir',str(tmp_path)])
    generator.main()
    captured=capsys.readouterr()
    summary=json.loads(captured.out)
    records=[json.loads(line) for line in captured.err.splitlines()]
    assert all(record['generator']=='energy-estimation' for record in records)
    assert [(record['stage'],record['event']) for record in records]==[('generation','started')]+[
        (stage,event) for stage in ['default-construction','recycled-clock-construction',
            'analytic-power-construction','drawing'] for event in ['started','completed']
    ]+[('generation','completed')]
    assert records[0]['simulation']==records[0]['scaling']==records[0]['estimate']==False
    assert records[-1]['primitive_operations']==summary['primitive_operations']
    for name in ['results.json','surface-code.json','scaling.csv','surface-code-scaling.json','scaling.svg','energy-distributions.svg']:
        assert not (tmp_path/name).exists()
    assert not case.exists()
    assert not (fast/'surface-code.json').exists()
    comparison=json.loads((tmp_path/'access-comparison.json').read_text())
    assert comparison['repeated']['logical_qubits']==comparison['analytic_powers']['logical_qubits']
    assert comparison['analytic_powers']['power_circuits_all_groups']<comparison['repeated']['power_circuits_all_groups']
    assert comparison['repeated']['status']==comparison['analytic_powers']['status']=='unavailable'
    ledger=json.loads((tmp_path/'logical-resources.json').read_text())
    assert ledger['sequential_equivalent_simulation'] is None
    assert ledger['physical_resource_estimate']['status']=='unavailable'
    dynamic=json.loads((tmp_path/'recycled-clock-resources.json').read_text())
    assert dynamic['physical_estimate'] is None
    assert dynamic['source_resets_per_group']>0


def test_output_guard_protects_other_algorithms_and_source():
    for path in [ROOT,ROOT.parent,ROOT/'assets/generated/grover',
                 ROOT/'assets/generated/energy-estimation/pauli-access']:
        with pytest.raises(ValueError):generator.guard_output(path)
    assert generator.guard_output(ROOT/'assets/generated/energy-estimation')==ROOT/'assets/generated/energy-estimation'


def test_compiled_exports_are_released_before_projection_without_changing_saved_counts(tmp_path,monkeypatch,capsys):
    original=generator.logical_case
    compiled_refs=[];qasm_refs=[];exports=[];projections=[]
    class ExportedQasm(str):pass
    def bounded_case(dimers,phase_bits,ideal_failure,analytic_powers=False):
        # Real small circuits audit lifetime and export pass-through; native
        # estimation and the full scaling-accuracy contract are not tested here.
        circuit,lowered,qasm,ledger=original(dimers,2,.1,analytic_powers)
        compiled_refs.append(weakref.ref(lowered))
        qasm=ExportedQasm(qasm)
        qasm_refs.append(weakref.ref(qasm))
        exports.append({'dimers':dimers,'qasm_bytes':qasm.encode(),'operations':len(lowered.data),
            'sha256':ledger['qasm_sha256']})
        return circuit,lowered,qasm,ledger
    def projection(circuit,contract,cache):
        projections.append((circuit.num_qubits,contract))
        # The default and four scaling exports are dead before every query.
        # The separate analytic-power export is outside this cleanup's scope.
        assert all(reference() is None for reference in compiled_refs[:5])
        assert all(reference() is None for reference in qasm_refs[:5])
        return {'status':'computed_conditional_model','pareto_points':[],
            'resource_estimate_kind':'test_stub'}
    monkeypatch.setattr(generator,'logical_case',bounded_case)
    monkeypatch.setattr(generator,'projected',projection)
    monkeypatch.setattr(sys,'argv',['generate_energy_estimation.py','--phase-bits','2',
        '--ideal-failure','.1','--no-sim','--output-dir',str(tmp_path)])
    generator.main()
    summary=json.loads(capsys.readouterr().out)
    assert len(projections)==6
    paths=[tmp_path]+[tmp_path/f'n{2*dimers}' for dimers in [1,2,3,4]]+[tmp_path/'analytic-powers']
    for path,export in zip(paths,exports,strict=True):
        assert (path/'measured-group.qasm').read_bytes()==export['qasm_bytes']
        saved=json.loads((path/'logical-resources.json').read_text())
        assert saved['primitive_operations_per_group']==export['operations']
        assert saved['qasm_sha256']==export['sha256']
    comparison=json.loads((tmp_path/'access-comparison.json').read_text())
    assert comparison['repeated']['operations']==summary['primitive_operations']==exports[0]['operations']
    assert comparison['analytic_powers']['operations']==exports[-1]['operations']


@pytest.mark.parametrize('pauli_access',[False,True])
@pytest.mark.parametrize('source_layout',[False,True])
def test_public_generator_has_null_git_provenance_under_unrelated_repository(tmp_path,pauli_access,source_layout):
    ancestor=tmp_path/'unrelated';ancestor.mkdir()
    subprocess.run(['git','init','-q',str(ancestor)],check=True)
    (ancestor/'unrelated.txt').write_text('This repository does not own the downloaded quantum code.\n')
    subprocess.run(['git','-C',str(ancestor),'add','unrelated.txt'],check=True)
    subprocess.run(['git','-C',str(ancestor),'-c','user.name=Test Fixture',
        '-c','user.email=fixture@example.invalid','commit','-qm','unrelated fixture'],check=True)
    download=ancestor/'download';download.mkdir()
    scripts=download/'scripts' if source_layout else download
    modules=download/'src/openquantum' if source_layout else download
    scripts.mkdir(parents=True,exist_ok=True);modules.mkdir(parents=True,exist_ok=True)
    if source_layout:(modules/'__init__.py').write_text('')
    module_name='energy_estimation_pauli.py' if pauli_access else 'energy_estimation.py'
    for name in [module_name,'qft_qpe.py','resource_estimation.py']:
        shutil.copy2(ROOT/'src/openquantum'/name,modules/name)
    generator_name='generate_energy_estimation_pauli.py' if pauli_access else 'generate_energy_estimation.py'
    shutil.copy2(ROOT/'scripts'/generator_name,scripts/generator_name)
    # Source bundles include this helper; individual downloads may omit it.
    provenance_helper=ROOT/'scripts/source_provenance.py'
    if source_layout and provenance_helper.is_file():
        shutil.copy2(provenance_helper,scripts/'source_provenance.py')
    shutil.copy2(ROOT/'LICENSE',download/'LICENSE')
    output=download/'results'
    command=[sys.executable,str(scripts/generator_name),'--phase-bits','2','--no-sim','--no-scaling','--output-dir',str(output)]
    command+=['--steps','1'] if pauli_access else ['--ideal-failure','.1','--no-estimate']
    environment=os.environ.copy()
    environment.update({'PYTHONPATH':'','MPLBACKEND':'Agg','GIT_DIR':str(ancestor/'.git'),'GIT_WORK_TREE':str(ancestor)})
    completed=subprocess.run(command,cwd=download,env=environment,capture_output=True,text=True,timeout=90)
    assert completed.returncode==0,completed.stderr
    if not pauli_access:
        summary=json.loads(completed.stdout)
        records=[json.loads(line) for line in completed.stderr.splitlines()]
        assert records[0]['stage']=='generation' and records[0]['event']=='started'
        assert records[-1]['stage']=='generation' and records[-1]['event']=='completed'
        assert records[-1]['logical_group_width']==summary['logical_group_width']
    ledger=json.loads((output/'logical-resources.json').read_text())
    assert ledger['source_commit'] is None
    assert ledger['source_dirty'] is None
    assert ledger['source_sha256']
    assert all(value is None or len(value)==64 for value in ledger['source_sha256'].values())
