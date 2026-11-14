"""Saved fixtures and synthetic subprocesses; no real producer or Native work."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import textwrap

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tested_dynamics_verifier', ROOT/'scripts/verify_dynamics_resources.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
gate = v.gate


def summary():
    return '\n'.join([
        '| Method | Cases | Measured circuit/approximation evidence | Physical evidence |',
        '|---|---:|---|---|',
        '| Product formulas | 18 | Actual n=2–6 size cases, t=.5/1/2 and accuracy .025/.05/.1 slices; analytic step bounds | 0 conditional profile queries |',
        '| Complex QSP and integer OAA | 4 | Three-spin t=1 accepted blocks, rounds 0–3; every failed attempt charged | 0 conditional profile queries; three-spin amplification slice |',
        '| Retargeted complex QSP | 2 additional | n2/n4 at fixed J1,h.7,t1; all actual accepted-block basis columns checked, alongside n3 above | 0 conditional profile queries; bounded three-size slice |',
        '', 'The mean synthesis fit is not an error certificate. Initialization/reset and classical communication costs remain excluded.',
    ])+'\n'


@pytest.fixture
def saved_replay(tmp_path):
    records=ROOT/gate.RECORD_DIRECTORY; output=tmp_path/'replay'; output.mkdir()
    for name in v.LOGICAL_FILES:
        target=output/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(records/name,target)
    sources=gate.source_hashes(ROOT,gate.REPLAY_SOURCE_FILES)
    native=gate.load_json(records/'manifest.json'); replay=copy.deepcopy(native)
    replay['status']='logical_only'; replay['actual_physical_queries']=0
    replay['QSP_physical_size_sweep']='Not computed; n2,n3,n4 logical circuit checks at fixed J1,h.7,t1, with no physical projection'
    replay['provenance']['source_commit']=None
    replay['provenance']['source_sha256']={p:sources[p] for p in gate.NATIVE_SOURCE_FILES}
    for case in replay['cases']:
        case['provenance']=copy.deepcopy(replay['provenance']); case['physical_reports']=[]
        case['cpu_timings_seconds']={p:1.25 for p in case['cpu_timings_seconds']}
        (output/case['case_id']/'logical-resources.json').write_text(json.dumps(case))
    (output/'manifest.json').write_text(json.dumps(replay)); (output/'summary.md').write_text(summary())
    shutil.copy2(ROOT/'src/openquantum/dynamics_resources.py',output/'dynamics_resources.py')
    manifest={'native_provenance':native['provenance'],'replay_sources':sources}
    return records,output,manifest,replay


def test_saved_comparison_preserves_every_numerical_value(saved_replay):
    records,output,manifest,_=saved_replay; before=gate.core_inventory(records)
    result=v.compare_logical_outputs(records,output,manifest,ROOT)
    assert result['numeric_allowances']==[] and not result['original_cost_budgets_repriced']
    assert gate.core_inventory(records)==before


@pytest.mark.parametrize('field',['reference','budget','cap','phase'])
def test_one_bit_numeric_or_workload_changes_fail(saved_replay,field):
    records,output,manifest,replay=saved_replay
    case=replay['cases'][22 if field=='phase' else 18]
    if field=='reference':
        key='seeded_full_state_lowering_error'; case['numerical_checks'][key]=math.nextafter(case['numerical_checks'][key],math.inf)
    elif field=='budget':
        key='synthesis_model_error_per_attempt'; case['postselection_contract'][key]=math.nextafter(case['postselection_contract'][key],math.inf)
    elif field=='cap':case['complete_attempts']+=1
    else:case['phase_evidence']['cosine_phases'][0]=math.nextafter(case['phase_evidence']['cosine_phases'][0],math.inf)
    (output/'manifest.json').write_text(json.dumps(replay))
    (output/case['case_id']/'logical-resources.json').write_text(json.dumps(case))
    with pytest.raises(v.LogicalReplayMismatch) as caught:v.compare_logical_outputs(records,output,manifest,ROOT)
    assert caught.value.details['files'] and not caught.value.details['original_cost_budgets_repriced']


@pytest.mark.parametrize('field',['source','timer','environment','Git','physical'])
def test_ignored_metadata_is_checked_before_normalizing(saved_replay,field):
    _,_,manifest,replay=saved_replay;case=replay['cases'][0]
    expected={p:manifest['replay_sources'][p] for p in gate.NATIVE_SOURCE_FILES}
    if field=='source':case['provenance']['source_sha256']['requirements-lock.txt']='a'*64
    elif field=='timer':case['cpu_timings_seconds']['construction_and_lowering']=True
    elif field=='environment':case['provenance']['environment']['seed']=True
    elif field=='Git':case['provenance']['source_commit']='invalid'
    else:case['physical_reports']=['frozen-p-1e-3.json']
    with pytest.raises(ValueError):v.normalized_case(case,expected,replay=True)


def test_typed_diff_bounds_and_private_strings():
    diff=v.typed_json_diff({'a':1,'b':True,'text':'private-one'},{'a':1.0,'b':1,'text':'private-two'})
    assert diff['different_entries']==3 and diff['differences'][0]['reason']=='type'
    assert 'private' not in json.dumps(diff)
    assert v.typed_json_diff(-0.0,0.0)['different_entries']==1
    assert v.typed_json_diff({'a/b~':None},{})['differences'][0]['path']=='/a~1b~0'
    assert v.typed_json_diff([0]*4,[1]*4,max_rows=2)['truncated']
    with pytest.raises(ValueError):v.typed_json_diff(float('inf'),1.0)
    with pytest.raises(ValueError):v.typed_json_diff([0]*4,[0]*4,max_nodes=2)


def test_inventory_rejects_extra_native_outputs(saved_replay):
    _,output,_,_=saved_replay
    extra=output/gate.CASE_IDS[0]/'frozen-p-1e-3.json';extra.write_text('{}')
    with pytest.raises(ValueError,match='unexpected replay file'):v.output_inventory(output,complete=True)
    extra.unlink();path=output/'summary.md';path.unlink();path.symlink_to(output/'manifest.json')
    with pytest.raises(ValueError,match='linked replay output'):v.output_inventory(output,complete=True)


def synthetic(tmp_path,body):
    source=tmp_path/'source';(source/'scripts').mkdir(parents=True)
    (source/v.PRODUCER).write_text(textwrap.dedent(body))
    work=tmp_path/'work';work.mkdir()
    return source.resolve(),work.resolve(),(work/'logical-replay').resolve()


def test_timeout_cleans_separate_session_descendant(tmp_path):
    source,work,output=synthetic(tmp_path,"""
        from pathlib import Path
        import subprocess,sys,time
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        Path('descendant.pid').write_text(str(p.pid));time.sleep(60)
    """)
    with pytest.raises(TimeoutError) as caught:v.run_producer(source,output,work,1)
    pid=int((work/'descendant.pid').read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
    assert caught.value.replay_metrics['Native_estimates_run'] is False


def test_log_abort_retains_failure_metrics(tmp_path,monkeypatch):
    source,work,output=synthetic(tmp_path,"""
        import sys,time
        sys.stdout.write('x'*2048);sys.stdout.flush();time.sleep(60)
    """)
    monkeypatch.setattr(v,'MAX_LOG_BYTES',1024)
    with pytest.raises(ValueError,match='log size') as caught:v.run_producer(source,output,work,2)
    assert caught.value.replay_metrics['exit_status'] is not None


def test_term_error_still_forces_kill(tmp_path,monkeypatch):
    source,work,output=synthetic(tmp_path,'import time;time.sleep(60)')
    real=v._signal_owned_group;calls=[]
    def failure(child,tracked,number):
        calls.append(number)
        if number==signal.SIGTERM:raise OSError('synthetic TERM error')
        return real(child,tracked,number)
    monkeypatch.setattr(v,'_signal_owned_group',failure)
    with pytest.raises(OSError,match='TERM error'):v.run_producer(source,output,work,.2)
    assert signal.SIGKILL in calls


def test_log_prefix_capture_survives_oversize_abort(tmp_path,monkeypatch):
    work=tmp_path/'work';capture=tmp_path/'capture';work.mkdir();capture.mkdir()
    (work/'producer.stdout').write_bytes(b'x'*200);monkeypatch.setattr(v,'MAX_LOG_BYTES',128)
    v._retain(work,work/'replay',capture,{},ValueError('synthetic failure'))
    assert (capture/'producer.stdout').stat().st_size==128
    assert not (capture/'receipt.json').exists()


def test_blocked_deadline_is_rejected():
    old=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM})
    try:
        with pytest.raises(ValueError,match='blocked wall timer'):
            with v.wall_limit(1):raise AssertionError('must not enter')
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)


@pytest.mark.parametrize('wall',[0,-1,601,float('nan'),True])
def test_bad_total_wall_limits_fail_before_replay(wall):
    with pytest.raises(ValueError,match='total wall cap'):
        v.verify_record(ROOT/gate.RECORD_DIRECTORY,ROOT/gate.RECORD_DIRECTORY/gate.MANIFEST_NAME,'a'*64,max_wall_seconds=wall)


def test_rss_abort_uses_sampled_process_tree_and_cleans(tmp_path,monkeypatch):
    source,work,output=synthetic(tmp_path,'import time; data=bytearray(64*1024*1024); time.sleep(60)')
    monkeypatch.setattr(v,'MAX_RSS_BYTES',16*1024*1024)
    with pytest.raises(ValueError,match='RSS abort') as caught:
        v.run_producer(source,output,work,2,rss_limit=v.MAX_RSS_BYTES)
    assert caught.value.replay_metrics['sampled_process_tree_peak_rss_bytes']>v.MAX_RSS_BYTES
    assert caught.value.replay_metrics['exit_status'] is not None


@pytest.mark.parametrize('kind',['JSON','aggregate'])
def test_synthetic_output_bounds_abort_and_reap(tmp_path,monkeypatch,kind):
    if kind=='JSON':
        monkeypatch.setattr(gate,'MAX_JSON_BYTES',128)
        body="from pathlib import Path; import sys,time; p=Path(sys.argv[-1]);p.mkdir();(p/'manifest.json').write_text('x'*1024);time.sleep(60)"
    else:
        monkeypatch.setattr(v,'MAX_OUTPUT_BYTES',128)
        body="from pathlib import Path; import sys,time; p=Path(sys.argv[-1]);p.mkdir();(p/'summary.md').write_text('x'*1024);time.sleep(60)"
    source,work,output=synthetic(tmp_path,body)
    with pytest.raises(ValueError,match='size limit') as caught:
        v.run_producer(source,output,work,2,output_limit=v.MAX_OUTPUT_BYTES)
    assert caught.value.replay_metrics['exit_status'] is not None


def test_late_exit_zero_does_not_bypass_child_deadline(tmp_path,monkeypatch):
    source,work,output=synthetic(tmp_path,'pass')
    class Child:
        pid=999999999;returncode=0
        def poll(self):return 0
    monkeypatch.setattr(v.subprocess,'Popen',lambda *a,**k:Child())
    monkeypatch.setattr(v,'_owned_processes',lambda child,tracked:0)
    monkeypatch.setattr(v,'cleanup_owned_group',lambda child,tracked:None)
    monkeypatch.setattr(v,'output_inventory',lambda *a,**k:{})
    times=iter([0.,1.01,1.02]);monkeypatch.setattr(v.time,'monotonic',lambda:next(times,1.02))
    with pytest.raises(TimeoutError,match='wall allowance') as caught:v.run_producer(source,output,work,1)
    assert caught.value.replay_metrics['exit_status']==0


def test_reused_group_leader_is_not_signalled(monkeypatch):
    class Process:
        def create_time(self):return 200.
    class Child:pid=12345
    monkeypatch.setattr(v.psutil,'Process',lambda pid:Process())
    monkeypatch.setattr(v.os,'killpg',lambda *a:pytest.fail('must not signal reused PID group'))
    v._signal_owned_group(Child(),{12345:100.},signal.SIGKILL)


def test_available_json_survives_aggregate_capture_abort(tmp_path):
    work=tmp_path/'work';capture=tmp_path/'capture';output=work/'replay'
    work.mkdir();capture.mkdir();output.mkdir()
    (output/'manifest.json').write_text('{"completed_check": true}')
    case=output/gate.CASE_IDS[0];case.mkdir()
    with (case/'measured-trial.qasm').open('wb') as handle:handle.truncate(v.MAX_OUTPUT_BYTES+1)
    v._retain(work,output,capture,{},ValueError('synthetic output abort'))
    assert json.loads((capture/'logical-replay/manifest.json').read_text())=={'completed_check':True}
    assert (capture/'capture-output-error.json').exists()
    assert json.loads((capture/'partial-json-capture.json').read_text())['partial'] is True


@pytest.mark.parametrize('mutation',['historical','baseline'])
def test_post_replay_input_mutation_produces_failure_receipt(tmp_path,monkeypatch,mutation):
    records=tmp_path/'records';shutil.copytree(ROOT/gate.RECORD_DIRECTORY,records)
    source=tmp_path/'source';source.mkdir()
    names=set(gate.REPLAY_SOURCE_FILES)|set(gate.VERIFIER_FILES)|set(gate.FROZEN_FILES)|{gate.HISTORICAL_ADAPTER_PATH}
    for name in names:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    original=gate.load_json(records/'manifest.json')
    manifest={'native_provenance':original['provenance'],'replay_sources':gate.source_hashes(source,gate.REPLAY_SOURCE_FILES),
              'verifier_sources':gate.source_hashes(source,gate.VERIFIER_FILES),'frozen_files':gate.frozen_hashes(records,source)}
    (records/gate.MANIFEST_NAME).write_text(json.dumps(manifest))
    monkeypatch.setattr(gate,'verify_dynamics_record_identity',lambda *a,**k:{'physical_validation':{}})
    monkeypatch.setattr(v,'locked_environment',lambda source:{'python':'3.12.14','packages':{}})
    monkeypatch.setattr(v,'compare_logical_outputs',lambda *a,**k:{'status':'synthetic comparison passed'})
    def fake_run(source_root,output,workdir,timeout):
        path=source_root/(gate.HISTORICAL_ADAPTER_PATH if mutation=='historical' else 'scripts/estimate_resources.py')
        path.write_text('# changed synthetic input\n')
        return {'Native_estimates_run':False,'phase_fitter_run':False}
    monkeypatch.setattr(v,'run_producer',fake_run)
    capture=tmp_path/'capture'
    with pytest.raises(ValueError,match='sources changed'):
        v.verify_record(records,records/gate.MANIFEST_NAME,'a'*64,source_root=source,capture_dir=capture,max_wall_seconds=10)
    receipt=json.loads((capture/'receipt.json').read_text())
    assert receipt['status']=='dynamics verification failed'
    assert receipt['strict_comparison_passed'] is False


def test_hard_total_deadline_covers_retention_after_child_timeout(tmp_path,monkeypatch):
    import time
    records=tmp_path/'records';shutil.copytree(ROOT/gate.RECORD_DIRECTORY,records)
    original=gate.load_json(records/'manifest.json')
    manifest={'native_provenance':original['provenance'],'replay_sources':gate.source_hashes(ROOT,gate.REPLAY_SOURCE_FILES),
              'verifier_sources':gate.source_hashes(ROOT,gate.VERIFIER_FILES),'frozen_files':gate.frozen_hashes(records,ROOT)}
    (records/gate.MANIFEST_NAME).write_text(json.dumps(manifest))
    monkeypatch.setattr(gate,'verify_dynamics_record_identity',lambda *a,**k:{'physical_validation':{}})
    monkeypatch.setattr(v,'locked_environment',lambda source:{'python':'3.12.14','packages':{}})
    def child_timeout(*a,**k):
        time.sleep(.12)
        raise TimeoutError('synthetic replay timeout')
    monkeypatch.setattr(v,'run_producer',child_timeout)
    entered=[]
    def delayed_retention(*a,**k):
        entered.append(True)
        time.sleep(5)
    monkeypatch.setattr(v,'_retain',delayed_retention)
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        v.verify_record(records,records/gate.MANIFEST_NAME,'a'*64,max_wall_seconds=.7)
    assert entered and time.monotonic()-started<2
