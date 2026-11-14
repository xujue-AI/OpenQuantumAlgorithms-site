# SPDX-License-Identifier: MIT
"""Small mocked checks of bounded output, source identity and final publication."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


RESERVATION = 128*1024


def load_generator():
    path = Path(__file__).resolve().parents[1]/'scripts/generate_simulation_suzuki.py'
    spec = importlib.util.spec_from_file_location('_suzuki_generator_release_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator():
    return load_generator()


@pytest.fixture
def fake_checkout(generator, monkeypatch, tmp_path):
    root = tmp_path/'checkout'
    root.mkdir()
    committed = {}
    for index, name in enumerate(generator.SOURCE_FILES):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        committed[name] = f'producer-{index}\n'.encode()
        path.write_bytes(committed[name])
    head = 'a'*40
    calls = []
    def git_run(command, *, cwd, capture_output, timeout, text=False):
        assert Path(cwd) == root and capture_output and timeout == 5
        calls.append(command)
        if command == ['git', 'rev-parse', '--show-toplevel']:
            stdout = str(root)+'\n'
        elif command == ['git', 'rev-parse', 'HEAD']:
            stdout = head+'\n'
        elif command[:2] == ['git', 'show']:
            revision, name = command[2].split(':', 1)
            assert revision == head
            return subprocess.CompletedProcess(command, 0, committed[name], b'')
        else:
            raise AssertionError(f'unexpected command {command}')
        assert text
        return subprocess.CompletedProcess(command, 0, stdout, '')
    monkeypatch.setattr(generator, 'ROOT', root)
    monkeypatch.setattr(generator.subprocess, 'run', git_run)
    return SimpleNamespace(root=root, committed=committed, head=head, calls=calls)


@pytest.fixture
def mocked_generation(generator, fake_checkout, monkeypatch, tmp_path):
    reference = tmp_path/'pinned-reference.py'
    reference.write_bytes(b'pinned Suzuki reference\n')
    monkeypatch.setattr(generator.inspect, 'getfile', lambda cls: str(reference))
    monkeypatch.setattr(generator, 'QISKIT_SOURCE_SHA', hashlib.sha256(reference.read_bytes()).hexdigest())
    monkeypatch.setattr(generator.importlib.metadata, 'version', lambda name: '2.6.2' if name == 'qiskit' else 'mocked')
    monkeypatch.setattr(generator, 'request', lambda *args, **kwargs: None)
    monkeypatch.setattr(generator, '_dense_request', lambda *args, **kwargs: None)
    calls = {'circuits': 0, 'checks': 0, 'exports': 0}
    class Figure:
        def savefig(self, handle, **kwargs):
            handle.write(b'<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    def circuit(*args, **kwargs):
        calls['circuits'] += 1
        return SimpleNamespace(draw=lambda **unused: Figure())
    def export(unused):
        calls['exports'] += 1
        return 'OPENQASM 3.0;\nqubit[1] q;\n'
    def ledger(chain, time, repetitions, order, **kwargs):
        return {'complete_measured_primitive_counts': {'reset': chain.qubits, 'measure': chain.qubits, 'rx': chain.qubits},
                'arbitrary_rotations': chain.qubits, 'source_dependency_depth': 1,
                'source_parameter_rounding': {'negative_actual_substeps': 0, 'actual_absolute_substep_time_sum_exact': '1',
                                             'total_source_parameter_operator_bound_upper_float': 0.0}}
    def comparison(*args, **kwargs):
        calls['checks'] += 1
        return {'operator_error_vs_exact_evolution': 0.0, 'state_vector_error_vs_exact_evolution': 0.0,
                'observable_bias': 0.0, 'exact_z0': 1.0, 'circuit_z0': 1.0, 'source_to_qiskit_suzuki_operator_error': 0.0}
    monkeypatch.setattr(generator, 'suzuki_circuit', circuit)
    monkeypatch.setattr(generator.qasm3, 'dumps', export)
    monkeypatch.setattr(generator, 'logical_ledger', ledger)
    monkeypatch.setattr(generator, 'compare_case', comparison)
    monkeypatch.setattr(generator.plt, 'close', lambda unused: None)
    monkeypatch.setattr(generator, 'coefficient_figure', lambda output: generator.save_svg(output, 'signed-substeps.svg', Figure()))
    def convergence(output, rows):
        assert rows
        generator.save_svg(output, 'error-vs-steps.svg', Figure())
        generator.save_svg(output, 'error-vs-work.svg', Figure())
    monkeypatch.setattr(generator, 'convergence_figures', convergence)
    output = tmp_path/'new-output'
    return SimpleNamespace(output=output, calls=calls, checkout=fake_checkout, reference=reference)


def arguments(monkeypatch, output, *, canonical=False):
    args = ['generate_simulation_suzuki.py', '--output-dir', str(output)]
    if not canonical:
        args += ['--no-sim', '--qubits', '1', '--orders', '2', '--repetitions', '1']
    monkeypatch.setattr(sys, 'argv', args)


def test_incremental_payload_cap_reserves_manifest_and_rejects_before_write(generator, tmp_path):
    assert generator.MAX_OUTPUT_BYTES == 64*1024*1024
    output = tmp_path/'output'; output.mkdir()
    filler = output/'existing-payload.bin'
    # Sparse file: checks the true64MiB stat boundary without a64MiB allocation.
    with filler.open('wb') as handle:
        handle.truncate(generator.MAX_OUTPUT_BYTES-RESERVATION-3)
    target = output/'next.bin'
    with pytest.raises(RuntimeError, match='reservation'):
        generator.write_payload(output, target, b'four')
    assert not target.exists() and not (output/'manifest.json').exists()
    generator.write_payload(output, target, b'yes')
    assert target.read_bytes() == b'yes'
    # Replacement subtracts only the replaced file's real previous byte count.
    generator.write_payload(output, target, b'ok')
    assert target.read_bytes() == b'ok'
    with pytest.raises(RuntimeError, match='reservation'):
        generator.write_payload(output, target, b'four')
    assert target.read_bytes() == b'ok'


def test_exact_checkout_root_binds_head_and_all_four_source_files(generator, fake_checkout):
    assert generator.committed_source() == fake_checkout.head
    shows = [call[2].split(':', 1)[1] for call in fake_checkout.calls if call[:2] == ['git', 'show']]
    assert shows == list(generator.SOURCE_FILES)


def test_parent_git_root_is_not_misattributed_as_source_commit(generator, fake_checkout, monkeypatch):
    calls = []
    def parent_root(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, str(fake_checkout.root.parent)+'\n', '')
    monkeypatch.setattr(generator.subprocess, 'run', parent_root)
    assert generator.committed_source() is None
    assert calls == [['git', 'rev-parse', '--show-toplevel']]


@pytest.mark.parametrize('source_index', range(4))
def test_dirty_producer_rejected_before_output_creation(generator, mocked_generation, monkeypatch, source_index):
    state = mocked_generation
    changed = state.checkout.root/generator.SOURCE_FILES[source_index]
    changed.write_bytes(changed.read_bytes()+b'dirty\n')
    arguments(monkeypatch, state.output)
    with pytest.raises(ValueError, match='producer source differs'):
        generator.main()
    assert not state.output.exists()
    assert state.calls == {'circuits': 0, 'checks': 0, 'exports': 0}


def test_late_payload_budget_failure_leaves_no_completed_marker(generator, mocked_generation, monkeypatch, capsys):
    state = mocked_generation
    arguments(monkeypatch, state.output)
    def fill_remaining(output):
        used = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
        with (output/'last-figure.svg').open('wb') as handle:
            handle.truncate(generator.MAX_OUTPUT_BYTES-RESERVATION-used-1)
    monkeypatch.setattr(generator, 'coefficient_figure', fill_remaining)
    with pytest.raises(RuntimeError, match='reservation'):
        generator.main()
    assert (state.output/'n1-s2-r1/measured-trial.qasm').is_file()
    assert (state.output/'n1-s2-r1/record.json').is_file()
    assert not (state.output/'manifest.json').exists()
    assert not (state.output/'.manifest-write.tmp').exists()
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('stage', ['fsync', 'replace'])
def test_atomic_manifest_failure_cleans_temporary_and_never_completes(generator, mocked_generation, monkeypatch, capsys, stage):
    state = mocked_generation
    arguments(monkeypatch, state.output)
    def fail(*args, **kwargs):
        raise OSError(f'{stage} failed')
    monkeypatch.setattr(generator.os, stage, fail)
    with pytest.raises(OSError, match=f'{stage} failed'):
        generator.main()
    assert (state.output/'n1-s2-r1/record.json').is_file()
    assert not (state.output/'manifest.json').exists()
    assert not (state.output/'.manifest-write.tmp').exists()
    assert capsys.readouterr().out == ''


def test_final_manifest_size_failure_never_writes_completion_marker(generator, mocked_generation, monkeypatch):
    state = mocked_generation
    arguments(monkeypatch, state.output)
    monkeypatch.setattr(generator, 'MAX_OUTPUT_BYTES', RESERVATION+2048)
    original = generator.json.dumps
    def enlarged_manifest(value, *args, **kwargs):
        if isinstance(value, dict) and value.get('schema') == 'openquantum-suzuki-study-v1':
            return 'x'*generator.MAX_OUTPUT_BYTES
        return original(value, *args, **kwargs)
    monkeypatch.setattr(generator.json, 'dumps', enlarged_manifest)
    with pytest.raises(RuntimeError, match='no completed manifest'):
        generator.main()
    assert not (state.output/'manifest.json').exists()
    assert not (state.output/'.manifest-write.tmp').exists()


def test_mocked_canonical_inventory_marker_binds_exact_36_cases(generator, mocked_generation, monkeypatch):
    state = mocked_generation
    arguments(monkeypatch, state.output, canonical=True)
    generator.main()
    manifest = json.loads((state.output/'manifest.json').read_text())
    expected = [f'n{width}-s{order}-r{repetitions}' for width in (1,2,3) for order in (2,4,6) for repetitions in (1,2,4,8)]
    assert manifest['case_ids'] == expected and manifest['canonical_case_count'] == 36
    assert manifest['status'] == 'generated_and_numerically_checked'
    assert manifest['source_commit'] == state.checkout.head
    assert manifest['source_sha256'] == {name: hashlib.sha256(body).hexdigest() for name, body in state.checkout.committed.items()}
    assert state.calls == {'circuits': 36, 'checks': 36, 'exports': 36}
    assert manifest['native_estimator_calls'] == manifest['factory_searches'] == manifest['gpu_jobs'] == 0
    assert manifest['payload_bytes'] == sum(record['bytes'] for record in manifest['files'].values())
    for name, record in manifest['files'].items():
        body = (state.output/name).read_bytes()
        assert record == {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    assert not (state.output/'.manifest-write.tmp').exists()
    # This test mocks the scientific functions. It establishes inventory and
    # completion protocol only; actual evolution is tested in the separate file.


def test_late_source_drift_rejected_before_atomic_completion(generator, mocked_generation, monkeypatch, capsys):
    state = mocked_generation
    arguments(monkeypatch, state.output)
    original = generator.coefficient_figure
    def drift(output):
        original(output)
        source = state.checkout.root/generator.SOURCE_FILES[0]
        source.write_bytes(source.read_bytes()+b'late source drift\n')
    monkeypatch.setattr(generator, 'coefficient_figure', drift)
    with pytest.raises(ValueError, match='source|producer'):
        generator.main()
    assert (state.output/'n1-s2-r1/record.json').exists()
    assert not (state.output/'manifest.json').exists()
    assert not (state.output/'.manifest-write.tmp').exists()
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('identity_change', ['HEAD', 'root'])
def test_late_checkout_identity_change_never_completes(generator, mocked_generation, monkeypatch, identity_change):
    state = mocked_generation
    arguments(monkeypatch, state.output)
    changed = {'active': False}
    original_git = generator.subprocess.run
    other_head = 'b'*40
    def shifted_identity(command, **kwargs):
        if changed['active']:
            if identity_change == 'root' and command == ['git', 'rev-parse', '--show-toplevel']:
                return subprocess.CompletedProcess(command, 0, str(state.checkout.root.parent)+'\n', '')
            if identity_change == 'HEAD' and command == ['git', 'rev-parse', 'HEAD']:
                return subprocess.CompletedProcess(command, 0, other_head+'\n', '')
            if identity_change == 'HEAD' and command[:2] == ['git', 'show'] and command[2].startswith(other_head+':'):
                name = command[2].split(':', 1)[1]
                return subprocess.CompletedProcess(command, 0, state.checkout.committed[name], b'')
        return original_git(command, **kwargs)
    monkeypatch.setattr(generator.subprocess, 'run', shifted_identity)
    original_figure = generator.coefficient_figure
    def shift(output):
        original_figure(output)
        changed['active'] = True
    monkeypatch.setattr(generator, 'coefficient_figure', shift)
    with pytest.raises(ValueError, match='exact checkout changed'):
        generator.main()
    assert not (state.output/'manifest.json').exists()
    assert not (state.output/'.manifest-write.tmp').exists()


def test_late_pinned_reference_change_never_completes(generator, mocked_generation, monkeypatch):
    state = mocked_generation
    arguments(monkeypatch, state.output)
    original = generator.coefficient_figure
    def drift(output):
        original(output)
        state.reference.write_bytes(b'changed reference\n')
    monkeypatch.setattr(generator, 'coefficient_figure', drift)
    with pytest.raises(ValueError, match='pinned reference source changed'):
        generator.main()
    assert not (state.output/'manifest.json').exists()
    assert not (state.output/'.manifest-write.tmp').exists()
