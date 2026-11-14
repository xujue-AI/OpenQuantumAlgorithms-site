#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate bounded qDRIFT demos, independent references and complete ledgers.

Default: two noncommuting one/two-spin cases, 32 fresh trials each, 8 exponentials
per trial. No Native estimator, phase synthesis, factory search or remote job.
Use --qubits N --no-sim for a separately guarded larger circuit export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import qasm3

from openquantum.simulation.qdrift import (
    Hamiltonian, MAX_DENSE_QUBITS, MAX_EXPONENTIALS, MAX_GENERATOR_QUBITS,
    MAX_PRIMITIVE_GATES, channel_comparison, chain_hamiltonian, empirical_study,
    integer, rational, logical_ledger, qdrift_circuit, sample_schedule,
)

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_STUDY_TRAJECTORIES = 64
SOURCE_FILES = ('src/openquantum/simulation/qdrift.py', 'scripts/generate_simulation_qdrift.py', 'requirements-lock.txt', 'LICENSE')
REFERENCE_DOI = '10.1103/PhysRevLett.123.070503'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_commit():
    top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, capture_output=True, text=True, timeout=5)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT.resolve():
        # A downloaded package nested inside another checkout is not that
        # checkout's source, even when its producer bytes happen to match.
        return None
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    for name in SOURCE_FILES:
        blob = subprocess.run(['git', 'show', f'{commit}:{name}'], cwd=ROOT, capture_output=True, timeout=5)
        if blob.returncode != 0 or sha(blob.stdout) != sha((ROOT / name).read_bytes()):
            raise ValueError('producer source differs from recorded commit; commit the source before generation')
    return commit


def guard_output(path):
    path = Path(path).absolute()
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError('output must not use a redirected path')
        current = current.parent
    destination = path.resolve()
    generated = (ROOT / 'assets/generated').resolve()
    own = (generated / 'simulation/qdrift').resolve()
    if generated == destination or generated in destination.parents:
        if destination != own and own not in destination.parents:
            raise ValueError('qDRIFT must not overwrite another algorithm artifact tree')
    for relative in ('src', 'scripts', 'tests', 'notebooks', 'algorithms', 'references', '.git'):
        protected = (ROOT / relative).resolve()
        if destination == protected or protected in destination.parents:
            raise ValueError('output must not overwrite source or private repository data')
    if destination == ROOT.resolve():
        raise ValueError('output must be a dedicated artifact directory')
    return destination


def normalized_svg(path):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def export_case(output, case_id, spec, simulation_time, steps, seed, trajectories, initial_basis, simulate):
    case = output / case_id
    case.mkdir()
    if simulate:
        study = empirical_study(spec, simulation_time, steps, trajectories, seed, initial_basis=initial_basis)
        first_seed = study['records'][0]['trajectory_seed']
        ledgers = [record['logical_ledger'] for record in study['records']]
    else:
        first_seed = seed
        ledger = logical_ledger(spec, simulation_time, steps, seed, initial_basis=initial_basis)
        ledgers = [ledger]
        study = {'hamiltonian': spec.as_dict(), 'time_exact': str(simulation_time), 'steps': steps,
                 'initial_basis': initial_basis, 'trajectory_seed': seed,
                 'validation': {'performed': False, 'reason': '--no-sim; scalable construction is separate from bounded dense checks'},
                 'physical_resource_estimate': None}
    for index, ledger in enumerate(ledgers):
        selected = tuple(ledger['schedule_indices'])
        circuit = qdrift_circuit(spec, simulation_time, steps, ledger['trajectory_seed'], initial_basis=initial_basis, measure=True, schedule=selected)
        text = qasm3.dumps(circuit)
        name = f'trial-{index:03d}.qasm'
        (case / name).write_text(text)
        ledger['measured_qasm_file'] = name
        ledger['measured_qasm_sha256'] = sha(text.encode())
    study['logical_ledgers'] = ledgers
    if simulate:
        study['validation'] = {'performed': True, 'trajectories_checked': trajectories,
                               'full_channel_spanning_tests': (2**spec.qubits)**2,
                               'numerical_tolerance': 1e-11, 'formal_certificate': False}
        rows = []
        for count in (1, 2, 4, 8, 16, 32, 64, 128):
            if count <= MAX_EXPONENTIALS:
                row = channel_comparison(spec, simulation_time, count, initial_basis=initial_basis)
                rows.append({k: row[k] for k in ('steps', 'hs_induced_superoperator_error', 'normalized_choi_trace_distance', 'fixed_input_trace_distance', 'observable_bias')})
        with (case / 'channel-convergence.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
        for metric, label in (('hs_induced_superoperator_error', 'HS-induced superoperator error'), ('fixed_input_trace_distance', 'Fixed-input trace distance'), ('observable_bias', 'Z0 bias')):
            axes[0].loglog([r['steps'] for r in rows], [max(r[metric], 1e-17) for r in rows], 'o-', label=label)
        axes[0].set(xlabel='Term exponentials per trajectory', ylabel='Numerical error'); axes[0].grid(alpha=.18); axes[0].legend(fontsize=8)
        values = [record['circuit_z0_expectation'] for record in study['records']]
        axes[1].plot(range(len(values)), values, '.', color='#346b8a', label='Individual trajectory mean')
        axes[1].axhline(study['ensemble_reference']['ensemble_observable'], color='#be6b35', label='Ideal mixture mean')
        axes[1].axhline(study['sample_mean'], color='#42765c', linestyle='--', label='One readout per trajectory')
        axes[1].set(xlabel='Saved independent-seed trajectory', ylabel='Z0 expectation / sample mean', ylim=(-1.05, 1.05)); axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(case / 'channel-and-samples.svg'); plt.close(fig)
        normalized_svg(case / 'channel-and-samples.svg')
        if spec.qubits <= 2:
            drawing = qdrift_circuit(spec, simulation_time, steps, first_seed, initial_basis=initial_basis, measure=True)
            fig = drawing.draw(output='mpl', fold=35, scale=.8)
            fig.savefig(case / 'first-complete-trial.svg', bbox_inches='tight'); plt.close(fig)
            normalized_svg(case / 'first-complete-trial.svg')
    dump(case / 'study.json', study)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', type=int)
    parser.add_argument('--time', default='1/2')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--trajectories', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--initial-basis', type=int, default=0)
    parser.add_argument('--coupling', default='1')
    parser.add_argument('--x-field', default='7/10')
    parser.add_argument('--z-field', default='0')
    parser.add_argument('--no-sim', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets/generated/simulation/qdrift')
    args = parser.parse_args()
    integer(args.trajectories, 'trajectories', 1, MAX_STUDY_TRAJECTORIES)
    args.time = str(rational(args.time, 'time'))
    if args.qubits is None:
        if (args.coupling, args.x_field, args.z_field) != ('1', '7/10', '0'):
            parser.error('custom chain coefficients require --qubits')
        if args.no_sim:
            parser.error('--no-sim requires an explicit --qubits construction request')
        cases = [('one-spin', Hamiltonian.from_terms(1, [('7/10', 'X'), ('3/10', 'Z')]), args.seed),
                 ('two-spin', chain_hamiltonian(2), args.seed + 1)]
    else:
        cases = [(f'chain-n{args.qubits}', chain_hamiltonian(args.qubits, args.coupling, args.x_field, args.z_field), args.seed)]
    if not args.no_sim and args.trajectories * len(cases) > MAX_STUDY_TRAJECTORIES:
        parser.error('the total recorded study is capped at 64 independently seeded trajectories')
    if not args.no_sim and any(spec.qubits > MAX_DENSE_QUBITS for _, spec, _ in cases):
        parser.error('dense checks stop at three qubits; use --no-sim for larger circuit export')
    # Validate all circuit requests before the output directory is created.
    for _, spec, seed in cases:
        logical_ledger(spec, args.time, args.steps, seed, initial_basis=args.initial_basis)
    commit = source_commit()
    output = guard_output(args.output_dir)
    if output.exists():
        raise ValueError('output must be absent; retain prior runs and use a new output directory')
    output.mkdir(parents=True)
    start = time.perf_counter()
    summaries = []
    for case_id, spec, seed in cases:
        study = export_case(output, case_id, spec, args.time, args.steps, seed, args.trajectories, args.initial_basis, not args.no_sim)
        if not args.no_sim:
            summaries.append({'case': case_id, 'qubits': spec.qubits, 'lambda_exact': str(spec.l1_norm), 'steps': args.steps,
                              'trajectories': args.trajectories, 'ensemble_Z0': study['ensemble_reference']['ensemble_observable'],
                              'exact_Z0': study['ensemble_reference']['exact_observable'], 'ensemble_bias': study['ensemble_reference']['observable_bias'],
                              'sample_mean': study['sample_mean'], 'hoeffding_radius': study['ideal_iid_hoeffding']['absolute_radius'],
                              'hs_induced_superoperator_error': study['ensemble_reference']['hs_induced_superoperator_error']})
    for relative in SOURCE_FILES[:2]:
        shutil.copyfile(ROOT / relative, output / Path(relative).name)
    shutil.copyfile(ROOT / 'LICENSE', output / 'LICENSE.txt')
    elapsed = time.perf_counter() - start
    files = {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(output.rglob('*')) if p.is_file()}
    size = sum(row['bytes'] for row in files.values())
    if size > MAX_OUTPUT_BYTES:
        raise RuntimeError('generated output exceeds 64 MiB; no completed manifest written')
    manifest = {
        'schema': 'openquantum-qdrift-study-v1', 'status': 'generated_and_ideal_checked' if not args.no_sim else 'generated_only_no_simulation',
        'reference_doi': REFERENCE_DOI,
        'source_commit': commit, 'git_producer_bindings_checked': commit is not None, 'source_sha256': {name: sha((ROOT / name).read_bytes()) for name in SOURCE_FILES},
        'source_scope': 'Exact checkout-root commit when Git top-level equals this package root; null for detached downloads. Content hashes identify actual producer files. Generated notebook/page output is a later artifact.',
        'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ('qiskit', 'numpy', 'scipy', 'matplotlib')},
        'device': 'CPU', 'seed': args.seed, 'cases': [name for name, _, _ in cases], 'time_exact': args.time,
        'steps': args.steps, 'independent_seed_trajectories_recorded': 0 if args.no_sim else args.trajectories * len(cases),
        'scientific_wall_seconds': elapsed,
        'generation_timing_scope': 'After request preflight and output-directory creation; includes circuit/channel checks, exports, figures and source copies; excludes interpreter imports, preflight and final metadata serialization.', 'native_estimator_calls': 0, 'factory_searches': 0, 'gpu_jobs': 0,
        'physical_resource_estimate': None,
        'command': {'script': 'scripts/generate_simulation_qdrift.py', 'qubits': args.qubits, 'time_exact': args.time,
                    'steps': args.steps, 'trajectories_per_case': args.trajectories, 'seed': args.seed,
                    'initial_basis': args.initial_basis, 'coupling': args.coupling, 'x_field': args.x_field,
                    'z_field': args.z_field, 'no_sim': args.no_sim, 'output_path': 'omitted; dedicated artifact directory'},
        'generator_bounds': {'qubits': MAX_GENERATOR_QUBITS, 'term_exponentials': MAX_EXPONENTIALS, 'primitive_operations': MAX_PRIMITIVE_GATES},
        'simulation_bounds': {'qubits': MAX_DENSE_QUBITS, 'total_study_trajectories': MAX_STUDY_TRAJECTORIES},
        'files': files, 'payload_bytes': size,
        'trust_boundary': 'Original finite circuit/channel numerical checks and seeded classical readouts. No physical device, formal proof, interval certificate, Native resource estimate or demonstrated advantage.'}
    if summaries:
        with (output / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(summaries)
        lines = ['| Case | lambda | N | Fresh trials | Exact Z0 | Ensemble Z0 | Bias | Sample mean |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        lines += [f"| {r['case']} | {r['lambda_exact']} | {r['steps']} | {r['trajectories']} | {r['exact_Z0']:.9f} | {r['ensemble_Z0']:.9f} | {r['ensemble_bias']:.3g} | {r['sample_mean']:.4f} |" for r in summaries]
        (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    # The inventory includes every final public payload, except its own manifest.
    final_files = {str(p.relative_to(output)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    manifest['files'] = final_files; manifest['payload_bytes'] = sum(row['bytes'] for row in final_files.values())
    encoded = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
    if manifest['payload_bytes'] + len(encoded) > MAX_OUTPUT_BYTES:
        raise RuntimeError('final output exceeds 64 MiB; no completed manifest written')
    temporary = output / '.manifest-write.tmp'
    try:
        with temporary.open('xb') as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output / 'manifest.json')
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({'status': manifest['status'], 'cases': len(cases), 'trajectories': manifest['independent_seed_trajectories_recorded'], 'wall_seconds': elapsed, 'payload_bytes': manifest['payload_bytes']}, indent=2))


if __name__ == '__main__':
    main()
