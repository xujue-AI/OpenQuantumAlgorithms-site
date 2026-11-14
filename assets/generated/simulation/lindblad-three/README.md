# Literal n3/r134 Lindblad physical workload

The fixed three-system, six-wire Z0 task is implemented here. All 15,756
complete fresh +i/+/+ input trials retain 673 resets and six readouts each.
The repeated population and hardware schedules are calculated, unexecuted.
The prior n2/r69 bundle remains unchanged and is referenced without rerunning.
The n4/r211 source ensemble check still rejects at its 256-branch cap;
its eight-wire density check also rejects the explicit six-wire ceiling.
Its synthesis/physical work remains unimplemented and allocation reserved.

Raw source, actual full routed rotations and actual ancilla-free Ross–Selinger
Clifford+T circuits are exported separately. An explicit opt-in six-wire
full-density CPTP checker retains all three system correlations and the full
joint output law, comparing with independent GKSL and gate-derived references.
Verifier-only local-matrix batching does not remove gates from the exported
circuits or resource counts. Resets are unobserved Kraus sums; reset histories
are not supplied as measured classical outputs. No postselection is applied.

Full-unitary lowering disables clean-input assumptions. SABRE seed 7 and level 0
preserve all reset/readout boundaries. Native synthesis exposes no seed: saved
sequences/hashes define its realization. Each QASM omits overall phase and
specifies complete measured trials, not a phase-preserving controlled oracle;
all three in-memory global phases are recorded in results.json.

The target is .01 splitting+.01 realization+.03 statistics=.05 absolute Z0
error. Actual guarded synthesis error, requested allowance 5e-5, representation
reserve 1e-7 and SDK/control/routing reserve 1e-7 remain separate. The reserves
are conditional norm premises, not inferred from density agreement; no full
representation/synthesis/hardware certificate is claimed. Deterministic bias
is not a stochastic physical fault and is not multiplied by the trial count.

Family budgets remain statistics .005, physical .0025 and factory readiness
.0025, divided across three cases: 1/600, 1/1200 and 1/1200 per case. Unused
allocations are not reassigned. There is no algorithm postselection or retry.
Complete factory retries retain their own accounting.

Physical error 1e-3/1e-4 uses the frozen default controller 1e-15 and unchanged
timing/interface profile through the serialized QDK model. Default-controller
infeasibility, if found, is retained explicitly with no fabricated space/time
point or substituted controller scenario. Startup, ignored readouts/resets,
idle/controller exposure, neighbor routing, transport/injection, port recycling
and full factory retries enter the model whenever a profile is evaluated.
The initial distance-independent controller diagnostic can reject a profile
before native factory search; other physical contributions then remain
unevaluated. Frozen Grover/PSSPC remains unchanged. Factory output quality is
charged once; readiness is separate. Offline classical preprocessing/reporting,
layout certification, decoder throughput and hardware calibration remain
explicit exclusions/assumptions, not quantum runtime.

Caps: one six-wire circuit; 20,000 source instructions; 1,000,000 routed,
synthesized and density-validation instructions; preparation/checking 1200 s,
each physical profile 180 s, family diagnostic 60 s and hard parent 1800 s.
Native rotation synthesis permits at most 10,720 attempts, including retries.
Each profile permits 24 top-level factory and 24 SurfaceCode queries, 1000
returned candidates per query and 24,000 total. These count adapter calls;
internal native construction before the first yield is bounded by the child
timeout. No candidate subset is truncated. Direct Trace calls are bounded by
one per returned candidate. Managed JSON/QASM is at most 32 MiB each; private
checkpoints at most 128 MiB. Timeout, overflow, errors and infeasibility remain
explicit with no stale or partial success.

Reproduce in the locked Python 3.12 environment:

    PYTHONPATH=src python scripts/generate_simulation_lindblad_three.py
    PYTHONPATH=src python -m pytest tests/test_simulation_lindblad_three_physical.py tests/test_simulation_bounded_density_instrument.py tests/test_simulation_lindblad_three_runner.py

Standalone: put all seven copied Python sources, requirements-lock.txt,
LICENSE.txt, references.bib and the notebook together. Default output is a new
subfolder. --reuse-prepared uses only runner-created private checkpoints keyed
by exact preparation source, native guard/caps, package versions, schema and
package/standalone import namespace, then checked against their saved hashes.
The default private cache is a task-specific subfolder of Python's canonical
temporary directory. Cache and public output are disjoint; no checkpoint/log
is distributed. Unknown user files are preserved and omitted from manifests.

artifact-inventory.csv/json list each managed payload file, category, logical
bytes and SHA-256. Recursive inventory/manifest metadata and volatile progress
are explicitly excluded there; the manifest attests inventory files. Baseline
asset totals are local measurements from worktree creation, not public-hosting
occupancy. Originals are preserved and publication layout remains undecided.
Source/package/QASM hashes identify the run; no-Git downloads record null
revision. Original MIT code, text and figures; primary algorithms and exact
upstream package sources are credited in references.bib.
