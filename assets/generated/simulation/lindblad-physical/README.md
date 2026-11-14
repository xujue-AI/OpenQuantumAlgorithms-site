# Literal n2/r69 Lindblad physical workload

Only the fixed two-system, four-wire Z0 task is implemented. All 15,756
complete fresh product-input trials retain 209 resets and four readouts each.
The repeated population and hardware schedules are calculated, unexecuted.
The n3/r134 and n4/r211 source ensemble checks explicitly reject at 256 branches;
their physical/synthesis results remain unimplemented and budgets reserved.

Source, routed and actual ancilla-free Ross–Selinger Clifford+T instruments
are checked with the existing four-wire density/CPTP path and independent
GKSL/gate-contracted references. Full-unitary lowering disables clean-input
assumptions. SABRE seed 7 and level 0 preserve all reset/readout boundaries.
Native synthesis exposes no seed: the saved sequences/hashes define its
realization. QASM omits overall phase and specifies measured trials, not a
phase-preserving controlled oracle.

The target is .01 splitting+.01 realization+.03 statistics=.05 absolute Z0
error. The actual guarded synthesis sum, requested allowance 5e-5, separate
representation reserve 1e-7 and SDK/control/routing reserve 1e-7 are distinct.
The reserves are conditional norm premises, not inferred from agreement;
no full representation/synthesis/hardware certificate is claimed. Coherent
bias is not a stochastic physical failure and is not multiplied by trials.

Family budgets remain statistics .005, physical .0025 and factory readiness .0025,
divided across three cases: 1/600, 1/1200, 1/1200 per case. Only n2 is physically
estimated; unused n3/n4 allocations are not reassigned. There is no algorithm
postselection or retry. Complete factory retries retain their own accounting.

Physical error 1e-3/1e-4 uses the frozen default controller/timing/interface
profile through the existing serialized QDK model. Startup, every ignored
readout/reset, idle/controller exposure, neighbor routing, transport/injection,
port recycling and factory retries are charged. Frozen Grover/PSSPC is unchanged.
Factory output quality is charged once; readiness is separate. Offline classical
preprocessing/reporting, layout certification, decoder throughput and hardware
calibration remain explicit exclusions/assumptions, not quantum runtime.

Caps: one four-wire circuit; 20,000 source instructions; 500,000 routed/synthesis/
density-validation instructions; preparation 600 s, each profile 180 s, family
diagnostic 60 s and parent 1200 s. Native rotation synthesis permits at most
3312 attempts, including retries. Each profile permits 24 top-level factory/
SurfaceCode queries and 1000 returned candidates per query, 24,000 total.
These count adapter invocations; internal native construction before the first
yield is bounded by the child timeout. No candidate subset is truncated.
Direct Trace calls are bounded by one per returned candidate. Managed JSON/
QASM is at most 32 MiB, private checkpoints 64 MiB. Timeout, cap overflow and
infeasibility remain explicit with no partial success.

Reproduce in the locked Python 3.12 environment:

    PYTHONPATH=src python scripts/generate_simulation_lindblad_physical.py
    PYTHONPATH=src python -m pytest tests/test_simulation_lindblad_physical.py

Standalone: put all six copied Python sources, requirements-lock.txt, LICENSE.txt,
references.bib and the notebook together. The default output is a new subfolder.
--reuse-prepared uses only runner-created private checkpoints keyed by exact
runtime source/package/import identities and checked against their saved hashes.
Cache and public output are disjoint; no checkpoint/log is distributed. Unknown
user files are preserved and omitted from manifests. Source/package/QASM hashes
identify the saved run; no-Git downloads record null revision. Original MIT code,
text and figures; primary algorithm and exact upstream package sources are
credited in references.bib.
