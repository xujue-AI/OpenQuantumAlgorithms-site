# Literal n4/r211 correlated Lindblad workload

The fixed four-system, eight-wire Z0 task keeps t1, J0.6, hx0.35,
rates0.8/0.5/0.5/0.5 and +i/+/+/+ input. Every one of 15,756 complete
fresh trials retains1,481 resets and eight readouts. Repeated populations and
hardware schedules are calculated and unexecuted. Prior n2/n3 bundles and
all historical n4 rejection labels are preserved; no allocation is reassigned.

source-preflight.json and the source_preflight field in results.json retain
completed raw-source/routed-rotation evidence even if native preparation later
fails. Full Clifford+T and physical records exist only after their own completed
stages; missing/cap/timeout/numerical/unsupported/infeasible statuses remain
explicit. A parent watchdog failure discards all CT/physical success and can
retain only this invocation's hash-verified source evidence and snapshots.

The new density API requires explicit EightWireDensityLimits and preserves the
whole 256x256 complex128 state (1 MiB), all 16 four-system probabilities and the
full 16x16 reduced density. Reset sums both unobserved Kraus terms; ignored
readouts dephase. No reset-history bits, trajectory pruning or product-of-
marginals replacement is used. Classical local-matrix batching does not remove
quantum occurrences. Once-at-output numerical trace normalization and raw trace
are disclosed; they are not algorithm postselection. The predeclared 1e-8 state
trace guard and 1e-10 density-comparison guard are nominal numerical checks,
not representation/compiler/hardware certificates.

A separate fixed-task compiler API requires explicit source 20k/routed 100k/
Clifford+T 2m limits. Existing defaults, the six-wire density ceiling, 256-branch
ensemble ceiling and shared 1m materialization ceiling remain unchanged. The
actual completed CT count is separately tested against the old 1m guard; this
validation is not an executed legacy synthesis attempt. Full-unitary level0
lowering, fixed layout, bidirectional line and SABRE seed 7 preserve every source
reset/readout. Actual Ross--Selinger sequences receive the unchanged whole-
trial 5e-5 request; repeated targets retain every emitted gate. Native synthesis
has no seed API. Complete raw/routed/CT QASM hashes identify realizations;
measured-instrument QASM omits overall phase and is not a controlled oracle.
In-memory phases are recorded for coherent unitary-segment reuse.

Accuracy remains .01 splitting/.01 realization/.03 statistics, target .05.
Representation and SDK/control/routing norm premises remain separately 1e-7
each. Under those premises, unconditional Z0 bias is at most 2 eta; deterministic bias is
not a stochastic fault and is not multiplied by 15,756. Family allocations are
statistics .005/physical .0025/readiness .0025, divided into 1/600, 1/1200, 1/1200
per case. No better controller, physical timing, geometry or allocation is
substituted. Physical p1e-3/p1e-4 and controller 1e-15 are frozen. Actual event
and T-wire counts determine controller/factory-retry diagnostics. An excessive
controller term or unsupported native materialization domain has no fabricated
qubits/runtime. Other fault components remain unevaluated when early rejection
prevents native search. The serialized model includes startup, resets, ignored
readouts, dispatch, active/idle exposure, neighbor routing, transport/injection,
port recycling, handoff and complete capped pipeline retries when evaluated.
Output quality and readiness budgets stay separate; frozen Grover/PSSPC is
unchanged. These engineering-model terms are not measured hardware-failure
lower bounds. Offline preprocessing/reporting, layout/decoder certification
and calibration remain exclusions; no advantage is claimed.

Caps: source preflight and historical diagnostics share 180s, native preparation
1800s, each physical profile 180s and hard parent 2400s through artifact export.
Native attempts 23,632; odd distances 3..49; factory retries <=16. Each profile
permits 24 top-level factory/24 SurfaceCode queries, 1000 candidates/query and 24k
total. Internal native work before yield is timeout-bounded; no candidate
subset is truncated. Direct Trace calls are at most returned candidates.
JSON/QASM 32 MiB each, private prepared pickle 256 MiB with atomic bounded writes
and streaming hashes. Infeasible/failed stages never retain stale success.

Reproduce with the locked Python 3.12 environment:

    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/generate_simulation_lindblad_four.py

Standalone: place all nine Python snapshots, requirements-lock.txt, LICENSE.txt,
references.bib and the notebook together. Default output is a new subfolder.
The runner disables bytecode writes and uses a fresh task-private prefix if the
caller did not supply one; every child uses a fresh private bytecode prefix.
Actual imported module locations and source hashes are verified against the
own snapshot; no paths, logs, checkpoints or credentials are distributed.
--reuse-prepared accepts only exact source/import/schema/native-policy/cap-
bound checkpoints with matching hashes. Source preflight is fresh per invocation
and additionally bound to its invocation ID and both exported QASM hashes.
Private cache and public output are disjoint. Unknown user files are preserved
and omitted from the managed manifest.

artifact-inventory.csv/json itemize local logical bytes, SHA256 and categories.
Recursive inventory/manifest metadata and volatile progress are excluded there;
the manifest attests the inventory files. Baseline 7,159 files/358,968,120 bytes
is local tracked generated storage at worktree creation, not public hosting
occupancy. All originals remain; publication layout is coordinator-owned.
Text/circuits/figures are original MIT work; exact primary package/paper credits
are in references.bib. No main/shared-export/workflow/publication change occurs.
