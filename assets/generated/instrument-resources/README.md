# Complete measured instruments: conditional physical models

This bundle computes damping-Z, lindblad-Z, thermal-Z, thermal-X. It contains actual Clifford+T
circuits, deterministic small density-matrix checks, explicit operation ledgers
and conditional physical workloads. The shot ensembles and hardware schedules
have **not** been executed. A guarded floating synthesis error is a numerical
proposal, **not** a formal interval or combined confidence certificate.

## Task and construction

The fixed damping-Z task measures the exact finite-time one-qubit absorbing
channel at rate 0.8 and time 1 from a gate-prepared plus state. Lindblad-Z uses
32 positive-time Strang steps by default with H=0.7X+0.3Z and fresh environment
resets. Its reference is the independent matrix-exponential Liouvillian answer.
The thermal X/Z tasks use the explicitly constructed degree-6 Taylor LCU
filter for H=0.6Z+0.4X at beta=1.25, Bell input preparation, actual PREPARE/SELECT
and uncomputation, label readout, reference discard and system readout. The
weighted energy output is 0.6 mean(Z)+0.4 mean(X), from separate groups.
See each report's instance field; --lindblad-steps changes the fixed step count.

Resets and measurements remain in the compiled schedule. SABRE uses a fixed
logical-line layout, seed 7 and optimization level 0. Actual ancilla-free
Ross--Selinger approximants replace every remaining rotation; the algorithm
uses Qiskit's gridsynth_unitary. Its native rsgridsynth backend exposes no seed
parameter. The generated sequences and SHA-256 identities, rather than a
promised seeded rerun, define the saved circuit. Per-rotation Frobenius errors
plus a 1e-12 numerical guard are summed. This is not formal roundoff analysis.
Exported QASM omits global phase and is a measured-trial export; it cannot be
substituted for a phase-preserving controlled oracle.

## Full attempted workload and error allocation

Hoeffding's inequality sets accepted sample counts for one +/-1 Pauli mean
with absolute statistical error 0.03. A fixed schedule runs every rejected
trial in full. Thermal postselection uses the numerical lower proposal
(sqrt(p_ideal)-eta)^2 and conditional mean-bias proposal 4 eta/sqrt(p_ideal).
The conditional budgets separately include statistical failure, acceptance
failure, physical faults and factory availability. X/Z group budgets add to
0.01 for the energy task. Reports include the ideal algorithm discrepancy,
actual synthesized answer and proposed synthesis bias. These estimates rely
on stated perturbation and physical assumptions; no formal success guarantee
or classical advantage is claimed.

## Space and time model

This is a **separate serialized instrument scenario**, not the frozen
Grover/PSSPC surface-code baseline or a co-design improvement. For n logical
source wires and odd distance d, reserve 4n-1 data/workspace/bridge patches,
each of area 2d^2-1. Peak qubits add one QDK native factory's qubits. At fixed
distance the reserved data space is linear in n, but materialized gate counts,
transport, preparation and success depend on the algorithm. This bounded
bundle does not demonstrate an asymptotic Gibbs or Lindblad complexity claim.

Each operation's ledger charges its serialized time, logical preparation,
readouts and controller decisions. Neighbor CNOTs use three protected Pauli
parity slots; nonlocal source CNOTs are routed to actual neighbor operations.
Each T uses native factory supply, capped complete factory retries, neighboring
SWAP transport, injection CNOT, Z readout, conditional S and port initialization.
T-dagger also has a charged inverse Clifford-frame decision. A factory port
residence term is added outside native supply time. QDK's output-quality model
already includes its quantum factory operations and is not charged again as
factory-memory error. The source-level 1% readiness requirement is separated
from output quality; exact decimal-rational arithmetic selects the smallest
retry cap in 1..16 covering every required state.

For M accepted slots and complete-trial cap r, A=M*r attempts are charged.
Total runtime is A times the sum of count(operation)*time(operation), and
data memory exposure uses (4n-1)*ceil(time/cycle) per attempt. Factory states
and all event failure contributions are multiplied by A. The 100 ns gates and
initializations, 500 ns measurements, 1000 ns code cycles, 10000 ns controller
reaction, controller error 1e-15/decision, separate initialization residual,
independent majority-of-disjoint-paths readout and memory-distance fit are
explicit scenarios, not calibrated hardware parameters. Data/factory output
faults are union-bounded; factory readiness has its own budget. Direct native
QDK Trace accounting is checked against the ledger for time and error.

Tracked Clifford frames, protected Pauli bases, T-axis orientation and
factory/data handoff are **conditional geometry/interface assumptions, not
a layout certificate**. This does not model arbitrary source-level if/else or
adaptive quantum feedback. It supports fixed schedules plus the stated reset,
T correction and factory/trial dispatch macros.

## Scope, limitations and reproduction

Source limits are five qubits and 10000 operations; actual synthesis permits
100000 operations. Synthesis allowances are between 1e-9 and 0.1, subject to
the per-gate numerical guard. Dense instrument checks are separately bounded
by that same five-qubit width. Distances are odd through 49 by default and at
most 61; no unbounded GPU job is used. The four small tasks are simulation
checks. The large reported sample/retry workloads are calculated, unexecuted.
Offline coefficient generation, synthesis and final statistical-report wall
time are outside quantum runtime; actual synthesis time is recorded separately.
Decoder throughput, interface certification, formal synthesis proofs and
arbitrary adaptive control remain unavailable.

In the private source checkout, with the locked Python 3.12 environment:

```bash
PYTHONPATH=src python scripts/generate_instrument_resources.py
PYTHONPATH=src python -m pytest tests/test_instrument_resources.py
```

For standalone use, download all six Python files in this directory, the
project requirements-lock.txt and LICENSE.txt into one directory, then run
python generate_instrument_resources.py. --case damping, lindblad or thermal
replaces this bundle's managed task records and summaries with that subset;
unrelated user files are preserved. Use a separate output directory to keep
multiple selections. The private factory cache must remain outside output.
manifest.json records package/Python/device details, optional verified-root
Git revision, exact source hashes and available before/after frozen-baseline
fingerprints. A no-Git download records null revision rather than an ancestor's
HEAD. Local paths, credentials and remote coordinates are not in the bundle.

## References and licenses

Original implementation and generated text are MIT-licensed; see LICENSE.txt.
references.bib credits Litinski (2019), Horsman et al. (2012), Ross and Selinger
(2014), Qiskit 2.6.2 (Apache-2.0), rsgridsynth 0.2.2 (MIT, as pinned by Qiskit's
Cargo.lock) and Microsoft QDK 1.34.0 (MIT). The installed wheel does not directly
certify its native crate version. QDK source semantics were audited at a
separate recorded commit; it is not claimed as the release commit. Exact source
URLs and reference keys are also stored in the physical reports.
