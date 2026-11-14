# Interacting-chain workloads at common ideal accuracy

This bundle materializes n=2, 3, 4 from the predeclared
family n=2, 3, 4. Numerical checks are performed.
It preserves the older two-spin r=32 energy study as a separate task. Here the
output is one **local Z0 mean**, not total energy or tomography.

## Fixed problem and output contract

Use hbar=1, time 1, H=0.6 sum ZZ+0.35 sum X on an open chain, local lowering
jumps with rate 0.8 at site 0 and 0.5 elsewhere, and a product input +i on site 0
and + on the others. Coefficients/rates have inverse-time units. Actual H/S
gates prepare the known input; all systems begin with resets. The measured
circuits read all 2n system/environment wires in Z, retaining only system 0's
score +/-1. Other system bits and every environment bit are ignored.

Each requested mean has splitting allowance 0.01 and statistical error 0.03.
The **three-output family** has statistical failure budget 0.005, split into
1/600 per case, giving 15,756 complete trials per case. A union bound controls
all three output errors simultaneously, each at 0.04 in the ideal exact-angle model.
Positive exponential Taylor
terms verify the Hoeffding count with exact rational arithmetic. The large
shot ensembles are calculated and **unexecuted**. Finite-angle roundoff,
coherent synthesis and physical faults do not have a completed confidence
certificate, so no hardware success guarantee is claimed.

## Original uniform step policy, before observing numerical errors

The literal global Strang palindrome contains ZZ/2, X/2, all local damping
halves except the last, the last damping full step, then the reverse halves.
Each component receives one full step's exposure and all durations are positive.
The dissipator convention is L rho L† - 0.5{L†L,rho}.

On Hermitian matrices with Schatten trace norm, Hamiltonian generators have
map norm at most twice the Hamiltonian operator norm, 2||H||_infinity;
each lowering dissipator has norm at most 2 gamma.
Thus A=2(n-1)|J|+2n|hx|+2 sum gamma; this is 5.2, 8.1, 11 for n=2, 3, 4. These are
generator-norm **upper bounds**, not vectorized matrix 1-norms, diamond-norm
measurements or the previous reference-exposure diagnostic.

F(s) denotes the reduced-system channel palindrome after environment discard,
not the joint dilation unitary with square-root small-time angles. Each
component has total weight one, giving F'(0)=G=sum L. Palindrome symmetry
gives F''(0)=G^2. For this positive-time product, each exponential is CPTP and contractive.
The product-rule multinomial expansion bounds ||F'''(s)|| by A^3. The exact
E(s)=exp(s sum L) is CPTP and has ||E'''(s)||<=A^3. The palindrome matches
F/E derivatives through degree 2 at zero. Two integral Taylor remainders give
local trace-norm map error <=A^3 h^3/3; CPTP telescoping across r steps gives
B=min(2,(At)^3/(3r^2)). State trace distance is bounded by B/2, and a unit-norm
Pauli mean bias by B. This is an all-input **exact-arithmetic analytic bound**
for this bounded time-independent positive-time schedule, excluding initial
preparation faults and floating gate representation. It is not a Lean proof
or a formal interval analysis of the implemented gates.

Fraction arithmetic selects the smallest r with B<=0.01, before any numerical
curve is inspected: r=69, 134, 211. The previous integer fails the analytic policy
in each case. The envelope is conservative; it is not optimal locality-aware
complexity. At fixed coefficients A=O(n), this policy gives r=O(n^(3/2) t^(3/2)
/sqrt(epsilon_split)). Gates per trial are O(nr), and complete-workload logical
counts multiply by O(epsilon_stat^(-2) log(3/delta_family)). Fixed-Z0 sampling
has constant score range; the old extensive energy task has different shots.

## Actual circuits and independent bounded classical checks

Quantum implementation uses the original two_spin_lindblad.py generator:
actual CX-RZ-CX and RX layers plus fresh environment reset, CRY and reverse CX.
No dense Hamiltonian, channel or exponential is appended as a quantum gate.
The literal damping halves are retained even when separate-site dissipators
commute. System/environment initialization, discard-before-reuse, final readout
and all input gates are charged. The lowered QASM uses rotation/CX primitives,
with optimization level 1 and transpiler seed 7. Compiler barriers preserve
reset/measurement boundaries. Their count is recorded separately from logical
primitive counts; this bundle contains **no physical timing estimate**.

The independent full GKSL reference uses a column-vectorized dense generator.
The actual reduced schedule is independently evaluated from matrix operators
of the actual small gate components. A fresh environment may be traced
immediately after its dilation: it is untouched until its next reset or ignored
final measurement, so this causal contraction preserves subsequent system
state and readout. Whole2n-wire Qiskit density checks verify this construction
on short schedules. The complete selected schedules are evaluated by powers
of the gate-derived reduced channel, not a full 2n-wire gate-by-gate simulation.
Supplied mixed matrices initialize only the classical verifier; quantum trials
use the declared gate-prepared product input.
Source/circuit correctness, selected-state error and the uniform analytic bound
are reported separately. Small numerical agreement is not the proof of B.

Dense system states are at most 16x16, channels/Liouvillians at most 256x256;
whole-circuit densities are at most 256x256 with four-step checks. Numerical
convergence uses at most five fixed step points per case, with no adaptive
selection. Source generation is capped at 1024 steps and the existing 50,000
primitive-operation bound. This wrapper accepts only 2/3/4 systems. No GPU
or unbounded job is used.

## Missing physical layer

These are **logical, ideal-angle workloads**. The conditional instrument
backend defaults to five source wires and also provides opt-in weighted-Kraus
checks and QSP/thermal resource reports for other circuits. This wrapper uses
no opt-in validation of the six/eight-wire chain workloads and their long
repeated-reset schedules. The width fields below describe the default five-wire
configuration; a width-only check does not supply a chain physical report.
The n=2 width fits the default, but no case in this complete workload has a
reproduced Clifford+T/physical report. No distance, factory count, physical qubit number or
elapsed hardware time is extrapolated. A supported extension needs actual
rotation synthesis and a whole-workload error allocation, reset-aware routing,
checked wider instrument semantics, and a declared data/factory layout and
controller/reset/readout/idle-fault schedule. Generic adapter presence or
changing a width constant would not provide that evidence.

The logical ledger includes all complete trials and score/counter updates,
but excludes offline angle/circuit construction and final statistical wall time,
coherent synthesis, physical routing, factory transport, QEC and physical
initialization/reset/readout/controller faults and times. CPU build/reference
times are recorded separately. Qubit-layer volume is a dependency proxy,
not physical qubit-seconds. No matched co-design or advantage claim is made;
available frozen Grover fingerprints are unchanged.

## Reproduce, provenance and credit

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/generate_simulation_lindblad_workload.py
PYTHONPATH=src python -m pytest tests/test_simulation_lindblad_workload.py
```

For independent use, put lindblad_workload.py, two_spin_lindblad.py,
open_systems.py and the generator together with requirements-lock.txt and
LICENSE.txt. The optional source_provenance.py records a Git revision only
for a verified exact-root checkout; no-Git downloads retain source hashes and
null revision. Partial --sizes runs retain the full-family shot allocation and
remove only stale managed case records, preserving unrelated files. --no-sim
requires separate output and marks numerical checks unperformed.

Original code, diagrams, explanation and bound derivation are MIT work. Pinned
Qiskit 2.6.2 is Apache-2.0; NumPy 2.5.4 and SciPy 1.18.1 use BSD licenses. Exact
versions and source hashes are in manifest.json; upstream references are in
references.bib. Lindblad/GKS 1976, Strang 1968 and the Kliesch 2011/2012 erratum
provide context. No commutator constants are imported from that paper and no
survey wording or figures are copied. Formal/Lean coverage remains absent.
