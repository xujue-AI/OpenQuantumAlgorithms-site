---
title: "Detailed-balance quantum Gibbs sampling"
description: "Executed finite zero-correction channel, weighted-decay access and one-spin heralded correction step, with remaining general Gibbs-sampler requirements."
---

:::{note} Implementation status: partial

Original Gaussian filtered-jump access, a restricted finite weak reset
channel, finite weighted-decay block access and a one-spin correction block
with one heralded known-input step have executed small checks.
The general noncommuting correction,
uniform Gibbs-accuracy/mixing guarantees, rotation synthesis, physical
resources and mechanized proof remain planned. The
[existing Gibbs examples](gibbs-preparation.md) retain their separate loader
and heralded-filter scope.
:::

## Motivation and intuition

A Gibbs sampler drives a system toward the mixed state $\rho_\beta=e^{-\beta H}/\operatorname{Tr}(e^{-\beta H})$.

Frequency filters weight energy-changing proposals, while the quantum
generator's detailed balance fixes the thermal stationary state. A fresh
environment records the proposed transition and is then discarded/reset;
this turns a unitary system–environment circuit into a channel on the system.
Stationarity and the number of rounds needed to approach it are separate
questions. Our finite example makes both the transition construction and
its residual Gibbs error visible.

## Fundamental method and complexity

The 2025 *Efficient quantum thermal simulation* construction uses Gaussian operator-Fourier filters and a corrective coherent term to obtain a synthetic Lindbladian satisfying exact KMS detailed balance. Under the paper's normalized jump-access condition $\|\sum_a(A^a)^2\|\leq1$, evolution for time $t\geq1$ uses signed and controlled Hamiltonian simulation with total time $\widetilde O(\beta t)$; preparation still requires the problem's mixing time {cite:p}`frontier_thermal_chen_2025`.

Exact stationarity of the mathematical generator and accuracy of a finite digital circuit are separate statements. Discretizing filters, truncating local patches, synthesizing rotations and simulating the channel consume their own error budget. Unique stationarity does not establish rapid convergence. The high-temperature results specify regimes in which mixing is provably efficient {cite:p}`frontier_thermal_universal_2026,frontier_thermal_optimal_2026`.

## Important finite-jump variant

The finite-jump KMS family provides a second implementation choice. Its LCU-access and filter assumptions must be recorded alongside any simpler jump inventory; fewer jump labels alone are not a physical-resource comparison {cite:p}`frontier_thermal_kms_2025`.

### Planned local-circuit comparison

Hahn and colleagues provide another implementation route: truncate the
quasilocal KMS generator to overlapping Hamiltonian patches, sample
short-time local channels, and compile their dilations into local circuits
{cite:p}`frontier_thermal_local_circuits_2026`. Their high-temperature results
assume local finite-range lattice Hamiltonians and the stated jump-generator
conditions. Approximating the full Gibbs state and estimating a local
observable have different truncation-radius requirements; a fixed small
patch does not establish global accuracy at every system size.

This comparison remains **planned**. A local implementation must retain the
coherent correction,
all dilation/discard/reset operations, randomized-channel sampling and
readout. Its ledger must also include patch construction, circuit compilation
and any variational-fit error, then allocate truncation, digital-evolution,
synthesis, sampling and physical faults separately. The paper's local
compilation and numerical results have not been reproduced here, and no
physical improvement over the frozen surface-code baseline is claimed.

## Executed finite zero-correction channel

The first endpoint uses $H=Z$ and normalized Pauli proposals $X/\sqrt2$ and
$Z/\sqrt2$. More generally, the source accepts diagonal I/Z Pauli Hamiltonians
and Pauli proposals. Each filtered block has a fixed computational-basis
permutation multiplied by diagonal coefficients. Therefore the decay operator
is diagonal, commutes with $H$, and has exactly zero spectral coherent
correction. Noncommuting inputs are rejected by this endpoint; a dense
correction matrix is never appended as a gate.

The circuit applies the actual Gaussian access $U$, rotates a fresh weak flag
by $2\arcsin\sqrt{\delta\eta\gamma_l}$ under its frequency register, and applies
the full $U^\dagger$ only when the flag is zero. The mesh normalization
$\eta$ is retained, with $\delta\eta\max_l\gamma_l\leq1$. Every proposal,
frequency and flag outcome contributes to the reduced channel. Environmental
reset discards the labels; there is no accepted-only branch or algorithm
success retry. The controlled inverse reverses the actual Gaussian PREPARE,
signed product-formula evolution, Pauli SELECT and negative DFT, preserving
their phases and charging every added primitive control
{cite:p}`frontier_thermal_chen_2025`.

The independent reference retains **all** non-jump Kraus operators
$\langle e|U^\dagger\sqrt{I-\delta\eta\Gamma}U|0\rangle$ and all jump operators
$\sqrt{\delta\eta\gamma_e}B_e$. Nonzero environment labels in the non-jump
branch contribute second-order channel terms. Keeping only the zero label
would change the implemented channel.

(gibbs-channel-circuit)=

### Small circuit and independent checks

The [executed notebook](../../notebooks/simulation_detailed_balance_channel.ipynb)
draws a minimal three-wire **source illustration** with one X proposal and
one clock bit. Its quantitative channel example uses X/Z proposals and two
clock bits, with five wires. The full
[five-wire dilation QASM](../../assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.125-dilation.qasm),
[reset-round QASM](../../assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.125-reset-round.qasm)
and [eight-round preparation QASM](../../assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.125-preparation.qasm)
expose every primitive, initialization, reset and final readout.

```python
from openquantum.simulation.detailed_balance_channel import build_weak_round

circuit = build_weak_round(
    ((1.0, "Z"),), ("X", "Z"), clock_bits=2, beta=1.0, delta=0.125
)
circuit.draw("text", fold=80)
```

(gibbs-channel-checked-records)=

The retained CPU grid has eleven cases: clock sizes $M=4,8,16$ and
$\delta=1/16,1/8,1/4$ at $\beta=1$, plus $M=8$, $\delta=1/8$ at
$\beta=1/2,2$. It exports 33 actual dilation/reset/preparation circuits.
All 100 focused tests and the five notebook code cells passed. The
[scientific records](../../assets/generated/simulation/detailed-balance-channel/results.json)
keep actual gate channels, independent finite references and the analytic
continuum generator distinct.

| Small-check quantity | Retained result |
|---|---:|
| Maximum actual versus independent Kraus entry error | $3.97\times10^{-14}$ |
| Maximum full-channel Frobenius error | $1.55\times10^{-13}$ |
| Maximum trace-preservation entry error | $8.03\times10^{-14}$ |
| Minimum small Choi eigenvalue | $0.00145457$ |
| Maximum continuum-generator Gibbs residual | $5.56\times10^{-17}$ |
| Finite stationary-state Gibbs trace-distance range | $0.000734$–$0.051333$ |
| Largest recorded 64-round Gibbs trace distance | $0.143441$ |

(gibbs-channel-output-contracts)=

These last two error quantities have different meanings. The stationary bias
comes from the finite mesh; 64 rounds can also retain initial-state error.
The 64-round results are powers of the fully checked reduced one-round
channel on four spanning inputs, **not** density simulation of the long gate
stream. A separate actual two-round density-circuit test checks reset and
composition. Bloch contraction, $0.86204$–$0.96624$ across this grid, is an
uncertified floating-point diagnostic with stationary-residual accumulation.
It is not a general mixing certificate. The data establish no uniform
$0.05$ Gibbs-preparation claim.

The scalable generator also emitted a twelve-spin, seventeen-wire source
with 4,917 primitives for one preparation round. No large state or channel
was simulated. Its endpoint X proposals do not connect all twelve-spin basis
states, so that illustration claims neither general mixing nor thermal
preparation. Dense channel references stop at one system wire and seven
total wires; source materialization has its own 100,000-operation guard.

(gibbs-decay-block)=

## Executed finite weighted-decay block

A separate primitive unitary exposes the finite weighted decay needed by a
future coherent-correction circuit. For the **actual** Gaussian-access blocks
$B_{a,l}$, define
\[
D_{\rm fin}=\eta\sum_{a,l}\gamma_l B_{a,l}^{\dagger}B_{a,l},
\qquad \alpha=\eta\max_l\gamma_l.
\]
These blocks retain the ordered product-formula evolution. The independent
reference constructs its own Pauli products and signed Fourier sum;
exact-H finite targets are reported separately. This original access
construction uses the paper's weighted decay and spectral-correction
specification {cite:p}`frontier_thermal_chen_2025`.

With a weight wire $w$, set $V=R_wU$, using frequency-controlled RY angles
$2\arcsin\sqrt{\gamma_l/\max\gamma}$. The chronological circuit is
$H_s;V;CZ_{s,w};P_s(\pi);V^\dagger;H_s$. Both access copies are unconditional,
and the complete inverse preserves their phases. The selected block equals
$D_{\rm fin}/\alpha$ only when **clock, proposal, weight and selector all
project onto zero**. Omitting $P_s(\pi)$ produces the complementary block.
For a caller-supplied normalized input $\psi$, the selection probability is
$\|D_{\rm fin}\psi\|^2/\alpha^2$; this oracle alone supplies no deterministic
preparation channel.

The [retained grid](../../assets/generated/simulation/detailed-balance-decay/grid/results.json)
uses $\beta=1$, $m=2$ clock bits and $r=1$ product step, with
$\eta=\alpha=0.9775450868978823$. The access normalizes proposals as
$P_a/\sqrt{J}$; Pauli-label character $i$ acts on $q_i$.

| Hamiltonian and proposals | Wires | Primitives | Source |
|---|---:|---:|---|
| $H=Z$; $X,Z$ | 6 | 130 | [QASM](../../assets/generated/simulation/detailed-balance-decay/grid/diagonal-decay-block.qasm) |
| $H=0.6Z+0.8X$; $Z$ | 5 | 218 | [QASM](../../assets/generated/simulation/detailed-balance-decay/grid/tilted-decay-block.qasm) |
| $H=0.7ZI+0.4XX+0.3IX+0.19YZ+0.23IZ$; $XI,ZI,IX,IZ$ | 8 | 744 | [QASM](../../assets/generated/simulation/detailed-balance-decay/grid/complex-decay-block.qasm) |

Physical mesh mass $\eta$ and scale $\alpha$ were independently checked;
agreement of normalized blocks alone cannot detect dropping $\eta$.
The actual all-zero blocks agree with the independent finite-Trotter
reference to maximum entry error $9.33\times10^{-15}$ and Frobenius error
$1.78\times10^{-14}$. Exact-H finite-target Frobenius errors for the normalized
$D/\alpha$ blocks remain
$0.0942194$ for the tilted spin and $0.0383745$ for the complex two-spin case.
Their classical $\|[H,D]\|_F$ values are $0.424027$ and $0.160156$, with
$\|C\|_F$ values $0.0489876$ and $0.0183416$. These matrices were classical
diagnostics in the decay-access run; its workflow executed no $C$ circuit.
The separately retained one-spin correction step below has its own circuit
and checks. Raw complex entries and small roundoff are retained without clipping.

All 70 focused tests passed in 3.47 s. The
[executed notebook with five code cells](../../assets/generated/simulation/detailed-balance-decay/notebook/executed.ipynb)
is a repository companion: it draws the six-wire source and authenticates
saved grid/source records;
it performs no numerical-check replay. Rerunning this companion requires the
complete source tree, explicit saved-results paths and trusted manifest pins.
Presentation Markdown was updated after execution; code, outputs and
execution counts remain unchanged. The
[source notebook](../../assets/generated/simulation/detailed-balance-decay/notebook/source.ipynb)
and [original execution](../../assets/generated/simulation/detailed-balance-decay/notebook/executed-original.ipynb)
retain their original bytes. The [completion record](../../assets/generated/simulation/detailed-balance-decay/completion.json)
records a 13.70 s whole-CLI wall time for classical CPU work, not quantum
runtime. The original implementation uses the locked Qiskit 2.6.2 environment
{cite:p}`sim_qiskit_262`.

The logical ledger charges both Gaussian PREPARE/SELECT/evolution streams,
weight multiplexors, reflection and full uncomputation:
$N_K=2N_U+4M+6$, on $n+m+\log_2J+2$ wires. The entire doubled construction
is guarded at 100,000 primitives before materialization; source inputs
allow $n\le16$, while dense references stop at two system/eight total wires.
A separate [twelve-spin record](../../assets/generated/simulation/detailed-balance-decay/n12/results.json)
and [QASM](../../assets/generated/simulation/detailed-balance-decay/n12/chain-n12-decay-block.qasm)
emit 2,408 primitives on eighteen wires for the open chain
$\sum_{i=0}^{10}Z_iZ_{i+1}+0.7\sum_{i=0}^{11}X_i$, with endpoint X/Z proposals.
This is source generation without large-state simulation.

The [CLI](../../assets/generated/simulation/detailed-balance-decay/grid/generate_detailed_balance_decay.py),
[block source](../../assets/generated/simulation/detailed-balance-decay/grid/detailed_balance_decay.py),
[independent reference](../../assets/generated/simulation/detailed-balance-decay/grid/detailed_balance_decay_reference.py)
and [Gaussian source](../../assets/generated/simulation/detailed-balance-decay/grid/detailed_balance_access.py)
download alongside the [lock](../../assets/generated/simulation/detailed-balance-decay/grid/requirements-lock.txt)
and [MIT license](../../assets/generated/simulation/detailed-balance-decay/grid/LICENSE.txt).
The adjacent CLI files run without the complete repository. New experiments
use absent output directories:

```bash
python -B generate_detailed_balance_decay.py --tiny-grid --output-dir new-decay-grid
python -B generate_detailed_balance_decay.py --n-system 12 --no-sim --output-dir new-decay-n12
```

New outputs do not extend this grid's validation. External input and
zero-auxiliary preparation/reset, readout, repetitions/amplification,
synthesis, physical faults and full processing costs remain excluded.
General correction/channel implementation, Gibbs accuracy, KMS/mixing
certificates and physical accounting remain planned; the frozen
surface-code baseline is unchanged. The
[bundle manifest](../../assets/generated/simulation/detailed-balance-decay/artifact-manifest.json)
and [digest](../../assets/generated/simulation/detailed-balance-decay/artifact-manifest.sha256)
authenticate the retained files. The completion record keeps
`all_components_verified=false` and `new_algorithm_result=false`.

(gibbs-correction-step)=

## Executed one-spin correction access and heralded step

### Target and original primitive construction

The nonzero coherent correction rotates the system while the filtered jumps
supply dissipation. The next finite endpoint exposes that correction as an
actual primitive block and applies one heralded step to a known input. It uses
only $H=0.6Z+0.8X$ in Z-then-X term order, proposal $Z$, $\beta=1$, two clock
bits and one product step. Its target remains the actual finite-Trotter
$D=D_{\rm fin}$ above, with $\alpha=0.9775450868978823$ and $\tau=1/8$.

Equation 19 of *Efficient quantum thermal simulation* specifies
\[
C_{ij}=\frac{i}{2}\tanh\!\left(\frac{\beta(E_i-E_j)}4\right)D_{ij}
\]
in H's energy basis {cite:p}`frontier_thermal_chen_2025`. For this reflection
$H^2=I$, with energies $+1,-1$, it gives
\[
C=\frac{i}{4}\tanh(\beta/2)[H,D],\qquad
a=\frac{\alpha\tanh(\beta/2)}4=0.11293508916268465.
\]
Here $c_{\rm norm}=\|C\|=0.034639432828173826$ is the **spectral/operator
norm**; the earlier Frobenius diagnostic $\|C\|_F$ has a different meaning.

The original source reuses $V=R_wU$ and emits the centered reflection
$W_D=V^\dagger Z_wV$ through the chronological stream
$U;R_w;P_w(\pi);R_w^\dagger;U^\dagger$. Its zero-auxiliary block is
$I-2D/\alpha$. A new selector c applies Hadamard; controlled-on-one H;
the complete $W_D$; controlled-on-zero H; $R_{z,c}(+\pi)$; Hadamard.
Projecting clock, weight and selector together onto zero gives $C/a$.
Positivity $0\preceq D\preceq\alpha I$ makes the compressed $W_D$ a
contraction; its commutator with the reflection H gives $\|C\|\le a$.
With $\theta=\operatorname{atan2}(0.8,0.6)$, the controlled H reflection
uses chronological $R_y(-\theta)$, Hadamard, CX, Hadamard, $R_y(\theta)$
on the system; the zero control adds two selector X flips. The complete
primitive control lift carries every source phase, including a relative P
gate for a nonzero global phase.

An LCU selector is prepared with probabilities proportional to $1,\tau a$,
controls the complete correction source, receives $P(-\pi/2)$ and is
unprepared. The selected six-wire block is
\[
T_\tau=\frac{I-i\tau C}{1+\tau a}.
\]
Success requires **all five auxiliary bits zero**. The
[results](../../assets/generated/simulation/detailed-balance-correction/v2/results.json)
retain T and every other outcome Kraus block $F_z$, with
$T^\dagger T+\sum_{z\ne0}F_z^\dagger F_z=I$ checked. The success-conditioned
step and the complete success/failure instrument have distinct outputs.

### Actual small circuits and independent checks

| Retained primitive source | Wires | Operations | QASM |
|---|---:|---:|---|
| Centered $W_D$ | 4 | 213 | [source](../../assets/generated/simulation/detailed-balance-correction/v2/centered_decay.qasm) |
| Correction $C/a$ block | 5 | 228 | [source](../../assets/generated/simulation/detailed-balance-correction/v2/correction_block.qasm) |
| Complete controlled correction | 6 | 2,013 | [source](../../assets/generated/simulation/detailed-balance-correction/v2/controlled_correction.qasm) |
| Heralded LCU step | 6 | 2,016 | [source](../../assets/generated/simulation/detailed-balance-correction/v2/heralded_step.qasm) |
| Known-plus coherent check prefix | 6 | 2,017 | [source](../../assets/generated/simulation/detailed-balance-correction/v2/known_plus_prefix.qasm) |
| Complete reset/prepare/readout trial | 6 | 2,029 | [source](../../assets/generated/simulation/detailed-balance-correction/v2/known_plus_trial.qasm) |

These OpenQASM 3 streams contain the actual primitive gates and preserve
global phase. The [builder](../../assets/generated/simulation/detailed-balance-correction/v2/detailed_balance_correction.py)
materializes Gaussian loading, signed product-formula access, weighting,
full inverses, both reflection controls and the complete LCU preparation and
unpreparation. Weighting uses $O(M)$ source gates at $M=4$; its classical angle
preprocessing and every primitive control are accounted for. The endpoint is
fixed to this toy, guarded at six wires and 5,000 operations per source.
These counts provide no general-H scaling or fault-tolerant resource estimate.

The [independent reference](../../assets/generated/simulation/detailed-balance-correction/v2/detailed_balance_correction_reference.py)
rebuilds the signed Fourier/product formula and physical eta/alpha, transforms
D into the known energy basis, and applies Eq. 19 entry by entry. It does not
compute C through the production commutator formula. The separate exact-H
finite comparison is diagnostic. The old $0.0942194$ Frobenius discrepancy
described the normalized $D/\alpha$ block; the corresponding physical D
discrepancy is $0.09210373540320556$. Multiplying by alpha converts the first
quantity to the second. They describe the same finite target and do not show
an accuracy improvement or target change.

All 28 checks passed. Two correction columns, two LCU columns and one
known-plus prefix used five Statevector evolutions with at most 64 amplitudes
each; twelve tiny exponential calls and two interval matrix powers supplied
the references. All 32 Kraus blocks and outcome probabilities are retained.
Maximum entry errors for $C/a$, T and Kraus completeness were respectively
$8.05\times10^{-16}$, $3.45\times10^{-14}$ and $7.06\times10^{-14}$.
The prefix's measurement probabilities were derived from its statevector;
the full reset/readout stream was materialized and counted separately.
No shots or hardware readouts were performed.

### Conditional bias, retries and readout budget

For this traceless Hermitian one-spin C, $C^2=c_{\rm norm}^2I$, so the ideal
success probability is input independent:
\[
p=\frac{1+\tau^2c_{\rm norm}^2}{(1+\tau a)^2},\qquad
p_{\rm low}=(1+\tau a)^{-2}.
\]
The observed known-plus success probability was $0.9723712591335427$.
In the ideal model, the normalized successful map equals
$\exp[-i\arctan(\tau c_{\rm norm})C/c_{\rm norm}]$, with the continuous
identity limit at zero. Its ideal operator error relative to the finite-C
target $e^{-i\tau C}$ is at most $b=(\tau a)^3/3=9.37768\times10^{-7}$;
full-state trace norm and Pauli-mean bias are at most $2b=1.87554\times10^{-6}$.
Observed operator and Z-mean differences were $2.70593\times10^{-8}$ and
$5.06106\times10^{-8}$. These mathematical ideal bounds and floating-point
diagnostics do not certify rounding, synthesis or hardware error.

Each full trial resets all six wires, prepares $|+\rangle$ with H, applies
the whole LCU and measures every wire. Classical selection reads five
auxiliary bits and maps the successful system bit to Z values $+1,-1$.
Failure incurs the same 2,029-operation workload. A new attempt resets and
reprepares the **known** input; this cannot restore an unknown failed state.
Chaining K successful steps would require whole-trajectory restart with
success $p^K$, or a separately implemented recovery construction.

For additive Z error 0.1 and total failure 0.05, the ideal projection reserves
the $2b$ bias first, leaving sampling error $0.09999812446368668$. Allocating
$\delta_s=0.04$ to sampling and $\delta_h=0.01$ to heralding gives
\[
N=\left\lceil\frac{2\ln(2/\delta_s)}{(0.1-2b)^2}\right\rceil=783,
\quad
R=\left\lceil\frac{\ln(\delta_h/N)}{\ln(1-p_{\rm low})}\right\rceil=4.
\]
The projected maximum is 3,132 complete attempts, or 6,354,828 primitive
operations including failed resets, preparations and measurements. Exhaustion
is an explicit failure. These are evaluated ideal-model repetition formulas;
no sampling run was executed and no certified hardware accuracy is inferred.
Classical bit selection and attempt bookkeeping are specified; their separate
timing and hardware costs remain unmeasured.

### Saved evidence, downloads and limits

The [v1 failure report](../../assets/generated/simulation/detailed-balance-correction/v1/failure.json)
retains the output-directory identity failure and its partial source. It
establishes neither a numerical pass nor retained runtime counters. The
distinct [v2 completion](../../assets/generated/simulation/detailed-balance-correction/completion.json)
records successful outer and CLI exits, 5.048 s supervisor wall time,
2.562 s observed CPU and 164,052,992 bytes sampled peak RSS. These are
classical validation-work measurements. The declared envelope was 60 s
wall/CPU, 512 MiB RSS and 8 MiB combined outputs. Monitoring was sampled and supplies
no continuous resource or full process-lifetime certificate.

The [download packet](../../assets/generated/simulation/detailed-balance-correction-download.zip)
contains original source, all measured files and a stdlib saved-only reader.
Its reader authenticates retained identities and scope without importing a
scientific producer or replaying the check. The [bundle manifest](../../assets/generated/simulation/detailed-balance-correction/artifact-manifest.json)
and [v2 manifest](../../assets/generated/simulation/detailed-balance-correction/v2/manifest.json)
keep measured bytes fixed. The original [CLI](../../assets/generated/simulation/detailed-balance-correction/v2/generate_detailed_balance_correction.py)
and adjacent scientific source mode are prepared but unexecuted as a downloaded
package; new experiments require distinct absent outputs and their own bounded
check. Saved identity verification does not extend scientific validation.

The locked Python 3.12.14, Qiskit 2.6.2, NumPy 2.5.4 and SciPy 1.18.1 versions
are retained in the [lock](../../assets/generated/simulation/detailed-balance-correction/v2/requirements-lock.txt)
{cite:p}`sim_qiskit_262,sim_scipy_1181`. Our circuit sources are original and
use the [MIT license](../../assets/generated/simulation/detailed-balance-correction/v2/LICENSE.txt);
[upstream package credits](../../assets/generated/simulation/detailed-balance-correction/UPSTREAM-LICENSES.md)
identify their licenses. General-H correction/evolution, complete channel or
sampler implementation, continuum/KMS accuracy, mixing/output guarantees,
synthesis, physical resources, formal proof and error certification remain
planned. The frozen surface-code baseline is unchanged, and no hardware
advantage is claimed.

## End-to-end resources and fault tolerance

For the zero-correction channel above, the logical workload exposes system
size $n$, inverse temperature $\beta$,
proposal count $J$, clock size $M=2^m$, weak step $\delta$, number of rounds
$K$ and output. It uses $n+m+\log_2J+1$ wires. Every complete round includes
the actual access, frequency rotation, primitive conditional inverse and
explicit environmental resets. Known input initialization and final
measurement are counted separately. Explicit Gaussian loading and weak
multiplexing use $O(M)$ source rotations; the current Gray-Walsh angle
preprocessing uses $O(M^2)$ classical work. This is the ordinary prototype
circuit, rather than the paper's optimized near-linear Lindbladian simulator.

(gibbs-channel-resource-projection)=

The emitted readout circuits use **eight rounds**, with every gate and reset
materialized. For a Z mean of that finite circuit output, additive sampling
error $0.1$ and failure probability $0.05$ give 738 independent complete
preparations by Hoeffding. The logical projection multiplies every full
preparation, reset and readout by that count. No draws or hardware readouts
were performed. This sampling contract is separate from stationary mesh bias
and the recorded 64-round state errors; it is not a Gibbs-accuracy certificate.

Arbitrary-angle rotation synthesis, physical code/factory/controller models,
routing, idle faults, reset error, uniform quadrature bounds, general mixing
and measured classical processing time remain excluded. The frozen
surface-code baseline is unchanged. A physical comparison must fix the
Hamiltonian, output, temperature, accuracy, success and hardware before
charging all synthesis, factories, transport, initialization/reset and
repetitions. No physical estimate or co-design improvement is claimed here.

(gibbs-channel-saved-verification)=

### Saved validation and source reproduction

The default saved checker authenticates the immutable forty-file bundle and
all five current source/lock/license inputs, then checks the literal weak
flag/control/reset chronology, eight-round readout counts and the retained
output contracts. It imports no scientific producer and performs no new
simulation, synthesis or sampling. The [complete source download](../../assets/generated/openquantum-code.zip)
includes this checker and both original generators.

```bash
python scripts/verify_detailed_balance_channel.py
```

A new generator experiment uses a separate **absent** output directory. Keep
its source, parameters and results separate from the measured canonical grid;
regenerating a figure or record does not extend its original validation scope.

## Verification, formalization and implementation sources

The bounded calculations use the locked Qiskit 2.6.2, NumPy 2.5.4 and SciPy
1.18.1 environment on Python 3.12.14 {cite:p}`sim_qiskit_262`. Gate construction and references are
original; no upstream benchmark was reproduced. The bundle includes
[source generation](../../assets/generated/simulation/detailed-balance-channel/generate_detailed_balance_channel.py),
[channel code](../../assets/generated/simulation/detailed-balance-channel/detailed_balance_channel.py),
[access code](../../assets/generated/simulation/detailed-balance-channel/detailed_balance_access.py),
the lock and MIT license for our original sources. An independent copied-source replay passed without
borrowing a parent repository's Git provenance.

The cited proofs remain mathematical literature. No mechanized proof or
implementation bridge has been reproduced. The Nature paper offers
numerical-plot code upon request rather than an established public reusable
package. General noncommuting coherent-correction access/evolution, uniform
continuum accuracy, mixing/output guarantees and complete physical
accounting remain planned. The finite-jump KMS variant above is a separate
unimplemented method.

## References

```{bibliography}
```
