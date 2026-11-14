---
title: "Detailed-balance quantum Gibbs sampling"
description: "An executed finite zero-correction channel, with explicit Gaussian access and remaining general Gibbs-sampler requirements."
---

:::{note} Implementation status: partial

Original Gaussian filtered-jump access and a restricted finite weak reset
channel have executed small checks. The general noncommuting correction,
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

## End-to-end resources and fault tolerance

The logical workload exposes system size $n$, inverse temperature $\beta$,
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
