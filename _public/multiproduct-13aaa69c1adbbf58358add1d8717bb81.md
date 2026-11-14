---
title: "Multiproduct Hamiltonian simulation"
description: "Fixed two-term coherent multiproduct circuits and classical weighted observables, with distinct output and resource contracts."
downloads:
  - file: ../../notebooks/simulation_multiproduct.ipynb
    title: "Multiproduct notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Complete Python code package"
  - file: ../../assets/generated/simulation/multiproduct-bundle.zip
    title: "Checked small circuits, records and standalone generator"
  - file: ../../assets/generated/simulation/multiproduct-large-bundle.zip
    title: "32-spin source exports (no numerical simulation)"
---

## Motivation and intuition

Product formulas approximate evolution by alternating easy term exponentials.
Multiproduct formulas combine different step counts so that leading errors
cancel. A coherent combination uses an actual coefficient register and controlled
circuits; a classical combination estimates observables from separate circuits.
Their outputs and statistical costs differ
{cite:p}`ham_mpf_coherent_2019,ham_mpf_hybrid_2023`.

:::{note} Implementation status

The original primitive circuits, 103 focused tests, 14 small numerical cases,
seven notebook code cells and standalone replay have executed successfully.
A 32-spin circuit bundle was generated without numerical simulation.
The checked methods supply conditional states, a discarded-ancilla channel,
or classical weighted observables, as specified below. Rotation synthesis,
physical resource estimation, literal quantum sampling/retry populations,
mechanized proofs and the optimized high-order family remain planned.
:::

## Fundamental fixed two-term method

Use the same open chain as [Hamiltonian dynamics](../hamiltonian-simulation.md):

$$H=J\sum_{i=0}^{n-2}Z_iZ_{i+1}+h\sum_{i=0}^{n-1}X_i.$$

The units are $J=1$, $\hbar=1$, and the demonstration fixes $h=0.7$, $t=1$,
input $|0\cdots0\rangle$ and readout $Z_0$. Qiskit uses $q[0]$ as the least
significant bit; displayed basis strings read $q[n-1]\cdots q[0]$.
Let $S_2$ be the symmetric ZZ-half/X-full/ZZ-half Strang circuit. Define

$$U_f=S_2(t/(2r))^{2r},\qquad U_c=S_2(t/r)^r,
\qquad A_r=\frac{4U_f-U_c}{3}.$$

At fixed $H,t$ and sufficiently small $t/r$, the symmetric expansion gives
$U_r=U+r^{-2}F_2+r^{-4}F_4+\cdots$. The combination cancels the $r^{-2}$
term. This fixed fourth-order example does not implement the optimized
high-order/polylogarithmic-accuracy construction of the general algorithm
{cite:p}`ham_mpf_coherent_2019`.

### Explicit coherent coefficient access

The two coefficients have absolute sum $5/3$. We add cancelling identity
coefficients $+1/6$ and $-1/6$ to make normalization two. Two selector qubits
are prepared with probabilities $2/3,1/6,1/12,1/12$ for labels
fine, coarse, identity, identity. A selector Z gate supplies signs
$+,-,+,-$. Binary Ry preparation, both controlled Strang branches and inverse
preparation are primitive circuits; a clean work qubit computes each selector
predicate using explicit seven-T Toffolis. It is restored for every selector
and system input in its declared zero-work domain.

The unitary $B$ has zero-selector block $A_r/2$. One oblivious-amplification
round, with $R=I-2P$ and $P$ the zero-selector projector, is

$$G=-BRB^\dagger RB,\qquad PGP=F_r=
\frac32A_r-\frac12A_rA_r^\dagger A_r.$$

The leading minus sign and coherent relative phases are retained. The exported
OpenQASM 3 includes an explicit `gphase` when the SDK export would otherwise
omit the circuit phase. A general controlled-evolution export is not verified.

Writing $D=U_f-U_c$ yields $A_r^\dagger A_r=I+4D^\dagger D/9$.
Unamplified ideal acceptance is at least $1/4$; one OAA round has a uniform
ideal floor $25/729$. OAA does not always improve acceptance. The separate
literal fixture $U_f=I,U_c=-I$ decreases acceptance from $25/36$ to $25/729$.
Its normalized conditional state remains exact; normalization removes the
scalar block error, so this is an acceptance/raw-block counterexample.

### Classical observable combination

The classical estimator is
$\hat\mu=(4\hat\mu_f-\hat\mu_c)/3$, using independent complete fine and coarse
experiments. It prepares neither a coherent evolved state nor a controlled
oracle. The raw estimate is not clipped to $[-1,1]$. Coefficient conditioning
increases sampling work {cite:p}`ham_mpf_hybrid_2023`:

$$\Pr(|\hat\mu-\mu|\ge\epsilon)
\le2\exp\!\left[-\frac{\epsilon^2}{2\sum_jc_j^2/N_j}\right].$$

For bounded Z outcomes, allocate $N_j=\lceil N|c_j|/(5/3)\rceil$ with
$N\ge2(25/9)\log(2/\delta)/\epsilon^2$. Our planner uses a conservative exact
rational upper bound on the logarithm, rather than rounding a floating
threshold downward. Whole physical confidence is a separate requirement.

## Small circuit and independent checks

The executed notebook shows a complete two-spin/r1 LCU trial,
with input resets, actual access, work restoration and every measured bit.
The fixed grid uses $n=1,2,3$ and $r=1,2,4,8$, plus $n=2,3$ at $r=32$.
Independent Kronecker Hamiltonians, matrix exponentials, a full selector
preparation completion and Strang matrix products supply reference objects
{cite:p}`sim_expm_2009,sim_scipy_1181,sim_qiskit_262`.

![Complete two-spin, one-step LCU trial](../../assets/generated/simulation/multiproduct/small-lcu-circuit.svg)

The checks compare complete clean-domain operators with the independently
constructed access model, including $A/2$ and $F$ blocks,
relative/global phases, workspace leakage and all conditional/unconditional
outcomes. An actual reset-channel density check and a small primitive-QASM
replay are included. Operator error, normalized-state error, acceptance and
Z bias are distinct quantities. Single-spin commuting and zero-time cases,
the opposite-identity fixture and finite resource guards are retained.

The largest recorded primitive/reference residual is
$4.63\times10^{-12}$, below the declared $10^{-10}$ comparison tolerance.
This measures implementation agreement; approximation to the target
$U=e^{-iHt}$ is a separate question:

| Spins n | Coarse steps r | $\|U_c-U\|_2$ | $\|A-U\|_2$ | $\|F-U\|_2$ |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | $5.26\times10^{-1}$ | $4.16\times10^{-2}$ | $1.73\times10^{-2}$ |
| 2 | 32 | $4.18\times10^{-4}$ | $3.39\times10^{-8}$ | $2.59\times10^{-8}$ |
| 3 | 1 | $7.22\times10^{-1}$ | $1.19\times10^{-1}$ | $8.38\times10^{-2}$ |
| 3 | 32 | $5.03\times10^{-4}$ | $6.47\times10^{-8}$ | $5.37\times10^{-8}$ |

$F$ is the unnormalized accepted block. Its error above does not measure the
full joint $G$ output against a zero-ancilla ideal evolution. The numerical
channel-reference check compares the implemented discarded-ancilla channel
with an independent construction of that same channel; it does not measure
distance to the ideal U channel. Recorded reduced-state trace distance and Z
bias to the ideal evolution use the specified basis-zero input, with no
measured uniform channel or diamond-norm estimate.

![Fixed-size errors and numerical minimum OAA acceptance](../../assets/generated/simulation/multiproduct/convergence.svg)

At n=2, r=32 the conditional OAA Z bias is $1.14\times10^{-8}$ and the
classical weighted Z bias is $2.02\times10^{-8}$. The fixed-size curve is
consistent with fourth-order cancellation; the finite certificate below is
conservative second order. Complete values and source identities are in the
[study](../../assets/generated/simulation/multiproduct/study.json),
[CSV](../../assets/generated/simulation/multiproduct/convergence.csv) and
[validation summary](../../assets/generated/simulation/multiproduct-validation.json).

## End-to-end resources and fault tolerance

The complete source generators include initialization, both selected branches,
coefficient preparation, inverse access, reflections, work, rejection readouts
and final measurement. SELECT contains $2r+r=3r$ Strang steps; one OAA round
uses three B calls, six PREPARE/inverse calls, three SELECT calls and two
reflections. It does not cost only the longer fine branch.

For active J and h, one B has $18r(3n-2)+89$ primitive unitary instructions.
One OAA round has $54r(3n-2)+281$. B contains 16 T and 12 Tdg gates; the
amplified circuit has 44 T and 40 Tdg because one B call is inverted.
Initial resets and readouts add $2(n+3)$ operations for a zero-input complete
postselected trial. The unconditional OAA channel instead explicitly resets
its three access/work qubits and reads the system. These are source counts,
before arbitrary-rotation synthesis and physical scheduling.

Fresh input preparation is required for postselection retries. The planner
budgets all complete capped attempts for every requested accepted sample;
coherent synthesis bias is not multiplied by sampling count. The classical
variant separately counts its fine/coarse populations and every full circuit.
Its ideal and statistical allowances belong to the combined weighted task.
The fine/coarse physical-failure shares are $4/5$ and $1/5$ of its single
$0.005$ budget; their proposed $0.006$ realization biases combine to $0.01$.
For statistical error $0.03$ and failure $0.004$, the conservative rational
planner requests 14,000 accepted/unconditional samples, or 31,112 fine and
7,778 coarse samples for the weighted estimator. The generator supplies seeded
classical binomial-count illustrations from checked ideal circuit means.
Literal quantum trials and postselection rejection populations remain
unexecuted; the illustrations do not execute these complete workloads.

### Finite error and scaling contract

The supplied certificate avoids an unbounded nested-commutator series.
For the stated ZZ/X ordering, Strang error is bounded by
$C|t|^3/r^2$, where

$$C\le\frac{|J|^2|h|}{6}(4n-6)
+\frac{4|J|h^2}{3}(n-1)\quad(n\ge2),\qquad C=0\quad(n=1).$$

Thus $\|A_r-U\|\le2C|t|^3/(3r^2)$. This conservative finite certificate
is second order even when the fixed-size curve exhibits fourth-order
cancellation. A stronger finite MPF bound and applicable commutator scaling
are needed before claiming optimized large-size accuracy
{cite:p}`sim_trotter_error_2021,ham_mpf_commutator_2026`.

At fixed r the actual local source is $O(nr)$ gates and $n+3$ wires. Holding
accuracy fixed can require increasing r; the conservative certificate gives
$r=O(\sqrt{n|t|^3/\epsilon})$ for fixed couplings. Coefficient conditioning,
sample count, accepted-output handling and physical assumptions remain part
of the end-to-end cost. A single conditional physical point would not prove
that scaling. Onsite Z terms are outside this implementation and its bound.

The surface-code baseline is frozen. Complete logical ledgers and proposed
synthesis/physical budgets are recorded. Synthesized rotation implementations,
factories, controller/reset timing, physical qubits and runtime have not been
estimated. A future comparison must
match Hamiltonian, input/output, accuracy, success and hardware assumptions.
Physical initialization/reset timing, classical coefficient/readout latency,
uniform realized rotation errors and physical scheduling remain uncomputed.

## Classical simulation, formalization and reproduction

Dense verification is capped at three system plus three access/work qubits.
The circuit generator allows at most 61 system qubits, 4,096 coarse steps
and 200,000 complete primitive operations. Scalar counting allocates no
Hamiltonian, state, circuit or size-dependent table. The executed 32-spin
generation produced eight exports with 26,096 total instructions; its complete
OAA measured trial has 5,427 instructions on 35 logical wires. This checks
materialization and source counts, with no dense simulation or inherited
accuracy claim. [Download its source bundle](../../assets/generated/simulation/multiproduct-large-bundle.zip).

No Lean/Rocq theorem or source-to-proof bridge has been reproduced. The
symbolic coefficient, amplification and commutator derivations remain
mathematical exposition, distinct from finite numerical circuit checks.
All exposition, code and circuit figures are original; we use pinned
Qiskit 2.6.2 (Apache-2.0), NumPy 2.5.4 and SciPy 1.18.1 in the locked
Python 3.12.14 environment.

The [standalone generator](../../assets/generated/simulation/multiproduct/generate_simulation_multiproduct.py),
[original module](../../assets/generated/simulation/multiproduct/multiproduct.py),
[locked environment](../../assets/generated/simulation/multiproduct/requirements-lock.txt),
[MIT license](../../assets/generated/simulation/multiproduct/LICENSE.txt) and
[two-spin measured LCU QASM](../../assets/generated/simulation/multiproduct/n2-r1-lcu.qasm)
are downloadable. The small bundle includes the complete study and both SVGs;
the separate large bundle contains generation-only circuits and records.

```bash
python scripts/generate_simulation_multiproduct.py --grid --output-dir NEW_DIR
python scripts/generate_simulation_multiproduct.py --qubits 32 --steps 1 \
  --no-sim --no-figures --output-dir NEW_LARGE_DIR
pytest tests/test_simulation_multiproduct.py
```

Use a Python 3.12 kernel for the companion notebook. Set
`OPENQUANTUM_MULTIPRODUCT_RECORDS` to the new grid directory before executing
it. The generator preserves existing output directories, so a fresh path is
required for a reproduction.

## References

```{bibliography}
```
