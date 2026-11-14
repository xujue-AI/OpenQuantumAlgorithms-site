---
title: "Bayesian phase estimation on a circular grid"
description: "An original bounded posterior-only adaptive Hadamard experiment, with explicit access and complete trial accounting."
downloads:
  - file: ../../notebooks/bayesian_qpe.ipynb
    title: "Bayesian phase-estimation notebook"
  - file: ../../assets/generated/bayesian-qpe-validation/source32/source-only.qasm
    title: "33-wire source-only circuit"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Python implementations and tests"
---

## Motivation and intuition

An individual fringe cannot distinguish all phase aliases. Bayesian inference
keeps the alternatives as a probability distribution, updates it after a
measurement, and uses that distribution to choose another experiment.
Published Bayesian methods motivate this route
{cite:p}`frontier_qpe_bayesian_2016,frontier_qpe_mixed_2021`.

:::{note} Implementation status: bounded ideal-model study executed
The fixed study, all 99 tests, all five notebook code cells and a standalone
snapshot replay executed on 13 November 2026 with the locked Python 3.12.14
environment. The eight study outputs represent 384 fresh ideal-model trials;
tests and the separate replay bring the validation total to 456.
No physical estimate, synthesis sequence, hardware execution, confidence
theorem or local Lean proof is supplied.
:::

## Fundamental method and its limits

The model has one fixed unknown eigenphase. For a fresh eigenstate satisfying
$U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle$, a controlled-$U^k$ experiment
and control phase $-\beta$ give

$$p_0(\phi\mid k,\beta)=\tfrac12[1+\cos(2\pi k\phi-\beta)].$$

Represent the prior by nonnegative weights $w_i$ at the dyadic atoms
$\phi_i=i/G$. After outcome $b$, multiply by $p_b(\phi_i)$ and normalize.
An observed outcome with zero model evidence reports a contradiction;
the implementation does not reset the prior or add probability support.
Unsupported positive-support underflow is rejected.

Our original fixed-length greedy policy evaluates powers
$1,2,4,8,16,32$ and offsets $0,\pi/2,\pi,3\pi/2$. It scans candidates in
the supplied order and replaces its current choice only when the expected
Shannon-entropy reduction improves by more than $10^{-12}$ nats. It receives no true
phase. Finite-grid entropy can favor different powers than a continuous
prior; the rule is not a resource-optimal or noise-robust theorem. An
individual observation can increase entropy or widen the posterior.

The output contains a MAP atom, a circular mean when its resultant is not
numerically ambiguous, and a shortest contiguous credible **grid-cell** arc.
Half-cell endpoints give a minimum displayed width $1/G$. Such an arc can
span low-density regions in a multimodal posterior and is not a disjoint HPD
set. A posterior mass of 95% is not a uniform frequentist coverage guarantee,
a certified continuous-phase error interval or a hardware success bound.

## Important variants

This explicit grid method is an original bounded example, not a reproduction
of every procedure in the Bayesian literature. Wiebe--Granade's efficient
inference and the mixed-prior representation are separate method references
{cite:p}`frontier_qpe_bayesian_2016,frontier_qpe_mixed_2021`. Multimodal
uncertainty over a single fixed phase is different from a freshly prepared
state with several spectral components. Our single-phase likelihood does not
support the latter. Likelihood calibration, decoherence-aware experiment
choice, continuous representations and adaptive stopping require separate
contracts. Phayes remains an inspected external package; it is not used by
this implementation.

## Small circuit and prescribed study

The pedagogical family is
$U_n=\bigotimes_{j=1}^{n}P(\theta)$ and
$|\psi\rangle=|1\rangle^{\otimes n}$, with
$\theta=(2\pi\phi_{\rm nominal})/n$ stored as binary64. The stored-angle
eigenphase and nominal phase are reported separately. This classically easy
family supplies a transparent correctness example, not application advantage.

Every controlled base call uses three P gates and two CX gates, repeated
literally $k$ times. Each trial resets all $n+1$ wires, performs the $n$ X
preparation gates, applies the control phase and reads the control. No
$P(k\theta)$ shortcut, opaque matrix gate or free eigenstate oracle is used.
The diagram has two literal calls.

```{embed} #bayesian-qpe-circuit
```

The prescribed study uses phases
$0.3,0.995,0.125,0.5$, two seed namespaces per phase, $G=512$ and 48 fresh
single-shot observations per output. Root seeds are 7 and 19. Each actual
seed is the first eight bytes of
`SHA256(original-grid-bayes:root_seed:phase_index)`, interpreted as a
big-endian integer and masked by $2^{63}-1$. The saved producer's shorthand
"first 63 bits" refers to this code-defined mask; exact seed integers are
retained. No seed, outcome or numerical threshold was tuned.

```{embed} #bayesian-qpe-summary
```

All eight empirical errors are retained. Their maximum is approximately
$0.000970405$; the displayed credible-cell widths are $2/512$ or $3/512$.
This finite record does not establish coverage for other phases, seeds or
hardware. The study charges 10,446 literal controlled base calls and 384
fresh preparations/readouts.

```{embed} #bayesian-qpe-posterior
```

## End-to-end resource analysis

For a complete recorded transcript with $N$ observations and powers $k_j$,
let $K=\sum_jk_j$. This implementation has $n+1$ quantum wires and charges

$$N\ \text{fresh preparations/readouts},\qquad K\ \text{controlled base calls}.$$

The raw source totals are $3nK+N$ P, $2nK$ CX, $2N$ H, $nN$ X,
$(n+1)N$ resets and $N$ measurements. The P offset is emitted and counted
even at zero angle. These are unsynthesized source counts; Clifford-angle
P gates are not all T gates. The full source-operation total is
$5nK+(2n+5)N$.

The policy makes $N+1$ evaluations, including initial validation, and $N$
Bayes updates. For $C$ candidates these require $O(NGC)$ numerical work.
The final credible-cell scan is explicitly bounded at $O(G^2)$; its CPU time
is recorded separately. The online posterior uses $O(G)$ storage; retaining
all demonstration posterior histories uses $O(NG)$ storage. The structural quantum envelope is
$K\le Nk_{\max}$ and
$5nNk_{\max}+(2n+5)N$ source operations. This is a construction bound,
not a phase-precision or predictive hardware-confidence guarantee.

Physical reset and controller implementation, all policy processing latency,
idle exposure, rotation synthesis and its certificate, surface-code distance,
factory supply and physical faults remain uncomputed. The shared static
estimator rejects reset/dynamic schedules; dropping resets would change this
contract. A future physical instrument record must retain the complete
transcript or declared envelope and its classical pauses. No improvement to
the frozen surface-code baseline is claimed.

## Verification, classical simulation and formalization

Executed checks compare complete small circuit operators and probabilities
with an independent diagonal-power matrix, and recursive posteriors with a
direct scalar product of likelihoods. Tests cover circular wrap, opposite
modes, normalized weights, impossible observations, bounded types and
literal preparation/access/readout. The SDK Born normalization retains the
raw norm and rejects a deviation above $2\times10^{-11}$; a normalized
floating result is not a formal accuracy certificate. Across the prescribed
study, the maximum full-operator residual is $1.27\times10^{-15}$, the
source-to-lowered operator residual is $5.20\times10^{-15}$, and the scalar
probability residual is $2.73\times10^{-15}$. Recursive posteriors agree with
the independent batch reference within $2.50\times10^{-16}$.

The complete test file passed 99 cases without skips or deselection. A separate
48-observation replay from the four exported source/lock/license files
matched the first study record in every scientific field, excluding only
timing. The real static notebook executed all five code cells with no new
draws. These execution checks are distinct from the earlier source review.

The simulator is separately capped at three target wires. Source construction
allows 32 target wires and at most 128 literal calls per circuit, with a
200,000-operation circuit cap. Full transcripts stop at 128 observations and
1,000,000 operations; grid sizes are powers of two from 16 to 4096 with
$G\ge4k_{\max}$. This resolution rule is not a quadrature certificate.
The producer has a 120-second/16-MiB export envelope and separate larger
source-only mode. Generated-only sizes carry no numerical validation or
posterior result: the executed generator produced an eight-target/nine-total-wire
companion with 1,301 source operations and a 32-target/33-total-wire circuit
with 20,549 source operations. Neither larger size was densely simulated.
No Lean/Rocq proof or Python-to-proof correspondence runs
here; the [QPE chapter](../qft-qpe.md#verification-and-formalization) retains
its external proof references and their limits.

Code and diagrams are original MIT work. Qiskit 2.6.2 supplies circuits,
export and numerical operators under Apache-2.0
{cite:p}`prim_qiskit_2_6_2`; NumPy 2.5.4 and Matplotlib 3.11.2 supply the
pinned numerical arrays and original plots. No new dependency is introduced.

## Run and references

Use the existing locked Python 3.12.14 environment and an absent output path.
The saved-data verifier checks existing manifests, source snapshots and notebook
source projections without making new modeled draws.

```bash
python scripts/verify_bayesian_qpe_saved.py
python scripts/generate_bayesian_qpe.py --output-dir artifacts/bayes-study
pytest tests/test_bayesian_qpe.py
python scripts/generate_bayesian_qpe.py --source-only --target-qubits 32 \
  --power 128 --output-dir artifacts/bayes-n32
```

The notebook can reuse saved results in a static site. Bounded live controls
are explicitly disabled during static publication and require a trusted
local or Binder kernel when enabled.

```{bibliography}
```
