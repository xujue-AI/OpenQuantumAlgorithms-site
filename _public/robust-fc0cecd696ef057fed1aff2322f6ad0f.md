---
title: "Robust phase estimation with bounded probability bias"
description: "Fresh-preparation two-quadrature circuits, exact circular lift decisions, a conservative conditional confidence bound and complete logical accounting."
downloads:
  - file: ../../notebooks/robust_qpe.ipynb
    title: "Executable robust phase-estimation notebook"
---

## Motivation and intuition

Repeated interference experiments can identify an eigenphase even when the measurement probabilities have a bounded systematic distortion. A long controlled evolution magnifies a small phase difference; sine and cosine observations identify its circular angle, and shorter evolutions identify the correct alias. This is the robust phase-estimation idea used in gate calibration {cite:p}`frontier_qpe_robust_2015`.

:::{note} Implemented scope

This companion supplies original explicit fresh-preparation circuits, rational circular inference, a seeded small readout-confusion example and logical scaling. It uses an independently derived conservative finite-sample guarantee. The 2015 paper's original error analysis is not adopted. Circuits and readout laws are checked numerically; the arithmetic confidence argument is mathematical and is not a reproduced formal proof. Physical estimation and experimental calibration remain planned.
:::

## Fundamental algorithm and input contract

The input is an exactly prepared single eigenstate

$$U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle,\qquad\phi\in[0,1).$$

At each power $k_j=2^j$, $j=0,\ldots,L-1$, use $m$ fresh independent trials in each of two quadratures. The ideal zero probabilities are

$$p_{c,j}=\tfrac12[1+\cos(2\pi k_j\phi)],\qquad
p_{s,j}=\tfrac12[1+\sin(2\pi k_j\phi)].$$

Our probability model specifies an actual Bernoulli probability $q_{b,j}$ for each batch, with

$$|q_{b,j}-p_{b,j}|\le\beta<\tfrac18.$$

Trials are independent and identically distributed within each batch. A single bias bound must hold at every chosen power and quadrature; it may have different signs in different batches. The bound is supplied as an input promise. No experiment estimates it here. State-preparation, controlled-evolution or synthesis errors can enter this contract only after proving that their combined effect satisfies the probability bound. Correlated drift, arbitrary mixed spectral inputs and unbounded power-dependent dephasing are outside this single-phase guarantee.

With zero counts $z_{c,j},z_{s,j}$, construct the exact rational signals

$$c_j=2z_{c,j}/m-1,\qquad s_j=2z_{s,j}/m-1.$$

The code encloses $\operatorname{atan2}(s_j,c_j)/(2\pi)\bmod1$ in an exact rational interval $[a_j,b_j]$ and uses its midpoint $\widetilde\theta_j$. It begins with $\widehat\phi_0=\widetilde\theta_0$. At power $k_j>1$, choose a lift $(\widetilde\theta_j+\ell)/k_j$ nearest the preceding circular estimate. Every interval decision uses fractions. In particular, accept an integer $\ell$ only if

$$\ell-\tfrac12<k_j\widehat\phi_{j-1}-b_j
\quad\text{and}\quad
k_j\widehat\phi_{j-1}-a_j<\ell+\tfrac12.$$

This ensures that the same lift is nearest for every angle in the enclosure. Normalize the lifted midpoint modulo one. Zero signal, an enclosure outside its width guard or a half-spacing boundary returns an explicit inconclusive result, retaining all counts and completed decisions. An accepted result does not itself verify the unknown bias promise.

### Independently derived conditional confidence

Set $t=1/8-\beta>0$. Hoeffding applied to each of the $2L$ empirical proportions gives

$$\Pr(\text{some }|\widehat p_{b,j}-q_{b,j}|>t)
\le4L\exp(-2mt^2).$$

On the complementary event, each empirical signal component differs from its ideal value by at most $2(\beta+t)=1/4$. The complex-signal error is at most $\sqrt2/4$. Tangent geometry around the unit circle gives an angular error smaller than $1/16$ turn, since

$$\sqrt2/4<\sin(\pi/8).$$

The code requires an angle-enclosure width at most $1/96$ turn. Its midpoint therefore adds at most $1/192$ turn, giving a fringe-center error below $13/192<r=1/12$. The correct next lift is at distance at most $3r/k_j$ from the preceding estimate: the preceding error contributes $2r/k_j$, and the new center contributes $r/k_j$. Because $3r=1/4<1/2$, the correct lift is unique. The enclosure width also leaves a strict margin for the interval decision above. Induction yields

$$d_{\mathbb T}(\widehat\phi,\phi)\le\epsilon_\phi
=\frac1{12\,2^{L-1}}$$

except on an event of probability at most $4L\exp(-2mt^2)$. Inconclusive outputs belong to that failure event; they are never omitted from a reported study. The returned closed circular arc splits into two rational intervals when it crosses zero. This is a conditional uniform coverage argument over the stated probability model, not a Bayesian credible interval or an empirical coverage estimate.

The planner uses the positive 128-term rational Taylor sum $E_{128}(2mt^2)\le\exp(2mt^2)$ and accepts a shot count only when

$$\frac{4L}{E_{128}(2mt^2)}\le\delta.$$

All ceiling comparisons are exact. The finite-degree bound can be conservative and the finite trial cap can reject a request; the planner does not claim the globally optimal shot count. The angle calculation uses a 32-term alternating arctangent series with its signed next-term remainder, an exact range reduction to magnitude at most $1/2$, and Machin's identity $\pi=16\arctan(1/5)-4\arctan(1/239)$. Axes and quadrant decisions are exact. No floating-point `atan2` determines an alias.

### Important variants and the correction

The author-corrected paper, dated **21 October 2021**, explains that the original $\pi/(2k_j)$ local-error event was insufficient: an alias error could occur even inside that interval. The replacement analysis uses the stricter $\pi/(3k_j)$ condition and supersedes the original Sections V A and V B {cite:p}`frontier_qpe_robust_errata_2021`. Our conservative argument above independently controls both neighboring levels. It does not reuse the incorrect constants or claim the paper's full calibration procedure.

Published robust calibration protocols use other measurement realizations and level-dependent schedules, and discuss root-mean-square error and optimized accumulated evolution work. This implementation uses uniform shots with a union confidence bound. It has an inverse-precision coherent demand plus logarithmic confidence overhead; it does not inherit an optimal total-work Heisenberg-scaling claim. The [ideal Hadamard companion](hadamard.md) has a different ideal-only guarantee, while [Bayesian inference](bayesian.md) reports posterior grid-cell arcs. Neither posterior credibility nor a numerical consistency test establishes this bias-model promise.

## Small circuit and executed readout example

The circuit family is the explicit $n$-wire product

$$U_n=\bigotimes_{i=1}^{n}P(\theta),\qquad
|\psi\rangle=|1\rangle^{\otimes n},\qquad
\theta=(2\pi\phi_{\rm nominal})/n.$$

The angle is stored as binary64. The nominal phase and a numerical diagnostic from the actual stored angles are recorded separately. The constructor rejects unsupported subnormal/underflow cases. This commuting model is classically easy; its known phase supplies an independent reference and never enters count recovery.

Each trial resets every wire, prepares the data with $n$ X gates, applies H to the control, repeats the actual controlled base circuit $k$ times, optionally applies $S^\dagger$ for the sine basis, and applies H/readout. Every controlled $P(\theta)$ is decomposed into three P gates and two CX gates. There is no dense-unitary gate, phase-only symbolic oracle or analytical $P(k\theta)$ shortcut. The code reuses the unchanged original [Hadamard circuit implementation](../../assets/generated/robust-qpe/hadamard_qpe.py).

```{embed} #robust-qpe-circuit
```

The prescribed study uses phases $0.3,0.995,0.125,0.5$, seeds $7,19$, and $L=6$. It explicitly models a classical readout channel with $r_{01}=3/200$ and $r_{10}=1/100$:

$$q_0=(1-r_{01})p_0+r_{10}(1-p_0).$$

Its distortion is bounded by $\max(r_{01},r_{10})=0.015$, inside the declared conservative allowance $\beta=1/50=0.02$. This allowance leaves numerical slack; circuit/reference equality is checked numerically and is not a rigorous interval certificate for SDK gates. The whole eight-output study shares a failure target $0.01$. Counts are classically sampled from numerically checked circuit probabilities followed by the specified independent readout channel; these are represented fresh trials, not QPU measurements.

```{include} ../../assets/generated/robust-qpe/summary.md
```

```{embed} #robust-qpe-recovery
```

Every prescribed output and every aggregate count batch is retained, including any miss or inconclusive result. Complete small operators are checked against independently assembled block matrices; the readout probabilities are checked against an independent two-by-two confusion matrix/Bernoulli law. A separate [nine-wire measured circuit](../../assets/generated/robust-qpe/circuits/large-n8-sin-k32.qasm) is materialized and counted, with dense simulation explicitly unexecuted.

## End-to-end resource scaling

For one output, with $n$ target wires, $L$ levels and $m$ shots per quadrature/level,

$$M=2Lm\quad\text{fresh preparations and readouts},\qquad
Q=2m(2^L-1)\quad\text{literal controlled base-}U\text{ calls}.$$

The maximum coherent base-call demand is $k_{\max}=2^{L-1}$, and the width is $n+1$. The raw all-trial source counts are $3nQ$ P rotations, $2nQ$ CX, $2M$ H, $nM$ X, $Lm$ $S^\dagger$, $(n+1)M$ resets and $M$ measured bits. Raw P counts include special Clifford angles and are not synthesized T counts. Each exported circuit records separate source/lowered counts and dependency depths. All trials and earlier powers remain charged when the eventual output succeeds or is inconclusive.

For fixed $\beta<1/8$, the theoretical exponential law allows

$$L=O(\log(1/\epsilon_\phi)),\qquad
m=O\!\left((1/8-\beta)^{-2}\log(L/\delta)\right).$$

Consequently the product-family gate work is

$$O\!\left(n\epsilon_\phi^{-1}(1/8-\beta)^{-2}
\log\!\frac{\log(1/\epsilon_\phi)}{\delta}\right).$$

The [36-row scaling CSV](../../assets/generated/robust-qpe/scaling.csv) varies $n$, $L$ and $\beta$ at an explicit failure target. It is an analytic workload table for the constructor, not large-simulation or physical evidence. An application must substitute its actual preparation and controlled-access circuits and establish the single-eigenphase/bias promise.

Classical count accumulation corresponds to $2Lm$ represented bit observations; the bounded simulator draws only $2L$ binomial aggregates. Inference uses $O(L)$ fixed-degree rational operations, with arithmetic bit cost growing with the count precision. The record retains separate binomial-sampling and rational-inference CPU times. Storage includes every count, interval, alias decision and output. Classical compilation, communication and calibration time are unmeasured. **Calibration is supplied as a promise: no calibration run is performed or priced, and experimental calibration work is additional.**

### Fault-tolerant costing remains planned

This is a bounded probability/readout model, not physical fault tolerance. Continuous rotations are lowered to RZ but unsynthesized. No factory, code-distance, physical-qubit, routing or physical-runtime estimate is supplied, and the frozen surface-code baseline is untouched. Initial reset, X preparation, every controlled gate and final readout appear in the logical circuits and cannot be discarded to fit a static estimator.

A physical adapter must compile the rotations, retain all resets/repetitions/readouts and account for data exposure, factories and scheduling. It must separately bound any resulting preparation, evolution, synthesis and readout probability distortions; their combined allowance reduces the remaining sampling margin $t$. A gate-failure rate alone is not a per-fringe additive-bias certificate. Physical gains against textbook or retained-target QPE remain uncomputed. Neither $Q$ nor $k_{\max}$ equals physical wall time.

## Classical verification and formal scope

The small circuit family has an independent analytic answer and full bounded matrix checks. Tests cover off-grid phases, wraparound, axes and octants, adversarial fringe-center offsets within the sufficient condition, zero signals, exact half-spacing ties, malformed/unbounded count iterables, bias/circuit/sample caps and source ledgers checked against actual programs. An independent high-precision Decimal construction checks rational angle enclosures and conservative exponential ceilings. These numerical tests cover their instances; the elementary proof above supplies the general conditional mathematical argument.

No Lean/Rocq proof build or Python-to-formal correspondence is reproduced. The primary paper, exact rational computation, SDK realization and physical compilation are separate evidence layers. Sandia's `pyRPE` at `e09be5a83762c94de7880daf45230996b9f6cab3` remains an inspected BSD-3-Clause/Apache-2.0 inference reference; it is not installed or imported by this implementation. No external inference code, survey prose or figure is copied.

## Reproduce, finite limits and packages

Use the locked **Python 3.12.14** environment. The generator requires a new output directory and preserves existing records:

```bash
python scripts/generate_robust_qpe.py --single --phase 0.995 --levels 6 \
  --additive-bound 1/50 --zero-to-one 3/200 --one-to-zero 1/100 \
  --output-dir artifacts/robust-new-run
python scripts/generate_robust_qpe.py --single --target-qubits 8 --levels 6 \
  --no-sim --output-dir artifacts/robust-source-only
pytest tests/test_robust_qpe.py
```

Construction stops at 32 data wires, 16 dyadic levels, $2^{15}$ base calls per experiment, 200,000 operations per circuit/schedule and 64 distinct circuits per invocation. The whole producer stops at 100,000 represented trials, 16 MiB output and 120 seconds. Dense checks stop at three data wires. The confidence planner can reject tight requests before sampling. `--no-sim` creates source/count artifacts and reports zero sampled trials; it supplies no phase result at that size.

The [normal notebook](../../notebooks/robust_qpe.ipynb) executes gate checks and replays saved counts without new sampling. Download [results](../../assets/generated/robust-qpe/results.json), [manifest](../../assets/generated/robust-qpe/manifest.json), [producer](../../assets/generated/robust-qpe/generate_robust_qpe.py), [inference source](../../assets/generated/robust-qpe/robust_qpe.py), [lock](../../assets/generated/robust-qpe/requirements-lock.txt), [software versions](../../assets/generated/robust-qpe/software.json) and [MIT license](../../assets/generated/robust-qpe/LICENSE.txt). All figures and local algorithm code are original MIT work. Qiskit **2.6.2** supplies circuits, export and statevectors under Apache-2.0 {cite:p}`prim_qiskit_2_6_2`; NumPy **2.5.4** supplies arrays under BSD-3-Clause, and Matplotlib **3.11.2** supplies original diagrams under its license. The generator records these exact versions from the environment and ships both local source modules required for portable execution.

## References

```{bibliography}
```
