---
title: "Hadamard phase estimation with fresh preparations"
description: "Explicit repeated controlled powers, sine/cosine count data, bounded dyadic recovery and complete fresh-trial accounting."
downloads:
  - file: ../../notebooks/hadamard_qpe.ipynb
    title: "Executable phase-recovery notebook"
---

## Motivation and intuition

An eigenphase becomes visible through interference with a control qubit. Measuring that control in two bases gives sine and cosine information; experiments at several powers resolve the aliases of an individual fringe. Classical processing replaces the coherent inverse QFT in this Hadamard-test/Kitaev-style route {cite:p}`prim_qpe_1998`.

:::{note} Implemented scope

Explicit primitive experiments, bounded dyadic angle recovery, independent matrix/probability checks and a twelve-output seeded study are executed. Each represented quantum trial has a fresh prepared target. Counts are sampled classically from checked ideal circuit probabilities; no QPU execution, native physical estimate, synthesized rotation sequence or Lean proof is supplied. The ideal confidence argument has exact-input/exact-real assumptions, while SDK and inverse-tangent checks remain numerical.
:::

## Fundamental algorithm and input promise

For an exact prepared eigenstate $U|\psi\rangle=e^{2\pi i\phi}|\psi\rangle$, an H–controlled-$U^k$–H experiment gives

$$p_{\cos}(0)=\tfrac12[1+\cos(2\pi k\phi)].$$

Applying $S^\dagger$ to the control before the final H instead gives

$$p_{\sin}(0)=\tfrac12[1+\sin(2\pi k\phi)].$$

The implementation uses $k_j=2^j$, $j=0,\ldots,L-1$, and $m$ independent fresh trials per basis and power. From the zero counts, construct $c_j=2\widehat p_{\cos,j}-1$, $s_j=2\widehat p_{\sin,j}-1$, and $\theta_j=\operatorname{atan2}(s_j,c_j)/(2\pi)\bmod1$. Start with $\widehat\phi_0=\theta_0$; at each later power choose the circular lift $(\theta_j+r)/k_j$ nearest the preceding estimate. A zero empirical signal returns explicit ambiguity.

This is the specified dyadic recovery protocol, rather than a claim to implement every published Kitaev, Bayesian or robust estimator. Recovery receives only counts, powers and shot totals; it receives no true phase or eigenvalue. A non-eigenstate instead yields a weighted spectral signal, which need not lie on the unit circle. The single-phase argument does not cover such inputs. Freshly preparing a mixed state for every experiment also differs from the [retained-target feedback instrument](iterative.md) and the energy chapter's retained-target median protocol.

### Conditional ideal confidence

If both estimated probabilities at one power differ from their ideal values by at most $1/8$, the complex-signal error is at most $\sqrt2/4$. Its phase-angle error is therefore smaller than $1/8$ cycles. The correct next lift lies within $3/(8k_j)$ of the preceding estimate, strictly inside half the lift spacing; induction gives circular phase error at most

$$\epsilon_\phi=\frac1{8k_{\max}}=2^{-(L+2)}.$$

Hoeffding for the $2L$ proportions and a union bound give

$$\Pr(\text{failure})\le4L\exp(-m/32).$$

The shot planner uses an exact positive rational Taylor lower bound on $\exp(m/32)$, so its reciprocal is a conservative upper bound. The integer threshold is not selected by a rounded logarithm alone. The finite-degree planner is conservative; it may choose additional shots or reject extreme requests, and is not claimed globally optimal for the exact exponential law. This arithmetic bound assumes exact eigenstate preparation, exact controlled powers, IID fresh ideal trials and exact-real angular inference. It does not certify our floating-point `atan2`, circuit realization, synthesis or noisy hardware.

## Small circuit and executed recovery

The implemented family is $U_n=\bigotimes_{i=1}^{n}P(\theta)$, with the target $|1\rangle^{\otimes n}$ prepared by $n$ X gates and $\theta=(2\pi\phi_{\rm nominal})/n$ stored as binary64. Its true eigenphase comes from the stored angles; the nominal phase and numerical actual-angle reference are recorded separately. This commuting family is classically easy and provides no application advantage claim.

Each controlled $P(\theta)$ is explicitly decomposed into three P gates and two CX gates. The fundamental circuit repeats that construction **$k$ times**. It does not replace it with one $P(k\theta)$ gate or assume generic fast-forwarding. Every measured circuit starts with resets on all $n+1$ wires, applies the X preparation, and reads out the control. The diagram shows two literal controlled calls for one sine experiment.

```{embed} #hadamard-qpe-circuit
```

The prescribed study uses phases $0.3,0.995,0.125,0.5$, $L=6$ and seeds $7,19,41$, with separate case streams. Twelve outputs share a total ideal failure target $0.01$ by assigning $1/1200$ to each. The exact planner selects **329 shots per fringe**, giving **47,376 represented fresh trials** across 48 experiment circuits. The ideal-model radius is **$1/256$**; the largest observed circular error is approximately **0.000417456**.

```{include} ../../assets/generated/hadamard-qpe/summary.md
```

All prescribed outputs fall within the ideal-model radius in this record. That observation does not establish empirical confidence for future instances or hardware. The record retains every count batch, seed, power, lift decision, source/lowered circuit and numerical residual. Sine/cosine probabilities and complete small circuit operators are compared with an independent block-matrix expression.

## End-to-end resource analysis

Let $n$ be the data width, $L$ the number of dyadic powers and $m$ the shots per basis/power. One classical phase output uses

$$M=2Lm\quad\text{fresh preparations/readouts},\qquad Q=2m(2^L-1)\quad\text{literal controlled base-$U$ calls}.$$

For this product-phase implementation, raw source counts are $3nQ$ P gates, $2nQ$ CX, $2M$ H, $nM$ X, $Lm$ $S^\dagger$, $(n+1)M$ resets and $M$ readouts. The width is $n+1$ throughout. P counts include special Clifford-angle cases; they are not synthesized T counts. Source and lowered counts/dependency depths remain separate.

For the whole recorded twelve-output study, **497,448 shot-weighted controlled base calls** are charged. Reusing one favorable result or a late inference success does not erase prior trials. The maximum coherent demand is $k_{\max}=32$ base calls in one trial; this differs from the total repetition-weighted work. Neither quantity is physical wall time.

At radius $\epsilon_\phi$, the ideal exponential bound permits $L=O(\log(1/\epsilon_\phi))$ and $m=O(\log(L/\delta))$. This is the theoretical shot choice, rather than an asymptotic optimality claim for the bounded finite-degree planner. The product-family gate work is $O(n\epsilon_\phi^{-1}\log(\log(1/\epsilon_\phi)/\delta))$. A general application substitutes its actual controlled-access and preparation costs. Classical count accumulation and $O(L)$ angular/lift processing must be charged separately; the generator records their CPU time. Constructing probabilities from the known phase is used only as an independent reference, not by recovery.

The [source-count scaling table](../../assets/generated/hadamard-qpe/scaling.csv) varies $n$ and $L$ at stated failure targets. It is an analytic construction/workload table, not a physical or large-simulation result. A [nine-wire measured circuit](../../assets/generated/hadamard-qpe/circuits/large-n8-sin-k32.qasm) is actually generated by the same constructor; its dense simulation remains unexecuted under the separate three-data-wire check cap.

### Fault-tolerant analysis

Physical estimation remains **planned**. An adapter must retain the actual initial resets, target preparation, every controlled gate, synthesis, data exposure, all repetitions and final readout. The static QDK adapter rejects resets, so dropping them to reuse its numbers would remove required work. A suitable instrument model must also allocate gate/reset/readout faults and probability bias at the same precision/confidence target.

The frozen surface-code baseline is unchanged. Factory/code-distance choices, rotation synthesis, routing and reset/controller timings have not been computed for this protocol. Ideal probability concentration cannot be transferred to approximate or noisy trials without a new systematic-error allowance and confidence calculation. Physical initialization, communication and compilation latency are unmeasured. There is no physical gain ratio against textbook or retained-target QPE.

## Verification, record provenance and classical simulation

The gate/operator checks cover normal-angle one-, two- and three-data-wire cases, sine signs, controlled scalar phases and explicit measured/reset instruments. Recovery tests include circular wrap, exact/integer fringes, zero-signal ambiguity, malformed counts and bounded iterable consumption. The confidence planner's rational bound is compared with an independent high-precision decimal exponential.

The saved canonical producer is `def275c501c8c55a32e219674000b4199eaa16d4`; its source snapshots, counts, timings and manifest remain intact. The current source adds only a rejection guard for nonexact subnormal halving/underflow. A **zero-sampling replay** authenticates that immutable producer, proves every canonical component bypasses the guard, reproduces all 49 QASM exports, performs numerical probability/operator checks for the 48 small circuits, checks register/resource evidence and recovers all twelve saved outputs. The larger nine-wire export remains source/count-only. Producer and replay source identities are recorded separately; the replay is not a new sampled experiment.

No formalization or Lean proof is reproduced for these circuits or recovery. The circuit, numerical inference, exact bound arithmetic and physical model are distinct evidence layers.

## Reproduce, limits and packages

Use the locked Python 3.12 environment and complete public code package:

```bash
python scripts/verify_hadamard_qpe.py
python scripts/generate_hadamard_qpe.py --single --phase 0.3 --levels 6 \
  --output-dir artifacts/hadamard-new-run
python scripts/generate_hadamard_qpe.py --single --target-qubits 8 --levels 6 \
  --no-sim --output-dir artifacts/hadamard-large
pytest tests/test_hadamard_qpe.py tests/test_hadamard_qpe_generator.py \
  tests/test_hadamard_qpe_record.py
```

The [notebook](../../notebooks/hadamard_qpe.ipynb) executes saved replay and gate checks. Optional controls check gate probabilities and run new bounded experiments in a local/Binder kernel. Set `OPENQUANTUM_STATIC_BUILD=0` in that kernel to enable them; static publication keeps them disabled and needs no public backend.

Construction stops at 32 data wires, $2^{15}$ literal calls per individual experiment, 200,000 operations per circuit/schedule and 64 distinct experiments per generator invocation. Confidence workloads stop at 100,000 represented trials per producer; output stops at 64 MiB with a 180-second generator allowance. Dense references stop at three data wires. Stored nonzero phase components must permit exact binary64 halving; unsupported subnormal/underflow inputs are rejected. These guards are separate from a scalable-performance guarantee.

Code and figures are original MIT work. Qiskit **2.6.2** supplies circuits/export/statevectors under Apache-2.0 {cite:p}`prim_qiskit_2_6_2`; NumPy **2.5.4** supplies numerical arrays under BSD-3-Clause, and Matplotlib **3.11.2** supplies the original diagrams under its license. [Software/versions](../../assets/generated/hadamard-qpe/software.json), [lock](../../requirements-lock.txt), [historical producer source](../../assets/generated/hadamard-qpe/hadamard_qpe.py), [record manifest](../../assets/generated/hadamard-qpe/manifest.json) and [MIT license](../../assets/generated/hadamard-qpe/LICENSE.txt) are downloadable. Use the current full code package when exploring other inputs; the producer snapshot is an audit artifact.

## References

```{bibliography}
```
