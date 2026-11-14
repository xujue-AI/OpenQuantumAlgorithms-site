---
title: "Adaptive amplitude estimation"
description: "A bounded branch-adaptive sequence of explicit amplified flag measurements."
downloads:
  - file: ../../notebooks/amplitude_iterative.ipynb
    title: "Executable adaptive notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Python examples and tests"
---

Adaptive amplitude estimation uses repeated amplified flag measurements to narrow a probability interval. Each quantum trial uses only data and flag qubits; a classical controller chooses the next amplification power from the preceding interval. Grinko and colleagues established iterative amplitude estimation as a distinct approach {cite:p}`prim_iterative_ae_2021`; the bounded fresh-batch variant here specifies its own sampling and stopping rules.

:::{note} Implemented and checked scope

Explicit threshold circuits, their independent amplified-flag laws, seeded sampled trajectories, conservative inference arithmetic, and complete recorded-path accounting are implemented and checked. The default reaches its requested interval width after four fresh batches. A separate 10-case circuit suite and 224-run model-only coverage diagnostic retain their full records. Resource projections remain conditional on the recorded ideal path; no hardware confidence certificate or quantum advantage is established.
:::

## Task, input and intuition

Estimate $a=k/2^n$ for the [explicit uniform threshold preparation](../amplitude-estimation.md#task-and-intuition), with requested absolute error $\epsilon$ and ideal statistical failure allowance $\delta$. The threshold has an exact classical shortcut, so it is a circuit and inference example rather than an application benchmark.

Use the normalized angle $t=\arcsin(\sqrt a)/\pi\in[0,1/2]$, so $a=\sin^2(\pi t)$. After preparing $A|0\rangle$ and applying the actual Grover iterate $Q$ $r$ times, the flag probability is

$$p_r=\sin^2(\pi(2r+1)t).$$

Larger multipliers magnify an angle change, but also introduce aliases: several angles can have the same amplified probability. The controller maintains a valid angle interval and chooses the largest permitted odd multiplier whose entire **unwrapped** interval lies on one lobe $[j/2,(j+1)/2]$ of $\sin^2(\pi u)$. Exact rational comparisons establish this monotone branch. The known threshold answer checks the example, rather than choosing powers or stopping inference.

## Explicit circuits and sampled rounds

Every trial is a fresh $A|0\rangle$ preparation, followed by $Q^r$ and a computational-basis measurement of the flag. Preparation, inverse preparation and both reflections occur inside each actual iterate. Data wires are discarded; measuring only the flag is the requested scalar output. All powers are repeated actual gates, without a dense unitary or probability oracle standing in for the quantum circuit.

```{embed} #prim-amplitude-iterative-circuit
```

The default input is two data bits and threshold one, so $a=1/4$. It uses $\epsilon=0.02$, $\delta=0.01$, 256 fresh shots per round, at most 20 rounds, powers at most 64, total shots at most 8,192 and seed 314159. Counts are ideal binomial draws from the bounded circuit statevector probability. The independent sine-squared formula checks that probability; it is not a gate-construction input.

```{include} ../../assets/generated/amplitude-iterative/summary.md
```

```{embed} #prim-amplitude-iterative-trajectory
```

For each new round, the controller selects its power from the preceding interval, runs exactly one fresh fixed-size batch, and computes a two-sided Clopper–Pearson interval for the amplified flag probability. The round receives $\delta_j=\delta/T$, where $T$ is the declared maximum number of rounds. It inverts the interval on the previously chosen branch and intersects the resulting angle interval with the prior one. Counts from different rounds are not pooled or reused as a fixed-size confidence interval after optional stopping.

The algorithm reports a conclusive midpoint only when its **certified radius about the actual reported float is at most $\epsilon$**, which also ensures interval width at most $2\epsilon$. A power/round/shot cap can instead return **inconclusive**; an empty intersection is also retained explicitly. The program never quietly changes the requested accuracy or reports a capped wide interval as a successful estimate.

## Ideal confidence and variants

Conditional on every preceding history, the new fixed-size batch has binomial law with its predictably chosen power. Its Clopper–Pearson miss probability is at most $\delta/T$. A union bound therefore bounds the probability of any miss by $\delta$ over the capped run. On the event that every interval covers, monotone inversion and intersection preserve the true angle. A conclusive midpoint then has error at most $\epsilon$.

The resulting ideal statement is **$P(\text{conclusive and wrong})\le\delta$**. It is not a claim that $P(\text{correct}\mid\text{conclusive})\ge1-\delta$, and caps do not guarantee convergence.

SciPy beta quantiles seed the interval endpoints, which move outward until exact integer/rational binomial-tail inequalities verify the allocated error. The per-batch allocation is rounded **down** from $\delta/T$. Probability inversion uses conservative 192-bit fixed-point sine-squared enclosures, rational Machin bounds for $\pi$, and a bounded Taylor remainder; angle conversion and the radius around the reported floating midpoint are outward bounded. These are executed conservative arithmetic certificates, separate from a Lean proof or a certified iid PRNG/device sampler.

| Approach | Implemented contract | Main tradeoff |
|---|---|---|
| Fresh-batch branch-adaptive variant | Fixed fresh batches, explicit per-round confidence allocation, monotone branch inversion and honest capped status. | A chosen path can save phase qubits, but late powers increase coherent work and caps can remain inconclusive. |
| Original IQAE | Primary methodological reference; our schedule differs. | Its published complexity constants are not automatically inherited by a changed no-pooling capped schedule. |
| Standard QPE amplitude estimation | [Separate implemented companion](../amplitude-estimation.md). | Adds a coherent phase register and controlled amplification powers. |
| Classical flag sampling | Same preparation with no amplification. | Hoeffding needs $\lceil\log(2/\delta)/(2\epsilon^2)\rceil$ independent shots for the requested precision. |

At $\epsilon=0.02$, $\delta=0.01$, the classical sufficient count is 6,623 fresh preparations and flag readouts. The recorded adaptive path uses 1,024 trials, but charges 2,048 predicate/inverse accesses. A successful recorded path cannot establish a matched-output advantage against an always-returning classical estimator: the adaptive method's chance of remaining inconclusive and its cap behavior also need accounting. The known threshold already gives the exact classical probability without sampling.

## End-to-end resource analysis

For a recorded trial at power $r$, the circuit uses $n+1$ live qubits, one initial $A$, $r$ additional $A$, $r$ $A^\dagger$, both reflections $r$ times and one flag measurement. It therefore evaluates the threshold predicate or its inverse **$1+2r$ times**. Each selected shot pays that entire schedule, including the rounds that do not narrow the interval enough to stop.

The [logical ledger](../../assets/generated/amplitude-iterative/logical-resources.json) weights actual lowered circuits by **every recorded shot**, preserving per-power OpenQASM hashes, gate counts, dependency depth and all access/preparation/readout calls. The [trajectory](../../assets/generated/amplitude-iterative/trajectory.json) records all selected powers, confidence/branch decisions, counts, intervals, seeds and RNG provenance. Classical control and compilation time are distinct from modeled quantum runtime.

The default has three live qubits and powers $[0,0,0,2]$. The three power-zero batches have 768 shots; the power-two batch has 256. The complete workload contains 512 $Q$ calls, 2,048 threshold/inverse calls and 1,024 flag measurements:

| Trial power | Recorded shots | Operations/shot | Dependency depth/shot | All-shot operations |
|---:|---:|---:|---:|---:|
| 0 | 768 | 20 | 12 | 15,360 |
| 2 | 256 | 120 | 80 | 30,720 |

Thus the recorded total is **46,080 primitive operations**. Dependency depths describe these explicit circuits before physical scheduling. Primitive arbitrary rotations remain subject to the separate synthesis contract.

For system size $n$ and the actual sampled schedule $(N_j,r_j)$, a full gate-cost model is

$$
G_{\rm total}(n)=\sum_jN_j\left[C_A(n)+r_j\big(C_A(n)+C_{A^\dagger}(n)+C_0(n)+C_g(n)\big)+C_{\rm readout}(n)\right],\qquad Q(n)=n+1.
$$

Here each $C$ denotes the compiled cost of the supplied preparation, inverse, zero/good reflection or flag readout, including their actual comparator controls. The analogous serial dependency-depth sum is recorded separately; synthesis, code distance and factory scheduling change physical time and space. No $C_A$ or reflection is treated as a unit-cost oracle.

The declared caps supply the crude source-access bound $N_{\max}(1+2r_{\max})$, with no promise that the caps achieve precision. There is no empirical asymptotic quantum-runtime claim for this particular capped variant. Qubit width is $n+1$; primitive cost depends on the implemented threshold comparator, reflections, power and pinned decomposition.

The optional [physical report](../../assets/generated/amplitude-iterative/physical-resources.json) estimates each unique **full measured trial circuit** with the same illustrative surface-code profile. A whole-recorded-path physical-fault allowance $0.005$ is divided over all actual shots, so every execution receives $0.005/N$. A separate per-output synthesis-model allowance $10^{-5}$ remains an approximation-model scalar rather than a random fault probability.

A serial envelope combines the returned finite per-power Pareto frontiers: peak qubits are the maximum selected allocation and time is the sum of complete per-power trial times weighted by shots. Grouping is accounting only; the recorded adaptive round order is preserved. It assumes factory/layout reconfiguration between groups with unmeasured latency; initialization/reset, classical decision, decoding and compilation times are excluded. It retains the per-group code distances and factory allocations rather than adding logical widths. Every projected circuit's QASM hash is checked against its exported complete measured trial, and the resource adapter is included in the source hashes and snapshots.

With physical error $10^{-4}$, 100 ns gates and 500 ns measurements, the two bounded QDK queries return these recorded-path envelope points:

| Peak physical qubits | Serial modeled runtime, all 1,024 trials |
|---:|---:|
| 17,624 | 0.209408 s |
| 20,352 | 0.183808 s |
| 27,152 | 0.158208 s |
| 34,144 | 0.132608 s |

This is a **conditional recorded-ideal-path projection**. Hardware noise or approximation can change future powers, intervals and stopping, so it does not bound the runtime of every hardware path or certify the adaptive interval. QDK's mean-fit synthesis model provides neither a concrete synthesis certificate nor a runtime-tail guarantee. The [frozen Grover baseline](../grover.md) remains separate and unchanged.

## Verification and formalization

The generator's 10 predetermined threshold cases include exact endpoints, quarter/half examples and neighboring five-bit thresholds. All 10 selected runs reach conclusive intervals and use 7,680 ideal trials in total. Their actual bounded circuit probabilities agree with the independent amplified law within $2\times10^{-14}$. This checks the selected seeds, not guaranteed convergence for every seed or permitted instance.

A separate 224-run sweep covers seven amplitudes with 32 predetermined seeds each, using the same inference engine with an explicitly **model-only analytic flag law**. It models 111,104 shots; all selected runs are conclusive, with zero observed interval misses or conclusive errors above target. These frequencies are diagnostics, not a probability certificate or physical circuit experiment. Full seed/status/interval records distinguish them from the circuit suite.

The [executed notebook](../../notebooks/amplitude_iterative.ipynb) shows the circuit, every default sampled round and the weighted resource ledger. Static publication reuses saved outputs and needs no public kernel backend. Optional local/Binder controls regenerate bounded circuits and trajectories. Executed arithmetic certificates cover endpoint/alias/inversion/radius calculations; **no Lean formalization, verified compiler, iid sampler proof or hardware confidence proof is integrated**.

```bash
python scripts/generate_amplitude_iterative.py --estimate
python -m pytest tests/test_amplitude_iterative.py
python scripts/execute_notebooks.py
```

Circuit construction can also run independently of adaptive statevector inference:

```python
from openquantum.iterative_amplitude_estimation import build_amplified_trial
from openquantum.qft_qpe import lower_circuit
trial = build_amplified_trial(n_data=6, threshold=21, power=16)
compiled = lower_circuit(trial)
```

This bounded seven-wire export was generated and counted at 15,937 primitive operations without statevector validation. Its size is not a claim that every allowed larger input is affordable.

The generator supports explicit accuracy/shot/round/power caps. Circuit generation supports power up to 4,096 with a 250,000 source-instruction guard that counts MCX atomically. Bounded statevector inference supports at most eight data wires and power 128; hitting that bound returns its explicit simulation-cap status. Primitive export is separately capped at 250,000 after lowering, which can consume extra compilation work before the guard. The generator removes stale optional reports and old per-power exports, uses a temporary factory cache outside public assets, records environment/source/lock/RNG hashes and runs without a Git directory in the public code package.

The original implementation uses **Qiskit core 2.6.2** (Apache-2.0) for explicit gates, exports and ideal statevectors {cite:p}`prim_qiskit_2_6_2`, **SciPy 1.18.1** for initial binomial-interval quantile guesses {cite:p}`prim_scipy_1_18_1`, NumPy 2.5.4 for seeded counts and Matplotlib 3.11.2 for original figures. Resource projections use the audited Microsoft QDK 1.34.0 adapter. Source snapshots and the [MIT license](../../assets/generated/amplitude-iterative/LICENSE.txt) accompany the artifacts.

- [Executed adaptive notebook](../../notebooks/amplitude_iterative.ipynb)
- [Complete default sampled trajectory](../../assets/generated/amplitude-iterative/trajectory.json)
- [Full logical ledger and every power-circuit hash](../../assets/generated/amplitude-iterative/logical-resources.json)
- [Explicit endpoint and near-alias validation suite](../../assets/generated/amplitude-iterative/validation-suite.json)
- [Separately labeled model-only coverage sweep](../../assets/generated/amplitude-iterative/coverage.json)
- [Conditional recorded-path physical report](../../assets/generated/amplitude-iterative/physical-resources.json)
