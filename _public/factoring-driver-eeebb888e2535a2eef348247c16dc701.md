---
title: "A complete bounded factoring driver"
description: "Random bases, recycled-control phase estimation, failed trials, verified factors and explicit stopping limits."
downloads:
  - file: ../../notebooks/shor_driver.ipynb
    title: "Executable driver notebook"
---

A successful fixed-base order measurement is only one part of factoring. A driver must also recognize easy classical cases, choose bases, retain failed measurements, verify proposed factors and stop honestly when its budget is spent. This page implements that full control flow for bounded inputs, using the actual modular arithmetic from the [main chapter](../shor.md).

:::{note} Status and output contract

The driver returns a **verified nontrivial factor split**, an exact bounded prime classification, or an explicit failure/resource limit. A recursive wrapper continues splits until all factors are prime or some components remain unresolved. The circuits and traces below are executed ideal simulations; no generic factoring confidence theorem or adaptive fault-tolerant projection is supplied.
:::

## Classical preparation and random bases

All inputs are integers $2\le N\le4095$. The driver checks small primes, even inputs and exact perfect powers using integer roots. It then uses a deterministic two-base Miller–Rabin implementation in this bounded domain. The dedicated tests compare every input in the domain against an independent sieve; this is not an unbounded primality claim. Strong-pseudoprime tests provide the broader context {cite:p}`driver_jaeschke_1993`.

For an odd composite that survives these checks, bases are drawn **uniformly with replacement from $2,\ldots,N-1$** using a recorded NumPy seed. A nontrivial $\gcd(a,N)$ is an immediate classical factor. Coprime bases receive a bounded number of complete quantum trials. Every outcome is processed by continued fractions, checked with $a^d\bmod N$, and followed by gcd factor verification. The driver never enumerates the order to construct a circuit, sample a result or decide when to stop.

## Recycled-control phase estimation

The semiclassical inverse Fourier transform replaces the exponent register with one measured and recycled control qubit {cite:p}`shor_griffiths_niu_1996`. Our original Qiskit 2.6.2 implementation uses actual `reset`, measurement and classical `if_test` operations. Qiskit is Apache-2.0 software; [upstream](https://github.com/Qiskit/qiskit) is credited separately from our MIT code.

At output bit $i=0,\ldots,t-1$, apply H, controlled multiplication by $a^{2^{t-1-i}}\bmod N$, and the feedback phases

$$P\!\left(-\frac{\pi b_j}{2^{i-j}}\right),\qquad j<i,$$

where $b_j$ is a previous measured bit. Apply H and measure $b_i$, then reset the control before the next step. The output integer is $y=\sum_i b_i2^i$, so bits arrive least significant first. All registers reset and the modular-power register is prepared at one at the beginning of **each** trial. Accumulator and borrow work is uncomputed on every branch.

```{embed} #shor-driver-circuit
```

This four-stage illustration uses $N=15,a=2,t=4$. The complete driver defaults to $t=2n$, giving eight stages for $N=15$ and ten for $N=21$. That grid-resolution rule supports continued-fraction reconstruction of suitable samples; it does not guarantee a favorable period or factor on every trial.

## Executed examples and failed work

| Input and policy | Observed result | Complete quantum trials |
|---|---|---:|
| $N=21$, seed 1, default caps | Base 10, outcome 853/1024, verified period six, factors 3 and 7 | 1 |
| $N=15$, seed 1, three bases/one shot per base | Two failed order samples, then base 6 gives a classical gcd factor | 2 |
| $N=15$, seed 20261112, the same three-base cap | Three failed quantum trials, explicit budget exhaustion | 3 |

```{embed} #shor-driver-trace
```

The default $N=21$ path uses **13 logical qubits**, ten measurements and **22 resets**: thirteen initial preparations and nine control recycles. Its lowered observed path contains **75,799 primitive operations**, including arithmetic, feedback, preparation and readout. The two failed $N=15$ trials in the retry example still charge **17,302 operations**, sixteen measurements and 36 resets. Classical success on a later base does not erase that earlier quantum work.

The recursive wrapper verifies a prime/unresolved-product invariant throughout. The notebook factors 225 into $3,3,5,5$: perfect-power preprocessing alone first gives $15\times15$, and both composite components are subsequently resolved. Recursive runs have separate recorded seeds and one shared global quantum-trial cap. An incomplete run preserves its unresolved components instead of labeling the result a complete prime factorization.

## Whole-policy success and scaling

Let $B$ be the maximum base draws, $S$ the trials allowed per coprime base and $K$ the global quantum-trial cap. The driver follows all three caps, including cases where $K<BS$. We provide a separate **numerical ideal-policy audit** for $N\le63$ using an independent reference distribution; the actual driver does not call that reference.

For $b$ remaining bases and $q$ remaining quantum trials, the audit averages failures over all coprime bases. A noncoprime base contributes zero failure. For a coprime base with ideal success $p_a$, use $u=\min(S,q)$ trials; its failure contribution is $(1-p_a)^u$ times continuation $F(b-1,q-S)$ when $q\ge S$, or a terminal failure when $q<S$. Also $F(0,q)=1$. A cap reached between bases still permits the next base's classical gcd check; a cap reached inside a base terminates that run. This distinction is tested against independent finite-policy enumeration.

For example, $N=15,t\ge4$ has six immediately successful noncoprime bases, six coprime bases with $p_a=1/2$, and base 14 with zero factor probability. An uncensored $B=8,S=2,K=16$ policy has modeled ideal failure $(5/26)^8$. The calculation is numerical and conditional on independent fresh trials, exact arithmetic/angles and the declared base policy; it is **not** a synthesis, physical-fault or generic algorithm-confidence certificate.

Recycled-control width is $2n+3$, instead of $t+2n+2$ for the full exponent register. Each trial retains $O(tn^3+t^2)$ logical work with the exact Fourier arithmetic, plus $t$ measurements and $2n+t+2$ resets. A run with $R\le K$ completed trials charges every complete path; classical base selection, integer-root tests, modular exponentiation and gcd processing use integer arithmetic and are timed separately. The recursive wrapper permits at most $2n-1$ component calls, within its explicit domain cap. Seeded successful examples establish neither a quantum advantage nor a general runtime bound.

## Fault-tolerant analysis and verification

The [main chapter's surface-code report](../../assets/generated/shor/surface-code.json) models a different, fixed full-register circuit. It cannot be reused as this driver's physical estimate. The current shared QDK adapter rejects resets and intermediate/classically controlled measurements; omitting them would remove work required by the implementation. A complete adaptive estimate still needs initialization/reset error and timing models, measurement-conditioned schedules, classical feedback latency, approximation budgets and faults across all possible trials. Physical resources are therefore **pending**, while the recorded logical ledger is executed and complete for each observed ideal path.

Thirty-two dedicated tests cover the full bounded primality domain, exact roots, actual adaptive versus full-register circuit distributions, cap transitions, failed work, random-base success and recursive product invariants. Independent review additionally checked 54 exact policy comparisons and canonical sampled-path likelihoods. A branch verifier retains all positive-probability paths without truncation; raw state norms are checked and their floating-point normalization is recorded. This is numerical verification, not a theorem. No Lean formalization has been built for this driver.

## Reproduce and modify

```bash
python scripts/generate_shor_driver.py --recursive
python scripts/generate_shor_driver.py --modulus 15 --seed 1 \
  --max-base-draws 3 --shots-per-base 1 --max-quantum-trials 3 \
  --output-dir artifacts/shor-retry
pytest tests/test_shor_driver.py
```

The [notebook](../../notebooks/shor_driver.ipynb) provides bounded live controls in a local/Binder kernel. The static page requires no maintained server. Sampling is capped at 18 quantum wires; circuit construction has its own conservative operation guard. Branch enumeration has an immutable six-output-bit cap and a separate maximum-amplitude allocation. Unsupported quantum components are reported explicitly, even when their classical preliminary checks were supported.

- [Default driver trace, program hashes and numerical policy audit](../../assets/generated/shor/driver/workload.json)
- [Actual lowered adaptive program for base 10](../../assets/generated/shor/driver/adaptive-a10-t10.qasm)
- [Failed trials followed by classical gcd success](../../assets/generated/shor/driver/retry/workload.json)
- [Explicit exhausted run](../../assets/generated/shor/driver/exhausted/workload.json)
- [Original driver source](../../assets/generated/shor/driver/shor_driver.py) and [MIT license](../../assets/generated/shor/driver/LICENSE.txt)

Exported QASM retains all required reset/measurement/feedback operations. The pinned exporter omits overall and classically selected branch global phases; these are measured instrument specifications, not phase-preserving gates for a new coherent control.

## References

```{bibliography}
```
