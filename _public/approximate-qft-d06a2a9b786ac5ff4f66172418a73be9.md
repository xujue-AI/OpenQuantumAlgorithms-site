---
title: "Approximate Fourier transform"
description: "Small-angle omission with an explicit circuit and spectral-norm bound."
---

## Motivation and construction

An exact $m$-wire Fourier circuit has quadratic controlled-phase count. Its longest-range phases are exponentially small. Omitting those phases reduces gate construction and eventual synthesis work at a declared coherent approximation cost, following the approximate-transform idea of Coppersmith {cite:p}`prim_approx_qft_1994`.

`build_qft(m, max_distance=b)` retains the Hadamards and final swaps, and keeps only controlled phases with wire separation $d\leq b$. A retained gate has angle $\pi/2^d$. The retained count is $\sum_{d=1}^{b}(m-d)$, giving $O(mb)$ controlled phases rather than $O(m^2)$. Setting `b=m-1` is the exact circuit; `b=0` retains only Hadamards and swaps.

## Approximation and scaling

For an omitted phase $\theta$, $\|CP(\theta)-I\|_2=2|\sin(\theta/2)|\leq|\theta|$. A telescoping product of unitary gates therefore gives the original bound

$$
\|\widetilde F_N-F_N\|_2\leq
\min\!\left(2,\sum_{d=b+1}^{m-1}(m-d)\frac{\pi}{2^d}\right)
\leq\min(2,\pi m/2^b).
$$

The inverse has the same bound. Choosing $b\geq\lceil\log_2(\pi m/\eta)\rceil$, capped at $m-1$, meets a sufficient spectral-norm omission allowance $\eta$. This bound covers the omitted ideal phases. It does not cover inaccuracies in retained rotations, physical faults or application readout confidence.

The [generated table](../../assets/generated/qft-qpe/qft-scaling.csv) compares explicit source circuits. At $m=32$, cutoff $b=14$ retains 343 controlled phases instead of 496, with telescoping bound about $0.003067963$. These larger circuits were generated and counted, not densely simulated. Logical count reduction alone does not establish physical improvement: retained rotation synthesis and factory scheduling must be compared under the same application and accuracy contract.

## Small demo and verification

```{embed} #qft-approximation
```

The notebook compares a five-wire truncated circuit with an independent Fourier matrix and checks its spectral error against the formula. Dedicated tests cover every cutoff for one through five wires, while exact circuits through six wires match the defining matrix. Dense matrix checks stop at eight wires. No integrated Lean proof or per-angle synthesis certificate is supplied.

## End-to-end resource analysis

A QFT-only unitary approximation does not establish confidence in an application output. This comparison uses a complete QPE task: prepare $|1\rangle$, apply the actual phase gate $U=P(2\pi\times0.3)$ through repeated controlled calls, apply exact or truncated inverse QFT, read the phase register and take the scalar median of fresh trials. The phase is known to the constructor, so this is a circuit/resource example rather than an application advantage.

The predeclared grid uses $m=5,6,7$ phase wires, exact cutoff $b=m-1$ and truncated cutoff $b=m-2$. One longest-distance controlled phase is omitted in each truncated circuit. Within each pair, the target phase, target width, absolute phase error $\epsilon_\phi=2^{-m}$, whole-run failure target 0.01, synthesis allowance and hardware profile remain fixed. Two synthesis-model allowances, $10^{-4}$ and $10^{-5}$ per trial, and both illustrative physical error profiles are retained.

### Omission and the matched output contract

The spectral omission bound $\eta$ implies output-distribution total variation at most $\min(1,\eta)$: pure-state trace distance is at most the state-vector norm difference, and measurement contracts distance. Omission therefore worsens a conservative **single-trial bad-output bound**. It is not an additive numerical phase bias and is not subtracted from $\epsilon_\phi$.

Using $\pi<22/7$, the ordinary two-adjacent-grid-point QPE theorem gives a conservative rational single-run bad bound $23/121$. With a retained-rotation synthesis-TV allowance $\epsilon_s$, use

$$q_{\rm bad}\le23/121+\eta_{\rm TV}+\epsilon_s.$$

The dyadic omission sum uses the same rational upper bound for $\pi$. It is $11/56$, $11/112$ and $11/224$ at $m=5,6,7$. Exact Fraction arithmetic chooses the least odd number $R\le1001$ of fresh complete trials whose binomial majority tail is at most 0.0049. A single-run bound at least one half or an insufficient cap returns an error instead of weakening the confidence target. The remaining whole-run physical-fault allowance is 0.005, leaving the configured total below 0.01.

The scalar median requires the phase to lie at least $2^{-m}$ from the $[0,1)$ branch cut. These numerical phase circuits, synthesis-TV interpretation, IID fresh trials, physical fault model and correct classical decoding are assumptions. Exact rational budget arithmetic does not certify binary64 phase realization, a concrete synthesis sequence, synthesis runtime tails or hardware confidence.

```{embed} #qft-variant-truncated-circuit
```

```{include} ../../assets/generated/qft-variant-comparison/summary.md
```

Every full trial pays input preparation, $2^m-1$ controlled $U$ calls, retained inverse-QFT phases, swaps and $m$ phase readouts. All $R$ fresh trials are included. For this fixed access model, the logical cost is

$$R\left[C_{\rm prep}+(2^m-1)C_{cU}+C_{F^\dagger}(m,b)+C_{\rm readout}(m)\right],\qquad Q=m+1.$$

### Computed conditional physical result

The complete measured circuits use the same compilation settings and QDK adapter. Both physical profiles use 100 ns gates, 500 ns measurements, rotated surface code and round-based factories. The 0.005 whole-run physical allowance is divided across all fresh trials. Changing $R$ can change the required data distance or factory allocation; that effect is included in each returned frontier.

The 24 bounded queries return 123 Pareto points. Omitting one gate saves five primitive operations per trial, but the conservative confidence contract increases repetitions from 15 to 127, 35 or 21. At each case's fastest returned point, truncated/exact total serial-runtime ratios range from about **1.40 to 9.75**. This grid therefore shows no total-runtime improvement under its declared omission/inference policy.

![Same-output conditional runtime ratios at the fastest returned points](../../assets/generated/qft-variant-comparison/physical-comparison.svg)

Observed output-TV losses at phase 0.3 are much smaller than the uniform bounds: approximately 0.00907, 0.00343 and 0.000463. Omission slightly improves this selected phase's good-output mass at seven bits. These diagnostics do not choose the repetition policy. A sharper independently established application-specific omission or inference contract could change the comparison; no such improvement is validated here.

All Pareto points, code distances, factory choices, canonical counts and QASM/source hashes are retained. Peak qubits times total runtime is a rectangular space-time proxy. The QDK mean synthesis fit supplies no worst-case per-angle approximation or runtime-tail certificate. Physical initialization/reset, external loading and classical median/controller/decoder/verification latency are excluded. The existing Grover baseline remains unchanged.

### Reproduce and verify

```bash
python scripts/generate_qft_variant_comparison.py --estimate
python -m pytest tests/test_approximate_qft_resources.py
```

The new [executed notebook](../../notebooks/qft_variant_comparison.ipynb) checks actual complete circuits, allocation tables and ideal distributions, and reuses the saved physical reports. Optional bounded live controls do not reuse physical estimates after changing the declared phase. The [public code package](../../assets/generated/openquantum-code.zip) includes the generator, its dependencies and tests.

- [Predeclared study and exact allocation records](../../assets/generated/qft-variant-comparison/study.json)
- [Full logical/access and simulation table](../../assets/generated/qft-variant-comparison/logical.csv)
- [Every physical Pareto point](../../assets/generated/qft-variant-comparison/physical-pareto.csv)
- [Matched fastest-point comparisons](../../assets/generated/qft-variant-comparison/matched-comparison.csv)
- [All source and download hashes](../../assets/generated/qft-variant-comparison/manifest.json)

The implementation and figures are original. They use pinned Qiskit core 2.6.2, NumPy 2.5.4, Matplotlib 3.11.2 and the audited QDK 1.34.0 adapter; source snapshots and the MIT implementation license accompany the reports. Existing approximate-QFT and QPE references apply. No Lean proof or certified hardware implementation is integrated.

## Generate a larger transform

```python
from openquantum.qft_qpe import build_qft, qft_omission_bound
circuit = build_qft(32, max_distance=14)
print(qft_omission_bound(32, 14))
```

The code, environment, implementation license and [executed notebook](../../notebooks/qft_qpe.ipynb) are shared with the parent chapter.
