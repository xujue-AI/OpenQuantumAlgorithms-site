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

## Fault-tolerant analysis

There is no separate physical estimate for the truncated family in this increment. The omission allowance consumes coherent algorithm accuracy; the retained continuous rotations need a separate synthesis allowance. [Static QPE projections](../qft-qpe.md#end-to-end-resource-analysis) use the exact inverse QFT and cannot be relabeled as approximate-QFT estimates. The frozen Grover physical baseline remains unchanged.

```python
from openquantum.qft_qpe import build_qft, qft_omission_bound
circuit = build_qft(32, max_distance=14)
print(qft_omission_bound(32, 14))
```

The code, environment, implementation license and [executed notebook](../../notebooks/qft_qpe.ipynb) are shared with the parent chapter.
