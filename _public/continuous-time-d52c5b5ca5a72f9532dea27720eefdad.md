---
title: "Continuous-time walks on the hypercube"
description: "Explicit commuting rotations, independent reference checks, and observable-specific resource accounting."
---

## Motivation and intuition

A continuous-time walk spreads amplitudes by evolving under a graph's adjacency Hamiltonian. On the $n$-dimensional hypercube, a vertex is an $n$-bit string and each edge flips one bit. The adjacency therefore has a particularly simple representation: a sum of Pauli X operators. The terms commute, so every coordinate evolves independently. This supplies a transparent companion to the [coined cycle walk](../quantum-walks.md), whose coin and conditional shifts generate different interference.

Hypercube walks and their mixing behavior are studied by Moore and Russell {cite:p}`walkct_moore_russell2002`. Our explicit convention below fixes the edge weight and time units. This special factorization is also easy to simulate classically; it establishes no quantum advantage or generic graph-access implementation.

## Fundamental algorithm and model

Set $\hbar=1$ and specify

$$H=J\sum_{q=0}^{n-1}X_q,\qquad U(t)=e^{-iHt}.$$

The classical input is the register width $n$, coupling $J$, time $t$, and starting vertex $x$. Prepare $|x\rangle$ with one X gate per set bit.

Here $J$ specifies the energy scale and $t$ is in inverse-energy units with $\hbar=1$. This model-evolution time is separate from the physical duration of the compiled gate circuit. Since distinct-coordinate Pauli terms commute,

$$
U(t)=\bigotimes_{q=0}^{n-1}\left(\cos(Jt)I-i\sin(Jt)X\right)
=\prod_{q=0}^{n-1}R_X(2Jt).
$$

The circuit constructs these actual rotations. It does not append a dense matrix exponential or load graph eigenvectors. Basis state preparation is explicit; arbitrary amplitude loading is outside this example.

For a destination $y$, let $h=\operatorname{popcount}(x\mathbin{\mathrm{xor}}y)$. The analytic probability is

$$P(y,t)=\sin^{2h}(Jt)\cos^{2(n-h)}(Jt),\qquad P(x,t)=\cos^{2n}(Jt).$$

At $|Jt|=\pi/4$, all $2^n$ vertices have probability $2^{-n}$. At $Jt=\pi/2$, the walk reaches the bitwise antipode with amplitude $(-i)^n$. Global phases matter for a coherently controlled evolution: at $Jt=\pi$, $U=(-1)^nI$, not always $I$.

The graph's degree is $n$. Choosing normalized adjacency means $J=1/n$, so the uniform-time value becomes $t=\pi n/4$. Choosing $J=1$ instead gives $t=\pi/4$. Times from differently normalized models must not be compared directly.

## Small circuit and independent check

The canonical generated example has $n=3$, $J=1$, $t=.7$ and $x=000$. Its unitary consists of three RX rotations; the complete measured trial adds all three position readouts. The notebook separately checks the uniform distribution at $t=\pi/4$, where the return probability is $1/8$.

```{embed} #walk-continuous-circuit
```

```{embed} #walk-continuous-results
```

The circuit is checked with Qiskit Statevector and Operator. An independent reference constructs bit-flip adjacency edges and exponentiates that small matrix with SciPy. A separate Kronecker-matrix implementation is used in tests. Residuals are recorded for the generated parameters; these are floating-point checks, not formal or synthesis certificates.

## End-to-end resource analysis

For one trial, logical space is $n$ qubits. Input preparation uses $\operatorname{popcount}(x)$ X gates, evolution uses $n$ RX gates for nonzero represented $Jt$, and complete position readout uses $n$ measurements. The coordinate rotations can form one logical layer; factory throughput and error-correction scheduling still determine physical time. The ability to use $n$ rotations for arbitrary represented time is a consequence of this commuting Hamiltonian, not a fast-forwarding method for arbitrary graphs.

Our resource task is **one fixed return-probability mean**, obtained from the indicator $[y=x]\in[0,1]$. It is different from reconstructing a uniformly accurate histogram over all $2^n$ vertices. Independent Bernoulli shots obey

$$M\ge\left\lceil\frac{\log(2/\delta_S)}{2\epsilon_S^2}\right\rceil.$$

For $\epsilon_S=.03$ and $\delta_S=.005$, this requires **3,329 complete trials**. The canonical generator passes a separate $10^{-4}$ allowance for synthesis bias, giving an absolute-error target .0301. A separate .005 physical-fault allowance spans all trials; it is not added to the coherent bias as an independent random fault. The helper's configurable default synthesis allowance is .01; that different default is not the published canonical setting.

The ledger must multiply preparation, rotations and every position measurement by $M$. Classical decoding compares each $n$-bit output with $x$; its time and communication latency are unmeasured. The clean zero input is assumed, with initialization timing/faults excluded. The parent chapter's conditional surface-code reports specify code distances, factories, hardware error rates and complete serial workloads. A raw RX count or constant logical depth does not determine physical qubits or runtime.

QDK's arbitrary-rotation cost is an empirical mean-fit model. It does not supply a concrete Clifford+T approximation or certified output-error bound. The synthesis-bias allocation is therefore conditional on the model scalar bounding output-distribution variation. No certified overall output-confidence statement follows from that assumption. The [frozen Grover baseline](../grover.md#end-to-end-resource-analysis) supplies reporting conventions and hardware profiles, not physical counts for this walk.

## Verification and formalization

Checks retain complex amplitudes, global phases, nonzero starting vertices, negative couplings/times, normalized versus unnormalized time conventions, final measurement maps, and lowering equivalence. Uniformity, antipodal transfer, periodicity and dense adjacency structure have independent references. A deliberately disagreeing reference is rejected by the checker. Results must meet a declared $10^{-12}$ numerical tolerance before they are labeled checked.

No Lean formalization is integrated for this implementation. The factorization above is a mathematical explanation; a proof artifact would need a reproduced toolchain, theorem assumptions and an explicit circuit connection. General continuous-time graph walks additionally require implemented graph access and approximation analysis; they are not supplied by this hypercube circuit.

The implementation uses original MIT code with [Qiskit 2.6.2](https://github.com/Qiskit/qiskit/tree/2.6.2) ([Apache-2.0 license](https://github.com/Qiskit/qiskit/blob/2.6.2/LICENSE.txt)), NumPy and SciPy. Exact build versions, source hashes and circuit digests accompany the generated artifacts.

Optional background video: [MIT Lecture 12: Quantum Dynamics](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/resources/lecture-12-quantum-dynamics/) explains unitary time evolution. It supports the Hamiltonian viewpoint rather than validating this walk implementation or its resource estimate.

## Generate without simulating

```python
import math
from openquantum.quantum_walks_continuous import (
    build_continuous_walk, validate_continuous_walk,
)

small = build_continuous_walk(3, 0.7, measure=True)
print(validate_continuous_walk(time=0.7)["return_probability"])

larger = build_continuous_walk(32, time=0.5, measure=True)
# This creates gates without allocating a 2**32-entry statevector.
```

Generation is capped at 64 position qubits. A full analytic probability vector is separately capped at 16, statevector checks at 12 and dense graph references at eight. Dense matrix exponentiation also has a separate scaled-exposure bound $n|Jt|\le100$. Larger generated circuits are not labeled simulated. One specified analytic vertex probability can be evaluated without the full vector, reflecting this model's classical simplicity.
