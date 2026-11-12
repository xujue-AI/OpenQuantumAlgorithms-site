---
title: "Qubitization and quantum signal processing"
description: "Explicit block encoding, a cosine primitive and postselected complex Hamiltonian evolution."
downloads:
  - file: ../../notebooks/simulation_qsp_evolution.ipynb
    title: "Complex QSP notebook"
---

## Motivation and intuition

A product formula approximates evolution by many short term rotations. Quantum signal processing instead transforms a block encoding of the Hamiltonian into a polynomial in its normalized eigenvalues. Ancilla phase rotations choose the polynomial; repeated accesses apply it to every spectral component without loading eigenvectors. Qubitization supplies the invariant two-dimensional spaces on which that signal-processing picture applies {cite:p}`sim_qubitization_2019,sim_qsp_2017`.

This page belongs to [Hamiltonian dynamics](../hamiltonian-simulation.md). It distinguishes the checked cosine-filter primitive, the complex cosine/sine combination, and [oblivious amplification](amplification.md). Their normalization and success costs differ.

## Small circuit: a degree-two polynomial

The [small QSP notebook](../../notebooks/qsp_basics.ipynb) applies two explicit block-encoding calls and phases $(-\pi/2,\pi/2,0)$ to $H=0.6Z+0.4X$. Its zero-ancilla block is checked against $T_2(H)=2H^2-I=0.04I$. Acceptance is only $0.0016$ for every system input; this example demonstrates the polynomial mechanism rather than efficient dynamics. The drawn circuit is lowered into actual gates.

```{embed} #qsp-small-circuit
```

## Qubitization and QSP cosine filtering

The five Hamiltonian terms have $\alpha=\sum_j|c_j|=4.1$. Explicit PREPARE and signed Pauli SELECT give the Hermitian involution

$$B=\mathrm{PREPARE}^\dagger\mathrm{SELECT}\,\mathrm{PREPARE},\qquad\langle0|B|0\rangle=H/\alpha.$$

Tests check the actual block. A reflection about the zero-label subspace creates the qubitization walk and its invariant-subspace identity is checked {cite:p}`sim_qubitization_2019`.

The QSP module documents its scalar matrix convention and phase ordering. Fourteen calls to $B$ and fifteen frozen phases produce a degree-14 real polynomial approximating $\cos(4.1x)/2$ on $[-1,1]$. A flag coherently combines the $+\phi$ and $-\phi$ branches to extract the real part, sharing block calls. The phase JSON records a deterministic SciPy fit, independent dense-grid residual, Chebyshev-coefficient discrepancy and analytic polynomial-tail bound. Tests compare the **actual circuit block** to $\cos(H)/2$, separately from the scalar phase check {cite:p}`sim_qsp_2017,sim_qsvt_2019`.

```{embed} #simulation-qsp-results
```

```{embed} #simulation-qsp-polynomial
```

The circuit-block error is approximately $3.40\times10^{-9}$ and ideal acceptance on $|000\rangle$ is $0.1782034$. These are cosine-filter results: this primitive has conditional normalization and does not itself implement $e^{-iH}$. A single parity-constrained real polynomial does not supply both even cosine and odd sine. The separate complex construction below combines the two parities. Neither phase fit certifies arbitrary rotation synthesis.

## Complex QSP dynamics with explicit postselection

The next circuit completes the cosine/sine combination for the fixed dynamics task. Freeze an even degree-14 phase sequence for $C(x)\approx\cos(4.1x)/2$ and an odd degree-15 sequence for $S(x)\approx\sin(4.1x)/2$. A uniform selector and a relative $-i$ phase on its sine branch give the zero-selector block

$$K(x)=\frac{C(x)-iS(x)}{2}\approx\frac{e^{-i4.1x}}{4}.$$

This is the QSP/LCU normalization, not a deterministic unitary on the system {cite:p}`sim_qsp_2017,sim_qsvt_2019,sim_qsp_lcu_unification_2021`. The generator shares fourteen ordinary $B$ calls between the branches, adds one actual selector-controlled PREPARE/SELECT/inverse-PREPARE call, and implements every selector-conditioned projector phase. It does not append a matrix exponential gate. All three label bits, the real-part flag and the selector must be zero to accept; all relative branch phases are included in the circuit and in decomposition checks.

```{embed} #simulation-complex-qsp-polynomial
```

The measured trial uses eight logical wires: three system spins, three access labels, one real-part flag and one selector. All wires are freshly initialized for each complete attempt. Final measurements retain the system bits and select only the zero ancillary outcomes. The conditional-state check describes the system **before** final system readout; the exported complete trial produces measured outcomes.

```{embed} #simulation-complex-qsp-results
```

The actual gate-circuit block has spectral error about $1.70\times10^{-9}$ against $U/4$. Rescaling that block gives $\|4K-U\|_2\approx6.79\times10^{-9}$; this is a linear-map check, not a claim that $4K$ is exactly unitary. Conditional-state error is about $6.07\times10^{-9}$, and the $Z_0$ bias is about $1.15\times10^{-9}$. Tests also exercise complex input amplitudes, negative coefficients and Y-containing Hermitian Hamiltonians. Global-phase-aligned tests on separate branches would miss an incorrect relative phase, so the combined branch and primitive matrices are compared directly.

```{embed} #simulation-complex-qsp-state
```

The ideal exactly scaled unitary would accept with probability $1/16$. The simulated $|000\rangle$ probability is $0.0624999999393$; the smallest singular value of the finite Hamiltonian's accepted block gives a numerical minimum acceptance of about $0.0624999994691$. These are small-instance numerical checks. The Bessel-tail formula is an analytic truncation bound, while floating phase fits, coefficient reconstruction and the chosen reconstruction guard are numerical evidence. Any interval-wide accuracy/acceptance projection and its retry budget are explicitly **conditional**, with no coefficient or synthesis certificate.

```{embed} #simulation-complex-qsp-ledger
```

At the declared conditional acceptance, 72 complete attempts cap the chance of obtaining no accepted measured trial at 1%. Observable estimation needs many accepted trials and a separate capped-retry and sampling allocation. The generator includes an illustrative seeded classical draw from the ideal joint circuit probabilities; it is not a hardware run. The logical ledger charges the complete coherent access and readout on each attempt, separately from fresh initialization and pending physical timing.

Frozen phases target $\alpha t=4.1$. The parameterized generator rejects phases that fail the requested scalar target on its 10,001-point grid; changing time requires verified phases rather than silently reusing these ones. Source materialization caps terms, phase degree, system wires and operation count separately from the eight-total-wire dense-Operator and twelve-total-wire statevector checks.

The separate [oblivious-amplification increment](amplification.md) implements actual forward/inverse accesses and ancilla reflections, with success and approximation checks. Integer round 3 still has nonzero failure probability; **deterministic evolution remains planned**. Rotation synthesis and physical faults need their own postselection-sensitive precision contract. No physical-resource improvement is claimed relative to product formulas or Taylor LCU.

## Complexity and end-to-end resource analysis

For an explicit Pauli Hamiltonian $H=\sum_{j=0}^{L-1}c_jP_j$, the input normalization is $\alpha=\sum_j|c_j|$. Let $C_B(n,L)$ be the elementary cost of this page's actual PREPARE/SELECT/unprepare access and $C_\Pi$ the success-register reflection/phase cost. A degree-$d$ sequence uses $d$ block accesses and $d+1$ projector phases. Thus its circuit cost scales as $dC_B+(d+1)C_\Pi$, plus input preparation, parity selection and readout. A query count alone omits both access costs.

The implemented label-tree preparation and indexed Pauli SELECT explicitly enumerate the supplied terms. Label width grows as $\lceil\log_2 L\rceil$, while controlled gate decomposition and state-loading work grow with the actual term list. This is a bounded explicit-access generator, not an assumed constant-time Hamiltonian oracle. For a local spin chain $L=O(n)$ and $\alpha$ grows with $n$ at fixed couplings; fixed time therefore changes $\alpha t$ and the required polynomial/phase sequence. Reusing degree 14/15 at every size would not preserve accuracy.

The downloaded [complex circuit ledger](../../assets/generated/simulation/qsp-evolution/logical-resources.json) counts the eight-wire demo and capped complete attempts. The [amplification ledger](../../assets/generated/simulation/amplification/logical-resources.json) counts all forward/inverse accesses, reflections and rejected trials. At $r$ integer amplification rounds, the coherent access is used $2r+1$ times and the actual success projection is reflected $2r$ times. Round count is a separate parameter from Hamiltonian size and degree.

The fixed demo's numerical approximation and acceptance do not certify arbitrary $n$. A size-dependent physical frontier requires validated phase/approximation and postselection-sensitive synthesis allowances, a declared output observable and shot budget, and physical-fault allocation across every attempt. That frontier remains pending. The [in-page dynamics resource contract](../hamiltonian-simulation.md#end-to-end-resource-analysis) records these boundaries.

## Verification and formalization

Actual circuit blocks are compared with independent spectral cosine and matrix-exponential references, while scalar response, coefficient reconstruction and truncation evidence are recorded separately. Additional negative/Y coefficients, complex inputs and nonnormal amplification tests check phase and conditioning behavior. These are bounded numerical checks. No reproduced Lean theorem or certified synthesis sequence is included.

[Cosine notebook](../../notebooks/hamiltonian_simulation.ipynb) · [Complex QSP notebook](../../notebooks/simulation_qsp_evolution.ipynb) · [Frozen phases](../../assets/generated/simulation/qsp-evolution/phases.json) · [Measured primitive circuit](../../assets/generated/simulation/qsp-evolution/evolution.qasm)
