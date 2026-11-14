---
title: "Quantum simulation"
description: "Checked dynamics/amplification, Lindblad channel splitting and bounded diagonal/noncommuting thermal preparation."
---

## Motivation and intuition

Quantum simulation asks for a specified state transformation or an observable of a physical model. This umbrella separates three tasks because their inputs, quantum circuits, accuracy requirements and readout costs differ.

| Track | Task | Current evidence |
|---|---|---|
| [Hamiltonian dynamics](hamiltonian-simulation.md) | Approximate $e^{-iHt}$ or measure a stated observable after evolution | Checked product formulas, [fixed two-term multiproduct circuits](simulation/multiproduct.md), [word-indexed Taylor LCU](simulation/word-taylor.md), [retargetable QSP](simulation/retargetable-qsp.md) and [oblivious amplification](simulation/amplification.md); bounded logical and conditional physical studies |
| [Open systems](simulation/open-systems.md) | Implement a specified quantum channel or Lindblad evolution | Explicit damping, [one-spin splitting](simulation/lindblad.md) and [interacting chains](simulation/lindblad-chain.md); uniform ideal-channel workloads for two, three and four spins, with wider physical studies pending |
| [Thermal/Gibbs preparation](simulation/gibbs-preparation.md) | Prepare a thermal state, its purification, or estimate a thermal observable | Diagonal preparation, [Pauli filtering](simulation/thermal-filter.md), [thermal spectral filtering](simulation/thermal-spectral.md) and [known-input amplification](simulation/thermal-amplification.md); bounded conditional physical reports, with general filtering planned |

Each example fixes the physical model, input, units, requested output and measurement protocol. Hamiltonian dynamics uses an open noncommuting spin chain. Open-system dynamics includes absorbing decay and positive-time splitting of driven and interacting spin models. Thermal preparation separates exhaustive diagonal coefficient loading from a Bell-input filter and a reflection about that known prepared input. These are different tasks and are not an equal-task resource comparison.

The dense classical answers check actual gate/dilation/preparation circuits; they are not quantum algorithm implementations. Parameterized generators and simulators have separate limits. Classical weights for the diagonal thermal example also permit direct classical sampling, so that circuit supplies no efficiency claim.

Postselected circuits include acceptance/retry accounting. Complex QSP and word-Taylor dynamics have actual integer-round oblivious-amplification paths; the thermal filter has a separate known-input amplification circuit that preserves the accepted purification's direction and can overrotate. Deterministic amplification, general scalable thermal filtering and certified synthesis remain planned. Channel and thermal circuits distinguish retained state output from measured trials, count or disclose preparation/reset/discard/readout, and separate approximation, coherent synthesis, statistical error and physical faults. No example supplies a completed fault-tolerant application pipeline or a claimed hardware advantage.

## Fundamental methods and circuit demos

Use [product formulas and the executed spin-chain circuit](hamiltonian-simulation.md#fundamental-method-product-formulas) for unitary evolution; [dilation and its two-wire circuit](simulation/open-systems.md#fundamental-algorithm-and-small-circuit) for an irreversible channel; and [purification and its circuit](simulation/gibbs-preparation.md#fundamental-algorithm-and-small-circuit) for a thermal mixed state. Important methods have their own pages: [structured word-Taylor access](simulation/word-taylor.md), [QSP](simulation/qsp.md), [oblivious amplification](simulation/amplification.md), [one-spin Lindblad splitting](simulation/lindblad.md), [coupled-spin dynamics](simulation/lindblad-chain.md), [noncommuting thermal filtering](simulation/thermal-filter.md) and [known-input thermal amplification](simulation/thermal-amplification.md).

## Further algorithm families and recent thermal methods

The [Hamiltonian method map](hamiltonian-simulation.md#hamiltonian-method-map) extends the
implemented dynamics examples with checked bounded randomized and
[fixed two-term multi-product methods](simulation/multiproduct.md).
Interaction-picture methods, optimized high-order multi-product methods and
matched fault-tolerant architecture comparisons remain planned.

Recent Gibbs preparation has separate method pages. The
[filtered-jump detailed-balance method](simulation/detailed-balance-gibbs.md),
following *Efficient quantum thermal simulation*
{cite:p}`frontier_thermal_chen_2025`, now has checked Gaussian access and a
restricted finite diagonal zero-C reset channel. Its finite mesh bias,
64-round numerical channel powers and eight-round readout projection have
separate contracts; general correction, mixing and physical accounting remain
planned. The
[forward evolution with a reusable bath qubit](simulation/reusable-bath.md),
following *Simple and efficient end-to-end quantum thermal and ground state
preparation* {cite:p}`frontier_thermal_bath_2026`, has bounded one-spin gate and
finite-channel components plus a portable saved notebook. A separate
[local-spin-chain companion](simulation/reusable-bath-local-spin.md) adds
finite n=1/2 checks and a source generator through n=64; its four-round
outputs retain large target biases. General preparation guarantees,
continuum/total accuracy and physical accounting remain planned. Mixing time, temperature,
access and low-energy assumptions remain part of each task. Our diagonal
loader and Bell-input filter do not implement these algorithms, and no
universal efficient low-temperature preparation is claimed.

## End-to-end resource analysis

### Scaling parameters

Hamiltonian dynamics depends on system size, term structure, time and accuracy. Open-system work adds jump operators, channel discretization and fresh environment cost. Thermal preparation adds inverse temperature, normalization and accepted-output probability. The diagonal thermal loader explicitly takes exponential classical work; the tiny noncommuting filter is bounded. Each track keeps its own space/time ledger and simulator limits. Their toy outputs do not establish one common size-dependent physical frontier.

The [Hamiltonian-dynamics report](hamiltonian-simulation.md#end-to-end-resource-analysis) contains logical workloads and bounded conditional physical studies for product formulas and QSP. Structured Taylor and interacting-channel pages also provide prescribed size/time/accuracy workloads. The retargetable-QSP and thermal-spectral pages include actual Clifford+T measured circuits and complete-attempt physical ledgers for one and two spins. All three tracks retain explicit limits; larger physical studies remain pending where reset, conditioning or synthesis contracts are unsupported.

| Track | Required output contract | Costs that must remain explicit |
|---|---|---|
| Hamiltonian dynamics | Unitary/state evolution or a measured observable at a stated time | Preparation, access, approximation, synthesis, postselection/amplification, measurements and repetitions |
| Open systems | A specified channel/state or observable | Environment/dilation preparation, reset/discard, channel approximation and measurement |
| Gibbs preparation | A density operator, purification or thermal observable | Temperature, Hamiltonian access, normalization/overlap assumptions, preparation and success costs |

These are different tasks. A cosine filter is not full complex evolution, and a classical matrix exponential is a reference rather than a quantum preparation algorithm. Align the task and accuracy before comparing costs.

Use separate approximation, sampling and physical-fault budgets. The static arbitrary-rotation adapter supplies conditional mean-fit projections. A separate serialized instrument model charges reset, readout, controller, idle and factory work for its checked bounded circuits. Actual synthesis sequences in that model still rely on numerical guards and stated routing/layout premises; they do not certify hardware confidence. The [fault-tolerance appendix](../appendices/fault-tolerance.md) explains these boundaries.

[Verification](#verification-and-formalization) · [Frozen illustrative hardware profiles](grover.md#end-to-end-resource-analysis)

## Verification and formalization

[Hamiltonian-dynamics verification](hamiltonian-simulation.md#verification-and-formalization) records the available circuit, operator/state, postselection and polynomial checks. Its evidence includes complex QSP and probabilistic amplification; certified physical synthesis, deterministic amplification and a formal proof remain absent.

Open-system circuits are compared with independent Kraus, Bloch-equation and Liouvillian references. Checks preserve the selected input, rates, time, observable and error metric, and distinguish a complete channel from a selected measurement branch. Additional Choi checks test complete positivity and trace preservation on the bounded models.

Thermal preparation compares actual prepared states and observables with normalized matrix-exponential targets at the same Hamiltonian and temperature. Classical preprocessing, normalization, acceptance and purification assumptions remain visible. General many-body channel access and scalable Gibbs preparation remain planned. No reproduced Lean formalization or synthesis certificate is included.

[Physics background](../appendices/quantum-physics.md) · [Mathematics](../appendices/mathematics.md) · [Evidence policy](../verification.md)
