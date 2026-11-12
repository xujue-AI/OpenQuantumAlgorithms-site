---
title: "Quantum simulation"
description: "Checked dynamics/amplification, Lindblad channel splitting and bounded diagonal/noncommuting thermal preparation."
---

## Motivation and intuition

Quantum simulation asks for a specified state transformation or an observable of a physical model. This umbrella separates three tasks because their inputs, quantum circuits, accuracy requirements and readout costs differ.

| Track | Task | Current evidence |
|---|---|---|
| [Hamiltonian dynamics](hamiltonian-simulation.md) | Approximate $e^{-iHt}$ or measure a stated observable after evolution | Checked product formulas, Taylor LCU, complex QSP and [oblivious amplification](simulation/amplification.md); a separate cosine-filter primitive |
| [Open systems](simulation/open-systems.md) | Implement a specified quantum channel or Lindblad evolution | Explicit amplitude-damping dilation and [Hamiltonian/damping channel splitting](simulation/lindblad.md) with Liouvillian checks; general many-body methods planned |
| [Thermal/Gibbs preparation](simulation/gibbs-preparation.md) | Prepare a thermal state, its purification, or estimate a thermal observable | Bounded diagonal preparation and a [noncommuting Pauli thermal filter](simulation/thermal-filter.md); scalable general filtering planned |

Each example fixes the physical model, input, units, requested output and measurement protocol. Hamiltonian dynamics uses an open noncommuting spin chain. Open-system dynamics includes absorbing decay and a driven-qubit Lindblad splitting. Thermal preparation separates exhaustive diagonal coefficient loading from a Bell-input filter for a noncommuting Pauli Hamiltonian. These are different tasks and are not an equal-task resource comparison.

The dense classical answers check actual gate/dilation/preparation circuits; they are not quantum algorithm implementations. Parameterized generators and simulators have separate limits. Classical weights for the diagonal thermal example also permit direct classical sampling, so that circuit supplies no efficiency claim.

Postselected circuits include acceptance/retry accounting, and complex QSP now has an actual integer-round oblivious-amplification path. Deterministic amplification, amplified thermal filtering and certified synthesis remain planned. Channel and thermal circuits distinguish retained state output from measured trials, count or disclose preparation/reset/discard/readout, and separate approximation, coherent synthesis, statistical error and physical faults. No example supplies a completed fault-tolerant application pipeline or a claimed hardware advantage.

## Fundamental methods and circuit demos

Use [product formulas and the executed spin-chain circuit](hamiltonian-simulation.md#fundamental-method-product-formulas) for unitary evolution; [dilation and its two-wire circuit](simulation/open-systems.md#fundamental-algorithm-and-small-circuit) for an irreversible channel; and [purification and its circuit](simulation/gibbs-preparation.md#fundamental-algorithm-and-small-circuit) for a thermal mixed state. Important methods have their own pages: [QSP](simulation/qsp.md), [amplification](simulation/amplification.md), [Lindblad splitting](simulation/lindblad.md) and [noncommuting thermal filtering](simulation/thermal-filter.md).

## End-to-end resource analysis

### Scaling parameters

Hamiltonian dynamics depends on system size, term structure, time and accuracy. Open-system work adds jump operators, channel discretization and fresh environment cost. Thermal preparation adds inverse temperature, normalization and accepted-output probability. The diagonal thermal loader explicitly takes exponential classical work; the tiny noncommuting filter is bounded. Each track keeps its own space/time ledger and simulator limits. Their toy outputs do not establish one common size-dependent physical frontier.

The [Hamiltonian-dynamics report](hamiltonian-simulation.md#end-to-end-resource-analysis) contains implemented logical workloads for product formulas, Taylor-LCU attempts and QSP components. All three tracks now have actual circuit/logical-workload records. Their physical estimates remain pending where reset, conditioning or synthesis contracts are unsupported.

| Track | Required output contract | Costs that must remain explicit |
|---|---|---|
| Hamiltonian dynamics | Unitary/state evolution or a measured observable at a stated time | Preparation, access, approximation, synthesis, postselection/amplification, measurements and repetitions |
| Open systems | A specified channel/state or observable | Environment/dilation preparation, reset/discard, channel approximation and measurement |
| Gibbs preparation | A density operator, purification or thermal observable | Temperature, Hamiltonian access, normalization/overlap assumptions, preparation and success costs |

These are different tasks. A cosine filter is not full complex evolution, and a classical matrix exponential is a reference rather than a quantum preparation algorithm. Align the task and accuracy before comparing costs.

Use separate approximation, sampling and physical-fault budgets. The static arbitrary-rotation adapter supplies only conditional mean-fit projections and cannot currently cost dynamic measurement/reset instruments. The [fault-tolerance appendix](../appendices/fault-tolerance.md) explains these boundaries.

[Verification](#verification-and-formalization) · [Frozen illustrative hardware profiles](grover.md#end-to-end-resource-analysis)

## Verification and formalization

[Hamiltonian-dynamics verification](hamiltonian-simulation.md#verification-and-formalization) records the available circuit, operator/state, postselection and polynomial checks. Its evidence includes complex QSP and probabilistic amplification; certified physical synthesis, deterministic amplification and a formal proof remain absent.

Open-system circuits are compared with independent Kraus, Bloch-equation and Liouvillian references. Checks preserve the selected input, rates, time, observable and error metric, and distinguish a complete channel from a selected measurement branch. Additional Choi checks test complete positivity and trace preservation on the bounded models.

Thermal preparation compares actual prepared states and observables with normalized matrix-exponential targets at the same Hamiltonian and temperature. Classical preprocessing, normalization, acceptance and purification assumptions remain visible. General many-body channel access and scalable Gibbs preparation remain planned. No reproduced Lean formalization or synthesis certificate is included.

[Physics background](../appendices/quantum-physics.md) · [Mathematics](../appendices/mathematics.md) · [Evidence policy](../verification.md)
