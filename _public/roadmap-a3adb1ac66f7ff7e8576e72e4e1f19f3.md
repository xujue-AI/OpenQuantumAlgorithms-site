---
title: "Roadmap"
description: "Implementation priorities and publication criteria for the handbook."
---

The project grows by completing reproducible examples within a stable chapter structure. The survey {cite:p}`survey2025` guides coverage; implementation order also reflects which algorithms share reusable circuit and estimation components.

## Release sequence

The current release has bounded implementations across the listed algorithm
families: Grover, exact/approximate QFT, textbook/feedback QPE, coherent/adaptive
amplitude estimation, reversible arithmetic and bounded factoring, structured
HHL/QSVT and Neumann LCU inverses, Hamiltonian dynamics and amplification,
channel/Lindblad simulation, thermal preparation, explicit quantum walks and
exact/finite-shot QAOA/VQE with digitized annealing. Their analysis and verification are integrated into
each algorithm article; significant methods have child pages.

Computed physical reports distinguish static arbitrary-rotation mean-fit costs
from a bounded serialized model with actual Clifford+T sequences and reset,
controller and factory accounting. Both retain explicit conditional assumptions.
Certified hardware realization, general application access, scalable thermal
methods and rigorous co-design comparisons remain under development. The frozen
Grover baseline stays fixed.
A pinned Lean artifact checks a finite classical C4 reference; it does not
verify the Python implementation, quantum circuits or physical estimates.

| Stage | Deliverable | Completion condition |
|:---|:---|:---|
| 1. Foundation | Public handbook, Grover example, baseline resource model | The site builds; the small example runs from a clean environment; assumptions and evidence limits are visible |
| 2. Shared primitives | Quantum Fourier transform, phase estimation, reversible arithmetic | Parameterized generators and checks agree with independent small-instance calculations |
| 3. Applications | Order finding and factoring; Hamiltonian simulation | Problem preparation, oracle costs, repetitions, and classical processing appear in the ledger |
| 4. Modern estimation | Linear systems and amplitude estimation | An observable or scalar output is specified, with accuracy and success requirements |
| 5. Comparative studies | Optimization examples; alternative fault-tolerant designs | Baselines and alternatives solve the same task under a documented comparison protocol |
| 6. Verification extensions | Selected formal developments and additional simulators | Artifacts build at pinned revisions and their claim scope is explained |

These stages are implementation milestones. They do not imply that later families are less important or that a quantum advantage is established.

## Coverage beyond the survey

The survey guides the initial coverage; it does not limit later algorithms.
The current literature expansion supplies planned method contracts alongside
the executed examples:

| Track | Added coverage | Next bounded implementation |
|---|---|---|
| [Hamiltonian dynamics](algorithms/hamiltonian-simulation.md#hamiltonian-method-map) | Bounded Suzuki, qDRIFT and [fixed two-term coherent/classical multiproduct circuits](algorithms/simulation/multiproduct.md) checked | Optimized high-order multiproduct, interaction-picture and matched physical studies |
| [Phase estimation](algorithms/qft-qpe.md#important-variants) | Hadamard-test, Bayesian, corrected robust, statistical energy and window/filter methods | Shared controlled-power likelihood experiments with separate coherence and shot accounting |
| [Factoring](algorithms/shor.md#important-variants) | Arithmetic/recovery improvements, Regev-style alternatives, external resource models and Coq/Lean developments | Compare bounded order recovery on the same circuit-generated samples |
| [Decoded quantum interferometry](algorithms/optimization/dqi.md) | Fixed three-variable coherent decoder and public companion checked {cite:p}`frontier_dqi2025` | Structured larger decoder families and Hamiltonian HDQI; synthesis and physical accounting |
| [Detailed-balance Gibbs sampling](algorithms/simulation/detailed-balance-gibbs.md) | Synthetic filtered-jump thermalization {cite:p}`frontier_thermal_chen_2025` | Finite diagonal zero-C channel checked; next: general correction, uniform accuracy and mixing |
| [Reusable bath qubit](algorithms/simulation/reusable-bath.md) | Bounded one-spin gate and finite-channel components checked {cite:p}`frontier_thermal_bath_2026` | Continuum/total accuracy, growing system size and full physical accounting |

These method families include checked increments and **planned** extensions;
the linked pages state their exact evidence. The finite detailed-balance
channel remains partial, with no uniform Gibbs-accuracy or physical gain claim. Thermal convergence retains the papers' temperature,
mixing, access and gap conditions. Factoring proofs remain external until
pinned reproduction and a connection to our generators are recorded. A
fault-tolerant comparison keeps input, output, accuracy, success and physical
assumptions fixed; it cannot inherit another paper's advantage ratio.

## Chapter completion criteria

A chapter is complete when it has:

1. A precise task, access model, input instance, accuracy target, and success criterion.
2. A small circuit generated from the same implementation used for larger sizes.
3. A downloadable notebook and a parameterized code entry point with a pinned environment.
4. A check against a classical result or another independent reference.
5. Logical-resource accounting that includes the oracle or state preparation.
6. A physical-resource estimate with its code, hardware assumptions, compilation choices, and error budget.
7. Paper citations, software attribution, licenses, and a record of the tested version.
8. Motivation and intuition, fundamental complexity or scoped empirical performance, important variants, and size-dependent space/time accounting with other parameters explicit.

An incomplete item remains marked **planned**, **partial**, or **unavailable**. A successful small simulation does not establish scalable performance.

## Reusable components

Early work should create components that later chapters can reuse:

- Circuit generators with explicit register conventions and ancilla cleanup.
- State preparation, controlled operations, and reversible arithmetic.
- A common logical-resource record: qubits, operations, depth, precision, and repetitions.
- A physical-estimation interface that records the model rather than hiding its assumptions.
- A consistent method for saving figures, results, environment versions, and references.

## Baseline before co-design

The initial comparison point is a named surface-code model. Later work may change synthesis, arithmetic, scheduling, factory choices, error correction, or the application representation. Each comparison will preserve the task's input, output, accuracy, and success requirements, then report both the resource change and the extra assumptions.

See [resources](algorithms/grover.md#end-to-end-resource-analysis) for the baseline boundary and [verification](verification.md) for evidence levels.

## References

```{bibliography}
```
