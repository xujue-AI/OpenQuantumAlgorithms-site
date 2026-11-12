---
title: "Roadmap"
description: "Implementation priorities and publication criteria for the handbook."
---

The project grows by completing reproducible examples within a stable chapter structure. The survey {cite:p}`survey2025` guides coverage; implementation order also reflects which algorithms share reusable circuit and estimation components.

## Release sequence

| Stage | Deliverable | Completion condition |
|:---|:---|:---|
| 1. Foundation | Public handbook, Grover example, baseline resource model | The site builds; the small example runs from a clean environment; assumptions and evidence limits are visible |
| 2. Shared primitives | Quantum Fourier transform, phase estimation, reversible arithmetic | Parameterized generators and checks agree with independent small-instance calculations |
| 3. Applications | Order finding and factoring; Hamiltonian simulation | Problem preparation, oracle costs, repetitions, and classical processing appear in the ledger |
| 4. Modern estimation | Linear systems and amplitude estimation | An observable or scalar output is specified, with accuracy and success requirements |
| 5. Comparative studies | Optimization examples; alternative fault-tolerant designs | Baselines and alternatives solve the same task under a documented comparison protocol |
| 6. Verification extensions | Selected formal developments and additional simulators | Artifacts build at pinned revisions and their claim scope is explained |

These stages are implementation milestones. They do not imply that later families are less important or that a quantum advantage is established.

## Chapter completion criteria

A chapter is complete when it has:

1. A precise task, access model, input instance, accuracy target, and success criterion.
2. A small circuit generated from the same implementation used for larger sizes.
3. A downloadable notebook and a parameterized code entry point with a pinned environment.
4. A check against a classical result or another independent reference.
5. Logical-resource accounting that includes the oracle or state preparation.
6. A physical-resource estimate with its code, hardware assumptions, compilation choices, and error budget.
7. Paper citations, software attribution, licenses, and a record of the tested version.

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

See [resources](resources.md) for the baseline boundary and [verification](verification.md) for evidence levels.

## References

```{bibliography}
```
