---
title: "QFT and phase estimation: resource analysis"
description: "Planned precision, controlled-evolution, and physical-resource accounting."
---

**Status: resource contract planned.** A complete fault-tolerant report will accompany the [implementation](../qft-qpe.md) after its circuit, precision rule, and output procedure are reviewed.

## Specify the task

For a transform, record register width, forward/inverse convention, and an operator approximation target. For phase estimation, record the implemented unitary, requested phase precision, failure probability, and how the input eigenstate is prepared. If the input is a superposition of eigenstates, its overlap with the desired eigenstate affects the probability of a useful output and must enter repetition accounting.

## Charge the complete circuit

Count preparation, controlled powers of the unitary, inverse Fourier processing, ancillas, measurements, and classical phase interpretation. Powers represented by a high-level gate still require an actual implementation. Approximate Fourier transforms must state which rotations are removed and justify the associated error. Remaining arbitrary rotations need their own synthesis allowance.

Conventional and iterative phase estimation have different control and measurement structures. The current static resource adapter accepts bound circuits with final measurements; mid-circuit resets and feedback require an extended, branch-aware model. An iterative circuit cannot silently reuse a static import that omits such work.

## Physical report to publish

The ledger will separate phase/algorithm approximation, synthesis approximation, useful-output probability, sampling failure, and modeled physical faults. It will publish logical counts plus conditional surface-code qubit/time frontiers under named profiles. Mean-fit arbitrary-rotation costs require additional approximation evidence before any certified bound is claimed.

Use the [fault-tolerance appendix](../../appendices/fault-tolerance.md) for terminology. The [Grover report](../../resources.md) illustrates the reporting format; its counts are not an estimate for phase estimation.

## Contextual lesson

[Qiskit: Phase Estimation and Factoring, Lesson 07](https://www.youtube.com/watch?v=4nT0BTUxhJY) introduces the phase-estimation component whose controlled-evolution cost this page will account for. It is an optional video link, not implementation evidence.
