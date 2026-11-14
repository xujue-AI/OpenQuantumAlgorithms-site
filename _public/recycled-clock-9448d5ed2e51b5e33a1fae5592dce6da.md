---
title: "Retained targets and recycled clocks"
description: "Actual measurement/reset energy-estimation rounds and their confidence and resource boundaries."
---

## Motivation and fundamental construction

The [energy protocol](../energy-estimation.md) needs several phase measurements conditional on the same energy component. Retaining the data while resetting a clock supplies those measurements with fewer live wires. HF is prepared once per group; it is not reloaded for every round.

`build_recycled_clock_qpe` explicitly resets data at group start, prepares X(q0) per dimer, then resets $m$ clock wires before each QPE round. Each measured clock output occupies its own classical bit segment. No measurement-dependent change is applied to the retained Hamiltonian data.

The diagram is a small three-round, two-clock-bit instrument. Its resets and intermediate measurements are actual operations.

![Small retained-target reset instrument](../../assets/generated/energy-estimation/recycled-clock.svg)

## End-to-end resource analysis

Fresh-clock static width is $n+Rm$; recycling uses $n+m$. With the defaults $n=2$, $m=5$ and $R=19$, that is 97 versus 7. The recycled group has $n+Rm=97$ explicit source resets and 95 clock readouts, one HF preparation and 589 repeated U calls. All $K=2$ groups still run; recycling does not remove confidence or preparation work.

The [reset ledger](../../assets/generated/energy-estimation/recycled-clock-resources.json) separates source/primitive gates, resets and output bits. Dynamic lowering preserves initial resets rather than deleting them under an implicit-zero assumption. Reset faults, preparation timing, reuse/feedback communication and physical scheduling need an instrument-aware backend. The current static QDK importer rejects this circuit; its physical estimate is **unavailable**.

[Analytic powers](analytic-powers.md) can also replace repeated U calls on this known family. One-ancilla feedback QPE has a further width tradeoff, introduced in the [QPE chapter](../qft-qpe/iterative.md); it is not silently included in this $m$-clock protocol.

## Verification and formalization

Small actual measurement/reset circuits and their primitive lowering are compared with complete deferred-clock output distributions and independent spectral-mixture laws. Conditional target collapse preserves the joint energy law. Measuring the target or repreparing HF within a group would change that law.

The canonical sequential simulator applies gates and projects/resets measured clocks with a fixed seed. It verifies bounded output behavior without allocating the full deferred register. No synthesized physical instrument or Lean proof is claimed. [Measured reset circuit](../../assets/generated/energy-estimation/recycled-clock.qasm) and the [notebook](../../notebooks/energy_estimation.ipynb) expose the implementation.
