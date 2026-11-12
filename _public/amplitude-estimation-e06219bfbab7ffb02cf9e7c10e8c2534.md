---
title: "Amplitude estimation"
description: "Planned probability-estimation examples with explicit accuracy and oracle costs."
---

:::{note} Planned chapter

Circuit examples and the comparison with classical sampling are planned. Numerical query or runtime claims will be added with their assumptions and evidence.
:::

Amplitude estimation extends the search components into a scalar estimation task. The chapter will connect preparation, amplification, measurement, and confidence requirements using the survey's application perspective {cite:p}`survey2025`.

## Task and mathematical contract

The preparation circuit $A$ produces a state of the form

$$
A\lvert0\rangle
=\sqrt{1-a}\,\lvert\psi_0\rangle\lvert0\rangle
+\sqrt{a}\,\lvert\psi_1\rangle\lvert1\rangle.
$$

The task is to estimate $a$ with a specified error and confidence level. The preparation circuit and its inverse are part of the implementation contract.

## Implementation target

- A small known-probability example with exact classical reference values.
- Conventional phase-estimation-based amplitude estimation.
- An iterative variant after its sampling and interval procedure is implemented.
- A comparison with classical sampling of the same preparation task.

## End-to-end accounting

The ledger will count preparation and inverse-preparation calls, reflections, controlled operations where used, shot counts, and classical estimation. It will distinguish query complexity from compiled gate cost and physical execution time. Confidence requirements will be shared across the quantum and classical comparison.

## Verification target

The estimator will be tested against exact probabilities for several small instances. Coverage checks will assess the reported intervals over repeated experiments. Resource estimates will include the implemented preparation circuit rather than treating it as a unit-cost oracle.

## References

```{bibliography}
```
