---
title: "Optimization and variational methods"
description: "Planned QAOA and VQE examples with classical optimization and measurement costs."
---

:::{note} Planned chapter

Small optimization examples and comparative resource records are planned. No approximation guarantee or quantum advantage is claimed for the proposed examples.
:::

This chapter will cover representative hybrid workflows: a small MaxCut problem using QAOA and a small variational energy problem using VQE. The survey supplies context for choosing application tasks and interpreting their full costs {cite:p}`survey2025`.

## Task and output contract

For each example, specify the instance, objective, required solution quality, and the classical result used for comparison. Record both the value returned and the computational effort required to obtain it.

For a variational state $\lvert\psi(\theta)\rangle$, an energy objective takes the form

$$
E(\theta)=\langle\psi(\theta)\rvert H\lvert\psi(\theta)\rangle.
$$

The ansatz, optimizer, initialization, stopping condition, and measurement procedure are part of the algorithm.

## Implementation target

- A small MaxCut graph with exact classical enumeration.
- A small Hamiltonian with an exact reference ground-state energy.
- Parameterized circuit generators with reproducible optimizer settings.
- A complete run record containing iterations, objective evaluations, shots, and achieved quality.

## End-to-end accounting

The ledger will include all optimization iterations and measurement effort, not only one ansatz evaluation. It will record classical optimizer time, parameter changes, state preparation, and any mitigation or fault-tolerant compilation used. Resource projections will keep empirical convergence observations separate from assumptions about larger instances.

## Verification target

Small results will be checked against exact classical optima or eigenvalues. Repeated runs will report variability from initialization and sampling. Later comparisons will match target quality and success probability and identify which quantities are measured or extrapolated.

## References

```{bibliography}
```
