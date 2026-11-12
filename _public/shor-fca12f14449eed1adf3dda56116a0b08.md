---
title: "Factoring and order finding"
description: "Planned implementation of order finding, modular arithmetic, and classical factoring post-processing."
---

:::{note} Planned chapter

The implementation and resource estimates are planned. This page defines the work required for an end-to-end factoring example.
:::

The chapter will connect reversible modular arithmetic, phase estimation, and classical post-processing rather than presenting only a precompiled small factoring circuit. The survey provides the broader application and complexity context {cite:p}`survey2025`.

## Task and mathematical contract

Given an integer $N$ and a base $a$ coprime to $N$, order finding seeks the smallest positive $r$ such that

$$
a^r \equiv 1 \pmod N.
$$

Factoring combines sampled information about the order with classical number-theoretic processing. A complete run must include unsuccessful base choices and sampling outcomes in its success accounting.

## Implementation target

- A small order-finding instance with explicit reversible modular multiplication.
- A factoring notebook that shows the samples, rational reconstruction, candidate order checks, and factor checks.
- A generator parameterized by the input integer and arithmetic design.
- A comparison between a pedagogical arithmetic circuit and an arithmetic implementation chosen for resource estimation.

## End-to-end accounting

The resource ledger will count modular arithmetic, control registers, ancillas, uncomputation, phase precision, and repetitions. It will report classical preprocessing and factor reconstruction separately. Small demonstration instances will be labeled with any simplifications that prevent their direct use as scaling estimates.

## Verification target

Classical modular exponentiation will check small reversible arithmetic blocks over their valid input domain. The run will verify the recovered order and factors. Register layout and ancilla cleanup will be tested before physical-resource estimates are published.

## References

```{bibliography}
```
