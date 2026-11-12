---
title: "Fourier transform and phase estimation"
description: "Planned implementations of the quantum Fourier transform and quantum phase estimation."
---

:::{note} Planned chapter

The circuits, notebooks, and numerical resource estimates for this chapter are being specified. No executed result is presented here.
:::

The quantum Fourier transform and phase estimation are shared components for several applications covered by the survey {cite:p}`survey2025`. This chapter will implement them as reusable circuit generators before using them inside larger algorithms.

## Task and mathematical contract

For (N=2^n), the quantum Fourier transform maps

$$
\lvert x\rangle \longmapsto \frac{1}{\sqrt{N}}
\sum_{y=0}^{N-1} e^{2\pi ixy/N}\lvert y\rangle.
$$

For phase estimation, the input includes a unitary (U) and a prepared eigenstate satisfying

$$
U\lvert\psi\rangle=e^{2\pi i\phi}\lvert\psi\rangle.
$$

The desired output is an estimate of $\phi$ with a stated error and success probability. The implementation will specify bit order, the Fourier-transform convention, and how the eigenstate is obtained.

## Implementation target

- An exact small Fourier-transform circuit checked against its matrix definition.
- A phase-estimation example with a known phase and an explicitly prepared eigenstate.
- Parameterized generators for exact and approximate transforms.
- A comparison of conventional and iterative phase-estimation variants once both are implemented.

## End-to-end accounting

Controlled powers of $U$ will be charged to the estimate rather than treated as free gates. The ledger will include eigenstate preparation, controlled evolution, rotation synthesis, measurement, repetitions, and classical phase interpretation. Approximation errors will be separated from statistical failure.

## Verification target

Small matrices will check the transform's convention and inverse. Known-phase examples will check the output distribution and bit ordering. Later applications will verify that their controlled-unitary construction has the claimed semantics.

## References

```{bibliography}
```
