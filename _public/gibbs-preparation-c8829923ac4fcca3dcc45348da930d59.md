---
title: "Thermal and Gibbs state preparation"
description: "Planned preparation circuits and explicit thermal-state accuracy and success contracts."
---

:::{note} Planned
This track has no implemented Gibbs preparation circuit, validated thermal result or resource estimate. The normalized matrix exponential is a future classical comparison, not a quantum preparation algorithm.
:::

For a finite Hamiltonian and a specified inverse temperature $\beta$ in inverse energy units, the target state is

$$
\rho_\beta=\frac{e^{-\beta H}}{Z_\beta},\qquad Z_\beta=\operatorname{Tr}(e^{-\beta H}).
$$

The implementation will distinguish preparing this density operator, preparing a purification, and estimating a chosen thermal observable. It must specify the representation of $H$, its energy reference, the input state, approximation metric, confidence and failure allowance. The preparation method and its normalization, initial overlap and any spectral assumptions remain to be chosen.

The ledger will include actual Hamiltonian access, filtering or channel construction, purification ancillas, postselection/amplification if used, discarded systems, measurement, repetitions and classical preprocessing. A tiny normalized matrix exponential will check the selected circuit once it exists. Low-temperature preparation and high-temperature observable estimation will not be treated as the same workload.

[Return to quantum simulation](../quantum-simulation.md).
