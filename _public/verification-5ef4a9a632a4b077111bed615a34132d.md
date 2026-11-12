---
title: "Quantum simulation: verification"
description: "Evidence scopes across the implemented and planned simulation tracks."
---

[Hamiltonian-dynamics verification](../hamiltonian-simulation/verification.md) records the available circuit, operator/state, postselection and polynomial checks. Its numerical evidence does not establish full complex QSP, amplification, physical synthesis or a formal proof.

Open-system work will compare an actual quantum channel or dilation with an independent analytic/density-matrix reference. Checks must preserve the selected input, rates, time, observable and error metric, and distinguish a complete channel from a selected measurement branch.

Thermal preparation will compare the actual prepared state or observable with the normalized Gibbs target at the same Hamiltonian and temperature. Classical preprocessing, normalization and purification assumptions will remain visible. Both broader tracks are planned until their circuit and checks are published.

[Physics background](../../appendices/quantum-physics.md) · [Mathematics](../../appendices/mathematics.md) · [Evidence policy](../../verification.md)
