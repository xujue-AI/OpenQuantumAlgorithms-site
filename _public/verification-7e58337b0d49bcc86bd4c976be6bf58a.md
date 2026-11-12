---
title: "Linear systems: verification"
description: "Planned residual, normalized-state, observable, and access checks."
---

**Status: verification plan.** The [implementation](../linear-systems.md) will begin with a small matrix whose solution can be computed independently.

Solve $Ax=b$ classically and check its residual. Separately normalize the solution to obtain the target quantum state. Compare the circuit's successful output branch with that state using a declared metric, and compare the requested observable with its value on the classical solution state. State agreement and a small scalar-estimation error are different contracts.

Test the actual preparation of $|b\rangle$, the matrix-access or block-encoding circuit, and the success flag. Postselection probability and the number of rejected trials must be recorded. For a block encoding with normalization factor $\alpha$, compare its selected matrix block with $A/\alpha$; do not hide this normalization in a downstream solver comparison.

The advertised conditioning and precision ranges need multiple small cases, including cases near the chosen spectral boundary. A numerical polynomial fit must be checked on its declared spectral interval. Small residuals alone do not verify data access, state preparation, or output confidence.

The [HHL paper](https://arxiv.org/abs/0811.3171) and [QSVT framework](https://arxiv.org/abs/1806.01838) supply method references. No integrated formal artifact is claimed for this chapter. The [resource plan](resources.md) includes preparation and readout costs that remain outside a matrix-level check.
