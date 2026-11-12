---
title: "Amplitude estimation: verification"
description: "Planned exact-probability, amplification, and interval-coverage checks."
---

**Status: verification plan.** Checked results will be attached after the [preparation and estimator implementations](../amplitude-estimation.md) are integrated.

For tiny examples, compute $a$ directly from an independent state or classical probability distribution. Verify the good-state flag and check that $A^\dagger A$ acts as the identity. Test the implemented amplification operator against its expected action in the good/bad subspace, including phase conventions and controlled powers where used.

Compare estimated values with exact probabilities for interior values and boundary cases near zero and one. Separate a deterministic ideal-distribution check from finite-shot statistical validation. Record random seeds and repeat the entire interval-producing procedure to assess empirical coverage; one successful estimate does not establish the claimed confidence level.

For iterative or adaptive schedules, save every chosen power, shot count, observation, and interval update. Check that stopping uses the declared tolerance and confidence allocation. These records are also needed by the [resource ledger](resources.md).

Any future formal artifact must cover the estimator's actual state-preparation and confidence assumptions. No reproduced proof is included here. The [original method paper](https://arxiv.org/abs/quant-ph/0005055) defines the algorithmic context; the [mathematics appendix](../../appendices/mathematics.md) supplies probability and norm background.
