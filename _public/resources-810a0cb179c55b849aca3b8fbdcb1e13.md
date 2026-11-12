---
title: "Hamiltonian dynamics: fault-tolerant analysis"
description: "Implemented logical workloads and the limits of current physical projections."
---

The dynamics examples publish decomposed circuit counts and separate output contracts. Their physical-resource estimates remain **pending**. Arbitrary rotations are not free T gates, and an instrument with reset or intermediate measurement requires a suitable estimator.

| Component | Implemented workload | Boundary |
|---|---|---|
| Product formulas | State preparation, every term exponential, output measurement, step-dependent counts | Rotations are unsynthesized; a fixed observable needs a shot/confidence allocation |
| Taylor LCU | Explicit PREPARE/SELECT/uncomputation, all segments and resets in a complete attempt, readout and capped retries | Postselection is charged; oblivious amplification and physical costing are not implemented |
| QSP/qubitization | Implemented access block, walk/reflections, frozen phases and a real cosine filter | This filter is a different task from full complex unitary dynamics; amplification and physical costing remain pending |

[Product-formula ledger](../../assets/generated/hamiltonian-simulation/logical-resources.json) · [LCU ledger](../../assets/generated/simulation/lcu/logical-resources.json) · [QSP ledger](../../assets/generated/simulation/qsp/logical-resources.json)

The LCU ledger reports complete-attempt acceptance and capped repetition budgets. These can dominate the workload. A calculated budget for many accepted readouts is not an executed large experiment. State errors describe the conditional state before final system readout; exported measured attempts produce classical outcomes.

The shared static resource adapter can project a bound product-formula circuit with final measurements. Its arbitrary-rotation cost is a conditional mean fit, not a concrete synthesis sequence or a certified worst-case error bound. Mid-circuit LCU measurement/reset is outside that static adapter. QSP filter costing must retain its own output and success contract; it cannot inherit a dynamics accuracy budget.

The [frozen Grover report](../../resources.md) supplies illustrative surface-code hardware profiles, not reusable physical counts for this task. Compare any later co-design at the same Hamiltonian, time, input, output, accuracy, confidence and hardware assumptions.

[Verification](verification.md) · [Fault-tolerance background](../../appendices/fault-tolerance.md) · [Dynamics implementation](../hamiltonian-simulation.md)
