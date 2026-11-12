---
title: "Optimization: verification"
description: "Independent circuit, exact-reference, optimizer and grouped-readout checks."
---

The current implementation has **40 optimization tests**, plus an independently executed notebook. QAOA is checked against exhaustive classical cut enumeration and independently assembled dense evolution, including asymmetric multiple-layer angles. Tests verify cost signs, angle conventions, bit ordering and complete evaluation traces.

The two-qubit VQE example is checked against the full Hamiltonian spectrum and an independent analytic energy expression. Its ansatz contains this particular toy ground state; that does not imply general VQE expressibility or convergence. All optimizer starts, rejected proposals, stopping conditions and objective calls are recorded.

Readout tests retain the covariance of jointly measured Z terms, independent X/Z streams, reproducible counts, bounded-outcome Hoeffding calculations and shot-weighted gate totals. Sample standard errors are descriptive; the stated ideal bounds are per fixed estimate, not simultaneous guarantees across every seed or optimizer parameter.

The chapter explicitly distinguishes exact-statevector optimization from ideal sampled final validation. A noisy/finite-shot optimizer trajectory, hardware execution, physical estimates and formal proof are not established by this increment. A separate 32-qubit generation run verifies export/counts without simulation or optimization.

[Executed notebook](../../notebooks/optimization.ipynb) · [Run records](../../assets/generated/optimization/runs.json) · [Public code and tests](../../assets/generated/openquantum-code.zip) · [Resource boundaries](resources.md)

See the [mathematics appendix](../../appendices/mathematics.md) for spectra and confidence, and the [verification policy](../../verification.md) for evidence scopes.
