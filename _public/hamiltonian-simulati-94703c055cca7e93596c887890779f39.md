---
title: "Hamiltonian simulation"
description: "Planned circuit generators and resource estimates for simulated quantum dynamics."
---

:::{note} Planned chapter

The initial model and comparison protocol are specified below. Circuit outputs and estimates will be added after implementation and validation.
:::

This chapter will start with a small spin Hamiltonian, then extend to structured generators and additional simulation methods. The survey motivates accounting for the Hamiltonian representation and the full application task {cite:p}`survey2025`.

## Task and mathematical contract

Given a described Hamiltonian $H$, an input state, and a simulation time $t$, implement an approximation to

$$
U(t)=e^{-iHt}.
$$

The instance must state the representation of $H$, its coefficients and units, the error metric, and the output to be measured. Producing a state and estimating an observable will be treated as distinct deliverables.

## Implementation target

- A small spin model with a circuit compared to exact matrix exponentiation.
- A parameterized product-formula generator.
- A later block-encoding and qubitization example with explicit access circuits.
- Observable estimates with a specified measurement and repetition procedure.

## End-to-end accounting

The ledger will include state preparation, Hamiltonian access, term selection or block encoding, evolution, synthesis, measurement, and repetitions. It will distinguish representation error, simulation error, synthesis error, and statistical error. Classical coefficient preparation and any preprocessing will be recorded.

## Verification target

Small-system state and observable results will be compared with exact evolution. Convergence checks will vary the algorithm's accuracy parameter. Comparisons between methods will preserve the same Hamiltonian, time, input, observable, and accuracy requirement.

## References

```{bibliography}
```
