---
title: "Amplitude estimation: resource analysis"
description: "Planned preparation, reflection, precision, and confidence accounting."
---

**Status: full resource ledger planned.** The [chapter](../amplitude-estimation.md) will specify one probability-estimation task with absolute tolerance and confidence.

## Implement the probability source

Name the preparation circuit $A$, the good-state predicate, input data representation, and desired estimate of $a$. Charge data loading, $A$, $A^\dagger$, good-state and zero-state reflections, ancillas, and uncomputation. Every controlled amplification power used by a phase-estimation version needs a concrete decomposition. Its preparation query count is only one layer of the resource record.

## Match precision and confidence

Record amplification schedules, circuit depths, shot counts, and classical interval construction. Iterative variants can use multiple circuit depths and adaptive choices; all parameter settings and executions belong in the ledger. A confidence interval must follow the implemented statistical procedure, including stopping rules and behavior near $a=0$ or $a=1$.

Algorithmic estimation error, coherent synthesis approximation, sampling confidence, and physical faults will be kept separate. Synthesis approximations change the probability being estimated and require a bias allowance. The current bounded-observable helper covers a fixed scalar mean under declared assumptions; it does not automatically establish confidence for an adaptive amplitude-estimation protocol.

## Report and compare

Publish decomposed logical counts and a conditional physical frontier for the complete schedule. Compare classical sampling of the same preparation task at the same output tolerance and confidence. Include preprocessing and classical postprocessing before drawing an application-level conclusion.

The [original amplitude-amplification and estimation paper](https://arxiv.org/abs/quant-ph/0005055) is the method reference. See the [fault-tolerance appendix](../../appendices/fault-tolerance.md) and the [Grover reporting example](../../resources.md); the latter is not an amplitude-estimation resource bound.
