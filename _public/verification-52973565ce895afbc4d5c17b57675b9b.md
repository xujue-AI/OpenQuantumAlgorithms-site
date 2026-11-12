---
title: "Grover: verification"
description: "Executed circuit checks, resource-contract checks, and their limits."
---

**Status: small circuit and resource contracts checked; no integrated formal proof.** The [implementation](../grover.md) uses an explicit equality phase oracle, whose marked integer is known to the generator. This is a teaching instance for circuit correctness.

## What is checked

The 22 dedicated Grover test cases cover oracle signs on computational basis states, invalid inputs, bit ordering, ideal success probabilities, and decomposition into the supported logical basis. Oracle matrices are compared with an independently specified diagonal matrix. Simulation results are compared with

$$
P(m)=\sin^2\!\left((2r+1)\arcsin(1/\sqrt{2^n})\right).
$$

The three-qubit case with target `101` and two iterations has ideal probability $121/128=0.9453125$. The compiled circuit is compared with the original unitary up to global phase. These are numerical checks over the tested instances, not a proof for every register size.

The notebook also checks 30 bounded cases for its live controls. Every interactive evaluation verifies the simulated marked-state probability against the same analytic expression. Saved circuit and probability outputs are reused in the chapter; a static page does not execute the Python controls.

## What resource checks establish

The 13 baseline-contract test cases check repetition arithmetic, allocation between ideal sampling failure and modeled physical faults, and rejection of opaque or unmeasured inputs. The [frozen resource report](../../resources.md) records the actual circuit digest and assumptions.

The seven general-resource-helper cases separately check static lowering, budget semantics, rejected operations, and the audited arbitrary-rotation model. That helper's mean-fit projections are conditional and do not certify concrete synthesis sequences. Its tests do not expand the proof scope of the frozen Grover instance.

## Remaining evidence

A practical search predicate needs its own reversible implementation, independent semantic checks, loading cost, and classical verification procedure. An integrated theorem would need to identify its mathematical object, assumptions, toolchain, and connection to this Python circuit. External [SQIR/VOQC proofs](https://github.com/inQWIRE/SQIR) are relevant references; this project has not reproduced a proof that covers the implementation and its physical compilation.

See [fault-tolerance background](../../appendices/fault-tolerance.md) for the distinction between circuit accuracy, sampling confidence, and modeled physical reliability.
