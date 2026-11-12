---
title: "Verification: order finding and factoring"
description: "Acceptance checks for reversible arithmetic, phase samples, and recovered factors."
---

:::{note} Verification protocol
These checks are planned acceptance criteria. This page contains no executed Shor result or reproduced formal proof. Method-specific evidence will link to the implemented circuit, pinned environment, and report.
:::

## Check arithmetic before factoring

For small registers, compare each modular multiplication with classical integer arithmetic on every valid basis input. Define the circuit's behavior on basis states outside $0\leq x<N$ as well: a reversible circuit must have a consistent unitary extension over the entire register. Test the control-off identity, the control-on permutation, register ordering, and restoration of every clean ancilla.

Compare the explicit circuit with its decomposed export up to global phase. A simulator's classical arithmetic callback or a dense matrix representing the desired permutation is a reference oracle, not evidence that the scalable arithmetic generator has been implemented.

## Check the complete attempt

Choose tiny instances with a classically known multiplicative order. Record the phase samples, the continued-fraction procedure, and every candidate order rejected or accepted. Verify $a^r\bmod N=1$ and, when the claimed result is the order, exclude smaller candidates. Validate that returned factors are nontrivial and divide $N$ exactly.

The success test must include failed samples and base choices. Do not select a favorable sample after seeing the answer or compile the answer into the demonstration circuit. Fix a stopping/retry procedure before collecting evidence, then check its declared failure calculation.

## Scope of the evidence

[Shor's paper](https://arxiv.org/abs/quant-ph/9508027) is the algorithm reference. Small exhaustive arithmetic checks cover those tested register sizes; they do not prove correctness for every integer or certify a surface-code realization. A future formal artifact must identify its arithmetic semantics, theorem, toolchain, and connection to the generator.

Resource verification additionally checks that the exported circuit includes preparation, arithmetic, readout, and the repetitions in the [resource contract](resources.md). The frozen [Grover report](../../resources.md) is a reporting example, and the [fault-tolerance appendix](../../appendices/fault-tolerance.md) explains the separate physical-model assumptions.
