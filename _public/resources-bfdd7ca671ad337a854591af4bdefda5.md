---
title: "Linear systems: resource analysis"
description: "Planned access, conditioning, solution-state, and observable accounting."
---

**Status: full resource report planned.** The [chapter](../linear-systems.md) will define one small task before attaching logical and physical counts.

## Input and output boundary

Record matrix dimension, spectrum/conditioning assumptions, normalization, and the implemented matrix-access model. Preparing the right-hand-side state, loading coefficients, and constructing a block encoding or sparse-access oracle are part of the workload. An abstract matrix query is not an implemented access circuit.

For invertible $A$ and nonzero input state $|b\rangle$, a solution-state target is

$$
|x\rangle=\frac{A^{-1}|b\rangle}{\|A^{-1}|b\rangle\|}.
$$

Preparing this normalized state differs from returning all classical entries of $x$. The report will select a state metric or a named observable and include its measurement procedure. If the requested answer needs a norm, an entire vector, or several observables, that additional readout must be specified and charged.

## Algorithm and implementation costs

The historical [HHL method](https://arxiv.org/abs/0811.3171) and modern [QSVT methods](https://arxiv.org/abs/1806.01838) have different primitives and precision requirements. The chosen route will count controlled evolution or block-encoding uses, inversion approximation, ancillas, uncomputation, postselection or amplification, and repetitions. Polynomial or spectral assumptions must justify algorithm bias independently of synthesis cost.

## Physical projection status

A surface-code report needs the concrete circuit, a requested output tolerance, sampling confidence, and physical-fault allocation. A mean-fit synthesis projection for arbitrary rotations remains conditional until independent approximation evidence is supplied. No complete estimate or classical/quantum advantage comparison is reported here.

The [fault-tolerance appendix](../../appendices/fault-tolerance.md) defines the reporting contract. The [Grover baseline](../../resources.md) illustrates it; its physical resources do not bound a linear-system task.
