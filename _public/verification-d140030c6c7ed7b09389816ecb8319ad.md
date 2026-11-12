---
title: "QFT and phase estimation: verification"
description: "Planned independent transform, phase-distribution, and precision checks."
---

**Status: verification plan.** Results will be linked here after the [circuit implementation](../qft-qpe.md) is integrated and its checks are run.

For a small register, build the reference Fourier matrix directly from its mathematical definition. Compare the circuit with that matrix, check unitarity and inverse behavior, and test both basis states and superpositions. The reference must specify the sign convention and bit-reversal permutation independently of the circuit generator.

Phase-estimation checks will cover phases exactly representable on the chosen grid and phases between grid points. An explicitly prepared eigenstate supplies a known answer. For a superposition input, compare the full ideal output distribution with the expected eigenphase weights. Controlled powers must agree with independently computed matrix powers in these tiny cases.

An approximate transform needs a declared error metric and tests across its advertised precision range. Numerical agreement at one size does not certify a truncation or synthesis bound. Sampled phase intervals additionally need a stated confidence procedure; deterministic matrix checks cannot validate interval coverage.

Later formal work may connect to [SQIR/VOQC](https://github.com/inQWIRE/SQIR). Any proof claim must identify the theorem, source circuit, toolchain, and compilation boundary. The current page includes no reproduced proof artifact.

The [resource plan](resources.md) records work outside the unitary check, including input preparation and repetitions. [Mathematical background](../../appendices/mathematics.md) explains the norms and probability distinctions used in these tests.
