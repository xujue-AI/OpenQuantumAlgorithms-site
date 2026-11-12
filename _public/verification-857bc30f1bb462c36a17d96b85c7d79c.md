---
title: "Hamiltonian dynamics: verification"
description: "Independent small-system checks for product formulas, LCU and QSP components."
---

The implementation has **55 simulation tests**: 20 product-formula checks and 35 LCU/QSP checks. The notebook executes the examples independently with a fresh kernel. These are numerical and functional checks, not formal proof certificates.

Product formulas are compared with independently assembled Pauli matrices and exact matrix exponentiation. Tests retain operator error, state error, infidelity and observable error as separate quantities. For the selected input, boundary phases make the two splitting orders produce the same Z measurement even while their state/operator errors differ. One matching observable is therefore insufficient evidence of matching evolution.

LCU tests check the actual encoded Taylor block, normalization, conditional state, acceptance probabilities and complete-attempt/capped-retry accounting. The quantum access circuits are implemented; the dense classical calculation is a separate reference. Its input and output contracts retain postselection.

QSP checks compare the circuit block with an independent spectral cosine function, verify scalar/Chebyshev response and frozen phase reproduction, and test additional Hermitian instances. Numerical coefficient reconstruction and a sampled grid are not a formal interval certificate. The implemented cosine primitive does not establish full complex evolution or an amplified pipeline.

A separate 32-spin generation run produced circuits and metadata while explicitly reporting simulation as unperformed. Materialization and dense-reference limits are separate; a large circuit export does not demonstrate scalable classical simulation.

[Executed notebook](../../notebooks/hamiltonian_simulation.ipynb) · [Public code and tests](../../assets/generated/openquantum-code.zip) · [Resource boundaries](resources.md)

Optional background: [MIT's quantum-dynamics lecture](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/resources/lecture-12-quantum-dynamics/) and the [mathematics appendix](../../appendices/mathematics.md). Formalization and physical synthesis certification remain future work.
