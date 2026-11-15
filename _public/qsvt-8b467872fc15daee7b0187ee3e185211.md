---
title: "QSVT linear inverse"
description: "A spectrum-matched inverse polynomial with explicit Pauli access and readout."
---

QSVT transforms singular values through a sequence of a block-encoded matrix's access circuit and projector phases {cite:p}`linsys_qsvt2019`. For Hermitian matrices this applies a parity-constrained polynomial to signed eigenvalues. It avoids a phase-estimation register, while retaining the costs of normalization, spectral gap, polynomial accuracy, input access and success/readout.

## An inverse on a specified spectrum

The [main page](../linear-systems.md) fixes $A=.375I-.125X$, $b=|0\rangle$ and normalized-solution $Z$ readout. Set $B=A/.5=.75I-.25X$, with spectrum $\{.5,1\}$. Our original cubic is

$$p(x)=\frac{5x-4x^3}{4}=\frac12T_1(x)-\frac14T_3(x).$$

It is odd and bounded by $5\sqrt{15}/36<1$ on $[-1,1]$. At $x=.5,1$, $p(x)=1/(4x)$, so $p(B)=B^{-1}/4=A^{-1}/8$. The normalized accepted state is $(3,1)^T/\sqrt{10}$; success is $5/32$.

This is a **spectrum-matched inverse**. Maximum reciprocal error across $[.5,1]$ is about .1823. The separate [interval-valid reciprocal method](reciprocal-qsvt.md) implements a broader polynomial and an exact ideal-phase certificate; this cubic does not inherit that method's interval guarantee.

```{embed} #linear-systems-qsvt-polynomial
```

## Access and projector phases

Actual Pauli-LCU PREPARE loads label probabilities .75 and .25 with RY. SELECT applies the signed I or X term conditioned on that label. Inverse PREPARE gives a Hermitian involution whose zero-label block is $B$. No dense matrix or inverse constructs a gate.

Four numerical phases and three complete access calls implement a QSP polynomial. Another ancilla combines positive/negative phase sequences to extract the real polynomial. The signed projector phase is $\exp(i\phi(2\Pi_0-I)Z_f)$, with branch-global phases explicitly retained. Label and real-part flag zero select $p(B)$.

```{embed} #linear-systems-qsvt-circuit
```

The [phase record](../../assets/generated/linear-systems/qsvt/phases.json) provides frozen phases, Chebyshev coefficients, 10,001-point checks and a disclosed floating reconstruction guard. A capped deterministic SciPy fitter reproduces phases; verifying frozen phases does not require fitting. Algebraic polynomial facts and numerical phase agreement are distinct.

## Scaling and variants

The generator handles $B=.75I-.25P$ for any nonidentity Pauli word $P$, preserving spectrum .5,1 and condition number 2. All three SELECT calls implement that word. Width is $n+2$, degree is 3, and access work scales with the word's nonidentity positions. The [scaling CSV](../../assets/generated/linear-systems/qsvt/scaling.csv) records actual primitive circuits through 24 data wires and named small checks. This is scaling at fixed condition and a special spectrum, not arbitrary-matrix or accuracy scaling.

A general degree-$d$ inverse polynomial requires $d$ full block accesses and $d+1$ projector phases. Its degree must be chosen from a promised normalized spectral gap and error. The accepted norm determines repetition/amplification work; arbitrary input and coefficient loading need actual constructions. General non-Hermitian inversion, singular-vector projectors and higher-accuracy phase synthesis remain outside this example.

## End-to-end resource analysis

The default Z trial has 11 CX, 12 continuous rotations, three measurements and dependency depth 28. The [ledger](../../assets/generated/linear-systems/qsvt/logical-resources.json) includes all access/unprepare calls, four projector phases, both flag Hadamards and every failed full attempt.

At .03 statistical error and separate .005 statistical/no-acceptance allowances, its unexecuted ideal proposal uses 13,315 accepted samples and 88 attempts per slot: 1,171,720 complete attempts. Fresh input preparation is assumed; a single unknown specimen cannot be copied for retries.

The [surface-code report](../../assets/generated/linear-systems/qsvt/surface-code.json) supplies seven conditional Pareto points under the main page's illustrative parameters. The .005 physical-fault allowance spans all attempts; the $10^{-6}$ synthesis-model scalar is per attempt. It reports factories, distances, peak qubits and serial runtime. Lower source counts or width do not alone establish a physical improvement.

**No certified synthesis or conditional-output guarantee is provided.** The QDK mean fit is a cost model. Clean initialization timing/faults, classical preprocessing/compilation/readout and concrete synthesis are excluded. The main page states the separate norm certificate postselection would require. QASM is scoped to measured trials; the compiled global phase is recorded and may be omitted by export. It is not a phase-preserving controlled inverse oracle.

## Verification and formalization

Actual accepted branches are compared with $(5B-4B^3)b/4$ and a normalized classical solve on the stated spectrum. Tests include X, Y and multi-qubit words, real/complex preparation, X/Y/Z basis changes, recursive rejection of hidden nonunitary preparation, lowering and deterministic phase reproduction. A generic-spectrum test detects that the cubic is not an inverse. Independent review checked 15 Hermitian Pauli sums against independent Kronecker matrices.

Dense checks allow six data/12 total wires; generated circuits are separately bounded. No Lean or other integrated formal proof covers this circuit. The [notebook](../../notebooks/linear_systems.ipynb) and [public package](../../assets/generated/openquantum-code.zip) supply the original implementation, shared Pauli/QSP dependencies and tests. Run `python scripts/generate_linear_systems_qsvt.py` with `--word` to specify actual access. Versions, licenses and primary sources are recorded on the main page and artifacts.
