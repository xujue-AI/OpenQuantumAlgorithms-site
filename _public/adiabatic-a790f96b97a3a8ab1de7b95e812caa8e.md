---
title: "Adiabatic intuition and a digitized anneal"
description: "An explicit four-qubit MaxCut anneal, continuous numerical reference, separate digitization checks, and a qualified gap diagnostic."
---

Adiabatic optimization starts from an easily prepared low-energy state and changes the Hamiltonian toward one whose low-energy states encode solutions {cite:p}`opt_adiabatic2000`. Interference arises through that evolution rather than a classical search over independent layer angles. How slowly the path must be followed depends on its spectral structure and smoothness. There is no generic polynomial runtime promise for an arbitrary optimization problem.

This example uses the same four-vertex cycle and classical MaxCut reference as [QAOA](qaoa.md). It implements a digitized time-dependent anneal with explicit gates, not an analog-hardware experiment. The target, time, and number of slices are chosen before examining the studies.

## Hamiltonian path and units

Let the cycle have $n=4$ vertices and $m=4$ edges. Its cut operator is $C=\sum_{(u,v)}(I-Z_uZ_v)/2$. Define

$$
H_D=-\sum_{j=0}^{n-1}X_j,\qquad
H_P=\frac12\sum_{(u,v)}Z_uZ_v=\frac m2 I-C,
$$

$$
H(s)=(1-s)H_D+sH_P,\qquad s=t/T.
$$

The initial state $|+\rangle^{\otimes4}$ is the ground state of $H_D$ and needs four H gates from $|0000\rangle$. Minimizing $H_P$ maximizes the cut. Its ground space consists of the two alternating strings `0101` and `1010`, checked by enumeration of all 16 assignments. Success is the total measurement probability of that ground space, not a preference for either string.

Set $\hbar=1$ and use the stated dimensionless energy coefficients. The path time $T$ is in corresponding inverse-energy units; it is not a hardware execution time in nanoseconds.

## Midpoint slices and explicit signs

For $L$ slices, take $\Delta t=T/L$ and $s_j=(j+1/2)/L$. One midpoint Strang slice is

$$
U_j=
e^{-i\Delta t(1-s_j)H_D/2}
e^{-i\Delta t s_jH_P}
e^{-i\Delta t(1-s_j)H_D/2}.
$$

The chronological circuit applies a driver half-step, the edge interactions, and a second driver half-step. With $R_P(\alpha)=e^{-i\alpha P/2}$, each driver half uses `RX(-dt*(1-s))` on every vertex. Each edge uses `CX(u,v); RZ(dt*s,v); CX(u,v)`. The signs follow from the negative-X driver and positive-ZZ problem Hamiltonian. There is no symbolic access oracle.

The figure shows **two slices** to make the repeating gate pattern readable. The fixed main run has **256 slices**, and the exported main QASM contains all of them.

![](#opt-annealing-circuit)

Before compilation or merging adjacent rotations, the source construction has $n$ H, $2nL$ RX, $mL$ RZ, and $2mL$ CX, followed by the requested readout. Thus circuit generation is $O(L(n+m))$ gates. At $n=m=4,L=256$, the recorded source has four H, 2,048 RX, 1,024 RZ, 2,048 CX, and four measurements. Compilation choices and rotation synthesis can change the subsequent cost.

## Separate path quality from digitization error

The fixed main input is $T=16,L=256$ with a predeclared ideal maximum-cut probability target of at least $0.95$. Its executed circuit reaches approximately $0.987022729$. That is a checked result on this graph and path, rather than an adiabatic-runtime guarantee.

The continuous reference numerically integrates the Schrödinger equation with **SciPy 1.18.1 DOP853**, using the tolerances and solver diagnostics recorded in the run JSON {cite:p}`opt_scipy1181`. DOP853 provides an independent high-accuracy numerical reference; it is not an exact solution or a global error certificate. A separate dense-matrix implementation of the same Strang slices checks the gate construction up to global phase.

The time study holds $L=256$ and uses $T=1,2,4,8,16$. The slice study holds $T=4$ and uses $L=8,16,32,64,128$. Increasing path time addresses the ideal evolution's preparation quality. Refining slices addresses digitization relative to the continuous numerical reference. They are different changes with different errors, and the plots retain that distinction.

```{include} ../../assets/generated/optimization/annealing/summary.md
```

![](#opt-annealing-convergence)

## Endpoint degeneracy and a symmetry-sector diagnostic

The full spectral gap $E_1(s)-E_0(s)$ is zero at $s=1$, because the two optimal cuts are degenerate. The global bit flip $G=X^{\otimes4}$ commutes with both Hamiltonians, and the initial uniform state lies in its $G=+1$ sector. Exact evolution and the stated digitized gates preserve that symmetry.

The report scans 201 interpolation points and displays both the full gap and the gap within the eight-dimensional $G=+1$ sector. This helps explain why a vanishing full endpoint gap need not describe the accessible-sector behavior of this example. A finite grid does not certify a lower gap bound between its points, an adiabatic runtime theorem, robustness to symmetry-breaking errors, or scaling on another graph.

![](#opt-annealing-gap)

## Readout and complete fixed-circuit workload

Fresh ideal readout uses 4,096 samples for each seed `7`, `19`, and `41`, totaling 12,288 modeled preparations of the complete fixed circuit. Each outcome is evaluated against the explicit classical cut predicate. The source probabilities are computed classically; repeated QPU executions are not performed.

![](#opt-annealing-readout)

The [resource record](../../assets/generated/optimization/annealing/resources.json) counts every preparation, driver rotation, edge interaction, and measurement at those repetitions. CPU simulation and continuous-reference work are recorded separately from unmeasured quantum-device latency. The time and slice scans are diagnostic classical computations; they do not silently become hardware executions in the fixed-circuit report.

Statevector simulation needs exponentially many amplitudes, and the dense references need exponentially large matrices. The statevector helper has a configured 12-qubit limit with a hard cap of 16 and $T\le32$; dense and ODE references have a hard eight-qubit cap and $T\le32$. The displayed example uses four qubits. The constructor separately caps graph width at 32, $T$ at 64, and $L$ at 4,096, subject to an aggregate 100,000-unitary-gate limit. Exporting a larger circuit does not validate its adiabatic quality or imply affordable simulation.

Fault-tolerant projection is discussed [in the main chapter](../optimization.md#fault-tolerant-resources-and-complete-workloads). Arbitrary-angle synthesis there is a conditional expected-cost model, not a concrete sequence or a certified cut-probability error bound. A fixed-circuit physical report is also different from the [recorded adaptive-trajectory replay](finite-shot.md#recorded-trajectory-resources).

## Reproduce and inspect

```bash
python scripts/generate_optimization_annealing.py
python scripts/execute_notebooks.py
```

Use the pinned Python 3.12 environment. The [executed notebook](../../notebooks/optimization_annealing.ipynb), [matching module](../../assets/generated/optimization/annealing/optimization.py), [generator](../../assets/generated/optimization/annealing/generate_optimization_annealing.py), and [MIT license](../../assets/generated/optimization/annealing/LICENSE.txt) support the standalone checks. Circuit operations use Qiskit core 2.6.2; dense calculations use NumPy 2.5.4. SciPy's [exact 1.18.1 source](https://github.com/scipy/scipy/tree/v1.18.1) and [BSD-3-Clause license](https://github.com/scipy/scipy/blob/v1.18.1/LICENSE.txt) identify the numerical ODE backend.

- [Full runs and numerical-reference diagnostics](../../assets/generated/optimization/annealing/runs.json)
- [Logical fixed-circuit workload](../../assets/generated/optimization/annealing/resources.json)
- [Complete 256-slice measured QASM](../../assets/generated/optimization/annealing/annealing.qasm)
- [Time study](../../assets/generated/optimization/annealing/time-study.csv)
- [Slice study](../../assets/generated/optimization/annealing/step-study.csv)
- [Finite-grid gap diagnostic](../../assets/generated/optimization/annealing/gap.csv)

## References

```{bibliography}
```
