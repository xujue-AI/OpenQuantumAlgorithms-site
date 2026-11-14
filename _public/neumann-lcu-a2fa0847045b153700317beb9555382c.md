---
title: "Neumann-series linear inverse"
description: "A uniformly bounded reciprocal polynomial, explicit Pauli LCU and known-input amplitude amplification."
downloads:
  - file: ../../notebooks/linear_inverse_lcu.ipynb
    title: "Executable notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Implementation and tests"
---

The spectrum-matched [QSVT cubic](qsvt.md) solves a special two-eigenvalue example. A broader inverse needs an approximation valid across a promised spectral interval. This page implements a finite **Neumann-series inverse with Pauli LCU**. It is a distinct method from a general QSVT reciprocal compiler.

The intuition is classical and transparent: if $C=I-B$ is a contraction, repeated powers of $C$ form a convergent geometric series for $B^{-1}$. We compute the finite polynomial's Pauli coefficients, construct its actual PREPARE/SELECT/unprepare circuit, and measure a property of the accepted solution state. LCU supplies a quantum implementation of a linear combination; related quantum linear-system algorithms use other reciprocal expansions {cite:p}`sim_taylor_lcu_2015,linsys_cks2017`. Our finite Neumann choice, exact coefficient aggregation and example are original.

:::{note} Implemented scope

A positive identity-dominant Hermitian Pauli matrix, a complex input circuit, a finite reciprocal polynomial, explicit access and ordinary known-input amplification are implemented. The analytic polynomial error is uniform across a specified interval. Tiny circuits are independently checked; larger generated circuits are marked separately. Classical Pauli aggregation is capped and can grow exponentially. General sparse matrix loading, a QSVT reciprocal phase compiler, certified hardware synthesis and Lean formalization remain unavailable.
:::

## Task, input and spectral promise

Write an explicit matrix

$$
A=aI+\sum_j c_jP_j,\qquad r=\sum_j|c_j|<a,\qquad
\alpha=a+r,\quad B=A/\alpha,\quad
\delta=\frac{a-r}{a+r}.
$$

The coefficient dominance promise ensures $\operatorname{spec}(B)\subseteq[\delta,1]$ and condition number at most $1/\delta$. The source accepts exact decimal or rational coefficients; no eigenvectors, inverse matrix or solution state constructs a gate.

The default is

$$A=.75I-.1\,IX-.1\,ZI-.05\,XX.$$

Qiskit's rightmost Pauli acts on q0. This matrix is noncommuting and has four distinct eigenvalues, approximately .53820, .73820, .76180 and .96180. Its promised gap is $\delta=.5$, while its actual condition number is about 1.787. The input uses RY(.63), RZ(.41) on q0 and RY(.37) on q1, so the prepared amplitudes are complex and unbalanced.

The output task is **one conditional $Z(q_0)$ expectation** of the normalized solution $A^{-1}|b\rangle/\|A^{-1}b\|$. A quantum state does not return every solution entry or the overall solution norm. The tiny classical solve provides an independent reference to this same normalized output.

## Uniform reciprocal approximation

Use the scaled polynomial

$$
T_K(B)=\delta\sum_{k=0}^K(I-B)^k
       =\delta B^{-1}\bigl[I-(I-B)^{K+1}\bigr].
$$

With $\gamma=1-\delta$ and $E=\gamma^{K+1}$, the spectral promise gives

$$\|T_K(B)-\delta B^{-1}\|\le E.$$

This is an analytic bound across the whole promised interval, rather than agreement only at chosen eigenvalues. Our exact rational Pauli arithmetic preserves the finite polynomial before its coefficients are converted into numerical gate angles.

```{embed} #linear-inverse-reciprocal
```

There is a sharper normalized-state bound than dividing this operator error by a small acceptance amplitude. Relative to the exact solution, the positive factor $F=I-(I-B)^{K+1}$ has eigenvalues in $[1-E,1]$. For a normalized vector $|\psi\rangle$, the angle between $|\psi\rangle$ and $F|\psi\rangle/\|F|\psi\rangle\|$ obeys

$$
D_{\rm tr}\le\frac{E}{2-E},\qquad
|\Delta\langle P\rangle|\le\frac{2E}{2-E}
$$

for a Pauli observable $P$. To see the angle bound, for positive $F$ with spectrum $[u,v]$, $F^2\le(u+v)F-uvI$. Optimizing the resulting lower bound on $\langle F\rangle^2/\langle F^2\rangle$ gives $4uv/(u+v)^2$; pure-state trace distance is therefore at most $(v-u)/(v+u)$. Here $u=1-E,v=1$. This concerns the exact polynomial and ideal gates. Numerical coefficient loading, amplification implementation and physical synthesis require separate evidence.

For $K=12,\delta=.5$, $E=1/8192$. The uniform conditional Pauli-bias bound is about $1.221\times10^{-4}$. The actual default circuit has trace distance about $1.46\times10^{-5}$ and Z bias about $2.07\times10^{-5}$ relative to the independent solve.

```{embed} #linear-inverse-results
```

## Explicit polynomial access

The generator multiplies Pauli words using exact rational coefficients. Since each accumulated power is a polynomial in $C$, multiplication can use $(CP+PC)/2$; anticommuting pairs cancel algebraically. No tolerance-based term pruning is used.

Let $T_K=\sum_\ell t_\ell P_\ell$ and $s=\sum_\ell|t_\ell|$. The actual circuit prepares label amplitudes $\sqrt{|t_\ell|/s}$, applies the signed Pauli word selected by that label, and reverses PREPARE. All-zero labels select $T_K/s$. The exact coefficient bound is

$$\delta\le s\le1-E.$$

Because $T_K$ has eigenvalues at least $\delta$, unamplified success is at least $(\delta/s)^2$. The tiny numerical default acceptance is about .80094. All label and data wires are measured at the end, including rejected trials.

```{embed} #linear-inverse-circuit
```

## Known-input amplitude amplification

A nonunitary inverse must not be amplified by assuming it is a scaled unitary. Here $W$ includes the **actual input preparation and LCU**. Define a good reflection $S_g=I-2\Pi_{\rm labels=0}$ and an initial reflection $S_0=I-2|0\cdots0\rangle\langle0\cdots0|$ on every initial wire. Both are explicit X/MCZ/X circuits.

After $W$, one chronological round applies $S_g,W^\dagger,S_0,W$ and the global minus sign. This is ordinary amplitude amplification of the known prepared state {cite:p}`linsys_amplitude2002`. Its accepted direction remains the finite-polynomial solution in the ideal model; it is not oblivious amplification of an arbitrary inverse operator.

For known input success $p$, $r$ rounds give $\sin^2((2r+1)\arcsin\sqrt p)$ and require $2r+1$ complete forward/inverse source calls plus $2r$ reflections. Success is nonmonotone: a first round can overshoot. The notebook checks actual circuits against this formula and compares the accepted state before and after amplification.

```{embed} #linear-inverse-amplification
```

The [amplification CSV](../../assets/generated/linear-systems/neumann-lcu/amplification.csv) records every selected round's success and primitive counts. The additional [one-round QASM](../../assets/generated/linear-systems/neumann-lcu/amplified-trial.qasm) contains actual source inverses and both reflection types. It is a measured-trial export. A single unknown specimen cannot be copied for retries; this procedure requires the stated reproducible preparation.

## Space, time and accuracy scaling

An unscaled reciprocal error bound is $E/\delta$. For a requested conditional Pauli-bias allowance $\epsilon_A$, selecting

$$E\le\frac{2\epsilon_A}{2+\epsilon_A}$$

gives degree $K=O(\delta^{-1}\log(1/\epsilon_A))$ for small gap. The module computes this degree and verifies the inequality using exact rational arithmetic. A request exceeding the degree-64 cap is rejected.

The circuit uses $n+\lceil\log_2M\rceil$ logical wires, with $M$ nonzero aggregated polynomial terms. Explicit SELECT work depends on the Pauli word widths, while label PREPARE has controlled rotations. Each full amplified source must be charged. If $L$ is the complement input-term count and $M_{\max}$ is the largest intermediate power term count, aggregation can require $O(K L M_{\max})$ word products before this circuit exists. Each product manipulates $n$-letter strings and growing rational coefficients. Final $M$ and intermediate $M_{\max}$ can grow exponentially; the generator rejects more than 256 aggregate terms. Reducing the polynomial to a small Pauli list is classical preprocessing, rather than a free quantum oracle.

The [width CSV](../../assets/generated/linear-systems/neumann-lcu/width-scaling.csv) records actual circuits at $n=2,3,4,6,8,12,24$ using three specified words: X on q0, Z on q(n−1), and X on every wire. This fixed algebra can be classically tractable, so its width scaling establishes circuit-generation behavior without establishing a general advantage. The [condition CSV](../../assets/generated/linear-systems/neumann-lcu/condition-scaling.csv) changes the promised gap and selects degree at fixed bias allowance. The [degree CSV](../../assets/generated/linear-systems/neumann-lcu/degree-convergence.csv) compares actual small-instance errors with the uniform bound.

![Neumann degree, generated width cost and amplification acceptance](../../assets/generated/linear-systems/neumann-lcu/convergence-and-cost.svg)

Generation allows 24 data wires, 32 input terms, degree 64, 256 aggregate terms and 200,000 expanded primitive operations. Input rational numerators/denominators are capped at 512 bits, their normalized common denominator at 128 bits, literals at 192 characters and scientific exponents at magnitude 128. Reference solves allow six data wires and statevectors at most 12 total wires. Named large rows are circuit-only exports. The separate bounded physical study below covers selected size/gap cases; the larger logical grids remain unprojected.

## End-to-end resource analysis

```{embed} #linear-inverse-ledger
```

The [logical ledger](../../assets/generated/linear-systems/neumann-lcu/logical-resources.json) includes input gates, every Pauli access, any complete source inverses/reflections, and all final measurements. Exact coefficient preprocessing and source generation/lowering CPU times are measured separately in one local run; these activities overlap in implementation and are not added as disjoint end-to-end timings.

For one Pauli mean, .03 statistical error and .005 statistical failure require 13,315 accepted samples. A separate .005 no-acceptance allowance sets the fresh-attempt cap. The default numerical success gives ten attempts per slot, or 133,150 capped full attempts. These are **unexecuted ideal workload proposals**. All rejected attempts pay the entire schedule. Amplified workloads use the checked prepared-input probability; the unamplified lower bound cannot be blindly substituted into the nonmonotone amplification formula.

The [surface-code report](../../assets/generated/linear-systems/neumann-lcu/surface-code.json) supplies seven conditional Pareto points for the default unamplified task. It uses the existing illustrative $10^{-4}$ physical error rate, 100 ns gates/500 ns measurements, separate .005 physical-fault allowance over all attempts, and a $10^{-6}$ per-attempt synthesis-model allowance. Factory, distance, peak-qubit and serial-runtime results are in the report; the frozen Grover baseline is unchanged.

**The analytic polynomial bound is not a hardware certificate.** The ledger separately bounds exact rational coefficient conversion to its floating values and reports a conversion-adjusted state bound in an otherwise exact-angle model. That coefficient-only correction does not certify numerical rotation construction or subsequent synthesis. The full numerical gate implementation and QDK's mean-fit rotation model do not certify an operator norm, conditional mean, acceptance rate or actual synthesis sequence. Clean initialization timing/faults and classical reference/readout processing time are excluded. Physical runtime excludes the separately measured coefficient construction and compilation. Dependency depth is not a complete hardware schedule.

The pinned QASM exporter may omit an overall compiled global phase; the ledger records that phase. All exports are scoped to complete measured trials, rather than phase-preserving coherently controlled inverse oracles.

### Computed physical size and gap slices

The [separate bounded study](../../assets/generated/linear-systems/physical-sweeps/manifest.json) costs n=2,3,4 at promised gap 1/2, and n=2 at gaps 3/4,1/2,1/4,1/8. Degree is selected from a uniform 0.001 Pauli-bias ceiling: 4,9,24,51 across those gaps. All selected circuits and readout laws have independent small-instance checks. The HHL panels use different matrices and do not establish a matched algorithm-advantage comparison.

The common normalized-Z tolerance is 0.05: 0.01 algorithm bias, 0.01 postselection-adjusted synthesis bias and 0.03 sampling. Statistical, no-acceptance and physical-fault allowances are each 0.005, totaling 0.015 per separate case. There is no simultaneous guarantee across the grid. The profile is fixed at physical error $10^{-4}$, 100 ns gates and 500 ns measurements.

Every case charges all preparation, access, readout and rejected complete attempts. The uniform acceptance lower bound determines retries throughout; chosen-input numerical probabilities do not replace it. The $10^{-6}$ per-attempt QDK synthesis scalar is only hypothetically treated as a complete coherent error bound. No actual synthesis sequence or hardware-confidence certificate is supplied. Fresh logical-zero initialization/reset, classical preprocessing/readout, routing and decoder realization remain excluded.

![Conditional size and conditioning resource slices](../../assets/generated/linear-systems/physical-sweeps/physical-scaling.svg)

[Complete workloads and selected physical points](../../assets/generated/linear-systems/physical-sweeps/summary.csv) · [Reports and circuit links](../../assets/generated/linear-systems/physical-sweeps/summary.md). The earlier degree-12 default report remains a separate numerical-input proposal. The new study uses the same rational unamplified lower-bound rule throughout, and does not project the larger logical grids or amplified policies.

## Verification and formalization

Tests and independent reviews compare rational polynomial coefficients with separately assembled matrix powers, actual access blocks with the polynomial, and normalized accepted states with classical solves. They vary degree, conditioning, word width and complex inputs. Actual reflected circuits are checked against the amplitude formula and the unamplified accepted direction. X/Y/Z basis readout and invalid/oversized input rejection are covered. No Lean formalization or integrated synthesis theorem covers these gates.

The [executed notebook](../../notebooks/linear_inverse_lcu.ipynb) reuses the implementation and its outputs; optional live controls vary a bounded two-qubit example. Live Python needs a kernel, while the saved book is static. The [public code package](../../assets/generated/openquantum-code.zip) provides shared Pauli/preparation helpers and tests; a module snapshot alone does not include its dependencies.

```bash
python scripts/generate_linear_inverse_lcu.py
python scripts/generate_linear_inverse_lcu.py --delta 1/4 --pauli-bias .001
python scripts/generate_linear_inverse_lcu.py --rounds 2
python scripts/generate_linear_inverse_lcu.py --qubits 12 --no-sim \
  --no-scaling --no-estimate --output-dir artifacts/neumann-12
```

All code, circuit figures and explanations are original MIT work. The [locked environment](../../requirements-lock.txt) and generated provenance record Python, Qiskit, NumPy/SciPy versions, source/module/QASM hashes and CPU device. Qiskit is Apache-2.0; NumPy/SciPy are BSD-3-Clause. See the [main linear-systems page](../linear-systems.md) for the wider input/output context and exact upstream links.
