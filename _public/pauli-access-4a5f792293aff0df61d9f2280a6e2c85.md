---
title: "Energy estimation with Pauli product formulas"
description: "Explicit full-orbital H2 access, controlled term circuits, approximation bounds and conditional resources."
---

## Motivation and intuition

An electronic Hamiltonian can be supplied as a list $H=\sum_jc_jP_j$ of Pauli words. Each word is easy to exponentiate using a basis change and parity computation. A product formula combines those elementary evolutions into approximate Hamiltonian access, which phase estimation then turns into an energy sample. Noncommutation introduces evolution error; finite clock width introduces a separate readout error {cite:p}`sim_trotter_1959,sim_strang_1968,sim_trotter_error_2021`.

This page is a variant of [energy estimation](../energy-estimation.md). It uses the full four-spin-orbital H₂ fixture, whereas the main demonstration exploits an exactly implementable reduced dimer family. Its output is **one approximate eigenenergy sample**. Ground-state selection additionally requires preparation overlap and a valid repeated-readout procedure.

## Fundamental algorithm and explicit access

The input is a real coefficient list with Qiskit labels ordered `q[n-1] ... q[0]`. Duplicate labels are combined with `math.fsum`; the resulting binary64 coefficients define the normalized input model. The analytic approximation bounds target that model and exclude incoming duplicate-aggregation roundoff.

For a Pauli word $P$, the generator implements $e^{-ictP}$ by converting X with H and Y with S† then H, computing the parity of its nonidentity wires with CX, applying $R_Z(2ct)$, and undoing parity and basis changes. An identity term contributes a global phase. **Its phase must be retained when access is controlled.** No matrix exponential is appended as a quantum gate.

For $\delta=t/r$, first order applies every term exponential for $\delta$ in chronological list order. The symmetric version uses forward half-steps, a full final term and reversed half-steps. The circuit's matrix multiplication order is the reverse of chronological gate order; its symmetric structure is unchanged.

Controlled access uses the same data basis/parity gates unconditionally, controls only the central RZ, then uncomputes. If the control is zero, the data operations cancel. A scalar identity applies a phase gate to the control. The selectable `lift-all-gates` implementation controls every elementary source gate instead; it is checked against the same ideal unitary and provides a compiler-cost comparison.

For $L$ nonidentity terms of supports $w_j$, one access has $O(r\sum_jw_j)$ elementary operations. Static QPE uses $m+n$ live wires and $2^m-1$ complete controlled accesses, plus preparation, $O(m^2)$ Fourier work and $m$ phase readouts. These counts include Hamiltonian construction rather than assigning unit cost to an oracle.

## Verified H₂ input and a small circuit

The fixed input is neutral H₂ at **0.735 Å**, STO-3G, with a closed-shell RHF reference (`spin=0`) and four spin orbitals ordered as alpha then beta blocks. The [pinned Qiskit Nature fixture](https://github.com/qiskit-community/qiskit-nature/blob/d86d4a0306f8f4af8918c9eb9e3eae4622192f45/docs/tutorials/06_qubit_mappers.ipynb) supplies all 15 electronic Pauli coefficients {cite:p}`energy_pauli_fixture`. Nuclear repulsion **+0.7199689944489797 Ha** is added separately. Hartree–Fock occupation `[1,0,1,0]` is `|0101⟩`, prepared with X on q0 and q2. The $N_\alpha=N_\beta=1$ particle sector has $M_S=0$ and includes multiple total-spin sectors; it is not a singlet projection.

The inspected checkout is tag 0.7.2, released on February 23, 2024. The cached notebook execution dates to April 6, 2023 and records Qiskit Nature 0.6.0 and Python 3.9.16; its PySCF version is not recorded. All cited notebook cell indices are zero-based. We use its stored coefficients and do not rerun PySCF, integrals or fermion mapping. Their classical cost and molecular basis error are outside this supplied-Hamiltonian demonstration. The data source is Apache-2.0; the code and figures here are original MIT work.

The first circuit shows an actual YYYY term factor with basis/parity computation. The second shows the seven-wire QPE demo. Its controlled access boxes have explicit gate definitions and a complete primitive QASM export.

![Elementary gates for an H2 YYYY factor](../../assets/generated/energy-estimation/pauli-access/pauli-factor.svg)

![Seven-wire phase estimation with explicit Pauli product-formula access](../../assets/generated/energy-estimation/pauli-access/qpe-circuit.svg)

The default uses order 2, eight steps and three phase bits. It measures only the phase register. An independent classical reference forms the approximate product unitary using $\cos(ct)I-i\sin(ct)P$, then computes its normal-unitary Schur spectral law and HF weights. The actual gate-circuit distribution agrees within about $6.1\times10^{-13}$ in the executed example.

![Actual QPE probabilities and the independent approximate-unitary law](../../assets/generated/energy-estimation/pauli-access/distribution.svg)

## Approximation, precision and important variants

To encode energy, let $c_I$ be the identity coefficient and $\beta=\sum_{P_j\ne I}|c_j|$. Choose

$$E_{\rm shift}=c_I-2\beta,\qquad \Delta E=4\beta,\qquad U=e^{2\pi i(H-E_{\rm shift}I)/\Delta E}.$$

The true spectrum lies inside phases $[1/4,3/4]$. The generator implements positive energy phase with negative time in its $e^{-iHt}$ access convention. A measured $y$ decodes to $E_{\rm shift}+\Delta E\,y/2^m$.

The approximation bound uses exact rational sums of the **normalized binary64 coefficients**, then rounds the exported bound upward. First order obeys

$$\|U_1-U\|_2\le\frac{t^2}{2r}\sum_{j<k}\|[H_j,H_k]\|_2.$$

The symmetric bound uses the suffix nested-commutator sums with coefficients 1/12 and 1/24 {cite:p}`sim_trotter_error_2021`. Pauli commutator norms are evaluated without dense matrices. For more than 64 nonidentity terms, the code explicitly falls back to a conservative $O(t^2/r)$ two-half-step bound. Gate-angle arithmetic and Clifford+T synthesis are separate from these ideal analytic-angle bounds.

At the default base time, order-2 access has operator error about **$1.2652\times10^{-4}$** against exact H evolution, below its analytic bound **$2.5019\times10^{-4}$**. The approximate ground eigenenergy differs from the true sector ground energy by about **$1.7766\times10^{-5}$ Ha**. These are distinct from the very coarse **0.94725 Ha** phase-grid width. This small demonstration does not attain chemical accuracy.

![Operator approximation error and analytic bounds as steps increase](../../assets/generated/energy-estimation/pauli-access/convergence.svg)

A normal-unitary perturbation bound $\eta$ gives a spectral phase-neighbourhood bound $\arcsin(\eta/2)/\pi$. When it leaves a full $1/2^m$ margin from the scalar branch cut, one ideal QPE sample is within

$$\Delta E\left(2^{-m}+\frac{\arcsin(\eta/2)}{\pi}\right)$$

of **some** supplied-H eigenvalue with probability at least $8/\pi^2$. The default ideal-angle contract gives about 0.94755 Ha. It includes no ground-selection, synthesis or physical confidence guarantee. HF weights of true H and approximate U are separately recorded; an approximate-U distribution check alone cannot validate ground estimation.

| Variant | Evidence and remaining work |
|---|---|
| Exact reduced dimer access | Implemented on the main page, with its own retained-target repetition contract. |
| First/second Pauli product formulas | Actual circuits, independent approximation checks and ideal-angle error bounds here. |
| Termwise versus lifted controls | Same input, precision, order, steps and trial contract; actual primitive and conditional model costs compared. |
| QSP/qubitization energy access | General application integration remains planned; see [the dynamics method](../simulation/qsp.md). |
| High-confidence ground estimation with approximate access | Requires a matched preparation/overlap, approximation, repeated-readout and synthesis contract; not supplied by one sample here. |

## End-to-end resource analysis

The default termwise circuit has **12,027 primitives**, including 6,953 CX, 3,043 RZ, 896 RX, 792 H, 169 S and 169 S†, two HF X gates and three phase measurements. Recorded dependency depth is **9,989**. Every controlled access, term exponential, uncomputation and Fourier operation is included. Arbitrary rotations are not Clifford+T syntheses.

The [logical record](../../assets/generated/energy-estimation/pauli-access/logical-resources.json) retains the full coefficients, parameters, source and QASM hashes, environment, measured construction/compilation times and exclusions. Its [actual compiled scaling CSV](../../assets/generated/energy-estimation/pauli-access/scaling.csv) varies $n=2,4,6,8,12$ and steps 1/4 for a **synthetic local Pauli chain**, with three phase bits. These rows are circuit-only examples. Their grid width changes with $n$; they are not a matched absolute-accuracy or molecular-growth comparison.

The [surface-code report](../../assets/generated/energy-estimation/pauli-access/surface-code.json) costs **one full static trial**, using the existing illustrative physical error $10^{-4}$, 100 ns gates and 500 ns measurements, a separate 0.005 modeled physical-fault allowance and $10^{-5}$ per-trial synthesis-model allowance. It returns three Pareto points. The fastest enumerated point projects 29,391 peak physical qubits and 398.174 ms; these are model values.

The [lifted-control comparison](../../assets/generated/energy-estimation/pauli-access/comparison-surface-code.json) uses the same H₂ input, eight steps, three phase bits and both allowances. Its circuit has 84,484 primitives and depth 59,007, with the same ideal output distribution but different compiler overhead. Its fastest enumerated model point uses 27,451 peak qubits and 708.288 ms. Lower logical gate count does not imply lower physical peak qubits. Both projections use the audited mean rotation-cost fit; neither supplies a concrete synthesis, accuracy or runtime-tail certificate. The frozen Grover baseline is unchanged.

Physical initialization/reset faults, external SCF/mapping and hardware-specific routing/decoder/classical readout latency remain excluded. A full ground-estimation workload needs a separately justified number of complete trials and preparation quality. Multiplying this one-shot cost by an arbitrary repetition count would not establish that contract.

## Verification and formalization

The tests check X/Y basis conventions, active/inactive controls, identity phases, complete product formulas, actual QPE distributions and primitive lowering. Full four-qubit JW restriction to occupation indices `[6,5,10,9]` agrees with the reduced fixture. Both H and approximate U must preserve a selected particle sector before reference restriction. A regression rejects an accidentally invariant product unitary whose true H leaks from the proposed subspace.

Independent review checked Pauli/Strang commutator bounds, general Hermitian two-term cases, termwise controlled matrices and spectral-neighbourhood bounds. These are bounded numerical and mathematical checks. No reproduced Lean theorem or certified physical synthesis artifact covers this implementation.

```bash
python scripts/generate_energy_estimation_pauli.py --estimate --compare-controls
pytest tests/test_energy_estimation_pauli.py
python scripts/generate_energy_estimation_pauli.py --synthetic-chain --qubits 12 \
  --phase-bits 3 --steps 4 --no-sim --no-scaling \
  --output-dir artifacts/pauli-chain-12
```

Generation allows up to 24 data wires and 256 supplied terms, with separate source/primitive limits. Dense references stop at six data wires and statevectors at 12 total wires. One controlled access must be materialized and lowered before its expansion cost is known; final export guards do not promise inexpensive compilation of every allowed input.

[Complete primitive QASM](../../assets/generated/energy-estimation/pauli-access/qpe.qasm) · [Comparison QASM](../../assets/generated/energy-estimation/pauli-access/comparison-qpe.qasm) · [Source](../../assets/generated/energy-estimation/pauli-access/energy_estimation_pauli.py) · [Generator](../../assets/generated/energy-estimation/pauli-access/generate_energy_estimation_pauli.py) · [Results](../../assets/generated/energy-estimation/pauli-access/results.json) · [MIT license](../../assets/generated/energy-estimation/pauli-access/LICENSE.txt)

Qiskit **2.6.2** supplies circuits, lowering and ideal statevectors ([tagged source](https://github.com/Qiskit/qiskit/tree/2.6.2), Apache-2.0); NumPy/SciPy supply independent arrays, spectra and exponentials; Matplotlib supplies original figures. The locked environment and source hashes identify the executed versions.
