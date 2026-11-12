---
title: "Hamiltonian dynamics"
description: "Checked product formulas, explicit postselected Taylor LCU, and a verified QSP cosine primitive."
downloads:
  - file: ../notebooks/hamiltonian_simulation.ipynb
    title: "Executable notebook"
  - file: ../assets/generated/openquantum-code.zip
    title: "Python examples and tests"
---

This chapter implements a small dynamics task with original gate construction and an independent classical answer. It belongs to the [quantum-simulation umbrella](quantum-simulation.md); [open systems](simulation/open-systems.md) and [Gibbs preparation](simulation/gibbs-preparation.md) have separate planned tracks.

:::{note} Implementation status
First- and second-order product formulas implement dynamics with state, operator and observable checks. Taylor LCU has actual PREPARE/SELECT, uncomputation and a measured/reset postselection schedule. Qubitization access and a degree-14 QSP **cosine filter** are checked. Full complex QSP evolution and amplitude amplification remain planned. Physical estimates await a reviewed rotation-synthesis adapter. No hardware advantage or formal proof is claimed.
:::

## The fixed task

Use three spins on an open chain with

$$
H=A+B,\quad A=Z_0Z_1+Z_1Z_2,\quad B=0.7(X_0+X_1+X_2).
$$

The coupling $J=1$ defines the energy unit, $\hbar=1$, and time is $t=1$ in inverse coupling units. Prepare $|\psi_0\rangle=|000\rangle$. The state task approximates $U(t)|\psi_0\rangle$ for $U(t)=e^{-iHt}$; the observable task estimates $\langle Z_0\rangle$ after evolution. These outputs have different repetition costs. The exact small classical observable is approximately **0.404255857287**.

The input is an explicit Pauli coefficient list. Qiskit labels read `q[2] q[1] q[0]`: the terms are `IZZ`, `ZZI`, `IIX`, `IXI`, `XII`. The observable assigns $+1$ to bit 0 equal to zero and $-1$ to bit 0 equal to one. The zero basis input needs no preparation gates; a device still needs initialization. No external data loading is modeled.

For an approximation $V$, report spectral operator error $\|V-U\|_2$, state-vector error $\|V\psi_0-U\psi_0\|_2$, state infidelity and observable bias separately. Vector error retains phase; infidelity does not. For normalized states, a state-vector error $\epsilon$ bounds this observable bias by $2\epsilon$.

## Explicit product formulas

Let $\delta=t/r$. Applying an $A$ layer followed chronologically by a $B$ layer gives

$$V_1(t;r)=\left(e^{-iB\delta}e^{-iA\delta}\right)^r.$$

Applying half an $A$ layer, a full $B$ layer and half an $A$ layer gives

$$V_2(t;r)=\left(e^{-iA\delta/2}e^{-iB\delta}e^{-iA\delta/2}\right)^r.$$

These are Lie–Trotter and symmetric Strang splittings {cite:p}`sim_trotter_1959,sim_strang_1968`. Modern commutator bounds explain their approximation error {cite:p}`sim_trotter_error_2021`; the curves here are numerical results for this instance.

Terms within each layer commute. Each $e^{-ic\delta Z_iZ_{i+1}}$ uses CX–$R_Z(2c\delta)$–CX; each $e^{-ic\delta X_i}$ uses $R_X(2c\delta)$. Half layers halve the angle. The generator uses these primitive operations directly and never appends a dense evolution gate.

```{embed} #simulation-product-formula-circuit
```

The figure shows one symmetric step; the accuracy run uses $r=32$. An independent Kronecker-product Hamiltonian and SciPy matrix exponential supply the dense reference {cite:p}`sim_expm_2009,sim_scipy_1181`. Qiskit Operator and Statevector evaluate the actual circuit {cite:p}`sim_qiskit_262`.

```{embed} #simulation-product-formula-convergence
```

```{embed} #simulation-product-formula-results
```

At 32 symmetric steps, operator error is approximately $5.0261\times10^{-4}$, state-vector error $4.0545\times10^{-4}$ and observable bias $1.5912\times10^{-4}$. Doubling steps in the tested range approximately halves the first-order operator error and quarters the second-order operator error.

The two formulas give equal $Z_0$ expectations on this input although their states and operators differ. Here $V_2=e^{-iA\delta/2}V_1e^{+iA\delta/2}$; the initial state is an eigenstate of $A$ and $Z_0$ commutes with $A$. Boundary phases are invisible to this observable. An observable-only check would miss the first-order circuit's larger evolution error.

## Taylor LCU with actual access

For a segment $\delta=t/r$, form

$$T_K(\delta)=\sum_{k=0}^K\frac{(-i\delta H)^k}{k!}=\sum_j b_j P_j,\qquad s=\sum_j|b_j|.$$

Bounded classical preprocessing multiplies Pauli strings, combines identical strings and removes only exact floating-point zeros. Its actual normalization differs from the unaggregated word-series normalization. This preprocessing can grow exponentially; it is not the scalable word-indexed access model of the truncated-Taylor algorithm {cite:p}`sim_taylor_lcu_2015`.

A binary tree of controlled $R_Y$ rotations implements

$$\mathrm{PREPARE}|0\rangle=\sum_j\sqrt{|b_j|/s}|j\rangle.$$

Indexed X/Y/Z gates and coefficient phases implement $\mathrm{SELECT}=\sum_j|j\rangle\langle j|\otimes e^{i\arg b_j}P_j$. Padded labels apply identity. PREPARE, SELECT and inverse PREPARE give a unitary whose zero-label block is $T_K/s$. Tests compare that actual block with an independent polynomial matrix, including negative/complex coefficients and bit ordering.

For $K=4$ and segment counts $r=1,2,4,8$, we check the unnormalized polynomial operator against $U$, the conditional state before final system readout against $U|000\rangle$ and the observable against its exact value. Conditional-state error and acceptance probability are separate.

```{embed} #simulation-lcu-results
```

Every segment must return zero label bits. Ancillas are reset for reuse; an attempt runs all segments and finally measures every system wire in Z. The exported attempt circuits include initial resets, all access/uncomputation, label measurement/reset and final readout; their output is an accepted measured trial. The conditional state checked above describes the system immediately before that final readout. Whole-attempt acceptance is the product of state-dependent conditional probabilities. At eight segments, ideal acceptance is about $0.001015$, requiring thousands of complete attempts for one accepted measured trial with 99% confidence. The ledger also records a capped retry budget for many accepted observable shots, charging the complete schedule each time.

**Oblivious amplitude amplification remains planned.** It is not supplied by the postselection circuit. State/observable values come from ideal simulation; approximation errors exclude rotation synthesis, physical faults and finite-shot noise. This unamplified demonstration is not an efficiency comparison with product formulas.

## Qubitization and QSP cosine filtering

The five Hamiltonian terms have $\alpha=\sum_j|c_j|=4.1$. Explicit PREPARE and signed Pauli SELECT give the Hermitian involution

$$B=\mathrm{PREPARE}^\dagger\mathrm{SELECT}\,\mathrm{PREPARE},\qquad\langle0|B|0\rangle=H/\alpha.$$

Tests check the actual block. A reflection about the zero-label subspace creates the qubitization walk and its invariant-subspace identity is checked {cite:p}`sim_qubitization_2019`.

The QSP module documents its scalar matrix convention and phase ordering. Fourteen calls to $B$ and fifteen frozen phases produce a degree-14 real polynomial approximating $\cos(4.1x)/2$ on $[-1,1]$. A flag coherently combines the $+\phi$ and $-\phi$ branches to extract the real part, sharing block calls. The phase JSON records a deterministic SciPy fit, independent dense-grid residual, Chebyshev-coefficient discrepancy and analytic polynomial-tail bound. Tests compare the **actual circuit block** to $\cos(H)/2$, separately from the scalar phase check {cite:p}`sim_qsp_2017,sim_qsvt_2019`.

```{embed} #simulation-qsp-results
```

```{embed} #simulation-qsp-polynomial
```

The circuit-block error is approximately $3.40\times10^{-9}$ and ideal acceptance on $|000\rangle$ is $0.1782034$. These are filter results. **Full complex QSP dynamics remains planned:** this filter has conditional normalization and does not implement $e^{-iH}$. A single parity-constrained real polynomial does not supply both even cosine and odd sine. A later circuit must coherently combine them and count success/amplification. The current phase fit does not certify arbitrary rotation synthesis.

## Accuracy and resource handoff

A product-formula observable-accuracy proposal of 0.05 allocates 0.01 bias to Hamiltonian approximation, 0.01 to coherent synthesis and 0.03 to sampling. The checked $r=32$ operator error meets its allowance of 0.005; synthesis is pending. For independent $\pm1$ samples, Hoeffding gives

$$N\geq\left\lceil\frac{2\log(2/\eta_{\rm stat})}{\epsilon_{\rm stat}^2}\right\rceil=13315,
\quad\epsilon_{\rm stat}=0.03,\quad\eta_{\rm stat}=0.005.$$

Separately, a proposed total physical-failure probability of 0.005 gives a per-shot allowance $0.005/N$ by a union bound. Coherent norm error is not stochastic failure. A conservative synthesis proposal divides its 0.005 operator allowance across counted arbitrary rotations via a telescoping sum of gate errors. A reviewed synthesis adapter is required before surface-code estimates.

```{embed} #simulation-product-formula-ledger
```

The measured $r=32$ circuit has 3 logical qubits, no ancillas and 224 arbitrary rotations before synthesis. Metadata counts gates/depth per shot and across the proposed shot count. Depth is circuit dependency layering; no physical scheduling model is applied. LCU and QSP have separate access, ancilla, measurement and retry ledgers. Their targets differ, so these counts are not an equal-task comparison.

Coefficient handling, classical readout/postprocessing wall time and physical initialization time are unmeasured. Synthesis, error correction, factories, physical qubits and runtime are unavailable. The frozen Grover surface-code baseline is unchanged.

## Run and scale

With the pinned Python 3.12 environment:

```bash
python scripts/generate_hamiltonian_simulation.py
python scripts/generate_simulation_lcu.py
python scripts/generate_simulation_qsp.py
pytest tests/test_hamiltonian_simulation.py tests/test_simulation_lcu.py tests/test_simulation_qsp.py
```

Generate a larger circuit independently of simulation:

```bash
python scripts/generate_hamiltonian_simulation.py --qubits 32 --steps 8 \
  --no-sim --output-dir artifacts/spin-32
```

Product-formula construction stops at 64 spins, 20,000 steps or 200,000 term exponentials; dense references stop at eight spins. LCU allows at most 256 explicit terms and 32 system qubits, with a 12-total-qubit statevector cap and a Taylor degree cap. QSP declares its own bounds. Indexed SELECT materialization is costly; parameterization does not imply affordable large-scale access. No remote run is used.

The [notebook](../notebooks/hamiltonian_simulation.ipynb) supplies labeled chapter outputs and optional live circuit controls. Static execution with `OPENQUANTUM_STATIC_BUILD=1` omits widget objects. For independent downloads, save the three Python modules below beside the notebook and install the [locked environment](../requirements-lock.txt). Frozen phases are also included in the QSP module.

## Downloads and attribution

- {download}`Executable notebook <../notebooks/hamiltonian_simulation.ipynb>`
- [Product-formula source](../assets/generated/hamiltonian-simulation/hamiltonian_simulation.py), [measured OpenQASM 3](../assets/generated/hamiltonian-simulation/product-formula.qasm), [logical ledger](../assets/generated/hamiltonian-simulation/logical-resources.json), [convergence CSV](../assets/generated/hamiltonian-simulation/convergence.csv)
- [LCU source](../assets/generated/simulation/lcu/lcu.py), [LCU results/ledger](../assets/generated/simulation/lcu/logical-resources.json), [eight-segment measured/reset circuit](../assets/generated/simulation/lcu/taylor-degree4-segments8.qasm)
- [QSP source](../assets/generated/simulation/qsp/qsp.py), [phases](../assets/generated/simulation/qsp/phases.json), [QSP results/ledger](../assets/generated/simulation/qsp/logical-resources.json), [measured filter circuit](../assets/generated/simulation/qsp/cosine-filter.qasm)
- [Original implementation license (MIT)](../assets/generated/hamiltonian-simulation/LICENSE.txt)

The text and circuit constructions are original. Exact Qiskit 2.6.2 source and Apache-2.0 license, and SciPy 1.18.1 source and BSD-3-Clause license, are recorded in the bibliography. Generated artifacts record parameters, package versions, source hashes, compilation settings, seeds and CPU backend. Matrix exponentiation is exclusively a bounded classical reference. Formal verification is not integrated.
