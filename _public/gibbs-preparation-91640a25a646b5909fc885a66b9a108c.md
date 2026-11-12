---
title: "Thermal and Gibbs state preparation"
description: "Checked diagonal Gibbs purification and mixed-state/readout schedules, with scalable filtering planned."
---

:::{note} Implementation status
An explicit diagonal two-qubit Gibbs purification and its reduced thermal state are checked against an independent normalized matrix exponential. Separate circuits retain the mixed system state or produce measured thermal trials. Classical coefficient loading is exhaustive and grows exponentially. A separate [noncommuting thermal-filter example](thermal-filter.md) applies actual signed Pauli access to a Bell purification. General scalable filtering/mixing, certified synthesis and physical estimates remain planned.
:::

## Motivation and intuition

A thermal mixed state favors low-energy configurations without selecting one ground state. A purification stores those probabilities as amplitudes entangled with a reference; tracing out the reference gives the mixed state. Noncommuting Hamiltonians require an actual quantum filter or other preparation procedure rather than direct classical basis weights.

For a stated finite Hamiltonian and inverse temperature $\beta$ in inverse energy units, the thermal target is

$$
\rho_\beta=\frac{e^{-\beta H}}{Z_\beta},\qquad Z_\beta=\operatorname{Tr}(e^{-\beta H}).
$$

This page distinguishes that density operator, its purification, and measured thermal observables. Algorithmic Gibbs preparation requires access, normalization and spectral assumptions; the original preparation methods provide context {cite:p}`sim_gibbs_poulin_wocjan_2009`. Our first circuit instead uses an explicitly enumerated diagonal model and pays its classical coefficient-loading cost.

## Model, input and temperature

Use

$$H/E_0=0.6Z_0Z_1+0.35Z_0-0.2Z_1,\qquad E_0=1,\qquad k_B=1.$$

The coefficients are expressed in the chosen energy unit $E_0=1$. At $\beta=1.25/E_0$, temperature is $T=0.8E_0$. The Hamiltonian is diagonal in the computational basis, with no unknown eigenbasis or diagonalization step. In displayed bit order `q1 q0`, the energies of `00`, `01`, `10`, `11` are $(0.75,-1.15,-0.05,0.45)E_0$. Every system and reference wire starts in zero.

We compute all energies, shifted Boltzmann weights and rotation angles classically. With $E_{\min}$ as a stability shift,

$$p_x=\frac{e^{-\beta(E_x-E_{\min})}}{\sum_y e^{-\beta(E_y-E_{\min})}},\qquad
\log Z_\beta=-\beta E_{\min}+\log\sum_y e^{-\beta(E_y-E_{\min})}.$$

The shift preserves the state but its contribution to normalization is recorded. At the default temperature, probabilities are approximately $(0.0627972,0.6751331,0.1707004,0.0913693)$ and $\log Z=1.8303454$. Exhaustive energy enumeration takes $O(n2^n)$ operations and $O(2^n)$ storage; the loader is not a scalable Gibbs algorithm.

## Fundamental algorithm and small circuit

The original generator uses a binary tree of conditional probability masses. At each prefix it applies an explicit controlled $R_Y$ rotation with angle $2\operatorname{atan2}(\sqrt{p_{\rm right}},\sqrt{p_{\rm left}})$, preparing

$$\sum_x\sqrt{p_x}|x\rangle_S.$$

System-to-reference CX gates then produce

$$|\Psi_\beta\rangle=\sum_x\sqrt{p_x}|x\rangle_S|x\rangle_R,\qquad
\operatorname{Tr}_R|\Psi_\beta\rangle\langle\Psi_\beta|=\rho_\beta.$$

Controlled-rotation state loading is a standard constructive approach {cite:p}`sim_state_preparation_mottonen_2005`; the code here is original. It contains explicit gates rather than an opaque state-preparation or dense-unitary operation. System wires are `q0,q1` and reference wires `q2,q3`.

```{embed} #simulation-gibbs-circuit
```

There are three different output schedules:

| Output | Circuit and interpretation |
|---|---|
| Purification | Prepare the pure system-reference state; leave every wire unmeasured |
| Retained mixed system | Initialize all wires, purify, then irreversibly reset the reference without conditioning; leave system unmeasured |
| Measured thermal trial | Initialize all wires, purify, measure system wires in Z, and discard the reference without conditioning |

The reference must be traced out, discarded or reset without conditioning to obtain the thermal marginal. Uncomputing the copying CX gates restores a coherent amplitude state, not a Gibbs density operator. Conditioning on reference outcomes gives basis states rather than the thermal mixture. The separate reset and readout circuits make this distinction concrete.

## Verification and formalization

The tests compare the actual circuit's reduced density operator with normalized $e^{-\beta H}$ from a separate SciPy matrix exponential. That dense matrix is a bounded classical answer, never a quantum gate. For $\beta=0,0.25,1.25,5$, the checks include trace distance, trace, minimum eigenvalue and thermal means. They also verify the coherent purification amplitudes, bit ordering, energy-shift invariance, reference-reset channel, and primitive decomposition {cite:p}`sim_qiskit_262,sim_scipy_1181`.

```{embed} #simulation-gibbs-checks
```

```{embed} #simulation-gibbs-observables
```

The tested trace distances are below $2\times10^{-16}$. At $\beta=1.25$, the means are approximately $\langle H\rangle=-0.6967240$, $\langle Z_0\rangle=-0.5330049$, $\langle Z_1\rangle=0.4758606$, and $\langle Z_0Z_1\rangle=-0.6916670$. These are ideal circuit/reference comparisons, not hardware or finite-shot experiments. Numerical precision is not a formal interval certificate.

## End-to-end resource analysis

Ideal preparation succeeds deterministically: there is no postselection, amplification or required input-overlap estimate. The cost is moved into exhaustive classical normalization and angle loading. The four-qubit purification lowers to six CX, five arbitrary RY and two X gates, with circuit dependency depth 10. A retained mixed-state preparation adds four initial reset instructions and two reference reset instructions. A measured thermal trial adds four initial resets and two system measurements, and discards two reference wires. The exported circuits include those actual schedules.

```{embed} #simulation-gibbs-ledger
```

One computational-basis sample supplies both a bitstring and its known diagonal energy. For one fixed iid scalar mean in $[a,b]$, the separate Hoeffding proposal is

$$N\geq\left\lceil\frac{(b-a)^2\log(2/\eta)}{2\epsilon_{\rm stat}^2}\right\rceil.$$

With $\epsilon_{\rm stat}=0.03$ and $\eta=0.005$, a single $Z_0$ mean needs 13,315 trials. A single energy mean uses the full interval $[-1.15,0.75]$ and needs 12,017 trials. These proposals are unexecuted and each covers one mean; simultaneous confidence for several means requires reallocation. Repeated trials include fresh input/reference resets.

Representation and ideal preparation error, coherent synthesis error, statistical confidence and physical faults are separate. The ledger proposes a per-trial coherent operator allowance of 0.005, giving a conservative 0.01 bias allowance for a $Z_0$ mean, independently of its 0.03 statistical error. A separate proposed physical-failure probability of 0.005 across all trials is divided by the trial count. These budgets have not been realized by synthesis or a physical model. The angles remain unsynthesized; no concrete Clifford+T sequence or physical-error confidence is claimed. Reset/discard/readout wall time, hardware scheduling, factories, error correction and physical resources are unavailable. Classical energy/Boltzmann preprocessing and circuit materialization wall times are recorded by the generator, but they are not quantum runtime.

## Run, inspect and scale

```bash
python scripts/generate_simulation_gibbs.py
pytest tests/test_simulation_gibbs.py
```

The parameterized loader accepts explicit diagonal energy lists of at most eight system qubits, and the generated purification doubles that width. Dense circuit/reference checks stop at six system qubits. The floating preparation domain requires $\beta(E_{\max}-E_{\min})\leq700$, and exported primitive circuits are capped at 250,000 operations. These are materialization and numerical limits, not a statement of asymptotic feasibility.

The [notebook](../../notebooks/simulation_gibbs.ipynb) provides the labeled outputs above and optional live temperature controls. Static execution with `OPENQUANTUM_STATIC_BUILD=1` creates no widget objects. Independent notebook downloads need [gibbs.py](../../assets/generated/simulation/gibbs/gibbs.py) beside the notebook and the [locked dependencies](../../requirements-lock.txt).

- [Original Gibbs source](../../assets/generated/simulation/gibbs/gibbs.py) and [standalone generator](../../assets/generated/simulation/gibbs/generate_simulation_gibbs.py)
- [Purification OpenQASM 3](../../assets/generated/simulation/gibbs/purification.qasm)
- [Retained mixed-state reset schedule](../../assets/generated/simulation/gibbs/retained-mixed-state.qasm)
- [Measured thermal trial](../../assets/generated/simulation/gibbs/thermal-trial.qasm) and [static measured preparation](../../assets/generated/simulation/gibbs/thermal-sample.qasm)
- [Logical ledger, classical cost and numerical checks](../../assets/generated/simulation/gibbs/logical-resources.json), [thermal-check CSV](../../assets/generated/simulation/gibbs/thermal-checks.csv), [MIT license](../../assets/generated/simulation/gibbs/LICENSE.txt)

The [noncommuting thermal filter](thermal-filter.md) advances a separate bounded preparation path. General scalable filtering/mixing, low-temperature complexity and comparisons against a fixed surface-code model remain planned. This tiny diagonal loader does not establish an efficient Gibbs algorithm or quantum advantage.

[Return to quantum simulation](../quantum-simulation.md).
