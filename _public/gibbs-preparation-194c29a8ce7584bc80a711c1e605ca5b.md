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

The [known-input thermal-amplification method](thermal-amplification.md) now wraps the noncommuting Bell-input filter with its actual full-register input reflection. It preserves the approximate accepted purification while changing the probabilistic success; its method page includes overrotation, full readout/retry costs and bounded size counts. It does not remove filter approximation error.

This page distinguishes that density operator, its purification, and measured thermal observables. Algorithmic Gibbs preparation requires access, normalization and spectral assumptions; the original preparation methods provide context {cite:p}`sim_gibbs_poulin_wocjan_2009`. Our first circuit instead uses an explicitly enumerated diagonal model and pays its classical coefficient-loading cost.

## Important variants and recent methods

Gibbs preparation now has two distinct implementation routes worth adding beside heralded imaginary-time filtering. A dissipative sampler prepares a mixed state by repeatedly coupling to and discarding an environment; a coherent filter prepares a selected state or purification and must account for normalization and acceptance. Their outputs and costs must be matched before comparison.

| Method | Contribution to this book | Main parameter to expose | Status here |
|---|---|---|---|
| Poulin–Wocjan {cite:p}`sim_gibbs_poulin_wocjan_2009` | Early Gibbs and partition-function baseline | Hilbert-space dimension and thermal normalization | Reference; current loader does not implement it |
| Chowdhury–Somma {cite:p}`sim_gibbs_chowdhury_somma_2017` | Improved precision dependence for coherent Gibbs preparation | Dimension, inverse temperature and partition function | Reference; general algorithm planned |
| Chen–Kastoryano–Brandão–Gilyén {cite:p}`frontier_thermal_chen_2025` | Efficient synthetic Lindbladian with exact detailed balance | Temperature, access normalization and mixing time | [Finite zero-C channel checked; general sampler planned](detailed-balance-gibbs.md) |
| Ding–Li–Lin {cite:p}`frontier_thermal_kms_2025` | Finite filtered-jump construction with KMS balance | Filter regularity, jump access and mixing time | Finite-jump variant planned |
| Fang–Lu–Tong–Zhao {cite:p}`frontier_thermal_detectability_2026` | Detectability-lemma channel construction and frustration-free projection for local commuting Gibbs states | Local terms, access normalization, spectral gap and accuracy | Detectability-lemma implementation planned |
| Rouzé–França–Alhambra {cite:p}`frontier_thermal_universal_2026,frontier_thermal_optimal_2026` | Proven high-temperature convergence regimes | Temperature threshold, local interaction structure and rate normalization | Theorem context; sampler planned |
| Hahn–Sweke–Deshpande–Shtanko {cite:p}`frontier_thermal_local_circuits_2026` | Spatially truncated KMS dissipators and randomized local-channel circuits | High-temperature/local-interaction assumptions, patch radius, mixing and global-state versus local-observable error | [Local-circuit comparison planned](detailed-balance-gibbs.md#planned-local-circuit-comparison) |
| Hahn–Parameswaran–Placke {cite:p}`frontier_thermal_local_driving_2026` | Locally driven system–ancilla sampler | Mixing, tunable coupling schedule, rewinding and resets | Local-driving implementation planned |
| Ding–Chen–Lin {cite:p}`frontier_ground_lindblad_2024` | Single-ancilla dissipative ground-state route | Jump realization and dissipative convergence | Reference; ground-state implementation planned |
| Ding–Zhan–Preskill–Lin {cite:p}`frontier_thermal_bath_2026` | Forward system–bath evolution with a reusable qubit | Channel mixing, bath frequencies, temperature or ground-state gap | [One-spin components](reusable-bath.md) and [finite local-spin checks](reusable-bath-local-spin.md) completed; general method and total accuracy planned |
| Chen–Gilyén {cite:p}`frontier_thermal_observables_2026` | Detailed-balanced measurements of many thermal expectations | Independent Gibbs copies, temperature, simultaneous confidence, evolution and observable access | Preprint; peer-reviewed successor unverified; measurement method planned |

The [detailed-balance method](detailed-balance-gibbs.md) now supplies a
restricted finite diagonal channel with every weak flag outcome and
environment reset retained. Its retained mesh bias and numerical channel
powers remain distinct from its finite-output sampling projection. It has
no general Gibbs-accuracy, physical-resource or formal certificate.

The papers' mathematical guarantees do not certify our current circuits. They also do not provide a universal efficient algorithm for arbitrary low-temperature local Hamiltonians.

The single-ancilla ground-state paper links the authors' numerical repository [lin-lin/oneancillaground](https://github.com/lin-lin/oneancillaground) {cite:p}`frontier_ground_reference_code`. Its public existence is verified, but its code license, executable revision and toolchain remain unverified; it is a reference rather than an adopted package.

### Planned detectability-lemma sampler

Fang, Lu, Tong and Zhao's published method avoids the trajectory overhead of
Lindbladian simulation by using local channel constructions
{cite:p}`frontier_thermal_detectability_2026`. For local commuting Gibbs
states, it uses the detectability lemma and quantum singular value
transformation to approximate a frustration-free ground-space projector,
with improved spectral-gap dependence under the paper's assumptions.

This route remains **planned**. An implementation must specify locality,
commutation, access normalization, gap and accuracy; construct the local maps
and projection; and charge preparation, repeated access, synthesis and
readout. The commuting-case gap improvement is not a general noncommuting
mixing guarantee or a physical fault-tolerant comparison. No author code has
been adopted; circuit implementation and proof reproduction remain planned.

### Planned local-driving sampler

Hahn, Parameswaran and Placke's local-driving method was published in *Physical Review B* **114, 014302** on July 1, 2026 {cite:p}`frontier_thermal_local_driving_2026`. The technical context here comes from the publisher abstract and Sections III–IV of the officially linked [2505.22816v2 manuscript](https://arxiv.org/abs/2505.22816v2), whose title begins *Towards*. It uses analog system evolution, time-dependent local system–ancilla couplings and ancilla resets. Fixed-point proximity to the Gibbs state depends on mixing time, which can itself change with protocol parameters; rapid mixing is not guaranteed for arbitrary Hamiltonians. The manuscript's general protocol includes a unitary rewinding step, omitted only under an additional quasiparticle-picture assumption. A digital implementation and matched surface-code comparison remain planned and must charge those controls, rewinding, ancillas, resets, convergence and simulation error.

### Planned reuse for thermal observables

Chen and Gilyén's version 2 is cited as a **preprint; peer-reviewed successor unverified** {cite:p}`frontier_thermal_observables_2026`. It addresses estimating expectations from already prepared Gibbs states. Theorem 1 assumes independent Gibbs copies, controlled forward/backward evolution $e^{\pm iHt}$ and block encodings of Hermitian observables $A_i$ with $\|A_i\|\leq1$. For all $M$ expectations within additive error $\varepsilon$ and simultaneous failure probability at most $\delta$, it reports

$$
\begin{aligned}
\text{Gibbs copies}&=O\!\left(\frac{\log(M/\delta)}{\varepsilon^2}\right),\\
\text{observable accesses}&=\widetilde O\!\left(\frac{M}{\varepsilon^2}\right),\\
\text{total evolution time}&=\widetilde O\!\left(\frac{\beta M}{\varepsilon^2}\right).
\end{aligned}
$$

The copy reduction covers the estimates together; observable access and evolution still grow with $M$, and preparing the independent input copies remains an end-to-end cost. This is a separate planned protocol from the one-mean sampling proposals for our diagonal loader below. Detailed-balanced measurement channels preserve the Gibbs state after averaging over outcomes. A conditioned register can change, so successive recorded outcomes require the paper's sequence analysis rather than an iid assumption. Its copy-optimality statement also has a black-box, subexponential-evolution qualification. The bounded successor check was rate-limited; no publication successor was confirmed, and its successor status remains unverified.

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
