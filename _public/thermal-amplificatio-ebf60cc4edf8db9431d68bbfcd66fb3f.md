---
title: "Amplifying a known thermal preparation"
description: "Full-input reflections preserve a postselected thermal purification, with oscillating success and complete measured-trial costs."
---

A thermal filter can fail postselection even when its accepted state is accurate. The [noncommuting thermal-filter circuit](thermal-filter.md) supplies an actual preparation unitary, its inverse and a recognizable success pattern. These ingredients let us amplify a **known prepared input** without changing the accepted thermal state. The experiment below checks that preservation, counts the extra access and reflections, and shows why a higher one-attempt success rate need not give a cheaper workload.

:::{note} Implementation status
Known-input amplitude amplification is implemented with explicit full-register and label-register reflections. The one-qubit Bell-plus-filter example is checked for rounds 0–4, including negative global signs and full coherent source/primitive equivalence. Complete measured trials include initialization, label readout, reference reset and separate X/Z measurements. Degree and temperature bias remain unchanged. Adaptive/fixed-point amplification, certified synthesis, fault-tolerant physical estimates and a Lean proof remain planned.
:::

## Task, representation and sizes

Use the same noncommuting model and units as the thermal-filter page:

$$H/E_0=0.6Z+0.4X,\qquad E_0=1,\qquad k_B=1.$$

The default has $\beta=1.25/E_0$, Taylor degree $K=6$ and one system qubit. A second case uses $\beta=2/E_0$ to expose overrotation. The requested state is $\rho_\beta=e^{-\beta H}/\operatorname{Tr}(e^{-\beta H})$; the measured task estimates its energy from separate X and Z means.

Let $n$ be the number of system qubits, with another $n$ reference qubits. The filter polynomial has an explicit Pauli representation

$$T_K(H)=\sum_{k=0}^K\frac{(-\beta H/2)^k}{k!}=\sum_{j=0}^{L_K-1}b_jP_j,$$

normalization $s=\sum_j|b_j|$, $m=\lceil\log_2L_K\rceil$ label qubits and $N=2n+m$ total wires. Zero padding is used in the label register. The default aggregate has three Pauli terms, so $n=1$, $m=2$ and $N=4$.

The actual source circuit $A$ starts from $|0_N\rangle$, prepares the Bell purification $|\Phi_{2^n}\rangle$ with H and CX gates, and applies explicit PREPARE/SELECT/inverse-PREPARE access. Its success projector is

$$\Pi_f=I_{S,R}\otimes|0_m\rangle\langle0_m|.$$

The accepted purification has direction $(T_K\otimes I_R)|\Phi_{2^n}\rangle$ and reduced system state

$$\rho_K=\frac{T_KT_K^\dagger}{\operatorname{Tr}(T_K^\dagger T_K)}.$$

All-zero input preparation is repeatable. No classical eigenvectors or dense exponential are loaded into a quantum gate. The classical matrix answers are small independent references. Pauli coefficient aggregation remains a potentially exponential preprocessing step.

## Intuition: reflect about the prepared state through its preparation

Write the output of $A$ as

$$A|0_N\rangle=\sqrt p\,|g\rangle+\sqrt{1-p}\,|b\rangle,$$

where $|g\rangle$ is the normalized successful purification and $|b\rangle$ lies outside the zero-label subspace. The two reflections are different:

$$S_f=I-2\Pi_f,\qquad S_0=I-2|0_N\rangle\langle0_N|.$$

$S_f$ touches only the labels. **$S_0$ touches every input wire**, including the system and reference. Applying $A S_0 A^\dagger$ reflects about the actual prepared state. Combining this reflection with $S_f$ rotates the two-dimensional plane spanned by $|g\rangle$ and $|b\rangle$ {cite:p}`sim_amplitude_2002`.

Both reflections use X gates to turn their zero pattern into an all-one pattern, followed by an H–MCX–H multi-controlled Z and X gates to undo the conversion. An empty success register gives $S_f=-I$, recorded as a global phase. There is no dense reflection gate.

Starting with $A$, each chronological round is $S_f,A^\dagger,S_0,A$, plus the required global $\pi$ phase:

$$Q=-A S_0 A^\dagger S_f,\qquad A_r=Q^r A.$$

The figure groups **defined circuits** into blocks; their access, inverse access and reflections are expanded in the primitive exports.

```{embed} #simulation-preparation-amplification-circuit
```

Let $\theta=\arcsin\sqrt p$. The known-input identity is

$$\Pi_f A_r|0_N\rangle=
\frac{\sin[(2r+1)\theta]}{\sqrt p}\,\Pi_f A|0_N\rangle.$$

It requires a unitary preparation $A$, but does not require its accepted filter block to be proportional to a unitary. The entire accepted vector receives one scalar. A negative scalar gives a global sign; it leaves the reduced thermal density matrix unchanged. Tests first check that **signed raw vector identity**, then check phase-invariant purification distance and reduced-state distance.

The system and reference stay coherent through all rounds. Label measurement and unconditioned reference reset occur only after amplification. The inverse $A^\dagger$ acts on that complete unitary preparation; it does not invert a measured or discarded thermal state.

## Actual results and overrotation

The default source has $p\approx0.4685902$. One round raises it to about $0.5937335$, while round 2 lowers it to about $0.3454224$. At $\beta=2/E_0$:

| Rounds | Actual zero-label success | Complete $A/A^\dagger$ calls |
|---:|---:|---:|
| 0 | 0.4008333 | 1 |
| 1 | 0.7818968 | 3 |
| 2 | 0.0797281 | 5 |
| 3 | 0.9925196 | 7 |
| 4 | 0.0127281 | 9 |

The extra round after the high-success case nearly removes the successful branch. Integer amplification is oscillatory; a fixed round count must be justified for its preparation probability. The demonstration chooses rounds from bounded classical checks. General probability learning and adaptive schedules are not implemented.

```{embed} #simulation-preparation-amplification-results
```

```{embed} #simulation-preparation-amplification-success
```

For the default one-round case, the reduced-state distance from the original polynomial state is at floating-point scale. Its Gibbs trace distance stays about $3.93\times10^{-7}$, and its energy error stays about $5.67\times10^{-7}E_0$. At $\beta=2/E_0$ the degree-6 Gibbs trace distance stays about $4.88\times10^{-6}$. Amplification changes acceptance, **not the Taylor approximation**. Improving the thermal state still requires a better polynomial or another validated filtering method.

## Why the thermal preparation needs this reflection

[Oblivious amplification](amplification.md) leaves an unknown system input outside its initial reflection and uses an ancilla-only projector on both sides. For a nonunitary filter, that construction generally reshapes its unequal singular values and changes the thermal eigenweights {cite:p}`sim_oblivious_aa_2014,sim_taylor_lcu_2015`.

A bounded contrast replaces the full $S_0$ here by the labels-only $S_f$. For the default one-round case, the resulting reduced state is about $0.8133$ in trace distance from the original polynomial state. The correct known-input construction preserves it to numerical precision. These are different algorithmic contracts: the known Bell input makes the full-input reflection available, without a reflection about an unknown state or quantum cloning.

Implemented choices on this page are unamplified preparation ($r=0$), fixed integer known-input amplification and the explicitly labeled ancilla-only contrast. Fixed-point/phase-adjusted schedules, efficient estimation of unknown success, scalable thermal-access constructions and physical comparisons remain planned.

## Analysis: queries, reflections and size-dependent cost

In the ideal small-$p$ oracle model, a suitable round count uses $O(1/\sqrt p)$ complete preparation/inverse calls instead of $O(1/p)$ independent attempts. That statement counts access; it does not remove the preparation, inverse, reflection, coefficient-processing or readout costs.

For $r$ rounds the actual circuit contains:

- $r+1$ forward calls to the entire Bell-plus-filter $A$, and $r$ calls to $A^\dagger$;
- $r$ full-input reflections on $N=2n+m$ wires and $r$ success reflections on $m$ labels;
- one final label readout, unconditioned reference reset and system measurement per complete attempt.

Each full source call includes the Bell gates and one PREPARE/SELECT/inverse-PREPARE access. Its inverse is an actual inverse circuit. The source preparation and access are counted on all $2r+1$ calls; initialization and final readout are charged once per complete attempt. There are no intermediate resets or early-abort savings.

The full reflection can be substantially more expensive than the label reflection. The generated size family uses $H_n=\sum_i(0.6Z_i+0.4X_i)$ at the same $\beta$ and degree. It constructs actual Pauli access and lowers both reflections, before claiming any larger-system state validation:

| System qubits $n$ | Aggregate terms | Labels $m$ | Total wires $N$ | Full-input reflection gates | Label reflection gates |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 2 | 4 | 37 | 5 |
| 2 | 9 | 4 | 8 | 391 | 37 |
| 3 | 27 | 5 | 11 | 799 | 62 |

The family ledger also lowers the source Bell-plus-filter circuit and full unamplified X/Z trials. A complete attempt has $N$ initial resets, $n$ unconditioned reference resets, $m$ label measurements and $n$ system measurements. Amplification multiplies the coherent source/inverse workload and adds both reflections, while those terminal operations occur once per attempt.

The reusable coherent source and the final measured channel have different lowering contracts. Explicit initialization and reference discard can permit different optimizations in a complete trial, so its gate count need not equal the reusable source count plus a constant readout cost. The ledger reports both actual lowerings separately.

The one-qubit accepted-shot budget is not extrapolated to larger $n$. The family energy coefficient sum grows as $n$; fixed absolute energy precision requires a new measurement/error allocation, as well as the new preparation probability. Those larger-system shot workloads remain unimplemented. Holding $\beta$ and $K$ fixed as $\|H_n\|$ grows is also **not an equal-error scaling comparison**.

These are primitive logical counts from the pinned lowering, with no initial-clean-wire assumptions. Family rows are **generation/lowering evidence**, not executed larger-system Gibbs results. The independent local-spin family also has an easy factorized classical Gibbs answer, so it is not an advantage benchmark. For general inputs, classical Pauli aggregation, explicit label SELECT and its controlled synthesis can grow rapidly.

```{embed} #simulation-preparation-amplification-scaling
```

Higher acceptance is not automatically less work. With **uncapped independent retries**, at $\beta=2/E_0$, round 3 uses roughly $7/0.9925\approx7.05$ full source calls on average per accepted outcome; unamplified attempts use about $1/0.4008\approx2.49$, before reflections and reset/readout costs. These are geometric expectations; the finite retry protocol has a separate stopped expectation and a full-cap workload. No physical-cost optimum is claimed.

## Complete energy trials and separated errors

The exported X and Z trials reset all four wires, run the amplified unitary, measure the two label bits, reset the reference without conditioning, perform the requested basis change and measure the system. Only zero-label outcomes are accepted; every rejected trial runs the full schedule and is charged. A separate coherent file is supplied for circuit inspection.

Energy estimation uses independent accepted X and Z means,

$$\widehat E/E_0=0.6\widehat Z+0.4\widehat X.$$

A single Z-basis bitstring is not an energy sample for this noncommuting sum. The numerical workload proposal retains the thermal page's 0.03 statistical energy-error allocation, 0.005 statistical-failure allowance and a distinct 0.005 acceptance-failure allowance across the accepted-shot slots. It multiplies each actual measurement-group circuit by its retry count. Sampling is **unexecuted**; ideal statevector probabilities and numerical retry projections are not hardware confidence guarantees.

For $M$ requested measurement slots, one-trial acceptance $p_r$ and retry cap $R$, the ledger distinguishes three complete-trial counts:

$$\mathbb E[N_{\rm uncapped}]=\frac{M}{p_r},\qquad
\mathbb E[N_{\rm stopped}]=M\sum_{k=0}^{R-1}(1-p_r)^k
=\frac{M[1-(1-p_r)^R]}{p_r},\qquad N_{\rm full\ cap}=MR.$$

The stopped expectation includes slots that exhaust their cap, and assumes every requested slot is attempted. It is neither conditional on all slots succeeding nor the cost of aborting the whole workload at its first exhausted slot. The named `*_uncapped` and `*_stopped_at_cap` fields retain that distinction; legacy `expected_complete_trials` and `expected_complete_attempts` fields explicitly identify their $M/p_r$ values as uncapped. Full-cap instruction counts and the proposed physical-fault allocation continue to use $MR$.

```{embed} #simulation-preparation-amplification-ledger
```

Polynomial bias, coherent synthesis error, statistical error, acceptance failure and physical faults are separate. If source/inverse error were certified by $\eta_A$, and the two reflection errors by $\eta_0,\eta_f$, telescoping gives a hypothetical coherent-circuit allowance

$$\eta_r\leq(2r+1)\eta_A+r(\eta_0+\eta_f).$$

Conditioning on ideal success $p_r$ can then amplify state error: a conservative trace-distance bound is $\min(1,2\eta_r/\sqrt{p_r})$, with perturbed acceptance at least $(\sqrt{p_r}-\eta_r)^2$ when $\eta_r<\sqrt{p_r}$. Unconditioned reference discard cannot increase trace distance. These are **conditional proposals requiring operator-norm certificates**, not performed synthesis or a conversion of coherent error into a fault probability.

Continuous-angle synthesis, routing, factories, error correction, physical reset/discard/readout times and classical acceptance-processing time remain unavailable. A separate physical-fault allowance spans the capped workload, without a combined physical-success guarantee. The frozen Grover physical baseline is unchanged.

## Verification, exports and reproducibility

Tests compare signed accepted vectors with the known-input identity, phase-invariant purifications and reduced states with the original polynomial, and Gibbs states/means with independent matrix-Taylor and normalized-exponential references. Cases include Y terms, a two-qubit Hamiltonian, overrotation and $\beta=0$ with no label qubits. Complete X/Y/Z trial tests verify reset and measurement semantics through joint density-matrix probabilities.

For the small coherent circuits, source and primitive **full-unitary matrices** are compared with global phase retained. Lowering explicitly sets `qubits_initially_zero=False`, so reusable controls cannot silently depend on clean input wires. These checks are numerical evidence for the stated cases; no Lean proof or formal compiler certificate is supplied. Formalization remains planned.

The pinned OpenQASM exporter omits an overall circuit phase. The measured exports preserve their probabilities but are not phase-preserving coherently controlled oracle specifications. The recorded phase and source modules are required for controlled reuse or explicit phase restoration. In-memory signed-vector and full-unitary checks keep that phase.

```bash
python scripts/generate_simulation_preparation_amplification.py
pytest tests/test_simulation_preparation_amplification.py
```

`--no-sim` constructs and lowers the circuits while marking checks, acceptance and workloads unavailable; it removes stale validation-only plots and CSV from a reused output directory. It does not turn an unexecuted example into a checked preparation. `--no-scaling` omits the generation-only size table. The artifact CLI uses the one-qubit X/Z model; the reusable source retains the thermal module's four-system-qubit, 256-aggregate-term and degree-12 limits, with $\beta\sum|h_j|/2\leq10$. Amplification allows rounds 0–4 and at most 24 total wires, with 50,000 top-level source instructions and a 250,000-operation limit after lowering. Nested definitions may cost more before that latter limit is checked. Statevectors stop at 12 total wires and full-operator checks at six.

The [notebook](../../notebooks/simulation_preparation_amplification.ipynb) provides reusable labeled outputs. Fresh static execution uses `OPENQUANTUM_STATIC_BUILD=1` and contains no widget objects. Standalone reproduction needs the source modules and the [locked Python 3.12 environment](../../requirements-lock.txt). Code, explanations and figures are original; the exact Qiskit/SciPy packages and licenses are credited in the simulation bibliography {cite:p}`sim_qiskit_262,sim_scipy_1181`.

- [Preparation-amplification source](../../assets/generated/simulation/preparation-amplification/preparation_amplification.py), [thermal-filter dependency](../../assets/generated/simulation/preparation-amplification/thermal_filter.py), [LCU dependency](../../assets/generated/simulation/preparation-amplification/lcu.py), [standalone generator](../../assets/generated/simulation/preparation-amplification/generate_simulation_preparation_amplification.py)
- [Results](../../assets/generated/simulation/preparation-amplification/results.json), [logical/workload ledger](../../assets/generated/simulation/preparation-amplification/logical-resources.json), [rounds CSV](../../assets/generated/simulation/preparation-amplification/rounds.csv), [size/reflection CSV](../../assets/generated/simulation/preparation-amplification/scaling.csv)
- Default one-round exports: [coherent circuit](../../assets/generated/simulation/preparation-amplification/beta-1p25-round-1-coherent.qasm), [X measured trial](../../assets/generated/simulation/preparation-amplification/beta-1p25-round-1-trial-X.qasm), [Z measured trial](../../assets/generated/simulation/preparation-amplification/beta-1p25-round-1-trial-Z.qasm)
- [MIT implementation license](../../assets/generated/simulation/preparation-amplification/LICENSE.txt)

[Return to thermal filtering](thermal-filter.md).
