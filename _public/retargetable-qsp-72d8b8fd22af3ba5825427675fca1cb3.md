---
title: "Retargetable complex quantum signal processing"
description: "Generated even/odd phases, explicit signed access, interval logical-source error bounds and bounded fixed-coupling scaling."
---

Time evolution changes when the Hamiltonian size, coefficients or simulated time change. The [first QSP example](qsp.md) freezes phases for $\alpha t=4.1$; this method generates new phases for each requested normalization-time parameter. It retains explicit PREPARE/SELECT/unprepare access and checks the resulting complex circuit against small independent classical answers.

:::{note} Implementation status
Bounded numerical phase generation, a separate directed-interval coefficient validator, exact logical signed SELECT, interval PREPARE weight checks and complex selector circuits are implemented. Seven fixed-coupling size/time cases have circuit/reference checks. Circuit-only generation still fits and validates new phases. The complete-resource study below supplies actual Clifford+T measured trials and a separate conditional instrument model for one and two spins. Numerical synthesis and SDK realization premises remain distinct from the logical-source certificate; deterministic evolution and formal/Lean proof remain planned.
:::

## Task and intuition

The explicit input is a real Pauli list $H=\sum_jh_jP_j$. Interpret its supplied binary64 coefficients and time $t$ as exact real input numbers, and define the mathematical normalization $\alpha=\sum_j|h_j|$. The task approximates $U=e^{-iHt}$, or estimates a specified observable after that evolution. Here $\hbar=1$, the initial state is $|0^n\rangle$ and the measured observable is $Z_0$.

The demonstrated open chain uses fixed local couplings

$$H_n=\sum_{j=0}^{n-2}Z_jZ_{j+1}+0.7\sum_{j=0}^{n-1}X_j,$$

so $\alpha_n=(n-1)+0.7n$. Qiskit Pauli labels read the highest wire first; the rightmost letter acts on q0. Hamiltonian coefficients are in the chosen energy unit and time in its inverse.

The displayed 0.7 names the supplied binary64 value, rather than an additional certificate for the exact rational 7/10 or a physical coupling. Converting external decimal coefficients, experimental data or a continuous physical model into these inputs is a separate representation-error allowance. For Pauli inputs, a stated coefficient discrepancy can be bounded by $\|H_{\rm input}-H_{\rm model}\|\le\sum_j|h_j^{\rm input}-h_j^{\rm model}|$ and charged through the unitary Duhamel bound.

QSP applies one scalar polynomial to every normalized eigenvalue $x=E/\alpha\in[-1,1]$. Changing $n$ at fixed couplings changes $\tau=\alpha t$, so it changes the desired cosine and sine responses. A frozen sequence for one $\tau$ cannot implement this input family. Actual access and polynomial transformation are the algorithmic ingredients {cite:p}`sim_qsp_2017,sim_qubitization_2019,sim_qsvt_2019`.

## Generated phases and actual access

Use the signal convention

$$B_x=\begin{pmatrix}x&\sqrt{1-x^2}\\\sqrt{1-x^2}&-x\end{pmatrix},\qquad
D(\phi)=\operatorname{diag}(e^{i\phi},e^{-i\phi}),\qquad
V=D(\phi_0)B_x\cdots B_xD(\phi_d).$$

The fitting targets are $C(x)\simeq\cos(\tau x)/2$ with even degree $d$, and $S(x)\simeq\sin(\tau x)/2$ with odd degree $d+1$. Floating Bessel coefficients and an analytic truncation expression choose a candidate degree; the separate interval validator decides whether its final phases meet the requested allowance. The bounded least-squares solver uses a palindromic parameterization, a product-rule analytic Jacobian and deterministic seeds. Numerical phase-factor optimization has established context {cite:p}`sim_qsp_phase_fitting_2021`; this implementation does not claim universal convergence.

PREPARE is a tree of actual RY/control gates loading $|h_j|/\alpha$ approximately. Indexed Pauli operations implement SELECT, and every access unprepares the label register. Negative coefficients use exact named Clifford zero-pattern phases: Z for one label, H–MCX–H for multiple labels, or an explicit $ZXZX=-I$ sequence when no label exists. They do not use a binary64 approximation of a $\pi$ sign phase. Padded labels apply identity. For exact declared logical gates, SELECT and $B=\mathrm{PREPARE}^\dagger\mathrm{SELECT}\,\mathrm{PREPARE}$ are Hermitian involutions even when the loaded probabilities differ slightly from the intended weights.

A real-part flag combines opposite phase signs. A second selector combines $C$ and $S$ with an exact S-dagger relative phase:

$$K(x)=\frac{C(x)-iS(x)}2\simeq\frac{e^{-i\tau x}}4.$$

The circuit shares $d$ ordinary complete accesses and executes one actual controlled access for the odd branch. All label bits, the flag and selector must be zero to accept. The inverse preparations, both flag preparations, each conditioned phase and the final selector operation are actual circuit operations; no matrix exponential is a quantum gate.

```{embed} #simulation-retargetable-qsp-circuit
```

The displayed blocks have definitions; primitive exports expand them. In the selected diagram q0/q1 are systems, q2/q3 are labels, q4 is the real-part flag and q5 the selector. The selected example has two systems, $t=0.5$, $\alpha=2.4$, $\tau=1.2$ and degree 6/7. Every default size/time case requests the same decoded logical-source norm allowance $E_{\rm source}\le10^{-5}$. The separate source/primitive maximum-entry numerical residual guard is $5\times10^{-12}$ on the stated small cases; that numerical realization check is not a uniform compiler norm certificate.

## Interval bounds and realization stages

The validator treats the stored binary64 phases as exact real angles. It encloses their sine/cosine values with Taylor polynomials and real Lagrange remainder bounds, propagates the QSP polynomial coefficients using directed decimal arithmetic, then converts power coefficients to Chebyshev coefficients by an exact binomial identity. On $[-1,1]$, $|T_k(x)|\le1$, so the coefficient-residual l1 sum plus the separately enclosed target remainder gives a uniform error upper bound. Authoritative interval endpoints are decimal strings; binary64 summary bounds round outward {cite:p}`sim_python_decimal_31214`.

The phase fit's sampled residual is numerical evidence. Its coefficients, Jacobian and optimizer status do not replace the interval bound. The interval calculation independently encloses the analytic cosine/sine target, including its high-degree remainder; it does not trust SciPy's floating Bessel values as certified coefficients.

A second interval calculation propagates the actual label-only RY/X PREPARE at its stored angles. If $\widehat p_j$ are its probabilities and $p_j=|h_j|/\alpha$, including zero target weight on padded labels,

$$\delta_A\le\sum_j|\widehat p_j-p_j|$$

bounds the accepted normalized-Hamiltonian block error for the exact signed SELECT. Exact-rational arithmetic separately bounds $\delta_\tau=|\alpha t-\widehat\tau|$. If $E_P$ is the decoded polynomial error, functional calculus and the unitary Duhamel bound give

$$\|4K-U\|\le E_P+|\alpha t|\delta_A+\delta_\tau=:E_{\rm source}.$$

The constructor recomputes certificates from supplied phase plans and rejects a source bound exceeding the requested allowance. This covers declared logical Cliffords/conditional controls and stored rotation angles. Floating nested ControlledGate/MCX definitions and compiler transformations are another realization stage. The source/primitive matrix checks below are numerical evidence for that stage, not a compiler norm certificate. Synthesis and physical errors are further stages.

```{embed} #simulation-retargetable-qsp-certificate
```

Let $\epsilon=E_{\rm source}/4$. The exact target branch has norm $1/4$ for every normalized input. If $\epsilon<1/4$, acceptance is at least $(1/4-\epsilon)^2$; conditional vector error and $Z_0$ bias are bounded conservatively by $2E_{\rm source}$. If a realization worker supplies a certified whole-coherent-circuit error $\eta$, replace $\epsilon$ by $\epsilon+\eta$ before conditioning. Every access, inverse, flag and projector phase must be included in that realization contract. An average rotation-fit statistic is not such a norm bound. Initialization/readout instrument errors and physical faults remain separate.

## Executed small checks and selected output

The default generator checks $n=1,2$ at $t=0.25,0.5,1$ and $n=3$ at $t=0.5$. It constructs all their phases afresh. An independent Kronecker Hamiltonian and SciPy exponential provide bounded references. Actual source accepted blocks, conditional states, observables and acceptance are compared; full source/primitive operators retain global phase. Tests include negative coefficients, Y terms, complex inputs, exact sign branches, real/$\pm i$ selector choices and malformed or stale plans.

```{embed} #simulation-retargetable-qsp-results
```

```{embed} #simulation-retargetable-qsp-response
```

The conditional state means the system immediately before final system readout. The exported full attempt instead resets every wire, prepares the named basis input, runs the entire circuit and measures all wires in Z. It retains a system bitstring only when every success ancilla is zero. Numerical acceptance is near $1/16$; this remains probabilistic evolution. Retrying assumes a fresh re-preparable input, and cannot copy an unknown single specimen after rejection.

## Logical workload and size scaling

With $L=2n-1$ chain terms, there are $m=\lceil\log_2L\rceil$ labels and $n+m+2$ logical wires. Each of the $d+1$ accesses includes PREPARE, SELECT and inverse PREPARE; conditioned phase controls and the extra controlled access are lowered and counted. Direct SELECT scans every label and Pauli letter. Loading, control decomposition and data preparation are not free oracle operations.

```{embed} #simulation-retargetable-qsp-ledger
```

For the selected two-spin attempt, the compiled schedule has six logical wires, 1,665 operations, dependency depth 1,185, six resets and six final measurements. Arbitrary rotations remain unsynthesized. One accepted trial with 1% no-acceptance failure has a numerical cap of 72 complete attempts. The separate proposed $Z_0$ mean requires 13,315 accepted trials for statistical error 0.03/failure 0.005, and an acceptance-failure allocation 0.005 across all slots gives 230 attempts per slot: 3,062,450 capped complete attempts. These workloads are unexecuted; they charge rejected schedules in full and distinguish the numerical probability model from the logical-source lower bound and any physical realization.

```{embed} #simulation-retargetable-qsp-scaling
```

At fixed $J=1,h=0.7,t=0.5$, the executed $n=1,2,3$ counts use regenerated degree 4/5, 6/7, 8/9 phases. The executed circuit-only $n=8$ example targets $\tau=6.3$, with newly generated degree 16/17 phases, 14 wires and 56,574 measured operations. Its depth is 45,902 with 11,340 continuous rotations, 14 resets and 14 measurements. This exceeds the 12-wire statevector cap, so its state/operator checks and empirical acceptance are unperformed; the declared-source interval bounds are still evaluated. It uses a new target rather than the original 4.1 sequence. Requested logical-source accuracy is held fixed, but compilation/synthesis/physical accuracy is not certified by this sweep. The ideal query-degree dependence is separate from the measured direct-scan gate growth. No physical improvement or classical/hardware advantage is claimed.

Dense state storage grows as $2^n$ and a full circuit operator grows as $4^{n+m+2}$. Bounded references are verification tools, not a scalable classical benchmark. Phase fitting/certificate and coefficient-processing CPU times are recorded separately from logical gate counts. The separate one-/two-spin study below consumes these circuit/error contracts through the existing serialized instrument backend. It does not physically estimate the larger logical sweep. Formal/Lean evidence is planned.

## Complete observable workload and conditional physical resources

The [complete-resource bundle](../../assets/generated/simulation/complete-resources/README.md) fixes $J=1,h=0.7,t=0.5$, the fresh input $|0^n\rangle$ and the accepted $Z_0$ mean for $n=1,2$. Each attempt resets every wire, runs the full access/phase circuit, measures every success ancilla and reads all system wires. Success measurements precede system readout in this resource circuit. Final measurements on different wires commute; the bounded joint-distribution check establishes agreement with the earlier data-first trial. A rejection still incurs the complete schedule.

The source trial is routed on a bidirectional logical line with fixed initial layout, SABRE seed 7 and optimization level 0. Clean-input compiler assumptions are explicitly disabled for the coherent lowering. Every remaining rotation is replaced by an actual ancilla-free Ross–Selinger Clifford+T sequence using Qiskit 2.6.2 {cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`. Source, routed and synthesized instruments are checked against the independent matrix-exponential score. Reset/readout events retain their ideal instrument meaning in these checks. The native synthesizer has no seed API; recorded sequences and hashes define the saved realization.

The absolute-error target is 0.05, with allowances 0.01 for source bias, 0.01 for realization bias and 0.03 for statistical error. Let $\epsilon$ bound the encoded source error, $s=1/4$ be the exact target-branch norm and $W=2$ be the width of the $\{+1,-1\}$ score. The guarded sum of actual per-rotation numerical errors is $\eta_{\rm synth}$, within the requested $5\times10^{-5}$ allowance. A separate $10^{-7}$ SDK nested-control/routing reserve is an explicit conditional premise. With $\eta=\eta_{\rm synth}+10^{-7}$ and $\epsilon+\eta<s$, the study uses

$$p_{\rm lower}=(s-\epsilon-\eta)^2,\qquad
B_{\rm source}\le\frac{2W\epsilon}{s},\qquad
B_{\rm realization}\le\frac{2W\eta}{s-\epsilon}.$$

These conservative bounds apply to normalized accepted scores; coherent approximation is not charged as a stochastic fault. Outcome agreement does not prove the SDK reserve, and the numerical synthesis guard is not an interval norm certificate. Hoeffding's inequality requires 14,855 accepted samples. An outward interval calculation sets a complete retry cap for each accepted slot and charges their product. The overall conditional failure allowance 0.01 is split equally among statistics, acceptance, physical faults and factory availability. The sample ensemble and physical schedule are unexecuted.

| Spins | Wires | Actual T/T-dagger per attempt | Guarded synthesis estimate (rounded) | Conditional acceptance lower | Complete cap per accepted slot | Capped complete attempts |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 6,128 | $9.7257\times10^{-6}$ | 0.06249507 | 242 | 3,594,910 |
| 2 | 6 | 31,022 | $8.5174\times10^{-6}$ | 0.06249558 | 242 | 3,594,910 |

Acceptance entries are rounded downward; the JSON records retain authoritative decimal bounds and actual numerical synthesis values. The actual ideal Clifford+T means are 0.7648418454 and 0.7836347543; the independent exact-evolution answers are 0.7648421873 and 0.7836360901. Their numerical biases are much smaller than the separately allocated realization proposals. They do not prove the norm premise used in the retry budget.

```{embed} #simulation-complete-resources-dynamics
```

The physical reports use the existing serialized instrument scenario with direct QDK 1.34.0 Trace/ledger agreement {cite:p}`resource_qdk_134`. For $w$ source wires and odd distance $d$, the data/workspace/bridge reservation is $(4w-1)(2d^2-1)$ physical qubits, plus one native factory. Every full attempt charges workspace startup, source reset/readout, controller dispatch, idle code exposure, actual neighboring CNOTs and all magic-state supply. T/T-dagger macros include complete capped factory retries, SWAP transport, injection, readout, corrective frames, port initialization and extra port residence. Native factory output quality and factory availability are separate contributions.

The report retains physical-error scenarios $10^{-3}$ and $10^{-4}$ at controller failure $10^{-15}$ per decision, and separately evaluates the conditional controller premise $10^{-18}$. An infeasible result stays in the report. If the all-attempt controller contribution alone exceeds the allocated additive physical-model budget, changing distance or factory cannot make that declared model feasible. This is a contribution to the engineering estimate, not an experimentally established hardware failure lower bound.

These are the minimum-space selections from the retained qubit/time Pareto frontiers. Hours include every capped complete attempt and factory retry.

| Spins | Physical error | Controller error per decision | Distance | Peak physical qubits | All-attempt hours |
|---:|---:|---:|---:|---:|---:|
| 1 | $10^{-3}$ | $10^{-15}$ | 37 | 1,015,427 | 63,203.8 |
| 1 | $10^{-4}$ | $10^{-15}$ | 19 | 253,071 | 35,877.9 |
| 1 | $10^{-3}$ | $10^{-18}$ | 33 | 807,667 | 57,131.4 |
| 1 | $10^{-4}$ | $10^{-18}$ | 17 | 202,527 | 32,841.7 |
| 2 | $10^{-3}$ | $10^{-15}$ | — | — | Infeasible controller model |
| 2 | $10^{-4}$ | $10^{-15}$ | — | — | Infeasible controller model |
| 2 | $10^{-3}$ | $10^{-18}$ | 37 | 1,048,271 | 381,519 |
| 2 | $10^{-4}$ | $10^{-18}$ | 17 | 209,451 | 202,922 |

At the unchanged $10^{-15}$ controller premise, the two-spin all-attempt controller contribution is 0.0179800 against its 0.0025 physical budget. The one-spin contribution is 0.00246970, leaving little room for the other additive model terms. The $10^{-18}$ rows are separate reliability scenarios, rather than changes to the published baseline.

Timing remains 100 ns gates/initialization, 500 ns measurements, 1,000 ns code cycles and 10,000 ns controller reaction. Memory and initialization residuals have separate assumptions; readout uses an independent majority-path model. Protected Pauli-parity geometry, tracked Clifford frames and compatible factory/data handoff remain conditional interface assumptions {cite:p}`resource_litinski_2019,resource_horsman_2012`. Offline coefficient/synthesis preprocessing and final statistical-processing wall time are excluded from quantum runtime. Layout certification, decoder throughput and hardware calibration remain unavailable. This study separately caps total wires at eight, source instructions at 20,000, synthesized/check instructions at 500,000, weighted simulator branches at 256 and calculated complete attempts at $10^{12}$. The separate scenario preserves the frozen Grover/PSSPC baseline and supplies no matched co-design improvement claim.

- {download}`Complete-resource notebook <../../notebooks/simulation_complete_resources.ipynb>`
- [Workload/source/error records](../../assets/generated/simulation/complete-resources/results.json), [physical summary](../../assets/generated/simulation/complete-resources/summary.csv), [manifest](../../assets/generated/simulation/complete-resources/manifest.json)
- [Bounded generator](../../assets/generated/simulation/complete-resources/generate_simulation_complete_resources.py), [workload implementation](../../assets/generated/simulation/complete-resources/complete_resource_workloads.py), [instrument backend](../../assets/generated/simulation/complete-resources/instrument_resources.py)
- Two-spin [complete source QASM](../../assets/generated/simulation/complete-resources/dynamics-n2-Z/source-instrument.qasm) and [actual Clifford+T QASM](../../assets/generated/simulation/complete-resources/dynamics-n2-Z/clifford-t-instrument.qasm)

## Reproduce and bounds

```bash
python scripts/generate_simulation_retargetable_qsp.py
pytest tests/test_simulation_retargetable_qsp.py
python scripts/generate_simulation_retargetable_qsp.py --qubits 8 --time 0.5 \
  --no-sim --no-sweep --output-dir artifacts/retargetable-eight
```

The access constructor permits at most 16 systems, 32 explicit terms, $|\tau|\le8$, degree32 and 50,000 top-level source instructions. Adjacent parity pairs therefore stop at degree30/31. Primitive exports stop at 250,000 operations after lowering. Statevectors stop at 12 total wires, full operators at 8 and independent Hamiltonian references at 6 systems. The CLI limits its sweep to 12 cases. The phase fitter also has restart/evaluation/time caps; failure returns no validated phase plan. These materialization, phase-generation and simulator bounds are independent. Unsupported broader methods remain planned.

`--no-sim` still fits phases and runs coefficient/weight/parameter validation; quantum state/operator comparisons and empirical acceptance/workloads are marked unperformed. Reused output directories remove only previously manifested circuit/phase files and known generated plots, preserving unrelated files. Optional exact-root Git provenance is absent in standalone packages; source hashes remain authoritative.

The pinned Qiskit QASM exporter omits an overall circuit phase. It preserves the supplied measured-trial probabilities, but is not a phase-preserving controlled-oracle specification. Source/in-memory circuits or explicit phase restoration are required for coherent controlled reuse.

- {download}`Executable notebook <../../notebooks/simulation_retargetable_qsp.ipynb>`
- [Method source](../../assets/generated/simulation/retargetable-qsp/retargetable_qsp.py), [phase generator](../../assets/generated/simulation/retargetable-qsp/qsp_phase_generation.py), [interval validator](../../assets/generated/simulation/retargetable-qsp/qsp_certificate.py), [standalone generator](../../assets/generated/simulation/retargetable-qsp/generate_simulation_retargetable_qsp.py)
- [Results](../../assets/generated/simulation/retargetable-qsp/results.json), [complete logical ledger](../../assets/generated/simulation/retargetable-qsp/logical-resources.json), [size/time CSV](../../assets/generated/simulation/retargetable-qsp/scaling.csv)
- Selected [phase plan](../../assets/generated/simulation/retargetable-qsp/phases-n2-time0p5.json) and [measured QASM](../../assets/generated/simulation/retargetable-qsp/evolution-n2-time0p5.qasm)
- [Optional provenance helper](../../assets/generated/simulation/retargetable-qsp/artifact_provenance.py), [MIT license](../../assets/generated/simulation/retargetable-qsp/LICENSE.txt) and [locked environment](../../requirements-lock.txt)

Save all copied sources beside the notebook/generator for an independent run. Code, phases, text and figures are original. Exact Qiskit 2.6.2/Apache-2.0 and SciPy 1.18.1/BSD-3-Clause sources are credited {cite:p}`sim_qiskit_262,sim_scipy_1181`.

[Return to QSP](qsp.md).
