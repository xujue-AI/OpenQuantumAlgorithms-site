---
title: "Thermal spectral filtering with retargeted QSP"
description: "Generated even/odd phases, explicit Hamiltonian access and Bell preparation, with bounded thermal checks and logical-source error contracts."
---

A spectral filter can suppress high-energy amplitudes without expanding powers of the Hamiltonian into a Pauli-word table. This method applies an exponential-shaped QSP polynomial through the original Hamiltonian's access circuit. It advances the [aggregated Taylor thermal filter](thermal-filter.md) to newly generated phases for each Hamiltonian size and inverse temperature. The coefficients used for fitting are classical scalar data; no eigenvectors or matrix exponential are loaded into a gate.

:::{note} Implementation status
Actual phase generation, PREPARE/SELECT/unprepare access, a real parity selector, Bell preparation and complete measured trials are implemented. The generator supplies bounded gate/reference checks and separate larger circuit-only counts. Analytic/interval bounds cover the declared logical circuit with exact controls and stored binary64 rotation angles. The complete-resource study below supplies actual Clifford+T trials and a separate conditional instrument model for one and two spins. SDK realization and synthesis checks remain numerical; general thermal mixing, hardware validation and formal/Lean proofs remain planned.
:::

## Model, input and selected output

Use an open chain in units $E_0=1$, $\hbar=k_B=1$,

$$H_n=0.6\sum_{j=0}^{n-2}Z_jZ_{j+1}+0.35\sum_{j=0}^{n-1}X_j.$$

Coefficients are the exact real numbers represented by the supplied binary64 input list. Converting decimal literals, empirical inputs or estimated coefficients into that list has a separate representation error. Define the mathematical normalization $\alpha=\sum_j|h_j|$ by an exact rational sum. The displayed grid varies $n=1,2,3$ and $\beta=0,0.75,1.5$; the selected example is $n=2,\beta=1.5$. Every grid/count case requests decoded filter error $10^{-6}$ and declared-logical Gibbs trace-distance error $10^{-4}$; degree selection tightens its effective filter target when needed. It targets

$$\rho_\beta=\frac{e^{-\beta H}}{\operatorname{Tr}(e^{-\beta H})}.$$

Prepare $n$ Bell pairs using actual H/CX gates. System wires come first, $n$ reference wires next, then labels and the two QSP flags. The coherent circuit returns a conditional system-reference purification before final readout. Tracing/resetting the reference without conditioning gives the mixed system state. The downloadable X/Z trial circuits instead produce accepted classical outcomes; the reference is discarded without conditioning, and only zero QSP success bits are accepted.

## Shift the exponential into a bounded signal function

Since $H/\alpha$ has spectrum in $[-1,1]$, let $a=\beta\alpha/2$ and

$$F(x)=e^{-a(x+1)},\qquad 0<F(x)\le1.$$

The scalar shift gives $F(H/\alpha)=e^{-\beta(H+\alpha I)/2}$. It changes preparation success, but cancels when the thermal density is normalized. Shifted QSP imaginary-time primitives provide context {cite:p}`sim_qsp_imaginary_time_2023`; this circuit does not implement that paper's fragmentation protocol or inherit its resource claims.

A single real QSP polynomial has fixed parity. Split the filter into

$$C(x)=\frac{e^{-a}\cosh(ax)}2,\qquad S(x)=-\frac{e^{-a}\sinh(ax)}2,$$

then use a uniform real selector to produce

$$K(x)=\frac{C(x)+S(x)}2=\frac{F(x)}4.$$

Even and odd degree sequences share their common access calls; one actual controlled access supplies the extra odd degree. The thermal selector has relative phase $+1$. The [retargeted dynamics method](retargetable-qsp.md) uses the same construction with an exact S-dagger relative phase for cosine minus $i$ sine. The QSP convention and block transformation are specified there {cite:p}`sim_qsvt_2019`.

## Actual access, fitting and circuit

PREPARE loads $\sqrt{|h_j|/\alpha}$ through a controlled-RY tree. Indexed Pauli operations and exact Clifford pattern phases implement signed SELECT; unused labels apply identity. PREPARE, SELECT and inverse PREPARE form a Hermitian involution for the named logical gates. Its loaded magnitudes use stored binary64 angles and are independently enclosed by the weight validator.

The floating fitting targets use scaled modified-Bessel coefficients: $C_0=\operatorname{ive}(0,a)/2$, even $C_k=\operatorname{ive}(k,a)$ and odd $S_k=-\operatorname{ive}(k,a)$. Bounded symmetric least-squares optimization with an analytic Jacobian generates the phases {cite:p}`sim_qsp_phase_fitting_2021`. Fit convergence and sampled residuals are numerical evidence. A separate validator must accept the resulting stored phases before a circuit is constructed.

```{embed} #simulation-thermal-spectral-circuit
```

```{embed} #simulation-thermal-spectral-response
```

Every run retargets $a$ and the degree. The original $\alpha t=4.1$ dynamics phases are not used as a thermal or larger-system generator. At $\beta=0$ the desired Gibbs state is maximally mixed; this uniform postselected construction still pays its access/selector cost. A direct Bell preparation would be cheaper for that special task.

## Independent bounded checks

The actual accepted branch is compared with $\operatorname{vec}(F(H/\alpha))/(4\sqrt{2^n})$. Its squared norm is acceptance; its normalized reduced density is compared with a separate normalized SciPy exponential. Circuit-block error, purification error, Gibbs trace distance, energy bias and acceptance are separate metrics. The dense matrices are classical answers, not quantum operations.

```{embed} #simulation-thermal-spectral-results
```

```{embed} #simulation-thermal-spectral-accuracy
```

Across the nine numerical cases, the largest Gibbs trace-distance error is $9.4623\times10^{-8}$. At $n=2,\beta=1.5$, filter-block error is approximately $1.5916\times10^{-8}$, Gibbs trace distance $9.4623\times10^{-8}$, energy bias $1.9997\times10^{-8}$ and raw acceptance **0.0157915**. The declared-logical decoded-filter bound is $6.4760\times10^{-8}$, Gibbs bound $9.1035\times10^{-7}$ and conservative acceptance lower bound 0.00126512. These analytic bounds and numerical values have different scopes.

The nine displayed cases are checked on the actual source circuits. The accepted filter block is extracted from the provided circuit by inverse-Bell preparation and data-basis Statevector trajectories, within an effective system-plus-success width of eight and a full Bell-statevector width of twelve. This is a bounded filter-operator check, not a full larger-wire unitary check. For larger Bell-plus-filter circuits, compilation equivalence is checked only on the all-zero input state; it is not a full-unitary certificate. All results use ideal CPU simulation, without executed quantum shots or physical noise.

## Rigorous source contract and realization limits

The validator uses explicitly directed Decimal endpoint arithmetic and manual real Taylor enclosures for phase trigonometry and the exponential. It reconstructs the signal polynomial, converts monomials to Chebyshev coefficients using exact binomial factors, and bounds the continuum error by coefficient $\ell_1$ residual plus analytic Taylor remainder {cite:p}`sim_python_decimal_31214`. It does not turn a sampled phase-fit grid into a uniform certificate.

Let $\delta_e,\delta_o$ bound the two real responses, $\Delta a$ be the exact-rational parameter-rounding gap, and $w$ the certified loaded-weight $\ell_1$ error. For the declared logical encoded map,

$$\epsilon_K\le\frac{\delta_e+\delta_o}{2}+\frac{|\Delta a|}{2}+\frac{a w}{4}.$$

The last term follows from Duhamel's formula for the two shifted positive exponential contractions. The input normalization and $a$ multiplication are accounted separately from the actual PREPARE-angle probabilities. The named logical Cliffords/controls, stored phase angles and Bell H/CX gates define this certified model; Qiskit nested gate definitions are a separate realization.

For the exact target Bell branch, $s\ge e^{-2a}/4$. If $\epsilon_K<s$, a conservative perturbed acceptance bound is $(s-\epsilon_K)^2$ and a Gibbs trace-distance bound is $\min(1,2\epsilon_K/s)$. The latter remains valid after unconditioned reference discard. These are conservative source bounds; they can be much looser than the particular-instance numerical acceptance and density error.

SDK nested-control/MCX definitions and compiler-created/merged angles need additional norm bounds. Source/primitive comparisons with phase retained are numerical checks and do not supply those bounds. Rotation synthesis, physical faults and irreversible reset/readout errors require separate models. No mean-fit resource projection or error probability is substituted for a coherent norm certificate.

## Complete trial costs and size scaling

For $L$ Pauli terms and $m=\lceil\log_2L\rceil$, this preparation uses $2n+m+2$ logical wires. An even degree $d$ and odd degree $d+1$ require $d$ ordinary and one controlled full PREPARE/SELECT/unprepare access. Bell preparation adds $n$ H and $n$ CX gates. Direct indexed SELECT scans $L$ terms; for dense Pauli strings it can touch $nL$ letters, then multi-controls are lowered. Degree/query counts alone do not measure implementation cost.

```{embed} #simulation-thermal-spectral-scaling
```

The size table fixes $J=0.6,h=0.35,\beta=1.5$ and retargets $a$ and the phases as $n$ grows.

| Spins | Parameter $a$ | Even/odd degree | Wires | Complete Z-trial operations | Z-trial dependency depth | State check |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.2625 | 4/5 | 4 | 112 | 92 | Checked |
| 2 | 0.975 | 6/7 | 8 | 1,672 | 1,185 | Checked |
| 3 | 1.6875 | 8/9 | 11 | 5,868 | 4,239 | Checked |
| 4 | 2.4 | 10/11 | 13 | 8,958 | 6,551 | Unperformed |

 The $n=4$ row is circuit materialization only; its state and readout validation remain unperformed. A fixed error request is checked against the source certificate before counting. This is bounded logical scaling, with no optimal-complexity or hardware-advantage claim.

Each X/Z attempt resets every wire, prepares Bell pairs, runs all access/phase gates, measures all success bits, resets each reference without conditioning, applies the requested system basis change and measures the system. Every rejected attempt runs in full.

```{embed} #simulation-thermal-spectral-ledger
```

The raw accepted branch is used; no amplification or early-abort saving is supplied here. [Known-input amplification](thermal-amplification.md) is a separate method and cannot be assumed free.

Energy uses two bounded scores, $h\sum_jx_j$ and $J\sum_jz_jz_{j+1}$, from incompatible measurement groups. Correlated bits within a trial are allowed. The ledger allocates a total statistical error of 0.03 and failure 0.005 across groups, with a separate acceptance failure of 0.005 across the accepted-shot slots. It multiplies actual compiled group circuits by complete retry counts. For the selected instance, the proposal needs 29,116 accepted X trials and 21,391 accepted Z trials. Its numerical acceptance gives 1,014 complete attempts per accepted slot: **51,214,098 capped complete attempts**, with every rejection charged. Both compiled group trials contain 1,672 operations, including 10 resets, 6 measurements and 284 continuous rotations, with dependency depth 1,185. This is an unexecuted workload proposal based on numerical instance acceptance, not a hardware confidence claim. At fixed absolute extensive-energy error, conservative shot counts grow as $O(n^2)$ before retry and growing-degree access costs.

## Complete energy workload and conditional physical resources

The [complete-resource bundle](../../assets/generated/simulation/complete-resources/README.md) fixes $J=0.6,h=0.35,\beta=1.5$ for one and two spins. Its complete trials retain all-wire initialization, Bell preparation, every access/phase operation, all success measurements, unconditioned reference reset/discard, basis changes and every system readout. A rejected attempt still runs in full. The one-spin Hamiltonian contains only $0.35X$, so that task has one nonzero measurement group. The two-spin energy is the sum of scores $0.35(x_0+x_1)$ and $0.6z_0z_1$ from separate X/Z groups. Correlations between bits within one score are allowed.

Actual ancilla-free Ross–Selinger synthesis replaces every remaining rotation after bounded logical-line routing {cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`. Source, routed and actual Clifford+T instruments are checked against the independent normalized matrix exponential. The guarded sum of actual numerical synthesis errors is kept separate from a conditional $10^{-7}$ SDK nested-control/routing reserve. The requested synthesis allowance is $5\times10^{-5}$; numerical outcome agreement does not certify the reserve or a uniform compiler error. Exact saved sequences and their hashes define the unseeded native synthesis realization.

The target absolute energy error is 0.05: source and realization bias each have allowance 0.01, and statistics have allowance 0.03. For two spins each allowance is divided equally between X and Z groups. The total conditional failure allowance 0.01 is similarly divided among statistics, acceptance, physical faults and factory availability, then among the two groups. No source or realization norm error is interpreted as a stochastic physical fault.

For these fixed Pauli lists, every term is exactly traceless. With Bell input, $d=2^n$, $a=\beta\alpha/2$ and target filter $F/4$,

$$s_{\rm target}^2=\frac{e^{-2a}}{16d}\operatorname{Tr}(e^{-\beta H}),\qquad
\frac1d\operatorname{Tr}(e^{-\beta H})\ge
e^{-\beta\operatorname{Tr}(H)/d}=1.$$

Jensen's inequality therefore supplies the analytical lower bound $s=e^{-a}/4\le s_{\rm target}$ for this input family. The generic $e^{-2a}/4$ control bound remains recorded. The tighter bound follows from the exact trace-zero input, rather than the measured numerical acceptance. If $\epsilon$ is the encoded source error, $\eta$ is actual guarded synthesis error plus the SDK reserve, and $W$ is the scalar score width, the conditional workload uses

$$p_{\rm lower}=(s-\epsilon-\eta)^2,\qquad
B_{\rm source}\le\frac{2W\epsilon}{s},\qquad
B_{\rm realization}\le\frac{2W\eta}{s-\epsilon}.$$

The score widths are 0.7 for one-spin X, 1.4 for two-spin X and 1.2 for two-spin ZZ. Outward Hoeffding counts are 1,820 accepted samples for one-spin X, and 32,135 X plus 23,609 ZZ samples for two-spin energy. An outward interval retry calculation charges every accepted slot's capped complete attempts; the population and all physical schedules remain unexecuted. This separate confidence allocation and source/realization bound should not be substituted into the earlier fifth-increment numerical workload.

| Group | Wires | Actual T/T-dagger per attempt | Guarded synthesis estimate (rounded) | Conditional acceptance lower | Complete cap per accepted slot | Capped complete attempts |
|---|---:|---:|---:|---:|---:|---:|
| One-spin X | 4 | 5,488 | $1.2803\times10^{-5}$ | 0.0369672 | 359 | 653,380 |
| Two-spin X | 8 | 30,850 | $7.9263\times10^{-6}$ | 0.00889061 | 1,911 | 61,409,985 |
| Two-spin ZZ | 8 | 30,850 | $7.9263\times10^{-6}$ | 0.00889061 | 1,877 | 44,314,093 |

Acceptance entries are rounded downward; authoritative decimal bounds and actual synthesis values remain in the JSON records. The two-spin complete energy population is **105,724,078 capped attempts**. The recorded generic acceptance control would instead charge 745,947,244 attempts with the same saved synthesis realization and confidence allocations. This reduction changes the proven input-specific acceptance bound; it does not use empirical acceptance or remove rejected attempts. The actual ideal Clifford+T acceptance is 0.0157915007 for both two-spin groups. Their energy sum is approximately $-0.6582580972$, compared with the exact Gibbs answer $-0.6582579922$; this small numerical discrepancy does not certify the conditional norm premises.

```{embed} #simulation-complete-resources-thermal
```

The serialized physical scenario reuses the existing direct QDK 1.34.0 Trace backend and explicit macro ledger {cite:p}`resource_qdk_134`. For $w$ complete source wires, reserved data/workspace/bridge space is $(4w-1)(2d^2-1)$ qubits before adding one native factory. Every attempted preparation, discarded reference, readout, accept/retry/counter dispatch and idle code cycle is charged. T/T-dagger operations also charge capped complete factory retries, neighboring SWAP transport, injection/readout/correction, port recycling and extra handoff residence. Native output-quality faults and factory readiness are separate model contributions.

Two-spin energy uses one common distance and one common factory size, pipeline time and output error across its groups. Only candidates feasible for both X and Z are combined: elapsed times and allocated failure contributions add, while sequential peak qubits take the maximum. The per-group factory retry caps remain charged. Selecting incompatible per-group qubit/time minima would not define the reported shared-hardware workload.

These are minimum-space selections from the retained frontiers. The two-spin rows are the complete X-plus-Z energy task at a common hardware configuration.

| Task | Physical error | Controller error per decision | Distance | Peak physical qubits | All-attempt hours |
|---|---:|---:|---:|---:|---:|
| One-spin energy | $10^{-3}$ | $10^{-15}$ | 33 | 816,375 | 10,462.7 |
| One-spin energy | $10^{-4}$ | $10^{-15}$ | 17 | 204,835 | 6,099.57 |
| One-spin energy | $10^{-3}$ | $10^{-18}$ | 33 | 816,375 | 10,462.7 |
| One-spin energy | $10^{-4}$ | $10^{-18}$ | 17 | 204,835 | 6,099.57 |
| Two-spin energy | $10^{-3}$ | $10^{-15}$ | — | — | No common feasible configuration |
| Two-spin energy | $10^{-4}$ | $10^{-15}$ | — | — | No common feasible configuration |
| Two-spin energy | $10^{-3}$ | $10^{-18}$ | 41 | 1,314,151 | 14,600,446 |
| Two-spin energy | $10^{-4}$ | $10^{-18}$ | 19 | 267,491 | 7,762,795 |

At physical error $10^{-3}$, the independently minimum-space ZZ point uses distance 39, while X needs 41. The combined energy point therefore uses 41 for both, increasing ZZ runtime before summing. At $10^{-4}$ both groups share distance 19. The full reports retain three common energy Pareto points per feasible scenario, rather than only these minimum-space selections.

```{embed} #simulation-complete-resources-energy
```

All four physical/controller scenarios are retained: physical error $10^{-3}$ or $10^{-4}$, with the existing controller failure $10^{-15}$ per decision or the separate conditional premise $10^{-18}$. The report preserves infeasible scenarios. The distance-independent controller term is an additive model contribution; if it exceeds its allocated budget, a larger distance or factory search cannot repair that scenario. It is not a calibrated hardware failure lower bound.

For the two-spin groups at controller failure $10^{-15}$, those controller contributions are 0.379659 (X) and 0.273966 (ZZ), each exceeding its 0.00125 physical allocation. Their combined contribution is 0.653624 against the energy task's 0.0025 physical budget. The one-spin contribution is 0.000514620 against 0.0025. The conditional $10^{-18}$ premise lowers only this controller scenario parameter; all remaining physical and timing assumptions are held fixed.

Timing remains 100 ns gates/initialization, 500 ns measurement, 1,000 ns code cycles and 10,000 ns controller reaction. Protected Pauli-parity geometry, tracked Clifford frames and compatible factory/data handoff remain assumptions {cite:p}`resource_litinski_2019,resource_horsman_2012`. Memory, residual initialization and independent majority-path readout are distinct uncalibrated scenarios. Offline phase/coefficient and synthesis preprocessing, final statistical reporting, layout certification and decoder throughput are outside quantum runtime. This study separately caps total wires at eight, source instructions at 20,000, synthesized/check instructions at 500,000, weighted simulator branches at 256 and calculated complete attempts at $10^{12}$. It leaves the frozen Grover/PSSPC baseline unchanged; the larger logical sweep has no new physical estimate.

- {download}`Complete-resource notebook <../../notebooks/simulation_complete_resources.ipynb>`
- [Complete source/error/workload records](../../assets/generated/simulation/complete-resources/results.json), [physical summary](../../assets/generated/simulation/complete-resources/summary.csv), [manifest](../../assets/generated/simulation/complete-resources/manifest.json)
- [Bounded generator](../../assets/generated/simulation/complete-resources/generate_simulation_complete_resources.py), [workload implementation](../../assets/generated/simulation/complete-resources/complete_resource_workloads.py), [instrument backend](../../assets/generated/simulation/complete-resources/instrument_resources.py)
- Two-spin actual Clifford+T trials: [X](../../assets/generated/simulation/complete-resources/thermal-n2-X/clifford-t-instrument.qasm), [ZZ](../../assets/generated/simulation/complete-resources/thermal-n2-Z/clifford-t-instrument.qasm)

## Reproduce and download

```bash
python scripts/generate_simulation_thermal_spectral.py
pytest tests/test_simulation_thermal_spectral.py
python scripts/generate_simulation_thermal_spectral.py --n-system 4 --beta 1.5 \
  --no-sim --no-scaling --output-dir artifacts/thermal-spectral-n4
```

The source permits eight systems, 32 terms, 24 total wires, parameter $a\le16$, and adjacent degrees through 30/31 under the fitter's degree-32 cap. Top-level source and primitive export caps are 50,000 and 250,000 operations. Statevectors stop at 12 total wires, filter Operator checks at eight, dense system references at six, artifact grids at 12 cases and combined exports at one million operations. Numerical conditional-state checks also reject acceptance at or below $10^{-28}$. Unsupported parameters/errors fail closed. Dense density storage grows as $4^n$; neither larger simulation nor general thermal mixing is claimed.

`--no-sim` still generates and validates phases and materializes circuits. It marks circuit state/reference/readout checks and numerical retry workloads unperformed, removing managed stale validation files. Optional Git provenance is recorded only for the exact checkout root; source hashes remain usable in downloaded packages with no Git history.

- {download}`Executable notebook <../../notebooks/simulation_thermal_spectral.ipynb>`
- [Original thermal source](../../assets/generated/simulation/thermal-spectral/thermal_spectral.py), [shared exact access](../../assets/generated/simulation/thermal-spectral/retargetable_qsp.py), [phase fitter](../../assets/generated/simulation/thermal-spectral/qsp_phase_generation.py), [validator](../../assets/generated/simulation/thermal-spectral/qsp_certificate.py), [standalone generator](../../assets/generated/simulation/thermal-spectral/generate_simulation_thermal_spectral.py)
- [Results and source contracts](../../assets/generated/simulation/thermal-spectral/results.json), [logical ledger](../../assets/generated/simulation/thermal-spectral/logical-resources.json), [checked grid](../../assets/generated/simulation/thermal-spectral/accuracy.csv), [size counts](../../assets/generated/simulation/thermal-spectral/scaling.csv)
- [Selected coherent QASM](../../assets/generated/simulation/thermal-spectral/n2-beta-1p5-coherent.qasm), [X trial](../../assets/generated/simulation/thermal-spectral/n2-beta-1p5-trial-X.qasm), [Z trial](../../assets/generated/simulation/thermal-spectral/n2-beta-1p5-trial-Z.qasm), [MIT license](../../assets/generated/simulation/thermal-spectral/LICENSE.txt), [locked environment](../../requirements-lock.txt)

The pinned exporter omits overall circuit phase. Measured probabilities are preserved; coherently controlled reuse needs the source or explicit restoration of the recorded phase. Text, phases, code and figures are original. Exact Qiskit 2.6.2/Apache-2.0 and SciPy 1.18.1/BSD-3-Clause sources are credited in the simulation bibliography. Formal/Lean evidence and a complete physical implementation remain planned.
