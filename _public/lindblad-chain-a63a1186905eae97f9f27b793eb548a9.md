---
title: "Coupled-spin Lindblad dynamics"
description: "Checked coupled-spin dynamics, an n-chain gate generator and bounded two/three-spin physical workloads."
---

An interaction can correlate two spins before local damping acts on them. Treating each spin independently then misses the joint state and observables. This example advances the [single-qubit Lindblad method](lindblad.md) to a coupled two-spin model, using exact local components and a symmetric global schedule.

:::{note} Status
The two-spin gate/reset circuits are compared with an independently assembled Liouvillian exponential. Product and Bell inputs, local observables, ZZ and grouped energy readout are checked. A parameterized open-chain generator supports up to 16 systems; its larger circuit counts are executed separately from the bounded density study. A bounded local-observable study below covers $n=2,3,4$ with an exact-angle step policy and numerical reduced-state checks. The n2/r69 and n3/r134 physical slices are preserved. The separate literal n4/r211 trial has completed density/Clifford+T checks and an explicit controller-model rejection; its follow-on physical-wire-0 port has completed conditional reports for both frozen profiles. Full finite-angle certification, hardware validation and formal/Lean coverage remain planned.
:::

## Task and intuition

For two systems, set $\hbar=1$ and

$$H=0.6Z_0Z_1+0.35(X_0+X_1),\qquad
L_j=\sqrt{\gamma_j}|0\rangle_j\langle1|,$$

with **$(\gamma_0,\gamma_1)=(0.8,0.5)$** and $t=1$. These are asymmetric model damping rates, not device-fault probabilities. Hamiltonian coefficients and rates are in inverse-time units. The initial product state has $|+i\rangle$ on system0 and $|+\rangle$ on system1; an explicitly prepared Bell state is also checked. Displayed system bitstrings read q1q0.

The target is the reduced density operator obeying the standard GKSL equation

$$\dot\rho=-i[H,\rho]+\sum_{j=0}^1\left(L_j\rho L_j^\dagger-
\tfrac12\{L_j^\dagger L_j,\rho\}\right).$$

An observable task instead requests local X/Y/Z, ZZ, or the energy mean after this evolution. A retained state is defined before final system measurement. The exported circuits produce measured trials and ignore both environment outcomes.

GKSL theory supplies the model context {cite:p}`sim_lindblad_1976,sim_gks_1976`. Digital simulation alternates short channels whose individual construction is transparent. The symmetric sequence cancels the leading ordering bias while preserving positive substep times; the noncommuting components still produce a finite splitting error.

## Actual components and global order

The ZZ component uses CX–$R_Z(2Js)$–CX. The X layer uses $R_X(2h_xs)$ on each system; its local X terms commute. Each damping component uses the [survival-aware environment dilation](open-systems.md), with a reset immediately before that environment's next use. No dense Hamiltonian or channel matrix is a quantum gate.

For $\delta=t/r$, Lie order is chronological

$$ZZ(\delta),\ X(\delta),\ D_0(\delta),\ D_1(\delta).$$

The global second-order palindrome is

$$ZZ(\delta/2),\ X(\delta/2),\ D_0(\delta/2),\ D_1(\delta),\
D_0(\delta/2),\ X(\delta/2),\ ZZ(\delta/2).$$

Every component receives total time $\delta$. A half damping step has $p_j=1-e^{-\gamma_j\delta/2}$; it does not halve the full-step probability. Every substep is a valid positive-time channel. The two local damping channels commute, but this literal circuit still executes two $D_0$ halves and one $D_1$ full step.

```{embed} #simulation-two-spin-lindblad-circuit
```

System wires are q0/q1 and environments q2/q3. At $r$ symmetric steps, environment0 is initialized $2r$ times and environment1 $r$ times. Of these, $2r-1$ and $r-1$ resets discard previous states before reuse. Known-input trials also reset both systems and include final measurements on all four wires. There is no environment postselection.

This is a **global** splitting over Hamiltonian terms and dissipators. It does not construct an exact combined-H substep with a separately added inner error. The full error already includes Hamiltonian-term and Hamiltonian/damping commutators. A damping-disabled H-only diagnostic is reported for interpretation and is not added again.

Lie–Trotter, Strang and Liouvillian product-formula literature provide context {cite:p}`sim_trotter_1959,sim_strang_1968,sim_liouvillian_trotter_2011,sim_liouvillian_trotter_erratum_2012`. Their norm and dissipator conventions must match before importing constants; the numerical results below are not theorem or diamond-norm certificates.

## Checked state and observables

The independent reference is a $16\times16$ column-vectorized GKSL generator. It is assembled from time-scaled coefficients before exponentiation, avoiding large intermediate entries for short-time/large-coefficient inputs. Qiskit DensityMatrix evaluates the actual four-wire gates and resets, then traces q2/q3. The density reference is classical verification, not the quantum implementation {cite:p}`sim_qiskit_262,sim_scipy_1181,sim_expm_2009`.

Another independent matrix calculation composes the stated exact component channels and checks that the circuit implements the advertised schedule. Tests cover product, Bell and small mixed inputs, trace, positivity, local X/Y/Z, ZZ and energy; removing required environment resets changes the channel. Mixed density matrices supplied to a checker initialize that classical simulator and do not implement arbitrary quantum input loading.

```{embed} #simulation-two-spin-lindblad-results
```

```{embed} #simulation-two-spin-lindblad-convergence
```

At 32 symmetric steps, the product-input trace distance is approximately **$8.3511\times10^{-5}$** and the energy bias **$2.7376\times10^{-5}$**. In the tested range, doubling steps approximately halves Lie state error and quarters symmetric state error. These checks distinguish accuracy of the target GKSL evolution from correctness of the individual gate sequence.

```{embed} #simulation-two-spin-lindblad-density
```

## Analysis and verification: complete energy workload

X/Y/Z readout applies the appropriate basis change on each system, measures all wires, and ignores environment bits. Energy uses two incompatible measurement groups:

- X group: one joint outcome gives score $h_x(x_0+x_1)\in[-0.7,0.7]$.
- Z group: one joint outcome gives score $Jz_0z_1\in[-0.6,0.6]$.

The bits within one trial can be correlated. Hoeffding applies to the complete bounded scalar score from each independent trial, without assuming independent spins. With statistical error 0.015 and failure probability 0.0025 per group, the proposal uses **29,116 X trials and 21,391 Z trials: 50,507 complete circuits**. A union bound supplies total sampling error 0.03 and failure 0.005 for one fixed energy mean. Other local observables or settings need separate confidence allocations.

```{embed} #simulation-two-spin-lindblad-ledger
```

The logical ledger includes data preparation, all Hamiltonian/damping gates, every environment reset/discard, basis rotations and system/environment measurements for every group trial. It repeats the actual compiled counts across both shot allocations and records a logical qubit-layer volume. That volume is a dependency-layer proxy, not physical qubit-seconds: simulated time $t$ and gate depth are not hardware elapsed time.

The proposed total energy-error target is 0.05, split into 0.01 global systematic bias, 0.01 coherent synthesis bias and 0.03 statistics. A coefficient bound $\|H\|\leq |J|+2|h_x|=1.3$ converts selected-state trace distance $d$ to an energy-bias bound $2(1.3)d$. The default $r=32$ check meets that systematic allowance numerically. It does not certify all inputs, parameters or larger systems.

Physical-fault probability is separately proposed at 0.005 across all 50,507 trials. Continuous-angle synthesis is pending for this energy task. Intended damping, splitting error, representation rounding, coherent approximation, statistical confidence and physical faults are distinct. The static physical adapter rejects mid-circuit resets; no complete physical estimate is supplied for this r32 energy workload.

Classical coefficient/angle preparation and readout processing, physical reset/discard/measurement time, routing, factories and code distance are excluded from the logical cost. Classical construction/decomposition and small density-check wall times are recorded as CPU measurements, not quantum performance. No quantum advantage is claimed for a classically tiny instance. Formal/Lean proof coverage is **not integrated**.

## Size, accuracy and empirical scaling

For a general open chain, the executable generator uses $H=J\sum Z_jZ_{j+1}+h_x\sum X_j$ and one local damping environment per system. Its logical space is **$2n$ wires**. Before optimization, complete measured-source upper counts are

$$C_1\leq(10n-3)r+6n+8,\qquad C_2\leq(20n-12)r+6n+8.$$

The overhead includes preparation, system initialization, readout bases and all measurements; compiled operation counts are checked against an additional cap. Source calls therefore scale as $O(nr)$ at fixed $r$. Actual construction/decomposition is run for $n=1,2,3,4,8$ and several step counts.

```{embed} #simulation-two-spin-lindblad-scaling
```

The original scaling family uses the product input, order2, rate0.8 on site0 and rate0.5 elsewhere, with the chosen time/J/hx; those assumptions are separate from a custom selected instance. Its full combined-GKSL accuracy curves are **n=2**, with a one-system check and one six-wire three-system component-sequence check. Those fixed-r circuit counts are separate from the bounded equal-accuracy local-observable workloads below.

For fixed coefficients and absolute energy statistical error, general-n X/Z score widths are $2n|h_x|$ and $2(n-1)|J|$. A conservative grouped-shot bound grows as $O(n^2\epsilon_{stat}^{-2}\log(1/\delta_{stat}))$. Combining it with fixed-r $O(nr)$ gates gives an **unexecuted conditional** workload bound $O(n^3r\epsilon_{stat}^{-2}\log(1/\delta_{stat}))$. Selecting $r(n)$ for this extensive-energy output still requires its own systematic-error policy; the local $Z_0$ study below uses a unit-norm observable and a separate shot allocation. This is not an optimal complexity or fault-tolerant runtime claim.

Dense density memory grows as $4^n$, and a dense Liouvillian contains $16^n$ entries. The original CPU checks use n=2 and fixed small n=3 functionality. The separate study below adds bounded $n=2,3,4$ reduced-channel accuracy checks; larger-size validation, tensor-network alternatives and physical scheduling remain planned. A gate generator alone does not imply affordable classical or hardware execution.

(simulation-lindblad-chain-accuracy-workload)=
### Local-observable workloads at preserved ideal accuracy

The [bounded workload study](../../assets/generated/simulation/lindblad-workload/README.md)
materializes **$n=2,3,4$** at a common output contract. It keeps $J=0.6$,
$h_x=0.35$, $t=1$, rate 0.8 at site 0 and rate 0.5 elsewhere, with
$|+i\rangle$ on system 0 and $|+\rangle$ on the other systems. Each output is
one final **$\langle Z_0\rangle$**, measured from a complete fresh-input trial;
all other system and environment outcomes are ignored. This local-output task
has a constant score range and a different measurement budget from the total
energy task above.

The step count is chosen from an analytic bound before inspecting numerical
errors. Use the trace norm on Hermitian matrices and its induced map norm.
Hamiltonian commutators have norm at most $2\|H\|_\infty$ (operator norm), and a standard-GKSL local
lowering dissipator has norm at most $2\gamma_j$. A component envelope is

$$A_n=2(n-1)|J|+2n|h_x|+2\sum_j\gamma_j.$$

Here $F(s)$ is the literal reduced-system channel palindrome after environment
discard, not the joint dilation unitary whose small-time angle has square-root
behavior. Each component has total weight one, so $F'(0)=G=\sum_c\mathcal L_c$;
palindrome symmetry gives $F''(0)=G^2$. Every positive-time component
exponential is CPTP and contractive. The product-rule multinomial expansion
gives $\|F'''(s)\|\leq A_n^3$. The exact channel $E(s)$ has the same third
derivative bound, and palindrome symmetry makes their derivatives through
order two agree at zero. Applying both integral Taylor remainders and then
telescoping the $r$ contractive steps yields

$$\|F(t/r)^r(\rho)-e^{t\mathcal L}(\rho)\|_1
 \leq B_n(r)=\min\!\left(2,\frac{(A_nt)^3}{3r^2}\right).$$

This is a uniform **exact-arithmetic, exact-substep** evolution bound for all
input density operators under the stated time-independent positive-time model.
It excludes preparation faults, angle rounding, coherent synthesis and physical
faults. It is a conservative original derivation for this schedule, not an
imported Liouvillian commutator constant or a formal/Lean proof. The literature
provides the surrounding method context {cite:p}`sim_strang_1968,sim_liouvillian_trotter_2011,sim_liouvillian_trotter_erratum_2012`.
A unit-norm local Pauli bias is bounded by $B_n$, while state trace distance is
bounded by $B_n/2$.

Exact rational arithmetic selects the smallest $r$ with $B_n(r)\leq0.01$.
Statistical error 0.03 per mean and family failure 0.005 are allocated across
the three outputs: each case has failure $1/600$ and **15,756 complete trials**.
Hoeffding's threshold is checked with a positive exponential Taylor sum in
exact rational arithmetic. In the ideal-angle model, a union bound gives all
three means absolute error at most **0.04 each**, with total sampling failure
at most 0.005. The trial ensembles are calculated and unexecuted; finite-angle
or hardware confidence is not certified.

| Systems $n$ | Selected $r$ | Logical wires | Uniform trace-norm bound | Compiled operations/trial | Complete workload operations |
|---:|---:|---:|---:|---:|---:|
| 2 | 69 | 4 | 0.009844 | 1,737 | 27,368,172 |
| 3 | 134 | 6 | 0.009866 | 6,445 | 101,547,420 |
| 4 | 211 | 8 | 0.009965 | 14,365 | 226,334,940 |

![Logical steps and complete operation counts for the fixed local observable.](../../assets/generated/simulation/lindblad-workload/workload-scaling.svg)

The [two-system report](../../assets/generated/simulation/lindblad-workload/n2/report.json),
[three-system report](../../assets/generated/simulation/lindblad-workload/n3/report.json)
and [four-system report](../../assets/generated/simulation/lindblad-workload/n4/report.json)
include the exact policy fractions, previous-integer failures, numerical checks
and whole-trial ledgers. Counts include preparation, all literal damping
halves, environment reuse/reset, basis/readout operations and every trial.
Reset counts are 209, 673 and 1,481 per trial; all 4, 6 or 8 wires are measured.
Lowering uses the pinned rotation/CX basis with optimization level 1 and seed 7;
the table uses executed compiled counts, with compiler barriers recorded separately.

At the selected steps, the observed selected-input state trace-norm errors are
approximately $3.59\times10^{-5}$, $1.39\times10^{-5}$ and $6.66\times10^{-6}$.
These numerical values are separate from the uniform analytic bound. The exact
reference is an independently assembled full GKSL exponential. The implemented
reduced schedule is composed from matrix operators of the actual ZZ/RX and
environment-dilation gates. Each environment can be traced after its dilation,
because it is untouched until its next reset or ignored readout. Short whole
$2n$-wire density simulations check that causal contraction. The selected long
workloads use powers of the gate-derived reduced channel, not whole-device
gate-by-gate density simulation. Dense system states stop at $16\times16$;
channel/Liouvillian matrices at $256\times256$; whole-circuit checks at four steps.
No dense matrix is appended as a quantum operation.

For fixed coefficients the conservative envelope has $A_n=O(n)$, giving
$r=O(n^{3/2}t^{3/2}/\sqrt{\epsilon_{split}})$ and $O(nr)$ logical work per
trial. Fixed local readout requires
$O(\epsilon_{stat}^{-2}\log(3/\delta_{family}))$ trials per case. This gives a
conditional algebraic workload scaling under the stated assumptions; the
materialized evidence remains the three bounded sizes and is not optimal
locality-aware complexity or a physical scaling result.

:::{note} Scope of this logical family
The logical-family table supplies no physical qubits, code distance or hardware elapsed time.
The [single-spin conditional physical study](lindblad.md#simulation-lindblad-physical-study) covers its fixed two-wire task. The instrument backend defaults to five source wires and also has opt-in weighted-Kraus checks and QSP/thermal resource reports for other circuits. This logical wrapper itself supplies no coherent-synthesis or full physical report for its family. The following four-wire slice and explicit six-wire density slice have separate checks. This preserved wrapper has no eight-wire physical layer; the later literal n4 and port sections use separate opt-in domains. Extending the original wrapper requires actual synthesis/error budgets,
wider reset-aware instrument checks, routed schedules and explicit data/factory
geometry plus controller/reset/readout/idle-fault accounting. Changing a width
constant would not validate those assumptions. The following physical slices are limited to the literal n2/r69 and n3/r134 tasks.
:::

Offline coefficient/circuit construction and final statistical-processing wall
time, Clifford+T synthesis, physical routing, factories, error correction and
physical reset/readout/controller faults/times are excluded from the logical
counts. CPU check times and logical qubit-layer proxies have separate labels.
The frozen Grover baseline is unchanged; this is no matched co-design or
advantage claim.

- Source trials: [2 systems](../../assets/generated/simulation/lindblad-workload/n2/source-trial.qasm), [3 systems](../../assets/generated/simulation/lindblad-workload/n3/source-trial.qasm), [4 systems](../../assets/generated/simulation/lindblad-workload/n4/source-trial.qasm)
- Rotation/CX trials: [2 systems](../../assets/generated/simulation/lindblad-workload/n2/rotation-basis-trial.qasm), [3 systems](../../assets/generated/simulation/lindblad-workload/n3/rotation-basis-trial.qasm), [4 systems](../../assets/generated/simulation/lindblad-workload/n4/rotation-basis-trial.qasm)
- [Reusable helper](../../assets/generated/simulation/lindblad-workload/lindblad_workload.py), [generator](../../assets/generated/simulation/lindblad-workload/generate_simulation_lindblad_workload.py), [manifest](../../assets/generated/simulation/lindblad-workload/manifest.json), [summary CSV](../../assets/generated/simulation/lindblad-workload/summary.csv), [numerical convergence](../../assets/generated/simulation/lindblad-workload/convergence.csv)

(simulation-lindblad-chain-physical-workload)=
### Complete physical accounting for the literal two-spin workload

The [bounded physical bundle](../../assets/generated/simulation/lindblad-physical/README.md) fixes the local-output task above at **n=2, r=69 and 15,756 complete trials**. It keeps the +i/+ product input, $J=0.6$, $h_x=0.35$, rates 0.8/0.5, $t=1$ and final $Z_0$ score. Every trial retains both system initializations, all 138/69 environment initializations, each damping half and every readout. There are **209 resets and four measurements per trial**, or 3,293,004 resets and 63,024 readouts in the calculated population. There is no algorithm postselection or retry saving. The trial ensemble and hardware schedule remain unexecuted.

![Actual initialization and first literal source period at dt=1/69; the source continues for all 69 periods before final readout.](../../assets/generated/simulation/lindblad-physical/one-step.svg)

The four-wire density path evaluates the literal source/reset circuit, rather than expanding reset histories into a pure-state ensemble. Setting the generic eight-wire limit would redirect even this small source into the branch-limited checker. The study instead declares `max_qubits=4`, 20,000 source instructions and 500,000 synthesized/validation instructions. The existing shared checker is unchanged.

The wider literal schedules retain the prior explicit ensemble rejection records. Their 6/8-wire sources fit instruction preflight limits, but the 256-branch ensemble rejects n3/r134 at step3 and n4/r211 at step2. The same ensemble rejects n2/r69 at step4; its separate four-wire density path is the supported route here. No nonzero branch is pruned, and short schedules that fit the ensemble do not replace the uniform-accuracy step choices. The new n3 density slice below supplies a separate supported route. In this preserved bundle, n4 synthesis and estimates remain unimplemented; the later literal n4 and port records use their own opt-in domains.

```{embed} #simulation-lindblad-physical-source
```

The source and routed ideal means are approximately 0.7911400047. The actual Clifford+T mean is **0.7911402242**, versus the independent GKSL reference **0.7911337404**. The synthesized trial contains 204,807 non-barrier instructions, including **78,108 T/T-dagger** and 903 neighboring CNOTs; 426 compiler event barriers remain recorded for instruction caps. Multiplying the actual non-barrier counts gives 3,226,939,092 primitive occurrences across all 15,756 trials. These numerical checks and counts have different roles from a norm or hardware certificate.

The circuit is lowered with full-unitary compiler settings, routed on the bidirectional logical line with fixed initial layout and SABRE seed 7, then every remaining rotation is replaced by an actual ancilla-free Ross–Selinger Clifford+T sequence {cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`. Reset/readout boundaries are retained. The optimization-level-1 logical table above is a different compilation record: its 760 n2 rotation count is not the allocation for this full-unitary level 0 realization. Exact generated sequences and QASM hashes identify the synthesis; the native backend exposes no seed API. Measured-trial QASM exports omit overall phase; coherent controlled reuse of unitary segments needs source or explicit phase restoration.

The total absolute output-error target is **0.05**: splitting bias 0.01, realization bias 0.01 and statistical error 0.03. The source step policy remains the exact-decimal, exact-substep statement above. Stored binary64 sub-times and damping angles, SDK control/routing and numerical Clifford+T synthesis are separate realization stages. The record gives exact rational input and Hamiltonian-angle discrepancies; the damping-angle functions and full representation norm remain uncertified. The requested whole-circuit synthesis allowance is $5\times10^{-5}$; the guarded error sum of the actual generated sequences is retained. Separate $10^{-7}$ representation and $10^{-7}$ SDK/control/routing reserves are conditional premises, not interval or formal certificates.

If $\eta$ bounds the sum of those coherent realization contributions, contraction under the ideal reset/discard channels gives an unconditional $Z_0$ mean-change bound $2\eta$. It has no acceptance denominator and is not multiplied by the trial count. Source/routed/synthesized density agreement is bounded numerical evidence; it does not prove either reserve or a uniform compiler norm bound. Coherent approximation is not charged as a stochastic physical fault.

The saved guarded numerical synthesis sum is approximately $5.9539\times10^{-6}$, so the aggregate realization premise with both reserves is $6.1539\times10^{-6}$. Its Z0 mean-bias proposal is $1.2308\times10^{-5}$. Combining it with the uniform source bound and statistical allowance gives a conditional absolute-error proposal below **0.0398568**, rounded outward and below the retained 0.05 target. Six native rotation-synthesis attempts produced the six unique targets; every repeated sequence occurrence is still charged.

```{embed} #simulation-lindblad-physical-budget
```

The three-case statistical allocation remains 0.005, with $1/600$ per mean and 15,756 trials per case. Physical faults and factory readiness receive separate family allowances 0.0025 each, or $1/1200$ per case each. The n2 calculation uses those unchanged shares. In its preserved bundle, n3/n4 shares remain unused. The new n3 slice below uses its own unchanged allocation, and n4 remains unused; there is no simultaneous hardware-confidence claim for the full family. Counts, physical faults and factory availability multiply across the complete n2 population; deterministic approximation bias does not.

```{embed} #simulation-lindblad-physical-workload
```

The physical report reuses the serialized instrument scenario with direct QDK 1.34.0 Trace/ledger agreement {cite:p}`resource_qdk_134`. It retains the two frozen physical-error profiles $10^{-3}$ and $10^{-4}$ with controller failure $10^{-15}$ per decision. These profiles are alternative physical assumptions for the same complete task. Gates and initialization take 100 ns, measurement 500 ns, code cycles 1,000 ns and controller reaction 10,000 ns. Cap rejections and physical-model infeasibility remain explicit instead of changing assumptions or omitting cost.

Reserved data/workspace/bridge space uses 15 logical patches for four source wires, each of area $2d^2-1$, plus one native factory. The 209 source resets include the four source-wire first resets; 11 additional workspace/bridge initializations are charged separately per complete trial. Startup, every source reset, all ignored readouts, trial dispatch, active/idle code exposure and routed neighboring CNOTs are charged. Each T/T-dagger also includes full capped factory retries, transport SWAPs, injection/readout/corrective-frame work, port recycling and additional handoff residence. Native factory output quality is charged once; its readiness allowance is separate.

```{embed} #simulation-lindblad-physical-resources
```

The minimum-space points below include all complete trials and complete factory retries. Both feasible profiles retain three qubit/time Pareto points.

| Physical error | Code distance | Peak physical qubits | All-trial hours | Additive physical-fault model | Factory-readiness model |
|---:|---:|---:|---:|---:|---:|
| $10^{-3}$ | 33 | 816,375 | 3,336.90 | $3.3476\times10^{-4}$ | $1.2307\times10^{-5}$ |
| $10^{-4}$ | 17 | 204,835 | 1,928.89 | $1.6260\times10^{-4}$ | $1.2307\times10^{-5}$ |

The 1,230,669,648 required magic states each have a full pipeline retry cap of seven, giving 8,614,687,536 capped pipeline attempts. Readiness is bounded separately by the modeled 1% pipeline failure raised to that cap and union-bounded over all required states. The all-trial controller contribution is approximately 0.000152073 in either profile, within the unchanged $1/1200$ physical allocation. It is an additive engineering-model term, not an observed hardware-failure lower bound.

```{embed} #simulation-lindblad-physical-faults
```

Protected Pauli-parity geometry, tracked Clifford frames, compatible factory/data handoff, memory and residual initialization models and independent majority-path readout remain conditional engineering assumptions {cite:p}`resource_litinski_2019,resource_horsman_2012`. Offline coefficient/circuit/synthesis preprocessing and final statistical-reporting wall time are excluded from quantum runtime. Layout certification, decoder throughput and measured hardware calibration remain unavailable. This is a separate serialized scenario preserving the frozen Grover/PSSPC baseline; it is not a matched co-design or advantage claim.

The run is predeclared as one prepared circuit and two fixed physical profiles. Source diagnostics have a 60 s limit, the preparation/synthesis/check stage 600 s, each physical profile 180 s and the parent generation 1,200 s. Native rotation synthesis permits at most 3,312 attempts including retries. Each profile permits at most 24 top-level native factory and 24 top-level surface-code queries, with at most 1,000 factory candidates per query and 24,000 direct Trace estimates. Exceeding a cap rejects the complete stage; candidates are not silently truncated. Odd code distances stop at 49 and complete factory retries at 16. Managed JSON/QASM files stop at 32 MiB. These limits bound this small resource study rather than a wider-chain extrapolation.

- {download}`Complete physical notebook <../../notebooks/simulation_lindblad_physical.ipynb>`
- [Actual source/error/workload records](../../assets/generated/simulation/lindblad-physical/results.json), [physical summary](../../assets/generated/simulation/lindblad-physical/summary.csv), [manifest](../../assets/generated/simulation/lindblad-physical/manifest.json)
- [Bounded generator](../../assets/generated/simulation/lindblad-physical/generate_simulation_lindblad_physical.py), [workload implementation](../../assets/generated/simulation/lindblad-physical/lindblad_physical_workload.py), [instrument backend](../../assets/generated/simulation/lindblad-physical/instrument_resources.py)
- [Complete source QASM](../../assets/generated/simulation/lindblad-physical/source-trial.qasm) and [actual Clifford+T QASM](../../assets/generated/simulation/lindblad-physical/clifford-t-trial.qasm)

(simulation-lindblad-chain-three-physical-workload)=
### Literal three-spin workload with full density checks

The [three-system bundle](../../assets/generated/simulation/lindblad-three/README.md)
extends the same local-output contract to **n=3, r=134 and 15,756 complete
trials**. It keeps $t=1$, $J=0.6$, $h_x=0.35$, rates 0.8/0.5/0.5 and the
$|+i\rangle/|+\rangle/|+\rangle$ input. The Hamiltonian is
$H=0.6(Z_0Z_1+Z_1Z_2)+0.35(X_0+X_1+X_2)$. One trial gives one final
$Z_0$ score in $\{-1,+1\}$; the other system and environment readouts are
ignored for that score. All three system outcomes remain correlated in the
checker. The complete trial ensemble and hardware schedule are unexecuted.

The literal source has **4,435 instructions**, including **673 resets and six
readouts**. System preparation resets all three system wires. Environments
q3/q4/q5 are initialized 268/268/134 times; 267/267/133 of those resets discard
their previous states before reuse. The calculated population therefore has
**10,603,788 source resets and 94,536 readouts**. There is no algorithm
postselection, trajectory selection or shortened damping schedule.

![Initialization and the first literal three-system period; all 134 periods precede complete readout.](../../assets/generated/simulation/lindblad-three/one-step.svg)

The opt-in checker evolves the entire six-wire $64\times64$ density operator.
An unobserved reset on wire q applies the two Kraus terms

$$\mathcal R_q(\rho)=\sum_{b=0}^1
 (|0\rangle\langle b|)_q\,\rho\,(|b\rangle\langle0|)_q.$$

It sums both terms without recording a reset outcome. Ignored explicit
readouts dephase their wire. No reset history becomes a measured bit, and no
nonzero trajectory or physical operation is pruned. The checker may multiply
consecutive one-qubit matrices to reduce classical verification work; the
source and exported Clifford+T schedule still count every occurrence. The
independent reference composes matrices from the actual local gates and
environment dilations, and separately exponentiates the full GKSL generator
{cite:p}`sim_qiskit_262,sim_scipy_1181,sim_expm_2009`.

```{embed} #simulation-lindblad-three-source
```

Before system readout, the checker retains the full $8\times8$ system density
and all eight probabilities in q2q1q0 order. It compares these objects across
source, routed and actual Clifford+T schedules, as well as comparing the local
mean. The literal-source mean is **0.7910526261**, versus the independent GKSL
mean **0.7910509390**. The source versus gate-derived system trace-norm
residual is approximately $4.25\times10^{-14}$; its trace-norm error against
GKSL is approximately $1.3904\times10^{-5}$. These are bounded selected-input
numerical checks with distinct scope from the uniform source bound and the
conditional realization reserves.

A product of the three single-system marginals preserves the same $Z_0$ mean
but loses the joint state. For the literal source, $\langle Z_0Z_1\rangle$ is
approximately **0.43890186**, versus **0.36695894** from that product;
$\langle Z_1Z_2\rangle$ is **0.4570158**, versus **0.1891117**. The product
state's trace-norm discrepancy is approximately **0.7925898**. This diagnostic
explains why local-mean agreement alone cannot validate the interacting
channel. It does not replace the full state with a cheaper quantum circuit.

```{embed} #simulation-lindblad-three-correlations
```

The new checker keeps defaults at five wires, 10,000 source instructions and
100,000 validation instructions. This fixed case explicitly opts into six
wires, 20,000 source instructions and 1,000,000 routed/synthesized/validation
instructions. Its hard density width is six. A retained complex128 density
matrix occupies **64 KiB**; this is a single-array size, not a measured process
peak. The unitary-matrix cache has a 1 MiB default and 16 MiB hard ceiling.
Instruction and definition caps also apply to generic unitary operations. The
earlier ensemble backend and frozen n2 records are unchanged. The n4/r211
source is rejected by the six-wire width limit and by its separate 256-branch
ensemble check; n4 synthesis and physical estimation remain unimplemented.

The complete source is lowered at optimization level 0 with full-unitary
settings, fixed initial layout and SABRE seed 7 on the bidirectional six-wire
line. The local lowering has **6,445 non-barrier operations** and 1,358 event
barriers, with **2,680 rotation occurrences**: 1,340 RY, 804 RX and 536 RZ.
Routing gives **5,524 neighboring CNOTs**, including 2,442 routing additions,
and 10,245 instructions including barriers. The initialized-wire schedule
passes without extra resets. Every remaining rotation receives an actual
ancilla-free Ross–Selinger sequence; repeated occurrences are charged even
when the native synthesis target is cached
{cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`.
The exported source, routed-rotation and Clifford+T QASM identify three
different compilation stages. Measured-instrument QASM omits overall phase
and is not a controlled-oracle specification. Coherent reuse of a unitary
segment requires its phase convention and recorded in-memory phase.

The actual Clifford+T mean is **0.7910523096**. Its system trace-norm
discrepancy from the source is approximately $2.1483\times10^{-6}$, and its
trace-norm error against GKSL is approximately $1.2711\times10^{-5}$.
The complete emitted circuit has **707,581 instructions**, including 1,358
barriers: **706,223 non-barrier primitives** per trial, with **274,432 T**
gates and 5,524 neighboring CNOTs. The calculated population contains
**11,127,249,588 primitive occurrences**. Eight native synthesis calls
produce eight unique rotation targets; the repeated sequences all remain
in these counts.

```{embed} #simulation-lindblad-three-workload
```

The accuracy target and family allocations are unchanged: **0.01 splitting,
0.01 realization and 0.03 statistics**, with statistical failure $1/600$,
physical-fault allowance $1/1200$ and factory-readiness allowance $1/1200$ for
this case. The source step count still follows the exact-rational policy,
giving uniform splitting bias below 0.01. The guarded actual synthesis sum
and separate $10^{-7}$ representation and $10^{-7}$ SDK/control/routing
reserves form the coherent premise $\eta$, whose unconditional $Z_0$ bias is
at most $2\eta$. Neither reserve follows from numerical density agreement.
Exact binary-input and Hamiltonian-angle discrepancies have narrower
coverage; decimal damping-angle enclosure and full representation, compiler,
formal synthesis and hardware-confidence certificates are unavailable.
Deterministic bias is not multiplied by the trial count. The unused n4
allocations are not reassigned.

The saved guarded synthesis sum is approximately $8.62305\times10^{-6}$.
With both conditional reserves, $\eta$ is $8.82305\times10^{-6}$ and the
realization-bias proposal is $1.76461\times10^{-5}$. Combining that proposal
with the uniform source bound and the statistical allowance gives an
outward-rounded conditional absolute-error proposal below **0.0398833**.

```{embed} #simulation-lindblad-three-budget
```

The physical adapter retains the frozen $p=10^{-3}$ and $10^{-4}$ profiles and
controller failure $10^{-15}$ per decision. It reserves **23 logical patches**
for data, workspace and bridges, plus one native factory. Each trial includes
the six source-wire first resets and
**17 additional workspace/bridge initializations**. Startup, every reuse
reset, every ignored readout, trial dispatch, active/idle exposure and actual
neighboring CNOTs are charged. Each T/T-dagger carries complete capped factory
retries, neighboring transport, injection/readout/corrective-frame work, port
recycling and handoff residence. Output-quality and readiness budgets remain
separate. The controller contribution is an additive engineering-model term,
not a measured hardware-failure lower bound
{cite:p}`resource_qdk_134,resource_litinski_2019,resource_horsman_2012`.

```{embed} #simulation-lindblad-three-resources
```

Both unchanged profiles have conditional feasible points, with 54 and 119
feasible candidates and three qubit/time Pareto points each. The minimum-space
points below include all 15,756 trials and all capped factory retries.

| Physical error | Code distance | Peak physical qubits | All-trial hours | Additive physical-fault model | Factory-readiness model |
|---:|---:|---:|---:|---:|---:|
| $10^{-3}$ | 35 | 937,967 | 14,055.91 | $8.1134\times10^{-4}$ | $4.3240\times10^{-5}$ |
| $10^{-4}$ | 17 | 209,451 | 7,840.35 | $7.5830\times10^{-4}$ | $4.3240\times10^{-5}$ |

The 4,323,950,592 required magic states have a complete pipeline retry cap of
seven, charging **30,267,654,144 capped pipeline attempts**. Readiness is
bounded separately by the modeled 1% pipeline-failure probability raised to
that cap and union-bounded over all required states. Both profiles charge
692,984,803,680 controller decisions, giving contribution **0.00069298480368**
within the unchanged $1/1200$ physical allowance. The physical component sums
also fit that allowance. Infeasibility and cap failures remain explicit in
the adapter; no resource point establishes measured hardware confidence.

This run is bounded to one new prepared circuit and two fixed profiles.
Source diagnostics have 60 s, preparation/synthesis/checking 1,200 s, each
physical profile 180 s and the parent generation 1,800 s. Native rotation
synthesis permits at most **10,720 attempts**, including retries. Each profile
permits 24 top-level native factory and 24 surface-code queries, with 1,000
factory candidates per query and 24,000 per profile; candidates are not
silently truncated. Managed JSON/QASM stops at 32 MiB. The public artifact
inventory records byte sizes, hashes and reuse of prior bundles. Offline
coefficient/circuit/synthesis and final statistical-report wall time remain
outside quantum runtime; layout certification, decoder throughput and measured
hardware calibration remain unavailable. The frozen Grover/PSSPC baseline is
preserved, and this small conditional study makes no advantage or matched
co-design claim.

- {download}`Three-system notebook <../../notebooks/simulation_lindblad_three.ipynb>`
- [Source/error/workload and profile records](../../assets/generated/simulation/lindblad-three/results.json), [summary](../../assets/generated/simulation/lindblad-three/summary.csv), [manifest](../../assets/generated/simulation/lindblad-three/manifest.json), [artifact inventory](../../assets/generated/simulation/lindblad-three/artifact-inventory.csv)
- [Bounded generator](../../assets/generated/simulation/lindblad-three/generate_simulation_lindblad_three.py), [workload implementation](../../assets/generated/simulation/lindblad-three/lindblad_three_physical.py), [density checker](../../assets/generated/simulation/lindblad-three/bounded_density_instrument.py)
- [Source QASM](../../assets/generated/simulation/lindblad-three/source-trial.qasm), [routed rotation QASM](../../assets/generated/simulation/lindblad-three/routed-rotation-trial.qasm), [actual Clifford+T QASM](../../assets/generated/simulation/lindblad-three/clifford-t-trial.qasm)

## Reproduce and limits

```bash
python scripts/generate_simulation_two_spin_lindblad.py
pytest tests/test_simulation_two_spin_lindblad.py
python scripts/generate_simulation_two_spin_lindblad.py --n-system 8 --steps 8 \
  --no-sim --output-dir artifacts/lindblad-chain-8
```

Generation is capped at 16 systems, 1,024 steps and 50,000 estimated/compiled primitive operations. This original generator's combined-density checker stays at two systems and 128 steps; reference exposure is capped at 100 and validation grids at 16 counts per order. The n=2 energy-budget calculator rejects unrepresentable allocations and caps total proposed trials at $10^{12}$. Larger accuracy studies and general interacting models beyond this explicit chain family remain planned.

- {download}`Executable notebook <../../notebooks/simulation_two_spin_lindblad.ipynb>`
- [Original module](../../assets/generated/simulation/two-spin-lindblad/two_spin_lindblad.py), [damping dependency](../../assets/generated/simulation/two-spin-lindblad/open_systems.py), [standalone generator](../../assets/generated/simulation/two-spin-lindblad/generate_simulation_two_spin_lindblad.py)
- [Results and scope](../../assets/generated/simulation/two-spin-lindblad/results.json), [convergence](../../assets/generated/simulation/two-spin-lindblad/convergence.csv), [executed circuit scaling](../../assets/generated/simulation/two-spin-lindblad/scaling.csv), [logical workload ledger](../../assets/generated/simulation/two-spin-lindblad/logical-resources.json)
- Measured/reset OpenQASM 3: [X](../../assets/generated/simulation/two-spin-lindblad/two-spin-x.qasm), [Y](../../assets/generated/simulation/two-spin-lindblad/two-spin-y.qasm), [Z](../../assets/generated/simulation/two-spin-lindblad/two-spin-z.qasm)
- [MIT source license](../../assets/generated/simulation/two-spin-lindblad/LICENSE.txt) and [locked environment](../../requirements-lock.txt)

The notebook runs independently with both downloaded modules beside it. Text, circuits and figures are original. Package versions, source hashes, compilation seed and CPU measurements are recorded. Publication and any physical projection remain coordinator-reviewed steps.


(simulation-lindblad-chain-four-physical-workload)=
## Literal four-spin workload in a separate bounded domain

This section follows the earlier n2/n3 bundles. Their negative n4 labels remain
unchanged: the frozen six-wire density checker rejects eight wires, the
256-branch ensemble rejects the literal source, and the shared instrument
materializer keeps its one-million-operation ceiling. The new n4 path uses
separately named, mandatory opt-in APIs. It changes none of those defaults or
historical rejection records.

The fixed task is **n=4, r=211, local Z0 and 15,756 complete trials**, with
$t=1$, $J=0.6$, $h_x=0.35$, rates 0.8/0.5/0.5/0.5 and
$|+i\rangle/|+\rangle/|+\rangle/|+\rangle$ input. Each trial reports one
$\{-1,+1\}$ score from the first system bit. It retains all four system
outcomes, all four ignored environment readouts and every explicit reset.
There is no quantum postselection or shortened schedule. The trial population
and hardware schedule are unexecuted.

The literal source has **9,934 instructions**, **1,481 resets** and **eight
readouts**. Four system wires are prepared once each. Environment resets occur
422/422/422/211 times, including 421/421/421/210 discards before reuse.
Eight source first-wire initializations and 1,473 reuse resets remain distinct.
The calculated population has **23,334,636 source resets and 126,048 explicit
readouts**. Frozen data/workspace/bridge geometry reserves **31 patches**,
adding 23 workspace startups per trial: 362,388 workspace startups and 488,436
startup preparations across the complete population.

### Full correlated source verification

The new checker retains the entire eight-wire $256\times256$ complex128
density operator: **1 MiB** for that array. Reset sums both Kraus terms without
recording a trajectory history; ignored readouts dephase. Before the first
system readout it keeps the full $16\times16$ four-system state and all 16
joint probabilities in logical c3c2c1c0 order. A product of single-system
marginals is only a classical diagnostic: preserving Z0 does not preserve the
selected interacting joint state.

Independent tensor/Kraus source preflight agrees with the gate-derived system
state within approximately $7.73\times10^{-14}$ in trace norm. Its Z0 mean is
approximately **0.7910514391**, versus **0.7910507585** from a separately
assembled GKSL exponential; the full-state trace-norm error against GKSL is
approximately $6.66\times10^{-6}$
{cite:p}`sim_qiskit_262,sim_scipy_1181,sim_expm_2009`.
These selected-input numerical checks do not establish a uniform
representation, compiler or hardware certificate.

```{embed} #simulation-lindblad-four-source
```

```{embed} #simulation-lindblad-four-correlations
```

For this selected source state, the product of four single-system marginals
has the same Z0 mean but trace-norm discrepancy approximately **0.94795949**.
Adjacent ZZ witnesses are 0.43959675/0.42264155/0.43936007 in the source,
versus 0.36753310/0.19578362/0.17071618 in that product. These selected-state
diagnostics explain why agreement of one local mean is insufficient.


The explicit eight-wire API requires limits in its constructor and simulation
call. The literal case opts into eight wires, 20,000 source instructions and
2,000,000 expanded validation operations; no existing width or default is
raised. A 1 MiB matrix-cache budget has a 16 MiB hard ceiling. Ordered
one/two-qubit index maps have a 128 KiB ceiling and larger maps are recomputed.
The conservative 12 MiB dense-array workspace ledger excludes caches, Python
objects, circuits and process overhead. Local matrix products reduce classical
verification work without removing any quantum operation or modifying exports.

The state/trace guard is $10^{-8}$, fixed before observing the compiled result;
full source/routed equivalence uses $10^{-10}$. The checker records raw trace
and divides the captured reduced density by that trace once at output. With
no acceptance labels this is numerical normalization, not algorithm
postselection. Those guards are distinct from the unchanged error allowances
and conditional norm premises. No formal interval or hardware-confidence
certificate is supplied.

### Actual source routing and bounded synthesis

Full-unitary optimization-level0 lowering keeps **5,908 rotations**:
RX1,688, RY2,954 and RZ1,266. It has 17,343 instructions including 2,978
event barriers. Fixed-layout SABRE seed7 on the bidirectional eight-wire line
produces **24,510 routed rotation instructions** and **14,130 neighboring
CNOTs**, including 7,167 routing additions. The initialized-wire checks pass
without adding resets. The first-step drawing is exactly nine preparation
instructions and the 47-instruction first literal period.

![Actual preparation and first four-system literal period; all 211 periods precede complete readout.](../../assets/generated/simulation/lindblad-four/one-step.svg)

The separate compiler API has explicit 20,000-source, 100,000-routed and
2,000,000-emitted-Clifford+T bounds. It retains the frozen numerical
Ross–Selinger policy, whole-trial allocation and phase convention
{cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`.
A passing rotation-stage count is not a Clifford+T fit. An actual compiled
trial must retain every reset/readout and pass the full correlated checks.
A completed trial's real operation count is also checked against the old
one-million guard; this validation is not an executed legacy synthesis run.

```{embed} #simulation-lindblad-four-workload
```

The completed new-domain Clifford+T trial has **1,677,906 instructions**,
including 2,978 barriers, or **1,674,928 non-barrier primitives**: H661,278,
T642,284, S349,839, CX14,130, X5,908, reset1,481 and measure8. Eight native
synthesis calls produce eight unique targets; all 5,908 occurrences remain.
The calculated population contains **26,390,165,568 primitive occurrences**.
The unchanged old one-million validation guard rejects this completed
1,677,906-operation trial. That rejection is an executed validation check,
not a legacy synthesis or physical-estimation run.

Source/routed/actual Clifford+T means are approximately
**0.7910514391 / 0.7910514391 / 0.7910513461**. The source-to-actual system
trace-norm discrepancy is $1.94352\times10^{-6}$ and actual-to-GKSL error is
$5.69201\times10^{-6}$. The raw compiled trace is 0.9999999998272142:
drift $1.73\times10^{-10}$, within the predeclared $10^{-8}$ state guard.
Output normalization and raw trace are both retained. These are bounded
nominal checks of a complete emitted trial, not a hardware execution.


The unchanged error target is **0.01 splitting, 0.01 realization and 0.03
statistics**, with statistical failure $1/600$, physical allowance $1/1200$
and readiness allowance $1/1200$ for this case. The exact source splitting
policy selects r211, with bound $1331/133563\approx0.00996533471$; r210 exceeds
0.01. Requested whole-trial synthesis is $5\times10^{-5}$. Separate
$10^{-7}$ representation and $10^{-7}$ SDK/control/routing premises combine,
when a complete prepared trial exists, with its guarded actual synthesis
estimate to give conditional coherent premise $\eta$ and unconditional score-bias proposal $2\eta$. Neither reserve
is inferred from density agreement, and deterministic bias is not multiplied
by trials. No family allocation is relaxed or reassigned.

```{embed} #simulation-lindblad-four-budget
```

The actual guarded numerical synthesis sum is
$8.76937334\times10^{-6}$, under the unchanged $5\times10^{-5}$ request.
With both conditional reserves, the realization-bias proposal is
$1.79387467\times10^{-5}$ and the outward-rounded total absolute-error
proposal is **0.03998327346**. This ideal/conditional proposal does not
establish the full hardware-confidence target.


### Frozen physical model and explicit limits

The physical layer keeps $p=10^{-3}$ and $10^{-4}$, controller failure
$10^{-15}$ per decision, and the earlier timing, geometry, memory,
initialization, readout and factory-interface assumptions. It derives T/T-dagger
wire positions and initial-reset classification from the actual emitted trial.
The serialized model defines complete factory retries, startup,
discard/reset, ignored readouts, dispatch, transport, injection/correction,
recycling, handoff and active/idle exposure
{cite:p}`resource_qdk_134,resource_litinski_2019,resource_horsman_2012`.
An additive controller-model rejection is not a measured hardware-failure
lower bound. Missing or unsupported physical points have no invented qubit
count or runtime. If the controller-only diagnostic rejects its allocation,
other fault components, factory searches and space/time remain unevaluated.

The next paragraphs retain the proposal at the original study's completion.
The [separate completed port](#simulation-lindblad-chain-port-workload) below
implements its primary rewrite; the original rejection record remains intact.

The primary matched-task follow-on proposal adds an explicit eight-wire
validation adapter, then amortizes movement over each verified rotation block.
It moves the already initialized data wire to physical data wire0, nearest
the unchanged factory interface, with adjacent SWAPs. It applies the same
actual Clifford+T block using the existing T_q0 injection macro there,
and reverses the SWAP chain before the next source event. Every rewritten
block adds $6w$ neighboring CNOTs; none is free. The T_q0 macro retains its
nonzero factory/workspace/handoff route costs. Blocks crossing uninitialized
wires retain their original implementation. Actual sequences, phases, all
5,908 rotation occurrences, source events, error allocations and hardware
geometry remain unchanged. A secondary bounded comparison may test eight
predeclared cyclic initial layouts.

The eight individual blocks are not separate exports yet. The future bridge
must extract them from the full Clifford+T export using the recorded rotation
keys and segment lengths, then verify their phases, hashes and operators
before reuse. No new native synthesis is needed only after those checks pass.
Only a tiny full-unitary identity fixture supports the movement recipe; the
complete n4 rewrite and native validation adapter are planned and unexecuted. The
complete rewritten correlated instrument, added-CNOT/controller ledger,
all repetitions and factory retries must be audited before both frozen
physical/readiness sums are tested. A smaller controller count alone is
insufficient. No physical feasibility or co-design gain is claimed.

```{embed} #simulation-lindblad-four-resources
```

Both unchanged profiles return **infeasible_declared_controller_model**.
The actual ledger requires **123,409,809 controller decisions per trial**,
or **1,944,444,950,604** across all 15,756 trials. Its distance-independent
additive model contribution is **0.001944444950604**, exceeding the unchanged
$1/1200\approx0.0008333333333333333$ physical allowance. Code distance cannot
remove this controller term in the declared model. No native factory or
surface-code query was made, and no physical qubit count or runtime is
reported.

The readiness arithmetic uses **10,119,826,704 required magic states**,
minimal full-pipeline retry cap **7**, **70,838,786,928 capped pipeline
attempts** and bound **0.00010119826704**. That is a separate modeled
readiness union bound; native factory output quality and the remaining full
physical components are unevaluated. The 52,889,904-decision maximum requires
a reduction of at least **70,519,905 decisions per trial**, a necessary but
insufficient condition for the planned matched-task rewrite. The unchanged
budgets, physical assumptions and historical negative n4 records remain.


The new run is bounded to one prepared task: source/legacy diagnostics 180 s,
native preparation/checking 1,800 s, each physical child 180 s and a 2,400 s
hard parent through export. Native synthesis permits at most 23,632 attempts,
including retries. Each profile permits 24 top-level factory and 24 surface-code
queries, at most 1,000 candidates per query and 24,000 per profile; overflow
rejects the whole profile. Managed JSON/QASM has a 32 MiB cap and private
checkpoints a 256 MiB cap. Cap, numerical-check or controller failures remain
explicit; no stale success or partial trial is described as complete.

Offline coefficient/circuit/synthesis and reporting wall time is outside
quantum runtime. Certified layout, decoder throughput and measured hardware
calibration are unavailable. Full-family physical confidence, a feasible n4
hardware schedule and any matched co-design improvement remain unimplemented.

- {download}`Four-system notebook <../../notebooks/simulation_lindblad_four.ipynb>`
- [Source/routed/compiled checks and profile statuses](../../assets/generated/simulation/lindblad-four/results.json), [source-only preflight](../../assets/generated/simulation/lindblad-four/source-preflight.json), [summary](../../assets/generated/simulation/lindblad-four/summary.csv), [manifest](../../assets/generated/simulation/lindblad-four/manifest.json), [artifact inventory](../../assets/generated/simulation/lindblad-four/artifact-inventory.csv)
- [Separate bounded generator](../../assets/generated/simulation/lindblad-four/generate_simulation_lindblad_four.py), [workload implementation](../../assets/generated/simulation/lindblad-four/lindblad_four_physical.py), [explicit density checker](../../assets/generated/simulation/lindblad-four/lindblad_four_density.py), [explicit compiler](../../assets/generated/simulation/lindblad-four/lindblad_four_compile.py)
- [Complete source QASM](../../assets/generated/simulation/lindblad-four/source-trial.qasm), [routed rotation QASM](../../assets/generated/simulation/lindblad-four/routed-rotation-trial.qasm), [actual Clifford+T QASM when completed](../../assets/generated/simulation/lindblad-four/clifford-t-trial.qasm)

(simulation-lindblad-chain-port-workload)=
## Physical-wire-0 port: a model-feasible follow-on

Long Clifford+T words offer a different place to pay for transport. In the
frozen model, each T injection has wire-dependent transport and controller
costs. The port moves the affected state to the factory's data-wire-0
interface once for an entire word, executes that same word there, and restores
the wire positions before the next source event. Here “wire 0” is a routed
data-wire position; the encoded physical qubits are counted separately.

This is the same **n=4, r=211, local Z0 task and all 15,756 complete trials**,
with the same input, Hamiltonian, damping rates, error allocations, physical
profiles, bidirectional eight-wire line, 31 patches and factory interface as
the literal record above. The repeated trial population and hardware schedule
are unexecuted. The original implementation remains
**infeasible_declared_controller_model**, with no evaluated physical qubits
or runtime. The port supplies its own conditional model points.

### Actual words and transport construction

The [recovered catalogue](../../assets/generated/simulation/lindblad-port/word-catalog.json)
contains eight actual native words, with lengths
462/233/225/225/230/230/244/244: **2,093 distinct-word primitives**. All
**5,908 occurrences** match the saved complete stream. Their full phased
operators, explicit QASM `gphase` and binary64 phase sidecars are retained;
word recovery starts no new rotation synthesis
{cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`.

For an eligible initialized wire $w$, let $S_w$ be the descending adjacent
SWAP chain that moves its state to wire 0. Let $W_w$ be the bare token word.
The chronological construction is that chain, the unchanged bare word on
wire 0, then the reverse chain:

$$W_w\longmapsto S_w^\dagger W_0 S_w.$$

The swaps transport correlations as well as the local state. Each SWAP is
three neighboring CX gates, so the outward and return paths charge **$6w$
actual CX per word**. The T_q0 macro still charges its nonzero factory,
workspace, handoff, transport and injection costs. The recipe uses bare words
plus the authoritative baseline whole phase exactly once. The complete
candidate QASM retains one explicit overall `gphase`; the phase-bearing word
exports are operator references and their phase must not be added again.
All actual paths were eligible; the implementation
keeps the original word for unsafe paths, with that fallback covered by tiny
fixtures.

```{embed} #simulation-lindblad-port-words
```

```{embed} #simulation-lindblad-port-recipe
```

The [complete candidate](../../assets/generated/simulation/lindblad-port/candidate-trial.qasm)
has **1,801,770 instructions**, including the unchanged 2,978 barriers.
Its **1,798,792 non-barrier primitives** are H661,278/T642,284/S349,839/
X5,908/CX137,994/reset1,481/measure8. It adds **123,864 CX** and places all
**642,284 T gates on physical data wire 0**. Every initialization, reset,
readout and source event remains; the final physical c0..c7 map is
`[6,1,2,3,7,4,0,5]`. Multiplying the actual trial counts gives a calculated
population of **28,341,766,752 non-barrier primitives** and
**2,174,233,464 CX**. These are complete-workload counts, not an executed
hardware population.

### Correlated-state and error checks

Only this new candidate received one complete eight-wire density run.
The raw and once-normalized $16\times16$ system outputs agree with saved
Clifford+T data within respectively $1.44619\times10^{-14}$ and
$1.42743\times10^{-14}$ in trace norm. All 16 joint probabilities agree
within maximum residual $1.4988\times10^{-15}$; raw trace difference is
$7.7716\times10^{-16}$. The candidate-to-saved-GKSL normalized trace-norm
error is approximately $5.69201\times10^{-6}$.

```{embed} #simulation-lindblad-port-density
```

The source splitting allowance 0.01, realization allowance 0.01, statistics
0.03, statistical failure $1/600$ and physical/readiness shares $1/1200$
remain fixed. The $5\times10^{-5}$ synthesis request and separate
$10^{-7}$ representation/SDK premises are also unchanged. Selected-input
agreement checks do not establish a uniform representation, interval,
compiler or hardware certificate. Deterministic bias is not multiplied by
the trial count.

```{embed} #simulation-lindblad-port-budget
```

### Complete conditional surface-code accounting

The port requires **50,713,680 controller decisions per trial** and
**799,044,742,080** across the calculated population. Its controller-model
contribution **0.00079904474208** meets the unchanged $1/1200$ share; the
original **123,409,809 decisions per trial** still fails that share. This
controller calculation is an eligibility check. The complete physical
reports additionally charge all source operations, added CX, startup,
resets/readouts, dispatch, transport/injection, full factory retries,
storage and active/idle exposure
{cite:p}`resource_qdk_134,resource_litinski_2019,resource_horsman_2012`.

```{embed} #simulation-lindblad-port-controller
```

Both frozen profiles completed with status
**computed_conditional_port_surface_code_model**. The $p=10^{-3}$ profile
retains **45 feasible points** and $p=10^{-4}$ retains **112**; each has
three Pareto points. All 157 rows remain available. The smallest-qubit
Pareto points are:

| Physical error profile | Data distance | Conditional physical qubits | Calculated all-trial worst-case runtime (ns) | Full physical failure sum |
|---|---:|---:|---:|---:|
| $10^{-3}$ | 37 | 1,070,167 | 90,942,060,411,477,600 | 0.0008277944270374766 |
| $10^{-4}$ | 19 | 267,491 | 50,695,791,106,365,600 | 0.0008006256336973444 |

```{embed} #simulation-lindblad-port-resources
```

All six Pareto points, both full feasible sets and every failure component
are retained in [results.json](../../assets/generated/simulation/lindblad-port/results.json).
Required magic states remain **10,119,826,704**, minimal full-pipeline retry
cap **7**, and separate readiness bound **0.00010119826704**. Offline word
reconstruction, coefficient/circuit/synthesis preprocessing and final
statistical reporting wall time are excluded from quantum runtime. Interface
certification, measured decoder throughput and hardware calibration remain
unavailable.

These are model-feasible points for one matched task. The original schedule
has no evaluated qubit, runtime or spacetime denominator, so no relative
savings ratio is reported. The large serialized runtimes are retained;
the reports do not establish measured hardware advantage or full-family
physical confidence.

### Reproduction and evidence limits

The [saved companion notebook](../../notebooks/simulation_lindblad_port.ipynb)
checks retained word matrices, correlated densities/laws and accounting.
It has eight executed code cells and three saved figures, with repository
and standalone-download replays recorded. It imports no implementation
source and repeats no full candidate density run, synthesis or Native query.
The separate immutable release gate requires both complete profile statuses,
all 33 bundle files, source/recipe identities, complete feasible sets,
counters and budgets before site publication.

The **157 Native Trace digests are audit-trusted**. Raw Native Trace JSON and
the inline audit script/hash were not saved. Saved-record integrity and
accounting checks therefore do not claim independent raw Trace replay.
The two retained profile-object serialization digests are checked separately
under the frozen writer format; this does not verify a private checkpoint's
contents or reconstruct a Native Trace. No Lean formalization is integrated.

Candidate build/check/export and controller eligibility took **310.19 s**;
the two Native children took **14.75/13.24 s**. The recorded whole-worker
peak was **1,526,071,296 bytes**, below its 2 GiB cap. The candidate-boundary
**365,543,424-byte** value was measured before artifact serialization and
is not the whole-worker peak. These are classical job measurements, separate
from the modeled quantum space and time.

- [Complete bundle README](../../assets/generated/simulation/lindblad-port/README.md), [manifest](../../assets/generated/simulation/lindblad-port/manifest.json), [inventory](../../assets/generated/simulation/lindblad-port/artifact-inventory.csv), [fixed recipe](../../assets/generated/simulation/lindblad-port/recipe.json), [source proof](../../assets/generated/simulation/lindblad-port/source-proof.json)
- [Phase-aware word catalogue](../../assets/generated/simulation/lindblad-port/word-catalog.json), [example word QASM](../../assets/generated/simulation/lindblad-port/words/word-00.qasm), [complete candidate QASM](../../assets/generated/simulation/lindblad-port/candidate-trial.qasm)
- [Bounded repository runner](../../assets/generated/simulation/lindblad-port/run_simulation_lindblad_port.py), [word recovery](../../assets/generated/simulation/lindblad-port/lindblad_port_words.py), [initialized-path rewrite](../../assets/generated/simulation/lindblad-port/lindblad_port_rewrite.py), [conditional resource adapter](../../assets/generated/simulation/lindblad-port/lindblad_port_resources.py)
