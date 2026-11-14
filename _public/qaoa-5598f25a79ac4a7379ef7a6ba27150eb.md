---
title: "QAOA: objective phases and mixing"
description: "The QAOA circuit family, an independently checked four-vertex MaxCut example, and full hybrid-workload accounting."
---

QAOA encodes a discrete objective in phases, then mixes computational-basis amplitudes so those phases can change the output probabilities {cite:p}`opt_qaoa2014`. A classical routine adjusts the phase and mixing angles. The quantum circuit supplies an objective distribution; finding useful angles, estimating that distribution, and returning a good assignment remain parts of the computation.

This page develops the QAOA example introduced in the [optimization chapter](../optimization.md). Its four-vertex cycle is small enough to enumerate classically and to compare every ideal objective call with an independent formula. The example checks the implementation and this circuit family on the stated input. It supplies no general optimizer-convergence or quantum-advantage claim.

## Input, objective, and angle conventions

An unweighted MaxCut input is a classical graph with $n$ vertices and $m$ edges. An assignment $z\in\{0,1\}^n$ receives one point for each edge crossing the partition:

$$
C(z)=\sum_{(u,v)\in\mathcal E}[z_u\ne z_v],\qquad
C=\sum_{(u,v)\in\mathcal E}\frac{I-Z_uZ_v}{2}.
$$

Define the mixer $B=\sum_{j=0}^{n-1}X_j$. Prepare $|+\rangle^{\otimes n}$ and apply layers in order $1,\ldots,p$:

$$
|\psi_p\rangle=
e^{-i\beta_pB}e^{-i\gamma_pC}\cdots
e^{-i\beta_1B}e^{-i\gamma_1C}|+\rangle^{\otimes n}.
$$

We maximize $\langle C\rangle$ by **minimizing** $f=-\langle C\rangle$. All angles are radians. Qiskit's half-angle convention is $R_P(\alpha)=e^{-i\alpha P/2}$ {cite:p}`opt_qiskit262`. Consequently, the mixer uses `RX(2*beta)` on every vertex. For each edge, `CX(u,v); RZ(-gamma,v); CX(u,v)` implements $e^{+i\gamma Z_uZ_v/2}$. The identity term in $C$ contributes only an omitted global phase. The graph phases are explicit gates rather than an opaque oracle or dense-matrix operation.

The generator accepts the edge list and equally long gamma/beta sequences. H preparation, every edge interaction, every mixer operation, and optional final measurements are materialized in the circuit.

## Four vertices: mean quality and best-cut probability

Use

$$
\mathcal E=\{(0,1),(1,2),(2,3),(0,3)\}.
$$

Enumerating all 16 bit assignments gives classical maximum four, attained by the two alternating assignments. Qiskit wire zero is the least significant bit, so displayed strings read $z_3z_2z_1z_0$; the maximum-cut strings are `0101` and `1010`. The classical check scores the edge predicate directly.

The saved circuit below is produced from the same implementation used for the numerical search and exports.

![](#opt-qaoa-circuit)

For one layer on this cycle, the independent ideal expression is

$$
\langle C\rangle=2+\sin(4\beta)\sin(2\gamma).
$$

At $\gamma=\pi/4$, $\beta=\pi/8$, the mean is three, or $3/4$ of the classical maximum. This is an expected-objective ratio. The distribution at these analytic angles gives cut sizes zero, two, and four with probabilities $1/32$, $14/32$, and $17/32$, respectively. Each maximum-cut bitstring has probability $17/64$.

![](#opt-qaoa-probabilities)

The recorded optimizer fits angles numerically. Its output distribution and exported QASM use the primary fitted angles; the analytic optimum is an independent check rather than a replacement for the recorded search. The requested ideal target is $\langle C\rangle\ge3-10^{-8}$.

For a fixed fitted state with maximum-cut probability $p_\star$, $N$ independent ideal draws contain at least one maximum cut with probability $1-(1-p_\star)^N$. The [run record](../../assets/generated/optimization/runs.json) supplies the actual fitted $p_\star$ and repetition calculation. This describes readout at fixed angles, rather than the probability that the optimizer finds those angles.

## Exact fitting and separate finite-shot readout

The first workflow uses a deterministic coordinate polling routine with initial points $(0.2,0.1)$, $(0.6,0.4)$, and $(-0.3,0.25)$. The primary output is the first initialization, selected before execution. Every objective call uses an exact ideal statevector with **zero shots**, and is checked against the scalar cycle formula.

Each sweep evaluates negative and positive displacements along every coordinate from the same incumbent. It accepts the best improvement exceeding $10^{-12}$; otherwise it halves the step. The initial step is $0.4$ radians, angles are reduced to $[-\pi,\pi)$, and stopping is step below $10^{-6}$ radians or 128 sweeps. The [complete trace](../../assets/generated/optimization/evaluations.csv) retains rejected candidates as well as accepted ones. Local optimizer and classical-control times are software measurements.

Only after fitting, the primary state receives separate ideal finite-shot readout with seeds `7`, `19`, and `41`, using 2,048 shots per seed. All edges are diagonal in the all-Z basis: each outcome scores the entire cut, so shared-edge covariance is retained in the variance of the joint score. Counts are drawn classically from statevector probabilities. These readout experiments do not feed sampled values back into the first optimizer.

The [finite-shot increment](finite-shot.md) performs actual sampled parameter updates and records all their shots, while [portability](portability.md) tests a triangle input with fixed starts across seeds. Neither is a theorem about convergence on larger graphs.

## Empirical performance across sizes and starts

The two shared finite-shot studies use fixed recipes and retain every prescribed outcome. Their complete records appear in the [size study](finite-shot.md#quality-and-workload-across-sizes) and [initialization diagnostics](finite-shot.md#initialization-and-sampling-diagnostics). Counts are generated classically from ideal circuit probabilities. These are bounded implementation studies, not hardware measurements, convergence theorems, or quantum-advantage evidence.

The size study uses unweighted paths and cycles with $n=3,4,5,6,8$ vertices at depths $p=1,2$. The initialization diagnostics use cycles C4 and C6 at those two depths. Both studies request normalized expected cut $\langle C\rangle/C_\star\ge0.80$, where classical enumeration supplies $C_\star$. This is a different quality goal from the first one-layer C4 demonstration's mean $3-10^{-8}$; its maximum ratio is $3/4$, below the studies' preserved target.

The locked recipe has been executed for all **72 runs**. Independent post-training audits meet the target in **19 of 60 QAOA runs** and **4 of 12 VQE runs**. Simultaneous holdout intervals support the target in **18 QAOA runs** and **no VQE runs**; every run supports the separately requested $\pm0.05$ normalized readout precision. All predetermined runs and misses are retained, with no retuning of initializations, schedule, targets, or readout allocation.

The frozen numerical recipe has executed all **36 sampled runs** and **24 separately labeled deterministic CPU controls**. Independent post-training audits meet the target in **4 of 24 QAOA runs** and **2 of 12 VQE runs**. Simultaneous holdout intervals support the target in **three QAOA runs** and **no VQE runs**; all 36 runs support the separately requested $\pm0.05$ normalized readout precision. Every prescribed run and miss remains in the record.


The achieved quality depends on the tested input and recipe. Paths at $n=3$ meet the target in all six depth/seed runs; among the larger paths, only the $n=4,p=2$, seed `41` run meets it. The odd cycles at $n=3,5$ meet the target in all twelve runs, while every tested even-cycle run misses it. Odd cycles have $C_\star=n-1$, so their ratio denominator differs from that of an even cycle; these results do not imply a monotone size trend.

All target misses remain visible. They can reflect the circuit family, initialization, finite search budget, and sampled decisions. One limitation is already known independently: the depth-one C4 formula in the [main chapter](../optimization.md#qaoa-phases-mixing-and-a-small-maxcut-circuit) has maximum mean three, hence ratio $3/4$, below this study's $0.80$ target. A perfect angle optimizer cannot make that circuit family meet the target. The study does not identify the cause of every other miss or estimate a general success probability.

All twelve depth-one QAOA runs miss the target, as required by the [C4/C6 circuit ceiling](finite-shot.md#sampled-diagnostics-equal-training-preparations-and-sampled-decisions). At C4 depth two, all three runs from the alternative start reach ratio $0.988199$, with simultaneous holdout support, while none from the original start reaches $0.80$. Both deterministic controls show the same start contrast. This is behavior of these two prescribed starts on one case, not an estimated population success probability.

The C6 depth-two results differ. The original start has one exact target hit, seed `19`, at ratio $0.805210$; its holdout interval crosses $0.80$ and is inconclusive. All three alternative-start sampled runs miss the exact target. For that alternative start, the exact objective control reaches $0.830926$ after eight accepted moves, while the expected margin control reaches $0.794687$ after two. The successful exact control supplies a state in this circuit family above the target, and the margin changes this fixed search path. Neither fact predicts what another sampled trajectory will select.

The size study keeps the original simultaneous family allocation $0.05/84$, and the diagnostic study keeps $0.05/48$. Showing one method's outcomes here does not reduce either family denominator, recalculate its intervals, or reinterpret holdout support as an optimizer success probability. Deterministic CPU controls consume zero quantum preparations; their virtual shot allocations are diagnostic inputs rather than measured workloads.

## End-to-end resource analysis

Before optional measurement, the direct $p$-layer source contains $n$ H gates, $np$ RX gates, $mp$ RZ gates, and $2mp$ CX gates. Circuit construction therefore has $O(n+p(n+m))$ gate work and $n$ logical qubits. Compilation, routing, synthesis, and scheduling can change the eventual physical cost; these source counts are not T counts or a physical runtime.

For a sampled hybrid run with objective calls $j$ and shot allocations $N_j$, state preparation and the whole ansatz repeat $\sum_jN_j$ times. Cut scoring requires $O(m)$ classical work per recorded assignment. Classical search effort and its number of objective calls are additional costs. A fixed layer depth does not establish a bound on how many evaluations achieve a requested cut quality.

The first demonstrated QAOA workload contains all three exact searches, 495 exact simulated ansatz evaluations, and 6,144 modeled final-readout preparations. The [logical ledger](../../assets/generated/optimization/logical-resources.json) separates simulated gate work from shot-weighted gates and measurements. The conditional recorded-trajectory projection and the actual Clifford+T comparison appear below; the exported fitted RX/RZ angles remain unsynthesized in this original logical circuit.

The generator caps construction at 4,096 qubits and 100,000 unitary gates before final measurements. Ideal statevector expectation defaults to at most 12 qubits, with a hard cap of 16; classical enumeration has its own hard cap of 20. Statevector memory scales as $2^n$, and the enumerated cut table also has $2^n$ entries. These limits separate circuit export from bounded classical validation; they do not constitute a large-input simulation benchmark.

(qaoa-historical-resources)=
### Recorded finite-shot resources: historical conditional model

This historical projection concerns the original one-layer C4 QAOA runs and their recorded 32-sweep sampled schedule. It does not estimate the larger inputs or the later depth-two compiled transcript. The [complete recorded-trajectory analysis](finite-shot.md#recorded-trajectory-resources) maps every initial call, incumbent remeasurement, accepted or rejected proposal, and holdout to an explicit measured circuit. The QAOA task has 473,856 preparations and 486 all-Z batches. These are modeled repetitions of stored circuits, not predictions of a fresh hardware-driven adaptive path.

| Task | Physical error | Declared serial choice | All modeled preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-task physical-fault score |
|---|---:|---|---:|---:|---:|---:|
| QAOA | 0.001 | fewest qubits per cost signature | 473,856 | 1,131,935 | 1088.640000 | 0.000154 |
| QAOA | 0.001 | shortest runtime per cost signature | 473,856 | 1,328,815 | 914.457600 | 0.000348 |
| QAOA | 0.0001 | fewest qubits per cost signature | 473,856 | 131,595 | 1088.640000 | 3.76e-05 |
| QAOA | 0.0001 | shortest runtime per cost signature | 473,856 | 270,375 | 479.001600 | 5.55e-05 |

These are the unchanged QAOA rows from the original replay, not a new estimate. Each task retains its separate whole-task physical-fault allowance $10^{-3}$ and per-preparation synthesis-model scalar $10^{-5}$. The scalar is a coherent approximation-model quantity and is not summed over shots as a stochastic failure probability. Profiles retain physical error $10^{-3}$ or $10^{-4}$, 100 ns gates and 500 ns measurements.

Times sum every preparation's model runtime; peaks take the maximum of independently selected class footprints. A common physical configuration is not verified. Physical initialization/reset, transport, routing, reconfiguration, communication and adaptive feedback remain unmeasured or excluded. The PSSPC rotation fit supplies expected resource cost, not concrete synthesized sequences, worst-case bounds, runtime-tail guarantees, or an output-accuracy certificate. CPU controller, reference and simulator times remain separate.

The larger shared size-study replay also retains all 72 runs and 5,859,840 preparations, including every quality miss. Its 84 cost classes and 85 searches remain a separate frozen record with independently selected class configurations. They are not a time-to-quality estimate or a physical proxy for the later initialization diagnostics, which have no new physical projection.

(complete-clifford-t-workload)=
### Complete Clifford+T workload for one recorded QAOA transcript

The [finite-shot diagnostics](finite-shot.md#initialization-and-sampling-diagnostics) recorded a successful small QAOA trajectory, together with every query that preceded its final state. This replay asks a narrower implementation question: what happens when each of those fixed, bound circuits is routed and its arbitrary rotations are replaced by actual Clifford+T sequences? QAOA's objective-dependent phase and mixer are explicit circuits {cite:p}`opt_qaoa2014`; neither the oracle construction nor the rejected optimizer work is an opaque free operation.

:::{note} Actual compilation and conditional model completed

The selected sampled trajectory is an executed ideal numerical record. All **54 distinct source circuits** have been routed, synthesized to actual Clifford+T sequences, and passed their tiny numerical checks. A complete **conditional surface-code workload** now accounts for all 110 training/holdout groups and 82,032 preparations. An initial physical-result representation-limit failure remains preserved; one separately declared computation-only recovery completed the lossless report without repeating synthesis or optimization.

Replay fixes the recorded classical decisions and angles. It does not establish that an optimizer supplied with different sampled counts, synthesis error, or hardware noise would follow that path. The reported shot workload and physical schedule are modeled; no QPU or calibrated hardware-noise experiment is performed. All prior records and their negative outcomes remain unchanged.
:::

(qaoa-compiled-selected-task-and-immutable-trajectory)=
#### Selected task and immutable trajectory

The input is the unweighted four-vertex cycle with edges $(0,1),(1,2),(2,3),(0,3)$. It has classical optimum $C_\star=4$. The requested output remains a normalized expected cut, with quality target $\langle C\rangle/C_\star\ge0.80$, normalized readout precision $\pm0.05$, and a declared conditional failure allowance of $0.05$.

The selected record is `qaoa-cycle-4-p2-alternate-seed-7`. It uses two QAOA layers and the predeclared interleaved start $(\gamma_0,\beta_0,\gamma_1,\beta_1)=(0.6,0.4,0.6,0.4)$. Selection takes the first genuinely holdout-supported four-wire run in the frozen case/start/seed order. Its eligibility depends on the observed holdout status, so the new statistical allocation retains the full 48-group union bound rather than treating it as an outcome-independent single-run selection. It is not the best angle vector or a new optimizer initialization.

The recorded final angles are approximately $(0.8,0.6,1.2,0.4)$; the implementation must bind the exact saved floating-point values, rather than substitute these rounded values. The independent ideal expected-cut ratio is $0.988198950$. Its saved holdout estimate is $0.988403320$, with simultaneous interval $[0.973214003,1.003592637]$ under the original 48-group allocation. The lower endpoint supports the $0.80$ quality target. The untruncated upper endpoint remains visible even though a cut ratio is at most one.

The [parent raw study](../../assets/generated/optimization/diagnostics/raw-study.json), [recipe](../../assets/generated/optimization/diagnostics/recipe.json), [source producer](../../assets/generated/optimization/diagnostics/optimization.py), and [source circuit manifest](../../assets/generated/optimization/diagnostics/circuit-manifest.json) define the original record. New exports must bind every source angle, basis, count batch, circuit and decision to those immutable identities. The original sampler used NumPy PCG64 ideal probabilities with seed `7`, case/start stream `33`, and separate training and holdout families.

The other two alternate-start C4 depth-two seeds also have saved target support. All three original-start C4 depth-two runs miss the ideal target. These comparisons remain in the parent record; compiling one supported path does not turn that selected path into a population optimizer-success estimate. The sixth study still has **zero holdout-supported VQE targets**. Its two ideal VQE hits remain inconclusive, and no VQE interval is relabeled by this QAOA selection.

(qaoa-compiled-every-training-and-readout-group-is-charged)=
#### Every training and readout group is charged

The recorded search has an initial 128-shot call, then twelve sweeps: four each at coordinate steps $0.4$, $0.2$, and $0.1$ radians, with 260, 520, and 1,040 shots per call. Each sweep evaluates a fresh incumbent and both directions along all four coordinates. It exhausts the fixed schedule without a convergence stopping rule.

| Recorded quantity | Complete selected trajectory |
|---|---:|
| Training objective calls / all-Z groups | 109 |
| Accepted / rejected coordinate proposals | 4 / 92 |
| Final independent all-Z holdout | 1 group, 16,384 shots |
| Training preparations | 65,648 |
| Training plus holdout preparations | 82,032 |
| Individual source-qubit readouts | 328,128 |

The source ledger includes every initial, incumbent, accepted proposal, rejected proposal, and final holdout. For $n=m=4$ and $p=2$, one measured source circuit contains four H, eight RX, eight RZ, sixteen CX, and four final measurements. Its complete source-unitary workload is therefore 2,953,152 shot-weighted gates. Four explicit initial resets per preparation add 328,128 source initialization events in the complete-instrument specification; these resets were previously an explicitly unmeasured physical preparation assumption.

All-Z readout has no additional basis rotation. The initial H gates, objective edge phases, mixer rotations and every measurement are already source operations. Actual logical routing can add neighboring operations to realize the source's nonlocal edge interaction; those extra operations must be counted from the routed circuit. Actual synthesized T and T-dagger counts come from the generated sequences, never an expected T-per-rotation fit.

The simulator's original optimizer, objective-evaluation and classical-control timings remain separate from hardware time. The [classical-work record](../../assets/generated/optimization/compiled-replay/classical-work.json) retains the original optimizer wall time of 0.070226 s, objective-evaluation time of 0.069346 s and control time of 0.000879 s, together with actual compilation/check timing. Physical scoring, optimization arithmetic, parameter transmission, inter-call communication and hardware optimizer latency remain unmeasured. A quantum-operation total must not silently absorb these exclusions.

(qaoa-compiled-actual-synthesis-and-tiny-independent-checks)=
#### Actual synthesis and tiny independent checks

The owned wrapper prepended canonical reset on every used wire, retained all final measurement maps, and passed each exact bound circuit to the frozen complete-instrument backend with `preserve_full_unitary=True`. This disables clean-input compiler assumptions for the coherent segment. Its lowering retains reset/readout boundaries. Logical routing uses a bidirectional line, fixed initial layout, SABRE seed 7, and optimization level zero.

Remaining RX/RZ rotations were replaced by actual ancilla-free Ross--Selinger approximants generated through Qiskit 2.6.2 `gridsynth_unitary` {cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`. The native synthesis entry point exposes no seed parameter. Exact generated sequences, global phases and hashes define the saved result; a future seeded rerun is not promised.

Independent four-qubit checks must compare the $16\times16$ coherent matrices, prepared state and joint cut distribution of the source, routed and synthesized circuits. Operator and state comparisons must restore the recorded global phase and account for the final routing permutation; joint-output comparisons must retain the classical measurement destinations. The routed circuit must agree with its source to numerical tolerance. Each generated Clifford+T circuit must satisfy its allocated spectral approximation check, preserve source initialization/measurement events, and contain only supported Clifford+T primitives plus reset, readout and zero-cost barriers. Every CX in the routed circuit must be between neighboring logical wires.

The backend checks each rotation's floating-point Frobenius discrepancy and adds a $10^{-12}$ numerical guard before summing occurrence-weighted errors. Such a guarded numerical estimate is evidence of the actual approximation, **not a formal interval certificate**. A requested tolerance alone does not establish that an approximant meets it.

Measured-trial OpenQASM exports omit overall global phase. Separate metadata must retain that phase for numerical checks. The exports specify this initialized, measured replay and must not be presented as phase-preserving controlled-oracle implementations.

The completed compilation checkpoints contain 306,508 materialized Clifford+T instrument operations across 54 distinct circuits, generated with 193 native synthesis calls. Multiplying each circuit's actual counts by all its recorded shots gives **182,646,240 T-state consumptions**; there are no T-dagger operations in these saved approximants. The total guarded fixed-script variation estimate is $0.0009485412$, below the $0.005$ allocation. Maximum routed/source spectral discrepancy is approximately $8.27\times10^{-16}$, and maximum actual Clifford+T/source spectral discrepancy is approximately $7.89\times10^{-9}$. These are numerical checks of actual saved sequences, not a mean-fit T estimate or formal certificate.

The final compiled circuit's exact numerical ratio is $0.9881989495$, with observed normalized bias approximately $9.99\times10^{-12}$ against the immutable source. The compiled shot ensemble has not been executed, so this is a numerical expectation rather than a newly sampled confidence interval. The [synthesis timing](../../assets/generated/optimization/compiled-replay/timing/synthesis.json) records 122.475877 s of parent-invocation wall time, including its bounded workers and checkpoint writes; this is not quantum runtime.

![Selected named source and Clifford+T operation types, weighted by every recorded preparation.](../../assets/generated/optimization/compiled-replay/workload.svg)

The figure's Clifford+T panel displays selected named operation types. The [complete weighted ledger](../../assets/generated/optimization/compiled-replay/resources.json) also includes 401,440 X operations in addition to the plotted types; all contribute to the workload accounting.

(qaoa-compiled-realization-sampling-and-physical-allowances)=
#### Realization, sampling and physical allowances

The frozen statistical, fixed-script realization-transfer, physical-fault and factory-availability allocations are

$$
\delta_{\rm sampling}=0.04,\qquad
\varepsilon_{\rm transcript}=0.005,\qquad
\delta_{\rm physical}=0.0025,\qquad
\delta_{\rm factory}=0.0025.
$$

Their sum is the requested $0.05$ conditional allowance for a prospective fixed-script execution. The realization term is a bound on distribution variation, not a stochastic fault event; it can bound the increase in probability of a fixed bad-report event when transferring an ideal fixed-script ensemble to its compiled counterpart. It is accounted once in that transfer, while physical faults and factory availability remain separate conditional event models. Historical confidence inference and a future execution are different probability experiments, as distinguished below.

(qaoa-compiled-archived-selection-aware-inference)=
##### Archived selection-aware inference

The statistical allowance $0.04$ covers all 48 original final groups, with $\delta_{\rm group}=0.04/48$. This selection-aware family scope protects choosing a supported trajectory after reading the holdouts. It must not be replaced by the selected group's individual failure allocation. The physical/factory workload covers this one complete selected trajectory. Its normalized Hoeffding radius at 16,384 shots is

$$
r_{\rm sampling}=\sqrt{\frac{\log(2/\delta_{\rm group})}{2\cdot16{,}384}}.
$$

The cut score lies in $[0,4]$, and normalization by $C_\star=4$ cancels that range width. This more conservative allocation gives $r_{\rm sampling}=0.0154118509$. Applied to the archived counts, it is a separate selection-aware inference about all 48 ideal final-state means; it does not overwrite the parent study's saved intervals or target labels. Hoeffding coverage concerns that family of fixed final states and fresh ideal independent samples, conditional on their training histories {cite:p}`opt_hoeffding1963`. It does not certify adaptive training queries.

If the actual coherent-segment discrepancy has a valid spectral upper bound, it deterministically shifts the inferred final mean by at most that bound. The pre-execution illustration uses the frozen uniform allowance $\eta_{\max}=0.005/82{,}032$, giving radius $r_{\rm sampling}+\eta_{\max}\approx0.0154119118<0.05$. Applied to the archived ideal-sampling estimate, it gives an illustrative lower endpoint of approximately $0.9729914085>0.80$ for an inferred compiled mean, conditional on a valid bound. These values use the allocation ceiling, rather than the smaller actual guarded bound reported below. This is not a confidence interval from newly sampled compiled-circuit counts: no such shot ensemble has been executed. This archived inference does not receive the fixed-script event-transfer term, because its historical adaptive transcript is not a product of independent fixed-angle draws.

(qaoa-compiled-prospective-fixed-script-realization)=
##### Prospective fixed-script realization

A coherent realization allocation is assigned to the full **fixed** transcript:

$$
\varepsilon_{\rm transcript}=0.005,\qquad
\eta_g\le\frac{0.005}{82{,}032},\qquad
\sum_gN_g\eta_g\le0.005.
$$

Here $N_g$ retains every group's recorded multiplicity. If $\eta_g$ truly upper-bounds phase-aligned spectral operator error of the coherent unitary segment after identical initial resets and before final readout, its outcome-distribution variation and normalized cut-mean bias are at most $\eta_g$. Operator checks must restore the recorded global phase and account for routing permutations. The full measured circuit is an instrument, not a unitary operator. The sum above compares hypothetical ideal and compiled ensembles for an externally fixed sequence of preparations. It does not describe the historical adaptive transcript distribution conditioned on producing its saved angles or a calibrated probability that another optimizer makes the same choices.

For a hypothetical fresh ideal fixed-script ensemble, a fixed final-report error event has probability at most the declared statistical ceiling $0.04$; the one final group uses the smaller $0.04/48$ allocation. A valid total-variation bound then increases that event probability by at most $0.005$ in the compiled fixed-script ensemble. Adding the separately modeled physical and factory events gives the conservative prospective sum $0.04+0.005+0.0025+0.0025=0.05$. This is a contract for a new externally fixed script, not a transfer of the historical selection event or a probability of repeating the original adaptive optimizer path.

The actual Clifford+T mean ratio is checked numerically from its small circuit. For prospective fresh ideal Clifford+T readouts, direct Hoeffding concentration is centered on that mean: $|\widehat R_{\rm CT}-R_{\rm CT}|\le r_{\rm sampling}$. On this event, a reported interval of radius $r_{\rm sampling}+\eta_{\rm final}$ supports the quality target if $R_{\rm CT}-2r_{\rm sampling}-\eta_{\rm final}>0.80$. The [conditional-readout record](../../assets/generated/optimization/compiled-replay/conditional-readout.json) gives this lower value as $0.9573752364$ and the actual guarded precision radius as $0.0154118622<0.05$. This sufficient test uses direct compiled-ensemble concentration, separately from the full-script event-transfer allocation. It is distinct from displaying the old ideal counts as though they were new compiled measurements.

The final-state mean-bias shift $\eta_{\rm final}$ and the full-transcript event transfer have distinct roles; the $0.005$ transfer is not added again to the mean-error radius. These proposals require actual synthesis checks, valid numerical operator-error upper bounds, physical scenario assumptions and a successful complete-workload allocation. They do not claim that the archived inference and a hypothetical future report jointly fail with probability at most $0.05$; such a joint statement would require a further statistical allocation. A floating-point numerical guard does not turn either experiment into a theorem or hardware guarantee.

(qaoa-compiled-conditional-surface-code-workload)=
#### Conditional surface-code workload

The frozen instrument backend uses direct QDK 1.34.0 Trace accounting, including reset and measured-trial events {cite:p}`resource_qdk_134`. It does not substitute the earlier OpenQASM-import/PSSPC mean-fit representation for the generated Clifford+T sequences. This is a separate serialized instrument scenario, preserving the frozen [Grover baseline](../grover.md#end-to-end-resource-analysis), not a matched-task co-design improvement.

The fixed profile keeps physical gate and initialization times at 100 ns, measurement time at 500 ns, code-cycle time at 1,000 ns, and controller reaction at 10,000 ns, with physical error rate $10^{-4}$. It uses the backend's declared memory, initialization, readout and controller-error scenarios. Those parameters are conditional engineering assumptions, not calibration measurements {cite:p}`resource_litinski_2019,resource_horsman_2012`.

For four source wires the model reserves fifteen data/workspace/bridge patches. At odd code distance $d$, their area is $15(2d^2-1)$ before the native factory footprint. Source preparations/readouts remain distinct from the additional scratch preparations, parity readouts, transport, injection and controller decisions in the physical macro ledger.

Initial encoded-patch startup uses the backend's `initial_reset` macro: a nominal initialization interval, one controller decision and zero explicit readouts. Its residual-initialization error and subsequent code exposure are assumed separately. This is not an implemented or certified fault-tolerant encoder, and its internal startup syndrome/micro-operation counts are unavailable.

Each generated T or T-dagger consumes native factory supply. The model charges complete capped factory retries, neighboring logical transport, injection CX/readout/correction, port reinitialization and extra handoff residence. Native factory output error and its separate 1% pipeline-readiness failure assumption are different quantities; the factory's included quantum operations must not be charged again as a duplicate memory error. Retry caps cover every required state in all repetitions.

The native factory ISA supplies conditional time, space and output-quality costs. An internal factory gate/readout/preparation breakdown and a compiled factory circuit are not exposed by this backend. The surrounding supply, retry and handoff macros are charged, but that unsupported internal layer must not be described as an exported gate-level factory implementation.

Physical failure allocations must add over every group and preparation. Factory availability allocations must add over every required T state. A common code-distance/factory configuration must be feasible for the whole workload; choosing independent minima for unrelated classes would not establish one common machine. Matched native cost inputs alone do not certify identical factory geometry.

The [physical summary](../../assets/generated/optimization/compiled-replay/physical-summary.json) and [CSV](../../assets/generated/optimization/compiled-replay/physical-summary.csv) retain two declared selections from the bounded common-configuration domain. Both use distance 15, fifteen data/workspace patches and a whole-workload factory retry cap of six. The factory footprint is 152,660 physical qubits for the minimum-qubit selection and 236,623 for the shortest model-time selection; data/workspace adds 6,735 qubits in each case. The complete T-state demand gives at most 1,095,877,440 pipeline attempts under that retry cap and factory-availability bound $0.00018264624$.

| Common selection within the frozen domain | Code distance | Peak physical qubits | Capped serial quantum-model time (s) | Whole physical-fault model | Factory-availability bound |
|---|---:|---:|---:|---:|---:|
| Fewest physical qubits | 15 | 159,395 | 863,080.953 | 0.000153065358 | 0.000182646240 |
| Shortest model time | 15 | 243,358 | 797,328.307 | 0.000143202461 | 0.000182646240 |

These times multiply the complete capped pipeline-attempt and algorithm-preparation counts by serialized operation costs. Native factory supply time is a nominal/expected calibration returned by `expect_time`, not a sampled tail bound. The rows are therefore capped serial **quantum-model times**, not hardware measurements or worst wall-duration guarantees. Their physical-fault terms rely on the stated native, memory, initialization, readout and controller models.

Both selections charge 9,379,512,720 preparation events, 4,780,464,240 readouts and 22,072,802,800 controller decisions in the serialized data/workspace/injection/control macro ledger. These are model-event counts, distinct from 82,032 source shots and 328,128 output-qubit readouts; they exclude the unavailable internal factory and startup-encoder breakdowns described above. They must not be presented as a complete microscopic hardware-operation count.

The physical parity, injection, protected-basis and factory/data interfaces are modeled macros, **not a certified physical lattice layout**. The logical routed circuit is an actual export; the physical macro geometry remains conditional. Arbitrary hardware adaptive control, decoder-throughput validation, measured interfaces and formal synthesis certificates remain unsupported and must be reported plainly.

(qaoa-compiled-preserved-representation-limit-failure)=
##### Preserved representation-limit failure

The first physical phase used 49 native accounting classes, 1,176 factory-frontier queries and 6,174 direct Trace estimates. It completed those native queries within their limits, but the result's expanded JSON representation exceeded the frozen 64 MiB decoded-record cap. The [retained failure](../../assets/generated/optimization/compiled-replay/checkpoints/failures/physical-workload.json) records the compact counters and the cap error; complete query records from that attempt were unavailable because of the same representation limit. This was not a completed physical artifact, and no partial point was published as the result.

The [computation-only amendment](../../assets/generated/optimization/compiled-replay/completion-amendment.json), frozen before the recovery, permits one bounded physical repeat with cumulative top-level and frontier caps of 128 and 3,072, while retaining the separate 52,224 caps for frontier yields, pipeline candidates and Trace estimates. Its lossless DAG representation completed the report. Cumulative work was 98 top-level calls, 2,352 frontier queries, 24,696 frontier yields, 816 pipeline candidates and 12,348 Trace estimates. The [completion timing](../../assets/generated/optimization/compiled-replay/completion-timing.json) records 46.301233 s against the 120 s cumulative physical allowance, including the failed phase's 11.716275 s.

The source circuits, compiled sequences, case, statistical/realization allocations and hardware profile are unchanged; synthesis was not repeated. The repeated work was CPU resource estimation, not another execution of the 82,032-preparation quantum script. The initial failure and all 281 first-stage files and their bytes remain preserved. The completed DAG was checked exactly against the second live producer result; no equality is claimed to the unavailable complete first result. The physical report's current status is complete under these conditional models, while the initial failed record remains incomplete history.

(qaoa-compiled-bounded-execution-archives-and-artifact-size)=
#### Bounded execution, archives and artifact size

The original wrapper and figures are MIT-licensed and use the pinned Python 3.12 environment. Qiskit core 2.6.2 is Apache-2.0; its Cargo lock pins the MIT-licensed `rsgridsynth` crate 0.2.2, although the installed wheel exposes no direct native-crate version certificate. QDK 1.34.0 is MIT-licensed; its frozen native-source audit pointer identifies inspected semantics and is not claimed to be the package release commit. The [main chapter's package credits](../optimization.md#reproduce-packages-and-planned-extensions) also record NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2. Exact source hashes and the lock must accompany the generated result.

Before compilation, the recipe froze the selected transcript, source and backend hashes, package/lock identities, accuracy/failure allocations, materialization and validation bounds, synthesis attempt/wall limits, native query/candidate caps, and output-byte cap. Native searches or representations that hit their cap remain incomplete; a partial frontier is not a completed resource result.

The frozen materialization limits are 110 measured group records and 54 distinct bound source circuits, four source wires and sixteen joint-output outcomes. Per-circuit operation caps are 128 source, 1,024 routed and 10,000 Clifford+T operations, with at most 540,000 Clifford+T operations materialized across distinct circuits. The compilation phase has a 300 s wall allowance and each class a 20 s watchdog; the physical phase has a 120 s allowance. Initial native limits were 64 top-level estimator calls, 1,536 frontier queries, and separate caps of 52,224 for frontier yields, pipeline candidates, and Trace estimates; those three counters are not one aggregated allowance. The representation-only completion amendment above changes only the cumulative top-level/frontier caps. Per-class and complete-record decoded representation caps remain 4 MiB and 64 MiB. Numerical arrays are losslessly serialized with real/imaginary components, retaining the instrument-ensemble checks. These are execution caps, not predicted resource costs or validation evidence.

The original archive and its negative controls are referenced by immutable identities rather than copied into a second large bundle. The [manifest](../../assets/generated/optimization/compiled-replay/manifest.json), [selected transcript](../../assets/generated/optimization/compiled-replay/selected-transcript.json), [group CSV](../../assets/generated/optimization/compiled-replay/groups.csv) and [class CSV](../../assets/generated/optimization/compiled-replay/classes.csv) map the distinct source/routed/Clifford+T circuits to every recorded group. The [negative-control CSV](../../assets/generated/optimization/compiled-replay/negative-controls.csv) preserves the original comparison labels. Identical QASM byte deduplication saves storage; it removes no preparations or readouts.

The [artifact-size record](../../assets/generated/optimization/compiled-replay/artifact-size.json) reports **9,614,924 new generated-bundle bytes**, including that size record and all 301 files, below the frozen **16 MiB** cap. The inherited large archive and private native cache are not copied into this count. This is the added generated bundle's byte count, not a prediction of the whole rendered site's size.

No GPU job is needed for four-qubit circuit checks. Local numerical/synthesis/native-query work must remain bounded, with complete provenance and retained failures.

(qaoa-compiled-reproduce-and-inspect-the-lossless-record)=
#### Reproduce and inspect the lossless record

The public [completion reader](../../assets/generated/optimization/compiled-replay/complete_optimization_compiled.py) supplies `load_completed_replay(output)`. It combines the [completed-replay metadata](../../assets/generated/optimization/compiled-replay/completed-replay.json.gz), all immutable synthesis checkpoints and the [physical JSON-DAG](../../assets/generated/optimization/compiled-replay/physical-dag.json.gz) using the matching [codec](../../assets/generated/optimization/compiled-replay/optimization_compiled_codec.py). The reader checks content hashes, preserved source identities, all classes and cumulative timing/query limits. Public reading needs no private pickle or native recomputation; decoded shared subtrees are read-only record data.

| Reproducibility object | SHA-256 |
|---|---|
| Completed replay transport | `91fe2c423d66e682bcfba4b78142fa4192a60bf02609d4e6b526b946259d3c53` |
| Physical JSON-DAG transport | `731791448fcfd3e7fc2df18efa0d7d0db12c49254a709bb58d0286a93d1cca34` |
| Physical record logical-content identity | `c84c1c85514d7620482a2fb25f6e2d2c772a6a6bb6f9ceff207e2e31adc2904b` |
| Completion reader source | `0782405325b69630b0ef34ad34b4a97d2ea852c42a1ec629716fac50960d7d7c` |
| JSON-DAG codec source | `d722622fd850b5b10d29898eaba38538dd257c870e09df1e28856fa1829cf4f3` |

The logical-content identity is the codec's typed content digest, not a checksum of an expanded JSON file. The exact native factory ISA and T-instruction hashes for each common selection are retained in the physical summary; the full physical record includes the frozen installed-native source fingerprints. The [original freeze](../../assets/generated/optimization/compiled-replay/freeze.json) and [completion freeze](../../assets/generated/optimization/compiled-replay/completion-freeze.json) bind the original and recovery producers separately. A representation change does not rewrite the executed synthesis bytes.

Save the complete generated directory, including its checkpoints, circuits, freezes, recipes, codec and source snapshots, for standalone reading. Place the [verification notebook](../../notebooks/optimization_compiled.ipynb) inside that directory or execute it from the source checkout with the pinned environment. Saved-record verification must not call the optimizer or synthesize replacement sequences.

```bash
python -m pytest tests/test_optimization_compiled.py
python scripts/complete_optimization_compiled.py --verify-only
```

For the matching standalone bundle, run `python complete_optimization_compiled.py --output-dir . --verify-only`. Checkout and isolated standalone notebook execution, figure QA, integrated tests and site checks remain prerequisites for publication.

`--verify-only` validates the saved records and deterministically re-exports the
derived summaries and figure. Run it on a copy when preserving the original
bundle bytes. The release verifier performs that work in a temporary copy,
compares every file with the original, and starts no optimizer, synthesis or
native physical-estimator pass.

**Classical C4 reference coverage only.** The existing Lean reference does not prove QAOA dynamics, this sampled optimizer, synthesis approximants, physical macro geometry, confidence intervals, or fault tolerance.

(qaoa-compiled-matched-word-transport-comparison)=
#### Matched word-transport comparison

The [word data-shuttle variant](qaoa.md#matched-word-transport) changes one implementation
choice on this exact frozen transcript: it moves selected uninterrupted
Clifford+T words to the magic port and returns the data afterward. It preserves
all 54 source classes, 49 cost parents, 110 training/holdout groups, 82,032
preparations, T-state demands, source phases, requested output and budgets.

Its checked ideal transformation and separately captured fixed-machine Native
results reduce the **conditional serialized model time and spacetime by
24.439396%** at the same 243,358-qubit footprint. The comparison uses the saved
minimum-runtime configuration, without another code/factory search. It is a
result for this one C4/depth-two transcript; other QAOA/VQE outcomes and the
original physical representation-limit failure remain separate unchanged
evidence. Public saved verification rederives the scientific records without
Native estimation, while private operational receipts retain their audited
hash and final-seal trust boundary.

(matched-word-transport)=
### Matched word-transport comparison

(qaoa-word-motivation-and-intuition)=
#### Motivation and intuition

Moving a freshly supplied magic state to a distant data wire can cost more
than the T gate it enables. When several T gates lie in one uninterrupted
single-wire word, moving the data state to the magic port once and returning it
after the word can reduce transport. Spectator data can be entangled: the
transformation must preserve the whole operator, rather than assume clean
intermediate wires.

:::{note} Executed scope and evidence

This comparison fixes **one** four-vertex, depth-two QAOA transcript,
`qaoa-cycle-4-p2-alternate-seed-7`, from the
[complete Clifford+T replay](qaoa.md#complete-clifford-t-workload). All 109 training groups, one
holdout, 54 source circuits, 49 original cost parents and 82,032 preparations
remain charged. The ideal logical transformation is checked against saved
full-operator and joint-readout references. A separately bounded local QDK
resource-model pass returned all 54 individual class results.

The physical result is a matched **conditional serialized model** on one
fixed machine. It is not a QPU experiment, a new factory/code search, a result
over all QAOA/VQE runs, or a prediction of a new adaptive optimizer path.
:::

(qaoa-word-fundamental-transformation-and-cost)=
#### Fundamental transformation and cost

Partition each saved stream into maximal contiguous single-wire words of
H, X, Y, Z, S, S-dagger, T and T-dagger gates. A different wire, CX, reset,
measurement or barrier ends a word. Select a word only when its wire is
$w>0$ and its T/T-dagger count $k\ge3$. The rule is fixed; there is no angle
optimization, word-selection search or new synthesis.

Use $w$ adjacent SWAPs to bring that data state to D0, execute the exact saved
word, and reverse the SWAP path. Each SWAP is three explicit CX gates. The
boundary permutation is the identity, and conjugating the word by the forward
and reverse permutation reproduces the original operation on Dw, including its
phase, on arbitrary entangled data. All original source events and final
measurement maps remain in order.

The frozen backend uses **one-way** fresh-magic delivery. For this word its
baseline charges $k(w+1)$ delivery SWAPs. The candidate retains one port
delivery SWAP per T and adds $2w$ data SWAPs, saving $w(k-2)$ SWAPs, or
$3w(k-2)$ CX macros. Injection, correction, port reinitialization, supply
retries and handoff residence remain charged. The parity, routing and protected
operation assumptions are those of the unchanged instrument model
{cite:p}`resource_horsman_2012,resource_litinski_2019`.

At source level, a selected word adds $6w$ CX gates and no data wires. For a
line of $n$ data wires and a declared word catalogue, added gate work is
$O(\sum w)$ and word execution retains its original gate count. The implemented
study is bounded to four data wires; this formula does not establish an
efficient routed design or physical gain for arbitrary larger layouts.

(qaoa-word-circuit-excerpt-and-independent-checks)=
#### Circuit excerpt and independent checks

The original 54 classes contain **744 selected words**. Their materialized
circuits add **7,020 actual CX gates** while preserving every T gate, phase,
source event and joint readout. The complete source/candidate operation counts
are 306,508 and 313,528 across distinct circuits; every class has its actual
preparation weight.

```{embed} #opt-word-transport-circuit
```

This folded figure is the **smallest selected real word**, with 227 local
operations and 94 T gates on D1, plus six explicit data-transport CX gates.
Idle spectator wires are omitted from the drawing. It is a word excerpt,
with arbitrary boundary data, rather than a complete initialization/readout
trial. Full trial QASM and operation/segment records remain downloadable for
all 54 classes.

Recorded independent checks compare phase-restored full 16-by-16 operators and
all 16 joint-Z outcomes, without fitting global phase. Short corrected-injection
Kraus and density checks cover arbitrary entangled data and initialized
factory/port interfaces. These checks assume ideal supplied magic inputs;
they do not simulate a physical encoder, decoder or factory circuit.

(qaoa-word-end-to-end-resources-and-fault-tolerance)=
#### End-to-end resources and fault tolerance

The exact selected workload, source angles, quality/readout targets and
selection-aware statistical allocation remain unchanged. It retains all
65,648 training preparations and 16,384 holdout preparations, all 328,128
data-qubit readouts and all 182,646,240 T-state demands. The 49 original cost
parents stay distinct even where candidate signatures coincide.

The machine is frozen at distance 15, fifteen data/workspace/bridge patches,
6,735 data/workspace qubits and 236,623 factory qubits: **243,358** in total.
Physical error $10^{-4}$, 100 ns gate/initialization time, 500 ns measurement,
1,000 ns code cycles and 10,000 ns controller reaction retain the original
profile. The source cost identity fixes the same ISA/T instruction and retry
cap six. No code distance, factory or pipeline search is performed.

For each complete preparation, sum all serialized event times, take the
integer code-cycle ceiling, and only then multiply by the class's shots.
The candidate has **602,466,111,572** weighted exposure cycles. Ceilings after
global weighting would undercharge 26,624 cycles. The 784-cycle-per-T output
handoff exposure remains separate, as do initialization, readout, controller,
factory-quality and availability terms.

```{include} ../../assets/generated/optimization/word-transport/summary.md
```

```{embed} #opt-word-transport-runtime
```

Across this frozen workload, the changed transport reduces 788,596,608 CX
macros and 8,674,562,688 controller decisions under the unchanged macro model.
These are model operation counts, not microscopic factory or hardware
measurements. The per-parent CSV retains exact allowances; the source-accounting
record also preserves the original recorded floats. Factory availability is
unchanged and separate from the
reported physical-fault score.

Classical input handling, compilation/synthesis preprocessing, hardware
optimizer latency and final statistical-processing wall time remain outside
the modeled quantum schedule. Protected parity geometry, tracked frames,
data/factory handoff, initialization/memory/readout residuals and decoder
throughput are conditional engineering assumptions. This result gives neither
a new Pareto frontier nor global resource optimality. The Grover baseline and
the separate QAOA/VQE negative controls retain their own tasks and evidence.

(qaoa-word-verification-provenance-and-formalization)=
#### Verification, provenance and formalization

The actual local resource-model pass completed 54 Trace calls in approximately
12.05 seconds of parent wall time; that local wall time differs from the modeled
602,466-second serialized quantum schedule. Full captured scientific Trace,
ISA/T input and typed returned records are retained byte-for-byte. Public
saved verification reconstructs every circuit word and all class/parent/whole
workload accounting with **zero Native calls**.

Private approval, process, deadline, permit and canonical-state files remain
private. Their audited hashes and final guardian-seal scope appear in a
sanitized execution summary. The original persisted parent/journal kept their
provisional status; final completion is supported by the matching post-export
seal and successful CLI, not a rewritten record. Public scientific verification
does not independently reproduce those private operational clock checks.

Factory `runs`, `states` and returned `error_rate` remain captured, semantically
unvalidated metadata. Exactly-one-copy validation and interpreted ISA/T
time, space and output-error costs remain enforced. A fresh graph restores
the evaluated cost identities; the original pipeline/layout graph is
unavailable, so its equivalence is unclaimed.

Formal coverage remains **classical C4 reference coverage only**. The recorded
ideal operator/channel checks do not prove the Python compiler, physical
schedule, hardware confidence or adaptive optimizer behavior in Lean.

(qaoa-word-reproduce-and-inspect)=
#### Reproduce and inspect

```bash
python scripts/verify_optimization_word_records.py
python scripts/generate_optimization_word_report.py --output-dir artifacts/word-report
pytest tests/test_optimization_word_records.py tests/test_optimization_word_native.py
```

Use Python 3.12.14 and the [locked environment](../../requirements-lock.txt).
Choose a new output directory for report formatting; the command refuses
existing directories to preserve scientific inputs and prior outputs.
The verifier is standard-library-only; the report uses the credited Qiskit
2.6.2 and Matplotlib 3.11.2 for original circuit/plot rendering
{cite:p}`opt_qiskit262,opt_matplotlib3112`.
The existing transport source supplies the bounded primitive transformation;
the saved bundle also retains exact producer snapshots and complete 54-class
trial exports. Generating a larger variant does not inherit this one-machine
physical result.

The original prepare/run scripts are historical workflow snapshots; public
saved reproduction uses the verifier and report commands above.

- [Immutable scientific manifest](../../assets/generated/optimization/word-transport/record-manifest.json)
- [Aggregate comparison](../../assets/generated/optimization/word-transport/comparison.csv),
  [54 class results](../../assets/generated/optimization/word-transport/classes.csv) and
  [49 parent budgets](../../assets/generated/optimization/word-transport/parents.csv)
- [Full Native scientific inputs](../../assets/generated/optimization/word-transport/native-inputs.json.gz),
  [captured producer returns](../../assets/generated/optimization/word-transport/producer-returns.jsonl.gz) and
  [checked Native result](../../assets/generated/optimization/word-transport/native-result.json)
- [Exact word excerpt QASM](../../assets/generated/optimization/word-transport/word-excerpt.qasm)
  and [full scientific record README](../../assets/generated/optimization/word-transport/README.md)
- [Public execution summary](../../assets/generated/optimization/word-transport/execution-summary.json)
  and [metadata transformation map](../../assets/generated/optimization/word-transport/record-transformations.json)

## Verification and formalization

Independent checks compare explicit circuits with classical references, preserve preparation and measurement maps, and replay recorded sampled decisions from their counts. The finite-shot protocol's fresh holdout bounds concern selected fixed states conditional on their training histories; they do not certify adaptive training queries, convergence, or physical hardware. Circuit export and bounded classical simulation have separate limits.

**Classical C4 reference coverage only.** The reproduced Lean artifact establishes finite C4 cut values and the stated classical equality predicate. It does not connect those definitions to Python or prove QAOA circuits, VQE ansätze, stochastic decisions, synthesis, physical schedules, or hardware confidence. [Proof source](../../assets/generated/optimization/formal/C4Reference.lean) and [verification record](../../assets/generated/optimization/formal/verification.json) retain the exact existing coverage.

## Reproduce and inspect

Use the pinned Python 3.12 environment and the Qiskit core/NumPy versions credited in the [parent chapter](../optimization.md). The implementation, figures, and this exposition are original; the implementation uses the project's MIT license.

```python
from math import pi
from openquantum.optimization import build_qaoa_circuit

edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
circuit = build_qaoa_circuit(4, edges, [pi / 4], [pi / 8], measure=True)
```

- [Executed notebook](../../notebooks/optimization.ipynb)
- [Complete runs and package provenance](../../assets/generated/optimization/runs.json)
- [Every objective evaluation](../../assets/generated/optimization/evaluations.csv)
- [Primary fitted measured QASM](../../assets/generated/optimization/qaoa.qasm)
- [Standalone implementation](../../assets/generated/optimization/optimization.py)
- [Artifact generator](../../assets/generated/optimization/generate_optimization.py)
- [Logical resources](../../assets/generated/optimization/logical-resources.json)

## References

```{bibliography}
```
