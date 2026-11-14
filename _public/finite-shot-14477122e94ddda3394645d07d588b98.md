---
title: "Finite-shot optimization and independent readout"
description: "Sampled QAOA and VQE parameter trajectories, complete shot records, fresh holdout estimates, and conditional fixed-readout resources."
---

The [first optimization examples](../optimization.md) fit angles using exact statevector expectations and sample only the final readout. This increment instead makes each parameter decision from finite outcomes. It uses the same four-vertex MaxCut cycle, the same two-qubit Hamiltonian $H=-Z_0-Z_1+\tfrac12X_0X_1$, and the same explicit QAOA and VQE circuits {cite:p}`opt_qaoa2014,opt_vqe2014`.

:::{note} Scope of the experiment

These are reproducible ideal finite-shot simulations. Counts are drawn classically from explicit circuit statevector probabilities; they are not hardware measurements. Training uses the sampled objectives and their descriptive standard errors. Exact diagnostics and fresh final readout are reported separately and do not choose the training moves.

The finite schedule and acceptance rule are heuristics. The observed trajectories do not certify convergence, optimal angles, or a success probability for the optimizer. The original fixed-readout projection concerns the final readout only; the later sections separately cost complete stored trajectories and the size-study workloads. The [complete recorded-trajectory replay](finite-shot.md#recorded-trajectory-resources) maps training and holdout batches as fixed circuit inputs. Predicting the cost of a new hardware-driven adaptive path remains **planned**.
:::

## Tasks, targets, and predetermined runs

For QAOA, maximize expected cut on edges $(0,1),(1,2),(2,3),(0,3)$ by minimizing $f_Q=-\langle C\rangle$. For VQE, minimize $f_V=\langle H\rangle$. QAOA uses one layer with phase `CX; RZ(-gamma); CX` per edge and mixer `RX(2*beta)`. VQE uses `RY(theta)` on qubit zero followed by `CX(0,1)`. All angles are radians. Input preparation, phase operations, mixer, basis changes, and measurement are explicit in the circuits.

| Seed | QAOA initial $(\gamma,\beta)$ | VQE initial $\theta$ |
|---|---:|---:|
| 7 | $(0.2,0.1)$ | $0.6$ |
| 19 | $(0.6,0.4)$ | $-1.1$ |
| 41 | $(-0.3,0.25)$ | $2.2$ |

These three seed/start pairs are declared before running either task. Seed 7 is the primary example and the input to the fixed-readout resource projection; no run is chosen after comparing outcomes. The requested toy quality targets are final exact expected cut at least $2.90$ and final exact VQE energy gap at most $0.03$ above $E_0=-\sqrt{17}/2$. They are evaluated **after training** with independent analytic and matrix references. They do not stop the optimizer, certify finite-shot confidence, or impose an angle-accuracy requirement.

```{include} ../../assets/generated/optimization/sampled-summary.md
```

## What one sampled objective contains

QAOA uses one all-Z group. Each bitstring yields its complete cut size $C(z)$, sharing the outcomes across all four edges. The minimized estimate is $\widehat f_Q=-\overline C$. Its descriptive standard error is $s_Q=\sqrt{\widehat{\mathrm{Var}}(C)/N}$, using sample variance of whole cut scores.

VQE uses two independently sampled groups. All-Z outcomes score $Y_Z=-Z_0-Z_1\in[-2,2]$. All-X outcomes, obtained by H basis changes on both qubits, score $Y_X=\tfrac12X_0X_1\in[-1/2,1/2]$. Then

$$
\widehat f_V=\overline Y_Z+\overline Y_X,\qquad
s_V=\sqrt{\frac{\widehat{\mathrm{Var}}(Y_Z)}{N_Z}
+\frac{\widehat{\mathrm{Var}}(Y_X)}{N_X}}.
$$

The two Z terms are scored **jointly** from the same bitstrings, so their covariance is retained. They are not treated as independent term measurements. Every VQE objective call uses the stated shot count in **each** of its two groups. No symmetry filtering, error mitigation, device noise, or extra postselection is applied.

## A fixed, fully recorded training schedule

The sampled polling optimizer makes one initial objective call at 256 shots per group. It then runs 32 sweeps in four blocks. Each block uses its stated step and shot count for all eight sweeps; lack of an accepted move does not alter the schedule.

| Sweeps | Coordinate displacement, radians | Shots per group per call |
|---|---:|---:|
| 1–8 | $0.4$ | 256 |
| 9–16 | $0.2$ | 512 |
| 17–24 | $0.1$ | 1,024 |
| 25–32 | $0.05$ | 2,048 |

Every sweep remeasures the incumbent at that sweep's shot count, then evaluates both negative and positive displacements along each coordinate from that same incumbent. QAOA therefore uses five objective calls per sweep: one incumbent and four proposals. VQE uses three: one incumbent and two proposals. Angles are reduced modulo $2\pi$.

Let $\widehat f_i,s_i$ denote the new incumbent estimate and standard error, and let $\widehat f_c,s_c$ denote the best sampled proposal. A move requires

$$
\widehat f_i-\widehat f_c>2\sqrt{s_i^2+s_c^2}.
$$

Otherwise the parameters stay at the incumbent. The run stops when the schedule is exhausted. The combined standard error is descriptive: the rule is not a calibrated hypothesis test or a 95% acceptance guarantee. Selecting among proposals, repeating adaptive comparisons, and occasionally obtaining zero empirical variance prevent that interpretation. An accepted sampled improvement can be an exact deterioration, and a rejected move can be an exact improvement.

NumPy `SeedSequence` separates random streams with entropy `[master_seed, algorithm_id]` and spawn keys `(family, evaluation_index, group_index)` {cite:p}`opt_seedsequence254`. Algorithm IDs are zero for QAOA and one for VQE; family zero is training and family one is fresh final validation. Keys distinguish calls and measurement groups. They support reproducibility and avoid accidental stream reuse; they do not mathematically prove independence of a pseudorandom generator. The statistical statements here use the ideal independent-sampling model.

The [training record](../../assets/generated/optimization/sampled-runs.json) retains every sampled call, candidate parameters, group counts, group means, standard errors, stream identifiers, accepted and rejected moves, stopping reason, classical optimizer time, and post-training diagnostics. No exact expectation or analytic derivative enters the acceptance rule.

![](#opt-sampled-trajectories)

The plotted exact cut or energy along a trajectory is a diagnostic, rather than an objective available to the sampled optimizer. The exact optimizer from the first increment remains an independent reference. Three predetermined trajectories show behavior on these instances; the fraction meeting a target is not a certified population success probability.

## Fresh final readout and what its intervals mean

After training, each selected final state receives a fresh 4,096-shot all-Z QAOA readout or fresh 4,096-shot-per-group VQE readout. Validation streams are distinct from all training streams. Exact expectation, sampled estimate, descriptive standard error, and a conservative bounded-score interval are reported separately.

Hoeffding's inequality bounds independent bounded samples without a normal approximation {cite:p}`opt_hoeffding1963`. For one final QAOA holdout with $N$ scores in $[0,4]$, a two-sided interval of failure probability at most $\delta$ has radius

$$
\epsilon_Q=4\sqrt{\frac{\log(2/\delta)}{2N}}.
$$

For the sum of the independently sampled VQE group means, the group score ranges are four and one. A direct bound for the sum has radius

$$
\epsilon_V=\sqrt{\frac12\left(\frac{16}{N_Z}+\frac1{N_X}\right)
\log(2/\delta)}.
$$

At 4,096 shots per group and $\delta=0.05$, these radii are about $0.0849$ in cut units and $0.0875$ in energy units. They refer to **one holdout's** expectation estimate. Six individually 95% intervals do not provide a simultaneous 95% statement for all runs. The VQE formula already bounds the combined Z/X estimate; it does not need an additional group split for this statistical interval.

Fresh holdout coverage remains valid conditional on the training history and selected angles under the independent-sampling model. It bounds readout error at those angles. It does not prove that training achieved a target, bound the optimizer's failure probability, or turn the observed mean into certified optimal angles. In particular, a holdout interval can be too wide to certify the requested $0.03$ VQE energy gap even when the toy's exact diagnostic meets it. Sampled energy can also fall below the exact ground energy through readout noise.

![](#opt-sampled-readout)

## End-to-end resource analysis

All proposals and all incumbent remeasurements count, including rejected moves and the initial call. With the default schedule, one QAOA run has 161 objective calls and 153,856 training shots. One VQE run has 97 objective calls, each with two groups, and 184,832 training shots. These totals follow from the executed trace, rather than from the number of accepted updates.

| Workload for the three predetermined runs | QAOA shots | VQE shots | Total shots |
|---|---:|---:|---:|
| Training | 461,568 | 554,496 | 1,016,064 |
| Fresh final holdout | 12,288 | 24,576 | 36,864 |
| Training plus holdout | 473,856 | 579,072 | 1,052,928 |

Each shot represents a new state preparation in the ideal circuit protocol. The implementation draws counts classically from circuit probabilities, so these are protocol shot counts rather than physical circuit executions. The [sampled resource ledger](../../assets/generated/optimization/sampled-resources.json) weights the actual circuit gates by every call's shot count, separates training from validation, includes basis changes and qubit readout, and records unsynthesized rotations. A single final ansatz evaluation is not this hybrid run.

CPU timings measure the local simulator, random sampling, and classical control. They do not measure a quantum device's reset, routing, gate, measurement, communication, or feedback latency. Input parsing and separate classical reference timing are disclosed where unmeasured. The exact diagnostics and figure-generation work are additional classical checks rather than hidden quantum objective evaluations. The [recorded-trajectory replay](finite-shot.md#recorded-trajectory-resources) accounts for this fixed training record; it does not predict a fresh hardware-driven adaptive execution.

(conditional-resources-for-a-fixed-final-readout)=
### Conditional resources for a fixed final readout

The projection takes seed 7's fitted final angles as fixed classical inputs. It estimates only 4,096 QAOA all-Z executions and, separately, 4,096 VQE all-Z plus 4,096 VQE all-X executions. It includes the explicit preparations, rotations, basis changes, and readout of those exported circuits. The sampled training decisions and their million-shot workload are excluded from this physical projection.

Two frozen physical error scenarios, $10^{-3}$ and $10^{-4}$, use the existing baseline's 100 ns gate and 500 ns measurement timing assumptions. The declared physical-fault budget is $10^{-3}$ for the whole QAOA fixed-readout task and separately $10^{-3}$ for VQE's fixed-readout task; VQE allocates half to each measurement group. That is a physical-fault budget, distinct from the holdout interval's sampling failure probability and from rotation approximation.

The projections use the pinned **Microsoft QDK local resource estimator 1.34.0**, released under [MIT](https://github.com/microsoft/qdk/blob/81de0d2e5347e1e9f10e05066320fffdb3b7058c/LICENSE.txt) {cite:p}`opt_qdk1340`. The linked audited source commit is a code pointer, not a claimed package release commit. The native PSSPC rotation model uses a mean empirical fit associated with mixed-fallback gate approximation {cite:p}`opt_rotation2022`.

Each measured circuit has a per-execution synthesis-model allowance $10^{-5}$ with its recorded allocation/count assumptions. It is a conditional expected-cost model, rather than an executed Clifford+T synthesis or a certificate of the resulting cut or energy error. Fitted mean quality does not certify angle accuracy, optimizer success, or convergence. The projection record must be read with those limits and its exact circuit and estimator provenance.

```{include} ../../assets/generated/optimization/sampled-projections.md
```

The [projection JSON](../../assets/generated/optimization/sampled-projections.json) records actual estimator results and exclusions. The [resource discussion](../optimization.md#end-to-end-resource-analysis) explains the synthesis boundary. The Grover surface-code baseline is unchanged; these fixed-readout reports do not compare different optimization schedules or demonstrate a physical advantage for QAOA or VQE.

The [portability increment](portability.md) keeps starts fixed while varying seeds on a triangle and a second Hamiltonian. Those changed tasks have their own results and logical records; they do not inherit the original fixed-readout physical estimates.

## Reproduce and inspect

Use the pinned Python 3.12 environment and the original Qiskit core, NumPy, and Matplotlib packages credited in the [parent chapter](../optimization.md). No optimizer package or new dependency is required.

```bash
python scripts/generate_optimization_sampled.py
python scripts/generate_optimization_projections.py
python scripts/execute_notebooks.py
```

The physical-projection command requires a source checkout containing the shared `src/openquantum/resource_estimation.py` adapter and the audited QDK environment. A standalone export of that physical adapter remains planned. The sampled generator and notebook can be reproduced separately from the public downloads below.

The [sampled notebook](../../notebooks/optimization_sampled.ipynb) runs the bounded experiment and supplies the labeled figures reused above. Saved outputs serve the static site without a live kernel. To reproduce its public downloads, save the [matching standalone module](../../assets/generated/optimization/sampled/optimization.py) and [generator](../../assets/generated/optimization/sampled/generate_optimization_sampled.py) together with the [implementation license](../../assets/generated/optimization/sampled/LICENSE.txt), using the pinned environment linked by the parent chapter. The generators and records define materialization and simulation limits; no larger-instance convergence behavior is inferred from these examples.

- [All sampled runs and stream metadata](../../assets/generated/optimization/sampled-runs.json)
- [Every sampled objective evaluation, CSV](../../assets/generated/optimization/sampled-evaluations.csv)
- [Complete training and validation resource ledger](../../assets/generated/optimization/sampled-resources.json)
- [Conditional fixed-readout projection](../../assets/generated/optimization/sampled-projections.json)
- [Executable sampled notebook](../../notebooks/optimization_sampled.ipynb)
- [Sampled implementation export](../../assets/generated/optimization/sampled/optimization.py)
- [Sampled artifact generator](../../assets/generated/optimization/sampled/generate_optimization_sampled.py)

(recorded-trajectory-resources)=
## Complete recorded-trajectory resources

The [fixed final-readout projection](finite-shot.md#conditional-resources-for-a-fixed-final-readout) omits sampled training. This report instead maps every measured group in the original four-vertex-cycle QAOA and first two-qubit VQE records, including rejected proposals, incumbent remeasurements, initial calls, and fresh final validation. It treats the stored angles and shot allocations as fixed inputs to a **recorded-trajectory replay**.

:::{note} What is estimated

Replay means preparing the recorded circuits with their recorded repetitions. Fresh hardware measurements are not allowed to choose different parameters in this cost calculation. A hardware-driven optimizer could follow a different angle path, with different synthesis costs and achieved quality, even when its call and shot schedule is fixed.

The circuit mapping and conditional estimator results have separate status in the generated summary below. A checked mapping is not an executed fault-tolerant run. Physical results use an expected synthesis-cost model; they certify neither concrete synthesized sequences nor worst-case work, runtime tails, convergence, or output-error confidence.
:::

(sampled-trajectory-count-the-entire-record)=
### Count the entire record

The original experiment contains three predetermined seed/start pairs for each task. Its 32-sweep sampled schedule and holdout are documented in the [finite-shot chapter](finite-shot.md). Here the workload is the complete collection of those recorded runs, with a separate resource budget for each task.

| Task | Training calls | Holdout calls | All objective calls | Measured group batches | Modeled shots |
|---|---:|---:|---:|---:|---:|
| Four-vertex QAOA | 483 | 3 | 486 | 486 | 473,856 |
| First two-qubit VQE | 291 | 3 | 294 | 588 | 579,072 |
| Complete collection | 774 | 6 | 780 | 1,074 | 1,052,928 |

A VQE objective call contains two groups, all-Z and all-X. QAOA contains one all-Z group. Batch repetition weights retain the exact per-call shot allocation rather than multiplying one final circuit by an optimizer iteration count. The ideal protocol repeats input-state preparation and all basis changes for each shot. Samples in the source experiment were drawn classically from statevector probabilities; these totals are not measured QPU consumption.

```{include} ../../assets/generated/optimization/trajectory-resources/summary.md
```

(sampled-trajectory-from-every-sampled-batch-to-an-explicit-circuit)=
### From every sampled batch to an explicit circuit

The [manifest](../../assets/generated/optimization/trajectory-resources/manifest.json) binds the replay to the source run record, generator, environment, and implementation. The [execution mapping](../../assets/generated/optimization/trajectory-resources/execution-mapping.csv) identifies the task, run, seed, training or validation phase, call, measurement group, parameters, and shot repetitions for every batch.

Each batch has source and lowered measured OpenQASM files identified by SHA-256. QAOA includes H preparation, edge phases, and mixer rotations. VQE includes its RY/CX preparation and the all-X H basis changes where required. Final qubit-to-classical measurement maps are retained. Initial all-zero availability is an assumption; physical reset time is excluded where unmeasured.

For every batch, the small-system check compares source and lowered unitary operators up to a global phase before final measurement, and verifies the final measurement map separately. That check validates numerical lowering of each particular circuit. It does not validate an approximate Clifford+T replacement of its arbitrary angles.

Identical QASM content can share a stored file by its exact hash. Every original batch still has an execution-mapping entry and its own repetition weight. Exact file deduplication therefore reduces artifact duplication without removing objective evaluations or shots.

(sampled-trajectory-share-estimator-work-only-when-the-native-cost-model-agrees)=
### Share estimator work only when the native cost model agrees

Different fitted angles generally produce different unitaries. Resource-model grouping makes no claim that those unitaries are equivalent. Equal gate counts alone also do not establish equal cost: native trace ordering, repetitions, properties, and resource requirements can affect the estimator.

The [model-signature record](../../assets/generated/optimization/trajectory-resources/model-signatures.json) groups batches after comparing cost-canonical native and PSSPC trace JSON, with the recorded derived metadata. Complete instrumented trace JSON is retained. The cost signature excludes only native `EVALUATION_TIME` instrumentation, property 7, whose recorded pinned-source audit establishes that it is not consumed by the physical cost calculation. All other fields and properties remain in the comparison. This CPU instrumentation time is not quantum runtime.

The grouping uses the same synthesis setting, physical profile, and per-execution budget. Its equality is equality of inputs relevant to this pinned **cost model**, rather than equality of circuit behavior at different angles. A bounded second-QASM probe for a shared model class is recorded separately, comparing actual physical Pareto fields under the same budget and profile. Its scope and result are retained rather than treated as a general circuit-equivalence proof.

QDK's default OpenQASM importer constructs a canonical trace from logical counts, already abstracting continuous angles and original Clifford topology. Comparing those native traces checks equality of inputs inside the declared estimator model; it does not certify a routed schedule for every original gate. The source and lowered QASM, gate counts, measurement maps, and numerical lowering checks preserve the original circuit evidence separately.

Each group retains a representative input and maps all constituent batches to the corresponding estimator report. Their repetition weights are summed only after the model-signature comparison. The [projection reports](../../assets/generated/optimization/trajectory-resources/projection-reports.json) retain the actual estimator outputs and model provenance. The distinction between source-circuit equivalence checks and cost-model grouping is essential: neither substitutes for the other.

(sampled-trajectory-keep-physical-faults-and-rotation-approximation-separate)=
### Keep physical faults and rotation approximation separate

The replay declares a whole-task physical-fault allowance of $\delta_Q=10^{-3}$ for all 473,856 QAOA shots and, separately, $\delta_V=10^{-3}$ for all 579,072 VQE shots. Each task allocates a uniform per-execution allowance,

$$
\delta_{\alpha,\mathrm{execution}}=
\frac{10^{-3}}{N_\alpha},\qquad
\alpha\in\{Q,V\}.
$$

The report checks this allocation across all batches. A union-bound interpretation requires the estimator's per-execution physical contribution to be a calibrated upper bound for the hardware/model assumptions; the replay does not supply that experimental calibration. Independence of physical faults is not needed for the union bound.

Each prepared circuit also receives a **per-execution synthesis-model allowance** of $10^{-5}$. This is a coherent approximation-model quantity, not a stochastic failure probability. It is not multiplied by the million-shot total and labeled whole-run failure. Each shot prepares a fresh state; repeated preparations do not form one long coherent computation on a shared state. Repetition also does not remove systematic observable bias from an approximate circuit. Adaptive classical feedback can turn changed outcomes into a different parameter path; no approximation or accuracy bound for that complete classical–quantum process is established here.

The pinned Microsoft QDK resource estimator 1.34.0 and native PSSPC model use the audited expected rotation-cost fit {cite:p}`opt_qdk1340,opt_rotation2022`. The fit is not an executed synthesis sequence, an instance-specific approximation certificate, a worst-case cost bound, or a bound on fallback-protocol runtime tails. The $10^{-5}$ model allowance consequently does not certify the final cut value or energy error. Sampling uncertainty, fitted quality, and physical-fault allocation remain separate quantities.

The two physical profiles retain the frozen $10^{-3}$ and $10^{-4}$ physical error rates, 100 ns gate time, and 500 ns measurement time. The Grover baseline remains unchanged. This report does not alter requested problem quality or hardware assumptions to manufacture an improvement.

(sampled-trajectory-aggregate-serial-time-and-peak-footprint-with-explicit-exclusions)=
### Aggregate serial time and peak footprint with explicit exclusions

For a declared model-point selection, serial quantum time is weighted by every batch's shot count and peak footprint is the maximum selected footprint:

$$
T_{\mathrm{serial}}=\sum_b N_b\,t_b,\qquad
Q_{\mathrm{peak}}=\max_b q_b.
$$

The declared choices can select independent per-model-signature Pareto points. Configuration changes, transport, communication, routing, physical reset, and feedback overhead are excluded where unmeasured. A single common physical layout or factory configuration for the complete trajectory is not certified. Thus these aggregate choices are conditional serial model costs, rather than a measured machine schedule or a globally optimized architecture.

The local CPU-control and objective-simulation timings from the original experiment are reported separately. A simulator's control time is not a measurement of classical feedback latency in a fault-tolerant processor, and it is not silently added to or hidden inside quantum runtime. Concurrent execution, reuse across shots, and latency overlap require separate scheduling evidence.

(sampled-trajectory-reproduce-and-inspect)=
### Reproduce and inspect

Reproduction requires the original sampled run record, its matching implementation, and a source checkout containing the shared `src/openquantum/resource_estimation.py` adapter in the pinned Python 3.12/QDK environment. The manifest and signature records identify the executed code, declared model settings, and input hashes. Standalone sampled experiment downloads remain available through the [finite-shot chapter](finite-shot.md); exporting the shared physical adapter is a separate integration step.

The recorded replay concerns only the original four-vertex cycle and first Hamiltonian. The [portability cases](portability.md) are different inputs with their own logical records, and do not inherit these physical estimates.

- [Replay manifest and execution contract](../../assets/generated/optimization/trajectory-resources/manifest.json)
- [Every batch and its source/lowered QASM mapping](../../assets/generated/optimization/trajectory-resources/execution-mapping.csv)
- [Native canonical/PSSPC model signatures](../../assets/generated/optimization/trajectory-resources/model-signatures.json)
- [Actual conditional estimator reports](../../assets/generated/optimization/trajectory-resources/projection-reports.json)

(quality-and-workload-across-sizes)=
## Quality and workload across sizes

The earlier examples establish inspectable circuits and small independent checks. This study varies input size and QAOA depth under a fixed sampling protocol, then asks whether the selected states meet stated quality targets. It uses the original QAOA and VQE constructions {cite:p}`opt_qaoa2014,opt_vqe2014`; the figures and implementation are original.

:::{note} Study status and interpretation

The locked recipe has been executed for all **72 runs**. Independent post-training audits meet the target in **19 of 60 QAOA runs** and **4 of 12 VQE runs**. Simultaneous holdout intervals support the target in **18 QAOA runs** and **no VQE runs**; every run supports the separately requested $\pm0.05$ normalized readout precision. All predetermined runs and misses are retained, with no retuning of initializations, schedule, targets, or readout allocation.

The experiment uses ideal counts drawn classically from circuit probabilities. Simultaneous holdout bounds concern the selected states under that sampling model, not a population success rate, optimizer convergence, physical hardware, or quantum advantage.
:::

(sampled-quality-fixed-inputs-and-targets)=
### Fixed inputs and targets

QAOA uses unweighted paths and cycles with $n=3,4,5,6,8$ vertices and depths $p=1,2$. A path has edges $(j,j+1)$; a cycle adds $(n-1,0)$. Each depth starts at the interleaved angle vector $(0.2,0.1)$ repeated $p$ times, in order $(\gamma_0,\beta_0,\ldots)$. Three distinct seed namespaces, `7`, `19`, and `41`, give 60 runs across the 20 graph/depth cases.

The minimized training estimate is the negative joint cut mean. All edges are scored from each all-Z bitstring. Full classical enumeration supplies $C_\star$, independently corroborating the simple references: $n-1$ for a path, $n$ for an even cycle, and $n-1$ for an odd cycle. The post-training quality target is

$$
\frac{\langle C\rangle}{C_\star}\ge0.80.
$$

These graph families are classically tractable; the experiment tests implementation behavior and resource growth rather than hard-instance advantage.

VQE uses open chains of $n=2,3,4,6$ qubits with dimensionless Hamiltonian

$$
H_n=-\sum_{j=0}^{n-1}Z_j
+\frac12\sum_{j=0}^{n-2}X_jX_{j+1}.
$$

Prepare $|0\rangle^{\otimes n}$, apply one `RY(theta[j])` per qubit, and then the ascending CX chain $(0,1),(1,2),\ldots,(n-2,n-1)$. All angles initially equal $0.6$. This is a concrete shallow ansatz, not a claim that every chain ground state lies in its circuit family. The three seed namespaces give 12 VQE runs. Dense diagonalization supplies an independent ground-energy reference $E_0$, and the post-training target is energy-density gap

$$
\frac{\langle H_n\rangle-E_0}{n}\le0.05.
$$

Normalization permits a common density target across sizes; it does not fix the same absolute energy tolerance, since the allowed absolute gap is $0.05n$.

The two diagrams show the final all-Z circuits of the **first prescribed case and seed** for each method: path $n=3,p=1$ QAOA and chain $n=2$ VQE, both seed `7`. They are selected by recipe order, with every other parameter/basis circuit retained in the manifest. VQE all-X groups additionally apply H to every qubit.

![First prescribed QAOA final circuit.](../../assets/generated/optimization/quality-scaling/qaoa-path3-p1-circuit.svg)

![First prescribed VQE final all-Z circuit.](../../assets/generated/optimization/quality-scaling/vqe-chain2-z-circuit.svg)

(sampled-quality-decisions-use-sampled-objectives)=
### Decisions use sampled objectives

Each run makes an initial call with 128 shots per measurement group and then 12 sweeps:

| Sweeps | Coordinate step, radians | Shots per group per objective call |
|---|---:|---:|
| 1–4 | $0.4$ | 256 |
| 5–8 | $0.2$ | 512 |
| 9–12 | $0.1$ | 1,024 |

Every sweep freshly samples its incumbent and both coordinate displacements along every parameter, from that same incumbent. The best sampled proposal needs a reduction exceeding twice the combined descriptive standard error. The only stopping rule is schedule exhaustion. This adaptive comparison is a heuristic, not a calibrated test or convergence criterion. Exact selected-state expectations and classical references are kept separate from the sampled quantities used for decisions.

VQE uses two groups. The all-Z score is $Y_Z=-\sum_jZ_j$, with range $[-n,n]$. The all-X score is $Y_X=\tfrac12\sum_jX_jX_{j+1}$ after H basis changes on every qubit, with range $[-(n-1)/2,(n-1)/2]$. Each score is formed **jointly from one bitstring**, preserving covariance among the Z terms and among the overlapping XX terms. The two group means are added; group independence does not make terms within a group independent.

The record retains case definitions, parameter order, every call and proposal, counts, group scores and variance, accepted/rejected moves, and random-stream metadata. Case-specific namespaces distinguish repeated seed labels across inputs, and fresh holdout streams are distinct from training streams {cite:p}`opt_seedsequence254`. No outcome-based case selection is applied.

(sampled-quality-simultaneous-holdout-precision-and-selected-state-quality)=
### Simultaneous holdout precision and selected-state quality

After training, all 72 selected states receive a fresh **16,384-shot-per-group** holdout. There are 60 QAOA groups and 24 VQE groups, giving 84 predetermined groups. The allocation was chosen analytically before outcomes. Set

$$
\delta_{\rm group}=\frac{0.05}{84},\qquad
a=\sqrt{\frac{\log(2/\delta_{\rm group})}{2\cdot16384}}
\approx0.01574146.
$$

Hoeffding's inequality and a union bound give simultaneous coverage of all 84 group means with failure probability at most $0.05$, under fresh ideal independent samples within each group {cite:p}`opt_hoeffding1963`. The statement remains conditional on the selected angles and training histories. Between-run independence is not needed for the union bound; valid fresh conditional sampling for each group is needed.

The recorded QAOA group uses the conservative input edge-count range $[0,m]$, so its ratio radius is $(m/C_\star)a$. The enumerated $C_\star$ normalizes the post-training report; it does not enter sampled optimization decisions. For VQE, adding the two group intervals and dividing by $n$ gives energy-density radius

$$
\epsilon_{\rm density}=(3-1/n)a.
$$

It ranges from about $0.03935$ at $n=2$ to $0.04460$ at $n=6$. These radii support the separately declared simultaneous $\pm0.05$ readout-precision allocation. This does **not** imply that every selected state achieves the optimization target.

The result table therefore separates three statements: post-training exact audit meets the target; the simultaneous holdout interval supports that selected-state target; and the measurement allocation supports the requested expectation precision. Sampled support requires a QAOA lower ratio endpoint at least $0.80$, or a VQE upper density-gap endpoint at most $0.05$. An interval crossing the target is inconclusive; lack of support is not itself proof of a target miss. Raw estimates and interval endpoints remain visible even if shot noise gives a negative energy-density gap. A negative lower endpoint does not establish energy below the ground state.

```{include} ../../assets/generated/optimization/quality-scaling/summary.md
```

![](#opt-quality-scaling-quality)

![](#opt-quality-scaling-coverage)

The achieved quality depends on the tested input and recipe. Paths at $n=3$ meet the target in all six depth/seed runs; among the larger paths, only the $n=4,p=2$, seed `41` run meets it. The odd cycles at $n=3,5$ meet the target in all twelve runs, while every tested even-cycle run misses it. Odd cycles have $C_\star=n-1$, so their ratio denominator differs from that of an even cycle; these results do not imply a monotone size trend.

The VQE density gap is $0.010811$–$0.011081$ at $n=2$, and $0.179165$–$0.235975$ at $n=6$. The recorded zero-state baseline at $n=6$ has density gap $0.0525804$, so the three fitted six-spin states also miss that simple baseline. The fixed search is retained rather than restarted from a more favorable point. Its four exact target hits remain distinct from the zero simultaneous holdout supports: a valid expectation-precision allocation can still leave a quality threshold inconclusive.

All target misses remain visible. They can reflect the circuit family, initialization, finite search budget, and sampled decisions. One limitation is already known independently: the depth-one C4 formula in the [main chapter](../optimization.md#qaoa-phases-mixing-and-a-small-maxcut-circuit) has maximum mean three, hence ratio $3/4$, below this study's $0.80$ target. A perfect angle optimizer cannot make that circuit family meet the target. The study does not identify the cause of every other miss or estimate a general success probability.

(sampled-quality-equal-sweeps-are-not-equal-workloads)=
### Equal sweeps are not equal workloads

Let $S=4(256+512+1024)=7,168$. A $D$-parameter run has $1+12(1+2D)$ training objective calls and $128+(1+2D)S$ training shots **per group**. QAOA has $D=2p$ and one group; VQE has $D=n$ and two groups.

| Run type | Training calls | Training shots | Holdout shots |
|---|---:|---:|---:|
| QAOA $p=1$ | 61 | 35,968 | 16,384 |
| QAOA $p=2$ | 109 | 64,640 | 16,384 |
| VQE $n=2$ | 61 | 71,936 | 32,768 |
| VQE $n=3$ | 85 | 100,608 | 32,768 |
| VQE $n=4$ | 109 | 129,280 | 32,768 |
| VQE $n=6$ | 157 | 186,624 | 32,768 |

The complete executed collection has 4,483,584 training shots and 1,376,256 holdout shots: **5,859,840 modeled preparations**, 6,408 objective calls including 72 holdouts, and 7,656 measured group batches. These include initial calls, rejected proposals, incumbent remeasurements, and final validation. The generated ledger checks the executed record against the fixed allocation.

![](#opt-quality-scaling-workload)

At depth two, more parameters cause more calls and shots, and each circuit also has more layers. A quality improvement at $p=2$ would therefore be observed under a larger workload, not an equal-budget comparison. Larger VQE inputs likewise increase the proposal count and per-shot circuit width.

Before compilation, one QAOA preparation uses $n$ H, $np$ RX, $mp$ RZ, and $2mp$ CX. The stated VQE ansatz uses $n$ RY and $n-1$ CX; its all-X group adds $n$ H, and every group measures $n$ qubits. Source gates must be weighted by every recorded group's shots. CPU simulation/control times, reference calculation, and postprocessing are separate from hypothetical quantum gate/measurement time. The recorded numerical workflow took **4.704195 s of CPU-host wall time** and **4.701387 s of process CPU time**; trace export and plotting are outside those study timers. Neither time measures hardware feedback. Statevectors and dense references still grow exponentially; this bounded study is not an asymptotic performance test.

(sampled-quality-conditional-physical-replay-and-verification-limits)=
### Conditional physical replay and verification limits

Physical projection concerns the **complete frozen recorded workloads**, rather than one final ansatz. The replay fixes stored parameters, measurement groups, and shot weights. Fresh hardware outcomes could lead to another adaptive trajectory and another resource cost. These are costs of replaying the protocol, not certified costs of attaining its targets. A lower-cost run that misses quality cannot establish an improvement at matched requested quality.

The physical layer now reports a **computed conditional replay of all 72 recorded runs**. It covers all 5,859,840 preparations and 84 model cost classes. The original 64-search guard stopped before any physical-estimator search, and that complete pending result remains in the [cap-64 archive](../../assets/generated/optimization/quality-resources-cap64/manifest.json). A [recorded computation-only amendment](../../assets/generated/optimization/quality-resources/amendment.json) raised the search limit to 96 **before new searches**, without changing cases, seeds, targets, shots, hardware profiles, or error allowances. The amended plan completed 85 physical-estimator searches, including the distinct-QASM cost-consistency probe. Profiles remain frozen at physical error $10^{-3}$ or $10^{-4}$, with 100 ns gates and 500 ns measurements.

The displayed table and manifest come from a fresh full 85-search audit with the current shared adapter. The original cap-64 archive and its amendment preserve the earlier computation history; the 72 executed training runs and their source snapshots remain unchanged.

```{include} ../../assets/generated/optimization/quality-resources/summary.md
```

The [replay manifest](../../assets/generated/optimization/quality-resources/manifest.json), [all execution mappings](../../assets/generated/optimization/quality-resources/execution-mapping.csv), [model signatures](../../assets/generated/optimization/quality-resources/model-signatures.json), and [full projection records](../../assets/generated/optimization/quality-resources/projection-reports.json) expose the fixed workload, allocations, selected code distances/factories, and provenance. The [conditional choices CSV](../../assets/generated/optimization/quality-resources/choices.csv) contains the four displayed aggregates. The separately reported 3,746 trace-signature audits and internal adapter/native operations are not physical-estimator searches and are not represented by the 96-search cap.

Source/lowered hashes and prepared-state/readout checks identify what was replayed. Prepared-state checks on the recorded initial input are not all-input unitary proofs. The executed distinct-QASM probe compares exact model signatures and selected physical-cost fields within the recorded numerical tolerance; it does not claim unitary equivalence between different angles. Native LogicalCounts-derived signatures abstract original gate topology.

The aggregate serial quantum-model time weights every recorded group by its shots; peak footprint is the maximum independently selected class footprint. A common physical configuration is **not verified**, and layout changes, physical initialization/reset, routing, transport, communication, and adaptive hardware feedback remain unmeasured or excluded. The whole-study physical-fault allowance $10^{-3}$ and per-preparation synthesis-model scalar $10^{-5}$ remain separate from sampling confidence. The synthesis scalar is a mean-fit resource allowance, with no concrete angle sequence or output-error certificate {cite:p}`opt_qdk1340,opt_rotation2022`.

Measured CPU controller/reference and replay/search timings remain in the records and are separate from modeled quantum time. These costs replay the fixed protocol, including all quality misses; they are not measured time-to-quality, a new hardware-driven adaptive optimization, or an improvement at matched successful output quality.

**Classical C4 reference coverage only.** The existing Lean result is not a uniform theorem for these graph families, a VQE-chain proof, or a proof of the new circuits, stochastic decisions, simultaneous intervals, or physical models.

(sampled-quality-reproduce-and-inspect)=
### Reproduce and inspect

Use the pinned Python 3.12 environment and original package credits in the [main chapter](../optimization.md). The [recipe](../../assets/generated/optimization/quality-scaling/recipe.json), [all runs](../../assets/generated/optimization/quality-scaling/runs.json), [resource ledger](../../assets/generated/optimization/quality-scaling/resources.json), [objective-call CSV](../../assets/generated/optimization/quality-scaling/evaluations.csv), [group CSV](../../assets/generated/optimization/quality-scaling/groups.csv), and [all final qualities and intervals](../../assets/generated/optimization/quality-scaling/results.csv) expose the complete experiment.

The [circuit manifest](../../assets/generated/optimization/quality-scaling/circuit-manifest.json) maps the exact QASM hashes to cases, parameters, groups, and repetition counts. Standalone [implementation](../../assets/generated/optimization/quality-scaling/optimization.py), [generator](../../assets/generated/optimization/quality-scaling/generate_optimization_quality_scaling.py), and [MIT license](../../assets/generated/optimization/quality-scaling/LICENSE.txt) accompany the [executed verification notebook](../../notebooks/optimization_quality_scaling.ipynb). The notebook loads matching `runs.json` and module snapshots, checks their hashes, and independently replays the recorded decisions and references without repeating training. For standalone execution, save both `runs.json` and `optimization.py` beside the notebook and use the pinned environment.

```bash
python scripts/verify_optimization_checkpoint.py
python -m pytest tests/test_optimization_quality_scaling.py tests/test_optimization_scaling_vqe.py
python scripts/generate_optimization_quality_resources.py --max-queries 96 --max-wall-seconds 1200 --require-complete
```

The resource driver requires the audited shared source-checkout adapter. Its [source](../../assets/generated/optimization/quality-resources/generate_optimization_quality_resources.py) and [MIT license](../../assets/generated/optimization/quality-resources/LICENSE.txt) are available for inspection; it supplies no actual synthesized angle sequences or reset model.

The generator preserves the numerical checkpoint and refuses to overwrite an existing study with a fresh run. `--resume` completes exports from matching snapshots; a new numerical reproduction needs a separate output directory. [The raw checkpoint](../../assets/generated/optimization/quality-scaling/raw-study.json) supports that recorded-workflow replay.

Saved notebook outputs serve the static site without a public kernel. Larger generators, alternative schedules, and any physical adaptive experiment require separate records and validation.

(initialization-and-sampling-diagnostics)=
## Initialization and sampling diagnostics

The [quality-and-workload study](finite-shot.md#quality-and-workload-across-sizes) retained many target misses under its fixed sampling schedule. Those results leave several possible explanations: an unreachable target in the circuit family, an unfavorable initialization, a limited search, or sampled decisions. This diagnostic study fixes two starts and equal training preparation counts for a smaller set of inputs. It uses the same original QAOA and VQE constructions {cite:p}`opt_qaoa2014,opt_vqe2014`; the implementation and figures are original.

:::{note} Study status and interpretation

The frozen numerical recipe has executed all **36 sampled runs** and **24 separately labeled deterministic CPU controls**. Independent post-training audits meet the target in **4 of 24 QAOA runs** and **2 of 12 VQE runs**. Simultaneous holdout intervals support the target in **three QAOA runs** and **no VQE runs**; all 36 runs support the separately requested $\pm0.05$ normalized readout precision. Every prescribed run and miss remains in the record.

Counts are generated classically from ideal circuit probabilities; no QPU or calibrated hardware noise is measured. The earlier 72-run study, its misses, source records, and conditional physical replay remain unchanged.

Equal training preparation counts do not imply equal objective calls, measured qubits, source gates, synthesis, physical qubits, or runtime. These bounded ideal numerical diagnostics cannot establish calibrated hardware-noise behavior, general optimizer convergence, or quantum advantage.
:::

(sampled-diagnostics-fixed-cases-starts-and-targets)=
### Fixed cases, starts, and targets

QAOA uses the unweighted cycles C4 and C6, each at depths $p=1,2$. For $n$ vertices the cycle edges are $(j,j+1)$ for $j=0,\ldots,n-2$, together with $(n-1,0)$. Each run minimizes the negative joint cut mean. The two predetermined starts, in interleaved $(\gamma_0,\beta_0,\ldots)$ order, repeat either $(0.2,0.1)$ or $(0.6,0.4)$ in every layer.

VQE uses open chains of $n=4,6$ qubits with dimensionless Hamiltonian

$$
H_n=-\sum_{j=0}^{n-1}Z_j
+\frac12\sum_{j=0}^{n-2}X_jX_{j+1}.
$$

Prepare $|0\rangle^{\otimes n}$, apply `RY(theta[j])` to each qubit, and then the ascending CX chain $(0,1),(1,2),\ldots,(n-2,n-1)$. The two starts set every angle to $0.6$ or every angle to zero. They are prescribed before sampling; diagnostic witnesses and angles selected by the earlier study never initialize these runs.

The six cases, two starts, and seed labels `7`, `19`, and `41` give 36 sampled runs. Distinct case/start namespaces and separate training/holdout streams distinguish repeated seed labels {cite:p}`opt_seedsequence254`. Every prescribed run and every failed or inconclusive outcome is retained.

The quality targets preserve those of the earlier study:

$$
\text{QAOA: }\frac{\langle C\rangle}{C_\star}\ge0.80,
\qquad
\text{VQE: }\frac{\langle H_n\rangle-E_0}{n}\le0.05.
$$

Classical enumeration supplies $C_\star=n$ for these even cycles, and independent dense diagonalization supplies $E_0$. The references normalize post-training quality; they do not supply sampled optimization decisions. The density tolerance is an absolute energy tolerance of $0.05n$, so it varies with $n$.

(sampled-diagnostics-equal-training-preparations-and-sampled-decisions)=
### Equal training preparations and sampled decisions

Every sampled run has an initial evaluation and then twelve sweeps: four at each coordinate step $0.4$, $0.2$, and $0.1$ radians. The three blocks use $b$, $2b$, and $4b$ shots per group per call. With $D$ parameters and $g$ measurement groups, the fixed allocation is

$$
N_{\rm train}=g\left[s_{\rm initial}+(1+2D)\,28b\right]
=65{,}648.
$$

| Case | Groups $g$ | Parameters $D$ | Initial shots/group | $b$ | Training calls |
|---|---:|---:|---:|---:|---:|
| QAOA $p=1$ | 1 | 2 | 128 | 468 | 61 |
| QAOA $p=2$ | 1 | 4 | 128 | 260 | 109 |
| VQE $n=4$ | 2 | 4 | 64 | 130 | 109 |
| VQE $n=6$ | 2 | 6 | 64 | 90 | 157 |

Each sweep freshly evaluates its incumbent and both displacements along every coordinate from that incumbent. Polling visits coordinates in ascending order, minus then plus. It chooses the smallest estimated proposal objective and accepts only when

$$
\widehat f_{\rm incumbent}-\widehat f_{\rm proposal}
>2\sqrt{\widehat{\mathrm{SE}}_{\rm incumbent}^{\,2}
+\widehat{\mathrm{SE}}_{\rm proposal}^{\,2}}.
$$

Angles wrap to $[-\pi,\pi)$. The schedule is exhausted without an outcome-based stopping rule. Choosing among proposals and adapting the angles makes this two-standard-error margin a descriptive heuristic, not a calibrated hypothesis test or convergence certificate.

For QAOA, each all-Z bitstring jointly scores every cut edge. VQE has an all-Z score $Y_Z=-\sum_jZ_j$ and an all-X score $Y_X=\tfrac12\sum_jX_jX_{j+1}$, after H basis changes on all qubits. Joint bitstring scores retain covariance among terms within each group. The two group means add to the energy, with independent fresh samples between groups.

The sampled optimizer uses only these counts, estimated objectives, and descriptive errors. Exact expectations of selected states are post-training audits. The record must retain every incumbent, proposal, count batch, margin, and rejection, so an observed success cannot hide the effort or failures preceding it.

One obstruction is known independently of the search. For both tested depth-one cycles, conjugating each edge observable through the mixer and cost layer gives mean cut contribution $1/2+\sin(4\beta)\sin(2\gamma)/4$. Summing the $n$ edges gives

$$
\langle C\rangle=\frac n2+\frac n4\sin(4\beta)\sin(2\gamma),
\qquad n\in\{4,6\}.
$$

Since both sine factors are bounded by one, its maximum ratio is $3/4$, attained at $\gamma=\pi/4,\beta=\pi/8$ and below the preserved $0.80$ target. These angles are an explanatory analytic anchor, never a sampled or control start. More shots or a perfect angle optimizer cannot lift that circuit-family ceiling. Deterministic angle checks corroborate the formula rather than prove a universal bound; the bound follows from the expression itself. This circuit limit is separate from an unsuccessful local search, and deterministic controls do not themselves prove infeasibility for other cases.

(sampled-diagnostics-two-deterministic-cpu-controls)=
### Two deterministic CPU controls

For each of the twelve case/start pairs, two deterministic controls follow the same coordinate order, step schedule, and strict acceptance rule. They have no seed replication because no random outcomes are drawn.

| Control | Objective used for decisions | Effective standard error |
|---|---|---|
| Exact objective | Circuit expectation | Zero |
| Expected margin | Circuit expectation | $\sqrt{\sum_g\operatorname{Var}(Y_g)/s_g}$ at the prescribed virtual group allocations |

For the second mode, if $Y_g$ is a group score and $s_g$ is its virtual allocation, the squared effective error is $\sum_g\operatorname{Var}(Y_g)/s_g$. It evaluates the acceptance margin at exact means. It is **not the expectation of the noisy adaptive path**: sampling can change the winning proposal, its estimated error, and every later angle.

Both controls consume **zero quantum preparations**. Their virtual sampling allocations define a CPU diagnostic margin; they are never counted as actual QPU shots or treated as free quantum expectations. Circuit expectations, population moments, and independent references are classically calculated on the bounded inputs. Their CPU times and virtual budgets must remain separate from the sampled workload and physical source metadata.

A difference between starts shows behavior of this prescribed search on these cases. A difference between the exact objective and expected margin controls helps expose the effect of the acceptance rule. A difference between either control and sampled runs includes selection and adaptive-path effects; it does not identify a calibrated hardware-noise law or isolate a unique cause for every miss.

(sampled-diagnostics-fresh-holdout-precision-and-selected-state-quality)=
### Fresh holdout precision and selected-state quality

Every sampled selected state receives a fresh **16,384-shot-per-group** holdout. There are 24 QAOA final groups and 24 VQE final groups, so the family has 48 predetermined groups. Allocate

$$
\delta_{\rm group}=\frac{0.05}{48},\qquad
a=\sqrt{\frac{\log(2/\delta_{\rm group})}{2\cdot16{,}384}}.
$$

Hoeffding's inequality and a union bound provide simultaneous coverage of those 48 group means with failure probability at most $0.05$, conditional on the selected states and training histories under fresh ideal independent samples within each group {cite:p}`opt_hoeffding1963`. The union bound does not require independence between runs. It does require valid fresh conditional sampling for every group.

The conservative QAOA score range $[0,m]$ gives ratio radius $(m/C_\star)a=a$ for these even cycles. The VQE group ranges $[-n,n]$ and $[-(n-1)/2,(n-1)/2]$ give energy-density radius $(3-1/n)a$. These radii assess the separately requested simultaneous $\pm0.05$ expectation precision.

Selected-state quality is a separate statement. A QAOA lower ratio endpoint at least $0.80$ supports the target, while an upper endpoint below $0.80$ supports a miss. A VQE upper density-gap endpoint at most $0.05$ supports the target, while a lower endpoint above $0.05$ supports a miss. An interval crossing the threshold is inconclusive. Raw means and endpoints remain visible, including negative sampled energy-density gaps; shot noise cannot establish an expectation below the ground energy.

The final report therefore distinguishes exact target hits, holdout-supported targets, holdout-supported misses, inconclusive intervals, and expectation precision. Deterministic controls have no sampled holdout confidence claim.

(sampled-diagnostics-recorded-outcomes-and-what-they-distinguish)=
### Recorded outcomes and what they distinguish

```{include} ../../assets/generated/optimization/diagnostics/summary.md
```

![Every prescribed sampled final state and both deterministic controls, by case and start.](../../assets/generated/optimization/diagnostics/quality.svg)

All twelve depth-one QAOA runs miss the target, as required by the analytic circuit ceiling above. At C4 depth two, all three runs from the alternative start reach ratio $0.988199$, with simultaneous holdout support, while none from the original start reaches $0.80$. Both deterministic controls show the same start contrast. This is behavior of these two prescribed starts on one case, not an estimated population success probability.

The C6 depth-two results differ. The original start has one exact target hit, seed `19`, at ratio $0.805210$; its holdout interval crosses $0.80$ and is inconclusive. All three alternative-start sampled runs miss the exact target. For that alternative start, the exact objective control reaches $0.830926$ after eight accepted moves, while the expected margin control reaches $0.794687$ after two. The successful exact control supplies a state in this circuit family above the target, and the margin changes this fixed search path. Neither fact predicts what another sampled trajectory will select.

For four-spin VQE, the zero start gives two sampled exact target hits, seeds `7` and `41`, at density gap $0.037892$; seed `19` ends at $0.050844$ and misses. All three holdout intervals are inconclusive. The zero state's exact density gap is already $0.047100$: its expected margin control accepts no move, while its exact objective control reaches $0.032233$. From the original all-$0.6$ start, the sampled gaps are $0.123778$–$0.229939$, all with holdout-supported misses. Its exact objective control also reaches $0.032233$, but its expected margin control accepts no move and stays at $0.564205$.

Every six-spin sampled run misses the target. The original-start gaps are $0.498964$–$0.596566$, with holdout-supported misses. The zero-start gaps are $0.052580$–$0.055903$, with inconclusive holdouts; two sampled trajectories worsen the zero-state baseline, and seed `41` stays at it. From zero, the exact objective control reaches $0.042669$ and meets the target, while the expected margin control accepts no move and retains the baseline $0.052580$. The exact controls thus exhibit target-reaching states for both VQE inputs without claiming that either ansatz contains its full ground state. The failed original-start six-spin exact control, at gap $0.087448$, is a local-search outcome rather than an infeasibility result.

Across the 36 sampled final states, holdouts support **three targets**, **24 misses**, and leave **nine inconclusive**. All 36 support the separate expectation-precision allocation. The figure below retains accepted-move counts for every sampled seed and both deterministic modes. Accepted moves need not improve the exact objective when their decisions use sampled counts, and a control with no accepted moves still consumes its full CPU evaluation schedule.

![Accepted moves for every case, start, sampled seed, and deterministic control.](../../assets/generated/optimization/diagnostics/margins.svg)

(sampled-diagnostics-complete-workload-and-physical-limits)=
### Complete workload and physical limits

The executed 36-run sampled allocation has **2,363,328 training preparations** and **786,432 holdout preparations**, for **3,149,760 total modeled preparations**. It makes **3,672 sampled objective calls**, including final validation, and **5,280 measured group batches**. These counts include initial calls, fresh incumbents, every rejected and accepted proposal, and final readout; the executed record reconciles with the fixed allocation.

Equal training counts leave important workload differences. A QAOA preparation uses $n$ H, $np$ RX, $mp$ RZ, and $2mp$ CX source gates. A VQE preparation uses $n$ RY and $n-1$ CX; its all-X group adds $n$ H. Each group measures all $n$ qubits. The complete source ledger weights every recorded parameter/basis circuit by its group's shots, including the full trajectory and final readout. The VQE holdout also uses two groups per selected state, so equal training counts do not give equal full readout workloads.

The [executed source ledger](../../assets/generated/optimization/diagnostics/resources.json) records **82,488,288 shot-weighted unitary gates**, **15,748,800 individual qubit readouts**, and peak width six. The [circuit manifest](../../assets/generated/optimization/diagnostics/circuit-manifest.json) maps all 5,280 sampled group batches to 1,147 unique source-QASM exports, parameters, bases, hashes, and shot weights. Deduplicating an export does not remove any repetitions from the ledger.

![Preparation counts and complete source-gate workloads for each prescribed case.](../../assets/generated/optimization/diagnostics/workload.svg)

The input is an explicitly generated graph or the stated local Hamiltonian. QAOA objective phases, initial H or RY preparation, CX chains, VQE basis changes, repetitions, and readout belong to the source workload. Dense references, statevectors, population moments, sampling, and controller wall/CPU times are classical costs recorded separately.

Circuit generation and simulation have separate limits. The source generators cap materialization at 100,000 unitary gates per circuit, and the chain-VQE generator supports 2–32 qubits. The sampled and dense-reference routines cap inputs at eight qubits; this study uses at most six. These four- and six-qubit controls do not remove the exponential cost of generic statevector simulation or dense reference calculations.

The deterministic controls make **2,424 exact CPU objective evaluations**, plus 24 independent final audits, with zero quantum preparations or measured groups. Their virtual training allocations are excluded from the sampled ledger. The [numerical timing record](../../assets/generated/optimization/diagnostics/numerical-timing.json) reports **3.855905 s of CPU-host wall time** and **3.773667 s of process CPU time**, including numerical checks and checkpoint writes. Trace reconstruction, export, and plotting have separate timings in the full run record. None of these times measures quantum hardware or physical feedback.

This study has no new physical projection. Arbitrary source rotations remain unsynthesized. Physical initialization/reset, routing, transport, communication, device latency, and adaptive hardware feedback are unmeasured, and no common physical configuration is verified. The [earlier conditional physical replay](finite-shot.md#sampled-quality-conditional-physical-replay-and-verification-limits) remains a separate frozen record, rather than a physical-cost proxy for this new recipe. A quality miss cannot establish an improvement at matched successful output quality.

(sampled-diagnostics-bounded-verification-and-reproducibility)=
### Bounded verification and reproducibility

The generator reconciles all 36 sampled histories with the fixed starts, schedules, namespaces, group allocations, and recorded decision rule. Counts replay the sampled decisions without exact means, while independent cut enumeration, circuit expectations, dense ground energies, and joint score moments audit final quality and control definitions. All 24 control paths retain their proposals and rejections as well.

The [recipe](../../assets/generated/optimization/diagnostics/recipe.json), [raw numerical checkpoint](../../assets/generated/optimization/diagnostics/raw-study.json), [full runs](../../assets/generated/optimization/diagnostics/runs.json), [sampled objectives](../../assets/generated/optimization/diagnostics/evaluations.csv), [measured groups](../../assets/generated/optimization/diagnostics/groups.csv), and [final qualities and intervals](../../assets/generated/optimization/diagnostics/results.csv) expose the sampled calculation. The [control results](../../assets/generated/optimization/diagnostics/controls.csv) and [every control evaluation](../../assets/generated/optimization/diagnostics/control-evaluations.csv) keep exact means, population variances, virtual allocations, and effective decision margins visible.

Matching standalone [implementation](../../assets/generated/optimization/diagnostics/optimization.py), [generator](../../assets/generated/optimization/diagnostics/generate_optimization_diagnostics.py), [environment lock](../../assets/generated/optimization/diagnostics/requirements-lock.txt), and [MIT license](../../assets/generated/optimization/diagnostics/LICENSE.txt) accompany the [executed verification notebook](../../notebooks/optimization_diagnostics.ipynb). The notebook checks matching hashes and replays saved sampled decisions, controls, references, holdouts, and workloads without repeating training. For standalone execution, copy the complete diagnostics artifact directory, place the notebook inside it, and run there using the pinned environment. The audit requires the matching implementation and generator snapshots, `runs.json`, `resources.json`, `circuit-manifest.json`, and its `circuits/` QASM files.

From the source checkout:

```bash
python -m pytest tests/test_optimization_diagnostics.py
python scripts/generate_optimization_diagnostics.py --resume
```

The generator's `--resume` mode preserves the frozen numerical checkpoint and matching producer bytes while completing verification or exports. It refuses a changed recipe or producer. Standalone resume additionally requires the complete saved artifact directory, including `freeze.json`, `recipe.json`, `raw-study.json`, `numerical-timing.json`, every file under `checkpoints/`, the license, and the environment lock. From that copied directory, run `python generate_optimization_diagnostics.py --output-dir . --resume`. A fresh numerical reproduction requires a separate output directory, a preserved recipe/producer freeze, and the same bounded execution checks.

Use the pinned Python 3.12 environment and exact package credits in the [main chapter](../optimization.md). Saved notebook outputs serve the static site without a public kernel. Larger inputs, alternative starts or schedules, calibrated hardware sampling, and physical adaptive execution require new records and validation.

**Classical C4 reference coverage only.** The existing Lean artifact does not prove these QAOA circuits, VQE ansätze, stochastic searches, deterministic controls, simultaneous intervals, or resource models.

## Verification and formalization

The saved audits check explicit source and lowered circuits, original measurement maps, recorded sampled decisions, fresh holdout allocations and complete repetition ledgers. The three recipes retain different inputs, schedules and confidence families. Their simultaneous selected-state intervals do not certify training convergence, a fresh adaptive hardware path or calibrated physical error.

**Classical C4 reference coverage only.** The existing reproduced Lean artifact covers finite cut values and the stated classical equality predicate. It does not prove the QAOA or VQE circuits, stochastic or deterministic search paths, simultaneous intervals, rotation synthesis or physical resource models. The [proof source](../../assets/generated/optimization/formal/C4Reference.lean) and [verification record](../../assets/generated/optimization/formal/verification.json) retain that exact coverage.

## References

```{bibliography}
```
