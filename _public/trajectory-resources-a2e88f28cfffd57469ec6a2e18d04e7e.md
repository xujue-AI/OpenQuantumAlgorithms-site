---
title: "Resources for a recorded optimization trajectory"
description: "Complete circuit-batch mapping and conditional physical costs for replaying the original sampled QAOA and VQE records."
---

The [fixed final-readout projection](finite-shot.md#conditional-resources-for-a-fixed-final-readout) omits sampled training. This report instead maps every measured group in the original four-vertex-cycle QAOA and first two-qubit VQE records, including rejected proposals, incumbent remeasurements, initial calls, and fresh final validation. It treats the stored angles and shot allocations as fixed inputs to a **recorded-trajectory replay**.

:::{note} What is estimated

Replay means preparing the recorded circuits with their recorded repetitions. Fresh hardware measurements are not allowed to choose different parameters in this cost calculation. A hardware-driven optimizer could follow a different angle path, with different synthesis costs and achieved quality, even when its call and shot schedule is fixed.

The circuit mapping and conditional estimator results have separate status in the generated summary below. A checked mapping is not an executed fault-tolerant run. Physical results use an expected synthesis-cost model; they certify neither concrete synthesized sequences nor worst-case work, runtime tails, convergence, or output-error confidence.
:::

## Count the entire record

The original experiment contains three predetermined seed/start pairs for each task. Its 32-sweep sampled schedule and holdout are documented in the [finite-shot chapter](finite-shot.md). Here the workload is the complete collection of those recorded runs, with a separate resource budget for each task.

| Task | Training calls | Holdout calls | All objective calls | Measured group batches | Modeled shots |
|---|---:|---:|---:|---:|---:|
| Four-vertex QAOA | 483 | 3 | 486 | 486 | 473,856 |
| First two-qubit VQE | 291 | 3 | 294 | 588 | 579,072 |
| Complete collection | 774 | 6 | 780 | 1,074 | 1,052,928 |

A VQE objective call contains two groups, all-Z and all-X. QAOA contains one all-Z group. Batch repetition weights retain the exact per-call shot allocation rather than multiplying one final circuit by an optimizer iteration count. The ideal protocol repeats input-state preparation and all basis changes for each shot. Samples in the source experiment were drawn classically from statevector probabilities; these totals are not measured QPU consumption.

```{include} ../../assets/generated/optimization/trajectory-resources/summary.md
```

## From every sampled batch to an explicit circuit

The [manifest](../../assets/generated/optimization/trajectory-resources/manifest.json) binds the replay to the source run record, generator, environment, and implementation. The [execution mapping](../../assets/generated/optimization/trajectory-resources/execution-mapping.csv) identifies the task, run, seed, training or validation phase, call, measurement group, parameters, and shot repetitions for every batch.

Each batch has source and lowered measured OpenQASM files identified by SHA-256. QAOA includes H preparation, edge phases, and mixer rotations. VQE includes its RY/CX preparation and the all-X H basis changes where required. Final qubit-to-classical measurement maps are retained. Initial all-zero availability is an assumption; physical reset time is excluded where unmeasured.

For every batch, the small-system check compares source and lowered unitary operators up to a global phase before final measurement, and verifies the final measurement map separately. That check validates numerical lowering of each particular circuit. It does not validate an approximate Clifford+T replacement of its arbitrary angles.

Identical QASM content can share a stored file by its exact hash. Every original batch still has an execution-mapping entry and its own repetition weight. Exact file deduplication therefore reduces artifact duplication without removing objective evaluations or shots.

## Share estimator work only when the native cost model agrees

Different fitted angles generally produce different unitaries. Resource-model grouping makes no claim that those unitaries are equivalent. Equal gate counts alone also do not establish equal cost: native trace ordering, repetitions, properties, and resource requirements can affect the estimator.

The [model-signature record](../../assets/generated/optimization/trajectory-resources/model-signatures.json) groups batches after comparing cost-canonical native and PSSPC trace JSON, with the recorded derived metadata. Complete instrumented trace JSON is retained. The cost signature excludes only native `EVALUATION_TIME` instrumentation, property 7, whose recorded pinned-source audit establishes that it is not consumed by the physical cost calculation. All other fields and properties remain in the comparison. This CPU instrumentation time is not quantum runtime.

The grouping uses the same synthesis setting, physical profile, and per-execution budget. Its equality is equality of inputs relevant to this pinned **cost model**, rather than equality of circuit behavior at different angles. A bounded second-QASM probe for a shared model class is recorded separately, comparing actual physical Pareto fields under the same budget and profile. Its scope and result are retained rather than treated as a general circuit-equivalence proof.

QDK's default OpenQASM importer constructs a canonical trace from logical counts, already abstracting continuous angles and original Clifford topology. Comparing those native traces checks equality of inputs inside the declared estimator model; it does not certify a routed schedule for every original gate. The source and lowered QASM, gate counts, measurement maps, and numerical lowering checks preserve the original circuit evidence separately.

Each group retains a representative input and maps all constituent batches to the corresponding estimator report. Their repetition weights are summed only after the model-signature comparison. The [projection reports](../../assets/generated/optimization/trajectory-resources/projection-reports.json) retain the actual estimator outputs and model provenance. The distinction between source-circuit equivalence checks and cost-model grouping is essential: neither substitutes for the other.

## Keep physical faults and rotation approximation separate

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

## Aggregate serial time and peak footprint with explicit exclusions

For a declared model-point selection, serial quantum time is weighted by every batch's shot count and peak footprint is the maximum selected footprint:

$$
T_{\mathrm{serial}}=\sum_b N_b\,t_b,\qquad
Q_{\mathrm{peak}}=\max_b q_b.
$$

The declared choices can select independent per-model-signature Pareto points. Configuration changes, transport, communication, routing, physical reset, and feedback overhead are excluded where unmeasured. A single common physical layout or factory configuration for the complete trajectory is not certified. Thus these aggregate choices are conditional serial model costs, rather than a measured machine schedule or a globally optimized architecture.

The local CPU-control and objective-simulation timings from the original experiment are reported separately. A simulator's control time is not a measurement of classical feedback latency in a fault-tolerant processor, and it is not silently added to or hidden inside quantum runtime. Concurrent execution, reuse across shots, and latency overlap require separate scheduling evidence.

## Reproduce and inspect

Reproduction requires the original sampled run record, its matching implementation, and a source checkout containing the shared `src/openquantum/resource_estimation.py` adapter in the pinned Python 3.12/QDK environment. The manifest and signature records identify the executed code, declared model settings, and input hashes. Standalone sampled experiment downloads remain available through the [finite-shot chapter](finite-shot.md); exporting the shared physical adapter is a separate integration step.

The recorded replay concerns only the original four-vertex cycle and first Hamiltonian. The [portability cases](portability.md) are different inputs with their own logical records, and do not inherit these physical estimates.

- [Replay manifest and execution contract](../../assets/generated/optimization/trajectory-resources/manifest.json)
- [Every batch and its source/lowered QASM mapping](../../assets/generated/optimization/trajectory-resources/execution-mapping.csv)
- [Native canonical/PSSPC model signatures](../../assets/generated/optimization/trajectory-resources/model-signatures.json)
- [Actual conditional estimator reports](../../assets/generated/optimization/trajectory-resources/projection-reports.json)

## References

```{bibliography}
```
