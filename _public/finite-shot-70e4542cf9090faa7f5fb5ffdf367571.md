---
title: "Finite-shot optimization and independent readout"
description: "Sampled QAOA and VQE parameter trajectories, complete shot records, fresh holdout estimates, and conditional fixed-readout resources."
---

The [first optimization examples](../optimization.md) fit angles using exact statevector expectations and sample only the final readout. This increment instead makes each parameter decision from finite outcomes. It uses the same four-vertex MaxCut cycle, the same two-qubit Hamiltonian $H=-Z_0-Z_1+\tfrac12X_0X_1$, and the same explicit QAOA and VQE circuits {cite:p}`opt_qaoa2014,opt_vqe2014`.

:::{note} Scope of the experiment

These are reproducible ideal finite-shot simulations. Counts are drawn classically from explicit circuit statevector probabilities; they are not hardware measurements. Training uses the sampled objectives and their descriptive standard errors. Exact diagnostics and fresh final readout are reported separately and do not choose the training moves.

The finite schedule and acceptance rule are heuristics. The observed trajectories do not certify convergence, optimal angles, or a success probability for the optimizer. Physical projections below concern the fixed final readout only. The [complete recorded-trajectory replay](trajectory-resources.md) maps training and holdout batches as fixed circuit inputs. Predicting the cost of a new hardware-driven adaptive path remains **planned**.
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

## The entire sampled workload

All proposals and all incumbent remeasurements count, including rejected moves and the initial call. With the default schedule, one QAOA run has 161 objective calls and 153,856 training shots. One VQE run has 97 objective calls, each with two groups, and 184,832 training shots. These totals follow from the executed trace, rather than from the number of accepted updates.

| Workload for the three predetermined runs | QAOA shots | VQE shots | Total shots |
|---|---:|---:|---:|
| Training | 461,568 | 554,496 | 1,016,064 |
| Fresh final holdout | 12,288 | 24,576 | 36,864 |
| Training plus holdout | 473,856 | 579,072 | 1,052,928 |

Each shot represents a new state preparation in the ideal circuit protocol. The implementation draws counts classically from circuit probabilities, so these are protocol shot counts rather than physical circuit executions. The [sampled resource ledger](../../assets/generated/optimization/sampled-resources.json) weights the actual circuit gates by every call's shot count, separates training from validation, includes basis changes and qubit readout, and records unsynthesized rotations. A single final ansatz evaluation is not this hybrid run.

CPU timings measure the local simulator, random sampling, and classical control. They do not measure a quantum device's reset, routing, gate, measurement, communication, or feedback latency. Input parsing and separate classical reference timing are disclosed where unmeasured. The exact diagnostics and figure-generation work are additional classical checks rather than hidden quantum objective evaluations. The [recorded-trajectory replay](trajectory-resources.md) accounts for this fixed training record; it does not predict a fresh hardware-driven adaptive execution.

## Conditional resources for a fixed final readout

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

## References

```{bibliography}
```
