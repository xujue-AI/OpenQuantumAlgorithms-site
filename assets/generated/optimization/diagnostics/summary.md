The frozen study retains all **36 sampled trajectories**: C4/C6 QAOA at p = 1, 2 and open-chain VQE at n = 4, 6, with two fixed starts and seeds 7, 19, 41. No fifth-increment winning angles or analytic witnesses initialize a run. Each sampled run has exactly **65,648 training preparations**, then 16,384 fresh shots per measurement group. All accepted and rejected candidates remain in the trace.

The **24 deterministic CPU controls** use the same 12 case/start combinations and fixed polling schedule. Exact-objective controls set the comparison standard error to zero. Expected-margin controls use exact means and true population variances divided by the prescribed virtual group allocation. These controls contain no random counts and use **zero actual quantum preparations**; their allocations are prospective bookkeeping.

| Case | Start | Sampled exact quality range (3 seeds) | Exact target hits | Holdout supports / misses / inconclusive | Exact-objective CPU quality | Expected-margin CPU quality |
|---|---|---|---:|---|---:|---:|
| QAOA C4, p=1 | original | 0.732910–0.749787 | 0/3 | 0 / 3 / 0 | 0.749787 | 0.732910 |
| QAOA C4, p=1 | alternate | 0.732910–0.749787 | 0/3 | 0 / 3 / 0 | 0.749787 | 0.732910 |
| QAOA C4, p=2 | original | 0.762274–0.769969 | 0/3 | 0 / 3 / 0 | 0.787504 | 0.762274 |
| QAOA C4, p=2 | alternate | 0.988199–0.988199 | 3/3 | 3 / 0 / 0 | 0.997847 | 0.988199 |
| QAOA C6, p=1 | original | 0.746257–0.749787 | 0/3 | 0 / 3 / 0 | 0.749787 | 0.746257 |
| QAOA C6, p=1 | alternate | 0.749787–0.749787 | 0/3 | 0 / 3 / 0 | 0.749787 | 0.749787 |
| QAOA C6, p=2 | original | 0.756137–0.805210 | 1/3 | 0 / 2 / 1 | 0.756137 | 0.756137 |
| QAOA C6, p=2 | alternate | 0.749787–0.787033 | 0/3 | 0 / 1 / 2 | 0.830926 | 0.794687 |
| VQE chain n=4 | original | 0.123778–0.229939 | 0/3 | 0 / 3 / 0 | 0.032233 | 0.564205 |
| VQE chain n=4 | alternate | 0.037892–0.050844 | 2/3 | 0 / 0 / 3 | 0.032233 | 0.047100 |
| VQE chain n=6 | original | 0.498964–0.596566 | 0/3 | 0 / 3 / 0 | 0.087448 | 0.667283 |
| VQE chain n=6 | alternate | 0.052580–0.055903 | 0/3 | 0 / 0 / 3 | 0.042669 | 0.052580 |

QAOA quality means mean cut divided by the enumerated optimum (target ≥0.80). VQE quality means energy-density gap above the dense ground energy (target ≤0.05). Fresh holdout bounds allocate total failure 0.05 across all 48 final groups, conditional on fixed trained parameters. The accepted-move margin is a heuristic; these finite controls do not certify the global ansatz optimum, population optimizer-success probability or convergence.

Actual ideal-sampled workload: **3,672 objective calls**, **5,280 measured groups**, **3,149,760 preparations**. The deterministic controls record **2,424 exact CPU oracle evaluations** with zero actual measured groups/preparations. Counts are simulated from ideal probabilities; no QPU or hardware noise was measured. Actual source costs include **82,488,288 shot-weighted unitary gates** and **15,748,800 qubit readouts**.

Input/state preparation gates, measurement-basis changes, every proposed circuit and all final readout shots are included in the actual source/readout ledger. Equal preparation budgets do not imply equal gate or physical cost. Rotations remain unsynthesized; physical initialization/reset, routing, communication and feedback time are unmeasured, and no physical projection or hardware advantage is inferred. All CPU timings, exact-QASM exports and producer snapshots are recorded separately. Existing Lean coverage remains classical C4 reference coverage only.
