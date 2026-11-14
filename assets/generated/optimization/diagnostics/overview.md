All 36 fixed sampled trajectories and 24 deterministic CPU controls are retained. Every sampled trajectory uses 65,648 training preparations; exact controls use zero actual quantum preparations.

| Method | Sampled runs | Post-run exact target hits | Whole-study holdout supports target | CPU control target hits (exact / expected-margin) | Actual preparations |
|---|---:|---:|---:|---:|---:|
| QAOA | 24 | 4 | 3 | 2/8 / 1/8 | 1,968,768 |
| VQE | 12 | 2 | 0 | 3/4 / 1/4 | 1,180,992 |

Complete actual trace: 3,672 sampled calls, 5,280 measured groups and 3,149,760 ideal preparations. Matched preparations leave different gate workloads; these bounded observations do not establish general convergence or quantum advantage.
