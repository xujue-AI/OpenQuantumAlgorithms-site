This portability check applies the same bounded sampled optimizer to two new toy inputs: the three-vertex triangle MaxCut graph with all three edges, and the Hamiltonian H = -0.8 Z0 - 0.6 Z1 + 0.4 X0X1 with the same RY/CX ansatz. All eight classical cut assignments give optimum 2; full matrix diagonalization gives ground energy -sqrt(2.12). Changed problem inputs prevent a same-task resource comparison with earlier reports.

These runs optimize **sampled** objectives. Ideal measurement outcomes are generated from statevector probabilities; exact expectations are used only for post-decision diagnostics. They are modeled ideal measurements, rather than executed hardware shots.

The bounded schedule contains four blocks of eight sweeps: (step radians, shots/group) = (0.4, 256), (0.2, 512), (0.1, 1024), (0.05, 2048). The initial objective uses 256 shots/group. Each sweep freshly measures its incumbent and both coordinate proposals for every parameter, including all rejected proposals. A candidate is accepted only when the sampled decrease exceeds twice the combined estimated standard error; this margin is a heuristic, without a confidence or convergence guarantee across adaptive choices. Every run stops after the prescribed 32 sweeps.

Seeds 7, 19 and 41 use the same fixed initialization for each task: QAOA (0.2, 0.1), VQE (0.6). Only the seeded measurement randomness changes within each task. Three observed runs do not certify a population success probability. No target-based retuning is performed. Fresh validation uses 4,096 shots per group and a separate keyed random-stream family after training ends.

| Task / seed | Initial angles | Final angles (radians) | Training calls / shots | Fresh validation shots | Total modeled shots | Training optimizer wall (s) | Classical control (s) |
|---|---|---|---:|---:|---:|---:|---:|
| QAOA / 7 | 0.2, 0.1 | 0.60000000, 0.30000000 | 161 / 153856 | 4096 | 157952 | 0.144288 | 0.003262 |
| QAOA / 19 | 0.2, 0.1 | 0.60000000, 0.30000000 | 161 / 153856 | 4096 | 157952 | 0.142378 | 0.003305 |
| QAOA / 41 | 0.2, 0.1 | 0.60000000, 0.30000000 | 161 / 153856 | 4096 | 157952 | 0.168337 | 0.003369 |
| VQE / 7 | 0.6 | -0.25000000 | 97 / 184832 | 8192 | 193024 | 0.082099 | 0.001860 |
| VQE / 19 | 0.6 | -0.30000000 | 97 / 184832 | 8192 | 193024 | 0.082355 | 0.001808 |
| VQE / 41 | 0.6 | -0.30000000 | 97 / 184832 | 8192 | 193024 | 0.082154 | 0.001691 |

Fresh validation estimates the energy or expected cut of the final, already selected state. The estimated standard error includes within-group covariance; it is not a rigorous interval. A zero empirical standard error can occur when a finite sample contains only one score and does not certify the exact mean. The separate Hoeffding radius is an ideal-sampling 95% bound for each validation row, not simultaneous table coverage.

| Task / seed | Post-run exact diagnostic | Fresh validation mean | Error from diagnostic | Estimated SE | Hoeffding 95% radius | Post-run toy target met | ±0.10 readout target supported |
|---|---:|---:|---:|---:|---:|---|---|
| QAOA / 7 | 1.999052281 | 1.99951172 | +0.000459 | 0.000488 | 0.042441 | True | True |
| QAOA / 19 | 1.999052281 | 2.00000000 | +0.000948 | 0.000000 | 0.042441 | True | True |
| QAOA / 41 | 1.999052281 | 1.99902344 | -0.000029 | 0.000690 | 0.042441 | True | True |
| VQE / 7 | -1.455438974 | -1.44042969 | +0.015009 | 0.008224 | 0.061795 | True | True |
| VQE / 19 | -1.455679167 | -1.45683594 | -0.001157 | 0.008810 | 0.061795 | True | True |
| VQE / 41 | -1.455679167 | -1.45253906 | +0.003140 | 0.008870 | 0.061795 | True | True |

The post-run numerical targets are exact expected cut at least 1.90 for QAOA and exact energy gap at most 0.03 for VQE. These diagnostics assess the selected toy states after training; target flags are not guarantees derived from the validation confidence bound. Separately, the fresh validation has an absolute expectation-error target of 0.10 at 95% confidence for one fixed estimate. The displayed Hoeffding radii determine whether that readout allocation supports this precision target.

QAOA seed 7: fresh validation observed best cut 2 (classical optimum 2); the sampled optimal-cut fraction was 0.999756. The fixed final state's exact optimal-cut probability, audited after training, was 0.999526.

QAOA seed 19: fresh validation observed best cut 2 (classical optimum 2); the sampled optimal-cut fraction was 1.000000. The fixed final state's exact optimal-cut probability, audited after training, was 0.999526.

QAOA seed 41: fresh validation observed best cut 2 (classical optimum 2); the sampled optimal-cut fraction was 0.999512. The fixed final state's exact optimal-cut probability, audited after training, was 0.999526.

The complete record contains **780 sampled objective calls** including six fresh validations and **1,052,928 modeled circuit shots**. The full CPU workflow took 0.709759 s. Simulator/probability generation and classical search work are software timings rather than quantum runtime. Logical counts repeat every preparation, measurement basis change, and readout for all training and validation shots. Arbitrary rotations remain unsynthesized. Physical reset/measurement timing, routing, faults, error correction, factories, hardware latency, noise and mitigation are excluded; physical resource estimates remain pending.
