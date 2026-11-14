These runs optimize **sampled** objectives. Ideal measurement outcomes are generated from statevector probabilities; exact expectations are used only for post-decision diagnostics. They are modeled ideal measurements, rather than executed hardware shots.

The bounded schedule contains four blocks of eight sweeps: (step radians, shots/group) = (0.4, 256), (0.2, 512), (0.1, 1024), (0.05, 2048). The initial objective uses 256 shots/group. Each sweep freshly measures its incumbent and both coordinate proposals for every parameter, including all rejected proposals. A candidate is accepted only when the sampled decrease exceeds twice the combined estimated standard error; this margin is a heuristic, without a confidence or convergence guarantee across adaptive choices. Every run stops after the prescribed 32 sweeps.

Each seed is paired with a different fixed initialization. These six cases jointly vary initialization and measurement randomness; they do not isolate the two effects. Fresh validation uses 4,096 shots per group and a separate keyed random-stream family after training ends.

| Task / seed | Initial angles | Final angles (radians) | Training calls / shots | Fresh validation shots | Total modeled shots | Training optimizer wall (s) | Classical control (s) |
|---|---|---|---:|---:|---:|---:|---:|
| QAOA / 7 | 0.2, 0.1 | 0.80000000, 0.40000000 | 161 / 153856 | 4096 | 157952 | 0.299646 | 0.004565 |
| QAOA / 19 | 0.6, 0.4 | 0.70000000, 0.40000000 | 161 / 153856 | 4096 | 157952 | 0.297557 | 0.004675 |
| QAOA / 41 | -0.3, 0.25 | -0.70000000, -0.40000000 | 161 / 153856 | 4096 | 157952 | 0.332488 | 0.004605 |
| VQE / 7 | 0.6 | -0.20000000 | 97 / 184832 | 8192 | 193024 | 0.132981 | 0.002202 |
| VQE / 19 | -1.1 | -0.20000000 | 97 / 184832 | 8192 | 193024 | 0.133128 | 0.002263 |
| VQE / 41 | 2.2 | -0.20000000 | 97 / 184832 | 8192 | 193024 | 0.132287 | 0.002226 |

Fresh validation estimates the energy or expected cut of the final, already selected state. The estimated standard error includes within-group covariance; it is not a rigorous interval. The separate Hoeffding radius is an ideal-sampling 95% bound for each validation row, not simultaneous table coverage.

| Task / seed | Post-run exact diagnostic | Fresh validation mean | Error from diagnostic | Estimated SE | Hoeffding 95% radius | Post-run toy target met |
|---|---:|---:|---:|---:|---:|---:|
| QAOA / 7 | 2.999147388 | 2.99511719 | -0.004030 | 0.017648 | 0.084881 | True |
| QAOA / 19 | 2.985029537 | 2.95605469 | -0.028975 | 0.016931 | 0.084881 | True |
| QAOA / 41 | 2.985029537 | 3.00439453 | +0.019365 | 0.016818 | 0.084881 | True |
| VQE / 7 | -2.059467821 | -2.05322266 | +0.006245 | 0.010104 | 0.087494 | True |
| VQE / 19 | -2.059467821 | -2.07568359 | -0.016216 | 0.009267 | 0.087494 | True |
| VQE / 41 | -2.059467821 | -2.06030273 | -0.000835 | 0.009820 | 0.087494 | True |

The post-run numerical targets are exact expected cut at least 2.90 for QAOA and exact energy gap at most 0.03 for VQE. These diagnostics assess the selected toy states after training; target flags are not guarantees derived from the validation confidence bound.

QAOA seed 7: fresh validation observed best cut 4 (classical optimum 4); the sampled optimal-cut fraction was 0.531982. The fixed final state's exact optimal-cut probability, audited after training, was 0.534524.

QAOA seed 19: fresh validation observed best cut 4 (classical optimum 4); the sampled optimal-cut fraction was 0.500000. The fixed final state's exact optimal-cut probability, audited after training, was 0.515878.

QAOA seed 41: fresh validation observed best cut 4 (classical optimum 4); the sampled optimal-cut fraction was 0.521973. The fixed final state's exact optimal-cut probability, audited after training, was 0.515878.

The complete record contains **780 sampled objective calls** including six fresh validations and **1,052,928 modeled circuit shots**. The full CPU workflow took 1.342414 s. Simulator/probability generation and classical search work are software timings rather than quantum runtime. Logical counts repeat every preparation, measurement basis change, and readout for all training and validation shots. Arbitrary rotations remain unsynthesized. Physical reset/measurement timing, routing, faults, error correction, factories, hardware latency, noise and mitigation are excluded; physical resource estimates remain pending.
