These are **conditional mean-fit resource projections for the first predeclared case's fixed final readout**. They exclude all finite-shot optimizer training calls and are not QPU measurements, certified angle syntheses, worst-case bounds, convergence guarantees, or guaranteed observable accuracy.

The profiles use the frozen physical assumptions: error rates $10^{-3}$ and $10^{-4}$, 100 ns gate time and 500 ns measurement time. Each circuit has 4,096 modeled serial executions. The QAOA task receives physical-fault allowance $10^{-3}$; the VQE task receives $10^{-3}$ split equally between its Z and X groups. Each circuit separately receives synthesis-model scalar allowance $10^{-5}$ per execution.

| Task | Physical error | Declared group selection | Modeled preparations | Peak physical qubits | Serial readout time (ms) | Physical-fault score, all groups |
|---|---:|---|---:|---:|---:|---:|
| QAOA | 0.001 | fewest qubits per group | 4096 | 485,615 | 9420.800000 | 0.000289 |
| QAOA | 0.001 | shortest runtime per group | 4096 | 767,335 | 6406.144000 | 0.000459 |
| VQE | 0.001 | fewest qubits per group | 8192 | 377,873 | 2367.488000 | 0.000321 |
| VQE | 0.001 | shortest runtime per group | 8192 | 445,867 | 2088.960000 | 0.000659 |
| QAOA | 0.0001 | fewest qubits per group | 4096 | 38,975 | 7913.472000 | 2.95e-05 |
| QAOA | 0.0001 | shortest runtime per group | 4096 | 63,595 | 3391.488000 | 4.65e-05 |
| VQE | 0.0001 | fewest qubits per group | 8192 | 14,893 | 2367.488000 | 0.000248 |
| VQE | 0.0001 | shortest runtime per group | 8192 | 28,033 | 974.848000 | 0.000624 |

These two declared choices select each group's point separately; they do not claim a full combined Pareto frontier. Serial group time is summed and peak physical qubits use the maximum, rather than adding independent peaks. This assumes reuse or reconfiguration between independent group layouts/factories; configuration-change and communication overhead are excluded, and a common joint hardware layout is not certified.

The separate ideal fixed-state 95% Hoeffding radii are 0.084881 for the mean cut and 0.087494 for energy. The VQE bound uses two independent bounded groups and preserves Z-term covariance. No certified confidence combining sampling, synthesis and physical faults is reported.

The full JSON records all six circuit/profile reports, every returned per-variant Pareto point, bound circuit/QASM hashes, source hashes, raw/synthesized QDK traces, model budgets and limitations. The full adaptive optimizer physical layer remains **pending**.
