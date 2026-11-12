This is a **conditional resource-model replay of the recorded ideal finite-shot trajectory**. It costs all training, rejected proposals, incumbent remeasurements, and independent validation. It does not predict that a synthesized or noisy hardware optimizer would follow the same path or return the same outcomes.

The validated ledger contains 780 calls, 1,074 measured-group batches and 1,052,928 modeled preparations. Every batch has a full small-circuit Operator check up to global phase and an unchanged final measurement map. Only then are 127 exact source QASM and 127 exact lowered QASM artifacts deduplicated.

Resource-query reuse is separate from exact-QASM deduplication. Cost-canonical raw and transformed PSSPC Trace JSON must match exactly in ordered instructions, repetitions, qubits, base error, resource states and all cost properties under the same profile and budgets. Only the documented QDK `EVALUATION_TIME` CPU-import instrumentation is omitted from the cost key; its complete original JSON is retained. Different-angle unitaries are not declared equal.

The distinct-QASM physical-cost probe retains every fingerprint field. Only positive finite factory error-rate floats may agree within four adjacent binary64 values; all other values, types and structures remain exact. Both full fingerprints and every differing float path, raw value and ULP distance are preserved. This narrowly bounded numeric agreement is not a synthesis or accuracy certificate.

There are 8 verified cost signatures and 9 bounded real QDK queries, including a second distinct-QASM probe of one shared cost signature. The task physical-fault allowance is $10^{-3}$ uniformly across all preparations: QAOA has 473,856 and VQE 579,072 preparations. Each execution separately receives synthesis-model scalar allowance $10^{-5}$; that coherent model scalar is not summed over shots as a stochastic fault probability.

| Task | Physical error | Declared serial choice | All modeled preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-task physical-fault score |
|---|---:|---|---:|---:|---:|---:|
| QAOA | 0.001 | fewest qubits per cost signature | 473,856 | 1,131,935 | 1088.640000 | 0.000154 |
| QAOA | 0.001 | shortest runtime per cost signature | 473,856 | 1,328,815 | 914.457600 | 0.000348 |
| VQE | 0.001 | fewest qubits per cost signature | 579,072 | 671,529 | 201.436672 | 0.000205 |
| VQE | 0.001 | shortest runtime per cost signature | 579,072 | 781,123 | 182.305280 | 0.000438 |
| QAOA | 0.0001 | fewest qubits per cost signature | 473,856 | 131,595 | 1088.640000 | 3.76e-05 |
| QAOA | 0.0001 | shortest runtime per cost signature | 473,856 | 270,375 | 479.001600 | 5.55e-05 |
| VQE | 0.0001 | fewest qubits per cost signature | 579,072 | 24,029 | 201.174528 | 0.000444 |
| VQE | 0.0001 | shortest runtime per cost signature | 579,072 | 46,529 | 86.386176 | 0.000703 |

Profiles retain physical error $10^{-3}$ or $10^{-4}$, 100 ns gates and 500 ns measurements. Times sum per-preparation model runtimes; peaks take the maximum across independently selected signature layouts/factories. Configuration changes, transport, resets and adaptive feedback are unmeasured, and no common physical configuration or joint Pareto frontier is certified.

The recorded CPU controller/evaluator times are supplied separately in the manifest; they are not added to quantum-model runtime. The PSSPC rotation costs are a mean-fit projection, not concrete certified synthesis sequences, worst-case bounds, runtime-tail guarantees, or observable/optimizer accuracy guarantees. Statistical uncertainty, coherent synthesis-model allowance and stochastic physical-fault allocations remain distinct.
