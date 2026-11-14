This is a **conditional resource-model replay of the entire fixed 72-run study**. All 5,859,840 recorded preparations are counted, including rejected proposals and final holdouts. The recorded angles are replayed; a fresh noisy or synthesized hardware optimizer could choose another path.

Physical status: **computed_conditional_recorded_replay**. Physical-estimator search limit: 96; searches actually requested: 85. Complete cost classes: 84. The separate trace-signature audit count is 3,746; trace audits and internal adapter/native calls are not represented by the physical-search cap.

The whole-study physical-fault allowance is $10^{-3}$, uniformly allocated over every preparation. The separate per-preparation $10^{-5}$ synthesis-model scalar is a mean-fit cost allowance, not a stochastic fault or concrete angle certificate.

| Physical error | Independently chosen class configuration | Preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-study physical-fault score |
|---:|---|---:|---:|---:|---:|
| 0.001 | fewest qubits per class | 5,859,840 | 4,080,140 | 16640.261632 | 0.000692 |
| 0.001 | shortest runtime per class | 5,859,840 | 7,670,740 | 15694.630656 | 0.000789 |
| 0.0001 | fewest qubits per class | 5,859,840 | 420,725 | 16631.084288 | 0.000489 |
| 0.0001 | shortest runtime per class | 5,859,840 | 1,149,445 | 7584.243200 | 0.000612 |

The peak is the maximum selected footprint and serial time is the weighted sum. Class code distances/factories are independently selected; a common physical configuration is **not verified**. Layout changes, physical initialization/reset, routing, transport and adaptive feedback are unmeasured and excluded. The native LogicalCounts/PSSPC abstraction does not preserve original circuit topology.

All source/lowered prepared states and readout maps are checked up to global phase within eight qubits, with independent algebraic/dense references. Full eight-qubit Operators are not allocated; this numerical scope is not a circuit theorem. Cost canonicalization removes only audited EVALUATION_TIME. One distinct-QASM empirical cost probe uses exact cost signatures and relative float tolerance $10^{-12}$ with zero absolute tolerance, not unitary equivalence.

Measured local CPU controller/reference timings remain separate from modeled quantum time. Actual synthesis sequences, reset models, hardware feedback and general optimizer/quality guarantees remain outside this conditional report. Missed-quality runs cannot establish a matched-quality improvement.
