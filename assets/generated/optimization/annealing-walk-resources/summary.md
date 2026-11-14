These are conditional surface-code/factory projections for two **fixed measured circuits**, each repeated 12,288 times across the three predeclared 4,096-shot readouts. Annealing time/step studies are classical simulation checks with zero quantum-shot workload; the evolution parameter $T=16$ is not hardware wall time.

The frozen profiles use physical error $10^{-3}$ or $10^{-4}$, 100 ns gates and 500 ns measurements. Each algorithm receives whole-readout physical-fault allowance $10^{-3}$ across all 12,288 preparations. Digitized annealing separately receives arbitrary-rotation model scalar allowance $10^{-5}$ per execution, not a stochastic failure probability or angle certificate.

| Fixed method | Physical error | Declared selection | All preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-readout physical-fault score |
|---|---:|---|---:|---:|---:|---:|
| annealing | 0.001 | fewest physical qubits | 12,288 | 1,317,615 | 8502.067200 | 0.000294 |
| annealing | 0.001 | shortest modeled runtime | 12,288 | 1,422,975 | 7821.901824 | 0.000432 |
| walk | 0.001 | fewest physical qubits | 12,288 | 704,817 | 123.801600 | 0.000105 |
| walk | 0.001 | shortest modeled runtime | 12,288 | 876,993 | 94.089216 | 0.000594 |
| annealing | 0.0001 | fewest physical qubits | 12,288 | 131,595 | 8502.067200 | 0.000278 |
| annealing | 0.0001 | shortest modeled runtime | 12,288 | 260,115 | 3740.909568 | 0.000431 |
| walk | 0.0001 | fewest physical qubits | 12,288 | 40,101 | 64.376832 | 0.000271 |
| walk | 0.0001 | shortest modeled runtime | 12,288 | 47,173 | 44.568576 | 0.000761 |

The walk supplies an explicit exact Clifford+T gate sequence: 56 source CCX gates use 392 T/T† gates per preparation. No-op compiler fences preserve this sequence through shared lowering. The actual estimated canonical trace has **T=392, arbitrary RZ=0, and 11 measurements**; it therefore uses zero arbitrary-rotation model allowance. This is a logical sequence check, not a certification of hardware faults or a formal walk proof.

The annealing source counts preparation, all driver/problem layers and final measurements. The walk counts uniform preparation, the explicit known-target equality oracle and AND uncomputation, coin, shift, all five work qubits and final readout. The known target is a teaching input, not a database benchmark.

Classical input/compilation/scoring time, physical initialization/reset, routing, communication and adaptive feedback are unmeasured or excluded from these quantum-model times. QEC/factory reliability remains conditional on the declared models. Digitization and state-specific adiabatic diagnostics are separate from sampling confidence, coherent synthesis-model allowance and physical faults; no generic spectral-gap, success, speedup or optimizer guarantee is inferred.
