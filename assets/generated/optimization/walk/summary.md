The implemented teaching task is coined search on a 4-dimensional hypercube with 16 vertices, a supplied marked vertex `0101` (integer 5), and a prescribed four-step final readout. The target value is used to construct an explicit equality predicate. This is a quantum-walk primitive, with no application benchmark or optimization-performance claim.

The active register has two binary coin bits and four position bits. Five clean workspace bits implement the joint reflection `I − 2|s_coin,5〉〈s_coin,5|` through AND computation, phase and uncomputation. Each step applies this oracle, the coin `I − 2|s〉〈s|`, and an explicitly matched shift. The step differs from the standard SKW step by a global minus sign, which is harmless for this uncontrolled search. The implementation does not promise phase-correct controlled reuse.

The independent 64 × 64 active-space matrix gives marked-position probability **25/64 = 0.390625** after four steps. Across the fixed scan of zero through eight steps, maximum circuit/reference amplitude error was 1.47e-15, and maximum workspace leakage was 0. The scan was a diagnostic; it did not select the four-step input. Every one of the 64 clean-work active basis inputs also passes the independently formed joint-oracle check. These are finite numerical checks, without a formal proof.

Three seeded final readouts use 4,096 modeled ideal measurements each. All 11 qubits are measured; workspace outcomes are retained rather than postselected. The target is an absolute frequency error of 0.025 at 95% confidence for each fixed estimate. The separate Hoeffding bound has outcome range one; it does not provide simultaneous coverage of all rows. The random contract is NumPy PCG64 with `SeedSequence([seed,5])`.

| Seed | Marked counts / shots | Marked frequency | Error from 25/64 | Hoeffding 95% half width | Nonzero-work outcomes |
|---:|---:|---:|---:|---:|---:|
| 7 | 1605 / 4096 | 0.391846 | +0.001221 | 0.021220 | 0 |
| 19 | 1629 / 4096 | 0.397705 | +0.007080 | 0.021220 | 0 |
| 41 | 1647 / 4096 | 0.402100 | +0.011475 | 0.021220 | 0 |

For this fixed validated state, **10 independent preparations** give ideal probability 0.992939 of at least one marked readout, exceeding the prescribed 0.99 target. This formula uses the verified fixed-state probability and equality checks after measurement; it is not a universal walk or optimizer success guarantee. Every trial includes preparation, four oracle/coin/shift calls and all-qubit readout.

| Logical workload | Per preparation | Three 4,096-shot readouts | Fixed 0.99 at-least-one task |
|---|---:|---:|---:|
| Preparations | 1 | 12,288 | 10 |
| Oracle calls | 4 | 49,152 | 40 |
| Coin calls | 4 | 49,152 | 40 |
| Shift calls | 4 | 49,152 | 40 |
| Individual qubit measurements | 11 | 135,168 | 110 |
| T / T† gates | 392 | 4,816,896 | 3,920 |

The 11-wire main circuit uses 56 explicit CCX gates (40 in the four joint oracles and 16 in the shifts). Deterministic optimization-level-zero compilation gives **392 T/T† gates**, 340 CX gates, and no arbitrary rotations. The logical ledger includes input preparation, the constructed oracle, work uncomputation, coin, shift, all repeated trials, and all final measurements. Arbitrary-rotation synthesis error is zero for this literal Clifford+T implementation. CPU simulation, classical equality-check time, routing, error correction, faults, factories and physical timing are outside this logical ledger; the main chapter supplies the separate conditional surface-code analysis.

The source generator supports dimensions 2, 4, 8 and 16 and zero through 32 steps, subject to a conservative 100,000 primitive-gate cap (CCX costs 15; CZ costs 3; other source Clifford gates cost one). The total width is `2(d + log2(d)) − 1`. Statevector execution is separately capped at 12 total qubits, and the independent dense reference at six active bits. Dimensions 8 and 16 have exported four-step source and exact Clifford+T circuits **without simulation**. The marked reflection uses `2(d+r−1)` CCX per step; for `r≥2` the shift uses `d(2r−3)` CCX, with `r=log2(d)`. The coin adds `2(r−1)` CCX for `r≥3`, one CZ for `r=2`, or one Z for `r=1`; wrappers and Hadamards are also counted. Consequently source size per step is O(d log d); dense probabilities remain exponentially bounded.

The complete generator ran in 1.196660 s on the recorded CPU environment. Artifact JSON records the source commit, actual source/generator/lock hashes, versions, seeds and timing scope. Qiskit 2.6.2 (Apache-2.0) and NumPy 2.5.4 (BSD-3-Clause) are credited in the provenance; this original implementation and generator are provided under the repository MIT license. Algorithm source: Shenvi, Kempe and Whaley, Physical Review A 67, 052307 (2003), DOI `10.1103/PhysRevA.67.052307`, arXiv `quant-ph/0210064`.
