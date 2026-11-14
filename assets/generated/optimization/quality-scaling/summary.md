This frozen, bounded study retains every predetermined run and every target miss. It covers 60 QAOA runs on path/cycle graphs with n = 3, 4, 5, 6, 8 and p = 1, 2, plus 12 VQE runs on n = 2, 3, 4, 6 chains. Each case uses seeds 7, 19 and 41 with the same initialization. QAOA angles start at interleaved (0.2, 0.1) per layer. VQE uses H = -sum Z + 0.5 sum nearest-neighbor XX, one layer of n RY gates followed by a CX chain, and all initial angles 0.6.

Training uses raw sampled cut or energy scores. Exact classical optima/eigenvalues are post-run references, never optimizer decisions. The fixed schedule has four sweeps each at (step radians, shots/group) = (0.4, 256), (0.2, 512), (0.1, 1024), with 128 shots/group for the initial call. Every sweep remeasures the incumbent and both displacements of every coordinate. The two-standard-error acceptance margin remains a heuristic without a convergence guarantee.

Fresh holdouts use 16,384 shots per measurement group. Their pointwise per-run bounds are reported separately from the whole-study allocation δ = 0.05 across 84 holdout groups. The resulting group intervals combine into conditional fixed-state ratio/energy-density intervals. A ±0.05 expectation precision bound is distinct from reaching a quality target. The audited quality targets are QAOA mean cut / classical optimum ≥0.80 and VQE energy-density gap ≤0.05.

| Case | n | p | Audited quality range over three seeds | Exact target met | Whole-study interval supports target | Modeled shots per run | Source unitary gates per run |
|---|---:|---:|---|---:|---:|---:|---|
| QAOA cycle | 3 | 1 | 0.999526–0.999526 | 3/3 | 3/3 | 52,352 | 785,280 |
| QAOA cycle | 3 | 2 | 0.996508–0.999526 | 3/3 | 3/3 | 81,024 | 2,187,648 |
| QAOA cycle | 4 | 1 | 0.711875–0.749787 | 0/3 | 0/3 | 52,352 | 1,047,040 |
| QAOA cycle | 4 | 2 | 0.762274–0.762274 | 0/3 | 0/3 | 81,024 | 2,916,864 |
| QAOA cycle | 5 | 1 | 0.937234–0.937234 | 3/3 | 3/3 | 52,352 | 1,308,800 |
| QAOA cycle | 5 | 2 | 0.945072–0.945072 | 3/3 | 3/3 | 81,024 | 3,646,080 |
| QAOA cycle | 6 | 1 | 0.749787–0.749787 | 0/3 | 0/3 | 52,352 | 1,570,560 |
| QAOA cycle | 6 | 2 | 0.755416–0.756137 | 0/3 | 0/3 | 81,024 | 4,375,296 |
| QAOA cycle | 8 | 1 | 0.749787–0.749787 | 0/3 | 0/3 | 52,352 | 2,094,080 |
| QAOA cycle | 8 | 2 | 0.755078–0.756137 | 0/3 | 0/3 | 81,024 | 5,833,728 |
| QAOA path | 3 | 1 | 0.804156–0.823892 | 3/3 | 2/3 | 52,352 | 628,224 |
| QAOA path | 3 | 2 | 0.828513–0.830124 | 3/3 | 3/3 | 81,024 | 1,701,504 |
| QAOA path | 4 | 1 | 0.791670–0.791670 | 0/3 | 0/3 | 52,352 | 889,984 |
| QAOA path | 4 | 2 | 0.798900–0.821446 | 1/3 | 1/3 | 81,024 | 2,430,720 |
| QAOA path | 5 | 1 | 0.775560–0.776971 | 0/3 | 0/3 | 52,352 | 1,151,744 |
| QAOA path | 5 | 2 | 0.780628–0.787088 | 0/3 | 0/3 | 81,024 | 3,159,936 |
| QAOA path | 6 | 1 | 0.765893–0.771534 | 0/3 | 0/3 | 52,352 | 1,413,504 |
| QAOA path | 6 | 2 | 0.781507–0.783820 | 0/3 | 0/3 | 81,024 | 3,889,152 |
| QAOA path | 8 | 1 | 0.764521–0.765321 | 0/3 | 0/3 | 52,352 | 1,937,024 |
| QAOA path | 8 | 2 | 0.770132–0.774052 | 0/3 | 0/3 | 81,024 | 5,347,584 |
| VQE chain | 2 | one layer | 0.010811–0.011081 | 3/3 | 0/3 | 104,704 | 418,816 |
| VQE chain | 3 | one layer | 0.036397–0.085680 | 1/3 | 0/3 | 133,376 | 866,944 |
| VQE chain | 4 | one layer | 0.057899–0.133581 | 0/3 | 0/3 | 162,048 | 1,458,432 |
| VQE chain | 6 | one layer | 0.179165–0.235975 | 0/3 | 0/3 | 219,392 | 3,071,488 |

Observed exact target passes: QAOA 19/60, VQE 4/12. These counts describe the fixed seed set, rather than a population optimizer-success probability. Misses can reflect ansatz restrictions or optimizer behavior; this experiment does not separate their causes.

The record contains **6,408 sampled objective calls**, **7,656 group batches**, and **5,859,840 modeled circuit preparations** (4,483,584 training and 1,376,256 fresh holdout shots). All counts are classically sampled from ideal circuit probabilities; no QPU execution or noise is measured. The source manifest links every group's parameters, basis, RNG stream, shots, exact OpenQASM hash, depth and logical gate counts. Byte-identical source circuits share a file only after SHA-256 comparison.

The p=2 QAOA runs use more coordinates, proposals and shots than p=1 under this fixed sweep schedule. Their different costs are explicit; this is not an equal-budget comparison. Circuit loading/preparation, basis changes and every rejected/accepted evaluation are included in the source ledger. Rotations are unsynthesized, routing/reset/device latency are unmeasured, and no physical-resource comparison is inferred. CPU timings measure the simulator and classical control, not quantum hardware time.
