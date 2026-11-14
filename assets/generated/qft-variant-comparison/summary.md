The predeclared comparison uses phase **0.3**, one explicitly prepared target wire, phase sizes **m=5,6,7**, exact cutoff **m−1** and truncated cutoff **m−2**, with one omitted controlled phase. Both variants keep the same phase target 2⁻ᵐ, synthesis allowance, 1% whole-run failure target and hardware profile. The exact rational confidence allocation charges omission as a single-trial output-TV loss, separately from synthesis and whole-run physical faults.

| m | Variant | Fresh trials | Operations/trial | Operations/all trials | Omission TV upper |
|---:|---|---:|---:|---:|---:|
| 5 | exact | 15 | 227 | 3,405 | 0 |
| 5 | truncated | 127 | 222 | 28,194 | 0.196429 |
| 6 | exact | 15 | 418 | 6,270 | 0 |
| 6 | truncated | 35 | 413 | 14,455 | 0.0982143 |
| 7 | exact | 15 | 771 | 11,565 | 0 |
| 7 | truncated | 21 | 766 | 16,086 | 0.0491071 |

Omission does not change the declared numerical phase tolerance. Its uniform distribution bound increases repetitions. Binary64 phase angles, actual synthesis sequences/runtime tails and hardware confidence are not certified; physical initialization/reset and classical decoding/verification/controller latency are excluded.

The 24 bounded resource queries returned **123 Pareto points**. At the fastest returned points, truncated/exact total-runtime ratios range from **1.399** to **9.749**. These conditional mean-fit results retain all fresh trials and do not establish an application advantage.
