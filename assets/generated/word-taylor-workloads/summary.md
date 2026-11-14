# Accuracy-preserving word-Taylor workloads

Fixed local couplings are J=3/5 and h=7/20. Each cell chooses its segment
count and smallest degree from uniform exact-rational ideal-access bounds.
All complete attempts reset the known input, execute every segment,
measure all access ancillas, reset them between segments and read out Z.

The 16-cell grid was prescribed before generation. Materialization
rejections remain explicit. Circuit-only cells are not simulated results.
Sampling and physical resource estimation are unperformed.

| Cell | n | Time | Accuracy | r | K | Ideal acceptance lower | Complete attempt cap | Evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| size-n1-native | 1 | 1 | 0.05 | 1 | 3 | 0.496388 | 331464 | tiny-circuit-checked |
| size-n1-normalization2-oaa1 | 1 | 1 | 0.05 | 1 | 3 | 0.998654 | 41433 | tiny-circuit-checked |
| size-n2-native | 2 | 1 | 0.05 | 3 | 4 | 0.074252 | 2955554 | circuit-only |
| size-n2-normalization2-oaa1 | 2 | 1 | 0.05 | 3 | 4 | 0.999177 | 41433 | circuit-only |
| size-n3-native | 3 | 1 | 0.05 | 5 | 4 | 0.0111023 | 20343603 | circuit-only |
| size-n3-normalization2-oaa1 | 3 | 1 | 0.05 | 5 | 4 | 0.998338 | 41433 | circuit-only |
| size-n4-native | 4 | 1 | 0.05 | 7 | 4 | 0.00166001 | 136673656 | circuit-only |
| size-n4-normalization2-oaa1 | 4 | 1 | 0.05 | 7 | 4 | 0.997481 | 41433 | materialization-rejected: primitive export operation limit exceeded |
| time-half-native | 2 | 0.5 | 0.05 | 2 | 3 | 0.272381 | 718172 | tiny-circuit-checked |
| time-half-normalization2-oaa1 | 2 | 0.5 | 0.05 | 2 | 3 | 0.998011 | 41433 | tiny-circuit-checked |
| time-two-native | 2 | 2 | 0.05 | 6 | 4 | 0.00551336 | 41073914 | circuit-only |
| time-two-normalization2-oaa1 | 2 | 2 | 0.05 | 6 | 4 | 0.998354 | 41433 | circuit-only |
| accuracy-025-native | 2 | 1 | 0.025 | 3 | 4 | 0.074252 | 12815912 | circuit-only |
| accuracy-025-normalization2-oaa1 | 2 | 1 | 0.025 | 3 | 4 | 0.999177 | 165723 | circuit-only |
| accuracy-100-native | 2 | 1 | 0.1 | 3 | 3 | 0.0740186 | 676788 | tiny-circuit-checked |
| accuracy-100-normalization2-oaa1 | 2 | 1 | 0.1 | 3 | 3 | 0.990364 | 13812 | tiny-circuit-checked |

Exact bounds describe a mathematical access model. The saved
floating-point normalization, probability/angle construction and
compiler do not carry a uniform norm certificate. Rotation
synthesis and physical faults/timings/routing/factories remain
separate and unmodeled. Dependency depth is not hardware time.

Source snapshots and the MIT license support standalone
reproduction with the repository's locked Python 3.12 environment.
