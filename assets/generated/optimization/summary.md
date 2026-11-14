All objective calls below used exact ideal statevectors and **zero shots**. Three initializations were fixed before execution; the first is the reported circuit. Times are local CPU/software measurements.

| Algorithm / start | Final parameters (radians) | Exact value | Quality check | Sweeps / calls | Stop | Optimizer wall (s) | Classical control (s) |
|---|---|---:|---|---:|---|---:|---:|
| QAOA / 1 | 0.785397339, 0.392698669 | 3.0000000000 | ratio 0.750000000; target met True | 42 / 169 | step_tolerance | 0.175387 | 0.001879 |
| QAOA / 2 | 0.785397339, 0.392698669 | 3.0000000000 | ratio 0.750000000; target met True | 38 / 153 | step_tolerance | 0.160019 | 0.001642 |
| QAOA / 3 | -0.785397339, -0.392698669 | 3.0000000000 | ratio 0.750000000; target met True | 43 / 173 | step_tolerance | 0.177439 | 0.001821 |
| VQE / 1 | -0.244978333 | -2.0615528128 | ground gap 1.13e-13; target met True | 31 / 63 | step_tolerance | 0.022153 | 0.000670 |
| VQE / 2 | -0.244978333 | -2.0615528128 | ground gap 1.12e-13; target met True | 32 / 65 | step_tolerance | 0.022709 | 0.000684 |
| VQE / 3 | -0.244978333 | -2.0615528128 | ground gap 1.13e-13; target met True | 35 / 71 | step_tolerance | 0.024589 | 0.000743 |

The complete ideal toy-workflow execution took 0.591914 s on this CPU, including exact reference checks, post-fit quality checks and probability generation. Artifact drawing/export and logical-ledger compilation are outside that workflow timer.

Finite-shot results evaluate the first optimized state only. These are independent ideal multinomial samples, not executed hardware shots. The standard error is estimated from the joint measured observable; it is not a rigorous confidence interval. The Hoeffding radius separately gives an ideal-sampling 95% bound for each listed estimate, not simultaneous coverage across this table.

| Algorithm / seed | Total modeled circuit shots | Estimate | Error from exact mean | Estimated SE | Hoeffding 95% radius |
|---|---:|---:|---:|---:|---:|
| QAOA / 7 | 2048 | 3.02832031 | +0.028320 | 0.023998 | 0.120040 |
| QAOA / 19 | 2048 | 2.99804688 | -0.001953 | 0.024865 | 0.120040 |
| QAOA / 41 | 2048 | 2.97656250 | -0.023437 | 0.024860 | 0.120040 |
| VQE / 7 | 4096 | -2.10156250 | -0.040010 | 0.013327 | 0.123735 |
| VQE / 19 | 4096 | -2.06689453 | -0.005342 | 0.015636 | 0.123735 |
| VQE / 41 | 4096 | -2.04101562 | +0.020537 | 0.016169 | 0.123735 |

The first QAOA state's exact probability of an optimal cut is 0.531249794; 7 independent ideal draws suffice for at least one optimum with probability at least 0.99 at these fixed parameters. This is a readout statement, not an optimizer guarantee.

| Whole recorded workload | Exact simulated ansatz evaluations | Exact-objective shots | Modeled held-out circuit shots | Modeled qubit readouts | Shot-weighted CX | Shot-weighted arbitrary rotations |
|---|---:|---:|---:|---:|---:|---:|
| QAOA | 495 | 0 | 6144 | 24576 | 49152 | 49152 |
| VQE | 199 | 0 | 12288 | 24576 | 12288 | 12288 |

The ledger also stores each exact evaluation's parameters, logical gates and depth. These simulated circuit counts are separate from shot-weighted readout counts. Arbitrary rotations remain unsynthesized; physical qubits, T counts, factories and runtime are pending. Reset timing, routing, graph/input transfer, noise and mitigation are excluded. Reference checks are included in whole CPU workflow time but are not timed separately.
