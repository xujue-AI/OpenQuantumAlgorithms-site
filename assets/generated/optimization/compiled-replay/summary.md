The replay fixes **qaoa-cycle-4-p2-alternate-seed-7**, the first heldout-supported four-wire QAOA run in the frozen order. It includes **109 training calls**, **one holdout**, **110 measured groups**, and all **82,032 preparations**, including every rejected proposal. The saved ideal-sampling target status remains **supported**; VQE inconclusive labels and every other archived negative control remain unchanged.

The **54 exact source classes** have actual Clifford+T approximants and actual logical-line routing. Weighted across every recorded group, the replay uses **182,646,240 actual T/T-dagger state consumptions** and **328,128 original output-qubit readouts**. Explicit input resets are added for every preparation; the separate instrument model also charges reserved workspace starts, magic-state supply/injection, macro readout, and classical dispatch.

The summed guarded numerical realization estimate over the fixed full transcript is **0.0009485412**, within the frozen 0.005 allocation. The final compiled exact ratio is **0.9881989495**. Its prospective fixed-state precision radius is **0.01541186**, below 0.05. These use actual saved sequences and tiny independent numerical checks, not a mean-fit T proxy or formal interval certificate.

The conditional allocation reserves 0.04 for the full 48-group sampling family, 0.005 for fixed-script realization distance, 0.0025 for whole-workload physical faults, and 0.0025 for factory availability. The surface-code layer remains the frozen serialized instrument scenario with physical error 10^-4. It is separate from the Grover/PSSPC baseline, and its geometry, controller, factory/interface and independence assumptions are not hardware guarantees.

The original counts were generated from ideal source probabilities. They were **not** redrawn from these approximants. A compiled replay conditional on those recorded classical choices does not prove that a changed noisy optimizer would take the same path. Offline CPU optimizer/synthesis/verification timing is recorded separately; hardware optimizer/scoring/communication and decoder latency remain unmeasured. No convergence or quantum advantage follows.

The physical estimator was evaluated twice. The original representation-limit failure remains visible and its measured time/query counts are charged. One explicitly declared computation-only recovery reused all 54 compiled checkpoints; quantum preparations and T requirements are counted once. The lossless DAG was checked against the second live producer result. Complete first-pass point records were unavailable, so no equality to that missing result is claimed.

| Common selection | Distance | Peak physical qubits | Serial model time (s) | Whole physical-fault model | Factory availability bound |
|---|---:|---:|---:|---:|---:|
| minimum_qubits | 15 | 159,395 | 863080.953 | 0.000153065 | 0.000182646 |
| minimum_runtime | 15 | 243,358 | 797328.307 | 0.000143202 | 0.000182646 |
