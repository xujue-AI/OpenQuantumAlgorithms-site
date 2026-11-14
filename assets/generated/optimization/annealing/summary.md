The fixed main digitized anneal uses **T=16, L=256**, chosen before the studies. The input is the four-vertex cycle and the initial state is |+> on each qubit. H_D = -sum X, H_P = 0.5 sum ZZ = 2-C, and the linear path is H(s) = (1-s) H_D + s H_P. Evolution times are Hamiltonian units with ℏ=1, rather than hardware execution times.

The main circuit's exact ideal maximum-cut probability is **0.987022729** and mean cut is **3.974032281**. The predeclared probability target ≥0.95 was met: **True**. The independent continuous DOP853 reference has maximum-cut probability 0.987036431. Circuit-versus-reference state infidelity is 1.19e-08; absolute mean-cut error is 2.74e-05. These are state-specific numerical checks, rather than operator-error or global adiabatic guarantees.

| Time study, fixed L=256 | Circuit P(max cut) | Continuous ODE P(max cut) | State infidelity vs ODE | Absolute mean-cut error vs ODE |
|---:|---:|---:|---:|---:|
| 1 | 0.204310889 | 0.204308801 | 1.62e-11 | 6.65e-06 |
| 2 | 0.375910127 | 0.375906317 | 8.88e-11 | 9.36e-06 |
| 4 | 0.615426395 | 0.615424443 | 2.08e-10 | 5.44e-06 |
| 8 | 0.898700623 | 0.898730931 | 4.51e-09 | 6.02e-05 |
| 16 | 0.987022729 | 0.987036431 | 1.19e-08 | 2.74e-05 |

The main continuous reference uses SciPy DOP853 with rtol=1e-11, atol=1e-13, 1973 derivative evaluations, and recorded norm drift 6.1e-13. Solver tolerances are numerical settings, not certified error bounds.

| Step study, fixed T=4 | Circuit P(max cut) | State infidelity vs ODE | Absolute mean-cut error vs ODE |
|---:|---:|---:|---:|
| 8 | 0.616978549 | 0.000192 | 0.00449 |
| 16 | 0.615899648 | 1.33e-05 | 0.00133 |
| 32 | 0.615547862 | 8.47e-07 | 0.000345 |
| 64 | 0.615455578 | 5.32e-08 | 8.69e-05 |
| 128 | 0.615432244 | 3.33e-09 | 2.18e-05 |

The 201-point gap scan has a smallest sampled G=+1 sector gap of 1.010939045. The full gap is zero at s=1 because two cut states are degenerate. The initial state lies in the conserved G=X⊗4, G=+1 sector. A positive finite-grid sector diagnostic does not certify a gap everywhere, an adiabatic runtime theorem, or behavior on another graph.

All time/step studies and dense ODE calculations are classical simulations with **zero measurement shots**. The three main-circuit readouts below use ideal multinomial counts, not hardware executions. Their sample variance scores each complete cut jointly; estimated standard errors are descriptive.

| Seed | Modeled main-circuit shots | Sample mean cut | Estimated SE | Maximum-cut fraction | Best observed cut |
|---:|---:|---:|---:|---:|---:|
| 7 | 4096 | 3.97656250 | 0.003363 | 0.988281 | 4 |
| 19 | 4096 | 3.97265625 | 0.003629 | 0.986328 | 4 |
| 41 | 4096 | 3.97167969 | 0.003693 | 0.985840 | 4 |

Per main circuit: H 4, RX 2048, RZ 1024, CX 2048, measurement 4. The modeled protocol totals **12,288 fresh main-circuit preparations** and **49,152 qubit readouts**. The ledger repeats every preparation, driver rotation, explicit edge interaction, and final measurement for all shots. Physical reset/measurement time, routing, rotation synthesis, fault correction and hardware runtime are excluded. Arbitrary rotations remain unsynthesized in this logical ledger. Any conditional physical projection is a separate report.

The recorded bounded CPU workflow took 1.359715 s for circuit simulation, dense references, gap diagnostics and ideal sampling. Artifact drawing, export and logical-ledger compilation are outside that timer; it is not quantum hardware time.
