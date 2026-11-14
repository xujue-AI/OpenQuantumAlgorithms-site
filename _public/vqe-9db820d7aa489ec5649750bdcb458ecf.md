---
title: "VQE: vary a state to lower its energy"
description: "An explicit two-qubit variational ansatz, an independently checked energy, and complete optimization and measurement accounting."
---

VQE uses a classical optimizer to choose the parameters of a quantum state-preparation circuit. Each proposed state is prepared repeatedly, its Hamiltonian terms are measured, and the estimated energy becomes the optimizer's objective {cite:p}`opt_vqe2014`. A short preparation circuit does not by itself determine the cost of finding good parameters: all proposals, measurement groups, shots, and classical work belong to the workflow.

For a Hermitian Hamiltonian $H$ with lowest eigenvalue $E_0$, every normalized trial state satisfies

$$
E(\theta)=\langle\psi(\theta)|H|\psi(\theta)\rangle\ge E_0.
$$

The ansatz determines which states can be reached. Optimization can reduce energy only within that family, and finding its global minimum is a separate problem. Our small example chooses a family whose exact energy is easy to check. It establishes no general convergence guarantee or quantum advantage.

## State the Hamiltonian and requested result

Use dimensionless coefficients and

$$
H_1=-Z_0-Z_1+\frac12X_0X_1.
$$

In basis order $|q_1q_0\rangle=|00\rangle,|01\rangle,|10\rangle,|11\rangle$, its full matrix is

$$
H_1=\begin{pmatrix}
-2&0&0&1/2\\
0&0&1/2&0\\
0&1/2&0&0\\
1/2&0&0&2
\end{pmatrix}.
$$

Exact matrix diagonalization gives eigenvalues $-\sqrt{17}/2,-1/2,1/2,\sqrt{17}/2$, with reference ground energy $E_0=-\sqrt{17}/2\approx-2.061552813$. The requested numerical optimization quality is an **ideal expectation** gap $E-E_0\le10^{-8}$. The returned angles and exact objective are recorded separately from finite-shot estimates and their uncertainty.

## Build the ansatz from two gates

Initialize $|00\rangle$, apply `ry(theta, 0)`, and then `cx(0, 1)`. Angles are radians, with $R_Y(\theta)=e^{-i\theta Y/2}$. The prepared state is

$$
|\psi(\theta)\rangle=
\cos(\theta/2)|00\rangle+\sin(\theta/2)|11\rangle.
$$

The circuit below is the saved output of the [executed optimization notebook](../../notebooks/optimization.ipynb), using the same original implementation as the artifact generator.

![](#opt-vqe-circuit)

Its independent scalar energy formula is

$$
E_1(\theta)=-2\cos\theta+\frac12\sin\theta.
$$

The minimum occurs at $\theta=-\arctan(1/4)$ modulo $2\pi$ and equals the full matrix ground energy. Thus this particular Hamiltonian's ground state lies in the chosen even-parity ansatz. The circuit generator exposes a parameterized **two-qubit** family; it is not a general scalable VQE ansatz generator.

![](#opt-vqe-energy)

The circuit expectation, scalar formula, and full matrix eigenvalue are independently compared. The [portability example](portability.md) changes the Hamiltonian coefficients while keeping this ansatz and the initial angle fixed across seeds. Its separate results do not extend this expressibility statement to arbitrary Hamiltonians.

## Optimize exactly, then sample the final state

The first implementation uses exact ideal statevector energies in a deterministic coordinate-polling optimizer. Its three initial angles are $0.6,-1.1,2.2$, declared before the runs. Starting with a $0.4$-radian step, each sweep evaluates both negative and positive displacements from the incumbent. It accepts the best decrease exceeding $10^{-12}$; otherwise it halves the step. Angles are reduced to $[-\pi,\pi)$. It stops when the step falls below $10^{-6}$ radians or after 128 sweeps.

Every proposed and rejected energy evaluation uses zero measurement shots and appears in the [complete run record](../../assets/generated/optimization/runs.json), together with stopping reasons, achieved gaps, evaluation counts, and CPU timings. Those timings measure this classical implementation, including its statevector calculations.

The first initialization is chosen as the primary result before comparing outcomes. Its fitted angle is then evaluated in separate finite-shot experiments with seeds `7`, `19`, and `41`, each using 2,048 shots **per measurement group**. Samples are drawn classically from ideal circuit probabilities; repeated QPU executions are not performed. These samples do not feed back into the exact optimizer. The [finite-shot increment](finite-shot.md) records a separate optimizer whose decisions actually use sampled objectives.

## Measure two groups and retain covariance

The all-Z group scores each bitstring jointly as $Y_Z=-Z_0-Z_1$. The all-X group applies H to both qubits before computational-basis readout and scores $Y_X=\tfrac12X_0X_1$. Their independently sampled means give

$$
\widehat E=\overline Y_Z+\overline Y_X,\qquad
\widehat{\mathrm{SE}}(\widehat E)=
\sqrt{\frac{\widehat{\mathrm{Var}}(Y_Z)}{N_Z}
+\frac{\widehat{\mathrm{Var}}(Y_X)}{N_X}}.
$$

The Z terms use the same bitstrings, so the variance of their joint score retains covariance. Treating them as independent term measurements would misstate uncertainty. Both groups require a fresh ansatz preparation per shot; the H basis changes and final measurements are explicit in the [all-Z](../../assets/generated/optimization/vqe-z.qasm) and [all-X](../../assets/generated/optimization/vqe-x.qasm) circuit exports.

Across three seeds, this final-readout protocol represents 12,288 two-qubit preparations. Its sampled energy can lie below $E_0$ through shot noise even though the exact expectation satisfies the variational bound. The finite-shot estimates do not certify the $10^{-8}$ ideal ground-energy target.

## Empirical performance across sizes and starts

The two shared finite-shot studies use fixed recipes and retain every prescribed outcome. Their complete records appear in the [size study](finite-shot.md#quality-and-workload-across-sizes) and [initialization diagnostics](finite-shot.md#initialization-and-sampling-diagnostics). Counts are generated classically from ideal circuit probabilities. These are bounded implementation studies, not hardware measurements, convergence theorems, or quantum-advantage evidence.

The size study uses open chains at $n=2,3,4,6$, and the initialization diagnostics use $n=4,6$. Their Hamiltonian is $H_n=-\sum_j Z_j+\tfrac12\sum_{j=0}^{n-2}X_jX_{j+1}$. Each study circuit applies one RY per qubit followed by the ascending CX chain; this concrete ansatz does not establish ground-state expressibility for every chain. The requested quality is density gap $(\langle H_n\rangle-E_0)/n\le0.05$, with an absolute allowance $0.05n$. It differs from the first two-qubit demonstration's $10^{-8}$ ideal energy-gap target.

The locked recipe has been executed for all **72 runs**. Independent post-training audits meet the target in **19 of 60 QAOA runs** and **4 of 12 VQE runs**. Simultaneous holdout intervals support the target in **18 QAOA runs** and **no VQE runs**; every run supports the separately requested $\pm0.05$ normalized readout precision. All predetermined runs and misses are retained, with no retuning of initializations, schedule, targets, or readout allocation.

The frozen numerical recipe has executed all **36 sampled runs** and **24 separately labeled deterministic CPU controls**. Independent post-training audits meet the target in **4 of 24 QAOA runs** and **2 of 12 VQE runs**. Simultaneous holdout intervals support the target in **three QAOA runs** and **no VQE runs**; all 36 runs support the separately requested $\pm0.05$ normalized readout precision. Every prescribed run and miss remains in the record.


The VQE density gap is $0.010811$–$0.011081$ at $n=2$, and $0.179165$–$0.235975$ at $n=6$. The recorded zero-state baseline at $n=6$ has density gap $0.0525804$, so the three fitted six-spin states also miss that simple baseline. The fixed search is retained rather than restarted from a more favorable point. Its four exact target hits remain distinct from the zero simultaneous holdout supports: a valid expectation-precision allocation can still leave a quality threshold inconclusive.

For four-spin VQE, the zero start gives two sampled exact target hits, seeds `7` and `41`, at density gap $0.037892$; seed `19` ends at $0.050844$ and misses. All three holdout intervals are inconclusive. The zero state's exact density gap is already $0.047100$: its expected margin control accepts no move, while its exact objective control reaches $0.032233$. From the original all-$0.6$ start, the sampled gaps are $0.123778$–$0.229939$, all with holdout-supported misses. Its exact objective control also reaches $0.032233$, but its expected margin control accepts no move and stays at $0.564205$.

Every six-spin sampled run misses the target. The original-start gaps are $0.498964$–$0.596566$, with holdout-supported misses. The zero-start gaps are $0.052580$–$0.055903$, with inconclusive holdouts; two sampled trajectories worsen the zero-state baseline, and seed `41` stays at it. From zero, the exact objective control reaches $0.042669$ and meets the target, while the expected margin control accepts no move and retains the baseline $0.052580$. The exact controls thus exhibit target-reaching states for both VQE inputs without claiming that either ansatz contains its full ground state. The failed original-start six-spin exact control, at gap $0.087448$, is a local-search outcome rather than an infeasibility result.

The size study keeps the original simultaneous family allocation $0.05/84$, and the diagnostic study keeps $0.05/48$. Showing one method's outcomes here does not reduce either family denominator, recalculate its intervals, or reinterpret holdout support as an optimizer success probability. Deterministic CPU controls consume zero quantum preparations; their virtual shot allocations are diagnostic inputs rather than measured workloads.

## End-to-end resource analysis

For a general VQE study, let $A(n,\ell)$ be the chosen ansatz's unitary gate count, $G$ its measurement-group count, $B_g(n)$ the basis-change gate count for group $g$, and $N_{kg}$ its shots at objective evaluation $k$. The circuit-operation ledger has the form

$$
\sum_{k=1}^{K}\sum_{g=1}^{G}
N_{kg}\,[A(n,\ell)+B_g(n)+\text{readout}_g].
$$

This is an accounting pattern, not a measured large-instance cost or a bound on the required number $K$ of optimizer evaluations. The ansatz, grouping, requested precision, initialization, stopping rule, and classical optimizer must be specified before such an estimate becomes a workflow contract. The current demo has one RY and one CX per preparation, two groups, and explicit two-qubit readout.

The [logical ledger](../../assets/generated/optimization/logical-resources.json) distinguishes exact simulated objectives from the modeled sampled protocol. The [conditional recorded-trajectory accounting below](#vqe-historical-resources) adds the synthesis, physical-fault, factory, and scheduling qualifications. An arbitrary-angle RY is not a validated T count, and a final circuit is not an estimate of the entire optimizer.

(vqe-historical-resources)=
### Recorded finite-shot resources: historical conditional model

This historical projection concerns the first two-qubit Hamiltonian $H_1$ and its one-angle RY/CX ansatz under the recorded 32-sweep sampled schedule. It does not estimate the larger chain cases or the later initialization diagnostics. The [complete recorded-trajectory analysis](finite-shot.md#recorded-trajectory-resources) maps every initial call, incumbent remeasurement, accepted or rejected proposal, and holdout to an explicit measured circuit. The VQE task has 579,072 preparations and 588 all-Z/all-X batches. These are modeled repetitions of stored circuits, not predictions of a fresh hardware-driven adaptive path.

| Task | Physical error | Declared serial choice | All modeled preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-task physical-fault score |
|---|---:|---|---:|---:|---:|---:|
| VQE | 0.001 | fewest qubits per cost signature | 579,072 | 671,529 | 201.436672 | 0.000205 |
| VQE | 0.001 | shortest runtime per cost signature | 579,072 | 781,123 | 182.305280 | 0.000438 |
| VQE | 0.0001 | fewest qubits per cost signature | 579,072 | 24,029 | 201.174528 | 0.000444 |
| VQE | 0.0001 | shortest runtime per cost signature | 579,072 | 46,529 | 86.386176 | 0.000703 |

These are the unchanged VQE rows from the original replay, not a new estimate. Each task retains its separate whole-task physical-fault allowance $10^{-3}$ and per-preparation synthesis-model scalar $10^{-5}$. The scalar is a coherent approximation-model quantity and is not summed over shots as a stochastic failure probability. Profiles retain physical error $10^{-3}$ or $10^{-4}$, 100 ns gates and 500 ns measurements.

Times sum every preparation's model runtime; peaks take the maximum of independently selected class footprints. A common physical configuration is not verified. Physical initialization/reset, transport, routing, reconfiguration, communication and adaptive feedback remain unmeasured or excluded. The PSSPC rotation fit supplies expected resource cost, not concrete synthesized sequences, worst-case bounds, runtime-tail guarantees, or an output-accuracy certificate. CPU controller, reference and simulator times remain separate.

The larger shared size-study replay also retains all 72 runs and 5,859,840 preparations, including every quality miss. Its 84 cost classes and 85 searches remain a separate frozen record with independently selected class configurations. They are not a time-to-quality estimate or a physical proxy for the later initialization diagnostics, which have no new physical projection.

## Verification and formalization

Independent checks compare explicit circuits with classical references, preserve preparation and measurement maps, and replay recorded sampled decisions from their counts. The finite-shot protocol's fresh holdout bounds concern selected fixed states conditional on their training histories; they do not certify adaptive training queries, convergence, or physical hardware. Circuit export and bounded classical simulation have separate limits.

**Classical C4 reference coverage only.** The reproduced Lean artifact establishes finite C4 cut values and the stated classical equality predicate. It does not connect those definitions to Python or prove QAOA circuits, VQE ansätze, stochastic decisions, synthesis, physical schedules, or hardware confidence. [Proof source](../../assets/generated/optimization/formal/C4Reference.lean) and [verification record](../../assets/generated/optimization/formal/verification.json) retain the exact existing coverage.

## Reproduce and inspect

Use the pinned Python 3.12 environment and original Qiskit core 2.6.2 and NumPy 2.5.4 implementation {cite:p}`opt_qiskit262,opt_numpy254`. Package sources and licenses are credited in the [main chapter](../optimization.md). The exposition, implementation, and figures here are original.

```bash
python scripts/generate_optimization.py
python -m pytest tests/test_optimization.py
python scripts/execute_notebooks.py
```

- [Executed notebook](../../notebooks/optimization.ipynb)
- [Angles, full traces, exact checks, and sampled results](../../assets/generated/optimization/runs.json)
- [Every objective evaluation, CSV](../../assets/generated/optimization/evaluations.csv)
- [Logical resources and explicit exclusions](../../assets/generated/optimization/logical-resources.json)
- [Standalone implementation](../../assets/generated/optimization/optimization.py)
- [Artifact generator](../../assets/generated/optimization/generate_optimization.py)
- [MIT implementation license](../../assets/generated/optimization/LICENSE.txt)

## References

```{bibliography}
```
