---
title: "Portability with fixed starts"
description: "The same bounded sampled optimizer on a triangle MaxCut task and a second stated two-qubit Hamiltonian."
---

The [finite-shot examples](finite-shot.md) couple three sampling seeds to different initial angles. Here we reuse the same circuit generators and sampled polling rule on two changed inputs, while holding each task's initial angles fixed across seeds. This separates sampling-seed variation from initialization changes within these runs. It tests a small amount of implementation portability rather than large-instance performance of QAOA or VQE {cite:p}`opt_qaoa2014,opt_vqe2014`.

:::{note} Fixed experiment and evidence

The three seeds, initial angles, 32-sweep schedule, measurement groups, and post-training targets are declared before the runs. Results are reported for every seed, including target misses. The inputs and schedule are not retuned after seeing outcomes.

Counts are ideal samples drawn classically from explicit circuit probabilities. Exact reference values are post-training diagnostics. A successful toy run does not establish convergence, a hardware success probability, or an advantage over the independent classical solution.
:::

## A triangle MaxCut input

Use three vertices and edges

$$
\mathcal E_\triangle=\{(0,1),(1,2),(0,2)\}.
$$

Exhaustive enumeration of its eight assignments gives cut values $[0,2,2,2,2,2,2,0]$ in integer bitstring order. The classical maximum is two. Any assignment with one vertex separated from the other two achieves it; `000` and `111` have cut zero. Qiskit strings read $q_2q_1q_0$ with wire zero least significant.

The one-layer QAOA generator prepares $|+\rangle^{\otimes3}$ with H gates, uses `CX; RZ(-gamma); CX` for each edge, and uses `RX(2*beta)` for the mixer. The minimized objective remains the negative mean cut. An all-Z outcome scores all three edges jointly. The requested post-training exact mean is at least $1.90$, a gap at most $0.10$ from this input's classical maximum two. It is separate from the probability of sampling a maximum-cut bitstring.

## A second stated Hamiltonian

Use dimensionless energy coefficients

$$
H_2=-0.8Z_0-0.6Z_1+0.4X_0X_1.
$$

In basis order $|q_1q_0\rangle=|00\rangle,|01\rangle,|10\rangle,|11\rangle$,

$$
H_2=\begin{pmatrix}
-1.4&0&0&0.4\\
0&0.2&0.4&0\\
0&0.4&-0.2&0\\
0.4&0&0&1.4
\end{pmatrix}.
$$

Exact diagonalization gives eigenvalues $-\sqrt{2.12},-\sqrt{0.20},\sqrt{0.20},\sqrt{2.12}$ and ground energy $E_{0,2}=-\sqrt{2.12}\approx-1.456021978$. The asymmetric middle diagonal entries follow from the qubit ordering and the separately supplied Z coefficients.

The same `RY(theta)` on qubit zero followed by `CX(0,1)` prepares

$$
|\psi(\theta)\rangle=\cos(\theta/2)|00\rangle+\sin(\theta/2)|11\rangle.
$$

Its independent energy formula is

$$
E_2(\theta)=-1.4\cos\theta+0.4\sin\theta.
$$

This has its minimum at $\theta=-\arctan(2/7)$ modulo $2\pi$, equal to the matrix ground energy. Thus this second Hamiltonian still has a ground state in the same even-parity ansatz family. It changes the supplied coefficients and measurement scores; it does not test an ansatz able to represent arbitrary two-qubit ground states. The post-training target is $E_2-E_{0,2}\le0.03$.

All-Z outcomes jointly score $-0.8Z_0-0.6Z_1$, retaining covariance. The independent all-X group scores $0.4X_0X_1$ after H basis changes on both qubits. Each VQE call uses its allocated shots in both groups. Matrix diagonalization and the scalar expectation are independent checks of the circuit energy.

## Reuse the protocol and vary only the seed

Every QAOA run starts at $(\gamma,\beta)=(0.2,0.1)$ and every VQE run starts at $\theta=0.6$. Seeds are `7`, `19`, and `41`. NumPy `SeedSequence` uses algorithm IDs two and three for these portability tasks, keeping their streams separate from IDs zero and one used in the original tasks {cite:p}`opt_seedsequence254`. Training and fresh validation also use distinct stream families. Within each task, the supplied input, initial point, optimizer settings, and measurement allocation stay fixed as the seed changes.

The initial call uses 256 shots per group. Four eight-sweep blocks use coordinate step/shot pairs $(0.4,256),(0.2,512),(0.1,1024),(0.05,2048)$. Every sweep samples a fresh incumbent and all negative/positive coordinate proposals. A move requires the sampled reduction to exceed twice the combined descriptive standard error. The only stopping rule is schedule exhaustion. The [finite-shot chapter](finite-shot.md) details why this adaptive rule is not a calibrated statistical test or convergence certificate.

Every final state receives a fresh 4,096-shot-per-group holdout. This separate readout task asks for absolute expectation error at most $0.10$ at 95% confidence for one fixed estimate. The bounded-score radii are about $0.0424$ for triangle cuts and $0.0618$ for the second energy, supporting that readout allocation under the ideal independent-sampling model {cite:p}`opt_hoeffding1963`.

These bands refer to the expectation at the selected angles. They are not optimizer-success bounds, and six individually 95% intervals are not jointly 95%. Exact post-training quality and finite-shot estimates are shown separately. The VQE holdout band is wider than the requested $0.03$ ground-energy gap; supporting the separate $0.10$ expectation-estimation task does not certify that ground-energy target.

```{include} ../../assets/generated/optimization/portability/summary.md
```

![](#opt-portability-trajectories)

The exact trajectory values in the plot are attached after decisions; they do not guide the sampled optimizer. Fixed starts make seed variation easier to inspect, but three predetermined seeds are a small set of reproducible cases rather than a distributional benchmark.

![](#opt-portability-readout)

Seed 19's QAOA holdout illustrates the difference between descriptive error and a bounded interval: all 4,096 draws have cut two, giving sampled mean two and estimated standard error zero. The final state's exact mean is approximately $1.999052281$, and the Hoeffding radius remains about $0.0424$. Zero empirical variance does not certify an exact mean or eliminate readout uncertainty.

## Record the complete workload and comparison limits

The schedule uses 161 QAOA training calls and 97 VQE training calls per seed. All proposals, incumbent remeasurements, the initial call, and fresh final holdout count. Across the three seeds, the circuit protocol represents 473,856 QAOA shots and 579,072 VQE shots, totaling 1,052,928. The [resource ledger](../../assets/generated/optimization/portability/resources.json) maps every measured group and its basis changes, gates, and qubit readout. These protocol totals are not measured hardware consumption; the implementation generates counts classically.

The original four-vertex cycle and first Hamiltonian remain separate baseline cases. A smaller triangle uses fewer qubits and edges, but it is a different problem with a different objective maximum and target. The two Hamiltonians also have different coefficients and energy scales. The original runs vary initialization, and the new cases use separate random streams; comparing equal seed labels across experiments mixes input, initialization, and random-stream changes. Equal optimizer schedules do not turn their logical counts or empirical target outcomes into matched-problem physical improvements. No claim of quantum advantage, transferable convergence, or a general success rate follows from these comparisons.

The [complete recorded-trajectory resource replay](trajectory-resources.md) concerns the original tasks. It is not a physical projection for these new inputs. Portability results here contain logical workload and measured local CPU effort; their physical resources remain planned. CPU timings of simulation and classical control are kept separate from unmeasured quantum gate, reset, routing, and feedback latency.

## Reproduce and inspect

Use the same pinned Python 3.12 environment and Qiskit core, NumPy, and Matplotlib versions credited in the [parent chapter](../optimization.md). No new dependency is introduced.

```bash
python scripts/generate_optimization_portability.py
python scripts/execute_notebooks.py
```

The [portability notebook](../../notebooks/optimization_portability.ipynb) supplies the labeled outputs reused above. Saved outputs serve a static site without a live kernel. The [standalone implementation](../../assets/generated/optimization/portability/optimization.py), [generator](../../assets/generated/optimization/portability/generate_optimization_portability.py), and [MIT license](../../assets/generated/optimization/portability/LICENSE.txt) support reproduction from public downloads. The bounded generators and simulators retain their explicit limits.

- [Runs, fixed settings, seeds, and exact checks](../../assets/generated/optimization/portability/runs.json)
- [Every objective evaluation, CSV](../../assets/generated/optimization/portability/evaluations.csv)
- [Measured-group logical resource ledger](../../assets/generated/optimization/portability/resources.json)
- [Seed 7 QAOA measured circuit](../../assets/generated/optimization/portability/qaoa-seed7.qasm)
- [Seed 7 VQE all-Z circuit](../../assets/generated/optimization/portability/vqe-z-seed7.qasm)
- [Seed 7 VQE all-X circuit](../../assets/generated/optimization/portability/vqe-x-seed7.qasm)
- [Executable portability notebook](../../notebooks/optimization_portability.ipynb)

## References

```{bibliography}
```
