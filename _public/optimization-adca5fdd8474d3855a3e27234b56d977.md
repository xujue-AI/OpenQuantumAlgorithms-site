---
title: "Optimization and variational methods"
description: "Checked four-qubit QAOA and two-qubit VQE examples, with complete optimization traces and separate sampled readout."
downloads:
  - file: ../notebooks/optimization.ipynb
    title: "Executable notebook"
  - file: ../assets/generated/openquantum-code.zip
    title: "Python examples and tests"
---

QAOA alternates an objective-dependent phase with a mixer, while VQE changes a trial state to reduce its energy {cite:p}`opt_qaoa2014,opt_vqe2014`. Both are hybrid workflows: classical parameter search and repeated state preparations needed to evaluate an objective belong to the algorithm. Here we make those operations inspectable on two small problems.

:::{note} Implementation and evidence

The first examples on this page use explicit Qiskit circuits, exact statevector objectives, independent classical references, and separately seeded final-readout experiments. The [finite-shot optimization increment](optimization/finite-shot.md) adds parameter updates driven by sampled objectives on the same tasks. The downloadable traces record every objective evaluation and all shot use. They check these particular circuits and runs; they establish no large-instance convergence guarantee or quantum advantage.

The first resource record contains logical circuit inputs and the complete demonstrated workload. Arbitrary rotations remain unsynthesized. The finite-shot increment adds conditional physical projections of a fixed final readout. The [recorded-trajectory replay](optimization/trajectory-resources.md) maps all stored training and holdout circuits, while the [portability cases](optimization/portability.md) test changed inputs with fixed starts. Predictive physical estimates of a fresh hardware-driven adaptive optimization remain **planned**.
:::

## Two stated tasks

| Example | Classical input and requested output | Independent reference |
|---|---|---|
| QAOA | An unweighted four-vertex cycle; optimize expected cut value, then sample bitstrings and report the best cut observed. | Enumerate all 16 assignments and evaluate their cut sizes. |
| VQE | The explicit two-qubit Hamiltonian below; optimize a stated ansatz and estimate its energy. | Diagonalize its full $4\times4$ matrix and compare with a separate scalar energy formula. |

The exact optimization targets are $\langle C\rangle\ge3-10^{-8}$ for the depth-one cycle circuit and $E-E_0\le10^{-8}$ for VQE, in the units stated below. These are numerical targets for ideal expectations, separate from the finite-shot estimation errors. The primary run is the first predetermined initialization for each task. The other initializations test the same small optimizer; the primary result is not selected after comparing outcomes. Achieved quality, stopping reason, run times, and every evaluation appear in the [structured run record](../assets/generated/optimization/runs.json).

```{include} ../assets/generated/optimization/summary.md
```

## QAOA: MaxCut on a four-vertex cycle

Number the vertices $0,1,2,3$ and supply the edge list

$$
\mathcal E=\{(0,1),(1,2),(2,3),(0,3)\}.
$$

For an assignment $z=(z_0,z_1,z_2,z_3)$, the cut size and cost operator are

$$
C(z)=\sum_{(u,v)\in\mathcal E}[z_u\ne z_v],\qquad
C=\sum_{(u,v)\in\mathcal E}\frac{I-Z_uZ_v}{2}.
$$

The classical maximum is four, attained by the two alternating assignments. Qiskit wire `q[0]` is the least significant bit, so displayed strings read $z_3z_2z_1z_0$; the optimal strings are `0101` and `1010`. Enumeration evaluates the edge predicate directly rather than consulting the quantum objective routine.

Starting from $|+\rangle^{\otimes4}$, one QAOA layer applies

$$
|\psi(\gamma,\beta)\rangle=
e^{-i\beta\sum_j X_j}e^{-i\gamma C}|+\rangle^{\otimes4}.
$$

The optimizer **minimizes** $f(\gamma,\beta)=-\langle C\rangle$. Angles are radians. Our gate convention is $R_P(\alpha)=e^{-i\alpha P/2}$: the mixer uses `rx(2*beta)` on every vertex. For each edge, the phase circuit is `cx(u,v); rz(-gamma,v); cx(u,v)`. This realizes $e^{+i\gamma Z_uZ_v/2}$; the omitted identity contribution changes only the global phase. There is no dense-matrix oracle or unexplained access operation.

![](#opt-qaoa-circuit)

For this cycle and one layer, an independent expression for the expected cut is

$$
\langle C\rangle=2+\sin(4\beta)\sin(2\gamma).
$$

Thus $\gamma=\pi/4$, $\beta=\pi/8$ gives mean cut three, compared with the classical maximum four. The ratio $3/4$ describes an **expected cut** on this instance. It does not mean every draw has cut three. At these analytic angles, the probabilities of cut sizes zero, two, and four are $1/32$, $14/32$, and $17/32$. Each maximum-cut bitstring has probability $17/64$. The saved plot and run record report the corresponding numerical distribution for the fitted angles.

![](#opt-qaoa-probabilities)

All edges are diagonal in the computational basis. One all-Z measurement produces an assignment and evaluates all four edge contributions together. For a fixed optimized state with best-cut probability $p_\star$, $N$ independent ideal draws contain a maximum cut with probability $1-(1-p_\star)^N$. At the analytic angles, seven draws give success above $0.99$. This calculation assumes ideal preparations at those fixed angles and independent draws; it does not predict optimizer success or noisy-device performance.

## VQE: an inspectable two-qubit energy

Use dimensionless coefficients and the Hamiltonian

$$
H=-Z_0-Z_1+\frac12 X_0X_1.
$$

In basis order $|q_1q_0\rangle=|00\rangle,|01\rangle,|10\rangle,|11\rangle$, its matrix is

$$
H=\begin{pmatrix}
-2&0&0&1/2\\
0&0&1/2&0\\
0&1/2&0&0\\
1/2&0&0&2
\end{pmatrix}.
$$

Its eigenvalues are $-\sqrt{17}/2,-1/2,1/2,\sqrt{17}/2$, so the reference ground energy is $E_0=-\sqrt{17}/2\approx-2.061552813$. This is a stated teaching Hamiltonian, rather than a molecule or chemistry workload with uncounted integral preparation.

Prepare $|00\rangle$, apply `ry(theta,0)`, and then `cx(0,1)`. The resulting state is

$$
|\psi(\theta)\rangle=
\cos(\theta/2)|00\rangle+\sin(\theta/2)|11\rangle.
$$

![](#opt-vqe-circuit)

The independent scalar expectation is

$$
E(\theta)=-2\cos\theta+\frac12\sin\theta.
$$

Its minimum is the full matrix ground energy, attained at $\theta=-\arctan(1/4)$ modulo $2\pi$. This ansatz can represent the exact ground state of this particular Hamiltonian. The generator and notebook compare the circuit expectation, matrix eigenvalue, and scalar formula; this expressibility statement does not extend to a general Hamiltonian or ansatz.

![](#opt-vqe-energy)

Finite-shot energy evaluation uses two groups. In the all-Z group, one bitstring yields the **joint** value $-Z_0-Z_1$; its sample variance includes covariance of the Z terms. In the all-X group, apply H to both qubits before computational-basis measurement and score $\tfrac12X_0X_1$. Adding the group means estimates total energy. The measurement protocol requires a fresh ansatz preparation for each shot, and H basis changes and final measurements appear in the exports. The implementation obtains each group's statevector probabilities and draws its counts classically. Shot noise can put a sampled energy below $E_0$ even though the exact expectation obeys the variational bound.

## A complete, deterministic parameter search

The original optimizer is a small coordinate polling routine. It evaluates the initial point, then every negative and positive coordinate displacement in a fixed order. If the best candidate reduces the minimized objective by more than $10^{-12}$, it accepts that candidate; otherwise it halves the step. The initial step is $0.4$ radians. It stops when the step falls below $10^{-6}$ radians or after 128 sweeps, and records which condition stopped each run. Coordinates are wrapped modulo $2\pi$.

QAOA initial points are $(\gamma,\beta)=(0.2,0.1),(0.6,0.4),(-0.3,0.25)$. VQE initial angles are $0.6,-1.1,2.2$. Every optimization objective uses an exact ideal statevector, with zero measurement shots. The run record retains accepted points, rejected candidates, iterations, evaluation counts, parameters, elapsed objective time, and total classical optimization time. Wall-clock timings include the classical simulator and are machine-dependent; they are not hardware execution times.

![](#opt-convergence)

The primary fitted angles are then evaluated in separate ideal finite-shot experiments with seeds `7`, `19`, and `41`, using 2,048 shots **per measurement group per seed**. Samples are drawn classically from statevector probabilities; repeated physical circuits are not executed. These first-increment experiments do not feed sampled objectives back to the optimizer. They report counts, sampled objectives, exact expectations, sample standard errors, and achieved output quality. Changing initialization probes optimization variability; changing the sampling seed probes readout variability. The [next increment](optimization/finite-shot.md) records an actual sampled parameter search with fresh final validation. Both increments use ideal simulation and exclude device noise and error mitigation.

## Logical workload and the pending physical handoff

The [logical resource record](../assets/generated/optimization/logical-resources.json) accompanies actual measured OpenQASM 3 circuits. It records gate counts, depth, width, input preparation, all sampling repetitions, basis changes, and readout. The graph is a classical edge list and the Hamiltonian is three explicit Pauli terms; no data-loading oracle is assumed. Classical input parsing, physical reset and measurement time, and device routing are excluded. Reference calculations are included in the whole CPU workflow timer, but their separate time is unmeasured.

The first demonstrated workload contains six exact classical optimization runs with their entire evaluation traces, plus three seeded readout experiments for each primary state. The ideal readout protocol represents 6,144 four-qubit QAOA preparations and 12,288 two-qubit VQE preparations, including both measurement bases. The generated ledger weights the explicit circuits by these shot counts; the samples themselves are drawn classically from statevector probabilities. Exact optimization evaluations use zero shots. The [finite-shot increment](optimization/finite-shot.md) provides a separate measured trace and shot allocation for every sampled training call. Multiplying one final circuit by an optimizer iteration count would not establish that workload's convergence or accuracy.

The elementary exports retain H, CX, and one-qubit rotations. Arbitrary-angle RX, RY, and RZ gates are **not** synthesized into Clifford+T. The inputs support a later synthesis audit, but supply no validated T count, code distance, factory allocation, physical-qubit count, or quantum runtime. Coherent rotation approximation error and stochastic fault probability need separate budgets before a surface-code comparison. The existing [Grover baseline](grover.md#end-to-end-resource-analysis) remains a separate frozen example.

## Generate, reproduce, and inspect

From a source checkout with the pinned Python 3.12 environment:

```bash
python scripts/generate_optimization.py
python -m pytest tests/test_optimization.py
python scripts/execute_notebooks.py
```

The [notebook](../notebooks/optimization.ipynb) uses the same implementation as the artifact generator and provides the reusable outputs embedded above. Saved outputs serve a static website; no live Python kernel is required. To run the downloaded notebook separately, place the [standalone module](../assets/generated/optimization/optimization.py) beside it and install the [pinned environment](../requirements-lock.txt).

The QAOA generator accepts a supplied graph and layer angles; the VQE generator exposes its ansatz parameter. Materialization is capped at 100,000 unitary gates before optional final measurements; the ledger includes those measurements. QAOA exact statevector simulation defaults to a small configured bound with a hard cap of 16 qubits; exhaustive enumeration has its own hard cap of 20. Circuit export is separate from these classical computations. Larger exports do not imply affordable simulation or project the number of optimization evaluations needed at larger sizes.

The implementation and figures are original and licensed under [MIT](../assets/generated/optimization/LICENSE.txt). Circuit construction, export, and ideal statevectors use **Qiskit core 2.6.2**, [tagged source](https://github.com/Qiskit/qiskit/tree/2.6.2), [Apache-2.0 license](https://github.com/Qiskit/qiskit/blob/2.6.2/LICENSE.txt) {cite:p}`opt_qiskit262`. Classical arrays and diagonalization use **NumPy 2.5.4**, [tagged source](https://github.com/numpy/numpy/tree/v2.5.4), [BSD-3-Clause license](https://github.com/numpy/numpy/blob/v2.5.4/LICENSE.txt) {cite:p}`opt_numpy254`. The original plots use **Matplotlib 3.11.2**, [tagged source](https://github.com/matplotlib/matplotlib/tree/v3.11.2), [Matplotlib license](https://github.com/matplotlib/matplotlib/blob/v3.11.2/LICENSE/LICENSE) {cite:p}`opt_matplotlib3112`. Metadata records actual Python and package versions. The original parameter search does not depend on `qiskit-algorithms`.

- [Executed notebook](../notebooks/optimization.ipynb)
- [Complete optimization and finite-shot records](../assets/generated/optimization/runs.json)
- [Every objective evaluation, CSV](../assets/generated/optimization/evaluations.csv)
- [Logical resource inputs and exclusions](../assets/generated/optimization/logical-resources.json)
- [QAOA measured circuit](../assets/generated/optimization/qaoa.qasm)
- [VQE all-Z measured circuit](../assets/generated/optimization/vqe-z.qasm)
- [VQE all-X measured circuit](../assets/generated/optimization/vqe-x.qasm)
- [Original implementation](../assets/generated/optimization/optimization.py)
- [Artifact-generation script](../assets/generated/optimization/generate_optimization.py)

## End-to-end resource analysis

### Space/time scaling and empirical performance

A depth-$p$ QAOA circuit for a graph with $n$ vertices and $m$ edges uses $n$ prepared wires, $pm$ problem rotations, $pn$ mixer rotations and two CX gates per edge per layer before lowering. It finally measures all $n$ wires. Routing and parallelism depend on the graph and hardware; an $O(p(n+m))$ gate count does not specify physical depth. Multiply each parameter/group circuit by its actual shots and optimizer evaluation count. The optimizer's required evaluations and attained quality as $n$ grows are empirical quantities, not fixed by this circuit formula.

VQE space follows the chosen ansatz and Hamiltonian; time also includes ansatz depth, separately measured noncommuting groups, covariance-aware estimation, shots and parameter search. The two-qubit ansatz is a bounded model. Classical exact diagonalization/exhaustive MaxCut are reference computations with independent scale limits.

The published circuit-scaling exports and changed toy instances provide bounded logical/empirical evidence. Conditional physical reports cover specified final readouts and recorded trajectories at those sizes. They are not a physical scaling law, predicted future adaptive path or general optimizer guarantee.

### Complete workflows and conditional fault-tolerant analysis

The original exact workflows separate **694 exact statevector objective evaluations** from **18,432 ideal sampled readout outcomes**. The exact optimizer uses zero shots. The final samples model fixed-parameter circuit executions; they are not QPU measurements or a finite-shot optimizer trajectory. The subsequent sampled workflow and conditional resource replay are described below.

:::{include} ../assets/generated/optimization/summary.md
:::

The [run record](../assets/generated/optimization/runs.json) preserves initialization, optimizer sweeps, every accepted/rejected objective call, seeds, fitted parameters and CPU timings. The whole CPU workflow timer includes its toy reference checks, which are not separately timed; export/plot/file I/O is outside that timer.

The [logical ledger](../assets/generated/optimization/logical-resources.json) keeps exact evaluation circuits separate from modeled shot-weighted readout. QAOA measures the joint cut score in the computational basis. VQE shares shots between its Z terms, retains their covariance, and uses an independent X-basis group. Preparation, basis changes, each circuit and individual qubit readouts are included in the declared logical workload.

A physical projection must account for each parameter setting and measurement group, synthesis choices, all repetitions, classical-control latency and achieved output quality. Multiplying a final ansatz by the number of exact optimizer calls does not establish a real finite-shot trajectory. The [finite-shot trajectories](optimization/finite-shot.md) now record that layer: 774 training calls plus six independent holdouts, totaling 1,052,928 ideal modeled preparations. The [trajectory replay](optimization/trajectory-resources.md) maps all 1,074 measurement groups to exact circuit/source hashes and conditional resource queries. The [second-instance cases](optimization/portability.md) separately test a triangle and an asymmetric two-qubit Hamiltonian with fixed starts.

The shared rotation adapter supplies conditional mean-fit cost projections, not certified synthesis or an optimizer-convergence guarantee. A confidence statement for one fixed bounded scalar mean does not silently cover adaptive parameters or a noncommuting energy sum. Any later projection must preserve those distinctions and the [fault-tolerance appendix](../appendices/fault-tolerance.md)'s separate budgets.

[Implementation](optimization.md) · [Verification](#verification-and-formalization) · [Frozen illustrative hardware profiles](grover.md#end-to-end-resource-analysis)

## Verification and formalization

The original exact workflow has 40 dedicated checks. Additional sampled, projection, replay and second-instance checks exercise the extended workflows; each notebook is freshly executed before publication. QAOA is checked against exhaustive classical cut enumeration and independently assembled dense evolution, including asymmetric multiple-layer angles. Tests verify cost signs, angle conventions, bit ordering and complete evaluation traces.

The two-qubit VQE example is checked against the full Hamiltonian spectrum and an independent analytic energy expression. Its ansatz contains this particular toy ground state; that does not imply general VQE expressibility or convergence. All optimizer starts, rejected proposals, stopping conditions and objective calls are recorded.

Readout tests retain the covariance of jointly measured Z terms, independent X/Z streams, reproducible counts, bounded-outcome Hoeffding calculations and shot-weighted gate totals. Sample standard errors are descriptive; the stated ideal bounds are per fixed estimate, not simultaneous guarantees across every seed or optimizer parameter.

The chapter explicitly distinguishes exact-statevector optimization from ideal sampled final validation. Actual ideal finite-shot trajectories are now recorded. Conditional projections replay their fixed measured paths; a hardware/noisy optimizer trajectory, certified synthesis and formal proof remain unestablished. A separate 32-qubit generation run verifies export/counts without simulation or optimization.

[Executed notebook](../notebooks/optimization.ipynb) · [Run records](../assets/generated/optimization/runs.json) · [Public code and tests](../assets/generated/openquantum-code.zip) · [Resource boundaries](#end-to-end-resource-analysis)

See the [mathematics appendix](../appendices/mathematics.md) for spectra and confidence, and the [verification policy](../verification.md) for evidence scopes.

## References

```{bibliography}
```
