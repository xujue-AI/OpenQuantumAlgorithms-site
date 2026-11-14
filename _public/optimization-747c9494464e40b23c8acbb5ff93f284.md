---
title: "Quantum optimization"
description: "QAOA, VQE, and digitized adiabatic optimization, with explicit circuits, complete workloads, and classical C4 reference coverage."
---

## Motivation and intuition

Optimization asks for a useful classical answer: a good assignment, an energy estimate, or a verified solution. For $n$ binary variables there are $2^n$ assignments. Quantum algorithms prepare amplitudes over candidates and change them through objective-dependent phases, interference, or Hamiltonian evolution. Measurement still returns a limited classical result. Input construction, state preparation, output quality, repetitions, and classical control therefore determine whether a proposed improvement is meaningful.

This chapter checks transparent small circuits for three approaches and adds an [executed fixed DQI circuit and public companion](optimization/dqi.md). QAOA and VQE use a classical parameter search; a digitized anneal follows a prescribed Hamiltonian path. The original algorithm papers provide the theoretical context {cite:p}`opt_qaoa2014,opt_vqe2014,opt_adiabatic2000`. The implementations, exposition, and figures here are original.

:::{note} Evidence and scope

The demonstrations are executed ideal numerical checks with independent classical references and downloadable circuits. Sample counts are generated classically from circuit probabilities, rather than measured on a QPU. Small results and parameterized generators do not establish large-instance convergence or quantum advantage.

Fault-tolerant reports are conditional resource models. **Classical C4 reference coverage only.** The reproduced Lean artifact checks finite cut values and the stated classical equality predicate; its coverage excludes QAOA and annealing correctness and physical estimates. These distinctions remain explicit below.
:::

## Fundamental algorithms and requested outputs

| Approach | How the state changes | What is checked here |
|---|---|---|
| QAOA | Alternate phases from a classical objective with a mixer; fit layer angles. | Four-vertex MaxCut, exact enumeration, and separate exact/sampled optimizer records. |
| VQE | Prepare a parameterized ansatz and reduce its measured energy. | A stated two-qubit Hamiltonian, exact diagonalization, and covariance-aware grouped readout. |
| Adiabatic optimization | Change an easy driver Hamiltonian into a problem Hamiltonian; digitize the evolution. | A fixed C4 path, numerical continuous reference, and separate time/slice studies. |
| [Decoded quantum interferometry](optimization/dqi.md) | Coherently decode a low-weight syndrome state, then Fourier-transform it into an objective-biased distribution. | Nine-wire default circuit and ordinary saved-artifact notebook executed. Larger families and physical estimates remain planned. |

An expected cut, a probability of obtaining a maximum cut, and an energy error are different outputs. A fair comparison fixes the problem, required quality, confidence, and physical assumptions. The small cases below demonstrate behavior; they are not interchangeable benchmarks.

## QAOA: phases, mixing, and a small MaxCut circuit

For a graph with $n$ vertices and $m$ unweighted edges, write

$$
C(z)=\sum_{(u,v)}[z_u\ne z_v],\qquad
C=\sum_{(u,v)}\frac{I-Z_uZ_v}{2}.
$$

QAOA prepares $|+\rangle^{\otimes n}$ and alternates $e^{-i\gamma_jC}$ with $e^{-i\beta_j\sum_kX_k}$ for $p$ layers. The phase changes amplitudes' relative phases according to cut value; the mixer converts those phases into a different outcome distribution. The optimizer minimizes $-\langle C\rangle$. With rotations defined by $R_P(\alpha)=e^{-i\alpha P/2}$, each edge uses `CX; RZ(-gamma); CX` and each mixer uses `RX(2*beta)`.

The checked cycle has edges $(0,1),(1,2),(2,3),(0,3)$. Enumerating all 16 assignments gives maximum cut four at `0101` and `1010`, with strings ordered $q_3q_2q_1q_0$. For one layer,

$$
\langle C\rangle=2+\sin(4\beta)\sin(2\gamma).
$$

At $\gamma=\pi/4,\beta=\pi/8$, the mean cut is three and the maximum-cut probability is $17/32$. An expected approximation ratio $3/4$ is consequently distinct from per-draw success.

![](#opt-qaoa-circuit)

The [QAOA method page](optimization/qaoa.md) develops the circuit, analytic check, probability distribution, and generator. General angle fitting is heuristic: the measured convergence of these tiny instances does not bound the effort needed on a larger graph.

## VQE: a variational energy and explicit measurements

VQE minimizes $E(\theta)=\langle\psi(\theta)|H|\psi(\theta)\rangle$. For a normalized state this is at least the ground energy, but a restricted ansatz or an unsuccessful optimizer can prevent reaching it. Estimating the expectation also requires enough repetitions of its measurement groups {cite:p}`opt_vqe2014`.

Our first input is

$$
H=-Z_0-Z_1+\tfrac12X_0X_1.
$$

`RY(theta)` on qubit zero followed by `CX(0,1)` prepares $\cos(\theta/2)|00\rangle+\sin(\theta/2)|11\rangle$. Its independently calculated energy is $-2\cos\theta+\tfrac12\sin\theta$, whose minimum matches the full matrix ground energy $-\sqrt{17}/2$. This particular ansatz contains this particular ground state; it is not a general two-qubit ansatz.

![](#opt-vqe-circuit)

One all-Z group jointly scores $-Z_0-Z_1$, retaining covariance between the terms. A separate all-X group applies H to both qubits and scores $\tfrac12X_0X_1$. A sampled energy can lie below the ground energy through shot noise even though the exact expectation satisfies the variational bound. The [VQE method page](optimization/vqe.md) gives the matrix reference, angle conventions, grouping, and full run contract.

### Exact fitting and sampled decisions

The first variational run uses deterministic coordinate polling with three fixed initializations, every exact-statevector objective recorded, and separate 2,048-shot-per-group final readout for seeds `7`, `19`, and `41`. The ideal targets are expected cut at least $3-10^{-8}$ and VQE ground-energy gap at most $10^{-8}$.

```{include} ../assets/generated/optimization/summary.md
```

The [finite-shot workflow](optimization/finite-shot.md) instead makes each accepted or rejected move from fresh sampled objectives. Its six recorded runs use 780 objective calls including final holdouts, 1,074 measured group batches, and 1,052,928 modeled preparations. Its repeated two-standard-error acceptance margin is descriptive, not a calibrated convergence test. Fresh final holdouts have separate fixed-expectation confidence bounds.

The [portability cases](optimization/portability.md) reuse the schedule on a triangle and a second stated Hamiltonian, fixing starts while varying seeds. Their changed inputs and quality targets prevent a same-problem resource-improvement claim.

The executed [quality-and-workload study](optimization/finite-shot.md#quality-and-workload-across-sizes) extends this to paths/cycles of three to eight vertices at QAOA depths one and two, and a shallow chain-VQE ansatz on two to six qubits. Its locked recipe retains all 72 runs, including 41 QAOA and eight VQE target misses in the independent post-training audits. The 84 fresh holdout groups have a simultaneous confidence allocation; expectation precision and selected-state quality are assessed separately.

```{include} ../assets/generated/optimization/quality-scaling/overview.md
```

![](#opt-quality-scaling-quality)

More layers and larger parameter vectors consume more sampled calls under the fixed sweep schedule. These results therefore show size-dependent achieved quality under explicit workloads, rather than equal-budget improvement or general convergence. The six-spin VQE runs also end above the recorded zero-state energy baseline; those outcomes stay in the record.

The executed [initialization and finite-shot diagnostic study](optimization/finite-shot.md#initialization-and-sampling-diagnostics) fixes 65,648 training preparations for each of 36 sampled runs on C4/C6 QAOA and four-/six-spin VQE, with 24 separately labeled deterministic CPU controls. It preserves the earlier targets and records; equal training preparation counts leave different circuit and readout workloads. Both depth-one cycles have an analytic ratio ceiling $0.75$, below the preserved $0.80$ target.

```{include} ../assets/generated/optimization/diagnostics/overview.md
```

The [actual Clifford+T trajectory replay](optimization/qaoa.md#complete-clifford-t-workload) has compiled all 54 distinct circuits for the first holdout-supported four-wire QAOA diagnostic run and computed its complete conditional instrument workload. All 110 groups and 82,032 preparations remain charged, requiring 182,646,240 actual T states. The 9,614,924-byte bundle preserves an initial representation-limit failure and its bounded recovery. Physical timing, encoder/factory internals and geometry remain model assumptions; the compiled replay does not establish that changed synthesis or hardware noise would reproduce the optimizer's recorded decisions. The VQE inconclusive labels remain unchanged.

## Adiabatic optimization: follow a path, then digitize it

For the same C4 MaxCut task, choose

$$
H_D=-\sum_jX_j,\qquad H_P=\tfrac12\sum_{(u,v)}Z_uZ_v=\tfrac m2I-C,
$$

$$
H(s)=(1-s)H_D+sH_P,\qquad s=t/T.
$$

The easy driver ground state is $|+\rangle^{\otimes4}$. A sufficiently suitable path and evolution time can transfer weight toward the problem ground space. Adiabatic reasoning depends on spectral and path assumptions; it gives no generic polynomial runtime guarantee for arbitrary optimization instances {cite:p}`opt_adiabatic2000`.

The implementation uses midpoint Strang slices. With $\Delta t=T/L$, each slice applies driver-half `RX(-dt*(1-s))`, then an explicit `CX; RZ(dt*s); CX` interaction per edge, then the second driver half. The source has $n$ H, $2nL$ RX, $mL$ RZ, and $2mL$ CX before final measurement.

The fixed main run is **$T=16,L=256$**, selected before the studies. Its ideal maximum-cut probability is approximately **0.987022729**, meeting the predeclared target $0.95$. $T$ is in inverse-energy units with $\hbar=1$, not hardware nanoseconds. The following diagram shows **two slices** for readability; the exported main circuit contains all 256.

![](#opt-annealing-circuit)

A high-accuracy SciPy DOP853 numerical integration supplies the continuous-evolution reference. A separate dense Strang implementation checks the circuit construction. Varying $T$ probes ideal path quality; varying $L$ probes digitization error relative to the continuous numerical reference. Those effects are reported separately.

![](#opt-annealing-convergence)

The full gap vanishes at the endpoint because two optimal cuts are degenerate. The initial state lies in the preserved global-bit-flip $G=X^{\otimes4}$ sector with eigenvalue $+1$. A 201-point sector-gap scan is a numerical diagnostic, not a lower-gap certificate or a runtime theorem. The [adiabatic method page](optimization/adiabatic.md) contains that scan, the path contract, independent-reference tolerances, readout, and scale limits.

## Decoded quantum interferometry: optimization through decoding

[DQI](optimization/dqi.md) adds a recent approach that connects optimization to
coding theory {cite:p}`frontier_dqi2025`. Its quantum circuit must contain the
reversible decoder as well as state preparation, syndrome computation and
Fourier interference. A classical decoder runtime is not that circuit's
fault-tolerant cost. The method page distinguishes the paper's algebraically
structured result from sparse max-XORSAT experiments, where a tailored
classical solver beats the showcased DQI result. The fixed nine-wire coherent
circuit and ordinary public saved-artifact notebook have executed. The earlier
combined attempt remains incomplete after its notebook timeout; the later
notebook run checks preserved evidence without repeating the quantum science.
Scalable decoder families and full physical estimates remain **planned**.

## End-to-end resource analysis

### Space and time as the input grows

Let $n$ be the objective's data-qubit count, $m$ its graph edges, $p$ QAOA layers, $L$ annealing slices, and $A(n,\ell)$ a chosen VQE ansatz's gate count. These are **source-construction scalings**, separate from synthesis, scheduling, and measured larger-instance behavior.

| Construction | Logical width | Source gates for one preparation | Remaining accuracy/workload dependence |
|---|---|---|---|
| QAOA | $n$ | $n$ H, $mp$ RZ, $np$ RX, $2mp$ CX: $O(n+p(n+m))$. | Number of optimizer evaluations, groups, repetitions, and shots; no general convergence bound here. |
| VQE | Ansätze-dependent; two qubits in this demo. | $A(n,\ell)$ plus input preparation and basis changes. | Hamiltonian terms/groups, variance and accuracy, ansatz expressibility, and optimizer work. |
| Digitized anneal | $n$ | $O(L(n+m))$ with the explicit driver and edge gates above. | Path time, digitization tolerance, readout, and gap/path assumptions. |

For an entire hybrid run, sum the actual group circuits weighted by **every** call's shots, including rejected candidates and final validation. Classical optimizer and postprocessing effort must be added separately. At one fixed parameter setting, a bounded score of range width $W$ needs $N\ge W^2\log(2/\delta)/(2\epsilon^2)$ shots for a two-sided Hoeffding error radius $\epsilon$. This fixed-mean relation does not certify an adaptive search or determine its evaluation count.

The available checks use CPU Qiskit Statevector and bounded NumPy/SciPy references. A statevector with $w$ total qubits stores $2^w$ amplitudes; a dense operator stores $4^w$ entries. Exact MaxCut enumeration examines $2^n$ assignments. The generators have independent materialization guards: annealing permits $n\le32,T\le64,L\le4096$ subject to 100,000 unitary gates; its statevector helper defaults to 12 qubits with a hard cap of 16, while dense/ODE checks cap at eight and both simulation paths require $T\le32$. A larger export is not evidence of affordable simulation or successful optimization.

### Fault-tolerant resources and complete workloads

Logical gate counts are circuit inputs to a physical model. The fixed annealing report includes **12,288 complete measured preparations**, with a whole-method physical-fault allowance of $10^{-3}$. The frozen scenarios use physical error rates $10^{-3}$ and $10^{-4}$, **100 ns gate time** and **500 ns measurement time**.

The main annealing input is fixed at $T=16,L=256$, with the same preparation and all-Z readout for every execution. Its source contains 3,072 arbitrary rotations; shared lowering merges adjacent driver rotations and the canonical estimator trace has 2,052 RZ rotations and four measurements. The model includes surface-code and factory costs under the recorded assumptions.

| Physical error | Declared selection | Preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-readout physical-fault score |
|---:|---|---:|---:|---:|---:|
| 0.001 | fewest physical qubits | 12,288 | 1,317,615 | 8502.067200 | 0.000294 |
| 0.001 | shortest modeled runtime | 12,288 | 1,422,975 | 7821.901824 | 0.000432 |
| 0.0001 | fewest physical qubits | 12,288 | 131,595 | 8502.067200 | 0.000278 |
| 0.0001 | shortest modeled runtime | 12,288 | 260,115 | 3740.909568 | 0.000431 |

These rows use the entries labeled `annealing` in the frozen report. The serial quantum-model time excludes unmeasured physical initialization/reset, classical input/compilation/scoring, routing, communication, and feedback. It is distinct from the Hamiltonian evolution parameter $T=16$.

The [frozen physical report](../assets/generated/optimization/annealing-walk-resources/report.json) contains code distances, factories, Pareto points, budgets, and provenance; the [summary CSV](../assets/generated/optimization/annealing-walk-resources/summary.csv) provides the declared selections. Both retain the earlier search-control records alongside the annealing entries. Circuit inputs remain inspectable:

| Fixed method | Source QASM | Lowered QASM | Actual estimator input |
|---|---|---|---|
| Annealing | [Source](../assets/generated/optimization/annealing-walk-resources/annealing-source.qasm) | [Lowered](../assets/generated/optimization/annealing-walk-resources/annealing-lowered.qasm) | [Estimated input](../assets/generated/optimization/annealing-walk-resources/annealing-estimated-source.qasm) |

Annealing rotations use a per-execution synthesis-model allowance $10^{-5}$ under the audited QDK expected-cost fit {cite:p}`opt_qdk1340,opt_rotation2022`. It is not a stochastic failure probability, a concrete synthesized sequence, a worst-case cost bound, or a certificate of cut-probability error.

The [complete variational trajectory replay](optimization/finite-shot.md#recorded-trajectory-resources) maps all stored batches to source and lowered QASM. Replay fixes those stored angles; it does not predict a new hardware-driven adaptive path. Native cost grouping preserves the pinned cost-model inputs while excluding only audited CPU `EVALUATION_TIME` instrumentation. The default OpenQASM importer first builds a LogicalCounts-derived canonical trace, which already abstracts original gate topology. Equal model signatures do not certify original routing, gate ordering, or unitary equivalence at different angles.

For the size study, the complete source/lowered mapping covers all 5,859,840 preparations across 72 recorded runs. The original 64-search result remains in an [unchanged archive](../assets/generated/optimization/quality-resources-cap64/manifest.json). A [computation-only amendment](../assets/generated/optimization/quality-resources/amendment.json) raised the limit to 96 before searches while preserving every case, seed, target, shot, profile, and allowance. The amended [physical replay](../assets/generated/optimization/quality-resources/manifest.json) completed all 85 physical-estimator searches for 84 classes plus the cost-consistency probe. These conditional aggregates include every target miss and use independently selected class configurations, with common hardware configuration and reset/feedback/reconfiguration time unverified.

```{include} ../assets/generated/optimization/quality-resources/summary.md
```

Serial time is repetition-weighted and peak footprint is the maximum selected footprint. CPU control, physical reset, communication, feedback, routing, and reconfiguration overhead are excluded where unmeasured. Independently selected model points do not certify one common physical layout. Reports expose these assumptions and preserve the [Grover baseline](grover.md#end-to-end-resource-analysis); different tasks do not establish a co-design improvement at matched output quality.

## Verification and formalization

Numerical coverage includes classical enumeration and scalar expectation formulas for QAOA, dense Hamiltonian eigenvalues for VQE, gate-versus-dense Strang checks and a continuous numerical reference for annealing. Export/lowering checks compare the implemented circuits up to global phase with measurement maps retained. Finite-shot bounds apply to fresh fixed-state estimates and remain separate from optimizer success and physical faults. The size study additionally checks all graph optima by enumeration, all chain ground energies by dense diagonalization, selected-state circuits against independent references, and every measured-decision trace from its counts. Its simultaneous 84-group holdout bounds do not cover adaptive training queries or extend the classical C4 proof.

**Classical C4 reference coverage only.** The finite reference has been reproduced in **Lean 4 v4.34.0-rc2**, using an explicitly pinned existing toolchain {cite:p}`opt_lean434rc2`.

```{include} ../assets/generated/optimization/formal/summary.md
```

Its checked declarations establish that every `Fin 16` cycle assignment has cut at most four, that the optimum occurs exactly at integers 5 or 10, and the stated classical marked-equality definition. This does not connect the Lean definitions to the Python implementation or prove QAOA or annealing circuits, quantum amplitudes, reversible uncomputation, convergence/gaps, sampling, or fault tolerance. [Download the proof source](../assets/generated/optimization/formal/C4Reference.lean) and [verification record](../assets/generated/optimization/formal/verification.json) to inspect the exact coverage.

## Reproduce, packages, and planned extensions

Use the pinned Python 3.12 environment and execute notebooks in fresh kernels:

```bash
python scripts/generate_optimization.py
python scripts/generate_optimization_sampled.py
python scripts/generate_optimization_portability.py
python scripts/generate_optimization_annealing.py
python scripts/execute_notebooks.py
```

The original code and figures are MIT-licensed. Qiskit core **2.6.2** supplies circuits, exports, and ideal statevectors ([source](https://github.com/Qiskit/qiskit/tree/2.6.2), [Apache-2.0](https://github.com/Qiskit/qiskit/blob/2.6.2/LICENSE.txt)) {cite:p}`opt_qiskit262`. NumPy **2.5.4** supplies classical arrays and diagonalization ([source](https://github.com/numpy/numpy/tree/v2.5.4), [BSD-3-Clause](https://github.com/numpy/numpy/blob/v2.5.4/LICENSE.txt)) {cite:p}`opt_numpy254`. SciPy **1.18.1** supplies the DOP853 numerical reference ([source](https://github.com/scipy/scipy/tree/v1.18.1), [BSD-3-Clause](https://github.com/scipy/scipy/blob/v1.18.1/LICENSE.txt)) {cite:p}`opt_scipy1181`. Matplotlib **3.11.2** renders the original figures ([source](https://github.com/matplotlib/matplotlib/tree/v3.11.2), [license](https://github.com/matplotlib/matplotlib/blob/v3.11.2/LICENSE/LICENSE)) {cite:p}`opt_matplotlib3112`. Reports record actual versions, source hashes, parameters, seeds, and commands.

The [variational](../notebooks/optimization.ipynb), [sampled](../notebooks/optimization_sampled.ipynb), [portability](../notebooks/optimization_portability.ipynb), and [annealing](../notebooks/optimization_annealing.ipynb) notebooks are downloadable. Each method page or generated record links matching standalone code, circuit exports, and licenses. Saved outputs serve the static site without a public kernel. Physical projection commands require the shared source-checkout resource adapter and audited QDK environment; their invocation and inputs are recorded separately.

Richer VQE ansätze, larger matched-instance optimizer studies, schedule design with certified spectral assumptions, optimization-specific walk objective predicates and maximum finding, a formal bridge to executable circuits, and predictive physical adaptive execution remain **planned**.

## References

```{bibliography}
```
