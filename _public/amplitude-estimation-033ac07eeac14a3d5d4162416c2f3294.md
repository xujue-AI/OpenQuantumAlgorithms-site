---
title: "Amplitude estimation"
description: "Explicit threshold counting, phase-estimation circuits, independent checks and conditional surface-code costs."
downloads:
  - file: ../notebooks/amplitude_estimation.ipynb
    title: "Executable notebook"
  - file: ../assets/generated/openquantum-code.zip
    title: "Python examples and tests"
---

Amplitude estimation turns a flag's probability into a coherent circuit's rotation angle. Instead of observing the flag repeatedly, the fundamental algorithm applies controlled amplification steps and estimates their eigenphases {cite:p}`prim_amplitude_2002`. This is useful when sample preparation or predicate evaluation is expensive and coherent access and inverse access are available. Those operations are explicit parts of the circuit here.

:::{note} Implemented scope

Standard phase-estimation amplitude estimation, preparation/inverse preparation, a threshold predicate, both reflections, complete controlled powers, final measurements, and an ideal median-confidence contract are implemented. Bounded circuits and their primitive lowering are checked against independent analytic distributions; a separately labeled classical categorical experiment checks median coverage.

The resource section contains a **conditional mean-fit surface-code projection**. There is no generated Clifford+T synthesis certificate or hardware amplitude-confidence guarantee. A separate [branch-adaptive amplitude-estimation companion](amplitude-estimation/iterative.md) implements fresh sampled batches and exact conservative interval arithmetic. Practical application predicates and formal Lean coverage remain planned.
:::

## Task and intuition

Given a preparation circuit

$$
A|0\rangle=\sqrt{1-a}|\psi_0\rangle|0\rangle+
\sqrt a|\psi_1\rangle|1\rangle,
$$

estimate $a$ with a stated absolute error and confidence. Write $a=\sin^2\theta$. Reflection about the prepared state followed by reflection of the good subspace rotates their two-dimensional span by $2\theta$. Its two eigenphases are $\pm\theta/\pi$ in units of a full turn. Both decode to the same probability through $\sin^2(\pi\phi)$.

Our classical input is an $n$-bit uniform register and threshold $k$; the reversible flag predicate is $[x<k]$. The independent answer is $a=k/2^n$. Because that threshold probability is already known, this is an implementation/accounting example with no application advantage. A practical integration, finance or counting problem needs its own reversible predicate and loading cost.

The comparator toggles the flag when the first bit differing from the threshold is zero and the threshold bit is one. At most $n$ disjoint prefix terms use X and multi-controlled-X, preserving all data bits with no work register or enumerated truth table. `q[0]` is the least significant data bit; `q[n]` is the flag.

## Fundamental algorithm and small circuit

Let $S_0=I-2|0\rangle\langle0|$ and $S_g=I-2\Pi_g$. Our iterate is

$$Q=-A S_0 A^\dagger S_g.$$

Chronologically it applies $S_g,A^\dagger,S_0,A$, and a global minus sign. **That sign matters once $Q$ is controlled.** Dropping it shifts the estimated phase and gives a wrong amplitude; the $a=1$ test catches this error. $S_0$ is actual X/MCZ/X, with MCZ lowered through H/MCX/H; $S_g$ is Z on the flag.

Prepare $A|0\rangle$, add $m$ phase qubits in $|+\rangle$, apply controlled $Q^{2^j}$ from wire $j$, then inverse QFT and phase measurement. Each power is actual repeated $Q$ access, rather than an opaque matrix. For $M=2^m$, output $y$ decodes as

$$\widehat a(y)=\sin^2(\pi y/M).$$

The default uses two data qubits, threshold one, a flag and four phase qubits: seven logical qubits, $a=1/4$, one initial preparation and 15 controlled iterates.

```{embed} #prim-amplitude-preparation
```

```{embed} #prim-amplitude-circuit
```

The phase law is the equal mixture of independent geometric-sum QPE laws at $\phi=1/6$ and $5/6$. Peaks $y=3,13$ each have probability about $0.344269$ and decode to $0.308658284$, an error about $0.0586583$. The mean over all decoded outcomes is $0.296875$; this coarse estimator is not unbiased.

```{embed} #prim-amplitude-distribution
```

## Accuracy, repetitions and classical comparison

BHMT Theorem 12 gives single-run success at least $p_0=8/\pi^2$ for

$$|\widehat a-a|\le 2\pi\sqrt{a(1-a)}/M+\pi^2/M^2.$$

Our answer-independent contract uses $\epsilon_M=\min(1,\pi/M+\pi^2/M^2)$. At $m=4$ it is **0.234902683**, so the example deliberately has coarse precision. The bound can be conservative or trivial at very small $m$.

Run the complete circuit independently an odd number $R$ of times and take the median of decoded outcomes. If a majority lies in the interval, its median does too. A binomial majority-tail bound is

$$\sum_{j=(R+1)/2}^{R}{R\choose j}(1-p_0)^j p_0^{R-j}.$$

Requested **ideal** failure $0.005$ needs $R=15$ under this bound; the value is about $0.00296846$. The generator also computes the exact ideal interval-failure probability for this discrete distribution and checks 10,000 classical categorical ensembles at seed 8127. These draw from the exact ideal law; physical circuits are not run.

Classical Bernoulli sampling of the **same preparation and flag** needs $N\ge\log(2/\delta)/(2\epsilon_M^2)$ shots by Hoeffding. At equal $\epsilon_M$ and $\delta=0.005$, this is 55 preparations/readouts. The quantum proposal uses 15 full QPE runs including 465 predicate or inverse-predicate calls. Query complexity does not imply a lower compiled cost here, and the known threshold already gives an exact classical shortcut.

For constant success probability, standard amplitude estimation needs $O(1/\epsilon)$ amplification queries and $O(n+\log(1/\epsilon))$ live qubits. This independent median multiplies work by $O(\log(1/\delta))$. The QFT costs $O(m^2)$ controlled phases. Each query contains preparation, its inverse and both reflections. Its primitive cost depends on the comparator controls and pinned compiler; the compiled ledger assigns no oracle unit cost.

## Variants and reusable components

| Method | Coverage | Tradeoff |
|---|---|---|
| Standard QPE amplitude estimation | Implemented explicit threshold access, exact-sign $Q$, QFT, decoding and median contract. | Coherent depth grows with $M$; adds $m$ phase qubits. |
| Classical flag sampling | Matching precision/workload formula for the same preparation. | Simpler measured circuits and $O(\epsilon^{-2})$ sampling bound. |
| [Branch-adaptive amplitude estimation](amplitude-estimation/iterative.md) | Implemented fresh amplified-flag batches, unambiguous branch inversion and conservative interval updates. | Removes the phase register; full sampled paths are charged, and caps may return inconclusive. |
| Application-specific predicates | Planned. | Loading, reversible arithmetic and output interpretation must be built and charged. |

The [QFT/QPE chapter](qft-qpe.md) implements conventional and iterative phase estimation. Iterative QPE alone does not implement an iterative amplitude-confidence procedure. The threshold comparator and Grover operator here are reusable independently of the capped simulator.

## End-to-end resource analysis

The default measured circuit lowers to 2,529 CX, 1,234 RZ, 210 RX, 47 RY, 1,083 T, 813 T-dagger, 504 H, two X and four measurements: **6,426 operations**, dependency depth **5,151**. Arbitrary rotations remain unsynthesized.

One complete run includes one uncontrolled $A$, 15 controlled $A$, 15 controlled $A^\dagger$, both controlled reflections 15 times, all 15 controlled minus signs and the inverse QFT. The 15-run ideal median proposal includes 225 controlled iterates, **465 predicate/inverse calls**, 60 phase measurements and 96,390 primitive operations. Data and flag are discarded after phase readout. Fresh zero-state input is assumed; physical initialization/reset faults and classical preparation/decoding/compilation times are unmeasured.

The [scaling CSV](../assets/generated/amplitude-estimation/scaling.csv) contains 16 **actually compiled** rows over $n=1,2,3,4$ and $m=2,3,4,5$. Extra rows are not claimed to be simulated. Selected counts show input access matters:

| $n$ | Threshold | $m$ | Live qubits | $Q$ calls/run | Operations/run | All-run operations ($R=15$) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4 | 6 | 15 | 1,013 | 15,195 |
| 2 | 1 | 4 | 7 | 15 | 6,426 | 96,390 |
| 3 | 2 | 4 | 8 | 15 | 8,875 | 133,125 |
| 4 | 5 | 4 | 9 | 15 | 40,502 | 607,530 |
| 2 | 1 | 5 | 8 | 31 | 13,233 | 198,495 |

The [QDK report](../assets/generated/amplitude-estimation/physical-resources.json) projects **15 complete default static runs**, with illustrative physical error $10^{-4}$, 100 ns gates and 500 ns measurements. A whole-workload physical-fault allowance $0.005$ is split among the runs. A separate per-run synthesis-model allowance $10^{-4}$ selects the audited mean-fit rotation cost.

| Peak physical qubits | Data code distance | Runtime/run | Serial runtime, 15 runs |
|---:|---:|---:|---:|
| 23,003 | 11 | 204.82 ms | 3.0723 s |
| 25,043 | 9 | 167.58 ms | 2.5137 s |
| 29,391 | 7 | 130.34 ms | 1.9551 s |

The [bounded physical scaling report](../assets/generated/amplitude-estimation/surface-code-scaling.json) also executes four additional static QDK queries. Every row uses the same $R=15$, ideal failure target $0.005$, physical-fault allowance $0.005$, and separate per-run synthesis-model allowance $10^{-4}$. Each retains its full accuracy contract and Pareto frontier. Selecting the **fastest returned point** gives:

| $n$ | Threshold | $m$ | Ideal absolute-error bound | Peak physical qubits at fastest point | Serial runtime, 15 runs |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4 | 0.234903 | 81,480 | 0.041625 s |
| 2 | 1 | 3 | 0.546912 | 29,100 | 0.870240 s |
| 2 | 1 | 4 | 0.234903 | 29,391 | 1.955100 s |
| 2 | 1 | 5 | 0.107813 | 60,385 | 4.032105 s |
| 3 | 2 | 4 | 0.234903 | 27,645 | 2.821560 s |

Factory allocation and code distance can differ between independently selected points, so peak qubits need not grow monotonically across this table. In particular, the one-data-bit fastest point spends more qubits for a short runtime; its smallest-qubit alternative has 11,060 physical qubits and runtime 0.074925 s. Additional rows are computed resource models, not additional statevector or device experiments. These bounded samples supply no physical asymptotic complexity or advantage theorem.

These are **conditional mean-fit resource projections** from QDK 1.34.0, including surface-code layout and magic-state factories. Its PSSPC formula fits mean fallback work for random angles; it supplies neither a concrete per-angle synthesis certificate nor a runtime-tail guarantee, and abstracts Clifford scheduling. Combining the ideal and physical allowances therefore does not establish a hardware amplitude interval: certified synthesis and its effect on the phase law are still required. Classical comparator construction, compilation, median sorting and external input loading timings are excluded. The separate [Grover baseline](grover.md) stays frozen.

## Verification and formalization

Tests compare every threshold for one through four data bits against an independently specified reversible permutation, including both incoming flag values and endpoint thresholds. They check $A|0\rangle$ against a flagged uniform vector and $Q$ against $(2|\psi\rangle\langle\psi|-I)S_g$, keeping the exact sign. Several full QPE laws include endpoint phases. Brute-force three-run enumeration independently checks the median-interval formula.

The default distribution agrees with its analytic mixture within $4\times10^{-14}$; lowering preserves the prepared state up to a global scalar within $5\times10^{-14}$. These bounded numerical checks do not prove every generated circuit. **No Lean formalization is integrated**; a proof would need an explicit link to this generator and its compiler output.

```bash
python scripts/generate_amplitude_estimation.py --estimate
python -m pytest tests/test_amplitude_estimation.py
python scripts/execute_notebooks.py
```

```python
from openquantum.amplitude_estimation import build_amplitude_estimation
circuit = build_amplitude_estimation(n_data=2, threshold=1, phase_bits=4,
                                     measure=True)
```

Generate a larger circuit without a statevector or extra scaling sweep:

```bash
python scripts/generate_amplitude_estimation.py --data-qubits 5 \
  --threshold 11 --phase-bits 4 --no-sim --no-scaling \
  --output-dir artifacts/amplitude-5
```

Data width is capped at 20, phase width at ten and repeated source instructions at 250,000; the shared QPE builder also applies its 200,000 source-instruction cap. These source counts treat each MCX atomically. Primitive export is capped at 250,000 after lowering; controlled-gate compilation can consume extra work before that final guard. Simulation has a separate **12 total-qubit** cap. These are rejection limits, not affordability promises for every permitted-size instance. Larger exports/scaling rows carry no simulation claim. Each generation removes its own optional outputs before writing, so disabled estimates or plots cannot leave an older instance's report behind.

The [notebook](../notebooks/amplitude_estimation.ipynb) shares the implementation, reuses outputs above and offers bounded optional live controls. Static publication needs no server. To execute independently, unpack the [code bundle](../assets/generated/openquantum-code.zip), install its package with `python -m pip install .`, and use the [locked environment](../requirements-lock.txt).

Circuit construction/export/statevectors use **Qiskit core 2.6.2**, [tagged repository](https://github.com/Qiskit/qiskit/tree/2.6.2), [Apache-2.0 license](https://github.com/Qiskit/qiskit/blob/2.6.2/LICENSE.txt) {cite:p}`prim_qiskit_2_6_2`. Arrays and seeded sampling use NumPy 2.5.4; original plots use Matplotlib 3.11.2. Conditional estimates use the separately audited Microsoft QDK 1.34.0 adapter. This original implementation is [MIT licensed](../assets/generated/amplitude-estimation/LICENSE.txt); source hashes, device, versions and seeds are recorded.

- [Executed notebook](../notebooks/amplitude_estimation.ipynb)
- [Actual measured primitive circuit, OpenQASM 3](../assets/generated/amplitude-estimation/circuit.qasm)
- [Checks, environment and full logical/ideal-workload ledger](../assets/generated/amplitude-estimation/logical-resources.json)
- [Width/resolution scaling CSV](../assets/generated/amplitude-estimation/scaling.csv)
- [Conditional physical report](../assets/generated/amplitude-estimation/physical-resources.json)
- [Complete bounded physical scaling projections](../assets/generated/amplitude-estimation/surface-code-scaling.json)
- [Amplitude module](../assets/generated/amplitude-estimation/amplitude_estimation.py) and [QFT/QPE module](../assets/generated/amplitude-estimation/qft_qpe.py)
