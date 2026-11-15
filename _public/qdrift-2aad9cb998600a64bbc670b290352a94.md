---
title: "qDRIFT: randomized Hamiltonian evolution"
description: "Primitive Pauli circuits, exact rational term sampling and separately checked averaged channels."
downloads:
  - file: ../../notebooks/simulation_qdrift.ipynb
    title: "Executable qDRIFT notebook"
---

## Motivation and intuition

qDRIFT replaces a prescribed product-formula ordering with randomly chosen term exponentials. Larger coefficients are selected more often. Averaging over newly drawn complete circuits approximates the desired dynamics even when one circuit is a poor approximation to the evolution operator {cite:p}`ham_qdrift_2019`.

:::{note} Evidence and output contract

Executed evidence consists of bounded ideal circuit/channel checks and seeded classical sampling, with actual circuit exports and a reproduced companion notebook. The small study contains one readout from each freshly selected trajectory. Its outputs are a channel and a measured observable, rather than one deterministic coherent evolution oracle. Physical estimates and formalization remain planned.
:::

## Fundamental algorithm and complexity

For a supplied Pauli expansion,

$$
H=\sum_{j=1}^{m}h_jP_j,\qquad\lambda=\sum_j|h_j|,\qquad p_j=|h_j|/\lambda,
$$

choose $j$ independently at every step and apply

$$
U_j=\exp[-i\,\operatorname{sgn}(h_j)\lambda tP_j/N].
$$

A complete trajectory has $N$ such gates in chronological order. For column-vectorized density matrices, the averaged elementary channel and complete channel are

$$
S=\sum_jp_j(\overline{U_j}\otimes U_j),\qquad\mathcal E_N=S^N.
$$

The algorithm's channel-error complexity scales as $O(\lambda^2t^2/\epsilon)$ term exponentials in the stated ideal access model {cite:p}`ham_qdrift_2019`. This asymptotic relation is not a certificate that our chosen small $N$ meets a specified diamond-error target, or that one saved circuit has operator error $\epsilon$.

The code treats decimal/rational strings, including `str(float)`, as exact fractions. It combines repeated Pauli labels before computing $\lambda$ and drops exactly zero coefficients. Integer cumulative weights implement the declared categorical probabilities without a floating-point probability table. PRNG seeds support reproduction; mathematical independence remains an explicit ideal-randomness assumption. Identity terms stay in the distribution and contribute circuit global phase.

### Primitive rotations and input access

For a Pauli with $k$ nonidentity wires, X wires use H basis changes, and Y wires use SDG–H followed by H–S on the way back. A parity ladder uses $2(k-1)$ CX gates around one $R_Z(2\operatorname{sgn}(h_j)\lambda t/N)$. The generator appends these gates directly. It never inserts a dense Hamiltonian or an opaque evolution gate.

The measured trial resets all $n$ wires, prepares the declared computational basis state with X gates, runs every exponential, and measures every wire in Z. Coefficients and Pauli strings are explicit classical inputs. Their table construction and all term draws remain classical work; no data-loading oracle is assumed free.

## Small circuits and independent answers

The fixed demonstrations use $t=1/2$, $N=8$, initial state zero, and 32 fresh trials per case:

- One spin: $H=0.7X+0.3Z$, with $\lambda=1$.
- Two spins: $H=Z_0Z_1+0.7(X_0+X_1)$, with $\lambda=12/5$.

```{include} ../../assets/generated/simulation/qdrift/summary.md
```

The complete first trial for each case is shown below; all 64 measured circuits and their term schedules are saved.

![Complete one-spin trial](../../assets/generated/simulation/qdrift/one-spin/first-complete-trial.svg)

![Complete two-spin trial](../../assets/generated/simulation/qdrift/two-spin/first-complete-trial.svg)

Every selected circuit is compared with an independent dense product using $P^2=I$. The actual reset/preparation prefix is also simulated as a density operator and compared with that product's prepared state. A separate single-step mixture is raised to $N$ without enumerating exponentially many complete schedules. Tests independently enumerate every two-step trajectory for a tiny noncommuting example and check the full channel on matrix units, rather than only the zero-state observable.

![Two-spin channel convergence and saved trajectories](../../assets/generated/simulation/qdrift/two-spin/channel-and-samples.svg)

The records distinguish the averaged channel, the empirical average of saved circuit channels and individual trajectory means. The plots show channel-error curves, individual trajectory expectations, the ideal mixture mean and the mean of one classically sampled full Z-readout per trajectory, scored by bit zero. These are ideal simulation results, not QPU measurements.

The reported spectral superoperator norm is the Hilbert–Schmidt-induced $2\to2$ norm. Normalized Choi trace distance, fixed-input trace distance and observable bias are separate diagnostics. None is a computed diamond norm. CP/TP follow analytically from a mixture of ideal unitary channels; floating-point Choi eigenvalue and trace checks are implementation diagnostics, not interval or formal certificates.

## Important variants and sampling contracts

Repeating readout on one saved seed estimates that circuit's expectation. Freshly drawing a complete schedule before each readout instead estimates the averaged-channel expectation. Those experiments have different joint distributions and must not share an unsupported confidence claim.

For ideal independent $\pm1$ readouts from fresh trajectories, Hoeffding gives

$$
R\ge\left\lceil\frac{2\log(2/\eta)}{\epsilon_{\rm stat}^2}\right\rceil.
$$

The 32-trial examples deliberately have broad confidence intervals. Their seeded realization does not empirically validate the analytic coverage statement. Hamiltonian channel bias, scalar gate rounding, synthesis error, statistical precision and physical failures have separate budgets. Controlled qDRIFT for phase estimation, grouped/importance-sampled variants and hardware-specific compilation remain planned.

## End-to-end resource analysis

### Space and time as the system grows

The logical width is $n$ data wires with no ancillas for these Pauli circuits. One support-$k_j$ sampled exponential uses one RZ, $2(k_j-1)$ CX and its actual basis changes; an identity term carries only global phase. The saved ledger sums the actual instructions of every complete trajectory rather than multiplying one favorable circuit by the trial count.

For $R$ fresh trials, charge $R$ complete preparations, $RN$ categorical draws/exponentials, every basis/parity gate and $Rn$ final measurements. Explicit input processing depends on $m$, Pauli representation and the bit length of the integer weight table. The generator limits that representation independently of dense simulation.

For the local chain with X and optional Z fields,

$$
\lambda=|J|(n-1)+n|h_X|+n|h_Z|.
$$

Each local term has bounded support, so $N$ exponentials require $O(N+n)$ gates per measured trial before synthesis. At fixed couplings and time, the ideal qDRIFT query scaling can consequently grow quadratically with $n$ for a fixed channel-error target. Arbitrary dense Pauli strings can add $O(n)$ work to each exponential. These source scalings do not establish an advantage over another compiled algorithm.

### Fault-tolerant handoff

No new physical estimate was executed. The next surface-code adapter must cost the complete measured workload, including resets, all draws' circuit constructions, routing/idle exposure, readout and factories. Reset and feedforward assumptions must be explicit; an adapter that drops the initial resets does not cost the exported trial.

The scalar RZ conversion bound recorded in the ledger concerns binary64 angle conversion only. It excludes a concrete Clifford+T synthesis proof and Qiskit's overall-phase accumulation/reduction for identity terms. Overall phase is immaterial to these measured channels, but exported QASM must not silently become a coherently controlled oracle. Any later physical comparison will preserve the frozen task, output, accuracy, success and hardware assumptions.

## Classical simulation and formalization

The circuit generator stops at 64 wires, 256 input terms, 512 exponentials and 100,000 primitive operations. Dense trajectory/channel references stop at three qubits **before allocation**; an empirical study stops at 64 trajectories. A density matrix or unitary has $4^n$ entries, while the full superoperator has $16^n$ entries. These limits are deliberately different from circuit materialization limits.

No Lean or Rocq proof is reproduced for qDRIFT here. The algebraic channel identity and numerical checks do not establish a compiler or physical fault-tolerance theorem.

## Reproduce, packages and downloads

Use Python 3.12 and the [locked environment](../../requirements-lock.txt). Preserve the recorded output and write each new run to a fresh directory:

```bash
python scripts/generate_simulation_qdrift.py --output-dir artifacts/qdrift-replay
pytest tests/test_simulation_qdrift.py
python scripts/generate_simulation_qdrift.py --qubits 32 --steps 64 \
  --no-sim --output-dir artifacts/qdrift-chain-32
```

The original implementation and figures are MIT-licensed. Qiskit 2.6.2 ([source](https://github.com/Qiskit/qiskit/tree/2.6.2), Apache-2.0) supplies primitive circuit containers and ideal Operator/Statevector/DensityMatrix checks {cite:p}`sim_qiskit_262`. NumPy 2.5.4 ([source](https://github.com/numpy/numpy/tree/v2.5.4), BSD-3-Clause), SciPy 1.18.1 ([source](https://github.com/scipy/scipy/tree/v1.18.1), BSD-3-Clause) and Matplotlib 3.11.2 ([source/license](https://github.com/matplotlib/matplotlib/tree/v3.11.2), Matplotlib license) supply arrays, the independent matrix exponential and original plots {cite:p}`opt_numpy254,sim_scipy_1181,opt_matplotlib3112`. The recorded manifest gives actual environment versions, source commit/hashes, seeds, parameters, device and timing scope. Qiskit's separate upstream QDrift synthesizer is a reference; it is not the generator used here.

- [Recorded study manifest](../../assets/generated/simulation/qdrift/manifest.json)
- [One-spin study and complete logical ledgers](../../assets/generated/simulation/qdrift/one-spin/study.json)
- [Two-spin study and complete logical ledgers](../../assets/generated/simulation/qdrift/two-spin/study.json)
- [Original module snapshot](../../assets/generated/simulation/qdrift/qdrift.py), [generator snapshot](../../assets/generated/simulation/qdrift/generate_simulation_qdrift.py) and [MIT license](../../assets/generated/simulation/qdrift/LICENSE.txt)
- [Executable companion notebook](../../notebooks/simulation_qdrift.ipynb) and [complete public code package](../../assets/generated/openquantum-code.zip)

The copied generator is a source snapshot; its CLI expects the complete package's source layout. The module also supports direct use beside a downloaded notebook. Static pages and saved figures require no public kernel; live notebook controls use a local kernel or the configured Binder environment.

## References

```{bibliography}
```
