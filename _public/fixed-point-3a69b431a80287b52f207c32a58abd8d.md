---
title: "Fixed-point amplitude amplification"
description: "Explicit known-input preparation and phased projectors, an independently checked Chebyshev success band, and conditional surface-code scaling."
downloads:
  - file: ../../notebooks/fixed_point_amplification.ipynb
    title: "Executable fixed-point notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Python examples and tests"
---

## Motivation and intuition

Ordinary [Grover search](../grover.md) can overshoot: repeating its two reflections past the success peak reduces success again. Fixed-point amplitude amplification uses a sequence of different reflection phases to obtain high ideal success throughout a promised range of initial probabilities. Yoder, Low and Chuang give the phase sequence and its Chebyshev success law {cite:p}`amp_ylc_2014`.

:::{note} Implemented and checked scope

The preparation, inverse preparation, good-register phase and complete all-zero phase are actual gates. The three-wire example and five selected sizes are simulated and checked against an independent polynomial recurrence; small full operators retain their global phase in a separate projector-matrix check. Generated circuits and conditional surface-code projections are downloadable. The ideal theorem does not certify binary64 angles, a concrete Clifford+T synthesis, hardware confidence or quantum advantage.

Physical initialization/reset, external loading and classical verification/controller/decoder latency are excluded. The input model assumes clean logical zero qubits and a supplied coherent preparation circuit.
:::

## Task, input model and fundamental algorithm

Supply a known unitary preparation $A$ on $q$ wires, so $|s\rangle=A|0^q\rangle$, and a computational-basis projector $\Pi_g$ identifying good outputs. Initial success is $p=\langle s|\Pi_g|s\rangle$. The schedule accepts a **promise** $p\ge w>0$ and a failure amplitude $0<\delta<1$; it does not measure or compute exact $p$ to choose phases. The requested ideal output failure probability is at most $\delta^2$.

This interface requires reusable $A$ and $A^\dagger$. An unknown specimen supplied once cannot replace that access model. Work qubits used by preparation belong to the coherent register; the reflection about the initial zero state must cover them too. The general amplitude-amplification framework motivates this known-preparation model {cite:p}`amp_amplification_2002`.

The actual projector phase is

$$S_g(\beta)=I+(e^{i\beta}-1)\Pi_g.$$

With $S_0(\phi)=I+(e^{i\phi}-1)|0^q\rangle\langle0^q|$, our convention is

$$G_j=-A S_0(-\alpha_j)A^\dagger S_g(\beta_j).$$

After the initial preparation, round $j$ chronologically applies $S_g(\beta_j)$, $A^\dagger$, $S_0(-\alpha_j)$, $A$, and global phase $\pi$. Round one acts first. Preserving the global minus sign matters for coherent reuse, even though it does not affect final measured probabilities.

### Phase schedule and complexity

Let $L=2\ell+1$ and

$$\gamma=\operatorname{sech}\left[\frac{\operatorname{acosh}(1/\delta)}L\right].$$

The phases are

$$\alpha_j=-\beta_{\ell-j+1}
=2\operatorname{arccot}\left[\tan(2\pi j/L)\sqrt{1-\gamma^2}\right],\quad j=1,\ldots,\ell.$$

The ideal success law is

$$P_L(p)=1-\delta^2 T_L\left(\frac{\sqrt{1-p}}\gamma\right)^2,$$

where $T_L$ is the degree-$L$ Chebyshev polynomial. For $p\ge1-\gamma^2$, its argument lies in $[0,1]$, so ideal failure is at most $\delta^2$. Choose a positive odd length satisfying

$$L\ge\frac{\operatorname{acosh}(1/\delta)}{\operatorname{atanh}\sqrt w}.$$

The $w=1$ case needs only the initial preparation. The bounded generator chooses the smallest permitted odd length and numerically rechecks its width. It computes the phase factor as $\tanh[\operatorname{acosh}(1/\delta)/L]$ to avoid subtracting nearly equal numbers. Phases remain binary64 values, without interval enclosures {cite:p}`amp_ylc_2014`.

For small $w$ and $\delta$, the preparation-access scaling is $O(\log(1/\delta)/\sqrt w)$. Actual calls here are **$\ell+1$ forward $A$ and $\ell$ inverse $A^\dagger$**, totaling $L$, with $\ell$ good phases and $\ell$ full zero phases. The paper's $L-1$ bit-oracle query convention counts compute/uncompute within each good phase; it must not be equated with these $A/A^\dagger$ calls.

“Fixed point” describes an expanding guaranteed success band as the full schedule is recomputed. Individual phases are not prefixes of a longer schedule, and success need not increase pointwise with $L$. Inputs below the promise still have a defined circuit and success curve, but lose the uniform guarantee. $\delta=0$ has no finite uniform zero-error schedule for $p<1$.

### Related variants and boundaries

| Construction | Status and role |
|---|---|
| Ordinary $\pi$-phase amplification | Implemented in [Grover search](../grover.md); a known success angle determines the first peak. |
| Yoder–Low–Chuang fixed-point phases | Implemented here; a lower-bound promise determines a uniform ideal success band. |
| Oblivious amplification | Separate [Hamiltonian-simulation method](../simulation/amplification.md); its block-encoding and output contracts differ. |
| Exact success-dependent phases | Planned; knowing the initial success allows a different exact-amplification contract. |

## Small circuit and independent classical checks

The demo uses two uniform data bits and a reversible flag $[x<1]$, giving $p=1/4$. The explicit preparation applies Hadamards and a comparator built from X and controlled-X gates; the threshold is known and has a trivial classical shortcut. This is an implementation/accounting example.

```{embed} #amp-fixed-point-preparation
```

Use $w=0.1$, $\delta=0.05$, $L=13$ and six phased rounds. Good means flag wire `q[2]` equals one. The all-zero projector covers **all three wires**, and the final measured trial reads both data and flag.

```{embed} #amp-fixed-point-circuit
```

```{include} ../../assets/generated/fixed-point-amplification/summary.md
```

The independent ideal success is approximately **0.997979172871**, compared with the promised lower bound 0.9975. Conditional on a good flag, the data state preserves the normalized original good component; amplification changes its weight rather than changing relative amplitudes inside that component.

```{embed} #amp-fixed-point-band
```

The comparison curve uses ordinary amplification with the same six reflection rounds. This is an ideal probability comparison, not a matched physical-resource or application-speedup claim. The notebook checks actual one-qubit circuits at eight probabilities, including zero, one and values below the promise. The test suite also compares complex multi-qubit operators with independently specified projector matrices and verifies conditional good-state direction.

## End-to-end resource analysis

For one complete trial, with the actual preparation and projectors at size $n$,

$$G(n)=C_A(n)+\ell[C_{A^\dagger}(n)+C_0(n)+C_A(n)+C_g(n)]+C_{\rm readout}(n).$$

For $R$ fresh serial trials, every term is paid $R$ times. This release uses **$R=1$**. No oracle, preparation, inverse or multi-controlled reflection is assigned unit gate cost. A different reversible application predicate or loading circuit changes these costs and must be explicitly supplied. Here $n$ means data bits, and live width is $q=n+1$ including the flag.

The [logical report](../../assets/generated/fixed-point-amplification/logical-resources.json) records preparation calls, both projectors, all final readouts, primitive counts and dependency depth. [The complete primitive circuit](../../assets/generated/fixed-point-amplification/circuit.qasm) retains arbitrary rotations. The resource adapter additionally optimizes that same complete trial at level one; its [estimated circuit](../../assets/generated/fixed-point-amplification/estimated-circuit.qasm) has its own hash and a checked ideal output. OpenQASM's omitted overall phase is irrelevant to this measured trial; the export is not a controlled-oracle specification.

### Computed scaling with system size

The sweep uses $n\in\{2,3,4,6,8\}$ and threshold $2^{n-2}$, holding $p=1/4$, $w=0.1$, $\delta=0.05$, $L=13$ and one trial fixed. Every selected circuit is materialized, lowered and bounded-statevector checked. Counts increase from **396** to **4,452** primitive operations including readout, and live width from three to nine wires. These are five computed instances, without an asymptotic fit or simulated large-system claim.

![Original compiled-circuit scaling under fixed promise and ideal failure amplitude](../../assets/generated/fixed-point-amplification/logical-scaling.svg)

The threshold family's comparator only controls a few high bits, while the full zero reflection still covers all live wires. A general threshold, work register or application predicate can have a different scaling law. [All logical counts](../../assets/generated/fixed-point-amplification/scaling.csv) and individual complete circuits are retained.

### Surface-code model and failure contract

The [computed physical reports](../../assets/generated/fixed-point-amplification/physical-resources.json) use the audited Microsoft QDK 1.34.0 adapter {cite:p}`amp_qdk_1_34_0`. Its two fixed profiles have physical error rates $10^{-3}$ and $10^{-4}$, 100 ns gates and 500 ns measurements, with rotated surface code and round-based magic-state factories. The existing [Grover baseline](../grover.md#end-to-end-resource-analysis) remains unchanged.

The default canonical QDK trial trace contains **91 exact T states, 48 arbitrary rotations and three measurements**. PSSPC assigns 17 T states per rotation, using a summed synthesis-model scalar about $6.11\times10^{-6}$, within the configured $10^{-5}$ allowance. These are modeled mean synthesis costs, without a generated per-angle Clifford+T sequence, approximation certificate or runtime-tail guarantee.

Allocate ideal failure $\delta^2=0.0025$, synthesis-model allowance $10^{-5}$ and physical-fault allowance 0.005. Their sum, **0.00751**, is a conditional one-trial failure ceiling only if exact phases meet the theorem, the synthesis scalar upper-bounds output-distribution total variation, the physical model bounds faults and classical verification is correct. Binary64 phase evaluation supplies no rigorous angle certificate. **Certified hardware confidence remains unavailable.**

The default fastest returned point at physical error $10^{-4}$ uses 14,308 peak physical qubits and 4.28 ms modeled serial runtime; at $10^{-3}$ it uses 175,452 qubits and 7.704 ms. The five-size sweep returns **52 Pareto points** across the two profiles. All points, code distances, factories and model provenance are retained; the plot selects each size/profile's fastest returned point.

![Conditional surface-code scaling at the fastest returned points](../../assets/generated/fixed-point-amplification/physical-scaling.svg)

Peak qubits times runtime is a rectangular space-time proxy, not an integral of changing occupancy. Clifford scheduling is abstracted by the estimator. Physical initialization/reset, data loading, controller/decoder/verification latency and synthesis runtime tails remain outside the measured boundary; these projections cannot establish an end-to-end application advantage. [All physical scaling points](../../assets/generated/fixed-point-amplification/physical-scaling.csv) are downloadable.

## Reproduce, simulate and inspect

The original implementation uses Qiskit core **2.6.2** for gates, statevectors, lowering and exports {cite:p}`amp_qiskit_2_6_2`; NumPy **2.5.4** supplies independent matrix/recurrence arithmetic {cite:p}`amp_numpy_2_5_4`, and Matplotlib **3.11.2** creates the original figures {cite:p}`amp_matplotlib_3_11_2`. Package versions, source/generator/lock hashes, CPU device and transpiler seed are recorded. The algorithm code and circuit figures are original; upstream algorithm code is not copied.

```bash
python scripts/generate_fixed_point_amplification.py --estimate
python -m pytest tests/test_fixed_point_amplification.py
```

To generate larger circuits without statevector execution:

```bash
python scripts/generate_fixed_point_amplification.py --data-qubits 8 \
  --threshold 64 --no-sim --no-scaling --output-dir artifacts/fixed-point-n8
```

The source builder accepts at most 12 live wires, $L\le255$, a 200,000 recursive source/expansion preflight and definition nesting depth at most 64. Primitive exports are capped at 250,000 operations after lowering; this is a guard, not a compilation-runtime proof. Statevector probability checks are separately capped at ten live wires. Tiny failure amplitudes whose squared bound underflows binary64 are rejected. The generator removes stale optional physical/scaling artifacts when invocation changes.

The [notebook](../../notebooks/fixed_point_amplification.ipynb) includes bounded live controls that regenerate actual circuits and recheck probabilities. Saved outputs work on a static site. A live local or Binder kernel enables the controls; it is not required for public reading. The notebook also runs beside the [fixed-point module](../../assets/generated/fixed-point-amplification/fixed_point_amplification.py) and [preparation module](../../assets/generated/fixed-point-amplification/amplitude_estimation.py), using the scientific dependencies from the [Binder environment](../../binder/environment.yml). Estimation additionally needs the full locked environment.

## Verification and formalization

The dedicated tests check independent Chebyshev recurrence values, exact good-register bit ordering, complete complex operators and global phases, lowered-state equivalence, conditional good-state direction, final measurement mapping, invalid/noncoherent preparation and generation/simulation bounds. These are deterministic numerical and functional checks over selected instances. The executable notebook and generation reports record which simulations ran; an unexecuted export has no simulation badge.

No Lean formalization, proof-producing phase generator, verified compiler or certified hardware implementation is integrated. The paper's mathematical theorem is a reference, and numerical agreement does not prove this Python implementation for every allowed input.

- [Executed notebook](../../notebooks/fixed_point_amplification.ipynb)
- [Logical/access report](../../assets/generated/fixed-point-amplification/logical-resources.json)
- [Conditional physical report](../../assets/generated/fixed-point-amplification/physical-resources.json)
- [Every artifact and source hash](../../assets/generated/fixed-point-amplification/manifest.json)
- [Original implementation and MIT license](../../assets/generated/fixed-point-amplification/LICENSE.txt)
