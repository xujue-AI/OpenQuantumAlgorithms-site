---
title: "Higher-order Suzuki formulas"
description: "Explicit positive and negative product-formula circuits, fixed-task error/work studies and scalar rounding analysis."
downloads:
  - file: ../../notebooks/simulation_suzuki.ipynb
    title: "Executable Suzuki notebook"
---

## Motivation and intuition

A symmetric product formula cancels leading splitting errors. Higher-order Suzuki formulas compose several symmetric steps, including backward-evolution steps, to cancel more terms. Extra cancellation can reduce approximation error while adding gates and longer absolute evolution demand. The useful choice depends on the requested accuracy and complete compiled workload {cite:p}`sim_trotter_error_2021,ham_comparison_2018`.

:::{note} Evidence and task contract

The canonical 36-case study and all eight notebook code cells have executed in the locked environment. The focused source and release checks passed (119 tests). A separate 32-wire, order-six circuit was generated without dense simulation. The study uses one fixed family, time, initial state and output before comparing orders. Physical projections, certified rotation synthesis and formalization remain planned.
:::

## Fundamental algorithm and supported family

Split an open-chain Hamiltonian into commuting groups,

$$
H=A+B,\qquad A=J\sum_{j=0}^{n-2}Z_jZ_{j+1}+h_Z\sum_{j=0}^{n-1}Z_j,
\qquad B=h_X\sum_{j=0}^{n-1}X_j.
$$

Terms within $A$ commute, as do terms within $B$. Coefficients are real; this implementation explicitly supports ZZ/Z/X chains, not arbitrary complex-weight Pauli lists. Couplings, time and basis inputs are declared before construction. There is no Hamiltonian or time rescaling; $\hbar=1$ and the Pauli coefficient norm is

$$\lambda=|J|(n-1)+n(|h_X|+|h_Z|).$$

The chronological base step applies $A/2$, then $B$, then $A/2$:

$$S_2(\delta)=e^{-iA\delta/2}e^{-iB\delta}e^{-iA\delta/2}.$$

For $k\ge2$, set

$$p_k=\frac1{4-4^{1/(2k-1)}},\qquad
S_{2k}(\delta)=S_{2k-2}(p_k\delta)^2
S_{2k-2}((1-4p_k)\delta)S_{2k-2}(p_k\delta)^2.$$

Repeat the full composition $r$ times at $\delta=t/r$. Orders two, four and six use respectively **1, 5 and 25 Strang blocks per macro-step**. At positive time, orders four and six have one and eight negative leaves. Two nested negative factors produce a positive leaf; each actual sign remains in the exported circuit.

![Signed normalized Strang substeps](../../assets/generated/simulation/suzuki/signed-substeps.svg)

### Gates and chronology

A diagonal-half ZZ term uses `CX; RZ(J*delta); CX`, a diagonal-half Z term uses `RZ(hZ*delta)`, and a full X term uses `RX(2*hX*delta)`. Both diagonal halves and every forward/backward block are materialized; adjacent halves are deliberately left unmerged for transparent source accounting. No dense evolution gate or opaque oracle enters our generator.

Each complete measured trial resets all system wires, prepares the declared computational basis state with X gates, runs the entire recursion and measures every wire in Z. Intermediate source negative durations are instructions for inverse evolution. Their gates still consume positive hardware time.

## Complexity, normalization and source rounding

For a fixed finite Hamiltonian and the ideal exact coefficients, the order-$2k$ global splitting error has the form $O(|t|^{2k+1}/r^{2k})$, with Hamiltonian/commutator-dependent constants. Those constants and the appropriate output norm matter when choosing $r$ {cite:p}`sim_trotter_error_2021`. A fitted slope on three spins is not a generic step-selection certificate.

The implementation interprets decimal/rational inputs as exact fractions. It encloses the cube/fifth root of four with 160 exact dyadic bisections and selects each binary64 recursive coefficient from the rational enclosure midpoint. The coefficient error is carried through signed interval multiplication. Actual binary64 leaf weights need not sum exactly to one: the ledger retains their exact rational signed sum and consistency defect instead of silently renormalizing them.

For leaf coefficient errors bounded by $d_\ell$, a source coefficient perturbation bound is

$$\epsilon_{\rm coeff}\le\lambda|t|\sum_\ell d_\ell.$$

Every RX/RZ angle has a separate conversion allowance $|\theta_{\rm float}-\theta_{\rm target}|/2$; the ledger sums every occurrence. These rational bounds compare the finite source parameters with mathematical Suzuki coefficients in an ideal gate model. They do not bound the formula's splitting error relative to exact dynamics, numerical reference error, Clifford+T synthesis or physical faults.

## Small circuits and independent performance

The canonical **36 cases** hold $J=1$, $h_X=0.7$, $h_Z=0$, $t=1$, input $|0\ldots0\rangle$ and output $\langle Z_0\rangle$ fixed, varying only

$$n\in\{1,2,3\},\qquad 2k\in\{2,4,6\},\qquad r\in\{1,2,4,8\}.$$

The three-spin Hamiltonian is the same task as the existing Lie/Strang demonstration. Its old source and records are preserved. The one-spin case has only X evolution and is a commuting limit: its tiny numerical errors should not be fitted as higher-order convergence.

```{include} ../../assets/generated/simulation/suzuki/summary.md
```

![Complete two-spin order-four measured circuit](../../assets/generated/simulation/suzuki/n2-order4-circuit.svg)

An independent Kronecker Hamiltonian and matrix exponential provide the exact-dynamics reference. A separate dense product of group exponentials checks the implemented chronology. The actual reset/preparation prefix is simulated as a density operator and compared with the corresponding prepared pure state.

A separately credited **Qiskit 2.6.2 SuzukiTrotter** implementation provides another numerical reference. It receives the nonempty commuting groups `[A, B]` with `preserve_order=True`; zero groups are omitted only as a declared commuting-limit shortcut. Its library coefficient calculation and atomic scheduling differ from our source. Operator agreement is tested at a stated tolerance; byte-identical circuits and physical-layout equivalence are not asserted {cite:p}`sim_qiskit_262`.

![Error against macro-step count](../../assets/generated/simulation/suzuki/error-vs-steps.svg)

![Error against complete measured primitive work](../../assets/generated/simulation/suzuki/error-vs-work.svg)

These plots compare actual finite circuits on the same task, including their extra gates. They are numerical work/error tradeoffs, not an unconditional recommendation of the highest order or a fault-tolerant speedup. State-vector error, infidelity and observable bias are recorded separately from spectral operator error. No finite-shot or physical measurements supply these plotted means.

## End-to-end space and time analysis

At $h_Z=0$, with nonzero $J,h_X$, one complete trial has

$$N_{\rm rot}=(3n-2)\,5^{k-1}r,\qquad
N_{\rm CX}=4(n-1)\,5^{k-1}r,$$

plus $n$ resets, the actual basis-preparation X gates and $n$ final measurements. A nonzero Z field adds $2n\,5^{k-1}r$ RZ rotations. Zero terms and the one-spin boundary use their actual counts. The width is $n$ system qubits with no ancillas.

Thus source construction/gate work is $O(5^{k-1}nr)$ at fixed order, before rotation synthesis. Maintaining accuracy as $n,t$ change can require a different $r$ and error bound. For $R$ observable readouts, repeat the complete preparation and circuit $R$ times; every failed or discarded physical attempt would also need accounting. The recorded Hoeffding proposal

$$R\ge\left\lceil2\log(2/\eta)/\epsilon_{\rm stat}^2\right\rceil$$

applies to independent fresh $\pm1$ readouts of one fixed circuit. Its repetitions are budgeted, not executed, and do not remove algorithmic bias. Source recursion and coefficient preprocessing, independent simulation, export and plotting local wall times are recorded separately from quantum-model time. The records retain the historical `cpu_timings_seconds` key; those timers measure elapsed wall time. External classical control/readout hardware latency is unmeasured.

### Fault-tolerant handoff

Physical resources remain planned. A matched adapter must include all positive/negative RX/RZ operations, complete initialization/readout, source and synthesis allowances, repetition workload, routing/idle exposure and factories at the frozen hardware assumptions. It must not interpret signed-time cancellation as hardware-time savings or remove reset instructions to fit a static importer.

A lower raw gate count or a smaller numerical operator error does not alone select a surface-code factory layout or an analog-rotation architecture. The existing surface-code baseline remains frozen. A later comparison must fix the Hamiltonian, output, accuracy, confidence and complete workload before changing order, merging or synthesis.

## Classical simulation and formalization

Circuit generation stops at 64 wires, 16,384 Strang blocks and 100,000 complete primitive operations. Dense references stop at three spins, 512 base blocks and 10,000 evolution-body operations before allocation; at most three preparation and six reset/readout operations are additional. Dense operators store $4^n$ entries and are independent of the larger circuit-generator guard.

The integer/rational source-bound checks and numerical matrix tests are not a Lean/Rocq proof of circuit correctness or a formal physical-resource theorem. Both formalization and certified synthesis are future work.

## Reproduce, packages and downloads

Use the locked Python 3.12 environment and a fresh output path:

```bash
python scripts/verify_simulation_suzuki.py
python scripts/generate_simulation_suzuki.py --output-dir artifacts/suzuki-replay
python scripts/verify_simulation_suzuki.py --replay-dir artifacts/suzuki-replay
pytest tests/test_simulation_suzuki.py tests/test_simulation_suzuki_release.py
python scripts/generate_simulation_suzuki.py --qubits 32 --orders 6 \
  --repetitions 1 --no-sim --output-dir artifacts/suzuki-chain32
```

The standard-library verifier authenticates the saved study, current producer
sources and record/QASM/summary bindings without repeating a simulation.
Fresh generation runs the numerical references; the subsequent comparison
keeps source scalars, primitive circuits and scope exact, with absolute
tolerance $10^{-11}$ only for the recorded numerical diagnostics. This
comparison tolerance supplies no algorithmic, synthesis or physical-error
certificate.

The original source and figures are MIT-licensed. Qiskit 2.6.2 ([source](https://github.com/Qiskit/qiskit/tree/2.6.2), Apache-2.0) supplies circuit containers, ideal state/operator checks and the separately inspected reference. NumPy 2.5.4 ([source](https://github.com/numpy/numpy/tree/v2.5.4), BSD-3-Clause), SciPy 1.18.1 ([source](https://github.com/scipy/scipy/tree/v1.18.1), BSD-3-Clause) and Matplotlib 3.11.2 ([source/license](https://github.com/matplotlib/matplotlib/tree/v3.11.2)) supply arrays, independent exponentials and original figures {cite:p}`sim_qiskit_262,opt_numpy254,sim_scipy_1181,opt_matplotlib3112`.

- [Study manifest and exact source/reference hashes](../../assets/generated/simulation/suzuki/manifest.json)
- [Saved verification script](../../code/scripts/verify_simulation_suzuki.py)
- [Complete 36-case summary](../../assets/generated/simulation/suzuki/summary.csv)
- [Three-spin order-six/r-eight circuit and ledger](../../assets/generated/simulation/suzuki/n3-s6-r8/record.json)
- [Original module snapshot](../../assets/generated/simulation/suzuki/suzuki.py), [generator snapshot](../../assets/generated/simulation/suzuki/generate_simulation_suzuki.py) and [MIT license](../../assets/generated/simulation/suzuki/LICENSE.txt)
- [Companion notebook](../../notebooks/simulation_suzuki.ipynb) and [complete public code package](../../assets/generated/openquantum-code.zip)

The generator snapshot expects the full package's source layout. The module can also sit beside a downloaded notebook. Saved pages/figures are static; optional notebook controls require a local or Binder kernel.

## References

```{bibliography}
```
