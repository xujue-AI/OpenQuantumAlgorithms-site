---
title: "Noncommuting Lindblad dynamics"
description: "Explicit Lie/Strang channel circuits for a one-qubit Hamiltonian and damping jump."
---

This example combines a Hamiltonian rotation with amplitude damping. Their generators do not commute, so alternating exact substeps introduces a controlled splitting approximation. It extends the [finite-time damping example](open-systems.md) with an actual quantum channel circuit for the combined dynamics.

:::{note} Implementation status
First- and second-order positive-time splitting circuits are implemented and compared with a small independently assembled Liouvillian exponential. Hamiltonian substeps use an exact analytic RY–RZ–RY construction; damping substeps use the checked environment dilation with a fresh reset each time. Original gates, resets, input preparation and final readout are counted. A separate conditional physical-resource study below supplies actual Clifford+T compilation for the fixed 32-step Z task. General Liouvillian access, hardware calibration and formal certificates remain planned.
:::

## Model and outputs

Set $\hbar=1$, use one system qubit and

$$H=0.7X+0.3Z,\qquad L=\sqrt{0.8}|0\rangle\langle1|.$$

The coefficients and rate have units of inverse time. The time-independent GKSL equation is

$$\dot\rho=\mathcal L(\rho)=-i[H,\rho]+L\rho L^\dagger-
\tfrac12\{L^\dagger L,\rho\},\qquad \rho(t)=e^{t\mathcal L}\rho(0).$$

The fixed task uses $t=1$ and initial $|+i\rangle$, prepared with H and S gates. The state output is the system's reduced density operator before final system readout. An observable output estimates one specified X, Y or Z mean. Each measured export produces a classical trial, with the environment outcome ignored. The code also checks $|0\rangle$, $|1\rangle$, $|+\rangle$ and supplied small mixed density inputs.

GKSL generators are the model context {cite:p}`sim_lindblad_1976,sim_gks_1976`. This is a single-qubit, single-jump construction; it does not implement arbitrary interacting open-system models.

## Exact Hamiltonian access

Let

$$\omega=\sqrt{h_x^2+h_z^2},\qquad\theta=\operatorname{atan2}(h_x,h_z).$$

Since $H=\omega R_Y(\theta)ZR_Y(-\theta)$,

$$U_H(s)=e^{-iHs}=R_Y(\theta)R_Z(2\omega s)R_Y(-\theta).$$

The chronological gate order starts with $R_Y(-\theta)$, followed by RZ and then $R_Y(\theta)$. There is no inner Hamiltonian product-formula approximation. Zero Hamiltonian or zero time skips this substep. The analytic circuit is independently compared with the matrix exponential for positive and negative Hamiltonian coefficients.

```{embed} #simulation-lindblad-hamiltonian
```

The exact gate convention is the pinned Qiskit 2.6.2 convention {cite:p}`sim_qiskit_262`. Matrix exponentiation is used solely as a bounded classical answer; no dense unitary gate or eigendecomposition supplies the quantum implementation.

## Lie and symmetric schedules

Write $\mathcal U_s(\rho)=U_H(s)\rho U_H(s)^\dagger$ and let $\mathcal D_s$ be the finite-time damping channel with rate $\gamma=0.8$. For $\delta=t/r$, the circuits implement

$$\mathcal F_1(t;r)=\left(\mathcal D_\delta\circ\mathcal U_\delta\right)^r,$$

$$\mathcal F_2(t;r)=\left(\mathcal U_{\delta/2}\circ\mathcal D_\delta\circ\mathcal U_{\delta/2}\right)^r.$$

Thus Lie applies H then damping chronologically; Strang applies half H, damping and half H. Every substep duration is nonnegative, and each substep is a valid quantum channel. Lie–Trotter and symmetric Strang splitting provide the algorithmic context {cite:p}`sim_trotter_1959,sim_strang_1968`. Liouvillian product-formula theory is discussed by Kliesch et al., together with its erratum {cite:p}`sim_liouvillian_trotter_2011,sim_liouvillian_trotter_erratum_2012`; the numerical curves here are implementation evidence.

Each damping step resets the environment to zero, then applies the survival-aware controlled-RY/reverse-CX dilation from the damping example. Resetting the reused environment discards its previous correlations. Keeping the old environment would implement a correlated repeated interaction instead of the intended Markovian composition; a test checks that removing these resets changes the result.

```{embed} #simulation-lindblad-circuit
```

The diagram shows one symmetric step and Z readout; the accuracy run uses $r=32$. Wire q0 is the system and q1 the reusable environment. A full trial includes one system initialization, $r$ fresh environment initializations, $r-1$ discard/reset operations before reuse and final system/environment measurements. The final environment bit is ignored. No environment outcome is postselected and no inverse damping interaction is applied.

## Independent Liouvillian reference

The classical reference uses column vectorization:

$$\operatorname{vec}(A\rho B)=(B^T\otimes A)\operatorname{vec}(\rho).$$

Its $4\times4$ generator is assembled independently as

$$\mathbf L=-i(I\otimes H-H^T\otimes I)+L^*\otimes L-
\tfrac12\left[I\otimes L^\dagger L+(L^\dagger L)^T\otimes I\right].$$

The dense reference is $e^{t\mathbf L}\operatorname{vec}(\rho_0)$, evaluated with SciPy 1.18.1 {cite:p}`sim_scipy_1181,sim_expm_2009`. It assembles the time-scaled generator from $\gamma t,h_xt,h_zt$ before exponentiation, keeping small finite exposures representable even for large coefficients and short times. Unrepresentable raw matrix entries are rejected. Qiskit DensityMatrix instead evaluates the actual gate/reset circuit and traces the environment. Supplied mixed density matrices initialize this small classical checker; no arbitrary quantum mixed-state loading algorithm is claimed.

Four physically prepared inputs reconstruct the numerical channel matrix. We report selected-input trace distance and the spectral norm of the column-vectorized channel error. The latter is the induced Hilbert–Schmidt $2\to2$ norm; it is distinct from induced trace norm and diamond norm. It is not a worst-case observable-confidence certificate.

The primary Liouvillian paper uses a doubled dissipator $D[X](\rho)=2X\rho X^\dagger-\{X^\dagger X,\rho\}$, so its $X$ corresponds to $L/\sqrt2$ in our convention. Constants and norms must be reconciled before using a theorem from that source. Neither a plotted numerical curve nor the uncorrected paper supplies a general classical-simulation or quantum-advantage claim.

```{embed} #simulation-lindblad-results
```

```{embed} #simulation-lindblad-convergence
```

At 32 symmetric steps, the selected input's trace distance is approximately $1.3573\times10^{-5}$, the channel $2\to2$ error is $4.2320\times10^{-5}$ and the Z bias is $1.6045\times10^{-5}$. The exact Z mean is approximately **0.911296372329**. In the tested range, doubling steps approximately halves Lie error and quarters Strang error. Trace and positivity checks cover the tested inputs and times; commuting/zero-component cases agree with their exact reference.

```{embed} #simulation-lindblad-density
```

## Resources and separate errors

The measured trial includes named input preparation, all analytic Hamiltonian rotations, every damping dilation, environment reset/discard and X/Y/Z basis changes. The compiler can merge neighboring Hamiltonian half steps, so metadata distinguishes source-level substep calls from actual primitive counts. Depth is a circuit dependency depth, without a physical scheduling model.

```{embed} #simulation-lindblad-ledger
```

The proposed 0.05 error target for one fixed Z mean allocates 0.01 to outer splitting bias, 0.01 to coherent synthesis bias and 0.03 to sampling. The Hamiltonian substep has no inner splitting error in the exact-angle model. A selected-state trace distance $d$ bounds a unit-norm Pauli bias by $2d$; the measured $r=32$ value meets the proposed 0.01 splitting allowance numerically. This is a toy-instance numerical check, not a formal interval or diamond-norm certificate.

Sampling error 0.03 and statistical failure probability 0.005 give a Hoeffding proposal of **13,315 independent complete trials**. That budget applies to one fixed Z mean; X/Y tomography, other initial states, different times and multiple settings require separate confidence allocations. The ledger's seeded ideal-probability draw is illustrative.

Physical-fault probability is separately proposed at 0.005 across all trials. These original logical exports retain continuous-angle gates. A conservative synthesis proposal uses the sum of primitive-gate operator errors and contraction under intervening channel/discard operations; it does not supply a certified Clifford+T sequence. Representation rounding, outer splitting, coherent synthesis, statistical confidence and physical faults are distinct quantities.

Classical coefficient/angle preparation and readout wall time, physical reset/discard/measurement duration, routing, factories and error correction are excluded from the logical ledger. The static physical adapter rejects mid-circuit resets. The separate reset/instrument-aware study below projects the fixed 32-step Z task under its own sampling and physical-failure allocations; the original logical proposal above retains its stated budgets.

(simulation-lindblad-physical-study)=
### Conditional physical-resource study

The [complete-instrument study](../../assets/generated/instrument-resources/README.md) contains damping-Z, Lindblad-Z and thermal X/Z: four measured tasks, eight physical profiles and 23 retained Pareto points. Its [Lindblad report](../../assets/generated/instrument-resources/lindblad-Z/report.json) fixes $H=0.7X+0.3Z$, damping rate 0.8, $t=1$, input $|+i\rangle$, **32 Strang steps** and one Z mean. All source resets and final readouts are retained; the environment result is ignored. This covers the stated two-wire trial, with no physical projection of other step counts or interacting models.

The [source instrument](../../assets/generated/instrument-resources/lindblad-Z/source-instrument.qasm) and [actual Clifford+T instrument](../../assets/generated/instrument-resources/lindblad-Z/clifford-t-instrument.qasm) are separately downloadable. Compilation uses Qiskit 2.6.2 ancilla-free Ross–Selinger approximants {cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`. The saved attempt includes 19,840 T/T-dagger gates, 33 resets and two measurements. Compiler barriers preserve reset/measurement boundaries. Summed numerical Frobenius errors plus guards give $\eta\approx1.55\times10^{-4}$ and the unconditional mean-bias proposal $2\eta$. Small instrument checks compare source, routed and synthesized answers with the Liouvillian reference. These generated sequences provide numerical synthesis evidence beyond a mean-fit T-count proxy; they are not formally certified approximants.

The study sets sampling, physical-fault and factory-availability failure budgets to 0.0025 each. A 0.03 statistical-error target gives **14,855 complete trials**. The original 13,315-trial proposal above continues to use its separate 0.005 statistical-failure budget. No hardware shot ensemble was executed. The study's minimum-qubit retained points cost the entire 14,855-attempt schedule:

| Physical error | Data distance | Peak physical qubits | Capped runtime (hours) |
|---|---:|---:|---:|
| $10^{-3}$ | 29 | 616,927 | 537.805 |
| $10^{-4}$ | 15 | 155,803 | 310.352 |

The separate serialized surface-code scenario uses direct QDK 1.34.0 Trace/ledger agreement {cite:p}`resource_qdk_134`. It reserves seven data/workspace/bridge patches plus one factory and charges system initialization, every environment reuse/reset, final readout, Clifford-frame/controller work and memory exposure throughout the full schedule. Magic-state costs include complete capped factory retries, neighboring SWAP transport, injection, corrective frames and extra port residence. Factory output faults and factory readiness are allocated separately. A general $w$-wire reservation has area $(4w-1)(2d^2-1)$ before adding factory space; the current fixed-width points do not establish interacting-system scaling.

:::{note} Conditional physical assumptions
Protected Pauli-parity/CNOT geometry, tracked Clifford bases and compatible factory/data handoff are assumed {cite:p}`resource_litinski_2019,resource_horsman_2012`. Separate initialization-residual and memory scenarios, independent majority-path readout and controller failure $10^{-15}$ per decision remain uncalibrated. Timing assumes 100 ns gates/initialization, 500 ns readout, 1,000 ns code cycles and 10,000 ns controller reaction. This is not a layout or decoder certificate; arbitrary source-level adaptive feedback is outside the supported fixed schedules and reset/T-correction/dispatch macros.
:::

Offline angle/coefficient preparation, synthesis preprocessing and final statistical-processing wall time are excluded from quantum runtime; actual synthesis time is reported separately. Formal synthesis, interface certification and decoder throughput remain unavailable. See the [generator](../../assets/generated/instrument-resources/generate_instrument_resources.py), [manifest](../../assets/generated/instrument-resources/manifest.json) and README for reproduction and complete exclusions. The frozen Grover baseline is unchanged; this separate instrument scenario supplies no matched co-design improvement claim.

## Run and limits

```bash
python scripts/generate_simulation_lindblad.py
pytest tests/test_simulation_lindblad.py
```

For circuit-only generation beyond the dense-simulation cap:

```bash
python scripts/generate_simulation_lindblad.py --steps 512 --no-sim \
  --output-dir artifacts/lindblad-512
```

Generation is capped at 4,096 steps and an estimated 60,000 primitive operations. Dense circuit checks stop at 256 steps, validation grids at 16 counts per order, and the $4\times4$ reference at an evolution-scale bound of 100. The quantum model stays at one system plus one reusable environment. General interacting Lindblad methods, additional channels and higher-order dissipative algorithms remain planned until their circuits and contracts are implemented.

- {download}`Executable notebook <../../notebooks/simulation_lindblad.ipynb>`
- [Original module](../../assets/generated/simulation/lindblad/lindblad.py), [damping dependency](../../assets/generated/simulation/lindblad/open_systems.py), [standalone generator](../../assets/generated/simulation/lindblad/generate_simulation_lindblad.py)
- [Results](../../assets/generated/simulation/lindblad/results.json), [convergence CSV](../../assets/generated/simulation/lindblad/convergence.csv), [logical ledger](../../assets/generated/simulation/lindblad/logical-resources.json)
- Measured/reset OpenQASM 3: [X](../../assets/generated/simulation/lindblad/lindblad-x.qasm), [Y](../../assets/generated/simulation/lindblad/lindblad-y.qasm), [Z](../../assets/generated/simulation/lindblad/lindblad-z.qasm)
- [MIT implementation license](../../assets/generated/simulation/lindblad/LICENSE.txt) and [pinned environment](../../requirements-lock.txt)

Save the notebook, `lindblad.py` and `open_systems.py` together for an independent run. The text, analytic circuit construction and figures are original. Generated metadata records coefficients, reference convention, source hashes, package versions, seed and CPU backend. Numerical checks and the conditional physical projection have separately recorded scopes; neither supplies a formal certificate.
