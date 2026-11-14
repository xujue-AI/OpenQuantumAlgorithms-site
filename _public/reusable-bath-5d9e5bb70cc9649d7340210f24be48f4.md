---
title: "Thermal and ground states from a reusable bath qubit"
description: "A bounded one-spin full-XX bath implementation, saved finite-channel checks and complete primitive accounting."
---

:::{note} Implementation status: bounded component evidence

An original one-spin implementation has authenticated results from **165
selected tests, two complete source exports, 17 compact SDK/manual gate
comparisons, 81 distinct-frequency ODE solves and six reduced channels**.
These components completed during an overall **failed** validation. Neither
complete exported preparation was quantum simulated.

On **13 November 2026**, a subsequent ordinary local notebook replay
completed all **five code cells** with exit code 0. That replay authenticated
and displayed saved files; it performed no new scientific computation and
does not turn the earlier full validation into a pass. The saved records retain
`all_components_verified=false`.

The general algorithm, a generator for growing system size, full
fault-tolerant accounting, a continuum bridge and a reproduced proof remain
**planned**. The requested absolute-Z error of 0.05 is **uncertified**.
:::

## Motivation and intuition

The separate [local-spin-chain companion](reusable-bath-local-spin.md)
supplies n=1/2 finite-channel checks and generation-only sources through n=64.
Its short four-round examples retain large target biases and no total
Gibbs/ground-state accuracy or physical certificate. The parameters and
evidence below remain this one-spin study's own records.

Ding, Zhan, Preskill and Lin study thermal and ground-state preparation by
sampling a coupling operator and bath frequency, applying forward
system–bath evolution, discarding the bath and resetting one reusable qubit
{cite:p}`frontier_thermal_bath_2026`. Repeated interaction lets the system
exchange energy with fresh bath states. The desired output is the reduced
system state averaged over the random choices and unobserved bath discards.
A recorded choice sequence represents one preparation schedule; it is not a
purification of the averaged output.

This page implements a finite one-spin prototype of that idea. It has no
accepted/rejected branch or bath postselection. The
[thermal-filter examples](thermal-filter.md) use a different preparation
procedure, and the [detailed-balance method](detailed-balance-gibbs.md) has
separately checked Gaussian access and a restricted finite zero-C channel.
Its general correction and mixing remain planned.

## Fundamental method and theorem context

The paper proves end-to-end polynomial dependence on mixing time, inverse
accuracy and temperature, or inverse Hamiltonian gap for ground-state
preparation. For its stated thermal mixing guarantees, the explicit
noninteracting-spin and quadratic-fermion classes use $\beta=\Theta(1)$; a
separate guarantee covers local commuting systems at high temperature
{cite:p}`frontier_thermal_bath_2026`. These are literature conditions. The
finite one-spin checks below retain their own channel, numerical-proxy and
uncertified total-error scope; the general continuum-to-circuit bridge remains
planned.

## Finite task and forward channel

Set $\hbar=1$ and use one system qubit S and one bath qubit E:

$$
H_S=\frac{Z_S}{2},\qquad H_E(\omega)=-\frac{\omega Z_E}{2},\qquad
H_\omega(t)=H_S+H_E(\omega)+0.08 f(t)X_SX_E,
$$

$$
f(t)=\exp\!\left[-\frac{(t-8)^2}{8}\right],\qquad 0\leq t\leq16.
$$

The finite Gaussian has center 8 and width $\sigma=2$. The coupling law is
a point mass at $A_S=X$: this restricted example draws no random coupling
operator. It retains the full XX interaction. In the mathematical basis
$|s,e\rangle$, its even and odd parity sectors are

$$
H_{\mathrm{even}}(t)=\frac{1-\omega}{2}Z+0.08f(t)X
\quad\text{on }\{|00\rangle,|11\rangle\},
$$

$$
H_{\mathrm{odd}}(t)=\frac{1+\omega}{2}Z+0.08f(t)X
\quad\text{on }\{|01\rangle,|10\rangle\}.
$$

Both transitions are present, including the counter-rotating sector. The
source therefore does not replace XX by an energy-exchange-only interaction.

Every round draws a new frequency from the **17-point Gauss–Legendre law on
[0,2]**, whose weights represent the uniform density $1/2$. All 17 nodes
contribute to the channel; the resonant node $\omega=1$ is only one of them.
Thermal mode uses $\beta=1$ and prepares the bath bit with probability

$$p_1(\omega)=\frac{1}{1+e^\omega},\qquad
\rho_E(\omega)=\operatorname{diag}(1-p_1,p_1).$$

An initial reset prepares E in $|0\rangle$, and an explicit X gate prepares
$|1\rangle$ when the sampled bit is one. Ground mode always uses bath
$|0\rangle$. Each round ends with an unobserved bath discard/reset to zero.
After $K=192$ fresh-draw rounds, the system is measured in Z. The system
starts in $|0\rangle$, the excited state of $H_S$; its ground state is
$|1\rangle$.

For a one-round unitary $U_\omega$ and quadrature weights $w_j$, the finite
averaged channel is

$$
\Phi(\rho)=\sum_j w_j\operatorname{Tr}_E\!\left[
U_{\omega_j}\bigl(\rho\otimes\rho_E(\omega_j)\bigr)
U_{\omega_j}^{\dagger}\right].
$$

Fresh independent choices every round give $\Phi^{192}(\rho)$, rather than
an average of fixed-frequency channel powers. The saved deterministic
schedules use PCG64 with seed **20261113** and separate thermal/ground
streams. Those records audit the draws; they do not establish statistical
independence of a physical trial population.

## Explicit gates and the small circuit figure

Each round has $L=256$ positive-time steps, $\Delta t=16/256=1/16$.
For midpoint $t_\ell=(\ell+1/2)\Delta t$, the second-order Strang step is

$$
e^{-iH_0\Delta t/2}
e^{-i0.08f(t_\ell)X_SX_E\Delta t}
e^{-iH_0\Delta t/2},\qquad H_0=H_S+H_E.
$$

Each free half uses $RZ_S(1/32)$ and $RZ_E(-\omega/32)$. The central
$RXX(\theta_\ell)$ has angle $\theta_\ell=0.01f(t_\ell)$ and is emitted as
H on both qubits, CX from S to E, $RZ_E(\theta_\ell)$, CX, then H on both
qubits. Thus every step contains **five RZ, four H and two CX gates: 11
primitives**. Arbitrary rotations remain unsynthesized.

```{figure} ../../assets/generated/simulation/reusable-bath/fixture-thermal.svg
:name: simulation-reusable-bath-fixture
:alt: Original two-wire thermal bath-bit-one fixture with individual reset, X, RZ, H, CX and readout instructions.

The original saved source fixture shows one round, two midpoint steps and
the GL3 central node, with bath bit one. Its 27 instructions include three
resets and final system readout. It illustrates the decomposition and reset
contract; it is not the 192-round GL17 result circuit.
```

The [ground fixture](../../assets/generated/simulation/reusable-bath/fixture-ground.svg)
has 26 instructions because it omits the bath X. Both drawings preserve
the original saved bytes and display every primitive. The main request uses
the complete GL17 law and the finer 256-step schedule.

## Saved gate checks and independent numerical answers

The 17 compact SDK circuits each contain one frequency's complete 256-step
evolution, or 2,816 primitives. Qiskit and an independent manual gate-matrix
path each visited 47,872 primitives. Their largest same-schedule operator
residual is **$6.90\times10^{-15}$**. The comparison explicitly converts
Qiskit's $|e,s\rangle$ matrix order to the mathematical $|s,e\rangle$ order.
Compact operator checks contain evolution only; they do not simulate the
full reset/readout preparation.

Independent full-Hamiltonian SciPy DOP853 integration completed **81
distinct-frequency solves**, using $\mathrm{rtol}=10^{-12}$ and
$\mathrm{atol}=10^{-14}$ with no tolerance retry. Four physical inputs,
$|0\rangle$, $|1\rangle$, $|+\rangle$ and $|+i\rangle$, span each reduced
map. The saved six channels comprise thermal and ground versions of T17
(the midpoint GL17 channel), Q17_hat (the 17-node ODE proxy) and C65_hat
(the 65-node ODE proxy).

The Gibbs answer is independently computed and normalized:

$$
\rho_\beta=\frac{e^{-H_S}}{\operatorname{Tr}(e^{-H_S})}
=\operatorname{diag}(0.26894142136999516,\ 0.731058578630005),
\qquad \langle Z\rangle_\beta=-\tanh(1/2).
$$

The independent ground projector is $|1\rangle\langle1|$. The saved T17
endpoint from initially excited $|0\rangle$ is:

| Mode | Final Z | Target Z | Absolute Z residual | Ground population |
|---|---:|---:|---:|---:|
| Thermal, $\beta=1$ | −0.4505592849840573 | −0.46211715726000985 | 0.011557872275952563 | 0.7252796424716395 |
| Ground | −0.9955185814468179 | −1 | 0.0044814185531820705 | 0.9977592907030252 |

```{figure} ../../assets/generated/simulation/reusable-bath/endpoints.png
:name: simulation-reusable-bath-endpoints
:alt: Saved thermal and ground finite-channel endpoint and target comparisons with residual and proxy diagnostics.

Original saved finite-channel diagnostics. Negative Z endpoints and
nonzero target residuals are retained; these numerical comparisons are
not a continuum or total-accuracy certificate.
```

Channel outputs, Choi matrices and powered endpoint states are kept raw,
without trace renormalization. The T17 final traces are
0.9999999999592216 (thermal) and 0.9999999999592324 (ground). The independent
Gibbs target is normalized separately. Full-state trace-norm diagnostics
and absolute-Z residuals have separate output contracts in the
[results](../../assets/generated/simulation/reusable-bath/results.json).

For the same initially excited input after 192 rounds, the saved
trace-norm comparison proxies are:

| Comparison | Thermal | Ground |
|---|---:|---:|
| T17 versus Q17_hat | $1.1162087332\times10^{-7}$ | $2.1083499924\times10^{-7}$ |
| Q17_hat versus C65_hat | $7.1609385088\times10^{-14}$ | $1.5532020115\times10^{-13}$ |

These compare finite numerical constructions. They are neither rigorous
Trotter/quadrature bounds nor evidence that the finite pulse has the exact
Gibbs fixed point. T17's numerical fixed-point target-Z residual remains
0.011555558369699404 in thermal mode and 0.0044782353234742445 in ground
mode. The saved affine mixing diagnostic also excludes raw trace drift and
the exact continuous-frequency channel. Finite pulse/weak-coupling bias,
the continuum bridge, classical-law representation error, synthesis and
physical faults still require separate allowances. Small proxy differences
do not certify the requested combined error of 0.05.

## End-to-end resources and fault tolerance

The two emitted full source schedules include initialization, every forward
gate, optional bath X, every bath reset and final system measurement. They
contain one system qubit, one bath qubit and one output bit. Each complete
preparation has **194 resets**: one system reset and 193 bath resets,
including initialization and all 192 round-end discards. Readout is one
system measurement; there is no environment measurement or postselection.

| Primitive | Emitted ground schedule | Emitted thermal schedule |
|---|---:|---:|
| RZ | 245,760 | 245,760 |
| H | 196,608 | 196,608 |
| CX | 98,304 | 98,304 |
| Bath X | 0 | 55 |
| Reset | 194 | 194 |
| Measure | 1 | 1 |
| **Total instructions** | **540,867** | **540,922** |

Both emitted schedules require 192 frequency draws; thermal mode also
requires 192 bath-bit draws. Recorded choices are supplied to the emitter,
which draws no randomness itself. The two files contain **1,081,789
primitives in total** and are source exports, without a full-schedule
quantum simulation. The thermal total ranges from **540,867 to 541,059**
for zero through 192 bath X gates; 540,922 is the actual emitted example.
The [actual-emission ledger](../../assets/generated/simulation/reusable-bath/primitive-ledger.json)
distinguishes these counts from historical formula-only projections.

For the same reset-prepared $|0\rangle$ input, $K$ rounds and $L$ steps per
round give a complete ground preparation with
$5KL$ RZ, $4KL$ H, $2KL$ CX, $K+2$ resets and one measurement, totaling
$11KL+K+3$ instructions. Thermal preparation adds between zero and K X
gates. A repeated Z trial reruns the entire preparation. M frequency nodes
affect classical coefficient/law storage and reference work; they do not
multiply the gates in a sampled round. Scalar K/L/M/P counting is separate
from bounded source emission and does not generate coefficients, circuits
or reference jobs. This is one-spin parameter counting, **not a generator
for large n**.

The saved statistical workload is **29,958 complete preparations per mode,
59,916 readouts total**, with an all-mode upper bound of 32,412,339,108
instructions. It is an **unexecuted projection**, not a measured sampling
population. The 128 recorded schedules per mode are classical schedule
audits. They are not 128 prepared quantum states.

Clifford+T synthesis, T count/depth, routing, reset/controller noise and
timing, idle exposure, factories, code distance, physical qubits/runtime
and total physical success remain **uncomputed**. The primitive totals
therefore do not establish a fault-tolerant cost or superiority. Any future
co-design comparison must use the same task, accuracy, success and physical
assumptions; the frozen surface-code baseline is unchanged.

## Verification, formalization and implementation sources

The retained 165 selected tests comprise 142 source/runner tests and 23
reference tests. Their completion, source exports and finite numerical
checks are authenticated component evidence. The overall validation failed
during notebook artifact generation; the later successful saved-file
replay establishes notebook usability only. A reproduced mechanized proof,
general-system implementation and exact continuum-to-circuit bridge remain
planned.

The motivating paper is cited as external algorithmic evidence
{cite:p}`frontier_thermal_bath_2026`; this restricted prototype does not
reproduce its general theorem. The implementation, explanations and circuit
drawings are original. The recorded implementation environment is Python
**3.12.14**, NumPy **2.5.4** (BSD-3-Clause), SciPy **1.18.1** (BSD-3-Clause)
and Qiskit **2.6.2** (Apache-2.0). Exact SciPy/Qiskit upstream revisions are
listed in the bibliography {cite:p}`sim_scipy_1181,sim_qiskit_262`;
the saved [environment lock](../../assets/generated/simulation/reusable-bath/source/requirements-lock.txt)
and [scientific protocol](../../assets/generated/simulation/reusable-bath/source/reusable_bath_protocol.json)
record the package versions and finite parameters.

The preserved source exposes the following public, standard-library
interfaces. Load the sibling protocol **explicitly** from the downloaded
`source/` folder, using `load_protocol(Path("reusable_bath_protocol.json"))`;
this selects the supplied scientific record directly.

| Interface | Scope |
|---|---|
| `count_source(K, L, M, mode, P=1)` | Scalar counts only; no circuit, coefficient generation or simulation. |
| `iter_round_operations(omega, steps, protocol=protocol)` | One round's evolution primitives. |
| `iter_operations(protocol, frequency_indices, bath_bits, steps=..., nodes=..., mode=..., initial="0")` | Complete bounded preparation primitives, including reset and readout; choices are supplied by the caller. |
| `QASM_HEADER` and `operation_qasm(operation)` | Serialize those primitives with an ordinary caller-owned file writer. |

These are API descriptions, not a new executed example. Complete source
emission is restricted to one system spin, at most 192 rounds, 256 steps,
17 nodes and 600,000 instructions. Scalar counting is a separate interface
with a metadata-size limit. The preserved `write_qasm` helper belongs to
the managed exporter; it is not the ordinary public file-writing entry point.
The included reference-source snapshot is for inspection and expects a
custom bounded context, rather than supplying a standalone public validator.

Download the [companion notebook](../../notebooks/simulation_reusable_bath.ipynb)
and [saved evidence ZIP](../../assets/generated/simulation/reusable-bath-evidence.zip).
Unzip beside the notebook to create `reusable-bath-evidence/`, then open it
in the locked Python 3.12 Jupyter environment with the ordinary Python 3
(ipykernel) kernel. Run All authenticates the manifest and 18 payload files
and displays saved JSON, SVG, PNG and QASM metadata. It does not import the
scientific source or repeat a test, simulation, ODE solve, source export or
resource calculation. The five-cell ordinary local replay passed on
13 November 2026; full validation and the uncomputed scientific/physical
contracts retain their separate status.

- [Saved-file manifest](../../assets/generated/simulation/reusable-bath/artifact-manifest.json), [endpoint records](../../assets/generated/simulation/reusable-bath/endpoints.json) and [primitive ledger](../../assets/generated/simulation/reusable-bath/primitive-ledger.json)
- Complete emitted sources: [ground QASM](../../assets/generated/simulation/reusable-bath/source-ground.qasm), [thermal QASM](../../assets/generated/simulation/reusable-bath/source-thermal.qasm) and [export metadata](../../assets/generated/simulation/reusable-bath/source-exports.json)
- [Restricted one-spin source](../../assets/generated/simulation/reusable-bath/source/reusable_bath.py), [scientific protocol](../../assets/generated/simulation/reusable-bath/source/reusable_bath_protocol.json) and [MIT license](../../assets/generated/simulation/reusable-bath/source/LICENSE.txt)

## References

```{bibliography}
```
