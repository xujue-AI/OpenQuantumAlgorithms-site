---
title: "A reusable bath for local spin chains"
description: "Emitted primitive sources for a growing diagonal spin chain, completed bounded finite-channel checks and saved target-bias diagnostics."
---

:::{note} Status: bounded component validation completed

On **13 November 2026**, one bounded run completed all six stages with CLI
exit code 0 and externally observed elapsed time **9.481859583 seconds**.
It passed **10 source-contract tests**, emitted **nine complete QASM sources
with 66,680 instructions**, checked **nine compact SDK Operators, eight
finite averaged channels and four complete tiny density programs**, and
replayed **five saved-only notebook cells** in ordinary Python 3, version
**3.12.14**. Independent review of the saved evidence also passed; it
retains the finite-component scope.

The implementation comparisons passed, but the four-round outputs retain
**large target biases**. This is completed finite-component evidence, with
the raw results and poor preparation quality preserved below.

`all_components_verified=false` remains the scope of this work. A continuous
evolution comparison, detailed-balance or mixing proof, Gibbs or ground-state
accuracy certificate, sampling confidence and physical resource estimate
remain planned or uncomputed. The earlier
[one-spin reusable-bath page](reusable-bath.md) has separate parameters and
component records; none of its numerical outputs establishes evidence for
this model.
:::

## Motivation and literature context

Forward interaction with a qubit bath offers a small environment register:
choose a system coupling and bath frequency, prepare the bath, evolve the
joint system, then discard and reset the bath. Ding, Zhan, Preskill and Lin
analyze this approach for thermal and ground-state preparation
{cite:p}`frontier_thermal_bath_2026`. Here the coupling is an actual local X
operator on a sampled site, and the source expands every forward operation
into visible gates. Increasing the system size changes the Hamiltonian and
the emitted register and gates.

The paper's end-to-end guarantees depend on its access normalization,
temperature or spectral-gap assumptions and mixing conditions. Its stated
thermal mixing examples include noninteracting spins and quadratic fermions
at $\beta=\Theta(1)$, and a separate result for local commuting systems at
high temperature {cite:p}`frontier_thermal_bath_2026`. The diagonal chain
below is local and commuting, but that class membership does not establish
the paper's normalization, high-temperature or mixing hypotheses for this
particular pulse and sampler. No theorem is reproduced here. The finite
channel may retain bias relative to the Gibbs or ground target, even when
its circuit implementation agrees with the independent reference.

## Frozen computational task

Use $n$ system spins indexed from $j=0$ and one reusable bath qubit E at
wire index $n$. With $\hbar=1$, define the open chain

$$
H_S^{(n)}=\frac12\sum_{j=0}^{n-1}h_jZ_j
+J\sum_{j=0}^{n-2}Z_jZ_{j+1},\qquad
h_j=1+\frac{j\bmod 3}{8},\qquad J=\frac18,
$$

$$
H_E(\omega)=-\frac{\omega Z_E}{2},\qquad
H_{a,\omega}(t)=H_S^{(n)}+H_E(\omega)+g(t)X_aX_E,
\qquad g(t)=0.08e^{-2(t-1)^2},\quad 0\le t\le2.
$$

Every round chooses a uniform site $a\in\{0,\ldots,n-1\}$ and a fresh
frequency from the analytic GL3 law representing uniform density $1/2$ on
$[0,2]$:

$$
(\omega_0,\omega_1,\omega_2)
=\left(1-\sqrt{\frac35},\ 1,\ 1+\sqrt{\frac35}\right),
\qquad (w_0,w_1,w_2)=\left(\frac5{18},\frac49,\frac5{18}\right).
$$

Thermal mode fixes $\beta=1$ and samples the bath bit with
$p_1(\omega)=1/(1+e^\omega)$. Ground mode always uses bath $|0\rangle$.
All $n+1$ qubits are initially reset to zero; an explicit X on E prepares a
sampled bath one. Each round ends with an unobserved bath discard/reset,
including the last round. Terminal measurements map system $q_j$ to output
bit $c_j$ in Z. There is no bath measurement or postselection.

The output contract is the system state, its individual Z expectations and
the final computational-basis distribution after $K$ rounds. For the finite
$L$-step round unitary $U_{a,r,L}$, the averaged channel is

$$
\Phi_{n,m,L}(\rho)=\frac1n\sum_{a=0}^{n-1}\sum_{r=0}^{2}w_r
\sum_{b=0}^{1}p_m(b\mid\omega_r)
\operatorname{Tr}_E\!\left[
U_{a,r,L}(\rho\otimes|b\rangle\langle b|)U_{a,r,L}^{\dagger}\right].
$$

Fresh independent choices in the modeled preparation give
$\Phi_{n,m,L}^{K}(|0\cdots0\rangle\langle0\cdots0|)$.
A fixed seeded schedule instead gives its conditional sequence of channels.
Those are distinct objects, and a recorded schedule is not an experimental
population or a purification of the averaged output.

The scheduler uses Python's `random.Random(MT19937)` with choice
order site, frequency, then thermal bath bit. It records K logical site
calls and K frequency calls, plus K thermal bit calls. A site call is made
even at $n=1$; ground mode makes no bath-bit call. These count the public
`randrange`/`random` calls, without counting internal generator words or
measuring classical controller time. The emitter consumes recorded choices
and draws no randomness.

## Explicit positive-time primitives

Set $\Delta t=2/L$ and midpoint $t_\ell=(\ell+1/2)\Delta t$. Each step is
the unfused Strang product

$$
e^{-i(H_S^{(n)}+H_E)\Delta t/2}
e^{-ig(t_\ell)X_aX_E\Delta t}
e^{-i(H_S^{(n)}+H_E)\Delta t/2}.
$$

All terms in the free Hamiltonian commute, so each free half is represented
exactly by its listed logical rotations. This does not remove the splitting
error between the free and XX parts or the midpoint time-discretization
error. With $RZ(\theta)=e^{-i\theta Z/2}$, the source grammar is:

| Block | Explicit instruction sequence | Primitives |
|---|---|---:|
| Field $j$ in one free half | $RZ_j(h_j\Delta t/2)$ | 1 RZ |
| Bond $(j,j+1)$ in one free half | $CX_{j,j+1}$, $RZ_{j+1}(J\Delta t)$, $CX_{j,j+1}$ | 1 RZ, 2 CX |
| Bath in one free half | $RZ_E(-\omega\Delta t/2)$ | 1 RZ |
| Full XX middle | $H_a,H_E,CX_{a,E},RZ_E(2g(t_\ell)\Delta t),CX_{a,E},H_a,H_E$ | 1 RZ, 2 CX, 4 H |

The first free half lists fields, then bonds, then bath. The second reverses
that gate order **with the same rotation signs**. It is another positive-time
half, rather than an inverse evolution. The seven-instruction middle keeps
the full XX interaction, including its counter-rotating transitions.
Adjacent free halves are retained in the source without fusion.

Thus each step has $4n+1$ RZ, $4n-2$ CX and four H instructions, or
$8n+3$ unitary primitives. Arbitrary RZ gates are unsynthesized.

```{figure} ../../assets/generated/simulation/reusable-bath-local-spin-presentation/first-step.svg
:name: simulation-local-spin-first-step
:alt: The first Strang step of the saved two-system-spin ground-mode source, including three initial resets and nineteen RZ, CX and H instructions.

The first Strang step of the validated `n2-ground.qasm` program: q0 and q1
are system spins, and q2 is the bath. The excerpt preserves the three initial
resets and 19 unitary instructions. Qiskit's drawer rounds the displayed
angles; the complete QASM retains their stored precision. Later steps,
bath resets and terminal system readouts are outside this excerpt, with c0
and c1 reserved for those final readouts. This is a drawing of saved gates,
not a separate quantum or reference execution.
```

The [small presentation renderer](../../assets/generated/openquantum-code.zip)
reads the authenticated source prefix and uses Qiskit 2.6.2's circuit drawer
{cite:p}`sim_qiskit_262`. Its [presentation record](../../assets/generated/simulation/reusable-bath-local-spin-presentation/presentation.json)
is separate from the unchanged 21-file finite-validation packet.

## Growing source and complete preparation accounting

The standard-library generator supports the intersection of
$1\le n\le64$, $1\le K\le64$, $1\le L\le128$, **100,000 complete
instructions per file** and **8 MiB per QASM file**. The interval maxima
cannot all be selected together. Admission uses the thermal upper count
before source writing and checks actual counts and bytes during streaming.
The case specification is bounded to nine cases and 128 KiB. Generation
holds a bounded schedule and free-half list while streaming operations;
it does not allocate a statevector or density matrix.

For $x$ bath-one choices, a complete preparation has

$$
I(n,K,L,x)=(8n+3)KL+2n+K+1+x.
$$

This includes $n+1$ initial resets, K post-round bath resets and n terminal
measurements. Logical qubits are $n+1$ and output bits are n. The fixed
primitive counts are $(4n+1)KL$ RZ, $(4n-2)KL$ CX, $4KL$ H,
$n+K+1$ resets and n measurements, with x X gates. Ground mode has x=0;
thermal mode has $0\le x\le K$.

:::{warning} Scope of the cost

These formulas include preparation, coupling construction, every round,
reset and readout, plus separately recorded logical random-call counts.
Rotation synthesis, T count/depth, routing, scheduling, idle exposure,
controller/reset/readout timing and noise, sampling confidence, factories,
code distance, physical qubits/runtime and total physical success are
**uncomputed**. The ideal Hamiltonian duration $2K$ is not a physical runtime.
No fault-tolerant advantage or accuracy claim follows from the primitive count.
:::

`count_source(..., repetitions=P)` projects P complete preparations,
including fresh-choice calls, all resets and all readouts for every trial.
Write $B=(8n+3)KL+2n+K+1$ for the fixed instruction count. For one supplied
schedule at P=1, its recorded x gives the exact projected count $B+x$.
Fresh thermal repetitions with P>1 have total X count in $[0,PK]$ and
instruction range $[PB,P(B+K)]$; one recorded schedule's x is not multiplied
across fresh trials. Ground repetitions have count PB. Counting supports
$P\le1,000,000$ with a 64 KiB metadata limit and performs scalar counting
only. It neither generates the P schedules nor establishes a sample size or
confidence interval. `write_qasm` exports exactly one complete preparation
for one supplied schedule.
The three frequency nodes determine a classical law and reference workload;
they do not multiply the gates in a sampled round. Any later architecture
comparison must retain the same task, accuracy, success and physical
assumptions and the frozen surface-code baseline.

The executed frozen packet contains four small checks and five larger source exports:

| Case | K | L | Role | Emitted bath X | Emitted complete instructions |
|---|---:|---:|---|---:|---:|
| n=1, ground | 4 | 8 | Small validation | 0 | 359 |
| n=1, thermal | 4 | 8 | Small validation | 1 | 360 |
| n=2, ground | 4 | 8 | Small validation | 0 | 617 |
| n=2, thermal | 4 | 8 | Small validation | 1 | 618 |
| n=4, thermal | 4 | 16 | Generation only | 0 | 2,253 |
| n=8, thermal | 4 | 16 | Generation only | 0 | 4,309 |
| n=16, thermal | 4 | 16 | Generation only | 2 | 8,423 |
| n=32, thermal | 4 | 16 | Generation only | 1 | 16,646 |
| n=64, thermal | 4 | 16 | Generation only | 2 | 33,095 |

The [generation ledger](../../assets/generated/simulation/reusable-bath-local-spin/generation.json)
records **66,680 actually emitted instructions**, matching the fixed
classical schedules. The same cases range from **66,673 to 66,701** over
allowed thermal bits. The saved schedules record 36 logical site calls,
36 frequency calls and 28 thermal bit calls. Every $n>2$ case is
**generation only**, with no state or density simulation claim.

The ledger also retains the counting helper's
`status="formula_only_unexecuted_projection"` and
`accuracy.finite_circuit_reference="unexecuted"`. These annotate the
formula-only accounting object; the helper performs no reference check.
Actual small-case execution and its scope are recorded separately in
`reference.json` and `validation.json`. Repetition projections and all
uncomputed accuracy and physical allowances retain their separate status.

## Completed independent small checks

The independent reference is restricted to the four $n=1,2$, $K=4$, $L=8$
cases. Its joint dimension is at most 8, with at most three qubits; its
system dimension is at most 4. It requires actual emitted small-source
operation records. Missing records cause an error rather than substituted
fixtures or a newly generated schedule.

| Completed check | Bounded work | What it compares |
|---|---|---|
| Compact SDK `Operator` checks | Nine rounds; 1,176 primitive visits | Every site/frequency combination against independently constructed diagonal free factors and analytic XX matrix products; 72 Strang steps in total |
| Full matrix-unit channel construction | 432 raw branch outputs; eight averaged channels | SDK and reference linear maps on every $E_{ab}$, including off-diagonal matrix units |
| Raw channel powers and Choi diagnostics | Eight fourth powers and eight Choi checks | Finite-channel consistency, numerical complete positivity and trace preservation |
| Complete tiny SDK `DensityMatrix` checks | Four sources; 1,948 premeasurement primitive visits | Actual reset/X/evolution/reset sources against their conditional dense reference outputs |

The dense path enumerates the frozen Ising energies independently and uses
diagonal phases and the analytic XX cosine/sine update. It does not reuse the
generator's field, pulse, angle or gate-matrix helpers. It calls no ODE solver
or matrix exponential. SDK matrices are converted from Qiskit's bit order
to the mathematical order with $q_0$ the most significant system bit,
followed by the bath bit.

The four complete small sources contain 1,954 instructions. Only their six
terminal system measurements are removed for the density checks. All
$n+1$ initial resets and all four post-round bath resets remain, including
the last bath reset. The SDK starts from an external all-ones joint state
to check initialization. The final joint state is compared with the
conditional system output tensored with bath zero, so the final reset is
checked as an operation on the joint state. Readout mapping and analytic
Born probabilities are recorded without shot sampling.

The target comparison independently normalizes
$e^{-H_S^{(n)}}$ for the thermal target and constructs the ground projector
from the enumerated energies. It retains raw state traces, matrices,
individual-Z residuals, full-state trace norms and finite target bias.
No renormalization, symmetrization, clipping, outcome-based seed change,
tolerance retry or bias tuning was used. Compiler and finite-channel
agreement does not establish continuous-time accuracy, detailed balance,
mixing, convergence to Gibbs, or a combined error certificate.

The largest saved compact Operator residual is
$5.332170238093881\times10^{-15}$. The SDK/reference residual maxima are
$7.884250117626704\times10^{-15}$ for averaged channels,
$3.1086000605167327\times10^{-14}$ for fourth powers and
$2.864376955436136\times10^{-14}$ for averaged endpoint states. The four
complete tiny sources have maximum final joint and reduced-system residual
$2.7089456695977975\times10^{-14}$. These passed the fixed finite-consistency
guard of $10^{-10}$; that guard is not a target-accuracy allowance.

For the averaged **independent reference** channel applied four times to the
reset-prepared all-zero input, the saved target diagnostics are:

| Case | Final mean Z, real part | Target mean Z | Full trace-norm target bias $\|\rho-\rho_{\rm target}\|_1$ | Largest absolute site-Z error |
|---|---:|---:|---:|---:|
| n=1, ground | 0.9324348222920665 | −1 | 1.9324348222920666 | 1.9324348222920666 |
| n=1, thermal | 0.9405366069783381 | −0.46211715726000985 | 1.402653764238348 | 1.4026537642383479 |
| n=2, ground | 0.9664581568340309 | −1 | 1.999571493054993 | 1.9666779948274782 |
| n=2, thermal | 0.9712503417764292 | −0.43838497123434833 | 1.8239120562856725 | 1.4375961670069155 |

The trace norm is the **sum of singular values, without a factor of one
half**. These are poor preparation results at the chosen K=4 and L=8:
all outputs retain positive mean Z while the targets have negative mean Z.
The n=2 ground-state bias is nearly the full trace-norm separation of 2.
Raw traces, tiny imaginary parts and complete matrices remain in the
[reference report](../../assets/generated/simulation/reusable-bath-local-spin/reference.json);
the largest raw endpoint trace residual among these four references is
$1.9986368334542577\times10^{-15}$. No parameters were changed in response
to the observed target bias.

### Reproduce the finite checks with ordinary Python

Download the [reproduction harness](../../assets/generated/simulation/reusable-bath-local-spin-reproduction/source/reusable_bath_local_spin_reproduce.py),
the [frozen assertions](../../assets/generated/simulation/reusable-bath-local-spin-reproduction/source/reusable_bath_local_spin_checks.py)
and the existing [saved-evidence ZIP](../../assets/generated/simulation/reusable-bath-local-spin-evidence.zip).
Extract the ZIP beside the two Python files. Use Python 3.12 with the supplied
[environment lock](../../assets/generated/simulation/reusable-bath-local-spin/source/requirements-lock.txt).
The recorded run used Python **3.12.14**, NumPy **2.5.4**, Qiskit **2.6.2** and
psutil **7.2.2**. From that directory, run:

```sh
python reusable_bath_local_spin_reproduce.py \
  --records-dir reusable-bath-local-spin-evidence \
  --checks reusable_bath_local_spin_checks.py \
  --output-dir local-spin-reproduction-1 \
  --max-seconds 120
```

Use real filesystem paths without symbolic-link ancestors. The output folder
must be new and outside the accepted evidence folder. Run in a fresh Python
process on its main thread; concurrent or reentrant use and optimized Python
(`-O`/`-OO`) are refused. The harness supplies its bounded ledger and context
automatically, without a private driver, manual module setup, approval
argument, SSH connection or private environment stack.

The route verifies all **21 accepted input files**, loads the generator and
reference snapshots, and decodes the four actual small QASMs into **1,954
operations**. It calls **ten direct Python assertion contracts** and
`validate_small_cases` once. These direct calls have no pytest
setup/call/teardown report claim. Copies of the nine QASMs and cases are
staged in the fresh output; the originals and recorded choices remain
unchanged. Quantum matrix work is restricted to $n=1,2$, with maximum
joint/system/Choi dimensions **8/4/16**. The larger files receive source
checks only.

On **14 November 2026**, the single ordinary CLI exited with code **0** in
**4.17963525 externally observed seconds**. It completed nine compact
Operators, eight channels and four tiny density programs, with 1,948 density
primitive visits, 460 partial traces and 62 state checks. New source work
was **1,926 primitives**: 750 assertion-fixture primitives and 1,176 compact
primitives. The single newly generated QASM is a 15-operation fixture,
**290 bytes**. There were no new scientific RNG choices, production-export
regeneration, notebook, ODE/expm, shots, Native/synthesis or GPU/remote work.
The [CLI summary](../../assets/generated/simulation/reusable-bath-local-spin-reproduction/reproduction.json)
and [provenance](../../assets/generated/simulation/reusable-bath-local-spin-reproduction/provenance.json)
record these counters and the harness/input hashes. The
[finite replay report](../../assets/generated/simulation/reusable-bath-local-spin-reproduction/finite-reference.json)
is byte-identical to the accepted finite reference above. Independent saved-output
postflight passed; the [interface completion record](../../assets/generated/simulation/reusable-bath-local-spin-reproduction/interface-validation.json)
binds the observed CLI exit to this scope. This is public
interface evidence; the poor target biases and `all_components_verified=false`
remain explicit.

The 120-second API budget uses cooperative **15/30/60/15-second** stages,
an own-process RSS sampling threshold of **1 GiB**, and **16 MiB / 64-file** output
admission with a **1 MiB** failure reserve. These are guard checkpoints, not
hard native interruption or disk quotas. A caller-owned outer deadline must
also cover interpreter startup and terminal delivery outside the API clock.
The route has no automatic retry or tuning path. Mixing, continuum accuracy,
rotation synthesis and physical budgets remain the next scientific gaps.

## Public source and saved-evidence notebook

The public interfaces are ordinary Python APIs:

| Interface | Contract |
|---|---|
| `Case(case_id, n, rounds, steps, mode, seed)` | One bounded preparation with the fixed model, pulse, input and readout |
| `load_cases(path)` | Read the supplied bounded case JSON; no draws, gates or numerical states |
| `make_schedule(case)` | Record seeded site, frequency and bath-bit choices |
| `iter_operations(case, schedule)` | Stream the complete preparation primitives |
| `count_source(case, schedule=None, repetitions=1)` | Formula-only accounting, with thermal ranges when no schedule is supplied |
| `write_qasm(case, schedule, path)` | Write one bounded QASM source; ordinary callers need no custom ledger |

The generator imports only the standard library. The reference imports
NumPy and Qiskit lazily after admitting the strictly small inputs. The
recorded locked environment is Python **3.12.14** with NumPy **2.5.4** (BSD-3-Clause)
and Qiskit **2.6.2** (Apache-2.0); the exact Qiskit revision is credited in
the bibliography {cite:p}`sim_qiskit_262`. The implementation and explanation
are original, with the project's MIT license. Source presence is not an
execution record or a mechanized proof.

The [ordinary Python 3 companion notebook](../../notebooks/simulation_reusable_bath_local_spin.ipynb)
has five frozen code-cell sources that completed the saved-only replay in
ordinary Python 3.12.14. The source notebook begins with cleared outputs;
publication replays those same cells to display the saved records. They read
saved JSON and file metadata only and perform no new scientific computation.

Download the notebook and the
[saved-evidence ZIP](../../assets/generated/simulation/reusable-bath-local-spin-evidence.zip).
Extract the ZIP beside the notebook to create
`reusable-bath-local-spin-evidence/`, then select the ordinary Python 3
(ipykernel) kernel and Run All. The repository's
`assets/generated/simulation/reusable-bath-local-spin/` folder and running
from the evidence folder itself are also supported. The ZIP has **21 file
members** and is **93,215 bytes**, containing **2,066,516 bytes** of
saved files. The notebook code does not import
the source snapshots, construct gates, run the reference, choose a numerical
tolerance, draw trials or estimate resources.

The public packet contains these saved files:

- [Cases](../../assets/generated/simulation/reusable-bath-local-spin/cases.json), [generation ledger](../../assets/generated/simulation/reusable-bath-local-spin/generation.json), [recorded schedules](../../assets/generated/simulation/reusable-bath-local-spin/schedules.json), [finite reference report](../../assets/generated/simulation/reusable-bath-local-spin/reference.json) and [validation status](../../assets/generated/simulation/reusable-bath-local-spin/validation.json).
- Nine QASM files under `source-exports/`, for the nine case IDs in the table. These are generation artifacts; the five larger files have no state or density simulation claim.
- [Generator snapshot](../../assets/generated/simulation/reusable-bath-local-spin/source/reusable_bath_local_spin.py), [reference snapshot](../../assets/generated/simulation/reusable-bath-local-spin/source/reusable_bath_local_spin_reference.py), [case specification](../../assets/generated/simulation/reusable-bath-local-spin/source/cases.json), [environment lock](../../assets/generated/simulation/reusable-bath-local-spin/source/requirements-lock.txt) and [MIT license](../../assets/generated/simulation/reusable-bath-local-spin/source/LICENSE.txt).
- [Manifest](../../assets/generated/simulation/reusable-bath-local-spin/artifact-manifest.json) and its [public SHA-256 sidecar](../../assets/generated/simulation/reusable-bath-local-spin/artifact-manifest.sha256).

The notebook checks regular portable paths, byte lengths, payload hashes,
the manifest's `local-spin-bath-saved-evidence` kind and the frozen case-spec
digest. It admits at most 32 files and 32 MiB, and keeps
`all_components_verified=false`. The sidecar is a public integrity check,
not a cryptographic validation authority. The saved `validation.json`
provides `status="bounded_component_validation_completed"`; the reference
report records `status="finite_component_checks_passed"` and
`validation_passed=true`, with `all_components_verified=false`.
The notebook displays recorded evidence; its successful replay establishes
saved-file usability rather than a new scientific result.

## References

```{bibliography}
```
