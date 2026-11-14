---
title: "Interval reciprocal QSVT"
description: "An odd inverse approximation on a signed spectral interval, with explicit Pauli access, reproducible phases and full readout costs."
downloads:
  - file: ../../notebooks/linear_systems_qsvt_reciprocal.ipynb
    title: "Executable notebook"
  - file: ../../assets/generated/linear-systems/reciprocal-qsvt/measured-trial.qasm
    title: "Complete measured trial"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Implementation and tests"
---

A linear-system algorithm prepares a state proportional to $A^{-1}|b\rangle$. In the eigenbasis of a Hermitian $A$, this means multiplying each input amplitude by the reciprocal of its signed eigenvalue. Small eigenvalues need more amplification, so a lower bound on their magnitude is part of the input contract.

QSVT realizes a bounded polynomial transformation through repeated matrix-access circuits and projector phases {cite:p}`linsys_qsvt2019`. Here an odd polynomial approaches a scaled reciprocal across a **promised signed interval**. It applies to more than the special eigenvalues of the [spectrum-matched cubic](qsvt.md). Related linear-system algorithms and a tutorial on polynomial transformations provide the wider context {cite:p}`linsys_cks2017,reciprocal_qsvt_martyn2021`.

:::{note} Implemented scope

The implementation supplies an original reciprocal polynomial, bounded deterministic phase fitting, explicit Hermitian Pauli PREPARE/SELECT/unprepare, a gate-built complex input and complete measured trials. Small circuits have independent numerical reference checks. The polynomial error is analytic; numerical access-angle construction and physical synthesis have separate limits. General non-Hermitian loading, an asymptotically optimal reciprocal compiler and Lean formalization remain planned.
:::

## Input, output and motivation

Supply real coefficients and Pauli words,

$$
A=\sum_{j=0}^{M-1}a_jP_j,\qquad
\alpha=\sum_j|a_j|,\qquad B=A/\alpha,
$$

together with the promise $\delta\le|\lambda(B)|\le1$. Here $M$ counts nonzero terms after equal Pauli words have been combined. This allows positive and negative eigenvalues; $A$ must be Hermitian and nonsingular. The normalized condition number is at most $1/\delta$. The code accepts exact rational or decimal coefficient inputs, and builds gates from their explicit Pauli representation. Dense matrices, eigenvectors and inverse solutions are used only by bounded reference checks.

The default two-data-qubit example is

$$A=15\,II-2\,IX-2\,ZI-XX,\qquad\alpha=20,\qquad\delta=1/2.$$

Its four distinct normalized eigenvalues are approximately .53820, .73820, .76180 and .96180; its actual condition number is about 1.787. Qiskit's rightmost Pauli character acts on q0. Input preparation uses H, S and RY(.63) on q0, followed by RY(.37) on q1 and CX(q0,q1). This constructs a complex entangled input independently of the solution.

The requested output is one **conditional $Z(q_0)$ expectation** in the normalized inverse state. Preparing that state does not return all entries of a classical solution or its overall norm. The classical reference solves the same small matrix and compares the same normalized observable.

## A regularized reciprocal polynomial

Let $m=K+1$, $c=1/\lceil\sqrt m\rceil$, and define

$$
p_K(x)=c\,x\sum_{j=0}^{K}(1-x^2)^j
      =\frac{c}{x}\bigl[1-(1-x^2)^m\bigr]\quad(x\ne0).
$$

The finite sum also defines $p_K(0)=0$. It is odd with degree $d=2K+1$. For $0<|x|\le1$,

$$
\left|x\sum_{j=0}^{K}(1-x^2)^j\right|
\le\min(m|x|,1/|x|)\le\sqrt m,
$$

and the value at zero is separately zero. Thus $|p_K(x)|\le1$ across the whole signal interval, including the excluded gap around zero. This boundedness is essential to the QSVT construction; fitting only the wanted eigenvalues would not establish it.

Set $E=(1-\delta^2)^m$. On the promised signed spectrum,

$$
\|p_K(B)-cB^{-1}\|\le\frac{cE}{\delta}.
$$

The relative factor $F=I-(I-B^2)^m$ is positive with spectrum in $[1-E,1]$, even when $B$ is indefinite. For positive $F$ whose extreme eigenvalues are $u,v$, the angle between $|\psi\rangle$ and normalized $F|\psi\rangle$ gives pure-state trace distance at most $(v-u)/(v+u)$. Therefore the normalized inverse approximation obeys

$$
D_{\rm tr}\le\frac{E}{2-E},\qquad
|\Delta\langle P\rangle|\le\frac{2E}{2-E}
$$

for a Pauli observable $P$. These are exact target-polynomial bounds, independent of the particular prepared input.

```{embed} #reciprocal-qsvt-polynomial
```

To request conditional Pauli bias at most $\epsilon_A$, it suffices that $E\le2\epsilon_A/(2+\epsilon_A)$. An exact rational scan chooses the smallest supported $K$ satisfying this inequality. The default $\delta=.5$, $\epsilon_A=.01$ selects $K=16$, $d=33$, $c=1/5$. Its relative tail is about .00751695, scaled operator-error bound .00300678, trace-distance bound .00377265 and Pauli-bias bound .00754531.

This transparent geometric construction has $d=O(\delta^{-2}\log(1/\epsilon_A))$. More advanced reciprocal approximations improve the condition dependence {cite:p}`linsys_qsvt2019,linsys_cks2017`. This implementation does not claim their optimal query complexity.

## Actual access, phases and a small circuit

PREPARE uses a binary tree of controlled RY rotations to load label amplitudes $\sqrt{|a_j|/\alpha}$. SELECT applies the signed Pauli word controlled on its actual label; inverse PREPARE removes the loading. In the exact access model with real coefficients, this complete access is a Hermitian involution whose all-zero-label block is $B$. Numerical probability/sign-phase/angle construction is accounted for separately from that model.

Our scalar convention is

$$
\mathcal B(x)=\begin{pmatrix}x&\sqrt{1-x^2}\\\sqrt{1-x^2}&-x\end{pmatrix},\quad
D(\phi)=\operatorname{diag}(e^{i\phi},e^{-i\phi}),\quad
V=D(\phi_0)\mathcal B\cdots\mathcal B D(\phi_d).
$$

A real-part flag combines the positive and negative phase branches. Its projector operation is $\exp[i\phi(2\Pi_0-I)Z_f]$, retaining the branch phases. All-zero Pauli labels **and** flag select the realized real polynomial. Data values are unrestricted by the acceptance test.

The default uses 33 complete PREPARE/SELECT/unprepare calls and 34 projector phases, with both flag Hadamards. The notebook also expands a smaller one-data-qubit example $A=3I-X$, $K=1$, so every access operation can be inspected. That coarse example illustrates circuit construction; its lower degree has a correspondingly looser reciprocal bound.

```{embed} #reciprocal-qsvt-circuit
```

The [phase record](../../assets/generated/linear-systems/reciprocal-qsvt/phases.json) stores the default phases and exact target Chebyshev coefficients. An original analytical prefix/suffix Jacobian supplies a deterministic SciPy least-squares fit with one initialization and at most 500 function evaluations. The initial Hermitian-signal phases are mapped from the standard signal convention; an initialization near zero would lose Jacobian rank because $\mathcal B^2=I$.

Floating coefficient reconstruction and a 10,001-point grid test provide independent numerical evidence of phase agreement. A separate **rational ideal-signal certificate** treats every stored binary phase as an exact real number, encloses its complex exponential with a Taylor bound, and propagates the resulting polynomial coefficients exactly. The Chebyshev coefficient one-norm bounds the response error uniformly on $[-1,1]$, with a telescoping bound for the enclosed phase factors. The frozen default phases have certified uniform error $\eta\le5.476\times10^{-15}$.

This strengthens the mathematical phase guarantee without certifying a physical circuit. For exact matrix access, write $r=\eta/c$ and require $1-E-r>0$. The realized relative inverse factor has spectrum in $[1-E-r,1+r]$, so

$$
D_{\rm tr}\le\frac{E+2r}{2-E},\qquad
|\Delta\langle P\rangle|\le\frac{2(E+2r)}{2-E}.
$$

The default corrected Pauli bound is at most .007545306. Probability conversion, PREPARE angle construction, coherent compiler/synthesis errors and physical faults remain separate. The phase record includes the exact binary phase hex strings, rational certificate parameters, rounded upper bounds and scope needed to reproduce this distinction.

## Success and size scaling

The exact target polynomial has input-independent acceptance lower bound

$$
p_{\min}=c^2\min\left(\frac{1-E}{\delta},1\right)^2.
$$

For $g(x)=[1-(1-x^2)^m]/x$ on the positive interval, the derivative changes from positive to negative at most once, so the minimum occurs at $\delta$ or 1. Oddness gives the same magnitude on the negative interval. Thus the default bound is $1/25$, while the checked default prepared input has acceptance about .08225369. This distinction matters: a favorable small-input probability is not a uniform guarantee for every input.

With $M$ nonzero combined input terms, a source circuit uses $n+\lceil\log_2M\rceil+1$ logical wires: data, labels and a real-part flag. A one-term access needs no label register. At degree $d$, it pays $d$ full access calls and $d+1$ projector phases. Each SELECT contains the controlled nonidentity letters of the actual Pauli words; PREPARE, sign handling and uncomputation are included. Before any compiler cancellation, its complete cost is the sum of input preparation, $d$ times PREPARE+SELECT+unprepare, $d+1$ projector operations, and the final readout. Primitive gate costs depend on the chosen controlled-gate decomposition, so the CSVs report actual lowered circuits rather than treating an access call as free.

Once the relative tail is small at a fixed promised gap, $p_{\min}=\Theta(1/m)$ while $d=\Theta(m)$. Worst-input expected access calls per accepted state therefore scale as $O(d/p_{\min})=O(m^2)$ for this geometric method. For $N$ accepted samples and empty-slot allowance $\gamma$, finite-cap access work adds the factor $N\log(N/\gamma)$. This includes the decline in acceptance as accuracy increases; degree alone understates the workload.

The scalable active-spin family is

$$
A_n=3R\,I-\sum_{i=0}^{n-1}Z_i-\sum_{i=0}^{n-2}X_iX_{i+1},\quad
R=2n-1,\quad\alpha=4R.
$$

Its coefficient norm promises $\operatorname{spec}(A_n/\alpha)\subseteq[1/2,1]$. Every spin participates; local couplings are fixed while the identity coefficient and normalization grow with $n$. There are $M=2n$ explicit terms. At fixed gap and requested accuracy, $d$ is fixed, logical width grows as $n+O(\log n)$, and each access uses $O(n)$ controlled rotations/Pauli operations with at most $O(\log n)$ label controls, plus their control-setting gates. Lowering those controls determines the primitive-time scaling recorded in the width CSV. Total capped work multiplies the complete per-attempt circuit cost by the reported number of attempts. This is a structured circuit-scaling example, not an arbitrary matrix loader or a claim of quantum advantage.

The [width CSV](../../assets/generated/linear-systems/reciprocal-qsvt/width-scaling.csv) records generated circuits for this family. The [degree CSV](../../assets/generated/linear-systems/reciprocal-qsvt/degree-convergence.csv) changes accuracy at its fixed $n=2$ matrix, which differs from the default demonstration. These two CSVs state whether each circuit was simulated. The [condition CSV](../../assets/generated/linear-systems/reciprocal-qsvt/condition-scaling.csv) supplies exact target-degree choices from the promised gap; it does not generate or simulate those matrices. A tighter requested gap/accuracy requiring degree above 63 is marked unavailable rather than silently weakening the request.

![Reciprocal approximation and generated cost scaling](../../assets/generated/linear-systems/reciprocal-qsvt/polynomial-and-scaling.svg)

Circuit generation is bounded to 24 data wires, 48 input terms and odd degree 63. Reference checks allow five data wires and at most 12 total statevector wires. These limits are separate: generating a larger circuit does not validate its behavior by simulation. The bounded physical sweep below covers seven size/conditioning cases; the larger logical width grid remains circuit-only.

## End-to-end resource analysis

```{embed} #reciprocal-qsvt-ledger
```

The [logical ledger](../../assets/generated/linear-systems/reciprocal-qsvt/logical-resources.json) includes the actual input gates, all access and projector operations, final X/Y/Z basis changes when selected, and measurements of every data/label/flag wire. The default five-wire Z trial lowers to 1,391 CX, 1,756 continuous rotations and five measurements, with dependency depth 2,685. A Z trial needs no extra basis rotation; X uses H and Y uses S-dagger then H. Failed attempts pay the entire circuit. Each fresh retry assumes new clean logical zeros and repeats the specified input preparation.

For one conditional Pauli mean, the proposed total absolute-error budget is .05: .01 polynomial bias, .01 synthesis bias and .03 statistical error. Statistical failure is .004, empty-slot failure .001 and physical-fault allowance .005. Hoeffding sampling requires 13,811 accepted slots. If each slot is allowed at most $R$ complete trials and success is at least $p$, the empty-slot union bound is $N(1-p)^R$.

Under the exact target-polynomial lower bound $p=1/25$, $R=403$ gives 5,565,833 capped complete attempts. The uncapped expected-attempt upper bound is $N/p=345,275$; with the finite cap it is $N[1-(1-p)^R]/p$. The ideal stored-phase certificate lowers acceptance to at least $(\sqrt{p_{\min}}-\eta)^2$, about .0399999999999978 for the default. Rounding this bound down to a modest exact dyadic value leaves the same retry cap. The cost report states which lower bound it uses. Access/synthesis perturbations still require their own correction before these retry counts become guarantees for those perturbed circuits. The repeated measurement workload is an **unexecuted ideal proposal**.

The [surface-code report](../../assets/generated/linear-systems/reciprocal-qsvt/surface-code.json) supplies seven conditional Pareto points under the project's frozen illustrative physical model: $10^{-4}$ physical error, 100 ns gates and 500 ns measurements. The .005 physical-fault allowance spans all capped attempts; the $10^{-6}$ per-attempt QDK synthesis-model scalar is separate from the .01 output-bias allocation. The frontier ranges from about 284,000 peak physical qubits and 14.7 serial days to about 499,000 qubits and 7.64 serial days for the capped workload. These values include magic-state factories and describe a conditional cost projection. Other parameter requests can remain explicitly unavailable; their budget is not relaxed to obtain a result.

```{embed} #reciprocal-qsvt-physical
```

**No complete physical-output confidence certificate is provided.** A polynomial or stored-phase certificate does not certify floating PREPARE angles or an actual Clifford+T synthesis sequence. QDK's mean-fit synthesis scalar is a cost model, rather than an operator-norm certificate for the postselected observable. Clean initialization/reset timing and faults, classical coefficient preprocessing, compilation, reference solve and readout/postprocessing timing are excluded from physical runtime and disclosed in the ledger. Dependency depth alone is not a hardware schedule.

Lowering preserves coherent action on arbitrary prepared data by disabling the assumption that all input qubits are initially zero. The pinned QASM exporter may omit the recorded overall compiled global phase. Downloads are consequently scoped to complete measured trials; they must not be used as phase-preserving controlled inverse oracles without retaining that phase.

### Conditional physical size and conditioning sweep

The [predeclared sweep](../../assets/generated/linear-systems/reciprocal-qsvt-physical/predeclared-plan.json) supplies actual conditional physical estimates for four active-chain sizes at promised gap .5, and four gaps at fixed size $n=2$. One case is shared, giving seven supported cases. For exact $\delta=p/q$, its matrix family is

$$
A_{n,\delta}=R(q+p)I-(q-p)\left(\sum_{i=0}^{n-1}Z_i+\sum_{i=0}^{n-2}X_iX_{i+1}\right),
\qquad R=2n-1,\qquad\alpha=2qR.
$$

The coefficient norm proves $\operatorname{spec}(A/\alpha)\subseteq[\delta,1]$. Every spin is active. At $\delta=.5$ this is the earlier active-spin chain; these benchmark matrices differ from the separate four-eigenvalue default demonstration. Input preparation follows the same gate rule at each size.

Every case keeps normalized-inverse $Z(q_0)$ error target .05, algorithm/synthesis/sampling allowances .01/.01/.03, and statistical/empty-slot/physical failure allocations .004/.001/.005. The physical profile stays at $10^{-4}$ error, 100 ns gates and 500 ns measurement. These budgets apply to each benchmark case; there is no joint output-confidence certificate for the whole grid.

Degree is the smallest supported value whose exact rational **phase-adjusted uniform** Pauli-bias bound fits .01. Retry counts use the phase-adjusted interval success lower bound, rounded downward. Observed small-input success is recorded only for verification. The current method's frozen phases are retained at degree 33; other degrees use its existing deterministic symmetric fitter. Changing the degree also changes the actual projector angles. All 13,811 accepted slots, capped retries, fresh input preparation, access/unprepare, projector operations and final data/ancilla measurements are included in each complete workload. The repeated measurement workloads remain unexecuted conditional proposals.

The [summary](../../assets/generated/linear-systems/reciprocal-qsvt-physical/summary.csv) and [50 Pareto points](../../assets/generated/linear-systems/reciprocal-qsvt-physical/pareto.csv) give the resulting costs. For the size grid, the conservative cap is 5,565,833 complete attempts at degree 33:

| Active data qubits | Logical wires | Primitive operations per attempt | Smallest peak physical qubits | Fastest capped serial days |
| --- | --- | --- | --- | --- |
| 1 | 3 | 245 | 112,428 | 1.42 |
| 2 | 5 | 3,489 | 291,522 | 5.12 |
| 3 | 7 | 13,718 | 252,927 | 87.5 |
| 4 | 8 | 18,902 | 255,425 | 147 |

The smallest footprint and fastest runtime are separate extrema of each frontier, generally attained at different factory/code configurations. They are not one simultaneous operating point. The minimum footprint need not rise monotonically across this small grid because the selected frontier configuration can change. These are conditional model measurements on the stated matrices, rather than an asymptotic hardware scaling law.

At $n=2$, gaps .75/.6/.5/.4 select degrees 11/21/33/53. Their complete-attempt caps are 1,933,540/3,521,805/5,565,833/8,065,624; fastest modeled capped serial times are about .520/2.08/5.12/12.3 days. Gaps 1/3 and 1/4 cannot meet the fixed algorithm allocation at degree at most 63 and remain explicitly unsupported.

![Conditional physical size and conditioning grid](../../assets/generated/linear-systems/reciprocal-qsvt-physical/physical-scaling.svg)

The [manifest](../../assets/generated/linear-systems/reciprocal-qsvt-physical/manifest.json) links each supported case's complete measured QASM, logical/validation ledger and native surface-code report. All seven actual circuits were compared with independent Kronecker matrices and normalized classical solves. Independent NumPy replay checks the full measured distribution of the specified prepared input. Native QDK imports exactly those hashed QASM bytes. Numeric angle literals use Qiskit's `disable_constants=True` to avoid fuzzy symbolic-$\pi$ replacement while preserving the source's stored phases.

The physical queries keep the pinned model and budgets and are bounded to eight wires, 25,000 primitive operations and 60 seconds per case. Factory costs, distances and capped serial runtime appear in every report. The CSV's space-time proxy is peak footprint times serial duration, measured in qubit-seconds; it is not an integral of changing qubit occupancy.

The exact ideal-phase interval certificate remains separate from numerical loading/lowering and concrete synthesis. QDK's per-attempt $10^{-6}$ mean-fit cost scalar supplies no certified .01 postselected synthesis-bias guarantee. Fresh-zero reset/initialization timing and faults, classical processing time in physical runtime, and arbitrary matrix/vector loading remain excluded. No co-design improvement, optimal inverse complexity or hardware confidence claim follows from this fixed-contract scaling grid.

```bash
python scripts/generate_reciprocal_physical_sweep.py
```

## Verification and formalization

```{embed} #reciprocal-qsvt-results
```

The default normalized circuit output has $\langle Z\rangle\approx.23210229$, while the independent normalized classical solve gives .23294918. Their bias is about $8.47\times10^{-4}$, and trace distance about .00149385, within the target bounds. The accepted branch agrees with the independently evaluated spectral polynomial to numerical roundoff. These small-instance checks include the actual preparation and Pauli access; they are not a simulation of an abstract inverse oracle.

An indefinite example $A=7\,ZI+2\,IZ+XX$ has normalized eigenvalues approximately $\pm.50990,\pm.90554$ and is supported by the same signed-gap promise. The notebook checks it separately. Tests and independent reviews cover complex inputs, Y terms, different degrees, analytic phase derivatives, measured basis changes, gap-promise violations, coherent lowering and bounded workload guards. Dense amplitude/unitary loading is rejected by the preparation API.

The [Neumann LCU method](neumann-lcu.md) offers a distinct positive-spectrum construction and actual known-input amplitude amplification. This QSVT method uses repeated access to the original Pauli matrix instead of classically expanding the reciprocal into new Pauli coefficients. General non-Hermitian block embeddings, better condition-dependent polynomials and certified physical synthesis are further variants. No Lean theorem or integrated formal proof currently covers this entire circuit.

The [notebook](../../notebooks/linear_systems_qsvt_reciprocal.ipynb) reuses the module, displays circuits and compares the reference output. Optional bounded controls vary degree on a two-qubit instance. Saved outputs support a static book without a kernel backend; live Python needs a kernel. The [public code package](../../assets/generated/openquantum-code.zip) includes the shared access/preparation dependencies and tests.

```bash
python scripts/generate_linear_systems_qsvt_reciprocal.py
python scripts/generate_linear_systems_qsvt_reciprocal.py --qubits 12 \
  --no-sim --no-scaling --no-estimate --output-dir artifacts/reciprocal-12
```

The explanations, reciprocal construction, circuits and fitter are original MIT work. We use [Qiskit 2.6.2](https://github.com/Qiskit/qiskit) (Apache-2.0), [NumPy 2.5.4](https://github.com/numpy/numpy) and [SciPy 1.18.1](https://github.com/scipy/scipy) (BSD-3-Clause), [Matplotlib 3.11.2](https://github.com/matplotlib/matplotlib) (its PSF-based license), and [QDK 1.34.0](https://github.com/microsoft/qdk) for the separate resource projection. The [locked environment](../../requirements-lock.txt) and provenance record exact versions, module/QASM content hashes and generation device. No upstream reciprocal phase library is used.
