---
title: "Structured word-Taylor dynamics"
description: "Actual degree and term-register access, normalization padding, OAA and complete-trial evidence without exponential word enumeration."
---

Taylor simulation approximates the exponential by a polynomial. Expanding that polynomial into every ordered Hamiltonian word produces a list whose length grows as $L^K$ for $L$ terms and degree $K$. A quantum word register supplies those choices coherently: prepare the degree once, prepare one independent term register for each possible slot, and activate only the slots selected by the degree. This page implements that construction rather than constructing the full word list {cite:p}`sim_taylor_lcu_2015`.

:::{note} Implementation status
Structured PREPARE/SELECT, signed Pauli words, all uncomputation, complete measured/reset trials and integer-round oblivious amplification are implemented. A separate normalization-2 zero-LCU padding circuit is implemented and checked. The [accuracy-preserving workload study](#accuracy-preserving-complete-workloads) supplies 16 ideal-access contracts, 15 materialized circuits and six tiny numerical checks. A separate [one-system baseline](#conditional-one-system-fault-tolerant-baseline) supplies actual Clifford+T sequences and conditional physical projections. Wider-workload physical estimates, certified floating-point and synthesis errors, executed finite-shot sampling and formal verification remain unavailable. Nearly unit acceptance in a small instance does not establish deterministic evolution.
:::

## A fixed small dynamics task

Use two spins with the noncommuting Hamiltonian

$$H=0.6Z_0Z_1+0.35(X_0+X_1),\qquad t=1,\quad\hbar=1.$$

The coefficients are measured in a specified energy unit $E_0$ and time in $E_0^{-1}$. The initial state is $|00\rangle$, and the observable is $Z_0$. Qiskit labels are `ZZ`, `IX`, `XI`, with the rightmost character acting on wire 0. The exact small reference expectation is approximately **0.791510480584**. This example has two spins; the main [Hamiltonian-dynamics example](../hamiltonian-simulation.md) has three. Their different instances do not define an equal-task cost comparison.

For segment length $\delta=t/r$, let

$$T_K(\delta)=\sum_{k=0}^K\frac{(-i\delta H)^k}{k!},\quad
\alpha=\sum_{j=0}^{L-1}|c_j|,\quad
s=\sum_{k=0}^K\frac{(\alpha|\delta|)^k}{k!}.$$

The native access circuit has accepted block $T_K/s$. Accepted states are normalized before comparison with $e^{-itH}|00\rangle$. The unnormalized polynomial operator, conditional state-vector error, infidelity, observable bias and acceptance are different quantities and are recorded separately. Matrix exponentiation supplies a bounded classical answer; it is not a circuit operation {cite:p}`sim_expm_2009,sim_scipy_1181`.

## Degree preparation and independent word slots

Prepare the degree with probabilities

$$d_k=\frac{(\alpha|\delta|)^k}{k!s},\qquad 0\le k\le K.$$

Prepare every one of the $K$ term registers with probabilities $p_j=|c_j|/\alpha$. The original binary-tree circuit uses controlled $R_Y$ rotations and retains the actual access gates. It loads only $K+1$ degree weights and $L$ term weights. Reusing those term weights across slots does not require loading an exponentially long word table.

SELECT applies the first $k$ signed Pauli slots when the degree register holds $k$. For each slot, it enumerates a valid degree value and a term label, converts their zero bits with X gates, then applies controlled Pauli gates and the coefficient sign. A separate degree-conditioned phase supplies $(-i\,\mathrm{sign}\,\delta)^k$. At zero time, unreachable degree branches still have unit-modulus phases so SELECT remains a unitary.

The displayed boxes below have actual circuit definitions. Their complete gate bodies are lowered in the downloadable primitive exports.

```{embed} #simulation-word-taylor-circuit
```

Unprepare the degree and **every** term register. Inactive slots were prepared but never selected; their projected probabilities sum to one. Thus the zero-ancilla block is

$$\sum_{k=0}^K d_k(-i\,\mathrm{sign}\,\delta)^k
\left(\sum_j p_j\,\mathrm{sign}(c_j)P_j\right)^k
=\frac{T_K(\delta)}{s}.$$

Chronological slot 0 then slot 1 multiplies an individual two-slot matrix word as $P_{j_1}P_{j_0}$. The independent, identical slot distributions make the complete sum exactly $H^k$; exchanging that individual order must still preserve its Pauli phase. A test selects X followed by Y and verifies $(-i)^2YX=iZ$ directly. Other tests cover negative coefficients, Y terms, padded labels, zero degree, a one-term register, negative/zero time, duplicate labels and complex input amplitudes.

The native register width is

$$n+\lceil\log_2(K+1)\rceil+K\lceil\log_2L\rceil.$$

For $n=2$, $L=3$ and $K=3$, this is ten wires: two data, two degree and three two-wire term registers. All eight ancillary wires must be zero for acceptance.

## OAA and an actual normalization-2 variant

Let $A$ denote the full native access and $\Pi$ project all degree/word ancillas onto zero. The existing amplification module implements $R=I-2\Pi$ with X–MCZ–X on those ancillas and actual forward/inverse accesses. One round has accepted block

$$3B-4BB^\dagger B,\qquad B=T_K/s,$$

and later integer rounds are checked against the general matrix-polynomial recurrence {cite:p}`sim_oaa_2017,sim_taylor_lcu_2015`. The reflection acts as identity on unknown data; it uses no input-state reflection. The usual sine success formula assumes an exactly scaled unitary and is not used as the implemented polynomial's success result.

At four short segments, native $s\approx1.38353$. One round overshoots: whole-attempt acceptance falls from about 0.07445 to 0.03519, and conditional-state error increases. A useful alternative pads the normalization to two when **$s\le2$**. The one-segment default has $s>2$ and this variant rejects it.

The padding adds a signal flag with probability $s/2$ in zero and a sign qubit in $|+\rangle$. Signal zero controls every actual degree phase and word-Pauli predicate. Signal one applies a CZ to the sign qubit, giving $\langle+|Z|+\rangle=0$. Unprepare all word registers and both flags. The resulting accepted block is

$$\frac{s}{2}\frac{T_K}{s}+\left(1-\frac{s}{2}\right)0=\frac{T_K}{2}.$$

This is an implemented zero-LCU branch, not symbolic normalization. It adds two wires and keeps SELECT polynomial. One OAA round around **all ten ancillas** gives

$$B_2=\frac{3T_K-T_KT_K^\dagger T_K}{2}.$$

The finite Taylor polynomial is not exactly unitary; the circuit still has state-dependent acceptance and distortion. At four segments, the actual accepted map has operator error about $8.62\times10^{-5}$ against $e^{-iH}$, conditional-state error about $6.14\times10^{-5}$ and observable bias about $3.67\times10^{-5}$. Acceptance on $|00\rangle$ is approximately **0.9999999173**. It is a numerical small-instance result with nonzero rejection probability.

## Executable analysis and verification

The default sweeps $r=1,2,4$, $K=3$, native rounds 0/1, and the normalization-2 round-1 variant where valid. All accepted columns of each actual segment circuit are obtained by statevector evolution of the system basis with zero ancillary input. This avoids forming the unused columns of a $2^{12}\times2^{12}$ full unitary. Independent Kronecker Pauli matrices and a matrix Taylor recurrence supply the classical polynomial reference.

```{embed} #simulation-word-taylor-results
```

Actual native block discrepancies are below $6\times10^{-14}$, and padded/OAA block discrepancies are below $10^{-12}$ in the demonstrated checks. The repeated conditional state and total acceptance are independently checked against powers of the appropriate accepted matrix. Per-segment probabilities depend on the preceding conditional state; the implementation multiplies those measured conditional probabilities rather than assuming identical segment trials.

```{embed} #simulation-word-taylor-convergence
```

Primitive compilation sets `qubits_initially_zero=False` to preserve the whole coherent source on arbitrary register input. This matters when an access is inverted or coherently reused. Tests retain global phases and compare source/primitive matrices on tiny circuits and full-register statevectors on the ten/twelve-wire examples. Complete measured/reset instruments are also checked directly, including the acceptance bits and final unconditioned data readout. These are executable checks; no formal proof is integrated.

The pinned Qiskit exporter omits the circuit's overall phase. The ledgers record each omitted phase. The measured-trial QASM preserves these supplied trial probabilities; coherent controlled reuse needs the source circuit or explicit restoration of that phase {cite:p}`sim_qiskit_262`.

## Source complexity, bounded scaling and accuracy

The generator never constructs a list of $L^K$ words and never performs classical Pauli aggregation. Degree preparation has $O(K)$ tree nodes; $K$ term preparations have $O(KL)$ nodes. SELECT has exactly $LK(K+1)/2$ degree-slot-term predicates. Each controls the degree and **one** term register, using $O(\log K+\log L)$ pattern bits. For general n-wire Pauli strings, the source has $O(K^2Ln)$ controlled Pauli calls, plus pattern/phase operations and control lowering. The primitive circuit, depth and unsynthesized rotations are counted separately; source-call complexity is not a physical runtime estimate.

```{embed} #simulation-word-taylor-scaling
```

The generated scaling CSV holds $K=3$, $r=4$, $J=0.6$, $h=0.35$ and $t=1$ fixed while varying the open-chain width from one to eight spins. Logical width grows from 3 to 22; complete primitive measured-attempt size grows from 416 to 57,891 operations. Those larger circuits are generated and counted without statevector execution. As n grows, $\alpha$ grows, so fixed degree/steps do not preserve accuracy.

An analytic real-arithmetic bound for one unnormalized Taylor segment is

$$\epsilon_T\le e^x-\sum_{k=0}^K x^k/k!,\qquad x=\alpha|\delta|.$$

Since exact evolution has norm one, a corresponding full unnormalized bound is $r\epsilon_T(1+\epsilon_T)^{r-1}$. The original fixed-degree CSV evaluates these expressions numerically; it is not an interval certificate or a conditional-state error guarantee. The following study instead chooses $r$ and $K$ from exact rational uniform bounds before compiling. The earlier $K=3,r=4$ data remain construction evidence with a different accuracy contract.

## Accuracy-preserving complete workloads

The [workload manifest](../../assets/generated/word-taylor-workloads/manifest.json), [summary CSV](../../assets/generated/word-taylor-workloads/summary.csv) and [coverage table](../../assets/generated/word-taylor-workloads/summary.md) record a prescribed **16-contract grid: 15 complete circuits were materialized and six tiny cases were numerically checked**. Local couplings stay fixed at $J=3/5$, $h=7/20$, with a newly prepared zero input and output $\langle Z_0\rangle$. Both native unamplified access and normalization-two padding with one OAA round per segment are included. The size slice uses $n=1,2,3,4$, $t=1$, accuracy .05; additional two-spin slices use times .5/2 and accuracies .025/.1. Circuit-only rows report actual generated gates and ideal bounds, not simulated performance.

:::{warning} Ideal bounds, numerical circuits and hardware are separate
The exact arithmetic below describes **ideal access for the recorded binary64 coefficients and segment duration**, with a separate correction to the stated rational Hamiltonian/time. It does not certify floating normalization, probability/angle construction, coherent compiler arithmetic or rotation synthesis. Six tiny source/primitive and measured/reset checks provide a separate numerical evidence layer. Physical faults, reset/readout/gate time, routing and factories are unmodeled for this workload grid; the [one-system physical baseline](#conditional-one-system-fault-tolerant-baseline) has a separate scope and guarded retry budget. No hardware-confidence or formal-proof claim follows from the uniform ideal bounds.
:::

### Exact uniform approximation and acceptance bounds

Let $c_j,t$ be the stated rational coefficients and time, $c_j^{(b)}$ their recorded binary64 values interpreted exactly, and $\delta_b$ the binary64 segment duration. The helper chooses the **smallest $r\le32$** for which

$$x=\sum_j|c_j^{(b)}|\,|\delta_b|\le\tfrac12,$$

then the smallest degree $K\le8$ meeting the algorithm-bias allowance. This is a bounded selection rule, not a global circuit-cost optimization. For $H_b=\sum_jc_j^{(b)}P_j$, the exact exposure correction

$$\eta_{\rm model}=\sum_j|r\delta_b c_j^{(b)}-t c_j|$$

bounds $\|e^{-ir\delta_bH_b}-e^{-itH}\|$ by the Hermitian unitary Duhamel bound. It corrects the coefficient/time model only; it does not bound probability, angle or lowering arithmetic.

Define $T=\sum_{k=0}^K(-i\delta_bH_b)^k/k!$ and $s_K=\sum_{k=0}^K x^k/k!$. The geometric majorant

$$\tau=\frac{x^{K+1}/(K+1)!}{1-x/(K+2)},\qquad s_K\le2$$

bounds the Taylor remainder uniformly over $|\lambda|\le\sum_j|c_j^{(b)}|$, without subtracting approximate exponentials or inferring accuracy from one input. The next omitted term ratio is at most $x/(K+2)$.

For **native access**, let $D_0=(1+\tau)^r-1$. The comparison bound is

$$\|T^r-e^{-itH}\|\le D_{\rm native}=D_0+\eta_{\rm model},\qquad
p_{\min}\ge\left(\frac{1-D_0}{s_K^r}\right)^2\quad(D_0<1).$$

The accepted map is **$T^r/s_K^r$**. Its attenuation is included in acceptance and retries; the reported operator comparison concerns the unscaled $T^r$, not a claim that the attenuated map is close to $e^{-itH}$.

The **normalization-two** label refers to the pre-amplification source block $T/2$. After one OAA round, the accepted segment is

$$B=\frac{3T-TT^\dagger T}{2}.$$

Writing $T=U_\delta(I+F)$ for $U_\delta=e^{-i\delta_bH_b}$, $\|F\|\le\tau$, the linear perturbation is $(F-F^\dagger)/2$. Three quadratic terms and one cubic term give

$$\|B-U_\delta\|\le E=\tau\left(1+\tfrac32\tau+\tfrac12\tau^2\right).$$

Since the ideal accepted block is a contraction, $\|B\|\le1$, composition gives

$$\|B^r-e^{-itH}\|\le D_{\rm padded}=rE+\eta_{\rm model},\qquad
p_{\min}\ge(1-E)^{2r}\quad(E<1).$$

The nonlinear $B$ does not retain a fixed normalization-two interpretation and still has rejection and polynomial bias. For either mode, $D<1$ gives the conservative conditional unit-norm-Pauli bias bound $4D/(1-D)$; the chosen $K$ is the smallest candidate satisfying $4D/(1-D)\le\epsilon/5$. Unsafe candidates receive no positive acceptance bound by squaring a negative amplitude. The downloads retain exact fractions, using lossless hexadecimal integer strings for very large records; decimal values are presentation only.

### Accepted samples and conservative retry caps

The accuracy split is $\epsilon/5$ algorithm bias, $\epsilon/5$ reserved synthesis bias, and $3\epsilon/5$ sampling error. Sampling failure .004 and acceptance-cap failure .001 give an **ideal-access** bad-output/cap-exhaustion bound .005. Under that model, output error is at most $4\epsilon/5$; synthesis is unimplemented, and the separately recorded .005 physical allowance is unmodeled. These allocations do not constitute a combined hardware 99% guarantee.

For $\epsilon=.05$, Hoeffding requires **$M=13,811$ accepted independent samples**. The helper upper-bounds logarithms with 64 exact rational atanh-series terms, power-of-two reduction and a geometric remainder. Acceptance is rounded downward to a 64-bit dyadic $p$. Using eight rational terms for the lower logarithm, the conservative slot cap is

$$R=\left\lceil\frac{\log_{\uparrow}(M/.001)}{\log_{\downarrow}(1/(1-p))}\right\rceil,$$

with $R=1$ when $p=1$. Thus $M(1-p_{\rm actual})^R\le.001$ whenever the ideal actual acceptance is at least the bound. Every complete failed schedule is charged in **$MR$**. The uncapped $M/p$ quantity is an upper bound on an expectation, not measured work or a stopped-at-cap cost.

### Size, time and accuracy scaling

At fixed local couplings, $\alpha$ grows with $n$. The .05-accuracy size slice selects $r=1,3,5,7$ and $K=3,4,4,4$ for $n=1,2,3,4$, respectively, in both modes. Source width grows slowly compared with the native retry burden: its $s_K^r$ attenuation makes acceptance increasingly costly. Padding/OAA changes both the full gate body and its acceptance bound.

| Spins | Mode | Logical wires | Primitives per complete attempt | Capped complete attempts | Full-cap primitives |
|---:|---|---:|---:|---:|---:|
| 1 | Native | 3 | 107 | 331,464 | 35,466,648 |
| 1 | Padded OAA1 | 5 | 819 | 41,433 | 33,933,627 |
| 2 | Native | 13 | 9,582 | 2,955,554 | 28,320,118,428 |
| 2 | Padded OAA1 | 15 | 39,297 | 41,433 | 1,628,192,601 |
| 3 | Native | 18 | 35,150 | 20,343,603 | 715,077,645,450 |
| 3 | Padded OAA1 | 20 | 134,110 | 41,433 | 5,556,579,630 |
| 4 | Native | 19 | 70,263 | 136,673,656 | 9,603,101,091,528 |
| 4 | Padded OAA1 | Unavailable | Cap rejected | 41,433 ideal cap | Uncomputed |

**The four-spin padded/OAA contract was explicitly rejected at the unchanged 250,000-primitive export cap.** Its [exact contract and rejection record](../../assets/generated/word-taylor-workloads/size-n4-normalization2-oaa1/record.json) remain in the grid. No QASM, materialized count or simulation result is supplied for that row; accuracy and bounds were not relaxed to force it through.

![Accuracy-preserving logical word-Taylor size workloads](../../assets/generated/word-taylor-workloads/size-workloads.svg)

The two-spin time slice selects $(r,K)=(2,3)$ at $t=.5$ and $(6,4)$ at $t=2$. Native capped attempts increase from 718,172 to 41,073,914; padded/OAA uses 41,433 in both cases while its per-attempt primitive work increases from 12,414 to 78,588. At $n=2,t=1$, the accuracy slice shows both degree and sampling/retry costs:

| Requested accuracy | Degree $K$ | Accepted samples | Native capped attempts | Padded/OAA capped attempts |
|---:|---:|---:|---:|---:|
| .025 | 4 | 55,241 | 12,815,912 | 165,723 |
| .05 | 4 | 13,811 | 2,955,554 | 41,433 |
| .1 | 3 | 3,453 | 676,788 | 13,812 |

These are complete logical workloads under one accuracy contract, not physical time or hardware savings. Counts include known-input resets, every degree/term PREPARE and inverse, signed SELECT, source inverses, all-ancilla reflections/padding, segment measurements/resets and final data Z readout. Dependency layers are not routed hardware time. Wider reset/measurement circuits are outside the present five-wire physical-instrument scope; the one-system physical evidence below remains separate from this wider logical dataset.

Six tiny checks cover both modes at one spin/$t=1$/accuracy .05, two spins/$t=.5$/accuracy .05, and two spins/$t=1$/accuracy .1. They use independent bit-flip/ZZ Hamiltonians, a polynomial recurrence, SciPy matrix exponentials and separate eigendecomposition to check actual source blocks, seeded complete-register lowering and accepted measured/reset chronology. Other materialized rows, including two spins at $t=1$ and accuracy .05, are **circuit-only**. Physical estimation and finite-shot execution are unperformed for this dataset; numerical gates do not certify the uniform ideal-access bound at the hardware level.

[Workload generator](../../assets/generated/word-taylor-workloads/generate_word_taylor_workloads.py) · [Exact-bound helper](../../assets/generated/word-taylor-workloads/word_taylor_workloads.py) · [Tiny native measured trial](../../assets/generated/word-taylor-workloads/size-n1-native/measured-attempt.qasm) · [Its bound/check record](../../assets/generated/word-taylor-workloads/size-n1-native/record.json) · [Three-spin padded/OAA measured trial](../../assets/generated/word-taylor-workloads/size-n3-normalization2-oaa1/measured-attempt.qasm)

## End-to-end resource analysis

### Complete measured trials and error separation

Each attempt resets all data/access wires and prepares the known basis input. After each segment, measure **every** degree, word and selected padding ancilla; reset them before reuse. The explicit schedule runs all segments even after an earlier failure and accepts only all-zero ancillary outcomes. Final Z measurements retain the data bitstring. A retry needs a freshly preparable input; a single unknown specimen is not copied after rejection.

```{embed} #simulation-word-taylor-ledger
```

The four-segment normalization-2 round-1 attempt has 12 logical wires, 8 forward accesses, 4 inverse accesses and 8 all-ancilla reflections. It includes 24 degree PREPARE/inverse calls, 72 term PREPARE/inverse calls, 12 SELECT/inverse calls, padding preparations and zero-LCU CZ gates. Its compiled complete schedule has 9,252 CX, 972 arbitrary rotations, 42 measurements and 42 resets, with dependency depth 22,184. These are actual logical counts before synthesis, scheduling or error correction.

Observable estimation needs many **accepted** trials. The original fixed-$K,r$ demonstration uses a separate example allowance $\epsilon_{\rm stat}=0.03$, $\eta_{\rm stat}=0.005$, giving 13,315 accepted $\pm1$ Z samples by Hoeffding. Its ledger uses numerical finite-H minimum acceptance, separately caps retries for an acceptance-failure allocation of 0.005 across all shots, and charges complete attempts. This is an unexecuted conditional workload proposal. It is separate from the new uniform-bound study's 13,811 samples, .004 sampling failure and .001 acceptance failure; the original .005+.005 allocation cannot also absorb a .005 physical allowance without changing the contract.

Hamiltonian truncation bias is checked independently. Coherent synthesis can change acceptance and the accepted state's normalization, so it has a separate hypothetical norm-error allowance. The proposed bound $4\epsilon/(\sqrt{p_{\min}}-\epsilon)$ applies to observable bias if the required accepted-branch error and acceptance bounds are certified. For this original two-spin demonstration, those certificates, rotation synthesis and physical costing are not implemented. The following one-system slice supplies actual synthesis sequences and a conditional physical model with numerical guards. Classical coefficient handling/readout wall time remains separate. The frozen Grover physical baseline is unchanged.

### Conditional one-system fault-tolerant baseline

The two one-system cases use **$H=0.35X$**, $t=1$, a newly prepared $|0\rangle$ input and one $Z_0$ mean at absolute accuracy .05. This is separate from the earlier two-spin $0.6ZZ+0.35(X_0+X_1)$ demonstration. Native word-Taylor access uses three logical wires; normalization-two padding with one oblivious-amplification round uses five. This bounded slice uses the complete serialized instrument model, with actual Clifford+T sequences and a separately stated surface-code, factory and routing scenario {cite:p}`resource_litinski_2019,resource_ross_selinger_2014,resource_qiskit_262,resource_qdk_134`.

Both cases require 13,811 accepted samples. Allowing for the guarded synthesis perturbation changes the conservative cap to **25 complete attempts per slot for native access and four for amplification: 345,275 and 55,244 attempted circuits**, respectively. The earlier ideal-only caps 24/3 remain in the logical workload study and are not the caps costed here. Every attempt initializes its input, runs all access, inverse and reflection gates, measures all access ancillas, and reads out the system even when acceptance fails. The large sampling/retry ensembles are calculated workloads, not executed experiments.

With $a$ the ideal uniform accepted-amplitude lower bound, the norm reserve is $\eta=.01a/(4+.01)$. The source synthesis request is rounded downward from $\eta-10^{-10}$, retaining a numerical construction reserve for the final routed accepted block. **If those numerical norm estimates hold**, conditional synthesis bias is at most $4\eta/(a-\eta)=.01$ and acceptance is at least $(a-\eta)^2$; this guarded acceptance determines the new retry caps.

The total failure allocation per case and hardware profile is .004 for sampling, .001 for exhausted accepted-slot retries, .0025 for modeled physical faults and .0025 for factory readiness. The synthesis allocation is a separate .01 mean-bias allowance, rather than a stochastic fault or an error multiplied by the sample count. The accuracy split remains .01 algorithm bias, .01 synthesis bias and .03 sampling error.

The table's $p$ column denotes the physical error rate, $10^{-3}$ or $10^{-4}$.

```{include} ../../assets/generated/word-taylor-physical/table.md
```

![Conditional complete-workload physical comparison](../../assets/generated/word-taylor-physical/comparison.svg)

The table retains both the fewest-qubit and shortest-runtime frontier selections from the current generated backend; the figure shows the fewest-qubit selection with its own runtime. **Actual T/Tdag counts per attempt are 1,132 for native access and 35,778 for padded/OAA**, requiring 390,851,300 and 1,976,519,832 magic states across the capped workloads. Fewer amplified attempts therefore do not imply lower physical runtime: the larger synthesized trial and charged factory, transport and controller work enter the complete serialized cost.

The complete source, routed and synthesized gate-readout laws include every failed outcome. Output-bit maps are restored before comparison, and the final native or amplified accepted block is checked directly. The operator comparison concerns the coherent body **after initialization**, not a unitary representation of the reset channel on arbitrary input. The accumulated synthesis error and accepted-block construction guard are numerical estimates. Protected-patch geometry, controller reliability, factory interfaces and physical-error assumptions are conditional scenarios rather than hardware calibration or formal certificates.

Runtime includes initialization, reset, readout, logical routing, magic-state transport/injection, controller decisions and capped factory retries. Offline coefficient construction, compilation/synthesis and final statistical-report wall time remain separate from quantum-model runtime. **Larger-width word-Taylor physical estimates remain pending.** This one-system model does not cost the wider logical grid or establish hardware confidence, quantum advantage or an improvement over the different frozen Grover/PSSPC task.

- Native: [record](../../assets/generated/word-taylor-physical/size-n1-native/report.json), [source QASM](../../assets/generated/word-taylor-physical/size-n1-native/source-instrument.qasm), [routed QASM](../../assets/generated/word-taylor-physical/size-n1-native/routed-instrument.qasm), [Clifford+T QASM](../../assets/generated/word-taylor-physical/size-n1-native/clifford-t-instrument.qasm)
- Amplified: [record](../../assets/generated/word-taylor-physical/size-n1-normalization2-oaa1/report.json), [source QASM](../../assets/generated/word-taylor-physical/size-n1-normalization2-oaa1/source-instrument.qasm), [routed QASM](../../assets/generated/word-taylor-physical/size-n1-normalization2-oaa1/routed-instrument.qasm), [Clifford+T QASM](../../assets/generated/word-taylor-physical/size-n1-normalization2-oaa1/clifford-t-instrument.qasm)

[All profiles and provenance](../../assets/generated/word-taylor-physical/manifest.json) · [Summary CSV](../../assets/generated/word-taylor-physical/summary.csv)

## Reproduce and download

The source generator limits systems to 32 wires, terms to 64, degree to 8, total width to 96, SELECT predicates to 4,096 and source instructions to 100,000. Primitive exports stop at 250,000 operations. OAA reuses its own 24-wire/0–4-round limits. Projected statevector checks stop at twelve total wires and six system wires; full Operator tests use at most eight total wires. These limits are independent of circuit-only materialization.

The accuracy-preserving planner separately caps systems at six spins, segments at 32 and degree at eight; its numerical workload checks stop at twelve total wires and three system spins. It retains the unchanged 250,000-primitive export cap. These planner/check bounds do not extend the original generator's simulation or hardware coverage.

```bash
python scripts/generate_simulation_word_taylor.py
pytest tests/test_simulation_word_taylor.py
python scripts/generate_word_taylor_workloads.py
pytest tests/test_word_taylor_workloads.py
python scripts/generate_simulation_word_taylor.py --qubits 8 \
  --segments 4 --rounds 0 --no-sim --no-padding --no-scaling \
  --output-dir artifacts/word-taylor-spin8
```

The [notebook](../../notebooks/simulation_word_taylor.ipynb) supplies the labeled figures, tables and optional live control. Static execution with `OPENQUANTUM_STATIC_BUILD=1` creates no widget objects. The original demonstration uses its five copied Python files and the [locked environment](../../requirements-lock.txt). The workload generator supplies its own copied dependency set, [lock](../../assets/generated/word-taylor-workloads/requirements-lock.txt), [source hashes](../../assets/generated/word-taylor-workloads/manifest.json) and [MIT license](../../assets/generated/word-taylor-workloads/LICENSE.txt) for standalone reproduction. Text, gate construction and figures are original; Qiskit 2.6.2 and SciPy 1.18.1 are credited with their source versions and licenses in the simulation bibliography.

- [Notebook](../../notebooks/simulation_word_taylor.ipynb), [original source](../../assets/generated/simulation/word-taylor/word_taylor.py), [standalone generator](../../assets/generated/simulation/word-taylor/generate_simulation_word_taylor.py)
- [Results and provenance](../../assets/generated/simulation/word-taylor/results.json), [full logical ledger](../../assets/generated/simulation/word-taylor/logical-resources.json), [convergence CSV](../../assets/generated/simulation/word-taylor/convergence.csv), [fixed-degree scaling CSV](../../assets/generated/simulation/word-taylor/scaling.csv)
- [Native four-segment measured circuit](../../assets/generated/simulation/word-taylor/degree3-segments4-rounds0.qasm), [normalization-2 amplified four-segment measured circuit](../../assets/generated/simulation/word-taylor/degree3-segments4-rounds1-normalization2.qasm)
- Standalone dependencies: [LCU preparation helper](../../assets/generated/simulation/word-taylor/lcu.py), [amplification](../../assets/generated/simulation/word-taylor/amplification.py), [Pauli/reference helpers](../../assets/generated/simulation/word-taylor/hamiltonian_simulation.py), [MIT license](../../assets/generated/simulation/word-taylor/LICENSE.txt)

[Return to Hamiltonian dynamics](../hamiltonian-simulation.md).
