---
title: "Structured degree-one DQI"
description: "Executed clean-workspace DQI checks at n=2/3, generation at n=8, and conditional compiled resource results for the fixed small tasks."
---

## Motivation and status

This family makes the decoder and preparation in [decoded quantum interferometry](dqi.md) concrete as the number of assignment bits grows. The DQI formulation connects a coherently decoded low-weight clause state to an objective-shaped Fourier distribution {cite:p}`frontier_dqi2025`. Here the input family has a simple decoder whose actual gates can be generated without enumerating all syndromes.

:::{note} Implementation status: executed ideal and compiled checks, companion and conditional fixed physical model

On 13 November 2026, one bounded Linux CPU pass completed 39 focused contract tests, new coherent checks at $n=2,3$, generation-only export at $n=8$, and four ordinary notebook cells with locked Python 3.12.14. The pipeline recorded 6.438 seconds before its final seal. This is CPU validation time, not quantum hardware runtime. The wider $n\leq64$ generator remains source-supported; this pass executed its CLI only at $n=8$.

The separately validated nine-wire default retains its arbitrary-dirty-scratch decoder law, **377 unitary gates and 396 materialized operations**, original failed attempt and successful ordinary notebook evidence. It was not rerun. The new clean decoder has a different workspace contract and its own checks; sharing the $n=3$ ideal output law does not transfer evidence between them. Those historical results remain separate from the 14 November 2026 compiled-circuit increment below. That separate CPU increment passed its compiled-state and ledger checks and an independent saved-output audit. Hardware execution, a complete end-to-end hardware estimate and a formal theorem remain unavailable.
:::

## Task, register order and decoder promise

For integer $n\geq2$, set $m=n+1$, polynomial degree $\ell=1$,

$$
B=\begin{bmatrix}I_n\\\mathbf1_n^T\end{bmatrix},
\qquad v=(0,\ldots,0,1).
$$

The first $n$ constraints require individual assignment bits to be zero; the last requires odd parity. The objective $f(x)$ counts satisfied equations $B_i x=v_i$ over $\mathbb F_2$. Input $n=1$ is excluded because its unit row and all-ones row coincide. This is the stated family only; arbitrary matrices, signs, fields and higher-degree filters require other implementations.

Integer bit $j$ is assignment bit $x_j$; displayed assignments read $x_{n-1}\cdots x_0$. Clause bit $i$ belongs to row $i$, including the last clause at $i=n$. The complete circuit uses:

| Register | Wires | Initial condition |
|---|---|---|
| Clause | $0,\ldots,n$ | Zero |
| Assignment/syndrome | $n+1,\ldots,2n$ | Zero |
| Reused workspace | $2n+1,\ldots,3n-1$ | Zero |
| Classical assignment readout | $0,\ldots,n-1$ | Wire $n+1+j$ reads into bit $j$ |

There are $3n$ logical wires and $n$ classical bits. Every complete draw resets all logical wires and reads all assignment wires. Preparation needs at most $n-1$ clean workspace bits; decoding needs at most $n-2$. These scratch bits are reused and uncomputed between predicates.

Let $E_i$ be a clause unit vector and $e_i$ an assignment unit vector. The prepared domain is $D=\{0,E_0,\ldots,E_n\}$. Syndrome bit $j$ is $y_j\oplus y_n$, so

$$
\ker(B^T)=\{0^{n+1},1^{n+1}\},\qquad d_{\min}=n+1>2\ell.
$$

The kernel follows because a zero syndrome forces every $y_j$ to equal $y_n$. Thus vacuum and singleton syndromes are distinct. This distance statement establishes the degree-one promise; it does not implement a higher-degree decoder.

The total decoded clause function is

$$
d(s)=\begin{cases}
E_i,&s=e_i,\quad 0\leq i<n,\\
E_n,&s=1^n,\\
0,&\text{otherwise}.
\end{cases}
$$

It specifies off-promise syndromes as well as $d(0)=0$, and satisfies $d(B^T y)=y$ on $D$. No $2^n$-entry lookup table is required.

## Explicit preparation and reversible decoding

The clause state assigns amplitude $1/\sqrt{n+2}$ to the vacuum and each of the $n+1$ singletons. For target $i=0,\ldots,n$, its source angle is

$$
\theta_i=2\arcsin\!\left(\frac1{\sqrt{n+2-i}}\right).
$$

The first target uses RY directly. Each later target rotates only when all preceding clause bits are zero. X conjugations turn those negative controls into positive controls; a conjunction chain computes their AND into clean scratch, controls the rotation, and reverses the chain. Target $i\geq2$ uses $2(i-1)$ CCX blocks and $2i$ X gates. Target one needs no conjunction scratch.

Each controlled RY is emitted as two RY gates and two CX gates. The source rejects a non-finite angle or a binary angle for which $2(\theta/2)\ne\theta$ before emitting that controlled rotation. This finite representation check does not certify an irrational ideal angle or synthesize it into a fault-tolerant gate set.

Before target $i$, the remaining zero-prefix probability is $(n+2-i)/(n+2)$. Its rotation creates singleton probability $1/(n+2)$; after the final target the remaining vacuum has the same probability. The ideal signs are positive and the conjunction scratch is uncomputed.

Z on clause $n$ supplies the negative sign. The syndrome contains $2n$ CX gates: one from clause $j$ and one from clause $n$ into assignment wire $n+1+j$, for each $j$.

Decoding tests each of the $n$ unit patterns and the all-ones pattern. For a unit pattern, X conjugations negate its other $n-1$ syndrome bits. A positive $n$-control toggle computes an AND chain into $n-2$ clean scratch bits, toggles its clause target, and reverses the chain. It uses $2n-3$ CCX blocks, including the single-CCX case $n=2$. Each CCX is expanded into the exact 15-primitive H/CX/T/T-dagger sequence, with counts $2/6/4/3$; no opaque CCX or decoder instruction remains.

For arbitrary clause target $t$, the clean-workspace contract is

$$
U_d|s,t,0_w\rangle=|s,t\oplus d(s),0_w\rangle.
$$

The primitive sequence is unitary on all inputs and restores its conjunction scratch. For $n\geq3$, dirty scratch can change the target toggle. Starting from scratch $w$, a positive chain computes

$$
w'_0=w_0\oplus c_0c_1,\qquad
w'_j=w_j\oplus w'_{j-1}c_{j+1},\quad 1\leq j\leq n-3,
$$

then toggles by $w'_{n-3}c_{n-1}$ and reverses the chain. Negative controls use their conjugated literals. At $n=2$, there is no decoder scratch chain and the toggle is $c_0c_1$. This explicit extension preserves scratch but does not promise the clean XOR target law on general dirty inputs.

On the prepared domain, decoding clears the clause register. The remaining assignment state before the final Hadamards is

$$
\frac{|0\rangle+\sum_{i=0}^{n-1}|e_i\rangle-|1^n\rangle}{\sqrt{n+2}}.
$$

## Independent ideal law and executed small checks

For assignment $x$ of Hamming weight $k$,

$$
f(x)=n-k+\mathbf1_{k\text{ odd}},\qquad
S(x)=2f(x)-(n+1)=n-2k-(-1)^k.
$$

A direct sum of its $n+2$ Walsh characters gives the signed amplitude

$$
A(x)=\frac{1+\sum_{i=0}^{n-1}(-1)^{x_i}-(-1)^{\operatorname{parity}(x)}}{\sqrt{2^n(n+2)}}
=\frac{2f(x)-n}{\sqrt{2^n(n+2)}}.
$$

Those characters are distinct for $n\geq2$; orthogonality fixes normalization. The independent validator evaluated the first character sum and the individual equations directly, rather than call the production closed objective formula.

| Checked small fixture | Exact signed-amplitude reference in integer order | Ideal optimum probability | Ideal expected objective |
|---|---|---:|---:|
| $n=2$ | $(1,1,1,-1)/2$ | $3/4$ | $3/2$ |
| $n=3$ | $(3,3,3,-1,3,-1,-1,-1)/\sqrt{40}$ | $9/10$ | $14/5$ |

The $n=2$ output law is uniform, despite its negative final amplitude. It covers the empty decoder-chain boundary. The $n=3$ fixture covers a nonempty chain and an objective-biased law. The exact references matched the floating ideal CPU checks; no outcomes were sampled.

The new check retained complex phase, preparation support, decoded clause/workspace cleanup and the final assignment law. It tested the clean XOR decoder on every syndrome and every clause target: 32 clean basis columns at $n=2$ and 128 at $n=3$. Off-domain $n=3$ syndromes $3,5,6$ returned zero with clean workspace. Arbitrary clause targets were checked with clean scratch; no target-action guarantee was validated for initially dirty scratch. The old default's separate 512-input numerical check was not repeated.

| New case | Clean decoder columns | Maximum recorded check error | CPU expected objective | CPU maximum-objective probability |
|---|---:|---:|---:|---:|
| $n=2$ | 32 | `7.993605777301127e-15` | `1.499999999999992` | `0.749999999999996` |
| $n=3$ | 128 | `5.10702591327572e-14` | `2.7999999999999488` | `0.8999999999999835` |

All comparisons used finite absolute tolerance $2\times10^{-12}$ and zero relative tolerance. Decoder-entry errors were `1.4432899320127035e-15` and `5.551115123125783e-15`; the table's maxima include other checks such as the objective expectation. Final clause/workspace-zero probabilities were `0.9999999999999947` and `0.9999999999999817`. These floating elementwise diagnostics retain the negative amplitudes; they are not spectral-norm, interval or hardware-failure certificates.

The [small validation record](../../assets/generated/optimization/dqi-structured/small/validated-small.json) contains both complete outcome tables, state transitions, source/environment identities and actual primitive ledgers. The [n=2 QASM](../../assets/generated/optimization/dqi-structured/small/n2/dqi.qasm) and [n=3 QASM](../../assets/generated/optimization/dqi-structured/small/n3/dqi.qasm) expose every preparation, syndrome, decoder, reset and readout operation. The separate simulator ceilings remain nine wires, 512 statevector amplitudes and operator dimension 512. No larger dense simulation ran.

The original six-wire $n=2$ primitive circuit is available as a [full text drawing](../../assets/generated/optimization/dqi-structured/small/n2/circuit.txt). Its display abbreviates angles; QASM preserves their exported binary values.

```text
           ░   ┌─────────┐  ┌───┐                          ┌───┐┌───┐                                        ┌───┐ »
q_0: ─|0>──░───┤ Ry(π/3) ├──┤ X ├──■────────────────────■──┤ X ├┤ X ├─────────■─────────────────────■────■───┤ T ├─»
           ░ ┌─┴─────────┴─┐└───┘┌─┴─┐┌──────────────┐┌─┴─┐├───┤└───┘         │             ┌───┐   │  ┌─┴─┐┌┴───┴┐»
q_1: ─|0>──░─┤ Ry(0.61548) ├─────┤ X ├┤ Ry(-0.61548) ├┤ X ├┤ X ├──■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├»
           ░ └─┬─────────┬─┘     └───┘└──────────────┘└───┘└───┘  │           │         │   └───┘   │  └───┘└─────┘»
q_2: ─|0>──░───┤ Ry(π/4) ├────────────────────────────────────────┼───────────┼─────────┼───────────┼──────────────»
           ░   └─────────┘                                        │           │         │           │              »
q_3: ─|0>──░──────────────────────────────────────────────────────┼───────────┼─────────┼───────────┼──────────────»
           ░                                                      │           │         │           │              »
q_4: ─|0>──░──────────────────────────────────────────────────────┼───────────┼─────────┼───────────┼──────────────»
           ░      ┌───┐                                         ┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐ ┌───┐ »
q_5: ─|0>──░──────┤ H ├─────────────────────────────────────────┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├─»
           ░      └───┘                                         └───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘ »
c_0: ══════════════════════════════════════════════════════════════════════════════════════════════════════════════»
                                                                                                                   »
c_1: ══════════════════════════════════════════════════════════════════════════════════════════════════════════════»
                                                                                                                   »
«                                                                             ┌───┐      ┌───┐ ░       ░           »
«q_0: ──■──────────────────────────────────────■─────────────────────■────■───┤ T ├───■──┤ X ├─░───────░───■───────»
«     ┌─┴─┐                                    │             ┌───┐   │  ┌─┴─┐┌┴───┴┐┌─┴─┐├───┤ ░       ░   │       »
«q_1: ┤ X ├────────────────────────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├┤ X ├─░───────░───┼────■──»
«     ├───┤┌──────────┐┌───┐       │           │         │   └───┘   │  └───┘└─────┘└───┘└───┘ ░ ┌───┐ ░   │    │  »
«q_2: ┤ X ├┤ Ry(-π/4) ├┤ X ├───────┼───────────┼─────────┼───────────┼─────────────────────────░─┤ Z ├─░───┼────┼──»
«     └─┬─┘└──────────┘└─┬─┘       │           │         │           │                         ░ └───┘ ░ ┌─┴─┐  │  »
«q_3: ──┼────────────────┼─────────┼───────────┼─────────┼───────────┼─────────────────────────░───────░─┤ X ├──┼──»
«       │                │         │           │         │           │                         ░       ░ └───┘┌─┴─┐»
«q_4: ──┼────────────────┼─────────┼───────────┼─────────┼───────────┼─────────────────────────░───────░──────┤ X ├»
«       │                │  ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐ ┌───┐            ░       ░      └───┘»
«q_5: ──■────────────────■──┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├────────────░───────░───────────»
«                           └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘            ░       ░           »
«c_0: ═════════════════════════════════════════════════════════════════════════════════════════════════════════════»
«                                                                                                                  »
«c_1: ═════════════════════════════════════════════════════════════════════════════════════════════════════════════»
«                                                                                                                  »
«                ░ ┌───┐┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐┌───┐┌───┐ ┌───┐                                             »
«q_0: ───────────░─┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├─────────────────────────────────────────────»
«                ░ ├───┤└─┬─┘└─────┘└─┬─┘└───┘└─┬─┘└─────┘└─┬─┘└───┘ └───┘           ┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐»
«q_1: ───────────░─┤ H ├──┼───────────┼─────────┼───────────┼────────────────────────┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├»
«                ░ ├───┤  │           │         │           │                        └─┬─┘└─────┘└─┬─┘└───┘└─┬─┘└─────┘»
«q_2: ──■────■───░─┤ H ├──┼───────────┼─────────┼───────────┼──────────────────────────┼───────────┼─────────┼─────────»
«     ┌─┴─┐  │   ░ └───┘  │           │         │           │        ┌───┐      ┌───┐  │           │         │         »
«q_3: ┤ X ├──┼───░────────┼───────────■─────────┼───────────■────■───┤ T ├───■──┤ X ├──┼───────────■─────────┼─────────»
«     └───┘┌─┴─┐ ░ ┌───┐  │                     │   ┌───┐      ┌─┴─┐┌┴───┴┐┌─┴─┐├───┤  │                     │   ┌───┐ »
«q_4: ─────┤ X ├─░─┤ X ├──■─────────────────────■───┤ T ├──────┤ X ├┤ Tdg ├┤ X ├┤ X ├──■─────────────────────■───┤ T ├─»
«          └───┘ ░ └───┘                            └───┘      └───┘└─────┘└───┘└───┘                            └───┘ »
«q_5: ───────────░─────────────────────────────────────────────────────────────────────────────────────────────────────»
«                ░                                                                                                     »
«c_0: ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════»
«                                                                                                                      »
«c_1: ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════»
«                                                                                                                      »
«                                                                                         ░       ░        ░
«q_0: ────────────────────────────────────────────────────────────────────────────────────░───────░────────░─
«     ┌───┐┌───┐ ┌───┐                                                                    ░       ░        ░
«q_1: ┤ X ├┤ T ├─┤ H ├────────────────────────────────────────────────────────────────────░───────░────────░─
«     └─┬─┘└───┘ └───┘           ┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐┌───┐┌───┐ ┌───┐       ░       ░        ░
«q_2: ──┼────────────────────────┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├───────░───────░────────░─
«       │        ┌───┐      ┌───┐└─┬─┘└─────┘└─┬─┘└───┘└─┬─┘└─────┘└─┬─┘└───┘ ├───┤       ░ ┌───┐ ░ ┌─┐    ░
«q_3: ──■────■───┤ T ├───■──┤ X ├──┼───────────■─────────┼───────────■────■───┤ T ├───■───░─┤ H ├─░─┤M├────░─
«          ┌─┴─┐┌┴───┴┐┌─┴─┐└───┘  │                     │   ┌───┐      ┌─┴─┐┌┴───┴┐┌─┴─┐ ░ ├───┤ ░ └╥┘┌─┐ ░
«q_4: ─────┤ X ├┤ Tdg ├┤ X ├───────■─────────────────────■───┤ T ├──────┤ X ├┤ Tdg ├┤ X ├─░─┤ H ├─░──╫─┤M├─░─
«          └───┘└─────┘└───┘                                 └───┘      └───┘└─────┘└───┘ ░ └───┘ ░  ║ └╥┘ ░
«q_5: ────────────────────────────────────────────────────────────────────────────────────░───────░──╫──╫──░─
«                                                                                         ░       ░  ║  ║  ░
«c_0: ═══════════════════════════════════════════════════════════════════════════════════════════════╩══╬════
«                                                                                                       ║
«c_1: ══════════════════════════════════════════════════════════════════════════════════════════════════╩════
«
```

## Size, gates, workspace and repetitions

The preparation uses $C_p=n(n-1)$ CCX blocks, $n$ controlled RY blocks, one direct RY and $n(n+1)$ X gates. Its analytic primitive count is $P=16n^2-10n+1$. The decoder uses $C_d=(n+1)(2n-3)$ CCX blocks and $2n(n-1)$ X gates, giving $D=32n^2-17n-45$.

The generated $n=2,3,8$ workloads matched these source formulas; broader sizes were not executed. The complete per-draw source ledger is:

| Primitive or event | Analytic source count |
|---|---:|
| H | $6n^2-3n-6$ |
| CX | $18n^2-8n-18$ |
| T | $12n^2-8n-12$ |
| T-dagger | $9n^2-6n-9$ |
| X | $3n^2-n$ |
| RY, unsynthesized | $2n+1$ |
| Z | $1$ |
| Unitary primitives $U$ | $48n^2-24n-43$ |
| Reset events | $3n$ |
| Individual readouts | $n$ |
| Events excluding barriers | $48n^2-20n-43$ |
| Retained source barriers | $7$ |
| Materialized operations $M$ | $48n^2-20n-36$ |

All prefix controls, syndrome CXs, decoding, uncomputation and assignment readout are counted. The explicit T/T-dagger count is $21n^2-14n-21$, excluding synthesis of every RY. A serial primitive schedule has $U$ unitary slots; serialized reset and readout add $4n$ slots. This is a source scheduling bound, not hardware runtime or a parallel circuit-depth measurement.

| Generated example | Wires | Preparation | Decoder | Unitary $U$ | Total $M$ | Executed scope |
|---|---:|---:|---:|---:|---:|---|
| $n=2$ | 6 | 45 | 49 | 101 | 116 | Coherent small checks |
| $n=3$ | 9 | 115 | 192 | 317 | 336 | Coherent small checks |
| $n=8$ | 24 | 945 | 1867 | 2837 | 2876 | Generation only; no simulation |

The $n=3$ clean decoder has four fewer CCX blocks than the old dirty-scratch decoder. Its actual 317/336 counts do not revise the separately executed default's 377/396 counts or establish a physical improvement under the stronger decoder contract. Explicit T/T-dagger occurrences are 35, 126 and 1211 at $n=2,3,8$, with 5, 7 and 17 unsynthesized RY gates, respectively. None is full compiled T demand.

Maximum objective $n$ occurs exactly at Hamming weights zero and one. Hence

$$
p_{\rm opt}(n)=\frac{(n+1)n^2}{2^n(n+2)}.
$$

For independent ideal complete draws and desired miss probability $0<\delta_{\rm draw}<1$, the smallest required draw count is

$$
R=\left\lceil\frac{\log\delta_{\rm draw}}{\log(1-p_{\rm opt})}\right\rceil,
\qquad P(\text{at least one optimum})=1-(1-p_{\rm opt})^R.
$$

The source planner caps the search at 1024 draws and compares exact rational failure probabilities. If the target is infeasible within that cap, it reports the shortfall rather than accepting the requested confidence. Per-draw generation remains a separate task. For a feasible fixed $R$, charge $RU$ unitary primitives, $3nR$ resets, $nR$ readouts and $RM$ source operations; serial peak logical width remains $3n$. With $\delta_{\rm draw}=1/100$, the predicted $n=2,3,8$ draw counts are 4, 2 and 19, respectively. No such draws were sampled in this validation. The $n=8$ generation record charges 53,903 unitary primitives and 54,644 total source operations for that ideal 19-draw plan, without executing trials or validating the larger output law.

The exact miss target accepts a native `Fraction` with numerator and denominator at most 256 bits, or at most 64 ASCII characters in `numerator/denominator` form. Decimal and exponent text are rejected; the target must lie strictly between zero and one.

At fixed confidence, $R=\Theta(2^n/n^2)$, so the repeated ideal source-gate workload is $\Theta(2^n)$. Moreover, writing the all-zero assignment immediately gives an optimum: all $n+1$ equations cannot hold because the first $n$ force even parity. This family is an explicit circuit/decoder scaffold, separate from the Nature paper's hard structured problem and sparse benchmarks {cite:p}`frontier_dqi2025`. It supplies no optimization-advantage claim.

## Actual compilation and new coherent checks

On 14 November 2026, the corrected bounded Python 3.12.14 Darwin CPU pass
completed 83 pure boundary tests, eight native
synthesis calls and two direct QDK resource traces. It synthesized the exact
eight binary RY keys occurring in the accepted $n=2,3$ operation streams,
then checked the two newly compiled zero-input circuits. The earlier ideal
producer, decoder operator and companion notebook were not rerun. Five/three
assignment draws remain modeled; none was sampled.

The compiler requested epsilon $1/80000$ from Qiskit 2.6.2’s
`gridsynth_unitary` interface. Each returned Clifford+T word was reconstructed
independently, retaining its global phase, and compared with the analytic RY
matrix. Its Frobenius discrepancy plus a $10^{-12}$ guard had to fit
$1/20000$. The largest recorded guarded word discrepancy was
`8.424673794495145e-06`. There was one synthesis call per exact key, with no
epsilon retry or alternate backend
{cite:p}`resource_qiskit_262,resource_rsgridsynth_022,resource_ross_selinger_2014`.

Every source CX was routed on a logical line by deterministic neighboring
SWAP chains, each expanded into three CXs. The same swaps were undone after
the interaction, restoring logical placement and assignment-bit mapping.
All source preparation, sign, syndrome, decoder, uncomputation, reset,
readout and barrier events remain in order. No opaque rotation, decoder or
CCX remains in the compiled streams.

| Per complete compiled draw | $n=2$ | $n=3$ |
|---|---:|---:|
| Logical source wires | 6 | 9 |
| H | 173 | 310 |
| CX | 368 | 1704 |
| S | 105 | 158 |
| X / Z | 13 / 1 | 29 / 1 |
| T / T-dagger | 171 / 15 | 329 / 54 |
| Resets / assignment readouts / barriers | 6 / 2 / 7 | 9 / 3 / 7 |
| Materialized operations, including those events | 861 | 2604 |
| T-state demand per draw | 186 | 383 |
| Fixed complete draws | 5 | 3 |
| Complete-task T-state demand | 930 | 1149 |

The counts include the actual synthesized words at every occurrence. They
replace neither the accepted 101/317 ideal unitary counts nor the separate
dirty-scratch default’s 377-unitary record. The increased CX counts include
the selected line routing; they are not a measured hardware depth.

The independent checker reconstructed the routing actions and all event
ledgers, then compared full complex final states with Walsh references built
directly from the individual constraints. Assignment probabilities sum over
every clause and workspace value. Cleanup was not postselected, and global
phase was not normalized away.

| New compiled diagnostic | $n=2$ | $n=3$ |
|---|---:|---:|
| Full-state L2 discrepancy | `9.19641531153446e-06` | `1.0053162966495508e-05` |
| Unconditional assignment total variation | `7.303067726935342e-06` | `6.6970724739035675e-06` |
| Clause/workspace leakage probability | `3.469961287192496e-13` | `8.316980791226311e-13` |
| Compiled maximum-objective probability | `0.7500035191373087` | `0.9000017057044327` |
| Guarded occurrence-error sum plus reference padding | `2.0612256072003386e-05` | `2.9702927775257846e-05` |
| Required per-draw realization ceiling | $1/2000$ | $1/1200$ |

The realization allowance is the upward-rounded sum over all five/seven RY
occurrences, including reused words, plus $10^{-10}$ numerical reference padding. Its rationale
uses the operator-norm bound by Frobenius norm, telescoping over the unitary
sequence, and contraction through reset/readout. The numerical total-variation
diagnostic is separate from that conservative occurrence sum. These floating
guards and padding are numerical scenario assumptions, not proved interval
certificates. The assignment-readout task supplies no controlled-oracle
interface guarantee.

The [saved resource packet](../../assets/generated/optimization/dqi-structured-resources/manifest.json)
retains the exact [synthesized words](../../assets/generated/optimization/dqi-structured-resources/native-words.json),
compiled operation streams and phase metadata, [coherent checks](../../assets/generated/optimization/dqi-structured-resources/verification.json),
[physical ledgers](../../assets/generated/optimization/dqi-structured-resources/physical.json)
and a separate [computational acceptance record](../../assets/generated/optimization/dqi-structured-resources/acceptance.json).
Its fourteen-file manifest binds this new saved-evidence supplement; the earlier
25-file ideal bundle and notebook remain unchanged.

## Complete-task quality and fixed surface-code resources

For each size separately, the requested output is an assignment with score
$f(x)=n$ and conditional success at least $99/100$. Execute every fixed draw,
score every recorded assignment, and return the highest score, breaking ties
by lowest integer assignment. This contract is conditional on faithful
assignment-bit transfer and correct classical scoring/selection. It is not
a simultaneous 99% claim for both sizes.

The miss budget reserves four separate allowances of $1/400$: ideal miss,
realization discrepancy, physical faults and factory unavailability. The
accepted ideal law gives five draws at $n=2$ and three at $n=3$ to fit the
ideal-miss allowance. The earlier four/two-draw examples reserved only ideal
miss and therefore do not supply the new complete-task budget. The report
sets $\eta_n$ to the upward-rounded guarded occurrence sum plus reference padding
above, constrained by the respective per-draw ceiling, and uses

$$
P(\mathrm{miss})\leq(1-p_{\rm opt})^R+R\eta_n+
\delta_{\rm physical}+\delta_{\rm availability}.
$$

Reusing a synthesized word does not treat its coherent discrepancy as an
independent random fault. The reported physical and availability components
also remain distinct.

The physical scenario retains the serialized instrument baseline
{cite:p}`resource_litinski_2019,resource_horsman_2012,resource_qdk_134`:

| Frozen assumption | Value |
|---|---:|
| Physical error | $10^{-4}$ |
| Physical gate / measurement / initialization | 100 / 500 / 100 ns |
| Code cycle / controller reaction | 1000 / 10000 ns |
| Memory / initialization residual prefactors | 0.1 / 0.1 |
| Threshold | 0.01 |
| Controller failure per decision | $10^{-15}$ |
| Complete factory-pipeline rejection bound | 0.01 |
| Feedback rounds reserved per pipeline | 3 |
| Maximum code-distance field | 49 |

One candidate was fixed before the pass: data and distillation distance 15,
with one just-in-time factory. Its inherited native T instruction supplies
330000 ns nominal pipeline time, 236623 physical qubits and the exact output
error bits `0x1.88ea67f531f32p-56`. This restores recorded model-cost fields;
it does not reconstruct the original distillation pipeline or certify its
layout. Native result `copies=1` was required. The ambiguous `runs`, `states`
and `error_rate` result metadata are retained without using them in the
accounting. There was no factory search, Pareto comparison or optimality claim.

The layout account reserves $4Q-1$ data, bridge and transport patches, where
$Q$ is logical source width. It charges initialization of all 23/35 reserved
patches on every draw, including the additional 17/26 beyond the source
wires. Protected readout uses the majority of $d$ disjoint length-$d$ paths
with independent physical readout errors. Data-memory exposure per
patch/cycle and residual initialization use
$0.1(p/0.01)^{(d+1)/2}$.

All initial resets, assignment readouts, adjacent CXs, Clifford-frame
decisions, T/T-dagger transport and injection, feedforward, controller
dispatch, factory feedback reserves, data/workspace idling and output-port
residence are charged. For $N_T$ complete-task T states, the smallest allowed
number $K$ of complete pipeline attempts per state satisfies
$N_T(0.01)^K\leq1/400$. Both tasks require $K=3$, including the first attempt,
giving 2790/3447 reserved attempts. This assumes independent complete
attempts with rejection at most 0.01. It is separate from output-state
quality, and is not a whole-pipeline certificate derived from a native
per-round readiness setting.

| Fixed complete-task model | $n=2$ | $n=3$ |
|---|---:|---:|
| Complete draws | 5 | 3 |
| Reserved patches | 23 | 35 |
| Data/workspace physical qubits | 10327 | 15715 |
| Factory physical qubits | 236623 | 236623 |
| Peak physical qubits | 246950 | 252338 |
| Per-draw modeled runtime | 636118200 ns | 1952184000 ns |
| Whole-task modeled runtime | 3180591000 ns | 5856552000 ns |
| Whole-task physical failure | `8.476782446001864e-10` | `2.279408893274473e-09` |
| Exact factory-unavailability bound | $93/100000$ | $1149/1000000$ |
| Exact ideal miss | $1/1024$ | $1/1000$ |
| Conditional success lower bound | `0.9979903753719617` | `0.9977618889372653` |

The physical-failure ledger includes memory exposure, controller decisions,
initialization, protected readout, factory output quality and output
handoff/storage. Its largest component is data/workspace exposure:
`7.315368500000002e-10` and `2.0497932000000004e-09` across the complete tasks.
Each direct QDK `Trace.estimate` result matched the explicitly assembled
ledger’s runtime, qubit count and union-bound failure value. The existing
eight-wire instrument API was unchanged; a separate bounded adapter handled
these six/nine-wire inputs under the same assumptions.

Runtime follows a serial schedule with nominal factory-cost input and the
explicit complete-attempt reservations above. It supplies no measured
factory-timing tail or certified layout geometry.

These are conditional, unoptimized small-task results. Factory internal
circuits, startup encoder correctness, certified factory/data geometry,
hardware decoder throughput and measured controller latency remain
unavailable. Assignment transfer and host scoring/selection reliability or
latency, offline coefficient/circuit/synthesis work, final reporting time,
nominal factory-timing tails and formal interval bounds are excluded from a
hardware guarantee. Consequently, the table is a conditional
quantum/controller schedule, not complete hardware end-to-end wall time.

The native CPU command exited zero after an observed 6.151954624991049 seconds;
the saved parent reported 6.105968792006024 seconds of work before retention.
It sampled 80 child-RSS readings, with peak observed 142147584 bytes, within
the 1 GiB bound. The verifier’s conservative quantum-storage accounts were
114688/802816 bytes within the 32 MiB cap. Sampled RSS does not certify an
unobserved instantaneous peak, and these CPU observations are separate from
the table’s modeled quantum runtime. This pass executed neither larger
compiled cases nor the 5/3 assignment draws.

The classical optimum remains immediate: write $x=0^n$, which
satisfies $n$ equations; satisfying all $n+1$ would contradict parity.
Output writing and optional direct verification are linear in the declared
input/output size. This circuit family measures construction and accounting
overhead, with no optimization-advantage claim. Higher-degree preparation and
coherent decoding remain planned and require new instances, support
promises, phase checks and complete resource ledgers.

The standard-library degree-one generator and its existing bundle remain
portable source-generation interfaces. The native resource replay is bound
to the observed Darwin interpreter and installed binary fingerprints; its
version lock does not establish Linux or arbitrary-wheel replay. The new
`scripts/verify_dqi_structured_resource_saved.py` reads only the saved packet,
checking file hashes, schemas and scalar consistency with the standard
library. It launches no synthesis, state evolution or estimator. From the
source hierarchy, inspect the packet with:

```bash
python scripts/verify_dqi_structured_resource_saved.py assets/generated/optimization/dqi-structured-resources
```

## Portable companion and generation limits

The [ordinary notebook](../../notebooks/optimization_dqi_structured.ipynb) completed four cells in a standard Linux Python 3.12.14 kernel. It checks saved artifacts and presents the new small outcomes and text circuit; it does not repeat the quantum simulation. Keep the complete 25-file [portable bundle](../../assets/generated/optimization/dqi-structured/bundle.json) together. Its manifest binds source/data files and the notebook cell-type/source projection while allowing normal output and execution-counter updates.

The bundle includes the [module](../../assets/generated/optimization/dqi-structured/optimization_dqi_structured.py), [generator](../../assets/generated/optimization/dqi-structured/generate_optimization_dqi_structured.py), [small validator](../../assets/generated/optimization/dqi-structured/validate_optimization_dqi_structured.py), [saved verifier](../../assets/generated/optimization/dqi-structured/verify_dqi_structured_saved.py), [exact version lock](../../assets/generated/optimization/dqi-structured/requirements-lock.txt) and [project MIT license](../../assets/generated/optimization/dqi-structured/LICENSE.txt). Normal public paths require no private hook, environment variable or execution freeze.

The generator CLI streamed the $n=8$ [QASM](../../assets/generated/optimization/dqi-structured/large/dqi-structured.qasm) and [operation records](../../assets/generated/optimization/dqi-structured/large/operations.ndjson), with 24 wires and 2876 source operations. Its [generation-only receipt](../../assets/generated/optimization/dqi-structured/large/dqi-structured-generation.json) explicitly records no simulation or coherent validation. This invocation tightened limits to eight assignment bits, 24 wires, 4096 operations, 256 KiB export and ten seconds. It did not prepare a larger quantum state or measure a hardware result.

The wider source generator accepts this family through $n=64$, with hard ceilings of 192 wires, 200,000 operations and 32 MiB export; callers may tighten them. Its ten-second default/thirty-second maximum CLI deadline and cooperative API phase checks remain ordinary public interfaces. Only $n=8$ generator CLI execution is evidenced here, so this is not an executed sweep through all supported sizes or a universal timing/memory benchmark. Generation cannot launch a dense simulation implicitly.

From the complete downloaded directory, use fresh output directories:

```bash
python generate_optimization_dqi_structured.py --n 8 --output-dir dqi-structured-n8
python validate_optimization_dqi_structured.py --output-dir dqi-structured-small
python verify_dqi_structured_saved.py .
```

The validator accepts the fixed $n=2,3$ pair only, with a ten-second default/thirty-second maximum deadline, 192 KiB default export ceiling and 1 MiB hard maximum. Export receipts remain provisional until the CLI exits zero or the API returns normally. The retained [invocation-acceptance record](../../assets/generated/optimization/dqi-structured/invocation-acceptance.json) binds actual zero CLI exits for this pass; an export-status flag alone is insufficient. Successful generation remains generation-only and does not validate quantum probabilities.

The quantum-array source envelope remained within its 32 MiB cap. The recorded $n=3$ conservative quantum-buffer bound is 17,039,360 bytes. The whole pipeline sampled 254,144,512 bytes of concurrent target RSS over 608 samples; that scope includes its supervisors and held descendants, excludes SSH ancestors, and does not certify an unobserved instantaneous peak. These are CPU validation observations, separate from logical-qubit counts and physical estimates.

Input is the compact declared family, compiled into known sign and syndrome gates, rather than a QRAM or general matrix-loading oracle. Angle calculation, classical generation/export, compilation and scoring/reporting time remain separate from quantum hardware runtime. The 6.438-second pre-seal pipeline observation is not an end-to-end quantum runtime estimate.

The fixed $n=2,3$ resource increment above supplies new synthesized words, compiled-state diagnostics and a conditional physical scenario. It does not extend physical validation to the $n=8$ generation record or the wider supported generator. Hardware geometry, decoder throughput, measured controller and host latency, formal error bounds and a co-design frontier remain planned. The earlier ideal study and frozen baseline are retained.

## Sources, license and formal coverage

The construction, explanation and primitive source are original MIT project work. The generator is a standard-library stream; the small validator used the pinned Python 3.12 environment with Qiskit 2.6.2 (Apache-2.0) and NumPy 2.5.4 (top-level BSD-3-Clause with bundled license notices) {cite:p}`opt_qiskit262,opt_numpy254`. The supplied lock records these exact versions; the dependencies remain governed by their upstream license notices. No authors' benchmark code has been adopted.

The separate compiled resource pass used Qiskit 2.6.2 (Apache-2.0) and QDK 1.34.0 (MIT). Qiskit's 2.6.2 source lock pins rsgridsynth 0.2.2 (MIT); the installed wheel has no direct certificate of its native crate version. The exact wrapper and wheel identities were retained, alongside the versioned upstream references {cite:p}`resource_qiskit_262,resource_rsgridsynth_022,resource_qdk_134`.

The Nature paper provides the DQI formulation {cite:p}`frontier_dqi2025`. Its inspected versioned experiment archive remains an external MIT reference {cite:p}`frontier_dqi_archive`. This concrete family and its algebraic accounting are not attributed to an executed author benchmark. No local Lean, Rocq or other formal theorem has been reproduced.

## References

```{bibliography}
```
