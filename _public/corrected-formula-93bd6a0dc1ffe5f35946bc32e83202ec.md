---
title: "Corrected product formula with explicit pair boundaries"
description: "An executed disjoint-pair CPF2 circuit, independent small checks, scalable generation and explicit resource limits."
---

:::{note} Status: small circuit and classical checks executed

The original numerical component completed 24 tests and 32 paired CPF2/Strang
cases. A separately completed genuine notebook executed its five original code
cells, including one- and two-pair checks and the circuit drawing. Larger
circuits below are generation-only. These results cover the specified
noninteracting pair model; physical estimation, rotation synthesis and
formalization remain planned. The new standalone wrapper and adapted portable
notebook are prepared and unexecuted.
:::

## Motivation and intuition

A product formula's short alternating evolutions introduce commutator errors.
A corrector rotates the computational frame around that formula to cancel a
selected leading error. A symplectic corrector has an inverse on its other
side, so equal intermediate boundary operations cancel over a fixed sequence.
The remaining first and last operations still cost gates.

This page selects the first symplectic correction to Strang splitting from
Bagherimehrab and colleagues {cite:p}`ham_corrected_formulas_2026`. The model
has disjoint two-spin pairs: it is classically factorized and provides a
transparent circuit check, with no application-advantage claim.

## Task, fundamental formula and assumptions

For $m$ pairs and $n=2m$ wires, define

$$A=\sum_{j=0}^{m-1}a_jZ_{2j}Z_{2j+1},\qquad
B_0=\sum_{j=0}^{m-1}b_jX_{2j},\qquad H=A+\alpha B_0.$$

The rational inputs have $0\le\alpha\le1$, $|a_j|,|b_j|\le4$ and $|t|\le4$.
The energy unit and $\hbar=1$ fix the meaning of $t$. The known input uses H
on every wire and S on each lower pair wire, giving
$|+y\rangle_{2j}|+\rangle_{2j+1}$. The measured output is the $Z_0$ mean;
the circuit retains every system Z-readout bit.

Set $\delta=t/r$ and

$$S_2(\delta)=e^{-iA\delta/2}e^{-i\alpha B_0\delta}e^{-iA\delta/2}.$$

Equations 25/30 of the paper use $\lambda=-i\delta$ and
$C=-\alpha\lambda^2[A,B_0]/24$. Here

$$[A,B_0]=2i\sum_j a_jb_jY_{2j}Z_{2j+1},\qquad
C=\frac{i\alpha\delta^2}{12}\sum_j a_jb_jY_{2j}Z_{2j+1}.$$

The implemented complete evolution is

$$V_{\mathrm{CPF}}=e^C S_2(\delta)^r e^{-C}.$$

The chronological circuit starts with $e^{-C}$ and ends with $e^C$.
Different pairs commute, so each boundary is a product of exact declared
Pauli rotations. This restricted family needs no approximation to a
commutator exponential. Its gates still use binary64 angles and later need
synthesis and physical error accounting.

The paper's perturbative local residual has the form
$O(\alpha^2|\delta|^3+\alpha|\delta|^5)$ with model-dependent constants.
At fixed nonzero $\alpha$, this remains generally second order globally.
Small-$\alpha$ prefactor improvements require comparable unscaled norms.
Here $\|A\|=\sum_j|a_j|$ and $\|B_0\|=\sum_j|b_j|$; the ledger records their
ratio, including undefined cases when $A=0$. The saved study reports both
methods' errors; no universal CPF improvement or fourth-order claim follows
from this finite pair model.

## Literal boundary circuit and executed small demo

For a first wire $f=2j$ and partner $p=2j+1$, the sequence

```text
Sdag(f); H(f); CX(f,p); RZ(theta,p); CX(f,p); H(f); S(f)
```

implements $e^{-i\theta Y_fZ_p/2}$ in the ideal named-gate model. The first
boundary uses $\theta=+\alpha a_jb_j\delta^2/6$; the final boundary uses its
negative. Qiskit labels read higher wire first, so the literal pair Pauli
label is `ZY`. Each Strang pair-step uses two CX–RZ–CX ZZ half-rotations and
one RX on the first wire. No dense quantum gate or symbolic oracle appears.

The small demo fixes one pair, $a=b=1$, $\alpha=1/5$, $t=1$, $r=4$.
The [original executed notebook](../../notebooks/simulation_corrected_formula.ipynb)
contains the genuine output. This complete trial has 49 literal primitives,
including resets, preparation, both boundaries and both measurements.

```{figure} ../../assets/generated/simulation/corrected-formula/one-pair-complete-trial.png
:alt: Executed two-wire CPF2 circuit with preparation, first and final YZ correctors, four Strang steps, resets and readout.
:width: 100%

The original notebook's circuit output, extracted without redrawing. Its two
PNG outputs were byte-identical; this figure displays that saved image once.
```

The independent exact-evolution and circuit checks give the following saved
results for this fixed demo. Operator error is the spectral norm distance to
$e^{-iHt}$; observable bias compares the ideal $Z_0$ mean with exact dynamics.

| Saved quantity | Strang | CPF2 |
|---|---:|---:|
| Complete-trial primitives | 35 | 49 |
| Continuous rotation calls | 12 | 14 |
| Operator error against exact dynamics | $1.90117\times10^{-3}$ | $4.18460\times10^{-4}$ |
| $Z_0$ mean | 0.17693562 | 0.17511281 |
| Absolute $Z_0$ bias | $1.97925\times10^{-3}$ | $1.56449\times10^{-4}$ |
| Circuit/operator-formula entry discrepancy | $1.57009\times10^{-16}$ | $4.96507\times10^{-16}$ |

CPF2 improves these two errors on this instance while adding 14 primitives.
This comparison has no synthesized rotations, physical faults or sampled
readout population. The [exact saved case](../../assets/generated/simulation/corrected-formula/cases/case-06/record.json)
retains probabilities, all comparison metrics and both complete ledgers;
download its [CPF2 QASM](../../assets/generated/simulation/corrected-formula/cases/case-06/cpf2.qasm)
or [Strang QASM](../../assets/generated/simulation/corrected-formula/cases/case-06/strang.qasm).

The full saved study contains 32 fixed cases: one/two pairs;
$\alpha=1,1/5,1/20$; $r=1,2,4,8$ at $t=1$, followed by declared negative-time,
zero-time, zero-$\alpha$ and signed heterogeneous controls. No parameters
were selected from observed performance. Inspect the
[64-row comparison table](../../assets/generated/simulation/corrected-formula/case-summary.csv)
and [complete saved study](../../assets/generated/simulation/corrected-formula/saved-study.json)
for the controls and other parameters.

:::{important} Original evidence and adapted code have different status

The original notebook archive preserves every byte, including its frozen
pre-execution “planned” prose and metadata. Its later execution counts and
outputs are genuine. The [immutable archive](../../assets/generated/simulation/corrected-formula/original-executed.ipynb)
uses the original repository imports; it is an evidence record rather than a
standalone adaptation. The separate
[portable companion](../../assets/generated/simulation/corrected-formula/portable_demo.ipynb)
has a new file loader, empty outputs and null execution counts. It has not
executed. The original numerical run's completed components and the later
notebook completion are recorded separately; the first full run's notebook
startup failure has not been rewritten as success.
:::

## End-to-end resource analysis and size scaling

With every zero-angle call retained and no gate merging, the complete CPF
trial has the following literal source counts. The executed ledgers check
these counts on the saved cases and generation examples.

| Primitive or workload | Source count |
|---|---:|
| RX | $mr$ |
| RZ, including boundaries | $(2r+2)m$ |
| CX | $(4r+4)m$ |
| H / S / S-dagger | $6m$ / $3m$ / $2m$ |
| Initial resets / final measurements | $2m$ / $2m$ |
| All complete-trial primitives | $(7r+21)m$ |

The matched Strang trial uses $(7r+7)m$ primitives with the same input and
readout; the two boundary circuits add $14m$. Generator work is $O(mr)$
at fixed formula and parameters. Keeping an accuracy target as $m,t$ change
requires a separate error analysis and step choice. The current generator
supplies no unexecuted fixed-accuracy scaling guarantee.

For $M$ independent observable trials, preparation, both boundaries, every
step, reset and readout are charged $M$ times. Finite-shot populations have
not run. Classical coefficient handling and final report latency, rotation
synthesis, fault-tolerant routing, code cycles and factories need further
accounting. The surface-code baseline stays frozen until a matched physical
scenario is implemented. Raw gate counts are not a physical runtime.

At $a=b=1$, $\alpha=1/5$, $t=1$, $r=8$, the original generator materialized
the following complete CPF2 trials. These rows do not run dense simulation.

| Pairs $m$ | Wires $n=2m$ | Complete primitives | Continuous rotations | CX | Dependency depth |
|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 308 | 104 | 144 | 74 |
| 16 | 32 | 1,232 | 416 | 576 | 74 |
| 32 | 64 | 2,464 | 832 | 1,152 | 74 |

The constant recorded depth comes from independent disjoint pairs and fixed
$r$ in Qiskit's dependency graph. Routing, fault-tolerant operation durations,
synthesis and readout scheduling are absent. Download the
[saved generation records](../../assets/generated/simulation/corrected-formula/saved-generation.json)
and the [64-wire CPF2 circuit](../../assets/generated/simulation/corrected-formula/generation/m32/cpf2.qasm).

The exact sum of $|\theta_{\rm float}-\theta_{\rm rational}|/2$ over all
rotation occurrences is the ledger's ideal-gate conversion bound. It covers
both corrector boundaries. Splitting error, SDK arithmetic, synthesis and
physical faults remain separate; it is not a hardware certificate.

## Classical checks and formalization

The 24 executed tests compare all complex entries,
without phase alignment, against independently written $4\times4$ Pauli
references. For each pair, $H_j^2=(a_j^2+\alpha^2b_j^2)I$ gives a closed-form
exact evolution. A separate full bit-basis Hamiltonian and SciPy exponential
check that reference, while a closed-form Strang matrix checks chronology.
The producer requires the independent-reference, preparation and reset
discrepancies to be finite and at most $10^{-11}$.

Other executed checks cover the seven-gate YZ identity, corrector
conjugation, a signed two-pair model, a complex correlated input, preparation
and reset from a mixed input, and actual classical readout mappings. Missing
and wrong-sign boundary controls are frozen on a noncommuting one-pair case;
their saved full-complex-operator contrasts are 0.0275952 for missing
boundaries and 0.0549697 for wrong-sign boundaries; the correct construction's
entry discrepancy is $3.51083\times10^{-16}$. These are fixed numerical
controls, not formal or hardware certificates. The
[complete control record](../../assets/generated/simulation/corrected-formula/negative-controls.json)
retains its exact model and values.

No Lean/Rocq proof is supplied. The paper's mathematical result and the
independent small numerical checks do not prove full implementation
correctness for all supported inputs.

## Public code, reproducibility and prepared portability

Use Python 3.12.14 and the existing environment lock; no dependency change is
required. The original source is MIT-licensed and uses locked Qiskit 2.6.2
(Apache-2.0), NumPy 2.5.4 (BSD-3-Clause) and SciPy 1.18.1 (BSD-3-Clause)
{cite:p}`sim_qiskit_262,opt_numpy254,sim_scipy_1181`.

Download the [public packet](../../assets/generated/simulation/corrected-formula/cpf2-public-packet.zip),
[original scientific module](../../assets/generated/simulation/corrected-formula/corrected_product_formula.py),
[lock](../../assets/generated/simulation/corrected-formula/requirements-lock.txt)
and [license](../../assets/generated/simulation/corrected-formula/LICENSE.txt).
The [source provenance](../../assets/generated/simulation/corrected-formula/source-provenance.json)
records original source, data and executed-notebook hashes. Original module
and producer docstrings retain their source-freeze “unexecuted” descriptions;
they were not silently edited when the separately recorded runs completed.

The archived [original producer](../../assets/generated/simulation/corrected-formula/original_repo_producer.py)
uses Git and repository source files, including tests and the source notebook.
It is repository-dependent. The new
[standalone wrapper](../../assets/generated/simulation/corrected-formula/generate_cpf2_public.py)
loads the identical module beside itself and authenticates its source, lock
and license hashes. It retains the original input bounds, finite residual
checks, tolerances, fixed suite and negative controls. **This wrapper has not
executed.** The following are prepared commands for a later portability check:

```bash
# In the extracted packet, with the recorded environment installed:
python generate_cpf2_public.py --pairs 4 --repetitions 8 \
  --generation-only --output-dir ../cpf2-new-generation
# A separately bounded numerical run would use the default one-pair case:
python generate_cpf2_public.py --output-dir ../cpf2-new-small-check
```

The first command constructs two measured-trial QASMs and ledgers without
any dense check. Matching those QASM bytes with the original four-pair exports
is a proposed portability check. It has not run and would not validate the
adapted notebook's numerical cells. Every output directory must be fresh.
Generation stops at
32 pairs/64 wires, 1,024 steps and 100,000 complete primitives. Dense checks
stop separately at two pairs/four wires, 16 steps and 2,000 source operations.
Large generation does not imply large simulation. The wrapper has cooperative
90-second and 32 MiB/128-file output bounds; it makes no hard memory or process
containment claim. The [packet README](../../assets/generated/simulation/corrected-formula/README.md)
describes the original evidence, adaptations, dependency closure and limits.

## References

```{bibliography}
```
