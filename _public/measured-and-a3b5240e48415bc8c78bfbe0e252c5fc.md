---
title: "Measured-AND ripple arithmetic"
description: "Explicit four-T CCX replacement with measurement and feedback, full ripple counts and separate physical-model limits."
downloads:
  - file: ../../notebooks/shor_measured_and.ipynb
    title: "Measured-AND notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Complete Python source package"
  - file: ../../assets/generated/shor/measured-and-artifacts.zip
    title: "Measured-AND circuits, source snapshots and study"
  - file: ../../assets/generated/shor/measured-and-validation.json
    title: "Bounded validation summary"
  - file: ../../assets/generated/shor/measured-and/scaling.csv
    title: "Materialized logical scaling counts"
---

## Motivation and intuition

A temporary logical AND stores a predicate in a clean helper qubit. It can be
cheaper to compute that predicate, use it, and erase it by measurement than to
reverse its non-Clifford circuit. Gidney introduced this route for quantum
addition {cite:p}`shor_gidney_addition2018`.

Measurement does not have to destroy the data coherence. Measuring the helper
in the X basis adds a known conditional phase to the data; a Clifford CZ
correction removes that phase. The helper's measured value also tells us
whether an X is needed to return it to zero.

:::{note} Executed ideal-circuit checkpoint
The dedicated module tests passed **109 cases**. The generator executed six
bounded addition checks and two small order-finding checks, and the companion
notebook executed all seven code cells in a genuine Jupyter kernel, including
its circuit figure. The [validation summary](../../assets/generated/shor/measured-and-validation.json)
records exact source and artifact hashes. Larger circuit exports remain
generation-only. No physical estimate, hardware execution or mechanized proof
is claimed.
:::

## Fundamental construction and complexity

For data controls a,b and a clean helper h, our literal computation is

```text
H(h), T(h), CX(b,h), Tdg(h), CX(a,h), T(h),
CX(b,h), Tdg(h), H(h), Sdg(h).
```

The first nine gates are a four-T relative-phase construction. On clean-helper
inputs its |11,0> branch has phase i; the final Sdg removes that phase, giving
|a,b,0> → |a,b,a∧b>. No claim is made that this computation is an exact CCX on
dirty-helper inputs.

Copy the predicate into an arbitrary data target with CX(h,target), apply
H(h), and measure h into m. The two uncorrected data Kraus operators are

\[
K_m=2^{-1/2}\,CZ(a,b)^m\,CCX(a,b,target).
\]

Applying CZ iff m=1 makes both corrected branches CCX/√2. The data channel is
therefore exactly CCX in the ideal gate model, including coherent inputs and
entanglement with other registers. Applying X(h) iff m=1 clears the measured
helper without a second measurement. A separate literal-reset variant is
provided for comparison. Each lifetime keeps a and b unchanged until its
correction completes.

Our [ripple baseline](ripple-arithmetic.md) has 2w CCX and 4w CX gates before
lowering. Replacing each CCX gives **8w literal T/Tdg gates**, compared with
14w for the standard seven-T decomposition. This is an unoptimized MAJ/UMA
drop-in variant, **not Gidney's separately optimized 4w−4 adder**. Measurements,
helper storage and controller delays remain costs.

## Small circuits and important variants

The executed numerical checks cover all eight CCX data basis columns, both
measurement outcomes and coherent data states. They compare the actual
primitive circuit's Kraus operators with an independent classical CCX
permutation. Addition checks use independent integer arithmetic and coherent
permutation references. The small default is two-bit addition 3+1 modulo 4,
with carry-in zero; the first input and carry must be restored.

```{figure} ../../assets/generated/shor/measured-and/gadget.svg
:name: measured-and-gadget

The literal measured CCX replacement: clean helper initialization, four T/Tdg
gates, helper measurement and the actual conditional phase correction/clear.
```

```{figure} ../../assets/generated/shor/measured-and/addition.svg
:name: measured-and-addition

Executed two-bit addition 3+1 modulo 4. The helper is q5; c0 carries its four
successive feedback decisions, while c1 through c5 record the data readout.
Both input restoration and the sum register are checked independently.
```

The producer checks these fixed inputs with both the seven-T baseline and the
measured replacement:

| Width w | a | b | Carry-in | Checked sum modulo 2^w |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 |
| 2 | 3 | 1 | 0 | 0 |
| 3 | 5 | 3 | 1 | 1 |

Across the six checks, the maximum stored density-matrix error is
5.996×10⁻¹⁵. All six exact sparse checks report normalization exactly one;
the measured cases also execute helper-restoration assertions. This is
bounded ideal-circuit evidence.

The generator also transforms the existing controlled ripple modular powers
inside order finding. Static controls and inverses are expanded **before**
measurement-assisted replacement. The resulting dynamic circuit is not a
Gate that can be controlled or inverted by calling a generic method. Inverse
QFT P/CP phases remain present and have their own synthesis requirements.

Executed bounded full-circuit checks use N=15 and N=21, a=2 and four phase bits.
They compare the coherent circuit with an independent modular-permutation
reference before final readout; the exported QASM additionally includes four
phase measurements and their classical bits. The stored distribution errors
are below 10⁻¹⁶ for both cases.
Large 24-bit addition and 12-bit order-finding exports are separate
materializations, not validated large factoring runs.

## End-to-end resource accounting

Let W be the number of input X gates, namely popcount(a)+popcount(b)+carry.
Both complete addition demos reset all 2w+1 data qubits and measure all of
them at the end. The measured circuit additionally initializes one helper.
One classical feedback bit is reused after every completed correction.

| Complete prepared addition | Seven-T static baseline | Measured-AND default |
|---|---:|---:|
| Logical quantum wires | 2w+1 | 2w+2 |
| Additional feedback bits | 0 | 1 |
| Literal T/Tdg demand | 14w | 8w |
| Helper X-basis measurements/decisions | 0 | 2w |
| Initial data/helper reset instructions | 2w+1 | 2w+2 |
| Final data readout bits | 2w+1 | 2w+1 |
| Complete source instructions | 38w+2+W | 34w+3+W through 42w+3+W |

The measured instruction range corresponds to all-zero through all-one
feedback. Uniform ideal outcomes give expected count 38w+3+W. This expectation
is not a physical-runtime model. The helper's 2w semantic clears are actual
conditional X gates already included in the range, rather than uncounted
resets. The logical dependency ledger includes a conservative unit-cost
classical decision; it is not an optimized physical schedule.

The [materialized scaling data](../../assets/generated/shor/measured-and/scaling.csv)
contains 16 logical-count rows, comparing both methods at widths
1, 2, 3, 4, 8, 16, 24 and 31. These rows count constructed circuits; they are
not a physical resource sweep or quantum simulations at every width.

For order finding with t clock bits and n integer bits, the existing source
has O(t n²+t²) gates and t+3n+4 wires. Replacement adds one helper and O(t n²)
measurements/feedback opportunities. These are source-scaling statements;
QFT rotation synthesis, repeated quantum trials, classical recovery and
physical success require their own full factoring task contract.

### Fault-tolerant comparison boundary

No physical estimate has been computed for this instrument. The shared static
QDK adapter rejects intermediate measurements, resets and if/else. The fixed
instrument scheduler also rejects arbitrary source feedback and caps its
current source width at eight qubits. Neither can be bypassed by silently
deleting those operations.

A matched comparison needs an explicit controller/routing envelope, all
initializations and readouts, idle exposure, factory transport and fault
allocations. The old ripple PSSPC table, which assumes logical-zero inputs
without explicit resets, is not the comparator for this prepared instrument.
The surface-code assumptions must be frozen equally for a new baseline and
candidate. Lower T demand alone does not establish lower time, qubits or
spacetime volume.

## Classical simulation and formalization

Dense branch-density checks are bounded separately at ten quantum wires and
an event-by-matrix work ceiling. The larger small order circuits use an
original exact sparse interpreter in Q(ζ16), with ζ16⁸=−1. It applies every
literal gate and both measured branches; branches merge only after exact
equality and helper restoration. It does not substitute a Boolean CCX oracle
or truncate small amplitudes.

That interpreter recognizes the intended dyadic π/8 angle model from exact
Python parameter spellings. It is not an enclosure for binary-angle hardware
realization, a general phase simulator or a proof kernel. Numerical density
checks and the mathematical derivation are distinct evidence. No Lean or
Rocq proof is integrated for this replacement. The [Shor formal-reference
section](../shor.md#available-external-formalization) retains the scope of
external developments {cite:p}`shor_verified_peng2023`.

## Reproduce and limits

```bash
python scripts/generate_shor_measured_and.py --suite \
  --output-dir .cache/measured-and
pytest tests/test_shor_measured_and.py
export OPENQUANTUM_AND_ARTIFACT_DIR="$PWD/.cache/measured-and"
```

Set that environment variable before running the companion notebook so it
reads the same generated study. The output directory must be absent. Source materialization is capped at 64
total quantum wires and 500,000 primitive events; ripple width is at most 31.
Exact sparse simulation is capped at four phase bits, 8,192 amplitude states
and 20 million gate-state applications. Larger exported circuits are marked
unvalidated. Use the complete public code package for the src/openquantum and
scripts layout; individual flat source snapshots are not an installed package.

The [original executed notebook snapshot](../../assets/generated/shor/measured-and-executed.ipynb) retains the validation summary's exact PNG record. Publication may refresh the current notebook's figure output; the saved verifier authenticates its ordered sources, seven executed cells and current package provenance while preserving the original study. Routine saved verification uses `python scripts/verify_shor_measured_and_saved.py` and executes no quantum simulation.

The code, narrative and figures are original MIT work. Standard gates use
Qiskit 2.6.2 (Apache-2.0); the inspected [upstream relative-phase gate
definition](https://github.com/Qiskit/qiskit/blob/2.6.2/qiskit/circuit/library/standard_gates/x.py)
is credited as a semantic reference rather than adopted source code.

## References

```{bibliography}
```
