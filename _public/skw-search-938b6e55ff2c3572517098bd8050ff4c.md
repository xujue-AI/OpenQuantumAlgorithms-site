---
title: "Marked-hypercube search"
description: "An explicit bounded SKW-style search circuit, clean-workspace checks, exact Clifford+T lowering and fixed-workload surface-code estimates."
downloads:
  - file: ../../notebooks/optimization_walk.ipynb
    title: "Executable SKW search notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Python examples and tests"
---

## Motivation and intuition

A marked vertex changes the coin used by a quantum walk. Repeated coin and
shift operations make paths interfere, increasing the chance of reading that
vertex at suitable times. Shenvi, Kempe and Whaley introduced a hypercube
search algorithm with $O(\sqrt N)$ oracle calls in its stated search model
{cite:p}`opt_skw2003`. This page implements a bounded teaching instance with a
fully constructed predicate, rather than treating the marking operation as a
free box.

:::{note} Implemented scope

The default four-dimensional hypercube circuit and its independent classical
reference are executed. The supplied marked value is known to the constructor:
it specifies an equality predicate, not an unknown database or application
benchmark. Larger supported circuits have separate source-only exports. The
physical report is a conditional estimate of one fixed readout workload, and
there is no integrated Lean proof or hardware execution.
:::

## Fundamental algorithm and task contract

For hypercube dimension $d$, positions are $x\in\{0,\ldots,2^d-1\}$ and the
coin selects one of $d$ directions. This implementation uses $d=2^r$, with $r$
coin qubits, $d$ position qubits and $d+r-1$ clean work qubits. Coin bits occupy
the low wires, so a basis index is

$$j=c+2^r x+2^{d+r}w.$$

The work register begins at zero. Hadamards on all $d+r$ active wires prepare
uniform coin and position. With $|s\rangle=d^{-1/2}\sum_c|c\rangle$, one
implemented iteration is

$$U_{\rm impl}=S\widetilde G R,\qquad
R=I-2|s,m\rangle\langle s,m|,\qquad
\widetilde G=I-2|s\rangle\langle s|,$$

$$S|c,x\rangle=|c,x\oplus2^c\rangle.$$

The chronological order is marking, coin, shift. The joint reflection uses
coin Hadamards and X wrappers to convert the supplied target condition into
an all-one phase. A clean CCX chain computes the conjunction, Z marks its
last flag and reversed CCX gates uncompute every flag. The coin uses the same
reflection construction on its own register. Each shift explicitly matches a
binary coin label and flips the corresponding position bit.

The usual SKW coin is $G=2|s\rangle\langle s|-I$, with a $-I$ coin block at the
marked position. Consequently this circuit's iteration differs from that
standard step by an overall minus sign. The $(-1)^k$ phase leaves uncontrolled
search probabilities unchanged; phase-correct coherently controlled reuse is
outside the exported circuit contract.

A complete trial measures **every** coin, position and work wire. Decode the
position bits and compare with the supplied mark. Coin outcomes are summed,
and work outcomes are retained. Clean-work leakage is a circuit check, not a
postselection rule. The oracle's intended semantics require initially clean
work, rather than an arbitrary work-register input.

## Small circuit and empirical performance

The prescribed instance has $d=4$, marked vertex $m=5$ (`0101`) and $k=4$
iterations. There are two coin, four position and five work wires: **11 total**.
The following original figure shows uniform preparation and **one** explicit
iteration; the downloadable measured circuit contains all four iterations and
11 final measurements.

![Uniform preparation and one explicit joint-reflection, coin and shift iteration.](../../assets/generated/optimization/walk/one-step-circuit.svg)

An independently constructed $64\times64$ active-space coin matrix and shift
permutation give marked-position probability

$$p_m=25/64=0.390625.$$

Across the saved diagnostic scan from zero through eight iterations, the
largest phase-aligned circuit/reference amplitude discrepancy is about
$1.47\times10^{-15}$, with zero recorded work leakage. Four iterations were
fixed before this scan. More iterations do not monotonically improve success.
A separate $d=2$ four-step check returns marked probability $1/4$.

![Marked probability over a fixed zero-through-eight iteration scan.](../../assets/generated/optimization/walk/probability-scan.svg)

The three saved readouts draw from the checked ideal circuit probabilities,
using NumPy PCG64 and `SeedSequence([seed,5])`. Each uses 4,096 complete
preparations; none is a QPU experiment.

| Seed | Marked count / 4,096 | Marked frequency | Nonzero-work outcomes |
|---:|---:|---:|---:|
| 7 | 1,634 | 0.398926 | 0 |
| 19 | 1,635 | 0.399170 | 0 |
| 41 | 1,599 | 0.390381 | 0 |

For one fixed Bernoulli mean, the 95% Hoeffding half-width is
$\sqrt{\log(40)/(2\times4096)}\simeq0.021220$, below the specified absolute
frequency target 0.025. It is per estimate, rather than simultaneous coverage
of the three seeds. A separate ideal at-least-one-marked-readout task needs
$R=10$ complete independent trials, giving $1-(1-25/64)^{10}\simeq0.992939$.
That task has a different repetition ledger from the 12,288-readout study.

## Important variants and scope

This implementation covers a single supplied equality mark on a binary-coin
hypercube. The [unmarked coined-cycle walk](../quantum-walks.md) and
[continuous-time hypercube walk](continuous-time.md) use different graph,
coin/evolution and output contracts. Multiple marks, general graph access,
Szegedy transition-state preparation, application predicates and a larger
success-scheduled search study remain planned. The original paper's query
bound is external theory, not a scaling result measured by this four-step
example.

## End-to-end resource analysis

### Source size and bounded scaling

Total logical width is $Q=2(d+\log_2d)-1$. For $k$ iterations the source circuit
has $O(kd\log d)$ gates, plus $d+\log_2d$ preparation Hadamards and $Q$ final
measurements. Writing $r=\log_2d$, each joint mark has $2(d+r-1)$ CCX gates.
The shift has $d$ CX for $r=1$, or $d(2r-3)$ CCX for $r\ge2$. The coin has one Z
for $r=1$, one CZ for $r=2$, or $2(r-1)$ CCX for $r\ge3$. All H/X wrappers and
uncomputation are included in the actual export counts.

The main four-step source has **56 CCX**: 40 in marking and 16 in shifts.
Deterministic optimization-level-zero Qiskit lowering yields **392 T/T†**, 340
CX and no arbitrary-angle rotations. Its standard CCX sequence has seven
T/T† gates, so approximate rotation synthesis is unnecessary for this literal
logical circuit. This does not remove physical fault or factory costs.

| Workload | Preparations | Oracle/coin/shift calls, each | Qubit readouts | T/T† gates |
|---|---:|---:|---:|---:|
| One complete trial | 1 | 4 | 11 | 392 |
| Three seeded frequency estimates | 12,288 | 49,152 | 135,168 | 4,816,896 |
| Ideal 0.99 at-least-one task | 10 | 40 | 110 | 3,920 |

The saved larger exports retain four steps and mark 5:

| Dimension $d$ | Vertices $2^d$ | Total wires | Literal T/T† | Simulation status |
|---:|---:|---:|---:|---|
| 4 | 16 | 11 | 392 | Full bounded ideal/reference checks |
| 8 | 256 | 21 | 1,344 | Source and Clifford+T exports only |
| 16 | 65,536 | 39 | 3,472 | Source and Clifford+T exports only |

These counts are generator scaling, not a physical size sweep or a validated
success law for the larger circuits. Supported dimensions are $2,4,8,16$;
iteration counts are capped at 32. The conservative materialization guard is
100,000 primitive operations, charging CCX as 15 and CZ as three. Statevectors
are separately capped at 12 total wires; dense references at six active bits.
A larger generator or a higher caller limit does not bypass those simulation
caps.

### Fixed-workload surface-code analysis

The [saved surface-code/factory report](../../assets/generated/optimization/annealing-walk-resources/report.json)
costs all **12,288 complete preparations** of the fixed $d=4,m=5,k=4$ measured
circuit. Its two walk profiles use physical error $10^{-3}$ or $10^{-4}$,
100 ns gates and 500 ns measurements, with whole-readout physical-fault
allowance $10^{-3}$. It is separate from the ideal ten-trial search output and
from the report's annealing entries.

| Physical error | Saved selection | Peak physical qubits | Serial model time (s) | Whole-readout physical-fault score |
|---:|---|---:|---:|---:|
| $10^{-3}$ | Fewest qubits | 704,817 | 123.801600 | 0.000104727 |
| $10^{-3}$ | Shortest modeled time | 876,993 | 94.089216 | 0.000594491 |
| $10^{-4}$ | Fewest qubits | 40,101 | 64.376832 | 0.000271240 |
| $10^{-4}$ | Shortest modeled time | 47,173 | 44.568576 | 0.000761446 |

The original preprocessing supplies the exact Clifford+T sequence. No-op
compiler fences preserve its literal counts through the shared adapter. The
recorded estimator trace has T count 392, zero arbitrary RZ and 11 measurements,
so it uses zero arbitrary-rotation model allowance. The QDK LogicalCounts trace
abstracts Clifford scheduling and gate topology; the supplied sequence is not
a routed physical implementation {cite:p}`opt_qdk1340`.

:::{warning} Conditional model and exclusions

The complete logical circuit and ledger retain preparation, the explicit
oracle, work cleanup, coin/shift and readout. The table costs their QDK/PSSPC
count abstraction with modeled code distance, factories and reliability;
individual original H/CX operations are not separately charged or routed.
**Physical initialization/reset, routing, communication/feedback, classical
input/compilation/scoring time and hardware reliability certificates are
excluded or unmeasured.** Logical zero-input initialization is assumed.
The factory and code models do not certify hardware success. Peak qubits times
serial runtime is a reported resource product, not a measured active-qubit
integral. No physical size sweep, matched advantage claim or new factory search
is supplied here. The frozen Grover baseline remains unchanged.
:::

The report retains every Pareto point and the
[selection CSV](../../assets/generated/optimization/annealing-walk-resources/summary.csv).
Its original producer/adapter hashes and historical source revision are
preserved; the tables are saved evidence rather than newly regenerated
estimates.

## Classical simulation, verification and formalization

The saved oracle check evaluates all 64 clean-work active basis inputs against
an independently formed rank-one reflection. Tests independently build the
standard SKW coin blocks and shift, check coherent phases and work cleanup,
verify the local seven-T CCX operator, compare the selected full 11-wire
prepared state after lowering, retain every measurement outcome and reject
unsupported sizes. The physical preprocessing does not check the full
$2048\times2048$ walk operator. Numerical tests are bounded evidence, rather
than a theorem for all inputs.

A full statevector stores $2^Q$ complex amplitudes; a dense active-space matrix
stores $4^{d+r}$ entries. Those references are intentionally limited. No Lean
or Rocq proof is integrated for this walk. The separate finite classical C4
optimization proof does not prove its oracle, coin, shift, success or physical
resources.

## Reproduce, code and references

The [existing companion notebook](../../notebooks/optimization_walk.ipynb)
executes independent reference, sampled-readout and exact-lowering checks in a
fresh Python 3.12 kernel. Static GitHub Pages reuse its saved outputs; editable
execution needs a local or Binder kernel.

From the pinned source package, write new results to a separate directory:

```bash
python scripts/generate_optimization_walk.py --output artifacts/skw-check
pytest tests/test_optimization_walk.py
# Larger source and exact Clifford+T exports; no simulation:
python scripts/generate_optimization_walk.py --dimension 8 --marked 5 --steps 4 \
  --export-only --output artifacts/skw-d8
```

The [matching archived implementation](../../assets/generated/optimization/walk/optimization.py),
[standalone generator](../../assets/generated/optimization/walk/generate_optimization_walk.py),
[MIT license](../../assets/generated/optimization/walk/LICENSE.txt) and
[environment lock](../../requirements-lock.txt) reproduce the saved logical
workflow. Keep `optimization.py`, `generate_optimization_walk.py`,
`LICENSE.txt` and the lock together when using individual downloads. The
surface-code reproduction additionally needs the source-checkout shared
adapter; it is not a standalone notebook command.

The prose and circuit figures are original. Qiskit **2.6.2** supplies circuit
construction, exact primitive lowering and ideal statevectors under
[Apache-2.0](https://github.com/Qiskit/qiskit/blob/2.6.2/LICENSE.txt)
{cite:p}`walk_qiskit262`; NumPy **2.5.4** supplies classical matrices and ideal
sampling under BSD-3-Clause {cite:p}`opt_numpy254`. Saved provenance identifies
the actually executed source and package versions, rather than rebinding old
records to later whole-module edits.

- [Main source QASM](../../assets/generated/optimization/walk/walk-source.qasm) and [literal Clifford+T QASM](../../assets/generated/optimization/walk/walk-clifford-t.qasm)
- [Run/reference/readout records](../../assets/generated/optimization/walk/runs.json), [logical ledger](../../assets/generated/optimization/walk/resources.json), [readout CSV](../../assets/generated/optimization/walk/readout.csv)
- [Dimension-eight source](../../assets/generated/optimization/walk/scalability/d8-source.qasm) and [Clifford+T](../../assets/generated/optimization/walk/scalability/d8-clifford-t.qasm)
- [Dimension-sixteen source](../../assets/generated/optimization/walk/scalability/d16-source.qasm) and [Clifford+T](../../assets/generated/optimization/walk/scalability/d16-clifford-t.qasm)
- [Fenced estimator input](../../assets/generated/optimization/annealing-walk-resources/walk-estimated-source.qasm) and [lowered input](../../assets/generated/optimization/annealing-walk-resources/walk-lowered.qasm)

```{bibliography}
```
