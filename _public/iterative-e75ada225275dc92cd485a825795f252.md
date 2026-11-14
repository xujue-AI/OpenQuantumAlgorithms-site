---
title: "Iterative phase estimation with feedback"
description: "One reusable phase ancilla with actual measurement, reset and conditional phase correction."
downloads:
  - file: ../../notebooks/qpe_iterative.ipynb
    title: "Executable feedback-QPE notebook"
  - file: ../../notebooks/qpe_feedback_resources.ipynb
    title: "Complete static and feedback QPE accounting"
---

## Motivation and fundamental version

The phase register can be measured one bit at a time when its inverse Fourier transform is immediately followed by readout. Earlier outcomes then select later single-qubit corrections, replacing coherent Fourier interactions with classical feedback {cite:p}`prim_semiclassical_1996`.

Our explicit circuit uses $n+1$ quantum wires and $m$ classical output bits. In round $r=0,\ldots,m-1$ it resets the phase ancilla, applies H and $2^{m-r-1}$ controlled calls to $U$, applies corrections selected by previous recorded bits, applies H and measures into `c[r]`. A previous `c[l]=1` triggers

$$
P\!\left(-\frac{2\pi}{2^{r-l+1}}\right),\qquad l<r.
$$

The least-significant output bit is obtained first. Displaying `c[m-1]...c[0]` gives the usual binary integer $y$ and phase $y/2^m$. The eigenstate is prepared once and retained between rounds in the ideal instrument. For a superposition of eigenstates, the retained system and record become correlated; the checked final phase distribution is the expected mixture.

## Small circuit and independent simulation

```{embed} #qpe-iterative-circuit
```

```{embed} #qpe-iterative-results
```

The default is $U=P(2\pi\times0.3)$, prepared $\lvert1\rangle$, and five output bits. The actual `if_else`, measurement and reset instrument is evaluated by bounded branch enumeration: project the measured wire, weight each outcome, apply its selected correction and continue. It is not replaced by the known-phase distribution formula. The result matches conventional QPE and the independent Dirichlet reference within $2.4\times10^{-15}$.

## End-to-end space and time

The quantum width falls from $n+m$ to $n+1$ while retaining $m$ classical bits. Each complete execution still applies $2^m-1$ controlled $U$ calls, $2m$ H gates, $m$ reset instructions, $m$ measurements and up to $m(m-1)/2$ selected correction gates. Fifteen independent runs of the default use 465 controlled calls, 75 measurements and 75 resets, with fresh preparation for each run.

The source ledger separates unconditional primitive operations from conditional blocks. A conditional gate is charged on the corresponding branch; recording a worst-case branch bound does not produce an average latency model. The executed circuit includes reset even though an ideal measured eigenstate can allow equivalent feedback initialization strategies. Reset faults and classical feed-forward delays are not silently omitted.

## Fault-tolerant analysis and verification

**Complete physical accounting has executed for the matched three/four-bit workload below.** It uses actual conditional Clifford+T corrections and one conservatively provisioned schedule for every branch. The original five-bit example remains an ideal instrument check. The parent chapter's static mean-fit QPE numbers are separate records.

### Matched complete task and common feedback schedule

The comparison fixes $U=P(2\pi\times0.3)$, an X-prepared $|1\rangle$ target,
and $m=3$ or $4$. Each variant repeats its entire initialized, prepared and
measured circuit 15 times, then returns the scalar median of $y/2^m$, with
absolute phase error target $2^{-m}$. The promise
$2^{-m}\le\phi\le1-2^{-m}$ keeps the good interval away from the branch cut.
The usual ideal single-run probability bound $8/\pi^2$ and its 15-run
majority tail are separate from synthesis and physical faults
{cite:p}`prim_qpe_1998,prim_semiclassical_1996`.

Both variants explicitly reset every input wire before preparation and retain
all $2^m-1$ complete controlled calls. Feedback also retains all $m$ round
resets, $m$ readouts and $m(m-1)/2$ predicates. The target remains protected
during reset, readout, feedback, factory supply and routing time. Startup
workspace is initialized for every complete run.

```{embed} #qpe-feedback-physical-circuit
```

The compiler gives each local conditional correction an independent slot,
generates an actual ancilla-free Ross–Selinger Clifford+T sequence and
reconstructs its original `if_else`. It preserves the fixed two-wire layout
and every reset/readout boundary. The executable feedback circuit is distinct
from the separately labelled **all-corrections accounting envelope**. This
envelope charges every possible correction and provisions one fixed code
distance, factory configuration and retry cap before any result is observed.
False predicates still consume a controller decision, and skipped correction
slots retain their full reserved idle time. No branch chooses another machine
or supplies a lower expected-runtime claim {cite:p}`resource_ross_selinger_2014`.

The physical scenario uses the existing serialized surface-code instrument
model: physical error $10^{-4}$, 100 ns gates and initialization, 500 ns
measurements, 1,000 ns code cycles and 10,000 ns controller reactions.
Concrete synthesis receives a guarded numerical operator allowance
$2\times10^{-5}$ per complete run. Physical faults and capped factory
availability each receive a separate $0.0025$ allowance over all 15 runs.
The recorded ideal majority tail, $15$ times the synthesis allowance, and
these two fault allowances sum below $0.01$ under the stated assumptions
{cite:p}`resource_litinski_2019,resource_horsman_2012,resource_qdk_134`.

Neighbor routing, protected parity slots, magic-state transport/injection,
initialization, reset, measurement, every controller decision and capped
complete factory retries are charged. Early factory success injects
immediately and pads the remaining supply slot afterward, rather than holding
a magic state for an uncharged interval. Factory output quality and independent
pipeline availability are distinct terms. The code-distance domain is odd
distances 3 through 15, with one just-in-time factory.

Classical decoding is included once after all runs. It stores $15m$ output
bits and applies a fixed 105-comparator network: $15m$ bit-decoding slots,
$105(m+3)$ comparison/write slots and one median-selection slot. Each slot
uses the declared controller reaction and fault scenario; all reserved patches
remain charged during this pass. These are controller-ISA scenario costs,
not calibrated hardware latencies. The model assumes classical record storage
and Boolean logic reliability are covered by its per-decision controller
scenario. Separate idle classical-memory calibration and controller transistor
area are unavailable.

### Executed comparison and limits

The original source instruments agree with the independent Fourier law within
$1.8\times10^{-15}$. Actual synthesized feedback differs from its ideal source
distribution by total variation $1.20\times10^{-7}$ at three bits and
$7.20\times10^{-7}$ at four bits, below their respective guarded numerical
allocations. Removing the conditional corrections changes a probability by
about $0.373$ and $0.691$, demonstrating that the branch behavior is exercised.

```{embed} #qpe-feedback-physical-probabilities
```

The following table selects the smallest computed physical-qubit point for
each task and variant. All four selected points use distance 11, the same
4,820-qubit factory configuration and a four-pipeline-attempt cap per required
state. Every feedback transcript uses that provisioned machine and its padded
schedule. The full reports retain the other Pareto point and all failure terms.

| Phase bits $m$ | Variant | Actual T/T-dagger per run, upper bound | Peak physical qubits | Capped time, 15 full runs and median |
|---:|---|---:|---:|---:|
| 3 | Textbook | 4,074 | 8,435 | 165.908226 s |
| 3 | Feedback | 2,954 | 6,507 | 90.592059 s |
| 4 | Textbook | 8,958 | 9,399 | 394.159777 s |
| 4 | Feedback | 6,742 | 6,507 | 205.908351 s |

These are worst-case reserved quantum/controller times under the model,
including every capped factory supply slot and the final median pass. They
are calculated workloads, not elapsed hardware execution. The selected full
physical-fault terms range from about $6.81\times10^{-5}$ to $7.64\times10^{-4}$,
and factory-availability terms from $4.431\times10^{-4}$ to
$1.3437\times10^{-3}$, each within its separate $0.0025$ allowance. The
declared ideal-statistical, synthesis, physical and availability allowances
sum to about $0.00827$; floating checks do not certify that confidence on hardware.

The dedicated physical adapter supports one target and three/four phase bits,
five total validation wires, 100,000 expanded validation operations and 256
branches. It never prunes a computed nonzero branch. All 8 or 16 allowed
transcripts are covered by the accounting envelope, including records with
zero ideal probability. Arbitrary branching, multiwire corrections, adaptive
stopping and target replacement remain unsupported. The separate
[logical size ledger](../../assets/generated/qpe-feedback-resources/logical-size-scaling.json)
uses active phase-product circuits at $n=1,2,4,8$ and $m=3$; the extra widths
were generated but not simulated, synthesized or physically estimated here.

:::{warning} Conditional engineering comparison

Guarded floating matrix checks and small instrument enumeration are numerical
evidence, not a formal synthesis interval or hardware confidence certificate.
Encoder, readout, controller and code-exposure fits are declared scenarios.
Tracked Pauli bases, lattice geometry, decoder throughput and factory/data
handoff remain conditional assumptions. Offline compilation and
presentation-only final reporting time are excluded. Physical qubit counts
cover quantum patches and factories. This separate comparison preserves all
earlier static QPE projections and the frozen Grover baseline.
:::

The tested feedback distributions cover grid and off-grid phases, all tested output-bit sizes through five, global phases, a Y eigenstate and non-eigenstate mixtures. Primitive lowering preserves tested instrument probabilities. Independent review additionally checked entangled targets and conjugated multi-qubit eigenstates. The instrument checker is capped at eight quantum/classical bits, 4,096 top-level operations and 1,024 branches. It drops numerically zero branches below $10^{-15}$ and checks remaining total probability. These are bounded ideal numerical checks, not a physical simulator or a Lean theorem.

The scalar-median confidence contract and its phase branch-cut assumption are [defined in the parent page](../qft-qpe.md#fundamental-algorithm-and-task-contract). No noisy-hardware inference or circular median is claimed.

## Run and provenance

```python
from openquantum.qft_qpe import phase_gate_example, build_iterative_qpe, dynamic_distribution
unitary, preparation = phase_gate_example(0.3)
circuit = build_iterative_qpe(5, unitary, preparation)
probabilities = dynamic_distribution(circuit)
```

[Executed notebook](../../notebooks/qpe_iterative.ipynb), [lowered dynamic QASM](../../assets/generated/qft-qpe/iterative-qpe.qasm), [source snapshot](../../assets/generated/qft-qpe/qft_qpe.py), [results/ledgers](../../assets/generated/qft-qpe/results.json) and [MIT implementation license](../../assets/generated/qft-qpe/LICENSE.txt) share the pinned Qiskit 2.6.2 environment {cite:p}`prim_qiskit_2_6_2`. The measured/reset QASM is an instrument, not a coherent unitary oracle. Coherent use of a unitary subroutine requires its unmeasured source circuit with global phase preserved.

For the new physical comparison, use the [executed accounting notebook](../../notebooks/qpe_feedback_resources.ipynb),
[three-bit complete report](../../assets/generated/qpe-feedback-resources/n1-m3/report.json),
[four-bit complete report](../../assets/generated/qpe-feedback-resources/n1-m4/report.json),
[actual conditional Clifford+T QASM](../../assets/generated/qpe-feedback-resources/n1-m3/feedback-clifford-t.qasm),
[adapter source](../../assets/generated/qpe-feedback-resources/qpe_feedback_resources.py),
[provenance manifest](../../assets/generated/qpe-feedback-resources/manifest.json) and
[MIT license](../../assets/generated/qpe-feedback-resources/LICENSE.txt).
The [original executed notebook snapshot](../../assets/generated/qpe-feedback-validation/qpe_feedback_resources.ipynb) preserves the fixed study manifest. The routine adapter authenticates refreshed notebook sources and outputs against current package provenance, then checks the unchanged physical records with that original snapshot.

The reports credit Qiskit 2.6.2, its pinned rsgridsynth 0.2.2 dependency and
Microsoft QDK 1.34.0, including the installed-wheel identity limitation.

```bash
# Bounded new synthesis/physical computation; keep the private cache separate:
python scripts/generate_qpe_feedback_resources.py
# Routine retained-record verification; no synthesis or factory queries:
python scripts/verify_qpe_feedback_public.py
```
