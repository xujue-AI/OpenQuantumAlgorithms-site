---
title: "Phase estimation with Hamiltonian access"
description: "A checked two-spin controlled propagator, retained-target QPE instruments and fresh interference experiments."
downloads:
  - file: ../../assets/generated/qpe-application-tiny/kernel.qasm
    title: "Three-wire controlled Hamiltonian circuit"
  - file: ../../assets/generated/qpe-application-tiny/checks.json
    title: "All 15 saved tiny checks"
  - file: ../../assets/generated/qpe-application-tiny/src/openquantum/application_qpe_tiny.py
    title: "Original circuit generators"
  - file: ../../assets/generated/qpe-application-tiny/src/openquantum/application_qpe_tiny_reference.py
    title: "Independent mathematical references"
  - file: ../../assets/generated/qpe-application-tiny/scripts/check_qpe_application_tiny.py
    title: "Exact recorded checker source"
  - file: ../../assets/generated/qpe-application-tiny/requirements-lock.txt
    title: "Pinned Python dependencies"
---

## Motivation and intuition

An energy becomes a phase through Hamiltonian evolution. A useful QPE example
must construct that controlled evolution and specify the input state, because
the phase distribution depends on both {cite:p}`prim_qpe_1998,energy_eigenvalues_1999`.
This companion replaces the known phase-gate example with an explicit
noncommuting two-spin Hamiltonian. It also shows why keeping the target between
measurements and preparing it afresh produce different joint records.

:::{note} Checked scope
The saved study checks a full controlled three-wire kernel, two-bit/two-round
QPE instruments and six fresh interference experiments. Its **15 tiny ideal
circuit/instrument cases** passed numerical reference checks. It supplies no
ground-energy output or confidence experiment, compiled rotation sequence,
physical estimate or formal proof. The model is analytically solvable and is a
correctness example, with no application-advantage or chemical-accuracy claim.
:::

## Fundamental model and phase convention

With $\hbar=1$, use

$$H=-Z_0-Z_1+\tfrac14X_0X_1,\qquad |\psi_{\rm in}\rangle=|00\rangle.$$

The input circuits actually reset both data wires. They load no ground
eigenvector. Data index is $q_0+2q_1$. The even and odd blocks in bases
$(00,11)$ and $(01,10)$ are

$$H_{\rm even}=\begin{pmatrix}-2&1/4\\1/4&2\end{pmatrix},\qquad
H_{\rm odd}=\begin{pmatrix}0&1/4\\1/4&0\end{pmatrix}.$$

Writing $R=\sqrt{65}/4$, the spectrum is $(-R,-1/4,1/4,R)$. The reset input's
ground weight is $w_0=(1+8/\sqrt{65})/2$; its remaining weight is
$q=1-w_0<1/256$. This two-spin overlap does not establish the overlap of a
larger Hamiltonian family.

All circuits use $\tau_0=\pi/3$, reference energy 3 and spectral width 6:

$$U=e^{i3\tau_0}e^{-iH\tau_0}=-e^{-iH\tau_0},\qquad
\phi(E)=\frac{3-E}{6},\qquad E=3-6\phi.$$

The eigenphases lie inside $[1/8,7/8]$. The scalar minus sign becomes an
essential **Z on the phase control**. Dropping it changes the decoded ground
energy by three energy units. The full controlled comparison retains that
phase and performs no global-phase fit.

## Actual controlled evolution and small circuit

A CX from data wire 0 to data wire 1 stores parity and transforms H into
$-(1+Z_p)Z_d+X_d/4$. Set $\theta=\arctan(1/8)$. Chronological even-sector gates
$R_Y(\theta),R_Z(-2R\tau_0),R_Y(-\theta)$ implement its evolution; the odd sector
uses $H,R_Z(\tau_0/2),H$. Each selected rotation is controlled by the phase bit
and parity, using four quarter-angle rotations and four CX gates. An X pair
opens the even-parity control; the parity transformation is undone.

The [actual kernel](../../assets/generated/qpe-application-tiny/kernel.qasm)
contains **16 rotations +18 CX +2 H +2 X +1 Z =39 primitive instructions**.
Its wires are phase control `q[0]`, data 0 `q[1]` and data 1 `q[2]`.

![All 39 recorded gates on phase-control q0 and data q1/q2, in four ordered panels](../../assets/figures/qpe-application-tiny-kernel.svg)

The figure shows every literal gate in recorded order. Read each panel left
to right, then continue downward. Signed angle aliases match the
[full QASM export](../../assets/generated/qpe-application-tiny/kernel.qasm);
the final control Z is shown individually. This coherent kernel includes no
reset or readout; those appear in the complete trial circuits below.

The following is the recorded three-wire circuit source; rotations are unsynthesized.

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
cx q[1], q[2];
x q[2];
ry(0.03108874863669036) q[1];
cx q[2], q[1];
ry(-0.03108874863669036) q[1];
cx q[0], q[1];
ry(0.03108874863669036) q[1];
cx q[2], q[1];
ry(-0.03108874863669036) q[1];
cx q[0], q[1];
rz(-1.0553470713917545) q[1];
cx q[2], q[1];
rz(1.0553470713917545) q[1];
cx q[0], q[1];
rz(-1.0553470713917545) q[1];
cx q[2], q[1];
rz(1.0553470713917545) q[1];
cx q[0], q[1];
ry(-0.03108874863669036) q[1];
cx q[2], q[1];
ry(0.03108874863669036) q[1];
cx q[0], q[1];
ry(-0.03108874863669036) q[1];
cx q[2], q[1];
ry(0.03108874863669036) q[1];
cx q[0], q[1];
x q[2];
h q[1];
rz(pi/24) q[1];
cx q[2], q[1];
rz(-pi/24) q[1];
cx q[0], q[1];
rz(pi/24) q[1];
cx q[2], q[1];
rz(-pi/24) q[1];
cx q[0], q[1];
h q[1];
cx q[1], q[2];
z q[0];
```

Every power calls this entire base kernel literally. Direct longer-time block
formulas appear only in the independent reference. The construction uses this
special two-spin structure; it is not generic Hamiltonian fast-forwarding.

## Retained targets, feedback and fresh experiments

Let $P_j(y)$ be the two-bit QPE word law for eigencomponent j. Two rounds on
one retained target have

$$J(y_0,y_1)=\sum_j w_jP_j(y_0)P_j(y_1).$$

The latent eigenvalue is common to the rounds. Resetting the target between
rounds instead gives $J_{\rm fresh}(y_0,y_1)=p(y_0)p(y_1)$ with
$p(y)=\sum_jw_jP_j(y)$. The saved target-reset negative exposes this difference;
repeated rounds are not independent fresh draws from the marginal mixture.

The deferred six-wire circuit, four-wire clock-reuse circuit and three-wire
feedback circuit implement the same retained law. Feedback measures $U^2$
first as the low bit, applies $S^\dagger$ if that bit is one, and then measures
the $U^1$ high bit {cite:p}`prim_semiclassical_1996`. History restarts for the
next word; the data stay retained. All four joint outputs store 16 probabilities
in order $y_0+4y_1$, where $y_0=c_0+2c_1$ and $y_1=c_2+2c_3$.

Fresh Hadamard experiments reset all three wires for each trial and measure
cosine/sine fringes at powers 1, 2 and 4. Their signal is
$\langle00|U^k|00\rangle$, a spectral mixture. Its probability difference from
the ground-component fringe is at most q. The six checked fringes fit this
tiny-model contamination diagnostic; arbitrary mixed inputs need another
certificate before using [bounded-bias robust inference](robust.md).

## Recorded results and classical verification

The [complete saved checks](../../assets/generated/qpe-application-tiny/checks.json)
retain all 15 cases. Independent original-basis block matrices check the full
8-by-8 controlled operator, including control-zero identity and the odd sector
that has zero weight in the reset input. Unnormalized density maps evaluate
actual resets, readouts and conditional gates.

The fresh circuits give the following zero-outcome probabilities, rounded for
display. They are ideal circuit/projection results; no shots were sampled.

| Literal power k | Cosine Pr(0) | Sine Pr(0) |
|---:|---:|---:|
| 1 | 0.757024172428 | 0.074431262238 |
| 2 | 0.264245700849 | 0.062474189462 |
| 4 | 0.222320358274 | 0.912594363296 |

| Checked comparison | Largest recorded entry/probability error |
|---|---:|
| Full controlled powers 1, 2 and 4 | $2.4432\times10^{-15}$ |
| Four joint instruments against their respective laws | $3.3307\times10^{-16}$ |
| Six fresh zero-outcome probabilities | $2.2204\times10^{-16}$ |

All are inside the declared numerical-entry tolerance $2\times10^{-12}$.
The [missing-Z negative](../../assets/generated/qpe-application-tiny/kernel-without-reference-z.qasm)
changes a full-operator entry by about 1.989. The
[missing-odd-evolution negative](../../assets/generated/qpe-application-tiny/kernel-without-odd-evolution.qasm)
changes the odd block by about 0.2588 while preserving the even block. The
retained/reprepared joint laws differ by about 0.0015319. These controls exercise
errors that an input-only evolution check could miss.

The checked domain stops at **6 quantum wires, 16 records and power 4**, with
density dimension at most 64. The largest recorded validation count is 1248
applications under the 10000-per-case cap. That validator counter charges
every retained record, explicit ket-column traversal, reset Kraus map,
measurement projection and feedback predicate; it measures verification work,
separately from the algorithm's source instructions below.

## End-to-end accounting and planned fault tolerance

Every joint circuit uses six literal controlled base calls. Its complete source
includes the actual preparation/reset and readout instructions:

| Reduced source | Quantum wires | Unitary instructions, upper bound | Resets | Readouts | Feedback decisions |
|---|---:|---:|---:|---:|---:|
| [Deferred clocks](../../assets/generated/qpe-application-tiny/retained-deferred.qasm) | 6 | 258 | 6 | 4 | 0 |
| [Reused clocks](../../assets/generated/qpe-application-tiny/retained-reused.qasm) | 4 | 258 | 6 | 4 | 0 |
| [Feedback](../../assets/generated/qpe-application-tiny/feedback.qasm) | 3 | 244 | 6 | 4 | 2 |
| [Reprepared negative](../../assets/generated/qpe-application-tiny/retained-reprepared.qasm) | 4 | 258 | 8 | 4 | 0 |

Feedback counts both possible corrections in its source upper envelope. A
fresh cosine trial uses $39k+6$ source instructions, including three resets
and one readout; the sine trial adds one $S^\dagger$. These are source counts,
with continuous rotations. They are not T counts, physical cycles or a matched
fault-tolerant comparison.

For supplied n-wire access of cost $C_U^{(c)}(n)$, R retained QPE rounds require
$R(2^m-1)C_U^{(c)}(n)+O(Rm^2)$ gate work, plus preparation, resets and readout.
Deferred/reused/feedback clock widths are respectively $n+Rm$, $n+m$ and $n+1$.
Fresh experiments additionally charge every preparation and shot. This
two-spin study provides no validated scaling in n or general-Hamiltonian
generator. Larger access, overlap, alias windows and confidence must be supplied
for each family.

The full six-bit/41-round/two-group output and its energy-confidence target
remain planned. Synthesis/angle certificates, full repetitions, classical
inference, storage/controller latency, physical faults, routing and factory
availability must all be priced before an end-to-end physical claim. The
surface-code baseline is unchanged. No Lean or other formal theorem has been
reproduced for this implementation.

## Sources, reproducibility and evidence limits

The [public packet](../../assets/generated/qpe-application-tiny/README.md) supplies
the exact circuit/reference/checker sources, all 14 saved payloads, license,
lock and filename/hash joins. Original MIT code uses Qiskit **2.6.2**
(Apache-2.0) and NumPy **2.5.4** (BSD-3-Clause) {cite:p}`prim_qiskit_2_6_2`.
The independent references use block formulas and finite Fourier sums rather
than importing the circuit constructor. The packet README documents a manual
reproduction recipe using the exact checker and a fresh output directory. That
recipe was not executed during this projection; a replay notebook remains planned.

The saved checks SHA-256 is
`584640ed31039646b0764d5f325129f31371d7877b40915b73ff686bfbcca0b3`.
Raw files preserve authoring-era `private`/`unexecuted` wording, including
embedded `source.scope` labels. Those historical labels are not the status of
the completed tiny check and have not been silently rewritten. All saved final
record probabilities are nonzero: complete records are demonstrated here,
while exact-zero final-branch preservation remains a structural source property.
Numerical agreement supplies fixed-instance evidence, with full application
confidence, synthesis, physical resources and formal verification still planned.
