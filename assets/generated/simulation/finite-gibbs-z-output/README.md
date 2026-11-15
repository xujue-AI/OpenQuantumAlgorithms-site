# Fixed finite Gibbs Z-mean example

This packet presents the separately accepted **fixed, conditional ideal Z-mean
contract** for one spin. It contains original scientific sources, the exact
accepted primitive QASM exports and selected saved scientific evidence. The
public packaging and portable reproduction wrapper were authored and reviewed
as source only; they have not been executed as a downloaded package.

Use H=Z with coefficient one, beta=1, proposals X/sqrt(2) and Z/sqrt(2),
four clock bits (M=16), one product step, delta=1/8 and R=64. Resetting the
system and applying H prepares the known plus input. The output is the mean
of a final Z readout, compared with the analytic Gibbs target -tanh(1).
The exact diagonal-H/Pauli monomial restriction makes the coherent correction
C zero. All 32 nonjump and 32 jump Kraus blocks contribute; there is no
success postselection or deletion of plus-state coherences.

## Saved scientific contract

The saved interval and angle ledger, with their stated mathematical
assumptions, give the following upper endpoints. These values are copied from
the accepted saved evidence, without recalculation; full decimal intervals
are retained in [results.json](results.json).

| Contribution | Saved upper endpoint | Allowance |
|---|---:|---:|
| Complete 128A primitive-angle bridge | 9.4087761533880019e-13 | .002 |
| Finite stationary Z bias | .0014683286051066204 | .004 |
| Weak digital transient | .0013569783582174397 | .002 |
| Finite-time transient | .0091965252906841105 | .012 |
| Preparation total | .012021832254949048 | .020 |
| Preparation plus projected .030 measurement error | .042021832254949048 | .050 |
| Projected Hoeffding failure | .0012562140299529949 | .050 |

The measurement projection assumes 16,384 fresh independent complete
preparations followed by ideal Born Z readouts. **No shots were drawn.**
This fixed conditional observable contract establishes neither full-state
Gibbs accuracy nor a general thermal sampler, mixing theorem, uniform
continuum/KMS bridge, floating-backend error certificate, rotation synthesis,
physical resources or mechanized proof.

The original result status remains
`observed_fixed_finite_Gibbs_Z_components`, with
`finite_gibbs_Z_output_bound_observed=true`,
`finite_gibbs_Z_output_bound_passed=false` and
`whole_invocation_validated=false`. Broad implementation and certification
flags remain false. The projection records the separate fixed acceptance
alongside the saved flags; it does not replace them.

## Circuit and bounded validation work

The [original illustration](finite-gibbs-z-circuit.svg) groups one round;
the QASM files below retain every primitive and numeric literal.

| Exact accepted export | Primitive instances |
|---|---:|
| [access-U.qasm](circuits/access-U.qasm) | 140 |
| [weak-dilation.qasm](circuits/weak-dilation.qasm) | 1,493 |
| [reset-round.qasm](circuits/reset-round.qasm) | 1,505 |
| [preparation-R64.qasm](circuits/preparation-R64.qasm) | 96,323 |
| [preparation-R2.qasm](circuits/preparation-R2.qasm) | 3,013 |

Seven wires are reserved: system q0, clock q1..q4, proposal q5 and weak flag
q6. Each round resets all six auxiliaries before and after the complete
dilation; q0 stays live. Initial system reset/H and final Z measurement are
included in both preparation counts. The 64-round stream was emitted and
authenticated, **not evolved as a full primitive stream**.

The accepted check performed two seven-wire Statevector evolutions, one
seven-wire R2 density evolution, two exponent-64 powers of 4-by-4 channels,
one independent full-U/all-64-Kraus reference construction and one fixed
interval certificate evaluation. The density derivative removes only the
terminal Z measurement from R2, leaving 3,012 primitives and 25 resets.
The actual/reference matrices, all Kraus blocks, diagnostics and predicates
are retained in the scientific projection. Floating agreement is distinct
from the interval budget.

The projected readout workload is `16384*(64*1505+3)` primitive instances.
This is source-operation accounting. Classical preprocessing and readout
bookkeeping, arbitrary-angle synthesis, routing, idle/controller work,
physical faults and hardware timing remain unpriced. The frozen surface-code
baseline is unchanged; no physical comparison is made.

## Sources, evidence and reproduction

[REPRODUCTION.md](REPRODUCTION.md) gives the locked Python 3.12 recipe and
bounded portable driver. Its default input is the exact accepted QASM, so it
does not regenerate platform-dependent angles. A future run writes a new
result into a separate absent output directory and cannot extend the original
saved acceptance automatically.

Save the individually linked files in the displayed relative layout. The
five `circuits/` exports above, the six `sources/` files below, the lock and
license form the runnable source closure.

| Scientific source | Role |
|---|---|
| [sources/detailed_balance_access.py](sources/detailed_balance_access.py) | Original access generator; separate materialization/reference guards |
| [sources/detailed_balance_channel.py](sources/detailed_balance_channel.py) | Original weak-channel and preparation generator |
| [sources/finite_gibbs_interval.py](sources/finite_gibbs_interval.py) | Accepted directed scalar arithmetic and fixed budget definitions |
| [sources/finite_gibbs_angle_provenance.py](sources/finite_gibbs_angle_provenance.py) | Accepted complete symbolic primitive-angle ledger |
| [sources/finite_gibbs_reference.py](sources/finite_gibbs_reference.py) | Four verbatim accepted scientific reference definitions in a portable module |
| [sources/reproduce.py](sources/reproduce.py) | Newly authored, unexecuted portable wrapper |

[source-derivations.json](source-derivations.json) distinguishes unchanged
source copies from neutral public header edits and the newly authored
wrapper/reference extraction. Executable scientific bodies of the four
original helper modules are preserved. The complete private runtime driver
is not part of this scientific packet.

[summary.json](summary.json) supplies a small reading view;
[results.json](results.json) is an explicit scientific-field projection of
the accepted saved result. It retains scientific interval, angle, domain,
budget, matrix and diagnostic data while omitting operational records and
machine paths. [manifest.json](manifest.json) authenticates the public file
closure and records the accepted scientific source/result identities.

The circuit code, finite reference, explanation and illustration are original
Open Quantum Algorithms work under [LICENSE.txt](LICENSE.txt).
[UPSTREAM-LICENSES.md](UPSTREAM-LICENSES.md) credits exact package versions
and licenses; [references.bib](references.bib) retains the existing primary
paper and software citations. No paper figure, survey text or upstream
benchmark implementation was copied.
