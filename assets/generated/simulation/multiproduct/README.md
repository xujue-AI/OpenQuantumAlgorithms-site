# Fixed two-term multiproduct implementation

This original MIT implementation uses NumPy, SciPy and pinned Qiskit2.6.2.
Install requirements-lock.txt in a Python3.12.14 virtual environment. Run
`python generate_simulation_multiproduct.py --output-dir NEW_DIRECTORY`.
The generator refuses existing directories. A large circuit uses `--qubits 32
--steps 1 --no-sim --no-figures`; bounded numerical references stop at 3 spins.

Complete measured-trial QASMs contain literal primitive access, input resets
and readouts. State-instrument QASMs leave the system unmeasured; the separate
opposite-identity counterexamples are unitary fixtures.
Nonzero circuit phases are retained through explicit OpenQASM3 gphase.
LCU/OAA postselection, unconditional discarded-ancilla OAA, and classical
weighted means are different outputs. Both SELECT branches, padding,
PREPARE/uncomputation, reflections, work, repetitions and readout are counted.
The opposite-identity fixture is outside the common-H Strang family and
retains OAA's possible acceptance decrease without claiming state failure.

The grid checks implementation on smallinputs. Fixed-H,t fourthorder is an
asymptotic statement; the concrete commutator certificate is conservative
secondorder. Finite counts and proposed synthesis/physical allowances are
not executed physical estimates or hardware/accuracy certificates. Projected
sampling and rejection populations are unexecuted. No formal proof is built.

References: Low etal1907.11679v2; CarreraVazquez etalQuantum7,1067,
DOI10.22331/q-2023-07-25-1067; MizutaQuantum10,1974,
DOI10.22331/q-2026-01-19-1974. Upstream Qiskit is Apache2.0.
