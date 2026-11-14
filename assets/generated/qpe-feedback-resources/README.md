# Complete feedback-QPE accounting

These are two bounded, original MIT implementations, n=1 and m=3/4,
phi=.3, U=P(2pi phi), X eigenstate preparation, and15 fresh full runs.
The executable feedback QASM contains actual measurements, all round resets
and conditional Clifford+T corrections. The separately named all-corrections
envelope is accounting input and must not be run as feedback QPE.

Both variants use the same serialized surface-code instrument model and
hardware scenario: p=1e-4,100ns gates/init,500ns measurements,1000ns code
cycles,10000ns controller reaction,1e-15 controller fault per decision.
The existing static QDK mean-fit baseline and all earlier artifacts are
preserved. Every full run initializes every wire and repeats actual supplied
controlled U calls; workspace startup, target idling, preparation, readout,
reset, local Clifford frames, neighbor routing, T transport/injection and
capped factory retries are charged. Each predicate, even false, adds one
controller slot and whole-device idle exposure. A common distance/factory/
retry cap covers every allowed transcript. False bodies pad their reserved
slot; expected wall time is not presented as a lower resource estimate.

Scalar medians need phi in[2^-m,1-2^-m]. The ideal Fourier/majority argument,
actual guarded numerical Ross-Selinger synthesis, modeled physical faults
and factory availability are distinct. No floating check certifies hardware
confidence, a decoder, a lattice layout or a formal synthesis interval.
The exact sequences, hashes, source controls, all branch ledgers, failure
components and complete Pareto points are retained in the task reports.

Offline synthesis time is separately measured. The15-sample scalar median
is decoded by a fixed105-comparator m-bit network and charged once after all
runs:15*m bit-decoding slots,105*(m+3) comparison/write slots and1 selection
slot. Each uses the controller reaction/fault scenario; all quantum patches
remain charged during that time. Compilation and presentation-only final
reporting time are excluded from quantum wall time. Geometry,
tracked Pauli bases, factory/data handoff and encoder/readout/controller
reliability retain the explicit conditional assumptions of instrument_resources.
Classical record/logic reliability is part of the per-decision controller
scenario; independent idle classical-memory calibration and classical
controller transistor area are not supplied or included in physical qubits.
Conditional-body overall phases are discarded only across measured classical
branches; controlled source global phase is preserved. QASM exports are
complete measured instruments, not reusable coherent oracle specifications.

The logical-size file generates active phase-product access at n=1,2,4,8;
those extra widths were not simulated/synthesized/physically estimated.
Scalable generation retains the independent qft_qpe.py caps of64 wires,
16 phase bits,200000 source operations and1000000 decomposed operations.
The new physical/simulation adapter is limited to two feedback wires,
at most4 record bits,100000 total validation operations and256 branches.
Only previously measured one-bit, local one-arm corrections are supported.

Reproduce in the repository with the locked Python3.12.14 environment:
python scripts/generate_qpe_feedback_resources.py
pytest tests/test_qpe_feedback_resources.py
The producer's four new native queries search odd distances3..15. Keep the
private factory cache outside the public output. No existing producer is run.

Qiskit2.6.2 is Apache-2.0; rsgridsynth0.2.2 and QDK1.34.0 are MIT.
Exact upstream identities and the installed-wheel limitation appear in
synthesis/physical reports. references.bib and primitives.bib credit the
semiclassical QFT, QPE, Ross-Selinger and surface-code sources. All prose and
circuits here are original; LICENSE.txt covers our implementation.
