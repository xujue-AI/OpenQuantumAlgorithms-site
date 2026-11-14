# Original measured-AND arithmetic

The small circuits and source counts retain initialization, helper measurement, conditional CZ/X clearing and final readout. Literal T savings are not physical speedup. The unoptimized MAJ/UMA adder has 8w measured T gates versus 14w static T gates; this is not Gidney's optimized 4w-4 adder.

Large exports are circuit materializations, not validated large quantum runs. Exact sparse checks use the intended dyadic pi/8 angle model, not a formal proof or a certified floating/synthesis enclosure.

Use the full public code package for the src/openquantum and scripts layout. Flat snapshots here identify exact sources; they are not an automatically installed package. No private controller, approvals, logs or credentials are included.

References: Gidney, Quantum 2,74 (2018), doi:10.22331/q-2018-06-18-74. Qiskit2.6.2 (Apache-2.0) supplies standard gates; all wrapper code, explanations and figures are original MIT work.
