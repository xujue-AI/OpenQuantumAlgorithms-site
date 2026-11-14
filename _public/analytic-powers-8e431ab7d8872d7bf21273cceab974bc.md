---
title: "Analytic powers of dimer evolution"
description: "Explicit long-time dimer gates at matched energy precision and success assumptions."
---

## Motivation and fundamental variant

The [baseline energy protocol](../energy-estimation.md) implements $U^{2^j}$ by repeated U calls. Our specified dimer Hamiltonian also has an exact gate formula at any allowed time. Replacing the evolution time $t$ with $2^j t$ therefore constructs the power directly, including the controlled identity/shift phase. The code never inserts a matrix-power gate.

`build_energy_unitary(...,power=2**j)` builds the actual Pauli/parity rotations. `build_retained_target_qpe(...,analytic_powers=True)` uses one such controlled power circuit per clock bit. This fast-forwarding is specific to the analytically accessible independent-dimer family. It does not give a constant-cost power for an arbitrary molecular Hamiltonian.

## End-to-end resource analysis

The task, HF preparation, encoding, $m$, $R$, $K$, readout and error allocations stay fixed. Repeated access has $KR(2^m-1)$ actual base calls; direct powers have $KRm$ actual power circuits. The sum of simulated base periods remains $KR(2^m-1)$ and must not be confused with gate calls or hardware elapsed time.

With the defaults $m=5$, $R=19$ and $K=2$, these counts are 1,178 versus 190. Fresh-clock width stays 97, with 190 readouts and two HF preparations. Ideal energy resolution and the retained-target median theorem are unchanged. Long evolution angles still require synthesis precision; they are not free physical operations.

The [matched comparison](../../assets/generated/energy-estimation/access-comparison.json) preserves both full workloads and their conditional surface-code frontiers. Both use $p=10^{-4}$, 100 ns gates, 500 ns measurements, a 0.005 whole-workload physical allowance and $10^{-5}$ per-group synthesis-model allowance. The comparison retains the different mean-fit costs and Clifford counts of the actual circuits. Neither report supplies an angle/output certificate, initialization cost or measured hardware runtime.

The explicit generator supports circuit-only higher resolution when its width/operation caps allow it:

```bash
python scripts/generate_energy_estimation.py --analytic-powers --phase-bits 12 \
  --no-sim --no-scaling --no-estimate --output-dir artifacts/h2-high-resolution
```

The ideal grid bound at $m=12$ is about 0.000965 Ha; this command exports a circuit and marks its simulation and physical projection unperformed. It does not establish chemical accuracy on hardware.

## Verification and formalization

Actual analytic powers and their controls are compared with independent matrix powers, including scalar phases. Complete small QPE statevectors agree with repeated-access circuits at the same preparation and decoding convention. The numerical checks are scoped to tested powers and inputs; no all-angle synthesis or Lean certificate is provided.

[Direct-power measured group](../../assets/generated/energy-estimation/analytic-powers/measured-group.qasm) · [Logical ledger](../../assets/generated/energy-estimation/analytic-powers/logical-resources.json) · [Conditional physical report](../../assets/generated/energy-estimation/analytic-powers/surface-code.json)
