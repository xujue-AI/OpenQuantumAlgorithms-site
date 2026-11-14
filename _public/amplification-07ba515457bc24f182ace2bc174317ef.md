---
title: "Oblivious amplification of complex dynamics"
description: "Actual ancilla reflections and forward/inverse QSP access, with probabilistic success and full trial costs."
---

The [complex QSP circuit](qsp.md) has an accepted block $K\approx e^{-iH}/4$. This page implements a separate amplification circuit around that actual access. It raises acceptance with integer rounds and checks the accepted state and observable; it does not make the evolution deterministic.

:::{note} Implementation status
Ancilla-only reflections, forward access and inverse access are implemented and decomposed into gates. The default three-spin instance is checked before final system readout. The exported measured trials include all ancillary and system measurements. Numerical acceptance/retry projections remain separate from coherent synthesis and physical faults. Deterministic amplification and physical estimates remain planned.
:::

## Task and unknown input

Keep the three-spin open chain, $t=1$, $\hbar=1$, $\alpha=4.1$ and observable $Z_0$ from the dynamics page. Let $A$ be its eight-wire complex QSP circuit and $\Pi$ project the five access/flag/selector ancillas onto zero, while acting as identity on the three system spins. The amplification operator never reads or reflects about the system input state.

The tested input is $|000\rangle$. The operation can act on an unknown state, but retries require a fresh, re-preparable input. A single unknown specimen cannot be copied or automatically recovered after a failed postselection. Preparation and trial assumptions therefore remain part of the workload.

## Actual reflection and access sequence

Construct $R=I-2\Pi$ by X gates on every success ancilla, a multi-controlled Z, then X gates to undo the zero-pattern conversion. The multi-controlled Z is H–MCX–H and is decomposed before counting. No dense reflection matrix is a circuit gate.

Starting with $A$, each chronological round applies $R,A^\dagger,R,A$ and records the required global $\pi$ phase:

$$A_r=(-ARA^\dagger R)^r A.$$

The figure summarizes defined circuit blocks whose implementations are expanded in the primitive exports.

```{embed} #simulation-amplification-circuit
```

This is oblivious amplitude amplification {cite:p}`sim_oblivious_aa_2014,sim_oaa_2017`. In particular, one round has accepted block $3K-4KK^\dagger K$ {cite:p}`sim_taylor_lcu_2015`. Reapplying that cubic to its own output would produce a different angle schedule; our repeated circuit always uses the original $A$ and $A^\dagger$.

## Exact model and the approximate circuit

If $K=sU$ exactly for a unitary $U$, the accepted amplitude is $\sin[(2r+1)\arcsin(s)]$. With $s=1/4$:

| Rounds | Accepted amplitude | Ideal success | Complete $A/A^\dagger$ calls |
|---:|---:|---:|---:|
| 0 | $1/4$ | $1/16$ | 1 |
| 1 | $11/16$ | $121/256$ | 3 |
| 2 | $61/64$ | $3721/4096$ | 5 |
| 3 | $251/256$ | $63001/65536\approx0.961319$ | 7 |

The actual QSP block is approximate, so the formula alone does not validate acceptance or accuracy. Tests extract the accepted block by simulating every data-basis input and compare it with an independent matrix-polynomial recurrence. They separately check the conditional state against exact $e^{-iH}$ evolution, the observable and input-dependent acceptance from singular values.

```{embed} #simulation-amplification-results
```

```{embed} #simulation-amplification-success
```

The round-3 conditional-state error is approximately $4.21\times10^{-9}$ for the default input. This is a numerical state check before final system readout, not a formal uniform certificate. The approximate block can have input-dependent acceptance. It is also necessary to preserve relative phases inside the controlled QSP access; aligning separate branch matrices up to arbitrary phases would not validate the coherent construction.

## Complete trials and conditional workload

Round 3 has four forward and three inverse calls to the entire source circuit and six success-ancilla reflections. That includes 98 ordinary PREPARE/SELECT/inverse-PREPARE accesses and seven selector-controlled accesses, before charging reflection and readout gates. Inverses are actual inverse unitary circuits, not free oracle queries.

The compiled round-3 circuit has 61,303 primitive unitary gates, including 20,042 continuous-angle rotations, and dependency depth 46,783. Eight final measurements are added for a complete measured trial. These are logical counts before synthesis, scheduling and error correction.

```{embed} #simulation-amplification-ledger
```

At the numerical finite-H acceptance, two complete round-3 attempts suffice for a 1% cap on receiving no accepted measured trial. The circuit still fails with nonzero probability. A fixed accepted $Z_0$ mean needs many accepted trials and a separate capped-retry/statistical allocation; rejected attempts run and are charged in full. The ledgers keep those numerical projections distinct from a physical-fault allowance.

Higher one-attempt acceptance is not automatically the cheapest retry workload. Round 2 also has a conditional two-attempt cap for that 1% target, with 43,782 primitive unitary gates per attempt versus 61,303 at round 3. The table retains both choices. This is a logical tradeoff under the numerical ideal-instance assumptions, not a physical-cost optimum.

All wires need fresh initialization on each attempt. The measured export reads all eight wires in Z, accepts only the five-zero ancillary pattern, and retains the system bitstring. Continuous rotations, synthesis-induced acceptance/state changes, hardware resets, routing, factories and physical faults remain unmodeled. A shared estimator cannot treat the accepted mean as an unconditional scalar mean without a postselection-aware precision contract.

## Export phase and scale limits

The pinned Qiskit OpenQASM exporter omits an overall circuit phase. This does not change the supplied measured-trial probabilities, but the downloaded QASM is not a phase-preserving controlled-oracle specification. The generator records the compiled phase and the scope of each export; use the source modules, or explicitly restore that phase, for coherent controlled reuse. In-memory source and primitive phase checks keep it.

The parameterized generator limits width to 24 wires and rounds to 0–4. Its source-operation cap counts top-level instructions; nested definitions and transpilation can cost more before the 200,000-primitive-gate export limit is checked. The demonstrated source is bounded and verified. Dense checks stop at eight total wires and statevector checks at twelve. Counting a short arbitrary external gate definition does not establish cheap compilation.

Oblivious amplification assumes a scaled-unitary target for the ideal success formula. Applying it to a nonunitary [thermal filter](thermal-filter.md) generally changes singular values and the desired thermal state. The separate [known-input thermal-amplification method](thermal-amplification.md) implements the full prepared-input reflection and checks preservation of its accepted purification. A unitary environment dilation's inverse likewise does not invert an irreversible reduced Lindblad channel.

```bash
python scripts/generate_simulation_amplification.py
pytest tests/test_simulation_amplification.py
```

The [notebook](../../notebooks/simulation_amplification.ipynb) supplies the labeled outputs. Standalone reproduction needs the copied Python modules and [locked environment](../../requirements-lock.txt). Circuit text, phase conventions and figures are original; Qiskit 2.6.2 and SciPy 1.18.1 are credited in the existing simulation bibliography.

- [Original source](../../assets/generated/simulation/amplification/amplification.py)
- [Standalone generator](../../assets/generated/simulation/amplification/generate_simulation_amplification.py), [results](../../assets/generated/simulation/amplification/results.json), [rounds CSV](../../assets/generated/simulation/amplification/rounds.csv)
- Measured OpenQASM 3: [round 0](../../assets/generated/simulation/amplification/round-0.qasm), [round 1](../../assets/generated/simulation/amplification/round-1.qasm), [round 2](../../assets/generated/simulation/amplification/round-2.qasm), [round 3](../../assets/generated/simulation/amplification/round-3.qasm)
- [Logical counts, omitted overall phases and success/retry evidence](../../assets/generated/simulation/amplification/logical-resources.json)
- [MIT implementation license](../../assets/generated/simulation/amplification/LICENSE.txt)

[Return to Hamiltonian dynamics](../hamiltonian-simulation.md).
