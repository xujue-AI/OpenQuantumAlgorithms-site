---
title: "Ripple-carry modular arithmetic"
description: "An explicit Boolean carry alternative, its space/rotation tradeoff, and an equal-task conditional resource comparison."
downloads:
  - file: ../../notebooks/shor_ripple.ipynb
    title: "Executable ripple-arithmetic notebook"
---

The [factoring circuit](../shor.md) needs reversible modular multiplication. Fourier arithmetic stores addition in phases; ripple arithmetic carries ordinary binary information along a register using reversible Boolean gates. This variant makes the tradeoff visible: more work qubits and compiled operations, but many fewer general-angle rotations in the demonstrated circuits.

:::{note} Implementation status

The carry, modular-add and modular-multiply circuits are implemented and numerically checked. The full $N=15$ and $N=21$ circuits are simulated with a bounded complex-amplitude source simulator. Small coherent circuits are independently checked with Qiskit. Larger generators and their logical counts are separate from quantum simulation. Physical results below are an equal-task **conditional QDK model comparison**, not an observed hardware speedup or a synthesis certificate.
:::

## Fundamental carry construction

Our original implementation uses the MAJ/UMA idea of Cuccaro, Draper, Kutin and Moulton {cite:p}`shor_ripple_cuccaro_2004`. For bit wires $(a,b,c)$, MAJ applies CX from $a$ to $b$, CX from $a$ to $c$, and CCX from $(b,c)$ to $a$. The final $a$ stores the majority/carry. Reverse UMA restores $a$ and $c$ while replacing $b$ with their sum bit. Chaining MAJ from low to high bits and UMA in reverse gives

$$|a,b,c\rangle\longmapsto |a,(a+b+c)\bmod 2^m,c\rangle.$$

The circuit exposes no overflow output. It uses $2m+1$ wires, **$2m$ Toffoli gates and $4m$ CX gates**, with depth linear in $m$. These intentionally simple endpoints are not the paper's optimized gate count.

```{embed} #shor-ripple-adder
```

The five-wire example adds two-bit numbers modulo four, preserving the first register and carry input. The notebook checks all 32 basis inputs and includes a separate complex-state check of controlled modular multiplication.

## From addition to a clean modular multiplier

Let $n=\lceil\log_2N\rceil$ and $m=n+1$. A constant addition prepares the constant into an $m$-bit zero register, runs the carry adder, then unprepares the restored constant register. Preparation uses X, CX or CCX for zero, one or two external controls. The adder itself remains an explicitly defined Boolean circuit.

A modular addition first adds $b$, subtracts $N$, copies the accumulator's high bit into a borrow flag, and conditionally restores $N$. Subtracting $b$ reveals the complementary sign condition needed to erase the flag; adding $b$ restores the final result. The carry and constant register are clean after every addition. The subtract/restore work also cancels on an inactive control branch.

To multiply, add $(a2^j\bmod N)$ for each input bit into a zero accumulator, swap the input/result under the external control, then subtract the inverse multiplier's weighted constants to erase the old input. The circuit includes every control, constant preparation, swap and uncomputation. No dense unitary or arithmetic lookup table constructs a gate.

The valid contract is $x<N$, zero accumulator, borrow, constant and carry workspace. The same reversible circuit has an unspecified extension on invalid or dirty inputs. In-memory gates preserve the relative phases needed for coherent controls. Fully measured QASM exports omit the compiled overall global phase, recorded in their ledger; they are not phase-preserving specifications for another controlled oracle.

## Small factoring circuits and important variants

The ripple multiplier uses **$3n+5$ wires**, ordered as external control, $n$ input bits, $n+1$ accumulator bits, borrow flag, $n+1$ constant bits and carry. Full-register phase estimation uses **$t+3n+4$ wires**. Thus $N=15,a=2,t=4$ uses 20 wires, and $N=21,a=2,t=6$ uses 25.

```{embed} #shor-ripple-circuit
```

Both use the same modular input $1$, exponent precision, final exponent measurement and continued-fraction/gcd rule as the Fourier baseline. Their ideal factor success is respectively $0.5$ and about $0.287989$, requiring eight and sixteen complete trials for an ideal fixed-base sampling failure allowance of $0.005$.

| Arithmetic variant | Main tradeoff | Current status |
|---|---|---|
| Fourier constant addition | Fewer work registers; many phase rotations | Implemented baseline |
| This MAJ/UMA ripple addition | Extra constant/carry workspace; Boolean arithmetic | Implemented here |
| Optimized Cuccaro endpoints and specialized constant adders | Can reduce the coefficients of the Boolean counts | Planned comparison |
| Approximate Fourier arithmetic | Reduces rotations with a certified circuit-wide error budget | Planned |

## End-to-end space and time scaling

Constant addition has $O(n)$ gates and depth. Five additions plus borrow handling implement one modular addition; $2n$ modular additions and $n$ controlled swaps implement one nonidentity multiplier. Consequently the complete coherent circuit has

$$Q_{\rm ripple}=t+3n+4,\qquad G_{\rm ripple}=O(tn^2+t^2).$$

At $t=2n$, this is $O(n^3)$ logical gate growth, compared with the implemented exact-Fourier baseline's $O(n^4)$, at the cost of $n+2$ extra wires. This asymptotic comparison does not make the small ripple circuits cheaper in every metric. The following actual compiler counts include preparation, all work handling, inverse arithmetic, inverse QFT and readout:

| Task | Method | Logical wires | Primitive operations per full trial | Dependency depth | RX/RY/RZ operations before angle classification |
|---|---|---:|---:|---:|---:|
| $N=15,a=2,t=4$ | Fourier | 14 | 8,669 | 4,994 | 4,388 |
| Same task | Ripple | 20 | 16,473 | 9,978 | 121 |
| $N=21,a=2,t=6$ | Fourier | 18 | 45,276 | 23,665 | 23,220 |
| Same task | Ripple | 25 | 74,260 | 45,004 | 512 |

Some RX/RY/RZ angles are exact Clifford/T angles; these columns are **not canonical arbitrary-rotation counts**. QDK classifies them separately below. Dependency depth is a compiler graph measure, not physical elapsed time. Eight N15 ripple trials charge 131,784 primitive operations including readout; sixteen N21 trials charge 1,188,160. The [scaling CSV](../../assets/generated/shor/ripple/scaling.csv) materializes both methods at $t=2n$ for $N=7,15,21,33$, without claiming those larger circuits simulated. Identity multiplier constants affect instance-specific counts.

## Fault-tolerant analysis at an equal task

The [computed comparison](../../assets/generated/shor/ripple/surface-code-comparison.json) keeps $N=15,a=2,t=4$, input $1$, readout, eight complete trials and all requested budgets fixed. Both methods use the same illustrative surface-code profile: physical error $10^{-4}$, 100 ns gates and 500 ns measurement, a separate physical-fault allowance $0.005$ over all eight trials, and synthesis-model allowance $10^{-5}$ per trial. Ideal sampling failure is separately at most $0.005$. No generic random-base success guarantee is inferred.

| QDK canonical resource per trial | Fourier | Ripple |
|---|---:|---:|
| Exact T instructions | 1,599 | 6,897 |
| Arbitrary RZ instructions | 2,737 | 9 |
| Fixed mean-fit T states per arbitrary rotation | 20 | 16 |
| Total modeled T states | 56,339 | 7,041 |
| Modeled synthesis scalar | $6.887\times10^{-6}$ | $4.236\times10^{-6}$ |

The ripple model has two Pareto points: **30,034 physical qubits and 0.505584 seconds for all eight trials**, or **32,398 qubits and 0.393232 seconds**. The same computed Fourier comparison ranges from 32,880–40,740 qubits and 4.16832–2.24448 seconds for eight trials. Thus the declared model favors the lower synthesis workload here, despite larger logical width and depth.

This conclusion is conditional on QDK 1.34.0's PSSPC mean-fit angle synthesis, layout and factory models. It neither generates a concrete synthesis sequence nor certifies approximation error or runtime tails. QDK abstracts Clifford scheduling, which matters because ripple has more compiled gates and depth. The results include all supplied work qubits, preparation, controls, cleanup, final readout, serial trials and modeled factories; classical base selection, continued fractions, gcd, compilation and verification time are excluded from physical elapsed time. All qubits begin at zero in the model, with no explicit reset instructions. These numbers are not an observed physical speedup, and do not alter the frozen Grover baseline. See [fault-tolerance background](../../appendices/fault-tolerance.md).

## Classical simulation and verification

The Boolean simulator interprets the actual X/CX/CCX/SWAP/CSWAP source gates using integer bit operations. It checks active and inactive controls and clean work registers without allocating an exponentially large statevector. Basis checks alone are distinguished from coherent checks: tests also compare Qiskit complex amplitudes on superpositions, including relative phase, and compare a small lowered circuit with its original statevector.

The sparse source simulator applies the actual H, phase and Boolean gates to complex amplitudes without truncation. $N=15$ and $N=21$ probability discrepancies from the independent order/Fourier reference are about $3.33\times10^{-16}$ and $2.91\times10^{-16}$; dirty-work probability is zero. N21 uses at most 380 nonzero amplitudes and 818,384 gate-state applications, rather than a dense 25-wire allocation. The independent reviewer also compared this simulator with Qiskit on general small phase/Boolean circuits.

The 36 dedicated tests cover carries, controls, modular inverses, clean ancillas, coherent relative phase, source/compiled/sparse checks and limits. An independent interpreter additionally checked 43,688 carry-adder cases, 13,426 modular-add branches and 2,280 multiplier branches through $N=33$, plus inverse cases. These are scoped computational checks, not a machine-checked theorem for every width. No Lean proof is integrated for this variant.

## Reproduction, interaction and limits

The [executed notebook](../../notebooks/shor_ripple.ipynb) reuses the implementation and provides optional live Python controls for bounded N15/N21 cases. Static publication omits those kernel controls; a local or Binder kernel can regenerate the actual circuits and plots. The documentation itself remains static.

```bash
python scripts/generate_shor_ripple.py --physical
pytest tests/test_shor_ripple.py
python scripts/generate_shor_ripple.py --modulus 33 --base 2 --phase-bits 12 \
  --no-sim --no-extra --output-dir artifacts/ripple33
```

Input generation permits at most 12 integer bits and 24 exponent bits, with an arithmetic-work preflight and 500,000 compiled-operation cap. Source lowering is capped at 64 wires and 100,000 primitives. The quantum sparse simulator separately caps exponent precision at eight bits, support at 8,192 amplitudes and work at four million gate-state applications. Dense Qiskit checks cap both 20 wires and a separate gate-amplitude work estimate; a full N15 dense check is optional and took about 34 seconds in the recorded CPU environment. These caps reject costly requests instead of promising simulation from a circuit's logical width.

The generator's `--no-sim` skips quantum amplitude simulation; bounded Boolean checks remain. For $N\le33$ these cover every valid input/control branch; larger generated multipliers check four named input values only. Large circuit materialization is not a validated large factoring run.

- [Original module snapshot](../../assets/generated/shor/ripple/shor_ripple.py) and [MIT license](../../assets/generated/shor/ripple/LICENSE.txt); install the public code package to resolve the shared Shor module.
- [N15 complete measured QASM](../../assets/generated/shor/ripple/circuit.qasm) and [logical ledger](../../assets/generated/shor/ripple/logical-resources.json).
- [N21 QASM](../../assets/generated/shor/ripple/n21/circuit.qasm) and [logical ledger](../../assets/generated/shor/ripple/n21/logical-resources.json).
- [Matched conditional physical comparison](../../assets/generated/shor/ripple/surface-code-comparison.json) and [circuit-only scaling CSV](../../assets/generated/shor/ripple/scaling.csv).

The code and figures are original MIT work using Qiskit 2.6.2, credited through its [Apache-2.0 upstream repository](https://github.com/Qiskit/qiskit). Exact package versions and source/QASM hashes are recorded in the ledgers.

## References

```{bibliography}
```
