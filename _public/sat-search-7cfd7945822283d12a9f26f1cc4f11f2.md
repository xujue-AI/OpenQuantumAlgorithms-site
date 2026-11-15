---
title: "SAT predicates and unknown-count search"
description: "A reversible CNF oracle, multiple marked assignments, and bounded count-agnostic search."
downloads:
  - file: ../../notebooks/grover_sat.ipynb
    title: "SAT search notebook"
  - file: ../../assets/generated/openquantum-code.zip
    title: "Python implementations and tests"
---

Grover search needs a predicate that recognizes a valid answer. Here the input is a Boolean formula in conjunctive normal form (CNF), and the oracle evaluates that formula reversibly. Its satisfying assignments emerge from the supplied clauses; no solution bitstring is embedded in the oracle.

:::{note} Implemented; bounded checks and conditional replay estimates
The executed notebook checks the actual clause oracle and all predetermined searches. The downloadable reports account for complete fixed replays under the stated surface-code model. An inconclusive capped run does not certify unsatisfiability or establish an unknown-count search confidence.
:::

## Input and result

A clause is an OR of signed literals; the formula is the AND of its clauses. The assignment search space has $N=2^n$ elements for $n$ variables. Clause scratch increases circuit width and simulation cost, while the search window is still defined by the assignment space. Positive integer `k` means variable $x_{k-1}$ and negative integer `-k` means its negation. Variable 1 is the least significant circuit bit. The example is

$$
f(x)=(x_0\lor x_1)\land(\neg x_0\lor x_2)\land(\neg x_1\lor\neg x_2)\land x_3.
$$

```python
from openquantum.grover_sat import CNF, build_cnf_phase_oracle

cnf = CNF(4, [[1, 2], [-1, 3], [-2, -3], [4]])
oracle = build_cnf_phase_oracle(cnf)
```

The search returns a classically verified satisfying assignment or an **inconclusive** result when its attempt, oracle-call, or between-attempt soft wall cap is reached. A failed capped search does not certify unsatisfiability. The notebook's separate exhaustive reference can diagnose its tiny example, but that reference is not available to the search controller.

## Compute clauses, mark, uncompute

One scratch qubit stores each clause result, and a reusable clean pool holds temporary conjunctions. On zero-initialized scratch, the oracle implements

$$
\lvert x\rangle\lvert0^m\rangle\lvert0^a\rangle
\longmapsto(-1)^{f(x)}\lvert x\rangle\lvert0^m\rangle\lvert0^a\rangle.
$$

The generator computes every clause into its scratch bit, flips phase when all clause bits are true, and reverses the complete computation. For a nonempty clause, it forms the complement of the all-false condition using X masks and a controlled X. Each multi-control operation uses an exact Toffoli conjunction chain, followed by uncomputation; it does not hide an exponential no-ancilla decomposition. Empty clauses and tautologies retain their Boolean meanings. The clean pool is reused across clauses, final marking, and diffusion. The final inverse restores scratch, allowing assignments to interfere.

![The reversible clause oracle, including compute and uncompute.](../../assets/generated/grover-sat/oracle.svg)

Scratch initialization is part of the oracle's input promise. A correct phase on individual basis inputs is not enough: the notebook also compares a coherent superposition with an independent signed reference vector and checks that clause scratch remains zero.

## Multiple satisfying assignments

The diffuser reflects the assignment register and restores its temporary work. Amplitude amplification supports any marked subset represented by the supplied predicate {cite:p}`groverSatBrassard2002`. Our circuit implements $I-2|u\rangle\langle u|$, the negative of the usual diffuser. This gives a per-iteration global sign and leaves ordinary measurement probabilities unchanged; a coherent controlled use must retain that phase convention.

When an independent bounded reference supplies $M$ marked assignments among $N=2^n$, the ideal probability after $r$ iterations is

$$
P_r=\sin^2\!\left((2r+1)\arcsin\sqrt{M/N}\right).
$$

The notebook compares this formula with circuit simulation only after running the unknown-count searches. Exact enumeration is explicitly a small-instance diagnostic, with its own size limit. It is not free input to a large search.

![Ideal amplification probabilities compared with the bounded classical reference.](../../assets/generated/grover-sat/amplification.svg)

## Search without knowing the count

The bounded controller follows the randomized BBHT schedule {cite:p}`groverSatBoyer1998`: begin with window $m=1$, choose an integer $0\leq r<m$ uniformly, prepare a fresh assignment superposition, execute that many Grover iterations, measure, and evaluate the supplied CNF classically. After a failed verification, update $m$ to $\min(1.2m,\sqrt N)$. A real-valued window contains $\lceil m\rceil$ candidate iteration counts.

```python
from openquantum.bbht_search import BBHTConfig, run_bbht

result = run_bbht(cnf, BBHTConfig(seed=7, max_attempts=16, max_oracle_calls=32))
```

The predetermined seeds 7, 11, and 29 are all retained. Every recorded attempt contains its iteration count, measured assignment, verification result, and circuit provenance. The wall budget is checked between attempts; an in-progress trial can overrun it, so it is not a hard real-time limit. The contradictory formula `[[1], [-1]]` is also run under a cap. Its reference diagnosis and the controller's termination status are reported separately.

The executed SAT traces are:

| Seed | Grover iterations in executed attempts | Verified assignment | Oracle calls |
|---:|---|---|---:|
| 7 | 0 | `1010` | 0 |
| 11 | 0, 1 | `1101` | 1 |
| 29 | 0, 1 | `1010` | 1 |

The zero-iteration success for seed 7 is an initial uniform sample followed by classical verification; it is not evidence that quantum amplification improved that run. The contradictory two-variable case remained inconclusive after eight attempts and two oracle calls. Its separate exhaustive reference has no solutions. For the four-variable SAT example, the diagnostic finds two satisfying assignments, after the search traces were already recorded.

These are bounded implementation checks, not a benchmark against classical SAT solvers. BBHT's theoretical expected-query analysis is not a confidence guarantee for arbitrary implementation caps. This implementation does not infer an unknown-$M$ success confidence, optimizer-like convergence, or an absence certificate from its trace.

## Cost scaling with the supplied formula

Let $m$ be the number of clauses, $L$ the canonical literal count, and $k_j$ the number of distinct literals in each nonempty, non-tautological clause. Define $c(k)=0$ for $k\leq1$, and $c(k)=2k-3$ for $k\geq2$. A $k$-controlled X then uses $c(k)$ Toffolis and $\max(k-2,0)$ clean work bits. Marking an all-one register of $b$ bits uses $c(\max(b-1,0))$ Toffolis. The allocated width is

$$
W=n+m+a,\qquad
a=\max\!\left(0,n-3,m-3,\max_j(k_j-2)\right).
$$

One iteration, including predicate computation and uncomputation, uses

$$
C_{\mathrm{CCX}}=2\sum_j c(k_j)+c(\max(m-1,0))+c(\max(n-1,0)).
$$

X masks, Hadamards, and direct CNOTs add $O(L+m+n)$ work. The pinned level-0 lowering maps each Toffoli to an exact seven-T Clifford+T word, so $r$ iterations use $7rC_{\mathrm{CCX}}$ T/T-dagger gates. The example has nine declared logical qubits and 12 Toffolis per iteration, giving 84 T/T-dagger gates for one iteration. Preparation supplies $n$ Hadamards; final readout measures $n$ data bits. There are no arbitrary-angle rotations or angle-synthesis approximations in these circuits.

For $0<M<N$ satisfying assignments, the uncapped BBHT analysis gives expected $O(\sqrt{N/M})$ oracle queries {cite:p}`groverSatBoyer1998`. Combined with this construction, expected logical work scales as $O((L+m+n)\sqrt{2^n/M})$, plus repeated preparation, measurement, and classical verification. Finite caps need their own success analysis and do not inherit that expectation as a guarantee. The $M=0$ case gives no SAT witness.

The physical space-time projection additionally depends on distance, factory allocation, scheduling, and fault allowance. For a recorded prefix with $A$ attempts and $R=\sum r_i$ iterations, our static replay declares $AW$ qubits, $7RC_{\mathrm{CCX}}$ T/T-dagger gates, and $An$ measurements. Its reported peak-qubit-times-runtime product is a reproducible projection of that prefix. A general physical scaling law needs a specified family of CNFs and fixed success/error assumptions; the tiny reports do not supply such a law by extrapolation.

## Physical analysis of the complete recorded replay

The physical artifact estimates a **fixed replay of the recorded iteration circuits**. Each attempt occupies a fresh, disjoint assignment-and-scratch bank. Its preparation, predicate computation, phase, uncomputation, and output measurements are represented. Measurements are deferred to the end, so the replay satisfies the current static estimator's contract. The replay produces fresh outcomes: it fixes the recorded iteration counts, not the old measurement bitstrings.

That circuit is a resource artifact for a captured sequence, not the original adaptive controller. It does not implement stopping after an intermediate success, reuse banks through resets, or wait for an online verifier. QDK's canonical PSSPC input reduces the circuit to non-Clifford and measurement counts on placeholder wires; it does not retain the actual gate dependency graph. The reported replay time is a conditional count/layout projection, rather than a validated schedule of this circuit or serial adaptive BBHT runtime. The original Clifford work and exact gate ordering remain available in the separately audited QASM.

```{include} ../../assets/generated/grover-sat/resource-table.md
```

The complete fixed replay reserves a synthesis allowance of $10^{-4}$ and a modeled physical-fault allowance of 0.005. Exact Clifford+T lowering leaves zero arbitrary rotations, so the rotation-synthesis contribution is zero. QDK still abstracts Clifford scheduling and conditional surface-code/factory behavior. These allowances do not establish a success confidence for unknown-count search.

All banks are assumed initially in logical zero. Hadamard preparation is explicit; active physical initialization/reset beyond the estimator's layout model is not separately measured. Classical candidate verification is performed and recorded in the search trace. Its CPU diagnostics are separate from the static replay's physical clock; online controller latency and hardware-specific routing/decoder work are outside that clock.

The original three-qubit equality example and its frozen resource baseline are preserved. This SAT workload supplies a distinct predicate and recorded schedule, so its physical counts cannot be substituted for that baseline or presented as a demonstrated application speedup.

## Reproduce and inspect

Run the notebook from a source checkout or unpack the downloadable code bundle so that its `src` directory is available. In a checkout, the artifact generator is:

```bash
python scripts/generate_grover_sat.py
```

For a supplied JSON formula with `n_variables` and `clauses` fields, circuit-only generation is:

```bash
python scripts/generate_grover_sat.py --cnf formula.json --generate-only \
  --iterations 1 --output-dir artifacts/sat-circuit
```

The CLI limits a generated trial to 32 declared qubits, 64 requested iterations, a conservative 200,000-operation preflight, and 250,000 actual lowered operations. Count-agnostic runs default to a separate 14-qubit ideal simulation bound, 16 attempts, 32 oracle calls, and a between-attempt 30-second soft wall budget. Larger bounded circuit materialization remains separate from simulation. The notebook first runs the count-agnostic searches, then performs bounded exhaustive and coherent checks. It also exports a 27-qubit, ten-variable CNF circuit without allocating a statevector.

## Classical checks and formal coverage

The implementation tests compare coherent amplitudes with direct Boolean truth tables, including long clauses, tautologies, empty clauses, multiple solutions, and clean-work restoration. They also verify the diffuser's full-state phase convention, fresh-bank locality, product output laws, seeded controller caps, and exported/native-input hashes. Bounded Qiskit Statevector simulation is an exponential classical implementation check, not a scalable SAT solver or hardware measurement.

No integrated Lean, Rocq, or other formal proof is supplied for this CNF oracle, BBHT controller, or physical compilation. General amplitude-amplification proofs are references rather than proofs of this Python implementation.

- {download}`Executable notebook <../../notebooks/grover_sat.ipynb>`
- [Recorded search traces](../../assets/generated/grover-sat/search-traces.json)
- [Complete fixed-replay resource report](../../assets/generated/grover-sat/replay-resources.json)
- {download}`Generated iteration circuits <../../assets/generated/grover-sat/iteration-circuits.zip>`
- [Grover overview](../grover.md)

Implementation code is original and uses [Qiskit core](https://github.com/Qiskit/qiskit), released under Apache 2.0. Input formulas, seeds, gate counts, package versions, code revisions, and hashes belong with each generated report. A numerical semantic check is distinct from a formal proof; no integrated proof of this implementation or its physical compilation is claimed.

## References

```{bibliography}
```
