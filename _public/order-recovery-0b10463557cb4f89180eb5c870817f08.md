---
title: "Classical recovery from an order-finding sample"
description: "Bounded continued-fraction, offset and smooth-component recovery, exact-order certificates, and identical circuit-output comparisons."
downloads:
  - file: ../../notebooks/shor_recovery.ipynb
    title: "Recovery notebook"
---

## Motivation and intuition

An order-finding measurement can encode a reduced fraction: its denominator
may omit a common factor of the unknown order. Extra classical processing can
recover that missing component without rerunning the quantum circuit. Ekerå's
analysis combines a nearby-frequency search with a small-prime-power search
{cite:p}`shor_ekera_success2024`. This page implements a bounded
continued-fraction route and makes its classical work visible.

:::{note} Executed scope and the small-instance limit

The implementation uses original integer-only code, freshly checked arithmetic
circuits, identical samples for both recovery rules, and an additional bounded
exact-order certificate. It has no true-order oracle or input-factorization
routine. **All default orders are small and fully smooth for the chosen
bound.** The retained classical shortcut can recover them before using the
frequency; these examples demonstrate control flow and verification, not
quantum advantage. Physical costing and formal correspondence remain pending.
:::

## Fundamental algorithm and important variants

Let $a$ be coprime to $N$, with unknown order $r<2^m$, and let $t$ be the
number of measured phase bits. The implemented candidate route follows the
paper's continued-fraction method and Algorithms 3 and 4:

1. Examine every distinct frequency $j+u\pmod{2^t}$ for
   $u=0,+1,-1,\ldots,+B,-B$. There is no early peak-isolation heuristic.
2. For each frequency, compute the last continued-fraction denominator $q$
   with $q^2<2^t$, using exact integer comparisons. Retain $1\le q<2^m$.
3. Set $b=cm$ and
   $E=\prod_{p\le b}p^{\lfloor\log_p b\rfloor}$ over primes $p$.
   The code constructs the powers by integer multiplication, avoiding rounded
   logarithms. Keep candidates with $a^{Eq}\equiv1\pmod N$.
4. Algorithm 3 uses a product tree to isolate each prime-power component of
   $a^q$, then finds its required exponent within the bound. Complete every
   retained candidate and choose the smallest returned period.

Here **$b$-smooth means each entire prime power is at most $b$**. An order
component $16$ is not $5$-smooth, even though its only prime is two. The
implementation also retains the paper's useful shortcut: if $a^E=1$, apply
Algorithm 3 to candidate one and recover the wholly smooth order classically.

For an optimal nearby frequency, the CF route requires $t=m+l$ with
$1\le l\le m$ and $r^2<2^t$. If its missing component is $cm$-smooth,
the good candidate produces $r$. An arbitrary candidate can instead produce
$\operatorname{lcm}(r,q)$, a proper multiple of the order. The implementation
therefore distinguishes a verified period from an exact order. The lattice
variants and the complete general success bound are external references,
rather than implemented claims for every configured input
{cite:p}`shor_ekera_success2024`.

### A separately charged exact-order certificate

For a returned period $R$, the bounded certificate completely factors **$R$**,
checks $a^R=1\pmod N$, and checks $a^{R/p}\ne1\pmod N$ for each distinct
prime divisor $p$ of $R$. These witnesses establish minimality as described
in the paper's correctness-verification section. Trial division is capped at
$R\le10^6$; it factors the candidate period, not the input modulus. Larger
periods remain labeled `verified_period_only`. This extra certificate does
not make arbitrary large-period factorization efficient.

## Small circuits and identical-output comparison

The default fixed cases use the original [Fourier arithmetic](../shor.md),
with $a=2$ and $N=15,21,33$. Before examining the order, the generator sets

$$t=\min(2n,\,18-(2n+2)),\qquad n=\lceil\log_2N\rceil,$$

to respect its 18-wire full-statevector cap. Thus $t=8,6,4$, respectively.
Every controlled modular addition/multiply, accumulator cleanup, inverse QFT
and phase measurement is materialized in the downloadable QASM. The circuit
figure displays the actual subcircuits; their primitive definitions are in
the exported program and original source.

```{embed} #shor-recovery-circuit
```

Fresh Qiskit statevectors are compared with the independent Fourier
residue-class reference. Both postprocessors receive all the same measured
frequencies and the same 16 seeded draws per case. All outcomes, including
unsuccessful ones and tiny numerical probabilities, remain in the record.
The probability rows below come from the complete ideal distributions;
they are not QPU measurements or confidence bounds for a random-base driver.
The original-rule exact-order column is diagnosed against the independent
reference; only the new rule has an explicit minimality certificate.

```{include} ../../assets/generated/shor/recovery/summary.md
```

```{embed} #shor-recovery-success
```

The recovered $N=33,a=2$ order is ten, but its half-power is $32=-1\pmod{33}$;
factor extraction still fails. Its four-bit demonstration is below the
paper's CF precision regime. The smooth-only shortcut does not need that
frequency promise. The original higher-precision negative resource records
remain unchanged.

A separate integer-only example examines the supplied frequency $70$ at
$N=59,a=2,t=12$. Looking at its adjacent frequency can change recovery.
This exercises the nonsmooth candidate/offset path; it is explicitly a
synthetic classical frequency example, not another executed quantum circuit.

## End-to-end resource analysis

For $n$ input bits and $t$ phase bits, the existing circuit has $t+2n+2$
logical wires and source work $O(tn^3+t^2)$ with its exact-angle Fourier
arithmetic. Both postprocessors retain the same complete 16-trial quantum
workload here. The new recovery changes classical decisions and work; it does
not remove any preparation, rotation, uncomputation or readout from those
trials.

The CF loop records its divisions. At most $2B+1$ shifted frequencies and
that many distinct candidates are processed. The paper's Algorithm 3 has
total exponent-length scaling $O(cm\log(cm))$ per candidate; this code records
its actual modular-power calls, exponent lengths, sieve assignments and
products. The additional period certificate can use $O(\sqrt R)$ integer
divisions and separate modular witnesses. Those certificate costs are not
part of the paper's efficient smooth-completion bound.

```{embed} #shor-recovery-work
```

Recorded CPU time and selected algorithm-operation counters are separate from physical
quantum runtime; they are not CPython instruction counts or a portable CPU
speed benchmark and do not count every Python multiplication. Input handling, classical readout/feedback communication and
calibrated hardware latency still need an application model. A future
equal-quality comparison may change the number of quantum trials, but it must
first establish the new success contract, include every failed trial, and
hold the physical assumptions fixed.

The surface-code baseline stays frozen. **No new Native estimator or factory
search ran.** Existing physical reports have their own compiled traces and
repetition contracts; this new classical comparison cannot inherit their
runtime unchanged. Rotation synthesis, initialization, scheduling, factories,
routing and classical-interface costing remain pending for this complete
recovery policy. The logical counts and explicit exclusions are recorded.

## Classical verification and available formalization

Independent integer checks compare the counted baseline with the book's
original rule on every frequency in bounded grids. They check optimal
frequency/smoothness cases, false candidates that produce order multiples,
prime-power failures, exact minimality witnesses and resource-limit outputs.
The independent order-enumeration function is test-only; neither recovery
API reads its answer or factors $N$.

The [pinned Quaspy source](https://github.com/ekera/quaspy/tree/8040affdf08814ef7d451d9797d28f7645856151)
is a MIT-licensed algorithm reference. We inspected its CF, filtering and
Algorithm 3 routines, then wrote an integer-only implementation without
importing the package or adding its dependencies. The exact upstream license
is retained in the download. Qiskit, NumPy, SciPy and Matplotlib remain the
locked upstream circuit/reference/rendering packages already credited by the
[main chapter](../shor.md).

No Lean or Coq proof has been built for this Python recovery implementation.
The [external Shor formalizations](../shor.md#available-external-formalization)
have different proved models; their theorem coverage is not a correspondence
proof for these new functions or QASM exports.

## Reproduce, scale and download

```bash
python scripts/generate_shor_recovery.py
pytest tests/test_shor_recovery.py tests/test_shor_recovery_artifacts.py

# A supplied large frequency invokes classical recovery without a simulator.
python scripts/generate_shor_recovery.py --modulus 59 --base 2 \
  --phase-bits 12 --outcome 70 --offset-radius 1 \
  --output-dir artifacts/recovery-59
```

The classical API permits up to 128 modulus bits, 256 phase bits, offset
radius 32, $cm\le512$ and 8,192 modular-power calls per frequency, with an
explicit resource-limit result. These are materialization/work guards, not
large-instance performance evidence. Actual circuit simulation is independently
capped at 18 wires and 4,096 outcomes; the default complete generation has a
180-second limit and a 64 MiB output cap. Custom/materialization-only runs use
a separate output directory, preserving the default cohort.

The [notebook](../../notebooks/shor_recovery.ipynb) replays the saved checks and
executes the integer recovery examples. Optional local controls need a kernel;
the static book uses saved outputs and needs no maintained server.

- [Full checked records and all frequency pairs](../../assets/generated/shor/recovery/results.json)
- [N15/a2/t8 complete measured program](../../assets/generated/shor/recovery/n15-a2-t8.qasm)
- [Integer recovery source](../../assets/generated/shor/recovery/shor_recovery.py)
- [Circuit/reference source](../../assets/generated/shor/recovery/shor.py)
- [Generation script](../../assets/generated/shor/recovery/generate_shor_recovery.py)
- [Project license](../../assets/generated/shor/recovery/LICENSE.txt) and [upstream reference license](../../assets/generated/shor/recovery/Quaspy-LICENSE.txt)

## References

```{bibliography}
```
