---
title: "A fixed 22-wire path-majority DQI pass"
description: "Saved evidence for one ideal degree-two DQI pass on a mixed-parity ten-cycle, with all five original full vectors and explicit acceptance boundaries."
---

(dqi-path-majority-k2)=
## One complete fixed pass

Decoded quantum interferometry (DQI) combines signed clause-error contributions
by preparing low-weight words, computing their syndromes, coherently clearing
the error register and applying a Fourier transform {cite:p}`frontier_dqi2025`.
This page preserves one independently accepted **22-wire ideal primitive
pass** on a fixed two-path graph, recorded on **14 November 2026**. Its full
preparation and output probabilities were checked in the same forward pass.
The saved records and all five original vectors are available below.

This ten-cycle has a linear classical optimum construction. The example makes
preparation, coherent majority decoding and interference inspectable; it
establishes no optimization advantage or generic useful-output guarantee.

:::{important} Exact acceptance scope

Root acceptance covers **one fixed Q22/N10/m10/n9 ideal primitive pass**,
explicitly known-zero initialization and unconditional readout probabilities.
All **37 RY occurrences are unsynthesized**. No actual shots were sampled and
no reset-noise channel was modeled. Fault-tolerant costs, physical faults,
hardware execution, general DQI guarantees and reproduced formal proof remain
unavailable.

The [curated acceptance record](../../assets/generated/optimization/dqi-path-majority-k2/accepted-fixed-run.json)
identifies the later parent record by its exact hash. Original source headers,
ledger and diagnostic flags retain their earlier states: the raw diagnostics
still say `full_program_validated=false` and
`parent_acceptance_recorded=false`. This separate later acceptance does not
rewrite them. The [twelve-wire component and thirty-one-wire source](dqi-degree-two.md#dqi-path-majority-component)
retain their separate scopes; the **31-wire k=3 program remains generation
only**.
:::

## Fixed task and input coordinates

Two internally disjoint five-edge paths join root vertex zero to terminal
vertex one:

$$
P_0=(0,2,3,4,5,1),\qquad P_1=(0,6,7,8,9,1).
$$

The [unchanged input](../../assets/generated/optimization/dqi-path-majority-k2/scientific/source/input-k2.json)
is `k=2`, with ordered signs
$v=(1,1,0,0,0,1,0,0,0,0)$ along these paths. Thus $N=10$, $m=10$ and
$n=9$ after fixing $x_0=0$. Integer bit $j$ represents vertex $x_{j+1}$.
For the reduced edge incidence matrix $B$, the score is

$$
f(x)=\sum_{i=0}^{9}\mathbf1[B_ix=v_i],\qquad
S(x)=\sum_{i=0}^{9}(-1)^{v_i+B_ix}=2f(x)-10.
$$

The two path parities are zero and one. Either terminal bit therefore requires
one violation, yielding optimum score nine. The saved independent comparator
propagated a parity witness directly from signs and edges and scored its output;
it also scored all 512 gauge assignments independently. Those retained
comparisons are separate from the circuit emitter.

The prepared domain is global clause-error weight at most two. The fixed
coefficients are $(a_0,a_1,a_2)=(10,10,2)$ and $Z^2=1280$:

$$
|\psi\rangle=\frac{1}{\sqrt{1280}}
\sum_{|y|\leq2}a_{|y|}|y\rangle.
$$

This includes all 56 zero/singleton/pair words. The circuit uses clause wires
0–9, assignment wires 10–18 and three reused scratch wires 19–21. The scratch
wires first hold the weight counter/flag, then support decoding after the
counter is erased. Vertex one is assignment wire 10; each decoder preserves
that terminal syndrome while using its four internal syndrome wires.

## Original circuit and saved primitive stream

![Original grouped 22-wire circuit. Known-zero initialization precedes degree-two preparation, three sign Z gates, eighteen incidence CX gates, two clean path-majority decoders, nine assignment Hadamards and nine readout mappings. The source stream supplies every individual primitive.](../../assets/generated/optimization/dqi-path-majority-k2/circuit.svg)

This original schematic groups long stages for readability. The
[actual QASM](../../assets/generated/optimization/dqi-path-majority-k2/scientific/source/program.qasm),
[actual NDJSON](../../assets/generated/optimization/dqi-path-majority-k2/scientific/source/program.ndjson)
and [unchanged ledger](../../assets/generated/optimization/dqi-path-majority-k2/scientific/source/program-ledger.json)
preserve every emitted primitive, reset, barrier and readout mapping.

Preparation is counter-controlled and reverses its increments. Three Z gates
on clause wires 0, 1 and 5 encode the fixed signs. Eighteen CX gates explicitly
compute incidence syndromes, so access to the graph/sign data is part of the
circuit. Each five-edge path is recovered by computing four prefix parities
and their majority flag, XORing the decoded word into the clause targets,
then reversing the flag and prefix computation. Two **525-primitive** decoders
reuse clean scratch; their 68 full phase-exact CCX blocks are lowered into
H/CX/T/T-dagger gates in the complete stream. No opaque decoder oracle is charged
as a free operation.

| Saved unitary stage | Primitive applications |
|---|---:|
| Degree-two preparation and counter erasure | 955 |
| Sign access | 3 |
| Incidence access | 18 |
| Two clean path decoders | 1050 |
| Final assignment Hadamards | 9 |
| Complete unitary pass | **2035** |

The saved gate totals are H 253, CX 834, T 488, T-dagger 366, X 54, Z 3 and
RY 37. Twenty-two known-zero resets and nine individual readout mappings give
2,066 operational events; seven barriers give 2,073 source records.
Readout denotes the saved unconditional probability marginal, **not nine
sampled outcomes**. The initial resets were checked only on the explicitly
known zero input, not as a noisy channel or arbitrary-state reset procedure.

## Independent reference and reported results

The independent verifier evolved a 4,194,304-amplitude packed complex128 state
through all 2,035 unitary primitives once. Five checkpoints inspected that same
forward pass: preparation, sign, syndrome, decoder and final Hadamards. It
separately constructed the low-weight character reference directly from the
fixed edges/signs and compared it with the score polynomial

$$
A(x)=\frac{S(x)^2+10S(x)}{\sqrt{512\cdot1280}}.
$$

The saved reference work comprises 512 independent assignments, 5,120 edge
scores and 28,672 character terms. It calls no producer mass, incidence or
decoder helper. Comparisons used signed complex entries, with absolute
tolerance $2\times10^{-10}$ and relative tolerance zero, without global-phase
fitting, renormalization, sparse truncation or postselection.

| Reported saved quantity | Value |
|---|---:|
| Largest checkpoint signed-complex entry error (decoder) | `1.693090163914611e-14` |
| Final signed-complex entry error | `1.0325187949425617e-14` |
| Final unconditional auxiliary leakage | `3.587109805628593e-29` |
| Unconditional output-probability discrepancy | `3.670674875166924e-15` |
| Output probability normalization | `0.999999999999885` |
| Optimum probability, analytic target $81/256$ | `0.3164062499999637` |
| Expected score, analytic target $115/16$ | `7.187499999999175` |

These values are copied from the
[unchanged diagnostics](../../assets/generated/optimization/dqi-path-majority-k2/scientific/verification/diagnostics.json)
and [full 512-row output table](../../assets/generated/optimization/dqi-path-majority-k2/scientific/verification/output-table.json).
Public packaging and the saved checker performed no numerical recomputation.
The table marginalizes over **all** auxiliary values, retaining leakage rather
than conditioning on clean auxiliaries.

## Five original full vectors and byte authentication

The [dedicated vector ZIP](../../assets/generated/optimization/dqi-path-majority-k2/saved-full-vectors.zip)
keeps all five original files together: initial state, preparation reference,
preparation actual, final reference and final actual. Each holds 4,194,304
little-endian complex128 amplitudes as interleaved float64 real/imaginary
values, with wire zero least significant. Each raw member is 67,108,864 bytes;
the complete uncompressed payload is 335,544,320 bytes. Streaming compression
preserves every byte and each separately recorded SHA-256.

The [manifest](../../assets/generated/optimization/dqi-path-majority-k2/public-manifest.json)
binds all public scientific/source bytes and the five member hashes. The
[stdlib saved checker](../../assets/generated/optimization/dqi-path-majority-k2/code/scripts/verify_dqi_path_majority_k2_saved.py)
authenticates an externally pinned manifest, every file and all five ZIP
members in bounded streams. It replays no circuit, extracts no vector and
decodes no values. The [packet README](../../assets/generated/optimization/dqi-path-majority-k2/README.md)
explains inspection and the absent private execution inputs. The exact
[profile](../../assets/generated/optimization/dqi-path-majority-k2/code/src/openquantum/optimization_dqi_path_majority_k2.py),
[independent verifier](../../assets/generated/optimization/dqi-path-majority-k2/code/src/openquantum/optimization_dqi_path_majority_k2_verify.py)
and [driver](../../assets/generated/optimization/dqi-path-majority-k2/code/scripts/run_optimization_dqi_path_majority_k2.py)
are immutable source snapshots, not a new execution grant or runnable public
environment.

## Accounting, theory boundaries and credits

Preparation, sign/incidence access, coherent decoding, scratch erasure and
readout all belong to the recorded complete draw. Only one ideal pass was
executed. A future repetition plan would repeat these operations and add
assignment receipt, edge scoring and best-score selection; no actual shots or
matched quality/repetition resource comparison is supplied here.

The whole scientific process completed in `43.711875165987294` seconds,
including startup, output writes, EOF, reap and cleanup. Maximum sampled RSS
was 210,960,384 bytes; external sampled CPU was `41.71202464` seconds. These
are observations of one local CPU process, not portable benchmark promises or
unsampled-peak measurements. The declared 212,342,784-byte numerical-buffer
envelope includes reserve and is separate from RSS; it is not a runtime
internal-allocation proof. **No fault-tolerant cost estimate** is attached:
rotation synthesis, synthesis error allocation, scheduling, factories, code
distance and physical faults remain excluded. The existing surface-code
baseline is unchanged.

Successful decoding on the prepared weight-two domain does not establish the
stronger every-neighbor premise, OPI access assumptions or generic quality
guarantees in the cited cycle-code/OPI results
{cite:p}`frontier_dqi_cycle_codes_2026,frontier_dqi_rosmanis_2026`.
Larger full passes, decoder co-design and matched resource comparisons remain
planned. This fixed check supplies neither a hard-family advantage nor a
local formal theorem.

The implementation and figure are original project work under
[MIT](../../assets/generated/optimization/dqi-path-majority-k2/LICENSE.txt).
The accepted job used CPython 3.12.14, NumPy 2.5.4 (recorded git
`a3d0ef1df045b858e68115001fdabc930bb3b68c`) for local CPU primitive kernels and
psutil 7.2.2 for private monitoring. No quantum SDK, resource estimator or new
notebook ran for this increment. Exact source pins, package roles and complete
license notices appear in the
[credits](../../assets/generated/optimization/dqi-path-majority-k2/CREDITS.md)
and [curated environment record](../../assets/generated/optimization/dqi-path-majority-k2/environment-public.json).
The Nature method and versioned supporting archive remain external references;
their code, text, figures and benchmarks were not copied
{cite:p}`frontier_dqi2025,frontier_dqi_archive`.
