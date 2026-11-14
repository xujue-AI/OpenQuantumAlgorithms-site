# Classical C4 reference coverage only.

This original Lean file checks the classical input used by the C4 QAOA and
digitized annealing examples. An assignment is a member of `Fin 16`; vertex 0
is its least significant bit. The cut sums the four supplied edges
`(0,1), (1,2), (2,3), (0,3)`.

The Lean kernel checks these closed finite propositions with `by decide`:

- Every encoded assignment has cut at most four.
- A cut equals four exactly when the assignment integer is 5 or 10.
- Integers 5 and 10 each attain four.

The walk's known marked predicate is separately defined as equality to 5.
Its theorem records that definition; it does not prove an oracle circuit.
All five declaration dependency audits are empty. The proof uses Lean core
and the bundled `Std` library, with no mathlib dependency, `native_decide`,
user-declared axioms, or admitted proof. This is finite classical reference
coverage only: it does not connect the definitions to Python, prove quantum
circuit semantics, establish adiabatic or walk dynamics, bound sampling, or
validate fault-tolerant projections.

The executed existing toolchain is **Lean 4 v4.34.0-rc2**, source commit
`6a10ac8c22beadecabdbb0919c2b50214762f91d`. It is a pinned release candidate,
not a claim about the latest release. The recorded executable reports
`arm64-apple-darwin24.6.0`. Lean and bundled Std use the upstream Apache-2.0
license; this original artifact uses the project MIT license in `LICENSE.txt`.

- Exact upstream source: https://github.com/leanprover/lean4/tree/6a10ac8c22beadecabdbb0919c2b50214762f91d
- Exact upstream license: https://github.com/leanprover/lean4/blob/6a10ac8c22beadecabdbb0919c2b50214762f91d/LICENSE

Use an existing elan executable with this toolchain available. The check
script neither installs a toolchain nor changes elan's default. From the
downloaded directory:

```bash
python check.py --output checked
```

Or execute just the proof and dependency audit:

```bash
elan run leanprover/lean4:v4.34.0-rc2 lean C4Reference.lean
```

`--elan /path/to/elan` selects a pre-existing executable if it is not on PATH.
`elan run` is used without `--install`, so an absent toolchain fails the check.
`verification.json` records the actual version output, source hashes, scope,
command, successful exit status, and theorem audits. `check.log` contains the
actual successful checker output. The source checkout exports the same bundle
with:

```bash
python algorithms/optimization/formal/check.py --output assets/generated/optimization/formal
```
