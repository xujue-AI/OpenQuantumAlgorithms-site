# Kernel-compiled finite two-bit ripple reference

This original Lean/Std reference covers all 32 classical inputs of a five-wire
two-bit ripple adder. With a=(q0,q1), b=(q2,q3) and carry=q4, least significant
first, it preserves a and carry and writes `(a+b+carry) mod 4` into b. Carry-in
one is included. The literal native source has eight CX and four CCX gates.

The selected Lean proof subprocess exited zero and printed all eight
supporting/main theorem audits. `ripple_bits`, `ripple_two_bit` and
`ripple_two_bit_carry_one` use Lean's standard propositional-extensionality
axiom `propext`; the other five audits are empty. There are no user-declared
domain axioms, `sorryAx` or `native_decide`. This finite reference was accepted
with that explicit standard-foundation inventory.

The original verification helper exited one because its stricter empty-audit
policy was unsatisfied. `compilation.json` preserves `helper_complete=false`
and `empty_audit_contract_satisfied=false`. The earlier integer and later
explicit-match proof attempts timed out and are not accepted proof results.

The exact compiled Lean source and toolchain bytes are retained. Its header
describes the preparation-time status before the recorded compilation; the
current accepted status is in `compilation.json`. `axiom-audit.txt` contains
only the eight selected scientific declaration audits. No private caller,
grant, worktree, freeze or operational log is included.

The native QASM's literal gate list was matched with the formal source list
by a separate bounded check. That parser is a trusted bridge and is not
formally verified. The compiled theorem concerns classical Boolean-wire
semantics. It excludes Python/parser correctness, coherent quantum amplitudes
and phases, arbitrary widths, modular arithmetic, full Shor/order recovery,
Clifford+T lowering, synthesis and physical fault-tolerant resources.

Original source and this projection use the project MIT license in
`LICENSE.txt`. Lean and bundled Std use Apache-2.0. The recorded toolchain is
`leanprover/lean4:v4.34.0-rc2`, upstream source
[`6a10ac8c22beadecabdbb0919c2b50214762f91d`](https://github.com/leanprover/lean4/tree/6a10ac8c22beadecabdbb0919c2b50214762f91d).
These are exact recorded versions,
without a claim about the latest release.

Readers with that existing Lean/Std toolchain may check the preserved source:

```bash
elan run leanprover/lean4:v4.34.0-rc2 lean RippleTwoBitBool.lean
```

This command has not been run by the public-projection step. Reproduction
requires inspecting all eight audits, including the three `propext` entries;
it should not assert that every dependency inventory is empty.
