/-
Copyright (c) 2026 OpenQuantumAlgorithms contributors.
Released under the MIT license in the repository LICENSE.

Unexecuted Boolean-evaluator repair proposal for the same finite two-bit
ripple contract. The earlier integer-evaluator source and its timed-out
proof attempt remain separate evidence. This file has not been compiled or
kernel checked locally. It covers Boolean basis-state actions only, without
quantum amplitudes, relative phase, measurement-assisted AND, arbitrary-width
or modular arithmetic, Shor correctness, or physical resources.
-/

import Std

namespace ShorRippleBooleanReference

abbrev State := Fin 32
abbrev Wire := Fin 5

/-- Explicit Boolean wires in least-significant-first order. -/
structure Bits where
  q0 : Bool
  q1 : Bool
  q2 : Bool
  q3 : Bool
  q4 : Bool
  deriving DecidableEq

def asNat : Bool → Nat
  | false => 0
  | true => 1

/-- Read a validated wire; the last branch is wire 4 because i : Fin 5. -/
def read (i : Wire) (s : Bits) : Bool :=
  match i.val with
  | 0 => s.q0
  | 1 => s.q1
  | 2 => s.q2
  | 3 => s.q3
  | _ => s.q4

/-- X updates one explicit Boolean field, without integer bit arithmetic. -/
def toggle (i : Wire) (s : Bits) : Bits :=
  match i.val with
  | 0 => { s with q0 := !s.q0 }
  | 1 => { s with q1 := !s.q1 }
  | 2 => { s with q2 := !s.q2 }
  | 3 => { s with q3 := !s.q3 }
  | _ => { s with q4 := !s.q4 }

inductive Gate where
  | x (target : Wire)
  | cx (control target : Wire)
  | ccx (control₀ control₁ target : Wire)
  deriving DecidableEq

/-- Classical controlled flips only; no phase or channel semantics. -/
def applyGate : Gate → Bits → Bits
  | .x target, s => toggle target s
  | .cx control target, s =>
      if read control s then toggle target s else s
  | .ccx control₀ control₁ target, s =>
      if read control₀ s && read control₁ s then toggle target s else s

-- BEGIN SOURCE GATE LIST
def sourceGates : List Gate := [
    .cx 0 2,
    .cx 0 4,
    .ccx 2 4 0,
    .cx 1 3,
    .cx 1 0,
    .ccx 3 0 1,
    .ccx 3 0 1,
    .cx 1 0,
    .cx 0 3,
    .ccx 2 4 0,
    .cx 0 4,
    .cx 4 2
  ]
-- END SOURCE GATE LIST

def evalBits (s : Bits) : Bits :=
  sourceGates.foldl (fun state gate => applyGate gate state) s

def aBits (s : Bits) : Nat := asNat s.q0 + 2 * asNat s.q1
def bBits (s : Bits) : Nat := asNat s.q2 + 2 * asNat s.q3
def carryBits (s : Bits) : Nat := asNat s.q4

def decodeBits (s : Bits) : Nat × Nat × Nat :=
  (aBits s, bBits s, carryBits s)

/-- Proposed gate contract staged as 32 Boolean cases with small reductions. -/
theorem ripple_bits (s : Bits) :
    decodeBits (evalBits s) =
      (aBits s, (aBits s + bBits s + carryBits s) % 4, carryBits s) := by
  cases s with
  | mk q0 q1 q2 q3 q4 =>
      cases q0 <;> cases q1 <;> cases q2 <;> cases q3 <;> cases q4 <;> rfl

-- Source audit instructions only; no successful proof record is claimed.
#print axioms ShorRippleBooleanReference.ripple_bits

/-- Integer bit extraction occurs at the encoding boundary only. -/
def natBit (z : State) (i : Nat) : Nat :=
  (z.val / 2 ^ i) % 2

def ofState (z : State) : Bits :=
  ⟨natBit z 0 == 1, natBit z 1 == 1, natBit z 2 == 1,
    natBit z 3 == 1, natBit z 4 == 1⟩

def weighted (s : Bits) : Nat :=
  asNat s.q0 + 2 * asNat s.q1 + 4 * asNat s.q2 +
    8 * asNat s.q3 + 16 * asNat s.q4

theorem weighted_lt32 (s : Bits) : weighted s < 32 := by
  cases s with
  | mk q0 q1 q2 q3 q4 =>
      cases q0 <;> cases q1 <;> cases q2 <;> cases q3 <;> cases q4 <;> decide

#print axioms ShorRippleBooleanReference.weighted_lt32

def toState (s : Bits) : State :=
  ⟨weighted s, weighted_lt32 s⟩

/-- The output encoding preserves every Boolean wire. -/
theorem ofState_toState (s : Bits) : ofState (toState s) = s := by
  cases s with
  | mk q0 q1 q2 q3 q4 =>
      cases q0 <;> cases q1 <;> cases q2 <;> cases q3 <;> cases q4 <;> rfl

#print axioms ShorRippleBooleanReference.ofState_toState

/-- The input encoding preserves all 32 original integer basis labels. -/
theorem toState_ofState : ∀ z : State, toState (ofState z) = z := by
  decide

#print axioms ShorRippleBooleanReference.toState_ofState

/-- Link the Boolean registers to the original integer parity formulas. -/
theorem input_values : ∀ z : State,
    aBits (ofState z) = natBit z 0 + 2 * natBit z 1 ∧
    bBits (ofState z) = natBit z 2 + 2 * natBit z 3 ∧
    carryBits (ofState z) = natBit z 4 := by
  decide

#print axioms ShorRippleBooleanReference.input_values

def a (z : State) : Nat := natBit z 0 + 2 * natBit z 1
def b (z : State) : Nat := natBit z 2 + 2 * natBit z 3
def carry (z : State) : Nat := natBit z 4

def decode (z : State) : Nat × Nat × Nat :=
  (a z, b z, carry z)

theorem decode_ofState (z : State) : decodeBits (ofState z) = decode z := by
  have h := input_values z
  exact Prod.ext h.1 (Prod.ext h.2.1 h.2.2)

#print axioms ShorRippleBooleanReference.decode_ofState

/-- Same finite input/output type, with Boolean gates between the boundaries. -/
def eval (z : State) : State :=
  toState (evalBits (ofState z))

/-- Same Fin 32 addition/carry-restoration contract as the retained proposal. -/
theorem ripple_two_bit : ∀ z : State,
    decode (eval z) = (a z, (a z + b z + carry z) % 4, carry z) := by
  intro z
  have h := input_values z
  have hg := ripple_bits (ofState z)
  calc
    decode (eval z) = decodeBits (ofState (eval z)) :=
      (decode_ofState (eval z)).symm
    _ = decodeBits (evalBits (ofState z)) := by
      rw [eval, ofState_toState]
    _ = (a z, (a z + b z + carry z) % 4, carry z) := by
      simpa only [a, b, carry, h.1, h.2.1, h.2.2] using hg

#print axioms ShorRippleBooleanReference.ripple_two_bit

/-- Derived from the main contract instead of a second finite enumeration. -/
theorem ripple_two_bit_carry_one : ∀ z : State, carry z = 1 →
    decode (eval z) = (a z, (a z + b z + 1) % 4, 1) := by
  intro z hc
  simpa only [hc] using ripple_two_bit z

#print axioms ShorRippleBooleanReference.ripple_two_bit_carry_one

end ShorRippleBooleanReference
