/-
Copyright (c) 2026 OpenQuantumAlgorithms contributors.
Released under the MIT license in LICENSE.txt.

Finite classical reference for the four-cycle MaxCut instance.
Vertex 0 is the least significant bit of an assignment in Fin 16.
This file does not formalize the Python implementation, a quantum circuit,
adiabatic dynamics, sampling, or a physical resource estimate.
-/

import Std

namespace OptimizationReference

/-- A computational-basis assignment of four binary vertices. -/
abbrev Assignment := Fin 16

/-- Extract vertex `i` using the chapter's least-significant-bit convention. -/
def bit (z : Assignment) (i : Nat) : Nat :=
  (z.val / 2 ^ i) % 2

/-- One unweighted edge contributes one exactly when its endpoint bits differ. -/
def crossing (z : Assignment) (u v : Nat) : Nat :=
  if bit z u = bit z v then 0 else 1

/-- The supplied edge list is (0,1), (1,2), (2,3), (0,3). -/
def cut (z : Assignment) : Nat :=
  crossing z 0 1 + crossing z 1 2 + crossing z 2 3 + crossing z 0 3

/-- Kernel-checked finite enumeration bounds every assignment's cut by four. -/
theorem all_cuts_le_four : ∀ z : Assignment, cut z ≤ 4 := by
  decide

/-- The two alternating assignments are exactly the maximum cuts. -/
theorem optimal_iff : ∀ z : Assignment,
    cut z = 4 ↔ z.val = 5 ∨ z.val = 10 := by
  decide

/-- Assignment 5 is `0101` in displayed q3 q2 q1 q0 order. -/
theorem five_is_optimal : cut ⟨5, by decide⟩ = 4 := by
  decide

/-- Assignment 10 is `1010` in displayed q3 q2 q1 q0 order. -/
theorem ten_is_optimal : cut ⟨10, by decide⟩ = 4 := by
  decide

/-- The teaching walk's marked vertex is a supplied equality predicate. -/
def marked (z : Assignment) : Prop := z.val = 5

/-- The equality predicate marks exactly the supplied vertex. -/
theorem marked_iff : ∀ z : Assignment, marked z ↔ z.val = 5 := by
  intro z
  exact Iff.rfl

end OptimizationReference

-- These declarations make the check log include the theorem dependency audit.
-- `by decide` produces proof terms reduced and checked by the Lean kernel;
-- there is no `native_decide`, `sorry`, or user-declared axiom in this artifact.
#print axioms OptimizationReference.all_cuts_le_four
#print axioms OptimizationReference.optimal_iff
#print axioms OptimizationReference.five_is_optimal
#print axioms OptimizationReference.ten_is_optimal
#print axioms OptimizationReference.marked_iff
