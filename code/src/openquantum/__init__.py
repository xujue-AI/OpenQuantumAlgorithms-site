"""Reproducible quantum algorithm implementations for the documentation site."""

from .grover import (
    build_grover_circuit,
    decompose_to_rotation_basis,
    expected_success_probability,
    optimal_iterations,
    phase_oracle,
)

__all__ = [
    "build_grover_circuit",
    "decompose_to_rotation_basis",
    "expected_success_probability",
    "optimal_iterations",
    "phase_oracle",
]
