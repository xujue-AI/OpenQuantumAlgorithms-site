# Open Quantum Algorithms examples

Original public circuit examples; private project plans, credentials,
worker settings and Git history are excluded.

Use Python 3.12.14 and the included requirements lock for the full checked
environment. Verify a fresh extraction as described below, then install this
package with `python -m pip install --no-deps .`. Binder is configured to use
the same Python patch and requirements lock before installing the public package
without dependency resolution. Hosted Binder execution remains unvalidated;
the static site and browser controls do not require a live kernel.

Examples cover Grover, QFT/QPE, amplitude estimation, reversible-arithmetic
order finding and factoring, HHL, QSVT and Neumann-LCU inverses,
product formulas, Taylor-LCU/QSP, molecular-energy demos, amplification,
dissipative channels, thermal preparation, quantum walks, annealing and exact/finite-shot QAOA/VQE
toy workflows. Scope and limitations are documented at
https://xujue-ai.github.io/OpenQuantumAlgorithms-site/ . Arbitrary-rotation
resource projections remain conditional; no synthesis certificate is supplied.

Using the locked environment, run
`python scripts/verify_public_package.py --package-dir .` on a freshly
extracted, untouched copy to check the complete file manifest, isolated
imports and tests. Verify before installing this package or executing its
notebooks/generators. Installation and execution create unrecorded build or
cache files; repeat complete-content verification on a fresh extraction.
Run `python -m pytest`, `python scripts/execute_notebooks.py`, or the generator scripts.
Run `scripts/generate_grover.py` before its resource-estimation script.

Generated fixtures retain the exact source snapshots, measured QASM and raw
records used by the resource checks. Historical sampled optimization runs
are replayed with their matching source. Do not replace that source with a
new implementation without regenerating the corresponding run record.

The original Lean artifact proves five finite classical C4 reference facts.
Its toolchain, source, checker, license and recorded hashes are included.
`python scripts/verify_public_package.py --formal-only` checks those recorded
hashes; it does not rerun Lean. With an already installed pinned elan
toolchain, use `python algorithms/optimization/formal/check.py --output assets/generated/optimization/formal`
to reproduce the kernel check.
Coverage excludes Python equivalence, quantum circuits, sampling and
fault-tolerant models.
