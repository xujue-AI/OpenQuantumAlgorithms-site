# Open Quantum Algorithms examples

Original public circuit examples; private project plans, credentials,
worker settings and Git history are excluded.

Use Python 3.12. Install the included requirements lock for the full checked
environment, then `python -m pip install --no-deps .`. Binder uses the separately
documented smaller demo environment and installs this package without dependencies.

Examples cover Grover, QFT/QPE, amplitude estimation, reversible-arithmetic
order finding, HHL and QSVT,
product formulas, Taylor-LCU/QSP, amplification,
dissipative channels, thermal preparation and exact/finite-shot QAOA/VQE
toy workflows. Scope and limitations are documented at
https://xujue-ai.github.io/OpenQuantumAlgorithms-site/ . Arbitrary-rotation
resource projections remain conditional; no synthesis certificate is supplied.

Run `python -m pytest`, execute the notebooks, or use the generator scripts.
Run `scripts/generate_grover.py` before its resource-estimation script.
