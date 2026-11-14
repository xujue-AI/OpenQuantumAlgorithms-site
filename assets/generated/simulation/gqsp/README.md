# Given-angle GQSP: original checked files

This flat folder contains the exact original `gqsp.py`,
`generate_simulation_gqsp.py`, requirements lock and MIT license. The generator
loads its sibling module and uses its built-in case vector. No private
repository, planning document or controller is required in this layout.

`simulation_gqsp.ipynb` is the original executed notebook with five completed
code cells and the original circuit PNG. `source-notebook.ipynb` preserves
the frozen source before execution. Their joined cell sources, IDs and types
are identical; nbformat changed only source-array line segmentation. The
original notebook introduction and code comments retain their pre-execution
"planned/prototype" wording because these files are preserved byte for byte.
Current observed evidence is summarized in `checks-summary.json`.

Thirty-seven dedicated cases and the exact eight-case suite completed:
six small numerical comparisons and two generation-only exports. The enclosing
first full control run failed during notebook readiness/process observation.
A later notebook-only continuation completed without changing the scientific
module, generator, case vector or notebook source. These are distinct events;
the original failed outcome has not been rewritten. Private machine and
controller records are not in this folder.

## Reproduce with the original code

Use Python 3.12.14 and the supplied lock in a separate virtual environment.
For a downloaded copy, keep the four original code/environment files beside
the notebook and start Jupyter in this folder. The notebook chooses the
local original generator first. This is also the correct working directory
when these files are extracted from the complete book code package.

```bash
python -m pip install -r requirements-lock.txt
python -B generate_simulation_gqsp.py --suite --output-dir ../gqsp-recomputed
python -B generate_simulation_gqsp.py --n 64 --degree 6 --no-sim \
  --output-dir ../gqsp-n64-recomputed
```

Each output directory must be new. Keep new outputs outside this immutable
folder. `--no-sim` is required above four total quantum wires; construction
and simulation have separate limits. This generator transforms a supplied
unitary signal using supplied angles. It is not an inverse angle compiler
or a complete Hamiltonian-simulation algorithm.

## Read the saved evidence

- `study.json` and `manifest.json` are unchanged original suite outputs.
- `scaling.csv` transcribes actual complete-trial counts and small-check scope.
- `generation-scaling.csv` transcribes the notebook's count-only size slice.
- `circuit.png` is decoded from the original notebook display output, with
  no redraw or visual adaptation.
- `checks-summary.json` separates test/suite evidence, notebook evidence and
  incomplete physical/formal/application scope.
- `test-case-outcomes.json` retains all 37 original test names/outcomes with
  machine, timestamp and duration fields removed.
- `public-manifest.json` describes this closed public folder.
- `verify_saved.py` is a new pure saved-file checker; it does not import
  Qiskit or rerun circuits. Its final-folder execution is a separate
  portability/integrity check, not inherited scientific validation.

The original numerical comparisons use floating-point matrices and gates.
They are not a uniform interval certificate, calibrated hardware result,
fault-tolerant synthesis proof or application-advantage benchmark. Large
generation-only QASM files carry no dense-simulation claim.

The core source is original MIT code. Qiskit 2.6.2 is used under Apache-2.0;
NumPy supplies independent classical matrices. The primary algorithm
reference is Motlagh and Wiebe, *Generalized Quantum Signal Processing*,
PRX Quantum 5, 020368 (2024), DOI `10.1103/PRXQuantum.5.020368`.
