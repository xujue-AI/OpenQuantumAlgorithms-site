# Disjoint-pair first symplectic CPF2

This packet contains original MIT-licensed code, saved ideal-circuit evidence
and separately prepared portable adaptations. It implements the first
symplectic correction to Strang splitting from arXiv:2409.08265v4,
Equations 25/26/30, for disjoint ZZ/X pairs. It does not implement a general
commutator compiler or establish application advantage, generic fourth order,
fault-tolerant synthesis or a formal proof.

## Saved evidence

The immutable `corrected_product_formula.py` is the exact scientific module
used for the original 24 passing tests and 32 paired CPF2/Strang cases.
`cases/` retains the original case JSON and both measured-trial QASM files
without modifying parameters or values. `saved-study.json` retains all case
objects but removes controller details and machine timings; `case-summary.csv`
is a convenience view of those saved numbers. `negative-controls.json` is the
unchanged saved missing-boundary/wrong-sign comparison.

`generation/` and `saved-generation.json` contain the original four-, sixteen-
and thirty-two-pair exports at r=8. These were generation-only; they performed
no dense simulation. Complete trials include input resets, H/S preparation,
all Strang steps, both CPF2 boundaries, and every final Z readout bit.

`original-executed.ipynb` preserves the exact genuine five-code-cell notebook
execution, including its one- and two-pair numerical checks and its circuit
image. The frozen notebook prose and metadata still say "planned" because
source bytes were preserved across the later execution. Its actual execution
counts 1 through 5 and saved outputs record that execution. The original
repository-dependent loader was not adapted in this archive.

`one-pair-complete-trial.png` is decoded directly from the original notebook's
PNG output, without rerendering. Both original PNG occurrences have identical
bytes; the packet displays that image once. `source-provenance.json` binds the
original source, saved data, test inventory and genuine notebook hashes.
The first full run failed at notebook startup; its completed components and
the later successful notebook continuation are recorded separately.

## Prepared standalone adaptations

`generate_cpf2_public.py` is a new, **unexecuted** wrapper. It loads the exact
scientific module beside itself and checks its hash, the original lock/license
hashes, Python 3.12.14 and the Qiskit/NumPy/SciPy versions. It keeps the module's
input and operation bounds plus the original producer's finite residual
checks, tolerance 1e-11, fixed 32-case suite and negative controls. It uses no
repository Git state or private source contract. It has not passed a runtime
portability check.

`portable_demo.ipynb` is a separate, **unexecuted** companion with a sibling-file
loader. Its remaining numerical and generation cells retain the original
parameters. Its output lists are empty and execution counts null; original
outputs were not copied into adapted cells. A generation-only wrapper check
would not establish that this companion's numerical cells execute.

In a Python 3.12.14 virtual environment with `requirements-lock.txt` installed,
the following commands are prepared for separately bounded execution:

```bash
python generate_cpf2_public.py --pairs 4 --repetitions 8 \
  --generation-only --output-dir ../cpf2-new-generation
python generate_cpf2_public.py --output-dir ../cpf2-new-small-check
```

The first command generates both measured-trial QASMs and ledgers for the
original four-pair/r=8 instance, without dense simulation. Its exact QASM bytes
can be compared with `generation/m4/`. This is a proposed check, not an
executed validation. The default second command performs one small paired
numerical comparison. `--suite` would perform the original 32 tiny cases and
negative controls; it is outside that proposed one-case portability check.

The original `original_repo_producer.py` is retained byte-for-byte for source
attribution. It needs the repository layout, its tests/source notebook and Git
commit admission. It is not the standalone command above. Its old unexecuted
docstring and the module's source-freeze docstring are preserved deliberately;
the saved provenance states which original components later executed.

## Bounds, dependencies and licenses

Generation has separate limits of 32 pairs/64 wires, 1,024 steps and 100,000
complete primitives. Dense checks stop at two pairs/four wires, 16 steps and
2,000 complete source operations. The wrapper also keeps cooperative
90-second and 32 MiB/128-file output caps, and requires new output directories.
These are not hard memory/process-containment guarantees.

The scientific module needs NumPy 2.5.4, SciPy 1.18.1 and Qiskit 2.6.2.
Drawing notebook circuits additionally uses Matplotlib 3.11.2 and pylatexenc
2.11; the original full environment lock is included. Synthesis, surface-code
mapping, routing/factories, finite-shot sampling, hardware calibration and
classical report latency are absent. No Lean/Rocq proof is supplied.

Original code and wrapper: MIT, `LICENSE.txt`. Qiskit: Apache-2.0. NumPy and
SciPy: BSD-3-Clause. `references.bib` identifies the method and pinned packages.
The circuit figure and explanations are original project artifacts.
