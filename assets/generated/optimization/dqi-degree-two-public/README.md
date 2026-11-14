# Degree-two decoded quantum interferometry: accepted saved evidence

This download preserves the original bounded C5 numerical component, a
generation-only figure-eight export, and the genuine four-cell saved-data
companion. The packet records **170 passed pure test cases**, one complete
12-wire/4096-amplitude C5 forward evolution, 512 Boolean clean-workspace decoder
mappings and one 8-by-8 phase-exact CCX reference. No outcomes were sampled.

The original five-stage sequence returned exits `0, 0, 0, 0, 1`: its notebook
wrapper encountered a serialized source-array error. That attempt remains
failed and spent. A separately authorized, saved-notebook-only continuation
later returned zero and completed four ordinary Python cells with counts
`1, 2, 3, 4`. It repeated no test, generator, decoder check, phase reference or
state evolution. Parent acceptance covers the finite floating C5 component and
the separate genuine companion. It does not reclassify the earlier sequence
as successful.

## What is preserved

`assets/generated/optimization/dqi-degree-two/` remains an exact **22-file**
copy of the original sealed bundle. Its manifest SHA-256 is
`6981e4dba5502925dc4cd3ccc9d709e5be43adba7a9673b78ce22d54e9019395`.
Its historical component acceptance remains pending notebook acceptance and
has SHA-256
`158edceadd0ca586cd6da85b7e38c3420566a0c8c2c5b8f941210bacb6a29db1`.
The raw source notebook in that directory is unchanged, with empty outputs and
null counts; its SHA-256 is
`18ac25e51a13e6bf74b188c75ba4f58f2c699e0c734c9c6872dc9f9a7e062c66`.

`notebooks/optimization_dqi_degree_two.ipynb` is the genuine executed companion,
19,950 bytes with SHA-256
`e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9`.
The eight ordered cell identities and cell types are preserved. Its joined
source projection matches the raw template:
`3cc9682dded7381b2895cbca21f68fdc4581a94aac33ae36cad57c333577928f`.
The continuation's private runner joined source arrays in memory after
authentication; it did not rewrite this original template or sealed bundle.

The public `accepted-components.json` describes both events and their scoped
parent acceptance. Opaque source/receipt hashes bind their provenance. No
private control programs, raw process logs or machine coordinates are part of
this download.

## Check the saved packet

The ZIP preserves the repository hierarchy and contains exactly **38 ordinary
files**. Extract it into an empty directory, keep its hierarchy intact, and run
from that directory using Python 3.12:

```bash
python -I -S -B scripts/verify_dqi_degree_two_public_saved.py --artifact-root .
```

This check uses only the standard library. It verifies all extracted file
fingerprints, closed member sets, curated acceptance, the exact genuine
notebook, its cell counts/IDs/source projection, and the unchanged original
saved-reader source. That hash-bound reader inspects the preserved 22 files;
neither reader imports a scientific producer or reconstructs a quantum state,
matrix, angle or probability. A saved check executes no notebook or test and
does not certify fresh scientific replay, platform portability or publication.

For an extracted packet, the archive and `download.json` are intentionally
outside the ZIP to avoid self-reference. The check therefore needs neither
file. To additionally inspect the original archive, leave it outside the
extracted directory and pass `--archive OUTSIDE_PACKET/dqi-degree-two-download.zip`.
The archive is checked against the same closed member graph and exact bytes.
Compare the projection-manifest hash against the separately published
`download.json`; an explicit `--expected-manifest-sha256 PUBLISHED_SHA256` binds
the checker to that external identity.

In the full repository or static-site artifact tree, the sibling metadata
directory contains the ZIP plus its descriptor. The same checker then verifies
the six-file metadata set and both the repository file graph and closed ZIP.
`verify_public_packet(artifact_root, source_root=..., notebook_path=...)` allows
the coordinator's static-site checks to use exported code and the preserved
raw notebook download without refreshing a kernel.

## Source and portable context

The root `src/`, `scripts/` and `tests/` copies preserve the original seven Python
source/test files from source commit
`c69733230b09e25ae514a63e47cbcb1c3d01c130`. The two actual input graphs, source
NDJSON/QASM, lock and MIT license are included. Root `requirements-lock.txt`,
`LICENSE` and `pyproject.toml` match the accepted inputs byte-for-byte. This
hierarchy makes the scientific CLI's repository-relative source paths
explicit; the flat Python copies inside the original 22-file bundle are
evidence snapshots rather than alternate CLI entry points.

No scientific source adaptation was performed. The extracted scientific
generator/validator/test context has **not** been executed as a new experiment.
The earlier actual C5 and figure-eight generator invocations retain their own
evidence. A successful saved reader supplies no new scientific CLI deadline,
performance, environment installation or replay result.

The C5 source circuit contains 12 wires, 1,773 unitary primitives, 1,796 complete
operations and 17 unsynthesized RY gates. Its independent ideal optimum
probability is 18/19 and expected satisfied-clause score is 74/19; the recorded
final signed-complex entry discrepancy is approximately 3.7e-15 at absolute
tolerance 2e-10 and relative tolerance zero. Figure-eight has 24 wires and
13,020 source operations and was never simulated. No generic graph success or
repetition law is attached to it.

C5 is an affine/code-basis equivalent repetition-code toy with a linear-time
classical optimum. It fails the stronger every-neighbor recovery premise; no
semicircle theorem, hardness or optimization-advantage claim transfers. RY
synthesis, physical resource estimates, hardware/QPU execution and a local
formal proof or interval certificate remain unavailable. The original
degree-one ideal and conditional physical evidence is unchanged.

## Credits and environment

The generator, decoder, independent verifier and explanations are original
project work, distributed under the included MIT license. No author circuit,
survey text or figure was adopted. The method is grounded in Jordan and
colleagues' *Optimization by decoded quantum interferometry* (Nature,
DOI `10.1038/s41586-025-09527-5`); its structured OPI claims and sparse classical
benchmarks remain distinct from these binary graph fixtures. The article's
exact-version Rosmanis and Apte preprint citations retain their memory,
field/channel, ensemble and every-neighbor qualifications.

The accepted runs used local Darwin CPU and Python **3.12.14**. The original
scientific path uses the Python standard library. The ordinary companion used
nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, IPython 9.18.1,
jupyter_client 8.10.0, jupyter_core 5.9.1, traitlets 5.16.1 and pyzmq 27.3.0;
pytest 9.2.0 supplied the pure tests and psutil 7.2.2 supplied private monitoring.
`environment.json` records their exact distribution metadata/license/project
credits and selected hash roles, not installation paths or a complete imported
file attestation. Python is PSF-licensed; dependency licenses remain their
upstream licenses. The lock also contains optional Qiskit, NumPy and QDK
packages for the wider project; they were not called by this degree-two path
and supply no verified backend or physical-cost claim here.
