# Source, method and runtime credits

The circuit construction, independent verifier, driver, explanation, saved
checker and original circuit schematic are project work under the
[MIT license](LICENSE.txt), copyright 2026 Open Quantum Algorithms
contributors. The exact three scientific files retain their SPDX notices and
are copied without edits from `d80a2be51c5156283bc4912dc6ed9b48f8543d43`:

| Original repository path / local snapshot | Bytes | SHA-256 |
|---|---:|---|
| [src/openquantum/optimization_dqi_path_majority_k2.py](code/src/openquantum/optimization_dqi_path_majority_k2.py) | 26863 | `5fe7a85cf7046e50b072d46cbb865a39dbe3890567d09cd86f524aa9873dcd06` |
| [src/openquantum/optimization_dqi_path_majority_k2_verify.py](code/src/openquantum/optimization_dqi_path_majority_k2_verify.py) | 45182 | `57b207fb570321c6e44b6ed14fed4de915087f4017a83bb060c5e9017bb2c2f3` |
| [scripts/run_optimization_dqi_path_majority_k2.py](code/scripts/run_optimization_dqi_path_majority_k2.py) | 44738 | `3599a419ea418fb8f49f69bbb998e5fd089ef1b7d24212a6123f1d91152d09a8` |

The profile credits the project's earlier original path-majority construction,
scientific revision `7a4b1cdaa1354014886f87d14149bd3d2d5edb63`, core SHA-256
`ba3015dbdf35fe3ccec6a79503f2e911b735c2ea716969f690eb5fbc5e3c61bf`.
That [earlier source snapshot](../dqi-path-majority/code/src/openquantum/optimization_dqi_path_majority.py)
retains its separate local-component and generation-only scope.

The method reference is Jordan and colleagues, *Optimization by decoded quantum
interferometry*, Nature **646**, 831–836 (2025), DOI
`10.1038/s41586-025-09527-5`. The versioned supporting archive is DOI
`10.5281/zenodo.16741169`; it is an external reference. Neither author code,
paper text, figures nor benchmark data were adopted in these sources.
The theory-boundary references used by the existing chapter are Apte and
colleagues, *Cycle Codes and Decoded Quantum Interferometry*, arXiv
`2609.40036v2`, and Rosmanis, *A nearly linear-time Decoded Quantum
Interferometry algorithm for the Optimal Polynomial Intersection problem*,
arXiv `2601.15171v1`. Their stronger decoding/access assumptions are not local
guarantees. These are the exact reading versions recorded by the project.

The accepted fixed job used the following runtime; no package binaries are
redistributed or imported by this packet's saved-byte checker:

| Package | Exact identity and role | Exact notice |
|---|---|---|
| CPython | 3.12.14, upstream `python/cpython`, release tag `v3.12.14`; stdlib orchestration | [PSF agreement and bundled notices](licenses/CPython-LICENSE.txt) |
| NumPy | 2.5.4, upstream `numpy/numpy`, recorded git `a3d0ef1df045b858e68115001fdabc930bb3b68c`; local CPU complex128 primitive kernels | [Complete wheel notice](licenses/numpy/LICENSE.txt), BSD-3-Clause and bundled notices; all 16 additional shipped license/COPYING files are preserved under `licenses/numpy/numpy/` |
| psutil | 7.2.2, upstream `giampaolo/psutil`; private process monitoring | [BSD-3-Clause](licenses/psutil-LICENSE.txt) |

[Environment metadata](environment-public.json) records each notice's exact
size/hash and the admitted NumPy/psutil metadata pins. The copied CPython
notice has a new packaging hash; it is not claimed as a pinned run-admission
payload. The NumPy main wheel notice is 46,826 bytes, SHA-256
`649a589ab31076e79e9ce62ba24eae9959598124176640dbc95f87bd10f34fcf`;
the psutil notice is 1,548 bytes, SHA-256
`b89c063b3786e28e0c0a38f1931db61fed35e69dd2a2966fbecffee0f46c8d10`.
The [unchanged project lock](code/requirements-lock.txt) provides version
context. Its notebook, quantum SDK and estimator entries do not mean those
packages executed in the accepted k2 run.

NumPy's startup may perform native sanity work inside process timing. Selected
on-disk hashes do not establish a complete dynamic-library, OS-image or ABI
closure. The public projection omits private environment inventories and
operational receipts while retaining their provenance digests.
