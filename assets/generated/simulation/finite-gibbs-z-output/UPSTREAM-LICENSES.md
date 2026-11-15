# Implementation and upstream credits

The primitive circuit recipe, independent finite reference, interval and
angle helpers, portable wrapper, explanations and circuit illustration are
original Open Quantum Algorithms sources under the unchanged project MIT
[license](LICENSE.txt). Selected original helpers and reference functions
come from scientific source commit
`47ffd10368a738f9576053491a56c6d80a10e602`; the derivation and packaging
boundaries are recorded in [source-derivations.json](source-derivations.json).

The mathematical filtered-jump specification is Chen, Kastoryano, Brandão
and Gilyén, *Efficient quantum thermal simulation*, Nature 646, 561–566
(2025), DOI `10.1038/s41586-025-09583-x`. The finite gate recipe is original.
Neither the paper's text/figures nor its complete sampler or an upstream
benchmark implementation is reproduced by this fixed example.

The accepted calculation used Python 3.12.14 with the exact
[requirements-lock.txt](requirements-lock.txt). Package versions below are
the retained locked versions, rather than a claim about current releases.
The upstream revision/license records are retained from the project's
previously inspected locked distribution metadata.

| Package | Exact version | Upstream source and license |
|---|---|---|
| [Qiskit](https://github.com/Qiskit/qiskit/tree/2.6.2) | 2.6.2 | Apache-2.0; source revision `0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`; distribution `LICENSE.txt` |
| [NumPy](https://github.com/numpy/numpy) | 2.5.4 | BSD-3-Clause main project; retained distribution expression `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`, with component notices |
| [SciPy](https://github.com/scipy/scipy/tree/v1.18.1) | 1.18.1 | BSD-3-Clause main project and distribution/component notices; revision `e4e854eaa8f18d807cd3496028e257e36caa93cc` |

Qiskit supplies native primitive circuit objects and the bounded Statevector
and density backends. NumPy supplies arrays and the two 4-by-4 channel powers.
The independent fixed-H reference uses analytic diagonal phase evolution;
the general original generators also import SciPy for matrix-exponential
reference support. Python's Decimal arithmetic supplies directed scalar
interval operations. Package distribution notices apply when installing and
using the lock; this packet distributes original source and environment
references, rather than copies of the upstream distributions.

The new portable wrapper and public packaging remain source-only and
unexecuted. The accepted saved scientific evidence does not certify a new
installation's floating backend, physical faults, arbitrary-angle synthesis
or mechanized proof. No hardware-resource advantage is claimed.
