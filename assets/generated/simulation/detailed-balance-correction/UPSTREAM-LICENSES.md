# Implementation and upstream credits

The circuit recipe, finite Fourier reference, bounded checker and explanations
are original Open Quantum Algorithms sources. The copied source files carry
SPDX MIT notices; `v2/LICENSE.txt` is their unchanged project MIT license.

The correction specification is Chen et al., *Efficient quantum thermal
simulation*, Nature 646 (2025), DOI `10.1038/s41586-025-09583-x`, Eq. 19.
The gate recipe is original; neither the paper's text/figures nor an upstream
benchmark implementation is copied. This finite toy is not a reproduction of
the paper's complete synthetic Lindbladian or Gibbs sampler.

The original measured environment was Python 3.12.14 with the exact
`v2/requirements-lock.txt`. The main scientific dependencies and their upstream
sources are:

| Package | Exact version | Upstream license/metadata |
|---|---|---|
| [Qiskit](https://github.com/Qiskit/qiskit) | 2.6.2 | Apache-2.0; distribution license file `LICENSE.txt` |
| [NumPy](https://github.com/numpy/numpy) | 2.5.4 | Main project BSD-3-Clause; recorded distribution expression `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`, with component notices |
| [SciPy](https://github.com/scipy/scipy) | 1.18.1 | BSD-3-Clause main project license and distribution/component notices |

Package versions and license metadata were read from the locked distribution
metadata. Qiskit supplies primitive circuit objects and the bounded Statevector
backend; NumPy supplies tiny arrays/linear algebra; SciPy supplies the twelve
explicit 2×2 exponential calls. Upstream distribution notices apply when using
the locked packages. This download distributes original project source and
environment references.

The numerical checks are floating-point observations. No arbitrary-angle
synthesis, hardware resource estimate, error-correcting-code comparison or
mechanized proof was executed. Ideal bias and retry formulas remain separate
from floating-point, synthesis and hardware certification.
