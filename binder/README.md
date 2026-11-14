# Runnable public notebook

This Binder recipe is published beside the generated website in the public
`xujue-AI/OpenQuantumAlgorithms-site` repository's `gh-pages` branch. It uses
Python 3.12.14. The site's `code` directory is a source preview; the complete
package remains in `assets/generated/openquantum-code.zip`. Before installing
dependencies, `postBuild` checks the ZIP digest and complete file manifest,
then replaces the preview with the full verified extraction. A failed check
stops the build before installation. It installs the canonical
`requirements-lock.txt`, then the public example package without resolving
another dependency set. It
also retains the standalone Grover download, so a
public reader needs no access to the private development repository.

Launch:

https://mybinder.org/v2/gh/xujue-AI/OpenQuantumAlgorithms-site/gh-pages?urlpath=lab/tree/notebooks/grover.ipynb

The live controls are bounded to 2–6 qubits and 0–12 iterations. Every
evaluation checks the exact ideal statevector against the analytic marked-state
probability. The website reuses saved notebook outputs. Its browser explorer
uses the analytic probability formula; live Python controls run in Binder or a
local Jupyter kernel.

The locked Python packages include the QDK estimator, pandas and synthesis
dependencies used by the Hamiltonian, optimization and resource notebooks.
The Conda/JupyterLab and repo2docker base layers remain separately managed;
installing the Python lock does not certify the entire hosted container.
These notebook checks have run in the locked checkout and standalone Python
environments. A successful hosted Binder build/launch has not been observed.
Static notebook outputs and browser controls work independently of Binder.
Large simulations and new physical searches retain their explicit limits;
a live kernel does not establish hardware execution or a resource guarantee.
