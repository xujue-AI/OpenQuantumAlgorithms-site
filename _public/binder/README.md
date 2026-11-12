# Runnable public notebook

This Binder recipe is published beside the generated website in the public
`xujue-AI/OpenQuantumAlgorithms-site` repository's `gh-pages` branch. It uses
Python 3.12 and pins the scientific libraries used by the Grover notebook.
`postBuild` places the standalone implementation beside the notebook, so a
public reader needs no access to the private development repository.

Launch:

https://mybinder.org/v2/gh/xujue-AI/OpenQuantumAlgorithms-site/gh-pages?urlpath=lab/tree/notebooks/grover.ipynb

The live controls are bounded to 2–6 qubits and 0–12 iterations. Every
evaluation checks the exact ideal statevector against the analytic marked-state
probability. The website reuses saved notebook outputs. Its browser explorer
uses the analytic probability formula; live Python controls run in Binder or a
local Jupyter kernel.

This is a demo environment. It excludes the QDK resource estimator, and the
Conda/JupyterLab and repo2docker base dependencies are not a full environment
lock. Surface-code resource reports use the separate locked build environment
and scripts in the development project. A configured recipe is not evidence of
a successful live Binder build.
