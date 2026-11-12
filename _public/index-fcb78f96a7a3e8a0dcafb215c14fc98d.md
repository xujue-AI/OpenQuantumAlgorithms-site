---
title: "Quantum algorithms, end to end"
description: "Quantum circuit examples, classical checks, and fault-tolerant resource estimates."
---

:::{div}
:class: hero

Quantum algorithm implementations

## Circuits and resource estimates

Inspect a small working circuit, generate larger instances, and reproduce a resource estimate with its input assumptions and physical model.

**[Start with Grover's search](algorithms/grover.md)** · [Explore the roadmap](roadmap.md)
:::

This project uses *Quantum Algorithms: A Survey of Applications and End-to-end Complexities* as a guide to algorithm coverage and end-to-end questions {cite:p}`survey2025`. The explanations and implementations here are developed for this handbook; references identify the papers and software behind each example.

:::{note} Publication status

This is the first release of the handbook. Grover's search is the first worked example. The other algorithm chapters specify their planned implementations and checks. The resource chapter defines the surface-code baseline and identifies the limits of the current estimate.
:::

## What each implementation will provide

::::{div}
:class: feature-grid

:::{div}
:class: feature-card

### Understand the problem
An input and output contract, the access model, accuracy and success requirements, and a comparable classical task.
:::

:::{div}
:class: feature-card

### Inspect the circuit
A small reproducible circuit, a downloadable notebook, and code that generates larger instances.
:::

:::{div}
:class: feature-card

### Account for the resources
Logical circuit counts, compilation assumptions, and physical qubits and time under a named fault-tolerant model.
:::

:::{div}
:class: feature-card

### Check the evidence
Classical checks, cited implementations, and formal artifacts where a relevant verified development exists.
:::
::::

## Run the first example

[Grover's search](algorithms/grover.md) connects an explicit oracle to a circuit, simulated measurement probabilities, and logical operation counts. The [companion notebook](notebooks/grover.ipynb) displays the saved results; its download is available in the page's download panel.

Read [the resource baseline](resources.md) before interpreting physical estimates, and [verification](verification.md) for what each type of check establishes.

To run the notebook independently, save [the generator](assets/generated/grover/grover.py) beside it and install the [pinned Python 3.12 environment](requirements-lock.txt):

```bash
python -m pip install -r requirements-lock.txt
```

The published original code and documentation are covered by the [MIT license](LICENSE). Package licenses and attribution remain recorded with their implementations.

## Coverage

| Algorithm family | First implementation target | Status |
|:---|:---|:---|
| [Search and amplitude amplification](algorithms/grover.md) | Search with an explicit marked-state oracle | First worked example |
| [Fourier transform and phase estimation](algorithms/qft-qpe.md) | Known-phase unitary and eigenstate | Planned |
| [Factoring and order finding](algorithms/shor.md) | Reversible modular arithmetic | Planned |
| [Hamiltonian simulation](algorithms/hamiltonian-simulation.md) | Small spin model, then scalable generators | Planned |
| [Quantum linear systems](algorithms/linear-systems.md) | Small well-conditioned system and an observable | Planned |
| [Amplitude estimation](algorithms/amplitude-estimation.md) | A specified probability-estimation task | Planned |
| [Optimization](algorithms/optimization.md) | Small MaxCut and variational energy examples | Planned |

The [roadmap](roadmap.md) explains the release order and the evidence required to call a chapter complete.

## Open tools, attributed use

The first circuit implementation uses [Qiskit](https://github.com/Qiskit/qiskit). Later chapters will evaluate packages such as [Qualtran](https://github.com/quantumlib/Qualtran), [PennyLane](https://github.com/PennyLaneAI/pennylane), and [Yao](https://github.com/QuantumBFS/Yao.jl) where their abstractions fit the task. Each adopted implementation will record its repository, license, version, and any modifications. The site is built with [Jupyter Book 2](https://jupyterbook.org/stable/) and the MyST Document Engine.

## References

```{bibliography}
```
