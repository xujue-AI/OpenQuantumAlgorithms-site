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

This is an evolving handbook. Each listed family has bounded working circuits and independent checks, with more general methods still marked planned. The examples include coherent and adaptive estimation, a bounded factoring driver, structured linear inverses, dynamics/open-system/thermal simulation, quantum walks and QAOA/VQE. The in-page resource sections distinguish computed logical workloads, conditional physical models and unimplemented layers.
:::

## Reading the book

Algorithms are listed directly in the table of contents. Each article brings together **motivation and intuition**, the **fundamental algorithm and complexity**, **important variants**, a **small circuit**, **end-to-end resource analysis**, and **verification/formalization**. These are sections of the algorithm page. Substantial methods such as QSP have a separate page under their parent chapter; notebooks remain executable companions. Physical estimates and incomplete methods are labeled explicitly.

The appendices introduce [quantum physics](appendices/quantum-physics.md), [mathematics](appendices/mathematics.md), and [fault-tolerant computing](appendices/fault-tolerance.md). Reading and video links appear beside the concepts they support; videos are optional.

Download the [public Python package, notebooks and tests](assets/generated/openquantum-code.zip) to reproduce examples without accessing the private development repository.

The website's source preview keeps code files beside the
[complete download](assets/generated/openquantum-code.zip). Its
[manifest](https://xujue-ai.github.io/OpenQuantumAlgorithms-site/code/PREVIEW.json) maps omitted fixtures to identical public assets;
use the full download to run the complete tests and notebooks.

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

## Run the examples

[Grover's search](algorithms/grover.md) connects an explicit oracle to a circuit, simulated measurement probabilities, and logical operation counts. The [companion notebook](notebooks/grover.ipynb) displays the saved results; its download is available in the page's download panel.

The [dynamics chapter](algorithms/hamiltonian-simulation.md) and [optimization chapter](algorithms/optimization.md) add checked circuits and complete-attempt or hybrid-workload records. Optimization includes conditional model replays of recorded ideal paths. [Quantum walks](algorithms/quantum-walks.md) provide explicit graph circuits and size sweeps. [QFT/QPE](algorithms/qft-qpe.md), [order finding](algorithms/shor.md), [linear systems](algorithms/linear-systems.md) and [amplitude estimation](algorithms/amplitude-estimation.md) have their own worked circuits and bounded resource reports. A method's article identifies any missing synthesis, reset or adaptive scheduling layer.

Read [the resource baseline](algorithms/grover.md#end-to-end-resource-analysis) before interpreting physical estimates, and [verification](verification.md) for what each type of check establishes.

To run the notebook independently, save [the generator](assets/generated/grover/grover.py) beside it and install the [pinned Python 3.12 environment](requirements-lock.txt):

```bash
python -m pip install -r requirements-lock.txt
```

The published original code and documentation are covered by the [MIT license](LICENSE). Package licenses and attribution remain recorded with their implementations.

## Coverage

| Algorithm family | First implementation target | Status |
|:---|:---|:---|
| [Search and amplitude amplification](algorithms/grover.md) | Search with an explicit marked-state oracle | First worked example |
| [Fourier transform and phase estimation](algorithms/qft-qpe.md) | Exact/approximate transforms; textbook and feedback phase estimation | Checked bounded circuits; static conditional scaling reports |
| [Factoring and order finding](algorithms/shor.md) | Reversible modular arithmetic, ripple arithmetic and a bounded random-base driver | Checked circuits and failed-trial traces; adaptive physical projection pending |
| [Quantum simulation](algorithms/quantum-simulation.md) | Product formulas, LCU, QSP/amplification, channels and thermal preparation | Checked size/accuracy workloads and bounded conditional physical reports; wider access and certified hardware realization planned |
| [Energy estimation](algorithms/energy-estimation.md) | Retained-target QPE for a supplied H2 fixture; explicit Pauli access | Checked small circuits and conditional scaling; fresh molecular preprocessing and general ground-state preparation planned |
| [Quantum linear systems](algorithms/linear-systems.md) | HHL, interval-valid reciprocal QSVT and uniformly bounded Neumann LCU | Checked structured families and ideal-phase certificates; general access and optimal-degree compilation planned |
| [Amplitude estimation](algorithms/amplitude-estimation.md) | Reversible threshold predicate, coherent estimation and adaptive inference | Checked circuits, arithmetic intervals and recorded-path resources |
| [Quantum walks](algorithms/quantum-walks.md) | Coined cycle and continuous-time hypercube | Checked graph access, size sweeps and conditional physical reports |
| [Optimization](algorithms/optimization.md) | MaxCut, variational energy and digitized annealing | Checked exact and sampled workflows; bounded classical Lean reference |

The [roadmap](roadmap.md) explains the release order and the evidence required to call a chapter complete.

## Open tools, attributed use

The first circuit implementation uses [Qiskit](https://github.com/Qiskit/qiskit). Later chapters will evaluate packages such as [Qualtran](https://github.com/quantumlib/Qualtran), [PennyLane](https://github.com/PennyLaneAI/pennylane), and [Yao](https://github.com/QuantumBFS/Yao.jl) where their abstractions fit the task. Each adopted implementation will record its repository, license, version, and any modifications. The site is built with [Jupyter Book 2](https://jupyterbook.org/stable/) and the MyST Document Engine.

## References

```{bibliography}
```
