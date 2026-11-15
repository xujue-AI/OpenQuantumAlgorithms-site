---
title: "Verification and reproducibility"
description: "What simulation, resource checks, and formal artifacts establish."
---

Each implementation publishes its small example and the assumptions behind larger estimates. Verification has several scopes; each article identifies the checks that were actually run.

## Evidence levels

| Evidence | What it checks | Limit |
|:---|:---|:---|
| Classical reference | A known answer, exact matrix, or small classical solve | The check covers the tested instances |
| Circuit simulation | State evolution or sampled outcomes for an implemented circuit | Small-instance simulation does not demonstrate large-scale feasibility |
| Resource calculation | Counts and arithmetic under stated assumptions | A count is only as complete as its circuit and cost model |
| Formal artifact | A stated theorem under its definitions and assumptions | A proof does not automatically verify an external compiler or hardware model |

## Reproducible examples

Published examples record the input, seed where randomness is used, package versions, source hashes, and command that generated each figure or result. Notebooks execute from fresh kernels before publication. Larger generators state when construction itself becomes costly, even if simulation is intentionally limited to small instances.

The [public code package](assets/generated/openquantum-code.zip) includes the implementations, notebooks, tests and reviewed fixtures. Publication checks run its tests in an isolated extraction. Git history is intentionally absent: regenerated downloads record a null Git revision, while file hashes identify the source. A surrounding, unrelated Git repository is never used as the download's provenance.

The website's `code/` directory is a partial source preview. Its
[preview manifest](https://xujue-ai.github.io/OpenQuantumAlgorithms-site/code/PREVIEW.json) lists retained source files and maps each
omitted fixture to the identical canonical download under `assets/`.
[The full provenance](https://xujue-ai.github.io/OpenQuantumAlgorithms-site/code/PROVENANCE.json) describes the complete ZIP, which
remains the runnable package for tests and notebooks. Publication verifies
both the complete archive and those source/asset identities. Binder assembles
and verifies the full archive before installation; a hosted Binder launch
remains unvalidated.

[Grover's search](algorithms/grover.md) checks its analytic success probability. Other implemented chapters compare circuits with independent matrix exponentials, reversible permutations, classical solves, graph recurrences, objective calculations or likelihood/inference arithmetic. An executed check covers its declared instances and limits; a larger circuit-only row remains labeled separately.

## Classical simulation

State-vector simulation is a useful first check for small examples. Additional methods, such as stabilizer or tensor-network simulation, will be evaluated when they fit a circuit's structure. A simulator comparison must preserve the circuit, observables, accuracy, and measurement convention.

Every adopted simulator will be identified by package, version, configuration, and relevant reference. A link to a package will not be presented as a test result.

## Lean and other formal developments

Formalization is an optional extension for selected algorithms or reusable primitives. The following external developments are candidates for integration:

| Development | Proof environment | Integration status |
|:---|:---|:---|
| [SQIR / VOQC](https://github.com/inQWIRE/SQIR) | Coq / Rocq | External algorithm and circuit proofs; this project has not reproduced their proof build |
| [QEDIA quantum formal verification](https://github.com/qedia/quantum-formal-verification) | Lean 4 | Research candidate; license, toolchain, and theorem-scope audit pending |

These links identify external work. They do not establish that a proof covers this Python implementation or its resource model.

Each formal artifact will state:

- The repository, license, revision, and toolchain.
- The exact theorem and its mathematical assumptions.
- Whether it covers an abstract algorithm, circuit semantics, arithmetic block, or compilation step.
- The relation between the formal object and the executable implementation.
- Any parts of the end-to-end pipeline outside the proof.

**Current status: finite classical C4 reference coverage only.** An original
[Lean source](assets/generated/optimization/formal/C4Reference.lean) checks the
four-edge MaxCut input on all 16 assignments. It proves that every cut is at
most four and that precisely assignments 5 and 10 attain four, with a separate
declaration about the known marked vertex. All five saved declaration audits
report no axioms. The [verification record](assets/generated/optimization/formal/verification.json)
and [reproduction instructions](assets/generated/optimization/formal/README.md)
pin Lean `v4.34.0-rc2`, its upstream commit and licenses.

The proof build was reproduced with that existing toolchain. CI checks the
proof-source and recorded-artifact hashes; it does not install or rerun Lean.
There is no verified connection to Python, quantum gates, compilation,
sampling confidence or fault-tolerant resources. See the
[optimization chapter](algorithms/optimization.md#verification-and-formalization)
for the scope beside the corresponding example.

## Checking physical-resource claims

The [resource chapter](algorithms/grover.md#end-to-end-resource-analysis) defines the current model and its limitations. Later estimators will be checked for units, logical-to-physical conversion, scheduling, failure budget, and sensitivity to assumptions. Independent estimator comparisons will align their inputs before comparing outputs.

CI recomputes Grover's nine physical points and compares their physical quantities,
budgets and circuit identity with the frozen report. The published baseline files
are preserved. Importer wall time and an inactive rotation-search parameter are
excluded from that comparison only for the exact rotation-free baseline trace.
Changed physical costs or assumptions fail the check.

## Attribution and reference policy

Each chapter will cite its algorithm references and any adapted code. Reused software will retain its required notices. Original narrative, third-party source code, and third-party figures will have their licenses recorded separately where necessary.

Estimates will distinguish executed results, analytical calculations, and projected values. Dates and revisions will accompany results that depend on changing toolchains or hardware assumptions.
