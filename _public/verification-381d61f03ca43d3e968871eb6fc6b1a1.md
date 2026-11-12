---
title: "Verification and reproducibility"
description: "What simulation, resource checks, and formal artifacts establish."
---

Each implementation should publish enough information to reproduce its small example and audit the assumptions behind larger estimates. Verification has several scopes; the page will identify which checks were actually run.

## Evidence levels

| Evidence | What it checks | Limit |
|:---|:---|:---|
| Classical reference | A known answer, exact matrix, or small classical solve | The check covers the tested instances |
| Circuit simulation | State evolution or sampled outcomes for an implemented circuit | Small-instance simulation does not demonstrate large-scale feasibility |
| Resource calculation | Counts and arithmetic under stated assumptions | A count is only as complete as its circuit and cost model |
| Formal artifact | A stated theorem under its definitions and assumptions | A proof does not automatically verify an external compiler or hardware model |

## Reproducible examples

Published examples will record the input, seed where randomness is used, package versions, code revision, and command that generated each figure or result. Notebook runs should begin from a fresh kernel. Larger generators must state when construction itself becomes costly, even if simulation is intentionally limited to small instances.

The first implementation is [Grover's search](algorithms/grover.md). Its chapter describes its circuit and classical check. Other chapters remain marked planned until their artifacts and checks are available.

## Classical simulation

State-vector simulation is a useful first check for small examples. Additional methods, such as stabilizer or tensor-network simulation, will be evaluated when they fit a circuit's structure. A simulator comparison must preserve the circuit, observables, accuracy, and measurement convention.

Every adopted simulator will be identified by package, version, configuration, and relevant reference. A link to a package will not be presented as a test result.

## Lean and other formal developments

Formalization is an optional extension for selected algorithms or reusable primitives. The following external developments are candidates for integration:

| Development | Proof environment | Integration status |
|:---|:---|:---|
| [SQIR / VOQC](https://github.com/inQWIRE/SQIR) | Coq / Rocq | External algorithm and circuit proofs; this project has not reproduced their proof build |
| [QEDIA quantum formal verification](https://github.com/qedia/quantum-formal-verification) | Lean 4 | Research candidate; license, toolchain, and theorem-scope audit pending |

These links identify relevant external work. They do not establish that a proof covers this Python implementation or its resource model. A chapter will add an integrated proof artifact after its build is reproduced and its connection to the executable circuit is documented.

Each formal artifact will state:

- The repository, license, revision, and toolchain.
- The exact theorem and its mathematical assumptions.
- Whether it covers an abstract algorithm, circuit semantics, arithmetic block, or compilation step.
- The relation between the formal object and the executable implementation.
- Any parts of the end-to-end pipeline outside the proof.

**Current status:** no reproduced formal artifact is included in this release.

## Checking physical-resource claims

The [resource chapter](resources.md) defines the current model and its limitations. Later estimators will be checked for units, logical-to-physical conversion, scheduling, failure budget, and sensitivity to assumptions. Independent estimator comparisons will align their inputs before comparing outputs.

## Attribution and reference policy

Each chapter will cite its algorithm references and any adapted code. Reused software will retain its required notices. Original narrative, third-party source code, and third-party figures will have their licenses recorded separately where necessary.

Estimates will distinguish executed results, analytical calculations, and projected values. Dates and revisions will accompany results that depend on changing toolchains or hardware assumptions.
