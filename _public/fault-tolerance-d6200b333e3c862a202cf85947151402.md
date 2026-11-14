---
title: "Fault tolerance and resource estimation"
description: "From an ideal circuit to a conditional physical-resource model."
---

An ideal circuit specifies logical operations. A physical processor must implement them while qubits, gates, measurements, and idle periods can fail. Fault tolerance encodes information across many physical qubits and repeatedly measures error syndromes. These measurements reveal information about errors while preserving the encoded quantum information. The result is a logical operation with a modeled failure probability and a physical space and time cost.

## Logical operations and surface-code patches

The surface code uses local stabilizer checks on a two-dimensional layout. Its code distance characterizes the smallest undetectable logical error. Increasing distance consumes more physical qubits and more syndrome rounds; the required distance depends on the noise model, decoder, and permitted failure probability. A threshold is a property of those assumptions, rather than a guarantee supplied by the algorithm's logical qubit count. [Fowler and colleagues' surface-code review](https://arxiv.org/abs/1208.0928) introduces these constructions and their implementation costs.

Layout matters. Logical data patches, routing space, temporary ancillas, and measurement procedures contribute to the footprint. Operations performed through lattice surgery occupy patches for several syndrome rounds. A logical circuit's abstract gate depth therefore needs a timing and scheduling model before it becomes elapsed physical time.

## Why non-Clifford operations cost more

Clifford operations preserve stabilizer structure. Universal computation also needs non-Clifford operations, commonly T gates or equivalent injected resources. A factory distills noisy resource states into states suitable for logical computation. Its footprint, throughput, output error, and number of parallel copies must be included in the report. Factories can dominate a small demonstration.

Arbitrary-angle rotations introduce another approximation step: select a synthesis method and allocate precision, then account for the resulting gates or resource states. Counting an undecomposed multi-controlled operation as one physical gate hides substantial work. Likewise, a synthesis cost formula alone does not supply the actual gate sequence or certify its approximation error.

## Keep accuracy and failure separate

A useful ledger records four different quantities: algorithmic approximation, coherent synthesis approximation, measurement precision and confidence, and physical faults. For one scalar observable, independently justified bias bounds and a statistical interval can be allocated within an absolute error target. A physical fault allowance remains a probability; it is not interchangeable with an observable's absolute error.

For repeated executions, state whether trials are serial or parallel and allocate failure probability across the complete run. For a hybrid optimizer, include every parameter setting, measurement group, and stopping rule. Confidence for one fixed scalar mean does not establish optimizer convergence or simultaneous confidence for an entire energy sum.

## Read this site's reports

The [frozen Grover report](../algorithms/grover.md#end-to-end-resource-analysis) is a concrete illustration with explicit circuit and hardware assumptions. Other algorithms must supply their own preparation, access, approximation, and output costs. Their reports cannot inherit Grover's physical counts.

The general static resource helper uses a pinned QDK synthesis model based on a mean fit. Its arbitrary-rotation reports are **conditional resource projections**: that helper produces no concrete synthesis sequence or instance-level approximation certificate. Physical reliability also depends on the declared surface-code and factory models. See the [QDK model documentation](https://learn.microsoft.com/en-us/azure/quantum/qre-build-error-correction-models) and [reported quantities](https://learn.microsoft.com/en-us/azure/quantum/qre-estimation-results).

The separate serialized instrument model uses actual Ross–Selinger Clifford+T sequences {cite:p}`resource_ross_selinger_2014,resource_qiskit_262,resource_rsgridsynth_022`. The [retargetable-QSP](../algorithms/simulation/retargetable-qsp.md#end-to-end-resource-analysis) and [thermal-spectral](../algorithms/simulation/thermal-spectral.md#end-to-end-resource-analysis) studies check their bounded source, routed and synthesized measured circuits. Their ledgers charge complete failed attempts, reset/readout, controller decisions, idle exposure and factory retries. Numerical synthesis guards and explicit layout/controller assumptions remain conditional; a concrete sequence alone does not establish a certified hardware schedule. Infeasible scenarios remain in the reports.

Compare designs at the same problem instance, output, precision, confidence, and physical assumptions. Publish peak physical qubits and elapsed time as a Pareto frontier. Peak qubits multiplied by runtime is a useful named summary; a detailed occupation schedule provides the integral of occupied qubits over time. Neither metric alone establishes an application advantage.

## Optional lesson

[Qiskit: Fault Tolerant Quantum Computation, Lesson 16](https://www.youtube.com/watch?v=aeaqXh2XXMk) provides a video introduction to the mechanisms behind this appendix. The lesson opens only when selected; it is supplementary to the recorded assumptions in each resource report.
