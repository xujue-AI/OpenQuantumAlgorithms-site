# Bounded conditional physical sweeps

Each row requests one normalized-solution Z(q0) mean. The frozen target is 0.05 absolute error
with allocations 0.01 algorithm bias, 0.01 postselection-adjusted synthesis bias and 0.03 sampling.
Statistical, no-acceptance and whole-workload physical allowances are each 0.005.
The coherent-error interpretation of the fixed 1e-6 synthesis scalar is hypothetical; hardware
accuracy, acceptance and runtime-tail confidence are not certified. There is no simultaneous
cross-grid guarantee. Every rejected complete trial, preparation/access and final readout is charged.

This invocation made **13** QDK queries, bounded by **13**.

| Case | Wires | Primitives/attempt | Capped attempts | Fastest peak qubits | Serial seconds | Evidence |
|---|---:|---:|---:|---:|---:|---|
| hhl-size-n1 | 4 | 81 | 692,380 | 207,615 | 810.085 | [logical](hhl-size-n1/logical-resources.json), [reference](hhl-size-n1/reference.json), [physical](hhl-size-n1/surface-code.json) |
| hhl-size-n2 | 5 | 110 | 1,677,690 | 168,498 | 4669.01 | [logical](hhl-size-n2/logical-resources.json), [reference](hhl-size-n2/reference.json), [physical](hhl-size-n2/surface-code.json) |
| hhl-size-n3 | 7 | 284 | 3,062,450 | 249,143 | 48475.5 | [logical](hhl-size-n3/logical-resources.json), [reference](hhl-size-n3/reference.json), [physical](hhl-size-n3/surface-code.json) |
| hhl-size-n4 | 8 | 343 | 4,833,345 | 162,325 | 109644 | [logical](hhl-size-n4/logical-resources.json), [reference](hhl-size-n4/reference.json), [physical](hhl-size-n4/surface-code.json) |
| hhl-condition-k2 | 6 | 219 | 692,380 | 179,240 | 8309.25 | [logical](hhl-condition-k2/logical-resources.json), [reference](hhl-condition-k2/reference.json), [physical](hhl-condition-k2/surface-code.json) |
| hhl-condition-k4 | 7 | 349 | 3,062,450 | 210,743 | 27387.5 | [logical](hhl-condition-k4/logical-resources.json), [reference](hhl-condition-k4/reference.json), [physical](hhl-condition-k4/surface-code.json) |
| hhl-condition-k8 | 8 | 507 | 12,516,100 | 524,085 | 420929 | [logical](hhl-condition-k8/logical-resources.json), [reference](hhl-condition-k8/reference.json), [physical](hhl-condition-k8/surface-code.json) |
| neumann-size-n2 | 5 | 325 | 652,435 | 199,278 | 18767.3 | [logical](neumann-size-n2/logical-resources.json), [reference](neumann-size-n2/reference.json), [physical](neumann-size-n2/surface-code.json) |
| neumann-size-n3 | 6 | 384 | 652,435 | 199,760 | 21128.5 | [logical](neumann-size-n3/logical-resources.json), [reference](neumann-size-n3/reference.json), [physical](neumann-size-n3/surface-code.json) |
| neumann-size-n4 | 7 | 444 | 652,435 | 200,483 | 23209.7 | [logical](neumann-size-n4/logical-resources.json), [reference](neumann-size-n4/reference.json), [physical](neumann-size-n4/surface-code.json) |
| neumann-gap-3-4 | 5 | 325 | 239,670 | 199,278 | 6894.11 | [logical](neumann-gap-3-4/logical-resources.json), [reference](neumann-gap-3-4/reference.json), [physical](neumann-gap-3-4/surface-code.json) |
| neumann-gap-1-4 | 5 | 325 | 2,383,385 | 455,338 | 68558.1 | [logical](neumann-gap-1-4/logical-resources.json), [reference](neumann-gap-1-4/reference.json), [physical](neumann-gap-1-4/surface-code.json) |
| neumann-gap-1-8 | 5 | 325 | 6,550,980 | 395,566 | 222701 | [logical](neumann-gap-1-8/logical-resources.json), [reference](neumann-gap-1-8/reference.json), [physical](neumann-gap-1-8/surface-code.json) |

Fastest means the minimum runtime among enumerated points in the declared QDK models.
The peak-qubit × serial-runtime column in the CSV is a rectangular proxy, not an integrated layout volume.
HHL size changes conditioning too; Neumann's fixed Pauli algebra is not generic matrix loading or a quantum-advantage benchmark.
Initialization/reset faults, classical construction/compilation/reference/readout costs, routing and decoder realization remain excluded.
