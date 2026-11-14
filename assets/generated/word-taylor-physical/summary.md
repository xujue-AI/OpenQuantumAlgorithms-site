# One-spin Taylor instrument resource comparison

Both tasks use H=.35X, time=1, fresh zero input and one Z0 mean
with accuracy0.05. Actual Clifford+T sequences and full source/routed
readout laws are checked. Initialization, failed attempts, routing,
readout, controller work, injection and capped factory supply are charged.

The failure split per task/profile is .004 sampling +.001 exhausted
accepted-slot retries +.0025 physical faults +.0025 factory readiness =.01.
Numerical operator guards and the unchanged serialized geometry/fault
model are conditional assumptions, not hardware-success certificates.

| Case | p | Selection | Attempts | T/Tdag per attempt | Distance | Peak qubits | Full-cap hours |
|---|---:|---|---:|---:|---:|---:|---:|
| size-n1-native | 0.001 | fewest qubits | 345275 | 1132 | 31 | 712691 | 890.502 |
| size-n1-native | 0.001 | shortest runtime | 345275 | 1132 | 31 | 1093049 | 809.726 |
| size-n1-native | 0.0001 | fewest qubits | 345275 | 1132 | 15 | 157599 | 496.345 |
| size-n1-native | 0.0001 | shortest runtime | 345275 | 1132 | 15 | 241562 | 457.26 |
| size-n1-normalization2-oaa1 | 0.001 | fewest qubits | 55244 | 35778 | 33 | 825083 | 5371.97 |
| size-n1-normalization2-oaa1 | 0.001 | shortest runtime | 55244 | 35778 | 33 | 1256129 | 4937.14 |
| size-n1-normalization2-oaa1 | 0.0001 | fewest qubits | 55244 | 35778 | 15 | 161191 | 2884.15 |
| size-n1-normalization2-oaa1 | 0.0001 | shortest runtime | 55244 | 35778 | 15 | 245154 | 2686.49 |

Each physical report retains all native frontier points and exact macro
ledgers. These rows do not compare with the different Grover/PSSPC
task or establish quantum advantage. Larger-width Taylor schedules
remain physically pending. Offline coefficient/compilation/synthesis
and final statistical-report wall times are excluded from quantum-model
runtime and recorded separately where measured. See report assumptions
and the copied references for geometry, controller, factory and license scope.
