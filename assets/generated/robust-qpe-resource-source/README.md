# Robust phase-estimation source/export evidence

This original MIT package preserves one **generation-only** example: eight
target qubits plus one phase control, six dyadic levels, twelve fresh-preparation
fringes, and a 5,286-primitive macrocycle. The twelve source-fringe QASMs and
macrocycle QASM are actual exported circuits. Each controlled base call is
explicitly decomposed into three P and two CX gates; input resets, X preparation,
quadrature choice and readout are present. Raw P counts are not T counts.

The associated ledger designs 400 repetitions per fringe: 4,800 fresh trials
and 50,400 controlled base calls. Those shots were **not quantum simulated,
sampled or executed**. No phase result, synthesis certificate, physical qubit,
fault-tolerant runtime, hardware result or formal proof is supplied here.

## Failed and successful invocations

The initial source-only invocation exported the same scientific files but
failed its final 256 MiB self-RSS guard: exit 1, 287.078125 MiB recorded peak.
The repaired producer streamed file hashes and dropped completed source-export
references; its separately approved invocation exited 0, with 101.125 MiB
recorded peak. Both retained manifests were written before their final guards.
The initial raw manifest therefore **does not establish invocation success**.
Use source-export-record.json for the explicit failed/succeeded distinction.
Recorded CPU/source export wall times are not quantum or physical runtimes.

Seventy-nine source/guard tests passed on the earlier producer. They were not
rerun after the producer-only buffer repair; that delta was statically reviewed.
The successful source-only export then preserved all fourteen scientific
QASM/ledger files byte-for-byte. This is construction/export evidence, not new
numerical probability validation. The existing small robust-QPE chapter study
has a separate scope and source history.

## Code, environment and reproduction

The code/ directory retains the repository-shaped Python source, both original
source-test files, project configuration, locked requirements and MIT license.
The saved checker authenticates the closed file/ZIP projection and counts the
literal source statements without importing a quantum package or executing
source/physical producers. Its expected manifest digest is provided separately
by source-export-descriptor.json, which also binds the deterministic download ZIP.

Actual source/export software was Python 3.12.14, Qiskit 2.6.2 (Apache-2.0)
and NumPy 2.5.4 (BSD-3-Clause). The prepared physical adapter references the
recorded rsgridsynth 0.2.2 and Microsoft QDK 1.34.0 models; their synthesis and
physical-estimation paths were **not invoked** in either source-only run.
The exact installed-native-wheel identity limitation remains in the source.

The method references include Kimmel--Low--Yoder robust phase estimation and
the author's 2021 correction. The local conservative confidence derivation
remains distinct from quantum source export and does not reuse the superseded
error-event constants. references.bib retains exact source/package citations.

For a new source-only reproduction, use the existing locked environment,
enter code/, then run:

```bash
python scripts/generate_robust_qpe_resources.py --mode source-only \
  --target-qubits 8 --levels 6 --output-dir ../new-source-export \
  --max-wall-seconds 180
```

The producer requires a fresh destination. It has a 180-second cooperative
limit, 256 MiB observed self-RSS cap and 8 MiB/40-file output cap. The constructor
permits up to 32 target wires and two through six levels under a 200,000-operation
macrocycle cap. Larger generated sources acquire no phase, simulation, synthesis
or physical-validation claim. Input/controlled-access/readout costs remain
in the source ledger. Classical count accumulation, rational inference,
calibration and their hardware costs remain unmeasured in this projection.

The physical code remains a **planned follow-up**. A lower maximum coherent
power by itself is not a total-cost improvement. This package does not reprice
the frozen surface-code baseline or claim complete end-to-end confidence.

Raw operational logs, private caller arguments/paths, grants, plans, remote
coordinates and Git metadata are not included in the public projection.
