# Fixed k=2 path-majority DQI: saved scientific evidence

This packet preserves the one independently accepted **Q22/N10/m10/n9 ideal
primitive pass** from 14 November 2026, its explicitly known-zero initialization
and unconditional assignment probabilities. It contains immutable scientific
source and outputs plus separately curated provenance. It supplies no new run.

The fixed signs are `[1,1,0,0,0,1,0,0,0,0]`. The two length-five paths form a
classically easy ten-cycle. Acceptance covers this input and this finite pass;
the older twelve-wire local component and generation-only thirty-one-wire
`k=3` program retain their own evidence scopes.

## Contents and byte identity

- `code/src/openquantum/` contains the two exact scientific modules and
  `code/scripts/run_optimization_dqi_path_majority_k2.py` the exact driver from
  source commit `d80a2be51c5156283bc4912dc6ed9b48f8543d43`.
- `scientific/source/` contains the original input, actual OpenQASM 3 program,
  complete primitive NDJSON stream and unchanged source ledger.
- `scientific/verification/` contains unchanged `diagnostics.json` and the
  512-assignment unconditional `output-table.json`.
- `saved-full-vectors.zip` contains **only the five original binary vectors**,
  with original `verification/` member names. Each is 67,108,864 bytes; the
  complete uncompressed payload is 335,544,320 bytes. Streaming DEFLATE changes
  storage, not member bytes. No vectors were decoded or reconstructed to
  create this packet.
- `accepted-fixed-run.json` is a curated later acceptance summary.
  `environment-public.json`, [credits](CREDITS.md) and the exact license notices
  identify the original runtime and source provenance.
- `code/scripts/verify_dqi_path_majority_k2_saved.py` is a separate stdlib
  saved-byte checker. `public-manifest.json` pins every packet file and all
  five archive members; its externally recorded SHA-256 authenticates the
  manifest itself.

The exact three scientific files retain their historical **unexecuted**
headers. The source ledger retains `source_only_unexecuted_fixed_k2_profile`
and its original false flags. The raw diagnostics retain
`full_program_ideal_diagnostics_passed=true`, `full_program_validated=false`
and `parent_acceptance_recorded=false`. These earlier endpoints are preserved;
the later root acceptance is a separate record, pinned by SHA-256
`c8f5933fc6a4ddedc4802652ca2833f1b75a0ed429f925d1abf08c3bf466427d`.
This packet does not rewrite an earlier endpoint to manufacture acceptance.

## Inspect saved bytes

From this directory, using Python 3.12 and the manifest digest supplied with
the reviewed release:

```sh
python -I -S -B code/scripts/verify_dqi_path_majority_k2_saved.py \
  --bundle-root . --manifest-sha256 RELEASE_MANIFEST_SHA256
```

The checker authenticates every file, reads each ZIP member once in bounded
chunks and compares its length and SHA-256. It does not extract vectors,
interpret floating values, rescore assignments, recompute diagnostics, import
scientific modules, replay primitives or sample outcomes. Its whole-operation
wall ceiling is 55 seconds. A valid hash check authenticates the saved evidence;
it does not reproduce the scientific computation.

The original driver needs private admission and operational inputs that are
deliberately absent. The repository-shaped source snapshots and unchanged
project lock are provenance, **not a replacement execution environment or a
grant to run the driver**. Private driver receipts, grant/caller payloads,
absolute environment inventories and operational logs are excluded. Their
digests may appear as provenance; their contents do not.

## Scientific and resource limits

The accepted pass applied 2,035 unitary primitives, with 22 known-zero resets,
nine source readout mappings and seven barriers. All 37 RY occurrences remain
ideal and unsynthesized. The output table is a probability marginal over all
auxiliary values; no shots were sampled and no reset-noise channel was modeled.
The saved reported optimum probability is `0.3164062499999637`, compared with
`81/256`; the reported expected score is `7.187499999999175`, compared with
`115/16`. These values are copied from the accepted outputs.

There is no generic DQI quality/repetition guarantee, hard-family advantage,
hardware execution, reproduced formal proof, rotation synthesis, scheduling,
factory model or fault-tolerant cost estimate. The observed process wall and
sampled RSS describe one local CPU job; sampled peaks and the declared buffer
envelope are not physical quantum resources or portable performance promises.
