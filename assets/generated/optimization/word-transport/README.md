# Fixed QAOA word-transport record

This MIT-licensed scientific bundle records one frozen four-vertex, depth-two
QAOA transcript: `qaoa-cycle-4-p2-alternate-seed-7`. It retains all 109 training
groups, one independent holdout, 54 circuit classes, 49 cost-allocation parents,
82,032 complete preparations and 182,646,240 T-state demands.

The actual local QDK 1.34.0 resource-model pass returned 54 individual results on
the fixed distance-15, 243,358-qubit machine. Its serialized conditional model
time is 602,466.0849472 seconds, compared with the frozen baseline's
797,328.306784 seconds. This matched model time and spacetime reduction concerns
only this fixed transcript and machine. It establishes no hardware performance,
global resource optimum, population optimizer success or quantum advantage.

## Reproduce the saved checks

Use the complete downloadable examples package, Python 3.12.14 and its pinned
environment. Verify the complete extracted package before running any code.
Then run:

```bash
python scripts/verify_optimization_word_records.py
python scripts/generate_optimization_word_report.py --output-dir artifacts/word-report
```

The saved verifier uses the Python standard library and authenticated source
snapshots. It checks every exact circuit/phase/word, complete weighted workload,
Native Trace/ISA/T input and typed returned value, parent budget, integer cycle
ceiling and comparison. It never imports QDK or calls Native estimation,
synthesis, an optimizer or a private execution runner. The report command
formats the same saved data; it performs no new reference simulation.

## Evidence and transformation boundary

All 108 candidate circuit files, the three Native scientific transports, the
selected baseline archive and original producer snapshots retain their exact
bytes. `record-transformations.json` distinguishes original hashes from public
metadata hashes. Public JSON omits execution-policy fields; a read-only schema
module restores fixed historical non-execution values in memory solely to check
the exact original hashes. Those values grant no execution authority.

Actual approval, canonical one-use state, process identity, deadline,
permit/journal/configuration and raw-log records remain private.
`execution-summary.json` retains their audited hashes and states the trust
boundary. The persisted parent/journal had a provisional status; terminal
completion was independently established by the matching post-export guardian
seal and successful CLI result, without rewriting those historical bytes.

Factory `runs`, `states` and returned `error_rate` are captured, semantically
unvalidated informational fields. Exactly-one-copy checks and interpreted
ISA/T time, space and output-error costs are separate validated inputs. A fresh
graph restores those evaluated cost identities; the original pipeline/layout
graph is unavailable. Protected geometry, encoder/factory internals, decoder
throughput and hardware calibration retain the model's stated assumptions.

The original full-operator and short density checks are hash-bound recorded
ideal evidence. This bundle does not add a new simulation, kernel proof,
hardware execution, optimizer run or sampled holdout.
