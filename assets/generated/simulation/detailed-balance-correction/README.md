# Saved finite correction toy

This bundle retains one executed one-spin correction block and one heralded
known-input step. It preserves all eighteen v2 files byte-for-byte, including
six primitive OpenQASM 3 sources, seven Python sources, the environment lock,
MIT license, numerical records and nested manifest/sidecar. All 32 Kraus
outcomes and raw complex entries remain in `v2/results.json`.

The fixed input is `H=0.6 Z+0.8 X` in that order, proposal `Z`, `beta=1`,
two clock bits, one product step and `tau=1/8`. The source retains the actual
finite-Trotter Gaussian decay. The independent correction reference applies
Chen et al., *Efficient quantum thermal simulation*, Nature 646 (2025),
DOI `10.1038/s41586-025-09583-x`, Eq. 19, in the known energy basis.

The selected correction block is `C/a`, where
`a=alpha*tanh(beta/2)/4`; the LCU success block is
`(I-i*tau*C)/(1+tau*a)`. Success requires all five auxiliaries zero. Every
other auxiliary outcome has its own retained Kraus operator. The complete
known-plus trial includes all six resets, H preparation, the whole controlled
oracle and every wire's measurement. The source counts are 213 centered,
228 correction, 2,013 controlled correction, 2,016 LCU, 2,017 coherent prefix
and 2,029 complete trial primitives.

The recorded check used five Statevector evolutions of lengths
`[32,32,64,64,64]`, twelve tiny exponentials and two interval matrix powers.
All 28 check predicates passed. The original external observed sequence also
exited zero; `completion.json` gives the public-safe timing, work and scope
summary. Its monitoring observations are sampled and do not establish a
continuous memory quota or complete process-lifetime proof. The timings are
classical validation work, not quantum hardware runtime.

The v1 record remains a distinct failure. `v1/failure.json` preserves its
public-safe outcome and `v1/centered_decay.qasm` preserves its only saved
payload. Numerical success, the 5/12/2 runtime counters and the remaining
results were not retained for v1. The later v2 pass does not change that status.

## Saved-only reader

The standalone download contains this `detailed-balance-correction/` folder
and `verify_detailed_balance_correction_saved.py` beside it. From the unpacked
download directory, the prepared saved-reader invocation is:

```bash
python -B verify_detailed_balance_correction_saved.py
```

The reader uses only the Python standard library. It authenticates the closed
projection manifest, all original payload bytes, the nested v2 identity and
literal saved contracts. It performs no producer import, circuit evolution,
matrix exponential, matrix power, sampling or installation. Successful reading
authenticates saved evidence; it adds no numerical validation.

The repository invocation can additionally bind the retained source copies to
the selected public source tree and inspect the standalone ZIP:

```bash
python -B scripts/verify_detailed_balance_correction_saved.py \
  --bundle assets/generated/simulation/detailed-balance-correction \
  --source-root . \
  --download assets/generated/simulation/detailed-balance-correction-download.zip
```

## Source reproduction boundary

The seven original Python files, lock and license in `v2/` form the complete
adjacent source closure for the fixed CLI. The driver authenticates bare-module
snapshots and avoids borrowing parent Git provenance or package initialization.
This adjacent scientific mode is **prepared/unexecuted**: the measured v2
invocation used repository-selected copies and external run control. No portable
scientific wrapper or downloaded-source replay has been executed.

The fixed CLI requires Python 3.12.14, Qiskit 2.6.2, NumPy 2.5.4 and SciPy
1.18.1, bytecode/user-site loading disabled, seven CPU thread settings set to
one, and an absolute absent output path whose parent exists. Its checkpoints
are cooperative; whole-invocation resource and cleanup acceptance requires
separate reviewed control. New outputs do not extend the recorded validation.
See `UPSTREAM-LICENSES.md` for exact upstream credits and license metadata.

Known-input retries reset and reprepare the whole trial. They cannot restore
an unknown evolving input; chained successes require whole-trajectory restart
or a separately implemented recovery construction. The N/R sampling and retry
projection in the results draws no shots. Ideal bias formulas, floating-point
comparison tolerances, synthesis error and hardware error remain distinct.

The old normalized `D/alpha` finite-versus-exact-H Frobenius error is about
0.0942194. The physical D error is about 0.0921037, obtained by multiplication
by `alpha=0.9775450868978823`. This is a units/normalization distinction, not
an improved approximation or a changed target.

General-H correction and correction evolution, full channel/sampler,
continuum/KMS and mixing/output guarantees, rotation synthesis, physical
resources and formal proof remain unimplemented. All general and certification
flags stay false; the frozen surface-code baseline is unchanged.
