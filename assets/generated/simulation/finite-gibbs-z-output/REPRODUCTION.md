# Fixed ideal Gibbs Z mean: source reproduction

This portable wrapper is **source-only and unexecuted as a downloaded
package**. The accepted original calculation used scientific source commit
`47ffd10368a738f9576053491a56c6d80a10e602`; its saved results are separate
evidence. Source reading and static review of this wrapper do not extend that
execution's validation or accept a new result.

The fixed input is H=Z with coefficient one, beta=1, X/Z proposals normalized
once as X/sqrt(2) and Z/sqrt(2), four clock bits (M=16), product_steps=1,
delta=1/8, R=64 and known-plus initialization by system reset followed by H.
The exact Gibbs observable is Z mean `-tanh(1)`. The diagonal-H/Pauli monomial
restriction makes the spectral coherent correction exactly zero. The
preparation targets this single fixed ideal observable; general thermal
sampling, full-state accuracy and mixing remain outside its scope.

Save the individually linked files in the relative layout listed in
`README.md`, and keep that directory together. Its five files in `circuits/`
are the **exact accepted QASM bytes**. The default wrapper authenticates their
pinned hashes and parses their primitive-only syntax directly into native
Qiskit gates. No platform-dependent angle regeneration or optional QASM
importer is used. It checks each native base class, every canonical decimal
literal's binary64 round-trip, all source chronology and repeated-round
signatures before and after scientific calls. The complete original
angle/interval helpers independently reconstruct their symbolic targets and
outward rate/budget enclosures; saved floating rates or saved interval bounds
are never injected as certificate inputs.

## A new bounded run

The following recipe is prepared for a **future explicit run**. It was not
executed in this proposal. Use Python **3.12.14** and the complete bundled
`requirements-lock.txt`, which pins Qiskit 2.6.2, NumPy 2.5.4, SciPy 1.18.1
and psutil 7.2.2. Create a fresh local environment beside the download; no
system-wide package installation is needed. Both the environment name and
the new output name below must be distinct from existing evidence.

```bash
# Start in the downloaded finite-gibbs-z-output directory.
python3.12 -m venv ../finite-gibbs-venv
../finite-gibbs-venv/bin/python -m pip install -r requirements-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
../finite-gibbs-venv/bin/python -I -B sources/reproduce.py \
  --output-dir ../new-finite-gibbs-reproduction
```

The wrapper refuses a Python patch/package version mismatch, changed pinned
inputs, an existing output directory, an output within the downloaded
payload, a changed primitive signature, an invalid interval domain or an
exceeded direct-call budget. It retains a new failure record when a failure
occurs after output allocation; it never overwrites the accepted records.
Python isolation excludes user-site and Python-path inputs. The wrapper adds
only its authenticated adjacent scientific fragments to its import path.
Thread settings bound the intended local CPU configuration; no GPU,
subprocess, shots executor or remote job is started by the wrapper.

Its new `results.json` and `reproduction-manifest.json` describe this new
run. A true `finite_gibbs_Z_output_bound_observed` means the fixed numerical
diagnostics and interval inequalities were observed in this run. The
wrapper deliberately leaves `finite_gibbs_Z_output_bound_passed=false`,
`whole_invocation_validated=false` and `separate_acceptance_required=true`.
These files need their own scientific and operational review before a new
acceptance claim. They do not replace or modify any original raw flag.

## Fixed work and reference closure

The five source counts are access U: **140**, weak dilation: **1,493**, reset
round: **1,505**, complete R64 preparation/readout: **96,323** and R2
preparation/readout: **3,013**. Every reset round contains six initial
auxiliary resets, the complete dilation, and six terminal auxiliary resets.
The system remains live inside the rounds. Initialization adds reset/H and
final system Z measurement adds one operation.

The direct scientific inventory is fixed:

- Two seven-wire Statevector evolutions, each with 128 entries, reconstruct
  the complete 64 Kraus blocks from the dilation's two system basis columns.
- One independent full 64-by-64 Gaussian access U and all 32 nonjump plus 32
  jump blocks reconstruct the complete reference channel without production
  matrix helpers.
- Two exponent-64 powers of 4-by-4 channel matrices act on the full known-plus
  density, retaining its coherences. The **full R64 primitive stream is never
  evolved**.
- One seven-wire R2 DensityMatrix evolution starts from the 128-by-128
  density of `|1111111><1111111|`, whose only nonzero entry is (127,127)=1.
  System reset/H then prepares known plus. It checks reset/composition
  against two actual channel applications. Its
  source removes only the terminal system measurement: **3,012** primitives
  and **25** resets. SDK internal reset dispatches remain unmeasured.
- One fixed outward interval evaluation follows the complete angle audit.
  Its precision, series limits and 4,096 scalar-request guard remain fixed.

The portable wrapper admits no extra direct evolution, full-state simulator,
matrix power, certificate call or parameter sweep. Explicit reference
products stop at dimension 64 and 4,096 products; mesh, elementwise
exponential and scalar transcendental calls have fixed guards. Source
loading is bounded at 110,000 aggregate primitives; individual QASM/helper
files are pinned and limited to 4 MiB. New combined JSON outputs are limited
to 16 MiB. Cooperative wall/CPU limits are 120 seconds and the checked RSS
limit is 768 MiB. These are cooperative observations around the scientific
work, not a continuous resource or whole-process-lifetime certificate.
The direct portable version/native-class checks do not authenticate the
private execution's native runtime files or reproduce its caller controls.
The package lock fixes versions; it is not a platform-independent numerical
backend certificate.

All actual/reference Kraus, channel, R64 state and R2 joint/reduced data are
retained in the new result. The 18 implementation diagnostics use the fixed
1e-9 tolerance; they are distinct from the outward interval predicates.
There is no clipping, trace renormalization, phase fitting, coherence
deletion or postselection. The interval certificate retains eta and proposal
normalization once. It separates the complete angle bridge, stationary
finite-mesh Z bias, weak digital transient, finite-time transient and
preparation total.

N=16,384 is a **hypothetical** number of fresh independent complete
preparations and ideal Born Z readouts. No shots are drawn. The projected
work formula is `16384*(64*1505+3)`, including every preparation, reset and
terminal readout. The .030 measurement allowance and its Hoeffding failure
bound assume those future iid ideal measurements; they are not observed
sampling confidence or a hardware success guarantee.

## Original sources and packaging changes

`detailed_balance_access.py` and `detailed_balance_channel.py` are exact
original generator sources, downloadable separately from the fixed bounded
wrapper. Their own materialization/reference guards apply to any separately
authored experiment. The default wrapper imports neither generator and does
not run a scalable-generator experiment.

The interval and angle source modules retain every executable function,
class and constant from the accepted source; only their historical
module-level source-pass docstrings were clarified for public packaging.
`source-derivations.json` records that byte boundary. The four functions in
`finite_gibbs_reference.py` are copied **byte-for-byte**, including their
definitions, from the original checker:

| Function | Definition SHA-256 |
|---|---|
| `_matrix_record` | `20b3be3567c4905682669ac5be3c467a66402a55062c8afe51ae103b2562cc44` |
| `_max_entry` | `c2a4f40e9a251b902a951bf71c2d49b075cf4ab6f9d6ff13b93d331dbb63a93b` |
| `_full_reference` | `8eeaaa3f16f6a8b22377b46a45186f44836d60a83b57fa20dbe02fad0363823b` |
| `_superoperator` | `15f93280907c4d22e2ace2d127038dae247ee55a754c9cf81de3d4034630883c` |

Definitions are hashed from `def` through the last body line, excluding the
blank separator before the next definition. The original numerical `_run`
excerpt in `reproduce.py` retains its scientific chronology with exactly two
dispatch substitutions for the portable direct-call guards. Its surrounding
parser, import/setup, guard adapter, result writer and recipe are newly
authored and unexecuted packaging. Original source validation does not
validate those changes.

The original private checker is excluded from the public payload because its
runtime/caller/output controls are specific to that original invocation.
The scientific reference fragments and the portable guard adapter are
explicitly separate artifacts. Original sources and these new files use
the bundled MIT license; exact upstream package credits and licenses are
listed in `UPSTREAM-LICENSES.md`.

Full-state Gibbs accuracy, a general sampler/mixing or uniform KMS theorem,
rotation synthesis, floating-backend error certification, physical faults,
hardware timing, resource advantage and mechanized proof remain excluded.
The surface-code baseline is unchanged. Classical preparation/postprocessing
and implementation costs are described but are not a physical estimate.
