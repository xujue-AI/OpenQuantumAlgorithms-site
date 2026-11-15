# Fixed tiny application-QPE packet

This packet preserves fourteen accepted raw payloads and the three exact
scientific source files used for the fixed tiny check. It is a static public
projection proposal. Copying, hashing and metadata authoring added no numerical
computation, test result, reader result or execution mechanism.

## Fixed task and evidence scope

The specified two-spin model is H = -Z0 - Z1 + X0 X1/4, with the input
actually reset to |00>, tau0 = pi/3, Eref = 3, W = 6 and
U = -exp(-i H pi/3). The three-wire controlled base kernel has 39 primitive
instructions. The reduced QPE comparison uses two-bit words and two rounds;
the admitted controlled powers are 1, 2 and 4. The source/check limits are
six quantum wires, four classical bits, sixteen retained records and at most
10,000 expanded validation instructions per case.

The exact saved checks.json reports passed for fifteen bounded ideal
circuit/instrument cases. The raw files include the full controlled-kernel
and negative controls, deferred/reused/feedback/reprepared record comparisons,
and six fresh-preparation sine/cosine fringes. Their numerical comparisons and
allocation/count records are retained as originally written; they were not
recomputed during this projection. A numerical tolerance is not an interval
certificate, a hardware accuracy guarantee or a full application theorem.

The full m=6/R=41/K=2 application protocol is outside this fixed tiny tranche.
No full application output/confidence, shot sampling, classical phase inference,
rotation synthesis certificate, physical-resource estimate, system-size
scaling result, hardware result or reproduced formal proof is supplied.
Reported validation-map counts are not physical circuit costs or FLOP counts.

## Preserved authoring labels and final-record limitation

The exact source headers retain authoring-era descriptions such as "private"
and "unexecuted" and references to separately approved work. The raw nested
source.scope strings likewise retain "unexecuted bounded source". Those labels
have not been edited: they describe original source-authoring/counting metadata,
while checks.json records the accepted finite numerical-check result. Neither
set of labels expands the evidence to the full application protocol.

The checker retains prescribed record labels, including zero densities when
encountered, without pruning or normalization. The accepted fixture did not
exercise a zero-probability **final** record. The raw
zero_probability_records_retained flag therefore does not establish a dedicated
zero-final-record test or general handling of impossible records in arbitrary
programs. Intermediate zero densities and final-record coverage are different
questions. This limitation remains explicit in public-projection.json.

## Files and attribution

The fourteen raw payloads sit at packet root. The exact used core and
independent reference preserve src/openquantum/ layout; the scientific checker
preserves scripts/ layout. requirements-lock.txt and the original MIT
LICENSE.txt accompany them. The source uses Qiskit for circuit objects and
state/instrument evaluation, and NumPy for independent numerical arrays;
the precise locked dependency versions are preserved in the included lock.
All local scientific source is original MIT work.

The checker is included as exact provenance source, retaining its original
fixed startup/output assumptions. The documented manual recipe below uses
that same source; no wrapper or source controls were added. manifest.json binds
the raw payloads, source, lock, license and this README by filename/bytes/SHA-256;
public-projection.json binds that manifest and states the narrow evidence scope.
No source archive or private operational material is included.

## Reproduce the fixed tiny checks

Use a local copy of this whole packet, preserving the `src/openquantum/` and
`scripts/` paths. Run the following POSIX-shell recipe from that packet's root.
Start with **CPython 3.12.14**: replace `python3.12` below if its executable has
another name. The explicit version check rejects other patch versions.
The commands are documented here but **were not executed during this
projection**.

Create a locked environment and a fresh replay directory outside the packet:

```bash
set -eu
qpe_python312=python3.12
"$qpe_python312" -I -B -c 'import sys; assert sys.version_info[:3] == (3, 12, 14), "CPython 3.12.14 required"'
qpe_packet="$(pwd -P)"
[ "$qpe_packet" != / ]
qpe_workspace="$(cd "$qpe_packet/.." && pwd -P)"
qpe_env="$(mktemp -d "$qpe_workspace/qpe-env.XXXXXXXX")"
"$qpe_python312" -m venv "$qpe_env"
qpe_python="$qpe_env/bin/python"
"$qpe_python" -m pip install -r "$qpe_packet/requirements-lock.txt"
qpe_new_output="$(mktemp -d "$qpe_workspace/qpe-replay.XXXXXXXX")"
qpe_output="$(cd "$qpe_new_output" && pwd -P)"
```

The checker requires exactly fourteen **precreated empty, single-linked regular
files**. This standard-library setup checks a canonical, empty output directory
outside the packet and creates only the fixed filenames, without overwriting:

```bash
"$qpe_python" -I -B - "$qpe_packet" "$qpe_output" <<'PY'
import os
from pathlib import Path
import stat
import sys

packet = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
assert output.is_absolute() and output.resolve(strict=True) == output
assert output.is_dir() and not output.is_symlink()
assert not output.is_relative_to(packet), "Replay must stay outside the packet"
assert not list(output.iterdir()), "Fresh empty output directory required"
names = (
    "kernel.qasm", "kernel-without-reference-z.qasm",
    "kernel-without-odd-evolution.qasm", "retained-deferred.qasm",
    "retained-reused.qasm", "retained-reprepared.qasm", "feedback.qasm",
    "fringe-k1-cos.qasm", "fringe-k1-sin.qasm",
    "fringe-k2-cos.qasm", "fringe-k2-sin.qasm",
    "fringe-k4-cos.qasm", "fringe-k4-sin.qasm", "checks.json",
)
for name in names:
    fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)
    info = (output / name).lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == 0
assert {entry.name for entry in output.iterdir()} == set(names)
print(output)
PY
```

Run the **exact checker** in isolated, bytecode-disabled mode with its absolute
canonical output path and one-thread scientific settings:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  RAYON_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  "$qpe_python" -I -B "$qpe_packet/scripts/check_qpe_application_tiny.py" \
  --output-dir "$qpe_output"
```

A successful checker exits zero and writes the fourteen files, ending with
`checks.json` whose status is `passed` and scope is
`tiny-ideal-circuit-instrument`. Keep these new replay outputs separate from
the accepted files at packet root. **Never point the checker at this packet or
reuse a populated output directory.** A new attempt requires another fresh
directory and the same fourteen empty files.

Compare the resulting structures, case counts and declared numerical tolerance
with the accepted record. Floating results may vary in their last digits across
hosts; the original files/hashes above remain the retained evidence. This recipe
provides no hard wall/RSS limit, CPU-affinity guarantee or process-tree monitor;
use suitable local supervision for execution. The checker's deterministic
six-wire, sixteen-record, power-four and array/instruction guards remain fixed.
Manual replay supplies no full-application confidence, synthesis, physical or
formal claim.
