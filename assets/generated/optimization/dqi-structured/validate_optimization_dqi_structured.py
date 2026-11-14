#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicit fresh n=2/3 structured DQI coherent validation and small exports.

Keep this CLI beside optimization_dqi_structured_validation.py and
optimization_dqi_structured.py, or run it from the repository scripts directory.
The ordinary Python 3.12 lock supplies Qiskit 2.6.2 and NumPy 2.5.4. This command
validates the new clean-workspace family only; it never invokes the preserved
default proof, a larger simulation, synthesis, sampling, an optimizer or GPU.

The default allowance is ten seconds, with a thirty-second maximum. On POSIX
the hard alarm is anchored at script entry and remains active through parsing,
source loading, both checks, exports and final output. Timeout exits 124; a
fresh result's initial receipt remains incomplete until all checks finish.
The module API supplies cooperative deadline checks without a private recipe,
private environment variable or namespace hook.

The default cumulative export ceiling is 192 KiB, with a hard one-MiB maximum;
smaller requested ceilings only tighten the output allowance. Atomic receipt
replacement counts both the old receipt and its new temporary file.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys


MAX_SECONDS = 30.0
MAX_SOURCE_BYTES = 1 << 20
VALIDATOR_NAME = "optimization_dqi_structured_validation.py"


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Keep the ordinary validator source as a regular file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("The validator source exceeds its fixed byte ceiling")
    return raw


def _validator_source():
    here = Path(__file__).resolve().parent
    for candidate in (here / VALIDATOR_NAME, here.parent / "src/openquantum" / VALIDATOR_NAME):
        if candidate.is_file():
            return candidate, _read(candidate)
    raise FileNotFoundError("Place this CLI beside the validator/core or in repository scripts")


def _load(path, raw):
    name = "_structured_dqi_small_validator_" + hashlib.sha256(raw).hexdigest()[:16]
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise RuntimeError("Cannot create the new validator module specification")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("time-limit-seconds must be a finite number in (0,30]")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("time-limit-seconds must be a finite number in (0,30]") from error
    if not math.isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError("time-limit-seconds must be a finite number in (0,30]")
    return value


def _alarm(deadline):
    if os.name == "posix" and hasattr(signal, "ITIMER_REAL"):
        signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))


def main(argv=None):
    # This preliminary alarm also bounds parsing and validator-source loading.
    _alarm(ENTRY_STARTED + MAX_SECONDS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    parser.add_argument("--max-output-bytes", type=int, default=192 << 10)
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)
    try:
        seconds = _seconds(args.time_limit_seconds)
        deadline = ENTRY_STARTED + seconds
        if time.monotonic() >= deadline:
            raise TimeoutError("Structured validation entry allowance expired")
        _alarm(deadline)
        path, raw = _validator_source()
        module = _load(path, raw)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("No validation work allowance remains")
        record = module.validate_small_structured(args.output_dir,
                    time_limit_seconds=remaining, max_output_bytes=args.max_output_bytes,
                    source_commit=args.source_commit)
        if _read(path) != raw:
            raise RuntimeError("The recorded validator source changed during the call")
        if time.monotonic() >= deadline:
            raise TimeoutError("Final validation output exceeded its entry allowance")
        print(json.dumps({"status": record["status"], "cases": [case["n"] for case in record["cases"]],
                          "coherent_validation_performed": record["coherent_validation_performed"],
                          "old_default_reexecuted": False}), flush=True)
        if time.monotonic() >= deadline:
            raise TimeoutError("Final validation output exceeded its entry allowance")
        return 0
    except (ValueError, OSError, RuntimeError, AssertionError, TimeoutError, ImportError) as error:
        print("Structured DQI validation failed: " + str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    # Keep the POSIX hard alarm live through process termination. No private
    # deadline is inherited or required by this ordinary command.
    raise SystemExit(main())
