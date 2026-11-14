#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the immutable saved Native QAOA comparison without recomputation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "assets/generated/optimization/word-transport")
    args = parser.parse_args()
    path = Path(__file__).with_name("optimization_word_record_release.py")
    spec = importlib.util.spec_from_file_location("owned_word_record_release", path)
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    exec(compile(path.read_bytes(), str(path), "exec"), release.__dict__)
    try:
        result = release.verify_public_word_transport(args.records_dir)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "saved_record_verification_failed", "error_type": type(error).__name__}))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
