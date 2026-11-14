#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify retained instrument records without synthesis or Native estimation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tail_resource_records import KINDS, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
KIND = "instrument"
BUNDLE_RELATIVE = KINDS[KIND]["directory"]
# Reviewed immutable sidecar identity, independent of its hashed helper sources.
MANIFEST_SHA256 = "0333814a823ebece83910251328b1145fc84e3cfbdec68051af92e78280ba220"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--canonical-root", type=Path, help="Canonical assets root when --root is a partial source preview")
    parser.add_argument("--replay", action="store_true", help="Replay actual frozen source and Clifford+T QASM")
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    args = parser.parse_args()
    result = verify_bundle(args.records_dir or args.root / BUNDLE_RELATIVE, args.root,
                           kind=KIND, expected_sha256=MANIFEST_SHA256, replay=args.replay,
                           max_wall_seconds=args.max_wall_seconds, canonical_root=args.canonical_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
