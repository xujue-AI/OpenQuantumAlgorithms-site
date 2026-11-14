#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify complete saved workloads without phase fitting, synthesis or Native."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tail_resource_records import KINDS, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
KIND = "complete"
BUNDLE_RELATIVE = KINDS[KIND]["directory"]
MANIFEST_SHA256 = "86af1d97bad4ac1f4d8c5191d3e546f69ca82a5121bd8de7d7307081c70dc0b1"


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
