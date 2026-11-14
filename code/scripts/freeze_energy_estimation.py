#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create a reviewable identity manifest from an actually computed energy job.

Requires the separate completed job's environment and measurement receipts.
Does not run native estimation, change computed data, or replace an existing
manifest. Routine CI uses verify_energy_estimation.py with its reviewed SHA256.
"""
import argparse
import json
from pathlib import Path

from verify_energy_estimation import freeze_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-environment", type=Path, required=True)
    parser.add_argument("--native-measurement", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_record(args.records_dir, args.output, args.native_environment,
                                   args.native_measurement), indent=2))


if __name__ == "__main__":
    main()
