#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write saved-resource identity sidecars after independent record review.

This does not generate circuits or execute the scientific producers. Existing
61 record files, producer modules and Frozen5 are read only.
Failed candidates remain unvalidated; no mutable named target is removed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from tail_resource_records import (KINDS, PRODUCER_COMMIT, SIDECAR, TRUST_BOUNDARIES,
                                   _check_directory_chain, _directory_chain, _identity,
                                   collect_inputs, digest, inventory, read_regular, require)

ROOT = Path(__file__).resolve().parents[1]


def freeze(kind, root):
    records_dir = root / KINDS[kind]["directory"]
    records, sources, _ = collect_inputs(kind, records_dir, root, with_sidecar=False)
    record = {"schema": "openquantum-saved-tail-resource-v1", "kind": kind,
              "producer_commit": PRODUCER_COMMIT, "files": inventory(records),
              "sources": inventory(sources), "scope": KINDS[kind]["scope"],
              "trust_boundaries": TRUST_BOUNDARIES}
    data = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    # Exclusive creation keeps refreezing a deliberate reviewed operation.
    handles, edges = _directory_chain(records_dir)
    directory = handles[-1]
    descriptor = None
    try:
        descriptor = os.open(SIDECAR, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                             0o644, dir_fd=directory)
        created = os.fstat(descriptor)
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            require(count > 0, "sidecar write did not advance")
            written += count
        after_write = os.fstat(descriptor)
        _check_directory_chain(edges)
        final_records, final_sources, _ = collect_inputs(kind, records_dir, root)
        require(inventory(final_records) == record["files"] and inventory(final_sources) == record["sources"],
                "inputs changed during sidecar freezing")
        # Keep the created descriptor live through audit and readback. This
        # pins its inode and never authorizes deleting a mutable named target.
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = bytearray()
        while True:
            block = os.read(descriptor, min(65536, len(data) + 1 - len(observed)))
            if not block:
                break
            observed.extend(block)
            require(len(observed) <= len(data), "created sidecar grew during freezing")
        require(bytes(observed) == data and _identity(os.fstat(descriptor)) == _identity(after_write),
                "created sidecar changed during freezing")
        require(read_regular(records_dir, SIDECAR) == data, "named sidecar changed during freezing")
        present = os.stat(SIDECAR, dir_fd=directory, follow_symlinks=False)
        require((present.st_dev, present.st_ino) == (created.st_dev, created.st_ino),
                "created sidecar identity changed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for handle in reversed(handles):
            os.close(handle)
    return {"kind": kind, "files": len(records), "sha256": digest(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--kind", choices=KINDS, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.kind, args.root.absolute()), sort_keys=True))


if __name__ == "__main__":
    main()
