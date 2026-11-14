#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the closed saved GQSP folder without importing science.

This new reader checks fixed original and derived-report bytes, case scope,
source/output correspondence and the decoded circuit image. It performs no circuit generation, simulation,
test run, kernel execution, physical estimation or inverse phase finding.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat

MAX_TOTAL_BYTES = 4*1024*1024
MAX_FILE_BYTES = 2*1024*1024
MAX_FILES = 24
PUBLIC_FILES = frozenset({
    "gqsp.py", "generate_simulation_gqsp.py", "requirements-lock.txt", "LICENSE.txt",
    "degree0.qasm", "complex-degree1.qasm", "noncommuting-degree2.qasm",
    "coupled-n2-degree3.qasm", "coupled-n3-degree2.qasm", "identity-signal.qasm",
    "generation-n8.qasm", "generation-n64.qasm", "study.json", "manifest.json",
    "source-notebook.ipynb", "simulation_gqsp.ipynb", "circuit.png",
    "checks-summary.json", "test-case-outcomes.json", "scaling.csv", "generation-scaling.csv", "README.md",
    "verify_saved.py",
})
ORIGINAL_HASHES = {
    "gqsp.py":"a013aeae83968d071f76093647d0c9f68498718a035c215998c8c85d325a68d2",
    "generate_simulation_gqsp.py":"11db399ea75b755187656625bac227f73a0463e7a44d0c275ab7bd26775689b8",
    "requirements-lock.txt":"6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE.txt":"5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
    "study.json":"e46f3ca9b6d0e496cdc509930a5440bf02402f7488558d9204588ec49e845390",
    "manifest.json":"116cc8933398ee5de89b3f429048af1b79b89be3577a1b6d663632b2e92cd683",
    "source-notebook.ipynb":"c978f54e3612c27d5656a54a273acf60c7edf13678c6735d59d6519b65347a70",
    "simulation_gqsp.ipynb":"b900503b6b422af7948e7d2aec991f351046977fec3fb14fea5464ada0a49d6e",
    "circuit.png":"1e2c5e4d05b1bdcffb2d246844587e6e95322fbd47d12e52eefe39251df5a2b7",
}
# Independent anchors for the audited authored reports: the mutable public
# manifest is a member inventory, not the authority for their scientific scope.
FROZEN_DERIVED_HASHES = {
    "checks-summary.json":"e71d27247aaaf7ff7cec1fb94460deb5f720b33a713aaa6958a56da9cd2e0785",
    "test-case-outcomes.json":"3eb9ce4aa4409483467975ac84c9faf1638fd3b21b933a052a0af2aad5e47ec3",
    "scaling.csv":"f675411ffe106ceb804b176df7a24bdc772122517660a3dd8b79a262ead56b5a",
    "generation-scaling.csv":"b6789eceef87fc7a1ddbe2ff9bc5790aa9611772c1e6a3a776df94e6a0b2ff98",
    "README.md":"47d574a58cb59d78759d3d63588ce28ffb5dc38a938910d9ce45c8c69496a939",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_FILE_BYTES:
        raise ValueError("saved member is not a bounded regular file")
    with path.open("rb") as stream:
        data=stream.read(MAX_FILE_BYTES+1)
    if len(data)>MAX_FILE_BYTES:
        raise ValueError("saved member exceeded the read bound")
    return data


def load_json(data):
    def bad_constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(data,parse_constant=bad_constant)


def source_text(cell):
    return "".join(cell["source"]) if isinstance(cell["source"],list) else cell["source"]


def verify(bundle):
    bundle=Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("saved bundle must be a regular directory")
    children=list(bundle.iterdir())
    if len(children)>MAX_FILES or {p.name for p in children}!=PUBLIC_FILES|{"public-manifest.json"}:
        raise ValueError("closed public folder has missing or unexpected members")
    blobs={p.name:read(p) for p in children}
    if sum(map(len,blobs.values()))>MAX_TOTAL_BYTES:
        raise ValueError("public folder exceeds its saved-data cap")
    manifest=load_json(blobs["public-manifest.json"])
    if manifest["schema"]!="openquantum.gqsp.public-manifest.v1" or set(manifest["files"])!=PUBLIC_FILES:
        raise ValueError("public manifest has a wrong schema or member set")
    for name,record in manifest["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}",name):
            raise ValueError("unsafe manifest member")
        if len(blobs[name])!=record["bytes"] or digest(blobs[name])!=record["sha256"]:
            raise ValueError("saved public member hash mismatch")
    for name,expected in ORIGINAL_HASHES.items():
        if digest(blobs[name])!=expected:
            raise ValueError("original science/notebook/image bytes changed")
    for name,expected in FROZEN_DERIVED_HASHES.items():
        if digest(blobs[name])!=expected:
            raise ValueError("frozen public report or scope bytes changed")
    if digest(Path(__file__).read_bytes())!=manifest["files"]["verify_saved.py"]["sha256"]:
        raise ValueError("executing saved reader differs from the bundled reader")
    original=load_json(blobs["manifest.json"])
    if len(original["files"])!=13 or original["source_bytes_preserved"] is not True:
        raise ValueError("original suite closure changed")
    for name,record in original["files"].items():
        if name not in PUBLIC_FILES or record!=manifest["files"][name]:
            raise ValueError("original suite manifest differs from the unchanged members")
    study=load_json(blobs["study.json"])
    tree=ast.parse(blobs["generate_simulation_gqsp.py"].decode())
    case_vector=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CASE_VECTOR" for t in node.targets):
            case_vector=ast.literal_eval(node.value)
    if case_vector is None or len(case_vector)!=8 or [c["case"] for c in study["cases"]]!=case_vector:
        raise ValueError("saved cases differ from the literal original generator")
    if digest(json.dumps(case_vector,sort_keys=True,separators=(',',':'),allow_nan=False).encode())!=study["case_vector_sha256"]:
        raise ValueError("case-vector hash differs")
    numerical=[]
    generated=[]
    residual_names=("p_block_error","q_block_error","unitarity_error","scalar_product_error",
                    "17_point_circle_identity_residual","prepared_vector_error","actual_reset_program_error")
    for record in study["cases"]:
        case=record["case"];n=case["n"];degree=len(case["rotations_pi"])-1
        prep={"zero":0,"one":1,"plus_i":2,"plus_i_x":3}[case["input"]]
        count=degree*(25*n-14)+3*(degree+1)+prep+2*(n+1)
        if record["complete_trial_operations"]!=count or record["total_qubits"]!=n+1:
            raise ValueError("saved complete source count/mapping differs")
        validation=record["validation"]
        if validation["simulation_performed"]:
            numerical.append(case["id"])
            if n+1>4 or validation["status"]!="numerical_small_checks_passed":
                raise ValueError("saved small-check scope differs")
            if any(not math.isfinite(validation[key]) or validation[key]>5e-12 for key in residual_names):
                raise ValueError("saved numerical residual failed its original guard")
        else:
            generated.append(case["id"])
            if validation!={"status":"generation_only","simulation_performed":False}:
                raise ValueError("generation-only scope changed")
    if len(numerical)!=6 or generated!=["generation-n8","generation-n64"]:
        raise ValueError("saved numerical/generation partition changed")
    source=load_json(blobs["source-notebook.ipynb"])
    executed=load_json(blobs["simulation_gqsp.ipynb"])
    projection=lambda book:[(c.get("id"),c["cell_type"],source_text(c)) for c in book["cells"]]
    if projection(source)!=projection(executed):
        raise ValueError("original notebook IDs/types/joined sources changed")
    cells=[c for c in executed["cells"] if c["cell_type"]=="code"]
    if len(cells)!=5 or [c["execution_count"] for c in cells]!=[1,2,3,4,5]:
        raise ValueError("genuine notebook execution counters changed")
    if any(o["output_type"]=="error" for c in cells for o in c["outputs"]):
        raise ValueError("original notebook contains an error output")
    images=[o["data"]["image/png"] for c in cells for o in c["outputs"] if "image/png" in o.get("data",{})]
    if len(images)!=1:
        raise ValueError("original circuit-image count differs")
    image_data="".join(images[0]) if isinstance(images[0],list) else images[0]
    if base64.b64decode(image_data,validate=True)!=blobs["circuit.png"]:
        raise ValueError("circuit PNG differs from the original display output")
    summary=load_json(blobs["checks-summary.json"])
    if summary["dedicated_tests"]["executed_cases"]!=37 or summary["dedicated_tests"]["failures_errors_skips"]!=0:
        raise ValueError("public dedicated-test scope differs")
    tests=load_json(blobs["test-case-outcomes.json"])
    if tests["schema"]!="openquantum.gqsp.public-tests.v1" or len(tests["cases"])!=37 or any(c["status"]!="passed" for c in tests["cases"]):
        raise ValueError("sanitized original test outcomes differ")
    if digest(blobs["test-case-outcomes.json"])!=summary["dedicated_tests"]["public_case_records_sha256"]:
        raise ValueError("public test outcome record hash differs")
    if summary["validation_history"]["initial_full_run_status"]!="failed after passing tests and suite during notebook readiness/process observation":
        raise ValueError("historical failed run was relabeled")
    if summary["validation_history"]["later_notebook_only_continuation_status"]!="passed with unchanged scientific module, generator, case vector and notebook source":
        raise ValueError("separate notebook evidence was relabeled")
    if summary["public_packet_portability"]["adapted_scientific_source"] or summary["public_packet_portability"]["adapted_companion_with_copied_outputs"]:
        raise ValueError("adapted source was silently labeled as original evidence")
    rows=list(csv.DictReader(io.StringIO(blobs["scaling.csv"].decode())))
    if len(rows)!=8 or [r["case"] for r in rows]!=[c["id"] for c in case_vector]:
        raise ValueError("saved resource CSV case vector differs")
    for row,case in zip(rows,study["cases"]):
        if int(row["complete_trial_instructions"])!=case["complete_trial_operations"]:
            raise ValueError("saved CSV instruction count differs")
    size_rows=list(csv.DictReader(io.StringIO(blobs["generation-scaling.csv"].decode())))
    if [(int(r["data_qubits"]),int(r["complete_trial_instructions"])) for r in size_rows]!=[(1,93),(2,246),(3,398),(8,1158),(64,9670)]:
        raise ValueError("original notebook count-only size slice changed")
    if any(r["scope"]!="generation/counts only" or int(r["degree"])!=6 for r in size_rows):
        raise ValueError("count-only size scope was relabeled")
    return {"status":"saved_integrity_passed","scope":"original bytes and bounded saved claims; no new science",
            "public_files":len(blobs),"saved_suite_cases":8,"small_numerical_cases":6,
            "generation_only_cases":2,"genuine_notebook_cells":5,
            "new_scientific_execution":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    here=Path(__file__).resolve().parent
    default=here if (here/"public-manifest.json").is_file() else here.parent/"assets/generated/simulation/gqsp"
    parser.add_argument("--bundle",type=Path,default=default)
    args=parser.parse_args()
    print(json.dumps(verify(args.bundle),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
