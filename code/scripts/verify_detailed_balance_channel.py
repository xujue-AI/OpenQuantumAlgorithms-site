#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate retained finite zero-C channel records without source execution.

This stdlib data gate verifies immutable provenance and literal circuit/ledger
structure. Producer-attested numerical checks remain historical evidence; no
simulation, mixing certificate, physical estimate or formal proof is added.
"""
from __future__ import annotations
import argparse
from collections import Counter
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

BUNDLE_RELATIVE = "assets/generated/simulation/detailed-balance-channel"
MANIFEST_SHA256 = "57aa8fc1a053e108f3df67e1f02a74a6701a9e724afe4d00057fbd3763544a10"
PRODUCER_COMMIT = "4f54f1944c2b031a0b1bfe105fcc70d3b8d20824"
SOURCE_SHA256 = {
    "src/openquantum/simulation/detailed_balance_access.py": "7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77",
    "src/openquantum/simulation/detailed_balance_channel.py": "99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b",
    "scripts/generate_detailed_balance_channel.py": "f215acff31d87b3ecf8b0e6b7fa7de033c5754791bb15885a78b9737feb704c2",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
SNAPSHOT_SOURCES = {
    "detailed_balance_access.py": "src/openquantum/simulation/detailed_balance_access.py",
    "detailed_balance_channel.py": "src/openquantum/simulation/detailed_balance_channel.py",
    "generate_detailed_balance_channel.py": "scripts/generate_detailed_balance_channel.py",
    "requirements-lock.txt": "requirements-lock.txt", "LICENSE.txt": "LICENSE",
}
CASES = tuple((f"n1-m{m}-beta{beta:g}-delta{delta:g}", m, beta, delta)
              for m, beta, delta in [(m, 1.0, delta) for m in (2,3,4) for delta in (.0625,.125,.25)]
              + [(3,.5,.125),(3,2.0,.125)])
CASE_IDS = tuple(row[0] for row in CASES)
BUNDLE_FILES = frozenset({"manifest.json", "results.json", *SNAPSHOT_SOURCES,
                         *(case+"-"+kind+".qasm" for case in CASE_IDS for kind in ("dilation","reset-round","preparation"))})
MAX_FILE_BYTES = 256*1024
MAX_JSON_BYTES = 128*1024
MAX_BUNDLE_BYTES = 2*1024*1024
MAX_QASM_OPERATIONS = 100000
PRIMITIVES = frozenset({"h","x","s","sdg","t","tdg","p","ry","rz","cx","reset","measure"})
FALSE_FIELDS = ("finite_mesh_exact_kms","general_coherent_correction_implemented",
                "rotations_synthesized","physical_resources_computed","formal_proof_reproduced")


def _equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("Unexpected Gibbs "+label)
    if type(actual) is dict:
        for key in expected: _equal(actual[key],expected[key],label)
    elif type(actual) in (list,tuple):
        for left,right in zip(actual,expected): _equal(left,right,label)


def _number(value,label,minimum=None,maximum=None):
    if type(value) not in (int,float):
        raise ValueError("Nonfinite/untyped Gibbs "+label)
    try: finite=math.isfinite(value)
    except OverflowError as error: raise ValueError("Oversized Gibbs number "+label) from error
    if not finite: raise ValueError("Nonfinite Gibbs "+label)
    if minimum is not None and value<minimum or maximum is not None and value>maximum:
        raise ValueError("Out-of-range Gibbs "+label)
    return value


def _root(value):
    root = Path(value).absolute()
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise ValueError("Gibbs root must be an unaliased directory")
    return root


def _relative(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("Unsafe Gibbs relative path")
    return path


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(root, name, maximum=MAX_FILE_BYTES):
    """Read once through non-following directory descriptors, with a size cap."""
    relative = _relative(name)
    root = _root(root)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | os.O_NONBLOCK
    descriptors = []
    try:
        root_info = root.lstat()
        descriptor = os.open(root, directory)
        descriptors.append(descriptor)
        if _identity(os.fstat(descriptor)) != _identity(root_info):
            raise ValueError("Gibbs root changed before open")
        for part in relative.parts[:-1]:
            info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Linked or non-directory Gibbs ancestor")
            child = os.open(part, directory, dir_fd=descriptor)
            descriptors.append(child)
            if _identity(os.fstat(child)) != _identity(info):
                raise ValueError("Gibbs ancestor changed before open")
            descriptor = child
        info = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("Unsafe or oversized Gibbs file")
        handle = os.open(relative.name, os.O_RDONLY | nofollow | os.O_NONBLOCK, dir_fd=descriptor)
        descriptors.append(handle)
        if _identity(os.fstat(handle)) != _identity(info):
            raise ValueError("Gibbs file changed before open")
        content = bytearray()
        while len(content) <= maximum:
            chunk = os.read(handle, min(65536, maximum + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) != info.st_size or len(content) > maximum or _identity(os.fstat(handle)) != _identity(info):
            raise ValueError("Gibbs file changed during bounded read")
        return bytes(content)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _json(raw):
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Oversized Gibbs JSON")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate Gibbs JSON key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("Nonfinite Gibbs JSON constant")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid Gibbs JSON") from error
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop(); visited += 1
        if depth > 16 or visited > 20000:
            raise ValueError("Gibbs JSON structural limit")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str and len(item) > 8192:
            raise ValueError("Oversized Gibbs JSON string")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("Nonfinite Gibbs JSON number")
        elif type(item) is int and item.bit_length() > 4096:
            raise ValueError("Oversized Gibbs JSON integer")
    return value


def _mapping(value,label):
    if type(value) is not dict: raise ValueError("Gibbs "+label+" must be a mapping")
    return value


def _list(value,label):
    if type(value) is not list: raise ValueError("Gibbs "+label+" must be a list")
    return value


def _inventory(root):
    names=set();total=0;scanned=0
    with os.scandir(root) as entries:
        for entry in entries:
            scanned+=1
            if scanned>40 or entry.name in names: raise ValueError("Duplicate/overlimit Gibbs directory entry")
            info=entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
                raise ValueError("Nonregular Gibbs inventory member")
            names.add(entry.name);total+=info.st_size
            if len(names)>40 or total>MAX_BUNDLE_BYTES:
                raise ValueError("Gibbs inventory cap exceeded")
    _equal(names,set(BUNDLE_FILES),"closed40-file inventory")


def _authenticate(bundle):
    root=_root(bundle);raw=_read(root,"manifest.json",MAX_JSON_BYTES)
    _equal(hashlib.sha256(raw).hexdigest(),MANIFEST_SHA256,"immutable manifest pin")
    _equal(len(raw),6328,"immutable manifest size")
    manifest=_mapping(_json(raw),"manifest")
    _equal(set(manifest),{"status","files","total_payload_bytes","general_sampler_implemented","exact_finite_kms"},"manifest keys")
    _equal(manifest["status"],"checked_small_finite_diagonal_channel","historical status")
    for key in ("general_sampler_implemented","exact_finite_kms"): _equal(manifest[key],False,key)
    _mapping(manifest["files"],"manifest files")
    _equal(set(manifest["files"]),set(BUNDLE_FILES)-{"manifest.json"},"payload keys")
    _inventory(root);data={"manifest.json":raw};total=len(raw)
    for name,entry in manifest["files"].items():
        _mapping(entry,"file pin")
        _equal(set(entry),{"bytes","sha256"},"file pin keys")
        if type(entry["bytes"]) is not int or not 0<entry["bytes"]<=MAX_FILE_BYTES:
            raise ValueError("Invalid Gibbs file byte count")
        if type(entry["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}",entry["sha256"]):
            raise ValueError("Invalid Gibbs file hash")
        total+=entry["bytes"]
        if total>MAX_BUNDLE_BYTES: raise ValueError("Gibbs total byte limit")
        value=_read(root,name,entry["bytes"])
        _equal(len(value),entry["bytes"],"file size")
        _equal(hashlib.sha256(value).hexdigest(),entry["sha256"],"file digest")
        data[name]=value
    _equal(total,1717964,"immutable actual bundle bytes")
    _equal(manifest["total_payload_bytes"],total-len(raw),"payload bytes")
    for snapshot,live in SNAPSHOT_SOURCES.items():
        _equal(hashlib.sha256(data[snapshot]).hexdigest(),SOURCE_SHA256[live],"five-input snapshot "+snapshot)
    return manifest,data


def _sources(source_root):
    root=_root(source_root)
    for name,digest in SOURCE_SHA256.items():
        _equal(hashlib.sha256(_read(root,name)).hexdigest(),digest,"current source "+name)


def _angle(text):
    if len(text)>80: raise ValueError("Gibbs angle token cap")
    pi=re.fullmatch(r"([+-]?)pi(?:/([1-9]\d{0,3}))?",text)
    if pi: return ("pi",Fraction(-1 if pi[1]=="-" else 1,int(pi[2] or 1)))
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",text):
        raise ValueError("Unsupported Gibbs angle expression")
    value=_number(float(text),"QASM angle",-1024,1024)
    return ("binary64",Fraction.from_float(value))


def _neg(angle): return (angle[0],-angle[1])
def _half(angle): return (angle[0],angle[1]/2)
def _op(name,*wires,angle=None): return (name,tuple(wires),angle)


def _qasm(raw,wires,measured):
    try: lines=raw.decode("ascii").splitlines()
    except UnicodeError as error: raise ValueError("Non-ASCII Gibbs QASM") from error
    header=["OPENQASM 3.0;",'include "stdgates.inc";']
    if measured: header.append("bit[1] c;")
    header.append(f"qubit[{wires}] q;")
    _equal(lines[:len(header)],header,"QASM register/header")
    operations=[]
    for line in lines[len(header):]:
        one=re.fullmatch(r"(h|x|s|sdg|t|tdg|reset) q\[(\d+)\];",line)
        cx=re.fullmatch(r"cx q\[(\d+)\], q\[(\d+)\];",line)
        rotation=re.fullmatch(r"(p|ry|rz)\(([^()]+)\) q\[(\d+)\];",line)
        measurement=re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];",line)
        if one: operation=_op(one[1],int(one[2]))
        elif cx:
            if cx[1]==cx[2]: raise ValueError("Repeated Gibbs CX wire")
            operation=_op("cx",int(cx[1]),int(cx[2]))
        elif rotation: operation=_op(rotation[1],int(rotation[3]),angle=_angle(rotation[2]))
        elif measurement and measured and measurement[1]=="0" and measurement[2]=="0": operation=_op("measure",0)
        else: raise ValueError("Unsupported Gibbs QASM primitive")
        if any(wire>=wires for wire in operation[1]): raise ValueError("Out-of-range Gibbs wire")
        operations.append(operation)
        if len(operations)>MAX_QASM_OPERATIONS: raise ValueError("Gibbs QASM operation cap")
    return operations


def _inverse(operations):
    self_inverse={"h","x","cx"};swaps={"s":"sdg","sdg":"s","t":"tdg","tdg":"t"}
    result=[]
    for name,wires,angle in reversed(operations):
        if name in self_inverse: result.append((name,wires,angle))
        elif name in swaps: result.append((swaps[name],wires,angle))
        elif name in ("p","ry","rz"): result.append((name,wires,_neg(angle)))
        else: raise ValueError("Nonunitary Gibbs access prefix")
    return result


def _toffoli(a,b,t):
    return [_op("h",t),_op("cx",b,t),_op("tdg",t),_op("cx",a,t),_op("t",t),
            _op("cx",b,t),_op("tdg",t),_op("cx",a,t),_op("t",b),_op("t",t),
            _op("h",t),_op("cx",a,b),_op("t",a),_op("tdg",b),_op("cx",a,b)]


def _controlled_rotation(name,angle,c,t):
    half=_half(angle)
    return [_op(name,t,angle=half),_op("cx",c,t),_op(name,t,angle=_neg(half)),_op("cx",c,t)]


def _lift(operation,flag):
    name,wires,angle=operation
    if flag in wires: raise ValueError("Flag occurred inside Gibbs access prefix")
    if name=="cx": return _toffoli(flag,wires[0],wires[1])
    target=wires[0]
    if name=="x": return [_op("cx",flag,target)]
    if name=="h":
        return [_op("h",target),_op("cx",flag,target),_op("h",target)]+_controlled_rotation("ry",("pi",Fraction(1,2)),flag,target)
    if name in ("s","sdg","t","tdg","p"):
        constants={"s":Fraction(1,2),"sdg":Fraction(-1,2),"t":Fraction(1,4),"tdg":Fraction(-1,4)}
        angle=("pi",constants[name]) if name!="p" else angle
        half=_half(angle)
        return [_op("p",flag,angle=half),_op("cx",flag,target),_op("p",target,angle=_neg(half)),_op("cx",flag,target),_op("p",target,angle=half)]
    if name in ("ry","rz"): return _controlled_rotation(name,angle,flag,target)
    raise ValueError("Unsupported controlled Gibbs prefix primitive")


def _depth(operations,wires):
    levels=[0]*wires
    for _,targets,_ in operations:
        value=max(levels[t] for t in targets)+1
        for t in targets: levels[t]=value
    return max(levels)


def _integer(value,label,minimum=0):
    if type(value) is not int or not minimum<=value<=1000000: raise ValueError("Invalid Gibbs integer "+label)
    return value


DILATION_KEYS = frozenset(('access_operations', 'beta', 'clock_bits', 'controlled_inverse_operations', 'delta', 'environment_outcomes', 'eta', 'finite_mesh_exact_kms', 'formal_proof_reproduced', 'general_coherent_correction_implemented', 'hamiltonian', 'milestone', 'n_system', 'nonjump_inverse', 'physical_resources_computed', 'preflight_upper_bound', 'product_steps', 'proposal_bits', 'proposal_count', 'proposals', 'rotations_synthesized', 'source_operations', 'weak_flag', 'weak_operations', 'zero_correction_reason'))
VALIDATION_KEYS = frozenset(('actual_vs_independent_channel_frobenius_error', 'actual_vs_independent_kraus_max_entry_error', 'bloch_contraction_norm', 'channel_power_scope', 'continuum_generator_gibbs_residual', 'contraction_scope', 'decay_commutator', 'depth', 'finite_first_order_channel_error', 'finite_generator_gibbs_residual', 'finite_round_gibbs_residual', 'finite_semigroup_one_round_error', 'finite_stationary_channel_residual', 'finite_stationary_gibbs_trace_distance', 'finite_stationary_populations', 'metadata', 'minimum_choi_eigenvalue', 'nonzero_environment_nojump_weight', 'primitive_counts', 'remaining', 'rounds', 'spanning_input_channel_power_checks', 'status', 'trace_preservation_max_entry_error'))


def _meta(meta,clock,beta,delta):
    if type(meta) is not dict: raise ValueError("Gibbs metadata must be a mapping")
    expected=set(DILATION_KEYS)
    if meta.get("milestone")=="finite_diagonal_weak_reset_channel":
        expected|={"initialization","resets_per_round"}
    elif meta.get("milestone")=="finite_diagonal_preparation_source":
        expected|={"initialization","resets_per_round","rounds","initial","readout","full_round_operations","total_resets","mixing_certificate","repetitions"}
    _equal(set(meta),expected,"metadata keys")
    for key,value in (("n_system",1),("clock_bits",clock),("proposal_count",2),("proposal_bits",1),
                      ("beta",beta),("delta",delta),("weak_flag",clock+2),("product_steps",1),
                      ("hamiltonian",[[1.0,"Z"]]),("proposals",["X","Z"])):
        _equal(meta[key],value,key)
    for key in FALSE_FIELDS: _equal(meta[key],False,key)
    for key in ("access_operations","weak_operations","controlled_inverse_operations","source_operations","preflight_upper_bound"):
        _integer(meta[key],key,1)
    eta=_number(meta["eta"],"eta",0,2)
    if eta==0 or eta*delta>1: raise ValueError("Invalid retained weak-step domain")
    _equal(meta["zero_correction_reason"],"I/Z Hamiltonian; each Pauli proposal has a fixed computational permutation, so every B_dagger_B is diagonal","zero-C restriction")
    _equal(meta["environment_outcomes"],"all frequency/proposal and both weak flag outcomes retained","unconditioned outputs")
    _equal(meta["nonjump_inverse"],"complete primitive U inverse conditional on flag zero, including global phase","complete inverse scope")


def _circuit_records(case,data,clock,beta,delta):
    dilation,reset,prep=case["dilation"],case["reset_round"],case["preparation"]
    for meta in (dilation,reset,prep): _meta(meta,clock,beta,delta)
    _equal(dilation["milestone"],"finite_diagonal_weak_dilation","dilation milestone")
    _equal(reset["milestone"],"finite_diagonal_weak_reset_channel","reset milestone")
    _equal(prep["milestone"],"finite_diagonal_preparation_source","preparation milestone")
    wires=clock+3;flag=wires-1
    ops=_qasm(data[case["name"]+"-dilation.qasm"],wires,False)
    round_ops=_qasm(data[case["name"]+"-reset-round.qasm"],wires,False)
    prep_ops=_qasm(data[case["name"]+"-preparation.qasm"],wires,True)
    gu,gw,gc=(dilation[key] for key in ("access_operations","weak_operations","controlled_inverse_operations"))
    _equal(gw,2*(1<<clock),"weak multiplexor operations")
    _equal(dilation["source_operations"],len(ops),"dilation length")
    _equal(gu+gw+gc,len(ops),"dilation block accounting")
    _equal(dilation["preflight_upper_bound"],17*gu+2*(1<<clock)+4,"recorded materialization bound")
    prefix,weak,controlled=ops[:gu],ops[gu:gu+gw],ops[gu+gw:]
    _equal(prefix[0],_op("h",clock+1),"normalized proposal preparation")
    gray=[j^(j>>1) for j in range(1<<clock)]
    for index in range(1<<clock):
        name,targets,angle=weak[2*index]
        _equal((name,targets),("ry",(flag,)),"weak flag target")
        if angle is None: raise ValueError("Missing weak angle")
        changed=gray[index]^gray[(index+1)%len(gray)]
        _equal(weak[2*index+1],_op("cx",1+changed.bit_length()-1,flag),"weak Gray-cycle closure")
    expected_control=[_op("x",flag)]
    for operation in _inverse(prefix): expected_control.extend(_lift(operation,flag))
    expected_control.append(_op("x",flag))
    _equal(controlled,expected_control,"complete flag-zero controlled inverse chronology")
    _equal(gc,len(expected_control),"controlled inverse count")
    env=[_op("reset",wire) for wire in range(1,wires)]
    _equal(round_ops,env+ops+env,"unconditioned reset chronology")
    _equal(reset["source_operations"],len(round_ops),"reset source length")
    _equal(reset["resets_per_round"],2*(wires-1),"environment reset count")
    _equal(reset["initialization"],"arbitrary caller system; explicit environment reset before and after round","reset input scope")
    _equal(prep["initialization"],"explicit known system reset and optional X/H; environment initialized and discarded each round","known preparation input scope")
    _equal(prep["rounds"],8,"emitted K8")
    _equal(prep["initial"],"zero","known initial input")
    _equal(prep["readout"],True,"complete readout")
    _equal(prep["full_round_operations"],len(round_ops),"full round count")
    _equal(prep["total_resets"],1+8*len(env)*2,"total resets")
    _equal(prep_ops,[_op("reset",0)]+round_ops*8+[_op("measure",0)],"complete K8 preparation/readout chronology")
    _equal(prep["source_operations"],len(prep_ops),"preparation source length")
    _equal(case["preparation_primitive_counts"],dict(Counter(row[0] for row in prep_ops)),"preparation counts")
    _equal(case["preparation_depth"],_depth(prep_ops,wires),"preparation dependency depth")
    for key in set(dilation)-{"milestone","source_operations"}:
        _equal(reset[key],dilation[key],"reset inherited "+key)
        _equal(prep[key],dilation[key],"preparation inherited "+key)
    _equal(prep["mixing_certificate"],"none; bounded channel-power checks may be supplied separately","mixing exclusion")
    _equal(prep["repetitions"],"each observable draw reruns all rounds; no accepted-only branch or algorithm retry factor","fresh preparation contract")
    return ops,prep_ops,wires


def _validation(case,ops,wires):
    record=_mapping(case["check"],"retained validation")
    _equal(set(record),set(VALIDATION_KEYS),"retained validation keys")
    _equal(record["status"],"checked_small_finite_diagonal_weak_channel","retained check status")
    _equal(record["metadata"],case["dilation"],"check/dilation metadata")
    _equal(record["primitive_counts"],dict(Counter(row[0] for row in ops)),"check primitive counts")
    _equal(record["depth"],_depth(ops,wires),"dilation dependency depth")
    for key in ("actual_vs_independent_kraus_max_entry_error","actual_vs_independent_channel_frobenius_error",
                "trace_preservation_max_entry_error","continuum_generator_gibbs_residual","decay_commutator"):
        _number(record[key],key,0,1e-10)
    _number(record["minimum_choi_eigenvalue"],"retained Choi eigenvalue",-1e-10)
    if _number(record["nonzero_environment_nojump_weight"],"retained all nonjump labels",0)==0:
        raise ValueError("Dropped retained nonzero environment nojump weight")
    for key in ("finite_first_order_channel_error","finite_semigroup_one_round_error","finite_generator_gibbs_residual",
                "finite_round_gibbs_residual","finite_stationary_channel_residual"):
        _number(record[key],key,0)
    _number(record["finite_stationary_gibbs_trace_distance"],"finite mesh bias",0,1)
    _number(record["bloch_contraction_norm"],"numerical contraction",0,1+1e-10)
    populations=_list(record["finite_stationary_populations"],"stationary populations")
    _equal(len(populations),2,"retained stationary dimension")
    for value in populations: _number(value,"stationary population",0,1)
    if abs(sum(populations)-1)>1e-12: raise ValueError("Unnormalized retained stationary populations")
    _equal(record["rounds"],64,"recorded K64 channel powers")
    _equal(record["channel_power_scope"],"iterate actual full one-round channel after independent Kraus checks; no full statevector/density simulation of the large repeated circuit","channel power scope")
    _equal(record["contraction_scope"],"floating-point small-channel diagnostic with stationary-residual accumulation; no certified rounding enclosure or general mixing certificate","uncertified contraction scope")
    _equal(record["remaining"],["general noncommuting coherent-correction circuit","uniform finite mesh/window/aliasing error bound",
                              "general mixing guarantee","rotation synthesis and complete frozen surface-code model","mechanized proof bridge"],"unfinished endpoint")
    _list(record["spanning_input_channel_power_checks"],"spanning inputs")
    _equal(len(record["spanning_input_channel_power_checks"]),4,"spanning inputs")
    for row in record["spanning_input_channel_power_checks"]:
        _mapping(row,"spanning output")
        _equal(set(row),{"trace_error","minimum_eigenvalue","gibbs_trace_distance","finite_stationary_trace_distance","numerical_contraction_estimate_to_finite_stationary"},"spanning output keys")
        _number(row["trace_error"],"retained trace error",0,1e-10)
        _number(row["minimum_eigenvalue"],"retained output positivity",-1e-10,1+1e-10)
        for key in ("gibbs_trace_distance","finite_stationary_trace_distance","numerical_contraction_estimate_to_finite_stationary"):
            _number(row[key],key,0,1+1e-10)


def _readout(case,prep_ops):
    record=_mapping(case["logical_readout_contract"],"readout contract")
    _equal(set(record),{"observable","mean_additive_sampling_error","sampling_failure_probability","independent_complete_preparations",
                        "hoeffding_formula","total_primitive_counts","total_rounds","synthesis_and_physical_faults_included",
                        "continuum_gibbs_accuracy_certificate","exclusions"},"readout keys")
    _equal(record["observable"],"Z on q0 after the finite circuit's explicit K rounds","finite K8 observable")
    _equal(record["mean_additive_sampling_error"],.1,"finite-output sampling epsilon")
    _equal(record["sampling_failure_probability"],.05,"sampling failure")
    with localcontext() as context:
        context.prec=80
        trials=int((Decimal(2)*Decimal(40).ln()/Decimal("0.1")**2).to_integral_value(rounding=ROUND_CEILING))
    _equal(trials,738,"fixed Hoeffding calculation")
    _equal(record["independent_complete_preparations"],trials,"projected complete preparations")
    _equal(record["hoeffding_formula"],"ceil(2*ln(2/failure)/epsilon^2), Z in[-1,1]","sampling formula")
    _equal(record["total_primitive_counts"],{name:count*trials for name,count in Counter(row[0] for row in prep_ops).items()},"full readout projection")
    _equal(record["total_rounds"],8*trials,"projected K8 rounds")
    for key in ("synthesis_and_physical_faults_included","continuum_gibbs_accuracy_certificate"):
        _equal(record[key],False,key)
    _equal(record["exclusions"],["arbitrary rotation synthesis","physical code/factory/reset/controller model",
                               "general mixing and finite mesh certificate","classical preprocessing timing"],"resource exclusions")


def _records(manifest,data):
    report=_mapping(_json(data["results.json"]),"results report")
    _equal(set(report),{"status","paper","source_commit","source_dirty","source_revision_is_exact","environment","caps","source_files",
                        "environment_lock_sha256","wall_seconds","cases","scope","exact_finite_kms","general_sampler_implemented",
                        "rotations_synthesized","physical_resources_computed","formal_proof_reproduced"},"report keys")
    _equal(report["status"],"checked_small_finite_diagonal_channel","report status")
    _equal(report["source_commit"],PRODUCER_COMMIT,"historical producer")
    _equal(report["source_dirty"],False,"historical source dirty")
    _equal(report["source_revision_is_exact"],True,"historical exact source")
    _equal(report["environment"],{"python":"3.12.14","qiskit":"2.6.2","numpy":"2.5.4","scipy":"1.18.1","device":"CPU","seed":None,"random_sampling_performed":False},"historical environment")
    _equal(report["paper"],{"doi":"10.1038/s41586-025-09583-x","weak_round_equations":[28,29],"spectral_correction_equation":19,"coherent_correction":"exactly zero for the declared diagonal-Pauli family"},"paper/restriction")
    _equal(report["caps"],{"source_rounds":256,"primitive_operations_each":100000,"reference_system_wires":1,"reference_clock_bits":4,"reference_total_wires":7,"output_bytes":8388608},"producer caps")
    _equal(report["source_files"],{name:manifest["files"][name] for name in SNAPSHOT_SOURCES if name.endswith(".py")},"source3 hashes")
    _equal(report["environment_lock_sha256"],SOURCE_SHA256["requirements-lock.txt"],"lock binding")
    _number(report["wall_seconds"],"historical generation wall",0)
    _equal(report["scope"],"complete finite weak reset channel in a restricted zero-C family; all outcomes retained; channel-power checks separately labeled","recorded scope")
    for key in ("exact_finite_kms","general_sampler_implemented","rotations_synthesized","physical_resources_computed","formal_proof_reproduced"):
        _equal(report[key],False,key)
    _list(report["cases"],"retained cases")
    _equal(len(report["cases"]),11,"eleven retained cases")
    total_ops=0
    for case,(name,clock,beta,delta) in zip(report["cases"],CASES):
        _mapping(case,"retained case")
        _equal(set(case),{"name","terms","proposals","dilation","reset_round","preparation","preparation_primitive_counts",
                          "preparation_depth","numerical_checks_performed","check","logical_readout_contract"},"case keys")
        _equal(case["name"],name,"case ordering")
        _equal(case["terms"],[[1.0,"Z"]],"fixed H")
        _equal(case["proposals"],["X","Z"],"fixed normalized proposals")
        _equal(case["numerical_checks_performed"],True,"historical numerical evidence")
        ops,prep_ops,wires=_circuit_records(case,data,clock,beta,delta)
        _validation(case,ops,wires);_readout(case,prep_ops)
        total_ops+=len(ops)+len(prep_ops)+case["reset_round"]["source_operations"]
        if total_ops>200000: raise ValueError("Gibbs aggregate QASM operation cap")
    return report


def verify_bundle(bundle,*,source_root=None):
    manifest,data=_authenticate(bundle)
    if source_root is not None: _sources(source_root)
    _records(manifest,data)
    return {"status":"authenticated_saved_finite_zero_C_channel_records","manifest_sha256":MANIFEST_SHA256,
            "files":40,"bytes":1717964,"cases":11,"qasm_circuits":33,
            "current_source_inputs_checked":5 if source_root is not None else 0,
            "recorded_numerical_evidence_authenticated":True,"new_numerical_simulations":0,"native_estimator_calls":0,
            "physical_resource_estimate":None,"formal_certificate":False,"uniform_Gibbs_accuracy_certificate":False}


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir",type=Path,default=root/BUNDLE_RELATIVE)
    parser.add_argument("--source-root",type=Path,default=root)
    arguments=parser.parse_args()
    print(json.dumps(verify_bundle(arguments.records_dir,source_root=arguments.source_root),sort_keys=True))

if __name__=="__main__": main()
