#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded given-angle GQSP producer prototype. No run is yet validated.

Only explicit primitive unitary signals are generated. --suite has six tiny
numerical cases and two generation-only cases; it never fits inverse phases,
calls an estimator or claims complete Hamiltonian simulation.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/openquantum/simulation/gqsp.py"
SIBLING_MODULE = Path(__file__).with_name("gqsp.py")
CANONICAL_SCRIPT = ROOT / "scripts/generate_simulation_gqsp.py"
if MODULE_PATH.is_file() and Path(__file__).resolve()==CANONICAL_SCRIPT.resolve() and not SIBLING_MODULE.is_file():
    REPOSITORY = True
    sys.path.insert(0,str(ROOT/"src"))
    from openquantum.simulation import gqsp
    if Path(gqsp.__file__).resolve()!=MODULE_PATH.resolve():
        raise ImportError("repository GQSP import did not select the frozen source")
else:
    REPOSITORY = False
    MODULE_PATH = SIBLING_MODULE
    if not MODULE_PATH.is_file():
        raise ImportError("a detached GQSP generator requires its shipped sibling gqsp.py")
    loader = importlib.util.spec_from_file_location("openquantum_gqsp_snapshot", MODULE_PATH)
    gqsp = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = gqsp
    loader.loader.exec_module(gqsp)

MAX_OUTPUT_BYTES = 32*1024*1024
MAX_FILES = 40
MAX_INPUT_BYTES = 256*1024
MAX_CASES = 8
CASE_VECTOR = [
    {"id":"degree0", "n":1, "y_pi":["1/8"], "z_pi":["1/9"], "chi_pi":"1/6", "rotations_pi":[["1/7","1/3","-1/5"]], "input":"zero", "simulate":True},
    {"id":"complex-degree1", "n":1, "y_pi":["0"], "z_pi":["1/4"], "chi_pi":"1/4", "rotations_pi":[["1/4","0","0"],["1/4","0","1/2"]], "input":"plus_i", "simulate":True},
    {"id":"noncommuting-degree2", "n":1, "y_pi":["-1/9"], "z_pi":["1/7"], "chi_pi":"-1/5", "rotations_pi":[["1/6","-1/7","1/9"],["1/5","1/3","-1/6"],["-1/4","1/8","1/11"]], "input":"one", "simulate":True},
    {"id":"coupled-n2-degree3", "n":2, "y_pi":["1/8","-1/9"], "z_pi":["-1/7","1/11"], "chi_pi":"1/13", "rotations_pi":[["1/7","1/3","-1/11"],["-1/5","1/8","1/12"],["1/4","-1/9","1/5"],["1/6","1/10","-1/7"]], "input":"plus_i_x", "simulate":True},
    {"id":"coupled-n3-degree2", "n":3, "y_pi":["1/8","-1/9","1/10"], "z_pi":["-1/7","1/11","-1/12"], "chi_pi":"-1/13", "rotations_pi":[["1/6","1/7","-1/9"],["-1/5","1/8","1/6"],["1/4","-1/11","1/12"]], "input":"plus_i_x", "simulate":True},
    {"id":"identity-signal", "n":1, "y_pi":["0"], "z_pi":["0"], "chi_pi":"0", "rotations_pi":[["1/6","1/7","1/8"],["1/5","-1/9","1/11"],["1/4","1/12","-1/13"]], "input":"plus_i", "simulate":True},
    {"id":"generation-n8", "n":8, "y_pi":"1/8", "z_pi":"-1/9", "chi_pi":"1/6", "rotations_pi":[["1/7","1/3","-1/11"],["-1/5","1/8","1/12"],["1/4","-1/9","1/5"],["1/6","1/10","-1/7"],["-1/8","1/9","1/10"],["1/9","-1/11","1/12"],["1/10","1/12","-1/13"]], "input":"plus_i_x", "simulate":False},
    {"id":"generation-n64", "n":64, "y_pi":"1/8", "z_pi":"-1/9", "chi_pi":"1/6", "rotations_pi":[["1/7","1/3","-1/11"],["-1/5","1/8","1/12"],["1/4","-1/9","1/5"],["1/6","1/10","-1/7"],["-1/8","1/9","1/10"],["1/9","-1/11","1/12"],["1/10","1/12","-1/13"]], "input":"plus_i_x", "simulate":False},
]


def case_spec(case: dict):
    def angle(text):
        if not isinstance(text,str) or len(text)>32 or not re.fullmatch(r"-?[0-9]+(?:/[0-9]+)?",text):
            raise ValueError("pi multiples must be bounded rational strings")
        return float(Fraction(text))*math.pi
    gqsp.source_operations(case["n"],0)
    if not isinstance(case["rotations_pi"],list) or not 1<=len(case["rotations_pi"])<=gqsp.MAX_DEGREE+1:
        raise ValueError("case rotation list exceeds the finite source contract")
    if any(not isinstance(triple,list) or len(triple)!=3 for triple in case["rotations_pi"]):
        raise ValueError("each Euler entry has three supplied pi multiples")
    for key in ("y_pi","z_pi"):
        if not isinstance(case[key],(str,list)) or (isinstance(case[key],list) and len(case[key])!=case["n"]):
            raise ValueError("case access angles must match the bounded data width")
    gqsp.source_operations(case["n"],len(case["rotations_pi"])-1,case["input"],True)
    ys = [case["y_pi"]]*case["n"] if isinstance(case["y_pi"], str) else case["y_pi"]
    zs = [case["z_pi"]]*case["n"] if isinstance(case["z_pi"], str) else case["z_pi"]
    spec = gqsp.signal_spec(case["n"], [angle(x) for x in ys], [angle(x) for x in zs], angle(case["chi_pi"]))
    rotations = tuple(gqsp.EulerRotation(*(angle(x) for x in triple)) for triple in case["rotations_pi"])
    return spec, rotations


def _read_input(path):
    path=Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("snapshot inputs must be regular nonsymlink files")
    with path.open("rb") as stream:
        data=stream.read(MAX_INPUT_BYTES+1)
    if len(data)>MAX_INPUT_BYTES:
        raise ValueError("snapshot input exceeds the 256KiB bound")
    return data


def source_snapshot():
    """Capture committed repository inputs, or explicit detached bundle bytes."""
    paths={"gqsp.py":MODULE_PATH,"generate_simulation_gqsp.py":Path(__file__).resolve(),
           "requirements-lock.txt":ROOT/"requirements-lock.txt" if REPOSITORY else Path(__file__).with_name("requirements-lock.txt"),
           "LICENSE.txt":ROOT/"LICENSE" if REPOSITORY else Path(__file__).with_name("LICENSE.txt")}
    content={name:_read_input(path) for name,path in paths.items()}
    commit=None
    if REPOSITORY:
        result=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,timeout=5)
        commit=result.stdout.strip()
        if result.returncode or not re.fullmatch(r"[0-9a-f]{40}",commit):
            raise ValueError("commit the frozen GQSP source before calculation")
        for name,path in paths.items():
            relative=path.relative_to(ROOT).as_posix()
            blob=subprocess.run(["git","show",f"{commit}:{relative}"],cwd=ROOT,capture_output=True,timeout=5)
            if blob.returncode or blob.stdout!=content[name]:
                raise ValueError(f"uncommitted or changed frozen input: {relative}")
    return {"paths":paths,"content":content,"source_commit":commit}


def recheck_snapshot(snapshot):
    for name,path in snapshot["paths"].items():
        if _read_input(path)!=snapshot["content"][name]:
            raise ValueError("source/producer/lock/license changed during calculation")
    if snapshot["source_commit"] is not None:
        result=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,timeout=5)
        if result.returncode or result.stdout.strip()!=snapshot["source_commit"]:
            raise ValueError("frozen source commit changed during calculation")


def _output_location(path,fresh):
    destination=Path(path).absolute()
    current=destination
    while current!=current.parent:
        if current.is_symlink():
            raise ValueError("output path may not contain a symlink ancestor")
        current=current.parent
    destination=destination.resolve()
    if destination==ROOT or (fresh and destination.exists()):
        raise ValueError("output must be a fresh dedicated directory")
    for folder in (".git","src","scripts","tests","notebooks","algorithms","docs","references","assets"):
        protected=ROOT/folder
        if destination==protected or protected in destination.parents:
            raise ValueError("prototype output cannot enter protected source or historical assets")
    if not fresh and not destination.is_dir():
        raise ValueError("payload output must be an existing dedicated directory")
    return destination


def output_directory(path):
    return _output_location(path,True)


def _payload_name(name):
    if not isinstance(name,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}",name):
        raise ValueError("payload name must be a safe single relative file")
    if name in (".",".."):
        raise ValueError("unsafe relative payload name")
    return name


class Writer:
    def __init__(self,directory):
        self.directory=_output_location(directory,False)
        self.files=[]
        self.used=0
    def write(self,name,data):
        name=_payload_name(name)
        if any(old_name==name for old_name,_ in self.files):
            raise ValueError("payload names must be unique")
        reserve=0 if name=="manifest.json" else 65536
        encoded=data.encode() if isinstance(data,str) else data
        if len(self.files)>=MAX_FILES or self.used+len(encoded)>MAX_OUTPUT_BYTES-reserve:
            raise ValueError("managed output cap exceeded")
        current=self.directory
        while current!=current.parent:
            if current.is_symlink():
                raise ValueError("payload directory acquired a symlink ancestor")
            current=current.parent
        path=self.directory/name
        with path.open("xb") as stream:
            stream.write(encoded)
        self.files.append((name,len(encoded)))
        self.used+=len(encoded)


def _dense_signal(spec):
    """Independent classical formula assembled from bit-index two-by-two actions."""
    import numpy as np
    dimension = 1 << spec.n
    matrix = np.eye(dimension, dtype=complex)
    for j, (y, z) in enumerate(zip(spec.y, spec.z)):
        local = np.array([[math.cos(y)*complex(math.cos(z),-math.sin(z)),
                           -math.sin(y)*complex(math.cos(z),-math.sin(z))],
                          [math.sin(y)*complex(math.cos(z),math.sin(z)),
                           math.cos(y)*complex(math.cos(z),math.sin(z))]])
        action = np.zeros((dimension,dimension), dtype=complex)
        for lower in range(dimension):
            if lower & (1 << j):
                continue
            upper = lower | (1 << j)
            action[np.ix_([lower,upper],[lower,upper])] = local
        matrix = action @ matrix
    phases = [(-1)**sum(((index >> j)&1)*((index >> (j+1))&1) for j in range(spec.n-1)) for index in range(dimension)]
    return complex(math.cos(spec.chi),math.sin(spec.chi))*np.diag(phases)@matrix


def _euler_reference(rotation):
    import numpy as np
    z_left = np.diag([complex(math.cos(rotation.phi/2),-math.sin(rotation.phi/2)),
                      complex(math.cos(rotation.phi/2),math.sin(rotation.phi/2))])
    y = np.array([[math.cos(rotation.theta),-math.sin(rotation.theta)],
                  [math.sin(rotation.theta),math.cos(rotation.theta)]])
    z_right = np.diag([complex(math.cos(rotation.lam/2),-math.sin(rotation.lam/2)),
                       complex(math.cos(rotation.lam/2),math.sin(rotation.lam/2))])
    return z_left@y@z_right


def _matrix_polynomial(coefficients, matrix):
    import numpy as np
    result = np.zeros_like(matrix)
    identity = np.eye(len(matrix),dtype=complex)
    for coefficient in reversed(coefficients):
        result = result@matrix+coefficient*identity
    return result


def _input_reference(n, mode):
    import numpy as np
    state = np.zeros(1 << n, dtype=complex)
    if mode == "zero":
        state[0] = 1
    elif mode == "one":
        state[1] = 1
    else:
        offset = 2 if mode == "plus_i_x" else 0
        state[offset],state[offset+1] = 1/math.sqrt(2),1j/math.sqrt(2)
    return state


def small_report(spec, rotations, mode, snapshot=None):
    """Tiny circuit versus independent matrices; this call must be separately authorized."""
    gqsp.require_small(spec, rotations)
    snapshot=source_snapshot() if snapshot is None else snapshot
    recheck_snapshot(snapshot)
    import numpy as np
    from qiskit.quantum_info import DensityMatrix, Operator, Statevector
    circuit = gqsp.build_gqsp(spec,rotations)
    actual = Operator(circuit).data
    signal = _dense_signal(spec)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(signal)):
        raise AssertionError("nonfinite circuit or independent signal matrix")
    p,q = gqsp.polynomial_coefficients(rotations)
    expected_p,expected_q = _matrix_polynomial(p,signal),_matrix_polynomial(q,signal)
    p_error = float(np.max(np.abs(actual[0::2,0::2]-expected_p)))
    q_error = float(np.max(np.abs(actual[1::2,0::2]-expected_q)))
    unitarity_error = float(np.max(np.abs(actual.conj().T@actual-np.eye(len(actual)))))
    scalar_error, circle_residual = 0.0,0.0
    for k in range(17):
        angle = 2*math.pi*k/17
        z = complex(math.cos(angle),math.sin(angle))
        ref = _euler_reference(rotations[0])
        for rotation in rotations[1:]:
            ref = _euler_reference(rotation)@np.diag([z,1])@ref
        pv,qv = gqsp.polynomial_value(p,z),gqsp.polynomial_value(q,z)
        if not np.all(np.isfinite(ref)) or not all(math.isfinite(v) for v in (pv.real,pv.imag,qv.real,qv.imag)):
            raise AssertionError("nonfinite scalar polynomial/reference result")
        scalar_error = max(scalar_error,abs(pv-ref[0,0]),abs(qv-ref[1,0]))
        circle_residual = max(circle_residual,abs(abs(pv)**2+abs(qv)**2-1))
    initial = _input_reference(spec.n,mode)
    expected = np.empty(len(actual),dtype=complex)
    expected[0::2],expected[1::2] = expected_p@initial,expected_q@initial
    prepared = gqsp.build_prepared_gqsp(spec,rotations,mode)
    vector = Statevector.from_int(0,len(actual)).evolve(prepared).data
    trial = gqsp.build_measured_trial(spec,rotations,mode)
    reset_program = trial.remove_final_measurements(inplace=False)
    arbitrary_initial = DensityMatrix(np.eye(len(actual))/len(actual))
    final = arbitrary_initial.evolve(reset_program).data
    vector_error = float(np.max(np.abs(vector-expected)))
    reset_error = float(np.max(np.abs(final-np.outer(expected,expected.conj()))))
    joint = np.abs(expected)**2
    accepted_probability = float(np.sum(joint[0::2]))
    residuals=(p_error,q_error,unitarity_error,scalar_error,circle_residual,vector_error,reset_error)
    if not all(math.isfinite(v) for v in residuals) or not np.all(np.isfinite(joint)) or not math.isfinite(accepted_probability):
        raise AssertionError("nonfinite numerical residual or probability output")
    if np.any(joint<0) or np.any(joint>1+5e-12) or not 0<=accepted_probability<=1+5e-12 or abs(float(np.sum(joint))-1)>5e-12:
        raise AssertionError("independent joint probabilities violate the numerical normalization guard")
    if max(residuals)>5e-12:
        raise AssertionError("independent tiny GQSP circuit/reference checks failed")
    recheck_snapshot(snapshot)
    return {"status":"numerical_small_checks_passed", "evidence":"bounded floating-point comparison; no interval or formal certificate",
            "p_block_error":p_error,"q_block_error":q_error,"unitarity_error":unitarity_error,
            "scalar_product_error":float(scalar_error),"17_point_circle_identity_residual":float(circle_residual),
            "prepared_vector_error":vector_error,"actual_reset_program_error":reset_error,
            "accepted_probability":accepted_probability,"joint_readout_probabilities_lsb_ancilla":joint.tolist(),
            "conditional_state_available":accepted_probability>gqsp.MIN_CONDITIONAL_PROBABILITY,
            "measurements":spec.n+1,"resets":spec.n+1,"literal_finite_shot_sampling_performed":False}


def generate(output: Path, cases: list[dict], simulate: bool = True):
    from qiskit import qasm3
    output=output_directory(output)
    if not isinstance(cases,list) or not 1<=len(cases)<=MAX_CASES:
        raise ValueError("producer has at most eight finite cases")
    for case in cases:
        _payload_name(case["id"]+".qasm")
    if len({case["id"] for case in cases})!=len(cases):
        raise ValueError("case identifiers must be unique")
    snapshot=source_snapshot()
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    writer=Writer(output)
    write=writer.write
    for case in cases:
        spec,rotations = case_spec(case)
        core = gqsp.build_gqsp(spec,rotations)
        trial = gqsp.build_measured_trial(spec,rotations,case["input"])
        assert not set(trial.count_ops())-gqsp.PRIMITIVE_NAMES
        assert len(trial.data)==gqsp.source_operations(spec.n,len(rotations)-1,case["input"],True)
        assert float(trial.global_phase)==0
        qasm = qasm3.dumps(trial)
        write(case["id"]+".qasm",qasm)
        p,q = gqsp.polynomial_coefficients(rotations)
        record = {"case":case,"data_qubits":spec.n,"total_qubits":spec.n+1,"degree_upper_bound":len(rotations)-1,
                  "signal_calls":len(rotations)-1,"core_source_operations":len(core.data),"complete_trial_operations":len(trial.data),
                  "counts":dict(trial.count_ops()),"dependency_depth":trial.depth(),
                  "coefficients_ascending":{"p":[[z.real,z.imag] for z in p],"q":[[z.real,z.imag] for z in q]},
                  "qasm_global_phase":"zero; conditional chi retained by literal p gate",
                  "physical_resources":"uncomputed","inverse_phase_compiler":False,"hamiltonian_simulation_complete":False,
                  "validation":{"status":"generation_only","simulation_performed":False}}
        if simulate and case["simulate"]:
            record["validation"] = {**small_report(spec,rotations,case["input"],snapshot),"simulation_performed":True}
        records.append(record)
    recheck_snapshot(snapshot)
    frozen=snapshot["content"]
    lock_bytes=frozen["requirements-lock.txt"]
    summary = {"status":"bounded_given_angle_component","cases":records,"python":platform.python_version(),
               "qiskit":importlib.metadata.version("qiskit"),"source_sha256":hashlib.sha256(frozen["gqsp.py"]).hexdigest(),
               "numpy":importlib.metadata.version("numpy"),"lock_sha256":hashlib.sha256(lock_bytes).hexdigest(),
               "producer_sha256":hashlib.sha256(frozen["generate_simulation_gqsp.py"]).hexdigest(),
               "case_vector_sha256":hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
               "source_commit":snapshot["source_commit"],"source_commit_binding":"repository inputs equal captured Git blobs; detached reproduction binds content hashes",
               "error_scope":"Numerical primitive/reference checks only; no coherent synthesis, hardware or application certificate",
               "algorithm_scope":"given-angle polynomial of explicit unitary; no inverse compiler or full Hamiltonian simulation"}
    write("study.json",json.dumps(summary,indent=2,allow_nan=False)+"\n")
    write("gqsp.py",frozen["gqsp.py"])
    write("generate_simulation_gqsp.py",frozen["generate_simulation_gqsp.py"])
    write("LICENSE.txt",frozen["LICENSE.txt"])
    write("requirements-lock.txt",lock_bytes)
    recheck_snapshot(snapshot)
    manifest = {"files":{name:{"bytes":size,"sha256":hashlib.sha256((output/name).read_bytes()).hexdigest()} for name,size in writer.files},
                "output_bytes_before_manifest":writer.used,"maximum_bytes":MAX_OUTPUT_BYTES,"maximum_files":MAX_FILES,
                "source_bytes_preserved":True,"source_commit":snapshot["source_commit"]}
    write("manifest.json",json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",action="store_true")
    parser.add_argument("--n",type=int,default=1)
    parser.add_argument("--degree",type=int,default=1)
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    if sys.version_info[:3]!=(3,12,14) or importlib.metadata.version("qiskit")!="2.6.2":
        raise SystemExit("This frozen prototype run requires Python3.12.14 and Qiskit2.6.2")
    if args.suite:
        cases=CASE_VECTOR
    else:
        if not 1<=args.n<=gqsp.MAX_DATA_QUBITS or not 0<=args.degree<=gqsp.MAX_DEGREE:
            raise SystemExit("generator request exceeds declared width/degree bounds")
        template = CASE_VECTOR[6]
        rotations = (template["rotations_pi"]*((args.degree+7)//7))[:args.degree+1]
        cases=[{**template,"id":"custom-given-angle", "n":args.n,"rotations_pi":rotations,
                "input":"plus_i_x" if args.n>=2 else "plus_i","simulate":not args.no_sim}]
        if not args.no_sim and args.n+1>gqsp.MAX_DENSE_TOTAL_QUBITS:
            raise SystemExit("use --no-sim for larger generated circuits")
    result=generate(args.output_dir,cases,simulate=not args.no_sim)
    print(json.dumps({"status":result["status"],"cases":len(result["cases"]),"output_dir":str(args.output_dir)},indent=2,allow_nan=False))


if __name__=="__main__":
    main()
