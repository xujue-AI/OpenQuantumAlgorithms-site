# SPDX-License-Identifier: MIT
"""Independent NumPy replay of the exact bounded canonical QASM3 subset.

This is deliberately not a general OpenQASM parser. Unknown syntax rejects;
QDK separately imports the complete artifact for the physical projection.
"""
import ast
import math
import re

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator,Statevector

from openquantum.linear_systems_qsvt_reciprocal import (
    DEFAULT_TERMS,lower_reciprocal,measured_reciprocal_trial,prepared_reciprocal,
)


def constant_angle(text):
    def evaluate(node):
        if isinstance(node,ast.Constant) and type(node.value) in {int,float}: return node.value
        if isinstance(node,ast.Name) and node.id=="pi": return math.pi
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            value=evaluate(node.operand)
            return value if isinstance(node.op,ast.UAdd) else -value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            left,right=evaluate(node.left),evaluate(node.right)
            if isinstance(node.op,ast.Add): return left+right
            if isinstance(node.op,ast.Sub): return left-right
            if isinstance(node.op,ast.Mult): return left*right
            return left/right
        raise ValueError("unsupported QASM angle expression")
    value=float(evaluate(ast.parse(text,mode="eval").body))
    if not math.isfinite(value): raise ValueError("nonfinite angle")
    return value


def replay_canonical_qasm(text):
    width=int(re.search(r"qubit\[(\d+)\] q;",text).group(1))
    result=np.eye(1<<width,dtype=complex)
    measurements=[]
    for line in text.splitlines():
        if (line in {'OPENQASM 3.0;','include "stdgates.inc";'}
                or re.fullmatch(r"(?:qubit|bit)\[\d+\] \w+;",line)
                or line.startswith("barrier ")):
            continue
        match=re.fullmatch(r"meas\[(\d+)\] = measure q\[(\d+)\];",line)
        if match:
            measurements.append(tuple(map(int,match.groups())));continue
        match=re.fullmatch(r"cx q\[(\d+)\], q\[(\d+)\];",line)
        if match:
            control,target=map(int,match.groups())
            rows=np.arange(1<<width)
            low=rows[((rows>>control)&1).astype(bool)&~((rows>>target)&1).astype(bool)]
            high=low^(1<<target)
            result[low],result[high]=result[high].copy(),result[low].copy()
            continue
        rotation=re.fullmatch(r"(rx|ry|rz)\((.+)\) q\[(\d+)\];",line)
        plain=re.fullmatch(r"(x|h|s|sdg) q\[(\d+)\];",line)
        if rotation:
            name,angle,target=rotation.groups();target=int(target)
            angle=constant_angle(angle)
            pauli={"rx":np.array([[0,1],[1,0]]),"ry":np.array([[0,-1j],[1j,0]]),"rz":np.diag([1,-1])}[name]
            gate=math.cos(angle/2)*np.eye(2)-1j*math.sin(angle/2)*pauli
        elif plain:
            name,target=plain.groups();target=int(target)
            gate={"x":np.array([[0,1],[1,0]]),"h":np.array([[1,1],[1,-1]])/math.sqrt(2),
                  "s":np.diag([1,1j]),"sdg":np.diag([1,-1j])}[name]
        else: raise ValueError("unsupported canonical QASM line: "+line)
        rows=np.arange(1<<width);low=rows[((rows>>target)&1)==0];high=low^(1<<target)
        first,second=result[low].copy(),result[high].copy()
        result[low]=gate[0,0]*first+gate[0,1]*second
        result[high]=gate[1,0]*first+gate[1,1]*second
    assert sorted(measurements)==[(i,i) for i in range(width)]
    return result


@pytest.mark.parametrize("terms,k",[(DEFAULT_TERMS,16),(((3,"I"),(-1,"Y")),1)])
def test_source_full_operator_and_exported_measured_probabilities(terms,k):
    source=prepared_reciprocal(terms,k)
    trial=measured_reciprocal_trial(terms,k)
    exported=qasm3.dumps(lower_reciprocal(trial))
    replay=replay_canonical_qasm(exported)
    reference=Operator(source).data
    phase=np.angle(np.trace(reference.conj().T@replay))
    # Overall phase may be omitted by the pinned exporter. Full coherent
    # comparison is therefore modulo that one phase, never relative phases.
    assert np.linalg.norm(replay-reference*np.exp(1j*phase))<1e-11
    state=Statevector.from_instruction(source).data
    assert np.max(abs(abs(replay[:,0])**2-abs(state)**2))<1e-12
    n=len(terms[0][1]);accepted=replay[:1<<n,0]
    assert abs(np.linalg.norm(accepted)**2-np.linalg.norm(state[:1<<n])**2)<1e-12


def test_qasm_replay_rejects_unrecognized_syntax():
    with pytest.raises(ValueError): constant_angle("__import__('os')")
    with pytest.raises(ValueError): replay_canonical_qasm('OPENQASM 3.0;\nqubit[1] q;\nbit[1] meas;\nreset q[0];')
