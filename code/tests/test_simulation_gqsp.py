"""Pending independent numerical checks for supplied-angle GQSP.

These tests have not executed in the source-preparation task. The literal
complex degree-one reference, bit-index matrix reference, and mixed-state
reset chronology are independent of the gate builder and coefficient helper.
"""
import cmath
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import DensityMatrix, Operator, Statevector

from openquantum.simulation import gqsp

SOURCE_ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("gqsp_producer_test_snapshot",SOURCE_ROOT/"scripts/generate_simulation_gqsp.py")
producer = importlib.util.module_from_spec(loader)
sys.modules[loader.name]=producer
loader.loader.exec_module(producer)


def _reference_euler(r):
    # Three mathematical matrices, independently of production euler_entries.
    left=np.diag([cmath.exp(-.5j*r.phi),cmath.exp(.5j*r.phi)])
    middle=np.array([[math.cos(r.theta),-math.sin(r.theta)],
                     [math.sin(r.theta),math.cos(r.theta)]],dtype=complex)
    right=np.diag([cmath.exp(-.5j*r.lam),cmath.exp(.5j*r.lam)])
    return left@middle@right


def _bit_action(n,j,local):
    matrix=np.zeros((1<<n,1<<n),dtype=complex)
    for column in range(1<<n):
        bit=(column>>j)&1
        for new_bit in (0,1):
            row=(column&~(1<<j))|(new_bit<<j)
            matrix[row,column]=local[new_bit,bit]
    return matrix


def _reference_signal(spec):
    matrix=np.eye(1<<spec.n,dtype=complex)
    for j,(y,z) in enumerate(zip(spec.y,spec.z)):
        y_matrix=np.array([[math.cos(y),-math.sin(y)],[math.sin(y),math.cos(y)]])
        z_matrix=np.diag([cmath.exp(-1j*z),cmath.exp(1j*z)])
        matrix=_bit_action(spec.n,j,z_matrix)@_bit_action(spec.n,j,y_matrix)@matrix
    for basis in range(1<<spec.n):
        sign=(-1)**sum(((basis>>j)&1)*((basis>>(j+1))&1) for j in range(spec.n-1))
        matrix[basis,:]*=sign*cmath.exp(1j*spec.chi)
    return matrix


def _reference_full(spec,rotations,control_state=0):
    signal=_reference_signal(spec)
    dimension=len(signal)
    access=np.zeros((2*dimension,2*dimension),dtype=complex)
    access[control_state::2,control_state::2]=signal
    access[1-control_state::2,1-control_state::2]=np.eye(dimension)
    result=np.kron(np.eye(dimension),_reference_euler(rotations[0]))
    for r in rotations[1:]:
        result=np.kron(np.eye(dimension),_reference_euler(r))@access@result
    return result


@pytest.mark.parametrize("case",producer.CASE_VECTOR[:6],ids=[c["id"] for c in producer.CASE_VECTOR[:6]])
def test_explicit_signal_and_full_complex_gqsp_against_independent_matrices(case):
    spec,rotations=producer.case_spec(case)
    gqsp.require_small(spec,rotations)
    actual_signal=Operator(gqsp.build_signal(spec)).data
    np.testing.assert_allclose(actual_signal,_reference_signal(spec),atol=3e-12,rtol=0)
    core=gqsp.build_gqsp(spec,rotations)
    np.testing.assert_allclose(Operator(core).data,_reference_full(spec,rotations),atol=5e-12,rtol=0)
    assert float(core.global_phase)==0
    assert set(core.count_ops())<=gqsp.PRIMITIVE_NAMES
    assert len(core.data)==gqsp.source_operations(spec.n,len(rotations)-1)


@pytest.mark.parametrize("case",producer.CASE_VECTOR[:6],ids=[c["id"] for c in producer.CASE_VECTOR[:6]])
def test_forward_polynomial_against_independent_scalar_products(case):
    _,rotations=producer.case_spec(case)
    p,q=gqsp.polynomial_coefficients(rotations)
    for j in range(37):
        z=cmath.exp(2j*math.pi*j/37)
        matrix=_reference_euler(rotations[0])
        for r in rotations[1:]:
            matrix=_reference_euler(r)@np.diag([z,1])@matrix
        assert abs(gqsp.polynomial_value(p,z)-matrix[0,0])<5e-12
        assert abs(gqsp.polynomial_value(q,z)-matrix[1,0])<5e-12
        assert abs(abs(matrix[0,0])**2+abs(matrix[1,0])**2-1)<5e-12


def test_complex_mixed_parity_degree_one_has_literal_closed_form_and_readout_law():
    spec,rotations=producer.case_spec(producer.CASE_VECTOR[1])
    p,q=gqsp.polynomial_coefficients(rotations)
    minus,plus=(1-1j)/math.sqrt(2),(1+1j)/math.sqrt(2)
    np.testing.assert_allclose(p,[-plus/2,minus/2],atol=1e-14,rtol=0)
    np.testing.assert_allclose(q,[plus/2,minus/2],atol=1e-14,rtol=0)
    # Both degree0 and degree1 coefficients are nonzero and complex.
    assert min(abs(z) for z in p)>.49
    matrix=Operator(gqsp.build_gqsp(spec,rotations)).data
    np.testing.assert_allclose(matrix[0::2,0::2],np.diag([-1j/math.sqrt(2),0]),atol=2e-14,rtol=0)
    np.testing.assert_allclose(matrix[1::2,0::2],np.diag([1/math.sqrt(2),plus]),atol=2e-14,rtol=0)
    prepared=gqsp.build_prepared_gqsp(spec,rotations,"plus_i")
    actual=Statevector.from_int(0,4).evolve(prepared).data
    expected=np.array([-1j/2,1/2,0,1j*plus/math.sqrt(2)])
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)
    np.testing.assert_allclose(np.abs(actual)**2,[.25,.25,0,.5],atol=2e-14,rtol=0)
    # Literal amplitudes preserve phase; phase-aligned comparison is forbidden.
    assert abs(matrix[2,2])<1e-14


def test_wrong_control_order_and_dropped_signal_phase_change_exact_case():
    spec,rotations=producer.case_spec(producer.CASE_VECTOR[1])
    positive=QuantumCircuit(2)
    gqsp.append_euler(positive,rotations[0])
    gqsp.append_controlled_signal(positive,spec,0,(1,),control_state=1)
    gqsp.append_euler(positive,rotations[1])
    reverse=QuantumCircuit(2)
    gqsp.append_euler(reverse,rotations[1])
    gqsp.append_controlled_signal(reverse,spec,0,(1,))
    gqsp.append_euler(reverse,rotations[0])
    missing_phase=gqsp.build_gqsp(gqsp.signal_spec(1,spec.y,spec.z,0),rotations)
    assert abs(Operator(positive).data[2,2])**2==pytest.approx(1,abs=2e-14)
    assert abs(Operator(reverse).data[2,2])**2==pytest.approx(.5,abs=2e-14)
    good=Operator(gqsp.build_gqsp(spec,rotations)).data
    assert np.max(np.abs(good-Operator(missing_phase).data))>.1
    np.testing.assert_allclose(Operator(positive).data,_reference_full(spec,rotations,1),atol=2e-14,rtol=0)


@pytest.mark.parametrize("case",producer.CASE_VECTOR[:6],ids=[c["id"] for c in producer.CASE_VECTOR[:6]])
def test_actual_reset_readout_chronology_and_complete_primitive_counts(case):
    spec,rotations=producer.case_spec(case)
    trial=gqsp.build_measured_trial(spec,rotations,case["input"])
    assert len(trial.data)==gqsp.source_operations(spec.n,len(rotations)-1,case["input"],True)
    assert all(x.operation.name=="reset" for x in trial.data[:spec.n+1])
    assert [x.operation.name for x in trial.data[-(spec.n+1):]]==["measure"]*(spec.n+1)
    assert [(trial.find_bit(x.qubits[0]).index,trial.find_bit(x.clbits[0]).index)
            for x in trial.data[-(spec.n+1):]]==[(j,j) for j in range(spec.n+1)]
    initial=producer._input_reference(spec.n,case["input"])
    full=_reference_full(spec,rotations)
    reference=full[:,0::2]@initial
    dimension=len(full)
    mixed=DensityMatrix(np.eye(dimension)/dimension)
    state=mixed.evolve(trial.remove_final_measurements(inplace=False)).data
    np.testing.assert_allclose(state,np.outer(reference,reference.conj()),atol=5e-12,rtol=0)
    source=qasm3.dumps(trial)
    assert "gphase" not in source
    assert not {"unitary","hamiltonian","initialize","state_preparation"}&set(trial.count_ops())
    assert set(trial.count_ops())<=gqsp.PRIMITIVE_NAMES


@pytest.mark.parametrize("n,degree",[(8,6),(64,6),(64,64)])
def test_large_request_has_explicit_generation_scope_and_no_implicit_simulation(n,degree):
    spec=gqsp.signal_spec(n,[.125]*n,[-.25]*n,.3)
    rotations=(gqsp.EulerRotation(.2,.3,-.4),)*(degree+1)
    with pytest.raises(ValueError,match="dense simulation"):
        gqsp.require_small(spec,rotations)
    # This test does not materialize large circuits; the separate suite does n8/n64,d6.
    assert gqsp.source_operations(n,degree,"plus_i_x",True)<=gqsp.MAX_SOURCE_OPERATIONS


@pytest.mark.parametrize("bad",[True,0,65,1.0])
def test_invalid_width_rejected_before_circuit_creation(bad):
    with pytest.raises(ValueError):
        gqsp.signal_spec(bad,[0],[0])


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),101*math.pi,True,1j])
def test_invalid_angles_rejected(bad):
    with pytest.raises(ValueError):
        gqsp.signal_spec(1,[bad],[0])


def test_empty_oversized_or_invalid_rotation_and_preparation_contracts_rejected():
    spec=gqsp.signal_spec(1,[0],[0])
    for rotations in ([],[gqsp.EulerRotation(0,0,0)]*66,[(0,0,0)]):
        with pytest.raises(ValueError):
            gqsp.build_gqsp(spec,rotations)
    with pytest.raises(ValueError):
        gqsp.build_measured_trial(spec,[gqsp.EulerRotation(0,0,0)],"plus_i_x")
    with pytest.raises(ValueError):
        gqsp.signal_spec(2,[0],[0,0])


def test_detached_producer_selects_shipped_source_beneath_an_unrelated_checkout(tmp_path):
    # Provenance fixture only: this fake source has no quantum or numerical API.
    ambient=tmp_path/"src/openquantum/simulation"
    ambient.mkdir(parents=True)
    (ambient/"gqsp.py").write_text("raise RuntimeError('ambient source selected')\n")
    shipped=tmp_path/"bundle"
    shipped.mkdir()
    module=shipped/"gqsp.py"
    module.write_text("SENTINEL = 'shipped source'\n")
    script=shipped/"generate_simulation_gqsp.py"
    script.write_bytes((SOURCE_ROOT/"scripts/generate_simulation_gqsp.py").read_bytes())
    previous=sys.modules.get("openquantum_gqsp_snapshot")
    try:
        loader=importlib.util.spec_from_file_location("gqsp_detached_provenance_test",script)
        detached=importlib.util.module_from_spec(loader)
        loader.loader.exec_module(detached)
        assert detached.REPOSITORY is False
        assert detached.MODULE_PATH.resolve()==module.resolve()
        assert detached.gqsp.SENTINEL=="shipped source"
    finally:
        if previous is None:
            sys.modules.pop("openquantum_gqsp_snapshot",None)
        else:
            sys.modules["openquantum_gqsp_snapshot"]=previous


def test_output_gate_protects_sources_assets_existing_paths_and_symlink_ancestors(tmp_path,monkeypatch):
    root=tmp_path.resolve()/"source"
    root.mkdir()
    monkeypatch.setattr(producer,"ROOT",root)
    for folder in (".git","src","scripts","tests","notebooks","algorithms","docs","references","assets"):
        with pytest.raises(ValueError,match="protected"):
            producer.output_directory(root/folder/"new")
    existing=root/"old-run"
    existing.mkdir()
    with pytest.raises(ValueError,match="fresh"):
        producer.output_directory(existing)
    link=root/"linked-run"
    link.symlink_to(existing,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):
        producer.output_directory(link/"new")
    assert producer.output_directory(root/".cache/new-run")==root/".cache/new-run"


def test_payload_writer_rejects_escape_duplicate_and_existing_siblings(tmp_path):
    directory=tmp_path.resolve()/"payload"
    directory.mkdir()
    writer=producer.Writer(directory)
    for name in ("../other.txt","/absolute.txt","nested/file.txt",".hidden"):
        with pytest.raises(ValueError,match="safe"):
            writer.write(name,b"no")
    writer.write("first.txt",b"preserved")
    with pytest.raises(ValueError,match="unique"):
        writer.write("first.txt",b"replacement")
    assert (directory/"first.txt").read_bytes()==b"preserved"
    (directory/"external.txt").write_bytes(b"external")
    with pytest.raises(FileExistsError):
        writer.write("external.txt",b"replacement")
    assert (directory/"external.txt").read_bytes()==b"external"


def test_nonfinite_reference_cannot_produce_a_success_record(monkeypatch):
    spec,rotations=producer.case_spec(producer.CASE_VECTOR[1])
    monkeypatch.setattr(producer,"source_snapshot",lambda:{"paths":{},"content":{},"source_commit":None})
    monkeypatch.setattr(producer,"_dense_signal",lambda ignored:np.full((2,2),complex(float("nan"),0)))
    with pytest.raises(AssertionError,match="nonfinite"):
        producer.small_report(spec,rotations,"plus_i")
