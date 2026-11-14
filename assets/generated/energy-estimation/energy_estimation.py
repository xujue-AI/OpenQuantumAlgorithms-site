# SPDX-License-Identifier: MIT
"""Explicit dimer Hamiltonian access and robust retained-target energy QPE.

This module constructs gates, not matrix exponentials or eigenvector gates.
Dimer Pauli labels read q1 q0. Separate bounded NumPy references verify the
access and spectral output. Fresh-clock static circuits defer measurements;
sequential measurement/reset sampling has the same joint output law, tested
for small complete circuits. Physical initialization and synthesis remain
separate from ideal quantum confidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

if __package__:
    from .qft_qpe import build_qft
else:
    from qft_qpe import build_qft

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_DIMERS = 12
MAX_PHASE_BITS = 12
MAX_ROUNDS = 31
MAX_GROUPS = 256
MAX_STATIC_QUBITS = 256
MAX_PRIMITIVE_OPERATIONS = 750_000
MAX_SIMULATOR_QUBITS = 14
MAX_SAMPLED_ROUNDS = 1024


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    result = float(value)
    if abs(result) > 1e6:
        raise ValueError(f"{name} exceeds the finite model coefficient/time cap")
    return result


@dataclass(frozen=True)
class DimerHamiltonian:
    """H = (electronic_identity+nuclear) I + z0 Z0 + z1 Z1 + zz ZZ + xx XX.

    Generic inputs define this Pauli model. Molecular provenance is supplied
    separately by the publication fixture; arbitrary values are not called H2.
    """
    electronic_identity: float
    z0: float
    z1: float
    zz: float
    xx: float
    nuclear_repulsion: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            _real(value, name)

    @property
    def identity(self):
        return float(self.electronic_identity + self.nuclear_repulsion)

    @property
    def nonidentity_norm_bound(self):
        return float(abs(self.z0) + abs(self.z1) + abs(self.zz) + abs(self.xx))

    def analytic_spectrum_and_hf_overlap(self):
        """Two parity-block eigenvalues and |q1 q0>=|01> ground overlap."""
        even_norm = math.hypot(self.z0 + self.z1, self.xx)
        odd_norm = math.hypot(self.z0 - self.z1, self.xx)
        even = (self.identity + self.zz - even_norm, self.identity + self.zz + even_norm)
        odd = (self.identity - self.zz - odd_norm, self.identity - self.zz + odd_norm)
        ground = min(even[0], odd[0])
        overlap = (1.0 if odd_norm == 0 else (1 + (self.z0 - self.z1) / odd_norm) / 2)
        if odd[0] > ground:
            overlap = 0.0
        return {"eigenvalues": sorted([*even, *odd]), "even_parity_eigenvalues": list(even),
                "odd_parity_eigenvalues": list(odd), "ground_energy": ground,
                "hartree_fock_ground_overlap": float(overlap)}

    def pauli_terms(self, dimers=1):
        _integer(dimers, "dimers", 1, MAX_DIMERS)
        width = 2 * dimers
        terms = [(self.identity * dimers, "I" * width)]
        for dimer in range(dimers):
            for coefficient, local in ((self.z0, "IZ"), (self.z1, "ZI"),
                                       (self.zz, "ZZ"), (self.xx, "XX")):
                if coefficient:
                    label = ["I"] * width
                    label[width - 2 * dimer - 2:width - 2 * dimer] = local
                    terms.append((float(coefficient), "".join(label)))
        return tuple(terms)


# Stored numerical fixture, not a newly executed SCF calculation. Exact data:
# qiskit-community/qiskit-nature tag 0.7.2 commit
# d86d4a0306f8f4af8918c9eb9e3eae4622192f45,
# docs/tutorials/06_qubit_mappers.ipynb and 01_electronic_structure.ipynb.
# The stored notebook execution records Qiskit Nature 0.6.0. Upstream data
# attribution is Apache-2.0; all circuit code here is original MIT work.
H2_STO3G = DimerHamiltonian(
    electronic_identity=-1.0523732457728605,
    z0=0.39793742484317896,
    z1=-0.39793742484317896,
    zz=-0.01128010425623538,
    xx=0.18093119978423122,
    nuclear_repulsion=0.7199689944489797,
)
H2_FIXTURE_PROVENANCE = {
    "molecule": "H2", "coordinates_angstrom": [["H", 0., 0., 0.], ["H", 0., 0., .735]],
    "basis": "sto3g", "charge": 0, "spin": 0, "method": "RHF",
    "mapper": "ParityMapper(num_particles=(1,1))", "spin_orbital_order": "alpha block then beta block",
    "fixed_z_symmetry_values": [-1, 1], "reduced_hf_displayed_bits": "01",
    "fixture_repository": "qiskit-community/qiskit-nature", "fixture_checkout_tag": "0.7.2",
    "fixture_checkout_commit": "d86d4a0306f8f4af8918c9eb9e3eae4622192f45",
    "fixture_mapper_notebook_sha256": "560c1b4c4863cabef7b2cf8b2d5975695bfc0782fd1cbed2156aed6b778374bd",
    "fixture_electronic_structure_notebook_sha256": "d22a8646af6af5d83218c9351832e2b0f3e8dd21f6ae15896b205ab35d0bb33f",
    "fixture_release_date": "2024-02-23", "source_notebook_cell_indexing": "zero-based",
    "coefficient_source": "docs/tutorials/06_qubit_mappers.ipynb cell 12",
    "nuclear_source": "docs/tutorials/01_electronic_structure.ipynb cell 12",
    "removed_original_parity_wires": [1,3],
    "fixed_particle_numbers": {"alpha":1,"beta":1}, "mapped_space_is_singlet_only":False,
    "stored_execution_qiskit_nature_version": "0.6.0", "stored_execution_python_version": "3.9.16",
    "stored_execution_date": "2023-04-06",
    "particle_number_selection_scope": "parity tapering selects exact one-alpha/one-beta occupation only in this four-spin-orbital fixture",
    "pyscf_version_in_stored_output": None, "fresh_scf_or_mapping_executed_here": False,
    "upstream_license": "Apache-2.0", "electronic_identity_excludes_nuclear_repulsion": True,
    "pauli_label_order": "q1 q0",
}


@dataclass(frozen=True)
class EnergyEncoding:
    shift: float
    span: float

    def __post_init__(self):
        _real(self.shift, "shift")
        _real(self.span, "span")
        if self.span <= 0:
            raise ValueError("span must be positive")

    @property
    def time(self):
        return 2 * math.pi / self.span

    def energy(self, phase):
        return self.shift + self.span * np.asarray(phase)

    def phase(self, energy):
        return (np.asarray(energy) - self.shift) / self.span


def energy_encoding(model: DimerHamiltonian, dimers=1):
    _integer(dimers, "dimers", 1, MAX_DIMERS)
    beta = model.nonidentity_norm_bound * dimers
    center = model.identity * dimers
    if beta == 0:
        return EnergyEncoding(center - .5, 1.0)
    # Entire promised spectrum lies in phases [1/4,3/4], avoiding scalar-median
    # wrap for phase_bits>=2. This increases width/time relative to tight bounds.
    return EnergyEncoding(center - 2 * beta, 4 * beta)


def dimer_evolution(model: DimerHamiltonian, time: float):
    """Exact analytic-angle exp(-iHt), using parity CX and controlled axes.

    After CX(q0,q1), q1 stores parity. The target q0 sees XX -> X0 and
    z0 Z0+z1 Z1 -> (z0+(-1)^parity z1) Z0. No eigenvector/matrix gate appears.
    """
    time = _real(time, "time")
    circuit = QuantumCircuit(2, name="Dimer evolution")
    circuit.global_phase = -model.identity * time
    circuit.cx(0, 1)
    if model.zz * time:
        circuit.rz(2 * model.zz * time, 1)
    for parity, z_coefficient in ((0, model.z0 + model.z1), (1, model.z0 - model.z1)):
        omega = math.hypot(z_coefficient, model.xx)
        if omega == 0 or time == 0:
            continue
        theta = math.atan2(model.xx, z_coefficient)
        if parity == 0:
            circuit.x(1)
        circuit.cry(-theta, 1, 0)
        circuit.crz(2 * omega * time, 1, 0)
        circuit.cry(theta, 1, 0)
        if parity == 0:
            circuit.x(1)
    circuit.cx(0, 1)
    return circuit


def build_energy_unitary(model: DimerHamiltonian, dimers=1, encoding=None, *, power=1):
    """Explicit exp(2pi i (H-shift I)/span) for independent active dimers."""
    _integer(dimers, "dimers", 1, MAX_DIMERS)
    _integer(power, "power", 1, 2**MAX_PHASE_BITS)
    encoding = energy_encoding(model, dimers) if encoding is None else encoding
    circuit = QuantumCircuit(2 * dimers, name="Energy U")
    access = dimer_evolution(model, -encoding.time*power)
    for dimer in range(dimers):
        circuit.compose(access, [2 * dimer, 2 * dimer + 1], inplace=True)
    circuit.global_phase -= encoding.time * encoding.shift * power
    return circuit


def hartree_fock_preparation(dimers=1):
    """Prepare the explicit odd-parity product |01> for every active dimer."""
    _integer(dimers, "dimers", 1, MAX_DIMERS)
    circuit = QuantumCircuit(2 * dimers, name="HF preparation")
    circuit.x(range(0, 2 * dimers, 2))
    return circuit


def majority_failure(rounds: int, one_round_success=8 / math.pi**2):
    _integer(rounds, "rounds", 1, MAX_ROUNDS)
    if rounds % 2 == 0:
        raise ValueError("median rounds must be odd")
    p = _real(one_round_success, "one_round_success")
    if not .5 < p <= 1:
        raise ValueError("one-round success must be in (1/2,1]")
    return sum(math.comb(rounds, failures) * (1-p)**failures * p**(rounds-failures)
               for failures in range((rounds+1)//2, rounds+1))


def ground_energy_contract(model: DimerHamiltonian, dimers=1, phase_bits=5,
                           *, ideal_failure=.005, overlap_lower_bound=None):
    _integer(dimers, "dimers", 1, MAX_DIMERS)
    _integer(phase_bits, "phase_bits", 2, MAX_PHASE_BITS)
    delta = _real(ideal_failure, "ideal_failure")
    if not 0 < delta < .5:
        raise ValueError("ideal_failure must lie in (0,1/2)")
    exact_overlap = model.analytic_spectrum_and_hf_overlap()["hartree_fock_ground_overlap"]**dimers
    overlap = exact_overlap if overlap_lower_bound is None else _real(overlap_lower_bound, "overlap_lower_bound")
    if not 0 < overlap <= 1:
        raise ValueError("positive ground-overlap promise is required")
    if overlap > exact_overlap + 1e-14:
        raise ValueError("claimed overlap exceeds this explicit family's analytic overlap")
    overlap_failure = delta/2
    groups = 1 if overlap == 1 else max(1, math.ceil(math.log(overlap_failure)/math.log1p(-overlap)))
    if groups > MAX_GROUPS:
        raise ValueError("ground-overlap group requirement exceeds bounded workload")
    round_failure = delta/(2*groups)
    rounds = next((r for r in range(1, MAX_ROUNDS+1, 2) if majority_failure(r) <= round_failure), None)
    if rounds is None:
        raise ValueError("median confidence requires more than bounded rounds")
    encoding = energy_encoding(model, dimers)
    no_ground = (1-overlap)**groups
    median_failure = majority_failure(rounds)
    return {"kind": "minimum_of_retained-target_medians", "dimers": dimers, "data_qubits": 2*dimers,
            "phase_bits": phase_bits, "phase_width": 2**phase_bits,
            "encoding": asdict(encoding), "energy_absolute_error_ideal": encoding.span/2**phase_bits,
            "ground_overlap_lower_bound": overlap, "overlap_promise_source": "analytic two-parity dimer family; not generic chemistry state preparation",
            "independent_groups": groups, "qpe_rounds_per_retained_target": rounds,
            "one_round_phase_success_lower_bound": 8/math.pi**2,
            "median_failure_bound_per_group": median_failure, "no_ground_group_failure_bound": no_ground,
            "ideal_failure_union_bound": no_ground+groups*median_failure,
            "ideal_confidence_lower_bound": 1-no_ground-groups*median_failure,
            "requested_ideal_failure": delta, "total_qpe_rounds": groups*rounds,
            "total_base_U_equivalent_calls": groups*rounds*(2**phase_bits-1),
            "assumptions": ["exact analytic-angle Hamiltonian access", "independent HF preparations across groups",
                            "same target retained for all median rounds within each group", "spectrum maps into [1/4,3/4]",
                            "correct classical median/min decoding", "no physical faults or synthesis error"],
            "hardware_or_synthesis_confidence_certified": False}


def _controlled_access(model, dimers, encoding, power=1):
    return build_energy_unitary(model, dimers, encoding,power=power).to_gate(label=f"Energy U^{power}").control(1)


def build_retained_target_qpe(model: DimerHamiltonian, *, dimers=1, phase_bits=5, rounds=1,
                             measure=True, preparation=True, analytic_powers=False):
    """One group: fresh independent clocks act on the same retained target.

    Measurements are deferred until every round completes, making a static
    circuit with n+rounds*m wires. Repreparing HF for every round would destroy
    the median conditional-eigenvalue contract and is deliberately not done.
    """
    _integer(dimers, "dimers", 1, MAX_DIMERS)
    _integer(phase_bits, "phase_bits", 2, MAX_PHASE_BITS)
    _integer(rounds, "rounds", 1, MAX_ROUNDS)
    if rounds % 2 == 0:
        raise ValueError("rounds must be odd")
    if any(type(v) is not bool for v in (measure,preparation,analytic_powers)):
        raise ValueError("measure, preparation and analytic_powers must be booleans")
    clock_width = rounds*phase_bits
    width = 2*dimers+clock_width
    if width > MAX_STATIC_QUBITS:
        raise ValueError("deferred-clock circuit exceeds generation width")
    encoding = energy_encoding(model, dimers)
    controlled = _controlled_access(model, dimers, encoding)
    primitive_access = transpile(QuantumCircuit(2*dimers+1).compose(controlled, range(2*dimers+1)),
                                basis_gates=ROTATION_BASIS, optimization_level=1, seed_transpiler=7)
    power_gates=None
    if analytic_powers:
        power_gates=[_controlled_access(model,dimers,encoding,2**bit) for bit in range(phase_bits)]
        power_costs=[len(transpile(QuantumCircuit(2*dimers+1).compose(gate,range(2*dimers+1)),
            basis_gates=ROTATION_BASIS,optimization_level=1,seed_transpiler=7).data) for gate in power_gates]
        access_preflight=rounds*sum(power_costs)
    else:
        access_preflight=rounds*(2**phase_bits-1)*len(primitive_access.data)
    preflight = access_preflight + rounds*6*phase_bits**2+dimers
    if preflight > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("expanded controlled-access preflight exceeds primitive limit")
    circuit = QuantumCircuit(width, clock_width if measure else 0, name=f"Energy QPE x{rounds}")
    targets = list(range(clock_width,width))
    if preparation:
        circuit.compose(hartree_fock_preparation(dimers), targets, inplace=True)
    for round_index in range(rounds):
        clock = list(range(round_index*phase_bits,(round_index+1)*phase_bits))
        circuit.h(clock)
        for bit, qubit in enumerate(clock):
            if analytic_powers:
                circuit.append(power_gates[bit],[qubit,*targets])
            else:
                for _ in range(2**bit):
                    circuit.append(controlled, [qubit,*targets])
        circuit.compose(build_qft(phase_bits,inverse=True), clock, inplace=True)
    if measure:
        circuit.measure(range(clock_width),range(clock_width))
    circuit.metadata = {"dimers":dimers,"phase_bits":phase_bits,"rounds":rounds,
                        "encoding":asdict(encoding),"controlled_unitary_calls":rounds*phase_bits if analytic_powers else rounds*(2**phase_bits-1),
                        "retained_target":True,"hf_preparations":1 if preparation else 0,
                        "measurement_deferred":True,"primitive_preflight":preflight,
                        "access_mode":"analytic_dimer_powers" if analytic_powers else "repeated_U",
                        "controlled_power_circuits":rounds*phase_bits if analytic_powers else rounds*(2**phase_bits-1),
                        "base_U_equivalent_calls":rounds*(2**phase_bits-1)}
    return circuit


def build_recycled_clock_qpe(model,*,dimers=1,phase_bits=5,rounds=19,analytic_powers=False):
    """Actual measurement/reset variant, retaining data across median rounds.

    Every phase wire is reset before every round; data wires are reset and HF
    prepared once. This dynamic instrument has n+m wires and Rm output bits.
    It is not accepted by the current static QDK resource importer.
    """
    _integer(dimers,"dimers",1,MAX_DIMERS);_integer(phase_bits,"phase_bits",2,MAX_PHASE_BITS)
    _integer(rounds,"rounds",1,MAX_ROUNDS)
    if rounds%2==0:raise ValueError("rounds must be odd")
    body=build_retained_target_qpe(model,dimers=dimers,phase_bits=phase_bits,rounds=1,
                                  measure=False,preparation=False,analytic_powers=analytic_powers)
    circuit=QuantumCircuit(2*dimers+phase_bits,rounds*phase_bits,name=f"Recycled energy QPE x{rounds}")
    targets=list(range(phase_bits,circuit.num_qubits))
    circuit.reset(targets)
    circuit.compose(hartree_fock_preparation(dimers),targets,inplace=True)
    for r in range(rounds):
        circuit.reset(range(phase_bits))
        circuit.compose(body,range(circuit.num_qubits),inplace=True)
        circuit.measure(range(phase_bits),range(r*phase_bits,(r+1)*phase_bits))
    circuit.metadata={"dimers":dimers,"phase_bits":phase_bits,"rounds":rounds,
                      "retained_target":True,"hf_preparations":1,
                      "controlled_unitary_calls":rounds*phase_bits if analytic_powers else rounds*(2**phase_bits-1),
                      "access_mode":"analytic_dimer_powers" if analytic_powers else "repeated_U",
                      "base_U_equivalent_calls":rounds*(2**phase_bits-1),
                      "resets":2*dimers+rounds*phase_bits,"physical_resource_estimate":"unsupported_dynamic_instrument"}
    return circuit


def lower_circuit(circuit):
    # Keep explicit dynamic initialization/reset work; opt1 can delete known
    # initial resets under an implicit-zero assumption irrelevant to FT cost.
    level=0 if any(item.operation.name=='reset' for item in circuit.data) else 1
    compiled = transpile(circuit,basis_gates=ROTATION_BASIS,optimization_level=level,seed_transpiler=7)
    if len(compiled.data)>MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive export limit exceeded")
    if set(compiled.count_ops())-set(ROTATION_BASIS)-{"measure","barrier","reset"}:
        raise RuntimeError("lowering retained unsupported operations")
    return compiled


def reference_matrix(model: DimerHamiltonian,dimers=1):
    """Independent Kronecker Pauli reference, limited to <=8 data qubits."""
    _integer(dimers,"dimers",1,4)
    I=np.eye(2);X=np.array([[0,1],[1,0]]);Z=np.diag([1,-1])
    matrices={"I":I,"X":X,"Z":Z}
    size=2**(2*dimers)
    hamiltonian=np.zeros((size,size),complex)
    for coefficient,label in model.pauli_terms(dimers):
        term=np.array([[1.]],complex)
        for p in label:term=np.kron(term,matrices[p])
        hamiltonian+=coefficient*term
    return hamiltonian


def reference_spectral_law(model,dimers=1,phase_bits=5):
    _integer(phase_bits,"phase_bits",2,8)
    matrix=reference_matrix(model,dimers)
    eigenvalues,eigenvectors=np.linalg.eigh(matrix)
    hf_index=sum(1<<(2*d) for d in range(dimers))
    weights=np.abs(eigenvectors[hf_index,:])**2
    encoding=energy_encoding(model,dimers)
    phases=encoding.phase(eigenvalues)
    size=2**phase_bits
    powers=np.arange(size)
    laws=np.array([np.abs(np.exp(2j*np.pi*np.outer(powers,phase-np.arange(size)/size)).sum(axis=0)/size)**2
                   for phase in phases])
    laws/=laws.sum(axis=1,keepdims=True)
    return {"eigenvalues":eigenvalues,"weights":weights,"phases":phases,"conditional_laws":laws,
            "marginal":weights@laws,"ground_energy":float(eigenvalues[0])}


def joint_reference_distribution(model,*,dimers=1,phase_bits=3,rounds=3):
    """Spectral mixture of IID round laws, NOT fresh-HF marginal products."""
    _integer(rounds,"rounds",1,MAX_ROUNDS)
    _integer(phase_bits,"phase_bits",2,8)
    if phase_bits*rounds>12:
        raise ValueError("full joint reference limited to 12 clock bits")
    reference=reference_spectral_law(model,dimers,phase_bits)
    result=np.zeros(2**(phase_bits*rounds))
    for weight,law in zip(reference["weights"],reference["conditional_laws"],strict=True):
        term=np.array([1.])
        for _ in range(rounds):term=np.kron(law,term)
        result+=weight*term
    return result/result.sum()


def simulate_complete_group(model,*,dimers=1,phase_bits=3,rounds=3):
    circuit=build_retained_target_qpe(model,dimers=dimers,phase_bits=phase_bits,rounds=rounds,measure=False)
    if circuit.num_qubits>MAX_SIMULATOR_QUBITS:
        raise ValueError("complete-group statevector simulation exceeds 14 wires")
    return Statevector.from_instruction(circuit).probabilities(qargs=list(range(phase_bits*rounds)))


def sample_retained_target_groups(model,*,dimers=1,phase_bits=5,rounds=19,groups=2,seed=20261112,
                                  analytic_powers=False):
    """Actual sequential Qiskit state measurement with retained target.

    A single n+m-wire state is evolved/projected, clock outcomes are reset to
    zero by explicit X gates, and the SAME target proceeds to the next round.
    This implements the measured joint law of the deferred static circuit;
    it does not construct the answer from an eigenvalue reference.
    """
    _integer(dimers,"dimers",1,MAX_DIMERS);_integer(phase_bits,"phase_bits",2,MAX_PHASE_BITS)
    _integer(groups,"groups",1,MAX_GROUPS);_integer(rounds,"rounds",1,MAX_ROUNDS)
    if rounds%2==0:raise ValueError("rounds must be odd")
    if groups*rounds>MAX_SAMPLED_ROUNDS or 2*dimers+phase_bits>MAX_SIMULATOR_QUBITS:
        raise ValueError("sequential simulator exceeds bounded round/wire budget")
    body=build_retained_target_qpe(model,dimers=dimers,phase_bits=phase_bits,rounds=1,measure=False,
                                 preparation=False,analytic_powers=analytic_powers)
    hf_index=sum(1<<(phase_bits+2*d) for d in range(dimers))
    encoding=energy_encoding(model,dimers)
    rng=np.random.default_rng(seed)
    records=[]
    for group in range(groups):
        state=Statevector.from_int(hf_index,2**body.num_qubits)
        outcomes=[]
        for r in range(rounds):
            state=state.evolve(body)
            probabilities=state.probabilities(qargs=list(range(phase_bits)))
            probabilities=np.maximum(probabilities,0);probabilities/=probabilities.sum()
            outcome=int(rng.choice(2**phase_bits,p=probabilities))
            data=state.data.copy();indices=np.arange(data.size)
            data[(indices&((1<<phase_bits)-1))!=outcome]=0
            probability=float(np.vdot(data,data).real)
            state=Statevector(data/math.sqrt(probability))
            for bit in range(phase_bits):
                if (outcome>>bit)&1:
                    state=state.evolve(Operator.from_label("X"),qargs=[bit])
            outcomes.append(outcome)
        median=int(np.median(outcomes))
        records.append({"group":group,"outcomes":outcomes,"median_outcome":median,
                        "median_energy":float(encoding.energy(median/2**phase_bits))})
    return {"seed":seed,"groups":records,"estimate":min(item["median_energy"] for item in records),
            "simulator":"actual sequential Qiskit gate evolution/projection/reset; target retained",
            "full_deferred_register_statevector_allocated":False,
            "simulated_width":body.num_qubits,"total_qpe_rounds":groups*rounds}


def robust_energy_reference(model,*,dimers=1,phase_bits=5,rounds=19,groups=2):
    """Independent spectral/binomial law of minimum-of-median energy output.

    This computes a bounded classical reference, not algorithm output or a
    circuit. Conditional round distributions are IID within a retained
    spectral component; the unconditional round marginal is NOT multiplied.
    """
    _integer(rounds,"rounds",1,MAX_ROUNDS);_integer(groups,"groups",1,MAX_GROUPS)
    if rounds%2==0:raise ValueError("rounds must be odd")
    reference=reference_spectral_law(model,dimers,phase_bits)
    cdfs=np.cumsum(reference["conditional_laws"],axis=1)
    cdfs=np.clip(cdfs,0,1)
    median_cdf=sum(math.comb(rounds,k)*cdfs**k*(1-cdfs)**(rounds-k)
                   for k in range((rounds+1)//2,rounds+1))
    group_cdf=reference["weights"]@median_cdf
    minimum_cdf=1-(1-group_cdf)**groups
    probabilities=np.diff(np.r_[0.,minimum_cdf])
    if probabilities.min() < -1e-12:
        raise RuntimeError("reference CDF lost monotonicity")
    probabilities=np.maximum(probabilities,0);probabilities/=probabilities.sum()
    energies=energy_encoding(model,dimers).energy(np.arange(2**phase_bits)/2**phase_bits)
    tolerance=energy_encoding(model,dimers).span/2**phase_bits
    success=float(probabilities[np.abs(energies-reference["ground_energy"])<=tolerance+1e-13].sum())
    return {"energies":energies,"probabilities":probabilities,"ideal_success_within_grid_error":success,
            "ground_energy":reference["ground_energy"],"error_tolerance":tolerance,
            "method":"spectral mixture, conditional binomial median CDF and independent group minimum"}
