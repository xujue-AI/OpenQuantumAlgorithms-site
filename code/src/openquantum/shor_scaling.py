"""Bounded matched fixed-base workloads for existing Shor arithmetic circuits.

Circuits are actual Fourier/ripple gates. The independent classical FFT
reference supplies tiny ideal sampling contracts after materialization, rather
than participating in oracle construction or claiming a generic driver bound.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Statevector

from .resource_estimation import CircuitResourceBudget, prepare_measured_circuit
from .shor import (
    analytic_phase_distribution, build_order_finding_circuit,
    controlled_multiplier_gate, factors_from_order, reconstruct_order,
    validate_instance,
)
from .shor_ripple import (
    build_order_finding_circuit_ripple, controlled_multiplier_gate_ripple,
    simulate_boolean_batch, simulate_order_finding_ripple, source_primitives,
)

MAX_INTEGER_BITS = 6
MAX_PHASE_BITS = 12
MAX_GENERATED_QUBITS = 40
MAX_COMPILED_OPERATIONS = 250_000
MAX_CASES = 5
MAX_NATIVE_QUERIES = 8
MAX_NATIVE_SECONDS = 90
MAX_SERIAL_TRIALS = 64
MAX_BOOLEAN_VALIDATION_WORK = 8_000_000

BUILDERS = {"fourier": build_order_finding_circuit,
            "ripple": build_order_finding_circuit_ripple}


@dataclass(frozen=True)
class ScalingCase:
    modulus: int
    base: int
    phase_bits: int

    def __post_init__(self):
        n = validate_instance(self.modulus, self.base)
        if n > MAX_INTEGER_BITS:
            raise ValueError("Scaling study is capped at six integer bits")
        if isinstance(self.phase_bits, bool) or not isinstance(self.phase_bits, int) or not 1 <= self.phase_bits <= MAX_PHASE_BITS:
            raise ValueError("Scaling reference precision is capped at 1..12 bits")
        if self.phase_bits + 3*n + 4 > MAX_GENERATED_QUBITS:
            raise ValueError("Scaling study generated-width cap exceeded")
        # Both existing builders' independent source-work preflights.
        if 100*n*self.phase_bits*(n+1)**2 > 500_000 or 650*n*self.phase_bits*(n+1)+10*self.phase_bits**2 > 500_000:
            raise ValueError("Existing arithmetic materialization preflight rejects this case")

    @property
    def identifier(self):
        return f"n{self.modulus}-a{self.base}-t{self.phase_bits}"

    def as_dict(self):
        n=(self.modulus-1).bit_length()
        return {**asdict(self), "integer_bits":n, "id":self.identifier,
                "phase_grid_spacing":2**(-self.phase_bits),
                "standard_2n_resolution":self.phase_bits==2*n,
                "phase_bits_selected_using_order":False}


DEFAULT_CASES = (ScalingCase(15,2,8), ScalingCase(21,2,6),
                 ScalingCase(21,2,10), ScalingCase(35,2,12), ScalingCase(33,2,12))


@dataclass(frozen=True)
class MatchedBudget:
    ideal_sampling_failure_allowance: float = .005
    physical_failure_allowance: float = .005
    synthesis_model_allowance_whole_workload: float = 8e-5

    def __post_init__(self):
        for name in ("ideal_sampling_failure_allowance","physical_failure_allowance","synthesis_model_allowance_whole_workload"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<1:
                raise ValueError(f"{name} must be finite in (0,1)")

    def resource_budget(self, trials: int):
        if isinstance(trials,bool) or not isinstance(trials,int) or not 1<=trials<=MAX_SERIAL_TRIALS:
            raise ValueError("Complete serial-trial cap exceeded")
        return CircuitResourceBudget(self.synthesis_model_allowance_whole_workload/trials,
                                     self.physical_failure_allowance,trials)


def circuit_fingerprint(circuit: QuantumCircuit) -> str:
    value={"qubits":circuit.num_qubits,"clbits":circuit.num_clbits,
           "global_phase":float(circuit.global_phase),"operations":[]}
    for item in circuit.data:
        value["operations"].append({"name":item.operation.name,
            "parameters":[float(p) for p in item.operation.params],
            "qubits":[circuit.find_bit(q).index for q in item.qubits],
            "clbits":[circuit.find_bit(q).index for q in item.clbits]})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def materialize(case: ScalingCase, method: str):
    if method not in BUILDERS:
        raise ValueError("Unknown arithmetic variant")
    started=time.perf_counter_ns()
    circuit=BUILDERS[method](case.modulus,case.base,case.phase_bits,measure=True)
    construction_ns=time.perf_counter_ns()-started
    started=time.perf_counter_ns()
    compiled=prepare_measured_circuit(circuit,optimization_level=1)
    compilation_ns=time.perf_counter_ns()-started
    if compiled.num_qubits>MAX_GENERATED_QUBITS or len(compiled.data)>MAX_COMPILED_OPERATIONS:
        raise ValueError("Actual lowered width/operation cap exceeded")
    # Audit the complete static output mapping, with no work after readout.
    mapping=[]
    measuring=False
    for item in compiled.data:
        if item.operation.name=="measure":
            measuring=True
            mapping.append((compiled.find_bit(item.qubits[0]).index,compiled.find_bit(item.clbits[0]).index))
        elif measuring and item.operation.name!="barrier":
            raise AssertionError("Circuit has work after readout")
    if mapping!=[(i,i) for i in range(case.phase_bits)]:
        raise AssertionError("Full exponent readout is missing or remapped")
    constants=[pow(case.base,1<<j,case.modulus) for j in range(case.phase_bits)]
    source=qasm3.dumps(compiled)
    record={"method":method,"case":case.as_dict(),"status":"actual_circuit_materialized",
        "declared_qubits":compiled.num_qubits,"source_structural_sha256":circuit_fingerprint(circuit),
        "source_gate_counts":dict(circuit.count_ops()),"primitive_gate_counts":dict(compiled.count_ops()),
        "primitive_operations_per_trial":len(compiled.data),"dependency_depth":compiled.depth(),
        "data_readouts_per_trial":len(mapping),"readout_mapping":mapping,
        "controlled_multiplier_constants":constants,"nonidentity_multipliers":sum(c!=1 for c in constants),
        "qasm_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "compiled_global_phase_radians":float(compiled.global_phase),
        "qasm_global_phase_scope":"Pinned exporter omits overall global phase; complete measured-trial probabilities retained, not reusable controlled-oracle phase",
        "construction_wall_ns":construction_ns,"compilation_wall_ns":compilation_ns,
        "transpiler":{"optimization_level":1,"seed_transpiler":7},
        "included":["logical-zero input assumption; X prepares modular register1","t Hadamards prepare exponent superposition",
                    "every defined controlled modular multiply and inverse cleanup","all work/constant/carry/borrow registers",
                    "full inverse QFT","all t exponent measurements"],
        "hardware_initialization":"All wires assumed initially logical zero; no calibrated physical initialization/reset protocol"}
    return circuit,compiled,source,record


def independent_sampling_contract(case: ScalingCase, budget: MatchedBudget=MatchedBudget()) -> dict:
    """FFT of literal modular-power residue indicators, not the circuit builder.

    This deliberately bounded O(N*2^t*t) classical work is charged as diagnostic
    CPU time and never used to choose the circuit's modular powers or precision.
    """
    started=time.perf_counter_ns()
    dimension=1<<case.phase_bits
    residues=np.empty(dimension,dtype=np.int64)
    value=1
    for x in range(dimension):
        residues[x]=value
        value=value*case.base%case.modulus
    probabilities=np.zeros(dimension)
    for residue in np.unique(residues):
        indicator=(residues==residue).astype(float)
        probabilities+=np.abs(np.fft.fft(indicator)/dimension)**2
    value=1
    order=None
    for exponent in range(1,case.modulus):
        value=value*case.base%case.modulus
        if value==1:
            order=exponent
            break
    if order is None:
        raise AssertionError("Coprime reference orbit did not close")
    geometric=analytic_phase_distribution(order,case.phase_bits)
    discrepancy=float(np.max(np.abs(probabilities-geometric)))
    if discrepancy>1e-11 or abs(float(probabilities.sum())-1)>1e-11:
        raise AssertionError("Independent FFT and geometric reference disagree")
    successful=[]
    p=0.0
    validation_work=0
    for y,weight in enumerate(probabilities):
        period=reconstruct_order(y,case.phase_bits,case.modulus,case.base)
        factors=factors_from_order(case.modulus,case.base,period)
        validation_work+=1
        if any(case.modulus%factor or not 1<factor<case.modulus for factor in factors):
            raise AssertionError("Classical postprocessor returned an invalid factor")
        if factors:
            successful.append(y)
            p+=float(weight)
    if p<=0:
        trials=None
    elif p>=1:
        trials=1
    else:
        trials=math.ceil(math.log(budget.ideal_sampling_failure_allowance)/math.log1p(-p))
    supported=trials is not None and trials<=MAX_SERIAL_TRIALS
    return {"case":case.as_dict(),"reference_order":order,"independent_method":"FFT of actual classical modular-power residue indicator arrays",
        "fft_geometric_max_probability_error":discrepancy,"probabilities":probabilities.tolist(),
        "successful_exponent_outcomes":successful,"ideal_factor_success_probability_per_trial":p,
        "complete_serial_trials":trials,"finite_retry_projection_supported":supported,
        "ideal_sampling_failure_after_trials":(1-p)**trials if trials is not None else None,
        "requested_budgets":asdict(budget),"classical_reference_wall_ns":time.perf_counter_ns()-started,
        "synthesis_model_allowance_per_trial":budget.synthesis_model_allowance_whole_workload/trials if supported else None,
        "reference_modular_power_values_evaluated":dimension,"postprocessing_outcomes_evaluated":validation_work,
        "reference_used_to_construct_circuit":False,"reference_used_to_choose_precision":False,
        "scope":"bounded numerical ideal fixed-base contract with the actual CF/modular-verification/gcd rule; no random-base or recursive factoring-driver bound",
        "certified_combined_factor_success":None,
        "unsupported_reason":None if supported else "Zero ideal factor success" if trials is None else "Required complete trials exceed64"}


def tiny_variant_checks() -> dict:
    """Independently check both actual small circuits and clean controls."""
    from .shor import simulate_phase_distribution
    fourier=simulate_phase_distribution(3,2,2)
    ripple=simulate_order_finding_ripple(3,2,2,qiskit=True)
    difference=float(np.max(np.abs(np.array(fourier["probabilities"])-np.array(ripple["probabilities"]))))
    if difference>1e-11:
        raise AssertionError("Small actual arithmetic variants disagree")
    coherent={}
    for name,gate in (("fourier",controlled_multiplier_gate(3,2)),("ripple",controlled_multiplier_gate_ripple(3,2))):
        initial=np.zeros(1<<gate.num_qubits,dtype=complex)
        initial[2]=1/math.sqrt(2)
        initial[3]=1j/math.sqrt(2)
        expected=np.zeros_like(initial)
        expected[2]=1/math.sqrt(2)
        expected[5]=1j/math.sqrt(2)
        actual=Statevector(initial).evolve(gate)
        error=float(np.max(np.abs(actual.data-expected)))
        if error>1e-11:
            raise AssertionError("Coherent multiplier or clean-work restoration failed")
        coherent[name]={"max_complex_amplitude_error":error,"tested_modulus":3,"tested_multiplier":2}
    return {"status":"checked","actual_full_circuit_case":{"modulus":3,"base":2,"phase_bits":2},
            "fourier":fourier,"ripple":ripple,"variant_probability_difference":difference,"coherent_multipliers":coherent,
            "coverage":"Actual tiny phase and coherent-control checks; not simulation or proof of larger materialized circuits"}


def bounded_ripple_multiplier_check(case: ScalingCase) -> dict:
    values=list(range(case.modulus))
    pairs=[(control,value) for control in (0,1) for value in values]
    constants=sorted({pow(case.base,1<<j,case.modulus) for j in range(case.phase_bits)})
    work=0
    checks=[]
    for constant in constants:
        gate=controlled_multiplier_gate_ripple(case.modulus,constant)
        circuit=QuantumCircuit(gate.num_qubits)
        circuit.append(gate,range(gate.num_qubits))
        estimate=len(source_primitives(circuit))*len(pairs)
        if work+estimate>MAX_BOOLEAN_VALIDATION_WORK:
            return {"status":"partial_validation_capped","checked_multipliers":checks,
                    "all_materialized_power_constants_checked":False,"gate_input_work":work,
                    "reason":"Whole-case Boolean validation work cap reached; remaining multipliers not checked"}
        outputs=simulate_boolean_batch(circuit,[control+(value<<1) for control,value in pairs])
        expected=[control+(((constant*value%case.modulus) if control else value)<<1) for control,value in pairs]
        if outputs!=expected:
            raise AssertionError("Actual Boolean power multiplier disagrees with modular arithmetic")
        work+=estimate
        checks.append({"constant":constant,"control_input_cases":len(pairs),"all_valid_inputs_tested":True,
                       "all_tested_work_restored_zero":True,"gate_input_work":estimate})
    return {"status":"checked","input_values":values,"control_input_cases_per_constant":len(pairs),
            "distinct_power_constants":constants,"checked_multipliers":checks,"gate_input_work":work,
            "all_materialized_power_constants_checked":True,"all_valid_inputs_tested":True,"all_tested_work_restored_zero":True,
            "scope":"Exact Boolean trajectories; tiny checks separately test coherent amplitudes"}
