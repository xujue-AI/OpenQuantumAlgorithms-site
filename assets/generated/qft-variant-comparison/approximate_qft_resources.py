# SPDX-License-Identifier: MIT
"""Matched-output QPE comparisons for exact and omitted inverse-QFT phases.

The spectral omission bound is charged as a worst-case output-TV loss, not an
additive numerical phase error. The statistical allocation uses conservative
exact rational bounds; synthesis-TV and physical guarantees remain conditional.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from qiskit import QuantumCircuit

from .qft_qpe import _qpe_inputs, build_qft, qft_omission_bound, phase_product_example, simulate_qpe
from .resource_estimation import CircuitResourceBudget

MAX_PHASE_BITS = 8
MAX_TARGET_QUBITS = 3
MAX_SOURCE_OPERATIONS = 20_000
MAX_PRIMITIVE_OPERATIONS = 100_000
MAX_SIM_QUBITS = 10
MAX_REPETITIONS = 1001
PI_UPPER = Fraction(22,7)
BASE_SUCCESS_LOWER = Fraction(98,121)  # 8/(22/7)^2 <= 8/pi^2.
BASE_FAILURE_UPPER = 1-BASE_SUCCESS_LOWER


def _fraction_text(value:Fraction):
    """Keep exact large allocations serializable without changing process limits."""
    if max(value.numerator.bit_length(),value.denominator.bit_length())<=13_000:
        return str(value)
    return {"encoding":"exact_hex_rational","numerator_hex":hex(value.numerator),
            "denominator_hex":hex(value.denominator)}


def _integer(name,value,low,high):
    if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def _finite(name,value,low,high,*,strict_lower=False):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if not low<=value<high or (strict_lower and value==low):
        raise ValueError(f"{name} is outside its permitted range")


def omission_tv_upper(phase_bits:int,max_distance:int) -> Fraction:
    """Exact rational upper bound using pi<22/7 and measurement contractivity."""
    _integer("phase_bits",phase_bits,1,MAX_PHASE_BITS)
    _integer("max_distance",max_distance,0,phase_bits-1)
    return min(Fraction(1),sum((Fraction(phase_bits-d)*PI_UPPER/2**d
                              for d in range(max_distance+1,phase_bits)),Fraction(0)))


def majority_failure_upper(single_failure:Fraction,repetitions:int) -> Fraction:
    _integer("repetitions",repetitions,1,MAX_REPETITIONS)
    if repetitions%2!=1:
        raise ValueError("Scalar median needs an odd number of fresh trials")
    if not isinstance(single_failure,Fraction) or not 0<=single_failure<1:
        raise ValueError("Use an exact rational single-trial failure bound in [0,1)")
    return sum((Fraction(math.comb(repetitions,b))*single_failure**b*(1-single_failure)**(repetitions-b)
                for b in range((repetitions+1)//2,repetitions+1)),Fraction(0))


def median_trial_count(single_failure:Fraction,statistical_allowance:Fraction,
                       *,max_repetitions:int=MAX_REPETITIONS) -> int:
    if not isinstance(single_failure,Fraction) or not 0<=single_failure<Fraction(1,2):
        raise ValueError("No finite uniform median contract when single-trial bound is at least one half")
    if not isinstance(statistical_allowance,Fraction) or not 0<statistical_allowance<1:
        raise ValueError("Use an exact rational statistical allowance in (0,1)")
    _integer("max_repetitions",max_repetitions,1,MAX_REPETITIONS)
    for repetitions in range(1,max_repetitions+1,2):
        if majority_failure_upper(single_failure,repetitions)<=statistical_allowance:
            return repetitions
    raise ValueError("Required median count exceeds the declared repetition cap; no confidence claim was truncated")


@dataclass(frozen=True)
class QPEVariantContract:
    phase_bits:int
    max_distance:int
    phase:float=.3
    target_qubits:int=1
    synthesis_model_allowance_per_trial:float=1e-5
    statistical_failure_allowance:Fraction=Fraction(49,10000)
    physical_failure_allowance:float=.005
    total_failure_target:float=.01

    def __post_init__(self):
        _integer("phase_bits",self.phase_bits,1,MAX_PHASE_BITS)
        _integer("max_distance",self.max_distance,0,self.phase_bits-1)
        _integer("target_qubits",self.target_qubits,1,MAX_TARGET_QUBITS)
        _finite("phase",self.phase,0,1)
        if not self.phase_error_target<=self.phase<=1-self.phase_error_target:
            raise ValueError("Scalar phase median requires the complete good interval away from the branch cut")
        _finite("synthesis_model_allowance_per_trial",self.synthesis_model_allowance_per_trial,0,1,strict_lower=True)
        _finite("physical_failure_allowance",self.physical_failure_allowance,0,1,strict_lower=True)
        _finite("total_failure_target",self.total_failure_target,0,1,strict_lower=True)
        if not isinstance(self.statistical_failure_allowance,Fraction) or not 0<self.statistical_failure_allowance<1:
            raise ValueError("Statistical allowance must be an exact rational in (0,1)")
        if self.statistical_failure_allowance+Fraction.from_float(float(self.physical_failure_allowance))>Fraction.from_float(float(self.total_failure_target)):
            raise ValueError("Statistical plus physical allowances exceed the total failure target")
        # Fail at contract construction if omission leaves no permitted median.
        self.repetitions

    @property
    def phase_error_target(self):return 2.**(-self.phase_bits)

    @property
    def omission_output_tv_upper(self):return omission_tv_upper(self.phase_bits,self.max_distance)

    @property
    def single_trial_failure_upper(self):
        return BASE_FAILURE_UPPER+self.omission_output_tv_upper+Fraction.from_float(float(self.synthesis_model_allowance_per_trial))

    @property
    def repetitions(self):
        return median_trial_count(self.single_trial_failure_upper,self.statistical_failure_allowance)

    @property
    def resource_budget(self):
        return CircuitResourceBudget(self.synthesis_model_allowance_per_trial,
                                     self.physical_failure_allowance,self.repetitions)

    def as_metadata(self):
        tail=majority_failure_upper(self.single_trial_failure_upper,self.repetitions)
        return {
            "phase_bits":self.phase_bits,"max_distance":self.max_distance,"target_qubits":self.target_qubits,
            "phase":self.phase,"phase_error_target":self.phase_error_target,
            "variant":"exact" if self.max_distance==self.phase_bits-1 else "truncated",
            "total_failure_target":self.total_failure_target,
            "statistical_failure_allowance":float(self.statistical_failure_allowance),
            "statistical_allowance_exact_fraction":_fraction_text(self.statistical_failure_allowance),
            "physical_failure_allowance_all_trials":self.physical_failure_allowance,
            "synthesis_model_allowance_per_trial":self.synthesis_model_allowance_per_trial,
            "base_single_failure_upper":float(BASE_FAILURE_UPPER),
            "base_single_failure_exact_fraction":_fraction_text(BASE_FAILURE_UPPER),
            "existing_numeric_spectral_omission_bound":qft_omission_bound(self.phase_bits,self.max_distance),
            "omission_output_tv_upper":float(self.omission_output_tv_upper),
            "omission_output_tv_upper_exact_fraction":_fraction_text(self.omission_output_tv_upper),
            "single_failure_upper":float(self.single_trial_failure_upper),
            "single_failure_upper_exact_fraction":_fraction_text(self.single_trial_failure_upper),
            "odd_fresh_trials":self.repetitions,"median_failure_upper":float(tail),
            "median_failure_upper_exact_fraction":_fraction_text(tail),
            "configured_conditional_full_failure_upper":float(tail)+self.physical_failure_allowance,
            "scalar_median_branch_cut_assumption_satisfied":True,
            "output":"median of fresh scalar y/2^m values within absolute phase error 2^-m",
            "omission_semantics":"coherent spectral-norm loss converted to worst-case output TV, never an additive phase-angle bias",
            "arithmetic":"exact Fraction binomial allocation with conservative pi<22/7; no Lean proof",
            "confidence_scope":"conditional on ideal unitary/phase theorem, IID outputs, synthesis-TV scalar, correct decoder and physical-fault model",
            "certified_hardware_confidence":None,
        }


def build_variant_qpe(phase_bits:int,unitary:QuantumCircuit,preparation:QuantumCircuit,
                      max_distance:int,*,measure:bool=False) -> QuantumCircuit:
    """All controlled calls, supplied preparation, inverse QFT and final readout."""
    _integer("phase_bits",phase_bits,1,MAX_PHASE_BITS)
    _integer("max_distance",max_distance,0,phase_bits-1)
    _,_,source_upper=_qpe_inputs(phase_bits,unitary,preparation)
    if source_upper>MAX_SOURCE_OPERATIONS:
        raise ValueError("Repeated controlled access exceeds the variant source-operation cap")
    _integer("target_qubits",unitary.num_qubits,1,MAX_TARGET_QUBITS)
    if not isinstance(measure,bool):raise ValueError("measure must be bool")
    circuit=QuantumCircuit(phase_bits+unitary.num_qubits,phase_bits if measure else 0,name="QPE QFT variant")
    target=list(range(phase_bits,circuit.num_qubits))
    circuit.compose(preparation,target,inplace=True)
    circuit.h(range(phase_bits))
    controlled=unitary.to_gate(label="U").control(1)
    for bit in range(phase_bits):
        for _ in range(2**bit):circuit.append(controlled,[bit,*target])
    circuit.compose(build_qft(phase_bits,inverse=True,max_distance=max_distance),range(phase_bits),inplace=True)
    if len(circuit.data)>MAX_SOURCE_OPERATIONS:
        raise ValueError("Variant source circuit exceeds its source-operation cap")
    if measure:circuit.measure(range(phase_bits),range(phase_bits))
    circuit.metadata={"phase_bits":phase_bits,"max_distance":max_distance,
                      "input_preparation_calls":1,"controlled_unitary_calls":2**phase_bits-1,
                      "retained_inverse_qft_controlled_phases":sum(phase_bits-d for d in range(1,max_distance+1)),
                      "omitted_inverse_qft_controlled_phases":sum(phase_bits-d for d in range(max_distance+1,phase_bits)),
                      "phase_measurements":phase_bits if measure else 0,
                      "logical_zero_initialization_assumed":circuit.num_qubits,
                      "physical_initialization_cost_included":False}
    return circuit


def build_contract_trial(contract:QPEVariantContract,*,measure:bool=False):
    if not isinstance(contract,QPEVariantContract):raise ValueError("Supply a QPEVariantContract")
    unitary,preparation=phase_product_example(contract.target_qubits,contract.phase)
    return build_variant_qpe(contract.phase_bits,unitary,preparation,contract.max_distance,measure=measure)


def checked_distribution(circuit:QuantumCircuit,phase_bits:int):
    if not isinstance(circuit,QuantumCircuit):raise ValueError("Supply an explicit QuantumCircuit")
    if circuit.num_qubits>MAX_SIM_QUBITS:
        raise ValueError("Variant statevector checker exceeds its ten-wire cap")
    def count(block,depth=0):
        if depth>24 or block.num_clbits or block.num_parameters:
            raise ValueError("Variant simulation needs bounded coherent gate definitions")
        if not math.isfinite(float(block.global_phase)):
            raise ValueError("Variant simulation needs finite global phases")
        primitives={"h","x","y","z","s","sdg","t","tdg","rx","ry","rz","p","u",
                    "cx","cy","cz","ch","cp","crx","cry","crz","ccx","mcx","swap"}
        total=0
        for item in block.data:
            operation=item.operation
            if getattr(operation,"condition",None) is not None or getattr(operation,"blocks",None):
                raise ValueError("Variant simulation does not support classical control")
            if operation.name in {"reset","measure","initialize","unitary","delay","store"}:
                raise ValueError("Variant simulation needs explicit coherent gates")
            if any(not math.isfinite(float(value)) for value in operation.params):
                raise ValueError("Variant simulation parameters must be finite")
            if operation.name in primitives or operation.name=="barrier":total+=1
            elif operation.definition is not None:total+=count(operation.definition,depth+1)
            else:raise ValueError("Variant simulation does not support opaque gates")
            if total>MAX_SOURCE_OPERATIONS:
                raise ValueError("Variant statevector checker exceeds its source-operation cap")
        return total
    count(circuit)
    return simulate_qpe(circuit,phase_bits)


def fixed_grid():
    """Declared before execution: three sizes, two cutoffs and two precisions."""
    return tuple(QPEVariantContract(m,b,synthesis_model_allowance_per_trial=precision)
                 for m in (5,6,7) for b in (m-1,m-2) for precision in (1e-4,1e-5))
