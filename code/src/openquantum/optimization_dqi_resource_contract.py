"""Pure finite contract for the accepted n=2/3 structured-DQI resource task.

No quantum backend, estimator, producer, simulator or package initializer is
imported here.  Public runtime uses embedded scientific task/cost constants;
``load_design`` is an optional source-preparation fingerprint check only.
The private design JSON is neither a runtime dependency nor launch authority.
All future native calls must be charged before entry.  A failed native return
spends that exact key and never authorizes an implicit retry.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType


DESIGN_RELATIVE = "docs/coordination/optimization-dqi-resources/task-proposal.json"
DESIGN_SHA256 = "fdf573da42a069170c17984452dd4b3918532df09825b070fe1ce32e861b8eb3"
MAX_INPUT_FILE_BYTES = 6 << 20
MAX_JSON_NODES = 8000
MAX_JSON_DEPTH = 20
EPSILON_FLOAT_HEX = "0x1.a36e2eb1c432dp-17"
REFERENCE_PADDING = Fraction(1, 10_000_000_000)
MATRIX_GUARD = Fraction(1, 1_000_000_000_000)
WORD_ERROR_CEILING = Fraction(1, 20_000)
ALLOWANCE = Fraction(1, 400)
_TOTAL_MISS_TARGET = Fraction(1, 100)
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FRACTION_PATTERN = re.compile(r"[0-9]+/[0-9]+\Z")
_SOURCE_OPERATIONS = frozenset({"h", "x", "z", "t", "tdg", "ry", "cx", "reset", "measure", "barrier"})
_ONE_QUBIT = frozenset({"h", "x", "z", "t", "tdg"})
DESIGN_CANONICAL_SHA256 = '918f02c990e337d32535a598aa194064681a17c3bcc60a621850a1616b59b668'
PUBLIC_TASK_SHA256 = '5197f1ae5612153b57ff8a4ecc026fe9b9ee2ff5257a2a7ce796d81e786cb0ea'
_PUBLIC_TASK_JSON = '{"budget_per_case":{"factory_availability_whole_task":"1/400","formal_interval_certificate":false,"ideal_algorithmic_miss":"1/400","per_matrix_numerical_guard":"1/1000000000000","physical_fault_whole_task":"1/400","realization_whole_fixed_draws":"1/400","reference_numerical_padding":"1/10000000000","total_miss_target":"1/100"},"cases":[{"additional_patch_initializations_per_draw":17,"assignment_readout_mapping":[{"classical_bit":0,"wire":3},{"classical_bit":1,"wire":4}],"case_id":"n2_optimum99","compiled_result":null,"direct_trace_physical_max_error_fraction":"1/2000","fixed_complete_draws":5,"ideal_miss":"1/1024","ideal_optimum_probability":"3/4","n":2,"objective_target":2,"physical_result":null,"planned_source_counts_whole_task":{"barrier":35,"cx":190,"h":60,"measure":10,"reset":30,"ry":25,"t":100,"tdg":75,"x":50,"z":5},"realization_ceiling_per_draw":"1/2000","reserved_logical_patches":23,"rotation_occurrences":[{"angle_hex":"0x1.0c152382d7366p+0","operation_index":7,"wire":0},{"angle_hex":"0x1.3b2028082e8d5p-1","operation_index":9,"wire":1},{"angle_hex":"-0x1.3b2028082e8d5p-1","operation_index":11,"wire":1},{"angle_hex":"0x1.921fb54442d17p-1","operation_index":31,"wire":2},{"angle_hex":"-0x1.921fb54442d17p-1","operation_index":33,"wire":2}],"source_counts_per_draw":{"barrier":7,"cx":38,"h":12,"measure":2,"reset":6,"ry":5,"t":20,"tdg":15,"x":10,"z":1},"source_explicit_t_tdg_per_draw":35,"source_initial_reset_wires":[0,1,2,3,4,5],"source_operation_files":["assets/generated/optimization/dqi-structured/small/n2/dqi.qasm","assets/generated/optimization/dqi-structured/small/n2/operations.ndjson","assets/generated/optimization/dqi-structured/small/n2/ledger.json"],"source_operations_per_draw":116,"source_operations_whole_task":580,"source_unitary_per_draw":101,"source_unitary_whole_task":505,"source_wires":6},{"additional_patch_initializations_per_draw":26,"assignment_readout_mapping":[{"classical_bit":0,"wire":4},{"classical_bit":1,"wire":5},{"classical_bit":2,"wire":6}],"case_id":"n3_optimum99","compiled_result":null,"direct_trace_physical_max_error_fraction":"1/1200","fixed_complete_draws":3,"ideal_miss":"1/1000","ideal_optimum_probability":"9/10","n":3,"objective_target":3,"physical_result":null,"planned_source_counts_whole_task":{"barrier":21,"cx":360,"h":117,"measure":9,"reset":27,"ry":21,"t":216,"tdg":162,"x":72,"z":3},"realization_ceiling_per_draw":"1/1200","reserved_logical_patches":35,"rotation_occurrences":[{"angle_hex":"0x1.dac670561bb4fp-1","operation_index":10,"wire":0},{"angle_hex":"0x1.0c152382d7366p-1","operation_index":12,"wire":1},{"angle_hex":"-0x1.0c152382d7366p-1","operation_index":14,"wire":1},{"angle_hex":"0x1.3b2028082e8d5p-1","operation_index":34,"wire":2},{"angle_hex":"-0x1.3b2028082e8d5p-1","operation_index":36,"wire":2},{"angle_hex":"0x1.921fb54442d17p-1","operation_index":88,"wire":3},{"angle_hex":"-0x1.921fb54442d17p-1","operation_index":90,"wire":3}],"source_counts_per_draw":{"barrier":7,"cx":120,"h":39,"measure":3,"reset":9,"ry":7,"t":72,"tdg":54,"x":24,"z":1},"source_explicit_t_tdg_per_draw":126,"source_initial_reset_wires":[0,1,2,3,4,5,6,7,8],"source_operation_files":["assets/generated/optimization/dqi-structured/small/n3/dqi.qasm","assets/generated/optimization/dqi-structured/small/n3/operations.ndjson","assets/generated/optimization/dqi-structured/small/n3/ledger.json"],"source_operations_per_draw":336,"source_operations_whole_task":1008,"source_unitary_per_draw":317,"source_unitary_whole_task":951,"source_wires":9}],"compiler":{"adaptive_retry":false,"global_phase":"preserve in word/state record; export separately if QASM omits it","native_synthesis_attempts_per_unique_key":1,"opaque_instructions_allowed":false,"rotation_targets":[{"angle_hex":"0x1.0c152382d7366p+0","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"0x1.3b2028082e8d5p-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"-0x1.3b2028082e8d5p-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"0x1.921fb54442d17p-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"-0x1.921fb54442d17p-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"0x1.dac670561bb4fp-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"0x1.0c152382d7366p-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"},{"angle_hex":"-0x1.0c152382d7366p-1","epsilon_requested_float_hex":"0x1.a36e2eb1c432dp-17","epsilon_requested_fraction":"1/80000","gate":"ry","guarded_frobenius_ceiling_fraction":"1/20000"}],"routing":"deterministic adjacent SWAP chain, routed CX, inverse SWAP chain; restore source wire placement after each CX"},"cpu_pass_scope":{"compile_distinct_circuits":2,"direct_qdk_ledger_checks":2,"execute_qpu":false,"new_compiled_state_checks":2,"new_size_or_degree":false,"rerun_accepted_ideal_validator_or_decoder_operator":false,"sample_assignment_draws":false},"fixed_candidate":{"candidate_count":1,"data_code_distance":15,"factory_count":1,"identity":{"identity_scope":"exact native factory ISA/T cost-model values; no distillation-layout certificate","native_factory_isa":[{"arity":1,"encoding":1,"error_rate":{"float_hex":"0x1.88ea67f531f32p-56"},"expected_output_error_hex":"0x1.88ea67f531f32p-56","expected_space":236623,"expected_time_ns":330000,"id":1028,"space":{"integer":236623},"time":{"integer":330000}}],"native_factory_isa_sha256":"2c94c77a1cb4779ac6b4d2f49d9312d70974b9ff82650b4f252ea52fe939c102","native_sources":{"_isa_enumeration.py":"f35a4248c0d29174daac033d12f8296ad6c3b5a203a4866703d763a5b65aa97e","models/factories/_round_based.py":"c5fcb478c86534cf9888bc8a9b9d3e5a733355bf92c41072228d99fdc0d19110","models/qec/_surface_code.py":"3530f92f328e325d6d0c3e245f8688c11956fa6f63b1e25455b5d14cd844ed35"},"native_t_instruction":{"arity":1,"encoding":1,"error_rate":{"float_hex":"0x1.88ea67f531f32p-56"},"expected_output_error_hex":"0x1.88ea67f531f32p-56","expected_space":236623,"expected_time_ns":330000,"id":1028,"space":{"integer":236623},"time":{"integer":330000}},"native_t_instruction_sha256":"ad99b8ce36f8b556c05d877305e97e5e654f31c0d3756f0630c5bb2b44b184cb","surface_code_query":{"code_cycle_override":1000,"distance":15,"error_correction_threshold":0.01}},"native_factory_isa_sha256":"2c94c77a1cb4779ac6b4d2f49d9312d70974b9ff82650b4f252ea52fe939c102","native_result_copies_required":1,"native_result_fields_uninterpreted":["runs","states","error_rate"],"native_t_instruction_sha256":"ad99b8ce36f8b556c05d877305e97e5e654f31c0d3756f0630c5bb2b44b184cb","restore_scope":"exact cost fields in a fresh local QDK context; original pipeline/provenance graph not restored"},"frozen_profile":{"code_cycle_time_ns":1000,"controller_failure_per_decision":1e-15,"factory_feedback_rounds_reserved":3,"factory_pipeline_failure_bound":0.01,"initialization_residual_prefactor":0.1,"maximum_code_distance":49,"memory_prefactor":0.1,"memory_threshold":0.01,"physical_error_rate":0.0001,"physical_gate_time_ns":100,"physical_initialization_time_ns":100,"physical_measurement_time_ns":500,"reaction_time_ns":10000},"frozen_readiness_scenario":{"complete_pipeline_rejection_bound":"1/100","independent_complete_attempts_assumed":true,"native_per_round_requirement_is_whole_pipeline_certificate":false,"output_error_source":"archived native T instruction, separately counted once per state","retry_rule":"smallest K<=16 with N_T*(1/100)**K <= 1/400"},"limits":{"capture_log_bytes":65536,"cleanup_wall_seconds":5,"compiled_operations_per_case":20000,"compiled_operations_total":40000,"direct_native_trace_estimate_calls":2,"distinct_source_circuits":2,"factory_frontier_candidates":0,"factory_frontier_queries":0,"factory_pipeline_candidates":0,"factory_retry_cap":16,"internal_native_iteration_counter_available":false,"macro_classes_per_case":32,"native_call_wall_seconds":5,"outer_wall_seconds":66,"physical_configurations":1,"qdk_top_level_estimate_calls":0,"quantum_array_bytes":33554432,"retained_artifact_bytes":8388608,"retained_file_count":32,"retention_wall_seconds":1,"retry_or_fallback_authorized":false,"sampled_child_rss_bytes":1073741824,"source_operations_per_case":336,"source_qubits":9,"synthesis_native_calls":8,"synthesized_word_operations":2000,"task_candidate_evaluations":2,"unique_ry_targets":8,"unique_word_cache_operations":16000,"verification_matrix_dimension":2,"verification_state_amplitudes":512,"work_wall_seconds":60},"native_error_composition":"UnionBound; physical ledger only, synthesis discrepancy separately allocated","output":{"conditional_on":["numerical error estimates bounding realization discrepancy","frozen physical/controller/readiness scenario","faithful assignment-bit transfer","correct classical scoring and selection"],"quality":"return assignment with f(x)=n","simultaneous_case_success_claim":false,"success_target_per_case":"99/100","ties":"smallest integer assignment among maximum scores"},"parent_public_result_commit":"0758b8694b213eb0ee670df053233ba9f5fe36d6","source_inputs":{"LICENSE":{"bytes":1093,"sha256":"5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"},"assets/generated/optimization/compiled-replay/physical-summary.json":{"bytes":1980,"sha256":"33b6ba297cfd7313f0d29aff1f87e4f3b4b9e9290e0fcf2ff110866c370da7b1"},"assets/generated/optimization/dqi-structured/bundle.json":{"bytes":4601,"sha256":"1e01422fc8a0961bc1771ceedaffacfdf90a9c1a3bc9a0b99efc2aa3f4100850"},"assets/generated/optimization/dqi-structured/small/n2/dqi.qasm":{"bytes":1693,"sha256":"af09988e10bce0969e32527515b0a3a09a5105e35f0a253872dfa7730decd432"},"assets/generated/optimization/dqi-structured/small/n2/ledger.json":{"bytes":4529,"sha256":"94c4ad862c3939e0583dc23e116c2a7a29580730c32a9650c7e73030c2e4dad1"},"assets/generated/optimization/dqi-structured/small/n2/operations.ndjson":{"bytes":2247,"sha256":"6fa145372a48336ce4761ae142c16a3da50235f2355268c10823cbb45ed9ca8a"},"assets/generated/optimization/dqi-structured/small/n3/dqi.qasm":{"bytes":4299,"sha256":"e85801409197e7195f9320ecba963e72dcf3f5c4d4b7bec52ff78941c7a9f23a"},"assets/generated/optimization/dqi-structured/small/n3/ledger.json":{"bytes":4584,"sha256":"fb7d69f2b6cad7f550cfbacab0eaa660afdc917f24afaf7b95a0dda4c483429c"},"assets/generated/optimization/dqi-structured/small/n3/operations.ndjson":{"bytes":6191,"sha256":"3a9a0ddfefd2b4f468100458e784012593a2404691e3f1b53bd318b409b2cfaf"},"assets/generated/optimization/dqi-structured/small/validated-small.json":{"bytes":17715,"sha256":"e7aeadafbb838e00c22dbef400b7d0bf637d45a7aad4ff107dac79f7662b7d42"},"assets/generated/optimization/word-transport/baseline/physical-dag.json.gz":{"bytes":5027908,"sha256":"731791448fcfd3e7fc2df18efa0d7d0db12c49254a709bb58d0286a93d1cca34"},"requirements-lock.txt":{"bytes":1974,"sha256":"6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"},"src/openquantum/instrument_resources.py":{"bytes":52381,"sha256":"fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"},"src/openquantum/optimization_dqi_structured.py":{"bytes":21438,"sha256":"0daaf35c01fbf83bcdfad98c85f6af94420202d7b546c7892615bd18ff1ec028"}},"status":"source_prepared_unexecuted","task_id":"structured-dqi-fixed-d15-optimum99-v1"}'
_FROZEN_IDENTITY_JSON = '{"identity_scope":"exact native factory ISA/T cost-model values; no distillation-layout certificate","native_factory_isa":[{"arity":1,"encoding":1,"error_rate":{"float_hex":"0x1.88ea67f531f32p-56"},"expected_output_error_hex":"0x1.88ea67f531f32p-56","expected_space":236623,"expected_time_ns":330000,"id":1028,"space":{"integer":236623},"time":{"integer":330000}}],"native_factory_isa_sha256":"2c94c77a1cb4779ac6b4d2f49d9312d70974b9ff82650b4f252ea52fe939c102","native_sources":{"_isa_enumeration.py":"f35a4248c0d29174daac033d12f8296ad6c3b5a203a4866703d763a5b65aa97e","models/factories/_round_based.py":"c5fcb478c86534cf9888bc8a9b9d3e5a733355bf92c41072228d99fdc0d19110","models/qec/_surface_code.py":"3530f92f328e325d6d0c3e245f8688c11956fa6f63b1e25455b5d14cd844ed35"},"native_t_instruction":{"arity":1,"encoding":1,"error_rate":{"float_hex":"0x1.88ea67f531f32p-56"},"expected_output_error_hex":"0x1.88ea67f531f32p-56","expected_space":236623,"expected_time_ns":330000,"id":1028,"space":{"integer":236623},"time":{"integer":330000}},"native_t_instruction_sha256":"ad99b8ce36f8b556c05d877305e97e5e654f31c0d3756f0630c5bb2b44b184cb","surface_code_query":{"code_cycle_override":1000,"distance":15,"error_correction_threshold":0.01}}'
FROZEN_INPUTS = MappingProxyType({'LICENSE': (1093, '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'),
 'assets/generated/optimization/compiled-replay/physical-summary.json': (1980,
                                                                         '33b6ba297cfd7313f0d29aff1f87e4f3b4b9e9290e0fcf2ff110866c370da7b1'),
 'assets/generated/optimization/dqi-structured/bundle.json': (4601,
                                                              '1e01422fc8a0961bc1771ceedaffacfdf90a9c1a3bc9a0b99efc2aa3f4100850'),
 'assets/generated/optimization/dqi-structured/small/n2/dqi.qasm': (1693,
                                                                    'af09988e10bce0969e32527515b0a3a09a5105e35f0a253872dfa7730decd432'),
 'assets/generated/optimization/dqi-structured/small/n2/ledger.json': (4529,
                                                                       '94c4ad862c3939e0583dc23e116c2a7a29580730c32a9650c7e73030c2e4dad1'),
 'assets/generated/optimization/dqi-structured/small/n2/operations.ndjson': (2247,
                                                                             '6fa145372a48336ce4761ae142c16a3da50235f2355268c10823cbb45ed9ca8a'),
 'assets/generated/optimization/dqi-structured/small/n3/dqi.qasm': (4299,
                                                                    'e85801409197e7195f9320ecba963e72dcf3f5c4d4b7bec52ff78941c7a9f23a'),
 'assets/generated/optimization/dqi-structured/small/n3/ledger.json': (4584,
                                                                       'fb7d69f2b6cad7f550cfbacab0eaa660afdc917f24afaf7b95a0dda4c483429c'),
 'assets/generated/optimization/dqi-structured/small/n3/operations.ndjson': (6191,
                                                                             '3a9a0ddfefd2b4f468100458e784012593a2404691e3f1b53bd318b409b2cfaf'),
 'assets/generated/optimization/dqi-structured/small/validated-small.json': (17715,
                                                                             'e7aeadafbb838e00c22dbef400b7d0bf637d45a7aad4ff107dac79f7662b7d42'),
 'assets/generated/optimization/word-transport/baseline/physical-dag.json.gz': (5027908,
                                                                                '731791448fcfd3e7fc2df18efa0d7d0db12c49254a709bb58d0286a93d1cca34'),
 'requirements-lock.txt': (1974,
                           '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'),
 'src/openquantum/instrument_resources.py': (52381,
                                             'fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863'),
 'src/openquantum/optimization_dqi_structured.py': (21438,
                                                    '0daaf35c01fbf83bcdfad98c85f6af94420202d7b546c7892615bd18ff1ec028')})
LIMITS = MappingProxyType({'capture_log_bytes': 65536,
 'cleanup_wall_seconds': 5,
 'compiled_operations_per_case': 20000,
 'compiled_operations_total': 40000,
 'direct_native_trace_estimate_calls': 2,
 'distinct_source_circuits': 2,
 'factory_frontier_candidates': 0,
 'factory_frontier_queries': 0,
 'factory_pipeline_candidates': 0,
 'factory_retry_cap': 16,
 'internal_native_iteration_counter_available': False,
 'macro_classes_per_case': 32,
 'native_call_wall_seconds': 5,
 'outer_wall_seconds': 66,
 'physical_configurations': 1,
 'qdk_top_level_estimate_calls': 0,
 'quantum_array_bytes': 33554432,
 'retained_artifact_bytes': 8388608,
 'retained_file_count': 32,
 'retention_wall_seconds': 1,
 'retry_or_fallback_authorized': False,
 'sampled_child_rss_bytes': 1073741824,
 'source_operations_per_case': 336,
 'source_qubits': 9,
 'synthesis_native_calls': 8,
 'synthesized_word_operations': 2000,
 'task_candidate_evaluations': 2,
 'unique_ry_targets': 8,
 'unique_word_cache_operations': 16000,
 'verification_matrix_dimension': 2,
 'verification_state_amplitudes': 512,
 'work_wall_seconds': 60})
PROFILE = MappingProxyType({'code_cycle_time_ns': 1000,
 'controller_failure_per_decision': 1e-15,
 'factory_feedback_rounds_reserved': 3,
 'factory_pipeline_failure_bound': 0.01,
 'initialization_residual_prefactor': 0.1,
 'maximum_code_distance': 49,
 'memory_prefactor': 0.1,
 'memory_threshold': 0.01,
 'physical_error_rate': 0.0001,
 'physical_gate_time_ns': 100,
 'physical_initialization_time_ns': 100,
 'physical_measurement_time_ns': 500,
 'reaction_time_ns': 10000})


FROZEN_IDENTITY = json.loads(_FROZEN_IDENTITY_JSON)
CASE_PARAMETERS = tuple(json.loads(_PUBLIC_TASK_JSON)["cases"])
ROTATION_TARGETS = tuple(
    (target["gate"], target["angle_hex"], target["epsilon_requested_float_hex"])
    for target in json.loads(_PUBLIC_TASK_JSON)["compiler"]["rotation_targets"]
)
ALLOWED_SYNTHESIS_KEYS = frozenset(ROTATION_TARGETS)


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _case_n(n):
    if type(n) is not int or n not in (2, 3):
        raise ValueError("Only the fixed n=2/3 resource cases are supported")
    return n


def _finite_fraction(value, name):
    if type(value) is Fraction:
        if value.numerator.bit_length() > 256 or value.denominator.bit_length() > 256:
            raise ValueError(f"{name} fraction exceeds its 256-bit bound")
        result = value
    elif type(value) is int:
        if value.bit_length() > 256:
            raise ValueError(f"{name} integer exceeds its 256-bit bound")
        result = Fraction(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        result = Fraction.from_float(value)
    else:
        raise ValueError(f"{name} must be a bounded finite real number")
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def fraction_text(text):
    """Parse the finite exact fraction grammar used by the scientific recipe."""
    if (not isinstance(text, str) or len(text) > 64 or not text.isascii()
            or _FRACTION_PATTERN.fullmatch(text) is None):
        raise ValueError("Expected at most 64 ASCII numerator/denominator characters")
    numerator_text, denominator_text = text.split("/")
    numerator, denominator = int(numerator_text), int(denominator_text)
    if denominator == 0 or numerator.bit_length() > 256 or denominator.bit_length() > 256:
        raise ValueError("Fraction denominator/bit bound is invalid")
    return Fraction(numerator, denominator)


def canonical_bytes(value):
    _check_json_shape(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _no_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON key")
        output[key] = value
    return output


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")


def _check_json_shape(value):
    """Reject oversized/non-JSON values before canonical serialization."""
    count = 0
    encoded_upper = 0
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise ValueError("JSON structure exceeds its node/depth envelope")
        if type(item) is dict:
            if len(item) > MAX_JSON_NODES-count-len(pending):
                raise ValueError("JSON container exceeds its node envelope")
            if any(type(key) is not str or len(key) > 1024 for key in item):
                raise ValueError("JSON keys must be bounded strings")
            encoded_upper += 2 + sum(12*len(key)+6 for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if len(item) > MAX_JSON_NODES-count-len(pending):
                raise ValueError("JSON container exceeds its node envelope")
            encoded_upper += len(item)+2
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("JSON number must be finite")
        elif type(item) is int and item.bit_length() > 256:
            raise ValueError("JSON integer exceeds its bit bound")
        elif type(item) is str and len(item) > 65536:
            raise ValueError("JSON string exceeds its length bound")
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("Expected exact ordinary JSON value types")
        if type(item) is str:
            encoded_upper += 12*len(item)+2
        elif type(item) in (int, float, bool) or item is None:
            encoded_upper += 96
        if encoded_upper > MAX_INPUT_FILE_BYTES:
            raise ValueError("JSON canonical representation exceeds its byte envelope")
    return value


def parse_json(raw):
    """Decode bounded finite JSON without accepting duplicate keys or NaN."""
    if not isinstance(raw, bytes) or len(raw) > MAX_INPUT_FILE_BYTES:
        raise ValueError("JSON input exceeds its bounded byte envelope")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded JSON") from exc
    _check_json_shape(value)
    return value


def public_task():
    """Return fresh ordinary parameters; no private file or authorization hook."""
    return json.loads(_PUBLIC_TASK_JSON)


def frozen_identity():
    return json.loads(_FROZEN_IDENTITY_JSON)


def check_identity(identity):
    """Reject altered cost fields, float bits, types or identity provenance."""
    if canonical_bytes(identity) != _FROZEN_IDENTITY_JSON.encode():
        raise ValueError("The single frozen native factory cost identity changed")
    return identity


def check_task(task):
    try:
        digest = hashlib.sha256(canonical_bytes(task)).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Task parameters must be the exact finite public contract") from exc
    if digest != PUBLIC_TASK_SHA256:
        raise ValueError("The fixed public resource task parameters changed")
    return task


def check_design(design):
    """Check optional source-design identity; it grants no execution authority."""
    try:
        digest = hashlib.sha256(canonical_bytes(design)).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Source design must be the exact finite reviewed object") from exc
    if digest != DESIGN_CANONICAL_SHA256:
        raise ValueError("The immutable source-only ff87 resource design changed")
    return design


def _relative_parts(relative):
    if (not isinstance(relative, str) or not relative or len(relative) > 1024
            or not relative.isascii() or "\\" in relative or relative.startswith("/")):
        raise ValueError("Expected a bounded ordinary relative path")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Relative input path is not canonical or escapes its root")
    return parts


def bounded_read(root, relative, expected_sha256=None, expected_bytes=None, *, cap=MAX_INPUT_FILE_BYTES):
    """Read an exact regular file via no-follow directory descriptors.

    Every component under the supplied root is opened with O_NOFOLLOW.  The
    root itself must be a regular directory, not a symlink.  Bounded reads and
    post-read fstat/hash checks close size and byte-replacement races.
    """
    _integer(cap, "input byte cap", 1, MAX_INPUT_FILE_BYTES)
    parts = _relative_parts(relative)
    if expected_bytes is not None:
        _integer(expected_bytes, "expected input bytes", 0, cap)
    if expected_sha256 is not None and (not isinstance(expected_sha256, str)
            or _SHA_PATTERN.fullmatch(expected_sha256) is None):
        raise ValueError("Expected an exact lowercase SHA-256")
    try:
        root = Path(root).absolute()
    except (TypeError, ValueError) as exc:
        raise ValueError("Input root must be an ordinary path") from exc
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Input root must be an ordinary directory")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptors = []
    file_descriptor = None
    try:
        current = os.open(root, flags | os.O_DIRECTORY)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(component, flags | os.O_DIRECTORY, dir_fd=current)
            descriptors.append(current)
        file_descriptor = os.open(parts[-1], flags, dir_fd=current)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError("Input must be a regular file inside its byte cap")
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError("Input file size differs from the frozen value")
        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = None
            raw = handle.read(cap + 1)
            after = os.fstat(handle.fileno())
        if len(raw) > cap or len(raw) != before.st_size:
            raise ValueError("Input changed size or exceeded the read byte cap")
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Input file changed while being read")
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Input file hash differs from its frozen value")
        return raw
    except OSError as exc:
        raise ValueError("Input path is unavailable, nonregular or contains a symlink") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def load_design(root, relative=DESIGN_RELATIVE):
    """Optional source-preparation check; ordinary driver runtime never needs it."""
    return check_design(parse_json(bounded_read(root, relative, DESIGN_SHA256, 17981)))


def verify_frozen_inputs(root):
    """Hash the fourteen immutable inputs; execute none of their producers."""
    return {name: bounded_read(root, name, digest, size)
            for name, (size, digest) in FROZEN_INPUTS.items()}


def _case_parameters(n):
    _case_n(n)
    return next(case for case in public_task()["cases"] if case["n"] == n)


def synthesis_key(angle_hex):
    if not isinstance(angle_hex, str) or len(angle_hex) > 32:
        raise ValueError("An approved canonical binary angle is required")
    key = ("ry", angle_hex, EPSILON_FLOAT_HEX)
    if key not in ALLOWED_SYNTHESIS_KEYS:
        raise ValueError("Unknown rotation angle or synthesis allowance")
    return key


def validate_operations(n, records):
    """Validate the exact accepted serialized stream without loading a circuit."""
    n = _case_n(n)
    case = _case_parameters(n)
    if not isinstance(records, (list, tuple)) or len(records) != case["source_operations_per_draw"]:
        raise ValueError("Accepted source operation count changed")
    qwidth = 3*n
    immutable = []
    counts = Counter()
    readouts = []
    rotations = []
    measured = False
    for index, record in enumerate(records):
        if not isinstance(record, (list, tuple)) or len(record) != 4:
            raise ValueError("Source operation must have exactly four fields")
        name, qargs, cargs, params = record
        if not isinstance(name, str) or name not in _SOURCE_OPERATIONS:
            raise ValueError("Opaque or unsupported source operation")
        if not all(isinstance(item, (list, tuple)) for item in (qargs, cargs, params)):
            raise ValueError("Source operands and parameters must be finite sequences")
        if len(qargs) > qwidth or len(cargs) > n or len(params) > 1:
            raise ValueError("Source operation arity exceeds the fixed case")
        qargs = tuple(_integer(q, "source wire", 0, qwidth-1) for q in qargs)
        cargs = tuple(_integer(c, "source readout bit", 0, n-1) for c in cargs)
        params = tuple(params)
        if len(set(qargs)) != len(qargs) or len(set(cargs)) != len(cargs):
            raise ValueError("Duplicate source operands are not allowed")
        if name == "barrier":
            valid = qargs == tuple(range(qwidth)) and not cargs and not params
        elif name == "measure":
            valid = len(qargs) == len(cargs) == 1 and not params
            measured = True
            readouts.append({"wire": qargs[0] if qargs else None,
                             "classical_bit": cargs[0] if cargs else None})
        elif name == "reset":
            valid = len(qargs) == 1 and not cargs and not params and index < qwidth and qargs == (index,)
        elif name == "cx":
            valid = len(qargs) == 2 and not cargs and not params and not measured
        elif name == "ry":
            valid = (len(qargs) == 1 and not cargs and len(params) == 1
                     and type(params[0]) is float and math.isfinite(params[0]) and not measured)
            if valid:
                angle = params[0].hex()
                synthesis_key(angle)
                rotations.append({"operation_index": index, "wire": qargs[0], "angle_hex": angle})
        else:
            valid = len(qargs) == 1 and not cargs and not params and not measured
        if not valid:
            raise ValueError("Source operation schema or event order is invalid")
        counts[name] += 1
        immutable.append((name, qargs, cargs, params))
    if tuple(immutable[:qwidth]) != tuple(("reset", (q,), (), ()) for q in range(qwidth)):
        raise ValueError("Every source input wire must reset before preparation")
    if dict(counts) != case["source_counts_per_draw"]:
        raise ValueError("Accepted source primitive counts changed")
    if readouts != case["assignment_readout_mapping"] or rotations != case["rotation_occurrences"]:
        raise ValueError("Accepted source rotation/readout mapping changed")
    encoded = b"".join(json.dumps(record, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                       for record in immutable)
    name = case["source_operation_files"][1]
    if hashlib.sha256(encoded).hexdigest() != FROZEN_INPUTS[name][1]:
        raise ValueError("Accepted source instruction order/bytes changed")
    return tuple(immutable)


def accepted_cases(root, design=None):
    """Read/check fixed source evidence; never construct or simulate a circuit.

    Runtime defaults to embedded ordinary parameters.  An optional exact
    source-only design object can be checked during preparation, but no
    docs/coordination path is read here.
    """
    if design is not None:
        check_design(design)
    task = public_task()
    raw = verify_frozen_inputs(root)
    output = []
    for case in task["cases"]:
        n = case["n"]
        qasm_path, stream_path, ledger_path = case["source_operation_files"]
        lines = raw[stream_path].splitlines()
        if len(lines) != case["source_operations_per_draw"]:
            raise ValueError("Accepted source NDJSON line count changed")
        operations = validate_operations(n, [parse_json(line) for line in lines])
        ledger = parse_json(raw[ledger_path])
        actual = ledger.get("actual_ledger") if isinstance(ledger, dict) else None
        if not isinstance(actual, dict) or actual.get("primitive_counts") != case["source_counts_per_draw"]:
            raise ValueError("Accepted source ledger counts changed")
        if (actual.get("source_operations") != len(operations)
                or actual.get("unitary_primitives") != case["source_unitary_per_draw"]
                or actual.get("explicit_t_tdg") != case["source_explicit_t_tdg_per_draw"]):
            raise ValueError("Accepted source ledger task accounting changed")
        item = dict(case)
        item.update(operations=operations, qasm=raw[qasm_path], ledger=ledger,
                    validated_inputs={name: {"bytes": size, "sha256": digest}
                                      for name, (size, digest) in FROZEN_INPUTS.items()})
        output.append(item)
    return output


def guard_whole_error(n, sum_errors, padding=REFERENCE_PADDING):
    """Enforce sum over the complete case plus padding against exact eta.

    Float inputs are compared by their exact IEEE rational values; an error
    exactly at eta is rejected once mandatory reference padding is charged.
    This arithmetic guard is not a formal bound on a measured matrix norm.
    """
    n = _case_n(n)
    errors = _finite_fraction(sum_errors, "whole-case error sum")
    padding_value = _finite_fraction(padding, "reference padding")
    if type(padding) is float and padding.hex() == float(REFERENCE_PADDING).hex():
        padding_value = REFERENCE_PADDING
    if padding_value < REFERENCE_PADDING:
        raise ValueError("Reference padding cannot be omitted or reduced")
    eta = fraction_text(_case_parameters(n)["realization_ceiling_per_draw"])
    total = errors + padding_value
    if total > eta:
        raise ValueError("Whole-case error sum plus padding exceeds its eta allowance")
    represented_upper = float(total)
    if Fraction.from_float(represented_upper) < total:
        represented_upper = math.nextafter(represented_upper, math.inf)
    if Fraction.from_float(represented_upper) > eta:
        raise ValueError("Represented whole-case upper bound exceeds exact eta")
    return represented_upper


def guard_occurrence_errors(n, errors, padding=REFERENCE_PADDING):
    """Count every reused RY occurrence before applying the whole-case guard."""
    n = _case_n(n)
    expected_count = len(_case_parameters(n)["rotation_occurrences"])
    if not isinstance(errors, (list, tuple)) or len(errors) != expected_count:
        raise ValueError("Every source RY occurrence must contribute one guarded error")
    values = [_finite_fraction(value, "guarded word error") for value in errors]
    if any(value > WORD_ERROR_CEILING for value in values):
        raise ValueError("A guarded word error exceeds its per-word allowance")
    return guard_whole_error(n, sum(values, Fraction()), padding)


def whole_task_success_bound(n, guarded_draw_error, physical_failure, availability_failure):
    """Separate ideal/realization/physical/readiness allowances for one task."""
    n = _case_n(n)
    case = _case_parameters(n)
    eta = _finite_fraction(guarded_draw_error, "complete-draw guarded error")
    physical = _finite_fraction(physical_failure, "whole-task physical failure")
    availability = _finite_fraction(availability_failure, "whole-task factory availability failure")
    ceiling = fraction_text(case["realization_ceiling_per_draw"])
    if eta > ceiling or physical > ALLOWANCE or availability > ALLOWANCE:
        raise ValueError("A separate whole-task budget category is exceeded")
    total = fraction_text(case["ideal_miss"]) + case["fixed_complete_draws"]*eta + physical + availability
    if total > _TOTAL_MISS_TARGET:
        raise ValueError("Conditional whole-task miss bound exceeds one percent")
    miss_upper, success_lower = float(total), float(1-total)
    if Fraction.from_float(miss_upper) < total:
        miss_upper = math.nextafter(miss_upper, math.inf)
    if Fraction.from_float(success_lower) > 1-total:
        success_lower = math.nextafter(success_lower, -math.inf)
    exact_text = lambda value: f"{value.numerator}/{value.denominator}"
    return {"miss_upper_bound": miss_upper, "success_lower_bound": success_lower,
            "ideal_miss_fraction": case["ideal_miss"],
            "realization_failure_fraction": exact_text(case["fixed_complete_draws"]*eta),
            "physical_failure_fraction": exact_text(physical),
            "availability_failure_fraction": exact_text(availability),
            "miss_upper_bound_fraction": exact_text(total),
            "success_lower_bound_fraction": exact_text(1-total),
            "conditional_on_faithful_bit_transfer_and_correct_scoring": True,
            "formal_interval_certificate": False}


def factory_retry_cap(n, t_states_per_draw):
    """Recompute whole-task readiness; never inherit a prior task's retry cap."""
    n = _case_n(n)
    count = _integer(t_states_per_draw, "per-draw T demand", 0, LIMITS["compiled_operations_per_case"])
    states = count * _case_parameters(n)["fixed_complete_draws"]
    if states == 0:
        return {"retry_cap": 0, "states": 0, "worst_case_pipeline_attempts": 0,
                "availability_failure_fraction": "0/1"}
    rejection = Fraction(1, 100)
    for retries in range(1, LIMITS["factory_retry_cap"] + 1):
        failure = states * rejection**retries
        if failure <= ALLOWANCE:
            return {"retry_cap": retries, "states": states,
                    "worst_case_pipeline_attempts": states*retries,
                    "availability_failure_fraction": f"{failure.numerator}/{failure.denominator}"}
    raise ValueError("Factory retry cap cannot satisfy the frozen whole-task allowance")


class CounterGuard:
    """Precharge only the eight fixed synthesis keys and two fixed native tasks."""
    __slots__ = ("_limits", "_seen", "_calls", "_deadline", "_clock")

    def __init__(self, *, synthesis_limit=8, trace_limit=2, deadline=None, clock=time.monotonic):
        self._limits = {"synthesis": _integer(synthesis_limit, "synthesis limit", 0, 8),
                        "trace": _integer(trace_limit, "trace limit", 0, 2)}
        if deadline is not None and (type(deadline) not in (int, float)
                or type(deadline) is int and deadline.bit_length() > 63
                or not math.isfinite(deadline) or not 0 <= deadline <= 2**63):
            raise ValueError("Deadline must be a finite nonnegative monotonic time")
        if not callable(clock):
            raise ValueError("A monotonic clock callable is required")
        self._deadline, self._clock = deadline, clock
        self._seen = {"synthesis": set(), "trace": set()}
        self._calls = {"synthesis": 0, "trace": 0}

    def charge(self, kind, key):
        if type(kind) is not str or len(kind) > 16 or not kind.isascii() or kind not in self._limits:
            raise ValueError("No new native query class is allowed")
        if kind == "synthesis":
            if (type(key) is not tuple or len(key) != 3
                    or any(type(part) is not str or len(part) > 32 or not part.isascii() for part in key)
                    or key not in ALLOWED_SYNTHESIS_KEYS):
                raise ValueError("Native synthesis key is not the frozen angle/epsilon target")
        elif type(key) is not int or key not in (2, 3):
            raise ValueError("Direct Trace task must be exactly n=2 or n=3")
        if key in self._seen[kind]:
            raise ValueError("Native key is already spent; retry is forbidden")
        now = self._clock()
        if (type(now) not in (int, float) or type(now) is int and now.bit_length() > 63
                or not math.isfinite(now) or not 0 <= now <= 2**63):
            raise ValueError("Monotonic clock returned a nonfinite value")
        if self._deadline is not None and now >= self._deadline:
            raise TimeoutError("Native work deadline reached before entry")
        if self._calls[kind] >= self._limits[kind]:
            raise ValueError("Native call cap reached before entry")
        self._seen[kind].add(key)
        self._calls[kind] += 1
        return self._calls[kind]

    def snapshot(self):
        return {"synthesis_native_calls": self._calls["synthesis"],
                "direct_native_trace_estimate_calls": self._calls["trace"],
                "spent_synthesis_keys": [list(key) for key in sorted(self._seen["synthesis"])],
                "spent_trace_cases": sorted(self._seen["trace"]),
                "qdk_top_level_estimate_calls": 0,
                "factory_frontier_queries": 0, "factory_frontier_candidates": 0,
                "factory_pipeline_candidates": 0,
                "query_scope": "top-level native entries only; internal native iterations not counted"}

    def require_complete(self):
        if self._seen["synthesis"] != ALLOWED_SYNTHESIS_KEYS or self._seen["trace"] != {2, 3}:
            raise ValueError("Partial native work cannot be reported complete")
        return self.snapshot()
