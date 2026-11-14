"""Validate six frozen core-energy physical records without importing QDK.

The caller supplies immutable-file/source authority and replayed logical
ledgers.  This module checks record completeness, circuit bindings and the
pinned model arithmetic.  It does not rerun the native factory/Pareto search,
establish native execution provenance, or certify synthesized hardware.

Model equations are from Microsoft QDK commit
81de0d2e5347e1e9f10e05066320fffdb3b7058c, specifically
source/qre/src/trace/transforms/psspc.rs, source/qre/src/trace.rs, and
source/qdk_package/qdk/qre/models/qec/_surface_code.py.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re


AUDITED_SOURCE_COMMIT = "81de0d2e5347e1e9f10e05066320fffdb3b7058c"
ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
CASE_PARAMETERS = {
    "default": (1, 5, "repeated_U"),
    "n2": (1, 3, "repeated_U"),
    "n4": (2, 4, "repeated_U"),
    "n6": (3, 5, "repeated_U"),
    "n8": (4, 5, "repeated_U"),
    "analytic-powers": (1, 5, "analytic_dimer_powers"),
}
PROFILE_KEYS = {"name", "physical_error_rate", "gate_time_ns", "measurement_time_ns"}
LEDGER_KEYS = {
    "contract", "access_mode", "controlled_power_circuits_all_groups", "logical_qubits_per_group",
    "source_instruction_counts_per_group", "primitive_counts_per_group", "primitive_operations_per_group",
    "dependency_depth_per_group", "primitive_operations_all_groups", "qubit_measurements_all_groups",
    "hartree_fock_preparations_all_groups", "qasm_sha256", "compiled_global_phase_radians",
    "qasm_phase_policy", "construction_and_lowering_wall_seconds", "full_static_group_simulated",
    "sequential_equivalent_simulation", "included", "excluded_physical_costs", "source_commit",
    "source_dirty", "source_commit_scope", "source_sha256", "environment",
}
CONTRACT_KEYS = {
    "kind", "dimers", "data_qubits", "phase_bits", "phase_width", "encoding",
    "energy_absolute_error_ideal", "ground_overlap_lower_bound", "overlap_promise_source",
    "independent_groups", "qpe_rounds_per_retained_target", "one_round_phase_success_lower_bound",
    "median_failure_bound_per_group", "no_ground_group_failure_bound", "ideal_failure_union_bound",
    "ideal_confidence_lower_bound", "requested_ideal_failure", "total_qpe_rounds",
    "total_base_U_equivalent_calls", "assumptions", "hardware_or_synthesis_confidence_certified",
}
REPORT_KEYS = {
    "schema_version", "status", "resource_estimate_kind", "qdk_version",
    "audited_source_commit", "source_commit_is_package_release_commit",
    "qiskit_version", "transpiler", "qasm3", "qasm_sha256", "source_gate_counts",
    "decomposed_gate_counts", "canonical_gate_counts", "canonical_application_trace",
    "synthesized_application_trace", "synthesis", "budget", "observable_contract",
    "physical_profile", "error_composition", "isa_graph_pruning", "pareto_completeness",
    "included", "limitations", "sources", "pareto_points", "ideal_energy_contract",
    "initialization_included", "hardware_energy_confidence_certified",
}
SYNTHESIS_KEYS = {
    "model", "cost_semantics", "fit_domain", "num_arbitrary_rotations",
    "fixed_num_ts_per_rotation", "model_error_per_rotation", "model_error_per_execution",
    "actual_synthesis_sequence_generated", "instance_approximation_certificate_verified",
}
POINT_KEYS = {
    "peak_physical_qubits", "data_code_distance", "runtime_per_execution_ns",
    "serial_runtime_all_executions_ns", "qdk_combined_error_score_per_execution",
    "synthesis_model_error_per_execution", "physical_failure_model_per_execution",
    "physical_failure_union_bound_all_executions", "factories", "model_properties",
    "instruction_source_graph",
}
PROPERTY_KEYS = {
    "LOGICAL_MEMORY_QUBITS", "EVALUATION_TIME", "NUM_TS_PER_ROTATION",
    "ALGORITHM_COMPUTE_QUBITS", "PHYSICAL_COMPUTE_QUBITS", "ALGORITHM_MEMORY_QUBITS",
    "LOGICAL_COMPUTE_QUBITS", "PHYSICAL_FACTORY_QUBITS",
}
INCLUDED = [
    "all operations in supplied bound circuit",
    "explicit input preparation and output basis changes if supplied",
    "final measurements", "serial executions", "PSSPC logical layout",
    "surface-code data qubits", "round-based magic-state factories",
]
LIMITATIONS = [
    "The synthesis formula is a mean linear fit for random angles, not a worst-case bound or certified concrete Clifford+T implementation",
    "The fit models expected fallback-protocol work; no synthesis runtime-tail or budget-overrun confidence is certified",
    "The QDK trace abstracts Clifford work and scheduling into PSSPC rather than preserving an exact gate schedule",
    "Physical reliability is conditional on the declared hardware/QEC/factory models",
    "No algorithm approximation, optimizer convergence, or observable precision is inferred from the circuit",
    "Classical optimizer/preprocessing/compilation/verification costs and external data loading are excluded",
    "A bounded-observable contract covers one fixed scalar mean only when its declared assumptions hold",
]
SOURCES = [
    f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace/transforms/psspc.rs",
    f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace.rs",
    f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/result.rs",
    "https://learn.microsoft.com/en-us/azure/quantum/qre-estimation-results",
    "https://arxiv.org/abs/2203.10064",
]
STATUS = "computed_conditional_model"
KIND = "conditional_mean_fit_resource_projection"
MAX_RECORD_BYTES = 128 * 1024 * 1024


def _require(condition, label, message):
    if not condition:
        raise ValueError(f"{label}: {message}")


def _keys(value, expected, label):
    _require(isinstance(value, dict), label, "expected an object")
    _require(set(value) == set(expected), label, "missing or unexpected fields")


def _same(value, expected, label):
    """Compare JSON values without accepting True as 1 or 1.0 as 1."""
    _require(type(value) is type(expected), label, "value type differs")
    if isinstance(expected, dict):
        _keys(value, expected, label)
        for key, item in expected.items():
            _same(value[key], item, f"{label}/{key}")
    elif isinstance(expected, list):
        _require(len(value) == len(expected), label, "list length differs")
        for index, item in enumerate(expected):
            _same(value[index], item, f"{label}/{index}")
    else:
        _require(value == expected, label, "value differs")


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label, f"expected integer >= {minimum}")
    return value


def _number(value, label, minimum=None):
    _require(type(value) in (int, float), label, "expected a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, label, "number must be finite")
    if minimum is not None:
        _require(value >= minimum, label, f"number must be >= {minimum}")
    return value


def _close(value, expected, label):
    _number(value, label)
    _number(expected, label)
    _require(math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15), label,
             "pinned model arithmetic differs")


def _finite_tree(value, label):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{label}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}/{index}")
    elif type(value) is float:
        _number(value, label)


def _read_json(path, root):
    _require(path.is_file() and not path.is_symlink(), str(path), "missing regular record")
    _require(path.resolve().is_relative_to(root), str(path), "record escapes its directory")
    _require(path.stat().st_size <= MAX_RECORD_BYTES, str(path), "record size limit exceeded")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, str(path), f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"{path}: nonfinite JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                           parse_constant=invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: unreadable JSON record") from error
    _finite_tree(value, str(path))
    return value


def _counts(value, allowed, label):
    _require(isinstance(value, dict) and value, label, "expected nonempty counts")
    _require(set(value) <= set(allowed), label, "unsupported operation count")
    for key, count in value.items():
        _integer(count, f"{label}/{key}", 1)
    return value


def _angle(expression, label):
    _require(len(expression) <= 512, label, "angle expression too long")
    try:
        value = float(expression)
    except ValueError:
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, RecursionError) as error:
            raise ValueError(f"{label}: unsupported angle syntax") from error
        _require(sum(1 for _ in ast.walk(tree)) <= 64, label, "angle expression too complex")

        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                result = node.value
            elif isinstance(node, ast.Name) and node.id == "pi":
                result = math.pi
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = evaluate(node.operand)
                result = operand if isinstance(node.op, ast.UAdd) else -operand
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                else:
                    _require(right != 0, label, "zero angle denominator")
                    result = left / right
            else:
                raise ValueError(f"{label}: unsupported angle expression")
            return _number(result, label)

        value = evaluate(tree.body)
    _number(value, label)


def _qasm_counts(path, root, width, measurements):
    """Read the frozen flat QASM stream once; do not materialize a circuit."""
    _require(path.is_file() and not path.is_symlink(), str(path), "missing QASM record")
    _require(path.resolve().is_relative_to(root), str(path), "QASM escapes its directory")
    _require(path.stat().st_size <= MAX_RECORD_BYTES, str(path), "QASM size limit exceeded")
    expected_headers = ["OPENQASM 3.0;", 'include "stdgates.inc";',
                        f"bit[{measurements}] c;", f"qubit[{width}] q;"]
    gate_pattern = re.compile(r"([a-z]+)(?:\((.*)\))? (q\[\d+\](?:, q\[\d+\])?);")
    measure_pattern = re.compile(r"c\[(\d+)\] = measure q\[(\d+)\];")
    digest = hashlib.sha256()
    counts = Counter()
    seen_measurements = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                _require(raw.endswith(b"\n"), str(path), "QASM lines must end with LF")
                line = raw[:-1].decode("ascii")
                label = f"{path}:{line_number}"
                if line_number <= 4:
                    _same(line, expected_headers[line_number - 1], label)
                    continue
                measurement = measure_pattern.fullmatch(line)
                if measurement:
                    classical, quantum = map(int, measurement.groups())
                    _require(classical == quantum == seen_measurements and classical < measurements,
                             label, "unexpected final clock readout")
                    seen_measurements += 1
                    counts["measure"] += 1
                    continue
                _require(seen_measurements == 0, label, "operation follows a measurement")
                gate = gate_pattern.fullmatch(line)
                _require(gate is not None, label, "unsupported or opaque QASM statement")
                name, angle, operands = gate.groups()
                _require(name in ROTATION_BASIS, label, "unsupported primitive gate")
                wires = [int(value) for value in re.findall(r"q\[(\d+)\]", operands)]
                _require(len(wires) == (2 if name == "cx" else 1)
                         and len(set(wires)) == len(wires)
                         and all(0 <= wire < width for wire in wires), label, "invalid gate operands")
                _require((angle is not None) == (name in {"rx", "ry", "rz"}), label,
                         "unexpected gate parameters")
                if angle is not None:
                    _angle(angle, label)
                counts[name] += 1
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{path}: unreadable canonical QASM") from error
    _require(seen_measurements == measurements, str(path), "incomplete final measurements")
    _require(sum(counts.values()) <= 750000, str(path), "primitive export limit exceeded")
    return dict(counts), digest.hexdigest()


def _contract_and_ledger(ledger, case):
    label = f"{case}/logical-ledger"
    _keys(ledger, LEDGER_KEYS | ({"physical_resource_estimate"} if case == "default" else set()), label)
    _finite_tree(ledger, label)
    contract = ledger.get("contract")
    _keys(contract, CONTRACT_KEYS, f"{label}/contract")
    dimers, phase_bits, mode = CASE_PARAMETERS[case]
    for key, expected in {"kind": "minimum_of_retained-target_medians", "dimers": dimers,
                          "data_qubits": 2 * dimers, "phase_bits": phase_bits,
                          "phase_width": 2**phase_bits, "independent_groups": 2,
                          "qpe_rounds_per_retained_target": 19, "requested_ideal_failure": .005,
                          "overlap_promise_source": "analytic two-parity dimer family; not generic chemistry state preparation",
                          "assumptions": ["exact analytic-angle Hamiltonian access",
                                          "independent HF preparations across groups",
                                          "same target retained for all median rounds within each group",
                                          "spectrum maps into [1/4,3/4]", "correct classical median/min decoding",
                                          "no physical faults or synthesis error"],
                          "hardware_or_synthesis_confidence_certified": False}.items():
        _same(contract.get(key), expected, f"{label}/contract/{key}")
    groups, rounds = 2, 19
    measurements = rounds * phase_bits
    width = 2 * dimers + measurements
    base_calls = groups * rounds * (2**phase_bits - 1)
    power_calls = groups * rounds * (phase_bits if mode == "analytic_dimer_powers" else 2**phase_bits - 1)
    _same(contract.get("total_qpe_rounds"), groups * rounds, f"{label}/contract/total_qpe_rounds")
    _same(contract.get("total_base_U_equivalent_calls"), base_calls,
          f"{label}/contract/total_base_U_equivalent_calls")
    _same(ledger.get("access_mode"), mode, f"{label}/access_mode")
    _same(ledger.get("logical_qubits_per_group"), width, f"{label}/logical_qubits_per_group")
    _same(ledger.get("controlled_power_circuits_all_groups"), power_calls,
          f"{label}/controlled_power_circuits_all_groups")
    _same(ledger.get("qubit_measurements_all_groups"), groups * measurements,
          f"{label}/qubit_measurements_all_groups")
    _same(ledger.get("hartree_fock_preparations_all_groups"), groups,
          f"{label}/hartree_fock_preparations_all_groups")
    _same(ledger.get("full_static_group_simulated"), False, f"{label}/full_static_group_simulated")
    _number(ledger.get("compiled_global_phase_radians"), f"{label}/compiled_global_phase_radians")
    _integer(ledger["dependency_depth_per_group"], f"{label}/dependency_depth_per_group", 1)
    _number(ledger["construction_and_lowering_wall_seconds"], f"{label}/construction_and_lowering_wall_seconds", 0)
    _same(ledger["qasm_phase_policy"],
          "pinned exporter can omit overall compiled global phase; complete measured groups preserve probabilities; not a recontrolled oracle specification",
          f"{label}/qasm_phase_policy")
    _same(ledger["included"], ["one HF preparation per group", "all repeated controlled exact dimer accesses",
                               "identity/shift coherent phases", "every clock Hadamard and inverse QFT",
                               "all final clock measurements", "all serial median groups", "classical median/min rule"],
          f"{label}/included")
    _same(ledger["excluded_physical_costs"], ["fresh zero-state initialization/reset faults and timing",
                                             "classical coefficient/SCF/mapping construction",
                                             "compilation and decoder/routing implementation",
                                             "classical readout/median communication latency"],
          f"{label}/excluded_physical_costs")
    encoding = contract.get("encoding")
    _keys(encoding, {"shift", "span"}, f"{label}/contract/encoding")
    _number(encoding["shift"], f"{label}/contract/encoding/shift")
    span = _number(encoding["span"], f"{label}/contract/encoding/span", 0)
    _require(span > 0, label, "spectral span must be positive")
    _close(contract.get("energy_absolute_error_ideal"), span / 2**phase_bits,
           f"{label}/contract/energy_absolute_error_ideal")
    if case.startswith("n"):
        _require(contract["energy_absolute_error_ideal"] <= .5, label, "scaling accuracy target exceeded")
    overlap = _number(contract.get("ground_overlap_lower_bound"), f"{label}/contract/ground_overlap", 0)
    _require(overlap <= 1, label, "ground overlap exceeds one")
    p0 = 8 / math.pi**2
    median_failure = sum(math.comb(rounds, j) * (1 - p0)**j * p0**(rounds - j)
                         for j in range((rounds + 1) // 2, rounds + 1))
    no_ground = (1 - overlap)**groups
    failure = no_ground + groups * median_failure
    for key, expected in {"one_round_phase_success_lower_bound": p0,
                          "median_failure_bound_per_group": median_failure,
                          "no_ground_group_failure_bound": no_ground,
                          "ideal_failure_union_bound": failure,
                          "ideal_confidence_lower_bound": 1 - failure}.items():
        _close(contract.get(key), expected, f"{label}/contract/{key}")
    _require(failure <= .005, label, "ideal failure allocation exceeded")
    source = {"cEnergy U": power_calls // groups, "h": 2 * measurements,
              "cp": rounds * phase_bits * (phase_bits - 1) // 2,
              "measure": measurements, "swap": rounds * (phase_bits // 2), "x": dimers}
    _same(ledger.get("source_instruction_counts_per_group"), source,
          f"{label}/source_instruction_counts_per_group")
    primitive = _counts(ledger.get("primitive_counts_per_group"), ROTATION_BASIS + ["measure"],
                        f"{label}/primitive_counts_per_group")
    _same(primitive.get("measure"), measurements, f"{label}/primitive measurements")
    _same(ledger.get("primitive_operations_per_group"), sum(primitive.values()),
          f"{label}/primitive_operations_per_group")
    _same(ledger.get("primitive_operations_all_groups"), groups * sum(primitive.values()),
          f"{label}/primitive_operations_all_groups")
    return contract, width, measurements, primitive


def _raw_trace(text, width, counts, label):
    # These frozen records use QDK's logical-count rotation bucketization.
    pattern = (
        r"@compute_qubits\((\d+)\)\n\{\n"
        r"  repeat (\d+) \{\n    RZ\(0\)\n    RZ\(1\)\n  \}\n"
        r"  repeat (\d+) \{\n    RZ\(0\)\n  \}\n"
        r"  repeat (\d+) \{\n    T\(0\)\n  \}\n"
        r"  repeat (\d+) \{\n    MEAS_Z\(0\)\n  \}\n\}\n"
    )
    _require(isinstance(text, str), label, "missing canonical trace")
    match = re.fullmatch(pattern, text)
    _require(match is not None, label, "unsupported canonical trace grammar")
    qubits, paired, single, t_count, measurements = map(int, match.groups())
    _same(qubits, width, f"{label}/compute_qubits")
    _same({"RZ": 2 * paired + single, "T": t_count, "MEAS_Z": measurements}, counts,
          f"{label}/gate_counts")
    _require(paired > 0 and single > 0, label, "incomplete rotation buckets")
    return paired + single


def _synthesized_trace(text, logical_width, depth, states, synthesis_error, label):
    pattern = (r"@compute_qubits\((\d+)\)\n@base_error\(([-+0-9.eE]+)\)\n"
               r"@resource_state\(T, (\d+)\)\n\{\n  repeat (\d+) \{\n"
               r"    MULTI_PAULI_MEAS\(([0-9, ]+)\)\n  \}\n\}\n")
    _require(isinstance(text, str), label, "missing synthesized trace")
    match = re.fullmatch(pattern, text)
    _require(match is not None, label, "unsupported synthesized trace grammar")
    width, base, t_states, repeat, wires = match.groups()
    _same(int(width), logical_width, f"{label}/compute_qubits")
    _close(float(base), synthesis_error, f"{label}/base_error")
    _same(int(t_states), states, f"{label}/resource_state")
    _same(int(repeat), depth, f"{label}/logical_depth")
    _same(wires, ", ".join(map(str, range(logical_width))), f"{label}/operands")


def _source_graph(profile, distance):
    gates = (f"GateBased(error_rate={profile['physical_error_rate']}, gate_time={profile['gate_time_ns']}, "
             f"measurement_time={profile['measurement_time_ns']}, two_qubit_gate_time={profile['gate_time_ns']})")
    return (
        "LATTICE_SURGERY @ SurfaceCode(crossing_prefactor=0.03, error_correction_threshold=0.01, "
        "one_qubit_gate_depth=1, two_qubit_gate_depth=4, code_cycle_override=None, "
        f"code_cycle_offset=0, distance={distance})\n"
        f"  CNOT @ {gates}\n  H @ {gates}\n  MEAS_Z @ {gates}\n"
        "T @ RoundBasedFactory(code_query=_ComponentQuery(component=<class "
        "'qdk.qre.models.qec._surface_code.SurfaceCode'>, source=RootNode(), kwargs={}), "
        "physical_qubit_calculation=<built-in function sum>)\n"
        f"  T @ {gates}"
    )


def _validate_report(report, ledger, case, directory, root, profile):
    label = f"{case}/surface-code"
    _keys(report, REPORT_KEYS | ({"projection_wall_seconds"} if case == "default" else set()), label)
    for key, expected in {"schema_version": 1, "status": STATUS, "resource_estimate_kind": KIND,
                          "qdk_version": "1.34.0", "qiskit_version": "2.6.2",
                          "audited_source_commit": AUDITED_SOURCE_COMMIT,
                          "source_commit_is_package_release_commit": False,
                          "observable_contract": None, "isa_graph_pruning": False,
                          "initialization_included": False, "hardware_energy_confidence_certified": False,
                          "error_composition": "UnionBound; approximation base and physical contribution explicitly separated",
                          "pareto_completeness": "enumerated ISA combinations; limited to declared models/domains",
                          "included": INCLUDED, "limitations": LIMITATIONS, "sources": SOURCES}.items():
        _same(report[key], expected, f"{label}/{key}")
    _same(report["physical_profile"], profile, f"{label}/physical_profile")
    _same(report["budget"], {"synthesis_model_error_per_execution": 1e-5,
                           "physical_failure_probability": .005, "executions": 2}, f"{label}/budget")
    _same(report["transpiler"], {"optimization_level": 1, "seed_transpiler": 7,
                               "basis_gates": ROTATION_BASIS}, f"{label}/transpiler")
    if case == "default":
        _number(report["projection_wall_seconds"], f"{label}/projection_wall_seconds", 0)
    contract, width, measurements, primitive = _contract_and_ledger(ledger, case)
    _same(report["ideal_energy_contract"], contract, f"{label}/ideal_energy_contract")
    _same(report["source_gate_counts"], ledger["source_instruction_counts_per_group"],
          f"{label}/source_gate_counts")
    _same(report["decomposed_gate_counts"], primitive, f"{label}/decomposed_gate_counts")
    qasm_counts, qasm_digest = _qasm_counts(directory / "measured-group.qasm", root, width, measurements)
    _same(qasm_counts, primitive, f"{label}/QASM primitive counts")
    _same(report["qasm_sha256"], qasm_digest, f"{label}/qasm_sha256")
    _same(ledger.get("qasm_sha256"), qasm_digest, f"{case}/logical-ledger/qasm_sha256")
    _require(isinstance(report["qasm3"], str), label, "missing embedded QASM")
    _same(hashlib.sha256(report["qasm3"].encode("utf-8")).hexdigest(), qasm_digest,
          f"{label}/embedded QASM digest")
    counts = _counts(report["canonical_gate_counts"], {"T", "RZ", "MEAS_Z"}, f"{label}/canonical_gate_counts")
    _keys(counts, {"T", "RZ", "MEAS_Z"}, f"{label}/canonical_gate_counts")
    _same(counts["MEAS_Z"], measurements, f"{label}/canonical measurements")
    rotation_depth = _raw_trace(report["canonical_application_trace"], width, counts,
                                f"{label}/canonical_application_trace")
    synthesis = report["synthesis"]
    _keys(synthesis, SYNTHESIS_KEYS, f"{label}/synthesis")
    for key, expected in {"model": "QDK PSSPC mixed-fallback Clifford+T approximation",
                          "cost_semantics": "mean linear fit for random angles; not worst-case or certified per angle",
                          "fit_domain": "per-rotation modeled diamond-distance accuracy epsilon < 1e-4",
                          "actual_synthesis_sequence_generated": False,
                          "instance_approximation_certificate_verified": False}.items():
        _same(synthesis[key], expected, f"{label}/synthesis/{key}")
    rotations = counts["RZ"]
    _same(synthesis["num_arbitrary_rotations"], rotations, f"{label}/synthesis/num_arbitrary_rotations")
    num_ts = max(12, math.ceil(4.86 + .53 * (math.log2(rotations) - math.log2(1e-5))))
    while rotations * 2**((4.86 - num_ts) / .53) > 1e-5:
        num_ts += 1
    _same(synthesis["fixed_num_ts_per_rotation"], num_ts, f"{label}/synthesis/fixed_num_ts_per_rotation")
    per_rotation = 2**((4.86 - num_ts) / .53)
    synthesis_error = rotations * per_rotation
    _require(0 < per_rotation < 1e-4 and synthesis_error <= 1e-5, label, "synthesis allowance/domain exceeded")
    _close(synthesis["model_error_per_rotation"], per_rotation, f"{label}/synthesis/model_error_per_rotation")
    _close(synthesis["model_error_per_execution"], synthesis_error, f"{label}/synthesis/model_error_per_execution")
    _require(0 < synthesis["model_error_per_rotation"] < 1e-4
             and 0 <= synthesis["model_error_per_execution"] <= 1e-5,
             label, "reported synthesis allowance/domain exceeded")
    logical_width = 2 * width + math.ceil(math.sqrt(8 * width)) + 1
    logical_depth = measurements + counts["T"] + rotations + num_ts * rotation_depth
    states = counts["T"] + num_ts * rotations
    _synthesized_trace(report["synthesized_application_trace"], logical_width, logical_depth,
                       states, synthesis_error, f"{label}/synthesized_application_trace")
    _require(isinstance(report["pareto_points"], list) and len(report["pareto_points"]) == 1,
             label, "this frozen bundle requires one complete Pareto point")
    point = report["pareto_points"][0]
    _keys(point, POINT_KEYS, f"{label}/pareto_points/0")
    properties = point["model_properties"]
    _keys(properties, PROPERTY_KEYS, f"{label}/model_properties")
    for key, value in properties.items():
        _integer(value, f"{label}/model_properties/{key}")
    for key, expected in {"LOGICAL_MEMORY_QUBITS": 0, "ALGORITHM_MEMORY_QUBITS": 0,
                          "NUM_TS_PER_ROTATION": num_ts, "ALGORITHM_COMPUTE_QUBITS": width,
                          "LOGICAL_COMPUTE_QUBITS": logical_width}.items():
        _same(properties[key], expected, f"{label}/model_properties/{key}")
    distance = _integer(point["data_code_distance"], f"{label}/data_code_distance", 3)
    _require(distance <= 25 and distance % 2 == 1, label, "distance is outside pinned search domain")
    _same(point["instruction_source_graph"], _source_graph(profile, distance), f"{label}/instruction_source_graph")
    data_qubits = logical_width * (2 * distance**2 - 1)
    _same(properties["PHYSICAL_COMPUTE_QUBITS"], data_qubits, f"{label}/PHYSICAL_COMPUTE_QUBITS")
    cycle_ns = profile["gate_time_ns"] + 4 * profile["gate_time_ns"] + profile["measurement_time_ns"]
    runtime = logical_depth * distance * cycle_ns
    _number(point["runtime_per_execution_ns"], f"{label}/runtime_per_execution_ns", 0)
    _number(point["serial_runtime_all_executions_ns"], f"{label}/serial_runtime_all_executions_ns", 0)
    _require(point["runtime_per_execution_ns"] == runtime, label, "modeled per-execution runtime differs")
    _require(point["serial_runtime_all_executions_ns"] == 2 * runtime, label, "serial runtime scaling differs")
    _keys(point["factories"], {"T"}, f"{label}/factories")
    factory = point["factories"]["T"]
    _keys(factory, {"copies", "runs", "states", "error_rate"}, f"{label}/factories/T")
    for key in ("copies", "runs", "states"):
        _integer(factory[key], f"{label}/factories/T/{key}", 1)
    _same(factory["states"], states, f"{label}/factories/T/states")
    _same(factory["copies"], (states + factory["runs"] - 1) // factory["runs"], f"{label}/factories/T/copies")
    factory_error_rate = _number(factory["error_rate"], f"{label}/factories/T/error_rate", 0)
    _require(factory_error_rate < 1, label, "factory error rate exceeds probability domain")
    factory_qubits = properties["PHYSICAL_FACTORY_QUBITS"]
    _require(factory_qubits > 0 and factory_qubits % factory["copies"] == 0,
             label, "factory-space aggregate is inconsistent with identical copies")
    _same(_integer(point["peak_physical_qubits"], f"{label}/peak_physical_qubits", 1),
          data_qubits + factory_qubits, f"{label}/peak_physical_qubits")
    data_error = (logical_depth * logical_width * .03
                  * (profile["physical_error_rate"] / .01)**((distance + 1) // 2))
    physical_error = data_error + states * factory_error_rate
    _close(point["synthesis_model_error_per_execution"], synthesis_error, f"{label}/synthesis_model_error_per_execution")
    _close(point["physical_failure_model_per_execution"], physical_error, f"{label}/physical_failure_model_per_execution")
    _close(point["qdk_combined_error_score_per_execution"], synthesis_error + physical_error,
           f"{label}/qdk_combined_error_score_per_execution")
    _close(point["physical_failure_union_bound_all_executions"], 2 * physical_error,
           f"{label}/physical_failure_union_bound_all_executions")
    _require(0 <= physical_error <= .005 / 2
             and 0 <= point["physical_failure_model_per_execution"] <= .005 / 2
             and 0 <= point["physical_failure_union_bound_all_executions"] <= .005,
             label, "physical failure allowance exceeded")
    return {"contract": contract, "profile": profile, "budget": report["budget"],
            "points": report["pareto_points"], "width": width,
            "operations": ledger["primitive_operations_per_group"],
            "power_calls": ledger["controlled_power_circuits_all_groups"],
            "qasm_sha256": qasm_digest, "num_rotations": rotations,
            "num_ts_per_rotation": num_ts, "logical_depth": logical_depth}


def _validate_aggregates(root, validated):
    scaling = _read_json(root / "surface-code-scaling.json", root)
    _same(scaling, {"status": STATUS, "common_absolute_energy_error_target_Ha": .5,
                    "same_ideal_failure": .005, "hardware_energy_confidence_certified": False,
                    "points": [{"data_qubits": CASE_PARAMETERS[case][0] * 2,
                                "phase_bits": CASE_PARAMETERS[case][1],
                                "contract": validated[case]["contract"], "status": STATUS,
                                "pareto_points": validated[case]["points"]}
                               for case in ("n2", "n4", "n6", "n8")]}, "surface-code-scaling")
    repeated, analytic = validated["default"], validated["analytic-powers"]
    _same(analytic["contract"], repeated["contract"], "matched access ideal contract")
    comparison = _read_json(root / "access-comparison.json", root)

    def branch(record):
        return {"logical_qubits": record["width"], "operations": record["operations"],
                "power_circuits_all_groups": record["power_calls"], "status": STATUS,
                "pareto_points": record["points"]}

    _same(comparison, {"same_problem_and_contract": repeated["contract"],
                       "same_physical_profile": repeated["profile"],
                       "same_synthesis_model_allowance": 1e-5,
                       "same_whole_workload_physical_allowance": .005,
                       "hardware_or_synthesis_certificate": False,
                       "repeated": branch(repeated), "analytic_powers": branch(analytic),
                       "scope": "exact algebraic dimer fast-forwarding; not generic Hamiltonian-oracle access"},
          "access-comparison")
    base = repeated["contract"]
    for case in ("n2", "n4", "n6", "n8"):
        contract = validated[case]["contract"]
        dimers = CASE_PARAMETERS[case][0]
        _close(contract["encoding"]["span"], dimers * base["encoding"]["span"], f"{case}/spectral span scaling")
        _close(contract["encoding"]["shift"], dimers * base["encoding"]["shift"], f"{case}/spectral shift scaling")
        _close(contract["ground_overlap_lower_bound"], base["ground_overlap_lower_bound"]**dimers,
               f"{case}/ground overlap scaling")


def validate_physical_records(records_dir, logical_ledgers, expected_profile):
    """Validate the complete frozen default/scaling/analytic physical bundle.

    ``records_dir`` follows the generator's results layout and must contain
    each physical JSON report and measured-group QASM, plus the two physical
    aggregate JSON files. ``logical_ledgers`` maps the exact keys ``default``,
    ``n2``, ``n4``, ``n6``, ``n8`` and ``analytic-powers`` to caller-authorized
    replayed ledgers. ``expected_profile`` is the four-field profile dictionary
    bound by the caller's immutable manifest. The Pauli subtree is excluded.

    Replayed ledgers may record disabled physical estimation; the complete
    frozen reports supply physical status. Original file/source/command/run
    hashes and the simulation-evidence replay are the caller's responsibility.
    ``ValueError`` denotes a missing, stale, partial or inconsistent record.
    """
    root = Path(records_dir).resolve()
    _require(root.is_dir(), str(root), "missing records directory")
    _require(isinstance(logical_ledgers, Mapping), "logical_ledgers", "expected a case mapping")
    _require(set(logical_ledgers) == set(CASE_PARAMETERS), "logical_ledgers", "missing or unexpected case")
    _require(isinstance(expected_profile, Mapping), "expected_profile", "expected a profile mapping")
    profile = dict(expected_profile)
    _keys(profile, PROFILE_KEYS, "expected_profile")
    _require(isinstance(profile["name"], str) and profile["name"], "expected_profile/name", "missing name")
    physical_rate = _number(profile["physical_error_rate"], "expected_profile/physical_error_rate", 0)
    _require(0 < physical_rate < .01, "expected_profile", "physical error outside surface-code domain")
    _integer(profile["gate_time_ns"], "expected_profile/gate_time_ns", 1)
    _integer(profile["measurement_time_ns"], "expected_profile/measurement_time_ns", 1)
    validated = {}
    for case in CASE_PARAMETERS:
        directory = root if case == "default" else root / case
        report = _read_json(directory / "surface-code.json", root)
        validated[case] = _validate_report(report, logical_ledgers[case], case, directory, root, profile)
    _validate_aggregates(root, validated)
    return {
        "status": "validated_frozen_conditional_records", "native_estimates_run": False,
        "audited_source_commit": AUDITED_SOURCE_COMMIT,
        "physical_profile": profile,
        "cases": {case: {key: value for key, value in record.items()
                         if key not in {"contract", "profile", "budget"}}
                  for case, record in validated.items()},
        "factory_choice_status": "trusted immutable native result; factory/Pareto search not rerun",
        "hardware_energy_confidence_certified": False,
        "limitations": [
            "The caller must verify immutable source/file/run bindings and replay simulation evidence",
            "Canonical logical counts are bound to frozen QDK traces, not independently rederived from QASM",
            "Factory summaries omit full distillation internals; their native choice and search frontier are trusted frozen records",
            "Arithmetic consistency does not certify native execution, concrete synthesis or hardware success",
        ],
    }
