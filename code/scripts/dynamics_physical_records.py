#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate frozen dynamics-model arithmetic using only the standard library.

The caller authenticates original artifact/source bindings and performs the
bounded logical replay. Native trace bucketization and factory recipes remain
trusted frozen outputs. This module neither searches nor certifies synthesis,
finite-H numerical evidence, hardware confidence or global optimality.
"""
from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat

MAX_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
DISTANCES = list(range(3, 42, 2))
PROFILES = [dict(name="frozen-p-1e-3", physical_error_rate=.001, gate_time_ns=100, measurement_time_ns=500),
            dict(name="frozen-p-1e-4", physical_error_rate=.0001, gate_time_ns=100, measurement_time_ns=500)]
PRIMITIVES = {"h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"}
AUDITED = "81de0d2e5347e1e9f10e05066320fffdb3b7058c"
MODEL_SOURCES = [f"https://github.com/microsoft/qdk/blob/{AUDITED}/source/qre/src/trace/transforms/psspc.rs",
                 f"https://github.com/microsoft/qdk/blob/{AUDITED}/source/qre/src/trace.rs",
                 f"https://github.com/microsoft/qdk/blob/{AUDITED}/source/qre/src/result.rs",
                 "https://learn.microsoft.com/en-us/azure/quantum/qre-estimation-results",
                 "https://arxiv.org/abs/2203.10064"]
BOUND_SOURCE = {
    "paper": "Theory of Trotter Error with Commutator Scaling", "doi": "10.1103/PhysRevX.11.011020",
    "arxiv": "https://arxiv.org/abs/1912.08854v3",
    "PDF_sha256": "c76e2980dee799253ac3a5fa6176ac9005f78c844c1243dc2e579433137c7965",
    "equations": "Sec. 5.1, Eqs. (143) and (150), PDF pages 38/39, arXiv v3 (3 February 2021)",
    "step_composition": "Unitary telescoping over r equal-duration steps",
}
EXCLUSIONS = [
    "Clean zero states are supplied; physical initialization/reset time and faults are not modeled",
    "Classical coefficient handling, compilation, acceptance/readout communication and verification latency are not quantum-model time",
    "QDK uses an empirical mean synthesis fit; no concrete Clifford+T sequence, channel error certificate or runtime-tail bound is supplied",
    "QDK abstracts Clifford scheduling; the source dependency depth is not a physical routed schedule",
    "Reset, intermediate measurement and dynamic-feedback instruments are not costed by this study",
]
SPACETIME = "rectangular peak-times-capped-time proxy, not integrated time-dependent occupancy"
CSV_FIELDS = ("case_id,method,profile,order,rounds,system_qubits,logical_qubits,time,accuracy,steps,"
              "attempt_cap_per_slot,complete_attempts,choice,peak_physical_qubits,data_code_distance,"
              "capped_runtime_seconds,peak_time_qubit_seconds,modeled_physical_fault_union_bound,"
              "synthesis_model_error_per_attempt,certified_confidence").split(",")

# Fixed guarded acceptance inputs of this measured study. A fresh numerical
# p_min must not reprice its synthesis, retry or execution budgets.
FROZEN_POSTSELECTION = {
    "qsp-oaa0-n3-t1-eps0.05": (.06249999846905541, 257),
    "qsp-oaa1-n3-t1-eps0.05": (.472656245714564, 26),
    "qsp-oaa2-n3-t1-eps0.05": (.9084472614612636, 8),
    "qsp-oaa3-n3-t1-eps0.05": (.9613189648305525, 6),
    "qsp-retargeted-n2-t1-eps0.05": (.06249999895235418, 257),
    "qsp-retargeted-n4-t1-eps0.05": (.06249991235141292, 257),
}
PHASE_WORDS = {
    2: ([1.5095453382308335, .12450914609070969, .08900796123225115, -.26519182587457535,
         -.8483246360474818, -.5493325876775899, -.04734630889933234, .38534336252098517,
         .11799332499378123, -.15521598755842278, -.1612524924308159, .0337550177311136,
         .11150239513024923, .8767356241867994, .7266747184070741],
        [.5217097950176859, -.4696953574740585, .05088442937698719, .05164822160080615,
         -.06595061142587275, .597493180611017, .17502649397688486, .6147523942101356,
         -.9701312686621334, -.09461966678507572, -.4518240147390462, .4832706738096487,
         .007275549814451783, .3959545546102542, -.02453438642710989, .40503050311070815]),
    4: ([.7691511669005053, .34078985824366403, .25264716543713367, -.8090070326686101,
         -.45069238689313723, -.6334959578119661, -1.0411031835640414, .5136937650523742,
         .6776190730773453, -.18470749012375232, -.5209251892661454, .1930183661597749,
         .4831245748611812, .6121707253429074, .909839425867432],
        [1.0932060122023792, -.4305707303269108, .09396540278016041, .49548975270048334,
         -.34157201125313846, .9479366047160769, .3983175608314927, 1.5785869320125472,
         -1.0442244133337926, -.5944451384360948, -1.443496019688043, .6354207450852886,
         .18442694227476383, .39838183009243405, -.3766327294029219, .21044853994485918]),
}
REPORT_KEYS = set("schema_version status resource_estimate_kind qdk_version qiskit_version audited_source_commit source_commit_is_package_release_commit qasm3 qasm_sha256 source_gate_counts decomposed_gate_counts canonical_gate_counts canonical_application_trace synthesized_application_trace synthesis budget physical_profile observable_contract error_composition isa_graph_pruning surface_code_data_distance_domain factory_surface_code_distance_domain native_default_distance_domain distance_domain_scope pareto_completeness runtime_semantics limitations sources pareto_points dynamics_task postselection_contract exclusions projection_wall_seconds".split())
POINT_KEYS = set("peak_physical_qubits data_code_distance runtime_per_execution_ns serial_runtime_all_executions_ns qdk_combined_error_score_per_execution synthesis_model_error_per_execution physical_failure_model_per_execution physical_failure_union_bound_all_executions factories model_properties instruction_source_graph maximum_complete_attempts peak_times_capped_runtime_qubit_seconds spacetime_scope".split())
PROPERTY_KEYS = set("LOGICAL_MEMORY_QUBITS EVALUATION_TIME NUM_TS_PER_ROTATION ALGORITHM_COMPUTE_QUBITS PHYSICAL_COMPUTE_QUBITS ALGORITHM_MEMORY_QUBITS LOGICAL_COMPUTE_QUBITS PHYSICAL_FACTORY_QUBITS".split())
COMMON_CASE_KEYS = set("case_id method system_qubits logical_qubits time J h hbar task initial_state observable source_gate_counts source_depth compiled_gate_counts compiled_depth numerical_checks complete_attempts attempts_per_accepted_slot accepted_slots cpu_timings_seconds measurement input_preparation compiled_global_phase_radians qasm_global_phase_scope exclusions qasm_sha256 provenance physical_reports".split())
POST_KEYS = set("reference_acceptance_lower synthesis_model_error_per_attempt conditional_acceptance_lower_after_synthesis_model conditional_synthesis_Pauli_bias_allowance attempts_per_accepted_slot accepted_slots maximum_complete_attempts all_slots_no_acceptance_union_bound conditional_expected_stopped_attempts_all_slots physical_failure_allowance_per_complete_attempt certified_confidence_lower_bound conditional_confidence_lower_bound assumptions proof does_not_certify".split())
SYNTHESIS_KEYS = set("model cost_semantics fit_domain num_arbitrary_rotations fixed_num_ts_per_rotation model_error_per_rotation model_error_per_execution actual_synthesis_sequence_generated instance_approximation_certificate_verified".split())
OBSERVABLE_KEYS = set("absolute_error_target algorithm_bias_bound synthesis_bias_allowance sampling_error_allowance sampling_failure_probability physical_failure_probability outcome_min outcome_max kind shots_required confidence_status certified_confidence_lower_bound conditional_confidence_lower_bound assumptions does_not_cover".split())
POST_ASSUMPTIONS = [
    "The stated finite-H numerical minimum acceptance is a valid lower bound; it is not formally certified",
    "The QDK mean-fit scalar upper-bounds total variation of the complete final joint data/ancilla distribution",
    "The independently declared ideal algorithm bias bound applies to each conditional Pauli mean",
    "Every independent trial starts with newly supplied clean zero qubits",
    "The declared physical-fault model and classical acceptance/readout interpretation are correct",
]
POST_PROOF = "For event E, restricted l1 difference a and complement b satisfy |p-q|<=b; conditional TV<=(a+|p-q|)/(2p)<=TV(P,Q)/p"
MODEL_LIMITATIONS = [
    "No synthesis/channel error certificate; empirical mean-fit costs are conditional",
    "QDK abstracts Clifford scheduling rather than retaining an exact routed schedule",
    "Physical reliability is conditional on the hardware/QEC/factory model",
    "Clean-state initialization/reset and classical communication/control timing are excluded",
]
STUDY_KEYS = set("schema_version status canonical_cases actual_physical_queries product_formula_cases QSP_OAA_cases QSP_retargeted_size_cases profiles provenance common_data_and_factory_distance_domain search_domain_change budget_scope frozen_file_sha256_before frozen_file_sha256_after frozen_files_unchanged algorithm_evidence conditional_confidence_scope certified_confidence_lower_bound QSP_physical_size_sweep dynamic_instrument_physical_estimation included exclusions sources cases".split())


def _require(condition, label, message):
    if not condition:
        raise ValueError(f"{label}: {message}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(value, expected, label):
    _require(_canonical(value) == _canonical(expected), label, "typed value differs from frozen contract")


def _keys(value, expected, label):
    _require(isinstance(value, dict) and set(value) == set(expected), label, "missing or unexpected fields")


def _integer(value, label, minimum=0):
    _require(type(value) is int and minimum <= value <= 10**18, label, "invalid bounded integer")
    return value


def _number(value, label, minimum=None, maximum=None):
    _require(type(value) in (int, float) and math.isfinite(value), label, "expected finite number")
    _require(minimum is None or value >= minimum, label, "number below bound")
    _require(maximum is None or value <= maximum, label, "number above bound")
    return value


def _uncertified(value, label):
    pending, nodes = [(value, 0)], 0
    flags = {"certified", "acceptance_certified", "guard_is_certificate", "numerical_lowering_guard_is_certificate",
             "actual_synthesis_sequence_generated", "instance_approximation_certificate_verified",
             "source_commit_is_package_release_commit"}
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= 200000 and depth <= 40, label, "JSON traversal bound exceeded")
        if isinstance(item, dict):
            _require(len(item) <= 200000 - nodes, label, "JSON object traversal bound exceeded")
            for key, child in item.items():
                if key in flags:
                    _same(child, False, f"{label}/{key}")
                if key == "certified_confidence_lower_bound":
                    _same(child, None, f"{label}/{key}")
                pending.append((child, depth + 1))
        elif isinstance(item, list):
            _require(len(item) <= 200000 - nodes, label, "JSON array traversal bound exceeded")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _require(math.isfinite(item), label, "nonfinite JSON number")


def _file(root, relative):
    name = PurePosixPath(relative)
    _require(name.as_posix() == relative and not name.is_absolute() and ".." not in name.parts,
             relative, "unsafe relative path")
    path = root
    for part in name.parts:
        path /= part
        _require(not path.is_symlink(), relative, "linked file or ancestor")
    _require(path.is_file() and stat.S_ISREG(path.stat().st_mode) and path.resolve().is_relative_to(root),
             relative, "missing regular record")
    _require(path.stat().st_size <= MAX_BYTES, relative, "record size bound exceeded")
    return path


def _read_json(path):
    _require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
             path.name, "JSON must be a regular nonlinked file")
    _require(path.stat().st_size <= MAX_BYTES, path.name, "JSON size bound exceeded")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, path.name, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"{path.name}: nonfinite JSON constant")
    try:
        value = json.loads(_read_bounded(path).decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{path.name}: unreadable JSON") from error
    _uncertified(value, path.name)
    return value


def _sha(path):
    return hashlib.sha256(_read_bounded(path)).hexdigest()


def _read_bounded(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    _require(len(data) <= MAX_BYTES, path.name, "actual record read exceeds size bound")
    return data


def _inventory(root, names):
    result, total = {}, 0
    for name in sorted(names):
        data = _read_bounded(_file(root, name))
        total += len(data)
        _require(total <= MAX_TOTAL_BYTES, "records", "actual aggregate record read exceeds size bound")
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def _grid():
    values = {(n, 1., order, .05) for n in range(2, 7) for order in (1, 2)}
    values.update((3, t, order, .05) for t in (.5, 2.) for order in (1, 2))
    values.update((3, 1., order, accuracy) for accuracy in (.025, .1) for order in (1, 2))
    result = {f"pf{order}-n{n}-t{t:g}-eps{accuracy:g}": (n, t, order, accuracy)
              for n, t, order, accuracy in sorted(values)}
    result.update((f"qsp-oaa{r}-n3-t1-eps0.05", (3, 1, r, .05)) for r in range(4))
    result.update((f"qsp-retargeted-n{n}-t1-eps0.05", (n, 1, 0, .05)) for n in (2, 4))
    return result


def _task(accuracy):
    e = Fraction(str(accuracy))
    sampling = float(3 * e / 5)
    shots = math.ceil(2 * math.log(2 / .004) / sampling**2)
    _same(shots, {.025: 55241, .05: 13811, .1: 3453}[accuracy], "task/shots")
    return dict(absolute_error=float(e), algorithm_bias=float(e / 5), synthesis_bias=float(e / 5),
                sampling_error=sampling, sampling_failure=.004, acceptance_failure=.001,
                physical_failure=.005, accepted_samples=shots, whole_task_failure_allowance=.01,
                certified_confidence_lower_bound=None)


def _counts(value, allowed, label):
    _require(isinstance(value, dict) and value and set(value) <= allowed, label, "unsupported counts")
    for name, count in value.items():
        _integer(count, f"{label}/{name}", 1)
    return value


def _angle(expression, label):
    _require(len(expression) <= 512, label, "angle too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError(f"{label}: invalid angle") from error
    _require(sum(1 for _ in ast.walk(tree)) <= 64, label, "angle expression bound exceeded")
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
    evaluate(tree.body)


def _qasm(path, width):
    headers = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{width}] meas;", f"qubit[{width}] q;"]
    gate = re.compile(r"([a-z]+)(?:\((.*)\))? (q\[\d+\](?:, q\[\d+\])?);")
    measure = re.compile(r"meas\[(\d+)\] = measure q\[(\d+)\];")
    counts, depths, seen, barrier, operations = Counter(), [0] * width, 0, False, 0
    digest = hashlib.sha256()
    for index, raw in enumerate(_read_bounded(path).splitlines(keepends=True)):
            digest.update(raw)
            _require(raw.endswith(b"\n") and len(raw) <= 2048, path.name, "invalid QASM line")
            try:
                line = raw[:-1].decode("ascii")
            except UnicodeError as error:
                raise ValueError("QASM must be ASCII") from error
            if index < 4:
                _same(line, headers[index], f"{path.name}/header")
                continue
            label = f"{path.name}:{index + 1}"
            if line.startswith("barrier "):
                _require(not barrier and seen == 0, label, "unexpected barrier")
                _same(line, "barrier " + ", ".join(f"q[{wire}]" for wire in range(width)) + ";", label)
                barrier = True
                counts["barrier"] += 1
                continue
            found = measure.fullmatch(line)
            if found:
                _require(barrier and tuple(map(int, found.groups())) == (seen, seen) and seen < width,
                         label, "invalid final all-wire readout")
                depths[seen] += 1
                seen += 1
                counts["measure"] += 1
                continue
            _require(not barrier and seen == 0, label, "operation after final barrier/readout")
            found = gate.fullmatch(line)
            _require(found is not None, label, "opaque/unsupported QASM")
            name, angle, operands = found.groups()
            _require(name in PRIMITIVES, label, "unsupported primitive")
            wires = [int(value) for value in re.findall(r"q\[(\d+)\]", operands)]
            _require(len(wires) == (2 if name == "cx" else 1) and len(set(wires)) == len(wires)
                     and all(0 <= wire < width for wire in wires), label, "invalid operands")
            _require((angle is not None) == (name in {"rx", "ry", "rz"}), label, "invalid parameter count")
            if angle is not None:
                _angle(angle, label)
            depth = max(depths[wire] for wire in wires) + 1
            for wire in wires:
                depths[wire] = depth
            counts[name] += 1
            operations += 1
            _require(operations <= 100000, label, "primitive bound exceeded")
    _same(seen, width, path.name + "/measurements")
    return dict(counts), max(depths), digest.hexdigest()


def _source_pf_depth(n, steps, order):
    depths = [0] * n
    def append(wires):
        value = max(depths[wire] for wire in wires) + 1
        for wire in wires:
            depths[wire] = value
    def zz():
        for wire in range(n - 1):
            append((wire, wire + 1))
            append((wire + 1,))
            append((wire, wire + 1))
    for _ in range(steps):
        zz()
        for wire in range(n):
            append((wire,))
        if order == 2:
            zz()
    return max(depths)


def _pf(case, params, task):
    n, t, order, accuracy = params
    label = case["case_id"]
    _keys(case, COMMON_CASE_KEYS | set("order boundary steps terms analytic_error acceptance_probability no_acceptance_failure_actual".split()), label)
    for key, value in {"method": "product_formula", "system_qubits": n, "logical_qubits": n,
                       "time": t, "order": order, "boundary": "open"}.items():
        _same(case.get(key), value, label + "/" + key)
    allowance = Fraction(str(task["algorithm_bias"])) / 2
    reserve = allowance / 50
    ab = Fraction(14, 5) * (n - 1)
    aab = Fraction(28, 5) * (2 * n - 3)
    bba = Fraction(196, 25) * (n - 1)
    time = Fraction(str(t))
    ratio = (time**2 * ab / 2 if order == 1 else time**3 * (aab / 24 + bba / 12)) / (allowance - reserve)
    steps = -(-ratio.numerator // ratio.denominator) if order == 1 else math.isqrt(ratio.numerator // ratio.denominator)
    if order == 2 and Fraction(steps * steps) < ratio:
        steps += 1
    _same(case["steps"], steps, label + "/steps")
    bound = time**2 * ab / (2 * steps) if order == 1 else time**3 * (aab / 24 + bba / 12) / steps**2
    dt = t / steps
    rx, rz = 2 * .7 * dt, (2 if order == 1 else 1) * 1.0 * dt
    rx_count, rz_count = n * steps, (n - 1) * steps * order
    rounding = (rx_count * abs(Fraction.from_float(rx) - 2 * Fraction(7, 10) * time / steps)
                + rz_count * abs(Fraction.from_float(rz) - (2 if order == 1 else 1) * time / steps)) / 2
    _same(case["source_gate_counts"], dict(cx=2 * rz_count, rx=rx_count, rz=rz_count), label + "/source counts")
    _same(case["source_depth"], _source_pf_depth(n, steps, order), label + "/source depth")
    analytic = case["analytic_error"]
    _keys(analytic, set("commutator_Pauli_l1_bounds_exact product_formula_operator_bound_exact source_angle_rounding_operator_bound_exact operator_bound_upper_float Pauli_bias_bound_upper_float numerical_lowering_guard_added numerical_lowering_guard_is_certificate rounding_reserve scope source".split()), label + "/analytic")
    for key, value in {"commutator_Pauli_l1_bounds_exact": {"AB": str(ab), "AAB": str(aab), "BBA": str(bba)},
                       "product_formula_operator_bound_exact": str(bound),
                       "source_angle_rounding_operator_bound_exact": str(rounding),
                       "rounding_reserve": float(reserve), "numerical_lowering_guard_added": 1e-10,
                       "numerical_lowering_guard_is_certificate": False, "source": BOUND_SOURCE}.items():
        _same(analytic.get(key), value, label + "/" + key)
    upper = math.nextafter(float(bound + rounding), math.inf) + 1e-10
    _same(analytic["operator_bound_upper_float"], upper, label + "/operator bound")
    _same(analytic["Pauli_bias_bound_upper_float"], 2 * upper, label + "/bias bound")
    _require(bound + rounding <= allowance and rounding <= reserve and 2 * upper <= task["algorithm_bias"],
             label, "algorithm allowance exceeded")
    checks = case["numerical_checks"]
    _keys(checks, set("order steps operator_error state_vector_error state_infidelity observable exact_observable circuit_observable observable_error source_to_lowered_operator_error certified performed".split()), label + "/checks")
    _same(checks["observable"], "Z(q0)", label + "/check observable")
    _same(checks.get("certified"), False, label + "/certified")
    _same(checks.get("performed"), True, label + "/performed")
    _same(checks.get("order"), order, label + "/check order")
    _same(checks.get("steps"), steps, label + "/check steps")
    _number(checks["operator_error"], label + "/operator error", 0, math.nextafter(float(bound + rounding), math.inf) + 1e-10)
    _number(checks["source_to_lowered_operator_error"], label + "/lowering error", 0, 1e-10)
    _number(checks["state_vector_error"], label + "/state error", 0, 2)
    _number(checks["state_infidelity"], label + "/infidelity", 0, 1)
    for key in ("exact_observable", "circuit_observable"):
        _number(checks[key], label + "/" + key, -1, 1)
    _same(checks["observable_error"], abs(checks["exact_observable"] - checks["circuit_observable"]), label + "/observable error")
    for key, value in {"complete_attempts": task["accepted_samples"], "attempts_per_accepted_slot": 1,
                       "accepted_slots": task["accepted_samples"], "acceptance_probability": 1,
                       "no_acceptance_failure_actual": 0}.items():
        _same(case.get(key), value, label + "/" + key)
    return dict(synthesis_model_error_per_execution=task["synthesis_bias"] / 2,
                physical_failure_probability=.005, executions=task["accepted_samples"])


def _postselection(case, task):
    label = case["case_id"]
    contract = case["postselection_contract"]
    _keys(contract, POST_KEYS, label + "/postselection")
    _same(contract["assumptions"], POST_ASSUMPTIONS, label + "/postselection assumptions")
    _same(contract["proof"], POST_PROOF, label + "/postselection proof")
    _same(contract["does_not_certify"], ["concrete synthesis", "operator/diamond error", "runtime tails", "fresh-state/reset hardware cost"], label + "/postselection exclusions")
    p, cap = FROZEN_POSTSELECTION[label]
    epsilon, slots = task["synthesis_bias"] * p / 2, task["accepted_samples"]
    lower, attempts = p - epsilon, cap * slots
    for key, value in {"reference_acceptance_lower": p, "synthesis_model_error_per_attempt": epsilon,
                       "conditional_acceptance_lower_after_synthesis_model": lower,
                       "conditional_synthesis_Pauli_bias_allowance": 2 * epsilon / p,
                       "attempts_per_accepted_slot": cap, "accepted_slots": slots, "maximum_complete_attempts": attempts,
                       "all_slots_no_acceptance_union_bound": slots * (1 - lower)**cap,
                       "physical_failure_allowance_per_complete_attempt": .005 / attempts,
                       "conditional_confidence_lower_bound": .99, "certified_confidence_lower_bound": None}.items():
        _same(contract.get(key), value, label + "/" + key)
    _require(slots * (1 - lower)**cap <= .001 and (cap == 1 or slots * (1 - lower)**(cap - 1) > .001),
             label, "retry cap is not the frozen minimal cap")
    expected = -math.expm1(cap * math.log1p(-lower)) / lower * slots
    _require(math.isclose(contract["conditional_expected_stopped_attempts_all_slots"], expected, rel_tol=1e-15),
             label, "expected stopped attempts differ")
    for key, value in {"complete_attempts": attempts, "attempts_per_accepted_slot": cap, "accepted_slots": slots}.items():
        _same(case.get(key), value, label + "/" + key)
    _same(case["numerical_checks"]["finite_H_numerical_minimum_acceptance_probability"] - 1e-9,
          p, label + "/frozen guarded acceptance")
    return dict(synthesis_model_error_per_execution=epsilon, physical_failure_probability=.005, executions=attempts)


def _qsp(case, params, task):
    n, t, rounds, _ = params
    label = case["case_id"]
    retargeted = label.startswith("qsp-retargeted")
    expected_keys = COMMON_CASE_KEYS | set("phase_tau rounds access postselection_contract size_scaling_scope".split())
    _keys(case, expected_keys | ({"phase_evidence"} if retargeted else set()), label)
    width, tau = {2: 6, 3: 8, 4: 9}[n], {2: 2.4, 3: 4.1, 4: 5.8}[n]
    for key, value in {"method": "QSP_retargeted" if retargeted else "QSP_oblivious_amplification",
                       "system_qubits": n, "logical_qubits": width, "time": t, "rounds": rounds, "phase_tau": tau}.items():
        _same(case.get(key), value, label + "/" + key)
    _same(case["access"], dict(forward_A_calls=rounds + 1, inverse_A_calls=rounds,
        total_A_or_inverse_calls=2 * rounds + 1, zero_ancilla_reflections=2 * rounds,
        uncontrolled_block_encoding_calls=14 * (2 * rounds + 1), controlled_block_encoding_calls=2 * rounds + 1,
        branch_conditioned_projector_phase_calls=31 * (2 * rounds + 1)), label + "/access")
    checks = case["numerical_checks"]
    _counts(case["source_gate_counts"], PRIMITIVES | {"cPauli LCU", "cPauli LCU_dg", "cry", "crz", "ccry", "ccx", "mcx", "c3rz", "c4rz"}, label + "/source counts")
    _same(checks.get("guard_is_certificate"), False, label + "/guard certificate")
    _same(checks["acceptance_guard_subtracted"], 1e-9, label + "/acceptance guard")
    _number(checks["seeded_full_state_lowering_error"], label + "/full-register lowering", 0, 1e-10)
    pmin = _number(checks["finite_H_numerical_minimum_acceptance_probability"], label + "/minimum acceptance", 0, 1)
    pmax = _number(checks["finite_H_numerical_maximum_acceptance_probability"], label + "/maximum acceptance", pmin, 1)
    _number(checks["actual_input_acceptance_probability"], label + "/input acceptance", pmin, pmax)
    if retargeted:
        evidence = case["phase_evidence"]
        _keys(evidence, {"cosine_phases", "sine_phases", "cosine", "sine", "constructor_tolerance", "phase_fit_method", "certified"}, label + "/phase evidence")
        _keys(checks, set("accepted_block_basis_trajectories_per_source_or_lowering compiled_block_error_vs_exact_U_over4 source_to_compiled_accepted_block_operator_error seeded_full_state_lowering_error numerical_lowering_guard_added all_input_conditional_Pauli_bias_projection actual_input_acceptance_probability finite_H_numerical_minimum_acceptance_probability finite_H_numerical_maximum_acceptance_probability conditional_state_vector_error observable_exact observable_conditional observable_error acceptance_guard_subtracted guard_is_certificate".split()), label + "/checks")
        _same(evidence.get("certified"), False, label + "/phase certificate")
        _same(evidence["cosine_phases"], PHASE_WORDS[n][0], label + "/cosine words")
        _same(evidence["sine_phases"], PHASE_WORDS[n][1], label + "/sine words")
        _same(evidence["constructor_tolerance"], {2: 1e-8, 4: 1e-5}[n], label + "/phase tolerance")
        _same(evidence["phase_fit_method"], "deterministic analytic-Jacobian continuation from frozen tau4.1, maxstep.2/max500evals/per-component60s", label + "/phase provenance")
        for kind, degree in (("cosine", 14), ("sine", 15)):
            phase = evidence[kind]
            projection_key = "uniform_error_bound_with_numerical_reconstruction_guard" if kind == "cosine" else "conditional_numerical_uniform_error_projection"
            _keys(phase, set("degree tau target target_chebyshev_coefficients achieved_chebyshev_coefficients phase_fit_max_residual_10001_point_grid target_max_residual_10001_point_grid numerically_reconstructed_coefficient_l1_discrepancy analytic_chebyshev_truncation_bound coefficient_reconstruction_guard verification_scope certified".split()) | {projection_key}, label + "/phase component")
            _same(phase["target"], ("cos" if kind == "cosine" else "sin") + "(tau*x)/2 on [-1,1]", label + "/phase target")
            scope = ("analytic Bessel-tail bound plus floating coefficient reconstruction and dense-grid checks; no formal interval proof"
                     if kind == "cosine" else "analytic tail bound plus floating coefficient reconstruction and dense-grid check; no formal interval proof")
            _same(phase["verification_scope"], scope, label + "/phase verification scope")
            _same(phase.get("certified"), False, label + "/component certificate")
            _same(phase["degree"], degree, label + "/phase degree")
            _same(phase["tau"], tau, label + "/phase tau")
            for key in ("target_chebyshev_coefficients", "achieved_chebyshev_coefficients"):
                _require(isinstance(phase[key], list) and len(phase[key]) == degree + 1, label, "phase coefficient shape differs")
                for value in phase[key]:
                    _number(value, label + "/coefficient")
            _same(phase["coefficient_reconstruction_guard"], 1e-12, label + "/coefficient guard")
            _number(phase["phase_fit_max_residual_10001_point_grid"], label + "/phase residual", 0, evidence["constructor_tolerance"])
            target, achieved = phase["target_chebyshev_coefficients"], phase["achieved_chebyshev_coefficients"]
            for index, value in enumerate(target):
                if index % 2 != (0 if kind == "cosine" else 1):
                    _same(value, 0.0, label + "/target parity")
            discrepancy = sum(abs(left - right) for left, right in zip(target, achieved))
            reported = _number(phase["numerically_reconstructed_coefficient_l1_discrepancy"], label + "/coefficient discrepancy", 0)
            _require(abs(reported - discrepancy) <= 16 * math.ulp(max(reported, discrepancy)), label, "coefficient discrepancy arithmetic differs")
            tail = _number(phase["analytic_chebyshev_truncation_bound"], label + "/tail bound", 0)
            _same(phase[projection_key], tail + reported + 1e-12, label + "/phase error projection")
            _number(phase["target_max_residual_10001_point_grid"], label + "/target residual", 0, phase[projection_key])
        _same(case["source_depth"], {2: 321, 4: 755}[n], label + "/source depth")
        _same(checks["accepted_block_basis_trajectories_per_source_or_lowering"], 1 << n, label + "/basis columns")
        _number(checks["source_to_compiled_accepted_block_operator_error"], label + "/block lowering", 0, 1e-10)
        eta = _number(checks["compiled_block_error_vs_exact_U_over4"], label + "/block error", 0, .25 - 1e-10)
        bias = 4 * (eta + 1e-10) / (.25 - eta - 1e-10)
        _same(checks["numerical_lowering_guard_added"], 1e-10, label + "/lowering guard")
    else:
        _keys(checks, set("rounds ideal_input_amplitude ideal_scaled_unitary_amplified_amplitude ideal_scaled_unitary_success_probability ideal_success_formula_scope actual_input_acceptance_probability actual_minus_ideal_success_probability block_error_vs_ideal_amplitude_times_exact_U decoded_linear_map_operator_error conditional_state_check_scope conditional_state_vector_error conditional_state_infidelity observable_exact observable_conditional observable_error actual_block_singular_values finite_H_numerical_minimum_acceptance_probability finite_H_numerical_maximum_acceptance_probability finite_H_numerical_acceptance_input_dependence_span acceptance_certified accepted_block_polynomial_error all_input_conditional_Pauli_bias_projection seeded_full_state_lowering_error acceptance_guard_subtracted guard_is_certificate".split()), label + "/checks")
        amplitude = (.25, .6875, .953125, .98046875)[rounds]
        _same(case["source_depth"], 587 + 1184 * rounds, label + "/source depth")
        _same(checks.get("acceptance_certified"), False, label + "/acceptance certificate")
        _same(checks["rounds"], rounds, label + "/checked rounds")
        _same(checks["ideal_input_amplitude"], .25, label + "/ideal amplitude")
        _same(checks["ideal_scaled_unitary_amplified_amplitude"], amplitude, label + "/amplified amplitude")
        _same(checks["ideal_scaled_unitary_success_probability"], amplitude**2, label + "/ideal success")
        _number(checks["accepted_block_polynomial_error"], label + "/OAA polynomial", 0, 1e-11)
        eta = _number(checks["block_error_vs_ideal_amplitude_times_exact_U"], label + "/block error", 0, amplitude)
        bias = 4 * eta / (abs(amplitude) - eta)
        _same(checks["actual_minus_ideal_success_probability"], checks["actual_input_acceptance_probability"] - amplitude**2, label + "/success discrepancy")
        _same(checks["finite_H_numerical_acceptance_input_dependence_span"], pmax - pmin, label + "/acceptance span")
        singular = checks["actual_block_singular_values"]
        _require(isinstance(singular, list) and len(singular) == 8 and singular == sorted(singular, reverse=True), label, "singular-value evidence shape/order differs")
        for value in singular:
            _number(value, label + "/singular value", 0, 1)
        _require(abs(pmin - singular[-1]**2) <= 8 * math.ulp(pmin) and abs(pmax - singular[0]**2) <= 8 * math.ulp(pmax), label, "singular acceptance arithmetic differs")
        _number(checks["conditional_state_infidelity"], label + "/infidelity", 0, 1)
    _number(checks["conditional_state_vector_error"], label + "/conditional state error", 0, 2)
    for key in ("observable_exact", "observable_conditional"):
        _number(checks[key], label + "/" + key, -1, 1)
    _same(checks["observable_error"], abs(checks["observable_exact"] - checks["observable_conditional"]), label + "/observable error")
    _same(checks["all_input_conditional_Pauli_bias_projection"], bias, label + "/conditional bias")
    _require(bias <= task["algorithm_bias"], label, "conditional bias exceeds fixed allowance")
    return _postselection(case, task)


def _raw_trace(text, width, counts, label):
    _require(isinstance(text, str) and len(text) <= 65536, label, "invalid canonical trace")
    lines = text.splitlines()
    _same(lines[:2], [f"@compute_qubits({width})", "{"], label + "/header")
    _same(lines[-1:], ["}"], label + "/end")
    index, total, rotation_depth = 2, Counter(), 0
    while index < len(lines) - 1:
        match = re.fullmatch(r"  repeat (\d+) \{", lines[index])
        _require(match is not None, label, "unsupported trace block")
        repeat = _integer(int(match.group(1)), label + "/repeat", 1)
        index += 1
        operations = []
        while index < len(lines) - 1 and lines[index] != "  }":
            match = re.fullmatch(r"    (RZ|T|MEAS_Z)\((\d+)\)", lines[index])
            _require(match is not None, label, "unsupported trace operation")
            operations.append((match.group(1), int(match.group(2))))
            index += 1
        _require(index < len(lines) - 1 and operations, label, "unterminated/empty trace block")
        name = operations[0][0]
        _same(operations, [(name, wire) for wire in range(len(operations))], label + "/parallel bucket")
        _require(len(operations) <= width and (name == "RZ" or len(operations) == 1), label, "invalid bucket width")
        total[name] += repeat * len(operations)
        if name == "RZ":
            rotation_depth += repeat
        index += 1
    _same(dict(total), counts, label + "/counts")
    return rotation_depth


def _source_graph(profile, distance):
    gates = (f"GateBased(error_rate={profile['physical_error_rate']}, gate_time={profile['gate_time_ns']}, "
             f"measurement_time={profile['measurement_time_ns']}, two_qubit_gate_time={profile['gate_time_ns']})")
    return ("LATTICE_SURGERY @ SurfaceCode(crossing_prefactor=0.03, error_correction_threshold=0.01, "
        "one_qubit_gate_depth=1, two_qubit_gate_depth=4, code_cycle_override=None, "
        f"code_cycle_offset=0, distance={distance})\n  CNOT @ {gates}\n  H @ {gates}\n  MEAS_Z @ {gates}\n"
        "T @ RoundBasedFactory(code_query=_ComponentQuery(component=<class 'qdk.qre.models.qec._surface_code.SurfaceCode'>, "
        f"source=RootNode(), kwargs={{'distance': {DISTANCES}}}), physical_qubit_calculation=<built-in function sum>)\n  T @ {gates}")


def _observable(report, case, task, label):
    if case["method"] != "product_formula":
        _same(report["observable_contract"], None, label + "/observable")
        return
    contract = report["observable_contract"]
    _keys(contract, OBSERVABLE_KEYS, label + "/observable")
    _same(contract["confidence_status"], "conditional_only; independent approximation evidence is required", label + "/observable confidence")
    _same(contract["assumptions"], [
        "Independent identically distributed bounded shots at one fixed circuit/parameter setting",
        "The caller independently establishes the algorithm_bias_bound",
        "The synthesis model scalar upper-bounds output-distribution total variation; this module does not certify it",
        "The declared physical-fault model upper-bounds faults and the classical measurement interpretation is correct",
    ], label + "/observable assumptions")
    _same(contract["does_not_cover"], ["optimizer convergence", "noncommuting observable grouping", "multiple parameter settings", "adaptive stopping"], label + "/observable exclusions")
    for key, value in {"absolute_error_target": task["absolute_error"],
        "algorithm_bias_bound": case["analytic_error"]["Pauli_bias_bound_upper_float"],
        "synthesis_bias_allowance": task["synthesis_bias"], "sampling_error_allowance": task["sampling_error"],
        "sampling_failure_probability": .004, "physical_failure_probability": .005,
        "outcome_min": -1.0, "outcome_max": 1.0, "kind": "one_bounded_scalar_mean",
        "shots_required": task["accepted_samples"], "certified_confidence_lower_bound": None,
        "conditional_confidence_lower_bound": .991}.items():
        _same(contract.get(key), value, label + "/observable/" + key)


def _report(report, case, profile, qasm_digest, budget):
    label = case["case_id"] + "/" + profile["name"]
    _keys(report, REPORT_KEYS, label)
    for key, value in {"schema_version": 1, "status": "computed_conditional_model",
        "resource_estimate_kind": "conditional_mean_fit_resource_projection", "qdk_version": "1.34.0", "qiskit_version": "2.6.2",
        "audited_source_commit": AUDITED, "source_commit_is_package_release_commit": False,
        "physical_profile": profile, "budget": budget, "dynamics_task": case["task"],
        "postselection_contract": case.get("postselection_contract"), "exclusions": EXCLUSIONS,
        "surface_code_data_distance_domain": DISTANCES, "factory_surface_code_distance_domain": DISTANCES,
        "native_default_distance_domain": list(range(3, 26, 2)), "isa_graph_pruning": False, "sources": MODEL_SOURCES,
        "error_composition": "UnionBound; synthesis base and physical faults are explicitly separated",
        "pareto_completeness": "Enumerated listed domain only; not a global optimality certificate",
        "distance_domain_scope": "A common bounded extended search for all dynamics methods; unchanged physical/code formulas and task budgets; not the frozen Grover search domain",
        "runtime_semantics": "Capped circuit attempts times QDK mean-fit per-attempt model time, not a stochastic synthesis runtime-tail bound",
        "limitations": MODEL_LIMITATIONS}.items():
        _same(report.get(key), value, label + "/" + key)
    _observable(report, case, case["task"], label)
    _number(report["projection_wall_seconds"], label + "/projection wall", 0)
    _same(report["qasm_sha256"], qasm_digest, label + "/QASM digest")
    _require(isinstance(report["qasm3"], str), label, "missing embedded QASM")
    _same(hashlib.sha256(report["qasm3"].encode()).hexdigest(), qasm_digest, label + "/embedded QASM")
    _same(report["decomposed_gate_counts"], case["compiled_gate_counts"], label + "/primitive counts")
    _same(report["source_gate_counts"], {**case["source_gate_counts"], "barrier": 1, "measure": case["logical_qubits"]},
          label + "/measured source counts")
    counts = _counts(report["canonical_gate_counts"], {"RZ", "T", "MEAS_Z"}, label + "/canonical counts")
    _require(set(counts) == ({"RZ", "MEAS_Z"} if case["method"] == "product_formula" else {"RZ", "T", "MEAS_Z"}),
             label, "trace keys differ")
    _same(counts["MEAS_Z"], case["logical_qubits"], label + "/measurements")
    rotation_depth = _raw_trace(report["canonical_application_trace"], case["logical_qubits"], counts, label + "/raw trace")
    synthesis, rotations = report["synthesis"], counts["RZ"]
    _keys(synthesis, SYNTHESIS_KEYS, label + "/synthesis")
    for key, value in {
        "model": "Audited QDK PSSPC mixed-fallback mean fit for random angles",
        "cost_semantics": "Expected fallback work, not a worst-case or per-angle certificate",
        "fit_domain": "Per-rotation modeled accuracy < 1e-4",
    }.items():
        _same(synthesis[key], value, label + "/synthesis/" + key)
    ts = max(12, math.ceil(4.86 + .53 * (math.log2(rotations) - math.log2(budget["synthesis_model_error_per_execution"]))))
    while rotations * 2**((4.86 - ts) / .53) > budget["synthesis_model_error_per_execution"]:
        ts += 1
    per_rotation = 2**((4.86 - ts) / .53)
    base = rotations * per_rotation
    for key, value in {"num_arbitrary_rotations": rotations, "fixed_num_ts_per_rotation": ts,
                       "model_error_per_rotation": per_rotation, "model_error_per_execution": base,
                       "actual_synthesis_sequence_generated": False, "instance_approximation_certificate_verified": False}.items():
        _same(synthesis.get(key), value, label + "/synthesis/" + key)
    _require(0 < per_rotation < 1e-4 and base <= budget["synthesis_model_error_per_execution"], label, "synthesis allowance exceeded")
    width = 2 * case["logical_qubits"] + math.ceil(math.sqrt(8 * case["logical_qubits"])) + 1
    depth = counts["MEAS_Z"] + counts.get("T", 0) + rotations + ts * rotation_depth
    states = counts.get("T", 0) + ts * rotations
    pattern = r"@compute_qubits\((\d+)\)\n@base_error\(([-+0-9.eE]+)\)\n@resource_state\(T, (\d+)\)\n\{\n  repeat (\d+) \{\n    MULTI_PAULI_MEAS\(([0-9, ]+)\)\n  \}\n\}\n"
    match = re.fullmatch(pattern, report["synthesized_application_trace"])
    _require(match is not None, label, "unsupported synthesized trace")
    sw, se, ss, sd, wires = match.groups()
    _same([int(sw), float(se), int(ss), int(sd), wires],
          [width, base, states, depth, ", ".join(map(str, range(width)))], label + "/synthesized trace")
    points = report["pareto_points"]
    _require(isinstance(points, list) and 1 <= len(points) <= 128, label, "invalid Pareto point count")
    for index, point in enumerate(points):
        tag = f"{label}/point/{index}"
        _keys(point, POINT_KEYS, tag)
        properties = point["model_properties"]
        _keys(properties, PROPERTY_KEYS, tag + "/properties")
        for name, value in properties.items():
            _integer(value, tag + "/" + name)
        for name, value in {"LOGICAL_MEMORY_QUBITS": 0, "ALGORITHM_MEMORY_QUBITS": 0, "NUM_TS_PER_ROTATION": ts,
                            "ALGORITHM_COMPUTE_QUBITS": case["logical_qubits"], "LOGICAL_COMPUTE_QUBITS": width}.items():
            _same(properties[name], value, tag + "/" + name)
        distance = _integer(point["data_code_distance"], tag + "/distance", 3)
        _require(distance in DISTANCES, tag, "data distance outside frozen domain")
        _same(point["instruction_source_graph"], _source_graph(profile, distance), tag + "/source graph")
        data_qubits = width * (2 * distance**2 - 1)
        _same(properties["PHYSICAL_COMPUTE_QUBITS"], data_qubits, tag + "/data qubits")
        runtime, executions = float(depth * distance * 1000), budget["executions"]
        _same(point["runtime_per_execution_ns"], runtime, tag + "/runtime")
        _same(point["serial_runtime_all_executions_ns"], runtime * executions, tag + "/capped runtime")
        _same(point["maximum_complete_attempts"], executions, tag + "/complete attempts")
        _keys(point["factories"], {"T"}, tag + "/factories")
        factory = point["factories"]["T"]
        _keys(factory, {"copies", "runs", "states", "error_rate"}, tag + "/T")
        for name in ("copies", "runs", "states"):
            _integer(factory[name], tag + "/T/" + name, 1)
        _same(factory["states"], states, tag + "/T states")
        _same(factory["copies"], (states + factory["runs"] - 1) // factory["runs"], tag + "/T copies")
        factory_error = _number(factory["error_rate"], tag + "/T error", 0, 1)
        factory_qubits = properties["PHYSICAL_FACTORY_QUBITS"]
        _require(factory_qubits > 0 and factory_qubits % factory["copies"] == 0, tag, "factory space/copy aggregate differs")
        _same(point["peak_physical_qubits"], data_qubits + factory_qubits, tag + "/peak qubits")
        combined = _number(point["qdk_combined_error_score_per_execution"], tag + "/combined error", 0)
        physical = max(0., combined - base)
        _same(point["synthesis_model_error_per_execution"], base, tag + "/synthesis error")
        _same(point["physical_failure_model_per_execution"], physical, tag + "/error subtraction")
        derived = depth * width * .03 * (profile["physical_error_rate"] / .01)**((distance + 1) // 2) + states * factory_error
        # QDK composes the synthesis base with data/factory error; the producer
        # then subtracts that base. Stored subtraction stays exact. A separate
        # reconstruction can differ at base-scale ulps due to floating order,
        # as in the audited QDK 1.34 / 81de0d2e UnionBound result equations.
        _require(abs(physical - derived) <= 8 * max(math.ulp(base), math.ulp(combined)),
                 tag, "data/factory physical arithmetic differs")
        _same(point["physical_failure_union_bound_all_executions"], physical * executions, tag + "/physical union bound")
        _require(0 <= physical <= .005 / executions and physical * executions <= .005 and derived * executions <= .005,
                 tag, "whole physical allowance exceeded")
        _same(point["peak_times_capped_runtime_qubit_seconds"],
              point["peak_physical_qubits"] * point["serial_runtime_all_executions_ns"] / 1e9, tag + "/spacetime")
        _same(point["spacetime_scope"], SPACETIME, tag + "/spacetime scope")
    return points


def _selected(points):
    return {"fewest_qubits": min(points, key=lambda p: (p["peak_physical_qubits"], p["serial_runtime_all_executions_ns"])),
            "shortest_runtime": min(points, key=lambda p: (p["serial_runtime_all_executions_ns"], p["peak_physical_qubits"])),
            "smallest_peak_time_proxy": min(points, key=lambda p: p["peak_times_capped_runtime_qubit_seconds"])}


def validate_physical_records(records_dir, study_manifest):
    """Check the complete fixed saved study without dispatching computation."""
    supplied = Path(records_dir)
    _require(not supplied.is_symlink(), "records", "linked root")
    root = supplied.resolve()
    _require(root.is_dir(), "records", "missing record directory")
    _require(isinstance(study_manifest, Mapping), "manifest", "expected study mapping")
    study = _read_json(_file(root, "manifest.json"))
    _keys(study, STUDY_KEYS, "study")
    _same(study, dict(study_manifest), "study manifest")
    grid = _grid()
    names = {"manifest.json", "summary.md", "physical-summary.csv", "product-formula-scaling.svg",
             "qsp-amplification-resources.svg", "qsp-size-resources.svg", "dynamics_resources.py",
             "generate_dynamics_resources.py", "LICENSE.txt"}
    names |= {f"{case}/{name}" for case in grid for name in
              ("logical-resources.json", "measured-trial.qasm", "frozen-p-1e-3.json", "frozen-p-1e-4.json")}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    _require(actual - {"resource-record-manifest.json"} == names, "records", "missing or unexpected original files")
    directories = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}
    _same(sorted(directories), sorted(grid), "record directories")
    _require(sum(_file(root, name).stat().st_size for name in names) <= MAX_TOTAL_BYTES,
             "records", "aggregate record size bound exceeded")
    before = _inventory(root, names)
    if "resource-record-manifest.json" in actual:
        _read_bounded(_file(root, "resource-record-manifest.json"))
    try:
        for key, value in {"schema_version": 1, "status": "computed_conditional_model", "canonical_cases": 24,
            "actual_physical_queries": 48, "product_formula_cases": 18, "QSP_OAA_cases": 4, "QSP_retargeted_size_cases": 2,
            "profiles": PROFILES, "common_data_and_factory_distance_domain": DISTANCES,
            "frozen_files_unchanged": True, "certified_confidence_lower_bound": None, "exclusions": EXCLUSIONS}.items():
            _same(study.get(key), value, "study/" + key)
        _same(study["frozen_file_sha256_before"], study["frozen_file_sha256_after"], "frozen baseline")
        _require(isinstance(study["cases"], list) and [c.get("case_id") for c in study["cases"]] == list(grid),
                 "cases", "fixed grid/order differs")
        expected_rows, reports, point_count = [], 0, 0
        for case in study["cases"]:
            label, params = case["case_id"], grid[case["case_id"]]
            n, _, _, accuracy = params
            task = _task(accuracy)
            ledger = _read_json(_file(root, label + "/logical-resources.json"))
            _same(ledger, case, label + "/ledger")
            _same(case["provenance"], study["provenance"], label + "/provenance")
            for key, value in {"task": task, "J": 1, "h": .7, "hbar": 1, "initial_state": "0" * n,
                               "observable": "Z(q0)", "exclusions": EXCLUSIONS,
                               "physical_reports": [p["name"] + ".json" for p in PROFILES]}.items():
                _same(case.get(key), value, label + "/" + key)
            for value in case["cpu_timings_seconds"].values():
                _number(value, label + "/CPU timing", 0)
            _number(case["compiled_global_phase_radians"], label + "/phase")
            budget = _pf(case, params, task) if label.startswith("pf") else _qsp(case, params, task)
            primitive, depth, digest = _qasm(_file(root, label + "/measured-trial.qasm"), case["logical_qubits"])
            _same(primitive, case["compiled_gate_counts"], label + "/QASM counts")
            _same(depth, case["compiled_depth"], label + "/QASM depth")
            _same(digest, case["qasm_sha256"], label + "/QASM digest")
            for profile in PROFILES:
                report = _read_json(_file(root, label + "/" + profile["name"] + ".json"))
                points = _report(report, case, profile, digest, budget)
                reports += 1
                point_count += len(points)
                for choice, point in _selected(points).items():
                    expected_rows.append(dict(case_id=label, method=case["method"], profile=profile["name"], order=case.get("order", ""),
                        rounds=case.get("rounds", ""), system_qubits=n, logical_qubits=case["logical_qubits"], time=case["time"], accuracy=accuracy,
                        steps=case.get("steps", ""), attempt_cap_per_slot=case["attempts_per_accepted_slot"], complete_attempts=case["complete_attempts"],
                        choice=choice, peak_physical_qubits=point["peak_physical_qubits"], data_code_distance=point["data_code_distance"],
                        capped_runtime_seconds=point["serial_runtime_all_executions_ns"] / 1e9,
                        peak_time_qubit_seconds=point["peak_times_capped_runtime_qubit_seconds"],
                        modeled_physical_fault_union_bound=point["physical_failure_union_bound_all_executions"],
                        synthesis_model_error_per_attempt=report["synthesis"]["model_error_per_execution"], certified_confidence="unavailable"))
        with _file(root, "physical-summary.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            _same(reader.fieldnames, CSV_FIELDS, "CSV fields")
            rows = list(reader)
        _same(rows, [{name: str(row[name]) for name in CSV_FIELDS} for row in expected_rows], "all CSV selections")
        _same([reports, point_count, len(rows)], [48, 489, 144], "study coverage")
    finally:
        _same(_inventory(root, names), before, "immutable original files")
    return {"status": "validated_frozen_conditional_dynamics_records", "cases": 24, "reports": 48,
            "pareto_points": 489, "csv_rows": 144, "native_estimates_run": False,
            "profiles": [dict(profile) for profile in PROFILES], "distance_domain": list(DISTANCES),
            "limitations": [
                "Caller must authenticate original artifact/source/receipt bindings and perform bounded logical replay",
                "Canonical Native RZ bucketization is trusted frozen evidence, not independently derived from QASM",
                "Factory recipes, configurations and Pareto choices are trusted immutable Native outputs; search is not repeated",
                "Stored finite-H acceptance and phase evidence remain numerical; no interval/channel/hardware certificate is produced",
                "Arithmetic consistency does not attest fresh Native execution, global optimality or actual synthesis",
            ]}
