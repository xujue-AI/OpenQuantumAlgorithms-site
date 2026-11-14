"""SOURCE-only contracts for the frozen local-spin generation driver.

The driver must populate ``sys.modules['_oqp_local_spin_test_context']``
before collection.  This module never imports the generator or the legacy
package, draws random choices, constructs a quantum state, or requests a
pytest fixture.  The ten test protocols are below the 64-protocol admission
cap.  Passing tests emit 750 new primitives: 35 + 55 + 645 + one 15-operation
writer fixture.  Every new primitive is charged to both driver counters;
the additional hard stop is 2,000 primitives.
"""

from collections import Counter
import hashlib
import json
import math
import re
import stat
import sys


TEST_CONTEXT_MODULE = "_oqp_local_spin_test_context"
TEST_PROTOCOL_LIMIT = 64
SOURCE_FIXTURE_PRIMITIVE_LIMIT = 2_000
SOURCE_FILE_BYTES_LIMIT = 8 * 1024 * 1024
SOURCE_PACKET_BYTES_LIMIT = 16 * 1024 * 1024
_fixture_source_primitives = 0


def _context():
    context = sys.modules.get(TEST_CONTEXT_MODULE)
    assert context is not None, "the frozen driver context is required"
    return context


def _frozen_rows():
    # These choices were admitted before generator/reference execution.
    return (
        ("n1-ground", 1, 8, "ground", 2026111301,
         (0, 0, 0, 0), (0, 2, 0, 2), (0, 0, 0, 0)),
        ("n1-thermal", 1, 8, "thermal", 2026111302,
         (0, 0, 0, 0), (0, 2, 1, 2), (1, 0, 0, 0)),
        ("n2-ground", 2, 8, "ground", 2026111303,
         (1, 1, 0, 0), (2, 2, 1, 0), (0, 0, 0, 0)),
        ("n2-thermal", 2, 8, "thermal", 2026111304,
         (0, 0, 0, 1), (2, 1, 1, 2), (0, 0, 1, 0)),
        ("n4-thermal-generation", 4, 16, "thermal", 2026111404,
         (1, 1, 2, 2), (0, 1, 2, 1), (0, 0, 0, 0)),
        ("n8-thermal-generation", 8, 16, "thermal", 2026111408,
         (6, 3, 4, 0), (0, 2, 1, 2), (0, 0, 0, 0)),
        ("n16-thermal-generation", 16, 16, "thermal", 2026111416,
         (10, 13, 3, 10), (1, 0, 2, 2), (0, 1, 0, 1)),
        ("n32-thermal-generation", 32, 16, "thermal", 2026111432,
         (1, 14, 13, 6), (2, 1, 1, 2), (0, 0, 0, 1)),
        ("n64-thermal-generation", 64, 16, "thermal", 2026111464,
         (42, 58, 41, 35), (0, 0, 1, 2), (0, 0, 1, 1)),
    )


def _expected_counts(case, bath_x):
    n, k, length = case.n, case.rounds, case.steps
    return {
        "rz": (4 * n + 1) * k * length,
        "h": 4 * k * length,
        "cx": (4 * n - 2) * k * length,
        "reset": n + k + 1,
        "measure": n,
        "x": bath_x,
    }


def _expected_total(case, bath_x):
    return ((8 * case.n + 3) * case.rounds * case.steps
            + 2 * case.n + case.rounds + 1 + bath_x)


def _signature(operation):
    return operation.name, operation.qubits, operation.angle, operation.clbit


def _rejects(error_type, action):
    try:
        action()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def _blocked(*args, **kwargs):
    raise AssertionError("a source, file, or RNG side effect was attempted")


def _without_source_side_effects(generator, action):
    saved = (
        generator.make_schedule, generator.iter_operations,
        generator.iter_round_operations, generator.write_qasm,
        generator.random.Random, generator.tempfile.mkstemp,
    )
    try:
        generator.make_schedule = _blocked
        generator.iter_operations = _blocked
        generator.iter_round_operations = _blocked
        generator.write_qasm = _blocked
        generator.random.Random = _blocked
        generator.tempfile.mkstemp = _blocked
        return action()
    finally:
        (generator.make_schedule, generator.iter_operations,
         generator.iter_round_operations, generator.write_qasm,
         generator.random.Random, generator.tempfile.mkstemp) = saved


def _fixture_schedule(context, case, site=0, frequency_index=0):
    original = context.schedules["n1-ground"]
    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "rng": original["rng"],
        "choice_order": original["choice_order"],
        "sites": [site],
        "frequency_indices": [frequency_index],
        "bath_bits": [0],
        "logical_rng_calls": {"site": 1, "frequency": 1, "bath_bit": 0, "total": 2},
    }


def _copy_schedule(schedule):
    return {
        **schedule,
        "sites": list(schedule["sites"]),
        "frequency_indices": list(schedule["frequency_indices"]),
        "bath_bits": list(schedule["bath_bits"]),
        "logical_rng_calls": dict(schedule["logical_rng_calls"]),
    }


def _charge_fixture_primitive(ledger, amount=1):
    global _fixture_source_primitives
    assert type(amount) is int and amount == 1
    assert _fixture_source_primitives + amount <= SOURCE_FIXTURE_PRIMITIVE_LIMIT
    ledger.consume("fixture_source_primitives", amount)
    ledger.consume("source_primitives", amount)
    _fixture_source_primitives += amount
    ledger.guard()


def _fixture_operations(context, case, schedule):
    for operation in context.generator.iter_operations(case, schedule):
        _charge_fixture_primitive(context.ledger)
        yield operation


class _FixtureWriterLedger:
    # Deliberately has no record_source_operation callback: the frozen small
    # source cache belongs to the driver and cannot receive fixture operations.
    def __init__(self, ledger):
        self.ledger = ledger
        self.primitives = 0
        self.bytes = 0

    def guard(self):
        self.ledger.guard()

    def consume(self, name, amount):
        assert type(amount) is int and amount >= 0
        if name == "source_primitives":
            _charge_fixture_primitive(self.ledger, amount)
            self.primitives += amount
        elif name == "source_bytes":
            self.ledger.consume("source_bytes", amount)
            self.bytes += amount
            self.ledger.guard()
        else:
            raise AssertionError(f"unexpected writer ledger counter: {name}")


def _owned_directory(context):
    directory = context.output / "cache" / "source-contracts"
    assert directory == context.source_contracts_dir
    info = directory.lstat()
    assert stat.S_ISDIR(info.st_mode) and not directory.is_symlink()
    return directory


def _read_source(path, expected_bytes):
    assert type(expected_bytes) is int and 0 < expected_bytes <= SOURCE_FILE_BYTES_LIMIT
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    assert not path.is_symlink() and info.st_size == expected_bytes
    with path.open("rb") as source:
        encoded = source.read(info.st_size + 1)
    assert len(encoded) == expected_bytes
    after = path.lstat()
    assert stat.S_ISREG(after.st_mode) and after.st_nlink == 1
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
    )
    return encoded


def _parse_qasm(encoded, n):
    # This parser is independent of qasm_header/operation_qasm.  Requiring a
    # complete token match rejects ignored statements and incorrect registers.
    assert encoded.endswith(b"\n")
    text = encoded.decode("ascii")
    assert "\r" not in text
    lines = text.splitlines(keepends=True)
    assert lines[:4] == [
        "OPENQASM 2.0;\n", 'include "qelib1.inc";\n',
        f"qreg q[{n + 1}];\n", f"creg c[{n}];\n",
    ]
    number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    parsed = []
    for line in lines[4:]:
        match = re.fullmatch(rf"rz\(({number})\) q\[([0-9]+)\];\n", line)
        if match:
            angle = float(match.group(1))
            assert math.isfinite(angle)
            item = ("rz", (int(match.group(2)),), angle, None)
        else:
            match = re.fullmatch(r"cx q\[([0-9]+)\],q\[([0-9]+)\];\n", line)
            if match:
                item = ("cx", (int(match.group(1)), int(match.group(2))), None, None)
            else:
                match = re.fullmatch(r"measure q\[([0-9]+)\] -> c\[([0-9]+)\];\n", line)
                if match:
                    item = ("measure", (int(match.group(1)),), None, int(match.group(2)))
                else:
                    match = re.fullmatch(r"(h|x|reset) q\[([0-9]+)\];\n", line)
                    assert match is not None, f"unsupported QASM statement: {line!r}"
                    item = (match.group(1), (int(match.group(2)),), None, None)
        assert all(0 <= qubit <= n for qubit in item[1])
        if item[0] == "cx":
            assert item[1][0] != item[1][1]
        if item[0] == "measure":
            assert 0 <= item[3] < n
        parsed.append(item)
    return tuple(parsed)


def test_frozen_cases_and_seeded_choice_coverage():
    context = _context()
    rows = _frozen_rows()
    assert len(context.cases) == len(rows) == 9
    identifiers = {row[0] for row in rows}
    assert set(context.schedules) == identifiers
    assert set(context.receipts) == identifiers
    totals = Counter()
    for case, row in zip(context.cases, rows, strict=True):
        case_id, n, steps, mode, seed, sites, frequencies, bits = row
        assert (case.case_id, case.n, case.rounds, case.steps, case.mode, case.seed) == (
            case_id, n, 4, steps, mode, seed,
        )
        schedule = context.schedules[case_id]
        assert set(schedule) == {
            "case_id", "seed", "rng", "choice_order", "sites", "frequency_indices",
            "bath_bits", "logical_rng_calls",
        }
        assert (schedule["case_id"], schedule["seed"]) == (case_id, seed)
        assert schedule["rng"] == "random.Random(MT19937)"
        assert schedule["choice_order"] == "site/frequency/bath_bit"
        assert tuple(schedule["sites"]) == sites
        assert tuple(schedule["frequency_indices"]) == frequencies
        assert tuple(schedule["bath_bits"]) == bits
        expected_calls = {
            "site": 4, "frequency": 4, "bath_bit": 4 if mode == "thermal" else 0,
            "total": 12 if mode == "thermal" else 8,
        }
        assert schedule["logical_rng_calls"] == expected_calls
        context.generator.validate_schedule(case, schedule)
        totals.update(schedule["logical_rng_calls"])
    assert dict(totals) == {"site": 36, "frequency": 36, "bath_bit": 28, "total": 100}


def test_nine_emitted_receipts_match_bytes_and_complete_source_formula():
    context = _context()
    packet_bytes = 0
    total_instructions = 0
    for case in context.cases:
        receipt = context.receipts[case.case_id]
        schedule = context.schedules[case.case_id]
        expected_counts = _expected_counts(case, sum(schedule["bath_bits"]))
        expected_total = _expected_total(case, sum(schedule["bath_bits"]))
        assert receipt["case_id"] == case.case_id
        assert type(receipt["file"]) is str
        assert "/" not in receipt["file"] and "\\" not in receipt["file"]
        assert receipt["file"] not in ("", ".", "..")
        assert receipt["primitive_counts"] == expected_counts
        assert receipt["instruction_count"] == expected_total == sum(expected_counts.values())
        assert receipt["source_preparations_materialized"] == 1
        assert receipt["status"] == "source_emitted_no_quantum_simulation_or_accuracy_certificate"
        encoded = _read_source(context.source_dir / receipt["file"], receipt["bytes"])
        assert hashlib.sha256(encoded).hexdigest() == receipt["sha256"]
        packet_bytes += len(encoded)
        assert packet_bytes <= SOURCE_PACKET_BYTES_LIMIT
        total_instructions += expected_total
    assert total_instructions == 66_680


def test_small_cache_contains_complete_reset_evolution_readout_preparations():
    context = _context()
    cache = context.ledger.context["small_source_operations"]
    assert cache is context.small_source_operations
    assert set(cache) == {"n1-ground", "n1-thermal", "n2-ground", "n2-thermal"}
    total = 0
    for case in context.cases[:4]:
        operations = cache[case.case_id]
        schedule = context.schedules[case.case_id]
        expected = _expected_counts(case, sum(schedule["bath_bits"]))
        assert all(type(operation) is context.generator.Operation for operation in operations)
        assert dict(Counter(operation.name for operation in operations)) == {
            name: count for name, count in expected.items() if count
        }
        assert len(operations) == _expected_total(case, sum(schedule["bath_bits"]))
        signatures = tuple(_signature(operation) for operation in operations)
        assert signatures[:case.n + 1] == tuple(
            ("reset", (qubit,), None, None) for qubit in range(case.n + 1)
        )
        assert signatures[-case.n:] == tuple(
            ("measure", (qubit,), None, qubit) for qubit in range(case.n)
        )
        cursor = case.n + 1
        for bit in schedule["bath_bits"]:
            if bit:
                assert signatures[cursor] == ("x", (case.n,), None, None)
                cursor += 1
            end = cursor + (8 * case.n + 3) * case.steps
            assert all(item[0] in ("rz", "h", "cx") for item in signatures[cursor:end])
            assert signatures[end] == ("reset", (case.n,), None, None)
            cursor = end + 1
        assert cursor == len(signatures) - case.n
        total += len(operations)
    assert total == 1_954


def test_complete_small_qasm_roundtrips_to_actual_cached_operations():
    context = _context()
    cache = context.small_source_operations
    for case in context.cases[:4]:
        receipt = context.receipts[case.case_id]
        encoded = _read_source(context.source_dir / receipt["file"], receipt["bytes"])
        parsed = _parse_qasm(encoded, case.n)
        actual = tuple(_signature(operation) for operation in cache[case.case_id])
        # Exact float equality is deliberate: 17 significant digits preserve
        # each source float, including negative bath RZ and XX-middle angles.
        assert parsed == actual
        assert len(parsed) == receipt["instruction_count"]


def test_non_power_sizes_and_maximum_n_emit_bounded_complete_sources():
    context = _context()
    fixture_total = 0
    for n, expected_total in ((3, 35), (5, 55), (64, 645)):
        case = context.generator.Case(f"fixture-n{n}", n, 1, 1, "ground", 1)
        schedule = _fixture_schedule(context, case, site=n - 1, frequency_index=1)
        context.generator.validate_schedule(case, schedule)
        operations = tuple(_fixture_operations(context, case, schedule))
        assert len(operations) == expected_total == _expected_total(case, 0)
        counts = dict(Counter(operation.name for operation in operations))
        assert counts == {name: count for name, count in _expected_counts(case, 0).items() if count}
        signatures = tuple(_signature(operation) for operation in operations)
        assert signatures[:n + 1] == tuple(
            ("reset", (qubit,), None, None) for qubit in range(n + 1)
        )
        assert signatures[-n - 1:] == (("reset", (n,), None, None),) + tuple(
            ("measure", (qubit,), None, qubit) for qubit in range(n)
        )
        assert all(0 <= qubit <= n for item in signatures for qubit in item[1])
        assert all(item[0] not in ("reset", "measure", "x") for item in signatures[n + 1:-n - 1])
        half_length = 4 * n - 2
        evolution = signatures[n + 1:-n - 1]
        first_half = evolution[:half_length]
        assert tuple((item[0], item[1]) for item in first_half[:n]) == tuple(
            ("rz", (qubit,)) for qubit in range(n)
        )
        assert tuple((item[0], item[1]) for item in first_half[n:-1]) == tuple(
            item for qubit in range(n - 1) for item in (
                ("cx", (qubit, qubit + 1)), ("rz", (qubit + 1,)), ("cx", (qubit, qubit + 1)),
            )
        )
        assert first_half[-1][0:2] == ("rz", (n,))
        assert tuple((item[0], item[1]) for item in evolution[half_length:half_length + 7]) == (
            ("h", (n - 1,)), ("h", (n,)), ("cx", (n - 1, n)), ("rz", (n,)),
            ("cx", (n - 1, n)), ("h", (n - 1,)), ("h", (n,)),
        )
        assert evolution[-half_length:] == tuple(reversed(first_half))
        report = context.generator.count_source(case, schedule)
        assert report["primitive_counts_per_preparation"] == _expected_counts(case, 0)
        assert report["instruction_count_per_preparation"] == expected_total
        fixture_total += len(operations)
    assert fixture_total == 735


def test_repetition_projection_counts_every_complete_preparation_without_emission():
    context = _context()
    generator = context.generator

    def check_projections():
        for case in context.cases:
            schedule = context.schedules[case.case_id]
            bath_x = sum(schedule["bath_bits"])
            counts = _expected_counts(case, bath_x)
            total = _expected_total(case, bath_x)
            single = generator.count_source(case, schedule)
            assert single["primitive_counts_per_preparation"] == counts
            assert single["instruction_count_per_preparation"] == total
            assert single["repeated_primitive_count_ranges"] == {
                name: [count, count] for name, count in counts.items()
            }
            assert single["repeated_instruction_count_range"] == [total, total]
            report = generator.count_source(case, schedule, repetitions=25)
            assert report["primitive_counts_per_preparation"] == counts
            assert report["instruction_count_per_preparation"] == total
            repeated_counts = {
                name: [25 * count, 25 * count] for name, count in counts.items()
            }
            if case.mode == "thermal":
                repeated_counts["x"] = [0, 25 * case.rounds]
                base = _expected_total(case, 0)
                repeated_total = [25 * base, 25 * (base + case.rounds)]
            else:
                repeated_total = [25 * total, 25 * total]
            assert report["repeated_primitive_count_ranges"] == repeated_counts
            assert report["repeated_instruction_count_range"] == repeated_total
            assert report["total_system_readouts"] == 25 * case.n
            assert report["repeated_rotation_count"] == 25 * counts["rz"]
            assert report["complete_preparations_projected"] == report["repetitions"] == 25
            assert report["initial"] == "reset_prepared_all_zero_only"
            assert report["status"] == "formula_only_unexecuted_projection"
            assert report["repetition_scope"] == "projection_only_full_preparation_each_trial_emitter_P1_only"
            assert report["all_components_verified"] is False
            assert report["accuracy"] == {
                "finite_circuit_reference": "unexecuted", "trotter_error": "uncertified",
                "quadrature_error": "uncertified", "finite_pulse_bias": "uncertified",
                "continuum_bridge": "unavailable", "mixing_certificate": False,
                "gibbs_accuracy_certificate": False, "sampling_confidence": "uncomputed",
            }
            assert report["excluded_costs"]["rotation_synthesis_T_count_and_depth"] == "uncomputed_unsynthesized_RZ"
            assert report["excluded_costs"]["physical_error_factories_distance_qubits_runtime_and_success"] == "uncomputed"
        thermal = context.cases[3]
        report = generator.count_source(thermal, repetitions=25)
        base = _expected_total(thermal, 0)
        assert report["instruction_count_per_preparation"] is None
        assert report["primitive_counts_per_preparation"]["x"] is None
        assert report["repeated_primitive_count_ranges"]["x"] == [0, 25 * thermal.rounds]
        assert report["repeated_instruction_count_range"] == [25 * base, 25 * (base + thermal.rounds)]
        for repetitions in (True, 1.0, 0, 1_000_001):
            _rejects(ValueError, lambda: generator.count_source(thermal, repetitions=repetitions))

    _without_source_side_effects(generator, check_projections)


def test_case_limits_and_plain_integer_contract_reject_before_source_or_rng():
    generator = _context().generator
    bad_cases = [
        ("limit-n", 65, 1, 1, "ground", 1),
        ("limit-k", 1, 65, 1, "ground", 1),
        ("limit-l", 1, 1, 129, "ground", 1),
        ("limit-intersection", 64, 64, 128, "ground", 1),
        ("bool-n", True, 1, 1, "ground", 1),
        ("float-n", 1.0, 1, 1, "ground", 1),
        ("bool-k", 1, True, 1, "ground", 1),
        ("float-k", 1, 1.0, 1, "ground", 1),
        ("bool-l", 1, 1, True, "ground", 1),
        ("float-l", 1, 1, 1.0, "ground", 1),
        ("bool-seed", 1, 1, 1, "ground", True),
        ("float-seed", 1, 1, 1, "ground", 1.0),
        ("../escape", 1, 1, 1, "ground", 1),
        ("a/b", 1, 1, 1, "ground", 1),
        ("\u00e9", 1, 1, 1, "ground", 1),
        (object(), 1, 1, 1, "ground", 1),
        ("bad-mode", 1, 1, 1, "unsupported", 1),
    ]

    def reject_cases():
        for values in bad_cases:
            _rejects(ValueError, lambda: generator.Case(*values))

    _without_source_side_effects(generator, reject_cases)


def test_invalid_schedules_reject_before_the_first_source_primitive():
    context = _context()
    case = context.cases[3]
    original = context.schedules[case.case_id]
    malformed = []
    missing = _copy_schedule(original)
    del missing["seed"]
    malformed.append(missing)
    extra = _copy_schedule(original)
    extra["unrequested"] = 1
    malformed.append(extra)
    for key, value in (
        ("seed", float(case.seed)), ("sites", [0]),
        ("sites", [0, 0, 0, 2]), ("sites", [0, 0, 0, True]),
        ("frequency_indices", [0, 1, 2, 3]), ("bath_bits", (0, 0, 1, 0)),
    ):
        schedule = _copy_schedule(original)
        schedule[key] = value
        malformed.append(schedule)
    wrong_calls = _copy_schedule(original)
    wrong_calls["logical_rng_calls"]["total"] = 11
    malformed.append(wrong_calls)
    for schedule in malformed:
        _rejects(ValueError, lambda: context.generator.validate_schedule(case, schedule))
        _rejects(ValueError, lambda: next(_fixture_operations(context, case, schedule)))
    ground_case = context.cases[0]
    ground_schedule = _copy_schedule(context.schedules[ground_case.case_id])
    ground_schedule["bath_bits"][0] = 1
    _rejects(ValueError, lambda: next(_fixture_operations(context, ground_case, ground_schedule)))


def test_operation_tokens_and_fixed_cases_schema_reject_invalid_inputs():
    context = _context()
    generator = context.generator
    for angle in (float("nan"), float("inf"), -float("inf"), True, 1j):
        _rejects(ValueError, lambda: generator.Operation("rz", (0,), angle=angle))
    for clbit in (True, 1.0, None, 64):
        _rejects(ValueError, lambda: generator.Operation("measure", (0,), clbit=clbit))
    _rejects(ValueError, lambda: generator.Operation("cx", (0, 0)))
    _rejects(ValueError, lambda: generator.Operation("h", (True,)))
    _rejects(ValueError, lambda: generator.Operation("h", (0,), clbit=0))
    loaded = generator.load_cases(context.case_file)
    assert tuple((case.case_id, case.n, case.rounds, case.steps, case.mode, case.seed) for case in loaded) == tuple(
        (case.case_id, case.n, case.rounds, case.steps, case.mode, case.seed) for case in context.cases
    )
    with context.case_file.open("rb") as source:
        raw = source.read(128 * 1024 + 1)
    assert len(raw) <= 128 * 1024
    payload = json.loads(raw)
    invalid_payloads = []
    for edit in ("version-type", "extra-key", "model", "role", "case-int", "method"):
        candidate = json.loads(json.dumps(payload, allow_nan=False))
        if edit == "version-type":
            candidate["schema_version"] = True
        elif edit == "extra-key":
            candidate["extra"] = 1
        elif edit == "model":
            candidate["system"]["zz_coefficient"] = 0.25
        elif edit == "role":
            candidate["cases"][0]["role"] = "generation_only"
        elif edit == "case-int":
            candidate["cases"][0]["n"] = 1.0
        else:
            candidate["method_id"] = "another-model"
        invalid_payloads.append(json.dumps(candidate, allow_nan=False).encode("utf-8"))
    compact = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    invalid_payloads.append(b'{"schema_version":1,' + compact[1:])
    destination = _owned_directory(context) / "invalid-cases.json"
    for encoded in invalid_payloads:
        assert len(encoded) <= 128 * 1024
        with destination.open("xb") as output:
            output.write(encoded)
        try:
            info = destination.lstat()
            assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            _rejects(ValueError, lambda: generator.load_cases(destination))
        finally:
            destination.unlink()


def test_ordinary_writer_boundaries_and_existing_destination_preservation():
    context = _context()
    generator = context.generator
    directory = _owned_directory(context)
    before = {entry.name for entry in directory.iterdir()}
    case = generator.Case("fixture-write", 1, 1, 1, "ground", 1)
    schedule = _fixture_schedule(context, case)
    adapter = _FixtureWriterLedger(context.ledger)
    long_name = directory / ("a" * 129)
    missing_parent = directory / "absent-parent" / "fixture-write.qasm"
    _rejects(ValueError, lambda: generator.write_qasm(case, schedule, long_name, ledger=adapter))
    _rejects(ValueError, lambda: generator.write_qasm(case, schedule, missing_parent, ledger=adapter))
    destination = directory / "fixture-write.qasm"
    _rejects(ValueError, lambda: generator.write_qasm(case, schedule, destination, ledger=object()))
    assert {entry.name for entry in directory.iterdir()} == before
    assert adapter.primitives == adapter.bytes == 0
    cache_before = {case_id: len(operations) for case_id, operations in context.small_source_operations.items()}
    receipt = generator.write_qasm(case, schedule, destination, ledger=adapter)
    assert receipt["case_id"] == "fixture-write" and receipt["file"] == destination.name
    assert receipt["instruction_count"] == adapter.primitives == 15
    assert receipt["primitive_counts"] == {"rz": 5, "h": 4, "cx": 2, "x": 0, "reset": 3, "measure": 1}
    assert receipt["source_preparations_materialized"] == 1
    assert receipt["status"] == "source_emitted_no_quantum_simulation_or_accuracy_certificate"
    encoded = _read_source(destination, receipt["bytes"])
    assert adapter.bytes == len(encoded)
    assert hashlib.sha256(encoded).hexdigest() == receipt["sha256"]
    parsed = _parse_qasm(encoded, 1)
    assert len(parsed) == 15
    assert parsed[:2] == (("reset", (0,), None, None), ("reset", (1,), None, None))
    assert parsed[-2:] == (("reset", (1,), None, None), ("measure", (0,), None, 0))
    _rejects(FileExistsError, lambda: generator.write_qasm(case, schedule, destination, ledger=adapter))
    assert _read_source(destination, receipt["bytes"]) == encoded
    assert adapter.primitives == 15 and adapter.bytes == len(encoded)
    assert {case_id: len(operations) for case_id, operations in context.small_source_operations.items()} == cache_before
    assert {entry.name for entry in directory.iterdir()} == before | {destination.name}
    assert not any(entry.name.endswith(".part") for entry in directory.iterdir())
