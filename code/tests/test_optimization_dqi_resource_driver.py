"""Pure control/storage boundaries; no process, SDK, simulation or test ran.

These definitions are for the later reviewed test command.  The script is
loaded as ordinary source, without its main entry or a package initializer.
Every file fixture is small and local; native calls and subprocesses are not
part of this file's tests.
"""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/generate_optimization_dqi_resources.py"
spec = importlib.util.spec_from_file_location("_dqi_resource_driver_boundary_test", SOURCE)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
NONCE = "a" * 32


class SpyGuard:
    def __init__(self, expected=()):
        self.expected = set(expected)
        self.seen = set()

    def charge(self, kind, key):
        pair = (kind, key)
        if pair not in self.expected or pair in self.seen:
            raise ValueError("Unexpected or repeated entry")
        self.seen.add(pair)

    def require_complete(self):
        if self.seen != self.expected:
            raise ValueError("Entries missing")

    def snapshot(self):
        return {"observed_entries": len(self.seen), "forbidden_queries": 0}


def event(event_type, at=11.0, **fields):
    return dict(event=event_type, nonce=NONCE, at_hex=float(at).hex(), **fields)


def test_many_complete_control_lines_in_one_read_are_valid():
    first = b"x" * 3000 + b"\n"
    second = b"y" * 3000 + b"\n"
    remainder, lines, total = driver.control_lines(b"", first + second, 0)
    assert remainder == b"" and lines == [first[:-1], second[:-1]]
    assert total == len(first + second)


def test_unterminated_control_line_is_bounded():
    with pytest.raises(ValueError):
        driver.control_lines(b"", b"x" * (driver.CONTROL_LINE_CAP + 1), 0)


def test_complete_control_line_is_bounded():
    with pytest.raises(ValueError):
        driver.control_lines(b"", b"x" * (driver.CONTROL_LINE_CAP + 1) + b"\n", 0)


def test_control_total_bound_cannot_be_reset_by_line_splitting():
    with pytest.raises(ValueError):
        driver.control_lines(b"", b"ok\n", driver.CONTROL_CAP)


def test_orphan_native_return_is_rejected():
    watch = driver.CallWatch(NONCE, SpyGuard(), 10.0, 70.0)
    with pytest.raises(ValueError):
        watch.accept(event("native_end", sequence=1, status="returned"), 11.0)


def test_overlap_is_rejected_before_second_charge():
    guard = SpyGuard({("trace", 2), ("trace", 3)})
    watch = driver.CallWatch(NONCE, guard, 10.0, 70.0)
    watch.accept(event("native_begin", sequence=1, kind="trace", key=2), 11.0)
    with pytest.raises(ValueError):
        watch.accept(event("native_begin", at=12, sequence=2, kind="trace", key=3), 12.0)
    assert guard.seen == {("trace", 2)}


def test_repeated_spent_entry_remains_rejected_after_a_return():
    guard = SpyGuard({("trace", 2)})
    watch = driver.CallWatch(NONCE, guard, 10.0, 70.0)
    watch.accept(event("native_begin", sequence=1, kind="trace", key=2), 11.0)
    watch.accept(event("native_end", at=12, sequence=1, status="returned"), 12.0)
    with pytest.raises(ValueError):
        watch.accept(event("native_begin", at=13, sequence=2, kind="trace", key=2), 13.0)


def test_completion_cannot_replace_missing_entries():
    guard = SpyGuard({("trace", 2)})
    watch = driver.CallWatch(NONCE, guard, 10.0, 70.0)
    with pytest.raises(ValueError):
        watch.accept(event("complete", counts=guard.snapshot()), 11.0)


def test_false_zero_counter_does_not_match_integer_zero():
    guard = SpyGuard()
    watch = driver.CallWatch(NONCE, guard, 10.0, 70.0)
    with pytest.raises(ValueError):
        watch.accept(event("complete", counts={"observed_entries": 0, "forbidden_queries": False}), 11.0)


def test_native_return_clock_includes_the_whole_entry():
    guard = SpyGuard({("trace", 2)})
    watch = driver.CallWatch(NONCE, guard, 10.0, 70.0)
    watch.accept(event("native_begin", sequence=1, kind="trace", key=2), 11.0)
    with pytest.raises(ValueError):
        watch.accept(event("native_end", at=16.1, sequence=1, status="returned"), 16.1)


def test_integer_sequence_rejects_boolean_alias():
    guard = SpyGuard({("trace", 2)})
    watch = driver.CallWatch(NONCE, guard, 10.0, 70.0)
    watch.accept(event("native_begin", sequence=1, kind="trace", key=2), 11.0)
    with pytest.raises(ValueError):
        watch.accept(event("native_end", at=12, sequence=True, status="returned"), 12.0)


def test_control_timestamp_cannot_precede_the_task():
    watch = driver.CallWatch(NONCE, SpyGuard(), 10.0, 70.0)
    with pytest.raises(ValueError):
        watch.accept(event("complete", at=9, counts={}), 11.0)


def test_native_capture_failure_retains_bounded_partial_metadata():
    class BadWord:
        num_qubits = 2
        num_clbits = 0
        data = ()
    raw = driver.failed_return_capture(BadWord(), ValueError())
    assert raw["complete"] is False and raw["validated"] is False
    assert raw["num_qubits"] == 2 and raw["operation_count"] == 0


def test_native_error_supplied_raw_fields_are_retained_unvalidated():
    error = ValueError()
    error.raw_result = {"runtime": {"type": "int", "value": 330000},
                        "error": {"type": "float", "hex": "0x1p-20"}}
    raw = driver.failed_return_capture(None, error)
    assert raw["provided_raw_result"] == error.raw_result
    assert raw["complete"] is False and raw["validated"] is False


@pytest.mark.parametrize("opt_in", [None, False, 1])
def test_callable_parent_rejects_missing_or_boolean_alias_opt_in_before_work(opt_in):
    with pytest.raises(PermissionError):
        driver.parent(SimpleNamespace(execute=opt_in))


def test_file_read_rejects_symlink(tmp_path):
    target = tmp_path / "tiny"
    target.write_bytes(b"abc")
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        driver.bounded_regular_read(alias, 8)


def test_atomic_replacement_reserves_both_byte_copies(tmp_path):
    store = driver.FlatStore(tmp_path)
    try:
        store.byte_cap = 10
        store.put("record.json", b"123456")
        with pytest.raises(ValueError):
            store.put("record.json", b"abcdef")
        assert (tmp_path / "record.json").read_bytes() == b"123456"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"]
    finally:
        store.close()


def test_exclusive_startup_record_cannot_be_reused(tmp_path):
    store = driver.FlatStore(tmp_path)
    try:
        store.put("started.json", b"{}", exclusive=True)
        with pytest.raises((OSError, ValueError)):
            store.put("started.json", b"{}", exclusive=True)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["started.json"]
    finally:
        store.close()


def test_owned_directory_alias_change_is_fail_closed(tmp_path):
    output = tmp_path / "output"
    other = tmp_path / "other"
    output.mkdir()
    other.mkdir()
    store = driver.FlatStore(output)
    try:
        output.rename(tmp_path / "renamed")
        output.symlink_to(other, target_is_directory=True)
        with pytest.raises(ValueError):
            store.put("record.json", b"{}")
        assert not list(other.iterdir())
    finally:
        store.close()


def test_mirrored_input_root_does_not_unprotect_real_source(tmp_path, monkeypatch):
    actual = tmp_path / "actual"
    mirror = tmp_path / "mirror"
    (actual / "assets").mkdir(parents=True)
    mirror.mkdir()
    monkeypatch.setattr(driver, "ROOT", actual)
    with pytest.raises(ValueError):
        driver.fresh_output(mirror, actual / "assets" / "new-output")
    assert not (actual / "assets" / "new-output").exists()
