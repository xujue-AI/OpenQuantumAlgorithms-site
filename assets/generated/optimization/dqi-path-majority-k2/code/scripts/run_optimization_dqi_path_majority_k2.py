# SPDX-License-Identifier: MIT
"""Source-only future driver for one fixed full k=2 primitive check.

This file has not been executed. Loading it is not authorization to construct
source, import NumPy, create vectors, or run a circuit. A future invocation must
have an explicit external authorization and --execute, exact source/input/
environment identities, isolated site-disabled startup, and a fresh private
output. No legacy source, accepted record, test, notebook or public asset is
modified. This is one ordinary driver; the enclosing caller owns the complete
process 360-second ceiling and 256-KiB combined EOF capture.

NumPy import can execute its own native array/dot sanity work. That work is
startup, charged to CPU/wall/sampled RSS separately from 2,035 circuit gates.
Selected on-disk admission is not a portable loaded-library closure: macOS
dyld-cache/framework images are unobserved platform dependencies. No runtime
readiness, measured performance, synthesis, physical estimate or proof follows
from this unexecuted source. Failures are spent; no retry or backend fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import sys
import threading
import time
from types import ModuleType


ENTRY_TIME = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/openquantum/optimization_dqi_path_majority_k2.py"
VERIFIER = ROOT / "src/openquantum/optimization_dqi_path_majority_k2_verify.py"
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
FIXED_INPUT = {"k": 2, "signs": [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]}
EXPECTED_COUNTS = {"h": 253, "cx": 834, "t": 488, "tdg": 366, "x": 54,
                   "ry": 37, "z": 3, "reset": 22, "measure": 9, "barrier": 7}
SOURCE_CAP = 1 << 20
INVENTORY_CAP = 1 << 20
ENV_FILE_CAP = 64 << 20
ENV_TOTAL_CAP = 256 << 20
ENV_FILE_COUNT = 2000
OUTPUT_CAP = 512 << 20
OUTPUT_FILES = 64
BUFFER_CAP = 256 << 20
RSS_CAP = 1 << 30
CPU_CAP = 300
WALL_CAP = 360.0
CLEANUP_RESERVE = 5.0
DISK_FREE_START = 2 << 30
DISK_RESERVE = 1 << 30
CAPTURE_CAP = 256 << 10
VECTOR_BYTES = 64 << 20
VECTOR_NAMES = {"initial-state.bin", "preparation-reference.bin", "preparation-actual.bin",
                "final-reference.bin", "final-actual.bin"}


class DriverError(ValueError):
    """A fixed source/admission/process/output condition was rejected."""


def require(condition, message):
    if not condition:
        raise DriverError(message)


def digest(value, label="SHA-256"):
    require(type(value) is str and SHA.fullmatch(value), label + " must be lowercase SHA-256")
    return value


def integer(value, lower, upper, label):
    require(type(value) is int and lower <= value <= upper, label + " has an invalid integer bound")
    return value


def finite_seconds(value, maximum, label):
    require(type(value) in (int, float) and math.isfinite(value)
            and CLEANUP_RESERVE < value <= maximum, label + " is outside its hard ceiling")
    return float(value)


def json_value(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    def constant(_value):
        raise DriverError("Nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def regular_directory(path):
    path = Path(path)
    require(path.is_absolute(), "An absolute ordinary directory is required")
    for item in (path, *path.parents):
        require(stat.S_ISDIR(item.lstat().st_mode), "A directory path contains a symlink or non-directory")
    return path


def regular_pin(path, cap, check=None, expected=None):
    """Bounded read/hash; no target import and no whole-file native allocation."""
    path = Path(path)
    require(path.is_absolute(), "An absolute regular file is required")
    regular_directory(path.parent)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= cap,
            "A regular file exceeds its byte ceiling")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    hasher, total = hashlib.sha256(), 0
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode), "Opened file is not regular")
        while True:
            if check:
                check()
            chunk = os.read(fd, min(65536, cap + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            require(total <= cap, "A file exceeded its read ceiling")
            hasher.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                              value.st_size, value.st_mtime_ns)
    require(identity(before) == identity(opened) == identity(after) and total == after.st_size,
            "File identity changed during read")
    result = {"bytes": total, "sha256": hasher.hexdigest()}
    if expected is not None:
        require(result == expected, "A file differs from its exact pin")
    return result


def read_pinned(path, cap, expected_sha, check=None):
    digest(expected_sha)
    path = Path(path)
    require(path.is_absolute(), "An absolute regular file is required")
    regular_directory(path.parent)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= cap,
            "A pinned file exceeds its byte ceiling")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    chunks, hasher, total = [], hashlib.sha256(), 0
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode), "Opened pinned file is not regular")
        while True:
            if check:
                check()
            chunk = os.read(fd, min(65536, cap + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            require(total <= cap, "Pinned read exceeded its byte ceiling")
            chunks.append(chunk)
            hasher.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode,
                              value.st_size, value.st_mtime_ns)
    require(identity(before) == identity(opened) == identity(after) and total == after.st_size,
            "Pinned file identity changed during read")
    pin = {"bytes": total, "sha256": hasher.hexdigest()}
    require(pin["sha256"] == expected_sha, "A source/input/inventory digest differs")
    return b"".join(chunks), pin


def check_pin_shape(value):
    require(type(value) is dict and set(value) == {"bytes", "sha256"}, "Invalid environment pin fields")
    integer(value["bytes"], 0, ENV_FILE_CAP, "Environment file bytes")
    digest(value["sha256"])


def tree_members(directory, check, *, stdlib_directory):
    directory = regular_directory(directory)
    members, stack, visited = [], [(directory, 0)], 0
    while stack:
        current, depth = stack.pop()
        require(depth <= 12, "Package tree depth exceeds the admission bound")
        with os.scandir(current) as entries:
            for entry in entries:
                check()
                visited += 1
                require(visited <= 10000, "Package tree inventory exceeds the admission bound")
                if entry.name == "__pycache__":
                    require(entry.is_dir(follow_symlinks=False), "Unexpected cache path")
                    continue
                require(not entry.is_symlink(), "Package tree contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    if directory == stdlib_directory and Path(entry.path) == stdlib_directory / "site-packages":
                        continue
                    stack.append((Path(entry.path), depth + 1))
                else:
                    require(entry.is_file(follow_symlinks=False), "Package tree has a nonregular member")
                    members.append(Path(entry.path).relative_to(directory).as_posix())
    return sorted(members)


def startup_flags():
    require(sys.version_info[:3] == (3, 12, 14), "The locked Python 3.12.14 is required")
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode,
            "The command requires -I -S -B")
    require(type(sys.pycache_prefix) is str and Path(sys.pycache_prefix).is_absolute()
            and not os.path.lexists(sys.pycache_prefix), "A fresh absent -X pycache_prefix is required")
    require(not any(name == "numpy" or name.startswith("numpy.") or name == "psutil"
                    or name.startswith("psutil.") for name in sys.modules),
            "A target package was already loaded")


def admit_environment(value, check):
    require(type(value) is dict
            and value.get("schema") == "openquantum.dqi.path_majority_k2_environment_proposal.v1"
            and value.get("status") == "source_only_unexecuted"
            and value.get("execution_authorized") is False
            and value.get("runtime_verified") is False,
            "Environment source-only scope is invalid")
    require(value.get("python_version") == "3.12.14"
            and value.get("program") == sys.executable
            and value.get("resolved_program") == str(Path(sys.executable).resolve()),
            "Interpreter identity/path differs")
    chain = value.get("program_symlink_chain")
    require(type(chain) is list and 1 <= len(chain) <= 8, "Interpreter symlink chain is absent")
    for item in chain:
        require(type(item) is dict and set(item) == {"path", "target"}
                and type(item["path"]) is str and type(item["target"]) is str
                and stat.S_ISLNK(Path(item["path"]).lstat().st_mode)
                and os.readlink(item["path"]) == item["target"], "Interpreter symlink chain differs")
    files = value.get("environment_files")
    require(type(files) is dict and 1 <= len(files) <= ENV_FILE_COUNT, "Selected environment count differs")
    total = 0
    for name, pin in files.items():
        check_pin_shape(pin)
        total += pin["bytes"]
        require(total <= ENV_TOTAL_CAP, "Selected environment byte ceiling exceeded")
        regular_pin(name, ENV_FILE_CAP, check, pin)
    require(value.get("selection_summary", {}).get("selected_regular_files") == len(files)
            and value.get("selection_summary", {}).get("selected_total_bytes") == total,
            "Selected environment counters differ")
    inventories = value.get("package_tree_inventories")
    require(type(inventories) is list and 1 <= len(inventories) <= 8, "Package inventories are absent")
    for tree in inventories:
        require(type(tree) is dict and tree.get("snapshot_members_closed") is True,
                "Package inventory is not closed")
        require(tree.get("ignored_component") == "__pycache__", "Package cache exclusion differs")
        names = tree_members(tree["directory"], check,
                             stdlib_directory=Path(value["resolved_program"]).parents[1] / "lib/python3.12")
        require(not any(name.endswith((".pyc", ".pyo")) for name in names),
                "Noncache sourceless bytecode is not admitted")
        require(names == tree["noncache_members"] and len(names) == tree["member_count"],
                "Package noncache membership differs")
        list_digest = hashlib.sha256((json.dumps(names, separators=(",", ":")) + "\n").encode()).hexdigest()
        require(list_digest == tree["member_list_sha256"], "Package membership hash differs")
        selected = tree["selected_members"]
        require(type(selected) is list and len(selected) == tree["selected_member_count"]
                and len(set(selected)) == len(selected) and set(selected) <= set(names),
                "Selected package members differ")
        require(all(str(Path(tree["directory"]) / name) in files for name in selected),
                "A selected package member lacks a payload pin")
    policy = value.get("startup_policy")
    require(type(policy) is dict and policy.get("pth_processing_permitted") is False
            and policy.get("site_addsitedir_permitted") is False
            and policy.get("fresh_pycache_prefix_required") is True,
            "Unsafe site/cache policy")
    for key in policy["environment_overrides_to_remove"]:
        require(key not in os.environ, "An excluded startup environment override is present")
    for key, expected in policy["environment_to_set"].items():
        require(os.environ.get(key) == expected, "A fixed startup/thread environment value differs")
    prefixes = policy.get("environment_prefixes_to_remove")
    require(prefixes == ["PYTHON", "PYTEST", "DYLD_", "LD_", "NPY_", "NUMPY_"],
            "Startup environment-prefix exclusions differ")
    for key in os.environ:
        if any(key.startswith(prefix) for prefix in prefixes):
            require(key in policy["environment_to_set"]
                    and os.environ[key] == policy["environment_to_set"][key],
                    "An inherited startup override is not admitted")
    sites = policy["explicit_site_packages"]
    require(type(sites) is list and len(sites) == 1, "Exactly one admitted site directory is required")
    site = str(regular_directory(sites[0]))
    require(site == str(Path(value["program"]).parents[1] / "lib/python3.12/site-packages"),
            "The admitted site is outside the locked virtual environment")
    stdlib = str(Path(value["resolved_program"]).parents[1] / "lib/python3.12")
    paths = [stdlib, str(Path(stdlib) / "lib-dynload"), site]
    require(policy.get("post_admission_sys_path") == paths, "Post-admission module path differs")
    for path in paths:
        regular_directory(path)
    absent = policy.get("required_absent_paths")
    expected_absent = {str(Path(value["resolved_program"]).parents[1] / "lib/python312.zip"),
                       str(Path(stdlib) / "sitecustomize.py"), str(Path(stdlib) / "usercustomize.py"),
                       str(Path(site) / "sitecustomize.py"), str(Path(site) / "usercustomize.py"),
                       str(Path(site) / "numpy/_distributor_init_local.py")}
    require(type(absent) is list and len(absent) == len(expected_absent)
            and set(absent) == expected_absent, "Absent startup paths are not recorded exactly")
    for path in absent:
        require(type(path) is str and Path(path).is_absolute() and not os.path.lexists(path),
                "A prohibited startup path now exists")
    packages = value.get("packages", {})
    require(packages.get("numpy", {}).get("version") == "2.5.4"
            and packages.get("psutil", {}).get("version") == "7.2.2", "Package versions differ")
    require(value.get("observed_loaded_libraries") == [], "Static proposal must not invent loaded libraries")
    sys.path[:] = paths
    return {"selected_files": len(files), "selected_bytes": total, "package_trees": len(inventories),
            "post_admission_sys_path": paths, "runtime_verified_before_import": False,
            "portable_native_closure": False, "platform_scope_limits": value["scope_limits"]}


def load_source(name, path, raw, check):
    """Only the admitted, future --execute branch calls this target loader."""
    check()
    require(name not in sys.modules, "Target source was already loaded")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    check()
    return module


class ProcessGuard:
    def __init__(self, output, wall, cpu, rss):
        self.output = output
        self.deadline = ENTRY_TIME + wall - CLEANUP_RESERVE
        self.cpu_limit, self.rss_limit = cpu, rss
        self.process = None
        self.samples, self.max_rss = 0, 0
        self.failure = None
        self.stop_event = threading.Event()
        self.thread = None
        self.last_disk_check = 0.0
        self.sample_lock = threading.Lock()
        self.last_rss_sample = 0.0
        self.last_cpu_check = 0.0
        self.installed_cpu_hard_limit = None

    def sample(self):
        require(self.process is not None, "RSS monitoring is unavailable")
        value = self.process.memory_info().rss
        require(type(value) is int and value >= 0, "Invalid RSS observation")
        with self.sample_lock:
            self.samples += 1
            self.max_rss = max(self.max_rss, value)
            self.last_rss_sample = time.monotonic()
        require(value <= self.rss_limit, "Sampled RSS ceiling reached")

    def start_monitor(self, psutil):
        require(getattr(psutil, "__version__", None) == "7.2.2", "Loaded monitor version differs")
        self.process = psutil.Process(os.getpid())
        self.sample()
        def watch():
            try:
                while not self.stop_event.wait(0.05):
                    self.sample()
            except BaseException as error:
                self.failure = type(error).__name__ + ": " + str(error)
        self.thread = threading.Thread(target=watch, name="k2-private-rss", daemon=True)
        self.thread.start()

    def check(self):
        now = time.monotonic()
        require(now < self.deadline, "Cooperative inner deadline reached")
        if now - self.last_cpu_check >= 0.05:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            require(usage.ru_utime + usage.ru_stime <= self.cpu_limit, "CPU ceiling reached")
            self.last_cpu_check = now
        require(self.failure is None, "RSS monitor failed: " + str(self.failure))
        if self.process is not None and now - self.last_rss_sample >= 0.05:
            self.sample()
        if now - self.last_disk_check >= 1.0:
            require(shutil.disk_usage(self.output).free >= DISK_RESERVE, "Private disk reserve reached")
            output_size_only(self.output)
            self.last_disk_check = now

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=0.2)
            require(not self.thread.is_alive(), "RSS monitor did not stop")
        require(self.failure is None, "RSS monitor ended with an error")

    def observation(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"cpu_seconds": usage.ru_utime + usage.ru_stime,
                "driver_elapsed_seconds": time.monotonic() - ENTRY_TIME,
                "rss_samples": self.samples, "maximum_sampled_rss_bytes": self.max_rss,
                "rss_requested_cadence_seconds": 0.05, "rss_failure": self.failure,
                "cpu_poll_requested_cadence_seconds": 0.05,
                "kernel_cpu_hard_limit_also_installed": self.installed_cpu_hard_limit is not None,
                "installed_cpu_hard_limit_seconds": self.installed_cpu_hard_limit,
                "sampled_rss_is_not_an_unsampled_peak": True}


def output_size_only(root):
    """Bounded sampled size/count guard; writers retain their own byte guards."""
    count, total, stack, visited = 0, 0, [(root, 0)], 0
    while stack:
        current, depth = stack.pop()
        require(depth <= 6, "Private output directory depth exceeded")
        with os.scandir(current) as entries:
            for entry in entries:
                visited += 1
                require(visited <= 128 and not entry.is_symlink(), "Private output inventory differs")
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
                else:
                    require(entry.is_file(follow_symlinks=False), "Private output must be regular")
                    count += 1
                    total += entry.stat(follow_symlinks=False).st_size
                    require(count <= OUTPUT_FILES and total <= OUTPUT_CAP, "Private output ceiling reached")


class Writer:
    def __init__(self, root, check):
        self.root, self.check = root, check
        self.source_bytes = 0

    def binary(self, relative, raw, *, source=False):
        self.check()
        require(type(raw) is bytes, "A bounded byte output is required")
        require(len(raw) <= SOURCE_CAP, "Driver metadata/source output exceeds its cap")
        if source:
            self.source_bytes += len(raw)
            require(self.source_bytes <= SOURCE_CAP, "Combined source export exceeds 1 MiB")
        path = self.root / relative
        regular_directory(path.parent)
        with path.open("xb") as stream:
            require(stream.write(raw) == len(raw), "Output byte write did not complete")
        self.check()
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def json(self, relative, value, *, source=False):
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        return self.binary(relative, raw, source=source)


def output_inventory(root, check):
    files, total, stack, visited = {}, 0, [(root, 0)], 0
    while stack:
        current, depth = stack.pop()
        require(depth <= 6, "Private output directory depth exceeded")
        with os.scandir(current) as entries:
            for entry in entries:
                check()
                visited += 1
                require(visited <= 128 and not entry.is_symlink(), "Private output inventory differs")
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
                else:
                    require(entry.is_file(follow_symlinks=False), "Private output must contain regular files")
                    require(len(files) < OUTPUT_FILES, "Private regular-file ceiling reached")
                    total += entry.stat(follow_symlinks=False).st_size
                    require(total <= OUTPUT_CAP, "Private output byte ceiling reached")
                    p = Path(entry.path)
                    files[p.relative_to(root).as_posix()] = regular_pin(p, OUTPUT_CAP, check)
    return files, total


def write_failure(output, value):
    """Small best-effort failure evidence; does not reconstruct success."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(raw) <= 64 << 10, "Failure metadata cap reached")
    with (output / "private/failure-receipt.json").open("xb") as stream:
        stream.write(raw)


def verifier_result(value, snapshot):
    require(type(value) is dict
            and value.get("schema") == "openquantum.dqi.path_majority_k2_full_validation.v1"
            and value.get("status") == "complete_ideal_full_program_check_pending_parent_acceptance"
            and value.get("full_program_ideal_diagnostics_passed") is True
            and value.get("full_program_validated") is False
            and value.get("parent_acceptance_recorded") is False,
            "The full verifier has not completed ideal diagnostics")
    for key in ("global_phase_fitted", "state_renormalized", "auxiliary_postselection",
                "sparse_truncation", "dense_operator_constructed", "sampling_performed",
                "physical_resources_estimated", "rotation_synthesis_validated",
                "formal_proof_reproduced", "hardware_executed", "runtime_readiness_claimed",
                "publication_complete"):
        require(value.get(key) is False, "An unsupported verifier claim is present")
    require(type(snapshot) is dict and snapshot.get("complete") is True
            and snapshot.get("active") is False and snapshot.get("spent") is True
            and value.get("execution_budget") == snapshot, "The once-only budget has not closed")
    exact = {"verification_invocations": 1, "complete_statevector_evolutions": 1,
             "primitive_unitary_applications": 2035, "source_records_consumed": 2073,
             "known_zero_resets": 22, "barriers": 7, "readout_mappings": 9,
             "checkpoint_stages": 5, "checkpoint_full_state_scans": 5,
             "state_scan_chunks": 80, "state_scan_amplitudes": 20971520,
             "scalar_reference_angle_evaluations": 19, "degree_two_words_checked": 56,
             "promised_word_direct_decoder_mappings": 56, "independent_assignments_scanned": 512,
             "independent_edge_scores": 5120, "independent_character_terms": 28672,
             "parity_sign_reads": 10, "parity_assignment_internal_steps": 8,
             "parity_witness_edge_scores": 10, "reference_buffer_zeroing_passes": 5,
             "reference_entries_written": 736, "vector_files_completed": 5, "output_files_created": 7}
    require(all(type(snapshot.get(key)) is int and snapshot[key] == expected
                for key, expected in exact.items()), "The completed role counters differ")
    for key in ("numpy_kernel_calls", "numpy_kernel_elements", "numpy_reduction_calls",
                "numpy_reduction_input_elements", "primitive_pair_chunks", "primitive_pairs_addressed"):
        require(type(snapshot.get(key)) is int and snapshot[key] > 0, "Actual kernel/reduction work is absent")
    require(type(snapshot.get("output_bytes_written")) is int
            and snapshot["output_bytes_written"] <= OUTPUT_CAP, "Verifier output bytes exceed their ceiling")
    pins = value.get("file_pins")
    require(type(pins) is dict and set(pins) == VECTOR_NAMES | {"output-table.json", "diagnostics.json"},
            "The exact seven verifier outputs differ")
    for name, pin in pins.items():
        check_pin_shape(pin)
        if name in VECTOR_NAMES:
            require(pin["bytes"] == VECTOR_BYTES, "A required full vector shape differs")
        else:
            require(0 < pin["bytes"] <= SOURCE_CAP, "Verifier metadata output exceeds its ceiling")
    buffer = value.get("buffer_accounting")
    require(type(buffer) is dict and type(buffer.get("charged_peak_bytes")) is int
            and buffer["charged_peak_bytes"] <= BUFFER_CAP
            and buffer.get("limit_bytes") == BUFFER_CAP
            and buffer.get("whole_process_rss_is_separate") is True,
            "The declared buffer accounting does not close")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--input-sha256", required=True)
    result.add_argument("--environment-inventory", type=Path, required=True)
    result.add_argument("--environment-sha256", required=True)
    result.add_argument("--recipe", type=Path, required=True)
    result.add_argument("--recipe-sha256", required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--driver-sha256", required=True)
    result.add_argument("--profile-sha256", required=True)
    result.add_argument("--verifier-sha256", required=True)
    result.add_argument("--wall-limit-seconds", type=float, default=WALL_CAP)
    result.add_argument("--cpu-limit-seconds", type=int, default=CPU_CAP)
    result.add_argument("--rss-limit-bytes", type=int, default=RSS_CAP)
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.execute is not True:
        command.error("External authorization and explicit --execute are required before source/input access")
    startup_flags()
    for name in ("input_sha256", "environment_sha256", "recipe_sha256", "driver_sha256",
                 "profile_sha256", "verifier_sha256"):
        digest(getattr(args, name), name)
    require(COMMIT.fullmatch(args.source_commit), "An exact externally admitted source commit is required")
    wall = finite_seconds(args.wall_limit_seconds, WALL_CAP, "Wall limit")
    cpu = integer(args.cpu_limit_seconds, 1, CPU_CAP, "CPU limit")
    rss = integer(args.rss_limit_bytes, 1, RSS_CAP, "Sampled RSS limit")
    require(args.output.is_absolute() and not os.path.lexists(args.output), "Fresh absent output is required")
    require(Path(sys.pycache_prefix).is_relative_to(args.output),
            "The fresh pycache prefix must lie inside the private output")
    regular_directory(args.output.parent)
    require(shutil.disk_usage(args.output.parent).free >= DISK_FREE_START, "At least 2 GiB free is required")
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    for name in ("source", "source/code", "verification", "private"):
        (args.output / name).mkdir(mode=0o700)
    claim = {"schema": "openquantum.dqi.path_majority_k2_execution_claim.v1",
             "spent": True, "complete": False, "retry_authorized": False,
             "argv": list(sys.orig_argv), "source_commit": args.source_commit,
             "source_commit_admission": "external caller; exact source bytes independently checked",
             "output": str(args.output), "wall_limit_seconds": wall, "cpu_limit_seconds": cpu,
             "sampled_rss_limit_bytes": rss, "buffer_limit_bytes": BUFFER_CAP,
             "combined_outer_capture_limit_bytes": CAPTURE_CAP,
             "outer_capture_and_complete_process_ceiling_owned_by_caller": True,
             "new_legacy_component_tests_or_notebook": False}
    Writer(args.output, lambda: None).json("private/execution-claim.json", claim)
    guard = ProcessGuard(args.output, wall, cpu, rss)
    previous_alarm = None
    writer = Writer(args.output, guard.check)
    budget = None
    sources, admission, imports = {}, None, None
    try:
        require(hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
                and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
                "The CLI requires a fresh POSIX scheduled alarm")
        previous_alarm = signal.getsignal(signal.SIGALRM)
        def timeout(_number, _frame):
            raise TimeoutError("Scheduled inner deadline reached")
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, guard.deadline - time.monotonic()))
        old_cpu_limits = resource.getrlimit(resource.RLIMIT_CPU)
        hard_cpu = cpu if old_cpu_limits[1] == resource.RLIM_INFINITY else min(cpu, old_cpu_limits[1])
        resource.setrlimit(resource.RLIMIT_CPU, (hard_cpu, hard_cpu))
        guard.installed_cpu_hard_limit = hard_cpu
        input_raw, input_pin = read_pinned(args.input, SOURCE_CAP, args.input_sha256, guard.check)
        input_value = json_value(input_raw)
        require(type(input_value) is dict and set(input_value) == {"k", "signs"}
                and type(input_value["k"]) is int and input_value["k"] == 2
                and type(input_value["signs"]) is list
                and all(type(x) is int for x in input_value["signs"])
                and input_value == FIXED_INPUT, "The fixed input differs")
        own_raw, own_pin = read_pinned(Path(__file__).resolve(), SOURCE_CAP, args.driver_sha256, guard.check)
        profile_raw, profile_pin = read_pinned(PROFILE, SOURCE_CAP, args.profile_sha256, guard.check)
        verifier_raw, verifier_pin = read_pinned(VERIFIER, SOURCE_CAP, args.verifier_sha256, guard.check)
        sources = {"driver": {"path": str(Path(__file__).resolve()), **own_pin},
                   "profile": {"path": str(PROFILE), **profile_pin},
                   "verifier": {"path": str(VERIFIER), **verifier_pin}}
        env_raw, env_pin = read_pinned(args.environment_inventory, INVENTORY_CAP,
                                     args.environment_sha256, guard.check)
        environment = json_value(env_raw)
        recipe_raw, recipe_pin = read_pinned(args.recipe, SOURCE_CAP, args.recipe_sha256, guard.check)
        recipe = json_value(recipe_raw)
        require(type(recipe) is dict
                and recipe.get("schema") == "openquantum.dqi.path_majority_k2_full_verification_recipe.v1"
                and recipe.get("executed") is False and recipe.get("execution_authorized") is False
                and recipe.get("dimensions", {}).get("Q") == 22
                and recipe.get("future_work_counters", {}).get("primitive_unitary_applications") == 2035,
                "A pinned fixed source-only recipe is required")
        preserved_names = {"PLAN.md", "algorithms/optimization/dqi-degree-two.md", "requirements-lock.txt",
                           "scripts/generate_optimization_dqi_path_majority.py",
                           "scripts/validate_optimization_dqi_path_majority.py",
                           "src/openquantum/optimization_dqi_path_majority.py",
                           "src/openquantum/optimization_dqi_path_majority_verify.py"}
        require(type(recipe.get("source_pins")) is dict and set(recipe["source_pins"]) == preserved_names,
                "Recipe protected-source identities differ")
        for name, expected in recipe["source_pins"].items():
            check_pin_shape(expected)
            regular_pin(ROOT / name, ENV_FILE_CAP, guard.check, expected)
        admission = admit_environment(environment, guard.check)
        import_start = time.monotonic()
        import psutil
        require(str(Path(psutil.__file__).resolve())
                == str(Path(environment["packages"]["psutil"]["directory"]) / "__init__.py"),
                "Monitor package loaded from an unadmitted source")
        guard.start_monitor(psutil)
        psutil_ready = time.monotonic()
        import numpy
        require(numpy.__version__ == "2.5.4", "Loaded NumPy version differs")
        require(str(Path(numpy.__file__).resolve())
                == str(Path(environment["packages"]["numpy"]["directory"]) / "__init__.py"),
                "NumPy loaded from an unadmitted source")
        guard.check()
        numpy_ready = time.monotonic()
        imports = {"psutil_import_seconds": psutil_ready - import_start,
                   "numpy_import_seconds": numpy_ready - psutil_ready,
                   "numpy_native_sanity_work_is_startup": True,
                   "numpy_native_sanity_work_count_observed": False,
                   "startup_work_included_in_process_cpu_wall_sampled_rss": True,
                   "psutil_import_before_first_psutil_rss_sample": True,
                   "startup_work_is_not_circuit_primitive_applications": True,
                   "observed_loaded_os_images": None, "portable_native_closure": False}
        profile = load_source("_openquantum_k2_profile", PROFILE, profile_raw, guard.check)
        verifier = load_source("_openquantum_k2_verifier", VERIFIER, verifier_raw, guard.check)
        limits = profile.SourceLimits(api_wall_seconds=min(30.0, guard.deadline - time.monotonic()))
        plan = profile.make_plan(input_value, limits=limits)
        records, record_bytes, observed = [], 0, {key: 0 for key in EXPECTED_COUNTS}
        for operation in profile.iter_operations(plan, deadline=guard.deadline):
            guard.check()
            record = operation.record()
            require(len(records) < 2073 and record.get("gate") in observed, "Source event ceiling reached")
            encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            record_bytes += len(encoded)
            require(record_bytes <= SOURCE_CAP, "NDJSON source ceiling reached")
            records.append(record)
            observed[record["gate"]] += 1
        require(len(records) == 2073 and observed == EXPECTED_COUNTS, "Complete emitted source counts differ")
        ndjson = b"".join((json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
                           + "\n").encode() for record in records)
        require(len(ndjson) == record_bytes, "Retained source serialization differs")
        qasm_chunks, qasm_bytes = [], 0
        for chunk in profile.iter_qasm_records(records, deadline=guard.deadline, max_export_bytes=SOURCE_CAP):
            guard.check()
            require(type(chunk) is bytes, "QASM serializer yielded nonbytes")
            qasm_bytes += len(chunk)
            require(qasm_bytes <= SOURCE_CAP, "QASM source ceiling reached")
            qasm_chunks.append(chunk)
        qasm = b"".join(qasm_chunks)
        ledger = profile.plan_summary(plan)
        writer.binary("source/program.ndjson", ndjson, source=True)
        writer.binary("source/program.qasm", qasm, source=True)
        writer.json("source/program-ledger.json", ledger, source=True)
        writer.binary("source/input-k2.json", input_raw, source=True)
        writer.binary("source/recipe.json", recipe_raw, source=True)
        writer.binary("source/environment-inventory.json", env_raw, source=True)
        writer.binary("source/code/driver.py", own_raw, source=True)
        writer.binary("source/code/profile.py", profile_raw, source=True)
        writer.binary("source/code/verifier.py", verifier_raw, source=True)
        source_export = {"source_construction_invocations": 1, "retained_record_list": 1,
                         "record_count": len(records), "unitary_primitives": 2035,
                         "operational_events_excluding_barriers": 2066,
                         "counts": observed, "ndjson_bytes": len(ndjson), "qasm_bytes": len(qasm),
                         "qasm_serializes_same_records_without_second_emission": True,
                         "combined_driver_source_export_bytes": writer.source_bytes}
        budget = verifier.ExecutionBudget(True)
        result = verifier.verify_full_program_records(records, input_value, args.output / "verification",
            numpy_module=numpy, allow_numerical_execution=True, check_callback=guard.check, budget=budget)
        guard.check()
        require(type(result) is dict, "Full verifier result must be typed metadata")
        verifier_result(result, budget.snapshot())
        for source in sources.values():
            regular_pin(source["path"], SOURCE_CAP, guard.check,
                        {"bytes": source["bytes"], "sha256": source["sha256"]})
        regular_pin(args.input, SOURCE_CAP, guard.check, input_pin)
        regular_pin(args.environment_inventory, INVENTORY_CAP, guard.check, env_pin)
        regular_pin(args.recipe, SOURCE_CAP, guard.check, recipe_pin)
        for name, expected in environment["environment_files"].items():
            regular_pin(name, ENV_FILE_CAP, guard.check, expected)
        for name, expected in recipe["source_pins"].items():
            regular_pin(ROOT / name, ENV_FILE_CAP, guard.check, expected)
        guard.stop()
        files, total = output_inventory(args.output, guard.check)
        vectors = ["verification/" + name for name in sorted(VECTOR_NAMES)]
        require(all(files.get(name) == result["file_pins"][name.split("/", 1)[1]] for name in vectors)
                and all(files.get("verification/" + name) == expected
                        for name, expected in result["file_pins"].items()),
                "Verifier outputs differ from their actual complete pins")
        receipt = {"schema": "openquantum.dqi.path_majority_k2_driver_receipt.v1",
            "status": "verifier_phase_complete_pending_output_closure_and_parent_acceptance",
            "intended_return_code_if_final_closure_completes": 0,
            "actual_process_exit_observed_by_driver": False,
            "argv": list(sys.orig_argv), "source_commit": args.source_commit,
            "source_commit_admission": "external caller; exact source bytes checked",
            "sources": sources, "input_pin": input_pin, "environment_pin": env_pin, "recipe_pin": recipe_pin,
            "environment_admission": admission, "target_import_startup": imports,
            "source_export": source_export, "verification": result, "budget": budget.snapshot(),
            "process_observation": guard.observation(), "files_before_receipt_and_manifest": files,
            "process_observation_scope": "through pre-receipt closure; final stdout observation and outer caller complete it",
            "output_bytes_before_receipt_and_manifest": total,
            "captured_stdout_stderr": "owned by enclosing caller; not synthesized by driver",
            "combined_outer_capture_limit_bytes": CAPTURE_CAP,
            "capture_hashes_and_eof_observed_by_driver": False,
            "complete_process_wall_limit_seconds": wall, "scheduled_inner_cleanup_reserve_seconds": CLEANUP_RESERVE,
            "numerical_buffer_limit_bytes": BUFFER_CAP, "sampled_rss_limit_bytes": rss,
            "cpu_limit_seconds": cpu, "limits_are_abort_ceilings_not_readiness_or_completion": True,
            "new_legacy_component_test_invocations": 0, "kernel_invocations": 0,
            "gpu_or_remote_work": False, "sampling_performed": False,
            "native_estimator_invocations": 0, "rotation_synthesis_validated": False,
            "formal_proof_reproduced": False, "physical_resources_estimated": False,
            "publication_complete": False, "parent_acceptance": False,
            "retry_authorized": False, "runtime_startup_shutdown_portability_certified": False}
        writer.json("private/driver-receipt.json", receipt)
        files, total = output_inventory(args.output, guard.check)
        manifest = {"schema": "openquantum.dqi.path_majority_k2_run_manifest.v1",
                    "status": "full_k2_manifest_written_pending_final_closure_and_parent_acceptance",
                    "files": files, "file_count_excluding_manifest": len(files),
                    "bytes_excluding_manifest": total, "sources": sources,
                    "vector_files": sorted(vectors), "source_commit": args.source_commit,
                    "parent_acceptance": False, "publication_complete": False}
        manifest_pin = writer.json("manifest.json", manifest)
        final_files, final_total = output_inventory(args.output, guard.check)
        require(final_files["manifest.json"] == manifest_pin
                and {name: value for name, value in final_files.items() if name != "manifest.json"} == files,
                "Final closed output inventory differs")
        require(final_total <= OUTPUT_CAP and len(final_files) <= OUTPUT_FILES, "Final output ceiling differs")
        line = (json.dumps({"status": "complete_full_k2_check_pending_parent_acceptance", "manifest": manifest_pin,
                           "output_files": len(final_files), "output_bytes": final_total,
                           "final_driver_observation_before_return": guard.observation(),
                           "parent_acceptance": False, "publication_complete": False}, sort_keys=True) + "\n").encode()
        require(len(line) <= 4096, "Driver stdout summary exceeds 4 KiB")
        offset = 0
        while offset < len(line):
            guard.check()
            count = os.write(1, line[offset:])
            require(count > 0, "Driver stdout did not advance")
            offset += count
        return 0
    except BaseException as error:
        try:
            guard.stop()
        except BaseException:
            pass
        failure = {"schema": "openquantum.dqi.path_majority_k2_driver_failure.v1",
                   "status": "failed_spent_full_k2_invocation", "intended_return_code": 1,
                   "actual_process_exit_observed_by_driver": False,
                   "argv": list(sys.orig_argv), "error": type(error).__name__,
                   "message": str(error)[:1000], "sources_admitted": sources,
                   "environment_admission": admission, "target_import_startup": imports,
                   "budget": None if budget is None else budget.snapshot(),
                   "process_observation": guard.observation(), "complete": False,
                   "retry_authorized": False, "parent_acceptance": False, "publication_complete": False,
                   "outer_caller_owns_actual_eof_capture_exit_and_cleanup_observation": True}
        try:
            write_failure(args.output, failure)
        except BaseException:
            pass
        line = (json.dumps({"status": failure["status"], "error": type(error).__name__}, sort_keys=True) + "\n").encode()
        try:
            os.write(1, line)
        except OSError:
            pass
        return 1
    finally:
        if previous_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
        # Hard CPU limit cannot be restored above the installed limit. No retry is attempted.


if __name__ == "__main__":
    raise SystemExit(main())
