"""Saved-evidence registration boundaries using captured AST and inert bytes.

No scientific module, accepted reader, generator, numerical backend, Lean,
QASM parser, notebook, full package verifier or subprocess runs in these tests.
Only synthetic helper definitions may be compiled to test authenticated dispatch.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import unquote, urlsplit
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("thermal-two-round", "shor-finite-adder", "dqi-path-majority", "rpe-resource-source")
BASES = {
    "thermal-two-round": "assets/generated/simulation/detailed-balance-two-round",
    "shor-finite-adder": "assets/generated/shor/finite-adder-reference",
    "dqi-path-majority": "assets/generated/optimization/dqi-path-majority",
    "rpe-resource-source": "assets/generated/robust-qpe-resource-source",
}
PUBLIC_SYMBOLS = {
    "thermal-two-round": "PUBLIC_TWO_ROUND_FILES",
    "shor-finite-adder": "PUBLIC_SHOR_FINITE_ADDER_FILES",
    "dqi-path-majority": "PUBLIC_DQI_PATH_MAJORITY_FILES",
    "rpe-resource-source": "PUBLIC_RPE_RESOURCE_SOURCE_FILES",
}
ASSET_NAMES = {
    "thermal-two-round": frozenset({
        "LICENSE.txt", "README.md", "UPSTREAM-LICENSES.md", "requirements-lock.txt",
        "manifest.json", "manifest.sha256", "results.json", "summary.json", "two-round-circuit.svg",
        "circuits/correction-heralded-step.qasm", "circuits/correction_lcu.qasm",
        "circuits/round_macro.qasm", "circuits/two_round_trial.qasm",
        "circuits/weak_dilation.qasm", "circuits/weak_reset_prefix.qasm",
        "sources/detailed_balance_access.py", "sources/detailed_balance_channel.py",
        "sources/detailed_balance_correction.py", "sources/detailed_balance_decay.py",
        "sources/detailed_balance_thermal_two_round.py",
        "sources/detailed_balance_thermal_two_round_reference.py", "sources/thermal_two_round_contract.py",
    }),
    "shor-finite-adder": frozenset({
        "LICENSE.txt", "README.md", "RippleTwoBitBool.lean", "axiom-audit.txt",
        "compilation.json", "lean-toolchain", "manifest.json", "ripple-two-bit-source.qasm",
    }),
    "dqi-path-majority": frozenset({
        "README.md", "accepted-components.json", "companion-spec.json", "component-circuit.svg",
        "component-input-recipe.json", "environment.json", "input-k3.json", "public-manifest.json",
        "results-summary.json", "generated/component-ledger.json", "generated/component.ndjson",
        "generated/component.qasm", "generated/manifest.json", "generated/program-ledger.json",
        "generated/program.ndjson", "generated/program.qasm", "validation/manifest.json",
        "validation/validation.json", "code/scripts/generate_optimization_dqi_path_majority.py",
        "code/scripts/validate_optimization_dqi_path_majority.py",
        "code/src/openquantum/optimization_dqi_path_majority.py",
        "code/src/openquantum/optimization_dqi_path_majority_verify.py",
        "code/tests/test_optimization_dqi_path_majority.py",
        "code/tests/test_optimization_dqi_path_majority_boundaries.py",
    }),
    "rpe-resource-source": frozenset({
        "LICENSE.txt", "README.md", "references.bib", "projection-manifest.json",
        "source-export-descriptor.json", "source-export-download.zip", "source-export-record.json",
        "code/LICENSE", "code/pyproject.toml", "code/requirements-lock.txt",
        "code/scripts/generate_robust_qpe_resources.py", "code/scripts/verify_robust_qpe_resource_source_saved.py",
        "code/src/openquantum/__init__.py", "code/src/openquantum/grover.py",
        "code/src/openquantum/hadamard_qpe.py", "code/src/openquantum/instrument_resources.py",
        "code/src/openquantum/robust_qpe.py", "code/src/openquantum/robust_qpe_resources.py",
        "code/tests/test_robust_qpe_resources.py", "code/tests/test_robust_qpe_resources_producer.py",
        "evidence/initial-producer.py", "evidence/initial-source-manifest.json",
        "evidence/producer-README.md", "evidence/repaired-source-manifest.json",
        "source/source-macrocycle.qasm", "source/source-ledger.json",
        "source/source-fringe-0.qasm", "source/source-fringe-1.qasm", "source/source-fringe-2.qasm",
        "source/source-fringe-3.qasm", "source/source-fringe-4.qasm", "source/source-fringe-5.qasm",
        "source/source-fringe-6.qasm", "source/source-fringe-7.qasm", "source/source-fringe-8.qasm",
        "source/source-fringe-9.qasm", "source/source-fringe-10.qasm", "source/source-fringe-11.qasm",
    }),
}
PUBLIC_FILES = {kind: frozenset(BASES[kind] + "/" + name for name in names)
                for kind, names in ASSET_NAMES.items()}
HELPERS = {
    "thermal-two-round": "scripts/verify_thermal_two_round_public.py",
    "shor-finite-adder": "scripts/verify_shor_finite_adder_reference.py",
    "dqi-path-majority": "scripts/display_dqi_path_majority_saved.py",
    "rpe-resource-source": "scripts/verify_robust_qpe_resource_source_saved.py",
}
HELPER_PINS = {
    "thermal-two-round": "128a4b64d0de76137b1bbdb49aa95cfc7ecdda45b66f1d70af8abae66f3d31c9",
    "shor-finite-adder": "0549789470c469e3b5e6ac86e40a1ba92cd2d19064afd07fe1f049b9a2a2f9fb",
    "dqi-path-majority": "1e82bad0820cac97efd7a4805585a1b70aa89413de7d4af117170d2e230b219a",
    "rpe-resource-source": "c5019ed950e7dfd445135036dbe49c873db6e67c9d1f63caa19c69cf658c8267",
}
THERMAL_DISPLAY = "scripts/display_thermal_two_round_saved.py"
THERMAL_DISPLAY_PIN = "0e2caf47a6225d9d7c8ab8b4e23f72ee6193dd1b6dd7cf0a521eae40d3922fbc"
MANIFEST_PINS = {
    "thermal-two-round": "e2f3c836fa05c6a0ae6be5daa18684a6b9ec96c6033f5bd34420a2413fe54377",
    "shor-finite-adder": "a1b76407144da82df6065dd666e85da106c3e404dbd3af25b05b6e81e78164eb",
    "dqi-path-majority": "b6f6ad7f3e4914a6af7c34a9c852652e623b9d22de07635378ad379e816363d9",
    "rpe-resource-source": "aaa9eebc661db8e2c8a41764eb8497fa66d810934d9dd1f4bc3b414805378e33",
}
SOURCE_NAMES = {
    "thermal-two-round": frozenset({THERMAL_DISPLAY, HELPERS["thermal-two-round"]}),
    "shor-finite-adder": frozenset({HELPERS["shor-finite-adder"]}),
    "dqi-path-majority": frozenset({
        HELPERS["dqi-path-majority"], "scripts/generate_optimization_dqi_path_majority.py",
        "scripts/validate_optimization_dqi_path_majority.py", "scripts/plot_dqi_path_majority_component_svg.py",
        "src/openquantum/optimization_dqi_path_majority.py",
        "src/openquantum/optimization_dqi_path_majority_verify.py",
        "tests/test_dqi_path_majority_public_projection.py", "tests/test_optimization_dqi_path_majority.py",
        "tests/test_optimization_dqi_path_majority_boundaries.py",
    }),
    "rpe-resource-source": frozenset({
        "LICENSE", "requirements-lock.txt", "scripts/generate_robust_qpe_resources.py",
        HELPERS["rpe-resource-source"], "src/openquantum/__init__.py", "src/openquantum/grover.py",
        "src/openquantum/hadamard_qpe.py", "src/openquantum/instrument_resources.py",
        "src/openquantum/robust_qpe.py", "src/openquantum/robust_qpe_resources.py",
        "tests/test_robust_qpe_resources.py", "tests/test_robust_qpe_resources_producer.py",
    }),
}
NEXT_CI_COMMANDS = (
    "python scripts/verify_shor_finite_adder_reference.py --bundle assets/generated/shor/finite-adder-reference",
    "python scripts/verify_thermal_two_round_public.py --bundle assets/generated/simulation/detailed-balance-two-round",
    "python -I -S -B scripts/display_dqi_path_majority_saved.py --display --artifact-root . --source-root . "
    "--manifest-sha256 b6f6ad7f3e4914a6af7c34a9c852652e623b9d22de07635378ad379e816363d9 --time-limit-seconds 5",
)
RPE_CI_COMMANDS = (
    "python scripts/verify_robust_qpe_resource_source_saved.py --bundle assets/generated/robust-qpe-resource-source "
    "--expected-manifest-sha256 aaa9eebc661db8e2c8a41764eb8497fa66d810934d9dd1f4bc3b414805378e33 "
    "--source-root . --max-seconds 30",
)
ARTICLE_LABELS = {
    "algorithms/simulation/detailed-balance-gibbs/index.html": frozenset({
        "gibbs-two-round-finite", "gibbs-two-round-circuit"}),
    "algorithms/shor/index.html": frozenset({"shor-finite-adder-formal-reference"}),
    "algorithms/optimization/dqi-degree-two/index.html": frozenset({"dqi-path-majority-component"}),
    "algorithms/qft-qpe/robust/index.html": frozenset({"robust-resource-source-export"}),
}
RPE_CONFIG_SNAPSHOT = BASES["rpe-resource-source"] + "/code/pyproject.toml"
RPE_ARCHIVE = BASES["rpe-resource-source"] + "/source-export-download.zip"
RPE_CONFIG_PIN = {"bytes": 705, "sha256": "fb96cdbcbb8c032581d62d8d04f261e988fced3e66a0f560ea79d7e9c2d72404"}
RPE_ARCHIVE_PIN = {"bytes": 81554, "sha256": "2e49377067a3c2a02a55f172cebcc192fc2159e7f6ac682d6893967f9b801706"}
BOOTSTRAP_PIN = {"bytes": 351, "sha256": "3be26040a631217836c04f479db98d08c5c64edbe2d9bb352e869b2e10e1bdda"}
RPE_SOURCE_COUNTS = {
    "actual_macrocycle_operations": 5286, "actual_unitary_operations": 5166,
    "reset_operations": 108, "readout_operations": 12, "source_files_checked": 12,
    "bundle_members_checked": 38, "zip_members_checked": 36,
    "producer_or_test_invocations": 0, "quantum_package_imports": 0,
}
OLD_GIBBS_LABELS = frozenset({
    "gibbs-channel-circuit", "gibbs-channel-checked-records", "gibbs-channel-output-contracts",
    "gibbs-decay-block", "gibbs-correction-step", "gibbs-channel-resource-projection", "gibbs-channel-saved-verification",
})
ORIGINAL_CI_COUNT = 86
ORIGINAL_CI_SHA256 = "6651b51693a743d2558c60e0aa5dd193c5961961794d5b72f1a7d77aee4f2fab"
FORBIDDEN_IMPORTS = frozenset({
    "openquantum", "numpy", "scipy", "qiskit", "qdk", "cirq", "torch", "jax", "cupy", "lean",
    "subprocess", "display_thermal_two_round_saved", "verify_thermal_two_round_public",
    "verify_shor_finite_adder_reference", "display_dqi_path_majority_saved",
    "verify_robust_qpe_resource_source_saved", "generate_robust_qpe_resources",
})


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SourceSnapshot:
    """Capture one source once and select definitions without sibling imports."""

    def __init__(self, path):
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        self.tree = ast.parse(self.raw, filename=str(self.path))

    def function(self, name):
        return next(node for node in self.tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    def module(self, name, **extra):
        selected = [node for node in self.tree.body if isinstance(
            node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
        module = ModuleType(name)
        module.__file__ = str(self.path)
        os_proxy = SimpleNamespace(
            open=os.open, close=os.close, fdopen=os.fdopen, fstat=os.fstat,
            read=os.read, stat=os.stat, lstat=os.lstat, scandir=os.scandir,
            O_RDONLY=os.O_RDONLY, O_DIRECTORY=getattr(os, "O_DIRECTORY", 0),
            O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 0), O_NONBLOCK=getattr(os, "O_NONBLOCK", 0),
            environ={},
        )
        module.__dict__.update(
            Path=Path, PurePosixPath=PurePosixPath, hashlib=SimpleNamespace(sha256=hashlib.sha256),
            json=json, io=io, os=os_proxy, stat=stat, math=math, HTMLParser=HTMLParser,
            importlib=SimpleNamespace(util=SimpleNamespace(
                spec_from_file_location=importlib.util.spec_from_file_location,
                module_from_spec=importlib.util.module_from_spec)),
            sys=SimpleNamespace(argv=[], executable=sys.executable), zipfile=zipfile,
            unquote=unquote, urlsplit=urlsplit, LINDBLAD_PORT_FILES=frozenset(),
            git_revision=lambda _: "a" * 40,
            subprocess=SimpleNamespace(run=lambda *_, **__: pytest.fail("Registration reached a subprocess")),
        )
        module.__dict__.update(extra)
        original_import = builtins.__import__
        def guarded(import_name, *args, **kwargs):
            if import_name.split(".", 1)[0] in FORBIDDEN_IMPORTS:
                raise AssertionError("Registration reached a scientific or accepted-reader import")
            return original_import(import_name, *args, **kwargs)
        safe_builtins = dict(vars(builtins), __import__=guarded)
        module.__dict__["__builtins__"] = safe_builtins
        exec(compile(tree, str(self.path), "exec"), module.__dict__)
        return module


@pytest.fixture(scope="module")
def snapshots():
    return {"exporter": SourceSnapshot(ROOT / "scripts/export_code.py"),
            "checked": SourceSnapshot(ROOT / "scripts/verify_public_package.py")}


@pytest.fixture(scope="module")
def exporter(snapshots):
    return snapshots["exporter"].module("next_registration_exporter")


@pytest.fixture
def checked(snapshots):
    return snapshots["checked"].module("next_registration_package")


@pytest.fixture(scope="module")
def site_snapshot():
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site tooling is outside the complete public example package")
    return SourceSnapshot(path)


@pytest.fixture
def checker(site_snapshot, checked, exporter):
    values = {name: value for name, value in vars(exporter).items() if name.startswith("PUBLIC_")}
    values.update(ALGORITHMS=[], REDIRECTS={}, _site_redirects=SimpleNamespace(REDIRECT_FRAGMENTS={}),
                  HAMILTONIAN_SOURCE_HASHES={}, HAMILTONIAN_PACKETS={},
                  NEXT_SAVED_PACKETS=checked.NEXT_SAVED_PACKETS,
                  checked_directory=checked.checked_directory, is_runtime_artifact=lambda _: False,
                  verify_preview=lambda _: {}, _read_bounded_public_member=checked._read_bounded_public_member,
                  _load_next_saved_gate=checked._load_next_saved_gate,
                  _verify_next_saved_identity=checked._verify_next_saved_identity)
    for tag in ("N2", "N3", "N4", "PORT", "QDRIFT"):
        values[tag + "_FILES"] = frozenset()
        values[tag + "_PATH"] = "unused-" + tag.lower()
    values.update(PORT_SOURCE_FILES={}, OPT7_PATH="unused-compiled",
                  ENERGY_CORE_FILES=frozenset(), ENERGY_MANIFEST_NAME="unused.json",
                  ENERGY_RECORD_DIRECTORY="unused-energy", ENERGY_SOURCE_FILES=(), ENERGY_VERIFIER_FILES=(),
                  DYNAMICS_CORE_FILES=frozenset(), DYNAMICS_MANIFEST_NAME="unused.json",
                  DYNAMICS_RECORD_DIRECTORY="unused-dynamics", DYNAMICS_SOURCE_FILES=(), DYNAMICS_VERIFIER_FILES=())
    for name in ("verify_n2_bundle", "verify_n3_bundle", "verify_n4_bundle", "verify_port_bundle",
                 "verify_qdrift_bundle", "verify_hadamard_record", "verify_opt7_bundle"):
        values[name] = lambda *_, **__: {}
    for name in ("site_energy_record_errors", "site_dynamics_record_errors"):
        values[name] = lambda *_, **__: []
    return site_snapshot.module("next_registration_site", **values)


SYNTHETIC_RESULTS = {
    "thermal-two-round": {"full_byte_authenticated": True, "scientific_work_reexecuted": False},
    "shor-finite-adder": {
        "status": "authenticated_recorded_finite_classical_reference", "members": 8,
        "helper_complete": False, "lean_executed_by_this_check": False,
        "qasm_parsed_by_this_check": False, "quantum_work_performed": False,
    },
    "dqi-path-majority": {
        "status": "displayed_authenticated_saved_component", "asset_files": 24, "source_files": 9,
        "scientific_work_reexecuted": False, "full_program_validated": False, "notebook_executed": False,
    },
    "rpe-resource-source": {
        "status": "verified_saved_source_export_bytes_recipe_and_outcome_scope",
        "source_only_invocation_succeeded": True, "prior_source_only_invocation_succeeded": False,
        "algorithm_numerically_validated": False, "physical_resource_estimate_computed": False,
        **RPE_SOURCE_COUNTS,
    },
}


def synthetic_helper():
    """Tiny helper source, deliberately unrelated to every accepted reader."""
    return ("TOKEN='synthetic next saved evidence'\nCALLS=[]\n"
            "def verify_full_bundle(bundle, **kwargs):\n"
            "    CALLS.append(('thermal-two-round',str(bundle),kwargs))\n"
            "    return " + repr(SYNTHETIC_RESULTS["thermal-two-round"]) + "\n"
            "def verify_reference(bundle):\n"
            "    CALLS.append(('shor-finite-adder',str(bundle),{}))\n"
            "    return " + repr(SYNTHETIC_RESULTS["shor-finite-adder"]) + "\n"
            "def display_saved_component(root, **kwargs):\n"
            "    CALLS.append(('dqi-path-majority',str(root),kwargs))\n"
            "    return " + repr(SYNTHETIC_RESULTS["dqi-path-majority"]) + "\n"
            "def verify_saved_source_export(bundle, manifest, **kwargs):\n"
            "    CALLS.append(('rpe-resource-source',str(bundle),dict(kwargs,manifest=manifest)))\n"
            "    return " + repr(SYNTHETIC_RESULTS["rpe-resource-source"]) + "\n").encode()


def write_files(root, files):
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def synthetic_case(tmp_path, checked, monkeypatch, kind, *, split=True):
    """Supply an explicit tiny trust root; retain real descriptor path shapes."""
    root = (tmp_path / "site").resolve()
    root.mkdir()
    source = root / "code" if split else root
    source.mkdir(exist_ok=True)
    packet = copy.deepcopy(checked.NEXT_SAVED_PACKETS[kind])
    assets = {name: ("inert asset " + name + "\n").encode() for name in PUBLIC_FILES[kind]}
    sources = {name: ("# inert retained source " + name + "\n").encode() for name in SOURCE_NAMES[kind]}
    sources[packet["helper"]] = synthetic_helper()
    packet["asset_hashes"] = {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in assets.items()}
    packet["source_hashes"] = {name: sha(raw) for name, raw in sources.items()}
    packet["helper_sha256"] = sha(sources[packet["helper"]])
    packet["manifest_sha256"] = packet["asset_hashes"][packet["manifest"]]["sha256"]
    monkeypatch.setitem(checked.NEXT_SAVED_PACKETS, kind, packet)
    write_files(root, assets)
    write_files(source, sources)
    hashes = {name: sha(raw) for name, raw in {**assets, **sources}.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    raw_provenance = (json.dumps(provenance, sort_keys=True) + "\n").encode()
    (source / "PROVENANCE.json").write_bytes(raw_provenance)
    omitted = {name: {"site_path": name, "sha256": sha(raw)} for name, raw in assets.items()}
    retained = {**packet["source_hashes"], "PROVENANCE.json": sha(raw_provenance)}
    preview = {"schema": "openquantum-source-preview-v1", "retained_sha256": retained,
               "omitted_assets": omitted}
    return SimpleNamespace(root=root, source=source, packet=packet, assets=assets, sources=sources,
                           provenance=provenance, omitted=omitted, preview=preview)


def names_loaded(node):
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


def calls_named(node, name):
    return [item for item in ast.walk(node) if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name) and item.func.id == name]


@pytest.mark.parametrize("kind", KINDS)
def test_each_public_family_is_exactly_closed(exporter, kind):
    expected = PUBLIC_FILES[kind]
    assert getattr(exporter, PUBLIC_SYMBOLS[kind]) == expected
    assert len(expected) == {"thermal-two-round": 22, "shor-finite-adder": 8,
                             "dqi-path-majority": 24, "rpe-resource-source": 38}[kind]
    assert all(exporter.include_generated_file(name) for name in expected)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("extra", (
    "extra.json", "extra.py", "extra.qasm", "extra.csv", "extra.svg", "extra.md",
    "extra.lean", "extra.txt", "extra.sha256", "extra.ndjson", "extra.log", "extra.zip", "lean-toolchain-extra",
    "private/receipt.json", "sources/private.py", "code/private/test_secret.py",
))
def test_extra_suffix_files_do_not_enter_closed_public_families(exporter, kind, extra):
    assert not exporter.include_generated_file(BASES[kind] + "/" + extra)


@pytest.mark.parametrize("kind", KINDS)
def test_each_frozen_saved_helper_is_exported_once(exporter, checked, kind):
    name = HELPERS[kind].removeprefix("scripts/")
    assert exporter.PUBLIC_SCRIPTS.count(name) == 1
    actual = sha((ROOT / HELPERS[kind]).read_bytes())
    assert actual == checked.NEXT_SAVED_PACKETS[kind]["helper_sha256"]
    assert actual == HELPER_PINS[kind]


def test_frozen_thermal_display_stays_separate_from_full_byte_gate(exporter, checked):
    assert exporter.PUBLIC_SCRIPTS.count(THERMAL_DISPLAY.removeprefix("scripts/")) == 1
    assert sha((ROOT / THERMAL_DISPLAY).read_bytes()) == THERMAL_DISPLAY_PIN
    packet = checked.NEXT_SAVED_PACKETS["thermal-two-round"]
    assert packet["helper"] == HELPERS["thermal-two-round"] and packet["helper"] != THERMAL_DISPLAY
    assert packet["source_hashes"][THERMAL_DISPLAY] == THERMAL_DISPLAY_PIN


@pytest.mark.parametrize("kind", KINDS)
def test_registered_paths_and_source_pins_are_closed(checked, kind):
    assert tuple(checked.NEXT_SAVED_PACKETS) == KINDS
    packet = checked.NEXT_SAVED_PACKETS[kind]
    expected_fields = {"helper", "helper_sha256", "prefix", "manifest", "manifest_sha256",
                       "asset_hashes", "source_hashes"}
    if kind == "rpe-resource-source":
        expected_fields.add("source_config_snapshot_only")
        assert packet["source_config_snapshot_only"] == RPE_CONFIG_SNAPSHOT
    assert set(packet) == expected_fields
    assert packet["helper"] == HELPERS[kind] and packet["prefix"] == BASES[kind]
    manifest_name = {"dqi-path-majority": "public-manifest.json", "rpe-resource-source": "projection-manifest.json"}.get(kind, "manifest.json")
    assert packet["manifest"] == BASES[kind] + "/" + manifest_name
    assert packet["manifest_sha256"] == MANIFEST_PINS[kind]
    assert set(packet["asset_hashes"]) == PUBLIC_FILES[kind]
    assert set(packet["source_hashes"]) == SOURCE_NAMES[kind]
    assert packet["source_hashes"][packet["helper"]] == packet["helper_sha256"]
    for name, pin in packet["asset_hashes"].items():
        raw = (ROOT / name).read_bytes()
        assert pin == {"bytes": len(raw), "sha256": sha(raw)}
    for name, pin in packet["source_hashes"].items():
        assert sha((ROOT / name).read_bytes()) == pin


@pytest.mark.parametrize("kind", KINDS)
def test_only_synthetic_authenticated_helpers_are_compiled(checked, tmp_path, monkeypatch, kind):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    gate = checked._load_next_saved_gate(case.source, case.provenance, kind)
    assert gate.TOKEN == "synthetic next saved evidence" and gate.CALLS == []
    assert gate.__name__ not in sys.modules


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", ("refreshed-helper", "helper-pin", "schema", "missing-source-join"))
def test_forged_provenance_fails_before_helper_compile(checked, tmp_path, monkeypatch, kind, mutation):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    if mutation == "refreshed-helper":
        path = case.source / case.packet["helper"]
        raw = path.read_bytes() + b"# forged helper\n"
        path.write_bytes(raw)
        case.provenance["file_sha256"][case.packet["helper"]] = sha(raw)
        case.provenance["source_sha256"][case.packet["helper"]] = sha(raw)
    elif mutation == "helper-pin":
        case.packet["helper_sha256"] = "0" * 64
    elif mutation == "schema":
        case.provenance["schema"] = "unaccepted-v1"
    else:
        del case.provenance["file_sha256"][case.packet["helper"]]
    compiled = []
    def forbidden_compile(*args, **kwargs):
        compiled.append(args)
        raise AssertionError("Forged helper reached compilation")
    monkeypatch.setattr(checked, "compile", forbidden_compile, raising=False)
    with pytest.raises(ValueError):
        checked._load_next_saved_gate(case.source, case.provenance, kind)
    assert compiled == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("split", (False, True), ids=("package", "served-preview"))
def test_identity_dispatch_reads_assets_and_sources_from_their_canonical_roots(checked, tmp_path, monkeypatch, kind, split):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind, split=split)
    gate = checked._load_next_saved_gate(case.source, case.provenance, kind)
    reads = []
    actual_read = checked._read_bounded_public_member
    def tracked_read(root, name, maximum):
        reads.append((Path(root), name))
        return actual_read(root, name, maximum)
    monkeypatch.setattr(checked, "_read_bounded_public_member", tracked_read)
    result = checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
        provenance=case.provenance, kind=kind, omitted_assets=case.omitted if split else None)
    assert result == SYNTHETIC_RESULTS[kind]
    assert reads == [(case.root, name) for name in case.packet["asset_hashes"]] + [
        (case.source, name) for name in case.packet["source_hashes"]]
    assert len(gate.CALLS) == 1
    call_kind, target, kwargs = gate.CALLS[0]
    assert call_kind == kind
    if kind == "dqi-path-majority":
        assert target == str(case.root)
        assert kwargs == {"source_root": case.source,
                          "expected_manifest_sha256": case.packet["manifest_sha256"], "time_limit_seconds": 5.0}
    elif kind == "rpe-resource-source":
        assert target == str(case.root / case.packet["prefix"])
        assert kwargs == {"source_root": case.source, "max_seconds": 30,
                          "manifest": case.packet["manifest_sha256"]}
    else:
        assert target == str(case.root / case.packet["prefix"])
        assert kwargs == ({"provenance": case.provenance, "omitted_assets": case.omitted if split else None}
                          if kind == "thermal-two-round" else {})
    if split:
        assert not (case.source / case.packet["prefix"]).exists()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", ("absent", "missing-member", "wrong-canonical-path", "refreshed-digest"))
def test_split_roots_require_exact_omitted_preview_joins(checked, tmp_path, monkeypatch, kind, mutation):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    gate = checked._load_next_saved_gate(case.source, case.provenance, kind)
    omitted = copy.deepcopy(case.omitted)
    name = next(iter(case.packet["asset_hashes"]))
    if mutation == "absent":
        omitted = None
    elif mutation == "missing-member":
        del omitted[name]
    elif mutation == "wrong-canonical-path":
        omitted[name]["site_path"] = "code/" + name
    else:
        omitted[name]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind=kind, omitted_assets=omitted)
    assert gate.CALLS == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", ("asset", "refreshed-asset-provenance", "retained-source"))
def test_identity_changes_are_rejected_before_saved_dispatch(checked, tmp_path, monkeypatch, kind, mutation):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    gate = checked._load_next_saved_gate(case.source, case.provenance, kind)
    if mutation == "retained-source":
        path = case.source / case.packet["helper"]
        path.write_bytes(path.read_bytes() + b"# changed retained bytes\n")
    else:
        name = next(iter(case.packet["asset_hashes"]))
        path = case.root / name
        raw = path.read_bytes() + b"changed asset\n"
        path.write_bytes(raw)
        if mutation == "refreshed-asset-provenance":
            case.provenance["file_sha256"][name] = sha(raw)
            case.omitted[name]["sha256"] = sha(raw)
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind=kind, omitted_assets=case.omitted)
    assert gate.CALLS == []


@pytest.mark.parametrize("kind", KINDS)
def test_saved_dispatch_cannot_expand_the_accepted_scope(checked, tmp_path, monkeypatch, kind):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    gate = checked._load_next_saved_gate(case.source, case.provenance, kind)
    method = {"thermal-two-round": "verify_full_bundle", "shor-finite-adder": "verify_reference",
              "dqi-path-majority": "display_saved_component",
              "rpe-resource-source": "verify_saved_source_export"}[kind]
    actual = getattr(gate, method)
    def forged_result(*args, **kwargs):
        result = actual(*args, **kwargs)
        key = {"thermal-two-round": "scientific_work_reexecuted", "shor-finite-adder": "helper_complete",
               "dqi-path-majority": "full_program_validated",
               "rpe-resource-source": "algorithm_numerically_validated"}[kind]
        result[key] = True
        return result
    monkeypatch.setattr(gate, method, forged_result)
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind=kind, omitted_assets=case.omitted)
    assert len(gate.CALLS) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_site_joins_provenance_retained_code_and_served_assets(checker, checked, tmp_path, monkeypatch, kind):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    monkeypatch.setattr(checker, "NEXT_SAVED_PACKETS", {kind: case.packet})
    calls = []
    def load(source, provenance, selected):
        assert source == case.source and provenance == case.provenance and selected == kind
        calls.append("load")
        return checked._load_next_saved_gate(source, provenance, selected)
    def verify(gate, root, **kwargs):
        assert root == case.root
        assert kwargs == {"source_root": case.source, "provenance": case.provenance,
                          "kind": kind, "omitted_assets": case.omitted}
        calls.append("verify")
        return checked._verify_next_saved_identity(gate, root, **kwargs)
    monkeypatch.setattr(checker, "_load_next_saved_gate", load)
    monkeypatch.setattr(checker, "_verify_next_saved_identity", verify)
    assert checker.site_next_saved_errors(case.root, case.preview) == []
    assert calls == ["load", "verify"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mutation", (
    "schema", "provenance-digest", "retained-source", "missing-omitted",
    "omitted-canonical-path", "omitted-digest",
))
def test_site_rejects_unjoined_preview_before_loading_helpers(checker, checked, tmp_path, monkeypatch, kind, mutation):
    case = synthetic_case(tmp_path, checked, monkeypatch, kind)
    monkeypatch.setattr(checker, "NEXT_SAVED_PACKETS", {kind: case.packet})
    preview = copy.deepcopy(case.preview)
    name = next(iter(case.packet["asset_hashes"]))
    if mutation == "schema":
        preview["schema"] = "unaccepted-preview-v1"
    elif mutation == "provenance-digest":
        preview["retained_sha256"]["PROVENANCE.json"] = "0" * 64
    elif mutation == "retained-source":
        preview["retained_sha256"][case.packet["helper"]] = "0" * 64
    elif mutation == "missing-omitted":
        del preview["omitted_assets"]
    elif mutation == "omitted-canonical-path":
        preview["omitted_assets"][name]["site_path"] = "code/" + name
    else:
        preview["omitted_assets"][name]["sha256"] = "0" * 64
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("An unjoined preview reached saved dispatch")
    monkeypatch.setattr(checker, "_load_next_saved_gate", forbidden)
    monkeypatch.setattr(checker, "_verify_next_saved_identity", forbidden)
    errors = checker.site_next_saved_errors(case.root, preview)
    assert len(errors) == 1 and "Saved next public evidence verification failed" in errors[0]
    assert calls == []


def bootstrap_bytes(snapshot):
    """Read the literal package bootstrap from AST, never run package_files."""
    assignments = [node for node in ast.walk(snapshot.function("package_files"))
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                       and target.value.id == "files" and isinstance(target.slice, ast.Constant)
                       and target.slice.value == "pyproject.toml" for target in node.targets)]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant) and type(assignments[0].value.value) is bytes
    raw = assignments[0].value.value
    assert {"bytes": len(raw), "sha256": sha(raw)} == BOOTSTRAP_PIN
    return raw


def test_rpe_excludes_only_original_config_snapshot_and_preserves_package_bootstrap(
        checked, snapshots, tmp_path, monkeypatch):
    approved = checked.NEXT_SAVED_PACKETS["rpe-resource-source"]
    code_names = {name.removeprefix(BASES["rpe-resource-source"] + "/code/")
                  for name in approved["asset_hashes"]
                  if name.startswith(BASES["rpe-resource-source"] + "/code/")}
    assert len(code_names) == 13
    assert set(approved["source_hashes"]) == code_names - {"pyproject.toml"} == SOURCE_NAMES["rpe-resource-source"]
    assert approved["source_config_snapshot_only"] == RPE_CONFIG_SNAPSHOT
    assert approved["asset_hashes"][RPE_CONFIG_SNAPSHOT] == RPE_CONFIG_PIN
    case = synthetic_case(tmp_path, checked, monkeypatch, "rpe-resource-source")
    bootstrap = bootstrap_bytes(snapshots["exporter"])
    bootstrap_path = case.source / "pyproject.toml"
    bootstrap_path.write_bytes(bootstrap)
    snapshot_path = case.root / RPE_CONFIG_SNAPSHOT
    original_snapshot = snapshot_path.read_bytes()
    assert original_snapshot != bootstrap
    reads = []
    actual_read = checked._read_bounded_public_member
    def tracked(root, name, maximum):
        reads.append((Path(root), name))
        return actual_read(root, name, maximum)
    monkeypatch.setattr(checked, "_read_bounded_public_member", tracked)
    gate = checked._load_next_saved_gate(case.source, case.provenance, "rpe-resource-source")
    checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
        provenance=case.provenance, kind="rpe-resource-source", omitted_assets=case.omitted)
    assert (case.source, "pyproject.toml") not in reads
    assert (case.root, RPE_CONFIG_SNAPSHOT) in reads
    assert bootstrap_path.read_bytes() == bootstrap
    assert snapshot_path.read_bytes() == original_snapshot
    assert len(gate.CALLS) == 1


@pytest.mark.parametrize("name", sorted(SOURCE_NAMES["rpe-resource-source"]))
def test_rpe_keeps_every_nonconfig_source_join_frozen(checked, tmp_path, monkeypatch, name):
    case = synthetic_case(tmp_path, checked, monkeypatch, "rpe-resource-source")
    gate = checked._load_next_saved_gate(case.source, case.provenance, "rpe-resource-source")
    path = case.source / name
    raw = path.read_bytes() + b"# refreshed nonconfig source\n"
    path.write_bytes(raw)
    case.provenance["file_sha256"][name] = sha(raw)
    if name.endswith(".py"):
        case.provenance["source_sha256"][name] = sha(raw)
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind="rpe-resource-source", omitted_assets=case.omitted)
    assert gate.CALLS == []


@pytest.mark.parametrize("name", (RPE_CONFIG_SNAPSHOT, RPE_ARCHIVE))
@pytest.mark.parametrize("refresh_provenance", (False, True))
def test_rpe_snapshot_and_zip_remain_full_byte_authenticated(
        checked, tmp_path, monkeypatch, name, refresh_provenance):
    case = synthetic_case(tmp_path, checked, monkeypatch, "rpe-resource-source")
    gate = checked._load_next_saved_gate(case.source, case.provenance, "rpe-resource-source")
    path = case.root / name
    raw = path.read_bytes() + b"changed pinned saved member\n"
    path.write_bytes(raw)
    if refresh_provenance:
        case.provenance["file_sha256"][name] = sha(raw)
        case.omitted[name]["sha256"] = sha(raw)
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind="rpe-resource-source", omitted_assets=case.omitted)
    assert gate.CALLS == []


def test_rpe_admits_one_exact_zip_and_rejects_unregistered_archives(exporter, checked):
    family = exporter.PUBLIC_RPE_RESOURCE_SOURCE_FILES
    assert {name for name in family if name.endswith(".zip")} == {RPE_ARCHIVE}
    assert checked.NEXT_SAVED_PACKETS["rpe-resource-source"]["asset_hashes"][RPE_ARCHIVE] == RPE_ARCHIVE_PIN
    assert exporter.include_generated_file(RPE_ARCHIVE)
    assert not exporter.include_generated_file(BASES["rpe-resource-source"] + "/other-download.zip")
    assert not exporter.include_generated_file(BASES["rpe-resource-source"] + "/code/source-export-download.zip")


@pytest.mark.parametrize("name", sorted(RPE_SOURCE_COUNTS))
@pytest.mark.parametrize("mutation", ("integer", "boolean", "float"))
def test_rpe_source_only_counts_are_exact_typed_integers(checked, tmp_path, monkeypatch, name, mutation):
    case = synthetic_case(tmp_path, checked, monkeypatch, "rpe-resource-source")
    gate = checked._load_next_saved_gate(case.source, case.provenance, "rpe-resource-source")
    actual = gate.verify_saved_source_export
    def forged(*args, **kwargs):
        result = actual(*args, **kwargs)
        result[name] = (RPE_SOURCE_COUNTS[name] + 1 if mutation == "integer" else
                        True if mutation == "boolean" else float(RPE_SOURCE_COUNTS[name]))
        return result
    monkeypatch.setattr(gate, "verify_saved_source_export", forged)
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind="rpe-resource-source", omitted_assets=case.omitted)
    assert len(gate.CALLS) == 1


@pytest.mark.parametrize("name", (
    "algorithm_numerically_validated", "physical_resource_estimate_computed",
    "prior_source_only_invocation_succeeded",
))
@pytest.mark.parametrize("replacement", (True, 0, None))
def test_rpe_negative_scope_flags_remain_exact_false(checked, tmp_path, monkeypatch, name, replacement):
    case = synthetic_case(tmp_path, checked, monkeypatch, "rpe-resource-source")
    gate = checked._load_next_saved_gate(case.source, case.provenance, "rpe-resource-source")
    actual = gate.verify_saved_source_export
    def forged(*args, **kwargs):
        result = actual(*args, **kwargs)
        result[name] = replacement
        return result
    monkeypatch.setattr(gate, "verify_saved_source_export", forged)
    with pytest.raises(ValueError):
        checked._verify_next_saved_identity(gate, case.root, source_root=case.source,
            provenance=case.provenance, kind="rpe-resource-source", omitted_assets=case.omitted)
    assert len(gate.CALLS) == 1


def test_package_dispatch_follows_the_prior_hamiltonian_loop_before_backend_stages(snapshots):
    function = snapshots["checked"].function("verify_package")
    prior = [node for node in function.body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Name) and node.iter.id == "HAMILTONIAN_PACKETS"]
    next_loops = [node for node in function.body if isinstance(node, ast.For)
                  and isinstance(node.iter, ast.Name) and node.iter.id == "NEXT_SAVED_PACKETS"]
    assert len(prior) == len(next_loops) == 1
    loop = next_loops[0]
    assert prior[0].end_lineno < loop.lineno
    loads = calls_named(loop, "_load_next_saved_gate")
    identities = calls_named(loop, "_verify_next_saved_identity")
    assert len(loads) == len(identities) == 1 and loads[0].lineno < identities[0].lineno
    assert [ast.unparse(arg) for arg in loads[0].args] == ["package", "provenance", "kind"]
    assert [ast.unparse(arg) for arg in identities[0].args] == ["next_gate", "package"]
    assert {item.arg: ast.unparse(item.value) for item in identities[0].keywords} == {
        "source_root": "package", "provenance": "provenance", "kind": "kind"}
    backend_environment = [node for node in function.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "environment" for target in node.targets)]
    assert len(backend_environment) == 1 and loop.end_lineno < backend_environment[0].lineno


def test_site_dispatch_is_inside_successful_preview_branch_after_prior_packets(site_snapshot):
    functions = [node for node in site_snapshot.tree.body if isinstance(node, ast.FunctionDef)
                 and calls_named(node, "verify_preview")]
    assert len(functions) == 1
    preview_tries = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Try)
                     and any(calls_named(child, "verify_preview") for child in node.body)]
    assert len(preview_tries) == 1
    accepted_branch = ast.Module(body=preview_tries[0].orelse, type_ignores=[])
    prior = calls_named(accepted_branch, "site_hamiltonian_packet_errors")
    calls = calls_named(accepted_branch, "site_next_saved_errors")
    assert len(prior) == len(calls) == 1 and prior[0].lineno < calls[0].lineno
    assert [ast.unparse(arg) for arg in calls[0].args] == ["root", "preview_identity"]


def test_download_requirements_and_new_article_labels_retain_previous_labels(checker, site_snapshot, exporter):
    required = set(checker.required_artifacts())
    assert set().union(*PUBLIC_FILES.values()) <= required
    expected_sources = set().union(*SOURCE_NAMES.values()) | {
        "tests/test_shor_finite_adder_reference.py", "tests/test_thermal_two_round_public_static.py",
        "tests/test_robust_qpe_resource_source_saved.py"}
    assert exporter.PUBLIC_NEXT_EVIDENCE_SOURCE_PATHS == expected_sources
    assert {"code/" + name for name in expected_sources} <= required
    used = names_loaded(site_snapshot.function("required_artifacts"))
    assert set(PUBLIC_SYMBOLS.values()) | {"PUBLIC_NEXT_EVIDENCE_SOURCE_PATHS"} <= used
    for route, labels in ARTICLE_LABELS.items():
        assert checker.NEXT_EVIDENCE_ARTICLE_LABELS[route] == labels
    old_registered = (checker.GIBBS_CHANNEL_ARTICLE_LABELS | checker.DECAY_ARTICLE_LABELS
                      | checker.CORRECTION_ARTICLE_LABELS)
    assert OLD_GIBBS_LABELS <= old_registered
    source = (ROOT / "algorithms/simulation/detailed-balance-gibbs.md").read_text()
    labels = {line[1:-2] for line in source.splitlines() if line.startswith("(") and line.endswith(")=")}
    labels |= {line.removeprefix(":name: ") for line in source.splitlines() if line.startswith(":name: ")}
    assert OLD_GIBBS_LABELS | ARTICLE_LABELS["algorithms/simulation/detailed-balance-gibbs/index.html"] <= labels
    assert "(shor-finite-adder-formal-reference)=" in (ROOT / "algorithms/shor.md").read_text()
    dqi = (ROOT / "algorithms/optimization/dqi-degree-two.md").read_text()
    assert "(dqi-path-majority-component)=" in dqi and "## Path-local majority decoder component" in dqi
    assert "(robust-resource-source-export)=" in (ROOT / "algorithms/qft-qpe/robust.md").read_text()
    label_loops = [node for node in ast.walk(site_snapshot.tree) if isinstance(node, ast.For)
                   and "NEXT_EVIDENCE_ARTICLE_LABELS" in names_loaded(node.iter)]
    assert len(label_loops) == 1
    assert any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
               and ast.unparse(node.left) == "labels" and ast.unparse(node.right) == "identifiers"
               for node in ast.walk(label_loops[0]))
    loaded = names_loaded(site_snapshot.tree)
    assert {"GIBBS_CHANNEL_ARTICLE_LABELS", "DECAY_ARTICLE_LABELS", "CORRECTION_ARTICLE_LABELS"} <= loaded


def test_exact_named_ci_projection_preserves_the_original_eighty_six_commands():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file():
        pytest.skip("Publication workflow is outside the complete public example package")
    commands = [line.strip().removeprefix("run ") for line in workflow.read_text().splitlines()
                if line.startswith("          run python ")]
    new_commands = NEXT_CI_COMMANDS + RPE_CI_COMMANDS
    assert len(commands) == ORIGINAL_CI_COUNT + len(new_commands)
    assert all(commands.count(command) == 1 for command in new_commands)
    projected = [command for command in commands if command not in new_commands]
    assert len(projected) == ORIGINAL_CI_COUNT
    assert sha(("\n".join(projected) + "\n").encode()) == ORIGINAL_CI_SHA256
    predecessors = (
        "python scripts/verify_shor_measured_and_saved.py",
        "python scripts/verify_detailed_balance_correction_saved.py --bundle assets/generated/simulation/detailed-balance-correction "
        "--source-root . --download assets/generated/simulation/detailed-balance-correction-download.zip",
        "python scripts/verify_dqi_degree_two_public_saved.py --artifact-root . --source-root . "
        "--notebook-path notebooks/optimization_dqi_degree_two.ipynb "
        "--expected-manifest-sha256 ecd71a350153cba5a17ae1451a1545b703e604608b3d95abee815cabbcc26f6c "
        "--archive assets/generated/optimization/dqi-degree-two-public/dqi-degree-two-download.zip --time-limit-seconds 10",
    )
    for command, previous in zip(NEXT_CI_COMMANDS, predecessors, strict=True):
        assert commands.index(command) == commands.index(previous) + 1
    assert commands.index(RPE_CI_COMMANDS[0]) == commands.index("python scripts/verify_robust_qpe_saved.py") + 1
