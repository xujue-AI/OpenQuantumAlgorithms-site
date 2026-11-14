"""Static public projection checks; no producer or saved display is executed."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets/generated/simulation/detailed-balance-two-round"
DISPLAY = ROOT / "scripts/display_thermal_two_round_saved.py"
ARTICLE = ROOT / "algorithms/simulation/detailed-balance-gibbs.md"
SCOPE = "fixed one-system two-round finite composition; no Gibbs/mixing/physical/formal guarantee"
SOURCE_PINS = {
    "thermal_two_round_contract.py": (7277, "2a5f41bf8e36981ef9adac3928aaede05b984b094771320158d45f771f2d9fb1"),
    "detailed_balance_thermal_two_round.py": (22928, "45ae8cc6b55d39caf29b80a8cf1aef770b7dae29d45d6e6bc41e341890c3b388"),
    "detailed_balance_thermal_two_round_reference.py": (39381, "98dfc9f75e6849419427ec2256a51adde1a52b1ac2bd6f8ee59ffaaa35ca8b52"),
    "detailed_balance_access.py": (25361, "7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77"),
    "detailed_balance_decay.py": (12324, "82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1"),
    "detailed_balance_channel.py": (24362, "99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b"),
    "detailed_balance_correction.py": (24902, "62ede4bbafe79a15e7c66068f0dc2b54a32322bbbff4015f00e7a0ba64e3eb48"),
}
CIRCUIT_NAMES = (
    "weak_dilation.qasm", "weak_reset_prefix.qasm", "correction_lcu.qasm",
    "round_macro.qasm", "two_round_trial.qasm", "correction-heralded-step.qasm",
)
PAYLOAD_NAMES = {
    *("sources/" + name for name in SOURCE_PINS),
    *("circuits/" + name for name in CIRCUIT_NAMES),
    "requirements-lock.txt", "LICENSE.txt", "UPSTREAM-LICENSES.md", "README.md",
    "results.json", "summary.json", "two-round-circuit.svg",
}
CORE_NAMES = PAYLOAD_NAMES | {"manifest.json", "manifest.sha256"}
FALSE_FLAGS = (
    "whole_invocation_validated", "all_components_verified", "new_algorithm_result",
    "channel_implemented", "coherent_correction_implemented",
    "coherent_correction_evolution_implemented", "general_channel_implemented",
    "general_coherent_correction_implemented", "exact_finite_kms",
    "finite_mesh_exact_kms", "mixing_certificate", "rotations_synthesized",
    "physical_resources_computed", "formal_proof_reproduced",
    "floating_point_error_certified", "hardware_error_certified",
    "random_sampling_performed", "sampler_implemented",
)
DIAGNOSTIC_MAPPING = {
    "weak_kraus_max_error": "actual_weak_kraus_entry_error",
    "terminal_completeness_error": "terminal_full_readout_completeness_error",
    "final_state_error": "accepted_final_vs_ideal_phi_squared_error",
    "z_bias_against_finite_target": "actual_z_vs_finite_psi_squared_bias_diagnostic",
    "correction_only_bound_4b": "ideal_two_round_correction_bias_bound_scalar",
    "gibbs_frobenius_distance": "optional_gibbs_state_residual_frobenius_diagnostic",
    "gibbs_generator_residual": "finite_generator_gibbs_residual_frobenius_diagnostic",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def saved_json(name):
    return json.loads((BUNDLE / name).read_bytes())


def display_tree_and_constants():
    """Inspect literal assignments without importing or executing the script."""
    tree = ast.parse(DISPLAY.read_text())
    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    return tree, constants


def test_public_core_is_closed_and_every_payload_matches_its_manifest():
    found = set()
    for path in BUNDLE.rglob("*"):
        info = path.lstat()
        assert path.resolve() == path
        if stat.S_ISDIR(info.st_mode):
            assert path.relative_to(BUNDLE).as_posix() in {"sources", "circuits"}
            continue
        assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        found.add(path.relative_to(BUNDLE).as_posix())
    assert found == CORE_NAMES
    manifest = saved_json("manifest.json")
    assert set(manifest) == {
        "schema", "accepted_source_commit", "scope", "raw_flags_unchanged",
        "original_accepted_payloads", "summary_path", "scientific_result_path",
        "files", "file_count", "selected_payload_bytes",
    }
    assert manifest["schema"] == "openquantum.fixed-two-round.public-manifest.v1"
    assert manifest["accepted_source_commit"] == "02c6bce80784b249927223b08b39174396d6e184"
    assert manifest["scope"] == SCOPE and manifest["raw_flags_unchanged"] is True
    assert manifest["original_accepted_payloads"] == {"files": 23, "bytes": 1755875}
    assert manifest["summary_path"] == "summary.json"
    assert manifest["scientific_result_path"] == "results.json"
    assert set(manifest["files"]) == PAYLOAD_NAMES
    assert type(manifest["file_count"]) is int and manifest["file_count"] == 20
    total = 0
    for name, binding in manifest["files"].items():
        assert set(binding) == {"bytes", "sha256"} and type(binding["bytes"]) is int
        raw = (BUNDLE / name).read_bytes()
        assert binding == {"bytes": len(raw), "sha256": digest(raw)}
        total += len(raw)
    assert type(manifest["selected_payload_bytes"]) is int
    assert manifest["selected_payload_bytes"] == total
    manifest_raw = (BUNDLE / "manifest.json").read_bytes()
    assert (BUNDLE / "manifest.sha256").read_text() == digest(manifest_raw) + "  manifest.json\n"


def test_original_scientific_sources_remain_byte_identical():
    for name, pin in SOURCE_PINS.items():
        raw = (BUNDLE / "sources" / name).read_bytes()
        assert (len(raw), digest(raw)) == pin
    for name, pin in {
        "requirements-lock.txt": (1974, "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"),
        "LICENSE.txt": (1093, "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"),
    }.items():
        raw = (BUNDLE / name).read_bytes()
        assert (len(raw), digest(raw)) == pin


def test_summary_preserves_raw_flags_and_saved_values_without_recalculation():
    summary, result = saved_json("summary.json"), saved_json("results.json")
    assert summary["schema"] == "openquantum.fixed-two-round.public-summary.v1"
    assert summary["status"] == "parent_accepted_fixed_finite_composition_only"
    assert summary["scope"] == result["parent_acceptance"]["scope"] == SCOPE
    assert set(summary["flags"]) == set(FALSE_FLAGS) | {
        "successful_completion_claimed", "finite_toy_checks_passed",
    }
    assert summary["flags"]["successful_completion_claimed"] is True
    assert summary["flags"]["finite_toy_checks_passed"] is True
    for name in FALSE_FLAGS:
        assert summary["flags"][name] is False
        assert result[name] is False
        if name in result["check"]:
            assert result["check"][name] is False
    assert summary["parameters"] == result["parameters"]
    assert summary["provenance"] == result["projection_provenance"]
    assert summary["checks"] == {"passed": 66, "total": 66}
    assert len(result["check"]["checks"]) == 66
    assert all(value is True for value in result["check"]["checks"].values())
    assert result["check"]["failed_checks"] == []
    for public_name, saved_name in DIAGNOSTIC_MAPPING.items():
        assert summary["diagnostics"][public_name] == result["check"]["floating_diagnostics"][saved_name]
        assert type(summary["diagnostics"][public_name]) is float
    assert summary["diagnostics"]["two_success_probability"] == result["check"]["actual_round_probability_weights"]["p1_times_p2"]
    for name, value in summary["source_accounting"].items():
        assert type(value) is int and value == result["source_accounting"][name]
    for public_name, saved_name in {
        "weak_statevectors": "statevector_evolutions",
        "external_density_stages": "external_density_evolutions",
        "tiny_expm": "tiny_expm_calls", "interval_powers": "interval_matrix_power_calls",
        "generator_expm": "finite_generator_expm_calls",
    }.items():
        assert summary["observed_calls"][public_name] == result["work_inventory"][saved_name]
        assert type(summary["observed_calls"][public_name]) is int
    assert summary["internal_reset_dispatches"] == {"predicted": 16, "measured": False}
    assert result["work_inventory"]["internal_dispatches_measured"] is False
    assert result["work_inventory"]["expected_internal_partial_reset_evolve_dispatches"] == 16
    proposal = result["check"]["ideal_bias_retry_projection"]
    assert proposal["N_evaluated"] is False and proposal["R_evaluated"] is False
    assert proposal["sampling_performed"] is False
    assert result["sampling_proposal"]["numerical_repetitions_evaluated"] is False


def test_public_projection_excludes_private_operational_fields_and_paths():
    for name in CORE_NAMES:
        raw = (BUNDLE / name).read_bytes()
        for forbidden in (
            b"/private/tmp/", b"/Users/", b"PAGES_DEPLOY_KEY",
            b"__CF_USER_TEXT_ENCODING", b"approved_once", b"numerical-grant",
            b"startup_environment_policy", b"thermal_two_round_once_supervisor",
            b"thermal_environment_checked_driver", b"grant_file_expected",
        ):
            assert forbidden not in raw, (name, forbidden)
    result = saved_json("results.json")
    assert not {"environment", "caps", "source_files", "source_records", "source_commit",
                "cpu_seconds_at_report", "wall_seconds_at_report"} & set(result)
    assert result["retained_correction_dependence"]["fresh_independent_failure_blocks"] is False


def test_display_source_is_only_a_bounded_metadata_display():
    tree, constants = display_tree_and_constants()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "argparse", "hashlib", "json", "math", "os", "pathlib", "stat"}
    forbidden_calls = {
        "exec", "eval", "compile", "__import__", "system", "popen", "fork", "kill",
        "write_bytes", "write_text", "mkdir", "unlink", "rename", "replace",
        "expm", "matrix_power", "eigh", "eig", "evolve", "reset", "run_check",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            assert name not in forbidden_calls
    verify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_metadata")
    reads = [node for node in ast.walk(verify) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_read_file"]
    assert [ast.literal_eval(node.args[1]) for node in reads] == ["manifest.json", "summary.json"]
    report = next(node.value for node in ast.walk(verify)
                  if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict))
    values = {key.value: item for key, item in zip(report.keys, report.values)}
    assert ast.literal_eval(values["metadata_identity_verified"]) is True
    assert ast.literal_eval(values["full_bundle_verified"]) is False
    assert ast.literal_eval(values["numerical_replay_performed"]) is False
    manifest_raw = (BUNDLE / "manifest.json").read_bytes()
    assert constants["MANIFEST_SHA256"] == digest(manifest_raw)
    assert constants["MANIFEST_BYTES"] == len(manifest_raw)
    assert constants["MAX_MANIFEST_BYTES"] == 65536
    assert constants["MAX_SUMMARY_BYTES"] == constants["MAX_DISPLAY_BYTES"] == 16384
    assert constants["SCOPE"] == SCOPE
    assert set(constants["FALSE_FLAGS"]) == set(FALSE_FLAGS)
    assert set(saved_json("summary.json")) == set(constants["SUMMARY_KEYS"])


def test_original_figure_is_accessible_and_has_no_active_or_external_content():
    raw = (BUNDLE / "two-round-circuit.svg").read_bytes()
    root = ET.fromstring(raw)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.get("role") == "img" and root.get("viewBox")
    labels = root.get("aria-labelledby", "").split()
    by_id = {node.get("id"): node for node in root.iter() if node.get("id")}
    assert len(labels) >= 2
    assert all(label in by_id and "".join(by_id[label].itertext()).strip() for label in labels)
    assert {by_id[label].tag.rsplit("}", 1)[-1] for label in labels} >= {"title", "desc"}
    for node in root.iter():
        assert node.tag.rsplit("}", 1)[-1] not in {"script", "foreignObject"}
        for name, value in node.attrib.items():
            assert not name.lower().startswith("on")
            if name.rsplit("}", 1)[-1] == "href":
                assert value.startswith("#")


def test_article_preserves_existing_labels_and_links_only_to_closed_projection_members():
    if not ARTICLE.is_file() and not (ROOT / "myst.yml").is_file():
        provenance_path = ROOT / "PROVENANCE.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_bytes())
            if (type(provenance) is dict
                    and provenance.get("schema") == "openquantum-public-package-v2"
                    and type(provenance.get("file_sha256")) is dict
                    and ARTICLE.relative_to(ROOT).as_posix() not in provenance["file_sha256"]):
                pytest.skip("Book article is deliberately outside the standalone code package")
    article = ARTICLE.read_text()
    for label in (
        "gibbs-channel-circuit", "gibbs-channel-checked-records",
        "gibbs-channel-output-contracts", "gibbs-channel-resource-projection",
        "gibbs-channel-saved-verification", "gibbs-decay-block", "gibbs-correction-step",
    ):
        assert "(" + label + ")=" in article
    assert "## Fixed two-round finite composition" in article
    section = article.split("## Fixed two-round finite composition", 1)[1].split("## End-to-end resources", 1)[0]
    assert all(name in section for name in ("whole_invocation_validated", "all_components_verified", "new_algorithm_result"))
    assert "predicted" in section and "staged" in section.lower()
    links = re.findall(r"\]\((\.\./\.\./assets/generated/simulation/detailed-balance-two-round/[^)]+)\)", section)
    assert links
    for link in links:
        destination = (ARTICLE.parent / link.split("#", 1)[0]).resolve()
        assert destination.is_relative_to(BUNDLE)
        assert destination.relative_to(BUNDLE).as_posix() in CORE_NAMES
        assert destination.is_file()


@pytest.mark.parametrize("context", ["package", "book", "unmarked", "included", "unknown_schema"])
def test_absent_article_scope_does_not_hide_missing_book_inputs(tmp_path, monkeypatch, context):
    article = tmp_path / "algorithms/simulation/detailed-balance-gibbs.md"
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "ARTICLE", article)
    if context == "book":
        (tmp_path / "myst.yml").write_text("project: {}\n")
    if context != "unmarked":
        provenance = {
            "schema": "unknown" if context == "unknown_schema" else "openquantum-public-package-v2",
            "file_sha256": {article.relative_to(tmp_path).as_posix(): "0" * 64} if context == "included" else {},
        }
        (tmp_path / "PROVENANCE.json").write_text(json.dumps(provenance))
    expected = pytest.skip.Exception if context == "package" else FileNotFoundError
    with pytest.raises(expected):
        test_article_preserves_existing_labels_and_links_only_to_closed_projection_members()
