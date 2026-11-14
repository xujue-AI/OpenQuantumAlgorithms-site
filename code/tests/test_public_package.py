"""Public downloads must retain reproducible inputs and fail on identity changes."""
from __future__ import annotations

from types import SimpleNamespace

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load_script("export_code")
verify = load_script("verify_public_package")

WORKLOAD_SCRIPTS = {
    "generate_word_taylor_workloads.py", "generate_shor_scaling.py",
    "generate_reciprocal_physical_sweep.py", "generate_simulation_lindblad_workload.py",
    "verify_optimization_checkpoint.py",
    "verify_optimization_diagnostics.py", "generate_simulation_lindblad_physical.py",
    "generate_simulation_complete_resources.py",
    "generate_word_taylor_physical.py",
    "generate_simulation_lindblad_three.py", "verify_simulation_lindblad_three.py",
    "verify_simulation_lindblad_physical.py",
    "generate_simulation_lindblad_four.py", "verify_simulation_lindblad_four.py",
    "generate_optimization_compiled.py", "complete_optimization_compiled.py", "verify_optimization_compiled.py",
    "run_simulation_lindblad_port.py", "verify_simulation_lindblad_port.py",
    "generate_simulation_qdrift.py", "verify_simulation_qdrift.py",
    "generate_simulation_suzuki.py", "verify_simulation_suzuki.py",
    "generate_detailed_balance_access.py", "generate_detailed_balance_channel.py", "verify_detailed_balance_channel.py",
    "generate_detailed_balance_decay.py", "verify_detailed_balance_decay.py",
    "generate_detailed_balance_correction.py", "verify_detailed_balance_correction_saved.py",
    "generate_optimization_dqi_degree_two.py", "validate_optimization_dqi_degree_two.py",
    "verify_dqi_degree_two_saved.py", "verify_dqi_degree_two_public_saved.py",
    "generate_shor_recovery.py", "generate_hadamard_qpe.py", "verify_hadamard_qpe.py",
    "generate_bayesian_qpe.py", "verify_bayesian_qpe_saved.py",
    "generate_robust_qpe.py", "verify_robust_qpe_saved.py",
    "generate_qpe_feedback_resources.py", "verify_qpe_feedback_saved.py", "verify_qpe_feedback_public.py",
    "generate_shor_measured_and.py", "verify_shor_measured_and_saved.py",
    "prepare_optimization_word_transport.py", "run_optimization_word_native.py",
    "optimization_word_record_schema.py", "optimization_word_records.py",
    "optimization_word_record_release.py", "verify_optimization_word_records.py",
    "generate_optimization_word_report.py",
}
GIBBS_CHANNEL_CI_COMMAND = "python scripts/verify_detailed_balance_channel.py"
GIBBS_PRIOR_CI_SHA256 = "3ef8e4fd0942022521192f1195f6695222b953e091056ed9313b96ac53021cf6"

SHOR_HADAMARD_CI_COMMANDS = (
    "python scripts/generate_shor_recovery.py",
    "python scripts/verify_hadamard_qpe.py",
)
SHOR_RECOVERY_BUNDLE_FILES = {
    "LICENSE.txt", "Quaspy-LICENSE.txt", "README.md", "circuit.svg", "generate_shor_recovery.py",
    "n15-a2-t8.qasm", "n21-a2-t6.qasm", "n33-a2-t4.qasm", "results.json", "shor.py",
    "shor_recovery.py", "success.svg", "summary.csv", "summary.md",
}
HADAMARD_BUNDLE_FILES = {
    "manifest.json", "circuit.svg", "summary.md", "results.json", "scaling.csv", "recovery.svg",
    "README.md", "LICENSE.txt", "requirements-lock.txt", "software.json", "logical-resources.json",
    "hadamard_qpe.py", "generate_hadamard_qpe.py", "circuits/large-n8-sin-k32.qasm",
} | {
    f"circuits/phase-{case}-{basis}-k{power}.qasm"
    for case in range(4) for basis in ("cos", "sin") for power in (1, 2, 4, 8, 16, 32)
}
QDRIFT_CI_COMMANDS = (
    "python scripts/generate_simulation_qdrift.py --output-dir .cache/qdrift-ci-replay",
    "python scripts/verify_simulation_qdrift.py --replay-dir .cache/qdrift-ci-replay",
)
QDRIFT_BUNDLE_FILES = {
    "manifest.json", "qdrift.py", "generate_simulation_qdrift.py", "LICENSE.txt", "summary.csv", "summary.md",
} | {
    f"{case}/{name}" for case in ("one-spin", "two-spin")
    for name in ("study.json", "channel-convergence.csv", "channel-and-samples.svg", "first-complete-trial.svg")
} | {
    f"{case}/trial-{index:03d}.qasm" for case in ("one-spin", "two-spin") for index in range(32)
}
SUZUKI_CI_COMMANDS = (
    "python scripts/generate_simulation_suzuki.py --output-dir .cache/suzuki-ci-replay",
    "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay",
)
# Reviewed bundles use saved verification in CI; original producers remain exported.
CI_WORKLOAD_SCRIPTS = WORKLOAD_SCRIPTS - {
    "generate_detailed_balance_access.py", "generate_detailed_balance_channel.py",
    "generate_detailed_balance_decay.py",
    "generate_simulation_lindblad_three.py", "generate_simulation_lindblad_physical.py",
    "generate_simulation_lindblad_four.py", "generate_optimization_compiled.py", "complete_optimization_compiled.py",
    "run_simulation_lindblad_port.py",
    "generate_hadamard_qpe.py", "generate_bayesian_qpe.py", "generate_robust_qpe.py",
    "generate_qpe_feedback_resources.py", "verify_qpe_feedback_saved.py",
    "generate_shor_measured_and.py",
    "prepare_optimization_word_transport.py", "run_optimization_word_native.py",
    "optimization_word_record_schema.py", "optimization_word_records.py",
    "optimization_word_record_release.py", "generate_optimization_word_report.py",
    "generate_dynamics_resources.py", "generate_instrument_resources.py", "generate_simulation_complete_resources.py",
    "generate_detailed_balance_correction.py",
    "generate_optimization_dqi_degree_two.py", "validate_optimization_dqi_degree_two.py",
    "verify_dqi_degree_two_saved.py",
} | {
    "verify_dynamics_resources.py", "verify_instrument_resource_records.py", "verify_simulation_complete_resource_records.py",
    "verify_dqi_saved.py",
    "verify_simulation_multiproduct_saved.py",
}
WORKLOAD_SENTINELS = {
    "assets/generated/word-taylor-workloads/manifest.json",
    "assets/generated/word-taylor-workloads/summary.csv",
    "assets/generated/word-taylor-workloads/summary.md",
    "assets/generated/word-taylor-workloads/size-workloads.svg",
    "assets/generated/word-taylor-workloads/size-n1-native/measured-attempt.qasm",
    "assets/generated/word-taylor-workloads/size-n4-normalization2-oaa1/record.json",
    "assets/generated/word-taylor-physical/manifest.json",
    "assets/generated/word-taylor-physical/table.md",
    "assets/generated/word-taylor-physical/summary.csv",
    "assets/generated/word-taylor-physical/summary.md",
    "assets/generated/word-taylor-physical/comparison.svg",
    "assets/generated/shor-scaling/study.json",
    "assets/generated/shor-scaling/tiny-verification.json",
    "assets/generated/shor-scaling/scaling.csv",
    "assets/generated/shor-scaling/resource-table.md",
    "assets/generated/shor-scaling/physical-scaling.svg",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/predeclared-plan.json",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/manifest.json",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/summary.csv",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/pareto.csv",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/physical-scaling.svg",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/n2-gap1of3/unsupported.json",
    "assets/generated/linear-systems/reciprocal-qsvt-physical/n2-gap1of4/unsupported.json",
    "assets/generated/simulation/lindblad-workload/manifest.json",
    "assets/generated/simulation/lindblad-workload/summary.csv",
    "assets/generated/simulation/lindblad-workload/convergence.csv",
    "assets/generated/simulation/complete-resources/manifest.json",
    "assets/generated/simulation/complete-resources/results.json",
    "assets/generated/simulation/complete-resources/summary.csv",
    "assets/generated/simulation/complete-resources/summary.md",
    "assets/generated/simulation/complete-resources/attempts.svg",
    "assets/generated/simulation/complete-resources/space-time.svg",
    "assets/generated/simulation/complete-resources/accuracy.svg",
}
WORKLOAD_SENTINELS |= {
    f"assets/generated/optimization/diagnostics/{name}"
    for name in ("freeze.json", "recipe.json", "raw-study.json", "runs.json", "resources.json",
                 "numerical-timing.json", "circuit-manifest.json", "results.csv", "controls.csv",
                 "evaluations.csv", "groups.csv", "control-evaluations.csv", "summary.md",
                 "overview.md", "quality.svg", "margins.svg", "workload.svg")
}
WORKLOAD_SENTINELS |= {
    f"assets/generated/simulation/lindblad-physical/{name}"
    for name in ("manifest.json", "results.json", "summary.csv", "family-status.csv", "README.md",
                 "source-trial.qasm", "clifford-t-trial.qasm", "one-step.svg", "counts.svg",
                 "error-budget.svg", "failures.svg", "space-time.svg")
}
WORKLOAD_SENTINELS |= {
    f"assets/generated/simulation/lindblad-three/{name}"
    for name in (
        "manifest.json", "progress.json", "results.json", "summary.csv", "family-status.csv", "README.md",
        "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
        "one-step.svg", "counts.svg", "error-budget.svg", "failures.svg", "space-time.svg",
        "artifact-inventory.csv", "artifact-inventory.json",
        "generate_simulation_lindblad_three.py", "bounded_density_instrument.py",
        "lindblad_three_physical.py", "instrument_resources.py", "lindblad_workload.py",
        "two_spin_lindblad.py", "open_systems.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    )
}
WORKLOAD_SENTINELS |= {
    f"assets/generated/simulation/lindblad-four/{name}"
    for name in (
        "manifest.json", "progress.json", "results.json", "source-preflight.json", "summary.csv", "family-status.csv", "README.md",
        "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
        "one-step.svg", "counts.svg", "error-budget.svg", "failures.svg", "space-time.svg",
        "artifact-inventory.csv", "artifact-inventory.json", "generate_simulation_lindblad_four.py",
        "lindblad_four_compile.py", "lindblad_four_density.py", "lindblad_four_physical.py",
        "bounded_density_instrument.py", "instrument_resources.py", "lindblad_workload.py",
        "two_spin_lindblad.py", "open_systems.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    )
}
WORKLOAD_SENTINELS |= {
    f"assets/generated/optimization/compiled-replay/{name}"
    for name in (
        "freeze.json", "recipe.json", "selected-transcript.json", "source-classes.json.gz", "manifest.json", "resources.json",
        "conditional-readout.json", "classical-work.json", "completed-replay.json.gz", "physical-dag.json.gz",
        "physical-summary.json", "physical-summary.csv", "groups.csv", "classes.csv", "negative-controls.csv",
        "summary.md", "workload.svg", "artifact-size.json", "completion-amendment.json", "completion-recipe.json",
        "completion-freeze.json", "completion-timing.json", "timing/synthesis.json", "timing/physical.json",
        "checkpoints/failures/physical-workload.json", "optimization_compiled.py", "generate_optimization_compiled.py",
        "complete_optimization_compiled.py", "optimization_compiled_codec.py", "instrument_resources.py",
        "parent_optimization.py", "parent_generate_optimization_diagnostics.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    )
}
WORKLOAD_SENTINELS |= export.PUBLIC_GIBBS_CHANNEL_FILES
WORKLOAD_SENTINELS |= export.PUBLIC_DECAY_FILES
WORKLOAD_SENTINELS |= export.PUBLIC_CORRECTION_FILES
WORKLOAD_SENTINELS |= export.PUBLIC_DQI_DEGREE_TWO_FILES | export.PUBLIC_DQI_DEGREE_TWO_PUBLIC_FILES
WORKLOAD_SENTINELS |= export.PUBLIC_COMPRESSED_JSON_FILES
WORKLOAD_SENTINELS |= export.PUBLIC_LINDBLAD_PORT_FILES
WORKLOAD_SENTINELS |= {f"assets/generated/shor/recovery/{name}" for name in SHOR_RECOVERY_BUNDLE_FILES}
WORKLOAD_SENTINELS |= {f"assets/generated/hadamard-qpe/{name}" for name in HADAMARD_BUNDLE_FILES}
WORKLOAD_SENTINELS |= {"assets/generated/simulation/qdrift/" + name for name in QDRIFT_BUNDLE_FILES}
WORKLOAD_SENTINELS |= {
    f"assets/generated/simulation/complete-resources/{case}/{name}"
    for case in ("dynamics-n1-Z", "dynamics-n2-Z", "thermal-n1-X", "thermal-n2-X", "thermal-n2-Z")
    for name in ("source-instrument.qasm", "clifford-t-instrument.qasm", "report.json")
}
WORKLOAD_SENTINELS |= {
    f"assets/generated/word-taylor-physical/{case}/{name}"
    for case in ("size-n1-native", "size-n1-normalization2-oaa1")
    for name in ("source-instrument.qasm", "routed-instrument.qasm", "clifford-t-instrument.qasm", "report.json")
}
STANDALONE_WORKLOAD_FILES = {
    "simulation/detailed-balance-correction/v2": (
        'detailed_balance_access.py',
        'detailed_balance_channel.py',
        'detailed_balance_decay.py',
        'detailed_balance_decay_reference.py',
        'detailed_balance_correction.py',
        'detailed_balance_correction_reference.py',
        'generate_detailed_balance_correction.py',
        'requirements-lock.txt',
        'LICENSE.txt',
    ),
    'simulation/detailed-balance-decay/grid': ('detailed_balance_access.py', 'detailed_balance_decay.py', 'detailed_balance_decay_reference.py', 'generate_detailed_balance_decay.py', 'requirements-lock.txt', 'LICENSE.txt'),
    'simulation/detailed-balance-decay/n12': ('detailed_balance_access.py', 'detailed_balance_decay.py', 'detailed_balance_decay_reference.py', 'generate_detailed_balance_decay.py', 'requirements-lock.txt', 'LICENSE.txt'),
    "simulation/detailed-balance-channel": ("detailed_balance_access.py", "detailed_balance_channel.py",
        "generate_detailed_balance_channel.py", "requirements-lock.txt", "LICENSE.txt"),
    "simulation/qdrift": ("qdrift.py", "generate_simulation_qdrift.py", "LICENSE.txt"),
    "optimization/diagnostics": (
        "optimization.py", "generate_optimization_diagnostics.py", "requirements-lock.txt", "LICENSE.txt",
    ),
    "simulation/lindblad-physical": (
        "generate_simulation_lindblad_physical.py", "lindblad_physical_workload.py",
        "instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py",
        "open_systems.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "simulation/lindblad-three": (
        "generate_simulation_lindblad_three.py", "bounded_density_instrument.py",
        "lindblad_three_physical.py", "instrument_resources.py", "lindblad_workload.py",
        "two_spin_lindblad.py", "open_systems.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "simulation/lindblad-four": (
        "generate_simulation_lindblad_four.py", "lindblad_four_compile.py", "lindblad_four_density.py", "lindblad_four_physical.py",
        "bounded_density_instrument.py", "instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py",
        "open_systems.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "optimization/compiled-replay": (
        "optimization_compiled.py", "generate_optimization_compiled.py", "complete_optimization_compiled.py",
        "optimization_compiled_codec.py", "instrument_resources.py", "parent_optimization.py",
        "parent_generate_optimization_diagnostics.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "simulation/lindblad-port": (
        "run_simulation_lindblad_port.py", "lindblad_port_words.py", "lindblad_port_rewrite.py", "lindblad_port_resources.py",
        "instrument_resources.py", "lindblad_workload.py", "two_spin_lindblad.py", "open_systems.py",
        "bounded_density_instrument.py", "lindblad_four_density.py", "lindblad_four_compile.py", "lindblad_four_physical.py",
        "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "word-taylor-workloads": (
        "generate_word_taylor_workloads.py", "word_taylor_workloads.py", "word_taylor.py",
        "amplification.py", "lcu.py", "hamiltonian_simulation.py", "source_provenance.py",
        "requirements-lock.txt", "LICENSE.txt",
    ),
    "word-taylor-physical": (
        "word_taylor_physical.py", "instrument_resources.py", "word_taylor_workloads.py",
        "word_taylor.py", "amplification.py", "lcu.py", "hamiltonian_simulation.py",
        "source_provenance.py", "generate_word_taylor_physical.py",
        "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "simulation/lindblad-workload": (
        "generate_simulation_lindblad_workload.py", "lindblad_workload.py", "two_spin_lindblad.py",
        "open_systems.py", "source_provenance.py", "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
    "simulation/complete-resources": (
        "instrument_resources.py", "hamiltonian_simulation.py", "lcu.py", "retargetable_qsp.py",
        "thermal_spectral.py", "qsp_phase_generation.py", "qsp_certificate.py",
        "complete_resource_workloads.py", "generate_simulation_complete_resources.py",
        "requirements-lock.txt", "references.bib", "LICENSE.txt",
    ),
}


def test_gibbs_channel_export_contract_is_exact():
    gate = load_script("verify_detailed_balance_channel")
    expected = {gate.BUNDLE_RELATIVE + "/" + name for name in gate.BUNDLE_FILES}
    assert export.PUBLIC_GIBBS_CHANNEL_FILES == expected
    assert len(expected) == 40
    assert gate.BUNDLE_RELATIVE + "/requirements-lock.txt" in export.PUBLIC_DEPENDENCY_FILES
    for name in expected:
        assert export.include_generated_file(name)
    for name in ("private.json", "unreviewed.md", "extra.csv", "log.txt", "verifier.py", "subdir/results.json"):
        assert not export.include_generated_file(gate.BUNDLE_RELATIVE + "/" + name)
    assert {"generate_detailed_balance_access.py", "generate_detailed_balance_channel.py",
            "verify_detailed_balance_channel.py"} <= set(export.PUBLIC_SCRIPTS)
    assert "verify_detailed_balance_channel.py" in CI_WORKLOAD_SCRIPTS
    assert not {"generate_detailed_balance_access.py", "generate_detailed_balance_channel.py"} & CI_WORKLOAD_SCRIPTS


def test_decay_export_contract_is_exact_and_keeps_generators_out_of_ci():
    gate = load_script("verify_detailed_balance_decay")
    expected = {gate.BUNDLE_RELATIVE + "/" + name for name in gate.BUNDLE_FILES}
    assert export.PUBLIC_DECAY_FILES == expected and len(expected) == 28
    for name in expected:
        assert export.include_generated_file(name)
    for name in ("private.json", "unreviewed.md", "log.txt", "extra.ipynb", "extra.sha256", "grid/verifier.py", "grid/subdir/results.json"):
        assert not export.include_generated_file(gate.BUNDLE_RELATIVE + "/" + name)
    assert {"generate_detailed_balance_decay.py", "verify_detailed_balance_decay.py"} <= set(export.PUBLIC_SCRIPTS)
    assert "verify_detailed_balance_decay.py" in CI_WORKLOAD_SCRIPTS
    assert "generate_detailed_balance_decay.py" not in CI_WORKLOAD_SCRIPTS


def test_gibbs_channel_ci_preserves_current74_command_vector():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file():
        pytest.skip("Publication workflow is outside the downloaded code package")
    commands = [line.strip().removeprefix("run ") for line in workflow.read_text().splitlines()
                if line.strip().startswith("run python ")]
    degree_two = 'python scripts/verify_dqi_degree_two_public_saved.py --artifact-root . --source-root . --notebook-path notebooks/optimization_dqi_degree_two.ipynb --expected-manifest-sha256 ecd71a350153cba5a17ae1451a1545b703e604608b3d95abee815cabbcc26f6c --archive assets/generated/optimization/dqi-degree-two-public/dqi-degree-two-download.zip --time-limit-seconds 10'
    assert len(commands) == 86 and commands.count(degree_two) == 1
    assert commands.index(degree_two) == commands.index("python scripts/verify_dqi_saved.py assets/generated/optimization/dqi") + 1
    commands = [value for value in commands if value != degree_two]
    correction = 'python scripts/verify_detailed_balance_correction_saved.py --bundle assets/generated/simulation/detailed-balance-correction --source-root . --download assets/generated/simulation/detailed-balance-correction-download.zip'
    assert len(commands) == 85 and commands.count(correction) == 1
    assert commands.index(correction) == commands.index("python scripts/verify_detailed_balance_decay.py") + 1
    commands = [value for value in commands if value != correction]
    hamiltonian_saved = (
        "python scripts/verify_simulation_cpf_public.py --bundle assets/generated/simulation/corrected-formula",
        "python scripts/verify_simulation_gqsp_public.py --bundle assets/generated/simulation/gqsp",
    )
    assert len(commands) == 84
    assert all(commands.count(value) == 1 for value in hamiltonian_saved)
    assert commands.index(hamiltonian_saved[0]) == commands.index(
        "python scripts/verify_simulation_multiproduct_saved.py") + 1
    assert commands.index(hamiltonian_saved[1]) == commands.index(hamiltonian_saved[0]) + 1
    commands = [value for value in commands if value not in hamiltonian_saved]
    decay = "python scripts/verify_detailed_balance_decay.py"
    assert len(commands) == 82 and commands.count(decay) == 1
    assert commands.index(decay) == commands.index("python scripts/verify_detailed_balance_channel.py") + 1
    commands = [value for value in commands if value != decay]
    measured_and = "python scripts/verify_shor_measured_and_saved.py"
    assert len(commands) == 81 and commands.count(measured_and) == 1
    assert commands.index(measured_and) == commands.index("python scripts/generate_shor_recovery.py") + 1
    commands = [value for value in commands if value != measured_and]
    feedback = "python scripts/verify_qpe_feedback_public.py"
    assert len(commands) == 80 and commands.count(feedback) == 1
    assert commands.index(feedback) == commands.index("python scripts/verify_robust_qpe_saved.py") + 1
    commands = [value for value in commands if value != feedback]
    robust = "python scripts/verify_robust_qpe_saved.py"
    assert len(commands) == 79 and commands.count(robust) == 1
    assert commands.index(robust) == commands.index("python scripts/verify_bayesian_qpe_saved.py") + 1
    commands = [value for value in commands if value != robust]
    bayesian = "python scripts/verify_bayesian_qpe_saved.py"
    assert len(commands) == 78 and commands.count(bayesian) == 1
    assert commands.index(bayesian) == commands.index("python scripts/verify_hadamard_qpe.py") + 1
    commands = [value for value in commands if value != bayesian]
    assert len(commands) == 77
    assert hashlib.sha256(("\n".join(commands) + "\n").encode()).hexdigest() == (
        "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032")
    multiproduct = "python scripts/verify_simulation_multiproduct_saved.py"
    assert commands.count(multiproduct) == 1
    assert commands.index(multiproduct) == commands.index(
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay") + 1
    commands = [value for value in commands if value != multiproduct]
    assert len(commands) == 76
    assert hashlib.sha256(("\n".join(commands) + "\n").encode()).hexdigest() == (
        "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af")
    dqi = "python scripts/verify_dqi_saved.py assets/generated/optimization/dqi"
    assert commands.count(dqi) == 1
    commands = [value for value in commands if value != dqi]
    assert commands.count(GIBBS_CHANNEL_CI_COMMAND) == 1
    assert commands.index(GIBBS_CHANNEL_CI_COMMAND) == commands.index("python scripts/generate_simulation_gibbs.py") + 1
    replacements = {
        "python scripts/verify_instrument_resource_records.py --replay": "python scripts/generate_instrument_resources.py",
        "python scripts/verify_simulation_complete_resource_records.py --replay": "python scripts/generate_simulation_complete_resources.py",
    }
    for replacement, original in replacements.items():
        assert commands.count(replacement) == 1 and original not in commands
    restored = [replacements.get(value, value) for value in commands]
    old = [line for line in restored if line != GIBBS_CHANNEL_CI_COMMAND]
    assert len(old) == 74
    assert hashlib.sha256(("\n".join(old) + "\n").encode()).hexdigest() == GIBBS_PRIOR_CI_SHA256


def test_gibbs_helper_loads_bound_bytes_and_ignores_module_cache(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    package = tmp_path.resolve() / "package"
    source = package / "scripts/verify_detailed_balance_channel.py"
    source.parent.mkdir(parents=True)
    raw = b"marker = 'authenticated'\n"
    source.write_bytes(raw)
    provenance = {"schema": "openquantum-public-package-v2",
                  "file_sha256": {"scripts/verify_detailed_balance_channel.py": hashlib.sha256(raw).hexdigest()}}
    monkeypatch.setitem(sys.modules, "_openquantum_verified_gibbs_channel_record_gate", SimpleNamespace(marker="unbound-cache"))
    gate = verify._load_gibbs_channel_record_gate(package, provenance)
    assert gate.marker == "authenticated"
    for bad in ({}, {"schema": "wrong", "file_sha256": provenance["file_sha256"]},
                {"schema": provenance["schema"], "file_sha256": {"scripts/verify_detailed_balance_channel.py": "0"*64}}):
        with pytest.raises(ValueError):
            verify._load_gibbs_channel_record_gate(package, bad)
    source.write_bytes(raw + b"raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError):
        verify._load_gibbs_channel_record_gate(package, provenance)


def test_gibbs_package_dispatch_starts_after_complete_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(verify, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(verify, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(verify, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    calls = []
    monkeypatch.setattr(verify, "verify_manifest", lambda *args: (_ for _ in ()).throw(ValueError("bad complete manifest")))
    monkeypatch.setattr(verify, "_load_gibbs_channel_record_gate", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="bad complete manifest"):
        verify.verify_package(tmp_path.resolve(), run_tests=False)
    assert calls == []


def test_gibbs_site_gate_requires_authenticated_preview_provenance(monkeypatch, tmp_path):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    from types import SimpleNamespace
    checker = load_script("check_site")
    root = tmp_path.resolve()
    code = root / "code"
    code.mkdir()
    raw = json.dumps({"schema": "openquantum-public-package-v2", "file_sha256": {}}).encode()
    (code / "PROVENANCE.json").write_bytes(raw)
    calls = []
    def load(package, provenance):
        calls.append((package, provenance))
        return SimpleNamespace(BUNDLE_RELATIVE="assets/generated/simulation/detailed-balance-channel",
            verify_bundle=lambda directory, *, source_root: calls.append((directory, source_root)))
    monkeypatch.setattr(checker, "_load_gibbs_channel_record_gate", load)
    assert checker.site_gibbs_channel_record_errors(root, {})
    assert calls == []
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": "0"*64}}
    assert checker.site_gibbs_channel_record_errors(root, identity)
    assert calls == []
    identity["retained_sha256"]["PROVENANCE.json"] = hashlib.sha256(raw).hexdigest()
    assert checker.site_gibbs_channel_record_errors(root, identity) == []
    assert calls[0][0] == code
    assert calls[1] == (root / "assets/generated/simulation/detailed-balance-channel", code)


def test_gibbs_required_routes_and_article_labels_are_scoped():
    if not (ROOT / "scripts/check_site.py").is_file() or not (ROOT / "algorithms/simulation/detailed-balance-gibbs.md").is_file():
        pytest.skip("Book routes and articles are deliberately outside the public example package")
    checker = load_script("check_site")
    required = set(checker.required_artifacts())
    assert {checker.GIBBS_CHANNEL_ARTICLE_ROUTE, checker.GIBBS_CHANNEL_NOTEBOOK_ROUTE,
            "notebooks/simulation_detailed_balance_channel.ipynb"} <= required
    assert export.PUBLIC_GIBBS_CHANNEL_FILES <= required
    assert {"code/scripts/" + name for name in checker.GIBBS_CHANNEL_PUBLIC_SCRIPTS} <= required
    article = (ROOT / "algorithms/simulation/detailed-balance-gibbs.md").read_text()
    for label in checker.GIBBS_CHANNEL_ARTICLE_LABELS:
        assert "(" + label + ")=" in article


def test_gibbs_recorded_notebook_cells_are_closed():
    notebook = json.loads((ROOT / "notebooks/simulation_detailed_balance_channel.ipynb").read_text())
    assert [cell["id"] for cell in notebook["cells"]] == [f"filtered-channel-{index}" for index in range(7)]
    assert len([cell for cell in notebook["cells"] if cell["cell_type"] == "code"]) == 5


def test_public_bundle_retains_required_scientific_inputs_and_excludes_operational_files():
    files = export.package_files()
    required = {
        "scripts/source_provenance.py", "scripts/generate_amplitude_iterative.py",
        "scripts/build_book.py", "scripts/public_assets.py",
        "scripts/generate_shor_driver.py", "scripts/generate_shor_ripple.py",
        "scripts/generate_linear_inverse_lcu.py", "scripts/generate_quantum_walks.py",
        "scripts/generate_energy_estimation.py", "scripts/generate_energy_estimation_pauli.py",
        "scripts/generate_grover_sat.py", "scripts/generate_dynamics_resources.py",
        "scripts/generate_fixed_point_amplification.py", "scripts/generate_instrument_resources.py",
        "assets/generated/simulation/qsp-scaling/results.json",
        "assets/generated/simulation/qsp-scaling/logical-resources.json",
        "assets/generated/simulation/qsp-scaling/generate_simulation_qsp_scaling.py",
        "assets/generated/simulation/qsp-scaling/evolution-n5.qasm",
        "assets/generated/simulation/qsp-evolution/phases.json",
        "assets/generated/optimization/annealing/runs.json",
        "assets/generated/optimization/annealing/resources.json",
        "assets/generated/optimization/annealing/annealing.qasm",
        "assets/generated/optimization/walk/runs.json",
        "assets/generated/optimization/walk/resources.json",
        "assets/generated/optimization/walk/walk-clifford-t.qasm",
        "assets/generated/optimization/walk/walk-source.qasm",
        "assets/generated/optimization/sampled/optimization.py",
        "assets/generated/optimization/sampled-runs.json",
        "assets/generated/quantum-walks/surface-code-scaling.json",
        "assets/generated/energy-estimation/molecular-fixture.json",
        "assets/generated/energy-estimation/measured-group.qasm",
        "assets/generated/energy-estimation/pauli-access/qpe.qasm",
        "assets/generated/grover-sat/iteration-circuits.zip",
        "assets/generated/dynamics-resources/manifest.json",
        "assets/generated/fixed-point-amplification/manifest.json",
        "assets/generated/instrument-resources/manifest.json",
        "assets/generated/instrument-resources/requirements-lock.txt",
        "assets/generated/instrument-resources/references.bib",
        "algorithms/optimization/formal/C4Reference.lean",
        "algorithms/optimization/formal/lean-toolchain",
        "assets/generated/optimization/formal/verification.json",
    }
    required |= WORKLOAD_SENTINELS | {"scripts/" + name for name in WORKLOAD_SCRIPTS}
    for directory, names in STANDALONE_WORKLOAD_FILES.items():
        required |= {f"assets/generated/{directory}/{name}" for name in names}
    assert required <= files.keys()
    cpf_archive = "assets/generated/simulation/corrected-formula/cpf2-public-packet.zip"
    archive_names = {name for name in files if name.endswith(".zip")}
    assert cpf_archive in archive_names and cpf_archive in export.PUBLIC_CORRECTED_FORMULA_FILES
    assert archive_names - {cpf_archive} == export.PUBLIC_ARCHIVES
    assert all(not name.endswith(".log") for name in files)
    assert "assets/generated/publication.json" not in files
    assert all(not name.startswith(("docs/", ".git/", ".cache/")) for name in files)
    # The replay fixture must keep its historical implementation bytes.
    historical = "assets/generated/optimization/sampled/optimization.py"
    assert files[historical] == (ROOT / historical).read_bytes()
    record = json.loads(files["PROVENANCE.json"])
    assert set(record["file_sha256"]) == set(files) - {"PROVENANCE.json"}


def test_workload_registration_selects_complete_standalone_dependencies():
    assert WORKLOAD_SCRIPTS <= set(export.PUBLIC_SCRIPTS)
    assert "generate_optimization_diagnostics.py" in export.PUBLIC_SCRIPTS
    assert {"word-taylor-workloads", "shor-scaling", "hadamard-qpe"} <= set(export.PUBLIC_ARTIFACT_TREES)
    assert {path for path in export.PUBLIC_DEPENDENCY_FILES if path.startswith("assets/generated/hadamard-qpe/")} == {
        "assets/generated/hadamard-qpe/requirements-lock.txt",
    }
    assert {path for path in export.PUBLIC_DEPENDENCY_FILES if path.startswith("assets/generated/shor/recovery/")} == {
        "assets/generated/shor/recovery/Quaspy-LICENSE.txt",
    }
    assert {
        path for path in export.PUBLIC_DEPENDENCY_FILES
        if path.startswith("assets/generated/simulation/lindblad-three/")
    } == {
        "assets/generated/simulation/lindblad-three/requirements-lock.txt",
        "assets/generated/simulation/lindblad-three/references.bib",
    }
    for directory in ("simulation/lindblad-four", "simulation/lindblad-port", "optimization/compiled-replay"):
        prefix = f"assets/generated/{directory}/"
        assert {path for path in export.PUBLIC_DEPENDENCY_FILES if path.startswith(prefix)} == {
            prefix + "requirements-lock.txt", prefix + "references.bib",
        }
    for path in WORKLOAD_SENTINELS:
        assert export.include_generated_file(path), path
    for directory, names in STANDALONE_WORKLOAD_FILES.items():
        for name in names:
            path = f"assets/generated/{directory}/{name}"
            assert export.include_generated_file(path), path


@pytest.mark.parametrize("relative", [
    "assets/generated/word-taylor-workloads/notes.txt",
    "assets/generated/word-taylor-workloads/private.bib",
    "assets/generated/word-taylor-workloads/unreviewed.toml",
    "assets/generated/word-taylor-workloads/private.log",
    "assets/generated/simulation/lindblad-three/notes.txt",
    "assets/generated/simulation/lindblad-three/private.bib",
    "assets/generated/simulation/lindblad-four/notes.txt",
    "assets/generated/simulation/lindblad-four/private.bib",
    "assets/generated/optimization/compiled-replay/notes.txt",
    "assets/generated/optimization/compiled-replay/private.bib",
    "assets/generated/shor-scaling/unreviewed.zip",
    "assets/generated/shor-scaling/.cache/raw.json",
    "assets/generated/shor/recovery/notes.txt",
    "assets/generated/shor/n21/Quaspy-LICENSE.txt",
    "assets/generated/hadamard-qpe/notes.txt",
    "assets/generated/hadamard-qpe/private.bib",
])
def test_dependency_exceptions_do_not_widen_formats_or_archives(relative):
    assert not export.include_generated_file(relative)


def test_compiled_compressed_registration_matches_the_complete_reviewed_inventory():
    directory = ROOT / "assets/generated/optimization/compiled-replay"
    expected = {path.relative_to(ROOT).as_posix() for path in directory.rglob("*.json.gz") if path.is_file()}
    assert export.PUBLIC_COMPRESSED_JSON_FILES - export.PUBLIC_WORD_COMPRESSED_FILES == expected
    assert len(expected) == 57
    prefix = "assets/generated/optimization/compiled-replay/"
    assert {path.removeprefix(prefix) for path in expected if "/checkpoints/" not in path} == {
        "source-classes.json.gz", "completed-replay.json.gz", "physical-dag.json.gz",
    }
    assert len({path for path in expected if path.startswith(prefix + "checkpoints/synthesis/")}) == 54
    for relative in expected:
        assert export.include_generated_file(relative), relative


@pytest.mark.parametrize("relative", [
    "assets/generated/optimization/compiled-replay/unreviewed.json.gz",
    "assets/generated/optimization/compiled-replay/checkpoints/synthesis/" + "0" * 64 + ".json.gz",
    "assets/generated/optimization/compiled-replay/checkpoints/failures/physical-workload.json.gz",
    "assets/generated/optimization/compiled-replay/./physical-dag.json.gz",
    "assets/generated//optimization/compiled-replay/physical-dag.json.gz",
    "assets/generated/optimization/compiled-replay/../compiled-replay/physical-dag.json.gz",
    "assets/generated/optimization/diagnostics/raw-study.json.gz",
    "assets/generated/simulation/lindblad-four/results.json.gz",
    "assets/generated/optimization/compiled-replay/physical-dag.gz",
])
def test_compressed_exceptions_reject_unreviewed_files_and_alias_spellings(relative):
    assert not export.include_generated_file(relative)


def test_workload_ci_replays_the_checkpoint_before_bounded_resources():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file():
        pytest.skip("Publication workflow is deliberately outside the public example package")
    import yaml
    build = yaml.safe_load(workflow.read_text())["jobs"]["build"]
    assert build["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    steps = build["steps"]
    assert all(
        step.get("env", {}).get(
            "PYTHONDONTWRITEBYTECODE", build["env"]["PYTHONDONTWRITEBYTECODE"]
        ) == "1"
        for step in steps if "run" in step
    )
    contracts = [step for step in steps if step.get("name") == "Check algorithm and resource contracts"]
    assert len(contracts) == 1
    assert contracts[0]["run"] == "python -m pytest -q"
    commands = next(step["run"] for step in steps
                    if step.get("name") == "Generate verified circuit and resource artifacts").splitlines()
    commands = [command.removeprefix("run ") for command in commands]
    python_commands = [command for command in commands if command.startswith("python ")]
    all_python_commands = python_commands
    added_notebooks = {
        "notebooks/simulation_corrected_formula.ipynb",
        "notebooks/simulation_gqsp.ipynb",
        "notebooks/simulation_detailed_balance_decay.ipynb",
    }
    notebook_paths = sorted(path.relative_to(ROOT).as_posix()
                            for path in (ROOT / "notebooks").glob("*.ipynb"))
    preserved_notebooks = {"notebooks/optimization_dqi_degree_two.ipynb"}
    assert len(notebook_paths) == 61 and added_notebooks | preserved_notebooks <= set(notebook_paths)
    refreshed_notebooks = [name for name in notebook_paths if name not in preserved_notebooks]
    assert len(refreshed_notebooks) == 60
    preserved_raw = (ROOT / "notebooks/optimization_dqi_degree_two.ipynb").read_bytes()
    assert len(preserved_raw) == 19950
    assert hashlib.sha256(preserved_raw).hexdigest() == "e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9"
    assert "notebooks/simulation_corrected_formula_portable.ipynb" not in notebook_paths
    prior_notebooks = [name for name in refreshed_notebooks if name not in added_notebooks]
    assert len(prior_notebooks) == 57
    assert hashlib.sha256(("\n".join(prior_notebooks) + "\n").encode()).hexdigest() == (
        "c0534b35c3c2fdca0ed8961dbc340ef22d2134b53021b1441ec20583f008a896")
    toc_notebooks = []
    def collect_toc_notebooks(entries):
        for entry in entries:
            name = entry.get("file", "")
            if name.endswith(".ipynb"):
                toc_notebooks.append(name)
            collect_toc_notebooks(entry.get("children", []))
    collect_toc_notebooks(yaml.safe_load((ROOT / "myst.yml").read_text())["project"]["toc"])
    assert len(toc_notebooks) == len(set(toc_notebooks)) == 60
    assert preserved_notebooks <= set(toc_notebooks)
    assert set(notebook_paths) - set(toc_notebooks) == {
        "notebooks/simulation_detailed_balance_decay.ipynb"
    }
    assert sorted(name for name in toc_notebooks if name not in added_notebooks | preserved_notebooks) == prior_notebooks
    degree_two = 'python scripts/verify_dqi_degree_two_public_saved.py --artifact-root . --source-root . --notebook-path notebooks/optimization_dqi_degree_two.ipynb --expected-manifest-sha256 ecd71a350153cba5a17ae1451a1545b703e604608b3d95abee815cabbcc26f6c --archive assets/generated/optimization/dqi-degree-two-public/dqi-degree-two-download.zip --time-limit-seconds 10'
    assert len(python_commands) == 86 and python_commands.count(degree_two) == 1
    assert python_commands.index(degree_two) == python_commands.index("python scripts/verify_dqi_saved.py assets/generated/optimization/dqi") + 1
    python_commands = [value for value in python_commands if value != degree_two]
    correction = 'python scripts/verify_detailed_balance_correction_saved.py --bundle assets/generated/simulation/detailed-balance-correction --source-root . --download assets/generated/simulation/detailed-balance-correction-download.zip'
    assert len(python_commands) == 85 and python_commands.count(correction) == 1
    assert python_commands.index(correction) == python_commands.index("python scripts/verify_detailed_balance_decay.py") + 1
    python_commands = [value for value in python_commands if value != correction]
    hamiltonian_saved = (
        "python scripts/verify_simulation_cpf_public.py --bundle assets/generated/simulation/corrected-formula",
        "python scripts/verify_simulation_gqsp_public.py --bundle assets/generated/simulation/gqsp",
    )
    assert len(python_commands) == 84
    assert all(python_commands.count(value) == 1 for value in hamiltonian_saved)
    assert python_commands.index(hamiltonian_saved[0]) == python_commands.index(
        "python scripts/verify_simulation_multiproduct_saved.py") + 1
    assert python_commands.index(hamiltonian_saved[1]) == python_commands.index(hamiltonian_saved[0]) + 1
    python_commands = [value for value in python_commands if value not in hamiltonian_saved]
    decay = "python scripts/verify_detailed_balance_decay.py"
    assert len(python_commands) == 82 and python_commands.count(decay) == 1
    assert python_commands.index(decay) == python_commands.index("python scripts/verify_detailed_balance_channel.py") + 1
    python_commands = [value for value in python_commands if value != decay]
    measured_and = "python scripts/verify_shor_measured_and_saved.py"
    assert len(python_commands) == 81 and python_commands.count(measured_and) == 1
    assert python_commands.index(measured_and) == python_commands.index("python scripts/generate_shor_recovery.py") + 1
    python_commands = [value for value in python_commands if value != measured_and]
    feedback = "python scripts/verify_qpe_feedback_public.py"
    assert len(python_commands) == 80 and python_commands.count(feedback) == 1
    assert python_commands.index(feedback) == python_commands.index("python scripts/verify_robust_qpe_saved.py") + 1
    python_commands = [value for value in python_commands if value != feedback]
    robust = "python scripts/verify_robust_qpe_saved.py"
    assert len(python_commands) == 79 and python_commands.count(robust) == 1
    assert python_commands.index(robust) == python_commands.index("python scripts/verify_bayesian_qpe_saved.py") + 1
    python_commands = [value for value in python_commands if value != robust]
    bayesian = "python scripts/verify_bayesian_qpe_saved.py"
    assert len(python_commands) == 78 and python_commands.count(bayesian) == 1
    assert python_commands.index(bayesian) == python_commands.index("python scripts/verify_hadamard_qpe.py") + 1
    python_commands = [value for value in python_commands if value != bayesian]
    assert len(python_commands) == 77
    assert hashlib.sha256(("\n".join(python_commands) + "\n").encode()).hexdigest() == (
        "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032")
    multiproduct = "python scripts/verify_simulation_multiproduct_saved.py"
    assert python_commands.count(multiproduct) == 1
    assert python_commands.index(multiproduct) == python_commands.index(
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay") + 1
    python_commands = [value for value in python_commands if value != multiproduct]
    assert len(python_commands) == 76
    assert hashlib.sha256(("\n".join(python_commands) + "\n").encode()).hexdigest() == (
        "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af")
    dqi = "python scripts/verify_dqi_saved.py assets/generated/optimization/dqi"
    assert python_commands.count(dqi) == 1
    assert python_commands.index(dqi) == python_commands.index("python scripts/verify_optimization_word_records.py") + 1
    python_commands = [value for value in python_commands if value != dqi]
    tail_replacements = {
        "python scripts/verify_instrument_resource_records.py --replay": "python scripts/generate_instrument_resources.py",
        "python scripts/verify_simulation_complete_resource_records.py --replay": "python scripts/generate_simulation_complete_resources.py",
    }
    for replacement, original in tail_replacements.items():
        assert python_commands.count(replacement) == 1 and original not in python_commands
    python_commands = [tail_replacements.get(value, value) for value in python_commands]
    assert python_commands.count(GIBBS_CHANNEL_CI_COMMAND) == 1
    assert python_commands.index(GIBBS_CHANNEL_CI_COMMAND) == python_commands.index("python scripts/generate_simulation_gibbs.py") + 1
    python_commands = [command for command in python_commands if command != GIBBS_CHANNEL_CI_COMMAND]
    assert len(python_commands) == 74
    assert hashlib.sha256(("\n".join(python_commands) + "\n").encode()).hexdigest() == GIBBS_PRIOR_CI_SHA256
    dynamics_replacement = ("python scripts/verify_dynamics_resources.py --canonical-qasm-replay "
                            "--records-dir assets/generated/dynamics-resources --manifest-sha256 "
                            "0596353af49a0de40176069756cd9398749c21ede68e057ef84611bb6d5f4265 "
                            "--max-wall-seconds 600")
    assert python_commands.count(dynamics_replacement) == 1
    assert "python scripts/generate_dynamics_resources.py" not in python_commands
    python_commands = ["python scripts/generate_dynamics_resources.py" if value == dynamics_replacement else value
                       for value in python_commands]
    for command in SUZUKI_CI_COMMANDS:
        assert python_commands.count(command) == 1
    assert python_commands.index(SUZUKI_CI_COMMANDS[1]) == python_commands.index(SUZUKI_CI_COMMANDS[0]) + 1
    assert python_commands.index(SUZUKI_CI_COMMANDS[0]) == python_commands.index(QDRIFT_CI_COMMANDS[1]) + 1
    python_commands = [command for command in python_commands if command not in SUZUKI_CI_COMMANDS]
    assert hashlib.sha256(("\n".join(python_commands) + "\n").encode()).hexdigest() == (
        "4af3666293942bcc65be97ac01078f93358db2e3dd44b7a09fcd2571b64030bd"
    )
    word_command = "python scripts/verify_optimization_word_records.py"
    assert len(python_commands) == 72 and python_commands.count(word_command) == 1
    assert python_commands.index(word_command) == python_commands.index(
        "python scripts/verify_optimization_compiled.py") + 1
    python_commands = [command for command in python_commands if command != word_command]
    assert len(python_commands) == 71
    assert hashlib.sha256(("\n".join(python_commands) + "\n").encode()).hexdigest() == (
        "50acb5da4238bfdec719a89b18469e68dfe6881c4483c8df70790b6d4e45962c"
    )
    for command in SHOR_HADAMARD_CI_COMMANDS:
        assert python_commands.count(command) == 1
    qdrift_and_prior_commands = [command for command in python_commands if command not in SHOR_HADAMARD_CI_COMMANDS]
    assert len(qdrift_and_prior_commands) == 69
    for command in QDRIFT_CI_COMMANDS:
        assert python_commands.count(command) == 1
    assert python_commands.index(QDRIFT_CI_COMMANDS[1]) == python_commands.index(QDRIFT_CI_COMMANDS[0]) + 1
    assert python_commands.index(QDRIFT_CI_COMMANDS[0]) == python_commands.index(
        "python scripts/generate_hamiltonian_simulation.py") + 1
    preserved_commands = [command for command in qdrift_and_prior_commands if command not in QDRIFT_CI_COMMANDS]
    port_command = "python scripts/verify_simulation_lindblad_port.py"
    assert len(preserved_commands) == 67 and preserved_commands.count(port_command) == 1
    assert preserved_commands.index(port_command) == preserved_commands.index("python scripts/verify_simulation_lindblad_four.py") + 1
    prior_commands = [command for command in preserved_commands if command != port_command]
    assert hashlib.sha256(("\n".join(prior_commands) + "\n").encode()).hexdigest() == (
        "6d2b2979fdb474301a25d2e5479efa06c2cb1227ffbca786e8d552cb74391013"
    )
    for name in CI_WORKLOAD_SCRIPTS:
        prefix = "python scripts/" + name
        assert any(command == prefix or command.startswith(prefix + " ") for command in all_python_commands)
    assert not any("generate_simulation_lindblad_three.py" in command for command in commands)
    assert not any("generate_simulation_lindblad_physical.py" in command for command in commands)
    assert not any("generate_simulation_lindblad_four.py" in command for command in commands)
    assert not any("run_simulation_lindblad_port.py" in command for command in commands)
    assert not any("generate_hadamard_qpe.py" in command for command in commands)
    assert not any("generate_optimization_compiled.py" in command or "complete_optimization_compiled.py" in command
                   for command in commands)
    assert not any("prepare_optimization_word_transport.py" in command
                   or "run_optimization_word_native.py" in command
                   or "generate_optimization_word_report.py" in command for command in commands)
    verification = commands.index("python scripts/verify_optimization_checkpoint.py")
    assert commands[verification + 1] == (
        "python scripts/generate_optimization_quality_resources.py "
        "--max-queries 96 --max-wall-seconds 1200 --require-complete"
    )
    assert not any("generate_optimization_quality_scaling.py" in command or "--resume" in command
                   for command in commands)


def test_qdrift_public_registration_preserves_the_complete_record_and_notebook_labels():
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Rendered-site checker is deliberately outside the public example package")
    checker = load_script("check_site")
    required = set(checker.required_artifacts())
    assert len(QDRIFT_BUNDLE_FILES) == 78
    assert set(checker.QDRIFT_FILES) == QDRIFT_BUNDLE_FILES
    assert {f"assets/generated/simulation/qdrift/{name}" for name in QDRIFT_BUNDLE_FILES} <= required
    assert {
        "algorithms/simulation/qdrift/index.html", "notebooks/simulation_qdrift.ipynb",
        checker.QDRIFT_NOTEBOOK_ROUTE, "code/src/openquantum/simulation/qdrift.py",
        "code/scripts/generate_simulation_qdrift.py", "code/scripts/verify_simulation_qdrift.py",
    } <= required
    notebook = json.loads((ROOT / "notebooks/simulation_qdrift.ipynb").read_text())
    labels = {cell.get("metadata", {}).get("label") for cell in notebook["cells"]}
    assert checker.QDRIFT_EMBED_LABELS <= labels
    assert len(checker.QDRIFT_EMBED_LABELS) == 5


def test_shor_hadamard_public_registration_retains_complete_records_and_notebook_labels():
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Rendered-site checker is deliberately outside the public example package")
    checker = load_script("check_site")
    required = set(checker.required_artifacts())
    assert len(SHOR_RECOVERY_BUNDLE_FILES) == 14
    assert len(HADAMARD_BUNDLE_FILES) == 62
    assert checker.SHOR_RECOVERY_REQUIRED_FILES == SHOR_RECOVERY_BUNDLE_FILES
    assert checker.HADAMARD_REQUIRED_FILES == HADAMARD_BUNDLE_FILES
    for directory, expected in (("shor/recovery", SHOR_RECOVERY_BUNDLE_FILES), ("hadamard-qpe", HADAMARD_BUNDLE_FILES)):
        root = ROOT / "assets/generated" / directory
        assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == expected
        assert {f"assets/generated/{directory}/{name}" for name in expected} <= required
    manifest_path = ROOT / "assets/generated/hadamard-qpe/manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "9e451d29d041746701686fe9ae7e17304fefabc61702a9f51e20d8ead10a2776"
    )
    assert set(json.loads(manifest_path.read_text())["file_sha256"]) | {"manifest.json"} == HADAMARD_BUNDLE_FILES
    assert {
        "algorithms/shor/order-recovery/index.html", "notebooks/shor_recovery.ipynb",
        checker.SHOR_RECOVERY_NOTEBOOK_ROUTE, "code/src/openquantum/shor.py",
        "code/src/openquantum/shor_recovery.py", "code/scripts/generate_shor_recovery.py",
        "algorithms/qft-qpe/hadamard/index.html", "notebooks/hadamard_qpe.ipynb",
        checker.HADAMARD_NOTEBOOK_ROUTE, "code/src/openquantum/hadamard_qpe.py",
        "code/scripts/generate_hadamard_qpe.py", "code/scripts/verify_hadamard_qpe.py",
    } <= required
    for name, expected, count in (
        ("shor_recovery", checker.SHOR_RECOVERY_EMBED_LABELS, 5),
        ("hadamard_qpe", checker.HADAMARD_EMBED_LABELS, 4),
    ):
        notebook = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
        labels = {cell.get("metadata", {}).get("label") for cell in notebook["cells"]}
        assert expected <= labels and len(expected) == count


def test_immutable_hadamard_figures_keep_bytes_when_other_figures_are_normalized(tmp_path):
    import runpy
    script = tmp_path / "scripts/normalize_figures.py"
    script.parent.mkdir()
    script.write_bytes((ROOT / "scripts/normalize_figures.py").read_bytes())
    protected = [tmp_path / "assets/generated/hadamard-qpe" / name for name in ("circuit.svg", "recovery.svg")]
    original = b"<svg>  \n  historical producer bytes  \n</svg>\n\n"
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original)
    recorded_relative = "assets/generated/optimization/compiled-replay/workload.svg"
    recorded = tmp_path / recorded_relative
    recorded.parent.mkdir(parents=True)
    recorded_original = (ROOT / recorded_relative).read_bytes()
    recorded.write_bytes(recorded_original)
    ordinary = tmp_path / "assets/generated/shor/recovery/circuit.svg"
    ordinary.parent.mkdir(parents=True)
    ordinary.write_bytes(original)
    runpy.run_path(str(script))
    assert all(path.read_bytes() == original for path in protected)
    assert recorded.read_bytes() == recorded_original
    assert ordinary.read_bytes() == b"<svg>\n  historical producer bytes\n</svg>\n\n"


def test_site_contract_requires_new_records_and_standalone_dependencies(tmp_path, monkeypatch):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Rendered-site checker is deliberately outside the public example package")
    checker = load_script("check_site")
    monkeypatch.setattr(checker, "site_dqi_degree_two_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_correction_record_errors", lambda *args, **kwargs: [])
    required = set(checker.required_artifacts())
    assert WORKLOAD_SENTINELS <= required
    assert {"code/scripts/" + name for name in WORKLOAD_SCRIPTS} <= required
    for directory, names in STANDALONE_WORKLOAD_FILES.items():
        assert {f"assets/generated/{directory}/{name}" for name in names} <= required
    assert not any(path.startswith(("code/assets/generated/simulation/lindblad-four/",
                                    "code/assets/generated/simulation/lindblad-port/",
                                    "code/assets/generated/optimization/compiled-replay/",
                                    "code/assets/generated/shor/recovery/",
                                    "code/assets/generated/hadamard-qpe/")) for path in required)
    current = "assets/generated/optimization/quality-resources"
    assert {f"{current}/{name}" for name in (
        "manifest.json", "model-signatures.json", "projection-reports.json",
    )} <= required

    source_files = {
        "resource_generator_sha256": "code/scripts/generate_optimization_quality_resources.py",
        "shared_adapter_sha256": "code/src/openquantum/resource_estimation.py",
        "audited_trace_helper_sha256": "code/scripts/generate_optimization_trajectory_resources.py",
        "environment_lock_sha256": "code/requirements-lock.txt",
        "recorded_implementation_sha256": "assets/generated/optimization/quality-scaling/optimization.py",
    }
    input_files = {
        name: f"assets/generated/optimization/quality-scaling/{name}.json"
        for name in ("runs", "resources", "circuit-manifest")
    }
    publication_files = {
        path: (f"# Recorded fixture for {path}\n").encode()
        for path in source_files.values()
    }
    publication_files.update({path: json.dumps({"recorded_input": name}).encode()
                              for name, path in input_files.items()})
    snapshot = f"{current}/generate_optimization_quality_resources.py"
    publication_files[snapshot] = publication_files[source_files["resource_generator_sha256"]]
    assert set(publication_files) <= required
    for relative, content in publication_files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    provenance = {
        field: hashlib.sha256(publication_files[path]).hexdigest()
        for field, path in source_files.items()
    }
    provenance["input_sha256"] = {
        name: hashlib.sha256(publication_files[path]).hexdigest()
        for name, path in input_files.items()
    }

    signature_ids = [f"signature-{index}" for index in range(84)]
    first = signature_ids[0]
    records = {
        "manifest.json": {
            "status": "computed_conditional_recorded_replay", "pending_reason": None,
            "query_limit": 96, "native_query_count": 85, "planned_native_queries": 85,
            "cost_signature_count": 84, "total_preparations": 5_859_840,
            "prepared_state_checks": 1_873, "query_plan_complete": True,
            "source_provenance": provenance,
            "choices": [{"preparations": 5_859_840} for _ in range(4)],
            "audit_counts": {
                "recorded_groups_bound_to_numerical_calls": 7_656,
                "source_group_reconstructions": 7_656,
                "unique_prepared_state_checks": 1_873,
                "canonical_trace_signature_audits": 3_746,
                "distinct_qasm_trace_reparse_audits": 1,
                "physical_estimator_search_requests": 85,
            },
        },
        "model-signatures.json": {
            "signatures": {identity: {"id": identity} for identity in signature_ids},
        },
        "projection-reports.json": {
            "status": "computed_conditional_recorded_replay",
            "reports": {identity: {
                "signature_id": identity, "report": {"status": "computed_conditional_model"},
            } for identity in signature_ids},
            "probe": {
                "signature_id": first, "distinct_source_qasm": True, "equivalent_costs": True,
                "relative_float_tolerance": 1e-12, "absolute_float_tolerance": 0,
                "representative_fingerprint": {"cost": 1}, "alternate_fingerprint": {"cost": 1},
                "alternate_report": {"status": "computed_conditional_model"},
            },
        },
    }
    directory = tmp_path / current
    directory.mkdir(parents=True, exist_ok=True)

    def write_records(contents):
        for name, record in contents.items():
            (directory / name).write_text(json.dumps(record))

    write_records(records)
    archive = tmp_path / "assets/generated/optimization/quality-resources-cap64"
    archive.mkdir()
    (archive / "manifest.json").write_text(json.dumps({
        "status": "pending_physical_layer", "native_query_count": 0, "choices": [],
    }))
    assert checker.current_quality_resource_errors(tmp_path) == []

    for relative, content in publication_files.items():
        path = tmp_path / relative
        path.write_bytes(content + b"\nchanged bytes\n")
        assert checker.current_quality_resource_errors(tmp_path), relative
        path.write_bytes(content)
        path.unlink()
        assert checker.current_quality_resource_errors(tmp_path), relative
        path.write_bytes(content)
        assert checker.current_quality_resource_errors(tmp_path) == [], relative
    for field in (*source_files, "input_sha256"):
        corrupted = json.loads(json.dumps(records))
        del corrupted["manifest.json"]["source_provenance"][field]
        write_records(corrupted)
        assert checker.current_quality_resource_errors(tmp_path), field
    for name in input_files:
        corrupted = json.loads(json.dumps(records))
        del corrupted["manifest.json"]["source_provenance"]["input_sha256"][name]
        write_records(corrupted)
        assert checker.current_quality_resource_errors(tmp_path), name
    for field in source_files:
        corrupted = json.loads(json.dumps(records))
        corrupted["manifest.json"]["source_provenance"][field] = "0" * 64
        write_records(corrupted)
        assert checker.current_quality_resource_errors(tmp_path), field
    for name in input_files:
        corrupted = json.loads(json.dumps(records))
        corrupted["manifest.json"]["source_provenance"]["input_sha256"][name] = "0" * 64
        write_records(corrupted)
        assert checker.current_quality_resource_errors(tmp_path), name
    stale_generator = publication_files[snapshot] + b"\n# Historical standalone generator\n"
    (tmp_path / snapshot).write_bytes(stale_generator)
    corrupted = json.loads(json.dumps(records))
    corrupted["manifest.json"]["source_provenance"]["resource_generator_sha256"] = hashlib.sha256(stale_generator).hexdigest()
    write_records(corrupted)
    assert checker.current_quality_resource_errors(tmp_path)
    (tmp_path / snapshot).write_bytes(publication_files[snapshot])
    write_records(records)
    assert checker.current_quality_resource_errors(tmp_path) == []
    updated_generator = publication_files[snapshot] + b"\n# Updated current generator\n"
    generator_path = source_files["resource_generator_sha256"]
    for relative in (generator_path, snapshot):
        (tmp_path / relative).write_bytes(updated_generator)
    updated = json.loads(json.dumps(records))
    updated["manifest.json"]["source_provenance"]["resource_generator_sha256"] = hashlib.sha256(updated_generator).hexdigest()
    write_records(updated)
    assert checker.current_quality_resource_errors(tmp_path) == []
    for relative in (generator_path, snapshot):
        (tmp_path / relative).write_bytes(publication_files[relative])
    write_records(records)
    assert checker.current_quality_resource_errors(tmp_path) == []

    corruptions = [
        (("manifest.json", "source_provenance"), None, "source_provenance"),
        (("manifest.json", "source_provenance", "input_sha256"), [], "input_sha256"),
        (("manifest.json", "status"), "pending_physical_layer", "status"),
        (("manifest.json", "pending_reason"), "wall time exhausted", "pending_reason"),
        (("manifest.json", "query_limit"), 64, "query_limit"),
        (("manifest.json", "native_query_count"), 0, "native_query_count"),
        (("manifest.json", "planned_native_queries"), 84, "planned_native_queries"),
        (("manifest.json", "cost_signature_count"), 83, "cost_signature_count"),
        (("manifest.json", "total_preparations"), 1, "total_preparations"),
        (("manifest.json", "prepared_state_checks"), 0, "prepared_state_checks"),
        (("manifest.json", "query_plan_complete"), False, "query_plan_complete"),
        (("manifest.json", "choices"), [], "four aggregate choices"),
        (("manifest.json", "choices"), [{}, {}, {}, {}], "four aggregate choices"),
        (("manifest.json", "choices"), [{}, {}, {}, None], "four aggregate choices"),
        (("manifest.json", "audit_counts"), None, "audit_counts"),
        (("manifest.json", "audit_counts", "distinct_qasm_trace_reparse_audits"), True,
         "distinct_qasm_trace_reparse_audits"),
        (("manifest.json", "audit_counts", "physical_estimator_search_requests"), 0,
         "physical_estimator_search_requests"),
        (("model-signatures.json", "signatures"), {}, "84 signatures"),
        (("model-signatures.json", "signatures", first, "id"), "other", "model signature ID"),
        (("model-signatures.json", "signatures", first), [], "model signature ID"),
        (("projection-reports.json", "status"), "pending_physical_layer", "non-pending status"),
        (("projection-reports.json", "pending_reason"), "interrupted", "non-pending status"),
        (("projection-reports.json", "reports"), {}, "84 reports"),
        (("projection-reports.json", "reports", first, "signature_id"), "other", "report signature ID"),
        (("projection-reports.json", "reports", first), [], "report signature ID"),
        (("projection-reports.json", "reports", first, "report", "status"), "pending", "physical report"),
        (("projection-reports.json", "reports", first, "report", "pending_reason"), "interrupted", "physical report"),
        (("projection-reports.json", "probe"), None, "distinct-QASM probe"),
        (("projection-reports.json", "probe", "signature_id"), "other", "probe signature ID"),
        (("projection-reports.json", "probe", "signature_id"), [], "probe signature ID"),
        (("projection-reports.json", "probe", "distinct_source_qasm"), False, "distinct_source_qasm"),
        (("projection-reports.json", "probe", "equivalent_costs"), 1, "equivalent_costs"),
        (("projection-reports.json", "probe", "relative_float_tolerance"), 1e-6, "relative_float_tolerance"),
        (("projection-reports.json", "probe", "absolute_float_tolerance"), 1, "absolute_float_tolerance"),
        (("projection-reports.json", "probe", "representative_fingerprint"), {}, "representative_fingerprint"),
        (("projection-reports.json", "probe", "alternate_fingerprint"), None, "alternate_fingerprint"),
        (("projection-reports.json", "probe", "alternate_fingerprint"), {"cost": 2}, "cost fingerprints differ"),
        (("projection-reports.json", "probe", "alternate_report"), {}, "alternate report"),
        (("projection-reports.json", "probe", "alternate_report", "pending_reason"), "interrupted", "alternate report"),
    ]
    for keys, value, expected_error in corruptions:
        corrupted = json.loads(json.dumps(records))
        destination = corrupted
        for key in keys[:-1]:
            destination = destination[key]
        destination[keys[-1]] = value
        write_records(corrupted)
        errors = checker.current_quality_resource_errors(tmp_path)
        assert any(expected_error in error for error in errors), (keys, errors)

    rounded = json.loads(json.dumps(records))
    probe = rounded["projection-reports.json"]["probe"]
    probe["representative_fingerprint"] = {"costs": [1, {"runtime": 1.0}]}
    probe["alternate_fingerprint"] = {"costs": [1, {"runtime": 1.0 + 5e-13}]}
    write_records(rounded)
    assert checker.current_quality_resource_errors(tmp_path) == []
    for alternate in (
        {"costs": [2, {"runtime": 1.0}]},
        {"costs": [True, {"runtime": 1.0}]},
        {"costs": [1, {"runtime": 1.0 + 2e-12}]},
        {"costs": [1, {"runtime": float("inf")}]},
    ):
        probe["alternate_fingerprint"] = alternate
        write_records(rounded)
        assert any("cost fingerprints differ" in error for error in checker.current_quality_resource_errors(tmp_path))
    probe["representative_fingerprint"] = {"cost": 0.0}
    probe["alternate_fingerprint"] = {"cost": 1e-30}
    write_records(rounded)
    assert any("cost fingerprints differ" in error for error in checker.current_quality_resource_errors(tmp_path))

    corrupted = json.loads(json.dumps(records))
    del corrupted["manifest.json"]["pending_reason"]
    write_records(corrupted)
    assert any("pending_reason" in error for error in checker.current_quality_resource_errors(tmp_path))
    corrupted = json.loads(json.dumps(records))
    signatures = corrupted["model-signatures.json"]["signatures"]
    signatures["other"] = signatures.pop(first)
    signatures["other"]["id"] = "other"
    write_records(corrupted)
    assert any("signature IDs differ" in error for error in checker.current_quality_resource_errors(tmp_path))
    for name in records:
        write_records(records)
        (directory / name).unlink()
        assert any(f"cannot read {name}" in error for error in checker.current_quality_resource_errors(tmp_path))
        write_records(records)
        (directory / name).write_text("{")
        assert any(f"cannot read {name}" in error for error in checker.current_quality_resource_errors(tmp_path))
        (directory / name).write_text("[]")
        assert any(f"{name} must contain an object" in error for error in checker.current_quality_resource_errors(tmp_path))

    write_records(records)
    (tmp_path / "index.html").write_text("<html><body>Verified fixture</body></html>")
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "ALGORITHMS", [])
    # This synthetic tree exercises the quality gate's main() integration.
    # Independent preview and frozen-bundle gates have complete fixtures.
    monkeypatch.setattr(checker, "verify_preview", lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "site_word_record_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_suzuki_record_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_gibbs_channel_record_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_decay_record_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_tail_record_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_dqi_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_bath_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_multiproduct_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_bayesian_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_robust_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_qpe_feedback_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_dqi_structured_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_dqi_resource_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_local_spin_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_local_spin_reproduction_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_measured_and_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_hamiltonian_packet_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "MEASURED_AND_FIGURE_LABELS", frozenset())
    monkeypatch.setattr(checker, "verify_n3_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "verify_n2_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "verify_n4_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "verify_port_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "verify_qdrift_bundle", lambda *args, **kwargs: {})
    hadamard_calls = []
    def mocked_hadamard_record(record_dir, *, current_source):
        hadamard_calls.append((record_dir, current_source))
        return {}
    monkeypatch.setattr(checker, "verify_hadamard_record", mocked_hadamard_record)
    monkeypatch.setattr(checker, "verify_opt7_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "N3_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "N4_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "PORT_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "QDRIFT_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "SHOR_RECOVERY_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "HADAMARD_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "BAYESIAN_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "ROBUST_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "QPE_FEEDBACK_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "OPT8_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "SUZUKI_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "GIBBS_CHANNEL_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(checker, "DECAY_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(checker, "CORRECTION_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(checker, "site_energy_record_errors", lambda root: [])
    monkeypatch.setattr(checker, "site_dynamics_record_errors", lambda root: [])
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(tmp_path)])
    checker.main()
    assert hadamard_calls == [
        (tmp_path / "assets/generated/hadamard-qpe", tmp_path / "code/src/openquantum/hadamard_qpe.py"),
    ]
    records["manifest.json"]["status"] = "pending_physical_layer"
    write_records(records)
    with pytest.raises(SystemExit, match="manifest status"):
        checker.main()


@pytest.fixture
def manifest_package(tmp_path):
    package = tmp_path / "package"
    (package / "src").mkdir(parents=True)
    (package / "src/example.py").write_text("print('original')\n")
    (package / "measurement.json").write_text('{"shots": 256}\n')
    hashes = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in ("src/example.py", "measurement.json")
    }
    provenance = {
        "schema": "openquantum-public-package-v2", "source_commit": None,
        "file_sha256": hashes,
        "source_sha256": {"src/example.py": hashes["src/example.py"]},
    }
    (package / "PROVENANCE.json").write_text(json.dumps(provenance))
    return package


def test_full_manifest_accepts_original_recorded_inputs(manifest_package):
    assert verify.verify_manifest(manifest_package)["source_commit"] is None


@pytest.mark.parametrize("name", ["src/example.py", "measurement.json"])
def test_full_manifest_rejects_changed_source_or_measurements(manifest_package, name):
    (manifest_package / name).write_text("changed bytes\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify.verify_manifest(manifest_package)


def test_full_manifest_rejects_missing_or_unrecorded_files(manifest_package):
    (manifest_package / "measurement.json").unlink()
    with pytest.raises(ValueError, match="contents"):
        verify.verify_manifest(manifest_package)
    (manifest_package / "extra.txt").write_text("unrecorded")
    with pytest.raises(ValueError, match="contents"):
        verify.verify_manifest(manifest_package)


@pytest.fixture
def formal_package(tmp_path):
    for relative in ("algorithms/optimization/formal", "assets/generated/optimization/formal"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path


def test_recorded_classical_lean_sources_match_both_published_copies(formal_package):
    verify.verify_formal_record(formal_package)


@pytest.mark.parametrize("directory", ["algorithms/optimization/formal", "assets/generated/optimization/formal"])
@pytest.mark.parametrize("name", ["C4Reference.lean", "lean-toolchain", "check.py", "LICENSE.txt", "README.md"])
def test_formal_identity_check_rejects_any_changed_recorded_source(formal_package, directory, name):
    path = formal_package / directory / name
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify.verify_formal_record(formal_package)


@pytest.mark.parametrize("key,value", [
    ("status", "planned"), ("exit_code", 1), ("kernel_checked_closed_propositions", False),
    ("native_decide_used", True), ("toolchain", "leanprover/lean4:stable"),
    ("scope", "Full quantum algorithm correctness"), ("exclusions", []),
    ("axioms_by_theorem", {"all_cuts_le_four": ["sorryAx"]}),
    ("source_sha256", {}),
])
def test_formal_identity_check_rejects_incomplete_or_different_check_record(formal_package, key, value):
    path = formal_package / "assets/generated/optimization/formal/verification.json"
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Lean"):
        verify.verify_formal_record(formal_package)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "openquantum-examples/../../escape.py", "wrong-root/example.py"])
def test_archive_extraction_rejects_paths_outside_the_public_package(tmp_path, name):
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(name, b"changed")
    with pytest.raises(ValueError, match="Unsafe"):
        verify.extract_package(archive, tmp_path / "output")


def test_archive_extraction_rejects_symlink_members(tmp_path):
    archive = tmp_path / "package.zip"
    member = zipfile.ZipInfo("openquantum-examples/source.py")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr(member, b"/outside/source.py")
    with pytest.raises(ValueError, match="Unsafe"):
        verify.extract_package(archive, tmp_path / "output")
