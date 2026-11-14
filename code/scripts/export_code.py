#!/usr/bin/env python3
"""Export deliberate public examples without private project files."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_provenance import git_revision
from verify_simulation_lindblad_port import BUNDLE_FILES as LINDBLAD_PORT_FILES


PUBLIC_SCRIPTS = (
    "generate_grover.py", "estimate_resources.py", "generate_grover_scaling.py",
    "generate_qft_qpe.py", "generate_amplitude_estimation.py", "generate_amplitude_iterative.py",
    "generate_qft_variant_comparison.py",
    "generate_hadamard_qpe.py", "verify_hadamard_qpe.py",
    "generate_bayesian_qpe.py", "verify_bayesian_qpe_saved.py",
    "generate_robust_qpe.py", "verify_robust_qpe_saved.py",
    "generate_qpe_feedback_resources.py", "verify_qpe_feedback_saved.py", "verify_qpe_feedback_public.py",
    "generate_shor.py", "generate_shor_driver.py", "generate_shor_ripple.py",
    "generate_shor_scaling.py",
    "generate_shor_recovery.py",
    "generate_shor_measured_and.py", "verify_shor_measured_and_saved.py",
    "generate_linear_systems.py", "generate_linear_systems_qsvt.py", "generate_linear_inverse_lcu.py",
    "generate_linear_systems_qsvt_reciprocal.py",
    "generate_linear_systems_physical.py",
    "generate_reciprocal_physical_sweep.py",
    "generate_hamiltonian_simulation.py", "generate_simulation_lcu.py", "generate_simulation_qsp.py",
    "generate_simulation_qdrift.py", "verify_simulation_qdrift.py",
    "generate_simulation_suzuki.py", "verify_simulation_suzuki.py",
    "generate_detailed_balance_access.py", "generate_detailed_balance_channel.py",
    "verify_detailed_balance_channel.py",
    "generate_simulation_qsp_evolution.py", "generate_simulation_qsp_scaling.py",
    "generate_simulation_retargetable_qsp.py", "generate_simulation_thermal_spectral.py",
    "generate_simulation_complete_resources.py",
    "generate_simulation_lindblad_physical.py",
    "verify_simulation_lindblad_physical.py",
    "generate_simulation_lindblad_three.py", "verify_simulation_lindblad_three.py",
    "generate_simulation_lindblad_four.py", "verify_simulation_lindblad_four.py",
    "run_simulation_lindblad_port.py", "verify_simulation_lindblad_port.py",
    "generate_simulation_word_taylor.py", "generate_simulation_amplification.py",
    "generate_word_taylor_workloads.py",
    "generate_word_taylor_physical.py",
    "generate_simulation_open_systems.py", "generate_simulation_lindblad.py",
    "generate_simulation_two_spin_lindblad.py", "generate_simulation_gibbs.py",
    "generate_simulation_lindblad_workload.py",
    "generate_simulation_thermal_filter.py", "generate_simulation_preparation_amplification.py",
    "generate_optimization.py", "generate_optimization_sampled.py", "generate_optimization_projections.py",
    "generate_optimization_trajectory_resources.py", "generate_optimization_portability.py",
    "generate_optimization_annealing.py", "generate_optimization_walk.py",
    "generate_optimization_annealing_walk_resources.py", "generate_quantum_walks.py",
    "generate_optimization_quality_scaling.py", "generate_optimization_quality_resources.py",
    "generate_optimization_diagnostics.py",
    "generate_optimization_compiled.py", "complete_optimization_compiled.py", "verify_optimization_compiled.py",
    "verify_optimization_diagnostics.py",
    "prepare_optimization_word_transport.py", "run_optimization_word_native.py",
    "optimization_word_records.py", "optimization_word_record_schema.py",
    "optimization_word_record_release.py", "verify_optimization_word_records.py",
    "generate_optimization_word_report.py",
    "verify_optimization_checkpoint.py",
    "generate_energy_estimation.py", "generate_energy_estimation_pauli.py",
    "verify_energy_estimation.py", "freeze_energy_estimation.py",
    "energy_physical_records.py", "energy_record_release.py",
    "generate_grover_sat.py", "generate_dynamics_resources.py",
    "verify_dynamics_resources.py", "freeze_dynamics_resources.py",
    "dynamics_physical_records.py", "dynamics_record_release.py",
    "dynamics_canonical_replay.py", "dynamics_source_transport.py", "tail_resource_qasm.py",
    "generate_fixed_point_amplification.py", "generate_instrument_resources.py",
    "source_provenance.py", "normalize_figures.py", "execute_notebooks.py",
    "export_code.py", "verify_public_package.py", "verify_frozen_baseline.py",
    "build_book.py", "public_assets.py",
    "compact_site.py", "public_code_preview.py",
    "tail_resource_records.py", "tail_resource_model.py", "tail_resource_replay.py",
    "tail_resource_numerical.py", "freeze_tail_resource_records.py",
    "verify_instrument_resource_records.py", "verify_simulation_complete_resource_records.py",
    "generate_optimization_dqi.py", "verify_dqi_saved.py",
    "generate_optimization_dqi_structured.py", "validate_optimization_dqi_structured.py",
    "verify_dqi_structured_saved.py",
    "generate_optimization_dqi_resources.py", "verify_dqi_structured_resource_saved.py",
    "generate_simulation_multiproduct.py", "verify_simulation_multiproduct_saved.py",
    "render_reusable_bath_local_spin.py", "verify_reusable_bath_local_spin_saved.py",
)

# These algorithm output trees contain the reviewed data and source snapshots
# needed by the tests, notebooks, and recorded resource/QASM hash checks.
# A publication record or recursively included code ZIP is never an input.
PUBLIC_ARTIFACT_TREES = (
    "grover", "grover-scaling", "resources", "qft-qpe", "hadamard-qpe", "bayesian-qpe", "bayesian-qpe-validation", "robust-qpe", "qpe-feedback-resources", "qpe-feedback-validation", "amplitude-estimation",
    "amplitude-iterative", "shor", "linear-systems", "hamiltonian-simulation",
    "simulation", "optimization", "quantum-walks", "energy-estimation", "qft-variant-comparison",
    "grover-sat", "dynamics-resources", "fixed-point-amplification", "instrument-resources",
    "dynamics-recorded-source",
    "dynamics-replay-evidence",
    "word-taylor-workloads", "word-taylor-physical", "shor-scaling",
)
ARTIFACT_SUFFIXES = frozenset({".py", ".qasm", ".json", ".csv", ".svg", ".md"})
PUBLIC_ARCHIVES = frozenset({"assets/generated/shor/measured-and-artifacts.zip",
                             "assets/generated/grover-sat/iteration-circuits.zip",
                             "assets/generated/simulation/reusable-bath-evidence.zip",
                             "assets/generated/simulation/reusable-bath-local-spin-evidence.zip",
                             "assets/generated/simulation/multiproduct-bundle.zip",
                             "assets/generated/simulation/multiproduct-large-bundle.zip"})
PUBLIC_DEPENDENCY_FILES = frozenset({
    "assets/generated/optimization/dqi-structured-resources/LICENSE.txt",
    "assets/generated/optimization/dqi-structured-resources/requirements-lock.txt",
    "assets/generated/shor/measured-and/requirements-lock.txt",
    "assets/generated/shor/measured-and/LICENSE.txt",
    "assets/generated/qpe-feedback-resources/requirements-lock.txt",
    "assets/generated/qpe-feedback-resources/LICENSE.txt",
    "assets/generated/simulation/reusable-bath-local-spin/source/requirements-lock.txt",
    "assets/generated/simulation/reusable-bath-local-spin/source/LICENSE.txt",
    "assets/generated/robust-qpe/requirements-lock.txt",
    "assets/generated/robust-qpe/LICENSE.txt",
    "assets/generated/bayesian-qpe/requirements-lock.txt",
    "assets/generated/bayesian-qpe/LICENSE.txt",
    "assets/generated/bayesian-qpe-validation/source32/requirements-lock.txt",
    "assets/generated/bayesian-qpe-validation/source32/LICENSE.txt",
    "assets/generated/bayesian-qpe-validation/replay/requirements-lock.txt",
    "assets/generated/bayesian-qpe-validation/replay/LICENSE.txt",
    "assets/generated/simulation/multiproduct/requirements-lock.txt",
    "assets/generated/simulation/multiproduct/LICENSE.txt",
    "assets/generated/simulation/multiproduct-large/requirements-lock.txt",
    "assets/generated/simulation/multiproduct-large/LICENSE.txt",
    "assets/generated/simulation/reusable-bath/source/requirements-lock.txt",
    "assets/generated/simulation/reusable-bath/source/LICENSE.txt",
    "assets/generated/simulation/reusable-bath/source/PROTOCOL-NOTE.md",
    "assets/generated/optimization/dqi/requirements-lock.txt",
    "assets/generated/optimization/dqi/LICENSE.txt",
    "assets/generated/optimization/dqi/optimization_dqi.ipynb",
    "assets/generated/optimization/dqi-structured/requirements-lock.txt",
    "assets/generated/optimization/dqi-structured/LICENSE.txt",
    "assets/generated/optimization/dqi-structured/large/requirements-lock.txt",
    "assets/generated/optimization/dqi-structured/large/LICENSE.txt",
    "assets/generated/simulation/detailed-balance-channel/requirements-lock.txt",
    "assets/generated/hadamard-qpe/requirements-lock.txt",
    "assets/generated/shor/recovery/Quaspy-LICENSE.txt",
    "assets/generated/instrument-resources/requirements-lock.txt",
    "assets/generated/instrument-resources/references.bib",
    "assets/generated/optimization/diagnostics/requirements-lock.txt",
    "assets/generated/optimization/compiled-replay/requirements-lock.txt",
    "assets/generated/optimization/compiled-replay/references.bib",
    "assets/generated/optimization/word-transport/requirements-lock.txt",
    "assets/generated/optimization/word-transport/references.bib",
    "assets/generated/word-taylor-workloads/requirements-lock.txt",
    "assets/generated/word-taylor-physical/requirements-lock.txt",
    "assets/generated/word-taylor-physical/references.bib",
    "assets/generated/simulation/lindblad-workload/requirements-lock.txt",
    "assets/generated/simulation/lindblad-workload/references.bib",
    "assets/generated/simulation/lindblad-physical/requirements-lock.txt",
    "assets/generated/simulation/lindblad-physical/references.bib",
    "assets/generated/simulation/lindblad-three/requirements-lock.txt",
    "assets/generated/simulation/lindblad-three/references.bib",
    "assets/generated/simulation/lindblad-four/requirements-lock.txt",
    "assets/generated/simulation/lindblad-four/references.bib",
    "assets/generated/simulation/lindblad-port/requirements-lock.txt",
    "assets/generated/simulation/lindblad-port/references.bib",
    "assets/generated/simulation/complete-resources/requirements-lock.txt",
    "assets/generated/simulation/complete-resources/references.bib",
    "assets/generated/linear-systems/physical-sweeps/code/requirements-lock.txt",
    "assets/generated/linear-systems/physical-sweeps/code/citations.bib",
    "assets/generated/linear-systems/physical-sweeps/code/references/simulation.bib",
    "assets/generated/linear-systems/physical-sweeps/code/references/resources.bib",
    "assets/generated/linear-systems/physical-sweeps/code/references/linear-systems.bib",
    "assets/generated/linear-systems/physical-sweeps/code/references/amplification.bib",
})
# Exact compressed inputs from the reviewed 301-file compiled-replay bundle.
# New gzip files require an explicit reviewed registration.
PUBLIC_COMPRESSED_JSON_FILES = frozenset({
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/022bc4e612330e512612e741e8d37e71dae9beb51214b08e4d0d3d826b5e7716.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/0b49afcc07950445e3090a0d3385a37dcb6e81a64e5702522e8822ac6cc7000f.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/0e1b386c66fa83756c964e5cec7f213aa2209b1e811df74d021f85f21e921822.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/0e38baa8f9556fc33364365edc104bcdceaaf1676a6d21b3afd93031c21fdf31.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/13517cb73a0d157af136bcd7614928ecfb95416afda1975246c557a4fc9ce75a.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/183bee4820cc9db9160bd5d91d37e3c79e52495e47f3dacf7decec0df26538b7.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/1d11ccc11221636a165999a7629c3e9f613973d82fb70d276c4a7f5960a79c66.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/1db5e92725d8d9474198cce47cb104cb3991e6349b8620a86e8f5a1bcb93cc75.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/1ecb125df99753b77404b64697a7d5271471fad450eeb32599d7b2e1346611b2.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/2049ffeb8aae6985917a7f889ac6f647615e4f967f6fcfa59f8ffef293389e0c.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/2f8e806dda25fc7e8f5a4b4222493ed988b0ba60b5ea071e29f7493df1c25baf.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/439605dbdb9d5902c9b8e85d2f556eb3df5b0920925815fc23a7d9ab4639a986.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/4400b42c551aab4267c905a52463f77eb314d6ec7dc24af67588b4f8607b5a47.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/44eaf8c927344805b416c7431451cef2143579b0dab91788ddce3e7746fa8cc6.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/4c7779db4942f82ee77bf118a637a3875289aef4b3a69b66799c6c89aa42a380.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/4ff5ef50478c6da2610cf206856efeafc2eac8b4bcec589d8658be5c7a941365.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/51a245f482a4f3c7badd7720ae30b28c7a404ae7160714cdd55d515f22e9ee7f.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/550e032e8bdb453809fac762c567824c564b9ae3c9091cf9185725b48aa0507e.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/5623940e635764c58e29118e2d965f97d1e991db7f3e82756d5c4cde4ef68e2e.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/5b138994f07d2fc3fd5a55c313e312ee9e293582d2361f8c52ceb23123ff315e.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/6167ca4cc2931823c284778476fa83ae31df547d559a06b8ca4017c098506b06.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/66966a894ddedc83494653782892819c275b7834b1abf27d0bfe777d28e5341b.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/742d6fcd3a7236e61a32b2cc7b1d37f16db1874e0569c5901c15c9d46f0380af.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/774f19cf7fb51fb541fa59ed7234845281f4f359570cf8029631ec4596d1ed44.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/80380412a7023080af405ecd115be27abf42f63f5acdb2183cc2481ad3639ebc.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/86e1e5f849ff221d026d404ab98826fc0852febaf194f83731bd4089f85cc3de.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/901bc9d8ea8952aff5d725a0c63623ab5314f3e42105ee4261342b1006a2069a.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/91093d828e1796df46811716a41f8ddc2a973c2eb79fc6a77818a7bd8f91a34d.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/9d464cd381e4b3ceee6dde197b3144852d1b96072a089d9ff3d8b85372c1719c.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/a5d71b83ebf11631c344ca68c021a21846b8db682152c634634371af40e30de9.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/ae6b115f895e84c79f6205b33bd1b3ca29921081178ce6dd535a2a73ccdabf25.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/b3b088059765455b97b3c1e47a481f0515f4c22dd73951c6836827bab9ecbd83.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/b69e67ce0ca023f7a952396f6d2eff3e9bf171aa1c7b37e4d4732585fc2d2408.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/b8c61cc319f67f066ed734ff411f6b511dc181a18a7a0d48d35d3b5fae43d4cb.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/bc0bcf17fcb5a4861c2cee7f5e19a41b93266211acc30d7641dd8809d6f7f7bc.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/bed3b72756d275b4811edc8a0e986645faaeae8e3b2195f2ddd6966684e2a05a.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/bed7fcf00e412d5d511fac57e5adbd19ebfe1dddfa702f7ea62b09209d4ac435.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/c09b6df6291b1b8401978e889f48d78d9478eff468091a01fe9603ecdda5637a.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/c479c9e8ef56c684cd656cf250e02765c5ad6b866e4e68ad1cf45026f14dcd50.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/c6f5c27e7fe1712b35dea11e55d5bb544123fea9aee970e040c686a3543034d9.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/c9baae512c6dfdc94e19e7acfa9d8375a55695f24531e3f8771c1b4b8d2b0b71.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/cd867fceafe96b434398881b8bdfb397aa63f34c6d3570af57ea01ab39c26826.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/d2be017f42415c07acef481014b1c47ee887743765360ad8a84aeb7234fd8342.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/d575f94c35ee5fdcd65ca2cccbffeb0801e4c11b95ba80bcf6af9cb7cd44bd14.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/df262b18ddac25476536fbcb4be9c6c50fce9152e1576904517568b6b8c44f84.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/e162c55393e6ac1be43b157f4fd0566a40cd1d19e939ae600c5fda8b4eed3847.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/e1c9f761f046559de2b7d595cf47d025f4ff7765ecc8ab0aa276d11b1610fab3.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/e52e94b18e3baa061cd58154b81a831f5e1aa968e0f9a65425b2a6f567459946.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/e59cd31db0fea68aaecec71d4d4806e4dbca492c5b84e2c3bca6b6f7e6352fb8.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/edbd80dcaf25795af29eee2d5dd59ef31d144648cff93267564b1419486d5251.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/f423d9ec800c52460edba258050faa9f15d27ebd0eb99fd427237fd223e73a40.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/f5373d39f0da20198e832cfc7933956b92abee49590a386ecc8d0643e4168b6b.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/fb7c9e22b7c2d0cf3e22282f35f9ce9ca761c00bccbe9ec6771eec08d26257ed.json.gz',
    'assets/generated/optimization/compiled-replay/checkpoints/synthesis/ff0c16b82fb8e7ab06b758bd9898fab810e7a1f2a4b5686f8a74eb7b4b3c2ca1.json.gz',
    'assets/generated/optimization/compiled-replay/completed-replay.json.gz',
    'assets/generated/optimization/compiled-replay/physical-dag.json.gz',
    'assets/generated/optimization/compiled-replay/source-classes.json.gz',
})
# Exact immutable public files for the reviewed saved Native word comparison.
# Later files, including operational state, are not inferred from their suffix.
PUBLIC_OPTIMIZATION_WORD_FILES = frozenset({
    'assets/generated/optimization/word-transport/LICENSE.txt',
    'assets/generated/optimization/word-transport/README.md',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/022bc4e612330e512612e741e8d37e71dae9beb51214b08e4d0d3d826b5e7716.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/0b49afcc07950445e3090a0d3385a37dcb6e81a64e5702522e8822ac6cc7000f.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/0e1b386c66fa83756c964e5cec7f213aa2209b1e811df74d021f85f21e921822.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/0e38baa8f9556fc33364365edc104bcdceaaf1676a6d21b3afd93031c21fdf31.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/13517cb73a0d157af136bcd7614928ecfb95416afda1975246c557a4fc9ce75a.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/183bee4820cc9db9160bd5d91d37e3c79e52495e47f3dacf7decec0df26538b7.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/1d11ccc11221636a165999a7629c3e9f613973d82fb70d276c4a7f5960a79c66.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/1db5e92725d8d9474198cce47cb104cb3991e6349b8620a86e8f5a1bcb93cc75.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/1ecb125df99753b77404b64697a7d5271471fad450eeb32599d7b2e1346611b2.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/2049ffeb8aae6985917a7f889ac6f647615e4f967f6fcfa59f8ffef293389e0c.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/2f8e806dda25fc7e8f5a4b4222493ed988b0ba60b5ea071e29f7493df1c25baf.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/439605dbdb9d5902c9b8e85d2f556eb3df5b0920925815fc23a7d9ab4639a986.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/4400b42c551aab4267c905a52463f77eb314d6ec7dc24af67588b4f8607b5a47.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/44eaf8c927344805b416c7431451cef2143579b0dab91788ddce3e7746fa8cc6.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/4c7779db4942f82ee77bf118a637a3875289aef4b3a69b66799c6c89aa42a380.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/4ff5ef50478c6da2610cf206856efeafc2eac8b4bcec589d8658be5c7a941365.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/51a245f482a4f3c7badd7720ae30b28c7a404ae7160714cdd55d515f22e9ee7f.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/550e032e8bdb453809fac762c567824c564b9ae3c9091cf9185725b48aa0507e.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/5623940e635764c58e29118e2d965f97d1e991db7f3e82756d5c4cde4ef68e2e.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/5b138994f07d2fc3fd5a55c313e312ee9e293582d2361f8c52ceb23123ff315e.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/6167ca4cc2931823c284778476fa83ae31df547d559a06b8ca4017c098506b06.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/66966a894ddedc83494653782892819c275b7834b1abf27d0bfe777d28e5341b.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/742d6fcd3a7236e61a32b2cc7b1d37f16db1874e0569c5901c15c9d46f0380af.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/774f19cf7fb51fb541fa59ed7234845281f4f359570cf8029631ec4596d1ed44.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/80380412a7023080af405ecd115be27abf42f63f5acdb2183cc2481ad3639ebc.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/86e1e5f849ff221d026d404ab98826fc0852febaf194f83731bd4089f85cc3de.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/901bc9d8ea8952aff5d725a0c63623ab5314f3e42105ee4261342b1006a2069a.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/91093d828e1796df46811716a41f8ddc2a973c2eb79fc6a77818a7bd8f91a34d.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/9d464cd381e4b3ceee6dde197b3144852d1b96072a089d9ff3d8b85372c1719c.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/a5d71b83ebf11631c344ca68c021a21846b8db682152c634634371af40e30de9.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/ae6b115f895e84c79f6205b33bd1b3ca29921081178ce6dd535a2a73ccdabf25.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/b3b088059765455b97b3c1e47a481f0515f4c22dd73951c6836827bab9ecbd83.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/b69e67ce0ca023f7a952396f6d2eff3e9bf171aa1c7b37e4d4732585fc2d2408.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/b8c61cc319f67f066ed734ff411f6b511dc181a18a7a0d48d35d3b5fae43d4cb.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/bc0bcf17fcb5a4861c2cee7f5e19a41b93266211acc30d7641dd8809d6f7f7bc.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/bed3b72756d275b4811edc8a0e986645faaeae8e3b2195f2ddd6966684e2a05a.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/bed7fcf00e412d5d511fac57e5adbd19ebfe1dddfa702f7ea62b09209d4ac435.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/c09b6df6291b1b8401978e889f48d78d9478eff468091a01fe9603ecdda5637a.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/c479c9e8ef56c684cd656cf250e02765c5ad6b866e4e68ad1cf45026f14dcd50.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/c6f5c27e7fe1712b35dea11e55d5bb544123fea9aee970e040c686a3543034d9.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/c9baae512c6dfdc94e19e7acfa9d8375a55695f24531e3f8771c1b4b8d2b0b71.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/cd867fceafe96b434398881b8bdfb397aa63f34c6d3570af57ea01ab39c26826.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/d2be017f42415c07acef481014b1c47ee887743765360ad8a84aeb7234fd8342.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/d575f94c35ee5fdcd65ca2cccbffeb0801e4c11b95ba80bcf6af9cb7cd44bd14.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/df262b18ddac25476536fbcb4be9c6c50fce9152e1576904517568b6b8c44f84.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/e162c55393e6ac1be43b157f4fd0566a40cd1d19e939ae600c5fda8b4eed3847.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/e1c9f761f046559de2b7d595cf47d025f4ff7765ecc8ab0aa276d11b1610fab3.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/e52e94b18e3baa061cd58154b81a831f5e1aa968e0f9a65425b2a6f567459946.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/e59cd31db0fea68aaecec71d4d4806e4dbca492c5b84e2c3bca6b6f7e6352fb8.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/edbd80dcaf25795af29eee2d5dd59ef31d144648cff93267564b1419486d5251.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/f423d9ec800c52460edba258050faa9f15d27ebd0eb99fd427237fd223e73a40.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/f5373d39f0da20198e832cfc7933956b92abee49590a386ecc8d0643e4168b6b.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/fb7c9e22b7c2d0cf3e22282f35f9ce9ca761c00bccbe9ec6771eec08d26257ed.json.gz',
    'assets/generated/optimization/word-transport/baseline/checkpoints/synthesis/ff0c16b82fb8e7ab06b758bd9898fab810e7a1f2a4b5686f8a74eb7b4b3c2ca1.json.gz',
    'assets/generated/optimization/word-transport/baseline/manifest.json',
    'assets/generated/optimization/word-transport/baseline/negative-controls.csv',
    'assets/generated/optimization/word-transport/baseline/physical-dag.json.gz',
    'assets/generated/optimization/word-transport/baseline/recipe.json',
    'assets/generated/optimization/word-transport/candidate/circuits/022bc4e612330e512612e741e8d37e71dae9beb51214b08e4d0d3d826b5e7716.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/022bc4e612330e512612e741e8d37e71dae9beb51214b08e4d0d3d826b5e7716.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/0b49afcc07950445e3090a0d3385a37dcb6e81a64e5702522e8822ac6cc7000f.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/0b49afcc07950445e3090a0d3385a37dcb6e81a64e5702522e8822ac6cc7000f.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/0e1b386c66fa83756c964e5cec7f213aa2209b1e811df74d021f85f21e921822.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/0e1b386c66fa83756c964e5cec7f213aa2209b1e811df74d021f85f21e921822.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/0e38baa8f9556fc33364365edc104bcdceaaf1676a6d21b3afd93031c21fdf31.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/0e38baa8f9556fc33364365edc104bcdceaaf1676a6d21b3afd93031c21fdf31.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/13517cb73a0d157af136bcd7614928ecfb95416afda1975246c557a4fc9ce75a.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/13517cb73a0d157af136bcd7614928ecfb95416afda1975246c557a4fc9ce75a.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/183bee4820cc9db9160bd5d91d37e3c79e52495e47f3dacf7decec0df26538b7.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/183bee4820cc9db9160bd5d91d37e3c79e52495e47f3dacf7decec0df26538b7.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/1d11ccc11221636a165999a7629c3e9f613973d82fb70d276c4a7f5960a79c66.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/1d11ccc11221636a165999a7629c3e9f613973d82fb70d276c4a7f5960a79c66.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/1db5e92725d8d9474198cce47cb104cb3991e6349b8620a86e8f5a1bcb93cc75.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/1db5e92725d8d9474198cce47cb104cb3991e6349b8620a86e8f5a1bcb93cc75.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/1ecb125df99753b77404b64697a7d5271471fad450eeb32599d7b2e1346611b2.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/1ecb125df99753b77404b64697a7d5271471fad450eeb32599d7b2e1346611b2.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/2049ffeb8aae6985917a7f889ac6f647615e4f967f6fcfa59f8ffef293389e0c.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/2049ffeb8aae6985917a7f889ac6f647615e4f967f6fcfa59f8ffef293389e0c.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/2f8e806dda25fc7e8f5a4b4222493ed988b0ba60b5ea071e29f7493df1c25baf.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/2f8e806dda25fc7e8f5a4b4222493ed988b0ba60b5ea071e29f7493df1c25baf.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/439605dbdb9d5902c9b8e85d2f556eb3df5b0920925815fc23a7d9ab4639a986.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/439605dbdb9d5902c9b8e85d2f556eb3df5b0920925815fc23a7d9ab4639a986.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/4400b42c551aab4267c905a52463f77eb314d6ec7dc24af67588b4f8607b5a47.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/4400b42c551aab4267c905a52463f77eb314d6ec7dc24af67588b4f8607b5a47.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/44eaf8c927344805b416c7431451cef2143579b0dab91788ddce3e7746fa8cc6.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/44eaf8c927344805b416c7431451cef2143579b0dab91788ddce3e7746fa8cc6.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/4c7779db4942f82ee77bf118a637a3875289aef4b3a69b66799c6c89aa42a380.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/4c7779db4942f82ee77bf118a637a3875289aef4b3a69b66799c6c89aa42a380.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/4ff5ef50478c6da2610cf206856efeafc2eac8b4bcec589d8658be5c7a941365.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/4ff5ef50478c6da2610cf206856efeafc2eac8b4bcec589d8658be5c7a941365.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/51a245f482a4f3c7badd7720ae30b28c7a404ae7160714cdd55d515f22e9ee7f.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/51a245f482a4f3c7badd7720ae30b28c7a404ae7160714cdd55d515f22e9ee7f.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/550e032e8bdb453809fac762c567824c564b9ae3c9091cf9185725b48aa0507e.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/550e032e8bdb453809fac762c567824c564b9ae3c9091cf9185725b48aa0507e.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/5623940e635764c58e29118e2d965f97d1e991db7f3e82756d5c4cde4ef68e2e.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/5623940e635764c58e29118e2d965f97d1e991db7f3e82756d5c4cde4ef68e2e.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/5b138994f07d2fc3fd5a55c313e312ee9e293582d2361f8c52ceb23123ff315e.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/5b138994f07d2fc3fd5a55c313e312ee9e293582d2361f8c52ceb23123ff315e.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/6167ca4cc2931823c284778476fa83ae31df547d559a06b8ca4017c098506b06.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/6167ca4cc2931823c284778476fa83ae31df547d559a06b8ca4017c098506b06.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/66966a894ddedc83494653782892819c275b7834b1abf27d0bfe777d28e5341b.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/66966a894ddedc83494653782892819c275b7834b1abf27d0bfe777d28e5341b.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/742d6fcd3a7236e61a32b2cc7b1d37f16db1874e0569c5901c15c9d46f0380af.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/742d6fcd3a7236e61a32b2cc7b1d37f16db1874e0569c5901c15c9d46f0380af.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/774f19cf7fb51fb541fa59ed7234845281f4f359570cf8029631ec4596d1ed44.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/774f19cf7fb51fb541fa59ed7234845281f4f359570cf8029631ec4596d1ed44.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/80380412a7023080af405ecd115be27abf42f63f5acdb2183cc2481ad3639ebc.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/80380412a7023080af405ecd115be27abf42f63f5acdb2183cc2481ad3639ebc.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/86e1e5f849ff221d026d404ab98826fc0852febaf194f83731bd4089f85cc3de.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/86e1e5f849ff221d026d404ab98826fc0852febaf194f83731bd4089f85cc3de.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/901bc9d8ea8952aff5d725a0c63623ab5314f3e42105ee4261342b1006a2069a.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/901bc9d8ea8952aff5d725a0c63623ab5314f3e42105ee4261342b1006a2069a.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/91093d828e1796df46811716a41f8ddc2a973c2eb79fc6a77818a7bd8f91a34d.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/91093d828e1796df46811716a41f8ddc2a973c2eb79fc6a77818a7bd8f91a34d.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/9d464cd381e4b3ceee6dde197b3144852d1b96072a089d9ff3d8b85372c1719c.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/9d464cd381e4b3ceee6dde197b3144852d1b96072a089d9ff3d8b85372c1719c.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/a5d71b83ebf11631c344ca68c021a21846b8db682152c634634371af40e30de9.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/a5d71b83ebf11631c344ca68c021a21846b8db682152c634634371af40e30de9.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/ae6b115f895e84c79f6205b33bd1b3ca29921081178ce6dd535a2a73ccdabf25.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/ae6b115f895e84c79f6205b33bd1b3ca29921081178ce6dd535a2a73ccdabf25.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/b3b088059765455b97b3c1e47a481f0515f4c22dd73951c6836827bab9ecbd83.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/b3b088059765455b97b3c1e47a481f0515f4c22dd73951c6836827bab9ecbd83.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/b69e67ce0ca023f7a952396f6d2eff3e9bf171aa1c7b37e4d4732585fc2d2408.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/b69e67ce0ca023f7a952396f6d2eff3e9bf171aa1c7b37e4d4732585fc2d2408.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/b8c61cc319f67f066ed734ff411f6b511dc181a18a7a0d48d35d3b5fae43d4cb.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/b8c61cc319f67f066ed734ff411f6b511dc181a18a7a0d48d35d3b5fae43d4cb.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/bc0bcf17fcb5a4861c2cee7f5e19a41b93266211acc30d7641dd8809d6f7f7bc.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/bc0bcf17fcb5a4861c2cee7f5e19a41b93266211acc30d7641dd8809d6f7f7bc.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/bed3b72756d275b4811edc8a0e986645faaeae8e3b2195f2ddd6966684e2a05a.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/bed3b72756d275b4811edc8a0e986645faaeae8e3b2195f2ddd6966684e2a05a.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/bed7fcf00e412d5d511fac57e5adbd19ebfe1dddfa702f7ea62b09209d4ac435.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/bed7fcf00e412d5d511fac57e5adbd19ebfe1dddfa702f7ea62b09209d4ac435.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/c09b6df6291b1b8401978e889f48d78d9478eff468091a01fe9603ecdda5637a.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/c09b6df6291b1b8401978e889f48d78d9478eff468091a01fe9603ecdda5637a.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/c479c9e8ef56c684cd656cf250e02765c5ad6b866e4e68ad1cf45026f14dcd50.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/c479c9e8ef56c684cd656cf250e02765c5ad6b866e4e68ad1cf45026f14dcd50.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/c6f5c27e7fe1712b35dea11e55d5bb544123fea9aee970e040c686a3543034d9.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/c6f5c27e7fe1712b35dea11e55d5bb544123fea9aee970e040c686a3543034d9.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/c9baae512c6dfdc94e19e7acfa9d8375a55695f24531e3f8771c1b4b8d2b0b71.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/c9baae512c6dfdc94e19e7acfa9d8375a55695f24531e3f8771c1b4b8d2b0b71.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/cd867fceafe96b434398881b8bdfb397aa63f34c6d3570af57ea01ab39c26826.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/cd867fceafe96b434398881b8bdfb397aa63f34c6d3570af57ea01ab39c26826.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/d2be017f42415c07acef481014b1c47ee887743765360ad8a84aeb7234fd8342.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/d2be017f42415c07acef481014b1c47ee887743765360ad8a84aeb7234fd8342.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/d575f94c35ee5fdcd65ca2cccbffeb0801e4c11b95ba80bcf6af9cb7cd44bd14.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/d575f94c35ee5fdcd65ca2cccbffeb0801e4c11b95ba80bcf6af9cb7cd44bd14.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/df262b18ddac25476536fbcb4be9c6c50fce9152e1576904517568b6b8c44f84.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/df262b18ddac25476536fbcb4be9c6c50fce9152e1576904517568b6b8c44f84.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/e162c55393e6ac1be43b157f4fd0566a40cd1d19e939ae600c5fda8b4eed3847.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/e162c55393e6ac1be43b157f4fd0566a40cd1d19e939ae600c5fda8b4eed3847.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/e1c9f761f046559de2b7d595cf47d025f4ff7765ecc8ab0aa276d11b1610fab3.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/e1c9f761f046559de2b7d595cf47d025f4ff7765ecc8ab0aa276d11b1610fab3.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/e52e94b18e3baa061cd58154b81a831f5e1aa968e0f9a65425b2a6f567459946.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/e52e94b18e3baa061cd58154b81a831f5e1aa968e0f9a65425b2a6f567459946.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/e59cd31db0fea68aaecec71d4d4806e4dbca492c5b84e2c3bca6b6f7e6352fb8.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/e59cd31db0fea68aaecec71d4d4806e4dbca492c5b84e2c3bca6b6f7e6352fb8.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/edbd80dcaf25795af29eee2d5dd59ef31d144648cff93267564b1419486d5251.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/edbd80dcaf25795af29eee2d5dd59ef31d144648cff93267564b1419486d5251.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/f423d9ec800c52460edba258050faa9f15d27ebd0eb99fd427237fd223e73a40.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/f423d9ec800c52460edba258050faa9f15d27ebd0eb99fd427237fd223e73a40.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/f5373d39f0da20198e832cfc7933956b92abee49590a386ecc8d0643e4168b6b.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/f5373d39f0da20198e832cfc7933956b92abee49590a386ecc8d0643e4168b6b.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/fb7c9e22b7c2d0cf3e22282f35f9ce9ca761c00bccbe9ec6771eec08d26257ed.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/fb7c9e22b7c2d0cf3e22282f35f9ce9ca761c00bccbe9ec6771eec08d26257ed.qasm',
    'assets/generated/optimization/word-transport/candidate/circuits/ff0c16b82fb8e7ab06b758bd9898fab810e7a1f2a4b5686f8a74eb7b4b3c2ca1.json.gz',
    'assets/generated/optimization/word-transport/candidate/circuits/ff0c16b82fb8e7ab06b758bd9898fab810e7a1f2a4b5686f8a74eb7b4b3c2ca1.qasm',
    'assets/generated/optimization/word-transport/classes.csv',
    'assets/generated/optimization/word-transport/code/optimization_compiled_codec.py',
    'assets/generated/optimization/word-transport/code/optimization_word_accounting.py',
    'assets/generated/optimization/word-transport/code/optimization_word_native.py',
    'assets/generated/optimization/word-transport/code/optimization_word_record_schema.py',
    'assets/generated/optimization/word-transport/code/optimization_word_records.py',
    'assets/generated/optimization/word-transport/code/optimization_word_transport.py',
    'assets/generated/optimization/word-transport/comparison.csv',
    'assets/generated/optimization/word-transport/execution-summary.json',
    'assets/generated/optimization/word-transport/native-inputs.json.gz',
    'assets/generated/optimization/word-transport/native-plan.json',
    'assets/generated/optimization/word-transport/native-result.json',
    'assets/generated/optimization/word-transport/parents.csv',
    'assets/generated/optimization/word-transport/producer-returns.jsonl.gz',
    'assets/generated/optimization/word-transport/producer-source/scripts/prepare_optimization_word_transport.py',
    'assets/generated/optimization/word-transport/producer-source/scripts/run_optimization_word_native.py',
    'assets/generated/optimization/word-transport/producer-source/src/openquantum/optimization_word_accounting.py',
    'assets/generated/optimization/word-transport/producer-source/src/openquantum/optimization_word_native.py',
    'assets/generated/optimization/word-transport/producer-source/src/openquantum/optimization_word_native_runner.py',
    'assets/generated/optimization/word-transport/producer-source/src/openquantum/optimization_word_transport.py',
    'assets/generated/optimization/word-transport/producer-source/src/openquantum/optimization_word_validation.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_accounting.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_bootstrap.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_driver.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_native.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_native_runner.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_transport.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_validation.py',
    'assets/generated/optimization/word-transport/producer-source/tests/test_optimization_word_watchdog_review.py',
    'assets/generated/optimization/word-transport/record-manifest.json',
    'assets/generated/optimization/word-transport/record-transformations.json',
    'assets/generated/optimization/word-transport/references.bib',
    'assets/generated/optimization/word-transport/requirements-lock.txt',
    'assets/generated/optimization/word-transport/runtime-comparison.svg',
    'assets/generated/optimization/word-transport/source-accounting.json',
    'assets/generated/optimization/word-transport/source-artifact-manifest.json',
    'assets/generated/optimization/word-transport/source-recipe.json',
    'assets/generated/optimization/word-transport/source-validation.json',
    'assets/generated/optimization/word-transport/summary.md',
    'assets/generated/optimization/word-transport/word-excerpt.json',
    'assets/generated/optimization/word-transport/word-excerpt.qasm',
    'assets/generated/optimization/word-transport/word-excerpt.svg',
})
PUBLIC_WORD_COMPRESSED_FILES = frozenset(
    name for name in PUBLIC_OPTIMIZATION_WORD_FILES if name.endswith(".gz"))
PUBLIC_COMPRESSED_JSON_FILES |= PUBLIC_WORD_COMPRESSED_FILES

# Exact retained higher-order Suzuki study, including its source snapshots.
PUBLIC_SUZUKI_FILES = frozenset(
    "assets/generated/simulation/suzuki/" + name for name in (
        "LICENSE.txt", "error-vs-steps.svg", "error-vs-work.svg",
        "generate_simulation_suzuki.py", "manifest.json", "n1-order4-circuit.svg",
        "n2-order4-circuit.svg", "n3-order4-circuit.svg", "signed-substeps.svg",
        "summary.csv", "summary.md", "suzuki.py",
    )
) | frozenset(
    f"assets/generated/simulation/suzuki/n{width}-s{order}-r{repetitions}/{name}"
    for width in (1, 2, 3) for order in (2, 4, 6) for repetitions in (1, 2, 4, 8)
    for name in ("record.json", "measured-trial.qasm")
)

PUBLIC_GIBBS_CHANNEL_FILES = frozenset({
    'assets/generated/simulation/detailed-balance-channel/LICENSE.txt',
    'assets/generated/simulation/detailed-balance-channel/detailed_balance_access.py',
    'assets/generated/simulation/detailed-balance-channel/detailed_balance_channel.py',
    'assets/generated/simulation/detailed-balance-channel/generate_detailed_balance_channel.py',
    'assets/generated/simulation/detailed-balance-channel/manifest.json',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.0625-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.0625-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.0625-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.125-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.125-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.125-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.25-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.25-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m2-beta1-delta0.25-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta0.5-delta0.125-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta0.5-delta0.125-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta0.5-delta0.125-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.0625-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.0625-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.0625-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.125-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.125-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.125-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.25-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.25-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta1-delta0.25-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta2-delta0.125-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta2-delta0.125-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m3-beta2-delta0.125-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.0625-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.0625-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.0625-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.125-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.125-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.125-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.25-dilation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.25-preparation.qasm',
    'assets/generated/simulation/detailed-balance-channel/n1-m4-beta1-delta0.25-reset-round.qasm',
    'assets/generated/simulation/detailed-balance-channel/requirements-lock.txt',
    'assets/generated/simulation/detailed-balance-channel/results.json',
})

FORMAL_FILES = ("C4Reference.lean", "lean-toolchain", "check.py", "LICENSE.txt", "README.md")
PUBLIC_DQI_FILES = frozenset(
    "assets/generated/optimization/dqi/" + name for name in (
        "LICENSE.txt", "bundle.json", "dqi-circuit.svg", "dqi-validated.json", "dqi.qasm",
        "generate_optimization_dqi.py", "optimization_dqi.ipynb", "optimization_dqi.py",
        "requirements-lock.txt", "verify_dqi_saved.py",
    )
)
# The clean-workspace family's executed output has a closed portable graph.
# Future private controls or output files are not admitted by their suffix.
PUBLIC_DQI_STRUCTURED_FILES = frozenset(
    "assets/generated/optimization/dqi-structured/" + name for name in (
        "bundle.json", "LICENSE.txt", "requirements-lock.txt",
        "optimization_dqi_structured.py", "optimization_dqi_structured_validation.py",
        "generate_optimization_dqi_structured.py", "validate_optimization_dqi_structured.py",
        "verify_dqi_structured_saved.py", "optimization_dqi_structured.ipynb",
        "invocation-acceptance.json", "small/validated-small.json",
        "small/n2/dqi.qasm", "small/n2/operations.ndjson", "small/n2/ledger.json", "small/n2/circuit.txt",
        "small/n3/dqi.qasm", "small/n3/operations.ndjson", "small/n3/ledger.json",
        "large/dqi-structured-generation.json", "large/dqi-structured.qasm", "large/operations.ndjson",
        "large/optimization_dqi_structured.py", "large/generate_optimization_dqi_structured.py",
        "large/requirements-lock.txt", "large/LICENSE.txt",
    )
)
PUBLIC_BATH_FILES = frozenset(
    "assets/generated/simulation/reusable-bath/" + name for name in (
        "artifact-manifest.json", "endpoints.json", "endpoints.png", "endpoints.svg",
        "fixture-ground.svg", "fixture-operations.json", "fixture-thermal.svg", "primitive-ledger.json",
        "results.json", "sampled-schedules.json", "source-exports.json", "source-ground.qasm",
        "source-thermal.qasm", "source/LICENSE.txt", "source/PROTOCOL-NOTE.md",
        "source/requirements-lock.txt", "source/reusable_bath.py", "source/reusable_bath_protocol.json",
        "source/reusable_bath_reference.py",
    )
)
MULTIPRODUCT_CASES = tuple((n, r) for n in (1, 2, 3) for r in (1, 2, 4, 8)) + ((2, 32), (3, 32))
PUBLIC_MULTIPRODUCT_FILES = frozenset(
    "assets/generated/simulation/multiproduct/" + name for name in (
        "manifest.json", "multiproduct.py", "generate_simulation_multiproduct.py", "requirements-lock.txt",
        "LICENSE.txt", "README.md", "study.json", "scalar-scaling.json", "convergence.csv", "convergence.svg",
        "small-lcu-circuit.svg", "opposite-identity-lcu.qasm", "opposite-identity-oaa.qasm",
        "state-n2-r4-lcu.qasm", "state-n2-r4-oaa.qasm", "state-n2-r4-oaa-channel.qasm",
        *(f"n{n}-r{r}-{method}.qasm" for n, r in MULTIPRODUCT_CASES
          for method in ("lcu", "oaa", "oaa-channel", "fine", "coarse")),
    )
) | frozenset(
    "assets/generated/simulation/multiproduct-large/" + name for name in (
        "manifest.json", "multiproduct.py", "generate_simulation_multiproduct.py", "requirements-lock.txt",
        "LICENSE.txt", "README.md", "study.json", "scalar-scaling.json", "n32-r1-lcu.qasm", "n32-r1-oaa.qasm",
        "n32-r1-oaa-channel.qasm", "n32-r1-fine.qasm", "n32-r1-coarse.qasm", "state-n32-r1-lcu.qasm",
        "state-n32-r1-oaa.qasm", "state-n32-r1-oaa-channel.qasm",
    )
)
PUBLIC_LINDBLAD_PORT_FILES = frozenset(
    "assets/generated/simulation/lindblad-port/" + name for name in LINDBLAD_PORT_FILES)
PUBLIC_BOOTSTRAP_FILES = ("binder/assemble_code.py",)
PUBLIC_TAIL_RESOURCE_POLICY_FILES = (
    "scripts/tail_resource_instrument_policy.json",
    "scripts/tail_resource_complete_policy.json",
)


def is_public_artifact(path):
    """Allow deliberate generated formats, without caches, logs, or archives."""
    if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
        return False
    if path.name == "publication.json":
        return False
    return (path.suffix in ARTIFACT_SUFFIXES or path.name == "LICENSE.txt"
            or path.name in {"C4Reference.lean", "lean-toolchain"})


PUBLIC_BAYESIAN_FILES = frozenset({
    'assets/generated/bayesian-qpe-validation/replay/LICENSE.txt',
    'assets/generated/bayesian-qpe-validation/replay/README.md',
    'assets/generated/bayesian-qpe-validation/replay/bayesian_qpe.py',
    'assets/generated/bayesian-qpe-validation/replay/circuit.svg',
    'assets/generated/bayesian-qpe-validation/replay/demo.qasm',
    'assets/generated/bayesian-qpe-validation/replay/generate_bayesian_qpe.py',
    'assets/generated/bayesian-qpe-validation/replay/large-n8-k32.qasm',
    'assets/generated/bayesian-qpe-validation/replay/manifest.json',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k1-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k1-offset1.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k16-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k16-offset1.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k2-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k2-offset0.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k2-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k2-offset1.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k32-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k32-offset0.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k32-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k32-offset1.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k4-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k4-offset1.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k8-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k8-offset0.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k8-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/replay/phase0-k8-offset1.qasm',
    'assets/generated/bayesian-qpe-validation/replay/posterior.svg',
    'assets/generated/bayesian-qpe-validation/replay/requirements-lock.txt',
    'assets/generated/bayesian-qpe-validation/replay/results.json',
    'assets/generated/bayesian-qpe-validation/replay/summary.csv',
    'assets/generated/bayesian-qpe-validation/source32/LICENSE.txt',
    'assets/generated/bayesian-qpe-validation/source32/README.md',
    'assets/generated/bayesian-qpe-validation/source32/bayesian_qpe.py',
    'assets/generated/bayesian-qpe-validation/source32/generate_bayesian_qpe.py',
    'assets/generated/bayesian-qpe-validation/source32/manifest.json',
    'assets/generated/bayesian-qpe-validation/source32/requirements-lock.txt',
    'assets/generated/bayesian-qpe-validation/source32/results.json',
    'assets/generated/bayesian-qpe-validation/source32/source-only-lowered.qasm',
    'assets/generated/bayesian-qpe-validation/source32/source-only.qasm',
    'assets/generated/bayesian-qpe-validation/validation.json',
    'assets/generated/bayesian-qpe/LICENSE.txt',
    'assets/generated/bayesian-qpe/README.md',
    'assets/generated/bayesian-qpe/bayesian_qpe.py',
    'assets/generated/bayesian-qpe/circuit.svg',
    'assets/generated/bayesian-qpe/demo.qasm',
    'assets/generated/bayesian-qpe/generate_bayesian_qpe.py',
    'assets/generated/bayesian-qpe/large-n8-k32.qasm',
    'assets/generated/bayesian-qpe/manifest.json',
    'assets/generated/bayesian-qpe/phase0-k1-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k1-offset1.qasm',
    'assets/generated/bayesian-qpe/phase0-k16-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k16-offset1.qasm',
    'assets/generated/bayesian-qpe/phase0-k2-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k2-offset0.qasm',
    'assets/generated/bayesian-qpe/phase0-k2-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k2-offset1.qasm',
    'assets/generated/bayesian-qpe/phase0-k32-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k32-offset0.qasm',
    'assets/generated/bayesian-qpe/phase0-k32-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k32-offset1.qasm',
    'assets/generated/bayesian-qpe/phase0-k4-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k4-offset1.qasm',
    'assets/generated/bayesian-qpe/phase0-k8-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k8-offset0.qasm',
    'assets/generated/bayesian-qpe/phase0-k8-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase0-k8-offset1.qasm',
    'assets/generated/bayesian-qpe/phase1-k1-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k1-offset0.qasm',
    'assets/generated/bayesian-qpe/phase1-k16-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k16-offset0.qasm',
    'assets/generated/bayesian-qpe/phase1-k16-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k16-offset1.qasm',
    'assets/generated/bayesian-qpe/phase1-k2-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k2-offset0.qasm',
    'assets/generated/bayesian-qpe/phase1-k32-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k32-offset0.qasm',
    'assets/generated/bayesian-qpe/phase1-k32-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k32-offset1.qasm',
    'assets/generated/bayesian-qpe/phase1-k4-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k4-offset0.qasm',
    'assets/generated/bayesian-qpe/phase1-k8-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase1-k8-offset0.qasm',
    'assets/generated/bayesian-qpe/phase2-k1-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k1-offset0.qasm',
    'assets/generated/bayesian-qpe/phase2-k1-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k1-offset1.qasm',
    'assets/generated/bayesian-qpe/phase2-k16-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k16-offset0.qasm',
    'assets/generated/bayesian-qpe/phase2-k2-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k2-offset1.qasm',
    'assets/generated/bayesian-qpe/phase2-k32-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k32-offset0.qasm',
    'assets/generated/bayesian-qpe/phase2-k32-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k32-offset1.qasm',
    'assets/generated/bayesian-qpe/phase2-k4-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k4-offset0.qasm',
    'assets/generated/bayesian-qpe/phase2-k8-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase2-k8-offset0.qasm',
    'assets/generated/bayesian-qpe/phase3-k1-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k1-offset0.qasm',
    'assets/generated/bayesian-qpe/phase3-k16-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k16-offset0.qasm',
    'assets/generated/bayesian-qpe/phase3-k2-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k2-offset0.qasm',
    'assets/generated/bayesian-qpe/phase3-k32-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k32-offset0.qasm',
    'assets/generated/bayesian-qpe/phase3-k32-offset1-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k32-offset1.qasm',
    'assets/generated/bayesian-qpe/phase3-k4-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k4-offset0.qasm',
    'assets/generated/bayesian-qpe/phase3-k8-offset0-lowered.qasm',
    'assets/generated/bayesian-qpe/phase3-k8-offset0.qasm',
    'assets/generated/bayesian-qpe/posterior.svg',
    'assets/generated/bayesian-qpe/requirements-lock.txt',
    'assets/generated/bayesian-qpe/results.json',
    'assets/generated/bayesian-qpe/summary.csv',
})


PUBLIC_ROBUST_FILES = frozenset("assets/generated/robust-qpe/" + name for name in (
    "manifest.json", "robust_qpe.py", "hadamard_qpe.py", "generate_robust_qpe.py",
    "requirements-lock.txt", "LICENSE.txt", "circuit.svg", "recovery.svg", "results.json",
    "software.json", "scaling.csv", "summary.md", "circuits/large-n8-sin-k32.qasm",
)) | frozenset(
    f"assets/generated/robust-qpe/circuits/phase-{phase}-{basis}-k{power}.qasm"
    for phase in range(4) for basis in ("cos", "sin") for power in (1, 2, 4, 8, 16, 32))

PUBLIC_LOCAL_SPIN_FILES = frozenset("assets/generated/simulation/reusable-bath-local-spin/" + name for name in (
    "artifact-manifest.json", "artifact-manifest.sha256", "cases.json", "generation.json",
    "reference.json", "schedules.json", "validation.json",
    "source-exports/n1-ground.qasm", "source-exports/n1-thermal.qasm",
    "source-exports/n2-ground.qasm", "source-exports/n2-thermal.qasm",
    "source-exports/n4-thermal-generation.qasm", "source-exports/n8-thermal-generation.qasm",
    "source-exports/n16-thermal-generation.qasm", "source-exports/n32-thermal-generation.qasm",
    "source-exports/n64-thermal-generation.qasm", "source/LICENSE.txt", "source/cases.json",
    "source/requirements-lock.txt", "source/reusable_bath_local_spin.py",
    "source/reusable_bath_local_spin_reference.py",
))
PUBLIC_LOCAL_SPIN_PRESENTATION_FILES = frozenset(
    "assets/generated/simulation/reusable-bath-local-spin-presentation/" + name
    for name in ("first-step.svg", "presentation.json"))
PUBLIC_LOCAL_SPIN_SOURCE_FILES = (
    "src/openquantum/simulation/reusable_bath_local_spin_cases.json",
    "tests/isolated/reusable_bath_local_spin_checks.py",
)


PUBLIC_QPE_FEEDBACK_FILES = frozenset("assets/generated/qpe-feedback-resources/" + name for name in (
    "manifest.json", "summary.csv", "summary.md", "logical-size-scaling.json", "README.md", "LICENSE.txt",
    "references.bib", "primitives.bib", "qpe_feedback_resources.py", "qft_qpe.py", "instrument_resources.py",
    "generate_qpe_feedback_resources.py", "verify_qpe_feedback_saved.py", "test_qpe_feedback_resources.py",
    "requirements-lock.txt",
)) | frozenset(
    f"assets/generated/qpe-feedback-resources/n1-m{m}/{name}"
    for m in (3, 4) for name in ("report.json", "textbook-source.qasm", "feedback-source.qasm",
        "textbook-clifford-t.qasm", "feedback-clifford-t.qasm", "feedback-all-corrections-accounting-envelope.qasm")
) | frozenset({"assets/generated/qpe-feedback-validation/qpe_feedback_resources.ipynb"})


PUBLIC_MEASURED_AND_FILES = frozenset("assets/generated/shor/measured-and/" + name for name in (
    "manifest.json", "study.json", "scaling.csv", "README.md", "LICENSE.txt", "requirements-lock.txt",
    "__init__.py", "grover.py", "shor.py", "shor_ripple.py", "shor_measured_and.py", "generate_shor_measured_and.py",
    "gadget.svg", "addition.svg", "gadget-static.qasm", "gadget-feedback_x.qasm", "gadget-reset.qasm",
)) | frozenset(
    f"assets/generated/shor/measured-and/addition-w{width}-{method}.qasm"
    for width in (1, 2, 3, 24) for method in ("static", "measured")
) | frozenset(
    f"assets/generated/shor/measured-and/order-N{modulus}-t{bits}-{method}.qasm"
    for modulus, bits in ((15, 4), (21, 4), (4093, 1)) for method in ("static", "measured")
) | frozenset({"assets/generated/shor/measured-and-artifacts.zip", "assets/generated/shor/measured-and-validation.json",
    "assets/generated/shor/measured-and-notebook-circuit.png", "assets/generated/shor/measured-and-executed.ipynb"})


PUBLIC_LOCAL_SPIN_REPRODUCTION_FILES = frozenset(
    "assets/generated/simulation/reusable-bath-local-spin-reproduction/" + name for name in (
        "artifact-manifest.json", "artifact-manifest.sha256", "events.ndjson", "finite-reference.json",
        "interface-validation.json", "provenance.json", "reproduction.json", "source-contracts.json",
        "source/reusable_bath_local_spin_checks.py", "source/reusable_bath_local_spin_reproduce.py",
    ))


PUBLIC_DQI_RESOURCE_FILES = frozenset("assets/generated/optimization/dqi-structured-resources/" + name for name in (
    "manifest.json", "LICENSE.txt", "README.md", "acceptance.json", "n2-compiled.json", "n2-compiled.ndjson",
    "n3-compiled.json", "n3-compiled.ndjson", "native-words.json", "physical.json", "requirements-lock.txt",
    "results.json", "software.json", "verification.json",
))


def include_generated_file(relative):
    """Select a generated file from a registered tree and deliberate formats."""
    path = Path(relative)
    try:
        generated = path.relative_to("assets/generated")
    except ValueError:
        return False
    if not generated.parts or generated.parts[0] not in PUBLIC_ARTIFACT_TREES:
        return False
    if generated.parts[0] in ("bayesian-qpe", "bayesian-qpe-validation"):
        return str(relative) in PUBLIC_BAYESIAN_FILES
    if generated.parts[0] == "robust-qpe":
        return str(relative) in PUBLIC_ROBUST_FILES
    if generated.parts[0] in ("qpe-feedback-resources", "qpe-feedback-validation"):
        return str(relative) in PUBLIC_QPE_FEEDBACK_FILES
    if generated.parts[0] == "shor" and len(generated.parts) > 1 and generated.parts[1].startswith("measured-and"):
        return str(relative) in PUBLIC_MEASURED_AND_FILES
    # Only the finalized reviewed port bundle is deliberate publication data.
    # Raw Trace JSON, logs, checkpoints and plans are never inferred by suffix.
    if generated.parts[:2] == ("simulation", "lindblad-port"):
        return str(relative) in PUBLIC_LINDBLAD_PORT_FILES
    if generated.parts[:2] == ("optimization", "word-transport"):
        return str(relative) in PUBLIC_OPTIMIZATION_WORD_FILES
    if generated.parts[:2] == ("optimization", "dqi"):
        return str(relative) in PUBLIC_DQI_FILES
    if generated.parts[:2] == ("optimization", "dqi-structured-resources"):
        return str(relative) in PUBLIC_DQI_RESOURCE_FILES
    if generated.parts[:2] == ("optimization", "dqi-structured"):
        return str(relative) in PUBLIC_DQI_STRUCTURED_FILES
    if generated.parts[:2] == ("simulation", "suzuki"):
        return str(relative) in PUBLIC_SUZUKI_FILES
    if generated.parts[:2] == ("simulation", "detailed-balance-channel"):
        return str(relative) in PUBLIC_GIBBS_CHANNEL_FILES
    if generated.parts[:2] == ("simulation", "reusable-bath"):
        return str(relative) in PUBLIC_BATH_FILES
    if generated.parts[:2] == ("simulation", "reusable-bath-local-spin"):
        return str(relative) in PUBLIC_LOCAL_SPIN_FILES
    if generated.parts[:2] == ("simulation", "reusable-bath-local-spin-reproduction"):
        return str(relative) in PUBLIC_LOCAL_SPIN_REPRODUCTION_FILES
    if generated.parts[:2] == ("simulation", "reusable-bath-local-spin-presentation"):
        return str(relative) in PUBLIC_LOCAL_SPIN_PRESENTATION_FILES
    if generated.parts[:2] in (("simulation", "multiproduct"), ("simulation", "multiproduct-large")):
        return str(relative) in PUBLIC_MULTIPRODUCT_FILES
    return (is_public_artifact(generated)
            or path.as_posix() in PUBLIC_ARCHIVES | PUBLIC_DEPENDENCY_FILES
            or str(relative) in PUBLIC_COMPRESSED_JSON_FILES)


def package_files(root=None):
    root = ROOT if root is None else Path(root).resolve()
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "src/openquantum").rglob("*.py"))
    }
    for directory, pattern in (("tests", "*.py"), ("notebooks", "*.ipynb"), ("references", "*.bib")):
        for path in sorted((root / directory).glob(pattern)):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    for name in PUBLIC_SCRIPTS:
        files["scripts/" + name] = (root / "scripts" / name).read_bytes()
    for relative in PUBLIC_BOOTSTRAP_FILES:
        files[relative] = (root / relative).read_bytes()
    for relative in PUBLIC_TAIL_RESOURCE_POLICY_FILES:
        files[relative] = (root / relative).read_bytes()
    for relative in PUBLIC_LOCAL_SPIN_SOURCE_FILES:
        files[relative] = (root / relative).read_bytes()
    for name in ("citations.bib", "LICENSE", "requirements-lock.txt"):
        files[name] = (root / name).read_bytes()
    for name in ("assemble_code.py", "postBuild", "environment.yml", "README.md"):
        files["binder/" + name] = (root / "binder" / name).read_bytes()
    # Replay uses the exact implementation that generated the sampled path.
    # Its historical hash and measurements are audited before any cost reuse.
    for name in ("optimization.py",):
        relative = "assets/generated/optimization/sampled/" + name
        files[relative] = (root / relative).read_bytes()
    files["assets/generated/optimization/sampled-runs.json"] = (
        root / "assets/generated/optimization/sampled-runs.json"
    ).read_bytes()
    for relative in ("assets/generated/grover/circuit.qasm",
                     "assets/generated/resources/grover-surface-code.json",
                     "assets/generated/resources/grover-surface-code.csv"):
        files[relative] = (root / relative).read_bytes()
    for tree in PUBLIC_ARTIFACT_TREES:
        directory = root / "assets/generated" / tree
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and include_generated_file(relative):
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"Public artifact must be a local regular file: {path.relative_to(root)}")
                if path.stat().st_size >= 100 * 1024 ** 2:
                    raise ValueError(f"Public artifact exceeds the 100 MiB file limit: {relative}")
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    for name in FORMAL_FILES:
        relative = "algorithms/optimization/formal/" + name
        files[relative] = (root / relative).read_bytes()
    files["pyproject.toml"] = b'''[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "openquantum-examples"
version = "0.4.0"
requires-python = ">=3.12"
license = "MIT"

[tool.setuptools.packages.find]
where = ["src"]
include = ["openquantum*"]
namespaces = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
'''
    files["README.md"] = b'''# Open Quantum Algorithms examples

Original public circuit examples; private project plans, credentials,
worker settings and Git history are excluded.

Use Python 3.12.14 and the included requirements lock for the full checked
environment. Verify a fresh extraction as described below, then install this
package with `python -m pip install --no-deps .`. Binder is configured to use
the same Python patch and requirements lock before installing the public package
without dependency resolution. Hosted Binder execution remains unvalidated;
the static site and browser controls do not require a live kernel.

Examples cover Grover, QFT/QPE, amplitude estimation, reversible-arithmetic
order finding and factoring, HHL, QSVT and Neumann-LCU inverses,
product formulas, Taylor-LCU/QSP, molecular-energy demos, amplification,
dissipative channels, thermal preparation, quantum walks, annealing and exact/finite-shot QAOA/VQE
toy workflows. Scope and limitations are documented at
https://xujue-ai.github.io/OpenQuantumAlgorithms-site/ . Arbitrary-rotation
resource projections remain conditional; no synthesis certificate is supplied.

Using the locked environment, run
`python scripts/verify_public_package.py --package-dir .` on a freshly
extracted, untouched copy to check the complete file manifest, isolated
imports and tests. Verify before installing this package or executing its
notebooks/generators. Installation and execution create unrecorded build or
cache files; repeat complete-content verification on a fresh extraction.
Run `python -m pytest`, `python scripts/execute_notebooks.py`, or the generator scripts.
Run `scripts/generate_grover.py` before its resource-estimation script.

Generated fixtures retain the exact source snapshots, measured QASM and raw
records used by the resource checks. Historical sampled optimization runs
are replayed with their matching source. Do not replace that source with a
new implementation without regenerating the corresponding run record.

The original Lean artifact proves five finite classical C4 reference facts.
Its toolchain, source, checker, license and recorded hashes are included.
`python scripts/verify_public_package.py --formal-only` checks those recorded
hashes; it does not rerun Lean. With an already installed pinned elan
toolchain, use `python algorithms/optimization/formal/check.py --output assets/generated/optimization/formal`
to reproduce the kernel check.
Coverage excludes Python equivalence, quantum circuits, sampling and
fault-tolerant models.
'''
    provenance = {
        "schema": "openquantum-public-package-v2",
        "source_commit": git_revision(root),
        "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items() if name.endswith(".py")},
        "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["PROVENANCE.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    return files


def main():
    target = ROOT / "assets/generated/openquantum-code.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in package_files().items():
            info = zipfile.ZipInfo("openquantum-examples/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    print("Exported public example package", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
