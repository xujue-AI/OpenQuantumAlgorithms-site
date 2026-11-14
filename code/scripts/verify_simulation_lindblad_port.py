#!/usr/bin/env python3
"""Verify the immutable reviewed port release from saved bytes with stdlib only.

No implementation module is imported and no producer, synthesis, density or
Native query runs. Native Trace raw JSON and the inline audit script were not
saved: the 157 Trace digests are audit-trusted, not independently replayed or
raw-verified. Saved profile value JSON is verified separately. All costs are
conditional model results; the population/hardware schedule is unexecuted and
there is no gain ratio against the original controller-rejected baseline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_RELATIVE = "assets/generated/simulation/lindblad-port"
REVIEWED_MANIFEST_SHA256 = "7328a265464a9bbd25875c565d62ee4921f59d42af239deedec4951dbd17a83b"
RESULTS_SHA256 = "58ae0a0e7f5c2eeaf4ff3aae6df36fbd063a3e49b7c06362e6aa1c6ade678732"
PROGRESS_SHA256 = "9a675aafb5bbb71fffd0be6766034f8ab64427a257e28c427ced949faa5204fe"
RECIPE_SHA256 = "2e46759e17a71b98b7547e35b97b42976f840b350fbcceab6fd33eaec941a1b5"
CATALOG_FILE_SHA256 = "e40ee48d66035b3bd1a41f3b557c54bebddaf7f4306c318b79f341ed6cf53aac"
CATALOG_RECOVERY_SHA256 = "f8da75f64523c8e4f7999d356aae968f46c2da389f01e9bfcedcd204c157a767"
SOURCE_PROOF_SHA256 = "a07affb0f49c7866105c4b92484769e446fd2d5adbb274421569fafeb9d4d35a"
INVENTORY_IDENTITY = "85e150a5a7c58f67b77530bf84dfd567fd8eafde73ae23e511e4f9a21976c8fd"
FINGERPRINT = "e3f97b76033d5c32ce8cf6bdf3778ec21a36091f2cba9defb37fbb01d4e412aa"
GLOBAL_PHASE = "0x1.921fb5444274ep+2"
SOURCE_FILES = {
    "instrument_resources.py": "src/openquantum/instrument_resources.py",
    "run_simulation_lindblad_port.py": "scripts/run_simulation_lindblad_port.py",
    **{name + ".py": "src/openquantum/simulation/" + name + ".py" for name in (
        "lindblad_workload", "two_spin_lindblad", "open_systems", "bounded_density_instrument",
        "lindblad_four_density", "lindblad_four_compile", "lindblad_four_physical",
        "lindblad_port_words", "lindblad_port_rewrite", "lindblad_port_resources")},
    "LICENSE.txt": "LICENSE", "requirements-lock.txt": "requirements-lock.txt", "references.bib": None,
}
EXCLUDED_METADATA = frozenset({"manifest.json", "progress.json", "artifact-inventory.json", "artifact-inventory.csv"})
WORD_FILES = frozenset(f"words/word-{index:02d}.qasm" for index in range(8))
BUNDLE_FILES = frozenset(SOURCE_FILES) | WORD_FILES | EXCLUDED_METADATA | frozenset({
    "README.md", "candidate-trial.qasm", "results.json", "source-proof.json", "word-catalog.json", "recipe.json"})
SOURCE_COUNTS = {"cry": 1477, "cx": 4009, "h": 4, "measure": 8, "reset": 1481, "rx": 1688, "rz": 1266, "s": 1}
GATE_COUNTS = {"h": 661278, "s": 349839, "t": 642284, "x": 5908, "cx": 137994,
               "reset": 1481, "measure": 8, "barrier": 2978}
EVENT_COUNTS = {"initial_reset": 31, "h": 661278, "s": 349839, "cx": 137994, "t_q0": 642284,
                "x": 5908, "reset": 1473, "measure": 8, "trial_control": 1}
MACRO_SHAPES = {"cx": (3, 6, 11), "h": (0, 0, 1), "initial_reset": (0, 1, 1),
                "measure": (1, 2, 3), "reset": (1, 2, 5), "s": (0, 0, 1),
                "t_q0": (13, 25, 75), "trial_control": (0, 0, 1), "x": (0, 0, 1)}
TRIALS = 15756
STATES = 10119826704
BUDGET = math.nextafter(float(Fraction(1, 1200)), 0.0)
STATUS = "computed_conditional_port_surface_code_model"
PROFILE = {"physical_gate_time_ns": 100, "physical_measurement_time_ns": 500,
           "physical_initialization_time_ns": 100, "code_cycle_time_ns": 1000, "reaction_time_ns": 10000,
           "controller_failure_per_decision": 1e-15, "memory_prefactor": .1,
           "initialization_residual_prefactor": .1, "memory_threshold": .01,
           "factory_pipeline_failure_bound": .01, "factory_feedback_rounds_reserved": 3, "maximum_code_distance": 49}
CAPS = {"word_recipe_seconds": 180, "fixture_seconds": 180, "candidate_seconds": 1800,
        "native_profile_seconds": 180, "parent_seconds": 2400, "rss_bytes": 2147483648,
        "managed_bytes": 33554432, "log_bytes": 33554432, "checkpoint_bytes": 268435456,
        "recipe_bytes": 8388608, "factory_queries_per_profile": 24, "surface_queries_per_profile": 24,
        "factory_candidates_per_query": 1000, "factory_candidates_per_profile": 24000}
PACKAGES = {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2"}
# These are literal reviewed identities, never derived from a caller's canonical
# tree. There is deliberately no expected-hash argument or regeneration option.
RECORD_IDENTITIES = {'manifest.json': '48a5cf5882e8cc079a044760fbf6717b531b096f8a7f9427779cb6d625b8ea49',
 'results.json': '457bb6f57a1368d77a739bad02c993abb1d4b78e2801c54e0dceb63ae69c07f9',
 'progress.json': 'bf037b245087ddcfd0a31bddf33eed9f22f455fd0e4684f8a1344fe4629f8b53',
 'source-proof.json': '2067372540672a5dc659572f2b5365a2baa1ac0e500103ab3c1220f04e256265',
 'word-catalog.json': '1dcf8b3c893fe114a568c5e24ef2207f8d4cf6abb011c74c0c1fc797d0ccedeb',
 'recipe.json': 'b61f0d1d573516c25796c5faa4768ea1317d5578367c03392efaa8325e1157ae'}
ROW_IDENTITIES = [{'all_feasible_points': 'e41a5692bd6eebf1f3eb62f68cc130ac75c8611c10cf30db0f68c72f9f08fc60',
  'pareto_points': 'b11963bcd0600fadbda443b3403d538cfa676e5bd3cdbd820eed0a9cad8829e6'},
 {'all_feasible_points': '3eac6e013523bb04ff9f1c68b97b0c4ba71f24b33a7a4b5f61a51b7cdc3f9f8f',
  'pareto_points': 'd045faa387c7aeef79ac94fe1a780f7d81cbb734c1b89a7631a2d8fd779847f0'}]
PRESERVED_SCIENTIFIC_FILES = {'assets/generated/simulation/lindblad-four/LICENSE.txt': {'bytes': 1093,
                                                           'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'assets/generated/simulation/lindblad-four/README.md': {'bytes': 5877,
                                                         'sha256': '0e03a0e9f673f5b49dc122a51f5f8f7a240d23b49efe5430b6989b28f3d1092e'},
 'assets/generated/simulation/lindblad-four/artifact-inventory.csv': {'bytes': 2927,
                                                                      'sha256': '48f7c44dc1c67a4dd72d3666133bb590b561202b895919251c39ea5c2f48b13f'},
 'assets/generated/simulation/lindblad-four/artifact-inventory.json': {'bytes': 6482,
                                                                       'sha256': '3c995cca4ea1b751769dac45c45829b55c4c2476cea2e0461e1f42a7639d1617'},
 'assets/generated/simulation/lindblad-four/bounded_density_instrument.py': {'bytes': 23060,
                                                                             'sha256': '396b1d392d151c5211dc3b46fd2202ad3a9543e57485e8d3635bef5f7bac7517'},
 'assets/generated/simulation/lindblad-four/clifford-t-trial.qasm': {'bytes': 13671190,
                                                                     'sha256': '527e5a78aa0f4ed8a3394f040049028969644e43ed1763d4f0e81a0ac66c3009'},
 'assets/generated/simulation/lindblad-four/counts.svg': {'bytes': 31425,
                                                          'sha256': '6a2a4aeed53b87f4b091ea2480924da904d9cf6c7155c5e3588541c506890ebb'},
 'assets/generated/simulation/lindblad-four/error-budget.svg': {'bytes': 43572,
                                                                'sha256': '5e4d49f2e89c17f99090b4e37c3efa649986d16051cf4d8ee7a15cf19df27a6e'},
 'assets/generated/simulation/lindblad-four/failures.svg': {'bytes': 38220,
                                                            'sha256': '0b3955f4903126df106fc16b486445daca62ccf80d6e3b0c798d1e8d7b4f1ed8'},
 'assets/generated/simulation/lindblad-four/family-status.csv': {'bytes': 942,
                                                                 'sha256': 'd46e8d0a73c46fb73ef6467c24e5744aebb7c4c75523be751d26faa6735b292c'},
 'assets/generated/simulation/lindblad-four/generate_simulation_lindblad_four.py': {'bytes': 62145,
                                                                                    'sha256': 'cf4491b1cced1d712248cddff968610bbcee74060b666ec8efb34055d4d68b12'},
 'assets/generated/simulation/lindblad-four/instrument_resources.py': {'bytes': 52381,
                                                                       'sha256': 'fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863'},
 'assets/generated/simulation/lindblad-four/lindblad_four_compile.py': {'bytes': 22747,
                                                                        'sha256': '8036bd1e5e9c1ba89050c025224da80883c4af0296f915e7a531d03193290249'},
 'assets/generated/simulation/lindblad-four/lindblad_four_density.py': {'bytes': 28335,
                                                                        'sha256': '83c4245acc7a243fdd8a227b937fdbda2ccaf7e8e7498db3c9da4d73aa23fcf5'},
 'assets/generated/simulation/lindblad-four/lindblad_four_physical.py': {'bytes': 55208,
                                                                         'sha256': '636c95a24eeb10d0d6ab01a46125adc9b4897cbaa7f555e664f57b05e15fa452'},
 'assets/generated/simulation/lindblad-four/lindblad_workload.py': {'bytes': 19899,
                                                                    'sha256': 'f6edf708b463018b699fb37c11a2bd89f6d8d6bc214e738377004b45c745d577'},
 'assets/generated/simulation/lindblad-four/manifest.json': {'bytes': 18652,
                                                             'sha256': '5e78c5deac53d8106539ae75c5edbab4027ead7439c2bd4a2e491ec37834296f'},
 'assets/generated/simulation/lindblad-four/one-step.svg': {'bytes': 112010,
                                                            'sha256': '5a2f260bf42cb38a72539a0b06c4cb7bef2ac5d9af165c2a71abe605ea2b721b'},
 'assets/generated/simulation/lindblad-four/open_systems.py': {'bytes': 14355,
                                                               'sha256': '30145a6206ba2616f9604dc15972a588821c44819408d6770193a63ccff41692'},
 'assets/generated/simulation/lindblad-four/progress.json': {'bytes': 9344,
                                                             'sha256': '23d6f92c1f6430aa8f09bf60fed10b6e6f2068022eb135d0951ddee150ffcf46'},
 'assets/generated/simulation/lindblad-four/references.bib': {'bytes': 10907,
                                                              'sha256': '7bd314a90054ec17e348ca5f4cf748b0100150f6c3f4eac6732335f077a6bc4a'},
 'assets/generated/simulation/lindblad-four/requirements-lock.txt': {'bytes': 1974,
                                                                     'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'assets/generated/simulation/lindblad-four/results.json': {'bytes': 269279,
                                                            'sha256': 'a7451fd56010cbcf6d0591e55c30d154c6e02fed4ce850a02a0a9c604e985002'},
 'assets/generated/simulation/lindblad-four/routed-rotation-trial.qasm': {'bytes': 464087,
                                                                          'sha256': 'd62bb9b41337af2e0a8eb2d1eb4b3f3a50f3b3bd46e772b54e9b7b752db53c5c'},
 'assets/generated/simulation/lindblad-four/source-preflight.json': {'bytes': 90465,
                                                                     'sha256': 'a70329ebb4f37ac3144ce73aee99c1249a8b653b2112a03ed1f5a582a152045d'},
 'assets/generated/simulation/lindblad-four/source-trial.qasm': {'bytes': 226930,
                                                                 'sha256': '6924fce307bb11bb111e6588482a240f172c77c6dd5f9e14cfce98ce2857ce9b'},
 'assets/generated/simulation/lindblad-four/space-time.svg': {'bytes': 31705,
                                                              'sha256': 'b8e921a5acdaa3250ecb7f55bfa90a5fea17880d6fc9eca673621747b0518305'},
 'assets/generated/simulation/lindblad-four/summary.csv': {'bytes': 448,
                                                           'sha256': 'd8d504f979b1b779ca787e7a190528b0a6dc57a81e1dabf7f49ac8adce2258d3'},
 'assets/generated/simulation/lindblad-four/two_spin_lindblad.py': {'bytes': 23120,
                                                                    'sha256': '06e8da24dadefaccbb0d28f540995eaf7f681ee772962bae35a79dddea06ef34'},
 'assets/generated/simulation/lindblad-physical/LICENSE.txt': {'bytes': 1093,
                                                               'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'assets/generated/simulation/lindblad-physical/README.md': {'bytes': 3865,
                                                             'sha256': '849ff516c47873728fb98c5296abbd4e0830e5a44a07c2ef6dc9b718b7e67b1b'},
 'assets/generated/simulation/lindblad-physical/clifford-t-trial.qasm': {'bytes': 1659357,
                                                                         'sha256': '6814df4b766b8d446da327763aea3147ba24b8e5ec3089bd33638b8ad1d5eafb'},
 'assets/generated/simulation/lindblad-physical/counts.svg': {'bytes': 29780,
                                                              'sha256': 'e9c1a8ec9780fd1a2238a79b30ec0e64546c1f7b2b3913092ec8254fc16d361b'},
 'assets/generated/simulation/lindblad-physical/error-budget.svg': {'bytes': 44203,
                                                                    'sha256': 'b12c7cecef72cc239e7d5afa16d952fd252c4df0982cb49fc0623d9e416dcb84'},
 'assets/generated/simulation/lindblad-physical/failures.svg': {'bytes': 52334,
                                                                'sha256': 'c9a7fcce02133d0d2da6c62c02358df9b1fb31bb116e9408d867eee52a6e6f4e'},
 'assets/generated/simulation/lindblad-physical/family-status.csv': {'bytes': 395,
                                                                     'sha256': '13712141f4ee25992a43cf2bef5b46394d4291b838c9755ea219bfeaf89f0648'},
 'assets/generated/simulation/lindblad-physical/generate_simulation_lindblad_physical.py': {'bytes': 35615,
                                                                                            'sha256': '6eb9bc13e7870ac5927f2ff8ac32fd942127a85cb77b513c986a7b0d4638ecde'},
 'assets/generated/simulation/lindblad-physical/instrument_resources.py': {'bytes': 52381,
                                                                           'sha256': 'fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863'},
 'assets/generated/simulation/lindblad-physical/lindblad_physical_workload.py': {'bytes': 26779,
                                                                                 'sha256': '667c7dff27d7e89df37393eb9766525f07de28620d8b25da2238c925d56af704'},
 'assets/generated/simulation/lindblad-physical/lindblad_workload.py': {'bytes': 19899,
                                                                        'sha256': 'f6edf708b463018b699fb37c11a2bd89f6d8d6bc214e738377004b45c745d577'},
 'assets/generated/simulation/lindblad-physical/manifest.json': {'bytes': 8005,
                                                                 'sha256': '1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595'},
 'assets/generated/simulation/lindblad-physical/one-step.svg': {'bytes': 59191,
                                                                'sha256': 'e17060fd71c677d2c617463ff7ce896476aa4f2f7912631d828ca56bde1f4b65'},
 'assets/generated/simulation/lindblad-physical/open_systems.py': {'bytes': 14355,
                                                                   'sha256': '30145a6206ba2616f9604dc15972a588821c44819408d6770193a63ccff41692'},
 'assets/generated/simulation/lindblad-physical/references.bib': {'bytes': 10907,
                                                                  'sha256': '7bd314a90054ec17e348ca5f4cf748b0100150f6c3f4eac6732335f077a6bc4a'},
 'assets/generated/simulation/lindblad-physical/requirements-lock.txt': {'bytes': 1974,
                                                                         'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'assets/generated/simulation/lindblad-physical/results.json': {'bytes': 1086428,
                                                                'sha256': 'dea4ae3b7f7474bda4b92a824008f60aa08a19828b409e2e6a6b9be105f34efc'},
 'assets/generated/simulation/lindblad-physical/source-trial.qasm': {'bytes': 30414,
                                                                     'sha256': '3fdb92c3a613eca1b85ab1db182a17e4ffccbd470074dd6c60078895735ab5e2'},
 'assets/generated/simulation/lindblad-physical/space-time.svg': {'bytes': 41224,
                                                                  'sha256': '47387d9c629386c52d2b8babb789ddc262d60529f415d05457af58a64f641185'},
 'assets/generated/simulation/lindblad-physical/summary.csv': {'bytes': 478,
                                                               'sha256': 'bf16db825b5764b794fb8ab6f7e68e8eec5cb662319ffa0e65a116933bee0606'},
 'assets/generated/simulation/lindblad-physical/two_spin_lindblad.py': {'bytes': 23120,
                                                                        'sha256': '06e8da24dadefaccbb0d28f540995eaf7f681ee772962bae35a79dddea06ef34'},
 'assets/generated/simulation/lindblad-three/LICENSE.txt': {'bytes': 1093,
                                                            'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'assets/generated/simulation/lindblad-three/README.md': {'bytes': 5474,
                                                          'sha256': 'a32d969e538651ce5c1ad1c728ca6af054669b9a9830b2b7a775db8d3773cac6'},
 'assets/generated/simulation/lindblad-three/artifact-inventory.csv': {'bytes': 2558,
                                                                       'sha256': 'f648727dd1d3fa03164318eeba990488c46720a98838b2c8009268c0312ad6d5'},
 'assets/generated/simulation/lindblad-three/artifact-inventory.json': {'bytes': 5850,
                                                                        'sha256': 'c53ee5c339e8b5256b3ec1c23ced4899d0e8ab3a6f0eeabf9d0000bce414798d'},
 'assets/generated/simulation/lindblad-three/bounded_density_instrument.py': {'bytes': 23060,
                                                                              'sha256': '396b1d392d151c5211dc3b46fd2202ad3a9543e57485e8d3635bef5f7bac7517'},
 'assets/generated/simulation/lindblad-three/clifford-t-trial.qasm': {'bytes': 5751034,
                                                                      'sha256': '2883ae69d5c197323a585561c1ff2532461ba017ec96c2ed31d5e1793b9022da'},
 'assets/generated/simulation/lindblad-three/counts.svg': {'bytes': 30767,
                                                           'sha256': 'b73ca31c73361509395ee2015bfa58fe62c2cacb7f62779b1e547f554cb83a02'},
 'assets/generated/simulation/lindblad-three/error-budget.svg': {'bytes': 44358,
                                                                 'sha256': 'd787f58e661bbb603229b4e7bbaff07daaa4c78ed4433eff5bdc44952e8fd91d'},
 'assets/generated/simulation/lindblad-three/failures.svg': {'bytes': 52336,
                                                             'sha256': '330bb84179abba6de935d06f01f6e9d0ee6e446b20e993940252c5cf6740ca06'},
 'assets/generated/simulation/lindblad-three/family-status.csv': {'bytes': 540,
                                                                  'sha256': '7b0767710cba1f62462c1d95f118fbf20797ba0633b603e7b3c0123c91585720'},
 'assets/generated/simulation/lindblad-three/generate_simulation_lindblad_three.py': {'bytes': 47139,
                                                                                      'sha256': '9dd09a8e60fe5071bb8642b5b9604e7112b773b1ea12ddb350cb8cc68f9ee962'},
 'assets/generated/simulation/lindblad-three/instrument_resources.py': {'bytes': 52381,
                                                                        'sha256': 'fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863'},
 'assets/generated/simulation/lindblad-three/lindblad_three_physical.py': {'bytes': 32959,
                                                                           'sha256': 'ee9b6a207811b9217cb679fad3ddf6ceaac6542774005d1a7226138926cf6d91'},
 'assets/generated/simulation/lindblad-three/lindblad_workload.py': {'bytes': 19899,
                                                                     'sha256': 'f6edf708b463018b699fb37c11a2bd89f6d8d6bc214e738377004b45c745d577'},
 'assets/generated/simulation/lindblad-three/manifest.json': {'bytes': 9646,
                                                              'sha256': '9dde563b53b186272edb724de1b61865d497a2d00ea3368e05c8162b01d837f7'},
 'assets/generated/simulation/lindblad-three/one-step.svg': {'bytes': 85626,
                                                             'sha256': '7e401e682faad1ae8a36f898da98e8d8131abade82606a873b6c4df048992b3c'},
 'assets/generated/simulation/lindblad-three/open_systems.py': {'bytes': 14355,
                                                                'sha256': '30145a6206ba2616f9604dc15972a588821c44819408d6770193a63ccff41692'},
 'assets/generated/simulation/lindblad-three/progress.json': {'bytes': 2347,
                                                              'sha256': '8c7650b872c64901348aaed47dc3f89474515d9fda7d96757b5e6c042d9f9bf3'},
 'assets/generated/simulation/lindblad-three/references.bib': {'bytes': 10907,
                                                               'sha256': '7bd314a90054ec17e348ca5f4cf748b0100150f6c3f4eac6732335f077a6bc4a'},
 'assets/generated/simulation/lindblad-three/requirements-lock.txt': {'bytes': 1974,
                                                                      'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'assets/generated/simulation/lindblad-three/results.json': {'bytes': 1174227,
                                                             'sha256': 'd45bc3a1d494e7280958c3414e48da29b9634bc668612fc82e979364df40154d'},
 'assets/generated/simulation/lindblad-three/routed-rotation-trial.qasm': {'bytes': 189258,
                                                                           'sha256': 'f51b3d81dac3d40decae9a7e9788f141659dd1b6e68d12858ceadee4a5cac727'},
 'assets/generated/simulation/lindblad-three/source-trial.qasm': {'bytes': 101558,
                                                                  'sha256': 'e85cc8f6a6a0e07a1b2e964755287a176370a35eeb0ba73247e1998d8a8dfe5c'},
 'assets/generated/simulation/lindblad-three/space-time.svg': {'bytes': 42190,
                                                               'sha256': '117afd684f067d9e3ec62ac60d46b61f189840e07a825de71e9bbcc94587d4af'},
 'assets/generated/simulation/lindblad-three/summary.csv': {'bytes': 589,
                                                            'sha256': 'fdeb9902f70698b39b975d0116b18817c7eec196192f9f2b75fd2c21e938a037'},
 'assets/generated/simulation/lindblad-three/two_spin_lindblad.py': {'bytes': 23120,
                                                                     'sha256': '06e8da24dadefaccbb0d28f540995eaf7f681ee772962bae35a79dddea06ef34'}}


def equal(actual, expected, label):
    if isinstance(expected, (set, frozenset)):
        valid = isinstance(actual, (set, frozenset)) and actual == expected
    else:
        valid = canonical(actual) == canonical(expected)
    if not valid:
        raise ValueError(f"port {label} differs from the immutable reviewed contract")


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("port record is malformed or nonfinite") from error


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def finite_number(value, label, *, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"port {label} is nonfinite or outside its recorded bound")


def close(actual, expected, label):
    finite_number(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-30):
        raise ValueError(f"port {label} does not reconcile with saved accounting")


def checked_directory(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_dir():
        raise ValueError("port root must be a regular directory without linked ancestors")
    return path


def checked_file(root, name):
    if not isinstance(name, str):
        raise ValueError("port relative path must be a string")
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("port unsafe relative path")
    root = checked_directory(root)
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("port linked file or ancestor")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError("port missing saved file: " + name) from error
    if not stat.S_ISREG(info.st_mode) or info.st_size > 32 * 1024**2 or info.st_nlink != 1:
        raise ValueError("port missing, linked, special or oversized saved file: " + name)
    return path


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    # source-proof is the largest saved JSON (6,448,519 bytes).
    path = checked_file(path.parent, path.name)
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("port saved JSON exceeds its eight-MiB parsing cap")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("port duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError("port nonfinite JSON value: " + value)
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result
    try:
        result = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float)
    except (json.JSONDecodeError, RecursionError, UnicodeError) as error:
        raise ValueError("port JSON malformed or exceeds nesting limit") from error
    if not isinstance(result, dict):
        raise ValueError("port saved JSON must contain an object")
    stack = [(result, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > 128:
            raise ValueError("port JSON exceeds its 128-level nesting cap")
        if isinstance(value, dict):
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
    return result


def verify_baseline(baseline):
    """Preserve both negative original profiles with their costs unassessed."""
    try:
        equal(len(baseline["physical_profiles"]), 2, "two original negative profiles")
        forbidden = {"peak_physical_qubits", "runtime_per_complete_attempt_ns", "worst_case_runtime_all_attempts_ns",
                     "data_code_distance", "full_physical_failure_model", "failure_components_all_attempts", "feasible_candidate_count"}
        for record, rate in zip(baseline["physical_profiles"], (.001, .0001)):
            equal(record["status"], "infeasible_declared_controller_model", "original controller rejection")
            equal(record["profile"], {**PROFILE, "physical_error_rate": rate}, "original frozen profile")
            for key in ("pareto_points", "all_feasible_points"):
                equal(record[key], [], "original uncomputed frontier")
            equal(record["native_estimation_performed"], False, "original native nonexecution")
            if forbidden & record.keys():
                raise ValueError("port original rejected profile must retain unassessed costs")
            diagnostic = record["controller_model_diagnostic"]
            for key, expected in (("controller_decisions_per_trial", 123409809),
                                  ("controller_decisions_all_trials", 1944444950604),
                                  ("maximum_allowed_controller_decisions_per_trial", 52889904), ("budget_exceeded", True)):
                equal(diagnostic[key], expected, "original controller " + key)
            proposal = record["matched_task_implementation_proposal"]
            equal(proposal["proposed_implementation_executed"], False, "original planned proposal")
            equal(proposal["feasibility_claimed"], False, "original absent feasibility claim")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("port original negative records are incomplete or malformed") from error


def verify_point(row, rate):
    """Reconcile saved literal macro/time/population ledgers without Trace replay."""
    equal(rate in (.001, .0001), True, "fixed physical scenario")
    for key in ("data_code_distance", "reserved_logical_patches", "peak_physical_qubits", "data_workspace_physical_qubits",
                "factory_physical_qubits", "runtime_per_complete_attempt_ns", "worst_case_runtime_all_attempts_ns",
                "code_cycles_per_complete_attempt"):
        equal(isinstance(row[key], int) and not isinstance(row[key], bool) and row[key] > 0, True, "integer cost " + key)
    d = row["data_code_distance"]
    equal(isinstance(d, int) and not isinstance(d, bool) and d in range(3, 50, 2), True, "odd data distance")
    equal(row["reserved_logical_patches"], 31, "reserved patches")
    equal(row["data_workspace_physical_qubits"], 31 * (2*d*d - 1), "data/workspace qubits")
    equal(row["peak_physical_qubits"], row["data_workspace_physical_qubits"] + row["factory_physical_qubits"], "total qubits")
    finite_number(row["factory_physical_qubits"], "factory qubits", minimum=1)
    supply = row["factory_supply"]
    equal(supply["fixed_logical_distillation_distance"], d, "native factory distillation distance")
    equal(isinstance(supply["pipeline_time_ns"], int) and not isinstance(supply["pipeline_time_ns"], bool)
          and supply["pipeline_time_ns"] > 0, True, "integer factory time")
    for key, expected in (("full_pipeline_retry_cap_per_state", 7), ("states_all_complete_attempts", STATES),
                          ("worst_case_pipeline_attempts", STATES*7), ("availability_failure_union_bound", STATES/100**7),
                          ("pipeline_feedback_rounds_reserved", 3), ("extra_init_readout_feedback_dispatch_ns_per_pipeline", 41800),
                          ("native_qec_crossing_prefactor", .03), ("native_qec_threshold", .01), ("native_code_cycle_time_ns", 1000)):
        equal(supply[key], expected, "factory supply " + key)
    cnot = 9000*d + 112100
    initial = 1000*d + 10100
    destructive = 10500
    equal(supply["extra_output_port_hold_ns_per_state"], 41800 + 3*cnot, "port residence")
    times = {"cx": cnot, "initial_reset": initial, "measure": 3000*d + 30700,
             "reset": 4000*d + 50700,
             "t_q0": 7*(supply["pipeline_time_ns"] + 41800) + 4*cnot + destructive + 10000 + initial}
    finite_number(supply["pipeline_time_ns"], "factory pipeline time", minimum=1)
    finite_number(supply["output_error"], "native factory output quality", maximum=1)
    ledger = row["macro_ledger_per_complete_attempt"]
    equal([item["operation"] for item in ledger], sorted(EVENT_COUNTS), "complete nine-macro ledger")
    runtime = 0
    decisions = 0
    readouts = 0
    preparations = 0
    logical = .1*(rate/.01)**((d + 1)//2)
    odd = 0.0
    for _ in range(d):
        odd = odd*(1-rate) + (1-odd)*rate
    readout_error = math.fsum(math.comb(d, k)*odd**k*(1-odd)**(d-k) for k in range((d+1)//2, d+1))
    for item in ledger:
        operation = item["operation"]
        equal(item["count"], EVENT_COUNTS[operation], "macro population " + operation)
        for key, expected in zip(("readouts", "preparations", "decisions"), MACRO_SHAPES[operation]):
            equal(item[key], expected, "macro " + key + " " + operation)
        equal(item["time_ns"], times.get(operation, 10000), "macro time " + operation)
        close(item["event_failure_model"], item["readouts"]*readout_error + item["preparations"]*logical
              + item["decisions"]*1e-15, "macro failure " + operation)
        runtime += item["count"] * item["time_ns"]
        decisions += item["count"] * item["decisions"]
        readouts += item["count"] * item["readouts"]
        preparations += item["count"] * item["preparations"]
    equal(decisions, 50713680, "all macro controller decisions")
    equal(row["runtime_per_complete_attempt_ns"], runtime, "complete trial runtime")
    equal(row["worst_case_runtime_all_attempts_ns"], runtime*TRIALS, "complete population runtime")
    equal(row["code_cycles_per_complete_attempt"], (runtime + 999)//1000, "integer code cycles")
    components = row["failure_components_all_attempts"]
    equal(set(components), {"data_workspace_code_exposure", "factory_output_handoff_storage", "logical_readout_scenario",
                            "logical_initialization_scenario", "controller_scenario", "factory_output_quality_model"}, "six failure components")
    for name, value in components.items():
        finite_number(value, "all-trial failure " + name, maximum=BUDGET)
    equal(components["controller_scenario"], 50713680*TRIALS*1e-15, "controller failure contribution")
    close(components["data_workspace_code_exposure"], 31*((runtime+999)//1000)*logical*TRIALS, "full data exposure")
    close(components["factory_output_handoff_storage"], 642284*((41800+3*cnot+999)//1000)*logical*TRIALS, "full port residence")
    close(components["logical_readout_scenario"], readouts*readout_error*TRIALS, "full readout failure")
    close(components["logical_initialization_scenario"], preparations*logical*TRIALS, "full preparation failure")
    close(components["factory_output_quality_model"], STATES*supply["output_error"], "factory quality population")
    if STATES*supply["output_error"] > BUDGET*.75 or supply["output_error"] >= 7.1*rate:
        raise ValueError("port saved factory quality violates fixed calibration eligibility")
    close(row["full_physical_failure_model"], sum(components.values()), "full union-bound sum")
    if not Fraction.from_float(row["full_physical_failure_model"]) <= Fraction(1, 1200):
        raise ValueError("port complete physical allowance exceeded")
    if not isinstance(row["accounting_trace_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["accounting_trace_sha256"]) is None:
        raise ValueError("port audit-trusted Trace digest is malformed")


def verify_profiles(results):
    try:
        profiles = results["physical_profiles"]
        equal(len(profiles), 2, "both complete physical profiles")
        for index, (profile, rate, count, observed) in enumerate(zip(profiles, (.001, .0001), (45, 112), (257, 252))):
            for key, expected in (("schema_version", 1), ("status", STATUS), ("resource_domain", "literal-n4-port-explicit-opt-in-v1"),
                                  ("profile", {**PROFILE, "physical_error_rate": rate}), ("complete_algorithm_attempts", TRIALS),
                                  ("physical_failure_budget", BUDGET), ("factory_availability_budget", BUDGET),
                                  ("source_event_counts", SOURCE_COUNTS), ("synthesized_event_counts", GATE_COUNTS),
                                  ("workspace_initializations_per_attempt", 23), ("feasible_candidate_count", count),
                                  ("verification_fingerprint", FINGERPRINT), ("actual_synthesis", results["candidate"]["candidate_report"])):
                equal(profile[key], expected, "profile " + key)
            rows = profile["all_feasible_points"]
            equal(len(rows), count, "complete feasible set")
            equal(digest(rows), ROW_IDENTITIES[index]["all_feasible_points"], "exact complete feasible-set identity")
            equal(len({digest(row) for row in rows}), count, "no duplicate physical rows")
            expected_distances = ({d: (3 if d == 37 else 7) for d in range(37, 50, 2)} if index == 0
                                  else {d: 7 for d in range(19, 50, 2)})
            equal(dict(Counter(row["data_code_distance"] for row in rows)), expected_distances, "complete distance population")
            for row in rows:
                verify_point(row, rate)
            frontier = [row for row in rows if not any(
                other["peak_physical_qubits"] <= row["peak_physical_qubits"]
                and other["runtime_per_complete_attempt_ns"] <= row["runtime_per_complete_attempt_ns"]
                and (other["peak_physical_qubits"] < row["peak_physical_qubits"]
                     or other["runtime_per_complete_attempt_ns"] < row["runtime_per_complete_attempt_ns"])
                for other in rows)]
            frontier.sort(key=lambda row: (row["peak_physical_qubits"], row["runtime_per_complete_attempt_ns"]))
            equal(profile["pareto_points"], frontier, "Pareto set from every saved feasible row")
            equal(len(frontier), 3, "three Pareto points")
            equal(digest(frontier), ROW_IDENTITIES[index]["pareto_points"], "exact Pareto identity")
            native = profile["native_execution"]
            for key, expected in (("factory_queries", 24), ("surface_code_queries", 24),
                                  ("factory_candidates_observed_within_cap", observed), ("trace_estimates", count),
                                  ("candidate_overflow_sentinel_observed", False), ("whole_profile_discarded_on_overflow", True),
                                  ("external_process_watchdog_required", True)):
                equal(native[key], expected, "native dispatch counter/guard " + key)
            equal(native["limits"], {"factory_queries": 24, "surface_code_queries": 24, "candidates_per_query": 1000,
                                     "candidates_per_profile": 24000, "traces_per_candidate": 1, "wall_seconds": 180,
                                     "rss_bytes": 2147483648}, "native search limits")
            finite_number(native["elapsed_seconds"], "native accounting time", maximum=180)
            execution = profile["execution_status"]
            equal(execution["status"], "completed", "native profile completion")
            finite_number(execution["wall_seconds"], "native whole-profile time", maximum=180)
            finite_number(execution["peak_rss_bytes"], "native whole-profile RSS", minimum=1, maximum=CAPS["rss_bytes"])
            # The original serialized profile value is present; raw Trace JSON is not.
            payload = json.dumps({k: v for k, v in profile.items() if k != "execution_status"}, indent=2, allow_nan=False) + "\n"
            equal(hashlib.sha256(payload.encode()).hexdigest(), execution["value_sha256"], "saved native profile value-JSON identity")
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("port physical completion/counter/ledger record is incomplete or malformed") from error


def verify_records(manifest, results, progress, source_proof, catalog, recipe, baseline):
    """Semantic completion checks, followed by immutable whole-record identities."""
    try:
        equal(manifest["status"], "completed_candidate_and_bounded_native_profiles", "manifest completion")
        equal(results["status"], manifest["status"], "results completion")
        equal(manifest["schema_version"], 1, "schema")
        equal(results["schema_version"], 1, "results schema")
        equal(manifest["python"], "3.12.14", "locked Python")
        equal(manifest["packages"], PACKAGES, "locked packages")
        equal(manifest["run_caps"], CAPS, "bounded execution caps")
        equal(manifest["source_only"], False, "final execution scope")
        equal(manifest["hardware_execution"], False, "unexecuted hardware")
        equal(set(manifest["artifacts_sha256"]), BUNDLE_FILES - {"manifest.json", "progress.json"}, "31-artifact inventory")
        equal(set(manifest["source_sha256"]), set(SOURCE_FILES) - {"run_simulation_lindblad_port.py", "lindblad_port_resources.py",
                                                                      "LICENSE.txt", "requirements-lock.txt", "references.bib"}, "ten-module freeze inventory")
        equal(progress["status"], "completed", "progress completion")
        equal(progress["candidate_status"], manifest["status"], "progress candidate completion")
        equal(progress["results_sha256"], RESULTS_SHA256, "progress results binding")
        finite_number(progress["peak_rss_bytes"], "whole-child peak RSS", minimum=1, maximum=CAPS["rss_bytes"])
        execution = results["execution"]
        for key, expected in (("candidate_density_executed_once", True), ("historical_densities_rerun", False),
                              ("new_synthesis_calls", 0), ("checkpoint_restores_verified_capability", False),
                              ("frozen_source_commit", "68ba6b4c580c3401cea9266efdee173c776085c0"), ("run_caps", CAPS)):
            equal(execution[key], expected, "execution " + key)
        finite_number(execution["candidate_wall_seconds"], "candidate time", maximum=1800)
        finite_number(execution["candidate_peak_rss_bytes"], "candidate boundary RSS", minimum=1, maximum=progress["peak_rss_bytes"])
        for record in (source_proof, catalog, recipe):
            for key in ("candidate_materialized", "candidate_full_density_executed"):
                if key in record:
                    equal(record[key], False, "frozen source-only " + key)
            for key in ("new_synthesis_calls", "native_physical_queries"):
                equal(record[key], 0, "frozen source-only " + key)
        equal(recipe["full_density_executed"], False, "source-only density")
        equal(recipe["hardware_workload_executed"], False, "source-only hardware")
        equal(recipe["physical_feasibility_assessed"], False, "source-only feasibility")
        equal(source_proof["planned_counts"], recipe["plan"], "source/recipe counts")
        plan = recipe["plan"]
        for key, expected in (("added_cx", 123864), ("eligible_rotation_blocks", 5908), ("fallback_rotation_blocks", 0),
                              ("planned_operations", 1801770), ("planned_gate_counts", GATE_COUNTS),
                              ("planned_t_wire_counts", [642284, 0, 0, 0, 0, 0, 0, 0])):
            equal(plan[key], expected, "frozen plan " + key)
        equal(len(plan["source_event_records"]), 4467, "all ordered source events")
        blocks = plan["rotation_blocks"]
        equal(len(blocks), 5908, "all word occurrences")
        equal(sum(block["added_cx"] for block in blocks), 123864, "all block transport CX")
        equal(catalog["word_count"], 8, "eight native words")
        equal(len(catalog["words"]), 8, "complete word catalogue")
        equal([word["operations"] for word in catalog["words"]], [462, 233, 225, 225, 230, 230, 244, 244], "word lengths")
        equal(sum(word["occurrences"] for word in catalog["words"]), 5908, "catalogue occurrence count")
        equal(sum(word["operations"] for word in catalog["words"]), 2093, "unique-word primitives")
        words = {word["word_id"]: word for word in catalog["words"]}
        equal(set(words), {f"word-{index:02d}" for index in range(8)}, "exact word identities")
        previous_end = 0
        previous_routed = -1
        occurrences = Counter()
        for block in blocks:
            wire = block["original_wire"]
            equal(isinstance(wire, int) and not isinstance(wire, bool) and 0 <= wire < 8, True, "word original wire")
            word = words[block["word_id"]]
            equal(block["eligible"], True, "initialized eligible word")
            equal(block["decision"], "path_initialized", "initialized word decision")
            equal(block["target_wire"], 0, "physical word port")
            equal(block["initialized_mask"] & ((1 << (wire+1))-1), (1 << (wire+1))-1, "complete initialized path")
            equal(block["forward_swap_edges"], [[edge, edge+1] for edge in range(wire-1, -1, -1)], "descending SWAP path")
            equal(block["reverse_swap_edges"], [[edge, edge+1] for edge in range(wire)], "restoring SWAP path")
            equal(block["added_cx"], 6*wire, "literal three-CX round trips")
            equal(block["ct_operations"], word["operations"], "complete native word length")
            equal(block["rotation_key"], [word["rotation_gate"], word["angle_hex"]], "word rotation identity")
            if not (previous_end <= block["ct_start"] and block["routed_index"] > previous_routed):
                raise ValueError("port recipe word ranges overlap or are reordered")
            previous_end = block["ct_start"] + block["ct_operations"]
            previous_routed = block["routed_index"]
            occurrences[block["word_id"]] += 1
        equal(dict(occurrences), {key: word["occurrences"] for key, word in words.items()}, "every catalogue occurrence")
        equal(catalog["recovery_sha256"], CATALOG_RECOVERY_SHA256, "logical recovered catalogue identity")
        candidate = results["candidate"]
        for key, expected in (("status", "verified_complete_port_candidate"), ("case_id", "lindblad-n4-r211-Z0"),
                              ("candidate_density_execution_count", 1), ("candidate_full_density_executed", True),
                              ("historical_density_execution_count", 0), ("new_synthesis_calls", 0), ("native_physical_queries", 0),
                              ("physical_feasibility_claimed", False), ("recipe_sha256", RECIPE_SHA256),
                              ("catalog_sha256", CATALOG_RECOVERY_SHA256)):
            equal(candidate[key], expected, "candidate " + key)
        report = candidate["candidate_report"]
        equal(report["gate_counts"], GATE_COUNTS, "actual primitive counts")
        equal(report["t_wire_counts"], [642284, 0, 0, 0, 0, 0, 0, 0], "all T on physical wire zero")
        equal(report["added_cx"], 123864, "actual added CX")
        equal(report["actual_operations"], sum(GATE_COUNTS.values()), "actual operations")
        equal(report["global_phase_binary64_hex"], GLOBAL_PHASE, "authoritative whole phase")
        equal(results["candidate_verification_fingerprint"], FINGERPRINT, "candidate fingerprint")
        workload = candidate["complete_workload"]
        baseline_prepared = baseline["prepared"]
        equal(results["baseline_saved_prepared"], baseline_prepared, "unaltered saved n4 prepared record")
        equal(candidate["saved_baseline_complete_workload"], baseline_prepared["complete_workload"], "original population table")
        equal(candidate["error_contract"], baseline_prepared["error_contract"], "same error contract")
        equal(recipe["matched_contract"]["complete_workload"], baseline_prepared["complete_workload"], "recipe same original task")
        for key, value in baseline_prepared["complete_workload"].items():
            if not key.startswith("primitive_counts_"):
                equal(workload[key], value, "unchanged workload " + key)
        primitive = {key: count for key, count in GATE_COUNTS.items() if key != "barrier"}
        equal(workload["primitive_counts_per_complete_trial"], primitive, "actual primitive table")
        equal(workload["primitive_counts_all_trials"], {key: count*TRIALS for key, count in primitive.items()}, "full population table")
        equal(sum(workload["primitive_counts_all_trials"].values()), 28341766752, "full nonbarrier population")
        for key, expected in (("physical_failure_exact", {"numerator": 1, "denominator": 1200}),
                              ("factory_availability_failure_exact", {"numerator": 1, "denominator": 1200}),
                              ("statistical_failure_exact", {"numerator": 1, "denominator": 600}),
                              ("complete_trials", TRIALS), ("whole_workload_executed", False),
                              ("deterministic_bias_multiplied_by_trials", False), ("family_allocations_reallocated", False)):
            equal(workload[key], expected, "fixed full workload " + key)
        equal(len(results["controller_eligibility"]), 2, "both controller eligibility records")
        for record in results["controller_eligibility"]:
            for key, expected in (("event_counts_per_trial", EVENT_COUNTS), ("required_magic_states_all_trials", STATES),
                                  ("controller_decisions_per_trial", 50713680), ("controller_decisions_all_trials", 50713680*TRIALS),
                                  ("maximum_allowed_controller_decisions_per_trial", 52889904), ("factory_retry_cap", 7),
                                  ("distance_independent_controller_model_contribution", 50713680*TRIALS*1e-15),
                                  ("physical_failure_budget", BUDGET), ("factory_availability_budget", BUDGET),
                                  ("factory_readiness_failure_union_bound", STATES/100**7), ("budget_exceeded", False),
                                  ("eligibility_is_physical_feasibility", False), ("verification_fingerprint", FINGERPRINT),
                                  ("status", "controller_eligible_for_bounded_native_search")):
                equal(record[key], expected, "controller/readiness " + key)
        if not Fraction(STATES, 100**7) <= Fraction(1, 1200) < Fraction(STATES, 100**6):
            raise ValueError("port exact readiness retry cap is not minimal")
        verify_baseline(baseline)
        verify_profiles(results)
        for name, record in zip(RECORD_IDENTITIES, (manifest, results, progress, source_proof, catalog, recipe)):
            equal(digest(record), RECORD_IDENTITIES[name], "immutable saved " + name + " identity")
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("port completion/identity/status record is incomplete or malformed") from error


def verify_inventory(bundle, manifest, inventory):
    try:
        equal(inventory["schema_version"], 1, "inventory schema")
        equal(inventory["payload_files"], 29, "inventory payload count")
        equal(inventory["payload_bytes"], 26275842, "inventory payload bytes")
        rows = inventory["files"]
        equal([row["file"] for row in rows], sorted(BUNDLE_FILES - EXCLUDED_METADATA), "complete inventory allowlist")
        for row in rows:
            equal(set(row), {"file", "bytes", "sha256", "category"}, "inventory row keys")
            path = checked_file(bundle, row["file"])
            equal(row["bytes"], path.stat().st_size, "inventory file bytes")
            equal(row["sha256"], manifest["artifacts_sha256"][row["file"]], "inventory manifest binding")
            equal(row["sha256"], sha256(path), "inventory actual file digest")
            category = ("source_snapshot" if row["file"].endswith(".py") else
                        "phased_word_export" if row["file"] in WORD_FILES else
                        "exact_recipe" if row["file"] == "recipe.json" else "result_or_provenance")
            equal(row["category"], category, "inventory category")
        equal(sum(row["bytes"] for row in rows), 26275842, "inventory total")
        with checked_file(bundle, "artifact-inventory.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            equal(reader.fieldnames, ["file", "bytes", "sha256", "category"], "CSV inventory columns")
            csv_rows = list(reader)
        for row in csv_rows:
            row["bytes"] = int(row["bytes"])
        equal(csv_rows, rows, "CSV/JSON inventory identity")
        equal(digest(inventory), INVENTORY_IDENTITY, "immutable complete inventory identity")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("port inventory identity is incomplete or malformed") from error


def verify_bundle(bundle, *, source_root=None, canonical_root=None):
    bundle = checked_directory(bundle)
    source_root = checked_directory(bundle.parents[3] if source_root is None else source_root)
    canonical_root = checked_directory(source_root if canonical_root is None else canonical_root)
    names = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("port linked bundle entry")
        if path.is_dir():
            if path.relative_to(bundle).as_posix() != "words":
                raise ValueError("port unrecorded bundle directory")
        else:
            names.add(path.relative_to(bundle).as_posix())
    equal(names, BUNDLE_FILES, "complete 33-file bundle")
    equal(sha256(checked_file(bundle, "manifest.json")), REVIEWED_MANIFEST_SHA256, "immutable reviewed manifest")
    manifest = load_json(checked_file(bundle, "manifest.json"))
    for name, expected in manifest["artifacts_sha256"].items():
        equal(sha256(checked_file(bundle, name)), expected, "attested artifact digest " + name)
    equal(sha256(checked_file(bundle, "progress.json")), PROGRESS_SHA256, "completed progress digest")
    equal(sum(checked_file(bundle, name).stat().st_size for name in BUNDLE_FILES), 26290618, "complete bundle bytes")
    for name, relative in SOURCE_FILES.items():
        raw = (checked_file(source_root, relative).read_bytes() if relative is not None else
               checked_file(source_root, "references/simulation.bib").read_bytes() + b"\n" +
               checked_file(source_root, "references/resources.bib").read_bytes())
        equal(hashlib.sha256(raw).hexdigest(), manifest["artifacts_sha256"][name], "current source digest " + name)
    # Public package verification binds every saved n2/n3/n4 scientific bundle.
    # Private historical plans are intentionally not an export requirement.
    for relative, expected in PRESERVED_SCIENTIFIC_FILES.items():
        path = checked_file(canonical_root, relative)
        equal(path.stat().st_size, expected["bytes"], "preserved scientific file size")
        equal(sha256(path), expected["sha256"], "preserved scientific file digest " + relative)
    baseline = load_json(checked_file(canonical_root, "assets/generated/simulation/lindblad-four/results.json"))
    records = [load_json(checked_file(bundle, name)) for name in RECORD_IDENTITIES]
    verify_records(*records, baseline)
    verify_inventory(bundle, manifest, load_json(checked_file(bundle, "artifact-inventory.json")))
    return {"files": 33, "bytes": 26290618, "manifest_sha256": REVIEWED_MANIFEST_SHA256,
            "saved_numerical_trial_checks_completed": True, "physical_status": STATUS,
            "feasible_points": [45, 112], "pareto_points": [3, 3], "complete_trials": TRIALS,
            "preserved_scientific_files": len(PRESERVED_SCIENTIFIC_FILES), "original_controller_status": "infeasible_declared_controller_model",
            "native_trace_digest_status": "audit-trusted; raw Trace JSON and inline audit script/hash were not saved",
            "saved_native_profile_value_json_verified": True, "native_calls": 0, "synthesis_calls": 0,
            "density_replays": 0, "repeated_trials_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Source/package root with reviewed canonical assets")
    parser.add_argument("--source-root", type=Path, help="Optional independent exported source-preview root")
    parser.add_argument("--canonical-root", type=Path, help="Optional root with saved prior scientific bundles")
    args = parser.parse_args()
    receipt = verify_bundle(args.root / BUNDLE_RELATIVE, source_root=args.source_root or args.root,
                            canonical_root=args.canonical_root or args.root)
    print("Verified immutable port saved evidence:", json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
