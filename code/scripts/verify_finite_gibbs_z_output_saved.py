#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the fixed finite-Gibbs public packet from saved bytes only.

This standard-library gate hashes the independently pinned manifest and every
listed file before checking saved scope metadata. Scientific source, QASM and
matrix/interval values remain evidence: no source module is imported, no circuit
or matrix is constructed, no numerical comparison or interval arithmetic runs,
and the prepared portable wrapper is never invoked. Recorded observed, final
and whole-invocation flags remain separate from the curated conditional Z-mean
acceptance. Byte authentication is not a new numerical or publication pass.

The API has cooperative checks within at most 30 seconds. The CLI additionally
installs a real-time alarm and a non-enlarging 30-second CPU ceiling. Neither
mode extracts files or writes a receipt. No invocation follows from this source.
"""
from __future__ import annotations

import time
_ENTRY_TIME = time.monotonic()

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys

BASE = "assets/generated/simulation/finite-gibbs-z-output"
MANIFEST_NAME = "manifest.json"
EXPECTED_MANIFEST_SHA256 = "a490a8c5c1c941ce7a1b284de40879047bc21a8d3491aeceee0f5b05a878faa6"
MAX_SECONDS = 30.0
CHUNK_BYTES = 64 << 10
MAX_FILE_BYTES = 2 << 20
MAX_MANIFEST_BYTES = 16 << 10
MAX_BUNDLE_BYTES = 4 << 20
MAX_FILES = 22
MAX_ENTRIES = 64
MAX_OUTPUT_BYTES = 32 << 10

EXPECTED_PAYLOAD_PINS = {'LICENSE.txt': {'bytes': 1093,
                 'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
 'README.md': {'bytes': 6886,
               'sha256': 'bec600aeae6f29a94473e82edd2a02bd69004456e8bcb3ff9aa828c5c4ac95c4'},
 'REPRODUCTION.md': {'bytes': 9577,
                     'sha256': '769e714b5972c1315633a224cf45c61290344c704f6dfe88979c462d966860f3'},
 'UPSTREAM-LICENSES.md': {'bytes': 2699,
                          'sha256': '3350b7fe3dcf0be4f4036ea10290c30cb60c3bbaf34ef9d031e0df466ca88eb9'},
 'circuits/access-U.qasm': {'bytes': 3065,
                            'sha256': '93ab81b8d5008d886a2096f97e69b590985980546a4b60fd68128db0b0d19a65'},
 'circuits/preparation-R2.qasm': {'bytes': 44741,
                                  'sha256': '83cefa8e3fef9920a72f4e58f1312b29a8f93f0b18c63dfb4ce7424ba713bedc'},
 'circuits/preparation-R64.qasm': {'bytes': 1428581,
                                   'sha256': 'e584fa94fffe1618f2029c8f311e5b0eb7a188f514ca019a4e34feee461076b8'},
 'circuits/reset-round.qasm': {'bytes': 22370,
                               'sha256': 'ee2404c83e2f075dba35dbf02555432b385b9c25dee26bfb4f0d0ee576f0268d'},
 'circuits/weak-dilation.qasm': {'bytes': 22226,
                                 'sha256': 'e11625b24787d2513ce3bb4af95742c31b90734209183b797cc92d3297f0ffc3'},
 'finite-gibbs-z-circuit.svg': {'bytes': 7173,
                                'sha256': '20b8a3a4c7f929cfdb667bd3f909f76ea431b2d65bbc433c8c5a3bded726f764'},
 'references.bib': {'bytes': 1277,
                    'sha256': '45237554b2e3b5e77575f7cb2d31f36a3ca6e3ace8c74419c0e494dccf141a94'},
 'requirements-lock.txt': {'bytes': 1974,
                           'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
 'results.json': {'bytes': 1715997,
                  'sha256': '8329d568e2df8f317f97610165385123f52dc0f5d47c6aa190838405966ca33f'},
 'source-derivations.json': {'bytes': 7075,
                             'sha256': '40c86eea4d560df0012d102970fa735857181a423b9498129ad2604d534865e3'},
 'sources/detailed_balance_access.py': {'bytes': 25361,
                                        'sha256': '7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77'},
 'sources/detailed_balance_channel.py': {'bytes': 24362,
                                         'sha256': '99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b'},
 'sources/finite_gibbs_angle_provenance.py': {'bytes': 27630,
                                              'sha256': '175de0ead4e4e32ee679c42831662682e58c1467688d4bfe08ffaec6265801c1'},
 'sources/finite_gibbs_interval.py': {'bytes': 28159,
                                      'sha256': '8f035148771d8659057cd7f013d0378ec5d9995f4e2fe255bc453b348d776362'},
 'sources/finite_gibbs_reference.py': {'bytes': 4693,
                                       'sha256': '1f82517c1992f93ec3fa7af49982286e64c7cf011b69a07bdc20ac1dbb956ac4'},
 'sources/reproduce.py': {'bytes': 29996,
                          'sha256': 'a368f179c797f25246cb1ef89055b582d7d26164f041a584c09ab22a57cb597f'},
 'summary.json': {'bytes': 11003,
                  'sha256': '1f974ae6eb52952fa8439b1526c359c47195d843488632372dfe2c694285a8cf'}}

EXPECTED_MANIFEST_HEADER = {'computation_replayed_for_projection': False,
 'data_whitelist_record': 'summary.json#data_projection',
 'date': '2026-11-15',
 'derivation_record': 'source-derivations.json',
 'exclusions': ['full-state accuracy',
                'general thermal sampler or mixing theorem',
                'uniform continuum/KMS bridge',
                'rotation synthesis',
                'physical faults/timing/independence',
                'FT resource advantage',
                'floating backend rounding certificate',
                'mechanized proof'],
 'fixed_conditional_Z_mean_contract_accepted': True,
 'floating_backend_error_certified': False,
 'formal_proof_reproduced': False,
 'full_R64_primitive_simulation_performed': False,
 'full_state_Gibbs_accuracy_certified': False,
 'general_sampler_or_mixing_theorem': False,
 'measurement_projection': {'iid_ideal_Born_assumption': True,
                            'samples': 16384,
                            'sampling_performed': False,
                            'total_work_formula': '16384*(64*1505+3)'},
 'original_raw_flags_preserved': {'finite_gibbs_Z_output_bound_observed': True,
                                  'finite_gibbs_Z_output_bound_passed': False,
                                  'status': 'observed_fixed_finite_Gibbs_Z_components',
                                  'whole_invocation_validated': False},
 'original_root_acceptance_sha256': '6d2c086b6e175e394b86b92fc9e28d2dc8008af9004f5d6c003314bad9a86f75',
 'original_saved_result_sha256': 'f5e0b772fa2005f415af19d3cb4ba94395753a9172a308dd868b8058acbb5129',
 'packaging_status': 'private_source_proposal_unexecuted',
 'payload_scope': 'Only listed regular public files; manifest.json itself is excluded from its '
                  'own payload hashes.',
 'physical_resources_computed': False,
 'portable_wrapper_executed': False,
 'portable_wrapper_runtime_validated': False,
 'publication_performed': False,
 'rotations_synthesized': False,
 'schema': 'openquantum.finite_gibbs.public_projection_manifest.v1',
 'scientific_inventory': {'external_density_evolutions': 1,
                          'fixed_interval_certificate_evaluations': 1,
                          'independent_full_U_Kraus_constructions': 1,
                          'matrix_power_calls': 2,
                          'statevector_evolutions': 2},
 'scientific_source_commit': '47ffd10368a738f9576053491a56c6d80a10e602',
 'scope': 'Separately accepted fixed ideal conditional Z-mean contract for H=Z, beta=1, '
          'known-plus input, M=16, delta=1/8, R=64 and 16384 hypothetical independent ideal '
          'Born measurements.',
 'shots_drawn': 0,
 'source_freeze_sha256': '08a18e7c6b6cf4a8c0e836ed09e7725dfaaadae4cfb1a6c5d11197ed8d2208f9',
 'source_operation_counts': {'access': 140,
                             'dilation': 1493,
                             'preparation_R2': 3013,
                             'preparation_R64': 96323,
                             'reset_round': 1505}}

EXPECTED_SUMMARY_FIELDS = {'budget_checks': {'angle_realization_at_most_0.002': True,
                   'digital_transient_at_most_0.002': True,
                   'finite_time_transient_at_most_0.012': True,
                   'measurement_failure_at_most_0.050': True,
                   'mesh_stationary_at_most_0.004': True,
                   'preparation_total_at_most_0.020': True,
                   'total_error_at_most_0.050': True},
 'data_projection': {'all_whitelisted_scientific_subtrees_copied_without_value_changes': True,
                     'omission_scope': 'Operational resource caps/timing, backend host/native '
                                       'origins, startup-policy records and delivery '
                                       'bookkeeping are omitted. Scientific scope flags are '
                                       'retained. Generic native identity requirements and '
                                       'numeric binary64 evidence are scientific authenticity '
                                       'fields.',
                     'original_top_level_fields_omitted': ['caps',
                                                           'cpu_seconds_at_report',
                                                           'decimal_backend',
                                                           'loaded_module_origins',
                                                           'resource_observation_scope',
                                                           'schema',
                                                           'startup_policy_after_import',
                                                           'startup_policy_entry',
                                                           'startup_policy_final',
                                                           'terminal_acceptance_record',
                                                           'terminal_acceptance_required',
                                                           'wall_seconds_at_report'],
                     'scoped_acceptance_whitelist': ['actual_shots',
                                                     'all_18_implementation_5_domain_7_budget_predicates_passed',
                                                     'date',
                                                     'fixed_conditional_Z_mean_contract_accepted',
                                                     'floating_backend_error_certified',
                                                     'formal_proof_reproduced',
                                                     'full_R64_primitive_simulation_performed',
                                                     'full_state_Gibbs_accuracy_certified',
                                                     'general_sampler_or_mixing_theorem',
                                                     'parameters',
                                                     'physical_resources_computed',
                                                     'raw_final_flags_unchanged',
                                                     'rotations_synthesized',
                                                     'source_commit',
                                                     'source_freeze_sha256',
                                                     'whole_invocation_validated'],
                     'selected_actual_basis_columns': ['0', '1'],
                     'selected_matrix_keys': ['R2_expected_joint_state',
                                              'R2_joint_state',
                                              'R2_reduced_state',
                                              'actual_R64_state',
                                              'actual_all64_kraus',
                                              'actual_channel',
                                              'reference_R64_state',
                                              'reference_all64_kraus',
                                              'reference_channel',
                                              'reference_full_U'],
                     'top_level_scientific_whitelist': ['accuracy_scope',
                                                        'actual_basis_columns',
                                                        'all_components_verified',
                                                        'angle_provenance',
                                                        'backend_internal_Operator_SuperOp_constructors',
                                                        'channel_implemented',
                                                        'coherent_correction_evolution_implemented',
                                                        'coherent_correction_implemented',
                                                        'counters',
                                                        'density_source',
                                                        'exact_finite_kms',
                                                        'excluded',
                                                        'expected_internal_reset_dispatches',
                                                        'finite_gibbs_Z_output_bound_observed',
                                                        'finite_gibbs_Z_output_bound_passed',
                                                        'finite_mesh_exact_kms',
                                                        'floating_point_error_certified',
                                                        'formal_proof_reproduced',
                                                        'general_channel_implemented',
                                                        'general_coherent_correction_implemented',
                                                        'hardware_error_certified',
                                                        'implementation_checks',
                                                        'implementation_checks_passed',
                                                        'internal_reset_dispatches_measured',
                                                        'interval_certificate',
                                                        'matrices',
                                                        'measurement_projection',
                                                        'mesh_record',
                                                        'mixing_certificate',
                                                        'new_algorithm_result',
                                                        'parameters',
                                                        'physical_resources_computed',
                                                        'preprocessing_counters',
                                                        'random_sampling_performed',
                                                        'raw_population_diagnostics',
                                                        'raw_residuals',
                                                        'rotations_synthesized',
                                                        'runtime_versions',
                                                        'sampler_implemented',
                                                        'source_commit',
                                                        'source_counts',
                                                        'source_freeze_sha256',
                                                        'source_realization',
                                                        'source_summaries',
                                                        'status',
                                                        'tolerances',
                                                        'whole_invocation_validated',
                                                        'zero_Operator_SuperOp_requests_scope']},
 'density_source': {'primitives': 3012, 'resets': 25, 'terminal_measures_removed': 1},
 'domain_checks': {'d_positive_lower': True,
                   'population_step_at_most_one': True,
                   'population_step_nonnegative': True,
                   'rate_sum_positive_lower': True,
                   'u_positive_lower': True},
 'excluded': ['full-state accuracy',
              'general thermal sampler or mixing theorem',
              'uniform continuum/KMS bridge',
              'rotation synthesis',
              'physical faults/timing/independence',
              'FT resource advantage',
              'floating backend rounding certificate',
              'mechanized proof'],
 'fixed_conditional_Z_mean_contract_accepted': True,
 'implementation_checks': {'R2_joint_entry_error': True,
                           'R2_joint_trace_error': True,
                           'R2_reduced_entry_error': True,
                           'R64_actual_hermiticity_error': True,
                           'R64_actual_trace_error': True,
                           'R64_reference_trace_error': True,
                           'actual_kraus_completeness_entry_error': True,
                           'actual_reference_R64_state_entry_error': True,
                           'actual_reference_channel_entry_error': True,
                           'all64_kraus_entry_error': True,
                           'reference_R64_Z_vs_exact_interval_diagnostic': True,
                           'reference_X_jump_diagonal_error': True,
                           'reference_Z_jump_offdiagonal_error': True,
                           'reference_d_vs_exact_interval_diagnostic': True,
                           'reference_full_U_unitarity_entry_error': True,
                           'reference_kraus_completeness_entry_error': True,
                           'reference_nonjump_offdiagonal_error': True,
                           'reference_u_vs_exact_interval_diagnostic': True},
 'measurement_projection': {'iid_ideal_Born_assumption': True,
                            'samples': 16384,
                            'sampling_performed': False,
                            'total_work_formula': '16384*(64*1505+3)',
                            'whole_preparation_primitives_each': 96323},
 'original_result_sha256': 'f5e0b772fa2005f415af19d3cb4ba94395753a9172a308dd868b8058acbb5129',
 'packaging_status': 'private_source_proposal_unexecuted',
 'parameters': {'M': 16,
                'beta': 1,
                'clock_bits': 4,
                'delta': 0.125,
                'initial': 'plus',
                'measurement_samples_projected': 16384,
                'product_steps': 1,
                'proposal_labels': ['X', 'Z'],
                'rounds': 64,
                'terms': [[1, 'Z']]},
 'raw_final_flags': {'finite_gibbs_Z_output_bound_observed': True,
                     'finite_gibbs_Z_output_bound_passed': False,
                     'status': 'observed_fixed_finite_Gibbs_Z_components',
                     'whole_invocation_validated': False},
 'schema': 'openquantum.finite_gibbs.public_summary.v1',
 'scientific_inventory': {'external_density_evolutions': 1,
                          'fixed_interval_certificate_evaluations': 1,
                          'independent_full_U_Kraus_constructions': 1,
                          'matrix_power_calls': 2,
                          'statevector_evolutions': 2},
 'scope': 'Fixed ideal conditional Z-mean contract for H=Z, beta=1, known-plus input, M=16, '
          'delta=1/8 and R=64; future fresh iid ideal Born measurements only.',
 'source_commit': '47ffd10368a738f9576053491a56c6d80a10e602',
 'source_counts': {'access': 140,
                   'dilation': 1493,
                   'preparation_R2': 3013,
                   'preparation_R64': 96323,
                   'reset_round': 1505},
 'source_freeze_sha256': '08a18e7c6b6cf4a8c0e836ed09e7725dfaaadae4cfb1a6c5d11197ed8d2208f9'}

EXPECTED_RESULT_FIELDS = {'accuracy_scope': 'fixed Gibbs Z observable of emitted ideal numeric primitives, conditional '
                   'on authenticated angle/interval proofs',
 'expected_internal_reset_dispatches': 25,
 'implementation_checks': {'R2_joint_entry_error': True,
                           'R2_joint_trace_error': True,
                           'R2_reduced_entry_error': True,
                           'R64_actual_hermiticity_error': True,
                           'R64_actual_trace_error': True,
                           'R64_reference_trace_error': True,
                           'actual_kraus_completeness_entry_error': True,
                           'actual_reference_R64_state_entry_error': True,
                           'actual_reference_channel_entry_error': True,
                           'all64_kraus_entry_error': True,
                           'reference_R64_Z_vs_exact_interval_diagnostic': True,
                           'reference_X_jump_diagonal_error': True,
                           'reference_Z_jump_offdiagonal_error': True,
                           'reference_d_vs_exact_interval_diagnostic': True,
                           'reference_full_U_unitarity_entry_error': True,
                           'reference_kraus_completeness_entry_error': True,
                           'reference_nonjump_offdiagonal_error': True,
                           'reference_u_vs_exact_interval_diagnostic': True},
 'implementation_checks_passed': True,
 'measurement_projection': {'iid_ideal_Born_assumption': True,
                            'samples': 16384,
                            'sampling_performed': False,
                            'total_work_formula': '16384*(64*1505+3)',
                            'whole_preparation_primitives_each': 96323},
 'original_result_schema': 'openquantum.finite_gibbs.fixed_output_result.v1',
 'original_result_sha256': 'f5e0b772fa2005f415af19d3cb4ba94395753a9172a308dd868b8058acbb5129',
 'packaging_status': 'private_source_proposal_unexecuted',
 'parameters': {'M': 16,
                'beta': 1,
                'clock_bits': 4,
                'delta': 0.125,
                'initial': 'plus',
                'measurement_samples_projected': 16384,
                'product_steps': 1,
                'proposal_labels': ['X', 'Z'],
                'rounds': 64,
                'terms': [[1, 'Z']]},
 'projection_operations': 'Saved JSON field selection and file copying only; no scientific '
                          'recomputation.',
 'schema': 'openquantum.finite_gibbs.public_scientific_projection.v1',
 'scoped_acceptance': {'acceptance_record_sha256': '6d2c086b6e175e394b86b92fc9e28d2dc8008af9004f5d6c003314bad9a86f75',
                       'actual_shots': 0,
                       'all_18_implementation_5_domain_7_budget_predicates_passed': True,
                       'date': '2026-11-15',
                       'fixed_conditional_Z_mean_contract_accepted': True,
                       'floating_backend_error_certified': False,
                       'formal_proof_reproduced': False,
                       'full_R64_primitive_simulation_performed': False,
                       'full_state_Gibbs_accuracy_certified': False,
                       'general_sampler_or_mixing_theorem': False,
                       'parameters': {'H': 'Z',
                                      'M': 16,
                                      'R': 64,
                                      'beta': 1,
                                      'delta': '1/8',
                                      'hypothetical_iid_measurements': 16384,
                                      'input': 'plus',
                                      'measurement_allowance': '0.030',
                                      'preparation_allowance': '0.020',
                                      'sampling_failure_allowance': '0.050',
                                      'total_error_allowance': '0.050'},
                       'physical_resources_computed': False,
                       'raw_final_flags_unchanged': True,
                       'rotations_synthesized': False,
                       'schema': 'openquantum.finite_gibbs.public_scoped_acceptance.v1',
                       'science_review_sha256': '915ad6f5ecccd2046d8bc8057597fdd3a06a5f033b82f6e370e23f9fa7294ec1',
                       'scope': 'Fixed ideal conditional Z-mean contract for H=Z, beta=1, '
                                'known-plus input, M=16, delta=1/8 and R=64; future fresh iid '
                                'ideal Born measurements only.',
                       'source_commit': '47ffd10368a738f9576053491a56c6d80a10e602',
                       'source_freeze_sha256': '08a18e7c6b6cf4a8c0e836ed09e7725dfaaadae4cfb1a6c5d11197ed8d2208f9',
                       'whole_invocation_validated': False},
 'source_commit': '47ffd10368a738f9576053491a56c6d80a10e602',
 'source_counts': {'access': 140,
                   'dilation': 1493,
                   'preparation_R2': 3013,
                   'preparation_R64': 96323,
                   'reset_round': 1505},
 'source_freeze_sha256': '08a18e7c6b6cf4a8c0e836ed09e7725dfaaadae4cfb1a6c5d11197ed8d2208f9',
 'source_realization': 'original native primitive graph; no clipping, trace renormalization or '
                       'phase fitting',
 'status': 'observed_fixed_finite_Gibbs_Z_components'}

RAW_FALSE_FLAGS = ('all_components_verified',
 'channel_implemented',
 'coherent_correction_evolution_implemented',
 'coherent_correction_implemented',
 'exact_finite_kms',
 'finite_gibbs_Z_output_bound_passed',
 'finite_mesh_exact_kms',
 'floating_point_error_certified',
 'formal_proof_reproduced',
 'general_channel_implemented',
 'general_coherent_correction_implemented',
 'hardware_error_certified',
 'internal_reset_dispatches_measured',
 'mixing_certificate',
 'new_algorithm_result',
 'physical_resources_computed',
 'random_sampling_performed',
 'rotations_synthesized',
 'sampler_implemented',
 'whole_invocation_validated')

RAW_FINAL_FLAGS = {'finite_gibbs_Z_output_bound_observed': True,
 'finite_gibbs_Z_output_bound_passed': False,
 'status': 'observed_fixed_finite_Gibbs_Z_components',
 'whole_invocation_validated': False}

SOURCE_NAMES = ('sources/detailed_balance_access.py',
 'sources/detailed_balance_channel.py',
 'sources/finite_gibbs_angle_provenance.py',
 'sources/finite_gibbs_interval.py',
 'sources/finite_gibbs_reference.py',
 'sources/reproduce.py')

SUMMARY_KEYS = ('schema',
 'packaging_status',
 'source_commit',
 'source_freeze_sha256',
 'original_result_sha256',
 'public_results_sha256',
 'fixed_conditional_Z_mean_contract_accepted',
 'scope',
 'parameters',
 'target',
 'source_counts',
 'density_source',
 'scientific_inventory',
 'measurement_projection',
 'raw_population_diagnostics',
 'raw_residuals',
 'implementation_checks',
 'domain_checks',
 'budget_checks',
 'saved_budget_terms',
 'saved_angle_error_sum',
 'saved_mu_beta_interval',
 'raw_final_flags',
 'excluded',
 'data_projection')

RESULT_KEYS = ('schema',
 'original_result_schema',
 'original_result_sha256',
 'packaging_status',
 'projection_operations',
 'scoped_acceptance',
 'accuracy_scope',
 'actual_basis_columns',
 'all_components_verified',
 'angle_provenance',
 'backend_internal_Operator_SuperOp_constructors',
 'channel_implemented',
 'coherent_correction_evolution_implemented',
 'coherent_correction_implemented',
 'counters',
 'density_source',
 'exact_finite_kms',
 'excluded',
 'expected_internal_reset_dispatches',
 'finite_gibbs_Z_output_bound_observed',
 'finite_gibbs_Z_output_bound_passed',
 'finite_mesh_exact_kms',
 'floating_point_error_certified',
 'formal_proof_reproduced',
 'general_channel_implemented',
 'general_coherent_correction_implemented',
 'hardware_error_certified',
 'implementation_checks',
 'implementation_checks_passed',
 'internal_reset_dispatches_measured',
 'interval_certificate',
 'matrices',
 'measurement_projection',
 'mesh_record',
 'mixing_certificate',
 'new_algorithm_result',
 'parameters',
 'physical_resources_computed',
 'preprocessing_counters',
 'random_sampling_performed',
 'raw_population_diagnostics',
 'raw_residuals',
 'rotations_synthesized',
 'runtime_versions',
 'sampler_implemented',
 'source_commit',
 'source_counts',
 'source_freeze_sha256',
 'source_realization',
 'source_summaries',
 'status',
 'tolerances',
 'whole_invocation_validated',
 'zero_Operator_SuperOp_requests_scope')

EXPECTED_DERIVATION_HEADER = {'packaging_executed': False,
 'schema': 'openquantum.finite_gibbs.public_source_derivations.v1',
 'source_commit': '47ffd10368a738f9576053491a56c6d80a10e602',
 'source_freeze_sha256': '08a18e7c6b6cf4a8c0e836ed09e7725dfaaadae4cfb1a6c5d11197ed8d2208f9'}

DERIVATION_FIELDS = {'sources/finite_gibbs_angle_provenance.py': {'all_executable_function_and_class_bodies_byte_identical': True,
                                              'original_source_commit': '47ffd10368a738f9576053491a56c6d80a10e602'},
 'sources/finite_gibbs_interval.py': {'all_executable_function_and_class_bodies_byte_identical': True,
                                      'original_source_commit': '47ffd10368a738f9576053491a56c6d80a10e602'},
 'sources/finite_gibbs_reference.py': {'definition_bodies_byte_identical': True,
                                       'original_source_commit': '47ffd10368a738f9576053491a56c6d80a10e602',
                                       'portable_packaging_executed': False},
 'sources/reproduce.py': {'extends_original_acceptance': False,
                          'portable_wrapper_executed': False,
                          'portable_wrapper_runtime_validated': False,
                          'scientific_source_helpers_executed_in_accepted_calculation': True}}

FILE_NAMES = frozenset(EXPECTED_PAYLOAD_PINS)
JSON_NAMES = frozenset({"results.json", "summary.json", "source-derivations.json"})

def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved-byte operation exceeded its finite deadline")


def _digest(value):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
             "An external lowercase SHA-256 is required")
    return value


def _pin(value):
    _require(type(value) is dict and set(value) == {"bytes", "sha256"},
             "A file pin must contain only bytes and SHA-256")
    _require(type(value["bytes"]) is int and 0 < value["bytes"] <= MAX_FILE_BYTES,
             "A file pin has an invalid finite byte count")
    _digest(value["sha256"])
    return value


def _relative(value):
    _require(type(value) is str and 0 < len(value) <= 256,
             "A bounded bundle-relative name is required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and bool(path.parts) and all(part not in ("", ".", "..") for part in path.parts)
             and chr(0) not in value and chr(92) not in value,
             "A bundle name cannot redirect")
    return value


def _stamp(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink,
            metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


def _root(value):
    _require(isinstance(value, (str, Path)) and len(str(value)) <= 4096,
             "A bounded bundle directory is required")
    if type(value) is str:
        _require("//" not in value and value == str(Path(value)),
                 "A bundle directory cannot use lexical aliases")
    path = Path(value).absolute()
    _require(".." not in path.parts and len(path.parts) <= 64,
             "A bundle directory cannot redirect")
    for parent in reversed((path, *path.parents)):
        _require(stat.S_ISDIR(parent.lstat().st_mode),
                 "Bundle ancestors must be ordinary directories")
    return path


def _directories(names):
    result = {""}
    for name in names:
        for parent in PurePosixPath(name).parents:
            if parent.as_posix() != ".":
                result.add(parent.as_posix())
    return result


@contextmanager
def _directory_descriptor(path, deadline):
    """Walk ordinary directories with no-follow descriptor-relative opens."""
    _require(hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"),
             "This checker requires no-follow directory opens")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            _check(deadline)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            _require(stat.S_ISDIR(os.fstat(descriptor).st_mode),
                     "An opened ancestor is not an ordinary directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _inventory(root, deadline):
    """Read a finite complete tree, rejecting links and unlisted empty trees."""
    expected = FILE_NAMES | {MANIFEST_NAME}
    allowed_directories = _directories(expected)
    files, directories = {}, {"": _stamp(root.lstat())}
    pending = [(root, "")]
    entries = 0
    total_bytes = 0
    while pending:
        _check(deadline)
        directory, prefix = pending.pop()
        with _directory_descriptor(directory, deadline) as descriptor:
            _require(_stamp(os.fstat(descriptor)) == directories[prefix],
                     "A bundle directory identity changed")
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    _check(deadline)
                    entries += 1
                    _require(entries <= MAX_ENTRIES, "Bundle directory entry bound exceeded")
                    name = _relative(prefix + "/" + entry.name if prefix else entry.name)
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(metadata.st_mode):
                        _require(name in allowed_directories,
                                 "Unexpected directory in the public bundle")
                        directories[name] = _stamp(metadata)
                        pending.append((root / name, name))
                    else:
                        _require(name in expected and stat.S_ISREG(metadata.st_mode)
                                 and metadata.st_nlink == 1,
                                 "Bundle files must be listed ordinary nonlinked files")
                        maximum = MAX_MANIFEST_BYTES if name == MANIFEST_NAME else MAX_FILE_BYTES
                        _require(0 < metadata.st_size <= maximum,
                                 "A public file exceeds its finite byte bound")
                        files[name] = _stamp(metadata)
                        total_bytes += metadata.st_size
                        _require(len(files) <= MAX_FILES and total_bytes <= MAX_BUNDLE_BYTES,
                                 "Public bundle bound exceeded")
            _require(_stamp(os.fstat(descriptor)) == directories[prefix],
                     "A bundle directory changed during inventory")
    _require(set(files) == expected and set(directories) == allowed_directories,
             "Public bundle membership is incomplete or changed")
    return files, directories, total_bytes


@contextmanager
def _opened(root, name, stamp, deadline):
    _check(deadline)
    path = root / _relative(name)
    _require(_stamp(path.lstat()) == stamp, "A public file identity changed")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    with _directory_descriptor(path.parent, deadline) as parent_descriptor:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        try:
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                metadata = os.fstat(stream.fileno())
                _require(_stamp(metadata) == stamp and stat.S_ISREG(metadata.st_mode)
                         and metadata.st_nlink == 1, "An opened public file identity changed")
                yield stream
                _check(deadline)
                final_metadata = os.stat(path.name, dir_fd=parent_descriptor,
                                         follow_symlinks=False)
                _require(_stamp(os.fstat(stream.fileno())) == stamp
                         and _stamp(final_metadata) == stamp,
                         "A public file changed during authentication")
        finally:
            if descriptor is not None:
                os.close(descriptor)


def _hash_file(root, name, stamp, deadline, *, retain=False):
    digest = hashlib.sha256()
    count = 0
    retained = bytearray() if retain else None
    expected_bytes = stamp[4]
    with _opened(root, name, stamp, deadline) as stream:
        while True:
            _check(deadline)
            chunk = stream.read(min(CHUNK_BYTES, expected_bytes - count + 1))
            _check(deadline)
            if not chunk:
                break
            count += len(chunk)
            _require(count <= expected_bytes, "A public file exceeded its admitted length")
            digest.update(chunk)
            if retained is not None:
                retained.extend(chunk)
    _require(count == expected_bytes, "A public file ended before its admitted length")
    return {"bytes": count, "sha256": digest.hexdigest()}, (
        bytes(retained) if retained is not None else None)


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        _require(name not in result, "Duplicate JSON object member")
        result[name] = value
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON constants are not admitted")


def _finite_float(token):
    _require(len(token) <= 128, "A saved JSON number exceeded its token bound")
    value = float(token)
    _require(math.isfinite(value), "Nonfinite saved JSON numbers are not admitted")
    return value


def _bounded_int(token):
    _require(len(token) <= 64, "A saved JSON integer exceeded its token bound")
    return int(token)


def _json(raw):
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                        parse_constant=_constant, parse_float=_finite_float,
                        parse_int=_bounded_int)
    _require(type(result) is dict, "Saved JSON must contain an ordinary object")
    return result


def _same_metadata(observed, expected):
    """Compare saved labels with exact JSON types, including nested lists."""
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return (observed.keys() == expected.keys()
                and all(_same_metadata(observed[name], value)
                        for name, value in expected.items()))
    if type(expected) is list:
        return (len(observed) == len(expected)
                and all(_same_metadata(left, right)
                        for left, right in zip(observed, expected, strict=True)))
    return observed == expected


def _subset(value, expected, message):
    _require(type(value) is dict, message)
    for name, wanted in expected.items():
        _require(name in value and _same_metadata(value[name], wanted), message)


def _manifest_contract(value):
    """Authenticate a fixed declaration; no scientific derivation is evaluated."""
    expected_keys = set(EXPECTED_MANIFEST_HEADER) | {"payloads", "payload_count", "payload_bytes"}
    _require(type(value) is dict and set(value) == expected_keys,
             "Saved manifest field closure changed")
    _subset(value, EXPECTED_MANIFEST_HEADER, "Saved manifest scope or provenance changed")
    declared = value["payloads"]
    _require(type(declared) is list and len(declared) == len(FILE_NAMES) == 21,
             "The fixed 21-payload declaration is required")
    pins = {}
    for item in declared:
        _require(type(item) is dict and set(item) == {"path", "bytes", "sha256"},
                 "A manifest payload declaration changed")
        name = _relative(item["path"])
        _require(name not in pins, "Duplicate manifest payload path")
        pin = _pin({"bytes": item["bytes"], "sha256": item["sha256"]})
        _require(name in EXPECTED_PAYLOAD_PINS
                 and _same_metadata(pin, EXPECTED_PAYLOAD_PINS[name]),
                 "An immutable public payload pin changed")
        pins[name] = pin
    _require(set(pins) == FILE_NAMES, "Manifest payload closure changed")
    _subset(value, {"payload_count": len(pins),
                    "payload_bytes": sum(pin["bytes"] for pin in pins.values())},
            "Manifest stored-byte accounting changed")
    return pins


def _saved_contracts(raw):
    """Read scope labels only; no matrix or interval arithmetic is performed."""
    summary = _json(raw["summary.json"])
    results = _json(raw["results.json"])
    derivations = _json(raw["source-derivations.json"])
    _require(set(summary) == set(SUMMARY_KEYS), "Summary field closure changed")
    _require(set(results) == set(RESULT_KEYS), "Scientific projection field closure changed")
    _subset(summary, EXPECTED_SUMMARY_FIELDS, "Saved summary scope changed")
    _subset(summary, {"public_results_sha256": EXPECTED_PAYLOAD_PINS["results.json"]["sha256"]},
            "Public and original result identities were not kept separate")
    _subset(results, EXPECTED_RESULT_FIELDS, "Saved scientific metadata changed")
    _subset(results, RAW_FINAL_FLAGS, "Historical observed/final/whole flags changed")
    _subset(results, dict.fromkeys(RAW_FALSE_FLAGS, False),
            "A historical broader scientific claim changed")
    _require(set(derivations) == set(EXPECTED_DERIVATION_HEADER) | {"derivations"},
             "Source derivation field closure changed")
    _subset(derivations, EXPECTED_DERIVATION_HEADER, "Source derivation scope changed")
    records = derivations["derivations"]
    _require(type(records) is list and len(records) == len(DERIVATION_FIELDS),
             "Source derivation record closure changed")
    seen = set()
    for item in records:
        _require(type(item) is dict, "A source derivation record changed")
        name = item.get("public_path")
        _require(type(name) is str and name in DERIVATION_FIELDS and name not in seen,
                 "Source derivation path closure changed")
        seen.add(name)
        _subset(item, DERIVATION_FIELDS[name], "Source derivation claim changed")
        _subset(item, {"public_source_bytes": EXPECTED_PAYLOAD_PINS[name]["bytes"],
                       "public_source_sha256": EXPECTED_PAYLOAD_PINS[name]["sha256"]},
                "Source derivation file identity changed")
    _require(seen == set(DERIVATION_FIELDS), "Source derivation record is incomplete")


def _seconds(value):
    _require(type(value) in (int, float) and 0 < value <= MAX_SECONDS
             and math.isfinite(value), "Saved-byte time limit must be in (0,30]")
    return float(value)


def verify_saved_bundle(bundle: Path, manifest_sha256: str, *, max_seconds=30) -> dict:
    """Authenticate the closed fixed packet without importing scientific sources."""
    seconds = _seconds(max_seconds)
    expected_manifest = _digest(manifest_sha256)
    _require(expected_manifest == EXPECTED_MANIFEST_SHA256,
             "The independently supplied manifest differs from the fixed public projection")
    started = time.monotonic()
    deadline = started + seconds
    root = _root(bundle)
    files_before, directories_before, total_bytes = _inventory(root, deadline)
    manifest_pin, manifest_raw = _hash_file(
        root, MANIFEST_NAME, files_before[MANIFEST_NAME], deadline, retain=True)
    _require(manifest_pin["sha256"] == expected_manifest,
             "The supplied public-manifest digest did not authenticate")
    _check(deadline)
    pins = _manifest_contract(_json(manifest_raw))
    _check(deadline)
    observed = {MANIFEST_NAME: manifest_pin}
    raw_json = {}
    for name in sorted(FILE_NAMES):
        _check(deadline)
        pin, value = _hash_file(root, name, files_before[name], deadline, retain=name in JSON_NAMES)
        _require(_same_metadata(pin, pins[name]), "An authenticated public payload changed")
        observed[name] = pin
        if value is not None:
            raw_json[name] = value
    _check(deadline)
    _saved_contracts(raw_json)
    _check(deadline)
    files_after, directories_after, final_bytes = _inventory(root, deadline)
    _require(files_after == files_before and directories_after == directories_before
             and final_bytes == total_bytes, "Public packet changed during authentication")
    _check(deadline)
    return {
        "status": "authenticated_saved_fixed_finite_gibbs_z_projection_bytes_without_replay",
        "manifest_sha256": expected_manifest,
        "scientific_source_commit": EXPECTED_MANIFEST_HEADER["scientific_source_commit"],
        "source_freeze_sha256": EXPECTED_MANIFEST_HEADER["source_freeze_sha256"],
        "original_saved_result_sha256": EXPECTED_MANIFEST_HEADER["original_saved_result_sha256"],
        "original_root_acceptance_sha256": EXPECTED_MANIFEST_HEADER["original_root_acceptance_sha256"],
        "fixed_conditional_Z_mean_contract_accepted": True,
        "packaging_status": EXPECTED_MANIFEST_HEADER["packaging_status"],
        "raw_flags": dict(RAW_FINAL_FLAGS),
        "raw_scope_flags": dict.fromkeys(RAW_FALSE_FLAGS, False),
        "files_authenticated": len(observed), "payloads_authenticated": len(pins),
        "source_snapshots_authenticated": len(SOURCE_NAMES),
        "bundle_bytes_authenticated": total_bytes, "files": observed,
        "source_operation_counts": dict(EXPECTED_MANIFEST_HEADER["source_operation_counts"]),
        "scientific_inventory": dict(EXPECTED_MANIFEST_HEADER["scientific_inventory"]),
        "measurement_projection": dict(EXPECTED_SUMMARY_FIELDS["measurement_projection"]),
        "saved_metadata_contracts_checked": True, "saved_json_parsed": True,
        "scientific_target_imports_performed": False, "scientific_replay_performed": False,
        "qasm_parsed": False, "matrix_arithmetic_performed": False,
        "source_derivation_equivalence_reproduced": False,
        "interval_arithmetic_recomputed": False, "sampling_performed": False,
        "portable_wrapper_executed": False, "portable_wrapper_runtime_validated": False,
        "full_R64_primitive_simulation_performed": False,
        "full_state_Gibbs_accuracy_certified": False,
        "general_sampler_or_mixing_theorem": False, "floating_backend_error_certified": False,
        "rotations_synthesized": False, "physical_resources_computed": False,
        "formal_proof_reproduced": False, "publication_performed": False,
        "whole_invocation_validated": False,
        "elapsed_seconds": time.monotonic() - started,
    }


def _output_bytes(result):
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _require(len(raw) <= MAX_OUTPUT_BYTES, "Saved-byte output exceeded its finite bound")
    return raw


def _cli_guard(seconds):
    import resource

    remaining = seconds - (time.monotonic() - _ENTRY_TIME)
    _require(remaining > 0, "CLI startup exhausted the finite deadline")
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    ceiling = math.ceil(seconds)
    hard_limit = min(ceiling, hard) if hard != resource.RLIM_INFINITY else ceiling
    soft_limit = min(hard_limit, soft) if soft != resource.RLIM_INFINITY else hard_limit
    resource.setrlimit(resource.RLIMIT_CPU, (soft_limit, hard_limit))

    def expired(_signal, _frame):
        raise TimeoutError("Saved-byte CLI exceeded its real-time limit")

    signal.signal(signal.SIGALRM, expired)
    timer, _ = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, timer) if timer > 0 else remaining)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=str,
                        default=str(Path(__file__).absolute().parents[1] / BASE))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    try:
        seconds = _seconds(args.max_seconds)
        _cli_guard(seconds)
        result = verify_saved_bundle(args.bundle, args.manifest_sha256, max_seconds=seconds)
        print(_output_bytes(result).decode("utf-8"), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, TimeoutError, RecursionError,
            RuntimeError, EOFError, UnicodeError) as error:
        print("Saved finite-Gibbs byte authentication failed: " + type(error).__name__,
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
