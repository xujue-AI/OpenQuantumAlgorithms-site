# SPDX-License-Identifier: MIT
"""Read-only restoration of historical non-execution schema values.

Public JSON contains scientific data only. These fixed false/unapproved
historical fields permit exact original source/plan hash checks in memory.
They are not an approval, current execution state, scheduling instruction,
or permission to call Native APIs. This module writes no files.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

_FIXED = {'source_recipe': {'status': 'frozen_source_candidate_physical_execution_unapproved',
                   'candidate_approval_received': False,
                   'physical_execution_enabled': False,
                   'seventh_clearance_received': True,
                   'seventh_cli_receipt_boundary': 'original --verify-only '
                                                   're-exports derived '
                                                   'artifacts; never invoked '
                                                   'here; seventh canonical '
                                                   'bundle is read-only'},
 'source_accounting': {'status': 'source_ledger_only_not_native_physically_executed',
                       'physical_execution_enabled': False,
                       'physical_execution_requires': 'coordinator '
                                                      'seventh-review '
                                                      'clearance and specific '
                                                      'frozen-candidate '
                                                      'approval'},
 'source_validation': {'status': 'validated_source_candidate_physical_unexecuted',
                       'physical_approval_requirement': 'specific coordinator '
                                                        'candidate approval; '
                                                        'seventh review '
                                                        'clearance alone is '
                                                        'insufficient'},
 'native_plan': {'status': 'planned_fixed_native_requests_not_executed',
                 'execution_enabled': False,
                 'approval_required': 'specific coordinator approval of this '
                                      'reviewed fixed adapter/plan'},
 'source_artifact_manifest': {'status': 'validated_source_candidate_physical_unexecuted'}}
_FILES = {'source_recipe': 'source-recipe.json',
 'source_accounting': 'source-accounting.json',
 'source_validation': 'source-validation.json',
 'native_plan': 'native-plan.json',
 'source_artifact_manifest': 'source-artifact-manifest.json'}

def _bytes(value, encoding):
    if encoding == "sorted-indented-json-newline":
        return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if encoding == "sorted-compact-json-newline":
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    raise ValueError("Unknown historical record encoding")

def restore_record(wrapper):
    kind = wrapper.get("record_kind")
    if (wrapper.get("schema") != "openquantum-word-scientific-record-v1" or kind not in _FIXED
            or wrapper.get("removed_fields") != list(_FIXED[kind])):
        raise ValueError("Historical scientific record schema changed")
    record = deepcopy(wrapper["record"])
    if set(record) & set(_FIXED[kind]):
        raise ValueError("Removed execution-policy fields were reintroduced")
    if kind == "native_plan":
        for name, child_kind in (("frozen_source_recipe", "source_recipe"),
                                 ("frozen_source_accounting", "source_accounting")):
            nested = record[name]
            if set(nested) & set(_FIXED[child_kind]):
                raise ValueError("Nested execution-policy fields were reintroduced")
            nested.update(deepcopy(_FIXED[child_kind]))
    record.update(deepcopy(_FIXED[kind]))
    if hashlib.sha256(_bytes(record, wrapper["original_encoding"])).hexdigest() != wrapper["original_sha256"]:
        raise ValueError("Public scientific fields do not reconstruct the original record hash")
    return record

def restore_recorded_plan(bundle):
    return restore_record(json.loads((Path(bundle) / _FILES["native_plan"]).read_bytes()))

def restore_source_recipe(bundle):
    return restore_record(json.loads((Path(bundle) / _FILES["source_recipe"]).read_bytes()))

def restore_source_accounting(bundle):
    return restore_record(json.loads((Path(bundle) / _FILES["source_accounting"]).read_bytes()))

def restore_source_validation(bundle):
    return restore_record(json.loads((Path(bundle) / _FILES["source_validation"]).read_bytes()))
