# SPDX-License-Identifier: MIT
"""Pure producer-boundary tests; no producer initializer or real Git calls.

Only three selected standard-library helper definitions are extracted. Their
subprocess dependency is a fake interface and all writes target pytest files.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PRODUCER = Path(__file__).resolve().parents[1]/"scripts/generate_robust_qpe_resources.py"


@pytest.fixture
def helpers(tmp_path):
    tree = ast.parse(PRODUCER.read_text(), filename=str(PRODUCER))
    names = {"source_revision", "selected_output", "inventory"}
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == 3 and {node.name for node in selected} == names
    assert all(not node.decorator_list for node in selected)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom))
                   for function in selected for node in ast.walk(function))
    source_paths = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "SOURCE_PATHS"
                                for target in node.targets))
    root = tmp_path.resolve()/"fixture-root"
    root.mkdir()
    calls, responses = [], []
    def run(argv, *, cwd, capture_output, text):
        calls.append((tuple(argv), cwd, capture_output, text))
        if not responses:
            raise AssertionError("Unexpected mocked Git call")
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response
    namespace = {
        "Path": Path, "ROOT": root, "SOURCE_PATHS": source_paths,
        "subprocess": SimpleNamespace(run=run),
        "MAX_OUTPUT_FILES": 2, "MAX_OUTPUT_BYTES": 8,
    }
    # Extracted function bodies only; top-level path injection, imports and main
    # are absent, so no quantum package or producer initializer can be entered.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PRODUCER), "exec"), namespace)
    return SimpleNamespace(root=root, calls=calls, responses=responses,
                           source_paths=source_paths, namespace=namespace,
                           **{name: namespace[name] for name in names})


def reply(code=0, stdout=""):
    return SimpleNamespace(returncode=code, stdout=stdout)


@pytest.mark.parametrize("case", ("no_checkout", "wrong_root", "head_failure", "invalid_head"))
def test_source_revision_requires_matching_git_root_and_verified_40hex_head(helpers, case):
    if case == "no_checkout":
        helpers.responses.append(reply(128))
        expected_calls = 1
    elif case == "wrong_root":
        helpers.responses.append(reply(stdout=str(helpers.root.parent/"another-root")))
        expected_calls = 1
    elif case == "invalid_head":
        for head in ("a"*39, "a"*41, "g"*40, "A"*40):
            helpers.responses.extend([reply(stdout=str(helpers.root)+"\n"),
                                      reply(stdout=head+"\n")])
            assert helpers.source_revision() == (None, None)
            assert helpers.calls[-2:] == [
                (("git", "rev-parse", "--show-toplevel"), helpers.root, True, True),
                (("git", "rev-parse", "HEAD"), helpers.root, True, True),
            ]
        assert len(helpers.calls) == 8 and helpers.responses == []
        return
    else:
        helpers.responses.extend([
            reply(stdout=str(helpers.root)+"\n"),
            reply(1),
        ])
        expected_calls = 2
    assert helpers.source_revision() == (None, None)
    assert len(helpers.calls) == expected_calls and helpers.responses == []
    assert helpers.calls[0] == (("git", "rev-parse", "--show-toplevel"), helpers.root, True, True)
    if expected_calls == 2:
        assert helpers.calls[1] == (("git", "rev-parse", "HEAD"), helpers.root, True, True)


@pytest.mark.parametrize("status_code,status_text,expected_dirty", [(0, "", False), (0, " M source.py\n", True), (1, "", None)])
def test_source_revision_records_clean_dirty_or_unavailable_status_only_after_valid_head(
        helpers, status_code, status_text, expected_dirty):
    revision = "0123456789abcdef"*2+"01234567"
    helpers.responses.extend([reply(stdout=str(helpers.root)+"\n"),
                              reply(stdout=revision+"\n"),
                              reply(status_code, status_text)])
    assert helpers.source_revision() == (revision, expected_dirty)
    assert len(helpers.calls) == 3 and helpers.responses == []
    assert helpers.calls[-1] == (("git", "status", "--porcelain", "--", *helpers.source_paths),
                                 helpers.root, True, True)


def test_source_revision_falls_back_to_byte_provenance_when_git_binary_is_absent(helpers):
    helpers.responses.append(FileNotFoundError("mocked missing git"))
    assert helpers.source_revision() == (None, None)
    assert len(helpers.calls) == 1 and helpers.responses == []


def test_selected_output_accepts_only_fresh_unredirected_nonprotected_paths(helpers):
    base = helpers.root.parent
    fresh = base/"new-output"
    assert helpers.selected_output(fresh) == fresh.resolve()
    assert not fresh.exists()
    existing = base/"existing-output"
    existing.mkdir()
    target = base/"real-directory"
    target.mkdir()
    alias = base/"redirected-parent"
    alias.symlink_to(target, target_is_directory=True)
    dangling = base/"dangling-output"
    dangling.symlink_to(base/"absent-target", target_is_directory=True)
    source = helpers.root/"src/openquantum/robust_qpe_resources.py"
    refused = [existing, alias/"new-output", dangling, helpers.root,
               helpers.root/"assets/generated/new-output", source, source.parent]
    for path in refused:
        with pytest.raises(ValueError):
            helpers.selected_output(path)
    assert not source.exists()
    assert helpers.calls == []


@pytest.mark.parametrize("violation", ("files", "bytes", "symlink"))
def test_inventory_checks_exact_boundaries_and_refuses_cap_or_redirect_violations(helpers, violation):
    output = helpers.root.parent/"output-inventory"
    output.mkdir()
    (output/"first").write_bytes(b"abcd")
    (output/"second").write_bytes(b"efgh")
    assert helpers.inventory(output) == (2, 8)
    if violation == "files":
        (output/"third").write_bytes(b"")
    elif violation == "bytes":
        (output/"first").write_bytes(b"abcde")
    else:
        (output/"redirect").symlink_to(output/"first")
    with pytest.raises(RuntimeError, match="symlink" if violation == "symlink" else "cap"):
        helpers.inventory(output)
    assert helpers.calls == []
