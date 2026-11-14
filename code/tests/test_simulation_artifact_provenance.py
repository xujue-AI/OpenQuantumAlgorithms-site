"""Optional exact-root provenance never makes a public download require Git."""
from types import SimpleNamespace
from unittest.mock import patch

from openquantum.simulation import artifact_provenance as provenance


def test_download_without_git_marker_never_queries_an_ancestor_checkout(tmp_path):
    with patch.object(provenance.subprocess, "run", side_effect=AssertionError("Git queried")):
        assert provenance.git_revision(tmp_path) is None
        assert provenance.git_dirty(tmp_path, ["module.py"]) is None


def test_git_metadata_is_optional_when_git_is_unavailable(tmp_path):
    (tmp_path/".git").write_text("gitdir: unavailable")
    with patch.object(provenance.subprocess, "run", side_effect=OSError("Git not installed")):
        assert provenance.git_revision(tmp_path) is None
        assert provenance.git_dirty(tmp_path, ["module.py"]) is None


def test_exact_root_is_verified_and_external_git_selectors_are_removed(tmp_path, monkeypatch):
    (tmp_path/".git").mkdir()
    monkeypatch.setenv("GIT_DIR", "/unrelated/checkout")
    monkeypatch.setenv("GIT_WORK_TREE", "/unrelated/checkout")
    revision = "a"*40
    def command(arguments, **kwargs):
        assert "GIT_DIR" not in kwargs["env"]
        assert "GIT_WORK_TREE" not in kwargs["env"]
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        return SimpleNamespace(returncode=0, stdout=f"{tmp_path}\n{revision}\n")
    with patch.object(provenance.subprocess, "run", side_effect=command):
        assert provenance.git_revision(tmp_path) == revision


def test_mismatched_top_level_or_invalid_revision_returns_none(tmp_path):
    (tmp_path/".git").mkdir()
    for output in [f"{tmp_path.parent}\n{'a'*40}\n", f"{tmp_path}\nnot-a-commit\n"]:
        with patch.object(provenance.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=output)):
            assert provenance.git_revision(tmp_path) is None


def test_dirty_status_is_scoped_to_requested_owned_paths(tmp_path):
    (tmp_path/".git").mkdir()
    def command(arguments, **kwargs):
        if "rev-parse" in arguments:
            return SimpleNamespace(returncode=0, stdout=f"{tmp_path}\n{'b'*40}\n")
        assert arguments[-3:] == ["--", "module.py", "method.md"]
        return SimpleNamespace(returncode=0, stdout="?? module.py\n")
    with patch.object(provenance.subprocess, "run", side_effect=command):
        assert provenance.git_dirty(tmp_path, ["module.py", "method.md"]) is True
