"""A partial source preview must preserve the complete archive's identities."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import warnings
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import public_code_preview as preview
from verify_public_package import verify_manifest


def put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


def archive_package(package, archive, *, extra=(), transform=None):
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for path in sorted(package.rglob("*")):
            if not path.is_file():
                continue
            name = "openquantum-examples/" + path.relative_to(package).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            if transform:
                info, data = transform(info, data)
            target.writestr(info, data)
        for info, data in extra:
            target.writestr(info, data)


@pytest.fixture
def site_package(tmp_path):
    site, package = tmp_path / "site", tmp_path / "package"
    for name, data in {
        "src/example.py": "VALUE = 7\n",
        "tests/test_example.py": "def test_value():\n    assert 7 == 7\n",
        "notebooks/example.ipynb": '{"cells":[]}\n',
        "requirements-lock.txt": "a-locked-package==1.0\n",
        "README.md": "The complete example package\n",
        "assets/generated/example/results.json": '{"checked":true,"shots":256}\n',
        "assets/generated/example/source.py": "SAVED_VALUE = 7\n",
        "assets/generated/example/requirements-lock.txt": "a-locked-package==1.0\n",
    }.items():
        put(package, name, data)
    hashes = {path.relative_to(package).as_posix(): preview.digest(path)
              for path in package.rglob("*") if path.is_file()}
    record = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
              "file_sha256": hashes,
              "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    put(package, "PROVENANCE.json", json.dumps(record, indent=2) + "\n")
    shutil.copytree(package / "assets", site / "assets")
    archive_package(package, site / preview.ARCHIVE_PATH)
    return site, package


def test_preview_preserves_original_sources_assets_archive_and_full_provenance(site_package):
    site, package = site_package
    before = {p.relative_to(site).as_posix(): p.read_bytes() for p in site.rglob("*") if p.is_file()}
    put(site, "code/assets/stale.json", "old duplicate")
    put(site, "code/stale.py", "old source")
    record = preview.write_preview(package, site)
    assert preview.verify_preview(site) == record
    assert record["scope"] == preview.SCOPE
    assert not (site / "code/assets").exists()
    assert not (site / "code/stale.py").exists()
    assert record["retained_sha256"]["PROVENANCE.json"] == preview.digest(package / "PROVENANCE.json")
    assert (site / "code/PROVENANCE.json").read_bytes() == (package / "PROVENANCE.json").read_bytes()
    assert record["archive"] == {"path": preview.ARCHIVE_PATH, "sha256": preview.digest(site / preview.ARCHIVE_PATH)}
    for name, entry in record["omitted_assets"].items():
        assert entry == {"site_path": name, "sha256": preview.digest(package / name)}
        assert (site / name).read_bytes() == (package / name).read_bytes()
    assert all((site / name).read_bytes() == data for name, data in before.items())
    assert verify_manifest(package)["file_sha256"]
    with pytest.raises(ValueError, match="contents"):
        verify_manifest(site / "code")


@pytest.mark.parametrize("name", ["src/example.py", "PROVENANCE.json", "requirements-lock.txt"])
@pytest.mark.parametrize("change", ["edit", "missing", "symlink"])
def test_changed_retained_source_or_provenance_is_rejected(site_package, name, change, tmp_path):
    site, package = site_package
    preview.write_preview(package, site)
    path = site / "code" / name
    if change == "edit":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        original = path.read_bytes()
        path.unlink()
        if change == "symlink":
            target = put(tmp_path, "external-file", original)
            path.symlink_to(target)
    with pytest.raises(ValueError, match="preview|package"):
        preview.verify_preview(site)


@pytest.mark.parametrize("change", ["edit", "missing", "file-link", "directory-link"])
def test_changed_or_linked_canonical_assets_are_rejected(site_package, change, tmp_path):
    site, package = site_package
    preview.write_preview(package, site)
    path = site / "assets/generated/example/results.json"
    if change == "edit":
        path.write_text("changed measured result")
    elif change == "missing":
        path.unlink()
    elif change == "file-link":
        data = path.read_bytes()
        path.unlink()
        path.symlink_to(put(tmp_path, "external-result", data))
    else:
        folder = path.parent
        original = tmp_path / "relocated-assets"
        folder.rename(original)
        folder.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="asset|package|preview"):
        preview.verify_preview(site)


@pytest.mark.parametrize("key,value", [
    ("schema", "complete-package"), ("scope", "Complete runnable directory"),
    ("source_commit", "b" * 40), ("retained_sha256", {}), ("omitted_assets", {}),
    ("archive", {"path": "../unrelated.zip", "sha256": "0" * 64}),
])
def test_stale_or_misleading_preview_manifest_is_rejected(site_package, key, value):
    site, package = site_package
    record = preview.write_preview(package, site)
    record[key] = value
    (site / "code/PREVIEW.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest differs"):
        preview.verify_preview(site)


def test_omitted_asset_map_cannot_redirect_to_another_equal_file(site_package):
    site, package = site_package
    record = preview.write_preview(package, site)
    name = next(iter(record["omitted_assets"]))
    entry = record["omitted_assets"][name]
    put(site, "assets/other.json", (site / name).read_bytes())
    entry["site_path"] = "assets/other.json"
    (site / "code/PREVIEW.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest differs"):
        preview.verify_preview(site)


@pytest.mark.parametrize("name", ["extra.py", "assets/generated/example/results.json"])
def test_unrecorded_preview_file_or_reintroduced_duplicate_is_rejected(site_package, name):
    site, package = site_package
    preview.write_preview(package, site)
    put(site / "code", name, "unexpected")
    with pytest.raises(ValueError, match="contents differ"):
        preview.verify_preview(site)


@pytest.mark.parametrize("extra_name", [
    "openquantum-examples/extra.py", "openquantum-examples/src/./example.py",
    "openquantum-examples/src/example.py", "wrong-root/example.py",
])
def test_extra_collapsed_duplicate_or_wrong_root_archive_member_is_rejected(site_package, extra_name):
    site, package = site_package
    put(site, "code/old.py", "keep prior preview on failure")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        archive_package(package, site / preview.ARCHIVE_PATH, extra=[(extra_name, b"VALUE = 7\n")])
    with pytest.raises(ValueError, match="archive members"):
        preview.write_preview(package, site)
    assert (site / "code/old.py").read_text() == "keep prior preview on failure"
    assert not (site / "code/PREVIEW.json").exists()


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO])
def test_archive_nonregular_member_is_rejected_without_touching_prior_output(site_package, mode):
    site, package = site_package
    put(site, "code/old.py", "preserve")
    def change(info, data):
        if info.filename.endswith("src/example.py"):
            info.external_attr = (mode | 0o644) << 16
        return info, data
    archive_package(package, site / preview.ARCHIVE_PATH, transform=change)
    with pytest.raises(ValueError, match="regular file"):
        preview.write_preview(package, site)
    assert (site / "code/old.py").read_text() == "preserve"


def test_archive_changed_member_cannot_be_bound_to_an_unrelated_extraction(site_package):
    site, package = site_package
    put(site, "code/old.py", "preserve")
    def change(info, data):
        return info, b"VALUE = 9\n" if info.filename.endswith("src/example.py") else data
    archive_package(package, site / preview.ARCHIVE_PATH, transform=change)
    with pytest.raises(ValueError, match="archive member differs"):
        preview.write_preview(package, site)
    assert (site / "code/old.py").read_text() == "preserve"


def test_failed_atomic_replacement_rolls_back_previous_code(site_package, monkeypatch):
    site, package = site_package
    put(site, "code/old.py", "preserve")
    original_rename = Path.rename
    def failed_new_rename(path, target):
        if path.name == "code" and path.parent.name.startswith(".openquantum-preview-"):
            raise OSError("simulated final rename failure")
        return original_rename(path, target)
    monkeypatch.setattr(Path, "rename", failed_new_rename)
    with pytest.raises(OSError, match="simulated"):
        preview.write_preview(package, site)
    assert (site / "code/old.py").read_text() == "preserve"
    assert not list(site.glob(".openquantum-preview-*"))


def test_failed_replacement_and_rollback_preserve_recoverable_previous_code_outside_site(site_package, monkeypatch):
    site, package = site_package
    put(site, "code/old.py", "preserve")
    original_rename = Path.rename
    def failed_new_and_previous_rename(path, target):
        if path.parent.name.startswith(".openquantum-preview-"):
            raise OSError("simulated replacement/rollback failure")
        return original_rename(path, target)
    monkeypatch.setattr(Path, "rename", failed_new_and_previous_rename)
    with pytest.raises(RuntimeError, match="previous code preserved at"):
        preview.write_preview(package, site)
    recovery = list(site.parent.glob(".openquantum-preview-*/previous"))
    assert len(recovery) == 1
    assert (recovery[0] / "old.py").read_text() == "preserve"
    assert not (site / "code").exists()
    assert not list(site.glob(".openquantum-preview-*"))


def test_code_directory_symlink_cannot_modify_external_files(site_package, tmp_path):
    site, package = site_package
    external = tmp_path / "external-code"
    put(external, "old.py", "preserve")
    (site / "code").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        preview.write_preview(package, site)
    assert (external / "old.py").read_text() == "preserve"


def test_site_directory_symlink_cannot_modify_external_files(site_package, tmp_path):
    site, package = site_package
    put(site, "code/old.py", "preserve")
    linked_site = tmp_path / "linked-site"
    linked_site.symlink_to(site, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        preview.write_preview(package, linked_site)
    with pytest.raises(ValueError, match="regular directory"):
        preview.verify_preview(linked_site)
    assert (site / "code/old.py").read_text() == "preserve"


def test_cli_checks_complete_archive_and_maps_without_installed_algorithm_package(site_package):
    site, package = site_package
    preview.write_preview(package, site)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/public_code_preview.py"), str(site)],
                            capture_output=True, text=True, cwd=site, check=True)
    assert "6 retained files, 3 canonical assets, complete archive manifest" in result.stdout


def test_export_and_site_contract_keep_complete_archive_and_retained_preview_sources():
    spec = importlib.util.spec_from_file_location("preview_export_tests", ROOT / "scripts/export_code.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    assert "public_code_preview.py" in export.PUBLIC_SCRIPTS
    if (ROOT / "scripts/check_site.py").is_file():
        spec = importlib.util.spec_from_file_location("preview_site_tests", ROOT / "scripts/check_site.py")
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        required = set(checker.required_artifacts())
        assert {"code/PREVIEW.json", "code/PROVENANCE.json", preview.ARCHIVE_PATH} <= required
        assert not any(name.startswith("code/assets/") for name in required)
