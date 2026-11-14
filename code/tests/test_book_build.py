"""Guard the verified renderer patch and bound complete rendering operations."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tested_book_build", ROOT / "scripts/build_book.py")
book = importlib.util.module_from_spec(spec)
spec.loader.exec_module(book)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "book"
    root.mkdir()
    (root / "myst.yml").write_text(f"version: 1\nsite:\n  template: {book.TEMPLATE_URL}\n")
    original = b"async function prerender(){" + book.ORIGINAL_EXPRESSION + b"}\n"
    patched = original.replace(book.ORIGINAL_EXPRESSION, book.PATCHED_EXPRESSION)
    monkeypatch.setattr(book, "ORIGINAL_SHA256", sha(original))
    monkeypatch.setattr(book, "PATCHED_SHA256", sha(patched))
    directory = root / book.TEMPLATE_DIRECTORY
    (directory / "renderer/server").mkdir(parents=True)
    renderer = root / book.RENDERER_RELATIVE
    renderer.write_bytes(original)
    (directory / "template.yml").write_text(f"build:\n  render: {book.RENDER_COMMAND}\n")
    (directory / ".env.prod").write_text("NODE_ENV=production\n")
    return root, renderer, original, patched


def test_exact_patch_is_atomic_idempotent_and_preserves_permissions_and_client(bundle):
    root, renderer, original, patched = bundle
    renderer.chmod(0o640)
    client = root / book.TEMPLATE_DIRECTORY / "renderer/client/example.js"
    client.parent.mkdir()
    client.write_bytes(b"unchanged client")
    assert book.patch_renderer(root) is True
    assert renderer.read_bytes() == patched
    assert renderer.stat().st_mode & 0o777 == 0o640
    assert book.patch_renderer(root) is False
    assert renderer.read_bytes() == patched
    assert client.read_bytes() == b"unchanged client"
    assert not list(renderer.parent.glob(".book-renderer-*"))


def test_unknown_renderer_is_rejected_without_mutation(bundle):
    root, renderer, _, _ = bundle
    renderer.write_bytes(b"unknown upstream bundle")
    with pytest.raises(ValueError, match="checksum"):
        book.patch_renderer(root)
    assert renderer.read_bytes() == b"unknown upstream bundle"


@pytest.mark.parametrize("count", [0, 2])
def test_ambiguous_original_is_rejected_even_if_its_checksum_is_known(bundle, monkeypatch, count):
    root, renderer, _, _ = bundle
    original = b"// fixture\n" + book.ORIGINAL_EXPRESSION * count
    renderer.write_bytes(original)
    monkeypatch.setattr(book, "ORIGINAL_SHA256", sha(original))
    with pytest.raises(ValueError, match="ambiguous"):
        book.patch_renderer(root)
    assert renderer.read_bytes() == original


def test_unexpected_patch_result_is_rejected_before_write(bundle, monkeypatch):
    root, renderer, original, _ = bundle
    monkeypatch.setattr(book, "PATCHED_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="verified fix"):
        book.patch_renderer(root)
    assert renderer.read_bytes() == original


def test_wrong_theme_and_wrong_render_command_fail_before_bootstrap(bundle, monkeypatch):
    root, renderer, original, _ = bundle
    def forbidden(*args):
        raise AssertionError("Bootstrap must not run with an unguarded input")
    monkeypatch.setattr(book, "run_step", forbidden)
    (root / "myst.yml").write_text("site:\n  template: https://example.invalid/other.zip\n")
    with pytest.raises(ValueError, match="theme URL"):
        book.build_book(root)
    (root / "myst.yml").write_text(f"site:\n  template: {book.TEMPLATE_URL}\n")
    (root / book.TEMPLATE_DIRECTORY / "template.yml").write_text("build:\n  render: different-command\n")
    with pytest.raises(ValueError, match="execute the guarded renderer"):
        book.build_book(root)
    assert renderer.read_bytes() == original


@pytest.mark.parametrize("relative", [Path("myst.yml"), Path("_build"),
                                    book.TEMPLATE_DIRECTORY / "renderer/server", book.RENDERER_RELATIVE])
def test_symlink_inputs_fail_before_bootstrap(bundle, monkeypatch, relative):
    root, _, _, _ = bundle
    target = root / relative
    moved = root.parent / ("moved-" + target.name)
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    monkeypatch.setattr(book, "run_step", lambda *args: pytest.fail("Bootstrap followed a symlink"))
    with pytest.raises(ValueError, match="Symlink"):
        book.build_book(root)


def test_bootstrap_failure_preserves_status_and_original_renderer(bundle, monkeypatch):
    root, renderer, original, _ = bundle
    calls = []
    def failed(command, directory, timeout, environment):
        calls.append(command)
        assert directory == root and timeout == book.BOOTSTRAP_TIMEOUT_SECONDS
        return 17
    monkeypatch.setattr(book, "run_step", failed)
    assert book.build_book(root) == 17
    assert calls == [[sys.executable, "-m", "jupyter_book", "build", "--site", "--strict", "--ci"]]
    assert renderer.read_bytes() == original


def test_html_failure_is_preserved_after_exact_bootstrap_and_patch(bundle, monkeypatch):
    root, renderer, _, patched = bundle
    calls = []
    def step(command, directory, timeout, environment):
        calls.append(command)
        assert directory == root and environment["HOST"]
        if len(calls) == 1:
            return 0
        assert renderer.read_bytes() == patched and timeout == book.HTML_TIMEOUT_SECONDS
        return 23
    monkeypatch.setattr(book, "run_step", step)
    assert book.build_book(root) == 23
    assert calls[1] == [sys.executable, "-m", "jupyter_book", "build", "--html", "--strict", "--ci", "--keep-host"]


def test_fresh_bootstrap_must_materialize_guarded_inputs(bundle, monkeypatch):
    root, renderer, original, _ = bundle
    directory = root / book.TEMPLATE_DIRECTORY
    shutil.rmtree(root / "_build")
    calls = []
    def step(command, *_):
        calls.append(command)
        if len(calls) == 1:
            renderer.parent.mkdir(parents=True)
            renderer.write_bytes(original)
            (directory / "template.yml").write_text(f"build:\n  render: {book.RENDER_COMMAND}\n")
            (directory / ".env.prod").write_text("NODE_ENV=production\n")
        return 0
    monkeypatch.setattr(book, "run_step", step)
    assert book.build_book(root) == 0 and len(calls) == 2


def test_generated_site_reset_preserves_pinned_templates(bundle):
    root, renderer, original, _ = bundle
    site = root / "_build/site/public"
    site.mkdir(parents=True)
    (site / "stale-code.zip").write_bytes(b"old generated alias")
    book.reset_generated_site_cache(root)
    assert not (root / "_build/site").exists()
    assert renderer.read_bytes() == original


def test_generated_site_reset_rejects_redirected_cache(bundle):
    root, renderer, original, _ = bundle
    foreign = root.parent / "foreign-site"
    foreign.mkdir()
    (foreign / "keep.json").write_bytes(b"untouched")
    (root / "_build/site").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        book.reset_generated_site_cache(root)
    assert (foreign / "keep.json").read_bytes() == b"untouched"
    assert renderer.read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="Publication uses POSIX process groups")
def test_child_exit_status_and_timeout_are_bounded(tmp_path):
    environment = dict(os.environ)
    assert book.run_step([sys.executable, "-I", "-B", "-c", "raise SystemExit(29)"], tmp_path, 5, environment) == 29
    began = time.monotonic()
    assert book.run_step([sys.executable, "-I", "-B", "-c", "import time;time.sleep(30)"], tmp_path, .05, environment) == 124
    assert time.monotonic() - began < 5


@pytest.mark.skipif(os.name != "posix", reason="Publication uses POSIX process groups")
@pytest.mark.parametrize("original_status", [None, 31])
def test_wait_errors_during_cleanup_still_kill_and_preserve_original_result(tmp_path, monkeypatch, original_status):
    signals = []
    class BrokenWait:
        pid = 12345
        calls = 0
        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1 and original_status is not None:
                return original_status
            raise RuntimeError("original wait error" if self.calls == 1 else "cleanup wait error")
    monkeypatch.setattr(book.subprocess, "Popen", lambda *args, **kwargs: BrokenWait())
    monkeypatch.setattr(book.os, "killpg", lambda pid, sig: signals.append(sig))
    if original_status is None:
        with pytest.raises(RuntimeError, match="original wait error"):
            book.run_step(["fixture"], tmp_path, 1, {})
    else:
        assert book.run_step(["fixture"], tmp_path, 1, {}) == original_status
    assert signals == [book.signal.SIGTERM, book.signal.SIGKILL]


@pytest.mark.skipif(os.name != "posix", reason="Publication uses POSIX process groups")
def test_timeout_stops_an_owned_descendant_that_ignores_termination(tmp_path):
    pid_file = tmp_path / "descendant.pid"
    heartbeat = tmp_path / "heartbeat.txt"
    child = (
        "import pathlib,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"p=pathlib.Path({str(heartbeat)!r});"
        "i=0\nwhile True:\n i+=1;p.write_text(str(i));time.sleep(.03)\n"
    )
    parent = (
        "import pathlib,subprocess,sys,time;"
        f"p=subprocess.Popen([sys.executable,'-I','-B','-c',{child!r}]);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));time.sleep(30)"
    )
    try:
        assert book.run_step([sys.executable, "-I", "-B", "-c", parent], tmp_path, .75, dict(os.environ)) == 124
        assert pid_file.is_file() and heartbeat.is_file()
        after_stop = heartbeat.read_bytes()
        time.sleep(.15)
        assert heartbeat.read_bytes() == after_stop
    finally:
        if pid_file.is_file():
            try:
                os.kill(int(pid_file.read_text()), book.signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("size", [0, 1, 5, 17])
def test_synthetic_javascript_bounds_the_whole_render_read_write_operation(bundle, monkeypatch, size):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the renderer concurrency check")
    root, renderer, _, _ = bundle
    before = f"""let active=0,peak=0;const started=[],finished=[];
const A=Array.from({{length:{size}}},(_,id)=>({{id,latency:(id*7)%5}}));
const N=null,t=null,f=null;
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function renderRenderablePath(handler,item,output,base){{
  active++;peak=Math.max(peak,active);started.push(item.id);
  await delay(item.latency); // render and fetch
  await delay((item.id+1)%4); // read the complete response body
  await delay((item.id+2)%3); // finish the output write
  finished.push(item.id);active--;
}}
async function prerender(){{""".encode()
    after = b"}\nawait prerender();console.log(JSON.stringify({active,peak,started,finished}));\n"
    original = before + book.ORIGINAL_EXPRESSION + after
    patched = original.replace(book.ORIGINAL_EXPRESSION, book.PATCHED_EXPRESSION)
    monkeypatch.setattr(book, "ORIGINAL_SHA256", sha(original))
    monkeypatch.setattr(book, "PATCHED_SHA256", sha(patched))
    renderer.write_bytes(original)
    book.patch_renderer(root)
    script = root / "concurrency.mjs"
    script.write_bytes(renderer.read_bytes())
    completed = subprocess.run([node, str(script)], capture_output=True, text=True, check=True, timeout=10)
    result = json.loads(completed.stdout)
    assert result["active"] == 0 and result["peak"] == min(size, 5)
    assert result["started"] == list(range(size))
    assert sorted(result["finished"]) == list(range(size))


def test_public_package_exports_the_build_and_runtime_cache_helpers():
    spec = importlib.util.spec_from_file_location("book_export_registration", ROOT / "scripts/export_code.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    assert {"build_book.py", "public_assets.py"} <= set(export.PUBLIC_SCRIPTS)
