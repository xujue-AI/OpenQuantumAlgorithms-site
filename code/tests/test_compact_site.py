"""Compact generated aliases while preserving scientific/download bytes."""
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_compaction_tests", ROOT / "scripts/compact_site.py")
compact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compact)


def put(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site"
    root.mkdir()
    put(root, "index.html", '<a href="/book/_public/result-' + 'a' * 32 + '.json?x=1#answer">Result</a>')
    put(root, "assets/result.json", b'{"verified":true}\n')
    put(root, "code/assets/result.json", b'{"verified":true}\n')
    put(root, "_public/result-" + 'a' * 32 + ".json", b'{"verified":true}\n')
    put(root, "_public/assets/result.json", b'{"verified":true}\n')
    put(root, "_public/unique.png", b'unique original figure')
    put(root, "assets/code.zip", b'complete original archive')
    put(root, "_public/code-" + 'b' * 32 + ".zip", b'complete original archive')
    return root


def test_compaction_preserves_downloads_and_rewrites_generated_files(site):
    before = compact.protected_manifest(site)
    alias = "_public/result-" + 'a' * 32 + ".json"
    for name, text in {
        "page.json": '{"url":"/book/' + alias + '"}',
        "bundle.js": 'const asset="/book/' + alias + '";',
        "style.css": 'body{background:url(../' + alias + ')}',
        "_public/unique.json": '{"source":"/book/' + alias + '"}',
    }.items():
        put(site, name, text)
    result = compact.compact_site(site)
    assert result["removed_aliases"] == 3
    assert result["after_bytes"] < result["before_bytes"]
    assert compact.protected_manifest(site) == before
    assert (site / "assets/code.zip").read_bytes() == b'complete original archive'
    assert (site / "_public/unique.png").read_bytes() == b'unique original figure'
    assert '/book/assets/result.json?x=1#answer' in (site / "index.html").read_text()
    for path in (site / "page.json", site / "bundle.js", site / "style.css", site / "_public/unique.json"):
        assert alias not in path.read_text()
    assert compact.compact_site(site)["removed_aliases"] == 0


@pytest.mark.parametrize("url,expected", [
    ('/book/_public/a.json?x=1#f', '/book/assets/a.json?x=1#f'),
    ('../../_public/a.json', '../../assets/a.json'),
    ('https://example.org/book/_public/a.json', 'https://example.org/book/assets/a.json'),
    (r'\/book\/_public\/a.json', r'\/book\/assets\/a.json'),
    (r'\u002fbook\u002f_public\u002fa.json', r'\u002fbook\u002fassets\u002fa.json'),
    ('%2Fbook%2F_public%2Fa.json', '%2Fbook%2Fassets%2Fa.json'),
])
def test_url_rewriting_handles_base_relative_escaped_and_encoded_paths(url, expected):
    text = '"' + url + '"'
    assert compact.rewrite_urls(text, {'_public/a.json': 'assets/a.json'}) == '"' + expected + '"'


def test_unrelated_paths_and_same_prefix_filenames_are_not_rewritten():
    text = '"/_public/a.json.old" "/_public/b.json"'
    assert compact.rewrite_urls(text, {'_public/a.json': 'assets/a.json'}) == text


@pytest.mark.parametrize("escape_count", [1, 2, 3])
def test_nested_hydration_quote_escapes_stay_outside_the_asset_url(escape_count):
    escape = "\\" * escape_count
    alias = '_public/logical-scaling-' + 'a' * 32 + '.svg'
    text = f'{escape}"src{escape}":{escape}"/book/{alias}{escape}"'
    expected = text.replace('/book/' + alias, '/book/assets/logical-scaling.svg')
    assert compact.references(text) == {alias}
    assert compact.rewrite_urls(text, {alias: 'assets/logical-scaling.svg'}) == expected
    assert not compact.references(expected)


def test_compaction_rewrites_visible_and_escaped_hydration_links(site):
    alias = '_public/result-' + 'a' * 32 + '.json'
    hydration = r'window.__data=JSON.parse("{\"url\":\"/book/ALIAS\"}");'.replace('ALIAS', alias)
    put(site, 'index.html', f'<a href="/book/{alias}">Data</a><script>{hydration}</script>')
    compact.compact_site(site)
    text = (site / 'index.html').read_text()
    assert alias not in text
    assert text.count('/book/assets/result.json') == 2
    assert not (site / alias).exists()


def test_notebook_downloads_are_preserved_but_generated_routes_are_rewritten(site):
    alias = '_public/result-' + 'a' * 32 + '.json'
    notebook = put(site, 'notebooks/example.ipynb', b'{"original_notebook":true}\n')
    route = put(site, 'notebooks/example/index.html', '<a href="/book/' + alias + '">Data</a>')
    before = notebook.read_bytes()
    assert compact.protected(notebook.relative_to(site))
    assert not compact.protected(route.relative_to(site))
    compact.compact_site(site)
    assert notebook.read_bytes() == before
    assert '/book/assets/result.json' in route.read_text()


def test_protected_references_keep_their_original_alias(site):
    alias = '_public/result-' + 'a' * 32 + '.json'
    put(site, 'code/fixture.json', '{"url":"/book/' + alias + '"}')
    before = compact.protected_manifest(site)
    result = compact.compact_site(site)
    assert result['retained_protected_references'] == 1
    assert (site / alias).is_file()
    assert compact.protected_manifest(site) == before


def test_same_original_basename_is_preferred_over_other_equal_bytes(site):
    put(site, 'assets/000-other.json', b'{"verified":true}\n')
    compact.compact_site(site)
    assert '/book/assets/result.json' in (site / 'index.html').read_text()


def test_symlink_and_source_root_guards_preserve_files(site):
    before = compact.protected_manifest(site)
    link = site / '_public/link.json'
    link.symlink_to(site / 'assets/result.json')
    with pytest.raises(ValueError, match='Symlink'):
        compact.compact_site(site)
    assert compact.protected_manifest(site) == before
    with pytest.raises(ValueError, match='source'):
        compact.compact_site(ROOT)


def test_root_symlink_and_missing_index_are_rejected(site, tmp_path):
    link = tmp_path / 'site-link'
    link.symlink_to(site, target_is_directory=True)
    with pytest.raises(ValueError, match='real directory'):
        compact.compact_site(link)
    (site / 'index.html').unlink()
    with pytest.raises(ValueError, match='index.html'):
        compact.compact_site(site)
