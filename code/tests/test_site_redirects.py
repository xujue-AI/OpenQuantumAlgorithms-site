"""Public bookmark compatibility; no scientific modules or site preparation."""
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/site_redirects.py"
if not SOURCE.is_file():
    pytest.skip("Publication redirects are outside the public example package", allow_module_level=True)
SPEC = importlib.util.spec_from_file_location("public_site_redirects_test", SOURCE)
redirects = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(redirects)


class RedirectPage(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.script = ""
        self.in_script = False
        self.noscript_depth = 0
        self.refreshes = []
        self.canonical = None
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "noscript":
            self.noscript_depth += 1
        if tag == "script":
            self.in_script = True
        if tag == "meta" and values.get("http-equiv") == "refresh":
            self.refreshes.append((self.noscript_depth, values["content"]))
        if tag == "link" and values.get("rel") == "canonical":
            self.canonical = values["href"]
        if tag == "a":
            self.links.append(values.get("href"))

    def handle_endtag(self, tag):
        if tag == "noscript":
            self.noscript_depth -= 1
        if tag == "script":
            self.in_script = False

    def handle_data(self, text):
        if self.in_script:
            self.script += text


def test_existing_resource_redirect_markup_is_unchanged():
    target = "/OpenQuantumAlgorithms-site/algorithms/grover/#end-to-end-resource-analysis"
    assert redirects.REDIRECTS["resources"] == "algorithms/grover/#end-to-end-resource-analysis"
    assert redirects.render_redirect(redirects.REDIRECTS["resources"], base="/OpenQuantumAlgorithms-site") == (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>Open Quantum Algorithms</title>'
        f'<link rel="canonical" href="{target}">'
        f'<meta http-equiv="refresh" content="0;url={target}">'
        f'<script>location.replace("{target}")</script></head>'
        f'<body><p><a href="{target}">Continue to the algorithm section</a>'
        '</p></body></html>\n'
    )


def test_new_fragment_maps_have_public_targets_and_a_no_script_fallback():
    assert set(redirects.REDIRECT_FRAGMENTS) <= set(redirects.REDIRECTS)
    for old_path, fragments in redirects.REDIRECT_FRAGMENTS.items():
        assert fragments
        assert all(value.startswith("algorithms/") and "/#" in value for value in fragments.values())
        page = RedirectPage(redirects.render_redirect(
            redirects.REDIRECTS[old_path], base="/book/", fragments=fragments,
        ))
        target = "/book/" + redirects.REDIRECTS[old_path]
        assert page.canonical == target
        assert page.links == [target]
        assert page.refreshes == [(1, "0;url=" + target)]


def test_actual_redirect_script_resolves_encoded_known_and_invalid_hashes():
    node = shutil.which("node")
    if node is None:
        pytest.skip("The isolated browser-script contract requires Node.js")
    old_path = "algorithms/optimization/compiled-replay"
    page = RedirectPage(redirects.render_redirect(
        redirects.REDIRECTS[old_path], base="/book", fragments=redirects.REDIRECT_FRAGMENTS[old_path],
    ))
    expected = "/book/algorithms/optimization/qaoa/#qaoa-compiled-preserved-representation-limit-failure"
    fallback = "/book/algorithms/optimization/qaoa/#complete-clifford-t-workload"
    cases = [
        ("#preserved-representation-limit-failure", expected),
        ("#%70reserved-representation-limit-failure", expected),
        ("#references", "/book/algorithms/optimization/qaoa/#references"),
        ("#__proto__", fallback),
        ("#unknown", fallback),
        ("#%", fallback),
        ("", fallback),
    ]
    program = (
        "const vm=require('node:vm');let raw='';process.stdin.setEncoding('utf8');"
        "process.stdin.on('data',v=>raw+=v);process.stdin.on('end',()=>{"
        "const input=JSON.parse(raw);const results=input.hashes.map(hash=>{"
        "let calls=[];vm.runInNewContext(input.script,{location:{hash,replace:v=>calls.push(v)}},"
        "{timeout:100});return calls});process.stdout.write(JSON.stringify(results))});"
    )
    completed = subprocess.run(
        [node, "-e", program], input=json.dumps({"script": page.script, "hashes": [v[0] for v in cases]}),
        text=True, capture_output=True, timeout=5, check=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == [[value] for _, value in cases]
