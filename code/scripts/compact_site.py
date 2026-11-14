#!/usr/bin/env python3
"""Remove byte-identical MyST public aliases without changing scientific files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

PROTECTED_DIRECTORIES = frozenset({"assets", "code", "interactive", "binder"})
PROTECTED_FILES = frozenset({"LICENSE", "requirements-lock.txt"})
GENERATED_TEXT = frozenset({".html", ".js", ".css", ".json"})
READABLE_TEXT = GENERATED_TEXT | {".md", ".ipynb", ".txt", ".svg", ".py", ".toml", ".bib"}
MARKER = re.compile(r"_public(?:/|\\/|\\u002[fF]|%2[fF])")
PATH_MARKER = re.compile(r"_public(?:/|%2[fF])")
DELIMITERS = frozenset("\t\r\n \"'`<>(){}[],;=")
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def protected(relative: Path) -> bool:
    return bool(relative.parts and (
        relative.parts[0] in PROTECTED_DIRECTORIES or relative.as_posix() in PROTECTED_FILES
        # Notebook downloads stay exact; MyST's generated notebook-route
        # HTML/JSON/JS/CSS must have the same alias rewriting as other pages.
        or (relative.parts[0] == "notebooks" and relative.suffix.lower() not in GENERATED_TEXT)
    ))


def validate_root(root: Path) -> Path:
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Publication root must be a real directory")
    root = root.resolve()
    if root == SOURCE_ROOT or SOURCE_ROOT.is_relative_to(root):
        raise ValueError("Do not compact source files or a source ancestor")
    for directory in ("src", "scripts", "tests", "assets", "notebooks"):
        if root.is_relative_to(SOURCE_ROOT / directory):
            raise ValueError("Do not compact a protected source directory")
    if not (root / "index.html").is_file():
        raise ValueError("Publication root requires a rendered index.html")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in publication tree: {path.relative_to(root)}")
    return root


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def protected_manifest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and protected(path.relative_to(root))}


def url_tokens(text: str):
    previous_end = -1
    for match in MARKER.finditer(text):
        if match.start() < previous_end:
            continue
        start, end = match.start(), match.end()
        while start and text[start - 1] not in DELIMITERS:
            start -= 1
        while end < len(text) and text[end] not in DELIMITERS:
            end += 1
        previous_end = end
        # Nested hydration JSON escapes its enclosing quotes. Those trailing
        # backslashes belong to the quote, not to the referenced file path.
        while end > match.end() and text[end - 1] == "\\":
            end -= 1
        yield start, end, text[start:end]


def decoded_token(token: str) -> str:
    return re.sub(r"\\u002[fF]", "/", token.replace("\\/", "/"))


def alias_from_token(token: str) -> str | None:
    try:
        parts = urlsplit(decoded_token(token.rstrip("\\")))
    except ValueError:
        return None
    marker = PATH_MARKER.search(parts.path)
    return unquote(parts.path[marker.start():]) if marker else None


def references(text: str) -> set[str]:
    return {alias for _, _, token in url_tokens(text)
            if (alias := alias_from_token(token)) is not None}


def rewrite_urls(text: str, aliases: dict[str, str]) -> str:
    pieces, previous = [], 0
    for start, end, token in url_tokens(text):
        alias = alias_from_token(token)
        if alias not in aliases:
            continue
        normalized = decoded_token(token)
        parts = urlsplit(normalized)
        marker = PATH_MARKER.search(parts.path)
        target = quote(aliases[alias], safe="/-._~")
        if marker.group().lower().endswith("%2f"):
            target = target.replace("/", "%2F")
        path = parts.path[:marker.start()] + target
        replacement = urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
        if "\\u002f" in token.lower():
            escape = "\\u002F" if "\\u002F" in token else "\\u002f"
            replacement = replacement.replace("/", escape)
        elif "\\/" in token:
            replacement = replacement.replace("/", "\\/")
        pieces.extend((text[previous:start], replacement))
        previous = end
    pieces.append(text[previous:])
    return "".join(pieces)


def target_rank(alias: Path, candidate: str):
    original_name = re.sub(r"-[0-9a-fA-F]{24,64}(?=\.[^.]+$)", "", alias.name)
    path = Path(candidate)
    direct = alias.relative_to("_public").as_posix()
    return (candidate != direct, path.name != original_name,
            not path.stem.startswith(Path(original_name).stem), path.parts[0] == "code",
            len(path.parts), candidate)


def compact_site(root: Path) -> dict:
    root = validate_root(root)
    before_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    before = protected_manifest(root)
    index = {}
    for name, digest in before.items():
        index.setdefault((root.joinpath(name).stat().st_size, digest), []).append(name)
    aliases, sizes = {}, {}
    for path in sorted((root / "_public").rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        matches = index.get((size, sha256(path)), [])
        if matches:
            name = path.relative_to(root).as_posix()
            aliases[name] = min(matches, key=lambda candidate: target_rank(Path(name), candidate))
            sizes[name] = size
    # Protected downloads are immutable, including any URLs written into them.
    blocked = set()
    for name in before:
        path = root / name
        if path.suffix.lower() in READABLE_TEXT:
            blocked |= references(path.read_text(encoding="utf-8", errors="surrogateescape")) & aliases.keys()
    for name in blocked:
        del aliases[name]
    updates = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or protected(relative) or relative.as_posix() in aliases
                or path.suffix.lower() not in READABLE_TEXT):
            continue
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        if path.suffix.lower() in GENERATED_TEXT:
            rewritten = rewrite_urls(text, aliases)
        else:
            rewritten = text
        residual = references(rewritten) & aliases.keys()
        if residual:
            raise ValueError(f"Unrewritten public alias in {relative}: {sorted(residual)[0]}")
        if rewritten != text:
            updates.append((path, rewritten.encode("utf-8", errors="surrogateescape")))
    for path, contents in updates:
        path.write_bytes(contents)
    for name in aliases:
        (root / name).unlink()
    for directory in sorted((root / "_public").rglob("*"), key=lambda path: len(path.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if protected_manifest(root) != before:
        raise ValueError("Compaction changed a canonical or code download")
    after_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {"before_bytes": before_bytes, "after_bytes": after_bytes,
            "removed_aliases": len(aliases), "removed_alias_bytes": sum(sizes[name] for name in aliases),
            "rewritten_generated_files": len(updates), "protected_files": len(before),
            "protected_bytes_preserved": True, "retained_protected_references": len(blocked),
            "unique_public_files": sum(path.is_file() for path in (root / "_public").rglob("*")),
            "below_one_gigabyte": after_bytes < 1_000_000_000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=SOURCE_ROOT / "_build/html")
    args = parser.parse_args()
    print(json.dumps(compact_site(args.path), indent=2))


if __name__ == "__main__":
    main()
