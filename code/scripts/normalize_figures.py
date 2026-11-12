#!/usr/bin/env python3
"""Normalize insignificant figure/CSV whitespace after canonical generation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "assets/generated").rglob("*"):
    if path.suffix == ".svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    elif path.suffix == ".csv":
        # Text-mode reading converts CRLF; cell values are left untouched.
        path.write_text(path.read_text())
print("Normalized generated SVG/CSV whitespace")
