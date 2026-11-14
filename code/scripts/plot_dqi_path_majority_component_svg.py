#!/usr/bin/env python3
"""Draw accepted saved circuit records; never import or simulate quantum code.

This original stdlib-only renderer expands no algorithm and evaluates no state.
It folds only exact, phase-preserving fifteen-primitive CCX words already in
the hash-bound saved stream. All 34 CCX and 15 other CX remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path

SOURCE_SHA = "3cbb0866a388adf23a5b27594d90a92db99cfbab4ad901d597eef39ae9a6f562"
SOURCE_PATH = "assets/generated/optimization/dqi-path-majority/generated/component.ndjson"
QASM_SHA = "7c4359771fd9274d9217cb60ce4066384b8aa2eaeffe219927e1c3bf0bb18091"
SCHEMA = "openquantum.dqi.path_majority_circuit_svg.v1"
STAGES = (("prefix", 0, 3), ("compute_flag", 3, 258),
          ("xor_targets", 258, 267), ("uncompute_flag", 267, 522),
          ("undo_prefix", 522, 525))
COUNTS = {"h": 68, "cx": 219, "t": 136, "tdg": 102}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate record key")
        result[key] = value
    return result


def _saved_records(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32768:
        raise ValueError("Expected a bounded ordinary saved stream")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("The saved component stream does not match acceptance")
    lines = raw.splitlines()
    if len(lines) != 525:
        raise ValueError("The component must contain exactly 525 records")
    records = []
    for line in lines:
        value = json.loads(line, object_pairs_hook=_object)
        if type(value) is not dict or set(value) != {"gate", "qubits"}:
            raise ValueError("Unexpected saved record shape")
        name, wires = value["gate"], value["qubits"]
        if type(name) is not str or name not in COUNTS or type(wires) is not list:
            raise ValueError("Only the accepted H/CX/T/TDG component is drawable")
        if len(wires) != (2 if name == "cx" else 1) or any(
                type(q) is not int or not 0 <= q < 12 for q in wires):
            raise ValueError("Invalid component operands")
        if len(set(wires)) != len(wires):
            raise ValueError("Repeated quantum operand")
        records.append((name, tuple(wires)))
    if dict(Counter(name for name, _ in records)) != COUNTS:
        raise ValueError("Primitive histogram differs from the accepted component")
    return records


def _ccx(a, b, t):
    return (("h", (t,)), ("cx", (b, t)), ("tdg", (t,)), ("cx", (a, t)),
            ("t", (t,)), ("cx", (b, t)), ("tdg", (t,)), ("cx", (a, t)),
            ("t", (b,)), ("t", (t,)), ("h", (t,)), ("cx", (a, b)),
            ("t", (a,)), ("tdg", (b,)), ("cx", (a, b)))


def _fold(records):
    logical, position = [], 0
    while position < len(records):
        name, wires = records[position]
        if name == "h":
            if position + 15 > len(records):
                raise ValueError("Truncated CCX word")
            target = wires[0]
            b, a = records[position + 1][1][0], records[position + 3][1][0]
            if len({a, b, target}) != 3 or tuple(records[position:position + 15]) != _ccx(a, b, target):
                raise ValueError("A folded CCX must be the complete exact word")
            gate, operands, stop = "ccx", (a, b, target), position + 15
        elif name == "cx":
            gate, operands, stop = "cx", wires, position + 1
        else:
            raise ValueError("A primitive was left outside its exact CCX word")
        stage = next(label for label, start, end in STAGES if start <= position < end)
        if stop > next(end for label, _, end in STAGES if label == stage):
            raise ValueError("A logical word crossed a stage boundary")
        logical.append({"logical_index": len(logical) + 1, "gate": gate,
                        "qubits": list(operands), "primitive_start": position,
                        "primitive_stop": stop, "stage": stage})
        position = stop
    if Counter(row["gate"] for row in logical) != {"ccx": 34, "cx": 15}:
        raise ValueError("Expected all 34 CCX and 15 CX glyphs")
    return logical


def _text(x, y, text, size=14, color="#243247", anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>')


def render_svg(logical):
    width, height, left, right = 1180, 1284, 148, 1020
    metadata = {"schema": SCHEMA, "source_file": SOURCE_PATH, "source_sha256": SOURCE_SHA,
                "paired_source_qasm": {"path": SOURCE_PATH.replace(".ndjson", ".qasm"),
                                       "bytes": 5921, "sha256": QASM_SHA},
                "wires": 12, "logical_gates": logical, "ccx_blocks": 34,
                "cx_outside_ccx": 15, "primitive_operations": 525,
                "primitive_counts": COUNTS, "stage_ranges": {n: [a, b] for n, a, b in STAGES},
                "scope": "fixed 12-wire local decoder component; 31-wire program not depicted",
                "arbitrary_targets": True, "initial_scratch": "000", "final_scratch": "000",
                "syndrome_preserved": True, "full_program_validated": False,
                "logical_symbols_use_full_phase_exact_ccx": True}
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="figure-title figure-description" '
           f'data-schema="{SCHEMA}" data-source-sha256="{SOURCE_SHA}" data-wires="12" '
           'data-logical-gates="49" data-ccx-blocks="34" data-cx-outside-ccx="15" '
           'data-primitive-operations="525" data-full-program-validated="false">',
           '<title id="figure-title">Path-local majority decoder: every logical gate</title>',
           '<desc id="figure-description">Read panels A, B and C in order. Four syndrome '
           'wires become prefix bits, seventeen exact CCX blocks compute the majority flag, '
           'nine CX gates XOR the decoded word into arbitrary targets, and reversed CCX '
           'and prefix gates restore the three scratch wires and original syndrome. All '
           '34 CCX and 15 CX glyphs are visible. Each CCX represents the full fifteen-'
           'primitive phase-exact lowering in the linked saved source, totaling 525 '
           'primitives. The full 31-wire program is generation-only and is not shown.</desc>',
           '<metadata id="circuit-metadata">' + escape(json.dumps(metadata, sort_keys=True,
                         separators=(",", ":"))) + '</metadata>',
           '<style>text{font-family:Arial,"DejaVu Sans",sans-serif} '
           '.wire{stroke:#65758a;stroke-width:1.25}.gate-link{stroke:#26354a;stroke-width:1.8} '
           '.control{fill:#26354a}.target{fill:white;stroke:#26354a;stroke-width:1.8}</style>',
           '<rect width="1180" height="1284" fill="#fff"/>',
           _text(30, 39, "Path-local majority decoder", 25, weight="bold"),
           _text(30, 68, "12 wires · arbitrary clause targets · clean scratch at the component boundaries", 15),
           _text(30, 90, "Read A → B → C. Wire indices continue unchanged between panels.", 14, "#596879")]
    panels = ((0, 20, "A", "Prefix and majority flag"),
              (20, 29, "B", "XOR the decoded word into arbitrary targets"),
              (29, 49, "C", "Erase the flag and restore the syndrome"))
    s = ["s₁", "s₂", "s₃", "s₄"]
    r = ["r₂", "r₃", "r₄", "r₅"]
    t = ["t₁", "t₂", "t₃", "t₄", "t₅"]
    d = ["t₁ ⊕ d₁", "t₂ ⊕ d₂", "t₃ ⊕ d₃", "t₄ ⊕ d₄", "t₅ ⊕ d₅"]
    inputs = (s + t + ["0", "0", "0"], r + t + ["0", "0", "M(r)"],
              r + d + ["0", "0", "M(r)"])
    outputs = (r + t + ["0", "0", "M(r)"], r + d + ["0", "0", "M(r)"],
               s + d + ["0", "0", "0"])
    stage_titles = {"prefix": "Prefix · 3 CX", "compute_flag": "Compute flag · 17 CCX",
                    "xor_targets": "Target XOR · 9 CX", "uncompute_flag": "Uncompute · 17 CCX",
                    "undo_prefix": "Undo prefix · 3 CX"}
    stage_colors = {"prefix": "#eef2f7", "compute_flag": "#edf4fa", "xor_targets": "#eef5ed",
                    "uncompute_flag": "#faf1e8", "undo_prefix": "#eef2f7"}
    for row, (start, stop, letter, title) in enumerate(panels):
        top, spacing = 108 + row * 346, (right - left) / (stop - start + 1)
        out += [f'<g class="circuit-panel" data-panel-row="{row}" data-logical-start="{start}" '
                f'data-logical-stop="{stop}">',
                f'<rect x="20" y="{top}" width="1140" height="334" rx="10" '
                'fill="#fff" stroke="#d9e1ea"/>',
                _text(38, top + 28, f"{letter}  {title}", 17, weight="bold")]
        for stage in dict.fromkeys(g["stage"] for g in logical[start:stop]):
            positions = [j for j, g in enumerate(logical[start:stop]) if g["stage"] == stage]
            x, band_width = left + (positions[0] + .5) * spacing, len(positions) * spacing
            out.append(f'<rect x="{x:.1f}" y="{top + 39}" width="{band_width:.1f}" height="21" '
                       f'rx="4" fill="{stage_colors[stage]}"/>')
            out.append(_text(x + band_width / 2, top + 54, stage_titles[stage], 11,
                             "#405169", "middle"))
        ys = [top + 80 + q * 20 for q in range(12)]
        for q, y in enumerate(ys):
            out.append(f'<g class="circuit-wire" data-panel-row="{row}" data-qubit="{q}">'
                       f'<line class="wire" x1="{left}" y1="{y}" x2="{right}" y2="{y}"/></g>')
            out.append(_text(left - 12, y + 4, f'q{q}  {inputs[row][q]}', 12, anchor="end"))
            out.append(_text(right + 12, y + 4, outputs[row][q], 12))
        for j, gate in enumerate(logical[start:stop]):
            x, qs, idx = left + (j + 1) * spacing, gate["qubits"], gate["logical_index"]
            first, last = gate["primitive_start"], gate["primitive_stop"]
            controls, target = qs[:-1], qs[-1]
            out.append(f'<g class="logical-gate {gate["gate"]}" id="logical-gate-{idx:03d}" '
                       f'data-logical-index="{idx}" data-gate="{gate["gate"]}" '
                       f'data-qubits="{" ".join(map(str, qs))}" data-primitive-start="{first}" '
                       f'data-primitive-stop="{last}" data-stage="{gate["stage"]}">')
            out.append(f'<title>{gate["gate"].upper()} {qs}; saved primitives [{first},{last})</title>')
            out.append(f'<line class="gate-link" x1="{x:.1f}" y1="{ys[min(qs)]}" '
                       f'x2="{x:.1f}" y2="{ys[max(qs)]}"/>')
            for q in controls:
                out.append(f'<circle class="control" data-wire="{q}" cx="{x:.1f}" cy="{ys[q]}" r="3.4"/>')
            y = ys[target]
            out += [f'<circle class="target" data-wire="{target}" cx="{x:.1f}" cy="{y}" r="6.5"/>',
                    f'<path class="gate-link" d="M{x - 4:.1f},{y}H{x + 4:.1f} M{x:.1f},{y - 4}V{y + 4}"/>', '</g>',
                    _text(x, top + 321, str(idx), 9, "#778396", "middle")]
        out.append('</g>')
    out += [_text(30, 1174, "M(r) = r₂r₃r₄ ⊕ r₂r₃r₅ ⊕ r₂r₄r₅ ⊕ r₃r₄r₅ ⊕ r₂r₃r₄r₅  (threshold ≥ 3)", 14),
            _text(30, 1200, "Every CCX uses H=2, CX=6, T=4, T†=3: 34 × 15 + 15 = 525 primitives.", 14),
            _text(30, 1226, "Clean-scratch action: |s,t,000⟩ → |s,t ⊕ d(s),000⟩. No preparation, reset or readout is in this component.", 13),
            _text(30, 1252, "Full phase-exact words and source indices are retained in component.qasm and component.ndjson.", 13, "#596879"), '</svg>']
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    svg = render_svg(_fold(_saved_records(args.input)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
