# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-modality scoring helpers for OmniDocBench: TEDS (tables) + formula/text edit distance.

OmniDocBench's canonical end2end score is per-modality, not one global edit distance. This
module provides:
  * ``table_teds``   - Tree-Edit-Distance-based Similarity over table HTML (the standard
    PubTabNet TEDS, via ``apted`` + ``lxml`` + ``Distance``);
  * ``formula_edit_distance`` / text edit distance - normalized Levenshtein (``rapidfuzz``);
  * extractors that split a model's markdown into table / formula / text streams.

All heavy deps are optional: ``composite_available()`` reports whether TEDS can run, so the
suite degrades to the global text edit distance when they are absent (install with
``pip install -e ".[omnidocbench]"`` from the gbench checkout).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

try:
    from apted import APTED, Config
    from apted.helpers import Tree
    import lxml.html as _lxml_html
    import distance as _distance
    _TEDS_OK = True
except Exception:                                    # optional extra not installed
    _TEDS_OK = False
    Config = object            # type: ignore
    Tree = object              # type: ignore

try:
    from rapidfuzz.distance import Levenshtein as _RF_Lev
    _RF_OK = True
except Exception:
    _RF_OK = False


def composite_available() -> bool:
    """Whether the TEDS backend (apted/lxml/Distance) is installed."""
    return _TEDS_OK


def _norm_edit(a: str, b: str) -> float:
    """Normalized Levenshtein distance in [0, 1]; 0 == identical."""
    a, b = a or "", b or ""
    if not a and not b:
        return 0.0
    if _RF_OK:
        return float(_RF_Lev.normalized_distance(a, b))
    import difflib
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# --------------------------------------------------------------------------- #
# TEDS (Tree-Edit-Distance-based Similarity) - standard PubTabNet implementation.
# --------------------------------------------------------------------------- #
if _TEDS_OK:
    class _TableTree(Tree):
        def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
            self.tag = tag
            self.colspan = colspan
            self.rowspan = rowspan
            self.content = content
            self.children = list(children)

        def bracket(self) -> str:
            if self.tag == "td":
                body = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                    self.tag, self.colspan or 1, self.rowspan or 1, self.content)
            else:
                body = '"tag": %s' % self.tag
            for child in self.children:
                body += child.bracket()
            return "{%s}" % body

    class _TedsConfig(Config):
        @staticmethod
        def maximum(*sequences):
            return max(map(len, sequences))

        def normalized_distance(self, *sequences):
            m = self.maximum(*sequences)
            return float(_distance.levenshtein(*sequences)) / m if m else 0.0

        def rename(self, node1, node2):
            if (node1.tag != node2.tag or node1.colspan != node2.colspan
                    or node1.rowspan != node2.rowspan):
                return 1.0
            if node1.tag == "td" and (node1.content or node2.content):
                return self.normalized_distance(node1.content or "", node2.content or "")
            return 0.0

    _STRUCT_TAGS = {"table", "thead", "tbody", "tr", "td", "th"}

    def _load_tree(node) -> "_TableTree":
        tag = "td" if node.tag == "th" else node.tag
        if tag == "td":
            content = " ".join(node.itertext()).strip()
            tree = _TableTree("td", int(node.get("colspan", 1) or 1),
                              int(node.get("rowspan", 1) or 1), content)
        else:
            tree = _TableTree(tag)
        for child in node.iterchildren():
            ctag = "td" if child.tag == "th" else child.tag
            if ctag in _STRUCT_TAGS:
                tree.children.append(_load_tree(child))
        return tree

    def _count(tree) -> int:
        return 1 + sum(_count(c) for c in tree.children)

    def _find_table(html: str):
        el = _lxml_html.fromstring(html)
        return el if el.tag == "table" else el.find(".//table")


def table_teds(pred_html: str, gold_html: str) -> Optional[float]:
    """TEDS similarity in [0, 1] between two table HTML strings (1 == identical structure+text).

    Returns None if the backend is unavailable or either side has no parseable <table>.
    """
    if not _TEDS_OK:
        return None
    try:
        g_tab = _find_table(gold_html or "")
    except Exception:
        g_tab = None
    if g_tab is None:
        return None                                  # no gold table -> not scorable
    try:
        p_tab = _find_table(pred_html or "")
    except Exception:
        p_tab = None
    if p_tab is None:
        return 0.0                                    # gold has a table, prediction has none
    try:
        tp, tg = _load_tree(p_tab), _load_tree(g_tab)
        n = max(_count(tp), _count(tg))
        if n == 0:
            return 1.0
        dist = APTED(tp, tg, _TedsConfig()).compute_edit_distance()
        return max(0.0, 1.0 - float(dist) / n)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Formula + text edit distance
# --------------------------------------------------------------------------- #
def _norm_latex(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"^\$+|\$+$", "", s)                  # strip $ / $$ delimiters
    s = re.sub(r"\\[\[\]()]", "", s)                 # \[ \] \( \)
    s = re.sub(r"\s+", "", s)                        # whitespace is not significant in LaTeX
    return s


def formula_edit_distance(pred_latex: str, gold_latex: str) -> float:
    """Normalized edit distance in [0, 1] between two LaTeX formulas (0 == identical)."""
    return _norm_edit(_norm_latex(pred_latex), _norm_latex(gold_latex))


# --------------------------------------------------------------------------- #
# Prediction-side block extraction (split a model's markdown into modality streams)
# --------------------------------------------------------------------------- #
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_FORMULA_RES = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL),
]


def _markdown_tables_to_html(text: str) -> Tuple[List[str], str]:
    """Find markdown pipe-tables, convert each to <table> HTML, and strip them from the text."""
    lines = text.split("\n")
    tables, out_lines, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        is_row = line.strip().startswith("|") and line.count("|") >= 2
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        is_sep = bool(re.match(r"^\s*\|?\s*:?-{2,}", nxt)) and "|" in nxt
        if is_row and is_sep:
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [r for r in block if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
            html = "<table>"
            for r in rows:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
            html += "</table>"
            tables.append(html)
        else:
            out_lines.append(line)
            i += 1
    return tables, "\n".join(out_lines)


def extract_pred_streams(text: str) -> Tuple[List[str], List[str], str]:
    """Split a model response into (tables_html, formulas_latex, text_remainder)."""
    text = text or ""
    tables = _HTML_TABLE_RE.findall(text)
    text = _HTML_TABLE_RE.sub(" ", text)
    md_tables, text = _markdown_tables_to_html(text)
    tables.extend(md_tables)
    formulas: List[str] = []
    for rx in _FORMULA_RES:
        for m in rx.findall(text):
            if m.strip():
                formulas.append(m)
        text = rx.sub(" ", text)
    return tables, formulas, text


def paired_mean(preds: List[str], golds: List[str], scorer, unmatched: float) -> Optional[float]:
    """Order-based pairing: mean scorer(pred_i, gold_i); unmatched golds score `unmatched`.

    (An approximation of OmniDocBench's content-similarity block matching; tables/formulas
    almost always appear in reading order.)
    """
    if not golds:
        return None
    total = 0.0
    for i, g in enumerate(golds):
        val = scorer(preds[i], g) if i < len(preds) else unmatched
        total += unmatched if val is None else val
    return total / len(golds)
