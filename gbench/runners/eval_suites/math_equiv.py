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

"""Symbolic/numeric answer equivalence for competition-math suites.

Suites like hmmt graded answers with regex string-normalization, so ``1/2`` != ``0.5`` and
``\\frac{\\sqrt2}{2}`` != ``\\frac{1}{\\sqrt2}`` were scored wrong even when correct. The
canonical graders (HMMT/AIME leaderboards, OpenR1/lighteval) use symbolic equivalence.

``math_equivalent`` layers three checks, most authoritative first:
  1. `math_verify` (HuggingFace) parse+verify - extracts the answer from the full model
     response (it looks for ``\\boxed{}``/``$...$``) and checks LaTeX-aware equivalence;
  2. a `sympy` fallback (via ``latex2sympy2_extended`` when present, else ``sympify``) that
     catches equivalences math_verify's verify is conservative about (radical fractions,
     symbolic-vs-numeric like ``2\\pi`` == ``6.2831853``);
  3. normalized-string equality.

``math_verify``/``latex2sympy2_extended`` are optional extras. ``math_equivalent`` itself
degrades to sympy and then to string comparison when they are absent, but a math SUITE
should call :func:`require_backend_or_skip` and SKIP rather than publish a string-match
number as if it were symbolic equivalence (see ``available``).
"""

from __future__ import annotations

import re
from typing import Any, Optional

try:                                   # HuggingFace math_verify (preferred)
    from math_verify import parse as _mv_parse, verify as _mv_verify
except Exception:                      # optional extra not installed
    _mv_parse = _mv_verify = None

try:
    import sympy as _sp
except Exception:
    _sp = None

try:                                   # robust LaTeX -> sympy (installed with math_verify)
    from latex2sympy2_extended import latex2sympy as _latex2sympy
except Exception:
    _latex2sympy = None


def _box(s: str) -> str:
    s = str(s).strip()
    return s if ("\\boxed" in s or s.startswith("$")) else "\\boxed{" + s + "}"


def last_boxed(text: str) -> Optional[str]:
    """Contents of the LAST balanced ``\\boxed{...}`` in *text*, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


_NORM_STRIP = (
    (r"\\boxed\s*", ""), (r"\\left", ""), (r"\\right", ""), (r"\\!", ""),
    (r"\\,", ""), (r"\\;", ""), (r"\\ ", " "), (r"\$", ""),
)


def _normalize(s: str) -> str:
    s = str(s).strip()
    box = last_boxed(s)
    if box is not None:
        s = box
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    for pat, repl in _NORM_STRIP:
        s = re.sub(pat, repl, s)
    s = s.strip().strip("{}").strip()
    s = s.rstrip(".").replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    # 1,000 -> 1000 (thousands separators only, not decimals)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
    return s.casefold()


def _to_sympy(expr: str) -> Any:
    if _sp is None:
        return None
    e = last_boxed(expr)
    e = (e if e is not None else expr).strip()
    e = re.sub(r"\\text\{[^{}]*\}", "", e)
    if _latex2sympy is not None:
        try:
            return _latex2sympy(e)
        except Exception:
            pass
    try:
        return _sp.sympify(e.replace("^", "**"))
    except Exception:
        return None


def _sym_equal(a: Any, b: Any) -> bool:
    if _sp is None or a is None or b is None:
        return False
    try:
        if _sp.simplify(a - b) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(float(_sp.N(a)) - float(_sp.N(b))) < 1e-6
    except Exception:
        return False


def _mv_equal(gold: str, pred: str) -> bool:
    if _mv_parse is None or _mv_verify is None:
        return False
    try:
        gp = _mv_parse(_box(gold))
        # pred is often the full response; parse() extracts its boxed/`$...$` answer. Only
        # box it when it carries no answer marker at all (a bare expression).
        pred_in = pred if ("\\boxed" in pred or "$" in pred) else _box(pred)
        pp = _mv_parse(pred_in)
        return bool(gp) and bool(pp) and bool(_mv_verify(gp, pp))
    except Exception:
        return False


def math_equivalent(pred: Any, gold: Any) -> bool:
    """True iff the model's answer *pred* is mathematically equivalent to *gold*.

    *pred* may be the full model response (the answer is extracted); *gold* is the reference
    answer (bare value or LaTeX).
    """
    if pred is None or gold is None:
        return False
    p, g = str(pred).strip(), str(gold).strip()
    if not p or not g:
        return False
    if _normalize(p) == _normalize(g):
        return True
    if _mv_equal(g, p):
        return True
    return _sym_equal(_to_sympy(g), _to_sympy(p))


def available() -> bool:
    """Whether the precise math backend (`math_verify`) is installed.

    sympy alone (a torch dependency, always present) cannot robustly parse competition
    LaTeX, so it is a fallback INSIDE math_equivalent but NOT enough to grade a math suite
    canonically - a suite should skip rather than report a string-match number as if it were
    symbolic equivalence.
    """
    return _mv_parse is not None


def require_backend(eval_name: str) -> None:
    """HARD-ERROR (infra_required) if the precise math backend (`math_verify`) is missing.

    No-skip policy: math suites FAIL HARD instead of silently downgrading to string matching
    when `math_verify` is not installed. sympy alone (always present via torch) cannot robustly
    parse competition LaTeX. Install with ``pip install -e ".[evals]"`` from the gbench checkout
    (pulls math_verify + latex2sympy2_extended + antlr4). Returns None when the backend IS
    available.
    """
    if available():
        return None
    from .swebench_common import infra_required
    raise infra_required(
        eval_name,
        "requires the 'math_verify' package for canonical symbolic answer equivalence "
        "(pip install -e \".[evals]\" from the gbench checkout); refusing to silently downgrade "
        "to string matching",
        f"docs/evals/{eval_name}.md")
