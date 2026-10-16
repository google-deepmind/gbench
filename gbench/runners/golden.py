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

"""Golden Set validation runner for gbench.

The Golden Set is a deterministic smoke test over basic capabilities.
Every case asserts one thing with a check that cannot be satisfied by
accident, and every case produces exactly one of three outcomes:

``passed``  the model answered correctly
``failed``  the model answered incorrectly
``error``   the harness never got an answer

Model failures and harness failures are deliberately never mixed. A
down endpoint must not look like a total regression.
"""

import ast
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.config import BenchmarkConfig, normalize_endpoint
from ..core.models import ModelConfig, ModelFormat

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30
MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0

# Bump when a change in this file could make the same answer score
# differently: the comparison rules, the tool-call matching, the
# normalisation applied before comparing, or what counts as a harness
# error rather than a failure. Do not bump for a docstring, a log line
# or a refactor that preserves verdicts.
#
# Declared rather than derived on purpose. The obvious alternative, a
# content hash of this module, moves the scaffold id on every unrelated
# edit and so breaks every historical series for changes that cannot
# affect a score. A version someone forgets to bump under-reports one
# change. A content hash over-reports every change, which trains people
# to ignore the id. `CONTRACT_VERSION` in core/scaffold.py is declared
# for the same reason. GOLDEN_SCORING_FP below is the reminder: a test
# fails when the scoring code changes until someone decides whether
# this needs a bump and refreshes the fingerprint.
#
# 2: cases whose modality the model lacks are not_applicable and leave
#    the denominator; python_exec without a code sandbox is an error, not
#    a failure; the endpoint is the run's own server or --remote-endpoint
#    (never whatever answers on a default port), and an endpoint that does
#    not serve the model under test is a harness error.
# 3: a case may forbid the served model's own chat-template tokens
#    (golden_truth.forbid_chat_template_tokens), read at run time from the
#    model's tokenizer, instead of a fixed list of one family's delimiters;
#    when that tokenizer cannot be loaded, or yields no token, the case is a
#    harness error, never a pass.
# 4: with a --chat-template of its own, the server's template also forbids its
#    delimiter-shaped literals (template_literal_delimiters) that the tokenizer
#    has no token for; the template and remote-code trust are read from any
#    spelling vLLM accepts, and --tokenizer-trust-remote-code sets the trust.
SCORING_VERSION = 4

# The functions whose code decides a verdict or what counts as a harness
# error (normalisation, comparison, tool-call matching, status assignment,
# the accuracy computation, the served-model check, the code sandbox), by
# qualified name, per file (relative to this package's runners directory).
_SCORING_CODE: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("golden.py", (
        "_normalize", "_extract_answer", "_extract_json_value", "_parse_tool_calls",
        "_argument_equal", "_tool_call_matches", "_split_response", "_verdict",
        "GoldenBenchmarkRunner.evaluate_task_response",
        "GoldenBenchmarkRunner._match_answer_exact",
        "GoldenBenchmarkRunner._match_json_exact",
        "GoldenBenchmarkRunner._match_contains_all",
        "GoldenBenchmarkRunner._match_regex",
        "GoldenBenchmarkRunner._match_refusal",
        "GoldenBenchmarkRunner._match_exact",
        "GoldenBenchmarkRunner._match_python_exec",
        "GoldenBenchmarkRunner._match_tool_call",
        "GoldenBenchmarkRunner._unsupported_modality",
        "GoldenBenchmarkRunner._request_json",
        "GoldenBenchmarkRunner._resolve_model_id",
        "_load_tokenizer", "_served_chat_template", "_chat_template_texts",
        "template_literal_delimiters", "chat_template_tokens",
        "GoldenBenchmarkRunner._template_leak_tokens",
    )),
    ("eval_suites/sandbox.py", ("run_sandboxed", "sandbox_skip_reason")),
)


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """``node`` with the docstring of every function and class in it removed."""
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = sub.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                sub.body = body[1:] or [ast.Pass()]
    return node


def _canon(node: Any) -> str:
    """A canonical text of an AST: node types and field values only (no positions), leaving out
    empty fields, so the same code gives the same text on every supported Python version."""
    if isinstance(node, ast.AST):
        parts = []
        for name, value in ast.iter_fields(node):
            if name in ("type_comment", "kind", "type_params") or value is None or value == []:
                continue
            parts.append(f"{name}={_canon(value)}")
        return f"{type(node).__name__}({','.join(parts)})"
    if isinstance(node, list):
        return "[" + ",".join(_canon(v) for v in node) + "]"
    return repr(node)


def _function_index(tree: ast.Module) -> Dict[str, ast.AST]:
    """Top-level functions and class methods of a module by qualified name."""
    index: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            index[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index[f"{node.name}.{item.name}"] = item
    return index


def scoring_fingerprint(root: Optional[Path] = None) -> str:
    """sha256 (first 16 hex) of the docstring-free code of _SCORING_CODE. Comments,
    docstrings, formatting and code outside those functions do not move it; any change
    to their code does. ``root`` is the runners directory (this file's by default)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    parts = []
    for rel, names in _SCORING_CODE:
        tree = ast.parse((base / rel).read_text(encoding="utf-8"))
        index = _function_index(tree)
        for name in names:
            node = index.get(name)
            parts.append(f"{rel}:{name}=" + (_canon(_strip_docstrings(node))
                                              if node is not None else "<missing>"))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


# The fingerprint of the scoring code SCORING_VERSION was last reviewed
# against. tests/test_serving_servers_and_golden_endpoint.py fails when scoring_fingerprint()
# differs: bump SCORING_VERSION if verdicts can change, then set this to the
# new value.
GOLDEN_SCORING_FP = "0be2c393a287f222"

# Sampling is pinned so a golden run is reproducible. A case may
# override any of these through its own top-level "sampling" block, and
# whatever is actually used is echoed back in the task result.
DEFAULT_SAMPLING = {"temperature": 0.0, "max_tokens": 512}

# Status codes worth another attempt. Everything else is a real answer
# from the server and retrying it just wastes time.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_ERROR = "error"
# A case whose required modality the model architecturally lacks (verified from
# the model config, e.g. no audio tower) is NOT scored: it is neither a pass, a
# fail, nor a harness error. It is excluded from the denominator so a model is
# graded only on the modalities it actually has. This is capability-gating, not
# a lazy skip: a model that CAN do the modality but whose SERVER is
# misconfigured still errors (a real signal).
STATUS_NOT_APPLICABLE = "not_applicable"

# Every shipped case declares a category, but a case authored elsewhere
# and pointed at with --golden-tasks might not. Bucket those rather than
# letting a None key reach the summary.
UNCATEGORIZED = "uncategorized"

# Message fields that are part of the OpenAI chat schema but are not
# role or content. An assistant turn that made a tool call carries
# "tool_calls", and the "tool" turn answering it carries "tool_call_id"
# and "name". They have to survive into the request body untouched,
# because the server-side chat template is what serializes them and a
# case that never sends them cannot tell you whether that works.
PASSTHROUGH_MESSAGE_KEYS = ("tool_calls", "tool_call_id", "name")

# golden_truth key. A case that sets it to true also forbids, in the visible
# answer, every chat-template token of the served model, read at run time
# from that model's own tokenizer (chat_template_tokens). A hand-written
# delimiter list only covers the model family it was written for; the
# tokenizer names the delimiters of whatever model is under test. The
# tokens actually checked are recorded in the summary
# (chat_template_leak_check).
FORBID_TEMPLATE_TOKENS = "forbid_chat_template_tokens"

# Longest tokenizer-load error message kept in a result (Hub errors run long).
_MAX_ERROR_CHARS = 400


class GoldenHarnessError(RuntimeError):
    """The harness could not obtain an answer from the model.

    Distinct from the model getting a case wrong. A connection failure,
    an auth failure or a malformed response says nothing about the
    model, so it must never be scored as one.
    """


def _gcloud_identity_token() -> Optional[str]:
    """Return a gcloud identity token, or None with a logged reason."""
    try:
        res = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as err:
        logger.warning(f"gcloud identity token lookup failed: {err}")
        return None

    if res.returncode != 0:
        detail = res.stderr.strip()[:200]
        logger.warning(
            "gcloud auth print-identity-token exited "
            f"{res.returncode}: {detail}"
        )
        return None

    token = res.stdout.strip() or None
    if token:
        # Not in the environment: name it a credential so no log line or result file keeps it.
        from ..utils.redaction import register_secret
        register_secret(token)
    return token


def _normalize(text: str, mode: str) -> str:
    """Normalize an answer before comparison.

    ``strip``       trim surrounding whitespace (default)
    ``whitespace``  additionally collapse internal whitespace runs
    ``lower``       collapse whitespace and lowercase
    ``code``        collapse whitespace and drop it around punctuation, so
                    ``s == s[::-1]`` and ``s==s[::-1]`` compare equal while
                    the boundary in ``return s`` is preserved
    """
    if mode == "whitespace":
        return " ".join(text.split())
    if mode == "lower":
        return " ".join(text.split()).lower()
    if mode == "code":
        collapsed = " ".join(text.split())
        return re.sub(r"\s*([^\w\s])\s*", r"\1", collapsed)
    return text.strip()


def _extract_answer(text: str, pattern: str) -> Optional[str]:
    """Return the LAST match of ``pattern`` in ``text``.

    Models routinely restate a value while reasoning, and the value they
    committed to is the final one. Taking the last match is what makes
    "I first computed 42, but the correct answer is 39" fail against a
    golden of 42, where a substring check would pass it.

    Returns the first capturing group if the pattern has one, otherwise
    the whole match. Returns None when the pattern never matches, which
    is itself a failure: the model did not answer in the required shape.
    """
    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
    if not matches:
        return None
    last = matches[-1]
    return last.group(1) if last.groups() else last.group(0)


def _extract_json_value(text: str) -> Tuple[bool, Any]:
    """Pull the last top-level JSON value out of ``text``.

    Returns ``(found, value)``. Scanning to the last top-level value
    rather than the first means a model that shows a wrong object then
    corrects itself is judged on the correction, consistent with
    ``_extract_answer``. Nested objects are skipped because the decoder
    consumes the whole enclosing value in one step.
    """
    decoder = json.JSONDecoder()
    found = False
    value: Any = None
    idx = 0
    while idx < len(text):
        if text[idx] not in "{[":
            idx += 1
            continue
        try:
            decoded, end = decoder.raw_decode(text[idx:])
        except ValueError:
            idx += 1
            continue
        found, value = True, decoded
        idx += end
    return found, value


def _parse_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI ``message.tool_calls`` into name/arguments dicts.

    ``arguments`` arrives as a JSON string on the wire. Arguments that
    do not parse are kept under ``__unparsed__`` so the failure detail
    can show what the model actually emitted.
    """
    parsed = []
    for call in raw or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function", {})
        if not isinstance(fn, dict):
            fn = {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"__unparsed__": args}
        if not isinstance(args, dict):
            args = {"__unparsed__": args}
        parsed.append({"name": fn.get("name", ""), "arguments": args})
    return parsed


def _argument_equal(got: Any, want: Any) -> bool:
    """Compare one tool-call argument value."""
    if isinstance(got, str) and isinstance(want, str):
        return _normalize(got, "lower") == _normalize(want, "lower")
    return got == want


def _tool_call_matches(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> Tuple[bool, str]:
    """Check one actual tool call against one expected tool call.

    Expected arguments must all be present and equal. Extra arguments
    the model volunteers (a unit, a format) are allowed, so a case does
    not have to enumerate every optional parameter in the schema.
    """
    if expected.get("name", "") != actual.get("name", ""):
        return False, f"name '{actual.get('name')}'"

    want_args = expected.get("arguments", {}) or {}
    got_args = actual.get("arguments", {}) or {}
    for key, want_val in want_args.items():
        if key not in got_args:
            return False, f"missing argument '{key}'"
        if not _argument_equal(got_args[key], want_val):
            return False, (
                f"argument '{key}' was {got_args[key]!r}, "
                f"expected {want_val!r}"
            )
    return True, "ok"


def _split_response(
    response: Union[str, Dict[str, Any], None]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Accept either a raw string or a parsed message dict."""
    if isinstance(response, dict):
        return response.get("content") or "", response.get("tool_calls") or []
    return response or "", []


def _verdict(counts: Dict[str, int], total: int) -> Tuple[str, Optional[float], int]:
    """(status, accuracy_percent, applicable cases) of a run from its per-status counts.

    Any error: status error and no accuracy (a partial run has no meaningful score). Else the
    denominator is the applicable cases (not_applicable ones leave it): failed cases make the
    status failed with passed/applicable; none make it passed at 100 (None when no case was
    applicable)."""
    errors = counts.get(STATUS_ERROR, 0)
    failed = counts.get(STATUS_FAILED, 0)
    passed = counts.get(STATUS_PASSED, 0)
    applicable = total - counts.get(STATUS_NOT_APPLICABLE, 0)
    if errors:
        return STATUS_ERROR, None, applicable
    if failed:
        return STATUS_FAILED, (round(passed / applicable * 100.0, 2) if applicable else None), \
            applicable
    return STATUS_PASSED, (100.0 if applicable else None), applicable


def _load_tokenizer(name: str, trust_remote_code: bool = False) -> Any:
    """Load the tokenizer ``name`` (a HuggingFace repo id or a local directory) with
    transformers, the same way for every model family. ``trust_remote_code``:
    config.tokenizer_trust_remote_code (--tokenizer-trust-remote-code, else --trust-remote-code in
    --vllm-serve-args). Raises on any failure; the caller reports it as a harness error."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)


def _served_chat_template(serve_args: Any) -> Optional[str]:
    """The template text a server gbench launched was told to use instead of the tokenizer's
    (--chat-template in --vllm-serve-args, any spelling vLLM accepts: `--chat-template X`,
    `--chat-template=X`, `--chat_template ...`; a file path, or the template itself), else
    None."""
    from ..core.config import serve_arg_value
    value = serve_arg_value(serve_args, "--chat-template")
    if not value:
        return None
    path = Path(value).expanduser()
    return path.read_text(encoding="utf-8") if path.is_file() else value


def _chat_template_texts(template: Any) -> List[str]:
    """The template source text(s) a tokenizer carries: one string, or several named
    templates (a mapping such as {"default": ..., "tool_use": ...}, or a list of
    {"name": ..., "template": ...} entries)."""
    if isinstance(template, str):
        return [template]
    if isinstance(template, dict):
        return [t for t in template.values() if isinstance(t, str)]
    if isinstance(template, (list, tuple)):
        return [item["template"] for item in template
                if isinstance(item, dict) and isinstance(item.get("template"), str)]
    return []


#: Delimiter-shaped literals of a chat template: `<|...|>`, and `<...>` / `</...>` names that
#: carry an underscore (`<start_of_turn>`, `<tool_call>`), which ordinary prose does not write.
_TEMPLATE_DELIMITER_RE = re.compile(r"<\|[^<>|\s]{1,64}\|>|</?[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*>")


def template_literal_delimiters(template: Optional[str], tokens: Any = ()) -> List[str]:
    """The delimiter-shaped literals (_TEMPLATE_DELIMITER_RE) a chat template's source writes
    that are not already among `tokens`: a --chat-template whose turn markers are literal
    strings rather than tokens of the tokenizer still renders them into the prompt, so they
    must not reach an answer either."""
    if not template:
        return []
    have = set(tokens or ())
    return sorted({m.group(0) for m in _TEMPLATE_DELIMITER_RE.finditer(str(template))} - have)


def chat_template_tokens(tokenizer: Any, template: Optional[str] = None) -> Dict[str, Any]:
    """The chat-template tokens of a tokenizer: the strings its template consumes.

    Every special token (``all_special_tokens``) plus every added token
    (``get_added_vocab()``) that occurs in the chat template text: ``template`` when the
    server was given one of its own, else the tokenizer's. Three kinds are left out because
    ordinary prose contains them, so forbidding them would fail correct answers:
    whitespace-only tokens, single characters, and tokens made only of letters, digits,
    underscores and spaces. Returns ``tokens`` (sorted), ``excluded`` (each with its reason)
    and ``chat_template_found`` (whether there was any template text at all).
    """
    specials = [t for t in (getattr(tokenizer, "all_special_tokens", None) or [])
                if isinstance(t, str)]
    get_added = getattr(tokenizer, "get_added_vocab", None)
    added = [t for t in ((get_added() if callable(get_added) else None) or {})
             if isinstance(t, str)]
    template_text = "\n".join(_chat_template_texts(
        template if template is not None else getattr(tokenizer, "chat_template", None)))
    candidates = set(specials) | {t for t in added if t and t in template_text}
    tokens: List[str] = []
    excluded: List[Dict[str, str]] = []
    for tok in sorted(candidates):
        core = tok.strip()
        if not core:
            excluded.append({"token": tok, "reason": "whitespace only"})
        elif len(core) == 1:
            excluded.append({"token": tok, "reason": "single character"})
        elif re.fullmatch(r"[\w ]+", core):
            excluded.append({"token": tok, "reason": "plain word"})
        else:
            tokens.append(tok)
    return {"tokens": tokens, "excluded": excluded,
            "chat_template_found": bool(template_text)}


class GoldenBenchmarkRunner:
    """Runner for Golden Set benchmark tasks."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize Golden Set runner.

        Args:
            config: Benchmark configuration instance
        """
        self.config = config
        self.dataset_dir = Path(__file__).parent.parent / "golden_dataset"
        # The resolved served id per (endpoint, model): one runner serves every model of a run,
        # each on its own server.
        self._resolved: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._model_id_source: Optional[str] = None
        # The chat-template tokens per tokenizer name, loaded once per run.
        self._template_tokens: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load_tasks(
        self, selected_tasks: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load Golden Set task definitions from JSON files.

        Args:
            selected_tasks: Optional filter for specific task IDs or file
                names.

        Returns:
            List of task definition dictionaries, sorted by task id so a
            run is ordered the same way every time.
        """
        tasks: List[Dict[str, Any]] = []
        if not self.dataset_dir.exists():
            logger.error(
                f"Golden dataset directory {self.dataset_dir} does not "
                "exist. If gbench was installed as a wheel this means the "
                "dataset was not packaged."
            )
            return tasks

        for file_path in sorted(self.dataset_dir.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as err:
                logger.error(f"Failed to parse golden task {file_path}: {err}")
                continue

            task_id = data.get("id", file_path.stem)
            if selected_tasks and (
                task_id not in selected_tasks
                and file_path.name not in selected_tasks
            ):
                continue
            tasks.append(data)

        return sorted(tasks, key=lambda t: str(t.get("id", "")))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_task_response(
        self,
        task: Dict[str, Any],
        response: Union[str, Dict[str, Any], None],
        template_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a model output against golden_truth expectations.

        Args:
            task: Task specification dictionary.
            response: Either the raw response text or the parsed message
                dict returned by :meth:`_query_remote_endpoint`.
            template_tokens: The served model's chat-template tokens
                (:func:`chat_template_tokens`). Required by a case that
                sets ``forbid_chat_template_tokens``; they are forbidden in
                the answer alongside its ``expected_not_contains``.

        Returns:
            Dictionary containing the verdict and metadata. ``status`` is
            one of ``passed``, ``failed`` or ``error``. An ``error`` here
            means the case itself is malformed, or its leak check has no
            tokens to check, not that the model was wrong.
        """
        golden = task.get("golden_truth", {})
        match_type = golden.get("match_type", "")
        text, tool_calls = _split_response(response)
        forbid_template = bool(golden.get(FORBID_TEMPLATE_TOKENS))

        handlers = {
            "answer_exact": self._match_answer_exact,
            "json_exact": self._match_json_exact,
            "contains_all": self._match_contains_all,
            "regex": self._match_regex,
            "refusal": self._match_refusal,
            "exact": self._match_exact,
            "python_exec": self._match_python_exec,
        }

        if forbid_template and not template_tokens:
            status, details = STATUS_ERROR, (
                f"Case '{task.get('id')}' forbids the served model's chat-template "
                "tokens, but none were derived from its tokenizer, so its leak check "
                "cannot run. This is a harness error, not a verdict on the model."
            )
        elif match_type == "tool_call":
            status, details = self._match_tool_call(golden, tool_calls)
        elif match_type in handlers:
            status, details = handlers[match_type](golden, text)
        else:
            status, details = (
                STATUS_ERROR,
                f"Unknown match_type '{match_type}' in case "
                f"'{task.get('id')}'",
            )

        # Two-sided assertion. Any match type may declare markers that
        # must be absent, which is what stops a refusal token embedded in
        # a compliance sentence from scoring as a refusal. A case may add the
        # served model's own chat-template tokens to the forbidden strings.
        if status == STATUS_PASSED:
            forbidden = list(golden.get("expected_not_contains", []) or [])
            if forbid_template:
                forbidden += [t for t in template_tokens or [] if t not in forbidden]
            lowered = text.lower()
            hits = [item for item in forbidden if item.lower() in lowered]
            if hits:
                status = STATUS_FAILED
                details = f"Forbidden content present: {hits}"

        result = {
            "task_id": task.get("id"),
            "title": task.get("title"),
            "category": task.get("category") or UNCATEGORIZED,
            "status": status,
            "passed": status == STATUS_PASSED,
            "match_type": match_type,
            "details": details,
            "response_text": text,
            "tool_calls": tool_calls,
        }
        if forbid_template:
            result["chat_template_tokens_checked"] = len(template_tokens or [])
        return result

    def _match_answer_exact(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Extract the answer span, then compare it exactly."""
        pattern = golden.get("answer_pattern")
        if not pattern:
            return STATUS_ERROR, "answer_exact requires 'answer_pattern'"

        mode = golden.get("normalize", "strip")
        want = _normalize(golden.get("canonical_answer", ""), mode)
        extracted = _extract_answer(text, pattern)
        if extracted is None:
            return STATUS_FAILED, (
                f"answer_pattern matched nothing, so no answer was given "
                f"in the required form. Expected '{want}'"
            )

        got = _normalize(extracted, mode)
        if got == want:
            return STATUS_PASSED, f"Extracted answer '{got}' matches"
        return STATUS_FAILED, f"Extracted '{got}', expected '{want}'"

    def _match_json_exact(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Parse both sides as JSON and compare the parsed values."""
        canonical = golden.get("canonical_answer", "")
        try:
            want = json.loads(canonical)
        except json.JSONDecodeError as err:
            return STATUS_ERROR, f"canonical_answer is not valid JSON: {err}"

        found, got = _extract_json_value(text)
        if not found:
            return STATUS_FAILED, "Response contained no parseable JSON value"
        if got == want:
            return STATUS_PASSED, f"JSON matches canonical: {canonical}"
        return STATUS_FAILED, (
            f"JSON mismatch. Got {json.dumps(got, sort_keys=True)}, "
            f"expected {json.dumps(want, sort_keys=True)}"
        )

    def _match_contains_all(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Every expected substring must be present.

        Only appropriate where the answer is a fixed token wrapped in
        free-form framing. Pair it with ``expected_not_contains``.
        """
        expected = golden.get("expected_outputs", []) or []
        if not expected:
            return STATUS_ERROR, "contains_all requires 'expected_outputs'"
        lowered = text.lower()
        missing = [i for i in expected if i.lower() not in lowered]
        if missing:
            return STATUS_FAILED, f"Missing expected substrings: {missing}"
        return STATUS_PASSED, f"All expected strings found: {expected}"

    def _match_regex(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Every declared pattern must match."""
        patterns = golden.get("regex_patterns", []) or []
        if not patterns:
            return STATUS_ERROR, "regex requires 'regex_patterns'"
        missing = [
            p for p in patterns if not re.search(p, text, re.IGNORECASE)
        ]
        if missing:
            return STATUS_FAILED, f"Missing patterns: {missing}"
        return STATUS_PASSED, f"All {len(patterns)} patterns matched"

    def _match_refusal(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """At least one refusal marker must be present.

        On its own this is weak, which is why ``expected_not_contains``
        is mandatory here. The pair is what makes the check two-sided.
        """
        markers = golden.get("expected_any", []) or []
        if not markers:
            return STATUS_ERROR, "refusal requires 'expected_any'"
        if not golden.get("expected_not_contains"):
            return STATUS_ERROR, (
                "refusal requires 'expected_not_contains'. A refusal "
                "marker alone passes on text that refuses and then "
                "complies anyway."
            )
        lowered = text.lower()
        hits = [m for m in markers if m.lower() in lowered]
        if not hits:
            return STATUS_FAILED, f"No refusal marker from {markers} found"
        return STATUS_PASSED, f"Refusal markers found: {hits}"

    def _match_exact(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Whole response must equal the canonical answer."""
        canonical = golden.get("canonical_answer", "")
        mode = golden.get("normalize", "strip")
        got = _normalize(text, mode)
        want = _normalize(canonical, mode)
        if got == want:
            return STATUS_PASSED, f"Exact match: '{want}'"
        return STATUS_FAILED, f"Expected '{want}', got '{got}'"

    def _match_python_exec(
        self, golden: Dict[str, Any], text: str
    ) -> Tuple[str, str]:
        """Extract Python code blocks and execute against unit_tests assertions."""
        unit_tests = golden.get("unit_tests", [])
        if not unit_tests:
            return STATUS_ERROR, "python_exec requires 'unit_tests'"

        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        code_to_exec = "\n\n".join(code_blocks) if code_blocks else text

        # Model-written code runs in a bubblewrap sandbox (subprocess), NEVER in-process:
        # exec()ing model output inside the gbench process is arbitrary code execution with
        # our filesystem + network. If the sandbox is unavailable this check reports an ERROR
        # (env/harness issue) instead of executing unsandboxed - see GBENCH_SANDBOX.
        from .eval_suites.sandbox import run_sandboxed, sandbox_skip_reason
        skip = sandbox_skip_reason()
        if skip:
            return STATUS_ERROR, f"python_exec needs the code sandbox: {skip}"
        script = code_to_exec + "\n\n" + "\n".join(str(t) for t in unit_tests) + "\n"
        try:
            res = run_sandboxed(
                [sys.executable, "-c", script], capture_output=True, text=True,
                timeout=int(os.environ.get("GOLDEN_EXEC_TIMEOUT_S", "30")))
            if res.returncode == 0:
                return STATUS_PASSED, f"Passed all {len(unit_tests)} unit test assertions"
            detail = (res.stderr or res.stdout or "").strip().splitlines()
            return STATUS_FAILED, (
                f"Execution/assertion failed: {detail[-1] if detail else 'nonzero exit'}")
        except Exception as e:
            return STATUS_FAILED, f"Execution/assertion failed: {e}"

    def _match_tool_call(
        self, golden: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Structured tool calls must match, prose about them must not."""
        expected = golden.get("expected_tool_calls", []) or []
        if not expected:
            return STATUS_ERROR, "tool_call requires 'expected_tool_calls'"
        if not tool_calls:
            return STATUS_FAILED, (
                "Model returned no structured tool_calls. Naming the "
                "function in prose does not count."
            )

        reasons = []
        for want in expected:
            hit = False
            for actual in tool_calls:
                ok, reason = _tool_call_matches(want, actual)
                if ok:
                    hit = True
                    break
                reasons.append(reason)
            if not hit:
                return STATUS_FAILED, (
                    f"No tool call matched {want.get('name')}"
                    f"({want.get('arguments', {})}). Rejected: {reasons}"
                )
        matched = f"All {len(expected)} expected tool calls matched"
        return STATUS_PASSED, matched

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Build request headers, resolving the bearer token once: the endpoint key
        (VLLM_API_KEY, else OPENAI_API_KEY; placeholders ignored: base.endpoint_api_key), else,
        for a remote endpoint with GBENCH_GCP_AUTH=1, a gcloud identity token."""
        if not hasattr(self, "_cached_api_key"):
            from .eval_suites.base import endpoint_api_key
            key = endpoint_api_key()
            if not key:
                endpoint = (self.config.remote_endpoint or "").lower()
                is_local = not endpoint or "localhost" in endpoint or "127.0.0.1" in endpoint or "0.0.0.0" in endpoint
                # GCP identity-token auto-auth is OPT-IN (GBENCH_GCP_AUTH=1): a public user pointing
                # at any remote endpoint should not silently spawn a `gcloud` subprocess. Mirrors the
                # quality runner's gate.
                if not is_local and os.environ.get("GBENCH_GCP_AUTH", "").strip().lower() in ("1", "true", "yes"):
                    key = _gcloud_identity_token()
                elif not is_local:
                    logger.info("Remote endpoint with no VLLM_API_KEY / OPENAI_API_KEY; GCP "
                                "identity-token auto-auth is opt-in (set GBENCH_GCP_AUTH=1). "
                                "Proceeding unauthenticated.")
            self._cached_api_key = key

        headers = {"Content-Type": "application/json"}
        if self._cached_api_key:
            headers["Authorization"] = f"Bearer {self._cached_api_key}"
        return headers

    def _request_json(
        self, req: urllib.request.Request, timeout: int = DEFAULT_TIMEOUT_S
    ) -> Dict[str, Any]:
        """Send a request, retrying transient network failures only.

        Application errors and malformed output are not retried. They are
        deterministic, so a second attempt only delays the report.

        Raises:
            GoldenHarnessError: on any failure to obtain parsed JSON.
        """
        delay = BACKOFF_BASE_S
        last_err: Optional[BaseException] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                last_err = err
                if err.code not in RETRYABLE_STATUS:
                    raise GoldenHarnessError(
                        f"HTTP {err.code} from {req.full_url}: {err.reason}"
                    ) from err
            except json.JSONDecodeError as err:
                raise GoldenHarnessError(
                    f"Endpoint {req.full_url} returned non-JSON: {err}"
                ) from err
            except (urllib.error.URLError, OSError) as err:
                last_err = err

            if attempt < MAX_ATTEMPTS:
                wait = min(delay, BACKOFF_CAP_S)
                logger.warning(
                    f"Attempt {attempt}/{MAX_ATTEMPTS} to {req.full_url} "
                    f"failed ({last_err}); retrying in {wait:.0f}s"
                )
                time.sleep(wait)
                delay *= 2

        raise GoldenHarnessError(
            f"{MAX_ATTEMPTS} attempts to {req.full_url} failed: {last_err}"
        )

    def _resolve_model_id(
        self, endpoint_url: str, model_config: ModelConfig
    ) -> Optional[str]:
        """Decide which model name to put in the request payload.

        Omitting it, or taking whatever the endpoint lists first, lets a
        multi-model endpoint, a router or an unrelated server on the port
        answer from a different checkpoint than the one the report names,
        which would invalidate the whole run. The id is resolved from the
        endpoint's /v1/models listing (utils.endpoint.resolve_served_model):
        --served-model-id (alias --golden-model-id) first, then the model's
        HF id, path, name and short name. The server gbench launched for
        this run (config.local_endpoint) must list it; a remote endpoint
        whose listing cannot be read is sent the configured id, and the
        summary records that.

        Raises:
            GoldenHarnessError: when the endpoint serves no model matching
                this run (a different model, or several and none of them
                this one), or the run's own server cannot be listed.
        """
        key = (endpoint_url, str(getattr(model_config, "name", "") or model_config))
        if key in self._resolved:
            model_id, self._model_id_source = self._resolved[key]
            return model_id

        from ..utils.endpoint import EndpointError, resolve_served_model
        override = (getattr(self.config, "served_model_id", None)
                    or getattr(self.config, "golden_model_id", None))
        local = not getattr(self.config, "remote_endpoint", None)
        candidates = [getattr(model_config, "hf_model_id", None)]
        try:
            candidates.append(model_config.get_model_path(ModelFormat.HF))
        except Exception:                                                  # noqa: BLE001
            pass
        candidates += [getattr(model_config, "name", None),
                       getattr(model_config, "short_name", None)]
        headers = {k: v for k, v in self._auth_headers().items() if k == "Authorization"}
        try:
            served = resolve_served_model(endpoint_url, candidates, override, local=local,
                                          headers=headers)
        except EndpointError as err:
            raise GoldenHarnessError(
                f"{err}. The Golden Set sends every case to the model under test only"
            ) from err
        self._resolved[key] = (served.id, served.source)
        self._model_id_source = served.source
        logger.info(f"Golden Set will request model '{served.id}' ({served.source})")
        return served.id

    def _template_leak_tokens(
        self, model_config: Any, requested_model: Optional[str]
    ) -> Dict[str, Any]:
        """The served model's chat-template tokens, for the cases that forbid them.

        The tokenizer is --tokenizer when given, else the model's id (the
        checkpoint gbench serves, or the one a remote endpoint was named
        for), else the id the endpoint serves it under; it is loaded once per
        run. The template text is the one a server gbench launched was given
        with --chat-template (in --vllm-serve-args), else the tokenizer's. A
        tokenizer that cannot be loaded, or that yields no token, sets
        ``error``: every case that needs the tokens is then a harness error,
        never a pass, because a leak check with nothing to check proves
        nothing. The returned record is what the summary stores as
        ``chat_template_leak_check``.
        """
        name = getattr(self.config, "tokenizer", None)
        source = getattr(self.config, "tokenizer_source", None) or "--tokenizer"
        if not name:
            name, source = getattr(model_config, "hf_model_id", None), "model id"
        if not name:
            name, source = requested_model, "served model id"
        if not name:
            return {
                "tokenizer": None, "tokenizer_source": None, "tokens": [],
                "error": ("the chat-template leak check needs the served model's tokenizer, "
                          "and this run names none: pass --tokenizer <HF repo id or local "
                          "directory of the served model's tokenizer>. This is a harness "
                          "error, not a verdict on the model."),
            }
        if name in self._template_tokens:
            return self._template_tokens[name]

        record: Dict[str, Any] = {"tokenizer": name, "tokenizer_source": source}
        # The trust for the tokenizer's code (--tokenizer-trust-remote-code, else the trust the
        # operator gave the model's server in --vllm-serve-args) and, for a server gbench
        # launched, any template it was told to use instead of the tokenizer's. A remote
        # endpoint's template is unknown, so its tokenizer's is read.
        from ..core.config import tokenizer_trust_remote_code
        serve_args = [str(a) for a in (getattr(self.config, "vllm_serve_args", None) or [])]
        remote = bool(getattr(self.config, "remote_endpoint", None))
        trust, trust_source = tokenizer_trust_remote_code(self.config)
        record["trust_remote_code"] = trust
        record["trust_remote_code_source"] = trust_source
        problem = None
        needs_trust = False
        try:
            template = None if remote else _served_chat_template(serve_args)
            record["chat_template_source"] = ("--chat-template" if template is not None
                                              else "tokenizer")
            tokenizer = _load_tokenizer(name, trust_remote_code=trust)
            record["tokenizer_class"] = type(tokenizer).__name__
            record.update(chat_template_tokens(tokenizer, template))
            literals = template_literal_delimiters(template, record.get("tokens"))
            if literals:
                # a --chat-template's literal turn markers the tokenizer has no token for
                record["template_literal_delimiters"] = literals
                record["tokens"] = sorted(set(record.get("tokens") or []) | set(literals))
        except Exception as err:                                           # noqa: BLE001
            detail = " ".join(str(err).split())
            needs_trust = "trust_remote_code" in detail and not trust
            if len(detail) > _MAX_ERROR_CHARS:
                detail = detail[:_MAX_ERROR_CHARS] + "..."
            problem = f"could not be loaded or read ({type(err).__name__}: {detail})"
        if problem is None and not record.get("tokens"):
            problem = "has no special token and no added token in its chat template"
        if problem is not None:
            record.setdefault("tokens", [])
            hint = (" If the tokenizer ships its own code (the error names trust_remote_code), "
                    "pass --tokenizer-trust-remote-code." if needs_trust else "")
            record["error"] = (
                f"the chat-template leak check needs the served model's tokenizer, and "
                f"'{name}' ({source}) {problem}. Pass --tokenizer <HF repo id or local "
                "directory of the served model's tokenizer> (transformers must be "
                f"installed).{hint} This is a harness error, not a verdict on the model."
            )
            logger.error(f"Golden Set: {record['error']}")
        else:
            logger.info(
                f"Golden Set chat-template leak check: {len(record['tokens'])} tokens from "
                f"tokenizer '{name}' ({source})"
            )
        self._template_tokens[name] = record
        return record

    def _format_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert task messages into OpenAI chat content parts.

        Fields outside role and content are copied across unchanged. A
        case that replays a tool call needs the assistant turn to keep
        its "tool_calls" and the tool turn to keep the "tool_call_id"
        tying it back, or the endpoint sees an unanswered call.
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            media = msg.get("media", [])

            entry: Dict[str, Any] = {"role": role, "content": raw_content}
            for key in PASSTHROUGH_MESSAGE_KEYS:
                if key in msg:
                    entry[key] = msg[key]

            if not media:
                formatted.append(entry)
                continue

            content_parts: List[Dict[str, Any]] = []
            if isinstance(raw_content, str) and raw_content:
                content_parts.append({"type": "text", "text": raw_content})

            for item in media:
                media_type = item.get("type", "")
                url = item.get("url") or item.get("data") or ""
                mime = item.get(
                    "mime_type",
                    "image/jpeg" if media_type == "image" else "audio/wav",
                )

                is_remote = url.startswith(
                    ("http://", "https://", "data:")
                )
                if url and not is_remote:
                    asset_path = self.dataset_dir / url
                    if not asset_path.exists():
                        raise GoldenHarnessError(
                            f"Golden asset {asset_path} is missing"
                        )
                    try:
                        with open(asset_path, "rb") as af:
                            b64 = base64.b64encode(af.read()).decode("utf-8")
                    except OSError as err:
                        raise GoldenHarnessError(
                            f"Failed to read golden asset {asset_path}: {err}"
                        ) from err
                is_audio = media_type == "audio" or mime.startswith("audio")
                if not is_remote:
                    if is_audio:
                        url = b64
                    else:
                        url = f"data:{mime};base64,{b64}"

                if media_type == "image" or mime.startswith("image"):
                    content_parts.append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                elif is_audio:
                    audio_data = url.split(",")[-1] if "," in url else url
                    content_parts.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": item.get("format", "wav"),
                        },
                    })

            entry["content"] = content_parts
            formatted.append(entry)
        return formatted

    def _query_remote_endpoint(
        self,
        messages: List[Dict[str, Any]],
        endpoint_url: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        sampling: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query an OpenAI-compatible endpoint and return its message.

        Args:
            messages: Task messages, optionally carrying media parts.
            endpoint_url: Base endpoint URL.
            tools: Optional OpenAI tool schemas to advertise.
            tool_choice: Optional tool_choice directive.
            sampling: Resolved sampling parameters.
            model_id: Model name to request.

        Returns:
            Dict with ``content``, ``tool_calls`` and ``served_model``.

        Raises:
            GoldenHarnessError: if no answer could be obtained. Callers
                must not treat this as the model being wrong.
        """
        target_url = endpoint_url.rstrip("/") + "/chat/completions"
        resolved = dict(DEFAULT_SAMPLING)
        resolved.update(sampling or {})

        body: Dict[str, Any] = {
            "messages": self._format_messages(messages),
            "temperature": resolved["temperature"],
            "max_tokens": resolved["max_tokens"],
        }
        if model_id:
            body["model"] = model_id
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        req = urllib.request.Request(
            target_url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._auth_headers(),
            method="POST",
        )

        data = self._request_json(req)
        choices = data.get("choices", [])
        if not choices:
            raise GoldenHarnessError(
                f"{target_url} returned no choices: "
                f"{json.dumps(data)[:200]}"
            )

        message = choices[0].get("message", {}) or {}
        return {
            "content": message.get("content") or "",
            "tool_calls": _parse_tool_calls(message.get("tool_calls")),
            "served_model": data.get("model", ""),
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _error_summary(
        self, model_name: str, message: str
    ) -> Dict[str, Any]:
        """Build a summary for a run that never got off the ground."""
        logger.error(message)
        return {
            "model": model_name,
            "status": STATUS_ERROR,
            "total_tasks": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "error_cases": 1,
            "accuracy_percent": None,
            "harness_errors": [message],
            "task_results": [],
        }

    @staticmethod
    def _unsupported_modality(
        task: Dict[str, Any], model_config: Any
    ) -> Optional[str]:
        """Return a modality the task requires but the model does not have, else
        None. Capability comes from the model config (``supports_audio`` /
        ``supports_multimodal``, derived from the checkpoint's audio_config /
        vision_config), NOT from the server, so a genuine architectural absence
        (e.g. gemma-4-26B has no audio tower) is marked not-applicable while a
        model that CAN do the modality but whose server is misconfigured still
        errors. Text-only cases require nothing and are always applicable."""
        required = set()
        for msg in task.get("conversation", {}).get("messages", []):
            for item in (msg.get("media") or []):
                mtype = (item.get("type") or "").lower()
                mime = (item.get("mime_type") or "").lower()
                if mtype == "audio" or mime.startswith("audio"):
                    required.add("audio")
                elif (mtype in ("image", "video")
                      or mime.startswith(("image", "video"))):
                    required.add("vision")
        if "audio" in required and not getattr(
            model_config, "supports_audio", False
        ):
            return "audio"
        if "vision" in required and not getattr(
            model_config, "supports_multimodal", False
        ):
            return "vision"
        return None

    def run(
        self,
        model_config: ModelConfig,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute Golden Set benchmarks for a model.

        Args:
            model_config: Model configuration
            tasks: Optional pre-loaded task list

        Returns:
            Summary dictionary. ``status`` is ``passed``, ``failed`` or
            ``error``. ``accuracy_percent`` is None whenever any case
            errored, because a partial run has no meaningful score.
            ``chat_template_leak_check`` records the tokenizer and the
            chat-template tokens the leak-checking cases forbade (None when
            no selected case needed them).
        """
        if tasks is None:
            tasks = self.load_tasks(self.config.selected_golden_tasks)

        model_name = getattr(model_config, "name", "") or getattr(
            model_config, "short_name", str(model_config)
        )

        if not tasks:
            return self._error_summary(
                model_name,
                f"No Golden Set tasks loaded from {self.dataset_dir}. The "
                "dataset is missing or --golden-tasks matched nothing. "
                "This is a harness error, not a score of zero.",
            )

        # The run's own endpoint: --remote-endpoint (normalized to end in /v1), else the
        # server the CLI launched for this model (config.local_endpoint). Never a default
        # port, where whatever happens to listen would answer.
        endpoint = (normalize_endpoint(getattr(self.config, "remote_endpoint", None))
                    or getattr(self.config, "local_endpoint", None))
        if not endpoint:
            return self._error_summary(
                model_name,
                "The Golden Set has no endpoint to query: pass --remote-endpoint, or run it "
                "through the gbench CLI, which launches the model's server (--num-gpus) and "
                "hands it to the Golden Set. This is a harness error, not a score.")
        try:
            requested_model = self._resolve_model_id(endpoint, model_config)
        except GoldenHarnessError as err:
            return self._error_summary(model_name, str(err))

        logger.info(
            f"Running Golden Set evaluation for {model_name} "
            f"({len(tasks)} tasks) against {endpoint}"
        )

        results: List[Dict[str, Any]] = []
        harness_errors: List[str] = []
        counts = {
            STATUS_PASSED: 0,
            STATUS_FAILED: 0,
            STATUS_ERROR: 0,
            STATUS_NOT_APPLICABLE: 0,
        }
        served_models = set()
        # The served model's chat-template tokens and where they came from,
        # recorded when a case needed them (None when none did).
        leak_check: Optional[Dict[str, Any]] = None

        for task in tasks:
            task_id = task.get("id")
            sampling = dict(DEFAULT_SAMPLING)
            sampling.update(task.get("sampling", {}) or {})

            # Capability gate: a case whose required modality the model lacks is
            # not applicable to this model. Excluded from the gate (not scored,
            # not counted in the denominator) so the model is judged only on the
            # modalities it has. Verified from the model config, never the server.
            unsupported = self._unsupported_modality(task, model_config)
            if unsupported:
                counts[STATUS_NOT_APPLICABLE] += 1
                results.append({
                    "task_id": task_id,
                    "title": task.get("title"),
                    "category": task.get("category") or UNCATEGORIZED,
                    "status": STATUS_NOT_APPLICABLE,
                    "passed": False,
                    "match_type": task.get("golden_truth", {}).get("match_type"),
                    "details": (
                        f"Not applicable: model '{model_name}' has no "
                        f"{unsupported} modality (its config declares no "
                        f"{unsupported} tower), so this "
                        f"{task.get('category') or 'case'} invariant cannot be "
                        f"evaluated. Excluded from the gate, not counted as a "
                        f"pass or a failure."
                    ),
                    "response_text": "",
                    "tool_calls": [],
                    "sampling": sampling,
                })
                logger.info(
                    f"Golden case '{task_id}' N/A for {model_name}: no "
                    f"{unsupported} modality; excluded from the gate."
                )
                continue

            # A case that forbids the served model's chat-template tokens
            # needs them before it is sent. Without them its leak check
            # cannot run, so it is a harness error and the endpoint is not
            # asked.
            harness_error: Optional[str] = None
            template_tokens: Optional[List[str]] = None
            if (task.get("golden_truth") or {}).get(FORBID_TEMPLATE_TOKENS):
                leak_check = self._template_leak_tokens(model_config, requested_model)
                template_tokens = leak_check.get("tokens") or []
                harness_error = leak_check.get("error")

            if harness_error is None:
                try:
                    response = self._query_remote_endpoint(
                        task.get("conversation", {}).get("messages", []),
                        endpoint,
                        tools=task.get("tools"),
                        tool_choice=task.get("tool_choice"),
                        sampling=sampling,
                        model_id=requested_model,
                    )
                except GoldenHarnessError as err:
                    harness_error = str(err)
            if harness_error is not None:
                harness_errors.append(f"{task_id}: {harness_error}")
                counts[STATUS_ERROR] += 1
                results.append({
                    "task_id": task_id,
                    "title": task.get("title"),
                    "category": task.get("category") or UNCATEGORIZED,
                    "status": STATUS_ERROR,
                    "passed": False,
                    "match_type": task.get("golden_truth", {}).get(
                        "match_type"
                    ),
                    "details": harness_error,
                    "response_text": "",
                    "tool_calls": [],
                    "sampling": sampling,
                })
                continue

            if response.get("served_model"):
                served_models.add(response["served_model"])

            eval_res = self.evaluate_task_response(
                task, response, template_tokens=template_tokens
            )
            eval_res["sampling"] = sampling
            if eval_res["status"] == STATUS_ERROR:
                harness_errors.append(f"{task_id}: {eval_res['details']}")
            counts[eval_res["status"]] += 1
            results.append(eval_res)

        if requested_model and served_models - {requested_model}:
            logger.warning(
                f"Requested model '{requested_model}' but the endpoint "
                f"reported serving {sorted(served_models)}"
            )

        errors = counts[STATUS_ERROR]
        failed = counts[STATUS_FAILED]
        passed = counts[STATUS_PASSED]
        not_applicable = counts[STATUS_NOT_APPLICABLE]
        # Denominator = cases actually applicable to this model's modalities.
        status, accuracy, applicable = _verdict(counts, len(tasks))

        # Counters are named *_cases rather than passed/failed because the
        # CLI splits results on a truthy top-level "failed" key, and an
        # integer count there would hide a partly-failing golden run.
        summary = {
            "model": model_name,
            "requested_model": requested_model,
            "served_model_id": requested_model,
            "served_model_id_source": self._model_id_source,
            "served_models": sorted(served_models),
            "endpoint": endpoint,
            "scoring_version": SCORING_VERSION,
            "status": status,
            "total_tasks": len(tasks),
            "applicable_tasks": applicable,
            "passed_cases": passed,
            "failed_cases": failed,
            "error_cases": errors,
            "not_applicable_cases": not_applicable,
            "accuracy_percent": accuracy,
            "harness_errors": harness_errors,
            "chat_template_leak_check": leak_check,
            "task_results": results,
        }

        if errors:
            logger.error(
                f"Golden Set ERRORED for {model_name}: {errors} of "
                f"{len(tasks)} cases never produced an answer. No score is "
                "reported."
            )
        else:
            verdict = "PASS" if status == STATUS_PASSED else "FAIL"
            na_note = (
                f" ({not_applicable} not applicable to this model's modalities)"
                if not_applicable else ""
            )
            logger.info(
                f"Golden Set {verdict} for {model_name}: "
                f"{passed}/{applicable} applicable cases passed{na_note}"
            )
        return summary
