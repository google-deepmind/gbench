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

"""Reference-free spec validation of an emitted FunctionCall against its declaration.

A rule-based check of each emitted call against the tool schemas the request declared. It
needs no ground truth at all - only the call the model emitted and the declarations.

``validate_emitted(calls, tools)`` returns ``(spec_valid, errors)``, with one error code per
violation: unparseable arguments, unknown tool, missing required argument, unknown argument
name, wrong argument type. Only top-level argument types are checked, and a null value is
accepted unless its schema sets ``nullable: false``. A caller that scores tool use can treat
any violation as a failed call and ask an LLM judge only whether calling the tool was the
right thing to do: a judge not shown the declarations cannot tell a real tool or argument
from a hallucinated one, so it fails plausible calls it cannot check.
"""

from typing import Any, Dict, List, Optional, Tuple

#: Error codes, one per class of spec violation. They are stable identifiers:
#: `error_unknown_argument_type` means the value does not match the declared type.
ERR_UNPARSEABLE = "error_unparseable_fc"
ERR_INVALID_NAME = "error_invalid_tool_name"
ERR_MISSING_REQUIRED = "error_missing_required_property"
ERR_UNKNOWN_ARG = "error_unknown_argument_name"
ERR_BAD_TYPE = "error_unknown_argument_type"

#: JSON-schema `type` -> Python types accepted for it. `integer` deliberately excludes
#: bool (a bool IS an int in Python, and `True` is not a valid integer argument).
_TYPES = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _declared(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """`{tool name: parameters schema}` from an OpenAI-style tools list."""
    out: Dict[str, Dict[str, Any]] = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if name:
            out[str(name)] = fn.get("parameters") or {}
    return out


def _type_ok(value: Any, spec: Dict[str, Any]) -> bool:
    """Whether `value` satisfies the declared type. Unknown/absent type => accept."""
    if value is None:
        return bool(spec.get("nullable", True))
    declared = spec.get("type")
    if isinstance(declared, str):
        declared = declared.lower()
    if not declared or declared not in _TYPES:
        return True                      # `any_of`, missing type, custom: not our call
    if declared == "integer" and isinstance(value, bool):
        return False
    if declared == "number" and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[declared])


def validate_call(call: Dict[str, Any],
                  tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Spec violations for one emitted call. Empty list means it conforms.

    `call` is the normalized form produced by `base.normalize_function_calls`:
    `{"name": str, "arguments": dict | None, "arguments_raw": str?}`.
    """
    errors: List[Dict[str, Any]] = []
    name = str(call.get("name") or "")
    args = call.get("arguments")

    if args is None:
        # Arguments were emitted but would not parse as JSON.
        return [{"code": ERR_UNPARSEABLE, "tool": name,
                 "detail": str(call.get("arguments_raw") or "")[:200]}]

    declared = _declared(tools)
    if name not in declared:
        # Only decide this when we actually know what was on offer. With no declarations
        # we cannot tell a hallucinated tool from an undeclared-but-real one.
        if declared:
            errors.append({"code": ERR_INVALID_NAME, "tool": name,
                           "detail": f"not among {len(declared)} declared tools"})
        return errors

    schema = declared[name] or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []

    for key in required:
        if key not in args:
            errors.append({"code": ERR_MISSING_REQUIRED, "tool": name, "argument": key})

    for key, value in (args.items() if isinstance(args, dict) else []):
        if props and key not in props:
            errors.append({"code": ERR_UNKNOWN_ARG, "tool": name, "argument": key})
            continue
        spec = props.get(key) or {}
        if not _type_ok(value, spec):
            errors.append({"code": ERR_BAD_TYPE, "tool": name, "argument": key,
                           "detail": f"expected {spec.get('type')}, got "
                                     f"{type(value).__name__}"})
    return errors


def validate_emitted(calls: Optional[List[Dict[str, Any]]],
                     tools: Optional[List[Dict[str, Any]]]) -> Tuple[bool, List[Dict[str, Any]]]:
    """`(spec_valid, errors)` for every call in a turn.

    A turn that emits NO call is spec-valid: whether a call was warranted is a judgement
    for whatever scores the turn, not for a schema check.
    """
    errors: List[Dict[str, Any]] = []
    for call in calls or []:
        errors.extend(validate_call(call, tools))
    return (not errors), errors
