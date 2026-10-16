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
"""The googletrans worker of gbench's api_bank suite (image gbench-api-bank:googletrans-3.1.0a0,
built from gbench/docker/api_bank.Dockerfile).

API-Bank's Translate API (api-bank/apis/translate.py at DAMO-ConvAI f30ccf22) calls
`googletrans.Translator().translate(text, dest=..., src=...).text`, the synchronous API of the
googletrans release its requirements.txt pins, 3.1.0a0 (with httpx 0.13.3). That release cannot be
installed next to gbench (it downgrades httpx), so it runs here, one request per container: gbench's
process runs upstream's translate.py unchanged and hands each googletrans call to this worker.

One JSON request on stdin, one JSON reply on stdout, on the line that starts with REPLY_MARK:

* {"op": "probe"}: the googletrans and httpx releases installed, this file's sha256 and
  googletrans.LANGUAGES / LANGCODES (translate.py reads LANGUAGES);
* {"op": "translate", "text": ..., "dest": ..., "src": ...}: Translator(raise_exception=True)
  .translate(text, dest=dest, src=src) -> {"result": {"text", "src", "dest", "origin",
  "pronunciation"}}. raise_exception=True makes a reply other than HTTP 200 raise ("Unexpected
  status code ..."); googletrans' default instead returns the untranslated text as if it were the
  translation.

Any exception becomes {"error": {"type": its class, "message": str(exception)}}, the text
translate.py records as the call's exception. The proxy variables of the container's environment
are honored by httpx as they are on the host. No key or token is in this container.
"""

import hashlib
import json
import sys

REPLY_MARK = "GBENCH_TRANSLATE_WORKER="


def _probe():
    from importlib import metadata

    import googletrans
    with open(__file__, "rb") as f:
        own = hashlib.sha256(f.read()).hexdigest()
    return {"googletrans": metadata.version("googletrans"), "httpx": metadata.version("httpx"),
            "worker_sha256": own, "languages": dict(googletrans.LANGUAGES),
            "langcodes": dict(googletrans.LANGCODES)}


def _translate(req):
    from googletrans import Translator
    translated = Translator(raise_exception=True).translate(
        req["text"], dest=req.get("dest", "en"), src=req.get("src", "auto"))
    return {"result": {"text": translated.text, "src": translated.src, "dest": translated.dest,
                       "origin": translated.origin,
                       "pronunciation": translated.pronunciation}}


def main():
    try:
        req = json.loads(sys.stdin.read() or "{}")
        op = req.get("op")
        if op == "probe":
            out = _probe()
        elif op == "translate":
            out = _translate(req)
        else:
            out = {"error": {"type": "ValueError", "message": f"unknown op {op!r}"}}
    except Exception as e:  # noqa: BLE001 - every failure is reported, as translate.py records it
        out = {"error": {"type": type(e).__name__, "message": str(e)}}
    sys.stdout.write(REPLY_MARK + json.dumps(out, ensure_ascii=True, default=str) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
