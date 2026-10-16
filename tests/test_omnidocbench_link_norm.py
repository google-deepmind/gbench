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
"""Regression: omnidocbench text normalization must keep a markdown link's display text, not delete
the whole link. Deleting `[text](url)` was ASYMMETRIC - a model rendering a bare URL as a
markdown link (`[http://x](http://x)`) lost the URL from the prediction while the gold's bare URL was
kept, inflating the edit distance (0.083 -> 0.218) and flipping a faithful page from PASS to FAIL."""

from gbench.runners.eval_suites.omnidocbench import _normalize_doc_text, normalized_edit_distance


def test_link_display_text_is_kept():
    assert "grammarly.com" in _normalize_doc_text("[http://www.grammarly.com/](http://www.grammarly.com/)")
    assert "grammarly" in _normalize_doc_text("[Grammarly](http://www.grammarly.com/)").lower()


def test_bare_url_prediction_matches_markdown_link():
    pred = ("* Better choices out there\n    * [http://www.grammarly.com/](http://www.grammarly.com/)\n"
            "    * [http://www.grammarcheck.net/](http://www.grammarcheck.net/)")
    gold = ("\t-  Better choices out there\n\t\t- http://www.grammarly.com/\n"
            "\t\t- http://www.grammarcheck.net/")
    assert normalized_edit_distance(pred, gold) <= 0.10   # was 0.218 (FAIL) before the fix
