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

"""textvqa labels each image with the MIME type its bytes carry: the pinned dataset's JPEG files go
out as data:image/jpeg, never under a hard-coded image/png. The HF download is faked; no network."""

from unittest import mock

from gbench.runners.eval_suites.textvqa import _load_textvqa_samples

_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 16
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def _load(rows):
    def fake(path, *a, **k):
        assert path == "lmms-lab/textvqa"
        return rows
    with mock.patch("datasets.load_dataset", autospec=True, side_effect=fake):
        return _load_textvqa_samples()


def _url(sample):
    return sample[0][0]["content"][0]["image_url"]["url"]


def test_jpeg_bytes_are_labelled_jpeg_and_png_bytes_png():
    samples = _load([{"question": "q1", "answers": ["a"], "image": {"bytes": _JPEG}},
                     {"question": "q2", "answers": ["b"], "image": {"bytes": _PNG}}])
    urls = sorted(_url(s).split(";", 1)[0] for s in samples)
    assert urls == ["data:image/jpeg", "data:image/png"]
    assert all(";base64," in _url(s) for s in samples)
