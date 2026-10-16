# gbench-tau2: the tau2-bench simulator (github.com/sierra-research/tau2-bench) for gbench's `tau2`
# and `tau3` suites, built LOCALLY (never pulled).
#
# The image holds a tau2-bench checkout at the pinned revision, installed the way upstream installs
# it (`uv sync --extra knowledge` from its own uv.lock, so litellm is upstream's locked 1.81.11 and
# not the gbench environment's), plus the tools the banking_knowledge shell tool needs:
# Anthropic's sandbox-runtime (`srt`) 0.0.23, the version tau2-bench requires, with ripgrep,
# bubblewrap and socat (src/tau2/knowledge/sandbox_manager.py). The checkout keeps its .git
# directory, so the suite can verify inside the container that HEAD is the pinned commit and that
# src/ and data/ are unmodified.
#
# gbench's own tau2 driver (gbench/runners/eval_suites/tau_common.py and the helpers it imports) is
# NOT copied in: every run mounts the current gbench source read-only, so the image never goes
# stale when that code changes. Rebuild only when the pin below changes.
#
# Build (context = docker/tau2_srt, which holds only srt's package.json and package-lock.json):
#   docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# tau2-bench main as of 2026-09-17 (package version 1.0.1). Keep in sync with
# TAU2_BENCH_REVISION in gbench/runners/eval_suites/tau_common.py.
ARG TAU2_REF=b7ea9074c1cba482b30687fecdb5c8425fd6f619
ARG UV_VERSION=0.12.19

LABEL org.opencontainers.image.title="gbench-tau2" \
      org.opencontainers.image.source="https://github.com/sierra-research/tau2-bench" \
      gbench.tau2.revision="${TAU2_REF}"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

# git: the revision check; ripgrep, bubblewrap, socat, nodejs/npm: sandbox-runtime and its tools.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates git ripgrep bubblewrap socat nodejs npm \
 && rm -rf /var/lib/apt/lists/*

# tau2-bench at the pinned revision. A depth-1 fetch of the commit keeps a real .git directory.
RUN git init -q /opt/tau2-bench \
 && git -C /opt/tau2-bench remote add origin https://github.com/sierra-research/tau2-bench \
 && git -C /opt/tau2-bench fetch -q --depth 1 origin "${TAU2_REF}" \
 && git -C /opt/tau2-bench checkout -q FETCH_HEAD \
 && test "$(git -C /opt/tau2-bench rev-parse HEAD)" = "${TAU2_REF}" \
 && git config --system --add safe.directory /opt/tau2-bench

# Upstream's install (README "1. Install": `uv sync --extra knowledge`) from its frozen uv.lock,
# into the image's system interpreter. --inexact keeps pip and uv, which the lock does not list.
# At this revision `import tau2` also needs `websockets` (tau2.data_model.simulation imports the
# voice stack at module level) although only the voice extra lists it, so upstream's text-mode
# install alone cannot import tau2. The lock's websockets (15.0.1, no dependencies) is added; no
# other voice package is installed or needed.
RUN pip install "uv==${UV_VERSION}" \
 && cd /opt/tau2-bench \
 && uv sync --frozen --inexact --no-dev --extra knowledge \
 && uv pip install --system --no-deps "websockets==15.0.1" \
 && test -z "$(git -C /opt/tau2-bench status --porcelain --untracked-files=no -- src data)"

# sandbox-runtime 0.0.23 with its dependencies locked (package-lock.json, integrity hashes).
COPY package.json package-lock.json /opt/srt/
RUN cd /opt/srt \
 && npm ci --omit=dev --no-audit --no-fund \
 && ln -s /opt/srt/node_modules/.bin/srt /usr/local/bin/srt \
 && test "$(node -p "require('@anthropic-ai/sandbox-runtime/package.json').version")" = "0.0.23"

# Fail the build unless tau2 imports from the checkout with upstream's locked litellm, and every
# binary sandbox_manager requires on Linux resolves.
RUN python -c "\
import os, shutil, tomllib; \
from importlib.metadata import version; \
import tau2; \
from tau2.knowledge.sandbox_manager import _REQUIRED_BINARIES_LINUX as req; \
assert os.path.realpath(tau2.__file__).startswith('/opt/tau2-bench/src/tau2/'), tau2.__file__; \
lock = tomllib.load(open('/opt/tau2-bench/uv.lock', 'rb'))['package']; \
locked = {p['name']: p['version'] for p in lock if 'version' in p}; \
bad = [(n, version(n), locked[n]) for n in ('litellm', 'openai', 'rank-bm25', 'websockets') \
       if version(n) != locked[n]]; \
assert not bad, bad; \
missing = [b for b in req if shutil.which(b) is None]; \
assert not missing, missing; \
print('tau2', version('tau2'), 'litellm', version('litellm'), 'binaries', req)"

ENV HOME=/tmp USER=gbench LOGNAME=gbench
USER 65534:65534
WORKDIR /tmp
CMD ["python", "-c", "print('gbench-tau2: gbench runs its tau2 driver here; see docs/evals/tau2.md')"]
