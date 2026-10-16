# Canonical MCP-Bench image for gbench's `mcp_bench` suite, built LOCALLY (never pulled).
#
# Bundles Accenture's MCP-Bench runner + all 28 vendored MCP servers, built from source by the
# upstream install.sh (which tolerates per-server failures: one flaky server does not abort the
# build). The gbench launcher (mcp_bench_run.py) injects the gbench Gemini cascade judge and
# delegates to upstream benchmark.runner.main(). The system interpreter's MCP client is pinned to
# mcp==1.16.0 (mcp_bench_constraints.txt explains why) and the rest of its packages to the set of
# the checked image; Node, pip, uv and the managed CPython builds are pinned too, and the build
# fails if any of them moves. The build's npm packages resolve as of a fixed date (recorded).
#
# Build (context = gbench/docker, so the adapter files are in scope):
#   docker build -t gbench-mcp-bench:7a8eaeae -f gbench/docker/mcp_bench.Dockerfile gbench/docker
#
# HEAVY image: builds ~9 TypeScript servers (tsc/tsup) + ~19 Python servers (uv), including the
# fragile BioMCP/alphagenome, metmuseum (hand-built tsc), and numpy/scipy servers. Tens of
# minutes. A graded run additionally needs a served model at /v1, GEMINI_API_KEY (the judge, which
# gbench keeps in its own process: the container gets a relay placeholder), the five server keys
# (gbench passes them only to a second container of this image, the credentialed-servers
# container, never to the one where model text runs), and network egress with name resolution
# (the online servers hit public APIs); see docs/evals/mcp_bench.md.
FROM ubuntu:22.04@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Make uv pip installs land in the system interpreter so `python <server>.py` servers resolve
# their deps (the `uv run` servers still use their own uv-managed venvs).
ENV UV_SYSTEM_PYTHON=1

# Base toolchain: Python 3.10, the version MCP-Bench's README installs into (`conda create -n
# mcpbench python=3.10`; Ubuntu 22.04's python3 is 3.10, so install.sh's ">= 3.10" check passes and
# it installs no other Python), as the only python/python3 in the image; build tools, git, sudo
# (install.sh calls sudo), and Node 22 LTS (NodeSource) at the build the checked image got
# (NODE_VERSION; NodeSource keeps every build in its repository, and an unpinned install now takes
# a newer one). Pre-installing these makes install.sh's version checks skip re-installing them.
# tzdata gives the image a configured local timezone (/etc/localtime, Etc/UTC), as any machine the
# README runs on has: Time MCP's server reads it at start-up (tzlocal get_localzone_name) and
# exits without one, so it would list no tools. The Ubuntu packages float with the Ubuntu archive;
# the installed versions are recorded in /opt/gbench/dpkg-versions.txt at the end of the build.
ARG NODE_VERSION=22.23.3-1nodesource1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git gnupg sudo tzdata \
        build-essential gcc g++ make \
        python3 python3-venv python3-dev python-is-python3 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends "nodejs=${NODE_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && test -e /etc/localtime \
    && python3 -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version" \
    && test "$(dpkg-query -W -f '${Version}' nodejs)" = "${NODE_VERSION}"

# pip for that interpreter (the README's conda env ships one), with the pip, setuptools and wheel
# releases of the checked image (get-pip.py installs the newest of each unless told which).
ARG PIP_VERSION=26.2.1
ARG SETUPTOOLS_VERSION=84.0.0
ARG WHEEL_VERSION=0.48.0
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py --no-cache-dir "pip==${PIP_VERSION}" \
        "setuptools==${SETUPTOOLS_VERSION}" "wheel==${WHEEL_VERSION}" \
    && rm -f /tmp/get-pip.py \
    && python3 -m pip --version | grep -q "^pip ${PIP_VERSION} "

# uv (astral) on PATH for all users, at a fixed release. The release also fixes which managed
# CPython builds uv downloads (their list is built into uv).
ARG UV_VERSION=0.12.19
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh \
    && /usr/local/bin/uv --version | grep -q "^uv ${UV_VERSION} "
ENV PATH="/usr/local/bin:${PATH}"

# Clone MCP-Bench at a pinned commit (self-contained; gbench never pulls a prebuilt image).
ARG MCPBENCH_REF=7a8eaeae83a842a2949080acc5473f65e1569daf
RUN git clone https://github.com/Accenture/mcp-bench /app/mcp-bench \
    && cd /app/mcp-bench && git checkout "${MCPBENCH_REF}"

WORKDIR /app/mcp-bench

# Constraints for every pip / uv pip install into the system interpreter: mcp==1.16.0, the release
# MCP-Bench's declared mcp[cli]>=1.9.0 resolved to on the pinned commit's date, with which
# upstream's connection check reports 28/28 servers connected with tools (see the file), and the
# rest of the resolved set of the image built 2026-09-28. `uv sync` does not read UV_CONSTRAINT,
# so the servers' own uv.lock environments are untouched.
COPY mcp_bench_constraints.txt /app/constraints.txt

# Every npm resolution of the build (install.sh's global pnpm/yarn/tsx and its `npm install
# typescript --save-dev` steps, a server without a lockfile, the guards below) takes the newest
# release published before NPM_CONFIG_BEFORE (npm's `before`), the time the checked image ran
# install.sh, instead of the newest release of the build day. A build argument, so it applies to
# the build's RUN steps only; nothing resolves npm packages when a server starts.
ARG NPM_CONFIG_BEFORE=2026-09-26T00:10:00Z

# uv prefers the image's own Python (3.10, the README env's interpreter), at build time and when a
# `uv run` server starts. It downloads a managed CPython only for an environment whose
# .python-version or requires-python 3.10 cannot satisfy: nasa-mcp's (.python-version 3.13; a
# `uv run` server, so it runs on it) and the ones install.sh's `uv sync` builds for
# huggingface-mcp-server (>=3.13), mcp-nixos (>=3.11) and medcalc (>=3.12), which get the newest
# managed CPython (3.14 today) but do not serve them: commands.json starts those three with
# `python`, the image's 3.10, as it does in the README's conda env. Without the preference, once a
# managed 3.14 has been fetched, the unpinned `uv run` servers switch to it too, and their locked
# dependencies without wheels for it (cffi 1.17.1 under wikipedia-mcp and mcp-reddit) fail to
# build, so those servers list no tools. The servers commands.json starts with `python` do not use
# uv at all.
ENV UV_PYTHON_PREFERENCE=system

# Build all 28 servers + install their dependencies with the upstream installer, every step of it.
# install.sh stops at its first failing command (`set -e`), and metmuseum-mcp's unguarded `npx tsc`
# exits non-zero on type errors under today's npm packages (it still emits dist/), which would end
# the installer before okx-mcp, mcp-google-map, openapi-mcp-server and every Python server. So the
# script runs with that one line turned into `set +e`: each step runs, a failed one is logged, and
# the build check below fails the build naming any server left unusable. $0 stays install.sh's
# own path, so its `cd "$(dirname "$0")"` still lands in mcp_servers/. The first step fails the
# build when install.sh no longer has exactly that one `set -e` line (upstream drifted), so the
# rewrite can never silently skip or change more than that line.
RUN [ "$(grep -c '^set -e' /app/mcp-bench/mcp_servers/install.sh)" = 1 ] \
    || { echo "install.sh does not have exactly one 'set -e' line (upstream drifted)" >&2; exit 1; }
RUN cd /app/mcp-bench/mcp_servers \
    && PIP_CONSTRAINT=/app/constraints.txt UV_CONSTRAINT=/app/constraints.txt \
       bash -c "$(sed 's/^set -e.*/set +e/' install.sh)" /app/mcp-bench/mcp_servers/install.sh; \
    true

# Post-install guards for servers the installer can leave incomplete on a fresh Linux image (each
# tolerant, so every guard runs; the build check below fails the build if one left its server
# unusable): OKX's tsc build, Google Maps' tsup build, OpenAPI's npm install, and a real .venv for
# the 4 servers commands.json starts with `uv run`.
RUN set -x; cd /app/mcp-bench/mcp_servers; \
    ( cd okx-mcp && npm install typescript --no-save && npx tsc ) || echo "WARN okx build failed"; \
    ( cd mcp-google-map && npm install && npm run build ) || echo "WARN google-map build failed"; \
    ( cd openapi-mcp-server && npm install ) || echo "WARN openapi npm failed"; \
    for d in wikipedia-mcp biomcp mcp-reddit nasa-mcp; do \
      ( cd "/app/mcp-bench/mcp_servers/$d" && env -u UV_SYSTEM_PYTHON uv sync ) || echo "WARN uv sync failed: $d"; \
    done; \
    true

# Runner + judge deps in the system interpreter, added only where the installer left them out
# (pip keeps any version already installed): the launcher/judge import openai + json_repair, the
# runner imports mcp/jsonschema/etc. Every server commands.json starts with `python` runs on this
# same interpreter, as in upstream's single environment. The build fails unless the client is
# mcp 1.16.0.
RUN export PIP_CONSTRAINT=/app/constraints.txt \
    && python3 -m pip install --no-cache-dir openai json_repair \
    && python3 -m pip install --no-cache-dir mcp jsonschema pyyaml aiohttp tenacity \
    && python3 -m pip install --no-cache-dir -e /app/mcp-bench/mcp_servers/time-mcp \
    && python3 -m pip install --no-cache-dir "mcp[cli]==1.16.0" \
    && python3 -c "import importlib.metadata as m; v = m.version('mcp'); assert v == '1.16.0', 'MCP client ' + v"

# Every server commands.json starts must be ready: its built JavaScript entry point, its script,
# its module (importable) or its .venv, checked in its working directory. install.sh and the guards
# above keep going past a failed step, so this is where such a failure fails the build, naming
# every server that is not ready (mcp_bench_build_check.py; it writes nothing into the checkout).
COPY mcp_bench_build_check.py /opt/gbench/mcp_bench_build_check.py
RUN python3 /opt/gbench/mcp_bench_build_check.py /app/mcp-bench/mcp_servers

# The build fails unless the system interpreter holds exactly the constrained set plus the four
# servers installed editable from the checkout, and the managed CPython builds uv downloaded are
# exactly the ones the pinned uv release chose for the checked image. Then the record of what this
# build installed, in /opt/gbench: the Python set, the Ubuntu packages, Node, npm and the global
# npm tools, and the changes install.sh made to the checkout (the typescript releases it added to
# six servers' package.json and lockfile).
ARG UV_MANAGED_PYTHONS="cpython-3.13.15-linux-x86_64-gnu cpython-3.14.7-linux-x86_64-gnu"
RUN mkdir -p /opt/gbench \
    && python3 -m pip freeze --all --exclude-editable | LC_ALL=C sort > /opt/gbench/pip-freeze.txt \
    && grep -v -e '^#' -e '^$' /app/constraints.txt | LC_ALL=C sort | diff - /opt/gbench/pip-freeze.txt \
    && test "$(python3 -m pip freeze --all | sed -n 's/^-e .*#egg=\([^&]*\).*/\1/p' | LC_ALL=C sort | tr '\n' ' ')" \
        = "bibliomantic_mcp_server mcp_reddit_server mcp_server_time unit_converter_mcp " \
    && test "$(ls "$(uv python dir)" | grep -E '^cpython-[0-9]+\.[0-9]+\.[0-9]+-' | LC_ALL=C sort | tr '\n' ' ')" \
        = "${UV_MANAGED_PYTHONS} " \
    && dpkg-query -W -f '${Package}=${Version}\n' > /opt/gbench/dpkg-versions.txt \
    && { node --version; npm --version; uv --version; npm ls -g --depth=0; } > /opt/gbench/toolchain-versions.txt \
    && git -C /app/mcp-bench status --porcelain > /opt/gbench/mcp-bench-checkout-status.txt \
    && git -C /app/mcp-bench diff > /opt/gbench/mcp-bench-checkout.diff
LABEL org.gbench.nodejs="${NODE_VERSION}" org.gbench.uv="${UV_VERSION}" \
      org.gbench.npm_before="${NPM_CONFIG_BEFORE}"

# gbench adapter (cascade judge + model request settings + launcher + entrypoint, and the two
# ends of the credentialed-server bridge), from the build context (gbench/docker). The label
# org.gbench.model-request-policy tells gbench this image applies the run's thinking switch,
# --thinking budget and request timeout to the model under test and counts its failed requests
# (mcp_bench_request_policy.py). The label org.gbench.mcp-bench-harness tells it the judge takes
# gbench's Gemini relay placeholder and relay URL (no real Gemini key enters the container), the
# five credentialed servers run in gbench's credentialed-servers container (a second container of
# this image, mcp_bench_credentialed.py) behind the bridge (mcp_bench_bridge.py) and no server key
# enters the model container, a session's credentialed side says when it is up before gbench's
# relay answers and the bridge records every session that failed outside the model (version 3:
# the session protocol of gbench's relay), the network check resolves the servers' API host names
# (no IP-only test), and the launcher keeps each task's harness errors out of the averages and
# writes /out/harness_log.json with the answering Gemini models per task (mcp_bench_run.py).
# gbench refuses an image without either label at these versions.
LABEL org.gbench.model-request-policy="2"
LABEL org.gbench.mcp-bench-harness="3"
RUN mkdir -p /app/adapter
COPY mcp_bench_cascade_judge.py /app/adapter/mcp_bench_cascade_judge.py
COPY mcp_bench_request_policy.py /app/adapter/mcp_bench_request_policy.py
COPY mcp_bench_run.py /app/adapter/mcp_bench_run.py
COPY mcp_bench_bridge.py /app/adapter/mcp_bench_bridge.py
COPY mcp_bench_credentialed.py /app/adapter/mcp_bench_credentialed.py
COPY mcp_bench_entrypoint.sh /app/adapter/mcp_bench_entrypoint.sh
RUN chmod +x /app/adapter/mcp_bench_entrypoint.sh \
    && python3 -m py_compile /app/adapter/mcp_bench_run.py /app/adapter/mcp_bench_bridge.py \
       /app/adapter/mcp_bench_credentialed.py /app/adapter/mcp_bench_cascade_judge.py \
       /app/adapter/mcp_bench_request_policy.py

ENV MCPBENCH_DIR=/app/mcp-bench
ENV ADAPTER_DIR=/app/adapter
ENTRYPOINT ["/app/adapter/mcp_bench_entrypoint.sh"]
